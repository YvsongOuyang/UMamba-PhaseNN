"""Focused tests for second-stage evaluation metrics."""

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from pytorch_autophasenn.evaluate_refiner import (
    ambiguity_aligned_reconstruction_metrics,
    build_autophasenn_dataset,
    phase_wca,
    prepare_targets,
    parse_args,
    relative_improvement,
    support_metrics,
)
from pytorch_autophasenn.momamba_refiner import ambiguity_aware_component_mae
from pytorch_autophasenn.reconstruction import realspace_from_modulus_phase
from pytorch_autophasenn.train_refiner import target_support
from pytorch_autophasenn.visualize_refiner import (
    align_reconstruction_for_display,
    twin_transform,
)


class EvaluateRefinerTest(unittest.TestCase):
    def test_autophasenn_evaluation_defaults_to_support_threshold_point_three(
        self,
    ) -> None:
        with patch(
            "sys.argv",
            [
                "evaluate_refiner",
                "--checkpoint",
                "checkpoint.pt",
                "--dataset-format",
                "autophasenn_memmap",
            ],
        ):
            args = parse_args()
        self.assertEqual(args.support_threshold, 0.3)

    def test_refiner_training_uses_dataset_specific_support(self) -> None:
        target = torch.zeros(1, 4, 4, 4, dtype=torch.complex64)
        target[:, 1:3, 1:3, 1:3] = 0.5 + 0.0j
        autophasenn_args = argparse.Namespace(
            data_format="autophasenn",
            support_threshold=0.1,
        )
        torch.testing.assert_close(
            target_support({}, target, autophasenn_args),
            target.abs() >= 0.1,
        )

        stored = torch.ones_like(target, dtype=torch.bool)
        author_args = argparse.Namespace(
            data_format="author_npz",
            support_threshold=0.1,
        )
        self.assertIs(target_support({"support": stored}, target, author_args), stored)

    def test_autophasenn_targets_derive_support_and_reciprocal_phase(self) -> None:
        realspace = torch.zeros(1, 4, 4, 4, dtype=torch.complex64)
        realspace[:, 1:3, 1:3, 1:3] = 0.5 + 0.25j
        args = argparse.Namespace(
            dataset_format="autophasenn_memmap",
            support_threshold=0.1,
        )
        target, support, target_phase = prepare_targets(
            {"realspace": realspace},
            args,
            torch.device("cpu"),
        )
        torch.testing.assert_close(target, realspace)
        torch.testing.assert_close(support, realspace.abs() >= 0.1)
        self.assertEqual(tuple(target_phase.shape), (1, 4, 4, 4))
        self.assertTrue(torch.isfinite(target_phase).all())
        self.assertEqual(float(target_phase[0, 2, 2, 2]), 0.0)

    def test_autophasenn_dataset_uses_shared_memmap_config(self) -> None:
        shape = (4, 4, 4)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            diffraction = np.memmap(
                root / "val_diff.npy",
                dtype="float32",
                mode="w+",
                shape=(2,) + shape,
            )
            diffraction[:] = 1.0
            diffraction.flush()
            realspace = np.memmap(
                root / "val_real.npy",
                dtype="complex64",
                mode="w+",
                shape=(2,) + shape,
            )
            realspace[:] = 1.0 + 0.0j
            realspace.flush()
            del diffraction, realspace
            config_path = root / "data.json"
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "dataset_name": "AutoPhaseNN",
                        "dataset_version": "test",
                        "root": str(root),
                        "shape": list(shape),
                        "dtypes": {
                            "diffraction": "float32",
                            "realspace": "complex64",
                        },
                        "splits": {
                            "val": {
                                "diffraction": "val_diff.npy",
                                "realspace": "val_real.npy",
                                "num_samples": 2,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                data_config=str(config_path),
                data_dir="",
                data_diff="",
                data_real="",
                split="val",
                num_samples=1,
            )
            model_args = argparse.Namespace(shape=4, input_log_data=True)
            dataset, manifest = build_autophasenn_dataset(args, model_args)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(manifest["selection"], "split_prefix")
            self.assertEqual(manifest["declared_split_samples"], 2)
            self.assertIn("diffraction", dataset[0])
            dataset.diffraction._mmap.close()
            dataset.realspace._mmap.close()
            del dataset

    def test_phase_wca_is_zero_for_matching_object(self) -> None:
        modulus = torch.rand(2, 1, 4, 4, 4) + 0.1
        target_phase = torch.randn(2, 4, 4, 4)
        center = (2, 2, 2)
        target_phase = target_phase - target_phase[(slice(None),) + center][
            :, None, None, None
        ]
        realspace = realspace_from_modulus_phase(modulus, target_phase)
        weights = torch.rand_like(modulus) + 0.1
        self.assertLess(float(phase_wca(realspace, target_phase, weights)), 1e-6)

    def test_support_metrics_use_unit_maximum_amplitude(self) -> None:
        amplitude = torch.tensor([[[[[1.0, 0.4, 0.1, 0.0]]]]])
        realspace = torch.complex(amplitude, torch.zeros_like(amplitude))
        support = torch.tensor([[[[[True, True, False, False]]]]])
        values = support_metrics(realspace, support, threshold=0.3)
        self.assertAlmostEqual(float(values["support_iou"]), 1.0)
        self.assertAlmostEqual(float(values["support_dice"]), 1.0)
        self.assertAlmostEqual(float(values["support_volume_ratio"]), 1.0)

    def test_reconstruction_metrics_ignore_scale_phase_and_twin_ambiguity(self) -> None:
        generator = torch.Generator().manual_seed(7)
        target = torch.complex(
            torch.rand(2, 4, 4, 4, generator=generator),
            torch.rand(2, 4, 4, 4, generator=generator),
        )
        support = torch.ones_like(target, dtype=torch.bool)
        direct = target[0:1] * torch.exp(torch.tensor(0.7j)) * 2.5
        twin = torch.conj(
            torch.roll(
                torch.flip(target[1:2], dims=(-3, -2, -1)),
                shifts=(1, 1, 1),
                dims=(-3, -2, -1),
            )
        )
        prediction = torch.cat((direct, twin), dim=0)
        object_loss = ambiguity_aware_component_mae(prediction, target, support)
        metrics = ambiguity_aligned_reconstruction_metrics(
            prediction,
            target,
            object_loss.selected_support,
            object_loss.selected_twin,
        )
        self.assertLess(float(metrics["complex_nrmse"]), 1e-6)
        self.assertGreater(float(metrics["amplitude_psnr_db"]), 100.0)
        torch.testing.assert_close(
            object_loss.selected_twin,
            torch.tensor([False, True]),
        )

    def test_reconstruction_metrics_report_degradation(self) -> None:
        target = torch.ones(1, 4, 4, 4, dtype=torch.complex64)
        prediction = target.clone()
        prediction[:, :2] = 0.0
        support = torch.ones_like(target, dtype=torch.bool)
        object_loss = ambiguity_aware_component_mae(prediction, target, support)
        metrics = ambiguity_aligned_reconstruction_metrics(
            prediction,
            target,
            object_loss.selected_support,
            object_loss.selected_twin,
        )
        self.assertGreater(float(metrics["complex_nrmse"]), 0.0)
        self.assertTrue(torch.isfinite(metrics["amplitude_psnr_db"]))
        self.assertLess(float(metrics["amplitude_psnr_db"]), 20.0)

    def test_relative_improvement_obeys_metric_direction(self) -> None:
        self.assertAlmostEqual(relative_improvement("complex_nrmse", 0.2, 0.1), 50.0)
        self.assertAlmostEqual(
            relative_improvement("amplitude_psnr_db", 20.0, 25.0),
            25.0,
        )
        self.assertAlmostEqual(relative_improvement("object_mae", 0.2, 0.1), 50.0)
        self.assertAlmostEqual(relative_improvement("support_iou", 0.4, 0.6), 50.0)
        self.assertAlmostEqual(
            relative_improvement("support_volume_ratio", 1.4, 1.2),
            50.0,
        )
        self.assertIsNone(relative_improvement("twin_fraction", 0.4, 0.5))

    def test_display_alignment_resolves_twin_and_global_phase(self) -> None:
        generator = torch.Generator().manual_seed(11)
        target = torch.complex(
            torch.rand(1, 1, 4, 4, 4, generator=generator),
            torch.rand(1, 1, 4, 4, 4, generator=generator),
        )
        support = torch.ones_like(target, dtype=torch.bool)
        phase = torch.exp(torch.tensor(0.7j))
        prediction = twin_transform(target) * phase * 2.0
        aligned, selected_twin, _ = align_reconstruction_for_display(
            prediction,
            target,
            support,
        )
        self.assertTrue(selected_twin)
        aligned = aligned / aligned.abs().square().mean().sqrt()
        target = target / target.abs().square().mean().sqrt()
        torch.testing.assert_close(aligned, target, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
