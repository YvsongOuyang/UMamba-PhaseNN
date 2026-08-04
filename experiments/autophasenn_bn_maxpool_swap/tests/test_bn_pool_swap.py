"""Operator- and model-level tests for the BN/MaxPool swap experiment."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
for import_root in (REPOSITORY_ROOT, EXPERIMENT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import torch
import torch.nn as nn
import torch.nn.functional as F

from bn_pool_experiment.model import (
    AFFECTED_BN_LAYERS,
    PoolBNOrder,
    PoolBNSwapAutoPhaseNN,
    audit_all_bn_scales,
    audit_bn_scales,
)
from bn_pool_experiment.metrics import complex_tensor_pair_metrics
from bn_pool_experiment.config import ExperimentConfig
from bn_pool_experiment.multi_dataset import (
    DatasetSpec,
    MultiDatasetConfig,
    build_experiment_config,
    build_multi_dataset_summary,
    load_multi_dataset_config,
    multi_dataset_metric_rows,
    resolve_base_config_path,
)


class PoolBNCommutationTest(unittest.TestCase):
    """Verify the mathematical precondition implemented by the experiment."""

    def setUp(self) -> None:
        torch.manual_seed(7)
        self.inputs = torch.randn(2, 3, 8, 8, 8)
        self.bn = nn.BatchNorm3d(3, eps=1e-3)
        self.bn.running_mean.copy_(torch.tensor([0.2, -0.4, 0.1]))
        self.bn.running_var.copy_(torch.tensor([0.5, 1.5, 2.0]))
        self.bn.bias.data.copy_(torch.tensor([0.1, -0.2, 0.3]))
        self.bn.eval()

    def test_positive_scale_commutes(self) -> None:
        self.bn.weight.data.copy_(torch.tensor([0.5, 1.0, 2.0]))
        bn_then_pool = F.max_pool3d(self.bn(self.inputs), 2, 2)
        pool_then_bn = self.bn(F.max_pool3d(self.inputs, 2, 2))
        self.assertTrue(torch.allclose(bn_then_pool, pool_then_bn, atol=1e-6, rtol=1e-6))

    def test_negative_scale_breaks_maxpool_equivalence(self) -> None:
        self.bn.weight.data.copy_(torch.tensor([0.5, -1.0, 2.0]))
        bn_then_pool = F.max_pool3d(self.bn(self.inputs), 2, 2)
        pool_then_bn = self.bn(F.max_pool3d(self.inputs, 2, 2))
        maximum_error = torch.max(torch.abs(bn_then_pool - pool_then_bn)).detach()
        self.assertGreater(float(maximum_error), 1e-3)

    def test_model_exposes_exactly_four_affected_bn_layers(self) -> None:
        model = PoolBNSwapAutoPhaseNN(order=PoolBNOrder.BN_THEN_POOL)
        self.assertEqual(len(AFFECTED_BN_LAYERS), 4)
        self.assertTrue(all(name in model.layers for name in AFFECTED_BN_LAYERS))
        audit = audit_bn_scales(model)
        self.assertTrue(all(item["positive_fraction"] == 1.0 for item in audit.values()))

    def test_order_switch_does_not_change_state_dict_schema(self) -> None:
        model = PoolBNSwapAutoPhaseNN(order=PoolBNOrder.BN_THEN_POOL)
        schema_before = tuple(model.state_dict())
        model.set_pool_bn_order(PoolBNOrder.POOL_THEN_BN)
        self.assertEqual(schema_before, tuple(model.state_dict()))

    def test_complex_output_metric_is_zero_for_identical_tensors(self) -> None:
        real = torch.randn(1, 1, 4, 4, 4)
        imag = torch.randn(1, 1, 4, 4, 4)
        value = torch.complex(real, imag)
        metrics = complex_tensor_pair_metrics(value, value.clone())
        self.assertTrue(all(metric == 0.0 for metric in metrics.values()))

    def test_full_model_contains_twenty_six_bn_layers(self) -> None:
        model = PoolBNSwapAutoPhaseNN()
        audit = audit_all_bn_scales(model)
        self.assertEqual(len(audit), 26)
        self.assertTrue(all(item["positive_fraction"] == 1.0 for item in audit.values()))


class MultiDatasetConfigurationTest(unittest.TestCase):
    """Verify that batch evaluation preserves the training pipeline data contract."""

    def setUp(self) -> None:
        self.multi = MultiDatasetConfig(
            base_config="default.yaml",
            checkpoint="checkpoint.pt",
            data_dir="/datasets/autophasenn",
            output_dir="outputs_multi",
            device="cuda",
            datasets=[
                DatasetSpec(
                    name="validation",
                    diff_file="val_diff.npy",
                    real_file="val_real.npy",
                    num_samples=5000,
                )
            ],
        )

    def test_dataset_override_matches_autophasenn_memmap_layout(self) -> None:
        config = build_experiment_config(
            ExperimentConfig(),
            self.multi,
            self.multi.datasets[0],
            limit=8,
        )
        self.assertEqual(config.data.mode, "memmap")
        self.assertEqual(config.data.data_dir, "/datasets/autophasenn")
        self.assertEqual(config.data.diff_file, "val_diff.npy")
        self.assertEqual(config.data.real_file, "val_real.npy")
        self.assertEqual(config.data.num_samples, 8)
        self.assertEqual(config.data.shape, [64, 64, 64])
        self.assertEqual(config.data.dtype_diff, "float32")
        self.assertEqual(config.data.dtype_real, "complex64")
        self.assertEqual(config.model.checkpoint, "checkpoint.pt")
        self.assertEqual(config.runtime.device, "cuda")

    def test_shipped_multi_dataset_config_is_valid(self) -> None:
        config_path = EXPERIMENT_ROOT / "configs" / "server_multi_dataset.yaml"
        config = load_multi_dataset_config(config_path)
        enabled_names = [item.name for item in config.datasets if item.enabled]
        self.assertEqual(enabled_names, ["train", "validation"])
        base_path = resolve_base_config_path(config_path, config.base_config)
        self.assertEqual(base_path.resolve(), (config_path.parent / "default.yaml").resolve())

    def test_aggregate_contains_reconstruction_and_output_diff_rows(self) -> None:
        statistic = {
            "mean": 0.1,
            "std": 0.01,
            "ci_low": 0.08,
            "ci_high": 0.12,
        }
        summary = {
            "num_samples": 2,
            "has_realspace_truth": True,
            "reconstruction": {
                "paper_modulus_mae": {
                    "direction": "lower",
                    "baseline": statistic,
                    "swapped": statistic,
                    "delta": {**statistic, "mean": 0.0},
                    "relative_change_percent": 0.0,
                    "degradation_percent": 0.0,
                }
            },
            "final_output_comparison": {
                "metrics": {
                    "mae": statistic,
                    "rmse": statistic,
                    "max_abs": statistic,
                    "relative_l1": statistic,
                    "relative_l2": statistic,
                    "pearson_corr": statistic,
                    "histogram_js_divergence": statistic,
                },
                "maximum_absolute_difference_over_all_samples": 0.2,
                "exactly_identical": False,
            },
            "acceptance": {
                "overall_pass": True,
                "primary_metrics_pass": True,
                "output_consistency_pass": True,
                "bn_scale_precondition_pass": True,
            },
        }
        summaries = {"validation": summary}
        rows = multi_dataset_metric_rows(summaries)
        self.assertEqual(
            {row["category"] for row in rows},
            {"reconstruction", "model_output_difference"},
        )
        aggregate = build_multi_dataset_summary(self.multi, summaries)
        self.assertEqual(aggregate["dataset_count"], 1)
        self.assertEqual(aggregate["total_samples"], 2)
        self.assertTrue(aggregate["all_datasets_pass"])


if __name__ == "__main__":
    unittest.main()
