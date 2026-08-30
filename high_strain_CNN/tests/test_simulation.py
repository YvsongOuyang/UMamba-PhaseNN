"""Shared NPZ, Fourier reconstruction, threshold, and cache regression tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from simulation.evaluate_paper_model import (
    _file_sha256,
    _prediction_identity,
    choose_threshold,
    load_sample,
    realspace_metrics,
    run_tensorflow_predictions,
)
from simulation.run_paper_model import prepare_model_input, reconstruct_object
from simulation.sample_io import SimulatedSample, save_sample


class SimulationTest(unittest.TestCase):
    def test_centered_fft_matches_kinematic_sum(self) -> None:
        size = 4
        coordinates = (np.indices((size,) * 3) - size // 2).reshape(3, -1).T
        rng = np.random.default_rng(4)
        obj = rng.normal(size=(size,) * 3) + 1j * rng.normal(size=(size,) * 3)
        direct = np.exp(-2j * np.pi * (coordinates @ coordinates.T) / size) @ obj.ravel()
        fft = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj)))
        np.testing.assert_allclose(fft.ravel(), direct, atol=1e-12)

    def test_support_threshold_uses_exact_generated_support(self) -> None:
        target_support = np.zeros((6, 6, 6), dtype=bool)
        target_support[2:4, 2:4, 2:4] = True
        target = target_support.astype(np.complex64)
        prediction_amplitude = target_support.astype(np.float32)
        prediction_amplitude[1:5, 1:5, 1:5] += 0.2
        prediction = prediction_amplitude.astype(np.complex64)
        low = realspace_metrics(target, prediction, target_support, 0.1)
        high = realspace_metrics(target, prediction, target_support, 0.3)
        self.assertLess(low["support_iou"], high["support_iou"])
        self.assertAlmostEqual(high["support_iou"], 1.0)

    def test_threshold_selection_uses_calibration_iou(self) -> None:
        summaries = [
            {
                "threshold": 0.1,
                "calibration_support_iou_mean": 0.7,
                "calibration_support_dice_mean": 0.8,
                "calibration_support_volume_ratio_mean": 1.0,
            },
            {
                "threshold": 0.3,
                "calibration_support_iou_mean": 0.9,
                "calibration_support_dice_mean": 0.94,
                "calibration_support_volume_ratio_mean": 1.1,
            },
        ]
        self.assertEqual(choose_threshold(summaries), 0.3)

    def test_threshold_selection_uses_volume_ratio_inside_iou_plateau(self) -> None:
        summaries = [
            {
                "threshold": 0.145,
                "calibration_support_iou_mean": 0.57326,
                "calibration_support_dice_mean": 0.7232,
                "calibration_support_volume_ratio_mean": 1.023,
            },
            {
                "threshold": 0.15,
                "calibration_support_iou_mean": 0.57325,
                "calibration_support_dice_mean": 0.7227,
                "calibration_support_volume_ratio_mean": 0.989,
            },
        ]
        self.assertEqual(choose_threshold(summaries, iou_tolerance=1e-3), 0.15)

    def test_sample_io_preserves_schema_and_arrays(self) -> None:
        support = np.zeros((64, 64, 64), dtype=bool)
        support[24:40, 24:40, 24:40] = True
        phase = np.zeros(support.shape, dtype=np.float32)
        phase[support] = np.linspace(-1, 1, support.sum())
        obj = (support * np.exp(1j * phase)).astype(np.complex64)
        spectrum = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj)))
        intensity = np.abs(spectrum) ** 2
        sample = SimulatedSample(
            intensity=intensity,
            reciprocal_phase=np.angle(spectrum),
            support=support,
            object_phase=phase,
            realspace_object=obj,
            clean_intensity=intensity,
            metadata={"route": "schema-test"},
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.npz"
            for extras in (False, True):
                with self.subTest(extras=extras):
                    save_sample(sample, path, save_extras=extras)
                    with np.load(path, allow_pickle=False) as stored:
                        expected = {"I", "phi", "metadata_json"}
                        if extras:
                            expected |= {"support", "object_phase", "object", "I_clean"}
                        self.assertEqual(set(stored.files), expected)
                        self.assertEqual(stored["I"].dtype, np.float32)
                        self.assertEqual(stored["phi"].dtype, np.float32)
                        np.testing.assert_array_equal(stored["I"], intensity.astype(np.float32))
                        np.testing.assert_array_equal(
                            stored["phi"], sample.reciprocal_phase.astype(np.float32)
                        )
                        self.assertEqual(json.loads(str(stored["metadata_json"])), sample.metadata)
                        model_input = prepare_model_input(stored["I"])
                        self.assertEqual(model_input.shape, (1, 64, 64, 64, 1))
                        self.assertEqual(model_input.dtype, np.float32)
                        self.assertGreaterEqual(float(model_input.min()), 0.0)
                        self.assertLessEqual(float(model_input.max()), 1.0)
                        if extras:
                            np.testing.assert_array_equal(stored["support"], support)
                            np.testing.assert_array_equal(stored["object"], obj)
                            np.testing.assert_array_equal(stored["object_phase"], phase)
                            np.testing.assert_allclose(
                                reconstruct_object(stored["I_clean"], stored["phi"]),
                                obj, rtol=2e-5, atol=2e-5,
                            )


class PredictionCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.sample = self.directory / "sample_00000.npz"
        self.model = self.directory / "model.h5"
        self.model.write_bytes(b"test-model-a")
        self.write_sample(1.0)
        self.prediction = self.directory / "predicted_reciprocal_phase.npy"
        np.save(self.prediction, np.zeros((1, 4, 4, 4), dtype=np.float32))
        self.manifest = {
            "content_identity": _prediction_identity([self.sample], self.model),
            "prediction_sha256": _file_sha256(self.prediction),
        }
        self.write_manifest()

    def write_sample(self, intensity: float) -> None:
        np.savez_compressed(
            self.sample, I=np.full((4, 4, 4), intensity, dtype=np.float32),
            phi=np.zeros((4, 4, 4), dtype=np.float32),
            object=np.ones((4, 4, 4), dtype=np.complex64),
            support=np.ones((4, 4, 4), dtype=np.uint8),
        )

    def write_manifest(self) -> None:
        (self.directory / "prediction_manifest.json").write_text(
            json.dumps(self.manifest), encoding="utf-8"
        )

    def reuse(self) -> None:
        predictions, _ = run_tensorflow_predictions(
            [self.sample], self.model, self.directory, 1, True
        )
        self.assertEqual(predictions.shape, (1, 4, 4, 4))
        del predictions

    def test_matching_cache_is_usable_without_tensorflow(self) -> None:
        self.reuse()

    def test_same_name_changed_sample_is_rejected(self) -> None:
        self.write_sample(2.0)
        with self.assertRaisesRegex(ValueError, "identity differs"):
            self.reuse()

    def test_same_path_changed_model_is_rejected(self) -> None:
        self.model.write_bytes(b"test-model-b")
        with self.assertRaisesRegex(ValueError, "identity differs"):
            self.reuse()

    def test_corrupt_prediction_is_rejected(self) -> None:
        np.save(self.prediction, np.ones((1, 4, 4, 4), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "incomplete or changed"):
            self.reuse()

    def test_legacy_cache_requires_fresh_inference(self) -> None:
        self.manifest.pop("content_identity")
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "identity differs"):
            self.reuse()

    def test_noisy_inverse_fft_is_not_accepted_as_clean_truth(self) -> None:
        np.savez(self.sample, I=np.ones((4, 4, 4)), phi=np.zeros((4, 4, 4)))
        with self.assertRaisesRegex(ValueError, "--save-extras"):
            load_sample(self.sample)


if __name__ == "__main__":
    unittest.main()
