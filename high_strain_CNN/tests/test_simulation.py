"""Numerical checks for the paper-style synthetic data generator."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from simulation.config import load_simulation_config
from simulation.generator import generate_sample, save_sample
from simulation.run_paper_model import prepare_model_input, reconstruct_object


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_DIR / "configs" / "simulation_paper.json"


class SimulationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_simulation_config(CONFIG_PATH)

    def test_all_paper_shape_and_phase_families(self) -> None:
        rng = np.random.default_rng(12345)
        for shape_type, phase_type in zip(
            self.config.shape.types,
            self.config.phase.types,
        ):
            with self.subTest(shape=shape_type, phase=phase_type):
                sample = generate_sample(
                    self.config,
                    rng,
                    shape_type=shape_type,
                    phase_type=phase_type,
                )
                expected_shape = (self.config.grid_size,) * 3
                self.assertEqual(sample.intensity.shape, expected_shape)
                self.assertEqual(sample.reciprocal_phase.shape, expected_shape)
                self.assertTrue(np.all(sample.intensity >= 0))
                self.assertGreater(np.count_nonzero(sample.support), 0)
                self.assertGreater(sample.metadata["oversampling_ratio"], 2.0)
                center = np.asarray(sample.metadata["final_support_center_of_mass"])
                np.testing.assert_allclose(
                    center,
                    np.full(3, self.config.grid_size // 2),
                    atol=0.75,
                )
                phase_span = np.ptp(sample.object_phase[sample.support])
                self.assertGreaterEqual(phase_span, 2.0 * np.pi - 1e-5)
                self.assertLessEqual(phase_span, 5.0 * np.pi + 1e-5)

    def test_clean_modulus_and_phase_reconstruct_object(self) -> None:
        sample = generate_sample(
            self.config,
            np.random.default_rng(7),
            shape_type="wulff",
            phase_type="double_gaussian",
        )
        reconstruction = reconstruct_object(
            sample.clean_intensity,
            sample.reciprocal_phase,
        )
        np.testing.assert_allclose(
            reconstruction,
            sample.realspace_object,
            rtol=2e-5,
            atol=2e-5,
        )

    def test_saved_npz_matches_official_loader_schema(self) -> None:
        sample = generate_sample(
            self.config,
            np.random.default_rng(19),
            shape_type="random_polyhedron",
            phase_type="gaussian_correlated",
        )
        destination = (
            PROJECT_DIR
            / "artifacts"
            / "simulation"
            / "test_sample_schema.npz"
        )
        try:
            destination = save_sample(sample, destination, save_extras=True)
            with np.load(destination) as stored:
                self.assertIn("I", stored.files)
                self.assertIn("phi", stored.files)
                self.assertEqual(stored["I"].dtype, np.float32)
                self.assertEqual(stored["phi"].dtype, np.float32)
                model_input = prepare_model_input(stored["I"])
                self.assertEqual(model_input.shape, (1, 64, 64, 64, 1))
                self.assertEqual(model_input.dtype, np.float32)
                self.assertGreaterEqual(float(model_input.min()), 0.0)
                self.assertLessEqual(float(model_input.max()), 1.0)
        finally:
            destination.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
