"""Check the AutoPhaseNN subset adapter independently of model inference."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from simulation.run_paper_model import prepare_model_input, reconstruct_object
from simulation import visualization
from tools.export_autophasenn_samples import adapt_sample, main


class AutoPhaseNNExportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.obj = np.zeros((16, 16, 16), dtype=np.complex64)
        self.obj[2:7, 6:11, 7:12] = 0.8 * np.exp(0.7j)
        self.obj[2:4, 6:11, 7:12] *= 0.5
        self.spectrum = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(self.obj)))
        self.modulus = np.abs(self.spectrum).astype(np.float32)

    def test_preserves_measurement_and_uses_clean_truth(self) -> None:
        noisy_modulus = np.floor(self.modulus + 1.0)
        original = self.obj.copy()
        sample = adapt_sample(noisy_modulus, self.obj)
        clean = adapt_sample(self.modulus, self.obj)
        np.testing.assert_array_equal(sample.intensity, noisy_modulus**2)
        np.testing.assert_array_equal(sample.realspace_object, clean.realspace_object)
        np.testing.assert_array_equal(self.obj, original)
        np.testing.assert_allclose(sample.clean_intensity, np.abs(self.spectrum)**2, rtol=2e-7)
        expected_input = np.log1p(noisy_modulus**2)
        expected_input = (expected_input - expected_input.min()) / np.ptp(expected_input)
        np.testing.assert_array_equal(prepare_model_input(sample.intensity)[0, ..., 0], expected_input)

    def test_fractional_translation_and_fourier_pair(self) -> None:
        sample = adapt_sample(self.modulus, self.obj)
        offset = np.array(sample.metadata["source_amplitude_center_offset_voxels"])
        frequencies = np.stack(np.meshgrid(*([np.fft.fftshift(np.fft.fftfreq(16))] * 3), indexing="ij"))
        corrected = self.spectrum * np.exp(2j * np.pi * np.einsum("i,idhw->dhw", offset, frequencies))
        corrected *= np.exp(-1j * np.angle(corrected[8, 8, 8]))
        expected = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(corrected)))
        np.testing.assert_allclose(sample.realspace_object, expected, atol=2e-6)
        np.testing.assert_allclose(
            reconstruct_object(sample.clean_intensity, sample.reciprocal_phase),
            sample.realspace_object, atol=2e-6,
        )
        amp = np.abs(sample.realspace_object)
        np.testing.assert_array_equal(sample.support, amp > 0.1 * amp.max())

    def test_rejects_invalid_arrays(self) -> None:
        for modulus, obj in [(self.modulus[0], self.obj), (-self.modulus, self.obj),
                             (np.full_like(self.modulus, np.nan), self.obj),
                             (self.modulus, np.zeros_like(self.obj))]:
            with self.subTest(shape=modulus.shape), self.assertRaises(ValueError):
                adapt_sample(modulus, obj)
        with self.assertRaises(ValueError):
            adapt_sample(self.modulus, self.obj, target_support_threshold=0)

    def test_plots_use_the_evaluation_target_support(self) -> None:
        sample = adapt_sample(self.modulus, self.obj)
        with tempfile.TemporaryDirectory() as temporary:
            arguments = dict(
                intensity=sample.intensity, target_object=sample.realspace_object,
                predicted_object=sample.realspace_object, target_support=sample.support,
                support_threshold=0.3,
            )
            with patch.object(visualization, "plot_five_panel_volume") as volume_plot:
                visualization.save_volume_overview(**arguments, destination=Path(temporary) / "3d.png")
                target_panel = volume_plot.call_args.kwargs["panel_rows"][0][0]
                np.testing.assert_array_equal(target_panel[0], sample.support)
                self.assertEqual(target_panel[3], 0.5)
            with patch.object(visualization, "_masked_phase", wraps=visualization._masked_phase) as masked:
                visualization.save_slice_overview(
                    **arguments, destination=Path(temporary) / "2d.png",
                    target_reciprocal_phase=sample.reciprocal_phase,
                    predicted_reciprocal_phase=sample.reciprocal_phase,
                )
                np.testing.assert_array_equal(masked.call_args_list[3].args[1], sample.support)

    def test_cli_records_sampling_and_keeps_source_readonly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = json.loads((Path(__file__).resolve().parents[1] / "configs/autophasenn_data.json").read_text())
            config.update(root=str(root), shape=[16, 16, 16])
            config["splits"] = {"val": {"diffraction": "val_diff.npy", "realspace": "val_real.npy", "num_samples": 3}}
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            self.modulus[None].repeat(3, axis=0).tofile(root / "val_diff.npy")
            self.obj[None].repeat(3, axis=0).tofile(root / "val_real.npy")
            source_bytes = (root / "val_diff.npy").read_bytes()
            output = root / "export"
            argv = ["export", "--data-config", str(config_path), "--data-dir", str(root),
                    "--output-dir", str(output), "--num-samples", "2", "--seed", "19"]
            with patch("sys.argv", argv):
                self.assertEqual(main(), 0)
            manifest = json.loads((output / "dataset_manifest.json").read_text())
            self.assertEqual(manifest["source_indices"], np.random.default_rng(19).choice(3, 2, replace=False).tolist())
            self.assertEqual(len(list(output.glob("sample_*.npz"))), 2)
            self.assertEqual((root / "val_diff.npy").read_bytes(), source_bytes)
            with patch("sys.argv", argv), self.assertRaises(FileExistsError):
                main()


if __name__ == "__main__":
    unittest.main()
