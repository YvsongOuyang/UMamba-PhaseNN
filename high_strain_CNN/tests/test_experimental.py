from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experimental.run_official_data import (
    intensity_center,
    load_experimental_intensity,
    resample_centered,
)


class ExperimentalDataTest(unittest.TestCase):
    def test_resampling_centers_an_off_axis_peak(self) -> None:
        volume = np.zeros((15, 19, 23), dtype=np.float32)
        volume[3, 12, 17] = 1.0
        center = intensity_center(volume)
        resized = resample_centered(volume, (9, 9, 9), source_center=center, order=1)
        peak = np.asarray(np.unravel_index(np.argmax(resized), resized.shape))
        np.testing.assert_array_equal(peak, np.asarray([4, 4, 4]))

    def test_complex_reinterpolation_preserves_type_and_center(self) -> None:
        volume = np.zeros((9, 9, 9), dtype=np.complex64)
        volume[4, 4, 4] = np.exp(1j * 0.7)
        resized = resample_centered(volume, (17, 13, 11), order=1)
        self.assertEqual(resized.dtype, np.complex64)
        peak = np.asarray(np.unravel_index(np.argmax(np.abs(resized)), resized.shape))
        np.testing.assert_array_equal(peak, np.asarray([8, 6, 5]))

    def test_loader_rejects_nonphysical_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.npy"
            np.save(path, np.asarray([[[-1.0, 1.0]]], dtype=np.float32))
            with self.assertRaises(ValueError):
                load_experimental_intensity(path)


if __name__ == "__main__":
    unittest.main()
