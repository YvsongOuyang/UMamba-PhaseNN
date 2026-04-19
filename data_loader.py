"""Memory-mapped diffraction dataset for AutoPhaseNN training."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


class Dataset(torch.utils.data.Dataset):
    """Load paired diffraction and real-space volumes without reading all data into RAM.

    The historical training scripts name the files ``*.npy`` but some datasets may
    actually be raw memmap blobs. This loader first tries ``np.load(..., mmap_mode='r')``
    for standard NumPy files and falls back to raw ``np.memmap`` when needed.
    """

    def __init__(
        self,
        diff_path,
        real_path,
        num_samples: Optional[int],
        shape_diff: Tuple[int, int, int] = (64, 64, 64),
        shape_real: Tuple[int, int, int] = (64, 64, 64),
        dtype_diff: str = "float32",
        dtype_real: str = "complex64",
        scale_I: float = 0,
        shuffle: bool = True,
        seed: int = 4,
    ):
        self.diff_path = Path(diff_path)
        self.real_path = Path(real_path)
        self.scale_I = scale_I

        self.mmap_diff = self._open_array(self.diff_path, dtype_diff, shape_diff, num_samples)
        self.mmap_real = self._open_array(self.real_path, dtype_real, shape_real, num_samples)

        if self.mmap_diff.shape[0] != self.mmap_real.shape[0]:
            raise ValueError(
                f"Diffraction and real-space sample counts differ: "
                f"{self.mmap_diff.shape[0]} vs {self.mmap_real.shape[0]}"
            )

        available = self.mmap_diff.shape[0]
        self.num_samples = available if num_samples is None else min(int(num_samples), available)

        self.indices = np.arange(self.num_samples)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(self.indices)

    @staticmethod
    def _open_array(path: Path, dtype: str, sample_shape: Tuple[int, int, int], num_samples: Optional[int]):
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.ndim == len(sample_shape):
                array = array.reshape((1,) + tuple(sample_shape))
            if array.shape[1:] != tuple(sample_shape):
                raise ValueError(
                    f"{path} has sample shape {array.shape[1:]}, expected {sample_shape}"
                )
            return array
        except Exception as npy_error:
            if num_samples is None:
                raise ValueError(
                    f"{path} is not a readable .npy file and num_samples was not provided "
                    "for raw memmap fallback."
                ) from npy_error

            expected_size = int(num_samples) * int(np.prod(sample_shape)) * np.dtype(dtype).itemsize
            actual_size = path.stat().st_size
            if actual_size < expected_size:
                raise ValueError(
                    f"{path} is too small for raw memmap fallback: "
                    f"{actual_size} bytes < expected {expected_size} bytes"
                ) from npy_error

            return np.memmap(
                path,
                dtype=dtype,
                mode="r",
                shape=(int(num_samples),) + tuple(sample_shape),
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        actual_idx = int(self.indices[index])

        diff = np.asarray(self.mmap_diff[actual_idx], dtype=np.float32)
        realspace = np.asarray(self.mmap_real[actual_idx], dtype=np.complex64)

        amp = np.abs(realspace).astype(np.float32, copy=False)
        phi = np.angle(realspace).astype(np.float32, copy=False)

        # Convert measured diffraction intensity to amplitude-like scale.
        diff = np.sqrt(np.clip(diff, a_min=0.0, a_max=None)).astype(np.float32, copy=False)
        if self.scale_I > 0:
            max_I = float(diff.max())
            if max_I > 0:
                diff = diff / max_I * float(self.scale_I)

        return (
            torch.from_numpy(diff.copy()).unsqueeze(0),
            torch.from_numpy(amp.copy()).unsqueeze(0),
            torch.from_numpy(phi.copy()).unsqueeze(0),
        )
