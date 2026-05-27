from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset


def _infer_num_samples(path, dtype, sample_shape):
    path = Path(path)
    item_bytes = np.dtype(dtype).itemsize * int(np.prod(sample_shape))
    file_bytes = path.stat().st_size
    if file_bytes % item_bytes != 0:
        raise ValueError(
            f"Cannot infer sample count for {path}: file size {file_bytes} is not "
            f"a multiple of one sample ({item_bytes} bytes)."
        )
    return file_bytes // item_bytes


class AutoPhaseDataset(Dataset):
    """Memmap AutoPhaseNN dataset with pipeline-compatible preprocessing."""

    def __init__(
        self,
        diff_path,
        real_path=None,
        num_samples=None,
        shape_diff=(64, 64, 64),
        shape_real=(64, 64, 64),
        dtype_diff="float32",
        dtype_real="complex64",
        scale_i=0.0,
        shuffle=True,
        seed=4,
        scale_I=None,
        cache_data=False,
        allow_missing_real=False,
    ):
        if scale_I is not None:
            scale_i = scale_I

        self.scale_i = float(scale_i)
        self.diff_path = Path(diff_path)
        self.real_path = Path(real_path) if real_path is not None else None
        if not self.diff_path.exists():
            raise FileNotFoundError(f"diff memmap not found: {self.diff_path}")
        if self.real_path is not None and not self.real_path.exists():
            if allow_missing_real:
                self.real_path = None
            else:
                raise FileNotFoundError(f"real memmap not found: {self.real_path}")

        if num_samples is None:
            num_samples = _infer_num_samples(self.diff_path, dtype_diff, shape_diff)
        self.num_samples = int(num_samples)

        self.mmap_diff = np.memmap(
            self.diff_path,
            dtype=dtype_diff,
            mode="r",
            shape=(self.num_samples,) + tuple(shape_diff),
        )
        self.mmap_real = None
        if self.real_path is not None:
            self.mmap_real = np.memmap(
                self.real_path,
                dtype=dtype_real,
                mode="r",
                shape=(self.num_samples,) + tuple(shape_real),
            )

        self.indices = list(range(self.num_samples))
        if shuffle:
            random.Random(seed).shuffle(self.indices)

        self.cache_data = bool(cache_data)
        self.cached_diff = None
        self.cached_real = None
        if self.cache_data:
            self.cached_diff = np.asarray(self.mmap_diff[self.indices]).copy()
            if self.mmap_real is not None:
                self.cached_real = np.asarray(self.mmap_real[self.indices]).copy()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        actual_idx = self.indices[index]
        if self.cache_data:
            diff = self.cached_diff[index]
        else:
            diff = np.array(self.mmap_diff[actual_idx])

        if self.mmap_real is None:
            realspace = None
        elif self.cache_data:
            realspace = self.cached_real[index]
        else:
            realspace = np.array(self.mmap_real[actual_idx])

        name = f"{self.diff_path.stem}_{actual_idx:06d}"
        return self._format_sample(diff, realspace, name)

    def _format_sample(self, diff, realspace, name):
        diff = np.asarray(diff, dtype=np.float32)
        if self.scale_i > 0:
            diff = diff / (float(diff.max()) + 1e-6) * self.scale_i

        if realspace is None:
            amp = np.zeros_like(diff, dtype=np.float32)
            phi = np.zeros_like(diff, dtype=np.float32)
        else:
            amp = np.abs(realspace).astype(np.float32)
            phi = np.angle(realspace).astype(np.float32)

        return {
            "diff": torch.from_numpy(diff[None]),
            "amp": torch.from_numpy(amp[None]),
            "phi": torch.from_numpy(phi[None]),
            "name": name,
        }
