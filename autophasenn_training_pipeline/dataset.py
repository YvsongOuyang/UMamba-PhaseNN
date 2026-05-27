from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class AutoPhaseDataset(Dataset):
    """Loads memmap AutoPhaseNN samples as channel-first tensors."""

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
    ):
        if scale_I is not None:
            scale_i = scale_I

        if num_samples is None:
            raise ValueError("num_samples is required for memmap loading.")

        self.scale_i = float(scale_i)
        self.diff_path = Path(diff_path)
        self.real_path = Path(real_path) if real_path is not None else None
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

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        actual_idx = self.indices[index]
        diff = np.array(self.mmap_diff[actual_idx])
        if self.mmap_real is None:
            realspace = None
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

