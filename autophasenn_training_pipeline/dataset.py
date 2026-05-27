from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset


def read_file_list(data_dir, data_list=None, limit=0):
    data_dir = Path(data_dir)
    if data_list is None:
        files = sorted(list(data_dir.glob("*.npz")) + list(data_dir.glob("*.npy")))
    else:
        list_path = Path(data_list)
        if not list_path.is_absolute():
            list_path = data_dir / list_path
        names = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
        files = []
        for name in names:
            path = Path(name)
            candidates = []
            if path.is_absolute():
                candidates.append(path)
            else:
                candidates.append(data_dir / path)
                candidates.append(data_dir / path.name)
            for candidate in candidates:
                if candidate.exists():
                    files.append(candidate)
                    break
            else:
                raise FileNotFoundError(f"Could not resolve data file listed as {name!r}")

    if limit and limit > 0:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No .npz/.npy data files found under {data_dir}")
    return files


def split_files(files, train_ratio=0.9, seed=4):
    files = list(files)
    order = list(range(len(files)))
    random.Random(seed).shuffle(order)
    files = [files[i] for i in order]
    n_train = int(len(files) * train_ratio)
    n_train = min(max(n_train, 1), len(files))
    train_files = files[:n_train]
    val_files = files[n_train:]
    if not val_files:
        val_files = train_files
    return train_files, val_files


def _realspace_from_complex_diff(complex_diff):
    return np.fft.ifftn(np.fft.ifftshift(complex_diff))


def _load_np_data(path):
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded as data:
            keys = set(data.files)
            if {"arr_0", "arr_1"}.issubset(keys):
                diff = data["arr_0"]
                realspace = data["arr_1"]
            elif "arr_0" in keys:
                arr = data["arr_0"]
                if np.iscomplexobj(arr):
                    diff = np.abs(arr)
                    realspace = _realspace_from_complex_diff(arr)
                else:
                    diff = arr
                    realspace = None
            else:
                arr = data[data.files[0]]
                if np.iscomplexobj(arr):
                    diff = np.abs(arr)
                    realspace = _realspace_from_complex_diff(arr)
                else:
                    diff = arr
                    realspace = None
    else:
        arr = loaded
        if np.iscomplexobj(arr):
            diff = np.abs(arr)
            realspace = _realspace_from_complex_diff(arr)
        else:
            diff = arr
            realspace = None
    return diff, realspace


class AutoPhaseDataset(Dataset):
    """Loads AutoPhaseNN samples as channel-first tensors.

    Supported sample formats:
    - memmap diffraction/real-space arrays using the same loading style as
      the root data_loader.py.
    - .npz with arr_0 diffraction amplitude and arr_1 complex real-space object.
    - .npz/.npy containing a complex diffraction array; real space is computed by ifft.
    """

    def __init__(
        self,
        files=None,
        real_path=None,
        num_samples=None,
        shape_diff=(64, 64, 64),
        shape_real=(64, 64, 64),
        dtype_diff="float32",
        dtype_real="complex64",
        scale_i=0.0,
        shuffle=None,
        seed=4,
        diff_path=None,
        scale_I=None,
    ):
        if scale_I is not None:
            scale_i = scale_I

        self.scale_i = float(scale_i)
        self.files = None
        self.mmap_diff = None
        self.mmap_real = None
        self.indices = None

        if diff_path is None and files is not None and (real_path is not None or num_samples is not None):
            diff_path = files
            files = None

        if diff_path is not None:
            if num_samples is None:
                raise ValueError("num_samples is required when loading memmap data.")
            self.mode = "memmap"
            self.diff_path = Path(diff_path)
            self.real_path = Path(real_path) if real_path is not None else None
            self.num_samples = int(num_samples)
            self.mmap_diff = np.memmap(
                self.diff_path,
                dtype=dtype_diff,
                mode="r",
                shape=(self.num_samples,) + tuple(shape_diff),
            )
            if self.real_path is not None:
                self.mmap_real = np.memmap(
                    self.real_path,
                    dtype=dtype_real,
                    mode="r",
                    shape=(self.num_samples,) + tuple(shape_real),
                )
            self.indices = list(range(self.num_samples))
            if shuffle is None:
                shuffle = True
            if shuffle:
                random.Random(seed).shuffle(self.indices)
            return

        if files is None:
            raise ValueError("Either files or diff_path must be provided.")
        self.mode = "files"
        self.files = [Path(f) for f in files]
        if shuffle:
            random.Random(seed).shuffle(self.files)

    def __len__(self):
        if self.mode == "memmap":
            return self.num_samples
        return len(self.files)

    def __getitem__(self, index):
        if self.mode == "memmap":
            actual_idx = self.indices[index]
            diff = np.array(self.mmap_diff[actual_idx])
            if self.mmap_real is None:
                realspace = None
            else:
                realspace = np.array(self.mmap_real[actual_idx])
            name = f"{self.diff_path.stem}_{actual_idx:06d}"
            return self._format_sample(diff, realspace, name)

        path = self.files[index]
        diff, realspace = _load_np_data(path)
        return self._format_sample(diff, realspace, path.name)

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

