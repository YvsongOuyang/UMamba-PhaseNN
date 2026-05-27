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
    - .npz with arr_0 diffraction amplitude and arr_1 complex real-space object.
    - .npz/.npy containing a complex diffraction array; real space is computed by ifft.
    """

    def __init__(self, files, scale_i=0.0):
        self.files = [Path(f) for f in files]
        self.scale_i = float(scale_i)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        diff, realspace = _load_np_data(path)
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
            "name": path.name,
        }

