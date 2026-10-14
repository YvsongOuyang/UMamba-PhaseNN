"""Read the e-CDI generator's HDF5 and its existing, immutable split indices."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _h5py():
    try:
        import h5py
    except (ImportError, ValueError) as error:
        raise RuntimeError(
            "e-CDI HDF5 reading requires h5py built for the installed NumPy; see ecdi_training/requirements/pytorch-2d.txt."
        ) from error
    return h5py


def _root_file(root: Path, name: str) -> Path:
    return (root / name.replace("\\", "/")).resolve()


def inspect_manifest(config: dict) -> dict:
    """Validate the complete partition, HDF5 identity, fields and valid flags."""

    root = Path(config["root"])
    path = _root_file(root, config["manifest"])
    content = path.read_bytes()
    manifest = json.loads(content)
    dataset_path = (path.parent / manifest["dataset_path"].replace("\\", "/")).resolve()
    splits = manifest["splits"]
    if set(splits) != {"train", "val", "test"}:
        raise ValueError(
            "Use splits_training_with_val.json with train, val and test, not the train/test-only manifest."
        )
    all_indices = []
    for split, indices in splits.items():
        if not indices or any(type(index) is not int or index < 0 for index in indices):
            raise ValueError(f"Invalid or empty indices for {split}.")
        if manifest["counts"][split] != len(indices):
            raise ValueError(f"Manifest count differs from the {split} index list.")
        all_indices.extend(indices)
    if len(all_indices) != len(set(all_indices)):
        raise ValueError("Split indices overlap or contain duplicates.")
    sample_count = manifest["sample_count"]
    if sorted(all_indices) != list(range(sample_count)):
        raise ValueError("Manifest must partition every HDF5 sample exactly once.")
    with _h5py().File(dataset_path, "r") as source:
        keys = (
            "observed_counts",
            "exit_wave_real",
            "exit_wave_imag",
            "ideal_intensity",
            "metadata_json",
            "valid",
        )
        missing = set(keys) - set(source)
        if missing:
            raise ValueError(f"Missing HDF5 fields: {sorted(missing)}.")
        for key in keys[:4]:
            if source[key].shape != (sample_count, 512, 512) or source[
                key
            ].dtype != np.dtype("float32"):
                raise ValueError(
                    f"{key} must be float32 [N,512,512] and match the manifest."
                )
        if source["valid"].shape != (sample_count,) or not source["valid"][:].all():
            raise ValueError(
                "The HDF5 has incomplete samples; finish generation before training."
            )
        if source["metadata_json"].shape != (sample_count,):
            raise ValueError("Metadata size does not match the manifest.")
        source_hash = str(source.attrs.get("config_sha256", ""))
        if not source_hash or source_hash != manifest.get("config_sha256"):
            raise ValueError("HDF5 config_sha256 does not match the split manifest.")
        source_config = json.loads(source.attrs["config_json"])
        if source_config["physics"]["imaging_mode"] != "single_shot_ecdi":
            raise ValueError("Expected coherent single-shot e-CDI exit-wave pairs.")
        if source_config["physics"]["frozen_phonon_configs"] != 1:
            raise ValueError(
                "Averaging intensities from multiple exit waves breaks a single-wave target."
            )
    report_path = config.get("quality_report")
    if report_path:
        report = json.loads(_root_file(root, report_path).read_text(encoding="utf-8"))
        if (
            report.get("passed") is not True
            or report.get("failures")
            or report.get("sample_count") != sample_count
            or report.get("config_sha256") != source_hash
        ):
            raise ValueError(
                "Generator quality report is incomplete or does not match this dataset."
            )
    return {
        "storage": "ecdi_hdf5",
        "manifest_path": str(path),
        "manifest_sha256": hashlib.sha256(content).hexdigest(),
        "dataset_path": str(dataset_path),
        "config_sha256": source_hash,
        "sample_count": sample_count,
        "splits": splits,
        "counts": manifest["counts"],
        "split_type": manifest.get("split_type"),
        "interpretation": manifest.get("interpretation"),
        "geometry": source_config.get("ecdi", {}).get("sample_geometry"),
        "input_field": "observed_counts",
        "target_fields": ["exit_wave_real", "exit_wave_imag"],
        "support_semantics": "stored support ignored; amplitude threshold is an effective illumination region",
    }


class EcdiHDF5Dataset(Dataset):
    """Lazy worker-local HDF5 handles; never copy/export the full dataset."""

    def __init__(
        self,
        config: dict,
        split: str,
        num_samples: int = 0,
        *,
        include_ideal: bool = False,
    ):
        info = inspect_manifest(config)
        available = info["splits"][split]
        if num_samples < 0 or num_samples > len(available):
            raise ValueError("num_samples must be zero (all) or within the split.")
        self.indices = available[:num_samples] if num_samples else available
        self.path = info["dataset_path"]
        self.include_ideal = include_ideal
        self._handle = None
        self._pid = None
        self.manifest = {key: value for key, value in info.items() if key != "splits"}
        self.manifest.update(
            split=split,
            available_samples=len(available),
            selected_samples=len(self.indices),
            selected_indices_sha256=hashlib.sha256(
                json.dumps(self.indices).encode()
            ).hexdigest(),
            shape=[512, 512],
            diffraction_kind="intensity",
        )

    def _open(self):
        if self._handle is None or self._pid != os.getpid():
            self.close()
            self._handle = _h5py().File(self.path, "r")
            self._pid = os.getpid()
        return self._handle

    def close(self):
        if self._handle is not None:
            self._handle.close()
        self._handle, self._pid = None, None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"], state["_pid"] = None, None
        return state

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        source = self._open()
        index = self.indices[position]
        if not source["valid"][index]:
            raise ValueError(f"Sample {index} is not complete.")
        real = np.asarray(source["exit_wave_real"][index])
        imag = np.asarray(source["exit_wave_imag"][index])
        sample = {
            "diffraction": torch.from_numpy(
                np.asarray(source["observed_counts"][index])
            )[None],
            "realspace": torch.complex(torch.from_numpy(real), torch.from_numpy(imag))[
                None
            ],
            "index": index,
        }
        if self.include_ideal:
            sample["ideal_intensity"] = torch.from_numpy(
                np.asarray(source["ideal_intensity"][index])
            )[None]
            metadata = json.loads(source["metadata_json"].asstr()[index])
            sample["material"] = metadata["structure"]["source_name"]
        return sample
