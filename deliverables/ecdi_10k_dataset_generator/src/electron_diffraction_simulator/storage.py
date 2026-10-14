"""Resumable HDF5 storage for generated diffraction datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .config import SimulationConfig


class H5DatasetWriter:
    """Fixed-size, resumable dataset writer with per-sample completion flags."""

    def __init__(self, path: Path, config: SimulationConfig) -> None:
        self.path = path
        self.config = config
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and config.output.overwrite:
            mode = "w"
        elif path.exists():
            mode = "a"
        else:
            mode = "w"
        self.file = h5py.File(path, mode)
        self._initialize_or_validate()

    def _initialize_or_validate(self) -> None:
        config_data = self.config.to_dict()
        config_json = json.dumps(config_data, sort_keys=True, ensure_ascii=False)
        hash_data = json.loads(config_json)
        hash_data["output"]["path"] = ""
        hash_data["output"]["overwrite"] = False
        hash_data["run"]["log_every"] = 0
        hash_data["run"]["cupy_cache_dir"] = ""
        hash_json = json.dumps(hash_data, sort_keys=True, ensure_ascii=False)
        config_hash = hashlib.sha256(hash_json.encode("utf-8")).hexdigest()
        expected_shape = (
            self.config.run.num_samples,
            self.config.physics.detector_size,
            self.config.physics.detector_size,
        )
        if "valid" in self.file:
            if tuple(self.file["observed_counts"].shape) != expected_shape:
                raise ValueError("existing dataset shape does not match configuration")
            previous_hash = self.file.attrs.get("config_sha256", "")
            if previous_hash != config_hash:
                raise ValueError(
                    "existing dataset was created with a different configuration; "
                    "choose another output path or set output.overwrite=true"
                )
            return

        compression = self.config.output.compression
        compression_options = (
            self.config.output.compression_level if compression == "gzip" else None
        )
        image_chunks = (
            1,
            min(self.config.physics.detector_size, 256),
            min(self.config.physics.detector_size, 256),
        )
        image_kwargs = {
            "shape": expected_shape,
            "dtype": np.float32,
            "chunks": image_chunks,
            "compression": compression,
            "compression_opts": compression_options,
        }
        self.file.create_dataset("ideal_intensity", **image_kwargs)
        self.file.create_dataset("observed_counts", **image_kwargs)
        self.file.create_dataset("observed_log", **image_kwargs)
        if self.config.physics.imaging_mode == "single_shot_ecdi":
            self.file.create_dataset(
                "support",
                shape=(self.config.run.num_samples, self.config.physics.grid_size, self.config.physics.grid_size),
                dtype=np.bool_,
                chunks=(1, min(self.config.physics.grid_size, 256), min(self.config.physics.grid_size, 256)),
                compression=compression,
                compression_opts=compression_options,
            )
        if self.config.output.save_exit_wave:
            exit_shape = (
                self.config.run.num_samples,
                self.config.physics.grid_size,
                self.config.physics.grid_size,
            )
            exit_chunks = (
                1,
                min(self.config.physics.grid_size, 256),
                min(self.config.physics.grid_size, 256),
            )
            self.file.create_dataset(
                "exit_wave_real",
                shape=exit_shape,
                dtype=np.float32,
                chunks=exit_chunks,
                compression=compression,
                compression_opts=compression_options,
            )
            self.file.create_dataset(
                "exit_wave_imag",
                shape=exit_shape,
                dtype=np.float32,
                chunks=exit_chunks,
                compression=compression,
                compression_opts=compression_options,
            )
        string_dtype = h5py.string_dtype(encoding="utf-8")
        self.file.create_dataset(
            "metadata_json", shape=(self.config.run.num_samples,), dtype=string_dtype
        )
        self.file.create_dataset(
            "valid", shape=(self.config.run.num_samples,), dtype=np.bool_
        )
        self.file.attrs["config_json"] = config_json
        self.file.attrs["config_sha256"] = config_hash
        self.file.attrs["format_version"] = "1.0"
        self.file.flush()

    def pending_indices(self) -> list[int]:
        valid = np.asarray(self.file["valid"][:], dtype=bool)
        return np.flatnonzero(~valid).astype(int).tolist()

    def write_sample(
        self,
        index: int,
        ideal_intensity: np.ndarray,
        observed_counts: np.ndarray,
        observed_log: np.ndarray,
        exit_wave: np.ndarray,
        metadata: dict[str, Any],
        support: np.ndarray | None = None,
    ) -> None:
        self.file["ideal_intensity"][index] = ideal_intensity
        self.file["observed_counts"][index] = observed_counts
        self.file["observed_log"][index] = observed_log
        if "support" in self.file:
            if support is None:
                raise ValueError("e-CDI sample requires a real-space support mask")
            self.file["support"][index] = support
        if self.config.output.save_exit_wave:
            self.file["exit_wave_real"][index] = exit_wave.real
            self.file["exit_wave_imag"][index] = exit_wave.imag
        self.file["metadata_json"][index] = json.dumps(
            metadata, ensure_ascii=False, sort_keys=True
        )
        self.file["valid"][index] = True
        self.file.flush()

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "H5DatasetWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
