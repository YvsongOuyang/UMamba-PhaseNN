"""Fixed author NPZ training data; no particle simulation in DataLoader workers."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from simulation.sample_io import load_reciprocal_phase


def prepare_author_training_sample(
    stored: Any, *, input_log_data: bool, shape: tuple[int, int, int]
) -> dict[str, torch.Tensor]:
    """Match source I/phi preprocessing; I is already noisy linear intensity."""
    intensity = np.asarray(stored["I"], dtype=np.float32)
    if intensity.shape != shape or not np.isfinite(intensity).all() or np.any(intensity < 0):
        raise ValueError(f"I must be finite nonnegative intensity with shape {shape}.")
    phase = load_reciprocal_phase(stored)
    if phase.shape != shape:
        raise ValueError("Intensity and reciprocal-phase shapes differ.")
    model_input = np.log1p(intensity) if input_log_data else intensity
    low, high = float(model_input.min()), float(model_input.max())
    if high <= low:
        raise ValueError("Constant intensity gives zero total WCA weight.")
    model_input = (model_input - low) / (high - low)
    phase = phase - phase[tuple(size // 2 for size in shape)]
    return {
        "input": torch.from_numpy(np.ascontiguousarray(model_input[None])),
        "target_phase": torch.from_numpy(np.ascontiguousarray(phase)),
    }


class AuthorNPZPhaseDataset(Dataset):
    """Read a particle-disjoint split of a generated author dataset manifest.

    Only filenames survive initialization, not the large per-sample metadata
    or open NPZ handles. Spawn workers therefore copy a small read-only index.
    """

    def __init__(
        self, root: str | Path, split: str, *, num_samples: int | None = None,
        shape: tuple[int, int, int] = (64, 64, 64), input_log_data: bool = True,
        min_oversampling: float | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.shape = tuple(shape)
        self.input_log_data = input_log_data
        if min_oversampling is not None and (not np.isfinite(min_oversampling) or min_oversampling <= 0):
            raise ValueError("Minimum oversampling must be finite and positive.")
        manifest_path = self.root / "dataset_manifest.json"
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
        if manifest.get("route") != "author_generator":
            raise ValueError("Expected an author_generator dataset manifest.")
        if manifest.get("split_unit") != "particle" or split not in manifest.get("splits", {}):
            raise ValueError("Generate particle-disjoint splits with --split-counts TRAIN VAL TEST.")
        records = manifest["samples"]
        if len(records) != manifest["num_samples"]:
            raise ValueError("Dataset manifest is incomplete.")
        filenames, seen_files, particles, counts = [], set(), {}, {}
        eligible_counts: dict[str, int] = {}
        for record in records:
            name, record_split = record["filename"], record["split"]
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe sample filename: {name}")
            if name in seen_files:
                raise ValueError(f"Duplicate sample: {name}")
            seen_files.add(name)
            metadata = record["metadata"]
            # Seeds identify geometry even if someone renumbers particle IDs.
            particle = (int(metadata["particle_seed"]), metadata["shape"])
            previous = particles.setdefault(particle, record_split)
            if previous != record_split:
                raise ValueError(f"Particle leakage between {previous} and {record_split}.")
            counts[record_split] = counts.get(record_split, 0) + 1
            eligible = True
            if min_oversampling is not None:
                measured = np.asarray(metadata.get("measured_object_oversampling_xyz"), dtype=np.float64)
                if measured.shape != (3,) or not np.isfinite(measured).all() or np.any(measured <= 0):
                    raise ValueError(f"Missing or invalid oversampling metadata for {name}; cannot apply filter.")
                eligible = bool(np.all(measured > min_oversampling))
            if eligible:
                eligible_counts[record_split] = eligible_counts.get(record_split, 0) + 1
            if record_split == split and eligible:
                filenames.append(name)
        declared = manifest["splits"]
        if (
            any(counts.get(key, 0) != count for key, count in declared.items())
            or set(counts) - set(declared)
        ):
            raise ValueError("Declared split sizes disagree with sample records.")
        if num_samples is not None:
            if not 0 < num_samples <= len(filenames):
                raise ValueError(f"Requested {num_samples} samples but {split} has {len(filenames)} eligible samples.")
            filenames = filenames[:num_samples]
        if not filenames:
            raise ValueError(f"Empty {split} split after selection/filtering.")
        for name in filenames:
            path = (self.root / name).resolve()
            if self.root not in path.parents or not path.is_file():
                raise ValueError(f"Missing sample or path outside dataset: {name}")
        self.filenames = tuple(filenames)
        self.manifest = {
            "format": "author_npz", "root": str(self.root), "split": split,
            "num_samples": len(self), "available_samples": eligible_counts[split],
            "unfiltered_samples": declared[split],
            "oversampling_filter": {
                "min_exclusive_per_axis": min_oversampling,
                "source": "manifest measured_object_oversampling_xyz; no array modification",
                "original_split_counts": declared,
                "eligible_split_counts": {key: eligible_counts.get(key, 0) for key in declared},
                "excluded_split_counts": {key: count - eligible_counts.get(key, 0) for key, count in declared.items()},
            },
            "selection": "manifest_order_prefix" if num_samples is not None else "all_eligible",
            "selected_filenames_sha256": hashlib.sha256(
                json.dumps(filenames, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "shape": list(self.shape), "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "storage_schema": manifest.get("storage_schema", "author_standard"),
            "generator_protocol": manifest.get("generator_protocol"),
            "source": manifest.get("author_source_manifest"),
            "backend": manifest.get("compatibility"),
            "input": "minmax(log1p(I))" if input_log_data else "minmax(I)",
            "target": "stored phi or clean object FFT; subtract central phase; no COM correction",
            "noise": "fixed stored I; no resampling",
            "split_unit": "particle",
        }

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> dict:
        filename = self.filenames[index]
        with np.load(self.root / filename, allow_pickle=False) as stored:
            sample = prepare_author_training_sample(
                stored, input_log_data=self.input_log_data, shape=self.shape
            )
        sample["name"] = filename
        return sample


def initialize_data_worker(worker_id: int) -> None:
    """One CPU compute thread per worker; workers never initialize CUDA."""
    torch.set_num_threads(1)
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
