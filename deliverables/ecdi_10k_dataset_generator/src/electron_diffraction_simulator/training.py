"""Quality checks and leakage-aware index manifests for exit-wave training."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def audit_pairs(path: Path, max_fourier_nrmse: float = 1e-5) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify the saved ideal pattern is the FFT intensity of its exit wave."""
    samples: list[dict[str, Any]] = []
    failures: list[str] = []
    errors: list[float] = []
    count_totals: list[float] = []
    gauge_coherences: list[float] = []
    fingerprints: set[str] = set()
    seeds: set[int] = set()
    with h5py.File(path, "r") as source:
        config_sha256 = str(source.attrs.get("config_sha256", ""))
        required = {"ideal_intensity", "observed_counts", "exit_wave_real",
                    "exit_wave_imag", "metadata_json", "valid"}
        missing = required.difference(source.keys())
        if missing:
            raise ValueError(f"Missing required HDF5 datasets: {sorted(missing)}")
        n = len(source["valid"])
        if not np.asarray(source["valid"][:], dtype=bool).all():
            failures.append("Some samples are not marked valid")
        shape = tuple(source["ideal_intensity"].shape[1:])
        if shape != tuple(source["exit_wave_real"].shape[1:]):
            raise ValueError("Detector and exit-wave grid shapes differ")
        for index in range(n):
            if not bool(source["valid"][index]):
                continue
            metadata = json.loads(source["metadata_json"].asstr()[index])
            ideal = np.asarray(source["ideal_intensity"][index], dtype=np.float64)
            counts = np.asarray(source["observed_counts"][index], dtype=np.float64)
            wave = (np.asarray(source["exit_wave_real"][index], dtype=np.float64)
                    + 1j * np.asarray(source["exit_wave_imag"][index], dtype=np.float64))
            if not (np.isfinite(ideal).all() and np.isfinite(counts).all()
                    and np.isfinite(wave).all()):
                failures.append(f"sample {index}: nonfinite values")
                continue
            if ideal.min() < 0 or counts.min() < 0 or ideal.sum() <= 0 or counts.sum() <= 0:
                failures.append(f"sample {index}: invalid nonnegative intensity/count totals")
                continue
            spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(wave)))
            expected = np.abs(spectrum) ** 2
            expected /= expected.sum()
            original_ideal_sum = float(ideal.sum())
            ideal /= ideal.sum()
            error = float(np.linalg.norm(expected - ideal) / np.linalg.norm(ideal))
            errors.append(error)
            if error > max_fourier_nrmse:
                failures.append(f"sample {index}: Fourier intensity NRMSE {error:.3g}")
            if abs(original_ideal_sum - 1.0) > 1e-5:
                failures.append(f"sample {index}: ideal intensity normalization failure")
            geometry = metadata["physics"].get("sample_geometry")
            if geometry != "extended_crystal":
                failures.append(f"sample {index}: unexpected geometry {geometry}")
            if metadata["physics"].get("exact_support") is not False:
                failures.append(f"sample {index}: expected exact_support=false")
            if "support" in source and not np.asarray(source["support"][index], dtype=bool).all():
                failures.append(f"sample {index}: extended-crystal support is not full-field")
            gauge_coherence = float(abs(np.sum(wave * np.abs(wave))) /
                                    max(float(np.sum(np.abs(wave) ** 2)), 1e-12))
            gauge_coherences.append(gauge_coherence)
            if gauge_coherence < 0.05:
                failures.append(f"sample {index}: unstable global-phase gauge")
            seed = int(metadata["sample_seed"])
            if seed in seeds:
                failures.append(f"sample {index}: duplicate sample seed {seed}")
            seeds.add(seed)
            fingerprint = hashlib.blake2b(wave.tobytes(), digest_size=16).hexdigest()
            if fingerprint in fingerprints:
                failures.append(f"sample {index}: duplicate exit wave")
            fingerprints.add(fingerprint)
            count_totals.append(float(counts.sum()))
            samples.append({
                "index": index,
                "material": str(metadata["structure"]["source_name"]),
                "sample_seed": seed,
                "fourier_intensity_nrmse": error,
                "total_counts": float(counts.sum()),
                "vacancy_fraction": float(metadata["structure"].get("vacancy_fraction", 0.0)),
                "removed_atoms": int(metadata["structure"].get("removed_atoms", 0)),
                "gauge_coherence": gauge_coherence,
            })
    report = {
        "dataset": str(path.resolve()),
        "config_sha256": config_sha256,
        "sample_count": len(samples),
        "shape": list(shape),
        "materials": dict(sorted(Counter(sample["material"] for sample in samples).items())),
        "max_fourier_intensity_nrmse": max(errors, default=None),
        "median_fourier_intensity_nrmse": float(np.median(errors)) if errors else None,
        "total_counts_min": min(count_totals, default=None),
        "total_counts_max": max(count_totals, default=None),
        "minimum_gauge_coherence": min(gauge_coherences, default=None),
        "samples_with_removed_atoms": sum(row["removed_atoms"] > 0 for row in samples),
        "failures": failures,
        "passed": not failures and len(samples) == n,
    }
    return report, samples


def stratified_split(samples: list[dict[str, Any]], seed: int,
                     train_fraction: float = 0.8, val_fraction: float = 0.1) -> dict[str, list[int]]:
    """Split independent simulated draws within each material (interpolation test)."""
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1 or train_fraction + val_fraction >= 1:
        raise ValueError("train/validation fractions must be positive with a positive test fraction")
    groups: dict[str, list[int]] = {}
    for row in samples:
        groups.setdefault(row["material"], []).append(int(row["index"]))
    rng = np.random.default_rng(seed)
    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for material, indices in sorted(groups.items()):
        if len(indices) < 3:
            raise ValueError(f"Material {material} needs at least 3 samples for a three-way split")
        shuffled = rng.permutation(indices).tolist()
        n_val = max(1, int(round(len(indices) * val_fraction)))
        n_test = max(1, int(round(len(indices) * (1 - train_fraction - val_fraction))))
        if n_val + n_test >= len(indices):
            raise ValueError(f"Material {material} has too few samples for these fractions")
        splits["val"].extend(shuffled[:n_val])
        splits["test"].extend(shuffled[n_val:n_val + n_test])
        splits["train"].extend(shuffled[n_val + n_test:])
    return {name: sorted(indices) for name, indices in splits.items()}


def material_holdout_split(samples: list[dict[str, Any]], val_material: str,
                           test_material: str) -> dict[str, list[int]]:
    """Keep entire materials unseen during training (extrapolation test)."""
    materials = {row["material"] for row in samples}
    if val_material == test_material or {val_material, test_material}.difference(materials):
        raise ValueError("Validation/test holdout materials must be different and present")
    return {
        "train": [row["index"] for row in samples
                  if row["material"] not in {val_material, test_material}],
        "val": [row["index"] for row in samples if row["material"] == val_material],
        "test": [row["index"] for row in samples if row["material"] == test_material],
    }


def write_split_manifest(path: Path, dataset_path: Path, splits: dict[str, list[int]],
                         split_type: str, seed: int | None) -> None:
    """Save lightweight indices without duplicating large image arrays."""
    path.parent.mkdir(parents=True, exist_ok=True)
    all_indices = [index for values in splits.values() for index in values]
    if len(all_indices) != len(set(all_indices)):
        raise ValueError("Split indices overlap")
    with h5py.File(dataset_path, "r") as source:
        config_sha256 = str(source.attrs.get("config_sha256", ""))
        dataset_size = len(source["valid"])
    if sorted(all_indices) != list(range(dataset_size)):
        raise ValueError("Split indices must partition every sample exactly once")
    manifest = {
        "format_version": "1.0",
        "dataset_path": os.path.relpath(dataset_path.resolve(), path.parent.resolve()),
        "config_sha256": config_sha256,
        "sample_count": dataset_size,
        "split_type": split_type,
        "seed": seed,
        "splits": splits,
        "counts": {name: len(values) for name, values in splits.items()},
        "interpretation": (
            "Independent draws of the same source CIFs; tests interpolation, not new-material generalization"
            if split_type == "stratified_by_material"
            else "Entire held-out materials; tests cross-material generalization"
        ),
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
