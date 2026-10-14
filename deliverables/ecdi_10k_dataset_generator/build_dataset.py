"""Generate, verify and index a 10,000-pair electron e-CDI dataset."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ecdi_dataset_builder.config import DatasetPlan  # noqa: E402
from ecdi_dataset_builder.splits import build_splits  # noqa: E402
from electron_diffraction_simulator.config import load_config  # noqa: E402
from electron_diffraction_simulator.pipeline import generate_dataset  # noqa: E402
from electron_diffraction_simulator.training import audit_pairs  # noqa: E402


LOGGER = logging.getLogger("ecdi_10k")
GIB = 1024 ** 3
MIB = 1024 ** 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation-config", type=Path, default=ROOT / "configs/ecdi_10k.json")
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/dataset_plan.json")
    parser.add_argument("--output-root", type=str)
    parser.add_argument("--device", choices=("cpu", "gpu"))
    parser.add_argument("--train-count", type=int)
    parser.add_argument("--test-count", type=int)
    parser.add_argument("--validation-count", type=int)
    parser.add_argument("--free-reserve-gib", type=float)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Audit and make manifests from a completed HDF5; no simulation")
    return parser.parse_args()


def nearest_existing(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise FileNotFoundError(f"No existing ancestor of {path}")
        candidate = candidate.parent
    return candidate


def pending_count(h5_path: Path, expected_count: int) -> int:
    if not h5_path.exists():
        return expected_count
    with h5py.File(h5_path, "r") as source:
        valid = np.asarray(source["valid"][:], dtype=bool)
    if len(valid) != expected_count:
        raise ValueError("Existing HDF5 length differs from the requested sample count")
    return int((~valid).sum())


def estimate_storage(plan: DatasetPlan, image_size: int, pending: int) -> dict[str, float]:
    # ideal/counts/log/real/imag are float32; full-field support is bool.
    raw_gib = plan.total * image_size**2 * (5 * 4 + 1) / GIB
    npy_gib = plan.total * image_size**2 * (3 * 4) / GIB
    compressed_gib = plan.total * plan.estimated_compressed_mib_per_sample / 1024
    required_free_gib = (max(
        pending * plan.estimated_compressed_mib_per_sample / 1024,
        pending * image_size**2 * (5 * 4 + 1) / GIB,
    ) + plan.free_space_reserve_gib)
    return {
        "raw_hdf5_upper_estimate_gib": raw_gib,
        "pilot_based_compressed_hdf5_gib": compressed_gib,
        "optional_three_array_npy_gib": npy_gib,
        "required_free_gib_before_generation": required_free_gib,
    }


def write_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_manifest(path: Path, dataset_path: Path, config_sha256: str,
                   splits: dict[str, list[int]], seed: int, description: str) -> None:
    indices = [index for group in splits.values() for index in group]
    count = sum(len(group) for group in splits.values())
    if count != len(set(indices)) or sorted(indices) != list(range(count)):
        raise ValueError("Split indices must be disjoint and cover every sample exactly once")
    write_json(path, {
        "format_version": "1.0",
        "dataset_path": os.path.relpath(dataset_path.resolve(), path.parent.resolve()),
        "config_sha256": config_sha256,
        "sample_count": count,
        "split_type": description,
        "seed": seed,
        "splits": splits,
        "counts": {name: len(indices) for name, indices in splits.items()},
        "interpretation": "Independent draws from the same six source structures, not new-material generalization",
    })


def prepare_dataset(root: Path, h5_path: Path, plan: DatasetPlan) -> None:
    LOGGER.info("Auditing Fourier pairing and sample integrity: %s", h5_path)
    report, rows = audit_pairs(h5_path, plan.max_fourier_nrmse)
    quality_dir = root / "quality"
    write_json(quality_dir / "validation_report.json", report)
    if not report["passed"]:
        raise ValueError(f"Dataset validation failed: {quality_dir / 'validation_report.json'}")
    if len(rows) != plan.total:
        raise ValueError(f"Expected {plan.total} valid pairs, found {len(rows)}")
    primary, training = build_splits(
        rows, plan.train_count, plan.test_count,
        plan.validation_count_within_train, plan.split_seed,
    )
    manifest_dir = root / "manifests"
    split_filename = f"splits_{plan.train_count}_{plan.test_count}.json"
    write_manifest(manifest_dir / split_filename, h5_path,
                   report["config_sha256"], primary, plan.split_seed,
                   "exact_train_test_stratified_by_material")
    if training is not None:
        write_manifest(manifest_dir / "splits_training_with_val.json", h5_path,
                       report["config_sha256"], training, plan.split_seed,
                       "train_val_test_stratified_by_material")
    quality_dir.mkdir(parents=True, exist_ok=True)
    index_path = quality_dir / "sample_index.jsonl"
    temporary = index_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(index_path)
    file_size = h5_path.stat().st_size
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hdf5_path": str(h5_path),
        "hdf5_bytes": file_size,
        "hdf5_gib": file_size / GIB,
        "sample_count": len(rows),
        "image_shape": report["shape"],
        "material_counts": dict(sorted(Counter(row["material"] for row in rows).items())),
        "train_test_counts": {name: len(values) for name, values in primary.items()},
        "training_counts": ({name: len(values) for name, values in training.items()}
                            if training is not None else None),
        "config_sha256": report["config_sha256"],
        "quality_passed": report["passed"],
        "max_fourier_intensity_nrmse": report["max_fourier_intensity_nrmse"],
    }
    write_json(root / "metadata/dataset_summary.json", summary)
    LOGGER.info("Ready: %d pairs, train=%d test=%d, %.2f GiB",
                len(rows), len(primary["train"]), len(primary["test"]), file_size / GIB)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    plan = DatasetPlan.load(args.plan).with_overrides(
        output_root=args.output_root,
        train_count=args.train_count,
        test_count=args.test_count,
        validation_count_within_train=args.validation_count,
        free_space_reserve_gib=args.free_reserve_gib,
    )
    root = Path(plan.output_root).expanduser().resolve()
    h5_path = root / "data/ecdi_10k.h5"
    simulation = load_config(args.simulation_config)
    simulation = replace(
        simulation,
        run=replace(simulation.run, num_samples=plan.total,
                    device=args.device or simulation.run.device,
                    cupy_cache_dir=str(root / "metadata/cupy_cache")),
        output=replace(simulation.output, path=str(h5_path), overwrite=False),
    )
    simulation.validate()
    if simulation.physics.imaging_mode != "single_shot_ecdi" or not simulation.output.save_exit_wave:
        raise ValueError("A complex exit-wave e-CDI configuration is required")
    pending = pending_count(h5_path, plan.total)
    storage = estimate_storage(plan, simulation.physics.detector_size, pending)
    free_gib = shutil.disk_usage(nearest_existing(root)).free / GIB
    LOGGER.info("Dataset: %d samples (%d train + %d test), %d pending, %s",
                plan.total, plan.train_count, plan.test_count, pending, root)
    LOGGER.info("Disk free %.2f GiB; estimated HDF5 %.2f GiB compressed / %.2f GiB raw; "
                "optional NPY %.2f GiB", free_gib,
                storage["pilot_based_compressed_hdf5_gib"],
                storage["raw_hdf5_upper_estimate_gib"],
                storage["optional_three_array_npy_gib"])
    if args.dry_run:
        return
    if not args.prepare_only and free_gib < storage["required_free_gib_before_generation"]:
        raise OSError("Insufficient free disk space for the estimated remaining samples "
                      f"plus reserve ({storage['required_free_gib_before_generation']:.2f} GiB needed)")
    if not args.prepare_only:
        run_record = {
            "simulation": simulation.to_dict(),
            "split": {"train_count": plan.train_count, "test_count": plan.test_count,
                      "validation_count_within_train": plan.validation_count_within_train,
                      "seed": plan.split_seed},
            "storage_estimate_gib": storage,
        }
        record_path = root / "metadata/generation_plan.json"
        if record_path.exists():
            prior = json.loads(record_path.read_text(encoding="utf-8"))
            if prior["simulation"] != run_record["simulation"] or prior["split"] != run_record["split"]:
                raise ValueError("Existing generation plan differs; choose a new output root")
        else:
            write_json(record_path, run_record)
        if simulation.run.device == "gpu":
            cache_path = Path(simulation.run.cupy_cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("CUPY_CACHE_DIR", str(cache_path))
        generate_dataset(simulation, working_directory=ROOT)
    if not h5_path.is_file():
        raise FileNotFoundError(h5_path)
    prepare_dataset(root, h5_path, plan)


if __name__ == "__main__":
    main()
