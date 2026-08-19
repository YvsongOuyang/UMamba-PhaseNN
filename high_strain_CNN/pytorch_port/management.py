"""Version, runtime, and dataset-manifest helpers."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_CONFIG = PROJECT_DIR / "configs" / "autophasenn_data.json"


def project_version() -> str:
    """Return the semantic version stored with this source tree."""

    return (PROJECT_DIR / "VERSION").read_text(encoding="utf-8").strip()


def git_commit() -> str | None:
    """Return the parent repository commit when Git metadata is available."""

    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_DIR), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def git_dirty() -> bool | None:
    """Report whether this vendored source directory has uncommitted changes."""

    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_DIR), "status", "--porcelain", "--", "."],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def load_data_config(path: str | Path = DEFAULT_DATA_CONFIG) -> dict[str, Any]:
    """Load and validate the shared AutoPhaseNN memmap configuration."""

    config_path = Path(path).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "dataset_name",
        "dataset_version",
        "root",
        "shape",
        "dtypes",
        "splits",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Data config is missing fields: {sorted(missing)}")
    shape = tuple(int(size) for size in config["shape"])
    if len(shape) != 3 or any(size < 1 for size in shape):
        raise ValueError("Data config shape must contain three positive dimensions.")
    for kind in ("diffraction", "realspace"):
        if kind not in config["dtypes"]:
            raise ValueError(f"Data config is missing dtype {kind!r}.")
        np.dtype(config["dtypes"][kind])
    for split_name, split in config["splits"].items():
        split_missing = {"diffraction", "realspace", "num_samples"}.difference(split)
        if split_missing:
            raise ValueError(
                f"Data split {split_name!r} is missing fields: {sorted(split_missing)}"
            )
        if int(split["num_samples"]) < 1:
            raise ValueError(f"Data split {split_name!r} must contain samples.")
    config["config_path"] = str(config_path)
    return config


def expected_memmap_bytes(
    num_samples: int,
    shape: tuple[int, int, int] | list[int],
    dtype: str,
) -> int:
    """Calculate the exact raw-memmap byte count for one dataset file."""

    return int(num_samples) * int(np.prod(shape)) * np.dtype(dtype).itemsize


def build_data_manifest(
    *,
    config: dict[str, Any],
    root: str | Path,
    shape: tuple[int, int, int],
    diffraction_dtype: str,
    realspace_dtype: str,
    splits: dict[str, dict[str, Any]],
    input_log_data: bool,
) -> dict[str, Any]:
    """Create a resolved, JSON-safe snapshot of the data used by a run."""

    root_path = Path(root).expanduser().resolve()
    resolved_splits: dict[str, dict[str, Any]] = {}
    for name, split in splits.items():
        count = int(split["num_samples"])
        diff_path = root_path / str(split["diffraction"])
        real_path = root_path / str(split["realspace"])
        resolved_splits[name] = {
            "num_samples": count,
            "diffraction": str(diff_path),
            "realspace": str(real_path),
            "expected_diffraction_bytes": expected_memmap_bytes(
                count,
                shape,
                diffraction_dtype,
            ),
            "expected_realspace_bytes": expected_memmap_bytes(
                count,
                shape,
                realspace_dtype,
            ),
        }
    return {
        "schema_version": int(config["schema_version"]),
        "dataset_name": str(config["dataset_name"]),
        "dataset_version": str(config["dataset_version"]),
        "source_config": str(config["config_path"]),
        "storage": config.get("storage", "raw_numpy_memmap"),
        "root": str(root_path),
        "shape": list(shape),
        "dtypes": {
            "diffraction": np.dtype(diffraction_dtype).name,
            "realspace": np.dtype(realspace_dtype).name,
        },
        "input_log_data": bool(input_log_data),
        "input_preprocessing": config.get("input_preprocessing", {}),
        "target_preprocessing": config.get("target_preprocessing", {}),
        "semantics": config.get("semantics", {}),
        "splits": resolved_splits,
    }


def data_file_status(
    manifest: dict[str, Any],
    *,
    require_exact_size: bool = False,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Inspect resolved files without loading their array contents."""

    status: dict[str, dict[str, dict[str, Any]]] = {}
    for split_name, split in manifest["splits"].items():
        split_status: dict[str, dict[str, Any]] = {}
        for kind, expected_key in (
            ("diffraction", "expected_diffraction_bytes"),
            ("realspace", "expected_realspace_bytes"),
        ):
            path = Path(split[kind])
            expected_bytes = int(split[expected_key])
            exists = path.is_file()
            actual_bytes = path.stat().st_size if exists else None
            size_valid = bool(
                actual_bytes == expected_bytes
                if require_exact_size
                else actual_bytes is not None and actual_bytes >= expected_bytes
            )
            split_status[kind] = {
                "path": str(path),
                "exists": exists,
                "expected_bytes": expected_bytes,
                "actual_bytes": actual_bytes,
                "size_valid": size_valid,
            }
        status[split_name] = split_status
    return status


def require_data_files(
    manifest: dict[str, Any],
    *,
    require_exact_size: bool = False,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Fail early when configured memmaps are absent or too small."""

    status = data_file_status(manifest, require_exact_size=require_exact_size)
    failures = [
        details
        for split in status.values()
        for details in split.values()
        if not details["size_valid"]
    ]
    if failures:
        messages = [
            (
                f"{item['path']}: expected "
                f"{'exactly ' if require_exact_size else 'at least '}"
                f"{item['expected_bytes']} bytes, found {item['actual_bytes']}"
            )
            for item in failures
        ]
        raise ValueError("Invalid AutoPhaseNN data files:\n" + "\n".join(messages))
    return status


def runtime_manifest(device: torch.device | None = None) -> dict[str, Any]:
    """Collect the software and accelerator versions for reproducibility."""

    resolved_device = device or torch.device("cpu")
    gpu_name = None
    if resolved_device.type == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(resolved_device)
    return {
        "project_version": project_version(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(resolved_device),
        "gpu_name": gpu_name,
    }
