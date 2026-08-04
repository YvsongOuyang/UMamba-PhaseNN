"""Reproducibility, environment, and serialization utilities."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


LOGGER = logging.getLogger(__name__)


def configure_logging(log_level: str, log_path: Path | None = None) -> None:
    """Configure console logging and an optional UTF-8 log file."""

    level = getattr(logging, log_level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def set_seed(seed: int, deterministic: bool) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible inference."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def choose_device(name: str) -> torch.device:
    """Resolve cpu/cuda/auto and fail clearly on an unavailable explicit GPU."""

    normalized = name.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "runtime.device is 'cuda', but CUDA is unavailable. Use --device cpu "
            "only for a small smoke run; full 64^3 evaluation is intended for a GPU."
        )
    if normalized not in {"cpu", "cuda"}:
        raise ValueError("runtime.device must be one of: auto, cpu, cuda.")
    return torch.device(normalized)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute a checkpoint content hash for experiment provenance."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_metadata(path: Path) -> dict[str, Any]:
    """Return path, size, and modification time without hashing large datasets."""

    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "modified_time_epoch": stat.st_mtime,
    }


def environment_metadata(device: torch.device) -> dict[str, Any]:
    """Collect software and accelerator information for reproducibility."""

    metadata: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "working_directory": os.getcwd(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "selected_device": str(device),
    }
    if device.type == "cuda":
        metadata["gpu_name"] = torch.cuda.get_device_name(device)
        metadata["gpu_capability"] = list(torch.cuda.get_device_capability(device))
    return metadata

def save_json(path: Path, payload: Any) -> None:
    """Write deterministic, human-readable UTF-8 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )


def save_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write a union-schema CSV from flat per-sample dictionaries."""

    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in materialized for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def validate_memmap_file(
    path: Path,
    num_samples: int,
    shape: tuple[int, int, int],
    dtype: str,
) -> dict[str, Any]:
    """Validate that a raw memmap file can contain the requested samples."""

    if not path.is_file():
        raise FileNotFoundError(f"Validation data file not found: {path}")
    expected = int(num_samples * np.prod(shape) * np.dtype(dtype).itemsize)
    actual = path.stat().st_size
    if actual < expected:
        raise ValueError(
            f"Data file {path} has {actual} bytes, smaller than the {expected} "
            "bytes required by the configured sample count, shape, and dtype."
        )
    metadata = file_metadata(path)
    metadata.update(
        {
            "expected_minimum_bytes": expected,
            "configured_dtype": dtype,
            "configured_shape_per_sample": list(shape),
            "configured_num_samples": num_samples,
        }
    )
    return metadata
