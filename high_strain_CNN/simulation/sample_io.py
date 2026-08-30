"""Shared NPZ schema used by the AutoPhaseNN export and evaluation tools."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


COMPACT_STORAGE = "author_compact_v1"


def load_reciprocal_phase(stored: Any) -> np.ndarray:
    """Read legacy phi or derive it from the clean object, without recentering.

    The compact object is complex64, so FFT labels agree numerically rather
    than bitwise with labels calculated before the author's complex128 cast.
    """
    if "phi" in stored:
        phase = np.asarray(stored["phi"], dtype=np.float32)
    elif "object" in stored:
        obj = np.asarray(stored["object"])
        if obj.ndim != 3 or not np.iscomplexobj(obj) or not np.isfinite(obj).all():
            raise ValueError("Compact labels require a finite complex 3D object.")
        axes = (-3, -2, -1)
        reciprocal = np.fft.ifftshift(
            np.fft.fftn(np.fft.fftshift(obj, axes=axes), axes=axes), axes=axes
        )
        phase = np.angle(reciprocal).astype(np.float32)
    else:
        raise ValueError("NPZ must contain phi or a clean complex object.")
    if phase.ndim != 3 or not np.isfinite(phase).all():
        raise ValueError("Reciprocal phase must be a finite 3D array.")
    return phase


@dataclass(frozen=True)
class SimulatedSample:
    """One author-loader-compatible diffraction sample and optional ground truth."""

    intensity: np.ndarray
    reciprocal_phase: np.ndarray
    support: np.ndarray
    object_phase: np.ndarray
    realspace_object: np.ndarray
    clean_intensity: np.ndarray
    metadata: dict[str, Any]


def save_sample(
    sample: SimulatedSample,
    path: str | Path,
    *,
    save_extras: bool = False,
) -> Path:
    """Write one compressed NPZ using the official loader's ``I``/``phi`` keys."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "I": sample.intensity.astype(np.float32, copy=False),
        "phi": sample.reciprocal_phase.astype(np.float32, copy=False),
        "metadata_json": np.asarray(json.dumps(sample.metadata, sort_keys=True)),
    }
    if save_extras:
        arrays.update(
            {
                "support": sample.support.astype(np.uint8, copy=False),
                "object_phase": sample.object_phase.astype(np.float32, copy=False),
                "object": sample.realspace_object.astype(np.complex64, copy=False),
                "I_clean": sample.clean_intensity.astype(np.float32, copy=False),
            }
        )
    np.savez_compressed(destination, **arrays)
    return destination
