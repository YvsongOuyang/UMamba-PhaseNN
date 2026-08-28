"""Configuration objects for the paper-style diffraction simulator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _float_pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a two-element JSON list.")
    lower, upper = float(value[0]), float(value[1])
    if lower > upper:
        raise ValueError(f"{name} lower bound must not exceed its upper bound.")
    return lower, upper


def _int_pair(value: Any, name: str) -> tuple[int, int]:
    lower, upper = _float_pair(value, name)
    if not lower.is_integer() or not upper.is_integer():
        raise ValueError(f"{name} bounds must be integers.")
    return int(lower), int(upper)


@dataclass(frozen=True)
class ShapeConfig:
    types: tuple[str, ...]
    oversampling_ratio: tuple[float, float]
    wulff_111_to_100_distance: tuple[float, float]
    wulff_110_to_100_distance: tuple[float, float]
    winterbottom_retained_height: tuple[float, float]
    random_plane_count: tuple[int, int]
    random_plane_offset_fraction: tuple[float, float]


@dataclass(frozen=True)
class PhaseConfig:
    types: tuple[str, ...]
    peak_to_peak_pi: tuple[float, float]
    gaussian_center_fraction: tuple[float, float]
    gaussian_sigma_fraction: tuple[float, float]
    cosine_cycles: tuple[float, float]
    correlation_length_voxels: tuple[float, float]


@dataclass(frozen=True)
class NoiseConfig:
    enabled: bool
    peak_photons: tuple[float, float]
    sample_log_uniform: bool


@dataclass(frozen=True)
class SimulationConfig:
    schema_version: str
    description: str
    grid_size: int
    shape: ShapeConfig
    phase: PhaseConfig
    noise: NoiseConfig
    provenance: dict[str, list[str]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "schema_version": self.schema_version,
            "description": self.description,
            "grid_size": self.grid_size,
            "shape": {
                "types": list(self.shape.types),
                "oversampling_ratio": list(self.shape.oversampling_ratio),
                "wulff_111_to_100_distance": list(
                    self.shape.wulff_111_to_100_distance
                ),
                "wulff_110_to_100_distance": list(
                    self.shape.wulff_110_to_100_distance
                ),
                "winterbottom_retained_height": list(
                    self.shape.winterbottom_retained_height
                ),
                "random_plane_count": list(self.shape.random_plane_count),
                "random_plane_offset_fraction": list(
                    self.shape.random_plane_offset_fraction
                ),
            },
            "phase": {
                "types": list(self.phase.types),
                "peak_to_peak_pi": list(self.phase.peak_to_peak_pi),
                "gaussian_center_fraction": list(
                    self.phase.gaussian_center_fraction
                ),
                "gaussian_sigma_fraction": list(
                    self.phase.gaussian_sigma_fraction
                ),
                "cosine_cycles": list(self.phase.cosine_cycles),
                "correlation_length_voxels": list(
                    self.phase.correlation_length_voxels
                ),
            },
            "noise": {
                "enabled": self.noise.enabled,
                "peak_photons": list(self.noise.peak_photons),
                "sample_log_uniform": self.noise.sample_log_uniform,
            },
            "provenance": self.provenance,
        }


def load_simulation_config(path: str | Path) -> SimulationConfig:
    """Load and validate a simulator JSON configuration."""

    config_path = Path(path).expanduser().resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    grid_size = int(raw["grid_size"])
    if grid_size < 8 or grid_size % 2:
        raise ValueError("grid_size must be an even integer of at least 8.")

    shape_raw = raw["shape"]
    shape = ShapeConfig(
        types=tuple(str(value) for value in shape_raw["types"]),
        oversampling_ratio=_float_pair(
            shape_raw["oversampling_ratio"], "shape.oversampling_ratio"
        ),
        wulff_111_to_100_distance=_float_pair(
            shape_raw["wulff_111_to_100_distance"],
            "shape.wulff_111_to_100_distance",
        ),
        wulff_110_to_100_distance=_float_pair(
            shape_raw["wulff_110_to_100_distance"],
            "shape.wulff_110_to_100_distance",
        ),
        winterbottom_retained_height=_float_pair(
            shape_raw["winterbottom_retained_height"],
            "shape.winterbottom_retained_height",
        ),
        random_plane_count=_int_pair(
            shape_raw["random_plane_count"], "shape.random_plane_count"
        ),
        random_plane_offset_fraction=_float_pair(
            shape_raw["random_plane_offset_fraction"],
            "shape.random_plane_offset_fraction",
        ),
    )
    if shape.oversampling_ratio[0] <= 2.0:
        raise ValueError("The paper requires oversampling_ratio > 2.")

    phase_raw = raw["phase"]
    phase = PhaseConfig(
        types=tuple(str(value) for value in phase_raw["types"]),
        peak_to_peak_pi=_float_pair(
            phase_raw["peak_to_peak_pi"], "phase.peak_to_peak_pi"
        ),
        gaussian_center_fraction=_float_pair(
            phase_raw["gaussian_center_fraction"],
            "phase.gaussian_center_fraction",
        ),
        gaussian_sigma_fraction=_float_pair(
            phase_raw["gaussian_sigma_fraction"],
            "phase.gaussian_sigma_fraction",
        ),
        cosine_cycles=_float_pair(
            phase_raw["cosine_cycles"], "phase.cosine_cycles"
        ),
        correlation_length_voxels=_float_pair(
            phase_raw["correlation_length_voxels"],
            "phase.correlation_length_voxels",
        ),
    )
    noise_raw = raw["noise"]
    noise = NoiseConfig(
        enabled=bool(noise_raw["enabled"]),
        peak_photons=_float_pair(noise_raw["peak_photons"], "noise.peak_photons"),
        sample_log_uniform=bool(noise_raw["sample_log_uniform"]),
    )
    return SimulationConfig(
        schema_version=str(raw["schema_version"]),
        description=str(raw["description"]),
        grid_size=grid_size,
        shape=shape,
        phase=phase,
        noise=noise,
        provenance={
            str(key): [str(item) for item in value]
            for key, value in raw.get("provenance", {}).items()
        },
    )
