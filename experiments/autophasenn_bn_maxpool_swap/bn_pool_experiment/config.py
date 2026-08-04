"""Typed YAML configuration for the BN/MaxPool swap experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml


@dataclass
class DataConfig:
    """Validation memmap configuration."""

    mode: str = "memmap"
    data_dir: str = "/data_ssd/oyys/autophasenn"
    diff_file: str = "val_diff.npy"
    real_file: Optional[str] = "val_real.npy"
    num_samples: int = 5000
    shape: list[int] = field(default_factory=lambda: [64, 64, 64])
    dtype_diff: str = "float32"
    dtype_real: str = "complex64"
    scale_i: float = 0.0
    random_distribution: str = "uniform"
    random_min: float = 0.0
    random_max: float = 1.0
    random_seed: int = 20260803


@dataclass
class ModelConfig:
    """Checkpoint and model-output configuration."""

    checkpoint: str = "REPLACE_WITH_PRETRAINED_CHECKPOINT.pt"
    threshold: Optional[float] = None
    strict_load: bool = True
    allow_unsafe_checkpoint: bool = False


@dataclass
class RuntimeConfig:
    """Runtime and reproducibility configuration."""

    device: str = "cuda"
    batch_size: int = 1
    num_workers: int = 0
    seed: int = 20260708
    deterministic: bool = True
    hash_checkpoint: bool = True
    log_level: str = "INFO"


@dataclass
class MetricsConfig:
    """Metric implementation parameters."""

    amplitude_data_range: float = 1.0
    ssim_window_size: int = 7
    histogram_bins: int = 64


@dataclass
class StatisticsConfig:
    """Paired bootstrap configuration."""

    bootstrap_samples: int = 2000
    confidence_level: float = 0.95
    seed: int = 20260708


@dataclass
class AcceptanceConfig:
    """Explicit engineering criteria for the phrase 'small impact'."""

    primary_metrics: list[str] = field(
        default_factory=lambda: [
            "paper_modulus_mae",
            "real_amp_psnr",
            "real_amp_ssim3d",
            "real_phase_mae_true_support",
        ]
    )
    max_primary_metric_degradation_percent: float = 1.0
    max_farfield_relative_l1: float = 0.001
    min_positive_bn_scale_fraction: float = 1.0


@dataclass
class OutputConfig:
    """Output directory configuration."""

    root_dir: str = "experiments/autophasenn_bn_maxpool_swap/outputs"
    run_name: Optional[str] = None


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)
    acceptance: AcceptanceConfig = field(default_factory=AcceptanceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration mapping."""

        return asdict(self)


def _build_section(section_type: type, raw: Any, section_name: str):
    if raw is None:
        return section_type()
    if not isinstance(raw, Mapping):
        raise TypeError(f"Configuration section {section_name!r} must be a mapping.")
    allowed = set(section_type.__dataclass_fields__)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown keys in configuration section {section_name!r}: {unknown}"
        )
    return section_type(**dict(raw))


def _validate(config: ExperimentConfig) -> None:
    if config.data.mode not in {"memmap", "random"}:
        raise ValueError("data.mode must be either 'memmap' or 'random'.")
    if config.data.random_distribution != "uniform":
        raise ValueError("Only data.random_distribution='uniform' is currently supported.")
    if config.data.random_max <= config.data.random_min:
        raise ValueError("data.random_max must be greater than data.random_min.")
    if config.data.num_samples <= 0:
        raise ValueError("data.num_samples must be positive.")
    if len(config.data.shape) != 3 or any(int(v) <= 0 for v in config.data.shape):
        raise ValueError("data.shape must contain three positive dimensions.")
    if config.runtime.batch_size <= 0:
        raise ValueError("runtime.batch_size must be positive.")
    if config.runtime.num_workers < 0:
        raise ValueError("runtime.num_workers cannot be negative.")
    if config.metrics.ssim_window_size < 3 or config.metrics.ssim_window_size % 2 == 0:
        raise ValueError("metrics.ssim_window_size must be an odd integer >= 3.")
    if config.metrics.histogram_bins < 2:
        raise ValueError("metrics.histogram_bins must be >= 2.")
    if config.statistics.bootstrap_samples < 0:
        raise ValueError("statistics.bootstrap_samples cannot be negative.")
    if not 0.0 < config.statistics.confidence_level < 1.0:
        raise ValueError("statistics.confidence_level must be between 0 and 1.")
    if config.acceptance.max_primary_metric_degradation_percent < 0.0:
        raise ValueError(
            "acceptance.max_primary_metric_degradation_percent cannot be negative."
        )
    if not 0.0 <= config.acceptance.min_positive_bn_scale_fraction <= 1.0:
        raise ValueError(
            "acceptance.min_positive_bn_scale_fraction must be between 0 and 1."
        )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML file."""

    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise TypeError("The root YAML value must be a mapping.")

    section_types = {
        "data": DataConfig,
        "model": ModelConfig,
        "runtime": RuntimeConfig,
        "metrics": MetricsConfig,
        "statistics": StatisticsConfig,
        "acceptance": AcceptanceConfig,
        "output": OutputConfig,
    }
    unknown = sorted(set(raw) - set(section_types))
    if unknown:
        raise ValueError(f"Unknown top-level configuration keys: {unknown}")

    config = ExperimentConfig(
        **{
            name: _build_section(section_type, raw.get(name), name)
            for name, section_type in section_types.items()
        }
    )
    _validate(config)
    return config
