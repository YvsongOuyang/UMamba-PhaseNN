"""Configuration and reports for evaluating several AutoPhaseNN datasets."""

from __future__ import annotations

import copy
import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from .config import ExperimentConfig, load_experiment_config


@dataclass
class DatasetSpec:
    """One memmap dataset following ``AutoPhaseDataset`` conventions."""

    name: str
    diff_file: str
    real_file: Optional[str]
    num_samples: int
    enabled: bool = True
    data_dir: Optional[str] = None
    shape: list[int] = field(default_factory=lambda: [64, 64, 64])
    dtype_diff: str = "float32"
    dtype_real: str = "complex64"
    scale_i: float = 0.0


@dataclass
class MultiDatasetConfig:
    """Top-level configuration for a multi-dataset paired evaluation."""

    base_config: str
    checkpoint: str
    data_dir: str
    output_dir: str
    datasets: list[DatasetSpec]
    run_name: Optional[str] = None
    device: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping."""

        return asdict(self)


def _build_dataset(raw: Any, index: int) -> DatasetSpec:
    if not isinstance(raw, Mapping):
        raise TypeError(f"datasets[{index}] must be a mapping.")
    allowed = set(DatasetSpec.__dataclass_fields__)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown keys in datasets[{index}]: {unknown}")
    required = {"name", "diff_file", "real_file", "num_samples"}
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"Missing keys in datasets[{index}]: {missing}")
    return DatasetSpec(**dict(raw))


def _validate(config: MultiDatasetConfig) -> None:
    if not config.base_config:
        raise ValueError("base_config cannot be empty.")
    if not config.checkpoint:
        raise ValueError("checkpoint cannot be empty.")
    if not config.data_dir:
        raise ValueError("data_dir cannot be empty.")
    if not config.output_dir:
        raise ValueError("output_dir cannot be empty.")
    if config.device is not None and config.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda.")
    if not config.datasets:
        raise ValueError("At least one dataset must be configured.")

    names: set[str] = set()
    enabled_count = 0
    for dataset in config.datasets:
        if not dataset.name or dataset.name in {".", ".."}:
            raise ValueError("Every dataset must have a non-empty safe name.")
        if Path(dataset.name).name != dataset.name:
            raise ValueError(f"Dataset name must not contain path separators: {dataset.name}")
        if dataset.name in names:
            raise ValueError(f"Duplicate dataset name: {dataset.name}")
        names.add(dataset.name)
        if dataset.enabled:
            enabled_count += 1
        if dataset.num_samples <= 0:
            raise ValueError(f"Dataset {dataset.name}: num_samples must be positive.")
        if len(dataset.shape) != 3 or any(int(value) <= 0 for value in dataset.shape):
            raise ValueError(
                f"Dataset {dataset.name}: shape must contain three positive dimensions."
            )
        if not dataset.diff_file:
            raise ValueError(f"Dataset {dataset.name}: diff_file cannot be empty.")
    if enabled_count == 0:
        raise ValueError("At least one dataset must be enabled.")


def load_multi_dataset_config(path: str | Path) -> MultiDatasetConfig:
    """Load and strictly validate a multi-dataset YAML file."""

    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise TypeError("The root YAML value must be a mapping.")
    allowed = {
        "base_config",
        "checkpoint",
        "data_dir",
        "output_dir",
        "run_name",
        "device",
        "datasets",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown top-level configuration keys: {unknown}")
    required = {"base_config", "checkpoint", "data_dir", "output_dir", "datasets"}
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"Missing top-level configuration keys: {missing}")
    datasets_raw = raw["datasets"]
    if not isinstance(datasets_raw, list):
        raise TypeError("datasets must be a list.")
    config = MultiDatasetConfig(
        base_config=str(raw["base_config"]),
        checkpoint=str(raw["checkpoint"]),
        data_dir=str(raw["data_dir"]),
        output_dir=str(raw["output_dir"]),
        run_name=raw.get("run_name"),
        device=raw.get("device"),
        datasets=[_build_dataset(item, index) for index, item in enumerate(datasets_raw)],
    )
    _validate(config)
    return config


def resolve_base_config_path(multi_path: Path, configured_path: str) -> Path:
    """Resolve a base config relative to the multi-dataset config when possible."""

    candidate = Path(configured_path)
    if candidate.is_absolute() or candidate.is_file():
        return candidate
    relative_candidate = multi_path.resolve().parent / candidate
    if relative_candidate.is_file():
        return relative_candidate
    return candidate


def build_experiment_config(
    base_config: ExperimentConfig,
    multi_config: MultiDatasetConfig,
    dataset: DatasetSpec,
    *,
    checkpoint: str | None = None,
    data_dir: str | None = None,
    device: str | None = None,
    limit: int | None = None,
) -> ExperimentConfig:
    """Create an independent single-dataset config from batch-level settings."""

    config = copy.deepcopy(base_config)
    config.data.mode = "memmap"
    config.data.data_dir = data_dir or dataset.data_dir or multi_config.data_dir
    config.data.diff_file = dataset.diff_file
    config.data.real_file = dataset.real_file
    config.data.num_samples = dataset.num_samples
    config.data.shape = list(dataset.shape)
    config.data.dtype_diff = dataset.dtype_diff
    config.data.dtype_real = dataset.dtype_real
    config.data.scale_i = dataset.scale_i
    config.model.checkpoint = checkpoint or multi_config.checkpoint
    if device or multi_config.device:
        config.runtime.device = device or multi_config.device or config.runtime.device
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive.")
        config.data.num_samples = min(config.data.num_samples, limit)
    return config


def load_base_experiment_config(
    multi_path: str | Path,
    multi_config: MultiDatasetConfig,
) -> ExperimentConfig:
    """Load the single-dataset template referenced by a multi-dataset config."""

    path = resolve_base_config_path(Path(multi_path), multi_config.base_config)
    return load_experiment_config(path)


def multi_dataset_metric_rows(
    dataset_summaries: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten reconstruction and model-output comparisons into a CSV table."""

    rows: list[dict[str, Any]] = []
    for dataset_name, summary in dataset_summaries.items():
        for metric_name, item in summary["reconstruction"].items():
            rows.append(
                {
                    "dataset": dataset_name,
                    "category": "reconstruction",
                    "metric": metric_name,
                    "direction": item["direction"],
                    "baseline_mean": item["baseline"]["mean"],
                    "swapped_mean": item["swapped"]["mean"],
                    "difference_mean": item["delta"]["mean"],
                    "difference_ci_low": item["delta"].get("ci_low"),
                    "difference_ci_high": item["delta"].get("ci_high"),
                    "relative_change_percent": item["relative_change_percent"],
                    "degradation_percent": item["degradation_percent"],
                }
            )
        final_output = summary["final_output_comparison"]
        for metric_name, item in final_output["metrics"].items():
            rows.append(
                {
                    "dataset": dataset_name,
                    "category": "model_output_difference",
                    "metric": f"farfield.{metric_name}",
                    "direction": "lower" if metric_name != "pearson_corr" else "higher",
                    "baseline_mean": None,
                    "swapped_mean": None,
                    "difference_mean": item["mean"],
                    "difference_ci_low": item.get("ci_low"),
                    "difference_ci_high": item.get("ci_high"),
                    "relative_change_percent": None,
                    "degradation_percent": None,
                }
            )
    return rows


def build_multi_dataset_summary(
    config: MultiDatasetConfig,
    dataset_summaries: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build a compact cross-dataset index while retaining each detailed summary."""

    return {
        "experiment": "autophasenn_bn_maxpool_swap_multi_dataset",
        "checkpoint": config.checkpoint,
        "dataset_count": len(dataset_summaries),
        "total_samples": sum(
            int(summary["num_samples"]) for summary in dataset_summaries.values()
        ),
        "all_datasets_pass": all(
            bool(summary["acceptance"]["overall_pass"])
            for summary in dataset_summaries.values()
        ),
        "datasets": dict(dataset_summaries),
    }


def write_multi_dataset_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a cross-dataset metrics table with a stable column order."""

    columns = [
        "dataset",
        "category",
        "metric",
        "direction",
        "baseline_mean",
        "swapped_mean",
        "difference_mean",
        "difference_ci_low",
        "difference_ci_high",
        "relative_change_percent",
        "degradation_percent",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


RECONSTRUCTION_METRICS = (
    "paper_modulus_mae",
    "chi2_modulus",
    "relative_l1_modulus",
    "pearson_corr",
    "voxel_rmse",
    "real_amp_psnr",
    "real_amp_ssim3d",
    "real_support_iou",
    "real_support_dice",
    "real_phase_mae_true_support",
)

OUTPUT_DIFFERENCE_METRICS = (
    "mae",
    "rmse",
    "max_abs",
    "relative_l1",
    "relative_l2",
    "pearson_corr",
    "histogram_js_divergence",
)


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "通过" if value else "未通过"
    if isinstance(value, (int, float)):
        return f"{value:.6g}"
    return str(value)


def write_multi_dataset_report(path: Path, summary: Mapping[str, Any]) -> None:
    """Write a concise Chinese comparison report across all configured datasets."""

    lines = [
        "# AutoPhaseNN BN/MaxPool 交换：多数据集量化对比",
        "",
        f"共评估 {summary['dataset_count']} 个数据集、{summary['total_samples']} 个样本。",
        f"全部数据集是否满足配置中的判据：**{_fmt(summary['all_datasets_pass'])}**。",
        "",
        "两个独立模型均加载同一检查点并完整执行 forward；唯一差别是四个编码块中 "
        "`BN -> MaxPool3d` 与 `MaxPool3d -> BN` 的顺序。",
        "",
        "## 两个模型相对真值的重建效果",
        "",
        "差值定义为“交换模型 − 基线模型”；退化率为正表示交换后变差。指标实现直接复用 "
        "`autophasenn_training_pipeline.losses`，并补充 3D PSNR/SSIM。",
        "",
        "| 数据集 | 指标 | 趋势 | 基线 | 交换后 | 差值 | 退化率 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset_name, dataset_summary in summary["datasets"].items():
        reconstruction = dataset_summary["reconstruction"]
        for metric_name in RECONSTRUCTION_METRICS:
            item = reconstruction.get(metric_name)
            if item is None:
                continue
            direction = {"lower": "越低越好", "higher": "越高越好"}.get(
                item["direction"], "诊断"
            )
            lines.append(
                f"| {dataset_name} | `{metric_name}` | {direction} | "
                f"{_fmt(item['baseline']['mean'])} | {_fmt(item['swapped']['mean'])} | "
                f"{_fmt(item['delta']['mean'])} | {_fmt(item['degradation_percent'])}% |"
            )

    lines.extend(
        [
            "",
            "## 两个模型最终衍射输出的直接差异",
            "",
            "下表不依赖真值，直接比较两个完整模型最终输出的 `farfield_modulus`。",
            "",
            "| 数据集 | MAE | RMSE | 最大绝对差 | 相对 L1 | 相对 L2 | Pearson | JS 散度 | 完全一致 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset_name, dataset_summary in summary["datasets"].items():
        comparison = dataset_summary["final_output_comparison"]
        metrics = comparison["metrics"]
        values = {
            name: metrics[name]["mean"] for name in OUTPUT_DIFFERENCE_METRICS
        }
        lines.append(
            f"| {dataset_name} | {_fmt(values['mae'])} | {_fmt(values['rmse'])} | "
            f"{_fmt(comparison['maximum_absolute_difference_over_all_samples'])} | "
            f"{_fmt(values['relative_l1'])} | {_fmt(values['relative_l2'])} | "
            f"{_fmt(values['pearson_corr'])} | "
            f"{_fmt(values['histogram_js_divergence'])} | "
            f"{_fmt(comparison['exactly_identical'])} |"
        )
    lines.extend(
        [
            "",
            "## 数据集级判定",
            "",
            "| 数据集 | 样本数 | 有实空间真值 | 重建指标通过 | 输出一致性通过 | BN 正缩放前提通过 | 总判定 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset_name, dataset_summary in summary["datasets"].items():
        acceptance = dataset_summary["acceptance"]
        lines.append(
            f"| {dataset_name} | {dataset_summary['num_samples']} | "
            f"{_fmt(dataset_summary['has_realspace_truth'])} | "
            f"{_fmt(acceptance['primary_metrics_pass'])} | "
            f"{_fmt(acceptance['output_consistency_pass'])} | "
            f"{_fmt(acceptance['bn_scale_precondition_pass'])} | "
            f"{_fmt(acceptance['overall_pass'])} |"
        )
    lines.extend(
        [
            "",
            "完整的逐样本指标、置信区间、BN 审计和环境信息保存在各数据集子目录中。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
