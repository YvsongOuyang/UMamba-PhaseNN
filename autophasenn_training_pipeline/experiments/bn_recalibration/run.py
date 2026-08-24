"""Compare checkpoint BatchNorm statistics with train-data recalibration."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from autophasenn_training_pipeline import evaluate as evaluation
from autophasenn_training_pipeline.dataset import AutoPhaseDataset
from autophasenn_training_pipeline.model_factory import (
    MODEL_VARIANTS,
    create_model,
    default_support_threshold,
)


LOGGER = evaluation.LOGGER
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    "/data_ssd/oyys/autophasenn/autophasenn_pipeline_output/"
    "mamba_skip_scratch_bs4_lr1e-3_20260823_155916/checkpoint_best.pt"
)
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results" / "bn_recalibration_mamba_skip"
BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

COMPARISON_METRICS = {
    "paper_modulus_mae": "lower",
    "chi2_modulus": "lower",
    "pearson_corr": "higher",
    "real_amp_l1": "lower",
    "real_amp_ssim": "higher",
    "real_support_iou": "higher",
    "real_support_dice": "higher",
    "real_support_volume_ratio": "target_one",
    "real_phase_mae_true_support": "lower",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-variant", choices=MODEL_VARIANTS, default="mamba_skip")
    parser.add_argument("--data-dir", default="/data_ssd/oyys/autophasenn/")
    parser.add_argument("--data-train-diff", default="train_diff.npy")
    parser.add_argument("--data-val-diff", default="val_diff.npy")
    parser.add_argument("--data-val-real", default="val_real.npy")
    parser.add_argument("--num-calibration-samples", type=int, default=25000)
    parser.add_argument("--num-val-samples", type=int, default=5000)
    parser.add_argument("--calibration-passes", type=int, default=1)
    parser.add_argument("--shape", type=int, default=64)
    parser.add_argument("--dtype-diff", default="float32")
    parser.add_argument("--dtype-real", default="complex64")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--postprocess-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Support threshold for both evaluations. By default the checkpoint's "
            "training threshold is used so BN is the only changed variable."
        ),
    )
    parser.add_argument("--ssim-window-size", type=int, default=7)
    parser.add_argument("--free-mask", default="")
    parser.add_argument("--free-fraction", type=float, default=0.05)
    parser.add_argument("--free-seed", type=int, default=42)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--non-strict-checkpoint", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args()
    validate_args(args, parser)
    return args


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    positive = {
        "--num-calibration-samples": args.num_calibration_samples,
        "--num-val-samples": args.num_val_samples,
        "--calibration-passes": args.calibration_passes,
        "--shape": args.shape,
        "--batch-size": args.batch_size,
        "--ssim-window-size": args.ssim_window_size,
    }
    for name, value in positive.items():
        if value < 1:
            parser.error(f"{name} must be positive.")
    if args.num_workers < 0 or args.postprocess_workers < 0:
        parser.error("Worker counts cannot be negative.")
    if args.ssim_window_size % 2 == 0 or args.ssim_window_size > args.shape:
        parser.error("--ssim-window-size must be odd and no larger than --shape.")
    if args.threshold is not None and (
        not math.isfinite(args.threshold) or args.threshold < 0
    ):
        parser.error("--threshold must be finite and nonnegative.")
    if not 0.0 <= args.free_fraction < 1.0:
        parser.error("--free-fraction must be in [0, 1).")


def configure_logging(output_dir: Path, level: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for handler in LOGGER.handlers:
        handler.close()
    LOGGER.handlers.clear()
    LOGGER.setLevel(getattr(logging, level))
    LOGGER.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_dir / "bn_recalibration.log", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def resolve_path(data_dir: Path, filename: str) -> Path:
    path = (data_dir / filename).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")
    return path


def load_checkpoint_model(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, dict[str, object], float, str]:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        checkpoint_threshold = checkpoint.get("threshold")
        metadata = {
            "epoch": checkpoint.get("epoch"),
            "threshold": checkpoint_threshold,
            "best_val": checkpoint.get("best_val"),
        }
    else:
        state_dict = checkpoint
        checkpoint_threshold = None
        metadata = {"epoch": None, "threshold": None, "best_val": None}

    if args.threshold is not None:
        threshold = float(args.threshold)
        threshold_source = "explicit_argument"
    elif checkpoint_threshold is not None:
        threshold = float(checkpoint_threshold)
        threshold_source = "checkpoint_metadata"
    else:
        threshold = default_support_threshold(args.model_variant)
        threshold_source = "model_variant_default"

    model = create_model(args.model_variant, threshold=threshold)
    incompatibility = model.load_state_dict(
        state_dict,
        strict=not args.non_strict_checkpoint,
    )
    if args.non_strict_checkpoint:
        LOGGER.warning(
            "Non-strict checkpoint load | missing=%s | unexpected=%s",
            incompatibility.missing_keys,
            incompatibility.unexpected_keys,
        )
    del checkpoint, state_dict
    return model.to(device), metadata, threshold, threshold_source


def batchnorm_modules(model: nn.Module) -> list[tuple[str, nn.modules.batchnorm._BatchNorm]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, BN_TYPES) and module.track_running_stats
    ]


def snapshot_batchnorm(model: nn.Module) -> dict[str, dict[str, object]]:
    snapshot: dict[str, dict[str, object]] = {}
    for name, module in batchnorm_modules(model):
        snapshot[name] = {
            "running_mean": module.running_mean.detach().cpu().clone(),
            "running_var": module.running_var.detach().cpu().clone(),
            "num_batches_tracked": int(module.num_batches_tracked.detach().cpu()),
            "momentum": module.momentum,
        }
    if not snapshot:
        raise RuntimeError("The selected model has no tracked BatchNorm layers.")
    return snapshot


@torch.no_grad()
def recalibrate_batchnorm(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    passes: int,
) -> dict[str, int]:
    """Reset BN buffers and estimate them cumulatively from training inputs."""

    modules = batchnorm_modules(model)
    original_momenta = {name: module.momentum for name, module in modules}
    model.eval()
    for _name, module in modules:
        module.reset_running_stats()
        module.momentum = None
        module.train()

    processed_samples = 0
    processed_batches = 0
    try:
        for pass_index in range(1, passes + 1):
            progress = tqdm(
                loader,
                desc=f"BN calibration {pass_index}/{passes}",
                unit="batch",
            )
            for batch in progress:
                inputs = batch["diff"].to(device, non_blocking=True).float()
                model(inputs)
                processed_samples += inputs.shape[0]
                processed_batches += 1
        evaluation.synchronize(device)
    finally:
        model.eval()
        for name, module in modules:
            module.momentum = original_momenta[name]

    return {
        "unique_samples": len(loader.dataset),
        "forwarded_samples": processed_samples,
        "forwarded_batches": processed_batches,
        "passes": passes,
    }


def batchnorm_drift_rows(
    before: dict[str, dict[str, object]],
    after: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in before:
        mean_before = before[name]["running_mean"].float()
        mean_after = after[name]["running_mean"].float()
        var_before = before[name]["running_var"].float()
        var_after = after[name]["running_var"].float()
        mean_delta = torch.abs(mean_after - mean_before)
        var_delta = torch.abs(var_after - var_before)
        rows.append(
            {
                "layer": name,
                "channels": mean_before.numel(),
                "mean_abs_running_mean_delta": float(mean_delta.mean()),
                "max_abs_running_mean_delta": float(mean_delta.max()),
                "relative_l2_running_mean_delta": float(
                    torch.linalg.vector_norm(mean_after - mean_before)
                    / torch.linalg.vector_norm(mean_before).clamp_min(1e-12)
                ),
                "mean_abs_running_var_delta": float(var_delta.mean()),
                "max_abs_running_var_delta": float(var_delta.max()),
                "relative_l2_running_var_delta": float(
                    torch.linalg.vector_norm(var_after - var_before)
                    / torch.linalg.vector_norm(var_before).clamp_min(1e-12)
                ),
                "batches_before": before[name]["num_batches_tracked"],
                "batches_after": after[name]["num_batches_tracked"],
            }
        )
    return rows


def compare_evaluations(
    normal_rows: list[dict[str, object]],
    recalibrated_rows: list[dict[str, object]],
) -> dict[str, dict[str, float | str]]:
    recalibrated_by_name = {row["name"]: row for row in recalibrated_rows}
    if set(recalibrated_by_name) != {row["name"] for row in normal_rows}:
        raise RuntimeError("Normal and recalibrated evaluations used different samples.")

    comparisons: dict[str, dict[str, float | str]] = {}
    for metric, direction in COMPARISON_METRICS.items():
        normal = np.asarray([float(row[metric]) for row in normal_rows])
        recalibrated = np.asarray(
            [float(recalibrated_by_name[row["name"]][metric]) for row in normal_rows]
        )
        delta = recalibrated - normal
        standard_error = float(np.std(delta, ddof=1) / math.sqrt(delta.size))
        if direction == "lower":
            improved = recalibrated < normal
            relative_improvement = (normal.mean() - recalibrated.mean()) / max(
                abs(normal.mean()), 1e-12
            )
        elif direction == "higher":
            improved = recalibrated > normal
            relative_improvement = (recalibrated.mean() - normal.mean()) / max(
                abs(normal.mean()), 1e-12
            )
        else:
            normal_error = np.abs(normal - 1.0)
            recalibrated_error = np.abs(recalibrated - 1.0)
            improved = recalibrated_error < normal_error
            relative_improvement = (
                normal_error.mean() - recalibrated_error.mean()
            ) / max(normal_error.mean(), 1e-12)
        comparisons[metric] = {
            "direction": direction,
            "normal_mean": float(normal.mean()),
            "recalibrated_mean": float(recalibrated.mean()),
            "raw_delta_after_minus_before": float(delta.mean()),
            "raw_delta_ci95_low": float(delta.mean() - 1.96 * standard_error),
            "raw_delta_ci95_high": float(delta.mean() + 1.96 * standard_error),
            "relative_improvement_percent": float(100.0 * relative_improvement),
            "samples_improved_percent": float(100.0 * improved.mean()),
        }
    return comparisons


def write_bn_drift_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_paired_csv(
    path: Path,
    normal_rows: list[dict[str, object]],
    recalibrated_rows: list[dict[str, object]],
) -> None:
    recalibrated_by_name = {row["name"]: row for row in recalibrated_rows}
    fieldnames = ["name"]
    for metric in COMPARISON_METRICS:
        fieldnames.extend((f"normal_{metric}", f"recalibrated_{metric}", f"delta_{metric}"))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for normal in normal_rows:
            recalibrated = recalibrated_by_name[normal["name"]]
            row: dict[str, object] = {"name": normal["name"]}
            for metric in COMPARISON_METRICS:
                before = float(normal[metric])
                after = float(recalibrated[metric])
                row[f"normal_{metric}"] = before
                row[f"recalibrated_{metric}"] = after
                row[f"delta_{metric}"] = after - before
            writer.writerow(row)


def render_summary(report: dict[str, object]) -> str:
    run = report["run"]
    calibration = report["calibration"]
    lines = [
        "# BatchNorm Recalibration Experiment",
        "",
        "## Setup",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Checkpoint | `{run['checkpoint']}` |",
        f"| Checkpoint epoch | {run['checkpoint_epoch']} |",
        f"| Model | `{run['model_variant']}` |",
        f"| Threshold | {run['threshold']:.6g} ({run['threshold_source']}) |",
        f"| Validation samples | {run['validation_samples']} |",
        f"| Calibration samples | {calibration['unique_samples']} |",
        f"| Calibration passes | {calibration['passes']} |",
        f"| Calibration batches | {calibration['forwarded_batches']} |",
        f"| BatchNorm layers | {calibration['batchnorm_layers']} |",
        "",
        "Only BatchNorm running mean, running variance, and batch counters are changed. "
        "All learned parameters and BatchNorm affine parameters remain untouched.",
        "",
        "## Paired Validation Results",
        "",
        "| Metric | Normal eval | BN recalibrated | Raw delta | Relative improvement | Samples improved |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric, values in report["comparison"].items():
        lines.append(
            f"| `{metric}` | {values['normal_mean']:.6g} | "
            f"{values['recalibrated_mean']:.6g} | "
            f"{values['raw_delta_after_minus_before']:+.6g} | "
            f"{values['relative_improvement_percent']:+.3f}% | "
            f"{values['samples_improved_percent']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "Positive relative improvement always means better according to the metric's "
            "direction. Raw delta is recalibrated minus normal.",
            "",
            "## Largest BN Buffer Changes",
            "",
            "| Layer | Mean relative L2 change | Variance relative L2 change |",
            "|---|---:|---:|",
        ]
    )
    drift_rows = sorted(
        report["batchnorm_drift"],
        key=lambda row: max(
            row["relative_l2_running_mean_delta"],
            row["relative_l2_running_var_delta"],
        ),
        reverse=True,
    )
    for row in drift_rows[:10]:
        lines.append(
            f"| `{row['layer']}` | {row['relative_l2_running_mean_delta']:.6g} | "
            f"{row['relative_l2_running_var_delta']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A material and sample-consistent improvement after recalibration supports a "
            "checkpoint BN-statistics mismatch. Little change, mixed change, or degradation "
            "means stale running statistics are unlikely to be the primary cause of the "
            "validation behavior.",
            "",
        ]
    )
    return "\n".join(lines)


def make_loader(
    dataset: AutoPhaseDataset,
    args: argparse.Namespace,
    device: torch.device,
    *,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(args.seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        generator=generator,
    )


def run_validation(
    args: argparse.Namespace,
    model: nn.Module,
    loader: DataLoader,
    dataset: AutoPhaseDataset,
    device: torch.device,
    free_mask: torch.Tensor | None,
    postprocess_workers: int,
) -> tuple[list[dict[str, object]], dict[str, dict[str, float]], float]:
    evaluation_args = argparse.Namespace(
        threshold=args.resolved_threshold,
        threshold_sweep=(),
        ssim_window_size=args.ssim_window_size,
    )
    model.eval()
    evaluation.warm_up_model(model, loader, device, args.warmup_batches)
    started = time.perf_counter()
    rows, _total, _sweep = evaluation.evaluate(
        evaluation_args,
        model,
        loader,
        dataset,
        device,
        free_mask,
        postprocess_workers,
    )
    wall_seconds = time.perf_counter() - started
    return rows, evaluation.metric_statistics(rows), wall_seconds


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    configure_logging(output_dir, args.log_level)
    torch.manual_seed(args.seed)
    device = evaluation.choose_device(args.device)
    data_dir = Path(args.data_dir).expanduser().resolve()
    shape = (args.shape, args.shape, args.shape)

    model, checkpoint_metadata, threshold, threshold_source = load_checkpoint_model(
        args,
        device,
    )
    args.resolved_threshold = threshold
    train_dataset = AutoPhaseDataset(
        resolve_path(data_dir, args.data_train_diff),
        num_samples=args.num_calibration_samples,
        shape_diff=shape,
        dtype_diff=args.dtype_diff,
        shuffle=False,
    )
    val_dataset = AutoPhaseDataset(
        resolve_path(data_dir, args.data_val_diff),
        resolve_path(data_dir, args.data_val_real),
        num_samples=args.num_val_samples,
        shape_diff=shape,
        shape_real=shape,
        dtype_diff=args.dtype_diff,
        dtype_real=args.dtype_real,
        shuffle=False,
    )
    calibration_loader = make_loader(
        train_dataset,
        args,
        device,
        shuffle=True,
    )
    val_loader = make_loader(val_dataset, args, device, shuffle=False)
    free_mask, free_mask_metadata = evaluation.resolve_free_mask(args, shape)
    postprocess_workers = evaluation.resolve_postprocess_workers(
        args.postprocess_workers,
        args.batch_size,
    )

    LOGGER.info(
        "BN experiment | model=%s | checkpoint_epoch=%s | threshold=%.6g (%s)",
        args.model_variant,
        checkpoint_metadata["epoch"],
        threshold,
        threshold_source,
    )
    LOGGER.info(
        "Data | calibration=%d train samples | validation=%d samples | batch=%d",
        len(train_dataset),
        len(val_dataset),
        args.batch_size,
    )

    before_bn = snapshot_batchnorm(model)
    LOGGER.info("Evaluating checkpoint running statistics.")
    normal_rows, normal_statistics, normal_wall = run_validation(
        args,
        model,
        val_loader,
        val_dataset,
        device,
        free_mask,
        postprocess_workers,
    )

    LOGGER.info("Recalibrating %d BatchNorm layers from training inputs.", len(before_bn))
    calibration = recalibrate_batchnorm(
        model,
        calibration_loader,
        device,
        args.calibration_passes,
    )
    after_bn = snapshot_batchnorm(model)
    drift_rows = batchnorm_drift_rows(before_bn, after_bn)

    LOGGER.info("Evaluating recalibrated running statistics.")
    recalibrated_rows, recalibrated_statistics, recalibrated_wall = run_validation(
        args,
        model,
        val_loader,
        val_dataset,
        device,
        free_mask,
        postprocess_workers,
    )
    comparison = compare_evaluations(normal_rows, recalibrated_rows)

    calibration["batchnorm_layers"] = len(before_bn)
    report: dict[str, object] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run": {
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "checkpoint_epoch": checkpoint_metadata["epoch"],
            "checkpoint_best_val": checkpoint_metadata["best_val"],
            "checkpoint_threshold": checkpoint_metadata["threshold"],
            "model_variant": args.model_variant,
            "threshold": threshold,
            "threshold_source": threshold_source,
            "device": str(device),
            "validation_samples": len(val_dataset),
            "batch_size": args.batch_size,
        },
        "configuration": {
            key: value
            for key, value in vars(args).items()
            if key != "resolved_threshold"
        },
        "calibration": calibration,
        "free_mask": free_mask_metadata,
        "timing": {
            "normal_validation_wall_seconds": normal_wall,
            "recalibrated_validation_wall_seconds": recalibrated_wall,
        },
        "normal_metric_statistics": normal_statistics,
        "recalibrated_metric_statistics": recalibrated_statistics,
        "comparison": comparison,
        "batchnorm_drift": drift_rows,
    }

    evaluation.write_sample_csv(output_dir / "evaluation_normal.csv", normal_rows)
    evaluation.write_sample_csv(
        output_dir / "evaluation_bn_recalibrated.csv",
        recalibrated_rows,
    )
    write_paired_csv(
        output_dir / "evaluation_bn_paired.csv",
        normal_rows,
        recalibrated_rows,
    )
    write_bn_drift_csv(output_dir / "bn_layer_drift.csv", drift_rows)
    evaluation.write_json(output_dir / "bn_recalibration_report.json", report)
    (output_dir / "bn_recalibration_summary.md").write_text(
        render_summary(report),
        encoding="utf-8",
    )
    torch.save(
        {
            "checkpoint": report["run"]["checkpoint"],
            "model_variant": args.model_variant,
            "threshold": threshold,
            "batchnorm_buffers": after_bn,
        },
        output_dir / "bn_recalibrated_buffers.pt",
    )

    primary = comparison["paper_modulus_mae"]
    LOGGER.info(
        "Complete | modulus MAE %.6g -> %.6g (%+.3f%%) | samples improved %.2f%%",
        primary["normal_mean"],
        primary["recalibrated_mean"],
        primary["relative_improvement_percent"],
        primary["samples_improved_percent"],
    )
    LOGGER.info("Artifacts: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
