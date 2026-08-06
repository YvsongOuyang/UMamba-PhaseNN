"""Evaluate an AutoPhaseNN checkpoint and write a readable artifact bundle."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .dataset import AutoPhaseDataset
    from .losses import (
        FIXED_METRIC_DESCRIPTIONS,
        METRIC_DESCRIPTIONS,
        fixed_metric_groups,
        format_fixed_metric_groups,
        free_metric_dict,
        group_metrics,
        metric_dict,
        realspace_metric_dict,
        scale_align_sum,
    )
    from .model_tf_compatible import TFCompatibleAutoPhaseNN, load_weights
except ImportError:
    from dataset import AutoPhaseDataset
    from losses import (
        FIXED_METRIC_DESCRIPTIONS,
        METRIC_DESCRIPTIONS,
        fixed_metric_groups,
        format_fixed_metric_groups,
        free_metric_dict,
        group_metrics,
        metric_dict,
        realspace_metric_dict,
        scale_align_sum,
    )
    from model_tf_compatible import TFCompatibleAutoPhaseNN, load_weights


LOGGER = logging.getLogger("autophasenn.evaluate")
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output" / "evaluate"
DEFAULT_CHECKPOINT = (
    "/data_ssd/oyys/autophasenn/autophasenn_pipeline_output/autophasenn_retrain_l1/checkpoint_best.pt"
)

PAPER_METRICS = {
    "paper_modulus_mae": {
        "paper_reference": "Eq. (1): MAE between measured and estimated diffraction modulus.",
        "direction": "lower",
    },
    "chi2_modulus": {
        "paper_reference": "Eq. (2): reciprocal-space chi-square on diffraction modulus.",
        "direction": "lower",
    },
    "real_amp_ssim": {
        "paper_reference": "Fig. 2: local-window 3D SSIM of real-space amplitude.",
        "direction": "higher",
    },
    "r_factor_free": {
        "paper_reference": "Free R-factor family evaluated on held-out reciprocal voxels.",
        "direction": "lower",
    },
    "llk_free": {
        "paper_reference": "Supplementary Note 3: free Poisson log-likelihood diagnostic.",
        "direction": "lower",
    },
    "chi2_free": {
        "paper_reference": "Supplementary Note 3: free chi-square diagnostic.",
        "direction": "lower",
    },
}


def optional_data_path(data_dir: Path, filename: str | None) -> Path | None:
    """Resolve an optional data filename relative to the data directory."""

    if filename is None or filename.lower() in {"", "none", "null"}:
        return None
    return data_dir / filename


def choose_device(name: str) -> torch.device:
    """Resolve the requested device with an explicit CUDA fallback."""

    if name == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA was requested but is unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(name)


def configure_logging(output_dir: Path, level: str) -> None:
    """Configure matching console and file logs for an evaluation run."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for handler in LOGGER.handlers:
        handler.close()
    LOGGER.handlers.clear()
    LOGGER.setLevel(getattr(logging, level.upper()))
    LOGGER.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(
        output_dir / "evaluation.log",
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(console_handler)
    LOGGER.addHandler(file_handler)


def parse_args() -> argparse.Namespace:
    """Parse all data, model, metric, and output parameters."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate AutoPhaseNN with paper metrics and write JSON, CSV, "
            "Markdown, and log artifacts."
        )
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", default="/data_ssd/oyys/autophasenn/")
    parser.add_argument("--data-diff", default="val_diff.npy")
    parser.add_argument("--data-real", default="val_real.npy")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate at most this many samples; zero uses --num-samples.",
    )
    parser.add_argument("--shape", type=int, default=64)
    parser.add_argument("--dtype-diff", default="float32")
    parser.add_argument("--dtype-real", default="complex64")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--scale-i", type=float, default=0.0)
    parser.add_argument("--scale-align-loss", action="store_true")
    parser.add_argument(
        "--ssim-window-size",
        type=int,
        default=7,
        help="Odd cubic window size for paper-style 3D amplitude SSIM.",
    )
    parser.add_argument(
        "--free-mask",
        default="",
        help=(
            "Optional NumPy .npy/.npz boolean holdout mask with spatial shape. "
            "It takes precedence over --free-fraction."
        ),
    )
    parser.add_argument(
        "--free-fraction",
        type=float,
        default=0.05,
        help="Deterministic diagnostic holdout fraction when --free-mask is absent; zero disables free metrics.",
    )
    parser.add_argument("--free-seed", type=int, default=42)
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=1,
        help="Untimed forward batches used to warm up the selected device.",
    )
    parser.add_argument(
        "--reference-runtime-ms",
        type=float,
        default=0.0,
        help="Optional per-sample baseline used to calculate a machine-specific speedup.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--non-strict-checkpoint",
        action="store_true",
        help="Allow checkpoint keys that do not exactly match the model.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Reject invalid metric and loader settings before allocating data."""

    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive.")
    if args.limit < 0:
        raise ValueError("--limit cannot be negative.")
    if args.shape < 1:
        raise ValueError("--shape must be positive.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if not 0.0 <= args.free_fraction < 1.0:
        raise ValueError("--free-fraction must be in [0, 1).")
    if args.ssim_window_size < 1 or args.ssim_window_size % 2 == 0:
        raise ValueError("--ssim-window-size must be a positive odd integer.")
    if args.ssim_window_size > args.shape:
        raise ValueError("--ssim-window-size cannot exceed --shape.")
    if args.warmup_batches < 0:
        raise ValueError("--warmup-batches cannot be negative.")
    if args.reference_runtime_ms < 0:
        raise ValueError("--reference-runtime-ms cannot be negative.")


def load_external_free_mask(
    mask_path: Path, shape: tuple[int, int, int]
) -> torch.Tensor:
    """Load and validate a standard NumPy reciprocal-space holdout mask."""

    loaded = np.load(mask_path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        keys = loaded.files
        if len(keys) != 1:
            loaded.close()
            raise ValueError("Free-mask .npz files must contain exactly one array.")
        array = np.asarray(loaded[keys[0]])
        loaded.close()
    else:
        array = np.asarray(loaded)
    array = np.squeeze(array)
    if tuple(array.shape) != shape:
        raise ValueError(
            f"Free mask has spatial shape {tuple(array.shape)}; expected {shape}."
        )
    mask = torch.from_numpy(np.asarray(array > 0, dtype=np.bool_)).reshape(
        (1, 1) + shape
    )
    if not bool(mask.any()):
        raise ValueError("Free mask does not select any reciprocal-space voxels.")
    return mask


def generate_free_mask(
    shape: tuple[int, int, int],
    fraction: float,
    seed: int,
) -> torch.Tensor | None:
    """Generate a deterministic diagnostic holdout mask shared by all samples."""

    if fraction <= 0.0:
        return None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    mask = torch.rand((1, 1) + shape, generator=generator) < fraction
    flat_mask = mask.reshape(-1)
    if not bool(flat_mask.any()):
        flat_mask[0] = True
    if bool(flat_mask.all()):
        flat_mask[-1] = False
    return mask


def resolve_free_mask(
    args: argparse.Namespace,
    shape: tuple[int, int, int],
) -> tuple[torch.Tensor | None, dict[str, object]]:
    """Resolve the free-mask source and provenance recorded in the report."""

    if args.free_mask:
        mask_path = Path(args.free_mask).expanduser().resolve()
        mask = load_external_free_mask(mask_path, shape)
        source = "external_holdout_mask"
        note = (
            "Metrics use the supplied mask. They are true free metrics only if these voxels "
            "were excluded from the reconstruction/training objective."
        )
        provenance: dict[str, object] = {"path": str(mask_path)}
    else:
        mask = generate_free_mask(shape, args.free_fraction, args.free_seed)
        source = "generated_diagnostic_mask" if mask is not None else "disabled"
        note = (
            "The generated mask is reproducible but was not automatically excluded during "
            "model training. Treat Rfree, LLKfree, and chi2free as diagnostics, not as a "
            "numerically comparable reproduction of the paper."
            if mask is not None
            else "Free metrics are disabled because --free-fraction is zero."
        )
        provenance = {
            "requested_fraction": args.free_fraction,
            "seed": args.free_seed,
        }
    actual_fraction = float(mask.float().mean()) if mask is not None else 0.0
    return mask, {
        "enabled": mask is not None,
        "source": source,
        "actual_fraction": actual_fraction,
        "selected_voxels": int(mask.sum()) if mask is not None else 0,
        "note": note,
        **provenance,
    }


def synchronize(device: torch.device) -> None:
    """Synchronize CUDA so model timing excludes asynchronous launch delay."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def unpack_outputs(outputs: object) -> tuple[torch.Tensor, ...]:
    """Validate the AutoPhaseNN forward-output contract."""

    if not isinstance(outputs, (tuple, list)) or len(outputs) < 5:
        raise RuntimeError(
            "Model must return (pred_diff, pred_obj, pred_amp, pred_phi, support)."
        )
    return tuple(outputs[:5])


@torch.inference_mode()
def warm_up_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    warmup_batches: int,
) -> None:
    """Run untimed forward passes before collecting performance metrics."""

    if warmup_batches <= 0:
        return
    completed = 0
    for batch in loader:
        diff = batch["diff"].to(device, non_blocking=True).float()
        model(diff)
        completed += 1
        if completed >= warmup_batches:
            break
    synchronize(device)
    LOGGER.info("Completed %d warmup batch(es).", completed)


def add_metrics(total: dict[str, float], metrics: dict[str, float]) -> None:
    """Accumulate scalar metric values."""

    for key, value in metrics.items():
        total[key] = total.get(key, 0.0) + float(value)


def metric_statistics(
    per_sample: list[dict[str, object]]
) -> dict[str, dict[str, float]]:
    """Calculate readable distribution statistics for every scalar metric."""

    metric_keys = sorted(
        {
            key
            for row in per_sample
            for key, value in row.items()
            if key != "name" and isinstance(value, (int, float))
        }
    )
    statistics: dict[str, dict[str, float]] = {}
    for key in metric_keys:
        values = np.asarray(
            [float(row[key]) for row in per_sample if key in row],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        statistics[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }
    return statistics


def write_json(path: Path, report: dict[str, object]) -> None:
    """Write the full machine-readable report."""

    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def ordered_metric_keys(per_sample: list[dict[str, object]]) -> list[str]:
    """Place paper metrics first and diagnostics afterward in tabular output."""

    available = {key for row in per_sample for key in row if key != "name"}
    paper_keys = [key for key in PAPER_METRICS if key in available]
    return paper_keys + sorted(available.difference(paper_keys))


def write_sample_csv(path: Path, per_sample: list[dict[str, object]]) -> None:
    """Write one evaluation row per sample for spreadsheet analysis."""

    fieldnames = ["name", *ordered_metric_keys(per_sample)]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(per_sample)


def format_number(value: object) -> str:
    """Format report scalars compactly without losing small values."""

    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        return f"{value:.6g}"
    return str(value)


def render_markdown(report: dict[str, object]) -> str:
    """Render a human-readable summary emphasizing paper-reported metrics."""

    run = report["run"]
    statistics = report["metric_statistics"]
    free_mask = report["free_mask"]
    timing = report["timing"]
    paper_coverage = report["paper_metric_coverage"]
    grouped_mean = report["mean_metric_groups"]
    lines = [
        "# AutoPhaseNN Evaluation Summary",
        "",
        "## Run",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Checkpoint | `{run['checkpoint']}` |",
        f"| Checkpoint epoch | {format_number(run['checkpoint_epoch'])} |",
        f"| Device | `{run['device']}` |",
        f"| PyTorch | `{run['torch_version']}` |",
        f"| CUDA runtime | {format_number(run['cuda_version'])} |",
        f"| GPU | {format_number(run['gpu_name'])} |",
        f"| Samples | {run['num_samples']} |",
        f"| Batch size | {run['batch_size']} |",
        f"| Support threshold | {format_number(run['support_threshold'])} |",
        f"| SSIM window | {run['ssim_window_size']} x {run['ssim_window_size']} x {run['ssim_window_size']} |",
        f"| Real-space ground truth | {run['realspace_metrics']} |",
        f"| Scale-aligned diffraction | {run['scale_align_loss']} |",
        f"| Total evaluation wall time | {format_number(timing['evaluation_wall_seconds'])} s |",
        f"| Mean model inference | {format_number(timing['mean_inference_ms_per_sample'])} ms/sample |",
        f"| Model throughput | {format_number(timing['throughput_samples_per_second'])} samples/s |",
        "",
        "## Paper Metric Coverage",
        "",
        "| Metric | Mean | Std | P50 | P95 | Better | Paper usage |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for key, metadata in paper_coverage.items():
        stats = statistics.get(key)
        if stats is None:
            values = ["n/a"] * 4
        else:
            values = [
                format_number(stats["mean"]),
                format_number(stats["std"]),
                format_number(stats["p50"]),
                format_number(stats["p95"]),
            ]
        lines.append(
            f"| `{key}` | {values[0]} | {values[1]} | {values[2]} | {values[3]} "
            f"| {metadata['direction']} | {metadata['paper_reference']} |"
        )

    lines.extend(
        [
            "",
            "## Free-Metric Provenance",
            "",
            f"- Source: `{free_mask['source']}`",
            f"- Selected voxels: {free_mask['selected_voxels']} "
            f"({format_number(100.0 * free_mask['actual_fraction'])}%)",
            f"- Interpretation: {free_mask['note']}",
            "",
            "## Mean Metrics by Group",
            "",
        ]
    )
    for group_name, metrics in grouped_mean.items():
        lines.extend([f"### {group_name.replace('_', ' ').title()}", ""])
        lines.extend(["| Metric | Mean | Meaning |", "|---|---:|---|"])
        for key, value in metrics.items():
            description = METRIC_DESCRIPTIONS.get(
                key, "Additional evaluation diagnostic."
            )
            lines.append(f"| `{key}` | {format_number(value)} | {description} |")
        lines.append("")

    lines.extend(
        [
            "## Files",
            "",
            "- `evaluation_results.json`: full configuration, provenance, distributions, and per-sample values.",
            "- `evaluation_samples.csv`: one row per evaluated sample.",
            "- `evaluation_summary.md`: this readable summary.",
            "- `evaluation.log`: execution log and resolved paths.",
            "",
        ]
    )
    return "\n".join(lines)


@torch.inference_mode()
def evaluate(
    args: argparse.Namespace,
    model: torch.nn.Module,
    loader: DataLoader,
    dataset: AutoPhaseDataset,
    device: torch.device,
    free_mask: torch.Tensor | None,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Evaluate all samples with model.eval() and inference-only autograd state."""

    model.eval()
    total: dict[str, float] = {}
    per_sample: list[dict[str, object]] = []
    has_realspace = dataset.mmap_real is not None
    device_free_mask = free_mask.to(device) if free_mask is not None else None
    progress = tqdm(loader, desc="AutoPhaseNN evaluation", unit="batch")
    for batch in progress:
        diff = batch["diff"].to(device, non_blocking=True).float()
        amp = batch["amp"].to(device, non_blocking=True).float()
        phi = batch["phi"].to(device, non_blocking=True).float()
        synchronize(device)
        inference_started = time.perf_counter()
        outputs = model(diff)
        synchronize(device)
        batch_inference_seconds = time.perf_counter() - inference_started
        pred_diff, _pred_obj, pred_amp, pred_phi, support = unpack_outputs(outputs)
        if args.scale_align_loss:
            pred_diff = scale_align_sum(diff, pred_diff)
        inference_ms_per_sample = 1000.0 * batch_inference_seconds / diff.shape[0]

        for index, name in enumerate(batch["name"]):
            sample_slice = slice(index, index + 1)
            metrics = metric_dict(diff[sample_slice], pred_diff[sample_slice])
            if device_free_mask is not None:
                metrics.update(
                    free_metric_dict(
                        diff[sample_slice],
                        pred_diff[sample_slice],
                        device_free_mask,
                    )
                )
            if has_realspace:
                metrics.update(
                    realspace_metric_dict(
                        amp[sample_slice],
                        phi[sample_slice],
                        pred_amp[sample_slice],
                        pred_phi[sample_slice],
                        support[sample_slice],
                        threshold=args.threshold,
                        ssim_window_size=args.ssim_window_size,
                    )
                )
            metrics["inference_ms"] = inference_ms_per_sample
            add_metrics(total, metrics)
            per_sample.append({"name": name, **metrics})
    return per_sample, total


def main() -> int:
    """Load inputs, run evaluation, and write the complete artifact bundle."""

    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    configure_logging(output_dir, args.log_level)
    device = choose_device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    sample_count = (
        min(args.num_samples, args.limit) if args.limit > 0 else args.num_samples
    )
    spatial_shape = (args.shape, args.shape, args.shape)
    data_dir = Path(args.data_dir).expanduser().resolve()
    data_diff_path = optional_data_path(data_dir, args.data_diff)
    data_real_path = optional_data_path(data_dir, args.data_real)
    if data_diff_path is None:
        raise ValueError("--data-diff is required.")

    LOGGER.info("Output directory: %s", output_dir)
    LOGGER.info("Checkpoint: %s", checkpoint_path)
    LOGGER.info("Diffraction data: %s", data_diff_path)
    LOGGER.info("Real-space data: %s", data_real_path or "disabled")
    LOGGER.info("Resolved sample count: %d", sample_count)

    dataset = AutoPhaseDataset(
        data_diff_path,
        data_real_path,
        num_samples=sample_count,
        shape_diff=spatial_shape,
        shape_real=spatial_shape,
        dtype_diff=args.dtype_diff,
        dtype_real=args.dtype_real,
        scale_i=args.scale_i,
        shuffle=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = TFCompatibleAutoPhaseNN(threshold=args.threshold).to(device)
    checkpoint = load_weights(
        model,
        checkpoint_path,
        strict=not args.non_strict_checkpoint,
        map_location=device,
    )
    checkpoint_epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    checkpoint_threshold = (
        checkpoint.get("threshold") if isinstance(checkpoint, dict) else None
    )
    checkpoint_training_args = (
        checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    )
    if checkpoint_threshold is not None and not math.isclose(
        float(checkpoint_threshold),
        args.threshold,
    ):
        LOGGER.warning(
            "Evaluation threshold %.6g differs from checkpoint threshold %.6g.",
            args.threshold,
            float(checkpoint_threshold),
        )
    model.eval()
    free_mask, free_mask_metadata = resolve_free_mask(args, spatial_shape)
    LOGGER.info(
        "Free metrics: source=%s, selected_voxels=%d, fraction=%.6f",
        free_mask_metadata["source"],
        free_mask_metadata["selected_voxels"],
        free_mask_metadata["actual_fraction"],
    )

    warm_up_model(model, loader, device, args.warmup_batches)
    evaluation_started = time.perf_counter()
    per_sample, total = evaluate(args, model, loader, dataset, device, free_mask)
    evaluation_wall_seconds = time.perf_counter() - evaluation_started
    if not per_sample:
        raise RuntimeError("Evaluation produced no samples.")

    statistics = metric_statistics(per_sample)
    mean_metrics = {key: values["mean"] for key, values in statistics.items()}
    inference_seconds = sum(float(row["inference_ms"]) / 1000.0 for row in per_sample)
    mean_inference_ms = statistics["inference_ms"]["mean"]
    throughput = len(per_sample) / max(inference_seconds, 1e-12)
    timing: dict[str, float] = {
        "evaluation_wall_seconds": evaluation_wall_seconds,
        "model_inference_seconds": inference_seconds,
        "mean_inference_ms_per_sample": mean_inference_ms,
        "throughput_samples_per_second": throughput,
    }
    if args.reference_runtime_ms > 0:
        timing["speedup_vs_reference"] = args.reference_runtime_ms / max(
            mean_inference_ms,
            1e-12,
        )

    paper_coverage = {
        key: {
            **metadata,
            "available": key in statistics,
            "description": METRIC_DESCRIPTIONS.get(key, ""),
        }
        for key, metadata in PAPER_METRICS.items()
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint_epoch,
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "num_samples": len(per_sample),
            "batch_size": args.batch_size,
            "support_threshold": args.threshold,
            "ssim_window_size": args.ssim_window_size,
            "realspace_metrics": dataset.mmap_real is not None,
            "scale_align_loss": args.scale_align_loss,
        },
        "checkpoint_metadata": {
            "epoch": checkpoint_epoch,
            "threshold": checkpoint_threshold,
            "training_args": checkpoint_training_args,
        },
        "configuration": vars(args),
        "data": {
            "diffraction": str(data_diff_path),
            "realspace": str(data_real_path) if data_real_path is not None else None,
            "shape": list(spatial_shape),
        },
        "paper_metric_coverage": paper_coverage,
        "free_mask": free_mask_metadata,
        "timing": timing,
        "fixed_metric_groups": fixed_metric_groups(mean_metrics),
        "fixed_metric_descriptions": FIXED_METRIC_DESCRIPTIONS,
        "mean_metric_groups": group_metrics(mean_metrics),
        "metric_descriptions": METRIC_DESCRIPTIONS,
        "metric_statistics": statistics,
        "sum": total,
        "mean": mean_metrics,
        "per_sample": per_sample,
        "notes": {
            "diffraction_storage": (
                "Dataset tensors store abs(FFT), so paper Eq. (1) is modulus MAE and "
                "intensity is modulus squared."
            ),
            "ssim": (
                "real_amp_ssim is local-window 3D SSIM with normalized amplitude data_range=1; "
                "real_amp_global_ssim is retained as a lightweight diagnostic."
            ),
            "phase": "Phase errors are wrapped to [-pi, pi] and evaluated on the true support.",
            "timing": (
                "CUDA is synchronized around each forward pass. Per-sample values divide "
                "batch latency evenly across the batch."
            ),
        },
    }

    json_path = output_dir / "evaluation_results.json"
    csv_path = output_dir / "evaluation_samples.csv"
    markdown_path = output_dir / "evaluation_summary.md"
    write_json(json_path, report)
    write_sample_csv(csv_path, per_sample)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")

    LOGGER.info(
        "\n%s",
        format_fixed_metric_groups(mean_metrics, title="Evaluation mean metrics"),
    )
    LOGGER.info("Wrote JSON report: %s", json_path)
    LOGGER.info("Wrote sample CSV: %s", csv_path)
    LOGGER.info("Wrote Markdown summary: %s", markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
