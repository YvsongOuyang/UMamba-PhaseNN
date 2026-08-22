"""Reconstruct and evaluate real-space objects on AutoPhaseNN memmap data."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pytorch_port.data import AutoPhaseNNPhaseDataset, reciprocal_phase_from_realspace
from pytorch_port.losses import phase_retrieval_wca_components
from pytorch_port.management import (
    DEFAULT_DATA_CONFIG,
    build_data_manifest,
    load_data_config,
    require_data_files,
    runtime_manifest,
)
from pytorch_port.model import (
    MODEL_VARIANTS,
    HighStrainPhaseUNet,
    count_parameters,
    infer_model_variant,
)
from pytorch_port.reconstruction import (
    farfield_modulus_from_realspace,
    realspace_from_modulus_phase,
)


PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = PROJECT_DIR.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from autophasenn_training_pipeline.evaluate import (  # noqa: E402
    center_of_mass_shifts,
    materialize_metric_rows,
    metric_statistics,
    official_post_process_tensor_batch,
    post_process_realspace_batch,
    resolve_postprocess_workers,
    scipy_wrap_shift_batch,
    unwrap_phase_volumes,
)
from autophasenn_training_pipeline.losses import (  # noqa: E402
    FIXED_METRIC_DESCRIPTIONS,
    METRIC_DESCRIPTIONS,
    fixed_metric_groups,
    format_fixed_metric_groups,
    metric_tensor_dict,
    realspace_metric_tensor_dict,
)


LOGGER = logging.getLogger("high_strain.evaluate_autophase")
DEFAULT_EVALUATE_ROOT = PROJECT_DIR / "evaluate"
DEFAULT_CHECKPOINT = (
    "/data_ssd/oyys/autophasenn/autophasenn_pipeline_output/high_strain_cnn/"
    "high_strain_reduced_centered_resume_old60_lr1e-4_plateau_20260820_232732/"
    "checkpoint_best.pt"
)

COMPARABLE_METRICS = {
    "phase_wca": ("lower", "Reciprocal-phase WCA objective used for training."),
    "real_amp_l1": ("lower", "Full-volume post-processed amplitude L1."),
    "real_amp_ssim": ("higher", "Local-window 3D amplitude SSIM."),
    "real_support_iou": ("higher", "Post-processed support intersection-over-union."),
    "real_support_dice": ("higher", "Post-processed support Dice score."),
    "real_support_volume_ratio": ("near 1", "Predicted/true support volume ratio."),
    "real_phase_mae_true_support": (
        "lower",
        "Wrapped phase MAE on the post-processed true support.",
    ),
}

REPROJECTION_METRICS = (
    "paper_modulus_mae",
    "relative_l1_modulus",
    "chi2_modulus",
    "pearson_corr",
    "voxel_mse",
    "voxel_rmse",
)

THRESHOLD_SWEEP_METRICS = (
    "real_amp_l1",
    "real_amp_ssim",
    "real_support_iou",
    "real_support_dice",
    "real_support_volume_ratio",
    "real_phase_mae_true_support",
)


def resolve_output_dir(args: argparse.Namespace, model_variant: str) -> Path:
    """Resolve an explicit directory or the model-specific default."""

    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else DEFAULT_EVALUATE_ROOT / f"evaluate_{model_variant}"
    )
    return output_dir.resolve()


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--data-config", default=str(DEFAULT_DATA_CONFIG))
    bootstrap_args, _ = bootstrap.parse_known_args()
    data_config = load_data_config(bootstrap_args.data_config)
    val_config = data_config["splits"]["val"]
    configured_shape = tuple(int(size) for size in data_config["shape"])
    if len(set(configured_shape)) != 1:
        raise ValueError("HighStrainPhaseUNet requires a cubic data shape.")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default=str(DEFAULT_DATA_CONFIG))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", default=data_config["root"])
    parser.add_argument("--data-diff", default=val_config["diffraction"])
    parser.add_argument("--data-real", default=val_config["realspace"])
    parser.add_argument(
        "--num-samples",
        type=int,
        default=int(val_config["num_samples"]),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shape", type=int, default=configured_shape[0])
    parser.add_argument("--dtype-diff", default=data_config["dtypes"]["diffraction"])
    parser.add_argument("--dtype-real", default=data_config["dtypes"]["realspace"])
    parser.add_argument(
        "--input-log-data",
        action=argparse.BooleanOptionalAction,
        default=data_config.get("input_preprocessing", {}).get("transform") == "log1p",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--postprocess-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--model-variant",
        choices=("auto", *MODEL_VARIANTS),
        default="auto",
        help="Infer the architecture from the checkpoint by default.",
    )
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument(
        "--threshold-sweep",
        type=float,
        nargs="+",
        default=(),
        metavar="T",
        help=(
            "Evaluate additional support thresholds in the same pass. The primary "
            "--threshold is always included and remains the headline result."
        ),
    )
    parser.add_argument("--ssim-window-size", type=int, default=7)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument(
        "--ambiguity-mode",
        choices=("twin_aligned", "raw"),
        default="twin_aligned",
        help=(
            "Use ground truth only during evaluation to resolve the conjugate/twin "
            "sign permitted by the published WCA loss, or reconstruct the raw phase."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Artifact directory; empty uses "
            "<project>/evaluate/evaluate_<checkpoint-variant>."
        ),
    )
    parser.add_argument(
        "--save-realspace",
        action="store_true",
        help="Write reconstructed complex64 objects as an AutoPhaseNN-style raw memmap.",
    )
    parser.add_argument(
        "--save-reciprocal-phase",
        action="store_true",
        help="Write the selected float32 reciprocal phases as a raw memmap.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args()
    args.data_config = str(Path(args.data_config).expanduser().resolve())
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive.")
    if args.limit < 0:
        raise ValueError("--limit cannot be negative.")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Batch size must be positive and workers cannot be negative.")
    thresholds = (args.threshold, *args.threshold_sweep)
    if any(not math.isfinite(value) or value < 0 for value in thresholds):
        raise ValueError("Support thresholds must be finite and nonnegative.")
    if args.ssim_window_size < 1 or args.ssim_window_size % 2 == 0:
        raise ValueError("--ssim-window-size must be a positive odd integer.")
    if args.ssim_window_size > args.shape:
        raise ValueError("--ssim-window-size cannot exceed --shape.")


def configure_logging(output_dir: Path, level: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.handlers.clear()
    LOGGER.setLevel(getattr(logging, level))
    LOGGER.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_dir / "evaluation.log", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    requested_variant: str = "auto",
) -> tuple[HighStrainPhaseUNet, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    checkpoint_variant = infer_model_variant(state_dict)
    if requested_variant != "auto" and requested_variant != checkpoint_variant:
        raise ValueError(
            f"Checkpoint uses model variant {checkpoint_variant!r}, but "
            f"{requested_variant!r} was requested."
        )
    model = HighStrainPhaseUNet(model_variant=checkpoint_variant).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    return model, metadata


def select_reciprocal_phase(
    predicted_phase: torch.Tensor,
    target_phase: torch.Tensor,
    weights: torch.Tensor,
    ambiguity_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the direct or conjugate/twin-equivalent reciprocal phase."""

    direct, inverted = phase_retrieval_wca_components(
        predicted_phase,
        target_phase,
        weights,
    )
    twin_selected = inverted < direct
    if ambiguity_mode == "twin_aligned":
        mask = twin_selected.reshape((-1,) + (1,) * (predicted_phase.ndim - 1))
        selected_phase = torch.where(mask, -predicted_phase, predicted_phase)
    else:
        selected_phase = predicted_phase
    return selected_phase, direct, inverted, twin_selected


def resolve_threshold_sweep(
    primary_threshold: float,
    requested_thresholds: tuple[float, ...] | list[float],
) -> tuple[float, ...]:
    """Return stable, unique sweep thresholds with the primary value included."""

    if not requested_thresholds:
        return ()
    resolved: list[float] = []
    for value in (primary_threshold, *requested_thresholds):
        threshold = float(value)
        if not any(math.isclose(threshold, current) for current in resolved):
            resolved.append(threshold)
    return tuple(sorted(resolved))


def unwrap_realspace_phase_pair(
    true_phi: torch.Tensor,
    predicted_phi: torch.Tensor,
    executor: ThreadPoolExecutor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unwrap target and prediction once for all support thresholds."""

    batch_size = predicted_phi.shape[0]
    phase_batch = np.concatenate(
        (
            true_phi.detach().to(device="cpu", dtype=torch.float32).numpy(),
            predicted_phi.detach().to(device="cpu", dtype=torch.float32).numpy(),
        ),
        axis=0,
    )
    unwrapped = unwrap_phase_volumes(phase_batch, executor=executor)
    unwrapped_tensor = torch.from_numpy(unwrapped).to(
        device=predicted_phi.device,
        dtype=predicted_phi.dtype,
    )
    return torch.split(unwrapped_tensor, [batch_size, batch_size], dim=0)


def post_process_unwrapped_realspace(
    true_amp: torch.Tensor,
    true_phi_unwrapped: torch.Tensor,
    predicted_amp: torch.Tensor,
    predicted_phi_unwrapped: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, ...]:
    """Apply comparable AutoPhaseNN post-processing at one support threshold."""

    true_amp_device = true_amp.to(
        device=predicted_amp.device,
        dtype=predicted_amp.dtype,
        non_blocking=predicted_amp.device.type == "cuda",
    )
    true_amp_post, true_phi_post = official_post_process_tensor_batch(
        true_amp_device,
        true_phi_unwrapped,
        threshold=threshold,
    )
    predicted_amp_post, predicted_phi_post = official_post_process_tensor_batch(
        predicted_amp,
        predicted_phi_unwrapped,
        threshold=threshold,
    )
    predicted_support = (predicted_amp >= threshold).float()
    support_shifts = center_of_mass_shifts(predicted_support)
    predicted_support_post = scipy_wrap_shift_batch(
        predicted_support,
        support_shifts,
    )
    return (
        true_amp_post,
        true_phi_post,
        predicted_amp_post,
        predicted_phi_post,
        predicted_support_post,
    )


def write_sample_csv(path: Path, rows: list[dict[str, object]]) -> None:
    metric_keys = sorted({key for row in rows for key in row if key != "name"})
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["name", *metric_keys])
        writer.writeheader()
        writer.writerows(rows)


def summarize_threshold_sweep(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    """Aggregate sweep metrics and identify two transparent operating points."""

    thresholds = sorted({float(row["threshold"]) for row in rows})
    summaries: list[dict[str, object]] = []
    for threshold in thresholds:
        threshold_rows = [
            row for row in rows if math.isclose(float(row["threshold"]), threshold)
        ]
        statistics = metric_statistics(threshold_rows)
        statistics.pop("threshold", None)
        summaries.append(
            {
                "threshold": threshold,
                "num_samples": len(threshold_rows),
                "mean": {
                    metric: statistics[metric]["mean"]
                    for metric in THRESHOLD_SWEEP_METRICS
                },
                "metric_statistics": statistics,
            }
        )

    best_iou = max(
        summaries,
        key=lambda item: item["mean"]["real_support_iou"],
    )
    closest_volume = min(
        summaries,
        key=lambda item: abs(item["mean"]["real_support_volume_ratio"] - 1.0),
    )
    return {
        "enabled": True,
        "selection_scope": "validation_diagnostic_only",
        "primary_threshold_unchanged": True,
        "metrics": list(THRESHOLD_SWEEP_METRICS),
        "summaries": summaries,
        "best_mean_iou_threshold": best_iou["threshold"],
        "closest_mean_volume_ratio_threshold": closest_volume["threshold"],
    }


def write_threshold_sweep_summary_csv(
    path: Path,
    sweep: dict[str, object],
) -> None:
    """Write one compact row per support threshold."""

    fieldnames = ["threshold", "num_samples", *THRESHOLD_SWEEP_METRICS]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for summary in sweep["summaries"]:
            writer.writerow(
                {
                    "threshold": summary["threshold"],
                    "num_samples": summary["num_samples"],
                    **summary["mean"],
                }
            )


def format_number(value: object) -> str:
    """Format report scalars compactly while preserving small values."""

    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}" if math.isfinite(value) else "n/a"
    return str(value)


def high_strain_metric_groups(mean: dict[str, float]) -> dict[str, dict[str, float]]:
    """Group learned-quality and construction-only metrics separately."""

    group_keys = {
        "phase_retrieval": (
            "phase_wca",
            "phase_wca_direct",
            "phase_wca_inverted",
            "twin_flip_selected",
        ),
        "realspace_primary": (
            "real_amp_l1",
            "real_amp_ssim",
            "real_amp_global_ssim",
            "real_support_iou",
            "real_support_dice",
            "real_support_pred_fraction",
            "real_support_volume_ratio",
            "real_phase_mae_true_support",
        ),
        "realspace_diagnostic": (
            "real_amp_mse",
            "real_amp_rmse",
            "real_amp_rel_l1",
            "real_support_l1",
            "real_support_rmse",
            "real_support_true_fraction",
            "real_phase_mae_intersection",
            "real_phase_rmse_true_support",
        ),
        "reprojection_identity": REPROJECTION_METRICS,
        "timing": ("inference_ms",),
    }
    return {
        group_name: {key: mean[key] for key in keys if key in mean}
        for group_name, keys in group_keys.items()
    }


def render_markdown(report: dict[str, object]) -> str:
    """Render a summary that distinguishes learned quality from identities."""

    run = report["run"]
    mean = report["mean"]
    statistics = report["metric_statistics"]
    timing = report["timing"]
    lines = [
        "# HighStrain AutoPhaseNN Evaluation Summary",
        "",
        "## Run",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Checkpoint | `{run['checkpoint']}` |",
        f"| Checkpoint epoch | {format_number(run['checkpoint_epoch'])} |",
        f"| Model variant | `{run['model_variant']}` |",
        f"| Model parameters | {run['model_parameters']:,} |",
        f"| Project version | `{run['project_version']}` |",
        f"| Git commit | `{run['git_commit']}` |",
        f"| Samples | {run['num_samples']} |",
        f"| Batch size | {run['batch_size']} |",
        f"| Device | `{run['device']}` |",
        f"| GPU | {format_number(run['gpu_name'])} |",
        f"| Ambiguity mode | `{run['ambiguity_mode']}` |",
        f"| Support threshold | {format_number(run['support_threshold'])} |",
        f"| Phase unwrap workers | {run['postprocess_workers']} |",
        f"| Evaluation wall time | {format_number(timing['evaluation_wall_seconds'])} s |",
        f"| Mean inference | {format_number(timing['mean_inference_ms_per_sample'])} ms/sample |",
        f"| Throughput | {format_number(timing['throughput_samples_per_second'])} samples/s |",
        "",
        "## Metric Interpretation",
        "",
        "The real-space metrics below use the same AutoPhaseNN post-processing and "
        "metric implementations and are directly comparable. Reciprocal modulus "
        "metrics are not model-quality measurements for this method: reconstruction "
        "explicitly reuses the measured modulus, so their near-zero errors only test "
        "FFT/reprojection consistency.",
        "",
        "## Comparable Quality Metrics",
        "",
        "| Metric | Mean | Std | P50 | P95 | Better | Meaning |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for key, (direction, description) in COMPARABLE_METRICS.items():
        stats = statistics.get(key)
        if stats is None:
            values = ["n/a"] * 4
        else:
            values = [
                format_number(stats[statistic])
                for statistic in ("mean", "std", "p50", "p95")
            ]
        lines.append(
            f"| `{key}` | {values[0]} | {values[1]} | {values[2]} | {values[3]} "
            f"| {direction} | {description} |"
        )

    threshold_sweep = report["threshold_sweep"]
    if threshold_sweep["enabled"]:
        lines.extend(
            [
                "",
                "## Support Threshold Sweep",
                "",
                "The primary support threshold remains unchanged. Sweep-selected "
                "operating points are validation diagnostics and do not replace the "
                "headline metrics above.",
                "",
                "| Threshold | Primary | Amp L1 | Amp SSIM | Support IoU | "
                "Support Dice | Volume ratio | Phase MAE |",
                "|---:|:---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for summary in threshold_sweep["summaries"]:
            threshold = summary["threshold"]
            values = summary["mean"]
            is_primary = "yes" if math.isclose(
                threshold,
                run["support_threshold"],
            ) else ""
            lines.append(
                f"| {format_number(threshold)} | {is_primary} | "
                f"{format_number(values['real_amp_l1'])} | "
                f"{format_number(values['real_amp_ssim'])} | "
                f"{format_number(values['real_support_iou'])} | "
                f"{format_number(values['real_support_dice'])} | "
                f"{format_number(values['real_support_volume_ratio'])} | "
                f"{format_number(values['real_phase_mae_true_support'])} |"
            )
        lines.extend(
            [
                "",
                "- Best mean-IoU threshold: "
                f"`{format_number(threshold_sweep['best_mean_iou_threshold'])}`.",
                "- Threshold with mean volume ratio closest to one: "
                f"`{format_number(threshold_sweep['closest_mean_volume_ratio_threshold'])}`.",
            ]
        )

    lines.extend(
        [
            "",
            "## AutoPhaseNN-Compatible Fixed Metrics",
            "",
            "The `FT` row is retained for file-format compatibility but is a "
            "reprojection identity, not an independently predicted quantity.",
            "",
            "| Group | Metric | Mean |",
            "|---|---|---:|",
        ]
    )
    for group_name, values in fixed_metric_groups(mean).items():
        display_group = "FT (reprojection only)" if group_name == "FT" else group_name
        for metric_name, value in values.items():
            lines.append(
                f"| {display_group} | {metric_name} | {format_number(value)} |"
            )

    lines.extend(["", "## Mean Metrics by Group", ""])
    descriptions = report["metric_descriptions"]
    for group_name, metrics in report["mean_metric_groups"].items():
        lines.extend(
            [
                f"### {group_name.replace('_', ' ').title()}",
                "",
                "| Metric | Mean | Meaning |",
                "|---|---:|---|",
            ]
        )
        for key, value in metrics.items():
            lines.append(
                f"| `{key}` | {format_number(value)} | "
                f"{descriptions.get(key, 'Additional diagnostic.')} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation Notes",
            "",
            "- `twin_aligned` uses the target only to choose between the two reciprocal-"
            "phase signs treated as equivalent by the published WCA loss.",
            "- `phase_wca`, amplitude SSIM, support IoU/Dice, and real-space phase error "
            "are the meaningful quality indicators for this architecture.",
            "- A support volume ratio far from one indicates that reciprocal-phase errors "
            "spread reconstructed energy outside the object.",
            "",
            "## Files",
            "",
            "- `evaluation_results.json`: configuration, provenance, distributions, and per-sample metrics.",
            "- `evaluation_samples.csv`: one row per sample.",
            "- `evaluation_summary.md`: this readable summary.",
            "- `evaluation.log`: execution log.",
        ]
    )
    if threshold_sweep["enabled"]:
        lines.extend(
            [
                "- `evaluation_threshold_sweep.csv`: one aggregate row per threshold.",
                "- `evaluation_threshold_sweep_samples.csv`: per-sample sweep metrics.",
            ]
        )
    return "\n".join(lines) + "\n"


@torch.inference_mode()
def evaluate(
    args: argparse.Namespace,
    model: HighStrainPhaseUNet,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    sample_count: int,
    postprocess_workers: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    model.eval()
    rows: list[dict[str, object]] = []
    threshold_sweep_rows: list[dict[str, object]] = []
    sweep_thresholds = resolve_threshold_sweep(
        args.threshold,
        args.threshold_sweep,
    )
    shape = (args.shape, args.shape, args.shape)
    realspace_writer = (
        np.memmap(
            output_dir / "predicted_realspace.npy",
            dtype="complex64",
            mode="w+",
            shape=(sample_count,) + shape,
        )
        if args.save_realspace
        else None
    )
    phase_writer = (
        np.memmap(
            output_dir / "predicted_reciprocal_phase.npy",
            dtype="float32",
            mode="w+",
            shape=(sample_count,) + shape,
        )
        if args.save_reciprocal_phase
        else None
    )
    executor = (
        ThreadPoolExecutor(
            max_workers=postprocess_workers,
            thread_name_prefix="phase-unwrap",
        )
        if postprocess_workers > 1
        else None
    )
    offset = 0
    try:
        for batch in tqdm(loader, desc="high_strain AutoPhaseNN evaluation", unit="batch"):
            model_input = batch["input"].to(device, non_blocking=True).float()
            measured_modulus = batch["diffraction"].to(
                device,
                non_blocking=True,
            ).float()
            true_object = batch["realspace"].to(device, non_blocking=True)
            target_reciprocal_phase = reciprocal_phase_from_realspace(true_object)

            synchronize(device)
            inference_started = time.perf_counter()
            predicted_phase = model(model_input)
            synchronize(device)
            inference_seconds = time.perf_counter() - inference_started

            selected_phase, wca_direct, wca_inverted, twin_selected = (
                select_reciprocal_phase(
                    predicted_phase,
                    target_reciprocal_phase,
                    model_input[:, 0],
                    args.ambiguity_mode,
                )
            )
            predicted_object = realspace_from_modulus_phase(
                measured_modulus,
                selected_phase,
            )
            predicted_modulus = farfield_modulus_from_realspace(predicted_object)
            true_object = true_object[:, None]
            true_amp = true_object.abs().float()
            true_phi = torch.angle(true_object).float()
            predicted_amp = predicted_object.abs().float()
            predicted_phi = torch.angle(predicted_object).float()

            if sweep_thresholds:
                true_phi_unwrapped, predicted_phi_unwrapped = (
                    unwrap_realspace_phase_pair(
                        true_phi,
                        predicted_phi,
                        executor,
                    )
                )
                primary_realspace_metrics = None
                for threshold in sweep_thresholds:
                    processed = post_process_unwrapped_realspace(
                        true_amp,
                        true_phi_unwrapped,
                        predicted_amp,
                        predicted_phi_unwrapped,
                        threshold,
                    )
                    realspace_metrics = realspace_metric_tensor_dict(
                        *processed,
                        threshold=threshold,
                        ssim_window_size=args.ssim_window_size,
                    )
                    if math.isclose(threshold, args.threshold):
                        primary_realspace_metrics = realspace_metrics
                    sweep_metrics = {
                        key: realspace_metrics[key]
                        for key in THRESHOLD_SWEEP_METRICS
                    }
                    batch_sweep_rows = materialize_metric_rows(sweep_metrics)
                    for name, metrics in zip(batch["name"], batch_sweep_rows):
                        threshold_sweep_rows.append(
                            {
                                "name": name,
                                "threshold": threshold,
                                **metrics,
                            }
                        )
                if primary_realspace_metrics is None:
                    raise RuntimeError("Primary threshold missing from threshold sweep.")
            else:
                predicted_support = (predicted_amp >= args.threshold).float()
                processed = post_process_realspace_batch(
                    true_amp,
                    true_phi,
                    predicted_amp,
                    predicted_phi,
                    predicted_support,
                    threshold=args.threshold,
                    executor=executor,
                )
                primary_realspace_metrics = realspace_metric_tensor_dict(
                    *processed,
                    threshold=args.threshold,
                    ssim_window_size=args.ssim_window_size,
                )

            metric_tensors = metric_tensor_dict(measured_modulus, predicted_modulus)
            metric_tensors.update(primary_realspace_metrics)
            metric_tensors.update(
                {
                    "phase_wca_direct": wca_direct,
                    "phase_wca_inverted": wca_inverted,
                    "phase_wca": torch.minimum(wca_direct, wca_inverted),
                    "twin_flip_selected": twin_selected.float(),
                    "inference_ms": torch.full_like(
                        wca_direct,
                        1000.0 * inference_seconds / model_input.shape[0],
                    ),
                }
            )
            batch_rows = materialize_metric_rows(metric_tensors)
            for name, metrics in zip(batch["name"], batch_rows):
                rows.append({"name": name, **metrics})

            batch_size = model_input.shape[0]
            if realspace_writer is not None:
                realspace_writer[offset : offset + batch_size] = (
                    predicted_object[:, 0].detach().cpu().numpy().astype(np.complex64)
                )
            if phase_writer is not None:
                phase_writer[offset : offset + batch_size] = (
                    selected_phase[:, 0].detach().cpu().numpy().astype(np.float32)
                )
            offset += batch_size
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        if realspace_writer is not None:
            realspace_writer.flush()
        if phase_writer is not None:
            phase_writer.flush()
    return rows, threshold_sweep_rows


@torch.inference_mode()
def warm_up(
    model: HighStrainPhaseUNet,
    loader: DataLoader,
    device: torch.device,
    batches: int,
) -> None:
    if batches <= 0:
        return
    completed = 0
    for batch in loader:
        model(batch["input"].to(device, non_blocking=True).float())
        completed += 1
        if completed >= batches:
            break
    synchronize(device)
    LOGGER.info("Completed %d warmup batch(es).", completed)


def main() -> int:
    args = parse_args()
    validate_args(args)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    device = choose_device(args.device)
    model, checkpoint_metadata = load_model(
        checkpoint_path,
        device,
        args.model_variant,
    )
    output_dir = resolve_output_dir(args, model.model_variant)
    configure_logging(output_dir, args.log_level)

    sample_count = min(args.num_samples, args.limit) if args.limit > 0 else args.num_samples
    data_config = load_data_config(args.data_config)
    data_dir = Path(args.data_dir).expanduser().resolve()
    shape = (args.shape, args.shape, args.shape)
    data_manifest = build_data_manifest(
        config=data_config,
        root=data_dir,
        shape=shape,
        diffraction_dtype=args.dtype_diff,
        realspace_dtype=args.dtype_real,
        splits={
            "val": {
                "diffraction": args.data_diff,
                "realspace": args.data_real,
                "num_samples": sample_count,
            }
        },
        input_log_data=args.input_log_data,
    )
    data_manifest["file_status"] = require_data_files(data_manifest)
    dataset = AutoPhaseNNPhaseDataset(
        data_dir / args.data_diff,
        data_dir / args.data_real,
        sample_count,
        shape=shape,
        diffraction_dtype=args.dtype_diff,
        realspace_dtype=args.dtype_real,
        input_log_data=args.input_log_data,
        return_diffraction_modulus=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    runtime = runtime_manifest(device)
    checkpoint_version = checkpoint_metadata.get("project_version")
    if checkpoint_version and checkpoint_version != runtime["project_version"]:
        LOGGER.warning(
            "Checkpoint project version %s differs from evaluator version %s.",
            checkpoint_version,
            runtime["project_version"],
        )
    LOGGER.info("Checkpoint: %s", checkpoint_path)
    LOGGER.info("Data: %s samples from %s", sample_count, data_dir)
    LOGGER.info(
        "Model: %s | %s parameters | device=%s | ambiguity=%s",
        model.model_variant,
        f"{count_parameters(model):,}",
        device,
        args.ambiguity_mode,
    )
    postprocess_workers = resolve_postprocess_workers(
        args.postprocess_workers,
        args.batch_size,
    )
    LOGGER.info(
        "Output: %s | post-process workers=%d",
        output_dir,
        postprocess_workers,
    )
    sweep_thresholds = resolve_threshold_sweep(
        args.threshold,
        args.threshold_sweep,
    )
    if sweep_thresholds:
        LOGGER.info(
            "Support threshold sweep: %s | primary=%.6g",
            ", ".join(format_number(value) for value in sweep_thresholds),
            args.threshold,
        )
    warm_up(model, loader, device, args.warmup_batches)

    started = time.perf_counter()
    rows, threshold_sweep_rows = evaluate(
        args,
        model,
        loader,
        device,
        output_dir,
        sample_count,
        postprocess_workers,
    )
    wall_seconds = time.perf_counter() - started
    if not rows:
        raise RuntimeError("Evaluation produced no samples.")
    threshold_sweep = (
        summarize_threshold_sweep(threshold_sweep_rows)
        if threshold_sweep_rows
        else {
            "enabled": False,
            "selection_scope": "disabled",
            "primary_threshold_unchanged": True,
            "metrics": list(THRESHOLD_SWEEP_METRICS),
            "summaries": [],
            "best_mean_iou_threshold": None,
            "closest_mean_volume_ratio_threshold": None,
        }
    )
    statistics = metric_statistics(rows)
    mean = {key: values["mean"] for key, values in statistics.items()}
    mean_inference_ms = statistics["inference_ms"]["mean"]
    model_inference_seconds = sum(
        float(row["inference_ms"]) / 1000.0 for row in rows
    )
    timing = {
        "evaluation_wall_seconds": wall_seconds,
        "model_inference_seconds": model_inference_seconds,
        "mean_inference_ms_per_sample": mean_inference_ms,
        "throughput_samples_per_second": len(rows)
        / max(model_inference_seconds, 1e-12),
    }
    metric_descriptions = {
        **METRIC_DESCRIPTIONS,
        "paper_modulus_mae": (
            "Measured-modulus reprojection L1; near zero by construction and not "
            "comparable to an independently predicted modulus."
        ),
        "relative_l1_modulus": "Scale-normalized reprojection consistency.",
        "chi2_modulus": "Reprojection chi-square consistency.",
        "pearson_corr": "Measured/reprojected modulus correlation.",
        "voxel_mse": "Measured/reprojected modulus voxel MSE.",
        "voxel_rmse": "Measured/reprojected modulus voxel RMSE.",
        "phase_wca": "Published symmetry-aware reciprocal-phase WCA loss.",
        "phase_wca_direct": "WCA error against the direct reciprocal phase.",
        "phase_wca_inverted": "WCA error against the conjugate/twin phase.",
        "twin_flip_selected": (
            "Fraction indicator for evaluation-time conjugate/twin sign selection."
        ),
        "inference_ms": "Mean model forward latency assigned to each sample.",
    }
    fixed_metric_descriptions = {
        **FIXED_METRIC_DESCRIPTIONS,
        "FT/L1": "Measured-modulus reprojection L1; construction-only consistency.",
        "FT/MSE": "Measured-modulus reprojection MSE; construction-only consistency.",
        "FT/RMSE": "Measured-modulus reprojection RMSE; construction-only consistency.",
        "FT/RelL1": "Relative reprojection L1; construction-only consistency.",
    }
    report: dict[str, object] = {
        "schema_version": 3,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint_metadata.get("epoch"),
            "checkpoint_project_version": checkpoint_version,
            "checkpoint_git_commit": checkpoint_metadata.get("git_commit"),
            "model_variant": model.model_variant,
            "model_parameters": count_parameters(model),
            "project_version": runtime["project_version"],
            "git_commit": runtime["git_commit"],
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": runtime["gpu_name"],
            "num_samples": len(rows),
            "batch_size": args.batch_size,
            "ambiguity_mode": args.ambiguity_mode,
            "support_threshold": args.threshold,
            "support_threshold_sweep": list(sweep_thresholds),
            "ssim_window_size": args.ssim_window_size,
            "realspace_post_process": "official_skimage_unwrap_batched_torch",
            "postprocess_tensor_device": str(device),
            "postprocess_workers": postprocess_workers,
        },
        "checkpoint_metadata": {
            "epoch": checkpoint_metadata.get("epoch"),
            "project_version": checkpoint_version,
            "git_commit": checkpoint_metadata.get("git_commit"),
            "training_args": checkpoint_metadata.get("args", {}),
        },
        "configuration": vars(args),
        "resolved_output_dir": str(output_dir),
        "runtime": runtime,
        "data": data_manifest,
        "reconstruction": {
            "spectrum": "measured_modulus * exp(1j * selected_reciprocal_phase)",
            "inverse": "fftshift(ifftn(ifftshift(spectrum)))",
            "selected_phase": args.ambiguity_mode,
        },
        "artifacts": {
            "predicted_realspace": (
                str(output_dir / "predicted_realspace.npy")
                if args.save_realspace
                else None
            ),
            "predicted_reciprocal_phase": (
                str(output_dir / "predicted_reciprocal_phase.npy")
                if args.save_reciprocal_phase
                else None
            ),
            "threshold_sweep_summary": (
                str(output_dir / "evaluation_threshold_sweep.csv")
                if threshold_sweep["enabled"]
                else None
            ),
            "threshold_sweep_samples": (
                str(output_dir / "evaluation_threshold_sweep_samples.csv")
                if threshold_sweep["enabled"]
                else None
            ),
            "memmap_shape": [sample_count, *shape],
        },
        "timing": timing,
        "comparison_eligibility": {
            "directly_comparable": list(COMPARABLE_METRICS),
            "construction_only": list(REPROJECTION_METRICS),
            "reason": (
                "The reconstructed spectrum reuses measured modulus; only its phase "
                "is predicted. Modulus reprojection errors therefore cannot be compared "
                "with AutoPhaseNN's independently predicted diffraction modulus."
            ),
        },
        "fixed_metric_groups": fixed_metric_groups(mean),
        "fixed_metric_descriptions": fixed_metric_descriptions,
        "mean_metric_groups": high_strain_metric_groups(mean),
        "metric_descriptions": metric_descriptions,
        "metric_statistics": statistics,
        "mean": mean,
        "threshold_sweep": threshold_sweep,
        "per_sample": rows,
        "notes": {
            "metric_compatibility": (
                "Real-space values use the same AutoPhaseNN official post-processing "
                "and metric functions as autophasenn_training_pipeline/evaluate.py."
            ),
            "reciprocal_metrics": (
                "Modulus metrics are FFT/reprojection checks only. They are nearly exact "
                "because measured modulus is reused and must not be interpreted as model "
                "prediction quality."
            ),
            "twin_alignment": (
                "twin_aligned uses target data only for evaluation-time choice between "
                "the two signs explicitly treated as equivalent by the published loss."
            ),
            "threshold_calibration": (
                "Threshold-sweep selections are validation diagnostics. The primary "
                "threshold remains the fair AutoPhaseNN comparison point."
            ),
        },
    }
    json_path = output_dir / "evaluation_results.json"
    csv_path = output_dir / "evaluation_samples.csv"
    markdown_path = output_dir / "evaluation_summary.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    write_sample_csv(csv_path, rows)
    if threshold_sweep["enabled"]:
        write_threshold_sweep_summary_csv(
            output_dir / "evaluation_threshold_sweep.csv",
            threshold_sweep,
        )
        write_sample_csv(
            output_dir / "evaluation_threshold_sweep_samples.csv",
            threshold_sweep_rows,
        )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    LOGGER.info("\n%s", format_fixed_metric_groups(mean, title="Mean metrics"))
    LOGGER.info(
        "Comparable quality | WCA=%.6g | amp SSIM=%.6g | support IoU=%.6g | "
        "phase MAE=%.6g",
        mean["phase_wca"],
        mean["real_amp_ssim"],
        mean["real_support_iou"],
        mean["real_phase_mae_true_support"],
    )
    LOGGER.info(
        "Reprojection modulus MAE %.6g is construction-only, not model quality.",
        mean["paper_modulus_mae"],
    )
    if threshold_sweep["enabled"]:
        LOGGER.info(
            "Threshold sweep | best mean IoU at %.6g | volume ratio closest to 1 at %.6g",
            threshold_sweep["best_mean_iou_threshold"],
            threshold_sweep["closest_mean_volume_ratio_threshold"],
        )
    LOGGER.info("Wrote results: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
