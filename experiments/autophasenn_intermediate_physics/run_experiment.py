"""Test a latent-space physics-consistency hypothesis in baseline AutoPhaseNN."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from autophasenn_training_pipeline.dataset import AutoPhaseDataset
from autophasenn_training_pipeline.model_tf_compatible import (
    TFCompatibleAutoPhaseNN,
    load_weights,
)


LOGGER = logging.getLogger("autophasenn.intermediate_physics")
EPS = 1e-8
VARIANTS = ("paired", "zero_phase", "rolled_phase")
METRIC_NAMES = (
    "normalized_mae",
    "normalized_rmse",
    "relative_l1",
    "relative_l2",
    "pearson",
    "cosine",
    "raw_rms_ratio",
)
ERROR_METRICS = {
    "normalized_mae",
    "normalized_rmse",
    "relative_l1",
    "relative_l2",
}


@dataclass(frozen=True)
class FeatureLevel:
    """One spatially matched encoder/amplitude/phase feature triplet."""

    name: str
    spatial_size: int
    encoder_bn: str
    amplitude_bn: str
    phase_bn: str


FEATURE_LEVELS = (
    FeatureLevel(
        "latent_8",
        8,
        "batch_normalization_7",
        "batch_normalization_11",
        "batch_normalization_19",
    ),
    FeatureLevel(
        "latent_16",
        16,
        "batch_normalization_5",
        "batch_normalization_13",
        "batch_normalization_21",
    ),
    FeatureLevel(
        "latent_32",
        32,
        "batch_normalization_3",
        "batch_normalization_15",
        "batch_normalization_23",
    ),
    FeatureLevel(
        "latent_64",
        64,
        "batch_normalization_1",
        "batch_normalization_17",
        "batch_normalization_25",
    ),
)


class BaselineFeatureCapture:
    """Capture matched baseline features without modifying the model forward path."""

    def __init__(self, model: TFCompatibleAutoPhaseNN) -> None:
        self.features: dict[str, torch.Tensor] = {}
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

        for level in FEATURE_LEVELS:
            self.handles.append(
                model.layers[level.encoder_bn].register_forward_hook(
                    self._make_output_hook(f"{level.name}.encoder")
                )
            )
            self.handles.append(
                model.layers[level.amplitude_bn].register_forward_hook(
                    self._make_output_hook(f"{level.name}.amplitude")
                )
            )
            self.handles.append(
                model.layers[level.phase_bn].register_forward_hook(
                    self._make_output_hook(f"{level.name}.phase")
                )
            )

    def _make_output_hook(self, name: str) -> Callable:
        def hook(_module, _inputs, output) -> None:
            self.features[name] = output.detach()

        return hook

    def clear(self) -> None:
        self.features.clear()

    def snapshot(self) -> dict[str, torch.Tensor]:
        expected = {
            f"{level.name}.{branch}"
            for level in FEATURE_LEVELS
            for branch in ("encoder", "amplitude", "phase")
        }
        missing = sorted(expected.difference(self.features))
        if missing:
            raise RuntimeError(f"Feature hooks did not capture: {missing}")
        return dict(self.features)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def configure_logging(output_dir: Path, level: str) -> None:
    """Write the same concise log to the console and experiment directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for handler in LOGGER.handlers:
        handler.close()
    LOGGER.handlers.clear()
    LOGGER.setLevel(getattr(logging, level))
    LOGGER.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    log_file = logging.FileHandler(output_dir / "run.log", mode="w", encoding="utf-8")
    log_file.setFormatter(formatter)
    LOGGER.addHandler(console)
    LOGGER.addHandler(log_file)


def choose_device(name: str) -> torch.device:
    """Resolve auto/CUDA selection with an explicit fallback."""

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA was requested but is unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(name)


def match_channels(
    feature: torch.Tensor,
    output_channels: int,
    reduction: str,
) -> torch.Tensor:
    """Match decoder width by deterministic grouping or channel repetition."""

    batch, channels, depth, height, width = feature.shape
    if channels < output_channels:
        if output_channels % channels != 0:
            raise ValueError(
                f"Cannot expand {channels} channels to {output_channels} evenly."
            )
        return feature.repeat_interleave(output_channels // channels, dim=1)
    if channels % output_channels != 0:
        raise ValueError(
            f"Cannot reduce {channels} channels to {output_channels} equal groups."
        )
    grouped = feature.reshape(
        batch,
        output_channels,
        channels // output_channels,
        depth,
        height,
        width,
    )
    if reduction == "mean":
        return grouped.mean(dim=2)
    if reduction == "rms":
        return grouped.square().mean(dim=2).clamp_min(EPS).sqrt()
    raise ValueError(f"Unknown channel reduction: {reduction}")


def spatial_rms_normalize(tensor: torch.Tensor) -> torch.Tensor:
    """Remove arbitrary per-channel scale while retaining spatial structure."""

    rms = tensor.square().mean(dim=(-3, -2, -1), keepdim=True).clamp_min(EPS).sqrt()
    return tensor / rms


def latent_forward_model(
    amplitude_feature: torch.Tensor,
    phase_feature: torch.Tensor,
    target_channels: int,
    support_threshold: float,
    variant: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply an explicit latent analogue of the AutoPhaseNN physics model.

    Decoder channels are matched to the encoder width. RMS is interpreted as a
    non-negative amplitude and phase features are mapped to [-pi, pi] with the
    baseline output nonlinearity. The FFT is applied independently to each
    resulting latent channel.
    """

    amplitude = match_channels(amplitude_feature, target_channels, reduction="rms")
    amplitude = amplitude / amplitude.amax(dim=(-3, -2, -1), keepdim=True).clamp_min(EPS)
    phase = math.pi * torch.tanh(
        match_channels(phase_feature, target_channels, reduction="mean")
    )

    if variant == "zero_phase":
        phase = torch.zeros_like(phase)
    elif variant == "rolled_phase":
        shifts = tuple(size // 2 for size in phase.shape[-3:])
        phase = torch.roll(phase, shifts=shifts, dims=(-3, -2, -1))
    elif variant != "paired":
        raise ValueError(f"Unknown latent-forward variant: {variant}")

    support = amplitude >= support_threshold
    latent_object = torch.complex(
        amplitude * torch.cos(phase),
        amplitude * torch.sin(phase),
    )
    latent_object = latent_object * support
    shifted = torch.fft.ifftshift(latent_object, dim=(-3, -2, -1))
    diffraction = torch.fft.fftn(shifted, dim=(-3, -2, -1))
    diffraction = torch.fft.fftshift(diffraction, dim=(-3, -2, -1)).abs()
    return diffraction.float(), support.float(), phase


def per_sample_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compare non-negative feature magnitudes after per-channel RMS alignment."""

    prediction_rms = prediction.square().mean(dim=(1, 2, 3, 4)).clamp_min(EPS).sqrt()
    target_rms = target.square().mean(dim=(1, 2, 3, 4)).clamp_min(EPS).sqrt()
    prediction = spatial_rms_normalize(prediction)
    target = spatial_rms_normalize(target)
    difference = prediction - target
    reduce_dims = (1, 2, 3, 4)

    pred_flat = prediction.flatten(start_dim=1)
    target_flat = target.flatten(start_dim=1)
    pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)

    return {
        "normalized_mae": difference.abs().mean(dim=reduce_dims),
        "normalized_rmse": difference.square().mean(dim=reduce_dims).sqrt(),
        "relative_l1": difference.abs().sum(dim=reduce_dims)
        / target.abs().sum(dim=reduce_dims).clamp_min(EPS),
        "relative_l2": difference.square().sum(dim=reduce_dims).sqrt()
        / target.square().sum(dim=reduce_dims).sqrt().clamp_min(EPS),
        "pearson": (pred_centered * target_centered).sum(dim=1)
        / (
            pred_centered.square().sum(dim=1).sqrt()
            * target_centered.square().sum(dim=1).sqrt()
        ).clamp_min(EPS),
        "cosine": (pred_flat * target_flat).sum(dim=1)
        / (
            pred_flat.square().sum(dim=1).sqrt()
            * target_flat.square().sum(dim=1).sqrt()
        ).clamp_min(EPS),
        "raw_rms_ratio": prediction_rms / target_rms,
    }


def tensor_values(tensor: torch.Tensor) -> list[float]:
    """Move a batch of scalar values to ordinary Python floats."""

    return [float(value) for value in tensor.detach().cpu().tolist()]


def analyze_level(
    level: FeatureLevel,
    features: dict[str, torch.Tensor],
    sample_names: list[str],
    support_threshold: float,
) -> list[dict[str, object]]:
    """Produce long-form rows for one spatial scale and its controls."""

    encoder = features[f"{level.name}.encoder"]
    amplitude = features[f"{level.name}.amplitude"]
    phase = features[f"{level.name}.phase"]
    if encoder.shape[-3:] != amplitude.shape[-3:] or encoder.shape[-3:] != phase.shape[-3:]:
        raise RuntimeError(
            f"Spatial mismatch at {level.name}: encoder={tuple(encoder.shape)}, "
            f"amplitude={tuple(amplitude.shape)}, phase={tuple(phase.shape)}"
        )

    target = encoder.abs()
    batch_size = encoder.shape[0]
    diagnostic_tensors = {
        "encoder_negative_fraction": (encoder < 0).float().mean(dim=(1, 2, 3, 4)),
        "amplitude_negative_fraction": (amplitude < 0).float().mean(dim=(1, 2, 3, 4)),
        "phase_outside_pi_fraction": (phase.abs() > math.pi).float().mean(dim=(1, 2, 3, 4)),
    }
    diagnostics = {name: tensor_values(value) for name, value in diagnostic_tensors.items()}
    rows: list[dict[str, object]] = []

    for variant in VARIANTS:
        prediction, support, _mapped_phase = latent_forward_model(
            amplitude,
            phase,
            target_channels=encoder.shape[1],
            support_threshold=support_threshold,
            variant=variant,
        )
        metrics = {
            name: tensor_values(value)
            for name, value in per_sample_metrics(prediction, target).items()
        }
        support_fraction = tensor_values(support.mean(dim=(1, 2, 3, 4)))

        for index in range(batch_size):
            row: dict[str, object] = {
                "sample": sample_names[index],
                "level": level.name,
                "spatial_size": encoder.shape[-1],
                "encoder_channels": encoder.shape[1],
                "amplitude_channels": amplitude.shape[1],
                "phase_channels": phase.shape[1],
                "variant": variant,
                "support_fraction": support_fraction[index],
            }
            row.update({name: values[index] for name, values in diagnostics.items()})
            row.update({name: values[index] for name, values in metrics.items()})
            rows.append(row)
    return rows


def scalar_stats(values: list[float]) -> dict[str, float]:
    """Summarize sample-level values without assuming a normal distribution."""

    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
    }


def summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate each scale/variant and quantify paired-control advantages."""

    summary: dict[str, object] = {"levels": {}}
    for level in FEATURE_LEVELS:
        level_rows = [row for row in rows if row["level"] == level.name]
        level_summary: dict[str, object] = {"variants": {}, "paired_advantage": {}}
        for variant in VARIANTS:
            variant_rows = [row for row in level_rows if row["variant"] == variant]
            level_summary["variants"][variant] = {
                metric: scalar_stats([float(row[metric]) for row in variant_rows])
                for metric in METRIC_NAMES
            }

        paired_by_sample = {
            str(row["sample"]): row
            for row in level_rows
            if row["variant"] == "paired"
        }
        for control in VARIANTS[1:]:
            control_by_sample = {
                str(row["sample"]): row
                for row in level_rows
                if row["variant"] == control
            }
            advantages: dict[str, object] = {}
            for metric in METRIC_NAMES[:-1]:
                values = []
                for sample, paired in paired_by_sample.items():
                    control_row = control_by_sample[sample]
                    if metric in ERROR_METRICS:
                        value = float(control_row[metric]) - float(paired[metric])
                    else:
                        value = float(paired[metric]) - float(control_row[metric])
                    values.append(value)
                advantages[metric] = scalar_stats(values)
            level_summary["paired_advantage"][f"vs_{control}"] = advantages

        paired_rows = [row for row in level_rows if row["variant"] == "paired"]
        level_summary["feature_diagnostics"] = {
            metric: scalar_stats([float(row[metric]) for row in paired_rows])
            for metric in (
                "support_fraction",
                "encoder_negative_fraction",
                "amplitude_negative_fraction",
                "phase_outside_pi_fraction",
            )
        }
        summary["levels"][level.name] = level_summary
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write all per-sample metrics in a plot-friendly long table."""

    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: dict[str, object]) -> None:
    """Write a compact report whose claims stay within the experiment's scope."""

    lines = [
        "# AutoPhaseNN intermediate-feature physics check",
        "",
        "This experiment tests an imposed latent-space analogy, not a physical law. "
        "Intermediate channels are learned features without guaranteed amplitude, phase, "
        "or reciprocal-space semantics.",
        "",
        "All comparison tensors are normalized to unit spatial RMS per channel. Therefore "
        "the error metrics measure shape consistency, while `raw_rms_ratio` records the "
        "discarded scale mismatch.",
        "",
        "## Mean results",
        "",
        "| Level | Variant | Normalized MAE | Relative L2 | Pearson | Cosine |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for level in FEATURE_LEVELS:
        level_summary = summary["analysis"]["levels"][level.name]
        for variant in VARIANTS:
            metrics = level_summary["variants"][variant]
            lines.append(
                f"| {level.name} | {variant} | "
                f"{metrics['normalized_mae']['mean']:.6g} | "
                f"{metrics['relative_l2']['mean']:.6g} | "
                f"{metrics['pearson']['mean']:.6g} | "
                f"{metrics['cosine']['mean']:.6g} |"
            )

    lines.extend(
        [
            "",
            "## Paired advantage over controls",
            "",
            "Positive values favor the paired amplitude/phase features. For errors this is "
            "`control - paired`; for correlation this is `paired - control`.",
            "",
            "| Level | Control | MAE advantage | Relative-L2 advantage | Pearson advantage |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for level in FEATURE_LEVELS:
        comparisons = summary["analysis"]["levels"][level.name]["paired_advantage"]
        for control, metrics in comparisons.items():
            lines.append(
                f"| {level.name} | {control} | "
                f"{metrics['normalized_mae']['mean']:.6g} | "
                f"{metrics['relative_l2']['mean']:.6g} | "
                f"{metrics['pearson']['mean']:.6g} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A lower paired error than both controls is evidence that this particular "
            "deterministic latent mapping captures a statistical relation. It does not prove "
            "that each hidden channel is a physical amplitude or phase. Similar paired and "
            "control results argue against using this mapping directly as a loss.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse the isolated experiment configuration."""

    parser = argparse.ArgumentParser(
        description="Measure baseline AutoPhaseNN intermediate-feature physics consistency."
    )
    parser.add_argument("--checkpoint", required=True, help="Baseline checkpoint path.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--data-diff", default="val_diff.npy")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=0, help="Zero evaluates all samples.")
    parser.add_argument("--shape", type=int, default=64)
    parser.add_argument("--dtype-diff", default="float32")
    parser.add_argument("--scale-i", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--non-strict-checkpoint", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Reject ambiguous or unsupported experiment settings early."""

    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive.")
    if args.limit < 0:
        raise ValueError("--limit cannot be negative.")
    if args.shape != 64:
        raise ValueError("This baseline layer correspondence is defined for --shape 64.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1].")


def resolve_output_dir(value: str) -> Path:
    """Keep generated artifacts inside this experiment unless overridden."""

    if value:
        return Path(value).expanduser().resolve()
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    return EXPERIMENT_ROOT / "outputs" / run_name


def run(args: argparse.Namespace) -> Path:
    """Execute inference-only feature capture and write all scalar evidence."""

    validate_args(args)
    output_dir = resolve_output_dir(args.output_dir)
    configure_logging(output_dir, args.log_level)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    diffraction_path = Path(args.data_dir).expanduser().resolve() / args.data_diff
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not diffraction_path.is_file():
        raise FileNotFoundError(f"Diffraction data not found: {diffraction_path}")

    evaluated_samples = min(args.num_samples, args.limit or args.num_samples)
    LOGGER.info("Experiment: baseline intermediate-feature physics consistency")
    LOGGER.info("Device: %s", device)
    LOGGER.info("Checkpoint: %s", checkpoint_path)
    LOGGER.info("Diffraction data: %s", diffraction_path)
    LOGGER.info("Samples: %d / %d", evaluated_samples, args.num_samples)

    dataset = AutoPhaseDataset(
        diffraction_path,
        real_path=None,
        num_samples=args.num_samples,
        shape_diff=(args.shape,) * 3,
        dtype_diff=args.dtype_diff,
        scale_i=args.scale_i,
        shuffle=False,
    )
    subset = Subset(dataset, range(evaluated_samples))
    loader = DataLoader(
        subset,
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
        map_location="cpu",
    )
    model.eval()
    checkpoint_epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    del checkpoint
    LOGGER.info("Checkpoint epoch: %s", checkpoint_epoch)

    rows: list[dict[str, object]] = []
    capture = BaselineFeatureCapture(model)
    try:
        with torch.inference_mode():
            for batch in tqdm(loader, desc="Intermediate physics", unit="batch"):
                capture.clear()
                diffraction = batch["diff"].to(device, non_blocking=True)
                model(diffraction)
                features = capture.snapshot()
                sample_names = [str(name) for name in batch["name"]]
                for level in FEATURE_LEVELS:
                    rows.extend(
                        analyze_level(
                            level,
                            features,
                            sample_names,
                            support_threshold=args.threshold,
                        )
                    )
    finally:
        capture.close()

    analysis = summarize_rows(rows)
    summary = {
        "experiment": "autophasenn_intermediate_physics",
        "model": "TFCompatibleAutoPhaseNN (baseline)",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "diffraction_data": str(diffraction_path),
        "num_samples": evaluated_samples,
        "support_threshold": args.threshold,
        "normalization": "unit spatial RMS per sample and channel",
        "channel_mapping": (
            "identity-width RMS for amplitude; identity-width mean for phase; "
            "latent_8 phase channels repeated from 128 to 256"
        ),
        "phase_mapping": "pi * tanh(feature)",
        "target_mapping": "absolute value of matched encoder feature",
        "feature_levels": [
            {
                "name": level.name,
                "spatial_size": level.spatial_size,
                "encoder_bn": level.encoder_bn,
                "amplitude_bn": level.amplitude_bn,
                "phase_bn": level.phase_bn,
            }
            for level in FEATURE_LEVELS
        ],
        "analysis": analysis,
    }
    write_csv(output_dir / "per_sample_metrics.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "resolved_args.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(output_dir / "report.md", summary)
    LOGGER.info("Completed. Report: %s", (output_dir / "report.md").resolve())
    return output_dir


def main() -> None:
    """CLI entry point."""

    run(parse_args())


if __name__ == "__main__":
    main()
