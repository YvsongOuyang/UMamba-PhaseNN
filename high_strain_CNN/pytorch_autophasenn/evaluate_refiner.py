"""Evaluate the frozen phase U-Net plus real-space MoMamba cascade."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .author_data import AuthorNPZPhaseDataset, initialize_data_worker
from .data import (
    AutoPhaseNNPhaseDataset,
    build_autophasenn_refinement_dataset,
    reciprocal_phase_from_realspace,
)
from .losses import phase_retrieval_wca_components
from .management import DEFAULT_DATA_CONFIG, runtime_manifest
from .momamba_refiner import (
    HighStrainMoMambaCascade,
    ambiguity_aware_component_mae,
    diffraction_modulus_mae,
)
from .reconstruction import reciprocal_field_from_realspace
from .train_refiner import build_cascade, checkpoint_state


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = (
    PROJECT_DIR / "artifacts" / "evaluations" / "pytorch_realspace_momamba"
)
DEFAULT_AUTHOR_DATA_DIR = "/data_ssd/oyys/high_strain_cnn/dataset"
LOGGER = logging.getLogger("high_strain.evaluate_refiner")

LOWER_IS_BETTER = {
    "complex_nrmse",
    "object_mae",
    "real_mae",
    "imag_mae",
    "fourier_modulus_mae",
    "phase_wca",
}
HIGHER_IS_BETTER = {"amplitude_psnr_db", "support_iou", "support_dice"}
NEAR_ONE_IS_BETTER = {"support_volume_ratio"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--dataset-format",
        choices=("author_npz", "autophasenn_memmap"),
        default="author_npz",
    )
    parser.add_argument("--data-config", default=str(DEFAULT_DATA_CONFIG))
    parser.add_argument(
        "--data-dir",
        default="",
        help=(
            "Dataset root. Empty uses the author-data default or the root from "
            "--data-config, according to --dataset-format."
        ),
    )
    parser.add_argument("--data-diff", default="")
    parser.add_argument("--data-real", default="")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Evaluate a manifest-order prefix; zero uses the complete split.",
    )
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--support-threshold",
        type=float,
        default=None,
        help=(
            "Threshold on unit-maximum predicted amplitude and, for AutoPhaseNN, "
            "stored target amplitude. Defaults to 0.3 for author NPZ and 0.1 for "
            "AutoPhaseNN."
        ),
    )
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args()
    if args.support_threshold is None:
        args.support_threshold = (
            0.1 if args.dataset_format == "autophasenn_memmap" else 0.3
        )
    if args.num_samples < 0:
        parser.error("--num-samples cannot be negative.")
    if min(args.batch_size, args.prefetch_factor, args.print_freq) < 1:
        parser.error("Batch size, prefetch factor, and print frequency must be positive.")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative.")
    if not math.isfinite(args.support_threshold) or args.support_threshold < 0:
        parser.error("--support-threshold must be finite and nonnegative.")
    return args


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def resolve_output_dir(args: argparse.Namespace, checkpoint_path: Path) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    run_name = checkpoint_path.parent.name
    dataset_label = (
        "autophasenn" if args.dataset_format == "autophasenn_memmap" else "author"
    )
    return (DEFAULT_OUTPUT_ROOT / f"{run_name}_{dataset_label}_{args.split}").resolve()


def configure_logging(output_dir: Path, level: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.handlers.clear()
    LOGGER.setLevel(getattr(logging, level))
    LOGGER.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "console.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(stream)
    LOGGER.addHandler(file_handler)


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[HighStrainMoMambaCascade, dict, argparse.Namespace, str]:
    checkpoint = checkpoint_state(checkpoint_path, torch.device("cpu"))
    training_config = checkpoint.get("args")
    if not isinstance(training_config, dict):
        raise ValueError("Refiner checkpoint does not contain its training args.")
    model_args = argparse.Namespace(**training_config)
    model, phase_variant = build_cascade(model_args, device, checkpoint)
    model.eval()
    return model, checkpoint, model_args, phase_variant


def build_author_dataset(
    args: argparse.Namespace,
    model_args: argparse.Namespace,
) -> tuple[AuthorNPZPhaseDataset, dict[str, object]]:
    root = args.data_dir or DEFAULT_AUTHOR_DATA_DIR
    dataset = AuthorNPZPhaseDataset(
        root,
        args.split,
        num_samples=args.num_samples or None,
        shape=(int(model_args.shape),) * 3,
        input_log_data=bool(model_args.input_log_data),
        min_oversampling=model_args.author_min_oversampling,
        return_refinement_targets=True,
    )
    return dataset, dataset.manifest


def build_autophasenn_dataset(
    args: argparse.Namespace,
    model_args: argparse.Namespace,
) -> tuple[AutoPhaseNNPhaseDataset, dict[str, object]]:
    model_shape = (int(model_args.shape),) * 3
    dataset, manifest = build_autophasenn_refinement_dataset(
        data_config=args.data_config,
        data_dir=args.data_dir or None,
        split=args.split,
        num_samples=args.num_samples or None,
        shape=model_shape,
        input_log_data=bool(model_args.input_log_data),
        diffraction_path=args.data_diff or None,
        realspace_path=args.data_real or None,
    )
    manifest["refinement_targets"] = {
        "realspace": "stored complex object",
        "reciprocal_phase": (
            "FFT of realspace after amplitude-COM translation canonicalization"
        ),
        "support": "stored amplitude >= support_threshold",
    }
    return dataset, manifest


def build_dataset(
    args: argparse.Namespace,
    model_args: argparse.Namespace,
) -> tuple[Dataset, dict[str, object]]:
    if args.dataset_format == "autophasenn_memmap":
        return build_autophasenn_dataset(args, model_args)
    return build_author_dataset(args, model_args)


def build_loader(
    dataset: Dataset,
    args: argparse.Namespace,
    device: torch.device,
) -> DataLoader:
    kwargs: dict[str, object] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers:
        kwargs.update(
            prefetch_factor=args.prefetch_factor,
            worker_init_fn=initialize_data_worker,
            multiprocessing_context="spawn",
        )
    return DataLoader(dataset, **kwargs)


def prepare_targets(
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = batch["realspace"].to(device, non_blocking=True)
    if args.dataset_format == "author_npz":
        support = batch["support"].to(device, non_blocking=True)
        target_phase = batch["target_phase"].to(
            device,
            non_blocking=True,
        ).float()
    else:
        support = target.abs() >= args.support_threshold
        target_phase = reciprocal_phase_from_realspace(target).float()
    return target, support, target_phase


def reciprocal_phase(realspace: torch.Tensor) -> torch.Tensor:
    phase = torch.angle(reciprocal_field_from_realspace(realspace))
    center = tuple(size // 2 for size in phase.shape[-3:])
    center_phase = phase[(slice(None), slice(None)) + center]
    return phase - center_phase[..., None, None, None]


def phase_wca(
    realspace: torch.Tensor,
    target_phase: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    direct, inverted = phase_retrieval_wca_components(
        reciprocal_phase(realspace),
        target_phase,
        weights,
    )
    return torch.minimum(direct, inverted).mean()


def support_metrics(
    realspace: torch.Tensor,
    selected_support: torch.Tensor,
    threshold: float,
) -> dict[str, torch.Tensor]:
    amplitude = realspace.abs()
    eps = torch.finfo(amplitude.dtype).eps
    amplitude = amplitude / amplitude.amax(
        dim=(-3, -2, -1), keepdim=True
    ).clamp_min(eps)
    prediction = amplitude >= threshold
    target = selected_support.bool()
    spatial_dims = (-3, -2, -1)
    intersection = (prediction & target).sum(dim=spatial_dims).float()
    prediction_count = prediction.sum(dim=spatial_dims).float()
    target_count = target.sum(dim=spatial_dims).float().clamp_min(1.0)
    union = (prediction | target).sum(dim=spatial_dims).float().clamp_min(1.0)
    return {
        "support_iou": (intersection / union).mean(),
        "support_dice": (
            2.0 * intersection / (prediction_count + target_count).clamp_min(1.0)
        ).mean(),
        "support_volume_ratio": (prediction_count / target_count).mean(),
    }


def ambiguity_aligned_reconstruction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    selected_support: torch.Tensor,
    selected_twin: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return scale-, phase-, and twin-aligned complex reconstruction metrics.

    Complex NRMSE uses the same per-object RMS normalization, support-weighted
    global phase alignment, and twin selection as the component-MAE objective.
    Amplitude PSNR compares independently unit-maximum-normalized full volumes,
    so its data range is one and its unit is decibels.
    """

    if prediction.ndim == 4:
        prediction = prediction[:, None]
    if target.ndim == 4:
        target = target[:, None]
    if selected_support.ndim == 4:
        selected_support = selected_support[:, None]
    if prediction.shape != target.shape or selected_support.shape != target.shape:
        raise ValueError("Prediction, target, and support shapes must match.")
    if not torch.is_complex(prediction) or not torch.is_complex(target):
        raise ValueError("Prediction and target must be complex tensors.")
    if selected_twin.ndim != 1 or selected_twin.shape[0] != target.shape[0]:
        raise ValueError("selected_twin must contain one decision per sample.")

    twin_target = torch.conj(
        torch.roll(
            torch.flip(target, dims=(-3, -2, -1)),
            shifts=(1, 1, 1),
            dims=(-3, -2, -1),
        )
    )
    selected_target = torch.where(
        selected_twin[:, None, None, None, None],
        twin_target,
        target,
    )
    spatial_dims = (-3, -2, -1)
    real_dtype = prediction.real.dtype
    eps = torch.finfo(real_dtype).eps
    prediction = prediction / prediction.abs().square().mean(
        dim=spatial_dims,
        keepdim=True,
    ).sqrt().clamp_min(eps)
    selected_target = selected_target / selected_target.abs().square().mean(
        dim=spatial_dims,
        keepdim=True,
    ).sqrt().clamp_min(eps)

    mask = selected_support.to(dtype=real_dtype)
    correlation = (selected_target.conj() * prediction * mask).sum(
        dim=spatial_dims,
        keepdim=True,
    )
    phase_offset = torch.angle(correlation)
    prediction = prediction * torch.complex(
        torch.cos(-phase_offset),
        torch.sin(-phase_offset),
    )

    squared_error = (prediction - selected_target).abs().square().sum(
        dim=spatial_dims,
    )
    target_energy = selected_target.abs().square().sum(dim=spatial_dims).clamp_min(
        eps
    )
    complex_nrmse = torch.sqrt(squared_error / target_energy)[:, 0]

    prediction_amplitude = prediction.abs()
    target_amplitude = selected_target.abs()
    prediction_amplitude = prediction_amplitude / prediction_amplitude.amax(
        dim=spatial_dims,
        keepdim=True,
    ).clamp_min(eps)
    target_amplitude = target_amplitude / target_amplitude.amax(
        dim=spatial_dims,
        keepdim=True,
    ).clamp_min(eps)
    amplitude_mse = (prediction_amplitude - target_amplitude).square().mean(
        dim=spatial_dims,
    )[:, 0]
    amplitude_psnr = 10.0 * torch.log10(
        amplitude_mse.clamp_min(torch.finfo(real_dtype).tiny).reciprocal()
    )
    return {
        "complex_nrmse": complex_nrmse.mean(),
        "amplitude_psnr_db": amplitude_psnr.mean(),
    }


def stage_metrics(
    realspace: torch.Tensor,
    target: torch.Tensor,
    support: torch.Tensor,
    modulus: torch.Tensor,
    target_phase: torch.Tensor,
    weights: torch.Tensor,
    model_args: argparse.Namespace,
    support_threshold: float,
) -> dict[str, torch.Tensor]:
    object_loss = ambiguity_aware_component_mae(
        realspace,
        target,
        support,
        outside_weight=float(model_args.outside_support_weight),
        loss_scope=str(model_args.object_loss_scope),
    )
    values = {
        "object_mae": object_loss.loss,
        "real_mae": object_loss.real_mae,
        "imag_mae": object_loss.imag_mae,
        "fourier_modulus_mae": diffraction_modulus_mae(
            realspace,
            modulus,
            object_loss.selected_support,
        ),
        "phase_wca": phase_wca(realspace, target_phase, weights),
        "twin_fraction": object_loss.twin_fraction,
    }
    values.update(
        ambiguity_aligned_reconstruction_metrics(
            realspace,
            target,
            object_loss.selected_support,
            object_loss.selected_twin,
        )
    )
    values.update(
        support_metrics(realspace, object_loss.selected_support, support_threshold)
    )
    return values


def relative_improvement(metric: str, initial: float, refined: float) -> float | None:
    if metric in LOWER_IS_BETTER:
        baseline = abs(initial)
        numerator = initial - refined
    elif metric in HIGHER_IS_BETTER:
        baseline = abs(initial)
        numerator = refined - initial
    elif metric in NEAR_ONE_IS_BETTER:
        initial_error = abs(initial - 1.0)
        refined_error = abs(refined - 1.0)
        baseline = initial_error
        numerator = initial_error - refined_error
    else:
        return None
    if baseline <= 1e-12:
        return None
    return 100.0 * numerator / baseline


def evaluate(
    model: HighStrainMoMambaCascade,
    loader: DataLoader,
    args: argparse.Namespace,
    model_args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    totals: dict[str, dict[str, float]] = {"initial": {}, "refined": {}}
    samples = 0
    started = time.monotonic()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            model_input = batch["input"].to(device, non_blocking=True).float()
            modulus = batch["diffraction"].to(device, non_blocking=True).float()
            target, support, target_phase = prepare_targets(batch, args, device)
            output = model(model_input, modulus)
            current = {
                "initial": stage_metrics(
                    output.initial_object,
                    target,
                    support,
                    modulus,
                    target_phase,
                    model_input,
                    model_args,
                    args.support_threshold,
                ),
                "refined": stage_metrics(
                    output.refined_object,
                    target,
                    support,
                    modulus,
                    target_phase,
                    model_input,
                    model_args,
                    args.support_threshold,
                ),
            }
            batch_size = model_input.shape[0]
            for stage, values in current.items():
                for name, value in values.items():
                    totals[stage][name] = totals[stage].get(name, 0.0) + (
                        float(value.detach()) * batch_size
                    )
            samples += batch_size
            if batch_index % args.print_freq == 0 or batch_index == len(loader):
                elapsed = time.monotonic() - started
                remaining = elapsed / batch_index * (len(loader) - batch_index)
                LOGGER.info(
                    "batch=%05d/%05d | samples=%d/%d | initial=%.5e | "
                    "refined=%.5e | elapsed=%.1fs | eta=%.1fs",
                    batch_index,
                    len(loader),
                    samples,
                    len(loader.dataset),
                    float(current["initial"]["object_mae"]),
                    float(current["refined"]["object_mae"]),
                    elapsed,
                    remaining,
                )

    means = {
        stage: {name: value / samples for name, value in values.items()}
        for stage, values in totals.items()
    }
    metric_names = sorted(set(means["initial"]) & set(means["refined"]))
    comparison = {
        name: {
            "initial": means["initial"][name],
            "refined": means["refined"][name],
            "refined_minus_initial": means["refined"][name]
            - means["initial"][name],
            "improvement_percent": relative_improvement(
                name,
                means["initial"][name],
                means["refined"][name],
            ),
        }
        for name in metric_names
    }
    return {
        "num_samples": samples,
        "num_batches": len(loader),
        "elapsed_seconds": time.monotonic() - started,
        "metrics": means,
        "comparison": comparison,
    }


def write_csv(path: Path, comparison: dict[str, dict[str, float | None]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "metric",
                "initial",
                "refined",
                "refined_minus_initial",
                "improvement_percent",
            ),
        )
        writer.writeheader()
        for metric, values in comparison.items():
            writer.writerow({"metric": metric, **values})


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    output_dir = resolve_output_dir(args, checkpoint_path)
    configure_logging(output_dir, args.log_level)
    device = choose_device(args.device)
    model, checkpoint, model_args, phase_variant = load_model(checkpoint_path, device)
    dataset, data_manifest = build_dataset(args, model_args)
    loader = build_loader(dataset, args, device)
    LOGGER.info(
        "Checkpoint=%s | epoch=%s | phase=%s (frozen) | data=%s/%s | "
        "samples=%d | batch=%d | device=%s",
        checkpoint_path,
        checkpoint.get("epoch"),
        phase_variant,
        args.dataset_format,
        args.split,
        len(dataset),
        args.batch_size,
        device,
    )
    result = evaluate(model, loader, args, model_args, device)
    report = {
        "schema_version": 1,
        "run": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "phase_model_variant": phase_variant,
            "dataset_format": args.dataset_format,
            "split": args.split,
            "support_threshold": args.support_threshold,
            "device": str(device),
        },
        "runtime": runtime_manifest(device),
        "data": data_manifest,
        **result,
        "metric_notes": {
            "object_mae": (
                "Training-aligned RMS-normalized real MAE plus imaginary MAE; "
                "lower is better."
            ),
            "fourier_modulus_mae": (
                "Unit-maximum Fourier-modulus MAE after applying the selected "
                "ground-truth support; lower is better."
            ),
            "phase_wca": (
                "Symmetry-aware WCA after Fourier-transforming each real-space "
                "object; lower is better."
            ),
            "complex_nrmse": (
                "Full-volume complex NRMSE after per-object RMS normalization, "
                "support-weighted global-phase alignment, and the component-MAE "
                "twin selection; lower is better."
            ),
            "amplitude_psnr_db": (
                "Full-volume amplitude PSNR in dB after independently normalizing "
                "the selected target and prediction amplitudes to unit maximum; "
                "higher is better."
            ),
            "support": (
                "Predicted support is unit-maximum amplitude >= support_threshold; "
                "AutoPhaseNN target support uses stored target amplitude >= the same "
                "threshold, while author NPZ uses its explicit stored support. "
                "IoU/Dice are higher-is-better and volume ratio is best near one."
            ),
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_csv(output_dir / "comparison.csv", report["comparison"])
    initial = report["metrics"]["initial"]
    refined = report["metrics"]["refined"]
    LOGGER.info(
        "Object MAE %.6g -> %.6g | WCA %.6g -> %.6g | "
        "complex NRMSE %.6g -> %.6g | amplitude PSNR %.4f -> %.4f dB | "
        "Fourier MAE %.6g -> %.6g | support IoU %.6g -> %.6g",
        initial["object_mae"],
        refined["object_mae"],
        initial["phase_wca"],
        refined["phase_wca"],
        initial["complex_nrmse"],
        refined["complex_nrmse"],
        initial["amplitude_psnr_db"],
        refined["amplitude_psnr_db"],
        initial["fourier_modulus_mae"],
        refined["fourier_modulus_mae"],
        initial["support_iou"],
        refined["support_iou"],
    )
    LOGGER.info("Results: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
