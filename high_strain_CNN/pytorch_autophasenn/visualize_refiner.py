"""Visualize the frozen phase U-Net and real-space MoMamba cascade."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch

from .data import (
    build_autophasenn_refinement_dataset,
    reciprocal_phase_from_realspace,
)
from .evaluate_refiner import choose_device, load_model
from .management import DEFAULT_DATA_CONFIG
from .momamba_refiner import ambiguity_aware_component_mae
from .reconstruction import reciprocal_field_from_realspace


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from autophasenn_training_pipeline.evaluate import official_post_process  # noqa: E402
from autophasenn_training_pipeline.visualize_postprocessed import (  # noqa: E402
    plot_five_panel_volume,
    wrap_phase,
)

from .visualize import (  # noqa: E402
    masked_phase,
    normalized_modulus,
    plot_image_grid,
)


LOGGER = logging.getLogger("high_strain.visualize_refiner")
DEFAULT_OUTPUT_ROOT = (
    PROJECT_DIR / "artifacts" / "visualizations" / "pytorch_realspace_momamba"
)


def configure_logging(output_dir: Path, level: str) -> None:
    """Log visualization progress to both the terminal and its artifact folder."""

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


def parse_args() -> argparse.Namespace:
    """Parse data, checkpoint, and rendering options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-config", default=str(DEFAULT_DATA_CONFIG))
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--data-diff", default="")
    parser.add_argument("--data-real", default="")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="+",
        default=(0, 1000, 2000),
        help="Zero-based indices within the selected split.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--support-threshold", type=float, default=0.3)
    parser.add_argument("--reciprocal-phase-threshold", type=float, default=0.02)
    parser.add_argument("--reciprocal-surface-level", type=float, default=0.02)
    parser.add_argument("--diffraction-difference-threshold", type=float, default=1e-6)
    parser.add_argument("--slice-axis", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument(
        "--slice-index",
        type=int,
        default=-1,
        help="Slice along --slice-axis; -1 uses the volume center.",
    )
    parser.add_argument("--amplitude-error-level", type=float, default=0.02)
    parser.add_argument("--max-volume-points", type=int, default=12000)
    parser.add_argument("--volume-point-size", type=float, default=2.0)
    parser.add_argument("--volume-alpha", type=float, default=0.5)
    parser.add_argument("--view-elevation", type=float, default=25.0)
    parser.add_argument("--view-azimuth", type=float, default=35.0)
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args()
    if not args.sample_indices or min(args.sample_indices) < 0:
        parser.error("--sample-indices must contain nonnegative indices.")
    if len(set(args.sample_indices)) != len(args.sample_indices):
        parser.error("--sample-indices cannot contain duplicates.")
    if not math.isfinite(args.support_threshold) or not 0 <= args.support_threshold < 1:
        parser.error("--support-threshold must be finite and lie in [0, 1).")
    if args.slice_index < -1:
        parser.error("--slice-index must be -1 or nonnegative.")
    if args.amplitude_error_level <= 0:
        parser.error("--amplitude-error-level must be positive.")
    if args.reciprocal_phase_threshold < 0:
        parser.error("--reciprocal-phase-threshold must be nonnegative.")
    if args.reciprocal_surface_level <= 0:
        parser.error("--reciprocal-surface-level must be positive.")
    if args.diffraction_difference_threshold <= 0:
        parser.error("--diffraction-difference-threshold must be positive.")
    if args.max_volume_points < 1 or args.volume_point_size <= 0:
        parser.error("3D point count and point size must be positive.")
    if not 0 < args.volume_alpha <= 1:
        parser.error("--volume-alpha must lie in (0, 1].")
    return args


def twin_transform(volume: torch.Tensor) -> torch.Tensor:
    """Apply the centered-grid BCDI conjugate-inversion ambiguity."""

    return torch.conj(
        torch.roll(
            torch.flip(volume, dims=(-3, -2, -1)),
            shifts=(1, 1, 1),
            dims=(-3, -2, -1),
        )
    )


def align_reconstruction_for_display(
    prediction: torch.Tensor,
    target: torch.Tensor,
    support: torch.Tensor,
) -> tuple[torch.Tensor, bool, float]:
    """Align twin orientation and global phase while preserving amplitude scale."""

    if prediction.ndim == 4:
        prediction = prediction[:, None]
    if target.ndim == 4:
        target = target[:, None]
    if support.ndim == 4:
        support = support[:, None]
    object_loss = ambiguity_aware_component_mae(prediction, target, support)
    twin_selected = bool(object_loss.selected_twin[0].item())
    aligned = twin_transform(prediction) if twin_selected else prediction
    mask = support.to(dtype=aligned.real.dtype)
    correlation = (target.conj() * aligned * mask).sum(dim=(-3, -2, -1), keepdim=True)
    phase_offset = torch.angle(correlation)
    aligned = aligned * torch.polar(torch.ones_like(phase_offset), -phase_offset)
    return aligned, twin_selected, float(object_loss.loss.item())


def post_process_object(
    object_volume: torch.Tensor,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize and apply the established AutoPhaseNN display processing."""

    volume = object_volume.detach().cpu()[0, 0]
    amplitude = torch.abs(volume).numpy().astype(np.float32, copy=False)
    maximum = max(float(amplitude.max()), np.finfo(np.float32).eps)
    amplitude = amplitude / maximum
    phase = torch.angle(volume).numpy().astype(np.float32, copy=False)
    return official_post_process(amplitude, phase, threshold=threshold, unwrap=True)


def take_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    """Return one 2D slice from a 3D volume."""

    return np.take(volume, index, axis=axis)


def resolve_output_dir(args: argparse.Namespace, checkpoint_path: Path) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    return (
        DEFAULT_OUTPUT_ROOT / f"{checkpoint_path.parent.name}_{args.split}"
    ).resolve()


@torch.inference_mode()
def main() -> int:
    """Run selected samples through both stages and write comparison figures."""

    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = resolve_output_dir(args, checkpoint_path)
    configure_logging(output_dir, args.log_level)
    device = choose_device(args.device)
    model, checkpoint, model_args, phase_variant = load_model(checkpoint_path, device)

    shape = (int(model_args.shape),) * 3
    if args.slice_index == -1:
        args.slice_index = shape[args.slice_axis] // 2
    if args.slice_index >= shape[args.slice_axis]:
        raise ValueError("--slice-index lies outside the configured volume.")

    dataset, manifest = build_autophasenn_refinement_dataset(
        data_config=args.data_config,
        data_dir=args.data_dir or None,
        split=args.split,
        num_samples=None,
        shape=shape,
        input_log_data=bool(model_args.input_log_data),
        diffraction_path=args.data_diff or None,
        realspace_path=args.data_real or None,
    )
    if max(args.sample_indices) >= len(dataset):
        raise ValueError(
            f"Sample index {max(args.sample_indices)} exceeds {args.split} size "
            f"{len(dataset)}."
        )

    LOGGER.info("Checkpoint: %s", checkpoint_path)
    LOGGER.info("Device: %s | phase model: %s", device, phase_variant)
    LOGGER.info("Samples: %s", ", ".join(map(str, args.sample_indices)))

    slice_rows: list[list[np.ndarray]] = []
    amplitude_rows: list[
        list[tuple[np.ndarray | None, np.ndarray | None, str, float]]
    ] = []
    phase_rows: list[list[tuple[np.ndarray | None, np.ndarray | None, str, float]]] = []
    reciprocal_slice_rows: list[list[np.ndarray]] = []
    diffraction_modulus_rows: list[
        list[tuple[np.ndarray | None, np.ndarray | None, str, float]]
    ] = []
    diffraction_phase_rows: list[
        list[tuple[np.ndarray | None, np.ndarray | None, str, float]]
    ] = []
    names: list[str] = []
    sample_metadata: list[dict[str, object]] = []

    for position, index in enumerate(args.sample_indices, start=1):
        sample = dataset[index]
        model_input = sample["input"].unsqueeze(0).to(device).float()
        diffraction_modulus = sample["diffraction"].unsqueeze(0).to(device).float()
        target = sample["realspace"].unsqueeze(0).unsqueeze(0).to(device)
        target_amplitude = target.abs()
        target_support = target_amplitude >= args.support_threshold

        output = model(model_input, diffraction_modulus)
        initial, initial_twin, initial_mae = align_reconstruction_for_display(
            output.initial_object,
            target,
            target_support,
        )
        refined, refined_twin, refined_mae = align_reconstruction_for_display(
            output.refined_object,
            target,
            target_support,
        )

        measured_np = diffraction_modulus.detach().cpu().numpy()[0, 0]
        measured_norm = normalized_modulus(measured_np)
        measured_scale = max(float(measured_np.max()), np.finfo(np.float32).eps)
        initial_reciprocal = reciprocal_field_from_realspace(initial)
        refined_reciprocal = reciprocal_field_from_realspace(refined)
        initial_modulus = initial_reciprocal.abs().detach().cpu().numpy()[0, 0]
        refined_modulus = refined_reciprocal.abs().detach().cpu().numpy()[0, 0]
        initial_modulus_norm = initial_modulus / measured_scale
        refined_modulus_norm = refined_modulus / measured_scale
        initial_modulus_error = initial_modulus_norm - measured_norm
        refined_modulus_error = refined_modulus_norm - measured_norm

        target_reciprocal_phase = reciprocal_phase_from_realspace(target[:, 0])
        initial_reciprocal_phase = reciprocal_phase_from_realspace(initial[:, 0])
        refined_reciprocal_phase = reciprocal_phase_from_realspace(refined[:, 0])
        target_reciprocal_phase_np = target_reciprocal_phase.detach().cpu().numpy()[0]
        initial_reciprocal_phase_np = initial_reciprocal_phase.detach().cpu().numpy()[0]
        refined_reciprocal_phase_np = refined_reciprocal_phase.detach().cpu().numpy()[0]
        initial_phase_geometry = np.minimum(measured_norm, initial_modulus_norm)
        refined_phase_geometry = np.minimum(measured_norm, refined_modulus_norm)
        target_reciprocal_phase_display = masked_phase(
            target_reciprocal_phase_np,
            measured_norm,
            args.reciprocal_phase_threshold,
        )
        initial_reciprocal_phase_display = masked_phase(
            initial_reciprocal_phase_np,
            initial_phase_geometry,
            args.reciprocal_phase_threshold,
        )
        refined_reciprocal_phase_display = masked_phase(
            refined_reciprocal_phase_np,
            refined_phase_geometry,
            args.reciprocal_phase_threshold,
        )
        initial_reciprocal_phase_error = np.where(
            initial_phase_geometry > args.reciprocal_phase_threshold,
            wrap_phase(initial_reciprocal_phase_np - target_reciprocal_phase_np),
            np.nan,
        ).astype(np.float32)
        refined_reciprocal_phase_error = np.where(
            refined_phase_geometry > args.reciprocal_phase_threshold,
            wrap_phase(refined_reciprocal_phase_np - target_reciprocal_phase_np),
            np.nan,
        ).astype(np.float32)

        target_amp, target_phase = post_process_object(target, args.support_threshold)
        initial_amp, initial_phase = post_process_object(
            initial,
            args.support_threshold,
        )
        refined_amp, refined_phase = post_process_object(
            refined,
            args.support_threshold,
        )
        initial_amp_error = initial_amp - target_amp
        refined_amp_error = refined_amp - target_amp
        initial_phase_error = wrap_phase(initial_phase - target_phase)
        refined_phase_error = wrap_phase(refined_phase - target_phase)

        slice_rows.append(
            [
                take_slice(target_amp, args.slice_axis, args.slice_index),
                take_slice(initial_amp, args.slice_axis, args.slice_index),
                take_slice(refined_amp, args.slice_axis, args.slice_index),
                take_slice(initial_amp_error, args.slice_axis, args.slice_index),
                take_slice(refined_amp_error, args.slice_axis, args.slice_index),
                take_slice(wrap_phase(target_phase), args.slice_axis, args.slice_index),
                take_slice(
                    wrap_phase(initial_phase), args.slice_axis, args.slice_index
                ),
                take_slice(
                    wrap_phase(refined_phase), args.slice_axis, args.slice_index
                ),
                take_slice(initial_phase_error, args.slice_axis, args.slice_index),
                take_slice(refined_phase_error, args.slice_axis, args.slice_index),
            ]
        )
        amplitude_rows.append(
            [
                (target_amp, target_amp, "Target", args.support_threshold),
                (initial_amp, initial_amp, "U-Net initial", args.support_threshold),
                (refined_amp, refined_amp, "MoMamba refined", args.support_threshold),
                (
                    np.abs(initial_amp_error),
                    initial_amp_error,
                    "Initial - target",
                    args.amplitude_error_level,
                ),
                (
                    np.abs(refined_amp_error),
                    refined_amp_error,
                    "Refined - target",
                    args.amplitude_error_level,
                ),
            ]
        )
        phase_rows.append(
            [
                (
                    target_amp,
                    wrap_phase(target_phase),
                    "Target",
                    args.support_threshold,
                ),
                (
                    initial_amp,
                    wrap_phase(initial_phase),
                    "U-Net initial",
                    args.support_threshold,
                ),
                (
                    refined_amp,
                    wrap_phase(refined_phase),
                    "MoMamba refined",
                    args.support_threshold,
                ),
                (
                    np.minimum(target_amp, initial_amp),
                    initial_phase_error,
                    "Initial - target",
                    args.support_threshold,
                ),
                (
                    np.minimum(target_amp, refined_amp),
                    refined_phase_error,
                    "Refined - target",
                    args.support_threshold,
                ),
            ]
        )
        reciprocal_slice_rows.append(
            [
                take_slice(
                    np.log10(np.clip(measured_norm, 1e-6, None)),
                    args.slice_axis,
                    args.slice_index,
                ),
                take_slice(
                    np.log10(np.clip(initial_modulus_norm, 1e-6, None)),
                    args.slice_axis,
                    args.slice_index,
                ),
                take_slice(
                    np.log10(np.clip(refined_modulus_norm, 1e-6, None)),
                    args.slice_axis,
                    args.slice_index,
                ),
                take_slice(
                    initial_modulus_error,
                    args.slice_axis,
                    args.slice_index,
                ),
                take_slice(
                    refined_modulus_error,
                    args.slice_axis,
                    args.slice_index,
                ),
                take_slice(
                    target_reciprocal_phase_display,
                    args.slice_axis,
                    args.slice_index,
                ),
                take_slice(
                    initial_reciprocal_phase_display,
                    args.slice_axis,
                    args.slice_index,
                ),
                take_slice(
                    refined_reciprocal_phase_display,
                    args.slice_axis,
                    args.slice_index,
                ),
                take_slice(
                    initial_reciprocal_phase_error,
                    args.slice_axis,
                    args.slice_index,
                ),
                take_slice(
                    refined_reciprocal_phase_error,
                    args.slice_axis,
                    args.slice_index,
                ),
            ]
        )
        diffraction_modulus_rows.append(
            [
                (
                    measured_norm,
                    np.log10(np.clip(measured_norm, 1e-6, None)),
                    "Measured modulus",
                    args.reciprocal_surface_level,
                ),
                (
                    initial_modulus_norm,
                    np.log10(np.clip(initial_modulus_norm, 1e-6, None)),
                    "U-Net initial reprojection",
                    args.reciprocal_surface_level,
                ),
                (
                    refined_modulus_norm,
                    np.log10(np.clip(refined_modulus_norm, 1e-6, None)),
                    "MoMamba refined reprojection",
                    args.reciprocal_surface_level,
                ),
                (
                    np.abs(initial_modulus_error),
                    initial_modulus_error,
                    "Initial - measured",
                    args.diffraction_difference_threshold,
                ),
                (
                    np.abs(refined_modulus_error),
                    refined_modulus_error,
                    "Refined - measured",
                    args.diffraction_difference_threshold,
                ),
            ]
        )
        diffraction_phase_rows.append(
            [
                (
                    measured_norm,
                    target_reciprocal_phase_np,
                    "Target reciprocal phase",
                    args.reciprocal_phase_threshold,
                ),
                (
                    initial_modulus_norm,
                    initial_reciprocal_phase_np,
                    "U-Net initial reciprocal phase",
                    args.reciprocal_phase_threshold,
                ),
                (
                    refined_modulus_norm,
                    refined_reciprocal_phase_np,
                    "MoMamba refined reciprocal phase",
                    args.reciprocal_phase_threshold,
                ),
                (
                    initial_phase_geometry,
                    initial_reciprocal_phase_error,
                    "Initial - target",
                    args.reciprocal_phase_threshold,
                ),
                (
                    refined_phase_geometry,
                    refined_reciprocal_phase_error,
                    "Refined - target",
                    args.reciprocal_phase_threshold,
                ),
            ]
        )
        sample_name = str(sample["name"])
        names.append(sample_name)
        sample_metadata.append(
            {
                "index": index,
                "name": sample_name,
                "initial_twin_aligned": initial_twin,
                "refined_twin_aligned": refined_twin,
                "initial_object_mae": initial_mae,
                "refined_object_mae": refined_mae,
                "initial_diffraction_modulus_mae": float(
                    np.mean(np.abs(initial_modulus_error))
                ),
                "refined_diffraction_modulus_mae": float(
                    np.mean(np.abs(refined_modulus_error))
                ),
            }
        )
        LOGGER.info(
            "Rendered %d/%d | %s | object_mae %.6f -> %.6f",
            position,
            len(args.sample_indices),
            sample_name,
            initial_mae,
            refined_mae,
        )

    plot_image_grid(
        slice_rows,
        names,
        [
            "Target amplitude",
            "U-Net initial amplitude",
            "MoMamba refined amplitude",
            "Initial amplitude error",
            "Refined amplitude error",
            "Target phase",
            "U-Net initial phase",
            "MoMamba refined phase",
            "Initial phase error",
            "Refined phase error",
        ],
        [
            "viridis",
            "viridis",
            "viridis",
            "coolwarm",
            "coolwarm",
            "twilight",
            "twilight",
            "twilight",
            "coolwarm",
            "coolwarm",
        ],
        output_dir / "visualization_2d.png",
        "MoMamba real-space refinement: centered comparison",
        phase_rows={5, 6, 7, 8, 9},
        difference_rows={3, 4, 8, 9},
    )
    plot_five_panel_volume(
        amplitude_rows,
        names,
        output_dir / "visualization_amplitude_3d.png",
        "Real-space amplitude",
        "viridis",
        "coolwarm",
        "Amplitude",
        "Signed amplitude difference",
        args.max_volume_points,
        args.volume_point_size,
        args.volume_alpha,
        args.view_elevation,
        args.view_azimuth,
        absolute_limits=(0.0, 1.0),
        difference_limits=(-1.0, 1.0),
        absolute_zero_minimum=True,
    )
    plot_five_panel_volume(
        phase_rows,
        names,
        output_dir / "visualization_phase_3d.png",
        "Real-space phase on amplitude support",
        "twilight",
        "coolwarm",
        "Wrapped phase (rad)",
        "Wrapped phase difference (rad)",
        args.max_volume_points,
        args.volume_point_size,
        args.volume_alpha,
        args.view_elevation,
        args.view_azimuth,
        absolute_limits=(-float(np.pi), float(np.pi)),
        difference_limits=(-float(np.pi), float(np.pi)),
    )
    plot_image_grid(
        reciprocal_slice_rows,
        names,
        [
            "Measured modulus (log10 normalized)",
            "U-Net initial modulus (log10 normalized)",
            "MoMamba refined modulus (log10 normalized)",
            "Initial modulus error",
            "Refined modulus error",
            "Target reciprocal phase",
            "U-Net initial reciprocal phase",
            "MoMamba refined reciprocal phase",
            "Initial reciprocal phase error",
            "Refined reciprocal phase error",
        ],
        [
            "magma",
            "magma",
            "magma",
            "coolwarm",
            "coolwarm",
            "twilight",
            "twilight",
            "twilight",
            "coolwarm",
            "coolwarm",
        ],
        output_dir / "visualization_reciprocal_2d.png",
        "MoMamba reciprocal-space reconstruction",
        phase_rows={5, 6, 7, 8, 9},
        difference_rows={3, 4, 8, 9},
    )
    plot_five_panel_volume(
        diffraction_modulus_rows,
        names,
        output_dir / "visualization_diffraction_modulus_3d.png",
        "Diffraction-space modulus",
        "magma",
        "coolwarm",
        "log10 normalized diffraction modulus",
        "Signed normalized modulus difference",
        args.max_volume_points,
        args.volume_point_size,
        args.volume_alpha,
        args.view_elevation,
        args.view_azimuth,
    )
    plot_five_panel_volume(
        diffraction_phase_rows,
        names,
        output_dir / "visualization_diffraction_phase_3d.png",
        "Diffraction-space phase",
        "twilight",
        "coolwarm",
        "Wrapped diffraction phase (rad)",
        "Wrapped diffraction-phase difference (rad)",
        args.max_volume_points,
        args.volume_point_size,
        args.volume_alpha,
        args.view_elevation,
        args.view_azimuth,
        absolute_limits=(-float(np.pi), float(np.pi)),
        difference_limits=(-float(np.pi), float(np.pi)),
    )

    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "phase_model_variant": phase_variant,
        "device": str(device),
        "split": args.split,
        "support_threshold": args.support_threshold,
        "slice_axis": args.slice_axis,
        "slice_index": args.slice_index,
        "alignment": (
            "Predictions are twin- and global-phase-aligned to the target, then "
            "target, initial, and refined objects are independently centered by "
            "the established AutoPhaseNN display post-processing."
        ),
        "dataset_manifest": manifest,
        "samples": sample_metadata,
        "outputs": {
            "slices": "visualization_2d.png",
            "amplitude_3d": "visualization_amplitude_3d.png",
            "phase_3d": "visualization_phase_3d.png",
            "reciprocal_slices": "visualization_reciprocal_2d.png",
            "diffraction_modulus_3d": "visualization_diffraction_modulus_3d.png",
            "diffraction_phase_3d": "visualization_diffraction_phase_3d.png",
        },
    }
    metadata_path = output_dir / "visualization_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Saved visualization directory: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
