"""Visualize HighStrain reciprocal-phase and reconstructed real-space outputs."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from pytorch_autophasenn.data import AutoPhaseNNPhaseDataset, reciprocal_phase_from_realspace
from pytorch_autophasenn.losses import phase_retrieval_wca_components
from pytorch_autophasenn.management import DEFAULT_DATA_CONFIG, load_data_config
from pytorch_autophasenn.model import MODEL_VARIANTS
from pytorch_autophasenn.reconstruction import (
    farfield_modulus_from_realspace,
    realspace_from_modulus_phase,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from autophasenn_training_pipeline.evaluate import (  # noqa: E402
    center_post_processed_object,
    official_post_process,
    official_post_process_before_shift,
    shift_support,
)
from autophasenn_training_pipeline.losses import (  # noqa: E402
    metric_dict,
    realspace_metric_dict,
)
from autophasenn_training_pipeline.visualize_postprocessed import (  # noqa: E402
    object_farfield,
    plot_3d_comparison,
    plot_3d_error_comparison,
    plot_five_panel_volume,
    volume_tensor,
    wrap_phase,
)

from pytorch_autophasenn.evaluate import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    choose_device,
    load_model,
    select_reciprocal_phase,
)


LOGGER = logging.getLogger("high_strain.visualize")
DEFAULT_VISION_ROOT = (
    PROJECT_DIR / "artifacts" / "visualizations" / "autophasenn_pytorch"
)
DEFAULT_OUTPUT_FILENAMES = {
    "output_png": "visualization_2d.png",
    "output_3d_png": "visualization_3d.png",
    "output_shift_3d_png": "visualization_shift_comparison_3d.png",
    "output_error_3d_png": "visualization_error_3d.png",
    "output_reciprocal_2d_png": "visualization_reciprocal_2d.png",
    "output_reciprocal_3d_png": "visualization_reciprocal_3d.png",
    "output_amplitude_3d_png": "visualization_amplitude_3d.png",
    "output_phase_3d_png": "visualization_phase_3d.png",
    "output_diffraction_3d_png": "visualization_diffraction_3d.png",
    "output_diffraction_phase_3d_png": "visualization_diffraction_phase_3d.png",
}


def configure_logging(level: str) -> None:
    """Configure concise console logging."""

    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def optional_output_path(value: str | None) -> Path | None:
    """Resolve an output path while accepting ``none`` to disable it."""

    if value is None or value.lower() in {"", "none", "null"}:
        return None
    return Path(value).expanduser().resolve()


def apply_default_output_paths(
    args: argparse.Namespace,
    model_variant: str,
) -> Path:
    """Place default images in the model-specific vision directory."""

    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else DEFAULT_VISION_ROOT / f"vision_{model_variant}"
    ).resolve()
    for argument, filename in DEFAULT_OUTPUT_FILENAMES.items():
        if not getattr(args, argument):
            setattr(args, argument, str(output_dir / filename))
    return output_dir


def normalized_modulus(volume: np.ndarray) -> np.ndarray:
    """Normalize one nonnegative reciprocal modulus by its maximum."""

    value = np.asarray(volume, dtype=np.float32)
    scale = max(float(np.max(value)), np.finfo(np.float32).eps)
    return value / scale


def masked_phase(
    phase: np.ndarray,
    modulus: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Wrap reciprocal phase and hide low-modulus voxels."""

    return np.where(
        modulus > threshold,
        wrap_phase(np.asarray(phase, dtype=np.float32)),
        np.nan,
    ).astype(np.float32)


def _finite_limits(
    images: list[np.ndarray],
    *,
    symmetric: bool = False,
    phase: bool = False,
) -> tuple[float | None, float | None]:
    if phase:
        return -float(np.pi), float(np.pi)
    values = [image[np.isfinite(image)] for image in images]
    values = [value for value in values if value.size]
    if not values:
        return None, None
    minimum = min(float(value.min()) for value in values)
    maximum = max(float(value.max()) for value in values)
    if symmetric:
        bound = max(abs(minimum), abs(maximum), np.finfo(np.float32).eps)
        return -bound, bound
    if minimum == maximum:
        maximum += np.finfo(np.float32).eps
    return minimum, maximum


def plot_image_grid(
    rows: list[list[np.ndarray]],
    names: list[str],
    row_titles: list[str],
    color_maps: list[str],
    output_png: Path,
    figure_title: str,
    phase_rows: set[int],
    difference_rows: set[int],
) -> None:
    """Render center-slice rows with comparable limits across samples."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample_count = len(rows)
    fig = plt.figure(figsize=(3.7 * sample_count + 1.8, 2.2 * len(row_titles)))
    grid = fig.add_gridspec(
        len(row_titles),
        sample_count + 1,
        width_ratios=[1.0] * sample_count + [0.045],
        wspace=0.08,
        hspace=0.16,
    )
    for row_index, row_title in enumerate(row_titles):
        row_images = [sample[row_index] for sample in rows]
        vmin, vmax = _finite_limits(
            row_images,
            symmetric=row_index in difference_rows,
            phase=row_index in phase_rows,
        )
        last_image = None
        for column, image in enumerate(row_images):
            axis = fig.add_subplot(grid[row_index, column])
            color_map = plt.get_cmap(color_maps[row_index]).copy()
            color_map.set_bad(color="white", alpha=0.0)
            last_image = axis.imshow(
                image,
                cmap=color_map,
                vmin=vmin,
                vmax=vmax,
                origin="lower",
            )
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                axis.set_ylabel(row_title, fontsize=8)
            if row_index == 0:
                axis.set_title(names[column], fontsize=8)
        if last_image is not None:
            color_axis = fig.add_subplot(grid[row_index, -1])
            colorbar_format = "%.2f"
            if row_index in difference_rows and max(abs(vmin), abs(vmax)) < 1e-2:
                colorbar_format = "%.1e"
            colorbar = fig.colorbar(last_image, cax=color_axis, format=colorbar_format)
            colorbar.ax.tick_params(labelsize=6)

    fig.suptitle(figure_title, fontsize=13, y=0.995)
    fig.subplots_adjust(left=0.23, right=0.94, bottom=0.015, top=0.955)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def select_indices(
    dataset_size: int,
    num_samples: int,
    mode: str,
    seed: int,
) -> list[int]:
    """Select deterministic first or seeded validation samples."""

    sample_count = min(dataset_size, num_samples)
    if mode == "first":
        return list(range(sample_count))
    generator = np.random.default_rng(seed)
    return [int(index) for index in generator.choice(dataset_size, sample_count, False)]


def parse_args() -> argparse.Namespace:
    """Parse data, model, display, and output options."""

    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--data-config", default=str(DEFAULT_DATA_CONFIG))
    bootstrap_args, _ = bootstrap.parse_known_args()
    data_config = load_data_config(bootstrap_args.data_config)
    val_config = data_config["splits"]["val"]
    configured_shape = tuple(int(size) for size in data_config["shape"])

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default=str(DEFAULT_DATA_CONFIG))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", default=data_config["root"])
    parser.add_argument("--data-diff", default=val_config["diffraction"])
    parser.add_argument("--data-real", default=val_config["realspace"])
    parser.add_argument("--dataset-size", type=int, default=int(val_config["num_samples"]))
    parser.add_argument("--shape", type=int, default=configured_shape[0])
    parser.add_argument("--dtype-diff", default=data_config["dtypes"]["diffraction"])
    parser.add_argument("--dtype-real", default=data_config["dtypes"]["realspace"])
    parser.add_argument(
        "--input-log-data",
        action=argparse.BooleanOptionalAction,
        default=data_config.get("input_preprocessing", {}).get("transform") == "log1p",
    )
    parser.add_argument(
        "--model-variant",
        choices=("auto", *MODEL_VARIANTS),
        default="auto",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--sample-mode", choices=("seeded", "first"), default="seeded")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--slice-index", type=int, default=configured_shape[0] // 2)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--reciprocal-phase-threshold", type=float, default=0.02)
    parser.add_argument("--reciprocal-surface-level", type=float, default=0.02)
    parser.add_argument("--amplitude-error-level", type=float, default=0.05)
    parser.add_argument("--diffraction-difference-threshold", type=float, default=1e-6)
    parser.add_argument("--max-volume-points", type=int, default=12000)
    parser.add_argument("--volume-point-size", type=float, default=3.0)
    parser.add_argument("--volume-alpha", type=float, default=0.4)
    parser.add_argument("--surface-step-size", type=int, default=2)
    parser.add_argument("--view-elevation", type=float, default=25.0)
    parser.add_argument("--view-azimuth", type=float, default=35.0)
    parser.add_argument(
        "--ambiguity-mode",
        choices=("twin_aligned", "raw"),
        default="twin_aligned",
    )
    parser.add_argument("--output-dir", default="")
    for argument, filename in DEFAULT_OUTPUT_FILENAMES.items():
        parser.add_argument(
            "--" + argument.replace("_", "-"),
            default="",
            help=f"Output path for {filename}; pass none to disable.",
        )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args()
    args.data_config = str(Path(args.data_config).expanduser().resolve())
    if args.dataset_size < 1 or args.num_samples < 1:
        parser.error("Dataset size and sample count must be positive.")
    if not 0 <= args.slice_index < args.shape:
        parser.error("--slice-index must lie inside the configured volume.")
    if args.surface_step_size < 1:
        parser.error("--surface-step-size must be positive.")
    if args.amplitude_error_level <= 0:
        parser.error("--amplitude-error-level must be positive.")
    if args.diffraction_difference_threshold <= 0:
        parser.error("--diffraction-difference-threshold must be positive.")
    if args.max_volume_points < 1:
        parser.error("--max-volume-points must be positive.")
    if args.volume_point_size <= 0:
        parser.error("--volume-point-size must be positive.")
    if not 0 < args.volume_alpha <= 1:
        parser.error("--volume-alpha must be in (0, 1].")
    return args


@torch.inference_mode()
def main() -> int:
    """Run inference and write model-specific visualization artifacts."""

    args = parse_args()
    configure_logging(args.log_level)
    device = choose_device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    model, checkpoint_metadata = load_model(
        checkpoint_path,
        device,
        args.model_variant,
    )
    output_dir = apply_default_output_paths(args, model.model_variant)
    outputs = {
        key: optional_output_path(getattr(args, key))
        for key in DEFAULT_OUTPUT_FILENAMES
    }
    if outputs["output_png"] is None:
        raise ValueError("--output-png cannot be disabled.")

    data_dir = Path(args.data_dir).expanduser().resolve()
    shape = (args.shape, args.shape, args.shape)
    dataset = AutoPhaseNNPhaseDataset(
        data_dir / args.data_diff,
        data_dir / args.data_real,
        args.dataset_size,
        shape=shape,
        diffraction_dtype=args.dtype_diff,
        realspace_dtype=args.dtype_real,
        input_log_data=args.input_log_data,
        return_diffraction_modulus=True,
    )
    indices = select_indices(
        len(dataset),
        args.num_samples,
        args.sample_mode,
        args.seed,
    )
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False)

    slice_rows: list[list[np.ndarray]] = []
    reciprocal_rows: list[list[np.ndarray]] = []
    object_rows: list[list[tuple[np.ndarray | None, np.ndarray | None, str]]] = []
    shift_rows: list[list[tuple[np.ndarray | None, np.ndarray | None, str]]] = []
    error_rows: list[list[tuple[np.ndarray | None, np.ndarray | None, str, float]]] = []
    reciprocal_3d_rows: list[
        list[tuple[np.ndarray | None, np.ndarray | None, str]]
    ] = []
    amplitude_volume_rows: list[
        list[tuple[np.ndarray | None, np.ndarray | None, str, float]]
    ] = []
    phase_volume_rows: list[
        list[tuple[np.ndarray | None, np.ndarray | None, str, float]]
    ] = []
    diffraction_volume_rows: list[
        list[tuple[np.ndarray | None, np.ndarray | None, str, float]]
    ] = []
    diffraction_phase_volume_rows: list[
        list[tuple[np.ndarray | None, np.ndarray | None, str, float]]
    ] = []
    metrics: list[dict[str, Any]] = []
    names: list[str] = []

    for batch in loader:
        model_input = batch["input"].to(device).float()
        measured_modulus = batch["diffraction"].to(device).float()
        true_object = batch["realspace"].to(device)
        target_phase = reciprocal_phase_from_realspace(true_object)
        predicted_phase = model(model_input)
        selected_phase, direct, inverted, twin_selected = select_reciprocal_phase(
            predicted_phase,
            target_phase,
            model_input[:, 0],
            args.ambiguity_mode,
        )
        predicted_object = realspace_from_modulus_phase(
            measured_modulus,
            selected_phase,
        )
        reprojected_modulus = farfield_modulus_from_realspace(predicted_object)

        true_amp = true_object.abs().cpu().numpy()[0]
        true_phi = torch.angle(true_object).cpu().numpy()[0]
        pred_amp = predicted_object.abs().cpu().numpy()[0, 0]
        pred_phi = torch.angle(predicted_object).cpu().numpy()[0, 0]
        pred_support = (pred_amp >= args.threshold).astype(np.float32)
        pred_amp_before, pred_phi_before = official_post_process_before_shift(
            pred_amp,
            pred_phi,
            threshold=args.threshold,
        )
        pred_amp_post, pred_phi_post = center_post_processed_object(
            pred_amp_before,
            pred_phi_before,
            threshold=args.threshold,
        )
        true_amp_post, true_phi_post = official_post_process(
            true_amp,
            true_phi,
            threshold=args.threshold,
        )
        support_post = shift_support(pred_support).astype(np.float32)

        sample_metrics = {
            "name": batch["name"][0],
            "phase_wca": float(torch.minimum(direct, inverted).cpu()),
            "phase_wca_direct": float(direct.cpu()),
            "phase_wca_inverted": float(inverted.cpu()),
            "twin_flip_selected": bool(twin_selected.cpu()),
        }
        sample_metrics.update(
            realspace_metric_dict(
                volume_tensor(true_amp_post, device),
                volume_tensor(true_phi_post, device),
                volume_tensor(pred_amp_post, device),
                volume_tensor(pred_phi_post, device),
                volume_tensor(support_post, device),
                threshold=args.threshold,
            )
        )
        reprojection_metrics = metric_dict(measured_modulus, reprojected_modulus)
        sample_metrics.update(
            {f"reprojection_{key}": value for key, value in reprojection_metrics.items()}
        )

        measured_np = measured_modulus.cpu().numpy()[0, 0]
        reprojected_np = reprojected_modulus.cpu().numpy()[0, 0]
        measured_norm = normalized_modulus(measured_np)
        measured_scale = max(float(np.max(measured_np)), np.finfo(np.float32).eps)
        reprojected_norm = reprojected_np / measured_scale
        pred_before_modulus, pred_before_farfield_phase = object_farfield(
            pred_amp_before,
            pred_phi_before,
            device=device,
        )
        pred_post_modulus, pred_post_farfield_phase = object_farfield(
            pred_amp_post,
            pred_phi_post,
            device=device,
        )
        _true_post_modulus, true_post_farfield_phase = object_farfield(
            true_amp_post,
            true_phi_post,
            device=device,
        )
        pred_before_modulus_norm = pred_before_modulus / measured_scale
        pred_post_modulus_norm = pred_post_modulus / measured_scale
        target_phase_np = target_phase.cpu().numpy()[0]
        selected_phase_np = selected_phase.cpu().numpy()[0, 0]
        target_phase_display = masked_phase(
            target_phase_np,
            measured_norm,
            args.reciprocal_phase_threshold,
        )
        predicted_phase_display = masked_phase(
            selected_phase_np,
            measured_norm,
            args.reciprocal_phase_threshold,
        )
        reciprocal_phase_difference = np.where(
            measured_norm > args.reciprocal_phase_threshold,
            wrap_phase(target_phase_np - selected_phase_np),
            np.nan,
        ).astype(np.float32)

        index = args.slice_index
        phase_mask = pred_amp_post[:, :, index] > args.threshold
        true_phase_slice = wrap_phase(true_phi_post[:, :, index])
        pred_phase_slice = np.where(
            phase_mask,
            wrap_phase(pred_phi_post[:, :, index]),
            np.nan,
        )
        phase_difference = np.where(
            phase_mask,
            wrap_phase(true_phase_slice - pred_phase_slice),
            np.nan,
        )
        slice_rows.append(
            [
                np.log10(np.clip(measured_norm[:, :, index], 1e-6, None)),
                target_phase_display[:, :, index],
                predicted_phase_display[:, :, index],
                reciprocal_phase_difference[:, :, index],
                true_amp_post[:, :, index],
                pred_amp_post[:, :, index],
                true_amp_post[:, :, index] - pred_amp_post[:, :, index],
                true_phase_slice,
                pred_phase_slice,
                phase_difference,
                support_post[:, :, index],
            ]
        )
        reciprocal_rows.append(
            [
                np.log10(np.clip(measured_norm[:, :, index], 1e-6, None)),
                np.log10(np.clip(reprojected_norm[:, :, index], 1e-6, None)),
                measured_norm[:, :, index] - reprojected_norm[:, :, index],
                target_phase_display[:, :, index],
                predicted_phase_display[:, :, index],
                reciprocal_phase_difference[:, :, index],
            ]
        )
        object_rows.append(
            [
                (true_amp_post, true_phi_post, "Ground truth after center shift"),
                (pred_amp_post, pred_phi_post, "Prediction after center shift"),
            ]
        )
        shift_rows.append(
            [
                (pred_amp_before, pred_phi_before, "Prediction before center shift"),
                (pred_amp_post, pred_phi_post, "Prediction after center shift"),
            ]
        )
        amplitude_error = true_amp_post - pred_amp_post
        phase_geometry = np.minimum(true_amp_post, pred_amp_post)
        error_rows.append(
            [
                (
                    np.abs(amplitude_error),
                    amplitude_error,
                    "Amplitude error: true - prediction",
                    args.amplitude_error_level,
                ),
                (
                    phase_geometry,
                    wrap_phase(true_phi_post - pred_phi_post),
                    "Wrapped phase error on support intersection",
                    args.threshold,
                ),
            ]
        )
        reciprocal_3d_rows.append(
            [
                (measured_norm, target_phase_np, "Measured modulus + target phase"),
                (
                    measured_norm,
                    selected_phase_np,
                    "Measured modulus reused + predicted phase",
                ),
            ]
        )

        amplitude_shift_difference = pred_amp_before - pred_amp_post
        amplitude_target_difference = pred_amp_post - true_amp_post
        amplitude_volume_rows.append(
            [
                (true_amp_post, true_amp_post, "Target", args.threshold),
                (
                    pred_amp_before,
                    pred_amp_before,
                    "Prediction before center shift",
                    args.threshold,
                ),
                (
                    pred_amp_post,
                    pred_amp_post,
                    "Prediction after center shift",
                    args.threshold,
                ),
                (
                    np.abs(amplitude_shift_difference),
                    amplitude_shift_difference,
                    "Difference: before - after",
                    args.amplitude_error_level,
                ),
                (
                    np.abs(amplitude_target_difference),
                    amplitude_target_difference,
                    "Difference: after - target",
                    args.amplitude_error_level,
                ),
            ]
        )

        phase_shift_geometry = np.minimum(pred_amp_before, pred_amp_post)
        phase_target_geometry = np.minimum(pred_amp_post, true_amp_post)
        phase_volume_rows.append(
            [
                (
                    true_amp_post,
                    wrap_phase(true_phi_post),
                    "Target",
                    args.threshold,
                ),
                (
                    pred_amp_before,
                    wrap_phase(pred_phi_before),
                    "Prediction before center shift",
                    args.threshold,
                ),
                (
                    pred_amp_post,
                    wrap_phase(pred_phi_post),
                    "Prediction after center shift",
                    args.threshold,
                ),
                (
                    phase_shift_geometry,
                    wrap_phase(pred_phi_before - pred_phi_post),
                    "Wrapped difference: before - after",
                    args.threshold,
                ),
                (
                    phase_target_geometry,
                    wrap_phase(pred_phi_post - true_phi_post),
                    "Wrapped difference: after - target",
                    args.threshold,
                ),
            ]
        )

        diffraction_shift_difference = (
            pred_before_modulus_norm - pred_post_modulus_norm
        )
        diffraction_target_difference = pred_post_modulus_norm - measured_norm
        diffraction_volume_rows.append(
            [
                (
                    measured_norm,
                    np.log10(np.clip(measured_norm, 1e-6, None)),
                    "Target measured modulus",
                    args.reciprocal_surface_level,
                ),
                (
                    pred_before_modulus_norm,
                    np.log10(np.clip(pred_before_modulus_norm, 1e-6, None)),
                    "Reconstruction before center shift",
                    args.reciprocal_surface_level,
                ),
                (
                    pred_post_modulus_norm,
                    np.log10(np.clip(pred_post_modulus_norm, 1e-6, None)),
                    "Reconstruction after center shift",
                    args.reciprocal_surface_level,
                ),
                (
                    np.abs(diffraction_shift_difference),
                    diffraction_shift_difference,
                    "Difference: before - after",
                    args.diffraction_difference_threshold,
                ),
                (
                    np.abs(diffraction_target_difference),
                    diffraction_target_difference,
                    "Difference: after - target",
                    args.diffraction_difference_threshold,
                ),
            ]
        )

        diffraction_phase_shift_geometry = np.minimum(
            pred_before_modulus_norm,
            pred_post_modulus_norm,
        )
        diffraction_phase_target_geometry = np.minimum(
            pred_post_modulus_norm,
            measured_norm,
        )
        diffraction_phase_volume_rows.append(
            [
                (
                    measured_norm,
                    true_post_farfield_phase,
                    "Target phase derived from real-space target",
                    args.reciprocal_phase_threshold,
                ),
                (
                    pred_before_modulus_norm,
                    pred_before_farfield_phase,
                    "Reconstruction before center shift",
                    args.reciprocal_phase_threshold,
                ),
                (
                    pred_post_modulus_norm,
                    pred_post_farfield_phase,
                    "Reconstruction after center shift",
                    args.reciprocal_phase_threshold,
                ),
                (
                    diffraction_phase_shift_geometry,
                    wrap_phase(pred_before_farfield_phase - pred_post_farfield_phase),
                    "Wrapped difference: before - after",
                    args.reciprocal_phase_threshold,
                ),
                (
                    diffraction_phase_target_geometry,
                    wrap_phase(pred_post_farfield_phase - true_post_farfield_phase),
                    "Wrapped difference: after - target",
                    args.reciprocal_phase_threshold,
                ),
            ]
        )
        names.append(batch["name"][0])
        metrics.append(sample_metrics)

    overview_titles = [
        "Measured modulus (log10 normalized)",
        "Target reciprocal phase (rad)",
        "Predicted reciprocal phase (rad)",
        "Wrapped reciprocal phase error",
        "True amplitude (post)",
        "Predicted amplitude (post)",
        "True - predicted amplitude",
        "True real-space phase (rad)",
        "Predicted real-space phase (rad)",
        "Wrapped real-space phase error",
        "Predicted support",
    ]
    plot_image_grid(
        slice_rows,
        names,
        overview_titles,
        [
            "magma",
            "twilight",
            "twilight",
            "coolwarm",
            "viridis",
            "viridis",
            "coolwarm",
            "twilight",
            "twilight",
            "coolwarm",
            "gray",
        ],
        outputs["output_png"],
        "HighStrain reciprocal-phase and real-space center slices",
        phase_rows={1, 2, 3, 7, 8, 9},
        difference_rows={3, 6, 9},
    )
    if outputs["output_reciprocal_2d_png"] is not None:
        plot_image_grid(
            reciprocal_rows,
            names,
            [
                "Measured modulus\n(log10 normalized)",
                "Reprojected modulus\n(measured reused)",
                "Measured - reprojected\nmodulus",
                "Target reciprocal phase (rad)",
                "Predicted reciprocal phase (rad)",
                "Wrapped reciprocal phase error",
            ],
            ["magma", "magma", "coolwarm", "twilight", "twilight", "coolwarm"],
            outputs["output_reciprocal_2d_png"],
            "HighStrain reciprocal-space reconstruction",
            phase_rows={3, 4, 5},
            difference_rows={2, 5},
        )
    if outputs["output_3d_png"] is not None:
        plot_3d_comparison(
            object_rows,
            names,
            outputs["output_3d_png"],
            args.threshold,
            args.surface_step_size,
            args.view_elevation,
            args.view_azimuth,
            "HighStrain post-processed 3D object comparison",
        )
    if outputs["output_shift_3d_png"] is not None:
        plot_3d_comparison(
            shift_rows,
            names,
            outputs["output_shift_3d_png"],
            args.threshold,
            args.surface_step_size,
            args.view_elevation,
            args.view_azimuth,
            "HighStrain center-of-mass shift",
        )
    if outputs["output_error_3d_png"] is not None:
        plot_3d_error_comparison(
            error_rows,
            names,
            outputs["output_error_3d_png"],
            args.surface_step_size,
            args.view_elevation,
            args.view_azimuth,
        )
    if outputs["output_reciprocal_3d_png"] is not None:
        plot_3d_comparison(
            reciprocal_3d_rows,
            names,
            outputs["output_reciprocal_3d_png"],
            args.reciprocal_surface_level,
            args.surface_step_size,
            args.view_elevation,
            args.view_azimuth,
            "Measured modulus with target vs predicted reciprocal phase",
        )
    if outputs["output_amplitude_3d_png"] is not None:
        plot_five_panel_volume(
            amplitude_volume_rows,
            names,
            outputs["output_amplitude_3d_png"],
            figure_title="HighStrain real-space amplitude 3D volumes",
            absolute_color_map="viridis",
            difference_color_map="coolwarm",
            absolute_colorbar_label="Amplitude",
            difference_colorbar_label="Signed amplitude difference",
            max_points=args.max_volume_points,
            point_size=args.volume_point_size,
            alpha=args.volume_alpha,
            elevation=args.view_elevation,
            azimuth=args.view_azimuth,
            absolute_zero_minimum=True,
        )
    if outputs["output_phase_3d_png"] is not None:
        plot_five_panel_volume(
            phase_volume_rows,
            names,
            outputs["output_phase_3d_png"],
            figure_title="HighStrain real-space phase 3D volumes",
            absolute_color_map="twilight",
            difference_color_map="coolwarm",
            absolute_colorbar_label="Wrapped phase (rad)",
            difference_colorbar_label="Wrapped phase difference (rad)",
            max_points=args.max_volume_points,
            point_size=args.volume_point_size,
            alpha=args.volume_alpha,
            elevation=args.view_elevation,
            azimuth=args.view_azimuth,
            absolute_limits=(-float(np.pi), float(np.pi)),
        )
    if outputs["output_diffraction_3d_png"] is not None:
        plot_five_panel_volume(
            diffraction_volume_rows,
            names,
            outputs["output_diffraction_3d_png"],
            figure_title="HighStrain diffraction-modulus 3D volumes",
            absolute_color_map="magma",
            difference_color_map="coolwarm",
            absolute_colorbar_label="log10 normalized diffraction modulus",
            difference_colorbar_label="Signed normalized modulus difference",
            max_points=args.max_volume_points,
            point_size=args.volume_point_size,
            alpha=args.volume_alpha,
            elevation=args.view_elevation,
            azimuth=args.view_azimuth,
        )
    if outputs["output_diffraction_phase_3d_png"] is not None:
        plot_five_panel_volume(
            diffraction_phase_volume_rows,
            names,
            outputs["output_diffraction_phase_3d_png"],
            figure_title="HighStrain derived diffraction-phase 3D volumes",
            absolute_color_map="twilight",
            difference_color_map="coolwarm",
            absolute_colorbar_label="Wrapped diffraction phase (rad)",
            difference_colorbar_label="Wrapped diffraction-phase difference (rad)",
            max_points=args.max_volume_points,
            point_size=args.volume_point_size,
            alpha=args.volume_alpha,
            elevation=args.view_elevation,
            azimuth=args.view_azimuth,
            absolute_limits=(-float(np.pi), float(np.pi)),
        )

    metadata_path = outputs["output_png"].with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": checkpoint_metadata.get("epoch"),
                "model_variant": model.model_variant,
                "ambiguity_mode": args.ambiguity_mode,
                "dataset_size": args.dataset_size,
                "sample_mode": args.sample_mode,
                "sample_indices": indices,
                "sample_names": names,
                "slice_index": args.slice_index,
                "threshold": args.threshold,
                "reciprocal_phase_threshold": args.reciprocal_phase_threshold,
                "outputs": {
                    key: str(path) if path is not None else None
                    for key, path in outputs.items()
                },
                "notes": {
                    "modulus": (
                        "The measured modulus is reused during reconstruction; its "
                        "reprojection error is not an independent model prediction."
                    ),
                    "postprocess": (
                        "Real-space values use the same official AutoPhaseNN unwrap, "
                        "phase-offset removal, and center-of-mass alignment."
                    ),
                },
                "per_sample": metrics,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    LOGGER.info("Saved HighStrain visualizations: %s", output_dir)
    LOGGER.info("Selected samples: %s", ", ".join(names))
    LOGGER.info("Saved visualization metadata: %s", metadata_path)
    for output_name, output_path in outputs.items():
        if output_path is not None:
            LOGGER.info("Saved %s: %s", output_name, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
