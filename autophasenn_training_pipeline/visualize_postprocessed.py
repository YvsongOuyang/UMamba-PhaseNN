"""Visualize AutoPhaseNN predictions with the evaluator's official post-processing."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

try:
    from .dataset import AutoPhaseDataset
    from .evaluate import (
        center_post_processed_object,
        choose_device,
        official_post_process,
        official_post_process_before_shift,
        optional_data_path,
        shift_support,
    )
    from .losses import metric_dict, realspace_metric_dict, scale_align_sum
    from .model_factory import MODEL_VARIANTS, create_model
    from .model_tf_compatible import load_weights
except ImportError:
    from dataset import AutoPhaseDataset
    from evaluate import (
        center_post_processed_object,
        choose_device,
        official_post_process,
        official_post_process_before_shift,
        optional_data_path,
        shift_support,
    )
    from losses import metric_dict, realspace_metric_dict, scale_align_sum
    from model_factory import MODEL_VARIANTS, create_model
    from model_tf_compatible import load_weights


LOGGER = logging.getLogger("autophasenn.visualize")
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output" / "evaluate"


def configure_logging(level: str) -> None:
    """Configure concise console logging for a visualization run."""

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def optional_output_path(value: str | None) -> Path | None:
    """Resolve an output path while accepting ``none`` to disable it."""

    if value is None or value.lower() in {"", "none", "null"}:
        return None
    return Path(value).expanduser().resolve()


def wrap_phase(phase: np.ndarray) -> np.ndarray:
    """Wrap phase to [-pi, pi] for cyclic color mapping."""

    return np.arctan2(np.sin(phase), np.cos(phase))


def volume_tensor(volume: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert one 3D NumPy volume to the model's ``(1, 1, D, H, W)`` layout."""

    contiguous = np.ascontiguousarray(volume[None, None], dtype=np.float32)
    return torch.from_numpy(contiguous).to(device=device)


def row_limits(
    images: list[np.ndarray], row_index: int
) -> tuple[float | None, float | None]:
    """Return common color limits so columns remain directly comparable."""

    if row_index in {6, 7, 8}:
        return -float(np.pi), float(np.pi)
    if row_index == 9:
        return 0.0, 1.0

    finite_values = [image[np.isfinite(image)] for image in images]
    finite_values = [values for values in finite_values if values.size]
    if not finite_values:
        return None, None
    minimum = min(float(values.min()) for values in finite_values)
    maximum = max(float(values.max()) for values in finite_values)
    if row_index in {2, 5}:
        bound = max(abs(minimum), abs(maximum), np.finfo(np.float32).eps)
        return -bound, bound
    if minimum == maximum:
        maximum = minimum + np.finfo(np.float32).eps
    return minimum, maximum


def plot_slice_rows(
    rows: list[list[np.ndarray]],
    names: list[str],
    output_png: Path,
) -> None:
    """Save a readable, row-normalized center-slice comparison."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_titles = [
        "Input FT (log10)",
        "Predicted FT (log10)",
        "Input - prediction FT",
        "True amplitude (post)",
        "Predicted amplitude (post)",
        "True - prediction amplitude",
        "True phase (post, rad)",
        "Predicted phase (post, rad)",
        "Wrapped true - prediction phase",
        "Predicted support (shifted)",
    ]
    color_maps = [
        "magma",
        "magma",
        "coolwarm",
        "viridis",
        "viridis",
        "coolwarm",
        "twilight",
        "twilight",
        "twilight",
        "gray",
    ]

    sample_count = len(rows)
    fig = plt.figure(figsize=(3.5 * sample_count + 1.2, 23.0))
    grid = fig.add_gridspec(
        len(row_titles),
        sample_count + 1,
        width_ratios=[1.0] * sample_count + [0.045],
        wspace=0.08,
        hspace=0.16,
    )
    axes = np.empty((len(row_titles), sample_count), dtype=object)
    for row_index, row_title in enumerate(row_titles):
        row_images = [sample[row_index] for sample in rows]
        vmin, vmax = row_limits(row_images, row_index)
        last_image = None
        for column, image in enumerate(row_images):
            axis = fig.add_subplot(grid[row_index, column])
            axes[row_index, column] = axis
            last_image = axis.imshow(
                image,
                cmap=color_maps[row_index],
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
            colorbar = fig.colorbar(last_image, cax=color_axis, format="%.2f")
            colorbar.ax.tick_params(labelsize=6)

    fig.suptitle("AutoPhaseNN post-processed center slices", fontsize=13, y=0.998)
    fig.subplots_adjust(left=0.14, right=0.94, bottom=0.015, top=0.985)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def extract_phase_surface(
    amplitude: np.ndarray,
    phase: np.ndarray,
    threshold: float,
    step_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Extract an amplitude isosurface and one cyclic phase color per face."""

    from skimage.measure import marching_cubes

    amplitude = np.asarray(amplitude, dtype=np.float32)
    phase = wrap_phase(np.asarray(phase, dtype=np.float32))
    minimum = float(np.nanmin(amplitude))
    maximum = float(np.nanmax(amplitude))
    if not minimum < threshold < maximum:
        return None

    vertices, faces, _normals, _values = marching_cubes(
        amplitude,
        level=threshold,
        step_size=step_size,
        allow_degenerate=False,
    )
    volume_indices = np.rint(vertices).astype(np.int64)
    for axis, size in enumerate(amplitude.shape):
        volume_indices[:, axis] = np.clip(volume_indices[:, axis], 0, size - 1)
    vertex_phase = phase[tuple(volume_indices[:, axis] for axis in range(3))]
    face_phase = np.angle(np.mean(np.exp(1j * vertex_phase[faces]), axis=1))

    # marching_cubes returns array-axis coordinates (axis0, axis1, axis2).
    # Matplotlib expects conventional plot coordinates (x, y, z).
    plot_vertices = vertices[:, [2, 1, 0]]
    return plot_vertices, faces, face_phase.astype(np.float32)


def add_phase_surface(
    axis: Any,
    amplitude: np.ndarray | None,
    phase: np.ndarray | None,
    threshold: float,
    step_size: int,
    elevation: float,
    azimuth: float,
    empty_message: str,
) -> None:
    """Draw one phase-colored 3D amplitude isosurface on a shared coordinate frame."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if amplitude is None or phase is None:
        axis.text2D(0.5, 0.5, empty_message, transform=axis.transAxes, ha="center")
        shape = (64, 64, 64)
        surface = None
    else:
        shape = amplitude.shape
        surface = extract_phase_surface(amplitude, phase, threshold, step_size)
        if surface is None:
            axis.text2D(
                0.5,
                0.5,
                "No amplitude above threshold",
                transform=axis.transAxes,
                ha="center",
            )

    if surface is not None:
        vertices, faces, face_phase = surface
        color_map = plt.get_cmap("twilight")
        normalizer = Normalize(vmin=-np.pi, vmax=np.pi)
        mesh = Poly3DCollection(
            vertices[faces],
            facecolors=color_map(normalizer(face_phase)),
            edgecolors="none",
            linewidths=0.0,
            alpha=0.92,
        )
        axis.add_collection3d(mesh)

    depth, height, width = shape
    axis.set_xlim(0, max(width - 1, 1))
    axis.set_ylim(0, max(height - 1, 1))
    axis.set_zlim(0, max(depth - 1, 1))
    axis.set_box_aspect((width, height, depth))
    axis.view_init(elev=elevation, azim=azimuth)
    axis.scatter(
        [width / 2.0],
        [height / 2.0],
        [depth / 2.0],
        color="red",
        marker="x",
        s=32,
        linewidths=1.5,
        depthshade=False,
    )
    center_x, center_y, center_z = width / 2.0, height / 2.0, depth / 2.0
    center_line_style = {
        "color": "red",
        "linestyle": "--",
        "linewidth": 0.8,
        "alpha": 0.75,
    }
    axis.plot(
        [0, width - 1],
        [center_y, center_y],
        [center_z, center_z],
        **center_line_style,
    )
    axis.plot(
        [center_x, center_x],
        [0, height - 1],
        [center_z, center_z],
        **center_line_style,
    )
    axis.plot(
        [center_x, center_x],
        [center_y, center_y],
        [0, depth - 1],
        **center_line_style,
    )
    axis.set_xlabel("X", fontsize=7, labelpad=-1)
    axis.set_ylabel("Y", fontsize=7, labelpad=-1)
    axis.set_zlabel("Z", fontsize=7, labelpad=-1)
    axis.tick_params(labelsize=6, pad=0)
    axis.grid(True, linewidth=0.35, alpha=0.45)


def plot_3d_comparison(
    panel_rows: list[list[tuple[np.ndarray | None, np.ndarray | None, str]]],
    names: list[str],
    output_png: Path,
    threshold: float,
    step_size: int,
    elevation: float,
    azimuth: float,
    figure_title: str,
) -> None:
    """Save paired phase-colored 3D surfaces for every selected sample."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    row_count = len(panel_rows)
    fig, axes = plt.subplots(
        row_count,
        2,
        figsize=(10.0, max(4.7 * row_count, 5.2)),
        squeeze=False,
        subplot_kw={"projection": "3d"},
    )
    for row_index, panels in enumerate(panel_rows):
        for column, (amplitude, phase, panel_title) in enumerate(panels):
            axis = axes[row_index, column]
            add_phase_surface(
                axis,
                amplitude,
                phase,
                threshold=threshold,
                step_size=step_size,
                elevation=elevation,
                azimuth=azimuth,
                empty_message="Ground truth unavailable",
            )
            axis.set_title(f"{names[row_index]}\n{panel_title}", fontsize=9, pad=3)

    mapper = ScalarMappable(norm=Normalize(vmin=-np.pi, vmax=np.pi), cmap="twilight")
    mapper.set_array([])
    color_axis = fig.add_axes((0.92, 0.16, 0.018, 0.68))
    colorbar = fig.colorbar(mapper, cax=color_axis)
    colorbar.set_label("Wrapped phase (rad)", fontsize=9)
    colorbar.ax.tick_params(labelsize=7)
    fig.suptitle(
        f"{figure_title} — isosurface={threshold:g}; red guides=volume center",
        fontsize=12,
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.02, right=0.89, bottom=0.025, top=0.89, wspace=0.02, hspace=0.12
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse and validate visualization command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Visualize AutoPhaseNN center slices and phase-colored 3D surfaces "
            "with evaluate.py post-processing."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "/data_ssd/oyys/autophasenn/autophasenn_pipeline_output/"
            "autophasenn_retrain_l1/checkpoint_best.pt"
        ),
    )
    parser.add_argument("--data-dir", default="/data_ssd/oyys/autophasenn/")
    parser.add_argument("--data-diff", default="val_diff.npy")
    parser.add_argument("--data-real", default="val_real.npy")
    parser.add_argument("--shape", type=int, default=64)
    parser.add_argument(
        "--model-variant",
        choices=MODEL_VARIANTS,
        default="baseline",
        help="Network architecture variant.",
    )
    parser.add_argument("--dtype-diff", default="float32")
    parser.add_argument("--dtype-real", default="complex64")
    parser.add_argument(
        "--output-png",
        default=str(DEFAULT_OUTPUT_DIR / "visualization_2d.png"),
        help="Center-slice PNG path.",
    )
    parser.add_argument(
        "--output-3d-png",
        default=str(DEFAULT_OUTPUT_DIR / "visualization_3d.png"),
        help="True/predicted 3D comparison path; pass none to disable.",
    )
    parser.add_argument(
        "--output-shift-3d-png",
        default=str(DEFAULT_OUTPUT_DIR / "visualization_shift_comparison_3d.png"),
        help="Prediction before/after center-shift 3D comparison; pass none to disable.",
    )
    parser.add_argument("--dataset-size", type=int, default=5000)
    parser.add_argument(
        "--overfit-samples",
        type=int,
        default=5000,
        help="Restrict the pool to the same first N samples used by train.py.",
    )
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-mode",
        choices=["seeded", "first"],
        default="seeded",
        help="Select deterministic random samples or the first N samples.",
    )
    parser.add_argument("--slice-index", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--surface-step-size", type=int, default=2)
    parser.add_argument("--view-elevation", type=float, default=25.0)
    parser.add_argument("--view-azimuth", type=float, default=35.0)
    parser.add_argument("--scale-i", type=float, default=0.0)
    parser.add_argument("--scale-align-loss", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    args = parser.parse_args()

    if args.shape <= 0:
        parser.error("--shape must be positive.")
    if not 0 <= args.slice_index < args.shape:
        parser.error("--slice-index must be between 0 and shape - 1.")
    if args.dataset_size <= 0:
        parser.error("--dataset-size must be positive.")
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive.")
    if args.surface_step_size <= 0:
        parser.error("--surface-step-size must be positive.")
    return args


@torch.inference_mode()
def main() -> int:
    """Run inference and write 2D, 3D, shift-comparison, and JSON outputs."""

    args = parse_args()
    configure_logging(args.log_level)

    output_png = optional_output_path(args.output_png)
    output_3d_png = optional_output_path(args.output_3d_png)
    output_shift_3d_png = optional_output_path(args.output_shift_3d_png)
    if output_png is None:
        raise ValueError("--output-png cannot be disabled.")

    device = choose_device(args.device)
    data_dir = Path(args.data_dir).expanduser()
    data_real_path = optional_data_path(data_dir, args.data_real)
    has_ground_truth = data_real_path is not None
    shape = (args.shape, args.shape, args.shape)
    sample_pool_size = args.dataset_size
    if args.overfit_samples > 0:
        sample_pool_size = min(sample_pool_size, args.overfit_samples)

    dataset = AutoPhaseDataset(
        data_dir / args.data_diff,
        data_real_path,
        sample_pool_size,
        shape_diff=shape,
        shape_real=shape,
        dtype_diff=args.dtype_diff,
        dtype_real=args.dtype_real,
        scale_i=args.scale_i,
        shuffle=args.sample_mode == "seeded",
        seed=args.seed,
    )
    sample_count = min(args.num_samples, len(dataset))
    loader = DataLoader(
        Subset(dataset, range(sample_count)),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = create_model(args.model_variant, threshold=args.threshold).to(device)
    load_weights(model, args.checkpoint, map_location=device)
    model.eval()

    slice_rows: list[list[np.ndarray]] = []
    object_3d_rows: list[list[tuple[np.ndarray | None, np.ndarray | None, str]]] = []
    shift_3d_rows: list[list[tuple[np.ndarray | None, np.ndarray | None, str]]] = []
    names: list[str] = []
    metrics: list[dict[str, Any]] = []
    slice_index = args.slice_index

    for batch in loader:
        name = batch["name"][0]
        diff = batch["diff"].to(device, non_blocking=True).float()
        true_amp = batch["amp"].numpy()[0, 0]
        true_phi = batch["phi"].numpy()[0, 0]

        pred_diff, _pred_obj, pred_amp, pred_phi, support = model(diff)[:5]
        pred_for_metric = (
            scale_align_sum(diff, pred_diff) if args.scale_align_loss else pred_diff
        )
        sample_metrics = {"name": name, **metric_dict(diff, pred_for_metric)}

        diff_np = diff.detach().cpu().numpy()[0, 0]
        pred_diff_np = pred_diff.detach().cpu().numpy()[0, 0]
        pred_amp_np = pred_amp.detach().cpu().numpy()[0, 0]
        pred_phi_np = pred_phi.detach().cpu().numpy()[0, 0]
        support_np = support.detach().cpu().numpy()[0, 0]

        pred_amp_before, pred_phi_before = official_post_process_before_shift(
            pred_amp_np,
            pred_phi_np,
            threshold=args.threshold,
        )
        pred_amp_post, pred_phi_post = center_post_processed_object(
            pred_amp_before,
            pred_phi_before,
            threshold=args.threshold,
        )
        support_shifted = shift_support(support_np)

        true_amp_post: np.ndarray | None = None
        true_phi_post: np.ndarray | None = None
        if has_ground_truth:
            true_amp_post, true_phi_post = official_post_process(
                true_amp,
                true_phi,
                threshold=args.threshold,
            )
            sample_metrics.update(
                realspace_metric_dict(
                    volume_tensor(true_amp_post, device),
                    volume_tensor(true_phi_post, device),
                    volume_tensor(pred_amp_post, device),
                    volume_tensor(pred_phi_post, device),
                    pred_support=volume_tensor(support_shifted, device),
                    threshold=args.threshold,
                )
            )

        true_amp_display = (
            true_amp_post if true_amp_post is not None else np.zeros_like(pred_amp_post)
        )
        true_phi_display = (
            true_phi_post if true_phi_post is not None else np.zeros_like(pred_phi_post)
        )
        phase_mask = (pred_amp_post[:, :, slice_index] > args.threshold).astype(
            np.float32
        )
        true_phase_slice = wrap_phase(true_phi_display[:, :, slice_index])
        pred_phase_slice = phase_mask * wrap_phase(pred_phi_post[:, :, slice_index])
        phase_difference = phase_mask * wrap_phase(true_phase_slice - pred_phase_slice)

        slice_rows.append(
            [
                np.log10(np.maximum(diff_np[:, :, slice_index], 0.0) + 1.0),
                np.log10(np.maximum(pred_diff_np[:, :, slice_index], 0.0) + 1.0),
                diff_np[:, :, slice_index] - pred_diff_np[:, :, slice_index],
                true_amp_display[:, :, slice_index],
                pred_amp_post[:, :, slice_index],
                true_amp_display[:, :, slice_index] - pred_amp_post[:, :, slice_index],
                true_phase_slice,
                pred_phase_slice,
                phase_difference,
                support_shifted[:, :, slice_index],
            ]
        )
        object_3d_rows.append(
            [
                (true_amp_post, true_phi_post, "Ground truth after center shift"),
                (pred_amp_post, pred_phi_post, "Prediction after center shift"),
            ]
        )
        shift_3d_rows.append(
            [
                (pred_amp_before, pred_phi_before, "Prediction before center shift"),
                (pred_amp_post, pred_phi_post, "Prediction after center shift"),
            ]
        )
        names.append(name)
        metrics.append(sample_metrics)

    plot_slice_rows(slice_rows, names, output_png)
    if output_3d_png is not None:
        plot_3d_comparison(
            object_3d_rows,
            names,
            output_3d_png,
            threshold=args.threshold,
            step_size=args.surface_step_size,
            elevation=args.view_elevation,
            azimuth=args.view_azimuth,
            figure_title="Post-processed 3D object comparison",
        )
    if output_shift_3d_png is not None:
        plot_3d_comparison(
            shift_3d_rows,
            names,
            output_shift_3d_png,
            threshold=args.threshold,
            step_size=args.surface_step_size,
            elevation=args.view_elevation,
            azimuth=args.view_azimuth,
            figure_title="Center-of-mass shift: before vs. after",
        )

    output_json = output_png.with_suffix(".json")
    output_json.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "checkpoint": str(Path(args.checkpoint).expanduser()),
                "model_variant": args.model_variant,
                "dataset_size": args.dataset_size,
                "sample_pool_size": sample_pool_size,
                "overfit_samples": args.overfit_samples,
                "sample_mode": args.sample_mode,
                "num_samples": sample_count,
                "sample_names": names,
                "slice_index": slice_index,
                "threshold": args.threshold,
                "surface_step_size": args.surface_step_size,
                "view_elevation": args.view_elevation,
                "view_azimuth": args.view_azimuth,
                "postprocess": "evaluate.py official CPU post-processing",
                "outputs": {
                    "slice_2d": str(output_png),
                    "object_3d": str(output_3d_png) if output_3d_png else None,
                    "shift_comparison_3d": (
                        str(output_shift_3d_png) if output_shift_3d_png else None
                    ),
                },
                "per_sample": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    LOGGER.info("Saved 2D visualization: %s", output_png)
    if output_3d_png is not None:
        LOGGER.info("Saved 3D object comparison: %s", output_3d_png)
    if output_shift_3d_png is not None:
        LOGGER.info("Saved center-shift 3D comparison: %s", output_shift_3d_png)
    LOGGER.info("Saved sample metrics: %s", output_json)
    LOGGER.info(
        "Visualization pool: diff=%s | real=%s | size=%d | mode=%s",
        args.data_diff,
        args.data_real,
        sample_pool_size,
        args.sample_mode,
    )
    LOGGER.info("Selected samples: %s", ", ".join(names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
