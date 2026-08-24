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
DEFAULT_VISION_ROOT = PROJECT_DIR / "vision"
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


def apply_default_output_paths(args: argparse.Namespace) -> None:
    """Fill unspecified image paths inside the model-specific vision directory."""

    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else DEFAULT_VISION_ROOT / f"vision_{args.model_variant}"
    )
    for argument, filename in DEFAULT_OUTPUT_FILENAMES.items():
        if not getattr(args, argument):
            setattr(args, argument, str(output_dir / filename))


def wrap_phase(phase: np.ndarray) -> np.ndarray:
    """Wrap phase to [-pi, pi] for cyclic color mapping."""

    return np.arctan2(np.sin(phase), np.cos(phase))


def volume_tensor(volume: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert one 3D NumPy volume to the model's ``(1, 1, D, H, W)`` layout."""

    contiguous = np.ascontiguousarray(volume[None, None], dtype=np.float32)
    return torch.from_numpy(contiguous).to(device=device)


def object_farfield(
    amplitude: np.ndarray,
    phase: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return centered diffraction modulus and phase using the model FFT convention."""

    amplitude_tensor = volume_tensor(amplitude, device)
    phase_tensor = volume_tensor(phase, device)
    object_tensor = torch.polar(amplitude_tensor, phase_tensor)
    shifted_object = torch.fft.ifftshift(object_tensor, dim=(-3, -2, -1))
    farfield = torch.fft.fftn(shifted_object, dim=(-3, -2, -1))
    farfield = torch.fft.fftshift(farfield, dim=(-3, -2, -1))
    modulus = torch.abs(farfield).to(torch.float32)
    farfield_phase = torch.angle(farfield).to(torch.float32)
    return (
        modulus.detach().cpu().numpy()[0, 0],
        farfield_phase.detach().cpu().numpy()[0, 0],
    )


def normalize_by_reference(
    reference: np.ndarray,
    *volumes: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Scale reciprocal-space volumes by one reference maximum."""

    scale = max(float(np.nanmax(np.abs(reference))), np.finfo(np.float32).eps)
    return tuple(np.asarray(volume, dtype=np.float32) / scale for volume in volumes)


def masked_wrapped_phase(
    phase: np.ndarray,
    normalized_modulus: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Mask undefined low-modulus Fourier phase and wrap valid values."""

    return np.where(
        normalized_modulus > threshold,
        wrap_phase(phase),
        np.nan,
    ).astype(np.float32)


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


def plot_reciprocal_slice_rows(
    rows: list[list[np.ndarray]],
    names: list[str],
    output_png: Path,
) -> None:
    """Save reciprocal-space modulus and derived Fourier-phase center slices."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_titles = [
        "Measured modulus (log10, normalized)",
        "Predicted modulus (log10, measured scale)",
        "Measured - predicted modulus",
        "GT-derived Fourier phase (rad)",
        "Prediction-derived Fourier phase (rad)",
        "Wrapped GT - prediction Fourier phase",
    ]
    color_maps = ["magma", "magma", "coolwarm", "twilight", "twilight", "coolwarm"]
    sample_count = len(rows)
    fig = plt.figure(figsize=(3.5 * sample_count + 1.2, 14.5))
    grid = fig.add_gridspec(
        len(row_titles),
        sample_count + 1,
        width_ratios=[1.0] * sample_count + [0.045],
        wspace=0.08,
        hspace=0.16,
    )

    for row_index, row_title in enumerate(row_titles):
        row_images = [sample[row_index] for sample in rows]
        finite_values = [image[np.isfinite(image)] for image in row_images]
        finite_values = [values for values in finite_values if values.size]
        if row_index >= 3:
            vmin, vmax = -float(np.pi), float(np.pi)
        elif row_index == 2 and finite_values:
            bound = max(
                max(float(np.max(np.abs(values))) for values in finite_values),
                np.finfo(np.float32).eps,
            )
            vmin, vmax = -bound, bound
        elif finite_values:
            vmin = min(float(values.min()) for values in finite_values)
            vmax = max(float(values.max()) for values in finite_values)
            if vmin == vmax:
                vmax = vmin + np.finfo(np.float32).eps
        else:
            vmin, vmax = None, None

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
            if not np.any(np.isfinite(image)):
                axis.text(
                    0.5,
                    0.5,
                    "Unavailable without real-space ground truth",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=7,
                    wrap=True,
                )
            if column == 0:
                axis.set_ylabel(row_title, fontsize=8)
            if row_index == 0:
                axis.set_title(names[column], fontsize=8)
        if last_image is not None:
            color_axis = fig.add_subplot(grid[row_index, -1])
            colorbar = fig.colorbar(last_image, cax=color_axis, format="%.2f")
            colorbar.ax.tick_params(labelsize=6)

    fig.suptitle(
        "Reciprocal-space modulus and derived Fourier phase",
        fontsize=13,
        y=0.997,
    )
    fig.subplots_adjust(left=0.16, right=0.94, bottom=0.02, top=0.955)
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


def extract_scalar_surface(
    geometry: np.ndarray,
    values: np.ndarray,
    level: float,
    step_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Extract an isosurface and linearly interpolate scalar colors per face."""

    from skimage.measure import marching_cubes

    geometry = np.asarray(geometry, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    minimum = float(np.nanmin(geometry))
    maximum = float(np.nanmax(geometry))
    if not minimum < level < maximum:
        return None

    vertices, faces, _normals, _surface_values = marching_cubes(
        geometry,
        level=level,
        step_size=step_size,
        allow_degenerate=False,
    )
    volume_indices = np.rint(vertices).astype(np.int64)
    for axis, size in enumerate(geometry.shape):
        volume_indices[:, axis] = np.clip(volume_indices[:, axis], 0, size - 1)
    vertex_values = values[tuple(volume_indices[:, axis] for axis in range(3))]
    face_values = np.mean(vertex_values[faces], axis=1)
    return vertices[:, [2, 1, 0]], faces, face_values.astype(np.float32)


def configure_3d_axis(
    axis: Any,
    shape: tuple[int, ...],
    elevation: float,
    azimuth: float,
) -> None:
    """Apply shared coordinates, camera, and center guides to a 3D axis."""

    depth, height, width = shape
    axis.set_xlim(0, max(width - 1, 1))
    axis.set_ylim(0, max(height - 1, 1))
    axis.set_zlim(0, max(depth - 1, 1))
    axis.set_box_aspect((width, height, depth))
    axis.view_init(elev=elevation, azim=azimuth)
    center_x, center_y, center_z = width / 2.0, height / 2.0, depth / 2.0
    axis.scatter(
        [center_x],
        [center_y],
        [center_z],
        color="red",
        marker="x",
        s=32,
        linewidths=1.5,
        depthshade=False,
    )
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

    configure_3d_axis(axis, shape, elevation=elevation, azimuth=azimuth)


def add_scalar_surface(
    axis: Any,
    geometry: np.ndarray | None,
    values: np.ndarray | None,
    level: float,
    step_size: int,
    elevation: float,
    azimuth: float,
    color_map: Any,
    normalizer: Any,
    empty_message: str,
) -> None:
    """Draw a scalar-colored isosurface for a signed 3D error volume."""

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if geometry is None or values is None:
        axis.text2D(0.5, 0.5, empty_message, transform=axis.transAxes, ha="center")
        shape = (64, 64, 64)
        surface = None
    else:
        shape = geometry.shape
        surface = extract_scalar_surface(geometry, values, level, step_size)
        if surface is None:
            axis.text2D(
                0.5,
                0.5,
                "No error surface at configured level",
                transform=axis.transAxes,
                ha="center",
            )

    if surface is not None:
        vertices, faces, face_values = surface
        mesh = Poly3DCollection(
            vertices[faces],
            facecolors=color_map(normalizer(face_values)),
            edgecolors="none",
            linewidths=0.0,
            alpha=0.92,
        )
        axis.add_collection3d(mesh)

    configure_3d_axis(axis, shape, elevation=elevation, azimuth=azimuth)


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


def plot_3d_error_comparison(
    error_rows: list[list[tuple[np.ndarray | None, np.ndarray | None, str, float]]],
    names: list[str],
    output_png: Path,
    step_size: int,
    elevation: float,
    azimuth: float,
) -> None:
    """Save signed real-space amplitude and wrapped-phase 3D error surfaces."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    amplitude_values = [
        panels[0][1]
        for panels in error_rows
        if panels[0][1] is not None and np.any(np.isfinite(panels[0][1]))
    ]
    amplitude_bound = max(
        (float(np.nanmax(np.abs(values))) for values in amplitude_values),
        default=np.finfo(np.float32).eps,
    )
    phase_values = [
        panels[1][1]
        for panels in error_rows
        if panels[1][1] is not None and np.any(np.isfinite(panels[1][1]))
    ]
    phase_bound = min(
        max(
            (float(np.nanmax(np.abs(values))) for values in phase_values),
            default=np.finfo(np.float32).eps,
        ),
        float(np.pi),
    )
    amplitude_normalizer = Normalize(vmin=-amplitude_bound, vmax=amplitude_bound)
    phase_normalizer = Normalize(vmin=-phase_bound, vmax=phase_bound)
    color_map = plt.get_cmap("coolwarm")

    row_count = len(error_rows)
    fig, axes = plt.subplots(
        row_count,
        2,
        figsize=(10.0, max(4.7 * row_count, 5.2)),
        squeeze=False,
        subplot_kw={"projection": "3d"},
    )
    for row_index, panels in enumerate(error_rows):
        for column, (geometry, values, panel_title, level) in enumerate(panels):
            axis = axes[row_index, column]
            add_scalar_surface(
                axis,
                geometry,
                values,
                level=level,
                step_size=step_size,
                elevation=elevation,
                azimuth=azimuth,
                color_map=color_map,
                normalizer=(amplitude_normalizer if column == 0 else phase_normalizer),
                empty_message="Unavailable without real-space ground truth",
            )
            axis.set_title(f"{names[row_index]}\n{panel_title}", fontsize=9, pad=3)

    amplitude_mapper = ScalarMappable(
        norm=amplitude_normalizer,
        cmap=color_map,
    )
    phase_mapper = ScalarMappable(norm=phase_normalizer, cmap=color_map)
    amplitude_mapper.set_array([])
    phase_mapper.set_array([])
    amplitude_color_axis = fig.add_axes((0.07, 0.035, 0.36, 0.015))
    phase_color_axis = fig.add_axes((0.52, 0.035, 0.36, 0.015))
    amplitude_colorbar = fig.colorbar(
        amplitude_mapper,
        cax=amplitude_color_axis,
        orientation="horizontal",
    )
    phase_colorbar = fig.colorbar(
        phase_mapper,
        cax=phase_color_axis,
        orientation="horizontal",
    )
    amplitude_colorbar.set_label(
        "Signed amplitude error: true - prediction", fontsize=8
    )
    phase_colorbar.set_label(
        "Wrapped phase error: true - prediction (rad)",
        fontsize=8,
    )
    amplitude_colorbar.ax.tick_params(labelsize=7)
    phase_colorbar.ax.tick_params(labelsize=7)
    fig.suptitle(
        "Post-processed 3D reconstruction errors — red guides=volume center",
        fontsize=12,
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        bottom=0.105,
        top=0.89,
        wspace=0.02,
        hspace=0.12,
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def collect_panel_value_limits(
    panel_rows: list[list[tuple[np.ndarray | None, np.ndarray | None, str, float]]],
    columns: range,
    *,
    symmetric: bool,
    zero_minimum: bool,
) -> tuple[float, float]:
    """Resolve shared color limits from visible voxels in selected columns."""

    visible_values: list[np.ndarray] = []
    for panels in panel_rows:
        for column in columns:
            geometry, values, _title, level = panels[column]
            if geometry is None or values is None:
                continue
            mask = (np.asarray(geometry) > level) & np.isfinite(values)
            selected = np.asarray(values)[mask]
            if selected.size:
                visible_values.append(selected)

    if not visible_values:
        epsilon = float(np.finfo(np.float32).eps)
        return (-epsilon, epsilon) if symmetric else (0.0, epsilon)
    minimum = min(float(values.min()) for values in visible_values)
    maximum = max(float(values.max()) for values in visible_values)
    if symmetric:
        bound = max(abs(minimum), abs(maximum), np.finfo(np.float32).eps)
        return -bound, bound
    if zero_minimum:
        minimum = 0.0
    if minimum == maximum:
        maximum = minimum + np.finfo(np.float32).eps
    return minimum, maximum


def add_volume_points(
    axis: Any,
    geometry: np.ndarray | None,
    values: np.ndarray | None,
    level: float,
    color_map: Any,
    normalizer: Any,
    max_points: int,
    point_size: float,
    alpha: float,
    elevation: float,
    azimuth: float,
    empty_message: str,
) -> None:
    """Render a deterministic sample of visible 3D voxels with scalar colors."""

    if geometry is None or values is None:
        axis.text2D(0.5, 0.5, empty_message, transform=axis.transAxes, ha="center")
        shape = (64, 64, 64)
    else:
        geometry = np.asarray(geometry, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        shape = geometry.shape
        visible = (geometry > level) & np.isfinite(values)
        flat_indices = np.flatnonzero(visible)
        if flat_indices.size > max_points:
            sample_indices = np.linspace(
                0,
                flat_indices.size - 1,
                num=max_points,
                dtype=np.int64,
            )
            flat_indices = flat_indices[sample_indices]
        if flat_indices.size:
            z, y, x = np.unravel_index(flat_indices, shape)
            axis.scatter(
                x,
                y,
                z,
                c=values.reshape(-1)[flat_indices],
                cmap=color_map,
                norm=normalizer,
                s=point_size,
                alpha=alpha,
                linewidths=0.0,
                depthshade=False,
                rasterized=True,
            )
        else:
            axis.text2D(
                0.5,
                0.5,
                "No voxels above display threshold",
                transform=axis.transAxes,
                ha="center",
            )
    configure_3d_axis(axis, shape, elevation=elevation, azimuth=azimuth)


def plot_five_panel_volume(
    panel_rows: list[list[tuple[np.ndarray | None, np.ndarray | None, str, float]]],
    names: list[str],
    output_png: Path,
    figure_title: str,
    absolute_color_map: str,
    difference_color_map: str,
    absolute_colorbar_label: str,
    difference_colorbar_label: str,
    max_points: int,
    point_size: float,
    alpha: float,
    elevation: float,
    azimuth: float,
    absolute_limits: tuple[float, float] | None = None,
    difference_limits: tuple[float, float] | None = None,
    absolute_zero_minimum: bool = False,
) -> None:
    """Save a five-column 3D volume comparison for every selected sample."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    if absolute_limits is None:
        absolute_limits = collect_panel_value_limits(
            panel_rows,
            range(3),
            symmetric=False,
            zero_minimum=absolute_zero_minimum,
        )
    if difference_limits is None:
        difference_limits = collect_panel_value_limits(
            panel_rows,
            range(3, 5),
            symmetric=True,
            zero_minimum=False,
        )

    absolute_normalizer = Normalize(*absolute_limits)
    difference_normalizer = Normalize(*difference_limits)
    absolute_map = plt.get_cmap(absolute_color_map)
    difference_map = plt.get_cmap(difference_color_map)

    row_count = len(panel_rows)
    fig, axes = plt.subplots(
        row_count,
        5,
        figsize=(21.0, max(4.35 * row_count, 5.4)),
        squeeze=False,
        subplot_kw={"projection": "3d"},
    )
    for row_index, panels in enumerate(panel_rows):
        for column, (geometry, values, panel_title, level) in enumerate(panels):
            axis = axes[row_index, column]
            add_volume_points(
                axis,
                geometry,
                values,
                level=level,
                color_map=absolute_map if column < 3 else difference_map,
                normalizer=(
                    absolute_normalizer if column < 3 else difference_normalizer
                ),
                max_points=max_points,
                point_size=point_size,
                alpha=alpha,
                elevation=elevation,
                azimuth=azimuth,
                empty_message="Unavailable without real-space ground truth",
            )
            axis.set_title(f"{names[row_index]}\n{panel_title}", fontsize=8, pad=2)

    absolute_mapper = ScalarMappable(norm=absolute_normalizer, cmap=absolute_map)
    difference_mapper = ScalarMappable(
        norm=difference_normalizer,
        cmap=difference_map,
    )
    absolute_mapper.set_array([])
    difference_mapper.set_array([])
    absolute_color_axis = fig.add_axes((0.055, 0.035, 0.52, 0.012))
    difference_color_axis = fig.add_axes((0.64, 0.035, 0.30, 0.012))
    absolute_colorbar = fig.colorbar(
        absolute_mapper,
        cax=absolute_color_axis,
        orientation="horizontal",
    )
    difference_colorbar = fig.colorbar(
        difference_mapper,
        cax=difference_color_axis,
        orientation="horizontal",
    )
    absolute_colorbar.set_label(absolute_colorbar_label, fontsize=8)
    difference_colorbar.set_label(difference_colorbar_label, fontsize=8)
    absolute_colorbar.ax.tick_params(labelsize=7)
    difference_colorbar.ax.tick_params(labelsize=7)
    fig.suptitle(
        f"{figure_title} — red guides=grid center",
        fontsize=13,
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.015,
        right=0.985,
        bottom=0.105,
        top=0.89,
        wspace=0.01,
        hspace=0.12,
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
            "/data_ssd/oyys/autophasenn/autophasenn_pipeline_output/mamba_skip_scratch_bs4_lr1e-3_20260823_155916/checkpoint_best.pt"
        ),
    )
    parser.add_argument("--data-dir", default="/data_ssd/oyys/autophasenn/")
    parser.add_argument("--data-diff", default="val_diff.npy")
    parser.add_argument("--data-real", default="val_real.npy")
    parser.add_argument("--shape", type=int, default=64)
    parser.add_argument(
        "--model-variant",
        choices=MODEL_VARIANTS,
        default="mamba_skip",
        help="Network architecture variant.",
    )
    parser.add_argument("--dtype-diff", default="float32")
    parser.add_argument("--dtype-real", default="complex64")
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Visualization directory; empty uses "
            "<project>/vision/vision_<model-variant>."
        ),
    )
    parser.add_argument(
        "--output-png",
        default="",
        help="Center-slice PNG path.",
    )
    parser.add_argument(
        "--output-3d-png",
        default="",
        help="True/predicted 3D comparison path; pass none to disable.",
    )
    parser.add_argument(
        "--output-shift-3d-png",
        default="",
        help="Prediction before/after center-shift 3D comparison; pass none to disable.",
    )
    parser.add_argument(
        "--output-error-3d-png",
        default="",
        help="Real-space amplitude/phase 3D error path; pass none to disable.",
    )
    parser.add_argument(
        "--output-reciprocal-2d-png",
        default="",
        help="Reciprocal modulus/Fourier-phase slice path; pass none to disable.",
    )
    parser.add_argument(
        "--output-reciprocal-3d-png",
        default="",
        help="Reciprocal modulus surfaces colored by Fourier phase; pass none to disable.",
    )
    parser.add_argument(
        "--output-amplitude-3d-png",
        default="",
        help="Five-panel real-space amplitude volume path; pass none to disable.",
    )
    parser.add_argument(
        "--output-phase-3d-png",
        default="",
        help="Five-panel real-space phase volume path; pass none to disable.",
    )
    parser.add_argument(
        "--output-diffraction-3d-png",
        default="",
        help="Five-panel diffraction-modulus volume path; pass none to disable.",
    )
    parser.add_argument(
        "--output-diffraction-phase-3d-png",
        default="",
        help="Five-panel derived diffraction-phase volume path; pass none to disable.",
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
    parser.add_argument(
        "--amplitude-error-level",
        type=float,
        default=0.05,
        help="Absolute-amplitude-error isosurface level for the 3D error plot.",
    )
    parser.add_argument(
        "--reciprocal-surface-level",
        type=float,
        default=0.02,
        help="Diffraction-modulus isosurface level relative to measured maximum.",
    )
    parser.add_argument(
        "--reciprocal-phase-threshold",
        type=float,
        default=0.02,
        help="Hide Fourier phase below this relative diffraction modulus.",
    )
    parser.add_argument(
        "--diffraction-difference-threshold",
        type=float,
        default=1e-6,
        help="Minimum absolute normalized diffraction difference shown in 3D.",
    )
    parser.add_argument(
        "--max-volume-points",
        type=int,
        default=12000,
        help="Maximum visible voxels rendered in each five-panel 3D subplot.",
    )
    parser.add_argument("--volume-point-size", type=float, default=3.0)
    parser.add_argument("--volume-alpha", type=float, default=0.4)
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

    apply_default_output_paths(args)

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
    if args.amplitude_error_level <= 0:
        parser.error("--amplitude-error-level must be positive.")
    if not 0 < args.reciprocal_surface_level < 1:
        parser.error("--reciprocal-surface-level must be between 0 and 1.")
    if not 0 <= args.reciprocal_phase_threshold < 1:
        parser.error("--reciprocal-phase-threshold must be in [0, 1).")
    if args.diffraction_difference_threshold <= 0:
        parser.error("--diffraction-difference-threshold must be positive.")
    if args.max_volume_points <= 0:
        parser.error("--max-volume-points must be positive.")
    if args.volume_point_size <= 0:
        parser.error("--volume-point-size must be positive.")
    if not 0 < args.volume_alpha <= 1:
        parser.error("--volume-alpha must be in (0, 1].")
    return args


@torch.inference_mode()
def main() -> int:
    """Run inference and write 2D, 3D, shift-comparison, and JSON outputs."""

    args = parse_args()
    configure_logging(args.log_level)

    output_png = optional_output_path(args.output_png)
    output_3d_png = optional_output_path(args.output_3d_png)
    output_shift_3d_png = optional_output_path(args.output_shift_3d_png)
    output_error_3d_png = optional_output_path(args.output_error_3d_png)
    output_reciprocal_2d_png = optional_output_path(args.output_reciprocal_2d_png)
    output_reciprocal_3d_png = optional_output_path(args.output_reciprocal_3d_png)
    output_amplitude_3d_png = optional_output_path(args.output_amplitude_3d_png)
    output_phase_3d_png = optional_output_path(args.output_phase_3d_png)
    output_diffraction_3d_png = optional_output_path(args.output_diffraction_3d_png)
    output_diffraction_phase_3d_png = optional_output_path(
        args.output_diffraction_phase_3d_png
    )
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
    error_3d_rows: list[
        list[tuple[np.ndarray | None, np.ndarray | None, str, float]]
    ] = []
    reciprocal_slice_rows: list[list[np.ndarray]] = []
    reciprocal_3d_rows: list[list[tuple[np.ndarray | None, np.ndarray | None, str]]] = (
        []
    )
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

        pred_before_farfield_modulus, pred_before_farfield_phase = object_farfield(
            pred_amp_before,
            pred_phi_before,
            device=device,
        )
        pred_farfield_modulus, pred_farfield_phase = object_farfield(
            pred_amp_post,
            pred_phi_post,
            device=device,
        )
        (
            measured_modulus_norm,
            pred_modulus_norm,
            pred_before_modulus_norm,
            pred_derived_modulus_norm,
        ) = normalize_by_reference(
            diff_np,
            diff_np,
            pred_diff_np,
            pred_before_farfield_modulus,
            pred_farfield_modulus,
        )
        pred_fourier_phase_display = masked_wrapped_phase(
            pred_farfield_phase,
            pred_derived_modulus_norm,
            threshold=args.reciprocal_phase_threshold,
        )

        true_farfield_phase: np.ndarray | None = None
        if true_amp_post is not None and true_phi_post is not None:
            _true_farfield_modulus, true_farfield_phase = object_farfield(
                true_amp_post,
                true_phi_post,
                device=device,
            )
            true_fourier_phase_display = masked_wrapped_phase(
                true_farfield_phase,
                measured_modulus_norm,
                threshold=args.reciprocal_phase_threshold,
            )
            reciprocal_phase_valid = (
                measured_modulus_norm > args.reciprocal_phase_threshold
            ) & (pred_derived_modulus_norm > args.reciprocal_phase_threshold)
            reciprocal_phase_difference = np.where(
                reciprocal_phase_valid,
                wrap_phase(true_farfield_phase - pred_farfield_phase),
                np.nan,
            ).astype(np.float32)
        else:
            unavailable = np.full_like(diff_np, np.nan, dtype=np.float32)
            true_fourier_phase_display = unavailable
            reciprocal_phase_difference = unavailable.copy()

        reciprocal_slice_rows.append(
            [
                np.log10(np.clip(measured_modulus_norm[:, :, slice_index], 1e-6, None)),
                np.log10(np.clip(pred_modulus_norm[:, :, slice_index], 1e-6, None)),
                measured_modulus_norm[:, :, slice_index]
                - pred_modulus_norm[:, :, slice_index],
                true_fourier_phase_display[:, :, slice_index],
                pred_fourier_phase_display[:, :, slice_index],
                reciprocal_phase_difference[:, :, slice_index],
            ]
        )
        reciprocal_3d_rows.append(
            [
                (
                    measured_modulus_norm,
                    true_farfield_phase,
                    "Measured modulus + GT-derived Fourier phase",
                ),
                (
                    pred_modulus_norm,
                    pred_farfield_phase,
                    "Predicted modulus + prediction-derived Fourier phase",
                ),
            ]
        )

        if true_amp_post is not None and true_phi_post is not None:
            amplitude_error = true_amp_post - pred_amp_post
            phase_intersection = (true_amp_post > args.threshold) & (
                pred_amp_post > args.threshold
            )
            phase_error = np.where(
                phase_intersection,
                wrap_phase(true_phi_post - pred_phi_post),
                0.0,
            ).astype(np.float32)
            phase_error_geometry = np.minimum(true_amp_post, pred_amp_post)
            error_3d_rows.append(
                [
                    (
                        np.abs(amplitude_error),
                        amplitude_error,
                        "Amplitude error surface: true - prediction",
                        args.amplitude_error_level,
                    ),
                    (
                        phase_error_geometry,
                        phase_error,
                        "Wrapped phase error on support intersection",
                        args.threshold,
                    ),
                ]
            )
        else:
            error_3d_rows.append(
                [
                    (
                        None,
                        None,
                        "Amplitude error unavailable",
                        args.amplitude_error_level,
                    ),
                    (None, None, "Phase error unavailable", args.threshold),
                ]
            )

        amplitude_shift_difference = pred_amp_before - pred_amp_post
        amplitude_target_difference = (
            pred_amp_post - true_amp_post if true_amp_post is not None else None
        )
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
                    (
                        np.abs(amplitude_target_difference)
                        if amplitude_target_difference is not None
                        else None
                    ),
                    amplitude_target_difference,
                    "Difference: after - target",
                    args.amplitude_error_level,
                ),
            ]
        )

        phase_shift_geometry = np.minimum(pred_amp_before, pred_amp_post)
        phase_shift_difference = wrap_phase(pred_phi_before - pred_phi_post)
        if true_amp_post is not None and true_phi_post is not None:
            phase_target_geometry = np.minimum(pred_amp_post, true_amp_post)
            phase_target_difference: np.ndarray | None = wrap_phase(
                pred_phi_post - true_phi_post
            )
        else:
            phase_target_geometry = None
            phase_target_difference = None
        phase_volume_rows.append(
            [
                (
                    true_amp_post,
                    wrap_phase(true_phi_post) if true_phi_post is not None else None,
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
                    phase_shift_difference,
                    "Wrapped difference: before - after",
                    args.threshold,
                ),
                (
                    phase_target_geometry,
                    phase_target_difference,
                    "Wrapped difference: after - target",
                    args.threshold,
                ),
            ]
        )

        diffraction_shift_difference = (
            pred_before_modulus_norm - pred_derived_modulus_norm
        )
        diffraction_target_difference = (
            pred_derived_modulus_norm - measured_modulus_norm
        )
        diffraction_volume_rows.append(
            [
                (
                    measured_modulus_norm,
                    np.log10(np.clip(measured_modulus_norm, 1e-6, None)),
                    "Target measured modulus",
                    args.reciprocal_surface_level,
                ),
                (
                    pred_before_modulus_norm,
                    np.log10(np.clip(pred_before_modulus_norm, 1e-6, None)),
                    "Prediction before center shift",
                    args.reciprocal_surface_level,
                ),
                (
                    pred_derived_modulus_norm,
                    np.log10(np.clip(pred_derived_modulus_norm, 1e-6, None)),
                    "Prediction after center shift",
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
            pred_derived_modulus_norm,
        )
        diffraction_phase_shift_difference = wrap_phase(
            pred_before_farfield_phase - pred_farfield_phase
        )
        if true_farfield_phase is not None:
            diffraction_phase_target_geometry: np.ndarray | None = np.minimum(
                pred_derived_modulus_norm,
                measured_modulus_norm,
            )
            diffraction_phase_target_difference: np.ndarray | None = wrap_phase(
                pred_farfield_phase - true_farfield_phase
            )
        else:
            diffraction_phase_target_geometry = None
            diffraction_phase_target_difference = None
        diffraction_phase_volume_rows.append(
            [
                (
                    measured_modulus_norm if true_farfield_phase is not None else None,
                    true_farfield_phase,
                    "Target phase derived from real-space target",
                    args.reciprocal_phase_threshold,
                ),
                (
                    pred_before_modulus_norm,
                    pred_before_farfield_phase,
                    "Prediction before center shift",
                    args.reciprocal_phase_threshold,
                ),
                (
                    pred_derived_modulus_norm,
                    pred_farfield_phase,
                    "Prediction after center shift",
                    args.reciprocal_phase_threshold,
                ),
                (
                    diffraction_phase_shift_geometry,
                    diffraction_phase_shift_difference,
                    "Wrapped difference: before - after",
                    args.reciprocal_phase_threshold,
                ),
                (
                    diffraction_phase_target_geometry,
                    diffraction_phase_target_difference,
                    "Wrapped difference: after - target",
                    args.reciprocal_phase_threshold,
                ),
            ]
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
    if output_error_3d_png is not None:
        plot_3d_error_comparison(
            error_3d_rows,
            names,
            output_error_3d_png,
            step_size=args.surface_step_size,
            elevation=args.view_elevation,
            azimuth=args.view_azimuth,
        )
    if output_reciprocal_2d_png is not None:
        plot_reciprocal_slice_rows(
            reciprocal_slice_rows,
            names,
            output_reciprocal_2d_png,
        )
    if output_reciprocal_3d_png is not None:
        plot_3d_comparison(
            reciprocal_3d_rows,
            names,
            output_reciprocal_3d_png,
            threshold=args.reciprocal_surface_level,
            step_size=args.surface_step_size,
            elevation=args.view_elevation,
            azimuth=args.view_azimuth,
            figure_title="Reciprocal-space modulus surfaces and Fourier phase",
        )
    if output_amplitude_3d_png is not None:
        plot_five_panel_volume(
            amplitude_volume_rows,
            names,
            output_amplitude_3d_png,
            figure_title="Real-space amplitude 3D volumes",
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
    if output_phase_3d_png is not None:
        plot_five_panel_volume(
            phase_volume_rows,
            names,
            output_phase_3d_png,
            figure_title="Real-space phase 3D volumes",
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
    if output_diffraction_3d_png is not None:
        plot_five_panel_volume(
            diffraction_volume_rows,
            names,
            output_diffraction_3d_png,
            figure_title="Diffraction-modulus 3D volumes",
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
    if output_diffraction_phase_3d_png is not None:
        plot_five_panel_volume(
            diffraction_phase_volume_rows,
            names,
            output_diffraction_phase_3d_png,
            figure_title="Derived diffraction-phase 3D volumes",
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
                "amplitude_error_level": args.amplitude_error_level,
                "reciprocal_surface_level": args.reciprocal_surface_level,
                "reciprocal_phase_threshold": args.reciprocal_phase_threshold,
                "diffraction_difference_threshold": (
                    args.diffraction_difference_threshold
                ),
                "max_volume_points": args.max_volume_points,
                "volume_point_size": args.volume_point_size,
                "volume_alpha": args.volume_alpha,
                "view_elevation": args.view_elevation,
                "view_azimuth": args.view_azimuth,
                "postprocess": "evaluate.py official CPU post-processing",
                "outputs": {
                    "slice_2d": str(output_png),
                    "object_3d": str(output_3d_png) if output_3d_png else None,
                    "shift_comparison_3d": (
                        str(output_shift_3d_png) if output_shift_3d_png else None
                    ),
                    "realspace_error_3d": (
                        str(output_error_3d_png) if output_error_3d_png else None
                    ),
                    "reciprocal_2d": (
                        str(output_reciprocal_2d_png)
                        if output_reciprocal_2d_png
                        else None
                    ),
                    "reciprocal_3d": (
                        str(output_reciprocal_3d_png)
                        if output_reciprocal_3d_png
                        else None
                    ),
                    "amplitude_3d": (
                        str(output_amplitude_3d_png)
                        if output_amplitude_3d_png
                        else None
                    ),
                    "phase_3d": (
                        str(output_phase_3d_png) if output_phase_3d_png else None
                    ),
                    "diffraction_3d": (
                        str(output_diffraction_3d_png)
                        if output_diffraction_3d_png
                        else None
                    ),
                    "diffraction_phase_3d": (
                        str(output_diffraction_phase_3d_png)
                        if output_diffraction_phase_3d_png
                        else None
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
    if output_error_3d_png is not None:
        LOGGER.info("Saved real-space 3D errors: %s", output_error_3d_png)
    if output_reciprocal_2d_png is not None:
        LOGGER.info("Saved reciprocal-space slices: %s", output_reciprocal_2d_png)
    if output_reciprocal_3d_png is not None:
        LOGGER.info("Saved reciprocal-space 3D surfaces: %s", output_reciprocal_3d_png)
    if output_amplitude_3d_png is not None:
        LOGGER.info("Saved five-panel amplitude 3D volume: %s", output_amplitude_3d_png)
    if output_phase_3d_png is not None:
        LOGGER.info("Saved five-panel phase 3D volume: %s", output_phase_3d_png)
    if output_diffraction_3d_png is not None:
        LOGGER.info(
            "Saved five-panel diffraction-modulus 3D volume: %s",
            output_diffraction_3d_png,
        )
    if output_diffraction_phase_3d_png is not None:
        LOGGER.info(
            "Saved five-panel derived diffraction-phase 3D volume: %s",
            output_diffraction_phase_3d_png,
        )
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
