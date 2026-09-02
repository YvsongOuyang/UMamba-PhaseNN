"""2D and 3D diagnostics for simulated HighStrain reconstructions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _masked_phase(phase: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask, np.angle(np.exp(1.0j * phase)), np.nan)


def save_slice_overview(
    *,
    intensity: np.ndarray,
    target_reciprocal_phase: np.ndarray,
    predicted_reciprocal_phase: np.ndarray,
    target_object: np.ndarray,
    predicted_object: np.ndarray,
    destination: str | Path,
    support_threshold: float = 0.1,
    target_support: np.ndarray | None = None,
    model_label: str = "HighStrain model",
    sample_label: str = "",
) -> Path:
    """Save matched center slices in reciprocal and real space."""

    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    center = intensity.shape[0] // 2
    normalized_intensity = intensity / max(float(intensity.max()), 1.0)
    reciprocal_mask = normalized_intensity > 1e-3
    target_amplitude = np.abs(target_object)
    predicted_amplitude = np.abs(predicted_object)
    target_mask = (
        np.asarray(target_support, dtype=bool) if target_support is not None
        else target_amplitude > 0.5 * max(float(target_amplitude.max()), 1e-12)
    )
    predicted_mask = predicted_amplitude > support_threshold * max(
        float(predicted_amplitude.max()), 1e-12
    )
    reciprocal_error = np.angle(
        np.exp(1.0j * (target_reciprocal_phase - predicted_reciprocal_phase))
    )

    panels = [
        (
            np.log10(np.clip(normalized_intensity[:, :, center], 1e-6, None)),
            "Measured intensity (log10)",
            "magma",
            None,
        ),
        (
            _masked_phase(target_reciprocal_phase, reciprocal_mask)[:, :, center],
            "Target reciprocal phase",
            "twilight",
            (-np.pi, np.pi),
        ),
        (
            _masked_phase(predicted_reciprocal_phase, reciprocal_mask)[:, :, center],
            "Predicted reciprocal phase",
            "twilight",
            (-np.pi, np.pi),
        ),
        (
            _masked_phase(reciprocal_error, reciprocal_mask)[:, :, center],
            "Wrapped reciprocal error",
            "coolwarm",
            (-np.pi, np.pi),
        ),
        (target_amplitude[:, :, center], "Target amplitude", "viridis",
         (0.0, max(1.0, float(target_amplitude.max())))),
        (
            predicted_amplitude[:, :, center],
            "Predicted amplitude",
            "viridis",
            (0.0, max(1.0, float(predicted_amplitude.max()))),
        ),
        (
            _masked_phase(np.angle(target_object), target_mask)[:, :, center],
            "Target real-space phase",
            "twilight",
            (-np.pi, np.pi),
        ),
        (
            _masked_phase(np.angle(predicted_object), predicted_mask)[:, :, center],
            "Predicted real-space phase",
            "twilight",
            (-np.pi, np.pi),
        ),
    ]

    figure, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for axis, (image, title, color_map, limits) in zip(axes.ravel(), panels):
        kwargs = {}
        if limits is not None:
            kwargs.update(vmin=limits[0], vmax=limits[1])
        artist = axis.imshow(image.T, origin="lower", cmap=color_map, **kwargs)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.02)
    title = f"Author simulation | {model_label}"
    if sample_label:
        title = f"{title}\n{sample_label}"
    figure.suptitle(title)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def _configure_volume_axis(
    axis: Any,
    shape: tuple[int, ...],
    elevation: float,
    azimuth: float,
) -> None:
    """Apply AutoPhaseNN coordinates, camera, and center guides."""

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
    line_style = {
        "color": "red",
        "linestyle": "--",
        "linewidth": 0.8,
        "alpha": 0.75,
    }
    axis.plot(
        [0, width - 1],
        [center_y, center_y],
        [center_z, center_z],
        **line_style,
    )
    axis.plot(
        [center_x, center_x],
        [0, height - 1],
        [center_z, center_z],
        **line_style,
    )
    axis.plot(
        [center_x, center_x],
        [center_y, center_y],
        [0, depth - 1],
        **line_style,
    )
    axis.set_xlabel("X", fontsize=7, labelpad=-1)
    axis.set_ylabel("Y", fontsize=7, labelpad=-1)
    axis.set_zlabel("Z", fontsize=7, labelpad=-1)
    axis.tick_params(labelsize=6, pad=0)
    axis.grid(True, linewidth=0.35, alpha=0.45)


def _visible_value_limits(
    panel_rows: list[list[tuple[np.ndarray, np.ndarray, str, float]]],
    columns: range,
    *,
    symmetric: bool,
    zero_minimum: bool,
) -> tuple[float, float]:
    """Resolve shared color limits from voxels visible in selected columns."""

    visible_values: list[np.ndarray] = []
    for panels in panel_rows:
        for column in columns:
            geometry, values, _title, level = panels[column]
            selected = np.asarray(values)[
                (np.asarray(geometry) > level) & np.isfinite(values)
            ]
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


def _add_volume_points(
    axis: Any,
    geometry: np.ndarray,
    values: np.ndarray,
    *,
    level: float,
    color_map: Any,
    normalizer: Any,
    max_points: int,
    point_size: float,
    alpha: float,
    elevation: float,
    azimuth: float,
) -> None:
    """Render a deterministic sample from all visible volume voxels."""

    geometry = np.asarray(geometry, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    visible_indices = np.flatnonzero((geometry > level) & np.isfinite(values))
    if visible_indices.size > max_points:
        selected_indices = np.linspace(
            0,
            visible_indices.size - 1,
            num=max_points,
            dtype=np.int64,
        )
        visible_indices = visible_indices[selected_indices]
    if visible_indices.size:
        z, y, x = np.unravel_index(visible_indices, geometry.shape)
        axis.scatter(
            x,
            y,
            z,
            c=values.reshape(-1)[visible_indices],
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
    _configure_volume_axis(
        axis,
        geometry.shape,
        elevation=elevation,
        azimuth=azimuth,
    )


def plot_five_panel_volume(
    *,
    panel_rows: list[list[tuple[np.ndarray, np.ndarray, str, float]]],
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
    absolute_zero_minimum: bool = False,
) -> None:
    """Save the AutoPhaseNN five-column full-volume comparison layout."""

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    absolute_limits = _visible_value_limits(
        panel_rows,
        range(3),
        symmetric=False,
        zero_minimum=absolute_zero_minimum,
    )
    difference_limits = _visible_value_limits(
        panel_rows,
        range(3, 5),
        symmetric=True,
        zero_minimum=False,
    )
    absolute_normalizer = Normalize(*absolute_limits)
    difference_normalizer = Normalize(*difference_limits)
    absolute_map = plt.get_cmap(absolute_color_map)
    difference_map = plt.get_cmap(difference_color_map)

    figure, axes = plt.subplots(
        len(panel_rows),
        5,
        figsize=(21.0, max(4.35 * len(panel_rows), 5.4)),
        squeeze=False,
        subplot_kw={"projection": "3d"},
    )
    for row_index, panels in enumerate(panel_rows):
        for column, (geometry, values, panel_title, level) in enumerate(panels):
            _add_volume_points(
                axes[row_index, column],
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
            )
            axes[row_index, column].set_title(
                f"{names[row_index]}\n{panel_title}",
                fontsize=8,
                pad=2,
            )

    absolute_mapper = ScalarMappable(norm=absolute_normalizer, cmap=absolute_map)
    difference_mapper = ScalarMappable(
        norm=difference_normalizer,
        cmap=difference_map,
    )
    absolute_mapper.set_array([])
    difference_mapper.set_array([])
    absolute_color_axis = figure.add_axes((0.055, 0.035, 0.52, 0.012))
    difference_color_axis = figure.add_axes((0.64, 0.035, 0.30, 0.012))
    absolute_colorbar = figure.colorbar(
        absolute_mapper,
        cax=absolute_color_axis,
        orientation="horizontal",
    )
    difference_colorbar = figure.colorbar(
        difference_mapper,
        cax=difference_color_axis,
        orientation="horizontal",
    )
    absolute_colorbar.set_label(absolute_colorbar_label, fontsize=8)
    difference_colorbar.set_label(difference_colorbar_label, fontsize=8)
    absolute_colorbar.ax.tick_params(labelsize=7)
    difference_colorbar.ax.tick_params(labelsize=7)
    figure.suptitle(
        f"{figure_title} - red guides=grid center",
        fontsize=13,
        y=0.985,
    )
    figure.subplots_adjust(
        left=0.015,
        right=0.985,
        bottom=0.105,
        top=0.89,
        wspace=0.01,
        hspace=0.12,
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=240, bbox_inches="tight")
    plt.close(figure)


def _relative_geometry(amplitude: np.ndarray) -> np.ndarray:
    """Normalize amplitude only for deciding which volume voxels are visible."""

    amplitude = np.asarray(amplitude, dtype=np.float32)
    return amplitude / max(float(amplitude.max()), 1e-12)


def _amplitude_volume_panels(
    *,
    target_object: np.ndarray,
    predicted_object_before_shift: np.ndarray,
    predicted_object_after_shift: np.ndarray,
    support_threshold: float,
    target_support: np.ndarray | None,
    amplitude_error_level: float,
) -> list[tuple[np.ndarray, np.ndarray, str, float]]:
    """Build the same five amplitude panels used by AutoPhaseNN visualization."""

    target_amplitude = np.abs(target_object).astype(np.float32, copy=False)
    before_amplitude = np.abs(predicted_object_before_shift).astype(
        np.float32, copy=False
    )
    after_amplitude = np.abs(predicted_object_after_shift).astype(
        np.float32, copy=False
    )
    if target_support is None:
        target_geometry = _relative_geometry(target_amplitude)
        target_level = support_threshold
    else:
        target_geometry = np.asarray(target_support, dtype=np.float32)
        if target_geometry.shape != target_amplitude.shape:
            raise ValueError("Target support and target object shapes differ.")
        target_level = 0.5

    shift_difference = before_amplitude - after_amplitude
    target_difference = after_amplitude - target_amplitude
    return [
        (target_geometry, target_amplitude, "Target", target_level),
        (
            _relative_geometry(before_amplitude),
            before_amplitude,
            "Prediction before center shift",
            support_threshold,
        ),
        (
            _relative_geometry(after_amplitude),
            after_amplitude,
            "Prediction after center shift",
            support_threshold,
        ),
        (
            np.abs(shift_difference),
            shift_difference,
            "Difference: before - after",
            amplitude_error_level,
        ),
        (
            np.abs(target_difference),
            target_difference,
            "Difference: after - target",
            amplitude_error_level,
        ),
    ]


def save_amplitude_volume_comparison(
    *,
    target_objects: list[np.ndarray],
    predicted_objects_after_shift: list[np.ndarray],
    destination: str | Path,
    support_threshold: float,
    names: list[str] | None = None,
    predicted_objects_before_shift: list[np.ndarray] | None = None,
    target_supports: list[np.ndarray | None] | None = None,
    model_label: str = "HighStrain model",
    amplitude_error_level: float = 0.05,
    max_volume_points: int = 7000,
    volume_point_size: float = 3.0,
    volume_alpha: float = 0.4,
    view_elevation: float = 25.0,
    view_azimuth: float = 35.0,
) -> Path:
    """Save a shared-limit, five-column amplitude volume comparison."""

    sample_count = len(target_objects)
    if sample_count < 1 or len(predicted_objects_after_shift) != sample_count:
        raise ValueError(
            "Target and prediction lists must have equal nonzero lengths."
        )
    if predicted_objects_before_shift is None:
        predicted_objects_before_shift = predicted_objects_after_shift
    if target_supports is None:
        target_supports = [None] * sample_count
    if names is None:
        names = [f"Sample {index + 1}" for index in range(sample_count)]
    if any(
        len(values) != sample_count
        for values in (predicted_objects_before_shift, target_supports, names)
    ):
        raise ValueError("Every comparison list must contain the same samples.")

    panel_rows = [
        _amplitude_volume_panels(
            target_object=target_object,
            predicted_object_before_shift=before_object,
            predicted_object_after_shift=after_object,
            support_threshold=support_threshold,
            target_support=target_support,
            amplitude_error_level=amplitude_error_level,
        )
        for target_object, before_object, after_object, target_support in zip(
            target_objects,
            predicted_objects_before_shift,
            predicted_objects_after_shift,
            target_supports,
        )
    ]
    destination = Path(destination).expanduser().resolve()
    plot_five_panel_volume(
        panel_rows=panel_rows,
        names=names,
        output_png=destination,
        figure_title=f"Real-space amplitude 3D volumes | {model_label}",
        absolute_color_map="viridis",
        difference_color_map="coolwarm",
        absolute_colorbar_label="Amplitude",
        difference_colorbar_label="Signed amplitude difference",
        max_points=max_volume_points,
        point_size=volume_point_size,
        alpha=volume_alpha,
        elevation=view_elevation,
        azimuth=view_azimuth,
        absolute_zero_minimum=True,
    )
    return destination


def save_volume_overview(
    *,
    intensity: np.ndarray,
    target_object: np.ndarray,
    predicted_object: np.ndarray,
    destination: str | Path,
    support_threshold: float = 0.1,
    max_surface_points: int = 7000,
    max_diffraction_points: int = 5000,
    target_support: np.ndarray | None = None,
    model_label: str = "HighStrain model",
    sample_label: str = "",
    predicted_object_before_shift: np.ndarray | None = None,
    amplitude_error_level: float = 0.05,
) -> Path:
    """Save one AutoPhaseNN-style five-column amplitude volume comparison."""

    del intensity, max_diffraction_points
    return save_amplitude_volume_comparison(
        target_objects=[target_object],
        predicted_objects_before_shift=[
            predicted_object
            if predicted_object_before_shift is None
            else predicted_object_before_shift
        ],
        predicted_objects_after_shift=[predicted_object],
        target_supports=[target_support],
        names=[sample_label or "Sample"],
        destination=destination,
        support_threshold=support_threshold,
        model_label=model_label,
        amplitude_error_level=amplitude_error_level,
        max_volume_points=max_surface_points,
    )
