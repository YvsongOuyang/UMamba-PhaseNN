"""2D and 3D diagnostics for simulated HighStrain reconstructions."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from scipy.ndimage import binary_erosion


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
) -> Path:
    """Save matched center slices in reciprocal and real space."""

    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    center = intensity.shape[0] // 2
    normalized_intensity = intensity / max(float(intensity.max()), 1.0)
    reciprocal_mask = normalized_intensity > 1e-3
    target_amplitude = np.abs(target_object)
    predicted_amplitude = np.abs(predicted_object)
    target_mask = target_amplitude > 0.5 * max(float(target_amplitude.max()), 1e-12)
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
        (target_amplitude[:, :, center], "Target amplitude", "viridis", (0.0, 1.0)),
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
    figure.suptitle("HighStrain simulated sample and official-model reconstruction")
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def _surface_points(
    amplitude: np.ndarray,
    phase: np.ndarray,
    threshold: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    maximum = max(float(amplitude.max()), 1e-12)
    support = amplitude > threshold * maximum
    surface = support & ~binary_erosion(support)
    points = np.argwhere(surface)
    values = phase[surface]
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = points[indices]
        values = values[indices]
    return points, values


def _format_3d_axis(axis: plt.Axes, size: int, title: str) -> None:
    axis.set_title(title)
    axis.set_xlim(0, size - 1)
    axis.set_ylim(0, size - 1)
    axis.set_zlim(0, size - 1)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel("z")
    axis.set_ylabel("y")
    axis.set_zlabel("x")
    axis.view_init(elev=24, azim=42)


def save_volume_overview(
    *,
    intensity: np.ndarray,
    target_object: np.ndarray,
    predicted_object: np.ndarray,
    destination: str | Path,
    support_threshold: float = 0.1,
    max_surface_points: int = 7000,
    max_diffraction_points: int = 5000,
) -> Path:
    """Save phase-colored object surfaces and a reciprocal-space point cloud."""

    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = int(intensity.shape[0])
    target_amplitude = np.abs(target_object)
    predicted_amplitude = np.abs(predicted_object)
    target_points, target_phase = _surface_points(
        target_amplitude,
        np.angle(target_object),
        support_threshold,
        max_surface_points,
    )
    predicted_points, predicted_phase = _surface_points(
        predicted_amplitude,
        np.angle(predicted_object),
        support_threshold,
        max_surface_points,
    )

    normalized = intensity / max(float(intensity.max()), 1.0)
    positive = normalized[normalized > 0]
    cutoff = float(np.quantile(positive, 0.97)) if positive.size else 1.0
    diffraction_points = np.argwhere(normalized >= cutoff)
    diffraction_values = np.log10(
        np.clip(normalized[normalized >= cutoff], 1e-6, None)
    )
    if len(diffraction_points) > max_diffraction_points:
        indices = np.linspace(
            0, len(diffraction_points) - 1, max_diffraction_points, dtype=int
        )
        diffraction_points = diffraction_points[indices]
        diffraction_values = diffraction_values[indices]

    figure = plt.figure(figsize=(16, 5.5), constrained_layout=True)
    phase_normalizer = colors.Normalize(vmin=-np.pi, vmax=np.pi)
    phase_map = plt.get_cmap("twilight")

    target_axis = figure.add_subplot(1, 3, 1, projection="3d")
    target_axis.scatter(
        target_points[:, 2],
        target_points[:, 1],
        target_points[:, 0],
        c=phase_map(phase_normalizer(target_phase)),
        s=3,
        alpha=0.8,
        linewidths=0,
    )
    _format_3d_axis(target_axis, size, "Target object surface (phase color)")

    diffraction_axis = figure.add_subplot(1, 3, 2, projection="3d")
    diffraction_artist = diffraction_axis.scatter(
        diffraction_points[:, 2],
        diffraction_points[:, 1],
        diffraction_points[:, 0],
        c=diffraction_values,
        cmap="magma",
        s=4,
        alpha=0.7,
        linewidths=0,
    )
    _format_3d_axis(diffraction_axis, size, "Measured diffraction (top 3%)")
    figure.colorbar(
        diffraction_artist,
        ax=diffraction_axis,
        fraction=0.035,
        pad=0.08,
        label="log10 normalized intensity",
    )

    predicted_axis = figure.add_subplot(1, 3, 3, projection="3d")
    predicted_axis.scatter(
        predicted_points[:, 2],
        predicted_points[:, 1],
        predicted_points[:, 0],
        c=phase_map(phase_normalizer(predicted_phase)),
        s=3,
        alpha=0.8,
        linewidths=0,
    )
    _format_3d_axis(predicted_axis, size, "Predicted object surface (phase color)")
    phase_scalar = plt.cm.ScalarMappable(norm=phase_normalizer, cmap=phase_map)
    figure.colorbar(
        phase_scalar,
        ax=[target_axis, predicted_axis],
        fraction=0.02,
        pad=0.04,
        label="wrapped phase (rad)",
    )
    figure.suptitle("3D simulation and official HighStrain model output")
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination
