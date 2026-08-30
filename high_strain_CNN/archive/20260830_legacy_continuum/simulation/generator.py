"""Generate centered 3D strained crystals and their noisy diffraction."""

from __future__ import annotations

import itertools
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.signal import fftconvolve
from scipy.spatial.transform import Rotation

from .config import SimulationConfig


@dataclass(frozen=True)
class SimulatedSample:
    """One author-loader-compatible diffraction sample and optional ground truth."""

    intensity: np.ndarray
    reciprocal_phase: np.ndarray
    support: np.ndarray
    object_phase: np.ndarray
    realspace_object: np.ndarray
    clean_intensity: np.ndarray
    metadata: dict[str, Any]


def _uniform(rng: np.random.Generator, limits: tuple[float, float]) -> float:
    return float(rng.uniform(limits[0], limits[1]))


def _unit_vector(rng: np.random.Generator) -> np.ndarray:
    vector = rng.normal(size=3)
    return vector / np.linalg.norm(vector)


def _coordinate_grid(size: int) -> np.ndarray:
    axis = np.arange(size, dtype=np.float32) - size // 2
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=0)


def _rotated_coordinates(
    coordinates: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = Rotation.random(random_state=rng).as_matrix().astype(np.float32)
    local = np.einsum("ij,jdhw->idhw", rotation.T, coordinates, optimize=True)
    return local, rotation


def _wulff_support(
    local: np.ndarray,
    half_extent: float,
    facet_ratio_111: float,
    facet_ratio_110: float,
) -> np.ndarray:
    support = np.max(np.abs(local), axis=0) <= half_extent
    distance_111 = facet_ratio_111 * half_extent
    root_three = np.sqrt(3.0)
    for signs in itertools.product((-1.0, 1.0), repeat=3):
        normal = np.asarray(signs, dtype=np.float32) / root_three
        projection = np.einsum("i,idhw->dhw", normal, local, optimize=True)
        support &= projection <= distance_111
    distance_110 = facet_ratio_110 * half_extent
    for zero_axis in range(3):
        active_axes = [axis for axis in range(3) if axis != zero_axis]
        for signs in itertools.product((-1.0, 1.0), repeat=2):
            normal = np.zeros(3, dtype=np.float32)
            normal[active_axes] = np.asarray(signs, dtype=np.float32) / np.sqrt(2.0)
            projection = np.einsum("i,idhw->dhw", normal, local, optimize=True)
            support &= projection <= distance_110
    return support


def _make_support(
    local: np.ndarray,
    shape_type: str,
    half_extent: float,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    shape_config = config.shape
    if shape_type in {"wulff", "winterbottom"}:
        facet_ratio_111 = _uniform(rng, shape_config.wulff_111_to_100_distance)
        facet_ratio_110 = _uniform(rng, shape_config.wulff_110_to_100_distance)
        support = _wulff_support(
            local,
            half_extent,
            facet_ratio_111,
            facet_ratio_110,
        )
        parameters: dict[str, Any] = {
            "wulff_111_to_100_distance": facet_ratio_111,
            "wulff_110_to_100_distance": facet_ratio_110,
        }
        if shape_type == "winterbottom":
            signs = rng.choice(np.asarray((-1.0, 1.0)), size=3)
            interface_normal = signs / np.sqrt(3.0)
            projection = np.einsum(
                "i,idhw->dhw", interface_normal, local, optimize=True
            )
            values = projection[support]
            retained = _uniform(rng, shape_config.winterbottom_retained_height)
            interface = float(values.max() - retained * np.ptp(values))
            support &= projection >= interface
            parameters.update(
                {
                    "interface_normal": interface_normal.tolist(),
                    "retained_height_fraction": retained,
                    "interface_coordinate": interface,
                }
            )
        return support, parameters

    if shape_type == "random_polyhedron":
        support = np.max(np.abs(local), axis=0) <= half_extent
        count = int(
            rng.integers(
                shape_config.random_plane_count[0],
                shape_config.random_plane_count[1] + 1,
            )
        )
        planes = []
        for _ in range(count):
            normal = _unit_vector(rng)
            offset_fraction = _uniform(
                rng, shape_config.random_plane_offset_fraction
            )
            offset = half_extent * offset_fraction
            projection = np.einsum("i,idhw->dhw", normal, local, optimize=True)
            support &= projection <= offset
            planes.append(
                {"normal": normal.tolist(), "offset_fraction": offset_fraction}
            )
        return support, {"planes": planes}

    raise ValueError(f"Unsupported shape type: {shape_type}")


def _double_gaussian_phase(
    local: np.ndarray,
    half_extent: float,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    normalized = local / max(half_extent, np.finfo(np.float32).eps)
    phase = np.zeros(local.shape[1:], dtype=np.float32)
    components = []
    for _ in range(2):
        center = rng.uniform(*config.phase.gaussian_center_fraction, size=3)
        sigma = rng.uniform(*config.phase.gaussian_sigma_fraction, size=3)
        amplitude = float(rng.uniform(-1.0, 1.0))
        if abs(amplitude) < 0.2:
            amplitude = 0.2 if amplitude >= 0 else -0.2
        exponent = np.sum(
            np.square((normalized - center[:, None, None, None]) / sigma[:, None, None, None]),
            axis=0,
        )
        phase += amplitude * np.exp(-0.5 * exponent)
        components.append(
            {
                "center_fraction": center.tolist(),
                "sigma_fraction": sigma.tolist(),
                "amplitude": amplitude,
            }
        )
    return phase, {"components": components}


def _double_cosine_phase(
    local: np.ndarray,
    half_extent: float,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    normalized = local / max(2.0 * half_extent, np.finfo(np.float32).eps)
    phase = np.zeros(local.shape[1:], dtype=np.float32)
    components = []
    for _ in range(2):
        direction = _unit_vector(rng)
        cycles = _uniform(rng, config.phase.cosine_cycles)
        offset = float(rng.uniform(0.0, 2.0 * np.pi))
        amplitude = float(rng.uniform(0.5, 1.0) * rng.choice((-1.0, 1.0)))
        argument = (
            2.0
            * np.pi
            * cycles
            * np.einsum("i,idhw->dhw", direction, normalized, optimize=True)
            + offset
        )
        phase += amplitude * np.cos(argument)
        components.append(
            {
                "direction": direction.tolist(),
                "cycles": cycles,
                "offset": offset,
                "amplitude": amplitude,
            }
        )
    return phase, {"components": components}


def support_extent(support: np.ndarray) -> np.ndarray:
    """Measure occupied-voxel widths, including both boundary voxels."""

    occupied = np.argwhere(support)
    if not occupied.size:
        raise ValueError("Cannot measure an empty support.")
    return np.ptp(occupied, axis=0) + 1


def _fit_support_to_oversampling(
    local: np.ndarray,
    half_extent: float,
    shape_type: str,
    parameters: dict[str, Any],
    oversampling: float,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Resize the same convex shape after rotation, without redrawing its facets."""

    absolute = np.abs(local)
    radius = np.max(absolute, axis=0)
    if shape_type in {"wulff", "winterbottom"}:
        radius = np.maximum(
            radius,
            np.sum(absolute, axis=0)
            / (np.sqrt(3.0) * parameters["wulff_111_to_100_distance"]),
        )
        radius = np.maximum(
            radius,
            (np.sum(absolute, axis=0) - np.min(absolute, axis=0))
            / (np.sqrt(2.0) * parameters["wulff_110_to_100_distance"]),
        )
        if shape_type == "winterbottom":
            interface_fraction = parameters["interface_coordinate"] / half_extent
            if interface_fraction >= 0:
                raise ValueError("Resizing requires a Winterbottom shape containing the origin.")
            projection = np.einsum(
                "i,idhw->dhw", parameters["interface_normal"], local, optimize=True
            )
            radius = np.maximum(radius, projection / interface_fraction)
    else:
        for plane in parameters["planes"]:
            projection = np.einsum(
                "i,idhw->dhw", plane["normal"], local, optimize=True
            )
            radius = np.maximum(radius, projection / plane["offset_fraction"])

    size = local.shape[1]
    allowed_width = int(np.floor(size / oversampling))
    if allowed_width < 3:
        raise ValueError("Requested oversampling leaves fewer than three object voxels.")
    lower, upper = 0.0, float(radius.max())
    # Twenty-four bisections resolve sub-voxel facet positions on float32 grids.
    for _ in range(24):
        candidate = (lower + upper) / 2.0
        mask = radius <= candidate
        if int(support_extent(mask).max()) <= allowed_width:
            lower = candidate
        else:
            upper = candidate
    support = radius <= lower
    parameters = dict(parameters)
    if shape_type == "winterbottom":
        parameters["interface_coordinate"] *= lower / half_extent
    return support, lower, parameters


def _author_correlated_noise(
    white_noise: np.ndarray,
    correlation_length: float,
) -> np.ndarray:
    """Evaluate the Gaussian-convolution formula in Patching_DL random_noise."""

    size = white_noise.shape[0]
    axis = np.linspace(-1.0, 1.0, size)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    kernel = np.exp(-(x * x + y * y + z * z) / (2.0 * correlation_length**2))
    noise = fftconvolve(kernel, white_noise, mode="same")
    maximum = float(noise.max())
    if abs(maximum) <= np.finfo(np.float64).tiny:
        raise ValueError("Degenerate correlated random field.")
    return noise / maximum


def _correlated_phase(
    local: np.ndarray,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    if config.phase.correlated_method == "author_convolution":
        size = local.shape[1]
        correlation = _uniform(rng, config.phase.correlation_length_fraction)
        field = _author_correlated_noise(rng.standard_normal((size,) * 3), correlation)
        # Sample in the particle frame so shape and phase share one rotation.
        phase = map_coordinates(
            field, local + size // 2, order=1, mode="constant", cval=0.0
        )
        return phase.astype(np.float32), {
            "correlated_method": "author_convolution",
            "kernel_sigma_fraction": correlation,
            "kernel_sigma_voxels": correlation * (size - 1) / 2.0,
            "convolution_boundary": "zero_extended_same",
            "field_frame": "particle",
        }
    correlation = rng.uniform(*config.phase.correlation_length_voxels, size=3)
    white_noise = rng.standard_normal(local.shape[1:]).astype(np.float32)
    phase = gaussian_filter(white_noise, sigma=correlation, mode="reflect")
    return phase.astype(np.float32), {
        "correlation_length_voxels": correlation.tolist(),
        "correlated_method": "legacy_filter",
    }


def _make_phase(
    local: np.ndarray,
    support: np.ndarray,
    half_extent: float,
    phase_type: str,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    if phase_type == "double_gaussian":
        raw, parameters = _double_gaussian_phase(local, half_extent, config, rng)
    elif phase_type == "double_cosine":
        raw, parameters = _double_cosine_phase(local, half_extent, config, rng)
    elif phase_type == "gaussian_correlated":
        raw, parameters = _correlated_phase(local, config, rng)
    else:
        raise ValueError(f"Unsupported phase type: {phase_type}")

    values = raw[support]
    raw_span = float(np.ptp(values))
    if raw_span <= np.finfo(np.float32).eps:
        raise RuntimeError("Generated phase field has no variation inside the support.")
    target_pi = _uniform(rng, config.phase.peak_to_peak_pi)
    target_span = target_pi * np.pi
    centered = raw - float(np.mean(values))
    phase = centered * (target_span / raw_span)
    phase = np.where(support, phase, 0.0).astype(np.float32)
    parameters.update(
        {
            "target_peak_to_peak_pi": target_pi,
            "actual_peak_to_peak_rad": float(np.ptp(phase[support])),
        }
    )
    return phase, parameters


def _roll_without_wrap(array: np.ndarray, shift: tuple[int, int, int]) -> np.ndarray:
    shifted = np.roll(array, shift=shift, axis=(0, 1, 2))
    for axis, amount in enumerate(shift):
        if amount == 0:
            continue
        selection = [slice(None)] * array.ndim
        selection[axis] = slice(0, amount) if amount > 0 else slice(amount, None)
        shifted[tuple(selection)] = 0
    return shifted


def _center_particle(
    support: np.ndarray,
    phase: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int], list[float]]:
    occupied = np.argwhere(support)
    if occupied.size == 0:
        raise RuntimeError("Generated support is empty.")
    center_of_mass = occupied.mean(axis=0)
    target = np.asarray(support.shape, dtype=np.float64) // 2
    shift_array = np.rint(target - center_of_mass).astype(int)
    shift = tuple(int(value) for value in shift_array)
    centered_support = _roll_without_wrap(support, shift).astype(bool)
    centered_phase = _roll_without_wrap(phase, shift).astype(np.float32)
    final_center = np.argwhere(centered_support).mean(axis=0).tolist()
    return centered_support, centered_phase, shift, final_center


def _sample_peak_photons(
    config: SimulationConfig,
    rng: np.random.Generator,
) -> float:
    lower, upper = config.noise.peak_photons
    if config.noise.sample_log_uniform:
        return float(np.exp(rng.uniform(np.log(lower), np.log(upper))))
    return float(rng.uniform(lower, upper))


def _intensity_centroid(intensity: np.ndarray) -> np.ndarray:
    total = float(np.sum(intensity, dtype=np.float64))
    if total <= 0:
        raise ValueError("Cannot center an empty diffraction pattern.")
    return np.asarray([
        np.dot(
            np.arange(intensity.shape[axis]),
            np.sum(intensity, axis=tuple(i for i in range(3) if i != axis), dtype=np.float64),
        ) / total
        for axis in range(3)
    ])


def _center_diffraction(
    reciprocal: np.ndarray,
    phase: np.ndarray,
    support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Shift the discrete q grid and apply its matching real-space phase ramp."""

    intensity = np.abs(reciprocal.astype(np.complex128)) ** 2
    before = _intensity_centroid(intensity)
    target = np.asarray(phase.shape) // 2
    shift = tuple(np.rint(target - before).astype(int))
    shifted = np.roll(reciprocal, shift, axis=(0, 1, 2))
    retained = np.sum(_roll_without_wrap(intensity, shift), dtype=np.float64)
    wrapped_fraction = max(0.0, 1.0 - float(retained / intensity.sum()))
    coordinates = _coordinate_grid(phase.shape[0]).astype(np.float64)
    ramp = 2.0 * np.pi * np.einsum(
        "i,idhw->dhw", np.asarray(shift) / np.asarray(phase.shape), coordinates
    )
    shifted_phase = np.where(support, phase + ramp, 0.0).astype(np.float32)
    return shifted, shifted_phase, {
        "method": "integer_periodic",
        "shift_voxels": [int(value) for value in shift],
        "clean_centroid_before": before.tolist(),
        "clean_centroid_after": _intensity_centroid(np.abs(shifted) ** 2).tolist(),
        "wrapped_edge_energy_fraction": wrapped_fraction,
        "phase_ramp_sign": "+2pi * shift dot centered_realspace_index / grid_size",
    }


def _diffract_sample(
    support: np.ndarray,
    phase: np.ndarray,
    metadata: dict[str, Any],
    rng: np.random.Generator,
) -> SimulatedSample:
    """Shared forward physics and photon sampling for strained/control objects."""

    centering_method = metadata["diffraction_centering"]["method"]
    if centering_method not in {"none", "integer_periodic"}:
        raise ValueError(f"Unsupported diffraction centering: {centering_method}")
    realspace = support.astype(np.float32) * np.exp(1.0j * phase)
    reciprocal = np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(realspace))
    ).astype(np.complex64)
    diffraction_centering: dict[str, Any] = {"method": "none"}
    if centering_method == "integer_periodic":
        reciprocal, phase, diffraction_centering = _center_diffraction(
            reciprocal, phase, support
        )
        realspace = support.astype(np.float32) * np.exp(1.0j * phase)
    metadata["phase_parameters"]["effective_peak_to_peak_rad"] = float(np.ptp(phase[support]))
    clean_intensity = np.square(np.abs(reciprocal)).astype(np.float32)

    peak_photons = metadata["noise"]["peak_photons"]
    expected_total = float(np.sum(clean_intensity))
    if metadata["noise"]["enabled"]:
        expected = clean_intensity / float(clean_intensity.max()) * peak_photons
        intensity = rng.poisson(expected).astype(np.float32)
        expected_total = float(np.sum(expected))
    else:
        intensity = clean_intensity.copy()

    reciprocal_phase = np.angle(reciprocal).astype(np.float32)
    center = tuple(size // 2 for size in support.shape)
    metadata.update({
        "diffraction_centering": diffraction_centering,
        "observed_diffraction_centroid": _intensity_centroid(intensity).tolist(),
        "reciprocal_center_phase_rad": float(reciprocal_phase[center]),
    })
    metadata["noise"].update({
        "expected_total_photons": expected_total,
        "observed_total_photons": float(np.sum(intensity)),
    })
    return SimulatedSample(
        intensity=intensity,
        reciprocal_phase=reciprocal_phase,
        support=support,
        object_phase=phase,
        realspace_object=realspace.astype(np.complex64),
        clean_intensity=clean_intensity,
        metadata=metadata,
    )


def generate_sample(
    config: SimulationConfig,
    rng: np.random.Generator,
    *,
    shape_type: str | None = None,
    phase_type: str | None = None,
) -> SimulatedSample:
    """Generate one centered crystal and author-compatible diffraction pair."""

    shape_type = shape_type or str(rng.choice(config.shape.types))
    phase_type = phase_type or str(rng.choice(config.phase.types))
    if shape_type not in config.shape.types:
        raise ValueError(f"Shape {shape_type!r} is not enabled by the configuration.")
    if phase_type not in config.phase.types:
        raise ValueError(f"Phase {phase_type!r} is not enabled by the configuration.")

    oversampling = _uniform(rng, config.shape.oversampling_ratio)
    half_extent = config.grid_size / (2.0 * oversampling)
    coordinates = _coordinate_grid(config.grid_size)
    local, rotation = _rotated_coordinates(coordinates, rng)
    support, shape_parameters = _make_support(
        local, shape_type, half_extent, config, rng
    )
    if config.shape.oversampling_mode == "measured":
        support, half_extent, shape_parameters = _fit_support_to_oversampling(
            local, half_extent, shape_type, shape_parameters, oversampling
        )
    phase, phase_parameters = _make_phase(
        local, support, half_extent, phase_type, config, rng
    )
    count_before_centering = int(np.count_nonzero(support))
    support, phase, center_shift, final_center = _center_particle(support, phase)
    actual_extent = support_extent(support)
    actual_oversampling = config.grid_size / actual_extent
    if config.shape.oversampling_mode == "measured":
        if np.any(actual_oversampling < oversampling):
            raise RuntimeError("Generated support violates measured oversampling.")
        if int(np.count_nonzero(support)) != count_before_centering:
            raise RuntimeError("Particle centering clipped the generated support.")

    metadata: dict[str, Any] = {
        "schema_version": config.schema_version,
        "shape_type": shape_type,
        "phase_type": phase_type,
        "grid_size": config.grid_size,
        "oversampling_ratio": oversampling,
        "oversampling_mode": config.shape.oversampling_mode,
        "actual_support_extent_voxels": actual_extent.tolist(),
        "actual_oversampling_ratio": actual_oversampling.tolist(),
        "half_extent_voxels": half_extent,
        "rotation_matrix": rotation.tolist(),
        "center_shift_voxels": list(center_shift),
        "final_support_center_of_mass": final_center,
        "support_center_residual_voxels": (
            np.asarray(final_center) - config.grid_size // 2
        ).tolist(),
        "diffraction_centering": {"method": config.diffraction_centering},
        "scattering_backend": "numpy_centered_fft_discrete_kinematic",
        "coordinate_convention": "real and reciprocal index origins at N//2; FFT sign -2pi*i",
        "support_voxels": int(np.count_nonzero(support)),
        "shape_parameters": shape_parameters,
        "phase_parameters": phase_parameters,
        "noise": {
            "enabled": config.noise.enabled,
            "definition": "expected photons at brightest detector voxel",
            "peak_photons": _sample_peak_photons(config, rng) if config.noise.enabled else None,
        },
        "paper_fixed": config.provenance.get("paper_fixed", []),
        "replication_assumptions": config.provenance.get(
            "replication_assumptions", []
        ),
    }
    return _diffract_sample(support, phase, metadata, rng)


def generate_unstrained_control(
    reference_path: str | Path,
    rng: np.random.Generator,
) -> SimulatedSample:
    """Reuse an existing simulated support and photon peak with real-space phase zero."""

    reference_path = Path(reference_path).expanduser().resolve()
    with np.load(reference_path, allow_pickle=False) as source:
        required = {"support", "object", "metadata_json"}
        if not required.issubset(source.files):
            raise ValueError("Paired controls need a generated reference saved with --save-extras.")
        raw_support = source["support"]
        support = raw_support.astype(bool)
        obj = source["object"]
        metadata = json.loads(str(source["metadata_json"].item()))
    if (
        support.ndim != 3 or len(set(support.shape)) != 1 or not support.any()
        or obj.shape != support.shape or not np.all(np.isfinite(obj))
        or not np.all((raw_support == 0) | (raw_support == 1))
        or not np.allclose(np.abs(obj), support.astype(np.float32), rtol=1e-6, atol=1e-7)
    ):
        raise ValueError("Reference must contain a finite, cubic, unit-amplitude supported object.")
    if metadata["grid_size"] != support.shape[0]:
        raise ValueError("Reference metadata grid size differs from the stored object.")
    peak = metadata["noise"]["peak_photons"]
    if metadata["noise"]["enabled"] and (
        peak is None or not np.isfinite(peak) or peak <= 0
    ):
        raise ValueError("Reference photon peak must be finite and positive when noise is enabled.")

    metadata["control"] = {
        "type": "zero_realspace_phase",
        "reference_sample": str(reference_path),
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "reference_phase_type": metadata["phase_type"],
        "amplitude": "same binary support and unit amplitude as reference",
        "noise": "same per-sample peak; independent Poisson draw from recomputed intensity",
    }
    metadata["phase_type"] = "unstrained"
    metadata["phase_parameters"] = {
        "target_peak_to_peak_pi": 0.0,
        "actual_peak_to_peak_rad": 0.0,
    }
    metadata["replication_assumptions"] = list(metadata.get("replication_assumptions", [])) + [
        "Unstrained diagnostic control, not the paper's high-strain training distribution."
    ]
    phase = np.zeros(support.shape, dtype=np.float32)
    return _diffract_sample(support, phase, metadata, rng)


def save_sample(
    sample: SimulatedSample,
    path: str | Path,
    *,
    save_extras: bool = False,
) -> Path:
    """Write one compressed NPZ using the official loader's ``I``/``phi`` keys."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "I": sample.intensity.astype(np.float32, copy=False),
        "phi": sample.reciprocal_phase.astype(np.float32, copy=False),
        "metadata_json": np.asarray(json.dumps(sample.metadata, sort_keys=True)),
    }
    if save_extras:
        arrays.update(
            {
                "support": sample.support.astype(np.uint8, copy=False),
                "object_phase": sample.object_phase.astype(np.float32, copy=False),
                "object": sample.realspace_object.astype(np.complex64, copy=False),
                "I_clean": sample.clean_intensity.astype(np.float32, copy=False),
            }
        )
    np.savez_compressed(destination, **arrays)
    return destination
