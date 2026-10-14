"""Empirical liquid-cell and detector response applied to ideal diffraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter

from .config import InstrumentConfig
from .sampling import sample_uniform


_FLOAT_EPS = np.finfo(np.float64).eps


@dataclass(frozen=True)
class InstrumentSample:
    """Sampled response parameters for one frame."""

    dqe: float
    mtf_sigma_px: float
    background_fraction: float
    liquid_ring_radius_fraction: float
    liquid_ring_sigma_fraction: float
    center_shift_x_px: float
    center_shift_y_px: float
    ellipticity: float
    rotation_deg: float
    gain: float
    offset_counts: float
    read_noise_sigma_counts: float
    saturation_counts: float
    bad_pixel_fraction: float
    log_normalization_percentile: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class InstrumentResult:
    """Raw and log-scaled detector frames."""

    observed_counts: np.ndarray
    observed_log: np.ndarray
    metadata: dict[str, float]


def sample_instrument(
    config: InstrumentConfig, rng: np.random.Generator
) -> InstrumentSample:
    """Draw one deterministic detector configuration."""

    shift_radius = sample_uniform(rng, config.center_shift_px)
    shift_angle = float(rng.uniform(0.0, 2.0 * math.pi))
    return InstrumentSample(
        dqe=sample_uniform(rng, config.dqe),
        mtf_sigma_px=sample_uniform(rng, config.mtf_sigma_px),
        background_fraction=sample_uniform(rng, config.background_fraction),
        liquid_ring_radius_fraction=sample_uniform(
            rng, config.liquid_ring_radius_fraction
        ),
        liquid_ring_sigma_fraction=sample_uniform(
            rng, config.liquid_ring_sigma_fraction
        ),
        center_shift_x_px=shift_radius * math.cos(shift_angle),
        center_shift_y_px=shift_radius * math.sin(shift_angle),
        ellipticity=sample_uniform(rng, config.ellipticity),
        rotation_deg=sample_uniform(rng, config.rotation_deg),
        gain=sample_uniform(rng, config.gain),
        offset_counts=sample_uniform(rng, config.offset_counts),
        read_noise_sigma_counts=sample_uniform(
            rng, config.read_noise_sigma_counts
        ),
        saturation_counts=sample_uniform(rng, config.saturation_counts),
        bad_pixel_fraction=sample_uniform(rng, config.bad_pixel_fraction),
        log_normalization_percentile=config.log_normalization_percentile,
    )


def _normalize_nonnegative(image: np.ndarray) -> np.ndarray:
    image = np.maximum(np.asarray(image, dtype=np.float64), 0.0)
    return image / max(float(image.sum()), _FLOAT_EPS)


def _liquid_background(shape: tuple[int, int], sample: InstrumentSample) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=np.float64)
    center_y = (shape[0] - 1) / 2.0
    center_x = (shape[1] - 1) / 2.0
    radius = np.hypot(yy - center_y, xx - center_x)
    radius /= max(min(shape) / 2.0, _FLOAT_EPS)

    ring_sigma = max(sample.liquid_ring_sigma_fraction, 1e-4)
    ring = np.exp(
        -0.5
        * ((radius - sample.liquid_ring_radius_fraction) / ring_sigma) ** 2
    )
    central_tail = 1.0 / (1.0 + (radius / max(ring_sigma, 1e-3)) ** 2)
    background = ring + central_tail
    return _normalize_nonnegative(background)


def _distort(image: np.ndarray, sample: InstrumentSample) -> np.ndarray:
    angle = math.radians(sample.rotation_deg)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    scale_x = max(1.0 + sample.ellipticity, 1e-3)
    scale_y = max(1.0 / scale_x, 1e-3)
    forward = rotation @ np.diag([scale_y, scale_x]) @ rotation.T
    inverse = np.linalg.inv(forward)

    center = (np.asarray(image.shape, dtype=np.float64) - 1.0) / 2.0
    shift = np.array(
        [sample.center_shift_y_px, sample.center_shift_x_px], dtype=np.float64
    )
    offset = center - inverse @ (center + shift)
    distorted = affine_transform(
        image,
        inverse,
        offset=offset,
        output_shape=image.shape,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return _normalize_nonnegative(distorted)


def render_detector_frame(
    ideal_intensity: np.ndarray,
    illuminated_area_a2: float,
    dose_e_per_a2: float,
    sample: InstrumentSample,
    rng: np.random.Generator,
) -> InstrumentResult:
    """Convert ideal normalized intensity to a noisy detector frame."""

    signal = _normalize_nonnegative(ideal_intensity)
    background = _liquid_background(signal.shape, sample)
    mixed = (
        (1.0 - sample.background_fraction) * signal
        + sample.background_fraction * background
    )
    if sample.mtf_sigma_px > 0:
        mixed = gaussian_filter(mixed, sigma=sample.mtf_sigma_px, mode="nearest")
    mixed = _distort(_normalize_nonnegative(mixed), sample)

    incident_electrons = max(dose_e_per_a2 * illuminated_area_a2, 0.0)
    expected_detected = mixed * incident_electrons * sample.dqe
    detected = rng.poisson(expected_detected).astype(np.float64)
    observed = detected * sample.gain + sample.offset_counts
    if sample.read_noise_sigma_counts > 0:
        observed += rng.normal(
            0.0, sample.read_noise_sigma_counts, size=observed.shape
        )
    observed = np.clip(observed, 0.0, sample.saturation_counts)

    bad_pixel_count = int(round(observed.size * sample.bad_pixel_fraction))
    if bad_pixel_count:
        flat_indices = rng.choice(observed.size, size=bad_pixel_count, replace=False)
        half = bad_pixel_count // 2
        observed.flat[flat_indices[:half]] = 0.0
        observed.flat[flat_indices[half:]] = sample.saturation_counts

    observed = observed.astype(np.float32)
    dark_corrected = np.maximum(
        observed.astype(np.float64) - sample.offset_counts, 0.0
    )
    observed_log = np.log1p(dark_corrected)
    robust_max = float(
        np.percentile(observed_log, sample.log_normalization_percentile)
    )
    observed_log = np.clip(observed_log / max(robust_max, _FLOAT_EPS), 0.0, 1.0)
    return InstrumentResult(
        observed_counts=observed,
        observed_log=observed_log.astype(np.float32),
        metadata={
            **sample.to_dict(),
            "illuminated_area_a2": float(illuminated_area_a2),
            "incident_electrons": float(incident_electrons),
            "expected_detected_electrons": float(incident_electrons * sample.dqe),
            "training_dark_correction_counts": sample.offset_counts,
        },
    )
