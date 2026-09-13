"""Reciprocal/real-space transforms using the AutoPhaseNN convention."""

from __future__ import annotations

import torch


SPATIAL_DIMS = (-3, -2, -1)


def _without_singleton_channel(
    value: torch.Tensor,
    name: str,
) -> tuple[torch.Tensor, bool]:
    if value.ndim == 5 and value.shape[1] == 1:
        return value[:, 0], True
    if value.ndim == 4:
        return value, False
    raise ValueError(f"{name} must have shape [B, D, H, W] or [B, 1, D, H, W].")


def reciprocal_field_from_modulus_phase(
    diffraction_modulus: torch.Tensor,
    reciprocal_phase: torch.Tensor,
) -> torch.Tensor:
    """Combine measured modulus and predicted phase into a complex spectrum."""

    modulus, modulus_had_channel = _without_singleton_channel(
        diffraction_modulus,
        "diffraction_modulus",
    )
    phase, phase_had_channel = _without_singleton_channel(
        reciprocal_phase,
        "reciprocal_phase",
    )
    if modulus.shape != phase.shape:
        raise ValueError("Diffraction modulus and reciprocal phase must share shape.")
    field = torch.complex(modulus * torch.cos(phase), modulus * torch.sin(phase))
    return field[:, None] if modulus_had_channel or phase_had_channel else field


def realspace_from_modulus_phase(
    diffraction_modulus: torch.Tensor,
    reciprocal_phase: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct the complex real-space object with the inverse 3D FFT."""

    reciprocal = reciprocal_field_from_modulus_phase(
        diffraction_modulus,
        reciprocal_phase,
    )
    field, had_channel = _without_singleton_channel(reciprocal, "reciprocal_field")
    shifted = torch.fft.ifftshift(field, dim=SPATIAL_DIMS)
    realspace = torch.fft.ifftn(shifted, dim=SPATIAL_DIMS)
    realspace = torch.fft.fftshift(realspace, dim=SPATIAL_DIMS)
    return realspace[:, None] if had_channel else realspace


def reciprocal_field_from_realspace(realspace: torch.Tensor) -> torch.Tensor:
    """Transform a complex real-space object to the centered reciprocal grid."""

    field, had_channel = _without_singleton_channel(realspace, "realspace")
    if not torch.is_complex(field):
        raise ValueError("realspace must be a complex tensor.")
    shifted = torch.fft.ifftshift(field, dim=SPATIAL_DIMS)
    reciprocal = torch.fft.fftn(shifted, dim=SPATIAL_DIMS)
    reciprocal = torch.fft.fftshift(reciprocal, dim=SPATIAL_DIMS)
    return reciprocal[:, None] if had_channel else reciprocal


def farfield_modulus_from_realspace(realspace: torch.Tensor) -> torch.Tensor:
    """Project a complex object back to diffraction modulus for evaluation."""

    return reciprocal_field_from_realspace(realspace).abs()


def project_to_measured_modulus(
    realspace: torch.Tensor,
    diffraction_modulus: torch.Tensor,
) -> torch.Tensor:
    """Apply one differentiable Fourier-modulus projection to an object."""

    reciprocal = reciprocal_field_from_realspace(realspace)
    field, had_channel = _without_singleton_channel(reciprocal, "reciprocal_field")
    modulus, _ = _without_singleton_channel(
        diffraction_modulus,
        "diffraction_modulus",
    )
    if field.shape != modulus.shape:
        raise ValueError("Real-space field and measured modulus must share shape.")
    projected = torch.complex(
        modulus * torch.cos(torch.angle(field)),
        modulus * torch.sin(torch.angle(field)),
    )
    shifted = torch.fft.ifftshift(projected, dim=SPATIAL_DIMS)
    result = torch.fft.ifftn(shifted, dim=SPATIAL_DIMS)
    result = torch.fft.fftshift(result, dim=SPATIAL_DIMS)
    return result[:, None] if had_channel else result


def complex_to_channels(realspace: torch.Tensor) -> torch.Tensor:
    """Convert complex ``[B, 1, D, H, W]`` data to two real channels."""

    field, _ = _without_singleton_channel(realspace, "realspace")
    if not torch.is_complex(field):
        raise ValueError("realspace must be a complex tensor.")
    return torch.stack((field.real, field.imag), dim=1)


def channels_to_complex(channels: torch.Tensor) -> torch.Tensor:
    """Convert real/imaginary ``[B, 2, D, H, W]`` channels to complex data."""

    if channels.ndim != 5 or channels.shape[1] != 2:
        raise ValueError("channels must have shape [B, 2, D, H, W].")
    return torch.complex(channels[:, 0], channels[:, 1])[:, None]
