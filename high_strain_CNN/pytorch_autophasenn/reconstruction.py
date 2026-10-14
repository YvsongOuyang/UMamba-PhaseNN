"""Reciprocal/real-space transforms using the AutoPhaseNN convention."""

from __future__ import annotations

import torch


SPATIAL_DIMS = (-3, -2, -1)


def _without_singleton_channel(
    value: torch.Tensor,
    name: str,
    spatial_dims: int = 3,
) -> tuple[torch.Tensor, bool]:
    if spatial_dims not in (2, 3):
        raise ValueError("spatial_dims must be 2 or 3.")
    if value.ndim == spatial_dims + 2 and value.shape[1] == 1:
        return value[:, 0], True
    if value.ndim == spatial_dims + 1:
        return value, False
    raise ValueError(
        f"{name} must be batched {spatial_dims}D data with an optional singleton channel."
    )


def reciprocal_field_from_modulus_phase(
    diffraction_modulus: torch.Tensor,
    reciprocal_phase: torch.Tensor,
    *,
    spatial_dims: int = 3,
) -> torch.Tensor:
    """Combine measured modulus and predicted phase into a complex spectrum."""

    modulus, modulus_had_channel = _without_singleton_channel(
        diffraction_modulus,
        "diffraction_modulus",
        spatial_dims,
    )
    phase, phase_had_channel = _without_singleton_channel(
        reciprocal_phase,
        "reciprocal_phase",
        spatial_dims,
    )
    if modulus.shape != phase.shape:
        raise ValueError("Diffraction modulus and reciprocal phase must share shape.")
    field = torch.complex(modulus * torch.cos(phase), modulus * torch.sin(phase))
    return field[:, None] if modulus_had_channel or phase_had_channel else field


def realspace_from_modulus_phase(
    diffraction_modulus: torch.Tensor,
    reciprocal_phase: torch.Tensor,
    *,
    spatial_dims: int = 3,
) -> torch.Tensor:
    """Reconstruct with a centered inverse FFT over two or three spatial axes."""

    reciprocal = reciprocal_field_from_modulus_phase(
        diffraction_modulus,
        reciprocal_phase,
        spatial_dims=spatial_dims,
    )
    dims = tuple(range(-spatial_dims, 0))
    field, had_channel = _without_singleton_channel(
        reciprocal, "reciprocal_field", spatial_dims
    )
    shifted = torch.fft.ifftshift(field, dim=dims)
    realspace = torch.fft.ifftn(shifted, dim=dims)
    realspace = torch.fft.fftshift(realspace, dim=dims)
    return realspace[:, None] if had_channel else realspace


def reciprocal_field_from_realspace(
    realspace: torch.Tensor, *, spatial_dims: int = 3
) -> torch.Tensor:
    """Transform a complex real-space object to the centered reciprocal grid."""

    field, had_channel = _without_singleton_channel(
        realspace, "realspace", spatial_dims
    )
    if not torch.is_complex(field):
        raise ValueError("realspace must be a complex tensor.")
    dims = tuple(range(-spatial_dims, 0))
    shifted = torch.fft.ifftshift(field, dim=dims)
    reciprocal = torch.fft.fftn(shifted, dim=dims)
    reciprocal = torch.fft.fftshift(reciprocal, dim=dims)
    return reciprocal[:, None] if had_channel else reciprocal


def farfield_modulus_from_realspace(
    realspace: torch.Tensor, *, spatial_dims: int = 3
) -> torch.Tensor:
    """Project a complex object back to diffraction modulus for evaluation."""

    return reciprocal_field_from_realspace(realspace, spatial_dims=spatial_dims).abs()


def project_to_measured_modulus(
    realspace: torch.Tensor,
    diffraction_modulus: torch.Tensor,
    *,
    spatial_dims: int = 3,
) -> torch.Tensor:
    """Apply one differentiable Fourier-modulus projection to an object."""

    reciprocal = reciprocal_field_from_realspace(realspace, spatial_dims=spatial_dims)
    field, had_channel = _without_singleton_channel(
        reciprocal, "reciprocal_field", spatial_dims
    )
    modulus, _ = _without_singleton_channel(
        diffraction_modulus,
        "diffraction_modulus",
        spatial_dims,
    )
    if field.shape != modulus.shape:
        raise ValueError("Real-space field and measured modulus must share shape.")
    projected = torch.complex(
        modulus * torch.cos(torch.angle(field)),
        modulus * torch.sin(torch.angle(field)),
    )
    dims = tuple(range(-spatial_dims, 0))
    shifted = torch.fft.ifftshift(projected, dim=dims)
    result = torch.fft.ifftn(shifted, dim=dims)
    result = torch.fft.fftshift(result, dim=dims)
    return result[:, None] if had_channel else result


def complex_to_channels(
    realspace: torch.Tensor, *, spatial_dims: int = 3
) -> torch.Tensor:
    """Convert batched complex 2D/3D objects to two real channels."""

    field, _ = _without_singleton_channel(realspace, "realspace", spatial_dims)
    if not torch.is_complex(field):
        raise ValueError("realspace must be a complex tensor.")
    return torch.stack((field.real, field.imag), dim=1)


def channels_to_complex(
    channels: torch.Tensor, *, spatial_dims: int = 3
) -> torch.Tensor:
    """Convert batched real/imaginary 2D/3D channels to complex data."""

    if channels.ndim != spatial_dims + 2 or channels.shape[1] != 2:
        raise ValueError(
            f"channels must have two channels and {spatial_dims} spatial dimensions."
        )
    return torch.complex(channels[:, 0], channels[:, 1])[:, None]
