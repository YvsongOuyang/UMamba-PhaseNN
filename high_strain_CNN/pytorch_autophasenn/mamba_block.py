"""Mamba-based global context for a compact 3D decoder feature map."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn


MambaFactory = Callable[..., nn.Module]


def build_official_mamba(**kwargs: int) -> nn.Module:
    """Construct the official selective state-space Mamba implementation."""

    try:
        from mamba_ssm import Mamba
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "The Mamba model variant requires the official mamba-ssm package. "
            "Install the pinned optional dependency described in "
            "requirements/mamba.txt and README.md."
        ) from error
    return Mamba(**kwargs)


class BidirectionalMamba3D(nn.Module):
    """Apply a shared Mamba mixer in both directions over flattened 3D tokens."""

    def __init__(
        self,
        channels: int,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        mixer_factory: MambaFactory | None = None,
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()
        if min(channels, d_model, d_state, d_conv, expand) < 1:
            raise ValueError("Mamba dimensions must all be positive.")

        factory = mixer_factory or build_official_mamba
        if spatial_dims not in (2, 3):
            raise ValueError("spatial_dims must be 2 or 3.")
        self.spatial_dims = spatial_dims
        conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        self.channels = channels
        self.input_projection = conv(channels, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)
        self.mixer = factory(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.output_projection = conv(d_model, channels, kernel_size=1)
        self.alpha = nn.Parameter(torch.zeros(()))
        self.reset_adapter_parameters()

    def reset_adapter_parameters(self) -> None:
        """Initialize only the projections and gate around the official mixer."""

        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        nn.init.zeros_(self.alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[B, C, D, H, W]`` to the same shape with global context."""

        if x.ndim != self.spatial_dims + 2 or x.shape[1] != self.channels:
            raise ValueError(
                f"Expected [B, {self.channels}, D, H, W], got {tuple(x.shape)}."
            )

        projected = self.input_projection(x)
        shape = projected.shape
        sequence = projected.flatten(2).transpose(1, 2).contiguous()
        sequence = self.norm(sequence)

        forward = self.mixer(sequence)
        reversed_sequence = torch.flip(sequence, dims=(1,)).contiguous()
        backward = torch.flip(
            self.mixer(reversed_sequence),
            dims=(1,),
        ).contiguous()
        mixed = 0.5 * (forward + backward)

        volume = mixed.transpose(1, 2).reshape(shape)
        delta = self.output_projection(volume)
        return x + torch.tanh(self.alpha) * delta
