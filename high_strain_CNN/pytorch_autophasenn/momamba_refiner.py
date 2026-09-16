"""Sparse-MoMamba refinement of complex BCDI objects after phase retrieval.

This is a clean-room adaptation of Yu et al.'s sparse Mixture of Mambas method.
Their AET network maps scalar density volumes to scalar density volumes; here
the same positional routing and top-k Mamba-expert mechanism operates on the
real and imaginary channels of a complex BCDI reconstruction.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn

from .mamba_block import MambaFactory, build_official_mamba
from .reconstruction import (
    channels_to_complex,
    complex_to_channels,
    farfield_modulus_from_realspace,
    project_to_measured_modulus,
    realspace_from_modulus_phase,
)


@dataclass
class RoutingLosses:
    """Unweighted router regularizers accumulated across MoM blocks."""

    load_balance: torch.Tensor
    router_z: torch.Tensor

    @classmethod
    def zeros(cls, reference: torch.Tensor) -> "RoutingLosses":
        zero = reference.new_zeros(())
        return cls(load_balance=zero, router_z=zero)

    def __add__(self, other: "RoutingLosses") -> "RoutingLosses":
        return RoutingLosses(
            load_balance=self.load_balance + other.load_balance,
            router_z=self.router_z + other.router_z,
        )


@dataclass
class CascadeOutput:
    """Named outputs from reciprocal prediction through real-space refinement."""

    reciprocal_phase: torch.Tensor
    initial_object: torch.Tensor
    proposal_object: torch.Tensor
    refined_object: torch.Tensor
    routing_losses: RoutingLosses


@dataclass
class ObjectLoss:
    """Ambiguity-aware component-MAE loss and selection diagnostics."""

    loss: torch.Tensor
    direct: torch.Tensor
    twin: torch.Tensor
    twin_fraction: torch.Tensor
    real_mae: torch.Tensor
    imag_mae: torch.Tensor
    selected_twin: torch.Tensor
    selected_support: torch.Tensor


class SinusoidalPositionEncoding(nn.Module):
    """Parameter-free absolute encoding used by the paper before routing."""

    def __init__(self, channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        if channels < 2 or channels % 2:
            raise ValueError(
                "Sinusoidal position encoding requires an even channel count."
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Position dropout must be in [0, 1).")
        self.channels = channels
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add flattened-position encodings to ``[B, tokens, channels]``."""

        if x.ndim != 3 or x.shape[-1] != self.channels:
            raise ValueError(f"Expected [B, T, {self.channels}], got {tuple(x.shape)}.")
        positions = torch.arange(x.shape[1], device=x.device, dtype=torch.float32)
        frequencies = torch.exp(
            torch.arange(0, self.channels, 2, device=x.device, dtype=torch.float32)
            * (-math.log(10000.0) / self.channels)
        )
        angles = positions[:, None] * frequencies[None, :]
        encoding = torch.empty(
            x.shape[1], self.channels, device=x.device, dtype=torch.float32
        )
        encoding[:, 0::2] = torch.sin(angles)
        encoding[:, 1::2] = torch.cos(angles)
        return self.dropout(x + encoding.to(dtype=x.dtype)[None])


class MultiHeadRouter(nn.Module):
    """Fused multihead expert router from the Sparse MoMambas formulation."""

    def __init__(
        self,
        channels: int,
        num_experts: int,
        num_heads: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if min(channels, num_experts, num_heads) < 1:
            raise ValueError("Router dimensions must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Router dropout must be in [0, 1).")
        self.num_experts = num_experts
        self.num_heads = num_heads
        self.input_projection = nn.Linear(channels, num_experts * num_heads)
        if num_heads > 1:
            self.activation = nn.GELU()
            self.output_projection = nn.Linear(num_experts * num_heads, num_experts)
            self.dropout = nn.Dropout(dropout)
        else:
            self.activation = None
            self.output_projection = None
            self.dropout = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return pre-softmax expert logits for each token."""

        logits = self.input_projection(x)
        if self.output_projection is not None:
            logits = self.output_projection(self.activation(logits))
            logits = self.dropout(logits)
        return logits


class SparseMixtureOfMambas(nn.Module):
    """Route each position-enhanced token to a sparse set of Mamba experts."""

    def __init__(
        self,
        channels: int,
        num_experts: int = 3,
        top_k: int = 2,
        router_heads: int = 2,
        router_dropout: float = 0.2,
        position_dropout: float = 0.1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        mixer_factory: MambaFactory | None = None,
    ) -> None:
        super().__init__()
        if num_experts < 1 or not 1 <= top_k <= num_experts:
            raise ValueError("Require 1 <= top_k <= num_experts.")
        factory = mixer_factory or build_official_mamba
        self.channels = channels
        self.num_experts = num_experts
        self.top_k = top_k
        self.position_encoding = SinusoidalPositionEncoding(
            channels,
            dropout=position_dropout,
        )
        self.router = (
            MultiHeadRouter(
                channels,
                num_experts,
                num_heads=router_heads,
                dropout=router_dropout,
            )
            if num_experts > 1
            else None
        )
        self.experts = nn.ModuleList(
            factory(
                d_model=channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(num_experts)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, RoutingLosses]:
        """Process ``[B, tokens, channels]`` while preserving token order."""

        if x.ndim != 3 or x.shape[-1] != self.channels:
            raise ValueError(f"Expected [B, T, {self.channels}], got {tuple(x.shape)}.")
        positioned = self.position_encoding(x)
        if self.router is None:
            return self.experts[0](positioned), RoutingLosses.zeros(x)

        logits = self.router(positioned)
        probabilities = torch.softmax(logits, dim=-1)
        _, top_indices = probabilities.topk(self.top_k, dim=-1)
        routed = torch.zeros_like(probabilities, dtype=torch.bool).scatter_(
            dim=-1,
            index=top_indices,
            value=True,
        )

        batch_outputs = []
        for batch_index in range(positioned.shape[0]):
            sample_output = torch.zeros_like(positioned[batch_index])
            for expert_index, expert in enumerate(self.experts):
                token_indices = torch.nonzero(
                    routed[batch_index, :, expert_index], as_tuple=False
                ).flatten()
                if token_indices.numel() == 0:
                    continue
                expert_input = positioned[batch_index].index_select(0, token_indices)[
                    None
                ]
                expert_output = expert(expert_input)[0]
                scores = probabilities[batch_index, token_indices, expert_index][
                    :, None
                ]
                sample_output = sample_output.index_add(
                    0,
                    token_indices,
                    expert_output * scores,
                )
            batch_outputs.append(sample_output)

        expert_fraction = routed.float().mean(dim=(0, 1))
        mean_probability = probabilities.mean(dim=(0, 1))
        losses = RoutingLosses(
            load_balance=self.num_experts
            * torch.sum(expert_fraction * mean_probability),
            router_z=torch.logsumexp(logits, dim=-1).square().mean(),
        )
        return torch.stack(batch_outputs, dim=0), losses


class MoMambaBlock3D(nn.Module):
    """Layer-normalized sparse MoM block with the paper's small residual scale."""

    def __init__(
        self,
        channels: int,
        *,
        num_experts: int,
        top_k: int,
        router_heads: int,
        router_dropout: float,
        position_dropout: float,
        d_state: int,
        d_conv: int,
        expand: int,
        mixer_factory: MambaFactory | None,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.norm = nn.LayerNorm(channels)
        self.mixture = SparseMixtureOfMambas(
            channels,
            num_experts=num_experts,
            top_k=top_k,
            router_heads=router_heads,
            router_dropout=router_dropout,
            position_dropout=position_dropout,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            mixer_factory=mixer_factory,
        )
        self.gamma = nn.Parameter(torch.full((channels,), 1e-6))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, RoutingLosses]:
        """Map an NCDHW feature volume to the same shape."""

        if x.ndim != 5 or x.shape[1] != self.channels:
            raise ValueError(
                f"Expected [B, {self.channels}, D, H, W], got {tuple(x.shape)}."
            )
        shape = x.shape
        sequence = x.flatten(2).transpose(1, 2).contiguous()
        mixed, losses = self.mixture(self.norm(sequence))
        sequence = sequence + self.gamma * mixed
        return sequence.transpose(1, 2).reshape(shape), losses


def _split_complex_channels(
    value: torch.Tensor,
    expected_channels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split concatenated real/imaginary feature channels."""

    if value.ndim != 5 or value.shape[1] != 2 * expected_channels:
        raise ValueError(
            f"Expected [B, {2 * expected_channels}, D, H, W], got "
            f"{tuple(value.shape)}."
        )
    return value[:, :expected_channels], value[:, expected_channels:]


def _merge_complex_channels(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
    if real.shape != imag.shape:
        raise ValueError("Real and imaginary feature tensors must share shape.")
    return torch.cat((real, imag), dim=1)


class ComplexConv3d(nn.Module):
    """Complex 3D convolution implemented with two real-valued kernels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        *,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels) < 1:
            raise ValueError("Complex convolution channel counts must be positive.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.real_kernel = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.imag_kernel = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.real_bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.imag_bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.xavier_uniform_(self.real_kernel.weight)
        nn.init.xavier_uniform_(self.imag_kernel.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real, imag = _split_complex_channels(x, self.in_channels)
        output_real = self.real_kernel(real) - self.imag_kernel(imag)
        output_imag = self.real_kernel(imag) + self.imag_kernel(real)
        if self.real_bias is not None:
            shape = (1, self.out_channels, 1, 1, 1)
            output_real = output_real + self.real_bias.view(shape)
            output_imag = output_imag + self.imag_bias.view(shape)
        return _merge_complex_channels(output_real, output_imag)


class ComplexConvTranspose3d(nn.Module):
    """Complex transposed 3D convolution with shared complex algebra."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        *,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels) < 1:
            raise ValueError("Complex convolution channel counts must be positive.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.real_kernel = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.imag_kernel = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.real_bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.imag_bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.xavier_uniform_(self.real_kernel.weight)
        nn.init.xavier_uniform_(self.imag_kernel.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real, imag = _split_complex_channels(x, self.in_channels)
        output_real = self.real_kernel(real) - self.imag_kernel(imag)
        output_imag = self.real_kernel(imag) + self.imag_kernel(real)
        if self.real_bias is not None:
            shape = (1, self.out_channels, 1, 1, 1)
            output_real = output_real + self.real_bias.view(shape)
            output_imag = output_imag + self.imag_bias.view(shape)
        return _merge_complex_channels(output_real, output_imag)


class ComplexBatchNorm3d(nn.Module):
    """Normalize real and imaginary components independently."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.real_norm = nn.BatchNorm3d(channels)
        self.imag_norm = nn.BatchNorm3d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real, imag = _split_complex_channels(x, self.channels)
        return _merge_complex_channels(self.real_norm(real), self.imag_norm(imag))


class ComplexLeakyReLU(nn.Module):
    """Apply the activation component-wise, as in the reference C-CNN."""

    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.activation = nn.LeakyReLU(negative_slope=negative_slope, inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x)


class ComplexDropout3d(nn.Module):
    """Drop matching real/imaginary feature maps with one shared mask."""

    def __init__(self, channels: int, probability: float) -> None:
        super().__init__()
        self.channels = channels
        self.probability = probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0:
            return x
        real, imag = _split_complex_channels(x, self.channels)
        mask = torch.ones_like(real)
        mask = torch.nn.functional.dropout3d(
            mask,
            p=self.probability,
            training=True,
        )
        return _merge_complex_channels(real * mask, imag * mask)


class ComplexResidualConvBlock3D(nn.Module):
    """Two complex convolutions with a complex residual projection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = ComplexConv3d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = ComplexBatchNorm3d(out_channels)
        self.conv2 = ComplexConv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = ComplexBatchNorm3d(out_channels)
        self.activation = ComplexLeakyReLU(negative_slope=0.01)
        self.projection = (
            nn.Sequential(
                ComplexConv3d(in_channels, out_channels, 1, bias=False),
                ComplexBatchNorm3d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.projection(x)
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)


class MoMCodec3D(nn.Module):
    """Joint-component MoM block followed by complex residual convolutions."""

    def __init__(
        self,
        channels: int,
        *,
        num_experts: int,
        top_k: int,
        router_heads: int,
        router_dropout: float,
        position_dropout: float,
        codec_dropout: float,
        d_state: int,
        d_conv: int,
        expand: int,
        mixer_factory: MambaFactory | None,
    ) -> None:
        super().__init__()
        real_channels = 2 * channels
        self.mom = MoMambaBlock3D(
            real_channels,
            num_experts=num_experts,
            top_k=top_k,
            router_heads=router_heads,
            router_dropout=router_dropout,
            position_dropout=position_dropout,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            mixer_factory=mixer_factory,
        )
        self.conv1 = ComplexResidualConvBlock3D(channels, channels)
        self.conv2 = ComplexResidualConvBlock3D(channels, channels)
        self.residual_projection = nn.Sequential(
            ComplexDropout3d(channels, codec_dropout),
            ComplexConv3d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, RoutingLosses]:
        x, losses = self.mom(x)
        residual = self.conv2(self.conv1(x))
        return x + self.residual_projection(residual), losses


class DownMoMStage(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, **mom_kwargs: object
    ) -> None:
        super().__init__()
        self.downsample = nn.Sequential(
            ComplexConv3d(in_channels, out_channels, 2, stride=2, bias=False),
            ComplexBatchNorm3d(out_channels),
            ComplexLeakyReLU(0.01),
        )
        self.codec = MoMCodec3D(out_channels, **mom_kwargs)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, RoutingLosses]:
        return self.codec(self.downsample(x))


class UpMoMStage(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, **mom_kwargs: object
    ) -> None:
        super().__init__()
        self.upsample = ComplexConvTranspose3d(in_channels, out_channels, 2, stride=2)
        self.fusion = ComplexResidualConvBlock3D(out_channels, out_channels)
        self.codec = MoMCodec3D(out_channels, **mom_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> tuple[torch.Tensor, RoutingLosses]:
        x = self.upsample(x)
        if x.shape != skip.shape:
            raise ValueError(
                f"Decoder and skip shapes differ: {tuple(x.shape)} vs {tuple(skip.shape)}."
            )
        return self.codec(self.fusion(x + skip))


class ComplexMoMambaRefiner(nn.Module):
    """U-shaped sparse-MoMamba residual refiner for a complex 3D object.

    The first MoM stage runs at half resolution, as in the authors' backbone;
    the full-resolution path remains convolutional to bound memory use.
    """

    def __init__(
        self,
        base_channels: int = 4,
        num_experts: Sequence[int] = (3, 3, 3, 3),
        top_k: int = 2,
        router_heads: int = 2,
        router_dropout: float = 0.2,
        position_dropout: float = 0.1,
        codec_dropout: float = 0.1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        mixer_factory: MambaFactory | None = None,
    ) -> None:
        super().__init__()
        if base_channels < 1:
            raise ValueError("base_channels must be positive.")
        if len(num_experts) != 4 or min(num_experts) < 1:
            raise ValueError("num_experts must contain four positive stage values.")
        channels = [base_channels * (2**index) for index in range(5)]
        self.stem = ComplexResidualConvBlock3D(1, channels[0])

        def stage_kwargs(stage: int) -> dict[str, object]:
            experts = int(num_experts[stage])
            return {
                "num_experts": experts,
                "top_k": min(top_k, experts),
                "router_heads": router_heads,
                "router_dropout": router_dropout,
                "position_dropout": position_dropout,
                "codec_dropout": codec_dropout,
                "d_state": d_state,
                "d_conv": d_conv,
                "expand": expand,
                "mixer_factory": mixer_factory,
            }

        self.encoder1 = DownMoMStage(channels[0], channels[1], **stage_kwargs(0))
        self.encoder2 = DownMoMStage(channels[1], channels[2], **stage_kwargs(1))
        self.encoder3 = DownMoMStage(channels[2], channels[3], **stage_kwargs(2))
        self.bottleneck = DownMoMStage(channels[3], channels[4], **stage_kwargs(3))
        self.decoder3 = UpMoMStage(channels[4], channels[3], **stage_kwargs(2))
        self.decoder2 = UpMoMStage(channels[3], channels[2], **stage_kwargs(1))
        self.decoder1 = UpMoMStage(channels[2], channels[1], **stage_kwargs(0))
        self.full_resolution_up = ComplexConvTranspose3d(
            channels[1], channels[0], 2, stride=2
        )
        self.full_resolution_fusion = ComplexResidualConvBlock3D(
            channels[0], channels[0]
        )
        self.output = ComplexConv3d(channels[0], 1, 1)
        nn.init.zeros_(self.output.real_kernel.weight)
        nn.init.zeros_(self.output.imag_kernel.weight)

    def forward(
        self,
        initial_object: torch.Tensor,
    ) -> tuple[torch.Tensor, RoutingLosses]:
        """Refine complex ``[B, 1, D, H, W]`` data without changing its shape."""

        if initial_object.ndim != 5 or initial_object.shape[1] != 1:
            raise ValueError("initial_object must have shape [B, 1, D, H, W].")
        if not torch.is_complex(initial_object):
            raise ValueError("initial_object must be complex.")
        if any(size % 16 for size in initial_object.shape[-3:]):
            raise ValueError("Every spatial dimension must be divisible by 16.")

        scale = (
            initial_object.abs()
            .square()
            .mean(dim=(-3, -2, -1), keepdim=True)
            .sqrt()
            .clamp_min(torch.finfo(initial_object.real.dtype).eps)
        )
        normalized = complex_to_channels(initial_object / scale)
        skip0 = self.stem(normalized)
        skip1, losses = self.encoder1(skip0)
        skip2, current = self.encoder2(skip1)
        losses = losses + current
        skip3, current = self.encoder3(skip2)
        losses = losses + current
        x, current = self.bottleneck(skip3)
        losses = losses + current
        x, current = self.decoder3(x, skip3)
        losses = losses + current
        x, current = self.decoder2(x, skip2)
        losses = losses + current
        x, current = self.decoder1(x, skip1)
        losses = losses + current
        x = self.full_resolution_up(x)
        x = self.full_resolution_fusion(x + skip0)
        delta = self.output(x)
        refined = normalized + delta
        return channels_to_complex(refined) * scale, losses


class HighStrainMoMambaCascade(nn.Module):
    """Frozen phase U-Net, inverse FFT, complex MoMamba refinement, projection."""

    def __init__(
        self,
        phase_model: nn.Module,
        refiner: ComplexMoMambaRefiner,
        *,
        freeze_phase_model: bool = True,
        project_measured_modulus: bool = False,
    ) -> None:
        super().__init__()
        self.phase_model = phase_model
        self.refiner = refiner
        self.freeze_phase_model = freeze_phase_model
        self.project_measured_modulus = project_measured_modulus
        if freeze_phase_model:
            for parameter in self.phase_model.parameters():
                parameter.requires_grad_(False)
            self.phase_model.eval()

    def train(self, mode: bool = True) -> "HighStrainMoMambaCascade":
        super().train(mode)
        if self.freeze_phase_model:
            self.phase_model.eval()
        return self

    def forward(
        self,
        model_input: torch.Tensor,
        diffraction_modulus: torch.Tensor,
    ) -> CascadeOutput:
        """Run the complete reciprocal-to-real-space refinement cascade."""

        if self.freeze_phase_model:
            with torch.no_grad():
                reciprocal_phase = self.phase_model(model_input)
        else:
            reciprocal_phase = self.phase_model(model_input)
        center = tuple(size // 2 for size in reciprocal_phase.shape[-3:])
        center_phase = reciprocal_phase[(slice(None), slice(None)) + center]
        reciprocal_phase = reciprocal_phase - center_phase[..., None, None, None]
        initial_object = realspace_from_modulus_phase(
            diffraction_modulus,
            reciprocal_phase,
        )
        proposal_object, routing_losses = self.refiner(initial_object)
        refined_object = (
            project_to_measured_modulus(proposal_object, diffraction_modulus)
            if self.project_measured_modulus
            else proposal_object
        )
        return CascadeOutput(
            reciprocal_phase=reciprocal_phase,
            initial_object=initial_object,
            proposal_object=proposal_object,
            refined_object=refined_object,
            routing_losses=routing_losses,
        )


def _as_complex_volume(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim == 4:
        value = value[:, None]
    if value.ndim != 5 or value.shape[1] != 1 or not torch.is_complex(value):
        raise ValueError(f"{name} must be complex [B, 1, D, H, W] or [B, D, H, W].")
    return value


def complex_component_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Yu et al. Eq. (1): MAE(real) + MAE(imaginary)."""

    prediction = _as_complex_volume(prediction, "prediction")
    target = _as_complex_volume(target, "target")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match.")
    return (prediction.real - target.real).abs().mean() + (
        prediction.imag - target.imag
    ).abs().mean()


def _candidate_component_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    support: torch.Tensor,
    outside_weight: float,
    loss_scope: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eps = torch.finfo(prediction.real.dtype).eps
    mask = support.to(dtype=prediction.real.dtype)
    pred_scale = (
        prediction.abs()
        .square()
        .mean(dim=(-3, -2, -1), keepdim=True)
        .sqrt()
        .clamp_min(eps)
    )
    target_scale = (
        target.abs().square().mean(dim=(-3, -2, -1), keepdim=True).sqrt().clamp_min(eps)
    )
    prediction = prediction / pred_scale
    target = target / target_scale
    correlation = (target.conj() * prediction * mask).sum(
        dim=(-3, -2, -1), keepdim=True
    )
    phase_offset = torch.angle(correlation)
    prediction = prediction * torch.complex(
        torch.cos(-phase_offset),
        torch.sin(-phase_offset),
    )
    real_error = (prediction.real - target.real).abs()
    imag_error = (prediction.imag - target.imag).abs()
    spatial_dims = (-3, -2, -1)
    if loss_scope == "full":
        real_mae = real_error.mean(dim=spatial_dims)[:, 0]
        imag_mae = imag_error.mean(dim=spatial_dims)[:, 0]
    elif loss_scope == "support_balanced":
        inside_count = mask.sum(dim=spatial_dims).clamp_min(1.0)
        outside_mask = 1.0 - mask
        outside_count = outside_mask.sum(dim=spatial_dims).clamp_min(1.0)
        real_mae = (
            (real_error * mask).sum(dim=spatial_dims) / inside_count
            + outside_weight
            * (real_error * outside_mask).sum(dim=spatial_dims)
            / outside_count
        )[:, 0]
        imag_mae = (
            (imag_error * mask).sum(dim=spatial_dims) / inside_count
            + outside_weight
            * (imag_error * outside_mask).sum(dim=spatial_dims)
            / outside_count
        )[:, 0]
    else:
        raise ValueError("loss_scope must be 'full' or 'support_balanced'.")
    return real_mae + imag_mae, real_mae, imag_mae


def ambiguity_aware_component_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    support: torch.Tensor,
    *,
    outside_weight: float = 0.25,
    loss_scope: str = "full",
) -> ObjectLoss:
    """Apply Eq. (1) after resolving unavoidable BCDI ambiguities.

    Each object is RMS-normalized, the prediction is aligned by one global
    phase, and the lower direct or conjugate-inverted target is selected. The
    default ``full`` scope then remains exactly real MAE plus imaginary MAE.
    """

    if outside_weight < 0:
        raise ValueError("outside_weight must be nonnegative.")
    prediction = _as_complex_volume(prediction, "prediction")
    target = _as_complex_volume(target, "target")
    if support.ndim == 4:
        support = support[:, None]
    if support.shape != target.shape:
        raise ValueError("support and target shapes must match.")
    support = support.bool()
    direct, direct_real, direct_imag = _candidate_component_mae(
        prediction,
        target,
        support,
        outside_weight,
        loss_scope,
    )
    # With an even, centered FFT grid, negating reciprocal phase maps to a
    # conjugate inversion plus one wrapped voxel along every spatial axis.
    twin_target = torch.conj(
        torch.roll(
            torch.flip(target, dims=(-3, -2, -1)),
            shifts=(1, 1, 1),
            dims=(-3, -2, -1),
        )
    )
    twin_support = torch.roll(
        torch.flip(support, dims=(-3, -2, -1)),
        shifts=(1, 1, 1),
        dims=(-3, -2, -1),
    )
    twin, twin_real, twin_imag = _candidate_component_mae(
        prediction,
        twin_target,
        twin_support,
        outside_weight,
        loss_scope,
    )
    select_twin = twin < direct
    selected = torch.where(select_twin, twin, direct)
    return ObjectLoss(
        loss=selected.mean(),
        direct=direct.mean(),
        twin=twin.mean(),
        twin_fraction=select_twin.float().mean(),
        real_mae=torch.where(select_twin, twin_real, direct_real).mean(),
        imag_mae=torch.where(select_twin, twin_imag, direct_imag).mean(),
        selected_twin=select_twin,
        selected_support=torch.where(
            select_twin[:, None, None, None, None],
            twin_support,
            support,
        ),
    )


def ambiguity_aware_complex_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    support: torch.Tensor,
    *,
    outside_weight: float = 0.25,
) -> ObjectLoss:
    """Backward-compatible name for the support-balanced legacy objective."""

    return ambiguity_aware_component_mae(
        prediction,
        target,
        support,
        outside_weight=outside_weight,
        loss_scope="support_balanced",
    )


def diffraction_modulus_mae(
    realspace: torch.Tensor,
    measured_modulus: torch.Tensor,
    support: torch.Tensor | None = None,
) -> torch.Tensor:
    """Yu et al. Eq. (2): normalized Fourier-modulus MAE.

    The known synthetic support is applied before the FFT, mirroring the fixed
    support used by the authors for experimental-data refinement. Both moduli
    are normalized to unit maximum per sample because the paper's measured
    diffraction amplitude is normalized to [0, 1].
    """

    realspace = _as_complex_volume(realspace, "realspace")
    if support is not None:
        if support.ndim == 4:
            support = support[:, None]
        if support.shape != realspace.shape:
            raise ValueError("support and realspace shapes must match.")
        realspace = realspace * support.to(dtype=realspace.real.dtype)
    predicted = farfield_modulus_from_realspace(realspace)
    if measured_modulus.ndim == 4:
        measured_modulus = measured_modulus[:, None]
    if predicted.shape != measured_modulus.shape:
        raise ValueError("Predicted and measured modulus shapes must match.")
    eps = torch.finfo(predicted.dtype).eps
    predicted_scale = predicted.amax(dim=(-3, -2, -1), keepdim=True).clamp_min(eps)
    measured_scale = measured_modulus.amax(dim=(-3, -2, -1), keepdim=True).clamp_min(
        eps
    )
    return (
        (predicted / predicted_scale - measured_modulus / measured_scale).abs().mean()
    )


def diffraction_modulus_rmse(
    realspace: torch.Tensor,
    measured_modulus: torch.Tensor,
) -> torch.Tensor:
    """Legacy scale-normalized reciprocal-modulus RMSE."""

    predicted = farfield_modulus_from_realspace(realspace)
    if predicted.shape != measured_modulus.shape:
        raise ValueError("Predicted and measured modulus shapes must match.")
    eps = torch.finfo(predicted.dtype).eps
    scale = (
        measured_modulus.square()
        .mean(dim=(-3, -2, -1), keepdim=True)
        .sqrt()
        .clamp_min(eps)
    )
    return ((predicted / scale - measured_modulus / scale).square().mean() + eps).sqrt()


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
