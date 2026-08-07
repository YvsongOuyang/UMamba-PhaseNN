"""Residual AutoPhaseNN variant with strided-convolution downsampling."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


BN_EPS = 1e-3
BN_MOMENTUM = 0.01
LEAKY_RELU_SLOPE = 0.01


class ResidualBlock3D(nn.Module):
    """Two-convolution 3D residual block with optional channel projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn1 = nn.BatchNorm3d(
            out_channels,
            eps=BN_EPS,
            momentum=BN_MOMENTUM,
            affine=True,
            track_running_stats=True,
        )
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn2 = nn.BatchNorm3d(
            out_channels,
            eps=BN_EPS,
            momentum=BN_MOMENTUM,
            affine=True,
            track_running_stats=True,
        )
        self.projection = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)
        )

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, negative_slope=LEAKY_RELU_SLOPE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, Cin, D, H, W)`` to ``(B, Cout, D, H, W)``."""

        identity = self.projection(x)
        residual = self.conv1(x)
        residual = self._activate(residual)
        residual = self.bn1(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return self._activate(residual + identity)


class ResidualAutoPhaseNN(nn.Module):
    """AutoPhaseNN with residual convolution blocks and learned downsampling."""

    def __init__(self, threshold: float = 0.1) -> None:
        super().__init__()
        self.threshold = threshold

        self.encoder_blocks = nn.ModuleList(
            [
                ResidualBlock3D(1, 32),
                ResidualBlock3D(32, 64),
                ResidualBlock3D(64, 128),
                ResidualBlock3D(128, 256),
            ]
        )
        self.downsample_layers = nn.ModuleList(
            [
                nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.Conv3d(128, 128, kernel_size=3, stride=2, padding=1),
                nn.Conv3d(256, 256, kernel_size=3, stride=2, padding=1),
            ]
        )
        self.bottleneck = ResidualBlock3D(256, 512)

        self.amplitude_blocks = nn.ModuleList(
            [
                ResidualBlock3D(512, 256),
                ResidualBlock3D(256, 128),
                ResidualBlock3D(128, 64),
                ResidualBlock3D(64, 32),
            ]
        )
        self.phase_blocks = nn.ModuleList(
            [
                ResidualBlock3D(512, 128),
                ResidualBlock3D(128, 128),
                ResidualBlock3D(128, 64),
                ResidualBlock3D(64, 32),
            ]
        )
        self.zero_pad = nn.ConstantPad3d(16, 0.0)
        self.amplitude_output = nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1)
        self.phase_output = nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, 1, 64, 64, 64)`` into ``(B, 512, 4, 4, 4)``."""

        for block, downsample in zip(self.encoder_blocks, self.downsample_layers):
            x = block(x)
            x = downsample(x)
        return self.bottleneck(x)

    @staticmethod
    def _upsample(x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2, mode="nearest")

    def decode_amplitude(self, encoded: torch.Tensor) -> torch.Tensor:
        """Decode the shared representation into normalized amplitude."""

        x = encoded
        for block in self.amplitude_blocks[:-1]:
            x = block(self._upsample(x))
        x = self.amplitude_blocks[-1](self.zero_pad(x))
        return torch.sigmoid(self.amplitude_output(x))

    def decode_phase(self, encoded: torch.Tensor) -> torch.Tensor:
        """Decode the shared representation into phase in ``[-pi, pi]``."""

        x = encoded
        for block in self.phase_blocks[:-1]:
            x = block(self._upsample(x))
        x = self.phase_blocks[-1](self.zero_pad(x))
        return math.pi * torch.tanh(self.phase_output(x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return the same six tensors as the baseline AutoPhaseNN.

        Args:
            x: Diffraction modulus with shape ``(B, 1, 64, 64, 64)``.

        Returns:
            ``(farfield, masked_obj, masked_amp, phi, support, amp)``; every
            tensor has shape ``(B, 1, 64, 64, 64)``.
        """

        encoded = self.encode(x)
        amp = self.decode_amplitude(encoded)
        phi = self.decode_phase(encoded)

        support = torch.where(
            amp >= self.threshold,
            torch.ones_like(amp),
            torch.zeros_like(amp),
        )
        obj = torch.complex(amp * torch.cos(phi), amp * torch.sin(phi))
        masked_obj = obj * support.to(torch.complex64)

        shifted = torch.fft.ifftshift(masked_obj, dim=(-3, -2, -1))
        farfield = torch.fft.fftn(shifted, dim=(-3, -2, -1))
        farfield = torch.fft.fftshift(farfield, dim=(-3, -2, -1))
        farfield = torch.abs(farfield).to(torch.float32)

        masked_amp = torch.abs(masked_obj).to(torch.float32)
        return farfield, masked_obj, masked_amp, phi, support, amp


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
