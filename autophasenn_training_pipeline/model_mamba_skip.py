"""AutoPhaseNN with independent Bi-PVM encoder-to-decoder skip bridges."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .model_tf_compatible import TFCompatibleAutoPhaseNN
except ImportError:
    from model_tf_compatible import TFCompatibleAutoPhaseNN


MAMBA_WIDTH = 32
PVM_GROUPS = 4
PVM_GROUP_WIDTH = MAMBA_WIDTH // PVM_GROUPS
MAMBA_SKIP_CONV_NAMES = (
    "conv3d_10",
    "conv3d_12",
    "conv3d_19",
    "conv3d_21",
)
MAMBA_SKIP_WEIGHT_KEYS = frozenset(
    f"layers.{name}.weight" for name in MAMBA_SKIP_CONV_NAMES
)

MambaFactory: TypeAlias = Callable[..., nn.Module]
MAMBA_SKIP_PREFIXES = (
    "amp_skip8.",
    "amp_skip16.",
    "phase_skip8.",
    "phase_skip16.",
)


def _resolve_mamba_factory() -> MambaFactory:
    """Load mamba-ssm only when the Bi-PVM model is instantiated."""

    try:
        from mamba_ssm import Mamba
    except ImportError as exc:
        raise ImportError(
            "The mamba_skip model requires the 'mamba-ssm' package used by "
            "the repository's existing UMamba implementation."
        ) from exc
    return Mamba


class _BidirectionalMambaGroup(nn.Module):
    """Apply independent forward and reverse Mamba operators to one group."""

    def __init__(
        self,
        mamba_factory: MambaFactory,
        *,
        d_model: int = PVM_GROUP_WIDTH,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.forward_mamba = mamba_factory(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.backward_mamba = mamba_factory(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Preserve ``(B, N, d_model)`` while mixing both scan directions."""

        forward = self.forward_mamba(x)
        backward = torch.flip(
            self.backward_mamba(torch.flip(x, dims=(1,))),
            dims=(1,),
        )
        return x + forward + backward


class BiPVMBridge(nn.Module):
    """Project a 3D encoder feature through one four-group Bi-PVM block."""

    def __init__(
        self,
        in_channels: int,
        *,
        width: int = MAMBA_WIDTH,
        groups: int = PVM_GROUPS,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        mamba_factory: MambaFactory | None = None,
    ) -> None:
        super().__init__()
        if width % groups != 0:
            raise ValueError(f"Bi-PVM width {width} must be divisible by {groups} groups.")

        group_width = width // groups
        factory = mamba_factory or _resolve_mamba_factory()
        self.width = width
        self.groups = groups
        self.input_projection = nn.Conv3d(in_channels, width, kernel_size=1)
        self.local_mixer = nn.Conv3d(
            width,
            width,
            kernel_size=3,
            padding=1,
            groups=width,
        )
        self.norm = nn.LayerNorm(width)
        self.pvm_groups = nn.ModuleList(
            [
                _BidirectionalMambaGroup(
                    factory,
                    d_model=group_width,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(groups)
            ]
        )
        self.output_projection = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, Cin, D, H, W)`` to ``(B, 32, D, H, W)``."""

        x = self.input_projection(x)
        x = x + self.local_mixer(x)
        batch_size, channels, depth, height, width = x.shape
        sequence = x.flatten(2).transpose(1, 2)
        sequence = self.norm(sequence)
        chunks = torch.chunk(sequence, self.groups, dim=-1)
        sequence = torch.cat(
            [block(chunk) for block, chunk in zip(self.pvm_groups, chunks)],
            dim=-1,
        )
        sequence = self.output_projection(sequence)
        return sequence.transpose(1, 2).reshape(
            batch_size,
            channels,
            depth,
            height,
            width,
        )


class AutoPhaseNNBiPVMSkip(TFCompatibleAutoPhaseNN):
    """Baseline AutoPhaseNN with independent 8^3 and 16^3 Bi-PVM skips."""

    def __init__(
        self,
        threshold: float = 0.3,
        *,
        mamba_factory: MambaFactory | None = None,
    ) -> None:
        super().__init__(threshold=threshold)

        for name in MAMBA_SKIP_CONV_NAMES:
            original = self.layers[name]
            self.layers[name] = nn.Conv3d(
                original.in_channels + MAMBA_WIDTH,
                original.out_channels,
                kernel_size=original.kernel_size,
                stride=original.stride,
                padding=original.padding,
                dilation=original.dilation,
                groups=original.groups,
                bias=original.bias is not None,
                padding_mode=original.padding_mode,
            )

        bridge_kwargs = {"mamba_factory": mamba_factory}
        self.amp_skip8 = BiPVMBridge(128, **bridge_kwargs)
        self.amp_skip16 = BiPVMBridge(64, **bridge_kwargs)
        self.phase_skip8 = BiPVMBridge(128, **bridge_kwargs)
        self.phase_skip16 = BiPVMBridge(64, **bridge_kwargs)

    def encode_with_skips(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return bottleneck plus post-pooling ``16^3`` and ``8^3`` features."""

        x = self._encoder_block(
            x,
            "conv3d",
            "batch_normalization",
            "conv3d_1",
            "batch_normalization_1",
        )
        e16 = self._encoder_block(
            x,
            "conv3d_2",
            "batch_normalization_2",
            "conv3d_3",
            "batch_normalization_3",
        )
        e8 = self._encoder_block(
            e16,
            "conv3d_4",
            "batch_normalization_4",
            "conv3d_5",
            "batch_normalization_5",
        )
        x = self._encoder_block(
            e8,
            "conv3d_6",
            "batch_normalization_6",
            "conv3d_7",
            "batch_normalization_7",
        )
        bottleneck = self._encoder_block(
            x,
            "conv3d_8",
            "batch_normalization_8",
            "conv3d_9",
            "batch_normalization_9",
            pool=False,
        )
        return bottleneck, e16, e8

    def _up_mamba_skip_lrelu_block(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        conv1: str,
        bn1: str,
        conv2: str,
        bn2: str,
    ) -> torch.Tensor:
        """Upsample, concatenate one Bi-PVM skip, and run a decoder block."""

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if x.shape[0] != skip.shape[0] or x.shape[-3:] != skip.shape[-3:]:
            raise ValueError(
                "Decoder and Bi-PVM skip features must match in batch and space; "
                f"got decoder={tuple(x.shape)}, skip={tuple(skip.shape)}."
            )
        x = torch.cat((x, skip), dim=1)
        x = self._conv_lrelu_bn(x, conv1, bn1)
        return self._conv_lrelu_bn(x, conv2, bn2)

    def decode_amplitude_with_skips(
        self,
        encoded: torch.Tensor,
        e16: torch.Tensor,
        e8: torch.Tensor,
    ) -> torch.Tensor:
        """Decode amplitude with independent Bi-PVM skips at 8^3 and 16^3."""

        x = self._up_mamba_skip_lrelu_block(
            encoded,
            self.amp_skip8(e8),
            "conv3d_10",
            "batch_normalization_10",
            "conv3d_11",
            "batch_normalization_11",
        )
        x = self._up_mamba_skip_lrelu_block(
            x,
            self.amp_skip16(e16),
            "conv3d_12",
            "batch_normalization_12",
            "conv3d_13",
            "batch_normalization_13",
        )
        x = self._up_lrelu_block(
            x,
            "conv3d_14",
            "batch_normalization_14",
            "conv3d_15",
            "batch_normalization_15",
        )
        x = self._pad_relu_block(
            x,
            "conv3d_16",
            "batch_normalization_16",
            "conv3d_17",
            "batch_normalization_17",
        )
        return torch.sigmoid(self.layers["conv3d_18"](x))

    def decode_phase_with_skips(
        self,
        encoded: torch.Tensor,
        e16: torch.Tensor,
        e8: torch.Tensor,
    ) -> torch.Tensor:
        """Decode phase with independent Bi-PVM skips at 8^3 and 16^3."""

        x = self._up_mamba_skip_lrelu_block(
            encoded,
            self.phase_skip8(e8),
            "conv3d_19",
            "batch_normalization_18",
            "conv3d_20",
            "batch_normalization_19",
        )
        x = self._up_mamba_skip_lrelu_block(
            x,
            self.phase_skip16(e16),
            "conv3d_21",
            "batch_normalization_20",
            "conv3d_22",
            "batch_normalization_21",
        )
        x = self._up_lrelu_block(
            x,
            "conv3d_23",
            "batch_normalization_22",
            "conv3d_24",
            "batch_normalization_23",
        )
        x = self._pad_relu_block(
            x,
            "conv3d_25",
            "batch_normalization_24",
            "conv3d_26",
            "batch_normalization_25",
        )
        return math.pi * torch.tanh(self.layers["conv3d_27"](x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return the baseline six outputs for ``(B, 1, 64, 64, 64)`` input."""

        encoded, e16, e8 = self.encode_with_skips(x)
        amp = self.decode_amplitude_with_skips(encoded, e16, e8)
        phi = self.decode_phase_with_skips(encoded, e16, e8)
        return self._apply_forward_physics(amp, phi)


def initialize_from_baseline_state_dict(
    model: AutoPhaseNNBiPVMSkip,
    baseline_state_dict: Mapping[str, torch.Tensor],
) -> None:
    """Copy baseline weights and zero the four added Mamba input slices."""

    target_state = model.state_dict()
    baseline_keys = set(baseline_state_dict)
    target_keys = set(target_state)
    unexpected = sorted(baseline_keys.difference(target_keys))
    missing = sorted(target_keys.difference(baseline_keys))
    expected_missing = sorted(
        key for key in target_keys if key.startswith(MAMBA_SKIP_PREFIXES)
    )
    if unexpected or missing != expected_missing:
        raise RuntimeError(
            "Baseline checkpoint keys do not match AutoPhaseNNBiPVMSkip: "
            f"missing={missing}, unexpected={unexpected}. Use --resume for an "
            "existing mamba_skip checkpoint."
        )

    adapted_state = dict(target_state)
    for key, source in baseline_state_dict.items():
        target = target_state[key]
        if key in MAMBA_SKIP_WEIGHT_KEYS:
            expected_target_shape = (
                source.shape[0],
                source.shape[1] + MAMBA_WIDTH,
                *source.shape[2:],
            )
            if tuple(target.shape) != expected_target_shape:
                raise RuntimeError(
                    f"Mamba-skip convolution {key} has shape {tuple(target.shape)}; "
                    f"expected {expected_target_shape}."
                )
            expanded = source.new_zeros(target.shape)
            expanded[:, : source.shape[1]] = source
            adapted_state[key] = expanded
            continue
        if source.shape == target.shape:
            adapted_state[key] = source
            continue
        raise RuntimeError(
            f"Unexpected baseline shape mismatch for {key}: "
            f"checkpoint={tuple(source.shape)}, model={tuple(target.shape)}."
        )
    model.load_state_dict(adapted_state, strict=True)


def load_baseline_weights(
    model: AutoPhaseNNBiPVMSkip,
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> object:
    """Initialize a Mamba-Skip model from a standard baseline checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    initialize_from_baseline_state_dict(model, state_dict)
    return checkpoint
