"""AutoPhaseNN with bidirectional decoder cross-concats at 8^3 and 16^3."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn

try:
    from .model_tf_compatible import TFCompatibleAutoPhaseNN
except ImportError:
    from model_tf_compatible import TFCompatibleAutoPhaseNN


EXPANDED_INPUT_CHANNELS = {
    "conv3d_12": (256, 128),
    "conv3d_21": (128, 256),
    "conv3d_14": (128, 128),
    "conv3d_23": (128, 128),
}


class DecoderCrossConcatAutoPhaseNN(TFCompatibleAutoPhaseNN):
    """Baseline AutoPhaseNN with bidirectional decoder feature concatenation."""

    def __init__(self, threshold: float = 0.1) -> None:
        super().__init__(threshold=threshold)
        for name, (branch_channels, cross_channels) in EXPANDED_INPUT_CHANNELS.items():
            original = self.layers[name]
            self.layers[name] = nn.Conv3d(
                branch_channels + cross_channels,
                original.out_channels,
                kernel_size=original.kernel_size,
                stride=original.stride,
                padding=original.padding,
                dilation=original.dilation,
                groups=original.groups,
                bias=original.bias is not None,
                padding_mode=original.padding_mode,
            )

    @staticmethod
    def _bidirectional_concat(
        amp: torch.Tensor,
        phase: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build both branch inputs simultaneously from unchanged features."""

        if amp.shape[0] != phase.shape[0] or amp.shape[-3:] != phase.shape[-3:]:
            raise ValueError(
                "Amplitude and phase decoder features must match in batch and space; "
                f"got amplitude={tuple(amp.shape)}, phase={tuple(phase.shape)}."
            )
        return torch.cat((amp, phase), dim=1), torch.cat((phase, amp), dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return the baseline six outputs with cross-concats at 8^3 and 16^3."""

        encoded = self.encode(x)

        amp = self._up_lrelu_block(
            encoded,
            "conv3d_10",
            "batch_normalization_10",
            "conv3d_11",
            "batch_normalization_11",
        )
        phase = self._up_lrelu_block(
            encoded,
            "conv3d_19",
            "batch_normalization_18",
            "conv3d_20",
            "batch_normalization_19",
        )
        amp, phase = self._bidirectional_concat(amp, phase)

        amp = self._up_lrelu_block(
            amp,
            "conv3d_12",
            "batch_normalization_12",
            "conv3d_13",
            "batch_normalization_13",
        )
        phase = self._up_lrelu_block(
            phase,
            "conv3d_21",
            "batch_normalization_20",
            "conv3d_22",
            "batch_normalization_21",
        )
        amp, phase = self._bidirectional_concat(amp, phase)

        amp = self._up_lrelu_block(
            amp,
            "conv3d_14",
            "batch_normalization_14",
            "conv3d_15",
            "batch_normalization_15",
        )
        phase = self._up_lrelu_block(
            phase,
            "conv3d_23",
            "batch_normalization_22",
            "conv3d_24",
            "batch_normalization_23",
        )
        amp = self._pad_relu_block(
            amp,
            "conv3d_16",
            "batch_normalization_16",
            "conv3d_17",
            "batch_normalization_17",
        )
        phase = self._pad_relu_block(
            phase,
            "conv3d_25",
            "batch_normalization_24",
            "conv3d_26",
            "batch_normalization_25",
        )

        amp = torch.sigmoid(self.layers["conv3d_18"](amp))
        phase = math.pi * torch.tanh(self.layers["conv3d_27"](phase))
        return self._apply_forward_physics(amp, phase)


def initialize_from_baseline_state_dict(
    model: DecoderCrossConcatAutoPhaseNN,
    baseline_state_dict: Mapping[str, torch.Tensor],
) -> None:
    """Copy baseline kernels and zero only the added cross-feature channels."""

    target_state = model.state_dict()
    missing = sorted(set(target_state).difference(baseline_state_dict))
    unexpected = sorted(set(baseline_state_dict).difference(target_state))
    if missing or unexpected:
        raise RuntimeError(
            "Baseline checkpoint keys do not match DecoderCrossConcatAutoPhaseNN: "
            f"missing={missing}, unexpected={unexpected}."
        )

    expanded_weights = {
        f"layers.{name}.weight": channels
        for name, channels in EXPANDED_INPUT_CHANNELS.items()
    }
    adapted_state: dict[str, torch.Tensor] = {}
    for key, target in target_state.items():
        source = baseline_state_dict[key]
        if key in expanded_weights:
            branch_channels, cross_channels = expanded_weights[key]
            expected_source_shape = (
                target.shape[0],
                branch_channels,
                *target.shape[2:],
            )
            expected_target_channels = branch_channels + cross_channels
            if tuple(source.shape) != expected_source_shape:
                raise RuntimeError(
                    f"Expected baseline convolution {key} with shape "
                    f"{expected_source_shape}, got {tuple(source.shape)}. Use --resume "
                    "for an existing decoder_cross_concat checkpoint."
                )
            if target.shape[1] != expected_target_channels:
                raise RuntimeError(
                    f"Cross-concat convolution {key} has {target.shape[1]} input "
                    f"channels; expected {expected_target_channels}."
                )
            expanded = source.new_zeros(target.shape)
            expanded[:, :branch_channels] = source
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
    model: DecoderCrossConcatAutoPhaseNN,
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> object:
    """Initialize a decoder-cross-concat model from a baseline checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    initialize_from_baseline_state_dict(model, state_dict)
    return checkpoint
