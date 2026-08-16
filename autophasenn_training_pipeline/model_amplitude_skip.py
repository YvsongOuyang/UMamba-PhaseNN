"""AutoPhaseNN with 8^3 and 16^3 encoder skips in the amplitude decoder."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .model_tf_compatible import TFCompatibleAutoPhaseNN
except ImportError:
    from model_tf_compatible import TFCompatibleAutoPhaseNN


EXPANDED_INPUT_CHANNELS = {
    "conv3d_10": (512, 256),
    "conv3d_12": (256, 128),
}


class AmplitudeSkipAutoPhaseNN(TFCompatibleAutoPhaseNN):
    """Baseline AutoPhaseNN with encoder skips only in the amplitude decoder.

    Pre-pooling 8^3 and 16^3 encoder features are concatenated into the
    amplitude branch at matching scales. The phase decoder remains identical
    to the baseline and receives only the shared bottleneck feature.
    """

    def __init__(self, threshold: float = 0.1) -> None:
        super().__init__(threshold=threshold)
        for name, (decoder_channels, skip_channels) in EXPANDED_INPUT_CHANNELS.items():
            original = self.layers[name]
            self.layers[name] = nn.Conv3d(
                decoder_channels + skip_channels,
                original.out_channels,
                kernel_size=original.kernel_size,
                stride=original.stride,
                padding=original.padding,
                dilation=original.dilation,
                groups=original.groups,
                bias=original.bias is not None,
                padding_mode=original.padding_mode,
            )

    def _encoder_block_without_pool(
        self,
        x: torch.Tensor,
        conv1: str,
        bn1: str,
        conv2: str,
        bn2: str,
    ) -> torch.Tensor:
        x = self._conv_lrelu_bn(x, conv1, bn1)
        return self._conv_lrelu_bn(x, conv2, bn2)

    def encode(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return bottleneck plus pre-pooling 8^3 and 16^3 encoder features."""

        x = self._encoder_block_without_pool(
            x,
            "conv3d",
            "batch_normalization",
            "conv3d_1",
            "batch_normalization_1",
        )
        x = F.max_pool3d(x, kernel_size=2, stride=2)
        x = self._encoder_block_without_pool(
            x,
            "conv3d_2",
            "batch_normalization_2",
            "conv3d_3",
            "batch_normalization_3",
        )
        x = F.max_pool3d(x, kernel_size=2, stride=2)
        skip_16 = self._encoder_block_without_pool(
            x,
            "conv3d_4",
            "batch_normalization_4",
            "conv3d_5",
            "batch_normalization_5",
        )
        x = F.max_pool3d(skip_16, kernel_size=2, stride=2)
        skip_8 = self._encoder_block_without_pool(
            x,
            "conv3d_6",
            "batch_normalization_6",
            "conv3d_7",
            "batch_normalization_7",
        )
        x = F.max_pool3d(skip_8, kernel_size=2, stride=2)
        bottleneck = self._encoder_block_without_pool(
            x,
            "conv3d_8",
            "batch_normalization_8",
            "conv3d_9",
            "batch_normalization_9",
        )
        return bottleneck, skip_8, skip_16

    def _up_skip_lrelu_block(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        conv1: str,
        bn1: str,
        conv2: str,
        bn2: str,
    ) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if x.shape[0] != skip.shape[0] or x.shape[-3:] != skip.shape[-3:]:
            raise ValueError(
                "Decoder and encoder skip shapes must match in batch and space; "
                f"got decoder={tuple(x.shape)}, skip={tuple(skip.shape)}."
            )
        x = torch.cat((x, skip), dim=1)
        x = self._conv_lrelu_bn(x, conv1, bn1)
        return self._conv_lrelu_bn(x, conv2, bn2)

    def decode_amplitude(
        self,
        encoded: torch.Tensor,
        skip_8: torch.Tensor,
        skip_16: torch.Tensor,
    ) -> torch.Tensor:
        """Decode amplitude with 8^3 and 16^3 encoder skips."""

        x = self._up_skip_lrelu_block(
            encoded,
            skip_8,
            "conv3d_10",
            "batch_normalization_10",
            "conv3d_11",
            "batch_normalization_11",
        )
        x = self._up_skip_lrelu_block(
            x,
            skip_16,
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return the baseline six-output contract for a diffraction volume."""

        encoded, skip_8, skip_16 = self.encode(x)
        amp = self.decode_amplitude(encoded, skip_8, skip_16)
        phi = self.decode_phase(encoded)
        return self._apply_forward_physics(amp, phi)


def initialize_from_baseline_state_dict(
    model: AmplitudeSkipAutoPhaseNN,
    baseline_state_dict: Mapping[str, torch.Tensor],
) -> None:
    """Copy baseline weights and zero only the two added amplitude skip kernels."""

    target_state = model.state_dict()
    missing = sorted(set(target_state).difference(baseline_state_dict))
    unexpected = sorted(set(baseline_state_dict).difference(target_state))
    if missing or unexpected:
        raise RuntimeError(
            "Baseline checkpoint keys do not match AmplitudeSkipAutoPhaseNN: "
            f"missing={missing}, unexpected={unexpected}."
        )

    adapted_state: dict[str, torch.Tensor] = {}
    expanded_weights = {
        f"layers.{name}.weight": channels
        for name, channels in EXPANDED_INPUT_CHANNELS.items()
    }
    for key, target in target_state.items():
        source = baseline_state_dict[key]
        if key in expanded_weights:
            decoder_channels, skip_channels = expanded_weights[key]
            expected_source_shape = (
                target.shape[0],
                decoder_channels,
                *target.shape[2:],
            )
            expected_target_channels = decoder_channels + skip_channels
            if tuple(source.shape) != expected_source_shape:
                raise RuntimeError(
                    f"Expected baseline convolution {key} with shape "
                    f"{expected_source_shape}, got {tuple(source.shape)}. Use --resume "
                    "for an existing amplitude_skip checkpoint."
                )
            if target.shape[1] != expected_target_channels:
                raise RuntimeError(
                    f"Amplitude-skip convolution {key} has {target.shape[1]} input "
                    f"channels; expected {expected_target_channels}."
                )
            expanded = source.new_zeros(target.shape)
            expanded[:, :decoder_channels] = source
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
    model: AmplitudeSkipAutoPhaseNN,
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> object:
    """Initialize an amplitude-skip model from a baseline checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    initialize_from_baseline_state_dict(model, state_dict)
    return checkpoint
