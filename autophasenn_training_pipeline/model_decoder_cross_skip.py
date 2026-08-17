"""AutoPhaseNN with bidirectional decoder cross-skips at 8^3 and 16^3."""

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


AMPLITUDE_DECODER_LAYERS = (
    "conv3d_10",
    "batch_normalization_10",
    "conv3d_11",
    "batch_normalization_11",
    "conv3d_12",
    "batch_normalization_12",
    "conv3d_13",
    "batch_normalization_13",
    "conv3d_14",
    "batch_normalization_14",
    "conv3d_15",
    "batch_normalization_15",
    "conv3d_16",
    "batch_normalization_16",
    "conv3d_17",
    "batch_normalization_17",
    "conv3d_18",
)

PHASE_DECODER_LAYERS = (
    "conv3d_19",
    "batch_normalization_18",
    "conv3d_20",
    "batch_normalization_19",
    "conv3d_21",
    "batch_normalization_20",
    "conv3d_22",
    "batch_normalization_21",
    "conv3d_23",
    "batch_normalization_22",
    "conv3d_24",
    "batch_normalization_23",
    "conv3d_25",
    "batch_normalization_24",
    "conv3d_26",
    "batch_normalization_25",
    "conv3d_27",
)

TRAINABLE_STAGES = ("cross_skip", "decoders", "all")


class CrossDecoderSkip(nn.Module):
    """Exchange same-scale amplitude and phase features by residual addition."""

    def __init__(self, amp_channels: int, phase_channels: int) -> None:
        super().__init__()
        self.phase_to_amp = nn.Conv3d(phase_channels, amp_channels, kernel_size=1)
        self.amp_to_phase = nn.Conv3d(amp_channels, phase_channels, kernel_size=1)
        self.alpha_amp = nn.Parameter(torch.zeros(1))
        self.alpha_phase = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        amp: torch.Tensor,
        phase: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update both branches simultaneously from their unchanged inputs."""

        amp_old = amp
        phase_old = phase
        amp_delta = self.phase_to_amp(phase_old)
        phase_delta = self.amp_to_phase(amp_old)
        return (
            amp_old + self.alpha_amp * amp_delta,
            phase_old + self.alpha_phase * phase_delta,
        )


class DecoderCrossSkipAutoPhaseNN(TFCompatibleAutoPhaseNN):
    """Baseline AutoPhaseNN with two bidirectional decoder cross-skips."""

    def __init__(self, threshold: float = 0.1) -> None:
        super().__init__(threshold=threshold)
        self.cross_skip_8 = CrossDecoderSkip(amp_channels=256, phase_channels=128)
        self.cross_skip_16 = CrossDecoderSkip(amp_channels=128, phase_channels=128)
        self.trainable_stage = "all"

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return the baseline six outputs with cross-skips at 8^3 and 16^3."""

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
        amp, phase = self.cross_skip_8(amp, phase)

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
        amp, phase = self.cross_skip_16(amp, phase)

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

    def train(self, mode: bool = True) -> DecoderCrossSkipAutoPhaseNN:
        """Set module mode while keeping frozen BatchNorm statistics fixed."""

        super().train(mode)
        if not mode or self.trainable_stage == "all":
            return self

        self.layers.eval()
        if self.trainable_stage == "decoders":
            for name in AMPLITUDE_DECODER_LAYERS + PHASE_DECODER_LAYERS:
                self.layers[name].train(True)
        return self

    def set_trainable_stage(self, stage: str) -> None:
        """Select cross-skip-only, decoder, or full-network fine-tuning."""

        if stage not in TRAINABLE_STAGES:
            raise ValueError(f"Unknown trainable stage: {stage}")

        self.trainable_stage = stage
        for parameter in self.parameters():
            parameter.requires_grad = stage == "all"
        if stage == "all":
            self.train(self.training)
            return

        for module in (self.cross_skip_8, self.cross_skip_16):
            for parameter in module.parameters():
                parameter.requires_grad = True
        if stage == "cross_skip":
            self.train(self.training)
            return

        for name in AMPLITUDE_DECODER_LAYERS + PHASE_DECODER_LAYERS:
            for parameter in self.layers[name].parameters():
                parameter.requires_grad = True
        self.train(self.training)

    def cross_skip_strengths(self) -> dict[str, float]:
        """Return scalar residual strengths for experiment tracking."""

        return {
            "8/phase_to_amp": float(self.cross_skip_8.alpha_amp.detach()),
            "8/amp_to_phase": float(self.cross_skip_8.alpha_phase.detach()),
            "16/phase_to_amp": float(self.cross_skip_16.alpha_amp.detach()),
            "16/amp_to_phase": float(self.cross_skip_16.alpha_phase.detach()),
        }


def initialize_from_baseline_state_dict(
    model: DecoderCrossSkipAutoPhaseNN,
    baseline_state_dict: Mapping[str, torch.Tensor],
) -> None:
    """Copy every baseline parameter while retaining new cross-skip values."""

    target_state = model.state_dict()
    baseline_keys = set(baseline_state_dict)
    target_keys = set(target_state)
    unexpected = sorted(baseline_keys.difference(target_keys))
    missing = sorted(target_keys.difference(baseline_keys))
    expected_missing = sorted(
        key
        for key in target_keys
        if key.startswith("cross_skip_8.") or key.startswith("cross_skip_16.")
    )
    if unexpected or missing != expected_missing:
        raise RuntimeError(
            "Baseline checkpoint keys do not match DecoderCrossSkipAutoPhaseNN: "
            f"missing={missing}, unexpected={unexpected}. Use --resume for an "
            "existing decoder_cross_skip checkpoint."
        )

    adapted_state = dict(target_state)
    for key, source in baseline_state_dict.items():
        target = target_state[key]
        if source.shape != target.shape:
            raise RuntimeError(
                f"Unexpected baseline shape mismatch for {key}: "
                f"checkpoint={tuple(source.shape)}, model={tuple(target.shape)}."
            )
        adapted_state[key] = source
    model.load_state_dict(adapted_state, strict=True)


def load_baseline_weights(
    model: DecoderCrossSkipAutoPhaseNN,
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> object:
    """Initialize a decoder-cross-skip model from a baseline checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    initialize_from_baseline_state_dict(model, state_dict)
    return checkpoint
