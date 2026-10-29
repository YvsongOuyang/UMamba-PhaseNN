"""Versioned phase U-Net with a full-resolution skip and safe weight migration.

The shared backbone is not modified.  ``outer_skip_v4`` adds a 16-channel
full-resolution path to the BatchNorm U-Net.  Its new fusion weights start at
zero, preserving the function of a migrated baseline checkpoint initially.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from pytorch_autophasenn.model import (
    BN_EPS,
    BN_MOMENTUM,
    HighStrainPhaseUNet,
    REDUCED_BN_NO_OUTER_SKIP_VARIANT,
    REDUCED_BN_RELU_NO_OUTER_SKIP_VARIANT,
    TensorFlowSameConv3d,
)


BASELINE_ARCHITECTURE = "baseline"
OUTER_SKIP_ARCHITECTURE = "outer_skip_v4"
PHASE_ARCHITECTURES = (BASELINE_ARCHITECTURE, OUTER_SKIP_ARCHITECTURE)
_FUSION_WEIGHT = "layers.conv3d_19.conv.weight"
_NEW_PREFIXES = ("layers.conv3d_12.", "batch_norms.conv3d_12.")


class HighStrainPhaseUNet2D(HighStrainPhaseUNet):
    """The unchanged 2D baseline, with explicit architecture metadata."""

    phase_architecture = BASELINE_ARCHITECTURE

    def __init__(self, model_variant=REDUCED_BN_NO_OUTER_SKIP_VARIANT, **kwargs):
        super().__init__(model_variant, spatial_dims=2, **kwargs)


class HighStrainPhaseUNetOuterSkip2D(HighStrainPhaseUNet2D):
    """Preserve 512-square features through a newly enabled outer skip.

    At initialization the 16 new fusion channels are zero, while the original
    32 channels and every existing BatchNorm layer retain their values.  The
    new fusion channels receive gradients immediately.  Gradients reach the
    new skip convolution after those fusion weights move away from zero.
    """

    phase_architecture = OUTER_SKIP_ARCHITECTURE

    def __init__(self, model_variant=REDUCED_BN_NO_OUTER_SKIP_VARIANT, **kwargs):
        if model_variant not in {
            REDUCED_BN_NO_OUTER_SKIP_VARIANT,
            REDUCED_BN_RELU_NO_OUTER_SKIP_VARIANT,
        }:
            raise ValueError(
                "outer_skip_v4 supports the reduced BatchNorm baseline variants "
                "without Mamba only."
            )
        super().__init__(model_variant, **kwargs)
        original_fusion = self.layers["conv3d_19"]
        # Do not shift the caller's random stream relative to the baseline.
        with torch.random.fork_rng(devices=[]):
            skip = TensorFlowSameConv3d(33, 16, 3, spatial_dims=2)
            nn.init.xavier_uniform_(skip.conv.weight)
            nn.init.zeros_(skip.conv.bias)
            fusion = TensorFlowSameConv3d(48, 16, 5, spatial_dims=2)
            with torch.no_grad():
                fusion.conv.weight.zero_()
                fusion.conv.weight[:, :32].copy_(original_fusion.conv.weight)
                fusion.conv.bias.copy_(original_fusion.conv.bias)
            skip_bn = nn.BatchNorm2d(
                16,
                eps=BN_EPS,
                momentum=BN_MOMENTUM,
                affine=True,
                track_running_stats=True,
            )
        self.layers["conv3d_12"] = skip
        self.layers["conv3d_19"] = fusion
        self.batch_norms["conv3d_12"] = skip_bn
        self.use_outer_skip = True


def phase_architecture(config: Mapping) -> str:
    """Read an architecture explicitly; legacy configs mean baseline."""

    architecture = config["training"].get("phase_architecture", BASELINE_ARCHITECTURE)
    if architecture not in PHASE_ARCHITECTURES:
        raise ValueError(f"Unknown phase_architecture {architecture!r}.")
    return architecture


def migrate_baseline_state_dict(
    model: HighStrainPhaseUNetOuterSkip2D,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    """Copy every old parameter and buffer, adding only the new skip tensors.

    Source tensors are never modified.  Missing or extra baseline tensors and
    unexpected shapes fail explicitly instead of silently dropping weights.
    """

    target = model.state_dict()
    original_keys = {key for key in target if not key.startswith(_NEW_PREFIXES)}
    if set(state_dict) != original_keys:
        missing = sorted(original_keys - set(state_dict))
        extra = sorted(set(state_dict) - original_keys)
        raise ValueError(f"Invalid baseline state: missing={missing}, extra={extra}.")
    migrated = dict(target)
    for key, source in state_dict.items():
        if key == _FUSION_WEIGHT:
            if tuple(source.shape) != (16, 32, 5, 5):
                raise ValueError(f"Unexpected baseline fusion shape: {tuple(source.shape)}.")
            expanded = target[key].clone()
            expanded.zero_()
            expanded[:, :32].copy_(source)
            migrated[key] = expanded
        else:
            if source.shape != target[key].shape:
                raise ValueError(
                    f"Checkpoint shape mismatch for {key}: "
                    f"{tuple(source.shape)} vs {tuple(target[key].shape)}."
                )
            migrated[key] = source
    model.load_state_dict(migrated, strict=True)


def build_phase_model(
    config: Mapping,
    checkpoint: Mapping | None = None,
    *,
    migrate_baseline: bool = False,
    mixer_factory=None,
) -> HighStrainPhaseUNet2D:
    """Build from ``training.phase_architecture`` and optionally load weights.

    ``checkpoint`` must be a phase-stage native 2D checkpoint.  Use
    ``migrate_baseline=True`` only for a *new* outer-skip experiment initialized
    from a baseline checkpoint.  Resume and evaluation use the saved v4 config
    and load strictly with the default ``False``.
    """

    architecture = phase_architecture(config)
    variant = config["training"]["model_variant"]
    cls = (
        HighStrainPhaseUNetOuterSkip2D
        if architecture == OUTER_SKIP_ARCHITECTURE
        else HighStrainPhaseUNet2D
    )
    model = cls(variant, mamba_factory=mixer_factory)
    if checkpoint is None:
        if migrate_baseline:
            raise ValueError("migrate_baseline=True requires a phase checkpoint.")
        return model
    if checkpoint.get("stage") != "phase" or checkpoint.get("spatial_dims") != 2:
        raise ValueError("A native 2D phase-stage checkpoint is required.")
    source_config = checkpoint["config"]
    source_architecture = phase_architecture(source_config)
    declared_architecture = checkpoint.get("phase_architecture", source_architecture)
    if declared_architecture != source_architecture:
        raise ValueError("Checkpoint architecture metadata conflicts with its config.")
    if source_config["training"]["model_variant"] != variant:
        raise ValueError("Checkpoint and requested model_variant differ.")
    if architecture == source_architecture:
        if migrate_baseline:
            raise ValueError("Migration requires baseline -> outer_skip_v4.")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    elif (
        architecture == OUTER_SKIP_ARCHITECTURE
        and source_architecture == BASELINE_ARCHITECTURE
        and migrate_baseline
    ):
        migrate_baseline_state_dict(model, checkpoint["model_state_dict"])
    else:
        raise ValueError(
            f"Cannot load {source_architecture!r} into {architecture!r}; "
            "baseline migration must be explicitly enabled for initialization."
        )
    return model
