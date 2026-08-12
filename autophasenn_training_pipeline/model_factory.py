"""Model selection shared by AutoPhaseNN training and evaluation entry points."""

import torch
from torch import nn

try:
    from .model_dual_skip import DualPairwiseSkipAutoPhaseNN, load_baseline_weights
    from .model_residual import ResidualAutoPhaseNN
    from .model_tf_compatible import TFCompatibleAutoPhaseNN, load_weights
except ImportError:
    from model_dual_skip import DualPairwiseSkipAutoPhaseNN, load_baseline_weights
    from model_residual import ResidualAutoPhaseNN
    from model_tf_compatible import TFCompatibleAutoPhaseNN, load_weights


MODEL_VARIANTS = ("baseline", "residual", "dual_skip")


def create_model(model_variant: str, threshold: float) -> nn.Module:
    """Construct a model with the common AutoPhaseNN forward contract."""

    if model_variant == "baseline":
        return TFCompatibleAutoPhaseNN(threshold=threshold)
    if model_variant == "residual":
        return ResidualAutoPhaseNN(threshold=threshold)
    if model_variant == "dual_skip":
        return DualPairwiseSkipAutoPhaseNN(threshold=threshold)
    raise ValueError(f"Unknown model variant: {model_variant}")


def load_pretrained_weights(
    model: nn.Module,
    model_variant: str,
    checkpoint_path: str,
    map_location: str | torch.device = "cpu",
) -> object:
    """Load standard weights or migrate a baseline checkpoint into dual-skip."""

    if model_variant == "dual_skip":
        if not isinstance(model, DualPairwiseSkipAutoPhaseNN):
            raise TypeError("dual_skip requires DualPairwiseSkipAutoPhaseNN.")
        return load_baseline_weights(model, checkpoint_path, map_location=map_location)
    return load_weights(model, checkpoint_path, map_location=map_location)
