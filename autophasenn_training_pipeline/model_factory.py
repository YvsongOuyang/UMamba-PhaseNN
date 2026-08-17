"""Model selection shared by AutoPhaseNN training and evaluation entry points."""

import torch
from torch import nn

try:
    from .model_amplitude_skip import (
        AmplitudeSkipAutoPhaseNN,
        load_baseline_weights as load_amplitude_skip_baseline_weights,
    )
    from .model_decoder_cross_skip import (
        DecoderCrossSkipAutoPhaseNN,
        load_baseline_weights as load_decoder_cross_skip_baseline_weights,
    )
    from .model_residual import ResidualAutoPhaseNN
    from .model_tf_compatible import TFCompatibleAutoPhaseNN, load_weights
except ImportError:
    from model_amplitude_skip import (
        AmplitudeSkipAutoPhaseNN,
        load_baseline_weights as load_amplitude_skip_baseline_weights,
    )
    from model_decoder_cross_skip import (
        DecoderCrossSkipAutoPhaseNN,
        load_baseline_weights as load_decoder_cross_skip_baseline_weights,
    )
    from model_residual import ResidualAutoPhaseNN
    from model_tf_compatible import TFCompatibleAutoPhaseNN, load_weights


MODEL_VARIANTS = ("baseline", "residual", "amplitude_skip", "decoder_cross_skip")


def create_model(model_variant: str, threshold: float) -> nn.Module:
    """Construct a model with the common AutoPhaseNN forward contract."""

    if model_variant == "baseline":
        return TFCompatibleAutoPhaseNN(threshold=threshold)
    if model_variant == "residual":
        return ResidualAutoPhaseNN(threshold=threshold)
    if model_variant == "amplitude_skip":
        return AmplitudeSkipAutoPhaseNN(threshold=threshold)
    if model_variant == "decoder_cross_skip":
        return DecoderCrossSkipAutoPhaseNN(threshold=threshold)
    raise ValueError(f"Unknown model variant: {model_variant}")


def load_pretrained_weights(
    model: nn.Module,
    model_variant: str,
    checkpoint_path: str,
    map_location: str | torch.device = "cpu",
) -> object:
    """Load standard weights or migrate a baseline checkpoint into a variant."""

    if model_variant == "amplitude_skip":
        if not isinstance(model, AmplitudeSkipAutoPhaseNN):
            raise TypeError("amplitude_skip requires AmplitudeSkipAutoPhaseNN.")
        return load_amplitude_skip_baseline_weights(
            model,
            checkpoint_path,
            map_location=map_location,
        )
    if model_variant == "decoder_cross_skip":
        if not isinstance(model, DecoderCrossSkipAutoPhaseNN):
            raise TypeError(
                "decoder_cross_skip requires DecoderCrossSkipAutoPhaseNN."
            )
        return load_decoder_cross_skip_baseline_weights(
            model,
            checkpoint_path,
            map_location=map_location,
        )
    return load_weights(model, checkpoint_path, map_location=map_location)
