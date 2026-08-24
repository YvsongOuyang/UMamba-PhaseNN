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
    from .model_decoder_cross_concat import (
        DecoderCrossConcatAutoPhaseNN,
        load_baseline_weights as load_decoder_cross_concat_baseline_weights,
    )
    from .model_mamba_skip import (
        AutoPhaseNNBiPVMSkip,
        load_baseline_weights as load_mamba_skip_baseline_weights,
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
    from model_decoder_cross_concat import (
        DecoderCrossConcatAutoPhaseNN,
        load_baseline_weights as load_decoder_cross_concat_baseline_weights,
    )
    from model_mamba_skip import (
        AutoPhaseNNBiPVMSkip,
        load_baseline_weights as load_mamba_skip_baseline_weights,
    )
    from model_residual import ResidualAutoPhaseNN
    from model_tf_compatible import TFCompatibleAutoPhaseNN, load_weights


MODEL_VARIANTS = (
    "baseline",
    "residual",
    "amplitude_skip",
    "decoder_cross_skip",
    "decoder_cross_concat",
    "mamba_skip",
)

DEFAULT_SUPPORT_THRESHOLDS = {
    model_variant: (0.3 if model_variant == "mamba_skip" else 0.1)
    for model_variant in MODEL_VARIANTS
}


def default_support_threshold(model_variant: str) -> float:
    """Return the validated operating threshold for one model variant."""

    try:
        return DEFAULT_SUPPORT_THRESHOLDS[model_variant]
    except KeyError as exc:
        raise ValueError(f"Unknown model variant: {model_variant}") from exc


def resolve_support_threshold(
    model_variant: str,
    threshold: float | None,
) -> float:
    """Use a caller override or the variant-specific operating threshold."""

    return (
        default_support_threshold(model_variant)
        if threshold is None
        else float(threshold)
    )


def create_model(model_variant: str, threshold: float | None = None) -> nn.Module:
    """Construct a model with the common AutoPhaseNN forward contract."""

    threshold = resolve_support_threshold(model_variant, threshold)

    if model_variant == "baseline":
        return TFCompatibleAutoPhaseNN(threshold=threshold)
    if model_variant == "residual":
        return ResidualAutoPhaseNN(threshold=threshold)
    if model_variant == "amplitude_skip":
        return AmplitudeSkipAutoPhaseNN(threshold=threshold)
    if model_variant == "decoder_cross_skip":
        return DecoderCrossSkipAutoPhaseNN(threshold=threshold)
    if model_variant == "decoder_cross_concat":
        return DecoderCrossConcatAutoPhaseNN(threshold=threshold)
    if model_variant == "mamba_skip":
        return AutoPhaseNNBiPVMSkip(threshold=threshold)
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
    if model_variant == "decoder_cross_concat":
        if not isinstance(model, DecoderCrossConcatAutoPhaseNN):
            raise TypeError(
                "decoder_cross_concat requires DecoderCrossConcatAutoPhaseNN."
            )
        return load_decoder_cross_concat_baseline_weights(
            model,
            checkpoint_path,
            map_location=map_location,
        )
    if model_variant == "mamba_skip":
        if not isinstance(model, AutoPhaseNNBiPVMSkip):
            raise TypeError("mamba_skip requires AutoPhaseNNBiPVMSkip.")
        return load_mamba_skip_baseline_weights(
            model,
            checkpoint_path,
            map_location=map_location,
        )
    return load_weights(model, checkpoint_path, map_location=map_location)
