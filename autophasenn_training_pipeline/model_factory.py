"""Model selection shared by AutoPhaseNN training and evaluation entry points."""

from torch import nn

try:
    from .model_residual import ResidualAutoPhaseNN
    from .model_tf_compatible import TFCompatibleAutoPhaseNN
except ImportError:
    from model_residual import ResidualAutoPhaseNN
    from model_tf_compatible import TFCompatibleAutoPhaseNN


MODEL_VARIANTS = ("baseline", "residual")


def create_model(model_variant: str, threshold: float) -> nn.Module:
    """Construct a model with the common AutoPhaseNN forward contract."""

    if model_variant == "baseline":
        return TFCompatibleAutoPhaseNN(threshold=threshold)
    if model_variant == "residual":
        return ResidualAutoPhaseNN(threshold=threshold)
    raise ValueError(f"Unknown model variant: {model_variant}")
