"""2D adapters sharing the existing high-strain network architecture."""

from __future__ import annotations

from pytorch_autophasenn.model import (
    HighStrainPhaseUNet,
    REDUCED_BN_NO_OUTER_SKIP_VARIANT,
)
from pytorch_autophasenn.momamba_refiner import (
    ComplexMoMambaRefiner,
    HighStrainMoMambaCascade,
)


class HighStrainPhaseUNet2D(HighStrainPhaseUNet):
    """Predict phase of shape [B, 1, 512, 512] from equal-shaped input.

    The reduced graph has encoder resolutions 512, 256, 128, 64, 32 and a
    16-square, 1024-channel bottleneck. Smaller multiples of 32 are accepted
    by the shared backbone for focused tests; production data are 512-square.
    """

    def __init__(self, model_variant=REDUCED_BN_NO_OUTER_SKIP_VARIANT, **kwargs):
        super().__init__(model_variant, spatial_dims=2, **kwargs)


class ComplexMoMambaRefiner2D(ComplexMoMambaRefiner):
    """Refine complex [B, 1, 512, 512] fields with 2D convolutions.

    Sparse Mamba stages operate at 256, 128, 64 and 32 square resolutions.
    The zero-initialized complex output preserves the initial object exactly.
    """

    def __init__(self, **kwargs):
        super().__init__(spatial_dims=2, **kwargs)


class HighStrainMoMambaCascade2D(HighStrainMoMambaCascade):
    """Phase prediction -> inverse FFT2 -> complex 2D refinement."""

    def __init__(self, phase_model, refiner, **kwargs):
        if refiner.spatial_dims != 2 or getattr(phase_model, "spatial_dims", 2) != 2:
            raise ValueError("The 2D cascade requires a 2D phase model and refiner.")
        super().__init__(phase_model, refiner, **kwargs)
