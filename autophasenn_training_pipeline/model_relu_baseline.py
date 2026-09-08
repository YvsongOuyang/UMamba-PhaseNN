"""AutoPhaseNN baseline variant using ReLU for all hidden activations."""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from .model_tf_compatible import TFCompatibleAutoPhaseNN
except ImportError:
    from model_tf_compatible import TFCompatibleAutoPhaseNN


class ReLUBaselineAutoPhaseNN(TFCompatibleAutoPhaseNN):
    """Keep the baseline architecture while replacing LeakyReLU with ReLU."""

    def _conv_lrelu_bn(
        self,
        x: torch.Tensor,
        conv_name: str,
        bn_name: str,
    ) -> torch.Tensor:
        """Apply the inherited Conv3D and BatchNorm with a ReLU activation."""

        x = self.layers[conv_name](x)
        x = F.relu(x)
        return self.layers[bn_name](x)
