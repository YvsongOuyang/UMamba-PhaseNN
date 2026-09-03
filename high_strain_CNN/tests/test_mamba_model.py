"""Structural tests for the optional 8-cubed Mamba model variant."""

import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from pytorch_autophasenn.mamba_block import BidirectionalMamba3D
from pytorch_autophasenn.model import (
    REDUCED_BN_NO_OUTER_SKIP_MAMBA8_VARIANT,
    REDUCED_BN_NO_OUTER_SKIP_VARIANT,
    HighStrainPhaseUNet,
    infer_model_variant,
)
from pytorch_autophasenn.train import parse_args


class IdentityMixer(nn.Module):
    """Small CPU-safe stand-in that follows the official Mamba shape contract."""

    def __init__(self, d_model: int, **_: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class MambaModelTest(unittest.TestCase):
    def test_block_is_identity_at_initialization_and_gate_gets_gradient(self) -> None:
        block = BidirectionalMamba3D(
            channels=8,
            d_model=4,
            d_state=2,
            d_conv=2,
            expand=2,
            mixer_factory=IdentityMixer,
        )
        x = torch.randn(2, 8, 3, 3, 3, requires_grad=True)
        output = block(x)
        torch.testing.assert_close(output, x)
        output.square().mean().backward()
        self.assertIsNotNone(block.alpha.grad)
        self.assertGreater(abs(block.alpha.grad.item()), 0.0)

    def test_scratch_cnn_initialization_matches_base_variant(self) -> None:
        torch.manual_seed(17)
        baseline = HighStrainPhaseUNet(REDUCED_BN_NO_OUTER_SKIP_VARIANT)
        baseline_next_random = torch.rand(4)
        torch.manual_seed(17)
        mamba = HighStrainPhaseUNet(
            REDUCED_BN_NO_OUTER_SKIP_MAMBA8_VARIANT,
            mamba_factory=IdentityMixer,
        )
        mamba_next_random = torch.rand(4)

        baseline_state = baseline.state_dict()
        mamba_state = mamba.state_dict()
        for name, value in baseline_state.items():
            torch.testing.assert_close(mamba_state[name], value)
        torch.testing.assert_close(mamba_next_random, baseline_next_random)

    def test_variant_has_bn_no_outer_skip_and_is_inferred_from_checkpoint(self) -> None:
        model = HighStrainPhaseUNet(
            REDUCED_BN_NO_OUTER_SKIP_MAMBA8_VARIANT,
            mamba_factory=IdentityMixer,
        )
        self.assertTrue(model.use_batch_norm)
        self.assertTrue(model.use_mamba)
        self.assertFalse(model.use_outer_skip)
        self.assertNotIn("conv3d_12", model.layers)
        self.assertEqual(
            infer_model_variant(model.state_dict()),
            REDUCED_BN_NO_OUTER_SKIP_MAMBA8_VARIANT,
        )

    def test_mamba_variant_defaults_to_requested_scratch_learning_rate(self) -> None:
        with patch(
            "sys.argv",
            [
                "train",
                "--model-variant",
                REDUCED_BN_NO_OUTER_SKIP_MAMBA8_VARIANT,
            ],
        ):
            args = parse_args()
        self.assertEqual(args.learning_rate, 1e-3)

    def test_full_model_preserves_volume_shape(self) -> None:
        with torch.device("meta"):
            model = HighStrainPhaseUNet(
                REDUCED_BN_NO_OUTER_SKIP_MAMBA8_VARIANT,
                mamba_factory=IdentityMixer,
            )
            output = model(torch.empty(1, 1, 64, 64, 64, device="meta"))
        self.assertEqual(tuple(output.shape), (1, 1, 64, 64, 64))


if __name__ == "__main__":
    unittest.main()
