"""Contract tests for the reciprocal-to-real-space MoMamba cascade."""

import unittest

import numpy as np
import torch
import torch.nn as nn

from pytorch_autophasenn.author_data import prepare_author_training_sample
from pytorch_autophasenn.momamba_refiner import (
    ComplexConv3d,
    ComplexMoMambaRefiner,
    HighStrainMoMambaCascade,
    SparseMixtureOfMambas,
    ambiguity_aware_component_mae,
    complex_component_mae,
    diffraction_modulus_mae,
)
from pytorch_autophasenn.reconstruction import (
    farfield_modulus_from_realspace,
    project_to_measured_modulus,
    reciprocal_field_from_realspace,
    realspace_from_modulus_phase,
)


class IdentityMixer(nn.Module):
    def __init__(self, d_model: int, **_: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class ZeroPhaseModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class TrainablePhaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MoMambaRefinerTest(unittest.TestCase):
    def test_compact_author_sample_exposes_physical_refinement_fields(self) -> None:
        shape = (4, 4, 4)
        stored = {
            "I": np.arange(1, 65, dtype=np.float32).reshape(shape),
            "phi": np.zeros(shape, dtype=np.float32),
            "object": np.ones(shape, dtype=np.complex64) * (1 + 2j),
            "support": np.ones(shape, dtype=np.uint8),
        }
        sample = prepare_author_training_sample(
            stored,
            input_log_data=True,
            shape=shape,
            return_refinement_targets=True,
        )
        torch.testing.assert_close(
            sample["diffraction"][0],
            torch.from_numpy(np.sqrt(stored["I"]) / np.sqrt(stored["I"]).max()),
        )
        self.assertTrue(torch.is_complex(sample["realspace"]))
        self.assertEqual(sample["support"].dtype, torch.bool)

    def test_sparse_router_preserves_shape_and_routes_top_k(self) -> None:
        layer = SparseMixtureOfMambas(
            channels=4,
            num_experts=3,
            top_k=2,
            router_heads=2,
            router_dropout=0.0,
            d_state=2,
            d_conv=2,
            expand=2,
            mixer_factory=IdentityMixer,
        )
        output, losses = layer(torch.randn(2, 12, 4))
        self.assertEqual(tuple(output.shape), (2, 12, 4))
        self.assertTrue(torch.isfinite(losses.load_balance))
        self.assertTrue(torch.isfinite(losses.router_z))

    def test_complex_convolution_uses_complex_multiplication(self) -> None:
        layer = ComplexConv3d(1, 1, 1, bias=False)
        with torch.no_grad():
            layer.real_kernel.weight.fill_(2.0)
            layer.imag_kernel.weight.fill_(3.0)
        value = torch.tensor([[[[[4.0]]], [[[5.0]]]]])
        output = layer(value)
        torch.testing.assert_close(output[:, 0], torch.tensor([[[[-7.0]]]]))
        torch.testing.assert_close(output[:, 1], torch.tensor([[[[22.0]]]]))

    def test_zero_initialized_output_is_exact_identity(self) -> None:
        model = ComplexMoMambaRefiner(
            base_channels=2,
            num_experts=(2, 2, 2, 2),
            top_k=1,
            router_heads=2,
            router_dropout=0.0,
            codec_dropout=0.0,
            d_state=2,
            d_conv=2,
            expand=2,
            mixer_factory=IdentityMixer,
        ).eval()
        initial = torch.complex(
            torch.randn(1, 1, 16, 16, 16),
            torch.randn(1, 1, 16, 16, 16),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        refined, _ = model(initial)
        torch.testing.assert_close(refined, initial)
        (refined.real.square().mean()).backward()
        gradient = model.output.real_kernel.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        refined, _ = model(initial)
        (refined.real.square().mean()).backward()
        upstream_gradient = model.stem.conv1.real_kernel.weight.grad
        self.assertIsNotNone(upstream_gradient)
        self.assertGreater(float(upstream_gradient.abs().sum()), 0.0)

    def test_cascade_keeps_phase_network_frozen(self) -> None:
        phase_model = TrainablePhaseModel()
        refiner = ComplexMoMambaRefiner(
            base_channels=1,
            num_experts=(1, 1, 1, 1),
            top_k=1,
            router_heads=1,
            router_dropout=0.0,
            codec_dropout=0.0,
            d_state=2,
            d_conv=2,
            expand=2,
            mixer_factory=IdentityMixer,
        )
        cascade = HighStrainMoMambaCascade(phase_model, refiner)
        cascade.train()
        self.assertFalse(cascade.phase_model.training)
        self.assertTrue(all(not p.requires_grad for p in phase_model.parameters()))
        model_input = torch.rand(2, 1, 16, 16, 16)
        modulus = torch.rand_like(model_input) + 0.1
        output = cascade(model_input, modulus)
        output.refined_object.real.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in phase_model.parameters()))
        self.assertTrue(any(p.grad is not None for p in refiner.parameters()))

    def test_cascade_and_projection_preserve_measured_modulus(self) -> None:
        refiner = ComplexMoMambaRefiner(
            base_channels=2,
            num_experts=(1, 1, 1, 1),
            top_k=1,
            router_heads=1,
            router_dropout=0.0,
            codec_dropout=0.0,
            d_state=2,
            d_conv=2,
            expand=2,
            mixer_factory=IdentityMixer,
        ).eval()
        cascade = HighStrainMoMambaCascade(
            ZeroPhaseModel(), refiner, project_measured_modulus=True
        ).eval()
        model_input = torch.rand(1, 1, 16, 16, 16)
        modulus = torch.rand_like(model_input) + 0.1
        output = cascade(model_input, modulus)
        torch.testing.assert_close(output.proposal_object, output.initial_object)
        torch.testing.assert_close(
            farfield_modulus_from_realspace(output.refined_object),
            modulus,
            rtol=2e-5,
            atol=2e-5,
        )

    def test_projection_round_trip_and_ambiguity_aware_loss(self) -> None:
        modulus = torch.rand(2, 1, 16, 16, 16) + 0.1
        phase = torch.randn_like(modulus)
        target = realspace_from_modulus_phase(modulus, phase)
        proposal = target * torch.exp(torch.tensor(0.7j))
        projected = project_to_measured_modulus(proposal, modulus)
        torch.testing.assert_close(
            farfield_modulus_from_realspace(projected),
            modulus,
            rtol=2e-5,
            atol=2e-5,
        )
        support = torch.ones(2, 16, 16, 16, dtype=torch.bool)
        loss = ambiguity_aware_component_mae(proposal, target[:, 0], support)
        self.assertLess(float(loss.loss), 1e-5)

        reciprocal = reciprocal_field_from_realspace(target)
        twin = realspace_from_modulus_phase(
            reciprocal.abs(),
            -torch.angle(reciprocal),
        )
        twin_loss = ambiguity_aware_component_mae(twin, target[:, 0], support)
        self.assertLess(float(twin_loss.loss), 1e-5)
        self.assertEqual(float(twin_loss.twin_fraction), 1.0)

    def test_yu_component_and_fourier_losses(self) -> None:
        prediction = torch.complex(
            torch.tensor([[[[[2.0, 0.0]]]]]),
            torch.tensor([[[[[1.0, 3.0]]]]]),
        )
        target = torch.complex(
            torch.tensor([[[[[1.0, 0.0]]]]]),
            torch.tensor([[[[[0.0, 1.0]]]]]),
        )
        self.assertAlmostEqual(float(complex_component_mae(prediction, target)), 2.0)

        realspace = torch.complex(
            torch.randn(2, 1, 4, 4, 4),
            torch.randn(2, 1, 4, 4, 4),
        )
        modulus = farfield_modulus_from_realspace(realspace)
        support = torch.ones(2, 4, 4, 4, dtype=torch.bool)
        self.assertLess(
            float(diffraction_modulus_mae(realspace, modulus, support)), 1e-6
        )


if __name__ == "__main__":
    unittest.main()
