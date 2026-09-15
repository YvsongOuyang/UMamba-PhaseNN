"""Focused tests for second-stage evaluation metrics."""

import unittest

import torch

from pytorch_autophasenn.evaluate_refiner import (
    phase_wca,
    relative_improvement,
    support_metrics,
)
from pytorch_autophasenn.reconstruction import realspace_from_modulus_phase


class EvaluateRefinerTest(unittest.TestCase):
    def test_phase_wca_is_zero_for_matching_object(self) -> None:
        modulus = torch.rand(2, 1, 4, 4, 4) + 0.1
        target_phase = torch.randn(2, 4, 4, 4)
        center = (2, 2, 2)
        target_phase = target_phase - target_phase[(slice(None),) + center][
            :, None, None, None
        ]
        realspace = realspace_from_modulus_phase(modulus, target_phase)
        weights = torch.rand_like(modulus) + 0.1
        self.assertLess(float(phase_wca(realspace, target_phase, weights)), 1e-6)

    def test_support_metrics_use_unit_maximum_amplitude(self) -> None:
        amplitude = torch.tensor([[[[[1.0, 0.4, 0.1, 0.0]]]]])
        realspace = torch.complex(amplitude, torch.zeros_like(amplitude))
        support = torch.tensor([[[[[True, True, False, False]]]]])
        values = support_metrics(realspace, support, threshold=0.3)
        self.assertAlmostEqual(float(values["support_iou"]), 1.0)
        self.assertAlmostEqual(float(values["support_dice"]), 1.0)
        self.assertAlmostEqual(float(values["support_volume_ratio"]), 1.0)

    def test_relative_improvement_obeys_metric_direction(self) -> None:
        self.assertAlmostEqual(relative_improvement("object_mae", 0.2, 0.1), 50.0)
        self.assertAlmostEqual(relative_improvement("support_iou", 0.4, 0.6), 50.0)
        self.assertAlmostEqual(
            relative_improvement("support_volume_ratio", 1.4, 1.2),
            50.0,
        )
        self.assertIsNone(relative_improvement("twin_fraction", 0.4, 0.5))


if __name__ == "__main__":
    unittest.main()
