import math

import torch

from autophasenn_training_pipeline.evaluate import generate_free_mask
from autophasenn_training_pipeline.losses import (
    chi2_free,
    free_metric_dict,
    llk_free,
    r_factor_free,
    windowed_ssim_3d,
)


def test_windowed_ssim_is_one_for_identical_volumes():
    volume = torch.rand((2, 1, 7, 7, 7), generator=torch.Generator().manual_seed(7))

    score = windowed_ssim_3d(volume, volume, window_size=7)

    assert torch.allclose(score, torch.tensor(1.0), atol=1e-6)


def test_free_metrics_are_zero_for_an_exact_match():
    volume = torch.ones((1, 1, 4, 4, 4))
    mask = torch.zeros_like(volume, dtype=torch.bool)
    mask[..., ::2, ::2, ::2] = True

    metrics = free_metric_dict(volume, volume, mask)

    assert metrics == {
        "r_factor_free": 0.0,
        "llk_free": 0.0,
        "chi2_free": 0.0,
    }


def test_free_metrics_follow_their_documented_formulas():
    measured_modulus = torch.ones((1, 1, 2, 2, 2))
    predicted_modulus = measured_modulus * 2.0
    mask = torch.ones_like(measured_modulus, dtype=torch.bool)
    expected_llk = 2.0 * (4.0 - 1.0 + math.log(1.0 / 4.0))

    assert torch.allclose(
        r_factor_free(measured_modulus, predicted_modulus, mask),
        torch.tensor(1.0),
    )
    assert torch.allclose(
        chi2_free(measured_modulus, predicted_modulus, mask),
        torch.tensor(1.0),
    )
    assert torch.allclose(
        llk_free(measured_modulus, predicted_modulus, mask),
        torch.tensor(expected_llk),
        atol=1e-6,
    )


def test_generated_free_mask_is_reproducible_and_nonempty():
    first = generate_free_mask((8, 8, 8), fraction=0.05, seed=42)
    second = generate_free_mask((8, 8, 8), fraction=0.05, seed=42)

    assert first is not None
    assert torch.equal(first, second)
    assert bool(first.any())
    assert not bool(first.all())
