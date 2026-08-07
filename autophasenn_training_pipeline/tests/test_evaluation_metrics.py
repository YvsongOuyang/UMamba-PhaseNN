import math

import numpy as np
import pytest
import torch

from autophasenn_training_pipeline.evaluate import (
    generate_free_mask,
    materialize_metric_rows,
    post_process_realspace_batch,
    post_process_realspace_sample,
    scipy_wrap_shift_batch,
)
from autophasenn_training_pipeline.losses import (
    chi2_free,
    free_metric_dict,
    free_metric_tensor_dict,
    llk_free,
    metric_dict,
    metric_tensor_dict,
    r_factor_free,
    realspace_metric_dict,
    realspace_metric_tensor_dict,
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


def test_batched_metric_tensors_match_sample_metric_dicts():
    generator = torch.Generator().manual_seed(19)
    true_diff = torch.rand((2, 1, 7, 7, 7), generator=generator) + 0.1
    pred_diff = torch.rand((2, 1, 7, 7, 7), generator=generator) + 0.1
    true_amp = torch.rand((2, 1, 7, 7, 7), generator=generator)
    pred_amp = torch.rand((2, 1, 7, 7, 7), generator=generator)
    true_phi = torch.rand((2, 1, 7, 7, 7), generator=generator) * math.tau - math.pi
    pred_phi = torch.rand((2, 1, 7, 7, 7), generator=generator) * math.tau - math.pi
    pred_support = (pred_amp >= 0.5).float()
    free_mask = torch.zeros((1, 1, 7, 7, 7), dtype=torch.bool)
    free_mask[..., ::2, ::2, ::2] = True

    tensors = metric_tensor_dict(true_diff, pred_diff)
    tensors.update(free_metric_tensor_dict(true_diff, pred_diff, free_mask))
    tensors.update(
        realspace_metric_tensor_dict(
            true_amp,
            true_phi,
            pred_amp,
            pred_phi,
            pred_support,
            threshold=0.1,
            ssim_window_size=7,
        )
    )
    rows = materialize_metric_rows(tensors)

    for index, row in enumerate(rows):
        sample = slice(index, index + 1)
        expected = metric_dict(true_diff[sample], pred_diff[sample])
        expected.update(
            free_metric_dict(true_diff[sample], pred_diff[sample], free_mask)
        )
        expected.update(
            realspace_metric_dict(
                true_amp[sample],
                true_phi[sample],
                pred_amp[sample],
                pred_phi[sample],
                pred_support[sample],
                threshold=0.1,
                ssim_window_size=7,
            )
        )
        assert row == pytest.approx(expected, rel=1e-5, abs=1e-6)


def test_torch_shift_matches_scipy_integer_wrap_semantics():
    scipy_ndimage = pytest.importorskip("scipy.ndimage")
    generator = torch.Generator().manual_seed(23)
    values = torch.rand((2, 1, 8, 9, 10), generator=generator)
    shifts = torch.tensor([[1, 2, -3], [-4, 0, 5]])

    actual = scipy_wrap_shift_batch(values, shifts).numpy()

    for index in range(values.shape[0]):
        expected = scipy_ndimage.shift(
            values[index, 0].numpy(),
            shift=tuple(int(value) for value in shifts[index]),
            mode="wrap",
        )
        assert np.array_equal(actual[index, 0], expected)


def test_batched_postprocess_matches_official_sample_implementation():
    pytest.importorskip("scipy.ndimage")
    pytest.importorskip("skimage.restoration")
    coordinates = torch.meshgrid(
        torch.arange(8),
        torch.arange(8),
        torch.arange(8),
        indexing="ij",
    )
    z, y, x = (coordinate.float() for coordinate in coordinates)
    amp_a = torch.exp(-((z - 2) ** 2 + (y - 3) ** 2 + (x - 4) ** 2) / 5)
    amp_b = torch.exp(-((z - 5) ** 2 + (y - 4) ** 2 + (x - 2) ** 2) / 4)
    true_amp = torch.stack((amp_a, amp_b))[:, None]
    pred_amp = torch.stack((amp_b, amp_a))[:, None]
    true_phi = torch.atan2(
        torch.sin(0.9 * z + 0.5 * y - 0.3 * x),
        torch.cos(0.9 * z + 0.5 * y - 0.3 * x),
    ).repeat(2, 1, 1, 1)[:, None]
    pred_phi = torch.atan2(
        torch.sin(0.7 * z - 0.4 * y + 0.6 * x),
        torch.cos(0.7 * z - 0.4 * y + 0.6 * x),
    ).repeat(2, 1, 1, 1)[:, None]
    pred_support = (pred_amp >= 0.1).float()

    actual = post_process_realspace_batch(
        true_amp,
        true_phi,
        pred_amp,
        pred_phi,
        pred_support,
        threshold=0.1,
    )
    expected_samples = [
        post_process_realspace_sample(
            true_amp[index : index + 1],
            true_phi[index : index + 1],
            pred_amp[index : index + 1],
            pred_phi[index : index + 1],
            pred_support[index : index + 1],
            threshold=0.1,
        )
        for index in range(true_amp.shape[0])
    ]
    expected = tuple(
        torch.cat([sample[output_index] for sample in expected_samples], dim=0)
        for output_index in range(len(actual))
    )

    for actual_tensor, expected_tensor in zip(actual, expected):
        assert torch.allclose(actual_tensor, expected_tensor, atol=1e-5, rtol=1e-5)
