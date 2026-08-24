import math

import numpy as np
import pytest
import torch

from autophasenn_training_pipeline.evaluate import (
    generate_free_mask,
    materialize_metric_rows,
    post_process_realspace_batch,
    post_process_realspace_sample,
    post_process_unwrapped_realspace,
    raw_amp_from_outputs,
    resolve_threshold_sweep,
    scipy_wrap_shift_batch,
    summarize_threshold_sweep,
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
from autophasenn_training_pipeline.model_factory import (
    default_support_threshold,
    resolve_support_threshold,
)
from autophasenn_training_pipeline.model_mamba_skip import (
    MAMBA_SKIP_CONV_NAMES,
    MAMBA_SKIP_PREFIXES,
    MAMBA_WIDTH,
    initialize_from_baseline_state_dict,
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


def test_threshold_sweep_includes_primary_and_removes_duplicates():
    thresholds = resolve_threshold_sweep(
        0.1,
        [0.2, 0.05, 0.1, 0.05, 0.4],
    )

    assert thresholds == (0.05, 0.1, 0.2, 0.4)
    assert resolve_threshold_sweep(0.1, []) == ()


def test_model_variant_support_threshold_defaults_are_isolated():
    assert default_support_threshold("mamba_skip") == pytest.approx(0.3)
    assert default_support_threshold("baseline") == pytest.approx(0.1)
    assert resolve_support_threshold("mamba_skip", 0.15) == pytest.approx(0.15)


def test_mamba_skip_baseline_initialization_zeroes_new_input_channels():
    class TinyMambaSkip(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleDict(
                {
                    name: torch.nn.Conv3d(2 + MAMBA_WIDTH, 2, kernel_size=1)
                    for name in MAMBA_SKIP_CONV_NAMES
                }
            )
            for prefix in MAMBA_SKIP_PREFIXES:
                setattr(self, prefix.removesuffix("."), torch.nn.Conv3d(1, 1, 1))

    model = TinyMambaSkip()
    initial_state = {key: value.clone() for key, value in model.state_dict().items()}
    baseline_state = {}
    for name in MAMBA_SKIP_CONV_NAMES:
        baseline_state[f"layers.{name}.weight"] = torch.full(
            (2, 2, 1, 1, 1),
            0.25,
        )
        baseline_state[f"layers.{name}.bias"] = torch.full((2,), 0.25)

    initialize_from_baseline_state_dict(model, baseline_state)
    initialized_state = model.state_dict()

    for name in MAMBA_SKIP_CONV_NAMES:
        weight_key = f"layers.{name}.weight"
        bias_key = f"layers.{name}.bias"
        expanded_weight = initialized_state[weight_key]
        assert torch.equal(expanded_weight[:, :2], baseline_state[weight_key])
        assert torch.count_nonzero(expanded_weight[:, 2:]).item() == 0
        assert torch.equal(initialized_state[bias_key], baseline_state[bias_key])
    for key, value in initialized_state.items():
        if key.startswith(MAMBA_SKIP_PREFIXES):
            assert torch.equal(value, initial_state[key])


def test_threshold_sweep_can_recover_pre_support_amplitude():
    masked_amp = torch.zeros((1, 1, 2, 2, 2))
    raw_amp = torch.full_like(masked_amp, 0.075)
    outputs = (None, None, masked_amp, None, None, raw_amp)

    assert raw_amp_from_outputs(outputs, masked_amp) is raw_amp
    assert raw_amp_from_outputs(outputs[:5], masked_amp) is masked_amp


def test_primary_threshold_sweep_matches_standard_postprocess_path():
    coordinates = torch.meshgrid(
        torch.arange(7),
        torch.arange(7),
        torch.arange(7),
        indexing="ij",
    )
    z, y, x = (coordinate.float() for coordinate in coordinates)
    true_amp = torch.exp(-((z - 3) ** 2 + (y - 2) ** 2 + (x - 4) ** 2) / 4)[
        None, None
    ]
    raw_amp = torch.exp(-((z - 2) ** 2 + (y - 4) ** 2 + (x - 3) ** 2) / 5)[
        None, None
    ]
    true_phi = (0.1 * z + 0.2 * y - 0.1 * x)[None, None]
    pred_phi = (0.2 * z - 0.1 * y + 0.1 * x)[None, None]
    threshold = 0.1
    support = (raw_amp >= threshold).float()
    masked_amp = raw_amp * support

    standard = post_process_unwrapped_realspace(
        true_amp,
        true_phi,
        masked_amp,
        pred_phi,
        threshold,
        pred_support=support,
    )
    swept = post_process_unwrapped_realspace(
        true_amp,
        true_phi,
        raw_amp,
        pred_phi,
        threshold,
    )

    for standard_tensor, swept_tensor in zip(standard, swept):
        assert torch.equal(standard_tensor, swept_tensor)


def test_threshold_sweep_summary_selects_iou_and_volume_operating_points():
    rows = []
    for threshold, iou, volume_ratio in (
        (0.05, 0.7, 1.2),
        (0.1, 0.8, 1.05),
        (0.2, 0.75, 0.99),
    ):
        rows.append(
            {
                "name": f"sample_{threshold}",
                "threshold": threshold,
                "real_amp_l1": threshold,
                "real_amp_ssim": 1.0 - threshold,
                "real_support_iou": iou,
                "real_support_dice": iou,
                "real_support_volume_ratio": volume_ratio,
                "real_phase_mae_true_support": threshold * 2,
            }
        )

    summary = summarize_threshold_sweep(rows)

    assert summary["best_mean_iou_threshold"] == pytest.approx(0.1)
    assert summary["closest_mean_volume_ratio_threshold"] == pytest.approx(0.2)
    assert [item["threshold"] for item in summary["summaries"]] == [
        0.05,
        0.1,
        0.2,
    ]


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
