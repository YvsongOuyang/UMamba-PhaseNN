"""Reconstruction, consistency, and paired-summary metrics."""

from __future__ import annotations

import math
import zlib
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from autophasenn_training_pipeline.losses import metric_dict, realspace_metric_dict


METRIC_DIRECTIONS = {
    "paper_modulus_mae": "lower",
    "chi2_modulus": "lower",
    "relative_l1_modulus": "lower",
    "relative_log_mse": "lower",
    "pearson_corr": "higher",
    "pearson_loss": "lower",
    "voxel_mse": "lower",
    "voxel_rmse": "lower",
    "real_amp_l1": "lower",
    "real_amp_mse": "lower",
    "real_amp_rmse": "lower",
    "real_amp_rel_l1": "lower",
    "real_amp_global_ssim": "higher",
    "real_amp_psnr": "higher",
    "real_amp_ssim3d": "higher",
    "real_support_l1": "lower",
    "real_support_mse": "lower",
    "real_support_rmse": "lower",
    "real_support_rel_l1": "lower",
    "real_support_iou": "higher",
    "real_support_dice": "higher",
    "real_support_true_fraction": "neutral",
    "real_support_pred_fraction": "neutral",
    "real_support_volume_ratio": "neutral",
    "real_phase_l1_true_support": "lower",
    "real_phase_mse_true_support": "lower",
    "real_phase_rel_l1_true_support": "lower",
    "real_phase_mae_true_support": "lower",
    "real_phase_mae_intersection": "lower",
    "real_phase_rmse_true_support": "lower",
}


def _flatten_sample(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(value.shape[0], -1).float()


@torch.no_grad()
def psnr3d(
    target: torch.Tensor,
    prediction: torch.Tensor,
    data_range: float,
) -> float:
    """Return mean full-volume 3D PSNR in dB."""

    mse = torch.mean((_flatten_sample(prediction) - _flatten_sample(target)) ** 2, dim=1)
    peak = torch.as_tensor(data_range**2, device=mse.device, dtype=mse.dtype)
    score = 10.0 * torch.log10(peak / mse.clamp_min(torch.finfo(mse.dtype).tiny))
    return float(score.mean().cpu())


@torch.no_grad()
def uniform_window_ssim3d(
    target: torch.Tensor,
    prediction: torch.Tensor,
    data_range: float,
    window_size: int,
) -> float:
    """Return uniform-window 3D SSIM averaged over all voxels and samples."""

    target = target.float()
    prediction = prediction.float()
    padding = window_size // 2
    mu_target = F.avg_pool3d(target, window_size, stride=1, padding=padding)
    mu_prediction = F.avg_pool3d(prediction, window_size, stride=1, padding=padding)
    target_sq = F.avg_pool3d(target * target, window_size, stride=1, padding=padding)
    prediction_sq = F.avg_pool3d(
        prediction * prediction, window_size, stride=1, padding=padding
    )
    cross = F.avg_pool3d(target * prediction, window_size, stride=1, padding=padding)

    var_target = torch.clamp(target_sq - mu_target * mu_target, min=0.0)
    var_prediction = torch.clamp(
        prediction_sq - mu_prediction * mu_prediction, min=0.0
    )
    covariance = cross - mu_target * mu_prediction
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2.0 * mu_target * mu_prediction + c1) * (2.0 * covariance + c2)
    denominator = (
        (mu_target.square() + mu_prediction.square() + c1)
        * (var_target + var_prediction + c2)
    )
    return float(torch.mean(numerator / denominator.clamp_min(1e-12)).cpu())


@torch.no_grad()
def reconstruction_metrics(
    true_diff: torch.Tensor,
    true_amp: torch.Tensor,
    true_phase: torch.Tensor,
    outputs: tuple[torch.Tensor, ...],
    has_realspace: bool,
    threshold: float,
    amplitude_data_range: float,
    ssim_window_size: int,
) -> dict[str, float]:
    """Compute one sample's reconstruction metrics against validation truth."""

    pred_diff, _pred_obj, pred_amp, pred_phase, pred_support = outputs[:5]
    result = metric_dict(true_diff, pred_diff)
    if has_realspace:
        result.update(
            realspace_metric_dict(
                true_amp,
                true_phase,
                pred_amp,
                pred_phase,
                pred_support,
                threshold=threshold,
            )
        )
        result["real_amp_psnr"] = psnr3d(
            true_amp, pred_amp, data_range=amplitude_data_range
        )
        result["real_amp_ssim3d"] = uniform_window_ssim3d(
            true_amp,
            pred_amp,
            data_range=amplitude_data_range,
            window_size=ssim_window_size,
        )
    return result


@torch.no_grad()
def tensor_pair_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float]:
    """Measure direct numerical consistency between two real-valued tensors."""

    ref = _flatten_sample(reference)
    cand = _flatten_sample(candidate)
    difference = cand - ref
    abs_difference = torch.abs(difference)
    ref_centered = ref - ref.mean(dim=1, keepdim=True)
    cand_centered = cand - cand.mean(dim=1, keepdim=True)
    corr_denom = torch.sqrt(
        torch.sum(ref_centered.square(), dim=1)
        * torch.sum(cand_centered.square(), dim=1)
    )
    corr = torch.sum(ref_centered * cand_centered, dim=1) / corr_denom.clamp_min(1e-12)
    identical_constant = (corr_denom <= 1e-12) & (
        torch.max(abs_difference, dim=1).values <= 1e-12
    )
    corr = torch.where(identical_constant, torch.ones_like(corr), corr)
    return {
        "mae": float(abs_difference.mean().cpu()),
        "rmse": float(torch.sqrt(torch.mean(difference.square())).cpu()),
        "max_abs": float(abs_difference.max().cpu()),
        "relative_l1": float(
            (torch.sum(abs_difference) / torch.sum(torch.abs(ref)).clamp_min(1e-12)).cpu()
        ),
        "relative_l2": float(
            (
                torch.sqrt(torch.sum(difference.square()))
                / torch.sqrt(torch.sum(ref.square())).clamp_min(1e-12)
            ).cpu()
        ),
        "pearson_corr": float(corr.mean().cpu()),
    }


@torch.no_grad()
def complex_tensor_pair_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float]:
    """Measure direct consistency between two complex-valued object tensors."""

    difference = candidate - reference
    absolute_difference = torch.abs(difference).float()
    reference_magnitude = torch.abs(reference).float()
    return {
        "mae": float(absolute_difference.mean().cpu()),
        "rmse": float(torch.sqrt(torch.mean(absolute_difference.square())).cpu()),
        "max_abs": float(absolute_difference.max().cpu()),
        "relative_l1": float(
            (
                torch.sum(absolute_difference)
                / torch.sum(reference_magnitude).clamp_min(1e-12)
            ).cpu()
        ),
        "relative_l2": float(
            (
                torch.sqrt(torch.sum(absolute_difference.square()))
                / torch.sqrt(torch.sum(reference_magnitude.square())).clamp_min(1e-12)
            ).cpu()
        ),
        "real_mae": float(torch.mean(torch.abs(difference.real)).cpu()),
        "imag_mae": float(torch.mean(torch.abs(difference.imag)).cpu()),
    }


@torch.no_grad()
def _uniform_histogram(
    values: torch.Tensor,
    bins: int,
    low: float,
    high: float,
) -> torch.Tensor:
    """Count uniform bins deterministically without CUDA ``torch.histc``."""

    scaled = (values - low) * (float(bins) / (high - low))
    bin_indices = torch.floor(scaled).to(torch.int64).clamp_(0, bins - 1)
    return torch.bincount(bin_indices, minlength=bins)


@torch.no_grad()
def _histogram_js_divergence(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    bins: int,
) -> float:
    ref = reference.detach().float().reshape(-1)
    cand = candidate.detach().float().reshape(-1)
    low = float(torch.minimum(ref.min(), cand.min()).cpu())
    high = float(torch.maximum(ref.max(), cand.max()).cpu())
    if math.isclose(low, high, rel_tol=0.0, abs_tol=1e-12):
        return 0.0
    ref_hist = _uniform_histogram(ref, bins=bins, low=low, high=high).float()
    cand_hist = _uniform_histogram(cand, bins=bins, low=low, high=high).float()
    p = ref_hist / ref_hist.sum().clamp_min(1.0)
    q = cand_hist / cand_hist.sum().clamp_min(1.0)
    midpoint = 0.5 * (p + q)
    p_term = torch.where(p > 0, p * torch.log(p / midpoint.clamp_min(1e-12)), 0.0)
    q_term = torch.where(q > 0, q * torch.log(q / midpoint.clamp_min(1e-12)), 0.0)
    return float((0.5 * (p_term.sum() + q_term.sum())).cpu())


@torch.no_grad()
def output_consistency_metrics(
    baseline: tuple[torch.Tensor, ...],
    swapped: tuple[torch.Tensor, ...],
    histogram_bins: int,
) -> dict[str, float]:
    """Compare end-to-end outputs without using validation labels."""

    result: dict[str, float] = {}
    for label, index in (("farfield", 0), ("amplitude", 2), ("raw_amplitude", 5)):
        pair = tensor_pair_metrics(baseline[index], swapped[index])
        result.update({f"{label}.{key}": value for key, value in pair.items()})
        result[f"{label}.mean_shift"] = float(
            (swapped[index].float().mean() - baseline[index].float().mean()).cpu()
        )
        result[f"{label}.std_shift"] = float(
            (
                swapped[index].float().std(unbiased=False)
                - baseline[index].float().std(unbiased=False)
            ).cpu()
        )
        result[f"{label}.histogram_js_divergence"] = _histogram_js_divergence(
            baseline[index], swapped[index], bins=histogram_bins
        )

    complex_pair = complex_tensor_pair_metrics(baseline[1], swapped[1])
    result.update(
        {f"complex_object.{key}": value for key, value in complex_pair.items()}
    )

    phase_delta = torch.atan2(
        torch.sin(swapped[3] - baseline[3]),
        torch.cos(swapped[3] - baseline[3]),
    ).abs()
    result["phase.wrapped_mae"] = float(phase_delta.mean().cpu())
    result["phase.wrapped_rmse"] = float(
        torch.sqrt(torch.mean(phase_delta.square())).cpu()
    )
    result["phase.wrapped_max_abs"] = float(phase_delta.max().cpu())
    result["support.disagreement_fraction"] = float(
        torch.mean((baseline[4] != swapped[4]).float()).cpu()
    )
    return result


def summarize_values(
    values: Iterable[float],
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float | int | None]:
    """Summarize values and bootstrap the confidence interval of their mean."""

    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
            "ci_low": None,
            "ci_high": None,
        }
    result: dict[str, float | int | None] = {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "ci_low": None,
        "ci_high": None,
    }
    if bootstrap_samples <= 0 or array.size < 2 or not np.all(np.isfinite(array)):
        return result

    generator = np.random.default_rng(seed)
    means = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(bootstrap_samples):
        sample_indices = generator.integers(0, array.size, size=array.size)
        means[index] = np.mean(array[sample_indices])
    tail = (1.0 - confidence_level) / 2.0
    result["ci_low"] = float(np.quantile(means, tail))
    result["ci_high"] = float(np.quantile(means, 1.0 - tail))
    return result


def stable_metric_seed(base_seed: int, metric_name: str) -> int:
    """Derive a process-independent seed for a named metric."""

    return int((base_seed + zlib.crc32(metric_name.encode("utf-8"))) % (2**32))
