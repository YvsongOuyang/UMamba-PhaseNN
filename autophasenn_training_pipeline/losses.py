import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-8
STANDARD_L1_LOSS = nn.L1Loss(reduction="mean")
STANDARD_MSE_LOSS = nn.MSELoss(reduction="mean")


def spatial_dims(y):
    return tuple(range(1, y.ndim))


def flatten_sample(y):
    return y.reshape(y.shape[0], -1)


def reduce_per_sample(values, reduction="mean"):
    if reduction == "none":
        return values
    if reduction == "sum":
        return torch.sum(values)
    if reduction == "mean":
        return torch.mean(values)
    raise ValueError(f"Unsupported reduction: {reduction}")


def _mean_per_sample(values):
    return torch.mean(flatten_sample(values), dim=1)


def scale_align_sum(y_true, y_pred, eps=1e-10):
    """Scale each predicted sample to have the same total modulus as y_true."""

    dims = spatial_dims(y_true)
    scale = torch.sum(y_true, dim=dims, keepdim=True) / (
        torch.sum(y_pred, dim=dims, keepdim=True) + eps
    )
    return y_pred * scale


def paper_modulus_mae(y_true, y_pred, reduction="mean"):
    """Paper Eq. (1) for tensors that already store diffraction modulus.

    The paper writes the loss as mean absolute error between sqrt(Ie) and
    sqrt(Im). In this codebase the network input/output are already abs(FFT),
    i.e. sqrt(intensity), so this is the per-voxel MAE of diffraction modulus.
    """

    return F.l1_loss(y_pred, y_true, reduction=reduction)


def intensity_sqrt_mae(y_true_intensity, y_pred_intensity, reduction="mean"):
    """Paper Eq. (1) if the tensors store intensity rather than modulus."""

    true_modulus = torch.sqrt(torch.clamp(y_true_intensity, min=0.0))
    pred_modulus = torch.sqrt(torch.clamp(y_pred_intensity, min=0.0))
    return paper_modulus_mae(true_modulus, pred_modulus, reduction=reduction)


def chi2_modulus(y_true, y_pred, reduction="mean"):
    """Paper Eq. (2) reciprocal-space chi2 for diffraction modulus tensors."""

    true = flatten_sample(y_true)
    pred = flatten_sample(y_pred)
    numerator = torch.sum(torch.pow(pred - true, 2), dim=1)
    denominator = torch.sum(torch.pow(true, 2), dim=1)
    return reduce_per_sample(numerator / (denominator + EPS), reduction)


def relative_l1_modulus(y_true, y_pred, reduction="mean"):
    true = flatten_sample(y_true)
    pred = flatten_sample(y_pred)
    numerator = torch.sum(torch.abs(pred - true), dim=1)
    denominator = torch.sum(torch.abs(true), dim=1)
    return reduce_per_sample(numerator / (denominator + EPS), reduction)


def relative_log_mse(y_true, y_pred, reduction="mean"):
    true = flatten_sample(torch.log10(y_true + 1.0))
    pred = flatten_sample(torch.log10(y_pred + 1.0))
    numerator = torch.sum(torch.pow(pred - true, 2), dim=1)
    denominator = torch.sum(torch.pow(true, 2), dim=1)
    return reduce_per_sample(numerator / (denominator + EPS), reduction)


def pearson_corr(y_true, y_pred, reduction="mean"):
    true = flatten_sample(y_true)
    pred = flatten_sample(y_pred)
    true = true - torch.mean(true, dim=1, keepdim=True)
    pred = pred - torch.mean(pred, dim=1, keepdim=True)
    numerator = torch.sum(true * pred, dim=1)
    denominator = torch.sqrt(
        torch.sum(torch.pow(true, 2), dim=1) * torch.sum(torch.pow(pred, 2), dim=1)
        + EPS
    )
    return reduce_per_sample(numerator / denominator, reduction)


def pearson_loss(y_true, y_pred, reduction="mean"):
    corr = pearson_corr(y_true, y_pred, reduction="none")
    return reduce_per_sample(1.0 - corr, reduction)


def voxel_mse(y_true, y_pred, reduction="mean"):
    return F.mse_loss(y_pred, y_true, reduction=reduction)


def voxel_rmse(y_true, y_pred, reduction="mean"):
    per_sample = torch.sqrt(
        torch.mean(torch.pow(flatten_sample(y_pred) - flatten_sample(y_true), 2), dim=1)
        + EPS
    )
    return reduce_per_sample(per_sample, reduction)


def global_ssim(y_true, y_pred, data_range=1.0, reduction="mean"):
    """Single-window SSIM over each full 3D volume.

    The paper reports amplitude SSIM. This global variant is lightweight and
    dependency-free; it is intended as a stable validation signal rather than a
    pixel-perfect replacement for windowed skimage SSIM.
    """

    true = flatten_sample(y_true.float())
    pred = flatten_sample(y_pred.float())
    mu_true = torch.mean(true, dim=1)
    mu_pred = torch.mean(pred, dim=1)
    var_true = torch.mean((true - mu_true[:, None]) ** 2, dim=1)
    var_pred = torch.mean((pred - mu_pred[:, None]) ** 2, dim=1)
    cov = torch.mean((true - mu_true[:, None]) * (pred - mu_pred[:, None]), dim=1)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    score = ((2 * mu_true * mu_pred + c1) * (2 * cov + c2)) / (
        (mu_true**2 + mu_pred**2 + c1) * (var_true + var_pred + c2) + EPS
    )
    return reduce_per_sample(score, reduction)


def windowed_ssim_3d(
    y_true,
    y_pred,
    data_range=1.0,
    window_size=7,
    reduction="mean",
):
    """Compute local-window 3D SSIM for normalized amplitude volumes.

    This follows the standard SSIM formulation with a uniform cubic window
    and sample covariance normalization. Inputs must be shaped ``(B, C, D,
    H, W)``. AutoPhaseNN amplitude is normalized to ``[0, 1]``, so the
    default data range matches the paper's simulated-object evaluation.
    """

    if y_true.shape != y_pred.shape:
        raise ValueError("SSIM inputs must have identical shapes.")
    if y_true.ndim != 5:
        raise ValueError("3D SSIM expects tensors shaped (B, C, D, H, W).")
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("SSIM window_size must be a positive odd integer.")
    if any(size < window_size for size in y_true.shape[-3:]):
        raise ValueError(
            f"SSIM window_size={window_size} exceeds spatial shape {tuple(y_true.shape[-3:])}."
        )

    true = y_true.float()
    pred = y_pred.float()
    kernel = (window_size, window_size, window_size)
    mu_true = F.avg_pool3d(true, kernel_size=kernel, stride=1)
    mu_pred = F.avg_pool3d(pred, kernel_size=kernel, stride=1)
    covariance_norm = (window_size**3) / max(window_size**3 - 1, 1)
    var_true = covariance_norm * (
        F.avg_pool3d(true * true, kernel_size=kernel, stride=1) - mu_true * mu_true
    )
    var_pred = covariance_norm * (
        F.avg_pool3d(pred * pred, kernel_size=kernel, stride=1) - mu_pred * mu_pred
    )
    covariance = covariance_norm * (
        F.avg_pool3d(true * pred, kernel_size=kernel, stride=1) - mu_true * mu_pred
    )
    var_true = var_true.clamp_min(0.0)
    var_pred = var_pred.clamp_min(0.0)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_true * mu_pred + c1) * (2 * covariance + c2)
    denominator = (mu_true**2 + mu_pred**2 + c1) * (var_true + var_pred + c2)
    score = numerator / denominator.clamp_min(EPS)
    per_sample = torch.mean(flatten_sample(score), dim=1)
    return reduce_per_sample(per_sample, reduction)


def _broadcast_metric_mask(mask, reference):
    mask = mask.to(device=reference.device, dtype=reference.dtype)
    try:
        return torch.broadcast_to(mask, reference.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"Metric mask shape {tuple(mask.shape)} cannot broadcast to {tuple(reference.shape)}."
        ) from exc


def _masked_sum_per_sample(values, mask):
    broadcast_mask = _broadcast_metric_mask(mask, values)
    return torch.sum(flatten_sample(values * broadcast_mask), dim=1)


def r_factor_free(y_true, y_pred, free_mask, reduction="mean"):
    """Amplitude R-factor evaluated only on held-out reciprocal voxels."""

    numerator = _masked_sum_per_sample(torch.abs(y_pred - y_true), free_mask)
    denominator = _masked_sum_per_sample(torch.abs(y_true), free_mask)
    return reduce_per_sample(numerator / (denominator + EPS), reduction)


def chi2_free(y_true, y_pred, free_mask, reduction="mean"):
    """Paper Eq. (2) restricted to held-out reciprocal voxels."""

    numerator = _masked_sum_per_sample(torch.pow(y_pred - y_true, 2), free_mask)
    denominator = _masked_sum_per_sample(torch.pow(y_true, 2), free_mask)
    return reduce_per_sample(numerator / (denominator + EPS), reduction)


def llk_free(y_true, y_pred, free_mask, reduction="mean"):
    """Mean Poisson deviance on held-out reciprocal-space intensities.

    The tensors in this project contain diffraction modulus, so they are
    squared before evaluating the Poisson statistic. The value is normalized
    per free voxel for comparability across masks; lower is better and zero is
    an exact match.
    """

    true_intensity = torch.pow(y_true.clamp_min(0.0), 2)
    pred_intensity = torch.pow(y_pred.clamp_min(0.0), 2).clamp_min(EPS)
    log_ratio = torch.where(
        true_intensity > 0,
        true_intensity * torch.log((true_intensity + EPS) / pred_intensity),
        torch.zeros_like(true_intensity),
    )
    poisson_deviance = 2.0 * (pred_intensity - true_intensity + log_ratio)
    numerator = _masked_sum_per_sample(poisson_deviance, free_mask)
    mask_count = _masked_sum_per_sample(torch.ones_like(y_true), free_mask).clamp_min(
        1.0
    )
    return reduce_per_sample(numerator / mask_count, reduction)


@torch.no_grad()
def free_metric_tensor_dict(y_true, y_pred, free_mask):
    """Return free metrics as one scalar tensor per sample."""

    return {
        "r_factor_free": r_factor_free(y_true, y_pred, free_mask, reduction="none"),
        "llk_free": llk_free(y_true, y_pred, free_mask, reduction="none"),
        "chi2_free": chi2_free(y_true, y_pred, free_mask, reduction="none"),
    }


@torch.no_grad()
def free_metric_dict(y_true, y_pred, free_mask):
    """Paper-referenced free R-factor diagnostics on a supplied holdout mask."""

    return {
        "r_factor_free": float(r_factor_free(y_true, y_pred, free_mask).detach().cpu()),
        "llk_free": float(llk_free(y_true, y_pred, free_mask).detach().cpu()),
        "chi2_free": float(chi2_free(y_true, y_pred, free_mask).detach().cpu()),
    }


def wrapped_phase_abs_error(y_true, y_pred):
    return torch.atan2(torch.sin(y_pred - y_true), torch.cos(y_pred - y_true)).abs()


def masked_reduce(values, mask, reduction="mean"):
    values = flatten_sample(values)
    mask = flatten_sample(mask.float())
    numer = torch.sum(values * mask, dim=1)
    denom = torch.sum(mask, dim=1).clamp_min(1.0)
    per_sample = numer / denom
    return reduce_per_sample(per_sample, reduction)


def relative_abs_error(y_true, abs_error, mask=None, reduction="mean"):
    if mask is not None:
        mask = mask.float()
        y_true = y_true * mask
        abs_error = abs_error * mask
    true = flatten_sample(y_true)
    err = flatten_sample(abs_error)
    numerator = torch.sum(err, dim=1)
    denominator = torch.sum(torch.abs(true), dim=1)
    return reduce_per_sample(numerator / (denominator + EPS), reduction)


@torch.no_grad()
def realspace_metric_tensor_dict(
    true_amp,
    true_phi,
    pred_amp,
    pred_phi,
    pred_support=None,
    threshold=0.1,
    ssim_window_size=7,
):
    """Return real-space metrics as one scalar tensor per sample."""

    true_support = (true_amp >= threshold).float()
    if pred_support is None:
        pred_support = (pred_amp >= threshold).float()
    else:
        pred_support = (pred_support >= 0.5).float()

    intersection = true_support * pred_support
    union = torch.clamp(true_support + pred_support, max=1.0)
    true_flat = flatten_sample(true_support)
    pred_flat = flatten_sample(pred_support)
    inter = torch.sum(flatten_sample(intersection), dim=1)
    true_count = torch.sum(true_flat, dim=1)
    pred_count = torch.sum(pred_flat, dim=1)
    union_count = torch.sum(flatten_sample(union), dim=1)

    amp_abs_error = torch.abs(pred_amp - true_amp)
    amp_sq_error = torch.pow(pred_amp - true_amp, 2)
    amp_mse = _mean_per_sample(amp_sq_error)
    phase_err = wrapped_phase_abs_error(true_phi, pred_phi)
    phase_sq = phase_err**2
    phase_mse_true_support = masked_reduce(phase_sq, true_support, reduction="none")
    support_abs_error = torch.abs(pred_support - true_support)
    support_mse = _mean_per_sample(torch.pow(support_abs_error, 2))

    return {
        "real_amp_l1": _mean_per_sample(amp_abs_error),
        "real_amp_mse": amp_mse,
        "real_amp_rmse": torch.sqrt(amp_mse + EPS),
        "real_amp_rel_l1": relative_abs_error(
            true_amp,
            amp_abs_error,
            reduction="none",
        ),
        "real_amp_ssim": windowed_ssim_3d(
            true_amp,
            pred_amp,
            window_size=ssim_window_size,
            reduction="none",
        ),
        "real_amp_global_ssim": global_ssim(
            true_amp,
            pred_amp,
            reduction="none",
        ),
        "real_support_l1": _mean_per_sample(support_abs_error),
        "real_support_mse": support_mse,
        "real_support_rmse": torch.sqrt(support_mse + EPS),
        "real_support_rel_l1": relative_abs_error(
            true_support,
            support_abs_error,
            reduction="none",
        ),
        "real_support_iou": inter / (union_count + EPS),
        "real_support_dice": (2 * inter) / (true_count + pred_count + EPS),
        "real_support_true_fraction": true_count / true_flat.shape[1],
        "real_support_pred_fraction": pred_count / pred_flat.shape[1],
        "real_support_volume_ratio": pred_count / (true_count + EPS),
        "real_phase_l1_true_support": masked_reduce(
            phase_err,
            true_support,
            reduction="none",
        ),
        "real_phase_mse_true_support": phase_mse_true_support,
        "real_phase_rel_l1_true_support": relative_abs_error(
            true_phi,
            phase_err,
            true_support,
            reduction="none",
        ),
        "real_phase_mae_true_support": masked_reduce(
            phase_err,
            true_support,
            reduction="none",
        ),
        "real_phase_mae_intersection": masked_reduce(
            phase_err,
            intersection,
            reduction="none",
        ),
        "real_phase_rmse_true_support": torch.sqrt(phase_mse_true_support + EPS),
    }


@torch.no_grad()
def realspace_metric_dict(
    true_amp,
    true_phi,
    pred_amp,
    pred_phi,
    pred_support=None,
    threshold=0.1,
    ssim_window_size=7,
):
    """Metrics for the reconstructed real-space object."""

    true_support = (true_amp >= threshold).float()
    if pred_support is None:
        pred_support = (pred_amp >= threshold).float()
    else:
        pred_support = (pred_support >= 0.5).float()

    intersection = true_support * pred_support
    union = torch.clamp(true_support + pred_support, max=1.0)
    true_flat = flatten_sample(true_support)
    pred_flat = flatten_sample(pred_support)
    inter_flat = flatten_sample(intersection)
    union_flat = flatten_sample(union)
    inter = torch.sum(inter_flat, dim=1)
    true_count = torch.sum(true_flat, dim=1)
    pred_count = torch.sum(pred_flat, dim=1)
    union_count = torch.sum(union_flat, dim=1)

    phase_err = wrapped_phase_abs_error(true_phi, pred_phi)
    phase_sq = phase_err**2
    phase_mse_true_support = masked_reduce(phase_sq, true_support)
    support_abs_error = torch.abs(pred_support - true_support)
    support_mse = F.mse_loss(pred_support, true_support)

    return {
        "real_amp_l1": float(F.l1_loss(pred_amp, true_amp).detach().cpu()),
        "real_amp_mse": float(F.mse_loss(pred_amp, true_amp).detach().cpu()),
        "real_amp_rmse": float(
            torch.sqrt(F.mse_loss(pred_amp, true_amp) + EPS).detach().cpu()
        ),
        "real_amp_rel_l1": float(
            relative_abs_error(true_amp, torch.abs(pred_amp - true_amp)).detach().cpu()
        ),
        "real_amp_ssim": float(
            windowed_ssim_3d(
                true_amp,
                pred_amp,
                window_size=ssim_window_size,
            )
            .detach()
            .cpu()
        ),
        "real_amp_global_ssim": float(global_ssim(true_amp, pred_amp).detach().cpu()),
        "real_support_l1": float(F.l1_loss(pred_support, true_support).detach().cpu()),
        "real_support_mse": float(support_mse.detach().cpu()),
        "real_support_rmse": float(torch.sqrt(support_mse + EPS).detach().cpu()),
        "real_support_rel_l1": float(
            relative_abs_error(true_support, support_abs_error).detach().cpu()
        ),
        "real_support_iou": float(
            torch.mean(inter / (union_count + EPS)).detach().cpu()
        ),
        "real_support_dice": float(
            torch.mean((2 * inter) / (true_count + pred_count + EPS)).detach().cpu()
        ),
        "real_support_true_fraction": float(
            torch.mean(true_count / true_flat.shape[1]).detach().cpu()
        ),
        "real_support_pred_fraction": float(
            torch.mean(pred_count / pred_flat.shape[1]).detach().cpu()
        ),
        "real_support_volume_ratio": float(
            torch.mean(pred_count / (true_count + EPS)).detach().cpu()
        ),
        "real_phase_l1_true_support": float(
            masked_reduce(phase_err, true_support).detach().cpu()
        ),
        "real_phase_mse_true_support": float(phase_mse_true_support.detach().cpu()),
        "real_phase_rel_l1_true_support": float(
            relative_abs_error(true_phi, phase_err, true_support).detach().cpu()
        ),
        "real_phase_mae_true_support": float(
            masked_reduce(phase_err, true_support).detach().cpu()
        ),
        "real_phase_mae_intersection": float(
            masked_reduce(phase_err, intersection).detach().cpu()
        ),
        "real_phase_rmse_true_support": float(
            torch.sqrt(phase_mse_true_support + EPS).detach().cpu()
        ),
    }


def chi2_pcc_loss(y_true, y_pred):
    return 0.5 * (chi2_modulus(y_true, y_pred) + pearson_loss(y_true, y_pred))


def sqrt_chi2_pcc_loss(y_true, y_pred):
    return 0.5 * (
        torch.sqrt(chi2_modulus(y_true, y_pred) + EPS) + pearson_loss(y_true, y_pred)
    )


def chi2_pcc_log_loss(y_true, y_pred):
    return (
        50.0 * chi2_modulus(y_true, y_pred)
        + 50.0 * pearson_loss(y_true, y_pred)
        + relative_log_mse(y_true, y_pred)
    ) / 101.0


# Backward-compatible aliases. They now use standard batch-mean reduction.
loss_paper = paper_modulus_mae
loss_sq = chi2_modulus
loss_mae = relative_l1_modulus
loss_log = relative_log_mse
loss_pcc = pearson_loss
loss_comb = chi2_pcc_loss
loss_comb2 = sqrt_chi2_pcc_loss
loss_comb_log = chi2_pcc_log_loss


METRIC_GROUPS = {
    "reciprocal_primary": [
        "paper_modulus_mae",
        "relative_l1_modulus",
        "chi2_modulus",
        "pearson_corr",
    ],
    "realspace_primary": [
        "real_amp_l1",
        "real_amp_ssim",
        "real_amp_global_ssim",
        "real_support_iou",
        "real_support_dice",
        "real_support_pred_fraction",
        "real_support_volume_ratio",
        "real_phase_mae_true_support",
    ],
    "reciprocal_diagnostic": [
        "relative_log_mse",
        "pearson_loss",
        "voxel_mse",
        "voxel_rmse",
    ],
    "realspace_diagnostic": [
        "real_amp_mse",
        "real_amp_rmse",
        "real_support_true_fraction",
        "real_phase_mae_intersection",
        "real_phase_rmse_true_support",
    ],
    "paper_free": [
        "r_factor_free",
        "llk_free",
        "chi2_free",
    ],
}


METRIC_DESCRIPTIONS = {
    "paper_modulus_mae": "Primary paper-style far-field modulus L1. Lower is better.",
    "relative_l1_modulus": "Scale-normalized far-field L1. Lower is better.",
    "chi2_modulus": "Paper chi-square style far-field error. Lower is better.",
    "pearson_corr": "Far-field Pearson correlation. Higher is better.",
    "relative_log_mse": "Log-domain far-field diagnostic. Lower is better.",
    "pearson_loss": "1 - pearson_corr. Lower is better.",
    "voxel_mse": "Raw far-field MSE on the current data scale. Lower is better.",
    "voxel_rmse": "Raw far-field RMSE on the current data scale. Lower is better.",
    "real_amp_l1": "Real-space full-volume amplitude L1. Lower is better, but can be small for sparse objects.",
    "real_amp_mse": "Real-space full-volume amplitude MSE. Lower is better.",
    "real_amp_rmse": "Real-space full-volume amplitude RMSE. Lower is better.",
    "real_amp_rel_l1": "Real-space amplitude L1 normalized by true amplitude sum. Lower is better.",
    "real_amp_ssim": "Local-window 3D amplitude SSIM reported by the paper. Higher is better.",
    "real_amp_global_ssim": "Global 3D amplitude SSIM-like score. Higher is better.",
    "real_support_l1": "Binary support mask L1. Lower is better.",
    "real_support_mse": "Binary support mask MSE. Lower is better.",
    "real_support_rmse": "Binary support mask RMSE. Lower is better.",
    "real_support_rel_l1": "Binary support mask L1 normalized by true support volume. Lower is better.",
    "real_support_iou": "Intersection-over-union between predicted and true support. Higher is better.",
    "real_support_dice": "Dice score between predicted and true support. Higher is better.",
    "real_support_true_fraction": "True support fraction in the 64^3 volume.",
    "real_support_pred_fraction": "Predicted support fraction in the 64^3 volume; should be close to true fraction.",
    "real_support_volume_ratio": "pred_support_fraction / true_support_fraction; ideal is near 1.",
    "real_phase_l1_true_support": "Wrapped phase L1 on the true support. Lower is better.",
    "real_phase_mse_true_support": "Wrapped phase MSE on the true support. Lower is better.",
    "real_phase_rel_l1_true_support": "Wrapped phase L1 normalized by target phase magnitude on the true support. Lower is better.",
    "real_phase_mae_true_support": "Wrapped phase MAE on the true support. Lower is better.",
    "real_phase_mae_intersection": "Wrapped phase MAE on support intersection. Lower is better.",
    "real_phase_rmse_true_support": "Wrapped phase RMSE on the true support. Lower is better.",
    "r_factor_free": "Amplitude R-factor on held-out reciprocal voxels. Lower is better.",
    "llk_free": "Mean Poisson deviance on held-out reciprocal voxels. Lower is better.",
    "chi2_free": "Paper Eq. (2) chi-square restricted to held-out reciprocal voxels. Lower is better.",
}


FIXED_EVALUATION_GROUPS = {
    "FT": [
        ("L1", "paper_modulus_mae"),
        ("MSE", "voxel_mse"),
        ("RMSE", "voxel_rmse"),
        ("RelL1", "relative_l1_modulus"),
    ],
    "Amplitude": [
        ("L1", "real_amp_l1"),
        ("MSE", "real_amp_mse"),
        ("RMSE", "real_amp_rmse"),
        ("RelL1", "real_amp_rel_l1"),
    ],
    "Phase": [
        ("L1", "real_phase_l1_true_support"),
        ("MSE", "real_phase_mse_true_support"),
        ("RMSE", "real_phase_rmse_true_support"),
        ("RelL1", "real_phase_rel_l1_true_support"),
    ],
    "Support": [
        ("L1", "real_support_l1"),
        ("MSE", "real_support_mse"),
        ("RMSE", "real_support_rmse"),
        ("RelL1", "real_support_rel_l1"),
    ],
}


FIXED_METRIC_DESCRIPTIONS = {
    "FT/L1": "Far-field diffraction modulus L1. This is the paper-style primary fitting loss.",
    "FT/MSE": "Far-field diffraction modulus MSE.",
    "FT/RMSE": "Far-field diffraction modulus RMSE.",
    "FT/RelL1": "Far-field L1 normalized by target modulus sum.",
    "Amplitude/L1": "Full-volume real-space amplitude L1.",
    "Amplitude/MSE": "Full-volume real-space amplitude MSE.",
    "Amplitude/RMSE": "Full-volume real-space amplitude RMSE.",
    "Amplitude/RelL1": "Full-volume amplitude L1 normalized by target amplitude sum.",
    "Phase/L1": "Wrapped phase L1 on true support.",
    "Phase/MSE": "Wrapped phase MSE on true support.",
    "Phase/RMSE": "Wrapped phase RMSE on true support.",
    "Phase/RelL1": "Wrapped phase L1 normalized by target phase magnitude on true support.",
    "Support/L1": "Binary support mask L1.",
    "Support/MSE": "Binary support mask MSE.",
    "Support/RMSE": "Binary support mask RMSE.",
    "Support/RelL1": "Binary support mask L1 normalized by true support volume.",
}


def fixed_metric_groups(metrics):
    grouped = {}
    for group_name, entries in FIXED_EVALUATION_GROUPS.items():
        group = {}
        for display_name, metric_key in entries:
            if metric_key in metrics:
                group[display_name] = metrics[metric_key]
        if group:
            grouped[group_name] = group
    return grouped


def format_fixed_metric_groups(metrics, title=None):
    lines = []
    if title:
        lines.append(title)
    for group_name, group in fixed_metric_groups(metrics).items():
        lines.append(f"{group_name}:")
        for key, value in group.items():
            if isinstance(value, (int, float)):
                lines.append(f"  {key}: {value:.6g}")
            else:
                lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def group_metrics(metrics):
    grouped = {}
    used = set()
    for group_name, keys in METRIC_GROUPS.items():
        group = {key: metrics[key] for key in keys if key in metrics}
        if group:
            grouped[group_name] = group
            used.update(group)
    extra = {key: value for key, value in metrics.items() if key not in used}
    if extra:
        grouped["other"] = extra
    return grouped


def format_metric_groups(metrics, title=None):
    labels = {
        "reciprocal_primary": "Reciprocal-space primary metrics",
        "realspace_primary": "Real-space primary metrics",
        "reciprocal_diagnostic": "Reciprocal-space diagnostics",
        "realspace_diagnostic": "Real-space diagnostics",
        "paper_free": "Paper free R-factor diagnostics",
        "other": "Other metrics",
    }
    lines = []
    if title:
        lines.append(title)
    for group_name, group in group_metrics(metrics).items():
        lines.append(labels.get(group_name, group_name))
        for key, value in group.items():
            if isinstance(value, (int, float)):
                lines.append(f"  {key}: {value:.6g}")
            else:
                lines.append(f"  {key}: {value}")
    return "\n".join(lines)


LOSS_REGISTRY = {
    "paper_mae": STANDARD_L1_LOSS,
    "modulus_mae": STANDARD_L1_LOSS,
    "l1": STANDARD_L1_LOSS,
    "paper": STANDARD_L1_LOSS,
    "intensity_sqrt_mae": intensity_sqrt_mae,
    "chi2": chi2_modulus,
    "chi2_modulus": chi2_modulus,
    "relative_l1": relative_l1_modulus,
    "relative_log_mse": relative_log_mse,
    "log": relative_log_mse,
    "pcc": pearson_loss,
    "pearson": pearson_loss,
    "mse": STANDARD_MSE_LOSS,
    "rmse": voxel_rmse,
    # Legacy names kept for old command lines, but not recommended as defaults.
    "sq": chi2_modulus,
    "mae": relative_l1_modulus,
    "comb": chi2_pcc_loss,
    "comb2": sqrt_chi2_pcc_loss,
    "comb_log": chi2_pcc_log_loss,
}


def get_loss(name):
    key = name.lower()
    if key not in LOSS_REGISTRY:
        allowed = ", ".join(sorted(LOSS_REGISTRY))
        raise ValueError(f"Unknown loss {name!r}. Allowed: {allowed}")
    return LOSS_REGISTRY[key]


@torch.no_grad()
def metric_tensor_dict(y_true, y_pred):
    """Return reciprocal-space metrics as one scalar tensor per sample."""

    abs_error = torch.abs(y_pred - y_true)
    sq_error = torch.pow(y_pred - y_true, 2)
    return {
        "paper_modulus_mae": _mean_per_sample(abs_error),
        "chi2_modulus": chi2_modulus(y_true, y_pred, reduction="none"),
        "relative_l1_modulus": relative_l1_modulus(
            y_true,
            y_pred,
            reduction="none",
        ),
        "relative_log_mse": relative_log_mse(
            y_true,
            y_pred,
            reduction="none",
        ),
        "pearson_corr": pearson_corr(y_true, y_pred, reduction="none"),
        "pearson_loss": pearson_loss(y_true, y_pred, reduction="none"),
        "voxel_mse": _mean_per_sample(sq_error),
        "voxel_rmse": voxel_rmse(y_true, y_pred, reduction="none"),
    }


@torch.no_grad()
def metric_dict(y_true, y_pred):
    return {
        "paper_modulus_mae": float(paper_modulus_mae(y_true, y_pred).detach().cpu()),
        "chi2_modulus": float(chi2_modulus(y_true, y_pred).detach().cpu()),
        "relative_l1_modulus": float(
            relative_l1_modulus(y_true, y_pred).detach().cpu()
        ),
        "relative_log_mse": float(relative_log_mse(y_true, y_pred).detach().cpu()),
        "pearson_corr": float(pearson_corr(y_true, y_pred).detach().cpu()),
        "pearson_loss": float(pearson_loss(y_true, y_pred).detach().cpu()),
        "voxel_mse": float(voxel_mse(y_true, y_pred).detach().cpu()),
        "voxel_rmse": float(voxel_rmse(y_true, y_pred).detach().cpu()),
    }
