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
        torch.sum(torch.pow(true, 2), dim=1) * torch.sum(torch.pow(pred, 2), dim=1) + EPS
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


def wrapped_phase_abs_error(y_true, y_pred):
    return torch.atan2(torch.sin(y_pred - y_true), torch.cos(y_pred - y_true)).abs()


def masked_reduce(values, mask, reduction="mean"):
    values = flatten_sample(values)
    mask = flatten_sample(mask.float())
    numer = torch.sum(values * mask, dim=1)
    denom = torch.sum(mask, dim=1).clamp_min(1.0)
    per_sample = numer / denom
    return reduce_per_sample(per_sample, reduction)


@torch.no_grad()
def realspace_metric_dict(true_amp, true_phi, pred_amp, pred_phi, pred_support=None, threshold=0.1):
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

    return {
        "real_amp_l1": float(F.l1_loss(pred_amp, true_amp).detach().cpu()),
        "real_amp_mse": float(F.mse_loss(pred_amp, true_amp).detach().cpu()),
        "real_amp_rmse": float(torch.sqrt(F.mse_loss(pred_amp, true_amp) + EPS).detach().cpu()),
        "real_amp_global_ssim": float(global_ssim(true_amp, pred_amp).detach().cpu()),
        "real_support_iou": float(torch.mean(inter / (union_count + EPS)).detach().cpu()),
        "real_support_dice": float(torch.mean((2 * inter) / (true_count + pred_count + EPS)).detach().cpu()),
        "real_support_true_fraction": float(torch.mean(true_count / true_flat.shape[1]).detach().cpu()),
        "real_support_pred_fraction": float(torch.mean(pred_count / pred_flat.shape[1]).detach().cpu()),
        "real_support_volume_ratio": float(torch.mean(pred_count / (true_count + EPS)).detach().cpu()),
        "real_phase_mae_true_support": float(masked_reduce(phase_err, true_support).detach().cpu()),
        "real_phase_mae_intersection": float(masked_reduce(phase_err, intersection).detach().cpu()),
        "real_phase_rmse_true_support": float(torch.sqrt(masked_reduce(phase_sq, true_support) + EPS).detach().cpu()),
    }


def chi2_pcc_loss(y_true, y_pred):
    return 0.5 * (chi2_modulus(y_true, y_pred) + pearson_loss(y_true, y_pred))


def sqrt_chi2_pcc_loss(y_true, y_pred):
    return 0.5 * (torch.sqrt(chi2_modulus(y_true, y_pred) + EPS) + pearson_loss(y_true, y_pred))


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
def metric_dict(y_true, y_pred):
    return {
        "paper_modulus_mae": float(paper_modulus_mae(y_true, y_pred).detach().cpu()),
        "chi2_modulus": float(chi2_modulus(y_true, y_pred).detach().cpu()),
        "relative_l1_modulus": float(relative_l1_modulus(y_true, y_pred).detach().cpu()),
        "relative_log_mse": float(relative_log_mse(y_true, y_pred).detach().cpu()),
        "pearson_corr": float(pearson_corr(y_true, y_pred).detach().cpu()),
        "pearson_loss": float(pearson_loss(y_true, y_pred).detach().cpu()),
        "voxel_mse": float(voxel_mse(y_true, y_pred).detach().cpu()),
        "voxel_rmse": float(voxel_rmse(y_true, y_pred).detach().cpu()),
    }
