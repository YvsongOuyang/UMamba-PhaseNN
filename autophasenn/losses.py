import torch
import torch.nn.functional as F


def spatial_dims(y):
    return tuple(range(1, y.ndim))


def scale_align_sum(y_true, y_pred, eps=1e-10):
    dims = spatial_dims(y_true)
    scale = torch.sum(y_true, dim=dims, keepdim=True) / (
        torch.sum(y_pred, dim=dims, keepdim=True) + eps
    )
    return y_pred * scale


def loss_log(y_true, y_pred):
    pred = torch.log10(y_pred + 1.0)
    true = torch.log10(y_true + 1.0)
    top = torch.sum(torch.pow(pred - true, 2))
    bottom = torch.sum(torch.pow(true, 2))
    return top / (bottom + 1e-8)


def loss_sq(y_true, y_pred):
    dims = spatial_dims(y_true)
    top = torch.sum(torch.pow(y_pred - y_true, 2), dim=dims, keepdim=True)
    bottom = torch.sum(torch.pow(y_true, 2), dim=dims, keepdim=True)
    return torch.sum(top / (bottom + 1e-8))


def loss_mae(y_true, y_pred):
    dims = spatial_dims(y_true)
    top = torch.sum(torch.abs(y_pred - y_true), dim=dims, keepdim=True)
    bottom = torch.sum(torch.abs(y_true), dim=dims, keepdim=True)
    return torch.sum(top / (bottom + 1e-8))


def loss_paper(y_true, y_pred):
    sqrt_true = torch.sqrt(torch.clamp(y_true, min=0.0))
    sqrt_pred = torch.sqrt(torch.clamp(y_pred, min=0.0))
    n_voxels = y_true.shape[-1] * y_true.shape[-2] * y_true.shape[-3]
    return torch.sum(torch.abs(sqrt_pred - sqrt_true)) / float(n_voxels)


def loss_pcc(y_true, y_pred):
    dims = spatial_dims(y_true)
    pred_centered = y_pred - torch.mean(y_pred, dim=dims, keepdim=True)
    true_centered = y_true - torch.mean(y_true, dim=dims, keepdim=True)
    top = torch.sum(pred_centered * true_centered, dim=dims, keepdim=True)
    pred_var_sum = torch.sum(torch.pow(pred_centered, 2), dim=dims, keepdim=True)
    true_var_sum = torch.sum(torch.pow(true_centered, 2), dim=dims, keepdim=True)
    bottom = torch.sqrt(pred_var_sum * true_var_sum + 1e-8)
    return torch.sum(1.0 - top / bottom)


def loss_comb(y_true, y_pred):
    return (loss_sq(y_true, y_pred) + loss_pcc(y_true, y_pred)) / 2.0


def loss_comb2(y_true, y_pred):
    return (torch.sqrt(loss_sq(y_true, y_pred) + 1e-8) + loss_pcc(y_true, y_pred)) / 2.0


def loss_comb_log(y_true, y_pred):
    return (
        50.0 * loss_sq(y_true, y_pred)
        + 50.0 * loss_pcc(y_true, y_pred)
        + loss_log(y_true, y_pred)
    ) / 101.0


def loss_l1_mean(y_true, y_pred):
    return F.l1_loss(y_pred, y_true)


LOSS_REGISTRY = {
    "l1": loss_l1_mean,
    "log": loss_log,
    "sq": loss_sq,
    "mae": loss_mae,
    "paper": loss_paper,
    "pcc": loss_pcc,
    "comb": loss_comb,
    "comb2": loss_comb2,
    "comb_log": loss_comb_log,
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
        "loss_log": float(loss_log(y_true, y_pred).detach().cpu()),
        "loss_sq": float(loss_sq(y_true, y_pred).detach().cpu()),
        "loss_mae": float(loss_mae(y_true, y_pred).detach().cpu()),
        "loss_paper": float(loss_paper(y_true, y_pred).detach().cpu()),
        "loss_pcc": float(loss_pcc(y_true, y_pred).detach().cpu()),
        "loss_comb": float(loss_comb(y_true, y_pred).detach().cpu()),
        "loss_comb2": float(loss_comb2(y_true, y_pred).detach().cpu()),
        "loss_comb_log": float(loss_comb_log(y_true, y_pred).detach().cpu()),
    }

