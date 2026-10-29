"""Differentiable phase objective using the existing evaluation alignment.

The established WCA and component-MAE terms are retained unchanged.  This
module adds only an optional aligned complex NRMSE term for phase fine-tuning.
"""

from __future__ import annotations

import logging
import math
import time
from itertools import islice

import torch

from pytorch_autophasenn.losses import phase_retrieval_wca_loss
from pytorch_autophasenn.momamba_refiner import ambiguity_aware_component_mae
from pytorch_autophasenn.reconstruction import realspace_from_modulus_phase

from .data_v2 import center_object, prepare_batch


LOGGER = logging.getLogger("high_strain.2d")


def aligned_complex_nrmse_loss(
    prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.3
) -> torch.Tensor:
    """Mean aligned complex NRMSE with gradients through the prediction.

Alignment matches ``evaluate.aligned_objects_2d``: amplitude-COM centering,
component-MAE twin selection, support-based global phase, and individual RMS
normalization.  Only the discrete twin decision is detached.  Adding 1e-12
under the final square root stabilizes perfect predictions (loss floor 1e-6).
This is a training loss; the unmodified evaluator reports the exact metric.
"""

    if not 0 < threshold < 1:
        raise ValueError("Support threshold must be in (0,1).")
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError("Prediction must have shape [B,1,H,W].")
    if prediction.shape != target.shape or not torch.is_complex(prediction) or not torch.is_complex(target):
        raise ValueError("Prediction and target must be equal-shaped complex tensors.")
    pred, truth = center_object(prediction), center_object(target)
    support = truth.abs() / truth.abs().amax((-2, -1), keepdim=True).clamp_min(1e-7) >= threshold
    with torch.no_grad():
        selected_twin = ambiguity_aware_component_mae(
            pred, truth, support, spatial_dims=2
        ).selected_twin
    twin_prediction = torch.conj(
        torch.roll(torch.flip(pred, (-2, -1)), (1, 1), (-2, -1))
    )
    pred = torch.where(selected_twin[:, None, None, None], twin_prediction, pred)
    correlation = (truth.conj() * pred * support).sum((-2, -1), keepdim=True)
    pred = pred * torch.exp(-1j * torch.angle(correlation))
    pred = pred / pred.abs().square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1e-7)
    truth = truth / truth.abs().square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1e-7)
    error = (pred - truth).abs().square().flatten(1).sum(1)
    energy = truth.abs().square().flatten(1).sum(1).clamp_min(1e-7)
    return (error / energy + 1e-12).sqrt().mean()


def run_phase_epoch_v5(
    model,
    loader,
    *,
    config,
    stage,
    device,
    optimizer=None,
    max_batches=0,
    print_freq=50,
) -> dict:
    """``train_v2.run_epoch`` compatible phase-only epoch implementation."""

    if stage != "phase":
        raise ValueError("run_phase_epoch_v5 supports only the phase stage.")
    if print_freq < 1 or max_batches < 0:
        raise ValueError("Print frequency must be positive and batch limit nonnegative.")
    settings = config["training"]["phase"]
    reconstruction_weight = settings.get("reconstruction_loss_weight", 0.1)
    nrmse_weight = settings.get("nrmse_loss_weight", 0.0)
    if any(not math.isfinite(w) or w < 0 for w in (reconstruction_weight, nrmse_weight)):
        raise ValueError("Loss weights must be finite and nonnegative.")
    threshold = config["training"]["support_threshold"]
    training = optimizer is not None
    model.train(training)
    count = min(len(loader), max_batches or len(loader))
    if count < 1:
        raise ValueError("The epoch requires at least one batch.")
    totals, samples = {}, 0
    started = time.monotonic()
    with torch.enable_grad() if training else torch.no_grad():
        for index, raw_batch in enumerate(islice(loader, count), 1):
            batch = prepare_batch(raw_batch, config, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            phase = model(batch["input"])
            wca = phase_retrieval_wca_loss(phase, batch["target_phase"], batch["input"])
            initial = realspace_from_modulus_phase(batch["diffraction"], phase, spatial_dims=2)
            target = batch["realspace"]
            amplitude = target.abs()
            support = amplitude / amplitude.amax((-2, -1), keepdim=True).clamp_min(1e-7) >= threshold
            reconstruction_mae = ambiguity_aware_component_mae(
                initial, target, support, spatial_dims=2
            ).loss
            if nrmse_weight > 0:
                nrmse = aligned_complex_nrmse_loss(initial, target, threshold)
            else:
                with torch.no_grad():
                    nrmse = aligned_complex_nrmse_loss(initial, target, threshold)
            loss = wca + reconstruction_weight * reconstruction_mae + nrmse_weight * nrmse
            values = {
                "loss": loss,
                "phase_wca": wca,
                "reconstruction_mae": reconstruction_mae,
                "complex_nrmse": nrmse,
                "nrmse_loss": nrmse,
            }
            if any(not bool(torch.isfinite(value)) for value in values.values()):
                raise ValueError("Nonfinite phase loss or metric; check data and optimizer settings.")
            if training:
                loss.backward()
                parameters = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
                clip = config["training"]["gradient_clip"]
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
                elif any(not bool(torch.isfinite(p.grad).all()) for p in parameters):
                    raise ValueError("Nonfinite phase gradient.")
                optimizer.step()
            size = batch["input"].shape[0]
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach()) * size
            samples += size
            if index % print_freq == 0 or index == count:
                elapsed = time.monotonic() - started
                LOGGER.info(
                    "%s | batch=%d/%d | loss=%.6g | wca=%.6g | reconstruction_mae=%.6g | nrmse=%.6g | elapsed=%.1fs | eta=%.1fs",
                    "TRAIN" if training else "VAL", index, count, float(loss.detach()),
                    float(wca.detach()), float(reconstruction_mae.detach()), float(nrmse.detach()),
                    elapsed, elapsed / index * (count - index),
                )
    return {
        **{key: value / samples for key, value in totals.items()},
        "samples": samples,
        "elapsed_seconds": time.monotonic() - started,
    }
