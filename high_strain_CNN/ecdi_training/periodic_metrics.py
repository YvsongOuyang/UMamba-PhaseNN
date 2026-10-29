"""Full-field complex metrics with periodic translation ambiguity resolved.

Unlike ordinary amplitude-COM centering, FFT correlation also handles objects
split across opposite image edges. Only integer circular translation, conjugate
inversion, one global phase, and whole-field RMS scale are resolved. No cropping,
support mask, clipping, rotation, or local phase correction is applied.
Reference fields are used for evaluation only, never as HIO inputs.
"""
from __future__ import annotations

import torch


METRIC_PROTOCOL = "periodic_integer_translation_twin_global_phase_rms_reference_peak"


@torch.no_grad()
def align_periodic_fields(prediction, target):
    """Return aligned [B,1,H,W] fields with a common reference peak of one.

For each direct/twin candidate, maximize absolute complex cross-correlation
over all integer circular shifts. Global phase is then fitted over the full
image. Select the candidate minimizing full-field complex MSE.
"""
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError("Expected matching complex [B,1,H,W] fields.")
    if not torch.is_complex(prediction) or not torch.is_complex(target):
        raise ValueError("Both fields must be complex.")
    if prediction.device != target.device:
        raise ValueError("Fields must be on the same device.")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("Nonfinite complex field.")
    aligned_predictions, references = [], []
    for sample in range(target.shape[0]):
        pred, truth = prediction[sample, 0], target[sample, 0]
        pred_rms = pred.abs().square().mean().sqrt()
        truth_rms = truth.abs().square().mean().sqrt()
        if not bool((pred_rms > 0) & (truth_rms > 0)):
            raise ValueError("Complex metrics require nonzero fields.")
        pred, truth = pred / pred_rms, truth / truth_rms
        truth_spectrum = torch.fft.fft2(truth)
        twin = torch.conj(torch.roll(torch.flip(pred, (-2, -1)), (1, 1), (-2, -1)))
        best_field, best_mse = None, None
        for candidate in (pred, twin):
            correlation = torch.fft.ifft2(truth_spectrum * torch.fft.fft2(candidate).conj())
            index = int(correlation.abs().argmax())
            shift_y, shift_x = divmod(index, truth.shape[-1])
            candidate = torch.roll(candidate, (shift_y, shift_x), (-2, -1))
            phase = torch.angle((truth.conj() * candidate).sum())
            candidate = candidate * torch.exp(-1j * phase)
            mse = (candidate - truth).abs().square().mean()
            if best_mse is None or bool(mse < best_mse):
                best_field, best_mse = candidate, mse
        peak = truth.abs().amax()
        aligned_predictions.append(best_field / peak)
        references.append(truth / peak)
    return torch.stack(aligned_predictions)[:, None], torch.stack(references)[:, None]


@torch.no_grad()
def complex_image_metrics(prediction, target):
    pred, truth = align_periodic_fields(prediction, target)
    error = (pred - truth).abs().to(torch.float64).flatten(1)
    mse = error.square().mean(1)
    return {
        "complex_psnr_db": -10 * mse.clamp_min(torch.finfo(mse.dtype).tiny).log10(),
        "complex_mae": error.mean(1),
        "complex_mse": mse,
        "complex_rmse": mse.sqrt(),
    }
