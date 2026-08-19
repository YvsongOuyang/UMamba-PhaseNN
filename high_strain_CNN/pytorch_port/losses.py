"""PyTorch equivalent of the TensorFlow weighted circular-average loss."""

from __future__ import annotations

import torch


def _weighted_circular_average(
    weights: torch.Tensor,
    target_phase: torch.Tensor,
    predicted_phase: torch.Tensor,
) -> torch.Tensor:
    spatial_dims = tuple(range(1, target_phase.ndim))
    global_shift = (target_phase - predicted_phase).mean(
        dim=spatial_dims,
        keepdim=True,
    )
    normalized_weights = weights / weights.sum(dim=spatial_dims, keepdim=True)
    phase_error = target_phase - predicted_phase - global_shift
    error = normalized_weights.to(torch.complex64) * torch.exp(
        torch.complex(torch.zeros_like(phase_error), phase_error)
    )
    return 1.0 - error.sum(dim=spatial_dims).abs()


def phase_retrieval_wca_loss(
    predicted_phase: torch.Tensor,
    target_phase: torch.Tensor,
    weights: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Symmetry-aware WCA loss from the TensorFlow implementation."""

    direct, inverted = phase_retrieval_wca_components(
        predicted_phase,
        target_phase,
        weights,
    )
    loss = torch.minimum(direct, inverted)
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Unsupported reduction: {reduction}")


def phase_retrieval_wca_components(
    predicted_phase: torch.Tensor,
    target_phase: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return direct and conjugate/twin WCA errors for every sample."""

    if predicted_phase.ndim == 5 and predicted_phase.shape[1] == 1:
        predicted_phase = predicted_phase[:, 0]
    if target_phase.ndim == 5 and target_phase.shape[1] == 1:
        target_phase = target_phase[:, 0]
    if weights.ndim == 5 and weights.shape[1] == 1:
        weights = weights[:, 0]
    if predicted_phase.shape != target_phase.shape or weights.shape != target_phase.shape:
        raise ValueError(
            "Predicted phase, target phase, and weights must have the same spatial shape."
        )

    direct = _weighted_circular_average(weights, target_phase, predicted_phase)
    inverted = _weighted_circular_average(weights, -target_phase, predicted_phase)
    return direct, inverted
