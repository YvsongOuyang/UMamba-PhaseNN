"""AutoPhaseNN model variant with a switchable BN/MaxPool order."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from autophasenn_training_pipeline.model_tf_compatible import TFCompatibleAutoPhaseNN


class PoolBNOrder(str, Enum):
    """Supported execution orders for pool-adjacent encoder BN layers."""

    BN_THEN_POOL = "bn_then_pool"
    POOL_THEN_BN = "pool_then_bn"


AFFECTED_BN_LAYERS = (
    "batch_normalization_1",
    "batch_normalization_3",
    "batch_normalization_5",
    "batch_normalization_7",
)


class PoolBNSwapAutoPhaseNN(TFCompatibleAutoPhaseNN):
    """AutoPhaseNN whose four encoder downsampling blocks can swap BN and pool."""

    def __init__(
        self,
        threshold: float = 0.1,
        order: PoolBNOrder | str = PoolBNOrder.BN_THEN_POOL,
    ) -> None:
        super().__init__(threshold=threshold)
        self.pool_bn_order = PoolBNOrder(order)
        self.capture_block_outputs = False
        self.capture_intrinsic_pairs = False
        self.last_block_outputs: dict[str, torch.Tensor] = {}
        self.last_intrinsic_pairs: dict[
            str, tuple[torch.Tensor, torch.Tensor]
        ] = {}

    def set_pool_bn_order(self, order: PoolBNOrder | str) -> None:
        """Select the BN/MaxPool order used by subsequent forward passes."""

        self.pool_bn_order = PoolBNOrder(order)

    def set_capture(self, block_outputs: bool, intrinsic_pairs: bool) -> None:
        """Enable intermediate tensors needed for quantitative comparison."""

        if intrinsic_pairs and self.training:
            raise RuntimeError("Intrinsic commutation capture is only valid in eval mode.")
        self.capture_block_outputs = bool(block_outputs)
        self.capture_intrinsic_pairs = bool(intrinsic_pairs)

    def _encoder_block(self, x, conv1, bn1, conv2, bn2, pool=True):
        x = self._conv_lrelu_bn(x, conv1, bn1)
        pre_bn_pool = self.layers[conv2](x)
        pre_bn_pool = F.leaky_relu(pre_bn_pool, negative_slope=0.01)

        if not pool:
            return self.layers[bn2](pre_bn_pool)

        bn_then_pool = None
        pool_then_bn = None
        if self.pool_bn_order is PoolBNOrder.BN_THEN_POOL or self.capture_intrinsic_pairs:
            bn_then_pool = F.max_pool3d(
                self.layers[bn2](pre_bn_pool), kernel_size=2, stride=2
            )
        if self.pool_bn_order is PoolBNOrder.POOL_THEN_BN or self.capture_intrinsic_pairs:
            pool_then_bn = self.layers[bn2](
                F.max_pool3d(pre_bn_pool, kernel_size=2, stride=2)
            )

        if self.capture_intrinsic_pairs:
            if bn_then_pool is None or pool_then_bn is None:
                raise AssertionError("Both intrinsic order outputs must be available.")
            self.last_intrinsic_pairs[bn2] = (
                bn_then_pool.detach(),
                pool_then_bn.detach(),
            )

        if self.pool_bn_order is PoolBNOrder.BN_THEN_POOL:
            output = bn_then_pool
        else:
            output = pool_then_bn
        if output is None:
            raise AssertionError("Selected encoder order did not produce an output.")

        if self.capture_block_outputs:
            self.last_block_outputs[bn2] = output.detach()
        return output

    def forward(self, x: torch.Tensor):
        """Reconstruct a 3D object from diffraction modulus.

        Args:
            x: Float tensor shaped ``(B, 1, D, H, W)``. The pretrained
                AutoPhaseNN expects ``D = H = W = 64``.

        Returns:
            Tuple containing far-field modulus, complex object, masked
            amplitude, phase, binary support, and raw amplitude, each with
            spatial shape ``(B, 1, 64, 64, 64)`` for the standard input.
        """

        self.last_block_outputs = {}
        self.last_intrinsic_pairs = {}
        return super().forward(x)


def load_checkpoint_file(path: str | Path, allow_unsafe: bool = False) -> Any:
    """Load a checkpoint using tensor-only mode unless explicitly overridden."""

    checkpoint_path = Path(path)
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        if not allow_unsafe:
            raise RuntimeError(
                "Safe checkpoint loading failed. If this is a trusted legacy "
                "checkpoint containing custom Python objects, set "
                "model.allow_unsafe_checkpoint=true explicitly."
            ) from exc
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    """Extract a model state dictionary from common checkpoint layouts."""

    if isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "state_dict", "network_weights", "model"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping) and value:
                return value
        if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return checkpoint
    raise TypeError(
        "Checkpoint does not contain a recognized state dict. Expected one of "
        "model_state_dict/state_dict/network_weights/model or a bare tensor mapping."
    )


def _strip_prefix(
    state_dict: Mapping[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    return {
        (key[len(prefix) :] if key.startswith(prefix) else key): value
        for key, value in state_dict.items()
    }


def load_state_dict_robust(
    model: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    strict: bool = True,
) -> dict[str, list[str]]:
    """Load weights after selecting the common prefix variant with best overlap."""

    expected = set(model.state_dict())
    candidates = [
        dict(state_dict),
        _strip_prefix(state_dict, "module."),
        _strip_prefix(state_dict, "model."),
        _strip_prefix(_strip_prefix(state_dict, "module."), "model."),
    ]
    selected = max(candidates, key=lambda item: len(expected.intersection(item)))
    incompatible = model.load_state_dict(selected, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if strict and (missing or unexpected):
        raise RuntimeError(
            "Strict checkpoint loading failed: "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}"
        )
    return {"missing_keys": missing, "unexpected_keys": unexpected}


def infer_threshold(checkpoint: Any, configured: float | None) -> float:
    """Resolve the support threshold from config or checkpoint metadata."""

    if configured is not None:
        return float(configured)
    if isinstance(checkpoint, Mapping):
        if checkpoint.get("threshold") is not None:
            return float(checkpoint["threshold"])
        args = checkpoint.get("args")
        if isinstance(args, Mapping) and args.get("threshold") is not None:
            return float(args["threshold"])
    return 0.1


@torch.no_grad()
def audit_bn_scales(model: PoolBNSwapAutoPhaseNN) -> dict[str, dict[str, float | int]]:
    """Audit the monotonicity precondition for every pool-adjacent BN layer."""

    audit: dict[str, dict[str, float | int]] = {}
    for name in AFFECTED_BN_LAYERS:
        layer = model.layers[name]
        scale = layer.weight.detach() / torch.sqrt(layer.running_var.detach() + layer.eps)
        positive = int(torch.sum(scale > 0).item())
        zero = int(torch.sum(scale == 0).item())
        negative = int(torch.sum(scale < 0).item())
        channels = int(scale.numel())
        audit[name] = {
            "channels": channels,
            "positive": positive,
            "zero": zero,
            "negative": negative,
            "positive_fraction": positive / channels,
            "min_effective_scale": float(scale.min().cpu()),
            "max_effective_scale": float(scale.max().cpu()),
            "mean_effective_scale": float(scale.mean().cpu()),
        }
    return audit


@torch.no_grad()
def audit_all_bn_scales(
    model: PoolBNSwapAutoPhaseNN,
) -> dict[str, dict[str, float | int]]:
    """Audit effective scales for every BatchNorm3d layer in the full model."""

    audit: dict[str, dict[str, float | int]] = {}
    for name, layer in model.layers.items():
        if not isinstance(layer, torch.nn.BatchNorm3d):
            continue
        scale = layer.weight.detach() / torch.sqrt(layer.running_var.detach() + layer.eps)
        positive = int(torch.sum(scale > 0).item())
        zero = int(torch.sum(scale == 0).item())
        negative = int(torch.sum(scale < 0).item())
        channels = int(scale.numel())
        audit[name] = {
            "channels": channels,
            "positive": positive,
            "zero": zero,
            "negative": negative,
            "positive_fraction": positive / channels,
            "min_effective_scale": float(scale.min().cpu()),
            "max_effective_scale": float(scale.max().cpu()),
            "mean_effective_scale": float(scale.mean().cpu()),
        }
    return audit
