"""Evaluate AutoPhaseNN with raw reciprocal and official post-processed real-space metrics."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import time
from concurrent.futures import Executor, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .dataset import AutoPhaseDataset
    from .losses import (
        FIXED_METRIC_DESCRIPTIONS,
        METRIC_DESCRIPTIONS,
        fixed_metric_groups,
        format_fixed_metric_groups,
        free_metric_tensor_dict,
        group_metrics,
        metric_tensor_dict,
        realspace_metric_tensor_dict,
    )
    from .model_factory import (
        MODEL_VARIANTS,
        create_model,
        resolve_support_threshold,
    )
    from .model_tf_compatible import load_weights
except ImportError:
    from dataset import AutoPhaseDataset
    from losses import (
        FIXED_METRIC_DESCRIPTIONS,
        METRIC_DESCRIPTIONS,
        fixed_metric_groups,
        format_fixed_metric_groups,
        free_metric_tensor_dict,
        group_metrics,
        metric_tensor_dict,
        realspace_metric_tensor_dict,
    )
    from model_factory import MODEL_VARIANTS, create_model, resolve_support_threshold
    from model_tf_compatible import load_weights


LOGGER = logging.getLogger("autophasenn.evaluate")
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_EVALUATE_ROOT = PROJECT_DIR / "evaluate"
DEFAULT_CHECKPOINT = "/data_ssd/oyys/autophasenn/autophasenn_pipeline_output/decoder_cross_concat_scratch_bs4_lr1e-3_20260818_151115/checkpoint_best.pt"

PAPER_METRICS = {
    "paper_modulus_mae": {
        "paper_reference": "Eq. (1): MAE between measured and estimated diffraction modulus.",
        "direction": "lower",
    },
    "chi2_modulus": {
        "paper_reference": "Eq. (2): reciprocal-space chi-square on diffraction modulus.",
        "direction": "lower",
    },
    "real_amp_ssim": {
        "paper_reference": "Fig. 2: local-window 3D SSIM of real-space amplitude.",
        "direction": "higher",
    },
    "r_factor_free": {
        "paper_reference": "Free R-factor family evaluated on held-out reciprocal voxels.",
        "direction": "lower",
    },
    "llk_free": {
        "paper_reference": "Supplementary Note 3: free Poisson log-likelihood diagnostic.",
        "direction": "lower",
    },
    "chi2_free": {
        "paper_reference": "Supplementary Note 3: free chi-square diagnostic.",
        "direction": "lower",
    },
}

THRESHOLD_SWEEP_METRICS = (
    "real_amp_l1",
    "real_amp_ssim",
    "real_support_iou",
    "real_support_dice",
    "real_support_volume_ratio",
    "real_phase_mae_true_support",
)


def optional_data_path(data_dir: Path, filename: str | None) -> Path | None:
    """Resolve an optional data filename relative to the data directory."""

    if filename is None or filename.lower() in {"", "none", "null"}:
        return None
    return data_dir / filename


def choose_device(name: str) -> torch.device:
    """Resolve the requested device with an explicit CUDA fallback."""

    if name == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA was requested but is unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(name)


def configure_logging(output_dir: Path, level: str) -> None:
    """Configure matching console and file logs for an evaluation run."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for handler in LOGGER.handlers:
        handler.close()
    LOGGER.handlers.clear()
    LOGGER.setLevel(getattr(logging, level.upper()))
    LOGGER.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(
        output_dir / "evaluation.log",
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(console_handler)
    LOGGER.addHandler(file_handler)


def resolve_output_dir(args: argparse.Namespace) -> Path:
    """Resolve an explicit directory or the model-specific evaluation default."""

    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else DEFAULT_EVALUATE_ROOT / f"evaluate_{args.model_variant}"
    )
    return output_dir.resolve()


def parse_args() -> argparse.Namespace:
    """Parse all data, model, metric, and output parameters."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate AutoPhaseNN with paper metrics and write JSON, CSV, "
            "Markdown, and log artifacts."
        )
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", default="/data_ssd/oyys/autophasenn/")
    parser.add_argument("--data-diff", default="val_diff.npy")
    parser.add_argument("--data-real", default="val_real.npy")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument(
        "--model-variant",
        choices=MODEL_VARIANTS,
        default="decoder_cross_concat",
        help="Network architecture variant.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate at most this many samples; zero uses --num-samples.",
    )
    parser.add_argument("--shape", type=int, default=64)
    parser.add_argument("--dtype-diff", default="float32")
    parser.add_argument("--dtype-real", default="complex64")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Support threshold; defaults to 0.3 for mamba_skip and 0.1 otherwise.",
    )
    parser.add_argument(
        "--threshold-sweep",
        type=float,
        nargs="+",
        default=(),
        metavar="T",
        help=(
            "Evaluate additional real-space support thresholds in the same pass. "
            "The primary --threshold is always included and remains the headline result."
        ),
    )
    parser.add_argument(
        "--ssim-window-size",
        type=int,
        default=7,
        help="Odd cubic window size for paper-style 3D amplitude SSIM.",
    )
    parser.add_argument(
        "--postprocess-workers",
        type=int,
        default=0,
        help=(
            "Threads for the exact skimage phase unwrap; zero selects up to "
            "eight workers automatically."
        ),
    )
    parser.add_argument(
        "--free-mask",
        default="",
        help=(
            "Optional NumPy .npy/.npz boolean holdout mask with spatial shape. "
            "It takes precedence over --free-fraction."
        ),
    )
    parser.add_argument(
        "--free-fraction",
        type=float,
        default=0.05,
        help="Deterministic diagnostic holdout fraction when --free-mask is absent; zero disables free metrics.",
    )
    parser.add_argument("--free-seed", type=int, default=42)
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=1,
        help="Untimed forward batches used to warm up the selected device.",
    )
    parser.add_argument(
        "--reference-runtime-ms",
        type=float,
        default=0.0,
        help="Optional per-sample baseline used to calculate a machine-specific speedup.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Artifact directory; empty uses "
            "<project>/evaluate/evaluate_<model-variant>."
        ),
    )
    parser.add_argument(
        "--non-strict-checkpoint",
        action="store_true",
        help="Allow checkpoint keys that do not exactly match the model.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    args = parser.parse_args()
    args.threshold = resolve_support_threshold(args.model_variant, args.threshold)
    return args


def validate_args(args: argparse.Namespace) -> None:
    """Reject invalid metric and loader settings before allocating data."""

    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive.")
    if args.limit < 0:
        raise ValueError("--limit cannot be negative.")
    if args.shape < 1:
        raise ValueError("--shape must be positive.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if args.postprocess_workers < 0:
        raise ValueError("--postprocess-workers cannot be negative.")
    thresholds = (args.threshold, *args.threshold_sweep)
    if any(not math.isfinite(value) or value < 0 for value in thresholds):
        raise ValueError("Support thresholds must be finite and nonnegative.")
    if not 0.0 <= args.free_fraction < 1.0:
        raise ValueError("--free-fraction must be in [0, 1).")
    if args.ssim_window_size < 1 or args.ssim_window_size % 2 == 0:
        raise ValueError("--ssim-window-size must be a positive odd integer.")
    if args.ssim_window_size > args.shape:
        raise ValueError("--ssim-window-size cannot exceed --shape.")
    if args.warmup_batches < 0:
        raise ValueError("--warmup-batches cannot be negative.")
    if args.reference_runtime_ms < 0:
        raise ValueError("--reference-runtime-ms cannot be negative.")


def resolve_threshold_sweep(
    primary_threshold: float,
    requested_thresholds: tuple[float, ...] | list[float],
) -> tuple[float, ...]:
    """Return sorted unique sweep thresholds with the primary value included."""

    if not requested_thresholds:
        return ()
    resolved: list[float] = []
    for value in (primary_threshold, *requested_thresholds):
        threshold = float(value)
        if not any(math.isclose(threshold, current) for current in resolved):
            resolved.append(threshold)
    return tuple(sorted(resolved))


def load_external_free_mask(
    mask_path: Path, shape: tuple[int, int, int]
) -> torch.Tensor:
    """Load and validate a standard NumPy reciprocal-space holdout mask."""

    loaded = np.load(mask_path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        keys = loaded.files
        if len(keys) != 1:
            loaded.close()
            raise ValueError("Free-mask .npz files must contain exactly one array.")
        array = np.asarray(loaded[keys[0]])
        loaded.close()
    else:
        array = np.asarray(loaded)
    array = np.squeeze(array)
    if tuple(array.shape) != shape:
        raise ValueError(
            f"Free mask has spatial shape {tuple(array.shape)}; expected {shape}."
        )
    mask = torch.from_numpy(np.asarray(array > 0, dtype=np.bool_)).reshape(
        (1, 1) + shape
    )
    if not bool(mask.any()):
        raise ValueError("Free mask does not select any reciprocal-space voxels.")
    return mask


def generate_free_mask(
    shape: tuple[int, int, int],
    fraction: float,
    seed: int,
) -> torch.Tensor | None:
    """Generate a deterministic diagnostic holdout mask shared by all samples."""

    if fraction <= 0.0:
        return None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    mask = torch.rand((1, 1) + shape, generator=generator) < fraction
    flat_mask = mask.reshape(-1)
    if not bool(flat_mask.any()):
        flat_mask[0] = True
    if bool(flat_mask.all()):
        flat_mask[-1] = False
    return mask


def resolve_free_mask(
    args: argparse.Namespace,
    shape: tuple[int, int, int],
) -> tuple[torch.Tensor | None, dict[str, object]]:
    """Resolve the free-mask source and provenance recorded in the report."""

    if args.free_mask:
        mask_path = Path(args.free_mask).expanduser().resolve()
        mask = load_external_free_mask(mask_path, shape)
        source = "external_holdout_mask"
        note = (
            "Metrics use the supplied mask. They are true free metrics only if these voxels "
            "were excluded from the reconstruction/training objective."
        )
        provenance: dict[str, object] = {"path": str(mask_path)}
    else:
        mask = generate_free_mask(shape, args.free_fraction, args.free_seed)
        source = "generated_diagnostic_mask" if mask is not None else "disabled"
        note = (
            "The generated mask is reproducible but was not automatically excluded during "
            "model training. Treat Rfree, LLKfree, and chi2free as diagnostics, not as a "
            "numerically comparable reproduction of the paper."
            if mask is not None
            else "Free metrics are disabled because --free-fraction is zero."
        )
        provenance = {
            "requested_fraction": args.free_fraction,
            "seed": args.free_seed,
        }
    actual_fraction = float(mask.float().mean()) if mask is not None else 0.0
    return mask, {
        "enabled": mask is not None,
        "source": source,
        "actual_fraction": actual_fraction,
        "selected_voxels": int(mask.sum()) if mask is not None else 0,
        "note": note,
        **provenance,
    }


def synchronize(device: torch.device) -> None:
    """Synchronize CUDA so model timing excludes asynchronous launch delay."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def unpack_outputs(outputs: object) -> tuple[torch.Tensor, ...]:
    """Validate the AutoPhaseNN forward-output contract."""

    if not isinstance(outputs, (tuple, list)) or len(outputs) < 5:
        raise RuntimeError(
            "Model must return (pred_diff, pred_obj, pred_amp, pred_phi, support)."
        )
    return tuple(outputs[:5])


def raw_amp_from_outputs(
    outputs: object,
    masked_amp: torch.Tensor,
) -> torch.Tensor:
    """Return pre-support amplitude when the six-output model contract provides it."""

    if isinstance(outputs, (tuple, list)) and len(outputs) >= 6:
        return outputs[5]
    return masked_amp


def shift_center_of_mass(
    amp: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Center amplitude and phase together using the official TF2 convention."""

    from scipy.ndimage import center_of_mass, shift

    center = center_of_mass(amp)
    if any(np.isnan(value) for value in center):
        return amp, phi
    deltas = tuple(
        int(round(size / 2.0 - coordinate))
        for size, coordinate in zip(amp.shape, center)
    )
    return (
        shift(amp, shift=deltas, mode="wrap"),
        shift(phi, shift=deltas, mode="wrap"),
    )


def shift_support(support: np.ndarray) -> np.ndarray:
    """Center predicted support like the official TF2 ``shift_sup`` helper."""

    from scipy.ndimage import center_of_mass, shift

    center = center_of_mass(support)
    if any(np.isnan(value) for value in center):
        return support
    deltas = tuple(
        int(round(size / 2.0 - coordinate))
        for size, coordinate in zip(support.shape, center)
    )
    return shift(support, shift=deltas, mode="wrap")


def official_post_process(
    amp: np.ndarray,
    phi: np.ndarray,
    threshold: float = 0.1,
    unwrap: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the official TF2 amplitude/phase test-time post-processing."""

    amp_out, phi_out = official_post_process_before_shift(
        amp,
        phi,
        threshold=threshold,
        unwrap=unwrap,
    )
    return center_post_processed_object(amp_out, phi_out, threshold=threshold)


def official_post_process_before_shift(
    amp: np.ndarray,
    phi: np.ndarray,
    threshold: float = 0.1,
    unwrap: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply official real-space post-processing up to the center shift."""

    from skimage.restoration import unwrap_phase

    amp = np.asarray(amp, dtype=np.float32)
    phi = np.asarray(phi, dtype=np.float32).reshape(amp.shape)
    if unwrap:
        phi = unwrap_phase(phi)

    mask = amp > threshold
    amp_out = np.where(mask, amp, 0.0)
    phi_out = np.where(mask, phi, 0.0)
    selected = amp_out > threshold
    mean_phi = float(np.mean(phi_out[selected])) if np.any(selected) else 0.0
    phi_out = phi_out - mean_phi
    return amp_out.astype(np.float32), phi_out.astype(np.float32)


def center_post_processed_object(
    amp: np.ndarray,
    phi: np.ndarray,
    threshold: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Center an already post-processed object and restore its support mask."""

    amp_out, phi_out = shift_center_of_mass(amp, phi)
    mask = amp_out > threshold
    amp_out = np.where(mask, amp_out, 0.0)
    phi_out = np.where(mask, phi_out, 0.0)
    return amp_out.astype(np.float32), phi_out.astype(np.float32)


def post_process_realspace_sample(
    true_amp: torch.Tensor,
    true_phi: torch.Tensor,
    pred_amp: torch.Tensor,
    pred_phi: torch.Tensor,
    pred_support: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, ...]:
    """Post-process one ``(1, 1, D, H, W)`` sample and restore tensor layout."""

    def as_volume(value: torch.Tensor) -> np.ndarray:
        return value.detach().cpu().numpy()[0, 0]

    def as_tensor(value: np.ndarray, reference: torch.Tensor) -> torch.Tensor:
        contiguous = np.ascontiguousarray(value[None, None], dtype=np.float32)
        return torch.from_numpy(contiguous).to(
            device=reference.device,
            dtype=reference.dtype,
        )

    true_amp_post, true_phi_post = official_post_process(
        as_volume(true_amp),
        as_volume(true_phi),
        threshold=threshold,
        unwrap=True,
    )
    pred_amp_post, pred_phi_post = official_post_process(
        as_volume(pred_amp),
        as_volume(pred_phi),
        threshold=threshold,
        unwrap=True,
    )
    support_post = shift_support(as_volume(pred_support)).astype(np.float32)
    return (
        as_tensor(true_amp_post, true_amp),
        as_tensor(true_phi_post, true_phi),
        as_tensor(pred_amp_post, pred_amp),
        as_tensor(pred_phi_post, pred_phi),
        as_tensor(support_post, pred_support),
    )


def resolve_postprocess_workers(configured: int, batch_size: int) -> int:
    """Resolve a bounded worker count for independent phase-unwrapping jobs."""

    if configured > 0:
        return configured
    return max(1, min(batch_size * 2, os.cpu_count() or 1, 8))


def unwrap_phase_volumes(
    phase_batch: np.ndarray,
    executor: Executor | None = None,
) -> np.ndarray:
    """Apply the official skimage unwrap independently to a batch of volumes."""

    from skimage.restoration import unwrap_phase

    phase_batch = np.asarray(phase_batch, dtype=np.float32)
    if phase_batch.ndim != 5 or phase_batch.shape[1] != 1:
        raise ValueError("Phase batch must have shape (B, 1, D, H, W).")
    volumes = [phase_batch[index, 0] for index in range(phase_batch.shape[0])]
    if executor is None:
        unwrapped = [unwrap_phase(volume) for volume in volumes]
    else:
        unwrapped = list(executor.map(unwrap_phase, volumes))
    return np.ascontiguousarray(
        np.stack(unwrapped, axis=0)[:, None],
        dtype=np.float32,
    )


def center_of_mass_shifts(values: torch.Tensor) -> torch.Tensor:
    """Calculate official integer center-of-mass shifts for a tensor batch."""

    if values.ndim != 5:
        raise ValueError("Center-of-mass input must have shape (B, C, D, H, W).")
    reduction_dims = tuple(range(1, values.ndim))
    mass = torch.sum(values, dim=reduction_dims)
    shifts: list[torch.Tensor] = []
    for axis, size in enumerate(values.shape[-3:]):
        dim = values.ndim - 3 + axis
        coordinate_shape = [1] * values.ndim
        coordinate_shape[dim] = size
        coordinates = torch.arange(
            size,
            device=values.device,
            dtype=values.dtype,
        ).reshape(coordinate_shape)
        center = torch.sum(values * coordinates, dim=reduction_dims) / mass
        valid = torch.isfinite(center) & torch.isfinite(mass) & (mass != 0)
        shift = torch.round(size / 2.0 - center)
        shifts.append(torch.where(valid, shift, torch.zeros_like(shift)).long())
    return torch.stack(shifts, dim=1)


def scipy_wrap_shift_batch(
    values: torch.Tensor,
    shifts: torch.Tensor,
) -> torch.Tensor:
    """Apply integer shifts with SciPy ``mode='wrap'`` boundary semantics."""

    if values.ndim != 5:
        raise ValueError("Shift input must have shape (B, C, D, H, W).")
    if shifts.shape != (values.shape[0], 3):
        raise ValueError("Shifts must have shape (B, 3).")
    shifted = values
    for axis, size in enumerate(shifted.shape[-3:]):
        if size <= 1:
            continue
        dim = shifted.ndim - 3 + axis
        source = torch.arange(size, device=shifted.device)[None, :]
        source = source - shifts[:, axis, None]
        outside = (source < 0) | (source > size - 1)
        source = torch.where(outside, torch.remainder(source, size - 1), source)
        index_shape = [1] * shifted.ndim
        index_shape[0] = shifted.shape[0]
        index_shape[dim] = size
        gather_index = source.reshape(index_shape).expand_as(shifted)
        shifted = torch.gather(shifted, dim=dim, index=gather_index)
    return shifted


def official_post_process_tensor_batch(
    amp: torch.Tensor,
    unwrapped_phi: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the official mask, phase offset, and centering steps in a batch."""

    if amp.shape != unwrapped_phi.shape or amp.ndim != 5:
        raise ValueError("Amplitude and phase must share shape (B, C, D, H, W).")
    mask = amp > threshold
    amp_out = torch.where(mask, amp, torch.zeros_like(amp))
    phi_out = torch.where(mask, unwrapped_phi, torch.zeros_like(unwrapped_phi))
    selected = amp_out > threshold
    selected_flat = selected.flatten(start_dim=1)
    selected_count = torch.sum(selected_flat, dim=1)
    phase_sum = torch.sum((phi_out * selected).flatten(start_dim=1), dim=1)
    mean_phi = torch.where(
        selected_count > 0,
        phase_sum / selected_count.clamp_min(1),
        torch.zeros_like(phase_sum),
    )
    phi_out = phi_out - mean_phi.reshape((-1,) + (1,) * (phi_out.ndim - 1))

    shifts = center_of_mass_shifts(amp_out)
    amp_out = scipy_wrap_shift_batch(amp_out, shifts)
    phi_out = scipy_wrap_shift_batch(phi_out, shifts)
    mask = amp_out > threshold
    return (
        torch.where(mask, amp_out, torch.zeros_like(amp_out)),
        torch.where(mask, phi_out, torch.zeros_like(phi_out)),
    )


def post_process_realspace_batch(
    true_amp: torch.Tensor,
    true_phi: torch.Tensor,
    pred_amp: torch.Tensor,
    pred_phi: torch.Tensor,
    pred_support: torch.Tensor,
    threshold: float,
    executor: Executor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Batch official post-processing while keeping tensor operations on device."""

    true_phi_unwrapped, pred_phi_unwrapped = unwrap_realspace_phase_pair(
        true_phi,
        pred_phi,
        executor=executor,
    )
    return post_process_unwrapped_realspace(
        true_amp,
        true_phi_unwrapped,
        pred_amp,
        pred_phi_unwrapped,
        threshold,
        pred_support=pred_support,
    )


def unwrap_realspace_phase_pair(
    true_phi: torch.Tensor,
    pred_phi: torch.Tensor,
    executor: Executor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unwrap target and prediction once for every requested support threshold."""

    batch_size = pred_phi.shape[0]
    phase_batch = np.concatenate(
        (
            true_phi.detach().to(device="cpu", dtype=torch.float32).numpy(),
            pred_phi.detach().to(device="cpu", dtype=torch.float32).numpy(),
        ),
        axis=0,
    )
    unwrapped = unwrap_phase_volumes(phase_batch, executor=executor)
    unwrapped_tensor = torch.from_numpy(unwrapped).to(
        device=pred_phi.device,
        dtype=pred_phi.dtype,
    )
    return torch.split(unwrapped_tensor, [batch_size, batch_size], dim=0)


def post_process_unwrapped_realspace(
    true_amp: torch.Tensor,
    true_phi_unwrapped: torch.Tensor,
    pred_amp: torch.Tensor,
    pred_phi_unwrapped: torch.Tensor,
    threshold: float,
    pred_support: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Apply the shared official processing after support threshold selection."""

    true_amp_device = true_amp.to(
        device=pred_amp.device,
        dtype=pred_amp.dtype,
        non_blocking=pred_amp.device.type == "cuda",
    )
    true_amp_post, true_phi_post = official_post_process_tensor_batch(
        true_amp_device,
        true_phi_unwrapped,
        threshold=threshold,
    )
    pred_amp_post, pred_phi_post = official_post_process_tensor_batch(
        pred_amp,
        pred_phi_unwrapped,
        threshold=threshold,
    )
    if pred_support is None:
        pred_support = (pred_amp >= threshold).float()
    support_shifts = center_of_mass_shifts(pred_support)
    pred_support_post = scipy_wrap_shift_batch(pred_support, support_shifts)
    return (
        true_amp_post,
        true_phi_post,
        pred_amp_post,
        pred_phi_post,
        pred_support_post,
    )


def materialize_metric_rows(
    metrics: dict[str, torch.Tensor],
) -> list[dict[str, float]]:
    """Transfer a complete batch of scalar metrics to CPU in one operation."""

    if not metrics:
        return []
    keys = list(metrics)
    columns = [metrics[key].reshape(-1) for key in keys]
    batch_size = columns[0].shape[0]
    if any(column.shape != (batch_size,) for column in columns):
        raise ValueError("Every metric tensor must contain one scalar per sample.")
    matrix = torch.stack(columns, dim=1).detach().cpu().numpy()
    return [
        {
            key: float(matrix[row_index, column_index])
            for column_index, key in enumerate(keys)
        }
        for row_index in range(batch_size)
    ]


def timed_model_forward(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
) -> tuple[object, float]:
    """Time one forward pass without a redundant pre-forward CUDA sync."""

    if device.type != "cuda":
        started = time.perf_counter()
        return model(inputs), time.perf_counter() - started
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    started.record()
    outputs = model(inputs)
    finished.record()
    finished.synchronize()
    return outputs, started.elapsed_time(finished) / 1000.0


@torch.inference_mode()
def warm_up_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    warmup_batches: int,
) -> None:
    """Run untimed forward passes before collecting performance metrics."""

    if warmup_batches <= 0:
        return
    completed = 0
    for batch in loader:
        diff = batch["diff"].to(device, non_blocking=True).float()
        model(diff)
        completed += 1
        if completed >= warmup_batches:
            break
    synchronize(device)
    LOGGER.info("Completed %d warmup batch(es).", completed)


def add_metrics(total: dict[str, float], metrics: dict[str, float]) -> None:
    """Accumulate scalar metric values."""

    for key, value in metrics.items():
        total[key] = total.get(key, 0.0) + float(value)


def metric_statistics(
    per_sample: list[dict[str, object]]
) -> dict[str, dict[str, float]]:
    """Calculate readable distribution statistics for every scalar metric."""

    metric_keys = sorted(
        {
            key
            for row in per_sample
            for key, value in row.items()
            if key != "name" and isinstance(value, (int, float))
        }
    )
    statistics: dict[str, dict[str, float]] = {}
    for key in metric_keys:
        values = np.asarray(
            [float(row[key]) for row in per_sample if key in row],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        statistics[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }
    return statistics


def summarize_threshold_sweep(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    """Aggregate sweep metrics and select transparent support operating points."""

    thresholds = sorted({float(row["threshold"]) for row in rows})
    summaries: list[dict[str, object]] = []
    for threshold in thresholds:
        threshold_rows = [
            row for row in rows if math.isclose(float(row["threshold"]), threshold)
        ]
        statistics = metric_statistics(threshold_rows)
        statistics.pop("threshold", None)
        summaries.append(
            {
                "threshold": threshold,
                "num_samples": len(threshold_rows),
                "mean": {
                    metric: statistics[metric]["mean"]
                    for metric in THRESHOLD_SWEEP_METRICS
                },
                "metric_statistics": statistics,
            }
        )

    best_iou = max(
        summaries,
        key=lambda item: item["mean"]["real_support_iou"],
    )
    closest_volume = min(
        summaries,
        key=lambda item: abs(item["mean"]["real_support_volume_ratio"] - 1.0),
    )
    return {
        "enabled": True,
        "selection_scope": "validation_diagnostic_only",
        "primary_threshold_unchanged": True,
        "metrics": list(THRESHOLD_SWEEP_METRICS),
        "summaries": summaries,
        "best_mean_iou_threshold": best_iou["threshold"],
        "closest_mean_volume_ratio_threshold": closest_volume["threshold"],
    }


def write_json(path: Path, report: dict[str, object]) -> None:
    """Write the full machine-readable report."""

    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def ordered_metric_keys(per_sample: list[dict[str, object]]) -> list[str]:
    """Place paper metrics first and diagnostics afterward in tabular output."""

    available = {key for row in per_sample for key in row if key != "name"}
    paper_keys = [key for key in PAPER_METRICS if key in available]
    return paper_keys + sorted(available.difference(paper_keys))


def write_sample_csv(path: Path, per_sample: list[dict[str, object]]) -> None:
    """Write one evaluation row per sample for spreadsheet analysis."""

    fieldnames = ["name", *ordered_metric_keys(per_sample)]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(per_sample)


def write_threshold_sweep_summary_csv(
    path: Path,
    sweep: dict[str, object],
) -> None:
    """Write one compact aggregate row per support threshold."""

    fieldnames = ["threshold", "num_samples", *THRESHOLD_SWEEP_METRICS]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for summary in sweep["summaries"]:
            writer.writerow(
                {
                    "threshold": summary["threshold"],
                    "num_samples": summary["num_samples"],
                    **summary["mean"],
                }
            )


def format_number(value: object) -> str:
    """Format report scalars compactly without losing small values."""

    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        return f"{value:.6g}"
    return str(value)


def render_markdown(report: dict[str, object]) -> str:
    """Render a human-readable summary emphasizing paper-reported metrics."""

    run = report["run"]
    statistics = report["metric_statistics"]
    free_mask = report["free_mask"]
    timing = report["timing"]
    paper_coverage = report["paper_metric_coverage"]
    grouped_mean = report["mean_metric_groups"]
    sweep_display = ", ".join(
        format_number(value) for value in run["support_threshold_sweep"]
    ) or "disabled"
    lines = [
        "# AutoPhaseNN Evaluation Summary",
        "",
        "## Run",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Checkpoint | `{run['checkpoint']}` |",
        f"| Checkpoint epoch | {format_number(run['checkpoint_epoch'])} |",
        f"| Model variant | `{run['model_variant']}` |",
        f"| Device | `{run['device']}` |",
        f"| PyTorch | `{run['torch_version']}` |",
        f"| CUDA runtime | {format_number(run['cuda_version'])} |",
        f"| GPU | {format_number(run['gpu_name'])} |",
        f"| Samples | {run['num_samples']} |",
        f"| Batch size | {run['batch_size']} |",
        f"| Support threshold | {format_number(run['support_threshold'])} |",
        f"| Support threshold sweep | {sweep_display} |",
        f"| SSIM window | {run['ssim_window_size']} x {run['ssim_window_size']} x {run['ssim_window_size']} |",
        f"| Real-space ground truth | {run['realspace_metrics']} |",
        f"| Real-space post-process | `{run['realspace_post_process']}` |",
        f"| Post-process tensor device | `{run['postprocess_tensor_device']}` |",
        f"| Phase unwrap workers | {run['postprocess_workers']} |",
        f"| Total evaluation wall time | {format_number(timing['evaluation_wall_seconds'])} s |",
        f"| Mean model inference | {format_number(timing['mean_inference_ms_per_sample'])} ms/sample |",
        f"| Model throughput | {format_number(timing['throughput_samples_per_second'])} samples/s |",
        "",
        "## Paper Metric Coverage",
        "",
        "| Metric | Mean | Std | P50 | P95 | Better | Paper usage |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for key, metadata in paper_coverage.items():
        stats = statistics.get(key)
        if stats is None:
            values = ["n/a"] * 4
        else:
            values = [
                format_number(stats["mean"]),
                format_number(stats["std"]),
                format_number(stats["p50"]),
                format_number(stats["p95"]),
            ]
        lines.append(
            f"| `{key}` | {values[0]} | {values[1]} | {values[2]} | {values[3]} "
            f"| {metadata['direction']} | {metadata['paper_reference']} |"
        )

    threshold_sweep = report["threshold_sweep"]
    if threshold_sweep["enabled"]:
        lines.extend(
            [
                "",
                "## Support Threshold Sweep",
                "",
                "The primary threshold remains the headline operating point. Sweep-selected "
                "thresholds are validation diagnostics and must not be treated as an "
                "independently tested hyperparameter choice.",
                "",
                "| Threshold | Primary | Amp L1 | Amp SSIM | Support IoU | "
                "Support Dice | Volume ratio | Phase MAE |",
                "|---:|:---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for summary in threshold_sweep["summaries"]:
            threshold = summary["threshold"]
            values = summary["mean"]
            is_primary = (
                "yes" if math.isclose(threshold, run["support_threshold"]) else ""
            )
            lines.append(
                f"| {format_number(threshold)} | {is_primary} | "
                f"{format_number(values['real_amp_l1'])} | "
                f"{format_number(values['real_amp_ssim'])} | "
                f"{format_number(values['real_support_iou'])} | "
                f"{format_number(values['real_support_dice'])} | "
                f"{format_number(values['real_support_volume_ratio'])} | "
                f"{format_number(values['real_phase_mae_true_support'])} |"
            )
        lines.extend(
            [
                "",
                "- Best mean-IoU threshold: "
                f"`{format_number(threshold_sweep['best_mean_iou_threshold'])}`.",
                "- Threshold with mean volume ratio closest to one: "
                f"`{format_number(threshold_sweep['closest_mean_volume_ratio_threshold'])}`.",
            ]
        )

    lines.extend(
        [
            "",
            "## Free-Metric Provenance",
            "",
            f"- Source: `{free_mask['source']}`",
            f"- Selected voxels: {free_mask['selected_voxels']} "
            f"({format_number(100.0 * free_mask['actual_fraction'])}%)",
            f"- Interpretation: {free_mask['note']}",
            "",
            "## Mean Metrics by Group",
            "",
        ]
    )
    for group_name, metrics in grouped_mean.items():
        lines.extend([f"### {group_name.replace('_', ' ').title()}", ""])
        lines.extend(["| Metric | Mean | Meaning |", "|---|---:|---|"])
        for key, value in metrics.items():
            description = METRIC_DESCRIPTIONS.get(
                key, "Additional evaluation diagnostic."
            )
            lines.append(f"| `{key}` | {format_number(value)} | {description} |")
        lines.append("")

    lines.extend(
        [
            "## Files",
            "",
            "- `evaluation_results.json`: full configuration, provenance, distributions, and per-sample values.",
            "- `evaluation_samples.csv`: one row per evaluated sample.",
            "- `evaluation_summary.md`: this readable summary.",
            "- `evaluation.log`: execution log and resolved paths.",
            "",
        ]
    )
    if threshold_sweep["enabled"]:
        lines.extend(
            [
                "- `evaluation_threshold_sweep.csv`: one aggregate row per threshold.",
                "- `evaluation_threshold_sweep_samples.csv`: per-sample sweep metrics.",
                "",
            ]
        )
    return "\n".join(lines)


@torch.inference_mode()
def evaluate(
    args: argparse.Namespace,
    model: torch.nn.Module,
    loader: DataLoader,
    dataset: AutoPhaseDataset,
    device: torch.device,
    free_mask: torch.Tensor | None,
    postprocess_workers: int,
) -> tuple[
    list[dict[str, object]],
    dict[str, float],
    list[dict[str, object]],
]:
    """Evaluate all samples with model.eval() and inference-only autograd state."""

    model.eval()
    total: dict[str, float] = {}
    per_sample: list[dict[str, object]] = []
    threshold_sweep_rows: list[dict[str, object]] = []
    sweep_thresholds = resolve_threshold_sweep(
        args.threshold,
        args.threshold_sweep,
    )
    has_realspace = dataset.mmap_real is not None
    device_free_mask = free_mask.to(device) if free_mask is not None else None
    progress = tqdm(loader, desc="AutoPhaseNN evaluation", unit="batch")
    unwrap_executor = (
        ThreadPoolExecutor(
            max_workers=postprocess_workers,
            thread_name_prefix="phase-unwrap",
        )
        if has_realspace and postprocess_workers > 1
        else None
    )
    try:
        for batch in progress:
            diff = batch["diff"].to(device, non_blocking=True).float()
            outputs, batch_inference_seconds = timed_model_forward(model, diff, device)
            pred_diff, _pred_obj, pred_amp, pred_phi, support = unpack_outputs(outputs)
            raw_pred_amp = raw_amp_from_outputs(outputs, pred_amp)
            inference_ms_per_sample = 1000.0 * batch_inference_seconds / diff.shape[0]

            metric_tensors = metric_tensor_dict(diff, pred_diff)
            if device_free_mask is not None:
                metric_tensors.update(
                    free_metric_tensor_dict(diff, pred_diff, device_free_mask)
                )
            if has_realspace and sweep_thresholds:
                true_phi_unwrapped, pred_phi_unwrapped = unwrap_realspace_phase_pair(
                    batch["phi"].float(),
                    pred_phi,
                    executor=unwrap_executor,
                )
                primary_realspace_metrics = None
                for threshold in sweep_thresholds:
                    processed = post_process_unwrapped_realspace(
                        batch["amp"].float(),
                        true_phi_unwrapped,
                        raw_pred_amp,
                        pred_phi_unwrapped,
                        threshold,
                    )
                    realspace_metrics = realspace_metric_tensor_dict(
                        *processed,
                        threshold=threshold,
                        ssim_window_size=args.ssim_window_size,
                    )
                    if math.isclose(threshold, args.threshold):
                        primary_realspace_metrics = realspace_metrics
                    sweep_metrics = {
                        key: realspace_metrics[key]
                        for key in THRESHOLD_SWEEP_METRICS
                    }
                    batch_sweep_rows = materialize_metric_rows(sweep_metrics)
                    for name, metrics in zip(batch["name"], batch_sweep_rows):
                        threshold_sweep_rows.append(
                            {
                                "name": name,
                                "threshold": threshold,
                                **metrics,
                            }
                        )
                if primary_realspace_metrics is None:
                    raise RuntimeError("Primary threshold missing from threshold sweep.")
                metric_tensors.update(primary_realspace_metrics)
            elif has_realspace:
                processed = post_process_realspace_batch(
                    batch["amp"].float(),
                    batch["phi"].float(),
                    pred_amp,
                    pred_phi,
                    support,
                    threshold=args.threshold,
                    executor=unwrap_executor,
                )
                metric_tensors.update(
                    realspace_metric_tensor_dict(
                        *processed,
                        threshold=args.threshold,
                        ssim_window_size=args.ssim_window_size,
                    )
                )
            batch_metrics = materialize_metric_rows(metric_tensors)
            for name, metrics in zip(batch["name"], batch_metrics):
                metrics["inference_ms"] = inference_ms_per_sample
                add_metrics(total, metrics)
                per_sample.append({"name": name, **metrics})
    finally:
        if unwrap_executor is not None:
            unwrap_executor.shutdown(wait=True)
    return per_sample, total, threshold_sweep_rows


def main() -> int:
    """Load inputs, run evaluation, and write the complete artifact bundle."""

    args = parse_args()
    validate_args(args)
    sweep_thresholds = resolve_threshold_sweep(
        args.threshold,
        args.threshold_sweep,
    )
    output_dir = resolve_output_dir(args)
    configure_logging(output_dir, args.log_level)
    device = choose_device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    sample_count = (
        min(args.num_samples, args.limit) if args.limit > 0 else args.num_samples
    )
    spatial_shape = (args.shape, args.shape, args.shape)
    data_dir = Path(args.data_dir).expanduser().resolve()
    data_diff_path = optional_data_path(data_dir, args.data_diff)
    data_real_path = optional_data_path(data_dir, args.data_real)
    if data_diff_path is None:
        raise ValueError("--data-diff is required.")

    LOGGER.info("Output directory: %s", output_dir)
    LOGGER.info("Checkpoint: %s", checkpoint_path)
    LOGGER.info("Diffraction data: %s", data_diff_path)
    LOGGER.info("Real-space data: %s", data_real_path or "disabled")
    LOGGER.info("Resolved sample count: %d", sample_count)

    dataset = AutoPhaseDataset(
        data_diff_path,
        data_real_path,
        num_samples=sample_count,
        shape_diff=spatial_shape,
        shape_real=spatial_shape,
        dtype_diff=args.dtype_diff,
        dtype_real=args.dtype_real,
        shuffle=False,
    )
    if sweep_thresholds and dataset.mmap_real is None:
        raise ValueError("--threshold-sweep requires --data-real ground truth.")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = create_model(args.model_variant, threshold=args.threshold).to(device)
    checkpoint = load_weights(
        model,
        checkpoint_path,
        strict=not args.non_strict_checkpoint,
        map_location=device,
    )
    checkpoint_epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    checkpoint_threshold = (
        checkpoint.get("threshold") if isinstance(checkpoint, dict) else None
    )
    checkpoint_training_args = (
        checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    )
    if checkpoint_threshold is not None and not math.isclose(
        float(checkpoint_threshold),
        args.threshold,
    ):
        LOGGER.warning(
            "Evaluation threshold %.6g differs from checkpoint threshold %.6g.",
            args.threshold,
            float(checkpoint_threshold),
        )
    model.eval()
    free_mask, free_mask_metadata = resolve_free_mask(args, spatial_shape)
    LOGGER.info(
        "Free metrics: source=%s, selected_voxels=%d, fraction=%.6f",
        free_mask_metadata["source"],
        free_mask_metadata["selected_voxels"],
        free_mask_metadata["actual_fraction"],
    )
    postprocess_workers = (
        resolve_postprocess_workers(args.postprocess_workers, args.batch_size)
        if dataset.mmap_real is not None
        else 0
    )
    LOGGER.info(
        "Real-space post-process: tensor_device=%s, phase_unwrap_workers=%d",
        device,
        postprocess_workers,
    )
    if sweep_thresholds:
        LOGGER.info(
            "Support threshold sweep: %s | primary=%.6g",
            ", ".join(format_number(value) for value in sweep_thresholds),
            args.threshold,
        )

    warm_up_model(model, loader, device, args.warmup_batches)
    evaluation_started = time.perf_counter()
    per_sample, total, threshold_sweep_rows = evaluate(
        args,
        model,
        loader,
        dataset,
        device,
        free_mask,
        postprocess_workers,
    )
    evaluation_wall_seconds = time.perf_counter() - evaluation_started
    if not per_sample:
        raise RuntimeError("Evaluation produced no samples.")

    threshold_sweep = (
        summarize_threshold_sweep(threshold_sweep_rows)
        if threshold_sweep_rows
        else {
            "enabled": False,
            "selection_scope": "disabled",
            "primary_threshold_unchanged": True,
            "metrics": list(THRESHOLD_SWEEP_METRICS),
            "summaries": [],
            "best_mean_iou_threshold": None,
            "closest_mean_volume_ratio_threshold": None,
        }
    )

    statistics = metric_statistics(per_sample)
    mean_metrics = {key: values["mean"] for key, values in statistics.items()}
    inference_seconds = sum(float(row["inference_ms"]) / 1000.0 for row in per_sample)
    mean_inference_ms = statistics["inference_ms"]["mean"]
    throughput = len(per_sample) / max(inference_seconds, 1e-12)
    timing: dict[str, float] = {
        "evaluation_wall_seconds": evaluation_wall_seconds,
        "model_inference_seconds": inference_seconds,
        "mean_inference_ms_per_sample": mean_inference_ms,
        "throughput_samples_per_second": throughput,
    }
    if args.reference_runtime_ms > 0:
        timing["speedup_vs_reference"] = args.reference_runtime_ms / max(
            mean_inference_ms,
            1e-12,
        )

    paper_coverage = {
        key: {
            **metadata,
            "available": key in statistics,
            "description": METRIC_DESCRIPTIONS.get(key, ""),
        }
        for key, metadata in PAPER_METRICS.items()
    }
    report: dict[str, object] = {
        "schema_version": 3,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint_epoch,
            "model_variant": args.model_variant,
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "num_samples": len(per_sample),
            "batch_size": args.batch_size,
            "support_threshold": args.threshold,
            "support_threshold_sweep": list(sweep_thresholds),
            "ssim_window_size": args.ssim_window_size,
            "realspace_metrics": dataset.mmap_real is not None,
            "realspace_post_process": "official_skimage_unwrap_batched_torch",
            "postprocess_tensor_device": str(device),
            "postprocess_workers": postprocess_workers,
        },
        "checkpoint_metadata": {
            "epoch": checkpoint_epoch,
            "threshold": checkpoint_threshold,
            "training_args": checkpoint_training_args,
        },
        "configuration": vars(args),
        "data": {
            "diffraction": str(data_diff_path),
            "realspace": str(data_real_path) if data_real_path is not None else None,
            "shape": list(spatial_shape),
        },
        "paper_metric_coverage": paper_coverage,
        "free_mask": free_mask_metadata,
        "timing": timing,
        "fixed_metric_groups": fixed_metric_groups(mean_metrics),
        "fixed_metric_descriptions": FIXED_METRIC_DESCRIPTIONS,
        "mean_metric_groups": group_metrics(mean_metrics),
        "metric_descriptions": METRIC_DESCRIPTIONS,
        "metric_statistics": statistics,
        "threshold_sweep": threshold_sweep,
        "sum": total,
        "mean": mean_metrics,
        "per_sample": per_sample,
        "notes": {
            "diffraction_storage": (
                "Dataset tensors store abs(FFT), so paper Eq. (1) is modulus MAE and "
                "intensity is modulus squared."
            ),
            "ssim": (
                "Both target and predicted real-space amplitude/phase use the official TF2 "
                "post_process before metrics. real_amp_ssim is local-window 3D SSIM with "
                "normalized amplitude data_range=1; real_amp_global_ssim is retained as a "
                "lightweight diagnostic."
            ),
            "phase": (
                "Official post_process unwraps phase, subtracts its support mean, and centers "
                "the object. Metric phase differences are then wrapped to [-pi, pi] and "
                "evaluated on the post-processed true support. The exact skimage phase "
                "unwrap remains on CPU; masking, mean removal, center-of-mass shifts, and "
                "metric tensors are batched on the selected torch device."
            ),
            "scaling": (
                "No input normalization or prediction scale alignment is performed by this "
                "evaluator. Reciprocal-space metrics use the raw model pred_diff output."
            ),
            "timing": (
                "CUDA events time each forward pass. Per-sample values divide batch latency "
                "evenly across the batch."
            ),
            "threshold_calibration": (
                "Threshold-sweep selections are validation diagnostics computed from the "
                "raw pre-support amplitude. The primary threshold remains unchanged."
            ),
        },
    }

    json_path = output_dir / "evaluation_results.json"
    csv_path = output_dir / "evaluation_samples.csv"
    markdown_path = output_dir / "evaluation_summary.md"
    write_json(json_path, report)
    write_sample_csv(csv_path, per_sample)
    if threshold_sweep["enabled"]:
        write_threshold_sweep_summary_csv(
            output_dir / "evaluation_threshold_sweep.csv",
            threshold_sweep,
        )
        write_sample_csv(
            output_dir / "evaluation_threshold_sweep_samples.csv",
            threshold_sweep_rows,
        )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")

    LOGGER.info(
        "\n%s",
        format_fixed_metric_groups(mean_metrics, title="Evaluation mean metrics"),
    )
    LOGGER.info("Wrote JSON report: %s", json_path)
    LOGGER.info("Wrote sample CSV: %s", csv_path)
    LOGGER.info("Wrote Markdown summary: %s", markdown_path)
    if threshold_sweep["enabled"]:
        LOGGER.info(
            "Threshold sweep | best mean IoU at %.6g | volume ratio closest to 1 at %.6g",
            threshold_sweep["best_mean_iou_threshold"],
            threshold_sweep["closest_mean_volume_ratio_threshold"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
