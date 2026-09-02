"""Evaluate TensorFlow or PyTorch phase models on generated paper-style samples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .run_paper_model import (
    _center_of_mass,
    _roll_without_wrap,
    _weighted_circular_average,
    prepare_model_input,
    reconstruct_object,
)
from .visualization import (
    save_amplitude_volume_comparison,
    save_phase_volume_comparison,
    save_slice_overview,
)
from .sample_io import load_reciprocal_phase


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_DIR / "artifacts" / "models" / "model_paper.h5"
DEFAULT_DATASET = PROJECT_DIR / "artifacts" / "simulation" / "paper_evaluation"
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "artifacts"
    / "evaluations"
    / "simulation_tensorflow"
    / "official_published"
)
DEFAULT_CACHE = (
    PROJECT_DIR
    / "artifacts"
    / "simulation"
    / "tensorflow_prediction_cache"
    / "official_published"
)
DEFAULT_VISUALIZATIONS = (
    PROJECT_DIR
    / "artifacts"
    / "visualizations"
    / "simulation_tensorflow"
    / "official_published"
)
DEFAULT_THRESHOLDS = (
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.075,
    0.1,
    0.125,
    0.15,
    0.175,
    0.2,
    0.225,
    0.25,
    0.275,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.6,
)
LOGGER = logging.getLogger("high_strain.simulation_evaluation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    parser.add_argument(
        "--backend",
        choices=("tensorflow", "pytorch"),
        default="tensorflow",
        help="Model runtime. TensorFlow remains the backward-compatible default.",
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument(
        "--visualization-dir",
        default=str(DEFAULT_VISUALIZATIONS),
    )
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
    )
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument(
        "--calibration-split",
        choices=("train", "val", "test"),
        default=None,
        help="Manifest split used only to select the support threshold.",
    )
    parser.add_argument(
        "--evaluation-split",
        choices=("train", "val", "test"),
        default=None,
        help="Disjoint manifest split used for final reported metrics.",
    )
    parser.add_argument("--iou-tolerance", type=float, default=1e-3)
    parser.add_argument("--visualize-samples", type=int, default=3)
    parser.add_argument("--reuse-predictions", action="store_true")
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help=(
            "Reuse an existing prediction cache and evaluation_results.json to "
            "redraw representative samples without recomputing all metrics."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args()
    if args.num_samples < 0:
        parser.error("--num-samples cannot be negative.")
    if args.batch_size < 1 or args.visualize_samples < 0:
        parser.error("Batch size must be positive and visualizations nonnegative.")
    if not 0.0 < args.calibration_fraction < 1.0:
        parser.error("--calibration-fraction must lie in (0, 1).")
    if (args.calibration_split is None) != (args.evaluation_split is None):
        parser.error("Set --calibration-split and --evaluation-split together.")
    if (
        args.calibration_split == args.evaluation_split
        and args.calibration_split is not None
    ):
        parser.error("Calibration and evaluation manifest splits must differ.")
    if args.calibration_split is not None and args.num_samples:
        parser.error("--num-samples is only available with the legacy fraction split.")
    if args.visualize_only and not args.reuse_predictions:
        parser.error("--visualize-only requires --reuse-predictions.")
    if args.visualize_only and args.visualize_samples < 1:
        parser.error("--visualize-only requires at least one visualization sample.")
    if args.iou_tolerance < 0.0:
        parser.error("--iou-tolerance cannot be negative.")
    if any(not np.isfinite(value) or not 0.0 < value < 1.0 for value in args.thresholds):
        parser.error("Every support threshold must be finite and lie in (0, 1).")
    args.thresholds = tuple(sorted(set(float(value) for value in args.thresholds)))
    return args


def configure_logging(
    output_dir: Path,
    level: str,
    filename: str = "evaluation.log",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.handlers.clear()
    LOGGER.setLevel(getattr(logging, level))
    LOGGER.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            output_dir / filename,
            mode="w",
            encoding="utf-8",
        ),
    ):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def load_sample(path: Path) -> dict[str, Any]:
    with np.load(path) as stored:
        if not {"object", "support"} <= set(stored.files):
            raise ValueError(
                f"{path} lacks clean object/support truth. Regenerate with --save-extras."
            )
        intensity = np.asarray(stored["I"], dtype=np.float32)
        target_phase = load_reciprocal_phase(stored)
        target_object = np.asarray(stored["object"], dtype=np.complex64)
        target_support = np.asarray(stored["support"], dtype=bool)
        metadata = (
            json.loads(str(stored["metadata_json"]))
            if "metadata_json" in stored.files
            else {}
        )
    if intensity.ndim != 3 or any(
        array.shape != intensity.shape for array in (target_phase, target_object, target_support)
    ):
        raise ValueError(f"{path} contains inconsistent 3D sample shapes.")
    if any(not np.all(np.isfinite(array)) for array in (intensity, target_phase, target_object)):
        raise ValueError(f"{path} contains non-finite sample values.")
    if np.any(intensity < 0) or not np.any(intensity > 0) or not np.any(target_support):
        raise ValueError(f"{path} has invalid intensity or an empty support.")
    center = tuple(size // 2 for size in intensity.shape)
    target_phase = target_phase - float(target_phase[center])
    return {
        "intensity": intensity,
        "target_phase": target_phase,
        "target_object": target_object,
        "target_support": target_support,
        "metadata": metadata,
    }


def align_geometry(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int], float]:
    target_amplitude = np.abs(target)
    predicted_amplitude = np.abs(prediction)
    shift = tuple(
        int(value)
        for value in np.rint(
            _center_of_mass(target_amplitude) - _center_of_mass(predicted_amplitude)
        )
    )
    aligned = _roll_without_wrap(prediction, shift)
    aligned_amplitude = np.abs(aligned)
    scale = float(
        np.sum(target_amplitude * aligned_amplitude)
        / max(float(np.sum(np.square(aligned_amplitude))), 1e-12)
    )
    return (aligned * scale).astype(np.complex64), shift, scale


def align_global_phase(
    prediction: np.ndarray,
    target: np.ndarray,
    target_support: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, float]:
    predicted_amplitude = np.abs(prediction)
    predicted_support = predicted_amplitude > threshold * max(
        float(predicted_amplitude.max()), 1e-12
    )
    intersection = target_support & predicted_support
    if not np.any(intersection):
        return prediction, 0.0
    phase_offset = float(
        np.angle(np.sum(target[intersection] * np.conj(prediction[intersection])))
    )
    return (prediction * np.exp(1.0j * phase_offset)).astype(np.complex64), phase_offset


def realspace_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    target_support: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    target_amplitude = np.abs(target)
    predicted_amplitude = np.abs(prediction)
    predicted_support = predicted_amplitude > threshold * max(
        float(predicted_amplitude.max()), 1e-12
    )
    intersection = target_support & predicted_support
    union = target_support | predicted_support
    true_count = float(np.count_nonzero(target_support))
    predicted_count = float(np.count_nonzero(predicted_support))
    intersection_count = float(np.count_nonzero(intersection))
    amplitude_difference = target_amplitude - predicted_amplitude
    phase_mae = float("nan")
    if intersection_count:
        phase_error = np.angle(
            np.exp(1.0j * (np.angle(target) - np.angle(prediction)))
        )
        phase_mae = float(np.mean(np.abs(phase_error[intersection])))
    return {
        "amplitude_nrmse": float(np.sqrt(np.mean(np.square(amplitude_difference))))
        / max(float(target_amplitude.max() - target_amplitude.min()), 1e-12),
        "amplitude_mae": float(np.mean(np.abs(amplitude_difference))),
        "wrapped_phase_mae_rad": phase_mae,
        "support_iou": intersection_count
        / max(float(np.count_nonzero(union)), 1.0),
        "support_dice": 2.0 * intersection_count
        / max(true_count + predicted_count, 1.0),
        "support_precision": intersection_count / max(predicted_count, 1.0),
        "support_recall": intersection_count / max(true_count, 1.0),
        "support_volume_ratio": predicted_count / max(true_count, 1.0),
    }


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {key: float("nan") for key in ("mean", "std", "median", "q05", "q95")}
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q95": float(np.quantile(array, 0.95)),
    }


def metric_statistics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    metric_names = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float, np.integer, np.floating))
            and key not in {"index", "threshold", "particle_seed"}
            and not isinstance(value, (bool, np.bool_))
        }
    )
    return {
        name: distribution([float(row[name]) for row in rows if name in row])
        for name in metric_names
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prediction_identity(
    sample_paths: list[Path],
    model_path: Path,
    backend: str = "tensorflow",
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "preprocessing_version": (
            "log1p_float32_minmax_volume_ndhwc_v1"
            if backend == "tensorflow"
            else "log1p_float32_minmax_volume_ncdhw_v1"
        ),
        "backend": backend,
        "model_sha256": _file_sha256(model_path),
        "samples": [
            {"name": path.name, "sha256": _file_sha256(path)} for path in sample_paths
        ],
    }


def run_tensorflow_predictions(
    sample_paths: list[Path],
    model_path: Path,
    cache_dir: Path,
    batch_size: int,
    reuse: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    prediction_path = cache_dir / "predicted_reciprocal_phase.npy"
    manifest_path = cache_dir / "prediction_manifest.json"
    identity = _prediction_identity(sample_paths, model_path, "tensorflow")
    shape = load_sample(sample_paths[0])["intensity"].shape
    if reuse:
        if not prediction_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError("--reuse-predictions requires prediction and manifest files.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("content_identity") != identity:
            raise ValueError("Cached model/data/preprocessing identity differs; rerun inference.")
        if manifest.get("prediction_sha256") != _file_sha256(prediction_path):
            raise ValueError("Cached prediction content is incomplete or changed.")
        predictions = np.load(prediction_path, mmap_mode="r")
        if predictions.shape != (len(sample_paths),) + shape or predictions.dtype != np.float32:
            raise ValueError("Cached prediction shape/dtype differs from selected samples.")
        if any(not np.all(np.isfinite(prediction)) for prediction in predictions):
            raise ValueError("Cached predictions contain non-finite values.")
        return predictions, manifest

    import tensorflow as tf

    cache_dir.mkdir(parents=True, exist_ok=True)
    # Invalidate a prior completion record before overwriting its prediction array.
    manifest_path.unlink(missing_ok=True)
    LOGGER.info("Loading official TensorFlow model: %s", model_path)
    model = tf.keras.models.load_model(model_path, compile=False)
    predictions = np.lib.format.open_memmap(
        prediction_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(sample_paths),) + shape,
    )
    inference_seconds = 0.0
    started = time.perf_counter()
    for start in range(0, len(sample_paths), batch_size):
        stop = min(start + batch_size, len(sample_paths))
        batch = np.concatenate(
            [
                prepare_model_input(load_sample(path)["intensity"])
                for path in sample_paths[start:stop]
            ],
            axis=0,
        )
        inference_started = time.perf_counter()
        output = model(batch, training=False).numpy()
        inference_seconds += time.perf_counter() - inference_started
        if output.shape != (stop - start,) + shape + (1,):
            raise ValueError(f"Unexpected TensorFlow output shape: {output.shape}")
        if not np.all(np.isfinite(output)):
            raise ValueError("TensorFlow returned non-finite predictions.")
        predictions[start:stop] = output[..., 0].astype(np.float32)
        predictions.flush()
        elapsed = time.perf_counter() - started
        rate = stop / max(elapsed, 1e-12)
        remaining = (len(sample_paths) - stop) / max(rate, 1e-12)
        LOGGER.info(
            "TensorFlow inference %d/%d | %.3f samples/s | ETA %.1f s",
            stop,
            len(sample_paths),
            rate,
            remaining,
        )
    manifest = {
        "backend": "tensorflow",
        "tensorflow_version": tf.__version__,
        "python_version": platform.python_version(),
        "model": str(model_path),
        "model_file_bytes": model_path.stat().st_size,
        "content_identity": identity,
        "prediction_sha256": _file_sha256(prediction_path),
        "parameter_count": int(model.count_params()),
        "sample_names": [path.name for path in sample_paths],
        "num_samples": len(sample_paths),
        "shape": list(shape),
        "prediction_path": str(prediction_path),
        "input_preprocessing": "log1p intensity + per-volume min-max, NDHWC",
        "inference_seconds": inference_seconds,
        "mean_inference_ms_per_sample": 1000.0 * inference_seconds / len(sample_paths),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return np.load(prediction_path, mmap_mode="r"), manifest


def run_pytorch_predictions(
    sample_paths: list[Path],
    model_path: Path,
    cache_dir: Path,
    batch_size: int,
    reuse: bool,
    device_name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run a converted PyTorch phase model with the same numerical preprocessing."""

    import torch

    from pytorch_autophasenn.model import (
        HighStrainPhaseUNet,
        count_parameters,
        infer_model_variant,
    )

    prediction_path = cache_dir / "predicted_reciprocal_phase.npy"
    manifest_path = cache_dir / "prediction_manifest.json"
    identity = _prediction_identity(sample_paths, model_path, "pytorch")
    shape = load_sample(sample_paths[0])["intensity"].shape
    if reuse:
        if not prediction_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError("--reuse-predictions requires prediction and manifest files.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("content_identity") != identity:
            raise ValueError("Cached model/data/preprocessing identity differs; rerun inference.")
        if manifest.get("prediction_sha256") != _file_sha256(prediction_path):
            raise ValueError("Cached prediction content is incomplete or changed.")
        predictions = np.load(prediction_path, mmap_mode="r")
        if predictions.shape != (len(sample_paths),) + shape or predictions.dtype != np.float32:
            raise ValueError("Cached prediction shape/dtype differs from selected samples.")
        if any(not np.all(np.isfinite(prediction)) for prediction in predictions):
            raise ValueError("Cached predictions contain non-finite values.")
        return predictions, manifest

    device = torch.device("cuda" if device_name == "gpu" else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device gpu requested but PyTorch CUDA is unavailable.")
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model_variant = infer_model_variant(state_dict)
    model = HighStrainPhaseUNet(model_variant=model_variant).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.unlink(missing_ok=True)
    predictions = np.lib.format.open_memmap(
        prediction_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(sample_paths),) + shape,
    )
    LOGGER.info(
        "Loading PyTorch model: %s | variant=%s | device=%s",
        model_path,
        model_variant,
        device,
    )
    inference_seconds = 0.0
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(sample_paths), batch_size):
            stop = min(start + batch_size, len(sample_paths))
            batch_ndhwc = np.concatenate(
                [
                    prepare_model_input(load_sample(path)["intensity"])
                    for path in sample_paths[start:stop]
                ],
                axis=0,
            )
            batch_ncdhw = np.moveaxis(batch_ndhwc, -1, 1)
            model_input = torch.from_numpy(
                np.ascontiguousarray(batch_ncdhw)
            ).to(device=device, dtype=torch.float32)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_started = time.perf_counter()
            output = model(model_input)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - inference_started
            if output.shape != (stop - start, 1) + shape:
                raise ValueError(f"Unexpected PyTorch output shape: {tuple(output.shape)}")
            output_numpy = output[:, 0].detach().cpu().numpy().astype(np.float32)
            if not np.all(np.isfinite(output_numpy)):
                raise ValueError("PyTorch returned non-finite predictions.")
            predictions[start:stop] = output_numpy
            predictions.flush()
            elapsed = time.perf_counter() - started
            rate = stop / max(elapsed, 1e-12)
            remaining = (len(sample_paths) - stop) / max(rate, 1e-12)
            LOGGER.info(
                "PyTorch inference %d/%d | %.3f samples/s | ETA %.1f s",
                stop,
                len(sample_paths),
                rate,
                remaining,
            )

    checkpoint_metadata = checkpoint if isinstance(checkpoint, dict) else {}
    manifest = {
        "backend": "pytorch",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "model": str(model_path),
        "model_file_bytes": model_path.stat().st_size,
        "content_identity": identity,
        "prediction_sha256": _file_sha256(prediction_path),
        "parameter_count": count_parameters(model),
        "model_variant": model_variant,
        "checkpoint_epoch": checkpoint_metadata.get("epoch"),
        "checkpoint_project_version": checkpoint_metadata.get("project_version"),
        "checkpoint_git_commit": checkpoint_metadata.get("git_commit"),
        "sample_names": [path.name for path in sample_paths],
        "num_samples": len(sample_paths),
        "shape": list(shape),
        "prediction_path": str(prediction_path),
        "input_preprocessing": "log1p intensity + per-volume min-max, NCDHW",
        "inference_seconds": inference_seconds,
        "mean_inference_ms_per_sample": 1000.0 * inference_seconds / len(sample_paths),
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return np.load(prediction_path, mmap_mode="r"), manifest


def summarize_thresholds(
    sweep_rows: list[dict[str, Any]],
    thresholds: tuple[float, ...],
) -> list[dict[str, Any]]:
    summaries = []
    for threshold in thresholds:
        row: dict[str, Any] = {"threshold": threshold}
        for split in ("calibration", "evaluation"):
            selected = [
                item
                for item in sweep_rows
                if item["split"] == split
                and np.isclose(float(item["threshold"]), threshold)
            ]
            stats = metric_statistics(selected)
            for metric in (
                "support_iou",
                "support_dice",
                "support_precision",
                "support_recall",
                "support_volume_ratio",
                "wrapped_phase_mae_rad",
            ):
                row[f"{split}_{metric}_mean"] = stats[metric]["mean"]
        summaries.append(row)
    return summaries


def choose_threshold(
    summaries: list[dict[str, Any]],
    iou_tolerance: float = 1e-3,
) -> float:
    best_iou = max(row["calibration_support_iou_mean"] for row in summaries)
    near_optimal = [
        row
        for row in summaries
        if row["calibration_support_iou_mean"] >= best_iou - iou_tolerance
    ]
    best = min(
        near_optimal,
        key=lambda row: (
            abs(row["calibration_support_volume_ratio_mean"] - 1.0),
            -row["calibration_support_iou_mean"],
            -row["calibration_support_dice_mean"],
            row["threshold"],
        ),
    )
    return float(best["threshold"])


def category_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return {
        str(value): {
            "num_samples": len(selected),
            "statistics": metric_statistics(selected),
        }
        for value in sorted({row[key] for row in rows})
        for selected in [[row for row in rows if row[key] == value]]
    }


def render_markdown(report: dict[str, Any]) -> str:
    selected = report["selected_threshold"]
    stats = report["evaluation_statistics"]
    dataset_name = (report.get("dataset_manifest") or {}).get("dataset_name", "Reproduced Simulations")
    backend = str(report["model_metadata"]["backend"]).replace("_", " ").title()
    return "\n".join(
        [
            f"# {backend} Model on {dataset_name}",
            "",
            "## Protocol",
            "",
            f"- Samples: {report['num_samples']} ({report['calibration_samples']} calibration, "
            f"{report['evaluation_samples']} held-out evaluation)",
            f"- Model parameters: {report['model_metadata']['parameter_count']:,}",
            f"- Selected support threshold: `{selected}`",
            f"- Selection rule: calibration IoU within {report['iou_tolerance']:.3g} "
            "of the maximum, then support-volume ratio closest to one.",
            f"- Target support: {report['target_support_definition']}.",
            "",
            "## Held-out Evaluation",
            "",
            "| Metric | Mean | Std | Median | 5% | 95% |",
            "|---|---:|---:|---:|---:|---:|",
            *[
                f"| {metric} | {values['mean']:.6g} | {values['std']:.6g} | "
                f"{values['median']:.6g} | {values['q05']:.6g} | {values['q95']:.6g} |"
                for metric, values in stats.items()
            ],
            "",
            "## Files",
            "",
            "- `evaluation_results.json`: full provenance, statistics, and per-sample rows.",
            "- `evaluation_samples.csv`: held-out per-sample metrics at the selected threshold.",
            "- `threshold_sweep.csv`: calibration and evaluation means for every threshold.",
            f"- `evaluation.log`: {backend} inference and evaluation progress.",
            "",
        ]
    )


def select_evaluation_samples(
    dataset_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[Path], int, dict[str, Any] | None, str]:
    """Select either explicit manifest splits or the legacy filename fraction."""

    manifest_path = dataset_dir / "dataset_manifest.json"
    dataset_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else None
    )
    if args.calibration_split is None:
        sample_paths = sorted(dataset_dir.glob("sample_*.npz"))
        if args.num_samples:
            sample_paths = sample_paths[: args.num_samples]
        if len(sample_paths) < 2:
            raise ValueError("At least two generated samples are required.")
        calibration_count = min(
            len(sample_paths) - 1,
            max(1, int(round(len(sample_paths) * args.calibration_fraction))),
        )
        return (
            sample_paths,
            calibration_count,
            dataset_manifest,
            "sorted sample filenames; first fraction calibrates threshold",
        )

    if dataset_manifest is None:
        raise FileNotFoundError("Manifest split selection requires dataset_manifest.json.")
    if dataset_manifest.get("split_unit") != "particle":
        raise ValueError("Manifest splits must be particle-disjoint.")
    declared_splits = dataset_manifest.get("splits", {})
    for split in (args.calibration_split, args.evaluation_split):
        if split not in declared_splits:
            raise ValueError(f"Manifest does not contain split {split!r}.")

    selected: dict[str, list[Path]] = {
        args.calibration_split: [],
        args.evaluation_split: [],
    }
    particles: dict[str, set[tuple[int, str]]] = {
        args.calibration_split: set(),
        args.evaluation_split: set(),
    }
    seen: set[str] = set()
    for record in dataset_manifest.get("samples", []):
        split = record.get("split")
        if split not in selected:
            continue
        filename = str(record["filename"])
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts or filename in seen:
            raise ValueError(f"Unsafe or duplicate manifest sample: {filename}")
        seen.add(filename)
        path = (dataset_dir / relative).resolve()
        if dataset_dir not in path.parents or not path.is_file():
            raise FileNotFoundError(f"Manifest sample is missing: {path}")
        metadata = record.get("metadata", {})
        particles[split].add((int(metadata["particle_seed"]), str(metadata["shape"])))
        selected[split].append(path)

    for split, paths in selected.items():
        if len(paths) != int(declared_splits[split]):
            raise ValueError(
                f"Manifest declares {declared_splits[split]} {split} samples but selected {len(paths)}."
            )
    overlap = particles[args.calibration_split] & particles[args.evaluation_split]
    if overlap:
        raise ValueError("Particle leakage exists between calibration and evaluation splits.")
    calibration_paths = selected[args.calibration_split]
    evaluation_paths = selected[args.evaluation_split]
    if not calibration_paths or not evaluation_paths:
        raise ValueError("Calibration and evaluation manifest splits must both be nonempty.")
    return (
        calibration_paths + evaluation_paths,
        len(calibration_paths),
        dataset_manifest,
        (
            f"manifest particle-disjoint splits: {args.calibration_split} calibrates; "
            f"{args.evaluation_split} reports final metrics"
        ),
    )


def select_visualization_rows(
    rows: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    """Prefer one median-WCA example per shape/phase pair, then fill by IoU."""

    if count <= 0 or not rows:
        return []
    count = min(count, len(rows))
    groups = sorted({str(row.get("shape_phase", "unknown/unknown")) for row in rows})
    selected: list[dict[str, Any]] = []
    selected_indices: set[int] = set()
    if count >= len(groups):
        for group in groups:
            candidates = [row for row in rows if str(row.get("shape_phase")) == group]
            median_wca = float(np.median([float(row["phase_wca"]) for row in candidates]))
            representative = min(
                candidates,
                key=lambda row: (
                    abs(float(row["phase_wca"]) - median_wca),
                    abs(float(row["support_volume_ratio"]) - 1.0),
                    str(row["name"]),
                ),
            )
            selected.append({**representative, "visualization_selection": "pair_median_wca"})
            selected_indices.add(int(representative["index"]))
            if len(selected) == count:
                return selected

    remaining = count - len(selected)
    candidates = sorted(
        (row for row in rows if int(row["index"]) not in selected_indices),
        key=lambda row: float(row["support_iou"]),
    )
    if remaining > 0 and candidates:
        positions = np.linspace(
            0,
            len(candidates) - 1,
            min(remaining, len(candidates)),
            dtype=int,
        )
        for position in positions:
            selected.append(
                {
                    **candidates[int(position)],
                    "visualization_selection": "support_iou_quantile",
                }
            )
    return selected


def _visualization_model_label(model_metadata: dict[str, Any]) -> str:
    backend = str(model_metadata.get("backend", "model"))
    if backend == "pytorch":
        variant = str(model_metadata.get("model_variant", "phase model"))
        return f"PyTorch {variant}"
    if backend == "tensorflow":
        return "Official TensorFlow HighStrain model"
    return backend.replace("_", " ").title()


def render_visualizations(
    *,
    rows: list[dict[str, Any]],
    sample_paths: list[Path],
    predictions: np.ndarray,
    selected_threshold: float,
    visualization_dir: Path,
    model_metadata: dict[str, Any],
    count: int,
) -> list[dict[str, Any]]:
    """Render representative held-out samples without retaining volumes in memory."""

    selected_rows = select_visualization_rows(rows, count)
    if not selected_rows:
        return []
    visualization_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*_representative_*_2d.png", "*_representative_*_3d.png"):
        for stale_path in visualization_dir.glob(pattern):
            stale_path.unlink()

    model_label = _visualization_model_label(model_metadata)
    records: list[dict[str, Any]] = []
    target_objects: list[np.ndarray] = []
    predictions_before_shift: list[np.ndarray] = []
    predictions_after_shift: list[np.ndarray] = []
    target_supports: list[np.ndarray | None] = []
    volume_names: list[str] = []
    for rank, row in enumerate(selected_rows, start=1):
        index = int(row["index"])
        sample = load_sample(sample_paths[index])
        raw_prediction = predictions[index]
        selected_phase = (
            -raw_prediction if bool(row["twin_flip_selected"]) else raw_prediction
        )
        reconstruction = reconstruct_object(sample["intensity"], selected_phase)
        geometry_aligned, _, amplitude_scale = align_geometry(
            reconstruction,
            sample["target_object"],
        )
        prediction_before_shift = (reconstruction * amplitude_scale).astype(
            np.complex64
        )
        aligned, phase_offset = align_global_phase(
            geometry_aligned,
            sample["target_object"],
            sample["target_support"],
            selected_threshold,
        )
        prediction_before_shift = (
            prediction_before_shift * np.exp(1.0j * phase_offset)
        ).astype(np.complex64)
        pair = str(row.get("shape_phase", "unknown/unknown"))
        pair_token = "_".join(
            "".join(character for character in part if character.isalnum() or character == "_")
            or "unknown"
            for part in pair.split("/")
        )
        stem = f"{pair_token}_sample_{index:05d}_representative_{rank}"
        sample_label = (
            f"{pair} | WCA={float(row['phase_wca']):.4f} | "
            f"support IoU={float(row['support_iou']):.4f} | threshold={selected_threshold:g}"
        )
        slice_path = save_slice_overview(
            intensity=sample["intensity"],
            target_reciprocal_phase=sample["target_phase"],
            predicted_reciprocal_phase=selected_phase,
            target_object=sample["target_object"],
            predicted_object=aligned,
            destination=visualization_dir / f"{stem}_2d.png",
            support_threshold=selected_threshold,
            target_support=sample["target_support"],
            model_label=model_label,
            sample_label=sample_label,
        )
        target_objects.append(sample["target_object"])
        predictions_before_shift.append(prediction_before_shift)
        predictions_after_shift.append(aligned)
        target_supports.append(sample["target_support"])
        volume_names.append(
            f"{pair} | sample {index:05d}\n"
            f"WCA={float(row['phase_wca']):.4f} | "
            f"IoU={float(row['support_iou']):.4f}"
        )
        records.append(
            {
                "index": index,
                "name": row["name"],
                "shape_phase": pair,
                "selection": row["visualization_selection"],
                "phase_wca": row["phase_wca"],
                "support_iou": row["support_iou"],
                "slice_overview": str(slice_path),
            }
        )
    amplitude_volume_path = save_amplitude_volume_comparison(
        target_objects=target_objects,
        predicted_objects_before_shift=predictions_before_shift,
        predicted_objects_after_shift=predictions_after_shift,
        target_supports=target_supports,
        names=volume_names,
        destination=visualization_dir / "representative_amplitude_3d.png",
        support_threshold=selected_threshold,
        model_label=model_label,
    )
    phase_volume_path = save_phase_volume_comparison(
        target_objects=target_objects,
        predicted_objects_before_shift=predictions_before_shift,
        predicted_objects_after_shift=predictions_after_shift,
        target_supports=target_supports,
        names=volume_names,
        destination=visualization_dir / "representative_phase_3d.png",
        support_threshold=selected_threshold,
        model_label=model_label,
    )
    for record in records:
        record["volume_overview"] = str(amplitude_volume_path)
        record["amplitude_volume_overview"] = str(amplitude_volume_path)
        record["phase_volume_overview"] = str(phase_volume_path)
    return records


def main() -> int:
    args = parse_args()
    if args.backend == "tensorflow" and args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    visualization_dir = Path(args.visualization_dir).expanduser().resolve()
    configure_logging(
        output_dir,
        args.log_level,
        filename="visualization.log" if args.visualize_only else "evaluation.log",
    )
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    sample_paths, calibration_count, dataset_manifest, split_rule = (
        select_evaluation_samples(dataset_dir, args)
    )
    LOGGER.info(
        "Dataset: %s | samples=%d | calibration/evaluation=%d/%d | %s",
        dataset_dir,
        len(sample_paths),
        calibration_count,
        len(sample_paths) - calibration_count,
        split_rule,
    )
    LOGGER.info("Thresholds: %s", ", ".join(f"{value:g}" for value in args.thresholds))

    if args.backend == "tensorflow":
        predictions, model_metadata = run_tensorflow_predictions(
            sample_paths,
            model_path,
            cache_dir,
            args.batch_size,
            args.reuse_predictions,
        )
    else:
        predictions, model_metadata = run_pytorch_predictions(
            sample_paths,
            model_path,
            cache_dir,
            args.batch_size,
            args.reuse_predictions,
            args.device,
        )
    if args.visualize_only:
        report_path = output_dir / "evaluation_results.json"
        if not report_path.is_file():
            raise FileNotFoundError(
                "--visualize-only requires evaluation_results.json in --output-dir."
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        previous_identity = report.get("model_metadata", {}).get("content_identity")
        if previous_identity != model_metadata.get("content_identity"):
            raise ValueError("Evaluation report and prediction cache identities differ.")
        final_rows = report.get("evaluation_per_sample", [])
        selected_threshold = float(report["selected_threshold"])
        visualization_records = render_visualizations(
            rows=final_rows,
            sample_paths=sample_paths,
            predictions=predictions,
            selected_threshold=selected_threshold,
            visualization_dir=visualization_dir,
            model_metadata=model_metadata,
            count=args.visualize_samples,
        )
        report["visualizations"] = visualization_records
        report["visualizations_updated_at"] = datetime.now(timezone.utc).isoformat()
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        LOGGER.info(
            "Redrew %d representative sample(s): %s",
            len(visualization_records),
            visualization_dir,
        )
        return 0

    sweep_rows: list[dict[str, Any]] = []
    sample_state: dict[int, dict[str, Any]] = {}
    for index, (sample_path, raw_prediction) in enumerate(zip(sample_paths, predictions)):
        sample = load_sample(sample_path)
        model_input = prepare_model_input(sample["intensity"])
        direct_wca = _weighted_circular_average(
            sample["target_phase"], raw_prediction, model_input[0, ..., 0]
        )
        inverted_wca = _weighted_circular_average(
            -sample["target_phase"], raw_prediction, model_input[0, ..., 0]
        )
        twin_selected = inverted_wca < direct_wca
        selected_phase = -raw_prediction if twin_selected else raw_prediction
        reconstruction = reconstruct_object(sample["intensity"], selected_phase)
        geometry_aligned, center_shift, amplitude_scale = align_geometry(
            reconstruction,
            sample["target_object"],
        )
        split = "calibration" if index < calibration_count else "evaluation"
        metadata = sample["metadata"]
        shape_type = metadata.get("shape_type", metadata.get("shape", "unknown"))
        phase_type = metadata.get(
            "phase_type",
            metadata.get("strain_argument", metadata.get("phase_family", "unknown")),
        )
        sample_state[index] = {
            "center_shift": center_shift,
            "amplitude_scale": amplitude_scale,
            "phase_wca": min(direct_wca, inverted_wca),
            "phase_wca_direct": direct_wca,
            "phase_wca_inverted": inverted_wca,
            "twin_flip_selected": twin_selected,
            "split": split,
        }
        for threshold in args.thresholds:
            aligned, phase_offset = align_global_phase(
                geometry_aligned,
                sample["target_object"],
                sample["target_support"],
                threshold,
            )
            sweep_rows.append(
                {
                    "index": index,
                    "name": sample_path.name,
                    "split": split,
                    "shape_type": shape_type,
                    "phase_type": phase_type,
                    "shape_phase": f"{shape_type}/{phase_type}",
                    "particle_seed": metadata.get("particle_seed"),
                    "threshold": threshold,
                    "phase_offset_rad": phase_offset,
                    **realspace_metrics(
                        sample["target_object"],
                        aligned,
                        sample["target_support"],
                        threshold,
                    ),
                }
            )
        if (index + 1) % 50 == 0 or index + 1 == len(sample_paths):
            LOGGER.info(
                "Evaluated sample %d/%d: %s",
                index + 1,
                len(sample_paths),
                sample_path.name,
            )

    threshold_summaries = summarize_thresholds(sweep_rows, args.thresholds)
    selected_threshold = choose_threshold(
        threshold_summaries,
        args.iou_tolerance,
    )
    LOGGER.info(
        "Selected support threshold %.6g from near-optimal calibration IoU "
        "and volume-ratio balance.",
        selected_threshold,
    )
    final_rows: list[dict[str, Any]] = []
    for index in range(calibration_count, len(sample_paths)):
        state = sample_state[index]
        matching = next(
            row
            for row in sweep_rows
            if row["index"] == index
            and np.isclose(float(row["threshold"]), selected_threshold)
        )
        final_rows.append(
            {
                **matching,
                "phase_wca": state["phase_wca"],
                "phase_wca_direct": state["phase_wca_direct"],
                "phase_wca_inverted": state["phase_wca_inverted"],
                "twin_flip_selected": state["twin_flip_selected"],
                "center_shift_voxels": json.dumps(list(state["center_shift"])),
                "amplitude_scale": state["amplitude_scale"],
            }
        )

    visualization_records = render_visualizations(
        rows=final_rows,
        sample_paths=sample_paths,
        predictions=predictions,
        selected_threshold=selected_threshold,
        visualization_dir=visualization_dir,
        model_metadata=model_metadata,
        count=args.visualize_samples,
    )

    evaluation_statistics = metric_statistics(final_rows)
    dataset_manifest_summary = (
        {key: value for key, value in dataset_manifest.items() if key != "samples"}
        if dataset_manifest is not None
        else None
    )
    if dataset_manifest_summary is not None:
        index_filter = dataset_manifest_summary.get("index_filter")
        if isinstance(index_filter, dict):
            dataset_manifest_summary["index_filter"] = {
                key: value
                for key, value in index_filter.items()
                if key != "excluded_filenames"
            }
        dataset_manifest_summary["manifest_sha256"] = _file_sha256(
            dataset_dir / "dataset_manifest.json"
        )
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "dataset_manifest": dataset_manifest_summary,
        "num_samples": len(sample_paths),
        "calibration_samples": calibration_count,
        "evaluation_samples": len(final_rows),
        "split_rule": split_rule,
        "threshold_selection_metric": (
            "calibration mean support IoU within tolerance of maximum, then "
            "support volume ratio closest to one"
        ),
        "iou_tolerance": args.iou_tolerance,
        "thresholds": list(args.thresholds),
        "selected_threshold": selected_threshold,
        "target_support_definition": (dataset_manifest_summary or {}).get(
            "target_support_definition", "exact boolean support saved by simulation generator"
        ),
        "model_metadata": model_metadata,
        "evaluation_statistics": evaluation_statistics,
        "by_shape_type": category_summary(final_rows, "shape_type"),
        "by_phase_type": category_summary(final_rows, "phase_type"),
        "by_shape_phase": category_summary(final_rows, "shape_phase"),
        "threshold_summaries": threshold_summaries,
        "evaluation_per_sample": final_rows,
        "visualizations": visualization_records,
        "configuration": vars(args),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / "evaluation_summary.md").write_text(
        render_markdown(report),
        encoding="utf-8",
    )
    write_csv(output_dir / "evaluation_samples.csv", final_rows)
    write_csv(output_dir / "threshold_sweep.csv", threshold_summaries)
    LOGGER.info(
        "Held-out mean | WCA=%.6g | support IoU=%.6g | Dice=%.6g | phase MAE=%.6g",
        evaluation_statistics["phase_wca"]["mean"],
        evaluation_statistics["support_iou"]["mean"],
        evaluation_statistics["support_dice"]["mean"],
        evaluation_statistics["wrapped_phase_mae_rad"]["mean"],
    )
    LOGGER.info("Results: %s", output_dir)
    LOGGER.info("Visualizations: %s", visualization_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
