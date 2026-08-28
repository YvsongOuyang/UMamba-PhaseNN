"""Evaluate the official TensorFlow model on generated paper-style samples."""

from __future__ import annotations

import argparse
import csv
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
from .visualization import save_slice_overview, save_volume_overview


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
    parser.add_argument("--iou-tolerance", type=float, default=1e-3)
    parser.add_argument("--visualize-samples", type=int, default=3)
    parser.add_argument("--reuse-predictions", action="store_true")
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
    if args.iou_tolerance < 0.0:
        parser.error("--iou-tolerance cannot be negative.")
    if any(not np.isfinite(value) or not 0.0 < value < 1.0 for value in args.thresholds):
        parser.error("Every support threshold must be finite and lie in (0, 1).")
    args.thresholds = tuple(sorted(set(float(value) for value in args.thresholds)))
    return args


def configure_logging(output_dir: Path, level: str) -> None:
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
            output_dir / "evaluation.log",
            mode="w",
            encoding="utf-8",
        ),
    ):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def load_sample(path: Path) -> dict[str, Any]:
    with np.load(path) as stored:
        intensity = np.asarray(stored["I"], dtype=np.float32)
        target_phase = np.asarray(stored["phi"], dtype=np.float32)
        target_object = (
            np.asarray(stored["object"], dtype=np.complex64)
            if "object" in stored.files
            else reconstruct_object(intensity, target_phase)
        )
        target_support = (
            np.asarray(stored["support"], dtype=bool)
            if "support" in stored.files
            else np.abs(target_object) > 0.5 * max(float(np.abs(target_object).max()), 1e-12)
        )
        metadata = (
            json.loads(str(stored["metadata_json"]))
            if "metadata_json" in stored.files
            else {}
        )
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
            and key not in {"index", "threshold"}
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


def run_tensorflow_predictions(
    sample_paths: list[Path],
    model_path: Path,
    cache_dir: Path,
    batch_size: int,
    reuse: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    prediction_path = cache_dir / "predicted_reciprocal_phase.npy"
    manifest_path = cache_dir / "prediction_manifest.json"
    if reuse:
        if not prediction_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError("--reuse-predictions requires prediction and manifest files.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_names = [path.name for path in sample_paths]
        if manifest.get("sample_names") != expected_names:
            raise ValueError("Cached predictions do not match the selected sample files.")
        return np.load(prediction_path, mmap_mode="r"), manifest

    import tensorflow as tf

    cache_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Loading official TensorFlow model: %s", model_path)
    model = tf.keras.models.load_model(model_path, compile=False)
    first = load_sample(sample_paths[0])
    shape = first["intensity"].shape
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
    return "\n".join(
        [
            "# Official TensorFlow Model on Reproduced Simulations",
            "",
            "## Protocol",
            "",
            f"- Samples: {report['num_samples']} ({report['calibration_samples']} calibration, "
            f"{report['evaluation_samples']} held-out evaluation)",
            f"- Model parameters: {report['model_metadata']['parameter_count']:,}",
            f"- Selected support threshold: `{selected}`",
            f"- Selection rule: calibration IoU within {report['iou_tolerance']:.3g} "
            "of the maximum, then support-volume ratio closest to one.",
            "- Target support: exact support saved by the simulation generator.",
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
            "- `evaluation.log`: TensorFlow inference and evaluation progress.",
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    visualization_dir = Path(args.visualization_dir).expanduser().resolve()
    configure_logging(output_dir, args.log_level)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    sample_paths = sorted(dataset_dir.glob("sample_*.npz"))
    if args.num_samples:
        sample_paths = sample_paths[: args.num_samples]
    if len(sample_paths) < 2:
        raise ValueError("At least two generated samples are required.")
    calibration_count = min(
        len(sample_paths) - 1,
        max(1, int(round(len(sample_paths) * args.calibration_fraction))),
    )
    LOGGER.info(
        "Dataset: %s | samples=%d | calibration/evaluation=%d/%d",
        dataset_dir,
        len(sample_paths),
        calibration_count,
        len(sample_paths) - calibration_count,
    )
    LOGGER.info("Thresholds: %s", ", ".join(f"{value:g}" for value in args.thresholds))

    predictions, model_metadata = run_tensorflow_predictions(
        sample_paths,
        model_path,
        cache_dir,
        args.batch_size,
        args.reuse_predictions,
    )
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
        shape_type = metadata.get("shape_type", "unknown")
        phase_type = metadata.get("phase_type", "unknown")
        sample_state[index] = {
            **sample,
            "selected_phase": selected_phase,
            "geometry_aligned": geometry_aligned,
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
        LOGGER.info("Evaluated sample %d/%d: %s", index + 1, len(sample_paths), sample_path.name)

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

    visualization_records = []
    if args.visualize_samples:
        ranked = sorted(final_rows, key=lambda row: row["support_iou"])
        positions = np.linspace(
            0,
            len(ranked) - 1,
            min(args.visualize_samples, len(ranked)),
            dtype=int,
        )
        visualization_dir.mkdir(parents=True, exist_ok=True)
        for pattern in (
            "sample_*_representative_*_2d.png",
            "sample_*_representative_*_3d.png",
        ):
            for stale_path in visualization_dir.glob(pattern):
                stale_path.unlink()
        for rank, position in enumerate(positions):
            row = ranked[int(position)]
            index = int(row["index"])
            state = sample_state[index]
            aligned, _ = align_global_phase(
                state["geometry_aligned"],
                state["target_object"],
                state["target_support"],
                selected_threshold,
            )
            stem = f"sample_{index:05d}_representative_{rank + 1}"
            slice_path = save_slice_overview(
                intensity=state["intensity"],
                target_reciprocal_phase=state["target_phase"],
                predicted_reciprocal_phase=state["selected_phase"],
                target_object=state["target_object"],
                predicted_object=aligned,
                destination=visualization_dir / f"{stem}_2d.png",
                support_threshold=selected_threshold,
            )
            volume_path = save_volume_overview(
                intensity=state["intensity"],
                target_object=state["target_object"],
                predicted_object=aligned,
                destination=visualization_dir / f"{stem}_3d.png",
                support_threshold=selected_threshold,
            )
            visualization_records.append(
                {
                    "index": index,
                    "name": row["name"],
                    "support_iou": row["support_iou"],
                    "slice_overview": str(slice_path),
                    "volume_overview": str(volume_path),
                }
            )

    evaluation_statistics = metric_statistics(final_rows)
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "dataset_manifest": (
            json.loads((dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
            if (dataset_dir / "dataset_manifest.json").is_file()
            else None
        ),
        "num_samples": len(sample_paths),
        "calibration_samples": calibration_count,
        "evaluation_samples": len(final_rows),
        "split_rule": "sorted sample filenames; first fraction calibrates threshold",
        "threshold_selection_metric": (
            "calibration mean support IoU within tolerance of maximum, then "
            "support volume ratio closest to one"
        ),
        "iou_tolerance": args.iou_tolerance,
        "thresholds": list(args.thresholds),
        "selected_threshold": selected_threshold,
        "target_support_definition": "exact boolean support saved by simulation generator",
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
