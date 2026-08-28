"""Run simulated diffraction through the official or converted paper model."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from .visualization import save_slice_overview, save_volume_overview


LOGGER = logging.getLogger("high_strain.paper_inference")
PROJECT_DIR = Path(__file__).resolve().parents[1]


def _minmax(value: np.ndarray) -> np.ndarray:
    minimum = float(value.min())
    span = float(value.max()) - minimum
    if span <= np.finfo(np.float32).eps:
        return np.zeros_like(value, dtype=np.float32)
    return ((value - minimum) / span).astype(np.float32)


def prepare_model_input(intensity: np.ndarray) -> np.ndarray:
    """Apply the paper's log transform and min-max normalization to NDHWC."""

    normalized = _minmax(np.log1p(intensity.astype(np.float32)))
    return normalized[None, ..., None]


def _weighted_circular_average(
    target: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
) -> float:
    global_shift = float(np.mean(target - prediction))
    normalized_weights = weights / max(float(np.sum(weights)), 1e-12)
    correlation = np.sum(
        normalized_weights * np.exp(1.0j * (target - prediction - global_shift))
    )
    return float(1.0 - np.abs(correlation))


def _roll_without_wrap(array: np.ndarray, shift: tuple[int, int, int]) -> np.ndarray:
    shifted = np.roll(array, shift=shift, axis=(0, 1, 2))
    for axis, amount in enumerate(shift):
        if amount == 0:
            continue
        selection = [slice(None)] * array.ndim
        selection[axis] = slice(0, amount) if amount > 0 else slice(amount, None)
        shifted[tuple(selection)] = 0
    return shifted


def _center_of_mass(amplitude: np.ndarray) -> np.ndarray:
    total = float(amplitude.sum())
    if total <= np.finfo(np.float32).eps:
        return np.asarray(amplitude.shape, dtype=np.float64) / 2.0
    coordinates = np.indices(amplitude.shape, dtype=np.float64)
    return np.asarray(
        [float(np.sum(coordinates[axis] * amplitude) / total) for axis in range(3)]
    )


def reconstruct_object(
    intensity: np.ndarray,
    reciprocal_phase: np.ndarray,
) -> np.ndarray:
    """Combine measured modulus and phase, then apply the inverse 3D FFT."""

    spectrum = np.sqrt(np.clip(intensity, 0.0, None)) * np.exp(
        1.0j * reciprocal_phase
    )
    return np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(spectrum))
    ).astype(np.complex64)


def _align_reconstruction(
    prediction: np.ndarray,
    target: np.ndarray,
    support_threshold: float,
) -> tuple[np.ndarray, tuple[int, int, int], float, float]:
    target_amplitude = np.abs(target)
    predicted_amplitude = np.abs(prediction)
    target_center = _center_of_mass(target_amplitude)
    prediction_center = _center_of_mass(predicted_amplitude)
    shift = tuple(int(value) for value in np.rint(target_center - prediction_center))
    aligned = _roll_without_wrap(prediction, shift)

    predicted_amplitude = np.abs(aligned)
    amplitude_scale = float(
        np.sum(target_amplitude * predicted_amplitude)
        / max(float(np.sum(np.square(predicted_amplitude))), 1e-12)
    )
    aligned = aligned * amplitude_scale
    target_support = target_amplitude > support_threshold * max(
        float(target_amplitude.max()), 1e-12
    )
    predicted_support = np.abs(aligned) > support_threshold * max(
        float(np.abs(aligned).max()), 1e-12
    )
    intersection = target_support & predicted_support
    if np.any(intersection):
        phase_offset = float(
            np.angle(np.sum(target[intersection] * np.conj(aligned[intersection])))
        )
        aligned = aligned * np.exp(1.0j * phase_offset)
    else:
        phase_offset = 0.0
    return aligned.astype(np.complex64), shift, amplitude_scale, phase_offset


def _realspace_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    support_threshold: float,
) -> dict[str, float]:
    target_amplitude = np.abs(target)
    predicted_amplitude = np.abs(prediction)
    target_support = target_amplitude > support_threshold * max(
        float(target_amplitude.max()), 1e-12
    )
    predicted_support = predicted_amplitude > support_threshold * max(
        float(predicted_amplitude.max()), 1e-12
    )
    intersection = target_support & predicted_support
    union = target_support | predicted_support
    amplitude_difference = target_amplitude - predicted_amplitude
    amplitude_nrmse = float(
        np.sqrt(np.mean(np.square(amplitude_difference)))
        / max(float(target_amplitude.max() - target_amplitude.min()), 1e-12)
    )
    phase_mae = float("nan")
    if np.any(intersection):
        phase_error = np.angle(
            np.exp(1.0j * (np.angle(target) - np.angle(prediction)))
        )
        phase_mae = float(np.mean(np.abs(phase_error[intersection])))
    return {
        "amplitude_nrmse": amplitude_nrmse,
        "amplitude_mae": float(np.mean(np.abs(amplitude_difference))),
        "wrapped_phase_mae_rad": phase_mae,
        "support_iou": float(np.count_nonzero(intersection))
        / max(float(np.count_nonzero(union)), 1.0),
        "support_volume_ratio": float(np.count_nonzero(predicted_support))
        / max(float(np.count_nonzero(target_support)), 1.0),
    }


def _run_tensorflow(
    model_path: Path,
    model_input: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path, compile=False)
    started = time.perf_counter()
    prediction = model(model_input, training=False).numpy()
    elapsed = time.perf_counter() - started
    metadata = {
        "backend": "tensorflow",
        "tensorflow_version": tf.__version__,
        "parameter_count": int(model.count_params()),
        "trainable_parameter_count": int(
            sum(np.prod(variable.shape) for variable in model.trainable_weights)
        ),
        "non_trainable_parameter_count": int(
            sum(np.prod(variable.shape) for variable in model.non_trainable_weights)
        ),
        "inference_seconds": elapsed,
        "input_shape": list(model_input.shape),
        "output_shape": list(prediction.shape),
    }
    return prediction[0, ..., 0].astype(np.float32), metadata


def _run_pytorch(
    checkpoint_path: Path,
    model_input: np.ndarray,
    device_name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    from pytorch_autophasenn.model import (
        HighStrainPhaseUNet,
        count_parameters,
        infer_model_variant,
    )

    device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    variant = infer_model_variant(state_dict)
    model = HighStrainPhaseUNet(model_variant=variant).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    tensor = torch.from_numpy(model_input).permute(0, 4, 1, 2, 3).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        prediction = model(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    output = prediction.cpu().permute(0, 2, 3, 4, 1).numpy()
    metadata = {
        "backend": "pytorch",
        "torch_version": torch.__version__,
        "model_variant": variant,
        "parameter_count": count_parameters(model),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "non_trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if not parameter.requires_grad
        ),
        "inference_seconds": elapsed,
        "device": str(device),
        "input_shape": list(model_input.shape),
        "output_shape": list(output.shape),
    }
    return output[0, ..., 0].astype(np.float32), metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", required=True, help="Generated sample NPZ.")
    parser.add_argument("--backend", choices=("tensorflow", "pytorch"), default="tensorflow")
    parser.add_argument(
        "--model",
        default=str(PROJECT_DIR / "artifacts" / "models" / "model_paper.h5"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            PROJECT_DIR / "artifacts" / "simulation" / "official_model"
        ),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--support-threshold", type=float, default=0.1)
    parser.add_argument("--no-visualizations", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sample_path = Path(args.sample).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not sample_path.is_file():
        raise FileNotFoundError(f"Sample not found: {sample_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(sample_path) as sample:
        intensity = np.asarray(sample["I"], dtype=np.float32)
        target_phase = np.asarray(sample["phi"], dtype=np.float32)
        target_object = (
            np.asarray(sample["object"], dtype=np.complex64)
            if "object" in sample.files
            else reconstruct_object(intensity, target_phase)
        )
        sample_metadata = (
            json.loads(str(sample["metadata_json"]))
            if "metadata_json" in sample.files
            else {}
        )

    model_input = prepare_model_input(intensity)
    center = (intensity.shape[0] // 2,) * 3
    target_phase = target_phase - float(target_phase[center])
    LOGGER.info("Loading %s model: %s", args.backend, model_path)
    if args.backend == "tensorflow":
        raw_prediction, model_metadata = _run_tensorflow(model_path, model_input)
    else:
        raw_prediction, model_metadata = _run_pytorch(
            model_path, model_input, args.device
        )
    LOGGER.info(
        "Inference complete | parameters=%s | time=%.3f s",
        f"{model_metadata['parameter_count']:,}",
        model_metadata["inference_seconds"],
    )

    weights = model_input[0, ..., 0]
    direct_wca = _weighted_circular_average(target_phase, raw_prediction, weights)
    inverted_wca = _weighted_circular_average(-target_phase, raw_prediction, weights)
    twin_selected = inverted_wca < direct_wca
    selected_prediction = -raw_prediction if twin_selected else raw_prediction
    reconstruction = reconstruct_object(intensity, selected_prediction)
    reconstruction, center_shift, amplitude_scale, phase_offset = _align_reconstruction(
        reconstruction,
        target_object,
        args.support_threshold,
    )
    metrics = {
        "phase_wca": min(direct_wca, inverted_wca),
        "phase_wca_direct": direct_wca,
        "phase_wca_inverted": inverted_wca,
        "twin_flip_selected": twin_selected,
        **_realspace_metrics(
            target_object,
            reconstruction,
            args.support_threshold,
        ),
    }
    np.save(output_dir / "model_input_ndhwc.npy", model_input)
    np.save(output_dir / "predicted_reciprocal_phase_raw.npy", raw_prediction)
    np.save(output_dir / "predicted_reciprocal_phase_selected.npy", selected_prediction)
    np.savez_compressed(
        output_dir / "reconstruction.npz",
        predicted_object=reconstruction,
        target_object=target_object,
        measured_intensity=intensity,
        target_reciprocal_phase=target_phase,
    )

    visualizations: dict[str, str] = {}
    if not args.no_visualizations:
        slice_path = save_slice_overview(
            intensity=intensity,
            target_reciprocal_phase=target_phase,
            predicted_reciprocal_phase=selected_prediction,
            target_object=target_object,
            predicted_object=reconstruction,
            destination=output_dir / "simulation_reconstruction_2d.png",
        )
        volume_path = save_volume_overview(
            intensity=intensity,
            target_object=target_object,
            predicted_object=reconstruction,
            destination=output_dir / "simulation_reconstruction_3d.png",
            support_threshold=args.support_threshold,
        )
        visualizations = {
            "slice_overview": str(slice_path),
            "volume_overview": str(volume_path),
        }

    report = {
        "sample": str(sample_path),
        "model": str(model_path),
        "model_file_bytes": model_path.stat().st_size,
        "preprocessing": "log1p intensity followed by per-volume min-max normalization",
        "reconstruction": (
            "sqrt(measured intensity) * exp(i * predicted phase), then centered "
            "inverse FFT"
        ),
        "support_threshold": args.support_threshold,
        "ambiguity_handling": (
            "Ground-truth reciprocal phase selects direct or conjugate/twin-equivalent sign."
        ),
        "model_metadata": model_metadata,
        "alignment": {
            "center_shift_voxels": list(center_shift),
            "amplitude_scale": amplitude_scale,
            "global_realspace_phase_offset_rad": phase_offset,
        },
        "metrics": metrics,
        "sample_metadata": sample_metadata,
        "outputs": visualizations,
    }
    report_path = output_dir / "inference_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOGGER.info("Saved inference report: %s", report_path)
    LOGGER.info("Metrics: %s", json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )
    raise SystemExit(main())
