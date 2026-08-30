"""Run the official TensorFlow model on the repository's experimental data."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.ndimage import affine_transform, binary_erosion, center_of_mass, label

from simulation.run_paper_model import prepare_model_input, reconstruct_object


LOGGER = logging.getLogger("high_strain.experimental")
PROJECT_DIR = Path(__file__).resolve().parents[1]
OFFICIAL_REPOSITORY = "https://github.com/matteomasto/high_strain_CNN"
OFFICIAL_SAMPLES = {
    "data1.npy": {
        "paper_name": "Particle 1",
        "material": "Pt nanoparticle on YSZ",
        "physical_size_nm": [250, 400, 410],
        "paper_original_shape": [118, 180, 230],
    },
    "data2.npy": {
        "paper_name": "Particle 2",
        "material": "dewetted Pt/Pd bilayer on sapphire",
        "physical_size_nm": [250, 730, 580],
        "paper_original_shape": [100, 232, 120],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=str(PROJECT_DIR / "artifacts" / "upstream_data" / "official_exp_data"),
        help="Directory containing official exp_data/*.npy files.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=list(OFFICIAL_SAMPLES),
        help="Official experimental files to process.",
    )
    parser.add_argument(
        "--model",
        default=str(PROJECT_DIR / "artifacts" / "models" / "model_paper.h5"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            PROJECT_DIR
            / "artifacts"
            / "evaluations"
            / "experimental_tensorflow"
            / "official_exp_data"
        ),
    )
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--support-threshold", type=float, default=0.1)
    parser.add_argument("--interpolation-order", type=int, choices=(0, 1, 3), default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def configure_logging(output_dir: Path, level: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.handlers.clear()
    LOGGER.setLevel(getattr(logging, level))
    LOGGER.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(output_dir / "inference.log", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_experimental_intensity(path: Path) -> np.ndarray:
    """Load and validate one nonnegative three-dimensional intensity volume."""

    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if value.ndim != 3:
        raise ValueError(f"Expected a 3D diffraction volume, got {value.shape}: {path}")
    if not np.issubdtype(value.dtype, np.number):
        raise TypeError(f"Expected a numeric diffraction volume: {path}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"Experimental volume contains NaN or infinity: {path}")
    if np.any(value < 0):
        raise ValueError(f"Experimental intensity cannot be negative: {path}")
    if float(np.max(value)) <= 0:
        raise ValueError(f"Experimental intensity must contain signal: {path}")
    return value


def intensity_center(value: np.ndarray) -> np.ndarray:
    """Return the raw-intensity center of mass in source voxel coordinates."""

    center = np.asarray(center_of_mass(value), dtype=np.float64)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("Could not determine a finite diffraction center of mass.")
    return center


def resample_centered(
    value: np.ndarray,
    output_shape: Sequence[int],
    *,
    source_center: Sequence[float] | None = None,
    order: int = 1,
) -> np.ndarray:
    """Center and resample a real or complex 3D volume using one affine transform."""

    if value.ndim != 3:
        raise ValueError("resample_centered expects a three-dimensional array.")
    output_shape = tuple(int(size) for size in output_shape)
    if len(output_shape) != 3 or any(size < 2 for size in output_shape):
        raise ValueError("output_shape must contain three dimensions of at least two.")
    input_shape = np.asarray(value.shape, dtype=np.float64)
    output_size = np.asarray(output_shape, dtype=np.float64)
    source = (
        (input_shape - 1.0) / 2.0
        if source_center is None
        else np.asarray(source_center, dtype=np.float64)
    )
    if source.shape != (3,) or not np.all(np.isfinite(source)):
        raise ValueError("source_center must contain three finite coordinates.")
    scale = (input_shape - 1.0) / (output_size - 1.0)
    output_center = (output_size - 1.0) / 2.0
    offset = source - scale * output_center
    matrix = np.diag(scale)

    def transform(component: np.ndarray) -> np.ndarray:
        return affine_transform(
            component,
            matrix=matrix,
            offset=offset,
            output_shape=output_shape,
            order=order,
            mode="constant",
            cval=0.0,
            prefilter=order > 1,
        )

    if np.iscomplexobj(value):
        return (
            transform(np.asarray(value.real, dtype=np.float32))
            + 1j * transform(np.asarray(value.imag, dtype=np.float32))
        ).astype(np.complex64)
    return transform(np.asarray(value, dtype=np.float32)).astype(np.float32)


def support_diagnostics(
    reconstruction: np.ndarray, threshold: float
) -> tuple[np.ndarray, dict[str, Any]]:
    amplitude = np.abs(reconstruction)
    support = amplitude >= threshold * max(float(amplitude.max()), 1e-12)
    coordinates = np.argwhere(support)
    if coordinates.size:
        lower = coordinates.min(axis=0)
        upper = coordinates.max(axis=0)
        bbox = (upper - lower + 1).astype(int)
    else:
        lower = upper = bbox = np.zeros(3, dtype=int)
    components, count = label(support)
    if count:
        sizes = np.bincount(components.ravel())[1:]
        largest_fraction = float(sizes.max() / max(int(support.sum()), 1))
    else:
        largest_fraction = 0.0
    amplitude_center = (
        np.asarray(center_of_mass(amplitude), dtype=np.float64)
        if float(amplitude.sum()) > 0
        else np.full(3, np.nan)
    )
    diagnostics = {
        "support_threshold_relative_to_max": threshold,
        "support_voxels": int(support.sum()),
        "support_fraction": float(support.mean()),
        "largest_connected_component_fraction": largest_fraction,
        "support_bbox_lower": lower.tolist(),
        "support_bbox_upper": upper.tolist(),
        "support_bbox_shape": bbox.tolist(),
        "amplitude_center_of_mass": amplitude_center.tolist(),
    }
    return support, diagnostics


def modulus_consistency(intensity: np.ndarray, reconstruction: np.ndarray) -> float:
    recovered = np.abs(
        np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(reconstruction)))
    )
    measured = np.sqrt(np.clip(intensity, 0.0, None))
    denominator = max(float(np.linalg.norm(measured.ravel())), 1e-12)
    return float(np.linalg.norm((recovered - measured).ravel()) / denominator)


def _central_slice(value: np.ndarray, axis: int, index: int) -> np.ndarray:
    selection = [slice(None)] * 3
    selection[axis] = int(index)
    return value[tuple(selection)]


def save_slice_overview(
    source_intensity: np.ndarray,
    source_center: np.ndarray,
    resized_intensity: np.ndarray,
    predicted_phase: np.ndarray,
    reconstruction: np.ndarray,
    support: np.ndarray,
    destination: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source_indices = np.rint(source_center).astype(int)
    center = tuple(size // 2 for size in resized_intensity.shape)
    amplitude = np.abs(reconstruction)
    phase = np.angle(reconstruction)
    wrapped_reciprocal = np.mod(predicted_phase - predicted_phase[center], 2 * np.pi)

    figure, axes = plt.subplots(2, 4, figsize=(18, 8.5), constrained_layout=True)
    for axis in range(3):
        image = np.log10(np.clip(_central_slice(source_intensity, axis, source_indices[axis]), 1e-6, None))
        handle = axes[0, axis].imshow(image, cmap="magma", origin="lower")
        axes[0, axis].set_title(f"Original intensity, axis {axis}")
        figure.colorbar(handle, ax=axes[0, axis], shrink=0.78)
    handle = axes[0, 3].imshow(
        np.log10(np.clip(resized_intensity[center[0]], 1e-6, None)),
        cmap="magma",
        origin="lower",
    )
    axes[0, 3].set_title("COM-centered and resized intensity")
    figure.colorbar(handle, ax=axes[0, 3], shrink=0.78)

    handle = axes[1, 0].imshow(
        wrapped_reciprocal[center[0]],
        cmap="twilight",
        origin="lower",
        vmin=0,
        vmax=2 * np.pi,
    )
    axes[1, 0].set_title("Predicted reciprocal phase, wrapped")
    figure.colorbar(handle, ax=axes[1, 0], shrink=0.78)
    handle = axes[1, 1].imshow(amplitude[center[0]], cmap="viridis", origin="lower")
    axes[1, 1].set_title("Reconstructed amplitude")
    figure.colorbar(handle, ax=axes[1, 1], shrink=0.78)
    direct_phase = np.ma.masked_where(
        ~support[center[0]], np.mod(phase[center[0]], 2 * np.pi)
    )
    handle = axes[1, 2].imshow(
        direct_phase, cmap="twilight", origin="lower", vmin=0, vmax=2 * np.pi
    )
    axes[1, 2].set_title("Reconstructed wrapped phase")
    figure.colorbar(handle, ax=axes[1, 2], shrink=0.78)
    axes[1, 3].imshow(support[center[0]], cmap="gray", origin="lower", vmin=0, vmax=1)
    axes[1, 3].set_title("Diagnostic support")
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(title, fontsize=15)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, facecolor="white")
    plt.close(figure)


def _surface_points(support: np.ndarray, maximum: int = 18000) -> np.ndarray:
    surface = support & ~binary_erosion(support)
    points = np.argwhere(surface)
    if len(points) > maximum:
        points = points[np.linspace(0, len(points) - 1, maximum, dtype=int)]
    return points


def save_volume_overview(
    resized_intensity: np.ndarray,
    reconstruction: np.ndarray,
    original_reconstruction: np.ndarray,
    support_threshold: float,
    destination: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(16.5, 5.5), constrained_layout=True)
    axes = [figure.add_subplot(1, 3, index + 1, projection="3d") for index in range(3)]

    positive = resized_intensity[resized_intensity > 0]
    cutoff = float(np.quantile(positive, 0.99)) if positive.size else 0.0
    diffraction_points = np.argwhere(resized_intensity >= cutoff)
    if len(diffraction_points) > 16000:
        diffraction_points = diffraction_points[
            np.linspace(0, len(diffraction_points) - 1, 16000, dtype=int)
        ]
    diffraction_values = np.log10(
        np.clip(resized_intensity[tuple(diffraction_points.T)], 1e-6, None)
    )
    axes[0].scatter(
        diffraction_points[:, 2],
        diffraction_points[:, 1],
        diffraction_points[:, 0],
        c=diffraction_values,
        cmap="magma",
        s=3,
        linewidths=0,
    )
    axes[0].set_title("Measured diffraction, top 1%")

    for axis, value, subtitle in (
        (axes[1], reconstruction, "64-cube DL reconstruction"),
        (axes[2], original_reconstruction, "Re-interpolated original grid"),
    ):
        amplitude = np.abs(value)
        support = amplitude >= support_threshold * max(float(amplitude.max()), 1e-12)
        points = _surface_points(support)
        colors = (
            np.mod(np.angle(value)[tuple(points.T)], 2 * np.pi)
            if len(points)
            else np.empty(0)
        )
        axis.scatter(
            points[:, 2],
            points[:, 1],
            points[:, 0],
            c=colors,
            cmap="twilight",
            vmin=0,
            vmax=2 * np.pi,
            s=3,
            linewidths=0,
        )
        axis.set_title(subtitle)
        axis.set_box_aspect((value.shape[2], value.shape[1], value.shape[0]))
    for axis in axes:
        axis.set_xlabel("z")
        axis.set_ylabel("y")
        axis.set_zlabel("x")
    figure.suptitle(title, fontsize=14)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, facecolor="white")
    plt.close(figure)


def _sample_report(
    path: Path,
    source: np.ndarray,
    source_center: np.ndarray,
    resized_intensity: np.ndarray,
    prediction: np.ndarray,
    reconstruction: np.ndarray,
    original_reconstruction: np.ndarray,
    threshold: float,
    order: int,
    model_metadata: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    support, support64 = support_diagnostics(reconstruction, threshold)
    _, support_original = support_diagnostics(original_reconstruction, threshold)
    identity = OFFICIAL_SAMPLES.get(path.name, {})
    report = {
        "schema_version": 1,
        "source": {
            "path": str(path),
            "repository": OFFICIAL_REPOSITORY,
            "sha256": file_sha256(path),
            "shape": list(source.shape),
            "dtype": str(source.dtype),
            "file_bytes": path.stat().st_size,
            "minimum": float(source.min()),
            "maximum": float(source.max()),
            "mean": float(source.mean()),
            "nonzero_fraction": float(np.count_nonzero(source) / source.size),
            **identity,
        },
        "preprocessing": {
            "source_center_of_mass": source_center.tolist(),
            "source_geometric_center": (
                (np.asarray(source.shape, dtype=np.float64) - 1.0) / 2.0
            ).tolist(),
            "resize": f"single affine interpolation to {resized_intensity.shape}",
            "interpolation_order": order,
            "model_input": "log1p intensity followed by per-volume min-max normalization",
        },
        "model": model_metadata,
        "reconstruction": {
            "definition": "sqrt(resized measured intensity) * exp(i * predicted reciprocal phase), then centered inverse FFT",
            "phase_output_shape": list(prediction.shape),
            "predicted_phase_min": float(prediction.min()),
            "predicted_phase_max": float(prediction.max()),
            "visual_phase_convention": "wrapped to [0, 2pi), matching paper figures",
            "object_64_shape": list(reconstruction.shape),
            "reinterpolated_shape": list(original_reconstruction.shape),
            "reinterpolation": "real and imaginary components interpolated separately",
            "modulus_consistency_relative_l2": modulus_consistency(
                resized_intensity, reconstruction
            ),
            "support_64": support64,
            "support_original_grid": support_original,
        },
        "evaluation_limit": (
            "Experimental data have no reciprocal-phase or real-space ground truth; "
            "WCA, IoU, NRMSE, and phase MAE are therefore not reported."
        ),
        "outputs": {
            "arrays": str(output_dir / f"{path.stem}_reconstruction.npz"),
            "slices": str(output_dir / f"{path.stem}_slices.png"),
            "volume": str(output_dir / f"{path.stem}_3d.png"),
        },
    }
    return report


def main() -> int:
    args = parse_args()
    if args.grid_size < 2:
        raise ValueError("--grid-size must be at least two.")
    if not 0 < args.support_threshold < 1:
        raise ValueError("--support-threshold must be between zero and one.")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    data_dir = Path(args.data_dir).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    configure_logging(output_dir, args.log_level)
    paths = [(data_dir / name).resolve() for name in args.files]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing official experimental files: " + ", ".join(missing))
    if not model_path.is_file():
        raise FileNotFoundError(f"Official model not found: {model_path}")

    prepared: list[dict[str, Any]] = []
    for path in paths:
        source = load_experimental_intensity(path)
        paper_shape = OFFICIAL_SAMPLES.get(path.name, {}).get("paper_original_shape")
        if paper_shape and list(source.shape) != paper_shape:
            raise ValueError(
                f"{path.name} shape {source.shape} differs from paper shape {paper_shape}."
            )
        source_center = intensity_center(source)
        resized = np.clip(
            resample_centered(
                source,
                (args.grid_size,) * 3,
                source_center=source_center,
                order=args.interpolation_order,
            ),
            0.0,
            None,
        ).astype(np.float32)
        prepared.append(
            {
                "path": path,
                "shape": tuple(source.shape),
                "center": source_center,
                "intensity": resized,
                "input": prepare_model_input(resized),
            }
        )
        LOGGER.info(
            "%s: source=%s, COM=%s, resized=%s",
            path.name,
            tuple(source.shape),
            np.round(source_center, 3).tolist(),
            resized.shape,
        )

    import tensorflow as tf

    LOGGER.info("Loading official TensorFlow model: %s", model_path)
    model = tf.keras.models.load_model(model_path, compile=False)
    batch = np.concatenate([item["input"] for item in prepared], axis=0)
    started = time.perf_counter()
    predictions = model(batch, training=False).numpy()[..., 0].astype(np.float32)
    inference_seconds = time.perf_counter() - started
    if predictions.shape != (len(paths),) + (args.grid_size,) * 3:
        raise ValueError(f"Unexpected model output shape: {predictions.shape}")
    if not np.all(np.isfinite(predictions)):
        raise ValueError("Official model returned non-finite phase values.")
    model_metadata = {
        "backend": "tensorflow",
        "tensorflow_version": tf.__version__,
        "path": str(model_path),
        "sha256": file_sha256(model_path),
        "parameter_count": int(model.count_params()),
        "input_shape": list(batch.shape),
        "output_shape": list(predictions.shape),
        "inference_seconds": inference_seconds,
    }

    reports = []
    for item, prediction in zip(prepared, predictions):
        path = item["path"]
        source = load_experimental_intensity(path)
        reconstruction = reconstruct_object(item["intensity"], prediction)
        original_reconstruction = resample_centered(
            reconstruction,
            item["shape"],
            order=args.interpolation_order,
        )
        support, _ = support_diagnostics(reconstruction, args.support_threshold)
        arrays_path = output_dir / f"{path.stem}_reconstruction.npz"
        np.savez_compressed(
            arrays_path,
            resized_intensity=item["intensity"],
            model_input=item["input"][0, ..., 0],
            predicted_reciprocal_phase=prediction,
            reconstructed_object_64=reconstruction,
            reconstructed_object_original_grid=original_reconstruction,
        )
        title = OFFICIAL_SAMPLES.get(path.name, {}).get("paper_name", path.stem)
        save_slice_overview(
            source,
            item["center"],
            item["intensity"],
            prediction,
            reconstruction,
            support,
            output_dir / f"{path.stem}_slices.png",
            f"{title}: official experimental data and TensorFlow reconstruction",
        )
        save_volume_overview(
            item["intensity"],
            reconstruction,
            original_reconstruction,
            args.support_threshold,
            output_dir / f"{path.stem}_3d.png",
            f"{title}: official HighStrain model on experimental BCDI",
        )
        report = _sample_report(
            path,
            source,
            item["center"],
            item["intensity"],
            prediction,
            reconstruction,
            original_reconstruction,
            args.support_threshold,
            args.interpolation_order,
            model_metadata,
            output_dir,
        )
        (output_dir / f"{path.stem}_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        reports.append(report)
        LOGGER.info(
            "%s: support fraction=%.5f, largest component=%.5f",
            path.name,
            report["reconstruction"]["support_64"]["support_fraction"],
            report["reconstruction"]["support_64"][
                "largest_connected_component_fraction"
            ],
        )

    dataset_report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "official_repository": OFFICIAL_REPOSITORY,
        "paper": "https://doi.org/10.1038/s41524-026-02017-w",
        "configuration": vars(args),
        "model": model_metadata,
        "samples": reports,
        "interpretation": (
            "These are inference-only experimental reconstructions. Without ground "
            "truth, visual compactness and Fourier modulus consistency are diagnostics, "
            "not reconstruction-accuracy metrics."
        ),
    }
    (output_dir / "experimental_inference_report.json").write_text(
        json.dumps(dataset_report, indent=2), encoding="utf-8"
    )
    LOGGER.info("Finished %d experimental volumes: %s", len(reports), output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
