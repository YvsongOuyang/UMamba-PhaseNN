"""Native configurable-resolution adaptation of the author's 2D generator.

DOMAIN: custom. Spatial geometry scales with image size; no image resizing is used.
The default phase convention retains the upstream exp(2*pi*i*angle(B)).
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
import hashlib
import json
import logging
import math
from pathlib import Path
import platform
import shutil
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE_URL = "https://github.com/XFELDataScience/Complex-NNphase/blob/main/Simulated%20Date/generate_data.py"


def fft_centered(obj: np.ndarray) -> np.ndarray:
    """Centered FFT over the last two axes; handles [N,H,W] arrays."""
    axes = (-2, -1)
    return np.fft.ifftshift(
        np.fft.fftn(np.fft.fftshift(obj, axes=axes), axes=axes), axes=axes
    )

def show_image(fig, ax, array, title, cmap="viridis", **kwargs):
    artist = ax.imshow(array, origin="upper", cmap=cmap, **kwargs)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    fig.colorbar(artist, ax=ax, shrink=0.8, pad=0.025)

def log_intensity(modulus, config):
    intensity = np.square(modulus.astype(np.float64))
    return np.log10(intensity / intensity.max() + config.log_intensity_floor)

LOGGER = logging.getLogger(__name__)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class GenerationConfig:
    image_size: int = 512
    seeds: int = 3
    samples_per_axis: int = 2
    reference_size: int = 64
    domain_width: int = 6
    domain_height: int = 8
    domain_sigma_x: float = 1.5
    domain_sigma_y: float = 3.0
    domain_rotation_degrees: float = -25.0
    domain_pitch: int = 3
    offset_scale: int = 10
    field_start_margin: int = 10
    field_end_margin: int = 20
    envelope_sigma_x: float = 9.0
    envelope_sigma_y: float = 30.0
    envelope_rotation_degrees: float = -35.0
    phase_convention: str = "author"
    preview_seeds: int = 3
    dpi: int = 160
    log_intensity_floor: float = 1e-8

    @property
    def scale(self):
        return self.image_size // self.reference_size

    @property
    def field_size(self):
        return (self.samples_per_axis + 2) * self.image_size

    @property
    def phase_factor(self):
        return 2 * math.pi if self.phase_convention == "author" else 1.0

    def validate(self):
        if self.reference_size != 64 or self.image_size < 64 or self.image_size % 64:
            raise ValueError("image_size must be a positive multiple of the 64-pixel upstream reference")
        if self.seeds < 1 or self.samples_per_axis < 1 or self.preview_seeds < 1:
            raise ValueError("seeds, samples_per_axis and preview_seeds must be positive")
        if self.phase_convention not in {"author", "radians"}:
            raise ValueError("phase_convention must be author or radians")
        if not 0 < self.log_intensity_floor < 1:
            raise ValueError("log_intensity_floor must be between 0 and 1")
        widths = (self.domain_sigma_x, self.domain_sigma_y, self.envelope_sigma_x, self.envelope_sigma_y)
        if min(widths) <= 0 or self.domain_pitch <= 0:
            raise ValueError("Gaussian widths and domain pitch must be positive")
        # Keep all shifted domains inside the canvas; avoids clipped slice shape errors.
        jitter = math.ceil(self.offset_scale / 2)
        if self.field_start_margin < jitter or self.field_end_margin < max(self.domain_width, self.domain_height) + jitter:
            raise ValueError("Field margins are too small for domain extent and random shifts")


def gaussian(width, height, sigma_x, sigma_y, rotation):
    """Return [height,width] Gaussian, retaining the upstream centering convention."""
    result = np.zeros((height, width), dtype=np.float64)
    for row in range(height):
        for col in range(width):
            x = (row - width / 2) * math.sin(rotation) + (col - width / 2) * math.cos(rotation)
            y = (row - height / 2) * math.cos(rotation) - (col - height / 2) * math.sin(rotation)
            result[row, col] = math.exp(-x**2 / sigma_x**2 - y**2 / sigma_y**2)
    return result


def kernels(config):
    s = config.scale
    # Match the author's operation order for degree-to-radian conversion.
    domain_rotation = math.pi * config.domain_rotation_degrees / 180
    envelope_rotation = math.pi * config.envelope_rotation_degrees / 180
    domain = gaussian(config.domain_width * s, config.domain_height * s,
                      config.domain_sigma_x * s, config.domain_sigma_y * s, domain_rotation)
    envelope = gaussian(config.image_size, config.image_size,
                        config.envelope_sigma_x * s, config.envelope_sigma_y * s, envelope_rotation)
    return domain, envelope


def random_field(seed, domain, config):
    """Generate complex128 [field_size,field_size] from native scaled domains."""
    rng = np.random.RandomState(seed)
    field = np.zeros((config.field_size, config.field_size), dtype=np.complex128)
    height, width = domain.shape
    positions = range(config.field_start_margin * config.scale,
                      config.field_size - config.field_end_margin * config.scale,
                      config.domain_pitch * config.scale)
    for row in positions:
        for col in positions:
            field[row:row + height, col:col + width] += domain * np.exp(2j * math.pi * rng.rand())
            ox = int(config.offset_scale * config.scale * (rng.rand() - 0.5))
            oy = int(config.offset_scale * config.scale * (rng.rand() - 0.5))
            field[row + oy:row + oy + height, col + ox:col + ox + width] += domain * np.exp(2j * math.pi * rng.rand())
    return field


def samples(field, envelope, config):
    """Return amplitude, radians phase and detector amplitude, all [N,H,W]."""
    shape = (config.samples_per_axis**2, config.image_size, config.image_size)
    amplitude, phase = np.empty(shape, np.float64), np.empty(shape, np.float64)
    # FFT return dtype follows NumPy, matching the author's float32 complex input.
    detector = []
    positions = range(config.image_size, config.field_size - config.image_size, config.image_size)
    for index, (row, col) in enumerate((r, c) for r in positions for c in positions):
        obj = field[row:row + config.image_size, col:col + config.image_size] * envelope
        amplitude[index], phase[index] = abs(obj), np.angle(obj)
        fft_object = amplitude[index].astype(np.float32) * np.exp(1j * config.phase_factor * phase[index].astype(np.float32))
        detector.append(abs(fft_centered(fft_object)))
    return {"object_modulus": amplitude, "object_phase": phase, "modulus_detector": np.asarray(detector)}


def verify(arrays, config):
    expected = (config.samples_per_axis**2, config.image_size, config.image_size)
    for name, arr in arrays.items():
        if arr.shape != expected or not np.isfinite(arr).all():
            raise ValueError(f"Invalid shape/nonfinite values: {name}")
    amp, phase, detector = (arrays[k] for k in ("object_modulus", "object_phase", "modulus_detector"))
    if (amp < 0).any() or (detector < 0).any() or (abs(phase) > math.pi).any():
        raise ValueError("Invalid amplitude or phase range")
    # Verify one sample at a time to bound additional FFT memory.
    for index in range(len(amp)):
        obj = amp[index].astype(np.float32) * np.exp(1j * config.phase_factor * phase[index].astype(np.float32))
        np.testing.assert_array_equal(abs(fft_centered(obj)), detector[index])


def upstream_parity():
    """Check the generalized geometry and FFT against the actual 64px source."""
    source = ROOT / "upstream" / "generate_data.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    namespace = {}
    exec(compile(tree, str(source), "exec"), namespace)
    config = GenerationConfig(image_size=64, seeds=1, samples_per_axis=14)
    domain, envelope = kernels(config)
    original_domain = namespace["generate_ellipse"](1.5, 3, 6, 8)
    original_envelope = namespace["generate_Gaussian_support"](-math.pi * 35 / 180)
    np.testing.assert_array_equal(domain, original_domain)
    np.testing.assert_array_equal(envelope, original_envelope)
    original_field = namespace["generate_random_phase"](0, 6, 8, original_domain)
    adapted_field = random_field(0, domain, config)
    # np.exp vs cmath.exp can differ by floating point rounding.
    np.testing.assert_allclose(adapted_field, original_field, rtol=1e-12, atol=1e-12)
    original_phase, original_amp, original_detector = namespace["Generate_synthetic_data"](original_field, original_envelope, 1024, 448)
    # Isolate extraction/FFT parity from small domain summation rounding differences.
    adapted = samples(original_field, envelope, config)
    for key, original in (("object_phase", original_phase), ("object_modulus", original_amp), ("modulus_detector", original_detector)):
        np.testing.assert_array_equal(adapted[key], np.asarray(original))
    return {"reference_size": 64, "samples_checked": 196, "domain_and_envelope": "bitwise equal", "random_field_max_abs_error": float(abs(adapted_field - original_field).max()), "crop_and_forward": "all 196 amplitude, phase and detector arrays bitwise equal"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "demo512.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int)
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--samples-per-axis", type=int)
    parser.add_argument("--phase-convention", choices=("author", "radians"))
    parser.add_argument("--verify-upstream", action="store_true")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    for key, value in (("image_size", args.size), ("seeds", args.seeds), ("samples_per_axis", args.samples_per_axis), ("phase_convention", args.phase_convention)):
        if value is not None:
            settings[key] = value
    config = GenerationConfig(**settings)
    config.validate()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "synthetic_data"
    raw.mkdir()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(output / "run.log", encoding="utf-8")])
    required_bytes = config.seeds * config.samples_per_axis**2 * config.image_size**2 * 24
    if shutil.disk_usage(output).free < required_bytes * 1.2:
        raise OSError("Insufficient disk space for the requested dataset")
    started = time.perf_counter()
    parity = upstream_parity() if args.verify_upstream else None
    if parity:
        LOGGER.info("64px upstream comparison passed: %s", parity)
    domain, envelope = kernels(config)
    records, previews = [], []
    generation_seconds = 0.0
    LOGGER.info("Native %dx%d generation: scale=%d, field=%dx%d, samples=%d, phase=%s", config.image_size, config.image_size, config.scale, config.field_size, config.field_size, config.seeds * config.samples_per_axis**2, config.phase_convention)
    for seed in range(config.seeds):
        seed_started = time.perf_counter()
        field = random_field(seed, domain, config)
        arrays = samples(field, envelope, config)
        if seed == 0:
            duplicate = random_field(seed, domain, config)
            np.testing.assert_array_equal(field, duplicate)
            del duplicate
        del field
        path = raw / f"data{seed}.npz"
        np.savez(path, **arrays)
        generation_seconds += time.perf_counter() - seed_started
        with np.load(path, allow_pickle=False) as saved:
            loaded = {key: saved[key] for key in saved.files}
        for key in arrays:
            np.testing.assert_array_equal(arrays[key], loaded[key])
        verify(loaded, config)
        if seed < config.preview_seeds:
            previews.append({key: arr[0].copy() for key, arr in arrays.items()})
        records.append({
            "seed": seed,
            "file": f"synthetic_data/{path.name}",
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "arrays": {
                key: {"shape": list(arr.shape), "dtype": str(arr.dtype),
                      "min": float(arr.min()), "max": float(arr.max())}
                for key, arr in arrays.items()
            },
            "forward_max_abs_error": 0.0,
        })
        del arrays, loaded
        LOGGER.info("Seed %d saved and verified: %d samples, %.2f MiB", seed, config.samples_per_axis**2, path.stat().st_size / 2**20)
    fig, axes = plt.subplots(len(previews), 4, figsize=(15, 3.6 * len(previews)), squeeze=False)
    for seed, item in enumerate(previews):
        amp, phase, detector = (item[key] for key in ("object_modulus", "object_phase", "modulus_detector"))
        show_image(fig, axes[seed, 0], amp, f"seed {seed}, sample 0\nObject amplitude")
        show_image(fig, axes[seed, 1], phase, "Object phase [rad]", cmap="twilight", vmin=-np.pi, vmax=np.pi)
        show_image(fig, axes[seed, 2], detector, "Detector amplitude")
        show_image(fig, axes[seed, 3], log_intensity(detector, config), "log10(I / Imax)", cmap="magma", vmin=np.log10(config.log_intensity_floor), vmax=0)
    fig.suptitle(f"Native {config.image_size} x {config.image_size} | scaled Gaussian domains | phase: {config.phase_convention}")
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.5)
    fig.savefig(output / "samples_overview.png", dpi=config.dpi)
    plt.close(fig)
    source = ROOT / "upstream" / "generate_data.py"
    manifest = {"status": "complete", "source_url": SOURCE_URL, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "config": asdict(config), "geometry": {"scale": config.scale, "field_size": config.field_size, "domain_shape": list(domain.shape), "domain_sigma": [config.domain_sigma_x * config.scale, config.domain_sigma_y * config.scale], "domain_pitch": config.domain_pitch * config.scale, "envelope_sigma": [config.envelope_sigma_x * config.scale, config.envelope_sigma_y * config.scale], "crop_stride": config.image_size}, "samples": config.seeds * config.samples_per_axis**2, "generation_seconds_including_seed0_repeat": generation_seconds, "total_seconds": time.perf_counter() - started, "total_data_bytes": sum(record["bytes"] for record in records), "environment": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__, "executable": sys.executable, "device": "CPU"}, "upstream_parity": parity, "seed0_reproducibility": "random field bitwise equal", "interpretation": "Native 2D arrays, not resized images. Geometry is scaled in pixels; no physical length calibration is implied. Detector arrays store amplitude, not intensity. Default author phase uses exp(2*pi*i*angle(B)); radians mode uses exp(i*angle(B)). Smaller canvas/sample count changes random sequence placement relative to the full author dataset.", "files": records}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    LOGGER.info("Complete: %d samples, %.2f MiB, %.2f seconds", manifest["samples"], manifest["total_data_bytes"] / 2**20, manifest["total_seconds"])


if __name__ == "__main__":
    main()
