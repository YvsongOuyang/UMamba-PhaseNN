"""Exploratory single-image inference with a trained 2D e-CDI cascade.

Run from the high_strain_CNN project root:
    python -m ecdi_training.infer_tiff --input-tiff ... --checkpoint ... --output-dir ...

The input must be one nonnegative 2D diffraction intensity image. Images larger
than 512x512 are center-cropped; smaller images are rejected. This script does
not calculate accuracy metrics because a ground-truth complex object is absent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .train import load_checkpoint, make_model


def read_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Input TIFF must contain exactly one image frame.")
        array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"Expected a single-channel 2D TIFF, got shape {array.shape}.")
    array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all() or np.any(array < 0) or float(array.max()) <= 0:
        raise ValueError("TIFF pixels must be finite, nonnegative and not all zero.")
    return array


def center_crop_512(array: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = array.shape
    if height < 512 or width < 512:
        raise ValueError(
            f"Input is {width}x{height}; both dimensions must be at least 512."
        )
    top, left = (height - 512) // 2, (width - 512) // 2
    return array[top : top + 512, left : left + 512].copy(), (top, left)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tiff", required=True, help="One 2D TIFF diffraction frame")
    parser.add_argument("--checkpoint", required=True, help="Refiner-stage checkpoint_best.pt")
    parser.add_argument("--output-dir", required=True, help="New or existing output directory")
    parser.add_argument("--reference-amplitude", default="", help="Optional color-rendered amplitude reference TIFF")
    parser.add_argument("--reference-phase", default="", help="Optional color-rendered phase reference TIFF")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    input_path = Path(args.input_tiff).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this environment.")
    output_dir.mkdir(parents=True, exist_ok=True)

    original = read_image(input_path)
    crop, (top, left) = center_crop_512(original)
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint.get("stage") != "refiner":
        raise ValueError("Use the full refiner-stage checkpoint, not a phase-only checkpoint.")
    config = checkpoint["config"]
    model = make_model("refiner", config, checkpoint=checkpoint)
    device = torch.device(args.device)
    model.to(device).eval()

    intensity = torch.from_numpy(crop)[None, None].to(device)
    logged = torch.log1p(intensity)
    minimum = logged.amin(dim=(-2, -1), keepdim=True)
    model_input = (logged - minimum) / (
        logged.amax(dim=(-2, -1), keepdim=True) - minimum
    ).clamp_min(torch.finfo(logged.dtype).eps)
    diffraction_modulus = intensity.sqrt()
    diffraction_modulus = diffraction_modulus / diffraction_modulus.amax(
        dim=(-2, -1), keepdim=True
    ).clamp_min(1e-7)

    with torch.inference_mode():
        result = model(model_input, diffraction_modulus)

    input_crop = crop.astype(np.float32)
    reciprocal_phase = result.reciprocal_phase[0, 0].detach().cpu().numpy()
    initial = result.initial_object[0, 0].detach().cpu().numpy()
    refined = result.refined_object[0, 0].detach().cpu().numpy()
    np.savez_compressed(
        output_dir / "reconstruction.npz",
        input_intensity=input_crop,
        reciprocal_phase=reciprocal_phase,
        initial_complex=initial,
        initial_amplitude=np.abs(initial),
        initial_phase=np.angle(initial),
        refined_complex=refined,
        refined_amplitude=np.abs(refined),
        refined_phase=np.angle(refined),
        original_image_shape=np.asarray(original.shape, dtype=np.int32),
        crop_top=np.int32(top),
        crop_left=np.int32(left),
    )

    def amplitude_for_display(field: np.ndarray) -> np.ndarray:
        amplitude = np.abs(field).astype(np.float32)
        amplitude /= max(float(amplitude.max()), 1e-12)
        return np.log10(np.maximum(amplitude, 1e-4))

    initial_amp = np.abs(initial)
    initial_amp /= max(float(initial_amp.max()), 1e-12)
    refined_amp = np.abs(refined)
    refined_amp /= max(float(refined_amp.max()), 1e-12)
    initial_phase_display = np.ma.masked_where(initial_amp < 0.05, np.angle(initial))
    refined_phase_display = np.ma.masked_where(refined_amp < 0.05, np.angle(refined))

    reference_paths = (args.reference_amplitude, args.reference_phase)
    if any(reference_paths) and not all(reference_paths):
        raise ValueError("Provide both reference TIFFs or neither.")
    refs = []
    if all(reference_paths):
        for name, reference_path in zip(("amplitude", "phase"), reference_paths):
            with Image.open(reference_path) as ref_image:
                if getattr(ref_image, "n_frames", 1) != 1:
                    raise ValueError(f"Reference {name} TIFF must contain one frame.")
                refs.append(np.asarray(ref_image.convert("RGB")))
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
        panels = [
            (input_crop, "gray", "Input diffraction (512² center crop)"),
            (refs[0], None, "Amplitude reference (color-rendered)"),
            (refs[1], None, "Phase reference (color-rendered)"),
            (amplitude_for_display(initial), "magma", "Model initial amplitude (log, peak-normalized)"),
            (amplitude_for_display(refined), "magma", "Model refined amplitude (log, peak-normalized)"),
            (refined_phase_display, "twilight", "Model refined phase (low-amplitude pixels masked)"),
        ]
    else:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
        panels = [
            (input_crop, "gray", "Input diffraction (512² center crop)"),
            (amplitude_for_display(initial), "magma", "Model initial amplitude (log, peak-normalized)"),
            (initial_phase_display, "twilight", "Model initial phase (low-amplitude pixels masked)"),
            (input_crop, "gray", "Input diffraction (512² center crop)"),
            (amplitude_for_display(refined), "magma", "Model refined amplitude (log, peak-normalized)"),
            (refined_phase_display, "twilight", "Model refined phase (low-amplitude pixels masked)"),
        ]
    for axis, (panel, cmap, title) in zip(axes.flat, panels):
        axis.imshow(panel, cmap=cmap, vmin=(-4 if "amplitude" in title else None), vmax=(0 if "amplitude" in title else None))
        axis.set_title(title, fontsize=9)
        axis.set_axis_off()
    fig.suptitle(
        f"512×512 center crop from {original.shape[1]}×{original.shape[0]} TIFF; "
        "exploratory inference; reference panels are visual only"
    )
    fig.savefig(output_dir / "reconstruction.png", dpi=160)
    plt.close(fig)

    print(f"Checkpoint stage={checkpoint['stage']} epoch={checkpoint.get('epoch')}")
    print(f"Input={input_path} | original={original.shape[1]}x{original.shape[0]} | crop x={left}:{left+512}, y={top}:{top+512}")
    print(f"Saved {output_dir / 'reconstruction.npz'}")
    print(f"Saved {output_dir / 'reconstruction.png'}")
    print("No PSNR/MAE reported: this TIFF has no paired ground-truth complex object.")
    if all(reference_paths):
        print("Reference TIFFs are rendered color images; shown for visual comparison only, not pixelwise metrics.")
    print("Treat as exploratory: the TIFF is a rendered 8-bit image, not confirmed raw calibrated detector counts.")


if __name__ == "__main__":
    main()
