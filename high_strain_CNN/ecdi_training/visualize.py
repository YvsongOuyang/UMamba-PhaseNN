"""Plot saved e-CDI input, amplitude, phase and diffraction comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .evaluate import aligned_objects_2d, unit_max
from .data import center_object


def render(path: Path, output: Path, threshold=0.3):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(path, allow_pickle=False) as sample:
        target = torch.from_numpy(sample["target"].copy())[None, None]
        stages = [name for name in ("initial", "refined") if name in sample]
        aligned = {
            name: aligned_objects_2d(
                torch.from_numpy(sample[name].copy())[None, None], target, threshold
            )
            for name in stages
        }
        truth = center_object(target)
        truth = truth / truth.abs().square().mean().sqrt().clamp_min(1e-7)
        objects = [("Target", truth, truth)] + [
            (name.title(), *aligned[name]) for name in stages
        ]
        common_error_max = max(
            float((prediction - selected_truth).abs().max())
            for _, prediction, selected_truth in objects
        )
        common_error_max = max(common_error_max, 1e-7)
        observed = sample["measured_modulus"].copy() ** 2
        observed = observed / max(float(observed.sum()), 1e-12)
        title = f"sample={int(sample['sample'])}"
        if "material" in sample:
            title += f" | {str(sample['material'])}"
        figure, axes = plt.subplots(
            len(objects),
            5,
            figsize=(16, 3.2 * len(objects)),
            squeeze=False,
            constrained_layout=True,
        )
        for row, (label, prediction, selected_truth) in enumerate(objects):
            amp = unit_max(prediction.abs())[0, 0].numpy()
            mask = amp >= threshold
            phase = np.where(mask, torch.angle(prediction)[0, 0].numpy(), np.nan)
            spectrum = np.fft.fftshift(
                np.fft.fft2(np.fft.ifftshift(prediction[0, 0].numpy()))
            )
            intensity = np.abs(spectrum) ** 2
            intensity /= max(float(intensity.sum()), 1e-12)
            error = (prediction - selected_truth).abs()[0, 0].numpy()
            panels = [
                (amp, "Amplitude", "viridis", 0, 1),
                (
                    phase,
                    "Phase (rad)",
                    "twilight",
                    -np.pi,
                    np.pi,
                ),
                (
                    np.log10(observed + 1e-12),
                    "Measured diffraction\nlog probability",
                    "magma",
                    -12,
                    0,
                ),
                (
                    np.log10(intensity + 1e-12),
                    "Object diffraction\nlog probability",
                    "magma",
                    -12,
                    0,
                ),
                (
                    error,
                    "Aligned complex error\nRMS scale",
                    "inferno",
                    0,
                    common_error_max,
                ),
            ]
            for column, (data, name, cmap, vmin, vmax) in enumerate(panels):
                image = axes[row, column].imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
                axes[row, column].set_title(f"{label}\n{name}", fontsize=9)
                axes[row, column].axis("off")
                figure.colorbar(image, ax=axes[row, column], fraction=0.046, pad=0.03)
        figure.suptitle(title)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160, bbox_inches="tight")
        plt.close(figure)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--support-threshold", type=float, default=0.3)
    args = parser.parse_args()
    if not 0 < args.support_threshold < 1:
        parser.error("Support threshold must be in (0,1).")
    root = Path(args.evaluation_dir)
    samples = sorted(root.glob("sample_*.npz"))
    if not samples:
        raise FileNotFoundError(
            "No saved sample NPZs; rerun evaluate with --save-samples > 0."
        )
    for path in samples:
        render(
            path, root / "visualizations" / f"{path.stem}.png", args.support_threshold
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
