"""Compare original and tuned Au5nm reconstructions before/after projection.

The supplied target is an uncalibrated color-derived pseudo-label. All scores
refer to one sample; projection enforces the *processed* diffraction amplitude,
including pixels clipped to zero during background subtraction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pytorch_autophasenn.momamba_refiner import ambiguity_aware_component_mae
from pytorch_autophasenn.reconstruction import (
    farfield_modulus_from_realspace,
    project_to_measured_modulus,
)
from .data import build_dataset, load_config, prepare_batch
from .train import build_loader, load_checkpoint, make_model


ROOT = "/data_ssd/oyys/high_strain_cnn/pipeline_output_2d"
ORIGINAL = f"{ROOT}/ecdi_2d512_20261014_224534_refiner/checkpoint_best.pt"
TUNED = f"{ROOT}/au5nm_fixed_bn_20261027/checkpoint_last.pt"
DATA = "/data_ssd/oyys/eCDI/external_test/au5nm/finetune_data/config.json"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original", default=ORIGINAL)
    p.add_argument("--tuned", default=TUNED)
    p.add_argument("--data-config", default=DATA)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--output-dir", default="artifacts/evaluations/au5nm_projection_compare")
    return p.parse_args()


def compute_metrics(field, batch, support):
    """Match the training pseudo-loss and scale-fitted amplitude error."""
    obj = ambiguity_aware_component_mae(
        field, batch["realspace"], support, spatial_dims=2
    )
    pred = farfield_modulus_from_realspace(field, spatial_dims=2)
    measured = batch["diffraction"]
    scale = (pred * measured).sum() / pred.square().sum().clamp_min(1e-20)
    error = torch.linalg.vector_norm(scale * pred - measured) / torch.linalg.vector_norm(measured).clamp_min(1e-20)
    return {"pseudo_label_loss": float(obj.loss), "diffraction_rel_l2": float(error)}


def save_figure(fields, target, metrics, path):
    labels = ["Pseudo-reference", "Original", "Original + projection", "Fine-tuned (epoch 30)", "Fine-tuned + projection"]
    names = [None, "original", "original_projected", "finetuned", "finetuned_projected"]
    reference = fields["original"]
    images = [target] + [fields[name] for name in names[1:]]
    fig, axes = plt.subplots(2, 5, figsize=(18, 7.5), constrained_layout=True)
    amp_artist = phase_artist = None
    for j, (image, label, name) in enumerate(zip(images, labels, names)):
        amplitude = np.abs(image)
        peak = float(amplitude.max())
        normalized = amplitude / max(peak, 1e-12)
        amp_artist = axes[0, j].imshow(normalized, cmap="magma", vmin=0, vmax=1)
        cross = np.vdot(reference, image)
        aligned = image * np.exp(-1j * np.angle(cross)) if abs(cross) > 0 else image
        phase = np.ma.masked_where(normalized < 0.05, np.angle(aligned))
        phase_artist = axes[1, j].imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        title = label
        if name is not None:
            item = metrics[name]
            title += f"\nLoss {item['pseudo_label_loss']:.4f} | Diff {item['diffraction_rel_l2']:.4f}"
        axes[0, j].set_title(title, fontsize=10)
        axes[0, j].axis("off")
        axes[1, j].axis("off")
    axes[0, 0].set_ylabel("Amplitude", fontsize=11)
    axes[1, 0].set_ylabel("Phase", fontsize=11)
    fig.colorbar(amp_artist, ax=axes[0, :].tolist(), shrink=0.75, label="Amplitude, per-image normalized")
    fig.colorbar(phase_artist, ax=axes[1, :].tolist(), shrink=0.75, label="Phase (radians)")
    fig.suptitle("Au5nm exploratory comparison: one sample and approximate pseudo-reference", fontsize=14)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu explicitly")
    device = torch.device(args.device)
    original = load_checkpoint(args.original)
    tuned = load_checkpoint(args.tuned)
    config = load_config(args.data_config)
    for checkpoint in (original, tuned):
        if checkpoint.get("stage") != "refiner":
            raise ValueError("Both checkpoints must contain the refiner cascade")
        for key in ("diffraction_kind", "input_log_data", "center_targets"):
            if config[key] != checkpoint["config"][key]:
                raise ValueError(f"Checkpoint/data preprocessing differs on {key}")
    config["training"] = original["config"]["training"]
    if config["training"]["physics_loss_weight"] != 0:
        raise ValueError("This comparison expects the original zero physics-loss weight")
    dataset = build_dataset(config, "val")
    if len(dataset) != 1:
        raise ValueError("Expected exactly one Au5nm sample")
    loader = build_loader(dataset, 1, 0, False, device)
    batch = prepare_batch(next(iter(loader)), config, device)
    target = batch["realspace"]
    support = target.abs() / target.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-7)
    support = support >= config["training"]["support_threshold"]
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}; choose --output-dir")
    output.mkdir(parents=True, exist_ok=True)
    fields, scores = {}, {}
    with torch.no_grad():
        model = make_model("refiner", config, checkpoint=original).to(device).eval()
        for name, checkpoint in (("original", original), ("finetuned", tuned)):
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            result = model(batch["input"], batch["diffraction"])
            unprojected = result.refined_object
            projected = project_to_measured_modulus(unprojected, batch["diffraction"], spatial_dims=2)
            for key, field in ((name, unprojected), (name + "_projected", projected)):
                scores[key] = compute_metrics(field, batch, support)
                fields[key] = field[0, 0].detach().cpu().numpy()
                print(f"{key} | pseudo_label_loss={scores[key]['pseudo_label_loss']:.6f} | diffraction_rel_l2={scores[key]['diffraction_rel_l2']:.8f}", flush=True)
    report = {
        "sample_count": 1,
        "original_checkpoint": str(Path(args.original).resolve()),
        "tuned_checkpoint": str(Path(args.tuned).resolve()),
        "tuned_epoch": tuned.get("epoch"),
        "metrics": scores,
        "notes": [
            "Pseudo-label is a color-derived approximation, not a calibrated real-space ground truth.",
            "Diffraction error is scale-fitted Fourier-amplitude relative L2 error against the processed input.",
            "Projection enforces all processed input pixels, including the central pixels clipped to zero; this is not proof of physical accuracy.",
            "Display amplitudes are normalized per image; phases have a global phase offset aligned to original and values below 5% amplitude are masked.",
        ],
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    save_figure(fields, target[0, 0].cpu().numpy(), scores, output / "comparison.png")
    print(f"Output={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
