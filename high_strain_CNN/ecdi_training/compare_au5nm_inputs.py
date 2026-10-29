"""Compare Au5nm input handling using the original refiner model only.

All three inputs use the same approximate color-derived real-space target.
The data are one 8-bit rendered diffraction image, not calibrated counts.
This comparison does not train a model or establish experimental accuracy.
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
from pytorch_autophasenn.reconstruction import farfield_modulus_from_realspace
from .data import build_dataset, load_config, prepare_batch
from .train import build_loader, load_checkpoint, make_model


ROOT = "/data_ssd/oyys/eCDI/external_test/au5nm/input_check"
OLD_CONFIG = "/data_ssd/oyys/eCDI/external_test/au5nm/finetune_data/config.json"
SOURCE = "/data_ssd/oyys/high_strain_cnn/pipeline_output_2d/ecdi_2d512_20261014_224534_refiner/checkpoint_best.pt"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", default=ROOT)
    p.add_argument("--old-config", default=OLD_CONFIG)
    p.add_argument("--checkpoint", default=SOURCE)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--output-dir", default="artifacts/evaluations/au5nm_input_check_comparison")
    args = p.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu explicitly")
    device = torch.device(args.device)
    root = Path(args.input_root)
    checkpoint = load_checkpoint(args.checkpoint)
    if checkpoint.get("stage") != "refiner":
        raise ValueError("Expected a refiner-stage checkpoint")

    paths = [("old", Path(args.old_config)), ("raw_crop", root / "config_raw.json"), ("subtracted_crop", root / "config_subtracted.json")]
    negative = np.load(root / "negative_subtraction_mask.npy", allow_pickle=False)
    if negative.shape != (1, 512, 512) or negative.dtype != np.bool_:
        raise ValueError("Expected the background-subtraction mask as bool [1,512,512]")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    model = None
    scores, images = {}, []
    common_valid = torch.from_numpy(~negative).to(device)[:, None]
    original_target = None
    with torch.no_grad():
        for name, path in paths:
            config = load_config(path)
            for key in ("diffraction_kind", "input_log_data", "center_targets"):
                if config[key] != checkpoint["config"][key]:
                    raise ValueError(f"Checkpoint and {name} data differ on {key}")
            config["training"] = checkpoint["config"]["training"]
            if config["training"]["physics_loss_weight"] != 0:
                raise ValueError("This comparison expects the original zero physics-loss weight")
            dataset = build_dataset(config, "val")
            if len(dataset) != 1:
                raise ValueError(f"Expected one sample for {name}")
            loader = build_loader(dataset, 1, 0, False, device)
            batch = prepare_batch(next(iter(loader)), config, device)
            if original_target is None:
                original_target = batch["realspace"].clone()
            elif not torch.allclose(original_target, batch["realspace"], rtol=1e-5, atol=1e-7):
                raise ValueError("Targets differ across input variants")
            if model is None:
                model = make_model("refiner", config, checkpoint=checkpoint).to(device).eval()
            result = model(batch["input"], batch["diffraction"])
            target = batch["realspace"]
            support = target.abs() / target.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-7)
            support = support >= config["training"]["support_threshold"]
            loss = ambiguity_aware_component_mae(result.refined_object, target, support, spatial_dims=2).loss
            predicted = farfield_modulus_from_realspace(result.refined_object, spatial_dims=2)
            measured = batch["diffraction"]
            def relative_error(valid: torch.Tensor) -> float:
                x, y = predicted[valid], measured[valid]
                if x.numel() == 0 or y.square().sum() <= 0:
                    raise ValueError(f"No nonzero measured amplitude for {name}")
                scale = (x * y).sum() / x.square().sum().clamp_min(1e-20)
                return float(torch.linalg.vector_norm(scale * x - y) / torch.linalg.vector_norm(y).clamp_min(1e-20))

            full_error = relative_error(torch.ones_like(measured, dtype=torch.bool))
            common_error = relative_error(common_valid)
            scores[name] = {"pseudo_label_loss": float(loss),
                            "diffraction_rel_l2_full": full_error,
                            "diffraction_rel_l2_common_unclipped": common_error,
                            "common_unclipped_fraction": float(common_valid.float().mean()),
                            "model_input_mean": float(batch["input"].mean())}
            images.append((batch["input"][0, 0].cpu().numpy(), result.refined_object[0, 0].cpu().numpy()))
            print(f"{name} | pseudo_label_loss={float(loss):.6f} | full_rel_l2={full_error:.6f} | common_unclipped_rel_l2={common_error:.6f}", flush=True)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    titles = ("Old: resize, subtract, max-normalize", "Raw: center crop", "Subtracted: center crop")
    for col, ((input_image, reconstruction), title) in enumerate(zip(images, titles)):
        axes[0, col].imshow(input_image, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(title, fontsize=10)
        amplitude = np.abs(reconstruction)
        amplitude /= max(float(amplitude.max()), 1e-12)
        axes[1, col].imshow(np.log10(np.maximum(amplitude, 1e-4)), cmap="magma", vmin=-4, vmax=0)
        for row in range(2):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    axes[0, 0].set_ylabel("Model input")
    axes[1, 0].set_ylabel("Output amplitude (log, peak-normalized)")
    fig.suptitle("Au5nm: original model on three input preparations", fontsize=14)
    fig.savefig(output / "input_comparison.png", dpi=160)
    plt.close(fig)
    report = {"sample_count": 1, "checkpoint": str(Path(args.checkpoint).resolve()),
              "metrics": scores, "notes": [
                  "Target is an uncalibrated color-derived pseudo-label; scores do not establish experimental accuracy.",
                  "Each variant is scored on both its whole 512x512 frame and the same spatial mask where background subtraction did not clip. Inputs differ, so neither comparison establishes physical correctness.",
                  "The mask is used for the common-mask diagnostic only; the model still receives zero at clipped pixels for the subtracted variant.",
                  "TIFF values are not confirmed linear detector counts and physical geometry is not calibrated.",
              ]}
    (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Output={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
