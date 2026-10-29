"""Fit absolute complex output of the original cascade to one Au5nm pseudo-label.

This is a one-sample fit, not an independent accuracy test. The model receives
the un-subtracted 512x512 crop. Its target was estimated from rendered color
TIFFs and is not a calibrated experimental complex object.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_autophasenn.momamba_refiner import (
    ambiguity_aware_component_mae,
)
from pytorch_autophasenn.reconstruction import farfield_modulus_from_realspace
from .data import build_dataset, load_config, prepare_batch
from .train import LOGGER, PROJECT_DIR, build_loader, load_checkpoint, make_model


SOURCE = "/data_ssd/oyys/high_strain_cnn/pipeline_output_2d/ecdi_2d512_20261014_224534_refiner/checkpoint_best.pt"
DATA = "/data_ssd/oyys/eCDI/external_test/au5nm/input_check/config_raw.json"
CHECKPOINT_ROOT = "/data_ssd/oyys/high_strain_cnn/pipeline_output_2d"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-checkpoint", default=SOURCE)
    p.add_argument("--data-config", default=DATA)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--run-name", default="au5nm_absolute_fit")
    p.add_argument("--checkpoint-root", default=CHECKPOINT_ROOT)
    p.add_argument("--runs-dir", default=str(PROJECT_DIR / "artifacts" / "training" / "ecdi_training"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.steps < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        p.error("steps and learning-rate must be positive")
    if Path(args.run_name).name != args.run_name or args.run_name in ("", ".", ".."):
        p.error("run-name must be a single directory name")
    return args


def absolute_fit_loss(prediction, target, support):
    """Compare complex values after global phase and twin alignment, retaining scale."""
    axes = (-2, -1)
    twin = torch.conj(torch.roll(torch.flip(target, dims=axes), shifts=(1, 1), dims=axes))
    twin_support = torch.roll(torch.flip(support, dims=axes), shifts=(1, 1), dims=axes)
    scale = target.abs().mean().clamp_min(1e-7)
    losses = []
    for candidate, mask in ((target, support), (twin, twin_support)):
        correlation = (candidate.conj() * prediction * mask).sum(dim=axes, keepdim=True)
        aligned = prediction * torch.exp(-1j * torch.angle(correlation))
        component = ((aligned.real - candidate.real).abs().mean()
                     + (aligned.imag - candidate.imag).abs().mean()) / scale
        amplitude = (prediction.abs() - candidate.abs()).abs().mean() / scale
        losses.append(component + 0.5 * amplitude)
    return torch.minimum(losses[0], losses[1])


def train_one(model, raw, config, device, optimizer):
    # eval() disables BN running-stat updates and dropout, but does not disable gradients.
    model.eval()
    batch = prepare_batch(raw, config, device)
    optimizer.zero_grad(set_to_none=True)
    output = model(batch["input"], batch["diffraction"])
    target = batch["realspace"]
    support = target.abs() / target.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-7)
    support = support >= config["training"]["support_threshold"]
    loss = absolute_fit_loss(output.refined_object, target, support)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite training loss")
    loss.backward()
    clip = config["training"]["gradient_clip"]
    if clip > 0:
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), clip)
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def fit_metrics(model, raw, config, device):
    model.eval()
    batch = prepare_batch(raw, config, device)
    target = batch["realspace"]
    prediction = model(batch["input"], batch["diffraction"]).refined_object
    support = target.abs() / target.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-7)
    support = support >= config["training"]["support_threshold"]
    objective = absolute_fit_loss(prediction, target, support)
    shape_only = ambiguity_aware_component_mae(prediction, target, support, spatial_dims=2).loss
    amplitude = (prediction.abs() - target.abs()).abs().mean() / target.abs().mean().clamp_min(1e-7)
    peak_ratio = prediction.abs().amax() / target.abs().amax().clamp_min(1e-7)
    return {"absolute_fit_loss": float(objective), "shape_only_loss": float(shape_only),
            "relative_amplitude_mae": float(amplitude), "peak_amplitude_ratio": float(peak_ratio)}


@torch.no_grad()
def measured_amplitude_error(model, raw, config, device):
    """Scale-fitted Fourier amplitude relative L2 error on processed input."""
    model.eval()
    batch = prepare_batch(raw, config, device)
    result = model(batch["input"], batch["diffraction"])
    predicted = farfield_modulus_from_realspace(result.refined_object, spatial_dims=2)
    measured = batch["diffraction"]
    numerator = (predicted * measured).sum()
    denominator = predicted.square().sum().clamp_min(1e-20)
    scale = numerator / denominator
    return float(torch.linalg.vector_norm(scale * predicted - measured) / torch.linalg.vector_norm(measured).clamp_min(1e-20))


@torch.no_grad()
def reconstructed_object(model, raw, config, device):
    model.eval()
    batch = prepare_batch(raw, config, device)
    result = model(batch["input"], batch["diffraction"])
    return result.refined_object[0, 0].detach().cpu().numpy().copy()


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def save_comparison(path, target, before, after):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), constrained_layout=True)
    amplitudes = [np.abs(x) for x in (target, before, after)]
    peak = max(float(x.max()) for x in amplitudes)
    for ax, title, image in zip(axes, ("Pseudo-label", "Original model", "Single-sample fit"), amplitudes):
        ax.imshow(image, cmap="magma", vmin=0, vmax=peak)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu explicitly")
    device = torch.device(args.device)
    source = load_checkpoint(args.source_checkpoint)
    if source.get("stage") != "refiner":
        raise ValueError("Source checkpoint must be a refiner checkpoint")
    config = load_config(args.data_config)
    for key in ("diffraction_kind", "input_log_data", "center_targets"):
        if config[key] != source["config"][key]:
            raise ValueError(f"Checkpoint and data configuration differ on {key}")
    config["training"] = source["config"]["training"]
    train_set = build_dataset(config, "train")
    val_set = build_dataset(config, "val")
    if len(train_set) != 1 or len(val_set) != 1:
        raise ValueError("Expected exactly one training and one validation sample")
    train_loader = build_loader(train_set, 1, 0, True, device)
    val_loader = build_loader(val_set, 1, 0, False, device)
    train_raw = next(iter(train_loader))
    val_raw = next(iter(val_loader))
    model = make_model("refiner", config, checkpoint=source).to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)

    run_dir = Path(args.runs_dir).expanduser().resolve() / args.run_name
    output_dir = Path(args.checkpoint_root).expanduser().resolve() / args.run_name
    if (output_dir.exists() and any(output_dir.iterdir())) or (run_dir.exists() and any(run_dir.iterdir())):
        raise FileExistsError("Run name already exists; provide a new --run-name")
    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, force=True, format="%(asctime)s | %(levelname)s | %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "console.log", encoding="utf-8")])

    baseline = fit_metrics(model, val_raw, config, device)
    baseline_loss = baseline["absolute_fit_loss"]
    baseline_modulus = measured_amplitude_error(model, val_raw, config, device)
    if not np.isfinite(baseline_loss) or not np.isfinite(baseline_modulus):
        raise ValueError("Nonfinite baseline metric")
    best_loss, best_step = baseline_loss, 0
    best_state = cpu_state(model)
    baseline_image = reconstructed_object(model, val_raw, config, device)
    history = []

    def checkpoint(step, state):
        return {"spatial_dims": 2, "stage": "refiner", "epoch": step, "config": config,
                "model_state_dict": state, "best_val_loss": best_loss,
                "history": history, "data_manifests": {"train": train_set.manifest, "val": val_set.manifest},
                "transfer_source_checkpoint": str(Path(args.source_checkpoint).resolve()),
                "baseline_loss": baseline_loss, "baseline_modulus_rel_l2": baseline_modulus,
                "single_sample_pseudolabel": True}

    LOGGER.info("Baseline | absolute_fit_loss=%.6f | shape_only_loss=%.6f | peak_ratio=%.6f | diffraction_rel_l2=%.6f", baseline_loss, baseline["shape_only_loss"], baseline["peak_amplitude_ratio"], baseline_modulus)
    LOGGER.info("Fixed BN and dropout; phase frozen; train/val are identical; selecting by scale-sensitive complex fit")
    for step in range(1, args.steps + 1):
        train_loss = train_one(model, train_raw, config, device, optimizer)
        current = fit_metrics(model, val_raw, config, device)
        current_loss = current["absolute_fit_loss"]
        current_modulus = measured_amplitude_error(model, val_raw, config, device) if step % 10 == 0 or step == 1 else float("nan")
        if not np.isfinite(current_loss):
            raise ValueError("Nonfinite evaluation metric")
        selected = current_loss < best_loss
        if selected:
            best_loss, best_step = current_loss, step
            best_state = cpu_state(model)
        history.append({"step": step, "train_loss": train_loss, **current,
                        "diffraction_rel_l2": current_modulus if np.isfinite(current_modulus) else None,
                        "selected": selected,
                        "lr": optimizer.param_groups[0]["lr"]})
        if step % 10 == 0 or step == 1 or step == args.steps:
            LOGGER.info("Step=%d/%d | absolute_fit_loss=%.6f | shape_only=%.6f | amplitude_mae=%.6f | peak_ratio=%.6f | diffraction_rel_l2=%.6f | best_step=%d", step, args.steps, current_loss, current["shape_only_loss"], current["relative_amplitude_mae"], current["peak_amplitude_ratio"], current_modulus, best_step)
        if step - best_step >= 50:
            LOGGER.info("Stopping after 50 steps without improvement")
            break
    model.load_state_dict(best_state, strict=True)
    fitted_image = reconstructed_object(model, val_raw, config, device)
    selected = fit_metrics(model, val_raw, config, device)
    selected_modulus = measured_amplitude_error(model, val_raw, config, device)
    torch.save(checkpoint(best_step, best_state), output_dir / "checkpoint_best.pt")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf-8")
    target_image = prepare_batch(val_raw, config, device)["realspace"][0, 0].detach().cpu().numpy()
    save_comparison(run_dir / "comparison.png", target_image, baseline_image, fitted_image)
    np.savez_compressed(run_dir / "reconstructions.npz", pseudo_target=target_image,
                        original=baseline_image, fitted=fitted_image)
    (run_dir / "metrics.json").write_text(json.dumps({
        "best_step": best_step,
        "original": baseline,
        "fitted": selected,
        "diffraction_rel_l2_original": baseline_modulus,
        "diffraction_rel_l2_fitted": selected_modulus,
        "same_sample_for_training_and_selection": True,
        "target_is_uncalibrated_color_derived_pseudolabel": True,
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    LOGGER.info("DONE | best_step=%d | absolute_fit_loss=%.6f -> %.6f | peak_ratio=%.6f -> %.6f | diffraction_rel_l2=%.6f -> %.6f | checkpoint=%s", best_step, baseline_loss, best_loss, baseline["peak_amplitude_ratio"], selected["peak_amplitude_ratio"], baseline_modulus, selected_modulus, output_dir / "checkpoint_best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
