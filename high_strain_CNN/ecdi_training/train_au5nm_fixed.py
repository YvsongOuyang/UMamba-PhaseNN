"""Au5nm single-sample transfer with fixed normalization statistics.

Uses the existing original refiner checkpoint. Train/validation reuse the same
uncalibrated pseudo-label; the reported losses therefore measure fit to that
one label, not verified real-space accuracy. The original checkpoint is an
eligible epoch-zero candidate. A candidate is accepted only when its label loss
improves and its processed diffraction-amplitude error does not exceed baseline.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from pytorch_autophasenn.momamba_refiner import (
    ambiguity_aware_component_mae,
    diffraction_modulus_mae,
)
from pytorch_autophasenn.reconstruction import farfield_modulus_from_realspace
from .data import build_dataset, load_config, prepare_batch
from .train import LOGGER, PROJECT_DIR, build_loader, load_checkpoint, make_model, run_epoch


SOURCE = "/data_ssd/oyys/high_strain_cnn/pipeline_output_2d/ecdi_2d512_20261014_224534_refiner/checkpoint_best.pt"
DATA = "/data_ssd/oyys/eCDI/external_test/au5nm/finetune_data/config.json"
CHECKPOINT_ROOT = "/data_ssd/oyys/high_strain_cnn/pipeline_output_2d"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-checkpoint", default=SOURCE)
    p.add_argument("--data-config", default=DATA)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--run-name", default="au5nm_fixed_bn_20261027")
    p.add_argument("--checkpoint-root", default=CHECKPOINT_ROOT)
    p.add_argument("--runs-dir", default=str(PROJECT_DIR / "artifacts" / "training" / "ecdi_training"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.epochs < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        p.error("epochs and learning-rate must be positive")
    if Path(args.run_name).name != args.run_name or args.run_name in ("", ".", ".."):
        p.error("run-name must be a single directory name")
    return args


def train_one(model, raw, config, device, optimizer):
    # eval() disables BN running-stat updates and dropout, but does not disable gradients.
    model.eval()
    batch = prepare_batch(raw, config, device)
    optimizer.zero_grad(set_to_none=True)
    output = model(batch["input"], batch["diffraction"])
    target = batch["realspace"]
    support = target.abs() / target.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-7)
    support = support >= config["training"]["support_threshold"]
    obj = ambiguity_aware_component_mae(output.refined_object, target, support, spatial_dims=2)
    weight = config["training"]["physics_loss_weight"]
    physics = diffraction_modulus_mae(output.proposal_object, batch["diffraction"], spatial_dims=2) if weight > 0 else obj.loss.new_zeros(())
    loss = obj.loss + weight * physics
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite training loss")
    loss.backward()
    clip = config["training"]["gradient_clip"]
    if clip > 0:
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), clip)
    optimizer.step()
    return float(loss.detach())


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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.1)

    run_dir = Path(args.runs_dir).expanduser().resolve() / args.run_name
    output_dir = Path(args.checkpoint_root).expanduser().resolve() / args.run_name
    if (output_dir.exists() and any(output_dir.iterdir())) or (run_dir.exists() and any(run_dir.iterdir())):
        raise FileExistsError("Run name already exists; provide a new --run-name")
    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, force=True, format="%(asctime)s | %(levelname)s | %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "console.log", encoding="utf-8")])

    def val_loss():
        return run_epoch(model, val_loader, config=config, stage="refiner", device=device)["loss"]

    baseline_loss = val_loss()
    baseline_modulus = measured_amplitude_error(model, val_raw, config, device)
    if not np.isfinite(baseline_loss) or not np.isfinite(baseline_modulus):
        raise ValueError("Nonfinite baseline metric")
    best_loss, best_epoch = baseline_loss, 0
    history = []

    def checkpoint(epoch):
        return {"spatial_dims": 2, "stage": "refiner", "epoch": epoch, "config": config,
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(), "best_val_loss": best_loss,
                "history": history, "data_manifests": {"train": train_set.manifest, "val": val_set.manifest},
                "transfer_source_checkpoint": str(Path(args.source_checkpoint).resolve()),
                "baseline_loss": baseline_loss, "baseline_modulus_rel_l2": baseline_modulus,
                "single_sample_pseudolabel": True}

    torch.save(checkpoint(0), output_dir / "checkpoint_best.pt")
    LOGGER.info("Baseline | pseudo_label_loss=%.6f | processed_diffraction_rel_l2=%.6f | epoch=0", baseline_loss, baseline_modulus)
    LOGGER.info("Fixed BN and dropout during optimization; phase frozen; train/val are the same sample; no independent accuracy estimate")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one(model, train_raw, config, device, optimizer)
        current_loss = val_loss()
        current_modulus = measured_amplitude_error(model, val_raw, config, device)
        if not np.isfinite(current_loss) or not np.isfinite(current_modulus):
            raise ValueError("Nonfinite evaluation metric")
        scheduler.step()
        accepted = current_loss < best_loss and current_modulus <= baseline_modulus + 1e-7
        if accepted:
            best_loss, best_epoch = current_loss, epoch
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": current_loss,
                        "processed_diffraction_rel_l2": current_modulus, "accepted": accepted,
                        "lr": optimizer.param_groups[0]["lr"]})
        if accepted:
            torch.save(checkpoint(epoch), output_dir / "checkpoint_best.pt")
        torch.save(checkpoint(epoch), output_dir / "checkpoint_last.pt")
        (run_dir / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf-8")
        LOGGER.info("Epoch=%d/%d | pseudo_label_loss=%.6f | diffraction_rel_l2=%.6f | best_epoch=%d | accepted=%s", epoch, args.epochs, current_loss, current_modulus, best_epoch, accepted)
    LOGGER.info("DONE | selected_epoch=%d | best_loss=%.6f | checkpoint=%s", best_epoch, best_loss, output_dir / "checkpoint_best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
