"""Train the 512-square phase U-Net or its frozen-U-Net MoMamba second stage."""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from datetime import datetime
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pytorch_autophasenn.losses import phase_retrieval_wca_loss
from pytorch_autophasenn.management import runtime_manifest
from pytorch_autophasenn.momamba_refiner import (
    ambiguity_aware_component_mae,
    diffraction_modulus_mae,
)
from .data import build_dataset, load_config, prepare_batch
from .model import (
    HighStrainPhaseUNet2D,
    ComplexMoMambaRefiner2D,
    HighStrainMoMambaCascade2D,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
TRAINING_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = TRAINING_DIR / "configs" / "diffraction_2d.json"
DEFAULT_CHECKPOINT_ROOT = "/data_ssd/oyys/high_strain_cnn/pipeline_output_2d"
LOGGER = logging.getLogger("high_strain.2d")


def load_checkpoint(path: str | Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("spatial_dims") != 2:
        raise ValueError(
            "A native 2D checkpoint is required; 3D weights cannot be loaded directly."
        )
    return checkpoint


def make_model(
    stage: str,
    config: dict,
    *,
    phase_checkpoint: dict | None = None,
    checkpoint: dict | None = None,
    mixer_factory=None,
):
    training = config["training"]
    phase = HighStrainPhaseUNet2D(
        training["model_variant"], mamba_factory=mixer_factory
    )
    if stage == "phase":
        model = phase
    else:
        if phase_checkpoint is not None:
            if phase_checkpoint["stage"] != "phase":
                raise ValueError("--phase-checkpoint must contain a phase-stage model.")
            source_config = phase_checkpoint["config"]
            for key in ("input_log_data", "center_targets", "diffraction_kind"):
                if source_config[key] != config[key]:
                    raise ValueError(
                        f"Phase checkpoint and refiner data disagree on {key}."
                    )
            phase.load_state_dict(phase_checkpoint["model_state_dict"], strict=True)
        elif checkpoint is None:
            raise ValueError(
                "A trained 2D phase checkpoint is required for the refiner stage."
            )
        refiner_keys = (
            "base_channels",
            "num_experts",
            "top_k",
            "router_heads",
            "router_dropout",
            "position_dropout",
            "codec_dropout",
            "d_state",
            "d_conv",
            "expand",
        )
        refiner = ComplexMoMambaRefiner2D(
            **{key: training[key] for key in refiner_keys}, mixer_factory=mixer_factory
        )
        model = HighStrainMoMambaCascade2D(phase, refiner, freeze_phase_model=True)
    if checkpoint is not None:
        if checkpoint["stage"] != stage:
            raise ValueError("Checkpoint training stage does not match --stage.")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model


def build_loader(dataset, batch_size: int, workers: int, training: bool, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        drop_last=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def verify_data_identity(checkpoint: dict | None, manifest: dict):
    if checkpoint is None:
        return
    previous = checkpoint.get("data_manifests", {}).get("train", {})
    for key in ("config_sha256", "manifest_sha256"):
        if key in previous and previous[key] != manifest.get(key):
            raise ValueError(
                f"Checkpoint and current dataset differ in {key}; use the same generation and split."
            )


def run_epoch(
    model,
    loader,
    *,
    config,
    stage,
    device,
    optimizer=None,
    max_batches=0,
    print_freq=50,
) -> dict:
    training = optimizer is not None
    model.train(training)
    count = min(len(loader), max_batches or len(loader))
    totals, samples = {}, 0
    started = time.monotonic()
    with torch.enable_grad() if training else torch.no_grad():
        for index, raw_batch in enumerate(islice(loader, count), 1):
            batch = prepare_batch(raw_batch, config, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            if stage == "phase":
                phase = model(batch["input"])
                loss = phase_retrieval_wca_loss(
                    phase, batch["target_phase"], batch["input"]
                )
                values = {"loss": loss, "phase_wca": loss}
            else:
                output = model(batch["input"], batch["diffraction"])
                target = batch["realspace"]
                support = target.abs() / target.abs().amax(
                    dim=(-2, -1), keepdim=True
                ).clamp_min(1e-7)
                support = support >= config["training"]["support_threshold"]
                obj = ambiguity_aware_component_mae(
                    output.refined_object, target, support, spatial_dims=2
                )
                physics_weight = config["training"]["physics_loss_weight"]
                physics = (
                    diffraction_modulus_mae(
                        output.proposal_object,
                        batch["diffraction"],
                        spatial_dims=2,
                    )
                    if physics_weight > 0 or not training
                    else obj.loss.new_zeros(())
                )
                loss = obj.loss + physics_weight * physics
                values = {
                    "loss": loss,
                    "object_mae": obj.loss,
                    "physics_mae": physics,
                    "real_mae": obj.real_mae,
                    "imag_mae": obj.imag_mae,
                }
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss; check data and optimizer settings.")
            if training:
                loss.backward()
                clip = config["training"]["gradient_clip"]
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        (p for p in model.parameters() if p.requires_grad), clip
                    )
                optimizer.step()
            size = batch["input"].shape[0]
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach()) * size
            samples += size
            if index % print_freq == 0 or index == count:
                elapsed = time.monotonic() - started
                LOGGER.info(
                    "%s | batch=%d/%d | loss=%.6g | elapsed=%.1fs | eta=%.1fs",
                    "TRAIN" if training else "VAL",
                    index,
                    count,
                    float(loss.detach()),
                    elapsed,
                    elapsed / index * (count - index),
                )
    return {
        **{k: v / samples for k, v in totals.items()},
        "samples": samples,
        "elapsed_seconds": time.monotonic() - started,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("phase", "refiner"), required=True)
    parser.add_argument("--data-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--phase-checkpoint", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-samples-train", type=int, default=0)
    parser.add_argument("--num-samples-val", type=int, default=0)
    parser.add_argument("--max-batches-per-epoch", type=int, default=0)
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--checkpoint-root", default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument(
        "--runs-dir",
        default=str(PROJECT_DIR / "artifacts" / "training" / "ecdi_training"),
    )
    args = parser.parse_args()
    if (
        min(args.batch_size, args.print_freq) < 1
        or min(
            args.num_workers,
            args.max_batches_per_epoch,
            args.num_samples_train,
            args.num_samples_val,
        )
        < 0
    ):
        parser.error("Batch/frequency must be positive and workers/counts nonnegative.")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive.")
    if args.learning_rate is not None and (
        args.learning_rate <= 0 or not np.isfinite(args.learning_rate)
    ):
        parser.error("--learning-rate must be finite and positive.")
    if args.stage == "refiner" and not (args.phase_checkpoint or args.resume):
        parser.error("Refiner training requires --phase-checkpoint or --resume.")
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; use --device cpu explicitly for a CPU run."
        )
    device = torch.device(args.device)
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    config = load_config(args.data_config)
    if checkpoint is not None:
        # Architecture and preprocessing resume from their saved definitions.
        config["training"] = checkpoint["config"]["training"]
        for key in ("diffraction_kind", "input_log_data", "center_targets"):
            if config[key] != checkpoint["config"][key]:
                raise ValueError(f"Resume data config changed {key}.")
    if args.data_dir:
        config["root"] = str(Path(args.data_dir).expanduser().resolve())
    settings = config["training"][args.stage]
    epochs = args.epochs or settings["epochs"]
    learning_rate = args.learning_rate or settings["learning_rate"]
    name = args.run_name or f"{args.stage}_2d512_{datetime.now():%Y%m%d_%H%M%S}"
    if Path(name).name != name:
        raise ValueError("--run-name must be a single directory name.")
    run_dir = Path(args.runs_dir) / name
    output_dir = Path(args.checkpoint_root) / name
    if (output_dir / "checkpoint_last.pt").exists() and not args.resume:
        raise FileExistsError(
            "Run already has checkpoints; choose a new --run-name or use --resume."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        force=True,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "console.log", encoding="utf-8"),
        ],
    )
    train_set = build_dataset(config, "train", args.num_samples_train)
    val_set = build_dataset(config, "val", args.num_samples_val)
    train_loader = build_loader(
        train_set, args.batch_size, args.num_workers, True, device
    )
    val_loader = build_loader(val_set, args.batch_size, args.num_workers, False, device)
    phase_checkpoint = (
        load_checkpoint(args.phase_checkpoint) if args.phase_checkpoint else None
    )
    verify_data_identity(checkpoint or phase_checkpoint, train_set.manifest)
    model = make_model(
        args.stage, config, phase_checkpoint=phase_checkpoint, checkpoint=checkpoint
    ).to(device)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=learning_rate
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5, min_lr=1e-6
        )
        if settings["scheduler"] == "plateau"
        else torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
    )
    first_epoch, best, history = 1, float("inf"), []
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        first_epoch, best, history = (
            checkpoint["epoch"] + 1,
            checkpoint["best_val_loss"],
            checkpoint["history"],
        )
        if args.learning_rate is not None:
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            scheduler.base_lrs = [learning_rate] * len(optimizer.param_groups)
        if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            scheduler.T_max = epochs
    if epochs < first_epoch:
        raise ValueError("--epochs is the final epoch and must exceed the saved epoch.")
    manifest = {
        "config": config,
        "args": vars(args),
        "runtime": runtime_manifest(device),
        "train": train_set.manifest,
        "val": val_set.manifest,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    }
    (run_dir / "config.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    LOGGER.info(
        "Stage=%s | shape=512x512 | batch=%d | parameters=%s | trainable=%s | dtype=float32/complex64",
        args.stage,
        args.batch_size,
        f'{manifest["parameters"]:,}',
        f'{manifest["trainable_parameters"]:,}',
    )
    LOGGER.info("Run=%s | checkpoints=%s", run_dir, output_dir)
    for epoch in range(first_epoch, epochs + 1):
        LOGGER.info(
            "Starting epoch=%03d/%03d | lr=%.3g",
            epoch,
            epochs,
            optimizer.param_groups[0]["lr"],
        )
        common = dict(
            config=config,
            stage=args.stage,
            device=device,
            max_batches=args.max_batches_per_epoch,
            print_freq=args.print_freq,
        )
        train = run_epoch(model, train_loader, optimizer=optimizer, **common)
        val = run_epoch(model, val_loader, **common)
        if settings["scheduler"] == "plateau":
            scheduler.step(val["loss"])
        else:
            scheduler.step()
        improved = val["loss"] < best
        best = min(best, val["loss"])
        history.append(
            {
                "epoch": epoch,
                "train": train,
                "val": val,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        saved = {
            "spatial_dims": 2,
            "stage": args.stage,
            "epoch": epoch,
            "config": config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best,
            "history": history,
            "data_manifests": {"train": train_set.manifest, "val": val_set.manifest},
        }
        torch.save(saved, output_dir / "checkpoint_last.pt")
        if improved:
            torch.save(saved, output_dir / "checkpoint_best.pt")
        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2, allow_nan=False), encoding="utf-8"
        )
        LOGGER.info(
            "Epoch=%d/%d | train=%.6g | val=%.6g | best=%.6g | lr=%.3g",
            epoch,
            epochs,
            train["loss"],
            val["loss"],
            best,
            optimizer.param_groups[0]["lr"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
