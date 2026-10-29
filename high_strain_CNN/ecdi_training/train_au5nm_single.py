"""Single-sample transfer fine-tuning for the Au5nm TIFF-derived pair.

This entry point initializes model weights from an existing refiner checkpoint,
starts a fresh optimizer, and writes a new checkpoint directory. It deliberately
does not compare dataset hashes because this is transfer to a new dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from pytorch_autophasenn.management import runtime_manifest

from .data import build_dataset, load_config
from .train import (
    LOGGER,
    PROJECT_DIR,
    build_loader,
    load_checkpoint,
    make_model,
    run_epoch,
)


DEFAULT_SOURCE = (
    "/data_ssd/oyys/high_strain_cnn/pipeline_output_2d/"
    "ecdi_2d512_20261014_224534_refiner/checkpoint_best.pt"
)
DEFAULT_DATA_CONFIG = "/data_ssd/oyys/eCDI/external_test/au5nm/finetune_data/config.json"
DEFAULT_CHECKPOINT_ROOT = "/data_ssd/oyys/high_strain_cnn/pipeline_output_2d"
DEFAULT_RUNS_DIR = PROJECT_DIR / "artifacts" / "training" / "ecdi_training"
DEFAULT_TUNED = (
    "/data_ssd/oyys/high_strain_cnn/pipeline_output_2d/"
    "au5nm_single_ft_20261027/checkpoint_best.pt"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", default=DEFAULT_SOURCE)
    parser.add_argument("--diagnose", action="store_true", help="Compare the source and tuned model, and isolate refiner BatchNorm buffer changes without training.")
    parser.add_argument("--tuned-checkpoint", default=DEFAULT_TUNED)
    parser.add_argument("--data-config", default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="au5nm_single_ft_20261027")
    parser.add_argument("--checkpoint-root", default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("epochs/batch-size must be positive and num-workers nonnegative.")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive.")
    if Path(args.run_name).name != args.run_name:
        parser.error("run-name must be a single directory name.")
    return args


def diagnose_batchnorm(source, config, val_loader, device, tuned_path):
    """Compare the same pair in evaluation mode, swapping only refiner BN buffers."""
    tuned = load_checkpoint(tuned_path)
    if tuned.get("stage") != "refiner":
        raise ValueError("tuned-checkpoint must be a refiner checkpoint.")
    for key in ("diffraction_kind", "input_log_data", "center_targets"):
        if config[key] != tuned["config"][key]:
            raise ValueError(f"Tuned checkpoint and data config differ on {key}.")
    before = source["model_state_dict"]
    after = tuned["model_state_dict"]
    if set(before) != set(after):
        raise ValueError("The checkpoints have different model state keys.")
    bn_keys = [
        key for key in before
        if key.startswith("refiner.")
        and key.endswith((".running_mean", ".running_var", ".num_batches_tracked"))
    ]
    if not bn_keys:
        raise ValueError("No refiner BatchNorm buffers found in the checkpoints.")
    changed_bn = sum(not torch.equal(before[key], after[key]) for key in bn_keys)
    phase_keys = [key for key in before if key.startswith("phase_model.")]
    changed_phase = sum(not torch.equal(before[key], after[key]) for key in phase_keys)
    model = make_model("refiner", config, checkpoint=source).to(device)

    def score():
        return run_epoch(model, val_loader, config=config, stage="refiner", device=device)["loss"]

    baseline_loss = score()
    model.load_state_dict(after, strict=True)
    tuned_loss = score()
    restored = dict(after)
    for key in bn_keys:
        restored[key] = before[key]
    model.load_state_dict(restored, strict=True)
    restored_loss = score()
    print(f"原模型 loss={baseline_loss:.6f}", flush=True)
    print(f"微调模型 loss={tuned_loss:.6f}", flush=True)
    print(f"微调模型恢复原始BN统计量 loss={restored_loss:.6f}", flush=True)
    print(f"refiner BN统计量变化={changed_bn}/{len(bn_keys)} | phase状态变化={changed_phase}/{len(phase_keys)}", flush=True)
    print("说明：三个数值针对同一张图及近似标签，BN恢复只用于诊断，没有训练或保存权重。", flush=True)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; choose --device cpu explicitly.")
    device = torch.device(args.device)

    source = load_checkpoint(args.source_checkpoint)
    if source.get("stage") != "refiner":
        raise ValueError("source-checkpoint must be a trained refiner checkpoint.")

    config = load_config(args.data_config)
    source_config = source["config"]
    for key in ("diffraction_kind", "input_log_data", "center_targets"):
        if config[key] != source_config[key]:
            raise ValueError(f"Source checkpoint and data config differ on {key}.")
    # Preserve source architecture/hyperparameter definitions while using the
    # new dataset root and paths from the supplied NPY configuration.
    config["training"] = source_config["training"]
    training = config["training"]["refiner"]

    train_set = build_dataset(config, "train")
    val_set = build_dataset(config, "val")
    if len(train_set) != 1 or len(val_set) != 1:
        raise ValueError(
            "This entry point is for exactly one paired sample in train and val."
        )
    train_loader = build_loader(
        train_set, args.batch_size, args.num_workers, True, device
    )
    val_loader = build_loader(
        val_set, args.batch_size, args.num_workers, False, device
    )

    if args.diagnose:
        diagnose_batchnorm(source, config, val_loader, device, args.tuned_checkpoint)
        return 0

    model = make_model("refiner", config, checkpoint=source).to(device)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    if training["scheduler"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5, min_lr=1e-7
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=min(args.learning_rate, 1e-7)
        )

    run_dir = Path(args.runs_dir).expanduser().resolve() / args.run_name
    output_dir = Path(args.checkpoint_root).expanduser().resolve() / args.run_name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output already exists; choose a new --run-name: {output_dir}"
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

    manifest = {
        "source_checkpoint": str(Path(args.source_checkpoint).resolve()),
        "source_epoch": source.get("epoch"),
        "transfer_initialization": True,
        "optimizer_state_reused": False,
        "single_sample": True,
        "train_and_val_are_same_sample": True,
        "config": config,
        "args": vars(args),
        "runtime": runtime_manifest(device),
        "train": train_set.manifest,
        "val": val_set.manifest,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }
    (run_dir / "config.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    LOGGER.info(
        "Single-sample transfer | source epoch=%s | fresh optimizer | epochs=%d | lr=%.3g",
        source.get("epoch"),
        args.epochs,
        args.learning_rate,
    )
    LOGGER.info(
        "Train and validation use the same sample; metrics measure fitting only, not generalization."
    )
    LOGGER.info("Run=%s | checkpoints=%s", run_dir, output_dir)

    best = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        LOGGER.info(
            "Starting epoch=%03d/%03d | lr=%.3g",
            epoch,
            args.epochs,
            optimizer.param_groups[0]["lr"],
        )
        common = {"config": config, "stage": "refiner", "device": device}
        train = run_epoch(model, train_loader, optimizer=optimizer, **common)
        val = run_epoch(model, val_loader, **common)
        if training["scheduler"] == "plateau":
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
            "stage": "refiner",
            "epoch": epoch,
            "config": config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best,
            "history": history,
            "data_manifests": {"train": train_set.manifest, "val": val_set.manifest},
            "transfer_source_checkpoint": str(Path(args.source_checkpoint).resolve()),
        }
        torch.save(saved, output_dir / "checkpoint_last.pt")
        if improved:
            torch.save(saved, output_dir / "checkpoint_best.pt")
        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2, allow_nan=False), encoding="utf-8"
        )
        LOGGER.info(
            "Epoch=%d/%d | train_loss=%.6g | val_loss=%.6g | best=%.6g | lr=%.3g",
            epoch,
            args.epochs,
            train["loss"],
            val["loss"],
            best,
            optimizer.param_groups[0]["lr"],
        )
    LOGGER.info("DONE | best checkpoint=%s", output_dir / "checkpoint_best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
