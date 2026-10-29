"""Short controlled phase fine-tuning with an optional full-resolution skip."""

from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pytorch_autophasenn.management import runtime_manifest

from .data_v2 import build_dataset, load_config, prepare_batch
from .evaluate_phase_v4 import evaluate_phase, read_thickness, save_records
from .model_v4 import build_phase_model
from .train import load_checkpoint, verify_data_identity
from .train_v2 import run_epoch


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "ecdi_2d_phase_v4.json"
LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-phase-checkpoint", required=True)
    parser.add_argument("--data-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--architecture", choices=("baseline", "outer_skip_v4"))
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
    parser.add_argument("--checkpoint-root", default="/data_ssd/oyys/high_strain_cnn/pipeline_output_2d")
    parser.add_argument("--runs-dir", default=str(PROJECT_DIR / "artifacts" / "training" / "ecdi_training"))
    args = parser.parse_args()
    if min(args.batch_size, args.print_freq) < 1 or min(
        args.num_workers, args.num_samples_train, args.num_samples_val,
        args.max_batches_per_epoch,
    ) < 0:
        parser.error("Batch/frequency must be positive and counts nonnegative.")
    if args.epochs is not None and args.epochs < 1:
        parser.error("Epochs must be positive.")
    if args.learning_rate is not None and (
        not np.isfinite(args.learning_rate) or args.learning_rate <= 0
    ):
        parser.error("Learning rate must be finite and positive.")
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    source = load_checkpoint(args.init_phase_checkpoint)
    if source.get("stage") != "phase":
        raise ValueError("Initialization must use a phase-stage checkpoint.")
    config = load_config(args.data_config)
    for key in ("diffraction_kind", "input_log_data", "center_targets"):
        if config[key] != source["config"][key]:
            raise ValueError(f"Source checkpoint and current data differ on {key}.")
    if config["training"]["model_variant"] != source["config"]["training"]["model_variant"]:
        raise ValueError("Source and current base U-Net variants differ.")
    if args.architecture is not None:
        config["training"]["phase_architecture"] = args.architecture
    if args.data_dir:
        config["root"] = str(Path(args.data_dir).expanduser().resolve())
    settings = config["training"]["phase"]
    epochs = args.epochs if args.epochs is not None else settings["epochs"]
    learning_rate = args.learning_rate if args.learning_rate is not None else settings["learning_rate"]
    settings["epochs"] = epochs
    settings["learning_rate"] = learning_rate
    architecture = config["training"]["phase_architecture"]
    name = args.run_name or f"phase_{architecture}_{datetime.now():%Y%m%d_%H%M%S}"
    if Path(name).name != name:
        raise ValueError("Run name must be a single directory name.")
    run_dir = Path(args.runs_dir) / name
    checkpoint_dir = Path(args.checkpoint_root) / name
    if (checkpoint_dir / "checkpoint_best.pt").exists():
        raise FileExistsError("Run exists; choose a new run name.")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, force=True,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "console.log", encoding="utf-8")],
    )
    train_set = build_dataset(config, "train", args.num_samples_train)
    val_set = build_dataset(config, "val", args.num_samples_val, include_ideal=True)
    verify_data_identity(source, train_set.manifest)
    # Use a separate RNG so adding model layers cannot alter the training order.
    data_generator = torch.Generator().manual_seed(args.seed)
    common_loader = {
        "batch_size": args.batch_size, "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda", "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, generator=data_generator, **common_loader)
    val_loader = DataLoader(val_set, shuffle=False, **common_loader)
    thickness = read_thickness(val_set)
    source_architecture = source["config"]["training"].get("phase_architecture", "baseline")
    migrate = architecture == "outer_skip_v4" and source_architecture == "baseline"
    model = build_phase_model(config, checkpoint=source, migrate_baseline=migrate).to(device)
    if migrate:
        # Check migration with the actual checkpoint and actual server inputs
        # before spending any training time on the modified architecture.
        reference = build_phase_model(source["config"], checkpoint=source).to(device).eval()
        model.eval()
        with torch.no_grad():
            probe = prepare_batch(next(iter(val_loader)), config, device)["input"][:1]
            expected, actual = reference(probe), model(probe)
            maximum_error = float((expected - actual).abs().max())
            if not torch.allclose(expected, actual, rtol=1e-4, atol=1e-4):
                raise ValueError(f"Initial model migration changed phase output: max_error={maximum_error}.")
        LOGGER.info("Migration equivalence checked on server input | max_phase_error=%.6g", maximum_error)
        del reference, probe, expected, actual
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    manifest = {
        "experiment_version": config["experiment_version"], "config": config,
        "args": vars(args), "runtime": runtime_manifest(device),
        "train": train_set.manifest, "val": val_set.manifest,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    (run_dir / "config.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    LOGGER.info("Experiment=%s | architecture=%s | parameters=%s | batch=%d", config["experiment_version"], architecture, f'{manifest["parameters"]:,}', args.batch_size)
    LOGGER.info("Initialized from %s (epoch=%s); optimizer starts fresh", args.init_phase_checkpoint, source["epoch"])
    LOGGER.info("Loss unchanged: log-WCA + %g * reconstruction MAE", settings["reconstruction_loss_weight"])
    LOGGER.info("Run=%s | checkpoints=%s", run_dir, checkpoint_dir)

    initial, initial_records = evaluate_phase(
        model, val_loader, config, device, thickness, include_zero=True
    )
    best = initial["phase"]["all"]["complex_nrmse"]
    history = [{"epoch": 0, "train": None, "val": initial["phase"], "lr": learning_rate}]
    (run_dir / "metrics_initial.json").write_text(json.dumps(initial, indent=2, allow_nan=False), encoding="utf-8")
    save_records(run_dir / "validation_initial.csv", initial_records)
    save_records(run_dir / "validation_best.csv", [row for row in initial_records if row["stage"] == "phase"])
    (run_dir / "metrics_best.json").write_text(json.dumps({"phase": initial["phase"]}, indent=2, allow_nan=False), encoding="utf-8")

    def checkpoint_payload(epoch):
        return {
            "experiment_version": config["experiment_version"], "spatial_dims": 2,
            "stage": "phase", "epoch": epoch, "config": config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_metric": best, "checkpoint_selection_metric": "complex_nrmse",
            "history": history,
            "data_manifests": {"train": train_set.manifest, "val": val_set.manifest},
        }

    # The starting model is eligible for best, so unsuccessful fine-tuning
    # cannot silently replace it with an inferior trained checkpoint.
    torch.save(checkpoint_payload(0), checkpoint_dir / "checkpoint_best.pt")
    LOGGER.info("Epoch=0/%d | val_complex_nrmse=%.6f | val_ssim=%.6f | val_offaxis_nrmse=%.6f | zero_phase_nrmse=%.6f", epochs, best, initial["phase"]["all"]["real_amp_ssim"], initial["phase"]["all"]["offaxis_nrmse_9x9"], initial["zero_phase"]["all"]["complex_nrmse"])

    for epoch in range(1, epochs + 1):
        LOGGER.info("Starting epoch=%03d/%03d | lr=%.3g", epoch, epochs, optimizer.param_groups[0]["lr"])
        train = run_epoch(
            model, train_loader, config=config, stage="phase", device=device,
            optimizer=optimizer, max_batches=args.max_batches_per_epoch,
            print_freq=args.print_freq,
        )
        summary, records = evaluate_phase(model, val_loader, config, device, thickness)
        val = summary["phase"]
        scheduler.step()
        improved = val["all"]["complex_nrmse"] < best
        best = min(best, val["all"]["complex_nrmse"])
        history.append({"epoch": epoch, "train": train, "val": val, "lr": optimizer.param_groups[0]["lr"]})
        payload = checkpoint_payload(epoch)
        torch.save(payload, checkpoint_dir / "checkpoint_last.pt")
        if improved:
            torch.save(payload, checkpoint_dir / "checkpoint_best.pt")
            save_records(run_dir / "validation_best.csv", records)
            (run_dir / "metrics_best.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        (run_dir / "history_phase.json").write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf-8")
        LOGGER.info("Epoch=%d/%d | val_complex_nrmse=%.6f | best=%.6f | val_ssim=%.6f | val_phase_mae=%.6f | val_thick_nrmse=%.6f | val_offaxis_nrmse=%.6f | lr=%.3g", epoch, epochs, val["all"]["complex_nrmse"], best, val["all"]["real_amp_ssim"], val["all"]["real_phase_mae_true_support"], val["thick_ge_34a"].get("complex_nrmse", float("nan")), val["all"]["offaxis_nrmse_9x9"], optimizer.param_groups[0]["lr"])
    LOGGER.info("Training complete | best_complex_nrmse=%.6f", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
