"""Train a sparse-MoMamba real-space refiner behind a phase-retrieval U-Net."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .author_data import AuthorNPZPhaseDataset, initialize_data_worker
from .management import runtime_manifest
from .model import MODEL_VARIANTS, HighStrainPhaseUNet, infer_model_variant
from .momamba_refiner import (
    ComplexMoMambaRefiner,
    HighStrainMoMambaCascade,
    ambiguity_aware_component_mae,
    count_trainable_parameters,
    diffraction_modulus_mae,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = PROJECT_DIR / "artifacts" / "training" / "pytorch_realspace_momamba"
DEFAULT_CHECKPOINT_ROOT = Path(
    "/data_ssd/oyys/autophasenn/autophasenn_pipeline_output/high_strain_cnn_refiner"
)
LOGGER = logging.getLogger("high_strain.train_refiner")


def configure_logging() -> None:
    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    LOGGER.addHandler(handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="/data_ssd/oyys/high_strain_cnn/dataset",
        help="Author-generated compact NPZ dataset root.",
    )
    parser.add_argument("--phase-checkpoint", default="")
    parser.add_argument(
        "--phase-model-variant",
        choices=("auto",) + MODEL_VARIANTS,
        default="auto",
    )
    parser.add_argument("--resume", default="")
    parser.add_argument("--num-samples-train", type=int, default=None)
    parser.add_argument("--num-samples-val", type=int, default=None)
    parser.add_argument("--shape", type=int, default=64)
    parser.add_argument("--author-min-oversampling", type=float, default=None)
    parser.add_argument(
        "--input-log-data",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--base-channels",
        type=int,
        default=4,
        help="Number of complex channels in the full-resolution refiner stage.",
    )
    parser.add_argument("--num-experts", type=int, nargs=4, default=(3, 3, 3, 3))
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--router-heads", type=int, default=2)
    parser.add_argument("--router-dropout", type=float, default=0.2)
    parser.add_argument("--position-dropout", type=float, default=0.1)
    parser.add_argument("--codec-dropout", type=float, default=0.1)
    parser.add_argument("--mamba-d-state", type=int, default=16)
    parser.add_argument("--mamba-d-conv", type=int, default=4)
    parser.add_argument("--mamba-expand", type=int, default=2)
    parser.add_argument(
        "--project-measured-modulus",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optional hard projection for ablation/inference; disabled in training by default.",
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--object-loss-scope",
        choices=("full", "support_balanced"),
        default="full",
        help="'full' follows Yu et al. Eq. (1); support_balanced is an ablation.",
    )
    parser.add_argument("--outside-support-weight", type=float, default=0.25)
    parser.add_argument(
        "--physics-loss-weight",
        type=float,
        default=0.0,
        help="Weight of optional Yu et al. Eq. (2) Fourier-modulus MAE.",
    )
    parser.add_argument("--load-balance-weight", type=float, default=0.0)
    parser.add_argument("--router-z-weight", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--lr-scheduler",
        choices=("none", "plateau", "cosine"),
        default="cosine",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--max-batches-per-epoch", type=int, default=0)
    args = parser.parse_args()

    if not args.phase_checkpoint and not args.resume:
        parser.error("Provide --phase-checkpoint for a new run, or --resume.")
    if min(args.epochs, args.batch_size, args.prefetch_factor) < 1:
        parser.error("Epochs, batch size, and prefetch factor must be positive.")
    if args.num_workers < 0 or args.max_batches_per_epoch < 0 or args.save_every < 0:
        parser.error("Workers, batch limit, and save interval must be nonnegative.")
    if args.shape % 16:
        parser.error("--shape must be divisible by 16.")
    if args.top_k < 1 or any(value < args.top_k for value in args.num_experts):
        parser.error("Each --num-experts value must be at least --top-k.")
    if args.base_channels < 1:
        parser.error("--base-channels must be positive.")
    if (
        not 0 <= args.router_dropout < 1
        or not 0 <= args.position_dropout < 1
        or not 0 <= args.codec_dropout < 1
    ):
        parser.error("Dropout values must be in [0, 1).")
    nonnegative = (
        args.weight_decay,
        args.outside_support_weight,
        args.physics_loss_weight,
        args.load_balance_weight,
        args.router_z_weight,
        args.gradient_clip,
    )
    if args.learning_rate <= 0 or min(nonnegative) < 0:
        parser.error(
            "Learning rate must be positive and loss/regularization values nonnegative."
        )
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def format_duration(seconds: float) -> str:
    total = max(int(round(seconds)), 0)
    days, remainder = divmod(total, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    value = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{days}d {value}" if days else value


def checkpoint_state(path: str | Path, device: torch.device) -> dict:
    checkpoint = torch.load(Path(path).expanduser(), map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint {path} is not a state dictionary container.")
    return checkpoint


def phase_state_dict(checkpoint: dict) -> dict[str, torch.Tensor]:
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, dict):
        raise ValueError("Phase checkpoint does not contain model_state_dict.")
    return state


def build_cascade(
    args: argparse.Namespace,
    device: torch.device,
    resume_checkpoint: dict | None,
) -> tuple[HighStrainMoMambaCascade, str]:
    if resume_checkpoint is not None:
        variant = str(resume_checkpoint["phase_model_variant"])
    else:
        source = checkpoint_state(args.phase_checkpoint, torch.device("cpu"))
        inferred = infer_model_variant(phase_state_dict(source))
        variant = (
            inferred if args.phase_model_variant == "auto" else args.phase_model_variant
        )
        if variant != inferred:
            raise ValueError(
                f"Phase checkpoint is {inferred!r}, not requested variant {variant!r}."
            )

    phase_model = HighStrainPhaseUNet(variant)
    refiner = ComplexMoMambaRefiner(
        base_channels=args.base_channels,
        num_experts=args.num_experts,
        top_k=args.top_k,
        router_heads=args.router_heads,
        router_dropout=args.router_dropout,
        position_dropout=args.position_dropout,
        codec_dropout=args.codec_dropout,
        d_state=args.mamba_d_state,
        d_conv=args.mamba_d_conv,
        expand=args.mamba_expand,
    )
    cascade = HighStrainMoMambaCascade(
        phase_model,
        refiner,
        freeze_phase_model=True,
        project_measured_modulus=args.project_measured_modulus,
    )
    if resume_checkpoint is not None:
        cascade.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
    else:
        phase_source = checkpoint_state(args.phase_checkpoint, torch.device("cpu"))
        phase_model.load_state_dict(phase_state_dict(phase_source), strict=True)
    return cascade.to(device), variant


def build_dataset(
    args: argparse.Namespace,
    split: str,
    num_samples: int | None,
) -> AuthorNPZPhaseDataset:
    return AuthorNPZPhaseDataset(
        args.data_dir,
        split,
        num_samples=num_samples,
        shape=(args.shape,) * 3,
        input_log_data=args.input_log_data,
        min_oversampling=args.author_min_oversampling,
        return_refinement_targets=True,
    )


def build_loader(
    dataset: AuthorNPZPhaseDataset,
    args: argparse.Namespace,
    device: torch.device,
    *,
    training: bool,
) -> DataLoader:
    kwargs: dict[str, object] = {
        "batch_size": args.batch_size,
        "shuffle": training,
        "drop_last": training,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers:
        kwargs.update(
            prefetch_factor=args.prefetch_factor,
            worker_init_fn=initialize_data_worker,
            multiprocessing_context="spawn",
        )
    loader = DataLoader(dataset, **kwargs)
    if not len(loader):
        raise ValueError(
            "No batches: reduce batch size or increase the selected split."
        )
    return loader


def run_epoch(
    model: HighStrainMoMambaCascade,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
    *,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float | int]:
    training = optimizer is not None
    model.train(training)
    split = "TRAIN" if training else "VAL"
    batch_limit = min(len(loader), args.max_batches_per_epoch or len(loader))
    totals = {
        "loss": 0.0,
        "object_loss": 0.0,
        "real_mae": 0.0,
        "imag_mae": 0.0,
        "physics_loss": 0.0,
        "load_balance_loss": 0.0,
        "router_z_loss": 0.0,
        "twin_fraction": 0.0,
    }
    samples = 0
    started = time.monotonic()
    batch_finished = started
    data_wait = 0.0
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for batch_index, batch in enumerate(islice(loader, batch_limit), start=1):
            data_wait += time.monotonic() - batch_finished
            model_input = batch["input"].to(device, non_blocking=True).float()
            modulus = batch["diffraction"].to(device, non_blocking=True).float()
            target = batch["realspace"].to(device, non_blocking=True)
            support = batch["support"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            output = model(model_input, modulus)
            object_loss = ambiguity_aware_component_mae(
                output.refined_object,
                target,
                support,
                outside_weight=args.outside_support_weight,
                loss_scope=args.object_loss_scope,
            )
            if args.physics_loss_weight > 0:
                physics_loss = diffraction_modulus_mae(
                    output.proposal_object,
                    modulus,
                    object_loss.selected_support,
                )
            elif not training:
                with torch.no_grad():
                    physics_loss = diffraction_modulus_mae(
                        output.proposal_object.detach(),
                        modulus,
                        object_loss.selected_support,
                    )
            else:
                physics_loss = object_loss.loss.new_zeros(())
            loss = (
                object_loss.loss
                + args.physics_loss_weight * physics_loss
                + args.load_balance_weight * output.routing_losses.load_balance
                + args.router_z_weight * output.routing_losses.router_z
            )
            if training:
                loss.backward()
                if args.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.refiner.parameters(),
                        max_norm=args.gradient_clip,
                    )
                optimizer.step()

            batch_size = model_input.shape[0]
            values = {
                "loss": loss,
                "object_loss": object_loss.loss,
                "real_mae": object_loss.real_mae,
                "imag_mae": object_loss.imag_mae,
                "physics_loss": physics_loss,
                "load_balance_loss": output.routing_losses.load_balance,
                "router_z_loss": output.routing_losses.router_z,
                "twin_fraction": object_loss.twin_fraction,
            }
            for name, value in values.items():
                totals[name] += float(value.detach()) * batch_size
            samples += batch_size

            if batch_index % max(args.print_freq, 1) == 0 or batch_index == batch_limit:
                elapsed = time.monotonic() - started
                remaining = elapsed / batch_index * (batch_limit - batch_index)
                LOGGER.info(
                    "%s | epoch=%03d/%03d | batch=%05d/%05d | total=%.5e | "
                    "object=%.5e | physics=%.5e | elapsed=%s | eta=%s | data_wait=%.2fs",
                    split,
                    epoch,
                    args.epochs,
                    batch_index,
                    batch_limit,
                    float(loss.detach()),
                    float(object_loss.loss.detach()),
                    float(physics_loss.detach()),
                    format_duration(elapsed),
                    format_duration(remaining),
                    data_wait,
                )
            batch_finished = time.monotonic()

    return {
        **{name: value / max(samples, 1) for name, value in totals.items()},
        "samples": samples,
        "batches": batch_limit,
        "elapsed_seconds": time.monotonic() - started,
        "data_wait_seconds": data_wait,
    }


def save_checkpoint(
    path: Path,
    model: HighStrainMoMambaCascade,
    optimizer: torch.optim.Optimizer,
    scheduler: (
        torch.optim.lr_scheduler.LRScheduler
        | torch.optim.lr_scheduler.ReduceLROnPlateau
        | None
    ),
    epoch: int,
    best_val_loss: float,
    history: dict[str, list[dict]],
    args: argparse.Namespace,
    phase_model_variant: str,
    run_manifest: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "phase_model_variant": phase_model_variant,
            "trainable_parameter_count": count_trainable_parameters(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_val_loss": best_val_loss,
            "history": history,
            "args": vars(args),
            "run_manifest": run_manifest,
        },
        path,
    )


def main() -> None:
    from torch.utils.tensorboard import SummaryWriter

    configure_logging()
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    resume_checkpoint = checkpoint_state(args.resume, device) if args.resume else None

    train_dataset = build_dataset(args, "train", args.num_samples_train)
    val_dataset = build_dataset(args, "val", args.num_samples_val)
    if (
        train_dataset.manifest["manifest_sha256"]
        != val_dataset.manifest["manifest_sha256"]
    ):
        raise ValueError("Dataset manifest changed while initializing splits.")
    args.num_samples_train = len(train_dataset)
    args.num_samples_val = len(val_dataset)
    model, phase_variant = build_cascade(args, device, resume_checkpoint)

    if not args.run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = (
            f"{timestamp}_realspace_momamba_{phase_variant}_bs-{args.batch_size}"
            f"_lr-{args.learning_rate:g}_seed-{args.seed}"
        )
    run_dir = Path(args.runs_dir).expanduser() / args.run_name
    checkpoint_dir = Path(args.checkpoint_root).expanduser() / args.run_name
    tensorboard_dir = run_dir / "tensorboard"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir = str(run_dir.resolve())
    args.checkpoint_dir = str(checkpoint_dir.resolve())

    run_manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime_manifest(device),
        "data": {
            "format": "author_npz_compact",
            "train": train_dataset.manifest,
            "val": val_dataset.manifest,
        },
        "pipeline": {
            "phase_model_variant": phase_variant,
            "phase_checkpoint": (
                str(Path(args.phase_checkpoint).expanduser().resolve())
                if args.phase_checkpoint
                else None
            ),
            "phase_model_frozen": True,
            "inverse_fft": "pytorch_autophasenn.reconstruction convention",
            "refiner": "complex-convolution sparse MoMamba 3D U-Net",
            "measured_modulus_projection": args.project_measured_modulus,
            "object_loss": {
                "name": "aligned real MAE + imaginary MAE",
                "paper_equation": 1,
                "scope": args.object_loss_scope,
                "alignment": "RMS scale, global phase, direct/twin selection",
            },
            "physics_loss": {
                "name": "normalized Fourier-modulus MAE",
                "paper_equation": 2,
                "weight": args.physics_loss_weight,
                "support": "stored synthetic ground-truth support",
            },
        },
        "training": vars(args),
    }
    (run_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2), encoding="utf-8"
    )

    train_loader = build_loader(train_dataset, args, device, training=True)
    val_loader = build_loader(val_dataset, args, device, training=False)
    optimizer = torch.optim.Adam(
        model.refiner.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5, min_lr=1e-6
        )
    elif args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-6,
        )
    else:
        scheduler = None
    history: dict[str, list[dict]] = {"train": [], "val": []}
    start_epoch = 1
    best_val_loss = float("inf")
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        if scheduler and resume_checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        history = resume_checkpoint.get("history", history)
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        best_val_loss = float(resume_checkpoint.get("best_val_loss", best_val_loss))
        LOGGER.info("Resumed %s at epoch %d", args.resume, start_epoch)

    LOGGER.info(
        "Run=%s | phase=%s (frozen) | refiner trainable parameters=%s | "
        "device=%s | train/val=%s/%s | checkpoints=%s",
        args.run_name,
        phase_variant,
        f"{count_trainable_parameters(model):,}",
        device,
        f"{len(train_dataset):,}",
        f"{len(val_dataset):,}",
        checkpoint_dir,
    )
    LOGGER.info(
        "MoM experts=%s | top-k=%d | router heads=%d | complex base channels=%d | "
        "position/router/codec dropout=%.2f/%.2f/%.2f | projection=%s | precision=float32",
        list(args.num_experts),
        args.top_k,
        args.router_heads,
        args.base_channels,
        args.position_dropout,
        args.router_dropout,
        args.codec_dropout,
        args.project_measured_modulus,
    )
    LOGGER.info(
        "Object loss=aligned component MAE (%s) | physics weight=%.3g | "
        "Fourier metric=%s",
        args.object_loss_scope,
        args.physics_loss_weight,
        "train+validation" if args.physics_loss_weight > 0 else "validation only",
    )
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    run_started = time.monotonic()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.monotonic()
        train_stats = run_epoch(
            model, train_loader, args, device, epoch, optimizer=optimizer
        )
        val_stats = run_epoch(model, val_loader, args, device, epoch, optimizer=None)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(float(val_stats["loss"]))
        elif scheduler:
            scheduler.step()
        history["train"].append({"epoch": epoch, **train_stats})
        history["val"].append({"epoch": epoch, **val_stats})
        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        for split, stats in (("train", train_stats), ("val", val_stats)):
            for name in (
                "loss",
                "object_loss",
                "real_mae",
                "imag_mae",
                "physics_loss",
                "load_balance_loss",
                "router_z_loss",
                "twin_fraction",
            ):
                writer.add_scalar(f"{split}/{name}", stats[name], epoch)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)
        writer.flush()

        improved = float(val_stats["loss"]) < best_val_loss
        if improved:
            best_val_loss = float(val_stats["loss"])
        save_checkpoint(
            checkpoint_dir / "checkpoint_last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_loss,
            history,
            args,
            phase_variant,
            run_manifest,
        )
        if improved:
            save_checkpoint(
                checkpoint_dir / "checkpoint_best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                history,
                args,
                phase_variant,
                run_manifest,
            )
        if args.save_every and epoch % args.save_every == 0:
            save_checkpoint(
                checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                history,
                args,
                phase_variant,
                run_manifest,
            )

        run_elapsed = time.monotonic() - run_started
        complete = epoch - start_epoch + 1
        remaining = run_elapsed / max(complete, 1) * max(args.epochs - epoch, 0)
        finish = datetime.now().astimezone() + timedelta(seconds=remaining)
        LOGGER.info(
            "Epoch %03d complete | train=%.5e | val=%.5e | best=%.5e | "
            "real=%.5e | imag=%.5e | lr=%.3e | epoch_time=%s | eta=%s | finish=%s",
            epoch,
            train_stats["loss"],
            val_stats["loss"],
            best_val_loss,
            val_stats["real_mae"],
            val_stats["imag_mae"],
            optimizer.param_groups[0]["lr"],
            format_duration(time.monotonic() - epoch_started),
            format_duration(remaining),
            finish.strftime("%Y-%m-%d %H:%M:%S %Z"),
        )

    writer.close()
    LOGGER.info("Training complete | best=%s", checkpoint_dir / "checkpoint_best.pt")


if __name__ == "__main__":
    main()
