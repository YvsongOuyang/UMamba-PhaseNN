"""Train the PyTorch PhaseUNet on AutoPhaseNN memmap data."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pytorch_autophasenn.data import AutoPhaseNNPhaseDataset, reciprocal_phase_from_realspace
from pytorch_autophasenn.losses import phase_retrieval_wca_loss
from pytorch_autophasenn.management import (
    DEFAULT_DATA_CONFIG,
    build_data_manifest,
    load_data_config,
    require_data_files,
    runtime_manifest,
)
from pytorch_autophasenn.model import (
    DEFAULT_MODEL_VARIANT,
    MODEL_VARIANTS,
    REDUCED_BN_NO_OUTER_SKIP_VARIANT,
    HighStrainPhaseUNet,
    count_parameters,
    infer_model_variant,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = (
    PROJECT_DIR / "artifacts" / "training" / "pytorch_autophasenn"
)
DEFAULT_CHECKPOINT_ROOT = Path(
    "/data_ssd/oyys/autophasenn/autophasenn_pipeline_output/high_strain_cnn"
)
LOGGER = logging.getLogger("high_strain.train")


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
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--data-config", default=str(DEFAULT_DATA_CONFIG))
    bootstrap_args, _ = bootstrap.parse_known_args()
    data_config = load_data_config(bootstrap_args.data_config)
    train_config = data_config["splits"]["train"]
    val_config = data_config["splits"]["val"]
    configured_shape = tuple(int(size) for size in data_config["shape"])
    if len(set(configured_shape)) != 1:
        raise ValueError("HighStrainPhaseUNet requires a cubic data shape.")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default=str(DEFAULT_DATA_CONFIG))
    parser.add_argument("--data-dir", default=data_config["root"])
    parser.add_argument("--train-diff", default=train_config["diffraction"])
    parser.add_argument("--train-real", default=train_config["realspace"])
    parser.add_argument("--val-diff", default=val_config["diffraction"])
    parser.add_argument("--val-real", default=val_config["realspace"])
    parser.add_argument(
        "--num-samples-train",
        type=int,
        default=int(train_config["num_samples"]),
    )
    parser.add_argument(
        "--num-samples-val",
        type=int,
        default=int(val_config["num_samples"]),
    )
    parser.add_argument("--shape", type=int, default=configured_shape[0])
    parser.add_argument(
        "--diffraction-dtype",
        default=data_config["dtypes"]["diffraction"],
    )
    parser.add_argument(
        "--realspace-dtype",
        default=data_config["dtypes"]["realspace"],
    )
    parser.add_argument(
        "--input-log-data",
        action=argparse.BooleanOptionalAction,
        default=data_config.get("input_preprocessing", {}).get("transform") == "log1p",
        help="Use normalized log1p(intensity), matching the published model.",
    )
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help=(
            "Initial learning rate. Defaults to 1e-3 for "
            "reduced_bn_no_outer_skip and 1e-4 otherwise."
        ),
    )
    parser.add_argument(
        "--model-variant",
        choices=MODEL_VARIANTS,
        default=DEFAULT_MODEL_VARIANT,
        help=(
            "Use reduced by default; reduced_bn_no_outer_skip adds BatchNorm and "
            "removes the full-resolution skip; published retains 2048 channels."
        ),
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=("none", "plateau"),
        default="plateau",
        help="Reduce LR when validation WCA plateaus, matching AutoPhaseNN training.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--max-batches-per-epoch", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    if args.learning_rate is None:
        args.learning_rate = (
            1e-3
            if args.model_variant == REDUCED_BN_NO_OUTER_SKIP_VARIANT
            else 1e-4
        )
    args.data_config = str(Path(args.data_config).expanduser().resolve())
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
    """Format a nonnegative duration as ``HH:MM:SS`` or ``Dd HH:MM:SS``."""

    total_seconds = max(int(round(seconds)), 0)
    days, remainder = divmod(total_seconds, 24 * 60 * 60)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    clock = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{days}d {clock}" if days else clock


def load_model_state(
    model: HighStrainPhaseUNet,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    checkpoint_variant = infer_model_variant(state_dict)
    if checkpoint_variant != model.model_variant:
        raise ValueError(
            f"Checkpoint uses model variant {checkpoint_variant!r}, but the requested "
            f"model is {model.model_variant!r}. Select the matching --model-variant."
        )
    model.load_state_dict(state_dict, strict=True)
    return checkpoint


def save_checkpoint(
    path: Path,
    model: HighStrainPhaseUNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_val_loss: float,
    history: dict[str, list[dict]],
    args: argparse.Namespace,
    run_manifest: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "model_variant": model.model_variant,
            "parameter_count": count_parameters(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_loss": best_val_loss,
            "history": history,
            "args": vars(args),
            "project_version": run_manifest["runtime"]["project_version"],
            "git_commit": run_manifest["runtime"]["git_commit"],
            "run_manifest": run_manifest,
        },
        path,
    )


def run_epoch(
    model: HighStrainPhaseUNet,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    epochs: int,
    print_freq: int,
    max_batches: int,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    fp16: bool = False,
) -> dict[str, float | int]:
    training = optimizer is not None
    model.train(training)
    split = "TRAIN" if training else "VAL"
    batch_limit = len(loader)
    if max_batches > 0:
        batch_limit = min(batch_limit, max_batches)
    total_loss = 0.0
    total_samples = 0
    processed_batches = 0
    started = time.monotonic()
    grad_context = torch.enable_grad() if training else torch.no_grad()

    with grad_context:
        for batch_index, batch in enumerate(loader, start=1):
            if batch_index > batch_limit:
                break
            model_input = batch["input"].to(device, non_blocking=True).float()
            realspace = batch["realspace"].to(device, non_blocking=True)
            target_phase = reciprocal_phase_from_realspace(realspace)
            weights = model_input[:, 0]

            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=fp16 and device.type == "cuda"):
                predicted_phase = model(model_input)
                loss = phase_retrieval_wca_loss(
                    predicted_phase,
                    target_phase,
                    weights,
                )
            if training:
                if fp16 and device.type == "cuda":
                    assert scaler is not None
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            batch_size = model_input.shape[0]
            total_loss += float(loss.detach()) * batch_size
            total_samples += batch_size
            processed_batches += 1
            if batch_index % max(print_freq, 1) == 0 or batch_index == batch_limit:
                elapsed_seconds = time.monotonic() - started
                seconds_per_batch = elapsed_seconds / max(processed_batches, 1)
                remaining_seconds = seconds_per_batch * max(
                    batch_limit - processed_batches,
                    0,
                )
                LOGGER.info(
                    "%s | epoch=%03d/%03d | batch=%05d/%05d | loss=%.6e | "
                    "stage_elapsed=%s | stage_eta=%s",
                    split,
                    epoch,
                    epochs,
                    batch_index,
                    batch_limit,
                    float(loss.detach()),
                    format_duration(elapsed_seconds),
                    format_duration(remaining_seconds),
                )

    elapsed_seconds = time.monotonic() - started
    return {
        "loss": total_loss / max(total_samples, 1),
        "samples": total_samples,
        "batches": processed_batches,
        "elapsed_seconds": elapsed_seconds,
    }


def main() -> None:
    configure_logging()
    args = parse_args()
    if args.pretrained and args.resume:
        raise ValueError("--pretrained and --resume cannot be used together.")
    set_seed(args.seed)
    device = choose_device(args.device)
    data_config = load_data_config(args.data_config)
    shape = (args.shape, args.shape, args.shape)
    data_manifest = build_data_manifest(
        config=data_config,
        root=args.data_dir,
        shape=shape,
        diffraction_dtype=args.diffraction_dtype,
        realspace_dtype=args.realspace_dtype,
        splits={
            "train": {
                "diffraction": args.train_diff,
                "realspace": args.train_real,
                "num_samples": args.num_samples_train,
            },
            "val": {
                "diffraction": args.val_diff,
                "realspace": args.val_real,
                "num_samples": args.num_samples_val,
            },
        },
        input_log_data=args.input_log_data,
    )
    data_manifest["file_status"] = require_data_files(data_manifest)
    run_manifest: dict[str, object] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime_manifest(device),
        "data": data_manifest,
        "training": vars(args),
    }
    initialization = "pretrained" if args.pretrained else "resume" if args.resume else "scratch"
    if not args.run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = (
            f"{timestamp}_high_strain_{args.model_variant}_{initialization}"
            f"_bs-{args.batch_size}"
            f"_lr-{args.learning_rate:g}_seed-{args.seed}"
        )

    run_dir = Path(args.runs_dir).expanduser() / args.run_name
    checkpoint_dir = Path(args.checkpoint_root).expanduser() / args.run_name
    tensorboard_dir = run_dir / "tensorboard"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir = str(run_dir)
    args.checkpoint_dir = str(checkpoint_dir)
    (run_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2),
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2),
        encoding="utf-8",
    )

    data_dir = Path(args.data_dir)
    dataset_kwargs = {
        "shape": shape,
        "diffraction_dtype": args.diffraction_dtype,
        "realspace_dtype": args.realspace_dtype,
        "input_log_data": args.input_log_data,
    }
    train_dataset = AutoPhaseNNPhaseDataset(
        data_dir / args.train_diff,
        data_dir / args.train_real,
        args.num_samples_train,
        **dataset_kwargs,
    )
    val_dataset = AutoPhaseNNPhaseDataset(
        data_dir / args.val_diff,
        data_dir / args.val_real,
        args.num_samples_val,
        **dataset_kwargs,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": True,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = HighStrainPhaseUNet(model_variant=args.model_variant).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-7,
    )
    scheduler = None
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")
    history: dict[str, list[dict]] = {"train": [], "val": []}
    start_epoch = 1
    best_val_loss = float("inf")

    if args.pretrained:
        load_model_state(model, args.pretrained, device)
        LOGGER.info("Loaded converted/pretrained weights: %s", args.pretrained)
    elif args.resume:
        checkpoint = load_model_state(model, args.resume, device)
        checkpoint_version = checkpoint.get("project_version")
        current_version = run_manifest["runtime"]["project_version"]
        if checkpoint_version and checkpoint_version != current_version:
            LOGGER.warning(
                "Checkpoint project version %s differs from current version %s.",
                checkpoint_version,
                current_version,
            )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler_state = checkpoint.get("scheduler_state_dict")
        if scheduler is not None and scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        history = checkpoint.get("history", history)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        LOGGER.info("Resumed checkpoint %s at epoch %d", args.resume, start_epoch)

    LOGGER.info(
        "Run=%s | model=%s | version=%s | commit=%s | device=%s | parameters=%s | train/val=%s/%s | checkpoints=%s",
        args.run_name,
        model.model_variant,
        run_manifest["runtime"]["project_version"],
        run_manifest["runtime"]["git_commit"],
        device,
        f"{count_parameters(model):,}",
        f"{len(train_dataset):,}",
        f"{len(val_dataset):,}",
        checkpoint_dir,
    )
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    run_started = time.monotonic()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.monotonic()
        train_stats = run_epoch(
            model,
            train_loader,
            device,
            epoch,
            args.epochs,
            args.print_freq,
            args.max_batches_per_epoch,
            optimizer=optimizer,
            scaler=scaler,
            fp16=args.fp16,
        )
        val_stats = run_epoch(
            model,
            val_loader,
            device,
            epoch,
            args.epochs,
            args.print_freq,
            args.max_batches_per_epoch,
        )
        if scheduler is not None:
            scheduler.step(float(val_stats["loss"]))
        history["train"].append({"epoch": epoch, **train_stats})
        history["val"].append({"epoch": epoch, **val_stats})
        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )
        writer.add_scalar("train/loss", train_stats["loss"], epoch)
        writer.add_scalar("val/loss", val_stats["loss"], epoch)
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
            scaler,
            epoch,
            best_val_loss,
            history,
            args,
            run_manifest,
        )
        if improved:
            save_checkpoint(
                checkpoint_dir / "checkpoint_best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_val_loss,
                history,
                args,
                run_manifest,
            )
        if epoch % args.save_every == 0:
            save_checkpoint(
                checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_val_loss,
                history,
                args,
                run_manifest,
            )
        epoch_seconds = time.monotonic() - epoch_started
        run_seconds = time.monotonic() - run_started
        completed_epochs = epoch - start_epoch + 1
        average_epoch_seconds = run_seconds / max(completed_epochs, 1)
        remaining_seconds = average_epoch_seconds * max(args.epochs - epoch, 0)
        estimated_finish = datetime.now().astimezone() + timedelta(
            seconds=remaining_seconds
        )
        LOGGER.info(
            "Epoch %03d complete | train=%.6e | val=%.6e | best=%.6e | "
            "lr=%.3e | epoch_time=%s | run_elapsed=%s | eta=%s | finish=%s",
            epoch,
            train_stats["loss"],
            val_stats["loss"],
            best_val_loss,
            optimizer.param_groups[0]["lr"],
            format_duration(epoch_seconds),
            format_duration(run_seconds),
            format_duration(remaining_seconds),
            estimated_finish.strftime("%Y-%m-%d %H:%M:%S %Z"),
        )

    writer.close()
    LOGGER.info("Training complete | best checkpoint=%s", checkpoint_dir / "checkpoint_best.pt")


if __name__ == "__main__":
    main()
