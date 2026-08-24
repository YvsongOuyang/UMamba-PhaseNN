import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import AutoPhaseDataset
from losses import get_loss, metric_dict, scale_align_sum
from model_factory import (
    MODEL_VARIANTS,
    create_model,
    load_pretrained_weights,
    resolve_support_threshold,
)
from model_tf_compatible import load_weights


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = PROJECT_DIR / "runs"
DEFAULT_CHECKPOINT_ROOT = Path(
    "/data_ssd/oyys/autophasenn/autophasenn_pipeline_output"
)
DEFAULT_RESUME_PATH = str(
    DEFAULT_CHECKPOINT_ROOT / "autophasenn_retrain_l1" / "checkpoint_last.pt"
)
LOG_WIDTH = 96
LOGGER = logging.getLogger("autophasenn.train")


def configure_logging():
    """Configure timestamped, line-buffered console logging for redirected runs."""

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


def format_duration(seconds):
    return str(timedelta(seconds=max(int(seconds), 0)))


def format_yes_no(value):
    return "yes" if value else "no"


def choose_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def optional_data_path(data_dir, filename):
    if filename is None or filename.lower() in {"", "none", "null"}:
        return None
    return data_dir / filename


def build_run_name(args):
    """Build a readable, filesystem-safe experiment name from key settings."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    initialization = "scratch" if not args.pretrained and not args.resume else "checkpointed"
    return (
        f"{timestamp}_{args.model_variant}_{initialization}"
        f"_loss-{args.loss_type}_{args.loss_scope}"
        f"_bs-{args.batch_size}_lr-{args.lr:g}"
        f"_{args.optimizer}_{args.lr_scheduler}"
        f"_T-{args.threshold:g}_seed-{args.seed}"
    )


def write_tensorboard_epoch(writer, epoch, train_stats, val_stats, learning_rate):
    """Record the epoch metrics already produced by the training loop."""

    writer.add_scalar("learning_rate", learning_rate, epoch)
    for key in ("loss", "loss_ft", "loss_amp", "loss_phase", "loss_support"):
        writer.add_scalar(f"train/{key}", train_stats[key], epoch)
        writer.add_scalar(f"val/{key}", val_stats[key], epoch)
    writer.flush()


def training_stage_for_epoch(args, epoch):
    """Return the decoder-cross-skip fine-tuning stage for an epoch."""

    if args.model_variant != "decoder_cross_skip":
        return "all"
    if epoch <= args.cross_skip_only_epochs:
        return "cross_skip"
    decoder_stage_end = args.cross_skip_only_epochs + args.decoder_finetune_epochs
    if epoch <= decoder_stage_end:
        return "decoders"
    return "all"


def configure_training_stage(args, model, epoch, previous_stage):
    """Apply staged freezing and log changes in trainable parameter scope."""

    stage = training_stage_for_epoch(args, epoch)
    if args.model_variant == "decoder_cross_skip":
        model.set_trainable_stage(stage)
    if stage != previous_stage:
        trainable = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        total = sum(parameter.numel() for parameter in model.parameters())
        LOGGER.info(
            "Trainable stage | epoch=%d | scope=%s | parameters=%s / %s",
            epoch,
            stage,
            f"{trainable:,}",
            f"{total:,}",
        )
    return stage


def record_cross_skip_strengths(writer, model, epoch):
    """Write learned cross-skip residual strengths when the model provides them."""

    if not hasattr(model, "cross_skip_strengths"):
        return
    strengths = model.cross_skip_strengths()
    for name, value in strengths.items():
        writer.add_scalar(f"cross_skip/{name}", value, epoch)
    writer.flush()
    LOGGER.info(
        "Cross-skip strengths | %s",
        " | ".join(f"{name}={value:+.4e}" for name, value in strengths.items()),
    )


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, history, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    best_val = min((item.get("loss", float("inf")) for item in history.get("val", [])), default=float("inf"))
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "history": history,
            "best_val": best_val,
            "args": vars(args),
            "threshold": args.threshold,
        },
        path,
    )


def make_scheduler(args, optimizer):
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    if args.lr_scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=args.gamma, patience=args.patience, min_lr=args.min_lr
        )
    raise ValueError(f"Unknown scheduler {args.lr_scheduler}")


def checkpoint_key_summary(checkpoint):
    if not isinstance(checkpoint, dict):
        return "non-dict checkpoint"
    keys = list(checkpoint.keys())
    preview = ", ".join(keys[:12])
    suffix = "..." if len(keys) > 12 else ""
    return f"{len(keys)} top-level keys: [{preview}{suffix}]"


def optimizer_lrs(optimizer):
    return [group.get("lr", None) for group in optimizer.param_groups]


def log_training_setup(
    args,
    device,
    run_dir,
    output_dir,
    checkpoint_dir,
    tensorboard_dir,
    train_samples,
    validation_samples,
    cache_data,
    model,
):
    mode = "from scratch" if not args.pretrained and not args.resume else "checkpointed"
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    lines = [
        "=" * LOG_WIDTH,
        "AutoPhaseNN training",
        "=" * LOG_WIDTH,
        "[Run]",
        f"  name              : {args.run_name}",
        f"  model             : {args.model_variant}",
        f"  initialization    : {mode}",
        f"  device / fp16     : {device} / {format_yes_no(args.fp16 and device.type == 'cuda')}",
        f"  parameters        : {parameter_count:,}",
        "[Data]",
        f"  directory         : {args.data_dir}",
        f"  train / validation: {train_samples:,} / {validation_samples:,}",
        f"  volume shape      : {args.shape} x {args.shape} x {args.shape}",
        f"  batch / workers   : {args.batch_size} / {args.num_workers}",
        f"  cache in memory   : {format_yes_no(cache_data)}",
        "[Optimization]",
        f"  loss / scope      : {args.loss_type} / {args.loss_scope}",
        f"  optimizer / lr    : {args.optimizer} / {args.lr:.3e}",
        f"  scheduler         : {args.lr_scheduler}",
        f"  support weight    : {args.support_weight:g}",
        f"  threshold         : {args.threshold:g}",
        f"  epochs            : {args.epochs}",
        "[Artifacts]",
        f"  run               : {run_dir}",
        f"  output            : {output_dir}",
        f"  checkpoints       : {checkpoint_dir}",
        f"  tensorboard       : {tensorboard_dir}",
        "=" * LOG_WIDTH,
    ]
    if args.model_variant == "decoder_cross_skip":
        artifacts_index = lines.index("[Artifacts]")
        lines.insert(
            artifacts_index,
            "  training stages   : "
            f"cross-skip={args.cross_skip_only_epochs}, "
            f"decoders={args.decoder_finetune_epochs}, then all",
        )
    LOGGER.info("\n%s", "\n".join(lines))


def log_resume_summary(
    args,
    checkpoint,
    optimizer_restored,
    scheduler_restored,
    scaler_restored,
    start_epoch,
    optimizer,
):
    checkpoint_epoch = checkpoint.get("epoch", None) if isinstance(checkpoint, dict) else None
    history = checkpoint.get("history", {}) if isinstance(checkpoint, dict) else {}
    train_history = len(history.get("train", [])) if isinstance(history, dict) else 0
    val_history = len(history.get("val", [])) if isinstance(history, dict) else 0
    best_val = checkpoint.get("best_val", None) if isinstance(checkpoint, dict) else None
    lines = [
        "-" * LOG_WIDTH,
        "Checkpoint resume",
        "-" * LOG_WIDTH,
        f"  path              : {args.resume}",
        f"  contents          : {checkpoint_key_summary(checkpoint)}",
        f"  restored states   : model=yes, optimizer={format_yes_no(optimizer_restored)}, "
        f"scheduler={format_yes_no(scheduler_restored)}, scaler={format_yes_no(scaler_restored)}",
        f"  epoch             : saved={checkpoint_epoch}, next={start_epoch}",
        f"  history entries   : train={train_history}, validation={val_history}",
        f"  best val loss     : {best_val}",
        f"  optimizer lr      : {optimizer_lrs(optimizer)}",
        "  resume granularity: epoch-level; training restarts from batch 1 of the next epoch",
        "-" * LOG_WIDTH,
    ]
    LOGGER.info("\n%s", "\n".join(lines))


def log_epoch_summary(
    args,
    epoch,
    train_stats,
    val_stats,
    learning_rate,
    best_val,
    best_updated,
    periodic_saved,
):
    header = (
        f"{'split':<10}{'total':>13}{'fourier':>13}{'amplitude':>13}"
        f"{'phase':>13}{'support':>13}{'time':>12}"
    )

    def metric_row(name, stats):
        return (
            f"{name:<10}{stats['loss']:>13.4e}{stats['loss_ft']:>13.4e}"
            f"{stats['loss_amp']:>13.4e}{stats['loss_phase']:>13.4e}"
            f"{stats['loss_support']:>13.4e}{format_duration(stats['elapsed_seconds']):>12}"
        )

    lines = [
        "-" * LOG_WIDTH,
        f"Epoch {epoch:03d}/{args.epochs:03d} summary",
        header,
        metric_row("train", train_stats),
        metric_row("validation", val_stats),
        f"lr={learning_rate:.3e} | best_val={best_val:.4e} | "
        f"best_updated={format_yes_no(best_updated)} | "
        f"periodic_checkpoint={format_yes_no(periodic_saved)}",
        "-" * LOG_WIDTH,
    ]
    LOGGER.info("\n%s", "\n".join(lines))


def unpack_outputs(outputs):
    if not isinstance(outputs, (tuple, list)) or len(outputs) < 5:
        raise RuntimeError(
            "Model must return (pred_diff, pred_obj, pred_amp, pred_phi, support)."
        )
    return outputs[:5]


def raw_amp_from_outputs(outputs, pred_amp):
    if isinstance(outputs, (tuple, list)) and len(outputs) >= 6:
        return outputs[5]
    return pred_amp


def balanced_support_bce(raw_amp, target_support):
    bce = F.binary_cross_entropy(
        raw_amp.clamp(1e-6, 1.0 - 1e-6),
        target_support,
        reduction="none",
    )
    target_support = target_support.float()
    pos = target_support
    neg = 1.0 - target_support
    pos_loss = (bce * pos).sum() / pos.sum().clamp_min(1.0)
    neg_loss = (bce * neg).sum() / neg.sum().clamp_min(1.0)
    return 0.5 * (pos_loss + neg_loss)


def compute_loss_bundle(args, loss_fn, diff, amp, phi, outputs):
    pred_diff, _obj, pred_amp, pred_phi, support = unpack_outputs(outputs)
    raw_amp = raw_amp_from_outputs(outputs, pred_amp)
    pred_for_loss = scale_align_sum(diff, pred_diff) if args.scale_align_loss else pred_diff
    loss_ft = loss_fn(diff, pred_for_loss)
    loss_amp = F.l1_loss(pred_amp, amp)
    loss_phase = F.l1_loss(pred_phi * support, phi * support)
    target_support = (amp >= args.threshold).float()
    loss_support = balanced_support_bce(raw_amp, target_support)
    if args.unsupervised or args.loss_scope == "diff":
        loss = loss_ft
    else:
        loss = (
            args.ft_weight * loss_ft
            + args.amp_weight * loss_amp
            + args.phase_weight * loss_phase
        )
    loss = loss + args.support_weight * loss_support
    return loss, loss_ft, loss_amp, loss_phase, loss_support, pred_for_loss


def run_epoch(args, model, loader, loss_fn, device, epoch=0, optimizer=None, scaler=None, train=True):
    model.train(train)
    use_amp = args.fp16 and device.type == "cuda"
    mode = "train" if train else "val"
    num_batches = len(loader)
    if args.max_batches_per_epoch and args.max_batches_per_epoch > 0:
        num_batches = min(num_batches, args.max_batches_per_epoch)
    print_freq = max(int(args.print_freq), 1)
    epoch_start_time = time.time()
    total = {
        "loss": 0.0,
        "loss_ft": 0.0,
        "loss_amp": 0.0,
        "loss_phase": 0.0,
        "loss_support": 0.0,
        "samples": 0,
        "batches": 0,
    }

    grad_context = torch.enable_grad() if train else torch.no_grad()
    with grad_context:
        for batch_index, batch in enumerate(loader, start=1):
            if batch_index > num_batches:
                break
            diff = batch["diff"].to(device, non_blocking=True).float()
            amp = batch["amp"].to(device, non_blocking=True).float()
            phi = batch["phi"].to(device, non_blocking=True).float()

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(diff)
                loss, loss_ft, loss_amp, loss_phase, loss_support, _pred_for_loss = compute_loss_bundle(
                    args, loss_fn, diff, amp, phi, outputs
                )
            if train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            batch_size = diff.shape[0]
            total["loss"] += float(loss.detach().cpu()) * batch_size
            total["loss_ft"] += float(loss_ft.detach().cpu()) * batch_size
            total["loss_amp"] += float(loss_amp.detach().cpu()) * batch_size
            total["loss_phase"] += float(loss_phase.detach().cpu()) * batch_size
            total["loss_support"] += float(loss_support.detach().cpu()) * batch_size
            total["samples"] += batch_size
            total["batches"] += 1

            if batch_index % print_freq == 0 or batch_index == num_batches:
                elapsed_seconds = time.time() - epoch_start_time
                avg_time_per_iter = elapsed_seconds / max(batch_index, 1)
                eta_seconds = (num_batches - batch_index) * avg_time_per_iter
                current_lr = optimizer.param_groups[0]["lr"] if optimizer is not None else 0.0
                progress = 100.0 * batch_index / max(num_batches, 1)
                lr_text = f" | lr={current_lr:.3e}" if train else ""
                LOGGER.info(
                    "%s | epoch=%03d/%03d | batch=%05d/%05d (%6.2f%%)%s | "
                    "elapsed=%s | eta=%s\n"
                    "    losses | total=%.4e | fourier=%.4e | amplitude=%.4e | "
                    "phase=%.4e | support=%.4e",
                    mode.upper(),
                    epoch,
                    args.epochs,
                    batch_index,
                    num_batches,
                    progress,
                    lr_text,
                    format_duration(elapsed_seconds),
                    format_duration(eta_seconds),
                    loss.item(),
                    loss_ft.item(),
                    loss_amp.item(),
                    loss_phase.item(),
                    loss_support.item(),
                )

    denom = max(total["samples"], 1)
    elapsed_total = time.time() - epoch_start_time
    stats = {
        "loss": total["loss"] / denom,
        "loss_ft": total["loss_ft"] / denom,
        "loss_amp": total["loss_amp"] / denom,
        "loss_phase": total["loss_phase"] / denom,
        "loss_support": total["loss_support"] / denom,
        "samples": total["samples"],
        "batches": total["batches"],
        "elapsed_seconds": elapsed_total,
    }
    LOGGER.info(
        "%s epoch %03d complete | batches=%d | samples=%d | time=%s",
        mode.upper(),
        epoch,
        stats["batches"],
        stats["samples"],
        format_duration(elapsed_total),
    )
    return stats


@torch.no_grad()
def one_batch_metrics(args, model, loader, device):
    batch = next(iter(loader))
    diff = batch["diff"].to(device).float()
    pred_diff = model(diff)[0]
    if args.scale_align_loss:
        pred_diff = scale_align_sum(diff, pred_diff)
    return metric_dict(diff, pred_diff)

def main():
    configure_logging()
    parser = argparse.ArgumentParser(description="Standalone AutoPhaseNN PyTorch training.")
    parser.add_argument("--data-dir", default="/data_ssd/oyys/autophasenn/")
    parser.add_argument("--data-train-diff", default="train_diff.npy")
    parser.add_argument("--data-train-real", default="train_real.npy")
    parser.add_argument("--data-val-diff", default="val_diff.npy")
    parser.add_argument("--data-val-real", default="val_real.npy")
    parser.add_argument("--num-samples-train", type=int, default=25000)
    parser.add_argument("--num-samples-val", type=int, default=5000)
    parser.add_argument("--shape", type=int, default=64)
    parser.add_argument(
        "--model-variant",
        choices=MODEL_VARIANTS,
        default="residual",
        help="Network architecture variant.",
    )
    parser.add_argument(
        "--cross-skip-only-epochs",
        type=int,
        default=0,
        help="Initial epochs training only decoder cross-skip modules.",
    )
    parser.add_argument(
        "--decoder-finetune-epochs",
        type=int,
        default=0,
        help="Additional epochs training both decoders and cross-skip modules.",
    )
    parser.add_argument("--dtype-diff", default="float32")
    parser.add_argument("--dtype-real", default="complex64")
    parser.add_argument(
        "--runs-dir",
        default=str(DEFAULT_RUNS_DIR),
        help="Parent directory for named training runs.",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Experiment directory name; empty builds one from key settings.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Config/history directory; empty uses <runs-dir>/<run-name>.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="",
        help="Checkpoint directory; empty uses <checkpoint-root>/<run-name>.",
    )
    parser.add_argument("--train-size", type=int, default=0)
    parser.add_argument(
        "--overfit-samples",
        type=int,
        default=0,
        help="Train and validate on the same first N training samples for fit debugging.",
    )
    parser.add_argument(
        "--cache-data",
        action="store_true",
        help="Load the selected memmap samples into RAM at dataset construction time.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Support threshold; defaults to 0.3 for mamba_skip and 0.1 otherwise.",
    )
    parser.add_argument("--scale-i", type=float, default=0.0)
    parser.add_argument("--scale-align-loss", action="store_true")
    parser.add_argument("--loss-type", default="l1")
    parser.add_argument(
        "--loss-scope",
        choices=["diff", "supervised"],
        default="diff",
        help="diff uses only reciprocal-space --loss-type; supervised also adds amp/phase losses.",
    )
    parser.add_argument(
        "--batch-average-loss",
        action="store_true",
        help="Deprecated; losses in losses.py already use batch-mean reduction.",
    )
    parser.add_argument("--unsupervised", action="store_true")
    parser.add_argument("--ft-weight", type=float, default=1.0)
    parser.add_argument("--amp-weight", type=float, default=1.0)
    parser.add_argument("--phase-weight", type=float, default=1.0)
    parser.add_argument(
        "--support-weight",
        type=float,
        default=0.0,
        help="Optional BCE(raw_amp, amp>=threshold) support-shape loss. Default keeps paper-style diff-only training.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    parser.add_argument("--lr-scheduler", choices=["none", "step", "plateau"], default="plateau")
    parser.add_argument("--step-size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--pretrained", default="")
    parser.add_argument("--resume", default=DEFAULT_RESUME_PATH)
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Require training from random initialization; incompatible with --pretrained or --resume.",
    )
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument("--max-batches-per-epoch", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.threshold = resolve_support_threshold(args.model_variant, args.threshold)

    if args.batch_average_loss:
        LOGGER.warning(
            "--batch-average-loss is deprecated and ignored; losses already use batch means."
        )
    if args.cross_skip_only_epochs < 0 or args.decoder_finetune_epochs < 0:
        raise ValueError("Cross-skip training-stage epoch counts cannot be negative.")
    if args.model_variant != "decoder_cross_skip" and (
        args.cross_skip_only_epochs or args.decoder_finetune_epochs
    ):
        raise ValueError(
            "Cross-skip training stages require --model-variant decoder_cross_skip."
        )
    if args.unsupervised:
        args.loss_scope = "diff"
    if args.from_scratch and args.resume == DEFAULT_RESUME_PATH:
        args.resume = ""
    if args.pretrained and args.resume == DEFAULT_RESUME_PATH:
        args.resume = ""
    if args.pretrained and args.resume:
        raise ValueError("--pretrained and --resume cannot be used together.")
    if args.from_scratch and (args.pretrained or args.resume):
        raise ValueError("--from-scratch cannot be combined with --pretrained or --resume.")

    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    args.run_name = args.run_name.strip() or build_run_name(args)
    run_dir = Path(args.runs_dir).expanduser() / args.run_name
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else run_dir
    checkpoint_dir = (
        Path(args.checkpoint_dir).expanduser()
        if args.checkpoint_dir
        else DEFAULT_CHECKPOINT_ROOT / args.run_name
    )
    tensorboard_dir = output_dir / "tensorboard"
    args.runs_dir = str(Path(args.runs_dir).expanduser())
    args.run_dir = str(run_dir)
    args.output_dir = str(output_dir)
    args.checkpoint_dir = str(checkpoint_dir)
    args.tensorboard_dir = str(tensorboard_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", vars(args))
    run_info = {
        "run_name": args.run_name,
        "created_at": datetime.now().astimezone().isoformat(),
        "model_variant": args.model_variant,
        "initialization": (
            "from_scratch" if not args.pretrained and not args.resume else "checkpointed"
        ),
        "loss": {"type": args.loss_type, "scope": args.loss_scope},
        "optimization": {
            "optimizer": args.optimizer,
            "learning_rate": args.lr,
            "scheduler": args.lr_scheduler,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "fp16": args.fp16,
            "cross_skip_only_epochs": args.cross_skip_only_epochs,
            "decoder_finetune_epochs": args.decoder_finetune_epochs,
        },
        "data": {
            "directory": args.data_dir,
            "train_samples": args.num_samples_train,
            "validation_samples": args.num_samples_val,
        },
        "paths": {
            "run": str(run_dir),
            "output": str(output_dir),
            "checkpoints": str(checkpoint_dir),
            "tensorboard": str(tensorboard_dir),
        },
    }
    save_json(output_dir / "run_info.json", run_info)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    writer.add_text("run/info", json.dumps(run_info, indent=2), 0)
    writer.flush()

    data_dir = Path(args.data_dir)
    shape = (args.shape, args.shape, args.shape)
    train_samples = args.num_samples_train
    if args.train_size and args.train_size > 0:
        train_samples = min(train_samples, args.train_size)
    overfit_samples = max(args.overfit_samples, 0)
    overfit_mode = overfit_samples > 0
    if overfit_mode:
        train_samples = min(train_samples, overfit_samples)
    cache_data = args.cache_data or overfit_mode
    validation_samples = train_samples if overfit_mode else args.num_samples_val

    train_dataset = AutoPhaseDataset(
        data_dir / args.data_train_diff,
        optional_data_path(data_dir, args.data_train_real),
        train_samples,
        shape_diff=shape,
        shape_real=shape,
        dtype_diff=args.dtype_diff,
        dtype_real=args.dtype_real,
        scale_i=args.scale_i,
        shuffle=False,
        cache_data=cache_data,
    )
    if overfit_mode:
        val_dataset = train_dataset
        LOGGER.warning(
            "Overfit mode enabled: validation reuses the same %d training samples.",
            train_samples,
        )
    else:
        val_dataset = AutoPhaseDataset(
            data_dir / args.data_val_diff,
            optional_data_path(data_dir, args.data_val_real),
            args.num_samples_val,
            shape_diff=shape,
            shape_real=shape,
            dtype_diff=args.dtype_diff,
            dtype_real=args.dtype_real,
            scale_i=args.scale_i,
            shuffle=False,
            cache_data=cache_data,
        )
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = create_model(args.model_variant, threshold=args.threshold).to(device)
    log_training_setup(
        args,
        device,
        run_dir,
        output_dir,
        checkpoint_dir,
        tensorboard_dir,
        train_samples,
        validation_samples,
        cache_data,
        model,
    )
    if args.pretrained:
        load_pretrained_weights(
            model,
            args.model_variant,
            args.pretrained,
            map_location="cpu",
        )
        if args.model_variant == "amplitude_skip":
            LOGGER.info(
                "Loaded baseline weights with zero-initialized amplitude skip kernels: %s",
                args.pretrained,
            )
        elif args.model_variant == "decoder_cross_skip":
            LOGGER.info(
                "Loaded baseline weights with zero-initialized cross-skip strengths: %s",
                args.pretrained,
            )
        elif args.model_variant == "decoder_cross_concat":
            LOGGER.info(
                "Loaded baseline weights with zero-initialized cross-concat kernels: %s",
                args.pretrained,
            )
        elif args.model_variant == "mamba_skip":
            LOGGER.info(
                "Loaded baseline backbone weights; Mamba bridges and fusion blocks "
                "retain their initialized values: %s",
                args.pretrained,
            )
        else:
            LOGGER.info("Loaded pretrained weights: %s", args.pretrained)

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = make_scheduler(args, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")
    loss_fn = get_loss(args.loss_type)
    history = {"train": [], "val": []}
    start_epoch = 1
    checkpoint = None

    if args.resume:
        LOGGER.info("Loading checkpoint for resume: %s", args.resume)
        checkpoint = load_weights(model, args.resume, map_location=device)
        optimizer_restored = False
        scheduler_restored = False
        scaler_restored = False
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            optimizer_restored = True
        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            scheduler_restored = True
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            scaler_restored = True
        history = checkpoint.get("history", history)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        log_resume_summary(
            args,
            checkpoint,
            optimizer_restored,
            scheduler_restored,
            scaler_restored,
            start_epoch,
            optimizer,
        )
    else:
        LOGGER.info(
            "Starting at epoch %d with randomly initialized optimizer state; lr=%s",
            start_epoch,
            optimizer_lrs(optimizer),
        )

    if args.dry_run:
        model.eval()
        metrics = one_batch_metrics(args, model, val_loader, device)
        save_json(output_dir / "dry_run_metrics.json", metrics)
        for key, value in metrics.items():
            writer.add_scalar(f"dry_run/{key}", value, 0)
        writer.close()
        LOGGER.info("Dry-run metrics:\n%s", json.dumps(metrics, indent=2))
        LOGGER.info("Dry run complete; no training was performed.")
        return

    best_val = checkpoint.get("best_val", float("inf")) if args.resume else float("inf")
    if best_val == float("inf"):
        best_val = min((item.get("loss", float("inf")) for item in history.get("val", [])), default=float("inf"))
    if best_val < float("inf"):
        LOGGER.info("Loaded best validation loss: %.6g", best_val)
    t0 = time.time()
    active_stage = None
    for epoch in range(start_epoch, args.epochs + 1):
        active_stage = configure_training_stage(
            args,
            model,
            epoch,
            previous_stage=active_stage,
        )
        train_stats = run_epoch(
            args, model, train_loader, loss_fn, device, epoch=epoch, optimizer=optimizer, scaler=scaler, train=True
        )
        val_stats = run_epoch(args, model, val_loader, loss_fn, device, epoch=epoch, train=False)

        if scheduler:
            if args.lr_scheduler == "plateau":
                scheduler.step(val_stats["loss"])
            else:
                scheduler.step()

        history["train"].append(
            {"epoch": epoch, "trainable_stage": active_stage, **train_stats}
        )
        history["val"].append({"epoch": epoch, **val_stats})
        save_json(output_dir / "history.json", history)
        write_tensorboard_epoch(
            writer,
            epoch,
            train_stats,
            val_stats,
            optimizer.param_groups[0]["lr"],
        )
        record_cross_skip_strengths(writer, model, epoch)

        save_checkpoint(
            checkpoint_dir / "checkpoint_last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            history,
            args,
        )
        periodic_saved = epoch % args.save_every == 0
        if periodic_saved:
            save_checkpoint(
                checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                history,
                args,
            )
        best_updated = val_stats["loss"] < best_val
        if best_updated:
            best_val = val_stats["loss"]
            save_checkpoint(
                checkpoint_dir / "checkpoint_best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                history,
                args,
            )

        log_epoch_summary(
            args,
            epoch,
            train_stats,
            val_stats,
            optimizer.param_groups[0]["lr"],
            best_val,
            best_updated,
            periodic_saved,
        )

    writer.close()
    lines = [
        "=" * LOG_WIDTH,
        "Training complete",
        "=" * LOG_WIDTH,
        f"  total time        : {format_duration(time.time() - t0)}",
        f"  best val loss     : {best_val:.6g}",
        f"  best checkpoint   : {checkpoint_dir / 'checkpoint_best.pt'}",
        f"  history           : {output_dir / 'history.json'}",
        f"  tensorboard       : {tensorboard_dir}",
        "=" * LOG_WIDTH,
    ]
    LOGGER.info("\n%s", "\n".join(lines))


if __name__ == "__main__":
    main()
