import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import AutoPhaseDataset
from losses import get_loss, metric_dict, scale_align_sum
from model_tf_compatible import TFCompatibleAutoPhaseNN, load_weights


def choose_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def optional_data_path(data_dir, filename):
    if filename is None or filename.lower() in {"", "none", "null"}:
        return None
    return data_dir / filename


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, history, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "history": history,
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


def run_epoch(args, model, loader, loss_fn, device, optimizer=None, scaler=None, train=True):
    model.train(train)
    use_amp = args.fp16 and device.type == "cuda"
    total = {
        "loss": 0.0,
        "loss_ft": 0.0,
        "loss_amp": 0.0,
        "loss_phase": 0.0,
        "loss_support": 0.0,
        "samples": 0,
        "batches": 0,
    }

    iterator = tqdm(loader, leave=False, desc="train" if train else "val")
    grad_context = torch.enable_grad() if train else torch.no_grad()
    with grad_context:
        for batch_index, batch in enumerate(iterator, start=1):
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

            iterator.set_postfix(
                loss=total["loss"] / max(total["samples"], 1),
                ft=total["loss_ft"] / max(total["samples"], 1),
                amp=total["loss_amp"] / max(total["samples"], 1),
                phase=total["loss_phase"] / max(total["samples"], 1),
                support=total["loss_support"] / max(total["samples"], 1),
            )
            if args.max_batches_per_epoch and batch_index >= args.max_batches_per_epoch:
                break

    denom = max(total["samples"], 1)
    stats = {
        "loss": total["loss"] / denom,
        "loss_ft": total["loss_ft"] / denom,
        "loss_amp": total["loss_amp"] / denom,
        "loss_phase": total["loss_phase"] / denom,
        "loss_support": total["loss_support"] / denom,
        "samples": total["samples"],
        "batches": total["batches"],
    }
    mode = "train" if train else "val"
    print(
        f"{mode} epoch optimization terms | OptTotal: {stats['loss']:.4e} | "
        f"FTLoss: {stats['loss_ft']:.4e} | "
        f"SupportWeighted: {(args.support_weight * stats['loss_support']):.4e}"
    )
    print(
        f"{mode} epoch real-space monitors | AmpL1Full: {stats['loss_amp']:.4e} | "
        f"PhaseL1PredSup: {stats['loss_phase']:.4e} | "
        f"SupportBCE: {stats['loss_support']:.4e}"
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
    parser = argparse.ArgumentParser(description="Standalone AutoPhaseNN PyTorch training.")
    parser.add_argument("--data-dir", default="/data_ssd/oyys/autophasenn/")
    parser.add_argument("--data-train-diff", default="train_diff.npy")
    parser.add_argument("--data-train-real", default="train_real.npy")
    parser.add_argument("--data-val-diff", default="val_diff.npy")
    parser.add_argument("--data-val-real", default="val_real.npy")
    parser.add_argument("--num-samples-train", type=int, default=25000)
    parser.add_argument("--num-samples-val", type=int, default=5000)
    parser.add_argument("--shape", type=int, default=64)
    parser.add_argument("--dtype-diff", default="float32")
    parser.add_argument("--dtype-real", default="complex64")
    parser.add_argument("--output-dir", default="./autophasenn_training_pipeline/output/")
    parser.add_argument("--train-size", type=int, default=0)
    parser.add_argument(
        "--overfit-samples",
        type=int,
        default=100,
        help="Train and validate on the same first N training samples for fit debugging.",
    )
    parser.add_argument(
        "--cache-data",
        action="store_true",
        help="Load the selected memmap samples into RAM at dataset construction time.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.1)
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
    parser.add_argument("--resume", default="/data_ssd/oyys/autophasenn/autophasenn.pth")
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Require training from random initialization; incompatible with --pretrained or --resume.",
    )
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max-batches-per-epoch", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.batch_average_loss:
        print("--batch-average-loss is deprecated and ignored; losses are batch-mean by default.")
    if args.unsupervised:
        args.loss_scope = "diff"
    if args.from_scratch and (args.pretrained or args.resume):
        raise ValueError("--from-scratch cannot be combined with --pretrained or --resume.")

    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", vars(args))

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
    print(
        f"Resolved memmap samples: train={train_samples}, "
        f"val={train_samples if overfit_mode else args.num_samples_val}"
    )
    print(
        "Training setup: {} | loss_scope={} | loss_type={} | cache_data={}".format(
            "from_scratch" if not args.pretrained and not args.resume else "checkpointed",
            args.loss_scope,
            args.loss_type,
            cache_data,
        )
    )

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
        print(f"Overfit mode enabled: validating on the same {train_samples} training samples.")
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

    model = TFCompatibleAutoPhaseNN(threshold=args.threshold).to(device)
    if args.pretrained:
        load_weights(model, args.pretrained, map_location=device)
        print(f"Loaded pretrained weights: {args.pretrained}")

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = make_scheduler(args, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")
    loss_fn = get_loss(args.loss_type)
    history = {"train": [], "val": []}
    start_epoch = 1

    if args.resume:
        checkpoint = load_weights(model, args.resume, map_location=device)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        history = checkpoint.get("history", history)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"Resumed checkpoint: {args.resume}")

    if args.dry_run:
        model.eval()
        metrics = one_batch_metrics(args, model, val_loader, device)
        save_json(output_dir / "dry_run_metrics.json", metrics)
        print(json.dumps(metrics, indent=2))
        print("Dry run complete; no training was performed.")
        return

    best_val = float("inf")
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        train_stats = run_epoch(
            args, model, train_loader, loss_fn, device, optimizer=optimizer, scaler=scaler, train=True
        )
        val_stats = run_epoch(args, model, val_loader, loss_fn, device, train=False)

        if scheduler:
            if args.lr_scheduler == "plateau":
                scheduler.step(val_stats["loss"])
            else:
                scheduler.step()

        history["train"].append({"epoch": epoch, **train_stats})
        history["val"].append({"epoch": epoch, **val_stats})
        save_json(output_dir / "history.json", history)

        print(
            f"epoch {epoch}/{args.epochs} "
            f"train_opt_total={train_stats['loss']:.6g} train_ft_loss={train_stats['loss_ft']:.6g} "
            f"train_support_weighted={(args.support_weight * train_stats['loss_support']):.6g} "
            f"train_amp_l1_full={train_stats['loss_amp']:.6g} "
            f"train_phase_l1_predsup={train_stats['loss_phase']:.6g} "
            f"train_support_bce={train_stats['loss_support']:.6g} "
            f"val_opt_total={val_stats['loss']:.6g} val_ft_loss={val_stats['loss_ft']:.6g} "
            f"val_support_weighted={(args.support_weight * val_stats['loss_support']):.6g} "
            f"val_amp_l1_full={val_stats['loss_amp']:.6g} "
            f"val_phase_l1_predsup={val_stats['loss_phase']:.6g} "
            f"val_support_bce={val_stats['loss_support']:.6g}"
        )

        save_checkpoint(
            output_dir / "checkpoint_last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            history,
            args,
        )
        if epoch % args.save_every == 0:
            save_checkpoint(
                output_dir / f"checkpoint_epoch_{epoch:04d}.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                history,
                args,
            )
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            save_checkpoint(
                output_dir / "checkpoint_best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                history,
                args,
            )

    print(f"Training complete in {time.time() - t0:.1f}s. Best val_loss={best_val:.6g}")


if __name__ == "__main__":
    main()
