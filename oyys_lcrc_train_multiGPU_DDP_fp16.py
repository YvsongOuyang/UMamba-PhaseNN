from __future__ import annotations

import argparse
import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_loader import Dataset


SCRIPT_DIR = Path(__file__).resolve().parent


def str2bool(value):
    """Accept both shell flags and explicit true/false strings."""
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def positive_spatial_dims(tensor: torch.Tensor) -> Tuple[int, ...]:
    return tuple(range(1, tensor.ndim))


def loss_log(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    y_true = torch.clamp(y_true.float(), min=0.0)
    y_pred = torch.clamp(y_pred.float(), min=0.0)
    pred = torch.log10(y_pred + 1.0)
    true = torch.log10(y_true + 1.0)
    return torch.sum((pred - true) ** 2) / (torch.sum(true ** 2) + 1e-8)


def loss_sq(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    dims = positive_spatial_dims(y_true)
    top = torch.sum((y_pred - y_true) ** 2, dim=dims, keepdim=True)
    bottom = torch.sum(y_true ** 2, dim=dims, keepdim=True)
    return torch.mean(top / (bottom + 1e-6))


def loss_mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    dims = positive_spatial_dims(y_true)
    top = torch.sum(torch.abs(y_pred - y_true), dim=dims, keepdim=True)
    bottom = torch.sum(torch.abs(y_true), dim=dims, keepdim=True)
    return torch.mean(top / (bottom + 1e-8))


def loss_paper(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    sqrt_true = torch.sqrt(torch.clamp(y_true, min=0.0))
    sqrt_pred = torch.sqrt(torch.clamp(y_pred, min=0.0))
    return torch.mean(torch.abs(sqrt_pred - sqrt_true))


def loss_pcc(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    dims = positive_spatial_dims(y_true)
    pred_centered = y_pred - torch.mean(y_pred, dim=dims, keepdim=True)
    true_centered = y_true - torch.mean(y_true, dim=dims, keepdim=True)
    numerator = torch.sum(pred_centered * true_centered, dim=dims, keepdim=True)
    pred_norm = torch.sqrt(torch.sum(pred_centered ** 2, dim=dims, keepdim=True) + 1e-8)
    true_norm = torch.sqrt(torch.sum(true_centered ** 2, dim=dims, keepdim=True) + 1e-8)
    pcc = torch.clamp(numerator / (pred_norm * true_norm), -1.0 + 1e-6, 1.0 - 1e-6)
    return torch.mean(1.0 - pcc)


def loss_comb(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return 0.5 * (loss_sq(y_true, y_pred) + loss_pcc(y_true, y_pred))


def loss_comb2(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.sqrt(loss_sq(y_true, y_pred) + 1e-8) + loss_pcc(y_true, y_pred))


def loss_comb_log(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return (50.0 * loss_sq(y_true, y_pred) + 50.0 * loss_pcc(y_true, y_pred) + loss_log(y_true, y_pred)) / 101.0


def build_reconstruction_loss(loss_type: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    loss_type = loss_type.lower()
    losses = {
        "mae": lambda y_true, y_pred: F.l1_loss(y_pred, y_true),
        "mae_norm": loss_mae,
        "mse": loss_sq,
        "huber": lambda y_true, y_pred: F.smooth_l1_loss(y_pred, y_true),
        "pcc": loss_pcc,
        "paper": loss_paper,
        "comb": loss_comb,
        "comb2": loss_comb2,
        "comb_log": loss_comb_log,
        "log": loss_log,
    }
    if loss_type not in losses:
        raise ValueError(f"Unsupported loss_type={loss_type!r}. Available: {', '.join(sorted(losses))}")
    return losses[loss_type]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train AutoPhaseNN/UMamba for 3D diffraction phase retrieval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu_device", type=int, default=0)
    parser.add_argument("--OutputFolder", type=str, default="/lcrc/project/AutoPhase/test_pytorch/")
    parser.add_argument("--DataFolder", type=str, default="/data_hdd/oyys/autophaseNN/")
    parser.add_argument("--plans_file", type=str, default="plans_diffraction_3d.json")

    parser.add_argument("--model_name", type=str, default="autophasenn", choices=["autophasenn", "umamba"])
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--resume_optimizer", nargs="?", const=True, default=True, type=str2bool)

    parser.add_argument("--data_train_diff", type=str, default="train_diff.npy")
    parser.add_argument("--data_train_real", type=str, default="train_real.npy")
    parser.add_argument("--data_val_diff", type=str, default="val_diff.npy")
    parser.add_argument("--data_val_real", type=str, default="val_real.npy")
    parser.add_argument("--num_samples_train", type=int, default=25000)
    parser.add_argument("--num_samples_val", type=int, default=5000)
    parser.add_argument("--train_size", type=int, default=0, help="Optional cap for train samples; 0 means use num_samples_train.")
    parser.add_argument("--train_perc", type=float, default=1.0, help="Legacy argument kept for old launch scripts.")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--save_model", type=int, default=1, help="Save every N epochs; <=0 disables periodic checkpoints.")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--persistent_workers", nargs="?", const=True, default=True, type=str2bool)

    parser.add_argument("--loss_type", type=str, default="mae")
    parser.add_argument("--monitor_metric", type=str, default="loss_comb2")
    parser.add_argument("--amp_loss_weight", type=float, default=1.0)
    parser.add_argument("--phase_loss_weight", type=float, default=1.0)
    parser.add_argument("--unsupervise", nargs="?", const=True, default=False, type=str2bool)

    parser.add_argument("--optim_type", type=str, default="adam", choices=["adam", "adamw"])
    parser.add_argument("--Initlr", type=float, default=1e-5)
    parser.add_argument("--lr_type", type=str, default="clr", choices=["clr", "step", "plateau", "none"])
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--fp16", nargs="?", const=True, default=False, type=str2bool)

    parser.add_argument("--shape", type=int, default=64)
    parser.add_argument("--T", type=float, default=0.1)
    parser.add_argument("--nconv", type=int, default=32)
    parser.add_argument("--use_down_stride", nargs="?", const=True, default=False, type=str2bool)
    parser.add_argument("--use_up_stride", nargs="?", const=True, default=False, type=str2bool)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--scale_I", type=float, default=1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_threads", type=int, default=0)
    parser.add_argument("--notes", type=str, default="test")
    parser.add_argument("--cudnn_benchmark", nargs="?", const=True, default=True, type=str2bool)
    parser.add_argument("--scale_prediction_in_validation", nargs="?", const=True, default=True, type=str2bool)

    return parser.parse_args()


def resolve_relative_to_script(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def path_in_folder(folder: Path, name: str) -> Path:
    path = Path(name)
    return path if path.is_absolute() else folder / path


def setup_runtime(args) -> torch.device:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    else:
        torch.set_num_threads(1)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"
        args.fp16 = False

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu_device)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)

    return device


def dataloader_kwargs(args, device: torch.device) -> Dict:
    kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
        kwargs["persistent_workers"] = args.persistent_workers
    return kwargs


def load_plans(args) -> dict:
    plans_path = resolve_relative_to_script(args.plans_file)
    with open(plans_path, "r", encoding="utf-8") as f:
        plans = json.load(f)
    plans["configurations"]["3d_fullres"]["batch_size"] = args.batch_size
    return plans


def build_model(args) -> nn.Module:
    if args.model_name == "autophasenn":
        from AutoPhaseNN_model_relu import Network

        return Network(args)

    if args.model_name == "umamba":
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        from UMambaEnc_3d import get_umamba_enc_3d_from_plans

        plans_manager = PlansManager(load_plans(args))
        config_manager = plans_manager.get_configuration("3d_fullres")
        dataset_json = {"labels": {"background": 0}, "num_segmentation_heads": 1}
        return get_umamba_enc_3d_from_plans(
            plans_manager=plans_manager,
            dataset_json=dataset_json,
            configuration_manager=config_manager,
            num_input_channels=1,
            deep_supervision=False,
        )

    raise ValueError(f"Unsupported model_name={args.model_name!r}")


def build_optimizer(args, model: nn.Module):
    if args.optim_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.Initlr)
    if args.optim_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.Initlr)
    raise ValueError(f"Unsupported optim_type={args.optim_type!r}")


def build_scheduler(args, optimizer, train_loader: Iterable):
    if args.lr_type == "none":
        return None
    if args.lr_type == "clr":
        step_size = max(1, 6 * len(train_loader))
        print(f"LR step size: {step_size} batches ({step_size / max(1, len(train_loader)):.1f} epochs)")
        return torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=args.Initlr / 10.0,
            max_lr=args.Initlr,
            step_size_up=step_size,
            cycle_momentum=False,
            mode="triangular2",
        )
    if args.lr_type == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    if args.lr_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6)
    raise ValueError(f"Unsupported lr_type={args.lr_type!r}")


def autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=enabled)
    return nullcontext()


def to_device(batch, device: torch.device):
    return tuple(t.to(device, non_blocking=device.type == "cuda") for t in batch)


def align_prediction_scale(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    scale = torch.sum(y_true, dim=positive_spatial_dims(y_true), keepdim=True)
    scale = scale / (torch.sum(y_pred, dim=positive_spatial_dims(y_pred), keepdim=True) + 1e-10)
    return y_pred * scale


def compute_validation_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "loss_mae": loss_mae(y_true, y_pred),
        "loss_mse": loss_sq(y_true, y_pred),
        "loss_huber": F.smooth_l1_loss(y_pred, y_true),
        "loss_pcc": loss_pcc(y_true, y_pred),
        "loss_comb": loss_comb(y_true, y_pred),
        "loss_comb2": loss_comb2(y_true, y_pred),
    }


def tensor_to_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def average_dict(total: Dict[str, float], count: int) -> Dict[str, float]:
    return {key: value / max(1, count) for key, value in total.items()}


def train_epoch(args, model, criterion, train_loader, optimizer, scheduler, scaler, writer, device, epoch):
    model.train()
    start_time = time.time()
    totals = {"loss_total": 0.0, "loss_ft": 0.0, "loss_amp": 0.0, "loss_phase": 0.0}

    progress = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch} train", dynamic_ncols=True)
    for step, batch in enumerate(progress, start=1):
        ft_images, amps, phs = to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, args.fp16):
            y, _, pred_amps, pred_phs, support = model(ft_images)
            loss_ft = criterion(ft_images, y)
            loss_amp = F.l1_loss(pred_amps, amps)
            loss_phase = F.l1_loss(pred_phs * support, phs * support)
            if args.unsupervise:
                loss = loss_ft
            else:
                loss = loss_ft + args.amp_loss_weight * loss_amp + args.phase_loss_weight * loss_phase

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

        if scheduler is not None and args.lr_type == "clr":
            scheduler.step()

        step_values = {
            "loss_total": tensor_to_float(loss),
            "loss_ft": tensor_to_float(loss_ft),
            "loss_amp": tensor_to_float(loss_amp),
            "loss_phase": tensor_to_float(loss_phase),
        }
        for key, value in step_values.items():
            totals[key] += value

        global_step = (epoch - 1) * len(train_loader) + step
        writer.add_scalar("train_step/loss_total", step_values["loss_total"], global_step)
        writer.add_scalar("train_step/loss_ft", step_values["loss_ft"], global_step)
        progress.set_postfix(
            loss=f"{step_values['loss_total']:.4e}",
            ft=f"{step_values['loss_ft']:.4e}",
            lr=f"{optimizer.param_groups[0]['lr']:.4e}",
        )

    metrics = average_dict(totals, len(train_loader))
    metrics["seconds"] = time.time() - start_time
    print(
        f"Epoch {epoch} train | total={metrics['loss_total']:.4e} "
        f"ft={metrics['loss_ft']:.4e} amp={metrics['loss_amp']:.4e} "
        f"phase={metrics['loss_phase']:.4e} time={metrics['seconds']:.2f}s"
    )
    return metrics


def validate(args, model, valid_loader, writer, device, epoch):
    model.eval()
    totals = {
        "loss_mae": 0.0,
        "loss_mse": 0.0,
        "loss_huber": 0.0,
        "loss_pcc": 0.0,
        "loss_comb": 0.0,
        "loss_comb2": 0.0,
        "loss_amp_l1": 0.0,
        "loss_phase_l1": 0.0,
    }

    progress = tqdm(valid_loader, total=len(valid_loader), desc=f"Epoch {epoch} val", dynamic_ncols=True)
    with torch.no_grad():
        for batch in progress:
            ft_images, amps, phs = to_device(batch, device)
            with autocast_context(device, args.fp16):
                y, _, pred_amps, pred_phs, support = model(ft_images)
                if args.scale_prediction_in_validation:
                    y = align_prediction_scale(ft_images, y)
                details = compute_validation_metrics(ft_images, y)
                details["loss_amp_l1"] = F.l1_loss(pred_amps, amps)
                details["loss_phase_l1"] = F.l1_loss(pred_phs * support, phs * support)

            for key in totals:
                totals[key] += tensor_to_float(details[key])
            progress.set_postfix(loss_comb2=f"{tensor_to_float(details['loss_comb2']):.4e}")

    metrics = average_dict(totals, len(valid_loader))
    print(
        f"Epoch {epoch} val | comb2={metrics['loss_comb2']:.4e} "
        f"mae={metrics['loss_mae']:.4e} pcc={metrics['loss_pcc']:.4e}"
    )
    writer.add_scalars("validation", metrics, epoch)
    return metrics


def clean_state_dict_keys(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model.", "net."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def load_checkpoint_if_available(args, model, optimizer, scheduler, scaler, device):
    checkpoint_path = args.checkpoint.strip()
    start_epoch = 0
    training_history = []
    validation_history = []
    best_val_loss = float("inf")

    if not checkpoint_path:
        print("No checkpoint provided; training starts from scratch.")
        return start_epoch, training_history, validation_history, best_val_loss

    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists() or checkpoint_file.is_dir():
        print(f"[WARN] Checkpoint not found or not a file: {checkpoint_file}. Training starts from scratch.")
        return start_epoch, training_history, validation_history, best_val_loss

    checkpoint = torch.load(checkpoint_file, map_location=device)
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))

    missing_keys, unexpected_keys = model.load_state_dict(clean_state_dict_keys(state_dict), strict=False)
    print(
        f"Loaded checkpoint weights from {checkpoint_file} "
        f"(missing={len(missing_keys)}, unexpected={len(unexpected_keys)})."
    )

    if isinstance(checkpoint, dict):
        if args.resume_optimizer:
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if scheduler is not None and checkpoint.get("scheduler") is not None:
                scheduler.load_state_dict(checkpoint["scheduler"])
            if "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = int(checkpoint.get("epoch", 0))
        training_history = checkpoint.get("training_losses", [])
        validation_history = checkpoint.get("validation_losses", [])
        best_val_loss = tensor_to_float(checkpoint.get("best_val_loss", best_val_loss))

    return start_epoch, training_history, validation_history, best_val_loss


def save_checkpoint(path: Path, epoch, model, optimizer, scheduler, scaler, training_history, validation_history, best_val_loss):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict(),
            "training_losses": training_history,
            "validation_losses": validation_history,
            "best_val_loss": float(best_val_loss),
        },
        path,
    )


def write_settings(args, output_folder: Path, device: torch.device):
    payload = vars(args).copy()
    payload["resolved_device"] = str(device)
    payload["script"] = str(Path(__file__).resolve())
    with open(output_folder / "setting.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)


def main():
    args = parse_args()
    print(f"Starting training | pytorch={torch.__version__}")
    device = setup_runtime(args)

    output_folder = Path(args.OutputFolder)
    output_folder.mkdir(parents=True, exist_ok=True)
    write_settings(args, output_folder, device)

    data_folder = Path(args.DataFolder)
    train_samples = args.num_samples_train if args.train_size <= 0 else min(args.train_size, args.num_samples_train)
    sample_shape = (args.shape, args.shape, args.shape)

    train_dataset = Dataset(
        path_in_folder(data_folder, args.data_train_diff),
        path_in_folder(data_folder, args.data_train_real),
        train_samples,
        shape_diff=sample_shape,
        shape_real=sample_shape,
        dtype_diff="float32",
        dtype_real="complex64",
        scale_I=args.scale_I,
        shuffle=True,
        seed=args.seed,
    )
    validation_dataset = Dataset(
        path_in_folder(data_folder, args.data_val_diff),
        path_in_folder(data_folder, args.data_val_real),
        args.num_samples_val,
        shape_diff=sample_shape,
        shape_real=sample_shape,
        dtype_diff="float32",
        dtype_real="complex64",
        scale_I=args.scale_I,
        shuffle=False,
        seed=args.seed,
    )

    loader_kwargs = dataloader_kwargs(args, device)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    if len(train_loader) == 0 or len(validation_loader) == 0:
        raise RuntimeError("Train/validation DataLoader is empty. Check sample counts and batch size.")

    model = build_model(args).to(device)
    print(f"Model: {args.model_name} | parameters={sum(p.numel() for p in model.parameters()):,}")

    criterion = build_reconstruction_loss(args.loss_type)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer, train_loader)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    start_epoch, training_history, validation_history, best_val_loss = load_checkpoint_if_available(
        args, model, optimizer, scheduler, scaler, device
    )

    writer = SummaryWriter(log_dir=str(output_folder / "tensorboard"), comment=Path(output_folder).name)
    t0 = time.time()
    try:
        for epoch in range(start_epoch + 1, args.epoch + 1):
            train_metrics = train_epoch(args, model, criterion, train_loader, optimizer, scheduler, scaler, writer, device, epoch)
            training_history.append(train_metrics)
            writer.add_scalars("train", {k: v for k, v in train_metrics.items() if k != "seconds"}, epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

            val_metrics = validate(args, model, validation_loader, writer, device, epoch)
            validation_history.append(val_metrics)

            if args.lr_type == "step" and scheduler is not None:
                scheduler.step()
            elif args.lr_type == "plateau" and scheduler is not None:
                scheduler.step(val_metrics[args.monitor_metric])

            current_val_loss = val_metrics.get(args.monitor_metric)
            if current_val_loss is None:
                raise KeyError(f"monitor_metric={args.monitor_metric!r} is not in validation metrics.")

            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                save_checkpoint(
                    output_folder / "best_model.pt",
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    training_history,
                    validation_history,
                    best_val_loss,
                )
                print(f"Saved new best model | {args.monitor_metric}={best_val_loss:.6f} | epoch={epoch}")

            if args.save_model > 0 and epoch % args.save_model == 0:
                checkpoint_path = output_folder / f"training_model_{epoch:06d}.pt"
                save_checkpoint(
                    checkpoint_path,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    training_history,
                    validation_history,
                    best_val_loss,
                )
                print(f"Saved checkpoint: {checkpoint_path}")
    finally:
        writer.flush()
        writer.close()

    total_time = time.time() - t0
    epochs_ran = max(1, args.epoch - start_epoch)
    print(f"Total running time: {total_time:.2f}s")
    print(f"Average time per epoch: {total_time / epochs_ran:.2f}s")


if __name__ == "__main__":
    main()
