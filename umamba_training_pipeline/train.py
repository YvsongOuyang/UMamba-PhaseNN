from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import AutoPhaseDataset  # noqa: E402
from losses import (  # noqa: E402
    chi2_modulus,
    chi2_pcc_loss,
    get_loss,
    metric_dict,
    pearson_loss,
    relative_l1_modulus,
    scale_align_sum,
    sqrt_chi2_pcc_loss,
)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y", "on"):
        return True
    if value in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def choose_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(name)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def safe_path_name(value):
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))
    return safe.strip("._") or "experiment"


def optional_data_path(data_dir, filename):
    if filename is None or str(filename).lower() in {"", "none", "null"}:
        return None
    path = Path(filename)
    if path.is_absolute():
        return path
    return Path(data_dir) / path


def clean_state_dict_keys(state_dict):
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "model.", "net."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        cleaned_state_dict[key] = value
    return cleaned_state_dict


def load_matching_model_weights(model, state_dict, max_report_keys=12):
    model_state = model.state_dict()
    matched_state = {}
    skipped_shape = []
    unexpected_keys = []
    non_tensor_keys = []

    for key, value in clean_state_dict_keys(state_dict).items():
        if key not in model_state:
            unexpected_keys.append(key)
            continue
        if not torch.is_tensor(value):
            non_tensor_keys.append(key)
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            skipped_shape.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        matched_state[key] = value

    missing_keys = [key for key in model_state if key not in matched_state]
    incompatible = model.load_state_dict(matched_state, strict=False)
    can_resume_state = (
        len(matched_state) == len(model_state)
        and not missing_keys
        and not skipped_shape
        and not unexpected_keys
        and not non_tensor_keys
    )

    print(
        "Checkpoint weight load summary: "
        f"loaded={len(matched_state)}/{len(model_state)} | "
        f"missing={len(missing_keys)} | unexpected={len(unexpected_keys)} | "
        f"shape_mismatch={len(skipped_shape)} | non_tensor={len(non_tensor_keys)}",
        flush=True,
    )
    if missing_keys:
        print(f"  missing sample: {missing_keys[:max_report_keys]}", flush=True)
    if unexpected_keys:
        print(f"  unexpected sample: {unexpected_keys[:max_report_keys]}", flush=True)
    if skipped_shape:
        print(f"  shape mismatch sample: {skipped_shape[:max_report_keys]}", flush=True)
    if non_tensor_keys:
        print(f"  non-tensor sample: {non_tensor_keys[:max_report_keys]}", flush=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            "  load_state_dict report: "
            f"missing={len(incompatible.missing_keys)} | "
            f"unexpected={len(incompatible.unexpected_keys)}",
            flush=True,
        )

    return {
        "loaded": len(matched_state),
        "total": len(model_state),
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "skipped_shape": skipped_shape,
        "non_tensor_keys": non_tensor_keys,
        "can_resume_state": can_resume_state,
    }


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model", "net"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def get_checkpoint_value(checkpoint, *keys, default=None):
    if not isinstance(checkpoint, dict):
        return default
    for key in keys:
        if key in checkpoint:
            return checkpoint[key]
    return default


def build_umamba_plans(args):
    shape = int(args.shape)
    return {
        "dataset_name": "Diffraction3D",
        "original_median_spacing_after_transp": [1.0, 1.0, 1.0],
        "original_median_shape_after_transp": [shape, shape, shape],
        "image_reader_writer": "SimpleITKIO",
        "transpose_forward": [0, 1, 2],
        "transpose_backward": [0, 1, 2],
        "configurations": {
            "3d_fullres": {
                "data_identifier": "nnUNetPlans_3d_fullres",
                "preprocessor_name": "DefaultPreprocessor",
                "batch_size": args.batch_size,
                "patch_size": [shape, shape, shape],
                "median_image_size_in_voxels": [float(shape), float(shape), float(shape)],
                "spacing": [1.0, 1.0, 1.0],
                "normalization_schemes": ["ZScoreNormalization"],
                "use_mask_for_norm": [False],
                "UNet_class_name": "PlainConvUNet",
                "UNet_base_num_features": 32,
                "n_conv_per_stage_encoder": [2, 2, 2, 2],
                "n_conv_per_stage_decoder": [2, 2, 2],
                "num_pool_per_axis": [4, 4, 4],
                "pool_op_kernel_sizes": [
                    [1, 1, 1],
                    [2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 2],
                ],
                "conv_kernel_sizes": [
                    [3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3],
                ],
                "unet_max_num_features": 320,
                "resampling_fn_data": "fast_resample_data_or_seg_to_shape",
                "resampling_fn_seg": "fast_resample_data_or_seg_to_shape",
                "resampling_fn_data_kwargs": {
                    "is_seg": False,
                    "order": 3,
                    "order_z": 3,
                    "force_separate_z": None,
                },
                "resampling_fn_seg_kwargs": {
                    "is_seg": True,
                    "order": 1,
                    "order_z": 1,
                    "force_separate_z": None,
                },
                "resampling_fn_probabilities": "fast_resample_data_or_seg_to_shape",
                "resampling_fn_probabilities_kwargs": {
                    "is_seg": False,
                    "order": 1,
                    "order_z": 1,
                    "force_separate_z": None,
                },
                "batch_dice": True,
            },
            "3d_cascade_fullres": {
                "inherits_from": "3d_fullres",
                "previous_stage": "3d_lowres",
            },
        },
        "experiment_planner_used": "ExperimentPlanner",
        "label_manager": "LabelManager",
        "foreground_intensity_properties_per_channel": {
            "0": {
                "max": 3071.0,
                "mean": 97.29716491699219,
                "median": 118.0,
                "min": -1024.0,
                "percentile_00_5": -958.0,
                "percentile_99_5": 270.0,
                "std": 137.8484649658203,
            }
        },
    }


def build_model(args, device):
    model_name = args.model_name.lower()
    if model_name == "umamba":
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        from umamba_training_pipeline.UMambaEnc_3d import get_umamba_enc_3d_from_plans

        plans_manager = PlansManager(build_umamba_plans(args))
        config_manager = plans_manager.get_configuration("3d_fullres")
        dataset_json = {"labels": {"background": 0}, "num_segmentation_heads": 1}
        model = get_umamba_enc_3d_from_plans(
            plans_manager=plans_manager,
            dataset_json=dataset_json,
            configuration_manager=config_manager,
            num_input_channels=1,
            deep_supervision=False,
            phase_activation=args.phase_activation,
            phase_logit_scale=args.phase_logit_scale,
            threshold=args.T,
            center_pad_last_upsample=args.center_pad_last_upsample,
        )
        print(
            f"Using UMamba | phase_activation={args.phase_activation} | "
            f"phase_logit_scale={args.phase_logit_scale:.4e} | "
            f"center_pad_last_upsample={args.center_pad_last_upsample}",
            flush=True,
        )
    elif model_name == "autophasenn":
        from AutoPhaseNN_model import Network as AutoPhaseNNNetwork

        model = AutoPhaseNNNetwork(args)
        print("Using AutoPhaseNN_model.py", flush=True)
    elif model_name == "autophasenn_relu":
        from AutoPhaseNN_model_relu import Network as AutoPhaseNNReluNetwork

        model = AutoPhaseNNReluNetwork(args)
        print("Using AutoPhaseNN_model_relu.py", flush=True)
    else:
        raise ValueError(f"Unsupported model_name: {args.model_name}")
    return model.to(device)


def build_optimizer_param_groups(model, lr, head_lr_mult):
    if head_lr_mult == 1.0:
        return model.parameters()

    head_params = []
    head_ids = set()
    for name in ("decoder1", "decoder2"):
        module = getattr(model, name, None)
        if module is None:
            continue
        for param_name, param in module.named_parameters():
            if param_name.startswith("encoder."):
                continue
            if param.requires_grad:
                head_params.append(param)
                head_ids.add(id(param))

    if not head_params:
        print(
            "[WARN] head_lr_mult requested but decoder1/decoder2 params were not found; "
            "using a single lr group.",
            flush=True,
        )
        return model.parameters()

    base_params = [
        param
        for param in model.parameters()
        if param.requires_grad and id(param) not in head_ids
    ]
    print(
        f"Optimizer param groups: base_lr={lr:.3e} ({len(base_params)} tensors) | "
        f"head_lr={lr * head_lr_mult:.3e} ({len(head_params)} tensors)",
        flush=True,
    )
    return [
        {"params": base_params, "lr": lr},
        {"params": head_params, "lr": lr * head_lr_mult},
    ]


def make_scheduler(args, optimizer, start_epoch, steps_per_epoch):
    if args.lr_type == "none":
        return None
    if args.lr_type == "clr":
        step_size = max(1, int(round(args.clr_step_epochs * max(1, steps_per_epoch))))
        print(
            f"LR step size is: {step_size}, which is every "
            f"{step_size / max(1, steps_per_epoch):.2f} epochs",
            flush=True,
        )
        return torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=args.lr / 10.0,
            max_lr=args.lr,
            step_size_up=step_size,
            cycle_momentum=False,
            mode="triangular2",
        )
    if args.lr_type == "cosine":
        remaining_epochs = max(1, args.epochs - start_epoch + 1)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=remaining_epochs, eta_min=args.min_lr
        )
    if args.lr_type == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    if args.lr_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=args.gamma, patience=args.patience, min_lr=args.min_lr
        )
    raise ValueError(f"Unsupported lr_type: {args.lr_type}")


def build_datasets(args):
    data_dir = Path(args.data_dir)
    shape = (args.shape, args.shape, args.shape)
    train_samples = args.num_samples_train
    if args.train_size and args.train_size > 0:
        train_samples = min(train_samples, args.train_size)

    overfit_samples = max(args.overfit_samples, args.debug_overfit_samples, 0)
    overfit_mode = overfit_samples > 0
    if overfit_mode:
        train_samples = min(train_samples, overfit_samples)
    cache_data = args.cache_data or overfit_mode

    train_dataset = AutoPhaseDataset(
        optional_data_path(data_dir, args.data_train_diff),
        optional_data_path(data_dir, args.data_train_real),
        num_samples=train_samples,
        shape_diff=shape,
        shape_real=shape,
        dtype_diff=args.dtype_diff,
        dtype_real=args.dtype_real,
        scale_i=args.scale_i,
        shuffle=False,
        seed=args.seed,
        cache_data=cache_data,
        allow_missing_real=args.allow_missing_real,
    )
    if overfit_mode:
        val_dataset = train_dataset
    else:
        val_dataset = AutoPhaseDataset(
            optional_data_path(data_dir, args.data_val_diff),
            optional_data_path(data_dir, args.data_val_real),
            num_samples=args.num_samples_val,
            shape_diff=shape,
            shape_real=shape,
            dtype_diff=args.dtype_diff,
            dtype_real=args.dtype_real,
            scale_i=args.scale_i,
            shuffle=False,
            seed=args.seed,
            cache_data=cache_data,
            allow_missing_real=args.allow_missing_real,
        )
    print(
        f"Resolved memmap samples: train={len(train_dataset)}, val={len(val_dataset)} | "
        f"overfit={overfit_mode} | cache_data={cache_data}",
        flush=True,
    )
    if overfit_mode:
        print(
            f"[DEBUG] Overfit mode enabled: using the same {len(train_dataset)} "
            "train samples for both train and validation.",
            flush=True,
        )
    return train_dataset, val_dataset, overfit_mode


def build_loaders(args):
    train_dataset, val_dataset, overfit_mode = build_datasets(args)
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": args.device == "cuda",
    }
    if args.num_workers > 0:
        kwargs.update({"prefetch_factor": 2, "persistent_workers": True})
    print(f"DataLoader kwargs: {kwargs}", flush=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        **kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **kwargs,
    )
    return train_loader, val_loader, overfit_mode


def unpack_batch(batch, device):
    diff = batch["diff"].to(device, non_blocking=True).float()
    amp = batch["amp"].to(device, non_blocking=True).float()
    phi = batch["phi"].to(device, non_blocking=True).float()
    return diff, amp, phi


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


def support_threshold(args):
    return float(getattr(args, "T", getattr(args, "threshold", 0.1)))


def compute_loss_bundle(args, loss_fn, diff, amp, phi, outputs):
    pred_diff, _pred_obj, pred_amp, pred_phi, support = unpack_outputs(outputs)
    raw_amp = raw_amp_from_outputs(outputs, pred_amp)
    pred_for_loss = scale_align_sum(diff, pred_diff) if args.scale_align_loss else pred_diff
    loss_ft = loss_fn(diff, pred_for_loss)
    loss_amp = F.l1_loss(pred_amp, amp)
    loss_phase = F.l1_loss(pred_phi * support, phi * support)
    target_support = (amp >= support_threshold(args)).float()
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


def debug_grad_norm_message(model):
    def norm_for_params(named_params):
        sq_sum = 0.0
        max_abs = 0.0
        tensors = 0
        for _name, param in named_params:
            if param.grad is None:
                continue
            grad = param.grad.detach()
            sq_sum += float(torch.sum(grad.float() ** 2).cpu())
            max_abs = max(max_abs, float(grad.abs().max().cpu()))
            tensors += 1
        return sq_sum ** 0.5, max_abs, tensors

    total = norm_for_params(model.named_parameters())
    groups = []
    for prefix in ("encoder", "decoder1", "decoder2"):
        params = [(n, p) for n, p in model.named_parameters() if n.startswith(prefix)]
        norm, max_abs, count = norm_for_params(params)
        groups.append(f"{prefix}:norm={norm:.3e},max={max_abs:.3e},n={count}")
    return f"total={total[0]:.3e} | " + " | ".join(groups)


def save_checkpoint(path, args, model, optimizer, scheduler, scaler, epoch, history, best_val_loss, global_step):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "training_losses": history["train"],
            "validation_losses": history["val"],
            "history": history,
            "best_val_loss": best_val_loss,
            "global_step": global_step,
            "args": vars(args),
        },
        path,
    )


def train_one_epoch(args, model, loss_fn, trainloader, optimizer, scheduler, epoch, scaler, device, writer=None, global_step=0):
    start_time = time.time()
    model.train()
    num_batches = len(trainloader)
    if args.debug_max_train_batches > 0:
        num_batches = min(num_batches, args.debug_max_train_batches)
    use_amp = args.fp16 and device.type == "cuda"
    epoch_start_time = time.time()
    loss_total = 0.0
    loss_ft_total = 0.0
    loss_amp_total = 0.0
    loss_phase_total = 0.0
    loss_support_total = 0.0

    for i, batch in enumerate(trainloader):
        if i >= num_batches:
            break
        diff, amp, phi = unpack_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(diff)
            loss, loss_ft, loss_amp, loss_phase, loss_support, pred_for_loss = compute_loss_bundle(
                args, loss_fn, diff, amp, phi, outputs
            )
            pred_diff, pred_obj, _pred_amp, pred_phi, support = unpack_outputs(outputs)

        track_output_delta = args.debug_output_delta and i < args.debug_output_delta_batches
        if track_output_delta:
            pre_loss = loss.detach().float()
            pre_y = pred_for_loss.detach().float()
            pre_raw_amp = torch.abs(pred_obj.detach()).float()
            pre_phs = pred_phi.detach().float()
            pre_support = support.detach().float()

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        if args.debug_grad_norm and i < args.debug_grad_norm_batches:
            print(
                f"[GRAD] Epoch[{epoch}] Batch[{i + 1}] | Loss={loss.item():.4e} | "
                f"{debug_grad_norm_message(model)}",
                flush=True,
            )

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

        has_grad = any(
            p.grad is not None and bool((p.grad.detach().abs().max() > 0).item())
            for p in model.parameters()
        )
        before = next(model.parameters()).data.flatten()[0].item()
        if scaler.is_enabled():
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_stepped = scaler.get_scale() >= scale_before
        else:
            optimizer.step()
            optimizer_stepped = True
        after = next(model.parameters()).data.flatten()[0].item()

        if track_output_delta:
            with torch.no_grad():
                after_outputs = model(diff)
                after_loss, _after_ft, _after_amp, _after_phase, _after_support, after_pred = compute_loss_bundle(
                    args, loss_fn, diff, amp, phi, after_outputs
                )
                _after_diff, pred_obj_after, _amp_after, pred_phs_after, support_after = unpack_outputs(after_outputs)
                post_y = after_pred.detach().float()
                post_raw_amp = torch.abs(pred_obj_after.detach()).float()
                post_phs = pred_phs_after.detach().float()
                post_support = support_after.detach().float()
            print(
                f"[OUTPUT_DELTA] Epoch[{epoch}] Batch[{i + 1}] | "
                f"LossBefore={pre_loss.item():.4e} | LossAfter={after_loss.item():.4e} | "
                f"DeltaLoss={(after_loss - pre_loss).item():+.4e} | "
                f"MeanAbsDeltaY={(post_y - pre_y).abs().mean().item():.4e} | "
                f"MeanAbsDeltaAmp={(post_raw_amp - pre_raw_amp).abs().mean().item():.4e} | "
                f"MeanAbsDeltaPh={(post_phs - pre_phs).abs().mean().item():.4e} | "
                f"MaxAbsDeltaPh={(post_phs - pre_phs).abs().max().item():.4e} | "
                f"MeanAbsDeltaSupport={(post_support - pre_support).abs().mean().item():.4e} | "
                f"PostAmpMean={post_raw_amp.mean().item():.4e} | "
                f"PostSupportMean={post_support.mean().item():.4e}",
                flush=True,
            )

        if optimizer_stepped:
            global_step += 1
        if optimizer_stepped and scheduler is not None and args.lr_type == "clr":
            scheduler.step()
        current_batch_lr = optimizer.param_groups[0]["lr"]
        if optimizer_stepped and writer is not None:
            writer.add_scalar("Lr", current_batch_lr, global_step)
            writer.add_scalar("loss/train_loss", loss.item(), global_step)
            writer.add_scalar("loss/train_loss_ft", loss_ft.item(), global_step)
            writer.add_scalar("loss/train_loss_amp", loss_amp.item(), global_step)
            writer.add_scalar("loss/train_loss_phase", loss_phase.item(), global_step)
            writer.add_scalar("loss/train_loss_support", loss_support.item(), global_step)

        loss_total += loss.detach().item()
        loss_ft_total += loss_ft.detach().item()
        loss_amp_total += loss_amp.detach().item()
        loss_phase_total += loss_phase.detach().item()
        loss_support_total += loss_support.detach().item()
        current_iter = i + 1
        elapsed_seconds = time.time() - epoch_start_time
        avg_time_per_iter = elapsed_seconds / current_iter
        eta_seconds = (num_batches - current_iter) * avg_time_per_iter
        elapsed_str = str(timedelta(seconds=int(elapsed_seconds)))
        eta_str = str(timedelta(seconds=int(eta_seconds)))

        if i % args.print_freq == 0 or current_iter == num_batches:
            print(
                f"Epoch[{epoch}] Batch[{current_iter}/{num_batches}] | "
                f"OptTotal: {loss.item():.4e} | FTLoss: {loss_ft.item():.4e} | "
                f"AmpL1Full: {loss_amp.item():.4e} | "
                f"PhaseL1PredSup: {loss_phase.item():.4e} | "
                f"SupportBCE: {loss_support.item():.4e} | "
                f"SupportWeighted: {(args.support_weight * loss_support.item()):.4e} | "
                f"BatchLR: {current_batch_lr:.3e} | "
                f"Grad: {str(has_grad):5s} | Update: {str(before != after):5s} | "
                f"Elapsed: {elapsed_str} | ETA: {eta_str}",
                flush=True,
            )

    loss_total /= max(num_batches, 1)
    loss_ft_total /= max(num_batches, 1)
    loss_amp_total /= max(num_batches, 1)
    loss_phase_total /= max(num_batches, 1)
    loss_support_total /= max(num_batches, 1)
    time_cost = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"Epoch {epoch} complete | total time: {time_cost:.2f}s | average OptTotal: {loss_total:.4e}")
    print(
        f"Average train optimization terms | FTLoss: {loss_ft_total:.4e} | "
        f"SupportWeighted: {(args.support_weight * loss_support_total):.4e}"
    )
    print(
        f"Average train real-space monitors | AmpL1Full: {loss_amp_total:.4e} | "
        f"PhaseL1PredSup: {loss_phase_total:.4e} | "
        f"SupportBCE: {loss_support_total:.4e}"
    )
    print("=" * 80 + "\n")
    return loss_total, global_step


def validate(args, model, loss_fn, validloader, epoch, device):
    model.eval()
    num_batches = len(validloader)
    if args.debug_max_val_batches > 0:
        num_batches = min(num_batches, args.debug_max_val_batches)
    use_amp = args.fp16 and device.type == "cuda"
    details_total = {
        "loss": 0.0,
        "loss_ft": 0.0,
        "loss_amp": 0.0,
        "loss_phase": 0.0,
        "loss_support": 0.0,
        "loss_l1": 0.0,
        "loss_mae": 0.0,
        "loss_mse": 0.0,
        "loss_huber": 0.0,
        "loss_pcc": 0.0,
        "loss_comb": 0.0,
        "loss_comb2": 0.0,
        "metric_log": 0.0,
        "voxel_mse": 0.0,
        "voxel_rmse": 0.0,
    }

    with torch.no_grad():
        for i, batch in enumerate(validloader):
            if i >= num_batches:
                break
            diff, amp, phi = unpack_batch(batch, device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(diff)
                loss, loss_ft, loss_amp, loss_phase, loss_support, pred_for_loss = compute_loss_bundle(
                    args, loss_fn, diff, amp, phi, outputs
                )
                details = {
                    "loss": loss,
                    "loss_ft": loss_ft,
                    "loss_amp": loss_amp,
                    "loss_phase": loss_phase,
                    "loss_support": loss_support,
                    "loss_l1": F.l1_loss(pred_for_loss, diff),
                    "loss_mae": relative_l1_modulus(diff, pred_for_loss),
                    "loss_mse": chi2_modulus(diff, pred_for_loss),
                    "loss_huber": nn.SmoothL1Loss()(pred_for_loss, diff),
                    "loss_pcc": pearson_loss(diff, pred_for_loss),
                    "loss_comb": chi2_pcc_loss(diff, pred_for_loss),
                    "loss_comb2": sqrt_chi2_pcc_loss(diff, pred_for_loss),
                }
                metrics = metric_dict(diff, pred_for_loss.detach())
                details["metric_log"] = torch.tensor(metrics["relative_log_mse"], device=device)
                details["voxel_mse"] = torch.tensor(metrics["voxel_mse"], device=device)
                details["voxel_rmse"] = torch.tensor(metrics["voxel_rmse"], device=device)

            for key in details_total:
                details_total[key] += float(details[key].detach().float().cpu().item())

            if (i + 1) % args.print_freq == 0 or (i + 1) == num_batches:
                pcc_corr = 1.0 - details["loss_pcc"].detach().float().item()
                print(
                    f"Validation Epoch [{epoch}] | Batch [{i + 1}/{num_batches}] | "
                    f"OptTotal: {details['loss']:.4e} | FTLoss: {details['loss_ft']:.4e} | "
                    f"AmpL1Full: {details['loss_amp']:.4e} | "
                    f"PhaseL1PredSup: {details['loss_phase']:.4e} | "
                    f"SupportBCE: {details['loss_support']:.4e} | "
                    f"FT_L1: {details['loss_l1']:.4e} | "
                    f"FT_RelL1: {details['loss_mae']:.4e} | "
                    f"FT_Chi2: {details['loss_mse']:.4e} | "
                    f"FT_PearsonCorr: {pcc_corr:.4e}",
                    flush=True,
                )

    for key in details_total:
        details_total[key] /= max(num_batches, 1)

    print("\n" + "=" * 80)
    print(f"Epoch {epoch} validation complete")
    print("Average validation metrics:")
    print(
        f"   Optimization: OptTotal={details_total['loss']:.4e} | "
        f"FTLoss={details_total['loss_ft']:.4e} | "
        f"SupportWeighted={(args.support_weight * details_total['loss_support']):.4e}"
    )
    print(
        f"   Real-space monitors: AmpL1Full={details_total['loss_amp']:.4e} | "
        f"PhaseL1PredSup={details_total['loss_phase']:.4e} | "
        f"SupportBCE={details_total['loss_support']:.4e}"
    )
    print(
        f"   Reciprocal primary: FT_L1={details_total['loss_l1']:.4e} | "
        f"FT_RelL1={details_total['loss_mae']:.4e} | "
        f"FT_Chi2={details_total['loss_mse']:.4e} | "
        f"FT_PearsonCorr={(1.0 - details_total['loss_pcc']):.4e}"
    )
    print(
        f"   Reciprocal diagnostics: FT_Huber={details_total['loss_huber']:.4e} | "
        f"FT_PearsonLoss={details_total['loss_pcc']:.4e} | "
        f"FT_LogRelMSE={details_total['metric_log']:.4e} | "
        f"FT_Comb={details_total['loss_comb']:.4e} | "
        f"FT_Comb2={details_total['loss_comb2']:.4e}"
    )
    print(
        f"   Raw-scale diagnostics: FT_VoxelMSE={details_total['voxel_mse']:.4e} | "
        f"FT_VoxelRMSE={details_total['voxel_rmse']:.4e}"
    )
    print("=" * 80 + "\n")
    return details_total


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid UMamba training pipeline.")
    parser.add_argument("--model-name", "--model_name", dest="model_name", choices=["umamba", "autophasenn", "autophasenn_relu"], default="umamba")
    parser.add_argument("--phase-activation", "--phase_activation", dest="phase_activation", choices=["tanh", "atan"], default="tanh")
    parser.add_argument("--phase-logit-scale", "--phase_logit_scale", dest="phase_logit_scale", type=float, default=1.0)
    parser.add_argument(
        "--center-pad-last-upsample",
        "--center_pad_last_upsample",
        dest="center_pad_last_upsample",
        type=str2bool,
        default=True,
        help="For UMamba, replace the final decoder upsample+outer skip with center zero-padding.",
    )
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
    parser.add_argument("--allow-missing-real", action="store_true")
    parser.add_argument("--output-dir", default="./umamba_training_pipeline/output/")
    parser.add_argument("--tensorboard-dir", default="runs")
    parser.add_argument("--train-size", type=int, default=0)
    parser.add_argument("--overfit-samples", type=int, default=0)
    parser.add_argument("--debug-overfit-samples", type=int, default=0)
    parser.add_argument("--cache-data", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--T", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--nconv", type=int, default=32)
    parser.add_argument("--use-down-stride", "--use_down_stride", dest="use_down_stride", type=str2bool, default=False)
    parser.add_argument("--use-up-stride", "--use_up_stride", dest="use_up_stride", type=str2bool, default=False)
    parser.add_argument("--n-blocks", "--n_blocks", dest="n_blocks", type=int, default=4)
    parser.add_argument("--unsupervise", type=str2bool, default=False)
    parser.add_argument("--unsupervised", action="store_true")
    parser.add_argument("--scale-i", "--scale-I", dest="scale_i", type=float, default=0.0)
    parser.add_argument("--scale-align-loss", action="store_true")
    parser.add_argument("--loss-type", default="l1")
    parser.add_argument("--loss-scope", choices=["diff", "supervised"], default="diff")
    parser.add_argument("--ft-weight", type=float, default=1.0)
    parser.add_argument("--amp-weight", type=float, default=1.0)
    parser.add_argument("--phase-weight", type=float, default=1.0)
    parser.add_argument(
        "--support-weight",
        type=float,
        default=0.0,
        help="Optional balanced BCE(raw_amp, amp>=T) support-shape loss. Default keeps paper-style diff-only training.",
    )
    parser.add_argument("--lr", "--Initlr", dest="lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", "--optim_type", dest="optimizer", choices=["adam", "adamw"], default="adam")
    parser.add_argument("--lr-type", "--lr_scheduler", "--lr-scheduler", dest="lr_type", choices=["none", "clr", "cosine", "step", "plateau"], default="plateau")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--clr-step-epochs", type=float, default=6.0)
    parser.add_argument("--step-size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--checkpoint", "--resume", dest="checkpoint",  default="/data_ssd/oyys/autophasenn/Unsupfalse_Dfalse_Ufalse_T0.1_comb2_batch8_plateau_Init1e-3_adam_scale1/best_model.pt")
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--head-lr-mult", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--save-model", type=int, default=20)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument("--debug-max-train-batches", type=int, default=0)
    parser.add_argument("--debug-max-val-batches", type=int, default=0)
    parser.add_argument("--debug-skip-scheduler", action="store_true")
    parser.add_argument("--debug-grad-norm", action="store_true")
    parser.add_argument("--debug-grad-norm-batches", type=int, default=1)
    parser.add_argument("--debug-output-delta", action="store_true")
    parser.add_argument("--debug-output-delta-batches", type=int, default=1)
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.unsupervised or args.unsupervise:
        args.loss_scope = "diff"
    if args.from_scratch and args.checkpoint:
        raise ValueError("--from-scratch cannot be combined with --checkpoint/--resume.")

    torch.manual_seed(args.seed)
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.cuda.set_device(0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "setting.json", vars(args))
    save_json(output_dir / "config.json", vars(args))

    print("Training setup")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print(f"use device: {device}", flush=True)
    print(f"Torch threads: {torch.get_num_threads()}", flush=True)

    experiment_name = safe_path_name(Path(output_dir).name)
    tb_run_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{experiment_name}"
    writer = SummaryWriter(log_dir=str(Path(args.tensorboard_dir) / tb_run_name))
    print(f"TensorBoard log dir: {Path(args.tensorboard_dir, tb_run_name).resolve()}", flush=True)

    model = build_model(args, device)
    print(f"model parameters: {sum(param.nelement() for param in model.parameters())}", flush=True)

    checkpoint = None
    load_success = False
    resume_state_success = False
    if args.checkpoint:
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            print(f"Checkpoint file loaded: {args.checkpoint}", flush=True)
            state_dict = extract_state_dict(checkpoint)
            report = load_matching_model_weights(model, state_dict)
            load_success = report["loaded"] > 0
            resume_state_success = (
                load_success
                and isinstance(checkpoint, dict)
                and report["can_resume_state"]
                and any(
                    key in checkpoint
                    for key in ("epoch", "optimizer_state_dict", "scheduler", "scheduler_state_dict")
                )
            )
        except Exception as exc:
            print(f"Checkpoint loading failed: {exc}", flush=True)

    if resume_state_success:
        print("Training mode: resume from checkpoint", flush=True)
    elif load_success:
        print("Training mode: pretrained weights only; optimizer/scheduler start fresh", flush=True)
    else:
        print("Training mode: from scratch", flush=True)

    train_loader, val_loader, overfit_mode = build_loaders(args)
    loss_fn = get_loss(args.loss_type)
    optimizer_params = build_optimizer_param_groups(model, args.lr, args.head_lr_mult)
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(optimizer_params, lr=args.lr)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(optimizer_params, lr=args.lr)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    start_epoch = 1
    best_val_loss = float("inf")
    global_step = 0
    history = {"train": [], "val": []}
    if resume_state_success and isinstance(checkpoint, dict):
        start_epoch = int(get_checkpoint_value(checkpoint, "epoch", default=0)) + 1
        best_val_loss = float(get_checkpoint_value(checkpoint, "best_val_loss", default=best_val_loss))
        global_step = int(get_checkpoint_value(checkpoint, "global_step", default=(start_epoch - 1) * len(train_loader)))
        history["train"] = list(get_checkpoint_value(checkpoint, "training_losses", default=[]))
        history["val"] = list(get_checkpoint_value(checkpoint, "validation_losses", default=[]))
        print(f"Resume metadata loaded: next_epoch={start_epoch} | global_step={global_step}", flush=True)
        if args.reset_optimizer:
            history = {"train": [], "val": []}
            best_val_loss = float("inf")
            print("Reset optimizer/scheduler/scaler; checkpoint weights and resume position were kept.", flush=True)

    scheduler = make_scheduler(args, optimizer, start_epoch, len(train_loader))
    print(f"LR scheduler: {args.lr_type} | init_lr={args.lr:.3e} | min_lr={args.min_lr:.3e}", flush=True)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.fp16 and device.type == "cuda"))
    if resume_state_success and isinstance(checkpoint, dict) and not args.reset_optimizer:
        optimizer_state = get_checkpoint_value(checkpoint, "optimizer_state_dict", default=None)
        scheduler_state = get_checkpoint_value(checkpoint, "scheduler", "scheduler_state_dict", default=None)
        scaler_state = get_checkpoint_value(checkpoint, "scaler", "scaler_state_dict", default=None)
        if optimizer_state is not None:
            try:
                optimizer.load_state_dict(optimizer_state)
                print("Optimizer state restored.", flush=True)
            except ValueError as exc:
                print(f"Optimizer state was not restored ({exc}); continuing fresh.", flush=True)
        if scheduler is not None and scheduler_state is not None:
            try:
                scheduler.load_state_dict(scheduler_state)
                print("Scheduler state restored.", flush=True)
            except ValueError as exc:
                print(f"Scheduler state was not restored ({exc}); continuing fresh.", flush=True)
        if scaler.is_enabled() and scaler_state is not None:
            scaler.load_state_dict(scaler_state)
            print("GradScaler state restored.", flush=True)

    limited_batch_debug_run = args.debug_max_train_batches > 0 or args.debug_max_val_batches > 0
    save_checkpoints = bool(args.save_model) and not limited_batch_debug_run
    if limited_batch_debug_run:
        print("[DEBUG] Limited batch debug run detected: checkpoint saving is disabled.", flush=True)

    t_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, global_step = train_one_epoch(
            args, model, loss_fn, train_loader, optimizer, scheduler, epoch, scaler, device, writer, global_step
        )
        history["train"].append(train_loss)
        val_details = validate(args, model, loss_fn, val_loader, epoch, device)
        history["val"].append(val_details)
        validation_objective = val_details["loss"]

        if args.debug_skip_scheduler:
            pass
        elif scheduler is not None and args.lr_type == "step":
            scheduler.step()
        elif scheduler is not None and args.lr_type == "cosine":
            scheduler.step()
        elif scheduler is not None and args.lr_type == "plateau":
            scheduler.step(validation_objective)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch} scheduler lr: {current_lr:.3e}", flush=True)
        writer.add_scalar("Lr_epoch", current_lr, epoch)
        writer.add_scalar("loss-coarse/train_loss_l1", train_loss, epoch)
        writer.add_scalar("loss-coarse-val/loss_l1", val_details["loss_l1"], epoch)
        writer.add_scalar("loss-coarse-val/loss_pcc", val_details["loss_pcc"], epoch)

        if save_checkpoints:
            if validation_objective < best_val_loss:
                best_val_loss = validation_objective
                save_checkpoint(
                    output_dir / "best_model.pt",
                    args,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    history,
                    best_val_loss,
                    global_step,
                )
                print(f"Saved new best model | best loss: {best_val_loss:.6f} | epoch: {epoch}", flush=True)
            if args.save_every > 0 and epoch % args.save_every == 0:
                save_checkpoint(
                    output_dir / "checkpoint.pt",
                    args,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    history,
                    best_val_loss,
                    global_step,
                )
                print("checkpoint saved!", flush=True)

        save_json(output_dir / "history.json", history)

    total_time = time.time() - t_start
    epochs_ran = max(1, args.epochs - start_epoch + 1)
    print(f"Total running time: {total_time} seconds", flush=True)
    print(f"Average time per epoch: {total_time / epochs_ran:.2f}s", flush=True)
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
