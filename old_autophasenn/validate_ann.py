"""
ANN validation entry for the archived AutoPhaseNN model.

This intentionally mirrors the ANN validation path from
D:/code/PYTHON/AutoPhaseNN-main/PyTorch/train.py:
  1. model.eval()
  2. force spiking layers, if any, to mode="ann"
  3. run a single forward pass
  4. compare y[0:1] with ft_images[0:1]
  5. align predicted diffraction energy by sum(target) / sum(pred)
  6. report the same loss group printed by the source evaluator

Example:
  python old_autophasenn/validate_ann.py \
    --dataset train \
    --checkpoint /data_ssd/oyys/autophasenn/run/best_model.pt \
    --max_batches 20
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


def import_local_module(module_name, file_name):
    module_path = THIS_DIR / file_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


old_model_module = import_local_module("old_autophasenn_model", "AutoPhaseNN_model.py")
old_data_loader = import_local_module("old_autophasenn_data_loader", "data_loader.py")
Network = old_model_module.Network


class AverageMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.total += float(value) * n
        self.count += n

    @property
    def global_avg(self):
        return self.total / max(1, self.count)


class MetricLogger:
    def __init__(self):
        self.loss = AverageMeter()

    def update(self, loss):
        self.loss.update(loss)

    def synchronize_between_processes(self):
        # This standalone validator is single-process by design.
        return


def loss_log(Y_true, Y_pred):
    pred = torch.log10(Y_pred + 1.0)
    true = torch.log10(Y_true + 1.0)

    top = torch.sum(torch.pow(pred - true, 2))
    bottom = torch.sum(torch.pow(true, 2))

    loss_value = top / (bottom + 1e-8)
    return loss_value


def loss_sq(Y_true, Y_pred):
    dims = tuple(range(1, Y_true.ndim))

    top = torch.sum(torch.pow(Y_pred - Y_true, 2), dim=dims, keepdim=True)
    bottom = torch.sum(torch.pow(Y_true, 2), dim=dims, keepdim=True)

    loss_value = torch.sum(top / (bottom + 1e-8))
    return loss_value


def loss_mae(Y_true, Y_pred):
    dims = tuple(range(1, Y_true.ndim))

    top = torch.sum(torch.abs(Y_pred - Y_true), dim=dims, keepdim=True)
    bottom = torch.sum(torch.abs(Y_true), dim=dims, keepdim=True)

    loss_value = torch.sum(top / (bottom + 1e-8))
    return loss_value


def loss_paper(Y_true, Y_pred):
    sqrt_true = torch.sqrt(torch.clamp(Y_true, min=0.0))
    sqrt_pred = torch.sqrt(torch.clamp(Y_pred, min=0.0))

    abs_error = torch.abs(sqrt_pred - sqrt_true)
    total_error = torch.sum(abs_error)

    loss_value = total_error / 262144.0
    return loss_value


def loss_pcc(Y_true, Y_pred):
    dims = tuple(range(1, Y_true.ndim))

    pred_mean = torch.mean(Y_pred, dim=dims, keepdim=True)
    true_mean = torch.mean(Y_true, dim=dims, keepdim=True)

    pred_centered = Y_pred - pred_mean
    true_centered = Y_true - true_mean

    top = torch.sum(pred_centered * true_centered, dim=dims, keepdim=True)

    pred_var_sum = torch.sum(torch.pow(pred_centered, 2), dim=dims, keepdim=True)
    true_var_sum = torch.sum(torch.pow(true_centered, 2), dim=dims, keepdim=True)

    bottom = torch.sqrt(pred_var_sum * true_var_sum + 1e-8)

    loss_value = torch.sum(1.0 - (top / bottom))
    return loss_value


def loss_comb(Y_true, Y_pred):
    l1 = loss_sq(Y_true, Y_pred)
    l2 = loss_pcc(Y_true, Y_pred)
    return (l1 + l2) / 2.0


def loss_comb2(Y_true, Y_pred):
    l1 = torch.sqrt(loss_sq(Y_true, Y_pred) + 1e-8)
    l2 = loss_pcc(Y_true, Y_pred)
    return (l1 + l2) / 2.0


def loss_comb_log(Y_true, Y_pred):
    l1 = loss_sq(Y_true, Y_pred)
    l2 = loss_pcc(Y_true, Y_pred)
    l3 = loss_log(Y_true, Y_pred)

    a1, a2, a3 = 50.0, 50.0, 1.0
    return (a1 * l1 + a2 * l2 + a3 * l3) / (a1 + a2 + a3)


def clean_state_dict_keys(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]
        cleaned[key] = value
    return cleaned


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("model_state_dict", "state_dict", "model", "network"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def print_key_examples(title, values, formatter=str, max_items=8):
    if not values:
        return
    print(f"  {title}: {len(values)}")
    for value in values[:max_items]:
        print(f"    - {formatter(value)}")
    if len(values) > max_items:
        print(f"    ... {len(values) - max_items} more")


def load_matching_model_weights(model, checkpoint_path, device, strict=False):
    print("\n" + "=" * 60)
    print("Pretrained checkpoint loading")
    print("=" * 60)
    print(f"Path: {checkpoint_path}")
    print(f"Strict load: {strict}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        checkpoint_keys = list(checkpoint.keys())
        print(f"Checkpoint type: dict | top-level keys: {checkpoint_keys}")
        for meta_key in ("epoch", "global_step", "best_val_loss"):
            if meta_key in checkpoint:
                print(f"{meta_key}: {checkpoint[meta_key]}")
    else:
        print(f"Checkpoint type: {type(checkpoint).__name__}")

    state_dict = clean_state_dict_keys(extract_state_dict(checkpoint))
    print(f"State dict keys in file: {len(state_dict)}")

    if strict:
        incompatible = model.load_state_dict(state_dict, strict=True)
        print("Checkpoint loaded strictly.")
        print(f"Missing keys: {len(incompatible.missing_keys)}")
        print(f"Unexpected keys: {len(incompatible.unexpected_keys)}")
        print("=" * 60 + "\n")
        return

    model_state = model.state_dict()
    matched = {}
    skipped_shape = []
    skipped_missing = []
    for key, value in state_dict.items():
        if key not in model_state:
            skipped_missing.append(key)
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            skipped_shape.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        matched[key] = value

    incompatible = model.load_state_dict(matched, strict=False)
    model_unloaded = [key for key in model_state if key not in matched]
    print(
        "Matched keys loaded: "
        f"{len(matched)}/{len(model_state)} "
        f"({len(matched) / max(1, len(model_state)):.1%})"
    )
    print(f"Checkpoint keys not found in model: {len(skipped_missing)}")
    print(f"Checkpoint keys skipped by shape mismatch: {len(skipped_shape)}")
    print(f"Model keys left at initialization: {len(model_unloaded)}")
    print(f"load_state_dict missing_keys report: {len(incompatible.missing_keys)}")
    print(f"load_state_dict unexpected_keys report: {len(incompatible.unexpected_keys)}")

    print_key_examples("checkpoint keys not found in model", skipped_missing)
    print_key_examples(
        "shape mismatches",
        skipped_shape,
        formatter=lambda item: f"{item[0]}: checkpoint{item[1]} != model{item[2]}",
    )
    print_key_examples("model keys left at init", model_unloaded)
    print("=" * 60 + "\n")


def evaluate_ann(model, data_loader, device, max_batches=0):
    """
    ANN validation logic copied from AutoPhaseNN-main/PyTorch/train.py.

    Keep the source behavior of taking y[0:1] and ft_images[0:1]. The original
    DataLoader uses batch_size=1 during this validation path.
    """
    model.eval()

    metric_logger = MetricLogger()
    loss_totals = {
        "paper": 0.0,
        "log": 0.0,
        "sq": 0.0,
        "mae": 0.0,
        "pcc": 0.0,
        "comb": 0.0,
        "comb2": 0.0,
        "comb_log": 0.0,
    }
    evaluated_batches = 0

    for module in model.modules():
        class_name = module.__class__.__name__
        if "SpikingNeuron" in class_name and hasattr(module, "mode"):
            module.mode = "ann"

    with torch.inference_mode():
        for i, (ft_images, amps, phs) in enumerate(data_loader):
            if max_batches > 0 and i >= max_batches:
                break

            ft_images = ft_images.to(device).float()
            amps = amps.to(device).float()
            phs = phs.to(device).float()

            y, _, pred_amps, pred_phs, support = model(ft_images)

            y_pred_raw = y[0:1].float()
            y_true_raw = ft_images[0:1].float()

            scale_raw = torch.sum(y_true_raw) / (torch.sum(y_pred_raw) + 1e-10)
            y_pred_raw_scaled = y_pred_raw * scale_raw
            l_paper = loss_paper(y_true_raw, y_pred_raw_scaled)
            l_log = loss_log(y_true_raw, y_pred_raw_scaled)
            l_sq = loss_sq(y_true_raw, y_pred_raw_scaled)
            l_mae = loss_mae(y_true_raw, y_pred_raw_scaled)
            l_pcc = loss_pcc(y_true_raw, y_pred_raw_scaled)
            l_comb = loss_comb(y_true_raw, y_pred_raw_scaled)
            l_comb2 = loss_comb2(y_true_raw, y_pred_raw_scaled)
            l_comb_log = loss_comb_log(y_true_raw, y_pred_raw_scaled)

            loss_totals["paper"] += l_paper.item()
            loss_totals["log"] += l_log.item()
            loss_totals["sq"] += l_sq.item()
            loss_totals["mae"] += l_mae.item()
            loss_totals["pcc"] += l_pcc.item()
            loss_totals["comb"] += l_comb.item()
            loss_totals["comb2"] += l_comb2.item()
            loss_totals["comb_log"] += l_comb_log.item()
            evaluated_batches += 1

            metric_logger.update(loss=l_comb.item())

    metric_logger.synchronize_between_processes()
    if evaluated_batches == 0:
        raise RuntimeError("No validation batches were evaluated.")
    return {key: value / evaluated_batches for key, value in loss_totals.items()}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate old AutoPhaseNN with the ANN validation logic from AutoPhaseNN PyTorch/train.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--checkpoint",
        "--pretrained_path",
        "--pretrain_path",
        dest="checkpoint",
        type=str,
        default="",
        help="pretrained/checkpoint file to load before ANN validation",
    )
    parser.add_argument("--strict_load", action="store_true")

    parser.add_argument("--DataFolder", type=str, default="/data_ssd/oyys/autophasenn")
    parser.add_argument("--dataset", choices=("train", "val"), default="val")
    parser.add_argument("--data_train_diff", type=str, default="train_diff.npy")
    parser.add_argument("--data_train_real", type=str, default="train_real.npy")
    parser.add_argument("--data_val_diff", type=str, default="val_diff.npy")
    parser.add_argument("--data_val_real", type=str, default="val_real.npy")
    parser.add_argument("--num_samples_train", type=int, default=25000)
    parser.add_argument("--num_samples_val", type=int, default=5000)
    parser.add_argument("--dtype_diff", type=str, default="float32")
    parser.add_argument("--dtype_real", type=str, default="complex64")
    parser.add_argument("--scale_I", type=int, default=1)
    parser.add_argument("--shuffle_dataset", action="store_true")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--max_batches", type=int, default=0)

    parser.add_argument("--shape", type=int, default=64)
    parser.add_argument("--T", type=float, default=0.1)
    parser.add_argument("--nconv", type=int, default=32)
    parser.add_argument("--use_down_stride", action="store_true")
    parser.add_argument("--use_up_stride", action="store_true")
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--unsupervise", action="store_true", default=True)

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"use device: {device}")
    print(f"ANN validation dataset: {args.dataset}")
    print("************************")
    print(f"Checkpoint: {args.checkpoint if args.checkpoint else '<none>'}")
    print(f"Model module: {THIS_DIR / 'AutoPhaseNN_model.py'}")
    print(f"DataLoader module: {THIS_DIR / 'data_loader.py'}")

    model = Network(args).to(device)
    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(args.checkpoint)
        load_matching_model_weights(model, args.checkpoint, device, strict=args.strict_load)
    else:
        print(
            "\nPretrained checkpoint loading: skipped "
            "(pass --checkpoint/--pretrained_path /path/to/model.pt to load weights)\n"
        )

    data_root = Path(args.DataFolder)
    if args.dataset == "train":
        diff_path = data_root / args.data_train_diff
        real_path = data_root / args.data_train_real
        num_samples = args.num_samples_train
    else:
        diff_path = data_root / args.data_val_diff
        real_path = data_root / args.data_val_real
        num_samples = args.num_samples_val

    dataset = old_data_loader.Dataset(
        str(diff_path),
        str(real_path),
        num_samples,
        dtype_diff=args.dtype_diff,
        dtype_real=args.dtype_real,
        scale_I=args.scale_I,
        shuffle=args.shuffle_dataset,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    avg_losses = evaluate_ann(model, data_loader, device, max_batches=args.max_batches)

    print("\n" + "=" * 60)
    print("ANN validation average losses")
    print("=" * 60)
    print(f"Average Loss Paper: {avg_losses['paper']:.6f}")
    print(f"Average Loss Log:   {avg_losses['log']:.6f}")
    print(f"Average Loss sq:   {avg_losses['sq']:.6f}")
    print(f"Average Loss mae:   {avg_losses['mae']:.6f}")
    print(f"Average Loss PCC:   {avg_losses['pcc']:.6f} (Target: < 0.2)")
    print(f"Average Real PCC:   {1.0 - avg_losses['pcc']:.6f} (Target: > 0.8)")
    print(f"Average Loss Comb:  {avg_losses['comb']:.6f} (Target: Small Number)")
    print(f"Average Loss Comb2: {avg_losses['comb2']:.6f}")
    print(f"Average Loss CombLog: {avg_losses['comb_log']:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
