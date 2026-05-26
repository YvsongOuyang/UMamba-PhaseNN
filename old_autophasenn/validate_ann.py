"""
ANN validation entry for the archived AutoPhaseNN model.

This intentionally mirrors the ANN validation path from
D:/code/PYTHON/AutoPhaseNN-main/PyTorch/train.py:
  1. model.eval()
  2. force spiking layers, if any, to mode="ann"
  3. run a single forward pass
  4. compare every sample in y with the matching ft_images sample
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


def compute_loss_values(y_true_raw, y_pred_raw_scaled):
    return {
        "paper": loss_paper(y_true_raw, y_pred_raw_scaled),
        "log": loss_log(y_true_raw, y_pred_raw_scaled),
        "sq": loss_sq(y_true_raw, y_pred_raw_scaled),
        "mae": loss_mae(y_true_raw, y_pred_raw_scaled),
        "pcc": loss_pcc(y_true_raw, y_pred_raw_scaled),
        "comb": loss_comb(y_true_raw, y_pred_raw_scaled),
        "comb2": loss_comb2(y_true_raw, y_pred_raw_scaled),
        "comb_log": loss_comb_log(y_true_raw, y_pred_raw_scaled),
    }


def print_sample_losses(sample_index, batch_index, batch_sample_index, losses):
    pcc_loss = losses["pcc"].item()
    print("\n" + "-" * 60)
    print(
        "Good ANN validation sample "
        f"| sample={sample_index} | batch={batch_index} | batch_sample={batch_sample_index}"
    )
    print(f"  Loss Paper: {losses['paper'].item():.6f}")
    print(f"  Loss Log:   {losses['log'].item():.6f}")
    print(f"  Loss sq:   {losses['sq'].item():.6f}")
    print(f"  Loss mae:   {losses['mae'].item():.6f}")
    print(f"  Loss PCC:   {pcc_loss:.6f} (Target: < 0.2)")
    print(f"  Real PCC:   {1.0 - pcc_loss:.6f} (Target: > 0.8)")
    print(f"  Loss Comb:  {losses['comb'].item():.6f} (Target: Small Number)")
    print(f"  Loss Comb2: {losses['comb2'].item():.6f}")
    print(f"  Loss CombLog: {losses['comb_log'].item():.6f}")
    print("-" * 60 + "\n")


def summarize_values(values):
    ordered = sorted(values)
    if not ordered:
        return {}

    def percentile(p):
        index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
        return ordered[index]

    return {
        "min": ordered[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


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


def evaluate_ann(
    model,
    data_loader,
    device,
    max_batches=0,
    good_pcc_loss_threshold=0.2,
    good_real_pcc_threshold=0.8,
    print_good_samples=True,
):
    """
    ANN validation logic copied from AutoPhaseNN-main/PyTorch/train.py.

    Evaluate every sample in every batch. This preserves the source behavior
    when batch_size=1, while avoiding silent sample drops for larger batches.
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
    loss_series = {key: [] for key in loss_totals}
    legacy_first_sample_totals = {key: 0.0 for key in loss_totals}
    legacy_first_sample_count = 0
    evaluated_batches = 0
    evaluated_samples = 0
    good_sample_count = 0

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

            evaluated_batches += 1

            for batch_sample_idx in range(ft_images.shape[0]):
                y_pred_raw = y[batch_sample_idx:batch_sample_idx + 1].float()
                y_true_raw = ft_images[batch_sample_idx:batch_sample_idx + 1].float()

                scale_raw = torch.sum(y_true_raw) / (torch.sum(y_pred_raw) + 1e-10)
                y_pred_raw_scaled = y_pred_raw * scale_raw
                losses = compute_loss_values(y_true_raw, y_pred_raw_scaled)

                for key, value in losses.items():
                    loss_value = value.item()
                    loss_totals[key] += loss_value
                    loss_series[key].append(loss_value)
                    if batch_sample_idx == 0:
                        legacy_first_sample_totals[key] += loss_value

                if batch_sample_idx == 0:
                    legacy_first_sample_count += 1

                evaluated_samples += 1
                metric_logger.update(loss=losses["comb"].item())

                pcc_loss = losses["pcc"].item()
                real_pcc = 1.0 - pcc_loss
                if (
                    pcc_loss < good_pcc_loss_threshold
                    and real_pcc > good_real_pcc_threshold
                ):
                    good_sample_count += 1
                    if print_good_samples:
                        print_sample_losses(
                            evaluated_samples,
                            i + 1,
                            batch_sample_idx + 1,
                            losses,
                        )

    metric_logger.synchronize_between_processes()
    if evaluated_samples == 0:
        raise RuntimeError("No validation samples were evaluated.")
    avg_losses = {key: value / evaluated_samples for key, value in loss_totals.items()}
    if legacy_first_sample_count > 0:
        avg_losses["_legacy_first_sample_avgs"] = {
            key: value / legacy_first_sample_count
            for key, value in legacy_first_sample_totals.items()
        }
        avg_losses["_legacy_first_sample_count"] = legacy_first_sample_count
    avg_losses["_loss_stats"] = {
        key: summarize_values(values)
        for key, values in loss_series.items()
    }
    avg_losses["_num_batches"] = evaluated_batches
    avg_losses["_num_samples"] = evaluated_samples
    avg_losses["_good_sample_count"] = good_sample_count
    return avg_losses


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
    parser.add_argument(
        "--good_pcc_loss_threshold",
        type=float,
        default=0.2,
        help="print a sample when its Loss PCC is lower than this value",
    )
    parser.add_argument(
        "--good_real_pcc_threshold",
        type=float,
        default=0.8,
        help="print a sample when its Real PCC is higher than this value",
    )
    parser.add_argument(
        "--disable_good_sample_print",
        action="store_true",
        help="disable per-sample loss printing for samples that satisfy the PCC targets",
    )

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
    print(f"Model module: {THIS_DIR / 'AutoPhaseNN_model.py'}")
    print(f"DataLoader module: {THIS_DIR / 'data_loader.py'}")

    model = Network(args).to(device)
    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(args.checkpoint)
        load_matching_model_weights(model, args.checkpoint, device, strict=args.strict_load)
    else:
        print("\nPretrained checkpoint loading: skipped (--checkpoint/--pretrained_path not provided)\n")

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

    avg_losses = evaluate_ann(
        model,
        data_loader,
        device,
        max_batches=args.max_batches,
        good_pcc_loss_threshold=args.good_pcc_loss_threshold,
        good_real_pcc_threshold=args.good_real_pcc_threshold,
        print_good_samples=not args.disable_good_sample_print,
    )

    print("\n" + "=" * 60)
    print("ANN validation average losses")
    print("=" * 60)
    print(f"Evaluated batches: {avg_losses['_num_batches']}")
    print(f"Evaluated samples: {avg_losses['_num_samples']}")
    print(f"Good samples meeting PCC targets: {avg_losses['_good_sample_count']}")
    print(f"Average Loss Paper: {avg_losses['paper']:.6f}")
    print(f"Average Loss Log:   {avg_losses['log']:.6f}")
    print(f"Average Loss sq:   {avg_losses['sq']:.6f}")
    print(f"Average Loss mae:   {avg_losses['mae']:.6f}")
    print(f"Average Loss PCC:   {avg_losses['pcc']:.6f} (Target: < 0.2)")
    print(f"Average Real PCC:   {1.0 - avg_losses['pcc']:.6f} (Target: > 0.8)")
    print(f"Average Loss Comb:  {avg_losses['comb']:.6f} (Target: Small Number)")
    print(f"Average Loss Comb2: {avg_losses['comb2']:.6f}")
    print(f"Average Loss CombLog: {avg_losses['comb_log']:.6f}")
    if "_legacy_first_sample_avgs" in avg_losses:
        legacy = avg_losses["_legacy_first_sample_avgs"]
        print("\nLegacy first-sample-per-batch averages")
        print(f"Legacy samples: {avg_losses['_legacy_first_sample_count']}")
        print(f"Legacy Loss Paper: {legacy['paper']:.6f}")
        print(f"Legacy Loss Log:   {legacy['log']:.6f}")
        print(f"Legacy Loss sq:   {legacy['sq']:.6f}")
        print(f"Legacy Loss mae:   {legacy['mae']:.6f}")
        print(f"Legacy Loss PCC:   {legacy['pcc']:.6f}")
        print(f"Legacy Real PCC:   {1.0 - legacy['pcc']:.6f}")
        print(f"Legacy Loss Comb:  {legacy['comb']:.6f}")
        print(f"Legacy Loss Comb2: {legacy['comb2']:.6f}")
        print(f"Legacy Loss CombLog: {legacy['comb_log']:.6f}")

    print("\nLoss distribution summary")
    for key in ("paper", "sq", "mae", "pcc", "comb"):
        stats = avg_losses["_loss_stats"][key]
        print(
            f"{key}: min={stats['min']:.6f} | p50={stats['p50']:.6f} | "
            f"p90={stats['p90']:.6f} | p99={stats['p99']:.6f} | max={stats['max']:.6f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
