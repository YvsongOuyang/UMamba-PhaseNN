import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import AutoPhaseDataset
from losses import metric_dict, scale_align_sum
from model_tf_compatible import TFCompatibleAutoPhaseNN, load_weights


def choose_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def add_metrics(total, metrics):
    for key, value in metrics.items():
        total[key] = total.get(key, 0.0) + float(value)


def optional_data_path(data_dir, filename):
    if filename is None or filename.lower() in {"", "none", "null"}:
        return None
    return data_dir / filename


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Evaluate an AutoPhaseNN PyTorch checkpoint.")
    parser.add_argument("--checkpoint", default="/data_ssd/oyys/autophasenn/autophasenn.pth")
    parser.add_argument("--data-dir", default="/data_ssd/oyys/autophasenn/")
    parser.add_argument("--data-diff", default="val_diff.npy")
    parser.add_argument("--data-real", default="val_real.npy")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--shape", type=int, default=64)
    parser.add_argument("--dtype-diff", default="float32")
    parser.add_argument("--dtype-real", default="complex64")
    parser.add_argument("--output-json", default="./output/evaluation_results.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--scale-i", type=float, default=0.0)
    parser.add_argument("--scale-align-loss", action="store_true")
    args = parser.parse_args()

    device = choose_device(args.device)
    data_dir = Path(args.data_dir)
    shape = (args.shape, args.shape, args.shape)
    num_samples = args.num_samples
    if args.limit and args.limit > 0:
        num_samples = min(num_samples, args.limit)
    dataset = AutoPhaseDataset(
        data_dir / args.data_diff,
        optional_data_path(data_dir, args.data_real),
        num_samples,
        shape_diff=shape,
        shape_real=shape,
        dtype_diff=args.dtype_diff,
        dtype_real=args.dtype_real,
        scale_i=args.scale_i,
        shuffle=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = TFCompatibleAutoPhaseNN(threshold=args.threshold).to(device)
    checkpoint = load_weights(model, args.checkpoint, map_location=device)
    epoch = checkpoint.get("epoch", None) if isinstance(checkpoint, dict) else None
    model.eval()

    total = {}
    per_sample = []
    for batch in tqdm(loader, desc="eval"):
        diff = batch["diff"].to(device).float()
        pred = model(diff)[0]
        if args.scale_align_loss:
            pred = scale_align_sum(diff, pred)

        for i, name in enumerate(batch["name"]):
            metrics = metric_dict(diff[i : i + 1], pred[i : i + 1])
            per_sample.append({"name": name, **metrics})
            add_metrics(total, metrics)

    n = max(len(per_sample), 1)
    report = {
        "checkpoint": str(args.checkpoint),
        "epoch": epoch,
        "num_samples": len(per_sample),
        "scale_align_loss": args.scale_align_loss,
        "sum": total,
        "mean": {key: value / n for key, value in total.items()},
        "per_sample": per_sample,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["mean"], indent=2))


if __name__ == "__main__":
    main()

