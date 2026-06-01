import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import center_of_mass, shift
from skimage.restoration import unwrap_phase
from torch.utils.data import DataLoader, Subset

from dataset import AutoPhaseDataset
from losses import metric_dict, scale_align_sum
from train import (
    build_model,
    choose_device,
    extract_state_dict,
    load_matching_model_weights,
    optional_data_path,
    str2bool,
)


def load_checkpoint_if_present(model, checkpoint_path, device):
    if not checkpoint_path:
        print("Checkpoint loading skipped: --checkpoint not provided.", flush=True)
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    print(f"Checkpoint file loaded: {checkpoint_path}", flush=True)
    load_matching_model_weights(model, state_dict)
    return checkpoint


def shift_com(amp, phi, shape):
    coms = center_of_mass(amp)
    if np.any(np.isnan(coms)):
        return amp, phi
    deltas = tuple(int(round(dim / 2 - com)) for dim, com in zip(shape, coms))
    return shift(amp, shift=deltas, mode="wrap"), shift(phi, shift=deltas, mode="wrap")


def shift_support(support, shape):
    coms = center_of_mass(support)
    if np.any(np.isnan(coms)):
        return support
    deltas = tuple(int(round(dim / 2 - com)) for dim, com in zip(shape, coms))
    return shift(support, shift=deltas, mode="wrap")


def post_process(amp, phi, shape, threshold=0.1):
    amp = np.asarray(amp).reshape(shape)
    phi = unwrap_phase(np.asarray(phi).reshape(shape))
    mask = amp > threshold
    amp_out = mask * amp
    phi_out = mask * phi
    if np.any(amp_out > threshold):
        phi_out = phi_out - np.mean(phi_out[amp_out > threshold])
    amp_out, phi_out = shift_com(amp_out, phi_out, shape)
    mask = amp_out > threshold
    return mask * amp_out, mask * phi_out


def plot_rows(rows, names, output_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_titles = [
        "Input FT log10",
        "Pred FT log10",
        "Input - Pred FT",
        "True amp post",
        "Pred amp post",
        "Amp diff post",
        "True phase post",
        "Pred phase post",
        "Phase diff post",
        "Support shifted",
    ]
    n = len(rows)
    fig, axes = plt.subplots(10, n, figsize=(4.0 * n, 22), squeeze=False)
    for col, images in enumerate(rows):
        for row, image in enumerate(images):
            ax = axes[row, col]
            im = ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(row_titles[row], fontsize=8)
            if row == 0:
                ax.set_title(names[col], fontsize=8)
            fig.colorbar(im, ax=ax, format="%.2f", fraction=0.046, pad=0.02)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Postprocessed UMamba/AutoPhaseNN visualization.")
    parser.add_argument("--model-name", choices=["umamba", "autophasenn", "autophasenn_relu"], default="umamba")
    parser.add_argument("--phase-activation", choices=["tanh", "atan"], default="tanh")
    parser.add_argument("--phase-logit-scale", type=float, default=1.0)
    parser.add_argument("--checkpoint", default="/home/oyys/code/UMamba-AutoPhaseNN/umamba_training_pipeline/output/checkpoint.pt")
    parser.add_argument("--data-dir", default="/data_ssd/oyys/autophasenn/")
    parser.add_argument("--data-diff", default="val_diff.npy")
    parser.add_argument("--data-real", default="val_real.npy")
    parser.add_argument("--shape", type=int, default=64)
    parser.add_argument("--dtype-diff", default="float32")
    parser.add_argument("--dtype-real", default="complex64")
    parser.add_argument("--allow-missing-real", action="store_true")
    parser.add_argument("--output-png", default="./umamba_training_pipeline/output/visualization.png")
    parser.add_argument("--dataset-size", type=int, default=5000)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--slice-index", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--T", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--nconv", type=int, default=32)
    parser.add_argument("--use-down-stride", "--use_down_stride", dest="use_down_stride", type=str2bool, default=False)
    parser.add_argument("--use-up-stride", "--use_up_stride", dest="use_up_stride", type=str2bool, default=False)
    parser.add_argument("--n-blocks", "--n_blocks", dest="n_blocks", type=int, default=4)
    parser.add_argument("--unsupervise", type=str2bool, default=False)
    parser.add_argument("--scale-i", "--scale-I", dest="scale_i", type=float, default=0.0)
    parser.add_argument("--scale-align-loss", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = choose_device(args.device)
    data_dir = Path(args.data_dir)
    shape = (args.shape, args.shape, args.shape)
    dataset = AutoPhaseDataset(
        optional_data_path(data_dir, args.data_diff),
        optional_data_path(data_dir, args.data_real),
        num_samples=args.dataset_size,
        shape_diff=shape,
        shape_real=shape,
        dtype_diff=args.dtype_diff,
        dtype_real=args.dtype_real,
        scale_i=args.scale_i,
        shuffle=True,
        seed=args.seed,
        allow_missing_real=args.allow_missing_real,
    )
    sample_count = min(args.num_samples, len(dataset))
    dataset = Subset(dataset, range(sample_count))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = build_model(args, device)
    load_checkpoint_if_present(model, args.checkpoint, device)
    model.eval()

    rows = []
    names = []
    metrics = []
    z = min(max(args.slice_index, 0), args.shape - 1)
    for batch in loader:
        diff = batch["diff"].to(device).float()
        true_amp = batch["amp"].numpy()[0, 0]
        true_phi = batch["phi"].numpy()[0, 0]
        pred_diff, _obj, pred_amp, pred_phi, support = model(diff)[:5]
        pred_for_metric = scale_align_sum(diff, pred_diff) if args.scale_align_loss else pred_diff
        metrics.append({"name": batch["name"][0], **metric_dict(diff, pred_for_metric)})

        diff_np = diff.cpu().numpy()[0, 0]
        pred_diff_np = pred_diff.cpu().numpy()[0, 0]
        pred_amp_np = pred_amp.cpu().numpy()[0, 0]
        pred_phi_np = pred_phi.cpu().numpy()[0, 0]
        support_np = support.cpu().numpy()[0, 0]

        true_amp_post, true_phi_post = post_process(true_amp, true_phi, shape, threshold=args.threshold)
        pred_amp_post, pred_phi_post = post_process(
            pred_amp_np, pred_phi_np, shape, threshold=args.threshold
        )
        support_shifted = shift_support(support_np, shape)
        phase_mask = (pred_amp_post[:, :, z] > args.threshold).astype(np.float32)
        rows.append(
            [
                np.log10(diff_np[:, :, z] + 1.0),
                np.log10(pred_diff_np[:, :, z] + 1.0),
                diff_np[:, :, z] - pred_diff_np[:, :, z],
                true_amp_post[:, :, z],
                pred_amp_post[:, :, z],
                true_amp_post[:, :, z] - pred_amp_post[:, :, z],
                true_phi_post[:, :, z],
                phase_mask * pred_phi_post[:, :, z],
                true_phi_post[:, :, z] - phase_mask * pred_phi_post[:, :, z],
                support_shifted[:, :, z],
            ]
        )
        names.append(batch["name"][0])

    output_png = Path(args.output_png)
    plot_rows(rows, names, output_png)
    output_json = output_png.with_suffix(".json")
    output_json.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "dataset_size": args.dataset_size,
                "num_samples": sample_count,
                "sample_names": names,
                "per_sample": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {output_png}", flush=True)
    print("Selected samples: {}".format(", ".join(names)), flush=True)


if __name__ == "__main__":
    main()
