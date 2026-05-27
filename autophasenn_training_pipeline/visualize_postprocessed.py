import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import center_of_mass, shift
from skimage.restoration import unwrap_phase
from torch.utils.data import DataLoader

from dataset import AutoPhaseDataset
from losses import metric_dict, scale_align_sum
from model_tf_compatible import TFCompatibleAutoPhaseNN, load_weights


H = W = D = 64


def choose_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def optional_data_path(data_dir, filename):
    if filename is None or filename.lower() in {"", "none", "null"}:
        return None
    return data_dir / filename


def shift_com(amp, phi):
    coms = center_of_mass(amp)
    if np.any(np.isnan(coms)):
        return amp, phi
    deltas = (
        int(round(H / 2 - coms[0])),
        int(round(W / 2 - coms[1])),
        int(round(D / 2 - coms[2])),
    )
    return shift(amp, shift=deltas, mode="wrap"), shift(phi, shift=deltas, mode="wrap")


def shift_support(support):
    coms = center_of_mass(support)
    if np.any(np.isnan(coms)):
        return support
    deltas = (
        int(round(H / 2 - coms[0])),
        int(round(W / 2 - coms[1])),
        int(round(D / 2 - coms[2])),
    )
    return shift(support, shift=deltas, mode="wrap")


def post_process(amp, phi, threshold=0.1):
    amp = np.asarray(amp).reshape(H, W, D)
    phi = unwrap_phase(np.asarray(phi).reshape(H, W, D))
    mask = amp > threshold
    amp_out = mask * amp
    phi_out = mask * phi
    if np.any(amp_out > threshold):
        phi_out = phi_out - np.mean(phi_out[amp_out > threshold])
    amp_out, phi_out = shift_com(amp_out, phi_out)
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


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="TF test-style PyTorch visualization.")
    parser.add_argument("--checkpoint", default="/data_ssd/oyys/autophasenn/autophasenn.pth")
    parser.add_argument("--data-dir", default="/data_ssd/oyys/autophasenn/")
    parser.add_argument("--data-diff", default="val_diff.npy")
    parser.add_argument("--data-real", default="val_real.npy")
    parser.add_argument("--shape", type=int, default=64)
    parser.add_argument("--dtype-diff", default="float32")
    parser.add_argument("--dtype-real", default="complex64")
    parser.add_argument("--output-png", default="./autophasenn_training_pipeline/output/visualization.png")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--slice-index", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--scale-i", type=float, default=0.0)
    parser.add_argument("--scale-align-loss", action="store_true")
    args = parser.parse_args()

    device = choose_device(args.device)
    data_dir = Path(args.data_dir)
    shape = (args.shape, args.shape, args.shape)
    dataset = AutoPhaseDataset(
        data_dir / args.data_diff,
        optional_data_path(data_dir, args.data_real),
        args.num_samples,
        shape_diff=shape,
        shape_real=shape,
        dtype_diff=args.dtype_diff,
        dtype_real=args.dtype_real,
        scale_i=args.scale_i,
        shuffle=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = TFCompatibleAutoPhaseNN(threshold=args.threshold).to(device)
    load_weights(model, args.checkpoint, map_location=device)
    model.eval()

    rows = []
    names = []
    metrics = []
    z = args.slice_index
    for batch in loader:
        diff = batch["diff"].to(device).float()
        true_amp = batch["amp"].numpy()[0, 0]
        true_phi = batch["phi"].numpy()[0, 0]
        pred_diff, _obj, pred_amp, pred_phi, support = model(diff)
        pred_for_metric = scale_align_sum(diff, pred_diff) if args.scale_align_loss else pred_diff
        metrics.append({"name": batch["name"][0], **metric_dict(diff, pred_for_metric)})

        diff_np = diff.cpu().numpy()[0, 0]
        pred_diff_np = pred_diff.cpu().numpy()[0, 0]
        pred_amp_np = pred_amp.cpu().numpy()[0, 0]
        pred_phi_np = pred_phi.cpu().numpy()[0, 0]
        support_np = support.cpu().numpy()[0, 0]

        true_amp_post, true_phi_post = post_process(true_amp, true_phi, threshold=args.threshold)
        pred_amp_post, pred_phi_post = post_process(
            pred_amp_np, pred_phi_np, threshold=args.threshold
        )
        support_shifted = shift_support(support_np)
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
    output_json.write_text(json.dumps({"per_sample": metrics}, indent=2), encoding="utf-8")
    print(f"Saved {output_png}")


if __name__ == "__main__":
    main()
