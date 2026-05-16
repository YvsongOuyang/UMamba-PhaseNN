import argparse
import csv
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from AutoPhaseNN_model import Network
from UMambaEnc_3d import get_umamba_enc_3d_from_plans
from data_loader import Dataset


def loss_sq(y_true, y_pred):
    dims = tuple(range(1, y_true.ndim))
    top = torch.sum(torch.pow(y_pred - y_true, 2), dim=dims, keepdim=True)
    bottom = torch.sum(torch.pow(y_true, 2), dim=dims, keepdim=True)
    return torch.mean(top / (bottom + 1e-6))


def loss_mae(y_true, y_pred):
    dims = tuple(range(1, y_true.ndim))
    top = torch.sum(torch.abs(y_pred - y_true), dim=dims, keepdim=True)
    bottom = torch.sum(torch.abs(y_true), dim=dims, keepdim=True)
    return torch.sum(top / (bottom + 1e-8))


def loss_paper(y_true, y_pred):
    sqrt_true = torch.sqrt(torch.clamp(y_true, min=0.0))
    sqrt_pred = torch.sqrt(torch.clamp(y_pred, min=0.0))
    abs_error = torch.abs(sqrt_pred - sqrt_true)
    return torch.sum(abs_error) / 262144.0


def loss_pcc_old(y_true, y_pred):
    dims = tuple(range(1, y_true.ndim))
    pred_mean = torch.mean(y_pred, dim=dims, keepdim=True)
    true_mean = torch.mean(y_true, dim=dims, keepdim=True)
    pred_centered = y_pred - pred_mean
    true_centered = y_true - true_mean
    top = torch.sum(pred_centered * true_centered, dim=dims, keepdim=True)
    pred_var_sum = torch.sum(torch.pow(pred_centered, 2), dim=dims, keepdim=True)
    true_var_sum = torch.sum(torch.pow(true_centered, 2), dim=dims, keepdim=True)
    bottom = torch.sqrt(pred_var_sum * true_var_sum + 1e-8)
    return torch.mean(1.0 - (top / bottom))


def loss_pcc(y_true, y_pred):
    dims = tuple(range(1, y_true.ndim))
    pred_mean = torch.mean(y_pred, dim=dims, keepdim=True)
    true_mean = torch.mean(y_true, dim=dims, keepdim=True)
    pred_centered = y_pred - pred_mean
    true_centered = y_true - true_mean
    top = torch.sum(pred_centered * true_centered, dim=dims, keepdim=True)
    pred_var_sum = torch.sum(pred_centered**2, dim=dims, keepdim=True)
    true_var_sum = torch.sum(true_centered**2, dim=dims, keepdim=True)
    bottom = torch.sqrt(pred_var_sum + 1e-8) * torch.sqrt(true_var_sum + 1e-8)
    pcc = torch.clamp(top / bottom, -1.0 + 1e-6, 1.0 - 1e-6)
    return torch.mean(1.0 - pcc)


def loss_comb(y_true, y_pred):
    return (loss_sq(y_true, y_pred) + loss_pcc_old(y_true, y_pred)) / 2.0


def loss_comb2(y_true, y_pred):
    return (torch.sqrt(loss_sq(y_true, y_pred) + 1e-8) + loss_pcc(y_true, y_pred)) / 2.0


def compute_losses(ft_images, pred_ft):
    sq = loss_sq(ft_images, pred_ft)
    return {
        "loss_l1": nn.L1Loss()(pred_ft, ft_images),
        "loss_mae": loss_mae(ft_images, pred_ft),
        "loss_sq": sq,
        "loss_mse": sq,
        "loss_paper": loss_paper(ft_images, pred_ft),
        "loss_huber": nn.SmoothL1Loss()(pred_ft, ft_images),
        "loss_pcc": loss_pcc(ft_images, pred_ft),
        "loss_comb": loss_comb(ft_images, pred_ft),
        "loss_comb2": loss_comb2(ft_images, pred_ft),
    }


def clean_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "model.", "net."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    return cleaned


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    missing, unexpected = model.load_state_dict(clean_state_dict(state_dict), strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    if missing:
        print("First missing keys:", missing[:10])
    if unexpected:
        print("First unexpected keys:", unexpected[:10])


def normalize_slice(slice_2d):
    array = np.asarray(slice_2d, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array)
    vmin = np.percentile(array[finite], 1)
    vmax = np.percentile(array[finite], 99)
    if vmax <= vmin:
        vmax = float(array[finite].max())
        vmin = float(array[finite].min())
    if vmax <= vmin:
        return np.zeros_like(array)
    return np.clip((array - vmin) / (vmax - vmin), 0.0, 1.0)


def aligned_slice_index(volume, input_size, slice_index=None, slice_fraction=0.5):
    volume = np.asarray(volume, dtype=np.float32)
    depth = volume.shape[0]
    if slice_index is not None:
        if input_size <= 1:
            fraction = 0.5
        else:
            fraction = float(slice_index) / float(input_size - 1)
    else:
        fraction = float(slice_fraction)
    fraction = min(max(fraction, 0.0), 1.0)
    return int(round(fraction * (depth - 1)))


def pick_aligned_slice(volume, input_size, slice_index=None, slice_fraction=0.5, normalize=True):
    volume = np.asarray(volume, dtype=np.float32)
    layer_index = aligned_slice_index(
        volume,
        input_size=input_size,
        slice_index=slice_index,
        slice_fraction=slice_fraction,
    )
    image = volume[layer_index]
    if normalize:
        image = normalize_slice(image)
    return layer_index, image


def save_slice_heatmap(volume, out_path, title, input_size, slice_index=None, slice_fraction=0.5):
    layer_index, image = pick_aligned_slice(
        volume,
        input_size=input_size,
        slice_index=slice_index,
        slice_fraction=slice_fraction,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4), dpi=160)
    plt.imshow(image, cmap="inferno")
    plt.title(f"{title} | z={layer_index}")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def volume_to_numpy(tensor):
    array = tensor.detach().float().cpu().numpy()
    return np.squeeze(array)


def save_notebook_style_slice_plots(records, out_dir, input_size, slice_index=None, slice_fraction=0.5):
    if not records:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    groups = [
        (
            "ft_comparison.png",
            [
                ("ft_true", "FT input"),
                ("ft_pred", "Pred FT"),
                ("ft_diff", "Diff FT"),
            ],
        ),
        (
            "amp_comparison.png",
            [
                ("support", "Support"),
                ("amp_true", "True Amp"),
                ("amp_pred", "Predicted Amp"),
            ],
        ),
        (
            "phase_comparison.png",
            [
                ("phase_true", "True ph"),
                ("phase_pred_masked", "Pred Ph"),
                ("phase_diff", "Diff Ph"),
            ],
        ),
    ]

    saved_paths = []
    for filename, columns in groups:
        ncols = len(records) * len(columns)
        fig, axes = plt.subplots(
            1,
            ncols,
            figsize=(3.2 * ncols, 3.4),
            constrained_layout=True,
            squeeze=False,
        )
        for record_index, record in enumerate(records):
            for col_index, (key, label) in enumerate(columns):
                axis = axes[0, record_index * len(columns) + col_index]
                layer_index, image = pick_aligned_slice(
                    record[key],
                    input_size=input_size,
                    slice_index=slice_index,
                    slice_fraction=slice_fraction,
                    normalize=False,
                )
                im = axis.imshow(image)
                axis.set_title(f"{label}, {record['sample_index']}\nz={layer_index}", fontsize=10)
                axis.axis("off")
                plt.colorbar(im, ax=axis, fraction=0.046, pad=0.04)

        path = out_dir / filename
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(str(path))
    return saved_paths


class LeakyReluRecorder:
    def __init__(self, model, input_size, slice_index=None, slice_fraction=0.5):
        self.model = model
        self.input_size = input_size
        self.slice_index = slice_index
        self.slice_fraction = slice_fraction
        self.names = OrderedDict()
        self.current = OrderedDict()
        self.sum_volumes = OrderedDict()
        self.counts = OrderedDict()
        self.handles = []

    def _hook(self, name):
        def fn(_module, _inputs, output):
            if not torch.is_tensor(output) or output.ndim < 5:
                return
            with torch.no_grad():
                vol = output.detach().float().abs().mean(dim=(0, 1)).cpu()
            if name not in self.sum_volumes:
                self.sum_volumes[name] = torch.zeros_like(vol)
                self.counts[name] = 0
            self.sum_volumes[name] += vol
            self.counts[name] += 1
            self.current[name] = vol.numpy()

        return fn

    def register(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.LeakyReLU):
                safe_name = name.replace(".", "_")
                self.names[name] = safe_name
                self.handles.append(module.register_forward_hook(self._hook(name)))
        if not self.names:
            raise RuntimeError("No nn.LeakyReLU layers found in the model.")
        print(f"Registered {len(self.names)} LeakyReLU hooks.")

    def clear_current(self):
        self.current = OrderedDict()

    def save_current(self, out_dir, batch_index):
        for name, volume in self.current.items():
            safe_name = self.names[name]
            out_path = out_dir / f"batch_{batch_index:04d}_{safe_name}.png"
            save_slice_heatmap(
                volume,
                out_path,
                f"{name} batch {batch_index}",
                input_size=self.input_size,
                slice_index=self.slice_index,
                slice_fraction=self.slice_fraction,
            )

    def save_aggregate(self, out_dir):
        aggregate_dir = out_dir / "aggregate"
        summary = {}
        for name, sum_volume in self.sum_volumes.items():
            count = max(self.counts[name], 1)
            volume = (sum_volume / count).numpy()
            safe_name = self.names[name]
            npy_path = aggregate_dir / f"{safe_name}.npy"
            png_path = aggregate_dir / f"{safe_name}.png"
            aggregate_dir.mkdir(parents=True, exist_ok=True)
            np.save(npy_path, volume)
            save_slice_heatmap(
                volume,
                png_path,
                f"{name} aggregate",
                input_size=self.input_size,
                slice_index=self.slice_index,
                slice_fraction=self.slice_fraction,
            )
            summary[name] = {
                "count": int(count),
                "shape": list(volume.shape),
                "png": str(png_path),
                "npy": str(npy_path),
            }
        return summary

    def close(self):
        for handle in self.handles:
            handle.remove()


def build_args():
    parser = argparse.ArgumentParser(
        description="Run AutoPhaseNN validation and save LeakyReLU middle-slice heatmaps."
    )
    parser.add_argument("--checkpoint", type=str, default="best_model.pt")
    parser.add_argument("--data_folder", type=str, default="CDI_simulation_upsamp_noise")
    parser.add_argument("--data_val_diff", type=str, default="val_diff.npy")
    parser.add_argument("--data_val_real", type=str, default="val_real.npy")
    parser.add_argument("--num_samples_val", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=str(SCRIPT_DIR / "autophasenn_heatmap_results"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--max_batches", type=int, default=0, help="0 means run the full validation set.")
    parser.add_argument("--save_batch_heatmaps", type=int, default=3)
    parser.add_argument(
        "--save_recon_plots",
        type=int,
        default=2,
        help="Number of validation samples to save with notebook-style FT/Amp/Phase slice plots.",
    )
    parser.add_argument(
        "--slice_index",
        type=int,
        default=None,
        help="Input-space z index to visualize. If omitted, --slice_fraction is used.",
    )
    parser.add_argument(
        "--slice_fraction",
        type=float,
        default=0.5,
        help="Relative z position to visualize when --slice_index is omitted.",
    )
    parser.add_argument("--shape", type=int, default=64)
    parser.add_argument("--T", type=float, default=0.1)
    parser.add_argument("--nconv", type=int, default=32)
    parser.add_argument("--use_down_stride", action="store_true", default=False)
    parser.add_argument("--use_up_stride", action="store_true", default=False)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--unsupervise", type=bool, default=True)
    parser.add_argument("--scale_I", type=int, default=1)
    return parser.parse_args()


def main():
    args = build_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Pass --checkpoint to point at the pretrained file."
        )

    plans = {
        "dataset_name": "Diffraction3D",
        "original_median_spacing_after_transp": [1.0, 1.0, 1.0],
        "original_median_shape_after_transp": [64, 64, 64],
        "image_reader_writer": "SimpleITKIO",
        "transpose_forward": [0, 1, 2],
        "transpose_backward": [0, 1, 2],

        "configurations": {
            "3d_fullres": {
                "data_identifier": "nnUNetPlans_3d_fullres",
                "preprocessor_name": "DefaultPreprocessor",
                "batch_size": args.batch_size,
                "patch_size": [64, 64, 64],
                "median_image_size_in_voxels": [64.0, 64.0, 64.0],
                "spacing": [1.0, 1.0, 1.0],
                "normalization_schemes": ["ZScoreNormalization"],
                "use_mask_for_norm": [False],
                "UNet_class_name": "PlainConvUNet",
                "UNet_base_num_features": 32,
                "n_conv_per_stage_encoder": [2, 2, 2, 2], 
                "n_conv_per_stage_decoder": [2, 2, 2],
                "num_pool_per_axis": [4, 4, 4],
                "pool_op_kernel_sizes": [[1, 1, 1],[2, 2, 2],[2, 2, 2],[2, 2, 2]],
                "conv_kernel_sizes": [[3, 3, 3],[3, 3, 3],[3, 3, 3],[3, 3, 3]],
                "unet_max_num_features": 320,
                "resampling_fn_data": "fast_resample_data_or_seg_to_shape",
                "resampling_fn_seg": "fast_resample_data_or_seg_to_shape",
                "resampling_fn_data_kwargs": {
                    "is_seg": False,
                    "order": 3,
                    "order_z": 3,
                    "force_separate_z": None
                },
                "resampling_fn_seg_kwargs": {
                    "is_seg": True,
                    "order": 1,
                    "order_z": 1,
                    "force_separate_z": None
                },
                "resampling_fn_probabilities": "fast_resample_data_or_seg_to_shape",
                "resampling_fn_probabilities_kwargs": {
                    "is_seg": False,
                    "order": 1,
                    "order_z": 1,
                    "force_separate_z": None
                },
                "batch_dice": True,
            },
            "3d_cascade_fullres": {
                "inherits_from": "3d_fullres",
                "previous_stage": "3d_lowres"
            }
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
                "std": 137.8484649658203
            }
        }
    }
    # 模拟dataset.json（标签配置）
    dataset_json = {"labels": {"background":0,}, "num_segmentation_heads":1}
    
    # 初始化PlansManager/ConfigurationManager（函数必需入参）
    plans_manager = PlansManager(plans)
    config_manager = plans_manager.get_configuration("3d_fullres")

    model = get_umamba_enc_3d_from_plans(
            plans_manager=plans_manager,
            dataset_json=dataset_json,
            configuration_manager=config_manager,
            num_input_channels=1,  # 单模态输入（如CT）
            deep_supervision=False
        ).to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    val_diff = os.path.join(args.data_folder, args.data_val_diff)
    val_real = os.path.join(args.data_folder, args.data_val_real)
    for data_file in (val_diff, val_real):
        if not os.path.exists(data_file):
            raise FileNotFoundError(
                f"Validation data file not found: {data_file}. "
                "Pass --data_folder/--data_val_diff/--data_val_real to match your dataset."
            )

    dataset = Dataset(
        val_diff,
        val_real,
        args.num_samples_val,
        shape_diff=(args.shape, args.shape, args.shape),
        shape_real=(args.shape, args.shape, args.shape),
        dtype_diff="float32",
        dtype_real="complex64",
        scale_I=args.scale_I,
        shuffle=False,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )

    recorder = LeakyReluRecorder(
        model,
        input_size=args.shape,
        slice_index=args.slice_index,
        slice_fraction=args.slice_fraction,
    )
    recorder.register()

    details_total = {
        "loss_l1": 0.0,
        "loss_mae": 0.0,
        "loss_sq": 0.0,
        "loss_mse": 0.0,
        "loss_paper": 0.0,
        "loss_huber": 0.0,
        "loss_pcc": 0.0,
        "loss_comb": 0.0,
        "loss_comb2": 0.0,
    }
    batch_rows = []
    recon_records = []
    total_batches = len(loader) if args.max_batches <= 0 else min(len(loader), args.max_batches)

    with torch.no_grad():
        progress = tqdm(enumerate(loader), total=total_batches, desc="Validation")
        for batch_index, (ft_images, amps, phs) in progress:
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            ft_images = ft_images.to(device=device, dtype=torch.float32, non_blocking=True)
            amps = amps.to(device=device, dtype=torch.float32, non_blocking=True)
            phs = phs.to(device=device, dtype=torch.float32, non_blocking=True)
            recorder.clear_current()

            with torch.cuda.amp.autocast(enabled=args.fp16 and device.type == "cuda"):
                pred_ft, obj, pred_amps, pred_phs, support = model(ft_images)
                scale_raw = torch.sum(ft_images) / (torch.sum(pred_ft) + 1e-10)
                pred_ft = pred_ft * scale_raw
                details = compute_losses(ft_images, pred_ft)

            if len(recon_records) < args.save_recon_plots:
                slots_left = args.save_recon_plots - len(recon_records)
                samples_this_batch = min(ft_images.shape[0], slots_left)
                for sample_offset in range(samples_this_batch):
                    global_index = batch_index * args.batch_size + sample_offset
                    masked_phase = pred_phs[sample_offset] * support[sample_offset]
                    recon_records.append(
                        {
                            "sample_index": int(global_index),
                            "ft_true": volume_to_numpy(ft_images[sample_offset]),
                            "ft_pred": volume_to_numpy(pred_ft[sample_offset]),
                            "ft_diff": volume_to_numpy(ft_images[sample_offset] - pred_ft[sample_offset]),
                            "support": volume_to_numpy(support[sample_offset]),
                            "amp_true": volume_to_numpy(amps[sample_offset]),
                            "amp_pred": volume_to_numpy(pred_amps[sample_offset]),
                            "phase_true": volume_to_numpy(phs[sample_offset]),
                            "phase_pred_masked": volume_to_numpy(masked_phase),
                            "phase_diff": volume_to_numpy(phs[sample_offset] - masked_phase),
                        }
                    )

            if batch_index < args.save_batch_heatmaps:
                recorder.save_current(output_dir / "per_batch", batch_index)

            row = {"batch": batch_index}
            for key, value in details.items():
                scalar = float(value.detach().cpu().item())
                details_total[key] += scalar
                row[key] = scalar
            batch_rows.append(row)
            progress.set_postfix({key: f"{row[key]:.3e}" for key in ("loss_l1", "loss_comb2")})

    for key in details_total:
        details_total[key] /= max(total_batches, 1)

    heatmap_summary = recorder.save_aggregate(output_dir)
    recon_plot_paths = save_notebook_style_slice_plots(
        recon_records,
        output_dir / "reconstruction_slices",
        input_size=args.shape,
        slice_index=args.slice_index,
        slice_fraction=args.slice_fraction,
    )
    recorder.close()

    results = {
        "checkpoint": args.checkpoint,
        "data_val_diff": val_diff,
        "data_val_real": val_real,
        "num_batches": total_batches,
        "batch_size": args.batch_size,
        "losses": details_total,
        "heatmaps": heatmap_summary,
        "reconstruction_slice_plots": recon_plot_paths,
    }

    json_path = output_dir / "validation_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    csv_path = output_dir / "validation_batches.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["batch"] + list(details_total.keys()))
        writer.writeheader()
        writer.writerows(batch_rows)

    print("\nAverage validation losses:")
    for key, value in details_total.items():
        print(f"  {key}: {value:.6e}")
    print(f"\nSaved summary: {json_path}")
    print(f"Saved batch losses: {csv_path}")
    print(f"Saved heatmaps under: {output_dir}")
    if recon_plot_paths:
        print("Saved notebook-style reconstruction slice plots:")
        for path in recon_plot_paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
