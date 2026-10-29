"""Evaluate and visualize one paired Au5nm sample with periodic alignment.

Reports full-field complex PSNR and complex MAE, including phase errors.
FFT correlation resolves circular translation, conjugate twin, global phase
and RMS scale, then uses the truth peak as the common PSNR/MAE scale.
This is a same-sample fitting assessment when the pair was used for training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import build_dataset, load_config, prepare_batch
from .periodic_metrics import METRIC_PROTOCOL, align_periodic_fields, complex_image_metrics
from .train import PROJECT_DIR, build_loader, load_checkpoint, make_model


def save_reconstruction_figure(prediction, target, hio, model_hio, metrics, output,
                              hio_metrics=None, model_hio_metrics=None):
    """Display the same aligned full fields used by the complex metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with torch.no_grad():
        pred, truth = align_periodic_fields(prediction, target)
        fields = [truth, pred]
        names = ["Reference", "Model"]
        if hio is not None:
            aligned_hio, _ = align_periodic_fields(hio, target)
            fields.append(aligned_hio)
            names.append("HIO")
        if model_hio is not None:
            aligned_model_hio, _ = align_periodic_fields(model_hio, target)
            fields.append(aligned_model_hio)
            names.append("Model + HIO")
        fields = [field[0, 0].detach().cpu().numpy() for field in fields]

    count = len(fields)
    phase_cmap = plt.get_cmap("twilight").copy()
    phase_cmap.set_bad("white")
    reference_amplitude = np.abs(fields[0])
    phase_support = reference_amplitude >= 0.05 * reference_amplitude.max()
    # Explicit colorbar columns keep all image panels square and equally sized.
    widths = [1] * count + [0.065, 0.24] + [1] * count + [0.065]
    fig = plt.figure(figsize=(4.5 * count + 1.3, 3.25), dpi=180)
    grid = fig.add_gridspec(1, len(widths), width_ratios=widths,
                           left=0.015, right=0.965, bottom=0.09, top=0.82,
                           wspace=0.12)
    amplitude_axes = [fig.add_subplot(grid[0, i]) for i in range(count)]
    phase_axes = [fig.add_subplot(grid[0, count + 2 + i]) for i in range(count)]
    amplitude_peak = max(float(np.abs(field).max()) for field in fields)
    for i, (name, field) in enumerate(zip(names, fields)):
        amplitude_image = amplitude_axes[i].imshow(
            np.abs(field), cmap="magma", vmin=0, vmax=amplitude_peak,
            interpolation="nearest")
        phase_image = phase_axes[i].imshow(
            np.ma.masked_where(~phase_support, np.angle(field)),
            cmap=phase_cmap, vmin=-np.pi, vmax=np.pi,
            interpolation="nearest")
        amplitude_axes[i].set_title(f"{name} · amplitude", fontsize=10, pad=7)
        phase_axes[i].set_title(f"{name} · phase", fontsize=10, pad=7)
        amplitude_axes[i].set_axis_off()
        phase_axes[i].set_axis_off()
    for index, method_metrics in ((1, metrics), (2, hio_metrics), (3, model_hio_metrics)):
        if method_metrics is not None:
            amplitude_axes[index].text(
                0.035, 0.965,
                f"PSNR  {method_metrics['complex_psnr_db']:.2f} dB\n"
                f"MAE   {method_metrics['complex_mae']:.4f}",
                transform=amplitude_axes[index].transAxes, va="top", ha="left",
                color="white", fontsize=9, linespacing=1.3,
                bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none", "pad": 3})
    fig.colorbar(amplitude_image, cax=fig.add_subplot(grid[0, count]),
                 label="Amplitude (reference peak = 1)")
    phase_bar = fig.colorbar(phase_image, cax=fig.add_subplot(grid[0, -1]),
                            label="Phase (radians)")
    phase_bar.set_ticks([-3, -2, -1, 0, 1, 2, 3])
    fig.suptitle("Au5nm reconstruction", fontsize=14, y=0.97)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="/data_ssd/oyys/high_strain_cnn/pipeline_output_2d/"
        "au5nm_from_original_1029/checkpoint_best.pt",
    )
    parser.add_argument(
        "--data-config",
        default="/data_ssd/oyys/eCDI/external_test/au5nm_ready_512/config.json",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--no-hio", action="store_true",
                        help="Plot reference and model only; skip HIO reconstruction.")
    parser.add_argument("--hio-method", choices=("fixed", "shrinkwrap"), default="fixed",
                        help="Fixed-support HIO/ER schedule or the earlier shrinkwrap comparison.")
    parser.add_argument(
        "--output",
        default=None,
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; select --device cpu if needed.")
    config = load_config(args.data_config)
    checkpoint = load_checkpoint(args.checkpoint)
    if checkpoint.get("stage") != "refiner":
        parser.error("Use a complete refiner checkpoint containing phase + refiner.")
    for key in ("diffraction_kind", "input_log_data", "center_targets"):
        if config[key] != checkpoint["config"][key]:
            parser.error(f"Checkpoint and data configuration disagree on {key}.")
    config["training"] = checkpoint["config"]["training"]
    dataset = build_dataset(config, "test")
    if len(dataset) != 1:
        parser.error(f"Expected exactly one paired sample; found {len(dataset)}.")
    raw = next(iter(build_loader(dataset, 1, 0, False, device)))
    model = make_model("refiner", config, checkpoint=checkpoint).to(device).eval()
    with torch.no_grad():
        batch = prepare_batch(raw, config, device)
        prediction = model(batch["input"], batch["diffraction"]).refined_object
        values = complex_image_metrics(prediction, batch["realspace"])
    metrics = {key: float(values[key].item())
               for key in ("complex_psnr_db", "complex_mae")}
    output = Path(args.output or PROJECT_DIR / "artifacts" / "evaluations"
                  / f"au5nm_from_original_1029_periodic_{args.hio_method}_model_hio"
                  / "metrics.json").expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Complex PSNR: {metrics['complex_psnr_db']:.6f} dB")
    print(f"Complex MAE: {metrics['complex_mae']:.8f}")
    print(f"Metric alignment: {METRIC_PROTOCOL}")
    hio = None
    model_hio = None
    hio_metrics = None
    model_hio_metrics = None
    if not args.no_hio:
        if args.hio_method == "fixed":
            from .hio_fixed import HIO_DEFAULTS, reconstruct_hio
        else:
            from .benchmark_selected100 import HIO_DEFAULTS, reconstruct_hio
        from .hio_fixed import make_model_hio_starts
        print(f"Reconstructing HIO ({args.hio_method}): {HIO_DEFAULTS}", flush=True)
        with torch.no_grad():
            hio = reconstruct_hio(batch["diffraction"], dict(HIO_DEFAULTS), seed=20261019)
            hio_values = complex_image_metrics(hio, batch["realspace"])
            hio_metrics = {key: float(hio_values[key].item())
                           for key in ("complex_psnr_db", "complex_mae")}
        print(f"HIO PSNR: {hio_metrics['complex_psnr_db']:.6f} dB")
        print(f"HIO MAE: {hio_metrics['complex_mae']:.8f}")
        model_starts = make_model_hio_starts(prediction, seed=20261020)
        with torch.no_grad():
            model_hio = reconstruct_hio(
                batch["diffraction"], dict(HIO_DEFAULTS), seed=20261019,
                initial_object=model_starts)
            model_hio_values = complex_image_metrics(model_hio, batch["realspace"])
            model_hio_metrics = {key: float(model_hio_values[key].item())
                                 for key in ("complex_psnr_db", "complex_mae")}
        print(f"Model + HIO PSNR: {model_hio_metrics['complex_psnr_db']:.6f} dB")
        print(f"Model + HIO MAE: {model_hio_metrics['complex_mae']:.8f}")
    report = {**metrics, "metric_protocol": METRIC_PROTOCOL}
    if hio_metrics is not None:
        report["hio"] = hio_metrics
        report["hio_method"] = args.hio_method
        report["hio_parameters"] = dict(HIO_DEFAULTS)
    if model_hio_metrics is not None:
        report["model_initialized_hio"] = model_hio_metrics
        report["model_initialized_hio_starts"] = 2
        report["model_initialized_hio_phase_perturbation_std_rad"] = 0.15
        report["model_initialized_hio_selection"] = "lowest measured-modulus residual"
        report["model_initialized_hio_initial_checkpoint"] = str(
            Path(args.checkpoint).expanduser().resolve())
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(f"Saved: {output}")
    figure = output.parent / "reconstruction.png"
    save_reconstruction_figure(prediction, batch["realspace"], hio, model_hio,
                               metrics, figure, hio_metrics, model_hio_metrics)
    print(f"Saved: {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
