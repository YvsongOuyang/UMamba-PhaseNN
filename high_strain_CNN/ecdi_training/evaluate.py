"""Paired initial/refined 2D metrics, with phase, translation and twin alignment."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pytorch_autophasenn.losses import phase_retrieval_wca_loss
from pytorch_autophasenn.momamba_refiner import ambiguity_aware_component_mae
from pytorch_autophasenn.reconstruction import (
    reciprocal_field_from_realspace,
    realspace_from_modulus_phase,
)
from .data import build_dataset, center_object, load_config, prepare_batch
from .train import (
    PROJECT_DIR,
    DEFAULT_CONFIG,
    build_loader,
    load_checkpoint,
    make_model,
)


def unit_max(value):
    return value / value.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-7)


def ssim_2d(target, prediction, window=7):
    """Local 2D SSIM with unit data range, averaged per image."""

    pool = lambda value: F.avg_pool2d(value, window, stride=1)
    mu_t, mu_p = pool(target), pool(prediction)
    var_t = (pool(target.square()) - mu_t.square()).clamp_min(0)
    var_p = (pool(prediction.square()) - mu_p.square()).clamp_min(0)
    cov = pool(target * prediction) - mu_t * mu_p
    score = ((2 * mu_t * mu_p + 0.01**2) * (2 * cov + 0.03**2)) / (
        (mu_t.square() + mu_p.square() + 0.01**2) * (var_t + var_p + 0.03**2)
    )
    return score.flatten(1).mean(dim=1)


@torch.no_grad()
def aligned_objects_2d(prediction, target, threshold=0.3):
    """Center and align the RMS scale, global phase and conjugate twin."""

    pred, truth = center_object(prediction), center_object(target)
    support = unit_max(truth.abs()) >= threshold
    loss = ambiguity_aware_component_mae(pred, truth, support, spatial_dims=2)
    twin_prediction = torch.conj(
        torch.roll(torch.flip(pred, (-2, -1)), (1, 1), (-2, -1))
    )
    pred = torch.where(loss.selected_twin[:, None, None, None], twin_prediction, pred)
    correlation = (truth.conj() * pred * support).sum((-2, -1), keepdim=True)
    pred = pred * torch.exp(-1j * torch.angle(correlation))
    pred = pred / pred.abs().square().mean((-2, -1), keepdim=True).sqrt().clamp_min(
        1e-7
    )
    truth = truth / truth.abs().square().mean((-2, -1), keepdim=True).sqrt().clamp_min(
        1e-7
    )
    return pred, truth


@torch.no_grad()
def metrics_2d(
    prediction, target, measured, weights, threshold=0.3, ideal_intensity=None
):
    """Return one value per image; no target support is imposed on the FFT."""

    raw_spectrum = reciprocal_field_from_realspace(prediction, spatial_dims=2)
    pred, truth = aligned_objects_2d(prediction, target, threshold)
    amp_p, amp_t = unit_max(pred.abs()), unit_max(truth.abs())
    predicted_support = amp_p >= threshold
    true_support = amp_t >= threshold
    true_count = true_support.flatten(1).sum(1).clamp_min(1)
    pred_count = predicted_support.flatten(1).sum(1)
    intersection = (predicted_support & true_support).flatten(1).sum(1)
    union = (predicted_support | true_support).flatten(1).sum(1).clamp_min(1)
    phase_error = torch.atan2(
        torch.sin(torch.angle(pred) - torch.angle(truth)),
        torch.cos(torch.angle(pred) - torch.angle(truth)),
    )
    phase_mae = (phase_error.abs() * true_support).flatten(1).sum(1) / true_count
    phase_mse = (phase_error.square() * true_support).flatten(1).sum(1) / true_count
    amp_error = amp_p - amp_t
    ft_error = unit_max(raw_spectrum.abs()) - unit_max(measured)
    ft_l1 = ft_error.abs().flatten(1).sum(1)
    measured_sum = unit_max(measured).flatten(1).sum(1).clamp_min(1e-7)
    target_phase = torch.angle(
        reciprocal_field_from_realspace(center_object(target), spatial_dims=2)
    )
    pred_phase = torch.angle(
        reciprocal_field_from_realspace(center_object(prediction), spatial_dims=2)
    )
    amp_mse = amp_error.square().flatten(1).mean(1)
    values = {
        "complex_nrmse": (
            (pred - truth).abs().square().flatten(1).sum(1)
            / truth.abs().square().flatten(1).sum(1).clamp_min(1e-7)
        ).sqrt(),
        "object_mae": ((pred.real - truth.real).abs() + (pred.imag - truth.imag).abs())
        .flatten(1)
        .mean(1),
        "phase_wca": phase_retrieval_wca_loss(
            pred_phase, target_phase, weights, reduction="none"
        ),
        "paper_modulus_mae": ft_error.abs().flatten(1).mean(1),
        "relative_l1_modulus": ft_l1 / measured_sum,
        "voxel_mse": ft_error.square().flatten(1).mean(1),
        "voxel_rmse": ft_error.square().flatten(1).mean(1).sqrt(),
        "real_amp_l1": amp_error.abs().flatten(1).mean(1),
        "real_amp_mse": amp_mse,
        "real_amp_rmse": amp_mse.sqrt(),
        "real_amp_rel_l1": amp_error.abs().flatten(1).sum(1)
        / amp_t.flatten(1).sum(1).clamp_min(1e-7),
        "amplitude_psnr_db": -10
        * amp_mse.clamp_min(torch.finfo(amp_mse.dtype).tiny).log10(),
        "real_amp_ssim": ssim_2d(amp_t, amp_p),
        "real_phase_mae_true_support": phase_mae,
        "real_phase_rmse_true_support": phase_mse.sqrt(),
        "real_phase_amplitude_weighted_mae": (phase_error.abs() * truth.abs())
        .flatten(1)
        .sum(1)
        / truth.abs().flatten(1).sum(1).clamp_min(1e-7),
        "real_support_iou": intersection / union,
        "real_support_dice": 2 * intersection / (true_count + pred_count).clamp_min(1),
        "real_support_volume_ratio": pred_count / true_count,
    }
    if ideal_intensity is not None:
        clean_modulus = unit_max(ideal_intensity.sqrt())
        clean_error = unit_max(raw_spectrum.abs()) - clean_modulus
        values["clean_modulus_rel_l1"] = clean_error.abs().flatten(1).sum(
            1
        ) / clean_modulus.flatten(1).sum(1).clamp_min(1e-7)
        predicted_intensity = raw_spectrum.abs().square()
        predicted_intensity = predicted_intensity / predicted_intensity.sum(
            (-2, -1), keepdim=True
        ).clamp_min(1e-7)
        clean_intensity = ideal_intensity / ideal_intensity.sum(
            (-2, -1), keepdim=True
        ).clamp_min(1e-7)
        values["clean_intensity_nrmse"] = (
            (predicted_intensity - clean_intensity).square().flatten(1).sum(1)
            / clean_intensity.square().flatten(1).sum(1).clamp_min(1e-12)
        ).sqrt()
    return values


def paired_comparison(records, metrics):
    if "refined" not in metrics:
        return {}
    pairs = {(row["sample"], row["stage"]): row for row in records}
    indices = sorted({row["sample"] for row in records})
    higher_is_better = {
        "amplitude_psnr_db",
        "real_amp_ssim",
        "real_support_iou",
        "real_support_dice",
    }
    result = {}
    for metric in metrics["initial"]:
        before = np.array([pairs[(index, "initial")][metric] for index in indices])
        after = np.array([pairs[(index, "refined")][metric] for index in indices])
        improvement = after - before if metric in higher_is_better else before - after
        if metric == "real_support_volume_ratio":
            improvement = np.abs(before - 1) - np.abs(after - 1)
        result[metric] = {
            "initial": float(before.mean()),
            "refined": float(after.mean()),
            "refined_minus_initial": float((after - before).mean()),
            "median_improvement": float(np.median(improvement)),
            "fraction_improved": float(np.mean(improvement > 0)),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--support-threshold", type=float, default=0.3)
    parser.add_argument("--save-samples", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    if (
        args.batch_size < 1
        or min(args.num_workers, args.num_samples, args.save_samples) < 0
    ):
        parser.error("Batch must be positive and counts nonnegative.")
    if not 0 < args.support_threshold < 1:
        parser.error("Support threshold must be in (0,1).")
    device = torch.device(args.device)
    checkpoint = load_checkpoint(args.checkpoint)
    config = load_config(args.data_config)
    if args.data_dir:
        config["root"] = str(Path(args.data_dir).expanduser().resolve())
    for key in ("diffraction_kind", "input_log_data", "center_targets"):
        if config[key] != checkpoint["config"][key]:
            raise ValueError(f"Evaluation and checkpoint data disagree on {key}.")
    config["training"] = checkpoint["config"]["training"]
    dataset = build_dataset(config, args.split, args.num_samples, include_ideal=True)
    from .train import verify_data_identity

    verify_data_identity(checkpoint, dataset.manifest)
    loader = build_loader(dataset, args.batch_size, args.num_workers, False, device)
    model = (
        make_model(checkpoint["stage"], config, checkpoint=checkpoint).to(device).eval()
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_DIR
        / "artifacts"
        / "evaluations"
        / "ecdi_training"
        / f"{Path(args.checkpoint).parent.name}_{args.split}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s"
    )
    records, saved_samples, offset = [], [], 0
    with torch.no_grad():
        for index, raw in enumerate(loader, 1):
            batch = prepare_batch(raw, config, device)
            if checkpoint["stage"] == "phase":
                phase = model(batch["input"])
                fields = {
                    "initial": realspace_from_modulus_phase(
                        batch["diffraction"], phase, spatial_dims=2
                    )
                }
            else:
                result = model(batch["input"], batch["diffraction"])
                fields = {
                    "initial": result.initial_object,
                    "refined": result.refined_object,
                }
            for stage, field in fields.items():
                values = metrics_2d(
                    field,
                    batch["realspace"],
                    batch["diffraction"],
                    batch["input"],
                    args.support_threshold,
                    batch.get("ideal_intensity"),
                )
                arrays = {k: v.cpu().tolist() for k, v in values.items()}
                for sample in range(field.shape[0]):
                    records.append(
                        {
                            "sample": (
                                int(batch["index"][sample])
                                if "index" in batch
                                else offset + sample
                            ),
                            "stage": stage,
                            "material": (
                                batch["material"][sample]
                                if "material" in batch
                                else "unspecified"
                            ),
                            **{k: v[sample] for k, v in arrays.items()},
                        }
                    )
            for sample in range(
                min(batch["input"].shape[0], args.save_samples - len(saved_samples))
            ):
                saved_samples.append(
                    {
                        "sample": (
                            int(batch["index"][sample])
                            if "index" in batch
                            else offset + sample
                        ),
                        "material": (
                            batch["material"][sample]
                            if "material" in batch
                            else "unspecified"
                        ),
                        "target": batch["realspace"][sample, 0].cpu().numpy(),
                        "measured_modulus": batch["diffraction"][sample, 0]
                        .cpu()
                        .numpy(),
                        **{k: v[sample, 0].cpu().numpy() for k, v in fields.items()},
                        **(
                            {
                                "ideal_intensity": batch["ideal_intensity"][sample, 0]
                                .cpu()
                                .numpy()
                            }
                            if "ideal_intensity" in batch
                            else {}
                        ),
                    }
                )
            offset += batch["input"].shape[0]
            if index % 20 == 0 or index == len(loader):
                logging.info(
                    "batch=%d/%d | samples=%d/%d",
                    index,
                    len(loader),
                    offset,
                    len(dataset),
                )
    stages = sorted({row["stage"] for row in records})
    metrics = {
        stage: {
            key: float(np.mean([row[key] for row in records if row["stage"] == stage]))
            for key in records[0]
            if key not in ("sample", "stage", "material")
        }
        for stage in stages
    }
    report = {
        "schema_version": 1,
        "spatial_dims": 2,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint["epoch"],
        "num_samples": len(dataset),
        "data": dataset.manifest,
        "support_threshold": args.support_threshold,
        "metrics": metrics,
        "paired_comparison": paired_comparison(records, metrics),
        "metrics_by_material": {
            material: {
                stage: {
                    key: float(
                        np.mean(
                            [
                                row[key]
                                for row in records
                                if row["material"] == material and row["stage"] == stage
                            ]
                        )
                    )
                    for key in metrics[stage]
                }
                for stage in stages
            }
            for material in sorted({row["material"] for row in records})
        },
        "notes": [
            "Real-space metrics align translation, scale, global phase and direct/twin ambiguity.",
            "FT metrics use the complete predicted object, without ground-truth support masking.",
            "The initial object's modulus equals the measured modulus by construction.",
            "For e-CDI, thresholded support means an effective illuminated region, not a crystal boundary; the HDF5 full-field support is ignored.",
            "clean_* Fourier metrics compare against ideal_intensity; it is never a model input or training target replacement.",
        ],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (output_dir / "per_sample.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    for sample in saved_samples:
        np.savez_compressed(output_dir / f'sample_{sample["sample"]:05d}.npz', **sample)
    logging.info("Metrics=%s | output=%s", metrics, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
