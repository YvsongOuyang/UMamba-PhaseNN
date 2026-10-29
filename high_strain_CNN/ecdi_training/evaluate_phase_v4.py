"""Evaluate baseline and v4 phase networks, including a zero-phase baseline."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch

from pytorch_autophasenn.reconstruction import (
    realspace_from_modulus_phase,
    reciprocal_field_from_realspace,
)

from .data_v2 import build_dataset, load_config, prepare_batch
from .evaluate import aligned_objects_2d, metrics_2d
from .model_v4 import build_phase_model
from .train import build_loader, load_checkpoint, verify_data_identity


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "ecdi_2d_phase_v4.json"


def read_thickness(dataset) -> dict[int, float]:
    """Read metadata only, without loading image arrays."""
    if not hasattr(dataset, "indices") or not hasattr(dataset, "path"):
        return {}
    import h5py

    with h5py.File(dataset.path, "r") as source:
        metadata = source["metadata_json"].asstr()
        return {
            int(index): float(json.loads(metadata[index])["structure"]["target_thickness_a"])
            for index in dataset.indices
        }


@torch.no_grad()
def offaxis_metrics(prediction, target, threshold=0.3, center_width=9):
    """Band error after the same object alignment as the standard metrics.

    The central square is excluded after alignment. No further band-specific
    scale, phase or twin fitting is performed. This is a diagnostic, not an
    absolute-resolution metric.
    """
    pred, truth = aligned_objects_2d(prediction, target, threshold)
    difference = reciprocal_field_from_realspace(pred - truth, spatial_dims=2)
    spectrum = reciprocal_field_from_realspace(truth, spatial_dims=2)
    mask = torch.ones_like(spectrum.real)
    cy, cx = spectrum.shape[-2] // 2, spectrum.shape[-1] // 2
    radius = center_width // 2
    mask[..., cy - radius : cy + radius + 1, cx - radius : cx + radius + 1] = 0
    error = (difference.abs().square() * mask).flatten(1).sum(1)
    energy = (spectrum.abs().square() * mask).flatten(1).sum(1)
    total_energy = spectrum.abs().square().flatten(1).sum(1)
    return {
        "offaxis_nrmse_9x9": (error / energy.clamp_min(1e-7)).sqrt(),
        "offaxis_target_energy_fraction_9x9": energy / total_energy.clamp_min(1e-7),
    }


def summarize_records(records):
    keys = [key for key in records[0] if key not in (
        "sample", "stage", "material", "thickness_a"
    )]

    def aggregate(rows):
        if not rows:
            return {"num_samples": 0}
        return {
            "num_samples": len(rows),
            **{key: float(np.mean([row[key] for row in rows])) for key in keys},
        }

    summary = {}
    for stage in sorted({row["stage"] for row in records}):
        rows = [row for row in records if row["stage"] == stage]
        summary[stage] = {
            "all": aggregate(rows),
            "thin_le_20a": aggregate([row for row in rows if row["thickness_a"] is not None and row["thickness_a"] <= 20]),
            "thick_ge_34a": aggregate([row for row in rows if row["thickness_a"] is not None and row["thickness_a"] >= 34]),
            "by_material": {
                material: aggregate([row for row in rows if row["material"] == material])
                for material in sorted({row["material"] for row in rows})
            },
        }
    return summary


@torch.no_grad()
def evaluate_phase(model, loader, config, device, thickness, *, include_zero=False):
    model.eval()
    threshold = config["training"].get("support_threshold", 0.3)
    records = []
    for batch_index, raw in enumerate(loader, 1):
        batch = prepare_batch(raw, config, device)
        phase = model(batch["input"])
        fields = {
            "phase": realspace_from_modulus_phase(batch["diffraction"], phase, spatial_dims=2)
        }
        if include_zero:
            fields["zero_phase"] = realspace_from_modulus_phase(
                batch["diffraction"], torch.zeros_like(phase), spatial_dims=2
            )
        for stage, field in fields.items():
            values = metrics_2d(
                field, batch["realspace"], batch["diffraction"], batch["input"],
                threshold, batch.get("ideal_intensity"),
            )
            values.update(offaxis_metrics(field, batch["realspace"], threshold))
            arrays = {key: value.cpu().tolist() for key, value in values.items()}
            for position in range(field.shape[0]):
                sample = int(batch["index"][position])
                records.append({
                    "sample": sample,
                    "stage": stage,
                    "material": batch["material"][position],
                    "thickness_a": thickness.get(sample),
                    **{key: values[position] for key, values in arrays.items()},
                })
        if batch_index % 20 == 0 or batch_index == len(loader):
            LOGGER.info("VAL | batch=%d/%d", batch_index, len(loader))
    return summarize_records(records), records


def save_records(path, records):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--include-zero", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("Batch size must be positive and workers nonnegative.")
    checkpoint = load_checkpoint(args.checkpoint)
    config = load_config(args.data_config)
    for key in ("diffraction_kind", "input_log_data", "center_targets"):
        if config[key] != checkpoint["config"][key]:
            raise ValueError(f"Checkpoint and evaluation differ on {key}.")
    config["training"] = checkpoint["config"]["training"]
    if args.data_dir:
        config["root"] = str(Path(args.data_dir).expanduser().resolve())
    dataset = build_dataset(config, args.split, include_ideal=True)
    verify_data_identity(checkpoint, dataset.manifest)
    device = torch.device(args.device)
    loader = build_loader(dataset, args.batch_size, args.num_workers, False, device)
    model = build_phase_model(config, checkpoint=checkpoint).to(device).eval()
    output = Path(args.output_dir)
    if (output / "metrics.json").exists() or (output / "per_sample.csv").exists():
        raise FileExistsError("Output exists; choose a new --output-dir.")
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    summary, records = evaluate_phase(
        model, loader, config, device, read_thickness(dataset), include_zero=args.include_zero
    )
    report = {
        "experiment_version": "phase_outer_skip_v4_evaluation",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint["epoch"],
        "phase_architecture": config["training"].get("phase_architecture", "baseline"),
        "split": args.split,
        "data": dataset.manifest,
        "metrics": summary,
        "offaxis_definition": "After full-object alignment and RMS normalization, exclude the centered 9x9 Fourier pixels and divide residual energy by target energy in the remaining band.",
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    save_records(output / "per_sample.csv", records)
    for stage, metrics in summary.items():
        LOGGER.info("%s | Metrics=%s", stage, metrics)
    LOGGER.info("Output=%s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
