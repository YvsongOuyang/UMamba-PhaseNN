"""Check whether blending or Fourier-modulus projection helps a trained refiner.

This is an evaluation-only diagnostic. It never changes model weights. Choose a
blend on the validation split; use the test split only for the chosen variant.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch

from pytorch_autophasenn.reconstruction import project_to_measured_modulus

from .data_v2 import build_dataset, load_config, prepare_batch
from .evaluate import metrics_2d
from .train import build_loader, load_checkpoint, make_model, verify_data_identity


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "ecdi_2d_v2.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--support-threshold", type=float, default=0.3)
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("Batch size must be positive and workers nonnegative.")
    if not 0 < args.support_threshold < 1:
        parser.error("Support threshold must be between 0 and 1.")
    try:
        args.alpha_values = [float(value) for value in args.alphas.split(",")]
    except ValueError:
        parser.error("--alphas must be a comma-separated list of numbers.")
    if not args.alpha_values or any(
        not np.isfinite(value) or not 0 <= value <= 1
        for value in args.alpha_values
    ):
        parser.error("Every blend alpha must lie between 0 and 1.")
    if len(set(args.alpha_values)) != len(args.alpha_values):
        parser.error("Blend alphas must be unique.")
    return args


def main() -> int:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    if checkpoint.get("stage") != "refiner":
        raise ValueError("This diagnostic requires a refiner checkpoint.")
    config = load_config(args.data_config)
    if args.data_dir:
        config["root"] = str(Path(args.data_dir).expanduser().resolve())
    for key in ("diffraction_kind", "input_log_data", "center_targets"):
        if config[key] != checkpoint["config"][key]:
            raise ValueError(f"Evaluation and checkpoint disagree on {key}.")
    config["training"] = checkpoint["config"]["training"]
    dataset = build_dataset(config, args.split, include_ideal=True)
    verify_data_identity(checkpoint, dataset.manifest)
    device = torch.device(args.device)
    loader = build_loader(dataset, args.batch_size, args.num_workers, False, device)
    model = make_model("refiner", config, checkpoint=checkpoint).to(device).eval()

    output_dir = Path(args.output_dir)
    if (output_dir / "metrics_v3.json").exists() or (
        output_dir / "per_sample_v3.csv"
    ).exists():
        raise FileExistsError("Output files already exist; choose a new --output-dir.")
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )
    records = []
    with torch.no_grad():
        for batch_index, raw in enumerate(loader, 1):
            batch = prepare_batch(raw, config, device)
            result = model(batch["input"], batch["diffraction"])
            initial = result.initial_object
            refined = result.refined_object
            for alpha in args.alpha_values:
                blend = initial + alpha * (refined - initial)
                variants = {
                    f"blend_{alpha:g}": blend,
                    f"projected_blend_{alpha:g}": project_to_measured_modulus(
                        blend, batch["diffraction"], spatial_dims=2
                    ),
                }
                for variant, field in variants.items():
                    values = metrics_2d(
                        field,
                        batch["realspace"],
                        batch["diffraction"],
                        batch["input"],
                        args.support_threshold,
                        batch.get("ideal_intensity"),
                    )
                    arrays = {key: value.cpu().tolist() for key, value in values.items()}
                    for position in range(field.shape[0]):
                        records.append(
                            {
                                "sample": int(batch["index"][position]),
                                "material": batch["material"][position],
                                "variant": variant,
                                **{key: values[position] for key, values in arrays.items()},
                            }
                        )
            if batch_index % 20 == 0 or batch_index == len(loader):
                LOGGER.info("batch=%d/%d", batch_index, len(loader))

    names = [f"{prefix}_{alpha:g}" for alpha in args.alpha_values for prefix in (
        "blend", "projected_blend"
    )]
    summary = {
        variant: {
            key: float(np.mean([row[key] for row in records if row["variant"] == variant]))
            for key in records[0]
            if key not in ("sample", "material", "variant")
        }
        for variant in names
    }
    report = {
        "schema_version": 1,
        "experiment_version": "refiner_postprocess_v3",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint["epoch"],
        "split": args.split,
        "num_samples": len(dataset),
        "alpha_definition": "initial + alpha * (refined - initial)",
        "projection_definition": "replace Fourier modulus with the measured modulus",
        "metrics": summary,
    }
    (output_dir / "metrics_v3.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (output_dir / "per_sample_v3.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    for variant in names:
        metrics = summary[variant]
        LOGGER.info(
            "%s | complex_nrmse=%.6f | amp_ssim=%.6f | support_iou=%.6f | measured_modulus_rel_l1=%.6f",
            variant,
            metrics["complex_nrmse"],
            metrics["real_amp_ssim"],
            metrics["real_support_iou"],
            metrics["relative_l1_modulus"],
        )
    LOGGER.info("Output=%s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
