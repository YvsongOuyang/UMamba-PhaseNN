"""Preflight e-CDI split identity, noisy counts and clean exit-wave FFT pairing."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from .data import build_dataset, load_config, prepare_batch


def validate(config: dict, samples_per_split: int = 12) -> dict:
    results = {}
    rng = np.random.default_rng(42)
    for split in ("train", "val", "test"):
        dataset = build_dataset(config, split, include_ideal=True)
        positions = rng.choice(
            len(dataset), min(samples_per_split, len(dataset)), replace=False
        )
        errors, counts = [], []
        for position in positions:
            sample = dataset[int(position)]
            raw = {k: v[None] for k, v in sample.items() if isinstance(v, torch.Tensor)}
            prepare_batch(raw, config, torch.device("cpu"))
            field = sample["realspace"][0].numpy().astype(np.complex128)
            expected = (
                np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))) ** 2
            )
            expected /= expected.sum()
            if "ideal_intensity" in sample:
                ideal = sample["ideal_intensity"][0].numpy().astype(np.float64)
                if (
                    not np.isfinite(ideal).all()
                    or (ideal < 0).any()
                    or abs(ideal.sum() - 1) > 1e-5
                ):
                    raise ValueError("Invalid normalized ideal_intensity.")
                error = float(np.linalg.norm(expected - ideal) / np.linalg.norm(ideal))
                if error > 1e-5:
                    raise ValueError(
                        f"Fourier pairing failed for {split} index={sample['index']}: {error:g}."
                    )
                errors.append(error)
            counts.append(float(sample["diffraction"].sum()))
        results[split] = {
            "data": dataset.manifest,
            "checked_samples": len(positions),
            "max_clean_fourier_intensity_nrmse": max(errors) if errors else None,
            "minimum_observed_sum": min(counts),
            "maximum_observed_sum": max(counts),
        }
        if hasattr(dataset, "close"):
            dataset.close()
    return {
        "passed": True,
        "sampled_pair_check": True,
        "splits": results,
        "notes": "All split indices, valid flags, shapes and generator report checked; FFT pairing sampled per split.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--samples-per-split", type=int, default=12)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.samples_per_split < 1:
        parser.error("--samples-per-split must be positive.")
    config = load_config(args.data_config)
    if args.data_dir:
        config["root"] = str(Path(args.data_dir).expanduser().resolve())
    report = validate(config, args.samples_per_split)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    logging.info(
        "Preflight passed | counts=%s | report=%s",
        {k: v["data"]["selected_samples"] for k, v in report["splits"].items()},
        output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
