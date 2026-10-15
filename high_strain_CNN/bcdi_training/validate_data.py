"""Validate prepared BCDI provenance, seed separation and Fourier pairing."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from ecdi_training.data import DiffractionDataset2D, load_config
from .prepare_data import sha256_file, modulus_from_object, SPLITS


def verify_manifest(config, *, check_hashes=False):
    root = Path(config["root"])
    provenance = config.get("provenance", {})
    if "manifest_sha256" not in provenance:
        raise ValueError(
            "Use training_config.json produced by bcdi_training.prepare_data."
        )
    path = root / provenance["manifest"]
    if sha256_file(path) != provenance["manifest_sha256"]:
        raise ValueError(
            "Prepared manifest hash differs from the training configuration."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest["status"] != "complete"
        or config["diffraction_kind"] != "modulus"
        or config.get("storage") != "npy"
    ):
        raise ValueError(
            "A completed modulus-valued prepared BCDI dataset is required."
        )
    all_seeds = []
    for split in SPLITS:
        seeds = manifest["seed_groups"][split]
        all_seeds.extend(seeds)
        rows = manifest["splits"][split]
        if not rows or len(rows) != manifest["counts"][split]:
            raise ValueError(f"Incorrect count for {split}.")
        if sorted({row["seed"] for row in rows}) != sorted(seeds):
            raise ValueError(f"Seed groups and sample mappings disagree for {split}.")
        pairs = {(row["seed"], row["source_index"]) for row in rows}
        if len(pairs) != len(rows) or [row["index"] for row in rows] != list(
            range(len(rows))
        ):
            raise ValueError(f"Duplicate or nonsequential sample mappings for {split}.")
        dataset = DiffractionDataset2D(config, split)
        if len(dataset) != len(rows):
            raise ValueError(f"Prepared array count differs for {split}.")
        for kind, record in manifest["arrays"][split].items():
            path = root / config["splits"][split][kind]
            if path.name != record["file"] or path.stat().st_size != record["bytes"]:
                raise ValueError(f"Prepared array filename/size differs: {path}")
            if check_hashes and sha256_file(path) != record["sha256"]:
                raise ValueError(f"Prepared array content hash differs: {path}")
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("A source random-field seed appears in multiple splits.")
    return manifest


def validate(config, *, samples_per_split=12, check_hashes=True):
    manifest = verify_manifest(config, check_hashes=check_hashes)
    results = {}
    for split in SPLITS:
        dataset = DiffractionDataset2D(config, split)
        positions = np.unique(
            np.linspace(
                0, len(dataset) - 1, min(samples_per_split, len(dataset)), dtype=int
            )
        )
        errors = []
        for position in positions:
            sample = dataset[int(position)]
            field = sample["realspace"][0].numpy()
            measured = sample["diffraction"][0].numpy()
            expected = modulus_from_object(field)
            norm = float(np.linalg.norm(measured.astype(np.float64)))
            if (
                norm <= 0
                or not np.isfinite(field).all()
                or not np.isfinite(measured).all()
                or (measured < 0).any()
            ):
                raise ValueError(f"Invalid prepared data in {split}, index={position}.")
            error = float(np.linalg.norm(expected.astype(np.float64) - measured) / norm)
            if not np.isfinite(error) or error > 2e-5:
                raise ValueError(
                    f"Prepared Fourier pairing failed: {split} index={position} error={error}"
                )
            errors.append(error)
        results[split] = {
            "num_samples": len(dataset),
            "seeds": manifest["seed_groups"][split],
            "checked_samples": len(errors),
            "max_fourier_modulus_relative_l2": max(errors),
        }
    return {
        "passed": True,
        "seed_disjoint": True,
        "prepared_file_hashes_checked": check_hashes,
        "source_phase_convention": manifest["phase_convention"],
        "splits": results,
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
        config["root"] = str(Path(args.data_dir).resolve())
    result = validate(config, samples_per_split=args.samples_per_split)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    logging.info("BCDI preflight passed | %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
