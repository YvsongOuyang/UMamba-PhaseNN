"""Generate paper-style synthetic samples accepted by the author's loader."""

from __future__ import annotations

import argparse
import json
import logging
from itertools import product
from pathlib import Path

import numpy as np

from .config import load_simulation_config
from .generator import generate_sample, save_sample


LOGGER = logging.getLogger("high_strain.simulation")
PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_DIR / "configs" / "simulation_paper.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_DIR / "artifacts" / "simulation" / "datasets"),
    )
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--balanced-categories",
        action="store_true",
        help="Cycle through every shape/phase Cartesian-product pair.",
    )
    parser.add_argument(
        "--save-extras",
        action="store_true",
        help="Also store support, real-space object/phase, and clean intensity.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    config_path = Path(args.config).expanduser().resolve()
    config = load_simulation_config(config_path)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("sample_*.npz"))
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} already contains generated samples; pass --overwrite."
        )
    if existing:
        for sample_path in existing:
            sample_path.unlink()
        LOGGER.info("Removed %d existing generated samples", len(existing))

    rng = np.random.default_rng(args.seed)
    balanced_pairs = tuple(product(config.shape.types, config.phase.types))
    records = []
    for index in range(args.num_samples):
        if args.balanced_categories:
            shape_type, phase_type = balanced_pairs[index % len(balanced_pairs)]
        else:
            shape_type, phase_type = None, None
        sample = generate_sample(
            config,
            rng,
            shape_type=shape_type,
            phase_type=phase_type,
        )
        destination = save_sample(
            sample,
            output_dir / f"sample_{index:05d}.npz",
            save_extras=args.save_extras,
        )
        records.append(
            {
                "index": index,
                "file": destination.name,
                "shape_type": sample.metadata["shape_type"],
                "phase_type": sample.metadata["phase_type"],
                "oversampling_ratio": sample.metadata["oversampling_ratio"],
                "peak_photons": sample.metadata["noise"]["peak_photons"],
                "support_voxels": sample.metadata["support_voxels"],
            }
        )
        LOGGER.info(
            "Generated %s | shape=%s | phase=%s | oversampling=%.3f",
            destination.name,
            sample.metadata["shape_type"],
            sample.metadata["phase_type"],
            sample.metadata["oversampling_ratio"],
        )

    manifest = {
        "generator": "high_strain_CNN.simulation",
        "config_path": str(config_path),
        "config": config.to_dict(),
        "seed": args.seed,
        "num_samples": args.num_samples,
        "balanced_categories": args.balanced_categories,
        "balanced_strategy": (
            "cartesian_product_cycle" if args.balanced_categories else None
        ),
        "save_extras": args.save_extras,
        "samples": records,
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOGGER.info("Saved dataset manifest: %s", manifest_path)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )
    raise SystemExit(main())
