"""Generate paper-style synthetic samples accepted by the author's loader."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from itertools import product
from pathlib import Path

import numpy as np

from .config import load_simulation_config
from .generator import generate_sample, generate_unstrained_control, save_sample


LOGGER = logging.getLogger("high_strain.simulation")
PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        help="Simulation config; defaults to configs/simulation_paper.json.",
    )
    parser.add_argument(
        "--unstrained-from",
        help="Existing generated dataset to pair with phase-zero controls; preserves sample names.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_DIR / "artifacts" / "simulation" / "datasets"),
    )
    parser.add_argument(
        "--num-samples", type=int,
        help="Defaults to 3 new samples, or all references with --unstrained-from.",
    )
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--shape-type",
        help="Fix one enabled shape family instead of drawing it randomly.",
    )
    parser.add_argument(
        "--phase-type",
        help="Fix one enabled phase family instead of drawing it randomly.",
    )
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
    reference_dir = None
    reference_paths = []
    reference_manifest_path = None
    config_path = None
    config = None
    if args.unstrained_from:
        if args.config or args.balanced_categories or args.shape_type or args.phase_type:
            raise ValueError(
                "Paired controls inherit references; omit --config, category balancing, "
                "and fixed category options."
            )
        reference_dir = Path(args.unstrained_from).expanduser().resolve()
        reference_paths = sorted(reference_dir.glob("sample_*.npz"))
        if not reference_paths:
            raise ValueError(f"No reference samples in {reference_dir}.")
        reference_manifest_path = reference_dir / "dataset_manifest.json"
        if not reference_manifest_path.is_file():
            raise FileNotFoundError(f"Missing reference provenance: {reference_manifest_path}")
        num_samples = len(reference_paths) if args.num_samples is None else args.num_samples
        if num_samples > len(reference_paths):
            raise ValueError("--num-samples exceeds the number of reference samples.")
    else:
        config_path = Path(
            args.config or PROJECT_DIR / "configs" / "simulation_paper.json"
        ).expanduser().resolve()
        config = load_simulation_config(config_path)
        if args.shape_type and args.shape_type not in config.shape.types:
            raise ValueError(f"Shape {args.shape_type!r} is not enabled by the configuration.")
        if args.phase_type and args.phase_type not in config.phase.types:
            raise ValueError(f"Phase {args.phase_type!r} is not enabled by the configuration.")
        if args.balanced_categories and (args.shape_type or args.phase_type):
            raise ValueError("Use either --balanced-categories or fixed category options, not both.")
        num_samples = 3 if args.num_samples is None else args.num_samples
    if num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir == reference_dir:
        raise ValueError("Paired controls must not overwrite their reference dataset.")
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
    balanced_pairs = tuple(product(config.shape.types, config.phase.types)) if config else ()
    save_extras = args.save_extras or reference_dir is not None
    records = []
    for index in range(num_samples):
        if reference_dir is not None:
            sample = generate_unstrained_control(reference_paths[index], rng)
            filename = reference_paths[index].name
        else:
            if args.balanced_categories:
                shape_type, phase_type = balanced_pairs[index % len(balanced_pairs)]
            else:
                shape_type, phase_type = args.shape_type, args.phase_type
            sample = generate_sample(
                config,
                rng,
                shape_type=shape_type,
                phase_type=phase_type,
            )
            filename = f"sample_{index:05d}.npz"
        destination = save_sample(
            sample,
            output_dir / filename,
            save_extras=save_extras,
        )
        records.append(
            {
                "index": index,
                "file": destination.name,
                "shape_type": sample.metadata["shape_type"],
                "phase_type": sample.metadata["phase_type"],
                "oversampling_ratio": sample.metadata["oversampling_ratio"],
                "actual_oversampling_ratio": sample.metadata["actual_oversampling_ratio"],
                "phase_span_before_centering_rad": sample.metadata["phase_parameters"]["actual_peak_to_peak_rad"],
                "phase_span_after_centering_rad": sample.metadata["phase_parameters"]["effective_peak_to_peak_rad"],
                "peak_photons": sample.metadata["noise"]["peak_photons"],
                "support_voxels": sample.metadata["support_voxels"],
            }
        )
        if "control" in sample.metadata:
            records[-1]["control"] = sample.metadata["control"]
        LOGGER.info(
            "Generated %s | shape=%s | phase=%s | requested OS=%.3f | actual OS=%s",
            destination.name,
            sample.metadata["shape_type"],
            sample.metadata["phase_type"],
            sample.metadata["oversampling_ratio"],
            "/".join(f"{value:.2f}" for value in sample.metadata["actual_oversampling_ratio"]),
        )

    manifest = {
        "generator": "high_strain_CNN.simulation",
        "config_path": str(config_path) if config_path else None,
        "config": config.to_dict() if config else None,
        "config_sha256": hashlib.sha256(
            json.dumps(config.to_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest() if config else None,
        "generator_source_sha256": hashlib.sha256(
            Path(__file__).with_name("generator.py").read_bytes()
        ).hexdigest(),
        "seed": args.seed,
        "num_samples": num_samples,
        "balanced_categories": args.balanced_categories,
        "balanced_strategy": (
            "cartesian_product_cycle" if args.balanced_categories else None
        ),
        "fixed_shape_type": args.shape_type,
        "fixed_phase_type": args.phase_type,
        "save_extras": save_extras,
        "samples": records,
    }
    if reference_manifest_path is not None:
        manifest["control"] = {
            "type": "zero_realspace_phase",
            "reference_dataset": str(reference_dir),
            "reference_manifest_sha256": hashlib.sha256(reference_manifest_path.read_bytes()).hexdigest(),
            "reference_config": json.loads(reference_manifest_path.read_text(encoding="utf-8"))["config"],
            "noise": "same peak photon counts, independent Poisson draws",
            "selection": "same sorted sample filenames; no random geometry or phase draw",
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
