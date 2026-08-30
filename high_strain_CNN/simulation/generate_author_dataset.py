"""Generate an NPZ dataset from the particle/diffraction code supplied by the authors."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .sample_io import COMPACT_STORAGE

from .author_generator import (
    AUTHOR_GENERATOR_PROTOCOL,
    AUTHOR_PHASE_SAMPLING,
    DEFAULT_AUTHOR_CODE_DIR,
    PAPER_SHAPES,
    PAPER_STRAINS,
    AuthorParticle,
    author_source_manifest,
    create_paper_particle,
    file_sha256,
    generate_notebook_sample,
    generate_paper_observation,
    load_author_modules,
    paper_category_for_index,
    save_author_sample,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_DIR / "artifacts" / "simulation" / "author_generator_paper"
LOGGER = logging.getLogger("high_strain.author_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--author-code-dir", default=str(DEFAULT_AUTHOR_CODE_DIR),
        help="Author source directory; defaults to the bundled vendor copy.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--profile", choices=("notebook", "paper"), default="paper")
    counts = parser.add_mutually_exclusive_group()
    counts.add_argument("--num-samples", type=int, default=None)
    counts.add_argument(
        "--split-counts", type=int, nargs=3, metavar=("TRAIN", "VAL", "TEST"),
        help="Exact split sizes; a particle is never shared between splits.",
    )
    parser.add_argument("--storage", choices=("standard", "compact"), default="standard")
    parser.add_argument(
        "--scattering-backend", choices=("compat", "pynx_cuda"), default="compat",
        help="CPU FFT/NUFFT compatibility or native author PyNX CUDA (Linux).",
    )
    parser.add_argument("--observations-per-particle", type=int, default=3)
    parser.add_argument(
        "--oversampling-policy", choices=("error", "record"), default="error",
        help="Stop on oversampling <=2, or retain and flag the unmodified source draw.",
    )
    parser.add_argument(
        "--category-sampling", choices=("balanced", "random"), default="random"
    )
    parser.add_argument(
        "--random-q-rotation",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--save-extras",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include support, complex object, and clean intensity; source phase is wrapped.",
    )
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args()
    if args.split_counts is not None:
        if any(count < 0 for count in args.split_counts) or not sum(args.split_counts):
            parser.error("Split counts must be nonnegative with a positive total.")
        if args.profile != "paper":
            parser.error("Particle-disjoint splits require the paper profile.")
        args.num_samples = sum(args.split_counts)
    elif args.num_samples is None:
        args.num_samples = 9
    if args.num_samples < 1 or args.observations_per_particle < 1:
        parser.error("Sample and observation counts must be positive.")
    if args.profile == "notebook" and args.random_q_rotation is True:
        parser.error("The notebook profile is fixed to its unrotated executed path.")
    if args.storage == "compact" and (args.profile != "paper" or not args.save_extras):
        parser.error("Compact storage requires the paper profile and object/support extras.")
    return args


def sample_schedule(
    num_samples: int, observations: int, split_counts: list[int] | None
) -> Iterator[tuple[int, str | None, int]]:
    """Yield index, split, category index; close a particle at every split boundary."""
    splits = (
        zip(("train", "val", "test"), split_counts)
        if split_counts is not None else [(None, num_samples)]
    )
    index = 0
    particle_offset = 0
    for split, count in splits:
        for local_index in range(count):
            yield index, split, particle_offset * observations + local_index
            index += 1
        particle_offset += (count + observations - 1) // observations


def backend_manifest(backend: str) -> dict[str, Any]:
    packages = ("numpy", "scipy", "ase", "finufft", "pynx", "pycuda")
    versions = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            pass
    manifest = {
        "name": backend, "python": sys.version, "packages": versions,
        "scattering": "native PyNX CUDA" if backend == "pynx_cuda" else "FFT/NUFFT; native PyNX parity not measured",
        "thomson_factor": "native PyNX" if backend == "pynx_cuda" else "constant omitted; cancels in normalized intensity",
        "coordinate_io": "in-memory positions; no eight-decimal LMP roundtrip",
    }
    if backend == "pynx_cuda":
        import pycuda.driver as cuda
        manifest["visible_devices"] = [cuda.Device(i).name() for i in range(cuda.Device.count())]
        manifest["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return manifest


def _configure_logging(output_dir: Path, level: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.handlers.clear()
    LOGGER.setLevel(getattr(logging, level))
    LOGGER.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_dir / "generation.log", mode="w", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def main() -> int:
    args = parse_args()
    author_code_dir = Path(args.author_code_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not author_code_dir.is_dir():
        raise FileNotFoundError(f"Author code directory not found: {author_code_dir}")
    existing_samples = sorted(output_dir.glob("sample_*.npz")) if output_dir.exists() else []
    if existing_samples or (output_dir / "dataset_manifest.json").exists():
        raise FileExistsError(
            f"Output directory already contains a generated dataset: {output_dir}"
        )
    _configure_logging(output_dir, args.log_level)
    modules = load_author_modules(author_code_dir, scattering_backend=args.scattering_backend)
    random_q_rotation = (
        args.random_q_rotation
        if args.random_q_rotation is not None
        else args.profile == "paper"
    )
    category_rng = random.Random(args.seed + 7919)
    random_shapes: dict[int, str] = {}
    particle_cache: dict[int, AuthorParticle] = {}
    sample_records: list[dict[str, Any]] = []
    pair_counts: Counter[str] = Counter()
    started = time.perf_counter()

    LOGGER.info(
        "Generating profile=%s | samples=%d | observations/particle=%d | storage=%s | backend=%s",
        args.profile,
        args.num_samples,
        args.observations_per_particle,
        args.storage,
        args.scattering_backend,
    )
    for index, split, category_index in sample_schedule(
        args.num_samples, args.observations_per_particle, args.split_counts
    ):
        if args.profile == "notebook":
            sample = generate_notebook_sample(
                author_code_dir,
                modules,
                args.seed + index,
            )
        else:
            particle_index, observation_index, shape, strain = paper_category_for_index(
                category_index,
                args.observations_per_particle,
                category_sampling=args.category_sampling,
                rng=category_rng,
                random_shapes=random_shapes,
            )
            if particle_index not in particle_cache:
                particle_cache.clear()
                particle_cache[particle_index] = create_paper_particle(
                    author_code_dir,
                    modules,
                    args.seed + 1_000_000 + particle_index,
                    shape,
                    particle_index,
                )
            sample = generate_paper_observation(
                modules,
                particle_cache[particle_index],
                args.seed + index,
                strain,
                observation_index,
                random_q_rotation=random_q_rotation,
                oversampling_policy=args.oversampling_policy,
            )
        sample = replace(sample, metadata={
            **sample.metadata, "split": split, "scattering_backend": args.scattering_backend,
        })
        if sample.metadata.get("satisfies_paper_oversampling") is False:
            LOGGER.warning(
                "Retaining source draw unchanged | sample=%d seed=%d | oversampling=%s",
                index, sample.metadata["seed"], sample.metadata["measured_object_oversampling_xyz"],
            )
        destination = output_dir / f"sample_{index:05d}.npz"
        save_author_sample(destination, sample, save_extras=args.save_extras, storage=args.storage)
        pair_counts[f"{sample.metadata['shape']}+{sample.metadata['strain_argument']}"] += 1
        sample_records.append(
            {
                "index": index,
                "split": split,
                "filename": destination.name,
                "bytes": destination.stat().st_size,
                "sha256": file_sha256(destination),
                "metadata": sample.metadata,
            }
        )
        elapsed = time.perf_counter() - started
        rate = (index + 1) / max(elapsed, 1e-12)
        eta = (args.num_samples - index - 1) / max(rate, 1e-12)
        LOGGER.info(
            "Generated %d/%d | %s/%s | nstep=%d | %.2f s | ETA %.1f s",
            index + 1,
            args.num_samples,
            sample.metadata["shape"],
            sample.metadata["strain_argument"],
            sample.metadata["nstep"],
            sample.metadata["generation_seconds"],
            eta,
        )

    manifest = {
        "schema_version": "1.3",
        "route": "author_generator",
        "profile": args.profile,
        "phase_sampling": AUTHOR_PHASE_SAMPLING if args.profile == "paper" else "notebook_overrides",
        "generator_protocol": AUTHOR_GENERATOR_PROTOCOL if args.profile == "paper" else "notebook_v1",
        "oversampling_policy": args.oversampling_policy
        if args.profile == "paper" else "unmodified notebook draw",
        "compatibility": backend_manifest(args.scattering_backend),
        "storage_schema": COMPACT_STORAGE if args.storage == "compact" else "author_standard",
        "split_unit": "particle",
        "splits": dict(zip(("train", "val", "test"), args.split_counts)) if args.split_counts is not None else {},
        "author_code_dir": str(author_code_dir),
        "author_source_manifest": author_source_manifest(author_code_dir),
        "seed": args.seed,
        "num_samples": args.num_samples,
        "observations_per_particle": args.observations_per_particle,
        "category_sampling": args.category_sampling,
        "random_q_rotation": random_q_rotation,
        "save_extras": args.save_extras,
        "shape_families": list(PAPER_SHAPES) if args.profile == "paper" else ["wulff"],
        "phase_families": list(PAPER_STRAINS) if args.profile == "paper" else ["random"],
        "combination_rule": "one shape x one phase per observation",
        "category_counts": dict(sorted(pair_counts.items())),
        "generation_seconds": time.perf_counter() - started,
        "samples": sample_records,
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOGGER.info("Complete | manifest=%s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
