"""Generate an NPZ dataset from the particle/diffraction code supplied by the authors."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .sample_io import COMPACT_STORAGE
from .generation_execution import dataset_lock, execute_jobs, sample_schedule

from .author_generator import (
    AUTHOR_GENERATOR_PROTOCOL,
    AUTHOR_PHASE_SAMPLING,
    DEFAULT_AUTHOR_CODE_DIR,
    PAPER_SHAPES,
    PAPER_STRAINS,
    author_source_manifest,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_DIR / "artifacts" / "simulation" / "author_generator_paper"
DEFAULT_LOG_ROOT = PROJECT_DIR / "artifacts" / "generation"
LOGGER = logging.getLogger("high_strain.author_dataset")
DATA_ARGUMENTS = (
    "author_code_dir", "output_dir", "profile", "num_samples", "split_counts", "storage",
    "scattering_backend", "observations_per_particle", "oversampling_policy",
    "category_sampling", "random_q_rotation", "save_extras", "seed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--author-code-dir", default=str(DEFAULT_AUTHOR_CODE_DIR),
        help="Author source directory; defaults to the bundled vendor copy.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--log-dir", default=None,
        help="Run logs/config directory; defaults to artifacts/generation/<timestamped-run>.",
    )
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
    parser.add_argument("--workers", type=int, default=1, help="Independent spawned particle workers, not a GPU batch size.")
    parser.add_argument("--worker-threads", type=int, default=1, help="Numeric CPU threads per spawned worker.")
    parser.add_argument("--print-freq", type=int, default=50, help="Progress interval in completed observations.")
    parser.add_argument(
        "--resume-from", default=None, metavar="CONFIG_JSON",
        help="Resume an incomplete dataset using a previous run's config.json; data settings cannot change.",
    )
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
    previous = None
    if args.resume_from:
        try:
            config = json.loads(Path(args.resume_from).expanduser().read_text(encoding="utf-8"))
            previous = {key: config["args"][key] for key in DATA_ARGUMENTS}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(f"Invalid resume config: {exc}")
        parser.set_defaults(**previous)
        args = parser.parse_args()
        if args.num_samples != previous["num_samples"]:
            parser.error("Resume cannot change --num-samples; keep the original dataset schedule.")
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
    if min(args.workers, args.worker_threads, args.print_freq) < 1:
        parser.error("Worker, thread, and print frequency values must be positive.")
    if args.profile == "notebook" and args.random_q_rotation is True:
        parser.error("The notebook profile is fixed to its unrotated executed path.")
    if args.storage == "compact" and (args.profile != "paper" or not args.save_extras):
        parser.error("Compact storage requires the paper profile and object/support extras.")
    args.random_q_rotation = args.random_q_rotation if args.random_q_rotation is not None else args.profile == "paper"
    if previous is not None:
        if args.profile != "paper":
            parser.error("Resume currently requires the paper profile.")
        previous["random_q_rotation"] = (
            previous["random_q_rotation"] if previous["random_q_rotation"] is not None else previous["profile"] == "paper"
        )
        for key in DATA_ARGUMENTS:
            old, new = previous[key], getattr(args, key)
            if key.endswith("_dir"):
                old, new = Path(old).expanduser().resolve(), Path(new).expanduser().resolve()
            if old != new:
                parser.error(f"Resume cannot change --{key.replace('_', '-')}: {old!r} -> {new!r}")
    return args


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
        cuda.init()
        manifest["visible_devices"] = [cuda.Device(i).name() for i in range(cuda.Device.count())]
        manifest["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return manifest


def _configure_logging(log_dir: Path, level: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    for handler in LOGGER.handlers:
        handler.close()
    LOGGER.handlers.clear()
    LOGGER.setLevel(getattr(logging, level))
    LOGGER.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "generation.log", mode="w", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def _generate(args: argparse.Namespace) -> int:
    author_code_dir = Path(args.author_code_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not author_code_dir.is_dir():
        raise FileNotFoundError(f"Author code directory not found: {author_code_dir}")
    existing_samples = sorted(output_dir.glob("sample_*.npz")) if output_dir.exists() else []
    if (existing_samples and not args.resume_from) or (output_dir / "dataset_manifest.json").exists():
        raise FileExistsError(
            f"Output directory already contains a generated dataset: {output_dir}"
        )
    for path in existing_samples:
        try:
            index = int(path.stem.removeprefix("sample_"))
        except ValueError:
            index = -1
        if not 0 <= index < args.num_samples or path.name != f"sample_{index:05d}.npz":
            raise ValueError(f"Unexpected sample outside the requested schedule: {path}")
    source_manifest = author_source_manifest(author_code_dir)
    previous_config = None
    if args.resume_from:
        args.resume_from = str(Path(args.resume_from).expanduser().resolve())
        previous_config = json.loads(Path(args.resume_from).read_text(encoding="utf-8"))
        if previous_config.get("author_source_manifest", source_manifest) != source_manifest:
            raise ValueError("Author source hashes differ from the previous run; refusing to mix generators.")
        if previous_config.get("generator_protocol", AUTHOR_GENERATOR_PROTOCOL) != AUTHOR_GENERATOR_PROTOCOL:
            raise ValueError("Generator protocol differs from the previous run.")
        # Old generators did not take a dataset lock. Detect their recorded Linux PID too.
        if sys.platform == "linux" and previous_config.get("hostname", socket.gethostname()) == socket.gethostname():
            pid = int(previous_config.get("pid", 0))
            try:
                command = Path(f"/proc/{pid}/cmdline").read_bytes()
            except FileNotFoundError:
                command = b""
            if b"simulation.generate_author_dataset" in command or b"generate_author_dataset.py" in command:
                raise RuntimeError(f"Previous generator PID {pid} is still running. Stop it and wait before resuming.")
    run_name = (
        f"author_{args.profile}_{args.storage}_seed{args.seed}_"
        f"{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    log_dir = (
        Path(args.log_dir).expanduser() if args.log_dir else DEFAULT_LOG_ROOT / run_name
    ).resolve()
    if (log_dir / "config.json").exists():
        raise FileExistsError(f"Choose a new log directory; refusing to overwrite {log_dir / 'config.json'}")
    _configure_logging(log_dir, args.log_level)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.author_code_dir = str(author_code_dir)
    args.output_dir = str(output_dir)
    args.log_dir = str(log_dir)
    (log_dir / "config.json").write_text(
        json.dumps({
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "author_source_manifest": source_manifest,
            "generator_protocol": AUTHOR_GENERATOR_PROTOCOL,
            "args": vars(args),
        }, indent=2), encoding="utf-8",
    )
    LOGGER.info("Data directory: %s", output_dir)
    LOGGER.info("Run logs/config: %s", log_dir)
    if previous_config is not None and "author_source_manifest" not in previous_config:
        LOGGER.warning("Legacy resume config has no source hashes; checking saved seeds/schema/settings, not historical source identity.")
    sample_records: list[dict[str, Any]] = []
    pair_counts: Counter[str] = Counter()
    stage_totals: Counter[str] = Counter()
    generated = reused = 0
    started = time.perf_counter()

    LOGGER.info(
        "Generating profile=%s | samples=%d | observations/particle=%d | storage=%s | backend=%s | workers=%d",
        args.profile,
        args.num_samples,
        args.observations_per_particle,
        args.storage,
        args.scattering_backend,
        args.workers,
    )
    if args.workers > 1:
        LOGGER.info("Bounded spawn workers | numeric threads/worker=%d | queued particles<=%d", args.worker_threads, 2 * args.workers)
    for record, was_reused in execute_jobs(args):
        metadata = record["metadata"]
        if metadata.get("satisfies_paper_oversampling") is False and not was_reused:
            LOGGER.warning(
                "Retaining source draw unchanged | sample=%d seed=%d | oversampling=%s",
                record["index"], metadata["seed"], metadata["measured_object_oversampling_xyz"],
            )
        pair_counts[f"{metadata['shape']}+{metadata['strain_argument']}"] += 1
        sample_records.append(record)
        reused += int(was_reused)
        generated += int(not was_reused)
        if not was_reused:
            stage_totals.update(record.get("timings", {}))
        completed = len(sample_records)
        if completed != 1 and completed % args.print_freq and completed != args.num_samples:
            continue
        elapsed = time.perf_counter() - started
        rate = generated / max(elapsed, 1e-12)
        remaining_new = args.num_samples - len(existing_samples) - generated
        eta = format_duration(remaining_new / rate) if rate else "warming up / validating"
        LOGGER.info(
            "Generated %d/%d | new=%d reused=%d | %.3f sample/s | elapsed %s | ETA %s",
            completed, args.num_samples, generated, reused, rate, format_duration(elapsed), eta,
        )
        if generated and stage_totals:
            LOGGER.info("Mean worker time/new sample (overlaps across workers) | %s", " | ".join(
                f"{stage}={seconds / generated:.3f}s" for stage, seconds in sorted(stage_totals.items())
            ))

    manifest = {
        "schema_version": "1.4",
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
        "generation_log_dir": str(log_dir),
        "author_source_manifest": source_manifest,
        "seed": args.seed,
        "num_samples": args.num_samples,
        "observations_per_particle": args.observations_per_particle,
        "category_sampling": args.category_sampling,
        "random_q_rotation": args.random_q_rotation,
        "save_extras": args.save_extras,
        "shape_families": list(PAPER_SHAPES) if args.profile == "paper" else ["wulff"],
        "phase_families": list(PAPER_STRAINS) if args.profile == "paper" else ["random"],
        "combination_rule": "one shape x one phase per observation",
        "category_counts": dict(sorted(pair_counts.items())),
        "generation_seconds": time.perf_counter() - started,
        "execution": {"workers": args.workers, "worker_threads": args.worker_threads, "generated": generated, "reused": reused, "resume_from": args.resume_from},
        "stage_worker_seconds": dict(stage_totals),
        "samples": sorted(sample_records, key=lambda record: record["index"]),
    }
    manifest_path = output_dir / "dataset_manifest.json"
    temporary_manifest = output_dir / ".dataset_manifest.tmp.json"
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)
    LOGGER.info("Complete | manifest=%s", manifest_path)
    return 0


def format_duration(seconds: float) -> str:
    hours, remainder = divmod(max(0, int(seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main() -> int:
    args = parse_args()
    previous_handler = signal.getsignal(signal.SIGTERM)

    def stop(signum, frame):
        raise KeyboardInterrupt("Termination requested; waiting for active particle workers to finish.")

    signal.signal(signal.SIGTERM, stop)
    try:
        with dataset_lock(Path(args.output_dir).expanduser().resolve()):
            return _generate(args)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())
