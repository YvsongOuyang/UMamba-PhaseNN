"""Managed, resumable CPU generation of native paired 512-square BCDI data."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import math
import multiprocessing
import os
import platform
import shutil
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np

from .prepare_data import (
    HERE,
    SPLITS,
    partition_seeds,
    sha256_file,
    object_from_labels,
    modulus_from_object,
)


SOURCE = HERE / "vendor" / "simulator_source.py"
DEFAULT_CONFIG = HERE / "configs" / "generation_512.json"
LOGGER = logging.getLogger("bcdi.generate")
KERNEL_DEFINITIONS = {
    "fft_centered",
    "GenerationConfig",
    "gaussian",
    "kernels",
    "random_field",
    "samples",
}


@lru_cache(maxsize=1)
def simulator():
    """Load the exact pinned NumPy kernels without Matplotlib or CLI effects."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name in KERNEL_DEFINITIONS
    ]
    if {node.name for node in tree.body} != KERNEL_DEFINITIONS:
        raise ValueError("Pinned simulator definitions have changed.")
    module = types.ModuleType("bcdi_training.vendor._active_kernel")
    module.__dict__.update(np=np, math=math, dataclass=dataclass)
    sys.modules[module.__name__] = module
    exec(compile(tree, str(SOURCE), "exec"), module.__dict__)
    return module


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def array_hash(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


@contextmanager
def dataset_lock(output):
    """Hold an OS lock that is released even if a generation process is killed."""
    handle = (output / ".generation.lock").open("a+b")
    try:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(
                "Another process is writing this dataset; only one generation controller is allowed."
            ) from error
        yield
    finally:
        handle.close()


def specification(config):
    """Include geometry, RNG environment and source identity in dataset versioning."""
    settings = simulator().GenerationConfig(
        seeds=config["num_seeds"], **config["simulation"]
    )
    settings.validate()
    if settings.image_size != 512 or config["num_seeds"] < 3:
        raise ValueError(
            "Managed training generation requires 512-square data and at least three seeds."
        )
    if (
        config["seed_start"] < 0
        or config["seed_start"] + config["num_seeds"] - 1 > np.iinfo(np.uint32).max
    ):
        raise ValueError("The RandomState seed range must fit uint32.")
    seeds = list(
        range(config["seed_start"], config["seed_start"] + config["num_seeds"])
    )
    groups = partition_seeds(
        seeds, config["split_seed"], config["val_fraction"], config["test_fraction"]
    )
    return {
        "schema_version": 1,
        "dataset_version": config["dataset_version"],
        "simulation": asdict(settings),
        "seed_start": config["seed_start"],
        "num_seeds": config["num_seeds"],
        "split_seed": config["split_seed"],
        "val_fraction": config["val_fraction"],
        "test_fraction": config["test_fraction"],
        "seed_groups": groups,
        "simulator_sha256": sha256_file(SOURCE),
        "generation_adapter_sha256": sha256_file(Path(__file__)),
        "label_adapter_sha256": sha256_file(HERE / "prepare_data.py"),
        "numpy": np.__version__,
        "diffraction_dtype": "float32",
        "realspace_dtype": "complex64",
    }


def generate_seed(seed, settings_dict):
    """Generate one random field with unchanged source draw order and formulas."""
    kernel = simulator()
    settings = kernel.GenerationConfig(**settings_dict)
    domain, envelope = kernel.kernels(settings)
    field = kernel.random_field(seed, domain, settings)
    arrays = kernel.samples(field, envelope, settings)
    del field
    measured = np.ascontiguousarray(arrays["modulus_detector"], dtype=np.float32)
    objects = object_from_labels(
        arrays["object_modulus"], arrays["object_phase"], settings.phase_convention
    )
    if (
        not np.isfinite(objects).all()
        or not np.isfinite(measured).all()
        or (measured < 0).any()
    ):
        raise ValueError(f"Seed {seed} produced invalid values.")
    maximum_error = 0.0
    for index in range(len(objects)):
        expected = modulus_from_object(objects[index])
        norm = float(np.linalg.norm(measured[index].astype(np.float64)))
        if norm <= 0 or np.abs(objects[index]).max() <= 0:
            raise ValueError(
                f"Seed {seed} produced an empty object or diffraction pattern."
            )
        error = float(
            np.linalg.norm(expected.astype(np.float64) - measured[index]) / norm
        )
        if not np.isfinite(error) or error > 2e-5:
            raise ValueError(f"Seed {seed} Fourier-pair error={error}.")
        maximum_error = max(maximum_error, error)
    return measured, objects, maximum_error


def layout(spec):
    per_seed = spec["simulation"]["samples_per_axis"] ** 2
    jobs = {}
    for split in SPLITS:
        for position, seed in enumerate(spec["seed_groups"][split]):
            jobs[seed] = {
                "split": split,
                "start": position * per_seed,
                "count": per_seed,
            }
    return jobs


def checked_record(path, seed, job, config_hash, arrays):
    """Verify flushed rows before accepting a completed seed on resume."""
    record = json.loads(path.read_text(encoding="utf-8"))
    if (
        record.get("config_hash") != config_hash
        or record.get("seed") != seed
        or any(record.get(key) != value for key, value in job.items())
    ):
        raise ValueError(f"Completed-seed metadata differs: {path}")
    rows = slice(job["start"], job["start"] + job["count"])
    for kind in ("diffraction", "realspace"):
        if array_hash(arrays[job["split"]][kind][rows]) != record[kind + "_sha256"]:
            raise ValueError(
                f"Completed seed {seed} data are corrupted; resume will not silently accept them."
            )
    return record


def generate(
    output, config, *, workers=4, resume=False, training_template=None, log_dir=None
):
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with dataset_lock(output):
        return _generate(
            output,
            config,
            workers=workers,
            resume=resume,
            training_template=training_template,
            log_dir=log_dir,
        )


def _generate(output, config, *, workers, resume, training_template, log_dir):
    if workers < 1:
        raise ValueError("workers must be positive.")
    spec = specification(config)
    config_hash = canonical_hash(spec)
    jobs = layout(spec)
    counts = {
        split: sum(job["count"] for job in jobs.values() if job["split"] == split)
        for split in SPLITS
    }
    output = Path(output).expanduser().resolve()
    spec_path = output / "generation_config.json"
    if resume:
        if (
            not spec_path.is_file()
            or json.loads(spec_path.read_text(encoding="utf-8")) != spec
        ):
            raise ValueError(
                "Resume configuration/source/NumPy version differs; use the original configuration or a new dataset directory."
            )
    elif any(path.name != ".generation.lock" for path in output.iterdir()):
        raise FileExistsError(
            "Dataset directory is not empty; use --resume with the same configuration, or a new directory."
        )
    output.mkdir(parents=True, exist_ok=True)
    log_dir = (
        Path(log_dir).resolve()
        if log_dir
        else HERE.parent / "artifacts" / "generation" / "bcdi_2d" / output.name
    )
    if log_dir == output or log_dir.is_relative_to(output):
        raise ValueError("Generation logs must be outside the dataset directory.")
    log_dir.mkdir(parents=True, exist_ok=True)
    template = json.loads(
        Path(training_template or HERE / "configs" / "bcdi_2d.json").read_text(
            encoding="utf-8"
        )
    )
    paths = {
        split: {
            kind: output / template["splits"][split][kind]
            for kind in ("diffraction", "realspace")
        }
        for split in SPLITS
    }
    total_bytes = sum(counts.values()) * 512 * 512 * 12
    if resume:
        allocated = sum(
            getattr(path.stat(), "st_blocks", path.stat().st_size / 512) * 512
            for split in paths.values()
            for path in split.values()
            if path.exists()
        )
    else:
        allocated = 0
    remaining_bytes = max(0, total_bytes - allocated)
    if shutil.disk_usage(output).free < remaining_bytes * 1.1 + 64 * 2**20:
        raise OSError(
            f"Insufficient disk: final arrays need {total_bytes / 2**30:.2f} GiB; remaining allocation needs headroom."
        )
    arrays = {}
    if not resume:
        atomic_json(spec_path, spec)
    records_dir = output / "seed_records"
    records_dir.mkdir(exist_ok=True)
    started = time.monotonic()
    try:
        for split in SPLITS:
            arrays[split] = {}
            for kind, path in paths[split].items():
                dtype = np.dtype("float32" if kind == "diffraction" else "complex64")
                expected = (counts[split], 512, 512)
                if resume and path.exists():
                    array = np.load(path, mmap_mode="r+", allow_pickle=False)
                    if array.shape != expected or array.dtype != dtype:
                        raise ValueError(f"Resume array shape/dtype differs: {path}")
                else:
                    if resume and any(records_dir.glob("seed_*.json")):
                        raise ValueError(
                            f"A completed dataset array is missing: {path}"
                        )
                    array = np.lib.format.open_memmap(
                        path, mode="w+", dtype=dtype, shape=expected
                    )
                arrays[split][kind] = array
        records, pending_seeds = {}, []
        for seed, job in sorted(jobs.items()):
            path = records_dir / f"seed_{seed}.json"
            if path.exists():
                records[seed] = checked_record(path, seed, job, config_hash, arrays)
            else:
                pending_seeds.append(seed)
        LOGGER.info(
            "Dataset=%s | samples=%d | splits=%s | workers=%d | storage=%.2f GiB | completed seeds=%d/%d",
            output,
            sum(counts.values()),
            counts,
            workers,
            total_bytes / 2**30,
            len(records),
            len(jobs),
        )
        atomic_json(
            log_dir / "generation_config.json",
            {
                "protocol": spec,
                "config_hash": config_hash,
                "output_dir": str(output),
                "workers": workers,
                "python": platform.python_version(),
                "executable": sys.executable,
            },
        )
        status_path = log_dir / "generation_status.json"
        initial_completed = len(records)

        def accept(seed, result):
            measured, objects, error = result
            job = jobs[seed]
            rows = slice(job["start"], job["start"] + job["count"])
            record = {
                "config_hash": config_hash,
                "seed": seed,
                **job,
                "max_fourier_modulus_relative_l2": error,
            }
            for kind, value in (("diffraction", measured), ("realspace", objects)):
                arrays[job["split"]][kind][rows] = value
                arrays[job["split"]][kind].flush()
                record[kind + "_sha256"] = array_hash(value)
            atomic_json(records_dir / f"seed_{seed}.json", record)
            records[seed] = record
            elapsed = time.monotonic() - started
            completed_now = len(records) - initial_completed
            eta = elapsed / completed_now * (len(jobs) - len(records))
            atomic_json(
                status_path,
                {
                    "status": "generating",
                    "config_hash": config_hash,
                    "completed_seeds": len(records),
                    "total_seeds": len(jobs),
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta,
                },
            )
            LOGGER.info(
                "Seeds=%d/%d | samples=%d/%d | seed=%d | split=%s | elapsed=%.1fs | eta=%.1fs",
                len(records),
                len(jobs),
                sum(value["count"] for value in records.values()),
                sum(counts.values()),
                seed,
                job["split"],
                elapsed,
                eta,
            )

        if workers == 1:
            for seed in pending_seeds:
                accept(seed, generate_seed(seed, spec["simulation"]))
        elif pending_seeds:
            # Bound in-flight results to one seed per worker; do not queue the
            # entire dataset as large pickle results in the parent process.
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                os.environ.setdefault(name, "1")
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                remaining = iter(pending_seeds)
                pending = {
                    pool.submit(generate_seed, seed, spec["simulation"]): seed
                    for seed in pending_seeds[:workers]
                }
                for _ in range(min(workers, len(pending_seeds))):
                    next(remaining)
                while pending:
                    ready, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in ready:
                        seed = pending.pop(future)
                        accept(seed, future.result())
                        next_seed = next(remaining, None)
                        if next_seed is not None:
                            pending[
                                pool.submit(
                                    generate_seed, next_seed, spec["simulation"]
                                )
                            ] = next_seed
        LOGGER.info(
            "All seeds complete; hashing final arrays and publishing the training manifest."
        )
        atomic_json(
            status_path,
            {
                "status": "hashing",
                "completed_seeds": len(records),
                "total_seeds": len(jobs),
            },
        )
        for split in arrays.values():
            for array in split.values():
                array.flush()
        report = {
            "schema_version": 1,
            "status": "complete",
            "dataset_name": "bcdi_simulator_512",
            "dataset_version": spec["dataset_version"],
            "source_directory": ".",
            "source_kind": "pinned_simulator_direct_npy",
            "source_manifest_sha256": sha256_file(spec_path),
            "generation_config_hash": config_hash,
            "phase_convention": spec["simulation"]["phase_convention"],
            "split_seed": spec["split_seed"],
            "seed_groups": spec["seed_groups"],
            "counts": counts,
            "max_fourier_modulus_relative_l2": max(
                value["max_fourier_modulus_relative_l2"] for value in records.values()
            ),
            "splits": {},
            "arrays": {},
        }
        for split in SPLITS:
            report["splits"][split] = [
                {
                    "index": jobs[seed]["start"] + index,
                    "seed": seed,
                    "source_file": f"procedural_seed_{seed}",
                    "source_index": index,
                }
                for seed in spec["seed_groups"][split]
                for index in range(jobs[seed]["count"])
            ]
            report["arrays"][split] = {
                kind: {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for kind, path in paths[split].items()
            }
        atomic_json(output / "dataset_manifest.json", report)
        template["root"], template["diffraction_kind"] = ".", "modulus"
        template["provenance"] = {
            "manifest": "dataset_manifest.json",
            "manifest_sha256": sha256_file(output / "dataset_manifest.json"),
            "source_phase_convention": report["phase_convention"],
            "target": "physical complex object used by the simulator FFT",
        }
        training_path = output / "training_config.json"
        if training_path.exists():
            existing = json.loads(training_path.read_text(encoding="utf-8"))
            if (
                existing.get("provenance", {}).get("manifest_sha256")
                != template["provenance"]["manifest_sha256"]
            ):
                raise ValueError(
                    "Existing training configuration points to a different dataset manifest."
                )
        else:
            atomic_json(training_path, template)
        atomic_json(
            status_path,
            {
                "status": "complete",
                "completed_seeds": len(records),
                "total_seeds": len(jobs),
                "elapsed_seconds": time.monotonic() - started,
                "training_config": str(output / "training_config.json"),
            },
        )
        LOGGER.info("Complete | training_config=%s", output / "training_config.json")
        return report
    except BaseException as error:
        atomic_json(
            log_dir / "generation_status.json",
            {
                "status": (
                    "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                ),
                "error": str(error),
                "resume": "Rerun with the same configuration and --resume.",
            },
        )
        raise
    finally:
        for split in arrays.values():
            for array in split.values():
                array._mmap.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--log-dir", default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-seeds", type=int)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--samples-per-axis", type=int)
    parser.add_argument("--phase-convention", choices=("author", "radians"))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    for key in ("num_seeds", "seed_start"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    for key in ("samples_per_axis", "phase_convention"):
        if getattr(args, key) is not None:
            config["simulation"][key] = getattr(args, key)
    log_dir = (
        Path(args.log_dir).resolve()
        if args.log_dir
        else HERE.parent
        / "artifacts"
        / "generation"
        / "bcdi_2d"
        / Path(args.output_dir).name
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        force=True,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "console.log", encoding="utf-8"),
        ],
    )
    generate(
        args.output_dir,
        config,
        workers=args.workers,
        resume=args.resume,
        log_dir=log_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
