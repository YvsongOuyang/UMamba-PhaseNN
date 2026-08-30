"""Bounded, particle-level execution; source RNGs stay isolated in spawned processes."""

from __future__ import annotations

import json
import multiprocessing
import os
import random
import time
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .author_generator import (
    AUTHOR_GENERATOR_PROTOCOL, AUTHOR_PHASE_SAMPLING, create_paper_particle,
    file_sha256, generate_notebook_sample, generate_paper_observation,
    load_author_modules, paper_category_for_index, save_author_sample,
)
from .sample_io import COMPACT_STORAGE


@dataclass(frozen=True)
class ObservationJob:
    index: int
    split: str | None
    particle_index: int
    observation_index: int
    shape: str
    strain: str


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


def particle_jobs(args: Any) -> Iterator[tuple[ObservationJob, ...]]:
    """Draw categories in original index order, never in worker completion order."""
    rng = random.Random(args.seed + 7919)
    shapes: dict[int, str] = {}
    group: list[ObservationJob] = []
    for index, split, category_index in sample_schedule(
        args.num_samples, args.observations_per_particle, args.split_counts
    ):
        if args.profile == "paper":
            particle, observation, shape, strain = paper_category_for_index(
                category_index, args.observations_per_particle,
                category_sampling=args.category_sampling, rng=rng, random_shapes=shapes,
            )
        else:
            particle, observation, shape, strain = index, 0, "wulff", "random"
        if group and particle != group[0].particle_index:
            yield tuple(group)
            group = []
        group.append(ObservationJob(index, split, particle, observation, shape, strain))
    if group:
        yield tuple(group)


def sample_record(path: Path, job: ObservationJob, metadata: dict) -> dict:
    return {
        "index": job.index, "split": job.split, "filename": path.name,
        "bytes": path.stat().st_size, "sha256": file_sha256(path), "metadata": metadata,
    }


def read_existing_sample(path: Path, job: ObservationJob, args: Any) -> dict:
    """Check old NPZ metadata, array schema and ZIP CRCs before reusing it."""
    expected = {
        "profile": args.profile, "seed": args.seed + job.index, "split": job.split,
        "scattering_backend": args.scattering_backend, "shape": job.shape,
        "strain_argument": job.strain, "particle_index": job.particle_index,
        "particle_seed": args.seed + 1_000_000 + job.particle_index,
        "observation_index": job.observation_index,
        "generator_protocol": AUTHOR_GENERATOR_PROTOCOL,
        "phase_sampling": AUTHOR_PHASE_SAMPLING,
        "random_q_rotation": args.random_q_rotation,
        "oversampling_policy": args.oversampling_policy,
    }
    try:
        with np.load(path, allow_pickle=False) as stored:
            metadata = json.loads(str(stored["metadata_json"]))
            for key, value in expected.items():
                if key not in metadata or metadata[key] != value:
                    raise ValueError(f"metadata {key}: expected {value!r}, got {metadata.get(key)!r}")
            fields = {"I": np.dtype("float32")}
            if args.storage == "compact":
                if metadata.get("storage_schema") != COMPACT_STORAGE:
                    raise ValueError("storage schema differs")
                fields.update(object=np.dtype("complex64"), support=np.dtype("bool"))
            else:
                fields["phi"] = np.dtype("float32")
                if args.save_extras:
                    fields.update(object=np.dtype("complex64"), support=np.dtype("bool"), I_clean=np.dtype("float32"))
            if set(stored.files) != {*fields, "metadata_json"}:
                raise ValueError("array fields differ from requested storage")
            for key, dtype in fields.items():
                array = stored[key]
                if array.shape != (64, 64, 64) or array.dtype != dtype or not np.isfinite(array).all():
                    raise ValueError(f"invalid {key} shape/dtype/values")
                if key in ("I", "I_clean") and np.any(array < 0):
                    raise ValueError(f"negative {key}")
    except Exception as exc:
        raise ValueError(
            f"Cannot reuse {path}: {exc}. No existing sample was overwritten. "
            "Check the configuration; move a confirmed incomplete file out of the dataset before retrying."
        ) from exc
    return sample_record(path, job, metadata)


def generate_particle_job(args: Any, jobs: tuple[ObservationJob, ...], modules: tuple) -> list[tuple[dict, bool]]:
    output = Path(args.output_dir)
    results = []
    pending = []
    for job in jobs:
        path = output / f"sample_{job.index:05d}.npz"
        if path.exists():
            if not args.resume_from:
                raise FileExistsError(f"Refusing to overwrite {path}")
            results.append((read_existing_sample(path, job, args), True))
        else:
            pending.append(job)
    if not pending:
        return results
    geometry_started = time.perf_counter()
    particle = None
    if args.profile == "paper":
        first = jobs[0]
        particle = create_paper_particle(
            Path(args.author_code_dir), modules,
            args.seed + 1_000_000 + first.particle_index, first.shape, first.particle_index,
        )
    geometry_seconds = time.perf_counter() - geometry_started
    for job in pending:
        if args.profile == "paper":
            sample = generate_paper_observation(
                modules, particle, args.seed + job.index, job.strain, job.observation_index,
                random_q_rotation=args.random_q_rotation, oversampling_policy=args.oversampling_policy,
            )
        else:
            sample = generate_notebook_sample(Path(args.author_code_dir), modules, args.seed + job.index)
        sample = replace(sample, metadata={
            **sample.metadata, "split": job.split, "scattering_backend": args.scattering_backend,
        })
        destination = output / f"sample_{job.index:05d}.npz"
        temporary = output / f".{destination.stem}.{os.getpid()}.tmp.npz"
        write_started = time.perf_counter()
        try:
            save_author_sample(temporary, sample, save_extras=args.save_extras, storage=args.storage)
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite {destination}")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        write_seconds = time.perf_counter() - write_started
        hash_started = time.perf_counter()
        record = sample_record(destination, job, sample.metadata)
        record["timings"] = {
            **sample.metadata.get("stage_seconds", {}),
            "geometry": geometry_seconds, "write": write_seconds,
            "hash": time.perf_counter() - hash_started,
        }
        geometry_seconds = 0.0
        results.append((record, False))
    return results


_WORKER_MODULES: tuple | None = None


def _initialize_worker(args: Any) -> None:
    global _WORKER_MODULES
    _WORKER_MODULES = load_author_modules(Path(args.author_code_dir), scattering_backend=args.scattering_backend)


def _worker_job(args: Any, jobs: tuple[ObservationJob, ...]) -> list[tuple[dict, bool]]:
    assert _WORKER_MODULES is not None
    return generate_particle_job(args, jobs, _WORKER_MODULES)


def execute_jobs(args: Any) -> Iterator[tuple[dict, bool]]:
    jobs = particle_jobs(args)
    if args.workers == 1:
        modules = load_author_modules(Path(args.author_code_dir), scattering_backend=args.scattering_backend)
        for group in jobs:
            yield from generate_particle_job(args, group, modules)
        return
    variables = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    previous = {name: os.environ.get(name) for name in variables}
    # Set before spawn/import so every numeric runtime sees the per-worker limit.
    os.environ.update({name: str(args.worker_threads) for name in variables})
    try:
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_worker, initargs=(args,),
        ) as pool:
            pending = set()
            try:
                for _ in range(2 * args.workers):
                    group = next(jobs, None)
                    if group is not None:
                        pending.add(pool.submit(_worker_job, args, group))
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        yield from future.result()
                        group = next(jobs, None)
                        if group is not None:
                            pending.add(pool.submit(_worker_job, args, group))
            finally:
                for future in pending:
                    future.cancel()
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def dataset_lock(output: Path):
    """OS-held lock; a crash releases ownership without deleting the lock inode."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".generation.lock").open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Another generator is writing to {output}; stop it before resuming.") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)
