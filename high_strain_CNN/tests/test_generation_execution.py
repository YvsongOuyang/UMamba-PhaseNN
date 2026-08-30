from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from simulation.author_generator import (
    AUTHOR_GENERATOR_PROTOCOL, AUTHOR_PHASE_SAMPLING, DEFAULT_AUTHOR_CODE_DIR,
    AuthorGeneratedSample, file_sha256, paper_category_for_index, save_author_sample,
)
from simulation.generate_author_dataset import DATA_ARGUMENTS, format_duration, parse_args
from simulation.generation_execution import (
    dataset_lock, generate_particle_job, particle_jobs, read_existing_sample, sample_schedule,
)


def arguments(tmp_path, **overrides):
    values = dict(
        author_code_dir=str(DEFAULT_AUTHOR_CODE_DIR), output_dir=str(tmp_path), profile="paper",
        seed=20260830, num_samples=12, split_counts=[5, 4, 3], storage="compact",
        scattering_backend="compat", observations_per_particle=3, oversampling_policy="record",
        category_sampling="random", random_q_rotation=True, save_extras=True,
        workers=1, worker_threads=1, print_freq=50, resume_from=None,
    )
    return Namespace(**{**values, **overrides})


def stored_sample(job, args):
    metadata = dict(
        profile="paper", seed=args.seed + job.index, split=job.split,
        scattering_backend=args.scattering_backend, shape=job.shape, strain_argument=job.strain,
        particle_index=job.particle_index, particle_seed=args.seed + 1_000_000 + job.particle_index,
        observation_index=job.observation_index, generator_protocol=AUTHOR_GENERATOR_PROTOCOL,
        phase_sampling=AUTHOR_PHASE_SAMPLING, random_q_rotation=args.random_q_rotation,
        oversampling_policy=args.oversampling_policy,
    )
    return AuthorGeneratedSample(
        intensity=np.ones((64,) * 3, dtype=np.float32),
        reciprocal_phase=np.zeros((64,) * 3, dtype=np.float32), metadata=metadata,
        realspace_object=np.ones((64,) * 3, dtype=np.complex64),
        support=np.ones((64,) * 3, dtype=bool), clean_intensity=np.ones((64,) * 3, dtype=np.float32),
    )


def test_category_sequence_matches_old_serial_loop(tmp_path):
    args = arguments(tmp_path)
    jobs = [job for group in particle_jobs(args) for job in group]
    rng, shapes = random.Random(args.seed + 7919), {}
    for job, (index, split, category) in zip(jobs, sample_schedule(12, 3, [5, 4, 3])):
        expected = paper_category_for_index(category, 3, category_sampling="random", rng=rng, random_shapes=shapes)
        assert (job.index, job.split) == (index, split)
        assert (job.particle_index, job.observation_index, job.shape, job.strain) == expected
    assert [len(group) for group in particle_jobs(args)] == [3, 2, 3, 1, 3]


def test_resume_config_preserves_data_settings(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    config = tmp_path / "config.json"
    previous = {key: getattr(args, key) for key in DATA_ARGUMENTS}
    previous["random_q_rotation"] = None  # Compatible with older configs.
    config.write_text(json.dumps({"args": previous}))
    monkeypatch.setattr(sys, "argv", ["generate", "--resume-from", str(config), "--workers", "4"])
    resumed = parse_args()
    assert resumed.workers == 4 and resumed.num_samples == 12 and resumed.random_q_rotation
    assert resumed.output_dir == str(tmp_path)
    for flag in ("--seed", "--num-samples", "--observations-per-particle"):
        monkeypatch.setattr(sys, "argv", ["generate", "--resume-from", str(config), flag, "1"])
        with pytest.raises(SystemExit):
            parse_args()


@pytest.mark.parametrize("storage", ["compact", "standard"])
def test_reuse_checks_arrays_and_metadata(tmp_path, storage):
    args = arguments(tmp_path, storage=storage, resume_from="previous/config.json")
    job = next(particle_jobs(args))[0]
    path = tmp_path / "sample_00000.npz"
    sample = stored_sample(job, args)
    save_author_sample(path, sample, storage=storage)
    record = read_existing_sample(path, job, args)
    assert record["sha256"] == file_sha256(path)
    sample.metadata["seed"] += 1
    save_author_sample(path, sample, storage=storage)
    with pytest.raises(ValueError, match="metadata seed"):
        read_existing_sample(path, job, args)
    path.write_bytes(b"interrupted file")
    with pytest.raises(ValueError, match="No existing sample was overwritten"):
        read_existing_sample(path, job, args)


def test_partial_particle_resume_keeps_existing_bytes(tmp_path, monkeypatch):
    import simulation.generation_execution as execution
    args = arguments(tmp_path, resume_from="previous/config.json")
    group = next(particle_jobs(args))
    path = tmp_path / "sample_00000.npz"
    save_author_sample(path, stored_sample(group[0], args), storage="compact")
    original = path.read_bytes()
    calls = []
    monkeypatch.setattr(execution, "create_paper_particle", lambda *a, **kw: calls.append("particle"))
    def observation(modules, particle, seed, strain, index, **kwargs):
        calls.append(seed)
        return stored_sample(group[index], args)
    monkeypatch.setattr(execution, "generate_paper_observation", observation)
    records = generate_particle_job(args, group, (None,) * 3)
    assert calls == ["particle", args.seed + 1, args.seed + 2]
    assert [reused for _, reused in records] == [True, False, False]
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp.npz"))
    calls.clear()
    assert all(reused for _, reused in generate_particle_job(args, group, (None,) * 3))
    assert not calls


def test_dataset_lock_released_after_failure(tmp_path):
    with dataset_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another generator"):
            with dataset_lock(tmp_path):
                pass
    with pytest.raises(ValueError):
        with dataset_lock(tmp_path):
            raise ValueError("test")
    with dataset_lock(tmp_path):
        pass


def test_atomic_sample_not_published_after_write_failure(tmp_path, monkeypatch):
    import simulation.generation_execution as execution
    args = arguments(tmp_path)
    group = next(particle_jobs(args))
    monkeypatch.setattr(execution, "create_paper_particle", lambda *a, **kw: None)
    monkeypatch.setattr(execution, "generate_paper_observation", lambda *a, **kw: stored_sample(group[0], args))
    def fail(path, *args, **kwargs):
        path.write_bytes(b"partial")
        raise OSError("disk full")
    monkeypatch.setattr(execution, "save_author_sample", fail)
    with pytest.raises(OSError, match="disk full"):
        generate_particle_job(args, group, (None,) * 3)
    assert not list(tmp_path.iterdir())


def test_duration_is_human_readable():
    assert format_duration(106598) == "29:36:38"


def test_serial_and_spawn_generation_and_resume(tmp_path):
    """Real author calls, all nine categories, with a split cutting a particle short."""
    project = Path(__file__).resolve().parents[1]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    outputs = []
    configs = []
    for workers in (1, 4):
        output, log = tmp_path / f"data{workers}", tmp_path / f"logs{workers}"
        command = [
            sys.executable, "-B", "-m", "simulation.generate_author_dataset",
            "--output-dir", str(output), "--log-dir", str(log), "--split-counts", "9", "2", "1",
            "--storage", "compact", "--oversampling-policy", "record", "--category-sampling", "balanced",
            "--seed", "20260830", "--workers", str(workers), "--print-freq", "5",
        ]
        result = subprocess.run(command, cwd=project, env=env, capture_output=True, text=True, timeout=240)
        assert result.returncode == 0, result.stdout + result.stderr
        manifest = json.loads((output / "dataset_manifest.json").read_text())
        assert [record["index"] for record in manifest["samples"]] == list(range(12))
        assert manifest["execution"]["workers"] == workers
        assert "scattering=" in (log / "generation.log").read_text()
        outputs.append(output)
        configs.append(log / "config.json")
    for index in range(12):
        with np.load(outputs[0] / f"sample_{index:05d}.npz") as serial, np.load(outputs[1] / f"sample_{index:05d}.npz") as parallel:
            np.testing.assert_array_equal(serial["I"], parallel["I"])
            np.testing.assert_array_equal(serial["support"], parallel["support"])
            np.testing.assert_allclose(serial["object"], parallel["object"], rtol=2e-6, atol=2e-6)
            metadata = [json.loads(str(item["metadata_json"])) for item in (serial, parallel)]
            for item in metadata:
                item.pop("generation_seconds")
                item.pop("stage_seconds")
            assert metadata[0] == metadata[1]
    output = outputs[1]
    preserved = (output / "sample_00000.npz").read_bytes()
    # Mimic an interruption leaving holes and part of a particle already complete.
    (output / "dataset_manifest.json").unlink()
    for index in (1, 8, 11):
        (output / f"sample_{index:05d}.npz").unlink()
    result = subprocess.run([
        sys.executable, "-B", "-m", "simulation.generate_author_dataset",
        "--resume-from", str(configs[1]), "--log-dir", str(tmp_path / "resumed"), "--workers", "2",
    ], cwd=project, env=env, capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr
    assert preserved == (output / "sample_00000.npz").read_bytes()
    manifest = json.loads((output / "dataset_manifest.json").read_text())
    assert manifest["execution"]["reused"] == 9 and manifest["execution"]["generated"] == 3
    for index in (1, 8, 11):
        with np.load(outputs[0] / f"sample_{index:05d}.npz") as expected, np.load(output / f"sample_{index:05d}.npz") as actual:
            np.testing.assert_array_equal(expected["I"], actual["I"])
