"""Generation parity, reproducibility, resumability, integrity and reader contracts."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from bcdi_training.generate_dataset import (
    DEFAULT_CONFIG,
    generate,
    specification,
    layout,
    dataset_lock,
)
from bcdi_training.prepare_data import sha256_file
from bcdi_training.validate_data import validate
from ecdi_training.data import load_config


@pytest.fixture
def small_config():
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["num_seeds"] = 3
    config["seed_start"] = 11
    config["simulation"]["samples_per_axis"] = 1
    return config


@pytest.fixture
def generated(tmp_path, small_config):
    output = tmp_path / "dataset"
    report = generate(output, small_config, workers=1, log_dir=tmp_path / "logs")
    return output, report


def test_default_layout_has_16000_2000_2000_without_allocating_data():
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    spec = specification(config)
    jobs = layout(spec)
    assert {
        split: sum(job["count"] for job in jobs.values() if job["split"] == split)
        for split in ("train", "val", "test")
    } == {"train": 16000, "val": 2000, "test": 2000}


def test_generated_dataset_passes_existing_reader(generated):
    output, report = generated
    assert report["status"] == "complete" and report["counts"] == {
        "train": 1,
        "val": 1,
        "test": 1,
    }
    result = validate(load_config(output / "training_config.json"))
    assert result["passed"] and result["seed_disjoint"]
    assert (
        max(
            value["max_fourier_modulus_relative_l2"]
            for value in result["splits"].values()
        )
        < 2e-5
    )
    assert not list(output.glob("*.npz"))


def test_parallel_and_serial_generation_are_identical(
    generated, small_config, tmp_path
):
    output, first = generated
    other = tmp_path / "parallel"
    second = generate(
        other, small_config, workers=2, log_dir=tmp_path / "parallel_logs"
    )
    assert first == second
    assert sha256_file(output / "dataset_manifest.json") == sha256_file(
        other / "dataset_manifest.json"
    )
    for split in ("train", "val", "test"):
        for suffix in ("diff", "real"):
            assert sha256_file(output / f"{split}_{suffix}.npy") == sha256_file(
                other / f"{split}_{suffix}.npy"
            )


def test_resume_repairs_only_uncommitted_seed_and_keeps_identity(
    generated, small_config, tmp_path
):
    output, report = generated
    before = sha256_file(output / "dataset_manifest.json")
    seed = report["seed_groups"]["train"][0]
    (output / "seed_records" / f"seed_{seed}.json").unlink()
    array = np.load(output / "train_diff.npy", mmap_mode="r+")
    array[:] = 0
    array.flush()
    array._mmap.close()
    config_path = output / "training_config.json"
    training = json.loads(config_path.read_text(encoding="utf-8"))
    training["training"]["phase"]["learning_rate"] = 5e-4
    config_path.write_text(json.dumps(training), encoding="utf-8")
    resumed = generate(
        output, small_config, workers=1, resume=True, log_dir=tmp_path / "resume_logs"
    )
    assert resumed == report and sha256_file(output / "dataset_manifest.json") == before
    assert load_config(config_path)["training"]["phase"]["learning_rate"] == 5e-4


def test_reject_changed_config_corrupt_completed_data_and_overwrite(
    generated, small_config, tmp_path
):
    output, _ = generated
    with pytest.raises(FileExistsError):
        generate(output, small_config, workers=1, log_dir=tmp_path / "logs")
    changed = copy.deepcopy(small_config)
    changed["seed_start"] += 1
    with pytest.raises(ValueError, match="Resume configuration"):
        generate(output, changed, resume=True, workers=1, log_dir=tmp_path / "logs")
    array = np.load(output / "val_real.npy", mmap_mode="r+")
    array[0, 0, 0] += 1
    array.flush()
    array._mmap.close()
    with pytest.raises(ValueError, match="corrupted"):
        generate(
            output, small_config, resume=True, workers=1, log_dir=tmp_path / "logs"
        )


def test_concurrent_writer_is_rejected(tmp_path, small_config):
    output = tmp_path / "locked"
    output.mkdir()
    with dataset_lock(output):
        with pytest.raises(RuntimeError, match="Another process"):
            generate(output, small_config, workers=1, log_dir=tmp_path / "logs")
