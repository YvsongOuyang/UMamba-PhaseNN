"""BCDI phase conventions, source integrity, grouped splits and training glue."""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from bcdi_training import prepare_data, train, evaluate
from bcdi_training._entry import check_arguments
from bcdi_training.validate_data import validate, verify_manifest
from bcdi_training.run_pipeline import pipeline_commands
from ecdi_training.data import load_config, build_dataset
from ecdi_training.train import build_loader, make_model, run_epoch


def make_source(root, convention="author"):
    root.mkdir()
    (root / "synthetic_data").mkdir()
    rng = np.random.default_rng(8)
    y, x = np.mgrid[:512, :512] - 256
    records = []
    for seed in range(3):
        amplitude = np.exp(-(x**2 / 90**2 + y**2 / 140**2))[None]
        phase = (0.003 * x + 0.006 * y + seed * 0.1 + rng.normal(0, 0.01, x.shape))[
            None
        ]
        field = amplitude.astype(np.float32) * np.exp(
            1j * (2 * np.pi if convention == "author" else 1) * phase.astype(np.float32)
        )
        measured = np.abs(
            np.fft.fftshift(
                np.fft.fft2(np.fft.ifftshift(field, axes=(-2, -1))), axes=(-2, -1)
            )
        ).astype(np.float32)
        arrays = {
            "object_modulus": amplitude,
            "object_phase": phase,
            "modulus_detector": measured,
        }
        path = root / "synthetic_data" / f"data{seed}.npz"
        np.savez(path, **arrays)
        records.append(
            {
                "seed": seed,
                "file": f"synthetic_data/{path.name}",
                "bytes": path.stat().st_size,
                "sha256": prepare_data.sha256_file(path),
                "arrays": {
                    key: {"shape": list(value.shape)} for key, value in arrays.items()
                },
            }
        )
    manifest = {
        "status": "complete",
        "config": {"image_size": 512, "phase_convention": convention},
        "samples": 3,
        "files": records,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


@pytest.fixture
def prepared(tmp_path):
    source = make_source(tmp_path / "source")
    output = tmp_path / "prepared"
    prepare_data.prepare(source, output)
    return source, output, load_config(output / "training_config.json")


@pytest.mark.parametrize("convention", ["author", "radians"])
def test_phase_conventions_and_seed_disjointness(tmp_path, convention):
    source = make_source(tmp_path / "source", convention)
    output = tmp_path / "prepared"
    report = prepare_data.prepare(source, output)
    config = load_config(output / "training_config.json")
    assert report["counts"] == {"train": 1, "val": 1, "test": 1}
    assert report["max_fourier_modulus_relative_l2"] < 2e-6
    checked = validate(config)
    assert checked["passed"] and checked["seed_disjoint"]
    groups = report["seed_groups"]
    assert set(groups["train"]).isdisjoint(groups["test"])
    with np.load(source / report["splits"]["train"][0]["source_file"]) as data:
        converted = np.load(output / "train_real.npy")
        expected = data["object_modulus"].astype(np.float32) * np.exp(
            1j
            * (2 * np.pi if convention == "author" else 1)
            * data["object_phase"].astype(np.float32)
        )
        np.testing.assert_allclose(converted, expected, atol=1e-6)
        if convention == "author":
            wrong = data["object_modulus"] * np.exp(1j * data["object_phase"])
            assert np.linalg.norm(converted - wrong) / np.linalg.norm(converted) > 0.1


def test_reject_mislabeled_convention_and_overwrite(tmp_path, prepared):
    source = make_source(tmp_path / "bad_source", "author")
    path = source / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["config"]["phase_convention"] = "radians"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Fourier pairing failed"):
        prepare_data.prepare(source, tmp_path / "bad_prepared")
    source, output, _ = prepared
    with pytest.raises(FileExistsError):
        prepare_data.prepare(source, output)
    with pytest.raises(ValueError, match="three"):
        prepare_data.partition_seeds([0, 1])


def test_reject_changed_source_and_prepared_arrays(prepared):
    source, output, config = prepared
    path = output / "test_diff.npy"
    array = np.load(path, mmap_mode="r+")
    array[0, 0, 0] += 1
    array.flush()
    array._mmap.close()
    with pytest.raises(ValueError, match="content hash"):
        verify_manifest(config, check_hashes=True)
    with (source / "synthetic_data" / "data0.npz").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="Source size/SHA256"):
        prepare_data.inspect_source(source)


def test_reject_checkpoint_from_different_data(prepared, tmp_path):
    _, output, _ = prepared
    path = tmp_path / "wrong.pt"
    torch.save(
        {"spatial_dims": 2, "config": {"provenance": {"manifest_sha256": "wrong"}}},
        path,
    )
    with pytest.raises(ValueError, match="different generation"):
        check_arguments(
            [
                "--data-config",
                str(output / "training_config.json"),
                "--checkpoint",
                str(path),
            ]
        )


def test_pipeline_wires_correct_dataset_and_modules(tmp_path):
    args = argparse.Namespace(
        data_config=str(tmp_path / "data" / "training_config.json"),
        data_dir="",
        run_name="bcdi_test",
        checkpoint_root=str(tmp_path / "weights"),
        num_workers=4,
        device="cuda",
        start_stage="phase",
        phase_batch_size=1,
        refiner_batch_size=1,
        phase_epochs=1,
        refiner_epochs=1,
        phase_resume="",
        refiner_resume="",
        max_batches_per_epoch=1,
        num_samples_train=1,
        num_samples_val=1,
        num_samples_test=1,
        phase_lr=1e-3,
        refiner_lr=None,
        source_dir=str(tmp_path / "source"),
    )
    commands = pipeline_commands(args, tmp_path / "run")
    assert [stage for stage, _ in commands] == [
        "prepare",
        "validate",
        "phase",
        "refiner",
        "evaluate",
        "visualize",
    ]
    assert all(
        command[command.index("-m") + 1].startswith("bcdi_training.")
        for _, command in commands
    )
    refinement = dict(commands)["refiner"]
    assert Path(refinement[refinement.index("--phase-checkpoint") + 1]).parts[-2:] == (
        "bcdi_test_phase",
        "checkpoint_best.pt",
    )
    args.start_stage, args.source_dir = "refiner", ""
    assert "phase" not in [
        stage for stage, _ in pipeline_commands(args, tmp_path / "run")
    ]


class TestMixer(nn.Module):
    __test__ = False

    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, value):
        return value * self.scale


def test_wrapper_training_evaluation_and_second_stage_gradients(
    prepared, tmp_path, monkeypatch
):
    torch.set_num_threads(4)
    _, output, config = prepared
    args = [
        "bcdi.train",
        "--stage",
        "phase",
        "--data-config",
        str(output / "training_config.json"),
        "--epochs",
        "1",
        "--device",
        "cpu",
        "--num-workers",
        "0",
        "--run-name",
        "bcdi_smoke",
        "--runs-dir",
        str(tmp_path / "runs"),
        "--checkpoint-root",
        str(tmp_path / "weights"),
    ]
    monkeypatch.setattr(sys, "argv", args)
    assert train.main() == 0
    checkpoint_path = tmp_path / "weights" / "bcdi_smoke" / "checkpoint_best.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bcdi.evaluate",
            "--data-config",
            str(output / "training_config.json"),
            "--checkpoint",
            str(checkpoint_path),
            "--device",
            "cpu",
            "--num-workers",
            "0",
            "--output-dir",
            str(tmp_path / "evaluation"),
        ],
    )
    assert evaluate.main() == 0
    report = json.loads((tmp_path / "evaluation" / "metrics.json").read_text())
    assert report["dataset"] == "bcdi_simulator_512" and report["num_samples"] == 1
    assert report["metrics"]["initial"]["paper_modulus_mae"] < 1e-6
    assert not any("e-CDI" in note for note in report["notes"])
    source_checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    config = copy.deepcopy(config)
    config["training"].update(base_channels=1, num_experts=[1, 1, 1, 1], top_k=1)
    model = make_model(
        "refiner", config, phase_checkpoint=source_checkpoint, mixer_factory=TestMixer
    )
    loader = build_loader(
        build_dataset(config, "train"), 1, 0, True, torch.device("cpu")
    )
    optimizer = torch.optim.Adam(model.refiner.parameters(), lr=2e-4)
    stats = run_epoch(
        model,
        loader,
        config=config,
        stage="refiner",
        device=torch.device("cpu"),
        optimizer=optimizer,
    )
    assert np.isfinite(stats["loss"])
    assert all(p.grad is None for p in model.phase_model.parameters())
    assert model.refiner.output.real_kernel.weight.grad.abs().sum() > 0
