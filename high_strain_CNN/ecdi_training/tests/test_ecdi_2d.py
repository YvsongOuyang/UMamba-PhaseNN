"""e-CDI HDF5 contracts, worker safety, noisy/clean metrics and pipeline order."""

import argparse
import copy
import json
import pickle
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from ecdi_training.data import build_dataset, prepare_batch
from ecdi_training.ecdi_data import inspect_manifest
from ecdi_training.evaluate import metrics_2d, aligned_objects_2d
from ecdi_training.run_pipeline import pipeline_commands
from ecdi_training.train import build_loader, run_epoch
from ecdi_training.model import ComplexMoMambaRefiner2D, HighStrainMoMambaCascade2D
from ecdi_training.validate_data import validate
from ecdi_training.visualize import render
from pytorch_autophasenn.reconstruction import realspace_from_modulus_phase


@pytest.fixture
def ecdi_config(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs" / "ecdi_2d.json").read_text(encoding="utf-8"))
    config["root"] = str(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "manifests").mkdir()
    (tmp_path / "quality").mkdir()
    config_hash = "synthetic_config_hash"
    coordinate = np.arange(512) - 256
    yy, xx = np.meshgrid(coordinate, coordinate, indexing="ij")
    rng = np.random.default_rng(7)
    targets = [
        np.exp(-((xx - i) ** 2 + yy**2) / (2 * 35**2))
        * np.exp(1j * (0.01 * xx + 0.02 * yy + i))
        for i in range(6)
    ]
    targets = np.stack(targets).astype(np.complex64)
    intensity = (
        np.abs(
            np.fft.fftshift(
                np.fft.fft2(np.fft.ifftshift(targets, axes=(-2, -1))), axes=(-2, -1)
            )
        )
        ** 2
    )
    intensity = (intensity / intensity.sum(axis=(-2, -1), keepdims=True)).astype(
        np.float32
    )
    counts = rng.poisson(intensity * 50000).astype(np.float32)
    path = tmp_path / "data" / "ecdi_10k.h5"
    with h5py.File(path, "w") as h5:
        for key, value in {
            "observed_counts": counts,
            "exit_wave_real": targets.real,
            "exit_wave_imag": targets.imag,
            "ideal_intensity": intensity,
            "support": np.ones_like(intensity, dtype=bool),
            "valid": np.ones(6, dtype=bool),
        }.items():
            h5.create_dataset(key, data=value)
        h5.create_dataset(
            "metadata_json",
            data=[
                json.dumps({"structure": {"source_name": "Si" if i % 2 else "Li"}})
                for i in range(6)
            ],
            dtype=h5py.string_dtype(),
        )
        h5.attrs["config_sha256"] = config_hash
        h5.attrs["config_json"] = json.dumps(
            {
                "physics": {
                    "imaging_mode": "single_shot_ecdi",
                    "frozen_phonon_configs": 1,
                },
                "ecdi": {"sample_geometry": "extended_crystal"},
            }
        )
    manifest = {
        "dataset_path": "..\\data\\ecdi_10k.h5",
        "config_sha256": config_hash,
        "sample_count": 6,
        "splits": {"train": [3, 0], "val": [5, 1], "test": [4, 2]},
        "counts": {"train": 2, "val": 2, "test": 2},
    }
    (tmp_path / config["manifest"]).write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / config["quality_report"]).write_text(
        json.dumps(
            {
                "passed": True,
                "sample_count": 6,
                "config_sha256": config_hash,
                "failures": [],
            }
        ),
        encoding="utf-8",
    )
    return config


def test_hdf5_reading_and_preflight(ecdi_config):
    dataset = build_dataset(ecdi_config, "train", include_ideal=True)
    sample = dataset[0]
    assert sample["index"] == 3 and sample["material"] == "Si"
    assert "support" not in sample
    assert torch.is_complex(sample["realspace"])
    assert "ideal_intensity" not in build_dataset(ecdi_config, "train")[0]
    report = validate(ecdi_config)
    assert (
        report["passed"]
        and report["splits"]["train"]["max_clean_fourier_intensity_nrmse"] < 1e-5
    )
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored._handle is None
    torch.testing.assert_close(restored[0]["diffraction"], sample["diffraction"])
    dataset.close()
    restored.close()


def test_worker_handles_do_not_share_parent_file(ecdi_config):
    dataset = build_dataset(ecdi_config, "train")
    assert dataset._handle is None
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=2, multiprocessing_context="spawn"
    )
    indices = [int(batch["index"][0]) for batch in loader]
    assert indices == [3, 0] and dataset._handle is None


def test_reject_overlap_hash_and_incomplete_data(ecdi_config):
    path = Path(ecdi_config["root"]) / ecdi_config["manifest"]
    original = path.read_text(encoding="utf-8")
    manifest = json.loads(original)
    manifest["splits"]["val"][0] = 3
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        inspect_manifest(ecdi_config)
    path.write_text(original, encoding="utf-8")
    h5path = Path(ecdi_config["root"]) / "data" / "ecdi_10k.h5"
    with h5py.File(h5path, "r+") as h5:
        h5.attrs["config_sha256"] = "wrong"
    with pytest.raises(ValueError, match="config_sha256"):
        inspect_manifest(ecdi_config)
    with h5py.File(h5path, "r+") as h5:
        h5.attrs["config_sha256"] = "synthetic_config_hash"
        h5["valid"][0] = False
    with pytest.raises(ValueError, match="incomplete"):
        inspect_manifest(ecdi_config)


def test_clean_metrics_distinguish_denoising_from_measured_consistency(
    ecdi_config, tmp_path
):
    torch.set_num_threads(4)
    dataset = build_dataset(ecdi_config, "test", include_ideal=True)
    raw = next(iter(build_loader(dataset, 1, 0, False, torch.device("cpu"))))
    batch = prepare_batch(raw, ecdi_config, torch.device("cpu"))
    initial = realspace_from_modulus_phase(
        batch["diffraction"], batch["target_phase"], spatial_dims=2
    )
    noisy_metrics = metrics_2d(
        initial,
        batch["realspace"],
        batch["diffraction"],
        batch["input"],
        ideal_intensity=batch["ideal_intensity"],
    )
    clean_metrics = metrics_2d(
        batch["realspace"],
        batch["realspace"],
        batch["diffraction"],
        batch["input"],
        ideal_intensity=batch["ideal_intensity"],
    )
    assert noisy_metrics["paper_modulus_mae"] < 1e-6
    assert clean_metrics["clean_intensity_nrmse"] < 1e-5
    assert clean_metrics["paper_modulus_mae"] > noisy_metrics["paper_modulus_mae"]
    sample_path = tmp_path / "sample_00004.npz"
    np.savez_compressed(
        sample_path,
        sample=4,
        material="Li",
        target=batch["realspace"][0, 0].numpy(),
        initial=initial[0, 0].numpy(),
        refined=batch["realspace"][0, 0].numpy(),
        measured_modulus=batch["diffraction"][0, 0].numpy(),
    )
    figure = render(sample_path, tmp_path / "comparison.png")
    assert figure.stat().st_size > 10000
    dataset.close()


def test_pipeline_order_and_skip_to_refiner(tmp_path):
    args = argparse.Namespace(
        data_config="ecdi_training/configs/ecdi_2d.json",
        checkpoint_root=str(tmp_path),
        run_name="test",
        data_dir="",
        num_workers=4,
        device="cuda",
        phase_batch_size=1,
        refiner_batch_size=1,
        phase_epochs=None,
        refiner_epochs=None,
        phase_resume="",
        refiner_resume="",
        start_stage="phase",
        max_batches_per_epoch=0,
        num_samples_train=0,
        num_samples_val=0,
        num_samples_test=0,
    )
    stages = pipeline_commands(args, tmp_path)
    assert [name for name, command in stages] == [
        "validate",
        "phase",
        "refiner",
        "evaluate",
        "visualize",
    ]
    assert "--phase-checkpoint" in stages[2][1] and "--split" in stages[3][1]
    args.start_stage = "refiner"
    assert [name for name, command in pipeline_commands(args, tmp_path)] == [
        "validate",
        "refiner",
        "evaluate",
        "visualize",
    ]


def test_twin_alignment_keeps_a_common_target_orientation():
    target = torch.zeros(1, 1, 32, 32, dtype=torch.complex64)
    phase = torch.arange(25, dtype=torch.float32).reshape(5, 5) * 0.17
    target[:, :, 14:19, 14:19] = torch.exp(1j * phase)
    twin = torch.conj(torch.roll(torch.flip(target, (-2, -1)), (1, 1), (-2, -1)))
    prediction, truth = aligned_objects_2d(twin * torch.exp(torch.tensor(0.7j)), target)
    torch.testing.assert_close(prediction, truth, atol=2e-5, rtol=2e-5)


def test_hdf5_refiner_epoch_freezes_phase_weights_and_bn(ecdi_config):
    torch.set_num_threads(4)
    dataset = build_dataset(ecdi_config, "train", 1)
    device = torch.device("cpu")
    loader = build_loader(dataset, 1, 0, True, device)
    phase = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1), torch.nn.BatchNorm2d(1))
    refiner = ComplexMoMambaRefiner2D(
        base_channels=1,
        num_experts=(1, 1, 1, 1),
        top_k=1,
        router_dropout=0,
        position_dropout=0,
        codec_dropout=0,
        mixer_factory=lambda d_model, **kwargs: torch.nn.Linear(
            d_model, d_model, bias=False
        ),
    )
    model = HighStrainMoMambaCascade2D(phase, refiner)
    before = {name: tensor.clone() for name, tensor in phase.state_dict().items()}
    optimizer = torch.optim.Adam(refiner.parameters(), lr=2e-4)
    result = run_epoch(
        model,
        loader,
        config=ecdi_config,
        stage="refiner",
        device=device,
        optimizer=optimizer,
    )
    assert np.isfinite(result["loss"]) and result["samples"] == 1
    assert all(
        torch.equal(before[name], tensor) for name, tensor in phase.state_dict().items()
    )
    assert refiner.output.real_kernel.weight.abs().sum() > 0
    dataset.close()
