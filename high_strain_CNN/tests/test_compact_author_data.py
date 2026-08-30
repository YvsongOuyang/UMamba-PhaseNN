from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from simulation.author_generator import AuthorGeneratedSample, save_author_sample
from simulation.generate_author_dataset import sample_schedule
from simulation.sample_io import COMPACT_STORAGE, load_reciprocal_phase
from pytorch_autophasenn.author_data import AuthorNPZPhaseDataset, prepare_author_training_sample
from pytorch_autophasenn.losses import phase_retrieval_wca_loss
from pytorch_autophasenn.train import build_loader, run_epoch


def make_sample(seed=123, size=8):
    rng = np.random.default_rng(seed)
    support = np.zeros((size,) * 3, dtype=bool)
    support[1:size // 2, 2:size // 2 + 1, 1:size // 2] = True
    obj = support * np.exp(1j * rng.normal(size=support.shape))
    reciprocal = np.fft.ifftshift(np.fft.fftn(np.fft.fftshift(obj)))
    clean = np.abs(reciprocal)**2
    intensity = rng.poisson(clean / clean.max() * 10000).astype(np.float32)
    return AuthorGeneratedSample(
        intensity=intensity, reciprocal_phase=np.angle(reciprocal).astype(np.float32),
        realspace_object=obj.astype(np.complex64), support=support,
        clean_intensity=clean.astype(np.float32), metadata={"seed": seed},
    )


def make_dataset(root: Path, counts=(8, 5, 3), storage="compact"):
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for index, split, category_index in sample_schedule(sum(counts), 3, counts):
        name = f"sample_{index:05d}.npz"
        save_author_sample(root / name, make_sample(index), storage=storage)
        records.append({
            "filename": name, "split": split,
            "metadata": {"particle_seed": 1000 + category_index // 3, "shape": "wulff"},
        })
    manifest = {
        "route": "author_generator", "split_unit": "particle", "num_samples": sum(counts),
        "splits": dict(zip(("train", "val", "test"), counts)), "samples": records,
        "storage_schema": COMPACT_STORAGE if storage == "compact" else "author_standard",
    }
    (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


@pytest.mark.parametrize("size", [7, 8])
def test_compact_phase_and_wca_match_standard(tmp_path, size):
    sample = make_sample(size=size)
    paths = [tmp_path / "standard.npz", tmp_path / "compact.npz"]
    for path, storage in zip(paths, ("standard", "compact")):
        save_author_sample(path, sample, storage=storage)
    inputs, phases = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as stored:
            item = prepare_author_training_sample(stored, input_log_data=True, shape=sample.intensity.shape)
            inputs.append(item["input"])
            phases.append(item["target_phase"])
            np.testing.assert_array_equal(stored["I"], sample.intensity)
    with np.load(paths[1], allow_pickle=False) as compact:
        assert set(compact.files) == {"I", "object", "support", "metadata_json"}
        assert json.loads(str(compact["metadata_json"]))["storage_schema"] == COMPACT_STORAGE
    assert torch.equal(inputs[0], inputs[1])
    delta = torch.angle(torch.exp(1j * (phases[0] - phases[1])))
    assert delta.abs().max() < 2e-5
    pred = torch.randn((1, size, size, size), generator=torch.Generator().manual_seed(1))
    losses = [phase_retrieval_wca_loss(pred, phase[None], inputs[0]) for phase in phases]
    torch.testing.assert_close(losses[0], losses[1], atol=3e-7, rtol=0)


def test_standard_phi_is_not_replaced_by_object_fft():
    sample = make_sample()
    phi = sample.reciprocal_phase + 0.25
    np.testing.assert_array_equal(load_reciprocal_phase({"phi": phi, "object": sample.realspace_object}), phi)


def test_shared_evaluator_reads_compact_truth(tmp_path):
    from simulation.evaluate_paper_model import load_sample
    sample = make_sample()
    path = tmp_path / "compact.npz"
    save_author_sample(path, sample, storage="compact")
    loaded = load_sample(path)
    np.testing.assert_array_equal(loaded["target_object"], sample.realspace_object)
    np.testing.assert_array_equal(loaded["target_support"], sample.support)
    with np.load(path, allow_pickle=False) as stored:
        training = prepare_author_training_sample(stored, input_log_data=True, shape=(8,) * 3)
    np.testing.assert_array_equal(loaded["target_phase"], training["target_phase"].numpy())


def test_compact_rejects_missing_truth(tmp_path):
    sample = AuthorGeneratedSample(np.ones((8,) * 3), np.zeros((8,) * 3), {})
    with pytest.raises(ValueError, match="clean object"):
        save_author_sample(tmp_path / "bad.npz", sample, storage="compact")


@pytest.mark.parametrize("bad", [{}, {"object": np.ones((8,) * 3)}, {"object": np.full((8,) * 3, complex(np.nan))}])
def test_invalid_phase_source_is_rejected(bad):
    with pytest.raises(ValueError):
        load_reciprocal_phase(bad)


def test_constant_or_negative_intensity_rejected():
    for intensity in (np.ones((8,) * 3), -np.ones((8,) * 3)):
        with pytest.raises(ValueError):
            prepare_author_training_sample({"I": intensity, "phi": np.zeros((8,) * 3)}, input_log_data=True, shape=(8,) * 3)


def test_source_intensity_is_not_squared():
    sample = make_sample()
    prepared = prepare_author_training_sample({"I": sample.intensity, "phi": sample.reciprocal_phase}, input_log_data=True, shape=(8,) * 3)
    expected = np.log1p(sample.intensity)
    expected = (expected - expected.min()) / (expected.max() - expected.min())
    np.testing.assert_array_equal(prepared["input"][0].numpy(), expected)


def test_partial_particles_do_not_cross_splits():
    schedule = list(sample_schedule(12, 3, [5, 4, 3]))
    groups = {split: {category // 3 for _, s, category in schedule if s == split} for split in ("train", "val", "test")}
    assert groups == {"train": {0, 1}, "val": {2, 3}, "test": {4}}
    assert list(sample_schedule(9, 3, None)) == [(i, None, i) for i in range(9)]


@pytest.mark.parametrize("storage", ["standard", "compact"])
def test_dataset_repeatable_and_partial_validation_included(tmp_path, storage):
    make_dataset(tmp_path, storage=storage)
    dataset = AuthorNPZPhaseDataset(tmp_path, "val", shape=(8,) * 3)
    assert len(dataset) == 5
    assert torch.equal(dataset[1]["input"], dataset[1]["input"])
    assert torch.equal(dataset[1]["target_phase"], dataset[1]["target_phase"])
    args = Namespace(batch_size=2, num_workers=0, prefetch_factor=2, data_format="author_npz")
    loader = build_loader(dataset, args, torch.device("cpu"), training=False)
    assert sum(batch["input"].shape[0] for batch in loader) == 5
    assert loader.prefetch_factor is None


@pytest.mark.parametrize("defect", ["leak", "count", "duplicate", "traversal", "missing"])
def test_bad_manifests_are_rejected(tmp_path, defect):
    manifest = make_dataset(tmp_path)
    if defect == "leak":
        manifest["samples"][8]["metadata"] = manifest["samples"][0]["metadata"]
    elif defect == "count":
        manifest["splits"]["train"] += 1
    elif defect == "duplicate":
        manifest["samples"][1]["filename"] = manifest["samples"][0]["filename"]
    elif defect == "traversal":
        manifest["samples"][0]["filename"] = "../outside.npz"
    else:
        manifest["samples"][0]["filename"] = "missing.npz"
    (tmp_path / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        AuthorNPZPhaseDataset(tmp_path, "train", shape=(8,) * 3)


def test_four_spawn_workers_two_epochs(tmp_path):
    make_dataset(tmp_path)
    dataset = AuthorNPZPhaseDataset(tmp_path, "val", shape=(8,) * 3)
    args = Namespace(batch_size=2, num_workers=4, prefetch_factor=2, data_format="author_npz")
    loader = build_loader(dataset, args, torch.device("cpu"), training=False)
    assert loader.persistent_workers and loader.num_workers == 4
    try:
        epochs = [list(loader), list(loader)]
        for batches in epochs:
            names = [name for batch in batches for name in batch["name"]]
            assert names == list(dataset.filenames)
            phases = torch.cat([batch["target_phase"] for batch in batches])
            torch.testing.assert_close(phases, torch.stack([dataset[i]["target_phase"] for i in range(len(dataset))]), rtol=0, atol=0)
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()


def test_train_and_val_use_author_phase_without_com_correction(tmp_path, monkeypatch):
    import pytorch_autophasenn.train as train
    make_dataset(tmp_path)
    dataset = AuthorNPZPhaseDataset(tmp_path, "val", shape=(8,) * 3)
    args = Namespace(batch_size=2, num_workers=0, prefetch_factor=2, data_format="author_npz")
    loader = build_loader(dataset, args, torch.device("cpu"), training=False)
    def forbidden(*args):
        raise AssertionError("AutoPhaseNN recentering must not run on author data")
    monkeypatch.setattr(train, "reciprocal_phase_from_realspace", forbidden)
    model = torch.nn.Conv3d(1, 1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    old_weight = model.weight.detach().clone()
    trained = run_epoch(model, loader, torch.device("cpu"), 1, 1, 50, 0, optimizer=optimizer)
    validated = run_epoch(model, loader, torch.device("cpu"), 1, 1, 50, 0)
    assert trained["samples"] == validated["samples"] == 5
    assert np.isfinite(trained["loss"]) and np.isfinite(validated["loss"])
    assert not torch.equal(model.weight.detach(), old_weight)
    assert not model.training
    assert validated["data_wait_seconds"] >= 0


def test_native_backend_dispatch_and_cuda_preflight(monkeypatch):
    import simulation.author_generator as module
    calls = []
    fake_cuda = SimpleNamespace(init=lambda: None, Device=SimpleNamespace(count=lambda: 1))
    fake_fhkl = SimpleNamespace(Fhkl_thread=lambda *a, **kw: calls.append(kw))
    def import_fake(name):
        return fake_cuda if name == "pycuda.driver" else fake_fhkl
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module.importlib, "import_module", import_fake)
    module._require_pynx_cuda()
    assert calls == [{"gpu_name": "", "language": "cuda"}]
    fake_cuda.Device.count = lambda: 0
    with pytest.raises(RuntimeError, match="preflight failed"):
        module._require_pynx_cuda()


def test_native_backend_fails_on_windows(monkeypatch):
    import simulation.author_generator as module
    monkeypatch.setattr(module.sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="Linux CUDA"):
        module._require_pynx_cuda()


def test_cli_defaults_and_routes(monkeypatch):
    from pytorch_autophasenn.train import parse_args
    monkeypatch.setattr(sys, "argv", ["train"])
    args = parse_args()
    assert args.num_workers == 4 and args.num_samples_train == 25000
    assert args.data_format == "autophasenn" and not args.fp16
    monkeypatch.setattr(sys, "argv", ["train", "--data-format", "author_npz", "--save-every", "0"])
    args = parse_args()
    assert args.num_samples_train is None and args.num_samples_val is None
    assert Path(args.runs_dir).name == "pytorch_simulation"
    assert args.save_every == 0


def test_generation_cli_exact_split_counts(monkeypatch):
    from simulation.generate_author_dataset import parse_args
    monkeypatch.setattr(sys, "argv", [
        "generate", "--author-code-dir", "unused", "--storage", "compact",
        "--split-counts", "95000", "4000", "3000", "--scattering-backend", "pynx_cuda",
    ])
    args = parse_args()
    assert args.num_samples == 102000 and args.storage == "compact"
    assert args.scattering_backend == "pynx_cuda"
    assert args.category_sampling == "random"
