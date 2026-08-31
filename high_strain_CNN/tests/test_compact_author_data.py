from __future__ import annotations

import json
import hashlib
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
            "metadata": {
                "particle_seed": 1000 + category_index // 3, "shape": "wulff",
                "measured_object_oversampling_xyz": [4.0, 4.0, 4.0],
            },
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
    dataset = AuthorNPZPhaseDataset(tmp_path, "val", shape=(8,) * 3, min_oversampling=2)
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


def test_oversampling_filter_is_strict_and_precedes_subset_limit(tmp_path):
    manifest = make_dataset(tmp_path)
    manifest["samples"][0]["metadata"]["measured_object_oversampling_xyz"] = [3, 2, 3]
    manifest["samples"][2]["metadata"]["measured_object_oversampling_xyz"] = [3, 1.9, 3]
    manifest["samples"][8]["metadata"]["measured_object_oversampling_xyz"] = [1.9, 3, 3]
    manifest["samples"][13]["metadata"]["measured_object_oversampling_xyz"] = [3, 3, 2]
    path = tmp_path / "dataset_manifest.json"
    path.write_text(json.dumps(manifest))
    before = {name.name: hashlib.sha256(name.read_bytes()).hexdigest() for name in tmp_path.iterdir()}
    train = AuthorNPZPhaseDataset(tmp_path, "train", shape=(8,) * 3, min_oversampling=2, num_samples=3)
    assert train.filenames == ("sample_00001.npz", "sample_00003.npz", "sample_00004.npz")
    assert train.manifest["available_samples"] == 6
    assert train.manifest["unfiltered_samples"] == 8
    assert train.manifest["oversampling_filter"]["eligible_split_counts"] == {"train": 6, "val": 4, "test": 2}
    assert train.manifest["oversampling_filter"]["excluded_split_counts"] == {"train": 2, "val": 1, "test": 1}
    for split, count in (("val", 4), ("test", 2)):
        assert len(AuthorNPZPhaseDataset(tmp_path, split, shape=(8,) * 3, min_oversampling=2)) == count
    assert len(AuthorNPZPhaseDataset(tmp_path, "train", shape=(8,) * 3)) == 8
    after = {name.name: hashlib.sha256(name.read_bytes()).hexdigest() for name in tmp_path.iterdir()}
    assert before == after
    with pytest.raises(ValueError, match="6 eligible"):
        AuthorNPZPhaseDataset(tmp_path, "train", min_oversampling=2, num_samples=7)


@pytest.mark.parametrize("measured", [None, [3, 3], [3, float("nan"), 3], [3, 0, 3]])
def test_filter_rejects_unknown_oversampling(tmp_path, measured):
    manifest = make_dataset(tmp_path)
    manifest["samples"][0]["metadata"]["measured_object_oversampling_xyz"] = measured
    (tmp_path / "dataset_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="oversampling metadata"):
        AuthorNPZPhaseDataset(tmp_path, "train", min_oversampling=2)


def test_filter_does_not_hide_particle_leakage(tmp_path):
    manifest = make_dataset(tmp_path)
    manifest["samples"][8]["metadata"] = dict(manifest["samples"][0]["metadata"])
    manifest["samples"][8]["metadata"]["measured_object_oversampling_xyz"] = [1, 1, 1]
    (tmp_path / "dataset_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Particle leakage"):
        AuthorNPZPhaseDataset(tmp_path, "train", min_oversampling=2)


def test_short_epochs_reshuffle_full_pool_and_keep_small_validation_complete(tmp_path):
    make_dataset(tmp_path, counts=(24, 5, 3))
    class TrackedDataset(AuthorNPZPhaseDataset):
        def __getitem__(self, index):
            self.requested.append(index)
            return super().__getitem__(index)
    train, val = [TrackedDataset(tmp_path, split, shape=(8,) * 3, min_oversampling=2) for split in ("train", "val")]
    args = Namespace(batch_size=2, num_workers=0, prefetch_factor=2, data_format="author_npz")
    loaders = [build_loader(dataset, args, torch.device("cpu"), training=training)
               for dataset, training in ((train, True), (val, False))]
    torch.manual_seed(42)
    model = torch.nn.Conv3d(1, 1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    memberships = []
    for epoch in (1, 2):
        train.requested, val.requested = [], []
        stats = run_epoch(model, loaders[0], torch.device("cpu"), epoch, 2, 50, 3, optimizer=optimizer)
        validation = run_epoch(model, loaders[1], torch.device("cpu"), epoch, 2, 50, 3)
        assert stats["samples"] == len(train.requested) == 6
        assert len(set(train.requested)) == 6
        assert validation["samples"] == 5 and val.requested == list(range(5))
        memberships.append(set(train.requested))
    assert memberships[0] != memberships[1]


def test_training_cli_oversampling_filter_and_batch_limit(monkeypatch, tmp_path):
    from pytorch_autophasenn.train import parse_args, build_datasets
    make_dataset(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "train", "--data-format", "author_npz", "--data-dir", str(tmp_path),
        "--author-min-oversampling", "2", "--max-batches-per-epoch", "1563", "--shape", "8",
    ])
    args = parse_args()
    train, val, _ = build_datasets(args)
    assert args.max_batches_per_epoch == 1563 and args.num_samples_train == 8
    assert train.manifest["oversampling_filter"] == val.manifest["oversampling_filter"]
    monkeypatch.setattr(sys, "argv", ["train", "--author-min-oversampling", "2"])
    with pytest.raises(SystemExit):
        parse_args()


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


def test_both_author_clis_default_to_bundled_source(monkeypatch):
    from simulation.author_generator import DEFAULT_AUTHOR_CODE_DIR
    from simulation.generate_author_dataset import parse_args as generation_args
    from simulation.evaluate_author_code import parse_args as evaluation_args
    monkeypatch.setattr(sys, "argv", ["author_cli"])
    assert Path(generation_args().author_code_dir) == DEFAULT_AUTHOR_CODE_DIR
    assert Path(evaluation_args().author_code_dir) == DEFAULT_AUTHOR_CODE_DIR
    assert (DEFAULT_AUTHOR_CODE_DIR / "ShapedParticle.py").is_file()


def test_bundled_source_bytes_and_potential_manifest():
    from simulation.author_generator import DEFAULT_AUTHOR_CODE_DIR, author_source_manifest
    expected = json.loads((DEFAULT_AUTHOR_CODE_DIR.parent / "author_source_manifest.json").read_text())
    assert len(expected["files"]) == 11
    recorded = {record["name"]: record for record in author_source_manifest(DEFAULT_AUTHOR_CODE_DIR)}
    assert set(recorded) == {record["path"] for record in expected["files"]}
    for record in expected["files"]:
        content = (DEFAULT_AUTHOR_CODE_DIR / record["path"]).read_bytes()
        assert len(content) == record["bytes"]
        assert hashlib.sha256(content).hexdigest() == record["sha256"]
        assert recorded[record["path"]]["sha256"] == record["sha256"]


@pytest.mark.parametrize("explicit_log_dir", [False, True])
def test_generation_keeps_logs_separate_from_samples(tmp_path, monkeypatch, explicit_log_dir):
    import simulation.generate_author_dataset as generation
    import simulation.generation_execution as execution
    data_dir = tmp_path / "ssd" / "dataset"
    log_root = tmp_path / "project" / "artifacts" / "generation"
    log_dir = log_root / "custom_run"
    monkeypatch.setattr(generation, "DEFAULT_LOG_ROOT", log_root)
    argv = ["generate", "--output-dir", str(data_dir), "--storage", "compact", "--num-samples", "1"]
    if explicit_log_dir:
        argv += ["--log-dir", str(log_dir)]
    monkeypatch.setattr(sys, "argv", argv)
    sample = make_sample()
    sample.metadata.update(shape="wulff", strain_argument="random", nstep=120, generation_seconds=0.001)
    monkeypatch.setattr(execution, "load_author_modules", lambda *a, **kw: (None, None, None))
    monkeypatch.setattr(execution, "create_paper_particle", lambda *a, **kw: None)
    monkeypatch.setattr(execution, "generate_paper_observation", lambda *a, **kw: sample)
    try:
        assert generation.main() == 0
        if not explicit_log_dir:
            directories = list(log_root.iterdir())
            assert len(directories) == 1
            log_dir = directories[0]
        assert {path.name for path in data_dir.iterdir()} == {"sample_00000.npz", "dataset_manifest.json", ".generation.lock"}
        assert {path.name for path in log_dir.iterdir()} == {"generation.log", "config.json"}
        config = json.loads((log_dir / "config.json").read_text())
        manifest = json.loads((data_dir / "dataset_manifest.json").read_text())
        assert config["args"]["output_dir"] == str(data_dir.resolve())
        assert config["args"]["log_dir"] == manifest["generation_log_dir"] == str(log_dir.resolve())
        assert "Generated 1/1" in (log_dir / "generation.log").read_text()
    finally:
        for handler in generation.LOGGER.handlers:
            handler.close()
        generation.LOGGER.handlers.clear()


def test_backend_failure_leaves_config_and_logs_in_run_directory(tmp_path, monkeypatch):
    import simulation.generate_author_dataset as generation
    import simulation.generation_execution as execution
    data_dir, log_dir = tmp_path / "dataset", tmp_path / "logs"
    monkeypatch.setattr(sys, "argv", ["generate", "--output-dir", str(data_dir), "--log-dir", str(log_dir)])
    def fail_backend(*args, **kwargs):
        raise RuntimeError("backend preflight failed")
    monkeypatch.setattr(execution, "load_author_modules", fail_backend)
    try:
        with pytest.raises(RuntimeError, match="backend preflight failed"):
            generation.main()
        assert (log_dir / "generation.log").is_file()
        assert (log_dir / "config.json").is_file()
        assert {path.name for path in data_dir.iterdir()} == {".generation.lock"}
    finally:
        for handler in generation.LOGGER.handlers:
            handler.close()
        generation.LOGGER.handlers.clear()


def test_existing_dataset_is_not_overwritten_when_log_dir_changes(tmp_path, monkeypatch):
    import simulation.generate_author_dataset as generation
    data_dir, log_dir = tmp_path / "dataset", tmp_path / "logs"
    data_dir.mkdir()
    marker = data_dir / "dataset_manifest.json"
    marker.write_text("existing dataset")
    monkeypatch.setattr(sys, "argv", ["generate", "--output-dir", str(data_dir), "--log-dir", str(log_dir)])
    with pytest.raises(FileExistsError, match="already contains"):
        generation.main()
    assert marker.read_text() == "existing dataset"
    assert not log_dir.exists()
