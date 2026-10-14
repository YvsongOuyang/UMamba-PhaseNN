"""Physical and training contracts for the native 512-square 2D pipeline."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from pytorch_autophasenn.momamba_refiner import ambiguity_aware_component_mae
from pytorch_autophasenn.reconstruction import (
    farfield_modulus_from_realspace,
    project_to_measured_modulus,
    realspace_from_modulus_phase,
)
from ecdi_training.data import DiffractionDataset2D, center_object, prepare_batch
from ecdi_training.evaluate import metrics_2d
from ecdi_training.model import (
    HighStrainPhaseUNet2D,
    ComplexMoMambaRefiner2D,
    HighStrainMoMambaCascade2D,
)
from ecdi_training.train import make_model, load_checkpoint
from ecdi_training import train, evaluate


class TestMixer(nn.Module):
    """Test-only token mixer; production always uses the official Mamba."""

    __test__ = False

    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * self.scale


class PhaseProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)
        self.norm = nn.BatchNorm2d(1)

    def forward(self, x):
        return self.norm(self.conv(x))


@pytest.fixture
def config():
    path = Path(__file__).resolve().parents[1] / "configs" / "diffraction_2d.json"
    return json.loads(path.read_text(encoding="utf-8"))


def refiner(**kwargs):
    return ComplexMoMambaRefiner2D(
        base_channels=2,
        num_experts=(2, 2, 2, 2),
        top_k=1,
        router_dropout=0,
        position_dropout=0,
        codec_dropout=0,
        d_state=2,
        mixer_factory=TestMixer,
        **kwargs,
    )


def test_512_phase_backbone_forward():
    torch.set_num_threads(4)
    model = HighStrainPhaseUNet2D().eval()
    assert not any(
        isinstance(m, (nn.Conv3d, nn.BatchNorm3d, nn.ConvTranspose3d))
        for m in model.modules()
    )
    with torch.no_grad():
        output = model(torch.rand(1, 1, 512, 512))
    assert output.shape == (1, 1, 512, 512)
    assert torch.isfinite(output).all()


def test_phase_training_backpropagates():
    model = HighStrainPhaseUNet2D().train()
    output = model(torch.rand(2, 1, 32, 32))
    output.square().mean().backward()
    grad = model.layers["conv3d"].conv.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_512_cascade_identity_freeze_fft_and_gradients():
    model = HighStrainMoMambaCascade2D(PhaseProbe(), refiner()).train()
    assert not model.phase_model.training
    assert all(not p.requires_grad for p in model.phase_model.parameters())
    measured = torch.rand(1, 1, 512, 512) + 0.1
    output = model(torch.rand_like(measured), measured)
    assert output.refined_object.shape == (1, 1, 512, 512)
    torch.testing.assert_close(output.refined_object, output.initial_object)
    torch.testing.assert_close(
        farfield_modulus_from_realspace(output.initial_object, spatial_dims=2),
        measured,
        atol=2e-5,
        rtol=2e-5,
    )
    output.refined_object.real.square().mean().backward()
    assert model.refiner.output.real_kernel.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.phase_model.parameters())


def test_centering_is_translation_invariant_and_preserves_modulus(config):
    target = torch.zeros(2, 1, 512, 512, dtype=torch.complex64)
    target[:, :, 240:263, 246:275] = 1 + 0.2j
    shifted = torch.roll(target, (7, -11), (-2, -1))
    centered = center_object(target)
    torch.testing.assert_close(center_object(shifted), centered, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        farfield_modulus_from_realspace(centered, spatial_dims=2),
        farfield_modulus_from_realspace(target, spatial_dims=2),
        atol=2e-3,
        rtol=2e-4,
    )
    measured = farfield_modulus_from_realspace(target, spatial_dims=2)
    a = prepare_batch(
        {"diffraction": measured.square(), "realspace": target},
        config,
        torch.device("cpu"),
    )
    modulus_config = copy.deepcopy(config)
    modulus_config["diffraction_kind"] = "modulus"
    b = prepare_batch(
        {"diffraction": measured, "realspace": target},
        modulus_config,
        torch.device("cpu"),
    )
    torch.testing.assert_close(a["input"], b["input"])
    torch.testing.assert_close(a["diffraction"], b["diffraction"])
    reconstructed = realspace_from_modulus_phase(
        a["diffraction"], a["target_phase"], spatial_dims=2
    )
    loss = ambiguity_aware_component_mae(
        reconstructed,
        a["realspace"],
        torch.ones_like(target, dtype=torch.bool),
        spatial_dims=2,
    )
    assert float(loss.loss) < 1e-5


def test_projection_and_ambiguity_metrics():
    generator = torch.Generator().manual_seed(5)
    target = torch.complex(
        torch.rand(2, 1, 32, 32, generator=generator),
        torch.rand(2, 1, 32, 32, generator=generator),
    )
    measured = farfield_modulus_from_realspace(target, spatial_dims=2)
    prediction = target * torch.exp(torch.tensor(0.7j)) * 2
    projected = project_to_measured_modulus(prediction, measured, spatial_dims=2)
    torch.testing.assert_close(
        farfield_modulus_from_realspace(projected, spatial_dims=2),
        measured,
        atol=2e-4,
        rtol=2e-4,
    )
    loss = ambiguity_aware_component_mae(
        prediction, target, torch.ones_like(target, dtype=torch.bool), spatial_dims=2
    )
    assert float(loss.loss) < 1e-5
    values = metrics_2d(prediction, target, measured, torch.ones_like(measured))
    assert torch.isfinite(torch.stack(list(values.values()))).all()
    assert values["complex_nrmse"].max() < 1e-5
    assert values["phase_wca"].max() < 1e-5
    assert values["real_amp_ssim"].min() > 0.999


def test_native_npy_dataset_and_reject_3d_checkpoint(tmp_path, config):
    config["root"] = str(tmp_path)
    np.save(tmp_path / "train_diff.npy", np.ones((1, 512, 512), dtype=np.float32))
    np.save(tmp_path / "train_real.npy", np.ones((1, 512, 512), dtype=np.complex64))
    dataset = DiffractionDataset2D(config, "train")
    assert len(dataset) == 1 and dataset[0]["realspace"].shape == (1, 512, 512)
    checkpoint = tmp_path / "old3d.pt"
    torch.save({"model_state_dict": {}}, checkpoint)
    with pytest.raises(ValueError, match="native 2D"):
        load_checkpoint(checkpoint)


def test_reject_bad_inputs_and_mismatched_preprocessing(config):
    batch = {
        "diffraction": torch.zeros(1, 1, 512, 512),
        "realspace": torch.ones(1, 1, 512, 512, dtype=torch.complex64),
    }
    with pytest.raises(ValueError, match="nonzero"):
        prepare_batch(batch, config, torch.device("cpu"))
    changed = copy.deepcopy(config)
    changed["center_targets"] = False
    with pytest.raises(ValueError, match="center_targets"):
        make_model(
            "refiner",
            changed,
            phase_checkpoint={"stage": "phase", "config": config},
            mixer_factory=TestMixer,
        )


def test_phase_training_checkpoint_resume_and_evaluation(tmp_path, config, monkeypatch):
    coordinate = np.arange(512, dtype=np.float32) - 256
    yy, xx = np.meshgrid(coordinate, coordinate, indexing="ij")
    target = ((np.abs(xx) < 50) & (np.abs(yy) < 40)) * np.exp(
        1j * (0.003 * xx + 0.01 * yy)
    )
    target = target.astype(np.complex64)[None]
    measured = (
        np.abs(
            np.fft.fftshift(
                np.fft.fft2(np.fft.ifftshift(target, axes=(-2, -1))), axes=(-2, -1)
            )
        )
        ** 2
    )
    measured = (measured / measured.max()).astype(np.float32)
    for split in ("train", "val"):
        np.save(tmp_path / f"{split}_diff.npy", measured)
        np.save(tmp_path / f"{split}_real.npy", target)
    config["root"] = str(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    arguments = [
        "train",
        "--stage",
        "phase",
        "--data-config",
        str(config_path),
        "--epochs",
        "1",
        "--num-workers",
        "0",
        "--device",
        "cpu",
        "--run-name",
        "smoke",
        "--runs-dir",
        str(tmp_path / "runs"),
        "--checkpoint-root",
        str(tmp_path / "checkpoints"),
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    assert train.main() == 0
    checkpoint = tmp_path / "checkpoints" / "smoke" / "checkpoint_last.pt"
    saved = load_checkpoint(checkpoint)
    assert saved["epoch"] == 1 and np.isfinite(saved["best_val_loss"])
    # Resume one more epoch, using the full saved optimizer and scheduler state.
    arguments[arguments.index("--epochs") + 1] = "2"
    monkeypatch.setattr(sys, "argv", arguments + ["--resume", str(checkpoint)])
    assert train.main() == 0
    assert load_checkpoint(checkpoint)["epoch"] == 2
    output = tmp_path / "evaluation"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--checkpoint",
            str(checkpoint),
            "--data-config",
            str(config_path),
            "--split",
            "val",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--output-dir",
            str(output),
        ],
    )
    assert evaluate.main() == 0
    report = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert report["num_samples"] == 1
    assert report["metrics"]["initial"]["paper_modulus_mae"] < 1e-6
    assert (output / "per_sample.csv").is_file()
    assert (output / "sample_00000.npz").is_file()
