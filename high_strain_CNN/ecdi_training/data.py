"""Explicit paired 2D NPY data; no reinterpretation of 3D memmaps or slices."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from pytorch_autophasenn.reconstruction import reciprocal_field_from_realspace


SHAPE = (512, 512)


def load_config(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("spatial_dims") != 2 or tuple(config.get("shape", ())) != SHAPE:
        raise ValueError(
            "2D configuration must specify spatial_dims=2 and shape=[512,512]."
        )
    if config.get("diffraction_kind") not in ("intensity", "modulus"):
        raise ValueError("diffraction_kind must explicitly be intensity or modulus.")
    config.setdefault("input_log_data", True)
    config.setdefault("center_targets", True)
    if config.get("storage", "npy") not in ("npy", "ecdi_hdf5"):
        raise ValueError("storage must be npy or ecdi_hdf5.")
    if (
        config.get("storage") == "ecdi_hdf5"
        and config["diffraction_kind"] != "intensity"
    ):
        raise ValueError(
            "The e-CDI observed_counts field contains intensity, not modulus."
        )
    training = config["training"]
    for stage in ("phase", "refiner"):
        settings = training[stage]
        if settings["scheduler"] not in ("plateau", "cosine"):
            raise ValueError("Scheduler must be plateau or cosine.")
        if (
            settings["epochs"] < 1
            or not np.isfinite(settings["learning_rate"])
            or settings["learning_rate"] <= 0
        ):
            raise ValueError("Epochs and learning rate must be positive and finite.")
    if not 0 < training["support_threshold"] < 1:
        raise ValueError("Support threshold must be in (0,1).")
    if (
        not np.isfinite(training["physics_loss_weight"])
        or training["physics_loss_weight"] < 0
    ):
        raise ValueError("physics_loss_weight must be finite and nonnegative.")
    if not np.isfinite(training["gradient_clip"]) or training["gradient_clip"] < 0:
        raise ValueError("gradient_clip must be finite and nonnegative.")
    root = Path(config["root"]).expanduser()
    config["root"] = str(
        (path.parent / root).resolve() if not root.is_absolute() else root
    )
    config["config_path"] = str(path)
    return config


def build_dataset(
    config: dict, split: str, num_samples: int = 0, *, include_ideal: bool = False
):
    if config.get("storage") == "ecdi_hdf5":
        from .ecdi_data import EcdiHDF5Dataset

        return EcdiHDF5Dataset(config, split, num_samples, include_ideal=include_ideal)
    return DiffractionDataset2D(config, split, num_samples)


class DiffractionDataset2D(Dataset):
    """Read headered [N,512,512] float diffraction and complex object arrays."""

    def __init__(self, config: dict, split: str, num_samples: int = 0):
        self.root = Path(config["root"])
        self.split = split
        self.paths = {
            kind: self.root / config["splits"][split][kind]
            for kind in ("diffraction", "realspace")
        }
        self.arrays = None
        arrays = self._open()
        diffraction, realspace = arrays
        if diffraction.ndim != 3 or tuple(diffraction.shape[1:]) != SHAPE:
            raise ValueError(
                "Diffraction must be a headered NPY array with shape [N,512,512]."
            )
        if realspace.shape != diffraction.shape or len(diffraction) == 0:
            raise ValueError(
                "Nonempty paired diffraction and realspace arrays must have equal shapes."
            )
        if diffraction.dtype != np.dtype("float32") or realspace.dtype != np.dtype(
            "complex64"
        ):
            raise ValueError(
                "Use diffraction float32 and realspace complex64 NPY arrays."
            )
        if num_samples < 0 or num_samples > len(diffraction):
            raise ValueError(
                "num_samples must be zero (all) or within the available split."
            )
        self.count = num_samples or len(diffraction)
        self.manifest = {
            "split": split,
            "shape": list(SHAPE),
            "selected_samples": self.count,
            "available_samples": len(diffraction),
            "paths": {k: str(v) for k, v in self.paths.items()},
            "diffraction_kind": config["diffraction_kind"],
            "storage": "headered_npy",
        }

    def _open(self):
        if self.arrays is None:
            self.arrays = tuple(
                np.load(self.paths[kind], mmap_mode="r", allow_pickle=False)
                for kind in ("diffraction", "realspace")
            )
        return self.arrays

    def __getstate__(self):
        state = self.__dict__.copy()
        state["arrays"] = None
        return state

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        diffraction, realspace = self._open()
        return {
            "diffraction": torch.from_numpy(np.array(diffraction[index], copy=True))[
                None
            ],
            "realspace": torch.from_numpy(np.array(realspace[index], copy=True))[None],
        }


def center_object(field: torch.Tensor) -> torch.Tensor:
    """Remove amplitude-COM translation from complex [B,1,H,W] objects."""

    if field.ndim != 4 or field.shape[1] != 1 or not torch.is_complex(field):
        raise ValueError("center_object requires complex [B,1,H,W].")
    amplitude = field.abs()
    # Accumulate the COM accurately on a 512-square grid; roundoff otherwise
    # becomes a spurious fractional-pixel phase ramp even for integer shifts.
    mass_amplitude = amplitude.to(torch.float64)
    mass = mass_amplitude.sum(dim=(-2, -1), keepdim=True)
    safe_mass = mass.clamp_min(torch.finfo(mass_amplitude.dtype).eps)
    ramp = torch.zeros_like(amplitude)
    for axis, size in enumerate(field.shape[-2:]):
        shape = [1, 1, 1, 1]
        shape[axis + 2] = size
        coordinates = torch.arange(
            size, device=field.device, dtype=mass_amplitude.dtype
        ).reshape(shape)
        offset = (mass_amplitude * coordinates).sum(
            dim=(-2, -1), keepdim=True
        ) / safe_mass - size // 2
        offset = torch.where(mass > 0, offset, torch.zeros_like(offset))
        offset = offset.to(amplitude.dtype)
        frequencies = torch.fft.fftshift(
            torch.fft.fftfreq(size, device=field.device, dtype=amplitude.dtype)
        ).reshape(shape)
        ramp = ramp + offset * frequencies
    spectrum = reciprocal_field_from_realspace(field, spatial_dims=2)
    spectrum = spectrum * torch.exp(
        torch.complex(torch.zeros_like(ramp), 2 * torch.pi * ramp)
    )
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(spectrum, dim=(-2, -1))), dim=(-2, -1)
    )


def prepare_batch(batch: dict, config: dict, device: torch.device) -> dict:
    """Perform physical preprocessing on the selected device, outside Dataset."""

    measured = batch["diffraction"].to(device, non_blocking=True)
    target = batch["realspace"].to(device, non_blocking=True)
    if (
        not torch.isfinite(measured).all()
        or (measured < 0).any()
        or not torch.isfinite(target).all()
    ):
        raise ValueError("Data contain negative diffraction, NaNs or infinities.")
    axes = (-2, -1)
    eps = torch.finfo(measured.dtype).eps
    if (measured.amax(dim=axes) <= 0).any() or (target.abs().amax(dim=axes) <= 0).any():
        raise ValueError("Each diffraction frame and target must be nonzero.")
    intensity = (
        measured if config["diffraction_kind"] == "intensity" else measured.square()
    )
    modulus = intensity.sqrt()
    modulus = modulus / modulus.amax(dim=axes, keepdim=True).clamp_min(eps)
    model_input = (
        torch.log1p(intensity) if config.get("input_log_data", True) else intensity
    )
    minimum = model_input.amin(dim=axes, keepdim=True)
    model_input = (model_input - minimum) / (
        model_input.amax(dim=axes, keepdim=True) - minimum
    ).clamp_min(eps)
    if (model_input.amax(dim=axes) <= 0).any():
        raise ValueError(
            "Constant diffraction frames produce zero WCA weights after min-max normalization."
        )
    if config.get("center_targets", True):
        target = center_object(target)
    spectrum = reciprocal_field_from_realspace(target, spatial_dims=2)
    phase = torch.angle(spectrum)
    center_phase = phase[..., phase.shape[-2] // 2, phase.shape[-1] // 2]
    phase = phase - center_phase[..., None, None]
    prepared = {
        "input": model_input,
        "diffraction": modulus,
        "realspace": target,
        "target_phase": phase,
    }
    for key in ("index", "material"):
        if key in batch:
            prepared[key] = batch[key]
    if "ideal_intensity" in batch:
        prepared["ideal_intensity"] = batch["ideal_intensity"].to(
            device, non_blocking=True
        )
    return prepared
