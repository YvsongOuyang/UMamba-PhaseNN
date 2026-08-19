"""AutoPhaseNN memmap adapter for reciprocal-space phase supervision."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class AutoPhaseNNPhaseDataset(Dataset):
    """Read AutoPhaseNN diffraction/object memmaps for the PhaseUNet."""

    def __init__(
        self,
        diffraction_path: str | Path,
        realspace_path: str | Path,
        num_samples: int,
        shape: tuple[int, int, int] = (64, 64, 64),
        diffraction_dtype: str = "float32",
        realspace_dtype: str = "complex64",
        input_log_data: bool = True,
        return_diffraction_modulus: bool = False,
    ) -> None:
        self.diffraction_path = Path(diffraction_path)
        self.realspace_path = Path(realspace_path)
        self.num_samples = int(num_samples)
        self.shape = tuple(shape)
        self.input_log_data = bool(input_log_data)
        self.return_diffraction_modulus = bool(return_diffraction_modulus)
        self.diffraction = np.memmap(
            self.diffraction_path,
            dtype=diffraction_dtype,
            mode="r",
            shape=(self.num_samples,) + self.shape,
        )
        self.realspace = np.memmap(
            self.realspace_path,
            dtype=realspace_dtype,
            mode="r",
            shape=(self.num_samples,) + self.shape,
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        diffraction_modulus = np.array(
            self.diffraction[index],
            dtype=np.float32,
            copy=True,
        )
        intensity = np.square(diffraction_modulus, dtype=np.float32)
        model_input = np.log1p(intensity) if self.input_log_data else intensity
        minimum = float(model_input.min())
        maximum = float(model_input.max())
        scale = maximum - minimum
        if scale > 0.0:
            model_input = (model_input - minimum) / scale
        else:
            model_input = np.zeros_like(model_input)

        realspace = np.array(self.realspace[index], dtype=np.complex64, copy=True)
        sample = {
            "input": torch.from_numpy(np.asarray(model_input, dtype=np.float32)[None]),
            "realspace": torch.from_numpy(realspace),
            "name": f"{self.diffraction_path.stem}_{index:06d}",
        }
        if self.return_diffraction_modulus:
            sample["diffraction"] = torch.from_numpy(diffraction_modulus[None])
        return sample


def reciprocal_phase_from_realspace(realspace: torch.Tensor) -> torch.Tensor:
    """Generate the centered reciprocal phase used by the original dataset."""

    if realspace.ndim != 4 or not torch.is_complex(realspace):
        raise ValueError("realspace must be a complex tensor with shape [B, D, H, W].")
    shifted = torch.fft.ifftshift(realspace, dim=(-3, -2, -1))
    reciprocal = torch.fft.fftn(shifted, dim=(-3, -2, -1))
    reciprocal = torch.fft.fftshift(reciprocal, dim=(-3, -2, -1))
    phase = torch.angle(reciprocal)
    center = tuple(size // 2 for size in phase.shape[-3:])
    center_phase = phase[(slice(None),) + center]
    return phase - center_phase[:, None, None, None]
