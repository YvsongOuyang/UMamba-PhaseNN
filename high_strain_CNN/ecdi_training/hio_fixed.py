"""Fixed-support complex HIO/ER comparison, with measurement-only selection."""
from __future__ import annotations

import math

import torch

from .benchmark_selected100 import fftc, ifftc


HIO_DEFAULTS = {
    "beta": 0.9,
    "restarts": 12,
    "initial_er_iterations": 20,
    "cycles": 10,
    "hio_iterations_per_cycle": 300,
    "er_iterations_per_cycle": 50,
    "final_er_iterations": 500,
    "initial_autocorrelation_threshold": 0.04,
    "support_update": "fixed",
}


@torch.no_grad()
def make_model_hio_starts(initial_object, seed, phase_noise_std=0.15):
    """Make two model-derived starts: direct and reproducibly phase-perturbed."""
    if (initial_object.ndim != 4 or initial_object.shape[0] != 1
            or initial_object.shape[1] != 1 or not torch.is_complex(initial_object)
            or not bool(torch.isfinite(initial_object).all())):
        raise ValueError("Expected finite complex model output [1,1,H,W].")
    generator = torch.Generator(device=initial_object.device).manual_seed(seed)
    noise = torch.randn(initial_object.shape, device=initial_object.device,
                        dtype=initial_object.real.dtype, generator=generator)
    perturbed = torch.polar(initial_object.abs(),
                            torch.angle(initial_object) + phase_noise_std * noise)
    return torch.cat((initial_object, perturbed), dim=0)


@torch.no_grad()
def reconstruct_hio(modulus, config, seed, initial_object=None):
    """Use measured modulus only; reference objects never enter reconstruction.

    A 4%-of-peak autocorrelation contour supplies the fixed initial support.
    Random-start mode selects among 12 starts by measured-modulus residual.
    If initial_object is provided, run one trajectory per supplied model-derived
    start and select by measured-modulus residual only.
    No positivity or real-object constraint is imposed on the complex field.
    """
    if config != HIO_DEFAULTS:
        raise ValueError("Use the fixed documented HIO/ER configuration.")
    if modulus.ndim != 4 or modulus.shape[:2] != (1, 1):
        raise ValueError("Expected a single [1,1,H,W] measured modulus.")
    if not bool(torch.isfinite(modulus).all() & (modulus >= 0).all() & (modulus.amax() > 0)):
        raise ValueError("Measured modulus must be finite, nonnegative and nonzero.")
    m = modulus[0, 0]
    h, w = m.shape
    count = config["restarts"] if initial_object is None else initial_object.shape[0]
    autocorrelation = ifftc(m.square()).abs()
    support = autocorrelation >= config["initial_autocorrelation_threshold"] * autocorrelation.amax()
    support = support[None].expand(count, h, w)
    if initial_object is None:
        generator = torch.Generator(device=m.device).manual_seed(seed)
        phases = (torch.rand((count, h, w), device=m.device,
                             dtype=m.dtype, generator=generator) * 2 - 1) * math.pi
        fields = ifftc(torch.polar(m.expand(count, h, w), phases))
    else:
        if (initial_object.ndim != 4 or initial_object.shape[1:] != modulus.shape[1:]
                or not 1 <= count <= config["restarts"] or not torch.is_complex(initial_object)
                or initial_object.device != modulus.device
                or not bool(torch.isfinite(initial_object).all())):
            raise ValueError("Model starts must be finite complex [N,1,H,W] on the same device.")
        fields = initial_object[:, 0].to(torch.complex64).clone()

    def iterate(field, iterations, use_hio):
        for _ in range(iterations):
            projected = ifftc(torch.polar(m.expand(count, h, w), torch.angle(fftc(field))))
            outside = field - config["beta"] * projected if use_hio else torch.zeros_like(projected)
            field = torch.where(support, projected, outside)
        return field

    fields = iterate(fields, config["initial_er_iterations"], False)
    for _ in range(config["cycles"]):
        fields = iterate(fields, config["hio_iterations_per_cycle"], True)
        fields = iterate(fields, config["er_iterations_per_cycle"], False)
    fields = iterate(fields, config["final_er_iterations"], False)
    residual = (fftc(fields).abs() - m).square().sum((-2, -1)) / m.square().sum()
    if not bool(torch.isfinite(fields).all() & torch.isfinite(residual).all()):
        raise RuntimeError("Nonfinite HIO reconstruction.")
    return fields[int(residual.argmin())][None, None]
