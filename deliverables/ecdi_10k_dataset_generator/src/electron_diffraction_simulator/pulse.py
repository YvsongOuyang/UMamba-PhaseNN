"""Reduced pulse-train model for dose and residual structural damage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .config import PulseConfig
from .sampling import sample_uniform


@dataclass(frozen=True)
class PulseSample:
    """One sampled pulse train and its observable structural consequences."""

    width_ps: float
    interval_ns: float
    relaxation_time_ns: float
    acquisition_time_ms: float
    dose_e_per_a2: float
    duty_cycle: float
    pulse_count: int
    recovery_fraction: float
    residual_damage_fraction: float
    damage_displacement_sigma_a: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def sample_pulse(config: PulseConfig, rng: np.random.Generator) -> PulseSample:
    """Sample pulse parameters and map them to residual atomic disorder.

    Pulse timing is not inserted as a fictitious detector filter. Its effect is
    represented through total dose and incomplete structural recovery between
    pulses, which are the two pathways observable in a final diffraction frame.
    """

    width_ps = sample_uniform(rng, config.width_ps)
    interval_ns = sample_uniform(rng, config.interval_ns)
    relaxation_time_ns = sample_uniform(rng, config.relaxation_time_ns)
    acquisition_time_ms = sample_uniform(rng, config.acquisition_time_ms)
    dose = sample_uniform(rng, config.dose_e_per_a2)

    width_ns = width_ps * 1e-3
    rest_time_ns = max(interval_ns - width_ns, 0.0)
    duty_cycle = min(width_ns / interval_ns, 1.0)
    pulse_count = max(int(round(acquisition_time_ms * 1e6 / interval_ns)), 1)
    recovery_fraction = 1.0 - math.exp(-rest_time_ns / relaxation_time_ns)
    dose_drive = dose / (dose + config.damage_threshold_e_per_a2)
    residual_damage = float(np.clip(dose_drive * (1.0 - recovery_fraction), 0.0, 1.0))
    damage_sigma = (
        config.maximum_damage_displacement_sigma_a * math.sqrt(residual_damage)
    )

    return PulseSample(
        width_ps=width_ps,
        interval_ns=interval_ns,
        relaxation_time_ns=relaxation_time_ns,
        acquisition_time_ms=acquisition_time_ms,
        dose_e_per_a2=dose,
        duty_cycle=duty_cycle,
        pulse_count=pulse_count,
        recovery_fraction=recovery_fraction,
        residual_damage_fraction=residual_damage,
        damage_displacement_sigma_a=damage_sigma,
    )
