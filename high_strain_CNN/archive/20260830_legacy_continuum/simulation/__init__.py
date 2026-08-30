"""Paper-style synthetic data generation for the HighStrain phase network."""

from .config import SimulationConfig, load_simulation_config
from .generator import SimulatedSample, generate_sample, save_sample

__all__ = [
    "SimulatedSample",
    "SimulationConfig",
    "generate_sample",
    "load_simulation_config",
    "save_sample",
]
