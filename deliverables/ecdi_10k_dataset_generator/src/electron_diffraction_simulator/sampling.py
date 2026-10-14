"""Deterministic parameter sampling helpers."""

from __future__ import annotations

import numpy as np

from .config import FloatRange


def sample_uniform(rng: np.random.Generator, value_range: FloatRange) -> float:
    """Sample a scalar uniformly from a configured closed interval."""

    if value_range.low == value_range.high:
        return float(value_range.low)
    return float(rng.uniform(value_range.low, value_range.high))
