"""Exact-size, reproducible stratified partitions without image duplication."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def stratified_holdout(rows: list[dict[str, Any]], holdout_count: int,
                       seed: int) -> tuple[list[int], list[int]]:
    """Return sorted remaining/holdout indices with proportional material quotas."""
    if not 0 < holdout_count < len(rows):
        raise ValueError("holdout_count must lie strictly between 0 and sample count")
    groups: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        groups[str(row["material"])].append(int(row["index"]))
    all_indices = [index for indices in groups.values() for index in indices]
    if len(set(all_indices)) != len(all_indices):
        raise ValueError("Duplicate sample indices")
    raw = {name: holdout_count * len(indices) / len(rows)
           for name, indices in groups.items()}
    quotas = {name: int(np.floor(value)) for name, value in raw.items()}
    leftovers = holdout_count - sum(quotas.values())
    ranking = sorted(groups, key=lambda name: (-(raw[name] - quotas[name]), name))
    for name in ranking[:leftovers]:
        quotas[name] += 1
    rng = np.random.default_rng(seed)
    remaining: list[int] = []
    holdout: list[int] = []
    for name in sorted(groups):
        shuffled = rng.permutation(groups[name]).tolist()
        holdout.extend(shuffled[:quotas[name]])
        remaining.extend(shuffled[quotas[name]:])
    if len(holdout) != holdout_count or set(holdout) & set(remaining):
        raise AssertionError("Stratified partition failed")
    return sorted(remaining), sorted(holdout)


def build_splits(rows: list[dict[str, Any]], train_count: int, test_count: int,
                 validation_count: int, seed: int
                 ) -> tuple[dict[str, list[int]], dict[str, list[int]] | None]:
    """Create exact 9000/1000 and optional disjoint train/val/test manifests."""
    if len(rows) != train_count + test_count:
        raise ValueError("Rows do not match the requested train/test counts")
    train, test = stratified_holdout(rows, test_count, seed)
    primary = {"train": train, "test": test}
    if validation_count == 0:
        return primary, None
    by_index = {int(row["index"]): row for row in rows}
    development_rows = [by_index[index] for index in train]
    fit, val = stratified_holdout(development_rows, validation_count, seed + 1)
    return primary, {"train": fit, "val": val, "test": test}
