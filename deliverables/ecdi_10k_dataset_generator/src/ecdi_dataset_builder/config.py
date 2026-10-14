"""Validated plan for a two-way train/test simulated diffraction corpus."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path


@dataclass(frozen=True)
class DatasetPlan:
    output_root: str
    train_count: int
    test_count: int
    validation_count_within_train: int
    split_seed: int
    max_fourier_nrmse: float
    estimated_compressed_mib_per_sample: float
    free_space_reserve_gib: float

    @classmethod
    def load(cls, path: Path) -> "DatasetPlan":
        plan = cls(**json.loads(path.read_text(encoding="utf-8")))
        plan.validate()
        return plan

    def with_overrides(self, **changes: object) -> "DatasetPlan":
        plan = replace(self, **{key: value for key, value in changes.items() if value is not None})
        plan.validate()
        return plan

    def validate(self) -> None:
        if not self.output_root:
            raise ValueError("output_root is required")
        if self.train_count < 2 or self.test_count < 1:
            raise ValueError("train_count must be >=2 and test_count >=1")
        if not 0 <= self.validation_count_within_train < self.train_count:
            raise ValueError("validation_count_within_train must be in [0, train_count)")
        if self.split_seed < 0 or self.max_fourier_nrmse <= 0:
            raise ValueError("Invalid split seed or Fourier tolerance")
        if self.estimated_compressed_mib_per_sample <= 0 or self.free_space_reserve_gib < 0:
            raise ValueError("Invalid storage estimate or free-space reserve")

    @property
    def total(self) -> int:
        return self.train_count + self.test_count
