"""Fast, pure-Python checks of dataset partition and size contracts."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecdi_dataset_builder.config import DatasetPlan  # noqa: E402
from ecdi_dataset_builder.splits import build_splits  # noqa: E402


def test_default_plan_exact_counts() -> None:
    plan = DatasetPlan.load(ROOT / "configs/dataset_plan.json")
    assert (plan.total, plan.train_count, plan.test_count) == (10000, 9000, 1000)
    assert plan.validation_count_within_train == 500


def test_stratified_splits_exact_disjoint_reproducible() -> None:
    rows = [{"index": index, "material": f"material_{index % 6}"}
            for index in range(120)]
    first, with_val = build_splits(rows, 108, 12, 6, seed=42)
    repeated, repeated_with_val = build_splits(rows, 108, 12, 6, seed=42)
    assert first == repeated and with_val == repeated_with_val
    assert len(first["train"]) == 108 and len(first["test"]) == 12
    assert set(first["train"]).isdisjoint(first["test"])
    assert sorted(first["train"] + first["test"]) == list(range(120))
    assert with_val is not None
    assert {key: len(value) for key, value in with_val.items()} == {
        "train": 102, "val": 6, "test": 12}
    assert sorted(sum(with_val.values(), [])) == list(range(120))
    for material in range(6):
        assert sum(index % 6 == material for index in first["test"]) == 2


def test_full_size_split_contract() -> None:
    rows = [{"index": index, "material": f"material_{index % 6}"}
            for index in range(10000)]
    primary, training = build_splits(rows, 9000, 1000, 500, seed=20261016)
    assert {key: len(value) for key, value in primary.items()} == {
        "train": 9000, "test": 1000}
    assert training is not None
    assert {key: len(value) for key, value in training.items()} == {
        "train": 8500, "val": 500, "test": 1000}
    assert training["test"] == primary["test"]
    assert sorted(sum(training.values(), [])) == list(range(10000))
    test_material_counts = [sum(index % 6 == material for index in primary["test"])
                            for material in range(6)]
    assert max(test_material_counts) - min(test_material_counts) <= 1
