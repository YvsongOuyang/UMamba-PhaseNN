"""Filter train/val/test manifest entries without touching generated NPZ files."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from simulation.generation_execution import dataset_lock


LOGGER = logging.getLogger("high_strain.filter_author_manifest")
BACKUP_NAME = "dataset_manifest.unfiltered.json"


def _select_records(manifest: dict[str, Any], root: Path, minimum: float) -> dict[str, Any]:
    """Validate the original splits before removing individual observations."""
    if manifest.get("route") != "author_generator" or manifest.get("split_unit") != "particle":
        raise ValueError("Expected an author_generator manifest with particle-disjoint splits.")
    splits = manifest.get("splits", {})
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("Expected train, val and test split counts.")
    records = manifest.get("samples", [])
    if len(records) != manifest.get("num_samples"):
        raise ValueError("Dataset manifest is incomplete.")
    counts, kept_counts, pairs = Counter(), Counter(), Counter()
    seen, particles, kept = set(), {}, []
    excluded = {split: [] for split in splits}
    for record in records:
        name, split, metadata = record["filename"], record["split"], record["metadata"]
        if split not in splits:
            raise ValueError(f"Unknown split for {name}: {split}")
        relative = Path(name)
        path = (root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or root not in path.parents:
            raise ValueError(f"Unsafe sample filename: {name}")
        if name in seen:
            raise ValueError(f"Duplicate sample: {name}")
        seen.add(name)
        if not path.is_file():
            raise ValueError(f"Missing sample: {name}")
        particle = (int(metadata["particle_seed"]), metadata["shape"])
        if particles.setdefault(particle, split) != split:
            raise ValueError(f"Particle leakage across splits: {name}")
        if metadata.get("split", split) != split:
            raise ValueError(f"Sample split disagrees with metadata: {name}")
        measured = metadata.get("measured_object_oversampling_xyz")
        if (
            not isinstance(measured, list) or len(measured) != 3
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value <= 0 for value in measured)
        ):
            raise ValueError(f"Missing or invalid oversampling metadata: {name}")
        counts[split] += 1
        if all(value > minimum for value in measured):
            kept.append(record)
            kept_counts[split] += 1
            pairs[f"{metadata['shape']}+{metadata['strain_argument']}"] += 1
        else:
            excluded[split].append(name)
    if any(counts[split] != count for split, count in splits.items()):
        raise ValueError("Declared split sizes disagree with sample records.")
    if any(count and not kept_counts[split] for split, count in splits.items()):
        raise ValueError("Filtering would empty a previously nonempty split; no manifest was changed.")
    return {
        "samples": kept, "num_samples": len(kept),
        "splits": {split: kept_counts[split] for split in splits},
        "category_counts": dict(sorted(pairs.items())),
        "excluded_filenames": excluded,
    }


def filter_manifest(root: str | Path, minimum: float = 2.0, *, dry_run: bool = False) -> dict[str, Any]:
    """Back up and atomically replace only the active sample manifest."""
    if not math.isfinite(minimum) or minimum <= 0:
        raise ValueError("Minimum oversampling must be finite and positive.")
    root = Path(root).expanduser().resolve()
    path, backup = root / "dataset_manifest.json", root / BACKUP_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {path}")
    with dataset_lock(root):
        active_raw = path.read_bytes()
        active = json.loads(active_raw)
        previous = active.get("index_filter")
        if previous is not None:
            if previous.get("min_exclusive_per_axis") != minimum:
                raise ValueError("Already filtered at a different threshold; restore the unfiltered manifest first.")
            source_raw = backup.read_bytes()
            if hashlib.sha256(source_raw).hexdigest() != previous.get("original_manifest_sha256"):
                raise ValueError("Unfiltered backup hash does not match the active manifest.")
        else:
            source_raw = active_raw
        source = json.loads(source_raw) if previous is not None else active
        selected = _select_records(source, root, minimum)
        excluded = selected.pop("excluded_filenames")
        selection = {
            "min_exclusive_per_axis": minimum,
            "source": "metadata.measured_object_oversampling_xyz",
            "original_manifest": BACKUP_NAME,
            "original_manifest_sha256": hashlib.sha256(source_raw).hexdigest(),
            "original_num_samples": source["num_samples"],
            "original_splits": source["splits"],
            "excluded_counts": {split: len(names) for split, names in excluded.items()},
            "excluded_filenames": excluded,
            "created_at_utc": previous["created_at_utc"] if previous else datetime.now(timezone.utc).isoformat(),
        }
        updated = {**source, **selected, "index_filter": selection}
        if previous is not None and updated != active:
            raise ValueError("Active manifest differs from filtering its backup; refusing to overwrite it.")
        for split, original in source["splits"].items():
            LOGGER.info("%-5s | original=%d | kept=%d | excluded=%d", split, original,
                        selected["splits"][split], len(excluded[split]))
        if dry_run or previous is not None:
            LOGGER.info("%s; no manifest changed.", "Dry run" if dry_run else "Already applied")
            return selection
        if backup.exists():
            if backup.read_bytes() != source_raw:
                raise FileExistsError(f"Refusing to replace a different backup: {backup}")
        else:
            with backup.open("xb") as handle:
                handle.write(source_raw)
                handle.flush()
                os.fsync(handle.fileno())
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root,
                                             prefix=".dataset_manifest.", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(updated, handle, indent=2, ensure_ascii=False, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            # Do not overwrite a manifest edited outside the cooperative lock.
            if path.read_bytes() != active_raw:
                raise RuntimeError("Manifest changed during filtering; active file was not replaced.")
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        LOGGER.info("Updated: %s | original backup: %s | NPZ files unchanged", path, backup)
        return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--min-oversampling", type=float, default=2.0,
                        help="Keep only observations strictly above this value on ALL axes (default: 2).")
    parser.add_argument("--dry-run", action="store_true", help="Validate and show counts without updating the manifest.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    filter_manifest(args.data_dir, args.min_oversampling, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
