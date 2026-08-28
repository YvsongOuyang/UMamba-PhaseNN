"""Validate AutoPhaseNN raw memmaps against the shared data configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pytorch_autophasenn.management import (
    DEFAULT_DATA_CONFIG,
    build_data_manifest,
    load_data_config,
    runtime_manifest,
)


LOGGER = logging.getLogger("high_strain.validate_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default=str(DEFAULT_DATA_CONFIG))
    parser.add_argument("--data-dir", default="", help="Override the configured root.")
    parser.add_argument("--splits", nargs="+", default=("train", "val"))
    parser.add_argument("--sample-checks", type=int, default=3)
    parser.add_argument("--sha256", action="store_true")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_indices(num_samples: int, checks: int) -> list[int]:
    if checks <= 0:
        return []
    count = min(num_samples, checks)
    return sorted(
        {
            int(index)
            for index in np.linspace(0, num_samples - 1, num=count, dtype=np.int64)
        }
    )


def inspect_memmap(
    path: Path,
    *,
    dtype: str,
    num_samples: int,
    shape: tuple[int, int, int],
    checks: int,
    nonnegative: bool,
) -> dict[str, object]:
    values = np.memmap(
        path,
        dtype=dtype,
        mode="r",
        shape=(num_samples,) + shape,
    )
    rows: list[dict[str, object]] = []
    all_finite = True
    all_nonnegative = True
    for index in sample_indices(num_samples, checks):
        sample = np.asarray(values[index])
        magnitude = np.abs(sample) if np.iscomplexobj(sample) else sample
        finite = bool(np.isfinite(sample).all())
        nonnegative_sample = bool(np.min(sample) >= 0) if nonnegative else True
        all_finite &= finite
        all_nonnegative &= nonnegative_sample
        rows.append(
            {
                "index": index,
                "finite": finite,
                "nonnegative": nonnegative_sample,
                "magnitude_min": float(np.min(magnitude)),
                "magnitude_max": float(np.max(magnitude)),
                "magnitude_mean": float(np.mean(magnitude)),
            }
        )
    del values
    return {
        "sample_indices": [row["index"] for row in rows],
        "all_finite": all_finite,
        "all_nonnegative": all_nonnegative,
        "samples": rows,
    }


def main() -> int:
    configure_logging()
    args = parse_args()
    if args.sample_checks < 0:
        raise ValueError("--sample-checks cannot be negative.")
    config = load_data_config(args.data_config)
    unknown_splits = set(args.splits).difference(config["splits"])
    if unknown_splits:
        raise ValueError(f"Unknown data splits: {sorted(unknown_splits)}")
    root = Path(args.data_dir or config["root"]).expanduser().resolve()
    shape = tuple(int(size) for size in config["shape"])
    dtypes = config["dtypes"]
    selected_splits = {name: config["splits"][name] for name in args.splits}
    manifest = build_data_manifest(
        config=config,
        root=root,
        shape=shape,
        diffraction_dtype=dtypes["diffraction"],
        realspace_dtype=dtypes["realspace"],
        splits=selected_splits,
        input_log_data=config.get("input_preprocessing", {}).get("transform") == "log1p",
    )

    checks: dict[str, dict[str, object]] = {}
    passed = True
    for split_name, split in manifest["splits"].items():
        split_checks: dict[str, object] = {}
        for kind, dtype, expected_key in (
            ("diffraction", dtypes["diffraction"], "expected_diffraction_bytes"),
            ("realspace", dtypes["realspace"], "expected_realspace_bytes"),
        ):
            path = Path(split[kind])
            expected_bytes = int(split[expected_key])
            exists = path.is_file()
            actual_bytes = path.stat().st_size if exists else None
            size_matches = actual_bytes == expected_bytes
            file_report: dict[str, object] = {
                "path": str(path),
                "exists": exists,
                "dtype": np.dtype(dtype).name,
                "expected_bytes": expected_bytes,
                "actual_bytes": actual_bytes,
                "size_matches": size_matches,
            }
            if exists and size_matches:
                file_report["sample_check"] = inspect_memmap(
                    path,
                    dtype=dtype,
                    num_samples=int(split["num_samples"]),
                    shape=shape,
                    checks=args.sample_checks,
                    nonnegative=kind == "diffraction",
                )
                sample_check = file_report["sample_check"]
                file_passed = bool(
                    sample_check["all_finite"] and sample_check["all_nonnegative"]
                )
                if args.sha256:
                    LOGGER.info("Calculating SHA256 for %s; this may take several minutes.", path)
                    file_report["sha256"] = file_sha256(path)
            else:
                file_passed = False
            file_report["passed"] = file_passed
            passed &= file_passed
            split_checks[kind] = file_report
            LOGGER.info(
                "%s/%s | expected=%.3f GiB | actual=%s | %s",
                split_name,
                kind,
                expected_bytes / 1024**3,
                f"{actual_bytes / 1024**3:.3f} GiB" if actual_bytes is not None else "missing",
                "PASS" if file_passed else "FAIL",
            )
        checks[split_name] = split_checks

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "runtime": runtime_manifest(),
        "data_manifest": manifest,
        "checks": checks,
    }
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        LOGGER.info("Wrote data validation report: %s", output)
    LOGGER.info("Data validation: %s", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
