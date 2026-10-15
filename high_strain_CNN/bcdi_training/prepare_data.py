"""Convert simulator NPZs to physical paired NPYs with seed-disjoint splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import shutil
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")
LOGGER = logging.getLogger("bcdi.prepare")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_from_labels(amplitude, phase, convention):
    """Use the generator's float32 operation order, including author 2*pi."""
    if convention not in ("author", "radians"):
        raise ValueError("A recorded author or radians phase convention is required.")
    factor = 2 * math.pi if convention == "author" else 1.0
    return (
        np.asarray(amplitude, dtype=np.float32)
        * np.exp(1j * factor * np.asarray(phase, dtype=np.float32))
    ).astype(np.complex64)


def modulus_from_object(field):
    return np.abs(
        np.fft.fftshift(
            np.fft.fft2(np.fft.ifftshift(field, axes=(-2, -1))), axes=(-2, -1)
        )
    )


def partition_seeds(seeds, split_seed=42, val_fraction=0.1, test_fraction=0.1):
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError(
            "At least three unique random-field seeds are required for disjoint splits."
        )
    if (
        not 0 < val_fraction < 1
        or not 0 < test_fraction < 1
        or val_fraction + test_fraction >= 1
    ):
        raise ValueError(
            "Validation/test fractions must be positive and sum to less than one."
        )
    val_count = max(1, int(round(len(seeds) * val_fraction)))
    test_count = max(1, int(round(len(seeds) * test_fraction)))
    if val_count + test_count >= len(seeds):
        raise ValueError(
            "Requested fractions leave no training seed; use smaller fractions."
        )
    shuffled = np.random.default_rng(split_seed).permutation(sorted(seeds)).tolist()
    return {
        "val": sorted(shuffled[:val_count]),
        "test": sorted(shuffled[val_count : val_count + test_count]),
        "train": sorted(shuffled[val_count + test_count :]),
    }


def inspect_source(source: Path):
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest["config"]["image_size"] != 512:
        raise ValueError(
            "Source manifest must describe a completed native 512-square generation."
        )
    convention = manifest["config"].get("phase_convention")
    if convention not in ("author", "radians"):
        raise ValueError(
            "The source manifest must explicitly record the phase convention."
        )
    records = manifest["files"]
    count = 0
    for record in records:
        relative = Path(record["file"].replace("\\", "/"))
        path = (source / relative).resolve()
        if not path.is_relative_to(source):
            raise ValueError("Source array path escapes the generation directory.")
        shape = tuple(record["arrays"]["modulus_detector"]["shape"])
        if len(shape) != 3 or shape[0] < 1 or shape[1:] != (512, 512):
            raise ValueError(f"Invalid source shape in {path}.")
        for key in ("object_modulus", "object_phase"):
            if tuple(record["arrays"][key]["shape"]) != shape:
                raise ValueError(
                    f"Source amplitude, phase and diffraction shapes differ in {path}."
                )
        if (
            path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(
                f"Source size/SHA256 differs from the generator manifest: {path}"
            )
        count += shape[0]
    if count != manifest["samples"]:
        raise ValueError("Source sample count does not match its manifest.")
    return manifest


def prepare(
    source,
    output,
    *,
    template=None,
    split_seed=42,
    val_fraction=0.1,
    test_fraction=0.1,
    pair_tolerance=2e-5,
):
    source, output = Path(source).resolve(), Path(output).resolve()
    manifest = inspect_source(source)
    groups = partition_seeds(
        [record["seed"] for record in manifest["files"]],
        split_seed,
        val_fraction,
        test_fraction,
    )
    if output == source or output.is_relative_to(source):
        raise ValueError(
            "Prepared data must be outside the source generation directory."
        )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite nonempty prepared-data directory: {output}"
        )
    counts = {
        split: sum(
            record["arrays"]["modulus_detector"]["shape"][0]
            for record in manifest["files"]
            if record["seed"] in seeds
        )
        for split, seeds in groups.items()
    }
    output.mkdir(parents=True, exist_ok=True)
    required_bytes = sum(counts.values()) * 512 * 512 * 12 + 8192
    if shutil.disk_usage(output).free < required_bytes * 1.1:
        raise OSError(
            f"Prepared arrays need about {required_bytes / 2**30:.2f} GiB plus headroom."
        )
    config = json.loads(
        Path(template or HERE / "configs" / "bcdi_2d.json").read_text(encoding="utf-8")
    )
    config["root"] = "."
    config["diffraction_kind"] = "modulus"
    report = {
        "schema_version": 1,
        "status": "preparing",
        "dataset_name": "bcdi_simulator_512",
        "source_directory": str(source),
        "source_manifest_sha256": sha256_file(source / "manifest.json"),
        "phase_convention": manifest["config"]["phase_convention"],
        "split_seed": split_seed,
        "seed_groups": groups,
        "counts": counts,
        "splits": {},
        "arrays": {},
        "max_fourier_modulus_relative_l2": 0.0,
    }
    report_path = output / "dataset_manifest.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        for split in SPLITS:
            records = sorted(
                (r for r in manifest["files"] if r["seed"] in groups[split]),
                key=lambda r: r["seed"],
            )
            paths = {
                kind: output / config["splits"][split][kind]
                for kind in ("diffraction", "realspace")
            }
            arrays = {
                kind: np.lib.format.open_memmap(
                    path,
                    mode="w+",
                    dtype="float32" if kind == "diffraction" else "complex64",
                    shape=(counts[split], 512, 512),
                )
                for kind, path in paths.items()
            }
            rows, offset = [], 0
            try:
                for record in records:
                    path = source / record["file"].replace("\\", "/")
                    with np.load(path, allow_pickle=False) as shard:
                        amplitude, phase, measured = (
                            shard[key]
                            for key in (
                                "object_modulus",
                                "object_phase",
                                "modulus_detector",
                            )
                        )
                        expected = (
                            record["arrays"]["modulus_detector"]["shape"][0],
                            512,
                            512,
                        )
                        if any(
                            value.shape != expected
                            for value in (amplitude, phase, measured)
                        ):
                            raise ValueError(
                                f"NPZ contents disagree with the recorded shapes: {path}"
                            )
                        for local in range(len(measured)):
                            if (
                                not all(
                                    np.isfinite(value[local]).all()
                                    for value in (amplitude, phase, measured)
                                )
                                or (amplitude[local] < 0).any()
                                or (measured[local] < 0).any()
                            ):
                                raise ValueError(
                                    f"Invalid values: {path} sample={local}"
                                )
                            obj = object_from_labels(
                                amplitude[local],
                                phase[local],
                                report["phase_convention"],
                            )
                            pred = modulus_from_object(obj)
                            norm = float(
                                np.linalg.norm(measured[local].astype(np.float64))
                            )
                            if norm <= 0 or np.abs(obj).max() <= 0:
                                raise ValueError(
                                    f"Zero object/diffraction: {path} sample={local}"
                                )
                            error = float(
                                np.linalg.norm(
                                    pred.astype(np.float64) - measured[local]
                                )
                                / norm
                            )
                            if error > pair_tolerance:
                                raise ValueError(
                                    f"Fourier pairing failed: {path} sample={local} error={error:.6g}; check phase convention."
                                )
                            report["max_fourier_modulus_relative_l2"] = max(
                                report["max_fourier_modulus_relative_l2"], error
                            )
                            arrays["diffraction"][offset] = measured[local]
                            arrays["realspace"][offset] = obj
                            rows.append(
                                {
                                    "index": offset,
                                    "seed": record["seed"],
                                    "source_file": record["file"],
                                    "source_index": local,
                                }
                            )
                            offset += 1
                    LOGGER.info(
                        "%s | source seed=%d | prepared=%d/%d",
                        split,
                        record["seed"],
                        offset,
                        counts[split],
                    )
                for array in arrays.values():
                    array.flush()
            finally:
                for array in arrays.values():
                    array._mmap.close()
            report["splits"][split] = rows
            report["arrays"][split] = {
                kind: {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for kind, path in paths.items()
            }
        report["status"] = "complete"
        report_path.write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
        config["provenance"] = {
            "manifest": "dataset_manifest.json",
            "manifest_sha256": sha256_file(report_path),
            "source_phase_convention": report["phase_convention"],
            "target": "physical complex object used by the simulator FFT",
        }
        (output / "training_config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
    except Exception as error:
        report.update(status="failed", error=str(error))
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise
    LOGGER.info(
        "Prepared counts=%s | maximum pair error=%.3g | config=%s",
        counts,
        report["max_fourier_modulus_relative_l2"],
        output / "training_config.json",
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Generation output directory containing manifest.json and synthetic_data/.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--training-template", default=str(HERE / "configs" / "bcdi_2d.json")
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    prepare(
        args.source_dir,
        args.output_dir,
        template=args.training_template,
        split_seed=args.split_seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
