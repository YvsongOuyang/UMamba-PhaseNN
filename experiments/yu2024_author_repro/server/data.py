"""Corrected data interface; reuse the upstream generator's functions and geometry."""
import argparse
import ast
import hashlib
import json
import logging
import math
from pathlib import Path
import shutil

import numpy as np
import torch
from sklearn.utils import shuffle

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "yu2024-radian-amplitude-v1"
KEYS = ("object_phase", "object_modulus", "modulus_detector")
LOG = logging.getLogger(__name__)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generator_functions():
    path = ROOT / "vendor/Complex-NNphase/Simulated Date/generate_data.py"
    source = path.read_text(encoding="utf-8")
    old = "np.exp(2j*math.pi*np.float32(np.angle(B)))"
    assert source.count(old) == 1, "Upstream generator changed; review patch"
    source = source.replace(old, "np.exp(1j*np.float32(np.angle(B)))")
    tree = ast.parse(source)
    # Import definitions only: do not execute upstream's hard-coded 100-file loop.
    tree.body = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    namespace = {}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


def split_ids(samples=15000, seed=1):
    if samples < 10 or samples % 10:
        raise ValueError("samples must be a positive multiple of 10")
    pool = samples * 9 // 10
    shuffled = shuffle(np.arange(pool), random_state=0)
    order = torch.randperm(pool, generator=torch.Generator().manual_seed(seed)).numpy()
    return dict(train=shuffled[order[:samples * 8 // 10]],
                validation=shuffled[order[samples * 8 // 10:]],
                test=np.arange(pool, samples))


def diffraction(objects):
    return np.abs(np.fft.ifftshift(np.fft.fft2(
        np.fft.fftshift(objects.astype(np.complex128), axes=(-2, -1))), axes=(-2, -1)))


def generate(directory, seeds, samples, seed):
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError("Use an EMPTY corrected dataset directory; old data must not be mixed in")
    if seeds * 196 < samples:
        raise ValueError("Not enough seeds: each seed supplies 196 samples")
    required = seeds * 16_100_000 + 100_000_000
    if shutil.disk_usage(directory).free < required:
        raise OSError(f"Insufficient disk space: need about {required / 2**30:.2f} GiB")
    splits = split_ids(samples, seed)
    fn = generator_functions()
    g1 = fn["generate_ellipse"](1.5, 3, 6, 8)
    g2 = fn["generate_Gaussian_support"](-math.pi * 35 / 180)
    files = []
    for index in range(seeds):
        field = fn["generate_random_phase"](index, 6, 8, g1)
        arrays = fn["Generate_synthetic_data"](field, g2, 1024, 448)
        path = directory / f"data{index}.npz"
        np.savez(path, **dict(zip(KEYS, arrays)))
        files.append(dict(name=path.name, sha256=sha256(path)))
        LOG.info("Generated %s (%d/%d)", path.name, index + 1, seeds)
    np.savez(directory / "split_indices.npz", **splits)
    manifest = dict(schema=SCHEMA, samples=samples, raw_samples=seeds * 196,
                    seed=seed, files=files, split_sha256=sha256(directory / "split_indices.npz"),
                    phase="radians", detector="abs(FFT(A * exp(1j * phase)))",
                    upstream_commit="bcccdef38ca7d0c35699036ceee629fecdfc76f7")
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    verify(directory)


def manifest_and_splits(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Dataset is not the corrected radian/amplitude dataset; regenerate it")
    if sha256(directory / "split_indices.npz") != manifest["split_sha256"]:
        raise ValueError("Split checksum mismatch")
    with np.load(directory / "split_indices.npz", allow_pickle=False) as data:
        splits = {key: data[key].copy() for key in ("train", "validation", "test")}
    expected = split_ids(manifest["samples"], manifest["seed"])
    for key in expected:
        if not np.array_equal(splits[key], expected[key]):
            raise ValueError(f"Invalid {key} split")
    return manifest, splits


def verify(directory):
    manifest, splits = manifest_and_splits(directory)
    worst = 0.0
    for entry in manifest["files"]:
        path = directory / entry["name"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key].astype(np.float32) for key in KEYS}
        for key, array in arrays.items():
            if array.shape != (196, 64, 64) or not np.isfinite(array).all():
                raise ValueError(f"Invalid {key} in {path}")
        if (arrays["object_modulus"] < 0).any() or (arrays["modulus_detector"] < 0).any():
            raise ValueError(f"Negative amplitude in {path}")
        obj = arrays["object_modulus"] * np.exp(1j * arrays["object_phase"])
        observed = arrays["modulus_detector"].astype(np.float64)
        chi = ((diffraction(obj) - observed)**2).sum((-2, -1)) / (observed**2).sum((-2, -1))
        if not np.isfinite(chi).all():
            raise ValueError(f"Non-finite forward error in {path}")
        worst = max(worst, float(chi.max()))
    if not np.isfinite(worst) or worst > 1e-10:
        raise ValueError(f"Forward/label consistency failed: max chi2={worst}")
    result = dict(schema=SCHEMA, max_forward_chi2=worst, checked_raw_samples=manifest["raw_samples"],
                  splits={key: len(value) for key, value in splits.items()})
    (directory / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    LOG.info("Verification: %s", result)
    return result


def load_arrays(directory):
    manifest, splits = manifest_and_splits(directory)
    arrays = {key: np.empty((manifest["samples"], 64, 64), dtype=np.float32) for key in KEYS}
    offset = 0
    for entry in manifest["files"]:
        if offset >= manifest["samples"]:
            break
        path = directory / entry["name"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as data:
            count = min(len(data[KEYS[0]]), manifest["samples"] - offset)
            for key in KEYS:
                arrays[key][offset:offset + count] = data[key][:count]
        offset += count
    if offset != manifest["samples"] or not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Incomplete or non-finite dataset")
    return arrays, splits, manifest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["generate", "verify"])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--samples", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.data_dir.resolve(), args.seeds, args.samples, args.seed)
    else:
        verify(args.data_dir.resolve())
