"""Export a reproducible AutoPhaseNN subset for the shared official-H5 evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch

from pytorch_autophasenn.data import (
    AutoPhaseNNPhaseDataset,
    _amplitude_center_offset,
    reciprocal_phase_from_realspace,
)
from pytorch_autophasenn.management import (
    DEFAULT_DATA_CONFIG,
    build_data_manifest,
    load_data_config,
    require_data_files,
)
from simulation.sample_io import SimulatedSample, save_sample
from simulation.run_paper_model import reconstruct_object


LOGGER = logging.getLogger("high_strain.export_autophasenn")


@torch.no_grad()
def adapt_sample(
    modulus: np.ndarray,
    realspace: np.ndarray,
    *,
    target_support_threshold: float = 0.1,
) -> SimulatedSample:
    """Convert [D,H,W] modulus/object arrays without changing measured intensity.

    The existing fractional-translation phase convention is applied to the clean
    object too, keeping the exported object and reciprocal phase a Fourier pair.
    """

    modulus = np.asarray(modulus, dtype=np.float32)
    realspace = np.asarray(realspace, dtype=np.complex64)
    if modulus.ndim != 3 or modulus.shape != realspace.shape:
        raise ValueError("Modulus and object must have the same 3D shape.")
    if not np.all(np.isfinite(modulus)) or not np.all(np.isfinite(realspace)):
        raise ValueError("Modulus and object must be finite.")
    if np.any(modulus < 0) or not np.any(modulus > 0) or not np.any(np.abs(realspace) > 0):
        raise ValueError("Modulus must be nonnegative; both volumes must be nonempty.")
    if not np.isfinite(target_support_threshold) or not 0 < target_support_threshold < 1:
        raise ValueError("Target support threshold must lie in (0, 1).")

    tensor = torch.from_numpy(np.array(realspace, copy=True))[None]
    phase = reciprocal_phase_from_realspace(tensor)[0].numpy()
    offset = _amplitude_center_offset(tensor)[0].tolist()
    clean_spectrum = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(realspace)))
    clean_modulus = np.abs(clean_spectrum)
    clean_intensity = np.square(clean_modulus).astype(np.float32)
    # Clean truth is shifted by the same Fourier ramp as the phase label. Never
    # reconstruct the target object from noisy measured modulus.
    centered_object = reconstruct_object(clean_intensity, phase)
    amplitude = np.abs(centered_object)
    support = amplitude > target_support_threshold * float(amplitude.max())
    intensity = np.square(modulus, dtype=np.float32)
    if not np.all(np.isfinite(intensity)) or not np.all(np.isfinite(clean_intensity)):
        raise ValueError("Squared modulus overflowed float32.")
    scale = float(np.sum(modulus * clean_modulus) / np.sum(clean_intensity, dtype=np.float64))
    mismatch = float(np.linalg.norm(modulus - scale * clean_modulus) / np.linalg.norm(modulus))
    metadata = {
        "dataset_name": "AutoPhaseNN validation subset",
        "shape_type": "autophasenn",
        "phase_type": "original_autophasenn_phase",
        "source_diffraction_semantics": "measured reciprocal modulus; squared once to get intensity",
        "intensity_peak": float(intensity.max()),
        "intensity_rescaling": "none",
        "additional_noise": "none; retain source observations",
        "translation_canonicalization": "existing fractional amplitude-COM ramp removal",
        "source_amplitude_center_offset_voxels": offset,
        "clean_truth": "original clean Fourier modulus combined with canonicalized phase",
        "support_definition": "abs(canonicalized clean object) > target_threshold * its maximum",
        "target_support_threshold": target_support_threshold,
        "source_modulus_to_clean_fft_best_scale": scale,
        "source_modulus_to_clean_fft_relative_error": mismatch,
        "source_modulus_sha256": hashlib.sha256(modulus.tobytes()).hexdigest(),
        "source_object_sha256": hashlib.sha256(realspace.tobytes()).hexdigest(),
    }
    return SimulatedSample(
        intensity=intensity,
        reciprocal_phase=phase,
        support=support,
        object_phase=np.angle(centered_object).astype(np.float32),
        realspace_object=centered_object,
        clean_intensity=clean_intensity,
        metadata=metadata,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default=str(DEFAULT_DATA_CONFIG))
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--target-support-threshold", type=float, default=0.1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_data_config(args.data_config)
    if args.split not in config["splits"]:
        raise ValueError(f"Unknown split: {args.split}")
    split = config["splits"][args.split]
    total = int(split["num_samples"])
    if not 1 <= args.num_samples <= total:
        raise ValueError(f"--num-samples must lie in [1, {total}].")
    if not np.isfinite(args.target_support_threshold) or not 0 < args.target_support_threshold < 1:
        raise ValueError("Target support threshold must lie in (0, 1).")
    shape = tuple(config["shape"])
    manifest = build_data_manifest(
        config=config, root=args.data_dir, shape=shape,
        diffraction_dtype=config["dtypes"]["diffraction"],
        realspace_dtype=config["dtypes"]["realspace"],
        splits={args.split: split}, input_log_data=True,
    )
    file_status = require_data_files(manifest, require_exact_size=True)
    paths = manifest["splits"][args.split]
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir == Path(args.data_dir).expanduser().resolve():
        raise ValueError("Export into a separate artifact directory, not the source dataset.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use a fresh output directory: {output_dir}")
    dataset = AutoPhaseNNPhaseDataset(
        paths["diffraction"], paths["realspace"], total, shape=shape,
        diffraction_dtype=config["dtypes"]["diffraction"],
        realspace_dtype=config["dtypes"]["realspace"],
        return_diffraction_modulus=True,
    )
    indices = np.random.default_rng(args.seed).choice(total, size=args.num_samples, replace=False)
    records = []
    for position, index in enumerate(indices):
        source = dataset[int(index)]
        sample = adapt_sample(
            source["diffraction"][0].numpy(), source["realspace"].numpy(),
            target_support_threshold=args.target_support_threshold,
        )
        sample.metadata.update({"source_index": int(index), "source_name": source["name"]})
        destination = save_sample(sample, output_dir / f"sample_{position:05d}.npz", save_extras=True)
        records.append({"index": position, "file": destination.name, **sample.metadata})
        LOGGER.info("Exported %d/%d | source index=%d | peak I=%.6g | modulus mismatch=%.4g",
                    position + 1, len(indices), index, sample.intensity.max(),
                    sample.metadata["source_modulus_to_clean_fft_relative_error"])
    result = {
        "dataset_name": "AutoPhaseNN Validation Subset",
        "source_data": manifest,
        "source_file_status": file_status,
        "seed": args.seed,
        "num_samples": len(indices),
        "sampling": "uniform without replacement; seeded draw order preserved in filenames",
        "source_indices": indices.tolist(),
        "target_support_definition": (
            f"abs(clean object after fractional COM centering) > {args.target_support_threshold:g} * its maximum"
        ),
        "save_extras": True,
        "export_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "samples": records,
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    LOGGER.info("Subset and provenance saved to %s", output_dir)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    raise SystemExit(main())
