"""End-to-end electron diffraction dataset generation pipeline."""

from __future__ import annotations

import importlib.metadata
import logging
from pathlib import Path
import time

import numpy as np

from .config import SimulationConfig
from .instrument import render_detector_frame, sample_instrument
from .physics import AbtemBackend
from .pulse import sample_pulse
from .storage import H5DatasetWriter
from .structures import prepare_structure


LOGGER = logging.getLogger(__name__)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _resolve_output_path(config: SimulationConfig, working_directory: Path) -> Path:
    output_path = Path(config.output.path)
    if not output_path.is_absolute():
        output_path = (working_directory / output_path).resolve()
    return output_path


def generate_dataset(
    config: SimulationConfig, working_directory: Path | None = None
) -> Path:
    """Generate or resume a configured HDF5 diffraction dataset."""

    working_directory = (working_directory or Path.cwd()).resolve()
    output_path = _resolve_output_path(config, working_directory)
    backend = AbtemBackend(config.physics, config.run.device, config.ecdi)
    start_time = time.perf_counter()

    with H5DatasetWriter(output_path, config) as writer:
        pending = writer.pending_indices()
        LOGGER.info(
            "Starting diffraction generation: output=%s pending=%d device=%s",
            output_path,
            len(pending),
            config.run.device,
        )
        for completed, index in enumerate(pending, start=1):
            sample_start = time.perf_counter()
            seed_sequence = np.random.SeedSequence([config.run.seed, index])
            sample_seed = int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
            rng = np.random.default_rng(seed_sequence)

            source_index = (
                index % len(config.structure.sources)
                if config.structure.source_sampling == "balanced"
                else int(rng.integers(0, len(config.structure.sources)))
            )
            source = config.structure.sources[source_index]
            pulse = sample_pulse(config.pulse, rng)
            prepared = prepare_structure(
                source,
                config.structure,
                pulse,
                rng,
                working_directory,
            )
            physics = backend.simulate(prepared.atoms, rng, sample_seed)
            instrument_sample = sample_instrument(config.instrument, rng)
            cell_lengths = prepared.metadata["actual_cell_lengths_a"]
            if config.physics.imaging_mode == "single_shot_ecdi":
                if physics.metadata.get("sample_geometry") == "extended_crystal":
                    illuminated_area_a2 = float(
                        physics.metadata["effective_illumination_area_a2"]
                    )
                else:
                    illuminated_area_a2 = float(
                        np.pi * (physics.metadata["aperture_diameter_a"] / 2) ** 2
                    )
            else:
                illuminated_area_a2 = float(cell_lengths[0] * cell_lengths[1])
            observed = render_detector_frame(
                physics.intensity,
                illuminated_area_a2,
                pulse.dose_e_per_a2,
                instrument_sample,
                rng,
            )
            sample_seconds = time.perf_counter() - sample_start
            metadata = {
                "sample_index": index,
                "sample_seed": sample_seed,
                "structure": prepared.metadata,
                "physics": physics.metadata,
                "pulse": pulse.to_dict(),
                "instrument": observed.metadata,
                "runtime_seconds": sample_seconds,
                "software": {
                    "abtem": _package_version("abtem"),
                    "ase": _package_version("ase"),
                    "numpy": _package_version("numpy"),
                    "cupy": _package_version("cupy-cuda12x"),
                },
            }
            writer.write_sample(
                index=index,
                ideal_intensity=physics.intensity,
                observed_counts=observed.observed_counts,
                observed_log=observed.observed_log,
                exit_wave=physics.exit_wave,
                support=physics.support,
                metadata=metadata,
            )
            if completed % config.run.log_every == 0 or completed == len(pending):
                elapsed = time.perf_counter() - start_time
                LOGGER.info(
                    "Generated %d/%d pending samples; index=%d sample=%.2fs elapsed=%.2fs",
                    completed,
                    len(pending),
                    index,
                    sample_seconds,
                    elapsed,
                )

    LOGGER.info("Dataset complete: %s", output_path)
    return output_path
