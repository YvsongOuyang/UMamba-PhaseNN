"""Atomic structure loading and physically meaningful randomization."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from .config import StructureConfig, StructureSourceConfig
from .pulse import PulseSample
from .sampling import sample_uniform


@dataclass(frozen=True)
class PreparedStructure:
    """Prepared ASE structure plus traceable generation metadata."""

    atoms: object
    metadata: dict[str, object]


def _load_atoms(source: StructureSourceConfig, working_directory: Path):
    from ase.build import bulk
    from ase.io import read

    if source.path is not None:
        source_path = Path(source.path)
        if not source_path.is_absolute():
            source_path = (working_directory / source_path).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"structure file not found: {source_path}")
        atoms = read(source_path)
        origin = str(source_path)
    else:
        atoms = bulk(
            source.builtin_symbol,
            source.crystal_structure,
            a=source.lattice_constant_a,
            cubic=source.cubic,
        )
        origin = f"ase.bulk:{source.builtin_symbol}:{source.crystal_structure}"
    return atoms, origin


def _apply_in_plane_deformation(
    atoms, strain_percent: float, shear_percent: float
) -> None:
    strain = strain_percent / 100.0
    shear = shear_percent / 100.0
    deformation = np.array(
        [[1.0 + strain, shear, 0.0], [shear, 1.0 - strain, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    atoms.set_cell(np.asarray(atoms.cell) @ deformation.T, scale_atoms=True)


def prepare_structure(
    source: StructureSourceConfig,
    config: StructureConfig,
    pulse: PulseSample,
    rng: np.random.Generator,
    working_directory: Path,
) -> PreparedStructure:
    """Load, tile, strain, damage and defect an atomic model."""

    import abtem

    atoms, origin = _load_atoms(source, working_directory)
    unknown_vacancy_symbols = set(source.vacancy_symbols).difference(atoms.get_chemical_symbols())
    if unknown_vacancy_symbols:
        raise ValueError(
            f"Vacancy symbols not present in {source.name}: {sorted(unknown_vacancy_symbols)}"
        )
    source_formula = atoms.get_chemical_formula(mode="hill", empirical=True)
    atoms = abtem.orthogonalize_cell(atoms)

    strain_percent = sample_uniform(rng, config.in_plane_strain_percent)
    shear_percent = sample_uniform(rng, config.shear_percent)
    _apply_in_plane_deformation(atoms, strain_percent, shear_percent)

    target_thickness = sample_uniform(rng, config.thickness_a)
    lengths = np.asarray(atoms.cell.lengths(), dtype=np.float64)
    repetitions = (
        max(int(math.ceil(config.minimum_lateral_size_a / lengths[0])), 1),
        max(int(math.ceil(config.minimum_lateral_size_a / lengths[1])), 1),
        max(int(math.ceil(target_thickness / lengths[2])), 1),
    )
    atoms = atoms.repeat(repetitions)

    static_sigma = sample_uniform(rng, config.static_displacement_sigma_a)
    total_sigma = math.sqrt(
        static_sigma**2 + pulse.damage_displacement_sigma_a**2
    )
    if total_sigma > 0:
        atoms.positions += rng.normal(0.0, total_sigma, size=atoms.positions.shape)

    vacancy_range = source.vacancy_fraction or config.vacancy_fraction
    vacancy_fraction = sample_uniform(rng, vacancy_range)
    removed_atoms = 0
    if vacancy_fraction > 0 and len(atoms) > 1:
        if source.vacancy_symbols:
            symbols = np.asarray(atoms.get_chemical_symbols())
            candidates = np.flatnonzero(np.isin(symbols, source.vacancy_symbols))
        else:
            candidates = np.arange(len(atoms))
        removed_atoms = min(
            int(round(len(candidates) * vacancy_fraction)),
            len(candidates),
            len(atoms) - 1,
        )
        if removed_atoms:
            indices = np.sort(rng.choice(candidates, size=removed_atoms, replace=False))
            del atoms[indices]
    atoms.wrap()

    cell_lengths = tuple(float(value) for value in atoms.cell.lengths())
    return PreparedStructure(
        atoms=atoms,
        metadata={
            "source_name": source.name,
            "source_origin": origin,
            "source_formula": source_formula,
            "target_thickness_a": target_thickness,
            "actual_cell_lengths_a": cell_lengths,
            "repetitions": repetitions,
            "atom_count": len(atoms),
            "strain_percent": strain_percent,
            "shear_percent": shear_percent,
            "static_displacement_sigma_a": static_sigma,
            "pulse_damage_displacement_sigma_a": pulse.damage_displacement_sigma_a,
            "total_displacement_sigma_a": total_sigma,
            "vacancy_fraction": vacancy_fraction,
            "vacancy_symbols": list(source.vacancy_symbols),
            "removed_atoms": removed_atoms,
        },
    )
