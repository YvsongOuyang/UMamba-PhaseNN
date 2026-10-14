"""Validated configuration objects for diffraction-data generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FloatRange:
    """Closed interval sampled uniformly during data generation."""

    low: float
    high: float

    def validate(self, name: str, *, nonnegative: bool = False) -> None:
        if self.low > self.high:
            raise ValueError(f"{name}.low must not exceed {name}.high")
        if nonnegative and self.low < 0:
            raise ValueError(f"{name} must be nonnegative")


@dataclass(frozen=True)
class StructureSourceConfig:
    """One atomic-structure source used by the generator."""

    name: str
    path: str | None = None
    builtin_symbol: str | None = "Si"
    crystal_structure: str = "diamond"
    lattice_constant_a: float = 5.431
    cubic: bool = True
    vacancy_symbols: tuple[str, ...] = ()
    vacancy_fraction: FloatRange | None = None

    def validate(self) -> None:
        if not self.name:
            raise ValueError("structure source name must not be empty")
        if self.path is None and self.builtin_symbol is None:
            raise ValueError(
                f"structure source {self.name!r} requires path or builtin_symbol"
            )
        if self.vacancy_fraction is not None:
            self.vacancy_fraction.validate(
                f"structure source {self.name!r} vacancy_fraction", nonnegative=True
            )
            if self.vacancy_fraction.high >= 1:
                raise ValueError("source vacancy_fraction must be below 1")
            if not self.vacancy_symbols:
                raise ValueError("source vacancy_fraction requires vacancy_symbols")
        if len(set(self.vacancy_symbols)) != len(self.vacancy_symbols):
            raise ValueError("source vacancy_symbols must be unique")


@dataclass(frozen=True)
class StructureConfig:
    """Atomic-model randomization parameters."""

    sources: tuple[StructureSourceConfig, ...]
    minimum_lateral_size_a: float
    thickness_a: FloatRange
    in_plane_strain_percent: FloatRange
    shear_percent: FloatRange
    static_displacement_sigma_a: FloatRange
    vacancy_fraction: FloatRange
    source_sampling: str = "random"

    def validate(self) -> None:
        if not self.sources:
            raise ValueError("at least one structure source is required")
        if self.source_sampling not in {"random", "balanced"}:
            raise ValueError("structure.source_sampling must be random or balanced")
        for source in self.sources:
            source.validate()
        if self.minimum_lateral_size_a <= 0:
            raise ValueError("minimum_lateral_size_a must be positive")
        self.thickness_a.validate("structure.thickness_a", nonnegative=True)
        self.in_plane_strain_percent.validate("structure.in_plane_strain_percent")
        self.shear_percent.validate("structure.shear_percent")
        self.static_displacement_sigma_a.validate(
            "structure.static_displacement_sigma_a", nonnegative=True
        )
        self.vacancy_fraction.validate("structure.vacancy_fraction", nonnegative=True)
        if self.vacancy_fraction.high >= 1:
            raise ValueError("structure.vacancy_fraction must be below 1")


@dataclass(frozen=True)
class PhysicsConfig:
    """Multislice and illumination settings."""

    energy_kev: FloatRange
    energy_spread_ev_std: float
    beam_tilt_std_mrad: float
    grid_size: int
    detector_size: int
    slice_thickness_a: float
    projection: str
    parametrization: str
    frozen_phonon_configs: int
    thermal_sigma_a: float
    imaging_mode: str = "plane_wave"

    def validate(self) -> None:
        self.energy_kev.validate("physics.energy_kev", nonnegative=True)
        if self.energy_kev.low <= 0:
            raise ValueError("physics.energy_kev must be positive")
        if self.energy_spread_ev_std < 0 or self.beam_tilt_std_mrad < 0:
            raise ValueError("energy spread and beam tilt standard deviations must be nonnegative")
        if self.grid_size < 32 or self.detector_size < 32:
            raise ValueError("physics grid and detector sizes must be at least 32")
        if self.slice_thickness_a <= 0:
            raise ValueError("physics.slice_thickness_a must be positive")
        if self.frozen_phonon_configs < 1:
            raise ValueError("physics.frozen_phonon_configs must be at least 1")
        if self.thermal_sigma_a < 0:
            raise ValueError("physics.thermal_sigma_a must be nonnegative")
        if self.imaging_mode not in {"plane_wave", "single_shot_ecdi"}:
            raise ValueError("physics.imaging_mode must be plane_wave or single_shot_ecdi")


@dataclass(frozen=True)
class EcdiConfig:
    """Single-shot electron CDI with isolated or extended crystal geometry."""

    field_size_a: float
    aperture_edge_a: float
    object_diameter_a: FloatRange | None = None
    aperture_diameter_a: FloatRange | None = None
    in_plane_rotation_deg: FloatRange = field(default_factory=lambda: FloatRange(0.0, 0.0))
    shape_anisotropy: FloatRange = field(default_factory=lambda: FloatRange(0.0, 0.0))
    edge_roughness_fraction: FloatRange = field(default_factory=lambda: FloatRange(0.0, 0.0))
    smooth_displacement_a: FloatRange = field(default_factory=lambda: FloatRange(0.0, 0.0))
    sample_geometry: str = "isolated_particle"
    illumination_fwhm_a: FloatRange | None = None

    def validate(self, structure: StructureConfig) -> None:
        if self.sample_geometry not in {"isolated_particle", "extended_crystal"}:
            raise ValueError("ecdi.sample_geometry must be isolated_particle or extended_crystal")
        self.in_plane_rotation_deg.validate("ecdi.in_plane_rotation_deg")
        self.shape_anisotropy.validate("ecdi.shape_anisotropy", nonnegative=True)
        self.edge_roughness_fraction.validate("ecdi.edge_roughness_fraction", nonnegative=True)
        self.smooth_displacement_a.validate("ecdi.smooth_displacement_a", nonnegative=True)
        if self.field_size_a <= 0 or self.aperture_edge_a < 0:
            raise ValueError("ecdi.field_size_a must be positive and aperture_edge_a nonnegative")
        if self.sample_geometry == "extended_crystal":
            if self.illumination_fwhm_a is None:
                raise ValueError("ecdi.illumination_fwhm_a is required for extended_crystal")
            self.illumination_fwhm_a.validate("ecdi.illumination_fwhm_a", nonnegative=True)
            if self.illumination_fwhm_a.low <= 0:
                raise ValueError("ecdi.illumination_fwhm_a must be positive")
            if self.illumination_fwhm_a.high > self.field_size_a / 2:
                raise ValueError("Gaussian illumination FWHM must be at most half the field")
            if structure.minimum_lateral_size_a < self.field_size_a:
                raise ValueError("Extended crystal must cover the full computational field")
            return
        if self.object_diameter_a is None or self.aperture_diameter_a is None:
            raise ValueError("Isolated-particle e-CDI requires object and aperture diameters")
        self.object_diameter_a.validate("ecdi.object_diameter_a", nonnegative=True)
        self.aperture_diameter_a.validate("ecdi.aperture_diameter_a", nonnegative=True)
        if self.object_diameter_a.low <= 0 or self.aperture_diameter_a.low <= 0:
            raise ValueError("e-CDI object and aperture diameters must be positive")
        if self.shape_anisotropy.high + self.edge_roughness_fraction.high >= 1:
            raise ValueError("e-CDI shape modulation must keep a positive particle radius")
        largest_particle_diameter = self.object_diameter_a.high * (
            1.0 + self.shape_anisotropy.high + self.edge_roughness_fraction.high
        ) + 2.0 * math.sqrt(2.0) * self.smooth_displacement_a.high
        if largest_particle_diameter >= self.aperture_diameter_a.low:
            raise ValueError("The entire deformed crystal must fit inside the smallest aperture")
        if self.aperture_diameter_a.high > self.field_size_a / 2:
            raise ValueError("Aperture diameter must be at most half the computational field")
        if self.aperture_edge_a * 2 >= self.aperture_diameter_a.low:
            raise ValueError("Aperture edge must be narrower than its radius")
        if largest_particle_diameter > structure.minimum_lateral_size_a:
            raise ValueError("Prepared structure must cover the e-CDI object diameter")


@dataclass(frozen=True)
class PulseConfig:
    """Reduced pulse-train and pulse-to-pulse recovery model."""

    width_ps: FloatRange
    interval_ns: FloatRange
    relaxation_time_ns: FloatRange
    acquisition_time_ms: FloatRange
    dose_e_per_a2: FloatRange
    damage_threshold_e_per_a2: float
    maximum_damage_displacement_sigma_a: float

    def validate(self) -> None:
        for name in (
            "width_ps",
            "interval_ns",
            "relaxation_time_ns",
            "acquisition_time_ms",
            "dose_e_per_a2",
        ):
            getattr(self, name).validate(f"pulse.{name}", nonnegative=True)
        if self.width_ps.low <= 0 or self.interval_ns.low <= 0:
            raise ValueError("pulse width and interval must be positive")
        if self.relaxation_time_ns.low <= 0:
            raise ValueError("pulse relaxation time must be positive")
        if self.damage_threshold_e_per_a2 <= 0:
            raise ValueError("pulse damage threshold must be positive")
        if self.maximum_damage_displacement_sigma_a < 0:
            raise ValueError("maximum damage displacement must be nonnegative")


@dataclass(frozen=True)
class InstrumentConfig:
    """Result-domain detector and liquid-cell response model."""

    dqe: FloatRange
    mtf_sigma_px: FloatRange
    background_fraction: FloatRange
    liquid_ring_radius_fraction: FloatRange
    liquid_ring_sigma_fraction: FloatRange
    center_shift_px: FloatRange
    ellipticity: FloatRange
    rotation_deg: FloatRange
    gain: FloatRange
    offset_counts: FloatRange
    read_noise_sigma_counts: FloatRange
    saturation_counts: FloatRange
    bad_pixel_fraction: FloatRange
    log_normalization_percentile: float

    def validate(self) -> None:
        nonnegative_names = (
            "dqe",
            "mtf_sigma_px",
            "background_fraction",
            "liquid_ring_radius_fraction",
            "liquid_ring_sigma_fraction",
            "gain",
            "offset_counts",
            "read_noise_sigma_counts",
            "saturation_counts",
            "bad_pixel_fraction",
        )
        for name in nonnegative_names:
            getattr(self, name).validate(f"instrument.{name}", nonnegative=True)
        self.center_shift_px.validate("instrument.center_shift_px")
        self.ellipticity.validate("instrument.ellipticity")
        self.rotation_deg.validate("instrument.rotation_deg")
        if self.dqe.high > 1:
            raise ValueError("instrument.dqe cannot exceed 1")
        if self.background_fraction.high >= 1:
            raise ValueError("instrument.background_fraction must be below 1")
        if self.bad_pixel_fraction.high >= 1:
            raise ValueError("instrument.bad_pixel_fraction must be below 1")
        if not 50.0 <= self.log_normalization_percentile <= 100.0:
            raise ValueError(
                "instrument.log_normalization_percentile must be in [50, 100]"
            )


@dataclass(frozen=True)
class OutputConfig:
    """Dataset persistence settings."""

    path: str
    save_exit_wave: bool
    compression: str | None
    compression_level: int | None
    overwrite: bool

    def validate(self) -> None:
        if not self.path:
            raise ValueError("output.path must not be empty")
        if self.compression_level is not None and not 0 <= self.compression_level <= 9:
            raise ValueError("output.compression_level must be in [0, 9]")


@dataclass(frozen=True)
class RunConfig:
    """Execution controls."""

    num_samples: int
    seed: int
    device: str
    log_every: int
    cupy_cache_dir: str

    def validate(self) -> None:
        if self.num_samples < 1:
            raise ValueError("run.num_samples must be at least 1")
        if self.device not in {"cpu", "gpu"}:
            raise ValueError("run.device must be 'cpu' or 'gpu'")
        if self.log_every < 1:
            raise ValueError("run.log_every must be at least 1")


@dataclass(frozen=True)
class SimulationConfig:
    """Complete, validated simulation configuration."""

    run: RunConfig
    structure: StructureConfig
    physics: PhysicsConfig
    pulse: PulseConfig
    instrument: InstrumentConfig
    output: OutputConfig
    ecdi: EcdiConfig | None = None
    source_path: Path | None = field(default=None, compare=False)

    def validate(self) -> None:
        self.run.validate()
        self.structure.validate()
        self.physics.validate()
        self.pulse.validate()
        self.instrument.validate()
        self.output.validate()
        if self.physics.imaging_mode == "single_shot_ecdi":
            if self.ecdi is None:
                raise ValueError("ecdi section is required for single_shot_ecdi mode")
            self.ecdi.validate(self.structure)
            if self.physics.frozen_phonon_configs != 1:
                raise ValueError("Single-shot e-CDI requires one coherent frozen-phonon configuration")
            if self.physics.grid_size != self.physics.detector_size:
                raise ValueError("Single-shot e-CDI requires equal wave and detector grids")
            if self.physics.projection != "finite":
                raise ValueError("Single-shot e-CDI requires finite projected potentials")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("source_path", None)
        for source in data["structure"]["sources"]:
            if source["vacancy_fraction"] is None:
                source.pop("vacancy_fraction")
            if not source["vacancy_symbols"]:
                source.pop("vacancy_symbols")
        if self.ecdi is None:
            data.pop("ecdi")
        elif self.ecdi.sample_geometry == "isolated_particle":
            # Keep hashes of pre-existing isolated-particle datasets stable.
            data["ecdi"].pop("sample_geometry", None)
            data["ecdi"].pop("illumination_fwhm_a", None)
        if self.physics.imaging_mode == "plane_wave":
            data["physics"].pop("imaging_mode")
        return data


def _float_range(data: dict[str, Any], key: str) -> FloatRange:
    value = data[key]
    return FloatRange(low=float(value["low"]), high=float(value["high"]))


def load_config(path: str | Path) -> SimulationConfig:
    """Load and validate a JSON simulation configuration."""

    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    structure_raw = raw["structure"]
    physics_raw = raw["physics"]
    pulse_raw = raw["pulse"]
    instrument_raw = raw["instrument"]

    sources = tuple(
        StructureSourceConfig(
            **{
                **{key: value for key, value in item.items()
                   if key not in {"vacancy_fraction", "vacancy_symbols"}},
                "vacancy_symbols": tuple(item.get("vacancy_symbols", ())),
                "vacancy_fraction": (
                    _float_range(item, "vacancy_fraction")
                    if "vacancy_fraction" in item else None
                ),
            }
        )
        for item in structure_raw["sources"]
    )
    structure = StructureConfig(
        sources=sources,
        minimum_lateral_size_a=float(structure_raw["minimum_lateral_size_a"]),
        thickness_a=_float_range(structure_raw, "thickness_a"),
        in_plane_strain_percent=_float_range(
            structure_raw, "in_plane_strain_percent"
        ),
        shear_percent=_float_range(structure_raw, "shear_percent"),
        static_displacement_sigma_a=_float_range(
            structure_raw, "static_displacement_sigma_a"
        ),
        vacancy_fraction=_float_range(structure_raw, "vacancy_fraction"),
        source_sampling=str(structure_raw.get("source_sampling", "random")),
    )
    physics = PhysicsConfig(
        energy_kev=_float_range(physics_raw, "energy_kev"),
        energy_spread_ev_std=float(physics_raw["energy_spread_ev_std"]),
        beam_tilt_std_mrad=float(physics_raw["beam_tilt_std_mrad"]),
        grid_size=int(physics_raw["grid_size"]),
        detector_size=int(physics_raw["detector_size"]),
        slice_thickness_a=float(physics_raw["slice_thickness_a"]),
        projection=str(physics_raw["projection"]),
        parametrization=str(physics_raw["parametrization"]),
        frozen_phonon_configs=int(physics_raw["frozen_phonon_configs"]),
        thermal_sigma_a=float(physics_raw["thermal_sigma_a"]),
        imaging_mode=str(physics_raw.get("imaging_mode", "plane_wave")),
    )
    pulse = PulseConfig(
        width_ps=_float_range(pulse_raw, "width_ps"),
        interval_ns=_float_range(pulse_raw, "interval_ns"),
        relaxation_time_ns=_float_range(pulse_raw, "relaxation_time_ns"),
        acquisition_time_ms=_float_range(pulse_raw, "acquisition_time_ms"),
        dose_e_per_a2=_float_range(pulse_raw, "dose_e_per_a2"),
        damage_threshold_e_per_a2=float(pulse_raw["damage_threshold_e_per_a2"]),
        maximum_damage_displacement_sigma_a=float(
            pulse_raw["maximum_damage_displacement_sigma_a"]
        ),
    )
    instrument = InstrumentConfig(
        **{
            key: _float_range(instrument_raw, key)
            for key in (
                "dqe",
                "mtf_sigma_px",
                "background_fraction",
                "liquid_ring_radius_fraction",
                "liquid_ring_sigma_fraction",
                "center_shift_px",
                "ellipticity",
                "rotation_deg",
                "gain",
                "offset_counts",
                "read_noise_sigma_counts",
                "saturation_counts",
                "bad_pixel_fraction",
            )
        },
        log_normalization_percentile=float(
            instrument_raw["log_normalization_percentile"]
        ),
    )
    ecdi_raw = raw.get("ecdi")
    ecdi = None if ecdi_raw is None else EcdiConfig(
        field_size_a=float(ecdi_raw["field_size_a"]),
        object_diameter_a=(
            _float_range(ecdi_raw, "object_diameter_a")
            if "object_diameter_a" in ecdi_raw else None
        ),
        aperture_diameter_a=(
            _float_range(ecdi_raw, "aperture_diameter_a")
            if "aperture_diameter_a" in ecdi_raw else None
        ),
        aperture_edge_a=float(ecdi_raw["aperture_edge_a"]),
        sample_geometry=str(ecdi_raw.get("sample_geometry", "isolated_particle")),
        illumination_fwhm_a=(
            _float_range(ecdi_raw, "illumination_fwhm_a")
            if "illumination_fwhm_a" in ecdi_raw else None
        ),
        **{
            key: _float_range(ecdi_raw, key)
            for key in (
                "in_plane_rotation_deg",
                "shape_anisotropy",
                "edge_roughness_fraction",
                "smooth_displacement_a",
            )
            if key in ecdi_raw
        },
    )
    config = SimulationConfig(
        run=RunConfig(**raw["run"]),
        structure=structure,
        physics=physics,
        pulse=pulse,
        instrument=instrument,
        output=OutputConfig(**raw["output"]),
        ecdi=ecdi,
        source_path=config_path,
    )
    config.validate()
    return config
