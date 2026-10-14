"""abTEM multislice backend for coherent electron diffraction."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any

import numpy as np

from .config import EcdiConfig, PhysicsConfig
from .sampling import sample_uniform


@dataclass(frozen=True)
class PhysicsResult:
    """Ideal diffraction intensity and its reference exit wave."""

    intensity: np.ndarray
    exit_wave: np.ndarray
    metadata: dict[str, Any]
    support: np.ndarray | None = None


def _to_numpy(array: Any) -> np.ndarray:
    module_name = type(array).__module__.split(".", maxsplit=1)[0]
    if module_name == "cupy":
        cupy = importlib.import_module("cupy")
        return cupy.asnumpy(array)
    return np.asarray(array)


def _particle_cut_mask(
    relative_xy: np.ndarray,
    diameter_a: float,
    anisotropy: float,
    roughness: float,
    shape_angle_rad: float,
    edge_phase_3_rad: float,
    edge_phase_5_rad: float,
) -> np.ndarray:
    """Select atoms inside a smooth, noncircular finite-particle outline."""

    radius = np.linalg.norm(relative_xy, axis=1)
    theta = np.arctan2(relative_xy[:, 1], relative_xy[:, 0])
    boundary = diameter_a / 2 * (
        1.0
        + anisotropy * np.cos(2 * (theta - shape_angle_rad))
        + roughness
        * (
            0.65 * np.cos(3 * theta + edge_phase_3_rad)
            + 0.35 * np.cos(5 * theta + edge_phase_5_rad)
        )
    )
    return radius <= boundary


class AbtemBackend:
    """Thin, testable wrapper around abTEM's plane-wave multislice solver."""

    def __init__(self, config: PhysicsConfig, device: str, ecdi: EcdiConfig | None = None) -> None:
        import abtem

        self.config = config
        self.device = device
        self.ecdi = ecdi
        fft_backend = "cupy" if device == "gpu" else "numpy"
        abtem.config.set(
            {
                "device": device,
                "fft": fft_backend,
                "diagnostics.progress_bar": False,
                "diagnostics.task_progress": False,
            }
        )
        self.abtem = abtem

    def simulate(self, atoms, rng: np.random.Generator, seed: int) -> PhysicsResult:
        """Propagate a plane electron wave and return a detector-sized pattern."""

        if self.config.imaging_mode == "single_shot_ecdi":
            return self._simulate_ecdi(atoms, rng, seed)

        base_energy_kev = sample_uniform(rng, self.config.energy_kev)
        energy_ev = max(
            base_energy_kev * 1e3
            + float(rng.normal(0.0, self.config.energy_spread_ev_std)),
            1.0,
        )
        tilt = tuple(
            float(value)
            for value in rng.normal(0.0, self.config.beam_tilt_std_mrad, size=2)
        )

        atom_ensemble = self.abtem.FrozenPhonons(
            atoms,
            num_configs=self.config.frozen_phonon_configs,
            sigmas=self.config.thermal_sigma_a,
            ensemble_mean=True,
            seed=seed,
        )
        potential = self.abtem.Potential(
            atom_ensemble,
            gpts=(self.config.grid_size, self.config.grid_size),
            slice_thickness=self.config.slice_thickness_a,
            parametrization=self.config.parametrization,
            projection=self.config.projection,
            device=self.device,
        )
        wave = self.abtem.PlaneWave(
            energy=energy_ev,
            tilt=tilt,
            device=self.device,
        )
        exit_waves = wave.multislice(potential, lazy=False)
        diffraction = exit_waves.diffraction_patterns(
            max_angle="full",
            fftshift=True,
            renormalize=True,
        )
        if tuple(diffraction.shape[-2:]) != (
            self.config.detector_size,
            self.config.detector_size,
        ):
            diffraction = diffraction.interpolate(
                gpts=(self.config.detector_size, self.config.detector_size)
            )

        intensity = _to_numpy(diffraction.array).astype(np.float32, copy=False)
        if intensity.ndim > 2:
            intensity = intensity.reshape((-1,) + intensity.shape[-2:]).mean(axis=0)
        intensity = np.maximum(intensity, 0.0)
        intensity /= max(float(intensity.sum()), np.finfo(np.float32).eps)

        exit_wave = _to_numpy(exit_waves.array)
        if exit_wave.ndim > 2:
            exit_wave = exit_wave.reshape((-1,) + exit_wave.shape[-2:])[0]
        exit_wave = exit_wave.astype(np.complex64, copy=False)

        reciprocal_sampling = tuple(
            float(value) for value in diffraction.sampling
        )
        return PhysicsResult(
            intensity=intensity,
            exit_wave=exit_wave,
            metadata={
                "illumination_mode": "plane_wave",
                "energy_ev": energy_ev,
                "beam_tilt_mrad": tilt,
                "reciprocal_sampling_inv_a": reciprocal_sampling,
                "grid_size": self.config.grid_size,
                "detector_size": self.config.detector_size,
                "slice_thickness_a": self.config.slice_thickness_a,
                "frozen_phonon_configs": self.config.frozen_phonon_configs,
                "thermal_sigma_a": self.config.thermal_sigma_a,
                "device": self.device,
            },
        )

    def _simulate_ecdi(self, atoms, rng: np.random.Generator, seed: int) -> PhysicsResult:
        """A finite nanocrystal selected by a compact exit-plane aperture.

        The aperture is applied to the complex exit wave, before the detector FFT.
        Its support is therefore the exact support of the wave generating the saved
        diffraction pattern. It is an idealized exit-plane aperture, not a claim
        that vacuum transmission behind an unilluminated crystal is zero.
        """

        if self.ecdi is None:
            raise ValueError("e-CDI configuration is missing")
        if self.ecdi.sample_geometry == "extended_crystal":
            return self._simulate_extended_ecdi(atoms, rng, seed)
        ecdi = self.ecdi
        if ecdi.object_diameter_a is None or ecdi.aperture_diameter_a is None:
            raise ValueError("Isolated-particle e-CDI requires object and aperture diameters")
        energy_ev = max(
            sample_uniform(rng, self.config.energy_kev) * 1e3
            + float(rng.normal(0.0, self.config.energy_spread_ev_std)),
            1.0,
        )
        tilt = tuple(
            float(value)
            for value in rng.normal(0.0, self.config.beam_tilt_std_mrad, size=2)
        )
        object_diameter_a = sample_uniform(rng, ecdi.object_diameter_a)
        aperture_diameter_a = sample_uniform(rng, ecdi.aperture_diameter_a)
        rotation_deg = sample_uniform(rng, ecdi.in_plane_rotation_deg)
        anisotropy = sample_uniform(rng, ecdi.shape_anisotropy)
        roughness = sample_uniform(rng, ecdi.edge_roughness_fraction)
        smooth_displacement_a = sample_uniform(rng, ecdi.smooth_displacement_a)
        shape_angle = float(rng.uniform(0, 2 * np.pi)) if anisotropy > 0 else 0.0
        edge_phase_3 = float(rng.uniform(0, 2 * np.pi)) if roughness > 0 else 0.0
        edge_phase_5 = float(rng.uniform(0, 2 * np.pi)) if roughness > 0 else 0.0
        displacement_angle = (
            float(rng.uniform(0, 2 * np.pi)) if smooth_displacement_a > 0 else 0.0
        )
        displacement_phase_x = (
            float(rng.uniform(0, 2 * np.pi)) if smooth_displacement_a > 0 else 0.0
        )
        displacement_phase_y = (
            float(rng.uniform(0, 2 * np.pi)) if smooth_displacement_a > 0 else 0.0
        )

        finite_atoms = atoms.copy()
        original_center = np.asarray(finite_atoms.cell.lengths(), dtype=np.float64)[:2] / 2
        relative_xy = finite_atoms.positions[:, :2] - original_center
        rotation_rad = np.deg2rad(rotation_deg)
        cos_rotation, sin_rotation = np.cos(rotation_rad), np.sin(rotation_rad)
        relative_xy = relative_xy @ np.array(
            [[cos_rotation, sin_rotation], [-sin_rotation, cos_rotation]]
        )
        if smooth_displacement_a > 0:
            cos_field, sin_field = np.cos(displacement_angle), np.sin(displacement_angle)
            field_x = relative_xy[:, 0] * cos_field + relative_xy[:, 1] * sin_field
            field_y = -relative_xy[:, 0] * sin_field + relative_xy[:, 1] * cos_field
            relative_xy += smooth_displacement_a * np.stack(
                (
                    np.sin(2 * np.pi * field_x / object_diameter_a + displacement_phase_x),
                    np.sin(2 * np.pi * field_y / object_diameter_a + displacement_phase_y),
                ),
                axis=1,
            )
        finite_atoms.positions[:, :2] = relative_xy + original_center
        keep = _particle_cut_mask(
            relative_xy,
            object_diameter_a,
            anisotropy,
            roughness,
            shape_angle,
            edge_phase_3,
            edge_phase_5,
        )
        del finite_atoms[np.flatnonzero(~keep)]
        if len(finite_atoms) == 0:
            raise ValueError("The finite e-CDI particle contains no atoms")
        displacement = ecdi.field_size_a / 2 - original_center
        finite_atoms.positions[:, :2] += displacement
        finite_atoms.positions[:, 2] -= float(finite_atoms.positions[:, 2].min()) - 2.0
        cell_z = float(finite_atoms.positions[:, 2].max()) + 2.0
        finite_atoms.set_cell((ecdi.field_size_a, ecdi.field_size_a, cell_z))
        finite_atoms.pbc = False

        atom_ensemble = self.abtem.FrozenPhonons(
            finite_atoms,
            num_configs=1,
            sigmas=self.config.thermal_sigma_a,
            ensemble_mean=True,
            seed=seed,
        )
        potential = self.abtem.Potential(
            atom_ensemble,
            gpts=(self.config.grid_size, self.config.grid_size),
            slice_thickness=self.config.slice_thickness_a,
            parametrization=self.config.parametrization,
            projection="finite",
            periodic=False,
            device=self.device,
        )
        wave = self.abtem.PlaneWave(energy=energy_ev, tilt=tilt, device=self.device)
        exit_waves = wave.multislice(potential, lazy=False)
        full_exit_wave = np.squeeze(_to_numpy(exit_waves.array)).astype(np.complex64)
        if full_exit_wave.shape != (self.config.grid_size, self.config.grid_size):
            raise ValueError(f"Unexpected e-CDI exit-wave shape: {full_exit_wave.shape}")

        grid = self.config.grid_size
        sampling_a = ecdi.field_size_a / grid
        coordinates = (np.arange(grid, dtype=np.float32) - (grid - 1) / 2) * sampling_a
        radius = np.hypot(coordinates[:, None], coordinates[None, :])
        outer = aperture_diameter_a / 2
        edge = ecdi.aperture_edge_a
        if edge == 0:
            aperture = (radius <= outer).astype(np.float32)
        else:
            inner = outer - edge
            fraction = np.clip((outer - radius) / edge, 0.0, 1.0)
            aperture = np.where(
                radius <= inner,
                1.0,
                0.5 - 0.5 * np.cos(np.pi * fraction),
            ).astype(np.float32)
        support = aperture > 0
        exit_wave = full_exit_wave * aperture
        spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(exit_wave)))
        intensity = np.abs(spectrum) ** 2
        intensity = intensity / max(float(intensity.sum()), np.finfo(np.float32).eps)

        return PhysicsResult(
            intensity=intensity.astype(np.float32),
            exit_wave=exit_wave,
            support=support,
            metadata={
                "imaging_mode": "single_shot_ecdi",
                "sample_geometry": "isolated_particle",
                "exact_support": True,
                "illumination_mode": "plane_wave_with_finite_exit_plane_aperture",
                "energy_ev": energy_ev,
                "beam_tilt_mrad": tilt,
                "thermal_sigma_a": self.config.thermal_sigma_a,
                "field_size_a": ecdi.field_size_a,
                "object_diameter_a": object_diameter_a,
                "in_plane_rotation_deg": rotation_deg,
                "shape_anisotropy": anisotropy,
                "edge_roughness_fraction": roughness,
                "shape_angle_rad": shape_angle,
                "edge_phase_3_rad": edge_phase_3,
                "edge_phase_5_rad": edge_phase_5,
                "smooth_displacement_a": smooth_displacement_a,
                "displacement_angle_rad": displacement_angle,
                "displacement_phase_x_rad": displacement_phase_x,
                "displacement_phase_y_rad": displacement_phase_y,
                "aperture_diameter_a": aperture_diameter_a,
                "aperture_edge_a": edge,
                "object_atom_count": len(finite_atoms),
                "real_space_sampling_a": sampling_a,
                "support_fraction": float(support.mean()),
                "linear_oversampling_ratio": ecdi.field_size_a / aperture_diameter_a,
                "grid_size": grid,
                "detector_size": self.config.detector_size,
                "frozen_phonon_configs": 1,
                "device": self.device,
            },
        )

    def _simulate_extended_ecdi(
        self, atoms, rng: np.random.Generator, seed: int
    ) -> PhysicsResult:
        """Multislice an extended crystal under a finite Gaussian incident wave.

        There is no physical particle cut and no exit-plane mask. The Gaussian
        probe is not compact, so the only mathematically exact support is the
        full calculation grid; legacy HIO with a tight exact support is invalid.
        """

        if self.ecdi is None or self.ecdi.illumination_fwhm_a is None:
            raise ValueError("Extended-crystal e-CDI requires an illumination FWHM")
        ecdi = self.ecdi
        grid = self.config.grid_size
        field_a = ecdi.field_size_a
        energy_ev = max(
            sample_uniform(rng, self.config.energy_kev) * 1e3
            + float(rng.normal(0.0, self.config.energy_spread_ev_std)),
            1.0,
        )
        tilt = tuple(
            float(value)
            for value in rng.normal(0.0, self.config.beam_tilt_std_mrad, size=2)
        )
        rotation_deg = sample_uniform(rng, ecdi.in_plane_rotation_deg)
        smooth_displacement_a = sample_uniform(rng, ecdi.smooth_displacement_a)
        beam_fwhm_a = sample_uniform(rng, ecdi.illumination_fwhm_a)

        extended_atoms = atoms.copy()
        original_center = np.asarray(extended_atoms.cell.lengths(), dtype=np.float64)[:2] / 2
        relative_xy = extended_atoms.positions[:, :2] - original_center
        rotation_rad = np.deg2rad(rotation_deg)
        cos_rotation, sin_rotation = np.cos(rotation_rad), np.sin(rotation_rad)
        relative_xy = relative_xy @ np.array(
            [[cos_rotation, sin_rotation], [-sin_rotation, cos_rotation]]
        )
        if smooth_displacement_a > 0:
            displacement_angle = float(rng.uniform(0, 2 * np.pi))
            displacement_phase_x, displacement_phase_y = rng.uniform(0, 2 * np.pi, size=2)
            cos_field, sin_field = np.cos(displacement_angle), np.sin(displacement_angle)
            field_x = relative_xy[:, 0] * cos_field + relative_xy[:, 1] * sin_field
            field_y = -relative_xy[:, 0] * sin_field + relative_xy[:, 1] * cos_field
            relative_xy += smooth_displacement_a * np.stack(
                (
                    np.sin(2 * np.pi * field_x / beam_fwhm_a + displacement_phase_x),
                    np.sin(2 * np.pi * field_y / beam_fwhm_a + displacement_phase_y),
                ),
                axis=1,
            )
        extended_atoms.positions[:, :2] = relative_xy + field_a / 2
        inside_cell = np.all(
            (extended_atoms.positions[:, :2] >= 0)
            & (extended_atoms.positions[:, :2] < field_a),
            axis=1,
        )
        del extended_atoms[np.flatnonzero(~inside_cell)]
        if len(extended_atoms) == 0:
            raise ValueError("Extended crystal contains no atoms inside the calculation cell")
        extended_atoms.positions[:, 2] -= float(extended_atoms.positions[:, 2].min()) - 2.0
        cell_z = float(extended_atoms.positions[:, 2].max()) + 2.0
        extended_atoms.set_cell((field_a, field_a, cell_z))
        extended_atoms.pbc = False

        atom_ensemble = self.abtem.FrozenPhonons(
            extended_atoms,
            num_configs=1,
            sigmas=self.config.thermal_sigma_a,
            ensemble_mean=True,
            seed=seed,
        )
        potential = self.abtem.Potential(
            atom_ensemble,
            gpts=(grid, grid),
            slice_thickness=self.config.slice_thickness_a,
            parametrization=self.config.parametrization,
            projection="finite",
            periodic=False,
            device=self.device,
        )
        coordinates = (np.arange(grid, dtype=np.float32) - (grid - 1) / 2) * field_a / grid
        radius_squared = coordinates[:, None] ** 2 + coordinates[None, :] ** 2
        illumination = np.exp(
            -4 * np.log(2) * radius_squared / beam_fwhm_a**2
        ).astype(np.float32)
        incident_plane = self.abtem.PlaneWave(
            extent=field_a,
            gpts=(grid, grid),
            energy=energy_ev,
            tilt=tilt,
            device=self.device,
        ).build(lazy=False)
        array_module = importlib.import_module("cupy") if self.device == "gpu" else np
        incident_array = incident_plane.array * array_module.asarray(illumination)
        incident_wave = self.abtem.Waves(
            array=incident_array,
            energy=energy_ev,
            extent=field_a,
        )
        exit_waves = incident_wave.multislice(potential)
        exit_wave = np.squeeze(_to_numpy(exit_waves.array)).astype(np.complex64)
        if exit_wave.shape != (grid, grid):
            raise ValueError(f"Unexpected extended e-CDI exit-wave shape: {exit_wave.shape}")
        spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(exit_wave)))
        intensity = np.abs(spectrum) ** 2
        intensity /= max(float(intensity.sum()), np.finfo(np.float32).eps)
        return PhysicsResult(
            intensity=intensity.astype(np.float32),
            exit_wave=exit_wave,
            support=np.ones((grid, grid), dtype=bool),
            metadata={
                "imaging_mode": "single_shot_ecdi",
                "sample_geometry": "extended_crystal",
                "exact_support": False,
                "illumination_mode": "finite_gaussian_incident_wave",
                "illumination_amplitude_fwhm_a": beam_fwhm_a,
                "effective_illumination_area_a2": float(
                    np.sum(illumination**2) * (field_a / grid) ** 2
                ),
                "energy_ev": energy_ev,
                "beam_tilt_mrad": tilt,
                "thermal_sigma_a": self.config.thermal_sigma_a,
                "field_size_a": field_a,
                "in_plane_rotation_deg": rotation_deg,
                "smooth_displacement_a": smooth_displacement_a,
                "simulated_atom_count": len(extended_atoms),
                "real_space_sampling_a": field_a / grid,
                "support_fraction": 1.0,
                "grid_size": grid,
                "detector_size": self.config.detector_size,
                "frozen_phonon_configs": 1,
                "device": self.device,
            },
        )
