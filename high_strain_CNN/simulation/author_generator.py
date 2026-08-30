"""Adapters for the particle and diffraction generator supplied by the authors."""

from __future__ import annotations

import importlib
import hashlib
import json
import os
import random
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import fft as scipy_fft

from .sample_io import COMPACT_STORAGE

PAPER_SHAPES = ("wulff", "winterbottom", "random")
PAPER_STRAINS = ("gauss", "cosine", "random")
AUTHOR_PHASE_SAMPLING = "author_function_defaults_v1"
AUTHOR_GENERATOR_PROTOCOL = "author_calls_v2"
DEFAULT_AUTHOR_CODE_DIR = (
    Path(__file__).resolve().parents[1] / "vendor" / "codes_for_BCDI_dataset_creation"
)


@dataclass(frozen=True)
class AuthorParticle:
    """One particle geometry reused for several diffraction observations."""

    positions: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class AuthorGeneratedSample:
    """One model-ready diffraction/phase pair and optional real-space truth."""

    intensity: np.ndarray
    reciprocal_phase: np.ndarray
    metadata: dict[str, Any]
    support: np.ndarray | None = None
    realspace_object: np.ndarray | None = None
    clean_intensity: np.ndarray | None = None


def _regular_step(values: np.ndarray) -> float:
    unique = np.unique(np.asarray(values, dtype=np.float64))
    differences = np.diff(unique)
    differences = differences[differences > np.finfo(np.float64).eps]
    if not differences.size:
        raise ValueError("Reciprocal grid has no finite nonzero step.")
    step = float(np.median(differences))
    if not np.allclose(differences, step, rtol=1e-6, atol=1e-12):
        raise ValueError("FFT compatibility backend requires an unrotated regular grid.")
    return step


def _validate_q_arrays(
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], int, tuple[int, int, int]]:
    q_arrays = tuple(np.asarray(value, dtype=np.float64) for value in (qx, qy, qz))
    if any(value.ndim != 3 or value.shape != q_arrays[0].shape for value in q_arrays):
        raise ValueError("qx, qy, and qz must be equally shaped 3D arrays.")
    if len(set(q_arrays[0].shape)) != 1:
        raise ValueError("The author compatibility backend requires a cubic q grid.")
    size_q = q_arrays[0].shape[0]
    if size_q < 2:
        raise ValueError("The reciprocal grid must contain at least two samples per axis.")
    return q_arrays, size_q, (size_q // 2,) * 3


def _q_grid_geometry(
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Recover Bragg center, spacing, and source rotation from an affine q grid."""

    q_arrays, size_q, center = _validate_q_arrays(qx, qy, qz)
    bragg_center = np.asarray([value[center] for value in q_arrays])
    basis_by_array_axis = []
    for array_axis in range(3):
        neighbor = list(center)
        neighbor[array_axis] += 1 if center[array_axis] + 1 < size_q else -1
        direction = 1.0 if neighbor[array_axis] > center[array_axis] else -1.0
        vector = np.asarray([value[tuple(neighbor)] for value in q_arrays])
        basis_by_array_axis.append((vector - bragg_center) / direction)
    steps = np.asarray([np.linalg.norm(vector) for vector in basis_by_array_axis])
    step = float(np.mean(steps))
    if step <= np.finfo(np.float64).eps or not np.allclose(
        steps, step, rtol=2e-5, atol=1e-12
    ):
        raise ValueError("Reciprocal-grid basis vectors do not have one common spacing.")

    # np.meshgrid's default indexing maps array axes to (qy, qx, qz).
    rotation = np.column_stack(
        (
            basis_by_array_axis[1] / step,
            basis_by_array_axis[0] / step,
            basis_by_array_axis[2] / step,
        )
    )
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=2e-5, atol=2e-5):
        raise ValueError("Reciprocal grid is not a rigidly rotated Cartesian grid.")
    if not np.isclose(abs(np.linalg.det(rotation)), 1.0, rtol=2e-5, atol=2e-5):
        raise ValueError("Reciprocal-grid rotation has an invalid determinant.")

    probe_indices = (
        (0, 0, 0),
        (size_q - 1, 0, size_q // 3),
        (size_q // 3, size_q - 1, size_q - 1),
    )
    for index in probe_indices:
        source_modes = np.asarray(
            [index[1] - center[1], index[0] - center[0], index[2] - center[2]],
            dtype=np.float64,
        )
        expected = bragg_center + step * (rotation @ source_modes)
        actual = np.asarray([value[index] for value in q_arrays])
        if not np.allclose(actual, expected, rtol=2e-5, atol=2e-10):
            raise ValueError("Reciprocal grid is not affine across the sampled cube.")
    return bragg_center, step, rotation


def fft_fhkl_thread(
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    occ: np.ndarray | None = None,
    **_: Any,
) -> tuple[np.ndarray, float]:
    """CPU equivalent of ``Fhkl_thread`` for an unrotated FCC reciprocal grid."""

    started = time.perf_counter()
    q_arrays, size_q, center = _validate_q_arrays(qx, qy, qz)
    bragg_center = np.asarray([value[center] for value in q_arrays])
    if np.any(np.abs(bragg_center) <= 1e-12):
        raise ValueError("Only a nonzero Bragg center is supported.")
    if not np.allclose(np.abs(bragg_center), np.abs(bragg_center[0]), rtol=1e-5):
        raise ValueError("Only the supplied generator's [111] reflection is supported.")

    reciprocal_step = float(np.median([_regular_step(value) for value in q_arrays]))
    lattice = float(1.0 / np.median(np.abs(bragg_center)))
    nstep = int(round(1.0 / (lattice * reciprocal_step)))
    grid_size = 2 * nstep
    if grid_size < size_q:
        raise ValueError("Direct-space FFT grid is smaller than the requested q cube.")

    positions = np.column_stack((x, y, z)).astype(np.float64, copy=False)
    half_lattice_coordinates = positions / (lattice / 2.0)
    lattice_offset = half_lattice_coordinates[0] - np.rint(half_lattice_coordinates[0])
    integer_coordinates = np.rint(
        half_lattice_coordinates - lattice_offset
    ).astype(np.int64)
    residual = float(
        np.max(
            np.abs(half_lattice_coordinates - lattice_offset - integer_coordinates)
        )
    )
    if residual > 2e-4:
        raise ValueError(
            "Atomic positions are not on the FCC half-lattice required by the FFT "
            f"backend (maximum residual {residual:.3g})."
        )

    if occ is None:
        occupancy_weights = np.ones(len(positions), dtype=np.float32)
    else:
        occupancy_weights = np.asarray(occ, dtype=np.float32)
        if occupancy_weights.shape != (len(positions),):
            raise ValueError("Occupancy must have one value per atomic position.")
    bragg_phase = np.exp(-2.0j * np.pi * (positions @ bragg_center))
    density = np.zeros((grid_size,) * 3, dtype=np.complex64)
    indices = tuple((integer_coordinates[:, axis] % grid_size) for axis in range(3))
    np.add.at(
        density,
        indices,
        occupancy_weights.astype(np.complex64) * bragg_phase.astype(np.complex64),
    )
    spectrum = scipy_fft.fftn(density, workers=1, overwrite_x=True)
    centered_frequencies = np.arange(-size_q // 2, size_q - size_q // 2)
    frequency_indices = centered_frequencies % grid_size
    sampled = spectrum[np.ix_(frequency_indices, frequency_indices, frequency_indices)]
    translation_phase = [
        np.exp(
            -2.0j
            * np.pi
            * centered_frequencies
            * lattice_offset[axis]
            / grid_size
        ).astype(np.complex64)
        for axis in range(3)
    ]
    sampled *= (
        translation_phase[0][:, None, None]
        * translation_phase[1][None, :, None]
        * translation_phase[2][None, None, :]
    )
    sampled = np.transpose(sampled, (1, 0, 2)).astype(np.complex64, copy=False)
    return sampled, time.perf_counter() - started


def nufft_fhkl_thread(
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    occ: np.ndarray | None = None,
    **_: Any,
) -> tuple[np.ndarray, float]:
    """Evaluate the atomic structure factor on a rigidly rotated q grid."""

    try:
        import finufft
    except ImportError as error:
        raise RuntimeError(
            "Random reciprocal-space rotation requires finufft. Install "
            "requirements/author-simulation.txt."
        ) from error

    started = time.perf_counter()
    _, size_q, _ = _validate_q_arrays(qx, qy, qz)
    bragg_center, reciprocal_step, rotation = _q_grid_geometry(qx, qy, qz)
    positions = np.ascontiguousarray(
        np.column_stack((x, y, z)).astype(np.float64, copy=False)
    )
    if occ is None:
        occupancy = np.ones(len(positions), dtype=np.float64)
    else:
        occupancy = np.asarray(occ, dtype=np.float64)
        if occupancy.shape != (len(positions),):
            raise ValueError("Occupancy must have one value per atomic position.")

    rotated_positions = positions @ rotation
    source_coordinates = np.ascontiguousarray(
        2.0 * np.pi * reciprocal_step * rotated_positions
    )
    strengths = np.ascontiguousarray(
        occupancy * np.exp(-2.0j * np.pi * (positions @ bragg_center)),
        dtype=np.complex128,
    )
    source_x = np.ascontiguousarray(source_coordinates[:, 0])
    source_y = np.ascontiguousarray(source_coordinates[:, 1])
    source_z = np.ascontiguousarray(source_coordinates[:, 2])
    sampled = finufft.nufft3d1(
        source_x,
        source_y,
        source_z,
        strengths,
        (size_q, size_q, size_q),
        eps=1e-6,
        isign=-1,
        modeord=0,
    )
    sampled = np.transpose(sampled, (1, 0, 2)).astype(np.complex64, copy=False)
    return sampled, time.perf_counter() - started


def compatible_fhkl_thread(*args: Any, **kwargs: Any) -> tuple[np.ndarray, float]:
    """Use the fast FCC FFT for identity orientation and NUFFT otherwise."""

    qx, qy, qz = args[:3]
    _, _, rotation = _q_grid_geometry(qx, qy, qz)
    if np.allclose(rotation, np.eye(3), rtol=1e-6, atol=1e-7):
        return fft_fhkl_thread(*args, **kwargs)
    return nufft_fhkl_thread(*args, **kwargs)


def _install_source_compatibility_modules() -> None:
    pynx = types.ModuleType("pynx")
    scattering = types.ModuleType("pynx.scattering")
    fhkl = types.ModuleType("pynx.scattering.fhkl")
    fthomson = types.ModuleType("pynx.scattering.fthomson")
    cdi = types.ModuleType("pynx.cdi")
    for module in (pynx, scattering, fhkl, fthomson, cdi):
        module._high_strain_compat = True
    fhkl.Fhkl_thread = compatible_fhkl_thread
    # This factor is constant over a volume and disappears during normalization.
    fthomson.f_thomson = lambda *_args, **_kwargs: 1.0
    pynx.scattering = scattering
    scattering.fhkl = fhkl
    scattering.fthomson = fthomson
    pynx.cdi = cdi
    sys.modules.update(
        {
            "pynx": pynx,
            "pynx.scattering": scattering,
            "pynx.scattering.fhkl": fhkl,
            "pynx.scattering.fthomson": fthomson,
            "pynx.cdi": cdi,
        }
    )
    if "ipywidgets" not in sys.modules:
        ipywidgets = types.ModuleType("ipywidgets")
        ipywidgets.interact = lambda function=None, **_kwargs: function
        sys.modules["ipywidgets"] = ipywidgets


def _require_pynx_cuda() -> None:
    """Load real PyNX/PyCUDA; never substitute the compatibility backend."""
    if not sys.platform.startswith("linux"):
        raise RuntimeError("pynx_cuda generation requires a Linux CUDA environment.")
    for name, module in list(sys.modules.items()):
        if name.startswith("pynx") and getattr(module, "_high_strain_compat", False):
            del sys.modules[name]
    try:
        cuda = importlib.import_module("pycuda.driver")
        cuda.init()
        if cuda.Device.count() < 1:
            raise RuntimeError("No CUDA device is visible to PyCUDA.")
        fhkl = importlib.import_module("pynx.scattering.fhkl")
        importlib.import_module("pynx.scattering.fthomson")
        # Exercise native kernel compilation before a long generation run.
        q = np.zeros(32, dtype=np.float32)
        xyz = np.zeros(32, dtype=np.float32)
        fhkl.Fhkl_thread(q, q, q, xyz, xyz, xyz, gpu_name="", language="cuda")
    except Exception as error:
        raise RuntimeError(
            "Native PyNX CUDA preflight failed. Install/test PyNX with PyCUDA "
            "and a working CUDA compiler, or explicitly select --scattering-backend compat."
        ) from error


def load_author_modules(
    author_code_dir: Path, *, scattering_backend: str = "compat"
) -> tuple[Any, Any, Any]:
    required = (
        "particle_and_diffraction.ipynb",
        "ShapedParticle.py",
        "diffraction_noise_functions.py",
        "Main_files/pot/GOLD/Au_GROCHOLA.eam",
    )
    missing = [name for name in required if not (author_code_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Author code directory is missing: {', '.join(missing)}")
    if scattering_backend == "compat":
        _install_source_compatibility_modules()
    elif scattering_backend == "pynx_cuda":
        _require_pynx_cuda()
    else:
        raise ValueError(f"Unknown scattering backend: {scattering_backend}")
    for module_name in (
        "ShapedParticle",
        "diffraction_noise_functions",
        "Object_utilities",
        "Particle",
        "Global_utilities",
        "Plot_utilities",
        "PostProcessing",
        "rotation_matrices",
    ):
        sys.modules.pop(module_name, None)
    sys.path.insert(0, str(author_code_dir))
    try:
        shaped_particle = importlib.import_module("ShapedParticle")
        diffraction = importlib.import_module("diffraction_noise_functions")
        object_utilities = importlib.import_module("Object_utilities")
    finally:
        sys.path.pop(0)
    return shaped_particle, diffraction, object_utilities


def file_sha256(path: Path) -> str:
    """Return a stable content fingerprint for source and generated files."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def author_source_manifest(author_code_dir: Path) -> list[dict[str, Any]]:
    """Fingerprint supplied source and potential resources used by the adapter."""

    paths = sorted(
        [path for path in author_code_dir.iterdir()
         if path.is_file() and path.suffix.lower() in {".py", ".ipynb"}]
        + [path for path in (author_code_dir / "Main_files" / "pot").rglob("*")
           if path.is_file()]
    )
    return [
        {
            "name": path.relative_to(author_code_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in paths
    ]


def save_author_sample(
    path: Path,
    sample: AuthorGeneratedSample,
    *,
    save_extras: bool = True,
    storage: str = "standard",
) -> None:
    """Write legacy I/phi or compact fixed I/object/support, without new noise."""

    if storage not in ("standard", "compact"):
        raise ValueError(f"Unknown storage format: {storage}")
    if storage == "compact":
        if sample.realspace_object is None or sample.support is None:
            raise ValueError("Compact storage requires a clean object and support (paper profile).")
        metadata = {
            **sample.metadata,
            "storage_schema": COMPACT_STORAGE,
            "phase_label": "angle(ifftshift(fftn(fftshift(object)))); no extra centering",
            "object_dtype": "complex64",
            "label_precision": "numerical agreement, not bitwise source-phi equivalence",
        }
        payload = {
            "I": sample.intensity.astype(np.float32, copy=False),
            "object": sample.realspace_object.astype(np.complex64, copy=False),
            "support": sample.support.astype(bool, copy=False),
            "metadata_json": json.dumps(metadata, ensure_ascii=False),
        }
        np.savez_compressed(path, **payload)
        return

    payload: dict[str, Any] = {
        "I": sample.intensity,
        "phi": sample.reciprocal_phase,
        "metadata_json": json.dumps(sample.metadata, ensure_ascii=False),
    }
    if save_extras and sample.support is not None:
        payload["support"] = sample.support
    if save_extras and sample.realspace_object is not None:
        payload["object"] = sample.realspace_object
    if save_extras and sample.clean_intensity is not None:
        payload["I_clean"] = sample.clean_intensity
    np.savez_compressed(path, **payload)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def generate_notebook_sample(
    author_code_dir: Path,
    modules: tuple[Any, Any, Any],
    seed: int,
) -> AuthorGeneratedSample:
    """Execute the effective parameter path in ``particle_and_diffraction.ipynb``."""

    shaped_particle_module, diffraction, object_utilities = modules
    np.random.seed(seed)
    random.seed(seed)
    potential_path = str(author_code_dir / "Main_files" / "pot") + os.sep
    started = time.perf_counter()
    particle = shaped_particle_module.ShapedParticle(
        "wulff",
        shape_parameters=None,
        NL=[40, 40, 40],
        element="Au",
        path_potential=potential_path,
        print_mode="silent",
    )
    positions = np.asarray(particle.u, dtype=np.float64)
    x, y, z = positions.T
    nstep = int(np.random.randint(80, 160))
    lattice = 4.080
    dq = 1.0 / lattice / nstep
    qx, qy, qz = diffraction.Createqxqyqz(
        dq,
        64,
        [1, 1, 1],
        "Au",
        random_rotation=False,
        random_shift=False,
    )
    diffracted_amplitude = diffraction.Create_diffraction(
        qx,
        qy,
        qz,
        x,
        y,
        z,
        [1, 1, 1],
        "Au",
        center_the_center_of_mass=False,
    )
    obj = np.fft.ifftshift(np.fft.fftn(np.fft.fftshift(diffracted_amplitude)))
    obj_centered = object_utilities.center_object(obj)

    unused_strain_draw = str(np.random.choice(["random", "gauss", "cosine"]))
    phase_range1 = float(np.random.uniform(5, 18))
    phase_range2 = float(np.random.uniform(5, 18))
    sigma1 = float(np.random.uniform(100, 800))
    sigma2 = float(np.random.uniform(100, 800))
    intensity, reciprocal_phase = diffraction.add_random_noise(
        obj_centered,
        strain="random",
        corr_phase=None,
        phase_range1=phase_range1,
        phase_range2=phase_range2,
        sigma1=sigma1,
        sigma2=sigma2,
        poisson_noise=True,
        scale_poisson=1e5,
        plot=False,
    )
    intensity = np.asarray(intensity, dtype=np.float32)
    reciprocal_phase = np.asarray(reciprocal_phase, dtype=np.float32)
    _validate_generated_arrays(intensity, reciprocal_phase)
    metadata = {
        "profile": "notebook",
        "seed": seed,
        "shape": "wulff",
        "particle_parameters": _jsonable(particle.shape_parameters),
        "atom_count": int(len(positions)),
        "nstep": nstep,
        "dq_inverse_angstrom": dq,
        "hkl": [1, 1, 1],
        "strain_argument": "random",
        "unused_notebook_strain_draw": unused_strain_draw,
        "phase_range1_rad": phase_range1,
        "phase_range2_rad": phase_range2,
        "sigma1": sigma1,
        "sigma2": sigma2,
        "poisson_scale": 1e5,
        "random_q_rotation": False,
        "generation_seconds": time.perf_counter() - started,
    }
    return AuthorGeneratedSample(intensity, reciprocal_phase, metadata)


def create_paper_particle(
    author_code_dir: Path,
    modules: tuple[Any, Any, Any],
    seed: int,
    shape: str,
    particle_index: int,
) -> AuthorParticle:
    """Create one of the three mutually exclusive particle families in the paper."""

    if shape not in PAPER_SHAPES:
        raise ValueError(f"Unsupported paper shape: {shape}")
    shaped_particle_module, _, _ = modules
    np.random.seed(seed)
    random.seed(seed)
    potential_path = str(author_code_dir / "Main_files" / "pot") + os.sep
    particle = shaped_particle_module.ShapedParticle(
        shape,
        shape_parameters=None,
        NL=[40, 40, 40],
        element="Au",
        path_potential=potential_path,
        print_mode="silent",
    )
    positions = np.asarray(particle.u, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions):
        raise ValueError("The author particle generator returned no valid atomic positions.")
    return AuthorParticle(
        positions=positions,
        metadata={
            "profile": "paper",
            "particle_index": particle_index,
            "particle_seed": seed,
            "shape": shape,
            "particle_parameters": _jsonable(particle.shape_parameters),
            "atom_count": int(len(positions)),
            "generator_protocol": AUTHOR_GENERATOR_PROTOCOL,
            "shape_source": "ShapedParticle with source defaults",
        },
    )


def paper_category_for_index(
    sample_index: int,
    observations_per_particle: int,
    *,
    category_sampling: str = "balanced",
    rng: random.Random | None = None,
    random_shapes: dict[int, str] | None = None,
) -> tuple[int, int, str, str]:
    """Assign one shape per particle and one phase family per observation."""

    if sample_index < 0 or observations_per_particle < 1:
        raise ValueError("Sample index must be nonnegative and observation count positive.")
    particle_index, observation_index = divmod(sample_index, observations_per_particle)
    if category_sampling == "random":
        if rng is None or random_shapes is None:
            raise ValueError("Random sampling requires a category RNG and shape cache.")
        if particle_index not in random_shapes:
            random_shapes[particle_index] = rng.choice(PAPER_SHAPES)
        return (
            particle_index,
            observation_index,
            random_shapes[particle_index],
            rng.choice(PAPER_STRAINS),
        )
    if category_sampling != "balanced":
        raise ValueError(f"Unknown category sampling: {category_sampling}")
    shape = PAPER_SHAPES[particle_index % len(PAPER_SHAPES)]
    strain = PAPER_STRAINS[observation_index % len(PAPER_STRAINS)]
    return particle_index, observation_index, shape, strain


def _create_rotated_q_grid(
    diffraction: Any,
    *,
    random_q_rotation: bool,
    lattice: float = 4.080,
    size_q: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float, np.ndarray]:
    """Call the source grid constructor after the notebook's unmodified nstep draw."""

    nstep = int(np.random.randint(80, 160))
    dq = 1.0 / lattice / nstep
    qx, qy, qz = diffraction.Createqxqyqz(
        dq, size_q, [1, 1, 1], "Au",
        random_rotation=random_q_rotation, random_shift=False,
    )
    # Read the rotation back for provenance without changing the source grid or RNG.
    _, _, rotation = _q_grid_geometry(qx, qy, qz)
    return qx, qy, qz, nstep, dq, rotation


def _author_amplitude_object(
    diffraction: Any,
    obj: np.ndarray,
) -> tuple[np.ndarray, dict[str, str]]:
    obj = diffraction.smooth_object(obj, plot=False)
    obj = diffraction.remove_real_space_module_out_support(
        obj,
        threshold_module=0.1,
        plot=False,
    )
    amplitude = diffraction.add_random_noise_module(
        np.abs(obj),
        plot=False,
    )
    amplitude_object = amplitude * np.exp(1.0j * np.angle(obj))
    return amplitude_object, {
        "sampling": "smooth_object and add_random_noise_module source defaults",
    }


def _author_phase_object(
    diffraction: Any,
    amplitude_object: np.ndarray,
    strain: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use source defaults and the post-phase ramp removal from add_random_noise."""

    if strain not in PAPER_STRAINS:
        raise ValueError(f"Unsupported paper strain family: {strain}")
    support = np.abs(amplitude_object) > 0.3 * float(np.max(np.abs(amplitude_object)))
    if not np.any(support):
        raise RuntimeError("Author amplitude processing produced an empty support.")
    oversampling = np.asarray(
        diffraction.compute_oversampling_ratio(amplitude_object), dtype=np.float64
    )
    correlation_length = float(1.0 / (np.mean(oversampling) * 2.4))
    parameters: dict[str, Any] = {
        "family": strain,
        "sampling": AUTHOR_PHASE_SAMPLING,
        "correlation_length_fraction": correlation_length,
        "rescaled_after_source": False,
        "unwrapped_phase_available": False,
    }
    if strain == "random":
        phased = diffraction.replace_phase_by_random_phase(
            amplitude_object,
            correlation_length=correlation_length,
            plot=False,
        )
        parameters["source_function"] = "replace_phase_by_random_phase"
    else:
        function_name = "simulate_strain_gauss" if strain == "gauss" else "simulate_strain_cosine"
        phased = getattr(diffraction, function_name)(amplitude_object, plot=False)
        parameters["source_function"] = function_name

    # add_random_noise performs this step once after every family function.
    phased = diffraction.remove_phase_ramp(
        phased,
        threshold_module=0.3,
        crop=False,
        return_ramp=False,
        method="fit",
        plot=False,
    )
    return phased, parameters


def generate_paper_observation(
    modules: tuple[Any, Any, Any],
    particle: AuthorParticle,
    seed: int,
    strain: str,
    observation_index: int,
    *,
    random_q_rotation: bool = True,
    oversampling_policy: str = "error",
) -> AuthorGeneratedSample:
    """Generate one paper-profile observation from a reusable source particle."""

    if oversampling_policy not in {"error", "record"}:
        raise ValueError(f"Unknown oversampling policy: {oversampling_policy}")
    _, diffraction, object_utilities = modules
    np.random.seed(seed)
    random.seed(seed)
    started = time.perf_counter()
    positions = particle.positions
    stage_started = time.perf_counter()
    qx, qy, qz, nstep, dq, rotation = _create_rotated_q_grid(
        diffraction,
        random_q_rotation=random_q_rotation,
    )
    stage_seconds = {"q_grid": time.perf_counter() - stage_started}
    stage_started = time.perf_counter()
    diffracted_amplitude = diffraction.Create_diffraction(
        qx,
        qy,
        qz,
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        [1, 1, 1],
        "Au",
        center_the_center_of_mass=False,
    )
    stage_seconds["scattering"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    obj = np.fft.ifftshift(np.fft.fftn(np.fft.fftshift(diffracted_amplitude)))
    obj_centered = object_utilities.center_object(obj)
    measured_oversampling = np.asarray(
        diffraction.compute_oversampling_ratio(obj_centered), dtype=np.float64
    )
    satisfies_paper_oversampling = bool(np.all(measured_oversampling > 2.0))
    if not satisfies_paper_oversampling and oversampling_policy == "error":
        raise RuntimeError(
            "Generated observation violates the paper's oversampling > 2 condition: "
            f"{measured_oversampling.tolist()}; seed={seed}, nstep={nstep}. "
            "The source draw was not rescaled or retried."
        )

    stage_seconds["reconstruction"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    amplitude_object, amplitude_parameters = _author_amplitude_object(
        diffraction, obj_centered
    )
    stage_seconds["amplitude"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    realspace_object, phase_parameters = _author_phase_object(
        diffraction, amplitude_object, strain
    )
    stage_seconds["phase"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    support = np.abs(amplitude_object) > 0.3 * float(np.max(np.abs(amplitude_object)))
    clean_reciprocal = np.fft.ifftshift(
        np.fft.fftn(np.fft.fftshift(realspace_object))
    )
    clean_intensity = np.square(np.abs(clean_reciprocal)).astype(np.float32)
    intensity, reciprocal_phase = diffraction.force_poisson_statistic(
        realspace_object,
        plot=False,
    )
    intensity = np.asarray(intensity, dtype=np.float32)
    reciprocal_phase = np.asarray(reciprocal_phase, dtype=np.float32)
    _validate_generated_arrays(intensity, reciprocal_phase)
    stage_seconds["noise_and_labels"] = time.perf_counter() - stage_started
    metadata = dict(particle.metadata)
    metadata.update(
        {
            "seed": seed,
            "observation_index": observation_index,
            "nstep": nstep,
            "dq_inverse_angstrom": dq,
            "hkl": [1, 1, 1],
            "strain_argument": strain,
            "phase_family": {
                "gauss": "double_gaussian",
                "cosine": "double_cosine",
                "random": "gaussian_correlated",
            }[strain],
            "phase_parameters": phase_parameters,
            "phase_sampling": AUTHOR_PHASE_SAMPLING,
            "amplitude_parameters": amplitude_parameters,
            "poisson_sampling": "force_poisson_statistic source defaults; scale not exposed",
            "random_q_rotation": random_q_rotation,
            "q_rotation_matrix": rotation.tolist(),
            "measured_object_oversampling_xyz": measured_oversampling.tolist(),
            "satisfies_paper_oversampling": satisfies_paper_oversampling,
            "oversampling_policy": oversampling_policy,
            "support_voxels": int(np.count_nonzero(support)),
            "stage_seconds": stage_seconds,
            "generation_seconds": time.perf_counter() - started,
        }
    )
    return AuthorGeneratedSample(
        intensity=intensity,
        reciprocal_phase=reciprocal_phase,
        metadata=metadata,
        support=support.astype(bool),
        realspace_object=realspace_object.astype(np.complex64),
        clean_intensity=clean_intensity,
    )


def _validate_generated_arrays(
    intensity: np.ndarray,
    reciprocal_phase: np.ndarray,
) -> None:
    if intensity.shape != (64, 64, 64) or reciprocal_phase.shape != intensity.shape:
        raise ValueError("Author generator returned an unexpected volume shape.")
    if np.any(intensity < 0) or not np.all(np.isfinite(intensity)):
        raise ValueError("Author generator returned invalid diffraction intensity.")
    if not np.all(np.isfinite(reciprocal_phase)):
        raise ValueError("Author generator returned invalid reciprocal phase.")
