from __future__ import annotations

import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from simulation.author_generator import (
    DEFAULT_AUTHOR_CODE_DIR,
    AuthorGeneratedSample,
    AuthorParticle,
    _author_amplitude_object,
    _author_phase_object,
    _create_rotated_q_grid,
    create_paper_particle,
    fft_fhkl_thread,
    generate_paper_observation,
    nufft_fhkl_thread,
    paper_category_for_index,
    load_author_modules,
    save_author_sample,
)


def test_fft_fhkl_matches_direct_atomic_sum() -> None:
    lattice = 4.08
    nstep = 8
    size_q = 6
    reciprocal_axis = (
        1.0 / lattice
        + (np.arange(size_q) - size_q // 2) / (lattice * nstep)
    )
    qx, qy, qz = np.meshgrid(reciprocal_axis, reciprocal_axis, reciprocal_axis)
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [lattice / 2, lattice / 2, 0.0],
            [lattice / 2, 0.0, lattice / 2],
            [0.0, lattice / 2, lattice / 2],
            [lattice, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    positions += np.asarray([lattice / 4, -lattice / 4, lattice / 4])
    actual, _ = fft_fhkl_thread(
        qx,
        qy,
        qz,
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
    )
    expected = np.zeros_like(actual, dtype=np.complex128)
    for atom in positions:
        expected += np.exp(
            -2.0j
            * np.pi
            * (qx * atom[0] + qy * atom[1] + qz * atom[2])
        )
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_nufft_fhkl_matches_direct_sum_on_rotated_grid() -> None:
    lattice = 4.08
    size_q = 6
    dq = 1.0 / (lattice * 10)
    angle = np.deg2rad(31.0)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    axis = (np.arange(size_q) - size_q // 2) * dq
    dqx, dqy, dqz = np.meshgrid(axis, axis, axis)
    rotated = np.einsum(
        "ij,...j->...i",
        rotation,
        np.stack((dqx, dqy, dqz), axis=-1),
    )
    qx = 1.0 / lattice + rotated[..., 0]
    qy = 1.0 / lattice + rotated[..., 1]
    qz = 1.0 / lattice + rotated[..., 2]
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [lattice / 2, lattice / 2, 0.0],
            [lattice, lattice / 2, -lattice / 2],
            [1.2, -0.7, 2.4],
        ]
    )
    actual, _ = nufft_fhkl_thread(
        qx,
        qy,
        qz,
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
    )
    expected = np.zeros_like(actual, dtype=np.complex128)
    for atom in positions:
        expected += np.exp(
            -2.0j
            * np.pi
            * (qx * atom[0] + qy * atom[1] + qz * atom[2])
        )
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_balanced_paper_schedule_crosses_shape_and_phase_families() -> None:
    assignments = [paper_category_for_index(index, 3) for index in range(9)]
    pairs = {(shape, strain) for _, _, shape, strain in assignments}
    assert len(pairs) == 9
    assert assignments[0][2:] == ("wulff", "gauss")
    assert assignments[-1][2:] == ("random", "random")


def test_random_categories_are_reproducible_without_forced_pair_counts() -> None:
    def draw() -> list[tuple[int, int, str, str]]:
        rng = random.Random(20260830)
        shapes: dict[int, str] = {}
        return [paper_category_for_index(index, 3, category_sampling="random",
                                        rng=rng, random_shapes=shapes)
                for index in range(900)]

    assignments = draw()
    assert assignments == draw()
    for start in range(0, 900, 3):
        assert len({item[2] for item in assignments[start:start + 3]}) == 1
    assert any(len({item[3] for item in assignments[start:start + 3]}) < 3
               for start in range(0, 900, 3))
    assert len({(item[2], item[3]) for item in assignments}) == 9


def test_evaluation_generator_is_lazy(monkeypatch) -> None:
    from pathlib import Path
    from simulation import evaluate_author_code as evaluation

    calls: list[int] = []

    def generate(_directory, _modules, seed):
        calls.append(seed)
        return AuthorGeneratedSample(
            np.ones((2, 2, 2), dtype=np.float32),
            np.zeros((2, 2, 2), dtype=np.float32),
            {"shape": "wulff", "strain_argument": "random", "atom_count": 1,
             "nstep": 80, "generation_seconds": 0.1},
        )

    monkeypatch.setattr(evaluation, "generate_notebook_sample", generate)
    monkeypatch.setattr(evaluation, "save_author_sample", lambda *args: None)
    args = SimpleNamespace(seed=123, num_samples=900, profile="notebook",
                           random_q_rotation=None)
    samples = evaluation._generate_samples(args, Path("unused"), (), Path("unused"))
    assert calls == []
    next(samples)
    assert calls == [123]
    next(samples)
    assert calls == [123, 124]
    samples.close()


def test_group_statistics_and_particle_bootstrap() -> None:
    from simulation.evaluate_author_code import _group_distributions, _particle_bootstrap_ci

    rows = [{"particle_index": 0, "shape": "wulff", "phase_wca": 0.2},
            {"particle_index": 0, "shape": "wulff", "phase_wca": 0.4},
            {"particle_index": 1, "shape": "random", "phase_wca": 0.8}]
    groups = _group_distributions(rows, "shape")
    assert groups["wulff"]["count"] == 2
    np.testing.assert_allclose(groups["wulff"]["mean"], 0.3)
    ci = _particle_bootstrap_ci(rows, seed=7)
    np.testing.assert_allclose(ci, [0.3, 0.8])


def test_paper_evaluator_uses_manifest_val_and_test_without_particle_leakage(tmp_path):
    from simulation.evaluate_paper_model import select_evaluation_samples

    records = []
    for index, (split, particle_seed) in enumerate(
        (("train", 10), ("val", 20), ("val", 21), ("test", 30))
    ):
        filename = f"sample_{index:05d}.npz"
        (tmp_path / filename).touch()
        records.append(
            {
                "filename": filename,
                "split": split,
                "metadata": {"particle_seed": particle_seed, "shape": "wulff"},
            }
        )
    manifest = {
        "split_unit": "particle",
        "splits": {"train": 1, "val": 2, "test": 1},
        "samples": records,
    }
    (tmp_path / "dataset_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        calibration_split="val",
        evaluation_split="test",
        num_samples=0,
        calibration_fraction=0.5,
    )
    paths, calibration_count, loaded_manifest, split_rule = (
        select_evaluation_samples(tmp_path.resolve(), args)
    )
    assert [path.name for path in paths] == [
        "sample_00001.npz",
        "sample_00002.npz",
        "sample_00003.npz",
    ]
    assert calibration_count == 2
    assert loaded_manifest == manifest
    assert "val calibrates" in split_rule
    assert "test reports" in split_rule


def test_coverage_lists_all_nine_pairs_even_for_small_random_sample():
    from simulation.evaluate_author_code import _category_coverage

    actual = _category_coverage([{"shape": "wulff", "strain": "gauss"},
                                 {"shape": "wulff", "strain": "gauss"},
                                 {"shape": "winterbottom", "strain": "random"}])
    assert len(actual) == 3
    assert sum(len(counts) for counts in actual.values()) == 9
    assert actual["wulff"]["gauss"] == 2
    assert actual["winterbottom"]["random"] == 1
    assert actual["random"]["cosine"] == 0
    assert sum(sum(counts.values()) for counts in actual.values()) == 3


def test_summary_reports_observed_pairs_and_rotation_without_hiding_empty_pairs(tmp_path):
    from simulation.evaluate_author_code import (
        _category_coverage, _distribution, _group_distributions, _oversampling_groups,
        _write_summary,
    )

    rows = [{"shape": "wulff", "strain": "gauss", "shape_phase_pair": "wulff+gauss",
             "phase_wca": 0.2}]
    report = {
        "generator": {"profile": "paper", "category_sampling": "random",
                      "generator_protocol": "author_calls_v2", "random_q_rotation": True,
                      "oversampling_policy": "record"},
        "num_samples": 1, "num_particles": 1, "num_shape_phase_pairs": 1,
        "random_q_rotation_count": 1, "wca": _distribution([0.2]),
        "wca_particle_bootstrap_95_ci": [0.2, 0.2],
        "category_coverage": _category_coverage(rows),
        "oversampling_groups": _oversampling_groups(rows),
        "groups": {key: _group_distributions(rows, key)
                   for key in ("shape", "strain", "shape_phase_pair")},
    }
    destination = tmp_path / "summary.md"
    _write_summary(destination, report)
    summary = destination.read_text(encoding="utf-8")
    assert "Observed shape/phase combinations: `1`" in summary
    assert "enabled for 1/1 observations" in summary
    assert "| wulff | 1 | 0 | 0 |" in summary
    assert "| winterbottom | 0 | 0 | 0 |" in summary
    assert "| random | 0 | 0 | 0 |" in summary
    assert "| not_measured | 1 | N/A |" in summary


def test_oversampling_diagnostic_keeps_invalid_and_unmeasured_separate():
    from simulation.evaluate_author_code import _oversampling_groups

    groups = _oversampling_groups([
        {"satisfies_paper_oversampling": True, "phase_wca": 0.2},
        {"satisfies_paper_oversampling": False, "phase_wca": 0.8},
        {"phase_wca": 0.5},
    ])
    assert groups["meets_paper_condition"]["count"] == 1
    assert groups["meets_paper_condition"]["mean"] == 0.2
    assert groups["violates_paper_condition"]["count"] == 1
    assert groups["violates_paper_condition"]["mean"] == 0.8
    assert groups["not_measured"] == {"count": 1}


@pytest.mark.parametrize("strain", ["gauss", "cosine", "random"])
def test_phase_adapter_defers_parameter_sampling_to_source(strain) -> None:
    calls = []
    amplitude = np.ones((4, 4, 4), dtype=np.complex128)
    expected = amplitude * np.exp(1j * np.linspace(-2.5, 2.5, 64).reshape(4, 4, 4))

    def phase_function(obj, **kwargs):
        calls.append(kwargs)
        return expected

    def remove_ramp(obj, **kwargs):
        assert obj is expected
        calls.append(kwargs)
        return obj

    source = SimpleNamespace(
        compute_oversampling_ratio=lambda obj: np.array([3., 3., 3.]),
        replace_phase_by_random_phase=phase_function,
        simulate_strain_gauss=phase_function,
        simulate_strain_cosine=phase_function,
        remove_phase_ramp=remove_ramp,
    )
    actual, metadata = _author_phase_object(source, amplitude, strain)
    assert actual is expected
    assert len(calls) == 2
    assert not {"phase_range", "phase_range1", "phase_range2", "sigma1", "sigma2"} & calls[0].keys()
    if strain == "random":
        assert calls[0]["correlation_length"] == 1 / (3 * 2.4)
    assert metadata["rescaled_after_source"] is False
    assert metadata["unwrapped_phase_available"] is False
    assert "target_peak_to_peak_pi" not in metadata


@pytest.fixture(scope="module")
def supplied_author_modules():
    return load_author_modules(supplied_author_directory())


def supplied_author_directory() -> Path:
    return Path(os.environ.get("HIGH_STRAIN_AUTHOR_CODE_DIR") or DEFAULT_AUTHOR_CODE_DIR)


@pytest.fixture(scope="module")
def supplied_diffraction_module(supplied_author_modules):
    return supplied_author_modules[1]


def assert_rng_state_matches(numpy_state, python_state):
    actual = np.random.get_state()
    np.testing.assert_array_equal(actual[1], numpy_state[1])
    assert actual[0] == numpy_state[0]
    assert actual[2:] == numpy_state[2:]
    assert random.getstate() == python_state


@pytest.mark.parametrize("shape,seed", [
    ("wulff", 42), ("winterbottom", 42), ("random", 42), ("random", 36),
])
def test_particle_construction_calls_source_unchanged(supplied_author_modules, shape, seed):
    directory = supplied_author_directory()
    np.random.seed(seed)
    random.seed(seed)
    # Seed 36 exercises the source's zero cut-direction draw; do not redraw it.
    with np.errstate(invalid="ignore"):
        expected = supplied_author_modules[0].ShapedParticle(
            shape, shape_parameters=None, NL=[40, 40, 40], element="Au",
            path_potential=str(directory / "Main_files" / "pot") + os.sep,
            print_mode="silent",
        )
    numpy_state, python_state = np.random.get_state(), random.getstate()
    with np.errstate(invalid="ignore"):
        actual = create_paper_particle(directory, supplied_author_modules, seed, shape, 0)
    np.testing.assert_array_equal(actual.positions, expected.u)
    assert_rng_state_matches(numpy_state, python_state)


@pytest.mark.parametrize("rotation", [False, True])
@pytest.mark.parametrize("seed", [42, 20260830])
def test_reciprocal_grid_calls_source_unchanged(supplied_diffraction_module, rotation, seed):
    source = supplied_diffraction_module
    np.random.seed(seed)
    random.seed(seed)
    nstep = int(np.random.randint(80, 160))
    dq = 1.0 / 4.080 / nstep
    expected = source.Createqxqyqz(dq, 6, [1, 1, 1], "Au",
                                   random_rotation=rotation, random_shift=False)
    numpy_state, python_state = np.random.get_state(), random.getstate()
    np.random.seed(seed)
    random.seed(seed)
    actual = _create_rotated_q_grid(source, random_q_rotation=rotation, size_q=6)
    for array, reference in zip(actual[:3], expected):
        np.testing.assert_array_equal(array, reference)
    assert actual[3] == nstep
    assert actual[4] == dq
    np.testing.assert_allclose(actual[5].T @ actual[5], np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(actual[5]), 1.0)
    assert np.allclose(actual[5], np.eye(3)) != rotation
    assert_rng_state_matches(numpy_state, python_state)


@pytest.fixture(scope="module", params=["wulff", "winterbottom", "random"])
def supplied_particle(supplied_author_modules, request):
    return create_paper_particle(supplied_author_directory(),
                                 supplied_author_modules, 42, request.param, 0)


@pytest.mark.parametrize("strain", ["gauss", "cosine", "random"])
def test_all_nine_combinations_match_direct_source_pipeline(
    supplied_author_modules, supplied_particle, strain,
):
    _, source, object_utilities = supplied_author_modules
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    nstep = int(np.random.randint(80, 160))
    grid = source.Createqxqyqz(1.0 / 4.080 / nstep, 64, [1, 1, 1], "Au",
                               random_rotation=True, random_shift=False)
    x, y, z = supplied_particle.positions.T
    scattering = source.Create_diffraction(*grid, x, y, z, [1, 1, 1], "Au",
                                            center_the_center_of_mass=False)
    obj = np.fft.ifftshift(np.fft.fftn(np.fft.fftshift(scattering)))
    centered = object_utilities.center_object(obj)
    expected_intensity, expected_phase = source.add_random_noise(centered, strain=strain)
    numpy_state, python_state = np.random.get_state(), random.getstate()

    actual = generate_paper_observation(supplied_author_modules, supplied_particle,
                                        seed, strain, 0)
    np.testing.assert_array_equal(actual.intensity, expected_intensity.astype(np.float32))
    # Independent multithreaded NUFFT calls differ at float32 roundoff, even
    # when the source pipeline itself is executed twice with the same seed.
    phase_error = np.angle(np.exp(1j * (actual.reciprocal_phase - expected_phase)))
    np.testing.assert_allclose(phase_error, 0.0, rtol=0, atol=1e-6)
    assert actual.metadata["random_q_rotation"] is True
    assert not np.allclose(actual.metadata["q_rotation_matrix"], np.eye(3))
    assert actual.metadata["phase_parameters"]["rescaled_after_source"] is False
    assert actual.metadata["generator_protocol"] == "author_calls_v2"
    assert_rng_state_matches(numpy_state, python_state)


def test_invalid_source_oversampling_is_reported_without_retry(monkeypatch):
    from simulation import author_generator as generator

    calls = []

    def grid(*args, **kwargs):
        calls.append("grid")
        volume = np.ones((4, 4, 4))
        return volume, volume, volume, 80, 1.0 / 4.080 / 80, np.eye(3)

    monkeypatch.setattr(generator, "_create_rotated_q_grid", grid)
    source = SimpleNamespace(
        Create_diffraction=lambda *args, **kwargs: np.ones((4, 4, 4)),
        compute_oversampling_ratio=lambda obj: np.array([1.8, 3., 3.]),
    )
    modules = (None, source, SimpleNamespace(center_object=lambda obj: obj))
    particle = AuthorParticle(np.zeros((1, 3)), {})
    with pytest.raises(RuntimeError, match="not rescaled or retried"):
        generate_paper_observation(modules, particle, 42, "random", 0)
    assert calls == ["grid"]


def test_record_policy_preserves_low_oversampling_source_draw(supplied_author_modules):
    directory = supplied_author_directory()
    particle = create_paper_particle(directory, supplied_author_modules, 21260834,
                                     "winterbottom", 4)
    _, source, object_utilities = supplied_author_modules
    np.random.seed(20260842)
    random.seed(20260842)
    nstep = int(np.random.randint(80, 160))
    grid = source.Createqxqyqz(1.0 / 4.080 / nstep, 64, [1, 1, 1], "Au",
                               random_rotation=True, random_shift=False)
    x, y, z = particle.positions.T
    scattering = source.Create_diffraction(*grid, x, y, z, [1, 1, 1], "Au",
                                            center_the_center_of_mass=False)
    centered = object_utilities.center_object(
        np.fft.ifftshift(np.fft.fftn(np.fft.fftshift(scattering)))
    )
    expected_intensity, expected_phase = source.add_random_noise(centered, strain="cosine")
    numpy_state, python_state = np.random.get_state(), random.getstate()
    actual = generate_paper_observation(supplied_author_modules, particle, 20260842,
                                        "cosine", 0, oversampling_policy="record")
    assert actual.metadata["satisfies_paper_oversampling"] is False
    assert actual.metadata["nstep"] == nstep == 83
    assert actual.metadata["oversampling_policy"] == "record"
    np.testing.assert_array_equal(actual.intensity, expected_intensity.astype(np.float32))
    error = np.angle(np.exp(1j * (actual.reciprocal_phase - expected_phase)))
    np.testing.assert_allclose(error, 0.0, rtol=0, atol=1e-6)
    assert_rng_state_matches(numpy_state, python_state)


@pytest.mark.parametrize("strain", ["gauss", "cosine", "random"])
@pytest.mark.parametrize("seed", [20260830, 42])
def test_perturbations_match_author_add_random_noise(supplied_diffraction_module, strain, seed):
    source = supplied_diffraction_module
    obj = np.zeros((64, 64, 64), dtype=np.complex128)
    obj[20:44, 18:46, 22:42] = 1.0 + 0.1j

    np.random.seed(seed)
    random.seed(seed)
    expected_intensity, expected_phase = source.add_random_noise(obj.copy(), strain=strain)
    expected_numpy_state = np.random.get_state()
    expected_python_state = random.getstate()

    np.random.seed(seed)
    random.seed(seed)
    amplitude, _ = _author_amplitude_object(source, obj.copy())
    phased, _ = _author_phase_object(source, amplitude, strain)
    actual_intensity, actual_phase = source.force_poisson_statistic(phased)

    np.testing.assert_array_equal(actual_intensity, expected_intensity)
    np.testing.assert_array_equal(actual_phase, expected_phase)
    assert_rng_state_matches(expected_numpy_state, expected_python_state)


def test_saved_source_truth_does_not_claim_unwrapped_phase(tmp_path):
    sample = AuthorGeneratedSample(
        intensity=np.ones((4, 4, 4), dtype=np.float32),
        reciprocal_phase=np.zeros((4, 4, 4), dtype=np.float32),
        metadata={"phase_sampling": "author_function_defaults_v1"},
        support=np.ones((4, 4, 4), dtype=bool),
        realspace_object=np.ones((4, 4, 4), dtype=np.complex64),
        clean_intensity=np.ones((4, 4, 4), dtype=np.float32),
    )
    path = tmp_path / "sample.npz"
    save_author_sample(path, sample)
    with np.load(path) as data:
        assert {"I", "phi", "metadata_json", "support", "object", "I_clean"} == set(data.files)
