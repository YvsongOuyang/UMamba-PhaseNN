"""Evaluate the official TensorFlow model on the authors' simulated particles."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import random
import sys
import time
from collections import Counter
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np

from .author_generator import (
    AUTHOR_GENERATOR_PROTOCOL,
    AUTHOR_PHASE_SAMPLING,
    PAPER_SHAPES,
    PAPER_STRAINS,
    AuthorGeneratedSample,
    AuthorParticle,
    author_source_manifest,
    create_paper_particle,
    file_sha256,
    generate_notebook_sample,
    generate_paper_observation,
    load_author_modules,
    paper_category_for_index,
    save_author_sample,
)
from .run_paper_model import (
    _weighted_circular_average,
    prepare_model_input,
    reconstruct_object,
)
from .visualization import save_slice_overview, save_volume_overview


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_DIR / "artifacts" / "models" / "model_paper.h5"
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "artifacts"
    / "evaluations"
    / "simulation_tensorflow"
    / "author_generator"
)
LOGGER = logging.getLogger("high_strain.author_generator_evaluation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--author-code-dir",
        required=True,
        help="Directory containing particle_and_diffraction.ipynb and its Python modules.",
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--profile",
        choices=("notebook", "paper"),
        default="notebook",
        help="Reproduce the executed notebook or enable paper categories with recorded assumptions.",
    )
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--observations-per-particle", type=int, default=3)
    parser.add_argument(
        "--oversampling-policy", choices=("error", "record"), default="error",
        help="Stop on oversampling <=2, or retain unmodified source draws and report violations.",
    )
    parser.add_argument(
        "--category-sampling",
        choices=("balanced", "random"),
        default="random",
        help="Random categories by default; balanced is an explicit branch-coverage diagnostic.",
    )
    parser.add_argument(
        "--random-q-rotation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Defaults to enabled for paper and disabled for notebook.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--visualize-samples", type=int, default=3)
    parser.add_argument("--support-threshold", type=float, default=0.1)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args()
    if args.num_samples < 1 or args.batch_size < 1:
        parser.error("Sample count and batch size must be positive.")
    if args.observations_per_particle < 1:
        parser.error("Observations per particle must be positive.")
    if args.visualize_samples < 0:
        parser.error("Visualization count cannot be negative.")
    if not 0.0 < args.support_threshold < 1.0:
        parser.error("Support threshold must lie in (0, 1).")
    if args.profile == "notebook" and args.random_q_rotation is True:
        parser.error("The notebook profile is fixed to its unrotated executed path.")
    return args


def configure_logging(output_dir: Path, level: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.handlers.clear()
    LOGGER.setLevel(getattr(logging, level))
    LOGGER.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_dir / "evaluation.log", mode="w", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q95": float(np.quantile(array, 0.95)),
    }


def _group_distributions(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(float(row["phase_wca"]))
    return {
        name: {"count": len(values), **_distribution(values)}
        for name, values in sorted(groups.items())
    }


def _category_coverage(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts = Counter((row["shape"], row["strain"]) for row in rows)
    return {shape: {strain: counts[shape, strain] for strain in PAPER_STRAINS}
            for shape in PAPER_SHAPES}


def _oversampling_groups(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {}
    for label, status in (("meets_paper_condition", True), ("violates_paper_condition", False)):
        selected = [row for row in rows if row.get("satisfies_paper_oversampling") is status]
        groups[label] = {"count": len(selected)}
        if selected:
            groups[label].update(_distribution([float(row["phase_wca"]) for row in selected]))
    groups["not_measured"] = {"count": sum(row.get("satisfies_paper_oversampling") is None
                                          for row in rows)}
    return groups


def _particle_bootstrap_ci(rows: list[dict[str, Any]], seed: int) -> list[float]:
    """Resample entire particles, not correlated observations of the same particle."""

    groups: dict[int, list[float]] = {}
    for row in rows:
        groups.setdefault(int(row["particle_index"]), []).append(float(row["phase_wca"]))
    sums = np.asarray([sum(values) for values in groups.values()])
    counts = np.asarray([len(values) for values in groups.values()])
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(2000):
        indices = rng.integers(0, len(groups), size=len(groups))
        means.append(float(sums[indices].sum() / counts[indices].sum()))
    return np.quantile(means, [0.025, 0.975]).tolist()


def _generate_samples(
    args: argparse.Namespace,
    author_code_dir: Path,
    modules: tuple[Any, Any, Any],
    sample_dir: Path,
) -> Iterator[AuthorGeneratedSample]:
    particle_cache: dict[int, AuthorParticle] = {}
    category_rng = random.Random(args.seed + 7919)
    random_shapes: dict[int, str] = {}
    random_q_rotation = (
        args.random_q_rotation
        if args.random_q_rotation is not None
        else args.profile == "paper"
    )
    started = time.perf_counter()
    for index in range(args.num_samples):
        if args.profile == "notebook":
            sample = generate_notebook_sample(
                author_code_dir,
                modules,
                args.seed + index,
            )
        else:
            particle_index, observation_index, shape, strain = paper_category_for_index(
                index,
                args.observations_per_particle,
                category_sampling=args.category_sampling,
                rng=category_rng,
                random_shapes=random_shapes,
            )
            if particle_index not in particle_cache:
                particle_cache.clear()
                particle_cache[particle_index] = create_paper_particle(
                    author_code_dir,
                    modules,
                    args.seed + 1_000_000 + particle_index,
                    shape,
                    particle_index,
                )
            sample = generate_paper_observation(
                modules,
                particle_cache[particle_index],
                args.seed + index,
                strain,
                observation_index,
                random_q_rotation=random_q_rotation,
                oversampling_policy=args.oversampling_policy,
            )
        save_author_sample(sample_dir / f"sample_{index:03d}.npz", sample)
        elapsed = time.perf_counter() - started
        rate = (index + 1) / max(elapsed, 1e-12)
        eta = (args.num_samples - index - 1) / max(rate, 1e-12)
        metadata = sample.metadata
        if metadata.get("satisfies_paper_oversampling") is False:
            LOGGER.warning(
                "Retaining source draw unchanged | sample=%d seed=%d | oversampling=%s",
                index, metadata["seed"], metadata["measured_object_oversampling_xyz"],
            )
        LOGGER.info(
            "Generated %d/%d | %s/%s | atoms=%s | nstep=%d | %.2f s | ETA %.1f s",
            index + 1,
            args.num_samples,
            metadata["shape"],
            metadata["strain_argument"],
            f"{metadata['atom_count']:,}",
            metadata["nstep"],
            metadata["generation_seconds"],
            eta,
        )
        yield sample


def _evaluate_prediction(
    sample: AuthorGeneratedSample,
    prediction: np.ndarray,
    weights: np.ndarray,
    index: int,
    args: argparse.Namespace,
    visualization_dir: Path,
) -> dict[str, Any]:
    center = tuple(size // 2 for size in sample.reciprocal_phase.shape)
    target_phase = sample.reciprocal_phase - float(sample.reciprocal_phase[center])
    direct = _weighted_circular_average(target_phase, prediction, weights)
    inverted = _weighted_circular_average(-target_phase, prediction, weights)
    twin_selected = inverted < direct
    selected_prediction = -prediction if twin_selected else prediction
    row = {
        "index": index,
        "seed": sample.metadata["seed"],
        "particle_index": sample.metadata.get("particle_index", index),
        "observation_index": sample.metadata.get("observation_index", 0),
        "shape": sample.metadata["shape"],
        "strain": sample.metadata["strain_argument"],
        "shape_phase_pair": f"{sample.metadata['shape']}+{sample.metadata['strain_argument']}",
        "atom_count": sample.metadata["atom_count"],
        "nstep": sample.metadata["nstep"],
        "phase_sampling": sample.metadata.get("phase_sampling", "notebook_overrides"),
        "poisson_scale": sample.metadata.get("poisson_scale"),
        "random_q_rotation": sample.metadata["random_q_rotation"],
        "generator_protocol": sample.metadata.get("generator_protocol", "notebook_v1"),
        "satisfies_paper_oversampling": sample.metadata.get("satisfies_paper_oversampling"),
        "min_object_oversampling": min(sample.metadata["measured_object_oversampling_xyz"])
        if "measured_object_oversampling_xyz" in sample.metadata else None,
        "phase_wca": min(direct, inverted),
        "phase_wca_direct": direct,
        "phase_wca_inverted": inverted,
        "twin_flip_selected": twin_selected,
    }
    np.save(visualization_dir / f"sample_{index:03d}_prediction.npy", prediction)
    if index < args.visualize_samples:
        target_object = reconstruct_object(sample.intensity, target_phase)
        predicted_object = reconstruct_object(sample.intensity, selected_prediction)
        save_slice_overview(
            intensity=sample.intensity,
            target_reciprocal_phase=target_phase,
            predicted_reciprocal_phase=selected_prediction,
            target_object=target_object,
            predicted_object=predicted_object,
            destination=visualization_dir / f"sample_{index:03d}_2d.png",
            support_threshold=args.support_threshold,
        )
        save_volume_overview(
            intensity=sample.intensity,
            target_object=target_object,
            predicted_object=predicted_object,
            destination=visualization_dir / f"sample_{index:03d}_3d.png",
            support_threshold=args.support_threshold,
        )
    return row


def _save_wca_overview(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    values = [float(row["phase_wca"]) for row in rows]
    axes[0].hist(values, bins=np.linspace(0, 1, 26), color="#287f83", edgecolor="white")
    axes[0].axvline(np.mean(values), color="#b84848", label=f"Mean {np.mean(values):.4f}")
    axes[0].set(xlabel="Phase WCA (lower is better)", ylabel="Sample count", xlim=(0, 1))
    axes[0].legend()
    families = sorted({str(row["strain"]) for row in rows})
    groups = [[float(row["phase_wca"]) for row in rows if row["strain"] == family]
              for family in families]
    axes[1].boxplot(groups, labels=[f"{family}\nn={len(group)}"
                                   for family, group in zip(families, groups)])
    axes[1].set(ylabel="Phase WCA", ylim=(0, 1), title="By phase family")
    fig.suptitle(f"Official TensorFlow model | {len(rows)} simulated observations")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_summary(path: Path, report: dict[str, Any]) -> None:
    wca = report["wca"]
    profile = report["generator"]["profile"]
    lines = [
        "# Author Generator TensorFlow Evaluation",
        "",
        "## Scope",
        "",
        "- Source: supplied `codes_for_BCDI_dataset_creation` directory.",
        f"- Generator profile: `{profile}`.",
        "- Inference: unchanged official TensorFlow `model_paper.h5`.",
        "- Metric: published sign-symmetric reciprocal-phase WCA.",
        "",
        "## Result",
        "",
        f"- Samples: `{report['num_samples']}`",
        f"- Unique particles: `{report['num_particles']}`",
        f"- Observed shape/phase combinations: `{report['num_shape_phase_pairs']}`",
        f"- WCA mean: `{wca['mean']:.6f}`",
        f"- WCA std: `{wca['std']:.6f}`",
        f"- WCA median: `{wca['median']:.6f}`",
        f"- WCA range: `{wca['min']:.6f}` to `{wca['max']:.6f}`",
        "",
        "## Combination Rule",
        "",
        "The paper describes one of three shapes (Wulff, Winterbottom, random planar cuts) paired with one of three phase families (double Gaussian, double cosine, Gaussian-correlated random). The two terms inside a Gaussian/cosine phase belong to that single field; the three phase families are never stacked together. This yields nine shape/phase combinations, not nine operations. Reciprocal-space rotation is a shared per-observation operation, not a fourth shape or phase family.",
        "",
        "## Compatibility Boundary",
        "",
        "Unrotated FCC structure factors use a regular-grid FFT. Rotated grids use a type-1 NUFFT for the same atomic sum because PyNX is unavailable on this Windows host. Neither is a measured bitwise-parity claim against native PyNX. Source particle construction, q-grid rotation, amplitude perturbation, phase templates, phase-ramp removal, and Poisson sampling run through the supplied modules. Atomic coordinates stay in memory instead of passing through the notebook's eight-decimal LMP serialization. The constant Thomson factor is omitted; normalized intensity is invariant to this scale, but absolute scattering units are not preserved.",
    ]
    if profile == "notebook":
        lines.extend(
            [
                "",
                "The notebook uses Wulff only, draws a random strain label but explicitly passes `strain='random'`, and disables q rotation; this profile preserves that executed behavior rather than covering all nine paper combinations.",
            ]
        )
    else:
        ci = report["wca_particle_bootstrap_95_ci"]
        lines.extend([
            "", "## Sampling Assumptions", "",
            f"- Category sampling: `{report['generator']['category_sampling']}`.",
            f"- Generator protocol: `{report['generator']['generator_protocol']}`.",
            f"- Random q rotation: `{report['generator']['random_q_rotation']}`; enabled for {report['random_q_rotation_count']}/{report['num_samples']} observations.",
            "- Random mode chooses shapes uniformly per particle and phase families uniformly per observation; it does not force pair counts or select by WCA.",
            "- Category probabilities and three observations per particle are reproduction assumptions, not published exact training proportions.",
            "- Phase parameters are sampled inside the supplied author functions with their defaults, including source ramp removal. No support-span rescaling or phase-span filtering is performed.",
            f"- Oversampling policy: `{report['generator']['oversampling_policy']}`. nstep is drawn once from integers [80, 160); no draw or rotation is rescaled or retried. `error` stops at a violation; `record` retains it unchanged, with a warning and an explicit flag.",
            "- The notebook profile separately preserves its explicit parameter overrides; its draws differ from the function defaults.",
            "- The source returns a complex object, not an unwrapped phase field. No exact unwrapped support span is claimed or used for grouping.",
            f"- Mean WCA particle-cluster bootstrap 95% interval: [{ci[0]:.6f}, {ci[1]:.6f}] (2000 replicates).",
        ])
        lines.extend(["", "## Nine-Combination Coverage", "",
                      "Counts below describe the generated data; zero means unsampled, not disabled.",
                      "Rotation is applied per observation when enabled, including every occupied cell.",
                      "", "| Shape | Gaussian | Cosine | Correlated Random |",
                      "| --- | ---: | ---: | ---: |"])
        for shape, counts in report["category_coverage"].items():
            lines.append(f"| {shape} | {counts['gauss']} | {counts['cosine']} | {counts['random']} |")
        lines.extend(["", "## Oversampling Diagnostic", "",
                      "The main result includes every generated observation. The following is a post-hoc diagnostic, not a filtered replacement dataset or proof of the paper's full data distribution.",
                      "", "| Source-draw subset | Count | Mean WCA |",
                      "| --- | ---: | ---: |"])
        for label, stats in report["oversampling_groups"].items():
            mean = f"{stats['mean']:.6f}" if "mean" in stats else "N/A"
            lines.append(f"| {label} | {stats['count']} | {mean} |")
    for label, key in (("Shape", "shape"), ("Phase Family", "strain"),
                       ("Shape x Phase", "shape_phase_pair")):
        lines.extend(["", f"## {label}", "", "| Group | Count | Mean WCA | Median |",
                      "| --- | ---: | ---: | ---: |"])
        for name, stats in report["groups"][key].items():
            lines.append(f"| {name} | {stats['count']} | {stats['mean']:.6f} | {stats['median']:.6f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    author_code_dir = Path(args.author_code_dir).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not author_code_dir.is_dir():
        raise FileNotFoundError(f"Author code directory not found: {author_code_dir}")
    if not model_path.is_file():
        raise FileNotFoundError(f"TensorFlow model not found: {model_path}")
    if (output_dir / "evaluation_results.json").exists() or any(
        (output_dir / "generated_samples").glob("sample_*.npz")
    ):
        raise FileExistsError(f"Output already contains evaluation data: {output_dir}")
    configure_logging(output_dir, args.log_level)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    LOGGER.info("Loading supplied author generator: %s", author_code_dir)
    LOGGER.info(
        "Profile=%s | category sampling=%s | observations/particle=%d",
        args.profile,
        args.category_sampling,
        args.observations_per_particle,
    )
    modules = load_author_modules(author_code_dir)
    sample_dir = output_dir / "generated_samples"
    visualization_dir = output_dir / "visualizations"
    sample_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)
    samples = _generate_samples(args, author_code_dir, modules, sample_dir)

    import tensorflow as tf

    LOGGER.info("Loading official TensorFlow model: %s", model_path)
    model = tf.keras.models.load_model(model_path, compile=False)
    inference_seconds = 0.0
    generation_seconds = 0.0
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    csv_path = output_dir / "evaluation_samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = None
        while batch := list(islice(samples, args.batch_size)):
            model_inputs = np.concatenate(
                [prepare_model_input(sample.intensity) for sample in batch], axis=0
            )
            inference_started = time.perf_counter()
            output = model(model_inputs, training=False).numpy()
            inference_seconds += time.perf_counter() - inference_started
            if output.shape != (len(batch), 64, 64, 64, 1) or not np.all(np.isfinite(output)):
                raise ValueError(f"Invalid TensorFlow output (shape={output.shape}).")
            for sample, prediction, inputs in zip(batch, output[..., 0], model_inputs):
                generation_seconds += float(sample.metadata["generation_seconds"])
                row = _evaluate_prediction(sample, prediction, inputs[..., 0], len(rows),
                                           args, visualization_dir)
                rows.append(row)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
            stream.flush()
            elapsed = time.perf_counter() - started
            LOGGER.info(
                "Evaluated %d/%d | mean WCA %.6f | elapsed %.1f s | ETA %.1f s",
                len(rows), args.num_samples, np.mean([row["phase_wca"] for row in rows]),
                elapsed, elapsed / len(rows) * (args.num_samples - len(rows)),
            )
    wca_distribution = _distribution([float(row["phase_wca"]) for row in rows])
    pair_counts = Counter(f"{row['shape']}+{row['strain']}" for row in rows)
    generator_report: dict[str, Any] = {
        "profile": args.profile,
        "hkl": [1, 1, 1],
        "nstep_interval": "integers [80, 160)",
        "poisson": "source log-uniform scale 10^[3.8, 5.5]"
        if args.profile == "paper"
        else "fixed scale 1e5",
    }
    if args.profile == "paper":
        generator_report.update(
            {
                "shape_families": list(PAPER_SHAPES),
                "phase_families": list(PAPER_STRAINS),
                "category_sampling": args.category_sampling,
                "observations_per_particle": args.observations_per_particle,
                "random_q_rotation": args.random_q_rotation is not False,
                "phase_sampling": AUTHOR_PHASE_SAMPLING,
                "generator_protocol": AUTHOR_GENERATOR_PROTOCOL,
                "oversampling_policy": args.oversampling_policy,
                "phase_span": "not forced; source functions do not expose unwrapped phase",
                "combination": "one shape x one phase per observation",
                "category_counts": dict(sorted(pair_counts.items())),
                "assumptions": [
                    "Uniform category probabilities; exact training proportions are unpublished.",
                    "Three observations per particle unless explicitly changed.",
                    "No selection, rejection, or weighting by model WCA.",
                ],
            }
        )
    report = {
        "route": "simulation_tensorflow/author_generator",
        "author_code_dir": str(author_code_dir),
        "author_source_manifest": author_source_manifest(author_code_dir),
        "generator": generator_report,
        "compatibility": {
            "unrotated_fhkl": "regular FCC-grid FFT evaluation of the same atomic sum",
            "rotated_fhkl": "FINUFFT type-1 evaluation of the same atomic sum",
            "native_pynx_parity": "not measured; compatibility backend in use",
            "coordinate_io": "in-memory positions; bypasses notebook eight-decimal LMP roundtrip",
            "thomson_factor": "constant omitted; cancels during normalization",
            "direct_source_components": [
                "ShapedParticle",
                "Createqxqyqz and RandomdqsRotation",
                "smooth_object and amplitude perturbation",
                "three source phase template functions",
                "phase-ramp removal",
                "Poisson sampling",
            ],
        },
        "model": str(model_path),
        "model_sha256": file_sha256(model_path),
        "model_parameters": int(model.count_params()),
        "tensorflow_version": tf.__version__,
        "python_version": platform.python_version(),
        "device": args.device,
        "seed": args.seed,
        "num_samples": args.num_samples,
        "num_particles": len({row["particle_index"] for row in rows}),
        "num_shape_phase_pairs": len(pair_counts),
        "category_coverage": _category_coverage(rows),
        "random_q_rotation_count": sum(bool(row["random_q_rotation"]) for row in rows),
        "input_preprocessing": "log1p intensity + per-volume min-max, NDHWC",
        "metric": "published sign-symmetric reciprocal-phase WCA",
        "wca": wca_distribution,
        "oversampling_groups": _oversampling_groups(rows),
        "wca_particle_bootstrap_95_ci": _particle_bootstrap_ci(rows, args.seed),
        "groups": {key: _group_distributions(rows, key)
                   for key in ("shape", "strain", "shape_phase_pair")},
        "generation_seconds": generation_seconds,
        "inference_seconds": inference_seconds,
        "total_seconds": time.perf_counter() - started,
        "batch_size": args.batch_size,
        "visualization_support_threshold": args.support_threshold,
        "adapter_source_sha256": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in ("author_generator.py", "evaluate_author_code.py")
        },
        "samples": rows,
    }
    report_path = output_dir / "evaluation_results.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_summary(output_dir / "evaluation_summary.md", report)
    _save_wca_overview(output_dir / "wca_distribution.png", rows)
    LOGGER.info(
        "Complete | WCA mean=%.6f std=%.6f median=%.6f range=[%.6f, %.6f]",
        wca_distribution["mean"],
        wca_distribution["std"],
        wca_distribution["median"],
        wca_distribution["min"],
        wca_distribution["max"],
    )
    LOGGER.info("Saved report: %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
