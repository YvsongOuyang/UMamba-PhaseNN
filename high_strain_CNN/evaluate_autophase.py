"""Reconstruct and evaluate real-space objects on AutoPhaseNN memmap data."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pytorch_port.data import AutoPhaseNNPhaseDataset, reciprocal_phase_from_realspace
from pytorch_port.losses import phase_retrieval_wca_components
from pytorch_port.management import (
    DEFAULT_DATA_CONFIG,
    build_data_manifest,
    load_data_config,
    require_data_files,
    runtime_manifest,
)
from pytorch_port.model import (
    MODEL_VARIANTS,
    HighStrainPhaseUNet,
    count_parameters,
    infer_model_variant,
)
from pytorch_port.reconstruction import (
    farfield_modulus_from_realspace,
    realspace_from_modulus_phase,
)


PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = PROJECT_DIR.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from autophasenn_training_pipeline.evaluate import (  # noqa: E402
    materialize_metric_rows,
    metric_statistics,
    post_process_realspace_batch,
    resolve_postprocess_workers,
)
from autophasenn_training_pipeline.losses import (  # noqa: E402
    FIXED_METRIC_DESCRIPTIONS,
    METRIC_DESCRIPTIONS,
    fixed_metric_groups,
    format_fixed_metric_groups,
    group_metrics,
    metric_tensor_dict,
    realspace_metric_tensor_dict,
)


LOGGER = logging.getLogger("high_strain.evaluate_autophase")
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output" / "evaluate_autophase"


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--data-config", default=str(DEFAULT_DATA_CONFIG))
    bootstrap_args, _ = bootstrap.parse_known_args()
    data_config = load_data_config(bootstrap_args.data_config)
    val_config = data_config["splits"]["val"]
    configured_shape = tuple(int(size) for size in data_config["shape"])
    if len(set(configured_shape)) != 1:
        raise ValueError("HighStrainPhaseUNet requires a cubic data shape.")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default=str(DEFAULT_DATA_CONFIG))
    parser.add_argument("--checkpoint", default=str(PROJECT_DIR / "model_paper_pytorch.pt"))
    parser.add_argument("--data-dir", default=data_config["root"])
    parser.add_argument("--data-diff", default=val_config["diffraction"])
    parser.add_argument("--data-real", default=val_config["realspace"])
    parser.add_argument(
        "--num-samples",
        type=int,
        default=int(val_config["num_samples"]),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shape", type=int, default=configured_shape[0])
    parser.add_argument("--dtype-diff", default=data_config["dtypes"]["diffraction"])
    parser.add_argument("--dtype-real", default=data_config["dtypes"]["realspace"])
    parser.add_argument(
        "--input-log-data",
        action=argparse.BooleanOptionalAction,
        default=data_config.get("input_preprocessing", {}).get("transform") == "log1p",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--postprocess-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--model-variant",
        choices=("auto", *MODEL_VARIANTS),
        default="auto",
        help="Infer the architecture from the checkpoint by default.",
    )
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--ssim-window-size", type=int, default=7)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument(
        "--ambiguity-mode",
        choices=("twin_aligned", "raw"),
        default="twin_aligned",
        help=(
            "Use ground truth only during evaluation to resolve the conjugate/twin "
            "sign permitted by the published WCA loss, or reconstruct the raw phase."
        ),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--save-realspace",
        action="store_true",
        help="Write reconstructed complex64 objects as an AutoPhaseNN-style raw memmap.",
    )
    parser.add_argument(
        "--save-reciprocal-phase",
        action="store_true",
        help="Write the selected float32 reciprocal phases as a raw memmap.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args()
    args.data_config = str(Path(args.data_config).expanduser().resolve())
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive.")
    if args.limit < 0:
        raise ValueError("--limit cannot be negative.")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Batch size must be positive and workers cannot be negative.")
    if args.threshold < 0:
        raise ValueError("--threshold cannot be negative.")
    if args.ssim_window_size < 1 or args.ssim_window_size % 2 == 0:
        raise ValueError("--ssim-window-size must be a positive odd integer.")
    if args.ssim_window_size > args.shape:
        raise ValueError("--ssim-window-size cannot exceed --shape.")


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
        logging.FileHandler(output_dir / "evaluation.log", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    requested_variant: str = "auto",
) -> tuple[HighStrainPhaseUNet, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    checkpoint_variant = infer_model_variant(state_dict)
    if requested_variant != "auto" and requested_variant != checkpoint_variant:
        raise ValueError(
            f"Checkpoint uses model variant {checkpoint_variant!r}, but "
            f"{requested_variant!r} was requested."
        )
    model = HighStrainPhaseUNet(model_variant=checkpoint_variant).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    return model, metadata


def select_reciprocal_phase(
    predicted_phase: torch.Tensor,
    target_phase: torch.Tensor,
    weights: torch.Tensor,
    ambiguity_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the direct or conjugate/twin-equivalent reciprocal phase."""

    direct, inverted = phase_retrieval_wca_components(
        predicted_phase,
        target_phase,
        weights,
    )
    twin_selected = inverted < direct
    if ambiguity_mode == "twin_aligned":
        mask = twin_selected.reshape((-1,) + (1,) * (predicted_phase.ndim - 1))
        selected_phase = torch.where(mask, -predicted_phase, predicted_phase)
    else:
        selected_phase = predicted_phase
    return selected_phase, direct, inverted, twin_selected


def write_sample_csv(path: Path, rows: list[dict[str, object]]) -> None:
    metric_keys = sorted({key for row in rows for key in row if key != "name"})
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["name", *metric_keys])
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(report: dict[str, object]) -> str:
    run = report["run"]
    mean = report["mean"]
    lines = [
        "# high_strain_CNN AutoPhaseNN evaluation",
        "",
        "## Run",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Checkpoint | `{run['checkpoint']}` |",
        f"| Model variant | `{run['model_variant']}` |",
        f"| Model parameters | {run['model_parameters']:,} |",
        f"| Project version | `{run['project_version']}` |",
        f"| Git commit | `{run['git_commit']}` |",
        f"| Samples | {run['num_samples']} |",
        f"| Device | `{run['device']}` |",
        f"| Ambiguity mode | `{run['ambiguity_mode']}` |",
        f"| Support threshold | {run['threshold']} |",
        "",
        "## AutoPhaseNN-scale metrics",
        "",
        "| Group | Metric | Mean |",
        "|---|---|---:|",
    ]
    for group_name, values in fixed_metric_groups(mean).items():
        for metric_name, value in values.items():
            lines.append(f"| {group_name} | {metric_name} | {value:.6g} |")
    lines.extend(
        [
            "",
            "## Phase retrieval diagnostics",
            "",
            f"- WCA loss: `{mean.get('phase_wca', float('nan')):.6g}`",
            f"- Twin/conjugate selection fraction: `{mean.get('twin_flip_selected', 0.0):.6g}`",
            f"- Mean model inference: `{mean.get('inference_ms', float('nan')):.6g} ms/sample`",
            "",
            "## Interpretation",
            "",
            "The real-space amplitude, phase, and support metrics use the same official "
            "AutoPhaseNN post-processing and metric functions. Reciprocal-space modulus "
            "metrics are expected to be nearly zero because reconstruction explicitly "
            "reuses the measured modulus; WCA is the meaningful reciprocal-phase metric.",
            "",
            "`twin_aligned` uses the real-space target only during evaluation to select "
            "between the two signs explicitly treated as equivalent by the published WCA "
            "loss. Use `raw` to measure the uncorrected model output.",
        ]
    )
    return "\n".join(lines) + "\n"


@torch.inference_mode()
def evaluate(
    args: argparse.Namespace,
    model: HighStrainPhaseUNet,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    sample_count: int,
) -> list[dict[str, object]]:
    model.eval()
    rows: list[dict[str, object]] = []
    shape = (args.shape, args.shape, args.shape)
    realspace_writer = (
        np.memmap(
            output_dir / "predicted_realspace.npy",
            dtype="complex64",
            mode="w+",
            shape=(sample_count,) + shape,
        )
        if args.save_realspace
        else None
    )
    phase_writer = (
        np.memmap(
            output_dir / "predicted_reciprocal_phase.npy",
            dtype="float32",
            mode="w+",
            shape=(sample_count,) + shape,
        )
        if args.save_reciprocal_phase
        else None
    )
    postprocess_workers = resolve_postprocess_workers(
        args.postprocess_workers,
        args.batch_size,
    )
    executor = (
        ThreadPoolExecutor(
            max_workers=postprocess_workers,
            thread_name_prefix="phase-unwrap",
        )
        if postprocess_workers > 1
        else None
    )
    offset = 0
    try:
        for batch in tqdm(loader, desc="high_strain AutoPhaseNN evaluation", unit="batch"):
            model_input = batch["input"].to(device, non_blocking=True).float()
            measured_modulus = batch["diffraction"].to(
                device,
                non_blocking=True,
            ).float()
            true_object = batch["realspace"].to(device, non_blocking=True)
            target_reciprocal_phase = reciprocal_phase_from_realspace(true_object)

            synchronize(device)
            inference_started = time.perf_counter()
            predicted_phase = model(model_input)
            synchronize(device)
            inference_seconds = time.perf_counter() - inference_started

            selected_phase, wca_direct, wca_inverted, twin_selected = (
                select_reciprocal_phase(
                    predicted_phase,
                    target_reciprocal_phase,
                    model_input[:, 0],
                    args.ambiguity_mode,
                )
            )
            predicted_object = realspace_from_modulus_phase(
                measured_modulus,
                selected_phase,
            )
            predicted_modulus = farfield_modulus_from_realspace(predicted_object)
            true_object = true_object[:, None]
            true_amp = true_object.abs().float()
            true_phi = torch.angle(true_object).float()
            predicted_amp = predicted_object.abs().float()
            predicted_phi = torch.angle(predicted_object).float()
            predicted_support = (predicted_amp >= args.threshold).float()

            (
                true_amp_post,
                true_phi_post,
                predicted_amp_post,
                predicted_phi_post,
                predicted_support_post,
            ) = post_process_realspace_batch(
                true_amp,
                true_phi,
                predicted_amp,
                predicted_phi,
                predicted_support,
                threshold=args.threshold,
                executor=executor,
            )
            metric_tensors = metric_tensor_dict(measured_modulus, predicted_modulus)
            metric_tensors.update(
                realspace_metric_tensor_dict(
                    true_amp_post,
                    true_phi_post,
                    predicted_amp_post,
                    predicted_phi_post,
                    predicted_support_post,
                    threshold=args.threshold,
                    ssim_window_size=args.ssim_window_size,
                )
            )
            metric_tensors.update(
                {
                    "phase_wca_direct": wca_direct,
                    "phase_wca_inverted": wca_inverted,
                    "phase_wca": torch.minimum(wca_direct, wca_inverted),
                    "twin_flip_selected": twin_selected.float(),
                    "inference_ms": torch.full_like(
                        wca_direct,
                        1000.0 * inference_seconds / model_input.shape[0],
                    ),
                }
            )
            batch_rows = materialize_metric_rows(metric_tensors)
            for name, metrics in zip(batch["name"], batch_rows):
                rows.append({"name": name, **metrics})

            batch_size = model_input.shape[0]
            if realspace_writer is not None:
                realspace_writer[offset : offset + batch_size] = (
                    predicted_object[:, 0].detach().cpu().numpy().astype(np.complex64)
                )
            if phase_writer is not None:
                phase_writer[offset : offset + batch_size] = (
                    selected_phase[:, 0].detach().cpu().numpy().astype(np.float32)
                )
            offset += batch_size
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        if realspace_writer is not None:
            realspace_writer.flush()
        if phase_writer is not None:
            phase_writer.flush()
    return rows


@torch.inference_mode()
def warm_up(
    model: HighStrainPhaseUNet,
    loader: DataLoader,
    device: torch.device,
    batches: int,
) -> None:
    if batches <= 0:
        return
    completed = 0
    for batch in loader:
        model(batch["input"].to(device, non_blocking=True).float())
        completed += 1
        if completed >= batches:
            break
    synchronize(device)
    LOGGER.info("Completed %d warmup batch(es).", completed)


def main() -> int:
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    configure_logging(output_dir, args.log_level)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    sample_count = min(args.num_samples, args.limit) if args.limit > 0 else args.num_samples
    data_config = load_data_config(args.data_config)
    data_dir = Path(args.data_dir).expanduser().resolve()
    shape = (args.shape, args.shape, args.shape)
    data_manifest = build_data_manifest(
        config=data_config,
        root=data_dir,
        shape=shape,
        diffraction_dtype=args.dtype_diff,
        realspace_dtype=args.dtype_real,
        splits={
            "val": {
                "diffraction": args.data_diff,
                "realspace": args.data_real,
                "num_samples": sample_count,
            }
        },
        input_log_data=args.input_log_data,
    )
    data_manifest["file_status"] = require_data_files(data_manifest)
    dataset = AutoPhaseNNPhaseDataset(
        data_dir / args.data_diff,
        data_dir / args.data_real,
        sample_count,
        shape=shape,
        diffraction_dtype=args.dtype_diff,
        realspace_dtype=args.dtype_real,
        input_log_data=args.input_log_data,
        return_diffraction_modulus=True,
    )
    device = choose_device(args.device)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model, checkpoint_metadata = load_model(
        checkpoint_path,
        device,
        args.model_variant,
    )
    runtime = runtime_manifest(device)
    checkpoint_version = checkpoint_metadata.get("project_version")
    if checkpoint_version and checkpoint_version != runtime["project_version"]:
        LOGGER.warning(
            "Checkpoint project version %s differs from evaluator version %s.",
            checkpoint_version,
            runtime["project_version"],
        )
    LOGGER.info("Checkpoint: %s", checkpoint_path)
    LOGGER.info("Data: %s samples from %s", sample_count, data_dir)
    LOGGER.info(
        "Model: %s | %s parameters | device=%s | ambiguity=%s",
        model.model_variant,
        f"{count_parameters(model):,}",
        device,
        args.ambiguity_mode,
    )
    warm_up(model, loader, device, args.warmup_batches)

    started = time.perf_counter()
    rows = evaluate(args, model, loader, device, output_dir, sample_count)
    wall_seconds = time.perf_counter() - started
    if not rows:
        raise RuntimeError("Evaluation produced no samples.")
    statistics = metric_statistics(rows)
    mean = {key: values["mean"] for key, values in statistics.items()}
    report: dict[str, object] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint_metadata.get("epoch"),
            "checkpoint_project_version": checkpoint_version,
            "checkpoint_git_commit": checkpoint_metadata.get("git_commit"),
            "model_variant": model.model_variant,
            "model_parameters": count_parameters(model),
            "project_version": runtime["project_version"],
            "git_commit": runtime["git_commit"],
            "device": str(device),
            "torch_version": torch.__version__,
            "num_samples": len(rows),
            "batch_size": args.batch_size,
            "ambiguity_mode": args.ambiguity_mode,
            "threshold": args.threshold,
            "ssim_window_size": args.ssim_window_size,
            "wall_seconds": wall_seconds,
        },
        "configuration": vars(args),
        "runtime": runtime,
        "data": data_manifest,
        "reconstruction": {
            "spectrum": "measured_modulus * exp(1j * selected_reciprocal_phase)",
            "inverse": "fftshift(ifftn(ifftshift(spectrum)))",
            "selected_phase": args.ambiguity_mode,
        },
        "artifacts": {
            "predicted_realspace": (
                str(output_dir / "predicted_realspace.npy")
                if args.save_realspace
                else None
            ),
            "predicted_reciprocal_phase": (
                str(output_dir / "predicted_reciprocal_phase.npy")
                if args.save_reciprocal_phase
                else None
            ),
            "memmap_shape": [sample_count, *shape],
        },
        "fixed_metric_groups": fixed_metric_groups(mean),
        "fixed_metric_descriptions": FIXED_METRIC_DESCRIPTIONS,
        "mean_metric_groups": group_metrics(mean),
        "metric_descriptions": {
            **METRIC_DESCRIPTIONS,
            "phase_wca": "Published symmetry-aware weighted circular-average loss.",
            "phase_wca_direct": "WCA error against the direct reciprocal phase.",
            "phase_wca_inverted": "WCA error against the conjugate/twin phase.",
            "twin_flip_selected": "One when twin alignment negates the predicted phase.",
        },
        "metric_statistics": statistics,
        "mean": mean,
        "per_sample": rows,
        "notes": {
            "metric_compatibility": (
                "Real-space values use the same AutoPhaseNN official post-processing "
                "and metric functions as autophasenn_training_pipeline/evaluate.py."
            ),
            "reciprocal_metrics": (
                "Modulus metrics are nearly exact by construction because measured "
                "modulus is reused; phase_wca evaluates the learned reciprocal phase."
            ),
            "twin_alignment": (
                "twin_aligned uses target data only for evaluation-time choice between "
                "the two signs explicitly treated as equivalent by the published loss."
            ),
        },
    }
    json_path = output_dir / "evaluation_results.json"
    csv_path = output_dir / "evaluation_samples.csv"
    markdown_path = output_dir / "evaluation_summary.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    write_sample_csv(csv_path, rows)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    LOGGER.info("\n%s", format_fixed_metric_groups(mean, title="Mean metrics"))
    LOGGER.info("WCA: %.6g | twin selection fraction: %.6g", mean["phase_wca"], mean["twin_flip_selected"])
    LOGGER.info("Wrote results: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
