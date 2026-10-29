"""Evaluate replacing only the phase model inside one frozen refiner cascade.

The old and new cascades share the same refiner checkpoint.  No weights are
trained, saved or changed on disk; targets are used only to calculate metrics.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch

from .data_v2 import build_dataset, load_config, prepare_batch
from .evaluate import metrics_2d
from .evaluate_phase_v4 import offaxis_metrics, read_thickness, save_records, summarize_records
from .train import build_loader, load_checkpoint, make_model, verify_data_identity


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "ecdi_2d_phase_v5.json"
STAGES = ("initial_old", "refined_old", "initial_new", "refined_new")


def refiner_digest(model):
    """Hash parameters and buffers, including BatchNorm running statistics."""
    digest = hashlib.sha256()
    for name, value in sorted(model.refiner.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_compatibility(refiner_checkpoint, phase_checkpoint, config):
    if refiner_checkpoint.get("stage") != "refiner":
        raise ValueError("--refiner-checkpoint must be a refiner-stage checkpoint.")
    if phase_checkpoint.get("stage") != "phase":
        raise ValueError("--phase-checkpoint must be a phase-stage checkpoint.")
    sources = (refiner_checkpoint["config"], phase_checkpoint["config"], config)
    for source in sources:
        if source["training"].get("phase_architecture", "baseline") != "baseline":
            raise ValueError("Only baseline phase architectures can be substituted.")
    for key in ("diffraction_kind", "input_log_data", "center_targets", "spatial_dims", "shape"):
        if any(source[key] != sources[0][key] for source in sources[1:]):
            raise ValueError(f"Checkpoint/evaluation preprocessing differs on {key}.")
    variants = [source["training"]["model_variant"] for source in sources]
    if len(set(variants)) != 1:
        raise ValueError("All model_variant values must match.")
    for checkpoint in (refiner_checkpoint, phase_checkpoint):
        previous = checkpoint.get("data_manifests", {}).get("train", {})
        if not all(previous.get(key) for key in ("config_sha256", "manifest_sha256")):
            raise ValueError("Both checkpoints must record generation and split hashes.")
    for key in ("config_sha256", "manifest_sha256"):
        if (refiner_checkpoint["data_manifests"]["train"][key]
                != phase_checkpoint["data_manifests"]["train"][key]):
            raise ValueError(f"Phase/refiner data identity differs on {key}.")


def build_cascades(refiner_checkpoint, phase_checkpoint, device):
    """Load both cascades from one source, then replace only new.phase_model."""
    config = refiner_checkpoint["config"]
    old = make_model("refiner", config, checkpoint=refiner_checkpoint)
    new = make_model("refiner", config, checkpoint=refiner_checkpoint)
    new.phase_model.load_state_dict(phase_checkpoint["model_state_dict"], strict=True)
    expected = refiner_digest(old)
    if refiner_digest(new) != expected:
        raise RuntimeError("Refiner parameters or buffers changed during phase substitution.")
    return old.to(device).eval(), new.to(device).eval(), expected


def paired_refined_comparison(records):
    pairs = {(row["sample"], row["stage"]): row for row in records}
    samples = sorted({row["sample"] for row in records})
    result = {}
    for metric, higher in (("complex_nrmse", False), ("real_amp_ssim", True),
                           ("real_phase_mae_true_support", False),
                           ("offaxis_nrmse_9x9", False)):
        delta = np.array([pairs[(sample, "refined_new")][metric]
                          - pairs[(sample, "refined_old")][metric] for sample in samples])
        result[metric] = {
            "new_minus_old_mean": float(delta.mean()),
            "new_minus_old_median": float(np.median(delta)),
            "new_fraction_better": float(np.mean(delta > 0 if higher else delta < 0)),
        }
    return result


@torch.no_grad()
def evaluate_cascades(old, new, loader, config, device, thickness):
    old.eval()
    new.eval()
    before = (refiner_digest(old), refiner_digest(new))
    if before[0] != before[1]:
        raise RuntimeError("Old and new refiner states differ before evaluation.")
    threshold = config["training"].get("support_threshold", 0.3)
    records = []
    for batch_index, raw in enumerate(loader, 1):
        batch = prepare_batch(raw, config, device)
        result_old = old(batch["input"], batch["diffraction"])
        result_new = new(batch["input"], batch["diffraction"])
        fields = dict(zip(STAGES, (result_old.initial_object, result_old.refined_object,
                                   result_new.initial_object, result_new.refined_object)))
        for stage, field in fields.items():
            values = metrics_2d(field, batch["realspace"], batch["diffraction"],
                                batch["input"], threshold, batch.get("ideal_intensity"))
            values.update(offaxis_metrics(field, batch["realspace"], threshold))
            arrays = {key: value.cpu().tolist() for key, value in values.items()}
            for position in range(field.shape[0]):
                sample = int(batch["index"][position])
                records.append({
                    "sample": sample, "stage": stage,
                    "material": batch["material"][position],
                    "thickness_a": thickness.get(sample),
                    **{key: value[position] for key, value in arrays.items()},
                })
        if batch_index % 20 == 0 or batch_index == len(loader):
            LOGGER.info("batch=%d/%d", batch_index, len(loader))
    if not records:
        raise ValueError("The evaluation split is empty.")
    if (refiner_digest(old), refiner_digest(new)) != before:
        raise RuntimeError("Refiner parameters or BatchNorm buffers changed during evaluation.")
    return summarize_records(records), records, paired_refined_comparison(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refiner-checkpoint", required=True)
    parser.add_argument("--phase-checkpoint", required=True)
    parser.add_argument("--data-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("Batch size must be positive and workers nonnegative.")
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("Output directory already exists; choose a new --output-dir.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    logging.basicConfig(level=logging.INFO, force=True,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")
    refiner_checkpoint = load_checkpoint(args.refiner_checkpoint)
    phase_checkpoint = load_checkpoint(args.phase_checkpoint)
    config = load_config(args.data_config)
    verify_compatibility(refiner_checkpoint, phase_checkpoint, config)
    # Inference must use the refiner's architecture and support threshold.
    config["training"] = copy.deepcopy(refiner_checkpoint["config"]["training"])
    if args.data_dir:
        config["root"] = str(Path(args.data_dir).expanduser().resolve())
    dataset = build_dataset(config, args.split, include_ideal=True)
    try:
        verify_data_identity(refiner_checkpoint, dataset.manifest)
        verify_data_identity(phase_checkpoint, dataset.manifest)
        loader = build_loader(dataset, args.batch_size, args.num_workers, False, device)
        old, new, state_digest = build_cascades(refiner_checkpoint, phase_checkpoint, device)
        LOGGER.info("Same refiner verified | sha256=%s | old_epoch=%s | new_phase_epoch=%s",
                    state_digest, refiner_checkpoint.get("epoch"), phase_checkpoint.get("epoch"))
        summary, records, paired = evaluate_cascades(
            old, new, loader, config, device, read_thickness(dataset)
        )
        report = {
            "experiment_version": "phase_substitution_v5_evaluation", "split": args.split,
            "refiner_checkpoint": str(Path(args.refiner_checkpoint).resolve()),
            "phase_checkpoint": str(Path(args.phase_checkpoint).resolve()),
            "refiner_epoch": refiner_checkpoint.get("epoch"),
            "new_phase_epoch": phase_checkpoint.get("epoch"),
            "refiner_state_sha256": state_digest, "data": dataset.manifest,
            "metrics": summary, "paired_refined_new_vs_old": paired,
            "offaxis_definition": "After full-object alignment and RMS normalization, exclude the centered 9x9 Fourier pixels and divide residual energy by target energy in the remaining band.",
        }
        output.mkdir(parents=True, exist_ok=False)
        (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        save_records(output / "per_sample.csv", records)
        for stage in STAGES:
            metrics = summary[stage]
            LOGGER.info("%s | complex_nrmse=%.6f | amp_ssim=%.6f | thick_nrmse=%.6f | offaxis_nrmse=%.6f",
                        stage, metrics["all"]["complex_nrmse"], metrics["all"]["real_amp_ssim"],
                        metrics["thick_ge_34a"].get("complex_nrmse", float("nan")),
                        metrics["all"]["offaxis_nrmse_9x9"])
        LOGGER.info("Refined new minus old | paired=%s", paired)
        LOGGER.info("Output=%s", output)
    finally:
        close = getattr(dataset, "close", None)
        if close is not None:
            close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
