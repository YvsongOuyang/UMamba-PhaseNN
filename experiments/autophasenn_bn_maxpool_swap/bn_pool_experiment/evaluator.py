"""Paired evaluator for AutoPhaseNN BN/MaxPool order variants."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from autophasenn_training_pipeline.dataset import AutoPhaseDataset

from .config import ExperimentConfig
from .metrics import (
    METRIC_DIRECTIONS,
    output_consistency_metrics,
    reconstruction_metrics,
    stable_metric_seed,
    summarize_values,
    tensor_pair_metrics,
)
from .model import (
    PoolBNOrder,
    PoolBNSwapAutoPhaseNN,
    audit_all_bn_scales,
    audit_bn_scales,
    extract_state_dict,
    infer_threshold,
    load_checkpoint_file,
    load_state_dict_robust,
)
from .utils import (
    environment_metadata,
    file_metadata,
    save_csv,
    save_json,
    set_seed,
    sha256_file,
    validate_memmap_file,
)


LOGGER = logging.getLogger(__name__)


class RandomDiffractionDataset(Dataset):
    """Generate deterministic nonnegative random 3D diffraction-modulus inputs."""

    def __init__(
        self,
        num_samples: int,
        shape: tuple[int, int, int],
        seed: int,
        minimum: float,
        maximum: float,
    ) -> None:
        self.num_samples = int(num_samples)
        self.shape = tuple(shape)
        self.seed = int(seed)
        self.minimum = float(minimum)
        self.maximum = float(maximum)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one deterministic random input shaped ``(1, D, H, W)``."""

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + int(index))
        sample = torch.rand((1, *self.shape), generator=generator, dtype=torch.float32)
        sample = self.minimum + sample * (self.maximum - self.minimum)
        zeros = torch.zeros_like(sample)
        return {
            "diff": sample,
            "amp": zeros,
            "phi": zeros,
            "name": f"random_uniform_{index:06d}",
        }


def _optional_path(data_dir: Path, filename: str | None) -> Path | None:
    if filename is None or filename.lower() in {"", "none", "null"}:
        return None
    return data_dir / filename


def _slice_outputs(outputs: tuple[torch.Tensor, ...], index: int):
    return tuple(value[index : index + 1] for value in outputs)


def _add_prefixed(target: dict[str, Any], prefix: str, values: dict[str, float]) -> None:
    target.update({f"{prefix}.{key}": value for key, value in values.items()})


def _summarize_columns(
    rows: list[dict[str, Any]],
    prefix: str,
    config: ExperimentConfig,
    should_bootstrap: Callable[[str], bool] | None = None,
) -> dict[str, dict[str, Any]]:
    keys = sorted(
        {
            key[len(prefix) + 1 :]
            for row in rows
            for key in row
            if key.startswith(prefix + ".")
        }
    )
    result = {}
    for key in keys:
        values = [float(row[f"{prefix}.{key}"]) for row in rows]
        bootstrap_enabled = should_bootstrap(key) if should_bootstrap else False
        result[key] = summarize_values(
            values,
            bootstrap_samples=(
                config.statistics.bootstrap_samples if bootstrap_enabled else 0
            ),
            confidence_level=config.statistics.confidence_level,
            seed=stable_metric_seed(config.statistics.seed, f"{prefix}.{key}"),
        )
    return result


def _reconstruction_summary(
    rows: list[dict[str, Any]],
    config: ExperimentConfig,
) -> dict[str, dict[str, Any]]:
    baseline = _summarize_columns(rows, "baseline", config)
    swapped = _summarize_columns(rows, "swapped", config)
    primary_metrics = set(config.acceptance.primary_metrics)
    delta = _summarize_columns(
        rows,
        "delta",
        config,
        should_bootstrap=lambda key: key in primary_metrics,
    )
    result: dict[str, dict[str, Any]] = {}
    for key in sorted(set(baseline).intersection(swapped)):
        baseline_mean = baseline[key]["mean"]
        swapped_mean = swapped[key]["mean"]
        direction = METRIC_DIRECTIONS.get(key, "neutral")
        relative_change = None
        degradation = None
        if baseline_mean is not None and swapped_mean is not None:
            denominator = max(abs(float(baseline_mean)), 1e-12)
            relative_change = 100.0 * (float(swapped_mean) - float(baseline_mean)) / denominator
            if direction == "lower":
                degradation = relative_change
            elif direction == "higher":
                degradation = -relative_change
        result[key] = {
            "direction": direction,
            "baseline": baseline[key],
            "swapped": swapped[key],
            "delta": delta[key],
            "relative_change_percent": relative_change,
            "degradation_percent": degradation,
        }
    return result


def _acceptance_summary(
    reconstruction: dict[str, dict[str, Any]],
    consistency: dict[str, dict[str, Any]],
    bn_audit: dict[str, dict[str, float | int]],
    config: ExperimentConfig,
    reconstruction_claim_supported: bool,
) -> dict[str, Any]:
    available_primary = [
        key for key in config.acceptance.primary_metrics if key in reconstruction
    ]
    missing_primary = [
        key for key in config.acceptance.primary_metrics if key not in reconstruction
    ]
    degradations = {
        key: reconstruction[key]["degradation_percent"] for key in available_primary
    }
    primary_pass = None
    if reconstruction_claim_supported:
        primary_pass = bool(available_primary) and not missing_primary and all(
            value is not None
            and float(value) <= config.acceptance.max_primary_metric_degradation_percent
            for value in degradations.values()
        )

    farfield_relative_l1 = consistency.get("farfield.relative_l1", {}).get("mean")
    output_pass = (
        farfield_relative_l1 is not None
        and float(farfield_relative_l1) <= config.acceptance.max_farfield_relative_l1
    )
    bn_pass = all(
        float(item["positive_fraction"])
        >= config.acceptance.min_positive_bn_scale_fraction
        for item in bn_audit.values()
    )
    return {
        "overall_pass": bool(
            output_pass and bn_pass and (primary_pass if reconstruction_claim_supported else True)
        ),
        "scope": (
            "reconstruction_and_consistency"
            if reconstruction_claim_supported
            else "synthetic_consistency_only"
        ),
        "reconstruction_claim_supported": reconstruction_claim_supported,
        "primary_metrics_pass": primary_pass,
        "output_consistency_pass": bool(output_pass),
        "bn_scale_precondition_pass": bool(bn_pass),
        "primary_metrics": list(config.acceptance.primary_metrics),
        "available_primary_metrics": available_primary,
        "missing_primary_metrics": missing_primary,
        "degradation_percent_by_primary_metric": degradations,
        "observed_farfield_relative_l1": farfield_relative_l1,
        "criteria": {
            "max_primary_metric_degradation_percent": config.acceptance.max_primary_metric_degradation_percent,
            "max_farfield_relative_l1": config.acceptance.max_farfield_relative_l1,
            "min_positive_bn_scale_fraction": config.acceptance.min_positive_bn_scale_fraction,
        },
    }


@torch.no_grad()
def run_evaluation(
    config: ExperimentConfig,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Run the paired baseline/swap evaluation and save machine-readable artifacts."""

    set_seed(config.runtime.seed, config.runtime.deterministic)
    data_dir = Path(config.data.data_dir)
    checkpoint_path = Path(config.model.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")

    shape = tuple(int(value) for value in config.data.shape)
    diff_path = data_dir / config.data.diff_file
    real_path = _optional_path(data_dir, config.data.real_file)
    if config.data.mode == "memmap":
        data_metadata = {
            "mode": "memmap",
            "diff": validate_memmap_file(
                diff_path,
                config.data.num_samples,
                shape,
                config.data.dtype_diff,
            ),
        }
        if real_path is not None:
            data_metadata["real"] = validate_memmap_file(
                real_path,
                config.data.num_samples,
                shape,
                config.data.dtype_real,
            )
    else:
        real_path = None
        data_metadata = {
            "mode": "random",
            "distribution": config.data.random_distribution,
            "minimum": config.data.random_min,
            "maximum": config.data.random_max,
            "seed": config.data.random_seed,
            "shape_per_sample": list(shape),
            "num_samples": config.data.num_samples,
            "dtype": "float32",
        }

    LOGGER.info("Loading checkpoint: %s", checkpoint_path)
    checkpoint = load_checkpoint_file(
        checkpoint_path,
        allow_unsafe=config.model.allow_unsafe_checkpoint,
    )
    threshold = infer_threshold(checkpoint, config.model.threshold)
    state_dict = extract_state_dict(checkpoint)
    baseline_model = PoolBNSwapAutoPhaseNN(
        threshold=threshold,
        order=PoolBNOrder.BN_THEN_POOL,
    )
    swapped_model = PoolBNSwapAutoPhaseNN(
        threshold=threshold,
        order=PoolBNOrder.POOL_THEN_BN,
    )
    baseline_load_result = load_state_dict_robust(
        baseline_model,
        state_dict,
        strict=config.model.strict_load,
    )
    swapped_load_result = load_state_dict_robust(
        swapped_model,
        state_dict,
        strict=config.model.strict_load,
    )
    baseline_state = baseline_model.state_dict()
    swapped_state = swapped_model.state_dict()
    state_dicts_identical = tuple(baseline_state) == tuple(swapped_state) and all(
        torch.equal(baseline_state[key], swapped_state[key]) for key in baseline_state
    )
    if not state_dicts_identical:
        raise RuntimeError("The two model instances do not contain identical parameters.")

    baseline_model = baseline_model.to(device)
    swapped_model = swapped_model.to(device)
    baseline_model.eval()
    swapped_model.eval()
    bn_audit = audit_bn_scales(baseline_model)
    all_bn_audit = audit_all_bn_scales(baseline_model)
    all_bn_summary = {
        "layer_count": len(all_bn_audit),
        "channel_count": sum(int(item["channels"]) for item in all_bn_audit.values()),
        "positive": sum(int(item["positive"]) for item in all_bn_audit.values()),
        "zero": sum(int(item["zero"]) for item in all_bn_audit.values()),
        "negative": sum(int(item["negative"]) for item in all_bn_audit.values()),
        "all_positive": all(
            int(item["positive"]) == int(item["channels"])
            for item in all_bn_audit.values()
        ),
        "global_min_effective_scale": min(
            float(item["min_effective_scale"]) for item in all_bn_audit.values()
        ),
        "global_max_effective_scale": max(
            float(item["max_effective_scale"]) for item in all_bn_audit.values()
        ),
        "layers_with_nonpositive_scales": [
            name
            for name, item in all_bn_audit.items()
            if int(item["zero"]) > 0 or int(item["negative"]) > 0
        ],
    }
    LOGGER.info(
        "Created two independent full-model instances: baseline=%s, swapped=%s, identical_state=%s",
        PoolBNOrder.BN_THEN_POOL.value,
        PoolBNOrder.POOL_THEN_BN.value,
        state_dicts_identical,
    )
    for layer_name, item in bn_audit.items():
        LOGGER.info(
            "%s effective BN scale: positive=%d/%d, zero=%d, negative=%d, range=[%.6g, %.6g]",
            layer_name,
            item["positive"],
            item["channels"],
            item["zero"],
            item["negative"],
            item["min_effective_scale"],
            item["max_effective_scale"],
        )

    if config.data.mode == "memmap":
        dataset: Dataset = AutoPhaseDataset(
            diff_path=diff_path,
            real_path=real_path,
            num_samples=config.data.num_samples,
            shape_diff=shape,
            shape_real=shape,
            dtype_diff=config.data.dtype_diff,
            dtype_real=config.data.dtype_real,
            scale_i=config.data.scale_i,
            shuffle=False,
        )
    else:
        dataset = RandomDiffractionDataset(
            num_samples=config.data.num_samples,
            shape=shape,
            seed=config.data.random_seed,
            minimum=config.data.random_min,
            maximum=config.data.random_max,
        )
    loader = DataLoader(
        dataset,
        batch_size=config.runtime.batch_size,
        shuffle=False,
        num_workers=config.runtime.num_workers,
        pin_memory=device.type == "cuda",
    )

    rows: list[dict[str, Any]] = []
    has_realspace = config.data.mode == "memmap" and real_path is not None
    reconstruction_claim_supported = config.data.mode == "memmap" and has_realspace
    output_names = (
        "farfield_modulus",
        "masked_complex_object",
        "masked_amplitude",
        "phase",
        "support",
        "raw_amplitude",
    )
    output_metadata: dict[str, dict[str, dict[str, Any]]] = {}
    input_count = 0
    input_sum = 0.0
    input_sum_squares = 0.0
    input_min = float("inf")
    input_max = float("-inf")
    for batch in tqdm(loader, desc="BN/MaxPool paired evaluation"):
        diff = batch["diff"].to(device, non_blocking=True).float()
        true_amp = batch["amp"].to(device, non_blocking=True).float()
        true_phase = batch["phi"].to(device, non_blocking=True).float()
        input_count += diff.numel()
        input_sum += float(diff.double().sum().cpu())
        input_sum_squares += float(diff.double().square().sum().cpu())
        input_min = min(input_min, float(diff.min().cpu()))
        input_max = max(input_max, float(diff.max().cpu()))

        baseline_model.set_capture(block_outputs=True, intrinsic_pairs=True)
        baseline_outputs = baseline_model(diff)
        baseline_blocks = dict(baseline_model.last_block_outputs)
        intrinsic_pairs = dict(baseline_model.last_intrinsic_pairs)

        swapped_model.set_capture(block_outputs=True, intrinsic_pairs=False)
        swapped_outputs = swapped_model(diff)
        swapped_blocks = dict(swapped_model.last_block_outputs)
        baseline_model.set_capture(block_outputs=False, intrinsic_pairs=False)
        swapped_model.set_capture(block_outputs=False, intrinsic_pairs=False)

        if not output_metadata:
            output_metadata = {
                "baseline": {
                    name: {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "is_complex": bool(value.is_complex()),
                    }
                    for name, value in zip(output_names, baseline_outputs)
                },
                "swapped": {
                    name: {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "is_complex": bool(value.is_complex()),
                    }
                    for name, value in zip(output_names, swapped_outputs)
                },
            }

        for sample_index, name in enumerate(batch["name"]):
            base_sample = _slice_outputs(baseline_outputs, sample_index)
            swap_sample = _slice_outputs(swapped_outputs, sample_index)
            base_metrics = reconstruction_metrics(
                diff[sample_index : sample_index + 1],
                true_amp[sample_index : sample_index + 1],
                true_phase[sample_index : sample_index + 1],
                base_sample,
                has_realspace=has_realspace,
                threshold=threshold,
                amplitude_data_range=config.metrics.amplitude_data_range,
                ssim_window_size=config.metrics.ssim_window_size,
            )
            swap_metrics = reconstruction_metrics(
                diff[sample_index : sample_index + 1],
                true_amp[sample_index : sample_index + 1],
                true_phase[sample_index : sample_index + 1],
                swap_sample,
                has_realspace=has_realspace,
                threshold=threshold,
                amplitude_data_range=config.metrics.amplitude_data_range,
                ssim_window_size=config.metrics.ssim_window_size,
            )
            row: dict[str, Any] = {"name": name}
            _add_prefixed(row, "baseline", base_metrics)
            _add_prefixed(row, "swapped", swap_metrics)
            _add_prefixed(
                row,
                "delta",
                {key: swap_metrics[key] - base_metrics[key] for key in base_metrics},
            )
            _add_prefixed(
                row,
                "consistency",
                output_consistency_metrics(
                    base_sample,
                    swap_sample,
                    histogram_bins=config.metrics.histogram_bins,
                ),
            )
            for layer_name, (bn_pool, pool_bn) in intrinsic_pairs.items():
                _add_prefixed(
                    row,
                    f"intrinsic.{layer_name}",
                    tensor_pair_metrics(
                        bn_pool[sample_index : sample_index + 1],
                        pool_bn[sample_index : sample_index + 1],
                    ),
                )
            for layer_name in sorted(set(baseline_blocks).intersection(swapped_blocks)):
                _add_prefixed(
                    row,
                    f"propagated.{layer_name}",
                    tensor_pair_metrics(
                        baseline_blocks[layer_name][sample_index : sample_index + 1],
                        swapped_blocks[layer_name][sample_index : sample_index + 1],
                    ),
                )
            rows.append(row)

    reconstruction = _reconstruction_summary(rows, config)
    consistency_ci_metrics = {
        "farfield.mae",
        "farfield.relative_l1",
        "farfield.pearson_corr",
        "farfield.histogram_js_divergence",
        "amplitude.mae",
        "amplitude.relative_l1",
        "amplitude.histogram_js_divergence",
        "complex_object.mae",
        "complex_object.relative_l2",
        "complex_object.max_abs",
        "phase.wrapped_mae",
        "support.disagreement_fraction",
    }
    consistency = _summarize_columns(
        rows,
        "consistency",
        config,
        should_bootstrap=lambda key: key in consistency_ci_metrics,
    )
    intrinsic = _summarize_columns(
        rows,
        "intrinsic",
        config,
        should_bootstrap=lambda key: key.endswith(".mae")
        or key.endswith(".relative_l2"),
    )
    propagated = _summarize_columns(
        rows,
        "propagated",
        config,
        should_bootstrap=lambda key: key.endswith(".mae"),
    )
    acceptance = _acceptance_summary(
        reconstruction,
        consistency,
        bn_audit,
        config,
        reconstruction_claim_supported=reconstruction_claim_supported,
    )
    final_output_metric_names = (
        "mae",
        "rmse",
        "max_abs",
        "relative_l1",
        "relative_l2",
        "pearson_corr",
        "histogram_js_divergence",
    )
    final_output_comparison = {
        "output_name": "farfield_modulus",
        "baseline_model_order": PoolBNOrder.BN_THEN_POOL.value,
        "swapped_model_order": PoolBNOrder.POOL_THEN_BN.value,
        "full_forward_called_for_each_model": True,
        "shape": output_metadata["baseline"]["farfield_modulus"]["shape"],
        "dtype": output_metadata["baseline"]["farfield_modulus"]["dtype"],
        "num_samples": len(rows),
        "compared_element_count": int(len(rows) * np.prod(shape)),
        "metrics": {
            name: consistency[f"farfield.{name}"]
            for name in final_output_metric_names
        },
        "maximum_absolute_difference_over_all_samples": max(
            float(row["consistency.farfield.max_abs"]) for row in rows
        ),
    }
    final_output_comparison["exactly_identical"] = bool(
        final_output_comparison["maximum_absolute_difference_over_all_samples"] == 0.0
    )
    input_mean = input_sum / max(input_count, 1)
    input_variance = max(input_sum_squares / max(input_count, 1) - input_mean**2, 0.0)
    input_statistics = {
        "count": input_count,
        "minimum": input_min,
        "maximum": input_max,
        "mean": input_mean,
        "std": input_variance**0.5,
    }

    checkpoint_metadata = file_metadata(checkpoint_path)
    checkpoint_metadata["sha256"] = (
        sha256_file(checkpoint_path) if config.runtime.hash_checkpoint else None
    )
    summary = {
        "experiment": "autophasenn_bn_maxpool_swap",
        "num_samples": len(rows),
        "data_mode": config.data.mode,
        "threshold": threshold,
        "has_realspace_truth": has_realspace,
        "reconstruction_claim_supported": reconstruction_claim_supported,
        "checkpoint": checkpoint_metadata,
        "checkpoint_load": {
            "baseline": baseline_load_result,
            "swapped": swapped_load_result,
        },
        "data": data_metadata,
        "input_statistics": input_statistics,
        "model_comparison": {
            "separate_model_instances": baseline_model is not swapped_model,
            "state_dicts_identical": state_dicts_identical,
            "baseline_order": PoolBNOrder.BN_THEN_POOL.value,
            "swapped_order": PoolBNOrder.POOL_THEN_BN.value,
            "full_forward_called_for_each_model": True,
            "outputs_compared": list(output_names),
            "output_metadata": output_metadata,
        },
        "final_output_comparison": final_output_comparison,
        "bn_scale_audit": bn_audit,
        "all_bn_scale_audit": all_bn_audit,
        "all_bn_scale_summary": all_bn_summary,
        "reconstruction": reconstruction,
        "output_consistency": consistency,
        "layer_intrinsic": intrinsic,
        "layer_propagated": propagated,
        "acceptance": acceptance,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(output_dir / "per_sample.csv", rows)
    save_json(output_dir / "summary.json", summary)

    environment = environment_metadata(device)
    environment["checkpoint"] = checkpoint_metadata
    environment["data"] = data_metadata
    save_json(output_dir / "environment.json", environment)
    return summary
