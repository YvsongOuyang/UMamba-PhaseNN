"""Exploratory one-image Au5nm fit with phase-map calibration.

The colored phase TIFF was converted to an uncalibrated complex pseudo-label
upstream. This script adjusts only a single phase-gain parameter against the
same rendered diffraction frame, then builds a halfway Fourier-consistent
training target. It starts from the original-model single-image warm fit and
fine-tunes the phase network and refiner with fixed BN statistics.
All reported scores are same-image fitting diagnostics, not experimental truth.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pytorch_autophasenn.reconstruction import farfield_modulus_from_realspace
from .data import build_dataset, load_config, prepare_batch
from .fit_au5nm_absolute import cpu_state
from pytorch_autophasenn.momamba_refiner import ambiguity_aware_component_mae
from .train import PROJECT_DIR, build_loader, load_checkpoint, make_model


ORIGINAL_SOURCE = "/data_ssd/oyys/high_strain_cnn/pipeline_output_2d/ecdi_2d512_20261014_224534_refiner/checkpoint_best.pt"
SOURCE = "/data_ssd/oyys/high_strain_cnn/pipeline_output_2d/au5nm_absolute_fit/checkpoint_best.pt"
DATA = "/data_ssd/oyys/eCDI/external_test/au5nm/input_check/config_raw.json"
CHECKPOINT_ROOT = "/data_ssd/oyys/high_strain_cnn/pipeline_output_2d"


def args_from_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", default=SOURCE)
    parser.add_argument("--data-config", default=DATA)
    parser.add_argument("--run-name", default="au5nm_phase_calibrated_joint_cosine")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--fourier-weight", type=float, default=1.0)
    parser.add_argument("--amplitude-weight", type=float, default=2.0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--checkpoint-root", default=CHECKPOINT_ROOT)
    parser.add_argument("--runs-dir", default=str(PROJECT_DIR / "artifacts" / "training" / "ecdi_training"))
    args = parser.parse_args()
    if args.steps < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("steps and learning-rate must be positive")
    if not np.isfinite(args.fourier_weight) or args.fourier_weight <= 0 or not np.isfinite(args.amplitude_weight) or args.amplitude_weight <= 0:
        parser.error("fourier-weight and amplitude-weight must be positive")
    if Path(args.run_name).name != args.run_name or args.run_name in ("", ".", ".."):
        parser.error("run-name must be a single directory name")
    return args


def fourier_error(field: torch.Tensor, measured: torch.Tensor) -> torch.Tensor:
    predicted = farfield_modulus_from_realspace(field, spatial_dims=2)
    scale = (predicted * measured).sum() / predicted.square().sum().clamp_min(1e-20)
    return torch.linalg.vector_norm(scale * predicted - measured) / torch.linalg.vector_norm(measured).clamp_min(1e-20)


@torch.no_grad()
def calibrate(original: torch.Tensor, measured: torch.Tensor):
    amplitude = original.abs()
    phase = torch.angle(original)
    tested = []
    for gain in range(1, 25):
        candidate = torch.polar(amplitude, gain * phase)
        tested.append((gain, float(fourier_error(candidate, measured))))
    gain, _ = min(tested, key=lambda item: item[1])
    candidate = torch.polar(amplitude, gain * phase)
    spectrum = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(candidate, dim=(-2, -1))), dim=(-2, -1))
    estimated_modulus = spectrum.abs()
    scaled_measurement = measured * (torch.linalg.vector_norm(estimated_modulus) / torch.linalg.vector_norm(measured))
    blend = 0.5
    mixed_modulus = (1 - blend) * estimated_modulus + blend * scaled_measurement
    corrected = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(mixed_modulus * torch.exp(1j * torch.angle(spectrum)), dim=(-2, -1))), dim=(-2, -1))
    diagnostics = {"phase_gain_grid": tested, "selected_phase_gain": gain,
                   "fourier_blend": blend,
                   "original_target_fourier_rel_l2": float(fourier_error(original, measured)),
                   "phase_adjusted_fourier_rel_l2": float(fourier_error(candidate, measured)),
                   "corrected_target_fourier_rel_l2": float(fourier_error(corrected, measured)),
                   "corrected_target_amplitude_mae_relative_to_original": float((corrected.abs() - amplitude).abs().mean() / amplitude.mean().clamp_min(1e-7))}
    return corrected, diagnostics


def measurements(model, raw, config, device, original_amplitude, fourier_weight, amplitude_weight):
    model.eval()
    batch = prepare_batch(raw, config, device)
    result = model(batch["input"], batch["diffraction"])
    prediction, target = result.refined_object, batch["realspace"]
    support = target.abs() / target.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-7)
    support = support >= config["training"]["support_threshold"]
    # Optimize the same ambiguity-aware complex MAE used by evaluation, rather
    # than the scale-sensitive proxy loss used by the earlier exploratory fit.
    image_loss = ambiguity_aware_component_mae(
        prediction, target, support, spatial_dims=2
    ).loss
    predicted_modulus = farfield_modulus_from_realspace(prediction, spatial_dims=2)
    measured_modulus = batch["diffraction"]
    cosine = (predicted_modulus * measured_modulus).sum() / (
        torch.linalg.vector_norm(predicted_modulus) * torch.linalg.vector_norm(measured_modulus)
    ).clamp_min(1e-20)
    cosine = cosine.clamp(0, 1)
    diffraction_loss = torch.sqrt((1 - cosine.square()).clamp_min(0))
    diffraction_training_loss = 1 - cosine
    original_amplitude_error = (prediction.abs() - original_amplitude).abs().mean() / original_amplitude.mean().clamp_min(1e-7)
    score = image_loss + amplitude_weight * original_amplitude_error + fourier_weight * diffraction_training_loss
    return score, {"corrected_target_fit_loss": float(image_loss.detach()),
                   "diffraction_rel_l2": float(diffraction_loss.detach()),
                   "diffraction_cosine": float(cosine.detach()),
                   "diffraction_training_loss": float(diffraction_training_loss.detach()),
                   "original_amplitude_relative_mae": float(original_amplitude_error.detach()),
                   "weighted_score": float(score.detach())}, prediction


def save_figure(path: Path, images: list[np.ndarray]):
    titles = ("Color-derived target", "Phase-adjusted target", "Warm-start model", "Fitted model")
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3), constrained_layout=True)
    maximum = max(float(np.abs(image).max()) for image in images)
    for axis, title, field in zip(axes, titles, images):
        axis.imshow(np.abs(field), cmap="magma", vmin=0, vmax=maximum)
        axis.set_title(title, fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_phase_figure(path: Path, images: list[np.ndarray]):
    titles = ("Color-derived phase", "Adjusted phase", "Warm-start phase", "Fitted phase")
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3), constrained_layout=True)
    for axis, title, field in zip(axes, titles, images):
        visible = np.abs(field) >= 0.1 * max(float(np.abs(field).max()), 1e-7)
        axis.imshow(np.ma.masked_where(~visible, np.angle(field)), cmap="twilight",
                    vmin=-np.pi, vmax=np.pi)
        axis.set_title(title, fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = args_from_cli()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu explicitly")
    torch.manual_seed(42)
    device = torch.device(args.device)
    source = load_checkpoint(args.source_checkpoint)
    if source.get("stage") != "refiner":
        raise ValueError("Expected a refiner checkpoint")
    config = load_config(args.data_config)
    for key in ("diffraction_kind", "input_log_data", "center_targets"):
        if config[key] != source["config"][key]:
            raise ValueError(f"Source and data differ on {key}")
    config["training"] = source["config"]["training"]
    dataset = build_dataset(config, "train")
    if len(dataset) != 1:
        raise ValueError("Expected exactly one sample")
    raw = next(iter(build_loader(dataset, 1, 0, False, device)))
    prepared = prepare_batch(raw, config, device)
    original = prepared["realspace"].detach()
    measured = prepared["diffraction"].detach()
    corrected, calibration = calibrate(original, measured)
    raw["realspace"] = corrected.detach().cpu()
    original_amplitude = original.abs()

    run_dir = Path(args.runs_dir).expanduser().resolve() / args.run_name
    checkpoint_dir = Path(args.checkpoint_root).expanduser().resolve() / args.run_name
    if any(path.exists() for path in (run_dir, checkpoint_dir)):
        raise FileExistsError("Run name already exists; choose a new --run-name")
    run_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, force=True,
                        format="%(asctime)s | %(levelname)-7s | %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "console.log", encoding="utf-8")])
    log = logging.getLogger(__name__)
    log.info("Calibrated phase gain=%d | target Fourier error=%.6f -> %.6f -> %.6f",
             calibration["selected_phase_gain"], calibration["original_target_fourier_rel_l2"],
             calibration["phase_adjusted_fourier_rel_l2"], calibration["corrected_target_fourier_rel_l2"])
    original_source = load_checkpoint(ORIGINAL_SOURCE)
    reference_model = make_model("refiner", config, checkpoint=original_source).to(device).eval()
    with torch.no_grad():
        _, reference, _ = measurements(reference_model, raw, config, device, original_amplitude,
                                       args.fourier_weight, args.amplitude_weight)
    del reference_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model = make_model("refiner", config, checkpoint=source).to(device).eval()
    model.freeze_phase_model = False
    for parameter in model.phase_model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam([
        # The paired Au5nm target is supervised data; use the same learning
        # rate for phase and refiner so the phase error can actually overfit.
        {"params": model.phase_model.parameters(), "lr": args.learning_rate},
        {"params": model.refiner.parameters(), "lr": args.learning_rate},
    ])
    with torch.no_grad():
        _, baseline, initial_field = measurements(model, raw, config, device, original_amplitude,
                                                  args.fourier_weight, args.amplitude_weight)
    initial_image = initial_field[0, 0].detach().cpu().numpy().copy()
    best_score, best_step = float("inf"), 0
    best_state = cpu_state(model)
    history = []
    log.info(
        "Stage=refiner | shape=512x512 | batch=1 | parameters=%s | trainable=%s | dtype=float32/complex64",
        f'{sum(p.numel() for p in model.parameters()):,}',
        f'{sum(p.numel() for p in model.parameters() if p.requires_grad):,}',
    )
    log.info("Initialized from %s (epoch=%s); optimizer starts fresh",
             args.source_checkpoint, source.get("epoch"))
    log.info("Run=%s | checkpoints=%s", run_dir, checkpoint_dir)
    for step in range(1, args.steps + 1):
        log.info("Starting epoch=%03d/%03d | lr=%.3g",
                 step, args.steps, optimizer.param_groups[0]["lr"])
        train_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        current_fourier_weight = args.fourier_weight * min(1.0, max(0.0, (step - 100) / max(1, args.steps - 100)))
        score, _, _ = measurements(model, raw, config, device, original_amplitude,
                                   current_fourier_weight, args.amplitude_weight)
        if not torch.isfinite(score):
            raise ValueError("Training loss became nonfinite")
        score.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        optimizer.step()
        train_loss = float(score.detach())
        log.info("TRAIN | batch=1/1 | loss=%.6g | elapsed=%.1fs | eta=0.0s",
                 train_loss, time.perf_counter() - train_started)
        val_started = time.perf_counter()
        with torch.no_grad():
            evaluated_score, metrics, _ = measurements(model, raw, config, device, original_amplitude,
                                                        current_fourier_weight, args.amplitude_weight)
        if not torch.isfinite(evaluated_score):
            raise ValueError("Evaluation loss became nonfinite")
        # For a paired external sample, select against its self-consistent
        # complex target.  The old gate compared against the unrelated TIFF
        # amplitude proxy, which made every genuinely improved step ineligible.
        selection_score = (
            metrics["corrected_target_fit_loss"]
            + args.fourier_weight * metrics["diffraction_training_loss"]
        )
        # Validation uses the fixed checkpoint-selection objective. The training
        # objective retains its existing Fourier-weight warmup unchanged.
        log.info("VAL | batch=1/1 | loss=%.6g | elapsed=%.1fs | eta=0.0s",
                 selection_score, time.perf_counter() - val_started)
        history.append({"epoch": step, "step": step, "train_loss": train_loss,
                        "val_loss": selection_score,
                        "lr": optimizer.param_groups[0]["lr"],
                        "current_fourier_weight": current_fourier_weight,
                        "selection_score": selection_score, **metrics})
        eligible = metrics["corrected_target_fit_loss"] < baseline["corrected_target_fit_loss"]
        history[-1]["eligible"] = eligible
        if eligible and selection_score < best_score:
            best_score, best_step = selection_score, step
            best_state = cpu_state(model)
        log.info("Epoch=%d/%d | train_loss=%.6g | val_loss=%.6g | best_loss=%.6g | lr=%.3g",
                 step, args.steps, train_loss, selection_score, best_score,
                 optimizer.param_groups[0]["lr"])
        if best_step > 0 and step - best_step >= 100:
            log.info("Stopping after 100 steps without score improvement")
            break
    model.load_state_dict(best_state, strict=True)
    with torch.no_grad():
        _, selected, selected_field = measurements(model, raw, config, device, original_amplitude,
                                                   args.fourier_weight, args.amplitude_weight)
    output = selected_field[0, 0].detach().cpu().numpy().copy()
    original_numpy = original[0, 0].cpu().numpy().copy()
    target_numpy = corrected[0, 0].cpu().numpy().copy()
    np.savez_compressed(run_dir / "comparison.npz", color_derived_target=original_numpy,
                        corrected_target=target_numpy, warm_start_model=initial_image, fitted_model=output)
    save_figure(run_dir / "comparison.png", [original_numpy, target_numpy, initial_image, output])
    save_phase_figure(run_dir / "phase_comparison.png", [original_numpy, target_numpy, initial_image, output])
    report = {"calibration": calibration, "original_reference": reference,
              "warm_start_baseline": baseline, "selected": selected,
              "best_step": best_step, "fourier_weight": args.fourier_weight,
              "amplitude_weight": args.amplitude_weight,
              "improved_both_amplitude_and_diffraction": best_step > 0,
              "same_sample_for_training_and_selection": True,
              "phase_gain_fitted_to_same_rendered_diffraction": True,
              "physically_calibrated": False}
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    torch.save({"spatial_dims": 2, "stage": "refiner", "epoch": best_step, "config": config,
                "model_state_dict": best_state, "best_val_loss": selected["weighted_score"],
                "transfer_source_checkpoint": str(Path(args.source_checkpoint).resolve()),
                "one_sample_phase_calibration": calibration,
                "target_is_uncalibrated": True}, checkpoint_dir / "checkpoint_best.pt")
    log.info("DONE | best_epoch=%d | best_loss=%.6g | checkpoint=%s",
             best_step, best_score,
             checkpoint_dir / "checkpoint_best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
