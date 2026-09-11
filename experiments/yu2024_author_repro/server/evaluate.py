"""Evaluate a corrected checkpoint on the held-out test set; save metrics and predictions."""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import diffraction, load_arrays, sha256
from train import Samples, Unet, finite_tensors


def evaluate(args):
    torch.set_num_threads(4)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if state["config"]["overfit_samples"]:
        raise ValueError("An overfit diagnostic checkpoint is not eligible for held-out evaluation")
    if sha256(args.data_dir / "manifest.json") != state["dataset_manifest_sha256"]:
        raise ValueError("Dataset differs from checkpoint")
    arrays, splits, _ = load_arrays(args.data_dir)
    device = torch.device(args.device)
    model = Unet().to(device).eval()
    model.load_state_dict(state["model_state_dict"], strict=True)
    loader = DataLoader(Samples(arrays, splits["test"], state["scale"]), batch_size=args.batch_size)
    predictions = []
    with torch.no_grad():
        for inputs, _ in loader:
            output = model(inputs.to(device))
            finite_tensors([output], "evaluation output")
            predictions.append(torch.complex(output[..., 0], output[..., 1]).cpu().numpy()[:, 0])
    prediction = np.concatenate(predictions).astype(np.complex128)
    ids = splits["test"]
    target = arrays["object_modulus"][ids].astype(np.float64) * np.exp(1j * arrays["object_phase"][ids].astype(np.float64))
    measured = arrays["modulus_detector"][ids].astype(np.float64)
    chi = ((diffraction(prediction) - measured)**2).sum((-2, -1)) / (measured**2).sum((-2, -1))
    mae = (abs(prediction.real - target.real) + abs(prediction.imag - target.imag)).mean((-2, -1))
    metrics = dict(checkpoint=str(args.checkpoint), epoch=state["epoch"], test_samples=len(ids),
                   supervised_mae=float(mae.mean()), zero_object_mae=float((abs(target.real) + abs(target.imag)).mean()),
                   chi2_mean=float(chi.mean()), chi2_std=float(chi.std()),
                   predicted_amplitude_mean=float(abs(prediction).mean()), target_amplitude_mean=float(abs(target).mean()),
                   note="Raw complex L1, no phase/translation alignment; chi2 is amplitude-domain normalized squared error.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    np.savez_compressed(args.output_dir / "test_predictions.npz", sample_ids=ids,
                        prediction=prediction.astype(np.complex64), mae=mae, chi2=chi)
    logging.info("Test metrics: %s", metrics)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    evaluate(parser.parse_args())
