"""Compare converted PyTorch output with an exported TensorFlow reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pytorch_port.model import PUBLISHED_MODEL_VARIANT, HighStrainPhaseUNet


def error_metrics(pytorch_value: np.ndarray, tensorflow_value: np.ndarray) -> dict:
    difference = pytorch_value - tensorflow_value
    max_abs = float(np.max(np.abs(difference)))
    reference_scale = float(np.max(np.abs(tensorflow_value)))
    return {
        "max_abs_error": max_abs,
        "mean_abs_error": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "reference_max_abs": reference_scale,
        "relative_max_error": max_abs / max(reference_scale, 1e-12),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="model_paper_pytorch.pt")
    parser.add_argument(
        "--reference",
        default="parity_output/tensorflow_reference.npz",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-abs-tolerance", type=float, default=1e-3)
    parser.add_argument("--report", default="parity_output/parity_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = np.load(args.reference)
    tensorflow_input = np.asarray(reference["input"], dtype=np.float32)
    tensorflow_output = np.asarray(reference["output"], dtype=np.float32)
    pytorch_input = torch.from_numpy(tensorflow_input).permute(0, 4, 1, 2, 3)

    device = torch.device(args.device)
    model = HighStrainPhaseUNet(model_variant=PUBLISHED_MODEL_VARIANT).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    model.eval()
    captured: dict[str, torch.Tensor] = {}
    hooks = []
    for name, layer in model.layers.items():
        hooks.append(
            layer.register_forward_hook(
                lambda _module, _inputs, output, layer_name=name: captured.__setitem__(
                    layer_name, output.detach().cpu()
                )
            )
        )
    with torch.no_grad():
        pytorch_output = model(pytorch_input.to(device)).cpu().permute(0, 2, 3, 4, 1).numpy()
    for hook in hooks:
        hook.remove()

    output_metrics = error_metrics(pytorch_output, tensorflow_output)
    intermediate_metrics = {}
    for key in reference.files:
        if not key.startswith("layer__"):
            continue
        layer_name = key.removeprefix("layer__")
        pytorch_value = captured[layer_name].permute(0, 2, 3, 4, 1).numpy()
        intermediate_metrics[layer_name] = {
            "shape": list(pytorch_value.shape),
            **error_metrics(pytorch_value, np.asarray(reference[key], dtype=np.float32)),
        }

    report = {
        "tensorflow_shape": list(tensorflow_output.shape),
        "pytorch_shape": list(pytorch_output.shape),
        **output_metrics,
        "intermediate_layers": intermediate_metrics,
        "tolerance": args.max_abs_tolerance,
        "passed": output_metrics["max_abs_error"] <= args.max_abs_tolerance,
    }
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(
            f"Parity failed: max_abs_error={output_metrics['max_abs_error']:.6g} exceeds "
            f"{args.max_abs_tolerance:.6g}."
        )


if __name__ == "__main__":
    main()
