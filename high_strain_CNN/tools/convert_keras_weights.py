"""Convert the published Keras H5 weights into a PyTorch checkpoint."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from pytorch_autophasenn.model import (
    PUBLISHED_MODEL_VARIANT,
    HighStrainPhaseUNet,
    count_parameters,
)
from pytorch_autophasenn.management import runtime_manifest


PROJECT_DIR = Path(__file__).resolve().parents[1]


LOGGER = logging.getLogger("high_strain.convert")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keras-h5",
        default=str(PROJECT_DIR / "artifacts" / "models" / "model_paper.h5"),
    )
    parser.add_argument(
        "--output",
        default=str(
            PROJECT_DIR / "artifacts" / "models" / "model_paper_pytorch.pt"
        ),
    )
    return parser.parse_args()


def convert_weights(keras_h5: str | Path, output: str | Path) -> Path:
    try:
        import h5py
    except (ImportError, ValueError) as exc:
        raise RuntimeError(
            "A NumPy-compatible h5py installation is required to read Keras weights."
        ) from exc

    keras_h5 = Path(keras_h5).resolve()
    output = Path(output).resolve()
    model = HighStrainPhaseUNet(model_variant=PUBLISHED_MODEL_VARIANT)
    weighted_layers = model.weighted_layers()

    with h5py.File(keras_h5, "r") as handle, torch.no_grad():
        root = handle["model_weights"]
        h5_weighted_layers = {
            name
            for name, group in root.items()
            if "kernel:0" in group and "bias:0" in group
        }
        expected_layers = set(weighted_layers)
        if h5_weighted_layers != expected_layers:
            raise RuntimeError(
                "Keras/PyTorch weighted-layer mismatch: "
                f"missing={sorted(expected_layers - h5_weighted_layers)}, "
                f"unexpected={sorted(h5_weighted_layers - expected_layers)}"
            )

        for name, module in weighted_layers.items():
            group = root[name]
            kernel = np.asarray(group["kernel:0"], dtype=np.float32)
            bias = np.asarray(group["bias:0"], dtype=np.float32)
            converted_kernel = torch.from_numpy(kernel).permute(4, 3, 0, 1, 2)
            if tuple(converted_kernel.shape) != tuple(module.weight.shape):
                raise RuntimeError(
                    f"Weight shape mismatch for {name}: H5={tuple(kernel.shape)}, "
                    f"converted={tuple(converted_kernel.shape)}, "
                    f"PyTorch={tuple(module.weight.shape)}"
                )
            module.weight.copy_(converted_kernel)
            module.bias.copy_(torch.from_numpy(bias))
            LOGGER.info("Converted %-22s %s", name, tuple(module.weight.shape))

    output.parent.mkdir(parents=True, exist_ok=True)
    runtime = runtime_manifest()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_variant": model.model_variant,
            "source_format": "Keras H5 / TensorFlow 2.10.1",
            "source_checkpoint": str(keras_h5),
            "parameter_count": count_parameters(model),
            "input_layout": "NCDHW",
            "output": "reciprocal_space_phase",
            "project_version": runtime["project_version"],
            "git_commit": runtime["git_commit"],
            "runtime": runtime,
        },
        output,
    )
    LOGGER.info("Saved PyTorch checkpoint: %s", output)
    return output


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )
    args = parse_args()
    convert_weights(args.keras_h5, args.output)


if __name__ == "__main__":
    main()
