"""Export a deterministic TensorFlow reference input/output pair."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keras-h5",
        default=str(PROJECT_DIR / "artifacts" / "models" / "model_paper.h5"),
    )
    parser.add_argument(
        "--output",
        default=str(
            PROJECT_DIR
            / "artifacts"
            / "parity"
            / "tensorflow_reference.npz"
        ),
    )
    parser.add_argument("--input", default="", help="Optional NDHWC NumPy input.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--include-intermediates",
        action="store_true",
        help="Also export every parameterized layer output for parity diagnosis.",
    )
    return parser.parse_args()


def main() -> None:
    import tensorflow as tf

    args = parse_args()
    if args.input:
        model_input = np.asarray(np.load(args.input), dtype=np.float32)
    else:
        generator = np.random.default_rng(args.seed)
        model_input = generator.random((1, 64, 64, 64, 1), dtype=np.float32)

    model = tf.keras.models.load_model(args.keras_h5, compile=False)
    arrays = {"input": model_input}
    if args.include_intermediates:
        weighted_layers = [layer for layer in model.layers if layer.weights]
        probe = tf.keras.Model(
            inputs=model.input,
            outputs=[model.output, *(layer.output for layer in weighted_layers)],
        )
        values = probe(model_input, training=False)
        arrays["output"] = values[0].numpy()
        arrays.update(
            {
                f"layer__{layer.name}": value.numpy()
                for layer, value in zip(weighted_layers, values[1:])
            }
        )
    else:
        arrays["output"] = model(model_input, training=False).numpy()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    print(f"TensorFlow reference saved: {destination.resolve()}")
    print(f"input={model_input.shape}, output={arrays['output'].shape}")
    if args.include_intermediates:
        print(f"intermediate layers={len(arrays) - 2}")


if __name__ == "__main__":
    main()
