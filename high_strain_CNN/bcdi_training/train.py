"""Train the existing 2D U-Net or frozen-U-Net MoMamba on prepared BCDI data."""

import sys

from ecdi_training.train import main as train_core
from ._entry import (
    DATA_CONFIG,
    CHECKPOINT_ROOT,
    PROJECT_DIR,
    configured_arguments,
    check_arguments,
    invoke,
)


def main():
    arguments = configured_arguments(
        {
            "--data-config": DATA_CONFIG,
            "--checkpoint-root": CHECKPOINT_ROOT,
            "--runs-dir": PROJECT_DIR / "artifacts" / "training" / "bcdi_2d" / "stages",
            "--batch-size": 16,
        }
    )
    if not {"-h", "--help"}.intersection(sys.argv):
        check_arguments(arguments)
    return invoke(train_core, arguments)


if __name__ == "__main__":
    raise SystemExit(main())
