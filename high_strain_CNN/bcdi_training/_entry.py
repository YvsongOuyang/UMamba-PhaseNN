"""Small entrypoint adapter; model/trainer implementations remain shared."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ecdi_training.data import load_config
from ecdi_training.train import load_checkpoint
from .validate_data import verify_manifest


PROJECT_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = "/data/wyz/checkpoints"
DATA_CONFIG = "/data/wyz/training_config.json"


def has_option(arguments, key):
    return any(arg.split("=", 1)[0] == key for arg in arguments)


def configured_arguments(defaults, argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    for name, value in defaults.items():
        if not has_option(arguments, name):
            arguments += [name, str(value)]
    return arguments


def check_arguments(arguments):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--phase-checkpoint", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--checkpoint", default="")
    args, _ = parser.parse_known_args(arguments)
    config = load_config(args.data_config)
    if args.data_dir:
        config["root"] = str(Path(args.data_dir).resolve())
    manifest = verify_manifest(config)
    expected = config["provenance"]["manifest_sha256"]
    for path in (args.phase_checkpoint, args.resume, args.checkpoint):
        if path:
            checkpoint = load_checkpoint(path)
            previous = checkpoint["config"].get("provenance", {}).get("manifest_sha256")
            if previous != expected:
                raise ValueError(
                    "Checkpoint belongs to a different generation or seed split."
                )
    return config, manifest


def invoke(entrypoint, arguments):
    original = sys.argv
    try:
        sys.argv = [original[0], *arguments]
        return entrypoint()
    finally:
        sys.argv = original
