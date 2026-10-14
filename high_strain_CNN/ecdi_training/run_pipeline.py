"""Sequential e-CDI preflight, phase training, frozen-phase refinement and test."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .train import PROJECT_DIR, TRAINING_DIR, DEFAULT_CHECKPOINT_ROOT


def pipeline_commands(args, run_dir: Path):
    config = str(Path(args.data_config).resolve())
    checkpoint_root = str(Path(args.checkpoint_root).resolve())
    phase_name, refiner_name = args.run_name + "_phase", args.run_name + "_refiner"
    phase_path = Path(checkpoint_root) / phase_name / "checkpoint_best.pt"
    refiner_path = Path(checkpoint_root) / refiner_name / "checkpoint_best.pt"
    data = ["--data-config", config]
    if args.data_dir:
        data += ["--data-dir", args.data_dir]
    training = data + [
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--checkpoint-root",
        checkpoint_root,
        "--runs-dir",
        str(run_dir / "stages"),
    ]
    commands = [
        (
            "validate",
            [
                sys.executable,
                "-u",
                "-m",
                "ecdi_training.validate_data",
                *data,
                "--output",
                str(run_dir / "data_validation.json"),
            ],
        )
    ]
    for stage, name, batch, epoch, source in (
        (
            "phase",
            phase_name,
            args.phase_batch_size,
            args.phase_epochs,
            args.phase_resume,
        ),
        (
            "refiner",
            refiner_name,
            args.refiner_batch_size,
            args.refiner_epochs,
            args.refiner_resume,
        ),
    ):
        if args.start_stage == "evaluate" or (
            args.start_stage == "refiner" and stage == "phase"
        ):
            continue
        command = [
            sys.executable,
            "-u",
            "-m",
            "ecdi_training.train",
            "--stage",
            stage,
            "--run-name",
            name,
            "--batch-size",
            str(batch),
            *training,
        ]
        if epoch is not None:
            command += ["--epochs", str(epoch)]
        if args.max_batches_per_epoch:
            command += ["--max-batches-per-epoch", str(args.max_batches_per_epoch)]
        if args.num_samples_train:
            command += ["--num-samples-train", str(args.num_samples_train)]
        if args.num_samples_val:
            command += ["--num-samples-val", str(args.num_samples_val)]
        if source:
            command += ["--resume", source]
        if stage == "refiner" and not source:
            command += ["--phase-checkpoint", str(phase_path)]
        commands.append((stage, command))
    output = run_dir / "evaluation"
    evaluation = [
        sys.executable,
        "-u",
        "-m",
        "ecdi_training.evaluate",
        "--checkpoint",
        str(refiner_path),
        *data,
        "--split",
        "test",
        "--batch-size",
        str(args.refiner_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--output-dir",
        str(output),
    ]
    if args.num_samples_test:
        evaluation += ["--num-samples", str(args.num_samples_test)]
    commands.append(("evaluate", evaluation))
    commands.append(
        (
            "visualize",
            [
                sys.executable,
                "-u",
                "-m",
                "ecdi_training.visualize",
                "--evaluation-dir",
                str(output),
            ],
        )
    )
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-config", default=str(TRAINING_DIR / "configs" / "ecdi_2d.json")
    )
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--run-name", default="")
    parser.add_argument(
        "--runs-dir", default=str(PROJECT_DIR / "artifacts" / "training" / "ecdi_2d")
    )
    parser.add_argument("--checkpoint-root", default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--phase-batch-size", type=int, default=1)
    parser.add_argument("--refiner-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--phase-epochs", type=int)
    parser.add_argument("--refiner-epochs", type=int)
    parser.add_argument("--phase-resume", default="")
    parser.add_argument("--refiner-resume", default="")
    parser.add_argument(
        "--start-stage", choices=("phase", "refiner", "evaluate"), default="phase"
    )
    parser.add_argument("--num-samples-train", type=int, default=0)
    parser.add_argument("--num-samples-val", type=int, default=0)
    parser.add_argument("--num-samples-test", type=int, default=0)
    parser.add_argument("--max-batches-per-epoch", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.refiner_resume and args.start_stage == "phase":
        args.start_stage = "refiner"
    args.run_name = args.run_name or f"ecdi_2d_{datetime.now():%Y%m%d_%H%M%S}"
    if Path(args.run_name).name != args.run_name:
        parser.error("--run-name must be a single directory name.")
    if (
        min(args.phase_batch_size, args.refiner_batch_size) < 1
        or min(
            args.num_workers,
            args.num_samples_train,
            args.num_samples_val,
            args.num_samples_test,
            args.max_batches_per_epoch,
        )
        < 0
    ):
        parser.error("Batch sizes must be positive and counts nonnegative.")
    if any(
        value is not None and value < 1
        for value in (args.phase_epochs, args.refiner_epochs)
    ):
        parser.error("Epochs must be positive.")
    run_dir = Path(args.runs_dir).resolve() / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "pipeline.log", encoding="utf-8"),
        ],
    )
    commands = pipeline_commands(args, run_dir)
    if not args.dry_run:
        import torch

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this Python environment.")
        try:
            from mamba_ssm import Mamba
        except (ImportError, OSError) as error:
            raise RuntimeError(
                "Install the official Mamba dependency in ecdi_training/requirements/mamba.txt before starting both stages."
            ) from error
    (run_dir / "pipeline_config.json").write_text(
        json.dumps({"args": vars(args), "commands": commands}, indent=2),
        encoding="utf-8",
    )
    for stage, command in commands:
        logging.info("Stage=%s | %s", stage, subprocess.list2cmdline(command))
        if args.dry_run:
            continue
        logging.info("Starting %s", stage)
        try:
            with subprocess.Popen(command, cwd=PROJECT_DIR) as child:
                previous_handler = signal.getsignal(signal.SIGTERM)
                signal.signal(signal.SIGTERM, lambda *_: child.terminate())
                (run_dir / "pipeline_status.json").write_text(
                    json.dumps(
                        {"status": "running", "stage": stage, "child_pid": child.pid}
                    ),
                    encoding="utf-8",
                )
                try:
                    return_code = child.wait()
                except KeyboardInterrupt:
                    child.terminate()
                    child.wait()
                    raise
                finally:
                    signal.signal(signal.SIGTERM, previous_handler)
                if return_code:
                    raise subprocess.CalledProcessError(return_code, command)
        except subprocess.CalledProcessError as error:
            logging.error(
                "%s failed with exit code %s; later stages were not started.",
                stage,
                error.returncode,
            )
            (run_dir / "pipeline_status.json").write_text(
                json.dumps(
                    {"status": "failed", "stage": stage, "exit_code": error.returncode}
                ),
                encoding="utf-8",
            )
            return error.returncode
        (run_dir / "pipeline_status.json").write_text(
            json.dumps({"status": "running", "completed_stage": stage}),
            encoding="utf-8",
        )
    if not args.dry_run:
        (run_dir / "pipeline_status.json").write_text(
            json.dumps({"status": "complete"}), encoding="utf-8"
        )
    logging.info(
        "Pipeline %s | run=%s", "planned" if args.dry_run else "complete", run_dir
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
