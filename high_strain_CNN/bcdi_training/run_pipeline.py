"""Prepare/check BCDI data, train both stages, evaluate the test split and plot."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from ecdi_training.run_pipeline import pipeline_commands as shared_commands
from ._entry import PROJECT_DIR, DATA_CONFIG, CHECKPOINT_ROOT


def pipeline_commands(args, run_dir):
    commands = shared_commands(args, run_dir)
    for _, command in commands:
        for index, token in enumerate(command):
            if token.startswith("ecdi_training."):
                command[index] = token.replace("ecdi_training.", "bcdi_training.", 1)
    for stage, command in commands:
        rate = (
            args.phase_lr
            if stage == "phase"
            else args.refiner_lr if stage == "refiner" else None
        )
        if rate is not None:
            command += ["--learning-rate", str(rate)]
    if args.source_dir:
        destination = (
            Path(args.data_dir).resolve()
            if args.data_dir
            else Path(args.data_config).resolve().parent
        )
        expected_config = destination / "training_config.json"
        if expected_config != Path(args.data_config).resolve():
            raise ValueError(
                "With --source-dir, --data-config must point to <prepared output>/training_config.json."
            )
        commands.insert(
            0,
            (
                "prepare",
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "bcdi_training.prepare_data",
                    "--source-dir",
                    str(Path(args.source_dir).resolve()),
                    "--output-dir",
                    str(destination),
                ],
            ),
        )
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        default="",
        help="Optional: prepare a new dataset first; destination must be empty.",
    )
    parser.add_argument("--data-config", default=DATA_CONFIG)
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--run-name", default="")
    parser.add_argument(
        "--runs-dir", default=str(PROJECT_DIR / "artifacts" / "training" / "bcdi_2d")
    )
    parser.add_argument("--checkpoint-root", default=CHECKPOINT_ROOT)
    parser.add_argument("--phase-batch-size", type=int, default=16)
    parser.add_argument("--refiner-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--phase-epochs", type=int)
    parser.add_argument("--refiner-epochs", type=int)
    parser.add_argument("--phase-lr", type=float)
    parser.add_argument("--refiner-lr", type=float, default=4e-4)
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
    if args.source_dir and args.start_stage != "phase":
        parser.error(
            "Prepare once at the start of a new run; omit --source-dir for resume/evaluation."
        )
    args.run_name = args.run_name or f"bcdi_2d512_{datetime.now():%Y%m%d_%H%M%S}"
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
        parser.error(
            "Batch sizes must be positive and sample/worker counts nonnegative."
        )
    if any(
        value is not None and value < 1
        for value in (args.phase_epochs, args.refiner_epochs)
    ):
        parser.error("Epochs must be positive.")
    import math

    if any(
        value is not None and (not math.isfinite(value) or value <= 0)
        for value in (args.phase_lr, args.refiner_lr)
    ):
        parser.error("Learning rates must be finite and positive.")
    run_dir = Path(args.runs_dir).resolve() / args.run_name
    commands = pipeline_commands(args, run_dir)
    if not args.dry_run:
        import torch

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this environment.")
        from pytorch_autophasenn.mamba_block import build_official_mamba

        # Validate the actual implementation before committing to stage 1.
        probe = build_official_mamba(d_model=8, d_state=16, d_conv=4, expand=2)
        del probe
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot
        except (ImportError, OSError) as error:
            raise RuntimeError(
                "Plotting dependencies are incompatible; install bcdi_training/requirements.txt before starting the full pipeline."
            ) from error
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        force=True,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "pipeline.log", encoding="utf-8"),
        ],
    )
    (run_dir / "pipeline_config.json").write_text(
        json.dumps({"args": vars(args), "commands": commands}, indent=2),
        encoding="utf-8",
    )
    for stage, command in commands:
        logging.info("Stage=%s | %s", stage, subprocess.list2cmdline(command))
        if args.dry_run:
            continue
        status = run_dir / "pipeline_status.json"
        with subprocess.Popen(command, cwd=PROJECT_DIR) as child:
            old_handler = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, lambda *_: child.terminate())
            status.write_text(
                json.dumps(
                    {"status": "running", "stage": stage, "child_pid": child.pid}
                ),
                encoding="utf-8",
            )
            try:
                code = child.wait()
            except KeyboardInterrupt:
                child.terminate()
                child.wait()
                status.write_text(
                    json.dumps({"status": "interrupted", "stage": stage}),
                    encoding="utf-8",
                )
                raise
            finally:
                signal.signal(signal.SIGTERM, old_handler)
        if code:
            status.write_text(
                json.dumps({"status": "failed", "stage": stage, "exit_code": code}),
                encoding="utf-8",
            )
            logging.error("%s failed (%s); pipeline stopped.", stage, code)
            return code
        status.write_text(
            json.dumps({"status": "running", "completed_stage": stage}),
            encoding="utf-8",
        )
    if not args.dry_run:
        (run_dir / "pipeline_status.json").write_text(
            json.dumps({"status": "complete"}), encoding="utf-8"
        )
    logging.info(
        "BCDI pipeline %s | %s", "planned" if args.dry_run else "complete", run_dir
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
