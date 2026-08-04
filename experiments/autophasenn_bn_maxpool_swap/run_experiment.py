"""CLI entry point for the AutoPhaseNN BN/MaxPool swap experiment."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
for import_root in (REPOSITORY_ROOT, EXPERIMENT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from bn_pool_experiment.config import load_experiment_config
from bn_pool_experiment.evaluator import run_evaluation
from bn_pool_experiment.report import (
    write_all_bn_scale_audit_report,
    write_final_output_comparison_report,
    write_markdown_report,
)
from bn_pool_experiment.utils import choose_device, configure_logging, save_json


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line overrides for the YAML configuration."""

    parser = argparse.ArgumentParser(
        description="Quantify AutoPhaseNN BN/MaxPool layer-order equivalence."
    )
    parser.add_argument(
        "--config",
        default=str(EXPERIMENT_ROOT / "configs" / "default.yaml"),
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N configured samples for a smoke run.",
    )
    return parser.parse_args()


def apply_overrides(config, args: argparse.Namespace) -> None:
    """Apply explicit CLI overrides without changing the source YAML."""

    if args.checkpoint is not None:
        config.model.checkpoint = args.checkpoint
    if args.data_dir is not None:
        config.data.data_dir = args.data_dir
    if args.output_dir is not None:
        config.output.root_dir = args.output_dir
    if args.run_name is not None:
        config.output.run_name = args.run_name
    if args.device is not None:
        config.runtime.device = args.device
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive.")
        config.data.num_samples = min(config.data.num_samples, args.limit)


def resolve_output_dir(config) -> Path:
    """Create a unique run directory unless output.run_name is configured."""

    run_name = config.output.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(config.output.root_dir) / run_name


def main() -> None:
    """Run evaluation, persist artifacts, and report the acceptance decision."""

    args = parse_args()
    config = load_experiment_config(args.config)
    apply_overrides(config, args)
    output_dir = resolve_output_dir(config)
    configure_logging(config.runtime.log_level, output_dir / "run.log")
    device = choose_device(config.runtime.device)
    LOGGER.info("Selected device: %s", device)
    LOGGER.info("Output directory: %s", output_dir.resolve())
    save_json(output_dir / "resolved_config.json", config.to_dict())

    summary = run_evaluation(config, output_dir=output_dir, device=device)
    write_markdown_report(output_dir / "report.md", summary)
    write_final_output_comparison_report(
        output_dir / "final_output_comparison.md",
        summary,
    )
    write_all_bn_scale_audit_report(
        output_dir / "all_bn_scale_audit.md",
        summary,
    )
    LOGGER.info(
        "Experiment complete: overall_pass=%s, report=%s",
        summary["acceptance"]["overall_pass"],
        (output_dir / "report.md").resolve(),
    )


if __name__ == "__main__":
    main()
