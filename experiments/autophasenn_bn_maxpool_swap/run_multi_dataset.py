"""Run paired BN/MaxPool evaluations over multiple AutoPhaseNN datasets."""

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

from bn_pool_experiment.evaluator import run_evaluation
from bn_pool_experiment.multi_dataset import (
    build_experiment_config,
    build_multi_dataset_summary,
    load_base_experiment_config,
    load_multi_dataset_config,
    multi_dataset_metric_rows,
    write_multi_dataset_csv,
    write_multi_dataset_report,
)
from bn_pool_experiment.report import (
    write_all_bn_scale_audit_report,
    write_final_output_comparison_report,
    write_markdown_report,
)
from bn_pool_experiment.utils import choose_device, configure_logging, save_json


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse batch-level path, device, and smoke-test overrides."""

    parser = argparse.ArgumentParser(
        description="Evaluate the AutoPhaseNN BN/MaxPool swap on several datasets."
    )
    parser.add_argument(
        "--config",
        default=str(EXPERIMENT_ROOT / "configs" / "server_multi_dataset.yaml"),
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the paired inference batch size for every dataset.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate at most N samples from each enabled dataset.",
    )
    return parser.parse_args()


def main() -> None:
    """Evaluate enabled datasets and save per-dataset plus aggregate reports."""

    args = parse_args()
    multi_path = Path(args.config)
    multi_config = load_multi_dataset_config(multi_path)
    if args.checkpoint is not None:
        multi_config.checkpoint = args.checkpoint
    if args.data_dir is not None:
        multi_config.data_dir = args.data_dir
    if args.output_dir is not None:
        multi_config.output_dir = args.output_dir
    if args.run_name is not None:
        multi_config.run_name = args.run_name
    if args.device is not None:
        multi_config.device = args.device
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be positive.")
        multi_config.batch_size = args.batch_size
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")

    base_config = load_base_experiment_config(multi_path, multi_config)
    run_name = multi_config.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(multi_config.output_dir) / run_name
    configure_logging(base_config.runtime.log_level, output_root / "run.log")
    device_name = multi_config.device or base_config.runtime.device
    device = choose_device(device_name)
    save_json(output_root / "resolved_multi_dataset_config.json", multi_config.to_dict())
    LOGGER.info("Selected device: %s", device)
    LOGGER.info("Output directory: %s", output_root.resolve())

    summaries: dict[str, dict] = {}
    for dataset in multi_config.datasets:
        if not dataset.enabled:
            LOGGER.info("Skipping disabled dataset: %s", dataset.name)
            continue
        config = build_experiment_config(
            base_config,
            multi_config,
            dataset,
            checkpoint=args.checkpoint,
            data_dir=args.data_dir,
            device=args.device,
            batch_size=args.batch_size,
            limit=args.limit,
        )
        dataset_output = output_root / dataset.name
        LOGGER.info(
            "Evaluating dataset=%s, samples=%d, batch_size=%d, diff=%s, real=%s",
            dataset.name,
            config.data.num_samples,
            config.runtime.batch_size,
            Path(config.data.data_dir) / config.data.diff_file,
            (
                Path(config.data.data_dir) / config.data.real_file
                if config.data.real_file
                else None
            ),
        )
        save_json(dataset_output / "resolved_config.json", config.to_dict())
        summary = run_evaluation(config, output_dir=dataset_output, device=device)
        write_markdown_report(dataset_output / "report.md", summary)
        write_final_output_comparison_report(
            dataset_output / "final_output_comparison.md", summary
        )
        write_all_bn_scale_audit_report(
            dataset_output / "all_bn_scale_audit.md", summary
        )
        summaries[dataset.name] = summary

    aggregate = build_multi_dataset_summary(multi_config, summaries)
    save_json(output_root / "multi_dataset_summary.json", aggregate)
    write_multi_dataset_csv(
        output_root / "multi_dataset_metrics.csv",
        multi_dataset_metric_rows(summaries),
    )
    write_multi_dataset_report(
        output_root / "multi_dataset_comparison.md",
        aggregate,
    )
    LOGGER.info(
        "Multi-dataset evaluation complete: datasets=%d, samples=%d, all_pass=%s",
        aggregate["dataset_count"],
        aggregate["total_samples"],
        aggregate["all_datasets_pass"],
    )


if __name__ == "__main__":
    main()
