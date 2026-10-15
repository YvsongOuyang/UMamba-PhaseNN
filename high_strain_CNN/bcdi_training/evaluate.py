"""Shared paired metrics enriched with BCDI seed and source-sample provenance."""

import argparse
import csv
import json
import sys
from pathlib import Path

from ecdi_training.evaluate import main as evaluate_core
from ._entry import (
    DATA_CONFIG,
    PROJECT_DIR,
    configured_arguments,
    check_arguments,
    invoke,
)


def main():
    arguments = configured_arguments({"--data-config": DATA_CONFIG, "--batch-size": 1})
    if {"-h", "--help"}.intersection(sys.argv):
        return invoke(evaluate_core, arguments)
    _, manifest = check_arguments(arguments)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default="")
    args, _ = parser.parse_known_args(arguments)
    output = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_DIR
        / "artifacts"
        / "evaluations"
        / "bcdi_2d"
        / f"{Path(args.checkpoint).parent.name}_{args.split}"
    )
    if not args.output_dir:
        arguments += ["--output-dir", str(output)]
    result = invoke(evaluate_core, arguments)
    path = output / "metrics.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["dataset"] = "bcdi_simulator_512"
    report["source_phase_convention"] = manifest["phase_convention"]
    report["notes"] = [
        note
        for note in report["notes"]
        if "e-CDI" not in note and "ideal_intensity" not in note
    ]
    report["notes"] += [
        "The source is a noise-free simplified 2D Gaussian-domain BCDI simulator.",
        "Target phase is arg(A*exp(i*phase_factor*stored_phase)), with phase_factor taken from the generator manifest.",
        "Support at threshold 0.3 describes the effective amplitude region under Gaussian illumination.",
    ]
    report["seed_groups"] = manifest["seed_groups"]
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    path = output / "per_sample.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    mappings = manifest["splits"][args.split]
    for row in rows:
        mapping = mappings[int(row["sample"])]
        row.update(
            seed=mapping["seed"],
            source_file=mapping["source_file"],
            source_index=mapping["source_index"],
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
