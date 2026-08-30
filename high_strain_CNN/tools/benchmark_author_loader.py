"""Measure fixed-author NPZ loading/FFT throughput without allocating a model/GPU."""

from __future__ import annotations

import argparse
import logging
import time

import torch

from pytorch_autophasenn.author_data import AuthorNPZPhaseDataset
from pytorch_autophasenn.train import build_loader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--num-samples", type=int, default=0, help="0 reads the full selected split.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--passes", type=int, default=2)
    args = parser.parse_args()
    if args.num_samples < 0 or args.num_workers < 0 or min(args.batch_size, args.prefetch_factor, args.passes) < 1:
        parser.error("Invalid sample, worker, batch, prefetch, or pass count.")
    args.data_format = "author_npz"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    logger = logging.getLogger("high_strain.loader_benchmark")
    dataset = AuthorNPZPhaseDataset(args.data_dir, args.split, num_samples=args.num_samples or None)
    loader = build_loader(dataset, args, torch.device("cpu"), training=False)
    for iteration in range(args.passes):
        started = time.perf_counter()
        samples = 0
        for batch in loader:
            samples += len(batch["name"])
        elapsed = time.perf_counter() - started
        rate = samples / elapsed
        logger.info(
            "Pass=%d | samples=%d | workers=%d | seconds=%.3f | samples/s=%.2f | "
            "projected full-split supply=%.1f min | %s",
            iteration + 1, samples, args.num_workers, elapsed, rate,
            dataset.manifest["available_samples"] / rate / 60,
            "includes worker startup" if iteration == 0 else "warm workers/filesystem cache",
        )


if __name__ == "__main__":
    main()
