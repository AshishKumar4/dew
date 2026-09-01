#!/usr/bin/env python3
"""Time the data pipeline on its own.

Iterates a grain loader for a fixed number of steps and reports throughput and
per-step latency. No model is built, so when a training step looks slow this
says whether the loader is the reason.

Dataset names come from dew.data.registry: datasetMap for the image loader,
mediaDatasetMap with --media.

Usage:
    python tools/benchmark_data.py --dataset oxford_flowers102 --batch-size 32 --steps 100
    python tools/benchmark_data.py --dataset voxceleb2 --media --sequence-length 16 \
        --dataset-path /mnt/data/voxceleb2
"""

import os

# Reading data needs no accelerator, and claiming one costs seconds of startup
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import time
from typing import Iterable, List

from absl import flags

# grain's worker processes read absl flags, and a plain script never parses them
flags.FLAGS.mark_as_parsed()


def measure(batches: Iterable, steps: int, warmup: int) -> List[float]:
    """Seconds spent waiting for each batch, warmup batches dropped.

    The first batches pay for grain's worker startup and for the first reads
    off the filesystem, which says nothing about steady-state throughput.
    """
    latencies = []
    iterator = iter(batches)
    for index in range(warmup + steps):
        start = time.perf_counter()
        if next(iterator, None) is None:
            break
        if index >= warmup:
            latencies.append(time.perf_counter() - start)
    return latencies


def percentile(latencies: List[float], fraction: float) -> float:
    ordered = sorted(latencies)
    return ordered[min(int(fraction * len(ordered)), len(ordered) - 1)]


def report(latencies: List[float], batch_size: int):
    if not latencies:
        raise SystemExit("the loader ran out of batches before the first measured step")
    print(f"steps measured:   {len(latencies)}")
    print(f"samples/sec:      {len(latencies) * batch_size / sum(latencies):.1f}")
    print(f"step latency p50: {percentile(latencies, 0.5) * 1e3:.2f} ms")
    print(f"step latency p95: {percentile(latencies, 0.95) * 1e3:.2f} ms")


def main(args):
    from dew.data.dataloaders import get_dataset_grain, get_media_dataset_grain

    if args.media:
        dataset = get_media_dataset_grain(
            args.dataset,
            batch_size=args.batch_size,
            media_scale=args.image_size,
            sequence_length=args.sequence_length,
            dataset_source=args.dataset_path,
            worker_count=args.worker_count,
            read_thread_count=args.read_thread_count,
        )
    else:
        # An unset path leaves the loader on its own default root
        root = {} if args.dataset_path is None else {"dataset_source": args.dataset_path}
        dataset = get_dataset_grain(
            args.dataset,
            batch_size=args.batch_size,
            image_scale=args.image_size,
            worker_count=args.worker_count,
            read_thread_count=args.read_thread_count,
            **root,
        )

    print(f"{args.dataset}: {dataset['train_len']} records, "
          f"batch {dataset['local_batch_size']} per process, "
          f"{args.worker_count} grain workers")
    report(measure(dataset["train"](), args.steps, args.warmup), dataset["local_batch_size"])


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark a dew data loader")
    parser.add_argument("--dataset", default="oxford_flowers102",
                        help="Dataset name from dew.data.registry")
    parser.add_argument("--batch-size", type=int, default=32, help="Global batch size")
    parser.add_argument("--steps", type=int, default=100, help="Steps to time")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Steps to run before timing starts")
    parser.add_argument("--image-size", type=int, default=128,
                        help="Resolution the augmenter resizes to")
    parser.add_argument("--dataset-path", default=None,
                        help="Root the source reads from, required with --media")
    parser.add_argument("--media", action="store_true",
                        help="Use the media loader (mediaDatasetMap) instead of the image one")
    parser.add_argument("--sequence-length", type=int, default=1,
                        help="Frames per sample, --media only")
    parser.add_argument("--worker-count", type=int, default=8, help="Grain worker processes")
    parser.add_argument("--read-thread-count", type=int, default=16, help="Grain read threads")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
