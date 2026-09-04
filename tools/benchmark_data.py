#!/usr/bin/env python3
"""Time the data pipeline on its own.

Iterates a dataset's training stream for a fixed number of steps and reports
throughput and per-step latency. No model is built, so when a training step
looks slow this says whether the loader is the reason.

The dataset is a registered spec and its fields are the command line, so the
knobs here are the knobs a run has:

    python tools/benchmark_data.py --steps 100 data:oxford-flowers --data.image-size 128
    python tools/benchmark_data.py data:vox-celeb2 --data.path /mnt/data/voxceleb2 --data.frames 16
"""

import os

# Reading data needs no accelerator, and claiming one costs seconds of startup
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from typing import Iterable, List  # noqa: E402

import tyro  # noqa: E402
from absl import flags  # noqa: E402

# grain's worker processes read absl flags, and a plain script never parses them
flags.FLAGS.mark_as_parsed()

import dew  # noqa: E402,F401  registers the datasets
from dew.data import OxfordFlowers  # noqa: E402
from dew.registry import datasets  # noqa: E402


@dataclass(frozen=True)
class Benchmark:
    data: datasets.union = field(default_factory=OxfordFlowers)
    """Which dataset, with its own fields as flags."""
    batch: int = 32
    """Global batch size."""
    steps: int = 100
    """Steps to time."""
    warmup: int = 5
    """Steps to run before timing starts."""


def measure(batches: Iterable, steps: int, warmup: int) -> List[float]:
    """Wall time per batch after warmup, in seconds."""
    latencies = []
    for index, _ in enumerate(batches):
        if index >= warmup + steps:
            break
        now = time.perf_counter()
        if index >= warmup:
            latencies.append(now - last)
        last = now
    return latencies


def percentile(values: List[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def report(latencies: List[float], batch_size: int) -> None:
    if not latencies:
        raise SystemExit("the loader ran out of batches before the first measured step")
    print(f"steps measured:   {len(latencies)}")
    print(f"samples/sec:      {len(latencies) * batch_size / sum(latencies):.1f}")
    print(f"step latency p50: {percentile(latencies, 0.5) * 1e3:.2f} ms")
    print(f"step latency p95: {percentile(latencies, 0.95) * 1e3:.2f} ms")


def main(config: Benchmark) -> None:
    dataset = config.data.load(batch=config.batch)
    print(f"{datasets.name_of(type(config.data))}: {dataset.records} records, "
          f"batch {dataset.batch} across every process")
    report(measure(dataset.train(), config.steps, config.warmup), dataset.batch)


if __name__ == "__main__":
    main(tyro.cli(tyro.conf.CascadeSubcommandArgs[Benchmark]))
