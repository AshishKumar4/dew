#!/usr/bin/env python3
"""Time one compiled forward pass of an inference module at its published size.

`tools/benchmark_step.py` times what trains; this times what only runs
forward: a vision tower over a batch of images and the Stable Diffusion
VAE's decoder over a batch of latents. The weights are random, since the cost
depends only on the shapes, so no row downloads anything. Each case compiles
once, runs `--warmup` calls, then reports the median of `--repeats` calls,
each synchronized on its output. Its measured peak device memory is a process of
its own's (tools/peak_memory.py), run before this one opens a backend: a row
is `{"ms", "peak_device_bytes"}`.

Usage:
    python tools/benchmark_forward.py
    python tools/benchmark_forward.py --cases "vision siglip-400m 384px b8" --json-out forward.json
"""

import argparse
import json
import os
import statistics
import sys

import jax
import jax.numpy as jnp
import numpy as np

from dew.telemetry.benchmark import samples

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import peak_memory
from benchmark_cases import FORWARD


def measure(name: str, warmup: int, repeats: int) -> float:
    """The median milliseconds of one forward call of case `name`."""
    module, shape = FORWARD[name]()
    inputs = jnp.asarray(np.random.default_rng(0).normal(size=shape), jnp.float32)
    variables = jax.jit(module.init)(jax.random.key(0), inputs)
    forward = jax.jit(module.apply)
    return 1e3 * statistics.median(samples(forward, variables, inputs, warmup=warmup, repeats=repeats))


def main() -> None:
    parser = argparse.ArgumentParser(prog="tools/benchmark_forward.py")
    parser.add_argument("--cases", nargs="*", default=list(FORWARD), choices=list(FORWARD))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--json-out")
    parser.add_argument("--peak-run", type=int,
                        help="run case N of --cases alone, for its measured peak (tools/peak_memory.py)")
    args = parser.parse_args()
    if args.peak_run is not None:
        measure(args.cases[args.peak_run], args.warmup, args.repeats)
        return
    # Before this process opens a backend, which would hold the device each case's peak process needs.
    peaks = [peak_memory.measured([os.path.abspath(__file__), *sys.argv[1:], "--peak-run", str(index)])
             for index in range(len(args.cases))]
    results = {}
    for name, peak in zip(args.cases, peaks, strict=True):
        results[name] = {"ms": measure(name, args.warmup, args.repeats),
                         "peak_device_bytes": None if None in peak else max(peak)}
        print(f"{name}: {results[name]['ms']:.2f} ms", flush=True)
    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
