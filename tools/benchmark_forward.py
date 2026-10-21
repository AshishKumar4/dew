#!/usr/bin/env python3
"""Time one compiled forward pass of an inference module at its published size.

`tools/benchmark_step.py` times what trains; this times what only runs
forward: a vision tower over a batch of images and the Stable Diffusion
VAE's decoder over a batch of latents. The weights are random, since the cost
depends only on the shapes, so no row downloads anything. Each case compiles
once, runs `--warmup` calls, then reports the median of `--repeats` calls,
each synchronized on its output.

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
    args = parser.parse_args()
    results = {}
    for name in args.cases:
        results[name] = measure(name, args.warmup, args.repeats)
        print(f"{name}: {results[name]:.2f} ms", flush=True)
    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
