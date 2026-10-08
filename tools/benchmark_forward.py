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
import statistics
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn


def _siglip() -> tuple[nn.Module, tuple[int, ...]]:
    from dew.nn.vision import SiglipVision

    # google/siglip-so400m-patch14-384, the tower Gemma 3 and PaliGemma 2 read.
    tower = SiglipVision(hidden_size=1152, intermediate_size=4304, num_layers=27, num_heads=16,
                         image_size=384, patch_size=14)
    return tower.build().clone(dtype=jnp.bfloat16), (8, 3, 384, 384)


def _qwen35() -> tuple[nn.Module, tuple[int, ...]]:
    from dew.nn.vision import Qwen35Vision

    # Qwen3.5's tower at its released width and depth, a 448px image a row.
    return Qwen35Vision().build().clone(dtype=jnp.bfloat16), (8, 3, 448, 448)


def _gemma4() -> tuple[nn.Module, tuple[int, ...]]:
    from dew.nn.vision import Gemma4Vision

    # Gemma 4's tower at its released width and depth, 42x42 patches a row.
    return Gemma4Vision().build().clone(dtype=jnp.bfloat16), (4, 3, 672, 672)


def _sd_decoder() -> tuple[nn.Module, tuple[int, ...]]:
    from dew.nn.autoencoders.vae import FlaxDecoder

    # stabilityai/sd-vae-ft-mse's decoder: a 64x64x4 latent to a 512px image.
    return FlaxDecoder(block_out_channels=(128, 256, 512, 512), dtype=jnp.bfloat16), (4, 64, 64, 4)


CASES: dict[str, Callable[[], tuple[nn.Module, tuple[int, ...]]]] = {
    "vision siglip-400m 384px b8": _siglip,
    "vision qwen3.5 448px b8": _qwen35,
    "vision gemma4 672px b4": _gemma4,
    "vae decode sd 512px b4": _sd_decoder,
}


def measure(name: str, warmup: int, repeats: int) -> float:
    """The median milliseconds of one forward call of case `name`."""
    module, shape = CASES[name]()
    inputs = jnp.asarray(np.random.default_rng(0).normal(size=shape), jnp.float32)
    variables = jax.jit(module.init)(jax.random.key(0), inputs)
    forward = jax.jit(module.apply)
    for _ in range(warmup):
        jax.block_until_ready(forward(variables, inputs))
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        jax.block_until_ready(forward(variables, inputs))
        times.append(1e3 * (time.perf_counter() - started))
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser(prog="tools/benchmark_forward.py")
    parser.add_argument("--cases", nargs="*", default=list(CASES), choices=list(CASES))
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
