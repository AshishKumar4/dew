#!/usr/bin/env python3
"""A MoE layer's grouped expert products at decode batch sizes (row 3622): rows = tokens x top-k, a few a
expert, each route of the grouped selector (`grouped_matmul.GROUPED`: Mosaic GPU's template, Triton's,
CUTLASS's by name, XLA's) timed on the device by CUPTI over routings drawn fresh for every call, as a decode
step's router draws them (a routing reused would time one placement of the rows). One JSON line a cell.

    python tools/benchmark_moe_decode.py --out moe_decode.jsonl
"""

import argparse
import json
import statistics

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.mosaic.gpu import profiler

from dew.nn.kernels import generation
from dew.nn.kernels.templates.gemm import template

MODELS = {
    "qwen3-30b-a3b": {"hidden": 2048, "intermediate": 768, "experts": 128, "top_k": 8},
    "gpt-oss-20b": {"hidden": 2880, "intermediate": 2880, "experts": 32, "top_k": 4},
    "deepseek-v3": {"hidden": 7168, "intermediate": 2048, "experts": 256, "top_k": 8},
}
"""Each model's expert shapes: its hidden width, an expert's intermediate width, its experts and top-k."""

TOKENS = (1, 2, 4, 8, 16, 32, 64)
"""A decode step's tokens, the default; `--tokens` names others (a prefill's or a small batch's)."""
ROUTES = ("mosaic_gpu", "triton", "cutlass", "xla")
ROUTINGS = 32
"""The routings drawn a cell, each call its own."""


def routings(tokens: int, experts: int, top_k: int, seed: int) -> list[np.ndarray]:
    """`ROUTINGS` routings of `tokens` tokens: each token's `top_k` distinct experts, uniform, as group sizes
    (each expert's rows)."""
    rng = np.random.default_rng(seed)
    drawn = []
    for _ in range(ROUTINGS):
        chosen = np.stack([rng.choice(experts, top_k, replace=False) for _ in range(tokens)])
        drawn.append(np.bincount(chosen.reshape(-1), minlength=experts).astype(np.int32))
    return drawn


def timed(fn, arguments: list[tuple]) -> list[float]:
    """Each call's device seconds (CUPTI, its kernels alone), one call an argument tuple, after one warm
    call of each."""
    for args in arguments:
        jax.block_until_ready(fn(*args))
    cupti = profiler.Cupti(finalize=False)
    seconds = []
    for args in arguments:
        _, read = cupti.measure(fn, aggregate=False)(*args)
        seconds.append(sum(ms for _, ms in read or []) / 1e3)
    profiler.Cupti().measure(fn, aggregate=False)(*arguments[0])
    return seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--tokens", default=",".join(map(str, TOKENS)))
    args = parser.parse_args()
    device = jax.devices()[0].device_kind
    with open(args.out, "w") as out:
        for name in args.models.split(","):
            shape = MODELS[name]
            hidden, inter, experts, top_k = (shape[key] for key in ("hidden", "intermediate", "experts",
                                                                     "top_k"))
            rng = np.random.default_rng(0)
            # The two projections a decode step's experts run: gate and up together, then down.
            weights = {"gate_up": jnp.asarray(rng.normal(size=(experts, hidden, 2 * inter)) / np.sqrt(hidden),
                                              jnp.bfloat16),
                       "down": jnp.asarray(rng.normal(size=(experts, inter, hidden)) / np.sqrt(inter),
                                           jnp.bfloat16)}
            for tokens in map(int, args.tokens.split(",")):
                rows = tokens * top_k
                sizes = [jnp.asarray(drawn) for drawn in routings(tokens, experts, top_k, seed=tokens)]
                for projection, w in weights.items():
                    x = jnp.asarray(rng.normal(size=(rows, w.shape[1])), jnp.bfloat16)
                    for route in ROUTES:
                        row = {"device": device, "model": name, "tokens": tokens, "rows": rows,
                               "projection": projection, "route": route}
                        try:
                            with generation.chosen(grouped_matmul=route):
                                fn = jax.jit(lambda x, w, s: template.gemm(x, w, s, out_dtype=jnp.bfloat16))
                                seconds = timed(fn, [(x, w, s) for s in sizes])
                            # Each call's weights read: its routing's active experts' matrices.
                            matrix = w.shape[1] * w.shape[2] * 2
                            read = [float((np.asarray(s) > 0).sum()) * matrix for s in sizes]
                            row.update(median_us=round(statistics.median(seconds) * 1e6, 2),
                                       least_us=round(min(seconds) * 1e6, 2),
                                       weight_tb_per_s=round(statistics.median(
                                           [b / t / 1e12 for b, t in zip(read, seconds, strict=True)]), 3))
                        except Exception as error:
                            row["error"] = f"{type(error).__name__}: {str(error)[:200]}"
                        print(json.dumps(row), flush=True)
                        out.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
