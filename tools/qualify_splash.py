#!/usr/bin/env python3
"""Qualify `attention_impl 'tpu'` on a real TPU: the parity suite against
Mosaic, then what the splash kernel buys at long context.

tests/test_attention_splash.py runs anywhere, because `tpu_attention` asks
pallas for its interpreter off a TPU backend. That proves the arithmetic and
the mask descriptor; it does not prove the Mosaic kernel, which is the one
that runs in training. So this tool runs the same file first, with the
backend that compiles it for real, and only then times it.

The timing is the case the change exists for: 16 heads of width 128 at 2k, 8k
and 32k keys, causal, bf16, against XLA's attention on the same shapes and
against the pallas flash kernel this path used to reach for unconditionally.
One example per row, because 32k x 16 x 128 in bf16 is 134 MiB per tensor and
the backward pass holds several. Splash's claim is the block-sparse one: a
causal row should cost about half its rectangle and pull further ahead as the
sequence grows, while flash pays the whole rectangle at every length and
cannot take a mask at all without materializing [B, H, Q, K].

A kernel that refuses a shape is a row with served=False and the reason,
which is part of the answer rather than an error.

Usage:
    JAX_PLATFORMS=tpu PYTHONPATH=src python tools/qualify_splash.py
    JAX_PLATFORMS=tpu PYTHONPATH=src python tools/qualify_splash.py \\
        --sequence-lengths 2048 8192 --skip-parity --out splash.json
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

# tests/conftest.py defaults the platform to cpu, and it is imported after
# this module, so the backend this asks for has to be named before jax opens
# one. Naming it here rather than requiring the variable keeps the parity run
# and the timing on the same devices.
os.environ.setdefault("JAX_PLATFORMS", "tpu")

import jax
import jax.numpy as jnp
import tyro

from dew.nn.attention import pallas_flash_attention, scaled_dot_product_attention

KERNELS = ('splash', 'flash', 'xla')


@dataclass(frozen=True)
class QualifyConfig:
    """Which shapes to time, and whether to run the parity suite first."""

    sequence_lengths: tuple[int, ...] = (2048, 8192, 32768)
    heads: int = 16
    head_dim: int = 128
    batch: int = 1
    kernels: tuple[str, ...] = KERNELS
    causal: bool = True
    skip_parity: bool = False
    """Time only. The parity run is the half that says the numbers are right."""
    forward_only: bool = False
    warmup: int = 2
    steps: int = 10
    out: str | None = None
    parity_args: tuple[str, ...] = ()
    """Extra arguments for the pytest run, `-k causal` for instance."""


def parity(config: QualifyConfig) -> dict[str, int | str]:
    """The parity suite, run in this process against this backend."""
    import pytest

    arguments = ["tests/test_attention_splash.py", "-q", *config.parity_args]
    code = int(pytest.main(arguments))
    return {"command": " ".join(["pytest", *arguments]), "exit_code": code,
            "verdict": "parity holds" if code == 0 else "parity failed"}


def kernel_fn(kernel: str, causal: bool):
    """One callable per kernel, all reading Dew's own [B, S, H, D] layout.

    'splash' and 'flash' are both `attention_impl 'tpu'`; the dispatch picks
    splash for a call this shaped, so the flash row calls the wrapper the
    dispatch keeps for a bias or a traced mask directly. That is the kernel
    this change moves off, measured as it was reached before.
    """
    if kernel == 'flash':
        return lambda q, k, v: pallas_flash_attention(q, k, v, None, None, causal, None)
    return lambda q, k, v: scaled_dot_product_attention(
        q, k, v, implementation='tpu' if kernel == 'splash' else 'xla', causal=causal)


def timed(run, *arrays, warmup: int, steps: int) -> float:
    """Milliseconds per call, after the compile and a warm cache."""
    for _ in range(warmup):
        jax.block_until_ready(run(*arrays))
    start = time.perf_counter()
    for _ in range(steps):
        jax.block_until_ready(run(*arrays))
    return (time.perf_counter() - start) * 1e3 / steps


def measure(kernel: str, seq: int, config: QualifyConfig) -> dict[str, float | bool | str | int]:
    """One row: kernel x sequence length, forward and forward+backward.

    jax raises ValueError for a shape a kernel's own checks refuse,
    NotImplementedError for one its lowering has no rule for, and
    JaxRuntimeError when the arrays or the materialized logits do not fit,
    which at 32k keys is what separates the two TPU kernels.
    """
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    shape = (config.batch, seq, config.heads, config.head_dim)
    arrays = tuple(jax.random.normal(key, shape, jnp.bfloat16) for key in keys)
    row: dict[str, float | bool | str | int] = {
        "kernel": kernel, "sequence_length": seq, "heads": config.heads,
        "head_dim": config.head_dim, "batch": config.batch, "causal": config.causal}
    attention = kernel_fn(kernel, config.causal)
    refusals = (ValueError, NotImplementedError, jax.errors.JaxRuntimeError)
    try:
        forward = jax.jit(attention)
        row["forward_ms"] = timed(forward, *arrays, warmup=config.warmup, steps=config.steps)
    except refusals as refused:
        return {**row, "served": False,
                "reason": f"{type(refused).__name__}: {str(refused).splitlines()[0][:200]}"}
    row["served"] = True
    if config.forward_only:
        return row
    def loss(q, k, v):
        return jnp.sum(attention(q, k, v).astype(jnp.float32) ** 2)
    try:
        backward = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))
        row["train_ms"] = timed(backward, *arrays, warmup=config.warmup, steps=config.steps)
    except refusals as refused:
        row["backward_reason"] = (
            f"{type(refused).__name__}: {str(refused).splitlines()[0][:200]}")
    return row


def main(config: QualifyConfig) -> int:
    backend = jax.default_backend()
    report: dict[str, object] = {
        "backend": backend, "devices": [str(device) for device in jax.devices()],
        "jax": jax.__version__}
    if backend != 'tpu':
        report["verdict"] = (
            f"this qualifies the Mosaic kernel and the backend is {backend}; "
            "run it on a TPU, or run tests/test_attention_splash.py alone for "
            "the interpreter's numbers")
        print(json.dumps(report, indent=2))
        return 2
    if not config.skip_parity:
        report["parity"] = parity(config)
    report["rows"] = [measure(kernel, seq, config)
                      for seq in config.sequence_lengths for kernel in config.kernels]
    print(json.dumps(report, indent=2))
    if config.out:
        with open(config.out, "w") as handle:
            json.dump(report, handle, indent=2)
    parity_code = report.get("parity")
    return 1 if isinstance(parity_code, dict) and parity_code["exit_code"] else 0


if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(QualifyConfig)))
