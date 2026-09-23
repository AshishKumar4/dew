#!/usr/bin/env python3
"""Qualify `attention_impl 'tpu'` on a real TPU: the parity suite against
Mosaic, then what the splash kernel buys over XLA's attention.

tests/test_attention_splash.py runs anywhere, because `tpu_attention` asks
pallas for its interpreter off a TPU backend. That proves the arithmetic and
the mask descriptor; it does not prove the Mosaic kernel, which is the one
that runs in training. So this tool runs the same file first, with the
backend that compiles it for real, and only then times it. This module
imports jax before pytest's conftest sets JAX_DEFAULT_MATMUL_PRECISION to
highest, which jax reads once, at import. Under highest Mosaic refuses
splash's bf16 matmuls ("Bad lhs type" for a bf16 `tpu.matmul` at fp32
contract precision, jax 0.11.1), so the parity runs at DEFAULT, the
precision training runs at.

Each row is one kernel, one call shape and one sequence length, at a fixed
number of tokens per step (the batch shrinks as the sequence grows), 16 heads
of width 128 over 8 key heads, bf16. The shapes are the calls 'auto' decides
between on a TPU:

- `causal`: a decoder's full attention;
- `softcap`: Gemma 2's tanh cap of 50 on the logits, causal;
- `sinks`: GPT-OSS's per-head sink logit, causal;
- `packed`: four documents per row by segment ids, causal;
- `packed_window`: the same rows under a 1024-key sliding window.

A row carries the forward time, the forward-plus-backward time and the
compiled executables' temporary HBM (`memory_analysis().temp_size_in_bytes`),
which is where XLA's [B, H, S, S] logits live. A kernel that refuses a shape,
or runs out of memory, is a row with served=False and the reason, which is
part of the answer rather than an error. `--xla-only-where-faster` is not a
flag here: which kernel `auto` picks is read off these rows by hand and
written, with the numbers, at `tpu_runs`.

Usage:
    JAX_PLATFORMS=tpu PYTHONPATH=src python tools/qualify_splash.py
    JAX_PLATFORMS=tpu PYTHONPATH=src python tools/qualify_splash.py \\
        --sequence-lengths 2048 8192 --shapes causal packed --skip-parity --out splash.json
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
import numpy as np
import tyro

from dew.nn.attention import pallas_flash_attention, scaled_dot_product_attention

KERNELS = ('splash', 'flash', 'xla')
SHAPES = ('causal', 'softcap', 'sinks', 'packed', 'packed_window')
DOCUMENTS = 4
WINDOW = 1024


@dataclass(frozen=True)
class QualifyConfig:
    """Which shapes to time, and whether to run the parity suite first."""

    sequence_lengths: tuple[int, ...] = (512, 1024, 2048, 4096, 8192, 16384)
    tokens: int = 16384
    """Tokens per step; each row's batch is this over its sequence length, at least 1."""
    heads: int = 16
    kv_heads: int = 8
    head_dim: int = 128
    kernels: tuple[str, ...] = KERNELS
    shapes: tuple[str, ...] = SHAPES
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


def call_arguments(shape: str, batch: int, seq: int, heads: int) -> dict[str, object]:
    """The keyword arguments of `scaled_dot_product_attention` one shape names."""
    arguments: dict[str, object] = {"causal": True}
    if shape == 'softcap':
        arguments["softcap"] = 50.0
    elif shape == 'sinks':
        arguments["sinks"] = jax.random.normal(jax.random.PRNGKey(3), (heads,), jnp.float32)
    elif shape in ('packed', 'packed_window'):
        documents = np.repeat(np.arange(1, DOCUMENTS + 1), seq // DOCUMENTS)
        arguments["segment_ids"] = jnp.asarray(np.broadcast_to(documents, (batch, seq)),
                                               jnp.int32)
        if shape == 'packed_window':
            arguments["sliding_window"] = WINDOW
    return arguments


def kernel_fn(kernel: str, arguments: dict[str, object]):
    """One callable per kernel, all reading Dew's own [B, S, H, D] layout.

    'splash' and 'flash' are both `attention_impl 'tpu'`; the dispatch picks
    splash for every call this tool makes, so the flash row calls the
    wrapper the dispatch keeps for a bias or a traced mask directly, with
    the segment ids the packed shapes carry.
    """
    if kernel == 'flash':
        if set(arguments) - {"causal", "segment_ids"}:
            raise ValueError("the flash kernel takes no softcap, sinks or window flag")
        segment_ids = arguments.get("segment_ids")
        return lambda q, k, v: pallas_flash_attention(q, k, v, None, None, True, None,
                                                      segment_ids)
    implementation = 'tpu' if kernel == 'splash' else 'xla'
    return lambda q, k, v: scaled_dot_product_attention(
        q, k, v, implementation=implementation, **arguments)


def timed(run, *arrays, warmup: int, steps: int) -> float:
    """Milliseconds per call, after the compile and a warm cache."""
    for _ in range(warmup):
        jax.block_until_ready(run(*arrays))
    start = time.perf_counter()
    for _ in range(steps):
        jax.block_until_ready(run(*arrays))
    return (time.perf_counter() - start) * 1e3 / steps


def compiled(fn, *arrays):
    """The executable and its temporary HBM in MiB."""
    executable = jax.jit(fn).lower(*arrays).compile()
    analysis = executable.memory_analysis()
    temp = None if analysis is None else analysis.temp_size_in_bytes / 2 ** 20
    return executable, temp


def measure(kernel: str, shape: str, seq: int,
            config: QualifyConfig) -> dict[str, float | bool | str | int | None]:
    """One row: kernel x shape x sequence length, forward and forward+backward.

    jax raises ValueError for a shape a kernel's own checks refuse,
    NotImplementedError for one its lowering has no rule for, and
    JaxRuntimeError when the arrays or the materialized logits do not fit.
    """
    batch = max(1, config.tokens // seq)
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    q = jax.random.normal(keys[0], (batch, seq, config.heads, config.head_dim), jnp.bfloat16)
    k, v = (jax.random.normal(key, (batch, seq, config.kv_heads, config.head_dim), jnp.bfloat16)
            for key in keys[1:])
    row: dict[str, float | bool | str | int | None] = {
        "kernel": kernel, "shape": shape, "sequence_length": seq, "batch": batch,
        "heads": config.heads, "kv_heads": config.kv_heads, "head_dim": config.head_dim}
    refusals = (ValueError, NotImplementedError, jax.errors.JaxRuntimeError)
    try:
        attention = kernel_fn(kernel, call_arguments(shape, batch, seq, config.heads))
        forward, row["forward_temp_mib"] = compiled(attention, q, k, v)
        row["forward_ms"] = timed(forward, q, k, v, warmup=config.warmup, steps=config.steps)
    except refusals as refused:
        return {**row, "served": False,
                "reason": f"{type(refused).__name__}: {str(refused).splitlines()[0][:200]}"}
    row["served"] = True
    if config.forward_only:
        return row

    def loss(q, k, v):
        return jnp.sum(attention(q, k, v).astype(jnp.float32) ** 2)

    try:
        backward, row["train_temp_mib"] = compiled(jax.grad(loss, argnums=(0, 1, 2)), q, k, v)
        row["train_ms"] = timed(backward, q, k, v, warmup=config.warmup, steps=config.steps)
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
    rows = []
    for shape in config.shapes:
        for seq in config.sequence_lengths:
            for kernel in config.kernels:
                rows.append(measure(kernel, shape, seq, config))
                print(json.dumps(rows[-1]), flush=True)
    report["rows"] = rows
    print(json.dumps(report, indent=2))
    if config.out:
        with open(config.out, "w") as handle:
            json.dump(report, handle, indent=2)
    parity_code = report.get("parity")
    return 1 if isinstance(parity_code, dict) and parity_code["exit_code"] else 0


if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(QualifyConfig)))
