"""Builders of the timing harness's cases (tools/timing.py): each `builder(cell, implementation)` returns
`(call, draw)`, a compiled call and a draw of its arguments by index.

`dense`: the dense cells of real model projections at a training step's tokens, each product a consumer
launches: the forward (x @ W) and input gradient (dy @ W.T) in bf16, and the kernel gradient (x.T @ dy) into
a training master's fp32 (`gemm_t`'s out_dtype, XLA's preferred_element_type). `implementation` is "xla"
(XLA's dot, cuBLAS) or a grouped matmul's kernel by name ("mosaic_gpu", "triton"), chosen while the call
lowers, which is when a choice acts.

    {"builder": "tools/timing_cases.py:dense", "cells": [{"shape": "llama-3-8b qkv", "product": "forward"}],
     "arms": [{"name": "xla", "implementation": "xla"}, {"name": "mosaic", "implementation": "mosaic_gpu"}]}
"""
import jax
import jax.numpy as jnp
import numpy as np

TOKENS = 8192
"""A training step's tokens per projection (2 x 4096)."""

DENSE = {
    "llama-3-8b qkv": (4096, 6144),
    "llama-3-8b o": (4096, 4096),
    "llama-3-8b gate_up": (4096, 28672),
    "llama-3-8b down": (14336, 4096),
    "qwen3-8b gate_up": (4096, 24576),
    "qwen3-8b down": (12288, 4096),
    "qwen3-32b qkv": (5120, 10240),
    "qwen3-32b gate_up": (5120, 51200),
    "qwen3-32b down": (25600, 5120),
    "llama-3-70b down": (28672, 8192),
    "qwen3.5-4b up": (2560, 9216),
    "qwen3.5-4b down": (9216, 2560),
    "qwen3.5-4b key": (2560, 1024),
}
"""Each projection's width in and width out (its weight's shape)."""


def _products(implementation: str) -> dict:
    """Each product's function of (x, w, dy): XLA's dot, or the GEMM template through its launch by name."""
    if implementation == "xla":
        return {"forward": lambda x, w, dy: jnp.dot(x, w),
                "input_gradient": lambda x, w, dy: jnp.dot(dy, w.T),
                "kernel_gradient": lambda x, w, dy: jnp.dot(x.T, dy, preferred_element_type=jnp.float32)}
    from dew.nn.kernels import gemm

    return {"forward": lambda x, w, dy: gemm.gemm(x, w, None, out_dtype=jnp.bfloat16),
            "input_gradient": lambda x, w, dy: gemm.gemm(dy, w, None, transpose_rhs=True,
                                                          out_dtype=jnp.bfloat16),
            "kernel_gradient": lambda x, w, dy: gemm.gemm_t(x, dy, None, out_dtype=jnp.float32)}


def dense(cell: dict, implementation: str) -> tuple:
    """`cell`'s product (`cell["product"]`) at its shape (`cell["shape"]`, `DENSE`), compiled for
    `implementation`, and its one draw of inputs (seed 0)."""
    from dew.nn.kernels import generation

    width_in, width_out = DENSE[cell["shape"]]
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(TOKENS, width_in)), jnp.bfloat16)
    w = jnp.asarray(rng.normal(size=(width_in, width_out)) / np.sqrt(width_in), jnp.bfloat16)
    dy = jnp.asarray(rng.normal(size=(TOKENS, width_out)), jnp.bfloat16)
    chosen = generation.chosen() if implementation == "xla" else generation.chosen(grouped_matmul=implementation)
    with chosen:
        compiled = jax.jit(_products(implementation)[cell["product"]]).lower(x, w, dy).compile()
    return compiled, lambda index: (x, w, dy)
