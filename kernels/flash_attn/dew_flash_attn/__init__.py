"""FlashAttention-2 for JAX: Dao-AILab's CUDA kernels (BSD-3), through
flash_attn_jax's XLA FFI port (BSD-3), as two FFI targets of a library with
no Python ABI. `flash_mha` takes `[B, S, H, D]` bf16 or fp16 arrays, with
grouped key heads, and differentiates through the kernel's own backward."""

import ctypes
import functools
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_library = ctypes.cdll.LoadLibrary(str(Path(__file__).with_name("libdew_flash_attn.so")))
for _name in ("dew_flash_mha_fwd", "dew_flash_mha_bwd"):
    jax.ffi.register_ffi_target(_name, jax.ffi.pycapsule(getattr(_library, _name)), platform="CUDA")


def _round(x: int, multiple: int) -> int:
    return -(-x // multiple) * multiple


@functools.cache
def _multiprocessors() -> int:
    return jax.devices("gpu")[0].core_count


def _splits(work: int, slots: int, blocks: int) -> int:
    """How many pieces the forward splits the keys into: flash_attn_jax's
    heuristic, the C++ library's, which the scratch buffers are sized by."""
    if work >= 0.8 * slots:
        return 1
    most = min(128, slots, blocks)

    def eligible(splits):
        return splits == 1 or -(-blocks // splits) != -(-blocks // (splits - 1))

    efficiency = [(work * s / slots) / math.ceil(work * s / slots) if eligible(s) else 0.0
                  for s in range(1, most + 1)]
    best = max(efficiency, default=0.0)
    return next((s for s, e in enumerate(efficiency, 1) if eligible(s) and e >= 0.85 * best), 1)


def _padded(*arrays):
    """The arrays with their last axis padded to a multiple of 8, which the kernels take."""
    pad = -arrays[0].shape[-1] % 8
    return tuple(jnp.pad(x, ((0, 0),) * (x.ndim - 1) + ((0, pad),)) for x in arrays) if pad else arrays


def _attributes(scale: float, causal: bool) -> dict:
    return {"softmax_scale": np.float64(scale), "is_causal": causal,
            "window_size_left": np.int64(-1), "window_size_right": np.int64(-1)}


def _forward(query, key, value, scale, causal):
    batch, queries, heads, width = query.shape
    keys = key.shape[1]
    block = 256 if width <= 64 else 128 if width <= 128 else 64
    blocks = max(1, -(-keys // block))
    splits = _splits(batch * heads * max(1, -(-queries // 64)), _multiprocessors() * 2, blocks)
    q, k, v = _padded(query, key, value)
    out, lse, _, _ = jax.ffi.ffi_call("dew_flash_mha_fwd", (
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        jax.ShapeDtypeStruct((batch, heads, queries), jnp.float32),
        jax.ShapeDtypeStruct((splits, batch, queries, heads, _round(width, 32)), jnp.float32),
        jax.ShapeDtypeStruct((splits, batch, heads, queries), jnp.float32),
    ))(q, k, v, **_attributes(scale, causal))
    return out[..., :width], lse


def _backward(grad, query, key, value, out, lse, scale, causal):
    batch, queries, heads, width = query.shape
    keys, key_heads = key.shape[1], key.shape[2]
    g, q, k, v, o = _padded(grad, query, key, value, out)
    padded = q.shape[-1]
    dq, dk, dv, _, _ = jax.ffi.ffi_call("dew_flash_mha_bwd", (
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        jax.ShapeDtypeStruct((batch, keys, heads, padded), q.dtype),
        jax.ShapeDtypeStruct((batch, keys, heads, padded), q.dtype),
        jax.ShapeDtypeStruct((batch, heads, _round(queries, 128)), jnp.float32),
        jax.ShapeDtypeStruct((batch, _round(queries, 128), heads, _round(padded, 32)), jnp.float32),
    ))(g, q, k, v, o, lse, **_attributes(scale, causal), deterministic=False)
    # The kernel writes a key gradient for each query head; a group of them shares a key head.
    group = heads // key_heads
    dk = dk.reshape(batch, keys, key_heads, group, padded).sum(3)
    dv = dv.reshape(batch, keys, key_heads, group, padded).sum(3)
    return dq[..., :width], dk[..., :width], dv[..., :width]


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _attend(query, key, value, scale, causal):
    return _forward(query, key, value, scale, causal)[0]


def _attend_forward(query, key, value, scale, causal):
    out, lse = _forward(query, key, value, scale, causal)
    return out, (query, key, value, out, lse)


def _attend_backward(scale, causal, residuals, grad):
    return _backward(grad, *residuals, scale, causal)


_attend.defvjp(_attend_forward, _attend_backward)


def flash_mha(query, key, value, softmax_scale: float | None = None, is_causal: bool = False):
    """Attention of `query` over `key` and `value`, `[B, S, H, D]`, scaled by
    `softmax_scale` (1/sqrt(D) by default). A causal mask is aligned to the
    last key, as FlashAttention-2 aligns it."""
    if query.dtype not in (jnp.bfloat16, jnp.float16) or not query.dtype == key.dtype == value.dtype:
        raise ValueError(f"flash_mha takes bf16 or fp16 inputs of one dtype, not {query.dtype}, "
                         f"{key.dtype} and {value.dtype}")
    if query.shape[2] % key.shape[2] or key.shape != value.shape or query.shape[3] > 256:
        raise ValueError(f"flash_mha takes key heads that divide the query's and heads up to 256 wide, "
                         f"not query {query.shape}, key {key.shape} and value {value.shape}")
    scale = 1.0 / math.sqrt(query.shape[-1]) if softmax_scale is None else softmax_scale
    return _attend(query, key, value, float(scale), bool(is_causal))
