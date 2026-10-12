"""FlashAttention for JAX: Dao-AILab's CUDA kernels (BSD-3) as FFI targets of
libraries with no Python ABI. FlashAttention-2 comes through flash_attn_jax's
XLA FFI port (BSD-3) and runs on sm8x and sm120; FlashAttention-3, Hopper's,
comes through flash-attn3-jax's port (BSD-3) and runs on sm90. `flash_mha`
takes `[B, S, H, D]` bf16 or fp16 arrays, with grouped key heads, and
`flash_mha_varlen` packed sequences `[T, H, D]` between cumulative offsets;
both run the kernel built for the device and differentiate through its own
backward."""

import ctypes
import functools
import math
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def _load(name: str, report: str) -> tuple[ctypes.CDLL, tuple[str, ...]]:
    """The library and the GPU generations it reports holding code for, its build's CUDA
    architectures ('80-real' is sm80, '90a-real' sm90)."""
    library = ctypes.cdll.LoadLibrary(str(Path(__file__).with_name(name)))
    getattr(library, report).restype = ctypes.c_char_p
    entries = getattr(library, report)().decode().split(",")
    return library, tuple(f"sm{re.match(r'[0-9]+', entry).group()}" for entry in entries)


_library, _two = _load("libdew_flash_attn.so", "dew_flash_architectures")
_library3, _three = _load("libdew_flash_attn3.so", "dew_flash3_architectures")
ARCHITECTURES = _two + _three
"""The GPU generations the libraries hold code for, as their builds compiled them ('sm80', 'sm90',
'sm120')."""
for _name in ("dew_flash_mha_fwd", "dew_flash_mha_bwd",
              "dew_flash_mha_varlen_fwd", "dew_flash_mha_varlen_bwd"):
    jax.ffi.register_ffi_target(_name, jax.ffi.pycapsule(getattr(_library, _name)), platform="CUDA")
for _name in ("dew_flash3_mha_fwd", "dew_flash3_mha_bwd", "dew_flash3_mha_varlen_fwd", "dew_flash3_mha_varlen_bwd"):
    jax.ffi.register_ffi_target(_name, jax.ffi.pycapsule(getattr(_library3, _name)), platform="CUDA")


def _round(x: int, multiple: int) -> int:
    return -(-x // multiple) * multiple


@functools.cache
def _multiprocessors() -> int:
    return jax.devices("gpu")[0].core_count


@functools.cache
def _hopper() -> bool:
    """Whether the device runs FlashAttention-3, which is built for sm90 alone."""
    return f"sm{jax.devices('gpu')[0].compute_capability.replace('.', '')}" in _three


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


def _forward3(query, key, value, starts, longest, scale, causal):
    """FlashAttention-3's forward over `[B, S, H, D]` arrays, or, given cumulative offsets `starts`,
    over packed `[T, H, D]` ones whose sequences hold at most `longest` tokens."""
    q, k, v = _padded(query, key, value)
    heads = q.shape[-2]
    packed = starts is not None
    results = (jax.ShapeDtypeStruct(q.shape, q.dtype),
               jax.ShapeDtypeStruct((heads, q.shape[0]) if packed else (q.shape[0], heads, q.shape[1]),
                                    jnp.float32),
               jax.ShapeDtypeStruct((1, 1, 1, 1), jnp.float32), jax.ShapeDtypeStruct((1, 1, 1), jnp.float32))
    attributes = {"softmax_scale": np.float64(scale), "is_causal": causal}
    if packed:
        out, lse, _, _ = jax.ffi.ffi_call("dew_flash3_mha_varlen_fwd", results)(
            q, k, v, starts, max_seqlen=np.int64(longest), **attributes)
    else:
        out, lse, _, _ = jax.ffi.ffi_call("dew_flash3_mha_fwd", results)(q, k, v, **attributes)
    return out[..., :query.shape[-1]], lse


def _backward3(grad, query, key, value, out, lse, starts, longest, scale, causal):
    """FlashAttention-3's backward, whose key and value gradients come per key head."""
    width = query.shape[-1]
    g, q, k, v, o = _padded(grad, query, key, value, out)
    packed = starts is not None
    # Two scratch buffers of rows per head, each sequence's queries rounded up to the backward's row
    # block, which the port picks from 64, 80, 96 and 128 by width and mask, and packed one block
    # more over all of them: 128 rows a sequence and 128 more hold either layout.
    tokens, sequences = (q.shape[0], starts.shape[0] - 1) if packed else (q.shape[0] * q.shape[1], q.shape[0])
    scratch = jax.ShapeDtypeStruct((q.shape[-2] * (tokens + 128 * (sequences + 1)),), jnp.float32)
    results = (jax.ShapeDtypeStruct(q.shape, q.dtype), jax.ShapeDtypeStruct(k.shape, q.dtype),
               jax.ShapeDtypeStruct(v.shape, q.dtype), scratch, scratch)
    attributes = {"softmax_scale": np.float64(scale), "is_causal": causal}
    if packed:
        dq, dk, dv, _, _ = jax.ffi.ffi_call("dew_flash3_mha_varlen_bwd", results)(
            g, q, k, v, o, lse, starts, max_seqlen=np.int64(longest), **attributes)
    else:
        dq, dk, dv, _, _ = jax.ffi.ffi_call("dew_flash3_mha_bwd", results)(g, q, k, v, o, lse, **attributes)
    return dq[..., :width], dk[..., :width], dv[..., :width]


def _forward(query, key, value, scale, causal):
    if _hopper():
        return _forward3(query, key, value, None, None, scale, causal)
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
    if _hopper():
        return _backward3(grad, query, key, value, out, lse, None, None, scale, causal)
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
    last key, as FlashAttention aligns it."""
    if query.dtype not in (jnp.bfloat16, jnp.float16) or not query.dtype == key.dtype == value.dtype:
        raise ValueError(f"flash_mha takes bf16 or fp16 inputs of one dtype, not {query.dtype}, "
                         f"{key.dtype} and {value.dtype}")
    if query.shape[2] % key.shape[2] or key.shape != value.shape or query.shape[3] > 256:
        raise ValueError(f"flash_mha takes key heads that divide the query's and heads up to 256 wide, "
                         f"not query {query.shape}, key {key.shape} and value {value.shape}")
    scale = 1.0 / math.sqrt(query.shape[-1]) if softmax_scale is None else softmax_scale
    return _attend(query, key, value, float(scale), bool(is_causal))


def _varlen_forward(query, key, value, starts, longest, scale, causal):
    if _hopper():
        return _forward3(query, key, value, starts, longest, scale, causal)
    heads, width = query.shape[1:]
    sequences = starts.shape[0] - 1
    q, k, v = _padded(query, key, value)
    # The packed forward never splits the keys, so its split scratch is a token buffer.
    out, lse, _, _ = jax.ffi.ffi_call("dew_flash_mha_varlen_fwd", (
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        jax.ShapeDtypeStruct((sequences, heads, longest), jnp.float32),
        jax.ShapeDtypeStruct((1, 1, 1, 1, 1), jnp.float32),
        jax.ShapeDtypeStruct((1, 1, 1, 1), jnp.float32),
    ))(q, k, v, starts, starts, max_seqlen_q=np.int32(longest), max_seqlen_k=np.int32(longest),
       zero_tensors=False, **_attributes(scale, causal))
    return out[..., :width], lse


def _varlen_backward(grad, query, key, value, out, lse, starts, longest, scale, causal):
    if _hopper():
        return _backward3(grad, query, key, value, out, lse, starts, longest, scale, causal)
    total, heads, width = query.shape
    key_heads = key.shape[1]
    sequences = starts.shape[0] - 1
    g, q, k, v, o = _padded(grad, query, key, value, out)
    padded = q.shape[-1]
    attributes = _attributes(scale, causal)
    attributes["softmax_scale"] = np.float32(scale)
    dq, dk, dv, _, _ = jax.ffi.ffi_call("dew_flash_mha_varlen_bwd", (
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        jax.ShapeDtypeStruct((key.shape[0], heads, padded), q.dtype),
        jax.ShapeDtypeStruct((key.shape[0], heads, padded), q.dtype),
        jax.ShapeDtypeStruct((sequences, heads, _round(longest, 128)), jnp.float32),
        jax.ShapeDtypeStruct((total + 128 * sequences, heads, _round(padded, 32)), jnp.float32),
    ))(g, q, k, v, o, lse, starts, starts, max_seqlen_q=np.int64(longest), max_seqlen_k=np.int64(longest),
       zero_tensors=False, deterministic=False, **attributes)
    group = heads // key_heads
    dk = dk.reshape(key.shape[0], key_heads, group, padded).sum(2)
    dv = dv.reshape(key.shape[0], key_heads, group, padded).sum(2)
    return dq[..., :width], dk[..., :width], dv[..., :width]


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def _attend_packed(query, key, value, starts, longest, scale, causal):
    return _varlen_forward(query, key, value, starts, longest, scale, causal)[0]


def _attend_packed_forward(query, key, value, starts, longest, scale, causal):
    out, lse = _varlen_forward(query, key, value, starts, longest, scale, causal)
    return out, (query, key, value, out, lse, starts)


def _attend_packed_backward(longest, scale, causal, residuals, grad):
    query, key, value, out, lse, starts = residuals
    dq, dk, dv = _varlen_backward(grad, query, key, value, out, lse, starts, longest, scale, causal)
    return dq, dk, dv, np.zeros(starts.shape, jax.dtypes.float0)


_attend_packed.defvjp(_attend_packed_forward, _attend_packed_backward)


def flash_mha_varlen(query, key, value, cu_seqlens, max_seqlen: int, softmax_scale: float | None = None,
                     is_causal: bool = False):
    """Attention within each of the sequences packed into `query`, `key` and `value`, `[T, H, D]`:
    sequence i holds tokens `cu_seqlens[i]` to `cu_seqlens[i + 1]` (int32, `[N + 1]`, from 0 to T),
    none longer than `max_seqlen`, and a sequence may be empty. A causal mask is each sequence's own."""
    if query.dtype not in (jnp.bfloat16, jnp.float16) or not query.dtype == key.dtype == value.dtype:
        raise ValueError(f"flash_mha_varlen takes bf16 or fp16 inputs of one dtype, not {query.dtype}, "
                         f"{key.dtype} and {value.dtype}")
    if (query.ndim != 3 or query.shape[1] % key.shape[1] or key.shape != value.shape
            or query.shape[0] != key.shape[0] or query.shape[2] > 256):
        raise ValueError(f"flash_mha_varlen takes packed [T, H, D] arrays whose key heads divide the query's "
                         f"and heads up to 256 wide, not query {query.shape}, key {key.shape} and value "
                         f"{value.shape}")
    scale = 1.0 / math.sqrt(query.shape[-1]) if softmax_scale is None else softmax_scale
    starts = jnp.asarray(cu_seqlens, jnp.int32)
    return _attend_packed(query, key, value, starts, int(max_seqlen), float(scale), bool(is_causal))
