"""A decode step's attention prologue as one Pallas Triton kernel, for CUDA.

One new token a row: the packed query, key and value heads off `qkv_proj`
go through the per-head RMSNorms, the rotate-half rotation and the dense
cache write, and the query leaves folded for cuDNN's GQA call (each key
head's query heads as its query positions, `Attention._attended`). XLA runs
that as five kernels a layer (two norms, the rotated key's scatter, the
value's, the query's rotation and fold), each a few microseconds of launch
for a few kilobytes; the kernel is one program a row. Its arithmetic is
`rms_normalized`'s and `apply_rotary`'s op for op, but its norm sums a head
in another order than XLA's reduction. What was measured on an RTX 4080:
the caches and query bitwise XLA's in tests/test_decode_prologue.py's
cases, Qwen3-0.6B's served tokens and log-probabilities bitwise in every
run, and 2^30 normed 128-wide values over 2^23 random heads matched to the
bit; but Qwen3-1.7B's served tokens differ, so it is off (`ADOPTED`). It
reads each key head's query heads with a strided slice, which Pallas's
interpreter does not take, so it runs on CUDA only.

JAX 0.11 deprecates the Pallas Triton backend this is written for; its
replacement, Mosaic GPU, targets sm90 and later, and the RTX 4080 is sm89.
Where the backend goes, `fits` is the one place to turn the kernel off.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

ADOPTED = False
"""Off. Serving Qwen3-1.7B on an RTX 4080 the kernel's tokens differ from the
unfused step's (1349 of 8192 at 32 slots, 3328 of 16384 at 64), though
Qwen3-0.6B's were bitwise in every run: its arithmetic is not yet XLA's
(docs/performance.md). On again when both models are bitwise."""

WARPS = 4
"""Warps a program: on an RTX 4080 at Qwen3-0.6B's widths, 28 layers took
0.097 ms at 4 against XLA's 0.191 (docs/performance.md)."""


def fits(heads: int, kv_heads: int, head_dim: int) -> bool:
    """Whether the kernel takes these widths: Triton's blocks are powers of two,
    and a row's packed heads, a group and a half head are each one block."""
    def power(n: int) -> bool:
        return n > 0 and n & (n - 1) == 0

    group, packed = heads // kv_heads, heads + 2 * kv_heads
    return (heads % kv_heads == 0 and power(packed) and power(group) and power(kv_heads)
            and power(head_dim // 2))


def decode_prologue(packed: jax.Array, q_weight: jax.Array, k_weight: jax.Array, cos: jax.Array,
                    sin: jax.Array, slots: jax.Array, key_cache: jax.Array, value_cache: jax.Array, *,
                    heads: int, epsilon: float, scale_after_cast: bool
                    ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Norm, rotate and store one token a row; return the caches and the folded query.

    `packed` is `[rows, heads + 2 * kv_heads, head_dim]`, query heads then key
    heads then value heads; `q_weight` and `k_weight` are the norms' weights
    `[head_dim]` (the stored scale, plus one where the norm offsets it); `cos`
    and `sin` are `[rows, head_dim // 2]` in fp32; `slots` `[rows]` is each
    row's cache slot, -1 to store nothing. The caches are `[rows, capacity,
    kv_heads, head_dim]` and come back written; the query comes back
    `[rows, heads // kv_heads, kv_heads, head_dim]`.
    """
    rows, width, head_dim = packed.shape
    kv_heads = (width - heads) // 2
    group, half = heads // kv_heads, head_dim // 2
    capacity = key_cache.shape[1]
    dtype = packed.dtype

    def kernel(slot_ref, packed_ref, q_weight_ref, k_weight_ref, cos_ref, sin_ref, _key, _value,
               key_ref, value_ref, query_ref):
        row = pl.program_id(0)
        cos, sin = cos_ref[0, :][None, :], sin_ref[0, :][None, :]

        def normed(rows_of, weight_ref):
            # rms_normalized with fp32 statistics, the weight after or before the cast.
            whole = packed_ref[0, rows_of, :].astype(jnp.float32)
            inverse = jax.lax.rsqrt(jnp.sum(whole * whole, axis=1) / head_dim + epsilon)[:, None]
            halves = []
            for offset in (0, half):
                y = packed_ref[0, rows_of, pl.ds(offset, half)].astype(jnp.float32) * inverse
                weight = weight_ref[pl.ds(offset, half)][None, :]
                halves.append((y.astype(dtype) * weight if scale_after_cast else y * weight).astype(dtype))
            return halves

        def rotated(first, second):
            # apply_rotary's rotate-half: [x1, x2] * cos + [-x2, x1] * sin, in fp32.
            first, second = first.astype(jnp.float32), second.astype(jnp.float32)
            return ((first * cos + (-second) * sin).astype(dtype),
                    (second * cos + first * sin).astype(dtype))

        slot = slot_ref[0]
        key_first, key_second = rotated(*normed(pl.ds(heads, kv_heads), k_weight_ref))

        @pl.when((slot >= 0) & (slot < capacity))
        def _():
            key_ref[row, slot, :, pl.ds(0, half)] = key_first.astype(key_ref.dtype)
            key_ref[row, slot, :, pl.ds(half, half)] = key_second.astype(key_ref.dtype)
            values = packed_ref[0, pl.ds(heads + kv_heads, kv_heads), :]
            value_ref[row, slot, :, :] = values.astype(value_ref.dtype)

        for member in range(group):
            # Query head `head * group + member` is member `member` of key head `head`.
            first, second = rotated(*normed(pl.ds(member, kv_heads, group), q_weight_ref))
            query_ref[0, member, :, pl.ds(0, half)] = first
            query_ref[0, member, :, pl.ds(half, half)] = second

    # Imported here: JAX 0.11 deprecates its Pallas Triton backend (in favour of
    # Mosaic GPU, which targets sm90 and later), and only this CUDA step needs it.
    from jax.experimental.pallas import triton as plgpu

    anywhere = pl.BlockSpec(memory_space=pl.ANY)
    return pl.pallas_call(
        kernel, grid=(rows,),
        out_shape=(jax.ShapeDtypeStruct(key_cache.shape, key_cache.dtype),
                   jax.ShapeDtypeStruct(value_cache.shape, value_cache.dtype),
                   jax.ShapeDtypeStruct((rows, group, kv_heads, head_dim), dtype)),
        in_specs=[pl.BlockSpec((1,), lambda row: (row,)),
                  pl.BlockSpec((1, width, head_dim), lambda row: (row, 0, 0)),
                  pl.BlockSpec((head_dim,), lambda row: (0,)), pl.BlockSpec((head_dim,), lambda row: (0,)),
                  pl.BlockSpec((1, half), lambda row: (row, 0)),
                  pl.BlockSpec((1, half), lambda row: (row, 0)),
                  anywhere, anywhere],
        out_specs=(anywhere, anywhere,
                   pl.BlockSpec((1, group, kv_heads, head_dim), lambda row: (row, 0, 0, 0))),
        input_output_aliases={6: 0, 7: 1},
        compiler_params=plgpu.CompilerParams(num_warps=WARPS),
    )(slots.astype(jnp.int32), packed, q_weight, k_weight, cos, sin, key_cache, value_cache)
