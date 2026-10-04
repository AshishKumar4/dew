"""A decode step's rotation, cache write and query fold as one Pallas Triton
kernel, for CUDA.

One new token a row: the normed query and key heads take the rotate-half
rotation, the key and value go into the dense cache, and the query leaves
folded for cuDNN's GQA call (each key head's query heads as its query
positions, `Attention._attended`). XLA ran the rotated key's scatter, the
value's, and the query's rotation and fold as three kernels a layer of one
to three microseconds each; this is one program a row. The norms stay XLA's
own fusions: a kernel that normed too summed each head in another order
than XLA's reduction and served Qwen3-1.7B other tokens
(docs/performance.md). The rotation is `apply_rotary`'s arithmetic op for
op, so the caches and query are bitwise the unfused step's.

It reads each key head's query heads with a strided slice, which Pallas's
interpreter does not take, so it runs on CUDA only. JAX 0.11 deprecates the
Pallas Triton backend it is written for; its replacement, Mosaic GPU,
targets sm90 and later, and the RTX 4080 is sm89. Where the backend goes,
`fits` is the one place to turn the kernel off.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

WARPS = 4
"""Warps a program."""


def fits(heads: int, kv_heads: int, head_dim: int) -> bool:
    """Whether the kernel takes these widths: Triton's blocks are powers of two,
    and the query heads, the key heads, a group and a half head are each one block."""
    def power(n: int) -> bool:
        return n > 0 and n & (n - 1) == 0

    return (heads % kv_heads == 0 and power(heads) and power(kv_heads) and power(heads // kv_heads)
            and power(head_dim // 2))


def decode_prologue(query: jax.Array, key: jax.Array, value: jax.Array, cos: jax.Array, sin: jax.Array,
                    slots: jax.Array, key_cache: jax.Array, value_cache: jax.Array
                    ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Rotate and store one token a row; return the caches and the folded query.

    `query` is `[rows, heads, head_dim]` and `key` and `value` `[rows, kv_heads,
    head_dim]`, the query and key normed; `cos` and `sin` are `[rows,
    head_dim // 2]` in fp32; `slots` `[rows]` is each row's cache slot, -1 to
    store nothing. The caches are `[rows, capacity, kv_heads, head_dim]` and
    come back written; the query comes back `[rows, heads // kv_heads,
    kv_heads, head_dim]`.
    """
    rows, heads, head_dim = query.shape
    kv_heads = key.shape[1]
    group, half = heads // kv_heads, head_dim // 2
    capacity = key_cache.shape[1]
    dtype = query.dtype

    def kernel(slot_ref, query_ref, key_ref, value_ref, cos_ref, sin_ref, _key, _value,
               key_out, value_out, query_out):
        row = pl.program_id(0)
        cos, sin = cos_ref[0, :][None, :], sin_ref[0, :][None, :]

        def rotated(ref, rows_of):
            # apply_rotary's rotate-half: [x1, x2] * cos + [-x2, x1] * sin, in fp32.
            first = ref[0, rows_of, pl.ds(0, half)].astype(jnp.float32)
            second = ref[0, rows_of, pl.ds(half, half)].astype(jnp.float32)
            return ((first * cos + (-second) * sin).astype(dtype),
                    (second * cos + first * sin).astype(dtype))

        slot = slot_ref[0]
        key_first, key_second = rotated(key_ref, pl.ds(0, kv_heads))

        @pl.when((slot >= 0) & (slot < capacity))
        def _():
            key_out[row, slot, :, pl.ds(0, half)] = key_first.astype(key_out.dtype)
            key_out[row, slot, :, pl.ds(half, half)] = key_second.astype(key_out.dtype)
            value_out[row, slot, :, :] = value_ref[0, :, :].astype(value_out.dtype)

        for member in range(group):
            # Query head `head * group + member` is member `member` of key head `head`.
            first, second = rotated(query_ref, pl.ds(member, kv_heads, group))
            query_out[0, member, :, pl.ds(0, half)] = first
            query_out[0, member, :, pl.ds(half, half)] = second

    # Imported here: JAX 0.11 deprecates its Pallas Triton backend, and only
    # this CUDA step needs it.
    from jax.experimental.pallas import triton as plgpu

    anywhere = pl.BlockSpec(memory_space=pl.ANY)
    return pl.pallas_call(
        kernel, grid=(rows,),
        out_shape=(jax.ShapeDtypeStruct(key_cache.shape, key_cache.dtype),
                   jax.ShapeDtypeStruct(value_cache.shape, value_cache.dtype),
                   jax.ShapeDtypeStruct((rows, group, kv_heads, head_dim), dtype)),
        in_specs=[pl.BlockSpec((1,), lambda row: (row,)),
                  pl.BlockSpec((1, heads, head_dim), lambda row: (row, 0, 0)),
                  pl.BlockSpec((1, kv_heads, head_dim), lambda row: (row, 0, 0)),
                  pl.BlockSpec((1, kv_heads, head_dim), lambda row: (row, 0, 0)),
                  pl.BlockSpec((1, half), lambda row: (row, 0)),
                  pl.BlockSpec((1, half), lambda row: (row, 0)),
                  anywhere, anywhere],
        out_specs=(anywhere, anywhere,
                   pl.BlockSpec((1, group, kv_heads, head_dim), lambda row: (row, 0, 0, 0))),
        input_output_aliases={6: 0, 7: 1},
        compiler_params=plgpu.CompilerParams(num_warps=WARPS),
    )(slots.astype(jnp.int32), query, key, value, cos, sin, key_cache, value_cache)
