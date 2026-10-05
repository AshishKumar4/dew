"""A decode step's attention over a row's own keys, as a Pallas kernel for CUDA.

A serving step decodes every slot of its cache. jax.nn's xla attention, and
the folded form that reads the cache in its own layout
(`dew.nn.attention.folded_attention`), compute every row over the cache's
whole capacity, idle rows and the slots past a row's length included.
Each program here takes one row and one key head, and reads only the blocks
of keys below the row's length, with an online softmax over them (flash
decoding without splits). Six of Qwen3.5-0.8B's 256-wide layers at a
384-slot capacity, 128 rows with 16 drawing, took 0.28 ms against the folded
form's 1.12 and jax.nn's 2.40 on an RTX 4080, and 0.20 against 0.33 and 0.41
at 32 rows (docs/performance.md). Its body is whole-block arithmetic and
dots, so a Mosaic GPU port takes it as is.

The arithmetic is jax.nn's but for the order of its sums: fp32 logits of
the operands, the scale after, an fp32 softmax, its probabilities in the
value's dtype for the value product, which accumulates in fp32 and is
normalized at the end rather than before the product.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

BLOCK = 32
"""Keys a step of the loop reads."""
QUERIES = 16
"""Query positions a program holds: the fewest a tensor-core dot takes."""
WARPS = 4


def fits(query: jax.Array, key: jax.Array) -> bool:
    """Whether the kernel takes this `[B, T, N, D]` query over `[B, S, N, D]`
    keys: CUDA, at most `QUERIES` positions, a power-of-two width up to 256,
    and a capacity of whole blocks."""
    positions, width = query.shape[1], query.shape[-1]
    return (jax.default_backend() == "gpu" and positions <= QUERIES and width & (width - 1) == 0
            and 16 <= width <= 256 and key.shape[1] % BLOCK == 0 and query.dtype == key.dtype)


def _kernel(lengths_ref, query_ref, key_ref, value_ref, out_ref, *, scale: float):
    from jax.experimental.pallas import triton as plgpu

    length = lengths_ref[0]
    query = query_ref[...]                                        # [QUERIES, D]
    width = query.shape[-1]

    def body(block, carry):
        out, peak, total = carry
        keys = pl.ds(block * BLOCK, BLOCK)
        logits = plgpu.dot(query, key_ref[keys, :].T).astype(jnp.float32) * scale   # [QUERIES, BLOCK]
        read = block * BLOCK + jnp.arange(BLOCK) < length
        logits = jnp.where(read[None, :], logits, -0.7 * float(jnp.finfo(jnp.float32).max))
        new_peak = jnp.maximum(peak, logits.max(axis=-1))
        correction = jnp.exp(peak - new_peak)
        weights = jnp.exp(logits - new_peak[:, None])
        value = value_ref[keys, :]
        out = correction[:, None] * out + plgpu.dot(weights.astype(value.dtype), value).astype(jnp.float32)
        return out, new_peak, correction * total + weights.sum(axis=-1)

    start = (jnp.zeros((QUERIES, width), jnp.float32),
             jnp.full((QUERIES,), float(jnp.finfo(jnp.float32).min), jnp.float32),
             jnp.zeros((QUERIES,), jnp.float32))
    out, _, total = jax.lax.fori_loop(0, pl.cdiv(length, BLOCK), body, start)
    out_ref[...] = (out / total[:, None]).astype(out_ref.dtype)


def attend(query: jax.Array, key: jax.Array, value: jax.Array, lengths: jax.Array) -> jax.Array:
    """Attention of `query` `[B, T, N, D]` over the first `lengths` `[B]` of
    `key` and `value` `[B, S, N, D]` (the same heads), `[B, T, N, D]` out."""
    from jax.experimental.pallas import triton as plgpu

    batch, positions, heads, width = query.shape
    padded = jnp.pad(query, ((0, 0), (0, QUERIES - positions), (0, 0), (0, 0)))
    keyed = pl.BlockSpec((None, key.shape[1], None, width), lambda row, head: (row, 0, head, 0))
    queried = pl.BlockSpec((None, QUERIES, None, width), lambda row, head: (row, 0, head, 0))
    out = pl.pallas_call(
        lambda *refs: _kernel(*refs, scale=1 / math.sqrt(width)),
        grid=(batch, heads),
        in_specs=[pl.BlockSpec((None, 1), lambda row, head: (row, 0)), queried, keyed, keyed],
        out_specs=queried,
        out_shape=jax.ShapeDtypeStruct(padded.shape, query.dtype),
        compiler_params=plgpu.CompilerParams(num_warps=WARPS, num_stages=2),
    )(jnp.asarray(lengths, jnp.int32)[:, None], padded, key, value)
    return out[:, :positions]
