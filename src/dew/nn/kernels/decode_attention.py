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

`attend_paged` is the same kernel over a paged cache: a program takes one
row and one key head with the query heads that share it, and reads the row's
pages through its page table. JAX's GPU paged kernel
(`jax.experimental.pallas.ops.gpu.paged_attention`) stores its unnormalized
sums and its split partials in the query's dtype and divides by a bf16
denominator, three roundings where this has one.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from .generation import triton_runs

BLOCK = 32
"""Keys a step of the loop reads."""
QUERIES = 16
"""Query positions a program holds: the fewest a tensor-core dot takes."""
WARPS = 4


def fits(query: jax.Array, key: jax.Array) -> bool:
    """Whether the kernel takes this `[B, T, N, D]` query over `[B, S, N, D]`
    keys: a GPU Dew's Triton kernels run on (`triton_runs`), at most
    `QUERIES` positions, a power-of-two width up to 256, and a capacity of
    whole blocks."""
    positions, width = query.shape[1], query.shape[-1]
    return (triton_runs() and positions <= QUERIES and width & (width - 1) == 0
            and 16 <= width <= 256 and key.shape[1] % BLOCK == 0 and query.dtype == key.dtype)


def _attend(query: jax.Array, length: jax.Array, read, block: int, scale: float) -> jax.Array:
    """`query` `[QUERIES, D]` over the first `length` keys, `block` at a time,
    `read(step)` giving a step's keys and values `[block, D]`: an online
    softmax, normalized at the end."""
    from jax.experimental.pallas import triton as plgpu

    width = query.shape[-1]

    def body(step, carry):
        out, peak, total = carry
        key, value = read(step)
        logits = plgpu.dot(query, key.T).astype(jnp.float32) * scale              # [QUERIES, block]
        reads = step * block + jnp.arange(block) < length
        logits = jnp.where(reads[None, :], logits, -0.7 * float(jnp.finfo(jnp.float32).max))
        new_peak = jnp.maximum(peak, logits.max(axis=-1))
        correction = jnp.exp(peak - new_peak)
        weights = jnp.exp(logits - new_peak[:, None])
        product = plgpu.dot(weights.astype(value.dtype), value).astype(jnp.float32)
        out = correction[:, None] * out + product
        return out, new_peak, correction * total + weights.sum(axis=-1)

    start = (jnp.zeros((QUERIES, width), jnp.float32),
             jnp.full((QUERIES,), float(jnp.finfo(jnp.float32).min), jnp.float32),
             jnp.zeros((QUERIES,), jnp.float32))
    out, _, total = jax.lax.fori_loop(0, pl.cdiv(length, block), body, start)
    return out / total[:, None]


def _kernel(lengths_ref, query_ref, key_ref, value_ref, out_ref, *, scale: float):
    def read(step):
        keys = pl.ds(step * BLOCK, BLOCK)
        return key_ref[keys, :], value_ref[keys, :]

    out_ref[...] = _attend(query_ref[...], lengths_ref[0], read, BLOCK, scale).astype(out_ref.dtype)


def _paged_kernel(lengths_ref, table_ref, query_ref, key_ref, value_ref, out_ref, *, pages: int,
                  scale: float):
    page_size, width = key_ref.shape[-2:]

    def read(step):
        held = table_ref[pl.ds(step * pages, pages)]
        return (key_ref[held].reshape(pages * page_size, width),
                value_ref[held].reshape(pages * page_size, width))

    out = _attend(query_ref[...], lengths_ref[0], read, pages * page_size, scale)
    out_ref[...] = out.astype(out_ref.dtype)


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


def fits_paged(query: jax.Array | jax.ShapeDtypeStruct, key_pages: jax.Array | jax.ShapeDtypeStruct) -> bool:
    """Whether `attend_paged` takes this `[rows, heads, D]` query over
    `[kv_heads, pages, page_size, D]` pools: a GPU Dew's Triton kernels run on
    (`triton_runs`), at most `QUERIES` query heads a key head, a power-of-two
    width up to 256, and a power-of-two page of at most `BLOCK` keys."""
    heads, width = query.shape[-2:]
    kv_heads, page_size = key_pages.shape[0], key_pages.shape[2]
    return (triton_runs() and heads % kv_heads == 0 and heads // kv_heads <= QUERIES
            and width & (width - 1) == 0 and 16 <= width <= 256 and page_size & (page_size - 1) == 0
            and page_size <= BLOCK and query.dtype == key_pages.dtype)


def attend_paged(query: jax.Array, key_pages: jax.Array, value_pages: jax.Array, table: jax.Array,
                 lengths: jax.Array) -> jax.Array:
    """Attention of one query a row `[rows, heads, D]` over the first `lengths`
    `[rows]` keys of that row's pages, `table` `[rows, per_row]` naming them in
    the `[kv_heads, pages, page_size, D]` pools; a query head reads the key head
    it shares with `heads // kv_heads` others. `[rows, heads, D]` out."""
    from jax.experimental.pallas import triton as plgpu

    rows, heads, width = query.shape
    kv_heads, count, page_size, _ = key_pages.shape
    group, pages = heads // kv_heads, BLOCK // page_size
    grouped = jnp.pad(query.reshape(rows, kv_heads, group, width),
                      ((0, 0), (0, 0), (0, QUERIES - group), (0, 0)))
    table = jnp.pad(table, ((0, 0), (0, -table.shape[1] % pages)))
    pooled = pl.BlockSpec((None, count, page_size, width), lambda row, head: (head, 0, 0, 0))
    queried = pl.BlockSpec((None, None, QUERIES, width), lambda row, head: (row, head, 0, 0))
    out = pl.pallas_call(
        lambda *refs: _paged_kernel(*refs, pages=pages, scale=1 / math.sqrt(width)),
        grid=(rows, kv_heads),
        in_specs=[pl.BlockSpec((None, 1), lambda row, head: (row, 0)),
                  pl.BlockSpec((None, table.shape[1]), lambda row, head: (row, 0)),
                  queried, pooled, pooled],
        out_specs=queried,
        out_shape=jax.ShapeDtypeStruct(grouped.shape, query.dtype),
        compiler_params=plgpu.CompilerParams(num_warps=WARPS, num_stages=2),
    )(jnp.asarray(lengths, jnp.int32)[:, None], table, grouped, key_pages, value_pages)
    return out[:, :, :group].reshape(rows, heads, width)
