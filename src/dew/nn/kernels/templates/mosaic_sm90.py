"""The wgmma instruction family's issue and layout facts (Hopper, sm90; `generation.family` 'wgmma') for the
templates' Mosaic mainloops, which take every other primitive from the shared toolkit (`mosaic`): a
warpgroup MMA reads its right operand from shared memory and its left from shared memory or registers,
accumulates in a warpgroup's registers in the WGMMA layout, and runs asynchronously until waited for."""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import mosaic

FAMILY = mosaic.Family("wgmma", plgpu.Layout.WGMMA, rows=64)
"""wgmma's facts: its accumulator's layout and a warpgroup's 64 rows."""
ROW = FAMILY.layout.reduce(1)
COLUMN = FAMILY.layout.reduce(0)
"""A per-row and a per-column vector's layouts in the accumulator's."""


def product(lhs, rhs, shape: tuple[int, int], then=None) -> jax.Array:
    """`lhs @ rhs` of shared-memory tiles (or `lhs` in registers), fp32 `shape`, with `then` issued while the
    warpgroup MMA runs."""
    def scores(acc_ref):
        plgpu.wgmma(acc_ref, lhs, rhs)
        if then is not None:
            then()
        return acc_ref[...]

    return pl.run_scoped(scores, plgpu.WGMMAAccumulatorRef(shape, jnp.float32))


def registers_product(lhs: jax.Array, rhs, interpret: bool) -> jax.Array:
    """`lhs @ rhs`, fp32, `lhs` in registers (16-bit) and `rhs` a shared-memory tile. Mosaic GPU's
    interpreter (jax 0.11.2) takes no register operand to a warpgroup MMA: interpreted, the product is the
    dot it would compute."""
    if interpret:
        return jnp.dot(lhs, rhs[...], preferred_element_type=jnp.float32)
    return product(lhs, rhs, (lhs.shape[0], rhs.shape[1]))


def register_products(acc, parts, rhs, interpret: bool):
    """`acc` with `parts`' products with `rhs` (a shared-memory tile) issued into it in place, in order: a
    warpgroup accumulator, or interpreted the array its value is (`walked_products`). Each part is a 16-bit
    tile in registers, the warpgroup MMA's left operand, with the power of two its product takes back, which
    only None (no scaling, bf16's parts, `attention_algebra.cotangent_parts`) issues in place. Mosaic GPU's
    interpreter (jax 0.11.2) takes no register operand to a warpgroup MMA: interpreted, each product is the
    dot it would compute, added."""
    if any(exponent is not None for _, exponent in parts):
        raise NotImplementedError("a scaled part's product is no in-place accumulation (fp16's)")
    for part, _ in parts:
        if interpret:
            acc = acc + jnp.dot(part, rhs[...], preferred_element_type=jnp.float32)
        else:
            plgpu.wgmma(acc, part, rhs)
    return acc


def staged_products(acc, parts, rhs, buffers, interpret: bool, *, transposed: bool = False):
    """`register_products` of `parts` each staged in shared memory first, a buffer of `buffers` apiece (its
    transpose the left operand where `transposed`), so a left operand the MMA takes only from shared memory
    (a transpose) is one. The buffers are free for the next tile's parts once these MMAs are waited for."""
    if any(exponent is not None for _, exponent in parts):
        raise NotImplementedError("a scaled part's product is no in-place accumulation (fp16's)")
    for place, (part, _) in enumerate(parts):
        buffers.at[place][...] = part.astype(buffers.dtype)
    plgpu.commit_smem()
    for place in range(len(parts)):
        lhs = buffers.at[place].transpose((1, 0)) if transposed else buffers.at[place]
        if interpret:
            acc = acc + jnp.dot(lhs[...], rhs[...], preferred_element_type=jnp.float32)
        else:
            plgpu.wgmma(acc, lhs, rhs)
    return acc


def walked_products[C](segments, start: C, shapes: tuple[tuple[int, int], ...], interpret: bool
                       ) -> tuple[C, tuple[jax.Array, ...]]:
    """A walk of `segments`, each `(step, lo, hi)` its steps `lo` to `hi` in order,
    `step(index, carry, accumulators) -> (carry, accumulators)`, from `start`, with fp32 accumulators of
    `shapes` (zero) its products issue into in place (`register_products`, `staged_products`): warpgroup
    accumulators scoped across the walk, read once its MMAs are waited for; interpreted, arrays the walk
    carries. Returns the carry and the accumulators' values."""
    if interpret:
        both = (start, tuple(zeros(shape) for shape in shapes))
        for step, lo, hi in segments:
            both = lax.fori_loop(lo, hi, lambda index, both, step=step: step(index, *both), both)
        return both

    def run(*accumulators):
        carry = start
        for step, lo, hi in segments:
            carry = lax.fori_loop(lo, hi, lambda index, carry, step=step: step(index, carry, accumulators)[0],
                                  carry)
        plgpu.wgmma_wait(0)
        return carry, tuple(acc[...] for acc in accumulators)

    return pl.run_scoped(run, *(plgpu.WGMMAAccumulatorRef(shape, jnp.float32) for shape in shapes))


def zeros(shape: tuple[int, ...]) -> jax.Array:
    """fp32 zeros in the accumulator's layout (`shape` two-dimensional) or a row's (one-dimensional)."""
    return plgpu.layout_cast(jnp.zeros(shape, jnp.float32), FAMILY.layout if len(shape) == 2 else ROW)


def iota(shape: tuple[int, int], dimension: int) -> jax.Array:
    """Indices along `dimension` of a tile in the accumulator's layout."""
    return plgpu.broadcasted_iota(jnp.int32, shape, dimension, layout=FAMILY.layout)


def finished(run, shape: tuple[int, int], barrier, finish) -> None:
    """A consumer's output tile: `run` on a scoped warpgroup accumulator of `shape`, its MMAs waited for,
    the tile reported drained (`barrier`, the producer's cue to reload its stage), then `finish` of its
    value."""
    @functools.partial(pl.run_scoped, acc_ref=plgpu.WGMMAAccumulatorRef(shape))
    def _(acc_ref):
        run(acc_ref)
        plgpu.wgmma_wait(0)
        plgpu.barrier_arrive(barrier)
        finish(acc_ref[...])
