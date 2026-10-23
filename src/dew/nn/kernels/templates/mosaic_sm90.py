"""The wgmma instruction family's issue and layout facts (Hopper, sm90; `generation.family` 'wgmma') for the
templates' Mosaic mainloops, which take every other primitive from the shared toolkit (`mosaic`): a
warpgroup MMA reads its right operand from shared memory and its left from shared memory or registers,
accumulates in a warpgroup's registers in the WGMMA layout, and runs asynchronously until waited for."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import mosaic

FAMILY = mosaic.Family("wgmma", plgpu.Layout.WGMMA, rows=64)
"""wgmma's facts: its accumulator's layout and a warpgroup's 64 rows."""
ROW = FAMILY.layout.reduce(1)
"""A per-row vector's layout in the accumulator's."""


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
