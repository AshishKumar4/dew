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

FAMILY = mosaic.Family(
    "wgmma", plgpu.Layout.WGMMA, plgpu.Layout.WGMMA.reduce(1), plgpu.Layout.WGMMA.reduce(0), rows=64,
    transforms=(plgpu.TilingTransform((8, 64)), plgpu.SwizzleTransform(128)))
"""wgmma's facts: a warpgroup's 64 rows, 16-bit tiles of 64-entry (128-byte) swizzled rows."""


def product(lhs, rhs, shape: tuple[int, int], then=None) -> jax.Array:
    """`lhs @ rhs` of shared-memory tiles (or `lhs` in registers), fp32 `shape`, with `then` issued while the
    warpgroup MMA runs."""
    def scores(acc_ref):
        plgpu.wgmma(acc_ref, lhs, rhs)
        if then is not None:
            then()
        return acc_ref[...]

    return pl.run_scoped(scores, plgpu.ACC(shape, jnp.float32))


def accumulate(acc: jax.Array, lhs: jax.Array, rhs, interpret: bool) -> jax.Array:
    """`acc + lhs @ rhs`, the warpgroup MMA accumulating into `acc` in place, `lhs` in registers (16-bit) and
    `rhs` a shared-memory tile. Mosaic GPU's interpreter (jax 0.11.2) takes no register operand to a
    warpgroup MMA: interpreted, the product is the dot it would compute."""
    if interpret:
        return acc + jnp.dot(lhs, rhs[...], preferred_element_type=jnp.float32)

    def accumulated(acc_ref):
        plgpu.wgmma(acc_ref, lhs, rhs)

    return pl.run_state(accumulated)(plgpu.ACC.init(acc))


def zeros(shape: tuple[int, ...]) -> jax.Array:
    """fp32 zeros in the accumulator's layout (`shape` two-dimensional) or a row's (one-dimensional)."""
    return plgpu.layout_cast(jnp.zeros(shape, jnp.float32), FAMILY.layout if len(shape) == 2 else FAMILY.row)


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
