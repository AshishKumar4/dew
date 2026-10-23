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


class Wgmma(mosaic.RingFamily):
    """wgmma's bundle: a warpgroup's 64 rows, TMA rings fed by a producer warpgroup (`mosaic.RingFamily`),
    products from shared memory or registers into a warpgroup accumulator in the WGMMA layout, asynchronous
    until waited for. Mosaic GPU's interpreter (jax 0.11.2) takes no register operand to a warpgroup MMA:
    interpreted, such a product is the dot it would compute."""

    name = "wgmma"
    rows = 64
    layout = plgpu.Layout.WGMMA
    row = plgpu.Layout.WGMMA.reduce(1)
    column = plgpu.Layout.WGMMA.reduce(0)

    def product(self, lhs, rhs, shape: tuple[int, int], then=None) -> jax.Array:
        def scores(acc_ref):
            plgpu.wgmma(acc_ref, lhs, rhs)
            if then is not None:
                then()
            return acc_ref[...]

        return pl.run_scoped(scores, plgpu.WGMMAAccumulatorRef(shape, jnp.float32))

    def registers_product(self, lhs: jax.Array, rhs, interpret: bool) -> jax.Array:
        if interpret:
            return jnp.dot(lhs, rhs[...], preferred_element_type=jnp.float32)
        return self.product(lhs, rhs, (lhs.shape[0], rhs.shape[1]))

    def register_products(self, acc, parts, rhs, interpret: bool):
        _unscaled(parts)
        for part, _ in parts:
            if interpret:
                acc = acc + jnp.dot(part, rhs[...], preferred_element_type=jnp.float32)
            else:
                plgpu.wgmma(acc, part, rhs)
        return acc

    def staged_products(self, acc, parts, rhs, buffers, interpret: bool, *, transposed: bool = False):
        _unscaled(parts)
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

    def settled(self) -> None:
        plgpu.wgmma_wait(0)

    def accumulated[C](self, segments, start: C, shapes: tuple, interpret: bool
                       ) -> tuple[C, tuple[jax.Array, ...]]:
        # Warpgroup accumulators scoped across the walk, read once its MMAs are done; interpreted, arrays
        # the walk carries.
        if interpret:
            both = (start, tuple(self.zeros(shape) for shape in shapes))
            for step, lo, hi in segments:
                both = lax.fori_loop(lo, hi, lambda index, both, step=step: step(index, *both), both)
            return both

        def run(*accumulators):
            carry = start
            for step, lo, hi in segments:
                def stepped(index, carry, step=step):
                    return step(index, carry, accumulators)[0]

                carry = lax.fori_loop(lo, hi, stepped, carry)
            plgpu.wgmma_wait(0)
            return carry, tuple(acc[...] for acc in accumulators)

        return pl.run_scoped(run, *(plgpu.WGMMAAccumulatorRef(shape, jnp.float32) for shape in shapes))

    def zeros(self, shape: tuple[int, ...]) -> jax.Array:
        return self.full(shape, 0.0)

    def full(self, shape: tuple[int, ...], value: float) -> jax.Array:
        layout = self.layout if len(shape) == 2 else self.row
        return plgpu.layout_cast(jnp.full(shape, value, jnp.float32), layout)

    def iota(self, shape: tuple[int, int], dimension: int) -> jax.Array:
        return plgpu.broadcasted_iota(jnp.int32, shape, dimension, layout=self.layout)

    def rows_of(self, ref) -> jax.Array:
        return plgpu.load(ref, layout=self.row, optimized=False)

    def as_row(self, vector: jax.Array) -> jax.Array:
        return plgpu.layout_cast(vector, self.row)


def _unscaled(parts) -> None:
    """A product's parts issued in place take no power of two back (bf16's, `cotangent_parts`)."""
    if any(exponent is not None for _, exponent in parts):
        raise NotImplementedError("a scaled part's product is no in-place accumulation (fp16's)")


WGMMA = Wgmma()
"""The wgmma family's bundle."""


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
