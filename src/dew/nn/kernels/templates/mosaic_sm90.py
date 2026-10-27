"""The wgmma instruction family's issue and layout facts (Hopper, sm90; `generation.family` 'wgmma') for the
templates' Mosaic mainloops, which take every other primitive from the shared toolkit (`mosaic`): a
warpgroup MMA reads its right operand from shared memory and its left from shared memory or registers,
accumulates in a warpgroup's registers in the WGMMA layout, and runs asynchronously until waited for."""


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
    takes_tf32 = True
    rows = 64
    consumers = 2
    role_rows = 64
    widest_product = 256
    most_shared = 227 * 1024  # sm90's (and sm100's) opt-in block

    @property
    def layout(self):
        return plgpu.Layout.WGMMA

    @property
    def row(self):
        return plgpu.Layout.WGMMA.reduce(1)

    @property
    def column(self):
        return plgpu.Layout.WGMMA.reduce(0)

    column_scratch = 4 * 32 * 2
    """A WGMMA tile's columns sum across its four warps, each lane's two entries a register."""

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

    def products(self, acc, lhs, rhs, interpret: bool):
        if interpret:
            return acc + jnp.dot(lhs[...], rhs[...], preferred_element_type=jnp.float32)
        plgpu.wgmma(acc, lhs, rhs)
        return acc

    def register_products(self, acc, parts, rhs, interpret: bool):
        for formed in parts:
            part = mosaic.unscaled_part(formed)
            if interpret:
                acc = acc + jnp.dot(part, rhs[...], preferred_element_type=jnp.float32)
            else:
                plgpu.wgmma(acc, part, rhs)
        return acc

    def summed(self, shape: tuple[int, int], issue, interpret: bool) -> jax.Array:
        """`Family.summed` into a scoped warpgroup accumulator: reading it (`acc_ref[...]`) waits on the
        products `issue` issued into it, so the value is theirs once they are done."""
        if interpret:
            return issue(self.zeros(shape))

        def run(acc_ref):
            issue(acc_ref)
            return acc_ref[...]

        return pl.run_scoped(run, plgpu.WGMMAAccumulatorRef(shape, jnp.float32))

    def staged_products(self, acc, parts, rhs, buffers, interpret: bool, *, transposed: bool = False):
        mosaic.unscaled(parts)
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

    def rescaled_products(self, acc, scale, parts, rhs, interpret: bool):
        if interpret:
            return self.register_products(scale(acc), parts, rhs, interpret)
        # The accumulator's value read once its products are done, scaled, stored back.
        plgpu.wgmma_wait(0)
        acc[...] = scale(acc[...])
        return self.register_products(acc, parts, rhs, interpret)

    def settled(self, pending: int = 0) -> None:
        plgpu.wgmma_wait(pending)

    def accumulated[C](self, segments, start: C, shapes: tuple, interpret: bool, initial: tuple | None = None
                       ) -> tuple[C, tuple[jax.Array, ...]]:
        # Warpgroup accumulators scoped across the walk, read once its MMAs are done; interpreted, arrays
        # the walk carries.
        initial = (None,) * len(shapes) if initial is None else initial
        if interpret:
            both = (start, tuple(self.zeros(shape) if value is None else value
                                 for shape, value in zip(shapes, initial, strict=True)))
            for step, lo, hi in segments:
                both = lax.fori_loop(lo, hi, lambda index, both, step=step: step(index, *both), both)
            return both

        def run(*accumulators):
            for acc, value in zip(accumulators, initial, strict=True):
                if value is not None:
                    acc[...] = value
            carry = start
            for step, lo, hi in segments:
                def stepped(index, carry, step=step):
                    return step(index, carry, accumulators)[0]

                carry = lax.fori_loop(lo, hi, stepped, carry)
            plgpu.wgmma_wait(0)
            return carry, tuple(acc[...] for acc in accumulators)

        return pl.run_scoped(run, *(plgpu.WGMMAAccumulatorRef(shape, jnp.float32) for shape in shapes))


WGMMA = Wgmma()
"""The wgmma family's bundle."""


