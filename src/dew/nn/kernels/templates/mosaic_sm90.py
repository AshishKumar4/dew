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
    multicast across a pair of multiprocessors where a cluster shares a ring's blocks, products from shared
    memory or registers into a warpgroup accumulator in the WGMMA layout, asynchronous until waited for.
    Mosaic GPU's interpreter (jax 0.11.2) takes no register operand to a warpgroup MMA: interpreted, such a
    product is the dot it would compute."""

    name = "wgmma"
    takes_tf32 = True
    rows = 64
    consumers = 2
    role_rows = 64
    widest_product = 256
    most_shared = 227 * 1024  # sm90's (and sm100's) opt-in block
    cluster = 2  # TMA multicast to a pair, CUTLASS's grouped GEMM's and cuBLASLt's on an H100

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

    def column_registers(self, columns: int, dtype) -> int:
        # The column layout repeats a vector over the warpgroup's four warps and eight lanes of each, so a
        # thread holds a quarter of its columns: FS-K01-CAUSE-VECTOR, an fp32 column vector's epilogue at
        # 128 and 256 columns needs 104 and 200 against a row vector's 96 and 160.
        return -(-columns * jnp.dtype(dtype).itemsize // 16)

    def row_registers(self, rows: int, dtype) -> int:
        # Each warp's 16 rows of an MMA, a thread two of them, so a thread holds a 32nd of a consumer's rows:
        # the fast attention backward's dQ consumer carries seven fp32 row vectors, 4 registers each at 128
        # rows (DQROWPROOF64: 260 against ptxas's 255 and 512 bytes spilled; DQROWPROOF128: 184 at 64 rows).
        return -(-rows * jnp.dtype(dtype).itemsize // 128)

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


class WgmmaFed(mosaic.ConsumerFed, Wgmma):
    """wgmma's bundle without a producer: its consumers feed their own rings (`mosaic.ConsumerFed`), each
    at its program's entry registers, the layout a tile whose consumers need more than a producer leaves
    them takes."""

    name = "wgmma.fed"


FED = WgmmaFed()
"""wgmma's bundle without a producer."""


