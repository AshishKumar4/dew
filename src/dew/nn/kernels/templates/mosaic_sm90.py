"""The wgmma instruction family's issue and layout facts (Hopper, sm90; `generation.family` 'wgmma') for the
templates' Mosaic mainloops, which take every other primitive from the shared toolkit (`mosaic`): a
warpgroup MMA reads its right operand from shared memory and its left from shared memory or registers,
accumulates in a warpgroup's registers in the WGMMA layout, and runs asynchronously until waited for."""


import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import mosaic, resources


class WgmmaIssuer(mosaic.Issuer):
    """wgmma's issuer and home: a warpgroup's 64 rows, products from shared memory or registers into a
    warpgroup accumulator in the WGMMA layout, asynchronous until waited for, one step's in flight
    (CUTLASS's sm90 `K_PIPE_MMAS`). Mosaic GPU's interpreter (jax 0.11.2) takes no register operand to a
    warpgroup MMA: interpreted, such a product is the dot it would compute."""

    name = "wgmma"
    takes_tf32 = True
    takes_fp8 = True
    rows = 64
    role_rows = 64
    widest_product = 256
    operands_in_registers = False
    products_overlap = True
    in_flight = 1

    @property
    def lane_columns(self) -> int:
        # A thread holds two adjacent columns of each 8 (the WGMMA layout's lane % 4 times two).
        return 6

    def held(self, ref):
        # An operand every step reads stays in shared memory, which its products read.
        return ref

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
    held_registers = 4
    scalars_whole = False
    """A consumer thread's own state beside its tiles: its lane's coordinates in its fragment and the walk's
    counters where they diverge. Descriptors, barriers and TMA coordinates are warp-uniform and live in the
    uniform registers (ET-BATCH1's dK/dV SASS: HGMMA gdesc[UR16], SYNCS.PHASECHK [UR17], TMA [UR48], the
    descriptor's UIADD3.64): the tensormap pair, the barrier and the warpgroup's id (3117's census of the
    sm90 recurrence, GEMM and attention launches). A store's staging and a hook's values are the
    launches' own phases, not this; its addresses into the tiles it reads and writes itself are its
    `scalar_registers`'."""

    def scalar_registers(self, addressed=()) -> int:
        # The core words and a thread's own shared-memory addresses (`resources.addresses`), which the
        # recurrence's states launches and the exact normed kernel gradient spilled without (28 to 392 bytes
        # at the four core words alone).
        return self.held_registers + resources.addresses(addressed)

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

    def product(self, lhs, rhs, shape: tuple[int, int], then=None, *, interpret: bool = False) -> jax.Array:
        del interpret  # wgmma lowers interpreted

        def scores(acc_ref):
            plgpu.wgmma(acc_ref, lhs, rhs)
            mosaic.committed()
            if then is not None:
                then()
            return acc_ref[...]

        return pl.run_scoped(scores, plgpu.WGMMAAccumulatorRef(shape, jnp.float32))

    def registers_product(self, lhs: jax.Array, rhs, interpret: bool) -> jax.Array:
        if interpret:
            return jnp.dot(lhs, rhs[...], preferred_element_type=jnp.float32)
        return self.product(lhs, rhs, (lhs.shape[0], rhs.shape[1]))

    def products(self, acc, lhs, rhs, interpret: bool, *, accumulate: bool = True):
        mosaic.committed()
        if interpret:
            fresh = jnp.dot(lhs[...], rhs[...], preferred_element_type=jnp.float32)
            return acc + fresh if accumulate else fresh
        if not accumulate:
            # A fresh product into a walk's accumulator: zeroed in its registers, then issued, no wait.
            acc[...] = jnp.zeros(acc.shape, jnp.float32)
        plgpu.wgmma(acc, lhs, rhs)
        return acc

    def read(self, acc, *, first: int = 0, columns: int | None = None, pending: int = 0) -> jax.Array:
        # A scoped warpgroup accumulator read once at most `pending` of the warpgroup's product groups are in
        # flight (`wgmma_accumulator_load`'s wait), the later ones running on; interpreted, its value.
        value = acc if isinstance(acc, jax.Array) else plgpu.wgmma_accumulator_load(acc, wait_n=pending)
        return super().read(value, first=first, columns=columns)

    def register_products(self, acc, parts, rhs, interpret: bool):
        for formed in parts:
            part = mosaic.unscaled_part(formed)
            mosaic.committed()
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

    def added_into(self, acc, value: jax.Array, interpret: bool):
        """`Family.added_into` for a warpgroup accumulator ref: its value read once its products are done,
        `value` added, stored back (interpreted, a register value's sum, `summed`'s)."""
        if interpret:
            return acc + self.as_tile(value)
        plgpu.wgmma_wait(0)
        acc[...] = acc[...] + value
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

    def accumulated[C](self, segments, start: C, shapes: tuple, interpret: bool, initial: tuple | None = None,
                       finish=None) -> tuple[C, tuple[jax.Array, ...]]:
        # Warpgroup accumulators scoped across the walk, read once its MMAs are done; interpreted, arrays
        # the walk carries.
        if interpret:
            return super().accumulated(segments, start, shapes, interpret, initial, finish)
        initial = (None,) * len(shapes) if initial is None else initial

        def run(*accumulators):
            for acc, value in zip(accumulators, initial, strict=True):
                if value is not None:
                    acc[...] = value
            carry = start
            for loop in segments:
                def stepped(index, carry, step=loop.step):
                    return step(index, carry, accumulators)[0]

                carry = lax.fori_loop(loop.lo, loop.hi, stepped, carry)
                if loop.then is not None:
                    loop.then()
            if finish is not None:
                carry = finish(carry, accumulators)[0]
            plgpu.wgmma_wait(0)
            return carry, tuple(acc[...] for acc in accumulators)

        return pl.run_scoped(run, *(plgpu.WGMMAAccumulatorRef(shape, jnp.float32) for shape in shapes))


class Wgmma(WgmmaIssuer, mosaic.RingFamily):
    """wgmma's bundle: its issuer and home (`WgmmaIssuer`) on TMA rings fed by a producer warpgroup
    (`mosaic.RingFamily`), multicast across a pair of multiprocessors where a cluster shares a ring's
    blocks."""

    consumers = 2
    cluster = 2  # TMA multicast to a pair, CUTLASS's grouped GEMM's and cuBLASLt's on an H100


WGMMA = Wgmma()
"""The wgmma family's bundle."""


class WgmmaFed(mosaic.ConsumerFed, Wgmma):
    """wgmma's bundle without a producer: its consumers feed their own rings (`mosaic.ConsumerFed`), each
    at its program's entry registers, the layout a tile whose consumers need more than a producer leaves
    them takes."""

    name = "wgmma.fed"


FED = WgmmaFed()
"""wgmma's bundle without a producer."""


