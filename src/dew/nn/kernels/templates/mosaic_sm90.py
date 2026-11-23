"""The wgmma instruction family's issue and layout facts (Hopper, sm90; `generation.family` 'wgmma') for the
templates' Mosaic mainloops, which take every other primitive from the shared toolkit (`mosaic`): a
warpgroup MMA reads its right operand from shared memory and its left from shared memory or registers,
accumulates in a warpgroup's registers in the WGMMA layout, and runs asynchronously until waited for."""

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax._src.state import types as state_types
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import launches
from dew.nn.kernels.templates import fed, mosaic, resources, ring_family


class WgmmaLayout:
    """A home whose 64-row tiles sit in registers in the WGMMA layout (`plgpu.Layout.WGMMA`), a thread two
    adjacent columns of each 8 in a warp's 16 rows: its tiles', rows' and columns' layouts and what a thread
    holds of them. wgmma's accumulators, and tcgen05's 64-row ones read out of tensor memory
    (`mosaic_sm100.Tcgen05Chunks`), whose half-lane layout loads into it."""

    @property
    def lane_columns(self) -> int:
        # A thread holds two adjacent columns of each 8 (the WGMMA layout's lane % 4 times two).
        return 6

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
class WgmmaIssuer(WgmmaLayout, mosaic.Issuer):
    """wgmma's issuer and home: a warpgroup's 64 rows, products from shared memory or registers into a
    warpgroup accumulator in the WGMMA layout, asynchronous until waited for, one step's in flight
    (CUTLASS's sm90 `K_PIPE_MMAS`). Mosaic GPU's interpreter (jax 0.11.2) takes no register operand to a
    warpgroup MMA: interpreted, such a product is the dot it would compute."""

    name = "wgmma"
    takes_tf32 = True
    fp8_formats = (jnp.dtype(jnp.float8_e4m3fn), jnp.dtype(jnp.float8_e5m2))
    rows = 64
    role_rows = 64
    widest_product = 256
    operands_in_registers = False
    products_overlap = True
    in_flight = 1
    retires_register_products = True

    assembled = 26
    """The registers past an MMA's fp32 accumulator a thread's register target holds at least, whatever its
    program's raises: the assembler's own floor, measured, not a model's choice. ptxas 12.9 (sm_90a) refused
    a kernel of one m64nNk16 MMA (bf16 operands from shared memory into fp32) every target under N / 2 + 26
    at N of 16, 32, 64 and 128, its one warpgroup's or under producer and consumer roles raised past it
    (C7602), and assembled it spill-free there; the values' backward's at 16 wide blocks, 32 registers at 8
    programs, at 34 (jsf-vback-root, M-JS-63). The CPU gate assembles it at the floor
    (tests/test_kernel_compile_registers.py), so a ptxas that moves it fails there."""

    def entry_floor(self, tiles) -> int:
        widest = max((columns for _, columns, dtype in tiles if jnp.dtype(dtype) == jnp.float32), default=0)
        accumulator = min(widest, self.widest_product or widest) * self.rows // resources.THREADS
        return accumulator + self.assembled if widest else 0

    def product(self, lhs, rhs, shape: tuple[int, int], then=None, *, pending: int = 0) -> jax.Array:

        def scores(acc_ref):
            plgpu.wgmma(acc_ref, lhs, rhs)
            mosaic.committed()
            if then is not None:
                then()
            return acc_ref[...] if pending == 0 else self.read(acc_ref, pending=pending)

        return pl.run_scoped(scores, plgpu.WGMMAAccumulatorRef(shape, jnp.float32))

    def summed_together(self, shapes: tuple[tuple[int, int], ...], issues: tuple, *, apart: bool = False,
                        then: tuple[Callable | None, ...] | None = None) -> tuple[Callable, ...]:
        # Every group issued into its own scoped accumulator, then each read as its own groups complete,
        # the later ones running on (`read`'s `pending`, counted by the groups every run commits), `then[i]`
        # of it run there, interpreted alike (the interpreter's own MMA, as `product`'s); `apart`, each summed
        # in turn.
        if apart:
            return super().summed_together(shapes, issues, apart=apart, then=then)
        after = (None,) * len(shapes) if then is None else then

        def scoped(*accumulators):
            ends = []
            for issue, acc in zip(issues, accumulators, strict=True):
                issue(acc)
                ends.append(mosaic.sure_commits())
            values = []
            for acc, end, each in zip(accumulators, ends, after, strict=True):
                # In flight past the group read: the groups after it every run commits, which complete
                # after it, whichever predicated groups a run commits too (`mosaic.sure_commits`).
                value = self.read(acc, pending=mosaic.sure_commits() - end)
                values.append(value if each is None else each(value))
            return tuple(values)

        values = pl.run_scoped(scoped, *(plgpu.WGMMAAccumulatorRef(shape, jnp.float32) for shape in shapes))
        return tuple(functools.partial(lambda value: value, value) for value in values)

    def registers_product(self, lhs: jax.Array, rhs) -> jax.Array:
        interpret = launches.interpreting()  # the launch's (`launches.interpreting`)
        if interpret:
            return jnp.dot(lhs, rhs[...], preferred_element_type=jnp.float32)
        return self.product(lhs, rhs, (lhs.shape[0], rhs.shape[1]))

    def products(self, acc, lhs, rhs, *, accumulate: bool = True):
        interpret = launches.interpreting()  # the launch's (`launches.interpreting`)
        mosaic.committed()
        if interpret and isinstance(acc, jax.Array):
            fresh = jnp.dot(lhs[...], rhs[...], preferred_element_type=jnp.float32)
            return acc + fresh if accumulate else fresh
        # A scoped accumulator, interpreted too: the interpreter's own MMA, as a fresh product's (`product`).
        if not accumulate:
            # A fresh product into a walk's accumulator: zeroed in its registers, then issued, no wait.
            acc[...] = jnp.zeros(acc.shape, jnp.float32)
        plgpu.wgmma(acc, lhs, rhs)
        return acc

    def issued_when(self, issuing, acc, issue):
        # The warpgroup's accumulator its MMAs write in place: issued under a branch on `issuing`, its groups
        # counted either way (`Issuer.settled` waits until at most so many are in flight, which fewer meet); a
        # value (a running sum's, an interpreted walk's) by the issuer's own branch.
        if not (isinstance(acc, state_types.TransformedRef)
                or isinstance(jax.typeof(acc), state_types.AbstractRef)):
            return super().issued_when(issuing, acc, issue)

        @pl.when(issuing)
        def _():
            mosaic.predicated(issue)  # into `acc` in place; its returned ref no branch's value

        return acc
    def read(self, acc, *, first: int = 0, columns: int | None = None, pending: int = 0) -> jax.Array:
        # A scoped warpgroup accumulator read once at most `pending` of the warpgroup's product groups are in
        # flight (`wgmma_accumulator_load`'s wait), the later ones running on; interpreted, its value.
        value = acc if isinstance(acc, jax.Array) else plgpu.wgmma_accumulator_load(acc, wait_n=pending)
        return super().read(value, first=first, columns=columns)

    def register_products(self, acc, parts, rhs):
        if not launches.interpreting():  # the launch's (`launches.interpreting`)
            for formed in parts:
                part = mosaic.unscaled_part(formed)
                mosaic.committed()
                plgpu.wgmma(acc, part, rhs)
                # Operand A is read from its registers until its group retires, and the next part or the
                # walk's next step writes them (`retires_register_products`): retired now, the groups
                # before it with it, as they retire in order.
                plgpu.wgmma_wait(0)
            return acc
        for formed in parts:
            part = mosaic.unscaled_part(formed)
            mosaic.committed()
            product = jnp.dot(part, rhs[...], preferred_element_type=jnp.float32)
            if isinstance(acc, jax.Array):
                acc = acc + product
            else:
                # A scoped accumulator interpreted: its value read and written back (`summed_together`'s).
                acc[...] = acc[...] + product
        return acc

    def summed(self, shape: tuple[int, int], issue) -> jax.Array:
        """`Family.summed` into a scoped warpgroup accumulator: reading it (`acc_ref[...]`) waits on the
        products `issue` issued into it, so the value is theirs once they are done."""
        interpret = launches.interpreting()  # the launch's (`launches.interpreting`)
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
            return self.register_products(scale(acc), parts, rhs)
        # The accumulator's value read once its products are done, scaled, stored back.
        plgpu.wgmma_wait(0)
        acc[...] = scale(acc[...])
        return self.register_products(acc, parts, rhs)

    def settled(self, pending: int = 0) -> None:
        plgpu.wgmma_wait(pending)

    def accumulated[C](self, segments, start: C, shapes: tuple, interpret: bool, initial: tuple | None = None
                       ) -> tuple[C, tuple[jax.Array, ...]]:
        # Warpgroup accumulators scoped across the walk, read once its MMAs are done; interpreted, arrays
        # the walk carries.
        if interpret:
            return super().accumulated(segments, start, shapes, interpret, initial)
        initial = (None,) * len(shapes) if initial is None else initial

        def run(*accumulators):
            for acc, value in zip(accumulators, initial, strict=True):
                if value is not None:
                    acc[...] = value
            carry = mosaic.looped(segments, start, accumulators)
            plgpu.wgmma_wait(0)
            return carry, tuple(acc[...] for acc in accumulators)

        return pl.run_scoped(run, *(plgpu.WGMMAAccumulatorRef(shape, jnp.float32) for shape in shapes))


class Wgmma(WgmmaIssuer, ring_family.RingFamily):
    """wgmma's bundle: its issuer and home (`WgmmaIssuer`) on TMA rings fed by a producer warpgroup
    (`ring_family.RingFamily`), multicast across a pair of multiprocessors where a cluster shares a ring's
    blocks."""

    # Three consumers beside the producer, 504 of a lane's 512 registers raised (160 each, the producer's 24):
    # FlashAttention 3's d64 forward's 192-row tile (hopper's tile_size_fwd_sm90, row 1299), the most a
    # program takes.
    consumers = 3
    cluster = 2  # TMA multicast to a pair, CUTLASS's grouped GEMM's and cuBLASLt's on an H100


WGMMA = Wgmma()
"""The wgmma family's bundle."""


class WgmmaFed(fed.ConsumerFed, Wgmma):
    """wgmma's bundle without a producer: its consumers feed their own rings (`fed.ConsumerFed`), each
    at its program's entry registers, the layout a tile whose consumers need more than a producer leaves
    them takes."""

    name = "wgmma.fed"


FED = WgmmaFed()
"""wgmma's bundle without a producer."""


