"""The templates' Pallas Mosaic GPU toolkit (design §6), every instruction family's: the launch,
warpgroup roles, barriers, shared-memory staging, operand streams and stores every Mosaic mainloop builds on.
The math is each template's algebra, written once; what an instruction family adds is its issue and its
layout facts alone (`Family`: mosaic_sm90's wgmma; mma.sync's and tcgen05's to come). Each template keeps
its own mainloop, product dependencies and launch DAG; only these primitives are shared.

A cooperative launch (`cooperative`) runs a producer warpgroup issuing the loads beside consumer warpgroups
computing, persistent over the multiprocessors, a pair of them a cluster where it shares one operand's blocks
(multicast, `multicast_axes`). The consumers report a tile drained (`drained`, `drain`) so a persistent
producer never loads a next tile's stage over one an MMA still reads, and meet (`meet`) where one consumer's
writes to shared memory are the other's operand. Each consumer's accumulator lives for one output tile
(`finished`), and a tile's rows reach global memory through `store_rows`, whose stores have static extents
however many rows a tile holds.

A mainloop that drives its own loads (the GEMM's kernel gradient, its stages streaming across tiles)
streams each operand through a `Ring` whose counter runs across tiles, and takes each warpgroup's role
(`roles`).
"""

import dataclasses
import math

import jax
from jax import lax, numpy as jnp
from jax._src.state import types as state_types
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

Ref = state_types.AbstractRef | state_types.TransformedRef
"""A scoped allocation a primitive takes: shared memory, transformed (tiled and swizzled) or not, or
barriers."""


def compiler_params(*, approx_math: bool = False) -> plgpu.CompilerParams:
    """A mainloop's compiler parameters: Mosaic GPU's warpgroup lowering, whose arrays are a warpgroup's,
    and with `approx_math` the hardware's approximate exponential and reciprocal (ex2.approx, as Triton's
    exp2 and FlashAttention's are)."""
    return plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Warpgroup, approx_math=approx_math)


def cooperative(body, out_type, *, consumers: int, cluster: int, multiprocessors: int, interpret):
    """A cooperative launch of `body`: `consumers` consumer warpgroups and a producer a block (warpgroup
    `consumers`, `drain`), persistent over `multiprocessors` ('cluster_grid'), a pair of them a cluster where
    `cluster` is 2 ('cluster')."""
    if multiprocessors < cluster:
        raise ValueError(f"a cluster of {cluster} pairs multiprocessors, and the launch has "
                         f"{multiprocessors} (an interpreter's one block does not model a cluster)")
    return plgpu.kernel(
        body,
        out_type=out_type,
        grid=(multiprocessors // cluster,),
        grid_names=("cluster_grid",),
        cluster=(cluster,),
        cluster_names=("cluster",),
        num_threads=consumers + 1,
        thread_name="wg",
        compiler_params=compiler_params(),
        interpret=interpret,
    )


def multicast_axes(cluster: int) -> tuple[str, ...]:
    """The axes a block a pair shares is loaded over, multicast: the cluster's, for a pair."""
    return ("cluster",) if cluster == 2 else ()


def stage_spec(axes: tuple, extent: dict, stepped: str, in_flight: int) -> plgpu.BlockSpec:
    """An operand's block in a pipeline's shared-memory stage: its tile at `extent` per launch axis, advancing
    along `stepped` (the pipeline's axis) and fixed along the others, held `in_flight` steps longer for the
    MMAs in flight."""
    return plgpu.BlockSpec(tuple(extent[axis] for axis in axes),
                           lambda step: tuple(step if axis == stepped else 0 for axis in axes),
                           memory_space=plgpu.SMEM, delay_release=in_flight)


def ignore(*_, **__) -> None:
    """A pipeline body or compute context that does nothing: the allocations a pipeline takes do not depend
    on them (`hopper_matmul_mgpu`'s `ignore`)."""


def drained(cluster: int):
    """The barrier two consumers report a tile drained at: both multiprocessors' of a pair (each producer's
    next loads are multicast into both), a block's own otherwise."""
    if cluster == 2:
        return plgpu.ClusterBarrier(collective_axes=("cluster",), num_arrivals=2)
    return plgpu.Barrier(num_arrivals=2)


def drain(barrier, wg, producer: int = 2) -> None:
    """The producer warpgroup (`producer`) waits until the consumers have finished the tile's MMAs (each
    arrives at `barrier` after its last wait; a pair's four, `drained`) before its next tile's loads reuse
    the pipeline's stages."""
    pl.when(wg == producer)(lambda: plgpu.barrier_wait(barrier))


def meet(halves) -> None:
    """The two consumer warpgroups wait for each other at `halves`, two barriers of two arrivals, in turn:
    neither arrives at a barrier's next phase before the other has seen its last, however the meets fall
    (the interpreter's invariant, and the hardware's parity waits)."""
    for barrier in (halves.at[0], halves.at[1]):
        plgpu.barrier_arrive(barrier)
        plgpu.barrier_wait(barrier)


def staging(name: str, shaped) -> dict:
    """Each consumer warpgroup's staging buffer for each operand a tile reads (`shaped`: its tile's shape and
    its operand) and a barrier for each, as `{name}_smems` and `{name}_barriers`."""
    shaped = list(shaped)
    return {f'{name}_smems': [plgpu.SMEM((2, *shape), gmem.dtype) for shape, gmem in shaped],
            f'{name}_barriers': [plgpu.Barrier(num_barriers=2) for _ in shaped]}


def streamed(rings: tuple, views, lo, hi) -> None:
    """A producer's loads of steps `lo` to `hi` into `rings`, each ring's step counted from `lo` and loaded
    from its view of `views(step)`, then each ring drained."""
    @pl.loop(lo, hi)
    def _(index):
        for ring, view in zip(rings, views(index), strict=True):
            ring.load(index - lo, view)

    for ring in rings:
        ring.drain(hi - lo)


def passed(rings: tuple, lo, hi, *, first: jax.Array | int = 0, between=None) -> None:
    """A consumer's steps `lo` to `hi` of `rings` it reads none of (another consumer's), each ring's load
    awaited and its slot released as a reading consumer's are (counted from `first`): a wait takes its
    barrier's next phase, so every consumer waits on every load; `between()` after each step."""
    @pl.loop(lo, hi)
    def _(index):
        for ring in rings:
            ring.acquire(index - first)
            ring.release(index - first)
        if between is not None:
            between()


def loaded(copies) -> None:
    """Each `(view, ref, barrier)` copied whole into shared memory, every copy issued before any is
    awaited."""
    for view, ref, barrier in copies:
        plgpu.copy_gmem_to_smem(view, ref, barrier)
    for _, _, barrier in copies:
        plgpu.barrier_wait(barrier)


def stored(parts, views, *, smem) -> None:
    """Each of `parts` written whole to its view of `views` through the shared buffer `smem`, in turn, the
    buffer reused once its copy is done."""
    for part, view in zip(parts, views, strict=True):
        smem[...] = part.astype(smem.dtype)
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(smem, view)
        plgpu.wait_smem_to_gmem(0)


def load_fixed(views: list, refs: list, barriers: list, wg) -> None:
    """Each operand a tile reads whole (`views`) into this warpgroup's `refs` (`staging`), awaited."""
    for view, ref, barrier in zip(views, refs, barriers, strict=True):
        plgpu.copy_gmem_to_smem(view, ref, barrier.at[wg])
        plgpu.barrier_wait(barrier.at[wg])


def store_rows(o_smem, o_gmem, row_start, first, stored, column, most: int) -> None:
    """Rows `first` to `first + stored` of `o_smem` to `o_gmem` at row `row_start + first` and `column`, in a
    power-of-two ladder of TMA stores (a descriptor's extents are static), the largest `most` rows'."""
    start, size = first, most
    while size:
        rung = 1 << int(math.log2(size))
        size //= 2

        @pl.when(stored & rung != 0)
        def _(start=start, rung=rung):
            plgpu.copy_smem_to_gmem(o_smem.at[pl.ds(start, rung)],
                                    o_gmem.at[pl.ds(row_start + start, rung), pl.ds(column, o_smem.shape[1])])

        start = start + (stored & rung)


@dataclasses.dataclass(frozen=True)
class Ring:
    """A ring of `stages` shared-memory slots one operand streams through: a producer loads step `counter`
    into slot `counter % stages` (`load`), a consumer waits for it (`acquire`) and gives it back
    (`release`). The counter runs across a persistent kernel's tiles, so a tile's first loads reuse the
    slots the last tile left, and each tile may take its own number of steps (a causal tile's trip range):
    a slot is free for its first fill and, after, once every consumer released it; the producer drains the
    ring after its last load (`drain`). Allocated with `ring`, its slot count the allocation's own."""
    smem: Ref
    full: state_types.AbstractRef
    consumed: state_types.AbstractRef

    @property
    def stages(self) -> int:
        """The ring's slots: its allocation's leading dimension."""
        return int(self.smem.shape[0])

    def slot(self, counter):
        """The slot step `counter` uses."""
        return lax.rem(counter, jnp.asarray(self.stages, counter.dtype))

    def load(self, counter, view) -> None:
        """The producer's load of step `counter` from `view`, once its slot is free."""
        slot = self.slot(counter)
        pl.when(counter >= self.stages)(lambda: plgpu.barrier_wait(self.consumed.at[slot]))
        plgpu.copy_gmem_to_smem(view, self.smem.at[slot], self.full.at[slot])

    def acquire(self, counter):
        """A consumer's wait for step `counter`'s load: its slot. Each load is waited for once (a wait takes
        the barrier's next phase)."""
        slot = self.slot(counter)
        plgpu.barrier_wait(self.full.at[slot])
        return self.smem.at[slot]

    def at(self, counter):
        """Step `counter`'s slot, its load waited for already (`acquire`, issued under an earlier step's
        products)."""
        return self.smem.at[self.slot(counter)]

    def release(self, counter) -> None:
        """A consumer's release of step `counter`'s slot, whose reads (and MMAs) are done."""
        plgpu.barrier_arrive(self.consumed.at[self.slot(counter)])

    def drain(self, end) -> None:
        """The producer's wait, after its last load (step `end - 1`), for the consumers' release of each slot
        it filled last, so no slot is freed (or reallocated) while a consumer still reads it, and every
        release is seen."""
        for back in range(self.stages):
            step = end - 1 - back
            pl.when(step >= 0)(lambda step=step: plgpu.barrier_wait(self.consumed.at[self.slot(step)]))


def ring(block: tuple, dtype, stages: int, *, consumers: int = 1) -> dict:
    """A `Ring`'s allocations for `pl.run_scoped`: `stages` slots of `block` in `dtype`, a load barrier
    each, and a release barrier each that `consumers` arrive at. `Ring(**allocated)` once scoped."""
    return {'smem': plgpu.SMEM((stages, *block), dtype),
            'full': plgpu.Barrier(num_barriers=stages),
            'consumed': plgpu.Barrier(num_arrivals=consumers, num_barriers=stages)}


@dataclasses.dataclass(frozen=True)
class Family:
    """An instruction family's facts a Mosaic mainloop issues by (`generation.family` names it): its
    accumulator's register `layout` and the rows one compute warpgroup's MMA covers (`rows`). Its issue
    functions are its module's (mosaic_sm90's wgmma); its operand tiles' shared-memory tiling and swizzle are
    the warpgroup lowering's inference (`compiler_params`)."""

    name: str
    layout: plgpu.Layout
    rows: int


PRODUCER_REGISTERS = 40
"""The registers a thread of a cooperative block's producer warpgroup holds: it only issues loads."""


def consumer_registers(consumers: int) -> int:
    """The registers a thread of each of `consumers` consumer warpgroups holds: the multiprocessor's 64K less
    the producer's, shared among them in setmaxnreg's steps of 8 and at most its 256 (two consumers' 232,
    JAX's warp-specialized pipeline's `_compute_registers(40, 2)`)."""
    share = (64 * 1024 // 128 - PRODUCER_REGISTERS) // consumers
    return min(256, share // 8 * 8)


CONSUMER_REGISTERS = consumer_registers(2)
"""Two consumer warpgroups' registers a thread (`consumer_registers`)."""


def roles(wg, consumers: int, consumer, producer) -> None:
    """Each warpgroup's role in a cooperative block: the `consumers` first run `consumer(wg)` with their
    register budget raised to their share (`consumer_registers`), the last runs `producer()` with its
    lowered to `PRODUCER_REGISTERS`."""
    @pl.when(wg < consumers)
    def _():
        plgpu.set_max_registers(consumer_registers(consumers), action="increase")
        consumer(wg)

    @pl.when(wg == consumers)
    def _():
        plgpu.set_max_registers(PRODUCER_REGISTERS, action="decrease")
        producer()
