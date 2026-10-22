"""Hopper's (sm90) toolkit for the templates' Pallas Mosaic GPU mainloops (design §6): the launch,
warpgroup roles, barriers, shared-memory staging and the TMA store every sm90 mainloop builds on. Each
template keeps its own mainloop, product dependencies and launch DAG; only these primitives are shared.

A cooperative launch (`cooperative`) runs a producer warpgroup issuing the loads beside consumer warpgroups
computing, persistent over the multiprocessors, a pair of them a cluster where it shares one operand's blocks
(multicast, `multicast_axes`). The consumers report a tile drained (`drained`, `drain`) so a persistent
producer never loads a next tile's stage over one an MMA still reads, and meet (`meet`) where one consumer's
writes to shared memory are the other's operand. Each consumer's accumulator lives for one output tile
(`finished`), and a tile's rows reach global memory through `store_rows`, whose stores have static extents
however many rows a tile holds.

A mainloop that drives its own loads (attention's K and V, released at different points of a step) streams
each operand through a `Ring` whose counter runs across tiles, takes each warpgroup's role (`roles`), lets
two consumers take turns at the tensor cores (`turn`), walks its tiles in its own order (`persistent`) and
stores per-row vectors (`store_vector`).
"""

import dataclasses
import functools
import math

from jax import lax, numpy as jnp
from jax._src.state import types as state_types
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

Ref = state_types.AbstractRef | state_types.TransformedRef
"""A scoped allocation a primitive takes: shared memory, transformed (tiled and swizzled) or not, or
barriers."""


def compiler_params() -> plgpu.CompilerParams:
    """A mainloop's compiler parameters: Mosaic GPU's warpgroup lowering, whose arrays are a warpgroup's."""
    return plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Warpgroup)


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


def staging(name: str, shaped) -> dict:
    """Each consumer warpgroup's staging buffer for each operand a tile reads (`shaped`: its tile's shape and
    its operand) and a barrier for each, as `{name}_smems` and `{name}_barriers`."""
    shaped = list(shaped)
    return {f'{name}_smems': [plgpu.SMEM((2, *shape), gmem.dtype) for shape, gmem in shaped],
            f'{name}_barriers': [plgpu.Barrier(num_barriers=2) for _ in shaped]}


def load_fixed(views: list, refs: list, barriers: list, wg) -> None:
    """Each operand a tile reads whole (`views`) into this warpgroup's `refs` (`staging`), awaited."""
    for view, ref, barrier in zip(views, refs, barriers, strict=True):
        plgpu.copy_gmem_to_smem(view, ref, barrier.at[wg])
        plgpu.barrier_wait(barrier.at[wg])


def store_rows(o_smem, o_gmem, row_start, first, stored, column, most: int, *, grouped: bool = False) -> None:
    """Rows `first` to `first + stored` of `o_smem` to `o_gmem` at row `row_start + first` and `column`, in a
    power-of-two ladder of TMA stores (a descriptor's extents are static), the largest `most` rows'. Each
    store is its own group of copies, or with `grouped` none is, the caller committing them together
    (`plgpu.commit_smem_to_gmem_group`), so one wait covers a tile's stores whatever rungs it took."""
    start, size = first, most
    while size:
        rung = 1 << int(math.log2(size))
        size //= 2

        @pl.when(stored & rung != 0)
        def _(start=start, rung=rung):
            plgpu.copy_smem_to_gmem(o_smem.at[pl.ds(start, rung)],
                                    o_gmem.at[pl.ds(row_start + start, rung), pl.ds(column, o_smem.shape[1])],
                                    commit_group=not grouped)

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
        """A consumer's wait for step `counter`'s load: its slot."""
        slot = self.slot(counter)
        plgpu.barrier_wait(self.full.at[slot])
        return self.smem.at[slot]

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


def ring(block: tuple, dtype, stages: int, *, consumers: int = 1, transforms: tuple = ()) -> dict:
    """A `Ring`'s allocations for `pl.run_scoped`: `stages` slots of `block` in `dtype` (`transforms` their
    shared-memory tiling and swizzle), a load barrier each, and a release barrier each that `consumers`
    arrive at. `Ring(**allocated)` once scoped."""
    return {'smem': plgpu.SMEM((stages, *block), dtype, transforms=transforms),
            'full': plgpu.Barrier(num_barriers=stages),
            'consumed': plgpu.Barrier(num_arrivals=consumers, num_barriers=stages)}


def roles(wg, consumers: int, consumer, producer, *, registers: tuple[int, int] = (232, 40)) -> None:
    """Each warpgroup's role in a cooperative block: the `consumers` first run `consumer(wg)` with their
    register budget raised to `registers[0]`, the last runs `producer()` with its lowered to
    `registers[1]` (the producer only issues loads)."""
    @pl.when(wg < consumers)
    def _():
        plgpu.set_max_registers(registers[0], action="increase")
        consumer(wg)

    @pl.when(wg == consumers)
    def _():
        plgpu.set_max_registers(registers[1], action="decrease")
        producer()


def schedule_barrier():
    """The barrier two consumers meet at (`rendezvous`): two arrivals."""
    return plgpu.Barrier(num_arrivals=2)


def rendezvous(barrier) -> None:
    """One consumer's meeting with the other at the schedule barrier: it arrives and waits until both
    have. Symmetric meetings order nothing; a ping-pong of tensor-core phases is the mainloop's schedule
    on top of them, its phases offset by a meeting one consumer takes before the loop and the other after
    it (as `attention_mgpu` does), which each template keeps."""
    plgpu.barrier_arrive(barrier)
    plgpu.barrier_wait(barrier)


def persistent(count: int, *, collective_axes, order=None):
    """A persistent loop over `count` tiles, each block taking its share: `body(tile, loop)` with the tile
    `order(index)` names for the loop's linear index (the index itself without one: an LPT or causal
    order reverses it, a GEMM's planar snake folds it)."""
    def decorator(body):
        @plgpu.nd_loop((count,), collective_axes=collective_axes)
        def _(loop: plgpu.NDLoopInfo):
            index = loop.index[0]
            body(index if order is None else order(index), loop)
    return decorator


def store_vector(vector, smem, view) -> None:
    """A per-row vector (an attention tile's running max or sum) to `view` in global memory through
    `smem`, its layout's relayout there, once the store before it has read `smem`. The caller waits for the
    last store (`plgpu.wait_smem_to_gmem`) before `smem`'s scope ends or the vector is read back."""
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)
    smem[...] = vector.astype(smem.dtype)
    plgpu.commit_smem()
    plgpu.copy_smem_to_gmem(smem, view)
