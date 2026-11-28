"""The mma.sync instruction family's bundle (Ampere and its consumer kin, sm80 to sm89 and consumer
Blackwell; `generation.family` 'mma.sync') for the templates' Mosaic mainloops (`mosaic.Family`): one
warpgroup a program, its operand tiles streamed into shared memory by its own cp.async walk (`_streamed`),
read into registers in the MMA layouts (ldmatrix), and multiplied synchronously (mma.sync) into an
accumulator the walk carries in registers. An accumulator's value is the next product's left operand in
registers (the fork's relayout, jax 0.11.2.post13 and later), as FlashAttention-2 reuses its fragments.
Its layouts are its operands' dtype's (`Layout.MMA_ACC(dtype)`), so a bundle is a dtype's (`Mma`)."""

import abc
import dataclasses
import functools
import importlib
import itertools
import operator
from collections.abc import Callable, Iterator, Sequence
from typing import Any, ClassVar, Self

import jax
import jax.numpy as jnp
from jax import lax
from jax._src import core as jax_core
from jax._src.pallas.mosaic_gpu import core as plgpu_core
from jax._src.state import types as state_types
from jax.experimental import pallas as pl
from jax.experimental.mosaic import gpu as mgpu
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend.mlir import ir

from dew.nn.kernels import generation, launches
from dew.nn.kernels.templates import (
    bodies,
    emission,
    handoffs,
    mosaic,
    outputs,
    persistent,
    report,
    resources,
    rings,
    stores,
    swizzling,
)
from dew.nn.kernels.templates.cost import Stated
from dew.nn.kernels.templates.product import Support


def _staging(block: tuple, swizzle: int | None) -> tuple:
    """A stream's tile's transforms: a tile's swizzled (its own, `swizzle` where a stream names one); a side
    vector's none, in whole 16-byte cp.async chunks, and a tile's none where it is read a row at a time
    (`swizzling.UNSWIZZLED`)."""
    shape, dtype = block
    if len(shape) == 1 or swizzle == swizzling.UNSWIZZLED:
        return ()
    return swizzled(shape, dtype) if swizzle is None else swizzling.swizzle_transforms(dtype, swizzle)


def _at(index_map, placed: Callable, trip):
    """A stream's block indices at a pipeline's trip `trip`: the walk's step `placed(trip)` (`walk`)."""
    return index_map(placed(trip))


@dataclasses.dataclass(frozen=True)
class _Contiguous:
    """A walk's copy policy for blocks its operands hold whole (`_filled`): each stream's block a cp.async
    copy and a group of its own, none committed past the walk's last trip, so a step's wait counts the later
    steps' groups (`_Walk.run`'s drain); the copy's native lowering guards the reads of the slot it refills.
    Its slots `_ring`'s, staged swizzled for ldmatrix."""

    sources: dict
    """Each stream's `[S, D]` operand and its block indices at a trip, by name."""

    empty_commits: ClassVar[bool] = False
    guards_refill: ClassVar[bool] = False

    def groups(self, pipeline: "Pipeline") -> int:
        return len(pipeline.blocks)

    def fill(self, walk: "_Walk", slot, trip, ordinal) -> None:
        """Trip `trip`'s blocks into `slot`, the walk's `ordinal`-th fill, which the walk issues only within
        its trips (`_Walk.fetched`, `prologue`)."""
        del ordinal
        _filled(walk.pipeline, self.sources, walk.slots, slot, trip, walk.interpret)


@dataclasses.dataclass(frozen=True)
class _Gathered:
    """A walk's copy policy for blocks whose rows a table gathers (`Pipeline.gathered`): a step's rows copied
    by the warpgroup a row a copy (`gathered.row_copies`, `_row_copied`), every stream's copies one cp.async
    group, committed empty past the walk's last trip so a step's wait count stays one immediate, and the
    warpgroup synced before a refill (`mosaic.synced`: no lowering guards its reads of that slot), a
    one-slot walk's step filling its own slot before its wait. Interpreted, each row copied by the program's
    own loads and stores. Its slots `_ring`'s, staged at the swizzle its rows' copy tiles."""

    views: dict
    """Each stream's operand and its gathered rows at a step (`Streams.views`), by name."""
    placed: Callable
    """A trip's step in the walk."""

    empty_commits: ClassVar[bool] = True
    guards_refill: ClassVar[bool] = True

    def groups(self, pipeline: "Pipeline") -> int:
        del pipeline
        return 1

    def fill(self, walk: "_Walk", slot, trip, ordinal) -> None:
        """Trip `trip`'s rows into `slot`, the walk's `ordinal`-th fill (a count past its trips, not wrapped):
        its copies where the ordinal is one of the walk's trips, its group committed either way."""
        from dew.nn.kernels.templates import gathered as gathered_rows  # it builds on the toolkit's rings

        @pl.when(ordinal < walk.trips)
        def _():
            for name, ring in walk.slots.items():
                ref, index_map = self.views[name]
                rows = index_map(self.placed(trip))
                [depth] = rows.rest
                height, width = walk.pipeline.blocks[name][0]
                source = ref.at[:, pl.ds(depth * width, width)]
                table = rows.table.at[pl.ds(rows.start, height)]
                # At the swizzle its ring is staged at (`_ring`), the one decision.
                _row_copied(source, table, ring.at[slot], swizzling.swizzle_of(ring), walk.interpret)

        if not walk.interpret:
            plgpu.inline_mgpu()(gathered_rows.CommitGroup())()


def _depth(pipeline: "Pipeline", steps: int | None) -> int:
    """The slots a stream's ring holds in a walk of `steps` steps over `pipeline`: a stage each, no more than
    the walk's steps where they are known."""
    return pipeline.stages if steps is None else min(pipeline.stages, steps)


def _ring(pipeline: "Pipeline", steps: int | None) -> tuple:
    """The shared memory a walk of `steps` steps over `pipeline`'s streams scopes (`_streamed`, which runs it,
    and `CpAsync.lifetimes`, which prices it): `_depth`'s slots of each stream's block, its tile staged for
    the copies that fill it (a gathered walk's at the swizzle its rows' copy tiles, `gathered.swizzled`;
    another's swizzled for ldmatrix, `_staging`); and, where several consumers run the walk together, the
    two barriers they meet at in turn, so none arrives at a barrier's next phase before every one has seen
    its last."""
    from dew.nn.kernels.templates import gathered as gathered_rows  # it builds on the toolkit's rings

    depth = _depth(pipeline, steps)
    if pipeline.gathered:
        slots = tuple(swizzling.staged((depth, *shape), dtype, gathered_rows.swizzled(shape, dtype))
                      for shape, dtype in pipeline.blocks.values())
    else:
        slots = tuple(plgpu.SMEM((depth, *shape), dtype,
                                 transforms=_staging((shape, dtype), pipeline.swizzles.get(name)))
                      for name, (shape, dtype) in pipeline.blocks.items())
    meets = () if pipeline.consumers == 1 else (plgpu.Barrier(num_arrivals=pipeline.consumers,
                                                             num_barriers=2),)
    return slots + meets


def _filled(pipeline: "Pipeline", sources: dict, slots: dict, slot, trip, interpret: bool) -> None:
    """Each stream's block of the walk's `trip` (`sources[name]`: its `[S, D]` operand and its block indices
    at a trip) copied into its `slot`: by cp.async, each copy a group of its own, every consumer its share of
    the block's rows (whole 8-row tiles of each of a warpgroup's four warps; a vector's whole 16-byte
    chunks) where several run the walk (the calling one's read where it copies, held live no longer);
    interpreted, loaded and stored by one consumer alone, so no two write one row. Each
    read inside its operand: the launch extends every operand its rows read past (`bounded_loads`,
    `mosaic_gpu._walked_rows`), so each copy is promised in bounds, unchecked."""
    for name, (shape, _) in pipeline.blocks.items():
        ref, index_map = sources[name]
        source = ref.at[tuple(pl.ds(index * size, size) for index, size in zip(index_map(trip), shape,
                                                                               strict=True))]
        destination = slots[name].at[slot]
        if interpret:
            def copied(source=source, destination=destination):
                destination[...] = source[...]

            # One consumer's store of the whole block, ordered before any reads by the meets.
            if pipeline.consumers == 1:
                copied()
            else:
                pl.when(lax.axis_index("wg") == 0)(copied)
            continue
        if pipeline.consumers == 1:
            plgpu.copy_gmem_to_smem(source, destination, None, oob_mode=plgpu.OOBFillMode.PROMISE_IN_BOUNDS)
            continue
        rows, threads = shape[0], pipeline.consumers
        granule = 4 * 8 if len(shape) > 1 else 16 // jnp.dtype(destination.dtype).itemsize
        if rows % (threads * granule):
            raise NotImplementedError(f"a block of {rows} rows splits into no {threads} shares of whole "
                                      f"{granule}-row chunks")
        part = rows // threads
        thread = lax.axis_index("wg")
        for share in range(threads):
            @pl.when(thread == share)
            def _(share=share, source=source, destination=destination, part=part):
                rows_of = pl.ds(share * part, part)
                plgpu.copy_gmem_to_smem(source.at[rows_of], destination.at[rows_of], None,
                                        oob_mode=plgpu.OOBFillMode.PROMISE_IN_BOUNDS)


def _streamed(pipeline: "Pipeline", copier: "_Contiguous | _Gathered", trips, body: Callable, carry, *,
              interpret: bool, prefetch: Callable | None = None):
    """A walk of `trips` steps over `pipeline`'s streams, its stages' slots (`_ring`) a ring each
    step's blocks are copied into by cp.async (`copier`'s policy: `_Contiguous` blocks, `_Gathered` rows),
    `body(trip, slots, carry)` giving the next carry from the step's slots (each stream's by name).
    CUTLASS's multistage order: a step waits for its own copies (`plgpu.wait_gmem_to_smem`, the later
    steps' groups in flight, its count an immediate: the drain's unrolled where the steps are known, each
    awaited whole where not, every step's the same where the copier commits a group every step), then
    fetches the step `stages - 1` ahead into the slot the step before read, the native cp.async lowering's
    barrier ahead of each copy the guard of those reads (or the copier's sync, `guards_refill`; on an A100
    the fetch after the body left warps waiting at three warpgroup barriers a step, 13.5% of a dense GEMM's
    stall samples, row 2152). Its slot is carried, not
    recomputed (three stages on an A100: the remainder each step ran a GEMM at 13.3 ms against 9.1). Several
    consumers running it together (`pipeline.consumers`, the kernel's thread axis 'wg') each copy their
    share of every block and meet past each wait and each step, so every share has landed before any reads
    and every one has read a slot before any refills it.

    With `prefetch(slots)` (two slots or more), a step's first fragments are read under the step before's
    last product: `body(trip, slots, carry, prefetched, boundary)` returns `(next carry, next prefetched)`,
    `boundary()`, called once in the body's own trace, awaiting the next step's slots and reading theirs
    (the step's own standing in past the last step); a walk of trips known only as it runs, none of them,
    fills no slot and reads none."""
    known = isinstance(trips, int)
    stages = _depth(pipeline, trips if known else None)
    if prefetch is not None and stages == 1 and not (known and trips == 1):
        raise ValueError("a walk's prefetch needs two slots or more: one is refilled past its step's body")

    def scoped(*refs):
        copies = len(pipeline.blocks)
        walk = _Walk(pipeline, copier, trips, stages, interpret, prefetch,
                     dict(zip(pipeline.blocks, refs[:copies], strict=True)),
                     refs[copies] if pipeline.consumers > 1 else None)
        walk.prologue()
        if prefetch is None or known:
            return walk.run(body, carry)
        # A walk of no steps fills no slot and reads none: its carry crosses the guard alone, at the types
        # the loop gives it (a weak float's promoted to a narrower one).
        typed = jax.eval_shape(functools.partial(walk.run, body), carry)
        return lax.cond(trips > 0, functools.partial(walk.run, body), lambda carried: jax.tree.map(
            lambda value, aval: lax.convert_element_type(value, aval.dtype), carried, typed), carry)

    return swizzling.run_scoped(scoped, *_ring(pipeline, trips if known else None),
                                collective_axes="wg" if pipeline.consumers > 1 else ())


@dataclasses.dataclass(frozen=True)
class _Walk:
    """`_streamed`'s walk within its scope: its streams' `slots` (each stream's ring by name) and, where
    several consumers run it, the two barriers they `meets` at in turn (`_ring`)."""

    pipeline: "Pipeline"
    copier: "_Contiguous | _Gathered"
    trips: "int | jax.Array"
    stages: int
    interpret: bool
    prefetch: Callable | None
    slots: dict
    meets: Any

    @property
    def known(self) -> bool:
        return isinstance(self.trips, int)

    @property
    def lead(self) -> int:
        """1 where the step fetches before its body (two slots or more, or a copier syncing before its
        refill, `guards_refill`), 0 where after it."""
        return int(self.stages > 1 or self.copier.guards_refill)

    @property
    def own(self) -> bool:
        """Whether a step fetches its own trip, before its wait: a one-slot walk fetching before its body."""
        return self.stages - self.lead == 0

    @property
    def copies(self) -> int:
        """The cp.async groups a step's fetch commits (`copier.groups`)."""
        return self.copier.groups(self.pipeline)

    def meet(self) -> None:
        if self.meets is not None:
            for barrier in (self.meets.at[0], self.meets.at[1]):
                plgpu.barrier_arrive(barrier)
                plgpu.barrier_wait(barrier)

    def landed(self, count: int) -> None:
        """This step's copies landed, each consumer's share of them, the later steps' `count` in flight."""
        if not self.interpret:
            plgpu.wait_gmem_to_smem(count)
        self.meet()

    def at(self, slot) -> dict:
        return {name: ring.at[slot] for name, ring in self.slots.items()}

    def following(self, slot):
        wrapped = slot + 1
        return jnp.where(wrapped == self.stages, 0, wrapped)

    def wrapped(self, trip):
        """The trip after `trip`, the first past the last: a count the compiler takes for no induction
        variable, so it peels no step off the walk's loop for a body's test of its trip and shares no address
        with the prologue's copies (an A100's recurrence, peeled, contracted its sums into fmas whose operands
        spilled; its fused backward held the prologue's operand base live through the walk, spilling 8
        bytes), as Mosaic GPU's pipelines carry their grid indices."""
        following = trip + 1
        return jnp.where(following == self.trips, 0, following)

    def fetched(self, index, ahead, slot) -> None:
        """The fetch of trip `ahead` (wrapped) into `slot`, the walk's `index`-th step's, the walk's fill
        `index + stages - lead` (not wrapped): a copier committing a group every step (`empty_commits`) fills
        each, past the last trip an empty one; another fills a trip the walk holds alone."""
        ordinal = index + (self.stages - self.lead)
        if self.copier.guards_refill:
            mosaic.synced(self.interpret)
        if self.copier.empty_commits:
            self.copier.fill(self, slot, ahead, ordinal)
            return
        lax.cond(ordinal < self.trips, lambda: self.copier.fill(self, slot, ahead, ordinal), lambda: None)

    def prologue(self) -> None:
        """The steps before the first fetched, a slot each, its trip carried (`wrapped`) beside the slot,
        the loop's index."""
        def filled(slot, trip):
            # The loop's index the fill's slot and its ordinal, its trip carried wrapped.
            self.copier.fill(self, slot, trip, slot)
            return self.wrapped(trip)

        # Every stage-ahead slot where each commits a group however few trips remain; else the trips there
        # are, a traced count in its own dtype (an int64 segment's under x64), which `lax.min` matches.
        if self.copier.empty_commits or isinstance(self.trips, int):
            count = self.stages - self.lead
            known = True
        else:
            count = lax.min(jnp.asarray(self.stages - self.lead, self.trips.dtype), self.trips)
            known = False
        lax.fori_loop(0, count, filled, jnp.int32(0), unroll=known)

    def boundary(self, index, slot, prefetched, next_wait: int) -> tuple[Callable, list]:
        """A step's boundary (`_streamed`'s `prefetch`) and the record of its calls: the next step's slots
        awaited and their first fragments read, once in the body's trace (a wait under a cond or a loop would
        run on some paths alone while the next step skips its own), the step's own standing in past the last
        step."""
        prefetch, outer, called = self.prefetch, jax_core.trace_ctx.trace, []
        assert prefetch is not None

        def boundary():
            if jax_core.trace_ctx.trace is not outer or called:
                raise ValueError("a walk's body calls its boundary once, in its own trace")
            called.append(True)
            self.landed(next_wait)
            return lax.cond(index + 1 < self.trips, lambda: prefetch(self.at(self.following(slot))),
                            lambda: prefetched)

        return boundary, called

    def step(self, body: Callable, index, state, *, wait: int, next_wait: int):
        """One step: its trip and the one it fetches carried (`wrapped`), as its slot and the one it fills
        are; its tests read the loop's `index`."""
        trip, ahead, carried, slot, fill_slot, prefetched = state
        if self.own:
            self.fetched(index, ahead, fill_slot)
        if self.prefetch is None:
            self.landed(wait)
        if self.lead and not self.own:
            self.fetched(index, ahead, fill_slot)
        if self.prefetch is None:
            carried = body(trip, self.at(slot), carried)
        else:
            boundary, called = self.boundary(index, slot, prefetched, next_wait)
            carried, prefetched = body(trip, self.at(slot), carried, prefetched, boundary)
            if not called:
                raise ValueError("a walk's body with a prefetch calls its boundary: the next step does not "
                                 "wait for its slots")
        # Every consumer has read the slot before any refills it, and before the slots leave scope.
        self.meet()
        if not self.lead:
            self.fetched(index, ahead, fill_slot)
        return (self.wrapped(trip), self.wrapped(ahead), carried, self.following(slot),
                self.following(fill_slot), prefetched)

    def run(self, body: Callable, carried):
        """The walk's steps from its first, its carry `carried`: the steady ones leaving every later group in
        flight, then the drain (unrolled at fewer where the steps are known, each awaited whole where not)."""
        stages, lead, copies = self.stages, self.lead, self.copies
        first = None
        if self.prefetch is not None:
            # The first step's slots, awaited before the loop as each later step's are by the one before (its
            # boundary): past the groups the prologue issued after the first step's, every one where the steps
            # are not known.
            self.landed((stages - lead - 1) * copies if self.known else 0)
            first = self.prefetch(self.at(jnp.int32(0)))
        ahead = jnp.int32(0)
        for _ in range(stages - lead):
            ahead = self.wrapped(ahead)
        state = (jnp.int32(0), ahead, carried, jnp.int32(0), jnp.int32((stages - lead) % stages), first)
        # The steps whose wait leaves every later group in flight, the last of them stopping one short where
        # a boundary awaits the next step's slots at the next step's count.
        steady = self.trips - (stages - 1 - lead) - (self.prefetch is not None)
        in_flight = max(0, stages - 1 - lead) * copies
        if self.copier.empty_commits:
            # Every step's wait the same: the later stages' groups always committed, empty past the last.
            state = lax.fori_loop(0, self.trips, functools.partial(self.step, body, wait=in_flight,
                                                                   next_wait=in_flight), state)
        elif isinstance(self.trips, int):
            state = lax.fori_loop(0, max(0, steady), functools.partial(self.step, body, wait=in_flight,
                                                                       next_wait=in_flight), state)
            # A wait's count an immediate: the drain unrolled, each step's fewer.
            for index in range(max(0, steady), self.trips):
                state = self.step(body, jnp.int32(index), state, wait=(self.trips - 1 - index) * copies,
                                  next_wait=max(0, self.trips - 2 - index) * copies)
        else:
            steady = jnp.maximum(steady, 0)
            state = lax.fori_loop(0, steady, functools.partial(self.step, body, wait=in_flight,
                                                               next_wait=in_flight), state)
            state = lax.fori_loop(steady, self.trips, functools.partial(self.step, body, wait=0, next_wait=0),
                                  state)
        # No copy outlives the walk's scope: every group its fills committed retired as the scope ends.
        if not self.interpret:
            plgpu.wait_gmem_to_smem(0)
        return state[2]


def _row_copied(source, table, slot, swizzle: int, interpret: bool) -> None:
    """`slot`'s rows (`[height, width]`), row `r` `source`'s row `table[r]`: the warpgroup's cp.async copies
    (`gathered.row_copies`), issued and not awaited; interpreted, each row loaded and stored."""
    from dew.nn.kernels.templates import gathered as gathered_rows  # it builds on the toolkit's rings

    if interpret:
        def copied(row, _):
            slot[pl.ds(row, 1), :] = source[pl.ds(table[row], 1), :]

        lax.fori_loop(0, slot.shape[0], copied, None)
        return
    transforms = swizzling.swizzle_transforms(source.dtype, swizzle)
    refs = (plgpu.RefType(), plgpu.RefType(), plgpu.RefType(transforms))
    plgpu.inline_mgpu(arg_types=refs)(gathered_rows.RowCopies(swizzle))(source, table, slot)


def swizzled(shape: tuple[int, ...], dtype) -> tuple:
    """A `[rows, columns]` shared-memory tile's transforms: 8-row tiles as wide as the swizzle its row
    bytes allow and its cp.async copy tiles (`swizzling.copied_swizzle`), swizzled (bank-conflict-free
    ldmatrix), as jax's Ampere examples stage their operands."""
    return swizzling.swizzle_transforms(dtype, swizzling.copied_swizzle(shape, dtype))


def _depth_granule(ref, axis: int, instruction: int) -> int:
    """The depth slice a ref's staging allows: one MMA, or a whole swizzle along its tiled columns (no
    narrower slice of a swizzled tile lowers, FS-NARROW-LOWER1), its staging and views as it reads them
    (`swizzling.staging_of`: JAX 0.12.1's refs state none of their staging as they trace)."""
    transforms = swizzling.staging_of(ref)
    if not transforms:
        return instruction
    transposed = sum(isinstance(transform, state_types.TransposeTransform)
                     and transform.permutation == (1, 0) for transform in transforms) % 2
    if axis != (0 if transposed else 1):
        return instruction
    return max(instruction, swizzling.swizzle_columns(ref.dtype, swizzling.swizzle_of(ref)))


@dataclasses.dataclass(frozen=True)
class Pipeline:
    """A program's operand streams as `Mma.streams` describes them: each stream's tile `(shape, dtype)` by
    name, the tiles its walk keeps in flight (`stages`, `_streamed`) and the consumer warpgroups that run it
    together (`consumers`, its thread axis 'wg' where two); it allocates their slots itself."""

    blocks: dict
    stages: int
    swizzles: dict = dataclasses.field(default_factory=dict)
    """The swizzle a stream's tiles stage at where `Mma.streams` names one (`swizzling.swizzle_for`)."""
    gathered: tuple = ()
    """The streams whose step's rows a table gathers (a `Rows` view of one row a run): the warpgroup's own
    cp.async copies, a row a copy (`gathered.row_copies`), into slots of the walk's own, in the pipeline's
    cp.async groups (`_Gathered`); a walk gathers every stream or none."""
    consumers: int = 1
    """The consumer warpgroups that run its walk together (`_streamed`, the kernel's thread axis 'wg' where
    two), each copying an equal share of every block and reading every slot whole."""


@dataclasses.dataclass(frozen=True)
class Streams:
    """A program's operand streams (`Mma.opened`): its pipeline and each stream's view `(ref, block)` by
    name, the `[S, D]` operand its steps' tiles come from in `block` rows."""

    pipeline: Pipeline
    views: dict
    agreed: tuple | None = None
    """The steps `(lo, hi)` every consumer of a pipeline two run loads together, where `walk_own` agrees
    them: a walk loads them in turn whatever its consumer's own segments (`CpAsync.walk`). Its callers'
    invariants: the streamed range is every consumer's alike, and a step's rendezvous across consumers (a
    sum across them, `across.summed_across`) is reached by every consumer at that step."""
    common: bool = False
    """Whether every consumer's segments of a walk are the same steps (`walk_own`'s `common`, its caller's
    statement: no bound derived from its consumer): a walk of several then runs a pipeline a segment, each
    keeping its prefetch, its consumers meeting paired at every one."""


@dataclasses.dataclass(frozen=True)
class MmaIssuer(mosaic.Issuer[jax.Array]):
    """mma.sync's issuer and home for operands of `dtype`: accumulators in registers, products synchronous,
    each operand loaded into its fragments from shared memory (ldmatrix), its warps `m_warps` along a tile's
    rows and the rest along its columns (JAX's `MMALayouts`; `specialized`)."""

    dtype: jnp.dtype

    m_warps: int = 4

    relay: Any = dataclasses.field(default=None, compare=False)
    """A consumer's shared-memory tile its register left operands relay through where its warps split their
    columns (`relaying`, `relay_bytes`)."""

    def published(self) -> None:
        """Nothing to publish: its products read shared memory through ldmatrix, the warpgroup's own loads,
        which the stores' barriers order, on any movement (cp.async's, TMA rings a consumer or a producer
        fills); an async proxy's fence would order no reader (a Kimi intra walk's three a step on a GH200,
        M-JS-272)."""

    row_bound: Any = None
    """The tile's rows its products issue before, an int32 (`bounded`): each warp's 16-row blocks at or past
    it issue none, its accumulator's rows kept; None issues every row. Its identity's: a family bounded at
    one constant is another program than at the next."""

    bound_registers = 16
    """Its row bound's registers (`bounded`): the runtime int32 bound and each 16-row block's predicate took
    an sm80 64 x 192 x 64 normed, biased one-consumer grouped forward from 244 modeled registers to 255, 32
    bytes spilled (rows 5623, 7091)."""

    def side_bytes(self, dtype) -> int:
        # A full register a value in its accumulator's MMA layout (`side_tile`): an sm86 residual forward
        # priced at a bf16 side's width spilled 648 bytes at 64 x 192 (row 12689's gate).
        del dtype
        return 4

    held_registers = 40
    """The registers a consumer thread holds beside its accumulator and operand fragments (addresses,
    indices, the walk's carry), a measured whole (`scalars_whole`): CUTLASS's cooperative sm90 kernel's 168
    less its accumulator's 128; an A100's attention dQ holds 13 to 43 integers live at its peaks, places and
    addresses together. Its derivations need it: without it the GEMM's gate picks on sm80, sm86, sm89 and
    sm120 spill 36 to 784 bytes, where its operands' fragments share the registers (FS-HELDREG). A
    pre-lowering prior, which certification overrules (`report.certified`): the count a lowering holds is
    not derived from the traced kernel, whose shared-memory tile addresses and loop carries predicted the
    lowered peak's scalars no better than a family's constant over 260 sm80 GEMM and recurrence kernels
    (FS-DERIVED-HELD2-sm80, row 8377: correlation 0.49 and 0.39; a recurrence's peak 4 to 97 scalars, two to
    five times its traced count, the lowering's shared-memory addresses and swizzles)."""

    scalars_whole = True

    def scalar_registers(self, addressed=()) -> int:
        # Its measured whole, its addresses among them.
        del addressed
        return self.held_registers

    rows = 16

    # Its kernels take hooks and exact sums: an epilogue's side values read in its accumulator's layouts
    # (`side_tile`, `chunks`), its running sum counted beside the fragments its products hold and one
    # step's fresh product (`summed_apart`, `grouped_matmul._Mosaic.registers`); no prologue over a side
    # tile (`takes_prologue_tiles`).
    takes_prologue_tiles = False

    role_rows = 4 * 16

    operands_in_registers = True

    def depths(self, dtype) -> tuple[int, ...]:
        # A step's fragments read by ldmatrix from a block staged at a 128- or 64-byte swizzle, cp.async's
        # (`Mma`, `swizzled`) or a TMA ring's at the widest that fits (`bundles.MmaFed`, `swizzling.staged`):
        # the shallower step holds a balanced tile's ring where the deeper does not (an RTX PRO 6000's 99 KB
        # block at 128 x 128, cuBLASLt's 32-deep rings).
        return tuple(swizzling.swizzle_columns(dtype, swizzle) for swizzle in (swizzling.SWIZZLE_BYTES, 64))

    def column_unit(self, dtype) -> int:
        # A warp's lanes, whatever loader stages its tiles: the guarded attention walk's 32-column key
        # blocks held where its 64 held no tile's registers at width 128 (its fragments are registers), a
        # TMA ring staging such a block at a 64-byte swizzle (`swizzling.staged`); cp.async's stores move rows
        # through registers, a row's columns over a warp's lanes at least (`swizzling.copied`).
        del dtype
        return resources.THREADS // 4

    products_overlap = False
    operand_steps_ahead = 1
    """`products` loads the following instruction's slices before issuing the current instruction."""

    takes_tf32 = True

    transposes_tf32 = True

    rounds_tf32 = True

    def for_dtype(self, dtype) -> Self:
        """The operand's issuer: fp32 products issue TF32 at its own depths, beside 16-bit products."""
        dtype = jnp.dtype(dtype)
        return self if self.dtype == dtype else dataclasses.replace(self, dtype=dtype)

    def specialized(self, rows: int, columns: int) -> Self:
        # Its warps split `m_warps` along a warpgroup's `rows` and the rest along its `columns` (`MMALayouts`;
        # each product states it, `plgpu.mma`'s `m_warps`): of the splits whose atom, `m_warps * 16` rows by
        # `(4 // m_warps) * 8` columns, divides the share, the one reading its operands from shared memory
        # fewest times a step, each warp reading its rows' share whole and its columns' share whole
        # (`rows * (4 // m) + columns * m`), the most along the rows on a tie. A 128 x 128 share so splits
        # 2 x 2 (512 reads a step, where four warps along its rows read 640), each warp holding half the
        # right operand's fragments.
        splits = [m for m in (4, 2, 1) if rows % (m * 16) == 0 and columns % ((4 // m) * 8) == 0]
        m_warps = min(splits, key=lambda m: (rows * (4 // m) + columns * m, -m)) if splits else 4
        return dataclasses.replace(self, m_warps=m_warps)

    @property
    def atom_columns(self) -> int:
        # A warp's atom `m_warps * 16` rows by `(4 // m_warps) * 8` columns (`specialized`): a 16-row
        # share's four warps along its columns tile 32 of them.
        return (4 // self.m_warps) * 8

    @property
    def transposed_rows(self) -> int:
        # The transposed stage casts to the products' left-operand layout in whole row atoms, `m_warps`
        # warps of 16 rows: a 32-row step at two row warps lowers, where the 64-row role it was held to
        # left an A100's dense kernel gradients 18 to 29% slower (BC-A100-DW-AB2: Llama 3 8B o's 1.680 to
        # 1.378 ms, gate_up's 10.95 to 7.80).
        return self.m_warps * 16

    @property
    def lhs_copies(self) -> int:
        return 4 // self.m_warps

    @property
    def rhs_copies(self) -> int:
        return self.m_warps

    def fragment_depth(self, depth: int) -> int:
        # The right operand's fragments are loaded one MMA depth at a time (16 for 16-bit operands:
        # `MMALayouts`' k), CUTLASS's warp-level k-iterations, sliced along the stage's rows, which its
        # 8-row tiles allow (a swizzled row takes no slice narrower than its tile).
        return min(depth, 8 * max(32 // (jnp.dtype(self.dtype).itemsize * 8), 1))

    def fed_columns(self, rows: int, columns: int) -> int:
        # Where a warp's rows are one instruction's (`m_warps` warps of 16 along the product's rows), each
        # right fragment, an n8 tile's, feeds one instruction: each column warp's load feeds two of its
        # (`ldmatrix`'s four 8 x 8 matrices of 16-bit rows; a 32-bit operand's own loads one), beside the
        # loads ahead of it (`operand_steps_ahead`), and ptxas sinks each load to its instruction, so a
        # k-step's loads over many columns are never live at once: the column warps' atom
        # (`atom_columns`, an n8 tile each) that many times. An sm120 forward 256 wide at 64 x 32 held 237
        # registers, 9 past its accumulator's growth from width 128, where its values' fragments over every
        # column priced 260 and refused its one launch (FS-14113-CENSUS2, row 14113). A warp of more rows
        # reads each fragment once a row tile and holds them over every column: a grouped GEMM 2880 wide so
        # priced took 64 x 320 tiles (one row warp) that spilled 112 to 3,732 bytes.
        if rows > self.m_warps * self.rows:
            return columns
        per_load = 2 if self._matrix_loads else 1
        return min(columns, self.atom_columns * per_load * (1 + self.operand_steps_ahead))

    def rhs_fragment_depth(self, depth: int, *, transposed: bool = False, swizzle: int | None = None) -> int:
        # A transposed stage's rows are its depth: a load takes a whole swizzle tile of it (`products`), one
        # instruction's at the narrowest swizzle (a stream staged so by `streams`' `swizzles`: D64's K in
        # QKᵀ at the widest held its whole 64 x 192 stage, 162 registers, census80). An ordinary one slices
        # its rows' 8-row tiles. Which slices of it stay live is `operand_live_depth`'s.
        return self._staged_depth(depth, swizzle) if transposed else self.fragment_depth(depth)

    def operand_live_depth(self, depth: int, loaded: int, *, retained: bool = False,
                           transposed: bool = False) -> int:
        # A transposed granule read once (not `retained`) that is its step's whole depth at two
        # instructions dies a slice at a time as its products issue: one instruction's live, an ordinary
        # one's lookahead (a transposed matrix's input gradient held as its forward is at a 32-deep step,
        # FS-5634-DX-sm80/-sm86). Deeper granules keep the rule below: a 64-deep step's input gradient ran
        # 24-40 registers past its forward, and a recurrence backward's 64-deep transposed reads compiled
        # spill-free at it (FS-5634-RECUR). A retained one is read again by a later product and keeps its
        # whole granule (`_query_key_held`).
        instruction = self.fragment_depth(depth)
        if transposed and not retained and loaded == depth == 2 * instruction:
            return instruction
        return super().operand_live_depth(depth, loaded, retained=retained, transposed=transposed)

    def lhs_fragment_depth(self, depth: int, *, swizzle: int | None = None) -> int:
        # A whole tile of the swizzle its stage takes: one instruction's depth where it is staged plain or
        # narrowest, as a D128 attention forward's queries are, which held 64 deep spilled the 16 registers
        # of its 128 x 64 tile (FS-4324-NARROW2).
        return self._staged_depth(depth, swizzle)

    def _staged_depth(self, depth: int, swizzle: int | None = None) -> int:
        """The depth a load of a stage whose rows are `depth` deep takes: one tile of its `swizzle` (bytes;
        the widest its row bytes allow where None, `swizzled`, `_depth_granule`), the whole depth where it
        is narrower."""
        swizzle = resources.widest_swizzle(self.dtype, depth) if swizzle is None else swizzle
        return max(self.fragment_depth(depth), min(depth, swizzling.swizzle_columns(self.dtype, swizzle)))

    reduction_scratch = plgpu.CompilerParams().reduction_scratch_bytes  # its kernels' default parameters

    column_scratch = 0  # its column sums take its kernels' reduction scratch

    @property
    def lane_columns(self) -> int:
        # A thread holds two adjacent columns of each 8 (lane % 4 times two) in its warp's 8 of each
        # `n_warps * 8` (`MMALayouts`' accumulator tiling).
        return 6 | 8 * (4 // self.m_warps - 1)

    def written(self, smem, value: jax.Array) -> None:
        # Its accumulator's layout at one or two row warps takes no optimized register-to-shared transfer
        # (FS-T1-ROWS2: a 32-row tile on TMA rings found no layout, "Failed to infer a possible set of
        # layouts"; written unoptimized it lowers at 8 to 64 rows, FS-T1-ROWS3), as its own stores take it.
        plgpu.store(smem, value.astype(smem.dtype), optimized=False)


    @property
    def layout(self):
        return plgpu.Layout.MMA_ACC(self.dtype, m_warps=self.m_warps)

    @property
    def row(self):
        return self.layout.reduce(1)

    @property
    def column(self):
        return self.layout.reduce(0)

    @property
    def lhs(self):
        return plgpu.Layout.MMA_LHS(self.dtype, m_warps=self.m_warps)

    def hook_columns(self, columns: int) -> int:
        # Its accumulator in registers: a hook takes `HOOK_COLUMNS` of it at a time (`WgmmaIssuer`'s).
        return min(columns, mosaic.HOOK_COLUMNS)

    def chunks(self, accumulators: tuple, columns: int | None = None) -> Iterator[tuple[int, tuple]]:
        # The accumulators in their own layout, stated: an epilogue's broadcast of a side vector across one
        # takes the accumulator's layout as its result's (`side_tile`'s vectors its reductions), where left to
        # inference it took the vector's ("Tiling rank different than broadcast result rank, (16,) vs
        # (64, 128)", a scaled forward on sm80, EM-hooks6).
        yield from super().chunks(tuple(plgpu.layout_cast(acc, self.layout) for acc in accumulators), columns)

    def side_tile(self, ref, held_axes: tuple):
        # In the accumulator's layout, or the accumulator's vector layout of
        # the axis it follows. Read back unannotated, Mosaic GPU's layout inference exhausted its search on
        # every epilogue over a side vector (a scaled and a row-scaled forward on sm80, EM-hooks5).
        if len(ref.shape) > 1:
            return plgpu.load(ref, layout=self.layout, optimized=False)
        return plgpu.load(ref, layout=self.row if held_axes == ('rows',) else self.column, optimized=False)

    @property
    def rhs(self):
        return plgpu.Layout.MMA_RHS(self.dtype, m_warps=self.m_warps)

    def column_sums(self, value: jax.Array, scratch, interpret: bool) -> jax.Array:
        # One warpgroup a program: Mosaic GPU's own reduction across its warps, through the scratch its
        # kernels' parameters reserve (`reduction_scratch`).
        del scratch, interpret
        return value.sum(axis=0)

    def left_of(self, ref) -> jax.Array:
        return self._left(ref)

    def _left(self, lhs):
        """A product's left operand in the MMA's layout: a shared-memory tile read, or a register value's (an
        accumulator's) cast, in registers where each warp holds whole rows of it or no relay tile is bound,
        else through the bound one (`relaying`): every warp's columns stored, then each warp's rows read
        whole, a sync after each so no warp reads a column not yet stored or stores over one another still
        reads."""
        if not isinstance(lhs, jax.Array):
            return plgpu.load(lhs, layout=self.lhs, optimized=self.rows_loaded)
        value = lhs.astype(self.dtype)
        if self.relay is None or not self.relay_bytes(*value.shape):
            # Unbound, the cast is Mosaic's own, which crosses warps only where its layouts allow.
            return plgpu.layout_cast(value, self.lhs)
        # A plain tile, read element-wise: a warp's replicated rows (one or two row warps) take no ldmatrix
        # free of bank conflicts from it, and a swizzled one takes no store of the accumulator's layout.
        tile = self.relay.at[:value.shape[0], :value.shape[1]]
        plgpu.store(tile, value, optimized=False)
        mosaic.synced(interpret=False)
        loaded = plgpu.load(tile, layout=self.lhs, optimized=False)
        mosaic.synced(interpret=False)
        return loaded

    @property
    def rows_whole(self) -> bool:
        # Its warps along a tile's rows alone (four of them) hold whole rows of the accumulator and the
        # operand alike; fewer split the accumulator's columns among warps.
        return self.m_warps == 4

    def relay_bytes(self, rows: int, depth: int) -> int:
        # No register cast crosses the warps a row's columns are split among.
        return 0 if self.rows_whole else rows * depth * self.dtype.itemsize

    def relaying(self, tile) -> Self:
        # A plain tile (`swizzling.plain`).
        return dataclasses.replace(self, relay=tile)

    @property
    def _matrix_loads(self) -> bool:
        """Whether its operands load by `ldmatrix`, 16-bit rows of 8: not a 32-bit operand's (TF32's), for
        which Mosaic GPU's layout inference found no layout (tf32-mini at 9d77b39e7), its threads loading
        their elements themselves."""
        return self.dtype.itemsize <= 2

    @property
    def rows_loaded(self) -> bool:
        """Whether 16-bit left operands load by `ldmatrix`: only where each warp holds whole rows
        (`rows_whole`); fewer row warps read element-wise, as the relay path does."""
        return self._matrix_loads and self.rows_whole

    def _store_staged(self, value: jax.Array, buffer) -> None:
        cast = value.astype(buffer.dtype)
        if self.rows_whole or launches.interpreting():
            buffer[...] = cast
        else:
            # Its warps split a row's columns: an unoptimized store, which Mosaic GPU brackets with the
            # warpgroup's barriers as it does any store (its automatic barriers), one before it for the reads
            # it overwrites and one after it for the reads past it.
            plgpu.store(buffer, cast, optimized=False)

    def product_with[T](self, lhs, rhs, shape: tuple[int, int], between: Callable[[], T]
                        ) -> tuple[T, jax.Array]:
        out = self.products(self.zeros(shape), lhs, rhs)
        return between(), out

    def products(self, acc, lhs, rhs, *, accumulate: bool = True, support=None):
        interpret = launches.interpreting()  # the launch's (`launches.interpreting`)
        if not accumulate:
            # A fresh product: onto zeros, its accumulator a value (`read` is that value).
            acc = self.zeros(acc.shape)
        if support is not None and not interpret:
            return self._supported(acc, lhs, rhs, support)
        if jnp.dtype(lhs.dtype) != self.dtype:
            # Another dtype's operands (fp32's, as TF32) by that dtype's issuer: its fragments' layouts and
            # depths are its own, its accumulator's the same.
            return self.for_dtype(lhs.dtype).products(acc, lhs, rhs)
        if interpret:
            # mma has no rule on a CPU: its sum, in fp32, its rows past a bound kept as `_mma` keeps them.
            summed = acc + jnp.dot(lhs[...], rhs[...], preferred_element_type=jnp.float32)
            if self.row_bound is None or self.dtype == jnp.float32:
                return summed
            starts = lax.broadcasted_iota(jnp.int32, summed.shape, 0) // self.rows * self.rows
            issued = starts < self.row_bound
            return jnp.where(issued, summed, acc)
        acc, last = self._all_but_last(acc, *self._fragments(lhs, rhs)[:2])
        return self._mma(acc, *last)

    def _all_but_last(self, acc, loaded: Callable, starts: range) -> tuple[jax.Array, tuple]:
        """`acc` with every instruction's product of `starts` issued but the last, each instruction's
        fragments loaded (`loaded`) before the product ahead of it issues, and the last instruction's
        fragments."""
        fragments = loaded(starts[0])
        for following in starts[1:]:
            ahead = loaded(following)
            acc = self._mma(acc, *fragments)
            fragments = ahead
        return acc, fragments

    def _fragments(self, lhs, rhs, seeded: tuple | None = None) -> tuple[Callable, range, Callable]:
        """`products`' fragment reads of `lhs @ rhs`: `loaded(start)`, instruction `start`'s fragments, its
        loads issued the first time a slice is asked for; the instructions' starts; and `first()`, the left
        and right slices instruction 0 reads, read (`first_fragments`), those `seeded` gives already read
        (`products_across`). The left operand loads a whole swizzle tile of its depth at a time
        (`lhs_fragment_depth`), as a transposed left operand loaded in instruction slices took the A100's
        kernel gradient 7% slower (FS-DENSE-AB1); the right loads only the fragments the next instructions
        consume, as narrow as its staging allows (`_depth_granule`), which took its input gradient 12%
        faster. A register operand (an accumulator's value) is cast once and sliced. Each instruction's
        fragments load before the product ahead of it issues, CUTLASS's warp-level double buffering: every
        dense kernel gradient on an A100 took 12 to 14% less time, its outputs bitwise the same, the forward
        and input gradient unchanged (FS-C1-NARROW1-a100)."""
        instruction = self.fragment_depth(int(rhs.shape[0]))
        lhs_depth = max(_depth_granule(lhs, 1, instruction),
                        self.lhs_fragment_depth(int(lhs.shape[1]), swizzle=swizzling.swizzle_of(lhs)))
        rhs_depth = _depth_granule(rhs, 0, instruction)
        held = self._left(lhs) if isinstance(lhs, jax.Array) else None
        lefts: dict[int, jax.Array] = {}
        rights: dict[int, jax.Array] = {}
        if seeded is not None:
            left, rights[0] = seeded
            if left is not None:
                lefts[0] = left

        def loaded(start: int) -> tuple[jax.Array, jax.Array]:
            left_at, right_at = start - start % lhs_depth, start - start % rhs_depth
            if held is None and left_at not in lefts:
                lefts[left_at] = self._left(lhs.at[:, pl.ds(left_at, lhs_depth)])
            if right_at not in rights:
                rights[right_at] = plgpu.load(rhs.at[pl.ds(right_at, rhs_depth)], layout=self.rhs,
                                              optimized=self._matrix_loads)
            left = held[:, start:start + instruction] if held is not None else lefts[left_at][
                :, start - left_at:start - left_at + instruction]
            return left, rights[right_at][start - right_at:start - right_at + instruction]

        def first() -> tuple[jax.Array | None, jax.Array]:
            loaded(0)
            return lefts.get(0), rights[0]

        return loaded, range(0, rhs.shape[0], instruction), first

    def first_fragments(self, lhs, rhs):
        if jnp.dtype(lhs.dtype) != self.dtype:
            return self.for_dtype(lhs.dtype).first_fragments(lhs, rhs)
        return self._fragments(lhs, rhs)[2]()

    def products_across(self, acc, lhs, rhs, prefetched, boundary: Callable):
        if jnp.dtype(lhs.dtype) != self.dtype:
            return self.for_dtype(lhs.dtype).products_across(acc, lhs, rhs, prefetched, boundary)
        acc, last = self._all_but_last(acc, *self._fragments(lhs, rhs, seeded=prefetched)[:2])
        # The last product's operands are in registers: the next step's slots awaited and its first
        # fragments read now, under it.
        following = boundary()
        return self._mma(acc, *last), following

    def looped_products(self, acc, lhs, rhs, *, accumulate: bool = True):
        """`products`, its instruction slices issued in a loop of their own, each slice's operands loaded from
        shared memory within it: unrolled, ptxas interleaved a fused backward step's three later products
        (dV, dK, dQ, independent of each other) and held 170 words of their fragments at once, an A100's step
        at 64 x 64, width 64, under a softcap spilling 96 bytes; looped, 20 (one slice's), none spilled, the
        same 320 mma.sync instructions issued (row 5477). Interpreted, `products`."""
        if launches.interpreting():
            return self.products(acc, lhs, rhs, accumulate=accumulate)
        if jnp.dtype(lhs.dtype) != self.dtype:
            return self.for_dtype(lhs.dtype).looped_products(acc, lhs, rhs, accumulate=accumulate)
        if not accumulate:
            acc = self.zeros(acc.shape)
        step = self.fragment_depth(int(rhs.shape[0]))

        def issue(index, acc):
            start = index * step
            left = self._left(lhs.at[:, pl.ds(start, step)])
            right = plgpu.load(rhs.at[pl.ds(start, step)], layout=self.rhs,
                               optimized=self._matrix_loads)
            return self._mma(acc, left, right)

        return lax.fori_loop(0, int(rhs.shape[0]) // step, issue, acc, unroll=False)

    def looped_product(self, lhs, rhs, shape: tuple[int, int]) -> jax.Array:
        return self.looped_products(self.zeros(shape), lhs, rhs)

    def bounded(self, rows) -> Self:
        # Each product it makes takes the bound: `plgpu.mma`'s `rows` (the fork's from 0.12.0.1.post125),
        # each warp's 16-row blocks at or past it issuing none (row 7091).
        return dataclasses.replace(self, row_bound=rows)

    def _mma(self, acc, left, right):
        """`acc + left @ right` by one mma.sync product of its operands' fragments: fp32's as TF32 through
        Mosaic GPU's lane-level `mma` (`inline_mgpu`; the fork's TF32 from 0.12.0.1.post42), the dialect's op
        whose operand types jaxlib fixes naming no fp32, every row issued; any other dtype's by Pallas's
        `mma`, within its row bound (`bounded`)."""
        if self.dtype != jnp.float32:
            return plgpu.mma(acc, left, right, m_warps=self.m_warps, rows=self.row_bound)
        issued: Callable[..., jax.Array] = plgpu.inline_mgpu(
            arg_types=(self.layout, self.lhs, self.rhs),
            return_type=plgpu.ShapeDtypeStruct(acc.shape, jnp.dtype(jnp.float32), layout=self.layout))(
            _tf32_product)
        # The accumulator laid as the product takes it: a fresh one's splat passed in whole left a cast no
        # pass lowers (tf32-dump-sm80 at ca3dd0f0a, a splat to the accumulator's registers).
        return issued(plgpu.layout_cast(acc, self.layout), left, right)

    def registers_product(self, lhs: jax.Array, rhs) -> jax.Array:
        return self.products(self.zeros((lhs.shape[0], rhs.shape[1])), lhs, rhs)

    sums_in_windows = True  # its accumulator a value carried across a window's steps

    def summed_apart(self, running: jax.Array, lhs, rhs, scale: Callable | None = None
                     ) -> jax.Array:
        # A step's fresh accumulator starts from zeros tied to the running sum (an optimization barrier),
        # so no step's products issue before the last step's add. Its products are values, independent of
        # the running sum, so the compiler otherwise issued an unrolled walk's steps at once
        # (`_streamed`'s drain), a whole accumulator live each: at five stages three beside the running
        # sum, 616 bytes spilled at the A100 forward's 128 x 128 tile; a traced grid's walk, its drain not
        # unrolled, spilled none (NSDBG-5).
        running, fresh = lax.optimization_barrier((running, self.zeros(running.shape)))
        fresh = self.products(fresh, lhs, rhs)
        # `scale(start, columns, partial)` scales the partial itself (`Issuer.summed_apart`, row 13257).
        return running + (fresh if scale is None else scale(0, running.shape[1], fresh))

    def register_products(self, acc, parts, rhs):
        for formed in parts:
            acc = self.products(acc, mosaic.unscaled_part(formed), rhs)
        return acc

    @property
    def term_tile(self) -> tuple[int, int] | None:
        # Each instruction's tile of the accumulator, `m_warps * 16` rows by `atom_columns` (`MMALayouts`'
        # base tile, every warp's 16 x 8 share of it issued together): one skipped is skipped for every warp.
        return self.m_warps * 16, self.atom_columns

    def _supported(self, acc, lhs, rhs, support: Support):
        """`products` of `support`'s terms alone, `rhs` staged and `lhs` staged or in registers: each
        instruction issued where a term of the support falls in its tile and depth (`Support.tiles` on
        `term_tile`), each slice of either operand loaded where one reads it, none of a depth none takes. The
        doubling's merges hold their terms in its blocks (`recurrence_algebra.doubled`): whole, an RTX PRO
        6000's gated delta net solve issued 544 TF32 instructions and loaded and rounded 1408 operand words a
        thread, 2.8 times fla's solve (M-JS-275)."""
        if jnp.dtype(lhs.dtype) != self.dtype:
            return self.for_dtype(lhs.dtype)._supported(acc, lhs, rhs, support)
        depth = self.fragment_depth(int(rhs.shape[0]))
        rows, columns = self.m_warps * 16, self.atom_columns
        if not support.divides(rows, columns, depth):
            # Its instructions do not tile the product: whole.
            return self.products(acc, lhs, rhs)
        issued = support.tiles(rows, columns, depth)
        held = self._left(lhs) if isinstance(lhs, jax.Array) else None
        for k in range(issued.shape[2]):
            tiles = [(m, n) for m in range(issued.shape[0]) for n in range(issued.shape[1])
                     if issued[m, n, k]]
            if not tiles:
                continue
            read = sorted({n for _, n in tiles})
            rights = [plgpu.load(rhs.at[pl.ds(k * depth, depth), pl.ds(n * columns, columns)],
                                 layout=self.rhs, optimized=self._matrix_loads) for n in read]
            product_of: Callable[..., jax.Array] = plgpu.inline_mgpu(
                arg_types=(self.layout, self.lhs, *(self.rhs,) * len(rights)),
                return_type=plgpu.ShapeDtypeStruct(acc.shape, jnp.dtype(jnp.float32), layout=self.layout))(
                functools.partial(_tiled_product, rows=rows, columns=columns, depth=depth,
                                  tiles=tuple((m, n, read.index(n)) for m, n in tiles)))
            left = (held[:, k * depth:(k + 1) * depth] if held is not None
                    else self._left(lhs.at[:, pl.ds(k * depth, depth)]))
            acc = product_of(plgpu.layout_cast(acc, self.layout), left, *rights)
        return acc

    def rescaled_products(self, acc, scale, parts, rhs, interpret: bool):
        return self.register_products(scale(acc), parts, rhs)

    def added_into(self, acc: jax.Array, value: jax.Array, interpret: bool) -> jax.Array:
        """`Issuer.added_into` for a register accumulator: its sum with `value`, in the accumulator's
        layout."""
        del interpret
        return acc + self.as_tile(value)

    def settled(self, pending: int = 0) -> None:
        del pending


_MOSAIC_MMA = importlib.import_module("jax.experimental.mosaic.gpu.mma")
"""Mosaic GPU's mma.sync module (the package's `mma` its function): its single-tile product and TF32 rounding,
which `_tiled_product` issues a tile at a time, the fork's own (private there: a pin renaming them fails
tests/test_supported_products.py's compile)."""


def _tiled_product(ctx, acc, left, *rights, rows: int, columns: int, depth: int, tiles: tuple):
    """`acc` plus one depth's products of `left` `[M, depth]` with `rights` (each `[depth, columns]`) on
    `tiles` alone (`MmaIssuer._supported`): each `(m, n, i)` its accumulator's tile at row tile `m` and column
    tile `n` taking `left`'s row tile `m` times `rights[i]`, one instruction each (`mma`'s base tile), fp32
    operands rounded to TF32 once."""
    del ctx
    if left.mlir_dtype == ir.F32Type.get():
        left, rights = _MOSAIC_MMA._rounded_to_tf32(left), tuple(map(_MOSAIC_MMA._rounded_to_tf32, rights))
    acc = acc.copy()
    for m, n, index in tiles:
        at, span = slice(m * rows, (m + 1) * rows), slice(n * columns, (n + 1) * columns)
        acc[at, span] = _MOSAIC_MMA._mma_single_tile(acc[at, span], left[at, slice(0, depth)], rights[index])
    return acc


def _tf32_product(ctx, acc, left, right):
    """`acc + left @ right`, fp32's operands as TF32 by one mma.sync (`MmaIssuer._mma`), inlined: `2 m n k`
    TF32 FLOPs, its accumulator `[m, n]` and its left operand `[m, k]`."""
    del ctx
    return mgpu.mma(acc, left, right)


class CpAsync(mosaic.Family[Pipeline, Streams, jax.Array]):
    """The execution policy before Hopper (`mosaic.Transport`): one or two compute warpgroups a program and
    no producer, its loads cp.async copies into each walk's own slots (`_streamed`; two consumers run one
    walk over their thread axis 'wg', each copying its share of every block and meeting the other before a
    slot is read and before it is refilled: FlashAttention 2's eight warps over one shared tile), its
    accumulators in the walk's carry (a register home), its output stored through registers. A
    family whose issuer half an issuer composed with it gives (`Mma`)."""

    walks_scoped = True  # `_streamed` allocates each walk's slots in its own scope

    stores_narrowed = True  # a narrowed output staged whole at its stored shape, at its caller's place

    walks_streamed = True  # consumers of one collective pipeline walk every streamed step (`walk_own`)

    @property
    def shared_consumers(self) -> int:
        # Two consumer warpgroups run one collective pipeline at most, each copying its share of every block
        # (`_streamed`'s thread axis), as FlashAttention 2's eight warps share a step's queries over two
        # key halves: a launch that names two (the A100 backward's dK/dV, its dQ adds halved, row 5272), not
        # a derivation's own pick (`consumers`).
        return 2

    def compiled_with(self, *, approx_math: bool) -> plgpu.CompilerParams:
        # Its kernels' lane-level lowering (`kernel`'s), Mosaic GPU's default.
        return plgpu.CompilerParams(approx_math=approx_math)
    prefetches_fragments = True  # `_streamed`'s `prefetch`: a step's first fragments under the last's

    # Its programs fix a tile up across programs through global-memory atomics (`Family.fixed_up`), which
    # sm_70 on issues: a balanced tail's and a decode's split partials merge in the launch.
    fixes_up = True

    bounded_loads = False

    # A step loading ahead of the one consumed leaves cp.async's latency unhidden: two steps 64 deep stalled
    # on barriers 1.1-1.5 an issue against XLA's 0.3-0.7 (FS-NCU-DW2-a100), and 32-deep steps three
    # deep took dense products 4-8% less time than 64 deep two (FS-DENSE-STEPS1-a100).
    loads_ahead = 2

    stores_through_registers = True

    copies_vectors = False  # cp.async spreads a copy over the warpgroup's threads, 128 elements at least
    gathers_rows = True  # its one warpgroup's cp.async copies gather a step's rows (`_Gathered`)
    # `_streamed` would allocate a folded once block a slot per stage in the walk's scope, where its
    # planned once allocation (`lifetimes`) holds one: no once stream walks here until the plan counts it.
    walks_once = False
    gathers_alone = False  # its walk gathers every stream's rows or none (`streams`)
    gathers_interpreted = True

    copies_in_pieces = False  # its pipeline loads a step's block whole, or gathers its rows


    adds_through_shared = False  # `added`'s atomic adds go from registers (`red.global`), its `smem` unread
    stages_vectors = False

    @property
    @abc.abstractmethod
    def lhs(self) -> plgpu_core.SomeLayout:
        """The layout a tile loaded through registers takes (`loaded`): its issuer's left operand's."""

    def registers(self, consumers: int, programs: int = 1, producer: int | None = None,
                  joined: tuple = ()) -> int:
        # Each role at its thread's share of the multiprocessor's 64K registers: two programs' 256 threads
        # hold 255 each.
        _most(self, consumers)
        handoffs.runs_alone(self.name, joined)
        return resources.entry_registers(consumers, programs)

    def kernel(self, entry, *, out_type, grid: tuple, grid_names: tuple, consumers: int,
               interpret: bool, approx_math: bool = True, name: str | None = None,
               cluster: int = 1, cost: pl.CostEstimate | Stated | None = None,
               model_flops: int | None = None,
               reducing_warpgroups: int = 0, resident: int = 1, scalars: int = 0,
               prefetched: Callable | None = None, registers: int | None = None, reduces: bool = True,
               producer: int | None = None, shared: int | None = None, tmem: int = 0,
               moves: bool | None = None, joined: tuple = ()) -> Callable:
        handoffs.runs_alone(self.name, joined)
        # Its warpgroups its consumers (two a thread axis 'wg'), no producer beside them, and no cluster;
        # its kernels' default parameters reserve the scratch an implicit reduction takes
        # (`reduction_scratch`), where it `reduces`. Its programs read a table from global memory as any
        # operand.
        del reducing_warpgroups, scalars, producer, tmem, moves  # consumers alone: no roles to move registers
        _most(self, consumers, clustered=cluster > 1)
        threads = {} if consumers == 1 else {"num_threads": consumers, "thread_name": "wg"}
        built = launches.kernel(bodies.dependent(entry, interpret, prefetched=prefetched),
                                name=name or self.name,
                                out_type=outputs.structs(out_type), grid=grid, grid_names=grid_names,
                                **threads,
                                compiler_params=dataclasses.replace(
                                    self.compiled_with(approx_math=approx_math),
                                    reduction_scratch_bytes=self.reduction_scratch if reduces else 0,
                                    min_blocks_per_sm=resident, metadata=report.launch_metadata()),
                                interpret=bodies.interpreted(interpret),
                                kernel_name=report.named(
                                    name, model=(None if registers is None
                                                 else resources.register_model(
                                                     self, consumers, registers, shared=shared,
                                                     target=generation.gpu_target()))),
                                cost_estimate=emission.built_with(cost))
        return mosaic.checked(emission.costed(built, cost), name or self.name, resident, model_flops,
                              out_type=out_type)

    def program_scoped(self, body, *allocations, streams):
        # Collective where its pipelines (`streams`, one or a tree of them) run on two consumers; a tree of
        # one-consumer and two-consumer pipelines is refused here, before its scope opens.
        return swizzling.run_scoped(lambda *refs: body(*refs, streams=streams), *allocations,
                                    **_collective(_consumers(streams)))

    def roles(self, consumers: int, producer: int | None = None) -> handoffs.Schedule:
        _most(self, consumers)
        if consumers == 1:
            return handoffs.Schedule((handoffs.Role("consumer"),), axis=None)
        return handoffs.Schedule((handoffs.Role("consumer", consumers),))

    def streams(self, blocks: dict, stages: int, consumers: int, multicast: tuple[str, ...] = (),
                swizzles: dict[str, int] | None = None, copies: dict[str, int] | None = None,
                gathered: tuple[str, ...] = (), joined: tuple[str, ...] = ()):
        del joined  # a pair a stream (`Family.streams`)
        _most(self, consumers, clustered=bool(multicast))
        if gathered and set(gathered) != set(blocks):
            raise NotImplementedError("an A100's walk gathers every stream's rows or none")
        if gathered and consumers > 1:
            raise NotImplementedError("an A100's gathered walk copies by one consumer's rows")
        if any(count > 1 for count in (copies or {}).values()):
            raise ValueError("an A100's pipeline loads a step's block in one copy, not in pieces")
        return Pipeline(blocks, stages, dict(swizzles or {}), tuple(gathered), consumers)


    def lifetimes(self, streams: Pipeline, steps: int | None = None) -> tuple[tuple, tuple]:
        # A walk's shared memory in its own scope, the one plan its walk allocates (`_ring`); none for a walk
        # of no steps, which the walk skips.
        if steps == 0:
            return (), ((),)
        return (), (_ring(streams, steps),)

    def opened(self, allocated, views: dict | None = None):
        if views is None:
            raise ValueError("a pipelined family's streams take their views where they open (its walk loads)")
        return Streams(allocated, views)

    def stored_swizzle(self, shape: tuple[int, int], dtype) -> int:
        """The swizzle an output tile of `shape` in `dtype` stages at (`output_staging`): plain where its
        issuer states none (`Mma.stored_swizzle`)."""
        del shape, dtype
        return swizzling.UNSWIZZLED

    def stream(self, streams, views, lo, hi, *, again=None) -> None:
        # No producer role: a decided walk's consumers feed their second pass themselves.
        del streams, views, lo, hi, again

    def persist(self, streams, tiles: tuple[int, ...], schedule: Callable, issue: Callable, finish: Callable,
                *, consumers: int, axis: str, outputs: tuple, interpret: bool, in_flight: int = 0,
                unrolled: bool = False, accumulators: tuple = (), producer: int | None = None,
                pingpong: bool = False, chunks: tuple[int, int] | None = None) -> None:
        del producer  # no producer role of its own
        self.pingponged(pingpong)
        _most(self, consumers)
        # Each tile's pipeline fills and drains before its output copy. Synchronize at that lifetime
        # boundary so the output's shared memory can reuse the pipeline's slots. Two consumers stage their
        # rows side by side in one collective scope, opened once a tile where both finish it (`finished`):
        # a store only one reaches (a balanced tail's, the last program to arrive for each consumer storing,
        # two programs' winners apart) opens none of its own, which would wait for the other forever.
        del in_flight, unrolled, accumulators  # its walk carries its accumulators
        index = 0 if consumers == 1 else lax.axis_index("wg")
        staging = self.output_staging(outputs, consumers, chunks)
        opened: list = []

        def store(output, value, ref, place, within=None):
            del within  # each tile staged whole (`stores_chunks` off): a hook reads it out whole
            # A narrowed output (`stores.Output.narrowed`) is staged whole at its stored shape, its `place`
            # its own columns (its caller's), as a chunk-staging family's stores take it (`stores.staged`).
            # Interpreted pipelines use barrier-tracked copies, not cp.async groups.
            if not interpret:
                plgpu.wait_gmem_to_smem(0)

            def copied(smem):
                plgpu.store(smem, value.astype(smem.dtype), optimized=False)
                swizzle = swizzling.UNSWIZZLED if len(smem.shape) != 2 else self.stored_swizzle(smem.shape,
                                                                                                   smem.dtype)
                if interpret or swizzle == swizzling.UNSWIZZLED:
                    # The row-copy ladder needs strided layouts, not optimized-only transfers.
                    stores.store_rows(smem, ref, *place, copy=swizzling.copied,
                                      widest=resources.copied_rows(smem.shape[1], smem.dtype))
                else:
                    # A swizzled tile's rows out by its threads, whole tiles read (`output_staging`).
                    stores.swizzled_rows_out(smem, ref, *place[:4], swizzle=swizzle)

            if consumers == 1:
                swizzling.run_scoped(copied, staging[output])
            else:
                copied(opened[-1][output].at[index])
            if not interpret:
                plgpu.wait_gmem_to_smem(0)

        def finished(index, tile, accumulators, store, *, active):
            if consumers == 1:
                finish(index, tile, accumulators, store, active=active)
                return

            def staged(*refs):
                opened.append(refs)
                finish(index, tile, accumulators, store, active=active)
                opened.pop()

            swizzling.run_scoped(staged, *staging, collective_axes="wg")

        def walking(counter, active, steps, views):
            del counter, active

            def walk(step, start, shapes, *, initial=None, prefetch=None):
                return self.walk(self.opened(streams, views), ((step, 0, steps),), start, shapes,
                                 interpret, initial=initial, prefetch=prefetch)

            return walk

        persistent.consumed(self, tiles, schedule, issue, finished, store, index, walking, axis=axis)

    def walk[C](self, streams, segments, start: C, shapes: tuple, interpret: bool, *,
                first: jax.Array | int = 0, in_flight: int = 0, initial: tuple | None = None,
                unrolled: bool = False, products: Sequence[mosaic.Product] = (),
                prefetch: Callable | None = None) -> tuple[C, tuple[jax.Array, ...]]:
        # Its products are synchronous: none stays in flight past its step, and `_streamed` gives
        # each stage back as its step ends, so no step holds one past it, nor reads one ahead. With
        # `prefetch(slots)` (`prefetches_fragments`), `_streamed` reads a step's first fragments under the
        # step before's last product: each step is given them and the boundary that reads the next's
        # (`step(..., prefetched=, boundary=)`, returning its carry and the next's fragments), where it keeps
        # two slots or more (its prefetch's own refill) and gathers no rows (`_Gathered`'s own copies);
        # elsewhere each step is called as without it.
        if mosaic.ahead_streams(products):
            raise NotImplementedError(f"{self.name}'s pipeline reads no stream ahead")
        del first, in_flight, unrolled
        segments = mosaic.stepped(segments)
        blocks = streams.pipeline.blocks
        names = tuple(blocks)
        carry = (start, self.started(shapes, initial))
        gathered = streams.pipeline.gathered
        prefetching = prefetch is not None and streams.pipeline.stages > 1 and not gathered

        def prefetched_of(slots: dict):
            assert prefetch is not None
            return prefetch(slots_of(slots))

        def slots_of(slots: dict) -> rings.WholeSlots:
            return rings.WholeSlots(slots)

        # Each pipeline's step, its trips' steps (a trip's place in the walk), and its trips.
        walks = [(step, functools.partial(operator.add, lo), hi - lo) for step, lo, hi in segments]
        if streams.pipeline.consumers > 1:
            # Consumers running one pipeline together meet at each of its trips and fill each slot together,
            # so they load one sequence of steps: the range `walk_own` agrees (`Streams.agreed`), each
            # consumer's step at each of its trips that of its own segment holding it, none where none does
            # (a gap, a step past its own segments). Segments ordered and apart; a trip reads no fragments
            # ahead of its own where its consumer's segments are not the agreed range itself (its boundary
            # would be called under its segment's choice). Segments every consumer's alike (`common`) each
            # run a pipeline of their own, as one consumer's do, keeping their prefetch.
            if streams.agreed is None:
                if len(segments) > 1:
                    raise NotImplementedError(f"two {self.name} consumers' walk of several segments "
                                              "loads the steps `walk_own` agrees")
            else:
                lo, hi = streams.agreed
                whole = len(segments) == 1 and segments[0][1] is lo and segments[0][2] is hi
                if not whole and not streams.common:
                    walks = [(_dispatched(segments), functools.partial(operator.add, lo), hi - lo)]
                    prefetching = False
        for step, placed, trips in walks:
            if isinstance(trips, int) and trips <= 0:
                # No steps: the carry and its zero accumulators as they start (no slot to fill).
                continue

            def body(trip, slots, carried, *ahead, step=step, placed=placed):
                carried, accumulators = carried
                if prefetching:
                    prefetched, boundary = ahead
                    return step(placed(trip), slots_of(slots), carried, accumulators, prefetched=prefetched,
                                boundary=boundary)
                return step(placed(trip), slots_of(slots), carried, accumulators)

            # A walk that prefetches reads a step's fragments only once a copy filled its slot (row 12695),
            # so a segment whose trips are known only as it runs (a common segment's) runs its walk whatever
            # their count: an empty one prefetches nothing (`_streamed`).
            copier = (_Gathered(streams.views, placed) if gathered else _Contiguous(
                {name: (streams.views[name][0], functools.partial(_at, streams.views[name][1], placed))
                 for name in names}))
            carry = _streamed(streams.pipeline, copier, trips, body, carry, interpret=interpret,
                              prefetch=prefetched_of if prefetching else None)
        return carry

    def passed(self, streams, lo, hi, *, first: jax.Array | int = 0, between=None) -> None:
        del streams, lo, hi, first, between

    def walk_own[R](self, streams, views: dict, own: tuple, streamed: tuple, wg,
                    walk: Callable[..., R], *, between=None, counted=0, common: bool = False) -> R:
        # Two consumers run one collective walk (`_streamed`), whose meets pair their steps: each
        # walks every streamed step, its step masking those outside its own range, as a consumer-fed
        # family's do (`ConsumerFed.walk_own`; a causal attention program's two consumers reach different
        # diagonals, and walks of their own ranges would meet unpaired). One consumer walks its own.
        if streams.pipeline.consumers == 1:
            return super().walk_own(streams, views, own, streamed, wg, walk, between=between, counted=counted)
        del views, own, wg, between, counted
        lo, hi = streamed
        return walk(dataclasses.replace(streams, agreed=(lo, hi), common=common), lo, hi, lo)

    def loaded(self, copies, *, issued: jax.Array | bool = True, axes: tuple | None = None) -> None:
        # Through registers (no TMA before Hopper), in a layout of the tile's rank: a side vector's the
        # accumulator's vector layout of the axis it follows (its rows', `row`, read as `s[:, None]`; its
        # columns', `column`, otherwise), a tile's the left operand's. Each caller its own: a shared `ref`
        # loaded by one role would need a barrier past its stores that no caller here waits on.
        if issued is not True:
            raise NotImplementedError(f"{self.name} loads no shared copy for its roles")
        for index, (view, ref, _) in enumerate(copies):
            follows = axes[index] if axes is not None else ('columns',)
            layout = self.lhs if len(ref.shape) > 1 else self.row if follows == ('rows',) else self.column
            ref[...] = plgpu.load(view, layout=layout, optimized=False)

    def stored(self, parts, views, *, smem, wait: bool = True) -> None:
        # Through `smem` where given, copied out in row blocks as `persist`'s stores are (`store_rows`): an
        # MMA-layout tile stored from registers writes partial 32-byte sectors, which JAX 0.12's global
        # transfers refuse, and its row-sliced copies lay out whole ones. Its copies go through registers,
        # done when they return: nothing stays in flight (`wait` asks nothing of it).
        # A vector (a row's per-row statistics, attention's `[bq]` and `[bk]`) is stored whole from its
        # registers: reloaded from shared memory it takes no layout Mosaic GPU's inference finds (a 64-entry
        # one has no strided layout, which wants a multiple of 128), where its registers keep their own.
        del wait
        for part, view in zip(parts, views, strict=True):
            if smem is None or len(smem.shape) == 1:
                if isinstance(part, mosaic.StagedWriter):
                    part.write(None)
                elif len(view.shape) == 1 and not launches.interpreting():
                    plgpu.store(view, part.astype(view.dtype), optimized=False)
                else:
                    view[...] = part.astype(view.dtype)
                continue
            if isinstance(part, mosaic.StagedWriter):
                part.write(smem)
            else:
                self.stage_output(part, smem)
            rows = smem.shape[0]
            # A landing wider than its output rounds each row block as it copies it (`swizzling.narrowed`).
            wider = jnp.dtype(smem.dtype).itemsize > jnp.dtype(view.dtype).itemsize
            stores.store_rows(smem, view, 0, 0, jnp.int32(rows), 0, rows,
                              copy=swizzling.narrowed if wider else swizzling.coalesced,
                              widest=resources.copied_rows(smem.shape[1], smem.dtype))

    def stage_output(self, value: jax.Array, smem) -> None:
        if launches.interpreting():
            smem[...] = value.astype(smem.dtype)
        else:
            plgpu.store(smem, value.astype(smem.dtype), optimized=False)

    def added(self, value: jax.Array, view, interpret: bool, *, smem, pending: int = 0) -> None:
        # Element-wise atomic adds into global memory (`red.global`; no TMA reduction before Hopper), so
        # every other role's and program's adds into the view stay whole, no buffer between them.
        del smem, pending
        if interpret:
            view[...] = view[...] + value
        else:
            plgpu.atomic_add(view, value, optimized=False)



@dataclasses.dataclass(frozen=True)
class Mma(MmaIssuer, CpAsync):
    """mma.sync's bundle: its issuer and register home (`MmaIssuer`) on its walk's cp.async policy
    (`CpAsync`)."""

    name = "mma.sync"

    # One consumer warpgroup a program where a derivation picks its own (`consumers`, what every derivation
    # reads): two hung on an A100 at the attention forward (FS-C2HANG-a100, M-ET-15) and spilled where the
    # forward derived them. A launch asks for two by name (`shared_consumers`).
    consumers = 1

    def output_staging(self, outputs: tuple, consumers: int, chunks: tuple[int, int] | None = None) -> list:
        plain = mosaic.refuse_staging(self, chunks, outputs)
        # Each output tile staged at the swizzle its accumulator's stores pass the banks fewest times at
        # (`stored_swizzle`), copied out a 16-byte piece a thread over whole tiles (`persist`'s store,
        # `stores.swizzled_rows_out`); two consumers' rows of it side by side (`persist`), one output a tile:
        # their staging opens together once a tile, each output's beside the others', which a launch's plan
        # charges one output at a time (`resources.scratch_bytes`' tiles after its walks).
        if consumers > 1 and len(plain) > 1:
            raise NotImplementedError(f"two {self.name} consumers stage one output a tile, not {len(plain)}")
        side = () if consumers == 1 else (consumers,)
        return [swizzling.staged((*side, *shape), dtype, self.stored_swizzle(shape, dtype)) if len(shape) == 2
                else plgpu.SMEM((*side, *shape), dtype) for shape, dtype in plain]

    def stored_swizzle(self, shape: tuple[int, int], dtype) -> int:
        """The swizzle an output tile of `shape` in `dtype` stages at: of the swizzles its rows fill whole
        (`swizzling.SWIZZLE_BYTES` down to the narrowest) and plain, the one its accumulator's stores pass
        shared memory's banks fewest times at (`swizzling.wavefronts`, by the staging's address arithmetic),
        the widest on a tie. A warp stores mma.sync's accumulator fragment an 8-row slab by 8 columns at a
        time, lane `l` two adjacent elements of row `l // 4` at column `2 (l % 4)`: plain, 64 16-bit columns
        put a slab's 8 rows in the same banks, 8 passes a store, where a 128-byte swizzle gives each row's
        piece its own banks, one pass (row 5271: the A100's dense kernel gradient's 4x cuBLAS's bank
        conflicts, FS-DW-NCU2)."""
        rows, columns = shape
        itemsize = jnp.dtype(dtype).itemsize
        stagings = [swizzle for swizzle in (swizzling.SWIZZLE_BYTES, 64, swizzling.NARROWEST_SWIZZLE_BYTES)
                    if (columns * itemsize) % swizzle == 0 and rows % swizzling.TILE_ROWS == 0]
        stagings.append(swizzling.UNSWIZZLED)

        def passes(swizzle: int) -> int:
            return sum(swizzling.wavefronts(
                [swizzling.staged_address(slab + lane // 4, (block + 2 * (lane % 4)) * itemsize,
                                          columns * itemsize, swizzle) for lane in range(32)], 2 * itemsize)
                for slab in range(0, rows, swizzling.TILE_ROWS) for block in range(0, columns, 8))

        return min(stagings, key=lambda swizzle: (passes(swizzle), -swizzle))


def _dispatched(segments: tuple) -> Callable:
    """One step taking each of `segments`' steps `(step, lo, hi)`, ordered and apart, at its own place: the
    step of the segment whose range holds its index, nothing where none does. Segments whose bounds are
    known are refused where they overlap or fall out of order."""
    known = [(lo, hi) for _, lo, hi in segments if isinstance(lo, int) and isinstance(hi, int)]
    nonempty = [(lo, hi) for lo, hi in known if hi > lo]
    overlapping = any(nxt_lo < hi for (_, hi), (nxt_lo, _) in itertools.pairwise(nonempty))
    if len(known) == len(segments) and overlapping:
        raise ValueError(f"a collective walk's segments lie ordered and apart, not {known}")

    def step(index, slots, carried, accumulators):
        # The first segment holding the step (each a step at most), 0 where none does.
        which, unheld = jnp.int32(0), jnp.ones((), jnp.bool_)
        for at, (_, lo, hi) in enumerate(segments):
            inside = (index >= lo) & (index < hi) & unheld
            which, unheld = jnp.where(inside, at + 1, which), unheld & ~inside
        branches = [lambda carried, accumulators: (carried, accumulators),
                    *(functools.partial(each, index, slots) for each, _, _ in segments)]
        return lax.switch(which, branches, carried, accumulators)

    return step


def _most(family: "CpAsync", consumers: int, *, clustered: bool = False) -> None:
    """A cp.async family's programs take one compute warpgroup up to its collective pipeline's
    (`shared_consumers`), no producer, and no cluster (`clustered`: cp.async loads its own program's shared
    memory alone)."""
    if consumers not in range(1, family.shared_consumers + 1) or clustered:
        raise NotImplementedError(f"an {family.name} program takes one to {family.shared_consumers} consumer "
                                  f"warpgroups, not {consumers}, and no cluster")


def _consumers(streams) -> int:
    """The consumers every one of `streams`' pipelines (one or a tree of them) runs on, one where none; a
    tree of pipelines on different counts refused (its scope is collective over all of them or none)."""
    pipelines = jax.tree.leaves(streams, is_leaf=lambda leaf: isinstance(leaf, Pipeline))
    counts = {pipeline.consumers for pipeline in pipelines if isinstance(pipeline, Pipeline)}
    if len(counts) > 1:
        raise NotImplementedError(f"an mma.sync program's pipelines run on one count of consumers, not "
                                  f"{sorted(counts)}")
    return counts.pop() if counts else 1


def _collective(consumers: int) -> dict:
    """A scope's allocations collective over the consumers' thread axis where two run (a threaded kernel's
    scoped allocations are its warpgroups' together)."""
    return {} if consumers == 1 else {"collective_axes": "wg"}

