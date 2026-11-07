"""The templates' Pallas Mosaic GPU toolkit (design §6), every instruction family's: the launch,
warpgroup roles, barriers, shared-memory staging, operand streams and stores every Mosaic mainloop builds on.
The math is each template's algebra, written once; what an instruction family adds is its issue and its
layout facts alone (`Family`: mosaic_sm90's wgmma; mma.sync's and tcgen05's to come). Each template keeps
its own mainloop, product dependencies and launch DAG; only these primitives are shared.

A launch takes each warpgroup's role (`roles`: a producer issuing the loads beside consumers computing where
the family's roles split the register file, `Family.consumers`), persistent over the multiprocessors, a pair
of them a cluster where they share one operand's blocks (multicast: `ring`'s, `Family.streams`'). The
consumers meet (`meet`) where one consumer's writes to shared memory are the other's operand, and a tile's
rows reach global memory through `store_rows`, whose stores have static extents however many rows a tile
holds. What a program asks of a multiprocessor, and the derivation's shared steps, are `resources`'.

A mainloop that drives its own loads (the GEMM's kernel gradient, its stages streaming across tiles)
streams each operand through a `Ring` whose counter runs across tiles, and takes each warpgroup's role
(`roles`).
"""

import abc
import contextvars
import dataclasses
import functools
import math
import operator
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, Self

import jax
from jax import lax, numpy as jnp
from jax._src.pallas import core as pallas_core
from jax._src.pallas.mosaic_gpu import core as plgpu_core
from jax._src.state import types as state_types
from jax.experimental import pallas as pl
from jax.experimental.mosaic.gpu import fragmented_array as mgpu_fragments, utils as mgpu_utils
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.experimental.xla_metadata import set_xla_metadata
from jax.extend.mlir import ir
from jax.extend.mlir.dialects import arith

from dew.nn.kernels import generation, launches
from dew.nn.kernels.templates import product, report, resources
from dew.telemetry.instrumentation import MODEL_FLOPS

if TYPE_CHECKING:
    from dew.nn.kernels.templates.fed import Feed

Ref = state_types.AbstractRef | state_types.TransformedRef
Operand = Ref | jax.Array
"""A product's operand: a tile in shared memory (a ref) or in registers (an array)."""


def compiler_params(*, approx_math: bool = False, reduction_scratch: int = 0,
                    resident: int = 1) -> plgpu.CompilerParams:
    """A mainloop's compiler parameters: Mosaic GPU's warpgroup lowering, whose arrays are a warpgroup's,
    and with `approx_math` the hardware's approximate exponential and reciprocal (ex2.approx, as Triton's
    exp2 and FlashAttention's are). A block's warpgroups reduce at once, so each takes its own scratch: a
    consumer's planned reductions its share of the kernel's allocations (`Family.column_sums`), the
    implicit ones its arithmetic forms (a unit row a hook slices) an equal share of `reduction_scratch`
    bytes, which Mosaic GPU splits between the block's warpgroups; with none, such a reduction fails to
    lower rather than share one. `resident`, the programs a multiprocessor holds at once, caps each thread's
    registers so that many fit its register file (CUDA's `__launch_bounds__` minimum blocks, Mosaic GPU's
    `min_blocks_per_sm`, nvvm.minctasm); at one, ptxas takes up to 255."""
    return plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Warpgroup, approx_math=approx_math,
                                reduction_scratch_bytes=reduction_scratch, min_blocks_per_sm=resident)


CLUSTER = "cluster"
"""The axis a launch's cluster of programs runs along (`Family.kernel`'s `cluster`)."""


def cluster_index(cluster: int) -> jax.Array | int:
    """This program's index among its cluster's `cluster` programs (`CLUSTER`): 0 alone."""
    return lax.axis_index(CLUSTER) if cluster > 1 else 0


STAGED_BYTES = 128
"""The widest row, in bytes, a consumer stages an output in at a time: a swizzle's row. A consumer's 256
bf16 columns written to shared memory as one 512-byte row conflicted on its banks 195 times as often as in
two 256-byte halves (ncu, Qwen3's down projection forward: 18.3M conflicts against 94K, the kernel 16%
slower); 128-byte chunks, two of them buffered (`output_staging`), stage a 256-column bf16 tile in half its
width, which leaves the ring a stage more (unmeasured for conflicts at 128 bytes)."""

OUTPUT_BUFFERS = 2
"""The chunk buffers a consumer stages an output tile through: one written while the last one's store
reads it."""


def staged_columns(columns: int, dtypes) -> int:
    """The columns of each chunk a consumer stages a `columns`-wide output tile in: the most that divide the
    tile within `STAGED_BYTES` of the widest output (a 192-column tile's bf16 output in chunks of 64)."""
    return math.gcd(columns, STAGED_BYTES // max(jnp.dtype(dtype).itemsize for dtype in dtypes))


def output_staging(columns: int, dtypes) -> tuple[int, int]:
    """The columns of each chunk a consumer stages a `columns`-wide output tile in (`staged_columns`) and
    the chunk buffers it holds (`OUTPUT_BUFFERS`, at most its chunks): the tile's output staged in two
    chunks' shared memory rather than its whole width, CUTLASS's sub-tiled epilogue, which leaves the
    mainloop's ring the rest of the block (its StageCountAutoCarveout)."""
    chunk = staged_columns(columns, dtypes)
    return chunk, min(columns // chunk, OUTPUT_BUFFERS)


def interpreted(interpret) -> plgpu.InterpretGPUParams | None:
    """A launch's interpreter: `interpret`'s own parameters where it gives them (a read past an operand
    poisoned, `grouped_matmul.INTERPRETED`), the default where it is True, none where False."""
    if isinstance(interpret, plgpu.InterpretGPUParams):
        return interpret
    return plgpu.InterpretGPUParams() if interpret else None


def kernel(entry, *, out_type, grid: tuple, grid_names: tuple, warpgroups: int,
           interpret, family: str, name: str | None = None, scratch: tuple = (),
           approx_math: bool = False, cluster: int = 1, cost: pl.CostEstimate | None = None,
           model_flops: int | None = None, reduction_scratch: int = 0, resident: int = 1) -> Callable:
    """Build a per-shard launch with the same reporting and interpreter semantics for every family, each call
    checked against the mesh it runs in (`launches.kernel`), named as the launch tracing it reports `name`
    (`report.named`), a launch lowered once for every call of its facts (`tuning.lowered_once`), its `cost`
    the hardware's work and `model_flops` the model's share of it (`checked`)."""
    clustered = {'cluster': (cluster,), 'cluster_names': (CLUSTER,)} if cluster > 1 else {}
    built = launches.kernel(dependent(entry, interpret), name=name or family, out_type=out_type, grid=grid,
                            grid_names=grid_names, num_threads=warpgroups, thread_name="wg",
                            scratch_types=scratch,
                            compiler_params=compiler_params(approx_math=approx_math, resident=resident,
                                                            reduction_scratch=reduction_scratch),
                            interpret=interpreted(interpret),
                            kernel_name=report.named(name), cost_estimate=cost, **clustered)

    return checked(built, name or family, resident, model_flops)


def dependent(entry: Callable, interpret) -> Callable:
    """`entry` as a dependent launch (programmatic dependent launch, PTX's `griddepcontrol`): it waits for the
    launch before it in its stream to finish and its writes to land (`wait`) before any memory access of its
    own, so it is ordered against any launch as without it, and once its own work is done it lets the next
    such launch start (`launch_dependents`), whose launch and setup then overlap this one's last programs'
    tails. Triggered at the work's end, not its start: a trigger at entry let a dependent's programs take the
    multiprocessors this launch's later waves needed, each waiting there on this launch to finish (an H100
    SXM's recurrence launches, wall time per call: Qwen3.5-27B's GDN forward and Nemotron-H 30B's SSD
    training 1.0 and 1.1% above no dependent launch at entry, within 0.5% of it at the end, pdl-sxm at
    b2b872c91; CUTLASS triggers after its mainloop for the same reason). Mosaic marks the launch
    (`mosaic_gpu.uses_pdl`) and the runtime launches it with programmatic stream serialization. Interpreted,
    or on a GPU without grid dependency control (`generation.Architecture.dependent_launch`, sm_90 on),
    `entry` as it is."""
    if interpreted(interpret) or not generation.gpu_target().architecture.dependent_launch:
        return entry

    def body(*refs):
        plgpu.griddepcontrol_wait()
        entry(*refs)
        plgpu.griddepcontrol_launch_dependents()

    return body


_RESIDENT: contextvars.ContextVar[int] = contextvars.ContextVar("resident", default=1)
"""The programs a multiprocessor holds at once of the launch being traced (`kernel`'s `resident`): its roles
raise their registers to that residency's share (`roles`)."""


def checked(built: Callable, kernel: str, resident: int = 1, model_flops: int | None = None) -> Callable:
    """The launch `built` (`launches.kernel`, which holds each call per shard), each call traced at its
    residency (`_RESIDENT`) under the name `kernel`: a launch built once for every call of its facts checks
    each call's own; and
    refused at its trace where two of the refs it adds into in place share a shape and a dtype, which
    `stacked_zeros` makes one ref (row 3180). Its call declares `model_flops`, the model's share of the work
    its cost states (the rest its own recomputation), where given, as its custom call's frontend attribute
    (`instrumentation.MODEL_FLOPS`), which Dew's FLOP count reads."""
    def launch(*operands):
        if model_flops is not None:
            with set_xla_metadata(**{MODEL_FLOPS: str(model_flops)}):
                return checked_launch(*operands)
        return checked_launch(*operands)

    def checked_launch(*operands):
        refs = [jax.typeof(operand) for operand in operands if _is_ref(operand)]
        kinds = [(ref.shape, ref.dtype) for ref in refs]
        if len(set(kinds)) < len(kinds):
            raise ValueError(f"{kernel} adds into two refs of one shape and dtype ({kinds}): XLA may take "
                             "their zeros for one value and pass one buffer for both (row 3180); stack them "
                             "into one ref (`stacked_zeros`)")
        token = _RESIDENT.set(resident)
        try:
            return built(*operands)
        finally:
            _RESIDENT.reset(token)

    return launch


def _is_ref(operand) -> bool:
    """Whether a launch's operand is a ref it adds into in place (`jax.new_ref`'s)."""
    return isinstance(jax.typeof(operand), state_types.AbstractRef)


def stacked_groups(shapes: Sequence[tuple[int, ...]]) -> tuple[tuple[int, ...], ...]:
    """The refs a launch adds into in place, one for each of `shapes`, grouped by shape, each group's indices
    in order: a group is one ref, a leading axis of them (`stacked_zeros`)."""
    grouped: dict[tuple[int, ...], list[int]] = {}
    for index, shape in enumerate(shapes):
        grouped.setdefault(tuple(shape), []).append(index)
    return tuple(tuple(group) for group in grouped.values())


def stacked_zeros(shapes: Sequence[tuple[int, ...]], groups: tuple[tuple[int, ...], ...]) -> list:
    """Zeroed fp32 refs a launch adds into in place, one for each of `shapes`, a ref a group of one shape
    (`stacked_groups`), a leading axis of them: two refs of equal zeros are one value to XLA, which a
    dynamic-slice fusion wrapping the launch passes once for both aliased operands, so both were one buffer
    (row 3180; `checked` refuses two such refs)."""
    return [jax.new_ref(jnp.zeros((len(group), *shapes[group[0]]), jnp.float32)) for group in groups]


def stacked_views(refs, groups: tuple[tuple[int, ...], ...]) -> list:
    """Each stacked ref's member (`stacked_zeros`) as the launch's program reads it, in its shapes' order:
    its place along its group's ref's leading axis."""
    views: list = [None] * sum(map(len, groups))
    for ref, group in zip(refs, groups, strict=True):
        for place, index in enumerate(group):
            views[index] = ref.at[place]
    return views


def stacked_read(refs, groups: tuple[tuple[int, ...], ...]) -> list[jax.Array]:
    """Each stacked ref's member (`stacked_zeros`) after the launch, an array, in its shapes' order."""
    found: list = [None] * sum(map(len, groups))
    for ref, group in zip(refs, groups, strict=True):
        stacked = jax.freeze(ref)
        for place, index in enumerate(group):
            found[index] = stacked[place]
    return found


@dataclasses.dataclass(frozen=True)
class Outbox:
    """A stage each of a program's consumers writes its share of a step's sum into, which the program's
    producer adds into global memory (TMA's reduction) and frees (FlashAttention 3's dQ store warp: the add's
    waits off the consumers' walk): a consumer `claim`s its stage once the producer has read its last share,
    at its step's start, and `put`s its share when formed (a claim beside the put, ptxas sank the wait below
    the share's products and held them across it, 352 bytes spilled and C7512 serializing the walk's
    products where at the step's start the walk spills none, a shared dQ at width 128); the producer
    `take`s a step once every share is staged, adds each into its view and frees the stage once the adds
    have read it (`streamed`'s `adds`), having freed it once before its first (`opened`). Allocated with
    `outbox`."""

    smem: jax.Ref
    full: jax.Ref
    empty: jax.Ref

    def opened(self) -> None:
        """The producer's every stage freed before its first step, so a consumer's first put waits as its
        others do, with no branch of its own."""
        for share in range(self.smem.shape[0]):
            plgpu.barrier_arrive(self.empty.at[share])

    def claim(self, share: int) -> None:
        """Consumer share `share`'s stage, once the producer has read its last."""
        plgpu.barrier_wait(self.empty.at[share])

    def put(self, share: int, value: jax.Array) -> None:
        """Consumer share `share`'s `value` staged in its claimed stage (`claim`)."""
        self.smem.at[share][...] = value
        plgpu.commit_smem()
        plgpu.barrier_arrive(self.full.at[share])

    def take(self, views: tuple) -> None:
        """The producer's add of each staged share into its view of `views`, the stage freed once read."""
        for share, view in enumerate(views):
            plgpu.barrier_wait(self.full.at[share])
            plgpu.copy_smem_to_gmem(self.smem.at[share], view, reduction_op="add")
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)
        for share in range(len(views)):
            plgpu.barrier_arrive(self.empty.at[share])


def outbox(shape: tuple, dtype, shares: int) -> list:
    """An `Outbox`'s allocations for `pl.run_scoped`: a stage of `shape` in `dtype` a share, and its staged
    and freed barriers, one a share."""
    return [plgpu.SMEM((shares, *shape), dtype), plgpu.Barrier(num_barriers=shares),
            plgpu.Barrier(num_barriers=shares)]


@dataclasses.dataclass(frozen=True)
class Exchange:
    """A shared-memory block each of a program's consumers stages its share of and every consumer then
    reads whole, a step at a time (FlashAttention 3's P and dS between its two MMA warpgroups at width 256):
    its barriers (`Family.exchange`), `staged` arrived by each consumer once its share is staged, `read`
    once its reads of the block are done. A step `claim`s the block (every consumer's reads of the last
    step's done) only before it stages its share, past its own products, then `publish`es it and reads
    the whole once every share is staged, then `release`s it; `open` frees the block before the first step."""

    family: "Family"
    staged: object
    read: object
    consumers: int

    def open(self) -> None:
        self.family.turn(self.read, self.consumers, wait=False)

    def claim(self) -> None:
        self.family.turn(self.read, self.consumers, arrive=False)

    def publish(self) -> None:
        self.family.turn(self.staged, self.consumers)

    def release(self) -> None:
        self.family.turn(self.read, self.consumers, wait=False)


def per_role(index, roles: int, run: Callable[[int, jax.Array], None]) -> None:
    """`run(role, index)` for the role `index` names among `roles`, each role's built at its own static
    index: Mosaic GPU slices a swizzled tile only at static columns (a role's share of a block's, shared
    keys at width 256: offsets from the traced index spilled 20 bytes, a static role's none)."""
    for role in range(roles):
        pl.when(index == role)(functools.partial(run, role, index))


def unwalked(family: "Family") -> Callable:
    """A walk of no steps (`Family.persist`'s, where a tile streams nothing): its carry and accumulators as
    they start."""
    def walk(step, start, shapes, *, initial=None):
        del step
        return start, family.started(shapes, initial)

    return walk


def tiled(tiles: tuple[int, ...], schedule: Callable, visit: Callable, *, axis: str) -> tuple[jax.Array, ...]:
    """A persistent kernel's tiles (`Family.persist`'s): the grid `tiles` spread over the launch's programs
    along `axis`, each tile's `schedule(index)`, `(active, steps, views, tile)`, visited as `visit(counter,
    active, steps, views, tile, done)` at the step its streams are at across the tiles (`counter`: the steps
    of the active tiles before it) and its turn round accumulators a tile takes whole (`done`: the active
    tiles before it). Returns both past the last tile."""
    @plgpu.nd_loop(tiles, collective_axes=axis, init_carry=(jnp.int32(0), jnp.int32(0)))
    def ended(loop: plgpu.NDLoopInfo, carry):  # nd_loop passes its carry by this name
        counter, done = carry
        active, steps, views, tile = schedule(loop.index)
        visit(counter, active, steps, views, tile, done)
        return counter + jnp.where(active, steps, 0), done + jnp.where(active, 1, 0)

    return ended


def consumed(family: "Family", tiles: tuple[int, ...], schedule: Callable, issue: Callable, finish: Callable,
             store: Callable, index, walking: Callable, *, axis: str) -> jax.Array:
    """Consumer role `index`'s tiles of a persistent kernel (`tiled`'s), issued and finished in turn
    (`Family.persist`'s `issue` and `finish`): each tile's accumulators `issue(index, tile, walk,
    active=...)`, `walk` its steps' (`walking(counter, active, steps, views)`, built where the tile is
    visited) where it streams any, a walk of none (`unwalked`) where not, then `finish(index, tile, accs,
    store, active=...)`. Returns the counter past the last tile."""
    def visit(counter, active, steps, views, tile, done):
        del done
        walk = walking(counter, active, steps, views)

        def run(walk, *, active: bool):
            finish(index, tile, issue(index, tile, walk, active=active), store, active=active)

        pl.when(active)(lambda: run(walk, active=True))
        pl.when(~active)(lambda: run(unwalked(family), active=False))

    return tiled(tiles, schedule, visit, axis=axis)[0]


class Pieces(tuple):
    """A step's tile loaded as several copies (`viewed`, `Ring.load`): its block's rows in as many equal
    parts, each a view of its own, in order (a `Rows` step's runs)."""


@dataclasses.dataclass(frozen=True)
class Rows:
    """A block's rows in runs at a table's entries, the toolkit's one table-indexed row form: its run `i`,
    `rows` rows, at run `table[start + i]` of the array's rows (`table` an int32 ref, its entries in runs),
    its other block indices `rest`, each run a copy of its own (`viewed`'s `Pieces`). A paged pool's key
    tile across the pages it spans takes a run a page."""

    table: Any
    start: Any
    rest: tuple
    rows: int


def viewed(view: tuple, block: tuple, step):
    """Step `step`'s tile of a stream's view `(ref, index_map)` (`Family.opened`): the `block`-shaped tile of
    `ref` at the block indices `index_map(step)` gives, as a `BlockSpec`'s index map does; where it gives
    `Rows`, each run's view at its table's entry; where `ref` is `Pieces` of refs (an fp32 operand's bf16
    parts, each its own array's), each one's tile at the indices, its share of the block's rows, in order."""
    ref, index_map = view
    indices = index_map(step)
    if isinstance(ref, Pieces):
        share = (block[0] // len(ref), *block[1:])
        return Pieces(viewed((each, index_map), share, step) for each in ref)

    def at(indices, block):
        return ref.at[tuple(pl.ds(index * extent, extent)
                            for index, extent in zip(indices, block, strict=True))]

    if isinstance(indices, Rows):
        run = (indices.rows, *block[1:])
        return Pieces(at((indices.table[indices.start + each], *indices.rest), run)
                      for each in range(block[0] // indices.rows))
    return at(indices, block)


def loads(rings: dict, counter, active, steps, views: dict, tile, done) -> None:
    """A producer's loads of one persistent tile (`tiled`'s visit): each of its steps' blocks into its
    stream's ring at the step the rings are at (`counter`), none where it streams nothing."""
    del tile, done

    @pl.loop(0, jnp.where(active, steps, 0))
    def _(step):
        for name, ring in rings.items():
            ring.load(ring.turn(counter + step), ring.viewed(views[name], step))


def streamed(rings: tuple, views, lo, hi, *, adds: tuple | None = None) -> None:
    """A producer's loads of steps `lo` to `hi` into `rings`, each ring's step counted from `lo` and loaded
    from its view of `views(step)`, then each ring drained; with `adds` (an `Outbox` and `targets(step)`, its
    views of a step's sum), each step's sum added, a ring's depth behind its loads: a step's sum is staged
    after its slots are released, so the producer waits for it no sooner than for the next load's slot."""
    lag = min(ring.stages for ring in rings) if adds is not None else 0
    if adds is not None:
        adds[0].opened()

    @pl.loop(lo, hi + lag)
    def _(index):
        @pl.when(index < hi)
        def _():
            for ring, view in zip(rings, views(index), strict=True):
                ring.load(ring.turn(index - lo), view)

        if adds is not None:
            box, targets = adds
            pl.when(index - lag >= lo)(lambda: box.take(targets(index - lag)))

    for ring in rings:
        ring.drain(hi - lo)


def passed(rings: tuple, lo, hi, *, first: jax.Array | int = 0, between=None) -> None:
    """A consumer's steps `lo` to `hi` of `rings` it reads none of (another consumer's), each ring's load
    awaited and its slot released as a reading consumer's are (counted from `first`): a wait takes its
    barrier's next phase, so every consumer waits on every load; `between()` after each step."""
    @pl.loop(lo, hi)
    def _(index):
        for ring in rings:
            ring.acquire(ring.turn(index - first))
            ring.release(ring.turn(index - first))
        if between is not None:
            between()


def rotated(turned: Callable, lo, hi, first, stages: int) -> tuple:
    """Steps `lo` to `hi` of a ring of `stages` slots (its counters from `first`) as a loop takes them,
    `(run, start, stop)`: a rotation of the ring a trip, `run(rotation, *state) -> state` its `stages`
    steps unrolled, each at its slot, a constant, and the trip's one parity (`Turn`), so no step forms a
    slot's address or tests a barrier's phase. `turned(index, turn, *state) -> state` takes each step the
    range holds: a trip's steps outside it (its first and last trips', where the range is not whole
    rotations) are another walk's turns, which it skips (`taken`). It serves where no warpgroup accumulator
    is carried (`rotations`: a Blackwell product warp's loop); a walk that carries one takes `stepping`, whose
    loops leave no cond between its products, which ptxas serializes them for."""
    def run(rotation, *state):
        for slot in range(stages):
            counter = rotation * stages + slot
            index = counter + first
            step = functools.partial(turned, index, Turn(counter, slot, rotation))
            state = taken((index >= lo) & (index < hi), step, *state)
        return state

    return run, (lo - first) // stages, (hi - first + stages - 1) // stages


class Loop(NamedTuple):
    """A loop a walk runs in turn (`Family.accumulated`): `step(index, carry, accumulators)` from `lo` to
    `hi`, then `then()` once inside the walk's scope where it is given (a ring segment's products done and
    its slots given back before the next segment's steps wait for them, `RingFamily._walked`)."""

    step: Callable
    lo: Any
    hi: Any
    then: Callable[[], None] | None = None


def stepping(turned: Callable, lo, hi, first, stages: int, *, unrolled: bool) -> tuple[Loop, ...]:
    """Steps `lo` to `hi` of a ring of `stages` slots (its counters from `first`) as the loops a walk runs in
    turn (`Loop`); `turned(index, turn, *state, before=0) -> state` takes each step at
    its `Turn`, `before` the steps of its trip's, in its range, known to precede it. A loop of a step a trip,
    its slot traced (`Turn.of`); or, `unrolled`, of a rotation of the ring a trip, its `stages` steps each at
    its slot, a constant, and the trip's one parity, so no step of it forms a slot's address by arithmetic
    or tests a barrier's phase (an H100's 128 x 256 GEMM mainloop took 15.5 instructions an HGMMA with a slot
    formed each step, against CUTLASS's 13.5 and this form's 10.4, K40's SASS), the fewer than `stages`
    steps before the range's whole rotations and after them one a trip. No step of a rotation is skipped by a
    guard (`taken`): a cond between a warpgroup's products is one ptxas serializes them for (note C7514: a
    wait after each HGMMA, 16 for 16, where the unguarded walk waits once a step, a dense 128 x 256 GEMM's
    mainloop, whose 10.7 instructions an HGMMA were so a serialized loop's).

    Unrolled only where a step is small beside its waits (a GEMM's: its products), as the copies multiply a
    step's code: an attention or recurrence step's `stages` of them spilled 20 to 210 bytes with every slot
    opaque (`opaque`), and 60 to 540 without, where a step a trip spilled none (K40-RING2)."""
    def single(index, *state):
        return turned(index, Turn.of(index - first, stages), *state)

    if not unrolled:
        return (Loop(single, lo, hi),)

    def whole(rotation, *state):
        for slot in range(stages):
            counter = rotation * stages + slot
            state = turned(counter + first, Turn(counter, slot, rotation), *state, before=slot)
        return state

    # The first rotation's start at or after `lo` and the last's end at or before `hi`, the first not past
    # `hi` and the last not before the first (a range inside one rotation takes its steps one at a time).
    begin = jnp.minimum(lo + (first - lo) % stages, hi)
    end = jnp.maximum(hi - (hi - first) % stages, begin)
    return (Loop(single, lo, begin), Loop(whole, (begin - first) // stages, (end - first) // stages),
            Loop(single, end, hi))


def rotations(lo, hi, first, stages: int, step: Callable[[jax.Array, "Turn"], None]) -> None:
    """`step(index, turn)` for each step `lo` to `hi` of a ring of `stages` slots (counters from `first`),
    a rotation of the ring a trip (`rotated`): a loop that carries nothing (a Blackwell product warp's)."""
    def turned(index, turn):
        step(index, turn)
        return ()

    run, start, stop = rotated(turned, lo, hi, first, stages)

    @pl.loop(start, stop)
    def _(rotation):
        run(rotation)


def taken(active, step: Callable, *state):
    """`step(*state)` where `active`, `state` as it is otherwise (`lax.cond`): its arrays carried through the
    branch, its refs (a warpgroup accumulator's) read where they are; `step` returns `state`'s structure,
    its refs the same."""
    leaves, tree = jax.tree.flatten(state)
    refs = [isinstance(leaf, state_types.TransformedRef)
            or isinstance(jax.typeof(leaf), state_types.AbstractRef) for leaf in leaves]

    def rebuilt(values):
        values = iter(values)
        return jax.tree.unflatten(tree, [leaf if ref else next(values)
                                         for leaf, ref in zip(leaves, refs, strict=True)])

    def stepped(*values):
        stepped = jax.tree.leaves(step(*rebuilt(values)))
        return [leaf for leaf, ref in zip(stepped, refs, strict=True) if not ref]

    carried = [leaf for leaf, ref in zip(leaves, refs, strict=True) if not ref]
    return rebuilt(lax.cond(active, stepped, lambda *values: list(values), *carried))


def loaded(copies, issued: jax.Array | bool = True) -> None:
    """Each `(view, ref, barrier)` copied whole into shared memory, every copy issued before any is
    awaited: issued where `issued` (a role loading `ref` for every role that shares it), awaited by every
    caller."""
    @pl.when(issued)
    def _():
        for view, ref, barrier in copies:
            plgpu.copy_gmem_to_smem(view, ref, barrier)

    for _, _, barrier in copies:
        plgpu.barrier_wait(barrier)


def column_chunks(value: jax.Array, width: int) -> list[tuple[int, jax.Array]]:
    """`value` `width` of its columns at a time, each chunk with its first column, the whole where it is no
    wider: a stage narrower than the tile it moves (CUTLASS's sub-tiled epilogue, `staged_columns`)."""
    if width >= value.shape[-1]:
        return [(0, value)]
    return [(first, value[..., first:first + width]) for first in range(0, value.shape[-1], width)]


def chunked(value: jax.Array, view, smem) -> list[tuple[jax.Array, Ref]]:
    """`value` and its `view` a chunk of `smem`'s columns at a time (`column_chunks`)."""
    width = smem.shape[-1]
    if width >= value.shape[-1]:
        return [(value, view)]
    return [(piece, view.at[..., pl.ds(first, width)]) for first, piece in column_chunks(value, width)]


def stored(parts, views, *, smem, wait: bool = True) -> None:
    """Each of `parts` written whole to its view of `views` through the shared buffer `smem`, in turn, a
    chunk of its columns at a time where `smem` is narrower (`chunked`), the buffer reused once its copy
    has read it; the last copy waited for unless `wait` is false, when it stays in flight, one store group
    (`Family.store_wait`)."""
    pieces = [piece for part, view in zip(parts, views, strict=True) for piece in chunked(part, view, smem)]
    last = len(pieces) - 1
    for index, (part, view) in enumerate(pieces):
        smem[...] = part.astype(smem.dtype)
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(smem, view)
        if index < last:
            # The next piece writes `smem`: this copy's reads of it done first.
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)
        elif wait:
            plgpu.wait_smem_to_gmem(0)


def written(smem, value: jax.Array) -> None:
    """`value` written to the shared buffer `smem` in its dtype by an optimized transfer from its layout."""
    smem[...] = value.astype(smem.dtype)


def staged_rows(smem, value: jax.Array, o_gmem, row_start, first, stored, column, most: int,
                least: int = 1, *, write: Callable = written) -> None:
    """`value` written to the shared buffer `smem` in its dtype (`write`: its issuer's, `Issuer.written`),
    made visible to TMA, and its rows `first` to `first + stored` stored to `o_gmem` at row
    `row_start + first` and `column` (`store_rows`); the copies stay in flight (the caller waits before
    `smem` is written again)."""
    write(smem, value)
    plgpu.commit_smem()
    store_rows(smem, o_gmem, row_start, first, stored, column, most, least)


def store_rows(o_smem, o_gmem, row_start, first, stored, column, most: int, least: int = 1,
               copy: Callable = plgpu.copy_smem_to_gmem, widest: int | None = None) -> None:
    """Rows `first` to `first + stored` of `o_smem` to `o_gmem` at row `row_start + first` and `column`, in a
    power-of-two ladder of static-size `copy`s (a TMA store's descriptor has static extents; `copied`
    moves the rows through registers where a GPU has no TMA), the largest `most` rows', the smallest
    `least` (`stored` a multiple of it: a rung below it is never taken, and none is emitted); with
    `widest`, in copies of at most `widest` rows, whole ones first (a copy through registers holds its
    rows: `resources.copied_rows`)."""
    start, size = first, most
    if widest is not None and widest < most:
        for whole in range(most // widest):
            @pl.when(stored // widest > whole)
            def _(whole=whole):
                copy(o_smem.at[pl.ds(first + whole * widest, widest)],
                     o_gmem.at[pl.ds(row_start + first + whole * widest, widest),
                               pl.ds(column, o_smem.shape[1])])

        start, size = first + stored // widest * widest, widest - 1
    while size >= least:
        rung = 1 << int(math.log2(size))
        size //= 2

        @pl.when(stored & rung != 0)
        def _(start=start, rung=rung):
            copy(o_smem.at[pl.ds(start, rung)],
                 o_gmem.at[pl.ds(row_start + start, rung), pl.ds(column, o_smem.shape[1])])

        start = start + (stored & rung)


def copied(source, destination) -> None:
    """`source` (in shared memory) to `destination` (in global memory) through registers: a GPU without
    TMA's `store_rows` copy, explicitly unoptimized. JAX 0.12 takes a transfer with no `optimized` flag as
    optimized-only (0.11 downgraded it), which a ladder rung of a row or two cannot be, so its layout
    inference searched every rung's layouts to exhaustion (ns-gate1: the sm80 forward failed after about
    1000 s; without the ladder, or with these transfers unoptimized, it lowers in seconds:
    ns-noladder2, ns-abort1). A block whose elements no warpgroup's strided layout holds, a multiple of its
    threads (a ragged store's one row of an odd multiple of 64 columns, as every tile of gpt-oss's 2880-wide
    outputs: row 4474), is copied in a row's layout (`_row_layout`)."""
    strided = math.prod(source.shape) % resources.THREADS == 0
    layout = None if strided else _row_layout(source.shape, source.dtype)
    plgpu.store(destination, plgpu.load(source, layout=layout, optimized=False), optimized=False)


def _row_layout(shape: tuple[int, ...], dtype) -> plgpu_core.SomeLayout:
    """A `[rows, columns]` block's layout a row at a time (`copied`'s): each row's columns over a warp's
    lanes, as many adjacent ones a lane as a whole number of them takes up to 16 bytes, and every warp of
    the warpgroup holding the same; their stores race, writing the same values, which the CUDA memory model
    allows (`fragmented_array`'s tiled store). `columns` a multiple of a warp's lanes."""
    lanes = mgpu_fragments.WARP_SIZE
    vector = math.gcd(shape[-1] // lanes, 16 // jnp.dtype(dtype).itemsize)
    return plgpu.Layout.TILED(mgpu_fragments.Tiling(((1, lanes * vector), (vector,))),
                              warp_dims=(mgpu_fragments.Replicated(mgpu_fragments.WARPS_IN_WARPGROUP),),
                              lane_dims=(-2,), vector_dim=-1)


def coalesced(source, destination) -> None:
    """`copied`, through registers in Mosaic GPU's shared-to-global copy layout (`Layout.SMEM_GMEM_COPY`)
    where `source` tiles it: each lane 16 bytes of a row, a warp's lanes along tiles of 8 rows and the
    widest swizzle a row's bytes are a multiple of, so its stores write whole sectors. The layout
    inference's own choice was the MMA fragment's, a lane's 4 bytes a row, so a warp wrote half sectors: an
    A100's forward of Qwen3.5-27B's gated delta net took 1.162 ms against 0.941 with every store so, its
    states launch 0.422 against 0.314 and its output launch 0.299 against 0.258, the port's 0.984
    (JS-a100copy-a100-0b4eb412b). A walk's 16-bit stores alone (`Mma.stored`): the attention template's
    forward, whose `persist` stores its output so, took 19% longer (JS-2075cells-a100-491c05798), and an fp32
    tile's copy so spilled 48 bytes in the training query/key launch (the recurrence resource gate, sm80)."""
    layout = _copy_layout(source.shape, source.dtype)
    if layout is None:
        copied(source, destination)
    else:
        plgpu.store(destination, plgpu.load(source, layout=layout, optimized=False), optimized=False)


def _copy_layout(shape: tuple[int, ...], dtype) -> plgpu_core.SomeLayout | None:
    """`coalesced`'s layout for a 16-bit tile of `shape`, None where the copy layout tiles none of it."""
    if len(shape) != 2 or jnp.dtype(dtype).itemsize != 2:
        return None
    swizzle = next((swizzle for swizzle in (128, 64, 32, 16) if copy_tiles(shape, dtype, swizzle)), None)
    return None if swizzle is None else plgpu.Layout.SMEM_GMEM_COPY(tuple(shape), jnp.dtype(dtype), swizzle)


def copy_tiles(shape: tuple[int, ...], dtype, swizzle: int) -> bool:
    """Whether Mosaic GPU's tiled copy layout (cp.async's into a swizzled block, and `coalesced`'s) tiles a
    `[rows, columns]` block of `dtype` staged at `swizzle` bytes: its 8-row tiles as wide as the swizzle,
    a warpgroup's four warps' tiles four down the rows, two by two, or four across
    (fragmented_array.tiled_copy_smem_gmem_layout, which refuses any other)."""
    rows, width = shape[-2], shape[-1] * jnp.dtype(dtype).itemsize
    if rows % 8 or width % swizzle:
        return False
    row_tiles, column_tiles = rows // 8, width // swizzle
    return row_tiles % 4 == 0 or (row_tiles % 2 == 0 and column_tiles % 2 == 0) or column_tiles % 4 == 0


def copied_swizzle(shape: tuple[int, ...], dtype) -> int:
    """The swizzle, in bytes, a `[rows, columns]` block of `dtype` a program copies in with cp.async stages
    at: its rows' widest (`resources.widest_swizzle`), or the widest narrower one the copy layout tiles the
    block at (`copy_tiles`) where it tiles none at that, as an A100's key memory backward's 8 rows of 64 bf16
    columns, one tile at 128 bytes, four at 32 (a decay a key dimension 16 wide, the recurrence stability
    sites)."""
    widest = resources.widest_swizzle(dtype, shape[-1])
    narrower = (swizzle for swizzle in (widest, widest // 2, widest // 4)
                if swizzle >= NARROWEST_SWIZZLE_BYTES)
    return next((swizzle for swizzle in narrower if copy_tiles(shape, dtype, swizzle)), widest)


@dataclasses.dataclass(frozen=True)
class Turn:
    """Step `counter`'s turn of a ring of `stages` slots (`Ring`), the counter running from the ring's first
    load: its slot, `counter % stages`, and its rotation, `counter // stages`, whose parity is the phase of
    the slot's barriers its waits take. Every slot's barriers complete one phase a rotation, so the one parity
    serves them all (`plgpu.barrier_wait`'s `parity`), with no phase tracked a barrier. A slot a Python int
    where a walk knows it (a rotation unrolled: `rotated`), traced otherwise; the parities integer arithmetic
    (a warp's Mosaic lowering takes no boolean not)."""
    counter: jax.Array | int
    slot: jax.Array | int
    rotation: jax.Array | int

    @classmethod
    def of(cls, counter, stages: int) -> "Turn":
        """Step `counter`'s turn of a ring of `stages` slots."""
        return cls(counter, counter % stages, counter // stages)

    @property
    def parity(self):
        """The parity of the phase its fill completes, which a wait for it takes."""
        return self.rotation % 2 == 1

    @property
    def freed(self):
        """The parity of the phase the rotation before completes at its slot's release, which its fill waits
        for (a second rotation's on)."""
        return (self.rotation + 1) % 2 == 1

    def back(self, steps: int, stages: int) -> "Turn":
        """The turn `steps` before it (fewer than `stages`), its slot constant where this one's is."""
        if isinstance(self.slot, int):
            return Turn(self.counter - steps, (self.slot - steps) % stages,
                        self.rotation if self.slot >= steps else self.rotation - 1)
        return Turn.of(self.counter - steps, stages)


_COMMITTED = contextvars.ContextVar("committed", default=0)
"""The product groups an asynchronous issuer has committed as a program traces (`committed`)."""


def committed() -> None:
    """One product group committed by an asynchronous issuer (a warpgroup MMA, which `Issuer.settled` waits
    on by groups): counted as the program traces, so a walk knows a step's groups (`commits`)."""
    _COMMITTED.set(_COMMITTED.get() + 1)


def commits() -> int:
    """The product groups committed so far in this trace (`committed`): a walk's completion tokens, the
    difference across a step its groups."""
    return _COMMITTED.get()


def opaque(slot) -> jax.Array:
    """A ring's `slot` (a Python int where a walk unrolls its steps, traced otherwise) as a value the compiler
    cannot see through (an optimization barrier), so every address a slot makes, a barrier's and an operand
    tile's, is formed where it is used. A slot the compiler knows (a constant, an unrolled rotation's) has
    them formed once, ahead of the persistent loop, and held in registers for the whole kernel, its epilogue
    too: 5 to 6 more registers live over most of a 128 x 256 forward than with a traced slot (33 against 28
    at one multiprocessor, 38 against 32 at a cluster's pair), beside an accumulator's 128 and the packed
    rows' 64 in an epilogue that has 168 (the GPU's compile ignores `setmaxnreg`,
    `resources.RAISES_HONORED`). It spilled 148 bytes at the one, 2428 at the pair (K40-RING2). The
    barrier costs the addresses' arithmetic back, 2.4 instructions an HGMMA of a dense 128 x 256 mainloop's
    10.4 (8.1 with constant slots)."""
    return lax.optimization_barrier(jnp.asarray(slot, jnp.int32))


@dataclasses.dataclass(frozen=True)
class Ring:
    """A ring of `stages` shared-memory slots one operand streams through: a producer loads step `counter`
    into slot `counter % stages` (`load`, at its `Turn`), a consumer waits for it (`acquire`) and gives it
    back (`release`). The counter runs across a persistent kernel's tiles, so a tile's first loads reuse the
    slots the last tile left, and each tile may take its own number of steps (a causal tile's trip range):
    a slot is free for its first fill and, after, once every consumer released it; the producer drains the
    ring after its last load (`drain`). Allocated with `ring`, its slot count the allocation's own.

    A ring a cluster's programs share (`ring`'s `multicast`) takes each step's block once between them:
    every program's producer loads the same view, its share of the block multicast into each program's slot,
    and each program's load barrier awaits the whole block. A release arrives at each program's barrier, so
    a slot is free once every consumer of every program has released it (CUTLASS's PipelineTmaAsync), and
    the producer's drain holds its program until the others' last releases have reached it.

    Its protocol is the one the kernel challenge's Lean model proves safe for every slot, consumer, producer,
    byte and stage count (`Ring.ring_safe`, okc/notes/gpu-theory.md), on that proof's hypotheses, which
    `ring` builds and every use keeps: a slot's load barrier expects the arrivals its step's copies make
    (`copies`, `Pieces`), and its release barrier the arrivals of every consumer that reads it (`consumers`,
    each program's with `multicast`, or the products' one commit); each wait is at the parity of its turn
    (`Turn`), which flips once a wrap of the ring. tests/test_ring_protocol.py holds those hypotheses over
    every interleaving of the toolkit's own walks, and fails a flipped parity, a missing arrival and an
    early release."""
    smem: jax.Ref | state_types.TransformedRef
    full: jax.Ref
    consumed: jax.Ref

    @property
    def stages(self) -> int:
        """The ring's slots: its allocation's leading dimension."""
        return int(self.smem.shape[0])

    @property
    def block(self) -> tuple:
        """The block a slot holds."""
        return self.smem.shape[1:]

    @property
    def multicast(self) -> tuple | None:
        """The cluster axes the ring's loads are multicast along: its release barrier's, a cluster's (`ring`);
        None where the ring is its program's own."""
        dtype = self.consumed.dtype
        return dtype.collective_axes if isinstance(dtype, plgpu_core.ClusterBarrierType) else None

    def turn(self, counter) -> Turn:
        """Step `counter`'s turn (traced, or a Python int)."""
        return Turn.of(counter, self.stages)

    def initialized(self) -> None:
        """The ring's barriers ready, by every warpgroup before the roles split: Pallas's own, allocated
        initialized."""

    def viewed(self, view: tuple, step):
        """Step `step`'s source of the stream's view (`viewed`), as `load` takes it."""
        return viewed(view, self.block, step)

    def load(self, turn: Turn, view) -> None:
        """The producer's load of `turn`'s step from `view`, once its slot is free: released by the last
        rotation's step, whose parity is the other; `Pieces` each into its part of the slot's rows, one
        arrival each at the slot's load barrier, which takes as many (`ring`'s `copies`)."""
        slot = opaque(turn.slot)
        pl.when(turn.counter >= self.stages)(lambda: plgpu.barrier_wait(
            self.consumed.at[slot], parity=turn.freed))
        if not isinstance(view, Pieces):
            plgpu.copy_gmem_to_smem(view, self.smem.at[slot], self.full.at[slot],
                                    collective_axes=self.multicast)
            return
        rows = self.block[0] // len(view)
        for index, piece in enumerate(view):
            for source, destination in self.parts(slot, index * rows, piece):
                plgpu.copy_gmem_to_smem(source, destination, self.full.at[slot],
                                        collective_axes=self.multicast)

    def parts(self, slot, start: int, piece) -> list[tuple]:
        """A piece's copies into `slot`'s rows from `start` (`load`), each a source and its destination: the
        slot's rows as its transforms index them, where they are whole tiles of rows (`TILE_ROWS`) or the slot
        is plain; else, a row's alone (a gathered stream's, `Rows` of one row), each of its tiles' columns
        into the slot's physical row at the slot's swizzle. TMA swizzles a row by its shared-memory address
        (the CUDA programming guide's swizzle pointer offset: a 128-byte-aligned row swizzled as the row of
        the pattern it lies at), so the row lands as its tile's row does."""
        rows = piece.shape[0]
        match self.smem.transforms if isinstance(self.smem, state_types.TransformedRef) else ():
            case (plgpu_core.UnswizzleRef() as unswizzled, plgpu_core.UntilingTransform(tiling=(tile, width)),
                  ) if rows % tile:
                physical = state_types.TransformedRef(self.smem.ref, (unswizzled,))
                return [(piece.at[pl.ds(row, 1), pl.ds(column * width, width)],
                         physical.at[slot, (start + row) // tile, column, pl.ds((start + row) % tile, 1)])
                        for row in range(rows) for column in range(self.block[1] // width)]
        return [(piece, self.smem.at[slot, pl.ds(start, rows)])]

    def acquire(self, turn: Turn):
        """A consumer's wait for `turn`'s load: its slot. Each load is waited for by each consumer, as every
        phase of a barrier is by each thread that waits on it; a wait again for the same turn returns at
        once, its slot not refilled until released."""
        slot = opaque(turn.slot)
        plgpu.barrier_wait(self.full.at[slot], parity=turn.parity)
        return self.smem.at[slot]

    def release(self, turn: Turn) -> None:
        """A consumer's release of `turn`'s slot, whose reads (and MMAs) are done."""
        plgpu.barrier_arrive(self.consumed.at[opaque(turn.slot)])

    def drain(self, end) -> None:
        """The producer's wait, after its last load (step `end - 1`), for the consumers' release of each slot
        it filled last, so no slot is freed (or reallocated) while a consumer still reads it, and every
        release is seen (`drained`)."""
        drained(self.consumed, end, self.stages)


def drained(barriers, end, stages: int) -> None:
    """The completions of the last `stages` of `end` turns round `barriers`, a slot's each (`Turn`), which no
    later turn of the slot waited for: each barrier ends at the phase its waiters saw."""
    for back in range(1, stages + 1):
        turn = Turn.of(end - back, stages)
        pl.when(turn.counter >= 0)(lambda turn=turn: plgpu.barrier_wait(barriers.at[opaque(turn.slot)],
                                                                       parity=turn.parity))


def ring(block: tuple, dtype, stages: int, *, consumers: int = 1, multicast: bool = False,
         swizzle: int | None = None, copies: int = 1, released_by_products: bool = False) -> dict:
    """A `Ring`'s allocations for `pl.run_scoped`: `stages` slots of `block` in `dtype` (their rows staged
    at `swizzle`, `staged`), a load barrier each that a step's `copies` arrive at (`Pieces`), and a release
    barrier each that `consumers` arrive at, with `multicast` the `consumers` of each program of the cluster
    (`CLUSTER`), whose programs share the ring's blocks; or, `released_by_products`, that the products
    reading a slot arrive at once by their commit (tensor-core ordered: tcgen05's). `Ring(**allocated)` once
    scoped."""
    if released_by_products:
        consumed = plgpu.Barrier(num_barriers=stages, orders_tensor_core=True)
    elif multicast:
        consumed = plgpu.ClusterBarrier(collective_axes=(CLUSTER,), num_arrivals=consumers,
                                        num_barriers=stages)
    else:
        consumed = plgpu.Barrier(num_arrivals=consumers, num_barriers=stages)
    return {'smem': staged((stages, *block), dtype, swizzle),
            'full': plgpu.Barrier(num_arrivals=copies, num_barriers=stages), 'consumed': consumed}


SWIZZLE_BYTES = 128
"""The widest swizzle a shared-memory block's rows are staged at, which an MMA reads conflict-free."""

NARROWEST_SWIZZLE_BYTES = 32
"""The narrowest swizzle Mosaic GPU stages a block's rows at (`plgpu.find_swizzle`'s least)."""

UNSWIZZLED = 0
"""A stream's swizzle (`Family.streams`' `swizzles`) where a step reads its tile a row at a time, at a traced
row (a scan's tokens): staged plain, no tiling or swizzle, which Mosaic GPU indexes at any row."""


def swizzle_columns(dtype, swizzle: int) -> int:
    """The elements of `dtype` a row of a `swizzle`-byte swizzle spans: a swizzled tile's width, so the least
    slice a load of a block swizzled along that axis takes."""
    return swizzle // jnp.dtype(dtype).itemsize


def swizzle_for(dtype, *contractions: int) -> int:
    """The widest swizzle, in bytes, a shared-memory block of `dtype` takes whose group of elements (the
    swizzle over the itemsize: Mosaic's wgmma issues a product's contraction in groups of it, one swizzle
    for both its shared operands, wgmma.py's `k_group_elems`) divides each of `contractions`, the elements
    the products it takes part in contract over a step. A product contracting over 32 rows of 16-bit
    operands so stages them at 64 bytes, at which its partner block must stage too (K32SWZ3: S = K Qᵀ
    over a 128-wide head lowers with both at 64 and both at 128, not mixed; dV += P dO over 32 rows lowers
    at 64, not 128)."""
    return next((swizzle for swizzle in (SWIZZLE_BYTES, SWIZZLE_BYTES // 2, NARROWEST_SWIZZLE_BYTES)
                 if all(contraction % swizzle_columns(dtype, swizzle) == 0 for contraction in contractions)),
                NARROWEST_SWIZZLE_BYTES)


TILING: Callable[[tuple[int, int]], object] = plgpu.TilingTransform
SWIZZLE: Callable[[int], object] = plgpu.SwizzleTransform
"""Mosaic GPU's tile transforms, which it declares as protocol classes (a type checker takes them for
uninstantiable)."""


TILE_ROWS = 8
"""The rows of a swizzled shared-memory tile (`swizzle_transforms`): a block's rows are sliced in whole
multiples of them (a ring slot's `Pieces`)."""


def swizzle_transforms(dtype, swizzle: int) -> tuple:
    """A shared-memory tile's transforms at `swizzle` bytes: `TILE_ROWS`-row tiles a swizzle's row wide,
    swizzled; a row narrower than the narrowest swizzle (16 bytes: an NVFP4 stage's block scales, row 3621)
    tiled alone, a 16-byte swizzle being no permutation, which Mosaic GPU's swizzle transform refuses."""
    tiling = TILING((TILE_ROWS, swizzle_columns(dtype, swizzle)))
    return (tiling,) if swizzle < NARROWEST_SWIZZLE_BYTES else (tiling, SWIZZLE(swizzle))


def plain(shape: tuple, dtype):
    """A shared-memory block laid out plainly, neither tiled nor swizzled, stated so (`plgpu.PlainTransform`):
    one read at row offsets no tiling divides, which layout inference would otherwise search every tiling
    and swizzle for, each subview of it apart (a Kimi delta attention intra launch's out of its fuel at 32-
    and 64-key blocks on an H100)."""
    return plgpu.SMEM(shape, dtype, transforms=(plgpu.PlainTransform(),))


def laid(condition, like: jax.Array) -> jax.Array:
    """The scalar `condition` laid over `like`, a vector predicate of its shape: Mosaic GPU's warpgroup
    semantics select under no scalar predicate (`jnp.where` passes a scalar condition as it is; the fused
    step's compile, "Implicit broadcast not implemented with warpgroup semantics"), where a scalar added to a
    vector is laid over it."""
    return (jnp.zeros_like(like, jnp.float32) + jnp.asarray(condition, jnp.float32)) != 0.0


def synced(interpret: bool) -> None:
    """The warpgroup's threads past their shared-memory reads and writes before any goes on: a tile one thread
    wrote that another reads, or one written again that another still reads (a walk's scratch tile). Mosaic
    GPU's loads and stores order no other thread's, and `plgpu.commit_smem`'s async-proxy fence exists from
    sm90 on (an A100's LLVM selects no `fence.proxy.async`); interpreted, a load and a store take the whole
    tile at once."""
    if interpret:
        return
    plgpu.inline_mgpu()(lambda ctx: mgpu_utils.warpgroup_barrier())()


MAGIC_INT, MAGIC_FLOAT = 0x4B400000, 12582912.0
"""An integer below 2**22 in magnitude added to the first in integer arithmetic is the fp32 whose value less
the second it is (2**23 + 2**22's exponent and mantissa: Family.converted's integer codes)."""


def converts(dtype) -> bool:
    """Whether the current device converts quantized codes of `dtype` to 16 bits in registers
    (`Family.converted`): integer codes on any, FP8 codes where its tensor instructions take 8-bit operands
    (`generation.Architecture.tensor_rates`' '8bit', sm_89 on), as PTX's e4m3x2 and e5m2x2 conversions
    require (an A100's ptxas refused them, dd-quant4's q3)."""
    if jnp.issubdtype(dtype, jnp.integer):
        return True
    architecture = generation.architecture(generation.gpu_target().generation)
    return architecture is not None and "8bit" in architecture.tensor_rates


def staged(shape: tuple, dtype, swizzle: int | None = None):
    """A shared-memory block of `shape` whose rows are staged at `swizzle` bytes: as layout inference stages
    them where unstated (the widest that fits); otherwise in 8-row tiles of the swizzle's row, explicitly,
    which inference does not take for a block whose products contract over fewer elements than the widest
    swizzle's row (K32SWZ5: a register operand at a 32-row contraction lowered only so); plain where
    `UNSWIZZLED`."""
    if swizzle == UNSWIZZLED:
        return plain(shape, dtype)
    if swizzle is None:
        return plgpu.SMEM(shape, dtype)
    return plgpu.SMEM(shape, dtype, transforms=swizzle_transforms(dtype, swizzle))


def roles(wg, consumers: int, consumer, producer) -> None:
    """Each warpgroup's role in a cooperative block: the `consumers` first run `consumer(wg)`, the last
    `producer()` (`tabled`'s one row)."""
    tabled(wg, ((consumers, consumer),), producer)


def tabled(wg, table: tuple, producer) -> None:
    """Each warpgroup's role in a cooperative block from a role table: each of `table`'s rows, `(count,
    body)` in turn, its `count` warpgroups running `body(index)` (`index` the warpgroup's own among
    them), every one with its register budget raised to its share at the launch's residency
    (`resources.raised_registers`, `_RESIDENT`); the warpgroup after them runs `producer()` with its
    lowered to `resources.producer_floor`, where the GPU's compile honors them
    (`resources.RAISES_HONORED`)."""
    total = sum(count for count, _ in table)
    first = 0
    for count, body in table:
        # The first row's warpgroups are those below its count, each its own index: no bound and no shift
        # is traced for it, so a one-row table builds the same kernel `roles` built (an attention
        # forward's spill-free registers on an H100 turn on it, test_attention_mosaic's compile gate).
        within = wg < first + count if first == 0 else (wg >= first) & (wg < first + count)

        @pl.when(within)
        def _(body=body, first=first):
            plgpu.set_max_registers(resources.raised_registers(total, _RESIDENT.get()), action="increase")
            body(wg if first == 0 else wg - first)

        first += count

    @pl.when(wg == total)
    def _():
        plgpu.set_max_registers(resources.producer_floor(), action="decrease")
        producer()


class Slots:
    """A walk step's operand slots (`Family.walk`), by stream name: each acquired when the step first reads
    it (`slots[name]`) and released at the step's end, once its MMAs are done, unless the step `release`s it
    sooner (a slot its last product has read)."""

    def __init__(self, rings: dict[str, Ring], turn: Turn):
        self._rings, self._turn = rings, turn
        self._held: dict[str, Ref] = {}
        self._released: set[str] = set()

    def __getitem__(self, name: str):
        if name not in self._held:
            self._held[name] = self._rings[name].acquire(self._turn)
        return self._held[name]

    def release(self, name: str) -> None:
        """Stream `name`'s slot given back now, its reads done (acquired first if the step read none)."""
        self[name]
        self._rings[name].release(self._turn)
        self._released.add(name)

    def done(self) -> None:
        """Every slot the step has not given back given back, each acquired: each consumer waits on every load
        (a wait takes its barrier's next phase), so a slot it never read is still acquired before it is
        released."""
        for name in self._rings:
            if name not in self._released:
                self.release(name)

    def held(self) -> None:
        """Every slot the step has not read acquired, none given back: a step whose products stay running
        past it (`Family.walk`'s `in_flight`) gives its slots back a step later (`released`)."""
        if self._released:
            raise ValueError("a step whose products run past it gives back no slot early")
        self.acquire_all()

    def acquire_all(self) -> None:
        """Every slot the step has not read acquired in the step's own trace: a slot first acquired inside a
        loop's body (a rolled `lax.fori_loop`) is kept by its first read, which would carry that body's slot
        index past it, a scalar the kernel then takes as an input (the Mosaic GPU lowering fails on it); a
        step that reads its slots in a loop acquires them before it."""
        for name in self._rings:
            self[name]


def released(rings: dict[str, Ring], turn: Turn) -> None:
    """Every ring's slot of `turn` given back: a step's slots its walk held past it (`Slots.held`)."""
    for ring in rings.values():
        ring.release(turn)


@dataclasses.dataclass(frozen=True)
class Walk:
    """A program's walk of its tile, its steps storing what it computes (no accumulator past them, and its
    `carry`, from its value here, only between them: a per-key memory's gradients across key blocks):
    the family's loads of its streams' `views` (`viewed`) and its consumer's walk of `step` over its `steps`
    steps (one by default), or of its `segments` `(step, lo, hi)` where given (a launch's own walk, a step
    of its own for some of them). Where a `prologue` is given, the consumer runs it before its walk waits on
    the streams' first tiles, and each step takes what it returns (`loaded`): its own loads from global
    memory in flight beside the streams', not after them. A launch runs it on its roles (`Transport.run`),
    a persistent launch's tile through `persist`'s walk (`persisted`)."""

    views: dict
    step: Callable
    steps: int = 1
    prologue: Callable | None = None
    segments: tuple = ()
    carry: object = ()

    def stepped(self) -> tuple:
        """Its segments, each step given its prologue's value, the prologue run here (in the consumer)."""
        loaded = None if self.prologue is None else self.prologue()
        return tuple((step if loaded is None else functools.partial(step, loaded=loaded), lo, hi)
                     for step, lo, hi in (self.segments or ((self.step, 0, self.steps),)))

    def persisted(self, walk: Callable) -> None:
        """Its one step walked by a persistent tile's `walk(step, start, shapes)` (`Transport.persist`'s)."""
        if self.segments:
            raise ValueError("a persistent tile walks its one step, not segments of its own")
        ((step, _, _),) = self.stepped()
        walk(step, self.carry, ())


class Issuer[Accumulator](abc.ABC):
    """An instruction family's issuer and accumulator home (design §6): its products (`product` into a fresh
    accumulator, `products`, `register_products` and `issued` in place, `settled`), the accumulator's home
    and its layouts (`layout`, `row`, `column`, which `zeros`, `full`, `iota`, `rows_of`, `columns_of`,
    `column_sums` and `as_row` read), how an operand reaches a product (`held`, `staged`), and the facts its
    instructions decide (row units, fragments, register costs). A `Family` composes it with a `Transport`.
    Generic in the accumulator its products issue into in place (`Accumulator`: a warpgroup accumulator ref,
    or a register array)."""

    name: str
    """The family's name (`bundles.named`), which its refusals state."""

    rows: int

    role_rows: int
    """The rows a consumer role's products cover at least: one warpgroup's (wgmma's one MMA of 64 rows,
    mma.sync's four warps of 16 each)."""

    @property
    def transposed_rows(self) -> int:
        """The rows a step of an operand its products read from registers transposed (a kernel gradient's
        `[rows, depth]` stage) takes at least: a role's (`role_rows`)."""
        return self.role_rows

    def specialized(self, rows: int, columns: int) -> Self:
        """The issuer for a consumer role's `rows` x `columns` share of a tile: itself, unless its layout
        arranges its warps by the share (mma.sync's)."""
        del rows, columns
        return self

    def relay_bytes(self, rows: int, depth: int) -> int:
        """The shared memory a consumer's register left operand of `rows` x `depth` (an accumulator's value a
        product reads) takes on its way to the product's layout (`relaying`): none where the cast stays in
        registers."""
        del rows, depth
        return 0

    def relaying(self, tile) -> Self:
        """The issuer whose register left operands relay through `tile`, a consumer's own plain
        shared-memory tile (`plain`) of at least `relay_bytes` (`[rows, depth]` of the operand's dtype):
        itself where none do."""
        del tile
        return self

    held_registers: int = 40
    """The registers a consumer thread holds beside its accumulator and operand fragments (addresses,
    indices, the walk's carry): CUTLASS's cooperative sm90 kernel's 168 less its accumulator's 128."""

    scalars_whole: bool = True
    """Whether `held_registers` is a measured whole of a consumer thread's scalars, its addresses, indices
    and a select's predicates among them, which a launch's own scalar terms do not add to (an A100's
    attention dQ: 13 to 43 integers live at its peaks, places and addresses together, no predicate saved
    to a register, within the 40)."""

    def scalar_registers(self, addressed=()) -> int:
        """The registers a consumer thread holds beside its tiles: its own scalars (`held_registers`), its
        addresses into the shared tiles it reads and writes through registers (`addressed`, each
        `(shape, dtype)`) among them where the family's count is a measured whole, as this one is."""
        del addressed
        return self.held_registers

    operands_in_registers: bool
    """Whether a product reads its operands from registers a consumer loads them into (mma.sync's), where
    warpgroup MMA reads both from shared memory."""

    in_flight: int = 0
    """The steps whose products a walk keeps running past their step (`walk`'s `in_flight`), each holding
    its stage: none where products are synchronous; one where they run asynchronously, CUTLASS's sm90
    mainloops' `K_PIPE_MMAS`, so a step's issue overlaps the previous step's products."""

    products_overlap: bool
    """Whether a warpgroup's products run asynchronously, an issued product's accumulator held while the
    program goes on (warpgroup MMA's), where each mma.sync product completes before the next: a launch's
    products issued in turn hold their accumulators at once on the one, one at a time on the other."""

    @property
    def lhs_copies(self) -> int:
        """The warps of a role each holding the left operand's fragment for the same rows (mma.sync's warps
        along a tile's columns, JAX's `MMALayouts`' `lhs` replicated over them): one, unless replicated."""
        return 1

    @property
    def rhs_copies(self) -> int:
        """The warps of a role each holding the right operand's fragment for the same columns (mma.sync's
        warps along a tile's rows, JAX's `MMALayouts`' `rhs` replicated over them): one, unless
        replicated."""
        return 1

    def fragment_depth(self, depth: int) -> int:
        """The depth of a step's right operand fragments a consumer holds at once, where its products read
        registers (`operands_in_registers`): the step's, unless the family loads them a sub-step at a
        time (the left operand's it holds a step deep)."""
        return depth

    def rhs_fragment_depth(self, depth: int, *, transposed: bool = False, swizzle: int | None = None) -> int:
        """The right operand's depth held in registers, by its storage orientation and its stage's swizzle
        (bytes; the family's default staging where None)."""
        del transposed, swizzle
        return self.fragment_depth(depth)

    def lhs_fragment_depth(self, depth: int, *, swizzle: int | None = None) -> int:
        """The left operand's depth held in registers by a product, by its stage's swizzle (bytes; the
        family's default staging where None)."""
        del swizzle
        return depth

    widest_product: int | None = None
    """The most columns one product spans (warpgroup MMA's 256, wgmma's widest N), which a tile's columns
    are at most; None where a tile's columns are many products' (mma.sync's 8 each), held by its
    registers alone (`resources.tile_registers`)."""

    takes_tf32: bool = False
    """Whether its products take fp32 operands, as TF32 (warpgroup MMA's kind::tf32: Mosaic's wgmma issues
    fp32 shared-memory operands so; mma.sync's m16n8k8.tf32, the fork's from 0.12.0.1.post42); where not, a
    template splits fp32 operands into bf16 parts instead."""

    takes_fp8: bool = False
    """Whether its products take FP8 operands (`product.EIGHT`: E4M3, E5M2), both read K-major as Hopper's
    warpgroup MMA reads them ('Only f16 WGMMA supports transposes'); mma.sync's FP8 forms are Ada's and
    consumer Blackwell's, which no bundle of Dew's runs yet (row 3620)."""

    takes_block_scales: bool = False
    """Whether its products scale each 32 depths of their operands by an E8M0 scale in the instruction (an MX
    product's, `scaled_products`): tcgen05's block-scaled MMA; no other family's products do (row 4519)."""

    def scaled_products(self, acc, lhs, rhs, scales: tuple, interpret: bool):
        """`products` of block-scaled operands (`takes_block_scales`), `scales` the step's two scale slots."""
        raise NotImplementedError(f"{self.name}'s products take no block scales")

    transposes_tf32: bool = False
    """Whether its TF32 products take an operand read transposed (`takes_tf32`): mma.sync's, whose threads
    load each fragment's elements themselves; not warpgroup MMA's ('Only f16 WGMMA supports transposes'), so
    a product of transposed fp32 operands there takes bf16 parts."""

    counts_hooks: bool = True
    """Whether its register model counts what a template's hooks hold (side operands' tiles), so a hooked
    call derives a tile that spills nothing; where not (mma.sync's), such a call is not its launch's."""

    counts_sums: bool = True
    """Whether its register model counts an exactly summed call's running sum (a second fp32 accumulator
    beside the first and each step's product), so such a call derives a tile that spills nothing: every
    family's, an fp32 output of 16-bit products being exactly summed (`templates.product.promoted`)."""

    accumulator_rows: int | None = None
    """The rows of a consumer's accumulator its products issue into, where the family fixes them (tcgen05's
    M on one CTA); None where a tile takes any whole roles of rows."""

    issues_in_consumers: bool = True
    """Whether a walk's step runs in the consumer that issues its products, so it can prepare their operands
    first: a prologue's rewrite, a kernel gradient's zeroed neighbouring rows, an exactly summed step's own
    product. Where not (tcgen05's: one warp issues them, the tensor core reading each stage as loaded), a
    call needing that is not a template's launch's there."""

    takes_pairs: bool = True
    """Whether a launch on it takes a paired product whole, its model pricing the second accumulator beside
    the first in a consumer's registers (`column_registers`); where not (tcgen05's, whose accumulators stay
    in tensor memory, read out in chunks), its two products run apart."""

    def tensor_columns(self, columns: int, accumulators: int) -> int:
        """The tensor memory columns a program's `accumulators` fp32 accumulators of `columns` columns each
        take (`generation.GpuTarget.tmem_columns` bounds them): none where its consumers hold them in
        registers."""
        del columns, accumulators
        return 0

    def column_registers(self, columns: int, dtype) -> int:
        """The registers a consumer thread holds for a vector of `columns` entries of `dtype` in the
        accumulator's column layout (`column`), broadcast over its rows: a family whose register model
        prices such a vector (a hook's, a template's per-row statistics) states it."""
        raise NotImplementedError(f"{self.name}'s register model takes no column vector")

    def row_registers(self, rows: int, dtype) -> int:
        """The registers a consumer thread holds for a vector of `rows` entries of `dtype` in the
        accumulator's row layout (`row`), broadcast over its columns: a family whose register model prices
        such a vector (a template's per-row statistics) states it."""
        raise NotImplementedError(f"{self.name}'s register model takes no row vector")

    def narrowest_columns(self, dtype) -> int:
        """The fewest columns a staged operand block of `dtype` takes: a row of the narrowest swizzle
        (`NARROWEST_SWIZZLE_BYTES`), 16 bf16 columns, which both families' products read (wgmma's N in
        eights, mma.sync's n of 8 twice through ldmatrix). A template splitting its columns finer than its
        stores' `column_unit` to fill the multiprocessors (the recurrence's value blocks) steps by it."""
        return swizzle_columns(dtype, NARROWEST_SWIZZLE_BYTES)

    @property
    @abc.abstractmethod
    def layout(self) -> plgpu_core.SomeLayout:
        """The accumulator's register layout, which the accumulator's layout members (`zeros` to `as_row`)
        read."""

    @property
    @abc.abstractmethod
    def row(self) -> plgpu_core.SomeLayout:
        """A row vector's layout: the accumulator's reduced over its columns."""

    @property
    @abc.abstractmethod
    def column(self) -> plgpu_core.SomeLayout:
        """A column vector's layout: the accumulator's reduced over its rows."""

    def summed(self, shape: tuple[int, int], issue, interpret: bool) -> jax.Array:
        """`issue(acc) -> acc` issuing products in place (`products`) into a fresh fp32 accumulator of
        `shape` (zero), its value once they are done: a register accumulator's, which a family whose
        accumulator is a ref scopes (wgmma's)."""
        del interpret
        return issue(self.zeros(shape))

    @abc.abstractmethod
    def added_into(self, acc: Accumulator, value: jax.Array, interpret: bool) -> Accumulator:
        """The accumulator `acc` a walk carries with `value` (fp32, its shape, a scaled product restored,
        `operands.Issuer.summed`) added in."""

    def written(self, smem, value: jax.Array) -> None:
        """`value`, a tile in the issuer's register layout, written to the shared buffer `smem` in its dtype
        (an output's staging): by an optimized transfer where the layout takes one."""
        written(smem, value)

    def chunk_columns(self, columns: int) -> int:
        """The columns of a tile's accumulator a consumer holds in registers at once (`chunks`): all of them,
        where the issuer accumulates in registers or in a ref its walk reads out whole."""
        return columns

    def chunks(self, accumulators: tuple) -> tuple:
        """A tile's accumulators (its walk's) as its consumer reads them out: `(first_column, values)` a
        chunk of `chunk_columns` columns each, `values` each accumulator's: the accumulators themselves,
        one chunk, where the walk returns their values."""
        return ((0, accumulators),)

    def started(self, shapes: tuple, initial: tuple | None = None) -> tuple[jax.Array, ...]:
        """Register accumulators of `shapes` as a walk starts: their `initial` values, zero where None."""
        initial = (None,) * len(shapes) if initial is None else initial
        return tuple(self.zeros(shape) if value is None else value
                     for shape, value in zip(shapes, initial, strict=True))

    def accumulated[C](self, segments, start: C, shapes: tuple, interpret: bool, initial: tuple | None = None
                       ) -> tuple[C, tuple[jax.Array, ...]]:
        """A walk of `segments` (`Loop`s) in turn, with the family's accumulators of `shapes` held across it
        from their `initial` values (`walk`): in registers, the walk's loop carrying them (mma.sync's), unless
        the issuer's home is a ref it scopes (wgmma's)."""
        del interpret
        both = (start, self.started(shapes, initial))
        for loop in segments:
            both = lax.fori_loop(loop.lo, loop.hi, lambda index, both, step=loop.step: step(index, *both),
                                 both)
            if loop.then is not None:
                loop.then()
        return both

    @abc.abstractmethod
    def held(self, ref: Ref) -> Operand:
        """An operand every step reads (`ref`, in shared memory) where the family keeps it: shared memory,
        or registers."""

    @abc.abstractmethod
    def product(self, lhs, rhs, shape: tuple[int, int], then=None, *, interpret: bool = False) -> jax.Array:
        """`lhs @ rhs` into a fresh fp32 accumulator of `shape`, its value, with `then` issued while the
        product runs; `interpret`, as `products`' (an instruction no CPU rule lowers computed as its sum)."""

    @abc.abstractmethod
    def registers_product(self, lhs: jax.Array, rhs, interpret: bool) -> jax.Array:
        """`lhs @ rhs` into a fresh fp32 accumulator, `lhs` in registers (16-bit), its value."""

    def summed_window(self, block_k: int) -> int:
        """The steps of `block_k` depths an exactly summed kernel gradient's partial covers before it is
        added into the running sum (`gemm.mosaic_gpu._walked_gradient`): one, each step's partial apart,
        unless its accumulator is a value it may carry across steps."""
        del block_k
        return 1

    def summed_apart(self, running: jax.Array, lhs, rhs, interpret: bool, scale: Callable | None = None
                     ) -> jax.Array:
        """`running` plus `lhs @ rhs` summed apart (exact sums, `templates.product.promoted`): the step's own
        fp32 sum of its products, times `scale(start, columns)` where given (its columns' multiplier from
        `start`: a blockwise-scaled FP8 product's, `mosaic_gpu.gmm`'s `step_scales`), added into `running`
        by an IEEE add."""
        rows, columns = running.shape
        fresh = self.product(lhs, rhs, (rows, columns), interpret=interpret)
        return running + (fresh if scale is None else fresh * scale(0, columns))

    @abc.abstractmethod
    def products(self, acc: Accumulator, lhs, rhs, interpret: bool, *, accumulate: bool = True
                 ) -> Accumulator:
        """`acc` with `lhs @ rhs` issued into it in place, both operands in shared memory (a GEMM's step);
        unless `accumulate`, the accumulator started from zero (a walk's own fresh product, `read` once
        done)."""

    def read(self, acc: Accumulator, *, first: int = 0, columns: int | None = None, pending: int = 0
             ) -> jax.Array:
        """The accumulator's columns `first` to `first + columns` (all of them where None), once at most
        `pending` of the program's product groups are still in flight: a register home's value sliced, its
        products done as they are issued."""
        del pending
        value = jnp.asarray(acc)
        if first == 0 and columns is None:
            return value
        return value[:, first:first + (value.shape[1] - first if columns is None else columns)]

    def issued(self, acc: Accumulator, left: tuple, right: tuple, interpret: bool) -> Accumulator:
        """`acc` with the products of `left`'s and `right`'s parts (`product.formed`, each a part's
        shared-memory tile) issued into it in place, the pairs `product.terms` gives, smallest first
        (`products`)."""
        for i, j in product.terms(len(left), len(right)):
            acc = self.products(acc, left[i], right[j], interpret)
        return acc

    def apart(self, shape: tuple[int, int], left: tuple, right: tuple, interpret: bool) -> jax.Array:
        """The products of `left`'s and `right`'s parts (`issued`'s), each into its own fresh fp32
        accumulator of `shape` (`summed`), added left to right in `product.terms`' order, so the smallest
        meet first: a tensor core's accumulation truncates (mma.sync's), so a part accumulated onto a larger
        one loses its last bit where a sum rounds it."""
        def issue(i: int, j: int, acc: Accumulator) -> Accumulator:
            return self.products(acc, left[i], right[j], interpret)

        return functools.reduce(jnp.add, (self.summed(shape, functools.partial(issue, i, j), interpret)
                                          for i, j in product.terms(len(left), len(right))))

    @abc.abstractmethod
    def register_products(self, acc: Accumulator, parts, rhs, interpret: bool) -> Accumulator:
        """`acc` with `parts`' products with `rhs` issued into it in place, each part's left operand in
        registers, `parts` an iterable that may form each as it is taken
        (`attention_algebra.formed_parts`)."""

    @abc.abstractmethod
    def rescaled_products(self, acc: Accumulator, scale, parts, rhs, interpret: bool) -> Accumulator:
        """`acc` taken by `scale(value)` in place, then `register_products` of `parts` with `rhs` into it."""

    @abc.abstractmethod
    def settled(self, pending: int = 0) -> None:
        """Every product issued so far done, but the last `pending` issued."""

    def zeros(self, shape: tuple[int, ...]) -> jax.Array:
        """fp32 zeros in the accumulator's layout (two-dimensional) or a row's (one-dimensional)."""
        return self.full(shape, 0.0)

    def full(self, shape: tuple[int, ...], value: float, dtype=jnp.float32) -> jax.Array:
        """`value` in `dtype` (fp32 unless given), a splat whose layout inference takes from its uses (the
        accumulator's or a row's)."""
        return jnp.full(shape, value, dtype)

    def iota(self, shape: tuple[int, ...], dimension: int) -> jax.Array:
        """Indices along `dimension` of a tile or a row, laid out by inference from the values they meet. An
        iota nothing reads fails inference (Mosaic's iota is not pure, so it survives the cast it would
        have), so a template forms only the indices it reads."""
        return plgpu.broadcasted_iota(jnp.int32, shape, dimension)

    @staticmethod
    def folds_indices(columns: int) -> bool:
        """Whether a tile `columns` wide forms its indices (`iota`, `column_offsets`) as a register's constant
        beside the part its thread decides: Mosaic GPU's layouts (`TiledLayout.thread_idxs`) divide each
        register's linear index by the row's length and take its remainder, which split the constant off
        only where the length is a power of two. Elsewhere each entry's index is formed apart, an int32 tile
        each index a step reads (a causal forward at 64 x 192 on one consumer: its key places 96 ORs a
        thread and its mask 192 subtractions and compares, on an A100 and an H100 alike; width 64 on an
        A100 255 registers and 1056 bytes spilled, its lowered live set 417, where 64 x 128 folded them at
        206; width 128 on an H100 816 bytes spilled, its live set 367 against 175 at 64 x 128)."""
        return columns & (columns - 1) == 0

    @property
    @abc.abstractmethod
    def lane_columns(self) -> int:
        """The bits of a column's index in a tile in the accumulator's layout that a thread's lane and warp
        decide; the others are the element's place among the thread's registers (`column_offsets`)."""

    def column_offsets(self, shape: tuple[int, int]) -> tuple[jax.Array, jax.Array]:
        """Each column index of a `shape` tile (`iota(shape, 1)`) as `(base, offsets)`, their sum the index:
        `base` the bits the thread's lane and warp decide (`lane_columns`), equal across a thread's elements,
        so a row's bound takes it once (`offsets <= bound - base`); `offsets` the bits its register decides, a
        constant the compiler folds into each comparison (FA3's causal mask, against a column index formed
        whole per element). Any layout inference gives the iota keeps the sum exact."""
        columns = self.iota(shape, 1)
        return columns & self.lane_columns, columns & ~self.lane_columns

    def rows_of(self, ref) -> jax.Array:
        """A row vector read from `ref` (`[rows]`) in a row's layout."""
        return plgpu.load(ref, layout=self.row, optimized=False)

    def columns_of(self, ref) -> jax.Array:
        """A column vector read from `ref` (`[columns]`) in a column's layout (a tile's keys')."""
        return plgpu.load(ref, layout=self.column, optimized=False)

    def tile_of(self, ref) -> jax.Array:
        """A tile read from `ref` (`[rows, columns]`) in the accumulator's layout."""
        return plgpu.load(ref, layout=self.layout, optimized=False)

    def scalar_of(self, ref) -> jax.Array:
        """The scalar at `ref[0]` (a block's index a launch reads from global memory): a GPU thread's load, a
        TPU core's copy into its scalar memory."""
        return ref[0]

    def key_magnitudes(self, tile) -> jax.Array:
        """Each key's largest value magnitude, fp32, from a values tile `tile` `[keys, columns]` in shared
        memory: read transposed, its keys an accumulator tile's columns, and reduced over its own columns,
        so the vector is in a column's layout (`column`, a logits tile's keys'), as `algebra.room` reads
        it; no pass over the values outside the launch. In the values' own dtype until the reduction is
        done (a magnitude's maximum is exact in any), so the tile holds half an fp32 tile's registers."""
        values = plgpu.load(tile.transpose((1, 0)), layout=self.layout, optimized=False)
        return jnp.max(jnp.abs(values), axis=0).astype(jnp.float32)

    column_scratch: int
    """The fp32 entries of shared memory a consumer's `column_sums` takes: a reduction across the warps
    holding a tile's rows takes one register's partial sums at a time."""

    reduction_scratch: int = 0
    """The bytes of shared memory the family's kernels reserve for Mosaic GPU's implicit reductions across
    warps (its compiler parameters' `reduction_scratch_bytes`) where they reduce (`resources.Scope.reduces`):
    none where a reduction takes the consumer's own (`column_sums`)."""

    warpgroup_reduction_scratch: int = 0
    """The bytes of Mosaic GPU's reduction scratch a warpgroup of a kernel whose arithmetic reduces across
    warps implicitly takes as its share (`kernel`'s `reducing_warpgroups`, `resources.Scope`'s): a family
    whose kernels reserve no scratch by default (`reduction_scratch`)."""

    def column_sums(self, value: jax.Array, scratch, interpret: bool) -> jax.Array:
        """The sums of `value`'s columns `[columns]` in a column's layout, `value` `[rows, columns]` in the
        accumulator's layout: across the warps holding its rows, through the consumer's own `scratch`
        (`column_scratch` fp32 entries), as the block's other warpgroups reduce at once."""
        if interpret:
            return value.sum(axis=0)

        def sums(ctx, tile, scratch):
            del ctx
            return tile.reduce("add", axis=0, scratch=scratch)

        # Mosaic GPU's own reduction across the warps, through the given scratch; inline_mgpu passes the
        # launch context first and takes the tile and its scratch.
        summed: Callable[..., jax.Array] = plgpu.inline_mgpu(
            arg_types=(self.layout, plgpu.RefType()),
            return_type=plgpu.ShapeDtypeStruct((value.shape[1],), jnp.dtype(jnp.float32),
                                               layout=self.column))(sums)
        return summed(value, scratch)

    def as_row(self, vector: jax.Array) -> jax.Array:
        """`vector` `[rows]` in a row's layout, whatever its own."""
        return plgpu.layout_cast(vector, self.row)

    def as_tile(self, value: jax.Array) -> jax.Array:
        """`value` `[rows, columns]` in the accumulator's layout, whatever its own (a walk's carry, a TF32
        product's accumulator on mma.sync)."""
        return plgpu.layout_cast(value, self.layout)


class Transport[Allocated, Opened](abc.ABC):
    """An execution policy (design §6): who loads a program's operands and how (`streams`, `opened`,
    `lifetimes`, `stream` the producer's, `fed` the consumers' own, `walk` and `passed` the consumer's),
    the roles and their launch (`roles`, `warpgroups`, `kernel`, `scoped`), the persistent walk over tiles and
    its stores (`persist`, `stored`, `added`, `output_staging`), and a tile fixed up across programs
    (`fixed_up`). A `Family` composes it with an `Issuer`; its walks read the issuer's home (`accumulated`,
    `settled`). Generic in its streams' allocations (`Allocated`, `streams`') and its opened streams
    (`Opened`, `opened`'s)."""

    name: str
    """The family's name (`bundles.named`), which its refusals state."""

    bounded_loads: bool
    """Whether a load past an operand's extent reads zeros (TMA's) rather than past it (cp.async's), so an
    operand need not be padded to whole tiles."""

    loads_ahead: int = 1
    """The steps a walk keeps loading ahead of the one its products consume to hide its loads' latency: one
    where a producer's asynchronous copies run beside the consumers (TMA), more where the consumers issue
    them between their own products (`Mma`'s cp.async)."""

    fixes_up: bool = False
    """Whether a tile's steps may be walked by several programs in turn (`tail.Tail`), each leaving its fp32
    partial in global memory and the last to arrive at the tile's GPU-scope counter merging them
    (`fixed_up`, `partials_ready`)."""

    stores_through_registers: bool
    """Whether a program copies its stored output tiles out through its registers (no TMA store before
    Hopper: `mosaic_sm80.Mma.persist`), so its epilogue holds its share of the stored tile beside the
    accumulator (`resources.tile_registers`)."""

    copy_extent: int | None = None
    """The most elements one asynchronous copy moves along a dimension of its box after the copy's tiling
    (TMA's, 256): an operand block tiled in 8 rows by a swizzled row stays far below it at any tile its
    registers hold, and a one-dimensional side operand's block, untiled, is at most it
    (`grouped_matmul._Mosaic.fitting`); None where cp.async moves a block (mma.sync's)."""

    copies_in_pieces: bool = True
    """Whether a step's block may arrive in several copies (`streams`' `copies`, `Pieces`): TMA's each into
    its own rows of the slot, awaited at the slot's one barrier; cp.async's each a block of its pipeline's
    own, joined into one tile through registers (`mosaic_sm80.CpAsync.walk`): a paged call's key tile of
    several pages takes a copy a page."""

    pieces_of_views: bool = True
    """Whether a block's pieces (`copies_in_pieces`) may be pieces of any view (a TMA ring's: an fp32
    operand's parts, `Pieces`), where cp.async's are a table's runs alone (`mosaic.Rows`, a paged call's
    pages: `mosaic_sm80._run`)."""

    gathers_rows: bool = False
    """Whether its producer warpgroup gathers a step's rows from a table (a gathered stream's `Rows` of one
    row: `gathered.Ring`, its threads' cp.async copies; a ring family's); where not (mma.sync's pipeline, no
    producer of its own), a gathered call is not its launch's."""

    copies_vectors: bool = True
    """Whether its copies move a one-dimensional block untiled (TMA's), where cp.async spreads a copy over a
    warpgroup's threads, 128 elements at least ('CP_ASYNC requires the number of elements to be a multiple
    of 128', R9-gpu-a100): a step reads a side vector it does not copy from global memory instead."""

    adds_vectors: bool = True
    """Whether `added` takes a vector of fewer entries than a warpgroup's threads (TMA's reduction from shared
    memory), where Mosaic GPU refuses atomic stores from a layout that replicates its entries across
    threads (mma.sync's row and column layouts): a program writes such a sum to a place of its own
    instead."""

    adds_through_shared: bool = True
    """Whether `added` reads its value through its `smem` (TMA's reduction from a staged tile), which is then
    live wherever a walk adds; an atomic add from registers (cp.async's `red.global`) reads none, so a tile
    staged for the program's stores alone may be scoped after its walks (`scoped`'s `after_walks`)."""

    cluster: int = 1
    """The programs a cluster of the family's launch takes where they read a block in common (`kernel`'s
    `cluster`), each loading its share of the block multicast into every one's shared memory (`streams`'
    `multicast`): TMA multicast's pair; 1 where a load reaches its own program's shared memory alone
    (cp.async's)."""

    walks_scoped: bool = False
    """Whether its walks scope their streams apart, each in its own scope (`emit_pipeline`'s slots): a tile
    scoped after a walk then takes the larger of the two (`resources.shared_bytes`)."""

    feeds_itself: bool = False
    """Whether its consumers issue their own rings' loads, no producer beside them (`ConsumerFed`): every
    consumer walks every step, so a template may split a step's work between them (attention's dK and dV:
    a key block shared, each slot's queries split) where with a producer each takes steps of its own."""

    budgets_registers: bool = True
    """Whether a kernel budgets its threads' registers (`registers`): a GPU's families; a TPU core's does
    not (`mosaic_tpu.VmemRings`: its compiler spills values to VMEM), so a launch on it states no registers
    (`base.TemplateOp.registers` None) and a core runs one program of it (`resources.register_programs`)."""

    @abc.abstractmethod
    def registers(self, consumers: int, programs: int = 1) -> int:
        """The registers a thread of each of `consumers` consumer roles may hold, `programs` programs a
        multiprocessor at once; asked of a family that budgets them alone (`budgets_registers`)."""

    def warpgroups(self, consumers: int) -> int:
        """The warpgroups a program of `consumers` compute roles runs: theirs, and a producer's where the
        family streams from a role of its own (`roles`)."""
        return consumers

    def program_index(self, name: str) -> jax.Array:
        """The program's coordinate on its launch grid's axis `name` (`kernel`'s `grid_names`): Mosaic GPU
        names a grid's axes as a mesh's, read by `lax.axis_index`."""
        return lax.axis_index(name)

    @abc.abstractmethod
    def kernel(self, entry, *, out_type, grid: tuple, grid_names: tuple, consumers: int,
               interpret: bool, approx_math: bool = True, name: str | None = None,
               cluster: int = 1, cost: pl.CostEstimate | None = None, model_flops: int | None = None,
               reducing_warpgroups: int = 0, resident: int = 1, scalars: int = 0) -> Callable:
        """`entry` launched over `grid`, `consumers` compute roles a program beside the family's own,
        `resident` programs a multiprocessor at once (its plan's, `resources.LaunchPlan.programs`: its
        compile fits its registers to them, `compiler_params`, and its roles raise to their share), with
        `approx_math` the hardware's approximate exponential and reciprocal (`compiler_params`), named
        `name` where given (Mosaic names its kernel and its compile reports by it); each of `grid`'s
        programs a cluster of `cluster` along `CLUSTER` (at most the family's `cluster`); its `cost`, the
        hardware's work it states to XLA and to Dew's FLOP count (the custom call's `cost_estimate_json`),
        and `model_flops`, the model's share of it, where its own recomputation is the rest (`checked`); its
        `reducing_warpgroups` (the block's, where its arithmetic may reduce across warps implicitly) each
        reserving its share of reduction scratch (`warpgroup_reduction_scratch`); its first `scalars` operands
        tables of integers its programs read an element at a time (a grouped schedule's), which a GPU reads
        from global memory and a TPU from its scalar memory."""

    def scoped(self, body, *allocations, streams, after_walks: tuple = ()):
        """`body(*refs, streams=...)` with `allocations` scoped across the program's roles and its streams'
        (`streams`), and what it returns. With `after_walks`, allocations no walk reads, `body` takes
        `after` too: `after(fn)` runs `fn(*refs)` of them once a role's walks have ended, in a scope of their
        own where the family scopes its walks apart (`walks_scoped`: the walks' slots reused, as
        `resources.Scope.after` counts them), the program's otherwise."""
        if not after_walks:
            return self.program_scoped(body, *allocations, streams=streams)
        if self.walks_scoped:
            # The after-walks scope is the program's own kind (`program_scoped`: collective where its
            # consumers share one).
            def after(fn):
                return self.program_scoped(lambda *refs, streams: fn(*refs), *after_walks, streams=streams)

            return self.program_scoped(functools.partial(body, after=after), *allocations, streams=streams)
        held = len(allocations)

        def program(*refs, streams):
            return body(*refs[:held], streams=streams, after=lambda fn: fn(*refs[held:]))

        return self.program_scoped(program, *allocations, *after_walks, streams=streams)

    @abc.abstractmethod
    def program_scoped(self, body, *allocations, streams):
        """`body(*refs, streams=...)` with `allocations` scoped across the program's roles and its streams'
        (`streams`), and what it returns: the program's one scope (`scoped`'s)."""

    @abc.abstractmethod
    def roles(self, consumers: int, consumer, producer) -> None:
        """Each role of a program: `consumer(index)` the compute roles', `producer()` the loads' where the
        family streams from a role of its own."""

    def lifetimes(self, streams: Allocated, steps: int | None = None) -> tuple[tuple, tuple]:
        """A program's `streams` (`streams`') by lifetime (`resources.Scope`): those the program holds whole,
        and each walk's own where its walks scope their streams apart (`walks_scoped`), in no more slots than
        a walk's `steps` where they are known. The one statement of where a family's streams live, which
        the planner reads and its emission (`scoped`) allocates."""
        del steps
        return (streams,), ()

    @abc.abstractmethod
    def streams(self, blocks: dict, stages: int, consumers: int,
                multicast: tuple[str, ...] = (), swizzles: dict[str, int] | None = None,
                copies: dict[str, int] | None = None) -> Allocated:
        """The scoped allocations of a stream of each `blocks[name]` `(shape, dtype)` in `stages` slots read
        by `consumers` (`opened` once scoped), each stream `multicast` names one the programs of a cluster
        share, loaded once between them (`kernel`'s `cluster`), each `swizzles` names staged at its swizzle
        (`swizzle_for`; inference's own otherwise), each `copies` names a step's tile loaded in that many
        copies (`Pieces`; one otherwise)."""

    @abc.abstractmethod
    def opened(self, allocated: Allocated, views: dict | None = None) -> Opened:
        """The streams of `streams`' allocations, scoped, by name; their `views` where the family's walk
        loads them (`viewed`: each a ref and the block indices of a step's tile in it)."""

    def fed(self, streams, views: dict, lo, hi, wg, *, refills: bool = True):
        """What a consumer's walk of steps `lo` to `hi` reads (`walk`'s `streams`), the walk's `first` its
        `lo`: on a layout with a producer, the streams, which its producer loads from `views` (`stream`);
        on one whose consumers feed them (`ConsumerFed.fed`), a `Feed`. Its rings fresh, one walk's a scope:
        a second walk on them counts its steps from its own first against barriers already past them."""
        del views, lo, hi, wg, refills
        return streams

    def walk_own[R](self, streams, views: dict, own: tuple, streamed: tuple, wg,
                    walk: Callable[..., R], *, between=None) -> R:
        """Consumer `wg`'s walk of its own steps `own` (`lo`, `hi`) among its program's `streamed` ones,
        `walk(streams, lo, hi, first)` its walk of steps `lo` to `hi` counted from `first`: on a layout with a
        producer, the steps before and after its own passed (`passed`, `between()` after each), which its
        partners read; on one whose consumers feed their rings (`ConsumerFed.walk_own`), every streamed step
        over its own feed (`fed`), a step none of its own masked by its step. Returns `walk`'s value."""
        del views, wg
        lo, hi = own
        self.passed(streams, streamed[0], lo, first=streamed[0], between=between)
        walked = walk(streams, lo, hi, streamed[0])
        self.passed(streams, hi, streamed[1], first=streamed[0], between=between)
        return walked

    def run(self, allocated: Allocated, walk: Walk, interpret: bool) -> None:
        """`walk` on a program's roles: the producer's loads of its views into `allocated`'s streams (opened
        here) where the family has one, and its consumer's walk of them, fed by the producer or by its own
        loads (`fed`)."""
        streams = self.opened(allocated, walk.views)

        def consumer(wg):
            self.walk(self.fed(streams, walk.views, 0, walk.steps, wg), walk.stepped(), walk.carry, (),
                      interpret)

        self.roles(1, consumer, lambda: self.stream(streams, walk.views, 0, walk.steps))

    @abc.abstractmethod
    def stream(self, streams, views: dict, lo, hi, *, adds: tuple | None = None) -> None:
        """The producer's loads of steps `lo` to `hi` from `views` (`opened`'s), where the family has one,
        and with `adds` each step's sum the consumers stage added (`streamed`)."""

    @abc.abstractmethod
    def persist(self, streams, tiles: tuple[int, ...], schedule: Callable, issue: Callable, finish: Callable,
                *, consumers: int, axis: str, outputs: tuple, interpret: bool, in_flight: int = 0,
                unrolled: bool = False, accumulators: tuple = ()) -> None:
        """A persistent kernel's tiles, the grid `tiles` spread over the launch's programs along `axis`, its
        streams (`streams`, as `scoped` hands them) running on across them: `schedule(index)` gives a tile's
        `(active, steps, views, tile)`, whether it streams anything, its step count, each stream's view
        (`viewed`: a ref and the block indices of the tile's step `j`, counted from 0) and what its
        consumers are given. Each tile of each consumer role is issued, then finished, once for each value
        of `active`: `issue(index, tile, walk, active=...)` walks the tile's steps by `walk(step, start,
        shapes, *, initial=None)`, as `walk` does, a step's index the tile's own (where not active it takes
        none, its accumulators as they start), and returns the accumulators the walk returns;
        `finish(index, tile, accs, store, active=...)` stores them by `store(output, value, ref, place)`,
        a consumer's `value` of output `output` (`outputs`' `(shape, dtype)`) to `ref` at `place`,
        `store_rows`' `(row_start, first, stored, column, most)` and, where the stored rows come in whole
        multiples, `least`. A family whose products a role of their own issues runs `issue` in that role
        and `finish` in its consumers'; a register family runs both in each consumer. The family owns the
        loads, the step counter across tiles, the streams' drain, the outputs' staging and the stores'
        completion. A cluster's programs run the same tiles: where a stream is multicast (`streams`),
        `schedule` gives each of them the same `active`, `steps` and view of it. `walk`'s `in_flight` and
        `unrolled` are its consumers' walks'. `accumulators` are the shapes of the fp32 accumulators a
        tile's walk issues into, which a family holding them outside its consumers' registers allocates
        (tcgen05's tensor memory, read out by `Issuer.chunks`); a register family's walk carries its own."""

    @abc.abstractmethod
    def walk[C](self, streams, segments, start: C, shapes: tuple, interpret: bool, *,
                first: jax.Array | int = 0, in_flight: int = 0, initial: tuple | None = None,
                unrolled: bool = False) -> tuple[C, tuple[jax.Array, ...]]:
        """A consumer's walk of `segments`, each `(step, lo, hi)`, `step(index, slots, carry, accumulators)
        -> (carry, accumulators)` given each stream's slots (`Slots`, the streams' steps counted from
        `first`), from `start`, with fp32 accumulators of `shapes` its products issue into in place, zero
        or their `initial` values (in the accumulator's layout; None, zero). `in_flight` steps' products stay
        running past their step (0 or 1): their slots go back once they are done, a step later, so the next
        step's operands are waited for while they run. A ring's `unrolled` walk takes a rotation of its
        slots a trip, each step at a slot it knows (`mosaic.stepping`): for a step small beside its waits (a
        GEMM's), where a ring's slots are few.
        Returns the carry and the accumulators' values."""

    @abc.abstractmethod
    def passed(self, streams, lo, hi, *, first: jax.Array | int = 0, between=None) -> None:
        """A consumer's steps `lo` to `hi` it reads none of, each load awaited and released as a reading
        consumer's (counted from `first`); `between()` after each step."""

    @abc.abstractmethod
    def loaded(self, copies, *, issued: jax.Array | bool = True, axes: tuple | None = None) -> None:
        """Each `(view, ref, barrier)` copied whole into its shared-memory `ref` (its barrier one of
        `load_barriers`'): where `issued` is a role's own (a traced predicate), that role loads `ref` for
        every role that shares it and each caller awaits it; by default each caller loads its own. `axes`,
        each copy's launch axes as its tile holds them (`templates.axes.held`), say which of the
        accumulator's axes a vector follows, where a family stages its loads through registers in the
        accumulator's layouts (`mosaic_sm80.Mma`); a vector whose axes go unstated follows its columns."""

    def shares_rows(self, rows: int, consumers: int) -> bool:
        """Whether a step's block of `rows` rows splits into the copies `consumers` consumers each make of it:
        always where a producer loads it whole (TMA's rings)."""
        del rows, consumers
        return True

    def load_barriers(self, count: int) -> pallas_core.ScratchShape:
        """The scoped allocation of `count` whole loads' completions (`loaded`'s barriers), which a program's
        allocations hold beside its streams (`scoped`, `resources.Scope`): a backend's own kind of scoped
        allocation (a GPU's barriers, a TPU's DMA semaphores)."""
        return plgpu.Barrier(num_barriers=count)

    def staging(self, name: str, shaped) -> dict:
        """The scoped allocations of each consumer's staging of each operand a tile reads whole (`shaped`:
        its tile's shape and its operand), one for each of up to two consumers, in the memory a program
        stages in (`staged_memory`), and the completion of its load (`loaded`'s barriers), as
        `{name}_smems` and `{name}_barriers`."""
        shaped = list(shaped)
        return {f'{name}_smems': [self.staged_memory((2, *shape), gmem.dtype) for shape, gmem in shaped],
                f'{name}_barriers': [self.load_barriers(2) for _ in shaped]}

    def staged_memory(self, shape: tuple[int, ...], dtype, swizzle: int | None = None):
        """A scoped allocation of `shape` in the memory a program stages tiles in: a GPU's shared memory, its
        rows at `swizzle` bytes (`staged`: the widest that fits where None, plain where `UNSWIZZLED`); a
        TPU's VMEM, which its compiler tiles."""
        return staged(shape, dtype, swizzle)

    def meeting(self, consumers: int) -> plgpu.Barrier:
        """The scoped allocation two consumers `meet` at: two barriers of `consumers` arrivals each."""
        return plgpu.Barrier(num_arrivals=consumers, num_barriers=2)

    def meet(self, halves) -> None:
        """The two consumers wait for each other at `halves` (`meeting`'s), two barriers in turn: neither
        arrives at a barrier's next phase before the other has seen its last, however the meets fall (the
        interpreter's invariant, and the hardware's parity waits)."""
        for barrier in (halves.at[0], halves.at[1]):
            plgpu.barrier_arrive(barrier)
            plgpu.barrier_wait(barrier)

    def schedule(self, consumers: int) -> pallas_core.ScratchShape:
        """The scoped allocation of `consumers`' turns on the tensor cores (`turn`): one barrier each of
        them arrives at (a TPU core's, one role a core, a semaphore it never signals)."""
        return plgpu.Barrier(num_arrivals=consumers)

    def turn(self, schedule, consumers: int, *, arrive: bool = True, wait: bool = True) -> None:
        """A consumer's turn on the tensor cores through `schedule` (`schedule`'s allocation), where a
        program has more than one: its arrival unless not `arrive`, then its wait for its partner's unless
        not `wait` (FlashAttention-3's ping-pong of products and softmax)."""
        if consumers > 1 and arrive:
            plgpu.barrier_arrive(schedule)
        if consumers > 1 and wait:
            plgpu.barrier_wait(schedule)

    def exchange(self, consumers: int) -> list:
        """An exchange's allocations (`Exchange`): its two barriers of `consumers` arrivals each, its staged
        tiles' and its reads' (`schedule`)."""
        return [self.schedule(consumers), self.schedule(consumers)]

    def output_staging(self, outputs: tuple, consumers: int) -> list:
        """The shared memory `persist` stages each of `outputs` (`(shape, dtype)`, a tile's) in, which a
        template's allocations count as its own (`resources.Scope`): the whole tile, where a family stages
        it whole."""
        del consumers
        return [plgpu.SMEM(shape, dtype) for shape, dtype in outputs]

    def persist_scratch(self, consumers: int) -> tuple[tuple, int]:
        """What `persist` scopes in shared memory besides its outputs' staging, which a template's
        allocations count (`resources.Scope`): its barriers, and its address words
        (`resources.Scope.addresses`); none where its accumulators live in registers."""
        del consumers
        return (), 0

    def published(self) -> None:
        """A consumer's writes to shared memory made visible to the readers that read it asynchronously
        (warpgroup MMA, TMA), before any of them reads it: a family whose products read shared memory
        through its consumers' own loads (mma.sync's) has nothing to publish."""
        plgpu.commit_smem()

    @abc.abstractmethod
    def stored(self, parts, views, *, smem, wait: bool = True) -> None:
        """Each of `parts` written whole to its view of `views`, through `smem` where the family stores so.
        Unless `wait`, the last copy out of `smem` stays in flight as one store group, so the caller's
        next work overlaps it, and the caller owns `smem`'s reuse: it waits (`store_wait`) before writing
        `smem` again, an earlier deferred store's or an add's (`added`) reads of it included, so stores
        into other tiles stay in flight past the call (`store_wait(stores - 1)` keeps one a tile). A call
        of several parts still waits for each part's reads before writing the next, every earlier store
        group with it."""

    def store_wait(self, pending: int) -> None:
        """Waits until at most `pending` of this thread's deferred stores (`stored(wait=False)`) still read
        their shared tiles: a family that stores through registers has none."""
        del pending

    @abc.abstractmethod
    def added(self, value: jax.Array, view, interpret: bool, *, smem, pending: int = 0) -> None:
        """`value` added into its view (fp32, in global memory) through `smem` (a chunk of its columns at a
        time where `smem` is narrower, `chunked`), atomically against every other role's adds into it;
        interpreted, a read and a write of the view, atomic against other
        programs' alone (the interpreter runs a program's roles at once and a grid's programs in turn).
        `smem` is written once at most `pending` of the program's earlier adds may still be reading theirs
        (each through a buffer of its own: a caller's slots of a stage in turn), 0 its own add's last."""

    def partials_ready(self, shape: tuple[int, ...], interpret: bool):
        """The semaphores of `shape` a tile's partials are signalled on across programs (`fixed_up`); none
        interpreted, where the programs' order stands in for them (`tail.ordered`)."""
        return None if interpret else pl.get_global(plgpu.SemaphoreType.REGULAR(shape))

    def ticketed(self, ready, ticket, body: Callable[[], None]) -> None:
        """`body`, an add into global memory (`added`), in its turn among the programs adding there: once
        `ticket` of them have signalled `ready` (one of `partials_ready`'s, counting the adds done), then its
        writes complete and `ready` signalled, so the adds land in a fixed order (a deterministic sum,
        FlashAttention 3's dQ semaphore); `body` alone where `ready` is None (interpreted, the programs'
        order standing in)."""
        if ready is None:
            body()
            return
        # Wait without taking: the semaphore counts the adds done, each program's turn the count before it.
        plgpu.semaphore_wait(ready, ticket, decrement=False, memory_scope="gpu")
        body()
        # Its writes complete in global memory before the next turn (`added_settled`).
        self.added_settled()
        plgpu.semaphore_signal(ready, 1, memory_scope="gpu")

    @abc.abstractmethod
    def added_settled(self) -> None:
        """A program's adds into global memory complete (`added`), past any copy of its still in flight (a
        TMA reduction's); a family that adds from its registers has none in flight."""

    def fixed_up(self, summed, partials, ready, program, first, gathered, store,
                 combine=operator.add, firsts=None) -> None:
        """A consumer's share of a tile that programs `first` to `gathered` (exclusive) each walk a part of
        (a balanced tail's, `tail.Tail`; a split walk's), `summed` from this program's steps: left in
        `partials` (each program's, this consumer's; the tile's first program's in `firsts` where given, a
        balanced tail's program first in one tile having left its partial of the tile before in its own
        slot), and the program that arrives last at the tile's counter (`arrived`, on `ready`'s semaphore
        of its last program, `partials_ready`) combines every partial in the programs' order, the same sum
        whichever arrives last, and stores it (`store`). No program waits on another, which the GPU may
        never have scheduled (row 4732). `combine(running, partial)` is the algebra's merge of two partials
        of one tile (a sum's addition by default; a split walk's softmax merge, a scan's composition), which
        the work's declaration supplies. `summed` may be a pytree (a merge's partial of several leaves):
        `partials` is then a ref a leaf, in its structure. Interpreted (`ready` None), the tile's first
        program combines, which the interpreter walks after every other (`tail.ordered`)."""
        opening = partials if firsts is None else firsts

        def left(refs) -> None:
            # Unoptimized: a partial's leaf may move less than a cache line (a 16-row decode's statistics,
            # 64 bytes), which an optimized global transfer refuses.
            jax.tree.map(lambda leaf, ref: plgpu.store(ref.at[program], leaf, optimized=False), summed, refs)

        def read(refs, other):
            # The partials' structure read off the running value's: a ref may itself be a tree.
            return jax.tree.map(lambda _, ref: plgpu.load(ref.at[other], optimized=False), summed, refs)

        def combined(running) -> None:
            store(lax.fori_loop(first + 1, gathered,
                                lambda other, running: combine(running, read(partials, other)), running))

        if ready is None:
            pl.when(program != first)(lambda: left(partials))
            pl.when(program == first)(lambda: combined(summed))
            return
        pl.when(gathered - first == 1)(lambda: store(summed))

        @pl.when(gathered - first > 1)
        def _():
            pl.when(program != first)(lambda: left(partials))
            pl.when(program == first)(lambda: left(opening))
            last = arrived(ready.at[gathered - 1], gathered - first)
            # The first program's partial its own value where it arrives last, so the running value keeps
            # its accumulator's layout whichever program merges.
            first_partial = jax.tree.map(lambda own, left: jnp.where(program == first, own, left), summed,
                                         read(opening, first))
            pl.when(last == 1)(lambda: combined(first_partial))


class Family[Allocated, Opened, Accumulator](Issuer[Accumulator], Transport[Allocated, Opened]):
    """An instruction family's bundle (design §6, one mainloop per overlap structure): an issuer and its
    accumulator home (`Issuer`) composed with an execution policy (`Transport`), so a mainloop holds no
    instruction, layout or barrier of its own. A policy is a family whose issuer half the issuer composed
    with it gives, its walks reading that issuer's home (`RingFamily`, `ConsumerFed`,
    `mosaic_sm80.CpAsync`); a bundle composes the two (`mosaic_sm90.Wgmma`: `WgmmaIssuer` on a
    `RingFamily`). It holds what both halves read: its name, its compute roles (`consumers`), a register
    step, its shared memory, its tiles' column unit and depths, a register value's way to a product
    (`staged`) and the programs a multiprocessor holds (`programs`). Generic in its streams' allocations
    (`Allocated`, `streams`'), its opened streams (`Opened`, `opened`'s) and the accumulator its products
    issue into in place (`Accumulator`: a warpgroup accumulator ref, or a register array)."""

    name: str

    consumers: int
    """The compute roles a program may take (sm90 two warpgroups, sm80 one)."""

    @property
    def register_step(self) -> int:
        """The allocator's registers per thread in one step (`generation.Multiprocessor.register_step`,
        setmaxnreg's eight-register granularity). A compile gate tests admits at their actual cap, refusals
        one such step below it."""
        return resources.facts().register_step

    @abc.abstractmethod
    def column_unit(self, dtype) -> int:
        """The columns a tile's width steps by for operands of `dtype`: the swizzle a block's rows are
        staged in, or the row its stores lay over a role's threads."""

    @abc.abstractmethod
    def depths(self, dtype) -> tuple[int, ...]:
        """The depths a step may take for operands of `dtype`, widest first: the swizzles its products
        read a block's rows at."""

    def staged(self, value: jax.Array, buffer) -> Ref:
        """`value` in the shared-memory tile `buffer`, in its dtype, for a product to read: a register
        operand's way to an MMA. Mosaic GPU brackets a store with the warpgroup's barriers (its automatic
        barriers), so every warp's reads of `buffer` are done before it and its writes seen after; the
        family publishes them to its asynchronous readers (`published`)."""
        buffer[...] = value.astype(buffer.dtype)
        self.published()
        return buffer

    def converted(self, codes, into, interpret: bool, *, row_scale: jax.Array | None = None) -> Ref:
        """A stage of quantized codes (`codes`, int8 or an FP8 in shared memory, `[rows, depth]`, a consumer's
        rows) as the tile a product reads (`into`, a shared tile of its 16-bit dtype): each code cast
        exactly, each row scaled by `row_scale` (fp32 `[rows]` in registers, a per-token scale) in fp32
        before its one rounding, written and published as `staged` writes a register operand."""
        if not converts(codes.dtype):
            raise ValueError(f"this device converts no {jnp.dtype(codes.dtype).name} codes in registers "
                             "(`converts`)")
        if interpret:
            value = codes[...].astype(jnp.float32)
            if row_scale is not None:
                value = value * row_scale[:, None]
        else:
            # Read in the accumulator's layout, whose rows a row vector broadcasts along (`as_row`); an 8-bit
            # stage takes no ldmatrix. Integer codes to fp32 by the exponent's magic number in integer
            # arithmetic (exact below 2**22), no integer-to-float conversion: an int-to-fp32 cast and the
            # cast to 16 bits after it folded into Mosaic's packed int8-to-bf16 conversion, sm_90's
            # sub.bf16x2, which an A100's ptxas refused (dd-q8).
            loaded = plgpu.load(codes, layout=self.layout, optimized=False)
            if jnp.issubdtype(codes.dtype, jnp.integer):
                biased = loaded.astype(jnp.int32) + jnp.int32(MAGIC_INT)
                value = lax.bitcast_convert_type(biased, jnp.float32) - jnp.float32(MAGIC_FLOAT)
            else:
                value = loaded.astype(jnp.float32)
            if row_scale is not None:
                value = value * self.as_row(row_scale)[:, None]
        if interpret:
            into[...] = value.astype(into.dtype)
            return into
        plgpu.store(into, value.astype(into.dtype), optimized=False)
        self.published()
        return into

    def programs(self, consumers: int, registers: int | None = None) -> int:
        """The most programs of one launch a multiprocessor's registers hold at once, each `consumers`
        consumer roles holding `registers` (None: the family's budget, `registers`): at warpgroup MMA's
        budget, one, its consumers raised to the register file's share beside its producer
        (`resources.consumer_registers`). Each stages its share of the multiprocessor's shared memory
        (`resources.program_shared`), which may hold no tile at that many."""
        return resources.register_programs(self, consumers, registers)


class RingFamily[Accumulator](Family[dict, dict[str, Ring], Accumulator]):
    """A family whose loads a producer warpgroup issues by TMA into `Ring`s, beside one or two consumer
    warpgroups at `setmaxnreg` budgets (`roles`): sm90's and sm100's. Its walk acquires each step's slots as
    the step reads them and releases them once its products are done (`Slots`). Its products read their
    operands from the rings, a block's rows staged at the widest swizzle, 128 bytes. Composed with an issuer
    (`Issuer`) its walks read the home of (`accumulated`, `settled`)."""
    bounded_loads = True
    fixes_up = True
    gathers_rows = True
    # Mosaic GPU's default for a warpgroup (its CompilerParams' 128 x 4 x 4 bytes).
    warpgroup_reduction_scratch = 2048
    stores_through_registers = False
    copy_extent = 256

    def registers(self, consumers: int, programs: int = 1) -> int:
        return resources.consumer_registers(consumers, programs)

    def column_unit(self, dtype) -> int:
        return swizzle_columns(dtype, SWIZZLE_BYTES)

    def depths(self, dtype) -> tuple[int, ...]:
        return (swizzle_columns(dtype, SWIZZLE_BYTES),)

    def warpgroups(self, consumers: int) -> int:
        return consumers + 1

    def kernel(self, entry, *, out_type, grid: tuple, grid_names: tuple, consumers: int,
               interpret: bool, approx_math: bool = True, name: str | None = None,
               cluster: int = 1, cost: pl.CostEstimate | None = None, model_flops: int | None = None,
               reducing_warpgroups: int = 0, resident: int = 1, scalars: int = 0) -> Callable:
        # Its programs read a table from global memory as any operand.
        del scalars
        if cluster > self.cluster:
            raise ValueError(f"{self.name} launches a cluster of at most {self.cluster}, not {cluster}")
        return kernel(entry, out_type=out_type, grid=grid, grid_names=grid_names,
                      warpgroups=self.warpgroups(consumers), family=self.name, interpret=interpret,
                      name=name, approx_math=approx_math, cluster=cluster, cost=cost,
                      model_flops=model_flops, resident=resident,
                      # The issuer's own reductions across warps (mma.sync's column sums take the kernel's
                      # scratch, `reduction_scratch`), and each implicitly reducing warpgroup's share.
                      reduction_scratch=(self.reduction_scratch
                                         + self.warpgroup_reduction_scratch * reducing_warpgroups))

    def program_scoped(self, body, *allocations, streams):
        return pl.run_scoped(lambda *refs, **rings: body(*refs, streams=rings), *allocations, **streams,
                             collective_axes="wg")

    def roles(self, consumers: int, consumer, producer) -> None:
        roles(lax.axis_index("wg"), consumers, consumer, producer)

    def streams(self, blocks: dict, stages: int, consumers: int, multicast: tuple[str, ...] = (),
                swizzles: dict[str, int] | None = None, copies: dict[str, int] | None = None,
                gathered: tuple[str, ...] = ()) -> dict:
        from dew.nn.kernels.templates import gathered as gathered_rows  # it builds on `Ring`

        return {name: gathered_rows.ring(shape, dtype, stages, consumers=consumers) if name in gathered
                else ring(shape, dtype, stages, consumers=consumers, multicast=name in multicast,
                          swizzle=(swizzles or {}).get(name), copies=(copies or {}).get(name, 1))
                for name, (shape, dtype) in blocks.items()}

    def opened(self, allocated: dict, views: dict | None = None) -> dict[str, Ring]:
        from dew.nn.kernels.templates import gathered  # it builds on `Ring`

        del views
        # A gathered stream's load barriers are shared memory of its own (`gathered.ring`).
        return {name: (Ring if isinstance(scoped['full'].dtype, plgpu_core.BarrierType) else gathered.Ring)(
                    **scoped) for name, scoped in allocated.items()}

    def stream(self, streams, views: dict, lo, hi, *, adds: tuple | None = None) -> None:
        rings = tuple(streams.values())

        def tiles(index):
            return tuple(ring.viewed(views[name], index) for name, ring in zip(streams, rings, strict=True))

        streamed(rings, tiles, lo, hi, adds=adds)

    def output_staging(self, outputs: tuple, consumers: int) -> list:
        # Each consumer's share of each output in two chunk buffers, not the whole tile (`output_staging`).
        return [plgpu.SMEM((consumers, output_staging(shape[1], (dtype,))[1], shape[0],
                            output_staging(shape[1], (dtype,))[0]), dtype)
                for shape, dtype in outputs]

    def persist(self, streams, tiles: tuple[int, ...], schedule: Callable, issue: Callable, finish: Callable,
                *, consumers: int, axis: str, outputs: tuple, interpret: bool, in_flight: int = 0,
                unrolled: bool = False, accumulators: tuple = ()) -> None:
        del accumulators  # its consumers' walks carry their own
        # The producer loads every tile's steps into the rings at a counter running across the tiles, so
        # a tile's first loads go out while the consumers store the last one (`_tiled`).
        rings = self.opened(streams)
        for ring_ in rings.values():
            ring_.initialized()
        staging = self.output_staging(outputs, consumers)

        def producer():
            loaded, _ = tiled(tiles, schedule, functools.partial(loads, rings), axis=axis)
            for ring in rings.values():
                ring.drain(loaded)

        def consuming(smems):
            return self._tiled(rings, tiles, schedule, issue, finish, smems, axis=axis, interpret=interpret,
                               in_flight=in_flight, unrolled=unrolled)

        pl.run_scoped(lambda *smems: self.roles(consumers, consuming(smems), producer), *staging,
                      collective_axes="wg")

    def _tiled(self, rings: dict[str, Ring], tiles: tuple[int, ...], schedule: Callable, issue: Callable,
               finish: Callable, smems, *, axis: str, interpret: bool, in_flight: int,
               unrolled: bool) -> Callable:
        """`persist`'s consumers: each walks its tiles' steps at a counter running across the tiles
        (`_tile_walk`) and stores its outputs (`_stores`); the rings are left as `_ended` leaves them."""
        def run(index):
            store = self._stores(smems, index)

            def walking(counter, active, steps, views):
                # `counter`: the step every ring is at.
                return self._tile_walk(rings, counter, jnp.where(active, steps, 0), views, index,
                                       interpret=interpret, in_flight=in_flight, unrolled=unrolled)

            ended = consumed(self, tiles, schedule, issue, finish, store, index, walking, axis=axis)
            self._ended(rings, ended, index)
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)

        return run

    def _stores(self, smems, index) -> Callable:
        """Consumer `index`'s `store(output, value, ref, place)` (`persist`'s): its share of each output
        staged in chunks (`output_staging`: two buffers of `staged_columns`, `smems`), written from its
        issuer's layout (`Issuer.written`), and stored by TMA, waiting only for a buffer's last store to read
        it before it writes it again."""
        def store(output, value, ref, place):
            smem = smems[output].at[index]
            buffers, width = smem.shape[0], smem.shape[-1]
            row_start, first, stored, column, *ladder = place
            for chunk, (start, piece) in enumerate(column_chunks(value, width)):
                if chunk == 0 or chunk >= buffers:
                    # The buffer's last store (the last tile's, at its first chunk) has read it.
                    plgpu.wait_smem_to_gmem(0, wait_read_only=True)
                buffer = smem.at[chunk % buffers]
                staged_rows(buffer, piece, ref, row_start, first, stored, column + start, *ladder,
                            write=self.written)

        return store

    def _tile_walk(self, rings: dict[str, Ring], counter, steps, views: dict, wg, *, interpret: bool,
                   in_flight: int, unrolled: bool) -> Callable:
        """`persist`'s walk of one tile's `steps` for consumer `wg` (its consumer's `walk(step, start,
        shapes, initial=)`), the rings at `counter`, each step's index counted from the tile's first, over
        the streams the layout gives the tile (`_tile_streams`)."""
        streams = self._tile_streams(rings, counter, steps, views, wg)

        def walk(step, start, shapes, *, initial=None):
            def stepped(at, slots, carry, accumulators):
                return step(at - counter, slots, carry, accumulators)

            return self.walk(streams, ((stepped, counter, counter + steps),), start, shapes, interpret,
                             in_flight=in_flight, initial=initial, unrolled=unrolled)

        return walk

    def _tile_streams(self, rings: dict[str, Ring], counter, steps, views: dict,
                      wg) -> "dict[str, Ring] | Feed":
        """The streams a tile's walk reads (`_tile_walk`): the rings, which the producer loads from
        `views`."""
        del counter, steps, views, wg
        return rings

    def _ended(self, rings: dict[str, Ring], end, wg) -> None:
        """Consumer `wg`'s rings after `persist`'s last tile, at counter `end`: the producer drains them."""
        del rings, end, wg

    def walk[C](self, streams, segments, start: C, shapes: tuple, interpret: bool, *,
                first: jax.Array | int = 0, in_flight: int = 0, initial: tuple | None = None,
                unrolled: bool = False) -> tuple[C, tuple[jax.Array, ...]]:
        return self._walked(streams, segments, start, shapes, interpret, first=first, in_flight=in_flight,
                            initial=initial, unrolled=unrolled)

    def _walked[C](self, streams: dict[str, Ring], segments, start: C, shapes: tuple, interpret: bool, *,
                   first: jax.Array | int = 0, in_flight: int = 0, initial: tuple | None = None,
                   freed: Callable | None = None, unrolled: bool = False) -> tuple[C, tuple[jax.Array, ...]]:
        """`walk`, `freed(counter)` run after each step's slots are given back (step `counter`'s). Each
        segment walks its steps a step a trip or, `unrolled`, a rotation of the rings (`stepping`)."""
        stages = min(ring.stages for ring in streams.values())
        if not 0 <= in_flight < stages:
            raise ValueError(f"a walk keeps fewer steps' products in flight than its rings' {stages} slots, "
                             f"none or some, not {in_flight}")
        def given(turn: Turn, *, release: bool = False) -> None:
            # `turn`'s slots given back (`release`: here, a step its products ran past), then `freed`.
            if release:
                released(streams, turn)
            if freed is not None:
                freed(turn.counter)

        def turned(step, lo, index, turn: Turn, carry, accumulators, *, before: int = 0):
            slots = Slots(streams, turn)
            groups = commits()
            carry, accumulators = step(index, slots, carry, accumulators)
            # The step's product groups (its completion tokens): a step `in_flight` back is done once no
            # more than this step's groups times `in_flight` are pending, however many products a step
            # issues (a walk's steps are one function each, their groups alike).
            groups = commits() - groups
            if not in_flight:
                # The step's products done before its slots are given back.
                self.settled()
                slots.done()
                given(turn)
                return carry, accumulators
            # The step's products run on while the next steps' operands are waited for: the step
            # `in_flight` back, done now, gives back its slots.
            slots.held()
            self.settled(in_flight * groups)
            if before >= in_flight:
                # The step `in_flight` back is the range's too: a rotation's, past its first steps.
                given(turn.back(in_flight, stages), release=True)
            else:
                pl.when(index >= lo + in_flight)(lambda: given(turn.back(in_flight, stages), release=True))
            return carry, accumulators

        def drained(lo, hi) -> Callable[[], None]:
            # A segment's last `in_flight` steps' products done and their slots given back, oldest first,
            # before the next segment's steps wait for those slots: its own releases start at its own `lo`.
            def drain():
                self.settled()
                for back in reversed(range(in_flight)):
                    pl.when(hi - back > lo)(lambda back=back: given(Turn.of(hi - 1 - back - first, stages),
                                                                    release=True))

            return drain

        loops: list[Loop] = []
        for step, lo, hi in segments:
            run = stepping(functools.partial(turned, step, lo), lo, hi, first, stages, unrolled=unrolled)
            loops += [*run[:-1], run[-1]._replace(then=drained(lo, hi) if in_flight else None)]
        return self.accumulated(tuple(loops), start, shapes, interpret, initial)

    def passed(self, streams, lo, hi, *, first: jax.Array | int = 0, between=None) -> None:
        passed(tuple(streams.values()), lo, hi, first=first, between=between)

    def loaded(self, copies, *, issued: jax.Array | bool = True, axes: tuple | None = None) -> None:
        del axes  # copied by TMA, bytes whatever their layout
        loaded(copies, issued)

    def stored(self, parts, views, *, smem, wait: bool = True) -> None:
        # An add still reading `smem` (`added`) done first, unless the caller defers and so waits itself
        # (`Family.stored`): waiting here would drain its other tiles' deferred stores too.
        if wait:
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)
        stored(parts, views, smem=smem, wait=wait)

    def store_wait(self, pending: int) -> None:
        plgpu.wait_smem_to_gmem(pending, wait_read_only=True)

    def added(self, value: jax.Array, view, interpret: bool, *, smem, pending: int = 0) -> None:
        if interpret:
            # Mosaic GPU's interpreter (jax 0.11.2) takes no TMA reduction.
            view[...] = view[...] + value
            return
        # A TMA reduce from `smem` a chunk of its columns at a time (`chunked`), the reads of it the add
        # before made done first (`pending` of the earlier ones' through their own buffers still going on);
        # its writes go on.
        for index, (piece, place) in enumerate(chunked(value, view, smem)):
            plgpu.wait_smem_to_gmem(pending if index == 0 else 0, wait_read_only=True)
            smem[...] = piece
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(smem, place, reduction_op="add")

    def added_settled(self) -> None:
        # Its adds are TMA reductions (`added`), each in flight past its issue until waited.
        plgpu.wait_smem_to_gmem(0)


def arrived(counter, contributors) -> jax.Array:
    """1 where this consumer warpgroup is the last of `contributors` to arrive at `counter` (a GPU-scope
    semaphore each arrives at once, `Transport.fixed_up`'s), else 0, an int32 scalar: one of its threads adds
    one to the count by an atomic that releases the warpgroup's writes before it (a barrier first, so every
    thread's are) and acquires every earlier arrival's; the last, which reads `contributors - 1` before its
    add, sets the count back to zero for the next launch, and a barrier's reduction tells every thread of
    the warpgroup which it is. One atomic and no wait: a program spinning on a peer the GPU never scheduled
    hangs (row 4732), where an atomic ticket's last arriver merges (CUTLASS's and Triton's split-K
    semaphores alike)."""
    scalar = plgpu.Layout.WG_SPLAT(())

    @plgpu.inline_mgpu(arg_types=(plgpu.RefType(), scalar),
                       return_type=plgpu.ShapeDtypeStruct((), jnp.dtype(jnp.int32), layout=scalar))
    def ticket(ctx, ref, count):
        del ctx
        i32 = ir.IntegerType.get_signless(32)
        mgpu_utils.warpgroup_barrier()
        thread = arith.remui(mgpu_utils.thread_idx(), mgpu_utils.c(resources.THREADS, i32))
        before_last = arith.subi(count.registers.item(), mgpu_utils.c(1, i32))
        last = mgpu_utils.inline_ptx(
            f"""{{
            .reg .pred elected, last;
            .reg .b32 before;
            setp.eq.u32 elected, $1, 0;
            mov.b32 before, 0;
            @elected atom.acq_rel.gpu.global.add.u32 before, [$2], 1;
            setp.eq.and.u32 last, before, $3, elected;
            @last st.relaxed.gpu.global.u32 [$2], 0;
            bar.red.or.pred last, $4, {resources.THREADS}, last;
            selp.u32 $0, 1, 0, last;
            }}""",
            thread, mgpu_utils.memref_ptr(ref), before_last, mgpu_utils.warpgroup_barrier_idx(sync=False),
            result_types=i32, has_side_effects=True, convergent=True)
        return mgpu_fragments.FragmentedArray.splat(last, (), is_signed=True)

    return ticket(counter, jnp.asarray(contributors, jnp.int32))


def summing(shape: tuple[int, ...], consumers: int) -> dict:
    """`summed_across`' allocations for `pl.run_scoped` (collective across the consumers, in a scope after
    the walk's, whose shared memory it reuses): every consumer's fp32 part but the first's, the barrier they
    arrive at once theirs is written, and the one the first arrives at once it has read them; none for one
    consumer, whose part is the sum."""
    if consumers == 1:
        return {}
    return {'parts': plgpu.SMEM((consumers - 1, *shape), jnp.float32),
            'ready': plgpu.Barrier(num_arrivals=consumers - 1, num_barriers=1),
            'read': plgpu.Barrier(num_arrivals=1, num_barriers=1)}


def summed_across(value, scratch: dict, wg, consumers: int, store: Callable) -> None:
    """The consumers' fp32 parts of one tile (each consumer's `value`) added and stored once: every consumer
    but the first leaves its part in `scratch` (`summing`'s, scoped) and arrives at its `ready`, then waits
    until the first has read every part (`read`), so the scratch serves the next sum; the first waits, adds
    them in the consumers' order (one sum, the same each run), arrives at `read` and stores the total
    (`store(total)`). One consumer stores its own."""
    if consumers == 1:
        store(value)
        return
    parts, ready, read = scratch['parts'], scratch['ready'], scratch['read']

    @pl.when(wg > 0)
    def _():
        parts[wg - 1] = value
        plgpu.barrier_arrive(ready.at[0])
        plgpu.barrier_wait(read.at[0])

    @pl.when(wg == 0)
    def _():
        plgpu.barrier_wait(ready.at[0])
        total = value
        for other in range(consumers - 1):
            total = total + parts[other]
        plgpu.barrier_arrive(read.at[0])
        store(total)


def unscaled(parts) -> None:
    """A product's parts issued in place take no power of two back (bf16's, `cotangent_parts`): none a
    `product.Scaled`."""
    if any(isinstance(part, product.Scaled) for part in parts):
        raise NotImplementedError("a scaled part's product is no in-place accumulation (fp16's)")


def unscaled_part(part) -> jax.Array:
    """`unscaled` of one part taken as it is formed: its operand."""
    unscaled((part,))
    return part
