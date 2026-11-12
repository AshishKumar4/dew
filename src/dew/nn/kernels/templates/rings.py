"""The Mosaic GPU toolkit's rings (docs/design/kernel-templates.md): a stream's shared-memory slots a
producer loads by TMA and its consumers read (`Ring`, allocated by `ring`), a step's turn round them
(`Turn`), the views a step's block is copied from (`viewed`: `Pieces`, `Rows`, `Paired`, `Split`), a
producer's loads and drain (`loads`, `streamed`, `drain`) and a consumer's passing of steps it reads none
of (`passed`). The families and walks that compose them are `mosaic`'s."""

import dataclasses
import os
from collections.abc import Iterable, Sequence
from typing import Any

import jax
from jax import lax, numpy as jnp
from jax._src.pallas.mosaic_gpu import core as plgpu_core
from jax._src.state import types as state_types
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import swizzling

Ref = state_types.AbstractRef | state_types.TransformedRef


CLUSTER = "cluster"
"""The axis a launch's cluster of programs runs along (`Family.kernel`'s `cluster`)."""


class Pieces(tuple):
    """A step's tile loaded as several copies (`viewed`, `Ring.load`): its block's rows in as many equal
    parts, each a view of its own, in order (a `Rows` step's runs)."""


@dataclasses.dataclass(frozen=True)
class Rows:
    """A block's rows in runs at a table's entries, the toolkit's one table-indexed row form: its run `i`,
    `rows` rows, at run `table[start + i]` of the array's rows (`table` an int32 ref, its entries in runs),
    its other block indices `rest`: a gathered stream's (`Transport.gathers_rows`), each run copied by the
    loading warpgroup (`gathered.Ring`), or interpreted where no lane-level copy runs, a piece of its own
    (`viewed`'s `Pieces`). A paged pool's key tile across the pages it spans takes a run a key
    (`attention_mosaic_forward.paged_reads`)."""

    table: Any
    start: Any
    rest: tuple
    rows: int


@dataclasses.dataclass(frozen=True)
class Paired:
    """A step's block indices in a pair's view, which the two CTAs of a pair (`Family.pairs_product`) load
    between them: `indices` in the pair's blocks, its block the slot's twice along `along` (`viewed`), each
    CTA loading its half of it into its own slot (`Split`)."""

    indices: tuple
    along: int


@dataclasses.dataclass(frozen=True)
class Split:
    """A pair's step block (`Paired`, `viewed`): `ref`, the pair's block, which each CTA of the pair loads
    its half of along `along`."""

    ref: Any
    along: int


def viewed(view: tuple, block: tuple, step):
    """Step `step`'s tile of a stream's view `(ref, index_map)` (`Family.opened`): the `block`-shaped tile of
    `ref` at the block indices `index_map(step)` gives, as a `BlockSpec`'s index map does; where it gives
    `Rows`, each run's view at its table's entry; where it gives `Paired`, the pair's block (`block` twice
    along its axis) as a `Split`, each CTA's half its own; where `ref` is `Pieces` of refs (an fp32 operand's
    bf16 parts, each its own array's), each one's tile at the indices, its share of the block's rows, in
    order."""
    ref, index_map = view
    indices = index_map(step)
    if isinstance(ref, Pieces):
        share = (block[0] // len(ref), *block[1:])
        return Pieces(viewed((each, index_map), share, step) for each in ref)

    def at(indices, block):
        return ref.at[tuple(pl.ds(index * extent, extent)
                            for index, extent in zip(indices, block, strict=True))]

    if isinstance(indices, Paired):
        pair = tuple(extent * 2 if axis == indices.along else extent for axis, extent in enumerate(block))
        return Split(at(indices.indices, pair), indices.along)
    if isinstance(indices, Rows):
        run = (indices.rows, *block[1:])
        return Pieces(at((indices.table[indices.start + each], *indices.rest), run)
                      for each in range(block[0] // indices.rows))
    return at(indices, block)


def loads(rings: dict, counter, active, steps, views: dict, tile, done) -> None:
    """A producer's loads of one persistent tile (`persistent.tiled`'s visit): each of its steps' blocks
    into its stream's ring at the step the rings are at (`counter`), none where it streams nothing."""
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

    drain(rings, hi - lo)


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
        if os.environ.get("DBG_LOGICAL_ONE"):
            return 1
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
        """A piece's copies into `slot`'s rows from `start` (`load`), each a source and its destination, as
        `piece_cells` lays them out: the slot's rows as its transforms index them, where they are whole tiles
        of rows (`TILE_ROWS`) or the slot is plain; else (`split`), a row's alone (a gathered stream's, `Rows`
        of one row), each of its tiles' columns into the slot's physical row at its swizzle. TMA swizzles a
        row by its address (a 128-byte-aligned row swizzled as the pattern's row it lies at), so it lands as
        its tile's row."""
        rows, smem = piece.shape[0], self.smem
        cells = piece_cells(self.block, smem.dtype, swizzling.swizzle_of(smem), rows)
        if cells is None or not isinstance(smem, state_types.TransformedRef):
            return [(piece, smem.at[slot, pl.ds(start, rows)])]
        unswizzled = next(transform for transform in smem.transforms
                          if isinstance(transform, plgpu_core.UnswizzleRef))
        physical, width = state_types.TransformedRef(smem.ref, (unswizzled,)), cells[0]
        return [(piece.at[pl.ds(row, 1), pl.ds(column * width, width)],
                 physical.at[slot, (start + row) // swizzling.TILE_ROWS, column,
                             pl.ds((start + row) % swizzling.TILE_ROWS, 1)])
                for row, column in cells[1]]

    def acquire(self, turn: Turn):
        """A consumer's wait for `turn`'s load: its slot. Each load is waited for by each consumer, as every
        phase of a barrier is by each thread that waits on it; a wait again for the same turn returns at
        once, its slot not refilled until released."""
        slot = opaque(turn.slot)
        plgpu.barrier_wait(self.full.at[slot], parity=turn.parity)
        if os.environ.get("DBG_SLOT_ZERO"):
            return self.smem.at[opaque(0)]
        return self.smem.at[slot]

    def release(self, turn: Turn) -> None:
        """A consumer's release of `turn`'s slot, whose reads (and MMAs) are done."""
        plgpu.barrier_arrive(self.consumed.at[opaque(turn.slot)])


def piece_cells(block: tuple, dtype, swizzle: int | None, rows: int
                ) -> tuple[int, tuple[tuple[int, int], ...]] | None:
    """The one geometry of a piece of `rows` rows copied into a slot of `block` staged at `swizzle`, which
    `Ring.parts` copies and `piece_copies` counts: None where it is copied once; where `split`, the columns of
    one of the slot's swizzled tiles and each copy's `(row, column tile)`, a row's tiles apart. Counted apart
    from its copies, a 256-deep gathered step's barrier waited forever (row 4516)."""
    if swizzle is None or not split(swizzle, rows):
        return None
    width = swizzling.swizzle_columns(dtype, swizzle)
    return width, tuple((row, column) for row in range(rows) for column in range(block[1] // width))


def piece_copies(block: tuple, dtype, swizzle: int | None, rows: int) -> int:
    """The copies of a piece of `rows` rows into a slot of `block` at `swizzle` (`piece_cells`), the
    arrivals its load barrier takes of it (`ring`'s `copies`)."""
    cells = piece_cells(block, dtype, swizzle, rows)
    return 1 if cells is None else len(cells[1])


def split(swizzle: int | None, rows: int) -> bool:
    """Whether `Ring.parts` copies a piece of `rows` rows apart, a row's tile at a time: rows no whole tiles
    (`TILE_ROWS`) of a swizzled slot (a stated swizzle of 32 bytes or more; 16 only tiles)."""
    return (swizzle not in (None, swizzling.UNSWIZZLED) and swizzle >= swizzling.NARROWEST_SWIZZLE_BYTES
            and rows % swizzling.TILE_ROWS != 0)


def drain(rings: Iterable[Ring], end) -> None:
    """The producer's wait, after its last load (step `end - 1`), for the consumers' release of each slot of
    `rings` it filled last, so no slot is freed (or reallocated) while a consumer still reads it, and every
    release is seen: the rings of a depth drained together (`drained`)."""
    rings = tuple(rings)
    for stages in sorted({ring.stages for ring in rings}):
        drained(tuple(ring.consumed for ring in rings if ring.stages == stages), end, stages)


def drained(barriers: Sequence, end, stages: int) -> None:
    """The completions of the last `stages` of `end` turns round each of `barriers` (`stages` a ring each), a
    slot's each (`Turn`), which no later turn of the slot waited for: each barrier ends at the phase its
    waiters saw. A slot's turn is every ring's, waited for under its one guard, so the guards live at once
    are one, whatever the depth: drained a ring after another, ptxas held each slot's guard from the first
    ring's wait to the last's, a predicate a slot, and a thread holds seven (P0 to P6), so an eight-stage
    tcgen05 kernel gradient's two rings spilled 16 bytes of predicates at every raise but 232
    (FS-TGMM-PRED3). With no depth-bound count left, no resource states predicates: the compile gate's
    deepest rings hold it (their spilled bytes)."""
    for back in range(1, stages + 1):
        turn = Turn.of(end - back, stages)

        @pl.when(turn.counter >= 0)
        def _(turn=turn):
            for barrier in barriers:
                plgpu.barrier_wait(barrier.at[opaque(turn.slot)], parity=turn.parity)


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
    return {'smem': swizzling.staged((stages, *block), dtype, swizzle),
            'full': plgpu.Barrier(num_arrivals=copies, num_barriers=stages), 'consumed': consumed}
