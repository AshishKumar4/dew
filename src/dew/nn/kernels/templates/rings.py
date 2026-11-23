"""The Mosaic GPU toolkit's rings (docs/design/kernel-templates.md): a stream's shared-memory slots a
producer loads by TMA and its consumers read (`Ring`, allocated by `ring`), a step's turn round them
(`Turn`), the views a step's block is copied from (`viewed`: `Pieces`, `Rows`, `Paired`, `Split`), a
producer's loads and drain (`loads`, `streamed`, `drain`), a consumer's passing of steps it reads none
of (`passed`), and a step's slots of them (`Slots`, `WholeSlots`; `Beside` its once streams' beside
them). The families and walks that compose them are `mosaic`'s."""

import dataclasses
import functools
from collections.abc import Callable, Iterable, Sequence
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
    (`attention_mosaic_forward.paged_reads`). `tiled`, whether its runs are the same at every step of a tile
    (a gathered product's rows: `start` the tile's, the step in `rest`), so a load reading them ahead reads
    them once a tile (`Ahead`)."""

    table: Any
    start: Any
    rest: tuple
    rows: int
    tiled: bool = False


@dataclasses.dataclass(frozen=True)
class Among:
    """A stream's view whose source is chosen at run time among several arrays' views, standing for the
    whole view (`viewed`): `which(step)` the index of the one step `step` reads (traced: a launch's column
    tile among matrices side by side, `mosaic_gpu.Beside`, the same at every step; the product a step of a
    concatenated contraction falls in, `mosaic_gpu.Chained`), each of `sources` a view `(ref, index_map)` of
    its own. At a step another source is chosen for, a source's map may give indices past its array: its
    tile stays address arithmetic, never copied or prefetched (`Chosen`)."""

    which: Callable
    sources: tuple


@dataclasses.dataclass(frozen=True)
class Chosen:
    """A step's tile of an `Among` view (`viewed`): `which`, the traced index of the source the step reads,
    and each source's tile at the step; a ring's load copies the chosen one's alone (`Ring.copy`), one copy a
    step as any other view's, and a prefetch reads it alone (`prefetch`)."""

    which: Any
    tiles: tuple


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


def prefetch(view) -> None:
    """An L2 prefetch of a step's view (`viewed`, `plgpu.async_prefetch`): a `Chosen` tile's chosen source's
    alone, as a ring's load copies it (`Ring.copy`)."""
    if isinstance(view, Chosen):
        for index, tile in enumerate(view.tiles):
            pl.when(view.which == index)(functools.partial(prefetch, tile))
        return
    plgpu.async_prefetch(view)


def viewed(view: "Among | tuple", block: tuple, step):
    """Step `step`'s tile of a stream's view `(ref, index_map)` (`Family.opened`): the `block`-shaped tile of
    `ref` at the block indices `index_map(step)` gives, as a `BlockSpec`'s index map does; where it gives
    `Rows`, each run's view at its table's entry; where it gives `Paired`, the pair's block (`block` twice
    along its axis) as a `Split`, each CTA's half its own; where `ref` is `Pieces` of refs (an fp32 operand's
    bf16 parts, each its own array's), each one's tile at the indices, its share of the block's rows, in
    order; where the view is an `Among`, each source's tile at the step and the one it reads (`Chosen`)."""
    if isinstance(view, Among):
        return Chosen(view.which(step), tuple(viewed(source, block, step) for source in view.sources))
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


def held_together(rings: dict) -> list[dict]:
    """`rings` by the barrier pair they hold (`joined`'s rings one pair, every other ring its own), each group
    its rings by their keys in `rings`' order: a step's slots of a group are loaded once their one slot is
    free, awaited at its one load barrier and given back at its one release barrier, once each a step."""
    groups: list[dict] = []
    for key, ring in rings.items():
        group = next((group for group in groups if _lead(group).full is ring.full), None)
        if group is None:
            groups.append({key: ring})
        else:
            group[key] = ring
    return groups


@dataclasses.dataclass(frozen=True)
class Ahead:
    """The next step's view of a ring whose views read memory to address a step (a `Rows` table's entries:
    `reads_ahead`), handed to its load (`Ring.load`'s `ahead`) so it reads those of the next step under this
    one's copies (`gathered.Committed`): `view` the next step's, resolved by the load that reads it ahead
    (a call: an ordinary ring, which reads no step ahead, never resolves it, so no step past the walk's last
    is addressed, base review 103007773 F1), `live` whether the walk has that step and reads it (none where
    its rows are the tile's, `Rows.tiled`: the step's own read for every step), `held` whether this step's
    own were read before (not a walk's first step), and `parity` the half of the ring's two steps' entries
    this step's are in (its counter's parity; a tiled view's the first, every step of its tile)."""

    view: Callable[[], Any]
    live: Any
    held: Any
    parity: Any


def reads_ahead(indices) -> bool:
    """Whether a stream's step `indices` (its view's index map's at the step) read memory to address it: a
    table's (`Rows`), whose entries its load reads at the step before (`Ahead`)."""
    return isinstance(indices, Rows)


def stepped_in(rings: dict, turn_of: Callable, view_of: Callable, ahead_of: Callable | None = None) -> None:
    """A step's loads into `rings` (`turn_of(key, ring)` its turn, `view_of(key, ring)` its source,
    `ahead_of(key, ring)` the next step's where its views read memory: `Ahead`, None otherwise), a barrier
    pair at a time (`held_together`): a ring holding its own pair by its own load (`Ring.load`, a gathered
    ring's its own copies, with the next step's view where it reads ahead), a joined pair's slot freed once
    and each member copied into it (`_joined_loaded`); the one loader a producer's and a loading consumer's
    walks (`fed.Feed`) take. Every ring's step is started (`Ring.started`: a ring reading its addresses
    ahead issues a tile's first step's own reads), then the load warp's reads drained once (`Ring.drained`,
    the first such ring's for all), before any ring loads, so no ring's wait for its own addresses waits on
    another's next-step reads issued after its load (base reviews 103007773 F4, b68fc2382 F6)."""
    groups = held_together(rings)
    # Each lone ring's step formed once, for its await and its load, by its group's place.
    lone = {at: _LoneStep(ring, turn_of(key, ring), view_of(key, ring),
                          None if ahead_of is None else ahead_of(key, ring))
            for at, group in enumerate(groups) if len(group) == 1 for key, ring in group.items()}
    for step in lone.values():
        step.ring.started(step.turn, step.view, step.ahead)
    ahead = [step for step in lone.values() if step.ahead is not None]
    if ahead:
        ahead[0].ring.drained([(step.view, step.ahead) for step in ahead])
    # The loads in the groups' own order.
    for at, group in enumerate(groups):
        if at in lone:
            step = lone[at]
            step.ring.load(step.turn, step.view, step.ahead)
        else:
            _joined_loaded(group, turn_of, view_of)


@dataclasses.dataclass(frozen=True)
class _LoneStep:
    """A ring holding its own barrier pair at a step (`stepped_in`): its turn, its view and the next step's
    (`Ahead`, None where it reads none ahead)."""

    ring: "Ring"
    turn: "Turn"
    view: Any
    ahead: "Ahead | None"


def _joined_loaded(group: dict, turn_of: Callable, view_of: Callable) -> None:
    """A joined pair's step loaded (`held_together`): its slot freed once and each member copied into it."""
    key, lead = next(iter(group.items()))
    # One slot for the pair (its rings hold one depth, `joined`), its index formed once (`Ring.load`).
    turn = turn_of(key, lead)
    slot = opaque(turn.slot)
    lead.free(turn, slot)
    for key, ring in group.items():
        ring.copy(slot, view_of(key, ring))


def _keyed(rings) -> dict:
    """`rings` (a sequence) keyed by their places, as `held_together` groups them."""
    return dict(enumerate(rings))


def _lead(group: dict) -> "Ring":
    """A group's first ring (`held_together`), which waits and arrives for every ring of it."""
    return next(iter(group.values()))


def loads(rings: dict, counter, active, steps, views: dict, tile, done) -> None:
    """A producer's loads of one persistent tile (`persistent.tiled`'s visit): each of its steps' blocks
    into its stream's ring at the step the rings are at (`counter`), none where it streams nothing; a ring
    whose views read memory to address a step (`reads_ahead`) given the next step's view (`Ahead`), so it
    reads that step's addresses under this one's copies (an emulated B200's gathered kernel gradient waited
    805 cycles of its load warps' 1,700-1,900 a step for its table entries, its slot's release 72: em7x-3)."""
    del tile, done
    walked = jnp.where(active, steps, 0)

    @pl.loop(0, walked)
    def _(step):
        # Each view its rings load, its index map evaluated once at the step (base review ae8a242ed F5), its
        # view fixed there; a view whose ring another warp loads is not evaluated here (row 8700).
        fixed = {name: view if not isinstance(view := views[name], tuple) else
                 (view[0], functools.partial(lambda indices, _: indices, view[1](step)))
                 for name in rings}

        def ahead(name, ring):
            view = fixed[name]
            if not (isinstance(view, tuple) and reads_ahead(indices := view[1](step))):
                return None
            # A tile's rows read at its first step and held for its others (base review ae8a242ed F6).
            live, parity = ((False, 0) if indices.tiled
                            else (step + 1 < walked, jnp.asarray(counter + step, jnp.int32) % 2))
            return Ahead(functools.partial(ring.viewed, views[name], step + 1), live, step > 0, parity)

        stepped_in(rings, lambda name, ring: ring.turn(counter + step),
                   lambda name, ring: ring.viewed(fixed[name], step), ahead)


def loaded_once(rings: dict, views: dict) -> None:
    """A producer's loads of a walk's once streams (`Walk.once`): each its block, which every step reads,
    into its one-deep ring's slot."""
    for name, ring in rings.items():
        ring.load(ring.turn(0), ring.viewed(views[name], 0))


def streamed(rings: tuple, views, lo, hi, *, counted=0) -> None:
    """A producer's loads of steps `lo` to `hi` into `rings`, each ring's step counted from `lo`, `counted`
    turns past the ring's first (a walk of the same steps again, its rotation continuing), and loaded from
    its view of `views(step)`, then each ring drained."""
    @pl.loop(lo, hi)
    def _(index):
        given = dict(zip(_keyed(rings), views(index), strict=True))
        stepped_in(_keyed(rings), lambda key, ring: ring.turn(index - lo + counted),
                   lambda key, ring: given[key])

    drain(rings, counted + hi - lo)


def passed(rings: tuple, lo, hi, *, first: jax.Array | int = 0, between=None) -> None:
    """A consumer's steps `lo` to `hi` of `rings` it reads none of (another consumer's), each ring's load
    awaited and its slot released as a reading consumer's are (counted from `first`): a wait takes its
    barrier's next phase, so every consumer waits on every load; `between()` after each step."""
    @pl.loop(lo, hi)
    def _(index):
        for group in held_together(_keyed(rings)):
            _lead(group).acquire(_lead(group).turn(index - first))
            _lead(group).release(_lead(group).turn(index - first))
        if between is not None:
            between()


@dataclasses.dataclass(frozen=True)
class Turn:
    """Step `counter`'s turn of a ring of `stages` slots (`Ring`), the counter running from the ring's first
    load: its slot, `counter % stages`, and its rotation, `counter // stages`, the phase of the slot's
    barriers its waits take. Every slot's barriers complete one phase a rotation, so the one rotation serves
    them all (`plgpu.barrier_wait`'s `phase`), with no phase tracked a barrier. A slot a Python int
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

    def ahead(self, steps: int, stages: int) -> "Turn":
        """The turn `steps` after it (fewer than `stages`), its slot constant where this one's is."""
        if isinstance(self.slot, int):
            slot = self.slot + steps
            return Turn(self.counter + steps, slot % stages, self.rotation + slot // stages)
        return Turn.of(self.counter + steps, stages)

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
        return int(self.smem.shape[0])

    def issued_by(self, index: int, count: int) -> "Ring | None":
        """The ring as load warp `index` of a producer's `count` issues into it: a ring's loads the first's
        alone (one issuing thread arrives at a slot's load barrier and copies its step, None for the
        others); a ring whose step spreads over them, each one's share (`gathered.Committed`)."""
        del count
        return self if index == 0 else None

    @property
    def block(self) -> tuple:
        """The block a slot holds."""
        return self.smem.shape[1:]

    @property
    def multicast(self) -> tuple | None:
        """The cluster axes the ring's loads are multicast along: its release barrier's, a cluster's (`ring`);
        None where the ring is its program's own."""
        return collective_axes(self.consumed)

    def turn(self, counter) -> Turn:
        """Step `counter`'s turn (traced, or a Python int)."""
        return Turn.of(counter, self.stages)

    def initialized(self) -> None:
        """The ring's barriers ready, by every warpgroup before the roles split: Pallas's own, allocated
        initialized."""

    def viewed(self, view: tuple, step):
        """Step `step`'s source of the stream's view (`viewed`), as `load` takes it."""
        return viewed(view, self.block, step)

    def started(self, turn: Turn, view, ahead: Ahead | None = None) -> None:
        """The step's addresses' reads a tile's first step issues before any ring of the step loads
        (`stepped_in`), where its load reads them ahead (`ahead`): nothing for a ring that reads none ahead
        (`gathered.Committed`'s own)."""
        del turn, view, ahead

    def drained(self, steps: list) -> None:
        """The step's address reads of every ring that reads them ahead (`steps`: each one's view and
        `Ahead`) awaited once, after every ring's `started` and before any loads (`stepped_in`): a ring that
        reads none ahead is never asked (`gathered.Committed`'s own)."""
        del steps

    def load(self, turn: Turn, view, ahead: Ahead | None = None) -> None:
        """The producer's load of `turn`'s step from `view`, once its slot is free (`free`, `copy`): the
        slot's index formed once (`opaque`) for its wait and its copies, as an index each formed apart repeats
        its arithmetic past the barrier (+16 mov, +8 each of four integer ops and a register at an H100's
        fused dK/dV, 2.9% of its time, M-ET-87). A view that reads its addresses (`Rows`, `ahead` the next
        step's) reads them as it copies: an interpreted walk's pieces, each run's entry read where it is
        copied; a ring reading a step's addresses under the step before's copies is its own
        (`gathered.Committed`)."""
        del ahead
        slot = opaque(turn.slot)
        self.free(turn, slot)
        self.copy(slot, view)

    def free(self, turn: Turn, slot) -> None:
        """The producer's wait for `turn`'s slot (`slot`, its index `opaque`) to be free: released by the last
        rotation's step, whose parity is the other; none for a slot's first fill."""
        pl.when(turn.counter >= self.stages)(lambda: plgpu.barrier_wait(
            self.consumed.at[slot], phase=turn.rotation - 1))

    def copy(self, slot, view) -> None:
        """A step copied from `view` into `slot` (its index `opaque`), freed (`free`): `Pieces` each into its
        part of the slot's rows, one arrival each at the slot's load barrier, which takes as many (`ring`'s
        `copies`, a joined pair's every member's, `joined`)."""
        if isinstance(view, Chosen):
            # The chosen source's copy alone, one arrival as any view's.
            for index, tile in enumerate(view.tiles):
                pl.when(view.which == index)(functools.partial(self.copy, slot, tile))
            return
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

    def slot(self, turn: Turn):
        """`turn`'s slot, its load awaited by the step already (a ring of a joined pair whose first ring's
        wait covers it, `Slots`)."""
        return self.smem.at[opaque(turn.slot)]

    def acquire(self, turn: Turn):
        """A consumer's wait for `turn`'s load: its slot. Each load is waited for by each consumer, as every
        phase of a barrier is by each thread that waits on it; a wait again for the same turn returns at
        once, its slot not refilled until released."""
        slot = opaque(turn.slot)
        plgpu.barrier_wait(self.full.at[slot], phase=turn.rotation)
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
    rings = tuple(_lead(group) for group in held_together(_keyed(rings)))
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
                plgpu.barrier_wait(barrier.at[opaque(turn.slot)], phase=turn.rotation)


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


JOINED = "+"
"""What joins a joined pair's stream names into its allocation's key (`joined`, `opened`), a name marked
multicast (`MULTICAST`) where its loads are a cluster's."""

MULTICAST = "*"


def joined(members: dict[str, dict], multicast: tuple[str, ...] = ()) -> tuple[str, dict]:
    """One barrier pair for the rings of `members` (each `ring`'s allocations, at one depth), CUTLASS's
    PipelineTmaAsync for a GEMM's operands: their slots as they are, a load barrier each slot that every
    member's copies of a step arrive at, its bytes theirs together, and a release barrier each slot that the
    consumers arrive at once a step for all of them, a cluster's where any member is multicast (its slot is
    free once every program's consumers released it, as a multicast ring's is, a stricter wait for a member
    of its own). Its key (`JOINED`-joined names, each `multicast` one marked `MULTICAST`) and allocations,
    which `opened` reads back as one `JoinedRing` a member, the pair shared. A step of the pair waits one
    load barrier and arrives at one release barrier where its members each did (`held_together`)."""
    stages = {int(allocated['smem'].shape[0]) for allocated in members.values()}
    if len(stages) != 1:
        raise ValueError(f"a joined pair's rings hold one depth, not {sorted(stages)}")
    clustered = [allocated['consumed'] for name, allocated in members.items() if name in multicast]
    consumed = clustered[0] if clustered else next(iter(members.values()))['consumed']
    copies = sum(allocated['full'].num_arrivals for allocated in members.values())
    key = JOINED.join(f"{name}{MULTICAST if name in multicast else ''}" for name in members)
    return key, {'smem': {name: allocated['smem'] for name, allocated in members.items()},
                 'full': plgpu.Barrier(num_arrivals=copies, num_barriers=stages.pop()), 'consumed': consumed}


def opened_joined(key: str, scoped: dict) -> dict[str, "JoinedRing"]:
    """A joined pair's rings (`joined`) by stream name, from its key and its allocations scoped."""
    names = key.split(JOINED)
    return {name.rstrip(MULTICAST): JoinedRing(scoped['smem'][name.rstrip(MULTICAST)], scoped['full'],
                                               scoped['consumed'], axes=collective_axes(scoped['consumed'])
                                               if name.endswith(MULTICAST) else None)
            for name in names}


def collective_axes(consumed) -> tuple | None:
    """The cluster axes a release barrier's arrivals reach (a `ring`'s multicast one's), None for a
    program's own: a ring's multicast (`Ring.multicast`, a joined member's, `opened_joined`)."""
    dtype = consumed.dtype
    return dtype.collective_axes if isinstance(dtype, plgpu_core.ClusterBarrierType) else None


@dataclasses.dataclass(frozen=True)
class JoinedRing(Ring):
    """A ring of a joined pair (`joined`): its own slots, the pair's barriers, multicast along `axes` alone
    (its loads a cluster's where it is a multicast member, its program's own otherwise), whatever the
    pair's release barrier."""

    axes: tuple | None = None

    @property
    def multicast(self) -> tuple | None:
        return self.axes


class WholeSlots:
    """A step's operand slots by stream name (`Slots`' interface) where the walk hands the step its blocks
    whole and refills a slot only after the step that read it (`emit_pipeline`'s on cp.async, a TPU core's
    walk awaiting its blocks): releasing one early is nothing."""

    def __init__(self, refs: dict):
        self._refs = refs

    def __getitem__(self, name: str):
        return self._refs[name]

    def release(self, name: str) -> None:
        del name

    def acquire_all(self) -> None:
        """Every slot acquired: the step's blocks are its whole."""


class Beside:
    """A step's slots (`Slots`) beside its walk's once streams' (`Walk.once`), by stream name: a once
    stream's the slot every step reads, which the walk gives back, every other the step's own."""

    def __init__(self, slots: "Slots", once: dict):
        self._slots, self._once = slots, once
        self.more, self.first = slots.more, slots.first

    def __getitem__(self, name: str):
        return self._once[name] if name in self._once else self._slots[name]

    def release(self, name: str) -> None:
        if name not in self._once:
            self._slots.release(name)

    def opened(self, name: str):
        return self._slots.opened(name)

    def following(self, name: str):
        return self._slots.following(name)

    def done(self) -> None:
        self._slots.done()

    def held(self) -> None:
        self._slots.held()

    def acquire_all(self) -> None:
        self._slots.acquire_all()


def besides(slots: "Slots", once: dict | None):
    """A step's `slots` beside its walk's once streams' (`Beside`), as they are where none."""
    return Beside(slots, once) if once else slots


class Slots:
    """A walk step's operand slots (`Family.walk`), by stream name: each acquired when the step first reads
    it (`slots[name]`) and released at the step's end, once its MMAs are done, unless the step `release`s it
    sooner (a slot its last product has read). A stream the walk reads a step `ahead` the step reads through
    the next step's slot (`following`, where `more` steps follow in its range): its own slot was acquired by
    the step before it, or by itself where it is its range's `first` (`opened`), and it gives that slot
    back as any other."""

    def __init__(self, rings: dict[str, Ring], turn: Turn, ahead: tuple[str, ...] = (), *, more=False,
                 first=True):
        self._rings, self._turn = rings, turn
        self._held: dict[str, Ref] = {}
        self._released: set[str] = set()
        # Each stream's barrier pair's streams (`held_together`): one wait and one release a step for them.
        self._together = {name: tuple(group) for group in held_together(rings) for name in group}
        self._ahead, self.more, self.first = ahead, more, first

    def __getitem__(self, name: str):
        if name in self._ahead:
            raise ValueError(f"{name}'s slot is read a step ahead: the step before read it (`following`), or "
                             "the range's first step reads its own (`opened`)")
        if name not in self._held:
            ring = self._rings[name]
            awaited = any(other in self._held for other in self._together[name])
            self._held[name] = ring.slot(self._turn) if awaited else ring.acquire(self._turn)
        return self._held[name]

    def opened(self, name: str):
        """Stream `name`'s own slot where the walk reads it ahead and no step before this one did: the range's
        first step's (`first`), within its branch."""
        return self._rings[name].acquire(self._turn)

    def following(self, name: str):
        """Stream `name`'s slot at the next step, which this one reads ahead of it where `more` steps follow:
        acquired here, given back by the next step."""
        stages = min(ring.stages for ring in self._rings.values())
        return self._rings[name].acquire(self._turn.ahead(1, stages))

    def release(self, name: str) -> None:
        """Stream `name`'s slot given back now, its reads done (acquired first if the step read none, or by
        the step before where the walk reads it ahead)."""
        if name not in self._ahead:
            self[name]
        self._released.add(name)
        # A joined pair's slots given back once, by its last stream's release.
        if all(other in self._released for other in self._together[name]):
            self._rings[name].release(self._turn)

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
        if self._released or self._ahead:
            raise ValueError("a step whose products run past it gives back no slot early nor reads one ahead")
        self.acquire_all()

    def acquire_all(self) -> None:
        """Every slot the step has not read acquired in the step's own trace: a slot first acquired inside a
        loop's body (a rolled `lax.fori_loop`) is kept by its first read, which would carry that body's slot
        index past it, a scalar the kernel then takes as an input (the Mosaic GPU lowering fails on it); a
        step that reads its slots in a loop acquires them before it. A stream read ahead is not its own to
        acquire."""
        for name in self._rings:
            if name not in self._ahead:
                self[name]


def released(rings: dict[str, Ring], turn: Turn) -> None:
    """Every ring's slot of `turn` given back, once a barrier pair (`held_together`): a step's slots its walk
    held past it (`Slots.held`)."""
    for group in held_together(rings):
        _lead(group).release(turn)
