"""Blackwell's bundle (Mosaic GPU, sm100 and sm103): tcgen05's products, issued by one thread into
accumulators in tensor memory, and the warpgroups a mainloop runs them in, which the GEMM template's
(the GEMM template's, on `RINGS`) and the attention template's Blackwell mainloops share.

A program runs two warpgroups (`roles`). The first issues: one warp's TMA loads fill the TMA rings' operand
stages (`RINGS`, `CommittedRing`), one warp issues each step's products once its stage is there, the products
releasing the stage when they have read it (a tensor-core-ordered barrier, `tcgen05_mma`'s). The second reads
finished accumulators out of tensor memory (`read`, in column chunks of tcgen05's register layout) and
releases them for the products after next (`Accumulators`, two or more in tensor memory's columns). A
barrier's completions no later turn waited for are observed before the program ends (`drain`).

Unlike Hopper's and Ampere's bundles (`mosaic.Family`), a thread holds no accumulator between products: the
registers a program takes are its warpgroups' entry share alone (`registers`), the issuing warpgroup's few
(addresses and counters) left to a per-launch rule once a liveness model prices its branch
(`resources.producer_floor`)."""

import contextlib
import contextvars
import dataclasses
import functools
from collections.abc import Callable, Iterator
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax._src.pallas.mosaic_gpu import core as plgpu_core
from jax._src.state import types as state_types
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import generation
from dew.nn.kernels.templates import handoffs, mosaic, mosaic_sm90, persistent, resources, rings, swizzling
from dew.nn.kernels.templates.rings import Ref

ISSUING = (handoffs.Role("load", warp=True, weight=0), handoffs.Role("mma", warp=True, weight=0))
"""The issuing warpgroup's roles after a template's own (`handoffs.Schedule`'s warp roles): its load warp's
TMA copies and its product warp's MMAs."""
TMEM = plgpu.Layout.TCGEN05_TMEM_NATIVE
"""A tile's register layout as tensor memory loads it natively (a thread a row)."""
STORED = plgpu.Layout.TCGEN05
"""An accumulator's register layout for an epilogue, the one a swizzled shared-memory tile takes."""


@dataclasses.dataclass(frozen=True)
class Topology:
    """The CTAs one product spans (tcgen05's collective form): one, or a pair of a cluster's
    (`rings.CLUSTER`), whose leader issues each product over both's rows into both's tensor memory, each
    CTA loading its own rows' block of one operand and its half of the other's, the leader's load barriers
    tracking both halves and its products' commits arriving at both (CUTLASS's 2-SM UMMA)."""
    ctas: int = 1

    @property
    def paired(self) -> bool:
        return self.ctas == 2

    def index(self):
        """This CTA's place in the pair (0, the leader, without one)."""
        return lax.axis_index(rings.CLUSTER) if self.paired else 0

    def leads(self):
        """Whether this CTA issues the products."""
        return lax.axis_index(rings.CLUSTER) == 0 if self.paired else True

    def copied(self, dimension: int) -> dict:
        """A load's partition across the pair: each CTA its share of the block along `dimension`, tracked
        by the leader's barrier."""
        if not self.paired:
            return {}
        return {'leader_tracked': plgpu.CopyPartition.PARTITIONED(dimension),
                'collective_axes': rings.CLUSTER}

    def issued(self) -> dict:
        """A product's or a commit's spread over the pair."""
        return {'collective_axis': rings.CLUSTER} if self.paired else {}


SINGLE = Topology()
"""A product on one CTA."""

PAIRED = Topology(2)
"""A product over a pair of CTAs, its leader issuing it."""

PRODUCT_COLUMNS = 256
"""tcgen05's widest product, its columns (N) (PTX ISA, tcgen05.mma's shapes): one CTA's, or a pair's, where
each CTA's shared memory holds half of the right operand's columns and its tensor memory all of them for its
own rows."""


def swizzled(dtype, contraction: int) -> tuple:
    """A stage's transforms for an operand block of `dtype` contracting `contraction` elements a step: 8-row
    tiles of the widest swizzle its rows take, swizzled (tcgen05 reads its shared operands so)."""
    return swizzling.swizzle_transforms(dtype, resources.widest_swizzle(dtype, contraction))


def operand(shape: tuple[int, int], dtype, contraction: int):
    """A shared-memory operand block of `shape` a product reads whole, its rows staged at the widest swizzle
    `contraction` elements take (`swizzled`)."""
    return plgpu.SMEM(shape, dtype, transforms=swizzled(dtype, contraction))


def operand_columns(rows: int, columns_: int, dtype):
    """A tensor-memory operand of `rows` x `columns_` in `dtype` a product reads (`S`'s narrowed weights,
    FA4's `P`), packed two to a column where its elements are 16-bit."""
    return plgpu.TMEM((rows, columns_), dtype, packed=jnp.dtype(dtype).itemsize < 4)


def vector(length: int, dtype=jnp.float32):
    """A shared-memory vector of `length` (a tile's rows' statistics on their way out or between layouts)."""
    return plgpu.SMEM((length,), dtype)


def accumulator_columns(rows: int, columns_: int):
    """An fp32 tensor-memory accumulator of `rows` x `columns_` (an output's, read out once)."""
    return plgpu.TMEM((rows, columns_), jnp.float32)


@dataclasses.dataclass(frozen=True)
class Signals:
    """`count` one-phase-a-turn barriers between a program's roles, a turn's slot its remainder (`Signal`):
    tensor-core ordered where a product's commit arrives at it or a role's tensor-memory stores precede its
    arrival (`tensor_core`), else a load's; `arrivals` arrive at a turn (`handoffs.HandOff`'s producer's)."""
    count: int = 1
    tensor_core: bool = True
    arrivals: int = 1

    def arrived(self, produced: int, released: int) -> "Signals":
        del released  # a signal is only filled
        return dataclasses.replace(self, arrivals=produced)

    def scratch(self, target: generation.GpuTarget | None = None):
        del target
        return plgpu.Barrier(num_arrivals=self.arrivals, num_barriers=self.count,
                             orders_tensor_core=self.tensor_core)

    def opened(self, barriers) -> "Signal":
        return Signal(self, barriers)


@dataclasses.dataclass(frozen=True)
class Signal:
    """An opened signal (`Signals.opened`): each turn arrived at once, by a load, a product's commit or a
    role's threads, and waited for at its turn's parity by every role that waits on it, at every turn."""
    shape: Signals
    barriers: state_types.AbstractRef

    def turn(self, counter) -> rings.Turn:
        return rings.Turn.of(counter, self.shape.count)

    def loaded(self, turn: rings.Turn, view, ref) -> None:
        """`view` copied into `ref` by TMA, its arrival `turn`'s."""
        plgpu.copy_gmem_to_smem(view, ref, self.barriers.at[turn.slot])

    def committed(self, turn: rings.Turn, topology: Topology = SINGLE) -> None:
        """`turn`'s arrival once every product the thread issued is done (the tensor core's commit)."""
        plgpu.tcgen05_commit_arrive(self.barriers.at[turn.slot], **topology.issued())

    def published(self, turn: rings.Turn) -> None:
        """`turn`'s arrival by the role's threads, their tensor-memory stores visible to the products first
        (`Tcgen05Issuer.write`)."""
        plgpu.commit_tmem()
        plgpu.barrier_arrive(self.barriers.at[turn.slot])

    def wait(self, turn: rings.Turn) -> None:
        plgpu.barrier_wait(self.barriers.at[turn.slot], parity=turn.parity)


@dataclasses.dataclass(frozen=True)
class Accumulators:
    """A ring of `count` fp32 accumulators of `rows` x `columns` in tensor memory side by side: a done barrier
    each, which the producing role arrives at (the product warp's commit once a turn's products are issued),
    and a read barrier each, which the roles that give the turn back arrive at (`released`: the reading
    warpgroups once they have read it out); both tensor-core ordered, their counts stated by the roles that
    fill and give back the ring (`handoffs.HandOff`, `arrived`). A turn is a counter round the ring; its
    accumulator is the turn's remainder. Over a pair of CTAs (`topology`) each holds its rows' accumulators,
    allocated together, and an accumulator is free once both CTAs have read theirs out (its read barrier
    the cluster's)."""
    count: int
    rows: int
    columns: int
    topology: Topology = SINGLE
    produced: int = 1
    released: int = 1
    running: bool = False
    """Whether a tile's running sums lie past the turns, as many columns again (an exactly summed walk's,
    each window's turn promoted into them: `Tcgen05Rings.persist`'s `summed`)."""

    def arrived(self, produced: int, released: int) -> "Accumulators":
        return dataclasses.replace(self, produced=produced, released=released)

    def barriers(self) -> tuple:
        """The ring's done and read barriers (`scratch`'s, which a launch's shared memory counts)."""
        done = plgpu.Barrier(num_arrivals=self.produced, num_barriers=self.count, orders_tensor_core=True)
        if self.topology.paired:
            return done, plgpu.ClusterBarrier(collective_axes=(rings.CLUSTER,), num_arrivals=self.released,
                                              num_barriers=self.count, orders_tensor_core=True)
        read = plgpu.Barrier(num_arrivals=self.released, num_barriers=self.count, orders_tensor_core=True)
        return done, read

    def scratch(self, target: generation.GpuTarget) -> tuple:
        columns = (self.count + self.running) * self.columns
        if target.tmem_columns is None:
            raise ValueError(f"{target.generation} has no tensor memory")
        if columns > target.tmem_columns:
            raise ValueError(f"{self.count + self.running} accumulators of {self.columns} columns exceed "
                             "tensor memory")
        tmem = plgpu.TMEM((self.rows, columns), jnp.float32, collective=self.topology.paired)
        return tmem, *self.barriers()

    def opened(self, tmem, done, read) -> "Tensor":
        return Tensor(self, tmem, done, read)


@dataclasses.dataclass(frozen=True)
class Tensor:
    """Opened accumulators (`Accumulators.opened`): the product warp waits for a turn's accumulator to be
    free (`freed`, `acquire`), issues into it and commits (`issued`); a reading role waits for it once
    (`ready`), reads its columns in parts (`slot`, `columns`, through the issuer's `read`) and gives it back
    (`release`)."""
    shape: Accumulators
    tmem: Ref
    done: state_types.AbstractRef
    read_done: state_types.AbstractRef

    def turn(self, counter) -> rings.Turn:
        """Turn `counter` round the accumulators (`rings.Turn`, its slot the accumulator's index)."""
        return rings.Turn.of(counter, self.shape.count)

    def freed(self, turn: rings.Turn) -> None:
        """The product warp waits for the turn's accumulator: given back by the turn `count` before."""
        pl.when(turn.counter >= self.shape.count)(lambda: plgpu.barrier_wait(
            self.read_done.at[turn.slot], parity=turn.freed))

    def acquire(self, turn: rings.Turn):
        """The product warp's turn: the turn's accumulator, once `freed`."""
        self.freed(turn)
        return column_span(self.tmem, turn.slot * self.shape.columns, self.shape.columns)

    def columns(self, turn: rings.Turn, first, count: int):
        """Columns `first` to `first + count` of the turn's accumulator (several products' accumulators side
        by side in one turn's columns: a launch's paired outputs)."""
        return column_span(self.tmem, turn.slot * self.shape.columns + first, count)

    def issued(self, turn: rings.Turn) -> None:
        """The product warp's turn's products all issued: the reader may wait on them."""
        plgpu.tcgen05_commit_arrive(self.done.at[turn.slot], **self.shape.topology.issued())

    def ready(self, turn: rings.Turn) -> None:
        """A reading role's turn: its products done, once, however many parts it reads (`slot`)."""
        plgpu.barrier_wait(self.done.at[turn.slot], parity=turn.parity)

    def running_columns(self, first, count: int):
        """Columns `first` to `first + count` of the tile's running sums (`Accumulators.running`)."""
        return column_span(self.tmem, self.shape.count * self.shape.columns + first, count)

    def slot(self, turn: rings.Turn) -> "Accumulator":
        """The turn's accumulator, which the issuer reads, writes and updates in column parts
        (`Tcgen05Issuer.read`, `write`, `update`) once it is `ready`."""
        shape = (self.shape.rows, self.shape.columns)
        return Accumulator(column_span(self.tmem, turn.slot * self.shape.columns, self.shape.columns), shape)

    def release(self, turn: rings.Turn) -> None:
        """A reading role's turn read out: its accumulator free for the turn `count` after."""
        plgpu.wait_load_tmem()
        plgpu.barrier_arrive(self.read_done.at[turn.slot])

    def drain(self, turns) -> None:
        """The product warp's last turns' give-backs, which no later product waited for (`rings.drained`)."""
        rings.drained((self.read_done,), turns, self.shape.count)


def column_span(tmem, first, count: int):
    """Columns `first` to `first + count` of tensor memory `tmem`, every row."""
    return tmem.at[:, pl.ds(first, count)]


@dataclasses.dataclass(frozen=True)
class Accumulator:
    """A tile in tensor memory, the tcgen05 bundle's walk's and `products`' value: an fp32 accumulator, or a
    product's left operand a role wrote there (`Tcgen05Issuer.write`: FlashAttention 4's narrowed weights,
    16-bit, two to a column); its columns (`ref`; None for a tile that multiplies nothing, whose chunks are
    zeros), its shape and dtype, and whether a product adds onto it (not at a tile's first step: tensor
    memory starts unset)."""
    ref: Ref | None
    shape: tuple[int, int]
    accumulate: jax.Array | bool = False
    dtype: Any = jnp.float32


class CommittedRing(rings.Ring):
    """A ring whose slots tcgen05's products give back: the product warp's commit arrives at a slot's release
    once every product it issued has read the slot (`tcgen05_commit_arrive`), where a register family's
    consumers arrive themselves (`rings.ring`'s `released_by_products`)."""

    def release(self, turn: rings.Turn) -> None:
        plgpu.tcgen05_commit_arrive(self.consumed.at[rings.opaque(turn.slot)])


@dataclasses.dataclass(frozen=True)
class PairedRing(CommittedRing):
    """A pair's ring (`Tcgen05Rings` over `PAIRED`): each step's block the pair's (`rings.Split`), each CTA
    loading its half into its own slot, the leader's load barrier tracking both halves (`Topology.copied`),
    which the leader's products read together; the leader's commit gives each slot back on both CTAs
    (`Topology.issued`), so each CTA's load waits for its own slot's release."""

    topology: Topology = PAIRED

    def load(self, turn: rings.Turn, view) -> None:
        if not isinstance(view, rings.Split):
            raise ValueError(f"a pair's step is the pair's block (`rings.Paired` indices), not {view}")
        slot = rings.opaque(turn.slot)
        pl.when(turn.counter >= self.stages)(lambda: plgpu.barrier_wait(
            self.consumed.at[slot], parity=turn.freed))
        plgpu.copy_gmem_to_smem(view.ref, self.smem.at[slot], self.full.at[slot],
                                **self.topology.copied(view.along))

    def release(self, turn: rings.Turn) -> None:
        plgpu.tcgen05_commit_arrive(self.consumed.at[rings.opaque(turn.slot)], **self.topology.issued())


SCALE_ROWS = 128
"""The rows of one block-scale tile (`scale_tiles`): tensor memory's lanes, one row's scales a lane
(`tcgen05.scales_layout`), so a block-scaled product's tile is whole 128s of rows and of columns."""

BLOCK_SCALED_COLUMNS = SCALE_ROWS
"""The columns of a block-scaled product's tile: its matrix's columns are its scales' rows (`SCALE_ROWS`)."""

SCALES_A_ROW = 4
"""Each row's scales one block-scale tile holds (`scale_tiles`): `.scale_vec::1X`'s four a 32-bit lane."""

SCALE_TILE = (32, 16)
"""A block-scale tile's shared-memory extent (`scale_tiles`, `tcgen05.scales_layout`'s): its 128 rows' four
scales each as 32 rows of 16, row `i`'s scale `j` at `[i % 32, (i // 32) % 4 * 4 + j]`, each warp's quarter
of the lanes a group of four columns."""


def scale_tiles(rows: int, scales: int) -> tuple[int, int, int, int]:
    """The shared-memory shape of `rows` rows' `scales` scales each as the block-scaled products read them
    (`tcgen05.scales_layout`; `async_copy_scales_to_tmem`'s `(MN // 128, K // 4, 32, 16)`): a tile
    (`SCALE_TILE`) each 128 rows' four scales."""
    return (rows // SCALE_ROWS, scales // SCALES_A_ROW, *SCALE_TILE)


def scales_tiled(scales: jax.Array) -> jax.Array:
    """Block scales `[..., rows, scales]` laid out in their tiles (`scale_tiles`): row `i`'s scale `j` at
    `[i // 128, j // 4, i % 32, (i // 32) % 4 * 4 + j % 4]`, as `async_copy_scales_to_tmem` documents their
    preparation."""
    *lead, rows, columns = scales.shape
    lanes = SCALE_TILE[0]
    tiled = scales.reshape(*lead, rows // SCALE_ROWS, SCALE_ROWS // lanes, lanes, columns // SCALES_A_ROW,
                           SCALES_A_ROW)
    order = tuple(range(len(lead)))
    tiled = jnp.transpose(tiled, (*order, *(len(lead) + axis for axis in (0, 3, 2, 1, 4))))
    return tiled.reshape(*lead, *scale_tiles(rows, columns))


def scale_memory(tiles: tuple[int, ...], stages: int) -> tuple[int, int]:
    """The tensor memory a ring of `stages` slots of block-scale tiles `tiles` (`scale_tiles`' shape) is
    copied to: a row a lane, each slot's scales its own columns (`ScaleRing._scales`)."""
    return (tiles[0] * SCALE_ROWS, stages * tiles[1] * SCALES_A_ROW)


@dataclasses.dataclass(frozen=True)
class ScaleRing(CommittedRing):
    """A block-scale stream's ring (`SCALE_TILE`): each slot's shared-memory tiles paired with the tensor
    memory its products read them from (`Tcgen05Issuer.scaled_products`), a slot's columns of `tmem`. A slot's
    scales are copied to its tensor memory by the product warp before its products (tcgen05.cp then
    tcgen05.mma, pipelined in issue order: PTX ISA 9.7.18.6.2), and again only once its products' commit has
    released the slot and the producer refilled it, so no copy overwrites scales a product still reads."""

    tmem: Any = None

    def acquire(self, turn: rings.Turn):
        return super().acquire(turn), self._scales(turn)

    def _scales(self, turn: rings.Turn):
        """`turn`'s slot's tensor memory: its columns of `tmem`."""
        columns = self.tmem.shape[1] // self.smem.shape[0]
        return self.tmem.at[:, pl.ds(rings.opaque(turn.slot) * columns, columns)]


CHUNK_COLUMNS = 64
"""The accumulator columns an epilogue reads out of tensor memory at once: 64 fp32 registers a thread of its
warpgroup, the GEMM epilogue's chunk (`chunk_columns`)."""


class Tcgen05Issuer(mosaic.Issuer[Accumulator]):
    """tcgen05's issuer and home: products issued by one warp into fp32 accumulators in tensor memory
    (`Accumulator`), read from shared memory, asynchronous; an epilogue reads the accumulators out in column
    chunks (`chunks`, in an epilogue's register layout, `STORED`)."""

    name = "tcgen05"
    loads_past_floor = True

    rows = 128  # a product's rows on one CTA (tcgen05's M without a CTA pair)

    role_rows = 128

    accumulator_rows = rows

    issues_in_consumers = False

    takes_pairs = False

    takes_register_left = False

    takes_block_scales = True

    operands_in_registers = False

    products_overlap = True

    topology: Topology = SINGLE
    """The CTAs its products span (`Topology`): one, or a pair's whose leader issues each (`Tcgen05Rings`'
    pair)."""

    _HOME = "tcgen05's accumulators live in tensor memory, issued into by a walk's persist (`Tcgen05Rings`)"

    @property
    def layout(self):
        return STORED

    @property
    def row(self):
        return STORED.reduce(1)

    @property
    def column(self):
        return STORED.reduce(0)

    @property
    def lane_columns(self) -> int:
        raise NotImplementedError(f"{self._HOME}, read out in chunks; no lane holds a column of one")

    def chunk_columns(self, columns: int) -> int:
        return min(columns, CHUNK_COLUMNS)

    def column_registers(self, columns: int, dtype) -> int:
        # An epilogue chunk's columns of the vector held at once (`chunk_columns`), in the epilogue's layout.
        return super().column_registers(self.chunk_columns(columns), dtype)

    def chunks(self, accumulators: tuple) -> Iterator[tuple[int, tuple]]:
        # Each chunk's columns read from each accumulator's tensor memory (`read`) as its consumer asks for
        # it, after the chunk before it is stored: read all at once, every load was in flight together and a
        # thread held the tile's whole row, ptxas's least spill-free cap 144 at 128 columns and 208 at 192
        # where a chunk's model holds 112 (the compile gate's caps scan, EM-least).
        width = accumulators[0].shape[1]
        columns = self.chunk_columns(width)
        for first in range(0, width, columns):
            yield first, tuple(self.read(acc, first=first, columns=columns) for acc in accumulators)

    def read(self, acc: Accumulator, *, first: int = 0, columns: int | None = None, pending: int = 0,
             layout=STORED) -> jax.Array:
        # Its columns out of tensor memory in `layout` (an epilogue's, `STORED`, or tensor memory's own,
        # `TMEM`), the load in flight until the value is read or `Signal.published`; zeros for a tile that
        # multiplies nothing. Its products are awaited by a signal their commit arrives at (`signals`,
        # `Tensor.ready`), not by `pending`.
        del pending
        rows, width = acc.shape
        count = width - first if columns is None else columns
        if acc.ref is None:
            return self.zeros((rows, count))
        return plgpu.async_load_tmem(column_span(acc.ref, first, count), layout=layout)

    def write(self, acc: Accumulator, value: jax.Array, *, first: int = 0) -> None:
        """`value` into the tile's columns from `first` in tensor memory, in its dtype, visible to a product
        once a signal publishes it (`Signal.published`): a softmax's narrowed weights for P·V, a state
        rescaled in place."""
        plgpu.async_store_tmem(column_span(_tensor_memory(acc), first, value.shape[-1]),
                               value.astype(jnp.dtype(acc.dtype)))

    def update(self, acc: Accumulator, fn: Callable[[jax.Array], jax.Array], *, first: int = 0,
               columns: int | None = None, layout=TMEM) -> None:
        """The tile's columns from `first` rewritten in place by `fn` (a correction's rescale of an output
        accumulator): read in `layout` and written back, the read awaited by its value's use. The products
        that last wrote them are the role's to await first (a signal their commit arrives at)."""
        self.write(acc, fn(self.read(acc, first=first, columns=columns, layout=layout)), first=first)

    def tensor_memory(self, shape: tuple[int, int], dtype=jnp.float32):
        """A scoped allocation of a tile of `shape` in tensor memory (`pl.run_scoped`'s), 16-bit elements two
        to a column; `Accumulator(ref, shape, dtype=dtype)` once scoped."""
        return plgpu.TMEM(shape, dtype, packed=jnp.dtype(dtype).itemsize < 4)

    def signals(self, count: int = 1, *, by_products: bool = True) -> "Signals":
        """`count` barriers between a program's roles, one a turn, each arrived at once a turn by the product
        warp's commit (`by_products`: tensor-core ordered, as a role's tensor-memory writes before its
        arrival are) or by a role's threads (`Signals`; `.scratch()` scoped, `.opened(refs)` a `Signal`)."""
        return Signals(count, tensor_core=by_products)

    def started(self, shapes: tuple, initial: tuple | None = None) -> tuple:
        # A walk's accumulators are the tensor memory its persist gives it (`initial`, each an `Accumulator`);
        # a tile that walks no step has none, its chunks zeros (`persistent.unwalked`).
        if initial is not None:
            if tuple(acc.shape for acc in initial) != tuple(shapes):
                given = [acc.shape for acc in initial]
                raise ValueError(f"a walk of {shapes} was given tensor memory of {given}")
            return tuple(initial)
        return tuple(Accumulator(None, shape) for shape in shapes)

    def accumulated[C](self, segments, start: C, shapes: tuple, interpret: bool, initial: tuple | None = None
                       ) -> tuple[C, tuple]:
        # The accumulators stay in tensor memory, out of the loops' carries: a step's products add onto them
        # but at the walk's first step, which a flag the loops carry tells. A loop's index is no step's
        # counter where the walk takes a rotation of its ring a trip (`mosaic.stepping`: the rotation's), so
        # an index against the first step's counter restarted the sum at every rotation of a tile past the
        # first whose steps no rotation's end meets (a 576-deep product at four stages lost its second
        # tile's first three steps).
        del interpret
        accumulators = self.started(shapes, initial)
        if any(acc.ref is None for acc in accumulators):
            raise ValueError(f"{self._HOME}: a walk takes the tensor memory its persist gives it")
        # The flag counts no step before a walk's first and one after it.
        state = (start, jnp.zeros((), jnp.int32))
        for loop in segments:
            def stepped(index, state, step=loop.step):
                carry, taken = state
                at = tuple(dataclasses.replace(acc, accumulate=taken > 0) for acc in accumulators)
                return step(index, carry, at)[0], jnp.ones((), jnp.int32)

            state = lax.fori_loop(loop.lo, loop.hi, stepped, state)
            if loop.then is not None:
                loop.then()
        return state[0], accumulators

    def products(self, acc: Accumulator, lhs, rhs, *, accumulate: bool = True, released=None
                 ) -> Accumulator:
        """`Issuer.products`: the left operand from shared memory, or from tensor memory (an `Accumulator` a
        role wrote, tcgen05's TS form: FlashAttention 4's P·V), the right from shared memory, issued over its
        `topology`'s CTAs; `released`, where given, a barrier the product arrives at once it has read its
        operands (a stage's release, which a walk of its own gives back by it: attention's on Blackwell)."""
        left = _tensor_memory(lhs) if isinstance(lhs, Accumulator) else lhs
        plgpu.tcgen05_mma(_tensor_memory(acc), left, rhs, released,
                          accumulate=acc.accumulate if accumulate else False, **self.topology.issued())
        return dataclasses.replace(acc, accumulate=True)

    def scaled_products(self, acc: Accumulator, lhs, rhs, scales: tuple) -> Accumulator:
        """`products` of block-scaled operands (an MX product's: each 32 depths of `lhs`'s rows and `rhs`'s
        columns times its E8M0 scale), `scales` the step's two `ScaleRing` slots, each its shared-memory tiles
        and their tensor memory: both copied in, then the products read them (`tcgen05_mma`'s `a_scale`,
        `b_scale`)."""
        (a_tiles, a_scale), (b_tiles, b_scale) = scales
        plgpu.async_copy_scales_to_tmem(a_tiles, a_scale)
        plgpu.async_copy_scales_to_tmem(b_tiles, b_scale)
        plgpu.tcgen05_mma(_tensor_memory(acc), lhs, rhs, a_scale=a_scale, b_scale=b_scale,
                          accumulate=acc.accumulate)
        return dataclasses.replace(acc, accumulate=True)

    def settled(self, pending: int = 0) -> None:
        # The walk's release of a step's slots is its products' commit (`CommittedRing`): nothing to await.
        del pending

    def zeros(self, shape: tuple[int, ...]) -> jax.Array:
        return plgpu.layout_cast(jnp.zeros(shape, jnp.float32), STORED)

    def added_into(self, acc, value, interpret: bool):
        raise NotImplementedError(f"{self._HOME}; nothing adds a register tile into one")

    def product(self, lhs, rhs, shape: tuple[int, int], then=None):
        raise NotImplementedError(f"{self._HOME}; a fresh product takes the launch's, not its own")

    def registers_product(self, lhs, rhs):
        raise NotImplementedError(f"{self._HOME}; its operands are read from shared or tensor memory")

    def register_products(self, acc, parts, rhs):
        raise NotImplementedError(f"{self._HOME}; its operands are read from shared or tensor memory")

    def rescaled_products(self, acc, scale, parts, rhs, interpret: bool):
        raise NotImplementedError(f"{self._HOME}; its operands are read from shared or tensor memory")


class Tcgen05Rings(Tcgen05Issuer, mosaic.RingFamily[Accumulator]):
    """tcgen05's bundle: its issuer and tensor-memory home (`Tcgen05Issuer`) on TMA rings a producer warpgroup
    fills (`mosaic.RingFamily`), whose slots its products release by their commit (`CommittedRing`). A
    persistent launch's tiles run three roles: the producer's load warp fills the rings, its product warp
    walks each tile's steps into one of two accumulator turns in tensor memory (`issue`), and the consumer
    warpgroup reads each finished tile's accumulators out in chunks and stores them (`finish`), freeing its
    turn for the tile after next, so a tile's epilogue runs while the next tile's products fill the other
    (JAX's `blackwell_ragged_dot_mgpu.py`'s schedule, CUTLASS's sm100 persistent GEMM). Over a pair of CTAs
    (`Tcgen05PairRings`), CUTLASS's 2-SM UMMA."""

    consumers = 1
    gathers_rows = True  # its load warp's TMA four-row gathers (`gathered.Committed`, row 1376)

    turns = 2
    """The accumulator turns a launch's tensor memory holds: an epilogue reads one while the products fill
    the other."""

    def column_unit(self, dtype) -> int:
        # A packed 4-bit operand's swizzled row spans 256 of them along its depth (K-major, as tcgen05 reads
        # E2M1), not along a tile's columns: the columns step by the 128 its block scales' tiles take.
        return min(super().column_unit(dtype), BLOCK_SCALED_COLUMNS)

    sums_in_persist = True  # each window's partial promoted by the epilogue warpgroup (`persist`'s `summed`)

    rewrites_in_persist = True  # a rewriting warpgroup between the loads and the products (`prepared`)

    @property
    def rewriting(self) -> mosaic.Issuer:
        """The layouts a rewriting warpgroup reads a stage's tiles in (`persist`'s `prepared`): wgmma's, whose
        64-row tiles a swizzled stage loads natively; tcgen05's own is tensor memory's 128 lanes."""
        return mosaic_sm90.WGMMA

    sums_in_windows = True  # a window's partials a turn of tensor memory, its epilogue's to promote

    def tensor_columns(self, columns: int, accumulators: int, summed: bool = False) -> int:
        # Each turn holds every accumulator of a tile (`persist`); exactly summed, the turns hold a window's
        # partials and the tile's running sums take as many columns again.
        return (self.turns + summed) * columns * accumulators

    def streams(self, blocks: dict, stages: int, consumers: int, multicast: tuple[str, ...] = (),
                swizzles: dict[str, int] | None = None, copies: dict[str, int] | None = None,
                gathered: tuple[str, ...] = ()) -> dict:
        from dew.nn.kernels.templates import gathered as gathered_rows  # it builds on `rings.Ring`

        if multicast:
            raise NotImplementedError("tcgen05's pair of CTAs shares a product, not a multicast ring "
                                      "(`PAIR`: each CTA loads its half of a step's block)")
        if self.topology.paired and (gathered or any(count > 1 for count in (copies or {}).values())
                                     or any(tuple(shape[-2:]) == SCALE_TILE for shape, _ in blocks.values())):
            raise NotImplementedError("a pair's streams are its blocks whole, each CTA's half "
                                      "(`PairedRing`): no gathered rows, copies in pieces or block scales")
        # A gathered stream's rows are its load warp's four-row gathers (`gathered.Committed`).
        streams = {name: gathered_rows.ring(shape, dtype, stages, released_by_products=True)
                   if name in gathered else
                   rings.ring(shape, dtype, stages, swizzle=(swizzles or {}).get(name),
                               copies=(copies or {}).get(name, 1), released_by_products=True)
                   for name, (shape, dtype) in blocks.items()}
        # A block-scale stream's slots each pair with the tensor memory its products read them from
        # (`scale_memory`).
        for name, (shape, dtype) in blocks.items():
            if tuple(shape[-2:]) == SCALE_TILE:
                streams[name]['tmem'] = plgpu.TMEM(scale_memory(shape, stages), dtype,
                                                   layout=plgpu.TMEMLayout.SCALES_LAYOUT)
        return streams

    def opened(self, allocated: dict, views: dict | None = None) -> dict:
        from dew.nn.kernels.templates import gathered  # it builds on `rings.Ring`

        del views
        if self.topology.paired:
            return {name: PairedRing(**scoped, topology=self.topology) for name, scoped in allocated.items()}
        # A gathered stream's load barriers are shared memory of its own (`gathered.ring`).
        return {name: ScaleRing(**scoped) if 'tmem' in scoped
                else CommittedRing(**scoped) if isinstance(scoped['full'].dtype, plgpu_core.BarrierType)
                else gathered.Committed(**scoped) for name, scoped in allocated.items()}

    def persist_schedule(self, consumers: int, width: int, *, summed: bool = False,
                         rewrites: bool = False, producer: int | None = None) -> handoffs.Schedule:
        """A persistent walk's roles: its `consumers` epilogue warpgroups, a rewriting warpgroup where its
        steps' slots are rewritten (`persist`'s `prepared`), and the issuing warpgroup's warps, lowered to
        `producer` (`handoffs.Schedule.floor`); and its accumulator turns (`turns` of `width` columns,
        `Accumulators`, their running sums past them where exactly `summed`), which the product warp fills
        and the epilogue gives back."""
        rewriting = (handoffs.Role("rewrite", 1),) if rewrites else ()
        roles = (handoffs.Role("epilogue", consumers), *rewriting, *ISSUING)
        return handoffs.Schedule(roles, (handoffs.HandOff(
            "accumulators", Accumulators(self.turns, self.rows, width, self.topology, running=summed), "mma",
            ("epilogue",)),), floor=producer)

    def persist_scratch(self, consumers: int, rewrites: tuple[int, ...] = ()) -> tuple[tuple, int]:
        # Its accumulator turns' barriers, filled and drained (`persist`), each rewritten ring's slots'
        # publish barriers (`prepared`), and its tensor memory's address.
        (accumulators,) = self.persist_schedule(consumers, 0).stated()
        return ((*accumulators.barriers(), *_publish_barriers(rewrites)),
                resources.TMEM_ADDRESS_BYTES)

    def persist(self, streams, tiles: tuple[int, ...], schedule: Callable, issue: Callable, finish: Callable,
                *, consumers: int, axis: str, outputs: tuple, interpret: bool, in_flight: int = 0,
                unrolled: bool = False, accumulators: tuple = (), summed: int = 0,
                prepared: Callable | None = None, producer: int | None = None) -> None:
        """`mosaic.Family.persist`. With `prepared`, `prepared(tile, index, slots)` rewrites each step's slots
        before its products read them (a kernel gradient's neighbouring groups' rows zeroed), run by a role
        of its own past the consumers (`warpgroups`: the launch's consumers one more): it waits for each
        step's loads, rewrites them, publishes them to the tensor cores and arrives at the step's `masked`
        barrier, which the product warp waits for in the loads' place (a ring of three parties: the
        producer's loads, the rewrite, the products). With `summed`, an exactly summed launch's
        (`exact_sums`): each tile's
        steps `summed` at a time, a window, each window's products into a turn's partials of their own (its
        first product not accumulating), which the epilogue warpgroup promotes into the tile's running sums
        in tensor memory (`promoted`: each chunk of the partial and the running sum out, an IEEE add, the
        sum back) once their commit arrives, freeing the turn for the window after next; the tile's store
        reads the running sums. Each window is a turn (`persistent.tiled`'s `turns`), so the products of a
        window run while the epilogue promotes the one before."""
        if not accumulators:
            raise ValueError(f"{self._HOME}: a persistent launch states their shapes (`accumulators`)")
        if self.topology.paired and (summed or prepared is not None):
            raise NotImplementedError("a pair's tiles are a plain product's: no exact sums or rewritten "
                                      "steps")
        rows = {shape[0] for shape in accumulators}
        if rows != {self.rows}:
            raise ValueError(f"a tcgen05 accumulator is {self.rows} rows on one CTA, not {sorted(rows)}")
        del in_flight  # the products' commit releases each step's slots
        opened = self.opened(streams)
        for ring in opened.values():
            ring.initialized()
        staging = self.output_staging(outputs, consumers)
        rewrites = () if prepared is None else tuple(ring.stages for ring in opened.values())
        width = sum(shape[1] for shape in accumulators)
        offsets = [sum(shape[1] for shape in accumulators[:index]) for index in range(len(accumulators))]
        def windows(steps):
            return 1 if not summed else (steps + summed - 1) // summed

        roles = self.persist_schedule(consumers, width, summed=bool(summed), rewrites=prepared is not None,
                                      producer=producer)

        def scoped(*allocated):
            (tensor,) = roles.opened(allocated[:1])
            # The product warp reads each step's slots once the rewrite publishes them (`prepared`): each
            # slot's `masked` barrier in its loads' place.
            masked = dict(zip(opened, allocated[1:1 + len(rewrites)], strict=True)) if rewrites else {}
            smems = allocated[1 + len(rewrites):]
            issued = ({name: _published(ring, masked[name]) for name, ring in opened.items()} if masked
                      else opened)

            def held(turn: rings.Turn) -> tuple:
                # The turn's accumulators: each its columns of the turn's share of tensor memory.
                return tuple(Accumulator(tensor.columns(turn, offset, shape[1]), shape)
                             for offset, shape in zip(offsets, accumulators, strict=True))

            # An exactly summed tile's running sums: the columns past the turns'.
            running = tuple(Accumulator(tensor.running_columns(offset, shape[1]), shape)
                            for offset, shape in zip(offsets, accumulators, strict=True)) if summed else ()

            products = self._issued(issued, issue, tensor, held, running, summed=summed, interpret=interpret,
                                    unrolled=unrolled)

            def epilogue(index):
                store = self._stores(smems, index)

                def visit(counter, active, steps, views, tile, done_turns):
                    del counter, views

                    @pl.when(active)
                    def _():
                        into, turn = self._awaited(tensor, held, running, done_turns, steps, summed)

                        def walk(step, start, shapes, *, initial=None):
                            # The products issued it: the tile's accumulators (its running sums), done.
                            del step, initial
                            if tuple(shapes) != accumulators:
                                raise ValueError(f"a tile walks {shapes}, its launch stated {accumulators}")
                            return start, into

                        finish(index, tile, issue(index, tile, walk, active=True), store, active=True)
                        if turn is not None:
                            tensor.release(turn)
                        else:
                            plgpu.wait_load_tmem()

                    pl.when(~active)(lambda: finish(index, tile, issue(index, tile, persistent.unwalked(self),
                                                                       active=False), store, active=False))

                persistent.tiled(tiles, schedule, visit, axis=axis, turns=windows)
                plgpu.wait_smem_to_gmem(0, wait_read_only=True)

            def load():
                loads = functools.partial(rings.loads, opened)
                loaded, _ = persistent.tiled(tiles, schedule, loads, axis=axis)
                rings.drain(opened.values(), loaded)

            def issue_all():
                _, computed = persistent.tiled(tiles, schedule, products, axis=axis, turns=windows)
                tensor.drain(computed)

            rewriting = {}
            if prepared is not None:
                rewritten = functools.partial(_rewritten, opened, masked, prepared)

                def rewrite(index):
                    del index
                    persistent.tiled(tiles, schedule, rewritten, axis=axis)

                rewriting["rewrite"] = rewrite
            roles.run(epilogue=epilogue, load=load, mma=issue_all, **rewriting)

        (accumulated,) = roles.scratch(generation.gpu_target())
        # Each ring's slots' rewrites published, a barrier of the ring's own (`prepared`).
        pl.run_scoped(scoped, accumulated, *_publish_barriers(rewrites), *staging, collective_axes="wg")

    def _issued(self, rings, issue: Callable, tensor: "Tensor", held: Callable, running: tuple, *,
                summed: int, interpret: bool, unrolled: bool) -> Callable:
        """`persist`'s product warp's visit of a tile (`persistent.tiled`'s): each active tile's windows (its
        steps, where not `summed`) into a turn's accumulators once the window `turns` back has been read out
        of them, each window then a commit the epilogue waits on."""
        def products(counter, active, steps, views, tile, done_turns):
            del views

            @pl.when(active)
            def _():
                # A tile not summed takes one turn, freed and held before its walk is issued.
                turn = None if summed else tensor.turn(done_turns)
                if turn is not None:
                    tensor.freed(turn)

                def walked(lo, hi, start, shapes, accumulators: tuple, step):
                    def stepped(at, slots, carry, accs):
                        return step(at - counter, slots, carry, accs)

                    return self.walk(rings, ((stepped, lo, hi),), start, shapes, interpret,
                                     initial=accumulators, unrolled=unrolled)

                def walk(step, start, shapes, *, initial=None):
                    if initial is not None:
                        raise NotImplementedError(f"{self._HOME}; a walk starts it from no value")
                    if turn is not None:
                        return walked(counter, counter + steps, start, shapes, held(turn), step)

                    def window(index, carry):
                        each = tensor.turn(done_turns + index)
                        tensor.freed(each)
                        lo = counter + index * summed
                        carry, _ = walked(lo, jnp.minimum(lo + summed, counter + steps), carry, shapes,
                                          held(each), step)
                        tensor.issued(each)
                        return carry

                    return lax.fori_loop(0, (steps + summed - 1) // summed, window, start), running

                # A pair's leader issues each product over both CTAs, its commit arriving at both; each CTA's
                # product warp waits for its own turn's read, so both keep the read barrier's phases.
                @pl.when(self.topology.leads())
                def _():
                    issue(0, tile, walk, active=True)
                    if turn is not None:
                        tensor.issued(turn)

        return products

    def _awaited(self, tensor: "Tensor", held: Callable, running: tuple, done_turns, steps,
                 summed: int) -> tuple:
        """`persist`'s epilogue's wait for a tile's products: its turn's accumulators once their commit
        arrives, and the turn the epilogue gives back once it has read them; exactly summed (`summed`), each
        window's partials promoted into the running sums as its commit arrives, each window's turn given
        back then, the running sums the tile's accumulators and no turn left to give back."""
        if not summed:
            turn = tensor.turn(done_turns)
            tensor.ready(turn)
            return held(turn), turn

        def window(at, carry):
            turn = tensor.turn(done_turns + at)
            tensor.ready(turn)
            self._promoted(held(turn), running, at == 0)
            tensor.release(turn)
            return carry

        lax.fori_loop(0, (steps + summed - 1) // summed, window, None)
        return running, None

    def _promoted(self, partials: tuple, totals: tuple, first) -> None:
        """A window's `partials` added into the tile's running sums `totals` in tensor memory by an IEEE add
        (the first window's, where `first`, the running sums as they are), a chunk at a time, every load
        done and every store visible once it returns (`persist`'s `summed`)."""
        for partial, total in zip(partials, totals, strict=True):
            span = self.chunk_columns(total.shape[1])
            for start in range(0, total.shape[1], span):
                part = plgpu.async_load_tmem(column_span(partial.ref, start, span), layout=STORED)
                into = column_span(total.ref, start, span)
                # The first window's partial is the sum; a later one's adds onto it (a branch: a scalar
                # selects no tile in warpgroup semantics, and the sum's columns hold nothing before it).
                pl.when(first)(lambda into=into, part=part: plgpu.async_store_tmem(into, part))
                pl.when(~first)(lambda into=into, part=part: plgpu.async_store_tmem(
                    into, plgpu.async_load_tmem(into, layout=STORED) + part))
        plgpu.commit_tmem()
        plgpu.wait_load_tmem()

class Tcgen05PairRings(Tcgen05Rings):
    """tcgen05's bundle over a pair of CTAs (`PAIRED`, CUTLASS's 2-SM UMMA): each step's blocks the pair's,
    each CTA loading its half (`PairedRing`); the leader's product warp issuing each product over both CTAs'
    rows into both's tensor memory, allocated together; each CTA's epilogue reading its own rows, a turn freed
    once both have (`Accumulators`' cluster read barrier). Its tile is twice a CTA's rows and twice its right
    operand's columns: a CTA stages half of the product's columns of the right operand (`widest_product`, a
    CTA's block of them) and holds all of them for its own rows in tensor memory, as the product it shares
    writes them; no block scales (a scale tile's tensor memory is one CTA's) and no gathered rows (a pair's
    step block is copied partitioned, whole)."""

    name = "tcgen05.pair"
    topology = PAIRED
    cluster = PAIRED.ctas
    pairs_product = True
    widest_product = PRODUCT_COLUMNS // PAIRED.ctas
    takes_block_scales = False
    gathers_rows = False


RINGS = Tcgen05Rings()
"""The tcgen05 bundle (`bundles.of` on sm100 and sm103)."""

PAIR = Tcgen05PairRings()
"""The tcgen05 bundle over a pair of CTAs: a launch's config names it ('tcgen05.pair', `bundles`)."""


def _published(ring: rings.Ring, masked) -> CommittedRing:
    """`ring` as the product warp reads it where a role rewrites its slots (`Tcgen05Rings.persist`'s
    `prepared`): each slot awaited at `masked`, the rewrite's publish barrier, in its loads' place, and
    released by the products' commit; a gathered ring's (its loads' barriers its own mbarriers,
    `gathered.Committed`) a committed ring of its slots and release barrier."""
    if isinstance(ring, CommittedRing):
        return dataclasses.replace(ring, full=masked)
    return CommittedRing(smem=ring.smem, full=masked, consumed=ring.consumed)


def _publish_barriers(rewrites: tuple[int, ...]) -> tuple:
    """Each rewritten ring's slots' publish barriers (`Tcgen05Rings.persist`'s `prepared`), by its slot count:
    the one statement `persist` scopes and its planner prices (`persist_scratch`)."""
    return tuple(plgpu.Barrier(num_barriers=stages) for stages in rewrites)


def _rewritten(opened: dict, masked: dict, prepared: Callable, counter, active, steps, views, tile,
               done) -> None:
    """A rewriting role's visit of a persistent tile (`Tcgen05Rings.persist`'s `prepared`): each of its steps'
    slots, once loaded, rewritten (`prepared(tile, index, slots)`), published to the tensor cores and each
    ring's `masked` barrier of the step arrived at; none where the tile streams nothing."""
    del views, done

    @pl.when(active)
    def _():
        def step(index, carry):
            turn = rings.Turn.of(counter + index, min(ring.stages for ring in opened.values()))
            slots = {name: ring.acquire(turn) for name, ring in opened.items()}
            prepared(tile, index, slots)
            plgpu.commit_smem()
            for barriers in masked.values():
                plgpu.barrier_arrive(barriers.at[rings.opaque(turn.slot)])
            return carry

        lax.fori_loop(0, steps, step, None)


class Tcgen05ChunksIssuer(mosaic_sm90.WgmmaLayout, mosaic.Issuer[Accumulator | jax.Array]):
    """tcgen05's issuer for 64-row tiles a consumer holds in registers (the recurrence's chunks, row 4747):
    each product into a scoped 64-row fp32 accumulator in tensor memory (one CTA's M=64 MMA, the half-lane
    layout `tcgen05.tmem_half_lane_layout`), its left operand from shared memory or, written there first,
    from tensor memory (the TS form, `products`' `Accumulator`), its right from shared memory; awaited by a
    tensor-core barrier its commit arrives at, then read into registers in the WGMMA layout, which a
    half-lane tile loads natively (`WgmmaLayout`). Its walks carry their sums in registers, as wgmma's
    consumers do: no accumulator outlives its product's read. Mosaic GPU's interpreter models tensor memory
    and tcgen05's products, so interpreted it runs the same."""

    name = "tcgen05.chunks"
    loads_past_floor = True
    # Mosaic GPU's tcgen05 lowering takes no TF32 operand ("Unsupported input element type: tf32",
    # jsf-gate100b): an fp32 product takes bf16 parts.
    takes_fp8 = True
    takes_register_left = False
    # A B200's training forward on three consumers at 64 x 64, width 64, priced 158 of 160 registers with
    # its products read back twice and spilled 20 bytes (et-h1gate3): its products' addresses and phase.
    product_control = 8
    rows = 64
    role_rows = 64
    widest_product = 256
    operands_in_registers = False
    products_overlap = False  # each product is awaited where its value is read (`summed`)

    def narrowest_columns(self, dtype) -> int:
        # A 64-row product splits its N columns between two lane groups, each a whole tile of 16 (Mosaic
        # GPU: "the M or N MMA instruction size was chosen to be 24, which is not a multiple of the tiling of
        # the non-contracting dimension 16" at 48 columns, and a 16-column one's swizzle past its half's 8,
        # jsf-gate100b): a staged block's columns step in 32.
        return max(2 * 16, super().narrowest_columns(dtype))

    def right_swizzle(self, dtype, columns: int) -> int:
        # A 64-row product splits its N columns between two lane groups, each reading whole tiles of its
        # right operand's swizzle (Mosaic GPU: "the M or N MMA instruction size was chosen to be 96, which is
        # not a multiple of the tiling of the non-contracting dimension 64" at 192 columns, jsf-trace100b).
        half = columns // 2
        return next((swizzle for swizzle in (swizzling.SWIZZLE_BYTES, swizzling.SWIZZLE_BYTES // 2,
                                             swizzling.NARROWEST_SWIZZLE_BYTES)
                     if half % swizzling.swizzle_columns(dtype, swizzle) == 0),
                    swizzling.NARROWEST_SWIZZLE_BYTES)

    def product_scratch(self, consumers: int, widths: tuple[int, ...], lefts: tuple = ()) -> dict:
        # A role's accumulator of each product width: a half-lane tile's columns split between its two lane
        # groups, so a narrower product's is no column slice of a wider one's (Mosaic GPU's layout inference
        # refused one, jsf-tmem4). A 16-column product is issued 32 wide, its right operand padded
        # (`_padded`), its sum's first columns read (`summed`). A role's shared tile of each register left
        # operand's depth and dtype in `lefts` (`register_products` stages one there).
        widths = tuple(sorted({max(width, PADDED) for width in widths}))
        return {"tmem": {width: tuple(accumulator_columns(self.rows, width) for _ in range(consumers))
                         for width in widths},
                "padded": tuple(plgpu.SMEM((self.rows, PADDED), jnp.bfloat16,
                                           transforms=swizzling.swizzle_transforms(jnp.bfloat16, PADDED))
                                for _ in range(consumers)),
                "lefts": {_left_key(depth, dtype): tuple(
                    plgpu.SMEM((self.rows, depth), dtype,
                               transforms=swizzling.swizzle_transforms(
                                   dtype, swizzling.swizzle_for(dtype, depth)))
                    for _ in range(consumers)) for depth, dtype in dict.fromkeys(lefts)},
                "done": plgpu.Barrier(num_barriers=consumers, orders_tensor_core=True)}

    def product_addresses(self, products: tuple | None) -> int:
        # Its accumulators' tensor memory keeps its address in a word: every launch's compile was those 4
        # bytes past its plan (the sm100 gate, jsf-g100fix).
        return 0 if products is None else resources.TMEM_ADDRESS_BYTES

    def product_shared(self, scratch) -> tuple:
        # Its accumulators are tensor memory: its padded right operands, its staged left operands and its
        # barrier are shared memory, as `product_scratch` allocates them.
        return (*scratch["padded"], *(tile for tiles in scratch["lefts"].values() for tile in tiles),
                scratch["done"])

    @contextlib.contextmanager
    def products_in(self, refs):
        token = _PRODUCTS.set(refs)
        try:
            yield
        finally:
            _PRODUCTS.reset(token)

    def _scoped(self) -> dict:
        """The program's product scratch (`product_scratch`'s, scoped, `products_in`)."""
        found = _PRODUCTS.get()
        if found is None:
            raise ValueError(f"{self.name}'s products take their program's product scratch (`products_in`)")
        return found

    def _scratch(self, width: int):
        """The building role's accumulator `width` columns wide (`PADDED` at least) and its barrier
        (`product_scratch`'s)."""
        width = max(width, PADDED)
        found = self._scoped()
        if width not in found["tmem"]:
            raise ValueError(f"{self.name} has no {width}-column accumulator in its program's product "
                             f"scratch (`product_scratch`'s widths)")
        role = mosaic.ROLE.get()
        return found["tmem"][width][role], found["done"].at[role]

    def summed(self, shape: tuple[int, int], issue) -> jax.Array:
        tmem, done = self._scratch(shape[1])
        issue(Accumulator(tmem, (shape[0], max(shape[1], PADDED))))
        found = _awaited(tmem, done)
        return found if shape[1] >= PADDED else found[:, :shape[1]]

    def products(self, acc: Accumulator | jax.Array, lhs, rhs, *, accumulate: bool = True
                 ) -> Accumulator | jax.Array:
        if not isinstance(acc, Accumulator):
            # A sum the walk carries in registers: the product in a fresh accumulator, added to it.
            return acc + self.summed((acc.shape[0], acc.shape[1]),
                                     lambda fresh: self.products(fresh, lhs, rhs))
        left = _tensor_memory(lhs) if isinstance(lhs, Accumulator) else lhs
        if rhs.shape[1] < PADDED:
            rhs = self._padded(rhs)
        plgpu.tcgen05_mma(_tensor_memory(acc), left, rhs, accumulate=acc.accumulate if accumulate else False)
        return dataclasses.replace(acc, accumulate=True)

    def _padded(self, rhs):
        """A right operand of fewer than `PADDED` columns in the building role's padded tile, zeros past
        its columns: a 64-row product splits its columns between two lane groups, and 16's halves of 8 are
        whole tiles of no shared-memory tiling a transposed operand takes (Mosaic GPU's layout inference
        refused Kimi delta attention's 16-wide sub-chunk pairs on a B200; row 4747's debugging at
        0e5d3aa9df8: padded, they lower)."""
        padded = self._scoped()["padded"][mosaic.ROLE.get()]
        rows, columns = rhs.shape
        if rows != self.rows:
            raise ValueError(f"{self.name} pads a {self.rows}-row right operand, not {rows} rows")
        padded[...] = jnp.zeros((rows, PADDED), padded.dtype)
        plgpu.store(padded.at[:, :columns], plgpu.load(rhs, layout=plgpu.Layout.WGMMA, optimized=False),
                    optimized=False)
        plgpu.commit_smem()
        return padded

    def register_products(self, acc: Accumulator | jax.Array, parts, rhs) -> Accumulator | jax.Array:
        # A 64-row product takes no left operand from tensor memory (tcgen05.py: M=128 alone) nor registers:
        # each part is staged into the building role's shared tile of its depth and dtype
        # (`product_scratch`'s lefts), made visible to the tensor core's proxy, multiplied from there, and
        # its product's commit awaited before the next part, or the caller, may write the tile again (the
        # tensor core reads it as it runs). A sum the walk carries in registers takes the parts' products
        # in a fresh accumulator, added to it (`summed`).
        if not isinstance(acc, Accumulator):
            return acc + self.summed((acc.shape[0], acc.shape[1]), lambda fresh: self.register_products(
                fresh, parts, rhs))
        _, done = self._scratch(acc.shape[1])
        for part in parts:
            tile = self._left(part)
            plgpu.store(tile, part, optimized=False)
            plgpu.commit_smem()
            acc = self.products(acc, tile, rhs)
            plgpu.tcgen05_commit_arrive(done)
            plgpu.barrier_wait(done)
        return acc

    def _left(self, part: jax.Array):
        """The building role's shared tile for a register left operand like `part` (`register_products`)."""
        rows, depth = part.shape
        tiles = self._scoped()["lefts"].get(_left_key(depth, part.dtype))
        if tiles is None or rows != self.rows:
            raise ValueError(f"{self.name} stages a {self.rows}-row register left operand of its program's "
                             f"product scratch (`product_scratch`'s lefts), not {part.shape} {part.dtype}")
        return tiles[mosaic.ROLE.get()]

    def registers_product(self, lhs: jax.Array, rhs) -> jax.Array:
        return self.summed((lhs.shape[0], rhs.shape[1]), lambda acc: self.register_products(
            acc, (lhs,), rhs))

    def product(self, lhs, rhs, shape: tuple[int, int], then=None) -> jax.Array:
        def issue(acc):
            self.products(acc, lhs, rhs)
            if then is not None:
                then()

        return self.summed(shape, issue)

    def added_into(self, acc: Accumulator | jax.Array, value: jax.Array, interpret: bool
                   ) -> Accumulator | jax.Array:
        # Its products done (a commit awaited), its value read, `value` added, written back; a sum the
        # walk carries in registers takes `value` there.
        if not isinstance(acc, Accumulator):
            return acc + value
        del interpret
        _, done = self._scratch(acc.shape[1])
        plgpu.tcgen05_commit_arrive(done)
        plgpu.barrier_wait(done)
        plgpu.async_store_tmem(_tensor_memory(acc), _read(_tensor_memory(acc)) + value)
        plgpu.commit_tmem()
        return dataclasses.replace(acc, accumulate=True)

    def rescaled_products(self, acc: Accumulator | jax.Array, scale, parts, rhs, interpret: bool
                          ) -> Accumulator | jax.Array:
        if not isinstance(acc, Accumulator):
            return self.register_products(scale(acc), parts, rhs)
        _, done = self._scratch(acc.shape[1])
        plgpu.tcgen05_commit_arrive(done)
        plgpu.barrier_wait(done)
        plgpu.async_store_tmem(_tensor_memory(acc), scale(_read(_tensor_memory(acc))))
        plgpu.commit_tmem()
        return self.register_products(dataclasses.replace(acc, accumulate=True), parts, rhs)

    def settled(self, pending: int = 0) -> None:
        # A product reads its shared-memory operands after its issue returns, until its commit arrives (PTX
        # ISA, tcgen05.mma: an operand is not written before then), so a buffer staged again under it (a
        # second value block's right operand, `operands._staged`) is read half overwritten: every product
        # the role issued committed and awaited. tcgen05 commits all or none: `pending` (warpgroup MMA's
        # groups left running) is none here. Outside its program's product scratch no product of it runs.
        del pending
        if _PRODUCTS.get() is None:
            return
        done = self._scoped()["done"].at[mosaic.ROLE.get()]
        plgpu.tcgen05_commit_arrive(done)
        plgpu.barrier_wait(done)


class Tcgen05Chunks(Tcgen05ChunksIssuer, mosaic.RingFamily):
    """The recurrence's bundle on Blackwell (row 4747): tcgen05's chunks issuer (`Tcgen05ChunksIssuer`) on TMA
    rings fed by a producer warpgroup (`mosaic.RingFamily`), as wgmma's bundle is: the one composition in
    which its products are the family's own, mma.sync's on TMA rings spilling its key-block, fp32 solve and
    scores' gradient launches at every geometry on a B200."""

    consumers = 3

    def streams(self, blocks: dict, stages: int, consumers: int, multicast: tuple[str, ...] = (),
                swizzles: dict[str, int] | None = None, copies: dict[str, int] | None = None,
                gathered: tuple[str, ...] = ()) -> dict:
        # A 64-row product reads a transposed operand as two halves of 32 rows and splits its columns
        # between two lane groups, each whole tiles of its swizzle: at 128 bytes the keys' (64, 8) transposed
        # tiling divides no half (Mosaic GPU's layout inference refused a 27B gated delta net's fp32 forward
        # on a B200; row 4747's debugging at 0e5d3aa9df8: at 64 bytes it lowers and compiles), a 96-column
        # operand's halves of 48 no 32-column tile (`right_swizzle`), a role's share of a stream's columns
        # (the SSD's values two roles read 32 of 64 each) whole tiles of its halves, and a stream read as a
        # transposed right operand (Kimi delta attention's memories, 32 of their rows its columns) whole
        # tiles of its narrower side's halves (the debugging's SSD and KDA cases).
        capped = {name: min((swizzles or {}).get(name, resources.widest_swizzle(dtype, shape[-1])),
                            swizzling.SWIZZLE_BYTES // 2, self.right_swizzle(dtype, shape[-1] // consumers),
                            self.right_swizzle(dtype, min(shape)))
                  for name, (shape, dtype) in blocks.items() if len(shape) == 2}
        return super().streams(blocks, stages, consumers, multicast, {**(swizzles or {}), **capped}, copies,
                               gathered)


PADDED = 32
"""The fewest columns a 64-row tcgen05 product takes (`Tcgen05ChunksIssuer._padded`)."""


def _left_key(depth: int, dtype) -> str:
    """A staged left operand's tile's name in a program's product scratch (`product_scratch`'s lefts)."""
    return f"{depth}:{jnp.dtype(dtype).name}"

CHUNKS = Tcgen05Chunks()
"""tcgen05's bundle for the recurrence's 64-row chunks."""

_PRODUCTS: contextvars.ContextVar[dict | None] = contextvars.ContextVar("products", default=None)
"""The program's product scratch while it is traced (`Tcgen05ChunksIssuer.products_in`)."""


def _read(tmem) -> jax.Array:
    """A 64-row accumulator's value in registers, in the WGMMA layout its half-lane tile loads into."""
    value = plgpu.async_load_tmem(tmem, layout=plgpu.Layout.WGMMA)
    plgpu.wait_load_tmem()
    return value


def _awaited(tmem, done) -> jax.Array:
    """The products issued into `tmem` committed to `done`, awaited, and their sum read (`_read`)."""
    plgpu.tcgen05_commit_arrive(done)
    plgpu.barrier_wait(done)
    return _read(tmem)


def _tensor_memory(acc: Accumulator) -> Ref:
    """A tile's tensor memory; a tile that multiplies nothing has none to issue into or write."""
    if acc.ref is None:
        raise ValueError("a tile that multiplies nothing holds no tensor memory")
    return acc.ref

