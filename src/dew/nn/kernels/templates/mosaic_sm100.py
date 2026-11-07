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

import dataclasses
import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax._src.state import types as state_types
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import generation
from dew.nn.kernels.templates import mosaic, resources
from dew.nn.kernels.templates.mosaic import Ref

ISSUE, READ = 0, 1
"""A program's warpgroups: the one issuing loads and products, and the one reading accumulators."""
LOAD_WARP, MMA_WARP = 0, 1
"""The issuing warpgroup's warps: the TMA loads' and the products'."""
TMEM = plgpu.Layout.TCGEN05_TMEM_NATIVE
"""A tile's register layout as tensor memory loads it natively (a thread a row)."""
STORED = plgpu.Layout.TCGEN05
"""An accumulator's register layout for an epilogue, the one a swizzled shared-memory tile takes."""
WARPGROUPS = 2
"""A program's warpgroups (`roles`)."""


@dataclasses.dataclass(frozen=True)
class Topology:
    """The CTAs one product spans (tcgen05's collective form): one, or a pair of a cluster's
    (`mosaic.CLUSTER`), whose leader issues each product over both's rows into both's tensor memory, each
    CTA loading its own rows' block of one operand and its half of the other's, the leader's load barriers
    tracking both halves and its products' commits arriving at both (CUTLASS's 2-SM UMMA)."""
    ctas: int = 1

    @property
    def paired(self) -> bool:
        return self.ctas == 2

    def index(self):
        """This CTA's place in the pair (0, the leader, without one)."""
        return lax.axis_index(mosaic.CLUSTER) if self.paired else 0

    def leads(self):
        """Whether this CTA issues the products."""
        return lax.axis_index(mosaic.CLUSTER) == 0 if self.paired else True

    def copied(self, dimension: int) -> dict:
        """A load's partition across the pair: each CTA its share of the block along `dimension`, tracked
        by the leader's barrier."""
        if not self.paired:
            return {}
        return {'leader_tracked': plgpu.CopyPartition.PARTITIONED(dimension),
                'collective_axes': mosaic.CLUSTER}

    def issued(self) -> dict:
        """A product's or a commit's spread over the pair."""
        return {'collective_axis': mosaic.CLUSTER} if self.paired else {}


SINGLE = Topology()
"""A product on one CTA."""


@dataclasses.dataclass(frozen=True)
class Tcgen05:
    """The tcgen05 bundle's facts and pieces (the module's account)."""

    name = "tcgen05"
    rows = 128
    """A product's rows on one CTA (tcgen05's M without a CTA pair)."""
    consumers = 1
    """The warpgroups reading accumulators: one."""
    cluster = 1

    @staticmethod
    def registers() -> int:
        """The registers a thread of a program holds: its warpgroups' entry share (setmaxnreg moves none
        where the GPU's compile ignores it, `resources.RAISES_HONORED`)."""
        return resources.entry_registers(WARPGROUPS)

    @staticmethod
    def columns(target: generation.GpuTarget) -> int:
        """Tensor memory's columns a multiprocessor holds (PTX ISA, tcgen05.alloc: 512)."""
        if target.tmem_columns is None:
            raise ValueError(f"{target.generation} has no tensor memory")
        return target.tmem_columns

    @staticmethod
    def kernel(entry: Callable, *, out_type, grid: tuple, grid_names: tuple, scratch: tuple,
               interpret, name: str | None = None, topology: Topology = SINGLE,
               warpgroups: int = WARPGROUPS) -> Callable:
        """`entry` as a program of `warpgroups` warpgroups (axis 'wg', `WARPGROUPS` by default: a template
        whose roles are more, `mosaic.tabled`'s), its kernel named `name`, `grid` the clusters of
        `topology`'s CTAs (one each without a pair)."""
        return mosaic.kernel(entry, out_type=out_type, grid=grid, grid_names=grid_names,
                             warpgroups=warpgroups, family=Tcgen05.name, scratch=scratch,
                             interpret=interpret, name=name, cluster=topology.ctas)

    @staticmethod
    def roles(load: Callable[[], None], products: Callable[[], None], read: Callable[[], None]) -> None:
        """The issuing warpgroup's load warp runs `load`, its product warp `products`; the other warpgroup
        runs `read`."""
        Tcgen05.issued(load, products)
        pl.when(lax.axis_index("wg") == READ)(read)

    @staticmethod
    def product(acc, lhs, rhs, released, *, accumulate, topology: Topology = SINGLE) -> None:
        """`acc` (`+`)= `lhs @ rhs`, issued by the product warp into tensor memory (onto `acc` where
        `accumulate`), `released` (a stage's release barrier) arrived at once the product has read its
        operands; over `topology`'s CTAs, the leader's product spanning both."""
        plgpu.tcgen05_mma(acc, lhs, rhs, released, accumulate=accumulate, **topology.issued())

    @staticmethod
    def load_columns(tmem: Ref, first, count: int, layout=STORED) -> jax.Array:
        """Columns `first` to `first + count` of tensor memory `tmem`, every row, loaded in `layout` (the
        load in flight until a read of its value or `Tensor.release`)."""
        return plgpu.async_load_tmem(columns(tmem, first, count), layout=layout)

    @staticmethod
    def store_columns(tmem: Ref, first, value: jax.Array) -> None:
        """`value` into tensor memory `tmem`'s columns from `first`, every row, visible to a product once a
        signal publishes it (`Signal.published`)."""
        plgpu.async_store_tmem(columns(tmem, first, value.shape[-1]), value)

    @staticmethod
    def loaded() -> None:
        """Every load of tensor memory this thread issued done (`load_columns`), before its columns are
        reused or released."""
        plgpu.wait_load_tmem()

    @staticmethod
    def relaid(value: jax.Array, smem, layout) -> jax.Array:
        """`value` in `layout`, through the shared buffer `smem` (a row vector's tensor-memory layout to an
        epilogue's, which no register relayout reaches)."""
        smem[...] = value
        return plgpu.load(smem, layout=layout)

    @staticmethod
    def zeros(shape: tuple[int, int]) -> jax.Array:
        """An fp32 accumulator chunk of zeros in an epilogue's layout (`STORED`): an output a tile that
        multiplies nothing stores."""
        return plgpu.layout_cast(jnp.zeros(shape, jnp.float32), STORED)

    @staticmethod
    def issued(load: Callable[[], None], products: Callable[[], None]) -> None:
        """The issuing warpgroup's warps alone (a program's end, `drain`'s, and each tile's in `roles`): its
        load warp runs `load`, its product warp `products`."""
        pl.when(lax.axis_index("wg") == ISSUE)(lambda: _issuing(load, products))

    @staticmethod
    def issuing(load: Callable[[], None], products: Callable[[], None]) -> None:
        """The issuing warpgroup's warps where a role table places it (`mosaic.tabled`'s producer, after the
        template's rows): its load warp runs `load`, its product warp `products`."""
        _issuing(load, products)


TCGEN05 = Tcgen05()
"""The bundle, which the templates' Blackwell mainloops import (no route reaches them yet)."""


def swizzled(dtype, contraction: int) -> tuple:
    """A stage's transforms for an operand block of `dtype` contracting `contraction` elements a step: 8-row
    tiles of the widest swizzle its rows take, swizzled (tcgen05 reads its shared operands so)."""
    return mosaic.swizzle_transforms(dtype, resources.widest_swizzle(dtype, contraction))


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
    arrival (`tensor_core`), else a load's."""
    count: int = 1
    tensor_core: bool = True

    def scratch(self):
        return plgpu.Barrier(num_barriers=self.count, orders_tensor_core=self.tensor_core)

    def opened(self, barriers) -> "Signal":
        return Signal(self, barriers)


@dataclasses.dataclass(frozen=True)
class Signal:
    """An opened signal (`Signals.opened`): each turn arrived at once, by a load, a product's commit or a
    role's threads, and waited for at its turn's parity by every role that waits on it, at every turn."""
    shape: Signals
    barriers: state_types.AbstractRef

    def turn(self, counter) -> mosaic.Turn:
        return mosaic.Turn.of(counter, self.shape.count)

    def loaded(self, turn: mosaic.Turn, view, ref) -> None:
        """`view` copied into `ref` by TMA, its arrival `turn`'s."""
        plgpu.copy_gmem_to_smem(view, ref, self.barriers.at[turn.slot])

    def committed(self, turn: mosaic.Turn, topology: Topology = SINGLE) -> None:
        """`turn`'s arrival once every product the thread issued is done (the tensor core's commit)."""
        plgpu.tcgen05_commit_arrive(self.barriers.at[turn.slot], **topology.issued())

    def published(self, turn: mosaic.Turn) -> None:
        """`turn`'s arrival by the role's threads, their tensor-memory stores visible to the products first
        (`TCGEN05.store_columns`)."""
        plgpu.commit_tmem()
        plgpu.barrier_arrive(self.barriers.at[turn.slot])

    def wait(self, turn: mosaic.Turn) -> None:
        plgpu.barrier_wait(self.barriers.at[turn.slot], parity=turn.parity)


@dataclasses.dataclass(frozen=True)
class Accumulators:
    """`count` fp32 accumulators of `rows` x `columns` in tensor memory side by side, a product barrier
    each the product warp commits to once a tile's products are issued and a read barrier each the reading
    warpgroup arrives at once it has read one out (both tensor-core ordered). A turn is a tile's index
    over the program's tiles that compute; its accumulator is the turn's remainder. Over a pair of CTAs
    (`topology`) each holds its rows' accumulators, allocated together, and an accumulator is free once both
    CTAs have read theirs out (its read barrier the cluster's)."""
    count: int
    rows: int
    columns: int
    topology: Topology = SINGLE

    def scratch(self, target: generation.GpuTarget) -> tuple:
        if self.count * self.columns > TCGEN05.columns(target):
            raise ValueError(f"{self.count} accumulators of {self.columns} columns exceed tensor memory")
        tensor_core = functools.partial(plgpu.Barrier, orders_tensor_core=True)
        read = (plgpu.ClusterBarrier(collective_axes=(mosaic.CLUSTER,), num_barriers=self.count,
                                     orders_tensor_core=True)
                if self.topology.paired else tensor_core(num_barriers=self.count))
        return (plgpu.TMEM((self.rows, self.count * self.columns), jnp.float32,
                           collective=self.topology.paired),
                tensor_core(num_barriers=self.count), read)

    def opened(self, tmem, done, read) -> "Tensor":
        return Tensor(self, tmem, done, read)


@dataclasses.dataclass(frozen=True)
class Tensor:
    """Opened accumulators (`Accumulators.opened`)."""
    shape: Accumulators
    tmem: Ref
    done: state_types.AbstractRef
    read_done: state_types.AbstractRef

    def turn(self, counter) -> mosaic.Turn:
        """Turn `counter` round the accumulators (`mosaic.Turn`, its slot the accumulator's index)."""
        return mosaic.Turn.of(counter, self.shape.count)

    def acquire(self, turn: mosaic.Turn):
        """The product warp's turn: the turn's accumulator once the turn `count` before has been read
        out of it."""
        pl.when(turn.counter >= self.shape.count)(lambda: plgpu.barrier_wait(
            self.read_done.at[turn.slot], parity=turn.freed))
        return columns(self.tmem, turn.slot * self.shape.columns, self.shape.columns)

    def issued(self, turn: mosaic.Turn) -> None:
        """The product warp's turn's products all issued: the reader may wait on them."""
        plgpu.tcgen05_commit_arrive(self.done.at[turn.slot], **self.shape.topology.issued())

    def read(self, turn: mosaic.Turn, chunk: int, layout=STORED) -> Callable[[int], jax.Array]:
        """The reading warpgroup's turn: once its products are done, each `chunk`-column slice of its
        accumulator by its index, loaded from tensor memory in `layout`."""
        plgpu.barrier_wait(self.done.at[turn.slot], parity=turn.parity)
        first = turn.slot * self.shape.columns
        return lambda part: plgpu.async_load_tmem(columns(self.tmem, first + part * chunk, chunk),
                                                  layout=layout)

    def release(self, turn: mosaic.Turn) -> None:
        """The reading warpgroup's turn read out: its accumulator free for the turn `count` after."""
        plgpu.wait_load_tmem()
        plgpu.barrier_arrive(self.read_done.at[turn.slot])

    def drain(self, turns) -> None:
        """The product warp's last reads, which no later product waited for (`mosaic.drained`)."""
        mosaic.drained(self.read_done, turns, self.shape.count)


def columns(tmem, first, count: int):
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


class CommittedRing(mosaic.Ring):
    """A ring whose slots tcgen05's products give back: the product warp's commit arrives at a slot's release
    once every product it issued has read the slot (`tcgen05_commit_arrive`), where a register family's
    consumers arrive themselves (`mosaic.ring`'s `released_by_products`)."""

    def release(self, turn: mosaic.Turn) -> None:
        plgpu.tcgen05_commit_arrive(self.consumed.at[mosaic.opaque(turn.slot)])


SCALES = jnp.dtype(jnp.float8_e8m0fnu)
"""A block-scale stream's dtype (`Tcgen05Rings.streams`): an MX product's E8M0 scales, tiled in shared memory
as `tcgen05.scales_layout` lays them out, `[rows / 128, depth / 128, 32, 16]` a step's block."""


@dataclasses.dataclass(frozen=True)
class ScaleRing(CommittedRing):
    """A block-scale stream's ring (`SCALES`): each slot's shared-memory tiles paired with the tensor memory
    its products read them from (`Tcgen05Issuer.scaled_products`), a slot's columns of `tmem`. A slot's
    scales are copied to its tensor memory by the product warp before its products (tcgen05.cp then
    tcgen05.mma, pipelined in issue order: PTX ISA 9.7.18.6.2), and again only once its products' commit has
    released the slot and the producer refilled it, so no copy overwrites scales a product still reads."""

    tmem: Any = None

    def acquire(self, turn: mosaic.Turn):
        return super().acquire(turn), self._scales(turn)

    def slot(self, turn: mosaic.Turn):
        return super().slot(turn), self._scales(turn)

    def _scales(self, turn: mosaic.Turn):
        """`turn`'s slot's tensor memory: its columns of `tmem`."""
        columns = self.tmem.shape[1] // self.smem.shape[0]
        return self.tmem.at[:, pl.ds(mosaic.opaque(turn.slot) * columns, columns)]


CHUNK_COLUMNS = 64
"""The accumulator columns an epilogue reads out of tensor memory at once: 64 fp32 registers a thread of its
warpgroup, the GEMM epilogue's chunk (`chunk_columns`)."""


class Tcgen05Issuer(mosaic.Issuer[Accumulator]):
    """tcgen05's issuer and home: products issued by one warp into fp32 accumulators in tensor memory
    (`Accumulator`), read from shared memory, asynchronous; an epilogue reads the accumulators out in column
    chunks (`chunks`, in an epilogue's register layout, `STORED`)."""

    name = "tcgen05"

    rows = 128  # a product's rows on one CTA (tcgen05's M without a CTA pair)

    role_rows = 128

    accumulator_rows = rows

    issues_in_consumers = False

    counts_hooks = False  # its model is an epilogue chunk's accumulator alone (`chunk_columns`)

    takes_pairs = False

    takes_block_scales = True

    operands_in_registers = False

    products_overlap = True

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

    def chunks(self, accumulators: tuple) -> tuple:
        # Each chunk's columns read from each accumulator's tensor memory (`read`).
        width = accumulators[0].shape[1]
        columns = self.chunk_columns(width)
        return tuple((first, tuple(self.read(acc, first=first, columns=columns) for acc in accumulators))
                     for first in range(0, width, columns))

    def read(self, acc: Accumulator, *, first: int = 0, columns: int | None = None, pending: int = 0
             ) -> jax.Array:
        # Its columns out of tensor memory in an epilogue's layout (`STORED`), the load in flight until the
        # value is read or `Signal.published`; zeros for a tile that multiplies nothing. Its products are
        # awaited by a signal their commit arrives at (`signals`), not by `pending`.
        del pending
        rows, width = acc.shape
        columns = width - first if columns is None else columns
        if acc.ref is None:
            return self.zeros((rows, columns))
        return plgpu.async_load_tmem(_columns(acc.ref, first, columns), layout=STORED)

    def write(self, acc: Accumulator, value: jax.Array, *, first: int = 0) -> None:
        """`value` into the tile's columns from `first` in tensor memory, in its dtype, visible to a product
        once a signal publishes it (`Signal.published`): a softmax's narrowed weights for P·V, a state
        rescaled in place."""
        plgpu.async_store_tmem(_columns(_tensor_memory(acc), first, value.shape[-1]),
                               value.astype(jnp.dtype(acc.dtype)))

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
        # a tile that walks no step has none, its chunks zeros (`mosaic.unwalked`).
        if initial is not None:
            if tuple(acc.shape for acc in initial) != tuple(shapes):
                given = [acc.shape for acc in initial]
                raise ValueError(f"a walk of {shapes} was given tensor memory of {given}")
            return tuple(initial)
        return tuple(Accumulator(None, shape) for shape in shapes)

    def accumulated[C](self, segments, start: C, shapes: tuple, interpret: bool, initial: tuple | None = None,
                       finish=None) -> tuple[C, tuple]:
        # The accumulators stay in tensor memory, out of the loops' carries: a step's products add onto them
        # but at the walk's first step.
        del interpret
        accumulators = self.started(shapes, initial)
        if any(acc.ref is None for acc in accumulators):
            raise ValueError(f"{self._HOME}: a walk takes the tensor memory its persist gives it")
        first = segments[0].lo if segments else 0
        carry = start
        for loop in segments:
            def stepped(index, carry, step=loop.step):
                at = tuple(dataclasses.replace(acc, accumulate=index > first) for acc in accumulators)
                return step(index, carry, at)[0]

            carry = lax.fori_loop(loop.lo, loop.hi, stepped, carry)
            if loop.then is not None:
                loop.then()
        return (carry, accumulators) if finish is None else finish(carry, accumulators)

    def products(self, acc: Accumulator, lhs, rhs, interpret: bool, *, accumulate: bool = True
                 ) -> Accumulator:
        # The left operand from shared memory, or from tensor memory (an `Accumulator` a role wrote,
        # tcgen05's TS form: FlashAttention 4's P·V); the right from shared memory.
        del interpret  # Mosaic GPU's interpreter models tensor memory and tcgen05's products
        left = _tensor_memory(lhs) if isinstance(lhs, Accumulator) else lhs
        plgpu.tcgen05_mma(_tensor_memory(acc), left, rhs, accumulate=acc.accumulate if accumulate else False)
        return dataclasses.replace(acc, accumulate=True)

    def scaled_products(self, acc: Accumulator, lhs, rhs, scales: tuple, interpret: bool) -> Accumulator:
        """`products` of block-scaled operands (an MX product's: each 32 depths of `lhs`'s rows and `rhs`'s
        columns times its E8M0 scale), `scales` the step's two `ScaleRing` slots, each its shared-memory tiles
        and their tensor memory: both copied in, then the products read them (`tcgen05_mma`'s `a_scale`,
        `b_scale`)."""
        del interpret  # Mosaic GPU's interpreter models scaled tcgen05 products
        (a_tiles, a_scale), (b_tiles, b_scale) = scales
        plgpu.async_copy_scales_to_tmem(a_tiles, a_scale)
        plgpu.async_copy_scales_to_tmem(b_tiles, b_scale)
        plgpu.tcgen05_mma(_tensor_memory(acc), lhs, rhs, a_scale=a_scale, b_scale=b_scale,
                          accumulate=acc.accumulate)
        return dataclasses.replace(acc, accumulate=True)

    def settled(self, pending: int = 0) -> None:
        # The walk's release of a step's slots is its products' commit (`CommittedRing`): nothing to await.
        del pending

    def held(self, ref):
        return ref

    def zeros(self, shape: tuple[int, ...]) -> jax.Array:
        return plgpu.layout_cast(jnp.zeros(shape, jnp.float32), STORED)

    def added_into(self, acc, value, interpret: bool):
        raise NotImplementedError(f"{self._HOME}; nothing adds a register tile into one")

    def product(self, lhs, rhs, shape: tuple[int, int], then=None, *, interpret: bool = False):
        raise NotImplementedError(f"{self._HOME}; a fresh product takes the launch's, not its own")

    def registers_product(self, lhs, rhs, interpret: bool):
        raise NotImplementedError(f"{self._HOME}; its operands are read from shared or tensor memory")

    def register_products(self, acc, parts, rhs, interpret: bool):
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
    (JAX's `blackwell_ragged_dot_mgpu.py`'s schedule, CUTLASS's sm100 persistent GEMM)."""

    consumers = 1
    gathers_rows = False  # its rings and roles are its own (`streams`, `persist`), a load warp's, no gather

    turns = 2
    """The accumulator turns a launch's tensor memory holds: an epilogue reads one while the products fill
    the other."""

    def tensor_columns(self, columns: int, accumulators: int) -> int:
        # Each turn holds every accumulator of a tile (`persist`).
        return self.turns * columns * accumulators

    def streams(self, blocks: dict, stages: int, consumers: int, multicast: tuple[str, ...] = (),
                swizzles: dict[str, int] | None = None, copies: dict[str, int] | None = None,
                gathered: tuple[str, ...] = ()) -> dict:
        if gathered:
            raise NotImplementedError("tcgen05's load warp gathers no rows (`gathers_rows`)")
        if multicast:
            raise NotImplementedError("tcgen05's pair of CTAs shares a product, not a multicast ring "
                                      "(`Topology`; not yet a stream of this bundle's)")
        streams = {name: mosaic.ring(shape, dtype, stages, swizzle=(swizzles or {}).get(name),
                                     copies=(copies or {}).get(name, 1), released_by_products=True)
                   for name, (shape, dtype) in blocks.items()}
        # A block-scale stream's slots each pair with the tensor memory its products read them from: its
        # block's rows (`[rows / 128, depth / 128, 32, 16]`) by its depth's scales, a slot's columns each.
        for name, (shape, dtype) in blocks.items():
            if jnp.dtype(dtype) == SCALES:
                streams[name]['tmem'] = plgpu.TMEM((shape[0] * 128, stages * shape[1] * 4), SCALES,
                                                   layout=plgpu.TMEMLayout.SCALES_LAYOUT)
        return streams

    def opened(self, allocated: dict, views: dict | None = None) -> dict:
        del views
        return {name: ScaleRing(**scoped) if 'tmem' in scoped else CommittedRing(**scoped)
                for name, scoped in allocated.items()}

    def roles_of(self, table: tuple, *, load: Callable[[], None], products: Callable[[], None]) -> None:
        """Each of a program's warpgroups its role (`mosaic.tabled`): the template's `table`, `(count,
        body)` in turn, `count` warpgroups each running `body(index)`, its index among them; then the
        issuing warpgroup, its load warp running `load()` and its product warp `products()` (`kernel`'s
        warpgroups: the table's and one)."""
        mosaic.tabled(lax.axis_index("wg"), table, lambda: _issuing(load, products))

    def persist_scratch(self, consumers: int) -> tuple[tuple, int]:
        # Its accumulator turns' barriers, filled and drained (`persist`), and its tensor memory's address.
        return ((plgpu.Barrier(num_barriers=self.turns, orders_tensor_core=True),
                 plgpu.Barrier(num_barriers=self.turns, num_arrivals=consumers, orders_tensor_core=True)),
                resources.TMEM_ADDRESS_BYTES)

    def persist(self, streams, tiles: tuple[int, ...], schedule: Callable, issue: Callable, finish: Callable,
                *, consumers: int, axis: str, outputs: tuple, interpret: bool, in_flight: int = 0,
                unrolled: bool = False, accumulators: tuple = ()) -> None:
        if not accumulators:
            raise ValueError(f"{self._HOME}: a persistent launch states their shapes (`accumulators`)")
        rows = {shape[0] for shape in accumulators}
        if rows != {self.rows}:
            raise ValueError(f"a tcgen05 accumulator is {self.rows} rows on one CTA, not {sorted(rows)}")
        del in_flight  # the products' commit releases each step's slots
        rings = self.opened(streams)
        staging = self.output_staging(outputs, consumers)
        width = sum(shape[1] for shape in accumulators)
        offsets = [sum(shape[1] for shape in accumulators[:index]) for index in range(len(accumulators))]

        def scoped(tmem, done, read, *smems):
            def turn_of(done_tiles) -> mosaic.Turn:
                return mosaic.Turn.of(done_tiles, self.turns)

            def held(turn: mosaic.Turn) -> tuple:
                # The turn's accumulators: each its columns of the turn's share of tensor memory.
                return tuple(Accumulator(columns(tmem, turn.slot * width + offset, shape[1]), shape)
                             for offset, shape in zip(offsets, accumulators, strict=True))

            def products(counter, active, steps, views, tile, done_tiles):
                # The product warp: each active tile's steps into its turn's accumulators once the tile
                # `turns` back has been read out of them, then a commit the epilogue waits on.
                @pl.when(active)
                def _():
                    turn = turn_of(done_tiles)
                    pl.when(turn.counter >= self.turns)(lambda: plgpu.barrier_wait(
                        read.at[turn.slot], parity=turn.freed))
                    into = held(turn)

                    def walk(step, start, shapes, *, initial=None):
                        if initial is not None:
                            raise NotImplementedError(f"{self._HOME}; a walk starts it from no value")

                        def stepped(at, slots, carry, accs):
                            return step(at - counter, slots, carry, accs)

                        return self.walk(rings, ((stepped, counter, counter + steps),), start, shapes,
                                         interpret, initial=into, unrolled=unrolled)

                    issue(0, tile, walk, active=True)
                    plgpu.tcgen05_commit_arrive(done.at[turn.slot])

            def epilogue(index):
                store = self._stores(smems, index)

                def visit(counter, active, steps, views, tile, done_tiles):
                    del counter, steps, views

                    @pl.when(active)
                    def _():
                        turn = turn_of(done_tiles)
                        plgpu.barrier_wait(done.at[turn.slot], parity=turn.parity)
                        into = held(turn)

                        def walk(step, start, shapes, *, initial=None):
                            # The products issued it: the tile's accumulators, done.
                            del step, initial
                            if tuple(shapes) != accumulators:
                                raise ValueError(f"a tile walks {shapes}, its launch stated {accumulators}")
                            return start, into

                        finish(index, tile, issue(index, tile, walk, active=True), store, active=True)
                        plgpu.wait_load_tmem()
                        plgpu.barrier_arrive(read.at[turn.slot])

                    pl.when(~active)(lambda: finish(index, tile, issue(index, tile, mosaic.unwalked(self),
                                                                       active=False), store, active=False))

                mosaic.tiled(tiles, schedule, visit, axis=axis)
                plgpu.wait_smem_to_gmem(0, wait_read_only=True)

            def load():
                loaded, _ = mosaic.tiled(tiles, schedule, functools.partial(mosaic.loads, rings), axis=axis)
                for ring in rings.values():
                    ring.drain(loaded)

            def issue_all():
                _, computed = mosaic.tiled(tiles, schedule, products, axis=axis)
                mosaic.drained(read, computed, self.turns)

            self.roles_of(((consumers, epilogue),), load=load, products=issue_all)

        pl.run_scoped(scoped, plgpu.TMEM((self.rows, self.turns * width), jnp.float32),
                      *self.persist_scratch(consumers)[0], *staging, collective_axes="wg")


RINGS = Tcgen05Rings()
"""The tcgen05 bundle (`bundles.of` on sm100 and sm103)."""


def _issuing(load: Callable[[], None], products: Callable[[], None]) -> None:
    """The issuing warpgroup's warps: its load warp runs `load`, its product warp `products`."""
    @plgpu.warp_map
    def _(warp):
        pl.when(warp == LOAD_WARP)(load)
        pl.when(warp == MMA_WARP)(products)


def _tensor_memory(acc: Accumulator) -> Ref:
    """A tile's tensor memory; a tile that multiplies nothing has none to issue into or write."""
    if acc.ref is None:
        raise ValueError("a tile that multiplies nothing holds no tensor memory")
    return acc.ref


def _columns(ref, first, columns: int):
    """Columns `first` to `first + columns` of a tile in tensor memory."""
    return ref.at[:, pl.ds(first, columns)]
