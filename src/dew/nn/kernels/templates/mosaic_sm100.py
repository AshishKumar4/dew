"""Blackwell's bundle (Mosaic GPU, sm100 and sm103): tcgen05's products, issued by one thread into
accumulators in tensor memory, and the warpgroups a mainloop runs them in, which the GEMM template's
(`gemm.mosaic_blackwell`) and the attention template's Blackwell mainloops share.

A program runs two warpgroups (`roles`). The first issues: one warp's TMA loads fill a ring of operand
stages (`Stages`), one warp issues each step's products once its stage is there, the products releasing
the stage when they have read it (a tensor-core-ordered barrier, `tcgen05_mma`'s). The second reads
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
               interpret, name: str | None = None, topology: Topology = SINGLE) -> Callable:
        """`entry` as a program of `WARPGROUPS` warpgroups (axis 'wg'), its kernel named `name`, `grid` the
        clusters of `topology`'s CTAs (one each without a pair)."""
        return mosaic.kernel(entry, out_type=out_type, grid=grid, grid_names=grid_names,
                             warpgroups=WARPGROUPS, family=Tcgen05.name, scratch=scratch,
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
        @pl.when(lax.axis_index("wg") == ISSUE)
        def _():
            @plgpu.warp_map
            def _(warp):
                pl.when(warp == LOAD_WARP)(load)
                pl.when(warp == MMA_WARP)(products)


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
class Stages:
    """A ring of `stages` operand stages in shared memory, one buffer an operand (`blocks`, each its
    block's shape, dtype and transforms), a load barrier an operand and stage, and a release barrier a
    stage the products arrive at once they have read it (tensor-core ordered). A turn is a step's index
    over the program's walk; its stage is the turn's remainder. Over a pair of CTAs (`topology`), each
    operand's block splits along its `partitions` dimension between them."""
    stages: int
    blocks: tuple
    topology: Topology = SINGLE
    partitions: tuple[int, ...] = ()

    def scratch(self) -> tuple:
        """The ring's shared memory and barriers, in `opened`'s order."""
        buffers = tuple(plgpu.SMEM((self.stages, *shape), dtype, transforms=transforms)
                        for shape, dtype, transforms in self.blocks)
        loaded = tuple(plgpu.Barrier(num_barriers=self.stages) for _ in self.blocks)
        return (buffers, loaded, plgpu.Barrier(num_barriers=self.stages, orders_tensor_core=True))

    def opened(self, buffers, loaded, consumed) -> "Ring":
        return Ring(self.stages, buffers, loaded, consumed, self.topology, self.partitions)


@dataclasses.dataclass(frozen=True)
class Ring:
    """An opened ring (`Stages.opened`)."""
    stages: int
    buffers: tuple[Ref, ...]
    loaded: tuple[state_types.AbstractRef, ...]
    consumed: state_types.AbstractRef
    topology: Topology = SINGLE
    partitions: tuple[int, ...] = ()

    def turn(self, counter) -> mosaic.Turn:
        """Step `counter`'s turn of the ring."""
        return mosaic.Turn.of(counter, self.stages)

    def load(self, turn: mosaic.Turn, views: tuple) -> None:
        """The load warp's turn: each operand's `views` entry into the turn's stage, once the products
        have read the stage's last ones (the rotation before, of the other parity)."""
        pl.when(turn.counter >= self.stages)(lambda: plgpu.barrier_wait(
            self.consumed.at[turn.slot], parity=turn.freed))
        partitions = self.partitions or (0,) * len(views)
        for view, buffer, barrier, along in zip(views, self.buffers, self.loaded, partitions, strict=True):
            plgpu.copy_gmem_to_smem(view, buffer.at[turn.slot], barrier.at[turn.slot],
                                    **self.topology.copied(along))

    def acquire(self, turn: mosaic.Turn) -> tuple:
        """The product warp's turn: each operand's stage once loaded, and the barrier its products
        release it at."""
        for barrier in self.loaded:
            plgpu.barrier_wait(barrier.at[turn.slot], parity=turn.parity)
        return tuple(buffer.at[turn.slot] for buffer in self.buffers), self.consumed.at[turn.slot]

    def drain(self, tiles, steps: int | jax.Array) -> None:
        """The load warp's last releases, which no later load waited for (`mosaic.drained`), after `tiles`
        tiles of `steps` turns each."""
        mosaic.drained(self.consumed, tiles * steps, self.stages)


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
