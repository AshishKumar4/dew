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
(`resources.PRODUCER_REGISTERS`)."""

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
from dew.nn.kernels.templates import resources
from dew.nn.kernels.templates.mosaic import SWIZZLE, TILING, Ref

ISSUE, READ = 0, 1
"""A program's warpgroups: the one issuing loads and products, and the one reading accumulators."""
LOAD_WARP, MMA_WARP = 0, 1
"""The issuing warpgroup's warps: the TMA loads' and the products'."""
TMEM = plgpu.Layout.TCGEN05_TMEM_NATIVE
"""A tile's register layout as tensor memory loads it natively (a thread a row)."""
ROW = TMEM.reduce(1)
"""A per-row vector's layout in `TMEM`."""
STORED = plgpu.Layout.TCGEN05
"""An accumulator's register layout for an epilogue, the one a swizzled shared-memory tile takes."""
WARPGROUPS = 2
"""A program's warpgroups (`roles`)."""


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
               interpret, name: str | None = None) -> Callable:
        """`entry` as a program of `WARPGROUPS` warpgroups (axis 'wg'), its kernel named `name`."""
        return plgpu.kernel(
            entry, out_type=out_type, grid=grid, grid_names=grid_names, num_threads=WARPGROUPS,
            thread_name="wg", scratch_types=scratch,
            compiler_params=plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Warpgroup),
            interpret=interpret, kernel_name=name)

    @staticmethod
    def roles(load: Callable[[], None], products: Callable[[], None], read: Callable[[], None]) -> None:
        """The issuing warpgroup's load warp runs `load`, its product warp `products`; the other warpgroup
        runs `read`."""
        wg = lax.axis_index("wg")

        @pl.when(wg == ISSUE)
        def _():
            @plgpu.warp_map
            def _(warp):
                pl.when(warp == LOAD_WARP)(load)
                pl.when(warp == MMA_WARP)(products)

        pl.when(wg == READ)(read)

    @staticmethod
    def issued(load: Callable[[], None], products: Callable[[], None]) -> None:
        """The issuing warpgroup's warps alone (a program's end, `drain`'s): its load warp runs `load`, its
        product warp `products`."""
        @pl.when(lax.axis_index("wg") == ISSUE)
        def _():
            @plgpu.warp_map
            def _(warp):
                pl.when(warp == LOAD_WARP)(load)
                pl.when(warp == MMA_WARP)(products)


TCGEN05 = Tcgen05()
"""The bundle (`bundles.tensor_memory`)."""


def swizzled(dtype, contraction: int) -> tuple:
    """A stage's transforms for an operand block of `dtype` contracting `contraction` elements a step: 8-row
    tiles of the widest swizzle its rows take, swizzled (tcgen05 reads its shared operands so)."""
    swizzle = plgpu.find_swizzle(contraction * jnp.dtype(dtype).itemsize * 8)
    return TILING((8, swizzle // jnp.dtype(dtype).itemsize)), SWIZZLE(swizzle)


@dataclasses.dataclass(frozen=True)
class Stages:
    """A ring of `stages` operand stages in shared memory, one buffer an operand (`blocks`, each its
    block's shape, dtype and transforms), a load barrier an operand and stage, and a release barrier a
    stage the products arrive at once they have read it (tensor-core ordered). A turn is a step's index
    over the program's walk; its stage is the turn's remainder."""
    stages: int
    blocks: tuple

    def scratch(self) -> tuple:
        """The ring's shared memory and barriers, in `opened`'s order."""
        buffers = tuple(plgpu.SMEM((self.stages, *shape), dtype, transforms=transforms)
                        for shape, dtype, transforms in self.blocks)
        loaded = tuple(plgpu.Barrier(num_barriers=self.stages) for _ in self.blocks)
        return (buffers, loaded, plgpu.Barrier(num_barriers=self.stages, orders_tensor_core=True))

    def opened(self, buffers, loaded, consumed) -> "Ring":
        return Ring(self.stages, buffers, loaded, consumed)


@dataclasses.dataclass(frozen=True)
class Ring:
    """An opened ring (`Stages.opened`)."""
    stages: int
    buffers: tuple[Ref, ...]
    loaded: tuple[state_types.AbstractRef, ...]
    consumed: state_types.AbstractRef

    def slot(self, turn):
        return lax.rem(turn, self.stages)

    def load(self, turn, views: tuple) -> None:
        """The load warp's turn: each operand's `views` entry into the turn's stage, once the products
        have read the stage's last ones."""
        slot = self.slot(turn)
        pl.when(turn >= self.stages)(functools.partial(plgpu.barrier_wait, self.consumed.at[slot]))
        for view, buffer, barrier in zip(views, self.buffers, self.loaded, strict=True):
            plgpu.copy_gmem_to_smem(view, buffer.at[slot], barrier.at[slot])

    def acquire(self, turn) -> tuple:
        """The product warp's turn: each operand's stage once loaded, and the barrier its products
        release it at."""
        slot = self.slot(turn)
        for barrier in self.loaded:
            plgpu.barrier_wait(barrier.at[slot])
        return tuple(buffer.at[slot] for buffer in self.buffers), self.consumed.at[slot]

    def drain(self, tiles, steps: int) -> None:
        """The load warp's last releases, which no later load waited for (`drain`), after `tiles` tiles of
        `steps` turns each."""
        drain(self.consumed, tiles * steps, self.stages)


@dataclasses.dataclass(frozen=True)
class Accumulators:
    """`count` fp32 accumulators of `rows` x `columns` in tensor memory side by side, a product barrier
    each the product warp commits to once a tile's products are issued and a read barrier each the reading
    warpgroup arrives at once it has read one out (both tensor-core ordered). A turn is a tile's index
    over the program's tiles that compute; its accumulator is the turn's remainder."""
    count: int
    rows: int
    columns: int

    def scratch(self, target: generation.GpuTarget) -> tuple:
        if self.count * self.columns > TCGEN05.columns(target):
            raise ValueError(f"{self.count} accumulators of {self.columns} columns exceed tensor memory")
        tensor_core = functools.partial(plgpu.Barrier, orders_tensor_core=True)
        return (plgpu.TMEM((self.rows, self.count * self.columns), jnp.float32),
                tensor_core(num_barriers=self.count), tensor_core(num_barriers=self.count))

    def opened(self, tmem, done, read) -> "Tensor":
        return Tensor(self, tmem, done, read)


@dataclasses.dataclass(frozen=True)
class Tensor:
    """Opened accumulators (`Accumulators.opened`)."""
    shape: Accumulators
    tmem: Ref
    done: state_types.AbstractRef
    read_done: state_types.AbstractRef

    def index(self, turn):
        return lax.rem(turn, self.shape.count)

    def acquire(self, turn):
        """The product warp's turn: the turn's accumulator once the turn `count` before has been read
        out of it."""
        index = self.index(turn)
        pl.when(turn >= self.shape.count)(functools.partial(plgpu.barrier_wait, self.read_done.at[index]))
        return columns(self.tmem, index * self.shape.columns, self.shape.columns)

    def issued(self, turn) -> None:
        """The product warp's turn's products all issued: the reader may wait on them."""
        plgpu.tcgen05_commit_arrive(self.done.at[self.index(turn)])

    def read(self, turn, chunk: int, layout=STORED) -> Callable[[int], jax.Array]:
        """The reading warpgroup's turn: once its products are done, each `chunk`-column slice of its
        accumulator by its index, loaded from tensor memory in `layout`."""
        index = self.index(turn)
        plgpu.barrier_wait(self.done.at[index])
        first = index * self.shape.columns
        return lambda part: plgpu.async_load_tmem(columns(self.tmem, first + part * chunk, chunk),
                                                  layout=layout)

    def release(self, turn) -> None:
        """The reading warpgroup's turn read out: its accumulator free for the turn `count` after."""
        plgpu.wait_load_tmem()
        plgpu.barrier_arrive(self.read_done.at[self.index(turn)])

    def drain(self, turns) -> None:
        """The product warp's last reads, which no later product waited for (`drain`)."""
        drain(self.read_done, turns, self.shape.count)


def columns(tmem, first, count: int):
    """Columns `first` to `first + count` of tensor memory `tmem`, every row."""
    return tmem.at[:, pl.ds(first, count)]


def drain(barriers: state_types.AbstractRef, turns, slots: int) -> None:
    """The completions of the last `slots` of `turns` turns taken round `barriers`, one a slot, which no
    later turn of the slot waited for: each barrier ends at the phase its waiters saw."""
    for back in range(1, slots + 1):
        turn = turns - back
        slot = lax.rem(jnp.maximum(turn, 0), slots)
        pl.when(turn >= 0)(functools.partial(plgpu.barrier_wait, barriers.at[slot]))
