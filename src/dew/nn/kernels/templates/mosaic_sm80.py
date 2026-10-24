"""The mma.sync instruction family's bundle (Ampere and its consumer kin, sm80 to sm89 and consumer
Blackwell; `generation.family` 'mma.sync') for the templates' Mosaic mainloops (`mosaic.Family`): one
warpgroup a program, its operand tiles streamed into shared memory by `plgpu.emit_pipeline` (cp.async),
read into registers in the MMA layouts (ldmatrix), and multiplied synchronously (mma.sync) into an
accumulator the walk carries in registers. An accumulator's value is the next product's left operand in
registers (the fork's relayout, jax 0.11.2.post13 and later), as FlashAttention-2 reuses its fragments.
Its layouts are its operands' dtype's (`Layout.MMA_ACC(dtype)`), so a bundle is a dtype's (`Mma`)."""

import dataclasses
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import mosaic

_TILING: Callable[[tuple[int, int]], object] = plgpu.TilingTransform
_SWIZZLE: Callable[[int], object] = plgpu.SwizzleTransform
"""Mosaic GPU's tile transforms, which it declares as protocol classes (a type checker takes them for
uninstantiable)."""


def swizzled(shape: tuple[int, int], dtype) -> tuple:
    """A `[rows, columns]` shared-memory tile's transforms: 8-row tiles as wide as the swizzle its row
    bytes allow, swizzled (bank-conflict-free ldmatrix), as jax's Ampere examples stage their operands."""
    swizzle = plgpu.find_swizzle(shape[1] * jnp.dtype(dtype).itemsize * 8)
    return _TILING((8, swizzle // jnp.dtype(dtype).itemsize)), _SWIZZLE(swizzle)


class Slots:
    """A pipelined step's operand slots by stream name (`mosaic.Slots`'s interface): `emit_pipeline` hands
    the step its shared-memory tiles and takes them back when the step returns, so releasing one early is
    nothing."""

    def __init__(self, refs: dict):
        self._refs = refs

    def __getitem__(self, name: str):
        return self._refs[name]

    def release(self, name: str) -> None:
        del name


@dataclasses.dataclass(frozen=True)
class Pipeline:
    """A program's operand streams as `Mma.streams` describes them: each stream's tile `(shape, dtype)` by
    name and the tiles `emit_pipeline` keeps in flight (`stages`); it allocates their slots itself."""

    blocks: dict
    stages: int


@dataclasses.dataclass(frozen=True)
class Streams:
    """A program's operand streams (`Mma.opened`): its pipeline and each stream's view `(ref, block)` by
    name, the `[S, D]` operand its steps' tiles come from in `block` rows."""

    pipeline: Pipeline
    views: dict


@dataclasses.dataclass(frozen=True)
class Mma(mosaic.Family):
    """mma.sync's bundle for operands of `dtype`: one compute warpgroup a program and no producer
    (`emit_pipeline` issues its loads), accumulators in registers, products synchronous."""

    dtype: jnp.dtype

    name = "mma.sync"
    rows = 16
    consumers = 1

    @property
    def layout(self):
        return plgpu.Layout.MMA_ACC(self.dtype)

    @property
    def row(self):
        return self.layout.reduce(1)

    @property
    def column(self):
        return self.layout.reduce(0)

    @property
    def lhs(self):
        return plgpu.Layout.MMA_LHS(self.dtype)

    @property
    def rhs(self):
        return plgpu.Layout.MMA_RHS(self.dtype)

    def kernel(self, entry, *, out_type, grid: tuple, grid_names: tuple, consumers: int,
               interpret: bool) -> Callable:
        _one(consumers)
        return plgpu.kernel(entry, out_type=out_type, grid=grid, grid_names=grid_names,
                            compiler_params=plgpu.CompilerParams(approx_math=True),
                            interpret=plgpu.InterpretGPUParams() if interpret else None)

    def scoped(self, body, *allocations, streams) -> None:
        pl.run_scoped(lambda *refs: body(*refs, streams=streams), *allocations)

    def roles(self, consumers: int, consumer, producer) -> None:
        _one(consumers)
        del producer
        consumer(0)

    def streams(self, blocks: dict, stages: int, consumers: int):
        _one(consumers)
        return Pipeline(blocks, stages)

    def opened(self, allocated, views: dict | None = None):
        if views is None:
            raise ValueError("a pipelined family's streams take their views where they open (its walk loads)")
        return Streams(allocated, views)

    def stream(self, streams, views, lo, hi) -> None:
        del streams, views, lo, hi

    def walk[C](self, streams, segments, start: C, shapes: tuple, interpret: bool, *,
                first: jax.Array | int = 0, in_flight: int = 0) -> tuple[C, tuple[jax.Array, ...]]:
        # Its products are synchronous: none stays in flight past its step, and `emit_pipeline` gives
        # each stage back as its step ends.
        del first, in_flight
        blocks = streams.pipeline.blocks
        names = tuple(blocks)
        carry = (start, tuple(self.zeros(shape) for shape in shapes))
        for step, lo, hi in segments:
            def body(indices, *refs, step=step, lo=lo):
                (index,), (carried, accumulators) = indices, refs[-1]
                slots = Slots(dict(zip(names, refs[:-1], strict=True)))
                return step(index + lo, slots, carried, accumulators)

            specs = [plgpu.BlockSpec(blocks[name][0],
                                     lambda j, lo=lo, index_map=streams.views[name][1]: index_map(j + lo),
                                     transforms=swizzled(*blocks[name]))
                     for name in names]
            carry = plgpu.emit_pipeline(body, grid=(hi - lo,), in_specs=specs,
                                        max_concurrent_steps=streams.pipeline.stages,
                                        init_carry=carry)(*(streams.views[name][0] for name in names))
        return carry

    def passed(self, streams, lo, hi, *, first: jax.Array | int = 0, between=None) -> None:
        del streams, lo, hi, first, between

    def loaded(self, copies) -> None:
        for view, ref, _ in copies:
            ref[...] = plgpu.load(view, layout=self.lhs, optimized=False)

    def added(self, value: jax.Array, view, interpret: bool, *, smem) -> None:
        del smem
        if not interpret:
            # No TMA reduction before Hopper, and no atomic add through registers in Mosaic GPU's
            # lowering (jax 0.11.2).
            raise NotImplementedError("mma.sync's bundle adds into global memory only interpreted")
        view[...] = view[...] + value

    def held(self, ref):
        return plgpu.load(ref, layout=self.lhs, optimized=False)

    def _left(self, lhs):
        """A product's left operand in the MMA's layout: a shared-memory tile read, or a register value's
        (an accumulator's, the fork's relayout) cast."""
        if isinstance(lhs, jax.Array):
            return plgpu.layout_cast(lhs.astype(self.dtype), self.lhs)
        return plgpu.load(lhs, layout=self.lhs)

    def product(self, lhs, rhs, shape: tuple[int, int], then=None) -> jax.Array:
        out = plgpu.mma(self.zeros(shape), self._left(lhs), plgpu.load(rhs, layout=self.rhs))
        if then is not None:
            then()
        return out

    def products(self, acc, lhs, rhs, interpret: bool):
        if interpret:
            # mma has no rule on a CPU: its sum, in fp32.
            return acc + jnp.dot(lhs[...], rhs[...], preferred_element_type=jnp.float32)
        return plgpu.mma(acc, self._left(lhs), plgpu.load(rhs, layout=self.rhs))

    def registers_product(self, lhs: jax.Array, rhs, interpret: bool) -> jax.Array:
        del interpret
        return self.product(lhs, rhs, (lhs.shape[0], rhs.shape[1]))

    def register_products(self, acc, parts, rhs, interpret: bool):
        del interpret
        mosaic.unscaled(parts)
        operand = plgpu.load(rhs, layout=self.rhs)
        for part, _ in parts:
            acc = plgpu.mma(acc, self._left(part), operand)
        return acc

    def staged_products(self, acc, parts, rhs, buffers, interpret: bool, *, transposed: bool = False):
        del interpret
        mosaic.unscaled(parts)
        operand = plgpu.load(rhs, layout=self.rhs)
        for place, (part, _) in enumerate(parts):
            buffers.at[place][...] = part.astype(buffers.dtype)
            staged = buffers.at[place].transpose((1, 0)) if transposed else buffers.at[place]
            acc = plgpu.mma(acc, plgpu.load(staged, layout=self.lhs), operand)
        return acc

    def rescaled_products(self, acc, scale, parts, rhs, interpret: bool):
        return self.register_products(scale(acc), parts, rhs, interpret)

    def settled(self, pending: int = 0) -> None:
        del pending


def _one(consumers: int) -> None:
    """mma.sync's programs take one compute warpgroup (`emit_pipeline`'s stages are no thread-collective
    allocation)."""
    if consumers != 1:
        raise NotImplementedError(f"an mma.sync program takes one consumer warpgroup, not {consumers}")

