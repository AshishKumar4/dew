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
from jax._src.state import types as state_types
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import cost, mosaic, resources
from dew.nn.sharding import per_shard

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
    (`emit_pipeline` issues its loads), accumulators in registers, products synchronous, its warps
    `m_warps` along a tile's rows and the rest along its columns (JAX's `MMALayouts`; `for_tile`)."""

    dtype: jnp.dtype
    m_warps: int = 4

    name = "mma.sync"
    rows = 16
    # One consumer warpgroup a program; a multiprocessor runs two programs (`programs`). Two consumers in
    # one program took 1.04 to 1.50 of one's time on an A100 (CONSUMERS1), their shared copies leaner or
    # not, where two programs took 0.73 to 0.89 (CONSUMERS3).
    consumers = 1
    # Its kernels take hooks and exact sums (their side loads, vector reads and running sum lower), but
    # its register model does not count what they hold, and its derived tiles spilled (ns-regs2: sm86's
    # normed forward 16 bytes, exact sums 136).
    counts_hooks = False
    walks_scoped = True  # `emit_pipeline` allocates each walk's slots in its own scope
    bounded_loads = False
    role_rows = 4 * 16
    operands_in_registers = True
    stores_through_registers = True

    @classmethod
    def for_tile(cls, dtype, rows: int, columns: int) -> "Mma":
        """The bundle for a warpgroup's `rows` x `columns` of a tile, its warps split `m_warps` along the
        rows and the rest along the columns (`MMALayouts`; each product states it, `plgpu.mma`'s
        `m_warps`): of the splits whose atom, `m_warps * 16` rows by `(4 // m_warps) * 8` columns, divides
        it, the one reading its operands from shared memory fewest times a step, each warp reading its
        rows' share whole and its columns' share whole (`rows * (4 // m) + columns * m`), the most along
        the rows on a tie. A 128 x 128 tile so splits 2 x 2 (512 reads a step, where four warps along its
        rows read 640), each warp holding half the right operand's fragments."""
        splits = [m for m in (4, 2, 1) if rows % (m * 16) == 0 and columns % ((4 // m) * 8) == 0]
        m_warps = min(splits, key=lambda m: (rows * (4 // m) + columns * m, -m)) if splits else 4
        return cls(jnp.dtype(dtype), m_warps)

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
    most_shared = 163 * 1024  # sm80's and sm87's opt-in block (sm86's and sm89's 99 KiB)
    copies_vectors = False  # cp.async spreads a copy over the warpgroup's threads, 128 elements at least
    reduction_scratch = plgpu.CompilerParams().reduction_scratch_bytes  # its kernels' default parameters
    column_scratch = 0  # its column sums take its kernels' reduction scratch

    def registers(self, consumers: int) -> int:
        # Each role at a thread's whole budget: two warpgroups' 256 threads hold 255 each of a
        # multiprocessor's 64K registers.
        _one(consumers)
        return 255

    def programs(self, consumers: int) -> int:
        # As many as the register file holds at its threads' budget: one warpgroup's 128 threads at 255
        # registers hold half of it, so two programs run on a multiprocessor, eight warps as cuBLAS's and
        # CUTLASS's Ampere kernels run, each in half its shared memory: an A100's 81 KiB, a 99 KiB block's
        # GPU's 49 (sm86, sm89), where a call whose tile it cannot hold (the compile gate's three forms)
        # keeps one program (`mosaic_gpu.derived`). On an A100 two took 0.73 to 0.93 of one program's
        # time, a product's geomean over 6 grouped and 4 dense rows, their outputs bitwise one program's
        # (CONSUMERS2); two consumers in one program took 1.04 to 1.50 (CONSUMERS1).
        return cost.REGISTERS // (consumers * resources.THREADS * self.registers(consumers))

    def column_unit(self, dtype) -> int:
        # Its stores move a row through registers, laid over the role's threads, one element or more each.
        del dtype
        return resources.THREADS

    def depths(self, dtype) -> tuple[int, ...]:
        # Its operand blocks' rows staged at a 128- or 64-byte swizzle (`swizzled`).
        return tuple(swizzle // jnp.dtype(dtype).itemsize for swizzle in (mosaic.SWIZZLE_BYTES, 64))

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

    @property
    def rhs(self):
        return plgpu.Layout.MMA_RHS(self.dtype, m_warps=self.m_warps)

    def kernel(self, entry, *, out_type, grid: tuple, grid_names: tuple, consumers: int,
               interpret: bool, approx_math: bool = True, name: str | None = None,
               cluster: int = 1, cost: pl.CostEstimate | None = None) -> Callable:
        # Its one warpgroup is its consumer, no producer beside it, and no cluster.
        _one(consumers, clustered=cluster > 1)
        per_shard(name or self.name)
        return plgpu.kernel(entry, out_type=out_type, grid=grid, grid_names=grid_names,
                            compiler_params=plgpu.CompilerParams(approx_math=approx_math),
                            interpret=mosaic.interpreted(interpret), kernel_name=name, cost_estimate=cost)

    def scoped(self, body, *allocations, streams) -> None:
        pl.run_scoped(lambda *refs: body(*refs, streams=streams), *allocations)

    def roles(self, consumers: int, consumer, producer) -> None:
        _one(consumers)
        del producer
        consumer(0)

    def streams(self, blocks: dict, stages: int, consumers: int, multicast: tuple[str, ...] = ()):
        _one(consumers, clustered=bool(multicast))
        return Pipeline(blocks, stages)

    def opened(self, allocated, views: dict | None = None):
        if views is None:
            raise ValueError("a pipelined family's streams take their views where they open (its walk loads)")
        return Streams(allocated, views)

    def stream(self, streams, views, lo, hi) -> None:
        del streams, views, lo, hi

    def persist(self, streams, tiles: tuple[int, ...], schedule: Callable, consumer: Callable, *,
                consumers: int, axis: str, outputs: tuple, interpret: bool, in_flight: int = 0) -> None:
        # Each tile's pipeline is its own (`emit_pipeline` fills and drains it), its step count static; an
        # output is staged after it in a buffer scoped beside the pipeline (Mosaic reserves both at once:
        # `grouped_matmul._Mosaic.staged_bytes`), and stored through registers (no TMA before Hopper).
        _one(consumers)
        del in_flight

        def tiled(*staging):
            def store(output, value, ref, place):
                smem = staging[output]
                # Unoptimized, as `copied`'s transfers: JAX 0.12 takes an unflagged store as optimized-only,
                # which its layout inference searched to exhaustion beside the ladder's copies (ns-gate3).
                plgpu.store(smem, value.astype(smem.dtype), optimized=False)
                mosaic.store_rows(smem, ref, *place, copy=mosaic.copied,
                                  widest=resources.copied_rows(smem.shape[1], smem.dtype))

            @plgpu.nd_loop(tiles, collective_axes=axis)
            def _(loop: plgpu.NDLoopInfo):
                active, steps, views, tile = schedule(loop.index)

                def walk(step, start, shapes, *, initial=None):
                    return self.walk(self.opened(streams, views), ((step, 0, steps),), start, shapes,
                                     interpret, initial=initial)

                pl.when(active)(lambda: consumer(0, tile, walk, store, active=True))
                pl.when(~active)(lambda: consumer(0, tile, mosaic.unwalked(self), store, active=False))

        pl.run_scoped(tiled, *(plgpu.SMEM(shape, dtype) for shape, dtype in outputs))

    def walk[C](self, streams, segments, start: C, shapes: tuple, interpret: bool, *,
                first: jax.Array | int = 0, in_flight: int = 0, initial: tuple | None = None
                ) -> tuple[C, tuple[jax.Array, ...]]:
        # Its products are synchronous: none stays in flight past its step, and `emit_pipeline` gives
        # each stage back as its step ends.
        del first, in_flight
        blocks = streams.pipeline.blocks
        names = tuple(blocks)
        initial = (None,) * len(shapes) if initial is None else initial
        carry = (start, tuple(self.zeros(shape) if value is None else value
                              for shape, value in zip(shapes, initial, strict=True)))
        for step, lo, hi in segments:
            if isinstance(lo, int) and isinstance(hi, int) and hi <= lo:
                # No steps: the carry and its zero accumulators as they start (`emit_pipeline` takes no
                # empty grid).
                continue

            def body(indices, *refs, step=step, lo=lo):
                (index,), (carried, accumulators) = indices, refs[-1]
                slots = Slots(dict(zip(names, refs[:-1], strict=True)))
                return step(index + lo, slots, carried, accumulators)

            # A tile's blocks staged swizzled for ldmatrix; a side vector's untiled, in whole 16-byte
            # cp.async chunks.
            specs = [plgpu.BlockSpec(blocks[name][0],
                                     lambda j, lo=lo, index_map=streams.views[name][1]: index_map(j + lo),
                                     transforms=swizzled(*blocks[name]) if len(blocks[name][0]) > 1 else ())
                     for name in names]
            pipeline = plgpu.emit_pipeline(body, grid=(hi - lo,), in_specs=specs,
                                           max_concurrent_steps=streams.pipeline.stages, init_carry=carry)
            carry = pipeline(*(streams.views[name][0] for name in names))
        return carry

    def passed(self, streams, lo, hi, *, first: jax.Array | int = 0, between=None) -> None:
        del streams, lo, hi, first, between

    def loaded(self, copies) -> None:
        # Through registers (no TMA before Hopper), in a layout of the tile's rank: a side vector's the
        # accumulator's column vector layout, a tile's the left operand's.
        for view, ref, _ in copies:
            layout = self.lhs if len(ref.shape) > 1 else self.column
            ref[...] = plgpu.load(view, layout=layout, optimized=False)

    def stored(self, parts, views, *, smem, wait: bool = True) -> None:
        # Through `smem` where given, copied out in row blocks as `persist`'s stores are (`store_rows`): an
        # MMA-layout tile stored from registers writes partial 32-byte sectors, which JAX 0.12's global
        # transfers refuse, and its row-sliced copies lay out whole ones. Its copies go through registers,
        # done when they return: nothing stays in flight (`wait` asks nothing of it).
        # A vector (a row's per-row statistics, attention's `[bq]` and `[bk]`) is copied out whole.
        del wait
        for part, view in zip(parts, views, strict=True):
            if smem is None:
                view[...] = part.astype(view.dtype)
                continue
            plgpu.store(smem, part.astype(smem.dtype), optimized=False)
            if len(smem.shape) == 1:
                mosaic.copied(smem, view)
                continue
            rows = smem.shape[0]
            mosaic.store_rows(smem, view, 0, 0, jnp.int32(rows), 0, rows, copy=mosaic.copied,
                              widest=resources.copied_rows(smem.shape[1], smem.dtype))

    def added(self, value: jax.Array, view, interpret: bool, *, smem) -> None:
        # Element-wise atomic adds into global memory (`red.global`; no TMA reduction before Hopper), so
        # every other role's and program's adds into the view stay whole.
        del smem
        if interpret:
            view[...] = view[...] + value
        else:
            plgpu.atomic_add(view, value, optimized=False)

    def column_sums(self, value: jax.Array, scratch, interpret: bool) -> jax.Array:
        # One warpgroup a program: Mosaic GPU's own reduction across its warps, through the scratch its
        # kernels' parameters reserve (`reduction_scratch`).
        del scratch, interpret
        return value.sum(axis=0)

    def published(self) -> None:
        # Its products read shared memory through ldmatrix, the warpgroup's own loads, which the store's
        # automatic barriers order.
        pass

    def held(self, ref):
        return plgpu.load(ref, layout=self.lhs, optimized=False)

    def _left(self, lhs):
        """A product's left operand in the MMA's layout: a shared-memory tile read, or a register value's
        (an accumulator's, the fork's relayout) cast."""
        if isinstance(lhs, jax.Array):
            return plgpu.layout_cast(lhs.astype(self.dtype), self.lhs)
        return plgpu.load(lhs, layout=self.lhs)

    def product(self, lhs, rhs, shape: tuple[int, int], then=None) -> jax.Array:
        out = plgpu.mma(self.zeros(shape), self._left(lhs), plgpu.load(rhs, layout=self.rhs),
                        m_warps=self.m_warps)
        if then is not None:
            then()
        return out

    def products(self, acc, lhs, rhs, interpret: bool):
        if interpret:
            # mma has no rule on a CPU: its sum, in fp32.
            return acc + jnp.dot(lhs[...], rhs[...], preferred_element_type=jnp.float32)
        if isinstance(lhs, jax.Array):
            return plgpu.mma(acc, self._left(lhs), plgpu.load(rhs, layout=self.rhs), m_warps=self.m_warps)
        # The left operand a step deep, sliced in registers along the depth at its layout's tiles; the
        # right one a sub-step at a time from the stage (`fragment_depth`), along its rows' 8-row tiles,
        # or, a transposed view (whose rows are a swizzled tile's columns, no narrower slice of which
        # lowers), a step deep and sliced in registers too.
        left, depth = self._left(lhs), self.fragment_depth(rhs.shape[0])
        if depth == rhs.shape[0]:
            return plgpu.mma(acc, left, plgpu.load(rhs, layout=self.rhs), m_warps=self.m_warps)
        transposed = isinstance(rhs, state_types.TransformedRef) and any(
            isinstance(transform, state_types.TransposeTransform) for transform in rhs.transforms)
        right = plgpu.load(rhs, layout=self.rhs) if transposed else None
        for start in range(0, rhs.shape[0], depth):
            part = (right[start:start + depth] if right is not None
                    else plgpu.load(rhs.at[pl.ds(start, depth)], layout=self.rhs))
            acc = plgpu.mma(acc, left[:, start:start + depth], part, m_warps=self.m_warps)
        return acc

    def registers_product(self, lhs: jax.Array, rhs, interpret: bool) -> jax.Array:
        del interpret
        return self.product(lhs, rhs, (lhs.shape[0], rhs.shape[1]))

    def register_products(self, acc, parts, rhs, interpret: bool):
        del interpret
        operand = plgpu.load(rhs, layout=self.rhs)
        for formed in parts:
            acc = plgpu.mma(acc, self._left(mosaic.unscaled_part(formed)), operand, m_warps=self.m_warps)
        return acc

    def staged_products(self, acc, parts, rhs, buffers, interpret: bool, *, transposed: bool = False):
        del interpret
        mosaic.unscaled(parts)
        operand = plgpu.load(rhs, layout=self.rhs)
        for place, (part, _) in enumerate(parts):
            buffers.at[place][...] = part.astype(buffers.dtype)
            staged = buffers.at[place].transpose((1, 0)) if transposed else buffers.at[place]
            acc = plgpu.mma(acc, plgpu.load(staged, layout=self.lhs), operand, m_warps=self.m_warps)
        return acc

    def rescaled_products(self, acc, scale, parts, rhs, interpret: bool):
        return self.register_products(scale(acc), parts, rhs, interpret)

    def settled(self, pending: int = 0) -> None:
        del pending


def _one(consumers: int, *, clustered: bool = False) -> None:
    """mma.sync's programs take one compute warpgroup (`Mma.consumers`), no producer, and no cluster
    (`clustered`: cp.async loads its own program's shared memory alone)."""
    if consumers != Mma.consumers or clustered:
        raise NotImplementedError(f"an mma.sync program takes one consumer warpgroup, not {consumers}, and "
                                  "no cluster")

