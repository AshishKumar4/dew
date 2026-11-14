"""The mma.sync instruction family's bundle (Ampere and its consumer kin, sm80 to sm89 and consumer
Blackwell; `generation.family` 'mma.sync') for the templates' Mosaic mainloops (`mosaic.Family`): one
warpgroup a program, its operand tiles streamed into shared memory by `plgpu.emit_pipeline` (cp.async),
read into registers in the MMA layouts (ldmatrix), and multiplied synchronously (mma.sync) into an
accumulator the walk carries in registers. An accumulator's value is the next product's left operand in
registers (the fork's relayout, jax 0.11.2.post13 and later), as FlashAttention-2 reuses its fragments.
Its layouts are its operands' dtype's (`Layout.MMA_ACC(dtype)`), so a bundle is a dtype's (`Mma`)."""

import abc
import dataclasses
import functools
from collections.abc import Callable, Iterator
from typing import Any, Self

import jax
import jax.numpy as jnp
from jax import lax
from jax._src.pallas.mosaic_gpu import core as plgpu_core
from jax._src.state import types as state_types
from jax.experimental import pallas as pl
from jax.experimental.mosaic import gpu as mgpu
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import launches
from dew.nn.kernels.templates import (
    bodies,
    emission,
    handoffs,
    mosaic,
    persistent,
    report,
    resources,
    swizzling,
)
from dew.nn.kernels.templates.cost import Stated, Work


def _staging(block: tuple, swizzle: int | None) -> tuple:
    """A stream's tile's transforms: a tile's swizzled (its own, `swizzle` where a stream names one); a side
    vector's none, in whole 16-byte cp.async chunks, and a tile's none where it is read a row at a time
    (`swizzling.UNSWIZZLED`)."""
    shape, dtype = block
    if len(shape) == 1 or swizzle == swizzling.UNSWIZZLED:
        return ()
    return swizzled(shape, dtype) if swizzle is None else swizzling.swizzle_transforms(dtype, swizzle)


def _at(index_map, lo, step):
    """A stream's block indices at step `step` of a walk from `lo`."""
    return index_map(step + lo)


def _gathered_walk(streams, segments, carry, interpret: bool):
    """`CpAsync.walk` over streams whose step's rows a table gathers (`Pipeline.gathered`), `segments` as
    `walk`'s: each stream's slots the walk's own, staged at the swizzle its rows' copy tiles
    (`gathered.swizzled`), a step's rows copied by the warpgroup a row a copy (`gathered.row_copies`), every
    stream's copies of a step one cp.async group, as `emit_pipeline` loads its own: a step waits for its own
    group (`plgpu.wait_gmem_to_smem`, the later groups in flight `stages - 2`, a group committed empty where
    no step remains to load, so the count stays an immediate), then issues the group of the step `stages - 1`
    ahead into the slot the step before read, once the warpgroup is past those reads (`mosaic.synced`).
    Interpreted, each row copied by the program's own loads and stores."""
    from dew.nn.kernels.templates import gathered as gathered_rows  # it builds on the toolkit's rings

    blocks = streams.pipeline.blocks
    names = tuple(blocks)
    swizzles = {name: gathered_rows.swizzled(shape, dtype) for name, (shape, dtype) in blocks.items()}
    for step, lo, hi in segments:
        if isinstance(lo, int) and isinstance(hi, int) and hi <= lo:
            continue
        stages = streams.pipeline.stages
        if isinstance(lo, int) and isinstance(hi, int):
            stages = min(stages, hi - lo)

        def walked(*slots, step=step, lo=lo, hi=hi, stages=stages, carry=carry):
            def load(index) -> None:
                # Step `index`'s rows of every stream into its slot, committed as one group (empty past `hi`).
                @pl.when(index < hi)
                def _():
                    for name, ring in zip(names, slots, strict=True):
                        ref, index_map = streams.views[name]
                        rows = index_map(index)
                        [depth] = rows.rest
                        height, width = blocks[name][0]
                        source = ref.at[:, pl.ds(depth * width, width)]
                        table = rows.table.at[pl.ds(rows.start, height)]
                        slot = ring.at[lax.rem(index - lo, stages)]
                        _row_copied(source, table, slot, swizzles[name], interpret)

                if not interpret:
                    plgpu.inline_mgpu()(gathered_rows.commit_group)()

            for ahead in range(stages - 1):
                load(lo + ahead)

            def body(index, carried):
                if stages == 1:
                    mosaic.synced(interpret)
                    load(index)
                if not interpret:
                    plgpu.wait_gmem_to_smem(max(0, stages - 2))
                if stages > 1:
                    # The slot the step before read is the one this step's copies `stages - 1` ahead fill.
                    mosaic.synced(interpret)
                    load(index + stages - 1)
                slot = lax.rem(index - lo, stages)
                started, accumulators = carried
                tiles = {name: ring.at[slot] for name, ring in zip(names, slots, strict=True)}
                return step(index, mosaic.WholeSlots(tiles), started, accumulators)

            return lax.fori_loop(lo, hi, body, carry)

        carry = pl.run_scoped(walked, *(swizzling.staged((stages, *shape), dtype, swizzles[name])
                                         for name, (shape, dtype) in blocks.items()))
    return carry


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
    plgpu.inline_mgpu(arg_types=refs)(
        lambda ctx, source, rows, slot: gathered_rows.row_copies(source, rows, slot, swizzle=swizzle))(
            source, table, slot)


def swizzled(shape: tuple[int, ...], dtype) -> tuple:
    """A `[rows, columns]` shared-memory tile's transforms: 8-row tiles as wide as the swizzle its row
    bytes allow and its cp.async copy tiles (`swizzling.copied_swizzle`), swizzled (bank-conflict-free
    ldmatrix), as jax's Ampere examples stage their operands."""
    return swizzling.swizzle_transforms(dtype, swizzling.copied_swizzle(shape, dtype))


def _depth_granule(ref, axis: int, instruction: int) -> int:
    """The depth slice a ref's staging allows: one MMA, or a whole swizzle along its tiled columns (no
    narrower slice of a swizzled tile lowers, FS-NARROW-LOWER1)."""
    if not isinstance(ref, state_types.TransformedRef):
        return instruction
    transposed = sum(isinstance(transform, state_types.TransposeTransform)
                     and transform.permutation == (1, 0) for transform in ref.transforms) % 2
    if axis != (0 if transposed else 1):
        return instruction
    return max(instruction, swizzling.swizzle_columns(ref.dtype, swizzling.swizzle_of(ref)))


@dataclasses.dataclass(frozen=True)
class Pipeline:
    """A program's operand streams as `Mma.streams` describes them: each stream's tile `(shape, dtype)` by
    name and the tiles `emit_pipeline` keeps in flight (`stages`); it allocates their slots itself."""

    blocks: dict
    stages: int
    swizzles: dict = dataclasses.field(default_factory=dict)
    """The swizzle a stream's tiles stage at where `Mma.streams` names one (`swizzling.swizzle_for`)."""
    gathered: tuple = ()
    """The streams whose step's rows a table gathers (a `Rows` view of one row a run): the warpgroup's own
    cp.async copies, a row a copy (`gathered.row_copies`), into slots of the walk's own, in the pipeline's
    cp.async groups (`_gathered_walk`); a walk gathers every stream or none."""


@dataclasses.dataclass(frozen=True)
class Streams:
    """A program's operand streams (`Mma.opened`): its pipeline and each stream's view `(ref, block)` by
    name, the `[S, D]` operand its steps' tiles come from in `block` rows."""

    pipeline: Pipeline
    views: dict


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

    def rhs_fragment_depth(self, depth: int, *, transposed: bool = False, swizzle: int | None = None) -> int:
        # A transposed stage's rows are its depth: a load takes a whole swizzle tile of it (`products`), one
        # instruction's at the narrowest swizzle (a stream staged so by `streams`' `swizzles`: D64's K in
        # QKᵀ at the widest held its whole 64 x 192 stage, 162 registers, census80). An ordinary one slices
        # its rows' 8-row tiles.
        return self._staged_depth(depth, swizzle) if transposed else self.fragment_depth(depth)

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

    def chunks(self, accumulators: tuple) -> Iterator[tuple[int, tuple]]:
        # The accumulators in their own layout, stated: an epilogue's broadcast of a side vector across one
        # takes the accumulator's layout as its result's (`side_tile`'s vectors its reductions), where left to
        # inference it took the vector's ("Tiling rank different than broadcast result rank, (16,) vs
        # (64, 128)", a scaled forward on sm80, EM-hooks6).
        yield 0, tuple(plgpu.layout_cast(acc, self.layout) for acc in accumulators)

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

    def _left(self, lhs):
        """A product's left operand in the MMA's layout: a shared-memory tile read, or a register value's (an
        accumulator's) cast, in registers where each warp holds whole rows of it or no relay tile is bound,
        else through the bound one (`relaying`): every warp's columns stored, then each warp's rows read
        whole, a sync after each so no warp reads a column not yet stored or stores over one another still
        reads."""
        if not isinstance(lhs, jax.Array):
            return plgpu.load(lhs, layout=self.lhs, optimized=self._matrix_loads)
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

    def relay_bytes(self, rows: int, depth: int) -> int:
        # Its warps along a tile's rows alone (four of them) hold whole rows of the accumulator and the
        # operand alike; fewer split the accumulator's columns among warps, which no register cast crosses.
        return 0 if self.m_warps == 4 else rows * depth * self.dtype.itemsize

    def relaying(self, tile) -> Self:
        # A plain tile (`swizzling.plain`).
        return dataclasses.replace(self, relay=tile)

    @property
    def _matrix_loads(self) -> bool:
        """Whether its operands load by `ldmatrix`, 16-bit rows of 8: not a 32-bit operand's (TF32's), for
        which Mosaic GPU's layout inference found no layout (tf32-mini at 9d77b39e7), its threads loading
        their elements themselves."""
        return self.dtype.itemsize <= 2

    def product(self, lhs, rhs, shape: tuple[int, int], then=None) -> jax.Array:
        out = self.products(self.zeros(shape), lhs, rhs)
        if then is not None:
            then()
        return out

    def products(self, acc, lhs, rhs, *, accumulate: bool = True):
        interpret = bodies.interpreting()  # the launch's (`bodies.interpreting`)
        if not accumulate:
            # A fresh product: onto zeros, its accumulator a value (`read` is that value).
            acc = self.zeros(acc.shape)
        if jnp.dtype(lhs.dtype) != self.dtype:
            # Another dtype's operands (fp32's, as TF32) by that dtype's issuer: its fragments' layouts and
            # depths are its own, its accumulator's the same.
            return self.for_dtype(lhs.dtype).products(acc, lhs, rhs)
        if interpret:
            # mma has no rule on a CPU: its sum, in fp32.
            return acc + jnp.dot(lhs[...], rhs[...], preferred_element_type=jnp.float32)
        # The left operand loads a whole swizzle tile of its depth at a time (`lhs_fragment_depth`), as a
        # transposed left operand loaded in instruction slices took the A100's kernel gradient 7% slower
        # (FS-DENSE-AB1); the right loads only the fragments the next instructions consume, as narrow as
        # its staging allows (`_depth_granule`), which took its input gradient 12% faster. A register operand
        # (an accumulator's value) is cast once and sliced. Each instruction's fragments load before the
        # product ahead of it issues, CUTLASS's warp-level double buffering: every dense kernel gradient on
        # an A100 took 12 to 14% less time, its outputs bitwise the same, the forward and input gradient
        # unchanged (FS-C1-NARROW1-a100).
        instruction = self.fragment_depth(int(rhs.shape[0]))
        lhs_depth = max(_depth_granule(lhs, 1, instruction),
                        self.lhs_fragment_depth(int(lhs.shape[1]), swizzle=swizzling.swizzle_of(lhs)))
        rhs_depth = _depth_granule(rhs, 0, instruction)
        held = self._left(lhs) if isinstance(lhs, jax.Array) else None
        lefts: dict[int, jax.Array] = {}
        rights: dict[int, jax.Array] = {}

        def loaded(start: int) -> tuple[jax.Array, jax.Array]:
            """Instruction `start`'s fragments, its loads issued the first time a slice is asked for."""
            left_at, right_at = start - start % lhs_depth, start - start % rhs_depth
            if held is None and left_at not in lefts:
                lefts[left_at] = self._left(lhs.at[:, pl.ds(left_at, lhs_depth)])
            if right_at not in rights:
                rights[right_at] = plgpu.load(rhs.at[pl.ds(right_at, rhs_depth)], layout=self.rhs,
                                              optimized=self._matrix_loads)
            left = held[:, start:start + instruction] if held is not None else lefts[left_at][
                :, start - left_at:start - left_at + instruction]
            return left, rights[right_at][start - right_at:start - right_at + instruction]

        starts = range(0, rhs.shape[0], instruction)
        fragments = loaded(starts[0])
        for following in starts[1:]:
            ahead = loaded(following)
            acc = self._mma(acc, *fragments)
            fragments = ahead
        return self._mma(acc, *fragments)

    def looped_products(self, acc, lhs, rhs, *, accumulate: bool = True):
        """`products`, its instruction slices issued in a loop of their own, each slice's operands loaded from
        shared memory within it: unrolled, ptxas interleaved a fused backward step's three later products
        (dV, dK, dQ, independent of each other) and held 170 words of their fragments at once, an A100's step
        at 64 x 64, width 64, under a softcap spilling 96 bytes; looped, 20 (one slice's), none spilled, the
        same 320 mma.sync instructions issued (row 5477). Interpreted, `products`."""
        if bodies.interpreting():
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

    def _mma(self, acc, left, right):
        """`acc + left @ right` by one mma.sync product of its operands' fragments: fp32's as TF32 through
        Mosaic GPU's lane-level `mma` (`inline_mgpu`; the fork's TF32 from 0.12.0.1.post42), the dialect's op
        whose operand types jaxlib fixes naming no fp32; any other dtype's by Pallas's `mma`."""
        if self.dtype != jnp.float32:
            return plgpu.mma(acc, left, right, m_warps=self.m_warps)
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
        # (`emit_pipeline`'s drain), a whole accumulator live each: at five stages three beside the running
        # sum, 616 bytes spilled at the A100 forward's 128 x 128 tile; a traced grid's walk, its drain not
        # unrolled, spilled none (NSDBG-5).
        running, fresh = lax.optimization_barrier((running, self.zeros(running.shape)))
        fresh = self.products(fresh, lhs, rhs)
        return running + (fresh if scale is None else fresh * scale(0, running.shape[1]))

    def register_products(self, acc, parts, rhs):
        for formed in parts:
            acc = self.products(acc, mosaic.unscaled_part(formed), rhs)
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


@emission.stating(lambda acc, left, right: Work(tf32_flops=2 * acc.shape[0] * acc.shape[1] * left.shape[1]))
def _tf32_product(ctx, acc, left, right):
    """`acc + left @ right`, fp32's operands as TF32 by one mma.sync (`MmaIssuer._mma`), inlined: `2 m n k`
    TF32 FLOPs, its accumulator `[m, n]` and its left operand `[m, k]`."""
    del ctx
    return mgpu.mma(acc, left, right)


class CpAsync(mosaic.Family[Pipeline, Streams, jax.Array]):
    """The execution policy before Hopper (`mosaic.Transport`): one compute warpgroup a program and no
    producer, its loads `emit_pipeline`'s cp.async copies into each walk's own slots, its accumulators in
    the pipeline's carry (a register home), its output stored through registers. A family whose issuer half
    an issuer composed with it gives (`Mma`)."""

    walks_scoped = True  # `emit_pipeline` allocates each walk's slots in its own scope

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
    gathers_rows = True  # its one warpgroup's cp.async copies gather a step's rows (`_gathered_walk`)
    gathers_alone = False  # its walk gathers every stream's rows or none (`streams`)
    gathers_interpreted = True

    copies_in_pieces = False  # its pipeline loads a step's block whole, or gathers its rows

    adds_vectors = False  # its atomic adds take no layout replicating a vector's entries (a [bk] key sum's)

    adds_through_shared = False  # `added`'s atomic adds go from registers (`red.global`), its `smem` unread

    @property
    @abc.abstractmethod
    def lhs(self) -> plgpu_core.SomeLayout:
        """The layout a tile loaded through registers takes (`loaded`): its issuer's left operand's."""

    def registers(self, consumers: int, programs: int = 1, producer: int | None = None) -> int:
        # Each role at its thread's share of the multiprocessor's 64K registers: two programs' 256 threads
        # hold 255 each.
        _one(consumers)
        return resources.entry_registers(consumers, programs)

    def kernel(self, entry, *, out_type, grid: tuple, grid_names: tuple, consumers: int,
               interpret: bool, approx_math: bool = True, name: str | None = None,
               cluster: int = 1, cost: pl.CostEstimate | Stated | None = None,
               model_flops: int | None = None,
               reducing_warpgroups: int = 0, resident: int = 1, scalars: int = 0,
               prefetched: Callable | None = None, registers: int | None = None,
               producer: int | None = None, shared: int | None = None,
               moves: bool | None = None) -> Callable:
        # Its one warpgroup its consumer, no producer beside it, and no cluster; its kernels' default
        # parameters reserve the scratch an implicit reduction takes (`reduction_scratch`).
        # Its programs read a table from global memory as any operand, and no producer runs beside them.
        del reducing_warpgroups, scalars, producer, moves  # one warpgroup: no roles to move registers between
        _one(consumers, clustered=cluster > 1)
        built = launches.kernel(bodies.dependent(entry, interpret, prefetched=prefetched),
                                name=name or self.name,
                                out_type=out_type, grid=grid, grid_names=grid_names,
                                compiler_params=plgpu.CompilerParams(approx_math=approx_math,
                                                                     min_blocks_per_sm=resident),
                                interpret=bodies.interpreted(interpret),
                                kernel_name=report.named(
                                    name, model=(None if registers is None
                                                 else resources.register_model(self, consumers, registers,
                                                                               shared=shared))),
                                cost_estimate=emission.built_with(cost))
        return mosaic.checked(emission.costed(built, cost), name or self.name, resident, model_flops)

    def program_scoped(self, body, *allocations, streams):
        return pl.run_scoped(lambda *refs: body(*refs, streams=streams), *allocations)

    def roles(self, consumers: int, producer: int | None = None) -> handoffs.Schedule:
        _one(consumers)
        return handoffs.Schedule((handoffs.Role("consumer"),), axis=None)

    def streams(self, blocks: dict, stages: int, consumers: int, multicast: tuple[str, ...] = (),
                swizzles: dict[str, int] | None = None, copies: dict[str, int] | None = None,
                gathered: tuple[str, ...] = ()):
        _one(consumers, clustered=bool(multicast))
        if gathered and set(gathered) != set(blocks):
            raise NotImplementedError("an A100's walk gathers every stream's rows or none")
        if any(count > 1 for count in (copies or {}).values()):
            raise ValueError("an A100's pipeline loads a step's block in one copy, not in pieces")
        return Pipeline(blocks, stages, dict(swizzles or {}), tuple(gathered))


    def lifetimes(self, streams: Pipeline, steps: int | None = None) -> tuple[tuple, tuple]:
        # `emit_pipeline` allocates a walk's slots in its own scope, no more of them than the walk's steps.
        slots = streams.stages if steps is None else min(streams.stages, steps)
        tiles = tuple(plgpu.SMEM((slots, *shape), dtype) for shape, dtype in streams.blocks.values())
        return (), (tiles,)

    def opened(self, allocated, views: dict | None = None):
        if views is None:
            raise ValueError("a pipelined family's streams take their views where they open (its walk loads)")
        return Streams(allocated, views)

    def stream(self, streams, views, lo, hi) -> None:
        del streams, views, lo, hi

    def persist(self, streams, tiles: tuple[int, ...], schedule: Callable, issue: Callable, finish: Callable,
                *, consumers: int, axis: str, outputs: tuple, interpret: bool, in_flight: int = 0,
                unrolled: bool = False, accumulators: tuple = (), producer: int | None = None,
                pingpong: bool = False) -> None:
        del producer  # no producer role of its own
        self.pingponged(pingpong)
        # Each tile's pipeline fills and drains before its output copy. Synchronize at that lifetime
        # boundary so the output's shared memory can reuse the pipeline's slots.
        del in_flight, unrolled, accumulators  # its walk carries its accumulators
        staging = self.output_staging(outputs, consumers)

        def store(output, value, ref, place):
            # Interpreted pipelines use barrier-tracked copies, not cp.async groups.
            if not interpret:
                plgpu.wait_gmem_to_smem(0)

            def copied(smem):
                # The row-copy ladder needs strided layouts, not optimized-only transfers.
                plgpu.store(smem, value.astype(smem.dtype), optimized=False)
                mosaic.store_rows(smem, ref, *place, copy=swizzling.copied,
                                  widest=resources.copied_rows(smem.shape[1], smem.dtype))

            pl.run_scoped(copied, staging[output])
            if not interpret:
                plgpu.wait_gmem_to_smem(0)

        def walking(counter, active, steps, views):
            del counter, active

            def walk(step, start, shapes, *, initial=None):
                return self.walk(self.opened(streams, views), ((step, 0, steps),), start, shapes,
                                 interpret, initial=initial)

            return walk

        persistent.consumed(self, tiles, schedule, issue, finish, store, 0, walking, axis=axis)

    def walk[C](self, streams, segments, start: C, shapes: tuple, interpret: bool, *,
                first: jax.Array | int = 0, in_flight: int = 0, initial: tuple | None = None,
                unrolled: bool = False, ahead: tuple[str, ...] = ()) -> tuple[C, tuple[jax.Array, ...]]:
        # Its products are synchronous: none stays in flight past its step, and `emit_pipeline` gives
        # each stage back as its step ends, so no step holds one past it, nor reads one ahead.
        if ahead:
            raise NotImplementedError(f"{self.name}'s pipeline reads no stream ahead")
        del first, in_flight, unrolled
        segments = mosaic.stepped(segments)
        blocks = streams.pipeline.blocks
        names = tuple(blocks)
        carry = (start, self.started(shapes, initial))
        if streams.pipeline.gathered:
            return _gathered_walk(streams, segments, carry, interpret)
        for step, lo, hi in segments:
            if isinstance(lo, int) and isinstance(hi, int) and hi <= lo:
                # No steps: the carry and its zero accumulators as they start (`emit_pipeline` takes no
                # empty grid).
                continue

            def body(indices, *refs, step=step, lo=lo):
                (index,), (carried, accumulators) = indices, refs[-1]
                slots = mosaic.WholeSlots(dict(zip(names, refs[:-1], strict=True)))
                return step(index + lo, slots, carried, accumulators)

            # A tile's blocks staged swizzled for ldmatrix (`_staging`), each read inside its operand: its
            # launch extends every operand its rows read past (`bounded_loads`, `mosaic_gpu._walked_rows`),
            # so cp.async's copies are promised in bounds, unchecked.
            specs = [plgpu.BlockSpec(blocks[name][0], functools.partial(_at, streams.views[name][1], lo),
                                     transforms=_staging(blocks[name], streams.pipeline.swizzles.get(name)),
                                     oob_fill_mode=plgpu.OOBFillMode.PROMISE_IN_BOUNDS)
                     for name in names]
            pipeline = plgpu.emit_pipeline(body, grid=(hi - lo,), in_specs=specs,
                                           max_concurrent_steps=streams.pipeline.stages, init_carry=carry)
            carry = pipeline(*(streams.views[name][0] for name in names))
        return carry

    def passed(self, streams, lo, hi, *, first: jax.Array | int = 0, between=None) -> None:
        del streams, lo, hi, first, between

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
            if smem is None:
                view[...] = part.astype(view.dtype)
                continue
            if len(smem.shape) == 1:
                plgpu.store(view, part.astype(view.dtype), optimized=False)
                continue
            plgpu.store(smem, part.astype(smem.dtype), optimized=False)
            rows = smem.shape[0]
            mosaic.store_rows(smem, view, 0, 0, jnp.int32(rows), 0, rows, copy=swizzling.coalesced,
                              widest=resources.copied_rows(smem.shape[1], smem.dtype))

    def added(self, value: jax.Array, view, interpret: bool, *, smem, pending: int = 0) -> None:
        # Element-wise atomic adds into global memory (`red.global`; no TMA reduction before Hopper), so
        # every other role's and program's adds into the view stay whole, no buffer between them.
        del smem, pending
        if interpret:
            view[...] = view[...] + value
        else:
            plgpu.atomic_add(view, value, optimized=False)

    def published(self) -> None:
        # Its products read shared memory through ldmatrix, the warpgroup's own loads, which the store's
        # automatic barriers order.
        pass



@dataclasses.dataclass(frozen=True)
class Mma(MmaIssuer, CpAsync):
    """mma.sync's bundle: its issuer and register home (`MmaIssuer`) on `emit_pipeline`'s cp.async policy
    (`CpAsync`)."""

    name = "mma.sync"

    # One consumer warpgroup a program. Two running one cp.async pipeline (`emit_pipeline`'s thread_axis)
    # hung on an A100 at every attention forward, one 64-key step among them (FS-C2HANG-a100, M-ET-15),
    # spilled where the forward derived them (sm80 width 64, split residuals), and no GEMM derived them;
    # FlashAttention 2's rows come from one warpgroup's wider tiles, not a second warpgroup.
    consumers = 1


def _one(consumers: int, *, clustered: bool = False) -> None:
    """A cp.async family's programs take one compute warpgroup, no producer, and no cluster (`clustered`:
    cp.async loads its own program's shared memory alone)."""
    if consumers != 1 or clustered:
        raise NotImplementedError(f"an mma.sync program takes one consumer warpgroup, not {consumers}, "
                                  "and no cluster")

