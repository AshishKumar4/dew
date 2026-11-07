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
from collections.abc import Callable
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
from dew.nn.kernels.templates import mosaic, product, report, resources


def _staging(block: tuple, swizzle: int | None) -> tuple:
    """A stream's tile's transforms: a tile's swizzled (its own, `swizzle` where a stream names one); a side
    vector's none, in whole 16-byte cp.async chunks, and a tile's none where it is read a row at a time
    (`mosaic.UNSWIZZLED`)."""
    shape, dtype = block
    if len(shape) == 1 or swizzle == mosaic.UNSWIZZLED:
        return ()
    return swizzled(shape, dtype) if swizzle is None else mosaic.swizzle_transforms(dtype, swizzle)


def _at(index_map, lo, step):
    """A stream's block indices at step `step` of a walk from `lo`."""
    return index_map(step + lo)


def _run(index_map, lo, each: int, step):
    """Run `each`'s block indices of a stream's `Rows` view at step `step` of a walk from `lo`: its table's
    entry (`mosaic.Rows`)."""
    rows = index_map(step + lo)
    if not isinstance(rows, mosaic.Rows):
        raise ValueError(f"a stream in pieces reads its runs through a table (`mosaic.Rows`), not {rows}")
    return (rows.table[rows.start + each], *rows.rest)


def _join_swizzle(block: tuple, streams, name: str) -> int:
    """The swizzle a stream's joined tile stages at: its own where it names one, else its copy's."""
    named = streams.pipeline.swizzles.get(name)
    return named if named is not None else mosaic.copied_swizzle(*block)


def _joined(pieces: list, joined, interpret: bool):
    """A step's pieces (`Rows`' runs, each landed in a slot of its own) copied into its one tile, in order,
    through registers a piece at a time, a row's columns over a warp's lanes (`mosaic._row_layout`: the one
    layout a plain piece's load and a swizzled tile's store both take on sm80, FS-PIECESV-1; each warp
    writing every row alike): a sync first, so no warp still reads the last step's tile, and after, so none
    reads a row not yet copied."""
    rows = pieces[0].shape[0]
    mosaic.synced(interpret)
    for each, piece in enumerate(pieces):
        target = joined.at[pl.ds(each * rows, rows)]
        if interpret:
            target[...] = piece[...]
        else:
            layout = mosaic._row_layout(piece.shape, piece.dtype)
            plgpu.store(target, plgpu.load(piece, layout=layout, optimized=False), optimized=False)
    mosaic.synced(interpret)
    return joined


def swizzled(shape: tuple[int, ...], dtype) -> tuple:
    """A `[rows, columns]` shared-memory tile's transforms: 8-row tiles as wide as the swizzle its row
    bytes allow and its cp.async copy tiles (`mosaic.copied_swizzle`), swizzled (bank-conflict-free
    ldmatrix), as jax's Ampere examples stage their operands."""
    return mosaic.swizzle_transforms(dtype, mosaic.copied_swizzle(shape, dtype))


def _swizzle(ref) -> int:
    """The swizzle, in bytes, a shared-memory operand's rows are staged at: 0 where it is staged plain."""
    if not isinstance(ref, state_types.TransformedRef):
        return mosaic.UNSWIZZLED
    return max((transform.swizzle for transform in ref.transforms
                if isinstance(transform, plgpu_core.UnswizzleRef)), default=mosaic.UNSWIZZLED)


def _depth_granule(ref, axis: int, instruction: int) -> int:
    """The depth slice a ref's staging allows: one MMA, or a whole swizzle along its tiled columns (no
    narrower slice of a swizzled tile lowers, FS-NARROW-LOWER1)."""
    if not isinstance(ref, state_types.TransformedRef):
        return instruction
    transposed = sum(isinstance(transform, state_types.TransposeTransform)
                     and transform.permutation == (1, 0) for transform in ref.transforms) % 2
    if axis != (0 if transposed else 1):
        return instruction
    return max(instruction, mosaic.swizzle_columns(ref.dtype, _swizzle(ref)))


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

    def acquire_all(self) -> None:
        """Every slot acquired: `emit_pipeline` hands the step its tiles whole."""


@dataclasses.dataclass(frozen=True)
class Pipeline:
    """A program's operand streams as `Mma.streams` describes them: each stream's tile `(shape, dtype)` by
    name and the tiles `emit_pipeline` keeps in flight (`stages`); it allocates their slots itself."""

    blocks: dict
    stages: int
    swizzles: dict = dataclasses.field(default_factory=dict)
    """The swizzle a stream's tiles stage at where `Mma.streams` names one (`mosaic.swizzle_for`)."""
    copies: dict = dataclasses.field(default_factory=dict)
    """The copies a stream's step arrives in where more than one (`mosaic.Pieces`, a `Rows` view's runs):
    each copied into a slot of its own, then joined into the stream's one tile a step reads (`walk`)."""


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

    # Its kernels take hooks and exact sums (their side loads, vector reads and running sum lower), but
    # its register model does not count what a hook holds, and its derived tiles spilled (ns-regs2: sm86's
    # normed forward 16 bytes). Its running sum is counted beside the fragments its products hold and one
    # step's fresh product (`summed_apart`, `grouped_matmul._Mosaic.registers`).
    counts_hooks = False

    role_rows = 4 * 16

    operands_in_registers = True

    products_overlap = False

    takes_tf32 = True

    transposes_tf32 = True

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
        return max(self.fragment_depth(depth), min(depth, mosaic.swizzle_columns(self.dtype, swizzle)))

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

    def column_registers(self, columns: int, dtype) -> int:
        # The accumulator's tiles are `m_warps * 16` rows by `n_warps * 8` columns, each warp's 16 x 8 a
        # thread's two columns a row (jax/experimental/mosaic/gpu/mma.py:63, `MMALayouts`' accumulator
        # tiling), so a column vector broadcast over the rows holds two entries a thread in each of the
        # vector's `columns / (n_warps * 8)` tiles: a quarter of its columns at four row warps, as
        # warpgroup MMA's, an eighth at two, a sixteenth at one.
        per_thread = -(-columns * 2 // (4 // self.m_warps * 8))
        return -(-per_thread * jnp.dtype(dtype).itemsize // 4)

    def row_registers(self, rows: int, dtype) -> int:
        # Each of the `m_warps` row warps' 16-row tiles a thread's two rows (`MMALayouts`' accumulator
        # tiling), so a thread holds two of each `m_warps * 16` rows.
        return -(-rows * 2 * jnp.dtype(dtype).itemsize // (self.m_warps * 16 * 4))

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

    def column_sums(self, value: jax.Array, scratch, interpret: bool) -> jax.Array:
        # One warpgroup a program: Mosaic GPU's own reduction across its warps, through the scratch its
        # kernels' parameters reserve (`reduction_scratch`).
        del scratch, interpret
        return value.sum(axis=0)

    def held(self, ref):
        # A resident operand stays in shared memory between products, not beside every loop carry.
        return ref

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
        # A plain tile (`mosaic.plain`).
        return dataclasses.replace(self, relay=tile)

    @property
    def _matrix_loads(self) -> bool:
        """Whether its operands load by `ldmatrix`, 16-bit rows of 8: not a 32-bit operand's (TF32's), for
        which Mosaic GPU's layout inference found no layout (tf32-mini at 9d77b39e7), its threads loading
        their elements themselves."""
        return self.dtype.itemsize <= 2

    def product(self, lhs, rhs, shape: tuple[int, int], then=None, *, interpret: bool = False) -> jax.Array:
        out = self.products(self.zeros(shape), lhs, rhs, interpret=interpret)
        if then is not None:
            then()
        return out

    def products(self, acc, lhs, rhs, interpret: bool, *, accumulate: bool = True):
        if not accumulate:
            # A fresh product: onto zeros, its accumulator a value (`read` is that value).
            acc = self.zeros(acc.shape)
        if jnp.dtype(lhs.dtype) != self.dtype:
            # Another dtype's operands (fp32's, as TF32) by that dtype's issuer: its fragments' layouts and
            # depths are its own, its accumulator's the same.
            return dataclasses.replace(self, dtype=jnp.dtype(lhs.dtype)).products(acc, lhs, rhs, interpret)
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
                        self.lhs_fragment_depth(int(lhs.shape[1]), swizzle=_swizzle(lhs)))
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

    def _mma(self, acc, left, right):
        """`acc + left @ right` by one mma.sync product of its operands' fragments: fp32's as TF32 through
        Mosaic GPU's lane-level `mma` (`inline_mgpu`; the fork's TF32 from 0.12.0.1.post42), the dialect's op
        whose operand types jaxlib fixes naming no fp32; any other dtype's by Pallas's `mma`."""
        if self.dtype != jnp.float32:
            return plgpu.mma(acc, left, right, m_warps=self.m_warps)
        issued: Callable[..., jax.Array] = plgpu.inline_mgpu(
            arg_types=(self.layout, self.lhs, self.rhs),
            return_type=plgpu.ShapeDtypeStruct(acc.shape, jnp.dtype(jnp.float32), layout=self.layout))(
            lambda ctx, acc, left, right: mgpu.mma(acc, left, right))
        # The accumulator laid as the product takes it: a fresh one's splat passed in whole left a cast no
        # pass lowers (tf32-dump-sm80 at ca3dd0f0a, a splat to the accumulator's registers).
        return issued(plgpu.layout_cast(acc, self.layout), left, right)

    def registers_product(self, lhs: jax.Array, rhs, interpret: bool) -> jax.Array:
        return self.products(self.zeros((lhs.shape[0], rhs.shape[1])), lhs, rhs, interpret)

    def summed_window(self, block_k: int) -> int:
        # Its accumulator is a value, carried across a window's steps and added into the running sum at
        # the window's end: a partial a step cost the A100's fp32 kernel gradients 2.13-2.26x their plain
        # sums on 13 dense shapes (M-FS-4, W13-a100-1). A window is `product.SUMMED_DEPTH` deep, its
        # truncating in-order adds no deeper than a promoted FP8 product's step.
        return max(1, product.SUMMED_DEPTH // block_k)

    def summed_apart(self, running: jax.Array, lhs, rhs, interpret: bool, scale: Callable | None = None
                     ) -> jax.Array:
        # A step's fresh accumulator starts from zeros tied to the running sum (an optimization barrier),
        # so no step's products issue before the last step's add. Its products are values, independent of
        # the running sum, so the compiler otherwise issued an unrolled walk's steps at once
        # (`emit_pipeline`'s drain), a whole accumulator live each: at five stages three beside the running
        # sum, 616 bytes spilled at the A100 forward's 128 x 128 tile; a traced grid's walk, its drain not
        # unrolled, spilled none (NSDBG-5).
        running, fresh = lax.optimization_barrier((running, self.zeros(running.shape)))
        fresh = self.products(fresh, lhs, rhs, interpret)
        return running + (fresh if scale is None else fresh * scale(0, running.shape[1]))

    def register_products(self, acc, parts, rhs, interpret: bool):
        for formed in parts:
            acc = self.products(acc, mosaic.unscaled_part(formed), rhs, interpret)
        return acc

    def rescaled_products(self, acc, scale, parts, rhs, interpret: bool):
        return self.register_products(scale(acc), parts, rhs, interpret)

    def added_into(self, acc: jax.Array, value: jax.Array, interpret: bool) -> jax.Array:
        """`Issuer.added_into` for a register accumulator: its sum with `value`, in the accumulator's
        layout."""
        del interpret
        return acc + self.as_tile(value)

    def settled(self, pending: int = 0) -> None:
        del pending


class CpAsync(mosaic.Family[Pipeline, Streams, jax.Array]):
    """The execution policy before Hopper (`mosaic.Transport`): one compute warpgroup a program and no
    producer, its loads `emit_pipeline`'s cp.async copies into each walk's own slots, its accumulators in
    the pipeline's carry (a register home), its output stored through registers. A family whose issuer half
    an issuer composed with it gives (`Mma`)."""

    walks_scoped = True  # `emit_pipeline` allocates each walk's slots in its own scope

    bounded_loads = False

    # A step loading ahead of the one consumed leaves cp.async's latency unhidden: two steps 64 deep stalled
    # on barriers 1.1-1.5 an issue against XLA's 0.3-0.7 (FS-NCU-DW2-a100), and 32-deep steps three
    # deep took dense products 4-8% less time than 64 deep two (FS-DENSE-STEPS1-a100).
    loads_ahead = 2

    stores_through_registers = True

    copies_vectors = False  # cp.async spreads a copy over the warpgroup's threads, 128 elements at least

    pieces_of_views = False  # its pieces are a table's runs (`_run`), a paged call's pages alone

    adds_vectors = False  # its atomic adds take no layout replicating a vector's entries (a [bk] key sum's)

    adds_through_shared = False  # `added`'s atomic adds go from registers (`red.global`), its `smem` unread

    @property
    @abc.abstractmethod
    def lhs(self) -> plgpu_core.SomeLayout:
        """The layout a tile loaded through registers takes (`loaded`): its issuer's left operand's."""

    def registers(self, consumers: int, programs: int = 1) -> int:
        # Each role at its thread's share of the multiprocessor's 64K registers: two programs' 256 threads
        # hold 255 each.
        _one(consumers)
        return resources.entry_registers(consumers, programs)

    def kernel(self, entry, *, out_type, grid: tuple, grid_names: tuple, consumers: int,
               interpret: bool, approx_math: bool = True, name: str | None = None,
               cluster: int = 1, cost: pl.CostEstimate | None = None, model_flops: int | None = None,
               reducing_warpgroups: int = 0, resident: int = 1, scalars: int = 0) -> Callable:
        # Its one warpgroup its consumer, no producer beside it, and no cluster; its kernels' default
        # parameters reserve the scratch an implicit reduction takes (`reduction_scratch`).
        del reducing_warpgroups, scalars  # its programs read a table from global memory as any operand
        _one(consumers, clustered=cluster > 1)
        built = launches.kernel(mosaic.dependent(entry, interpret), name=name or self.name,
                                out_type=out_type, grid=grid, grid_names=grid_names,
                                compiler_params=plgpu.CompilerParams(approx_math=approx_math,
                                                                     min_blocks_per_sm=resident),
                                interpret=mosaic.interpreted(interpret),
                                kernel_name=report.named(name), cost_estimate=cost)
        return mosaic.checked(built, name or self.name, resident, model_flops)

    def program_scoped(self, body, *allocations, streams):
        return pl.run_scoped(lambda *refs: body(*refs, streams=streams), *allocations)

    def roles(self, consumers: int, consumer, producer) -> None:
        _one(consumers)
        del producer
        consumer(0)

    def streams(self, blocks: dict, stages: int, consumers: int, multicast: tuple[str, ...] = (),
                swizzles: dict[str, int] | None = None, copies: dict[str, int] | None = None):
        _one(consumers, clustered=bool(multicast))
        # A stream's pieces each a block of `emit_pipeline`'s own, in whole 8-row tiles, joined by its
        # consumer.
        copies = {name: count for name, count in (copies or {}).items() if count > 1}
        for name, count in copies.items():
            rows = blocks[name][0][0]
            if rows % (count * mosaic.TILE_ROWS):
                raise ValueError(f"{name}'s {rows} rows arrive in no {count} pieces of whole "
                                 f"{mosaic.TILE_ROWS}-row tiles")
        return Pipeline(blocks, stages, dict(swizzles or {}), copies)


    def lifetimes(self, streams: Pipeline, steps: int | None = None) -> tuple[tuple, tuple]:
        # `emit_pipeline` allocates a walk's slots in its own scope, no more of them than the walk's steps.
        slots = streams.stages if steps is None else min(streams.stages, steps)
        tiles = tuple(plgpu.SMEM((slots, *shape), dtype) for shape, dtype in streams.blocks.values())
        # A stream in pieces: its slots the same bytes in pieces, and the one tile they join into (`walk`).
        joins = tuple(plgpu.SMEM(*streams.blocks[name]) for name in streams.copies)
        return (), ((*tiles, *joins),)

    def opened(self, allocated, views: dict | None = None):
        if views is None:
            raise ValueError("a pipelined family's streams take their views where they open (its walk loads)")
        return Streams(allocated, views)

    def stream(self, streams, views, lo, hi, *, adds: tuple | None = None) -> None:
        del streams, views, lo, hi, adds

    def persist(self, streams, tiles: tuple[int, ...], schedule: Callable, issue: Callable, finish: Callable,
                *, consumers: int, axis: str, outputs: tuple, interpret: bool, in_flight: int = 0,
                unrolled: bool = False, accumulators: tuple = ()) -> None:
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
                mosaic.store_rows(smem, ref, *place, copy=mosaic.copied,
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

        mosaic.consumed(self, tiles, schedule, issue, finish, store, 0, walking, axis=axis)

    def walk[C](self, streams, segments, start: C, shapes: tuple, interpret: bool, *,
                first: jax.Array | int = 0, in_flight: int = 0, initial: tuple | None = None,
                unrolled: bool = False) -> tuple[C, tuple[jax.Array, ...]]:
        # Its products are synchronous: none stays in flight past its step, and `emit_pipeline` gives
        # each stage back as its step ends, so no step holds one past it.
        del first, in_flight, unrolled
        blocks, copies = streams.pipeline.blocks, streams.pipeline.copies
        names = tuple(blocks)
        carry = (start, self.started(shapes, initial))
        for step, lo, hi in segments:
            if isinstance(lo, int) and isinstance(hi, int) and hi <= lo:
                # No steps: the carry and its zero accumulators as they start (`emit_pipeline` takes no
                # empty grid).
                continue

            def body(indices, *refs, joins: dict, step=step, lo=lo):
                (index,), (carried, accumulators) = indices, refs[-1]
                landed, slots = iter(refs[:-1]), {}
                for name in names:
                    pieces = [next(landed) for _ in range(copies.get(name, 1))]
                    slots[name] = pieces[0] if len(pieces) == 1 else _joined(pieces, joins[name], interpret)
                return step(index + lo, Slots(slots), carried, accumulators)

            # A tile's blocks staged swizzled for ldmatrix (`_staging`); a stream in pieces a block a run of
            # its `Rows` view, each staged plain for its join's register copy.
            specs = []
            for name in names:
                shape = blocks[name][0]
                index_map = streams.views[name][1]
                if name not in copies:
                    staging = _staging(blocks[name], streams.pipeline.swizzles.get(name))
                    specs.append(plgpu.BlockSpec(shape, functools.partial(_at, index_map, lo),
                                                 transforms=staging))
                    continue
                run = (shape[0] // copies[name], *shape[1:])
                specs += [plgpu.BlockSpec(run, functools.partial(_run, index_map, lo, each))
                          for each in range(copies[name])]
            operands = [streams.views[name][0] for name in names for _ in range(copies.get(name, 1))]

            def piped(*joined, body=body, specs=specs, carry=carry, hi=hi, lo=lo, operands=operands):
                pipeline = plgpu.emit_pipeline(
                    functools.partial(body, joins=dict(zip(copies, joined, strict=True))), grid=(hi - lo,),
                    in_specs=specs, max_concurrent_steps=streams.pipeline.stages, init_carry=carry)
                return pipeline(*operands)

            # The joined tiles staged as the stream's whole tiles are (`lifetimes`' walk scope).
            joins = (mosaic.staged(blocks[name][0], blocks[name][1],
                                   _join_swizzle(blocks[name], streams, name)) for name in copies)
            carry = pl.run_scoped(piped, *joins) if copies else piped()
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
            mosaic.store_rows(smem, view, 0, 0, jnp.int32(rows), 0, rows, copy=mosaic.coalesced,
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

    def column_unit(self, dtype) -> int:
        # Its stores move rows through registers, a row's columns over a warp's lanes at least
        # (`mosaic.copied`), each lane holding an element (the guarded attention walk's 32-column blocks,
        # whose 64 held no tile's registers at width 128).
        del dtype
        return resources.THREADS // 4

    def depths(self, dtype) -> tuple[int, ...]:
        # Its operand blocks' rows staged at a 128- or 64-byte swizzle (`swizzled`).
        return tuple(mosaic.swizzle_columns(dtype, swizzle) for swizzle in (mosaic.SWIZZLE_BYTES, 64))


def _one(consumers: int, *, clustered: bool = False) -> None:
    """A cp.async family's programs take one compute warpgroup, no producer, and no cluster (`clustered`:
    cp.async loads its own program's shared memory alone)."""
    if consumers != 1 or clustered:
        raise NotImplementedError(f"an mma.sync program takes one consumer warpgroup, not {consumers}, "
                                  "and no cluster")

