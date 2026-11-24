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
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Self

import jax
import jax.numpy as jnp
from jax import lax
from jax._src.pallas.mosaic_gpu import core as plgpu_core
from jax._src.state import types as state_types
from jax.experimental import pallas as pl
from jax.experimental.mosaic import gpu as mgpu
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import generation, launches
from dew.nn.kernels.templates import (
    bodies,
    emission,
    handoffs,
    mosaic,
    outputs,
    persistent,
    report,
    resources,
    rings,
    stores,
    swizzling,
)
from dew.nn.kernels.templates.cost import Stated, Work
from dew.nn.programs import meshless


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
    for step, lo, hi in segments:
        if isinstance(lo, int) and isinstance(hi, int) and hi <= lo:
            continue
        known = isinstance(lo, int) and isinstance(hi, int)
        scoped = _gathered_slots(streams.pipeline, hi - lo if known else None)
        stages = scoped[0].shape[0]

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
                        # At the swizzle its ring is staged at (`_gathered_slots`), the one decision.
                        _row_copied(source, table, slot, swizzling.swizzle_of(ring), interpret)

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
                return step(index, rings.WholeSlots(tiles), started, accumulators)

            return lax.fori_loop(lo, hi, body, carry)

        carry = pl.run_scoped(walked, *scoped)
    return carry


def _gathered_slots(pipeline: "Pipeline", steps: int | None) -> tuple:
    """A gathered walk's ring of each stream (`_gathered_walk`, which scopes them, and `CpAsync.lifetimes`,
    which sizes them): a slot of each block a stage, no more than the walk's `steps` where they are known,
    staged at the swizzle its rows' copy tiles (`gathered.swizzled`)."""
    from dew.nn.kernels.templates import gathered as gathered_rows  # it builds on the toolkit's rings

    stages = pipeline.stages if steps is None else min(pipeline.stages, steps)
    return tuple(swizzling.staged((stages, *shape), dtype, gathered_rows.swizzled(shape, dtype))
                 for shape, dtype in pipeline.blocks.values())


def _specs(pipeline: "Pipeline", index_maps: dict | None = None) -> list:
    """Each stream's block spec as the walk hands `emit_pipeline` its tiles (`CpAsync.walk`), and as its
    allocations are read (`_pipelined`, at no `index_maps`, which an allocation does not read): a tile's
    blocks staged swizzled for ldmatrix (`_staging`), each read inside its operand: its launch extends every
    operand its rows read past (`bounded_loads`, `mosaic_gpu._walked_rows`), so its copies are promised in
    bounds, unchecked."""
    return [plgpu.BlockSpec(shape, None if index_maps is None else index_maps[name],
                            transforms=_staging((shape, dtype), pipeline.swizzles.get(name)),
                            oob_fill_mode=plgpu.OOBFillMode.PROMISE_IN_BOUNDS)
            for name, (shape, dtype) in pipeline.blocks.items()]


CP_ASYNC_GENERATION = "sm80"
"""The generation an mma.sync walk's shared memory is planned for (`_pipelined`): its copies are cp.async's
on every device the family runs on (`bundles.legal`: no device moving its blocks by TMA runs it), so one
plan holds for each; Ampere's, the least of them (`generation.BF16_GPU`)."""


@functools.lru_cache(maxsize=256)
def _pipelined(key: tuple, steps: int | None) -> tuple:
    """The shared memory `emit_pipeline` scopes for a walk of `steps` steps (its stages where they are not
    known) over the streams `key` names (`_pipeline_key`), as JAX's own builder allocates them
    (`get_allocations`) for the copies the walk makes (`CP_ASYNC_GENERATION`'s, a slot of each block a stage,
    fewer for fewer steps, no barrier), whatever GPU, mesh or trace the caller is in: traced on an abstract
    device of that generation, in a trace of its own that runs nothing, once a walk of the candidate
    search's."""
    stages, streams = key
    pipeline = Pipeline({name: (shape, jnp.dtype(dtype)) for name, shape, dtype, _ in streams}, stages,
                        {name: swizzle for name, _, _, swizzle in streams if swizzle is not None})
    found = {}

    def allocated() -> None:
        built = plgpu.emit_pipeline(lambda *_: None, grid=(stages if steps is None else steps,),
                                    in_specs=_specs(pipeline), max_concurrent_steps=stages)
        found.update(built.get_allocations(*(jax.ShapeDtypeStruct(shape, dtype)
                                             for shape, dtype in pipeline.blocks.values())))

    planned = generation.gpu_mesh(jax.sharding.AbstractMesh((1,), ("walk",)), CP_ASYNC_GENERATION)
    with meshless(), jax.sharding.use_abstract_mesh(planned):
        jax.make_jaxpr(allocated)()
    leaves = (*found["in_smem_refs"], *found["out_smem_refs"], found["barrier_ref"])
    return tuple(leaf for leaf in leaves if leaf is not None)


def _pipeline_key(pipeline: "Pipeline") -> tuple:
    """`pipeline`'s streams as `_pipelined` caches their allocations: its stages, and each stream's name, tile
    shape, dtype and named swizzle."""
    return pipeline.stages, tuple((name, tuple(shape), jnp.dtype(dtype).name, pipeline.swizzles.get(name))
                                  for name, (shape, dtype) in pipeline.blocks.items())


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
    name, the tiles `emit_pipeline` keeps in flight (`stages`) and the consumer warpgroups that run it
    together (`consumers`, its thread axis 'wg' where two); it allocates their slots itself."""

    blocks: dict
    stages: int
    swizzles: dict = dataclasses.field(default_factory=dict)
    """The swizzle a stream's tiles stage at where `Mma.streams` names one (`swizzling.swizzle_for`)."""
    gathered: tuple = ()
    """The streams whose step's rows a table gathers (a `Rows` view of one row a run): the warpgroup's own
    cp.async copies, a row a copy (`gathered.row_copies`), into slots of the walk's own, in the pipeline's
    cp.async groups (`_gathered_walk`); a walk gathers every stream or none."""
    consumers: int = 1
    """The consumer warpgroups that run its pipeline together (`emit_pipeline`'s thread axis 'wg' where
    two), each copying an equal share of every block and reading every slot whole."""


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

    row_bound: Any = None
    """The tile's rows its products issue before, an int32 (`bounded`): each warp's 16-row blocks at or past
    it issue none, its accumulator's rows kept; None issues every row. Its identity's: a family bounded at
    one constant is another program than at the next."""

    held_registers = 40
    """The registers a consumer thread holds beside its accumulator and operand fragments (addresses,
    indices, the walk's carry), a measured whole (`scalars_whole`): CUTLASS's cooperative sm90 kernel's 168
    less its accumulator's 128; an A100's attention dQ holds 13 to 43 integers live at its peaks, places and
    addresses together. Its derivations need it: without it the GEMM's gate picks on sm80, sm86, sm89 and
    sm120 spill 36 to 784 bytes, where its operands' fragments share the registers (FS-HELDREG). A
    pre-lowering prior, which certification overrules (`report.certified`): the count a lowering holds is
    not derived from the traced kernel, whose shared-memory tile addresses and loop carries predicted the
    lowered peak's scalars no better than a family's constant over 260 sm80 GEMM and recurrence kernels
    (FS-DERIVED-HELD2-sm80, row 8377: correlation 0.49 and 0.39; a recurrence's peak 4 to 97 scalars, two to
    five times its traced count, the lowering's shared-memory addresses and swizzles)."""

    scalars_whole = True

    def scalar_registers(self, addressed=()) -> int:
        # Its measured whole, its addresses among them.
        del addressed
        return self.held_registers

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
    def atom_columns(self) -> int:
        # A warp's atom `m_warps * 16` rows by `(4 // m_warps) * 8` columns (`specialized`): a 16-row
        # share's four warps along its columns tile 32 of them.
        return (4 // self.m_warps) * 8

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
        # its rows' 8-row tiles. Which slices of it stay live is `operand_live_depth`'s.
        return self._staged_depth(depth, swizzle) if transposed else self.fragment_depth(depth)

    def operand_live_depth(self, depth: int, loaded: int, *, retained: bool = False,
                           transposed: bool = False) -> int:
        # A transposed granule read once (not `retained`) that is its step's whole depth at two
        # instructions dies a slice at a time as its products issue: one instruction's live, an ordinary
        # one's lookahead (a transposed matrix's input gradient held as its forward is at a 32-deep step,
        # FS-5634-DX-sm80/-sm86). Deeper granules keep the rule below: a 64-deep step's input gradient ran
        # 24-40 registers past its forward, and a recurrence backward's 64-deep transposed reads compiled
        # spill-free at it (FS-5634-RECUR). A retained one is read again by a later product and keeps its
        # whole granule (`_query_key_held`).
        instruction = self.fragment_depth(depth)
        if transposed and not retained and loaded == depth == 2 * instruction:
            return instruction
        return super().operand_live_depth(depth, loaded, retained=retained, transposed=transposed)

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

    def left_of(self, ref) -> jax.Array:
        return self._left(ref)

    def _left(self, lhs):
        """A product's left operand in the MMA's layout: a shared-memory tile read, or a register value's (an
        accumulator's) cast, in registers where each warp holds whole rows of it or no relay tile is bound,
        else through the bound one (`relaying`): every warp's columns stored, then each warp's rows read
        whole, a sync after each so no warp reads a column not yet stored or stores over one another still
        reads."""
        if not isinstance(lhs, jax.Array):
            return plgpu.load(lhs, layout=self.lhs, optimized=self.rows_loaded)
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

    @property
    def rows_whole(self) -> bool:
        # Its warps along a tile's rows alone (four of them) hold whole rows of the accumulator and the
        # operand alike; fewer split the accumulator's columns among warps.
        return self.m_warps == 4

    def relay_bytes(self, rows: int, depth: int) -> int:
        # No register cast crosses the warps a row's columns are split among.
        return 0 if self.rows_whole else rows * depth * self.dtype.itemsize

    def relaying(self, tile) -> Self:
        # A plain tile (`swizzling.plain`).
        return dataclasses.replace(self, relay=tile)

    @property
    def _matrix_loads(self) -> bool:
        """Whether its operands load by `ldmatrix`, 16-bit rows of 8: not a 32-bit operand's (TF32's), for
        which Mosaic GPU's layout inference found no layout (tf32-mini at 9d77b39e7), its threads loading
        their elements themselves."""
        return self.dtype.itemsize <= 2

    @property
    def rows_loaded(self) -> bool:
        """Whether 16-bit left operands load by `ldmatrix`: only where each warp holds whole rows
        (`rows_whole`); fewer row warps read element-wise, as the relay path does."""
        return self._matrix_loads and self.rows_whole

    def _store_staged(self, value: jax.Array, buffer) -> None:
        cast = value.astype(buffer.dtype)
        if self.rows_whole or launches.interpreting():
            buffer[...] = cast
            return
        # Unoptimized store is not bracketed by automatic barriers like `buffer[...] =`: the sync before
        # keeps a warp from overwriting a column another still reads (WAR), and the sync after keeps a
        # subsequent read from preceding every warp's store (RAW).
        mosaic.synced(interpret=False)
        plgpu.store(buffer, cast, optimized=False)
        mosaic.synced(interpret=False)

    def product_with[T](self, lhs, rhs, shape: tuple[int, int], between: Callable[[], T], *,
                        pending: int = 0) -> tuple[T, jax.Array]:
        if pending:
            raise ValueError(f"{type(self).__name__}.product_with is synchronous (pending={pending})")
        out = self.products(self.zeros(shape), lhs, rhs)
        return between(), out

    def products(self, acc, lhs, rhs, *, accumulate: bool = True):
        interpret = launches.interpreting()  # the launch's (`launches.interpreting`)
        if not accumulate:
            # A fresh product: onto zeros, its accumulator a value (`read` is that value).
            acc = self.zeros(acc.shape)
        if jnp.dtype(lhs.dtype) != self.dtype:
            # Another dtype's operands (fp32's, as TF32) by that dtype's issuer: its fragments' layouts and
            # depths are its own, its accumulator's the same.
            return self.for_dtype(lhs.dtype).products(acc, lhs, rhs)
        if interpret:
            # mma has no rule on a CPU: its sum, in fp32, its rows past a bound kept as `_mma` keeps them.
            summed = acc + jnp.dot(lhs[...], rhs[...], preferred_element_type=jnp.float32)
            if self.row_bound is None or self.dtype == jnp.float32:
                return summed
            starts = lax.broadcasted_iota(jnp.int32, summed.shape, 0) // self.rows * self.rows
            issued = starts < self.row_bound
            return jnp.where(issued, summed, acc)
        acc, last = self._all_but_last(acc, *self._fragments(lhs, rhs)[:2])
        return self._mma(acc, *last)

    def _all_but_last(self, acc, loaded: Callable, starts: range) -> tuple[jax.Array, tuple]:
        """`acc` with every instruction's product of `starts` issued but the last, each instruction's
        fragments loaded (`loaded`) before the product ahead of it issues, and the last instruction's
        fragments."""
        fragments = loaded(starts[0])
        for following in starts[1:]:
            ahead = loaded(following)
            acc = self._mma(acc, *fragments)
            fragments = ahead
        return acc, fragments

    def _fragments(self, lhs, rhs, seeded: tuple | None = None) -> tuple[Callable, range, Callable]:
        """`products`' fragment reads of `lhs @ rhs`: `loaded(start)`, instruction `start`'s fragments, its
        loads issued the first time a slice is asked for; the instructions' starts; and `first()`, the left
        and right slices instruction 0 reads, read (`first_fragments`), those `seeded` gives already read
        (`products_across`). The left operand loads a whole swizzle tile of its depth at a time
        (`lhs_fragment_depth`), as a transposed left operand loaded in instruction slices took the A100's
        kernel gradient 7% slower (FS-DENSE-AB1); the right loads only the fragments the next instructions
        consume, as narrow as its staging allows (`_depth_granule`), which took its input gradient 12%
        faster. A register operand (an accumulator's value) is cast once and sliced. Each instruction's
        fragments load before the product ahead of it issues, CUTLASS's warp-level double buffering: every
        dense kernel gradient on an A100 took 12 to 14% less time, its outputs bitwise the same, the forward
        and input gradient unchanged (FS-C1-NARROW1-a100)."""
        instruction = self.fragment_depth(int(rhs.shape[0]))
        lhs_depth = max(_depth_granule(lhs, 1, instruction),
                        self.lhs_fragment_depth(int(lhs.shape[1]), swizzle=swizzling.swizzle_of(lhs)))
        rhs_depth = _depth_granule(rhs, 0, instruction)
        held = self._left(lhs) if isinstance(lhs, jax.Array) else None
        lefts: dict[int, jax.Array] = {}
        rights: dict[int, jax.Array] = {}
        if seeded is not None:
            left, rights[0] = seeded
            if left is not None:
                lefts[0] = left

        def loaded(start: int) -> tuple[jax.Array, jax.Array]:
            left_at, right_at = start - start % lhs_depth, start - start % rhs_depth
            if held is None and left_at not in lefts:
                lefts[left_at] = self._left(lhs.at[:, pl.ds(left_at, lhs_depth)])
            if right_at not in rights:
                rights[right_at] = plgpu.load(rhs.at[pl.ds(right_at, rhs_depth)], layout=self.rhs,
                                              optimized=self._matrix_loads)
            left = held[:, start:start + instruction] if held is not None else lefts[left_at][
                :, start - left_at:start - left_at + instruction]
            return left, rights[right_at][start - right_at:start - right_at + instruction]

        def first() -> tuple[jax.Array | None, jax.Array]:
            loaded(0)
            return lefts.get(0), rights[0]

        return loaded, range(0, rhs.shape[0], instruction), first

    def first_fragments(self, lhs, rhs):
        if jnp.dtype(lhs.dtype) != self.dtype:
            return self.for_dtype(lhs.dtype).first_fragments(lhs, rhs)
        return self._fragments(lhs, rhs)[2]()

    def products_across(self, acc, lhs, rhs, prefetched, boundary: Callable):
        if jnp.dtype(lhs.dtype) != self.dtype:
            return self.for_dtype(lhs.dtype).products_across(acc, lhs, rhs, prefetched, boundary)
        acc, last = self._all_but_last(acc, *self._fragments(lhs, rhs, seeded=prefetched)[:2])
        # The last product's operands are in registers: the next step's slots awaited and its first
        # fragments read now, under it.
        following = boundary()
        return self._mma(acc, *last), following

    def looped_products(self, acc, lhs, rhs, *, accumulate: bool = True):
        """`products`, its instruction slices issued in a loop of their own, each slice's operands loaded from
        shared memory within it: unrolled, ptxas interleaved a fused backward step's three later products
        (dV, dK, dQ, independent of each other) and held 170 words of their fragments at once, an A100's step
        at 64 x 64, width 64, under a softcap spilling 96 bytes; looped, 20 (one slice's), none spilled, the
        same 320 mma.sync instructions issued (row 5477). Interpreted, `products`."""
        if launches.interpreting():
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

    def bounded(self, rows) -> Self:
        # Each product it makes takes the bound: `plgpu.mma`'s `rows` (the fork's from 0.12.0.1.post125),
        # each warp's 16-row blocks at or past it issuing none (row 7091).
        return dataclasses.replace(self, row_bound=rows)

    def _mma(self, acc, left, right):
        """`acc + left @ right` by one mma.sync product of its operands' fragments: fp32's as TF32 through
        Mosaic GPU's lane-level `mma` (`inline_mgpu`; the fork's TF32 from 0.12.0.1.post42), the dialect's op
        whose operand types jaxlib fixes naming no fp32, every row issued; any other dtype's by Pallas's
        `mma`, within its row bound (`bounded`)."""
        if self.dtype != jnp.float32:
            return plgpu.mma(acc, left, right, m_warps=self.m_warps, rows=self.row_bound)
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
    """The execution policy before Hopper (`mosaic.Transport`): one or two compute warpgroups a program and
    no producer, its loads `emit_pipeline`'s cp.async copies into each walk's own slots (two consumers run
    one pipeline over their thread axis 'wg', each copying its share of every block and meeting the other
    before a slot is read and before it is refilled: FlashAttention 2's eight warps over one shared tile),
    its accumulators in the pipeline's carry (a register home), its output stored through registers. A
    family whose issuer half an issuer composed with it gives (`Mma`)."""

    walks_scoped = True  # `emit_pipeline` allocates each walk's slots in its own scope

    walks_streamed = True  # consumers of one collective pipeline walk every streamed step (`walk_own`)

    @property
    def shared_consumers(self) -> int:
        # Two consumer warpgroups run one collective pipeline at most, each copying its share of every block
        # (`emit_pipeline`'s thread_axis), as FlashAttention 2's eight warps share a step's queries over two
        # key halves: a launch that names two (the A100 backward's dK/dV, its dQ adds halved, row 5272), not
        # a derivation's own pick (`consumers`).
        return 2

    def compiled_with(self, *, approx_math: bool) -> plgpu.CompilerParams:
        # Its kernels' lane-level lowering (`kernel`'s), Mosaic GPU's default.
        return plgpu.CompilerParams(approx_math=approx_math)
    prefetches_fragments = True  # `emit_pipeline`'s `prefetch`: a step's first fragments under the last's

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
    # `emit_pipeline` would allocate a folded once block a slot per stage in the walk's scope, where its
    # planned once allocation (`lifetimes`) holds one: no once stream walks here until the plan counts it.
    walks_once = False
    gathers_alone = False  # its walk gathers every stream's rows or none (`streams`)
    gathers_interpreted = True

    copies_in_pieces = False  # its pipeline loads a step's block whole, or gathers its rows

    adds_vectors = False  # its atomic adds take no layout replicating a vector's entries (a [bk] key sum's)

    adds_through_shared = False  # `added`'s atomic adds go from registers (`red.global`), its `smem` unread
    stages_vectors = False

    @property
    @abc.abstractmethod
    def lhs(self) -> plgpu_core.SomeLayout:
        """The layout a tile loaded through registers takes (`loaded`): its issuer's left operand's."""

    def registers(self, consumers: int, programs: int = 1, producer: int | None = None,
                  joined: tuple = ()) -> int:
        # Each role at its thread's share of the multiprocessor's 64K registers: two programs' 256 threads
        # hold 255 each.
        _most(self, consumers)
        handoffs.runs_alone(self.name, joined)
        return resources.entry_registers(consumers, programs)

    def kernel(self, entry, *, out_type, grid: tuple, grid_names: tuple, consumers: int,
               interpret: bool, approx_math: bool = True, name: str | None = None,
               cluster: int = 1, cost: pl.CostEstimate | Stated | None = None,
               model_flops: int | None = None,
               reducing_warpgroups: int = 0, resident: int = 1, scalars: int = 0,
               prefetched: Callable | None = None, registers: int | None = None, reduces: bool = True,
               producer: int | None = None, shared: int | None = None, tmem: int = 0,
               moves: bool | None = None, joined: tuple = ()) -> Callable:
        handoffs.runs_alone(self.name, joined)
        # Its warpgroups its consumers (two a thread axis 'wg'), no producer beside them, and no cluster;
        # its kernels' default parameters reserve the scratch an implicit reduction takes
        # (`reduction_scratch`), where it `reduces`. Its programs read a table from global memory as any
        # operand.
        del reducing_warpgroups, scalars, producer, tmem, moves  # consumers alone: no roles to move registers
        _most(self, consumers, clustered=cluster > 1)
        threads = {} if consumers == 1 else {"num_threads": consumers, "thread_name": "wg"}
        built = launches.kernel(bodies.dependent(entry, interpret, prefetched=prefetched),
                                name=name or self.name,
                                out_type=outputs.structs(out_type), grid=grid, grid_names=grid_names,
                                **threads,
                                compiler_params=dataclasses.replace(
                                    self.compiled_with(approx_math=approx_math),
                                    reduction_scratch_bytes=self.reduction_scratch if reduces else 0,
                                    min_blocks_per_sm=resident, metadata=report.launch_metadata()),
                                interpret=bodies.interpreted(interpret),
                                kernel_name=report.named(
                                    name, model=(None if registers is None
                                                 else resources.register_model(
                                                     self, consumers, registers, shared=shared,
                                                     target=generation.gpu_target()))),
                                cost_estimate=emission.built_with(cost))
        return mosaic.checked(emission.costed(built, cost), name or self.name, resident, model_flops,
                              out_type=out_type)

    def program_scoped(self, body, *allocations, streams):
        # Collective where any of its pipelines (`streams`, one or a tree of them) runs on two consumers.
        return pl.run_scoped(lambda *refs: body(*refs, streams=streams), *allocations,
                             **_collective(_consumers(streams)))

    def roles(self, consumers: int, producer: int | None = None) -> handoffs.Schedule:
        _most(self, consumers)
        if consumers == 1:
            return handoffs.Schedule((handoffs.Role("consumer"),), axis=None)
        return handoffs.Schedule((handoffs.Role("consumer", consumers),))

    def streams(self, blocks: dict, stages: int, consumers: int, multicast: tuple[str, ...] = (),
                swizzles: dict[str, int] | None = None, copies: dict[str, int] | None = None,
                gathered: tuple[str, ...] = (), joined: tuple[str, ...] = ()):
        del joined  # a pair a stream (`Family.streams`)
        _most(self, consumers, clustered=bool(multicast))
        if gathered and set(gathered) != set(blocks):
            raise NotImplementedError("an A100's walk gathers every stream's rows or none")
        if gathered and consumers > 1:
            raise NotImplementedError("an A100's gathered walk copies by one consumer's rows")
        if any(count > 1 for count in (copies or {}).values()):
            raise ValueError("an A100's pipeline loads a step's block in one copy, not in pieces")
        return Pipeline(blocks, stages, dict(swizzles or {}), tuple(gathered), consumers)


    def lifetimes(self, streams: Pipeline, steps: int | None = None) -> tuple[tuple, tuple]:
        # A walk's shared memory in its own scope: `emit_pipeline`'s, as JAX's builder allocates it for its
        # cp.async copies (`_pipelined`), or a gathered walk's own rings (`_gathered_slots`); none for a walk
        # of no steps, which the walk skips.
        if steps == 0:
            return (), ((),)
        if streams.gathered:
            return (), (_gathered_slots(streams, steps),)
        # Keyed by its slots, `steps` only where fewer than its stages (`emit_pipeline` allocates
        # min(stages, steps)): a call's every depth past them plans one walk, not one each (row 11639).
        # Two consumers running it together meet at two barriers in turn (its thread axis's).
        slots = steps if steps is not None and steps < streams.stages else None
        meet = (() if streams.consumers == 1
                else (plgpu.Barrier(num_arrivals=streams.consumers, num_barriers=2),))
        return (), ((*_pipelined(_pipeline_key(streams), slots), *meet),)

    def opened(self, allocated, views: dict | None = None):
        if views is None:
            raise ValueError("a pipelined family's streams take their views where they open (its walk loads)")
        return Streams(allocated, views)

    def stored_swizzle(self, shape: tuple[int, int], dtype) -> int:
        """The swizzle an output tile of `shape` in `dtype` stages at (`output_staging`): plain where its
        issuer states none (`Mma.stored_swizzle`)."""
        del shape, dtype
        return swizzling.UNSWIZZLED

    def stream(self, streams, views, lo, hi, *, again=None) -> None:
        # No producer role: a decided walk's consumers feed their second pass themselves.
        del streams, views, lo, hi, again

    def persist(self, streams, tiles: tuple[int, ...], schedule: Callable, issue: Callable, finish: Callable,
                *, consumers: int, axis: str, outputs: tuple, interpret: bool, in_flight: int = 0,
                unrolled: bool = False, accumulators: tuple = (), producer: int | None = None,
                pingpong: bool = False, chunks: tuple[int, int] | None = None) -> None:
        del producer  # no producer role of its own
        self.pingponged(pingpong)
        _most(self, consumers)
        # Each tile's pipeline fills and drains before its output copy. Synchronize at that lifetime
        # boundary so the output's shared memory can reuse the pipeline's slots; two consumers stage their
        # rows' share of it side by side, in one collective scope.
        del in_flight, unrolled, accumulators  # its walk carries its accumulators
        index = 0 if consumers == 1 else lax.axis_index("wg")
        staging = self.output_staging(outputs, consumers, chunks)

        def store(output, value, ref, place):
            # Interpreted pipelines use barrier-tracked copies, not cp.async groups.
            if not interpret:
                plgpu.wait_gmem_to_smem(0)

            def copied(staged):
                smem = staged if consumers == 1 else staged.at[index]
                plgpu.store(smem, value.astype(smem.dtype), optimized=False)
                swizzle = swizzling.UNSWIZZLED if len(smem.shape) != 2 else self.stored_swizzle(smem.shape,
                                                                                                   smem.dtype)
                if interpret or swizzle == swizzling.UNSWIZZLED:
                    # The row-copy ladder needs strided layouts, not optimized-only transfers.
                    stores.store_rows(smem, ref, *place, copy=swizzling.copied,
                                      widest=resources.copied_rows(smem.shape[1], smem.dtype))
                else:
                    # A swizzled tile's rows out by its threads, whole tiles read (`output_staging`).
                    stores.swizzled_rows_out(smem, ref, *place[:4], swizzle=swizzle)

            pl.run_scoped(copied, staging[output], **_collective(consumers))
            if not interpret:
                plgpu.wait_gmem_to_smem(0)

        def walking(counter, active, steps, views):
            del counter, active

            def walk(step, start, shapes, *, initial=None, prefetch=None):
                return self.walk(self.opened(streams, views), ((step, 0, steps),), start, shapes,
                                 interpret, initial=initial, prefetch=prefetch)

            return walk

        persistent.consumed(self, tiles, schedule, issue, finish, store, index, walking, axis=axis)

    def walk[C](self, streams, segments, start: C, shapes: tuple, interpret: bool, *,
                first: jax.Array | int = 0, in_flight: int = 0, initial: tuple | None = None,
                unrolled: bool = False, products: Sequence[mosaic.Product] = (),
                prefetch: Callable | None = None) -> tuple[C, tuple[jax.Array, ...]]:
        # Its products are synchronous: none stays in flight past its step, and `emit_pipeline` gives
        # each stage back as its step ends, so no step holds one past it, nor reads one ahead. With
        # `prefetch(slots)` (`prefetches_fragments`), `emit_pipeline` reads a step's first fragments under the
        # step before's last product: each step is given them and the boundary that reads the next's
        # (`step(..., prefetched=, boundary=)`, returning its carry and the next's fragments), where it keeps
        # two slots or more (its prefetch's own refill) and gathers no rows (`_gathered_walk`'s own copies);
        # elsewhere each step is called as without it.
        if mosaic.ahead_streams(products):
            raise NotImplementedError(f"{self.name}'s pipeline reads no stream ahead")
        del first, in_flight, unrolled
        segments = mosaic.stepped(segments)
        blocks = streams.pipeline.blocks
        names = tuple(blocks)
        carry = (start, self.started(shapes, initial))
        if streams.pipeline.gathered:
            return _gathered_walk(streams, segments, carry, interpret)
        prefetching = prefetch is not None and streams.pipeline.stages > 1

        def prefetched_of(indices, *slots):
            assert prefetch is not None
            del indices
            return prefetch(slots_of(slots))

        def slots_of(refs) -> rings.WholeSlots:
            return rings.WholeSlots(dict(zip(names, refs, strict=True)))

        for step, lo, hi in segments:
            if isinstance(lo, int) and isinstance(hi, int) and hi <= lo:
                # No steps: the carry and its zero accumulators as they start (`emit_pipeline` takes no
                # empty grid).
                continue

            def body(indices, *refs, step=step, lo=lo):
                (index,) = indices
                if prefetching:
                    *slots, (carried, accumulators), prefetched, boundary = refs
                    (carried, accumulators), following = step(
                        index + lo, slots_of(slots), carried, accumulators, prefetched=prefetched,
                        boundary=boundary)
                    return (carried, accumulators), following
                carried, accumulators = refs[-1]
                return step(index + lo, slots_of(refs[:-1]), carried, accumulators)

            specs = _specs(streams.pipeline,
                           {name: functools.partial(_at, streams.views[name][1], lo) for name in names})
            # Two consumers run one pipeline, each copying its share of every block (`thread_axis`).
            pipeline = plgpu.emit_pipeline(
                body, grid=(hi - lo,), in_specs=specs, max_concurrent_steps=streams.pipeline.stages,
                init_carry=carry, prefetch=prefetched_of if prefetching else None,
                thread_axis=None if streams.pipeline.consumers == 1 else "wg")
            carry = pipeline(*(streams.views[name][0] for name in names))
        return carry

    def passed(self, streams, lo, hi, *, first: jax.Array | int = 0, between=None) -> None:
        del streams, lo, hi, first, between

    def walk_own[R](self, streams, views: dict, own: tuple, streamed: tuple, wg,
                    walk: Callable[..., R], *, between=None, counted=0) -> R:
        # Two consumers run one collective pipeline (`thread_axis`), whose meets pair their steps: each
        # walks every streamed step, its step masking those outside its own range, as a consumer-fed
        # family's do (`ConsumerFed.walk_own`; a causal attention program's two consumers reach different
        # diagonals, and walks of their own ranges would meet unpaired). One consumer walks its own.
        if streams.pipeline.consumers == 1:
            return super().walk_own(streams, views, own, streamed, wg, walk, between=between, counted=counted)
        del views, own, wg, between, counted
        lo, hi = streamed
        return walk(streams, lo, hi, lo)

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
            if smem is None or len(smem.shape) == 1:
                if isinstance(part, mosaic.StagedWriter):
                    part.write(None)
                elif len(view.shape) == 1 and not launches.interpreting():
                    plgpu.store(view, part.astype(view.dtype), optimized=False)
                else:
                    view[...] = part.astype(view.dtype)
                continue
            if isinstance(part, mosaic.StagedWriter):
                part.write(smem)
            else:
                self.stage_output(part, smem)
            rows = smem.shape[0]
            # A landing wider than its output rounds each row block as it copies it (`swizzling.narrowed`).
            wider = jnp.dtype(smem.dtype).itemsize > jnp.dtype(view.dtype).itemsize
            stores.store_rows(smem, view, 0, 0, jnp.int32(rows), 0, rows,
                              copy=swizzling.narrowed if wider else swizzling.coalesced,
                              widest=resources.copied_rows(smem.shape[1], smem.dtype))

    def stage_output(self, value: jax.Array, smem) -> None:
        if launches.interpreting():
            smem[...] = value.astype(smem.dtype)
        else:
            plgpu.store(smem, value.astype(smem.dtype), optimized=False)

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

    # One consumer warpgroup a program where a derivation picks its own (`consumers`, what every derivation
    # reads): two hung on an A100 at the attention forward (FS-C2HANG-a100, M-ET-15) and spilled where the
    # forward derived them. A launch asks for two by name (`shared_consumers`).
    consumers = 1

    def output_staging(self, outputs: tuple, consumers: int, chunks: tuple[int, int] | None = None) -> list:
        mosaic.refuse_staging(self, chunks)
        # Each output tile staged at the swizzle its accumulator's stores pass the banks fewest times at
        # (`stored_swizzle`), copied out a 16-byte piece a thread over whole tiles (`persist`'s store,
        # `stores.swizzled_rows_out`); two consumers' rows of it side by side (`persist`).
        side = () if consumers == 1 else (consumers,)
        return [swizzling.staged((*side, *shape), dtype, self.stored_swizzle(shape, dtype)) if len(shape) == 2
                else plgpu.SMEM((*side, *shape), dtype) for shape, dtype in outputs]

    def stored_swizzle(self, shape: tuple[int, int], dtype) -> int:
        """The swizzle an output tile of `shape` in `dtype` stages at: of the swizzles its rows fill whole
        (`swizzling.SWIZZLE_BYTES` down to the narrowest) and plain, the one its accumulator's stores pass
        shared memory's banks fewest times at (`swizzling.wavefronts`, by the staging's address arithmetic),
        the widest on a tie. A warp stores mma.sync's accumulator fragment an 8-row slab by 8 columns at a
        time, lane `l` two adjacent elements of row `l // 4` at column `2 (l % 4)`: plain, 64 16-bit columns
        put a slab's 8 rows in the same banks, 8 passes a store, where a 128-byte swizzle gives each row's
        piece its own banks, one pass (row 5271: the A100's dense kernel gradient's 4x cuBLAS's bank
        conflicts, FS-DW-NCU2)."""
        rows, columns = shape
        itemsize = jnp.dtype(dtype).itemsize
        stagings = [swizzle for swizzle in (swizzling.SWIZZLE_BYTES, 64, swizzling.NARROWEST_SWIZZLE_BYTES)
                    if (columns * itemsize) % swizzle == 0 and rows % swizzling.TILE_ROWS == 0]
        stagings.append(swizzling.UNSWIZZLED)

        def passes(swizzle: int) -> int:
            return sum(swizzling.wavefronts(
                [swizzling.staged_address(slab + lane // 4, (block + 2 * (lane % 4)) * itemsize,
                                          columns * itemsize, swizzle) for lane in range(32)], 2 * itemsize)
                for slab in range(0, rows, swizzling.TILE_ROWS) for block in range(0, columns, 8))

        return min(stagings, key=lambda swizzle: (passes(swizzle), -swizzle))


def _most(family: "CpAsync", consumers: int, *, clustered: bool = False) -> None:
    """A cp.async family's programs take one compute warpgroup up to its collective pipeline's
    (`shared_consumers`), no producer, and no cluster (`clustered`: cp.async loads its own program's shared
    memory alone)."""
    if consumers not in range(1, family.shared_consumers + 1) or clustered:
        raise NotImplementedError(f"an {family.name} program takes one to {family.shared_consumers} consumer "
                                  f"warpgroups, not {consumers}, and no cluster")


def _consumers(streams) -> int:
    """The most consumers any of `streams`' pipelines (one or a tree of them) runs on."""
    pipelines = jax.tree.leaves(streams, is_leaf=lambda leaf: isinstance(leaf, Pipeline))
    return max((pipeline.consumers for pipeline in pipelines if isinstance(pipeline, Pipeline)), default=1)


def _collective(consumers: int) -> dict:
    """A scope's allocations collective over the consumers' thread axis where two run (a threaded kernel's
    scoped allocations are its warpgroups' together)."""
    return {} if consumers == 1 else {"collective_axes": "wg"}

