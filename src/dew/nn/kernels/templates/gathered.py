"""Rows where a table says (row 1376): a gathered ring (`Ring`), a stream whose rows each lie where a table
says, a mixture of experts' tokens read in their own order, loaded by the producer warpgroup's cp.async
copies (Hopper's) or by TMA's four-row gathers (`Committed`, tcgen05's load warp), and a scattered store
(`scattered_rows`), an output's rows each written where a permutation says, each through `inline_mgpu`'s
lane-level code (Pallas's warpgroup semantics reach no thread's own copy, address or mbarrier)."""

import dataclasses
import functools

import jax.numpy as jnp
import numpy as np
from jax._src.lib.mlir.dialects import gpu as mgpu_gpu, llvm as mgpu_llvm, nvvm as mgpu_nvvm, scf as mgpu_scf
from jax.experimental import pallas as pl
from jax.experimental.mosaic.gpu import (
    fragmented_array as mgpu_fragments,
    launch_context as mgpu_launch,
    utils as mgpu_utils,
)
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend.mlir import ir as mgpu_ir
from jax.extend.mlir.dialects import arith as mgpu_arith, memref as mgpu_memref, vector as mgpu_vector

from dew.nn.kernels.templates import mosaic


@dataclasses.dataclass(frozen=True)
class Ring(mosaic.Ring):
    """A `mosaic.Ring` of rows each read where a table says (`mosaic.Rows` of one row a run), loaded by the
    producer warpgroup's cp.async copies: each thread copies its 16-byte pieces of the slot's rows from the
    rows the table names (`gathered_copies`), tracked at the slot's load barrier, an mbarrier of shared
    memory (`ring`: Pallas's barriers take no lane-level arrival); the consumers wait on it (`acquire`) and
    release the slot as any ring's. TMA lands a copy in whole tiles of rows, a row alone only by copies
    Mosaic's layout inference refuses, and from one thread a row a copy (about 5k clocks a step against 1k
    of products); CUTLASS's sm90 gather is the producer warps' cp.async too."""

    def initialized(self) -> None:
        stages = self.stages
        plgpu.inline_mgpu(arg_types=(plgpu.RefType(),))(
            lambda ctx, barriers: barriers_initialized(barriers, stages))(self.full)

    def viewed(self, view: tuple, step):
        ref, index_map = view
        return ref, index_map(step)

    def load(self, turn: mosaic.Turn, view) -> None:
        slot = mosaic.opaque(turn.slot)
        pl.when(turn.counter >= self.stages)(lambda: plgpu.barrier_wait(
            self.consumed.at[slot], parity=turn.freed))
        tokens, rows = view
        height, width = self.block
        [depth] = rows.rest
        source = tokens.at[:, pl.ds(depth * width, width)]
        bytes_ = swizzled(self.block, tokens.dtype)
        swizzle = mosaic.swizzle_transforms(tokens.dtype, bytes_)
        copies = functools.partial(gathered_copies, swizzle=bytes_)
        refs = (plgpu.RefType(), plgpu.RefType(), plgpu.RefType(swizzle), plgpu.RefType())
        plgpu.inline_mgpu(arg_types=refs)(lambda ctx, *refs: copies(*refs))(
            source, rows.table.at[pl.ds(rows.start, height)], self.smem.at[slot],
            self.full.at[pl.ds(slot, 1)])

    def acquire(self, turn: mosaic.Turn):
        slot = mosaic.opaque(turn.slot)
        plgpu.inline_mgpu(arg_types=(plgpu.RefType(), plgpu.Layout.WG_SPLAT(())))(
            lambda ctx, barrier, parity: barrier_of(barrier).wait_parity(splat_value(parity)))(
                self.full.at[pl.ds(slot, 1)], jnp.asarray(turn.parity, jnp.bool_))
        return self.smem.at[slot]


def ring(block: tuple, dtype, stages: int, *, consumers: int = 1, released_by_products: bool = False) -> dict:
    """A gathered ring's allocations for `pl.run_scoped`: `stages` slots of `block` in `dtype`, their rows
    staged at the widest swizzle that fits (`swizzled`), a load mbarrier each in shared memory (`uint64`,
    `Ring.initialized`) and a release barrier each that `consumers` arrive at, or, `released_by_products`,
    that the products reading a slot arrive at once by their commit (`Committed`'s: tcgen05's, as
    `mosaic.ring`'s)."""
    consumed = (plgpu.Barrier(num_barriers=stages, orders_tensor_core=True) if released_by_products
                else plgpu.Barrier(num_arrivals=consumers, num_barriers=stages))
    return {'smem': mosaic.staged((stages, *block), dtype, swizzled(block, dtype)),
            'full': plgpu.SMEM((stages,), jnp.uint64), 'consumed': consumed}


def swizzled(block: tuple, dtype) -> int:
    """The swizzle a gathered ring's slots of `block` stage at: the widest whose row its columns hold
    whole (`mosaic.swizzle_for`), as layout inference stages an ungathered one: 128 bytes but a narrow
    block's (a 32-column bf16 cotangent of mma.sync's kernel gradient, 64)."""
    return mosaic.swizzle_for(dtype, block[1])


@dataclasses.dataclass(frozen=True)
class Committed(Ring):
    """A gathered ring tcgen05's load warp fills and its products give back (`mosaic_sm100.CommittedRing`'s
    release, their commit arriving at the slot's): each step's rows by TMA's four-row gathers (`tma_rows`,
    `cp.async.bulk.tensor.2d...tile::gather4`, sm100's), one thread issuing them all and arriving once at the
    slot's load mbarrier, expecting their bytes, which the copies complete. A row alone by tiled TMA is a
    copy of its own (an 8-row tile's slice no swizzled box takes); a gather moves four."""

    def load(self, turn: mosaic.Turn, view) -> None:
        slot = mosaic.opaque(turn.slot)
        pl.when(turn.counter >= self.stages)(lambda: plgpu.barrier_wait(
            self.consumed.at[slot], parity=turn.freed))
        tokens, rows = view
        height, width = self.block
        [depth] = rows.rest
        bytes_ = swizzled(self.block, tokens.dtype)
        swizzle = mosaic.swizzle_transforms(tokens.dtype, bytes_)
        refs = (plgpu.RefType(), plgpu.RefType(), plgpu.RefType(swizzle), plgpu.RefType(),
                plgpu.Layout.WG_SPLAT(()))
        copies = functools.partial(tma_rows, swizzle=bytes_)
        plgpu.inline_mgpu(arg_types=refs)(lambda ctx, *refs: copies(ctx, *refs))(
            tokens, rows.table.at[pl.ds(rows.start, height)], self.smem.at[slot],
            self.full.at[pl.ds(slot, 1)], jnp.asarray(depth * width, jnp.int32))

    def release(self, turn: mosaic.Turn) -> None:
        plgpu.tcgen05_commit_arrive(self.consumed.at[mosaic.opaque(turn.slot)])


def barrier_of(memref) -> mgpu_utils.BarrierRef:
    """The mbarrier at a one-entry shared memref of `uint64` (`Ring`'s), as Mosaic GPU's
    lane-level barrier: its arrivals and parity waits, no phases tracked."""
    i32 = mgpu_ir.IntegerType.get_signless(32)
    # No phases tracked: its waits name their parity (`wait_parity`), so the phases word is never read.
    zero = mgpu_utils.c(0, i32)
    return mgpu_utils.BarrierRef(mgpu_utils.memref_ptr(memref), zero, zero, 1)


def splat_value(array):
    """A splat scalar's value (a `WG_SPLAT` argument of `inline_mgpu`)."""
    [value] = array.registers.flat
    return value


def barriers_initialized(barriers, stages: int) -> None:
    """Each of `stages` mbarriers of `barriers` (shared `uint64`) initialized to one arrival, by one thread of
    the block, the block meeting after so none is waited on or arrived at first."""
    i32, i64 = mgpu_ir.IntegerType.get_signless(32), mgpu_ir.IntegerType.get_signless(64)
    base = mgpu_utils.memref_ptr(barriers)
    predicate = mgpu_utils.single_thread_predicate(scope=mgpu_utils.ThreadSubset.BLOCK)
    for stage in range(stages):
        mgpu_nvvm.mbarrier_init(mgpu_utils.getelementptr(base, [stage], i64), mgpu_utils.c(1, i32),
                                predicate=predicate)
    mgpu_gpu.barrier()


def gathered_copies(source, rows, slot, barrier, *, swizzle: int) -> None:
    """The producer warpgroup's cp.async copies of the slot's rows (`Ring.load`): row `r` of
    `slot` (`[height, width]`, 8-row tiles at `swizzle` bytes) row `rows[r]` of `source` (`[tokens, width]`,
    the step's columns of the tokens), each thread its 16-byte pieces in the tiled copy's layout
    (`tiled_copy_smem_gmem_layout`, as Mosaic GPU's own cp.async copy of a tile places them), then the
    copies tracked at `barrier` and the warpgroup's one arrival there once every thread has issued its own."""
    i32, i64 = mgpu_ir.IntegerType.get_signless(32), mgpu_ir.IntegerType.get_signless(64)
    index = mgpu_ir.IndexType.get()
    element = mgpu_ir.MemRefType(source.type).element_type
    bits = mgpu_utils.bitwidth(element)
    shape = tuple(mgpu_ir.MemRefType(slot.type).shape)
    height, width = shape[0] * shape[2], shape[1] * shape[3]
    layout = mgpu_fragments.tiled_copy_smem_gmem_layout(shape[0], shape[1], swizzle, bits)
    [row_stride, column_stride], _ = mgpu_ir.MemRefType(source.type).get_strides_and_offset()
    rank = 2

    def thread_part(strides):
        tiled = [mgpu_arith.constant(i32, stride) for stride in layout.tiling.tile_strides(strides)[rank:]]
        return mgpu_arith.addi(mgpu_utils.dyn_dot(layout.lane_indices(), tiled),
                               mgpu_utils.dyn_dot(layout.warp_indices(), tiled))

    thread_row, thread_column = thread_part((1, 0)), thread_part((0, 1))
    base = mgpu_llvm.addrspacecast(mgpu_llvm.PointerType.get(address_space=1),
                                   mgpu_utils.memref_ptr(source))
    gep = mgpu_ir.IntegerType.get_signless(8) if bits == 8 else element
    width_bytes = layout.vector_length * bits // 8
    for _, _, base_index, smem_ptr in mgpu_fragments.FragmentedArray.transfer_tiled(
            slot, swizzle, layout, (height, width), optimized=False):
        row_constant, column_constant = base_index()
        row = mgpu_arith.addi(thread_row, mgpu_arith.constant(i32, row_constant))
        token = mgpu_arith.extsi(i64, mgpu_memref.load(rows, [mgpu_arith.index_cast(index, row)]))
        column = mgpu_arith.extui(i64, mgpu_arith.addi(thread_column,
                                                       mgpu_arith.constant(i32, column_constant)))
        offset = mgpu_arith.addi(mgpu_arith.muli(token, mgpu_utils.c(row_stride, i64)),
                                 mgpu_arith.muli(column, mgpu_utils.c(column_stride, i64)))
        mgpu_nvvm.cp_async_shared_global(smem_ptr, mgpu_utils.getelementptr(base, [offset], gep), width_bytes,
                                         mgpu_nvvm.LoadCacheModifierKind.CG)
    tracked = barrier_of(barrier)
    # cp.async.mbarrier.arrive is no arrival: it holds the phase open until this thread's copies land.
    mgpu_nvvm.cp_async_mbarrier_arrive(tracked.get_ptr())
    mgpu_utils.warpgroup_barrier()
    tracked.arrive(predicate=mgpu_utils.single_thread_predicate(mgpu_utils.ThreadSubset.WARPGROUP))


def tma_rows(ctx, source, rows, slot, barrier, column, *, swizzle: int) -> None:
    """`Committed.load`'s copies: row `r` of `slot` (`[height, width]`, 8-row tiles at `swizzle` bytes)
    row `rows[r]` of `source` (the tokens, whole: TMA's descriptor is of a kernel argument) from `column`,
    each 8-row tile's rows by two four-row gathers (`LaunchContext.async_copy`'s gather, its indices
    `TMA_INDICES_4_LAYOUT`: every thread holds them, the warpgroup's first thread issues: the load warp's,
    `mosaic_sm100.LOAD_WARP`), after that thread's one arrival at `barrier` expecting every row's bytes."""
    i32 = mgpu_ir.IntegerType.get_signless(32)
    index = mgpu_ir.IndexType.get()
    shape = tuple(mgpu_ir.MemRefType(slot.type).shape)
    tile_rows, tile_columns = shape[2], shape[3]
    width = shape[1] * tile_columns
    element = mgpu_ir.MemRefType(source.type).element_type
    tracked = barrier_of(barrier)
    total = shape[0] * tile_rows * width * mgpu_utils.bitwidth(element) // 8
    mgpu_nvvm.mbarrier_arrive_expect_tx(tracked.get_ptr(), mgpu_utils.c(total, i32),
                                        predicate=mgpu_utils.single_thread_predicate(
                                            mgpu_utils.ThreadSubset.WARPGROUP))
    layout = mgpu_fragments.TMA_INDICES_4_LAYOUT
    vector = mgpu_ir.VectorType.get((4,), i32)
    start = splat_value(column)
    for tile in range(shape[0]):
        # The tile's entries, four a register, every thread the same (`layout`), each a load of its own: a
        # tile's first entry lies where its group's rows start, at no 16-byte alignment.
        registers = [mgpu_vector.from_elements(vector, [
            mgpu_memref.load(rows, [mgpu_utils.c(tile * tile_rows + 4 * part + lane, index)])
            for lane in range(4)]) for part in range(tile_rows // 4)]
        indices = mgpu_fragments.FragmentedArray(
            _registers=_registered(registers, layout, tile_rows), _layout=layout, _is_signed=False)
        ctx.async_copy(src_ref=source, dst_ref=mgpu_utils.memref_slice(slot, mgpu_utils.ds(tile, 1)),
                       gmem_slice=(indices, mgpu_utils.ds(mgpu_arith.index_cast(index, start), width)),
                       gmem_transform=mgpu_launch.TileTransform((tile_rows, tile_columns)), swizzle=swizzle,
                       barrier=tracked, arrive=False)


def _registered(registers: list, layout, rows: int):
    """`registers` (each four entries) as a `FragmentedArray`'s of `rows` entries in `layout`."""
    held = np.empty(layout.registers_shape((rows,)), dtype=object)
    for at, register in zip(np.ndindex(held.shape), registers, strict=True):
        held[at] = register
    return held


def scattered_rows(smem, out: mosaic.Scattered, row_start, first, stored, column) -> None:
    """Rows `first` to `first + stored` of `smem` (`[rows, width]`, unswizzled: a consumer's staged share of
    a tile, `mosaic.staged_rows`) each written to its row of `out` (`out.table[row_start + r]`, a negative
    entry nowhere) at `column`: by the warpgroup's threads, each a 16-byte piece of a row at a time
    (`_scattered_pieces`), or interpreted, a row at a time."""
    rows, width = smem.shape
    table = out.table.at[pl.ds(row_start, rows)]
    dest = out.ref.at[:, pl.ds(column, width)]
    if out.interpreted:
        @pl.loop(first, first + stored)
        def _(row):
            token = table[row]

            @pl.when(token >= 0)
            def _():
                dest[pl.ds(token, 1), :] = smem[pl.ds(row, 1), :]
        return
    scalar = plgpu.Layout.WG_SPLAT(())
    def pieces(ctx, smem_ref, table_ref, dest_ref, first_value, stored_value):
        del ctx
        _scattered_pieces(smem_ref, table_ref, dest_ref, splat_value(first_value), splat_value(stored_value))

    plgpu.inline_mgpu(arg_types=(plgpu.RefType(), plgpu.RefType(), plgpu.RefType(), scalar, scalar))(pieces)(
        smem, table, dest, jnp.asarray(first, jnp.int32), jnp.asarray(stored, jnp.int32))


def _scattered_pieces(smem, table, dest, first, stored) -> None:
    """`scattered_rows`' lane-level copy: the warpgroup's threads met (the share staged by all of them),
    then each of the share's 16-byte pieces, thread `t` taking pieces `t`, `t + 128`, ..., read from shared
    memory and written to its row's place in `dest` where its row is stored (`first` <= r < `first +
    stored`, its table entry not negative), the threads met again before the buffer is written."""
    i32, i64 = mgpu_ir.IntegerType.get_signless(32), mgpu_ir.IntegerType.get_signless(64)
    index = mgpu_ir.IndexType.get()
    element = mgpu_ir.MemRefType(smem.type).element_type
    rows, width = mgpu_ir.MemRefType(smem.type).shape
    lanes = 16 * 8 // mgpu_utils.bitwidth(element)
    per_row = width // lanes
    pieces = rows * per_row
    [row_stride, column_stride], _ = mgpu_ir.MemRefType(dest.type).get_strides_and_offset()
    vector = mgpu_ir.VectorType.get((4,), i32)
    source = mgpu_utils.memref_ptr(smem)
    base = mgpu_llvm.addrspacecast(mgpu_llvm.PointerType.get(address_space=1), mgpu_utils.memref_ptr(dest))
    thread = mgpu_arith.remui(mgpu_arith.index_cast(i32, mgpu_gpu.thread_id(mgpu_gpu.Dimension.x)),
                              mgpu_utils.c(128, i32))
    mgpu_utils.warpgroup_barrier()
    end = mgpu_arith.addi(first, stored)
    for turn in range(-(-pieces // 128)):
        piece = mgpu_arith.addi(thread, mgpu_utils.c(128 * turn, i32))
        row = mgpu_arith.divui(piece, mgpu_utils.c(per_row, i32))
        lane = mgpu_arith.remui(piece, mgpu_utils.c(per_row, i32))
        inside = mgpu_arith.andi(
            mgpu_arith.cmpi(mgpu_arith.CmpIPredicate.slt, piece, mgpu_utils.c(pieces, i32)),
            mgpu_arith.andi(mgpu_arith.cmpi(mgpu_arith.CmpIPredicate.sge, row, first),
                            mgpu_arith.cmpi(mgpu_arith.CmpIPredicate.slt, row, end)))
        branch = mgpu_scf.IfOp(inside)
        with mgpu_ir.InsertionPoint(branch.then_block):
            token = mgpu_memref.load(table, [mgpu_arith.index_cast(index, row)])
            nonnegative = mgpu_arith.cmpi(mgpu_arith.CmpIPredicate.sge, token, mgpu_utils.c(0, i32))
            stored_here = mgpu_scf.IfOp(nonnegative)
            with mgpu_ir.InsertionPoint(stored_here.then_block):
                offset = mgpu_arith.addi(mgpu_arith.muli(row, mgpu_utils.c(width, i32)),
                                         mgpu_arith.muli(lane, mgpu_utils.c(lanes, i32)))
                value = mgpu_llvm.load(vector, mgpu_utils.getelementptr(source, [offset], element))
                place = mgpu_arith.addi(
                    mgpu_arith.muli(mgpu_arith.extsi(i64, token), mgpu_utils.c(row_stride, i64)),
                    mgpu_arith.muli(mgpu_arith.extui(i64, mgpu_arith.muli(lane, mgpu_utils.c(lanes, i32))),
                                    mgpu_utils.c(column_stride, i64)))
                mgpu_llvm.store(value, mgpu_utils.getelementptr(base, [place], element))
                mgpu_scf.YieldOp([])
            mgpu_scf.YieldOp([])
    mgpu_utils.warpgroup_barrier()
