"""A gathered ring (`Ring`): a stream whose rows each lie where a table says, a mixture of experts' tokens
read in their own order (row 1376), loaded by the producer warpgroup's cp.async copies through
`inline_mgpu`'s lane-level code (Pallas's warpgroup semantics reach no thread's own copy or mbarrier)."""

import dataclasses
import functools

import jax.numpy as jnp
from jax._src.lib.mlir.dialects import gpu as mgpu_gpu, llvm as mgpu_llvm, nvvm as mgpu_nvvm
from jax.experimental import pallas as pl
from jax.experimental.mosaic.gpu import fragmented_array as mgpu_fragments, utils as mgpu_utils
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend.mlir import ir as mgpu_ir
from jax.extend.mlir.dialects import arith as mgpu_arith, memref as mgpu_memref

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
        swizzle = mosaic.swizzle_transforms(tokens.dtype, mosaic.SWIZZLE_BYTES)
        copies = functools.partial(gathered_copies, swizzle=mosaic.SWIZZLE_BYTES)
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


def ring(block: tuple, dtype, stages: int, *, consumers: int = 1) -> dict:
    """A gathered `Ring`'s allocations for `pl.run_scoped`: `stages` slots of `block` in `dtype`, their rows
    staged at the widest swizzle, a load mbarrier each in shared memory (`uint64`, `Ring.initialized`)
    and a release barrier each that `consumers` arrive at."""
    return {'smem': mosaic.staged((stages, *block), dtype, mosaic.SWIZZLE_BYTES),
            'full': plgpu.SMEM((stages,), jnp.uint64),
            'consumed': plgpu.Barrier(num_arrivals=consumers, num_barriers=stages)}


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
