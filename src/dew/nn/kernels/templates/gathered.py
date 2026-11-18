"""Rows where a table says (row 1376): a gathered ring (`Ring`), a stream whose rows each lie where a table
says, a mixture of experts' tokens read in their own order, loaded by the producer warpgroup's cp.async
copies (Hopper's) or by TMA's four-row gathers (`Committed`, tcgen05's load warp), and a scattered store
(`scattered_rows`), an output's rows each written where a permutation says, each through `inline_mgpu`'s
lane-level code (Pallas's warpgroup semantics reach no thread's own copy, address or mbarrier)."""

import contextlib
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

from dew.nn.kernels.templates import across, rings, stores, swizzling


@dataclasses.dataclass(frozen=True)
class Ring(rings.Ring):
    """A `rings.Ring` of rows each read where a table says (`rings.Rows` of one row a run), loaded by the
    producer warpgroup's cp.async copies: each thread copies its 16-byte pieces of the slot's rows from the
    rows the table names (`gathered_copies`), tracked at the slot's load barrier, an mbarrier of shared
    memory (`ring`: Pallas's barriers take no lane-level arrival); the consumers wait on it (`acquire`) and
    release the slot as any ring's. TMA lands a copy in whole tiles of rows, a row alone only by copies
    Mosaic's layout inference refuses, and from one thread a row a copy (about 5k clocks a step against 1k
    of products); CUTLASS's sm90 gather is the producer warps' cp.async too."""

    def initialized(self) -> None:
        plgpu.inline_mgpu(arg_types=(plgpu.RefType(),))(Initialized(self.stages))(self.full)

    def viewed(self, view: tuple, step):
        ref, index_map = view
        return ref, index_map(step)

    def load(self, turn: rings.Turn, view) -> None:
        slot = rings.opaque(turn.slot)
        self.free(turn, slot)
        tokens, rows = view
        height, width = self.block
        [depth] = rows.rest
        source = tokens.at[:, pl.ds(depth * width, width)]
        bytes_ = swizzled(self.block, tokens.dtype)
        swizzle = swizzling.swizzle_transforms(tokens.dtype, bytes_)
        copies = functools.partial(gathered_copies, swizzle=bytes_)
        refs = (plgpu.RefType(), plgpu.RefType(), plgpu.RefType(swizzle), plgpu.RefType())
        plgpu.inline_mgpu(arg_types=refs)(lambda ctx, *refs: copies(*refs))(
            source, rows.table.at[pl.ds(rows.start, height)], self.smem.at[slot],
            self.full.at[pl.ds(slot, 1)])

    def acquire(self, turn: rings.Turn):
        slot = rings.opaque(turn.slot)
        plgpu.inline_mgpu(arg_types=(plgpu.RefType(), plgpu.Layout.WG_SPLAT(())))(Waited())(
            self.full.at[pl.ds(slot, 1)], jnp.asarray(turn.rotation, jnp.int32))
        return self.smem.at[slot]


def ring(block: tuple, dtype, stages: int, *, consumers: int = 1, released_by_products: bool = False) -> dict:
    """A gathered ring's allocations for `pl.run_scoped`: `stages` slots of `block` in `dtype`, their rows
    staged at the widest swizzle that fits (`swizzled`), a load mbarrier each in shared memory (`uint64`,
    `Ring.initialized`) and a release barrier each that `consumers` arrive at, or, `released_by_products`,
    that the products reading a slot arrive at once by their commit (`Committed`'s: tcgen05's, as
    `rings.ring`'s)."""
    consumed = (plgpu.Barrier(num_barriers=stages, orders_tensor_core=True) if released_by_products
                else plgpu.Barrier(num_arrivals=consumers, num_barriers=stages))
    return {'smem': swizzling.staged((stages, *block), dtype, swizzled(block, dtype)),
            'full': plgpu.SMEM((stages,), jnp.uint64), 'consumed': consumed}


def swizzled(block: tuple, dtype) -> int:
    """The swizzle a gathered ring's slots of `block` stage at: the widest whose row its columns hold
    whole (`swizzling.swizzle_for`), as layout inference stages an ungathered one: 128 bytes but a narrow
    block's (a 32-column bf16 cotangent of mma.sync's kernel gradient, 64)."""
    return swizzling.swizzle_for(dtype, block[1])


@dataclasses.dataclass(frozen=True)
class Committed(Ring):
    """A gathered ring tcgen05's load warps fill and its products give back (`mosaic_sm100.CommittedRing`'s
    release, their commit arriving at the slot's): each step's rows by TMA's four-row gathers (`Gathers`,
    `cp.async.bulk.tensor.2d...tile::gather4`, sm100's), its 8-row tiles spread over the producer's load
    warps (`share`, `issued_by`: tile t the warp t mod their count's), each warp's first thread issuing its
    tiles' and the first warp's alone arriving once at the slot's load mbarrier, expecting every tile's bytes,
    which the copies complete. A row alone by tiled TMA is a copy of its own (an 8-row tile's slice no
    swizzled box takes); a gather moves four.

    One thread issuing every gather held an emulated B200's gathered kernel gradient on that thread: each
    gather4 cost about 17 instructions moving its row indices into uniform registers (R2UR), the step's
    indices spilled the warp's uniform registers, and the product warp's step ran 2,728 cycles to the rows
    gathered first's 387 (M-EM-61, em61map). A warp's uniform registers and issue slots are its own, so its
    share of a step's gathers is its share of that work."""
    share: tuple[int, int] = (0, 1)
    """This load warp's index among the producer's load warps, and their count (`issued_by`)."""

    def issued_by(self, index: int, count: int) -> "Committed":
        return dataclasses.replace(self, share=(index, count))

    def load(self, turn: rings.Turn, view) -> None:
        slot = rings.opaque(turn.slot)
        self.free(turn, slot)
        tokens, rows = view
        height, width = self.block
        [depth] = rows.rest
        bytes_ = swizzled(self.block, tokens.dtype)
        swizzle = swizzling.swizzle_transforms(tokens.dtype, bytes_)
        refs = (plgpu.RefType(), plgpu.RefType(), plgpu.RefType(swizzle), plgpu.RefType(),
                plgpu.Layout.WG_SPLAT(()))
        plgpu.inline_mgpu(arg_types=refs)(Gathers(bytes_, self.share))(
            tokens, rows.table.at[pl.ds(rows.start, height)], self.smem.at[slot],
            self.full.at[pl.ds(slot, 1)], jnp.asarray(depth * width, jnp.int32))

    def release(self, turn: rings.Turn) -> None:
        plgpu.tcgen05_commit_arrive(self.consumed.at[rings.opaque(turn.slot)])


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


@dataclasses.dataclass(frozen=True)
class Initialized:
    """`Ring.initialized`'s body (`plgpu.inline_mgpu`'s): each of a ring's `stages` load mbarriers (shared
    `uint64`) initialized to `arrivals` arrivals a phase, by one thread of the block, the block meeting after
    so none is waited on or arrived at first. One a phase: the producer warpgroup's once its cp.async copies
    are issued (`gathered_copies`), or the first load warp's expecting every gather's bytes (`Gathers`). The
    synchronization checker reads each inlined body's fields (tests/ring_protocol.py)."""

    stages: int
    arrivals: int = 1

    def __call__(self, ctx, barriers) -> None:
        del ctx
        i32, i64 = mgpu_ir.IntegerType.get_signless(32), mgpu_ir.IntegerType.get_signless(64)
        base = mgpu_utils.memref_ptr(barriers)
        predicate = mgpu_utils.single_thread_predicate(scope=mgpu_utils.ThreadSubset.BLOCK)
        for stage in range(self.stages):
            mgpu_nvvm.mbarrier_init(mgpu_utils.getelementptr(base, [stage], i64),
                                    mgpu_utils.c(self.arrivals, i32), predicate=predicate)
        mgpu_gpu.barrier()


@dataclasses.dataclass(frozen=True)
class Waited:
    """`Ring.acquire`'s body: the thread's wait at a slot's load mbarrier (a one-entry shared `uint64`) for
    the fill of the turn whose `rotation` it is given, at the rotation's parity (`rings.Turn.parity`,
    `wait_parity`'s: no phases tracked). Given the rotation whole, not its parity, the body states which
    completion it waits for, as a `plgpu.barrier_wait`'s phase does."""

    def __call__(self, ctx, barrier, rotation) -> None:
        del ctx
        parity = mgpu_arith.trunci(mgpu_ir.IntegerType.get_signless(1), splat_value(rotation))
        barrier_of(barrier).wait_parity(parity)


def gathered_copies(source, rows, slot, barrier, *, swizzle: int) -> None:
    """The producer warpgroup's cp.async copies of the slot's rows (`Ring.load`, `row_copies`), tracked at
    `barrier` and the warpgroup's one arrival there once every thread has issued its own."""
    row_copies(source, rows, slot, swizzle=swizzle)
    tracked = barrier_of(barrier)
    # cp.async.mbarrier.arrive is no arrival: it holds the phase open until this thread's copies land.
    mgpu_nvvm.cp_async_mbarrier_arrive(tracked.get_ptr())
    mgpu_utils.warpgroup_barrier()
    tracked.arrive(predicate=mgpu_utils.single_thread_predicate(mgpu_utils.ThreadSubset.WARPGROUP))


def commit_group(ctx) -> None:
    """The thread's cp.async copies issued since its last group committed as one (`plgpu.inline_mgpu`'s
    body, `ctx` its launch context): what `plgpu.wait_gmem_to_smem` counts, as Mosaic GPU's own cp.async
    copies commit theirs (`LaunchContext.async_copy`)."""
    del ctx
    mgpu_nvvm.cp_async_commit_group()


def row_copies(source, rows, slot, *, swizzle: int) -> None:
    """A warpgroup's cp.async copies of a slot's rows, a row a copy: row `r` of `slot` (`[height, width]`,
    8-row tiles at `swizzle` bytes) row `rows[r]` of `source` (`[tokens, width]`, the step's columns of the
    tokens), each thread its 16-byte pieces in the tiled copy's layout (`tiled_copy_smem_gmem_layout`, as
    Mosaic GPU's own cp.async copy of a tile places them), issued and not awaited: a ring's tracked at its
    load barrier (`gathered_copies`), an A100 pipeline's in its cp.async groups (`mosaic_sm80.CpAsync`)."""
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


@dataclasses.dataclass(frozen=True)
class Gathers:
    """`Committed.load`'s body: its copies by one of the producer's load warps (`share`: its index among them
    and their count): row `r` of `slot` (`[height, width]`, 8-row tiles at `swizzle` bytes) row `rows[r]` of
    `source` (the tokens, whole: TMA's descriptor is of a kernel argument) from `column`, each of the tiles
    the warp issues (`tiles`) by two four-row gathers (`LaunchContext.async_copy`'s gather, its indices
    `TMA_INDICES_4_LAYOUT`: every thread holds them, the warp's first thread issues), the first warp's first
    thread arriving once at `barrier` expecting every tile's bytes, every warp's (`expects`).

    The other warps' copies may land before that arrival: each completes its bytes on the barrier, its
    transaction count going below zero (PTX's mbarrier tx-count ranges over -(2^20 - 1) to 2^20 - 1), and the
    phase completes only once its one pending arrival (the first warp's, with its expected bytes) is made and
    the count is back at zero. A warp waits for the slot's release before its copies (`Ring.free`), so no
    copy lands in a phase but its step's: the slot's last phase completed before its consumers released it
    (tests/test_ring_protocol.py's gathered walk, every order of the warps' copies and the arrival)."""

    swizzle: int
    share: tuple[int, int]

    @property
    def expects(self) -> bool:
        """Whether the warp arrives at the slot's load barrier expecting every tile's bytes: the first's
        alone."""
        return self.share[0] == 0

    def tiles(self, count: int) -> range:
        """The tiles of a step's `count` the warp issues: tile t the warp t mod their count's."""
        warp, warps = self.share
        return range(warp, count, warps)

    def __call__(self, ctx, source, rows, slot, barrier, column) -> None:
        i32 = mgpu_ir.IntegerType.get_signless(32)
        index = mgpu_ir.IndexType.get()
        shape = tuple(mgpu_ir.MemRefType(slot.type).shape)
        tile_rows, tile_columns = shape[2], shape[3]
        width = shape[1] * tile_columns
        element = mgpu_ir.MemRefType(source.type).element_type
        tracked = barrier_of(barrier)
        total = shape[0] * tile_rows * width * mgpu_utils.bitwidth(element) // 8
        if self.expects:
            mgpu_nvvm.mbarrier_arrive_expect_tx(tracked.get_ptr(), mgpu_utils.c(total, i32),
                                                predicate=mgpu_utils.single_thread_predicate(
                                                    mgpu_utils.ThreadSubset.WARP))
        layout = mgpu_fragments.TMA_INDICES_4_LAYOUT
        vector = mgpu_ir.VectorType.get((4,), i32)
        # The step's column as the warp holds it uniformly, as its rows' entries are (`_entries`).
        start = across.lane_value(splat_value(column), 0)
        issuer = mgpu_utils.single_thread_predicate(mgpu_utils.ThreadSubset.WARP)
        tiles = list(self.tiles(shape[0]))
        entries = self._entries(rows, len(tiles), tile_rows)
        for at, tile in enumerate(tiles):
            # The tile's entries, four a register, every thread the same (`layout`).
            mine = entries[at * tile_rows:(at + 1) * tile_rows]
            registers = [mgpu_vector.from_elements(vector, mine[4 * part:4 * (part + 1)])
                         for part in range(tile_rows // 4)]
            indices = mgpu_fragments.FragmentedArray(
                _registers=_registered(registers, layout, tile_rows), _layout=layout, _is_signed=False)
            ctx.async_copy(src_ref=source, dst_ref=mgpu_utils.memref_slice(slot, mgpu_utils.ds(tile, 1)),
                           gmem_slice=(indices, mgpu_utils.ds(mgpu_arith.index_cast(index, start), width)),
                           gmem_transform=mgpu_launch.TileTransform((tile_rows, tile_columns)),
                           swizzle=self.swizzle, barrier=tracked, arrive=False, predicate=issuer)

    def _entries(self, rows, tiles: int, tile_rows: int) -> list:
        """The table entries of the warp's `tiles` tiles (`rows`, a step's table in global memory: tile t's
        `tile_rows` entries from row t * `tile_rows`, the warp's tiles those `tiles` names), in order, each as
        every lane of the warp holds it uniformly: its lanes load them a warp's width at a time, an entry a
        lane, and each is read from its lane (`across.lane_value`), which ptxas holds in a uniform register,
        where a TMA gather reads its indices. Loaded by every thread and read from a register a lane, each
        gather4's four were moved into uniform registers by R2UR (66% of a load warp's issues at the line
        that loaded them, the emulated B200's gathered kernel gradient, M-EM-71)."""
        i32, index = mgpu_ir.IntegerType.get_signless(32), mgpu_ir.IndexType.get()
        warp, warps = self.share
        count = tiles * tile_rows
        lane = mgpu_arith.remui(mgpu_utils.thread_idx(), mgpu_utils.c(mgpu_utils.WARP_SIZE, i32))
        found = []
        for first in range(0, count, mgpu_utils.WARP_SIZE):
            # The lane's entry: its tile among the warp's (tile `warp + q * warps`, `tiles`), its row within.
            position = mgpu_arith.addi(lane, mgpu_utils.c(first, i32))
            tile = mgpu_arith.addi(mgpu_utils.c(warp, i32), mgpu_arith.muli(
                mgpu_arith.divui(position, mgpu_utils.c(tile_rows, i32)), mgpu_utils.c(warps, i32)))
            row = mgpu_arith.addi(mgpu_arith.muli(tile, mgpu_utils.c(tile_rows, i32)),
                                  mgpu_arith.remui(position, mgpu_utils.c(tile_rows, i32)))
            # A lane past the warp's entries loads the table's first, which no gather reads.
            held = mgpu_arith.cmpi(mgpu_arith.CmpIPredicate.ult, position, mgpu_utils.c(count, i32))
            row = mgpu_arith.select(held, row, mgpu_utils.c(0, i32))
            loaded = mgpu_memref.load(rows, [mgpu_arith.index_cast(index, row)])
            found += [across.lane_value(loaded, at)
                      for at in range(min(mgpu_utils.WARP_SIZE, count - first))]
        return found


def _registered(registers: list, layout, rows: int):
    """`registers` (each four entries) as a `FragmentedArray`'s of `rows` entries in `layout`."""
    held = np.empty(layout.registers_shape((rows,)), dtype=object)
    for at, register in zip(np.ndindex(held.shape), registers, strict=True):
        held[at] = register
    return held


def scattered_rows(smem, out: stores.Scattered, row_start, first, stored, column) -> None:
    """Rows `first` to `first + stored` of `smem` (`[rows, width]`, unswizzled: a consumer's staged share of
    a tile, `stores.staged_rows`) each written to its row of `out` (`out.table[row_start + r]`, a negative
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
    """`scattered_rows`' lane-level copy (`row_pieces`): each stored row to its table entry's row of
    `dest`, none where the entry is negative."""
    i32 = mgpu_ir.IntegerType.get_signless(32)
    index = mgpu_ir.IndexType.get()

    def placed(row):
        token = mgpu_memref.load(table, [mgpu_arith.index_cast(index, row)])
        return token, mgpu_arith.cmpi(mgpu_arith.CmpIPredicate.sge, token, mgpu_utils.c(0, i32))

    row_pieces(smem, dest, first, stored, placed, swizzle=swizzling.UNSWIZZLED)


def swizzled_pieces(smem, dest, start, first, stored, *, swizzle: int) -> None:
    """`stores.swizzled_rows_out`' lane-level copy (`row_pieces`): each stored row `r` of a tile staged at
    `swizzle` bytes to row `start + r` of `dest`."""
    row_pieces(smem, dest, first, stored, lambda row: (mgpu_arith.addi(start, row), None), swizzle=swizzle)


def row_pieces(smem, dest, first, stored, placed, *, swizzle: int) -> None:
    """A staged tile's rows `first` to `first + stored` written to `dest` by the warpgroup's threads: the
    threads met (the tile staged by all of them), then each of its 16-byte pieces, thread `t` taking pieces
    `t`, `t + 128`, ... in a loop of its own over the turns (each turn's addresses live within it: unrolled,
    a 64 x 320 forward's spilled 1440 bytes on an A100), read where the staging put it
    (`swizzling.staged_address`: plain, or a swizzle's 8-row tiles with each 16-byte piece XORed by its row's
    place) and written to `dest`'s row `placed(r)` gives with its predicate (None for every stored row), the
    threads met again before the buffer is written."""
    i32, i64 = mgpu_ir.IntegerType.get_signless(32), mgpu_ir.IntegerType.get_signless(64)
    element = mgpu_ir.MemRefType(smem.type).element_type
    bits = mgpu_utils.bitwidth(element)
    shape = tuple(mgpu_ir.MemRefType(smem.type).shape)
    rows, width = (shape[0] * shape[2], shape[1] * shape[3]) if len(shape) == 4 else shape
    row_bytes = width * bits // 8
    per_row = row_bytes // 16
    pieces = rows * per_row
    [row_stride, column_stride], _offset = mgpu_ir.MemRefType(dest.type).get_strides_and_offset()
    vector = mgpu_ir.VectorType.get((4,), i32)
    byte = mgpu_ir.IntegerType.get_signless(8)
    source = mgpu_utils.memref_ptr(smem)
    base = mgpu_llvm.addrspacecast(mgpu_llvm.PointerType.get(address_space=1), mgpu_utils.memref_ptr(dest))
    thread = mgpu_arith.remui(mgpu_arith.index_cast(i32, mgpu_gpu.thread_id(mgpu_gpu.Dimension.x)),
                              mgpu_utils.c(128, i32))

    def c(value: int):
        return mgpu_utils.c(value, i32)

    def staged(row, column_bytes):
        # `swizzling.staged_address`, traced.
        if swizzle == swizzling.UNSWIZZLED:
            return mgpu_arith.addi(mgpu_arith.muli(row, c(row_bytes)), column_bytes)
        tile = mgpu_arith.muli(
            mgpu_arith.addi(mgpu_arith.muli(mgpu_arith.divui(row, c(swizzling.TILE_ROWS)),
                                            c(row_bytes // swizzle)),
                            mgpu_arith.divui(column_bytes, c(swizzle))),
            c(swizzling.TILE_ROWS * swizzle))
        in_tile = mgpu_arith.muli(mgpu_arith.remui(row, c(swizzling.TILE_ROWS)), c(swizzle))
        within = mgpu_arith.remui(column_bytes, c(swizzle))
        offset = mgpu_arith.addi(mgpu_arith.addi(tile, in_tile), within)
        return mgpu_arith.xori(offset, mgpu_arith.shli(mgpu_arith.andi(
            mgpu_arith.shrui(offset, c(7)), c(swizzle // 16 - 1)), c(4)))

    mgpu_utils.warpgroup_barrier()
    end = mgpu_arith.addi(first, stored)

    @mgpu_utils.fori(c(-(-pieces // 128)), ())
    def _turns(turn, carry):
        piece = mgpu_arith.addi(thread, mgpu_arith.muli(turn, c(128)))
        row = mgpu_arith.divui(piece, c(per_row))
        lane = mgpu_arith.remui(piece, c(per_row))
        inside = mgpu_arith.andi(
            mgpu_arith.cmpi(mgpu_arith.CmpIPredicate.slt, piece, c(pieces)),
            mgpu_arith.andi(mgpu_arith.cmpi(mgpu_arith.CmpIPredicate.sge, row, first),
                            mgpu_arith.cmpi(mgpu_arith.CmpIPredicate.slt, row, end)))
        branch = mgpu_scf.IfOp(inside)
        with mgpu_ir.InsertionPoint(branch.then_block):
            target, admitted = placed(row)
            written = mgpu_scf.IfOp(admitted) if admitted is not None else None
            scope = (contextlib.nullcontext() if written is None
                     else mgpu_ir.InsertionPoint(written.then_block))
            with scope:
                column_bytes = mgpu_arith.muli(lane, c(16))
                address = mgpu_utils.getelementptr(source, [staged(row, column_bytes)], byte)
                value = mgpu_llvm.load(vector, address)
                place = mgpu_arith.addi(
                    mgpu_arith.muli(mgpu_arith.extsi(i64, target), mgpu_utils.c(row_stride, i64)),
                    mgpu_arith.muli(mgpu_arith.extui(i64, mgpu_arith.muli(lane, c(128 // bits))),
                                    mgpu_utils.c(column_stride, i64)))
                mgpu_llvm.store(value, mgpu_utils.getelementptr(base, [place], element))
                if written is not None:
                    mgpu_scf.YieldOp([])
            mgpu_scf.YieldOp([])
        return carry

    mgpu_utils.warpgroup_barrier()

