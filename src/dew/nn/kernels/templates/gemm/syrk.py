"""Dew: symmetric rank-k update (SYRK) and fused symmetric polynomial epilogue on the shared Mosaic toolkit
(row 11941, Muon Newton-Schulz Gram `X @ X^T` / `X^T @ X` and `c0 * I + c1 * A + c2 * (A @ A)`):

- Computes only the upper triangle of square super-tiles (`ti <= tj`, where `tile_rows == tile_columns`) and
  mirrors each strictly upper tile (`ti < tj`) to `(tj, ti)` in the epilogue with zero extra shared memory:
  - `sm90` (`wgmma`) and `sm100` (`tcgen05`) 16-bit: direct transposed register -> SMEM write
    (`smem_t.T[...] = piece.astype(dtype)` -> `stmatrix.sync.aligned.m8n8.x4.trans.shared.b16`), reusing the
    existing `[64, 64]` staging on square Hopper tiles or per-buffer `plgpu.RefUnion` on non-square chunks,
    with interleaved `buffer_waits` across normal and mirror TMA stores.
  - `sm90` and `sm100` fp32: in-register `plgpu.layout_cast(..., WGMMA_TRANSPOSED / TCGEN05_TRANSPOSED)`.
  - `sm80`/`sm86`/`sm89`/`sm120` (`mma.sync`): `_swizzled_columns_out` drains the already-staged swizzled SMEM
    transposed with a 0-bank-conflict staggered bijection and coalesced 128B global stores.
  - TPU (`mxu`): native VMEM `value.T` mirror store.
"""
import dataclasses
import math
from collections.abc import Callable
from typing import ClassVar

import jax
from jax import numpy as jnp
from jax._src.lib.mlir.dialects import gpu as mgpu_gpu, llvm as mgpu_llvm, scf as mgpu_scf
from jax.experimental import pallas as pl
from jax.experimental.mosaic.gpu import utils as mgpu_utils
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend.mlir import ir as mgpu_ir
from jax.extend.mlir.dialects import arith as mgpu_arith, vector as mgpu_vector

from dew.nn.kernels import generation, grouped_launch, tuning
from dew.nn.kernels.templates import (
    bundles,
    cost as unit_cost,
    mosaic,
    report,
    resources,
    rings,
    stores,
    swizzling,
)
from dew.nn.kernels.templates.gemm import derivation
from dew.nn.kernels.templates.gemm.mosaic_gpu import (
    APPROX_MATH,
    MOVES,
    Allocations,
    multicast,
    operand_streams,
    pair_takes,
    row_unit,
)
from dew.nn.kernels.templates.gemm.pricing import (
    TilesTemplate,
    epilogue_registers,
    family_of,
    granular_tile,
    tile_columns,
    tile_rows,
)
from dew.nn.kernels.templates.gemm.row_tiles import TMA_ROWS
from dew.nn.kernels.templates.gemm.tiling import CONSUMERS, Tiles

SYRK = "dew_syrk"
"""The name of a symmetric Gram/polynomial kernel (`syrk`), its producer's registers keyed by it."""


def supports_out_dtype(out_dtype) -> bool:
    """Whether native SYRK supports storing `out_dtype`: 16-bit (`bf16`/`fp16`) or 32-bit (`fp32`) floats."""
    dtype = jnp.dtype(out_dtype)
    return jnp.issubdtype(dtype, jnp.floating) and dtype.itemsize in (2, 4)


def upper_tile(linear: jax.Array, side: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Map linear upper-triangle tile index `linear` in `[0, g * tri)` to `(batch, ti, tj)` with
    `0 <= ti <= tj < side`, exact branchless pure-int32 binary search (lowers under tcgen05 Warp semantics
    as well as warpgroup, TPU, and CPU)."""
    tri = side * (side + 1) // 2
    batch = linear // tri
    k = linear % tri
    r = tri - 1 - k
    d = jnp.zeros_like(r)
    for bit in range((side - 1).bit_length() - 1, -1, -1):
        cand = d + (1 << bit)
        d = jnp.where((cand < side) & (cand * (cand + 1) // 2 <= r), cand, d)
    ti = side - 1 - d
    tj = side - 1 - (r - d * (d + 1) // 2)
    return batch, ti, tj


def square_super_tile(tiles: Tiles, family: mosaic.Family) -> bool:
    """Whether `tiles` forms a square super-tile (`tile_rows == tile_columns`) required by SYRK."""
    return tile_rows(tiles, family) == tile_columns(tiles, family)


def _uses_custom_staging(family: mosaic.Family, interpret: bool) -> bool:
    """Whether SYRK manages its own normal+mirror SMEM staging outside `family.persist`."""
    return family.budgets_registers and not family.stores_through_registers and not interpret


def _mirror_staging(part: tuple[int, int], out_dtype, consumers: int, family: mosaic.Family, *,
                    interpret: bool) -> tuple:
    """SMEM staging buffers for compiled RingFamily SYRK (`0` extra SMEM vs normal `output_staging`)."""
    if not _uses_custom_staging(family, interpret):
        return ()
    share, width = part
    chunk, buffers = stores.output_staging(width, (out_dtype,))
    if share == chunk:
        return (plgpu.SMEM((consumers, buffers, share, chunk), out_dtype),)
    return tuple(
        plgpu.RefUnion(plgpu.SMEM((consumers, share, chunk), out_dtype),
                       plgpu.SMEM((consumers, chunk, share), out_dtype))
        for _ in range(buffers)
    )


def syrk_allocations(tiles: Tiles, family: mosaic.Family, *, dtype, out_dtype, transpose: bool,
                     polynomial: bool, interpret: bool = False) -> Allocations:
    """Operand ring blocks, polynomial side staging, and mirror output staging for `syrk` (shared by
    `syrk()` and `grouped_matmul.MosaicSyrk.launch_of`)."""
    block_m, block_n, block_k = tiles.block_m, tiles.block_n, tiles.block_k
    columns = tile_columns(tiles, family)
    consumers = tiles.warpgroups
    part = (tiles.share, columns)
    blocks = ({'rows': ((block_k, block_m), dtype), 'matrix': ((block_k, block_n), dtype)} if transpose
              else {'rows': ((block_m, block_k), dtype), 'matrix': ((block_n, block_k), dtype)})
    side_smems = [family.staged_memory((consumers, *part), dtype)] if polynomial else []
    side_barriers = [family.load_barriers(consumers)] if polynomial else []
    custom_staging = _mirror_staging(part, out_dtype, consumers, family, interpret=interpret)
    staged = (side_smems, side_barriers, custom_staging)
    # Report normal output staging in `outputs` when persist allocates it or for resource planning on CpAsync.
    persist_outputs = () if _uses_custom_staging(family, interpret) else ((part, out_dtype),)
    return Allocations(blocks=blocks, staged=staged, outputs=persist_outputs, shared=('matrix',))


def _swizzled_columns_out(smem, dest, row_start, col_start, *, swizzle: int) -> None:
    """Drain already-staged swizzled SMEM `[rows, cols]` transposed into `dest[col_start + c, row_start + r]`
    on `mma.sync` (`sm80`/`sm86`/`sm89`/`sm120`) with 0 bank conflicts and coalesced 128B global stores."""
    scalar = plgpu.Layout.WG_SPLAT(())

    def emit(ctx, smem_ref, dest_ref, r_start_val, c_start_val):
        del ctx
        from dew.nn.kernels.templates import gathered

        _emit_transposed_pieces(smem_ref, dest_ref, gathered.splat_value(r_start_val),
                                gathered.splat_value(c_start_val), swizzle=swizzle)

    transforms = ((plgpu.PlainTransform(),) if swizzle == swizzling.UNSWIZZLED
                  else swizzling.swizzle_transforms(smem.dtype, swizzle))
    plgpu.inline_mgpu(arg_types=(plgpu.RefType(transforms), plgpu.RefType(), scalar, scalar))(emit)(
        smem, dest, jnp.asarray(row_start, jnp.int32), jnp.asarray(col_start, jnp.int32))


def _i32(value: int, i32):
    """Constant signless 32-bit MLIR integer."""
    return mgpu_utils.c(value, i32)


def _swizzled_tile_anchor(slab, col_bytes, *, row_bytes: int, swizzle: int, i32):
    """Loop-invariant `(base, within, slab_rem)` for reading `swizzling.TILE_ROWS`-aligned slabs."""
    if swizzle == swizzling.UNSWIZZLED:
        return mgpu_arith.addi(mgpu_arith.muli(slab, _i32(row_bytes, i32)), col_bytes), None, None
    slab_tile = mgpu_arith.divui(slab, _i32(swizzling.TILE_ROWS, i32))
    slab_rem = mgpu_arith.remui(slab, _i32(swizzling.TILE_ROWS, i32))
    tile_index = mgpu_arith.addi(mgpu_arith.muli(slab_tile, _i32(row_bytes // swizzle, i32)),
                                 mgpu_arith.divui(col_bytes, _i32(swizzle, i32)))
    tile_base = mgpu_arith.muli(tile_index, _i32(swizzling.TILE_ROWS * swizzle, i32))
    return tile_base, mgpu_arith.remui(col_bytes, _i32(swizzle, i32)), slab_rem


def _swizzled_lane_offset(anchor: tuple, dr, *, row_bytes: int, swizzle: int, i32):
    """Byte offset in swizzled SMEM for row `slab + dr` (`anchor` from `_swizzled_tile_anchor`)."""
    base, within, slab_rem = anchor
    if swizzle == swizzling.UNSWIZZLED:
        return mgpu_arith.addi(base, mgpu_arith.muli(dr, _i32(row_bytes, i32)))
    in_tile = mgpu_arith.addi(slab_rem, dr)
    shift = (swizzling.SWIZZLE_BYTES // swizzle).bit_length() - 1
    shifted = mgpu_arith.shrui(in_tile, _i32(shift, i32)) if shift else in_tile
    xor_bits = mgpu_arith.shli(shifted, _i32(4, i32))
    return mgpu_arith.addi(mgpu_arith.addi(base, mgpu_arith.muli(in_tile, _i32(swizzle, i32))),
                           mgpu_arith.xori(within, xor_bits))


def _store_transposed_vec4(base, element, row_stride: int, col_stride: int, dest_row, dest_col, words,
                           *, i32, i64) -> None:
    """Pack 4 i32 words into a 16-byte global vector store at `(dest_row, dest_col)`."""
    vec4 = mgpu_ir.VectorType.get((4,), i32)
    packed = mgpu_vector.from_elements(vec4, words)
    place = mgpu_arith.addi(
        mgpu_arith.muli(mgpu_arith.extsi(i64, dest_row), mgpu_utils.c(row_stride, i64)),
        mgpu_arith.muli(mgpu_arith.extsi(i64, dest_col), mgpu_utils.c(col_stride, i64)))
    mgpu_llvm.store(packed, mgpu_utils.getelementptr(base, [place], element))


def _emit_transposed_16(source, store_vec, warp, c_grp, r_blk, row_start, col_start, *,
                        rows: int, cols: int, row_bytes: int, swizzle: int, i32, byte) -> None:
    """16-bit transposed drain: 64x8 warp tiles, staggered `dr` loads, barrel-shift + 2x `v4.u32` stores."""
    n_c = cols // 8
    total = (rows // 64) * n_c
    mask_lo, mask_hi = _i32(0xFFFF, i32), _i32(0xFFFF0000, i32)

    @mgpu_utils.fori(_i32(-(-total // 4), i32), ())
    def _turns16(turn, carry):
        blk = mgpu_arith.addi(warp, mgpu_arith.muli(turn, _i32(4, i32)))
        branch = mgpu_scf.IfOp(mgpu_arith.cmpi(mgpu_arith.CmpIPredicate.slt, blk, _i32(total, i32)))
        with mgpu_ir.InsertionPoint(branch.then_block):
            r_base = mgpu_arith.muli(mgpu_arith.divui(blk, _i32(n_c, i32)), _i32(64, i32))
            c_base = mgpu_arith.muli(mgpu_arith.remui(blk, _i32(n_c, i32)), _i32(8, i32))
            col = mgpu_arith.addi(c_base, mgpu_arith.shli(c_grp, _i32(1, i32)))
            slab = mgpu_arith.addi(r_base, mgpu_arith.shli(r_blk, _i32(3, i32)))
            anchor = _swizzled_tile_anchor(slab, mgpu_arith.shli(col, _i32(1, i32)),
                                           row_bytes=row_bytes, swizzle=swizzle, i32=i32)
            raw = [mgpu_llvm.load(i32, mgpu_utils.getelementptr(
                source, [_swizzled_lane_offset(anchor, mgpu_arith.andi(
                    mgpu_arith.addi(_i32(k, i32), r_blk), _i32(7, i32)),
                    row_bytes=row_bytes, swizzle=swizzle, i32=i32)], byte)) for k in range(8)]
            for shift in (1, 2, 4):
                bit = mgpu_arith.cmpi(mgpu_arith.CmpIPredicate.ne,
                                      mgpu_arith.andi(r_blk, _i32(shift, i32)), _i32(0, i32))
                raw = [mgpu_arith.select(bit, raw[(d - shift) & 7], raw[d]) for d in range(8)]
            lo = [mgpu_arith.ori(mgpu_arith.andi(raw[2 * i], mask_lo),
                                 mgpu_arith.shli(mgpu_arith.andi(raw[2 * i + 1], mask_lo), _i32(16, i32)))
                  for i in range(4)]
            hi = [mgpu_arith.ori(mgpu_arith.shrui(raw[2 * i], _i32(16, i32)),
                                 mgpu_arith.andi(raw[2 * i + 1], mask_hi)) for i in range(4)]
            d_row, d_col = mgpu_arith.addi(col_start, col), mgpu_arith.addi(row_start, slab)
            store_vec(d_row, d_col, lo)
            store_vec(mgpu_arith.addi(d_row, _i32(1, i32)), d_col, hi)
            mgpu_scf.YieldOp([])
        return carry


def _emit_transposed_32(source, store_vec, warp, c_grp, r_blk, row_start, col_start, *,
                        rows: int, cols: int, row_bytes: int, swizzle: int, i32, byte) -> None:
    """32-bit transposed drain: 32x4 warp tiles, staggered `dr` loads, barrel-shift + `v4.u32` store."""
    n_c = cols // 4
    total = (rows // 32) * n_c

    @mgpu_utils.fori(_i32(-(-total // 4), i32), ())
    def _turns32(turn, carry):
        blk = mgpu_arith.addi(warp, mgpu_arith.muli(turn, _i32(4, i32)))
        branch = mgpu_scf.IfOp(mgpu_arith.cmpi(mgpu_arith.CmpIPredicate.slt, blk, _i32(total, i32)))
        with mgpu_ir.InsertionPoint(branch.then_block):
            r_base = mgpu_arith.muli(mgpu_arith.divui(blk, _i32(n_c, i32)), _i32(32, i32))
            c_base = mgpu_arith.muli(mgpu_arith.remui(blk, _i32(n_c, i32)), _i32(4, i32))
            col = mgpu_arith.addi(c_base, c_grp)
            slab = mgpu_arith.addi(r_base, mgpu_arith.shli(r_blk, _i32(2, i32)))
            anchor = _swizzled_tile_anchor(slab, mgpu_arith.shli(col, _i32(2, i32)),
                                           row_bytes=row_bytes, swizzle=swizzle, i32=i32)
            raw = [mgpu_llvm.load(i32, mgpu_utils.getelementptr(
                source, [_swizzled_lane_offset(anchor, mgpu_arith.andi(
                    mgpu_arith.addi(_i32(k, i32), r_blk), _i32(3, i32)),
                    row_bytes=row_bytes, swizzle=swizzle, i32=i32)], byte)) for k in range(4)]
            for shift in (1, 2):
                bit = mgpu_arith.cmpi(mgpu_arith.CmpIPredicate.ne,
                                      mgpu_arith.andi(r_blk, _i32(shift, i32)), _i32(0, i32))
                raw = [mgpu_arith.select(bit, raw[(d - shift) & 3], raw[d]) for d in range(4)]
            store_vec(mgpu_arith.addi(col_start, col), mgpu_arith.addi(row_start, slab), raw)
            mgpu_scf.YieldOp([])
        return carry


def _emit_transposed_pieces(smem, dest, row_start, col_start, *, swizzle: int) -> None:
    """MLIR emission for `_swizzled_columns_out`: staggered `dr` bijection eliminates bank conflicts at
    `SWIZZLE_128B`, barrel-shifted in registers into coalesced `v4.u32` (16B) global stores."""
    mem_type = mgpu_ir.MemRefType(smem.type)
    element = mem_type.element_type
    elem_bits = mgpu_utils.bitwidth(element)
    dims = tuple(mem_type.shape)
    rows, cols = (dims[0] * dims[2], dims[1] * dims[3]) if len(dims) == 4 else dims
    row_bytes = cols * elem_bits // 8
    i32, i64, byte = (mgpu_ir.IntegerType.get_signless(w) for w in (32, 64, 8))
    strides, _ = mgpu_ir.MemRefType(dest.type).get_strides_and_offset()
    row_stride, col_stride = strides[0], strides[1]
    source = mgpu_utils.memref_ptr(smem)
    gmem_base = mgpu_llvm.addrspacecast(mgpu_llvm.PointerType.get(address_space=1),
                                        mgpu_utils.memref_ptr(dest))
    tid = mgpu_arith.index_cast(i32, mgpu_gpu.thread_id(mgpu_gpu.Dimension.x))
    thread = mgpu_arith.remui(tid, _i32(resources.THREADS, i32))
    warp = mgpu_arith.divui(thread, _i32(32, i32))
    lane = mgpu_arith.remui(thread, _i32(32, i32))
    c_grp = mgpu_arith.divui(lane, _i32(8, i32))
    r_blk = mgpu_arith.remui(lane, _i32(8, i32))

    def store_vec(dest_row, dest_col, words) -> None:
        _store_transposed_vec4(gmem_base, element, row_stride, col_stride, dest_row, dest_col, words,
                               i32=i32, i64=i64)

    mgpu_utils.warpgroup_barrier()
    emitter = _emit_transposed_16 if elem_bits == 16 else _emit_transposed_32
    emitter(source, store_vec, warp, c_grp, r_blk, row_start, col_start,
            rows=rows, cols=cols, row_bytes=row_bytes, swizzle=swizzle, i32=i32, byte=byte)
    mgpu_utils.warpgroup_barrier()


def _write_transposed(smem_t, piece: jax.Array, family: mosaic.Family, tiles: Tiles) -> None:
    """Write `piece` transposed into `smem_t` (`[chunk, share]`): 16-bit uses `stmatrix.trans` directly;
    fp32 casts to `WGMMA_TRANSPOSED` / `TCGEN05_TRANSPOSED` for in-register butterfly shuffle."""
    typed = piece.astype(smem_t.dtype)
    if jnp.dtype(smem_t.dtype).itemsize <= 2:
        smem_t.T[...] = typed
    else:
        layout = (plgpu.Layout.TCGEN05_TRANSPOSED if family.tensor_columns(tiles.block_n, 1)
                  else plgpu.Layout.WGMMA_TRANSPOSED)
        smem_t.T[...] = plgpu.layout_cast(typed, layout)


def _store_ring_symmetric(staging: tuple, wg, value: jax.Array, out_b, part_row, col_start, strictly_upper,
                          *, family: mosaic.Family, tiles: Tiles) -> None:
    """Store normal tile and conditional mirror tile on compiled `RingFamily` (`sm90`/`sm100`) with exact
    `buffer_waits` interleaving and zero extra SMEM."""
    share, width = value.shape
    out_dtype = out_b.dtype
    chunk = stores.staged_columns(width, (out_dtype,))
    chunks = width // chunk
    square = share == chunk
    buffers = staging[0].shape[1] if square else len(staging)

    def norm_buf(buf: int):
        return staging[0].at[wg, buf] if square else staging[buf][0].at[wg]

    def mirr_buf(buf: int):
        return staging[0].at[wg, buf] if square else staging[buf][1].at[wg]

    @pl.when(strictly_upper)
    def _upper():
        for c in range(chunks):
            piece = value[:, c * chunk:(c + 1) * chunk]
            wait_n = 0 if (buffers == 1 or (c == 0 and chunks % 2 == 1)) else 1
            plgpu.wait_smem_to_gmem(wait_n, wait_read_only=True)
            smem_n = norm_buf(0)
            family.written(smem_n, piece)
            plgpu.commit_smem()
            stores.store_rows(smem_n, out_b, part_row, 0, share, col_start + c * chunk, share,
                              min(share, TMA_ROWS))
            wait_m = 0 if buffers == 1 else 1
            plgpu.wait_smem_to_gmem(wait_m, wait_read_only=True)
            smem_m = mirr_buf(buffers - 1)
            _write_transposed(smem_m, piece, family, tiles)
            plgpu.commit_smem()
            stores.store_rows(smem_m, out_b, col_start + c * chunk, 0, chunk, part_row, chunk,
                              min(chunk, TMA_ROWS))

    @pl.when(~strictly_upper)
    def _diag():
        for c, wait in enumerate(stores.buffer_waits(chunks, buffers)):
            plgpu.wait_smem_to_gmem(wait, wait_read_only=True)
            smem_n = norm_buf(c % buffers)
            family.written(smem_n, value[:, c * chunk:(c + 1) * chunk])
            plgpu.commit_smem()
            stores.store_rows(smem_n, out_b, part_row, 0, share, col_start + c * chunk, share,
                              min(share, TMA_ROWS))


def _store_cp_async_symmetric(value: jax.Array, out_b, part_row, col_start, strictly_upper, *,
                              family: mosaic.Family) -> None:
    """Store normal tile and conditional mirror tile on compiled `CpAsync` (`sm80`/`sm86`/`sm89`/`sm120`)
    from one swizzled SMEM staging pass."""
    share = value.shape[0]
    plgpu.wait_gmem_to_smem(0)

    def copied(smem):
        plgpu.store(smem, value.astype(smem.dtype), optimized=False)
        swizzle = swizzling.swizzle_of(smem)
        if swizzle == swizzling.UNSWIZZLED:
            stores.store_rows(smem, out_b, part_row, 0, share, col_start, share, min(share, TMA_ROWS),
                              copy=swizzling.copied,
                              widest=resources.copied_rows(smem.shape[1], smem.dtype))
        else:
            stores.swizzled_rows_out(smem, out_b, part_row, 0, share, col_start, swizzle=swizzle)

        @pl.when(strictly_upper)
        def _mirror():
            _swizzled_columns_out(smem, out_b, part_row, col_start, swizzle=swizzle)

    pl.run_scoped(copied, family.output_staging(((value.shape, out_b.dtype),), 1)[0])
    plgpu.wait_gmem_to_smem(0)


def _store_interpreted_symmetric(value: jax.Array, out_b, part_row, col_start, strictly_upper, *,
                                 store, family: mosaic.Family) -> None:
    """Store normal + mirror on TPU (`VmemRings`) or CPU interpreter."""
    share, width = value.shape
    unit = row_unit(family, out_b.dtype)
    if not family.budgets_registers:
        store(0, value, out_b, (part_row, 0, share, col_start, share, unit))

        @pl.when(strictly_upper)
        def _tpu_mirror():
            store(0, value.T, out_b, (col_start, 0, width, part_row, width, unit))
        return
    # CPU interpreter on GPU families: direct ref writes (handles non-square consumer shares cleanly).
    typed = value.astype(out_b.dtype)
    out_b[pl.ds(part_row, share), pl.ds(col_start, width)] = typed

    @pl.when(strictly_upper)
    def _cpu_mirror():
        out_b[pl.ds(col_start, width), pl.ds(part_row, share)] = typed.T


def _syrk_finish_tile(wg, tile, accs, store, *, x_gmem, o_gmem, coeffs_smem, side_smems, side_barriers,
                      custom_staging, columns: int, family: mosaic.Family, tiles: Tiles,
                      interpret: bool) -> None:
    """Apply optional fused symmetric polynomial epilogue and store upper + mirror tiles for one
    super-tile."""
    batch, ti, tj, row_start, col_tile = tile
    part_row = row_start + tiles.share_row(wg)
    out_b = o_gmem.at[batch]
    strictly_upper = ti < tj
    poly = None
    if coeffs_smem is not None:
        src = x_gmem.at[batch, pl.ds(part_row, tiles.share), pl.ds(col_tile, columns)]
        barrier = side_barriers[0].at[wg] if side_barriers else None
        family.loaded([(src, side_smems[0].at[wg], barrier)], axes=(('rows', 'columns'),))
        poly = (coeffs_smem[0], coeffs_smem[1], coeffs_smem[2])
    for start, (chunk,) in family.chunks(accs):
        tile_out = chunk
        if poly is not None:
            c0, c1, c2 = poly
            a_tile = family.side_tile(side_smems[0].at[wg].at[:, slice(start, start + tile_out.shape[-1])],
                                      ('rows', 'columns')).astype(jnp.float32)
            tile_out = c2 * tile_out + c1 * a_tile
            r_idx = part_row + family.iota(tile_out.shape, 0)
            c_idx = col_tile + start + family.iota(tile_out.shape, 1)
            tile_out = jnp.where(r_idx == c_idx, tile_out + c0, tile_out)
        col = col_tile + start
        if _uses_custom_staging(family, interpret):
            _store_ring_symmetric(custom_staging, wg, tile_out, out_b, part_row, col, strictly_upper,
                                  family=family, tiles=tiles)
        elif family.stores_through_registers and not interpret:
            _store_cp_async_symmetric(tile_out, out_b, part_row, col, strictly_upper, family=family)
        else:
            _store_interpreted_symmetric(tile_out, out_b, part_row, col, strictly_upper, store=store,
                                         family=family)


def syrk(x: jax.Array, *, tiles: Tiles, transpose: bool = False, out_dtype, multiprocessors: int,
         interpret: bool | plgpu.InterpretGPUParams = False, poly_coeffs: jax.Array | None = None,
         registers: int | None = None, shared: int | None = None) -> jax.Array:
    """Batched symmetric Gram `X @ X^T` (`transpose=False`, `x` `[g, m, k]`), `X^T @ X` (`transpose=True`,
    `x` `[g, k, m]`), or fused symmetric polynomial `c0 * I + c1 * A + c2 * (A @ A)` (`poly_coeffs` `[3]`,
    `transpose=False`, `m == k`): computes upper-triangle square super-tiles and mirrors strictly upper tiles
    (row 11941)."""
    if x.ndim != 3:
        raise ValueError(f"syrk expects 3D [batches, m, k] (or [batches, k, m]), got {x.shape}")
    out_dtype = jnp.dtype(out_dtype)
    if not supports_out_dtype(out_dtype):
        raise ValueError(f"syrk supports 16-bit and 32-bit float outputs, got {out_dtype}")
    family = family_of(tiles, x.dtype)
    if not square_super_tile(tiles, family):
        raise ValueError(f"syrk requires a square super-tile (tile_rows == tile_columns), got {tiles}")
    if poly_coeffs is not None and (transpose or x.shape[1] != x.shape[2]):
        raise ValueError(f"symmetric polynomial expects square untransposed [g, m, m], got {x.shape}")
    g = x.shape[0]
    m, k = (x.shape[2], x.shape[1]) if transpose else (x.shape[1], x.shape[2])
    tile_side = tile_rows(tiles, family)
    m_pad = -(-m // tile_side) * tile_side
    # Polynomial shares `x` with the epilogue side load `[part_row:part_row+share, col_tile:col_tile+cols]`,
    # so its column axis must cover the padded square output grid `m_pad` as well as whole `block_k` steps.
    k_target = max(k, m_pad) if poly_coeffs is not None else k
    k_pad = -(-k_target // tiles.block_k) * tiles.block_k
    if m_pad != m or k_pad != k:
        pad = (((0, 0), (0, k_pad - k), (0, m_pad - m)) if transpose
               else ((0, 0), (0, m_pad - m), (0, k_pad - k)))
        x = jnp.pad(x, pad)
    coeffs = None if poly_coeffs is None else jnp.asarray(poly_coeffs, jnp.float32).reshape(3)
    out = _syrk_mainloop(x, coeffs, (g, m_pad, k_pad), tiles=tiles, transpose=transpose, out_dtype=out_dtype,
                         multiprocessors=multiprocessors, interpret=interpret,
                         polynomial=coeffs is not None, registers=registers, shared=shared)
    return out[:, :m, :m] if m_pad != m else out


@tuning.lowered_once(('shape', 'tiles', 'transpose', 'out_dtype', 'multiprocessors', 'interpret',
                      'polynomial', 'registers', 'shared'))
def _syrk_mainloop(x: jax.Array, coeffs: jax.Array | None, shape: tuple[int, int, int], *, tiles: Tiles,
                   transpose: bool, out_dtype, multiprocessors: int, interpret, polynomial: bool = False,
                   registers: int | None = None, shared: int | None = None) -> jax.Array:
    """Compiled/interpreted persistent SYRK upper-triangle mainloop."""
    del polynomial
    has_poly = coeffs is not None
    family = family_of(tiles, x.dtype)
    g, m, k = shape
    block_m, block_n, block_k = tiles.block_m, tiles.block_n, tiles.block_k
    columns = tile_columns(tiles, family)
    tile_side = tile_rows(tiles, family)
    consumers, pair = tiles.warpgroups, tiles.cluster
    part = (tiles.share, columns)
    side = m // tile_side
    tri = side * (side + 1) // 2
    total_tiles = g * tri
    steps = k // block_k
    allocations = syrk_allocations(tiles, family, dtype=x.dtype, out_dtype=out_dtype, transpose=transpose,
                                   polynomial=has_poly, interpret=bool(interpret))

    def body(*gmems):
        if has_poly:
            coeffs_smem, x_gmem, o_gmem = gmems
        else:
            coeffs_smem, (x_gmem, o_gmem) = None, gmems

        def schedule(index):
            batch, ti, tj = upper_tile(index[0], side)
            cta = mosaic.cluster_index(pair)
            row_tile, col_tile = ti * tile_side, tj * tile_side
            row_start = row_tile + cta * block_m
            x_b = x_gmem.at[batch]
            if family.pairs_product:
                def pair_step(j):
                    return rings.Paired((j, 0), 1) if transpose else rings.Paired((0, j), 0)

                row_span, col_span = pl.ds(row_tile, tile_side), pl.ds(col_tile, columns)
                row_ref = x_b.at[:, row_span] if transpose else x_b.at[row_span]
                col_ref = x_b.at[:, col_span] if transpose else x_b.at[col_span]
                views = {'rows': (row_ref, pair_step), 'matrix': (col_ref, pair_step)}
            elif transpose:
                views = {'rows': (x_b.at[:, pl.ds(row_start, block_m)], lambda j: (j, 0)),
                         'matrix': (x_b.at[:, pl.ds(col_tile, block_n)], lambda j: (j, 0))}
            else:
                views = {'rows': (x_b.at[pl.ds(row_start, block_m)], lambda j: (0, j)),
                         'matrix': (x_b.at[pl.ds(col_tile, block_n)], lambda j: (0, j))}
            return jnp.ones((), jnp.bool_), steps, views, (batch, ti, tj, row_start, col_tile)

        def scope(side_smems, side_barriers, custom_staging, *, streams):
            def issue(wg, tile, walk, *, active):
                del tile, active
                mine_rows = pl.ds(tiles.share_row(wg), tiles.share)

                def step(index, slots, carry, accs):
                    del index
                    left = (slots['rows'].at[:, mine_rows].transpose((1, 0)) if transpose
                            else slots['rows'].at[mine_rows])
                    right = slots['matrix'] if transpose else slots['matrix'].transpose((1, 0))
                    return carry, (family.products(accs[0], left, right),)

                _, (acc,) = walk(step, None, (part,))
                return (acc,)

            def finish(wg, tile, accs, store, *, active):
                del active
                _syrk_finish_tile(wg, tile, accs, store, x_gmem=x_gmem, o_gmem=o_gmem,
                                  coeffs_smem=coeffs_smem, side_smems=side_smems,
                                  side_barriers=side_barriers, custom_staging=custom_staging,
                                  columns=columns, family=family, tiles=tiles, interpret=bool(interpret))

            persist_outs = allocations.outputs if not family.budgets_registers else ()
            family.persist(streams, (total_tiles,), schedule, issue, finish, consumers=consumers,
                           axis="programs", outputs=persist_outs, interpret=bool(interpret),
                           in_flight=family.in_flight, accumulators=(part,),
                           producer=tiles.producers.of(SYRK))

        family.scoped(scope, *allocations.staged,
                      streams=family.streams(allocations.blocks, tiles.stages, tiles.sharing,
                                             multicast=multicast(tiles, family, allocations.shared),
                                             joined=operand_streams(rewritten=False)))

    out_type = jax.ShapeDtypeStruct((g, m, m), out_dtype)
    operands = (coeffs, x) if has_poly else (x,)
    programs = max(total_tiles, 1) if family.walks_scoped and not interpret else multiprocessors
    # Upper-triangle FLOPs: g * (tri * tile_side^2) * 2k (~half of dense 2*g*m*m*k).
    flops = 2 * g * tri * tile_side * tile_side * k
    kernel = family.kernel(body, out_type=out_type, grid=(programs // pair,), grid_names=("programs",),
                           consumers=consumers, interpret=interpret, approx_math=APPROX_MATH, name=SYRK,
                           cluster=pair, resident=tiles.programs, scalars=1 if has_poly else 0,
                           registers=registers, producer=tiles.producers.of(SYRK), shared=shared, moves=MOVES,
                           cost=unit_cost.estimate(flops, *operands, out_type))
    return kernel(*operands)


def syrk_registers(tiles: Tiles, dtype, out_dtype, *, transpose: bool, polynomial: bool,
                   target) -> int | None:
    """Consumer thread register count for `syrk` at `tiles` on `target` (`resources.tile_registers` +
    shared `pricing.epilogue_registers`, row 11982)."""
    family = family_of(tiles, dtype, target)
    if not family.budgets_registers or generation.tpu(target):
        return None
    columns = tile_columns(tiles, family)
    plain = resources.tile_registers(family, tiles.sharing, (tiles.block_m, columns, tiles.block_k), dtype,
                                     stored=(out_dtype,), rhs_transposed=not transpose)
    epilogue = epilogue_registers(family, tiles.share, columns, (out_dtype,),
                                  tile_count=1 if polynomial else 0, has_hook=polynomial)
    return plain + epilogue


def syrk_launch_of(tiles: Tiles, shape: tuple[int, int, int], dtype, out_dtype, *, transpose: bool,
                   polynomial: bool, target, interpret: bool = False) -> resources.Launch:
    """Resource planner `Launch` for `syrk` at `tiles` on `target` (`resources.plan`)."""
    _, m, k = shape
    family = family_of(tiles, dtype, target)
    tile_side = tile_rows(tiles, family)
    m_pad = -(-m // tile_side) * tile_side
    k_target = max(k, m_pad) if polynomial else k
    steps = -(-k_target // tiles.block_k)

    def scope(stages: int) -> resources.Scope:
        staged = dataclasses.replace(tiles, stages=stages)
        return syrk_allocations(staged, family, dtype=dtype, out_dtype=out_dtype, transpose=transpose,
                                polynomial=polynomial, interpret=interpret).scope(family, staged, steps)

    shared = resources.program_shared(generation.shared_budget(target), tiles.programs)
    most = min(shared // ((tiles.block_m + tiles.block_n) * tiles.block_k), max(2, steps))
    most = family.ring_slots or most
    regs = syrk_registers(tiles, dtype, out_dtype, transpose=transpose, polynomial=polynomial, target=target)
    return resources.Launch(scope, tiles.warpgroups, regs, most=most, producer=tiles.producers.of(SYRK))


def derive_syrk(shape: tuple[int, int, int], dtype, out_dtype, target, *, transpose: bool = False,
                polynomial: bool = False, interpret: bool = False,
                refused: Callable[[Tiles], bool] | None = None, ba=None) -> Tiles | None:
    """Derive a legal square-super-tile `Tiles` for `syrk` on `target` using `derivation.derived` and the
    single `MOSAIC_SYRK.candidate` predicate (row 11982)."""
    if not supports_out_dtype(out_dtype):
        return None
    g, m, k = shape
    groups = bundles.groups(target.generation, dtype, stated=True, tpu=generation.tpu(target))
    if not groups:
        return None
    if ba is None:
        x_shape = (g, k, m) if transpose else (g, m, k)
        poly = jax.ShapeDtypeStruct((3,), jnp.float32) if polynomial else None
        ba = MOSAIC_SYRK.bind(jax.ShapeDtypeStruct(x_shape, dtype), poly, transpose=transpose,
                              out_dtype=jnp.dtype(out_dtype).name, interpret=interpret)
    multiprocessors = target.multiprocessors or 1
    for tried in (tried for layouts in groups for tried in derivation.paired_first(layouts, m)):
        family0 = tried[0]
        super_unit = family0.role_rows * min(family0.consumers, CONSUMERS) * family0.cluster
        base_unit = math.lcm(family0.role_rows, family0.column_unit(dtype))
        align = min(super_unit, max(base_unit, 1 << max(0, m - 1).bit_length()))
        align = -(-align // base_unit) * base_unit
        aligned_m = -(-m // align) * align
        step_unit = family0.depths(dtype)[-1]
        aligned_k = -(-(max(k, aligned_m) if polynomial else k) // step_unit) * step_unit

        def candidate(tile: Tiles) -> bool:
            if refused is not None and refused(tile):
                return False
            return MOSAIC_SYRK.candidate(ba, tile, target)

        def launch_of(tile: Tiles, *, rows_m: int = aligned_m, depth_k: int = aligned_k) -> resources.Launch:
            return syrk_launch_of(tile, (g, rows_m, depth_k), dtype, out_dtype, transpose=transpose,
                                  polynomial=polynomial, target=target, interpret=interpret)

        tile = derivation.derived(tried, rows=aligned_m, depth=aligned_k, width=aligned_m, dtype=dtype,
                                  multiprocessors=multiprocessors, target=target, candidate=candidate,
                                  launch_of=launch_of, grouped=False, dense=True, ragged=True)
        if tile is not None:
            return tile
    return None


@dataclasses.dataclass(frozen=True)
class MosaicSyrk(TilesTemplate):
    """SYRK (`syrk`) on the shared `TilesTemplate`/`KernelOp` certification path (`report.launching`,
    `report.refused`), deriving legal square super-tiles and stepping past spilling tiles."""

    KERNEL: ClassVar[str] = SYRK
    config_cls: ClassVar[type[Tiles]] = Tiles

    def _supports_ba(self, ba) -> bool:
        return supports_out_dtype(ba.arguments["out_dtype"])

    def _shape(self, ba) -> tuple[int, int, int]:
        x = ba.arguments["x"]
        m, k = (x.shape[2], x.shape[1]) if ba.arguments["transpose"] else (x.shape[1], x.shape[2])
        return x.shape[0], m, k

    def launch_of(self, ba, config: Tiles, target=None) -> resources.Launch:
        target = generation.gpu_target() if target is None else target
        return syrk_launch_of(config, self._shape(ba), ba.arguments["x"].dtype, ba.arguments["out_dtype"],
                              transpose=ba.arguments["transpose"],
                              polynomial=ba.arguments["poly_coeffs"] is not None,
                              target=target, interpret=ba.arguments["interpret"])

    def registers(self, ba, config: Tiles, target=None) -> int | None:
        target = generation.gpu_target() if target is None else target
        return syrk_registers(config, ba.arguments["x"].dtype, ba.arguments["out_dtype"],
                              transpose=ba.arguments["transpose"],
                              polynomial=ba.arguments["poly_coeffs"] is not None, target=target)

    def candidate(self, ba, config: Tiles, target) -> bool:
        x = ba.arguments["x"]
        out_dtype = ba.arguments["out_dtype"]
        if not supports_out_dtype(out_dtype) or report.refused(self, ba, config, target):
            return False
        groups = bundles.groups(target.generation, x.dtype, stated=True, tpu=generation.tpu(target))
        family = next((b for group in groups for b in group if b.name == config.family), None)
        if family is None or config.cluster > family.cluster or not pair_takes(config, family):
            return False
        if (not square_super_tile(config, family) or config.splits != 1 or config.pingpong
                or config.swapped or (family.pairs_product and ba.arguments["interpret"])):
            return False

        def blocks():
            return syrk_allocations(
                config, family, dtype=x.dtype, out_dtype=out_dtype, transpose=ba.arguments["transpose"],
                polynomial=ba.arguments["poly_coeffs"] is not None,
                interpret=ba.arguments["interpret"]).blocks

        if not granular_tile(family, config, x.dtype, target=target, stream_blocks=blocks):
            return False
        specialized = family.specialized(config.share, tile_columns(config, family))
        return not (ba.arguments["transpose"] and specialized.operands_in_registers
                    and config.block_k % specialized.transposed_rows != 0)

    def derived(self, ba, target) -> Tiles | None:
        return derive_syrk(self._shape(ba), ba.arguments["x"].dtype, ba.arguments["out_dtype"], target,
                           transpose=ba.arguments["transpose"],
                           polynomial=ba.arguments["poly_coeffs"] is not None,
                           interpret=ba.arguments["interpret"],
                           refused=lambda candidate: report.refused(self, ba, candidate, target), ba=ba)

    def _fwd(self, x: jax.Array, poly_coeffs: jax.Array | None = None, *, transpose: bool = False,
             out_dtype: str = "bfloat16", interpret: bool = False, config: Tiles,
             return_residuals: bool = False):
        del return_residuals
        target = generation.gpu_target()
        bound = self.bind(x, poly_coeffs, transpose=transpose, out_dtype=out_dtype, interpret=interpret)
        mp = config.cluster if interpret else (target.multiprocessors or 1) * config.programs
        regs = self.registers(bound, config, target)
        shared = regs and self.staged_bytes(bound, config, target)
        out = syrk(x, tiles=config, transpose=transpose, out_dtype=out_dtype, multiprocessors=mp,
                   interpret=grouped_launch.INTERPRETED if interpret else False, poly_coeffs=poly_coeffs,
                   registers=regs, shared=shared)
        return out, None


MOSAIC_SYRK = MosaicSyrk()
"""The singleton Mosaic SYRK launch on the shared `Template`/`KernelOp` certification path."""


MEASURED_GRAM_CELLS: frozenset[tuple[str, tuple[int, int, int], bool, str, str]] = frozenset({
    (gen, shape, transpose, "bfloat16", "bfloat16")
    for gen in ("sm80", "sm90")
    for shape in ((28, 2560, 2560), (28, 2560, 4096), (36, 4096, 4096), (36, 4096, 14336))
    for transpose in (False, True)
})
"""Exact `(generation, shape, transpose, dtype, out_dtype)` stacked Gram cells proven 1.52-1.98x faster
than XLA with bitwise-equal digests in `M-NG-45-a100` (sm80, 8/8) and `M-NG-45b-gh200` (sm90, 8/8;
row 11941)."""


def measured_gram(target: generation.GpuTarget, shape: tuple[int, int, int], dtype, *,
                  transpose: bool, out_dtype=None) -> bool:
    """Whether symmetric Gram `(target, shape, transpose, dtype, out_dtype)` is admitted by real GPU
    receipts (`MEASURED_GRAM_CELLS`, row 11941)."""
    if generation.tpu(target):
        return False
    in_name = jnp.dtype(dtype).name
    out_name = in_name if out_dtype is None else jnp.dtype(out_dtype).name
    return (target.generation, tuple(shape), bool(transpose), in_name, out_name) in MEASURED_GRAM_CELLS
