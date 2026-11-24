"""The Mosaic GPU lowering of a split walk's merge (`merged.merged_padded`, row 9795): one launch on NVIDIA
sm80+ GPUs reading the walk's padded slices in place, a program a (batch, query, key head) row and a block
of its channels, zero shared memory and no warp shuffles. Its loads, stores and output hooks address each
row's channels by one geometry (`_thread_col`, `_i64_offset`); the law it runs on them is `merged`'s."""

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src.lib.mlir.dialects import llvm as mgpu_llvm, scf as mgpu_scf
from jax.experimental import pallas as pl
from jax.experimental.mosaic.gpu import fragmented_array as mgpu_fragments, utils as mgpu_utils
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend.mlir import ir as mgpu_ir
from jax.extend.mlir.dialects import arith as mgpu_arith, vector as mgpu_vector

from dew.algebra import windows
from dew.nn.kernels.templates import attention_algebra as algebra, mosaic
from dew.nn.kernels.templates.attention import Kind

THREADS = 4 * 32
"""A program's threads: Pallas Triton's default of four warps (`CompilerParams.num_warps`), the merge's."""


def split_extra(extra: Sequence, output_kinds: tuple[Kind, ...]) -> tuple[tuple, tuple]:
    """Split `extra` operands into optional `(init_m, init_t, init_acc)` and `output_params` arrays."""
    n_init = len(extra) - len(output_kinds)
    assert n_init in (0, 3), f"expected 0 or 3 initial-state operands, got {n_init}"
    return tuple(extra[:n_init]), tuple(extra[n_init:])


def _tile_layout(tile_block: int):
    """Thread-local slice axis, column-sharded channel axis (`tile_block >= 64` a power of two): zero SMEM
    and zero warp shuffles for every reduction and halving tree along axis 0 (row 9795)."""
    warps = mgpu_fragments.WARPS_IN_WARPGROUP
    lanes = mgpu_fragments.WARP_SIZE
    if tile_block >= THREADS:
        tiling = mgpu_fragments.Tiling(((1, THREADS), (lanes,), (1,)))
        return plgpu.Layout.TILED(tiling, warp_dims=(-3,), lane_dims=(-2,), vector_dim=-1)
    tiling = mgpu_fragments.Tiling(((1, tile_block), (lanes,), (1,)))
    return plgpu.Layout.TILED(
        tiling, warp_dims=(mgpu_fragments.Replicated(warps // (tile_block // lanes)), -3),
        lane_dims=(-2,), vector_dim=-1)

def _tile_1d_layout(tile_layout):
    """1D channel-sharded layout derived from `_tile_layout(block)` by reducing its leading slice axis."""
    reduced = tile_layout.to_mgpu().reduce((0,))
    return plgpu.Layout.TILED(reduced.tiling, warp_dims=reduced.warp_dims,
                              lane_dims=reduced.lane_dims, vector_dim=reduced.vector_dim)

def _gmem_ptr(ref):
    """Global-memory address-space-1 pointer to `ref`'s base."""
    return mgpu_llvm.addrspacecast(mgpu_llvm.PointerType.get(address_space=1), mgpu_utils.memref_ptr(ref))


def _i64_offset(indices: Sequence, strides: Sequence[int]):
    """64-bit element offset `sum(index * stride)` from i32 MLIR indices and static integer strides."""
    i64 = mgpu_ir.IntegerType.get_signless(64)
    total = mgpu_utils.c(0, i64)
    for index, stride in zip(indices, strides, strict=True):
        term = mgpu_arith.muli(mgpu_arith.extsi(i64, index), mgpu_utils.c(stride, i64))
        total = mgpu_arith.addi(total, term)
    return total


def _predicated_load(base, offset, cond, element, zero):
    """Predicated global load returning `zero` when `cond` is false, avoiding out-of-bounds memory access."""
    branch = mgpu_scf.IfOp(cond, [element], has_else=True)
    with mgpu_ir.InsertionPoint(branch.then_block):
        ptr = mgpu_utils.getelementptr(base, [offset], element)
        mgpu_scf.YieldOp([mgpu_llvm.load(element, ptr)])
    assert branch.else_block is not None
    with mgpu_ir.InsertionPoint(branch.else_block):
        mgpu_scf.YieldOp([zero])
    return branch.results[0]


def _thread_col(block: int):
    """Warpgroup thread index `tid` and its unique column offset within `min(block, THREADS)`."""
    i32 = mgpu_ir.IntegerType.get_signless(32)
    unique = min(block, THREADS)
    tid = mgpu_arith.remui(mgpu_utils.thread_idx(), mgpu_utils.c(THREADS, i32))
    col = tid if unique == THREADS else mgpu_arith.remui(tid, mgpu_utils.c(unique, i32))
    return tid, col, unique


def _columns(tile_block: int, block: int, width: int, c_start, stride: int):
    """Each column a thread holds of a row's block of channels (`_thread_col`'s, `tile_block // unique` of
    them): whether it lies in the block and the row's `width`, and its element offset along the row at
    `stride`; the one geometry the merge's stores and its output hooks' loads address a row by."""
    i32, i64 = mgpu_ir.IntegerType.get_signless(32), mgpu_ir.IntegerType.get_signless(64)
    _, tid_col, unique = _thread_col(tile_block)
    block_c, width_c = mgpu_utils.c(block, i32), mgpu_utils.c(width, i32)
    for rep in range(tile_block // unique):
        local_col = mgpu_arith.addi(tid_col, mgpu_utils.c(rep * unique, i32))
        col = mgpu_arith.addi(c_start, local_col)
        ok = mgpu_arith.andi(mgpu_arith.cmpi(mgpu_arith.CmpIPredicate.slt, local_col, block_c),
                             mgpu_arith.cmpi(mgpu_arith.CmpIPredicate.slt, col, width_c))
        yield ok, mgpu_arith.muli(mgpu_arith.extsi(i64, col), mgpu_utils.c(stride, i64))


def _mgpu_load_chunk(padded_ref, stacked_ref, b, q, h, start, c_start, *, chunk: int, block: int,
                     tile_block: int, slices: int, read_width: int, tile_layout):
    """Predicated coalesced GMEM load of one `[chunk, tile_block]` output chunk (gated by each column's place
    in the block and the row's `read_width`, `_columns`, and by `sl < slices`) and its replicated
    `[chunk, tile_block]` maxima and totals (row 9795): its columns' validity of the block's start and the
    thread's place, which a kernel's count reads lane by lane (`inline_work`)."""
    scalar = plgpu.Layout.WG_SPLAT(())
    out_ty = plgpu.ShapeDtypeStruct((chunk, tile_block), padded_ref.dtype, layout=tile_layout)
    stat_ty = plgpu.ShapeDtypeStruct((chunk, tile_block), stacked_ref.dtype, layout=tile_layout)

    def body(ctx, p_ref, s_ref, b_v, q_v, h_v, start_v, c_start_v):
        del ctx
        i32, i64 = mgpu_ir.IntegerType.get_signless(32), mgpu_ir.IntegerType.get_signless(64)
        b_i, q_i, h_i = (v.registers.item() for v in (b_v, q_v, h_v))
        st_i, cs_i = start_v.registers.item(), c_start_v.registers.item()

        p_elem = mgpu_ir.MemRefType(p_ref.type).element_type
        p_strides, _ = mgpu_ir.MemRefType(p_ref.type).get_strides_and_offset()
        p_base = _gmem_ptr(p_ref)
        p_row = _i64_offset((b_i, q_i, h_i), (p_strides[0], p_strides[2], p_strides[3]))
        p_zero = mgpu_arith.constant(p_elem, 0.0)
        p_vec = mgpu_ir.VectorType.get((1,), p_elem)
        p_mgpu = tile_layout.to_mgpu()
        p_regs = np.empty(p_mgpu.registers_shape((chunk, tile_block)), dtype=object)

        s_elem = mgpu_ir.MemRefType(s_ref.type).element_type
        s_strides, _ = mgpu_ir.MemRefType(s_ref.type).get_strides_and_offset()
        s_base = _gmem_ptr(s_ref)
        m_row = _i64_offset((b_i, h_i, q_i), (s_strides[1], s_strides[2], s_strides[4]))
        t_row = mgpu_arith.addi(m_row, mgpu_utils.c(s_strides[0], i64))
        s_zero = mgpu_arith.constant(s_elem, 0.0)
        s_vec = mgpu_ir.VectorType.get((1,), s_elem)
        m_regs = np.empty(p_mgpu.registers_shape((chunk, tile_block)), dtype=object)
        t_regs = np.empty(p_mgpu.registers_shape((chunk, tile_block)), dtype=object)
        slices_c = mgpu_utils.c(slices, i32)
        for s_idx in range(chunk):
            sl = mgpu_arith.addi(st_i, mgpu_utils.c(s_idx, i32))
            sl_ok = mgpu_arith.cmpi(mgpu_arith.CmpIPredicate.slt, sl, slices_c)
            sl64 = mgpu_arith.extsi(i64, sl)
            s_off = mgpu_arith.muli(sl64, mgpu_utils.c(s_strides[3], i64))
            m_val = _predicated_load(s_base, mgpu_arith.addi(m_row, s_off), sl_ok, s_elem, s_zero)
            t_val = _predicated_load(s_base, mgpu_arith.addi(t_row, s_off), sl_ok, s_elem, s_zero)
            m_regs[s_idx, ...] = mgpu_vector.from_elements(s_vec, [m_val])
            t_regs[s_idx, ...] = mgpu_vector.from_elements(s_vec, [t_val])
            p_sl_base = mgpu_arith.addi(p_row, mgpu_arith.muli(sl64, mgpu_utils.c(p_strides[1], i64)))
            columns = _columns(tile_block, block, read_width, cs_i, p_strides[4])
            for rep, (col_bit, col_off) in enumerate(columns):
                ok = mgpu_arith.andi(sl_ok, col_bit)
                loaded = _predicated_load(p_base, mgpu_arith.addi(p_sl_base, col_off), ok, p_elem, p_zero)
                p_regs[s_idx, rep, ...] = mgpu_vector.from_elements(p_vec, [loaded])

        return (mgpu_fragments.FragmentedArray(_registers=p_regs, _layout=p_mgpu, _is_signed=None),
                mgpu_fragments.FragmentedArray(_registers=m_regs, _layout=p_mgpu, _is_signed=None),
                mgpu_fragments.FragmentedArray(_registers=t_regs, _layout=p_mgpu, _is_signed=None))

    return plgpu.inline_mgpu(
        arg_types=(plgpu.RefType(), plgpu.RefType(), *(scalar,) * 5),
        return_type=(out_ty, stat_ty, stat_ty),
    )(body)(padded_ref, stacked_ref, b, q, h, start, c_start)


def _mgpu_load_initial(init_m_ref, init_t_ref, init_acc_ref, b, h, q, c_start, *, block: int,
                       tile_block: int, width: int, tile_1d):
    """Predicated GMEM load of one row's `Initial` state (`[tile_block]` acc, replicated m and total)."""
    scalar = plgpu.Layout.WG_SPLAT(())
    acc_ty = plgpu.ShapeDtypeStruct((tile_block,), init_acc_ref.dtype, layout=tile_1d)
    stat_ty = plgpu.ShapeDtypeStruct((tile_block,), init_m_ref.dtype, layout=tile_1d)

    def body(ctx, im_ref, it_ref, ia_ref, b_v, h_v, q_v, c_start_v):
        del ctx
        i32, i64 = mgpu_ir.IntegerType.get_signless(32), mgpu_ir.IntegerType.get_signless(64)
        b_i, h_i, q_i, cs_i = (v.registers.item() for v in (b_v, h_v, q_v, c_start_v))
        _, tid_col, unique = _thread_col(tile_block)
        reps = tile_block // unique

        s_elem = mgpu_ir.MemRefType(im_ref.type).element_type
        s_vec = mgpu_ir.VectorType.get((1,), s_elem)
        t1_mgpu = tile_1d.to_mgpu()
        reg_shape = t1_mgpu.registers_shape((tile_block,))
        im_strides, _ = mgpu_ir.MemRefType(im_ref.type).get_strides_and_offset()
        it_strides, _ = mgpu_ir.MemRefType(it_ref.type).get_strides_and_offset()
        m_val = mgpu_llvm.load(s_elem, mgpu_utils.getelementptr(
            _gmem_ptr(im_ref), [_i64_offset((b_i, h_i, q_i), im_strides)], s_elem))
        t_val = mgpu_llvm.load(s_elem, mgpu_utils.getelementptr(
            _gmem_ptr(it_ref), [_i64_offset((b_i, h_i, q_i), it_strides)], s_elem))
        m_regs = np.full(reg_shape, mgpu_vector.from_elements(s_vec, [m_val]), dtype=object)
        t_regs = np.full(reg_shape, mgpu_vector.from_elements(s_vec, [t_val]), dtype=object)

        a_elem = mgpu_ir.MemRefType(ia_ref.type).element_type
        a_strides, _ = mgpu_ir.MemRefType(ia_ref.type).get_strides_and_offset()
        a_base = _gmem_ptr(ia_ref)
        a_row = _i64_offset((b_i, h_i, q_i), a_strides[:3])
        a_zero = mgpu_arith.constant(a_elem, 0.0)
        a_vec = mgpu_ir.VectorType.get((1,), a_elem)
        a_regs = np.empty(reg_shape, dtype=object)
        block_c, width_c = mgpu_utils.c(block, i32), mgpu_utils.c(width, i32)
        for rep in range(reps):
            local_col = mgpu_arith.addi(tid_col, mgpu_utils.c(rep * unique, i32))
            col = mgpu_arith.addi(cs_i, local_col)
            ok = mgpu_arith.andi(
                mgpu_arith.cmpi(mgpu_arith.CmpIPredicate.slt, local_col, block_c),
                mgpu_arith.cmpi(mgpu_arith.CmpIPredicate.slt, col, width_c))
            col_off = mgpu_arith.muli(mgpu_arith.extsi(i64, col), mgpu_utils.c(a_strides[3], i64))
            loaded = _predicated_load(a_base, mgpu_arith.addi(a_row, col_off), ok, a_elem, a_zero)
            a_regs[rep, ...] = mgpu_vector.from_elements(a_vec, [loaded])

        return (mgpu_fragments.FragmentedArray(_registers=a_regs, _layout=t1_mgpu, _is_signed=None),
                mgpu_fragments.FragmentedArray(_registers=m_regs, _layout=t1_mgpu, _is_signed=None),
                mgpu_fragments.FragmentedArray(_registers=t_regs, _layout=t1_mgpu, _is_signed=None))

    return plgpu.inline_mgpu(
        arg_types=(plgpu.RefType(), plgpu.RefType(), plgpu.RefType(), *(scalar,) * 4),
        return_type=(acc_ty, stat_ty, stat_ty),
    )(body)(init_m_ref, init_t_ref, init_acc_ref, b, h, q, c_start)


def _mgpu_store_finished(merged_ref, acc, b, out_t, out_h, c_start, *, block: int, tile_block: int,
                         width: int, acc_layout):
    """Predicated coalesced GMEM store of one `[block]` slice of finished output (`[tile_block]` or
    row 0 of `[2, tile_block]`, row 9795)."""
    scalar = plgpu.Layout.WG_SPLAT(())

    def body(ctx, dest_ref, acc_v, b_v, t_v, h_v, c_start_v):
        del ctx
        pos0 = mgpu_ir.DenseI64ArrayAttr.get([0])
        i32 = mgpu_ir.IntegerType.get_signless(32)
        b_i, t_i, h_i, cs_i = (v.registers.item() for v in (b_v, t_v, h_v, c_start_v))
        tid, _, unique = _thread_col(tile_block)
        reps = tile_block // unique
        elem = mgpu_ir.MemRefType(dest_ref.type).element_type
        strides, _ = mgpu_ir.MemRefType(dest_ref.type).get_strides_and_offset()
        base = _gmem_ptr(dest_ref)
        row_base = _i64_offset((b_i, t_i, h_i), strides[:3])
        active_tid = mgpu_arith.cmpi(mgpu_arith.CmpIPredicate.slt, tid, mgpu_utils.c(min(block, unique), i32))
        acc_regs = (acc_v.registers[0] if len(acc_v.shape) == 2 else acc_v.registers).reshape(reps)
        for rep, (ok, col_off) in enumerate(_columns(tile_block, block, width, cs_i, strides[3])):
            branch = mgpu_scf.IfOp(mgpu_arith.andi(active_tid, ok))
            with mgpu_ir.InsertionPoint(branch.then_block):
                stored = mgpu_vector.extract(acc_regs[rep], dynamic_position=[], static_position=pos0)
                off = mgpu_arith.addi(row_base, col_off)
                mgpu_llvm.store(stored, mgpu_utils.getelementptr(base, [off], elem))
                mgpu_scf.YieldOp([])

    plgpu.inline_mgpu(
        arg_types=(plgpu.RefType(), acc_layout, *(scalar,) * 4),
    )(body)(merged_ref, acc, b, out_t, out_h, c_start)

def _mgpu_store_stats(max_ref, tot_ref, maximum, total, b, q, h, *, tile_1d):
    """Single-thread GMEM store of one row's scalar maximum and total (thread 0; caller gates `c_block`)."""
    scalar = plgpu.Layout.WG_SPLAT(())

    def body(ctx, m_ref, t_ref, m_v, t_v, b_v, q_v, h_v):
        del ctx
        pos0 = mgpu_ir.DenseI64ArrayAttr.get([0])
        i32 = mgpu_ir.IntegerType.get_signless(32)
        b_i, q_i, h_i = (v.registers.item() for v in (b_v, q_v, h_v))
        tid = mgpu_arith.remui(mgpu_utils.thread_idx(), mgpu_utils.c(THREADS, i32))
        elected = mgpu_arith.cmpi(mgpu_arith.CmpIPredicate.eq, tid, mgpu_utils.c(0, i32))
        branch = mgpu_scf.IfOp(elected)
        with mgpu_ir.InsertionPoint(branch.then_block):
            m_elem = mgpu_ir.MemRefType(m_ref.type).element_type
            t_elem = mgpu_ir.MemRefType(t_ref.type).element_type
            m_strides, _ = mgpu_ir.MemRefType(m_ref.type).get_strides_and_offset()
            t_strides, _ = mgpu_ir.MemRefType(t_ref.type).get_strides_and_offset()
            m_val = mgpu_vector.extract(m_v.registers.flat[0], dynamic_position=[], static_position=pos0)
            t_val = mgpu_vector.extract(t_v.registers.flat[0], dynamic_position=[], static_position=pos0)
            mgpu_llvm.store(m_val, mgpu_utils.getelementptr(
                _gmem_ptr(m_ref), [_i64_offset((b_i, q_i, h_i), m_strides)], m_elem))
            mgpu_llvm.store(t_val, mgpu_utils.getelementptr(
                _gmem_ptr(t_ref), [_i64_offset((b_i, q_i, h_i), t_strides)], t_elem))
            mgpu_scf.YieldOp([])

    plgpu.inline_mgpu(
        arg_types=(plgpu.RefType(), plgpu.RefType(), tile_1d, tile_1d, *(scalar,) * 3),
    )(body)(max_ref, tot_ref, maximum, total, b, q, h)


def _mgpu_load_output_param(param_ref, b, out_t, out_h, c_start, *, kind: Kind, block: int,
                            tile_block: int, width: int, tile_layout):
    """Predicated coalesced GMEM load of one `output_mod` parameter (`Kind.HEAD` as 0D scalar, `Kind.QUERY`
    as `[2, 1]` splat, `Kind.OUTPUT` as `[2, tile_block]` `tile_layout` fragment), widening 16-bit floats to
    fp32 in registers and preserving signedness for integer and Boolean operands (row 5385)."""
    scalar = plgpu.Layout.WG_SPLAT(())
    is_float = jnp.issubdtype(param_ref.dtype, jnp.floating)
    is_bool = param_ref.dtype == jnp.bool_
    widen = is_float and param_ref.dtype != jnp.float32
    out_dtype = jnp.dtype(jnp.float32) if is_float else param_ref.dtype
    is_signed = mgpu_utils.is_signed(out_dtype)
    is_scalar = kind is not Kind.OUTPUT
    if kind is Kind.HEAD:
        out_ty = plgpu.ShapeDtypeStruct((), out_dtype, layout=scalar)
    elif kind is Kind.QUERY:
        out_ty = plgpu.ShapeDtypeStruct((2, 1), out_dtype, layout=plgpu.Layout.WG_SPLAT((2, 1)))
    else:
        out_ty = plgpu.ShapeDtypeStruct((2, tile_block), out_dtype, layout=tile_layout)
    def body(ctx, p_ref, b_v, t_v, h_v, c_start_v):
        del ctx
        i1, i32 = (mgpu_ir.IntegerType.get_signless(bits) for bits in (1, 32))
        b_i, t_i, h_i, cs_i = (v.registers.item() for v in (b_v, t_v, h_v, c_start_v))
        mem_ty = mgpu_ir.MemRefType(p_ref.type)
        elem = mem_ty.element_type
        strides, _ = mem_ty.get_strides_and_offset()
        shape = mem_ty.shape
        base = _gmem_ptr(p_ref)
        f32 = mgpu_ir.F32Type.get()
        res_elem = i1 if is_bool else (f32 if widen else elem)
        zero_elem = mgpu_arith.constant(elem, 0.0 if is_float else 0)

        def cast_loaded(raw):
            if widen:
                return mgpu_arith.extf(f32, raw)
            if is_bool and elem != i1:
                return mgpu_arith.cmpi(mgpu_arith.CmpIPredicate.ne, raw, zero_elem)
            return raw

        if is_scalar:
            if kind is Kind.HEAD:
                off = _i64_offset((h_i,), strides)
                splat_shape: tuple[int, ...] = ()
            else:
                b_idx = mgpu_utils.c(0, i32) if shape[0] == 1 else b_i
                h_idx = mgpu_utils.c(0, i32) if shape[1] == 1 else h_i
                off = _i64_offset((b_idx, h_idx, t_i), strides)
                splat_shape = (2, 1)
            loaded = cast_loaded(mgpu_llvm.load(elem, mgpu_utils.getelementptr(base, [off], elem)))
            return mgpu_fragments.FragmentedArray.splat(loaded, splat_shape, is_signed=is_signed)
        vec = mgpu_ir.VectorType.get((1,), res_elem)
        t2_mgpu = tile_layout.to_mgpu()
        reg_shape = t2_mgpu.registers_shape((2, tile_block))
        b_idx = mgpu_utils.c(0, i32) if shape[0] == 1 else b_i
        row_off = _i64_offset((b_idx, t_i, h_i), strides[:3])
        regs = np.empty(reg_shape, dtype=object)
        for rep, (ok, col_off) in enumerate(_columns(tile_block, block, width, cs_i, strides[3])):
            raw = _predicated_load(base, mgpu_arith.addi(row_off, col_off), ok, elem, zero_elem)
            loaded = cast_loaded(raw)
            regs[:, rep, ...] = mgpu_vector.from_elements(vec, [loaded])
        return mgpu_fragments.FragmentedArray(_registers=regs, _layout=t2_mgpu, _is_signed=is_signed)

    return plgpu.inline_mgpu(
        arg_types=(plgpu.RefType(), *(scalar,) * 4),
        return_type=out_ty,
    )(body)(param_ref, b, out_t, out_h, c_start)


def launch(padded, stacked, *extra, units: windows.Units, queries: int, width: int, block: int,
                   raw_scale: float | None = None, out_dtype=None, out_heads: int | None = None,
                   output_mod: Callable | None = None, output_kinds: tuple[Kind, ...] = (),
                   family: str = "mma.sync"):
    """`_value` as one Mosaic GPU launch on NVIDIA sm80+ GPUs (row 9795): zero SMEM, zero warp shuffles,
    predicated coalesced GMEM loads/stores over arbitrary `width`/`slices`, executing the exact same
    `algebra.finished_merged_chunked` body as `_launch`, with optional fused `output_mod` epilogue
    (row 5385)."""
    batch, slices, _, heads, read_width = padded.shape
    init_args, _ = split_extra(extra, output_kinds)
    has_initial = bool(init_args)
    total_slices = slices + int(has_initial)
    chunk = max(windows.chunk_width(total_slices), 2)
    blocks = (1 << (width - 1).bit_length()) // block
    tile_block = max(block, 64)
    tile_layout = _tile_layout(tile_block)
    tile_1d = _tile_1d_layout(tile_layout)
    rows = (batch, queries, heads)
    n_init = len(init_args)
    n_params = len(output_kinds)
    def entry(padded_ref, stacked_ref, *rest):
        row = lax.axis_index("row")
        h, bq = lax.rem(row, heads), lax.div(row, heads)
        q, b = lax.rem(bq, queries), lax.div(bq, queries)
        c_block = lax.axis_index("block")
        c_start = c_block * block
        init_refs, param_refs, out_refs = (
            rest[:n_init], rest[n_init:n_init + n_params], rest[n_init + n_params:])
        if has_initial:
            init_m_ref, init_t_ref, init_acc_ref = init_refs
            raw_acc, raw_m, raw_t = _mgpu_load_initial(
                init_m_ref, init_t_ref, init_acc_ref, b, h, q, c_start, block=block, tile_block=tile_block,
                width=width, tile_1d=tile_1d)
            init_state: algebra.Partial | None = algebra.initial_state(
                units, algebra.Partial(raw_acc, raw_m, raw_t), raw_scale=raw_scale)
        else:
            init_state = None

        def read(start):
            out, maximum, total = _mgpu_load_chunk(
                padded_ref, stacked_ref, b, q, h, start, c_start, chunk=chunk, block=block,
                tile_block=tile_block, slices=slices, read_width=min(width, read_width),
                tile_layout=tile_layout)
            if init_state is not None:
                slice_pos = plgpu.layout_cast(
                    start + jax.lax.broadcasted_iota(jnp.int32, (chunk, tile_block), 0), tile_layout)
                is_init = slice_pos == slices
                out = jnp.where(is_init, init_state.acc, out)
                maximum = jnp.where(is_init, init_state.m, maximum)
                total = jnp.where(is_init, init_state.total, total)
            return out, maximum, total

        def load(c):
            step = jnp.int32(c) if isinstance(c, int) else mosaic.step_index(c)
            start = step * chunk
            valid = plgpu.layout_cast(
                start + jax.lax.broadcasted_iota(jnp.int32, (chunk, tile_block), 0) < total_slices,
                tile_layout)
            return (*read(start), valid)

        state = algebra.finished_merged_chunked(units, load, -(-total_slices // chunk))
        if out_dtype is not None:
            assert out_heads is not None
            group = out_heads // heads
            out_t, out_h = lax.div(q, group), h * group + lax.rem(q, group)
            (merged_ref,) = out_refs
            acc = plgpu.layout_cast(state.acc, tile_1d)
            if output_mod is not None:
                vals = tuple(
                    _mgpu_load_output_param(ref, b, out_t, out_h, c_start, kind=kind, block=block,
                                            tile_block=tile_block, width=width, tile_layout=tile_layout)
                    for ref, kind in zip(param_refs, output_kinds, strict=True))
                acc = output_mod(
                    plgpu.layout_cast(lax.broadcast_in_dim(acc, (2, tile_block), (1,)), tile_layout), *vals)
                store_layout = tile_layout
            else:
                store_layout = tile_1d
            written = plgpu.layout_cast(acc.astype(out_dtype), store_layout)
            _mgpu_store_finished(merged_ref, written, b, out_t, out_h, c_start, block=block,
                                 tile_block=tile_block, width=width, acc_layout=store_layout)
        else:
            merged_ref, maximum_ref, total_ref = out_refs
            written = plgpu.layout_cast(state.acc, tile_1d)
            _mgpu_store_finished(merged_ref, written, b, q, h, c_start, block=block, tile_block=tile_block,
                                 width=width, acc_layout=tile_1d)
            pl.when(c_block == 0)(
                lambda: _mgpu_store_stats(maximum_ref, total_ref, plgpu.layout_cast(state.m, tile_1d),
                                          plgpu.layout_cast(state.total, tile_1d), b, q, h, tile_1d=tile_1d))

    if out_dtype is not None:
        assert out_heads is not None
        group = out_heads // heads
        tokens = queries // group
        out_shape: tuple = (jax.ShapeDtypeStruct((batch, tokens, out_heads, width), out_dtype),)
    else:
        out_shape = (jax.ShapeDtypeStruct((*rows, width), padded.dtype),
                     jax.ShapeDtypeStruct(rows, stacked.dtype), jax.ShapeDtypeStruct(rows, stacked.dtype))
    launched = mosaic.kernel(
        entry, out_type=out_shape, grid=(batch * queries * heads, blocks),
        grid_names=("row", "block"), warpgroups=1, compiled=mosaic.compiler_params(approx_math=False),
        interpret=False, family=family, name="merged_padded")
    merged_out = launched(padded, stacked, *extra)
    return merged_out if isinstance(merged_out, tuple) else (merged_out,)
