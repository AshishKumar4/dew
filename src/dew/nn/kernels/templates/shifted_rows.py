"""A register tile read at a row shift (`Issuer.row_shifted`): each row taking another's values within its
group by the layout's own lane permutation, a warp's shuffles, where a shared memory round trip's shifted
plain reads met every bank at once (row 5305: neither a padded row, which Mosaic's tiling refuses at a
stride off its width, nor a swizzle, whose refs slice at 8-row atoms only, reads it free of them).

Composed from Mosaic GPU's pieces until its FragmentedArray holds the shift itself (the fork's row: a
register tile's row shift beside its layout operations), when this lowering is deleted for it.
"""

import dataclasses
import functools

import numpy as np
from jax._src.lib.mlir.dialects import nvvm as mgpu_nvvm
from jax.experimental.mosaic.gpu import fragmented_array as mgpu_fragments, utils as mgpu_utils
from jax.extend.mlir import ir as mgpu_ir
from jax.extend.mlir.dialects import arith as mgpu_arith, vector as mgpu_vector

WARP = 32
"""A warp's lanes."""
WARPS = 4
"""A warpgroup's warps."""


def elements(layout, shape: tuple[int, int]) -> np.ndarray:
    """Where each register element of a 2-D array in a tiled layout (`mgpu_fragments.TiledLayout`) lies, as
    the layout's own index map places it (`TiledLayout.thread_idxs`, its warp and lane indices'
    delinearization, `_delinearize_index`), evaluated in integers: `[warps, lanes, *registers, vector, 2]`,
    each element's row and column."""
    contiguous = tuple(mgpu_utils.get_contiguous_strides(shape))
    strides = layout.tiling.tile_strides(contiguous)
    tiled = layout.tiled_tiling_shape
    rank = len(tiled)
    dynamic = strides[-rank:]

    def delinearized(index: int, dims) -> list[int]:
        sizes = tuple(d.times if isinstance(d, mgpu_fragments.Replicated) else tiled[d] for d in dims)
        full = [0] * rank
        for d, stride, size in zip(dims, mgpu_utils.get_contiguous_strides(sizes), sizes, strict=True):
            if not isinstance(d, mgpu_fragments.Replicated):
                full[d] = index // stride % size
        return full

    registers, vector = layout.registers_shape(shape), layout.vector_length
    found = np.zeros((WARPS, WARP, *registers, vector, 2), np.int64)
    for warp in range(WARPS):
        warp_offset = int(np.dot(delinearized(warp, layout.warp_dims), dynamic))
        for lane in range(WARP):
            offset = warp_offset + int(np.dot(delinearized(lane, layout.lane_dims), dynamic))
            for at in np.ndindex(registers):
                start = offset + sum(i * s for i, s in zip(at, strides, strict=False))
                for element in range(vector):
                    found[(warp, lane, *at, element)] = divmod(start + element, shape[1])
    return found


@dataclasses.dataclass(frozen=True)
class Lanes:
    """A layout's rows within a warp, as a row shift reads them (`lanes`): a warp holds groups of `rows`
    rows, a lane's row within its group its register's `offsets[at]` plus `lane // columns` (`columns` lanes
    a row), its columns its register's and `lane % columns`'s alone; `peers[at]` each register holding the
    same columns of the same group, by its row offset."""

    columns: int
    rows: int
    offsets: dict
    peers: dict


@functools.cache
def lanes(layout, shape: tuple[int, int]) -> Lanes | None:
    """The layout's rows within a warp (`Lanes`), where every element lies as its form states (`elements`,
    the layout's own index map); None for a layout that holds no such form."""
    if not isinstance(layout, mgpu_fragments.TiledLayout):
        return None
    placed = elements(layout, shape)
    rows_of, columns_of = placed[..., 0], placed[..., 1]
    columns = next((n for n in (1, 2, 4, 8, 16, 32) if n == WARP or np.any(rows_of[:, n] != rows_of[:, 0])),
                   None)
    lane = np.arange(WARP).reshape(1, WARP, *(1,) * (rows_of.ndim - 2))
    registers = tuple(np.ndindex(rows_of.shape[2:-1]))
    # The largest group a warp holds whole (an m16 fragment's 16 rows, its two halves registers apart).
    for rows in (64, 32, 16, 8):
        if rows < WARP // columns:
            continue
        offsets = {at: int(rows_of[(0, 0, *at, 0)] % rows) for at in registers}
        offset = np.zeros(rows_of.shape[2:], np.int64)
        for at in registers:
            offset[at] = offsets[at]
        base = rows_of - rows_of % rows
        moved = columns_of[:, np.arange(WARP) % columns]
        if (np.all(rows_of % rows == offset[None, None] + lane // columns)
                and np.all(base == base[:, :1]) and np.all(columns_of == moved)):
            peers = {at: {offsets[other]: other for other in registers
                          if base[(0, 0, *other, 0)] == base[(0, 0, *at, 0)]
                          and np.array_equal(columns_of[(0, 0, *other)], columns_of[(0, 0, *at)])}
                     for at in registers}
            return Lanes(columns, rows, offsets, peers)
    return None


def source(row, shift, lane, form: Lanes, ops):
    """Where an element at row `row` of its warp's group (its register's offset and its lane's row) reads
    its value under `shift` (`row - shift`, within the group): its source lane and its source register's row
    offset, in `ops`' arithmetic (`Integers` or `Lowered`)."""
    per = WARP // form.columns
    taken = ops.rem(ops.sub(row, shift), form.rows)
    return (ops.add(ops.mul(ops.rem(taken, per), form.columns), ops.rem(lane, form.columns)),
            ops.sub(taken, ops.rem(taken, per)))


class Integers:
    """`source`'s arithmetic on Python integers."""

    @staticmethod
    def add(a, b):
        return a + b

    @staticmethod
    def sub(a, b):
        return a - b

    @staticmethod
    def mul(a, b):
        return a * b

    @staticmethod
    def rem(a, b):
        return a % b


class Lowered:
    """`source`'s arithmetic on i32 values, its moduli powers of two taken as masks."""

    @staticmethod
    def i32(value):
        return value if isinstance(value, mgpu_ir.Value) else mgpu_utils.c(
            value, mgpu_ir.IntegerType.get_signless(32))

    @staticmethod
    def add(a, b):
        return mgpu_arith.addi(Lowered.i32(a), Lowered.i32(b))

    @staticmethod
    def sub(a, b):
        return mgpu_arith.subi(Lowered.i32(a), Lowered.i32(b))

    @staticmethod
    def mul(a, b):
        return mgpu_arith.muli(Lowered.i32(a), Lowered.i32(b))

    @staticmethod
    def rem(a, b):
        return mgpu_arith.andi(Lowered.i32(a), Lowered.i32(b - 1))


def shuffled(value, lane):
    """`value` (a register, a vector of 32-bit words or wider) as lane `lane` holds it: a warp's
    `shfl.sync.idx` a word."""
    i32 = mgpu_ir.IntegerType.get_signless(32)
    count = mgpu_utils.bitwidth(value.type) // 32
    words = mgpu_utils.bitcast(value, mgpu_ir.VectorType.get((count,), i32))
    out = words
    for i in range(count):
        position = mgpu_ir.DenseI64ArrayAttr.get([i])
        word = mgpu_vector.extract(words, dynamic_position=[], static_position=position)
        moved = mgpu_nvvm.shfl_sync(mgpu_utils.c(0xFFFFFFFF, i32), word, lane, mgpu_utils.c(0x1F, i32),
                                    mgpu_nvvm.ShflKind.idx)
        out = mgpu_vector.insert(moved, out, dynamic_position=[], static_position=position)
    return mgpu_utils.bitcast(out, value.type)


def row_shifted(array, shift, period: int):
    """`array` (an `mgpu_fragments.FragmentedArray` in a layout `lanes` reads) with each row `i` holding row
    `i - shift`'s values within its `period`-row group (`shift` an i32), any value where that row leaves it:
    each register taken from the lane holding its source row, a shuffle for each row offset its group spans,
    the source's selected by its offset."""
    form = lanes(array.layout, array.shape)
    assert form is not None and form.rows % period == 0 and form.rows & (form.rows - 1) == 0, (
        f"a row shift reads a layout whose warps hold whole groups of {period} rows: {array.layout}")
    i32 = mgpu_ir.IntegerType.get_signless(32)
    lane = mgpu_arith.remui(mgpu_utils.thread_idx(), mgpu_utils.c(WARP, i32))
    lane_row = mgpu_arith.divui(lane, mgpu_utils.c(form.columns, i32))
    registers = array.registers
    out = np.empty(registers.shape, dtype=object)
    for at in np.ndindex(registers.shape):
        lane_from, offset_from = source(Lowered.add(lane_row, form.offsets[at]), shift, lane, form, Lowered)
        taken = None
        for peer_offset, peer in sorted(form.peers[at].items()):
            moved = shuffled(registers[peer], lane_from)
            here = mgpu_arith.cmpi(mgpu_arith.CmpIPredicate.eq, offset_from, Lowered.i32(peer_offset))
            taken = moved if taken is None else mgpu_arith.select(here, moved, taken)
        out[at] = taken
    return mgpu_fragments.FragmentedArray(_registers=out, _layout=array.layout, _is_signed=array.is_signed)
