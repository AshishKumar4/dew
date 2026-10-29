"""The GEMM template's Blackwell mainloop (Mosaic GPU, sm100 and sm103), first rung: `gmm`, `[rows, depth] x
[groups, depth, width]` over rows sorted by group (with `transpose_rhs`, `[groups, width, depth]`), on the
schedule of JAX's `blackwell_ragged_dot_mgpu.py`: tcgen05's products into an accumulator in tensor
memory, two of them, so a tile's epilogue runs while the next tile's products fill the other, on
Blackwell's bundle (`mosaic_sm100`).

Persistent over the output tiles in a planar snake, one CTA a tile, the bundle's two warpgroups
(`Tcgen05.roles`). The first issues: its load warp fills the bundle's ring of operand stages
(`mosaic_sm100.Stages`), its product warp issues each step's product once its stage is there, the
product releasing the stage when it has read it. The second stores: it reads a finished accumulator out
of tensor memory a column chunk at a time (`mosaic_sm100.Accumulators`), applies the epilogue there with
its side tiles, and stores each output's chunk in a power-of-two ladder of TMA stores
(`mosaic.store_rows`; a tile stores only its group's rows), then frees the accumulator for the tile after
next. Each row tile's group comes from the template's table (`mosaic_gpu.tile_groups`). Dew's, beside
upstream's:

- the epilogue hook, `epilogue(acc, *side_tiles)` on each fp32 chunk, returning one output or a tuple
  (`out_dtypes`), each side operand along 'rows' and 'columns' brought to shared memory by TMA;
- each writes the output dtype it is asked for, not the operands';
- the rows past the routed ones are one more group, whose tiles multiply nothing: each output there is
  the epilogue of a zero accumulator (zero without one); a grid slot past the last tile stores nothing;
- tiles that multiply nothing take neither a stage nor an accumulator, so the ring's and the
  accumulators' turns count the tiles that do (the loop's carry), where upstream counted every tile;
- the rows are padded to a multiple of 8, which a TMA descriptor's tiling needs, and to a tile's rows at
  least, which each tile's loads take.

Each sums exact products of its 16-bit operands in fp32 (the tensor memory accumulator) and rounds once.
The prologue, the kernel gradient and a pair of CTAs (tcgen05's two-CTA products) are later rungs.
"""

import dataclasses
import functools
from typing import NamedTuple

import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import generation
from dew.nn.kernels.templates import mosaic, mosaic_sm100
from dew.nn.kernels.templates.gemm.mosaic_gpu import GroupInfo, tile_groups
from dew.nn.kernels.templates.mosaic_sm100 import TCGEN05


@dataclasses.dataclass(frozen=True)
class Tiles:
    """One launch's output tile (`block_m` x `block_n`, tcgen05's M of one CTA and its N, two accumulators
    of it in tensor memory), contraction step (`block_k`), the steps in flight (`stages`), the column
    tiles in each band of the planar snake (`grid_block_n`) and the columns the store warpgroup takes at
    a time (`epilogue_n`)."""
    block_m: int = 128
    block_n: int = 128
    block_k: int = 64
    stages: int = 4
    grid_block_n: int = 1
    epilogue_n: int = 64


def gmm(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, *, tiles: Tiles, transpose_rhs: bool,
        out_dtype, target: generation.GpuTarget, interpret: bool | plgpu.InterpretGPUParams = False,
        epilogue=None, epilogue_operands: tuple = (), epilogue_axes: tuple = (),
        out_dtypes: tuple | None = None) -> jax.Array | tuple[jax.Array, ...]:
    """Each group's rows of `lhs` `[rows, depth]` times its matrix of `rhs`, `[groups, depth, width]` or with
    `transpose_rhs` `[groups, width, depth]`: `[rows, width]` in `out_dtype` (or a tuple in `out_dtypes`),
    each row past the routed ones the epilogue of a zero accumulator, over `target`'s multiprocessors.
    The depth must divide by `tiles.block_k` and the width by `tiles.block_n`, and `tiles.block_n` by
    `tiles.epilogue_n`."""
    block_m, block_n, block_k = tiles.block_m, tiles.block_n, tiles.block_k
    rows, depth = lhs.shape
    groups = rhs.shape[0]
    rhs_depth, width = (rhs.shape[2], rhs.shape[1]) if transpose_rhs else rhs.shape[1:]
    if (depth != rhs_depth or depth % block_k or width % block_n or block_n % tiles.epilogue_n
            or block_m != TCGEN05.rows):
        raise ValueError(f"a [{rows}, {depth}] x {list(rhs.shape)} product does not take {tiles}")
    lhs = _rows_padded(lhs, 0, block_m)
    m = lhs.shape[0]
    sizes = group_sizes.astype(jnp.int32)
    lengths = jnp.concatenate([sizes, jnp.maximum(m - jnp.sum(sizes), 0)[None]])
    row_tiles = pl.cdiv(m, block_m) + groups
    table = tile_groups(lengths, block_m, row_tiles)
    epilogue_operands = tuple(_rows_padded(side, axes.index('rows'), block_m) if 'rows' in axes else side
                              for side, axes in zip(epilogue_operands, epilogue_axes, strict=True))
    dtypes = tuple(jnp.dtype(dtype) for dtype in (out_dtypes or (out_dtype,)))
    grid = (row_tiles, width // block_n)
    transforms = mosaic_sm100.swizzled(lhs.dtype, block_k)
    side_extent = {'rows': block_m, 'columns': tiles.epilogue_n}
    shape = _Shape(m, groups, depth // block_k, block_n // tiles.epilogue_n, transpose_rhs, side_extent)
    stages = mosaic_sm100.Stages(tiles.stages, (
        ((block_m, block_k), lhs.dtype, transforms),
        ((block_n, block_k) if transpose_rhs else (block_k, block_n), lhs.dtype, transforms)))
    accumulators = mosaic_sm100.Accumulators(2, block_m, block_n)

    def kernel(table_gmem, lhs_gmem, rhs_gmem, *refs):
        sides = refs[:len(epilogue_operands)]
        outs = refs[len(epilogue_operands):len(epilogue_operands) + len(dtypes)]
        ring_refs, tensor_refs, o_smems, side_smems, side_full = refs[len(epilogue_operands) + len(dtypes):]
        scratch = _Scratch(stages.opened(*ring_refs), accumulators.opened(*tensor_refs), o_smems, side_smems,
                           side_full)

        @plgpu.nd_loop((grid[0] * grid[1],), collective_axes="sm", init_carry=jnp.int32(0))
        def computed(loop: plgpu.NDLoopInfo, carry):
            mi, ni = plgpu.planar_snake(loop.index[0], grid, 1, tiles.grid_block_n)
            rows_of = GroupInfo.create(table_gmem, block_m, mi)
            # The unrouted rows' group multiplies nothing, and a slot past the last tile has no rows; the
            # ring's turn is the steps the computed tiles before took, the accumulator's their count.
            tile = _Tile(rows_of, ni, (rows_of.group_id < groups) & (rows_of.actual_size > 0),
                         carry * shape.steps, carry)
            TCGEN05.roles(
                lambda: pl.when(tile.computes)(functools.partial(_load, tile, lhs_gmem, rhs_gmem, scratch,
                                                                 tiles=tiles, shape=shape)),
                lambda: pl.when(tile.computes)(functools.partial(_products, tile, scratch, shape=shape)),
                lambda: pl.when(rows_of.actual_size > 0)(functools.partial(
                    _stores, tile, sides, outs, scratch, epilogue, epilogue_axes, tiles=tiles, shape=shape)))
            return carry + tile.computes.astype(jnp.int32)

        # Each barrier's last completions, which no later turn waited for, observed, so every barrier
        # ends at the phase its waiters saw: the stages' last releases and the accumulators' last reads.
        TCGEN05.issued(lambda: scratch.ring.drain(computed, shape.steps),
                       lambda: scratch.tensor.drain(computed))

    scratch = (
        stages.scratch(), accumulators.scratch(target),
        tuple(plgpu.SMEM((block_m, tiles.epilogue_n), dtype) for dtype in dtypes),
        tuple(plgpu.SMEM(tuple(side_extent[axis] for axis in axes), side.dtype)
              for side, axes in zip(epilogue_operands, epilogue_axes, strict=True)),
        plgpu.Barrier(num_barriers=max(len(epilogue_operands), 1)),
    )
    outs = TCGEN05.kernel(
        kernel, out_type=tuple(jax.ShapeDtypeStruct((m, width), dtype) for dtype in dtypes),
        grid=(target.multiprocessors,), grid_names=("sm",), scratch=scratch, interpret=interpret,
        name="dew_gmm_blackwell")(table, lhs, rhs, *epilogue_operands)
    outs = tuple(out[:rows] for out in outs)
    return outs if out_dtypes else outs[0]


class _Shape(NamedTuple):
    """A launch's static facts its roles read: the padded rows, the groups, the depth steps of a tile, the
    epilogue's column chunks of one, the matrix's orientation and each side operand axis's extent."""
    rows: int
    groups: int
    steps: int
    chunks: int
    transpose_rhs: bool
    side_extent: dict


class _Scratch(NamedTuple):
    """The kernel's opened operand ring (`mosaic_sm100.Ring`) and accumulators (`mosaic_sm100.Tensor`),
    each output's and side operand's chunk, and the side tiles' loads."""
    ring: mosaic_sm100.Ring
    tensor: mosaic_sm100.Tensor
    o_smems: tuple
    side_smems: tuple
    side_full: object


class _Tile(NamedTuple):
    """One output tile: its rows' group facts, its column tile, whether it multiplies (`computes`), the
    ring's turn at its first step (`first`) and the computed tiles before it (`done`)."""
    info: GroupInfo
    column: jax.Array
    computes: jax.Array
    first: jax.Array
    done: jax.Array


def _load(tile: _Tile, lhs_gmem, rhs_gmem, scratch: _Scratch, *, tiles: Tiles, shape: _Shape) -> None:
    """The load warp: each step's operand blocks into the ring's next stage (`Ring.load`)."""
    block_m, block_n, block_k = tiles.block_m, tiles.block_n, tiles.block_k
    matrix = rhs_gmem.at[jnp.minimum(tile.info.group_id, shape.groups - 1)]
    columns = pl.ds(tile.column * block_n, block_n)

    @pl.loop(0, shape.steps)
    def _(step):
        depths = pl.ds(step * block_k, block_k)
        block = matrix.at[columns, depths] if shape.transpose_rhs else matrix.at[depths, columns]
        scratch.ring.load(tile.first + step, (lhs_gmem.at[pl.ds(tile.info.block_start, block_m), depths],
                                              block))


def _products(tile: _Tile, scratch: _Scratch, *, shape: _Shape) -> None:
    """The product warp: each step's product into the tile's accumulator once its stage is loaded,
    releasing the stage when read, once the accumulator's tile before last is read out."""
    acc = scratch.tensor.acquire(tile.done)

    @pl.loop(0, shape.steps)
    def _(step):
        (a_smem, b_smem), consumed = scratch.ring.acquire(tile.first + step)
        plgpu.tcgen05_mma(acc, a_smem, b_smem.T if shape.transpose_rhs else b_smem, consumed,
                          accumulate=step > 0)

    scratch.tensor.issued(tile.done)


def _stores(tile: _Tile, sides, outs, scratch: _Scratch, epilogue, epilogue_axes, *, tiles: Tiles,
            shape: _Shape) -> None:
    """The store warpgroup: the tile's accumulator out of tensor memory a chunk of columns at a time once
    its products are done, through the epilogue, each output's rows of the tile's group stored; the
    accumulator freed. A tile of the unrouted rows stores the epilogue of a zero accumulator."""
    rows_of, chunk_n = tile.info, tiles.epilogue_n

    def store(chunk_of):
        for chunk in range(shape.chunks):
            column = tile.column * tiles.block_n + chunk * chunk_n
            acc = chunk_of(chunk)
            values = (acc,) if epilogue is None else _epilogue(
                epilogue, acc, sides, scratch.side_smems, scratch.side_full, epilogue_axes,
                {'rows': rows_of.block_start, 'columns': column}, shape.side_extent)
            for o_smem, o_gmem, value in zip(scratch.o_smems, outs, values, strict=True):
                o_smem[...] = value.astype(o_smem.dtype)
                plgpu.commit_smem()
                mosaic.store_rows(o_smem, o_gmem, rows_of.block_start, rows_of.start_within_block,
                                  rows_of.actual_size, column, min(tiles.block_m, shape.rows))
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    @pl.when(tile.computes)
    def _():
        store(scratch.tensor.read(tile.done, chunk_n))
        scratch.tensor.release(tile.done)

    pl.when(~tile.computes)(lambda: store(lambda chunk: plgpu.layout_cast(
        jnp.zeros((tiles.block_m, chunk_n), jnp.float32), mosaic_sm100.STORED)))


def _epilogue(epilogue, acc, gmems, smems, barriers, axes_of, origins: dict, extent: dict) -> tuple:
    """`epilogue(acc, *side_tiles)` as a tuple of outputs, each side operand's tile at the chunk's rows and
    columns brought to shared memory by TMA first."""
    for index, (gmem, smem, axes) in enumerate(zip(gmems, smems, axes_of, strict=True)):
        plgpu.copy_gmem_to_smem(gmem.at[tuple(pl.ds(origins[axis], extent[axis]) for axis in axes)], smem,
                                barriers.at[index])
    for index in range(len(gmems)):
        plgpu.barrier_wait(barriers.at[index])
    outs = epilogue(acc, *(smem[...] for smem in smems))
    return outs if isinstance(outs, tuple) else (outs,)


def _rows_padded(array: jax.Array, axis: int, least: int) -> jax.Array:
    """`array` with zeros after its own along `axis` up to a multiple of 8, the rows a TMA descriptor tiles
    by, and to `least` rows at least."""
    rows = array.shape[axis]
    pad = max(-rows % 8, least - rows)
    widths = tuple((0, pad if dim == axis else 0) for dim in range(array.ndim))
    return jnp.pad(array, widths) if pad > 0 else array
