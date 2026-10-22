"""The GEMM template's Blackwell mainloop (Mosaic GPU, sm100), first rung: `gmm`, `[rows, depth] x [groups,
depth, width]` over rows sorted by group (with `transpose_rhs`, `[groups, width, depth]`), on the schedule of
JAX's `blackwell_ragged_dot_mgpu.py` (jax-v0.11.2): tcgen05's products into an accumulator in tensor
memory, two of them, so a tile's epilogue runs while the next tile's products fill the other.

Persistent over the output tiles in a planar snake, one CTA a tile, two warpgroups. The first issues: one
warp's TMA loads fill a ring of operand stages, one warp issues each step's product once its stage is
there, releasing the stage when the product has read it. The second stores: it reads a finished
accumulator out of tensor memory a column chunk at a time, applies the epilogue there with its side tiles,
and stores each output's chunk in a power-of-two ladder of TMA stores (a tile stores only its group's
rows), then frees the accumulator for the tile after next. Dew's, beside upstream's:

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
from jax import lax, numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import hopper
from dew.nn.kernels.templates.gemm.mosaic_gpu import GroupInfo

ISSUE, STORE = 0, 1
"""The warpgroups: the one issuing loads and products, and the one storing finished tiles."""
LOAD_WARP, MMA_WARP = 0, 1
"""The issuing warpgroup's warps: the TMA loads' and the products'."""


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
        out_dtype, multiprocessors: int, interpret: bool | plgpu.InterpretGPUParams = False, epilogue=None,
        epilogue_operands: tuple = (), epilogue_axes: tuple = (),
        out_dtypes: tuple | None = None) -> jax.Array | tuple[jax.Array, ...]:
    """Each group's rows of `lhs` `[rows, depth]` times its matrix of `rhs`, `[groups, depth, width]` or with
    `transpose_rhs` `[groups, width, depth]`: `[rows, width]` in `out_dtype` (or a tuple in `out_dtypes`),
    each row past the routed ones the epilogue of a zero accumulator. The depth must divide by
    `tiles.block_k` and the width by `tiles.block_n`, and `tiles.block_n` by `tiles.epilogue_n`."""
    block_m, block_n, block_k, stages = tiles.block_m, tiles.block_n, tiles.block_k, tiles.stages
    rows, depth = lhs.shape
    groups = rhs.shape[0]
    rhs_depth, width = (rhs.shape[2], rhs.shape[1]) if transpose_rhs else rhs.shape[1:]
    if depth != rhs_depth or depth % block_k or width % block_n or block_n % tiles.epilogue_n:
        raise ValueError(f"a [{rows}, {depth}] x {list(rhs.shape)} product does not take {tiles}")
    lhs = _rows_padded(lhs, 0, block_m)
    m = lhs.shape[0]
    sizes = group_sizes.astype(jnp.int32)
    lengths = jnp.concatenate([sizes, jnp.maximum(m - jnp.sum(sizes), 0)[None]])
    epilogue_operands = tuple(_rows_padded(side, axes.index('rows'), block_m) if 'rows' in axes else side
                              for side, axes in zip(epilogue_operands, epilogue_axes, strict=True))
    dtypes = tuple(jnp.dtype(dtype) for dtype in (out_dtypes or (out_dtype,)))
    grid = (pl.cdiv(m, block_m) + groups, width // block_n)
    steps = depth // block_k
    chunks = block_n // tiles.epilogue_n
    swizzle = plgpu.find_swizzle(block_k * jnp.dtype(lhs.dtype).itemsize * 8)
    transforms = (plgpu.TilingTransform((8, swizzle // jnp.dtype(lhs.dtype).itemsize)),
                  plgpu.SwizzleTransform(swizzle))
    side_extent = {'rows': block_m, 'columns': tiles.epilogue_n}

    shape = _Shape(m, groups, steps, chunks, transpose_rhs, side_extent)

    def kernel(lengths_gmem, lhs_gmem, rhs_gmem, *refs):
        sides = refs[:len(epilogue_operands)]
        outs = refs[len(epilogue_operands):len(epilogue_operands) + len(dtypes)]
        scratch = _Scratch(*refs[len(epilogue_operands) + len(dtypes):])
        wg = lax.axis_index("wg")

        @plgpu.nd_loop((grid[0] * grid[1],), collective_axes="sm", init_carry=jnp.int32(0))
        def _(loop: plgpu.NDLoopInfo, carry):
            mi, ni = plgpu.planar_snake(loop.index[0], grid, 1, tiles.grid_block_n)
            info = GroupInfo.create(lengths_gmem, block_m, mi)
            # The unrouted rows' group multiplies nothing, and a slot past the last tile has no rows; the
            # ring's turn is the steps the computed tiles before took, the accumulator's their parity.
            tile = _Tile(info, ni, (info.group_id < groups) & (info.actual_size > 0), carry * steps, carry)

            @pl.when(wg == ISSUE)
            def _():
                @plgpu.warp_map
                def _(warp):
                    pl.when((warp == LOAD_WARP) & tile.computes)(
                        functools.partial(_load, tile, lhs_gmem, rhs_gmem, scratch, tiles=tiles, shape=shape))
                    pl.when((warp == MMA_WARP) & tile.computes)(
                        functools.partial(_products, tile, scratch, tiles=tiles, shape=shape))

            pl.when((wg == STORE) & (info.actual_size > 0))(functools.partial(
                _stores, tile, sides, outs, scratch, epilogue, epilogue_axes, tiles=tiles, shape=shape))
            return carry + tile.computes.astype(jnp.int32)

    tensor_core = functools.partial(plgpu.Barrier, orders_tensor_core=True)
    scratch = (
        plgpu.SMEM((stages, block_m, block_k), lhs.dtype, transforms=transforms),
        plgpu.SMEM((stages, block_n, block_k) if transpose_rhs else (stages, block_k, block_n), lhs.dtype,
                   transforms=transforms),
        tuple(plgpu.SMEM((block_m, tiles.epilogue_n), dtype) for dtype in dtypes),
        tuple(plgpu.SMEM(tuple(side_extent[axis] for axis in axes), side.dtype)
              for side, axes in zip(epilogue_operands, epilogue_axes, strict=True)),
        plgpu.TMEM((block_m, 2 * block_n), jnp.float32),
        plgpu.Barrier(num_barriers=stages), plgpu.Barrier(num_barriers=stages),
        tensor_core(num_barriers=stages), tensor_core(num_barriers=2), tensor_core(num_barriers=2),
        plgpu.Barrier(num_barriers=max(len(epilogue_operands), 1)),
    )
    outs = plgpu.kernel(
        kernel, out_type=tuple(jax.ShapeDtypeStruct((m, width), dtype) for dtype in dtypes),
        grid=(multiprocessors,), grid_names=("sm",), num_threads=2, thread_name="wg", scratch_types=scratch,
        compiler_params=plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Warpgroup),
        interpret=interpret,
    )(lengths, lhs, rhs, *epilogue_operands)
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
    """The kernel's shared memory, tensor memory and barriers: the operand stages, each output's and side
    operand's chunk, the two accumulators, and the stages' loads (`a_full`, `b_full`) and releases
    (`consumed`), the accumulators' products (`mma_done`) and stores (`store_done`), the side tiles'
    loads."""
    a_smem: object
    b_smem: object
    o_smems: tuple
    side_smems: tuple
    acc_tmem: object
    a_full: object
    b_full: object
    consumed: object
    mma_done: object
    store_done: object
    side_full: object


class _Tile(NamedTuple):
    """One output tile: its rows' group facts, its column tile, whether it multiplies (`computes`), the
    ring's turn at its first step (`first`) and the computed tiles before it (`done`)."""
    info: GroupInfo
    column: jax.Array
    computes: jax.Array
    first: jax.Array
    done: jax.Array

    @property
    def accumulator(self):
        return lax.rem(self.done, 2)


def _load(tile: _Tile, lhs_gmem, rhs_gmem, scratch: _Scratch, *, tiles: Tiles, shape: _Shape) -> None:
    """The load warp: each step's operand blocks into the ring's next stage, once its products have
    read the stage's last ones."""
    block_m, block_n, block_k = tiles.block_m, tiles.block_n, tiles.block_k
    matrix = rhs_gmem.at[jnp.minimum(tile.info.group_id, shape.groups - 1)]
    columns = pl.ds(tile.column * block_n, block_n)

    @pl.loop(0, shape.steps)
    def _(step):
        turn = tile.first + step
        slot = lax.rem(turn, tiles.stages)
        pl.when(turn >= tiles.stages)(functools.partial(plgpu.barrier_wait, scratch.consumed.at[slot]))
        depths = pl.ds(step * block_k, block_k)
        plgpu.copy_gmem_to_smem(lhs_gmem.at[pl.ds(tile.info.block * block_m, block_m), depths],
                                scratch.a_smem.at[slot], scratch.a_full.at[slot])
        block = matrix.at[columns, depths] if shape.transpose_rhs else matrix.at[depths, columns]
        plgpu.copy_gmem_to_smem(block, scratch.b_smem.at[slot], scratch.b_full.at[slot])


def _products(tile: _Tile, scratch: _Scratch, *, tiles: Tiles, shape: _Shape) -> None:
    """The product warp: each step's product into the tile's accumulator once its stage is loaded,
    releasing the stage when read, once the accumulator's tile before last is stored."""
    pl.when(tile.done >= 2)(functools.partial(plgpu.barrier_wait, scratch.store_done.at[tile.accumulator]))
    acc = scratch.acc_tmem.at[:, pl.ds(tile.accumulator * tiles.block_n, tiles.block_n)]

    @pl.loop(0, shape.steps)
    def _(step):
        slot = lax.rem(tile.first + step, tiles.stages)
        plgpu.barrier_wait(scratch.a_full.at[slot])
        plgpu.barrier_wait(scratch.b_full.at[slot])
        matrix = scratch.b_smem.at[slot].T if shape.transpose_rhs else scratch.b_smem.at[slot]
        plgpu.tcgen05_mma(acc, scratch.a_smem.at[slot], matrix, scratch.consumed.at[slot],
                          accumulate=step > 0)

    plgpu.tcgen05_commit_arrive(scratch.mma_done.at[tile.accumulator])


def _stores(tile: _Tile, sides, outs, scratch: _Scratch, epilogue, epilogue_axes, *, tiles: Tiles,
            shape: _Shape) -> None:
    """The store warpgroup: the tile's accumulator out of tensor memory a chunk of columns at a time once
    its products are done, through the epilogue, each output's rows of the tile's group stored; the
    accumulator freed. A tile of the unrouted rows stores the epilogue of a zero accumulator."""
    info, chunk_n = tile.info, tiles.epilogue_n

    def store(chunk_of):
        for chunk in range(shape.chunks):
            column = tile.column * tiles.block_n + chunk * chunk_n
            acc = chunk_of(chunk)
            values = (acc,) if epilogue is None else _epilogue(
                epilogue, acc, sides, scratch.side_smems, scratch.side_full, epilogue_axes,
                {'rows': info.block * tiles.block_m, 'columns': column}, shape.side_extent)
            for o_smem, o_gmem, value in zip(scratch.o_smems, outs, values, strict=True):
                o_smem[...] = value.astype(o_smem.dtype)
                plgpu.commit_smem()
                hopper.store_rows(o_smem, o_gmem, info.block_start, info.start_within_block, info.actual_size,
                                  column, min(tiles.block_m, shape.rows))
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    @pl.when(tile.computes)
    def _():
        plgpu.barrier_wait(scratch.mma_done.at[tile.accumulator])
        first = tile.accumulator * tiles.block_n
        store(lambda chunk: plgpu.async_load_tmem(
            scratch.acc_tmem.at[:, pl.ds(first + chunk * chunk_n, chunk_n)], layout=plgpu.Layout.TCGEN05))
        plgpu.wait_load_tmem()
        plgpu.barrier_arrive(scratch.store_done.at[tile.accumulator])

    pl.when(~tile.computes)(lambda: store(lambda chunk: plgpu.layout_cast(
        jnp.zeros((tiles.block_m, chunk_n), jnp.float32), plgpu.Layout.TCGEN05)))

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
