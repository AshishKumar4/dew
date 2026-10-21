# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""JAX's Pallas Mosaic GPU grouped matmuls for Hopper, vendored: `gmm`, `[rows, depth] x [groups, depth,
width]` over rows sorted by group (with `transpose_rhs`, `[groups, width, depth]`), and `tgmm`, each
group's rows transposed times their cotangent rows, `[groups, depth, width]`.

They are `ragged_dot` of `pallas/ops/gpu/ragged_dot_mgpu.py` and `transposed_ragged_dot` of
`pallas/ops/gpu/transposed_ragged_dot_mgpu.py` in the jax-v0.11.2 source tree (tag jax-v0.11.2,
32544801; license header above), which ship as examples: wgmma over pipelined TMA loads, persistent
over the multiprocessors, a planar snake over the output tiles. A tile is a block-aligned row block of
one group, a block two groups share running once for each, and a tile stores only its group's rows, in
a power-of-two ladder of TMA stores. Changed for Dew, each marked `# Dew:`:

- the rows past the routed ones are one more group, whose tiles store zeros, so they come back zero,
  and a grid slot past the last tile, which upstream ran on group 0's matrix and stored nothing of,
  multiplies nothing;
- each writes the output dtype it is asked for (fp32 for a kernel gradient), not the operands';
- `tgmm` zeroes the neighbouring groups' rows of both operands by selection, where upstream
  multiplied one operand by a mask, which kept a neighbour's NaN or infinity (NaN * 0);
- the rows are padded to a multiple of 8, which a TMA descriptor's tiling needs (upstream's examples
  took row counts that were);
- the program count is the device's multiprocessors, where `ragged_dot` fixed an H100 SXM's 132;
- `tgmm`'s shared-memory tiling and swizzle are the warpgroup lowering's inference, as `gmm`'s are
  upstream, where upstream stated them with transforms whose type JAX leaves abstract.

Each sums exact products of its 16-bit operands in fp32 (wgmma's accumulator) and rounds once to the
output dtype.
"""

import dataclasses
import functools
import math

import jax
from jax import lax, numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu


@dataclasses.dataclass(frozen=True)
class Tiles:
    """One launch's output tile (`block_m` x `block_n`), contraction step (`block_k`), the steps in flight
    (`stages`, upstream's `max_concurrent_steps`) and the column tiles in each band of the planar snake
    (`grid_block_n`)."""
    block_m: int
    block_n: int
    block_k: int
    stages: int
    grid_block_n: int


@dataclasses.dataclass(frozen=True)
class GroupInfo:
    """Information regarding the group being processed in a block."""

    group_id: jax.Array
    block: jax.Array
    block_start: jax.Array
    actual_start: jax.Array
    actual_end: jax.Array
    start_within_block: jax.Array
    actual_size: jax.Array

    @classmethod
    def create(cls, group_lengths, tile, tid):
        """Get the group info for the current block."""

        tile = jnp.int32(tile)
        group_boundaries = [group_lengths[i] for i in range(len(group_lengths))]

        # We usually only have very few groups, so we unroll the loop processing
        # them. Normally we'd break out of the loop early, once we'd have found our
        # boundary, but we can't do that when unrolling, so we rely on many selects
        # to mask out the epilogue of the loop.
        group_end = group_start = block = group = end = jnp.array(0, dtype=jnp.int32)

        for i, b in enumerate(group_boundaries):
            # Start/end are inclusive
            start = end
            end = start + b
            final = end - 1
            start_block = lax.div(start, tile)
            final_block = lax.div(final, tile)
            block_end = final_block + 1
            tid_begin = start_block + i
            tid_end = block_end + i
            # How many blocks after is our block?
            this_is_group = (tid_begin <= tid) & (tid < tid_end)
            block = lax.select(this_is_group, tid - tid_begin + start_block, block)
            group = lax.select(this_is_group, jnp.int32(i), group)
            group_start = lax.select(this_is_group, start, group_start)
            group_end = lax.select(this_is_group, end, group_end)

        block_start = block * tile
        actual_start = jnp.maximum(group_start, block_start)
        actual_end = jnp.minimum(group_end, block_start + tile)
        start_within_block = actual_start - block_start
        actual_size = actual_end - actual_start
        return cls(group_id=group, block=block, block_start=block_start, actual_start=actual_start,
                   actual_end=actual_end, start_within_block=start_within_block, actual_size=actual_size)


def _compiler_params():
    return plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Warpgroup)


def gmm(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, *, tiles: Tiles, transpose_rhs: bool,
        out_dtype, multiprocessors: int, interpret: bool | plgpu.InterpretGPUParams = False,
        prologue=None, prologue_operands: tuple = (), prologue_axes: tuple = (), epilogue=None,
        epilogue_operands: tuple = (), epilogue_axes: tuple = (),
        out_dtypes: tuple | None = None) -> jax.Array | tuple[jax.Array, ...]:
    """Each group's rows of `lhs` `[rows, depth]` times its matrix of `rhs`, `[groups, depth, width]` or
    with `transpose_rhs` `[groups, width, depth]`: `[rows, width]` in `out_dtype`, the rows past the
    routed ones zero. The depth must divide by `tiles.block_k` and the width by `tiles.block_n`.

    Dew: `prologue(tile, *side_tiles)` transforms each `[block_m, block_k]` tile of `lhs` in its shared
    memory stage before the warpgroup MMA reads it; `epilogue(acc, *side_tiles)` each fp32
    accumulator tile, returning one output or a tuple of them (`out_dtypes`). A side operand's
    dimensions follow the launch axes its `*_axes` entry names, 'rows' and 'depth' for the prologue's,
    'rows' and 'columns' for the epilogue's, each brought to shared memory by TMA."""
    block_m, block_n, block_k = tiles.block_m, tiles.block_n, tiles.block_k
    rows, k = lhs.shape
    g, k2, n = rhs.shape
    if transpose_rhs:
        k2, n = n, k2
    if k != k2 or k % block_k or n % block_n:
        raise ValueError(f"a [{rows}, {k}] x [{g}, {k2}, {n}] product does not take {tiles}")
    # Dew: a TMA descriptor tiles the rows by 8, so they are padded to a multiple of 8, and the rows past
    # the routed ones, the padding among them, are a last group, whose tiles store zeros.
    lhs = _rows_padded(lhs)
    m = lhs.shape[0]
    lengths = jnp.concatenate([group_sizes.astype(jnp.int32),
                               jnp.maximum(m - jnp.sum(group_sizes, dtype=jnp.int32), 0)[None]])
    prologue_operands = tuple(_rows_padded(side) if axes[0] == 'rows' else side
                              for side, axes in zip(prologue_operands, prologue_axes, strict=True))
    epilogue_operands = tuple(_rows_padded(side) if axes[0] == 'rows' else side
                              for side, axes in zip(epilogue_operands, epilogue_axes, strict=True))
    dtypes = tuple(jnp.dtype(dtype) for dtype in (out_dtypes or (out_dtype,)))
    sides = len(prologue_operands) + len(epilogue_operands)

    def body(rows_per_expert_gmem, lhs_gmem, rhs_gmem, *refs):
        prologue_gmems = refs[:len(prologue_operands)]
        epilogue_gmems = refs[len(prologue_operands):sides]
        o_gmems = refs[sides:]
        grid_m = pl.cdiv(m, block_m) + g  # Dew: g groups and the unrouted rows
        grid_n = pl.cdiv(n, block_n)

        @plgpu.nd_loop((grid_m * grid_n,), collective_axes="sm")
        def mn_loop(loop_info: plgpu.NDLoopInfo):
            mi, ni = plgpu.planar_snake(loop_info.index[0], (grid_m, grid_n), 1, tiles.grid_block_n)
            group_info = GroupInfo.create(rows_per_expert_gmem, block_m, mi)
            # Dew: each prologue side operand's block at the launch axes it follows.
            side_block = {'rows': (block_m, lambda k: group_info.block), 'depth': (block_k, lambda k: k)}
            side_specs = [plgpu.BlockSpec(tuple(side_block[axis][0] for axis in axes),
                                          lambda k, axes=axes: tuple(side_block[axis][1](k) for axis in axes),
                                          delay_release=1)
                          for axes in prologue_axes]

            def acc_scope(acc_ref):
                def step(_, lhs_smem, rhs_smem, *side_smems):
                    if prologue is not None:
                        # Dew: the prologue rewrites the stage of `lhs` in place, which the one warpgroup
                        # owns, before the MMA reads it.
                        lhs_smem[...] = prologue(lhs_smem[...], *(side[...] for side in side_smems)).astype(
                            lhs_smem.dtype)
                        plgpu.commit_smem()
                    plgpu.wgmma(acc_ref, lhs_smem, rhs_smem.transpose((1, 0)) if transpose_rhs else rhs_smem)

                # Dew: the unrouted rows' tiles multiply nothing, their accumulator stays zero, and a grid
                # slot past the last tile no group has (GroupInfo's group 0, no rows) multiplies nothing.
                @pl.when((group_info.group_id < g) & (group_info.actual_size > 0))
                def _():
                    plgpu.emit_pipeline(
                        step,
                        grid=(k // block_k,),
                        in_specs=[
                            plgpu.BlockSpec((block_m, block_k), lambda k: (group_info.block, k),
                                            delay_release=1),
                            plgpu.BlockSpec((block_n, block_k) if transpose_rhs else (block_k, block_n),
                                            lambda k: (ni, k) if transpose_rhs else (k, ni), delay_release=1),
                            *side_specs,
                        ],
                        max_concurrent_steps=tiles.stages,
                    )(lhs_gmem, rhs_gmem.at[jnp.minimum(group_info.group_id, g - 1)], *prologue_gmems)
                return acc_ref[...]

            acc = pl.run_scoped(acc_scope, plgpu.WGMMAAccumulatorRef((block_m, block_n)))
            outs = (acc,) if epilogue is None else _epilogue(
                epilogue, acc, epilogue_gmems, epilogue_axes, group_info.block_start, ni * block_n,
                block_m, block_n)

            for o_gmem, out in zip(o_gmems, outs, strict=True):
                @functools.partial(pl.run_scoped, o_smem=plgpu.SMEM((block_m, block_n), dtype=o_gmem.dtype))
                def store_scope(o_smem, o_gmem=o_gmem, out=out):
                    o_smem[...] = out.astype(o_smem.dtype)
                    plgpu.commit_smem()

                    smem_start = group_info.start_within_block
                    remaining_rows = min(block_m, m)
                    # TMA descriptors need to be generated with static tile sizes along each axis, but we
                    # do not know at compile time how many rows we will need to store, only that it is at
                    # most min(block_m, m): a logarithmic ladder of descriptors, each storing 2**i rows,
                    # stores any count between 1 and min(block_m, m).
                    while remaining_rows > 0:
                        const_rows_len = 1 << int(math.log2(remaining_rows))
                        remaining_rows //= 2
                        _store_rows(o_smem, o_gmem, group_info, smem_start, const_rows_len, ni * block_n)
                        smem_start += group_info.actual_size & const_rows_len
                    plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    kernel = plgpu.kernel(
        body,
        # Dew: the output dtypes asked for, and the device's multiprocessors.
        out_type=tuple(jax.ShapeDtypeStruct((m, n), dtype) for dtype in dtypes),
        grid=(multiprocessors,),
        grid_names=("sm",),
        compiler_params=_compiler_params(),
        interpret=interpret,
    )
    outs = tuple(out[:rows] for out in kernel(lengths, lhs, rhs, *prologue_operands, *epilogue_operands))
    return outs if out_dtypes else outs[0]


def _epilogue(epilogue, acc, gmems, axes_of, row_start, column_start, block_m: int, block_n: int):
    """Dew: `epilogue(acc, *side_tiles)` as a tuple of outputs, each side operand's tile brought to shared
    memory by TMA at the output tile's rows and columns first."""
    origins = {'rows': (row_start, block_m), 'columns': (column_start, block_n)}
    shapes = [tuple(origins[axis][1] for axis in axes) for axes in axes_of]

    smems = [plgpu.SMEM(shape, gmem.dtype) for shape, gmem in zip(shapes, gmems, strict=True)]

    @functools.partial(pl.run_scoped, smems=smems, barriers=[plgpu.Barrier() for _ in gmems])
    def scope(smems, barriers):
        for gmem, smem, barrier, axes in zip(gmems, smems, barriers, axes_of, strict=True):
            plgpu.copy_gmem_to_smem(gmem.at[tuple(pl.ds(*origins[axis]) for axis in axes)], smem, barrier)
        for barrier in barriers:
            plgpu.barrier_wait(barrier)
        outs = epilogue(acc, *(smem[...] for smem in smems))
        return outs if isinstance(outs, tuple) else (outs,)

    return scope


def _store_rows(o_smem, o_gmem, group_info: GroupInfo, smem_start, rows: int, column) -> None:
    """One rung of the ladder: where the tile's group has `rows` rows left, store them from `smem_start`."""
    @pl.when(group_info.actual_size & rows != 0)
    def _():
        o_smem_slice = o_smem.at[pl.ds(smem_start, rows)]
        o_gref_slice = o_gmem.at[pl.ds(group_info.block_start + smem_start, rows),
                                 pl.ds(column, o_smem.shape[1])]
        plgpu.copy_smem_to_gmem(o_smem_slice, o_gref_slice)


def tgmm(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, *, tiles: Tiles, out_dtype,
         multiprocessors: int, interpret: bool | plgpu.InterpretGPUParams = False) -> jax.Array:
    """Each group's rows of `lhs` `[rows, depth]` transposed times its rows of `rhs` `[rows, width]`:
    `[groups, depth, width]` in `out_dtype`, an empty group's zero. The depth must divide by
    `tiles.block_m` and the width by `tiles.block_n`; `tiles.block_k` is the rows a step takes."""
    block_m, block_n, block_k = tiles.block_m, tiles.block_n, tiles.block_k
    k, m = lhs.shape
    k2, n = rhs.shape
    g = group_sizes.shape[0]
    if k != k2 or m % block_m or n % block_n:
        raise ValueError(f"a [{k}, {m}] by [{k2}, {n}] kernel gradient does not take {tiles}")
    # Dew: the rows padded to a multiple of 8, as `gmm`'s, past every group.
    lhs, rhs = _rows_padded(lhs), _rows_padded(rhs)
    k = lhs.shape[0]

    group_sizes = group_sizes.astype(jnp.int32)
    group_starts = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(group_sizes)[:-1]]).astype(jnp.int32)
    group_ends = jnp.cumsum(group_sizes).astype(jnp.int32)
    group_block_starts = group_starts // block_k * block_k
    group_block_ends = -(group_ends // -block_k) * block_k
    group_num_blocks = (group_block_ends - group_block_starts) // block_k

    out_dtype = jnp.dtype(out_dtype)

    def body(group_sizes_gmem, group_starts_gmem, group_ends_gmem, group_num_blocks_gmem,
             group_block_starts_gmem, lhs_gmem, rhs_gmem, o_gmem):
        grid_m = pl.cdiv(m, block_m)
        grid_n = pl.cdiv(n, block_n)

        @plgpu.nd_loop((g, grid_m * grid_n), collective_axes="sm")
        def mn_loop(loop_info: plgpu.NDLoopInfo):
            g_i = loop_info.index[0]
            m_i, n_i = plgpu.planar_snake(loop_info.index[1], (grid_m, grid_n), 1, tiles.grid_block_n)

            # This slice is potentially out of bounds, but we never access the out of bound part in
            # emit_pipeline.
            group_block_start = pl.multiple_of(group_block_starts_gmem[g_i], block_k)
            gmem_slice = pl.ds(group_block_start, k)

            def acc_scope(acc_ref):
                def block_matmul(block_idx, lhs_smem, rhs_smem):
                    block_idx = block_idx[0]

                    # Dew: the first and last blocks of the group may hold a neighbour's rows, which
                    # both operands zero by selection, so a neighbour's NaN or infinity stays out.
                    @pl.when(block_idx == 0)
                    def _():
                        start_index = lax.rem(group_starts_gmem[g_i], block_k)
                        _zero_rows(lhs_smem, rhs_smem, lambda rows: rows >= start_index, block_k, block_m,
                                   block_n)

                    @pl.when(block_idx == group_num_blocks_gmem[g_i] - 1)
                    def _():
                        last_index = lax.rem(group_ends_gmem[g_i] - 1, block_k)
                        _zero_rows(lhs_smem, rhs_smem, lambda rows: rows <= last_index, block_k, block_m,
                                   block_n)

                    plgpu.wgmma(acc_ref, lhs_smem.transpose((1, 0)), rhs_smem)
                    if tiles.stages == 1:
                        # Without delayed release, we won't have at least two separate smem blocks in
                        # flight, so we cannot rely on the implicit wait of wgmma to guarantee that the
                        # data in smem is ready to be overwritten by the next pipeline iteration.
                        plgpu.wgmma_wait(0)

                @pl.when(group_sizes_gmem[g_i] > 0)  # Skip the group if it is empty.
                def _():
                    plgpu.emit_pipeline(
                        block_matmul,
                        grid=(group_num_blocks_gmem[g_i],),
                        in_specs=[
                            plgpu.BlockSpec((block_k, block_m), lambda k_i: (k_i, m_i),
                                            delay_release=int(tiles.stages > 1)),
                            plgpu.BlockSpec((block_k, block_n), lambda k_i: (k_i, n_i),
                                            delay_release=int(tiles.stages > 1)),
                        ],
                        max_concurrent_steps=tiles.stages,
                    )(lhs_gmem.at[gmem_slice, :], rhs_gmem.at[gmem_slice, :])

                return acc_ref[...]

            acc = pl.run_scoped(acc_scope, plgpu.WGMMAAccumulatorRef((block_m, block_n)))

            # Dew: the output in its own dtype, an fp32 gradient above all.
            @functools.partial(pl.run_scoped, o_smem=plgpu.SMEM((block_m, block_n), dtype=o_gmem.dtype))
            def store_scope(o_smem):
                o_smem[...] = acc.astype(o_smem.dtype)
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(
                    o_smem, o_gmem.at[g_i, pl.ds(m_i * block_m, block_m), pl.ds(n_i * block_n, block_n)])
                plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    kernel = plgpu.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((g, m, n), out_dtype),
        grid=(multiprocessors,),
        grid_names=("sm",),
        compiler_params=_compiler_params(),
        interpret=interpret,
    )
    return kernel(group_sizes, group_starts, group_ends, group_num_blocks, group_block_starts, lhs, rhs)


def _rows_padded(array: jax.Array) -> jax.Array:
    """`array` with zero rows after its own up to a multiple of 8, the rows a TMA descriptor tiles by."""
    pad = -array.shape[0] % 8
    return jnp.pad(array, ((0, pad),) + ((0, 0),) * (array.ndim - 1)) if pad else array


def _zero_rows(lhs_smem, rhs_smem, kept, block_k: int, block_m: int, block_n: int) -> None:
    """Zero the rows of both staged blocks, `[block_k, block_m]` and `[block_k, block_n]`, that `kept`
    (of each row's index) turns down, by selection."""
    for ref, columns in ((lhs_smem, block_m), (rhs_smem, block_n)):
        values = ref[...]
        rows = plgpu.layout_cast(lax.broadcasted_iota(jnp.int32, (block_k, columns), 0), plgpu.Layout.WGMMA)
        ref[...] = jnp.where(kept(rows), values, jnp.zeros_like(values))
    plgpu.commit_smem()
