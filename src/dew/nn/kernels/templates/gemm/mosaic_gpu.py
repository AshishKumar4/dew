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

- the rows past the routed ones are one more group, whose tiles multiply nothing, so each output there
  is the epilogue of a zero accumulator (zero without one), and a grid slot past the last tile, which
  upstream ran on group 0's matrix and stored nothing of,
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
    (`grid_block_n`); Dew: the warpgroups computing a tile (`warpgroups`: 2 halve `block_n` between two
    consumer warpgroups beside a producer that issues the loads, CUTLASS's cooperative sm90 schedule), and
    the multiprocessors of a cluster (`cluster`: 2 pair row tiles of one group, which load each step's
    matrix block once between them, multicast)."""
    block_m: int
    block_n: int
    block_k: int
    stages: int
    grid_block_n: int
    warpgroups: int = 1
    cluster: int = 1


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
    prologue_operands = tuple(_side_padded(side, axes) for side, axes in zip(prologue_operands, prologue_axes,
                                                                               strict=True))
    epilogue_operands = tuple(_side_padded(side, axes) for side, axes in zip(epilogue_operands, epilogue_axes,
                                                                               strict=True))
    dtypes = tuple(jnp.dtype(dtype) for dtype in (out_dtypes or (out_dtype,)))
    sides = len(prologue_operands) + len(epilogue_operands)
    if tiles.warpgroups == 2:
        outs = _gmm_cooperative(lhs, rhs, lengths, (m, n, k, g), tiles=tiles, transpose_rhs=transpose_rhs,
                                dtypes=dtypes, multiprocessors=multiprocessors, interpret=interpret,
                                hooks=(prologue, prologue_axes, epilogue, epilogue_axes),
                                sides=(prologue_operands, epilogue_operands))
        outs = tuple(out[:rows] for out in outs)
        return outs if out_dtypes else outs[0]

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
            # Dew: the rows past the routed ones multiplied nothing: each output there is the epilogue of a
            # zero accumulator, zero without one.
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


def _gmm_cooperative(lhs, rhs, lengths, shape: tuple, *, tiles: Tiles, transpose_rhs: bool, dtypes: tuple,
                     multiprocessors: int, interpret, hooks: tuple, sides: tuple) -> tuple:
    """Dew: `gmm`'s product at a tile two consumer warpgroups compute, each a half of its columns, beside a
    producer warpgroup issuing the loads (`emit_pipeline_warp_specialized`, as JAX's
    `hopper_matmul_mgpu`), with `tiles.cluster` 2 a pair of multiprocessors on two row tiles of one group,
    which load each step's matrix block once between them, multicast. A prologue rewrites the stage of
    `lhs` both warpgroups read: each rewrites half its rows and waits for the other's half before its MMA.
    Each keeps one MMA in flight while it prepares the next stage. `lhs` is padded and `lengths` holds the
    unrouted rows' group last (`gmm`); the outputs keep the padded rows."""
    prologue, prologue_axes, epilogue, epilogue_axes = hooks
    prologue_operands, epilogue_operands = sides
    m, n, k, g = shape
    block_m, block_n, block_k, pair = tiles.block_m, tiles.block_n, tiles.block_k, tiles.cluster
    half, rows_half = block_n // 2, block_m // 2
    side_shapes = [tuple({'rows': block_m, 'columns': half}[axis] for axis in axes) for axes in epilogue_axes]
    # A pipeline's operands each follow its steps: a prologue side operand without depth is loaded once a
    # tile.
    stepped = tuple(i for i, axes in enumerate(prologue_axes) if 'depth' in axes)
    fixed = tuple(i for i, axes in enumerate(prologue_axes) if 'depth' not in axes)
    extent = {'rows': block_m, 'depth': block_k}
    pipeline = functools.partial(_gmm_pipeline, tiles, transpose_rhs, k // block_k,
                                 tuple(prologue_axes[i] for i in stepped))

    def body(lengths_gmem, lhs_gmem, rhs_gmem, *refs):
        prologue_gmems = refs[:len(prologue_operands)]
        side_gmems = refs[len(prologue_operands):len(prologue_operands) + len(epilogue_operands)]
        o_gmems = refs[len(prologue_operands) + len(epilogue_operands):]
        grid_m = pl.cdiv(m, block_m * pair) + g  # pairs of row tiles: g groups and the unrouted rows
        grid_n = n // block_n
        rhs_columns = (lambda gmem, ni: gmem.at[pl.ds(ni * block_n, block_n)]) if transpose_rhs else (
            lambda gmem, ni: gmem.at[:, pl.ds(ni * block_n, block_n)])

        def stepped_views(rows):
            return [prologue_gmems[i].at[tuple(rows if axis == 'rows' else slice(None)
                                               for axis in prologue_axes[i])] for i in stepped]

        @functools.partial(
            pl.run_scoped,
            allocations=pipeline(_ignore, _ignore).get_allocations(
                lhs_gmem.at[pl.ds(0, block_m)], rhs_columns(rhs_gmem.at[0], 0),
                *stepped_views(pl.ds(0, block_m))),
            o_smems=[plgpu.SMEM((2, block_m, half), gmem.dtype) for gmem in o_gmems],
            side_smems=[plgpu.SMEM((2, *shape), gmem.dtype) for shape, gmem in zip(side_shapes, side_gmems,
                                                                                    strict=True)],
            side_barriers=[plgpu.Barrier(num_barriers=2) for _ in side_gmems],
            fixed_smems=[plgpu.SMEM((2, *(extent[axis] for axis in prologue_axes[i])),
                                    prologue_gmems[i].dtype) for i in fixed],
            fixed_barriers=[plgpu.Barrier(num_barriers=2) for _ in fixed],
            halves=plgpu.Barrier(num_arrivals=2, num_barriers=2),
            collective_axes="wg",
        )
        def scope(allocations, o_smems, side_smems, side_barriers, fixed_smems, fixed_barriers, halves):
            wg = lax.axis_index("wg")
            cta = lax.axis_index("cluster") if pair == 2 else 0
            mine_rows = pl.ds(wg * rows_half, rows_half)
            fixed_refs = [smem.at[wg] for smem in fixed_smems]

            @plgpu.nd_loop((grid_m * grid_n,), collective_axes="cluster_grid")
            def mn_loop(loop_info: plgpu.NDLoopInfo):
                mi, ni = plgpu.planar_snake(loop_info.index[0], (grid_m, grid_n), 1, tiles.grid_block_n)
                group_info = GroupInfo.create(lengths_gmem, block_m * pair, mi)
                # This multiprocessor's row tile of the pair and its group's rows there.
                row_block = group_info.block * pair + cta
                row_start = row_block * block_m
                first = jnp.clip(group_info.actual_start - row_start, 0, block_m)
                stored = jnp.clip(group_info.actual_end - row_start, 0, block_m) - first
                column = ni * block_n + wg * half

                def finish(acc):
                    """The warpgroup's half tile: its epilogue, then its group's rows of each output."""
                    sides = []
                    origins = {'rows': pl.ds(row_start, block_m), 'columns': pl.ds(column, half)}
                    for gmem, smem, barrier, axes in zip(side_gmems, side_smems, side_barriers, epilogue_axes,
                                                         strict=True):
                        plgpu.copy_gmem_to_smem(gmem.at[tuple(origins[axis] for axis in axes)], smem.at[wg],
                                                barrier.at[wg])
                        plgpu.barrier_wait(barrier.at[wg])
                        sides.append(smem[wg])
                    outs = (acc,) if epilogue is None else epilogue(acc, *sides)
                    outs = outs if isinstance(outs, tuple) else (outs,)
                    for o_gmem, o_smem, out in zip(o_gmems, o_smems, outs, strict=True):
                        plgpu.wait_smem_to_gmem(0, wait_read_only=True)
                        o_smem[wg] = out.astype(o_smem.dtype)
                        plgpu.commit_smem()
                        _store_ladder(o_smem.at[wg], o_gmem, row_start, first, stored, column,
                                      min(block_m, m))

                def context(run):
                    for i, ref, barrier in zip(fixed, fixed_refs, fixed_barriers, strict=True):
                        plgpu.copy_gmem_to_smem(prologue_gmems[i].at[pl.ds(row_start, block_m)], ref,
                                                barrier.at[wg])
                        plgpu.barrier_wait(barrier.at[wg])

                    @functools.partial(pl.run_scoped, acc_ref=plgpu.ACC((block_m, half), jnp.float32))
                    def _(acc_ref):
                        run(acc_ref)
                        plgpu.wgmma_wait(0)
                        finish(acc_ref[...])

                def mma(_, lhs_smem, rhs_smem, *rest):
                    *side_smems, acc_ref = rest
                    if prologue is not None:
                        # This warpgroup's half of the rows, then the other's, before an MMA reads them all.
                        mine = lhs_smem.at[mine_rows]
                        tiles_of = dict(zip(stepped, side_smems, strict=True)) | dict(zip(fixed, fixed_refs,
                                                                                          strict=True))
                        mine[...] = prologue(mine[...], *(
                            tiles_of[i].at[tuple(mine_rows if axis == 'rows' else slice(None)
                                                 for axis in prologue_axes[i])][...]
                            for i in range(len(prologue_axes)))).astype(mine.dtype)
                        plgpu.commit_smem()
                        _meet(halves)
                    columns = pl.ds(wg * half, half)
                    matrix = (rhs_smem.at[columns].transpose((1, 0)) if transpose_rhs
                              else rhs_smem.at[:, columns])
                    plgpu.wgmma(acc_ref, lhs_smem, matrix)
                    # One MMA in flight: the next stage is prepared while this one runs.
                    plgpu.wgmma_wait(1)
                    return acc_ref

                # Both multiprocessors of a pair run its pipeline, or neither: they share its loads.
                active = (group_info.group_id < g) & (group_info.actual_size > 0)

                @pl.when(active)
                def _():
                    pipeline(mma, context)(
                        lhs_gmem.at[pl.ds(row_start, block_m)],
                        rhs_columns(rhs_gmem.at[jnp.minimum(group_info.group_id, g - 1)], ni),
                        *stepped_views(pl.ds(row_start, block_m)),
                        allocations=allocations)

                # The unrouted rows, and a slot past the last tile, multiply nothing.
                @pl.when(~active & (wg < 2))
                def _():
                    finish(jnp.zeros((block_m, half), jnp.float32))

        plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    kernel = plgpu.kernel(
        body,
        out_type=tuple(jax.ShapeDtypeStruct((m, n), dtype) for dtype in dtypes),
        grid=(multiprocessors // pair,),
        grid_names=("cluster_grid",),
        cluster=(pair,),
        cluster_names=("cluster",),
        num_threads=3,
        thread_name="wg",
        compiler_params=_compiler_params(),
        interpret=interpret,
    )
    return kernel(lengths, lhs, rhs, *prologue_operands, *epilogue_operands)


def _gmm_pipeline(tiles: Tiles, transpose_rhs: bool, steps: int, side_axes: tuple, body, context):
    """Dew: the cooperative `gmm`'s pipeline over the depth's `steps` blocks: the rows' stage, the matrix's
    (multicast to a pair), and each prologue side operand's that has depth, in a producer warpgroup; a
    stage stays one step longer, for the MMA in flight."""
    extent = {'rows': tiles.block_m, 'depth': tiles.block_k}

    def side_spec(axes):
        return plgpu.BlockSpec(tuple(extent[axis] for axis in axes),
                               lambda step: tuple(step if axis == 'depth' else 0 for axis in axes),
                               memory_space=plgpu.SMEM, delay_release=1)

    rhs_block = (tiles.block_n, tiles.block_k) if transpose_rhs else (tiles.block_k, tiles.block_n)
    return plgpu.emit_pipeline_warp_specialized(
        body,
        grid=(steps,),
        memory_registers=40,
        in_specs=[plgpu.BlockSpec((tiles.block_m, tiles.block_k), lambda step: (0, step),
                                  memory_space=plgpu.SMEM, delay_release=1),
                  plgpu.BlockSpec(rhs_block, _rhs_index(transpose_rhs), memory_space=plgpu.SMEM,
                                  delay_release=1, collective_axes=_shared(tiles)),
                  *(side_spec(axes) for axes in side_axes)],
        wg_axis="wg",
        num_compute_wgs=2,
        max_concurrent_steps=tiles.stages,
        compute_context=context,
    )


def _meet(halves) -> None:
    """Dew: the two consumer warpgroups wait for each other at `halves`, two barriers of two arrivals, in
    turn: neither arrives at a barrier's next phase before the other has seen its last, however the meets
    fall (the interpreter's invariant, and the hardware's parity waits)."""
    for barrier in (halves.at[0], halves.at[1]):
        plgpu.barrier_arrive(barrier)
        plgpu.barrier_wait(barrier)


def _shared(tiles: Tiles) -> tuple[str, ...]:
    """The axes a block a pair shares is loaded over, multicast: the cluster's, for a pair."""
    return ("cluster",) if tiles.cluster == 2 else ()


def _rhs_index(transpose_rhs: bool):
    """A pipeline step's block of the matrix's column tile: its depth block `step`."""
    return (lambda step: (0, step)) if transpose_rhs else (lambda step: (step, 0))


def _ignore(*_, **__) -> None:
    """A pipeline body or compute context that does nothing: the allocations a pipeline takes do not
    depend on them (`hopper_matmul_mgpu`'s `ignore`)."""


def _store_ladder(o_smem, o_gmem, row_start, first, stored, column, most: int) -> None:
    """Dew: rows `first` to `first + stored` of `o_smem` to `o_gmem` at row `row_start + first` and
    `column`, in a power-of-two ladder of TMA stores (a descriptor's extents are static), the largest
    `most` rows' (`gmm`'s ladder)."""
    start, size = first, most
    while size:
        rung = 1 << int(math.log2(size))
        size //= 2

        @pl.when(stored & rung != 0)
        def _(start=start, rung=rung):
            plgpu.copy_smem_to_gmem(o_smem.at[pl.ds(start, rung)],
                                    o_gmem.at[pl.ds(row_start + start, rung), pl.ds(column, o_smem.shape[1])])

        start = start + (stored & rung)


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
         multiprocessors: int, interpret: bool | plgpu.InterpretGPUParams = False, prologue=None,
         prologue_operands: tuple = (), prologue_axes: tuple = (), exact_sums: bool = False) -> jax.Array:
    """Each group's rows of `lhs` `[rows, depth]` transposed times its rows of `rhs` `[rows, width]`:
    `[groups, depth, width]` in `out_dtype`, an empty group's zero. The depth must divide by
    `tiles.block_m` and the width by `tiles.block_n`; `tiles.block_k` is the rows a step takes.

    Dew: `prologue(rows, *side_tiles)` transforms each `[block_k, block_m]` stage of `lhs`'s rows in
    place before the group masking and the MMA; side operands follow 'rows' and 'depth' as `gmm`'s.
    `exact_sums` adds each step's product into the fp32 sum by an IEEE add, which holds two
    accumulators in registers."""
    block_m, block_n, block_k = tiles.block_m, tiles.block_n, tiles.block_k
    k, m = lhs.shape
    k2, n = rhs.shape
    g = group_sizes.shape[0]
    if k != k2 or m % block_m or n % block_n:
        raise ValueError(f"a [{k}, {m}] by [{k2}, {n}] kernel gradient does not take {tiles}")
    # Dew: the rows padded to a multiple of 8, as `gmm`'s, past every group.
    lhs, rhs = _rows_padded(lhs), _rows_padded(rhs)
    k = lhs.shape[0]
    prologue_operands = tuple(_side_padded(side, axes) for side, axes in zip(prologue_operands, prologue_axes,
                                                                               strict=True))

    group_sizes = group_sizes.astype(jnp.int32)
    group_starts, group_ends, group_num_blocks, group_block_starts = _group_blocks(group_sizes, block_k)
    out_dtype = jnp.dtype(out_dtype)
    if tiles.warpgroups == 2:
        return _tgmm_cooperative(
            (lhs, rhs), (group_sizes, group_starts, group_ends, group_num_blocks, group_block_starts),
            (k, m, n, g), tiles=tiles, out_dtype=out_dtype, multiprocessors=multiprocessors,
            interpret=interpret, prologue=prologue, prologue_operands=prologue_operands,
            prologue_axes=prologue_axes, exact_sums=exact_sums)

    def body(group_sizes_gmem, group_starts_gmem, group_ends_gmem, group_num_blocks_gmem,
             group_block_starts_gmem, lhs_gmem, rhs_gmem, *refs):
        *side_gmems, o_gmem = refs
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

            side_specs, sides = _tgmm_sides(side_gmems, prologue_axes, gmem_slice, m_i, tiles)

            def staged(block_idx, lhs_smem, rhs_smem, side_smems):
                if prologue is not None:
                    # Dew: the prologue rewrites the rows' stage in place, which the one warpgroup owns.
                    lhs_smem[...] = prologue(lhs_smem[...], *(side[...] for side in side_smems)).astype(
                        lhs_smem.dtype)
                    plgpu.commit_smem()
                _zero_neighbours(lhs_smem, rhs_smem, block_idx, group_starts_gmem[g_i],
                                 group_ends_gmem[g_i], group_num_blocks_gmem[g_i], tiles)

            pipeline = functools.partial(
                plgpu.emit_pipeline, grid=(group_num_blocks_gmem[g_i],),
                in_specs=[plgpu.BlockSpec((block_k, block_m), lambda k_i: (k_i, m_i),
                                          delay_release=int(tiles.stages > 1)),
                          plgpu.BlockSpec((block_k, block_n), lambda k_i: (k_i, n_i),
                                          delay_release=int(tiles.stages > 1)),
                          *side_specs],
                max_concurrent_steps=tiles.stages)
            operands = (lhs_gmem.at[gmem_slice, :], rhs_gmem.at[gmem_slice, :], *sides)
            tile = o_gmem.at[g_i, pl.ds(m_i * block_m, block_m), pl.ds(n_i * block_n, block_n)]
            nonempty = group_sizes_gmem[g_i] > 0  # Skip the group if it is empty.

            summed = _exactly_summed_gradient if exact_sums else _summed_gradient
            summed(pipeline, operands, staged, tile, nonempty, tiles)

    kernel = plgpu.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((g, m, n), out_dtype),
        grid=(multiprocessors,),
        grid_names=("sm",),
        compiler_params=_compiler_params(),
        interpret=interpret,
    )
    return kernel(group_sizes, group_starts, group_ends, group_num_blocks, group_block_starts, lhs, rhs,
                  *prologue_operands)


def _tgmm_cooperative(operands: tuple, groups: tuple, shape: tuple, *, tiles: Tiles, out_dtype,
                      multiprocessors: int, interpret, prologue, prologue_operands: tuple,
                      prologue_axes: tuple, exact_sums: bool) -> jax.Array:
    """Dew: `tgmm`'s product at a tile two consumer warpgroups compute, each a half of its columns, beside a
    producer warpgroup issuing the loads, with `tiles.cluster` 2 a pair of multiprocessors on two depth
    tiles of one group, which load each step's cotangent block once between them, multicast. Both
    warpgroups read the whole stage of the rows: each prepares a depth half of it (the prologue, the
    neighbouring groups' rows zeroed) and its own column half of the cotangent's, and each waits for the
    other's half before its MMA. `operands` are padded and `groups` is `_group_blocks`' with the sizes
    first (`tgmm`)."""
    lhs, rhs = operands
    k, m, n, g = shape
    block_m, block_n, block_k, pair = tiles.block_m, tiles.block_n, tiles.block_k, tiles.cluster
    half, depth_half = block_n // 2, block_m // 2
    # A pipeline's operands each follow its steps: a side operand without rows is loaded once a tile.
    stepped = tuple(i for i, axes in enumerate(prologue_axes) if 'rows' in axes)
    fixed = tuple(i for i, axes in enumerate(prologue_axes) if 'rows' not in axes)
    extent = {'rows': block_k, 'depth': block_m}
    pipeline = functools.partial(_tgmm_pipeline, tiles, tuple(prologue_axes[i] for i in stepped))

    def body(sizes_gmem, starts_gmem, ends_gmem, blocks_gmem, block_starts_gmem, lhs_gmem, rhs_gmem, *refs):
        *side_gmems, o_gmem = refs
        grid_m, grid_n = m // (block_m * pair), n // block_n

        def views(rows, depth, columns):
            """The pipeline's operands at a group's rows (from its first block) and a tile's depth and
            columns."""
            sides = [side_gmems[i].at[tuple(rows if axis == 'rows' else depth for axis in prologue_axes[i])]
                     for i in stepped]
            return lhs_gmem.at[rows, depth], rhs_gmem.at[rows, columns], *sides

        @functools.partial(
            pl.run_scoped,
            allocations=pipeline(_ignore, _ignore, tiles.stages).get_allocations(
                *views(pl.ds(0, k), pl.ds(0, block_m), pl.ds(0, block_n))),
            o_smem=plgpu.SMEM((2, block_m, half), out_dtype),
            fixed_smems=[plgpu.SMEM((2, *(extent[axis] for axis in prologue_axes[i])), side_gmems[i].dtype)
                         for i in fixed],
            fixed_barriers=[plgpu.Barrier(num_barriers=2) for _ in fixed],
            halves=plgpu.Barrier(num_arrivals=2, num_barriers=2),
            collective_axes="wg",
        )
        def scope(allocations, o_smem, fixed_smems, fixed_barriers, halves):
            wg = lax.axis_index("wg")
            cta = lax.axis_index("cluster") if pair == 2 else 0
            mine_depth, mine_columns = pl.ds(wg * depth_half, depth_half), pl.ds(wg * half, half)
            fixed_refs = [smem.at[wg] for smem in fixed_smems]

            @plgpu.nd_loop((g, grid_m * grid_n), collective_axes="cluster_grid")
            def mn_loop(loop_info: plgpu.NDLoopInfo):
                g_i = loop_info.index[0]
                pair_i, n_i = plgpu.planar_snake(loop_info.index[1], (grid_m, grid_n), 1, tiles.grid_block_n)
                depth = pl.ds((pair_i * pair + cta) * block_m, block_m)
                # This slice may run past the rows; the pipeline reads only the group's blocks.
                rows = pl.ds(pl.multiple_of(block_starts_gmem[g_i], block_k), k)
                steps, start, end = blocks_gmem[g_i], starts_gmem[g_i], ends_gmem[g_i]

                def staged(step, lhs_smem, rhs_smem, side_smems):
                    """This warpgroup's depth half of the rows' stage and column half of the cotangent's,
                    ready; then the other's half, before an MMA reads the whole stage."""
                    mine = lhs_smem.at[:, mine_depth]
                    if prologue is not None:
                        sides = dict(zip(stepped, side_smems, strict=True)) | dict(zip(fixed, fixed_refs,
                                                                                        strict=True))
                        tiles_of = [sides[i].at[tuple(slice(None) if axis == 'rows' else mine_depth
                                                      for axis in prologue_axes[i])][...]
                                    for i in range(len(prologue_axes))]
                        mine[...] = prologue(mine[...], *tiles_of).astype(mine.dtype)
                    boundary = _zero_neighbours_of((mine, rhs_smem.at[:, mine_columns]), step, start, end,
                                                   steps, block_k)
                    plgpu.commit_smem()
                    if prologue is not None:
                        _meet(halves)
                    else:
                        pl.when(boundary)(lambda: _meet(halves))

                def product(lhs_smem, rhs_smem, acc_ref):
                    plgpu.wgmma(acc_ref, lhs_smem.transpose((1, 0)), rhs_smem.at[:, mine_columns])

                def mma(indices, lhs_smem, rhs_smem, *rest):
                    *side_smems, carry = rest
                    staged(indices[0], lhs_smem, rhs_smem, side_smems)
                    if not exact_sums:
                        product(lhs_smem, rhs_smem, carry)
                        # One MMA in flight: the next stage is prepared while this one runs.
                        plgpu.wgmma_wait(1)
                        return carry

                    def step_scope(step_ref):
                        product(lhs_smem, rhs_smem, step_ref)
                        plgpu.wgmma_wait(0)
                        return step_ref[...]

                    # Exact sums: the step's product apart, added into the running fp32 sum by an IEEE add.
                    return carry + pl.run_scoped(step_scope, plgpu.ACC((block_m, half), jnp.float32))

                def store(acc):
                    plgpu.wait_smem_to_gmem(0, wait_read_only=True)
                    o_smem[wg] = acc.astype(o_smem.dtype)
                    plgpu.commit_smem()
                    plgpu.copy_smem_to_gmem(o_smem.at[wg],
                                            o_gmem.at[g_i, depth, pl.ds(n_i * block_n + wg * half, half)])

                def context(run):
                    for i, ref, barrier in zip(fixed, fixed_refs, fixed_barriers, strict=True):
                        plgpu.copy_gmem_to_smem(side_gmems[i].at[tuple(depth for _ in prologue_axes[i])], ref,
                                                barrier.at[wg])
                        plgpu.barrier_wait(barrier.at[wg])
                    if exact_sums:
                        zeros = jnp.zeros((block_m, half), jnp.float32)
                        store(run(plgpu.layout_cast(zeros, plgpu.Layout.WGMMA)))
                        return

                    @functools.partial(pl.run_scoped, acc_ref=plgpu.ACC((block_m, half), jnp.float32))
                    def _(acc_ref):
                        run(acc_ref)
                        plgpu.wgmma_wait(0)
                        store(acc_ref[...])

                nonempty = sizes_gmem[g_i] > 0  # an empty group's tile is zero

                @pl.when(nonempty)
                def _():
                    pipeline(mma, context, steps)(*views(rows, depth, pl.ds(n_i * block_n, block_n)),
                                                  allocations=allocations)

                @pl.when(~nonempty & (wg < 2))
                def _():
                    store(jnp.zeros((block_m, half), jnp.float32))

        plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    kernel = plgpu.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((g, m, n), out_dtype),
        grid=(multiprocessors // pair,),
        grid_names=("cluster_grid",),
        cluster=(pair,),
        cluster_names=("cluster",),
        num_threads=3,
        thread_name="wg",
        compiler_params=_compiler_params(),
        interpret=interpret,
    )
    return kernel(*groups, lhs, rhs, *prologue_operands)


def _tgmm_pipeline(tiles: Tiles, side_axes: tuple, body, context, steps):
    """Dew: the cooperative `tgmm`'s pipeline over a group's `steps` row blocks: the rows' stage, the
    cotangent's (multicast to a pair), and each side operand's that has rows, in a producer warpgroup; a
    stage stays one step longer, for the MMA in flight."""
    extent = {'rows': tiles.block_k, 'depth': tiles.block_m}

    def side_spec(axes):
        return plgpu.BlockSpec(tuple(extent[axis] for axis in axes),
                               lambda step: tuple(step if axis == 'rows' else 0 for axis in axes),
                               memory_space=plgpu.SMEM, delay_release=1)

    return plgpu.emit_pipeline_warp_specialized(
        body,
        grid=(steps,),
        memory_registers=40,
        in_specs=[plgpu.BlockSpec((tiles.block_k, tiles.block_m), lambda step: (step, 0),
                                  memory_space=plgpu.SMEM, delay_release=1),
                  plgpu.BlockSpec((tiles.block_k, tiles.block_n), lambda step: (step, 0),
                                  memory_space=plgpu.SMEM, delay_release=1, collective_axes=_shared(tiles)),
                  *(side_spec(axes) for axes in side_axes)],
        wg_axis="wg",
        num_compute_wgs=2,
        max_concurrent_steps=tiles.stages,
        compute_context=context,
    )


def _zero_neighbours_of(refs: tuple, step, start, end, steps, block_k: int):
    """Dew: in a group's first and last row blocks, zero each of `refs` (`[block_k, columns]` stages) on the
    rows a neighbouring group holds, by selection; whether `step` is either block."""
    first, last = step == 0, step == steps - 1

    def zero(kept):
        for ref in refs:
            values = ref[...]
            rows = plgpu.layout_cast(lax.broadcasted_iota(jnp.int32, values.shape, 0), plgpu.Layout.WGMMA)
            ref[...] = jnp.where(kept(rows), values, jnp.zeros_like(values))

    pl.when(first)(lambda: zero(lambda rows: rows >= lax.rem(start, block_k)))
    pl.when(last)(lambda: zero(lambda rows: rows <= lax.rem(end - 1, block_k)))
    return first | last


def _group_blocks(group_sizes: jax.Array, block_k: int) -> tuple:
    """Each group's first row, end, count of `block_k`-row blocks, and its first block's first row."""
    group_starts = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(group_sizes)[:-1]]).astype(jnp.int32)
    group_ends = jnp.cumsum(group_sizes).astype(jnp.int32)
    group_block_starts = group_starts // block_k * block_k
    group_block_ends = -(group_ends // -block_k) * block_k
    return group_starts, group_ends, (group_block_ends - group_block_starts) // block_k, group_block_starts


def _tgmm_sides(side_gmems, axes_of, gmem_slice, m_i, tiles: Tiles) -> tuple[list, list]:
    """Dew: each prologue side operand's block spec and its view: its rows the group's slice as `lhs`'s,
    wherever they are among its dimensions, its depth the tile's."""
    block = {'rows': (tiles.block_k, lambda k_i: k_i), 'depth': (tiles.block_m, lambda k_i: m_i)}

    def spec(axes):
        return plgpu.BlockSpec(tuple(block[axis][0] for axis in axes),
                               lambda k_i: tuple(block[axis][1](k_i) for axis in axes),
                               delay_release=int(tiles.stages > 1))

    views = [side.at[tuple(gmem_slice if axis == 'rows' else slice(None) for axis in axes)]
             for side, axes in zip(side_gmems, axes_of, strict=True)]
    return [spec(axes) for axes in axes_of], views


def _zero_neighbours(lhs_smem, rhs_smem, block_idx, start, end, blocks, tiles: Tiles) -> None:
    """Dew: the first and last blocks of a group may hold a neighbour's rows, which both operands zero by
    selection, so a neighbour's NaN or infinity stays out."""
    @pl.when(block_idx == 0)
    def _():
        start_index = lax.rem(start, tiles.block_k)
        _zero_rows(lhs_smem, rhs_smem, lambda rows: rows >= start_index, tiles.block_k, tiles.block_m,
                   tiles.block_n)

    @pl.when(block_idx == blocks - 1)
    def _():
        last_index = lax.rem(end - 1, tiles.block_k)
        _zero_rows(lhs_smem, rhs_smem, lambda rows: rows <= last_index, tiles.block_k, tiles.block_m,
                   tiles.block_n)


def _summed_gradient(pipeline, operands, staged, tile, nonempty, tiles: Tiles) -> None:
    """Dew: a kernel gradient tile's sum over its group's row steps, each step's product into one
    accumulator by the warpgroup MMA, stored to `tile`; an empty group's zero."""
    def acc_scope(acc_ref):
        def block_matmul(block_idx, lhs_smem, rhs_smem, *side_smems):
            staged(block_idx[0], lhs_smem, rhs_smem, side_smems)
            plgpu.wgmma(acc_ref, lhs_smem.transpose((1, 0)), rhs_smem)
            if tiles.stages == 1:
                # Without delayed release, we won't have at least two separate smem blocks in flight, so
                # we cannot rely on the implicit wait of wgmma to guarantee that the data in smem is ready
                # to be overwritten by the next pipeline iteration.
                plgpu.wgmma_wait(0)

        @pl.when(nonempty)
        def _():
            pipeline(block_matmul)(*operands)

        return acc_ref[...]

    _store_gradient_tile(pl.run_scoped(acc_scope, plgpu.WGMMAAccumulatorRef((tiles.block_m, tiles.block_n))),
                         tile)


def _exactly_summed_gradient(pipeline, operands, staged, tile, nonempty, tiles: Tiles) -> None:
    """Dew: `_summed_gradient` with each step's product summed apart in a fresh accumulator, then added
    into the running fp32 sum, the pipeline's carry, by an IEEE add: a tensor core's accumulation
    truncates each add."""
    shape = (tiles.block_m, tiles.block_n)

    def exact_step(block_idx, lhs_smem, rhs_smem, *rest):
        *side_smems, running = rest
        staged(block_idx[0], lhs_smem, rhs_smem, side_smems)

        def step_scope(step_ref):
            plgpu.wgmma(step_ref, lhs_smem.transpose((1, 0)), rhs_smem)
            return step_ref[...]

        return running + pl.run_scoped(step_scope, plgpu.WGMMAAccumulatorRef(shape))

    zeros = plgpu.layout_cast(jnp.zeros(shape, jnp.float32), plgpu.Layout.WGMMA)

    @pl.when(nonempty)
    def _():
        _store_gradient_tile(pipeline(exact_step, init_carry=zeros)(*operands), tile)

    @pl.when(~nonempty)
    def _():
        _store_gradient_tile(zeros, tile)


def _store_gradient_tile(acc, o_gmem) -> None:
    """Dew: a kernel gradient's tile in the output's own dtype, an fp32 gradient above all."""
    @functools.partial(pl.run_scoped, o_smem=plgpu.SMEM(acc.shape, dtype=o_gmem.dtype))
    def store_scope(o_smem):
        o_smem[...] = acc.astype(o_smem.dtype)
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(o_smem, o_gmem)
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)


def _rows_padded(array: jax.Array, axis: int = 0) -> jax.Array:
    """`array` with zeros after its own along `axis` up to a multiple of 8, the rows a TMA descriptor tiles
    by."""
    pad = -array.shape[axis] % 8
    widths = tuple((0, pad if dim == axis else 0) for dim in range(array.ndim))
    return jnp.pad(array, widths) if pad else array


def _side_padded(side: jax.Array, axes: tuple) -> jax.Array:
    """A side operand padded as the rows are along the dimension that follows them, wherever it is."""
    return _rows_padded(side, axes.index('rows')) if 'rows' in axes else side


def _zero_rows(lhs_smem, rhs_smem, kept, block_k: int, block_m: int, block_n: int) -> None:
    """Zero the rows of both staged blocks, `[block_k, block_m]` and `[block_k, block_n]`, that `kept`
    (of each row's index) turns down, by selection."""
    for ref, columns in ((lhs_smem, block_m), (rhs_smem, block_n)):
        values = ref[...]
        rows = plgpu.layout_cast(lax.broadcasted_iota(jnp.int32, (block_k, columns), 0), plgpu.Layout.WGMMA)
        ref[...] = jnp.where(kept(rows), values, jnp.zeros_like(values))
    plgpu.commit_smem()
