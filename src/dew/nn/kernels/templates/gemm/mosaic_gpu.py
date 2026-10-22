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
  upstream, where upstream stated them with transforms whose type JAX leaves abstract;
- both take a cooperative tile (`Tiles.warpgroups`: two consumer warpgroups beside a producer, CUTLASS's
  cooperative sm90 schedule, as JAX's `hopper_matmul_mgpu`; `tgmm`'s operand stages stream across its
  persistent loop's tiles), `gmm` a pair of multiprocessors sharing one operand's blocks
  (`Tiles.cluster`), and the template's hooks (`gmm`, `tgmm`).

Each sums exact products of its 16-bit operands in fp32 (wgmma's accumulator) and rounds once to the
output dtype.
"""

import dataclasses
import functools

import jax
from jax import lax, numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import hopper


@dataclasses.dataclass(frozen=True)
class Tiles:
    """One launch's output tile (`block_m` x `block_n`), contraction step (`block_k`), the steps in flight
    (`stages`, upstream's `max_concurrent_steps`) and the column tiles in each band of the planar snake
    (`grid_block_n`); Dew: the warpgroups computing a tile (`warpgroups`: 2 halve `block_n` between two
    consumer warpgroups beside a producer that issues the loads, CUTLASS's cooperative sm90 schedule), and
    the multiprocessors of a cluster (`cluster`: 2 pair two row tiles of one group of `gmm`, which load
    their matrix block once between them, multicast), only for a cooperative tile; `tgmm` takes no pair."""
    block_m: int
    block_n: int
    block_k: int
    stages: int
    grid_block_n: int
    warpgroups: int = 1
    cluster: int = 1

    def __post_init__(self):
        paired = self.cluster == 2 and self.warpgroups == 1
        if self.warpgroups not in (1, 2) or self.cluster not in (1, 2) or paired:
            raise ValueError(f"{self}: one warpgroup alone, or two in a cluster of one or two "
                             "multiprocessors")
        if self.warpgroups == 2 and (self.block_n % 2 or self.block_m % 2):
            raise ValueError(f"{self}: two warpgroups halve the tile's columns and rows")
        if self.warpgroups == 2 and self.stages < 2:
            # A stage is released a step late (an MMA in flight), so one stage would wait on itself.
            raise ValueError(f"{self}: two warpgroups stream at least two stages")


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


def gmm(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, *, tiles: Tiles, transpose_rhs: bool,
        out_dtype, multiprocessors: int, interpret: bool | plgpu.InterpretGPUParams = False,
        prologue=None, prologue_operands: tuple = (), prologue_axes: tuple = (), epilogue=None,
        epilogue_operands: tuple = (), epilogue_axes: tuple = (),
        out_dtypes: tuple | None = None, layer: jax.Array | None = None) -> jax.Array | tuple[jax.Array, ...]:
    """Each group's rows of `lhs` `[rows, depth]` times its matrix of `rhs`, `[groups, depth, width]` or
    with `transpose_rhs` `[groups, width, depth]`: `[rows, width]` in `out_dtype`, the rows past the
    routed ones zero. The depth must divide by `tiles.block_k` and the width by `tiles.block_n`. Dew:
    with `layer`, an int32 scalar, `rhs` is a bank `[layers, groups, ...]` whose layer `layer` the kernel
    reads in place, its TMA loads offset by it, where slicing it out would copy it.

Dew: `prologue(tile, *side_tiles)` transforms each `[block_m, block_k]` tile of `lhs` (a cooperative
    tile's consumer, half its rows) in its shared memory stage before the warpgroup MMA reads it;
    `epilogue(acc, *side_tiles)` each fp32 accumulator tile (a cooperative consumer's, half its columns),
    returning one output or a tuple of them (`out_dtypes`). A side operand's dimensions follow the launch
    axes its `*_axes` entry names, 'rows' and 'depth' for the prologue's, 'rows' and 'columns' for the
    epilogue's, each brought to shared memory by TMA."""
    block_n, block_k = tiles.block_n, tiles.block_k
    rows, k = lhs.shape
    g, k2, n = rhs.shape[-3:]
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
    launch = _gmm_cooperative if tiles.warpgroups == 2 else _gmm_alone
    layers = () if layer is None else (jnp.reshape(layer.astype(jnp.int32), (1,)),)
    outs = launch(lhs, rhs, lengths, (m, n, k, g), tiles=tiles, transpose_rhs=transpose_rhs, dtypes=dtypes,
                  multiprocessors=multiprocessors, interpret=interpret,
                  hooks=(prologue, prologue_axes, epilogue, epilogue_axes),
                  sides=(prologue_operands, epilogue_operands), layers=layers)
    outs = tuple(out[:rows] for out in outs)
    return outs if out_dtypes else outs[0]


def _gmm_alone(lhs, rhs, lengths, shape: tuple, *, tiles: Tiles, transpose_rhs: bool, dtypes: tuple,
               multiprocessors: int, interpret, hooks: tuple, sides: tuple, layers: tuple = ()) -> tuple:
    """`gmm`'s product at a tile one warpgroup computes, loading and multiplying (upstream's kernel), with
    `_gmm_cooperative`'s arguments."""
    prologue, prologue_axes, epilogue, epilogue_axes = hooks
    prologue_operands, epilogue_operands = sides
    m, n, k, g = shape
    block_m, block_n, block_k = tiles.block_m, tiles.block_n, tiles.block_k
    side_count = len(prologue_operands) + len(epilogue_operands)

    def body(rows_per_expert_gmem, lhs_gmem, rhs_gmem, *refs):
        prologue_gmems = refs[:len(prologue_operands)]
        epilogue_gmems = refs[len(prologue_operands):side_count]
        o_gmems = refs[side_count:]
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

                    # TMA descriptors need static extents, but how many rows to store is known only at run
                    # time: a ladder of power-of-two stores (`hopper.store_rows`).
                    hopper.store_rows(o_smem, o_gmem, group_info.block_start, group_info.start_within_block,
                                  group_info.actual_size, ni * block_n, min(block_m, m))
                    plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    kernel = plgpu.kernel(
        _indexed(body, layers),
        # Dew: the output dtypes asked for, and the device's multiprocessors.
        out_type=tuple(jax.ShapeDtypeStruct((m, n), dtype) for dtype in dtypes),
        grid=(multiprocessors,),
        grid_names=("sm",),
        compiler_params=hopper.compiler_params(),
        interpret=interpret,
    )
    return kernel(lengths, *layers, lhs, rhs, *prologue_operands, *epilogue_operands)


def _gmm_cooperative(lhs, rhs, lengths, shape: tuple, *, tiles: Tiles, transpose_rhs: bool, dtypes: tuple,
                     multiprocessors: int, interpret, hooks: tuple, sides: tuple,
                     layers: tuple = ()) -> tuple:
    """Dew: `gmm` at a tile two consumer warpgroups compute (half its columns each) beside a producer
    issuing the loads (as `hopper_matmul_mgpu`); `tiles.cluster` 2 pairs two row tiles of one group, which
    load each matrix block once, multicast. A prologue's consumers each rewrite half the rows' shared stage,
    then meet. `lhs` is padded, `lengths` ends with the unrouted rows' group (`gmm`)."""
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
        prologue_gmems, side_gmems, o_gmems = _split(refs, len(prologue_operands), len(epilogue_operands))
        grid_m, grid_n = pl.cdiv(m, block_m * pair) + g, n // block_n  # pairs: g groups, the unrouted rows
        rhs_columns = functools.partial(_matrix_columns, block_n=block_n, transpose_rhs=transpose_rhs)

        def stepped_views(rows):
            return [prologue_gmems[i].at[tuple(rows if axis == 'rows' else slice(None)
                                               for axis in prologue_axes[i])] for i in stepped]

        @functools.partial(
            pl.run_scoped,
            allocations=pipeline(hopper.ignore, hopper.ignore).get_allocations(
                lhs_gmem.at[pl.ds(0, block_m)], rhs_columns(rhs_gmem.at[0], 0),
                *stepped_views(pl.ds(0, block_m))),
            o_smems=[plgpu.SMEM((2, block_m, half), gmem.dtype) for gmem in o_gmems],
            **hopper.staging('side', zip(side_shapes, side_gmems, strict=True)),
            **hopper.staging('fixed', [(tuple(extent[axis] for axis in prologue_axes[i]), prologue_gmems[i])
                                 for i in fixed]),
            halves=plgpu.Barrier(num_arrivals=2, num_barriers=2),
            drained=hopper.drained(tiles.cluster),
            collective_axes="wg",
        )
        def scope(allocations, o_smems, side_smems, side_barriers, fixed_smems, fixed_barriers, halves,
                  drained):
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
                    origins = {'rows': pl.ds(row_start, block_m), 'columns': pl.ds(column, half)}
                    views = [gmem.at[tuple(origins[axis] for axis in axes)]
                             for gmem, axes in zip(side_gmems, epilogue_axes, strict=True)]
                    _finish_half(acc, epilogue, (views, side_smems, side_barriers), (o_gmems, o_smems),
                                 (row_start, first, stored, column, min(block_m, m)), wg)

                def context(run):
                    hopper.load_fixed([prologue_gmems[i].at[pl.ds(row_start, block_m)] for i in fixed],
                                      fixed_refs, fixed_barriers, wg)

                    hopper.finished(run, (block_m, half), drained, finish)

                def mma(_, lhs_smem, rhs_smem, *rest):
                    *side_smems, acc_ref = rest
                    if prologue is not None:
                        # This warpgroup's half of the rows, then the other's, before an MMA reads them all.
                        tiles_of = dict(zip(stepped, side_smems, strict=True)) | dict(zip(fixed, fixed_refs,
                                                                                          strict=True))
                        _rewritten(prologue, prologue_axes, lhs_smem.at[mine_rows], tiles_of, 'rows',
                                   mine_rows)
                        hopper.meet(halves)
                    columns = pl.ds(wg * half, half)
                    matrix = (rhs_smem.at[columns].transpose((1, 0)) if transpose_rhs
                              else rhs_smem.at[:, columns])
                    plgpu.wgmma(acc_ref, lhs_smem, matrix)
                    # An MMA in flight: the next stage is prepared while this one runs.
                    plgpu.wgmma_wait(_IN_FLIGHT)
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
                    hopper.drain(drained, wg)

                # The unrouted rows, and a slot past the last tile, multiply nothing.
                @pl.when(~active & (wg < 2))
                def _():
                    finish(jnp.zeros((block_m, half), jnp.float32))

        plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    outs = tuple(jax.ShapeDtypeStruct((m, n), dtype) for dtype in dtypes)
    kernel = hopper.cooperative(_indexed(body, layers), outs,
                                consumers=2, cluster=tiles.cluster, multiprocessors=multiprocessors,
                                interpret=interpret)
    return kernel(lengths, *layers, lhs, rhs, *prologue_operands, *epilogue_operands)


def _indexed(body, layers: tuple):
    """Dew: a `gmm` kernel's `body` on a bank's layer where `layers` holds its index (`gmm`'s `layer`): the
    index read once, the matrices' ref offset by it, so every TMA load of a matrix block reads the layer
    in place."""
    if not layers:
        return body

    def on_layer(lengths_gmem, layer_gmem, lhs_gmem, rhs_gmem, *refs):
        return body(lengths_gmem, lhs_gmem, rhs_gmem.at[layer_gmem[0]], *refs)

    return on_layer


def _gmm_pipeline(tiles: Tiles, transpose_rhs: bool, steps: int, side_axes: tuple, body, context):
    """Dew: the cooperative `gmm`'s pipeline over the depth's `steps` blocks: the rows' stage, the matrix's
    (multicast to a pair), and each prologue side operand's that has depth, in a producer warpgroup; a
    stage stays one step longer, for the MMA in flight."""
    extent = {'rows': tiles.block_m, 'depth': tiles.block_k}

    def side_spec(axes):
        return hopper.stage_spec(axes, extent, 'depth', _IN_FLIGHT)

    rhs_block = (tiles.block_n, tiles.block_k) if transpose_rhs else (tiles.block_k, tiles.block_n)
    return plgpu.emit_pipeline_warp_specialized(
        body,
        grid=(steps,),
        memory_registers=hopper.PRODUCER_REGISTERS,
        in_specs=[plgpu.BlockSpec((tiles.block_m, tiles.block_k), lambda step: (0, step),
                                  memory_space=plgpu.SMEM, delay_release=_IN_FLIGHT),
                  plgpu.BlockSpec(rhs_block, _rhs_index(transpose_rhs), memory_space=plgpu.SMEM,
                                  delay_release=_IN_FLIGHT,
                                  collective_axes=hopper.multicast_axes(tiles.cluster)),
                  *(side_spec(axes) for axes in side_axes)],
        wg_axis="wg",
        num_compute_wgs=2,
        max_concurrent_steps=tiles.stages,
        compute_context=context,
    )


def _rewritten(prologue, axes_of: tuple, mine, tiles_of: dict, halved: str, half) -> None:
    """Dew: `mine`, this warpgroup's half of a stage both consumers read, rewritten by the prologue on its
    side tiles (`tiles_of`, by operand) cut to the same half along the `halved` launch axis."""
    cut = [tiles_of[i].at[tuple(half if axis == halved else slice(None) for axis in axes)][...]
           for i, axes in enumerate(axes_of)]
    mine[...] = prologue(mine[...], *cut).astype(mine.dtype)
    plgpu.commit_smem()


def _finish_half(acc, epilogue, sides: tuple, outs: tuple, place: tuple, wg) -> None:
    """Dew: a consumer warpgroup's half tile: the epilogue on its side tiles (`sides`: their views at the
    half tile, staging buffers and barriers), then its group's rows of each output (`outs`: the outputs
    and their staging buffers) by `hopper.store_rows` (`place`: the tile's first row, the group's first row
    in it and count, the half's first column, the most rows a rung stores)."""
    row_start, first, stored, column, most = place
    tiles_of = []
    for view, smem, barrier in zip(*sides, strict=True):
        plgpu.copy_gmem_to_smem(view, smem.at[wg], barrier.at[wg])
        plgpu.barrier_wait(barrier.at[wg])
        tiles_of.append(smem[wg])
    values = (acc,) if epilogue is None else epilogue(acc, *tiles_of)
    values = values if isinstance(values, tuple) else (values,)
    for o_gmem, o_smem, value in zip(*outs, values, strict=True):
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)
        o_smem[wg] = value.astype(o_smem.dtype)
        plgpu.commit_smem()
        hopper.store_rows(o_smem.at[wg], o_gmem, row_start, first, stored, column, most)


_IN_FLIGHT = 1
"""The MMAs the cooperative kernels keep in flight while they prepare the next stage. A pipeline's last
stage is released one step late, past its end, so a persistent kernel's producer would load the next
tile's first stages over one an MMA still reads: the consumers report each tile drained (`hopper.drain`;
a ring's producer waits for each slot's last release, `hopper.Ring.drain`)."""


def _split(refs: tuple, prologue: int, epilogue: int) -> tuple:
    """Dew: a kernel's refs after its operands: the prologue's side operands, the epilogue's, the outputs."""
    return refs[:prologue], refs[prologue:prologue + epilogue], refs[prologue + epilogue:]


def _matrix_columns(gmem, column_tile, *, block_n: int, transpose_rhs: bool):
    """Dew: a group's matrix at a column tile: its rows (`transpose_rhs`'s `[width, depth]`) or columns."""
    columns = pl.ds(column_tile * block_n, block_n)
    return gmem.at[columns] if transpose_rhs else gmem.at[:, columns]


def _rhs_index(transpose_rhs: bool):
    """A pipeline step's block of the matrix's column tile: its depth block `step`."""
    return (lambda step: (0, step)) if transpose_rhs else (lambda step: (step, 0))


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


def tgmm(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, *, tiles: Tiles, out_dtype,
         multiprocessors: int, interpret: bool | plgpu.InterpretGPUParams = False, prologue=None,
         prologue_operands: tuple = (), prologue_axes: tuple = (), exact_sums: bool = False) -> jax.Array:
    """Each group's rows of `lhs` `[rows, depth]` transposed times its rows of `rhs` `[rows, width]`:
    `[groups, depth, width]` in `out_dtype`, an empty group's zero. The depth must divide by
    `tiles.block_m` and the width by `tiles.block_n`, and the tile takes one multiprocessor;
    `tiles.block_k` is the rows a step takes.

    Dew: `prologue(rows, *side_tiles)` transforms each `[block_k, block_m]` stage of `lhs`'s rows (a
    cooperative tile's consumer, half its depth) in place before the group masking and the MMA; side
    operands follow 'rows' and 'depth' as `gmm`'s.
    `exact_sums` adds each step's product into the fp32 sum by an IEEE add, which holds two
    accumulators in registers."""
    block_m, block_n, block_k = tiles.block_m, tiles.block_n, tiles.block_k
    k, m = lhs.shape
    k2, n = rhs.shape
    g = group_sizes.shape[0]
    if k != k2 or m % block_m or n % block_n or tiles.cluster != 1:
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
                _zero_neighbours_of((lhs_smem, rhs_smem), block_idx, group_starts_gmem[g_i],
                                    group_ends_gmem[g_i], group_num_blocks_gmem[g_i], tiles.block_k)
                plgpu.commit_smem()

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
        compiler_params=hopper.compiler_params(),
        interpret=interpret,
    )
    return kernel(group_sizes, group_starts, group_ends, group_num_blocks, group_block_starts, lhs, rhs,
                  *prologue_operands)


def _tgmm_cooperative(operands: tuple, groups: tuple, shape: tuple, *, tiles: Tiles, out_dtype,
                      multiprocessors: int, interpret, prologue, prologue_operands: tuple,
                      prologue_axes: tuple, exact_sums: bool) -> jax.Array:
    """Dew: `tgmm`'s product at a tile two consumer warpgroups compute, each a half of its columns, beside a
    producer warpgroup issuing the loads (CUTLASS's cooperative sm90 schedule), the operand stages
    streaming across the persistent loop's tiles (`hopper.Ring`): the producer loads the next tile's first
    row blocks while the consumers store this one, which a group's few rows, a tile's few steps, make
    most of the time. Both warpgroups read the whole stage of the rows: each prepares a depth half of it
    (the prologue, the neighbouring groups' rows zeroed) and its own column half of the cotangent's, and
    each waits for the other's half before its MMA. A side operand with rows streams beside the operands;
    one without is loaded once a tile. One multiprocessor a tile. `operands` are padded and `groups` is
    `_group_blocks`' with the sizes first (`tgmm`)."""
    lhs, rhs = operands
    _, m, n, g = shape
    block_m, block_n, block_k = tiles.block_m, tiles.block_n, tiles.block_k
    half, depth_half = block_n // 2, block_m // 2
    grid = (m // block_m, n // block_n)
    stepped = tuple(i for i, axes in enumerate(prologue_axes) if 'rows' in axes)
    fixed = tuple(i for i, axes in enumerate(prologue_axes) if 'rows' not in axes)
    extent = {'rows': block_k, 'depth': block_m}

    def body(sizes_gmem, starts_gmem, ends_gmem, blocks_gmem, block_starts_gmem, lhs_gmem, rhs_gmem, *refs):
        *side_gmems, o_gmem = refs

        def tile(loop: plgpu.NDLoopInfo):
            """A tile's group, depth and columns, its group's first row block and its count (none for an
            empty group)."""
            g_i = loop.index[0]
            m_i, n_i = plgpu.planar_snake(loop.index[1], grid, 1, tiles.grid_block_n)
            first = pl.multiple_of(block_starts_gmem[g_i], block_k)
            steps = jnp.where(sizes_gmem[g_i] > 0, blocks_gmem[g_i], 0)
            return g_i, pl.ds(m_i * block_m, block_m), pl.ds(n_i * block_n, block_n), first, steps

        def views(rows, depth, columns):
            """Each streamed operand's block at a step's rows and a tile's depth and columns."""
            sides = [side_gmems[i].at[tuple(rows if axis == 'rows' else depth for axis in prologue_axes[i])]
                     for i in stepped]
            return lhs_gmem.at[rows, depth], rhs_gmem.at[rows, columns], *sides

        blocks = [((block_k, block_m), lhs_gmem), ((block_k, block_n), rhs_gmem),
                  *((tuple(extent[axis] for axis in prologue_axes[i]), side_gmems[i]) for i in stepped)]
        rings = {f'ring_{i}': hopper.ring(block, gmem.dtype, tiles.stages, consumers=2)
                 for i, (block, gmem) in enumerate(blocks)}

        @functools.partial(
            pl.run_scoped,
            o_smem=plgpu.SMEM((2, block_m, half), out_dtype),
            **hopper.staging('fixed', [(tuple(extent[axis] for axis in prologue_axes[i]), side_gmems[i])
                                 for i in fixed]),
            halves=plgpu.Barrier(num_arrivals=2, num_barriers=2),
            collective_axes="wg",
            **rings,
        )
        def scope(o_smem, fixed_smems, fixed_barriers, halves, **rings):
            streams = [hopper.Ring(**rings[f'ring_{i}']) for i in range(len(rings))]

            def producer():
                @plgpu.nd_loop((g, grid[0] * grid[1]), collective_axes="cluster_grid",
                               init_carry=jnp.int32(0))
                def loaded(loop: plgpu.NDLoopInfo, carry):
                    depth, columns, first, steps = tile(loop)[1:]

                    @pl.loop(0, steps)
                    def _(step):
                        rows = pl.ds(first + step * block_k, block_k)
                        for ring, view in zip(streams, views(rows, depth, columns), strict=True):
                            ring.load(carry + step, view)

                    return carry + steps

                for ring in streams:
                    ring.drain(loaded)

            def consumer(wg):
                mine_depth, mine_columns = pl.ds(wg * depth_half, depth_half), pl.ds(wg * half, half)
                fixed_refs = [smem.at[wg] for smem in fixed_smems]

                def store(acc, g_i, depth, columns):
                    plgpu.wait_smem_to_gmem(0, wait_read_only=True)
                    o_smem[wg] = acc.astype(o_smem.dtype)
                    plgpu.commit_smem()
                    mine = pl.ds(columns.start + wg * half, half)
                    plgpu.copy_smem_to_gmem(o_smem.at[wg], o_gmem.at[g_i, depth, mine])

                hook = None if prologue is None else (prologue, prologue_axes, stepped, fixed, fixed_refs)

                def release(counter):
                    for ring in streams:
                        ring.release(counter)

                @plgpu.nd_loop((g, grid[0] * grid[1]), collective_axes="cluster_grid",
                               init_carry=jnp.int32(0))
                def _(loop: plgpu.NDLoopInfo, carry):
                    g_i, depth, columns, _, steps = tile(loop)
                    start, end = starts_gmem[g_i], ends_gmem[g_i]
                    fixed_views = [side_gmems[i].at[tuple(depth for _ in prologue_axes[i])] for i in fixed]
                    # An empty group's tile reads none (`_streamed_gradient` stores its zeros).
                    pl.when(steps > 0)(lambda: hopper.load_fixed(fixed_views, fixed_refs, fixed_barriers, wg))
                    _streamed_gradient(
                        lambda counter, step: _staged_halves(
                            [ring.acquire(counter) for ring in streams], (step, start, end, steps, block_k),
                            (mine_depth, mine_columns), hook, halves),
                        release, steps, carry, (block_m, half), exact_sums,
                        lambda acc: store(acc, g_i, depth, columns))
                    return carry + steps

                plgpu.wait_smem_to_gmem(0, wait_read_only=True)

            hopper.roles(lax.axis_index("wg"), 2, consumer, producer)

    kernel = hopper.cooperative(body, jax.ShapeDtypeStruct((g, m, n), out_dtype), consumers=2, cluster=1,
                                multiprocessors=multiprocessors, interpret=interpret)
    return kernel(*groups, lhs, rhs, *prologue_operands)


def _staged_halves(stages: list, group: tuple, mine: tuple, hook, halves) -> tuple:
    """Dew: a streamed kernel gradient step's operands for one consumer warpgroup, from its `stages` (the
    rows', the cotangent's, then each side operand's with rows): this warpgroup's depth half of the rows
    rewritten by the prologue (`hook`: it, its operands' axes, which are stepped and which fixed, and the
    fixed ones' tiles), the neighbouring groups' rows zeroed in its halves (`group`: the step, the group's
    first and last row, its steps, the rows a step takes; `mine`: its depth and column halves), and the
    other's halves awaited at `halves` where it rewrote any: the rows (whole) and its cotangent columns."""
    rows_stage, cotangent_stage, *side_stages = stages
    mine_depth, mine_columns = mine
    if hook is not None:
        prologue, axes_of, stepped, fixed, fixed_refs = hook
        tiles_of = dict(zip(stepped, side_stages, strict=True)) | dict(zip(fixed, fixed_refs, strict=True))
        _rewritten(prologue, axes_of, rows_stage.at[:, mine_depth], tiles_of, 'depth', mine_depth)
    halves_of = (rows_stage.at[:, mine_depth], cotangent_stage.at[:, mine_columns])
    boundary = _zero_neighbours_of(halves_of, *group)
    plgpu.commit_smem()
    if hook is not None:
        hopper.meet(halves)
    else:
        pl.when(boundary)(lambda: hopper.meet(halves))
    return rows_stage.transpose((1, 0)), cotangent_stage.at[:, mine_columns]

def _streamed_gradient(staged, release, steps, first, shape: tuple, exact_sums: bool, store) -> None:
    """Dew: a streamed kernel gradient tile's `steps` steps (`staged(counter, step)` their operands at the
    ring's turn, `release(counter)` their stages back), summed into an fp32 tile of `shape` and stored: as
    the tensor cores add, an MMA in flight, or with `exact_sums` each step's product summed apart and
    added into the running sum by an IEEE add (a tensor core's accumulation truncates each add). An empty
    group's tile, no steps, is zero."""
    zeros = plgpu.layout_cast(jnp.zeros(shape, jnp.float32), plgpu.Layout.WGMMA)
    if exact_sums:
        def exact_step(step, running):
            rows_operand, cotangent_operand = staged(first + step, step)

            def step_scope(step_ref):
                plgpu.wgmma(step_ref, rows_operand, cotangent_operand)
                plgpu.wgmma_wait(0)
                return step_ref[...]

            product = pl.run_scoped(step_scope, plgpu.WGMMAAccumulatorRef(shape))
            release(first + step)
            return running + product

        store(lax.fori_loop(0, steps, exact_step, zeros))
        return

    @pl.when(steps > 0)
    def _():
        @functools.partial(pl.run_scoped, acc_ref=plgpu.WGMMAAccumulatorRef(shape))
        def _(acc_ref):
            _summed(acc_ref, staged, release, steps, first)
            store(acc_ref[...])

    pl.when(steps == 0)(lambda: store(zeros))


def _summed(acc_ref, staged, release, steps, first) -> None:
    """Dew: `steps` steps' products into `acc_ref` as the tensor cores add them, an MMA in flight:
    `staged(counter, step)` a step's operands at the ring's turn, `release(counter)` its stages back once
    the products have read them."""
    @pl.loop(0, steps)
    def _(step):
        plgpu.wgmma(acc_ref, *staged(first + step, step))
        # One MMA in flight: the step before's stages are read.
        plgpu.wgmma_wait(_IN_FLIGHT)
        pl.when(step > 0)(lambda: release(first + step - 1))

    plgpu.wgmma_wait(0)
    release(first + steps - 1)


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
