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
import math

import jax
from jax import lax, numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import mosaic, mosaic_sm80, mosaic_sm90
from dew.nn.kernels.templates.gemm import rows_padded

FAMILY: mosaic.Family = mosaic_sm90.WGMMA
"""Hopper's warpgroup MMA bundle, a wgmma tile's (`family_of`)."""


def family_of(tiles: "Tiles", dtype) -> mosaic.Family:
    """The bundle a tile runs on: its instruction family's (`Tiles.family`), mma.sync's a dtype's."""
    return mosaic_sm80.Mma(jnp.dtype(dtype)) if tiles.family == 'mma.sync' else FAMILY


@dataclasses.dataclass(frozen=True)
class Tiles:
    """One launch's output tile (`block_m` x `block_n`), contraction step (`block_k`), the steps in flight
    (`stages`, upstream's `max_concurrent_steps`) and the column tiles in each band of the planar snake
    (`grid_block_n`); Dew: the warpgroups computing a tile (`warpgroups`: 2 halve `block_m` between two
    consumer warpgroups beside a producer that issues the loads, CUTLASS's cooperative sm90 schedule), and
    the multiprocessors of a cluster (`cluster`: 2 pair two row tiles of one group of `gmm`, which load
    their matrix block once between them, multicast), only for a cooperative tile; `tgmm` takes no pair;
    and the instruction family's bundle that runs it (`family`: Hopper's warpgroup MMA, or Ampere's
    mma.sync, one warpgroup whose loads `emit_pipeline` issues, the plain product alone)."""
    block_m: int
    block_n: int
    block_k: int
    stages: int
    grid_block_n: int
    warpgroups: int = 1
    cluster: int = 1
    family: str = 'wgmma'

    def __post_init__(self):
        paired = self.cluster == 2 and self.warpgroups == 1
        if self.warpgroups not in (1, 2) or self.cluster not in (1, 2) or paired:
            raise ValueError(f"{self}: one warpgroup alone, or two in a cluster of one or two "
                             "multiprocessors")
        if self.family not in ('wgmma', 'mma.sync') or (self.family == 'mma.sync' and self.warpgroups != 1):
            raise ValueError(f"{self}: Hopper's warpgroup MMA, or Ampere's mma.sync on one warpgroup")
        if self.warpgroups == 2 and (self.block_m % 128 or self.block_n > 256):
            # Each consumer's half of the rows is warpgroup MMAs', a multiple of 64 rows by at most 256
            # columns.
            raise ValueError(f"{self}: two warpgroups halve the tile's rows into warpgroup MMAs")
        if self.warpgroups == 2 and self.stages < 2:
            # A stage is released a step late (an MMA in flight), so one stage would wait on itself.
            raise ValueError(f"{self}: two warpgroups stream at least two stages")


def derived(family, *, rows: int, depth: int | None, width: int, dtype, multiprocessors: int, holds,
            grouped: bool) -> Tiles | None:
    """The one tile a GEMM mainloop runs on `family` for a call whose output has `rows` rows (a group's,
    on average, where `grouped`: a product's rows split among its groups) and `width` columns and whose
    steps run over `depth` (None: a group's rows, padded to whole steps), from the family's facts, the
    call's extents and the device's multiprocessors alone (no table, no ranking;
    docs/design/kernel-templates.md §7). `holds(tile)` is the launch's legality on the device (its
    staged bytes within a block, its registers within a consumer's: `Template.legal`).

    - Columns: the widest the family's one product spans (`Family.widest_product`) that divides the
      call's width in its column unit, which reads its rows' slice fewest times a FLOP (each instruction
      reads the columns' block whole) and writes its output in the longest runs.
    - Consumers: as many roles as the family has (`Family.consumers`), a warpgroup each.
    - Rows: the most, in whole roles of its consumers, its consumers' registers hold
      (`mosaic.held_rows`, `mosaic.tile_registers`), and no more than the call's rows rounded up to a
      role's or, where grouped, half a group's in whole roles, at least one role (each group takes a
      partial tile more, so at half its rows that tile adds at most a third of its work).
    - Step: the family's widest depth dividing the call's (`Family.depths`) at those rows, a shallower
      one before fewer rows (a family whose products read registers holds a step's fragments).
    - Stages: the deepest a block holds (`holds`, `mosaic.deepest`), at least those the walk keeps busy
      (one loading ahead beside the one consumed and, on a family that keeps its products in flight,
      the one in flight: `_IN_FLIGHT`); a tile that cannot stage that deep gives way to the next
      narrower or shorter one.
    - Band: the planar snake's column tiles a band (`mosaic.band`), the rows' and the columns' blocks
      balanced over a wave or, grouped, wider while a wave spans each group's row tiles whole.

    None where no tile holds."""
    unit = family.column_unit(dtype)
    widest = min(width, family.widest_product or width)
    columns = [block_n for block_n in range(widest - widest % unit, 0, -unit) if width % block_n == 0]
    steps = [step for step in family.depths(dtype) if depth is None or depth % step == 0]
    consumers = family.consumers
    role = family.role_rows * consumers
    most_rows = max(role, rows // 2 // role * role if grouped else -(-rows // role) * role)
    busy = 2 + (0 if family.operands_in_registers else _IN_FLIGHT)
    fallback = None
    for block_n in columns:
        for block_m in range(min(most_rows, mosaic.held_rows(family, consumers, block_n)), 0, -role):
            band = mosaic.band(multiprocessors, block_m, block_n, width // block_n,
                               -(-rows // block_m) if grouped else None)
            # More rows before a deeper step: each reads the columns' block fewer times a FLOP, where a
            # step's depth only sets how many steps a tile takes (and, in registers, its fragments').
            for step in steps:
                tile = mosaic.deepest(
                    lambda stages, block_m=block_m, block_n=block_n, step=step, band=band: Tiles(
                        block_m, block_n, step, stages, band, warpgroups=consumers, family=family.name),
                    holds, family.most_shared // ((block_m + block_n) * step))
                if tile is None:
                    continue
                if tile.stages >= busy:
                    return tile
                fallback = fallback or tile
    return fallback


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
    def create(cls, table, tile, tid):
        """Dew: the group info of row tile `tid`, read from `tile_groups`' table (its group, its block, its
        group's first and past-last row) rather than walked over every group's length in each tile."""
        tile = jnp.int32(tile)
        group, block, group_start, group_end = (table[tid, column] for column in range(4))
        block_start = block * tile
        actual_start = jnp.maximum(group_start, block_start)
        actual_end = jnp.minimum(group_end, block_start + tile)
        start_within_block = actual_start - block_start
        actual_size = actual_end - actual_start
        return cls(group_id=group, block=block, block_start=block_start, actual_start=actual_start,
                   actual_end=actual_end, start_within_block=start_within_block, actual_size=actual_size)


def tile_groups(lengths: jax.Array, tile: int, tiles: int) -> jax.Array:
    """Dew: each row tile's group info for `GroupInfo.create`, in XLA before the launch: of tiles `0` to
    `tiles`, the group (`lengths`' index), its block of `tile` rows, and its group's first and past-last
    row, int32 `[tiles, 4]`; a tile no group reaches, zeros. Upstream's `GroupInfo.create` walked every
    group's length in every tile, unrolled: at 129 groups (Qwen3's 128 experts and the unrouted rows) two
    thirds of a down projection's instructions, and its global loads a third of its stalls (NCU-full)."""
    lengths = lengths.astype(jnp.int32)
    ends = jnp.cumsum(lengths)
    starts = ends - lengths
    index = jnp.arange(lengths.shape[0], dtype=jnp.int32)
    tile = jnp.int32(tile)
    # A group's tiles are its blocks shifted by its index (one tile per group boundary a block shares),
    # as upstream counted them, its divisions truncating as `lax.div` does.
    begin = lax.div(starts, tile) + index
    end = lax.div(ends - 1, tile) + 1 + index
    tids = jnp.arange(tiles, dtype=jnp.int32)[:, None]
    ours = (begin[None] <= tids) & (tids < end[None])
    # The last group whose range holds the tile, as upstream's selects let the last match win.
    group = (lengths.shape[0] - 1 - jnp.argmax(ours[:, ::-1], axis=1)).astype(jnp.int32)
    found = jnp.any(ours, axis=1)
    columns = (group, tids[:, 0] - group, starts[group], ends[group])
    return jnp.stack([jnp.where(found, column, 0) for column in columns], axis=1)

def gmm(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, *, tiles: Tiles, transpose_rhs: bool,
        out_dtype, multiprocessors: int, interpret: bool | plgpu.InterpretGPUParams = False,
        prologue=None, prologue_operands: tuple = (), prologue_axes: tuple = (), epilogue=None,
        epilogue_operands: tuple = (), epilogue_axes: tuple = (),
        out_dtypes: tuple | None = None) -> jax.Array | tuple[jax.Array, ...]:
    """Each group's rows of `lhs` `[rows, depth]` times its matrix of `rhs`, `[groups, depth, width]` or
    with `transpose_rhs` `[groups, width, depth]`: `[rows, width]` in `out_dtype`, the rows past the
    routed ones zero. The depth must divide by `tiles.block_k` and the width by `tiles.block_n`.

Dew: `prologue(tile, *side_tiles)` transforms each `[block_m, block_k]` tile of `lhs` (a cooperative
    tile's consumer, half its rows) in its shared memory stage before the warpgroup MMA reads it;
    `epilogue(acc, *side_tiles)` each fp32 accumulator tile (a cooperative consumer's, half its rows),
    returning one output or a tuple of them (`out_dtypes`). A side operand's dimensions follow the launch
    axes its `*_axes` entry names, 'rows' and 'depth' for the prologue's, 'rows' and 'columns' for the
    epilogue's, each brought to shared memory by TMA."""
    block_n, block_k = tiles.block_n, tiles.block_k
    rows, k = lhs.shape
    g, k2, n = rhs.shape
    if transpose_rhs:
        k2, n = n, k2
    if k != k2 or k % block_k or n % block_n:
        raise ValueError(f"a [{rows}, {k}] x [{g}, {k2}, {n}] product does not take {tiles}")
    # Dew: a TMA descriptor tiles the rows by 8, so they are padded to a multiple of 8, and the rows past
    # the routed ones, the padding among them, are a last group, whose tiles store zeros.
    lhs = rows_padded(lhs, 0, 8)  # the rows a TMA descriptor tiles by
    m = lhs.shape[0]
    lengths = jnp.concatenate([group_sizes.astype(jnp.int32),
                               jnp.maximum(m - jnp.sum(group_sizes, dtype=jnp.int32), 0)[None]])
    row_tile = tiles.block_m * tiles.cluster
    table = tile_groups(lengths, row_tile, pl.cdiv(m, row_tile) + g)
    prologue_operands = tuple(_side_padded(side, axes) for side, axes in zip(prologue_operands, prologue_axes,
                                                                               strict=True))
    epilogue_operands = tuple(_side_padded(side, axes) for side, axes in zip(epilogue_operands, epilogue_axes,
                                                                               strict=True))
    dtypes = tuple(jnp.dtype(dtype) for dtype in (out_dtypes or (out_dtype,)))
    outs = _gmm_mainloop(lhs, rhs, table, (m, n, k, g), tiles=tiles, transpose_rhs=transpose_rhs,
                         dtypes=dtypes, multiprocessors=multiprocessors, interpret=interpret,
                         hooks=(prologue, prologue_axes, epilogue, epilogue_axes),
                         sides=(prologue_operands, epilogue_operands))
    outs = tuple(out[:rows] for out in outs)
    return outs if out_dtypes else outs[0]


def _gmm_mainloop(lhs, rhs, table, shape: tuple, *, tiles: Tiles, transpose_rhs: bool, dtypes: tuple,
                  multiprocessors: int, interpret, hooks: tuple, sides: tuple) -> tuple:
    """Dew: `gmm` on its instruction family's bundle (`family_of`): `tiles.warpgroups` consumer roles, each
    its share of the rows by all the columns, one MMA a step (two: CUTLASS's cooperative split, 2-7% faster
    on every H100 scoreboard row than halving the columns), persistent over the tiles in a planar snake,
    the family owning the operands' streams across tiles, their staging and the outputs' stores
    (`mosaic.Family.persist`). Each consumer's prologue rewrites its share of the rows' slot, its epilogue
    takes its share of each side tile. `lhs` is padded; `table` is `tile_groups`' (`gmm`). One
    multiprocessor a tile."""
    family = family_of(tiles, lhs.dtype)
    if tiles.cluster != 1:
        raise ValueError(f"{tiles}: the bundles' streams go to one multiprocessor")
    prologue, prologue_axes, epilogue, epilogue_axes = hooks
    prologue_operands, epilogue_operands = sides
    m, n, k, g = shape
    block_m, block_n, block_k, consumers = tiles.block_m, tiles.block_n, tiles.block_k, tiles.warpgroups
    if not family.bounded_loads:
        # Its loads read past an operand unpredicated: a last row tile reads rows the padding holds.
        lhs = jnp.pad(lhs, ((0, -m % block_m), (0, 0)))
    share = block_m // consumers
    part = (share, block_n)
    side_shapes = [tuple({'rows': part[0], 'columns': part[1]}[axis] for axis in axes)
                   for axes in epilogue_axes]
    # A stream for each operand that follows the steps: a prologue side operand without depth is loaded
    # once a tile.
    stepped = tuple(i for i, axes in enumerate(prologue_axes) if 'depth' in axes)
    fixed = tuple(i for i, axes in enumerate(prologue_axes) if 'depth' not in axes)
    extent = {'rows': block_m, 'depth': block_k}
    steps = k // block_k
    grid = (pl.cdiv(m, block_m) + g, n // block_n)  # g groups' boundaries, and the unrouted rows
    rhs_block = (block_n, block_k) if transpose_rhs else (block_k, block_n)

    def body(table_gmem, lhs_gmem, rhs_gmem, *refs):
        prologue_gmems, side_gmems, o_gmems = _split(refs, len(prologue_operands), len(epilogue_operands))
        blocks = {'rows': ((block_m, block_k), lhs_gmem.dtype), 'matrix': (rhs_block, rhs_gmem.dtype),
                  **{f'side_{i}': (tuple(extent[axis] for axis in prologue_axes[i]), prologue_gmems[i].dtype)
                     for i in stepped}}

        def schedule(index):
            """A tile's whether it multiplies any row, its steps, its streams' views and its group and
            column tile."""
            mi, ni = plgpu.planar_snake(index[0], grid, 1, tiles.grid_block_n)
            group = GroupInfo.create(table_gmem, block_m, mi)
            views = _gmm_views(lhs_gmem, rhs_gmem, prologue_gmems, group, ni, tiles=tiles, groups=g,
                               transpose_rhs=transpose_rhs, axes_of={i: prologue_axes[i] for i in stepped})
            return (group.group_id < g) & (group.actual_size > 0), steps, views, (group, ni)

        def scope(side_smems, side_barriers, fixed_smems, fixed_barriers, *, streams):
            def consumer(wg, tile, walk, store, *, active):
                group, ni = tile
                row_start = group.block * block_m
                fixed_refs = [smem.at[wg] for smem in fixed_smems]
                if active:
                    mosaic.load_fixed([prologue_gmems[i].at[pl.ds(row_start, block_m)] for i in fixed],
                                      fixed_refs, fixed_barriers, wg)
                step = functools.partial(_gmm_step, family=family, prologue=prologue,
                                         prologue_axes=prologue_axes,
                                         fixed_tiles=dict(zip(fixed, fixed_refs, strict=True)),
                                         mine_rows=pl.ds(wg * share, share), transpose_rhs=transpose_rhs,
                                         interpret=bool(interpret))
                # The unrouted rows, and a slot past the last tile, multiply nothing: zeros.
                _, (acc,) = walk(step, None, (part,))
                # This consumer's share of the tile: its first row, and its group's rows there.
                part_row, column = row_start + wg * share, ni * block_n
                first = jnp.clip(group.actual_start - part_row, 0, part[0])
                stored = jnp.clip(group.actual_end - part_row, 0, part[0]) - first
                origins = {'rows': pl.ds(part_row, part[0]), 'columns': pl.ds(column, part[1])}
                family.loaded([(gmem.at[tuple(origins[axis] for axis in axes)], smem.at[wg], barrier.at[wg])
                               for gmem, axes, smem, barrier
                               in zip(side_gmems, epilogue_axes, side_smems, side_barriers, strict=True)])
                values = (acc,) if epilogue is None else epilogue(acc, *(smem[wg] for smem in side_smems))
                values = values if isinstance(values, tuple) else (values,)
                for output, (o_gmem, value) in enumerate(zip(o_gmems, values, strict=True)):
                    store(output, value, o_gmem, (part_row, first, stored, column, min(part[0], m)))

            family.persist(streams, (grid[0] * grid[1],), schedule, consumer, consumers=consumers,
                           axis="programs", outputs=tuple((part, dtype) for dtype in dtypes),
                           interpret=bool(interpret), in_flight=_IN_FLIGHT)

        staged_sides = mosaic.staging('side', zip(side_shapes, side_gmems, strict=True))
        staged_fixed = mosaic.staging('fixed', [(tuple(extent[axis] for axis in prologue_axes[i]),
                                                 prologue_gmems[i]) for i in fixed])
        family.scoped(scope, staged_sides['side_smems'], staged_sides['side_barriers'],
                      staged_fixed['fixed_smems'], staged_fixed['fixed_barriers'],
                      streams=family.streams(blocks, tiles.stages, consumers))

    kernel = family.kernel(body, out_type=tuple(jax.ShapeDtypeStruct((m, n), dtype) for dtype in dtypes),
                           grid=(multiprocessors,), grid_names=("programs",), consumers=consumers,
                           interpret=interpret, approx_math=False, name="dew_gmm")
    return kernel(table, lhs, rhs, *prologue_operands, *epilogue_operands)


def _gmm_views(lhs_gmem, rhs_gmem, prologue_gmems, group, column_tile, *, tiles: Tiles, groups: int,
               transpose_rhs: bool, axes_of: dict) -> dict:
    """Dew: `gmm`'s streams' views at a tile (`group`'s rows, `column_tile`; `mosaic.viewed`): its rows',
    its group's matrix's, and each prologue side operand's that follows the depth (`axes_of`, by
    operand), step `j` each one's `j`-th depth block."""
    rows = pl.ds(group.block * tiles.block_m, tiles.block_m)
    matrix = _matrix_columns(rhs_gmem.at[jnp.minimum(group.group_id, groups - 1)], column_tile,
                             block_n=tiles.block_n, transpose_rhs=transpose_rhs)
    return {'rows': (lhs_gmem.at[rows], lambda j: (0, j)), 'matrix': (matrix, _rhs_index(transpose_rhs)),
            **{f'side_{i}': (prologue_gmems[i].at[tuple(rows if axis == 'rows' else slice(None)
                                                        for axis in axes)],
                             lambda j, axes=axes: tuple(0 if axis == 'rows' else j for axis in axes))
               for i, axes in axes_of.items()}}


def _gmm_step(_, slots, nothing, accumulators, *, family: mosaic.Family, prologue, prologue_axes: tuple,
              fixed_tiles: dict, mine_rows, transpose_rhs: bool, interpret: bool):
    """Dew: a consumer's step of the cooperative `gmm` (`mosaic.Family.walk`): its half of the rows' slot,
    rewritten by the prologue on its side tiles (the streamed ones' slots, the fixed ones' `fixed_tiles`),
    times the matrix's slot into its accumulator."""
    stage = slots['rows']
    if prologue is not None:
        # This warpgroup's half of the rows, the only half its MMA reads.
        tiles_of = {i: fixed_tiles[i] if i in fixed_tiles else slots[f'side_{i}']
                    for i in range(len(prologue_axes))}
        _rewritten(prologue, prologue_axes, stage.at[mine_rows], tiles_of, 'rows', mine_rows)
        family.published()
    matrix = slots['matrix'].transpose((1, 0)) if transpose_rhs else slots['matrix']
    (acc,) = accumulators
    return nothing, (family.products(acc, stage.at[mine_rows], matrix, interpret),)


def _rewritten(prologue, axes_of: tuple, mine, tiles_of: dict, halved: str, half) -> None:
    """Dew: `mine`, this warpgroup's half of a stage, rewritten by the prologue on its side tiles
    (`tiles_of`, by operand) cut to the same half along the `halved` launch axis; its reader publishes
    the writes (`mosaic.Family.published`)."""
    cut = [tiles_of[i].at[tuple(half if axis == halved else slice(None) for axis in axes)][...]
           for i, axes in enumerate(axes_of)]
    mine[...] = prologue(mine[...], *cut).astype(mine.dtype)


_IN_FLIGHT = 1
"""The MMAs the kernels keep in flight while they prepare the next stage (`mosaic.Family.walk`): a step's
slots go back a step late, and a tile's last once its walk ends, before the next tile's loads reuse them."""


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
    # Dew: the rows padded to a multiple of 8, as `gmm`'s, past every group; to whole row blocks where the
    # family's loads read past an operand unpredicated (its last group's last block).
    multiple = 8 if family_of(tiles, lhs.dtype).bounded_loads else math.lcm(8, block_k)
    lhs, rhs = rows_padded(lhs, 0, multiple), rows_padded(rhs, 0, multiple)
    k = lhs.shape[0]
    prologue_operands = tuple(_side_padded(side, axes) for side, axes in zip(prologue_operands, prologue_axes,
                                                                               strict=True))

    group_sizes = group_sizes.astype(jnp.int32)
    group_starts, group_ends, group_num_blocks, group_block_starts = _group_blocks(group_sizes, block_k)
    out_dtype = jnp.dtype(out_dtype)
    return _tgmm_streamed(
        (lhs, rhs), (group_sizes, group_starts, group_ends, group_num_blocks, group_block_starts),
        (k, m, n, g), tiles=tiles, out_dtype=out_dtype, multiprocessors=multiprocessors,
        interpret=interpret, prologue=prologue, prologue_operands=prologue_operands,
        prologue_axes=prologue_axes, exact_sums=exact_sums)


def _tgmm_streamed(operands: tuple, groups: tuple, shape: tuple, *, tiles: Tiles, out_dtype,
                      multiprocessors: int, interpret, prologue, prologue_operands: tuple,
                      prologue_axes: tuple, exact_sums: bool) -> jax.Array:
    """Dew: `tgmm`'s product at a tile `tiles.warpgroups` consumer roles compute, each its share of the depth
    by all the columns, one MMA a step (two: CUTLASS's cooperative sm90 schedule), on its instruction
    family's bundle (`family_of`), its streams running across the persistent loop's tiles where the family
    streams so (the next tile's first row blocks load while the consumers store this one, which a group's
    few rows, a tile's few steps, make most of the time). Each consumer reads its depth share of the rows'
    stage, which it prepares (the prologue, the neighbouring groups' rows zeroed), and the whole
    cotangent's, a column share of which each zeroes, waiting for the other's where a group boundary falls
    in the step. A side operand with rows streams beside the operands; one without is loaded once a tile.
    One multiprocessor a tile. `operands` are padded and `groups` is `_group_blocks`' with the sizes first
    (`tgmm`)."""
    lhs, rhs = operands
    family = family_of(tiles, lhs.dtype)
    _, m, n, g = shape
    block_m, block_n, block_k = tiles.block_m, tiles.block_n, tiles.block_k
    consumers = tiles.warpgroups
    depth_half = block_m // consumers
    # Each consumer zeroes whole swizzled rows of the cotangent's stage (`_zeroed_share`), as many each as
    # the stage's divide among them, the last's share short where they do not divide evenly.
    atom = mosaic.SWIZZLE_BYTES // jnp.dtype(rhs.dtype).itemsize
    atoms_each = -(-(block_n // atom) // consumers)
    part = (depth_half, block_n)
    grid = (m // block_m, n // block_n)
    stepped = tuple(i for i, axes in enumerate(prologue_axes) if 'rows' in axes)
    fixed = tuple(i for i, axes in enumerate(prologue_axes) if 'rows' not in axes)
    extent = {'rows': block_k, 'depth': block_m}

    def body(sizes_gmem, starts_gmem, ends_gmem, blocks_gmem, block_starts_gmem, lhs_gmem, rhs_gmem, *refs):
        *side_gmems, o_gmem = refs

        def schedule(index):
            """A tile's whether its group has rows, its row blocks, its streams' views (step `j` its
            group's `j`-th row block at the tile's depth and columns), and its group, depth and columns."""
            g_i = index[0]
            m_i, n_i = plgpu.planar_snake(index[1], grid, 1, tiles.grid_block_n)
            depth, columns = pl.ds(m_i * block_m, block_m), pl.ds(n_i * block_n, block_n)
            first = pl.multiple_of(block_starts_gmem[g_i], block_k) // block_k
            steps = jnp.where(sizes_gmem[g_i] > 0, blocks_gmem[g_i], 0)

            def row_block(j):
                return first + j, 0

            views = {'ring_0': (lhs_gmem.at[:, depth], row_block),
                     'ring_1': (rhs_gmem.at[:, columns], row_block),
                     **{f'ring_{2 + position}': (
                         side_gmems[i].at[tuple(slice(None) if axis == 'rows' else depth
                                                for axis in prologue_axes[i])],
                         lambda j, axes=prologue_axes[i]: tuple(first + j if axis == 'rows' else 0
                                                                for axis in axes))
                        for position, i in enumerate(stepped)}}
            return steps > 0, steps, views, (g_i, depth, columns, steps)

        blocks = [((block_k, block_m), lhs_gmem), ((block_k, block_n), rhs_gmem),
                  *((tuple(extent[axis] for axis in prologue_axes[i]), side_gmems[i]) for i in stepped)]
        fixed_staging = mosaic.staging('fixed', [(tuple(extent[axis] for axis in prologue_axes[i]),
                                                  side_gmems[i]) for i in fixed])

        def scope(fixed_smems, fixed_barriers, halves, *, streams):
            names = tuple(f'ring_{i}' for i in range(len(blocks)))

            def consumer(wg, tile, walk, store, *, active):
                g_i, depth, columns, steps = tile
                mine_depth = pl.ds(wg * depth_half, depth_half)
                mine_columns = (wg * atoms_each, atoms_each, atoms_each * consumers > block_n // atom)
                fixed_refs = [smem.at[wg] for smem in fixed_smems]
                hook = None if prologue is None else (prologue, prologue_axes, stepped, fixed, fixed_refs)
                group = (starts_gmem[g_i], ends_gmem[g_i], steps, block_k)
                if active:
                    mosaic.load_fixed([side_gmems[i].at[tuple(depth for _ in prologue_axes[i])]
                                       for i in fixed], fixed_refs, fixed_barriers, wg)

                def staged_at(index, slots):
                    return _staged_halves(family, [slots[name] for name in names], (index, *group),
                                          (mine_depth, mine_columns), hook,
                                          halves if consumers == 2 else None)

                # An empty group's tile reads none and stores zeros.
                summed = _walked_gradient(family, walk, staged_at, part, exact_sums, bool(interpret))
                store(0, summed, o_gmem.at[g_i],
                      (depth.start + wg * depth_half, 0, depth_half, columns.start, depth_half))

            family.persist(streams, (g, grid[0] * grid[1]), schedule, consumer, consumers=consumers,
                           axis="programs", outputs=((part, out_dtype),), interpret=bool(interpret),
                           in_flight=_IN_FLIGHT)

        family.scoped(scope, fixed_staging['fixed_smems'], fixed_staging['fixed_barriers'],
                      mosaic.meeting(consumers),
                      streams=family.streams({f'ring_{i}': (block, gmem.dtype)
                                              for i, (block, gmem) in enumerate(blocks)}, tiles.stages,
                                             consumers))

    kernel = family.kernel(body, out_type=jax.ShapeDtypeStruct((g, m, n), out_dtype),
                           grid=(multiprocessors,), grid_names=("programs",), consumers=consumers,
                           interpret=interpret, approx_math=False, name="dew_tgmm")
    return kernel(*groups, lhs, rhs, *prologue_operands)


def _staged_halves(family: mosaic.Family, stages: list, group: tuple, mine: tuple, hook, halves) -> tuple:
    """Dew: a streamed kernel gradient step's operands for one consumer warpgroup, from its `stages` (the
    rows', the cotangent's, then each side operand's with rows): its depth half of the rows, rewritten by
    the prologue (`hook`: it, its operands' axes, which are stepped and which fixed, and the fixed ones'
    tiles), and the whole cotangent, the neighbouring groups' rows zeroed in its shares of both (`group`:
    the step, the group's first and last row, its steps, the rows a step takes; `mine`: its depth half,
    and its first swizzled row of the cotangent's, how many, and whether the last consumer's share runs
    short of them), the other's share awaited at `halves`
    where it zeroed any. Both consumers' MMAs read the whole cotangent stage, so each zeroes its share of
    it and they meet where a group boundary falls in the step, the kernel's one exchange between them."""
    rows_stage, cotangent_stage, *side_stages = stages
    mine_depth, (first_atom, atoms_each, short) = mine
    if hook is not None:
        prologue, axes_of, stepped, fixed, fixed_refs = hook
        tiles_of = dict(zip(stepped, side_stages, strict=True)) | dict(zip(fixed, fixed_refs, strict=True))
        _rewritten(prologue, axes_of, rows_stage.at[:, mine_depth], tiles_of, 'depth', mine_depth)
    atom = mosaic.SWIZZLE_BYTES // jnp.dtype(cotangent_stage.dtype).itemsize
    atoms = cotangent_stage.shape[1] // atom
    # Its depth half's swizzled rows, each its own, and its share of the cotangent's, the last consumer's
    # past the stage's end held (`atoms_each` rounds the share up).
    shares = [(rows_stage.at[:, pl.ds(mine_depth.start + column, atom)], None)
              for column in range(0, mine_depth.size, atom)]
    shares += [(cotangent_stage.at[:, pl.ds((first_atom + j) * atom, atom)],
                first_atom + j < atoms if short else None) for j in range(atoms_each)]
    boundary = _zero_neighbours_of(family, shares, *group)
    family.published()
    if halves is not None:
        pl.when(boundary)(lambda: mosaic.meet(halves))
    return rows_stage.at[:, mine_depth].transpose((1, 0)), cotangent_stage


def _walked_gradient(family: mosaic.Family, walk, staged, shape: tuple, exact_sums: bool, interpret: bool):
    """Dew: a streamed kernel gradient tile's steps (`walk`, a tile's of `mosaic.Family.persist`;
    `staged(index, slots)` their operands), summed into an fp32 tile of `shape`: as the tensor cores add, or
    with `exact_sums` each step's product summed apart and added into the running sum by an IEEE add (a
    tensor core's accumulation truncates each add). A tile of no steps is zero."""
    if exact_sums:
        def exact_step(index, slots, running, accumulators):
            lhs, rhs = staged(index, slots)
            if interpret:
                # The in-place product's own arithmetic: the CPU's dots sum in IEEE fp32 either way.
                return family.products(running, lhs, rhs, interpret=True), accumulators
            return running + family.product(lhs, rhs, shape), accumulators

        running, _ = walk(exact_step, family.zeros(shape), ())
        return running

    def step(index, slots, nothing, accumulators):
        (acc,) = accumulators
        lhs, rhs = staged(index, slots)
        return nothing, (family.products(acc, lhs, rhs, interpret),)

    _, (summed,) = walk(step, None, (shape,))
    return summed


def _zero_neighbours_of(family: mosaic.Family, refs: list, step, start, end, steps, block_k: int):
    """Dew: in a group's first and last row blocks, zero each of `refs` (`[block_k, columns]` stages, each
    one swizzled row wide, so one holds a thread's few registers whatever the tile's width; each with
    whether it is held, None where always) on the rows a neighbouring group holds, by selection; whether
    `step` is either block."""
    first, last = step == 0, step == steps - 1

    def zero(kept):
        for ref, held in refs:
            def zeroed(ref=ref):
                values = family.tile_of(ref)
                rows = family.iota((values.shape[0], values.shape[1]), 0)
                ref[...] = jnp.where(kept(rows), values, jnp.zeros_like(values))

            if held is None:
                zeroed()
            else:
                pl.when(held)(zeroed)

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


def _side_padded(side: jax.Array, axes: tuple) -> jax.Array:
    """A side operand padded as the rows are along the dimension that follows them, wherever it is."""
    return rows_padded(side, axes.index('rows'), 8) if 'rows' in axes else side
