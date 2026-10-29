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
from collections.abc import Callable

import jax
from jax import lax, numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import bundles, mosaic, mosaic_sm80, mosaic_sm90, resources, tail as tail_of
from dew.nn.kernels.templates.gemm import cost, rows_padded

FAMILY: mosaic.Family = mosaic_sm90.WGMMA
"""Hopper's warpgroup MMA bundle, a wgmma tile's (`family_of`)."""


def family_of(tiles: "Tiles", dtype) -> mosaic.Family:
    """The bundle a tile runs on: the one it names (`Tiles.family`, `bundles.named`), mma.sync's a dtype's
    with its warps arranged for a consumer's share of the tile (`mosaic_sm80.Mma.for_tile`)."""
    if tiles.family != mosaic_sm80.Mma.name:
        return bundles.named(tiles.family, dtype)
    return mosaic_sm80.Mma.for_tile(dtype, tiles.block_m // tiles.warpgroups, tiles.block_n)


@dataclasses.dataclass(frozen=True)
class Tiles:
    """One launch's output tile (`block_m` x `block_n`), contraction step (`block_k`), the steps in flight
    (`stages`, upstream's `max_concurrent_steps`) and the column tiles in each band of the planar snake
    (`grid_block_n`); Dew: the warpgroups computing a tile (`warpgroups`: 2 halve `block_m` between two
    consumer warpgroups beside a producer that issues the loads, CUTLASS's cooperative sm90 schedule), and
    the multiprocessors of a cluster (`cluster`: 2 pair two row tiles of one group of `gmm`, which load
    their matrix block once between them, multicast), only for a cooperative tile; `tgmm` takes no pair;
    and the bundle that runs it (`family`, `bundles.named`: Hopper's warpgroup MMA, its loads a producer
    warpgroup's or, 'wgmma.fed', its first consumer's, or Ampere's mma.sync, one warpgroup whose loads
    `emit_pipeline` issues, the plain product alone); and the programs
    a multiprocessor runs at once (`programs`, at most its family's, `mosaic.Family.programs`), each in its
    share of the multiprocessor's shared memory (`resources.program_shared`); and the slices a kernel
    gradient's contraction splits into (`splits`, `tgmm`: each group's rows in that many runs of whole steps,
    each run's fp32 partial summed after), where its output tiles underfill a wave (`split_count`)."""
    block_m: int
    block_n: int
    block_k: int
    stages: int
    grid_block_n: int
    warpgroups: int = 1
    cluster: int = 1
    family: str = 'wgmma'
    programs: int = 1
    splits: int = 1

    def __post_init__(self):
        paired = self.cluster == 2 and self.warpgroups == 1
        if self.warpgroups not in (1, 2) or self.cluster not in (1, 2) or paired:
            raise ValueError(f"{self}: one warpgroup alone, or two in a cluster of one or two "
                             "multiprocessors")
        warpgroup = self.family in (mosaic_sm90.WGMMA.name, mosaic_sm90.FED.name)
        if not warpgroup and self.family != mosaic_sm80.Mma.name:
            raise ValueError(f"{self}: Hopper's warpgroup MMA (a producer's or not), or Ampere's mma.sync")
        if self.cluster > bundles.named(self.family, jnp.bfloat16).cluster:
            raise ValueError(f"{self}: a cluster of at most its bundle's (`mosaic.Family.cluster`)")
        wide = warpgroup and self.block_n > 256
        if self.warpgroups == 2 and (self.block_m % 128 or wide):
            # Each consumer's half of the rows is a role's, a multiple of 64 rows, and on warpgroup MMA at
            # most 256 columns.
            raise ValueError(f"{self}: two warpgroups halve the tile's rows into whole roles")
        if self.warpgroups == 2 and self.stages < 2:
            # A stage is released a step late (an MMA in flight), so one stage would wait on itself.
            raise ValueError(f"{self}: two warpgroups stream at least two stages")


def derived(layouts: tuple, *, rows: int, depth: int | None, width: int, dtype, multiprocessors: int, target,
            candidate, launch_of, grouped: bool) -> Tiles | None:
    """The one tile a GEMM mainloop runs on an instruction family's `layouts` (`bundles.layouts`, its
    bundles in preference order, which share its facts but their loaders and registers) for a call whose
    output has `rows` rows (a group's,
    on average, where `grouped`: a product's rows split among its groups) and `width` columns and whose
    steps run over `depth` (None: a group's rows, padded to whole steps), from the family's facts, the
    call's extents and the device's multiprocessors alone (no table, no ranking;
    docs/design/kernel-templates.md §7). `candidate(tile)` is the launch's legality apart from its
    resources (its granularity, its blocks dividing the call, its side operands' copies), and
    `launch_of(tile)` the launch whose ring and residency the planner sizes (`resources.plan`: its
    allocations a stage, its registers within a consumer's) on `target`.

    - Layout: at each shape and step, the first layout whose planner holds it (`resources.placed`): a
      producer's where its consumers' registers hold the tile, else one whose first consumer issues the
      loads, its consumers holding the program's entry registers (`mosaic.ConsumerFed`).
    - Shape: the tile loading its operands fewest times a FLOP within its consumers' registers: an
      accumulator of `resources.held_area` elements A reads them 1/rows + 1/columns times a FLOP, least at
      rows = columns = sqrt(A). So the rows are sqrt(A) in whole roles (no more than the call's, below);
      the columns the widest dividing the call's width in its column unit, at most the family's one
      product's (`Family.widest_product`), that A holds at those rows; the rows then the most its
      registers hold at those columns (`resources.held_rows`). mma.sync's one consumer so takes 128 x 128
      where it took 64 x 256, whose four warps, splitting its rows, each read the columns' block whole:
      1.7x the shared memory a FLOP (SHAPE1-a100: 0.887 of its time). A tile its registers or a block
      cannot stage busy (`holds`, `resources.tile_registers`) gives way to the next narrower or shorter one.
    - Consumers: as many roles as the family has (`Family.consumers`), a warpgroup each, or one where
      rows a tile must divide (not grouped) take no whole role for each, or where no tile of them holds.
    - Programs: the most a multiprocessor runs at once that the family's registers hold at those
      consumers (`Family.programs`) and whose share of its shared memory (`resources.program_shared`) holds
      a tile, one at least; a wave is all of them.
    - Rows: no more than the call's rows rounded up to a role's or, where grouped, half a group's in
      whole roles, at least one role (each group takes a partial tile more, so at half its rows that tile
      adds at most a third of its work).
    - Step: the family's widest depth dividing the call's (`Family.depths`) at those rows, a shallower
      one before fewer rows (a family whose products read registers holds a step's fragments).
    - Stages: the deepest its program's share of shared memory holds at those programs (`resources.plan`,
      the planner every template's derivation reads), at least those the walk keeps busy (one loading
      ahead beside the one consumed and, on a family that keeps its products in flight, the one in
      flight: `_IN_FLIGHT`); a tile that cannot stage that deep gives way to the next narrower or
      shorter one.
    - Cluster: where grouped and a group's rows span at least as many row tiles as the family's cluster
      takes (`Family.cluster`: a pair on warpgroup MMA), that many multiprocessors a cluster, running row
      tiles of one group that load its matrix block once between them (multicast), CUTLASS's and
      cuBLASLt's pair on an H100; each multiprocessor's tile, stages and staged bytes as alone, its
      legality unchanged.
    - Band: the planar snake's column tiles a band (`resources.band`), the rows' and the columns' blocks
      balanced over a wave or, grouped, wider while a wave spans each group's row tiles whole, a
      cluster's tiles one program's.

    None where no tile holds."""
    # The facts the layouts share (their instruction family's); each layout's registers bound the shapes.
    family = layouts[0]
    unit = family.column_unit(dtype)
    widest = min(width, family.widest_product or width)
    columns = [block_n for block_n in range(widest - widest % unit, 0, -unit) if width % block_n == 0]
    steps = [step for step in family.depths(dtype) if depth is None or depth % step == 0]
    busy = 2 + (0 if family.operands_in_registers else _IN_FLIGHT)

    def tiled(consumers: int, programs: int) -> Tiles | None:
        role = family.role_rows * consumers
        most_rows = max(role, rows // 2 // role * role if grouped else -(-rows // role) * role)
        shaped = functools.partial(_shaped, consumers=consumers, programs=programs, rows=rows, width=width,
                                   multiprocessors=multiprocessors, grouped=grouped)

        held_tile = functools.partial(_held_tile, role=role, steps=steps, busy=busy, shaped=shaped,
                                      candidate=candidate, launch_of=launch_of, layouts=layouts,
                                      target=target, programs=programs)
        # The width whose tile reads its operands fewest times a FLOP (1/rows + 1/columns), the wider on a
        # tie, among those staging deep enough to stay busy where any does. mma.sync's one consumer at
        # Qwen3-8B's gate/up so takes 128 x 128, its 192-wide tile's rows shortened to 64 by its stages
        # (0.016 against 0.021 reads a FLOP; 64 x 192 ran 1.21-1.30x XLA's time against 128 x 128's
        # 1.10-1.14, FSA100POST32).
        held = [held_tile(block_n, held_rows)
                for block_n, held_rows in _held_widths(layouts, consumers, columns, role, most_rows)]
        busy_tiles = [tile for tile, staged in held if tile is not None and staged]
        tiles = busy_tiles or [tile for tile, _ in held if tile is not None]
        return min(tiles, key=lambda tile: (1 / tile.block_m + 1 / tile.block_n, -tile.block_n), default=None)

    # Rows a tile divides (a kernel gradient's) that take no whole role for each consumer take one, as
    # does a call no cooperative tile holds (a wide epilogue's outputs past a block).
    whole = grouped or rows % (family.role_rows * family.consumers) == 0
    return next((tile for consumers in dict.fromkeys((family.consumers if whole else 1, 1))
                 for programs in range(family.programs(consumers), 0, -1)
                 if (tile := tiled(consumers, programs)) is not None), None)


def _held_tile(block_n: int, held_rows: int, *, role: int, steps: list[int], busy: int, shaped: Callable,
               candidate: Callable, launch_of: Callable, layouts: tuple, target, programs: int
               ) -> tuple["Tiles | None", bool]:
    """`derived`'s tile at a width: the rows its registers hold there (`held_rows`), shortened where its
    stages do not fit, at the first step that plans (a step's depth sets how many steps a tile takes, not
    how many times a FLOP reads each operand); and whether it stages `busy` deep, enough to stay busy."""
    fallback = None
    for block_m in range(held_rows, 0, -role):
        for step in steps:
            def launches_of(layout, block_m=block_m, step=step):
                shape = shaped(layout, block_m, block_n, step)
                return [launch_of(shape)] if candidate(shape) else None

            found = resources.placed(launches_of, layouts, target, programs=programs)
            if found is None:
                continue
            layout, planned = found
            tile = dataclasses.replace(shaped(layout, block_m, block_n, step), stages=planned[0].stages)
            if tile.stages >= busy:
                return tile, True
            fallback = fallback or tile
    return fallback, False


def _held_widths(layouts: tuple, consumers: int, columns: list[int], role: int,
                 most_rows: int) -> list[tuple[int, int]]:
    """`derived`'s widths at `consumers` and the rows each holds in its consumers' registers (its Shape):
    every column width up to the widest that holds a role's rows, its registers the most generous of
    `layouts`' (a layout holding fewer gives way at placement, `resources.placed`)."""
    if not columns:
        return []
    area = max(resources.held_area(layout, consumers) for layout in layouts)
    side = max(role, min(most_rows, math.isqrt(area) // role * role))
    widest = next((block_n for block_n in columns if block_n * side <= area), columns[-1])

    def held(block_n: int) -> int:
        return min(most_rows, max(resources.held_rows(layout, consumers, block_n) for layout in layouts))

    return [(block_n, held(block_n)) for block_n in columns if block_n <= widest and held(block_n) >= role]


def _shaped(layout, block_m: int, block_n: int, step: int, *, consumers: int, programs: int, rows: int,
            width: int, multiprocessors: int, grouped: bool) -> Tiles:
    """`derived`'s tile of a shape and step on `layout`: a pair of multiprocessors a cluster runs two row
    tiles of one group where its layout's loads multicast and a group averages a tile's rows, which then
    read its matrix's blocks once; a pair is two consumers' (`Tiles`), so a one-consumer tile, the
    fallback, runs alone."""
    cluster = layout.cluster if grouped and consumers > 1 and rows >= block_m else 1
    band = resources.band(multiprocessors * programs, block_m, block_n, width // block_n,
                          -(-rows // (block_m * cluster)) if grouped else None, cluster=cluster)
    return Tiles(block_m, block_n, step, 2, band, warpgroups=consumers, cluster=cluster, family=layout.name,
                 programs=programs)


@dataclasses.dataclass(frozen=True)
class GroupInfo:
    """Information regarding the group being processed in a block."""

    group_id: jax.Array
    block_start: jax.Array
    actual_start: jax.Array
    actual_end: jax.Array
    start_within_block: jax.Array
    actual_size: jax.Array

    @classmethod
    def create(cls, table, tile, tid):
        """Dew: the group info of row tile `tid`, read from `tile_groups`' table (its group, its first row,
        its group's first and past-last row) rather than walked over every group's length in each tile."""
        tile = jnp.int32(tile)
        group, block_start, group_start, group_end = (table[tid, column] for column in range(4))
        # Mosaic's copies and stores read a slice's start as a multiple of the rows its tiling takes.
        block_start = pl.multiple_of(block_start, TMA_ROWS)
        actual_start = jnp.maximum(group_start, block_start)
        actual_end = jnp.minimum(group_end, block_start + tile)
        start_within_block = actual_start - block_start
        actual_size = actual_end - actual_start
        return cls(group_id=group, block_start=block_start, actual_start=actual_start,
                   actual_end=actual_end, start_within_block=start_within_block, actual_size=actual_size)

    @classmethod
    def dense(cls, tile: int, tid, rows: int):
        """A dense row tile's bounds, with a final tile clipped to the padded operand's rows."""
        start = pl.multiple_of(tid * tile, TMA_ROWS)
        size = jnp.minimum(tile, rows - start)
        return cls(group_id=jnp.int32(0), block_start=start, actual_start=start,
                   actual_end=start + size, start_within_block=jnp.int32(0), actual_size=size)


def tile_groups(lengths: jax.Array, tile: int, tiles: int) -> jax.Array:
    """Dew: each row tile's group info for `GroupInfo.create`, in XLA before the launch: of tiles `0` to
    `tiles`, the group (`lengths`' index), the tile's first row, and its group's first and past-last row,
    int32 `[tiles, 4]`; a tile no group reaches, zeros. A group's tiles start at its first row (rounded
    down to the 8 rows a TMA descriptor tiles by, `TMA_ROWS`), whole tiles after it, as CUTLASS's grouped
    GEMM tiles each group: a group of up to a tile's rows reads its matrix once, where tiles on the
    product's own row grid read it once a tile the group touches (two for most of Kimi Linear's
    128-row experts at 128-row tiles). The tiles number at most `cdiv(rows, tile)` plus the groups less
    one (a ceiling's sum exceeds the sum's ceiling by at most one a part). Upstream's
    `GroupInfo.create` walked every group's length in every tile, unrolled: at 129 groups (Qwen3's 128
    experts and the unrouted rows) two thirds of a down projection's instructions, and its global loads a
    third of its stalls (NCU-full)."""
    lengths = lengths.astype(jnp.int32)
    ends = jnp.cumsum(lengths)
    starts = ends - lengths
    rows = jnp.int32(tile)
    first_rows = starts // TMA_ROWS * TMA_ROWS
    counts = lax.div(ends - first_rows + rows - 1, rows) * (lengths > 0)
    firsts = jnp.cumsum(counts) - counts
    tids = jnp.arange(tiles, dtype=jnp.int32)[:, None]
    ours = (firsts[None] <= tids) & (tids < (firsts + counts)[None])
    group = jnp.argmax(ours, axis=1).astype(jnp.int32)
    found = jnp.any(ours, axis=1)
    block_start = first_rows[group] + (tids[:, 0] - firsts[group]) * rows
    columns = (group, block_start, starts[group], ends[group])
    return jnp.stack([jnp.where(found, column, 0) for column in columns], axis=1)


TMA_ROWS = 8
"""The rows a TMA descriptor tiles by, to which `gmm` pads its rows and a group's first tile starts."""


@dataclasses.dataclass(frozen=True)
class Allocations:
    """Dew: a GEMM launch's shared memory at its tile, as its kernel scopes it (`gmm_allocations`,
    `tgmm_allocations`): its streams' blocks `(shape, dtype)` by name (`mosaic.Family.streams`), what it
    scopes beside them in the kernel's order (`staged`: side operands' staging, `mosaic.staging`, a meeting's
    barriers), and its outputs' tiles `(shape, dtype)`, which the family stages (`mosaic.Family.persist`)."""
    blocks: dict
    staged: tuple
    outputs: tuple

    def scope(self, family: mosaic.Family, tiles: "Tiles", steps: int | None = None) -> resources.Scope:
        """Its allocations by lifetime (`resources.shared_bytes`): a ring family's streams the program's,
        a family whose walks scope their streams apart (`mosaic.Family.walks_scoped`) a walk's, in no more
        slots than a walk's `steps` where they are known (`emit_pipeline` allocates no more); each output
        staged as its family stages it (`mosaic.Family.output_staging`, the allocations `persist` makes)."""
        consumers = tiles.warpgroups
        staging = tuple(family.output_staging(self.outputs, consumers))
        if family.walks_scoped:
            slots = tiles.stages if steps is None else min(tiles.stages, steps)
            walk = tuple(plgpu.SMEM((slots, *shape), dtype) for shape, dtype in self.blocks.values())
            return resources.Scope(program=(self.staged, staging), walks=(walk,))
        multicast = ('matrix',) if tiles.cluster > 1 and 'matrix' in self.blocks else ()
        streams = family.streams(self.blocks, tiles.stages, consumers, multicast=multicast)
        return resources.Scope(program=(streams, self.staged, staging))


def _typed(dtype) -> jax.ShapeDtypeStruct:
    """A stand-in for an operand of `dtype`, as `mosaic.staging` reads an operand's dtype."""
    return jax.ShapeDtypeStruct((), dtype)


def gmm_allocations(tiles: "Tiles", family: mosaic.Family, *, dtype, rhs_dtype, transpose_rhs: bool,
                    prologue_axes: tuple, prologue_dtypes: tuple, epilogue_axes: tuple,
                    epilogue_dtypes: tuple, dtypes: tuple, paired: bool = False) -> Allocations:
    """Dew: `gmm`'s allocations at `tiles` (`Allocations`), for rows of `dtype` and a matrix of `rhs_dtype`
    (two with `paired`), its side operands' axes and dtypes and its outputs' dtypes: a stream for the rows,
    each matrix and each prologue side operand that follows the steps (`_stepped`), each epilogue side
    operand's staging at a consumer's share of the tile, each prologue side operand without depth's at the
    tile's rows."""
    block_m, block_n, block_k = tiles.block_m, tiles.block_n, tiles.block_k
    part = (block_m // tiles.warpgroups, block_n)
    stepped, _, fixed = _stepped(family, prologue_axes, 'depth')
    extent = {'rows': block_m, 'depth': block_k}
    rhs_block = (block_n, block_k) if transpose_rhs else (block_k, block_n)
    blocks = {'rows': ((block_m, block_k), dtype), 'matrix': (rhs_block, rhs_dtype),
              **({'paired': (rhs_block, rhs_dtype)} if paired else {}),
              **{f'side_{i}': (tuple(extent[axis] for axis in prologue_axes[i]), prologue_dtypes[i])
                 for i in stepped}}
    sides = mosaic.staging('side', [(tuple({'rows': part[0], 'columns': part[1]}[axis] for axis in axes),
                                     _typed(side))
                                    for axes, side in zip(epilogue_axes, epilogue_dtypes, strict=True)])
    held = mosaic.staging('fixed', [(tuple(extent[axis] for axis in prologue_axes[i]),
                                     _typed(prologue_dtypes[i])) for i in fixed])
    return Allocations(blocks, (sides['side_smems'], sides['side_barriers'], held['fixed_smems'],
                                held['fixed_barriers']), tuple((part, jnp.dtype(out)) for out in dtypes))


def tgmm_allocations(tiles: "Tiles", family: mosaic.Family, *, lhs_dtype, rhs_dtype,
                     prologue_axes: tuple, prologue_dtypes: tuple, out_dtype) -> Allocations:
    """Dew: `tgmm`'s allocations at `tiles` (`Allocations`): a ring for the rows, the cotangent and each
    prologue side operand with rows that follows the steps (`_stepped`), each side operand without rows'
    staging, the consumers' meeting, and its output's share of the tile."""
    block_m, block_n, block_k = tiles.block_m, tiles.block_n, tiles.block_k
    stepped, _, fixed = _stepped(family, prologue_axes, 'rows')
    extent = {'rows': block_k, 'depth': block_m}
    blocks = {'ring_0': ((block_k, block_m), lhs_dtype), 'ring_1': ((block_k, block_n), rhs_dtype),
              **{f'ring_{2 + position}': (tuple(extent[axis] for axis in prologue_axes[i]),
                                          prologue_dtypes[i]) for position, i in enumerate(stepped)}}
    held = mosaic.staging('fixed', [(tuple(extent[axis] for axis in prologue_axes[i]),
                                     _typed(prologue_dtypes[i])) for i in fixed])
    return Allocations(blocks, (held['fixed_smems'], held['fixed_barriers'],
                                mosaic.meeting(tiles.warpgroups)),
                       (((block_m // tiles.warpgroups, block_n), jnp.dtype(out_dtype)),))


def _stepped(family: mosaic.Family, axes_of: tuple, stepping: str) -> tuple[tuple, tuple, tuple]:
    """Dew: which side operands a launch streams with its steps (each with the `stepping` axis, but a vector a
    family does not copy, `mosaic.Family.copies_vectors`), which its step reads from global memory (such a
    vector), and which it loads once a tile (none with the `stepping` axis)."""
    stepped = tuple(i for i, axes in enumerate(axes_of)
                    if stepping in axes and (len(axes) > 1 or family.copies_vectors))
    read = tuple(i for i, axes in enumerate(axes_of) if stepping in axes and i not in stepped)
    fixed = tuple(i for i, axes in enumerate(axes_of) if stepping not in axes)
    return stepped, read, fixed


def gmm(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, *, tiles: Tiles, transpose_rhs: bool,
        out_dtype, multiprocessors: int, interpret: bool | plgpu.InterpretGPUParams = False,
        prologue=None, prologue_operands: tuple = (), prologue_axes: tuple = (), epilogue=None,
        epilogue_operands: tuple = (), epilogue_axes: tuple = (),
        out_dtypes: tuple | None = None, dense: bool = False,
        paired: jax.Array | None = None) -> jax.Array | tuple[jax.Array, ...]:
    """Each group's rows of `lhs` `[rows, depth]` times its matrix of `rhs`, `[groups, depth, width]` or
    with `transpose_rhs` `[groups, width, depth]`: `[rows, width]` in `out_dtype`, the rows past the
    routed ones zero. The depth must divide by `tiles.block_k` and the width by `tiles.block_n`.
    With `dense`, the one matrix multiplies every row without a grouped schedule. With `paired`, a second
    matrix of `rhs`'s shape, each tile multiplies its rows by both, one load of the rows, two
    accumulators, which the epilogue takes together (`epilogue(acc, paired_acc, *side_tiles)`: a gate and
    an up projection with their SwiGLU) or the outputs are, both (`out_dtypes`).

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
    if dense and g != 1:
        raise ValueError("a dense product takes one matrix")
    if paired is not None and paired.shape != rhs.shape:
        raise ValueError(f"a paired matrix {paired.shape} is not its product's {rhs.shape}")
    accumulators = 1 if paired is None else 2
    if k != k2 or k % block_k or n % block_n:
        raise ValueError(f"a [{rows}, {k}] x [{g}, {k2}, {n}] product does not take {tiles}")
    dtypes = tuple(jnp.dtype(dtype) for dtype in (out_dtypes or (out_dtype,)))
    if k == 0:
        # Dew: a product of no depth sums nothing, its rows zeros through the epilogue on whole arrays (its
        # hooks are a tile's elementwise arithmetic), with no launch: a walk of no steps has no block of the
        # matrix to view (JAX 0.12 refuses a zero-extent slice).
        zeros = (jnp.zeros((rows, n), jnp.float32),) * accumulators
        values = zeros if epilogue is None else epilogue(*zeros, *epilogue_operands)
        values = values if isinstance(values, tuple) else (values,)
        outs = tuple(value.astype(dtype) for value, dtype in zip(values, dtypes, strict=True))
        return outs if out_dtypes else outs[0]
    # Dew: a TMA descriptor tiles the rows by 8, so they are padded to a multiple of 8, and the rows past
    # the routed ones, the padding among them, are a last group, whose tiles store zeros. A group's tiles
    # start at its first row (`tile_groups`), so a tile reads up to a tile's rows past the last group's
    # first. Where a family's loads past an operand's rows read zeros (`Family.bounded_loads`: TMA's) the
    # operands stay as they are; elsewhere every row-bearing operand is allocated to the schedule's extent
    # (`row_extent`), zeros past its rows, which a group's stores never reach (a 128-row lhs of groups
    # [120, 8] at 64-row tiles read rows 120 to 183 unpredicated on an A100, cp.async's loads), as the
    # interpreter's are, whose slices past an array clamp rather than read zeros. On an H100 that copy was a
    # kernel of its own each call, 0.035 to 0.33 ms beside a dense product of 0.34 to 2.4 (BREAKDOWN1-h100:
    # 0.33 ms of Llama 3 8B gate_up's 2.69 ms input gradient). A dense call's tiles start at whole row
    # tiles, so it reads to its last tile's end alone, no copy where its rows fill whole tiles: on an A100
    # that copy was 0.094 ms of Llama 3 8B o's 1.52 ms forward (BC-A100-EXTENT-AB1, 6% of its time).
    lhs = rows_padded(lhs, 0, TMA_ROWS)
    m = lhs.shape[0]
    row_tile = tiles.block_m * tiles.cluster
    if dense:
        table = None
    else:
        lengths = jnp.concatenate([group_sizes.astype(jnp.int32),
                                   jnp.maximum(m - jnp.sum(group_sizes, dtype=jnp.int32), 0)[None]])
        table = tile_groups(lengths, row_tile, pl.cdiv(m, row_tile) + g)
    extent = (m if family_of(tiles, lhs.dtype).bounded_loads and not interpret
              else -(-m // row_tile) * row_tile if dense else row_extent(m, row_tile))
    lhs = _rows_extended(lhs, 0, extent)
    prologue_operands = tuple(_side_extended(side, axes, extent)
                              for side, axes in zip(prologue_operands, prologue_axes, strict=True))
    epilogue_operands = tuple(_side_extended(side, axes, extent)
                              for side, axes in zip(epilogue_operands, epilogue_axes, strict=True))
    outs = _gmm_mainloop(lhs, rhs, table, (m, n, k, g), tiles=tiles, transpose_rhs=transpose_rhs,
                         dtypes=dtypes, multiprocessors=multiprocessors, interpret=interpret,
                         hooks=(prologue, prologue_axes, epilogue, epilogue_axes),
                         sides=(prologue_operands, epilogue_operands), dense=dense, paired=paired)
    outs = tuple(out[:rows] for out in outs)
    return outs if out_dtypes else outs[0]


def _gmm_mainloop(lhs, rhs, table, shape: tuple, *, tiles: Tiles, transpose_rhs: bool, dtypes: tuple,
                  multiprocessors: int, interpret, hooks: tuple, sides: tuple, dense: bool = False,
                  paired: jax.Array | None = None) -> tuple:
    """Dew: `gmm` on its instruction family's bundle (`family_of`): `tiles.warpgroups` consumer roles, each
    its share of the rows by all the columns, one MMA a step (two: CUTLASS's cooperative split, 2-7% faster
    on every H100 scoreboard row than halving the columns), persistent over the tiles in a planar snake,
    the family owning the operands' streams across tiles, their staging and the outputs' stores
    (`mosaic.Family.persist`). Each consumer's prologue rewrites its share of the rows' slot, its epilogue
    takes its share of each side tile. `lhs` is padded; `table` is `tile_groups`' at the pair's rows (`gmm`).
    One multiprocessor a tile or, at `tiles.cluster` 2, a pair of them a cluster running two row tiles of
    one group, which load the group's matrix block once between them (`mosaic.Family.streams`' multicast);
    both run each pair's steps (or neither), its second's rows past the group stored nowhere. A `paired`
    matrix streams beside the first, its blocks multiplied by the same rows' into a second accumulator."""
    family = family_of(tiles, lhs.dtype)
    prologue, prologue_axes, epilogue, epilogue_axes = hooks
    prologue_operands, epilogue_operands = sides
    m, n, k, g = shape
    block_m, block_n, block_k, consumers = tiles.block_m, tiles.block_n, tiles.block_k, tiles.warpgroups
    pair = tiles.cluster
    share = block_m // consumers
    part = (share, block_n)
    # A stream for each operand that follows the steps, but a vector a family does not copy, which its
    # step reads from global memory; a prologue side operand without depth is loaded once a tile.
    stepped, read, fixed = _stepped(family, prologue_axes, 'depth')
    allocations = gmm_allocations(tiles, family, dtype=lhs.dtype, rhs_dtype=rhs.dtype,
                                  transpose_rhs=transpose_rhs, prologue_axes=prologue_axes,
                                  prologue_dtypes=tuple(side.dtype for side in prologue_operands),
                                  epilogue_axes=epilogue_axes,
                                  epilogue_dtypes=tuple(side.dtype for side in epilogue_operands),
                                  dtypes=dtypes, paired=paired is not None)
    matrices = ('matrix',) if paired is None else ('matrix', 'paired')
    steps = k // block_k
    # A pair's tiles: g groups' boundaries, and the unrouted rows.
    grid = (pl.cdiv(m, block_m * pair) + (0 if dense else g), n // block_n)

    def body(*gmems):
        table_gmem, gmems = (None, gmems) if dense else (gmems[0], gmems[1:])
        lhs_gmem, *matrix_gmems = gmems[:1 + len(matrices)]
        refs = gmems[1 + len(matrices):]
        prologue_gmems, side_gmems, o_gmems = _split(refs, len(prologue_operands), len(epilogue_operands))
        blocks = allocations.blocks

        def schedule(index):
            """A tile's whether it multiplies any row, its steps, its streams' views and its group, row
            block and column tile: its pair's but for this multiprocessor's row block."""
            mi, ni = plgpu.planar_snake(index[0], grid, 1, tiles.grid_block_n)
            group = (GroupInfo.dense(block_m * pair, mi, m) if dense
                     else GroupInfo.create(table_gmem, block_m * pair, mi))
            # This multiprocessor's row tile of its pair, from its group's first: a multiple of the rows a
            # TMA descriptor tiles by, as Mosaic's layout inference reads a slice's start.
            row_start = pl.multiple_of(group.block_start + mosaic.cluster_index(pair) * block_m, TMA_ROWS)
            views = _gmm_views(lhs_gmem, dict(zip(matrices, matrix_gmems, strict=True)), prologue_gmems,
                               group, row_start, ni, tiles=tiles, groups=g, transpose_rhs=transpose_rhs,
                               axes_of={i: prologue_axes[i] for i in stepped})
            active = jnp.ones((), jnp.bool_) if dense else (group.group_id < g) & (group.actual_size > 0)
            return active, steps, views, (group, row_start, ni)

        def scope(side_smems, side_barriers, fixed_smems, fixed_barriers, *, streams):
            def consumer(wg, tile, walk, store, *, active):
                group, row_start, ni = tile
                fixed_refs = [smem.at[wg] for smem in fixed_smems]
                if active:
                    family.loaded([(prologue_gmems[i].at[pl.ds(row_start, block_m)], ref, barrier.at[wg])
                                   for i, ref, barrier in zip(fixed, fixed_refs, fixed_barriers,
                                                              strict=True)])
                step = functools.partial(_gmm_step, family=family, prologue=prologue,
                                         prologue_axes=prologue_axes,
                                         fixed_tiles=dict(zip(fixed, fixed_refs, strict=True)),
                                         read_tiles={i: prologue_gmems[i] for i in read},
                                         mine_rows=pl.ds(wg * share, share), transpose_rhs=transpose_rhs,
                                         matrices=matrices, interpret=bool(interpret))
                # The unrouted rows, and a slot past the last tile, multiply nothing: zeros.
                _, accs = walk(step, None, (part,) * len(matrices))
                # This consumer's share of the tile: its first row, and its group's rows there.
                part_row, column = row_start + wg * share, ni * block_n
                if dense:
                    first = 0
                    stored = (part[0] if m % (block_m * pair) == 0
                              else jnp.clip(m - part_row, 0, part[0]))
                else:
                    first = jnp.clip(group.actual_start - part_row, 0, part[0])
                    stored = jnp.clip(group.actual_end - part_row, 0, part[0]) - first
                origins = {'rows': pl.ds(part_row, part[0]), 'columns': pl.ds(column, part[1])}
                family.loaded([(gmem.at[tuple(origins[axis] for axis in axes)], smem.at[wg], barrier.at[wg])
                               for gmem, axes, smem, barrier
                               in zip(side_gmems, epilogue_axes, side_smems, side_barriers, strict=True)])
                values = accs if epilogue is None else epilogue(*accs, *(smem[wg] for smem in side_smems))
                values = values if isinstance(values, tuple) else (values,)
                for output, (o_gmem, value) in enumerate(zip(o_gmems, values, strict=True)):
                    store(output, value, o_gmem, (part_row, first, stored, column, min(part[0], m),
                                                  TMA_ROWS if dense else 1))

            family.persist(streams, (grid[0] * grid[1],), schedule, consumer, consumers=consumers,
                           axis="programs", outputs=allocations.outputs,
                           interpret=bool(interpret), in_flight=_IN_FLIGHT)

        family.scoped(scope, *allocations.staged,
                      streams=family.streams(blocks, tiles.stages, consumers,
                                             multicast=matrices if pair > 1 else ()))

    out_type = tuple(jax.ShapeDtypeStruct((m, n), dtype) for dtype in dtypes)
    operands = (lhs, rhs) if paired is None else (lhs, rhs, paired)
    kernel = family.kernel(body, out_type=out_type,
                           grid=(multiprocessors // pair,), grid_names=("programs",), consumers=consumers,
                           interpret=interpret, approx_math=False, name="dew_gmm", cluster=pair,
                           cost=cost(2 * m * n * k * len(matrices), *operands, *prologue_operands,
                                      *epilogue_operands, *out_type))
    return kernel(*(operands if dense else (table, *operands)), *prologue_operands, *epilogue_operands)


def _gmm_views(lhs_gmem, matrix_gmems: dict, prologue_gmems, group, row_start, column_tile, *, tiles: Tiles,
               groups: int, transpose_rhs: bool, axes_of: dict) -> dict:
    """Dew: `gmm`'s streams' views at a tile (`group`'s rows from `row_start`, `column_tile`;
    `mosaic.viewed`): its rows', its group's matrix's (each of `matrix_gmems`, by stream: a paired
    product's two), and each prologue side operand's that follows the depth (`axes_of`, by operand), step
    `j` each one's `j`-th depth block."""
    rows = pl.ds(row_start, tiles.block_m)
    matrices = {name: (_matrix_columns(gmem.at[jnp.minimum(group.group_id, groups - 1)], column_tile,
                                       block_n=tiles.block_n, transpose_rhs=transpose_rhs),
                       _rhs_index(transpose_rhs)) for name, gmem in matrix_gmems.items()}
    return {'rows': (lhs_gmem.at[rows], lambda j: (0, j)), **matrices,
            **{f'side_{i}': (prologue_gmems[i].at[tuple(rows if axis == 'rows' else slice(None)
                                                        for axis in axes)],
                             lambda j, axes=axes: tuple(0 if axis == 'rows' else j for axis in axes))
               for i, axes in axes_of.items()}}


def _gmm_step(index, slots, nothing, accumulators, *, family: mosaic.Family, prologue, prologue_axes: tuple,
              fixed_tiles: dict, read_tiles: dict, mine_rows, transpose_rhs: bool, matrices: tuple,
              interpret: bool):
    """Dew: a consumer's step of the cooperative `gmm` (`mosaic.Family.walk`): its half of the rows' slot,
    rewritten by the prologue on its side tiles (the streamed ones' slots, the fixed ones' `fixed_tiles`,
    a depth vector the family does not copy read at step `index` from global memory: `read_tiles`),
    times each of `matrices`' slot into its accumulator (a paired product's two)."""
    stage = slots['rows']
    if prologue is not None:
        # This warpgroup's half of the rows, the only half its MMA reads.
        depth = stage.shape[1]
        tiles_of = {i: fixed_tiles[i] if i in fixed_tiles
                    else read_tiles[i].at[pl.ds(index * depth, depth)] if i in read_tiles
                    else slots[f'side_{i}'] for i in range(len(prologue_axes))}
        _rewritten(prologue, prologue_axes, stage.at[mine_rows], tiles_of, 'rows', mine_rows)
        family.published()
    oriented = {name: slots[name].transpose((1, 0)) if transpose_rhs else slots[name] for name in matrices}
    return nothing, tuple(family.products(acc, stage.at[mine_rows], oriented[name], interpret)
                          for acc, name in zip(accumulators, matrices, strict=True))


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
         prologue_operands: tuple = (), prologue_axes: tuple = (), exact_sums: bool = False,
         dense: bool = False) -> jax.Array:
    """Each group's rows of `lhs` `[rows, depth]` transposed times its rows of `rhs` `[rows, width]`:
    `[groups, depth, width]` in `out_dtype`, an empty group's zero. The depth must divide by
    `tiles.block_m` and the width by `tiles.block_n`, and the tile takes one multiprocessor;
    `tiles.block_k` is the rows a step takes.

    Dew: `prologue(rows, *side_tiles)` transforms each `[block_k, block_m]` stage of `lhs`'s rows (a
    cooperative tile's consumer, half its depth) in place before the group masking and the MMA; side
    operands follow 'rows' and 'depth' as `gmm`'s.
    `exact_sums` adds each step's product into the fp32 sum by an IEEE add, which holds two
    accumulators in registers. A `dense` call (one group, every row) on a family that fixes up a tile
    across programs (`balanced`) runs its output tiles' last wave balanced over the programs."""
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
    prologue_operands = tuple(_side_padded(side, axes, multiple)
                              for side, axes in zip(prologue_operands, prologue_axes, strict=True))

    group_sizes = group_sizes.astype(jnp.int32)
    out_dtype = jnp.dtype(out_dtype)
    if tiles.splits > 1:
        # Each group's rows in `splits` runs of whole steps from the group's first block, the first and last
        # clipped to its rows: each run a group of its own, its fp32 partial summed after.
        sliced = _sliced(group_sizes, block_k, tiles.splits)
        partials = tgmm(lhs, rhs, sliced, tiles=dataclasses.replace(tiles, splits=1), out_dtype=jnp.float32,
                        multiprocessors=multiprocessors, interpret=interpret, prologue=prologue,
                        prologue_operands=prologue_operands, prologue_axes=prologue_axes)
        return partials.reshape(g, tiles.splits, m, n).sum(axis=1).astype(out_dtype)
    group_starts, group_ends, group_num_blocks, group_block_starts = _group_blocks(group_sizes, block_k)
    tail = None
    # An exact sum's order is its contract, which a tile summed across programs reassociates.
    if dense and family_of(tiles, lhs.dtype).fixes_up and not exact_sums:
        tail = tail_of.balanced(m // block_m * (n // block_n), -(-k // block_k), multiprocessors)
    return _tgmm_streamed(
        (lhs, rhs), (group_sizes, group_starts, group_ends, group_num_blocks, group_block_starts),
        (k, m, n, g), tiles=tiles, out_dtype=out_dtype, multiprocessors=multiprocessors,
        interpret=interpret, prologue=prologue, prologue_operands=prologue_operands,
        prologue_axes=prologue_axes, exact_sums=exact_sums, tail=tail)


def _tgmm_streamed(operands: tuple, groups: tuple, shape: tuple, *, tiles: Tiles, out_dtype,
                      multiprocessors: int, interpret, prologue, prologue_operands: tuple,
                      prologue_axes: tuple, exact_sums: bool,
                      tail: tail_of.Tail | None = None) -> jax.Array:
    """Dew: `tgmm`'s product at a tile `tiles.warpgroups` consumer roles compute, each its share of the depth
    by all the columns, one MMA a step (two: CUTLASS's cooperative sm90 schedule), on its instruction
    family's bundle (`family_of`), its streams running across the persistent loop's tiles where the family
    streams so (the next tile's first row blocks load while the consumers store this one, which a group's
    few rows, a tile's few steps, make most of the time). Each consumer reads its depth share of the rows'
    stage, which it prepares (the prologue, the neighbouring groups' rows zeroed), and the whole
    cotangent's, a column share of which each zeroes, waiting for the other's where a group boundary falls
    in the step. A side operand with rows streams beside the operands; one without is loaded once a tile.
    One multiprocessor a tile, but where a dense call's last wave is balanced (`tail`): a tile then walked
    by several programs in turn, each but its first program's leaving its fp32 partial in a workspace and
    signalling a semaphore, which the first waits on and adds before storing the tile. Interpreted, one
    program walks every program's tiles in turn (the interpreter runs its programs one after another, each
    paying its own setup), a balanced tail's programs ordered so no semaphore is needed (`tail.ordered`).
    `operands` are padded and `groups` is `_group_blocks`' with the sizes first (`tgmm`)."""
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
    # A stream for each side operand with rows, but a vector a family does not copy, which its step reads
    # from global memory.
    stepped, read, fixed = _stepped(family, prologue_axes, 'rows')
    allocations = tgmm_allocations(tiles, family, lhs_dtype=lhs.dtype, rhs_dtype=rhs.dtype,
                                   prologue_axes=prologue_axes,
                                   prologue_dtypes=tuple(side.dtype for side in prologue_operands),
                                   out_dtype=out_dtype)

    def body(sizes_gmem, starts_gmem, ends_gmem, blocks_gmem, block_starts_gmem, lhs_gmem, rhs_gmem, *refs):
        *side_gmems, o_gmem = refs if tail is None else refs[:-1]
        ready = None if tail is None else family.partials_ready((multiprocessors, consumers), bool(interpret))

        schedule = functools.partial(
            _tgmm_scheduled, groups=(sizes_gmem, blocks_gmem, block_starts_gmem),
            operands=(lhs_gmem, rhs_gmem, side_gmems), tail=tail, programs=multiprocessors, grid=grid,
            tiles=tiles, stepped=stepped, prologue_axes=prologue_axes, interpreted=bool(interpret))

        def scope(fixed_smems, fixed_barriers, halves, *, streams):
            names = tuple(allocations.blocks)

            def consumer(wg, tile, walk, store, *, active):
                g_i, depth, columns, steps, start, gathered, program = tile
                if tail is not None and not active:
                    return  # past its program's range: another program walks the tile
                mine_depth = pl.ds(wg * depth_half, depth_half)
                mine_columns = (wg * atoms_each, atoms_each, atoms_each * consumers > block_n // atom)
                fixed_refs = [smem.at[wg] for smem in fixed_smems]
                group = (starts_gmem[g_i], ends_gmem[g_i], steps, block_k)
                first = pl.multiple_of(block_starts_gmem[g_i], block_k) // block_k
                if active:
                    family.loaded([(side_gmems[i].at[tuple(depth for _ in prologue_axes[i])], ref,
                                    barrier.at[wg])
                                   for i, ref, barrier in zip(fixed, fixed_refs, fixed_barriers,
                                                              strict=True)])

                def staged_at(index, slots):
                    # The step of its group's the program walks, from its first (`start`).
                    index = index + start
                    # A vector read from global memory at the step's row block (`read`).
                    read_refs = {i: side_gmems[i].at[pl.ds((first + index) * block_k, block_k)] for i in read}
                    hook = (None if prologue is None
                            else (prologue, prologue_axes, stepped, fixed, fixed_refs, read_refs))
                    return _staged_halves(family, [slots[name] for name in names], (index, *group),
                                          (mine_depth, mine_columns), hook,
                                          halves if consumers == 2 else None)

                # An empty group's tile reads none and stores zeros.
                summed = _walked_gradient(family, walk, staged_at, part, exact_sums, bool(interpret))
                place = (depth.start + wg * depth_half, 0, depth_half, columns.start, depth_half)
                if tail is None:
                    store(0, summed, o_gmem.at[g_i], place)
                    return
                family.fixed_up(summed, refs[-1].at[:, wg], None if ready is None else ready.at[:, wg],
                                program, start, gathered,
                          lambda value: store(0, value, o_gmem.at[g_i], place))

            walks = (g, grid[0] * grid[1]) if tail is None else (tail.segments, multiprocessors)
            family.persist(streams, walks, schedule, consumer, consumers=consumers,
                           axis="programs", outputs=allocations.outputs, interpret=bool(interpret),
                           in_flight=_IN_FLIGHT)

        family.scoped(scope, *allocations.staged,
                      streams=family.streams(allocations.blocks, tiles.stages, consumers))

    out_type = jax.ShapeDtypeStruct((g, m, n), out_dtype)
    # A balanced tail's partials: each program's, a consumer's share of a tile each.
    workspace = () if tail is None else (
        jax.ShapeDtypeStruct((multiprocessors, consumers, *part), jnp.float32),)
    kernel = family.kernel(body, out_type=(out_type, *workspace),
                           grid=(1 if interpret else multiprocessors,), grid_names=("programs",),
                           consumers=consumers,
                           interpret=interpret, approx_math=False, name="dew_tgmm",
                           cost=cost(2 * lhs.shape[0] * m * n, lhs, rhs, *prologue_operands, out_type))
    return kernel(*groups, lhs, rhs, *prologue_operands)[0]


def _tgmm_scheduled(index, *, groups: tuple, operands: tuple, tail: tail_of.Tail | None, programs: int,
                    grid: tuple, tiles: Tiles, stepped: tuple, prologue_axes: tuple,
                    interpreted: bool) -> tuple:
    """Dew: a streamed kernel gradient's tile at the persistent walk's `index` (`mosaic.Family.persist`'s
    schedule): whether its program walks any of its rows, how many steps, its streams' views (step `j` its
    group's `j`-th row block from the program's first, at the tile's depth and columns), and its group,
    depth and columns, its group's steps, the program's first of them, its partials' gatherer (`walked`)
    and its program. `groups` are the group sizes', block counts' and first blocks' references and
    `operands` the rows', cotangent's and side operands'."""
    sizes_gmem, blocks_gmem, block_starts_gmem = groups
    lhs_gmem, rhs_gmem, side_gmems = operands
    index = tail_of.ordered(tail, index, programs, interpreted)
    g_i, snaked, start, count, gathered = _segmented(tail, index, programs, grid[0] * grid[1])
    m_i, n_i = plgpu.planar_snake(snaked, grid, 1, tiles.grid_block_n)
    depth, columns = pl.ds(m_i * tiles.block_m, tiles.block_m), pl.ds(n_i * tiles.block_n, tiles.block_n)
    first = pl.multiple_of(block_starts_gmem[g_i], tiles.block_k) // tiles.block_k + start
    steps = jnp.where(sizes_gmem[g_i] > 0, blocks_gmem[g_i], 0)
    walks = steps if count is None else count

    def row_block(j):
        return first + j, 0

    views = {'ring_0': (lhs_gmem.at[:, depth], row_block), 'ring_1': (rhs_gmem.at[:, columns], row_block),
             **{f'ring_{2 + position}': (
                 side_gmems[i].at[tuple(slice(None) if axis == 'rows' else depth
                                        for axis in prologue_axes[i])],
                 lambda j, axes=prologue_axes[i]: tuple(first + j if axis == 'rows' else 0 for axis in axes))
                for position, i in enumerate(stepped)}}
    return walks > 0, walks, views, (g_i, depth, columns, steps, start, gathered, index[-1])


def _segmented(tail: tail_of.Tail | None, index, programs: int, tiles: int) -> tuple:
    """Dew: a streamed kernel gradient's walk at `index`: its group, its tile's place in the snake, the
    first of the tile's steps the program walks and how many (None: its group's), and its partials'
    gatherer's end (`Tail.segment`): a tile a program a wave where nothing balances."""
    if tail is None:
        return index[0], index[1], 0, None, None
    return 0, *tail.segment(index, programs, tiles)


def _staged_halves(family: mosaic.Family, stages: list, group: tuple, mine: tuple, hook, halves) -> tuple:
    """Dew: a streamed kernel gradient step's operands for one consumer warpgroup, from its `stages` (the
    rows', the cotangent's, then each side operand's with rows): its depth half of the rows, rewritten by
    the prologue (`hook`: it, its operands' axes, which are stepped and which fixed, the fixed ones' tiles
    and the vectors read from global memory, by operand), and the whole cotangent, the neighbouring
    groups' rows zeroed in its shares of both (`group`: the step, the group's first and last row, its
    steps, the rows a step takes; `mine`: its depth half, and its first swizzled row of the cotangent's,
    how many, and whether the last consumer's share runs short of them), the other's share awaited at
    `halves`
    where it zeroed any. Both consumers' MMAs read the whole cotangent stage, so each zeroes its share of
    it and they meet where a group boundary falls in the step, the kernel's one exchange between them."""
    rows_stage, cotangent_stage, *side_stages = stages
    mine_depth, (first_atom, atoms_each, short) = mine
    if hook is not None:
        prologue, axes_of, stepped, fixed, fixed_refs, read_refs = hook
        tiles_of = (dict(zip(stepped, side_stages, strict=True)) | dict(zip(fixed, fixed_refs, strict=True))
                    | read_refs)
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


def split_count(tiles: Tiles, *, groups: int, rows: int, depth: int, width: int, multiprocessors: int,
                step_seconds: float | None, bandwidth: float | None) -> int:
    """Dew: the slices a kernel gradient's contraction splits into at `tiles` (`Tiles.splits`, `tgmm`),
    from the call's extents and the device's facts. Over a wave (`multiprocessors` x `tiles.programs`)
    and `rows` in steps a group (on average), the busiest program walks ceil(tiles x s / wave) output
    tiles of ceil(steps / s) steps at s slices. The split is the least s whose walk, with the time its
    fp32 partials take written and read back (2s + 1 passes over the output at `bandwidth`, in steps of
    `step_seconds`), is shorter than the unsplit walk, each slice keeping at least the ring's stages and
    one more; one where none is, or where the device states no rate. A bounded comparison over the
    facts, no table, no timing: Key's 2560 x 1024 gradient over 24,576 rows on an H100 splits in three,
    2 x 128 steps against 384."""
    if step_seconds is None or bandwidth is None:
        return 1
    wave = multiprocessors * tiles.programs
    output = groups * (depth // tiles.block_m) * (width // tiles.block_n)
    steps = -(-rows // max(groups, 1) // tiles.block_k)

    def walk(splits: int) -> float:
        passes = 0 if splits == 1 else 2 * splits + 1
        partials = passes * groups * depth * width * 4 / bandwidth / step_seconds
        return -(-output * splits // wave) * -(-steps // splits) + partials

    return next((splits for splits in range(2, steps // (tiles.stages + 1) + 1) if walk(splits) < walk(1)), 1)


def _sliced(group_sizes: jax.Array, block_k: int, splits: int) -> jax.Array:
    """Dew: each group's rows as `splits` consecutive runs (group-major, a run's size each): the group's
    `block_k`-row blocks, from its first row's block, shared out in whole blocks as evenly as they divide, the
    first run starting at the group's first row and the last ending at its last; a run of a group with
    fewer blocks than runs is empty."""
    starts, ends, blocks, block_starts = _group_blocks(group_sizes, block_k)
    shares = jnp.arange(1, splits)[None, :] * blocks[:, None] // splits
    inner = jnp.clip(block_starts[:, None] + shares * block_k, starts[:, None], ends[:, None])
    bounds = jnp.concatenate([starts[:, None], inner, ends[:, None]], axis=1)
    return jnp.diff(bounds, axis=1).reshape(-1).astype(jnp.int32)


def _group_blocks(group_sizes: jax.Array, block_k: int) -> tuple:
    """Each group's first row, end, count of `block_k`-row blocks, and its first block's first row."""
    group_starts = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(group_sizes)[:-1]]).astype(jnp.int32)
    group_ends = jnp.cumsum(group_sizes).astype(jnp.int32)
    group_block_starts = group_starts // block_k * block_k
    group_block_ends = -(group_ends // -block_k) * block_k
    return group_starts, group_ends, (group_block_ends - group_block_starts) // block_k, group_block_starts


def row_extent(rows: int, row_tile: int) -> int:
    """Dew: the rows `gmm`'s tile schedule reads of `rows` rows (a multiple of `TMA_ROWS`) at tiles of
    `row_tile` rows: a tile starts at most `TMA_ROWS` before the rows end (`tile_groups`) and reads
    `row_tile` rows on."""
    return rows + row_tile - TMA_ROWS


def _rows_extended(array: jax.Array, axis: int, extent: int) -> jax.Array:
    """Dew: `array` with zeros after its own along `axis` up to `extent`."""
    pad = extent - array.shape[axis]
    widths = tuple((0, pad if dim == axis else 0) for dim in range(array.ndim))
    return jnp.pad(array, widths) if pad > 0 else array


def _side_extended(side: jax.Array, axes: tuple, extent: int) -> jax.Array:
    """Dew: a side operand allocated as the rows are along the dimension that follows them (`row_extent`)."""
    return _rows_extended(side, axes.index('rows'), extent) if 'rows' in axes else side


def _side_padded(side: jax.Array, axes: tuple, multiple: int = 8) -> jax.Array:
    """A side operand padded as the rows are along the dimension that follows them, wherever it is: to a
    multiple of 8 (a TMA descriptor's tiling), or of `multiple` (a family's unpredicated loads)."""
    return rows_padded(side, axes.index('rows'), multiple) if 'rows' in axes else side
