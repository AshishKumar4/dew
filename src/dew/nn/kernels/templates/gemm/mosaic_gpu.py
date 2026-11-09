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
from typing import Any

import jax
from jax import lax, numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.mosaic.gpu import utils as mgpu_utils
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.algebra import bits
from dew.nn.kernels import generation, tuning
from dew.nn.kernels.templates import (
    axes,
    bodies,
    bundles,
    cost as unit_cost,
    mosaic,
    resources,
    swizzling,
    tail as tail_of,
)
from dew.nn.kernels.templates.gemm import rows_extended, rows_padded, scattered, sides_extended, sides_padded
from dew.nn.kernels.templates.product import SCALE_DEPTH


def family_of(tiles: "Tiles", dtype) -> mosaic.Family:
    """The bundle a tile runs on: the one it names (`Tiles.family`, `bundles.named`), for a consumer's share
    of the tile's rows and the columns its accumulators hold (`tile_columns`, `mosaic.Issuer.specialized`)."""
    family = bundles.named(tiles.family, dtype)
    return family.specialized(tiles.block_m // tiles.warpgroups, tile_columns(tiles, family))


def tile_columns(tiles: "Tiles", family: mosaic.Family) -> int:
    """The output columns a tile's accumulators hold: its `block_n`, or both CTAs' of a pair whose family's
    products span it (`mosaic.Transport.pairs_product`, tcgen05's pair: its tile twice a CTA's rows and
    columns, each CTA's accumulators its own rows by all the pair's columns)."""
    return tiles.block_n * (tiles.cluster if family.pairs_product else 1)


@dataclasses.dataclass(frozen=True)
class Tiles:
    """One launch's output tile (`block_m` x `block_n`), contraction step (`block_k`), the steps in flight
    (`stages`, upstream's `max_concurrent_steps`) and the column tiles in each band of the planar snake
    (`grid_block_n`); Dew: the warpgroups computing a tile (`warpgroups`: 2 halve `block_m` between two
    consumer warpgroups beside a producer that issues the loads, CUTLASS's cooperative sm90 schedule), and
    the multiprocessors of a cluster (`cluster`: 2 pair two row tiles of one group of `gmm`, which load
    their matrix block once between them, multicast), only for a cooperative tile; `tgmm` takes no pair;
    and the bundle that runs it (`family`, `bundles.named`, which each tile names: Hopper's warpgroup MMA,
    its loads a producer warpgroup's or, 'wgmma.fed', its first consumer's, or Ampere's mma.sync, one
    warpgroup whose loads `emit_pipeline` issues, the plain product alone); and the programs
    a multiprocessor runs at once (`programs`, at most its family's, `mosaic.Family.programs`), each in its
    share of the multiprocessor's shared memory (`resources.program_shared`); and the slices a kernel
    gradient's contraction splits into (`splits`, `tgmm`: each group's rows in that many runs of whole steps,
    each run's fp32 partial summed after), where its output tiles underfill a wave (`split_count`); and
    whether a dense product runs swapped (`swapped`, CUTLASS's and cuBLAS's swap of A and B): its matrix the
    MMA's rows and the call's rows its columns, `block_m` then the matrix's and `block_n` the call's rows,
    where those are fewer than a role's (`grouped_matmul.MosaicGmm.derived`)."""
    block_m: int
    block_n: int
    block_k: int
    stages: int
    grid_block_n: int
    warpgroups: int = 1
    cluster: int = 1
    family: str = dataclasses.field(kw_only=True)
    programs: int = 1
    splits: int = 1
    swapped: bool = False

    def __post_init__(self):
        family = bundles.named(self.family, jnp.bfloat16)
        # A pair of multiprocessors shares a group's matrix block between two consumers' row tiles, or, on a
        # family whose products span the pair (`mosaic.Transport.pairs_product`, tcgen05's), one product.
        paired = self.cluster == 2 and self.warpgroups == 1 and not family.pairs_product
        if self.warpgroups not in (1, 2) or self.cluster not in (1, 2) or paired:
            raise ValueError(f"{self}: one warpgroup alone, or two in a cluster of one or two "
                             "multiprocessors")
        if self.cluster > family.cluster:
            raise ValueError(f"{self}: a cluster of at most its bundle's (`mosaic.Family.cluster`)")
        wide = family.widest_product is not None and self.block_n > family.widest_product
        if self.warpgroups == 2 and (self.block_m % (2 * family.role_rows) or wide):
            # Each consumer's half of the rows is a role's, and its columns at most one product's (warpgroup
            # MMA's 256).
            raise ValueError(f"{self}: two warpgroups halve the tile's rows into whole roles")
        if self.warpgroups == 2 and self.stages < 2:
            # A stage is released a step late (an MMA in flight), so one stage would wait on itself.
            raise ValueError(f"{self}: two warpgroups stream at least two stages")


CONSUMERS = 2
"""The most consumer warpgroups a GEMM tile takes: CUTLASS's sm90 cooperative kernel's two; a family's third
(wgmma's, FlashAttention 3's d64 forward's) unmeasured on a product."""


def derived(layouts: tuple, *, rows: int, depth: int | None, width: int, dtype, multiprocessors: int, target,
            candidate, launch_of, grouped: bool, depths: tuple[int, ...] | None = None,
            ragged: bool = False, units: tuple[int, int, int] = (1, 1, 1)) -> Tiles | None:
    """The one tile a GEMM mainloop runs on an instruction family's `layouts` (`bundles.layouts`, its
    bundles in preference order, which share its facts but their loaders and registers) for a call whose
    output has `rows` rows (a group's,
    on average, where `grouped`: a product's rows split among its groups) and `width` columns and whose
    steps run over `depth` (None: a group's rows, padded to whole steps), from the family's facts, the
    call's extents and the device's multiprocessors alone (no table, no ranking;
    docs/design/kernel-templates.md §7); `depths`, where given, the steps the launch's stage takes in place
    of the family's (a packed matrix's decode, `NVFP4_STEP`); `ragged`, rows a tile need not divide (a kernel
    gradient's whose last row tile is partial, `tgmm`); `units`, what the rows, the columns and the step are
    whole multiples of besides (a call's side vectors' copy units, `mosaic.Transport.vector_unit`).
    `candidate(tile)` is the launch's legality apart
    from its
    resources (its granularity, its blocks dividing the call, its side operands' copies), and
    `launch_of(tile)` the launch whose ring and residency the planner sizes (`resources.plan`: its
    allocations a stage, its registers within a consumer's) on `target`.

    - Layout: at each shape and step, the first layout whose planner holds it (`resources.placed`): a
      producer's where its consumers' registers hold the tile, else one whose first consumer issues the
      loads, its consumers holding the program's entry registers (`fed.ConsumerFed`).
    - Shape: the balanced tile within its consumers' registers: an accumulator of `resources.held_area`
      elements A reads its operands 1/rows + 1/columns times a FLOP, least at rows = columns = sqrt(A),
      which whole roles and the family's widths come near rather than reach. So the rows are sqrt(A) in
      whole roles (no more than the call's, below;
      `_balanced_side`); the columns the widest dividing the call's width in its column unit, at most the
      family's one product's (`Family.widest_product`), that A holds at those rows; the rows then the most
      its registers hold at those columns (`resources.held_rows`). mma.sync's one consumer so takes 128 x
      128 where it took 64 x 256, whose four warps, splitting its rows, each read the columns' block whole:
      1.7x the shared memory a FLOP (SHAPE1-a100: 0.887 of its time). A tile its registers or a block
      cannot stage busy at its balanced rows (`holds`, `resources.tile_registers`) gives way to the next
      narrower one, and where none does, to shorter rows.
    - Consumers: the most the family takes (`Family.consumers`, at most `CONSUMERS`), a warpgroup each,
      whose cooperative tile holds each consumer's registers, so its balanced side and its reads a FLOP are
      at least one consumer's; one where rows a tile must divide (not grouped) take no whole role for each,
      where a group averages fewer rows than one role (grouped), or where no cooperative tile holds.
    - Programs: the most a multiprocessor runs at once that the family's registers hold at those
      consumers (`Family.programs`) and whose share of its shared memory (`resources.program_shared`) holds
      a tile, one at least; a wave is all of them.
    - Rows: no more than the call's rows rounded up to a role's or, where grouped, half a group's in
      whole roles, at least one role (each group takes a partial tile more, so at half its rows that tile
      adds at most a third of its work).
    - Step: the family's widest depth dividing the call's (`Family.depths`) at those rows, a shallower
      one before fewer rows (a family whose products read registers holds a step's fragments).
    - Stages: the deepest its program's share of shared memory holds at those programs (`resources.plan`,
      the planner every template's derivation reads), at least those the walk keeps busy (those loading
      ahead, `Family.loads_ahead`, beside the one consumed and those whose products the family keeps in
      flight, `Family.in_flight`); a tile that cannot stage that deep gives way to the next
      narrower or shorter one, or a shallower step.
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
    columns = _derived_columns(family, width, dtype, units[1])
    steps = [step for step in depths or family.depths(dtype)
             if (depth is None or depth % step == 0) and step % units[2] == 0]
    busy = 1 + family.loads_ahead + family.in_flight

    def tiled(consumers: int, programs: int) -> Tiles | None:
        role = math.lcm(family.role_rows * consumers, units[0])
        most_rows = max(role, rows // 2 // role * role if grouped else -(-rows // role) * role)
        shaped = functools.partial(_shaped, consumers=consumers, programs=programs, rows=rows, width=width,
                                   multiprocessors=multiprocessors, grouped=grouped)

        held_tile = functools.partial(_held_tile, role=role, steps=steps, busy=busy, shaped=shaped,
                                      candidate=candidate, launch_of=launch_of, layouts=layouts,
                                      target=target, programs=programs)
        # The balanced tile: the widest width the area holds at the balanced rows (`_held_widths`), its
        # rows the most its registers hold there and its stages keep busy, rows short of that balance (the
        # width's held rows, or the balanced side where it holds more) giving way to the next narrower
        # width, so a tile keeps its rows and narrows rather than shortening. mma.sync's one consumer at
        # Qwen3-8B's gate/up so takes 128 x 128, its 192-wide tile's rows shortened to 64 by its stages
        # (64 x 192 ran 1.21-1.30x XLA's time against 128 x 128's 1.10-1.14, FSA100POST32). Where no width
        # stages busy at its balance, the widest staging busy at all, then the widest that holds.
        widths = _held_widths(layouts, consumers, columns, role, most_rows)
        side = _balanced_side(layouts, consumers, role, most_rows)
        busy_at, fallback = None, None
        for block_n, held_rows in widths:
            tile, staged = held_tile(block_n, held_rows)
            if tile is None:
                continue
            if staged and tile.block_m >= min(held_rows, side):
                return tile
            busy_at = busy_at or (tile if staged else None)
            fallback = fallback or tile
        return busy_at or fallback

    # Rows a tile divides (a kernel gradient's) that take no whole role for each consumer take one, as
    # does a call no cooperative tile holds (a wide epilogue's outputs past a block), and a grouped call
    # whose groups average fewer rows than one role (a decode step's experts, a few routed rows each), whose
    # further consumers would compute padding alone; where groups average a role's rows, a routing's groups
    # past it keep the further consumers' rows real. Row 3622, H100 (FS-3622-AB1/AB2-h100): one consumer ran
    # 0.73 to 0.999 of the cooperative tile's time at 1 to 32 rows a group, and 1.13 to 1.26 of it at 64.
    most = min(family.consumers, CONSUMERS)
    whole = rows >= family.role_rows if grouped else ragged or rows % (family.role_rows * most) == 0
    # The most consumers the call takes, a cooperative tile holding each consumer's registers so its
    # balanced side, and with it each operand's reads a FLOP, at least one consumer's (CUTLASS's split, 2-7%
    # faster on every H100 scoreboard row); where no tile of theirs holds, one. Each at the most programs
    # that hold its tile.
    for consumers in dict.fromkeys((most if whole else 1, 1)):
        tile = next((tile for programs in range(family.programs(consumers), 0, -1)
                     if (tile := tiled(consumers, programs)) is not None), None)
        if tile is not None:
            return tile
    return None


def _derived_columns(family: mosaic.Family, width: int, dtype, unit_of_call: int) -> list[int]:
    """Dew: `derived`'s widths, widest first: whole rows of the widest swizzle, as well as its column unit,
    where the call's width has one (the objective prices reads, not a narrower staging swizzle's, and an
    A100's 160-wide tiles, 64-byte swizzle, ran its forwards 1.13-1.24x their 128-wide time where the reads
    favoured them, M-FS-1, W3-a100); a narrower call (a swapped decode product's few rows) at its column
    unit; each a whole number of the call's own unit (`unit_of_call`, its side vectors')."""
    widest = min(width, family.widest_product or width)
    unit = family.column_unit(dtype)
    swizzled = math.lcm(unit, swizzling.SWIZZLE_BYTES // jnp.dtype(dtype).itemsize)
    columns = ([block_n for block_n in range(widest - widest % swizzled, 0, -swizzled)
                if width % block_n == 0]
               or [block_n for block_n in range(widest - widest % unit, 0, -unit) if width % block_n == 0])
    return [block_n for block_n in columns if block_n % unit_of_call == 0]


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
            if not candidate(tile):
                # A tile a compile of this process found spilling (`report.refused`) gives way to the next.
                continue
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
    side = _balanced_side(layouts, consumers, role, most_rows)
    widest = next((block_n for block_n in columns if block_n * side <= area), columns[-1])

    def held(block_n: int) -> int:
        return min(most_rows, max(resources.held_rows(layout, consumers, block_n) for layout in layouts))

    return [(block_n, held(block_n)) for block_n in columns if block_n <= widest and held(block_n) >= role]


def _balanced_side(layouts: tuple, consumers: int, role: int, most_rows: int) -> int:
    """`derived`'s balanced rows at `consumers`: the square root of the accumulator area their registers
    hold (`resources.held_area`, `layouts`' most generous), which reads each operand fewest times a FLOP
    (1/rows + 1/columns at rows x columns = area), in whole roles, at least one, no more than the call's."""
    area = max(resources.held_area(layout, consumers) for layout in layouts)
    return max(role, min(most_rows, math.isqrt(area) // role * role))


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
    matrix: jax.Array | None = None
    """Dew: the matrix of a stack the group reads, where a table's fifth column states it (`gmm`'s
    `first`); None where the group reads its own."""

    @classmethod
    def create(cls, table, tile, tid, unit: int = 8):
        """Dew: the group info of row tile `tid`, read from `tile_groups`' table (its group, its first row,
        its group's first and past-last row, and where `gmm` states one, its matrix of a stack) rather than
        walked over every group's length in each tile."""
        tile = jnp.int32(tile)
        group, block_start, group_start, group_end = (table[tid, column] for column in range(4))
        # Mosaic's copies and stores read a slice's start as a multiple of the rows its tiling takes
        # (`row_unit`).
        block_start = pl.multiple_of(block_start, unit)
        actual_start = jnp.maximum(group_start, block_start)
        actual_end = jnp.minimum(group_end, block_start + tile)
        start_within_block = actual_start - block_start
        actual_size = actual_end - actual_start
        return cls(group_id=group, block_start=block_start, actual_start=actual_start,
                   actual_end=actual_end, start_within_block=start_within_block, actual_size=actual_size,
                   matrix=table[tid, 4] if table.shape[1] > 4 else None)

    @classmethod
    def dense(cls, tile: int, tid, rows: int):
        """A dense row tile's bounds, with a final tile clipped to the padded operand's rows."""
        start = pl.multiple_of(tid * tile, TMA_ROWS)
        size = jnp.minimum(tile, rows - start)
        return cls(group_id=jnp.int32(0), block_start=start, actual_start=start,
                   actual_end=start + size, start_within_block=jnp.int32(0), actual_size=size)


def tile_groups(lengths: jax.Array, tile: int, tiles: int, unit: int = 8) -> jax.Array:
    """Dew: each row tile's group info for `GroupInfo.create`, in XLA before the launch: of tiles `0` to
    `tiles`, the group (`lengths`' index), the tile's first row, and its group's first and past-last row,
    int32 `[tiles, 4]`; a tile no group reaches, zeros. A group's tiles start at its first row (rounded
    down to the `unit` rows its family's copies start at, `row_unit`), whole tiles after it, as CUTLASS's
    grouped GEMM tiles each group: a group of up to a tile's rows reads its matrix once, where tiles on the
    product's own row grid read it once a tile the group touches (two for most of Kimi Linear's
    128-row experts at 128-row tiles). The tiles number at most `cdiv(rows, tile)` plus the groups less
    one (a ceiling's sum exceeds the sum's ceiling by at most one a part). Upstream's
    `GroupInfo.create` walked every group's length in every tile, unrolled: at 129 groups (Qwen3's 128
    experts and the unrouted rows) two thirds of a down projection's instructions, and its global loads a
    third of its stalls (NCU-full)."""
    lengths = lengths.astype(jnp.int32)
    ends = _prefix_sums(lengths)
    starts = ends - lengths
    rows = jnp.int32(tile)
    first_rows = starts // unit * unit
    counts = lax.div(ends - first_rows + rows - 1, rows) * (lengths > 0)
    firsts = _prefix_sums(counts) - counts
    tids = jnp.arange(tiles, dtype=jnp.int32)[:, None]
    ours = (firsts[None] <= tids) & (tids < (firsts + counts)[None])
    group = jnp.argmax(ours, axis=1).astype(jnp.int32)
    found = jnp.any(ours, axis=1)
    block_start = first_rows[group] + (tids[:, 0] - firsts[group]) * rows
    columns = (group, block_start, starts[group], ends[group])
    return jnp.stack([jnp.where(found, column, 0) for column in columns], axis=1)


def _prefix_sums(values: jax.Array) -> jax.Array:
    """`values`' inclusive prefix sums, as one sum over a lower-triangular mask: a group count's square is
    a few hundred thousand integers at most, which XLA fuses into the table's own kernels, where
    `jnp.cumsum` lowers to reduce-window kernels of its own. The table so took 6.4 to 7.2 us against 18 to
    20 at 129 and 257 groups, decode to training rows (row 3622, FS-3622-TABLE1-h100), the same table."""
    count = values.shape[0]
    held = jnp.arange(count)[None, :] <= jnp.arange(count)[:, None]
    return jnp.sum(jnp.where(held, values[None, :], 0), axis=1, dtype=values.dtype)


TMA_ROWS = 8
"""The rows a TMA descriptor tiles by, to which `gmm` pads its rows and a group's first tile starts."""


def row_unit(family: mosaic.Family, dtype) -> int:
    """Dew: the rows a tile's first row is a multiple of (`tile_groups`): a TMA descriptor's 8 (`TMA_ROWS`),
    or where a family's copies start at whole tiles of its layout (`mosaic.Transport.copies_tiled_rows`: a
    TPU core's 8 sublanes of 32-bit words, 16 rows of bf16), the rows of `dtype` they hold."""
    return 8 * 4 // jnp.dtype(dtype).itemsize if family.copies_tiled_rows else TMA_ROWS


@dataclasses.dataclass(frozen=True)
class Allocations:
    """Dew: a GEMM launch's shared memory at its tile, as its kernel scopes it (`gmm_allocations`,
    `tgmm_allocations`): its streams' blocks `(shape, dtype)` by name (`mosaic.Family.streams`), what it
    scopes beside them in the kernel's order (`staged`: side operands' staging, `mosaic.Transport.staging`,
    a meeting's barriers, `mosaic.Transport.meeting`), and its outputs' tiles `(shape, dtype)`, which the
    family stages (`mosaic.Family.persist`)."""
    blocks: dict
    staged: tuple
    outputs: tuple

    def scope(self, family: mosaic.Family, tiles: "Tiles", steps: int | None = None,
              rewritten: bool = False) -> resources.Scope:
        """Its allocations by lifetime (`resources.shared_bytes`): its streams where its family places them
        (`mosaic.Family.lifetimes`, in no more slots than a walk's `steps` where they are known), and each
        output staged as its family stages it (`mosaic.Family.output_staging`, the allocations `persist`
        makes), after the walks where the family scopes them apart, its pipeline's memory reused; with
        `rewritten`, its streams' slots rewritten by its family's own role (`persist`'s `prepared`), whose
        publish barriers `persist` scopes too."""
        consumers = tiles.warpgroups
        staging = tuple(family.output_staging(self.outputs, consumers))
        rewrites = tuple(tiles.stages for _ in self.blocks) if rewritten else ()
        persisted, addresses = family.persist_scratch(consumers, rewrites)
        multicast = ('matrix',) if tiles.cluster > 1 and 'matrix' in self.blocks else ()
        streams = family.streams(self.blocks, tiles.stages, consumers, multicast=multicast)
        held, walks = family.lifetimes(streams, steps)
        # A core holding its values in scratch memory holds them beside (`mosaic.Issuer.value_scratch`).
        values = family.value_scratch(self.blocks, self.outputs)
        if walks:
            return resources.Scope(program=(*held, self.staged, persisted, values), walks=walks,
                                   after=tuple((buffer.shape, buffer.dtype) for buffer in staging),
                                   addresses=addresses)
        return resources.Scope(program=(*held, self.staged, persisted, staging, values), addresses=addresses)


def _typed(dtype) -> jax.ShapeDtypeStruct:
    """A stand-in for an operand of `dtype`, as `mosaic.Transport.staging` reads an operand's dtype."""
    return jax.ShapeDtypeStruct((), dtype)


def gmm_allocations(tiles: "Tiles", family: mosaic.Family, *, dtype, rhs_dtype, transpose_rhs: bool,
                    prologue_axes: tuple, prologue_dtypes: tuple, epilogue_axes: tuple,
                    epilogue_dtypes: tuple, dtypes: tuple, paired: bool = False,
                    transpose_lhs: bool = False, packed: bool = False, step_scaled: bool = False,
                    block_scaled: bool = False, interpret: bool = False) -> Allocations:
    """Dew: `gmm`'s allocations at `tiles` (`Allocations`), for rows of `dtype` and a matrix of `rhs_dtype`
    (two with `paired`), its side operands' axes and dtypes and its outputs' dtypes: a stream for the rows,
    each matrix and each prologue side operand that follows the steps (`_stepped`), each epilogue side
    operand's staging at a consumer's share of the tile, each prologue side operand without depth's at the
    tile's rows; a `transpose_lhs` call's rows stream their `[depth, rows]` blocks. A `packed` matrix (an
    NVFP4 one's, `scales`' in `gmm`) streams each 128 depths' words (`words_{q}`, a column's sixteen a
    64-byte row, which cp.async's two-dimensional copies take and a register tile loads whole) and its block
    scales K-major, decoded a stage at a time into bfloat16 buffers (`_stage_decoded`): compiled with
    consumers splitting whole 64-column shares, one set the consumers share, each decoding its share of the
    columns (`_shared_decode`), two more than its family keeps product groups in flight
    (`mosaic.Family.in_flight`), as a consumer may decode a stage ahead of another's products; else each
    consumer's own, one more, so no decode overwrites a stage a product still reads."""
    block_m, block_n, block_k = tiles.block_m, tiles.block_n, tiles.block_k
    # A consumer's accumulators: its share of the rows by the tile's columns, a pair's both where its products
    # span it (`tile_columns`); each stream's slot a multiprocessor's block.
    part = (block_m // tiles.warpgroups, tile_columns(tiles, family))
    stepped, _, fixed = _stepped(family, prologue_axes, 'depth')
    extent = {'rows': block_m, 'depth': block_k}
    rhs_block = (block_n, block_k) if transpose_rhs else (block_k, block_n)
    lhs_block = (block_k, block_m) if transpose_lhs else (block_m, block_k)
    matrix = ({**{f'words_{q}': ((block_n, 16), jnp.uint32) for q in range(block_k // 128)},
               'scales': ((block_n, block_k // 16), jnp.float8_e4m3fn)} if packed
              else {'matrix': (rhs_block, rhs_dtype)})
    blocks = {'rows': (lhs_block, dtype), **matrix,
              **({'paired': (rhs_block, rhs_dtype)} if paired else {}),
              **(_scale_blocks(block_m, block_n) if step_scaled else {}),
              # A block-scaled product's rows' and matrix's scales, a step's tiles of each (`gmm`'s
              # `mx_scales`), four scales a tile's row (`block_format`).
              **(_scale_tiles(block_m, block_n, block_k, dtype) if block_scaled else {}),
              **{f'side_{i}': (axes.named(axes.held(prologue_axes[i]), extent), prologue_dtypes[i])
                 for i in stepped}}
    sides = family.staging('side', [(axes.named(axes.held(side_axes), {'rows': part[0], 'columns': part[1]}),
                                     _typed(side))
                                    for side_axes, side in zip(epilogue_axes, epilogue_dtypes, strict=True)])
    held = family.staging('fixed', [(axes.named(axes.held(prologue_axes[i]), extent),
                                     _typed(prologue_dtypes[i])) for i in fixed])
    shared = _shared_decode(tiles, interpret=interpret)
    decoded = ([] if not packed else [plgpu.SMEM((family.in_flight + 2, block_n, block_k), jnp.bfloat16)]
               if shared else [plgpu.SMEM((tiles.warpgroups, family.in_flight + 1, block_n, block_k),
                                          jnp.bfloat16)])
    return Allocations(blocks, (sides['side_smems'], sides['side_barriers'], held['fixed_smems'],
                                held['fixed_barriers'], decoded),
                       tuple((part, jnp.dtype(out)) for out in dtypes))


def block_format(dtype) -> tuple[jnp.dtype, int]:
    """Dew: a block-scaled product's scale dtype and the depths one scale covers, by its operands' `dtype`
    (`gmm`'s `mx_scales`): NVFP4's (E2M1 operands) an E4M3 scale every 16 (tcgen05's `scale_vec::4X`,
    `kind::mxf4nvf4`); MXFP8's (FP8 operands) an E8M0 scale every 32 (OCP MX, `scale_vec::1X`)."""
    if jnp.dtype(dtype) == jnp.dtype(jnp.float4_e2m1fn):
        return jnp.dtype(jnp.float8_e4m3fn), 16
    return jnp.dtype(jnp.float8_e8m0fnu), 32


def _scale_tiles(block_m: int, block_n: int, block_k: int, dtype) -> dict:
    """Dew: a block-scaled step's scale streams (`gmm_allocations`'): its rows' and its matrix's columns'
    scales in `tcgen05.scales_layout`'s tiles, `[128-row tiles, scale tiles, 32, 16]`, a tile four scales of
    each of 128 rows, so a step of `block_k` depths takes `block_k` over four blocks' depths of them
    (`block_format`)."""
    scale, per = block_format(dtype)
    depth_tiles = block_k // (4 * per)
    return {'a_scales': ((block_m // 128, depth_tiles, 32, 16), scale),
            'b_scales': ((block_n // 128, depth_tiles, 32, 16), scale)}


def _scale_blocks(block_m: int, block_n: int) -> dict:
    """Dew: a step-scaled call's scale streams (`gmm`'s and `tgmm`'s `step_scales`): a step's row of its
    rows' scales at the tile's rows and of its columns' at the tile's columns (a row of the matrix's coarser
    ones holding the tile's, `_column_scales`), unswizzled (`SCALE_SWIZZLES`)."""
    return {'row_scales': ((1, block_m), jnp.float32), 'column_scales': ((1, block_n), jnp.float32)}


SCALE_SWIZZLES = {'row_scales': swizzling.UNSWIZZLED, 'column_scales': swizzling.UNSWIZZLED}
"""Dew: the scale streams' swizzles (`_scale_blocks`): rows of fp32 a step, read a row at a time."""


def tgmm_allocations(tiles: "Tiles", family: mosaic.Family, *, lhs_dtype, rhs_dtype,
                     prologue_axes: tuple, prologue_dtypes: tuple, out_dtype, k_major: bool = False,
                     step_scaled: bool = False, unstaged: bool = False) -> Allocations:
    """Dew: `tgmm`'s allocations at `tiles` (`Allocations`): a ring for the rows, the cotangent and each
    prologue side operand with rows that follows the steps (`_stepped`), each side operand without rows'
    staging, the consumers' meeting where they prepare their stages (not `unstaged`, `rows_unstaged`), and
    its output's share of the tile."""
    block_m, block_n, block_k = tiles.block_m, tiles.block_n, tiles.block_k
    stepped, _, fixed = _stepped(family, prologue_axes, 'rows')
    extent = {'rows': block_k, 'depth': block_m}
    # A rows-contiguous call's blocks lie depth (or width) by rows, the rows their minor axis.
    blocks = {'ring_0': ((block_m, block_k) if k_major else (block_k, block_m), lhs_dtype),
              'ring_1': ((block_n, block_k) if k_major else (block_k, block_n), rhs_dtype),
              **(_scale_blocks(block_m, block_n) if step_scaled else {}),
              **{f'ring_{2 + position}': (axes.named(axes.held(prologue_axes[i]), extent),
                                          prologue_dtypes[i]) for position, i in enumerate(stepped)}}
    held = family.staging('fixed', [(axes.named(axes.held(prologue_axes[i]), extent),
                                     _typed(prologue_dtypes[i])) for i in fixed])
    # Two consumers meet at their halves' boundaries (`_staged_halves`); one meets none, nor do two reading
    # their stages as loaded.
    meeting = (family.meeting(tiles.warpgroups),) if tiles.warpgroups > 1 and not unstaged else ()
    return Allocations(blocks, (held['fixed_smems'], held['fixed_barriers'], *meeting),
                       (((block_m // tiles.warpgroups, block_n), jnp.dtype(out_dtype)),))


def _stepped(family: mosaic.Family, axes_of: tuple, stepping: str) -> tuple[tuple, tuple, tuple]:
    """Dew: which side operands a launch streams with its steps (each with the `stepping` axis, but a vector a
    family does not copy, `mosaic.Family.copies_vectors`, its tile's axes those it holds, `axes.held`), which
    its step reads from global memory (such a vector), and which it loads once a tile (none with the
    `stepping` axis)."""
    stepped = tuple(i for i, side_axes in enumerate(axes_of)
                    if stepping in side_axes and (len(axes.held(side_axes)) > 1 or family.copies_vectors))
    read = tuple(i for i, side_axes in enumerate(axes_of) if stepping in side_axes and i not in stepped)
    fixed = tuple(i for i, side_axes in enumerate(axes_of) if stepping not in side_axes)
    return stepped, read, fixed


def gmm(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, *, tiles: Tiles, transpose_rhs: bool,
        out_dtype, multiprocessors: int, interpret: bool | plgpu.InterpretGPUParams = False,
        prologue=None, prologue_operands: tuple = (), prologue_axes: tuple = (), epilogue=None,
        epilogue_operands: tuple = (), epilogue_axes: tuple = (),
        out_dtypes: tuple | None = None, dense: bool = False,
        paired: jax.Array | None = None, exact_sums: bool = False,
        transpose_lhs: bool = False, first: jax.Array | None = None,
        scales: jax.Array | None = None, step_scales: tuple | None = None, mx_scales: tuple | None = None,
        gather: jax.Array | None = None, scatter: jax.Array | None = None
        ) -> jax.Array | tuple[jax.Array, ...]:
    """Each group's rows of `lhs` `[rows, depth]` times its matrix of `rhs`, `[groups, depth, width]` or
    with `transpose_rhs` `[groups, width, depth]`: `[rows, width]` in `out_dtype`, the rows past the
    routed ones zero. The depth must divide by `tiles.block_k` and the width by `tiles.block_n`.
    With `dense`, the one matrix multiplies every row without a grouped schedule. With `paired`, a second
    matrix of `rhs`'s shape, each tile multiplies its rows by both, one load of the rows, two
    accumulators, which the epilogue takes together (`epilogue(acc, paired_acc, *side_tiles)`: a gate and
    an up projection with their SwiGLU) or the outputs are, both (`out_dtypes`). With `transpose_lhs` (a
    dense call without a prologue), `lhs` is `[depth, rows]`, its blocks read transposed by the products: a
    product whose rows are the call's weight (its swapped orientation, `Tiles.swapped`). With `first`, an
    int32 scalar, `rhs` is a stack of more matrices than `group_sizes` has groups, group `g` reading matrix
    `first + g` where it lies (a banked layer's experts); the grid and the tile table are the groups'.
    With `scales`, `rhs` is an NVFP4 matrix's (`gemm.NvFp4`), read transposed: its E2M1 codes eight a
    uint32 in the staged order (`[groups, width, depth / 8]`) and `scales` its blocks' E4M3 scales
    (`[groups, width, depth / 16]`); each step decodes its stage exactly into bfloat16 (each code times its
    block's scale over `bits.NVFP4_OWED`, `_stage_decoded`), so the product is the rows times those values,
    the deferred factor the caller's. Its depth step is whole 256s and its width tile whole 64s, with no
    hooks, pair, exact sums, cluster or split. With `step_scales` (`row_scales` `[depth / 128, rows]`,
    `column_scales` `[groups, depth / 128, width / c]`, fp32), each 128-deep step's exactly summed partial
    is multiplied by its rows' scales and its group's scales of its columns, `c` columns a scale, before it
    is added (`grouped_matmul.step_scaled`: DeepSeek-V3's blockwise FP8, row 4518); its step is 128 deep,
    with no hooks, pair, packed matrix or split. With `mx_scales`, an MX product (`_mx_refused`; row 4519).
    With `gather` (int32 `[rows]`), row `i` is `lhs[gather[i]]`, read where it lies by the producer
    (`gathered.Ring`; interpreted, `Ring.parts`); with `scatter` (int32 `[rows]`, a permutation), row `i`
    lands at the outputs' row `scatter[i]`, once (`mosaic.Scattered`): a mixture of experts' sort and its
    undoing fused into the loads and the stores (row 1376), each refused as `_tables_refused` says.

Dew: `prologue(tile, *side_tiles)` transforms each `[block_m, block_k]` tile of `lhs` (a cooperative
    tile's consumer, half its rows) in its shared memory stage before the warpgroup MMA reads it;
    `epilogue(acc, *side_tiles)` each fp32 accumulator tile (a cooperative consumer's, half its rows),
    returning one output or a tuple of them (`out_dtypes`). A side operand's dimensions follow the launch
    axes its `*_axes` entry names, 'rows' and 'depth' for the prologue's, 'rows' and 'columns' for the
    epilogue's, each brought to shared memory by TMA. On a family whose products span a pair of
    multiprocessors (`mosaic.Transport.pairs_product`, tcgen05's pair), a tile is the pair's: twice
    `tiles.block_m` rows by twice `tiles.block_n` columns, each multiprocessor storing its own rows."""
    block_n, block_k = tile_columns(tiles, family_of(tiles, lhs.dtype)), tiles.block_k
    if transpose_lhs and (not dense or prologue is not None):
        raise ValueError("a transposed lhs is a dense call's without a prologue")
    (k, rows), (matrices, k2, n) = lhs.shape if transpose_lhs else lhs.shape[::-1], rhs.shape
    # A dense call's one matrix is its one group, its sizes unread (a TPU core's dense launch passes none).
    g = matrices if group_sizes is None else group_sizes.shape[0]
    if transpose_rhs:
        k2, n = n, k2
    # A packed matrix's words hold eight depths each.
    k2 = k2 if scales is None else 8 * k2
    if scales is not None:
        _packed_taken(rhs, scales, tiles, transpose_rhs=transpose_rhs, hooked=(prologue, epilogue, paired),
                      transpose_lhs=transpose_lhs, depth=k2, width=n)
    _gmm_refused(rhs, tiles, (rows, k, matrices, k2, n), columns=block_n, dense=dense, first=first,
                 paired=paired)
    _pair_refused(tiles, family_of(tiles, lhs.dtype), transpose_lhs=transpose_lhs, exact_sums=exact_sums,
                  parts=(prologue, paired, scales, step_scales, mx_scales, gather))
    _tables_refused(gather, scatter, tiles, family_of(tiles, lhs.dtype), prologue=prologue, dense=dense,
                    transpose_lhs=transpose_lhs)
    rows = rows if gather is None else gather.shape[0]
    accumulators = 1 if paired is None else 2
    dtypes = tuple(jnp.dtype(dtype) for dtype in (out_dtypes or (out_dtype,)))
    if k == 0:
        # A scattered product's rows where its table says, as its stores put them.
        outs = scattered(_of_no_depth((rows, n), group_sizes, accumulators, epilogue, epilogue_operands,
                                      epilogue_axes, dtypes), scatter)
        return outs if out_dtypes else outs[0]
    plain = all(each is None for each in (prologue, epilogue, paired, scales)) and not transpose_lhs
    _mx_refused(mx_scales, tiles, family_of(tiles, lhs.dtype), transpose_rhs=transpose_rhs,
                plain=all(each is None for each in (prologue, epilogue, paired, scales, step_scales))
                and not (transpose_lhs or exact_sums))
    _gmm_split_refused(tiles, k, n, step_scales, exact_sums=exact_sums,
                       hooked=prologue is not None or epilogue is not None or paired is not None, plain=plain)
    # The rows padded and extended as the schedule walks them (`_walked_rows`); a family whose copies start
    # at whole tiles of its layout (`row_unit`) stores whole tiles of rows, its outputs at least a tile's.
    walked_family = family_of(tiles, lhs.dtype)
    unit, tiled = row_unit(walked_family, lhs.dtype), walked_family.copies_tiled_rows
    lhs, gather, scatter, m, extent = _walked_rows(
        lhs, gather, scatter, tiles, transpose_lhs=transpose_lhs, dense=dense,
        bounded=walked_family.bounded_loads and not interpret, unit=unit,
        least=tiles.block_m * tiles.cluster if tiled else 0)
    row_tile = tiles.block_m * tiles.cluster
    table = None if dense else _grouped_table(group_sizes, m, row_tile, first, unit)
    per = 1
    if step_scales is not None:
        column_scales, per = _column_scales(step_scales[1], n, block_n)
        step_scales = (rows_extended(rows_padded(step_scales[0], 1, TMA_ROWS), 1, extent), column_scales)
    prologue_operands = sides_extended(prologue_operands, prologue_axes, extent)
    epilogue_operands = sides_extended(epilogue_operands, epilogue_axes, extent)
    # A dense product's last wave balanced over the programs (`tail.balanced`), as the kernel gradient's: one
    # matrix, one accumulator, a tile a program, its programs' exact partials added by IEEE adds. A depth
    # split already shares its depth over the wave, its runs a schedule of their own (`_gmm_scheduled`), each
    # into its own slice of the outputs, which no tail shares.
    tail = None
    if dense and paired is None and tiles.cluster == 1 and tiles.splits == 1 and fixes_up(tiles, lhs.dtype):
        tail = balanced_tail(tiles, (-(-m // tiles.block_m), n // block_n), k // block_k,
                             (lhs.dtype.itemsize, rhs.dtype.itemsize), generation.gpu_target(),
                             multiprocessors)
    walked = (-(-m // row_tile) + (0 if dense else g)) * (n // block_n) * tiles.cluster * tiles.splits
    programs = launched(family_of(tiles, lhs.dtype), multiprocessors, walked, tail)
    outs = _gmm_mainloop(lhs, rhs, table, (m, n, k, g), tiles=tiles, transpose_rhs=transpose_rhs,
                         dtypes=dtypes if tiles.splits == 1 else (jnp.dtype(jnp.float32),),
                         multiprocessors=programs, interpret=interpret,
                         hooks=(prologue, prologue_axes, epilogue, epilogue_axes),
                         sides=(prologue_operands, epilogue_operands), dense=dense, paired=paired, tail=tail,
                         exact_sums=exact_sums, transpose_lhs=transpose_lhs, scales=scales,
                         step_scales=step_scales, scale_per=per, mx_scales=mx_scales, gather=gather,
                         scatter=scatter, stored=None if scatter is None else rows)
    if tiles.splits > 1:
        # Each run's fp32 partial, summed in its order (the same sums every call) and rounded once.
        total = functools.reduce(jnp.add, tuple(outs[0]))
        outs = tuple(total.astype(dtype) for dtype in dtypes)
    outs = tuple(out[:rows] for out in outs)
    return outs if out_dtypes else outs[0]


NVFP4_STEP = 256
"""The depth an NVFP4 matrix's step decodes (`gmm`'s `scales`): two 128-depth groups of its staged order, so
each column's block scales are a 16-byte row, a copy's least."""


def _packed_taken(rhs, scales, tiles: Tiles, *, transpose_rhs: bool, hooked: tuple, transpose_lhs: bool,
                  depth: int, width: int) -> None:
    """Dew: refuses a packed matrix's call (`gmm`'s `scales`) its decode does not take: its words read
    transposed, its scales its blocks', in `NVFP4_STEP`-deep steps and 64-column tiles, with no hook, pair,
    transposed rows, cluster, split or swap (its exact sums, each step's product apart, read the decoded
    stage as any)."""
    if (not transpose_rhs or rhs.dtype != jnp.uint32 or scales.shape != (rhs.shape[0], width, depth // 16)
            or tiles.block_k % NVFP4_STEP or tiles.block_n % 64 or any(hook is not None for hook in hooked)
            or transpose_lhs or tiles.cluster > 1 or tiles.splits > 1 or tiles.swapped):
        raise ValueError(f"an NVFP4 matrix {rhs.shape} with scales {scales.shape} is read transposed in "
                         f"256-deep steps and 64-column tiles with no hooks, pair, cluster or split, not by "
                         f"{tiles}")


def _of_no_depth(shape: tuple, group_sizes, accumulators: int, epilogue, epilogue_operands: tuple,
                 epilogue_axes: tuple, dtypes: tuple) -> tuple:
    """Dew: `gmm`'s outputs of a product of no depth, which sums nothing: its rows zeros through the epilogue
    on whole arrays (its hooks are a tile's elementwise arithmetic), with no launch (a walk of no steps has
    no block of the matrix to view: JAX 0.12 refuses a zero-extent slice); its side operands by group read at
    each row as XLA's program reads them (`axes.at_rows`, row 4489)."""
    rows = shape[0]
    zeros = (jnp.zeros(shape, jnp.float32),) * accumulators
    by_row = tuple(axes.at_rows(side, side_axes, group_sizes, rows) if axes.GROUPS in side_axes else side
                   for side, side_axes in zip(epilogue_operands, epilogue_axes, strict=True))
    values = zeros if epilogue is None else epilogue(*zeros, *by_row)
    values = values if isinstance(values, tuple) else (values,)
    return tuple(value.astype(dtype) for value, dtype in zip(values, dtypes, strict=True))


def split_runs(steps: int, splits: int) -> bool:
    """Dew: whether `steps` split into `splits` runs of a share each (the last's the rest, `_gmm_scheduled`)
    leaves every run some steps."""
    return (splits - 1) * -(-steps // splits) < steps


def launched(family: mosaic.Family, multiprocessors: int, walked: int, tail: tail_of.Tail | None) -> int:
    """Dew: the programs a launch walking `walked` tiles runs: the `multiprocessors` its residency holds at
    once, persistent, where its family streams across a program's tiles (a producer's ring loading the next
    tile's steps under the last one's stores) or a balanced tail splits its last wave (`tail`); a program a
    tile where each walk scopes its streams apart (`mosaic.Family.walks_scoped`: cp.async's pipeline,
    filled and drained a tile), which the block scheduler starts in the walk's order as multiprocessors free
    up. A static persistent walk's programs drift apart over many rounds, where the scheduler keeps the
    running tiles a window of the walk's order: on an A100, a program a tile ran 0.37x and 0.43x the
    persistent walk's time on Mixtral-8x7B's down and DeepSeek-V3's gate input gradients, 0.80x and 0.90x
    on Mixtral's and Qwen3-30B-A3B's gate forwards, 0.88-0.95x on four grouped kernel gradients and 0.63x on
    Qwen3-32B's gate_up input gradient (a 51200-deep walk), and 0.99-1.03x on three shorter dense rows, the
    outputs bitwise equal (NS-P8-WALK-a100, NS-P8-WALK2-a100)."""
    return multiprocessors if tail is not None or not family.walks_scoped else max(walked, 1)


@tuning.lowered_once(('shape', 'tiles', 'transpose_rhs', 'dtypes', 'multiprocessors', 'interpret', 'hooks',
                      'dense', 'tail', 'exact_sums', 'transpose_lhs', 'scale_per', 'stored'))
def _gmm_mainloop(lhs, rhs, table, shape: tuple, *, tiles: Tiles, transpose_rhs: bool, dtypes: tuple,
                  multiprocessors: int, interpret, hooks: tuple, sides: tuple, dense: bool = False,
                  transpose_lhs: bool = False, paired: jax.Array | None = None, tail=None,
                  exact_sums: bool = False, scales: jax.Array | None = None, step_scales: tuple | None = None,
                  scale_per: int = 1, mx_scales: tuple | None = None, gather=None, scatter=None,
                  stored: int | None = None) -> tuple:
    """Dew: `gmm` on its instruction family's bundle (`family_of`): `tiles.warpgroups` consumer roles, each
    its share of the rows by all the columns, one MMA a step (two: CUTLASS's cooperative split, 2-7% faster
    on every H100 scoreboard row than halving the columns), persistent over the tiles in a planar snake,
    the family owning the operands' streams across tiles, their staging and the outputs' stores
    (`mosaic.Family.persist`). Each consumer's prologue rewrites its share of the rows' slot, its epilogue
    takes its share of each side tile. `lhs` is padded; `table` is `tile_groups`' at the pair's rows (`gmm`).
    One multiprocessor a tile or, at `tiles.cluster` 2, a pair of them a cluster running two row tiles of
    one group, which load the group's matrix block once between them (`mosaic.Family.streams`' multicast);
    both run each pair's steps (or neither), its second's rows past the group stored nowhere; where the
    family's products span the pair (`mosaic.Transport.pairs_product`), its step blocks are the pair's
    (`_gmm_views`), one product over both, each one's accumulator its rows by the pair's columns. A
    `paired` matrix streams beside the first, its blocks multiplied by the same rows' into a second one.
    With a balanced `tail` (a dense product's), the tiles past the whole waves have their steps shared out
    over the programs, each tile's partials added by the program that walks its first step (`tail.Tail`,
    `mosaic.Family.fixed_up`), which runs the epilogue and stores it.
    Lowered once a configuration (`tuning.lowered_once`), however many layers call it."""
    family = family_of(tiles, lhs.dtype)
    prologue, prologue_axes, epilogue, epilogue_axes = hooks
    prologue_operands, epilogue_operands = sides
    m, n, k, g = shape
    block_m, block_n, block_k = tiles.block_m, tile_columns(tiles, family), tiles.block_k
    consumers, pair, share = tiles.warpgroups, tiles.cluster, tiles.block_m // tiles.warpgroups
    part = (share, block_n)
    # A stream for each operand that follows the steps, but a vector a family does not copy, which its
    # step reads from global memory; a prologue side operand without depth is loaded once a tile.
    stepped, read, fixed = _stepped(family, prologue_axes, 'depth')
    allocations = _gmm_allocated(tiles, family, (lhs, rhs, paired, scales, step_scales, mx_scales), hooks,
                                 sides, dtypes, transpose_rhs=transpose_rhs, transpose_lhs=transpose_lhs,
                                 interpret=bool(interpret))
    matrices = ('matrix',) if paired is None else ('matrix', 'paired')
    # The streams that follow the matrix's group and column tile: its own, and a packed one's scales.
    streamed = matrices + (() if scales is None else ('scales',))
    # A pair's tiles: g groups' boundaries, and the unrouted rows.
    grid = (pl.cdiv(m, block_m * pair) + (0 if dense else g), n // block_n)

    def body(*gmems):
        table_gmem, gmems = (None, gmems) if dense else (gmems[0], gmems[1:])
        (lhs_gmem, *matrix_gmems), refs = gmems[:1 + len(streamed)], gmems[1 + len(streamed):]
        (scale_gmems, mx_gmems, gather_gmem, scatter_gmem), refs = _scale_refs(refs, step_scales, mx_scales,
                                                                                 gather, scatter)
        prologue_gmems, side_gmems, o_gmems = _split(refs, len(prologue_operands), len(epilogue_operands))
        workspace = None
        if tail is not None:
            workspace, *o_gmems = o_gmems
        o_gmems = _outputs(tuple(o_gmems), scatter_gmem, bool(interpret))
        ready = None if tail is None else family.partials_ready((multiprocessors, consumers), bool(interpret))

        schedule = functools.partial(
            _gmm_scheduled, table_gmem=table_gmem, lhs_gmem=lhs_gmem, splits=tiles.splits, grid=grid,
            matrix_gmems=dict(zip(streamed, matrix_gmems, strict=True)), prologue_gmems=prologue_gmems,
            shape=(m, k // block_k, g), tiles=tiles, dense=dense, tail=tail, programs=multiprocessors,
            transpose_rhs=transpose_rhs, axes_of={i: prologue_axes[i] for i in stepped},
            interpreted=bool(interpret), transpose_lhs=transpose_lhs, traced=exact_sums,
            scale_gmems=scale_gmems, scale_per=scale_per, mx_gmems=mx_gmems, gather_gmem=gather_gmem)

        def scope(side_smems, side_barriers, fixed_smems, fixed_barriers, decoded, *, streams):
            def issue(wg, tile, walk, *, active):
                group, row_start, ni, start, *_ = tile
                if tail is not None and not active:
                    return None  # past its program's range: another program walks the tile
                fixed_refs = [smem.at[wg] for smem in fixed_smems]
                if active:
                    _fixed_loaded(family, prologue_gmems, prologue_axes, fixed, fixed_refs, fixed_barriers,
                                  wg=wg, rows=pl.ds(row_start, block_m))
                step = functools.partial(_gmm_step, family=family, prologue=prologue,
                                         prologue_axes=prologue_axes,
                                         fixed_tiles=dict(zip(fixed, fixed_refs, strict=True)),
                                         read_tiles=_reads(prologue_gmems, prologue_axes, read, group, g),
                                         mine_rows=pl.ds(wg * share, share), transpose_rhs=transpose_rhs,
                                         matrices=matrices, interpret=bool(interpret), first=start,
                                         transpose_lhs=transpose_lhs,
                                         decoded=_decoding(decoded, tiles, wg, bool(interpret)),
                                         scaling=_gmm_scaling(step_scales is not None, scale_per, block_n,
                                                              ni, pl.ds(wg * share, share)),
                                         block_scaled=mx_scales is not None)
                # The unrouted rows, and a slot past the last tile, multiply nothing: zeros.
                return _walked_products(family, walk, step, part, len(matrices), exact_sums)

            def finish(wg, tile, accs, store, *, active):
                if tail is not None and not active:
                    return
                _gmm_finished(accs, tile, wg, store, family=family, part=part, m=m, g=g, dense=dense,
                              row_tile=block_m * pair, epilogue=epilogue, o_gmems=o_gmems,
                              sides=tuple(zip(side_gmems, epilogue_axes, side_smems, side_barriers,
                                              strict=True)),
                              partials=None if tail is None else (workspace, ready))

            walks = (tiles.splits * grid[0] * grid[1],) if tail is None else (tail.segments, multiprocessors)
            family.persist(streams, walks, schedule, issue, finish, consumers=consumers,
                           axis="programs", outputs=allocations.outputs, interpret=bool(interpret),
                           in_flight=family.in_flight, unrolled=_unrolled(prologue, exact_sums),
                           accumulators=(part,) * len(matrices), **_summed(family, tiles, exact_sums))

        family.scoped(scope, *allocations.staged,
                      streams=family.streams(allocations.blocks, tiles.stages, consumers,
                                             multicast=matrices if pair > 1 and not family.pairs_product
                                             else (),
                                             swizzles=SCALE_SWIZZLES if step_scales is not None
                                             else None, **_gathered(gather, tiles, lhs.dtype, interpret)))

    out_type = _declared(m if stored is None else stored, n, tiles.splits, dtypes)
    operands = (lhs, rhs, *(() if paired is None else (paired,)), *(() if scales is None else (scales,)),
                *(step_scales or ()), *(mx_scales or ()), *(() if gather is None else (gather,)),
                *(() if scatter is None else (scatter,)))
    kernel = family.kernel(body, out_type=out_type,
                           grid=(multiprocessors // pair,), grid_names=("programs",), consumers=consumers,
                           interpret=interpret, approx_math=False, name="dew_gmm", cluster=pair,
                           resident=tiles.programs, scalars=0 if dense else 1,
                           cost=unit_cost.estimate(2 * m * n * k * len(matrices), *operands,
                                                   *prologue_operands, *epilogue_operands, *out_type),
                           **_prefetched(tiles, grid, k // block_k, transpose_rhs, interpret,
                                         dense=dense, parts=(tail, scales, mx_scales, gather)))
    return kernel(*(operands if dense else (table, *operands)), *prologue_operands, *epilogue_operands,
                  *_workspace(tail, (multiprocessors, 2, consumers, *part), interpret))


def _prefetched(tiles: Tiles, grid: tuple, steps: int, transpose_rhs: bool, interpret, *, dense: bool,
                parts: tuple) -> dict:
    """Dew: a dense `gmm`'s matrix prefetched into L2 before its launch's wait for the one before it
    (`bodies.dependent`'s `prefetched`, `bodies.waits`): each program's first tile's matrix blocks of its
    first stages, so its first loads hit L2 rather than wait on memory under the previous launch's tail
    (M-FS-12: a decode GEMM's prologue exposed 3-5%; row 3601). A plain dense product's alone (no tail,
    scales or gathered rows: `parts` all None), whose first tile a program's index states (its persistent
    walk's first, `persistent.tiled`); a grouped one's needs its routing, written before the wait. A
    prefetch reads nothing into the program, so a matrix the launch before it still writes is prefetched
    early, never read stale."""
    plain = dense and all(part is None for part in parts)
    if not (plain and tiles.splits == 1 and tiles.cluster == 1
            and bodies.waits(interpret)):
        return {}
    block = (tiles.block_n, tiles.block_k) if transpose_rhs else (tiles.block_k, tiles.block_n)

    def prefetched(lhs_gmem, matrix_gmem, *refs):
        # The matrix is the launch's second operand, after the rows (`gmm`'s operands: a dense call's
        # take no table).
        del lhs_gmem, refs
        _, column_tile = tail_of.planar_snake(lax.axis_index("programs"), grid, 1, tiles.grid_block_n)
        view = (_matrix_columns(matrix_gmem.at[0], column_tile, block_n=tiles.block_n,
                                transpose_rhs=transpose_rhs), _rhs_index(transpose_rhs))
        return [mosaic.viewed(view, block, step) for step in range(min(tiles.stages, steps))]

    return {'prefetched': prefetched}


def _declared(rows: int, n: int, splits: int, dtypes: tuple) -> tuple:
    """Dew: a `gmm` launch's outputs, `[rows, n]` each, a depth split's each run's partial a slice of its own
    (`[splits, rows, n]`). A scattered product declares its stored rows alone, which its table's entries land
    on, each once: its padded rows land nowhere, so rows past them would be declared and never written
    (NSNAN-2)."""
    return tuple(jax.ShapeDtypeStruct((rows, n) if splits == 1 else (splits, rows, n), dtype)
                 for dtype in dtypes)


def _gmm_finished(accs: tuple, tile: tuple, wg, store, *, family: mosaic.Family, part: tuple, m: int, g: int,
                  dense: bool, row_tile: int, epilogue, o_gmems: tuple, sides: tuple, partials) -> None:
    """Dew: a consumer's share of a `gmm` tile stored (`_finished`) from its accumulators: at once, or,
    a balanced tail's tile (`partials`: its workspace and readiness flags), once its program's partial and
    the other programs' are added (`mosaic.Family.fixed_up`)."""
    group, row_start, ni, _, gathered, program, split = tile
    # A depth split's run stores its partial in its own slice of each output.
    o_gmems = o_gmems if split is None else tuple(
        dataclasses.replace(o_gmem, ref=o_gmem.ref.at[split]) if isinstance(o_gmem, mosaic.Scattered)
        else o_gmem.at[split] for o_gmem in o_gmems)
    # This consumer's share of the tile: its first row, and its group's rows there.
    part_row, column = row_start + wg * part[0], ni * part[1]
    first, stored = _stored_rows(group, part_row, part[0], m, dense=dense, row_tile=row_tile)
    finished = functools.partial(
        _finished, family=family, wg=wg, epilogue=epilogue, o_gmems=o_gmems, store=store, sides=sides,
        origins={'rows': pl.ds(part_row, part[0]), 'columns': pl.ds(column, part[1])},
        group=_side_group(group, g),
        place=(part_row, first, stored, column, min(part[0], m), TMA_ROWS if dense else 1))
    if partials is None or partials[0] is None:
        finished(*accs)
        return
    workspace, ready = partials
    first, gathered = gathered
    family.fixed_up(accs[0], workspace.at[:, 0, wg], None if ready is None else ready.at[:, wg], program,
                    first, gathered, finished, firsts=workspace.at[:, 1, wg])


def _stored_rows(group, part_row, rows: int, m: int, *, dense: bool, row_tile: int) -> tuple:
    """Dew: a consumer's share of a tile, `rows` from `part_row`: the first of them its group's, and how many
    (a dense product's every row to the padded rows `m`, whole where they fill `row_tile`s)."""
    if dense:
        return 0, (rows if m % row_tile == 0 else jnp.clip(m - part_row, 0, rows))
    first = jnp.clip(group.actual_start - part_row, 0, rows)
    return first, jnp.clip(group.actual_end - part_row, 0, rows) - first


def _fixed_loaded(family: mosaic.Family, prologue_gmems, prologue_axes: tuple, fixed, refs, barriers, *, wg,
                  rows) -> None:
    """Dew: a consumer's prologue side tiles without depth (`fixed`: loaded once a tile, at its `rows`) into
    its `refs`, each in the layout of its own axes (`Family.loaded`)."""
    family.loaded([(prologue_gmems[i].at[rows], ref, barrier.at[wg])
                   for i, ref, barrier in zip(fixed, refs, barriers, strict=True)],
                  axes=tuple(axes.held(prologue_axes[i]) for i in fixed))


def _finished(*accs, family: mosaic.Family, wg, epilogue, o_gmems: tuple, store, sides: tuple, origins: dict,
              group, place: tuple) -> None:
    """Dew: a consumer's share of a `gmm` tile from its accumulators (`accs`), read out a chunk of columns
    at a time (`mosaic.Issuer.chunks`: all of them at once where its registers hold them): its epilogue on
    its side tiles' same columns (`sides`, each its global ref, axes, staging and barrier, loaded at
    `origins`, a side operand by group at the tile's `group`, the unrouted rows' the last's, as XLA's
    program reads them, and handed the epilogue at each of the share's rows), then each output stored at
    `place`, its column the chunk's (`mosaic.Family.persist`'s `store`)."""
    family.loaded([(gmem.at[axes.named(side_axes, origins | axes.at_group(side_axes, gmem.shape, group))],
                    smem.at[wg], barrier.at[wg])
                   for gmem, side_axes, smem, barrier in sides],
                  axes=tuple(axes.held(side_axes) for _, side_axes, _, _ in sides))
    rows = origins['rows'].size
    row_start, first, stored, column, *ladder = place
    for start, chunk in family.chunks(accs):
        columns = {'columns': slice(start, start + chunk[0].shape[-1])}
        values = chunk if epilogue is None else epilogue(*chunk, *(
            axes.by_rows(family.side_tile(smem.at[wg].at[axes.placed(axes.held(side_axes), columns)],
                                          axes.held(side_axes)), side_axes, rows)
            for _, side_axes, smem, _ in sides))
        values = values if isinstance(values, tuple) else (values,)
        for output, (o_gmem, value) in enumerate(zip(o_gmems, values, strict=True)):
            store(output, value, o_gmem, (row_start, first, stored, column + start, *ladder))


def _gmm_scheduled(index, *, table_gmem, lhs_gmem, matrix_gmems: dict, prologue_gmems, shape: tuple,
                   grid: tuple, tiles: Tiles, dense: bool, tail: tail_of.Tail | None, programs: int,
                   transpose_rhs: bool, axes_of: dict, interpreted: bool,
                   transpose_lhs: bool = False, splits: int = 1, traced: bool = False,
                   scale_gmems=None, scale_per: int = 1, mx_gmems=None, gather_gmem=None) -> tuple:
    """Dew: `gmm`'s tile at the persistent walk's `index` (`mosaic.Family.persist`'s schedule; its first row
    a multiple of its family's `row_unit`): whether it
    multiplies any row, its steps, its streams' views (`_gmm_views`) and its group, row block and column
    tile: its pair's but for this multiprocessor's row block; in a balanced tail, the first of the tile's
    steps its program walks and how many, its partials' gatherer and its program (`tail.Tail.segment`);
    with `splits`, its depth's run: the walk every tile's first run, then every
    tile's second (as many launches of a run each would walk them), each run a share of the steps, the
    last's the rest; `traced`,
    its steps a count the compiler cannot see (`mosaic.opaque`), which a walk takes as a loop's bound (an
    exactly summed walk's: cp.async's pipeline unrolled a static walk's drain, each drain step's fresh
    product its own, and at five stages the A100 forward's 128 x 128 x 64 tile spilled 40 bytes beside the
    running sum; an opaque count's drain, a loop, spilled none, and a merely traced constant left ptxas
    spilling 32 bytes at the input gradient's 64 x 160 x 32 at five stages, NSDBG-5, NSDBG-DR1).
    `shape` is the padded rows, the steps and the groups."""
    m, steps, groups = shape
    pair, unit = tiles.cluster, row_unit(family_of(tiles, lhs_gmem.dtype), lhs_gmem.dtype)
    index = tail_of.ordered(tail, index, programs, interpreted)
    split = None
    if tail is not None:
        snaked, start, count, gathered = tail.segment(index, programs, grid[0] * grid[1])
    elif splits == 1:
        snaked, start, count, gathered = index[0], 0, mosaic.opaque(steps) if traced else steps, None
    else:
        share = -(-steps // splits)
        split, snaked = index[0] // (grid[0] * grid[1]), index[0] % (grid[0] * grid[1])
        start, gathered = split * share, None
        count = jnp.minimum(share, steps - start)
    mi, ni = tail_of.planar_snake(snaked, grid, 1, tiles.grid_block_n)
    group = (GroupInfo.dense(tiles.block_m * pair, mi, m) if dense
             else GroupInfo.create(table_gmem, tiles.block_m * pair, mi, unit))
    # This multiprocessor's row tile of its pair, from its group's first: a multiple of the rows a TMA
    # descriptor tiles by, as Mosaic's layout inference reads a slice's start.
    row_start = pl.multiple_of(group.block_start + mosaic.cluster_index(pair) * tiles.block_m, unit)
    views = _gmm_views(lhs_gmem, matrix_gmems, prologue_gmems, group, row_start, ni, tiles=tiles,
                       groups=groups, transpose_rhs=transpose_rhs, axes_of=axes_of, first=start,
                       transpose_lhs=transpose_lhs, scale_gmems=scale_gmems, scale_per=scale_per,
                       mx_gmems=mx_gmems, gather_gmem=gather_gmem)
    active = (count > 0 if tail is not None else jnp.ones((), jnp.bool_) if dense
              else (group.group_id < groups) & (group.actual_size > 0))
    return active, count, views, (group, row_start, ni, start, gathered, index[-1], split)


def _gmm_split_refused(tiles: Tiles, k: int, n: int, step_scales, *, exact_sums: bool, hooked: bool,
                       plain: bool) -> None:
    """Dew: `gmm`'s refusal of a depth split but a product's of no hooks or pair, each run some steps (a
    grouped one's runs over each tile its group's rows, the rows past every group's zeros in every run, so
    their partials sum to the product: a decode step's experts, row 3622; an exact sum's runs summed exactly,
    their partials by IEEE adds), of step scales but on an
    exactly summed product of no hook, pair, packed matrix or split (`_step_scaled_takes`)."""
    if tiles.splits > 1 and (hooked or not split_runs(k // tiles.block_k, tiles.splits)):
        raise ValueError(f"a depth split is a product's of no hooks or pair, each run some steps, not "
                         f"{tiles} of a {k}-deep one")
    if step_scales is not None and not _step_scaled_takes(step_scales, tiles, k, n, exact_sums=exact_sums,
                                                          plain=plain):
        raise ValueError(f"a step-scaled product sums each {SCALE_DEPTH}-deep step exactly with no "
                         f"hook, pair, packed matrix or split, its scales covering its steps and columns, "
                         f"not {tiles}")


def _gmm_allocated(tiles: Tiles, family: mosaic.Family, operands: tuple, hooks: tuple, sides: tuple,
                   dtypes: tuple, *, transpose_rhs: bool, transpose_lhs: bool,
                   interpret: bool = False) -> Allocations:
    """Dew: `gmm_allocations` for `_gmm_mainloop`'s call: its `operands` (rows, matrix, paired matrix, packed
    scales, step scales), its hooks' axes and side operands' dtypes, and its outputs' `dtypes`."""
    lhs, rhs, paired, scales, step_scales, mx_scales = operands
    _, prologue_axes, _, epilogue_axes = hooks
    prologue_operands, epilogue_operands = sides
    return gmm_allocations(tiles, family, dtype=lhs.dtype, rhs_dtype=rhs.dtype, transpose_rhs=transpose_rhs,
                           prologue_axes=prologue_axes,
                           prologue_dtypes=tuple(side.dtype for side in prologue_operands),
                           epilogue_axes=epilogue_axes,
                           epilogue_dtypes=tuple(side.dtype for side in epilogue_operands), dtypes=dtypes,
                           paired=paired is not None, transpose_lhs=transpose_lhs, packed=scales is not None,
                           step_scaled=step_scales is not None, block_scaled=mx_scales is not None,
                           interpret=interpret)


def _mx_refused(mx_scales, tiles: Tiles, family: mosaic.Family, *, transpose_rhs: bool, plain: bool) -> None:
    """Dew: `gmm`'s refusal of a block-scaled product (`mx_scales`: the rows' and the matrices' scales, one
    every block of depths, `block_format`'s, tiled as `tcgen05.scales_layout` lays them out, `[rows / 128,
    scale tiles, 32, 16]` and `[groups, width / 128, scale tiles, 32, 16]`, the rows' in the product's row
    order, a gathered product's its tokens' at the table (the caller's gather, `gemm._block_scaled`, row
    4851); FP8 or E2M1 rows times a K-major
    matrix of theirs, each block of each scaled in the products, `mosaic.Issuer.scaled_products`) it does
    not take: but on a family whose
    products take block scales, its matrix K-major, with no hook, pair, packed matrix, step scales,
    transposed rows, exact sums or split, its tile whole 128s of rows, columns and depth (the scales'
    tiles), each group's rows whole 128-row tiles (the caller's)."""
    if mx_scales is None:
        return
    if not (family.takes_block_scales and transpose_rhs and plain and tiles.splits == 1
            and tiles.block_m % 128 == 0 and tiles.block_n % 128 == 0 and tiles.block_k % 128 == 0):
        raise ValueError(f"a block-scaled product runs on a family whose products take block scales "
                         f"({family.name}'s do not, or its call or {tiles} is not one it takes)")


def _outputs(o_gmems: tuple, scatter_gmem, interpret: bool) -> tuple:
    """Dew: a `gmm` kernel's outputs, each `mosaic.Scattered` through `scatter_gmem` where the call has
    one."""
    if scatter_gmem is not None:
        return tuple(mosaic.Scattered(o_gmem, scatter_gmem, interpret) for o_gmem in o_gmems)
    return o_gmems


def _workspace(tail, shape: tuple, interpret) -> tuple:
    """Dew: a balanced tail's partials (`tail.Tail`): each program's, a consumer's share of a tile each, of
    the tile its range starts inside and of the one it walks the first step of (`Family.fixed_up`'s
    `firsts`), a ref the kernel writes in place, the last of its operands; none without a tail. Its slots
    are written only where programs share a tile, so it is no declared output, which would leave the rest
    unwritten (NSNAN-2), and it starts uninitialized (`mosaic.workspace`)."""
    return () if tail is None else (mosaic.workspace(shape, jnp.float32, interpret=interpret),)


def _scale_refs(refs: tuple, step_scales, mx_scales, gather, scatter) -> tuple:
    """Dew: a `gmm` kernel's step scales', MX scales' (each two), gather table's and scatter table's
    references, leading the rest in that order (None where the call has none), and the references after
    them."""
    step, refs = (refs[:2], refs[2:]) if step_scales is not None else (None, refs)
    mx, refs = (refs[:2], refs[2:]) if mx_scales is not None else (None, refs)
    gathered, refs = (refs[0], refs[1:]) if gather is not None else (None, refs)
    scattered, refs = (refs[0], refs[1:]) if scatter is not None else (None, refs)
    return (step, mx, gathered, scattered), refs


def _gathered(gather, tiles: Tiles, dtype, interpret) -> dict:
    """Dew: a gathered product's rows stream (`family.streams`' arguments): its producer's gather
    (`gathered.Ring`), or interpreted, where no lane-level copy runs, each row's copies (`Ring.parts`, its
    slot staged where layout inference lays it out, at no stated swizzle: `mosaic.piece_copies`), that many
    arrivals at its load barrier; none ungathered."""
    if gather is None:
        return {}
    if interpret:
        copies = mosaic.piece_copies((tiles.block_m, tiles.block_k), dtype, None, 1)
        return {'copies': {'rows': tiles.block_m * copies}}
    return {'gathered': ('rows',)}


def _grouped_table(group_sizes: jax.Array, m: int, row_tile: int, first: jax.Array | None,
                   unit: int = TMA_ROWS) -> jax.Array:
    """A grouped call's tile table (`tile_groups`) over its `m` padded rows at `row_tile`-row tiles: its
    groups', then the rows past the routed ones a last group, whose tiles store zeros."""
    g = group_sizes.shape[0]
    lengths = jnp.concatenate([group_sizes.astype(jnp.int32),
                               jnp.maximum(m - jnp.sum(group_sizes, dtype=jnp.int32), 0)[None]])
    table = tile_groups(lengths, row_tile, pl.cdiv(m, row_tile) + g, unit)
    if first is None:
        return table
    # Each tile's matrix, its group's of the stack from `first` (`GroupInfo.matrix`): the group itself stays
    # the call's, which its validity and its side operands read.
    return jnp.concatenate([table, jnp.minimum(table[:, :1], g - 1) + first], axis=1)


def _walked_rows(lhs, gather, scatter, tiles: Tiles, *, transpose_lhs: bool, dense: bool,
                 bounded: bool, unit: int = TMA_ROWS, least: int = 0) -> tuple:
    """Dew: `gmm`'s rows as its schedule walks them: `lhs`, or a gathered product's table (`gather`, a pad
    reading row 0 of the tokens, which stay as they are), padded to whole `unit`s of rows and to `least`
    (`m` of them; `row_unit`: a TMA descriptor's 8; a tile's rows where its family stores whole tiles) and
    extended to the schedule's extent where the family's loads past
    the rows are not zeros (`bounded`), and a scattered product's table to the extent, its padded and
    extended rows stored nowhere (`mosaic.Scattered`): `(lhs, gather, scatter, m, extent)`. The rows past
    the routed ones, the padding among them, are a last group, whose tiles store zeros; a group's tiles
    start at its first row (`tile_groups`), so a tile reads up to a tile's rows past the last group's first.
    Where the loads read zeros past an operand's rows (TMA's) the operands stay as they are; elsewhere every
    row-bearing operand is allocated to the extent, zeros past its rows, which a group's stores never reach
    (a 128-row lhs of groups [120, 8] at 64-row tiles read rows 120 to 183 unpredicated on an A100,
    cp.async's loads), as the interpreter's are, whose slices past an array clamp rather than read zeros. On
    an H100 that copy was a kernel of its own each call, 0.035 to 0.33 ms beside a dense product of 0.34 to
    2.4 (BREAKDOWN1-h100). A dense call's tiles start at whole row tiles, so it reads to its last tile's end
    alone, no copy where its rows fill whole tiles: on an A100 that copy was 0.094 ms of Llama 3 8B o's
    1.52 ms forward (BC-A100-EXTENT-AB1). A table is always extended: its threads read a tile's entries by
    plain loads, which no TMA bounds (a gathered product's last tile read past an unextended table on a
    GH200, CUDA_ERROR_ILLEGAL_ADDRESS, FS-4346-base-sm90)."""
    row_axis = 1 if transpose_lhs else 0
    walked, axis = (lhs, row_axis) if gather is None else (gather, 0)
    walked = rows_padded(walked, axis, unit)
    walked = rows_extended(walked, axis, max(walked.shape[axis], least))
    m = walked.shape[axis]
    row_tile = tiles.block_m * tiles.cluster
    tabled = gather is not None or scatter is not None
    extent = (m if bounded and not tabled else -(-m // row_tile) * row_tile if dense
              else row_extent(m, row_tile, unit))
    walked = rows_extended(walked, axis, extent)
    if scatter is not None:
        scatter = jnp.pad(scatter.astype(jnp.int32), (0, extent - scatter.shape[0]), constant_values=-1)
    return (walked, None, scatter, m, extent) if gather is None else (lhs, walked, scatter, m, extent)


def _tables_refused(gather, scatter, tiles: Tiles, family: mosaic.Family, *, prologue, dense: bool,
                    transpose_lhs: bool) -> None:
    """Dew: `gmm`'s refusal of a gathered product (`gather`) but on a family that gathers rows beside the
    matrix's whole loads (`Family.gathers_rows`, `gathers_alone`), of no prologue, transposed rows or dense
    call, and of a scattered one
    (`scatter`) but a grouped call's on such a family (its stores `mosaic.staged_rows`')."""
    gathers = family.gathers_rows and family.gathers_alone
    if gather is not None and (prologue is not None or dense or transpose_lhs or not gathers):
        raise ValueError(f"a gathered product's rows are a ring family's gather, with no prologue, "
                         f"transposed rows or dense call, not {tiles}'s")
    if scatter is not None and (dense or not gathers):
        raise ValueError(f"scattered rows are a grouped call's on a ring family, not {tiles}'s")


def _tgmm_tables_refused(tables: tuple, tiles: Tiles, family: mosaic.Family, *, prologue, dense: bool,
                         k_major: bool, step_scaled: bool) -> None:
    """Dew: `tgmm`'s refusal of a gathered operand (`tables`, each operand's) but on a family that gathers
    rows, of no rows-contiguous layout, step scales or dense call, and the rows' table of no prologue (which
    `gmm`'s refuses alike, `_tables_refused`)."""
    if any(table is not None for table in tables) and (
            dense or k_major or step_scaled or not (family.gathers_rows and family.gathers_alone)
            or (tables[0] is not None and prologue is not None)):
        raise ValueError(f"a gathered kernel gradient's rows are a ring family's gather, with no prologue on "
                         f"gathered rows, rows-contiguous layout, step scales or dense call, not {tiles}'s")


def depth_masked(prologue, tile, *sides):
    """Dew: `prologue` of a stage `tile` and its side tiles, the last a vector over the depth of ones where
    the call's depth runs and zeros past it (`grouped_matmul.MosaicGmm._aligned`'s, its padding), its result
    zeroed past it: the padding's zeros go into the product as its own, where a prologue of them (a
    reciprocal's infinities) would make NaNs of them."""
    *own, valid = sides
    transformed = prologue(tile, *own)
    # The vector broadcast straight to the tile's shape, as a side vector's arithmetic broadcasts it: a
    # `[1, depth]` reshape between took no layout Mosaic GPU infers ("Tiling rank different than broadcast
    # result rank, (8,) vs (1, 64)", the GEMM compile gate's unaligned projection on sm90).
    return jnp.where(valid > 0, transformed, jnp.zeros_like(transformed))


def pair_takes(tiles: Tiles, family: mosaic.Family) -> bool:
    """Dew: whether `tiles` is a tile of a family whose products span a pair
    (`mosaic.Transport.pairs_product`) where it is one: its cluster the pair, each CTA's columns at most its
    half of one product (`mosaic.Issuer.widest_product`, tcgen05's 256 over a pair); any tile on another
    family."""
    if not family.pairs_product:
        return True
    widest = family.widest_product
    return tiles.cluster == family.cluster and (widest is None or tiles.block_n <= widest)


def _pair_refused(tiles: Tiles, family: mosaic.Family, *, transpose_lhs: bool, exact_sums: bool,
                  parts: tuple) -> None:
    """Dew: `gmm`'s refusal of a pair's product (`mosaic.Transport.pairs_product`) but a plain one's at a
    pair's tile (`pair_takes`): its step views are its rows' and its matrix's alone (`_gmm_views`), so none of
    `parts` (a prologue, a second matrix, scales, gathered rows), transposed rows, exact sums or a split."""
    if family.pairs_product and (not pair_takes(tiles, family) or transpose_lhs or exact_sums
                                 or tiles.splits > 1 or any(part is not None for part in parts)):
        raise ValueError(f"a pair's product is a pair's tile's, of no prologue, transposed or gathered rows, "
                         f"second matrix, scales, exact sums or split, not {tiles}")


def _gmm_refused(rhs, tiles: Tiles, extents: tuple, *, columns: int, dense: bool, first, paired) -> None:
    """Dew: `gmm`'s refusal of a call its launch does not take: a dense product of several matrices, a stack
    read from its first matrix but a grouped product's of one, a paired matrix not its product's shape, or
    extents (`rows`, `depth`, `matrices`, the matrix's depth, `width`) its tile's step and `columns`
    (`tile_columns`) do not divide."""
    rows, k, matrices, k2, n = extents
    if dense and matrices != 1:
        raise ValueError("a dense product takes one matrix")
    if first is not None and (dense or paired is not None):
        raise ValueError("a stack read from its first matrix is a grouped product's of one matrix")
    if paired is not None and paired.shape != rhs.shape:
        raise ValueError(f"a paired matrix {paired.shape} is not its product's {rhs.shape}")
    if k != k2 or k % tiles.block_k or n % columns:
        raise ValueError(f"a [{rows}, {k}] x [{matrices}, {k2}, {n}] product does not take {tiles}")


def _step_scaled_takes(step_scales: tuple, tiles: Tiles, depth: int, width: int, *, exact_sums: bool,
                       plain: bool) -> bool:
    """Dew: whether `gmm` takes `step_scales` at `tiles` (`_step_scaling`): an exactly summed product of no
    hook, pair, packed matrix, transposed rows or split, a step 128 deep, a row scale each step, and a
    column scale each column or each multiple of the tile's width."""
    rows, columns = step_scales
    per = width // columns.shape[-1] if columns.shape[-1] and width % columns.shape[-1] == 0 else 0
    return (exact_sums and plain and tiles.splits == 1 and tiles.block_k == SCALE_DEPTH
            and rows.shape[0] * SCALE_DEPTH == depth
            and (per == 1 or (per > 0 and per % tiles.block_n == 0)))


def _matrix_of(group, groups: int):
    """Dew: the matrix a `gmm` tile of `group` multiplies: its group's, the unrouted rows' the last's, or the
    stack's from `first` (`GroupInfo.matrix`)."""
    return jnp.minimum(group.group_id, groups - 1) if group.matrix is None else group.matrix


def _unrolled(prologue, exact_sums: bool) -> bool:
    """Dew: whether a `gmm` walk takes a rotation of its ring a trip (`mosaic.stepping`): where its step is
    its products and its waits, small enough that a rotation unrolled takes fewer instructions an HGMMA;
    not a prologue's rewrite, nor an exactly summed step (its product apart, its IEEE add beside the running
    sum), a rotation of which spilled as the ring deepened (row 3620: FP8 at 128 x 128 x 128 spilled 152
    bytes at five stages and 424 at six, 34% slower than at four, FS-3620-SERIAL2, FS-3620-STAGES1-h100)."""
    return prologue is None and not exact_sums


def _zeroed_atoms(family: mosaic.Family, dtype, block_n: int, consumers: int) -> tuple[int, int]:
    """Dew: the columns a consumer zeroes of the cotangent's stage at a time (`zeroing_columns`) and how many
    such atoms each consumer zeroes (`_zeroed_share`): as many each as the stage's divide among them, the
    last's share short where they do not divide evenly."""
    atom = zeroing_columns(family, dtype)
    return atom, -(-(block_n // atom) // consumers)


def zeroing_columns(family: mosaic.Family, dtype) -> int:
    """Dew: the columns of a stage a consumer zeroes at a time (`_zero_neighbours_of`), so its registers hold
    a few whatever the tile's width: a swizzled row, never narrower than its family's column unit (a TPU
    core's slice of a VMEM block is whole lanes, 128 columns)."""
    return max(family.column_unit(dtype), swizzling.swizzle_columns(dtype, swizzling.SWIZZLE_BYTES))


def _dtypes_of(arrays) -> tuple:
    """Dew: each of `arrays`' dtype."""
    return tuple(array.dtype for array in arrays)


def _scales_apart(refs: list, scaled: bool) -> tuple:
    """Dew: a step-scaled kernel's two scale references, which lead its side operands' (`tgmm`'s
    `step_scales`), and the rest; None and all of them where it has no step scales."""
    return (tuple(refs[:2]), refs[2:]) if scaled else (None, refs)


def _column_scales(column_scales: jax.Array, width: int, block_n: int) -> tuple[jax.Array, int]:
    """Dew: a step-scaled `gmm`'s column scales (`[groups, depth / 128, width / per]`) as its stream reads
    them (`_scale_blocks`), and the columns `per` a scale covers: as they are where each covers a column;
    else each step's row of them padded to whole `block_n`s, a block of which holds a tile's scale
    (`_column_block`)."""
    per = width // column_scales.shape[-1]
    if per == 1:
        return column_scales, per
    return jnp.pad(column_scales, ((0, 0), (0, 0), (0, -column_scales.shape[-1] % block_n))), per


def _column_block(column_tile, block_n: int, per: int) -> tuple:
    """Dew: the block of a step's column scales `column_tile`'s stream reads (`_column_scales`'), and where
    in it the tile's own scale lies: its columns' block, at its first, where a scale covers a column; else
    the block holding the tile's one scale, `per` columns (whole tiles of them) a scale."""
    if per == 1:
        return column_tile, 0
    scale = column_tile * block_n // per
    return scale // block_n, scale % block_n


def _tgmm_scaling(scaled: bool, mine_depth) -> Callable | None:
    """Dew: a step-scaled `tgmm` tile's `_step_scaling` (`tgmm`'s `step_scales`), its depth rows' and its
    columns' scales streamed, a scale a column; none where the call has no step scales."""
    if not scaled:
        return None
    return functools.partial(_step_scaling, per=1, offset=0, mine_rows=mine_depth)


def _gmm_scaling(scaled: bool, per: int, block_n: int, column_tile, mine_rows) -> Callable | None:
    """Dew: a step-scaled `gmm` tile's `_step_scaling` for a consumer's share of the rows (`mine_rows`), its
    matrix's column scales a scale every `per` columns (`_column_block`); none where the call has no step
    scales."""
    if not scaled:
        return None
    return functools.partial(_step_scaling, per=per, offset=_column_block(column_tile, block_n, per)[1],
                             mine_rows=mine_rows)


def _step_scaling(index, slots, *, per: int, offset, mine_rows) -> Callable:
    """Dew: a step-scaled walk's multiplier of a step's exactly summed partial (`gmm`'s and `tgmm`'s
    `step_scales`, `mosaic.Issuer.summed_apart`'s `scale`): this consumer's rows' scales (the step's row of
    them streamed, `row_scales`) times the tile's columns' (streamed beside them, `column_scales`: the step's
    row of them at the tile's columns, one a column, or the block holding the tile's one, at `offset`, `per`
    columns a scale), in shared memory as the step's operands land: read from global memory, a step's
    scaling waited on its load (M-NS-13: the blockwise FP8 template 1.24-1.42x its unscaled exact product).
    `index` is the step's, whose scales are its streams'."""
    del index
    rows = slots['row_scales'].at[0, mine_rows][...]
    column_scales = slots['column_scales']
    if per == 1:
        return lambda start, columns: rows[:, None] * column_scales.at[0, pl.ds(start, columns)][...][None, :]
    # A weight block's one scale scales the rows' vector before it is broadcast over the columns: a multiply
    # a row, not one an element of the partial beside its own (the same products, so the same bits).
    scaled = rows * column_scales[0, offset]
    return lambda start, columns: scaled[:, None]


def _gmm_views(lhs_gmem, matrix_gmems: dict, prologue_gmems, group, row_start, column_tile, *, tiles: Tiles,
               groups: int, transpose_rhs: bool, axes_of: dict, first=0, transpose_lhs: bool = False,
               scale_gmems=None, scale_per: int = 1, mx_gmems=None, gather_gmem=None) -> dict:
    """Dew: `gmm`'s streams' views at a tile (`group`'s rows from `row_start`, `column_tile`;
    `mosaic.viewed`): its rows', its group's matrix's (each of `matrix_gmems`, by stream: a paired
    product's two), and each prologue side operand's that follows the depth (`axes_of`, by operand), step
    `j` each one's `first + j`-th depth block (`first` a balanced tail's program's first step); a
    `transpose_lhs` call's rows as `[depth, rows]` blocks. On a family whose products span a pair
    (`mosaic.Transport.pairs_product`), the rows' and the matrix's views are the pair's (`mosaic.Paired`):
    its two row tiles from the group's block and its columns (`tile_columns`), each multiprocessor loading
    its half."""
    rows = pl.ds(row_start, tiles.block_m)
    family = family_of(tiles, lhs_gmem.dtype)
    rhs_index = _rhs_index(transpose_rhs)
    if family.pairs_product:
        pair_rows = pl.ds(group.block_start, tiles.block_m * tiles.cluster)
        columns = tile_columns(tiles, family)
        return {'rows': (lhs_gmem.at[pair_rows], lambda j: mosaic.Paired((0, first + j), 0)),
                **{name: (_matrix_columns(gmem.at[_matrix_of(group, groups)], column_tile, block_n=columns,
                                          transpose_rhs=transpose_rhs),
                          lambda j: mosaic.Paired(rhs_index(first + j), 0 if transpose_rhs else 1))
                   for name, gmem in matrix_gmems.items()}}
    lhs_view = ((lhs_gmem.at[:, rows], lambda j: (first + j, 0)) if transpose_lhs
                else (lhs_gmem.at[rows], lambda j: (0, first + j)) if gather_gmem is None
                # A gathered product's rows each where its table says (`mosaic.Rows`, a row a run).
                else (lhs_gmem, lambda j: mosaic.Rows(gather_gmem, row_start, (first + j,), 1)))
    matrix = _matrix_of(group, groups)
    # Each stream's view and its step's block index (`(row, column)` of a matrix, four of a scale tile's).
    matrices: dict[str, tuple[object, Callable[..., tuple]]] = {
        name: (_matrix_columns(gmem.at[matrix], column_tile, block_n=tiles.block_n,
                               transpose_rhs=transpose_rhs),
               lambda j: rhs_index(first + j)) for name, gmem in matrix_gmems.items()}
    if 'scales' in matrix_gmems:
        # A packed matrix's words, a stream for each 128 depths of a step: step j's group q, sixteen words a
        # column at the tile's columns.
        groups_of = tiles.block_k // 128
        words = matrices.pop('matrix')[0]
        matrices |= {f'words_{q}': (words, lambda j, q=q: (0, groups_of * (first + j) + q))
                     for q in range(groups_of)}
    if scale_gmems is not None:
        # Its rows' scales and its matrix's columns', a step's row of each (`step_scales`): the rows' at the
        # tile's rows, the columns' at its columns or the block of them holding its one (`_column_block`).
        row_scales, column_scales = scale_gmems
        block, _ = _column_block(column_tile, tiles.block_n, scale_per)
        matrices['row_scales'] = (row_scales.at[:, rows], lambda j: (first + j, 0))
        matrices['column_scales'] = (column_scales.at[matrix], lambda j: (first + j, block))
    if mx_gmems is not None:
        # An MX product's scale tiles at step j: its rows' 128-row tiles from the tile's (whole tiles: a
        # group's rows are whole 128s) and its matrix's at the tile's columns.
        row_scales, matrix_scales = mx_gmems
        matrices['a_scales'] = (row_scales.at[pl.ds(row_start // 128, tiles.block_m // 128)],
                                lambda j: (0, first + j, 0, 0))
        matrices['b_scales'] = (matrix_scales.at[matrix, pl.ds(column_tile * (tiles.block_n // 128),
                                                                tiles.block_n // 128)],
                                lambda j: (0, first + j, 0, 0))
    # A side operand by group at the tile's group (`axes.at_group`), its groups' axis indexed away.
    return {'rows': lhs_view, **matrices,
            **{f'side_{i}': (prologue_gmems[i].at[axes.placed(side_axes, {'rows': rows} | axes.at_group(
                                 side_axes, prologue_gmems[i].shape, _side_group(group, groups)))],
                             lambda j, side_axes=side_axes: axes.placed(axes.held(side_axes), {'rows': 0},
                                                                        first + j))
               for i, side_axes in axes_of.items()}}


def _reads(prologue_gmems, prologue_axes: tuple, read: tuple, group, groups: int) -> dict:
    """Dew: the prologue side vectors a step reads from global memory (`read`, by operand), one by group at
    the tile's `group` of `groups` (`_side_group`, `axes.at_group`)."""
    return {i: prologue_gmems[i].at[axes.placed(prologue_axes[i], axes.at_group(
        prologue_axes[i], prologue_gmems[i].shape, _side_group(group, groups)))] for i in read}


def _side_group(group, groups: int):
    """Dew: the group a tile reads its side operands by group at (`axes.at_group`): its own, the unrouted
    rows' the last's, as XLA's program reads them (`axes.at_rows`)."""
    return jnp.minimum(group.group_id, groups - 1)


def _summed(family: mosaic.Family, tiles: Tiles, exact_sums: bool) -> dict:
    """Dew: the persist arguments of an exactly summed launch on a family that sums its windows itself
    (`mosaic.Family.sums_in_persist`): a window's steps (`summed_window`); none on any other."""
    return {'summed': family.summed_window(tiles.block_k)} if exact_sums and family.sums_in_persist else {}


def _walked_products(family: mosaic.Family, walk, step, shape: tuple, count: int, exact_sums: bool) -> tuple:
    """Dew: a cooperative `gmm` consumer's `count` products over its steps (`walk`, `step` a `_gmm_step`
    whose operands it reads), each into an fp32 tile of `shape`: as the tensor cores add, or with
    `exact_sums` each step's product summed apart and added into the running sum by an IEEE add, as
    `_walked_gradient`'s: an FP8 product's running sum promoted into fp32 every step, Hopper's FP8 tensor
    cores summing within one in a narrower accumulator (testing.md, FP8 products). A family whose persist
    sums the windows itself (`mosaic.Family.sums_in_persist`, `_summed`) walks the steps plainly."""
    if not exact_sums or family.sums_in_persist:
        _, accs = walk(step, None, (shape,) * count)
        return accs

    def exact_step(index, slots, running, accumulators):
        _, totals = step.func(index, slots, None, running, **step.keywords, summed=True)
        return totals, accumulators

    running, _ = walk(exact_step, tuple(family.zeros(shape) for _ in range(count)), ())
    return running


def _gmm_step(index, slots, nothing, accumulators, *, family: mosaic.Family, prologue, prologue_axes: tuple,
              fixed_tiles: dict, read_tiles: dict, mine_rows, transpose_rhs: bool, matrices: tuple,
              interpret: bool, first=0, summed: bool = False, transpose_lhs: bool = False,
              decoded=None, scaling=None, block_scaled: bool = False):
    """Dew: a consumer's step of the cooperative `gmm` (`mosaic.Family.walk`): its half of the rows' slot,
    transformed by the prologue on its side tiles (the streamed ones' slots, the fixed ones' `fixed_tiles`,
    a depth vector the family does not copy read at step `index` from global memory: `read_tiles`),
    times each of `matrices`' slot into its accumulator (a paired product's two); with `summed`, each
    product summed apart into its accumulator, the running sum (`_walked_products`' exact sums,
    `mosaic.Issuer.summed_apart`). The prologue's
    value stays in registers, where the products take it as their left operand (warpgroup MMA's register
    operand A, mma.sync's fragments): rewritten into the stage and read back, the gated prologue of a
    Llama-3-8B gate_up ran at 8.3x its plain product on an H100 (row 1336). With `decoded` (a packed
    matrix's, `_Decoding`), the step decodes its stage of words and scales, or its share of them, into the
    stage's buffer (`_stage_decoded`), published and met before its products read it as the matrix's
    stage."""
    stage = slots['rows']
    read = {name: slots[name] for name in matrices if decoded is None}
    if decoded is not None:
        read['matrix'] = decoded.decode(index, slots, interpret)
        family.published()
        decoded.met()
    oriented = {name: read[name].transpose((1, 0)) if transpose_rhs else read[name] for name in matrices}
    # This warpgroup's half of the rows, the only half its MMA reads (a `[depth, rows]` stage's, transposed).
    mine = stage.at[:, mine_rows].transpose((1, 0)) if transpose_lhs else stage.at[mine_rows]
    if prologue is not None:
        # This warpgroup's half of the rows, the only half its MMA reads.
        depth = stage.shape[1]
        tiles_of = {i: fixed_tiles[i] if i in fixed_tiles
                    else read_tiles[i].at[pl.ds((first + index) * depth, depth)] if i in read_tiles
                    else slots[f'side_{i}'] for i in range(len(prologue_axes))}
        left = _transformed(prologue, prologue_axes, stage.at[mine_rows], tiles_of, 'rows', mine_rows)
        if summed:
            return nothing, tuple(total + family.registers_product(left, oriented[name])
                                  for total, name in zip(accumulators, matrices, strict=True))
        return nothing, tuple(family.register_products(acc, (left,), oriented[name])
                              for acc, name in zip(accumulators, matrices, strict=True))
    if summed:
        scale = None if scaling is None else scaling(index, slots)
        return nothing, tuple(family.summed_apart(total, mine, oriented[name], scale)
                              for total, name in zip(accumulators, matrices, strict=True))
    if block_scaled:
        return nothing, tuple(family.scaled_products(acc, mine, oriented[name],
                                                     (slots['a_scales'], slots['b_scales']))
                              for acc, name in zip(accumulators, matrices, strict=True))
    return nothing, tuple(family.products(acc, mine, oriented[name])
                          for acc, name in zip(accumulators, matrices, strict=True))


def _decoding(decoded: list, tiles: Tiles, wg, interpret: bool) -> "_Decoding | None":
    """Dew: consumer `wg`'s decode of a packed matrix's stages into `decoded`'s buffers (`gmm_allocations`'),
    shared or its own (`_shared_decode`); None for a matrix of no words."""
    if not decoded:
        return None
    if _shared_decode(tiles, interpret=interpret):
        return _Decoding(decoded[0], wg, tiles.warpgroups)
    return _Decoding(decoded[0].at[wg], 0, 1)


def _shared_decode(tiles: Tiles, *, interpret) -> bool:
    """Dew: whether a packed matrix's consumers share its decoded stages, each decoding its share of the
    columns (`_Decoding`): compiled (they meet at a named barrier, lane-level code the interpreter does not
    run), on two consumers or more, each share whole 64-column fragments (the decode's register layout,
    `_stage_decoded`'s). Each decoding the whole stage, an H100 ran the packed kernel 5.2-5.3x the matrix
    decoded once a call at 4096 rows (W7-gh200-8-M-NS-8-h100), its decode unhidden under the products."""
    share = tiles.block_n // tiles.warpgroups
    return not interpret and tiles.warpgroups > 1 and share % 64 == 0


@dataclasses.dataclass(frozen=True)
class _Decoding:
    """Dew: a consumer's decode of a packed matrix's stages (`_gmm_step`'s): its `buffers` (`[stages,
    columns, depth]` bfloat16), and with `consumers` sharing them, its own share of the columns (`wg`'s),
    after which they meet (`met`)."""

    buffers: Any
    wg: Any
    consumers: int

    def decode(self, index, slots, interpret: bool):
        """Step `index`'s stage of words and scales (`slots`'), this consumer's share of it, decoded into
        its buffer, which it returns."""
        if self.consumers > 1:
            # A tile's first step restarts the buffers' rotation (`index` is the tile's own): every consumer
            # has drained its last tile's products, which read them, before any decodes into them.
            pl.when(index == 0)(self.met)
        buffer = self.buffers.at[index % self.buffers.shape[0]]
        columns = buffer.shape[0] // self.consumers
        mine = pl.ds(self.wg * columns, columns) if self.consumers > 1 else pl.ds(0, buffer.shape[0])
        words = [slots[f'words_{q}'].at[mine] for q in range(buffer.shape[-1] // 128)]
        _stage_decoded(words, slots['scales'].at[mine], buffer.at[mine], interpret)
        return buffer

    def met(self) -> None:
        """The consumers' meeting once each has decoded and published its share: a named barrier of all
        their threads (`DECODE_BARRIER`), none where one consumer decodes its own."""
        if self.consumers > 1:
            threads = self.consumers * WARPGROUP_THREADS
            plgpu.inline_mgpu()(lambda ctx: mgpu_utils.inline_ptx(
                f"bar.sync {DECODE_BARRIER}, {threads};", has_side_effects=True))()


DECODE_BARRIER = 15
"""Dew: the named barrier a packed matrix's consumers meet at (`_Decoding.met`): the last of a block's 16,
past Mosaic GPU's own (0, the block's; 1 on, a warpgroup's)."""

WARPGROUP_THREADS = 128
"""Dew: a warpgroup's threads."""


def _stage_decoded(words, scales, out, interpret: bool) -> None:
    """Dew: a stage of an NVFP4 matrix (`words`, each 128 depths' `[columns, 16]` uint32 in the staged order,
    `gemm.staged_rows`; `scales` `[columns, depth / 16]` E4M3) decoded into `out` (`[columns, depth]`
    bfloat16): each 128 depths' sixteen words a column give, code by code, its eight sixteen-depth slices
    (`bits.NVFP4_BFLOAT16_FORM`'s registers, each split into its two bfloat16 lanes, `_nvfp4_lanes`), each
    times its block's scale over `bits.NVFP4_OWED`, exactly: every value of an E2M1 code times a power-of-two
    scale within bfloat16's range. Its registers lie in the warpgroup MMA's fragment layout, in which a
    column's scale is a column vector and a lane's slice a contiguous store (the strided layout found no
    assignment for this decode)."""
    layout = None if interpret else plgpu.Layout.WGMMA

    def loaded(ref):
        return ref[...] if layout is None else plgpu.load(ref, layout=layout, optimized=False)
    owed = _e4m3_values(loaded(scales)) * jnp.float32(1 / bits.NVFP4_OWED)
    which = jax.lax.broadcasted_iota(jnp.int32, owed.shape, 1)
    for group, group_words in enumerate(words):
        for code, lane in enumerate(_nvfp4_lanes(loaded(group_words))):
            # The block's scale a column: its group's code-th, picked from the loaded row by a mask (a
            # column slice of a register tile has no layout).
            scale = jnp.sum(jnp.where(which == 8 * group + code, owed, 0.0), axis=1)
            out[:, pl.ds(128 * group + 16 * code, 16)] = (lane * scale[:, None]).astype(jnp.bfloat16)


def _e4m3_values(codes: jax.Array) -> jax.Array:
    """Dew: E4M3 codes (float8_e4m3fn) as float32 by integer arithmetic, exactly: a normal code's exponent
    (rebiased, 7 to 127) and mantissa moved into an fp32 pattern, a subnormal code's mantissa times 2^-9,
    its sign kept. No FP8 conversion, which takes sm89's `cvt ... .e4m3x2`: an A100's ptxas refused the
    packed NVFP4 kernel's scales cast so (W8-a100, M-NS-8). The NaN code (no block scale holds it) reads
    as 480."""
    word = jax.lax.bitcast_convert_type(codes, jnp.uint8).astype(jnp.uint32)
    exponent, mantissa = (word >> jnp.uint32(3)) & jnp.uint32(0xF), word & jnp.uint32(0x7)
    sign = (word & jnp.uint32(0x80)) << jnp.uint32(24)
    normal = jax.lax.bitcast_convert_type(
        sign | ((exponent + jnp.uint32(120)) << jnp.uint32(23)) | (mantissa << jnp.uint32(20)), jnp.float32)
    subnormal = mantissa.astype(jnp.float32) * jnp.float32(2.0 ** -9)
    return jnp.where(exponent == 0, jnp.where(sign != 0, -subnormal, subnormal), normal)


def _nvfp4_lanes(words: jax.Array) -> list[jax.Array]:
    """Dew: the eight codes of each of `words` (uint32), each code's E2M1 value in float32, code by code
    (`bits.NVFP4_BFLOAT16_FORM`'s four bfloat16x2 registers, each split low half first): each lane's bits
    as an fp32 pattern, its exponent restored by 2^126 (the form leaves each code times 2^-126, its
    subnormal codes `0.5` as a bfloat16 subnormal, restored from its mantissa), so `0.0`, `0.5` and each
    normal code come out exact."""
    lanes = []
    for register in bits.NVFP4_BFLOAT16_FORM.traced(words):
        for half in (register << jnp.uint32(16), register & jnp.uint32(0xFFFF0000)):
            sign = half & jnp.uint32(0x80000000)
            normal = jax.lax.bitcast_convert_type(half + jnp.uint32(126 << 23), jnp.float32)
            mantissa = (half >> jnp.uint32(16)) & jnp.uint32(0x7F)
            subnormal = mantissa.astype(jnp.float32) * jnp.float32(2.0 ** -7)
            subnormal = jnp.where(sign != 0, -subnormal, subnormal)
            lanes.append(jnp.where(half & jnp.uint32(0x7F800000) == 0, subnormal, normal))
    return lanes


def _transformed(prologue, axes_of: tuple, mine, tiles_of: dict, halved: str, half) -> jax.Array:
    """Dew: `mine`, this warpgroup's half of a stage, transformed by the prologue on its side tiles
    (`tiles_of`, by operand) cut to the same half along the `halved` launch axis, a side operand by group's
    at each of its rows (`axes.by_rows`), in the stage's dtype."""
    rows = mine.shape[0]
    cut = [axes.by_rows(tiles_of[i].at[axes.placed(axes.held(side_axes), {halved: half})][...], side_axes,
                        rows) for i, side_axes in enumerate(axes_of)]
    return prologue(mine[...], *cut).astype(mine.dtype)


def _rewritten(prologue, axes_of: tuple, mine, tiles_of: dict, halved: str, half) -> None:
    """Dew: `mine` rewritten in place by the prologue (`_transformed`); its reader publishes the writes
    (`mosaic.Family.published`)."""
    mine[...] = _transformed(prologue, axes_of, mine, tiles_of, halved, half)


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
         dense: bool = False, k_major: bool = False, step_scales: tuple | None = None,
         gather: tuple | None = None) -> jax.Array:
    """Each group's rows of `lhs` `[rows, depth]` transposed times its rows of `rhs` `[rows, width]`:
    `[groups, depth, width]` in `out_dtype`, an empty group's zero. The depth must divide by
    `tiles.block_m` and the width by `tiles.block_n`, and the tile takes one multiprocessor;
    `tiles.block_k` is the rows a step takes.

    Dew: `prologue(rows, *side_tiles)` transforms each `[block_k, block_m]` stage of `lhs`'s rows (a
    cooperative tile's consumer, half its depth) in place before the group masking and the MMA; side
    operands follow 'rows' and 'depth' as `gmm`'s.
    `exact_sums` adds each step's product into the fp32 sum by an IEEE add, which holds two
    accumulators in registers. A `dense` call (one group, every row) on a family that fixes up a tile
    across programs (`balanced`) runs its output tiles' last wave balanced over the programs; with no
    prologue, its stages are read as loaded (`rows_unstaged`).
    With `k_major`, `lhs` is `[depth, rows]` and `rhs` `[width, rows]`, the rows contiguous (an FP8
    product's operands, as Hopper's FP8 tensor cores read both), each group's rows whole steps of
    `tiles.block_k` (the caller's layout: no step straddles groups, so no stage is rewritten), with no
    prologue or split; with `step_scales` besides (`row_scales` `[rows / 128, depth]`, `column_scales`
    `[rows / 128, width]`, fp32; exact sums, 128-row steps), each step's partial times its depth rows' and
    columns' scales before its add (`gmm`'s `step_scales`: a blockwise FP8 recipe's kernel gradient, row
    4518). `gather`, a table for either operand (`(lhs_table, rhs_table)`, int32 `[rows]` or None), makes
    the rows that operand's rows at the table's entries, read where they lie (`gmm`'s `gather`: a mixture of
    experts' tokens, and a scattered product's cotangent at its slots, row 1376), on a ring family's
    producer, with no prologue, rows-contiguous layout, step scales or dense call."""
    block_m, block_n, block_k = tiles.block_m, tiles.block_n, tiles.block_k
    if k_major and (prologue is not None or tiles.splits > 1 or (step_scales is not None and (
            not exact_sums or block_k != SCALE_DEPTH))):
        raise ValueError(f"a rows-contiguous kernel gradient takes no prologue or split, and its step "
                         f"scales an exact {SCALE_DEPTH}-row step, not {tiles}")
    tables = (None, None) if gather is None else tuple(gather)
    _tgmm_tables_refused(tables, tiles, family_of(tiles, lhs.dtype), prologue=prologue, dense=dense,
                         k_major=k_major, step_scaled=step_scales is not None)
    m, k = lhs.shape if k_major else lhs.shape[::-1]
    n, k2 = rhs.shape if k_major else rhs.shape[::-1]
    k, k2 = (rows if table is None else table.shape[0] for rows, table in zip((k, k2), tables, strict=True))
    g = group_sizes.shape[0]
    # Dew: a depth the tile's rows do not divide takes a partial last row tile (`ragged_rows`).
    ragged = bool(m % block_m)
    if (k != k2 or n % block_n or tiles.cluster != 1
            or (ragged and not ragged_rows(m, prologue=prologue, k_major=k_major))):
        raise ValueError(f"a [{k}, {m}] by [{k2}, {n}] kernel gradient does not take {tiles}")
    group_sizes = group_sizes.astype(jnp.int32)
    out_dtype = jnp.dtype(out_dtype)
    if tiles.splits > 1:
        # Each group's rows in `splits` runs of whole steps from the group's first block, the first and last
        # clipped to its rows: each run a group of its own, its fp32 partial summed after. The operands go
        # as they came, the run's call padding them (a padded table and its ungathered operand would meet
        # at two row counts).
        # An exact sum's runs sum exactly too, and their partials by IEEE adds in the runs' order: a
        # reassociation of IEEE sums, which the exact-sum contract (`templates.product.promoted`) allows,
        # where the tensor cores' truncating adds are what it rules out.
        sliced = _sliced(group_sizes, block_k, tiles.splits)
        # A side operand by group follows its group into each of its runs: its value repeated a run each, in
        # the runs' group-major order, so a run's group reads its own group's (`axes.at_group`, modulo the
        # side's groups, as a stacked layer's call numbers them).
        prologue_operands = tuple(
            side if axes.GROUPS not in side_axes
            else jnp.repeat(side, tiles.splits, axis=list(side_axes).index(axes.GROUPS))
            for side, side_axes in zip(prologue_operands, prologue_axes, strict=True))
        partials = tgmm(lhs, rhs, sliced, tiles=dataclasses.replace(tiles, splits=1), out_dtype=jnp.float32,
                        multiprocessors=multiprocessors, interpret=interpret, prologue=prologue,
                        prologue_operands=prologue_operands, prologue_axes=prologue_axes,
                        exact_sums=exact_sums, gather=gather)
        return partials.reshape(g, tiles.splits, m, n).sum(axis=1).astype(out_dtype)
    # Dew: the rows padded to a multiple of 8, as `gmm`'s, past every group; to whole row blocks where the
    # family's loads read past an operand unpredicated (its last group's last block). A table is padded to
    # whole row blocks always, a pad reading row 0 of its operand, which stays as it is: its producer reads
    # a step's entries by plain loads, which no TMA bounds (`_walked_rows`).
    # A rows-contiguous call's stages are read whole as ever (`_whole_stages`).
    unstaged = not k_major and rows_unstaged(dense=dense, prologue=prologue,
                                             tabled=any(table is not None for table in tables))
    # A stage read as loaded (`rows_unstaged`) zeroes no rows past the call's, so its rows are padded to whole
    # steps wherever its loads past them do not read zeros (an interpreted slice clamps).
    bounds = family_of(tiles, lhs.dtype).bounded_loads and not (unstaged and interpret)
    multiple = 8 if bounds else math.lcm(8, block_k)
    row_axis = 1 if k_major else 0
    lhs, rhs = (rows_padded(operand, row_axis, multiple) if table is None else operand
                for operand, table in zip((lhs, rhs), tables, strict=True))
    tables = tuple(None if table is None else rows_padded(table.astype(jnp.int32), 0, math.lcm(8, block_k))
                   for table in tables)
    k = -(-k // multiple) * multiple
    prologue_operands = sides_padded(prologue_operands, prologue_axes, multiple)
    bounded = family_of(tiles, lhs.dtype).bounded_loads and not interpret and tables[0] is None
    if ragged and not bounded:
        # A partial row tile's depth past the rows read as zeros: TMA's loads past an operand read them; an
        # interpreted slice clamps, and cp.async reads past it (a gathered operand's producer's too,
        # `gathered.Ring`), so their depth is padded (`ragged_rows`).
        lhs = rows_padded(lhs, 1, block_m)
    group_starts, group_ends, group_num_blocks, group_block_starts = _group_blocks(group_sizes, block_k)
    # The groups walked longest first (`_segmented`), a group's tiles their steps (ties in their order).
    order = jnp.argsort(-group_num_blocks).astype(jnp.int32)
    tail = None
    # A tile summed across programs adds their exact partials by IEEE adds (`Family.fixed_up`), as a split's.
    if dense and fixes_up(tiles, lhs.dtype):
        tail = balanced_tail(tiles, (-(-m // block_m), n // block_n), -(-k // block_k),
                             (lhs.dtype.itemsize, rhs.dtype.itemsize), generation.gpu_target(),
                             multiprocessors)
    walked = g * -(-m // block_m) * (n // block_n)
    programs = launched(family_of(tiles, lhs.dtype), multiprocessors, walked, tail)
    return _tgmm_streamed(
        (lhs, rhs, *tables), (group_sizes, group_starts, group_ends, group_num_blocks, group_block_starts,
                              order),
        (k, m, n, g), tiles=tiles, out_dtype=out_dtype, multiprocessors=programs,
        interpret=interpret, prologue=prologue, prologue_operands=prologue_operands,
        prologue_axes=prologue_axes, exact_sums=exact_sums, tail=tail, k_major=k_major,
        step_scales=step_scales, unstaged=unstaged)


def rows_unstaged(*, dense: bool, prologue, tabled: bool) -> bool:
    """Dew: whether a `tgmm` stage is read as loaded, no consumer or role rewriting it: a dense call's (one
    group, every row: no step holds a neighbouring group's rows, and its rows past the call's are zeros,
    `tgmm`'s padding), with no prologue transforming it and no table gathering its rows (`tabled`). Its step
    is then its products alone (`_gradient_unrolled`)."""
    return dense and prologue is None and not tabled


def _rewrites(family: mosaic.Family, k_major: bool, tables: tuple, unstaged: bool) -> bool:
    """Dew: whether `tgmm`'s stage is rewritten by its family's own role (`mosaic.Family.rewrites_in_persist`,
    a role beside its consumers): its loads' rows, never rows its tables gather, a k-major launch's whole
    steps of one group, nor a stage read as loaded (`rows_unstaged`; `grouped_matmul._Mosaic.rewritten`)."""
    return (family.rewrites_in_persist and not k_major and all(table is None for table in tables)
            and not unstaged)


def _gradient_unrolled(family: mosaic.Family, *, unstaged: bool, exact_sums: bool, scaled: bool) -> bool:
    """Dew: whether a `tgmm` walk takes a rotation of its ring a trip (`mosaic.stepping`, `_unrolled`'s
    reason): where its step is its products, a stage read as loaded (`rows_unstaged`) summed by the tensor
    cores or by its family's persist (`mosaic.Family.sums_in_persist`). Not a stage its consumers zero
    (`_staged_halves`), nor a step's partial summed apart or a window's (`_walked_gradient`), nor a
    step-scaled one (`_unrolled`): wgmma's exact gradient, its windows' ends known at an unrolled rotation's
    slots, spilled 1028 to 2892 bytes at 128 x 128 x 64 (six stages), 128 x 192 x 64 and 64 x 192 x 64 (four),
    dense or grouped, where walked a step a trip it spills none (the GEMM compile gate, EM-G1, EM-G1-r2)."""
    return unstaged and not scaled and (not exact_sums or family.sums_in_persist)


def _whole_stages(slots: dict, mine_depth) -> tuple:
    """Dew: a `tgmm` step's two stages read whole as its products take them: rewritten by its family's own
    role (`mine_depth` None, `_rewritten_stage`), or a k-major launch's depth rows of the rows' block and
    the cotangent's block read transposed (whole steps of one group, nothing to rewrite)."""
    if mine_depth is None:
        return slots['ring_0'].transpose((1, 0)), slots['ring_1']
    return slots['ring_0'].at[mine_depth], slots['ring_1'].transpose((1, 0))


def _stage_rewrite(rewrites: bool, family: mosaic.Family, starts_gmem, ends_gmem, block_k: int) -> dict:
    """Dew: `tgmm`'s persist argument where its family rewrites its stage (`_rewrites`): each step's
    neighbouring groups' rows zeroed by the family's own role (`_rewritten_stage`); none elsewhere."""
    if not rewrites:
        return {}
    return {'prepared': functools.partial(_rewritten_stage, family, starts_gmem, ends_gmem, block_k)}


def _rewritten_stage(family: mosaic.Family, starts_gmem, ends_gmem, block_k: int, tile, index, slots) -> None:
    """Dew: a `tgmm` step's neighbouring groups' rows zeroed in both stages, whole (`_zero_neighbours_of`),
    by its family's rewriting role (`mosaic.Family.persist`'s `prepared`), in the layouts it reads them in
    (`rewriting`)."""
    g_i, _, _, steps, start, *_ = tile
    widths = [(stage, zeroing_columns(family, stage.dtype)) for stage in (slots['ring_0'], slots['ring_1'])]
    shares = [(stage.at[:, pl.ds(column, width)], None) for stage, width in widths
              for column in range(0, stage.shape[1], width)]
    _zero_neighbours_of(family.rewriting, shares, index + start, starts_gmem[g_i], ends_gmem[g_i], steps,
                        block_k, aligned_skip=True)


def _stored_chunks(family: mosaic.Family, summed, store: Callable, place: tuple) -> None:
    """Dew: a `tgmm` tile's columns stored as its family reads them out (`chunks`: a register home's whole,
    tensor memory's a chunk at a time), each at its column (`store(value, place)`)."""
    row_start, first_row, stored, column, *ladder = place
    for start_column, (chunk,) in family.chunks((summed,)):
        at = column + start_column if start_column else column
        store(chunk, (row_start, first_row, stored, at, *ladder))


@tuning.lowered_once(('shape', 'tiles', 'out_dtype', 'multiprocessors', 'interpret', 'prologue',
                      'prologue_axes', 'exact_sums', 'tail', 'k_major', 'unstaged'))
def _tgmm_streamed(operands: tuple, groups: tuple, shape: tuple, *, tiles: Tiles, out_dtype,
                   multiprocessors: int, interpret, prologue, prologue_operands: tuple, prologue_axes: tuple,
                   exact_sums: bool, tail=None, k_major: bool = False, step_scales=None,
                   unstaged: bool = False) -> jax.Array:
    """Dew: `tgmm`'s product at a tile `tiles.warpgroups` consumer roles compute, each its share of the depth
    by all the columns, one MMA a step (two: CUTLASS's cooperative sm90 schedule), on its instruction
    family's bundle (`family_of`), its streams running across the persistent loop's tiles where the family
    streams so (the next tile's first row blocks load while the consumers store this one, which a group's
    few rows, a tile's few steps, make most of the time). Each consumer reads its depth share of the rows'
    stage, which it prepares (the prologue, the neighbouring groups' rows zeroed), and the whole
    cotangent's, a column share of which each zeroes, waiting for the other's where a group boundary falls
    in the step; an `unstaged` stage (`rows_unstaged`) is read as loaded, prepared by none and met at no
    boundary. A side operand with rows streams beside the operands; one without is loaded once a tile.
    One multiprocessor a tile, but where a dense call's last wave is balanced (`tail`): a tile then walked
    by several programs in turn, each but its first program's leaving its fp32 partial in a workspace and
    signalling a semaphore, which the first waits on and adds before storing the tile. Interpreted, one
    program walks every program's tiles in turn (the interpreter runs its programs one after another, each
    paying its own setup), a balanced tail's programs ordered so no semaphore is needed (`tail.ordered`).
    `operands` are padded, then each one's gather table (or None); `groups` `_group_blocks`' (`tgmm`)."""
    lhs, rhs, tables = operands[0], operands[1], tuple(operands[2:])
    family = family_of(tiles, lhs.dtype)
    rewrites = _rewrites(family, k_major, tables, unstaged)
    k, m, n, g = shape
    block_m, block_n, block_k = tiles.block_m, tiles.block_n, tiles.block_k
    consumers = tiles.warpgroups
    depth_half = block_m // consumers
    atom, atoms_each = _zeroed_atoms(family, rhs.dtype, block_n, consumers)
    part = (depth_half, block_n)
    grid = (-(-m // block_m), n // block_n)
    # A stream for each side operand with rows, but a vector a family does not copy, which its step reads
    # from global memory.
    stepped, read, fixed = _stepped(family, prologue_axes, 'rows')
    allocations = tgmm_allocations(tiles, family, lhs_dtype=lhs.dtype, rhs_dtype=rhs.dtype,
                                   prologue_axes=prologue_axes, prologue_dtypes=_dtypes_of(prologue_operands),
                                   out_dtype=out_dtype, k_major=k_major, step_scaled=step_scales is not None,
                                   unstaged=unstaged)
    unrolled = _gradient_unrolled(family, unstaged=unstaged, exact_sums=exact_sums,
                                  scaled=step_scales is not None)

    def body(sizes_gmem, starts_gmem, ends_gmem, blocks_gmem, block_starts_gmem, order_gmem, lhs_gmem,
             rhs_gmem, *refs):
        table_gmems, refs = _tables_placed(tables, refs)
        *side_gmems, o_gmem = refs
        side_gmems, workspace = (side_gmems, None) if tail is None else (side_gmems[:-1], side_gmems[-1])
        scale_gmems, side_gmems = _scales_apart(side_gmems, step_scales is not None)
        ready = None if tail is None else family.partials_ready((multiprocessors, consumers), bool(interpret))

        schedule = functools.partial(
            _tgmm_scheduled, groups=(sizes_gmem, blocks_gmem, block_starts_gmem, order_gmem),
            operands=(lhs_gmem, rhs_gmem, side_gmems), tail=tail, programs=multiprocessors, grid=grid,
            tiles=tiles, stepped=stepped, prologue_axes=prologue_axes, interpreted=bool(interpret),
            k_major=k_major, scale_gmems=scale_gmems, table_gmems=table_gmems)

        def scope(fixed_smems, fixed_barriers, *meeting, streams):
            halves = meeting[0] if meeting else None
            names = tuple(allocations.blocks)

            def issue(wg, tile, walk, *, active):
                g_i, depth, _, steps, start, *_ = tile
                if tail is not None and not active:
                    return None  # past its program's range: another program walks the tile
                mine_depth = pl.ds(wg * depth_half, depth_half)
                mine_columns = (wg * atoms_each, atoms_each, atoms_each * consumers > block_n // atom)
                fixed_refs = [smem.at[wg] for smem in fixed_smems]
                group = (starts_gmem[g_i], ends_gmem[g_i], steps, block_k)
                first = pl.multiple_of(block_starts_gmem[g_i], block_k) // block_k
                if active:
                    # A side operand by group at the tile's (`axes.at_group`).
                    family.loaded([(side_gmems[i].at[axes.named(prologue_axes[i], {'depth': depth} | (
                                        axes.at_group(prologue_axes[i], side_gmems[i].shape, g_i)))],
                                    ref, barrier.at[wg])
                                   for i, ref, barrier in zip(fixed, fixed_refs, fixed_barriers,
                                                              strict=True)])

                def staged_at(index, slots):
                    # The step of its group's the program walks, from its first (`start`).
                    index = index + start
                    if rewrites or k_major:
                        return _whole_stages(slots, None if rewrites else mine_depth)
                    if unstaged:
                        return slots['ring_0'].at[:, mine_depth].transpose((1, 0)), slots['ring_1']
                    # A vector read from global memory at the step's row block (`read`).
                    read_refs = {i: side_gmems[i].at[pl.ds((first + index) * block_k, block_k)] for i in read}
                    hook = (None if prologue is None
                            else (prologue, prologue_axes, stepped, fixed, fixed_refs, read_refs))
                    return _staged_halves(family, [slots[name] for name in names], (index, *group),
                                          (mine_depth, mine_columns), hook, halves)

                # An empty group's tile reads none and stores zeros.
                scaling = _tgmm_scaling(step_scales is not None, mine_depth)
                return _walked_gradient(family, walk, staged_at, part, exact_sums, bool(interpret), scaling,
                                        family.summed_window(tiles.block_k))

            finish = functools.partial(_tgmm_finished, family=family, tiles=tiles, m=m, o_gmem=o_gmem,
                                       partials=None if tail is None else (workspace, ready))
            walks = (g * grid[0] * grid[1],) if tail is None else (tail.segments, multiprocessors)
            family.persist(streams, walks, schedule, issue, finish, consumers=consumers,
                           axis="programs", outputs=allocations.outputs, interpret=bool(interpret),
                           in_flight=family.in_flight, unrolled=unrolled, accumulators=(part,),
                           **_summed(family, tiles, exact_sums),
                           **_stage_rewrite(rewrites, family, starts_gmem, ends_gmem, block_k))

        family.scoped(scope, *allocations.staged,
                      streams=family.streams(allocations.blocks, tiles.stages, consumers,
                                             swizzles=SCALE_SWIZZLES if step_scales is not None else None,
                                             **_tgmm_gathered(tables, allocations.blocks, bool(interpret))))

    out_type = jax.ShapeDtypeStruct((g, m, n), out_dtype)
    kernel = family.kernel(body, out_type=out_type,
                           grid=(1 if interpret else multiprocessors,), grid_names=("programs",),
                           consumers=consumers + rewrites, interpret=interpret, approx_math=False,
                           name="dew_tgmm",
                           resident=tiles.programs, scalars=len(groups),
                           cost=unit_cost.estimate(2 * k * m * n, lhs, rhs, *prologue_operands, out_type))
    return kernel(*groups, lhs, rhs, *(table for table in tables if table is not None), *(step_scales or ()),
                  *prologue_operands,
                  *_workspace(tail, (multiprocessors, 2, consumers, *part), interpret))


def _tgmm_finished(wg, tile: tuple, summed, store, *, active, family: mosaic.Family, tiles: Tiles, m: int,
                   o_gmem, partials) -> None:
    """Dew: consumer `wg`'s share of a `tgmm` tile stored (`mosaic.Family.persist`'s `finish`): at once, or,
    a balanced tail's tile (`partials`: its workspace and readiness flags), its program's partial and the
    other programs' added first (`mosaic.Family.fixed_up`); nothing where its program walks none of it."""
    g_i, depth, columns, _, _, gathered, program = tile
    if partials is not None and not active:
        return
    place = _tgmm_place(m, depth.start + wg * (tiles.block_m // tiles.warpgroups), columns.start, tiles)
    out_gmem = o_gmem.at[g_i]
    if partials is None:
        _stored_chunks(family, summed, lambda value, at: store(0, value, out_gmem, at), place)
        return
    workspace, ready = partials
    first, gathered = gathered
    family.fixed_up(summed, workspace.at[:, 0, wg], None if ready is None else ready.at[:, wg], program,
                    first, gathered, lambda value: store(0, value, out_gmem, place),
                    firsts=workspace.at[:, 1, wg])


def ragged_rows(m: int, *, prologue, k_major: bool) -> bool:
    """Dew: whether `tgmm` takes a depth `m` (its output's rows) that its tile's rows do not divide, its
    last row tile partial (row 2295: GPT-OSS's 2880 divides by no cooperative wgmma tile's 128 rows, so its
    kernel gradient ran one consumer's 64 x 192 tile, 1.79x Triton's time): its depth's loads past the
    rows reading zeros (or padded), each consumer storing only its rows within the depth (`_tgmm_place`),
    in whole multiples of 8 rows a store takes; no prologue (whose side operands follow the depth) and no
    rows-contiguous layout (whose depth is its operands' outer axis, the step-scaled gradient's)."""
    return prologue is None and not k_major and m % TMA_ROWS == 0


def _tgmm_place(m: int, start, column, tiles: Tiles) -> tuple:
    """Dew: where a consumer's share of a kernel gradient's tile from row `start` and `column` is stored
    (`Family.persist`'s `store`): its rows within the depth `m` (`ragged_rows`: a partial last tile's), in
    `mosaic.store_rows`' ladder of the share's rows down to their least multiple the depth leaves; the share
    whole where the tile divides the depth."""
    share = tiles.block_m // tiles.warpgroups
    if m % tiles.block_m == 0:
        return start, 0, share, column, share
    # The ladder's rungs are powers of two: the least the largest dividing every count it stores.
    least = math.gcd(m, share)
    return start, 0, jnp.clip(m - start, 0, share), column, share, least & -least


def _tables_placed(tables: tuple, refs: tuple) -> tuple:
    """Dew: a `tgmm` kernel's gather tables' references, each operand's (None where it has none), leading
    the rest, and the references after them."""
    placed = []
    for table in tables:
        placed.append(None if table is None else refs[0])
        refs = refs if table is None else refs[1:]
    return tuple(placed), refs


def _tgmm_gathered(tables: tuple, blocks: dict, interpret: bool) -> dict:
    """Dew: a gathered kernel gradient's streams (`family.streams`' arguments, `_gathered`'s for `gmm`): each
    gathered operand's ring (`ring_0` the rows', `ring_1` the cotangent's) its producer's gather, or
    interpreted each of a step's rows its copies (`mosaic.piece_copies`, at no stated swizzle)."""
    names = tuple(name for name, table in zip(('ring_0', 'ring_1'), tables, strict=True) if table is not None)
    if not names:
        return {}
    if interpret:
        return {'copies': {name: blocks[name][0][0] * mosaic.piece_copies(blocks[name][0], blocks[name][1],
                                                                          None, 1) for name in names}}
    return {'gathered': names}


def _tgmm_scheduled(index, *, groups: tuple, operands: tuple, tail: tail_of.Tail | None, programs: int,
                    grid: tuple, tiles: Tiles, stepped: tuple, prologue_axes: tuple,
                    interpreted: bool, k_major: bool = False, scale_gmems=None,
                    table_gmems: tuple = (None, None)) -> tuple:
    """Dew: a streamed kernel gradient's tile at the persistent walk's `index` (`mosaic.Family.persist`'s
    schedule): whether its program walks any of its rows, how many steps, its streams' views (step `j` its
    group's `j`-th row block from the program's first, at the tile's depth and columns), and its group,
    depth and columns, its group's steps, the program's first of them, its partials' gatherer (`walked`)
    and its program. `groups` are the group sizes', block counts', first blocks' and walking order's
    references, `operands` the rows', cotangent's and side operands', and `table_gmems` each operand's
    gather table (a gathered operand's step its rows at the table's entries, `mosaic.Rows` of one row)."""
    sizes_gmem, blocks_gmem, block_starts_gmem, order_gmem = groups
    lhs_gmem, rhs_gmem, side_gmems = operands
    index = tail_of.ordered(tail, index, programs, interpreted)
    g_i, snaked, start, count, gathered = _segmented(tail, index, programs, grid[0] * grid[1], order_gmem,
                                                     sizes_gmem.shape[0])
    m_i, n_i = tail_of.planar_snake(snaked, grid, 1, tiles.grid_block_n)
    depth, columns = pl.ds(m_i * tiles.block_m, tiles.block_m), pl.ds(n_i * tiles.block_n, tiles.block_n)
    first = pl.multiple_of(block_starts_gmem[g_i], tiles.block_k) // tiles.block_k + start
    steps = jnp.where(sizes_gmem[g_i] > 0, blocks_gmem[g_i], 0)
    walks = steps if count is None else count

    def row_block(j):
        return first + j, 0

    def rows_block(j):
        return 0, first + j

    def gathered_rows(table, block):
        return lambda j: mosaic.Rows(table, (first + j) * tiles.block_k, (block,), 1)

    views = ({'ring_0': (lhs_gmem.at[depth], rows_block), 'ring_1': (rhs_gmem.at[columns], rows_block)}
             if k_major else
             {name: (gmem.at[:, extent], row_block) if table is None else (gmem, gathered_rows(table, block))
              for name, gmem, table, extent, block in zip(('ring_0', 'ring_1'), (lhs_gmem, rhs_gmem),
                                                          table_gmems, (depth, columns), (m_i, n_i),
                                                          strict=True)})
    if scale_gmems is not None:
        # Its depth rows' scales and its columns', a step's row of each at the tile's (`step_scales`).
        views['row_scales'] = (scale_gmems[0].at[:, depth], row_block)
        views['column_scales'] = (scale_gmems[1].at[:, columns], row_block)
    views = {**views,
             **{f'ring_{2 + position}': (
                 side_gmems[i].at[axes.placed(prologue_axes[i], {'rows': axes.WHOLE}, depth)],
                 lambda j, side_axes=prologue_axes[i]: axes.placed(side_axes, {'rows': first + j}, 0))
                for position, i in enumerate(stepped)}}
    return walks > 0, walks, views, (g_i, depth, columns, steps, start, gathered, index[-1])


def _segmented(tail: tail_of.Tail | None, index, programs: int, tiles: int, order, groups: int) -> tuple:
    """Dew: a streamed kernel gradient's walk at `index`: its group, its tile's place in the snake, the
    first of the tile's steps the program walks and how many (None: its group's), and its partials'
    gatherer's end (`Tail.segment`). Where nothing balances, a tile a program a round of the programs:
    the groups' tiles longest first (`order`, a group's tiles its steps each) and every other round taken
    by the programs in reverse, so each program's tiles sum to its share of the steps however the rows are
    routed: under qwen3-30b-a3b's gate gradient, groups as skewed as a gamma of CV 0.6 ran 1.143 of a
    uniform routing's time walked group by group in index order, 1.060 longest first (row 3614,
    FS-3614-SKEW2-h100), the alternating rounds bringing the slowest program's predicted share from 1.034
    of an even split to 1.014. A group's tiles stay a run of the walk, so a wave shares its rows in the
    L2."""
    if tail is not None:
        return 0, *tail.segment(index, programs, tiles)
    (step,) = index
    rounds, place = step // programs, step % programs
    length = jnp.minimum(programs, groups * tiles - rounds * programs)
    walked = rounds * programs + jnp.where(rounds % 2 == 0, place, length - 1 - place)
    return order[walked // tiles], walked % tiles, 0, None, None


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
    atom = zeroing_columns(family, cotangent_stage.dtype)
    atoms = cotangent_stage.shape[1] // atom
    # Its depth half's swizzled rows, each its own, and its share of the cotangent's, the last consumer's
    # past the stage's end held (`atoms_each` rounds the share up).
    shares = [(rows_stage.at[:, pl.ds(mine_depth.start + column, atom)], None)
              for column in range(0, mine_depth.size, atom)]
    shares += [(cotangent_stage.at[:, pl.ds((first_atom + j) * atom, atom)],
                first_atom + j < atoms if short else None) for j in range(atoms_each)]
    boundary = _zero_neighbours_of(family, shares, *group, aligned_skip=halves is None)
    family.published()
    if halves is not None:
        pl.when(boundary)(lambda: family.meet(halves))
    return rows_stage.at[:, mine_depth].transpose((1, 0)), cotangent_stage


def _walked_gradient(family: mosaic.Family, walk, staged, shape: tuple, exact_sums: bool, interpret: bool,
                     scaling=None, window: int = 1):
    """Dew: a streamed kernel gradient tile's steps (`walk`, a tile's of `mosaic.Family.persist`;
    `staged(index, slots)` their operands), summed into an fp32 tile of `shape`: as the tensor cores add, or
    with `exact_sums` each step's product summed apart and added into the running sum by an IEEE add (a
    tensor core's accumulation truncates each add), times `scaling(index, slots)`'s multiplier where given
    (`tgmm`'s `step_scales`); with a `window` of several steps (`mosaic.Issuer.summed_window`), the
    tensor cores' sum of each window's steps added at its end, the last window's however short. A tile of
    no steps is zero."""
    if exact_sums and family.sums_in_persist:
        # Its persist sums the windows itself (`_summed`): the steps walked plainly.
        exact_sums = False
    if exact_sums and scaling is None and window > 1:
        def windowed_step(index, slots, running, accumulators):
            (fresh,) = accumulators
            lhs, rhs = staged(index, slots)
            fresh = family.products(fresh, lhs, rhs)
            # A window's end adds its sum into the running sum and starts the next from zeros: a branch on
            # the step, the warpgroup's tiles not selected elementwise by a scalar.
            return lax.cond(lax.rem(index + 1, window) == 0,
                            lambda: (running + fresh, (family.zeros(shape),)),
                            lambda: (running, (fresh,)))

        running, (fresh,) = walk(windowed_step, family.zeros(shape), (shape,))
        return running + fresh
    if exact_sums:
        def exact_step(index, slots, running, accumulators):
            lhs, rhs = staged(index, slots)
            scale = None if scaling is None else scaling(index, slots)
            return family.summed_apart(running, lhs, rhs, scale), accumulators

        running, _ = walk(exact_step, family.zeros(shape), ())
        return running

    def step(index, slots, nothing, accumulators):
        (acc,) = accumulators
        lhs, rhs = staged(index, slots)
        return nothing, (family.products(acc, lhs, rhs),)

    _, (summed,) = walk(step, None, (shape,))
    return summed


def _zero_neighbours_of(family: mosaic.Issuer, refs: list, step, start, end, steps, block_k: int, *,
                        aligned_skip: bool = False):
    """Dew: in a group's first and last row blocks, zero each of `refs` (`[block_k, columns]` stages, each
    one swizzled row wide, so one holds a thread's few registers whatever the tile's width; each with
    whether it is held, None where always) on the rows a neighbouring group holds, by selection; whether
    `step` is either block. With `aligned_skip` (a one-consumer tile), a block the group starts or ends at
    a step's edge holds no neighbour's rows and is left as loaded: GPT-OSS-120B's 256-row groups at 64-row
    steps zeroed two of their four (3.18 ms against 2.52, DBG2295-BAND1). Two consumers zero as before:
    their zeroing is where they meet, and skipping it at aligned edges ran slower on DeepSeek-V3's and Kimi
    Linear's kernel gradients (DBG2295-FACTOR1), for a reason not established."""
    first, last = step == 0, step == steps - 1
    if aligned_skip:
        first = first & (lax.rem(start, block_k) != 0)
        last = last & (lax.rem(end, block_k) != 0)

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


def step_seconds(tiles: Tiles, target) -> float | None:
    """One MMA step of `tiles`, in seconds: its FLOPs at one multiprocessor's 16-bit tensor rate
    (`templates.cost.rates`, the architecture's a clock at the part's clock, whatever the multiprocessors a
    target states), the unit the kernel gradient's split (`split_count`) and a balanced tail
    (`handoff_steps`) price their walks in; None where the target states no rate."""
    rate = unit_cost.rates(target, device=False)[0]
    return None if rate is None else 2 * tiles.block_m * tiles.block_n * tiles.block_k / rate


def fixes_up(tiles: Tiles, dtype) -> bool:
    """Whether a tile's steps may be walked by several programs at `tiles`, each leaving its partial and the
    last to arrive adding them (`mosaic.Family.fixed_up`): its family's, where its consumers hold the tile's
    accumulator in registers, which `fixed_up` leaves and adds whole; not tcgen05's, whose accumulator stays
    in tensor memory (`Issuer.tensor_columns`), read out a chunk of columns at a time."""
    family = family_of(tiles, dtype)
    return family.fixes_up and not family.tensor_columns(tiles.block_n, 1)


def balanced_tail(tiles: Tiles, grid: tuple[int, int], steps: int, itemsizes: tuple[int, int], target,
                  programs: int) -> tail_of.Tail | None:
    """A dense product's last wave balanced over `programs` (`tail.balanced`, the launch's: its
    multiprocessors times `tiles.programs`) at `tiles` over its output tiles' `grid` and `steps` a tile,
    priced from the target's facts: its MMA step (`step_seconds`), its
    memory's rate and L2 (`GpuTarget.memory_rate`, `l2_bytes`), its operands' blocks a step (`itemsizes`,
    the left's and the right's) and its hand-off (`handoff_steps`); walked where the target states no
    rate."""
    step, rate = step_seconds(tiles, target), target.memory_rate
    if step is None or rate is None:
        return None
    left, right = itemsizes
    traffic = tail_of.Traffic(grid, tiles.grid_block_n, tiles.block_m * tiles.block_k * left,
                              tiles.block_n * tiles.block_k * right, step,
                              target.multiprocessors or programs, rate, target.l2_bytes)
    return tail_of.balanced(grid[0] * grid[1], steps, programs, traffic,
                            handoff_steps=handoff_steps(tiles, target))


def handoff_steps(tiles: Tiles, target) -> float:
    """A balanced tail's hand-off of a partial between programs (`GpuTarget.handoff_seconds`) in MMA steps
    of `tiles` (`step_seconds`), as `tail.balanced` prices it; none where the part's hand-off or its rate is
    unmeasured."""
    step, handoff = step_seconds(tiles, target), target.handoff_seconds
    return 0.0 if step is None or handoff is None else handoff / step


def split_count(tiles: Tiles, *, groups: int, rows: int, depth: int, width: int, itemsizes: tuple[int, int],
                multiprocessors: int, step_seconds: float | None, bandwidth: float | None,
                launch_seconds: float | None, partial_rows: int | None = None) -> int:
    """Dew: the slices a contraction of `rows` splits into at `tiles` (`Tiles.splits`: a kernel gradient's,
    `tgmm`, and a swapped decode product's, `gmm`), derived from the call's extents and the device's facts
    in MMA steps (`step_seconds`), nothing enumerated. Over a wave of P programs (`multiprocessors` x
    `tiles.programs`) and `rows` in steps a group (on average), the busiest program walks ceil(O s / P) of
    the O output tiles at ceil(steps / s) steps each at s slices, and streams as many shares of the operands
    (`rows` of `depth` and of `width` at `itemsizes` bytes, each read once) at its share of `bandwidth`
    (that of the multiprocessors the programs keep busy, over the programs in flight): the walk is the
    longer of the two, W unsplit. A split adds its fp32 partials' 2s + 1 passes over the output
    (`partial_rows` of it where they hold fewer than its `groups` x `depth`: a decode step's experts' few
    rows), p steps each at `bandwidth`, and the launch their sum adds (`launch_seconds`, the part's measured
    launch in a program, row 3613; none where unmeasured). The count fills k whole waves, s = kP / O, at the
    k where the last wave's idling and the partials' passes cost least (below), each slice keeping at least
    the ring's stages and one more steps and its runs sharing the steps whole (`split_runs`, as `gmm`'s runs
    share them); it splits where the walk it saves outweighs its partials and their launch, and is one where
    it does not, where the device states no rate, or where the call has no whole tile or no step. Key's
    2560 x 1024 gradient over 24,576 rows on an H100 splits in three, 2 x 128 steps against 384. Its steps
    priced at their MMA alone, a decode product's few tiles walked unsplit: an A100's 2048 x 2048 input
    gradient at 1-32 rows ran 12 programs on 108 multiprocessors, 2.8-3.0x XLA's 128 (M-NS-12, row 483),
    where its matrix's stream is the walk."""
    wave = multiprocessors * tiles.programs
    output = groups * -(-depth // tiles.block_m) * (width // tiles.block_n)
    steps = -(-rows // max(groups, 1) // tiles.block_k)
    if step_seconds is None or bandwidth is None or output == 0 or steps == 0:
        return 1
    operand_bytes = rows * (depth * itemsizes[0] + width * itemsizes[1])
    held = groups * depth if partial_rows is None else partial_rows
    passes = held * width * 4 / bandwidth / step_seconds

    def walk(splits: int) -> float:
        programs = output * splits
        waves = -(-programs // wave)
        busiest = waves * -(-steps // splits)
        # A program's share of the operands, at its share of the memory's rate.
        rate = bandwidth * min(programs, multiprocessors) / multiprocessors / min(programs, wave)
        stream = waves * operand_bytes / programs / rate / step_seconds
        return max(busiest, stream)

    # At s = kP / O slices, k whole waves, the busiest program walks the wave's share of every tile's steps,
    # W O / P, and its last wave idles half a tile's slices on average, W O^2 / 2kP^2; the partials' passes
    # add 2ps = 2pkP / O. Their sum is least at k = (O / P)^1.5 sqrt(W / 4p), and k holds two slices of every
    # tile at least; the slices fill every multiprocessor, so the operands stream at the whole rate too.
    # Each slice keeps at least the ring's stages and one more steps (`deepest` slices): where that bounds
    # the count, the most whole waves within it, or the bound itself where a wave holds two slices of none.
    waves = max(-(-2 * output // wave), round((output / wave) ** 1.5 * math.sqrt(walk(1) / (4 * passes))))
    deepest = steps // (tiles.stages + 1)
    splits = waves * wave // output
    if splits > deepest:
        within = deepest * output // wave * wave // output
        splits = within if within >= 2 else deepest
    if splits < 2:
        return 1
    # Its runs share the steps whole: as many runs as a share's ceil(steps / s) steps cover.
    splits = -(-steps // -(-steps // splits))
    launch = 0.0 if launch_seconds is None else launch_seconds / step_seconds
    saved = walk(1) - walk(splits)
    return splits if saved > (2 * splits + 1) * passes + launch else 1


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
    """Each group's first row, end, count of `block_k`-row blocks, and its first block's first row, from the
    groups' one prefix sum (`_prefix_sums`, as `tile_groups` reads them)."""
    group_sizes = group_sizes.astype(jnp.int32)
    group_ends = _prefix_sums(group_sizes)
    group_starts = group_ends - group_sizes
    group_block_starts = group_starts // block_k * block_k
    group_block_ends = -(group_ends // -block_k) * block_k
    return group_starts, group_ends, (group_block_ends - group_block_starts) // block_k, group_block_starts


def row_extent(rows: int, row_tile: int, unit: int = TMA_ROWS) -> int:
    """Dew: the rows `gmm`'s tile schedule reads of `rows` rows (a multiple of `unit`, `row_unit`) at tiles
    of `row_tile` rows: a tile starts at most `unit` before the rows end (`tile_groups`) and reads
    `row_tile` rows on."""
    return rows + row_tile - unit

