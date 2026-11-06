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

from dew.algebra import bits
from dew.nn.kernels import generation, tuning
from dew.nn.kernels.templates import axes, bundles, cost as unit_cost, mosaic, resources, tail as tail_of
from dew.nn.kernels.templates.gemm import rows_extended, rows_padded, sides_extended, sides_padded


def family_of(tiles: "Tiles", dtype) -> mosaic.Family:
    """The bundle a tile runs on: the one it names (`Tiles.family`, `bundles.named`), for a consumer's share
    of the tile (`mosaic.Issuer.specialized`)."""
    return bundles.named(tiles.family, dtype).specialized(tiles.block_m // tiles.warpgroups, tiles.block_n)


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
        paired = self.cluster == 2 and self.warpgroups == 1
        if self.warpgroups not in (1, 2) or self.cluster not in (1, 2) or paired:
            raise ValueError(f"{self}: one warpgroup alone, or two in a cluster of one or two "
                             "multiprocessors")
        family = bundles.named(self.family, jnp.bfloat16)
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


def derived(layouts: tuple, *, rows: int, depth: int | None, width: int, dtype, multiprocessors: int, target,
            candidate, launch_of, grouped: bool, depths: tuple[int, ...] | None = None) -> Tiles | None:
    """The one tile a GEMM mainloop runs on an instruction family's `layouts` (`bundles.layouts`, its
    bundles in preference order, which share its facts but their loaders and registers) for a call whose
    output has `rows` rows (a group's,
    on average, where `grouped`: a product's rows split among its groups) and `width` columns and whose
    steps run over `depth` (None: a group's rows, padded to whole steps), from the family's facts, the
    call's extents and the device's multiprocessors alone (no table, no ranking;
    docs/design/kernel-templates.md §7); `depths`, where given, the steps the launch's stage takes in place
    of the family's (a packed matrix's decode, `NVFP4_STEP`). `candidate(tile)` is the launch's legality apart
    from its
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
    - Consumers: of the counts the family takes (`Family.consumers`), a warpgroup each, the one whose tile
      reads its operands least (`_reads`: a share's fragment loads where the products read registers, then
      the tile's reads a FLOP), the most on a tie; one where rows a tile must divide (not grouped) take no
      whole role for each, or where a group averages fewer rows than one role (grouped).
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
    # Its widths in whole rows of the widest swizzle, as well as its column unit, where the call's width has
    # one: the objective below prices reads, not a narrower staging swizzle's, and an A100's 160-wide tiles
    # (64-byte swizzle) ran its forwards 1.13-1.24x their 128-wide time where the reads favoured them
    # (M-FS-1, W3-a100); a narrower call (a swapped decode product's few rows) at its column unit.
    widest = min(width, family.widest_product or width)
    unit = family.column_unit(dtype)
    swizzled = math.lcm(unit, mosaic.SWIZZLE_BYTES // jnp.dtype(dtype).itemsize)
    columns = ([block_n for block_n in range(widest - widest % swizzled, 0, -swizzled)
                if width % block_n == 0]
               or [block_n for block_n in range(widest - widest % unit, 0, -unit) if width % block_n == 0])
    steps = [step for step in depths or family.depths(dtype) if depth is None or depth % step == 0]
    busy = 1 + family.loads_ahead + family.in_flight

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
    # does a call no cooperative tile holds (a wide epilogue's outputs past a block), and a grouped call
    # whose groups average fewer rows than one role (a decode step's experts, a few routed rows each), whose
    # further consumers would compute padding alone; where groups average a role's rows, a routing's groups
    # past it keep the further consumers' rows real. Row 3622, H100 (FS-3622-AB1/AB2-h100): one consumer ran
    # 0.73 to 0.999 of the cooperative tile's time at 1 to 32 rows a group, and 1.13 to 1.26 of it at 64.
    whole = rows >= family.role_rows if grouped else rows % (family.role_rows * family.consumers) == 0
    # Each consumer count's tile at the most programs that hold one, the count the objective prices least
    # (`_reads`), the most consumers on a tie.
    counted = [next((tile for programs in range(family.programs(consumers), 0, -1)
                     if (tile := tiled(consumers, programs)) is not None), None)
               for consumers in range(family.consumers if whole else 1, 0, -1)]
    return min((tile for tile in counted if tile is not None), key=functools.partial(_reads, family),
               default=None)


def _reads(family: mosaic.Family, tile: "Tiles") -> tuple[float, float]:
    """Dew: `derived`'s objective over its consumer counts' tiles: the operand reads a FLOP each consumer's
    share issues from shared memory (1/rows + 1/columns of its share), where its products read registers
    its warps load (`mosaic.Issuer.operands_in_registers`: mma.sync's fragments, each an instruction its
    warps issue beside their products), then the tile's own reads a FLOP (1/rows + 1/columns), which a
    wider tile lowers. On an A100, two consumers' 64-row shares of a 128 x 128 tile issued 1.5x one
    consumer's fragment loads (176M against 118M LDSM, 0.0234 against 0.0156 reads a FLOP) and ran 2.78x
    its input gradient's time (NS-P2-NCU2-a100); their 128 x 256 tiles 1.355x one consumer's 128 x 128
    over 19 rows (NS-P2-CONSUMERS-a100). Warpgroup MMA reads both operands from shared memory itself, no
    instruction its warps issue: there the cooperative tile's fewer reads lead (CUTLASS's split, 2-7% faster
    on every H100 scoreboard row)."""
    share = 1 / (tile.block_m // tile.warpgroups) + 1 / tile.block_n
    return share if family.operands_in_registers else 0.0, 1 / tile.block_m + 1 / tile.block_n


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
    matrix: jax.Array | None = None
    """Dew: the matrix of a stack the group reads, where a table's fifth column states it (`gmm`'s
    `first`); None where the group reads its own."""

    @classmethod
    def create(cls, table, tile, tid):
        """Dew: the group info of row tile `tid`, read from `tile_groups`' table (its group, its first row,
        its group's first and past-last row, and where `gmm` states one, its matrix of a stack) rather than
        walked over every group's length in each tile."""
        tile = jnp.int32(tile)
        group, block_start, group_start, group_end = (table[tid, column] for column in range(4))
        # Mosaic's copies and stores read a slice's start as a multiple of the rows its tiling takes.
        block_start = pl.multiple_of(block_start, TMA_ROWS)
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
    ends = _prefix_sums(lengths)
    starts = ends - lengths
    rows = jnp.int32(tile)
    first_rows = starts // TMA_ROWS * TMA_ROWS
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

    def scope(self, family: mosaic.Family, tiles: "Tiles", steps: int | None = None) -> resources.Scope:
        """Its allocations by lifetime (`resources.shared_bytes`): its streams where its family places them
        (`mosaic.Family.lifetimes`, in no more slots than a walk's `steps` where they are known), and each
        output staged as its family stages it (`mosaic.Family.output_staging`, the allocations `persist`
        makes), after the walks where the family scopes them apart, its pipeline's memory reused."""
        consumers = tiles.warpgroups
        staging = tuple(family.output_staging(self.outputs, consumers))
        persisted, addresses = family.persist_scratch(consumers)
        multicast = ('matrix',) if tiles.cluster > 1 and 'matrix' in self.blocks else ()
        streams = family.streams(self.blocks, tiles.stages, consumers, multicast=multicast)
        held, walks = family.lifetimes(streams, steps)
        if walks:
            return resources.Scope(program=(*held, self.staged, persisted), walks=walks,
                                   after=tuple((buffer.shape, buffer.dtype) for buffer in staging),
                                   addresses=addresses)
        return resources.Scope(program=(*held, self.staged, persisted, staging), addresses=addresses)


def _typed(dtype) -> jax.ShapeDtypeStruct:
    """A stand-in for an operand of `dtype`, as `mosaic.Transport.staging` reads an operand's dtype."""
    return jax.ShapeDtypeStruct((), dtype)


def gmm_allocations(tiles: "Tiles", family: mosaic.Family, *, dtype, rhs_dtype, transpose_rhs: bool,
                    prologue_axes: tuple, prologue_dtypes: tuple, epilogue_axes: tuple,
                    epilogue_dtypes: tuple, dtypes: tuple, paired: bool = False,
                    transpose_lhs: bool = False, packed: bool = False, step_scaled: bool = False
                    ) -> Allocations:
    """Dew: `gmm`'s allocations at `tiles` (`Allocations`), for rows of `dtype` and a matrix of `rhs_dtype`
    (two with `paired`), its side operands' axes and dtypes and its outputs' dtypes: a stream for the rows,
    each matrix and each prologue side operand that follows the steps (`_stepped`), each epilogue side
    operand's staging at a consumer's share of the tile, each prologue side operand without depth's at the
    tile's rows; a `transpose_lhs` call's rows stream their `[depth, rows]` blocks. A `packed` matrix (an
    NVFP4 one's, `scales`' in `gmm`) streams each 128 depths' words (`words_{q}`, a column's sixteen a
    64-byte row, which cp.async's two-dimensional copies take and a register tile loads whole) and its block
    scales K-major, and each consumer
    decodes a stage into bfloat16 buffers of its own (`_stage_decoded`), one more than its family keeps
    product groups in flight (`mosaic.Family.in_flight`), so no decode overwrites a stage a product still
    reads."""
    block_m, block_n, block_k = tiles.block_m, tiles.block_n, tiles.block_k
    part = (block_m // tiles.warpgroups, block_n)
    stepped, _, fixed = _stepped(family, prologue_axes, 'depth')
    extent = {'rows': block_m, 'depth': block_k}
    rhs_block = (block_n, block_k) if transpose_rhs else (block_k, block_n)
    lhs_block = (block_k, block_m) if transpose_lhs else (block_m, block_k)
    matrix = ({**{f'words_{q}': ((block_n, 16), jnp.uint32) for q in range(block_k // 128)},
               'scales': ((block_n, block_k // 16), jnp.float8_e4m3fn)} if packed
              else {'matrix': (rhs_block, rhs_dtype)})
    blocks = {'rows': (lhs_block, dtype), **matrix,
              **({'paired': (rhs_block, rhs_dtype)} if paired else {}),
              **({'row_scales': ((1, block_m), jnp.float32)} if step_scaled else {}),
              **{f'side_{i}': (axes.named(prologue_axes[i], extent), prologue_dtypes[i]) for i in stepped}}
    sides = family.staging('side', [(axes.named(axes.held(side_axes), {'rows': part[0], 'columns': part[1]}),
                                     _typed(side))
                                    for side_axes, side in zip(epilogue_axes, epilogue_dtypes, strict=True)])
    held = family.staging('fixed', [(axes.named(prologue_axes[i], extent), _typed(prologue_dtypes[i]))
                                    for i in fixed])
    decoded = ([plgpu.SMEM((tiles.warpgroups, family.in_flight + 1, block_n, block_k), jnp.bfloat16)]
               if packed else [])
    return Allocations(blocks, (sides['side_smems'], sides['side_barriers'], held['fixed_smems'],
                                held['fixed_barriers'], decoded),
                       tuple((part, jnp.dtype(out)) for out in dtypes))


def tgmm_allocations(tiles: "Tiles", family: mosaic.Family, *, lhs_dtype, rhs_dtype,
                     prologue_axes: tuple, prologue_dtypes: tuple, out_dtype, k_major: bool = False,
                     step_scaled: bool = False) -> Allocations:
    """Dew: `tgmm`'s allocations at `tiles` (`Allocations`): a ring for the rows, the cotangent and each
    prologue side operand with rows that follows the steps (`_stepped`), each side operand without rows'
    staging, the consumers' meeting, and its output's share of the tile."""
    block_m, block_n, block_k = tiles.block_m, tiles.block_n, tiles.block_k
    stepped, _, fixed = _stepped(family, prologue_axes, 'rows')
    extent = {'rows': block_k, 'depth': block_m}
    # A rows-contiguous call's blocks lie depth (or width) by rows, the rows their minor axis.
    blocks = {'ring_0': ((block_m, block_k) if k_major else (block_k, block_m), lhs_dtype),
              'ring_1': ((block_n, block_k) if k_major else (block_k, block_n), rhs_dtype),
              **({'row_scales': ((1, block_m), jnp.float32)} if step_scaled else {}),
              **{f'ring_{2 + position}': (axes.named(prologue_axes[i], extent), prologue_dtypes[i])
                 for position, i in enumerate(stepped)}}
    held = family.staging('fixed', [(axes.named(prologue_axes[i], extent), _typed(prologue_dtypes[i]))
                                    for i in fixed])
    return Allocations(blocks, (held['fixed_smems'], held['fixed_barriers'],
                                family.meeting(tiles.warpgroups)),
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
        paired: jax.Array | None = None, exact_sums: bool = False,
        transpose_lhs: bool = False, first: jax.Array | None = None,
        scales: jax.Array | None = None, step_scales: tuple | None = None
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
    with no hooks, pair, packed matrix or split.

Dew: `prologue(tile, *side_tiles)` transforms each `[block_m, block_k]` tile of `lhs` (a cooperative
    tile's consumer, half its rows) in its shared memory stage before the warpgroup MMA reads it;
    `epilogue(acc, *side_tiles)` each fp32 accumulator tile (a cooperative consumer's, half its rows),
    returning one output or a tuple of them (`out_dtypes`). A side operand's dimensions follow the launch
    axes its `*_axes` entry names, 'rows' and 'depth' for the prologue's, 'rows' and 'columns' for the
    epilogue's, each brought to shared memory by TMA."""
    block_n, block_k = tiles.block_n, tiles.block_k
    if transpose_lhs and (not dense or prologue is not None):
        raise ValueError("a transposed lhs is a dense call's without a prologue")
    k, rows = lhs.shape if transpose_lhs else lhs.shape[::-1]
    matrices, k2, n = rhs.shape
    # A dense call's one matrix is its one group, its sizes unread (a TPU core's dense launch passes none).
    g = matrices if group_sizes is None else group_sizes.shape[0]
    if transpose_rhs:
        k2, n = n, k2
    # A packed matrix's words hold eight depths each.
    k2 = k2 if scales is None else 8 * k2
    if scales is not None:
        _packed_taken(rhs, scales, tiles, transpose_rhs=transpose_rhs, hooked=(prologue, epilogue, paired),
                      transpose_lhs=transpose_lhs, depth=k2, width=n)
    _gmm_refused(rhs, tiles, (rows, k, matrices, k2, n), dense=dense, first=first, paired=paired)
    accumulators = 1 if paired is None else 2
    dtypes = tuple(jnp.dtype(dtype) for dtype in (out_dtypes or (out_dtype,)))
    if k == 0:
        outs = _of_no_depth((rows, n), group_sizes, accumulators, epilogue, epilogue_operands, epilogue_axes,
                            dtypes)
        return outs if out_dtypes else outs[0]
    plain = all(each is None for each in (prologue, epilogue, paired, scales)) and not transpose_lhs
    _gmm_split_refused(tiles, k, n, step_scales, dense=dense, exact_sums=exact_sums,
                       hooked=prologue is not None or epilogue is not None or paired is not None, plain=plain)
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
    row_axis = 1 if transpose_lhs else 0
    lhs = rows_padded(lhs, row_axis, TMA_ROWS)
    m = lhs.shape[row_axis]
    row_tile = tiles.block_m * tiles.cluster
    if dense:
        table = None
    else:
        lengths = jnp.concatenate([group_sizes.astype(jnp.int32),
                                   jnp.maximum(m - jnp.sum(group_sizes, dtype=jnp.int32), 0)[None]])
        table = tile_groups(lengths, row_tile, pl.cdiv(m, row_tile) + g)
        if first is not None:
            # Each tile's matrix, its group's of the stack from `first` (`GroupInfo.matrix`): the group itself
            # stays the call's, which its validity and its side operands read.
            table = jnp.concatenate([table, jnp.minimum(table[:, :1], g - 1) + first], axis=1)
    extent = (m if family_of(tiles, lhs.dtype).bounded_loads and not interpret
              else -(-m // row_tile) * row_tile if dense else row_extent(m, row_tile))
    lhs = rows_extended(lhs, row_axis, extent)
    if step_scales is not None:
        step_scales = (rows_extended(rows_padded(step_scales[0], 1, TMA_ROWS), 1, extent), step_scales[1])
    prologue_operands = sides_extended(prologue_operands, prologue_axes, extent)
    epilogue_operands = sides_extended(epilogue_operands, epilogue_axes, extent)
    # A dense product's last wave balanced over the programs (`tail.balanced`), as the kernel gradient's: one
    # matrix, one accumulator, a tile a program.
    # An exact sum's order is its contract, which a tile summed across programs reassociates.
    tail = None
    if (dense and paired is None and tiles.cluster == 1 and family_of(tiles, lhs.dtype).fixes_up
            and not exact_sums):
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
                         step_scales=step_scales)
    if tiles.splits > 1:
        # Each run's fp32 partial, summed in its order (the same sums every call) and rounded once.
        total = functools.reduce(jnp.add, tuple(outs[0]))
        outs = tuple(total.astype(dtype) for dtype in dtypes)
    outs = tuple(out[:rows] for out in outs)
    return outs if out_dtypes else outs[0]


STEP_SCALE_DEPTH = 128
"""The depth one step scale covers (`gmm`'s `step_scales`), its step's: `grouped_matmul.SCALE_DEPTH`."""

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
                      'dense', 'tail', 'exact_sums', 'transpose_lhs'))
def _gmm_mainloop(lhs, rhs, table, shape: tuple, *, tiles: Tiles, transpose_rhs: bool, dtypes: tuple,
                  multiprocessors: int, interpret, hooks: tuple, sides: tuple, dense: bool = False,
                  transpose_lhs: bool = False,
                  paired: jax.Array | None = None, tail: tail_of.Tail | None = None,
                  exact_sums: bool = False, scales: jax.Array | None = None,
                  step_scales: tuple | None = None) -> tuple:
    """Dew: `gmm` on its instruction family's bundle (`family_of`): `tiles.warpgroups` consumer roles, each
    its share of the rows by all the columns, one MMA a step (two: CUTLASS's cooperative split, 2-7% faster
    on every H100 scoreboard row than halving the columns), persistent over the tiles in a planar snake,
    the family owning the operands' streams across tiles, their staging and the outputs' stores
    (`mosaic.Family.persist`). Each consumer's prologue rewrites its share of the rows' slot, its epilogue
    takes its share of each side tile. `lhs` is padded; `table` is `tile_groups`' at the pair's rows (`gmm`).
    One multiprocessor a tile or, at `tiles.cluster` 2, a pair of them a cluster running two row tiles of
    one group, which load the group's matrix block once between them (`mosaic.Family.streams`' multicast);
    both run each pair's steps (or neither), its second's rows past the group stored nowhere. A `paired`
    matrix streams beside the first, its blocks multiplied by the same rows' into a second accumulator.
    With a balanced `tail` (a dense product's), the tiles past the whole waves have their steps shared out
    over the programs, each tile's partials added by the program that walks its first step (`tail.Tail`,
    `mosaic.Family.fixed_up`), which runs the epilogue and stores it.
    Lowered once a configuration (`tuning.lowered_once`), however many layers call it."""
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
    allocations = _gmm_allocated(tiles, family, (lhs, rhs, paired, scales, step_scales), hooks, sides, dtypes,
                                 transpose_rhs=transpose_rhs, transpose_lhs=transpose_lhs)
    matrices = ('matrix',) if paired is None else ('matrix', 'paired')
    # The streams that follow the matrix's group and column tile: its own, and a packed one's scales.
    streamed = matrices + (() if scales is None else ('scales',))
    steps = k // block_k
    # A pair's tiles: g groups' boundaries, and the unrouted rows.
    grid = (pl.cdiv(m, block_m * pair) + (0 if dense else g), n // block_n)

    def body(*gmems):
        table_gmem, gmems = (None, gmems) if dense else (gmems[0], gmems[1:])
        lhs_gmem, *matrix_gmems = gmems[:1 + len(streamed)]
        refs = gmems[1 + len(streamed):]
        scale_gmems, refs = (refs[:2], refs[2:]) if step_scales is not None else ((None, None), refs)
        prologue_gmems, side_gmems, o_gmems = _split(refs, len(prologue_operands), len(epilogue_operands))
        o_gmems, workspace = (o_gmems, None) if tail is None else (o_gmems[:-1], o_gmems[-1])
        ready = None if tail is None else family.partials_ready((multiprocessors, consumers), bool(interpret))
        blocks = allocations.blocks

        schedule = functools.partial(
            _gmm_scheduled, table_gmem=table_gmem, lhs_gmem=lhs_gmem, splits=tiles.splits,
            matrix_gmems=dict(zip(streamed, matrix_gmems, strict=True)), prologue_gmems=prologue_gmems,
            shape=(m, steps, g), grid=grid, tiles=tiles, dense=dense, tail=tail, programs=multiprocessors,
            transpose_rhs=transpose_rhs, axes_of={i: prologue_axes[i] for i in stepped},
            interpreted=bool(interpret), transpose_lhs=transpose_lhs, traced=exact_sums,
            row_scales_gmem=scale_gmems[0])

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
                                         read_tiles={i: prologue_gmems[i] for i in read},
                                         mine_rows=pl.ds(wg * share, share), transpose_rhs=transpose_rhs,
                                         matrices=matrices, interpret=bool(interpret), first=start,
                                         transpose_lhs=transpose_lhs,
                                         decoded=decoded[0].at[wg] if decoded else None,
                                         scaling=_gmm_scaling(scale_gmems[1], (n, g, block_n, share),
                                                              group, ni, wg, start))
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
                           accumulators=(part,) * len(matrices))

        family.scoped(scope, *allocations.staged,
                      streams=family.streams(blocks, tiles.stages, consumers,
                                             multicast=matrices if pair > 1 else (),
                                             swizzles={'row_scales': mosaic.UNSWIZZLED}
                                             if step_scales is not None else None))

    # A depth split's outputs: each run's partial, a `[m, n]` slice of its own.
    out_type = tuple(jax.ShapeDtypeStruct((m, n) if tiles.splits == 1 else (tiles.splits, m, n), dtype)
                     for dtype in dtypes)
    # A balanced tail's partials: each program's, a consumer's share of a tile each.
    workspace = () if tail is None else (
        jax.ShapeDtypeStruct((multiprocessors, consumers, *part), jnp.float32),)
    operands = (lhs, rhs) if paired is None else (lhs, rhs, paired)
    operands = operands if scales is None else (*operands, scales)
    operands = operands if step_scales is None else (*operands, *step_scales)
    kernel = family.kernel(body, out_type=(*out_type, *workspace),
                           grid=(multiprocessors // pair,), grid_names=("programs",), consumers=consumers,
                           interpret=interpret, approx_math=False, name="dew_gmm", cluster=pair,
                           resident=tiles.programs, scalars=0 if dense else 1,
                           cost=unit_cost.estimate(2 * m * n * k * len(matrices), *operands,
                                                   *prologue_operands, *epilogue_operands, *out_type))
    outs = kernel(*(operands if dense else (table, *operands)), *prologue_operands, *epilogue_operands)
    return outs[:len(out_type)] if tail is not None else outs


def _gmm_finished(accs: tuple, tile: tuple, wg, store, *, family: mosaic.Family, part: tuple, m: int, g: int,
                  dense: bool, row_tile: int, epilogue, o_gmems: tuple, sides: tuple, partials) -> None:
    """Dew: a consumer's share of a `gmm` tile stored (`_finished`) from its accumulators: at once, or,
    a balanced tail's tile (`partials`: its workspace and readiness flags), once its program's partial and
    the other programs' are added (`mosaic.Family.fixed_up`)."""
    group, row_start, ni, start, gathered, program, split = tile
    # A depth split's run stores its partial in its own slice of each output.
    o_gmems = o_gmems if split is None else tuple(o_gmem.at[split] for o_gmem in o_gmems)
    # This consumer's share of the tile: its first row, and its group's rows there.
    part_row, column = row_start + wg * part[0], ni * part[1]
    first, stored = _stored_rows(group, part_row, part[0], m, dense=dense, row_tile=row_tile)
    finished = functools.partial(
        _finished, family=family, wg=wg, epilogue=epilogue, o_gmems=o_gmems, store=store, sides=sides,
        origins={'rows': pl.ds(part_row, part[0]), 'columns': pl.ds(column, part[1])},
        group=jnp.minimum(group.group_id, g - 1),
        place=(part_row, first, stored, column, min(part[0], m), TMA_ROWS if dense else 1))
    if partials is None or partials[0] is None:
        finished(*accs)
        return
    workspace, ready = partials
    family.fixed_up(accs[0], workspace.at[:, wg], None if ready is None else ready.at[:, wg], program, start,
                    gathered, finished)


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
            axes.by_rows(smem.at[wg].at[axes.placed(axes.held(side_axes), columns)][...], side_axes, rows)
            for _, side_axes, smem, _ in sides))
        values = values if isinstance(values, tuple) else (values,)
        for output, (o_gmem, value) in enumerate(zip(o_gmems, values, strict=True)):
            store(output, value, o_gmem, (row_start, first, stored, column + start, *ladder))


def _gmm_scheduled(index, *, table_gmem, lhs_gmem, matrix_gmems: dict, prologue_gmems, shape: tuple,
                   grid: tuple, tiles: Tiles, dense: bool, tail: tail_of.Tail | None, programs: int,
                   transpose_rhs: bool, axes_of: dict, interpreted: bool,
                   transpose_lhs: bool = False, splits: int = 1, traced: bool = False,
                   row_scales_gmem=None) -> tuple:
    """Dew: `gmm`'s tile at the persistent walk's `index` (`mosaic.Family.persist`'s schedule): whether it
    multiplies any row, its steps, its streams' views (`_gmm_views`) and its group, row block and column
    tile: its pair's but for this multiprocessor's row block; in a balanced tail, the first of the tile's
    steps its program walks and how many, its partials' gatherer and its program (`tail.Tail.segment`);
    with `splits`, its depth's run: the walk every tile's first run, then every tile's second (as many
    launches of a run each would walk them), each run a share of the steps, the last's the rest; `traced`,
    its steps a count the compiler cannot see (`mosaic.opaque`), which a walk takes as a loop's bound (an
    exactly summed walk's: cp.async's pipeline unrolled a static walk's drain, each drain step's fresh
    product its own, and at five stages the A100 forward's 128 x 128 x 64 tile spilled 40 bytes beside the
    running sum; an opaque count's drain, a loop, spilled none, and a merely traced constant left ptxas
    spilling 32 bytes at the input gradient's 64 x 160 x 32 at five stages, NSDBG-5, NSDBG-DR1).
    `shape` is the padded rows, the steps and the groups."""
    m, steps, groups = shape
    pair = tiles.cluster
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
             else GroupInfo.create(table_gmem, tiles.block_m * pair, mi))
    # This multiprocessor's row tile of its pair, from its group's first: a multiple of the rows a TMA
    # descriptor tiles by, as Mosaic's layout inference reads a slice's start.
    row_start = pl.multiple_of(group.block_start + mosaic.cluster_index(pair) * tiles.block_m, TMA_ROWS)
    views = _gmm_views(lhs_gmem, matrix_gmems, prologue_gmems, group, row_start, ni, tiles=tiles,
                       groups=groups, transpose_rhs=transpose_rhs, axes_of=axes_of, first=start,
                       transpose_lhs=transpose_lhs, row_scales_gmem=row_scales_gmem)
    active = (count > 0 if tail is not None else jnp.ones((), jnp.bool_) if dense
              else (group.group_id < groups) & (group.actual_size > 0))
    return active, count, views, (group, row_start, ni, start, gathered, index[-1], split)


def _gmm_split_refused(tiles: Tiles, k: int, n: int, step_scales, *, dense: bool, exact_sums: bool,
                       hooked: bool, plain: bool) -> None:
    """Dew: `gmm`'s refusal of a depth split but a dense product's of no hooks, pair or exact sums, each run
    some steps, and of step scales but on an exactly summed product of no hook, pair, packed matrix or
    split (`_step_scaled_takes`)."""
    if tiles.splits > 1 and (not dense or hooked or exact_sums or not split_runs(k // tiles.block_k,
                                                                                  tiles.splits)):
        raise ValueError(f"a depth split is a dense product's of no hooks, pair or exact sums, each run some "
                         f"steps, not {tiles} of a {k}-deep one")
    if step_scales is not None and not _step_scaled_takes(step_scales, tiles, k, n, exact_sums=exact_sums,
                                                          plain=plain):
        raise ValueError(f"a step-scaled product sums each {STEP_SCALE_DEPTH}-deep step exactly with no "
                         f"hook, pair, packed matrix or split, its scales covering its steps and columns, "
                         f"not {tiles}")


def _gmm_allocated(tiles: Tiles, family: mosaic.Family, operands: tuple, hooks: tuple, sides: tuple,
                   dtypes: tuple, *, transpose_rhs: bool, transpose_lhs: bool) -> Allocations:
    """Dew: `gmm_allocations` for `_gmm_mainloop`'s call: its `operands` (rows, matrix, paired matrix, packed
    scales, step scales), its hooks' axes and side operands' dtypes, and its outputs' `dtypes`."""
    lhs, rhs, paired, scales, step_scales = operands
    _, prologue_axes, _, epilogue_axes = hooks
    prologue_operands, epilogue_operands = sides
    return gmm_allocations(tiles, family, dtype=lhs.dtype, rhs_dtype=rhs.dtype, transpose_rhs=transpose_rhs,
                           prologue_axes=prologue_axes,
                           prologue_dtypes=tuple(side.dtype for side in prologue_operands),
                           epilogue_axes=epilogue_axes,
                           epilogue_dtypes=tuple(side.dtype for side in epilogue_operands), dtypes=dtypes,
                           paired=paired is not None, transpose_lhs=transpose_lhs, packed=scales is not None,
                           step_scaled=step_scales is not None)


def _gmm_refused(rhs, tiles: Tiles, extents: tuple, *, dense: bool, first, paired) -> None:
    """Dew: `gmm`'s refusal of a call its launch does not take: a dense product of several matrices, a stack
    read from its first matrix but a grouped product's of one, a paired matrix not its product's shape, or
    extents (`rows`, `depth`, `matrices`, the matrix's depth, `width`) its tile does not divide."""
    rows, k, matrices, k2, n = extents
    if dense and matrices != 1:
        raise ValueError("a dense product takes one matrix")
    if first is not None and (dense or paired is not None):
        raise ValueError("a stack read from its first matrix is a grouped product's of one matrix")
    if paired is not None and paired.shape != rhs.shape:
        raise ValueError(f"a paired matrix {paired.shape} is not its product's {rhs.shape}")
    if k != k2 or k % tiles.block_k or n % tiles.block_n:
        raise ValueError(f"a [{rows}, {k}] x [{matrices}, {k2}, {n}] product does not take {tiles}")


def _step_scaled_takes(step_scales: tuple, tiles: Tiles, depth: int, width: int, *, exact_sums: bool,
                       plain: bool) -> bool:
    """Dew: whether `gmm` takes `step_scales` at `tiles` (`_step_scaling`): an exactly summed product of no
    hook, pair, packed matrix, transposed rows or split, a step 128 deep, a row scale each step, and a
    column scale each column or each multiple of the tile's width."""
    rows, columns = step_scales
    per = width // columns.shape[-1] if columns.shape[-1] and width % columns.shape[-1] == 0 else 0
    return (exact_sums and plain and tiles.splits == 1 and tiles.block_k == STEP_SCALE_DEPTH
            and rows.shape[0] * STEP_SCALE_DEPTH == depth
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


def _zeroed_atoms(dtype, block_n: int, consumers: int) -> tuple[int, int]:
    """Dew: the columns of a swizzled row of the cotangent's stage and how many of them each consumer zeroes
    (`_zeroed_share`): whole swizzled rows, as many each as the stage's divide among them, the last's share
    short where they do not divide evenly."""
    atom = mosaic.swizzle_columns(dtype, mosaic.SWIZZLE_BYTES)
    return atom, -(-(block_n // atom) // consumers)


def _dtypes_of(arrays) -> tuple:
    """Dew: each of `arrays`' dtype."""
    return tuple(array.dtype for array in arrays)


def _scales_apart(refs: list, scaled: bool) -> tuple:
    """Dew: a step-scaled kernel's two scale references, which lead its side operands' (`tgmm`'s
    `step_scales`), and the rest; none and all of them where it has no step scales."""
    return (tuple(refs[:2]), refs[2:]) if scaled else ((None, None), refs)


def _tgmm_scaling(column_scales, column, mine_depth, first) -> Callable | None:
    """Dew: a step-scaled `tgmm` tile's `_step_scaling` (`tgmm`'s `step_scales`): its depth rows' scales
    streamed, its columns' from `column` read from global memory, a scale a column, a step's from `first`;
    none where the call has no step scales (`column_scales` None)."""
    if column_scales is None:
        return None
    return functools.partial(_step_scaling, column_scales=column_scales, per=1, matrix=None, column=column,
                             mine_rows=mine_depth, first=first)


def _gmm_scaling(column_scales, extents: tuple, group, column_tile, wg, start) -> Callable | None:
    """Dew: a step-scaled `gmm` tile's `_step_scaling` for consumer `wg` (its share of the rows), none where
    the call has no step scales (`column_scales` None): the tile's matrix's column scales from its column
    tile, a scale every `width / scales` columns. `extents` are the width, the groups, `block_n` and a
    consumer's share of the rows."""
    if column_scales is None:
        return None
    width, groups, block_n, share = extents
    return functools.partial(_step_scaling, column_scales=column_scales, per=width // column_scales.shape[-1],
                             matrix=_matrix_of(group, groups), column=column_tile * block_n,
                             mine_rows=pl.ds(wg * share, share), first=start)


def _step_scaling(index, slots, *, column_scales, per: int, matrix, column, mine_rows, first) -> Callable:
    """Dew: a step-scaled `gmm` step's multiplier of its exactly summed partial (`gmm`'s `step_scales`,
    `mosaic.Issuer.summed_apart`'s `scale`): this consumer's rows' scales at step `first + index` (the step's
    row of them streamed, `row_scales`) times the tile's matrix's scales of its columns from `column`
    (`column_scales`, read from global memory, a scale every `per` columns: one a column, or one the tile's
    whole columns, `per` a multiple of its width)."""
    rows = slots['row_scales'].at[0, mine_rows][...]
    step = first + index

    def scale(start: int, columns: int):
        if per == 1:
            # A kernel gradient's are its rows' blocks' alone (`tgmm`'s `step_scales`), no matrix's.
            at = (step,) if matrix is None else (matrix, step)
            scales = column_scales.at[(*at, pl.ds(column + start, columns))][...]
            return rows[:, None] * scales[None, :]
        return rows[:, None] * column_scales[matrix, step, lax.div(column + start, per)]

    return scale


def _gmm_views(lhs_gmem, matrix_gmems: dict, prologue_gmems, group, row_start, column_tile, *, tiles: Tiles,
               groups: int, transpose_rhs: bool, axes_of: dict, first=0, transpose_lhs: bool = False,
               row_scales_gmem=None) -> dict:
    """Dew: `gmm`'s streams' views at a tile (`group`'s rows from `row_start`, `column_tile`;
    `mosaic.viewed`): its rows', its group's matrix's (each of `matrix_gmems`, by stream: a paired
    product's two), and each prologue side operand's that follows the depth (`axes_of`, by operand), step
    `j` each one's `first + j`-th depth block (`first` a balanced tail's program's first step); a
    `transpose_lhs` call's rows as `[depth, rows]` blocks."""
    rows = pl.ds(row_start, tiles.block_m)
    lhs_view = ((lhs_gmem.at[:, rows], lambda j: (first + j, 0)) if transpose_lhs
                else (lhs_gmem.at[rows], lambda j: (0, first + j)))
    rhs_index = _rhs_index(transpose_rhs)
    matrix = _matrix_of(group, groups)
    matrices = {name: (_matrix_columns(gmem.at[matrix], column_tile,
                                       block_n=tiles.block_n, transpose_rhs=transpose_rhs),
                       lambda j: rhs_index(first + j)) for name, gmem in matrix_gmems.items()}
    if 'scales' in matrix_gmems:
        # A packed matrix's words, a stream for each 128 depths of a step: step j's group q, sixteen words a
        # column at the tile's columns.
        groups_of = tiles.block_k // 128
        words = matrices.pop('matrix')[0]
        matrices |= {f'words_{q}': (words, lambda j, q=q: (0, groups_of * (first + j) + q))
                     for q in range(groups_of)}
    if row_scales_gmem is not None:
        # Its rows' scales, a step's row of them (`step_scales`, `[depth / 128, rows]`).
        matrices['row_scales'] = (row_scales_gmem.at[:, rows], lambda j: (first + j, 0))
    return {'rows': lhs_view, **matrices,
            **{f'side_{i}': (prologue_gmems[i].at[axes.placed(side_axes, {'rows': rows})],
                             lambda j, side_axes=side_axes: axes.placed(side_axes, {'rows': 0}, first + j))
               for i, side_axes in axes_of.items()}}


def _walked_products(family: mosaic.Family, walk, step, shape: tuple, count: int, exact_sums: bool) -> tuple:
    """Dew: a cooperative `gmm` consumer's `count` products over its steps (`walk`, `step` a `_gmm_step`
    whose operands it reads), each into an fp32 tile of `shape`: as the tensor cores add, or with
    `exact_sums` each step's product summed apart and added into the running sum by an IEEE add, as
    `_walked_gradient`'s: an FP8 product's running sum promoted into fp32 every step, Hopper's FP8 tensor
    cores summing within one in a narrower accumulator (testing.md, FP8 products)."""
    if not exact_sums:
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
              decoded=None, scaling=None):
    """Dew: a consumer's step of the cooperative `gmm` (`mosaic.Family.walk`): its half of the rows' slot,
    transformed by the prologue on its side tiles (the streamed ones' slots, the fixed ones' `fixed_tiles`,
    a depth vector the family does not copy read at step `index` from global memory: `read_tiles`),
    times each of `matrices`' slot into its accumulator (a paired product's two); with `summed`, each
    product summed apart into its accumulator, the running sum (`_walked_products`' exact sums,
    `mosaic.Issuer.summed_apart`). The prologue's
    value stays in registers, where the products take it as their left operand (warpgroup MMA's register
    operand A, mma.sync's fragments): rewritten into the stage and read back, the gated prologue of a
    Llama-3-8B gate_up ran at 8.3x its plain product on an H100 (row 1336). With `decoded` (this consumer's
    buffers, a packed matrix's), the step decodes its stage of words and scales into its buffer
    (`_stage_decoded`), published before its products read it as the matrix's stage."""
    stage = slots['rows']
    read = {name: slots[name] for name in matrices if decoded is None}
    if decoded is not None:
        read['matrix'] = decoded.at[index % decoded.shape[0]]
        words = [slots[f'words_{q}'] for q in range(decoded.shape[-1] // 128)]
        _stage_decoded(words, slots['scales'], read['matrix'], interpret)
        family.published()
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
            return nothing, tuple(total + family.registers_product(left, oriented[name], interpret)
                                  for total, name in zip(accumulators, matrices, strict=True))
        return nothing, tuple(family.register_products(acc, (left,), oriented[name], interpret)
                              for acc, name in zip(accumulators, matrices, strict=True))
    if summed:
        scale = None if scaling is None else scaling(index, slots)
        return nothing, tuple(family.summed_apart(total, mine, oriented[name], interpret, scale)
                              for total, name in zip(accumulators, matrices, strict=True))
    return nothing, tuple(family.products(acc, mine, oriented[name], interpret)
                          for acc, name in zip(accumulators, matrices, strict=True))


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
    (`tiles_of`, by operand) cut to the same half along the `halved` launch axis, in the stage's dtype."""
    cut = [tiles_of[i].at[axes.placed(side_axes, {halved: half})][...] for i, side_axes in enumerate(axes_of)]
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
         dense: bool = False, k_major: bool = False, step_scales: tuple | None = None) -> jax.Array:
    """Each group's rows of `lhs` `[rows, depth]` transposed times its rows of `rhs` `[rows, width]`:
    `[groups, depth, width]` in `out_dtype`, an empty group's zero. The depth must divide by
    `tiles.block_m` and the width by `tiles.block_n`, and the tile takes one multiprocessor;
    `tiles.block_k` is the rows a step takes.

    Dew: `prologue(rows, *side_tiles)` transforms each `[block_k, block_m]` stage of `lhs`'s rows (a
    cooperative tile's consumer, half its depth) in place before the group masking and the MMA; side
    operands follow 'rows' and 'depth' as `gmm`'s.
    `exact_sums` adds each step's product into the fp32 sum by an IEEE add, which holds two
    accumulators in registers. A `dense` call (one group, every row) on a family that fixes up a tile
    across programs (`balanced`) runs its output tiles' last wave balanced over the programs.
    With `k_major`, `lhs` is `[depth, rows]` and `rhs` `[width, rows]`, the rows contiguous (an FP8
    product's operands, as Hopper's FP8 tensor cores read both), each group's rows whole steps of
    `tiles.block_k` (the caller's layout: no step straddles groups, so no stage is rewritten), with no
    prologue or split; with `step_scales` besides (`row_scales` `[rows / 128, depth]`, `column_scales`
    `[rows / 128, width]`, fp32; exact sums, 128-row steps), each step's partial times its depth rows' and
    columns' scales before its add (`gmm`'s `step_scales`: a blockwise FP8 recipe's kernel gradient, row
    4518)."""
    block_m, block_n, block_k = tiles.block_m, tiles.block_n, tiles.block_k
    if k_major and (prologue is not None or tiles.splits > 1 or (step_scales is not None and (
            not exact_sums or block_k != STEP_SCALE_DEPTH))):
        raise ValueError(f"a rows-contiguous kernel gradient takes no prologue or split, and its step "
                         f"scales an exact {STEP_SCALE_DEPTH}-row step, not {tiles}")
    m, k = lhs.shape if k_major else lhs.shape[::-1]
    n, k2 = rhs.shape if k_major else rhs.shape[::-1]
    g = group_sizes.shape[0]
    if k != k2 or m % block_m or n % block_n or tiles.cluster != 1:
        raise ValueError(f"a [{k}, {m}] by [{k2}, {n}] kernel gradient does not take {tiles}")
    # Dew: the rows padded to a multiple of 8, as `gmm`'s, past every group; to whole row blocks where the
    # family's loads read past an operand unpredicated (its last group's last block).
    multiple = 8 if family_of(tiles, lhs.dtype).bounded_loads else math.lcm(8, block_k)
    row_axis = 1 if k_major else 0
    lhs, rhs = rows_padded(lhs, row_axis, multiple), rows_padded(rhs, row_axis, multiple)
    k = lhs.shape[row_axis]
    prologue_operands = sides_padded(prologue_operands, prologue_axes, multiple)

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
    # The groups walked longest first (`_segmented`), a group's tiles their steps (ties in their order).
    order = jnp.argsort(-group_num_blocks).astype(jnp.int32)
    tail = None
    # An exact sum's order is its contract, which a tile summed across programs reassociates.
    if dense and family_of(tiles, lhs.dtype).fixes_up and not exact_sums:
        tail = balanced_tail(tiles, (m // block_m, n // block_n), -(-k // block_k),
                             (lhs.dtype.itemsize, rhs.dtype.itemsize), generation.gpu_target(),
                             multiprocessors)
    walked = g * (m // block_m) * (n // block_n)
    programs = launched(family_of(tiles, lhs.dtype), multiprocessors, walked, tail)
    return _tgmm_streamed(
        (lhs, rhs), (group_sizes, group_starts, group_ends, group_num_blocks, group_block_starts, order),
        (k, m, n, g), tiles=tiles, out_dtype=out_dtype, multiprocessors=programs,
        interpret=interpret, prologue=prologue, prologue_operands=prologue_operands,
        prologue_axes=prologue_axes, exact_sums=exact_sums, tail=tail, k_major=k_major,
        step_scales=step_scales)


@tuning.lowered_once(('shape', 'tiles', 'out_dtype', 'multiprocessors', 'interpret', 'prologue',
                      'prologue_axes', 'exact_sums', 'tail', 'k_major'))
def _tgmm_streamed(operands: tuple, groups: tuple, shape: tuple, *, tiles: Tiles, out_dtype,
                      multiprocessors: int, interpret, prologue, prologue_operands: tuple,
                      prologue_axes: tuple, exact_sums: bool,
                      tail: tail_of.Tail | None = None, k_major: bool = False,
                      step_scales: tuple | None = None) -> jax.Array:
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
    atom, atoms_each = _zeroed_atoms(rhs.dtype, block_n, consumers)
    part = (depth_half, block_n)
    grid = (m // block_m, n // block_n)
    # A stream for each side operand with rows, but a vector a family does not copy, which its step reads
    # from global memory.
    stepped, read, fixed = _stepped(family, prologue_axes, 'rows')
    allocations = tgmm_allocations(tiles, family, lhs_dtype=lhs.dtype, rhs_dtype=rhs.dtype,
                                   prologue_axes=prologue_axes, prologue_dtypes=_dtypes_of(prologue_operands),
                                   out_dtype=out_dtype, k_major=k_major, step_scaled=step_scales is not None)

    def body(sizes_gmem, starts_gmem, ends_gmem, blocks_gmem, block_starts_gmem, order_gmem, lhs_gmem,
             rhs_gmem, *refs):
        *side_gmems, o_gmem = refs if tail is None else refs[:-1]
        scale_gmems, side_gmems = _scales_apart(side_gmems, step_scales is not None)
        ready = None if tail is None else family.partials_ready((multiprocessors, consumers), bool(interpret))

        schedule = functools.partial(
            _tgmm_scheduled, groups=(sizes_gmem, blocks_gmem, block_starts_gmem, order_gmem),
            operands=(lhs_gmem, rhs_gmem, side_gmems), tail=tail, programs=multiprocessors, grid=grid,
            tiles=tiles, stepped=stepped, prologue_axes=prologue_axes, interpreted=bool(interpret),
            k_major=k_major, row_scales_gmem=scale_gmems[0])

        def scope(fixed_smems, fixed_barriers, halves, *, streams):
            names = tuple(allocations.blocks)

            def issue(wg, tile, walk, *, active):
                g_i, depth, columns, steps, start, *_ = tile
                if tail is not None and not active:
                    return None  # past its program's range: another program walks the tile
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
                    if k_major:
                        # Its depth rows of the rows' block and the cotangent's block read transposed: whole
                        # steps of one group, nothing to rewrite.
                        return slots['ring_0'].at[mine_depth], slots['ring_1'].transpose((1, 0))
                    # A vector read from global memory at the step's row block (`read`).
                    read_refs = {i: side_gmems[i].at[pl.ds((first + index) * block_k, block_k)] for i in read}
                    hook = (None if prologue is None
                            else (prologue, prologue_axes, stepped, fixed, fixed_refs, read_refs))
                    return _staged_halves(family, [slots[name] for name in names], (index, *group),
                                          (mine_depth, mine_columns), hook,
                                          halves if consumers == 2 else None)

                # An empty group's tile reads none and stores zeros.
                scaling = _tgmm_scaling(scale_gmems[1], columns.start, mine_depth, start + first)
                return _walked_gradient(family, walk, staged_at, part, exact_sums, bool(interpret), scaling)

            def finish(wg, tile, summed, store, *, active):
                g_i, depth, columns, _, start, gathered, program = tile
                if tail is not None and not active:
                    return
                place = (depth.start + wg * depth_half, 0, depth_half, columns.start, depth_half)
                if tail is None:
                    store(0, summed, o_gmem.at[g_i], place)
                    return
                family.fixed_up(summed, refs[-1].at[:, wg], None if ready is None else ready.at[:, wg],
                                program, start, gathered,
                          lambda value: store(0, value, o_gmem.at[g_i], place))

            walks = (g * grid[0] * grid[1],) if tail is None else (tail.segments, multiprocessors)
            family.persist(streams, walks, schedule, issue, finish, consumers=consumers,
                           axis="programs", outputs=allocations.outputs, interpret=bool(interpret),
                           in_flight=family.in_flight, accumulators=(part,))

        family.scoped(scope, *allocations.staged,
                      streams=family.streams(allocations.blocks, tiles.stages, consumers,
                                             swizzles={'row_scales': mosaic.UNSWIZZLED}
                                             if step_scales is not None else None))

    out_type = jax.ShapeDtypeStruct((g, m, n), out_dtype)
    # A balanced tail's partials: each program's, a consumer's share of a tile each.
    workspace = () if tail is None else (
        jax.ShapeDtypeStruct((multiprocessors, consumers, *part), jnp.float32),)
    kernel = family.kernel(body, out_type=(out_type, *workspace),
                           grid=(1 if interpret else multiprocessors,), grid_names=("programs",),
                           consumers=consumers,
                           interpret=interpret, approx_math=False, name="dew_tgmm",
                           resident=tiles.programs, scalars=len(groups),
                           cost=unit_cost.estimate(2 * lhs.shape[0] * m * n, lhs, rhs, *prologue_operands,
                                                   out_type))
    return kernel(*groups, lhs, rhs, *(step_scales or ()), *prologue_operands)[0]


def _tgmm_scheduled(index, *, groups: tuple, operands: tuple, tail: tail_of.Tail | None, programs: int,
                    grid: tuple, tiles: Tiles, stepped: tuple, prologue_axes: tuple,
                    interpreted: bool, k_major: bool = False, row_scales_gmem=None) -> tuple:
    """Dew: a streamed kernel gradient's tile at the persistent walk's `index` (`mosaic.Family.persist`'s
    schedule): whether its program walks any of its rows, how many steps, its streams' views (step `j` its
    group's `j`-th row block from the program's first, at the tile's depth and columns), and its group,
    depth and columns, its group's steps, the program's first of them, its partials' gatherer (`walked`)
    and its program. `groups` are the group sizes', block counts', first blocks' and walking order's
    references and `operands` the rows', cotangent's and side operands'."""
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

    views = ({'ring_0': (lhs_gmem.at[depth], rows_block), 'ring_1': (rhs_gmem.at[columns], rows_block)}
             if k_major else
             {'ring_0': (lhs_gmem.at[:, depth], row_block), 'ring_1': (rhs_gmem.at[:, columns], row_block)})
    if row_scales_gmem is not None:
        views['row_scales'] = (row_scales_gmem.at[:, depth], row_block)
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
    atom = mosaic.swizzle_columns(cotangent_stage.dtype, mosaic.SWIZZLE_BYTES)
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
                     scaling=None):
    """Dew: a streamed kernel gradient tile's steps (`walk`, a tile's of `mosaic.Family.persist`;
    `staged(index, slots)` their operands), summed into an fp32 tile of `shape`: as the tensor cores add, or
    with `exact_sums` each step's product summed apart and added into the running sum by an IEEE add (a
    tensor core's accumulation truncates each add), times `scaling(index, slots)`'s multiplier where given
    (`tgmm`'s `step_scales`). A tile of no steps is zero."""
    if exact_sums:
        def exact_step(index, slots, running, accumulators):
            lhs, rhs = staged(index, slots)
            scale = None if scaling is None else scaling(index, slots)
            return family.summed_apart(running, lhs, rhs, interpret, scale), accumulators

        running, _ = walk(exact_step, family.zeros(shape), ())
        return running

    def step(index, slots, nothing, accumulators):
        (acc,) = accumulators
        lhs, rhs = staged(index, slots)
        return nothing, (family.products(acc, lhs, rhs, interpret),)

    _, (summed,) = walk(step, None, (shape,))
    return summed


def _zero_neighbours_of(family: mosaic.Family, refs: list, step, start, end, steps, block_k: int, *,
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


def split_count(tiles: Tiles, *, groups: int, rows: int, depth: int, width: int, multiprocessors: int,
                step_seconds: float | None, bandwidth: float | None, launch_seconds: float | None) -> int:
    """Dew: the slices a kernel gradient's contraction splits into at `tiles` (`Tiles.splits`, `tgmm`),
    from the call's extents and the device's facts. Over a wave (`multiprocessors` x `tiles.programs`)
    and `rows` in steps a group (on average), the busiest program walks ceil(tiles x s / wave) output
    tiles of ceil(steps / s) steps at s slices. The split is the least s whose walk, with the time its
    fp32 partials take written and read back (2s + 1 passes over the output at `bandwidth`, in steps of
    `step_seconds`) and the launch their sum adds (`launch_seconds`, the part's measured launch in a
    program, row 3613; none where unmeasured), is shorter than the unsplit walk, each slice keeping at
    least the ring's stages and one more; one where none is, or where the device states no rate. A bounded
    comparison over the facts, no table, no timing: Key's 2560 x 1024 gradient over 24,576 rows on an H100
    splits in three, 2 x 128 steps against 384."""
    if step_seconds is None or bandwidth is None:
        return 1
    wave = multiprocessors * tiles.programs
    output = groups * (depth // tiles.block_m) * (width // tiles.block_n)
    steps = -(-rows // max(groups, 1) // tiles.block_k)

    def walk(splits: int) -> float:
        passes = 0 if splits == 1 else 2 * splits + 1
        partials = passes * groups * depth * width * 4 / bandwidth / step_seconds
        # A split runs its partials' launch and XLA's sum of them where the unsplit call runs one launch.
        launches = 0 if splits == 1 or launch_seconds is None else launch_seconds / step_seconds
        return -(-output * splits // wave) * -(-steps // splits) + partials + launches

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
    """Each group's first row, end, count of `block_k`-row blocks, and its first block's first row, from the
    groups' one prefix sum (`_prefix_sums`, as `tile_groups` reads them)."""
    group_sizes = group_sizes.astype(jnp.int32)
    group_ends = _prefix_sums(group_sizes)
    group_starts = group_ends - group_sizes
    group_block_starts = group_starts // block_k * block_k
    group_block_ends = -(group_ends // -block_k) * block_k
    return group_starts, group_ends, (group_block_ends - group_block_starts) // block_k, group_block_starts


def row_extent(rows: int, row_tile: int) -> int:
    """Dew: the rows `gmm`'s tile schedule reads of `rows` rows (a multiple of `TMA_ROWS`) at tiles of
    `row_tile` rows: a tile starts at most `TMA_ROWS` before the rows end (`tile_groups`) and reads
    `row_tile` rows on."""
    return rows + row_tile - TMA_ROWS

