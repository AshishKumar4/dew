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

"""Dew: the one tile a GEMM mainloop runs (`derived`): its shape, consumers, programs, step and stages from
the instruction family's facts, the device's and the call's extents alone (docs/design/kernel-templates.md
§7), for `mosaic_gpu.gmm` and `tgmm`."""

import dataclasses
import functools
import math
from collections.abc import Callable

from jax import numpy as jnp

from dew.nn.kernels.templates import axes, mosaic, resources, stores, swizzling
from dew.nn.kernels.templates.gemm import pricing
from dew.nn.kernels.templates.gemm.tiling import ACCUMULATOR_COLUMNS, CONSUMERS, Tiles


def derived(layouts: tuple, *, rows: int, depth: int | None, width: int, dtype, multiprocessors: int, target,
            candidate, launch_of, grouped: bool, depths: tuple[int, ...] | None = None, ragged: bool = False,
            units: tuple[int, int, int] = (1, 1, 1), dense: bool = False,
            programs: int | None = None, outputs: tuple = (), hooked: bool = False) -> Tiles | None:
    """The one tile a GEMM mainloop runs on an instruction family's `layouts` (`bundles.layouts`, its bundles
    in preference order, which share its facts but their loaders and registers) for a call whose output has
    `rows` rows (a group's, on average, where `grouped`: a product's rows split among its groups) and `width`
    columns and whose steps run over `depth` (None: a group's rows, padded to whole steps), from the family's
    facts, the call's extents and the device's multiprocessors alone (no table, no ranking;
    docs/design/kernel-templates.md §7); `depths`, where given, the steps the launch's stage takes in place of
    the family's (a packed matrix's decode, `NVFP4_STEP`); `ragged`, rows a tile need not divide (a kernel
    gradient's whose last row tile is partial, `tgmm`); `units`, what the rows, the columns and the step are
    whole multiples of besides (a call's side vectors' copy units, `mosaic.Transport.vector_unit`); `dense`, a
    call whose rows are one group's (`gmm`'s `dense`), its walk banded as a dense walk's (`resources.band`).
    `candidate(tile)` is the launch's legality apart from its resources (its granularity, its blocks dividing
    the call, its side operands' copies), and `launch_of(tile)` the launch whose ring and residency the
    planner sizes (`resources.plan`: its allocations a stage, its registers within a consumer's) on `target`;
    `outputs`, the dtypes its tiles store, where the derivation states their staging (`_ring_staged`);
    `hooked`, whether a column-local epilogue's hook computes them, its tile read out a chunk at a time
    (`hooks.ColumnLocal`: its side tiles and outputs a chunk's beside the ring).

    - Layout: at each shape and step, the first layout whose planner holds it (`resources.placed`): a
      producer's where its consumers' registers hold the tile, else one whose first consumer issues the
      loads, its consumers holding the program's entry registers (`fed.ConsumerFed`).
    - Shape: the balanced tile within its consumers' registers: an accumulator of `resources.held_area`
      elements A reads its operands 1/rows + 1/columns times a FLOP, least at rows = columns = sqrt(A), which
      whole roles and the family's widths come near rather than reach. So the rows are sqrt(A) in whole roles
      (no more than the call's, below; `_balanced_side`; a grouped call's weigh its groups' padding,
      `_padded_side`); the columns the widest dividing the call's width in its column unit, at most the
      family's one product's (`Family.widest_product`), that A holds at those rows; the rows then the most its
      registers hold at those columns (`resources.held_rows`). mma.sync's one consumer so takes 128 x 128
      where it took 64 x 256, whose four warps, splitting its rows, each read the columns' block whole: 1.7x
      the shared memory a FLOP (SHAPE1-a100: 0.887 of its time). A tile its registers or a block cannot stage
      busy at its balanced rows (`holds`, `resources.tile_registers`) gives way to the next narrower one, and
      where none does, to shorter rows.
    - Consumers: the most the family takes (`Family.consumers`, at most `CONSUMERS`), a warpgroup each,
      whose cooperative tile holds each consumer's registers, so its balanced side and its reads a FLOP are
      at least one consumer's; one where rows a tile must divide (not grouped) take no whole role for each,
      where a group averages fewer rows than one role (grouped), or where no cooperative tile holds.
    - Programs: the most a multiprocessor runs at once that the family's registers hold at those
      consumers (`Family.programs`) and whose share of its shared memory (`resources.program_shared`)
      stages the balanced tile busy (`tiled`'s balanced pass); where no count does, the most whose share
      holds a tile, one at least; a wave is all of them. Fewer programs before a smaller tile: programs
      share the multiprocessor's tensor rate (`pricing.SplitWalk.walk`), so a second one adds overlap, never a
      product's rate, where a halved share halves the tile and doubles its reads a FLOP: an RTX PRO
      6000's two programs a multiprocessor held 64 x 64 tiles at two stages in their 49 KB shares, its
      dense products 1.99x (forward), 3.00x (input gradient) and 1.72x (kernel gradient) cuBLASLt's time,
      geomean of 13 projections at 8192 tokens (M-FS-9, receipts dew-g4-measure2), where the balanced
      128 x 64 tile stages busy in that share at 32-deep steps (`MmaIssuer.depths`).
    - Rows: no more than the call's rows rounded up to a role's or, where grouped, half a group's in
      whole roles, at least one role (each group takes a partial tile more, so at half its rows that tile
      adds at most a third of its work).
    - Step: the family's widest depth dividing the call's (`Family.depths`) at those rows, a shallower
      one before fewer rows (a family whose products read registers holds a step's fragments).
    - Stages: the deepest its program's share of shared memory holds at those programs (`resources.plan`, the
      planner every template's derivation reads), at least those the walk keeps busy (those loading ahead,
      `Family.loads_ahead`, beside the one consumed and those whose products the family keeps in flight,
      `Family.in_flight`); a tile that cannot stage that deep gives way to the next narrower or shorter one,
      or a shallower step. A ring its consumers feed held at that least depth is deepened where a smaller
      output staging leaves it room (`_ring_staged`).
    - Cluster: where the call's rows (a group's on average, a kernel gradient's output rows) hold a tile's
      on a family whose cluster takes a pair (`Family.cluster`), a pair of multiprocessors whose row tiles
      load the block they share once (a matrix, a kernel gradient's cotangent: cuBLASLt's cluster on the
      4B's up dW read 35% less from device memory, its tensor pipe 98.7% busy against 82.4%, M-EM-48),
      each multiprocessor's tile as alone. On layouts whose products span a pair (tcgen05's 2-SM MMA,
      tried first where the rows hold its two row tiles, `paired_first`), every tile the pair's.
    - Band: the planar snake's column tiles a band (`resources.band`), the rows' and the columns' blocks
      balanced over a wave or, grouped, wider while a wave spans each group's row tiles whole, a
      cluster's tiles one program's.

    None where no tile holds."""
    # The facts the layouts share (their instruction family's); each layout's registers bound the shapes.
    family = layouts[0]
    columns = _derived_columns(family, width, dtype, units[1])
    steps = _steps(family, depths, depth, dtype, units[2])
    if not steps:
        return None  # no step the call's depth takes, so no tile (a 64-deep NVFP4 product on tcgen05's 256)

    def tiled(consumers: int, programs: int) -> tuple[Tiles | None, bool]:
        """The tile `consumers` take at `programs` a multiprocessor, and whether it is the balanced one
        staged busy: at the widest width its registers hold, its rows at their balance."""
        role = math.lcm(family.role_rows * consumers, units[0])
        most_rows = max(role, rows // 2 // role * role if grouped else axes.whole(rows, role))
        padded = grouped and not dense
        shaped = functools.partial(_shaped, consumers=consumers, programs=programs, rows=rows, width=width,
                                   multiprocessors=multiprocessors, grouped=padded)
        price_of = functools.partial(_tile_cost, family=family, target=target, dtype=dtype, rows=rows,
                                     width=width, padded=padded)
        held_tile = functools.partial(_held_tile, role=role, steps=steps, shaped=shaped,
                                      candidate=candidate, launch_of=launch_of, layouts=layouts,
                                      target=target, programs=programs, price_of=price_of,
                                      outputs=outputs, hooked=hooked)
        # Qwen3-8B's gate/up so takes 128 x 128, its 192-wide tile's rows shortened to 64 by its stages
        # (64 x 192 ran 1.21-1.30x XLA's time against 128 x 128's 1.10-1.14, FSA100POST32). Where no width
        # stages busy at its balance, the widest staging busy at all, then the widest that holds.
        side = _side(layouts, consumers, role, most_rows, columns, rows, target, dtype, steps, padded=padded)
        widths = _held_widths(layouts, consumers, columns, role, most_rows, side)
        return _best_width(widths, held_tile, side, price_of=price_of)
    # whose groups average fewer rows than one role (a decode step's experts, a few routed rows each), whose
    # further consumers would compute padding alone; where groups average a role's rows, a routing's groups
    # past it keep the further consumers' rows real. Row 3622, H100 (FS-3622-AB1/AB2-h100): one consumer ran
    # 0.73 to 0.999 of the cooperative tile's time at 1 to 32 rows a group, and 1.13 to 1.26 of it at 64.
    most = min(family.consumers, CONSUMERS)
    whole = rows >= family.role_rows if grouped else ragged or rows % (family.role_rows * most) == 0
    # balanced side, and with it each operand's reads a FLOP, at least one consumer's (CUTLASS's split, 2-7%
    # faster on every H100 scoreboard row); where no tile of theirs holds, one. Each at the most programs
    from dew.nn.kernels import generation
    with generation.as_target(target):
        for consumers in dict.fromkeys((most if whole else 1, 1)):
            # Default loop uses family's register-budget programs (raising cooperative wgmma/tgmm to
            # scalar-floor 2 programs drops caps to 80/56 regs, fatal ptxas C7602); `MosaicGmm._tile`
            # probes higher 1-consumer residencies via `pricing.max_programs` and prices plans together.
            program_counts = (range(family.programs(consumers), 0, -1)
                              if programs is None else (programs,))
            held = [tiled(consumers, count) for count in program_counts]
            tile = next((tile for tile, balanced in held if balanced), None)
            if tile := tile or next((tile for tile, _ in held if tile is not None), None):
                return tile
    return None


def paired_first(layouts: tuple, rows: int) -> tuple[tuple, ...]:
    """Dew: `derived`'s layouts in the order it tries them: those whose products span a pair
    (`mosaic.Transport.pairs_product`, tcgen05's 2-SM MMA) apart and first, where the call's rows hold a
    pair's two row tiles (each CTA's role), then the rest; each its own derivation, the first that holds a
    tile taken. A pair's product reads each CTA's half of the right operand's block, so a step stages half
    its columns a CTA. Measured: the 4B's dense products on a B200 ran 0.49-0.61 of the MMA's rate on one
    CTA's 128 x 128 and 128 x 192 tiles (M-EM-35). A hypothesis, not yet measured (M-EM-35's pinned pair arm
    and its ncu, rows 5671 and 5363): that one CTA's step is shared memory's, its operands passing it twice
    (staged, then read) at 128 bytes a clock, 0.50 and 0.60 of the MMA's rate, which the pair halves for the
    right operand."""
    paired = tuple(layout for layout in layouts if layout.pairs_product)
    alone = tuple(layout for layout in layouts if not layout.pairs_product)
    if not paired or rows < 2 * paired[0].role_rows:
        return (alone,) if alone else ()
    return (paired, alone) if alone else (paired,)


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


def _tile_cost(tile: Tiles, *, family: mosaic.Family, target, dtype, rows: int, width: int,
               padded: bool) -> tuple[float, float] | None:
    """One tile's seconds per useful output element-step (`pricing.step_seconds`, weighted by tile padding as
    `_padded_side` weighs grouped rows, tie-broken by operand reads per FLOP `1/M + 1/N`): the single cost
    model every family's derivation reads (docs/design/kernel-templates.md §7, rows 11944, 11983); None
    where `target` states no tensor rate. Leaving out the call's constant `rows` factor lets even `rows == 0`
    rank its tiles alike."""
    itemsize = jnp.dtype(dtype).itemsize
    seconds = pricing.step_seconds(tile, target, (itemsize, itemsize), issued=dtype)
    if seconds is None:
        return None
    row_tile, columns = pricing.tile_rows(tile, family), pricing.tile_columns(tile, family)
    row_work = (rows + row_tile / 2) if padded else (axes.whole(rows, row_tile))
    col_pad = (axes.whole(width, columns)) / max(width, 1)
    return (seconds * row_work * col_pad / (tile.block_m * columns * tile.block_k),
            1.0 / tile.block_m + 1.0 / columns)


def _held_tile(block_n: int, held_rows: int, *, role: int, steps: list[int], shaped: Callable,
               candidate: Callable, launch_of: Callable, layouts: tuple, target, programs: int,
               price_of: Callable, outputs: tuple = (), hooked: bool = False) -> tuple["Tiles | None", bool]:
    """`derived`'s tile at a width: the rows its registers or VMEM hold there (`held_rows`), shortened where
    its stages do not fit, at the first step that plans busy, taking the least-priced busy tile under the
    shared cost model (`_tile_cost`), and whether it stages deep enough to stay busy (those its family loads
    ahead, beside the one consumed and those it keeps in flight); a ring short of that depth, or one its
    consumers feed at it, deepened where a smaller staging of its outputs leaves room (`_ring_staged`)."""
    family = layouts[0]
    busy = 1 + family.loads_ahead + family.in_flight
    fallback = None
    best: tuple[tuple[float, float], Tiles] | None = None
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
            if outputs and ((tile.stages < busy and hooked and layout.stores_chunks)
                            or (tile.stages == busy and layout.feeds_itself)):
                # A column-local epilogue's ring short of busy where its outputs stage in chunks (its
                # outputs and side tiles beside it: the paired backward's pulled launch, three outputs and two
                # sides, row 1958), or one its consumers feed at its least busy depth, deepened by a smaller
                # staging of its outputs. An epilogue read out whole is not (the lightning flow's paired
                # gate/up, three outputs at 128 x 128 staged (64, 1), spilled 64 bytes at 168 registers).
                tile = _ring_staged(tile, layout, outputs, candidate=candidate, launch_of=launch_of,
                                    target=target, programs=programs,
                                    least=mosaic.HOOK_COLUMNS if tile.stages < busy else ACCUMULATOR_COLUMNS)
            if tile.stages >= busy:
                price = price_of(tile)
                if price is None:
                    return tile, True
                if best is None or price < best[0]:
                    best = (price, tile)
                break
            fallback = fallback or tile
    if best is not None:
        return best[1], True
    return fallback, False


def _ring_staged(tile: Tiles, layout: mosaic.Family, outputs: tuple, *, candidate: Callable,
                 launch_of: Callable, target, programs: int, least: int = ACCUMULATOR_COLUMNS) -> Tiles:
    """`tile`, whose ring is short of busy or its consumers feed (`mosaic.Transport.feeds_itself`) at its
    least busy depth, with its outputs (`outputs`, their dtypes) staged in smaller buffers where that leaves
    the ring deeper: of
    the stagings at the widest output's unstated chunk (`stores.output_staging`) and its halvings, in one
    buffer or two, legal for the tile's issuer as the tile specializes it and each output
    (`Tiles.staging_refusal`: whole accumulator atoms, whole TMA rows) and its buffers
    (`Family.output_staging`) smaller than the unstated ones, the most bytes first and one buffer before
    two on a tie, the first the planner places deeper, at the deepest ring it holds there
    (`resources.plan`) and no compile of this process refused (`candidate`). Its loading consumer issues a
    step's loads between its own products, no producer running ahead, so a ring at its least busy depth
    leaves a load's latency no slack: an RTX PRO 6000's input gradient at 128 x 256 x 32, two stages beside
    32 KB of output staging, slept on its full barriers in 14.4% of its samples (M-NG-1's ncu). The ring
    serves every step and the staging a tile's store once, CUTLASS's order (StageCountAutoCarveout: the
    epilogue's shared memory carved out first, the ring the rest). One buffer first: sm120's three-stage
    128 x 256 forward compiled spill-free at (64, 1) and (32, 1), 227-229 registers, where CUDA 12.9's
    allocator took (32, 2) to 255 and spilled 148 bytes, its live values no more (row 12667). The tile as
    it was where no smaller staging deepens it. Its chunks no narrower than `least` columns: a hooked
    epilogue's short ring no narrower than a hook's chunk (`mosaic.HOOK_COLUMNS`), where an 8-column
    staging deepened a row-scaled forward's ring past the registers its schedule left a consumer."""
    columns = pricing.tile_columns(tile, layout)
    piece = layout.chunk_columns(columns)
    shapes = tuple(((tile.share, columns), dtype) for dtype in outputs)

    def staged_bytes(staging: tuple[int, int] | None) -> int:
        return resources.allocated(*(resources.stored_bytes(buffer.shape, buffer.dtype)
                                     for buffer in layout.output_staging(shapes, tile.warpgroups, staging)))

    widest = max(stores.output_staging(columns, (dtype,), piece=piece)[0] for dtype in outputs)
    issuer = layout.specialized(tile.share, columns)
    halved = [(widest >> shift, count) for shift in range(widest.bit_length()) for count in (1, 2)
              if widest >> shift >= least]
    legal = [staging for staging in halved
             if dataclasses.replace(tile, staging=staging).staging_refusal(outputs, issuer) is None]
    unstated = staged_bytes(None)
    sized = {staging: size for staging in legal if (size := staged_bytes(staging)) < unstated}
    for staging in sorted(sized, key=lambda staging: (-sized[staging], staging[1])):
        staged = dataclasses.replace(tile, staging=staging)
        planned = resources.plan([launch_of(staged)], layout, target, programs=programs)
        if planned is None or planned[0].stages <= tile.stages:
            continue
        deeper = dataclasses.replace(staged, stages=planned[0].stages)
        if candidate(deeper):
            return deeper
    return tile


def _best_width(widths: list[tuple[int, int]], held_tile: Callable, side: int, *,
                price_of: Callable) -> tuple[Tiles | None, bool]:
    """Pick the least-priced busy tile across `widths` under the shared cost model (`_tile_cost`), or the
    first balanced busy tile where the target states no tensor rate."""
    busy_at, fallback = None, None
    best: tuple[tuple[float, float], Tiles] | None = None
    for block_n, held_rows in widths:
        tile, staged = held_tile(block_n, held_rows)
        if tile is not None and staged:
            price = price_of(tile)
            if price is None:
                if tile.block_m >= min(held_rows, side):
                    return tile, block_n == widths[0][0]
                busy_at = busy_at or tile
            elif best is None or price < best[0]:
                best = (price, tile)
        else:
            fallback = fallback or tile
    if best is not None:
        chosen = best[1]
        balanced = chosen.block_n == widths[0][0] and chosen.block_m >= min(widths[0][1], side)
        return chosen, balanced
    return busy_at or fallback, False


def _held_widths(layouts: tuple, consumers: int, columns: list[int], role: int, most_rows: int,
                 side: int) -> list[tuple[int, int]]:
    """`derived`'s widths at `consumers` and the rows each holds in its consumers' registers (its Shape):
    every column width up to the widest the area holds at the rows `side` (`_balanced_side`,
    `_padded_side`), each holding a role's rows, its registers the most generous of `layouts`' (a layout
    holding fewer gives way at placement, `resources.placed`)."""
    if not columns:
        return []
    area = max(resources.held_area(layout, consumers) for layout in layouts)
    widest = next((block_n for block_n in columns if block_n * side <= area), columns[-1])

    def held(block_n: int) -> int:
        return min(most_rows, max(resources.held_rows(layout, consumers, block_n) for layout in layouts))

    return [(block_n, held(block_n)) for block_n in columns if block_n <= widest and held(block_n) >= role]


def _steps(family: mosaic.Family, depths: tuple[int, ...] | None, depth: int | None, dtype,
           unit: int) -> list:
    """`derived`'s steps: the family's depths (or `depths`, a stage's own) the call's depth (where it has
    one) and the step `unit` divide, widest first as the family states them."""
    return [step for step in depths or family.depths(dtype)
            if (depth is None or depth % step == 0) and step % unit == 0]


def _side(layouts: tuple, consumers: int, role: int, most_rows: int, columns: list[int], rows: int, target,
          dtype, steps: list[int], *, padded: bool) -> int:
    """`derived`'s rows at `consumers`: a grouped call's weighing its groups' padding (`_padded_side`), every
    other call's balancing its reads (`_balanced_side`)."""
    if padded:
        return _padded_side(layouts, consumers, role, most_rows, columns, rows, target, dtype, steps)
    return _balanced_side(layouts, consumers, role, most_rows)


def _balanced_side(layouts: tuple, consumers: int, role: int, most_rows: int) -> int:
    """`derived`'s balanced rows at `consumers`: the square root of the accumulator area their registers
    hold (`resources.held_area`, `layouts`' most generous), which reads each operand fewest times a FLOP
    (1/rows + 1/columns at rows x columns = area), in whole roles, at least one, no more than the call's."""
    area = max(resources.held_area(layout, consumers) for layout in layouts)
    return max(role, min(most_rows, math.isqrt(area) // role * role))


def _padded_side(layouts: tuple, consumers: int, role: int, most_rows: int, columns: list[int], rows: int,
                 target, dtype, steps: list[int]) -> int:
    """`derived`'s rows at `consumers` for a grouped call of `rows` rows a group on average: each group's
    last row tile holds half a tile of rows of padding on average, which its products multiply as any
    rows, so a tile of m rows does r / (r + m/2) of its products' work for the group's rows; the rows, in
    whole roles up to `most_rows`, whose step on `target` (`pricing.step_seconds`: its products at the
    tensor rate or its operands through shared memory, the longer), at the widest of `columns` the
    consumers' accumulator area holds at those rows, costs least a row and a column of that useful work;
    the longest rows among equals. A tensor-bound step so takes shorter rows as a group's padding weighs
    more, where a step its shared memory bounds keeps the balanced side's fewer reads. An A100's mma.sync
    prices 64 x 256 and 128 x 128 alike a FLOP, its warps split to read their operands fewest times
    (`mosaic_sm80` `specialized`: 64 x 256 one warp a row block, 832 elements read a step of depth one
    against 768), so its groups' padding takes 64 rows: M-EM-69 ran its grouped forwards so 1.5-7.9%
    faster than 128 x 128 at Qwen3-30B-A3B's and GPT-OSS-20B's routings (SHAPE1-a100's 64 x 256 at 1.13x
    predates that split, its four warps each reading the columns' block whole). The balanced side
    (`_balanced_side`) where the target states no tensor rate."""
    balanced = _balanced_side(layouts, consumers, role, most_rows)
    area = max(resources.held_area(layout, consumers) for layout in layouts)
    best: tuple[float, int] | None = None
    # No rows past those the narrowest width's accumulator area holds: a taller tile holds no width.
    tallest = min(most_rows, area // columns[-1]) if columns else 0
    for block_m in range(role, tallest + 1, role):
        block_n = next(block_n for block_n in columns if block_n * block_m <= area)
        tiles = Tiles(block_m, block_n, steps[0], 2, 1, warpgroups=consumers, family=layouts[0].name)
        price = _tile_cost(tiles, family=layouts[0], target=target, dtype=dtype, rows=rows, width=block_n,
                           padded=True)
        if price is None:
            return balanced
        if best is None or price[0] <= best[0]:
            best = (price[0], block_m)
    return balanced if best is None else best[1]


def _shaped(layout, block_m: int, block_n: int, step: int, *, consumers: int, programs: int, rows: int,
            width: int, multiprocessors: int, grouped: bool) -> Tiles:
    """`derived`'s tile of a shape and step on `layout`: a pair of multiprocessors a cluster runs two row
    tiles of one group where its layout's loads multicast and the call's rows (a group's on average, a
    kernel gradient's output rows) hold a tile's, which then read the block they share once (a product's
    matrix, a kernel gradient's cotangent); a pair is two consumers' (`Tiles`), so a one-consumer tile, the
    fallback, runs alone; a layout whose products span a pair (`mosaic.Transport.pairs_product`, tcgen05's
    2-SM MMA) runs every tile on its pair."""
    multicast = consumers > 1 and rows >= block_m
    cluster = layout.cluster if layout.pairs_product or multicast else 1
    band = resources.band(multiprocessors * programs, block_m, block_n, width // block_n, grouped=grouped,
                          cluster=cluster)
    return Tiles(block_m, block_n, step, 2, band, warpgroups=consumers, cluster=cluster, family=layout.name,
                 programs=programs)
