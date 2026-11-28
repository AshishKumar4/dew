"""The attention template's Mosaic GPU mainloop's derivation (`attention_mosaic`'s `derived`, design §7): a
launch's config from its family's facts and the call's extents (`_derived`), the value columns and stages it
walks, and the facts and key its priced choices read (`_priced`, `_choice_key`)."""

import dataclasses
import functools
from collections.abc import Callable, Iterable

import jax.numpy as jnp
import tokamax

from dew._code import op_provenance
from dew.nn.kernels import generation, tuning
from dew.nn.kernels.templates import cost, mosaic, report, resources, swizzling
from dew.nn.kernels.templates.attention_mosaic_program import (
    BackwardTiles,
    Tiles,
    _width,
    extents,
    few_rows,
    most_consumers,
    query_rows,
    stagings,
    walked_extents,
)

WARPGROUP_WARPS = 4
"""The warps of a consumer role, a warpgroup's."""


def _column_blocks_of(family: mosaic.Family, dtype, value_width: int) -> list[int | None]:
    """The value columns a forward launch walks at once, widest first: the whole (None), then halved while a
    block stays whole swizzled columns, each block a launch (`Tiles.columns`), for a call no tile holds whole
    (an fp32 call's parts at width 128 on an H100: 64 x 64 staged 283,744 bytes of its 232,448, one
    consumer, et-fp32derive). A family that stages its operands swizzled takes a block at the narrowest
    swizzle its staged dtype has (`swizzling.NARROWEST_SWIZZLE_BYTES`, `swizzling.swizzle_for`'s least: a
    B200's scaled fp32 forward at width 128 holds 64 x 64 over 16 columns in 224 registers, where 32 spilled
    176 bytes, et-sneedfp32c); one whose products read registers, its own unit (`Family.column_unit`)."""
    whole = _width(value_width, family, dtype)
    # An fp32 call's operands stage as their bf16 parts (`MosaicForward.parts`).
    staged = jnp.bfloat16 if jnp.dtype(dtype) == jnp.dtype(jnp.float32) else dtype
    unit = (family.column_unit(dtype) if family.operands_in_registers
            else swizzling.swizzle_columns(staged, swizzling.NARROWEST_SWIZZLE_BYTES))
    return [None, *resources.halved_blocks(whole, unit)]


def _planned_stages(family: mosaic.Family, scope_of: Callable[[int], resources.Scope],
                    target: generation.GpuTarget, *, consumers: int, registers: int | None,
                    floor: int, most: int, resident: int) -> int:
    """Deepest ring in `[floor, most]` whose `scope_of(stages)` fits `target`'s share at `resident`
    programs per SM via `resources.plan` (rows 10948, 10962, 10993), every register fact `target`'s whatever
    GPU is in effect (an H100's backward deepened with an A100 in effect read the A100's setmaxnreg and
    raised, row 12800)."""
    if most <= floor or not family.budgets_registers:
        return floor
    with generation.as_target(target):
        capped = None if registers is None else min(registers, family.registers(consumers))
        launch = resources.Launch(scope=scope_of, consumers=consumers, registers=capped,
                                  floor=floor, most=most)
        planned = resources.plan((launch,), family, target, programs=resident)
    return floor if planned is None else planned[0].stages


def _priced(target: generation.GpuTarget) -> generation.GpuTarget:
    """The facts a derivation's priced choice reads on `target`: its generation's canonical part's
    (`generation.TARGETS`) at `target`'s size (its multiprocessors and shared memory), as a launch's identity
    reads it (`report._identity`), so the config a priced choice takes on one part is the one every part of
    its generation and size takes (row 8372: a GH200's own prices took 64-key tiles where the H100's took 128
    and named its kernels apart). A build of a stated part (`training.ahead.BuildTarget.device_kind`) prices
    as one of its generation still: a run of a canonical build reads the part it runs on, and a part's own
    facts priced there named a GH200's kernels apart from the H100 build it ran (row 13252, reopened).
    Pricing alone: a launch's other schedule decisions that read a part's live facts
    (`attention_mosaic_forward.rewalks_in_place`'s bandwidth and launch latency) still may differ. Its L2's
    read plateau none: a part's own (`Measured.l2_bytes_per_second`), which no other part of its generation
    reads (kernel-templates.md §6: a part no measurement names states none; Train 1508's F2, a Workstation
    part priced at the Server Edition's L2). The canonical part's other measured facts it reads still (row
    14210)."""
    canonical = generation.TARGETS.get(target.generation)
    if canonical is None:
        return target
    measured = {kind: dataclasses.replace(part, l2_bytes_per_second=None)
                for kind, part in canonical.measured.items()}
    return dataclasses.replace(canonical, multiprocessors=target.multiprocessors,
                               shared_limit=target.shared_limit, measured=measured)


def _choice_key(op, ba: tokamax.BoundArguments, candidates: tuple[Tiles, ...] | tuple[BackwardTiles, ...],
                target: generation.GpuTarget) -> str | None:
    """What decides `op`'s priced choice among `candidates` on `target` (`tuning.stored_choice`: a
    forward's consumer count, a backward's one launch or two), keyed through `report.stored_key`; None where
    the installation has no seal. Every fact `cost.launch_seconds` reads of `target` keys it: a
    multiprocessor's rates and the whole device's (its memory's among them), and its paths' latencies."""
    rates, device = (cost.rates(target, device=on_device)._asdict() for on_device in (False, True))
    latencies = (target.kernel_step_seconds, target.product_latency, target.barrier_seconds,
                 target.copy_latency)
    return report.stored_key(
        op_provenance(op, ba.arguments), report._identity(op, ba, target), target.generation,
        target.device_kind, target.multiprocessors, target.launch_seconds, rates, device, latencies,
        tuning.encoded(candidates))


def _derived[Config: (Tiles, BackwardTiles)](family: mosaic.Family, dtype, held: int, walked: int,
                                             tiles: Callable[[int, int, int], Config],
                                             fits: Callable[[Config], bool], *,
                                             overlapped: bool, walks: Callable = extents,
                                             rows: int | None = None,
                                             least_walked: Callable[[int], int] = lambda rows: 1,
                                             least_held: Callable[..., int] = query_rows,
                                             choose: Callable[[list[Config]], Config] | None = None
                                             ) -> Config | None:
    """A launch's config from its family's facts and the call's extents (design §7): its
    consumers up to the family's count (`Family.consumers`, at most a launch's `most_consumers`)
    which runs under its partner's products as the consumers take turns, FlashAttention-3's), else one (a
    step issuing its products back to back, which turns only make wait); fewer where the `held` rows a
    consumer accumulates (a forward's queries, a dK/dV launch's keys) fill fewer of the family's units; each
    consumer's rows the family's unit (`_extents`' least: the rows its products cover, `Family.role_rows`, in
    whole swizzled rows); the tile it walks the largest of its extents (`walks`: a backward's the family's
    unit, `extents`, its walked tiles' weights and cotangents feeding products from registers, which at fewer
    rows than the unit cross warps (fs-backward-specialized: mma.sync at 32-row shares, no relayout lowers);
    a forward's `walked_extents`, a product's depth in whole swizzled rows) within the `walked` extent (at
    least `least_walked(rows)`, the walked tile a consumer of `rows` rows takes at least) whose live set a
    consumer's registers hold and
    whose two stages the block holds (`fits`), two the ring's busy depth, a slot loading while one is read;
    where its family's products read operands from registers, a tile its transposed streams staged at the
    family's own swizzle do not fit fits them staged narrowest (`Tiles.narrow`), a wide stage's bandwidth
    traded for the larger tile; fewer consumers where no tile of the family's count holds the block (a GPU
    of a smaller one). Where an `overlapped` step's program runs one consumer on registers its family
    budgets (no partner whose products a softmax runs under: a family of one, or a family whose products read
    registers falling back to one, `wide`), its consumer's rows twice the unit too, each warp two of the
    products' row tiles, FlashAttention 2's 128-row tile on one warpgroup, first among one consumer's tiles,
    where its first key tile holds, each key tile's loads and fragments read once for twice the queries (an
    A100 at width 128, 128 x 32 against 64 x 64, ptxas 250 and 189 registers spilling none; width 64, 128 x
    64 against 64 x 128, two programs a multiprocessor against one, et-a100tile2); then the unit, then
    fewer rows. `tiles(rows, walked, consumers)` builds the config; `rows` a consumer's where given
    (a backward's `fewer_rows`), else `least_held`'s for the `held` rows: a forward's least query tile
    (`query_rows`: the issuer's rows for a decode step's few), a backward's a role's rows (`_role_rows`).
    None where no tile fits."""
    unit = least_held(family, dtype, held) if rows is None else rows
    most = min(most_consumers(family, tiles(unit, family.rows, 1)), family.consumers) if overlapped else 1
    # Twice the unit a consumer where the program runs one: a family of one consumer, or a family whose
    # products read registers at its fallback to one (a fed mma.sync's two consumers that hold no tile, its
    # one then FlashAttention 2's 128 rows, as a family of one derives it).
    wide = (rows is None and overlapped and family.budgets_registers
            and (most == 1 or family.operands_in_registers))

    def padded(rows: int, consumers: int) -> int:
        return -(-held // (rows * consumers)) * rows * consumers

    def counts(rows: int) -> range:
        return range(max(1, min(most, -(-held // rows))), 0, -1)

    def first(groups: Iterable[tuple[int, Iterable[int]]]) -> Config | None:
        picked: list[Config] = []
        for grp_rows, tried in groups:
            for consumers in tried:
                if (consumers > 1
                        and padded(grp_rows, consumers) - padded(grp_rows, consumers - 1) > held // 8):
                    # A count whose tile pads the held rows by more than an eighth of them beyond the next
                    # count's (a 192-row tile of 3 consumers over a 256-row shard pads it by half, where
                    # 2 consumers' 128 pad none): the call's padding is a copy and its last tile wasted.
                    continue
                # Each walked tile that fits, at its first staging, priced against the rest where `choose`
                # prices: a larger tile is not the faster at every call (M-NA-11b-gh200: a T512 training
                # walk of d128 with sinks at 64 x 128 took 72.80 us, at 64 x 64 47.49, the same residency).
                for extent in walks(family, dtype, max(walked, least_walked(grp_rows))):
                    for narrow in stagings(family):
                        config = dataclasses.replace(tiles(grp_rows, extent, consumers), narrow=narrow)
                        if fits(config):
                            if choose is None:
                                return config
                            picked.append(config)
                            break
        if not picked:
            return None
        return picked[0] if len(picked) == 1 or choose is None else choose(picked)

    # The unit at the counts above one first (FlashAttention 3's consumers); then one consumer's heights:
    # twice the unit where `wide` and the held rows take it, padding them by no more than an eighth beyond
    # the unit's, then the unit (priced together by `choose` so e.g. 64x64 can beat 128x32 on sm80, row
    # 6216); then, for a call of fewer rows than a role's, fewer of them a tile at every count, its issuer's
    # rows at least, where no tile of them all holds.
    top = (held if rows is None and not family.budgets_registers
           else 2 * unit if wide else unit)
    heights = ([] if rows is not None or top <= unit else
               [h for h in extents(family, dtype, top)
                if h > unit and padded(h, 1) - padded(unit, 1) <= held // 8])
    heights.append(unit)
    fewer = range(unit - family.rows, 0, -family.rows) if rows is None and few_rows(family, held) else ()
    tiers = (
        ((unit, [count for count in counts(unit) if count > 1]),),
        tuple((height, [1]) for height in heights),
        *(((r, counts(r)),) for r in fewer),
    )
    return next((config for config in (first(tier) for tier in tiers) if config is not None), None)


def _role_rows(family: mosaic.Family, dtype, held: int) -> int:
    """A backward launch's least held tile: a role's rows in whole swizzled rows (`extents`), however few
    the call's keys. Its two launches transpose its tiles (`backward_family`: dQ's rows its queries, dK/dV's
    its keys), so a side of fewer rows is both sides' (`fewer_rows`, `_sides`); the issuer's rows a forward
    takes for a few queries (`query_rows`) gave a 16-token training step's keys 16 rows under 64 queries, and
    neither launch lowered on mma.sync ("Cannot cast to layout" of the 16 key rows to the 64 a consumer's
    fragments tile; at 32 tokens the one launch's "cannot relayout", et-detbwd)."""
    return extents(family, dtype, held)[-1]


def _least_keys(family: mosaic.Family, rows: int) -> int:
    """The key tile a forward consumer of `rows` rows walks at least: a tile of fewer rows than a role's lays
    its warps along its keys, a whole 16-key product's columns each (`MosaicForward._launch_legal`), so a
    call of fewer keys pads to them (a 3-query call's 16-row tile over its 3 keys derived only 32-key tiles,
    each refused)."""
    return WARPGROUP_WARPS * 16 if rows < family.role_rows else 1


def _fewer_hooked(score_mod, family: mosaic.Family) -> bool:
    """Whether a backward of `score_mod` takes fewer rows a consumer than a role's on `family` (`fewer_rows`):
    a call of no score hook, or one whose family reads its products' fragments from registers
    (`Family.operands_in_registers`, mma.sync's), where `_score_logits` places query coordinates in the
    accumulator's layout so row-indexed hooks (`q_pos[:, :1]`) lower at 32-row tiles too (rows 7279, 7280,
    7360)."""
    return score_mod is None or family.operands_in_registers


def _sides(rows: int, family: mosaic.Family, dtype, extent: int) -> tuple[int, ...]:
    """A backward's walked tile of a consumer's `rows` below a role's (`fewer_rows`): whole multiples of them
    and of its swizzled rows, a launch's walked tile the other's consumer rows (`extents`' bounds)."""
    return tuple(side for side in walked_extents(family, dtype, extent) if side % rows == 0)


def _query_walk(rows: int | None) -> Callable[..., tuple[int, ...]]:
    """The query blocks a backward walks for a consumer of `rows` rows, largest first: a role's (None)
    `extents`, fewer rows' (`fewer_rows`) `_sides`; its derivation's walk and its ladder's (`ladder`)."""
    return extents if rows is None else functools.partial(_sides, rows)
