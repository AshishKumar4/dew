"""The attention template's Mosaic GPU mainloop as Template implementations (`MosaicForward`,
`MosaicBackward`; design §6 and §7): the forward, its training form and the backward over every call form,
each launch's one config derived from the instruction family's and the device's facts and the call's extents
(`derived`), legal where its programs stage within the device's block and a consumer's registers hold its live
set (`legal`): no table, no sweep, no space ranked. A GPU nobody measured of a family with a bundle here
(`bundle`) takes its own facts' config. The launches are `attention_mosaic_forward`'s and
`attention_mosaic_backward`'s; what they share, `attention_mosaic_program`'s."""

import dataclasses
import functools
import math
from collections.abc import Callable, Hashable, Sequence
from typing import ClassVar, Literal, overload

import jax
import jax.numpy as jnp
import tokamax

from dew._devices import deterministic_ops_requested
from dew.nn.kernels import generation, tuning
from dew.nn.kernels.selection import Refusal
from dew.nn.kernels.templates import attention_algebra as algebra, cost, mosaic, product, resources, swizzling
from dew.nn.kernels.templates.attention import (
    ACCURATE,
    DKV_GRADS,
    DQ_GRADS,
    NATURAL,
    TRACED_TILE,
    AttentionLaunch,
    Gradients,
    Hooks,
    Kept,
    KeySpans,
    Kind,
    Param,
    Split,
    gradients,
    kept_dtype,
    logit_scale,
    power_of_two,
    prescale,
)
from dew.nn.kernels.templates.attention_mosaic_backward import _launch_backward
from dew.nn.kernels.templates.attention_mosaic_coop import _coop_allocations
from dew.nn.kernels.templates.attention_mosaic_dkv import _dkv_allocations
from dew.nn.kernels.templates.attention_mosaic_dq import _dq_allocations
from dew.nn.kernels.templates.attention_mosaic_forward import (
    _codes_of,
    _forward_allocations,
    _plain_forward,
    _trained_forward,
    _whole,
    certified_walk,
    forward_products,
)
from dew.nn.kernels.templates.attention_mosaic_fused import _fused_allocations, staged_parts, stores_typed
from dew.nn.kernels.templates.attention_mosaic_program import (
    BACKWARD_CONSUMERS,
    ROUNDED,
    BackwardRegisters,
    BackwardTiles,
    ForwardRegisters,
    Tiles,
    _block,
    _padded,
    _width,
    backward_bundle,
    backward_layouts,
    bundle,
    bundle_of,
    extents,
    few_rows,
    fewer_rows,
    fuses,
    hook_reduces,
    launch_scope,
    layouts,
    loads_parts,
    most_consumers,
    placed_draws,
    query_rows,
    reducing,
    refused,
    score_places,
    split_queries,
    stagings,
    varying_slopes,
    walked_extents,
    whole_tiles,
)
from dew.nn.kernels.templates.base import Template, Unfitted


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


def _derived[Config: (Tiles, BackwardTiles)](family: mosaic.Family, dtype, held: int, walked: int,
                                                tiles: Callable[[int, int, int], Config],
                                                fits: Callable[[Config], bool], *,
                                                overlapped: bool, walks: Callable = extents,
                                                rows: int | None = None,
                                                least_walked: Callable[[int], int] = lambda rows: 1,
                                                least_held: Callable[..., int] = query_rows
                                                ) -> Config | None:
    """A launch's config from its family's facts and the call's extents (design §7), no space ranked: its
    consumers the family's (`Family.consumers`) where its step `overlapped` (a softmax between its products,
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
    of a smaller one). Where an `overlapped` step's family runs one consumer a program on registers it
    budgets (no partner whose products a softmax runs under: mma.sync's), its consumer's rows twice the unit
    too, each warp two of the products' row tiles, FlashAttention 2's 128-row tile on one warpgroup, first,
    where its first key tile holds, each key tile's loads and fragments read once for twice the queries (an
    A100 at width 128, 128 x 32 against 64 x 64, ptxas 250 and 189 registers spilling none; width 64, 128 x
    64 against 64 x 128, two programs a multiprocessor against one, et-a100tile2); then the unit, then
    fewer rows. `tiles(rows, walked, consumers)` builds the config; `rows` a consumer's where given
    (a backward's `fewer_rows`), else `least_held`'s for the `held` rows: a forward's least query tile
    (`query_rows`: the issuer's rows for a decode step's few), a backward's a role's rows (`_role_rows`).
    None where no tile fits."""
    unit = least_held(family, dtype, held) if rows is None else rows
    most = most_consumers(family, tiles(unit, family.rows, 1)) if overlapped else 1
    wide = rows is None and overlapped and most == 1 and family.budgets_registers

    def padded(rows: int, consumers: int) -> int:
        return -(-held // (rows * consumers)) * rows * consumers

    def first(rows: int) -> Config | None:
        for consumers in range(max(1, min(most, -(-held // rows))), 0, -1):
            if consumers > 1 and padded(rows, consumers) - padded(rows, consumers - 1) > held // 8:
                # A count whose tile pads the held rows by more than an eighth of them beyond the next
                # count's (a 192-row tile of three consumers over a 256-row shard pads it by half, where two
                # consumers' 128 pad none): the call's padding is a copy and its last tile's work wasted.
                continue
            for extent in walks(family, dtype, max(walked, least_walked(rows))):
                for narrow in stagings(family):
                    config = dataclasses.replace(tiles(rows, extent, consumers), narrow=narrow)
                    if fits(config):
                        return config
        return None

    # Twice the unit where the held rows take it, padding them by no more than an eighth beyond the unit's;
    # then the unit; then, for a call of fewer rows than a role's, fewer of them a tile, its issuer's rows at
    # least, where no tile of them all holds (a 48-row training walk's scaled op on mma.sync held 278
    # registers of 255 at 48 x 64, the least key tile so few rows take, `_least_keys`, et-sm86scaled). The
    # first that holds is the config: legality steps down, nothing is ranked (design §7).
    heights = [2 * unit] if wide and padded(2 * unit, 1) - padded(unit, 1) <= held // 8 else []
    heights.append(unit)
    fewer = range(unit - family.rows, 0, -family.rows) if rows is None and few_rows(family, held) else ()
    return next((config for config in map(first, (*heights, *fewer)) if config is not None), None)


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


def _sides(rows: int, family: mosaic.Family, dtype, extent: int) -> tuple[int, ...]:
    """A backward's walked tile of a consumer's `rows` below a role's (`fewer_rows`): whole multiples of them
    and of its swizzled rows, a launch's walked tile the other's consumer rows (`extents`' bounds)."""
    return tuple(side for side in walked_extents(family, dtype, extent) if side % rows == 0)


@dataclasses.dataclass(frozen=True)
class _Mainloop[Config: (Tiles, BackwardTiles)](AttentionLaunch, Template[Config]):
    """What this mainloop's launches share as Template implementations (`MosaicForward`, `MosaicBackward`):
    the call's own arrays as arguments, padded by a launch to its config's blocks; the score hook (and the
    forward's output hook) as fields keyed by what they compute (`hook_program`); the configs of the
    target's family (`fits`), a consumer's registers its roles' share (`register_budget`), where its bundle
    has a bundle here (`bundle`) and the call is one it takes (`refused`)."""

    name = "mosaic_gpu"
    GENERATES = True
    score_mod: Callable | None = None
    form: str = "spans"
    keys: int = 0
    score_kinds: tuple[Kind, ...] = ()
    score_dtypes: tuple[str, ...] = ()
    score_reduces: bool = False
    """Whether its score hook may reduce across warps (`hook_reduces`), its launches' warpgroups reserving
    their scratch (`reducing`), as its builder reads it off the hook: a keyed launch's hook is its
    fingerprint, which keeps no trace of it."""
    scale: float = 1.0
    natural: bool = NATURAL
    subtracted: bool = False
    """Whether an inference walk's fold takes its logits as a difference of raw products scaled after it is
    taken (`algebra.Forward.subtracted`: no fold's reach, no natural rewalk), a decode step's
    (`attention_decode`), whose walk the memory bounds."""
    per_span: int = 1
    """The queries each of the spans' intervals holds (`KeySpans.per_span`)."""

    def spans(self, start, end, query_ids, key_ids) -> KeySpans:
        """The call's spans from its arrays and the op's static form, key count and queries an interval."""
        return KeySpans(start, end, query_ids, key_ids, self.keys, self.form, self.per_span)

    UNBUNDLED: ClassVar[str] = "the Mosaic mainloop has no bundle for {name} yet"
    """Why a launch runs on no device of a family it has no bundle of (`declined`)."""

    @staticmethod
    def _bundle(generation_name: str | None, dtype) -> mosaic.Family | None:
        """The bundle the launch runs on (`bundle`)."""
        return bundle(generation_name, dtype)

    @staticmethod
    def _layouts(generation_name: str | None, dtype, rows: int | None = None) -> tuple[mosaic.Family, ...]:
        """The bundles the launch takes (`layouts`)."""
        return layouts(generation_name, dtype, rows)

    def padded(self, ba: tokamax.BoundArguments, config: Config) -> float | None:
        """The share of the score grid, each head's queries by its keys, that the launch's blocks pad
        past the call's (`_padded`, `tuning.KernelOp.padded`): its query and key blocks' tails."""
        queries, keys = ba.arguments["query"].shape[1], _keys(ba)
        padded_queries, padded_keys = _padded(queries, keys, config)
        return 1 - queries * keys / (padded_queries * padded_keys)

    def register_budget(self, config: Config) -> int:
        """A consumer thread's share of the registers (`Family.registers`): the family's roles share the
        file (`handoffs.Schedule`)."""
        # A consumer's budget is its family's, whatever the operands' dtype (`mosaic.Family.registers`).
        return bundle_of(config, jnp.bfloat16).registers(config.consumers)

    def eligible(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> bool:
        return self.declined(ba, target) is None

    def declined(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> Refusal | None:
        """Why the launch runs no call of `ba` on `target`, or None: its family has no bundle here, the call
        is one it does not take (`refusal`, each reason with its ledger row), it is interpreted in a GPU
        process (`grouped_matmul._Mosaic.eligible`: the interpreter's callbacks are ordered effects, which a
        GPU process's lowering cannot carry), or no config fits the device (`derived`)."""
        name = generation.family(target.generation)
        query = ba.arguments["query"]
        family = self._bundle(target.generation, query.dtype)
        if family is None:
            return Refusal('family', self.UNBUNDLED.format(name=name or target.generation))
        if (reason := self.refusal(ba, family)) is not None:
            return reason
        if ba.arguments["interpret"] and jax.default_backend() != "cpu":
            return Refusal('interpret', "an interpreted Mosaic launch runs in a CPU process alone")
        if self.derived(ba, target) is None:
            return Refusal('resources',
                           f"no config of the Mosaic mainloop fits {target.generation}'s block and registers")
        return None

    def legal(self, ba: tokamax.BoundArguments, config: Config, target: generation.GpuTarget) -> bool:
        """Whether a launch of `ba` runs at `config` on `target`: the target's family's bundle (`bundle`),
        tiles its program takes (`whole_tiles`), a call the launch takes (`refusal`), and the block and
        registers `Template.legal` holds it to. A pinned or stored config is held to it as a derived one."""
        query = ba.arguments["query"]
        found = {family.name: family
                 for family in self._layouts(target.generation, query.dtype, query.shape[1])}
        family = found.get(config.family)
        # A forward's and a dQ launch's layout streams through a producer (`Family.stream`); a layout whose
        # consumers feed their own rings (`split_queries`) runs the dK/dV launches alone, the keys-first one
        # and the fused one, each consumer its own keys over the stream they walk whole. The forward on it
        # has no use the producer's layout does not serve, and deadlocked (row 2599): it is no config.
        walks_whole = not isinstance(config, Tiles) and config.queries_first
        # One consumer feeding its own rings: no turns between consumers, whose dropped turns hung the
        # two-consumer fed forward; a decode step's 16-row tile on mma.sync's fed layout (an H100's,
        # `few_rows`), and every forward on a device whose bundle has no producer's layout (consumer
        # Blackwell's, an unseen NVIDIA GPU's: `bundles.of`, row 4727).
        alone = family is not None and isinstance(config, Tiles) and config.consumers == 1
        return (family is not None and (walks_whole or alone or not split_queries(family))
                and whole_tiles(family, config) and self.refusal(ba, family) is None
                and super().legal(ba, config, target))

    def _family(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> mosaic.Family | None:
        """The target's family's bundle where the launch takes the call there, else None."""
        query = ba.arguments["query"]
        family = self._bundle(target.generation, query.dtype)
        return None if family is None or self.refusal(ba, family) is not None else family

    WALKED: ClassVar[str] = "block_k"
    """The config's field naming the tile its consumers walk (a forward's keys, a backward's queries)."""

    def neighbours(self, ba: tokamax.BoundArguments, config: Config) -> Sequence[Config]:
        """The configs next to the derived `config` an opt-in tuning run times beside it (`variants`): the
        walked tile one unit size either side (`extents`), a stage fewer and more, a consumer fewer and
        more, each alone; `variants` keeps the legal ones."""
        query = ba.arguments["query"]
        family = bundle_of(config, query.dtype)
        walked = getattr(config, self.WALKED)
        reach = max(walked * 2, query.shape[1], _keys(ba))
        walks = walked_extents if isinstance(config, Tiles) else extents  # `_derived`'s `walks`
        sizes = [size for size in walks(family, query.dtype, reach) if size in (walked // 2, walked * 2)]
        changes = [*({self.WALKED: size} for size in sizes),
                   {"stages": config.stages - 1}, {"stages": config.stages + 1},
                   {"consumers": config.consumers - 1}, {"consumers": config.consumers + 1}]
        return tuple(dataclasses.replace(config, **change) for change in changes
                     if all(value > 0 for value in change.values())
                     and change.get("stages", 2) >= 2
                     and change.get("consumers", 1) <= most_consumers(family, config))

    def _get_heuristics_config(self, ba: tokamax.BoundArguments) -> Config:
        target = generation.gpu_target()
        query = ba.arguments["query"]
        family = self._bundle(target.generation, query.dtype)
        if family is not None and (reason := self.refusal(ba, family)):
            raise NotImplementedError(reason)
        config = self.derived(ba, target)
        if config is None:
            raise Unfitted(type(self).__name__, None, generation.shared_budget(target), target.generation)
        return config

    def refusal(self, ba: tokamax.BoundArguments, family: mosaic.Family) -> Refusal | None:
        """Why the launch takes no call of `ba` on `family`, or None (`refused`)."""
        raise NotImplementedError

    def resident(self, ba: tokamax.BoundArguments, config: Config, target: generation.GpuTarget) -> int:
        """The programs of its launch at `config` a multiprocessor holds at once: its family's residency at
        its registers (`Family.programs`) and its shared memory's, each program's reservation its own
        (`resources.shared_programs`); one at the least, a legal config's."""
        family = bundle_of(config, ba.arguments["query"].dtype)
        resident = family.programs(config.consumers, self.registers(ba, config))
        if target.shared_limit:
            staged = self.staged_bytes(ba, config, target)
            resident = max(1, resources.shared_programs(resident, staged, target.shared_limit))
        return resident


@dataclasses.dataclass(frozen=True)
class MosaicForward(_Mainloop[Tiles]):
    """The template's forward on this mainloop over the call's own arrays (`forward` gives them): queries
    `[B, Q, H, D]` over keys `[B, K, Hk, D]` and values `[B, K, Hk, Dv]`, each query's keys `start` to `end`
    `[B, Q]` and, with documents, its id `[B, Q]` against the keys' `[B, K]`; the score and output hooks'
    parameters (`score_params`, `output_params`, laid out by `score_kinds` and `output_kinds`) and the
    softmax's initial state (`Initial`, or None). Returns the output `[B, Q, H, Dv]` and, with `residuals`,
    the backward's residuals (`forward`'s). Its configs come from the family's facts and the call's extents
    (`derived`), legal within the device's block and a consumer's registers (`legal`)."""

    HOOKS = ("score_mod", "output_mod")
    config_cls = Tiles
    output_mod: Callable | None = None
    output_kinds: tuple[Kind, ...] = ()
    output_dtypes: tuple[str, ...] = ()
    output_width: int = TRACED_TILE
    out_dtype: str | None = None
    """The inference walk's output dtype, the query's where None: fp32 for a split launch's slices, which a
    merge takes (`attention_decode`)."""
    masked: bool = False
    """Whether a spans call's rewalk (the fold's natural walk, the headroom guard) runs as a launch every time
    over its rows alone, no branch (`algebra.walked_launches`' `masked`: a decode step's, row 1978)."""
    splits: int = 1
    """The slices each query tile's keys are cut into, each a program of its own (`_Program.splits`): a
    split-KV inference launch's, its outputs each slice's (`attention_decode` merges them)."""
    parts: int = 1
    """The bf16 parts each of an fp32 call's operands enters as (`product.EXACT_PARTS`, `product.formed`),
    stacked along the batch axis, part `p`'s rows from `p·B`: every product its parts' (`product.terms`),
    each 16-bit (attention.md 4b). One for a 16-bit call."""
    rounds: float | None = None
    """A quantized model's weights' bound (`Hooks.rounds`), which the anchored training walk rounds them to
    at each row's maximum: a call that keeps its split residuals' alone. None where its weights stay whole."""
    scaled: bool = False
    """Whether its kernels walk scaled (`algebra.SCALED_WALK`): the op a call some row of which can be scaled
    launches in its common op's place (`_fwd`, `algebra.may_scale`), its own config derived from its own
    registers and shared memory (`scales_in_kernel`), which may be smaller than the common op's."""
    bounded: bool = False
    """Whether its caller has bounded the call's values below where any row can be scaled (a cache's call
    over the largest magnitudes its cache states, `attention_decode.attend`'s `bounds`), so its kernels walk
    unscaled, a `cached` call's too, reading no value to choose them."""

    def _hooks(self, score_params, output_params, initial, drop) -> Hooks:
        """The call's hooks from its parameters, state and draw, its static hooks the op's."""
        return Hooks(self.score_mod, tuple(map(Param, score_params, self.score_kinds)), initial,
                     self.output_mod, tuple(map(Param, output_params, self.output_kinds)), drop, self.rounds)

    def _fwd(self, query, key, value, start, end, query_ids, key_ids, score_params, output_params, initial,
             drop=None, table=None, sources=None, scales=None, *, residuals: Kept, interpret: bool,
             config: Tiles, return_residuals: bool = False):
        hooks = self._hooks(score_params, output_params, initial, drop)
        if reason := refused(query.dtype, query.shape[3], value.shape[3], hooks, backward=False,
                             family=bundle_of(config, query.dtype)):
            raise NotImplementedError(reason)
        spans = self.spans(start, end, query_ids, key_ids)
        # A rare call's scaled op states no work to Dew's FLOP count, as the fold's rewalk does not: the
        # common op's is the step's (`_launch_flops`).
        form = {"scale": self.scale, "natural": self.natural, "tiles": config, "interpret": interpret,
                "counts": not self.scaled,
                "registered": self.registered(query, value, score_params, residuals, table, sources, scales,
                                              initial)}
        scaled = self.scaled or (self.cached(table, sources, scales) and not self.bounded)

        def launched():
            match residuals:
                case Kept.SPLIT:
                    if scales is not None:
                        raise NotImplementedError("a quantized cache's keys are read by inference alone")
                    return _trained_forward(query, key, value, spans, hooks, scaled=scaled, **form)
                case Kept.NONE | Kept.OUTPUT:
                    # The inference walk, its rows beside the output where it keeps them.
                    return _plain_forward(query, key, value, spans, hooks, rows=residuals is Kept.OUTPUT,
                                          out_dtype=self.out_dtype, masked=self.masked, splits=self.splits,
                                          table=table, sources=sources, subtracted=self.subtracted,
                                          scales=scales, parts=self.parts, scaled=scaled, **form)

        if scaled or self.bounded or hooks.rounds is not None or not algebra.headroom(value.dtype):
            return launched(), None
        # A call some row of which can be scaled (`algebra.may_scale`) launches the scaled op's kernels (its
        # own config, `scaled`), every other call, every ordinary input's, this op's unscaled ones, bit for
        # bit the scaled walk's there.
        may = algebra.may_scale(_whole(value, self.parts), keys_axis=1,
                                initial=None if initial is None else initial.acc,
                                amplified=1.0 if drop is None else drop.bound)
        arguments = (query, key, value, start, end, query_ids, key_ids, score_params, output_params, initial,
                     drop, table, sources, scales)

        def rare():
            # A pin of this op's where the scaled op can take it, else its own config, which derives wherever
            # this op's does (`tuning.pinned_where_taken`): its registers and shared memory are not the common
            # op's.
            scaled_op = dataclasses.replace(self, scaled=True)
            with tuning.pinned_where_taken(scaled_op, scaled_op.bind(*arguments, residuals=residuals,
                                                                     interpret=interpret)) as taken:
                return taken(*arguments, residuals=residuals, interpret=interpret)

        return jax.lax.cond(may, rare, launched), None

    def completed(self, ba: tokamax.BoundArguments, config: Hashable) -> Hashable:
        """A pin's or an explicit config's natural rewalk derived once where it names none (`_relaunched`,
        rows 2621, 2647); the config as it is where it names any of its rewalk's tiles (its key tile, its
        consumers, its value columns: row 2675) or none derives (a config of another type, which `pinnable`
        refuses)."""
        if (not isinstance(config, Tiles) or config.natural_form
                or (config.relaunch_k, config.relaunch_consumers, config.relaunch_columns) != (None,) * 3):
            return config
        return self._relaunched(ba, config, generation.gpu_target()) or config

    def legal(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget) -> bool:
        """Its launch's (`_launch_legal`), and a fold's natural rewalk at the tiles the config names (the
        derivation's, a tuning neighbour's, a pin's completed one, `completed`), their key tile dividing the
        config's and their consumers at most its, holding too (row 2584); a fold's config that names none
        rewalks at its own tiles, which must hold (rows 2621, 2647)."""
        if not self._launch_legal(ba, config, target):
            return False
        if config.natural_form or self._unrewalked(ba):
            return True
        relaunch = config.relaunch
        return (config.block_k % relaunch.block_k == 0 and relaunch.consumers <= config.consumers
                and self._launch_legal(ba, relaunch, target))

    def _launch_legal(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget) -> bool:
        """`_Mainloop.legal`'s for one launch at `config`."""
        if not super().legal(ba, config, target):
            return False
        query = ba.arguments["query"]
        # A quantized call's codes converted where the device converts them (`mosaic.converts`: an FP8 cache
        # on sm_89 on), its call the gathered keys' path elsewhere.
        codes = _codes_of(ba.arguments["key"], ba.arguments.get("scales"))
        if codes is not None and not mosaic.converts(codes):
            return False
        family = bundle_of(config, query.dtype)
        # A tile of fewer rows than a role's lays its warps along its keys, a whole 16-key product's columns
        # each (Mosaic's layouts take whole such tiles a warp: a 96-key tile's 24 refused, dd-q5p4; 64-key
        # tiles lower at widths 64 to 256 on sm80 and sm90, dd-k16b).
        if config.block_q < family.role_rows and config.block_k % (WARPGROUP_WARPS * 16):
            return False
        # Its key tiles (the launch's, its natural rewalk's) split among the consumers that copy them
        # (`Transport.shares_rows`: two mma.sync consumers' cp.async in whole 64-row parts, so no 32-key tile
        # of two consumers, fs-qualified6).
        if not (family.shares_rows(config.block_k, config.consumers)
                and family.shares_rows(config.relaunch_k or config.block_k,
                                       config.relaunch_consumers or config.consumers)):
            return False
        # Its products' right operands, the keys' tile in S = Q·Kᵀ (the heads' width deep, a key tile's
        # columns) and the values' in P·V (a key tile's rows of the launch's value columns, `Tiles.columns`),
        # ones its family's products take (`Issuer.takes_right`: tcgen05's 64-row products pad a narrow one
        # only at their own depth). Its natural rewalk's are judged as its launch's (`legal`).
        width = _width(query.shape[-1], family, query.dtype)
        value_columns = config.columns or _width(ba.arguments["value"].shape[-1], family, query.dtype)
        if not (family.takes_right(width, config.block_k)
                and family.takes_right(config.block_k, value_columns)):
            return False
        # A paged call's key tiles (the launch's, its natural rewalk's) whole in its pages or its pages whole
        # in them (`_pages_hold`).
        tiles = (config.block_k, config.relaunch_k or config.block_k)
        page = _page(ba)
        return page is None or all(_pages_hold(page, tile, family) for tile in tiles)

    def derived(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> Tiles | None:
        """Its tiles from the facts (`_derived`): each consumer a unit of queries, the key tile the largest
        its registers and the block hold, two stages; the family's consumers taking turns, a step's softmax
        running under the other's products, the training forward's too (its anchored walk's step is the
        plain one's, TWOPASS4-h100: 1.05 ms at 64 x 64 on two consumers against 1.41 at 64 x 128 on one,
        causal llama-d128 2 x 4096 on an H100 PCIe). Its family the first of the device's bundles, ranked
        by the share of their rows its queries fill (`layouts`: a decode step's few rows mma.sync's before
        wgmma's, row 4042), that takes the call and derives a config: a config legal with its natural
        rewalk's tiles (`_relaunched`), so a tile whose rewalk no tile holds gives way to a smaller (a 48-row
        training walk's scaled op on an A100 at width 32, legal at 48 x 64 and its rewalk not, derived none
        where 32 x 64 holds both, et-why48). A config is legal as it launches, its rewalk's tiles named
        (`_launchable`), so a compile's refusal of it (`report.refused`, keyed by the config the launch
        traced) steps its derivation: read at its own tiles alone, the refusal reached nothing and a B200's
        spilling 4B forward compiled twice, the same program (PF03)."""
        query = ba.arguments["query"]
        for family in layouts(target.generation, query.dtype, query.shape[1]):
            if self.refusal(ba, family) is not None:
                continue
            for columns in _column_blocks_of(family, query.dtype, ba.arguments["value"].shape[3]):
                def tiles(rows: int, walked: int, consumers: int, family=family, columns=columns) -> Tiles:
                    return Tiles(rows, walked, 2, consumers, family=family.name, columns=columns)

                def fits(config: Tiles) -> bool:
                    return self._launchable(ba, config, target) is not None

                config = _derived(family, query.dtype, query.shape[1], _keys(ba), tiles, fits,
                                  overlapped=True, walks=walked_extents,
                                  least_walked=functools.partial(_least_keys, family))
                if config is not None:
                    return self._launchable(ba, config, target)
        return None

    def _launchable(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget
                    ) -> Tiles | None:
        """`config` as it launches, its natural rewalk's tiles named (`_relaunched`), where that is legal
        (`legal`: a compile's refusal of the launch among its rules, keyed by this config); None where it is
        not or no tile holds the rewalk."""
        if not self._launch_legal(ba, config, target):
            return None
        launched = self._relaunched(ba, config, target)
        return launched if launched is not None and self.legal(ba, launched, target) else None

    def streaming(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget) -> Tiles:
        """`config` for a call whose walk the device's memory bounds (`streams`), its key tile the least its
        ring still streams at: a program pays its ring's first `stages` tiles before its first product, and
        a launch of more programs than the device runs at once pays it a program, so the least tile pays
        least, where its resident programs' rings in flight (`_staged_keys`) hold the bytes a
        multiprocessor's share of the device's stream keeps in flight (`GpuTarget.memory_rate` over its
        multiprocessors, for the part's measured `Measured.fixed`):
        a decode step of 128 rows over 4096 keys took 0.70 ms at 64-key tiles against 0.73 at 128 on an
        H100, and of 32 over 32768 1.35 against 1.38, its 128-wide K and V 32 KB a stage, two of them past
        the 45 KB a program keeps in flight (ET-STAGES2-h100).
        Its natural rewalk's tiles derived again (`_relaunched`); where no tile dividing the call's keys keeps
        the stream in flight, the largest that divides them, else `config` itself, as where the part's
        stream is unmeasured."""
        in_flight = self._in_flight(ba, config, target)
        if in_flight is None:
            return config
        query = ba.arguments["query"]
        family = bundle_of(config, query.dtype)
        # Its key tiles dividing the call's keys alone: a tile past them pads the whole cache, its keys and
        # values copied a step (`padded_call`; a 96-key tile over 32768, 383 us a step on an A100,
        # M-DD-5 W16, as long as its launch). A multiprocessor's bytes in flight are its resident programs'
        # together (`resident`): a larger tile holding fewer of them keeps no more in flight (an A100's
        # 16-row d64 decode at 256-key tiles staged 146 KB, one program a multiprocessor, 1.17x the hand
        # kernel's five resident programs of 20 KB, M-DD-7 W34).
        dividing = [launched for launched in (
                        self._launchable(ba, dataclasses.replace(config, block_k=extent), target)
                        for extent in reversed(walked_extents(family, query.dtype, config.block_k))
                        if self.keys % extent == 0)
                    if launched is not None]
        for candidate in dividing:
            held = self.resident(ba, candidate, target)
            if held * self._staged_keys(ba, candidate) >= in_flight:
                return candidate
        # Where none keeps the stream in flight, the largest dividing tile before a derived one that does not
        # divide the keys: a GH200's bounded decode at 192-key tiles over 32768 padded its cache, 84.5 of its
        # 134 us a step at width 128 (M-ET-70), where its 128-key walk alone is 49.5.
        if self.keys % config.block_k and dividing:
            return dividing[-1]
        return config

    def streamed(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget) -> int:
        """The programs a multiprocessor runs at once to keep its share of the device's stream in flight at
        `config` (`streaming`'s bytes in flight over a program's staged keys and values), at most those it
        holds (`resident`); those it holds where the stream is unmeasured. A memory-bound launch of more
        programs a multiprocessor streams no faster: a split decode's slices past them only lengthen its
        merge (an H100's B1 x 32768 decode at width 64 cut into 82 slices, five programs a multiprocessor
        where one keeps its stream, took 90.7 us against the hand kernel's 28.7, M-DD-4 W56)."""
        resident = self.resident(ba, config, target)
        in_flight = self._in_flight(ba, config, target)
        if in_flight is None:
            return resident
        return max(1, min(resident, math.ceil(in_flight / self._staged_keys(ba, config))))

    def _in_flight(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget
                   ) -> float | None:
        """The bytes a multiprocessor keeps in flight to stream its share of the device's memory rate for a
        call its memory bounds (`streams`: `GpuTarget.memory_rate` over its multiprocessors, for the part's
        measured `Measured.fixed`); None where the call is not so bound or the part's stream is
        unmeasured."""
        rate, fixed = target.memory_rate, target.fixed_seconds
        if not (streams(ba.arguments["query"], target) and rate and fixed and target.multiprocessors):
            return None
        return rate / target.multiprocessors * fixed

    def _staged_keys(self, ba: tokamax.BoundArguments, config: Tiles) -> int:
        """The bytes of keys and values a program's ring keeps in flight at `config`: its key tiles' rows of
        both, at the widths its family stages them (`_width`), in every slot but the one its walk reads
        while the others load (`stages - 1`): counted at all `stages`, an H100's decode step at 8 rows over
        32768 keys, width 64, was cut into slices of two programs a multiprocessor and streamed 69% of the
        device's memory rate where cuDNN's streamed 89% (232 against 181 us, M-ET-49b-h100)."""
        query, value = ba.arguments["query"], ba.arguments["value"]
        family = bundle_of(config, query.dtype)
        itemsize = jnp.dtype(query.dtype).itemsize
        per_key = sum(_width(array.shape[3], family, query.dtype) for array in (query, value)) * itemsize
        return max(1, config.stages - 1) * config.block_k * per_key

    def _relaunched(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget
                    ) -> Tiles | None:
        """`config` with its natural rewalk's tiles, a fold's (`Tiles.relaunch`): the largest key tile and the
        most consumers that hold its own registers, each dividing the config's (et-fwdhooks3: at the fold's
        tiles it spilled 2 KB at width 64); a call that keeps its logits natural rewalks nothing. None where
        no tile holds the rewalk."""
        config = dataclasses.replace(config, relaunch_k=None, relaunch_consumers=None, relaunch_columns=None)
        if self._unrewalked(ba):
            return config
        query, value = ba.arguments["query"], ba.arguments["value"]
        family = bundle_of(config, query.dtype)
        # Its value columns whole first, then in blocks (`_column_blocks_of`): a launch a block (row 2526: at
        # width 256 no tile held the anchored rewalk's whole sums).
        relaunches = (dataclasses.replace(config, block_k=block, consumers=consumers, natural_form=True,
                                          columns=columns)
                      for columns in _column_blocks_of(family, query.dtype, value.shape[-1])
                      for consumers in range(config.consumers, 0, -1)
                      for block in walked_extents(family, query.dtype, config.block_k)
                      if config.block_k % block == 0)
        relaunch = next((tiles for tiles in relaunches if self._launch_legal(ba, tiles, target)), None)
        if relaunch is None:
            return None
        return dataclasses.replace(config, relaunch_k=relaunch.block_k, relaunch_consumers=relaunch.consumers,
                                   relaunch_columns=relaunch.columns)

    def neighbours(self, ba: tokamax.BoundArguments, config: Tiles) -> Sequence[Tiles]:
        """`_Mainloop.neighbours`', each with its own natural rewalk's tiles (`_relaunched`), not the
        derived config's, which need not divide it (row 2584)."""
        target = generation.gpu_target()
        found = (self._relaunched(ba, neighbour, target) for neighbour in super().neighbours(ba, config))
        return tuple(neighbour for neighbour in found if neighbour is not None)

    def _natural(self, query: jax.Array) -> bool:
        """Whether the call of `query` keeps its logits natural, as its launch decides (`algebra.folds`, at
        the scale and power of two its queries take, `prescale`): its one form the natural one, no fold to
        rewalk."""
        power, scale = prescale(query.dtype, query.shape[3], self.scale)
        return not algebra.folds(scale, power, natural=self.natural, score_mod=self.score_mod)

    def _unrewalked(self, ba: tokamax.BoundArguments) -> bool:
        """Whether the call walks no natural rewalk: its logits natural (`_natural`), or a fold no logit's
        reach passes (`algebra.Forward.reaches`: an inference walk's subtracted fold of raw products, which a
        training forward does not take: its natural rewalk at tiles derived for it, an fp32 call's parts'
        at width 128 a block of the value columns, where the fold's whole width staged 280,672 bytes),
        at the scale its launch takes (`attention_mosaic_forward._prepared`)."""
        query = ba.arguments["query"]
        if self._natural(query):
            return True
        power, scale = prescale(query.dtype, query.shape[3], self.scale)
        taken = power if self.form != "spans" else 1.0
        subtracted = self.subtracted and ba.arguments.get("residuals") is not Kept.SPLIT
        return not algebra.Forward(self.form, self.keys, ids=False, scale=scale * taken,
                                   score_mod=self.score_mod, natural=self.natural, prescale=power / taken,
                                   subtracted=subtracted).reaches

    def staged_bytes(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget) -> int:
        """The program's shared memory (`_forward_allocations`, `resources.shared_bytes`), at the call's
        widths as the launch stages them (`_width`)."""
        family = bundle_of(config, ba.arguments["query"].dtype)
        query, value = ba.arguments["query"], ba.arguments["value"]
        width, value_width = (_width(array.shape[-1], family, query.dtype) for array in (query, value))
        # A launch of a block of the value columns (`Tiles.columns`) stages that block.
        value_width = config.columns or value_width
        steps = _padded(query.shape[1], _keys(ba), config)[1] // config.block_k
        out_dtype = None if self.out_dtype is None else jnp.dtype(self.out_dtype)
        # A paged call's key tile of several pages loaded as its launch loads it (`paged_reads`).
        page = _page(ba)
        allocations = _forward_allocations(family, config, width, value_width, query.dtype,
                                           ba.arguments.get("residuals") is Kept.SPLIT,
                                           min(config.stages, steps), out_dtype=out_dtype, page=page,
                                           codes=_codes_of(ba.arguments["key"], ba.arguments.get("scales")),
                                           parts=self.parts)
        # An inference launch may flag its rows for the fold's rewalk, and a cache's scaled walk reads its
        # keys' value magnitudes from its value slots (`algebra.SCALED_WALK`; another scaled walk the call's,
        # `attention_mosaic_forward._largest`), each a reduction across warps (`reducing`).
        flagged = ba.arguments.get("residuals") is not Kept.SPLIT
        arguments = ba.arguments
        cached = self.cached(arguments.get("table"), arguments.get("sources"), arguments.get("scales"))
        return resources.shared_bytes(family, launch_scope(family, allocations, reducing(
            family, config.consumers, self.score_reduces or self.output_mod is not None, flagged=flagged,
            magnitudes=cached and self.scales_in_kernel(ba)),
            products=forward_products(config, value_width, query.dtype, parts=self.parts)))

    def cached(self, table, sources, scales) -> bool:
        """Whether a call walks a cache's keys (masked, split, paged, by rows or over codes: a decode step's
        or a serving step's, which the memory bounds), whose kernel scales its rows' sums every tile where
        they could pass fp32's largest (`algebra.SCALED_WALK`): a pass over its values before it, to launch
        it unscaled, would cost the step's bandwidth. Every other call's common kernel holds no scale, a call
        some row of which can be scaled launching the scaled op's (`scaled`, `_fwd`)."""
        return (self.masked or self.splits > 1 or table is not None or sources is not None
                or scales is not None)

    def scales_in_kernel(self, ba: tokamax.BoundArguments) -> bool:
        """Whether the call's kernel walks scaled (the `scaled` op's, a `cached` call's; values that can pass
        fp32's largest, no rounded weights): it reads its keys' value magnitudes and holds its small weights'
        tier (`algebra.at_scale`)."""
        arguments = ba.arguments
        return self._scales(arguments.get("table"), arguments.get("sources"), arguments.get("scales"),
                            _computed(ba))

    def _scales(self, table, sources, scales, computed) -> bool:
        """`scales_in_kernel` of a call of `table`, `sources` and `scales`, its values read in `computed`."""
        return ((self.scaled or (self.cached(table, sources, scales) and not self.bounded))
                and self.rounds is None and algebra.headroom(computed))

    def registers(self, ba: tokamax.BoundArguments, config: Tiles, *, peaks: bool = False,
                  columns: int | None = None) -> int | None:
        """A consumer's registers at `config`: its walk's (`forward_registers`) over the values' columns
        its launch walks (`columns`, a block of them; the config's own by default), or its peak walk's
        (`peaks`, `peaks_registers`), which a training call takes first; None on a family that budgets
        none (`Family.budgets_registers`: a TPU core's values spill to VMEM), which holds every config.
        Its launches state the same at their own tiles as they lower (`registered`)."""
        arguments = ba.arguments
        registered = self.registered(arguments["query"], arguments["value"], arguments["score_params"],
                                     arguments.get("residuals"), arguments.get("table"),
                                     arguments.get("sources"), arguments.get("scales"),
                                     arguments.get("initial"))
        return registered.at(bundle_of(config, arguments["query"].dtype), config, peaks=peaks,
                             columns=columns)

    def registered(self, query, value, score_params, residuals, table=None, sources=None,
                   scales=None, initial=None) -> ForwardRegisters:
        """The call's facts its launches' registers read (`ForwardRegisters`): an `initial` state merges past
        an inference walk's keys (`attention_mosaic_forward._started`), a split training walk starting from
        it."""
        params = tuple(zip(self.score_kinds, (kept_dtype(param.dtype) for param in score_params),
                           strict=True))
        computed = query.dtype if scales is not None else value.dtype
        return ForwardRegisters(query.dtype, query.shape[-1], value.shape[-1], residuals is Kept.SPLIT,
                                self._natural(query), params, self.parts,
                                self._scales(table, sources, scales, computed), codes=scales is not None,
                                positions=math.ceil(query.shape[1] / self.per_span)
                                if self.form == "spans" and self.per_span > 1 else 0, per_span=self.per_span,
                                merged=initial is not None and residuals is not Kept.SPLIT,
                                started=initial is not None and residuals is Kept.SPLIT)


    def refusal(self, ba: tokamax.BoundArguments, family: mosaic.Family) -> Refusal | None:
        query, value = ba.arguments["query"], ba.arguments["value"]
        hooks = self._hooks(ba.arguments["score_params"], ba.arguments["output_params"],
                            ba.arguments["initial"], ba.arguments["drop"])
        if self.rounds is not None and (self.parts > 1 or ba.arguments["residuals"] is not Kept.SPLIT
                                        or ba.arguments["table"] is not None):
            # A quantized model's weights round at the row's maximum, which the anchored walk alone knows.
            return ROUNDED
        if self.parts > 1 and (ba.arguments["table"] is not None or ba.arguments["sources"] is not None
                               or ba.arguments["residuals"] is Kept.SPLIT or hooks.drop is not None):
            # A training forward's natural rewalk of parts spills past its model (an H100's at 64 x 64, width
            # 64: 48 bytes on two consumers, 24 on one): the Pallas-Triton mainloop's.
            return Refusal('dtype', "the Mosaic mainloop's fp32 parts walk an inference call over dense "
                           "operands, no dropout")
        if self.parts > 1 and not loads_parts(family):
            return Refusal('dtype', f"the Mosaic mainloop's fp32 parts load in pieces, which {family.name}'s "
                           "pipeline does not")
        return refused(query.dtype, query.shape[3], value.shape[3], hooks, backward=False, family=family)


def _pages_hold(page: int, tile: int, family: mosaic.Family) -> bool:
    """Whether a paged call's key tiles of `tile` keys lie in its pages of `page`: whole tiles a page, each
    read where it lies (`attention_mosaic_forward._paged_views`), or whole pages a tile, any page, its rows
    gathered through the table where `family` gathers rows (`attention_mosaic_forward.paged_reads`)."""
    return page % tile == 0 or (tile % page == 0 and family.gathers_rows)


def streams(query, target: generation.GpuTarget) -> bool:
    """Whether a call's walk the device's memory bounds: the products its keys and values feed, a query row
    of `query`'s `[B, Q, H, D]` each (4 flops a key entry, its logit's and its weighted value's), fewer a byte
    of them than the device's 16-bit products a byte its memory streams, both the cost model's device rates
    (`cost.rates`: the tensor rate its instruction family, clock and multiprocessors derive, no catalog's
    peak, row 3799): a decode step's few rows a key head, never a prefill's. False where either rate is
    unknown."""
    tensor, *_, memory = cost.rates(target, device=True)
    if not (tensor and memory):
        return False
    return 2 * query.shape[1] / jnp.dtype(query.dtype).itemsize < tensor / memory


def _keys(ba: tokamax.BoundArguments) -> int:
    """A call's keys, from its arrays: its key rows, or a paged call's table's slots, its pages a row times
    its pools' page (`forward`'s `table`)."""
    page = _page(ba)
    return ba.arguments["key"].shape[1] if page is None else ba.arguments["table"].shape[1] * page


WARPGROUP_WARPS = 4
"""The warps of a consumer role, a warpgroup's."""


def _computed(ba: tokamax.BoundArguments):
    """The dtype a call's values are read in: the query's where they are quantized codes (their `scales`
    given, `_codes_of`), their own otherwise."""
    query, value = ba.arguments["query"], ba.arguments["value"]
    return query.dtype if ba.arguments.get("scales") is not None else value.dtype


def _page(ba: tokamax.BoundArguments) -> int | None:
    """A paged call's keys a page, its pools' own (`forward`'s `table`); None for a call of rows of keys."""
    return None if ba.arguments.get("table") is None else ba.arguments["key"].shape[2]


@overload
def forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
            *, scale: float | None = None, natural: bool = NATURAL,
            residuals: Literal[Kept.NONE] = Kept.NONE, interpret: bool = False, out_dtype=None,
            masked: bool = False, splits: int = 1, table: jax.Array | None = None,
            sources: jax.Array | None = None, subtracted: bool = False,
            scales: tuple[jax.Array, jax.Array] | None = None, scaled: bool = False,
            bounded: bool = False) -> jax.Array:
    ...


@overload
def forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
            *, scale: float | None = None, natural: bool = NATURAL,
            residuals: Literal[Kept.SPLIT, Kept.OUTPUT], interpret: bool = False,
            out_dtype=None, masked: bool = False, splits: int = 1, table: jax.Array | None = None,
            sources: jax.Array | None = None, subtracted: bool = False,
            scales: tuple[jax.Array, jax.Array] | None = None, scaled: bool = False, bounded: bool = False
            ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, Split]:
    ...


def forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
            *, scale: float | None = None, natural: bool = NATURAL, residuals: Kept = Kept.NONE,
            interpret: bool = False, out_dtype=None, masked: bool = False, splits: int = 1,
            table: jax.Array | None = None, sources: jax.Array | None = None, subtracted: bool = False,
            scales: tuple[jax.Array, jax.Array] | None = None, scaled: bool = False, bounded: bool = False):
    """The template's forward on this mainloop (`MosaicForward`): `query` `[B, Q, H, D]` over `key`
    `[B, K, Hk, D]` and `value` `[B, K, Hk, Dv]`, and the keys `spans` leaves each query under `hooks`, as
    `dew.nn.kernels.templates.attention.reference` computes it; with `residuals` `Kept.SPLIT` the training
    forward's, its output with the backward's residuals as every mainloop's training forward gives them
    (`attention_portable.xla_mainloop(residuals=True)`): the maxima and denominators `[B, H, Q]`, each query's
    anchor key `[B, H, Q]` int32 and the split sums (`Split`); `Kept.OUTPUT` the same residuals of the
    inference walk, which keeps its output for the backward's D in place of an anchor (`output_split`,
    `takes_output_residuals`). `natural` keeps the logits natural
    (`algebra.Forward.base2`); `interpret` runs it on the CPU (Mosaic GPU's interpret mode), as a GPU of the
    family `generation.assumed` names; `out_dtype` the inference walk's output's (`MosaicForward.out_dtype`).
    A paged call's `key` and `value` are its pools `[Hk, P, page, D]` and `table` its page table `[B, pages
    a row]` int32: row `b`'s key `s` at offset `s % page` of page `table[b, s // page]`. With `sources`
    `[B]` int32 (a call of rows of keys, inference), batch row `b` reads key row `sources[b]` of `key` and
    `value` `[R, K, Hk, D]`: a serving step's work list over its cache's rows, no row copied. `subtracted`
    an inference walk's fold of raw products (`MosaicForward.subtracted`). With `splits`, each query tile's
    keys are cut into that many slices, every output the slices' own along the query axis
    (`MosaicForward.splits`), which XLA merges (`attention_decode`). With `scales` (an inference walk's),
    `key` and `value` are quantized codes, int8 or an FP8, and `scales` the keys' and the values'
    `[B, Hk, K]` fp32, one a key and key head in its row's key order: the call reads the keys and values they
    dequantize to, each converted to the query's dtype in its program (`_Codes`). `scaled` launches the
    scaled op whatever the call's values (`MosaicForward.scaled`: a mainloop whose own kernel holds no scale
    gives it a call some row of which can be scaled); `bounded` the unscaled
    one, a cache's call too, where its caller bounded its values (`MosaicForward.bounded`).
    """
    if residuals is Kept.SPLIT:
        # The training walk reads an interval a query (`KeySpans.per_query`); the inference walk by position.
        spans = spans.per_query()
    op, args = _forward_call(query, key, value, spans, hooks, scale=scale, natural=natural,
                             out_dtype=out_dtype, masked=masked, splits=splits, table=table, sources=sources,
                             subtracted=subtracted, scales=scales, scaled=scaled, bounded=bounded)
    return op(*args, residuals=residuals, interpret=interpret)


def forward_declined(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans,
                     hooks: Hooks | None = None, *, scale: float | None = None, natural: bool = NATURAL,
                     residuals: Kept = Kept.NONE) -> Refusal | None:
    """Why `forward` runs no call of these arrays compiled on this device, or None
    (`MosaicForward.declined`)."""
    spans = spans.per_query() if residuals is Kept.SPLIT else spans
    op, args = _forward_call(query, key, value, spans, hooks, scale=scale, natural=natural)
    return op.declined(op.bind(*args, residuals=residuals, interpret=False), generation.gpu_target())


def certified(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
              *, scale: float | None = None, natural: bool = NATURAL, interpret: bool = False):
    """The forward's certify form at the tiles its call derives (`MosaicForward.derived`), its walk the plain
    one (no P·V a step late): those tiles, and `certified_walk`'s output and its state after each of their key
    tiles. Off in production; tests/attention_witness.py holds each state to one step's enclosure of the one
    before. A call no config holds raises the reason the forward declines it (`MosaicForward.declined`: an
    unqualified family's, or the device's block and registers)."""
    op, args = _forward_call(query, key, value, spans, hooks, scale=scale, natural=natural)
    ba = op.bind(*args, residuals=Kept.NONE, interpret=interpret)
    target = generation.gpu_target()
    config = op.derived(ba, target)
    if config is None:
        raise ValueError(f"the Mosaic forward declines the call: {op.declined(ba, target)}")
    arguments = ba.arguments
    registered = op.registered(arguments["query"], arguments["value"], arguments["score_params"], Kept.NONE)
    return config, certified_walk(query, key, value, spans, hooks or Hooks(), scale=op.scale,
                                  natural=op.natural, tiles=config, interpret=interpret,
                                  registered=registered)


def _forward_call(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None, *,
                  scale: float | None, natural: bool, out_dtype=None, masked: bool = False,
                  splits: int = 1, table: jax.Array | None = None, sources: jax.Array | None = None,
                  subtracted: bool = False, scales: tuple[jax.Array, jax.Array] | None = None,
                  scaled: bool = False, bounded: bool = False) -> tuple[MosaicForward, tuple]:
    """`forward`'s op and its arguments, a dropping call's draw among them, which its legality rule refuses
    (`refused`): the op it launches, `scaled` or `bounded` as the call names (`MosaicForward.scaled`,
    `MosaicForward.bounded`), so a plan of it derives that launch's config and reads that launch's
    refusals (`attention_decode.plan`)."""
    hooks = Hooks() if hooks is None else hooks
    fp32 = all(jnp.dtype(x.dtype) == jnp.dtype(jnp.float32) for x in (query, key, value))
    parts = product.EXACT_PARTS if fp32 else 1
    if parts > 1:
        # fp32 operands as their bf16 parts, stacked along the batch axis (`MosaicForward.parts`); the
        # output fp32, the call's. Its fold's logits a difference of raw products scaled after it
        # (`MosaicForward.subtracted`): no reach, so no natural rewalk, whose guarded walk of parts held no
        # registers at width 64 (8 bytes spilled on an H100, et-fp32regs7).
        query, key, value = (jnp.concatenate(product.formed(x, parts), axis=0) for x in (query, key, value))
        out_dtype = jnp.float32 if out_dtype is None else out_dtype
        subtracted = True
    op = MosaicForward(score_mod=hooks.score_mod, output_mod=hooks.output_mod,
                       score_kinds=tuple(param.kind for param in hooks.score_params),
                       output_kinds=tuple(param.kind for param in hooks.output_params),
                       score_dtypes=tuple(kept_dtype(param.value.dtype) for param in hooks.score_params),
                       output_dtypes=tuple(kept_dtype(param.value.dtype) for param in hooks.output_params),
                       score_reduces=hook_reduces(hooks), output_width=power_of_two(value.shape[3]),
                       form=spans.form, keys=spans.keys,
                       scale=logit_scale(query.shape[3], scale), natural=natural,
                       out_dtype=None if out_dtype is None else jnp.dtype(out_dtype).name, masked=masked,
                       splits=splits, subtracted=subtracted, per_span=spans.per_span,
                       parts=parts, rounds=hooks.rounds, scaled=scaled, bounded=bounded)
    return op, (query, key, value, spans.start, spans.end, spans.query_ids, spans.key_ids,
                tuple(param.value for param in hooks.score_params),
                tuple(param.value for param in hooks.output_params), hooks.initial, hooks.drop,
                table, sources, scales)


@dataclasses.dataclass(frozen=True)
class MosaicBackward(_Mainloop[BackwardTiles]):
    """The template's backward on this mainloop (`gradients`) over the call's own arrays (`backward` gives
    them): its queries, keys, values and output cotangent, its spans' arrays, its score hook's parameters, its
    initial state (`Initial`, or None) and a training forward's residuals (`forward`'s). Returns its
    gradients (`Gradients`). Its config comes from the family's facts and the call's extents (`derived`: one
    launch for a fast-form call whose score hook takes no parameters, two in deterministic mode, for the
    accurate form and for a call whose hook does), legal within the device's block and a consumer's
    registers (`legal`)."""

    HOOKS = ("score_mod",)
    DIFFERENTIATED = True
    SPANS = 4
    """Its arrays' spans after the queries, keys, values and the output's cotangent (`_fwd`)."""
    config_cls = BackwardTiles
    WALKED = "block_q"
    accurate: bool = ACCURATE
    rounded: bool = True
    score_slopes: tuple[int, int] = (0, 0)
    """The tiles its score hook's pullback forms that vary in its dQ launch and its dK/dV launch
    (`varying_slopes`), as its builder reads them off the hook (`_backward_call`): a keyed launch's hook is
    its fingerprint, which keeps no trace of them."""
    score_reads: tuple[bool, bool] = (False, False)
    """Whether its score hook reads its queries' places and its keys' (`score_places`), as its builder reads
    them off the hook (`_backward_call`): a keyed launch's hook is its fingerprint, which keeps no trace of
    them, and its registers hold each it reads (`backward_registers`' `hooked`)."""

    def _fwd(self, query, key, value, dout, start, end, query_ids, key_ids, score_params, initial, residuals,
             drop=None, *, interpret: bool, config: BackwardTiles, return_residuals: bool = False):
        family = bundle_of(config, query.dtype)
        spans = self.spans(start, end, query_ids, key_ids)
        hooks = Hooks(self.score_mod, tuple(map(Param, score_params, self.score_kinds)), initial, drop=drop)
        if reason := refused(query.dtype, query.shape[3], value.shape[3], hooks, backward=True,
                             family=family):
            raise NotImplementedError(reason)
        # The call's query count where the launch pads past it (`algebra.Forward.queries`).
        queries = start.shape[1]
        padded_queries, padded_keys = _padded(queries, key.shape[1], config)
        # The form and geometry; the launch takes the scale and the query's power `gradients` gives it. Its
        # walks reach the padded keys (a common walk's holder wholly past the call's, `seen`).
        spec = algebra.Forward(spans.form, spans.keys, ids=query_ids is not None, scale=self.scale,
                               score_mod=self.score_mod, natural=self.natural, initial=initial is not None,
                               queries=queries if padded_queries > queries else None,
                               walked_keys=padded_keys if padded_keys > spans.keys else None)
        launch = functools.partial(_launch_backward, spec=spec, accurate=self.accurate, tiles=config,
                                   family=family, interpret=interpret,
                                   drop=placed_draws(drop, padded_queries, padded_keys), rounded=self.rounded,
                                   registered=self.registered(query, value, score_params, drop, config))
        return gradients(launch, query, key, value, spans, hooks, dout, residuals,
                         block=_block(config), scale=self.scale, rounded=self.rounded), None

    UNBUNDLED: ClassVar[str] = ("the Mosaic backward runs on no bundle of {name}'s: its launches scope no "
                                "issuer's product scratch, which tcgen05's chunks keep (row 4839, its "
                                "tensor-memory form)")

    @staticmethod
    def _bundle(generation_name: str | None, dtype) -> mosaic.Family | None:
        """The bundle the backward's dQ launch runs on (`backward_bundle`)."""
        return backward_bundle(generation_name, dtype)

    @staticmethod
    def _layouts(generation_name: str | None, dtype, rows: int | None = None) -> tuple[mosaic.Family, ...]:
        """The bundles the backward's launches take (`backward_layouts`), whatever its rows."""
        del rows
        return backward_layouts(generation_name, dtype)

    def legal(self, ba: tokamax.BoundArguments, config: BackwardTiles, target: generation.GpuTarget) -> bool:
        # The one launch adds each tile's dQ under the first pass's D: the fast form's, a call's of equal
        # head widths, its dK/dV program queries first on the dQ launch's layout or keys first on
        # one whose consumers share a key block (`split_queries`), each consumer's share of a slot's queries
        # a warpgroup MMA's rows (dQ's product's M). A score hook's and an initial state's only
        # queries first (at width 256 its consumers sharing a key block, `attention_mosaic_coop`), and a
        # hook of no parameters, whose gradients no launch writes (Gemma 2's softcap, FlashAttention 3's);
        # the state's (gpt-oss's sinks) the first pass's D and the keys' share of the weights give
        # (`gradient.split_mass`), its weights' denominator the forward's. A dropping call's only queries
        # first, each consumer its own keys and its own draw, under the dropping call's D (FlashAttention
        # 2's).
        hooked, dropped = bool(self.score_kinds), ba.arguments["drop"] is not None
        if config.fused and (self.score_mod is not None or ba.arguments["initial"] is not None or dropped
                             ) and not config.queries_first:
            return False
        if config.fused and dropped and config.shared_keys:
            return False
        found = {family.name: family
                 for family in self._layouts(target.generation, ba.arguments["query"].dtype)}
        dkv_family = found.get(config.dkv.family)
        # A family the device has no bundle of (a pin of an H100's wgmma on an A100) is not read here: the
        # mainloop's legality refuses it (`_Mainloop.legal`).
        if dkv_family is None or not whole_tiles(dkv_family, config.dkv):
            return False
        # The one launch's tiles a role's rows at least (`_role_rows`): no issuer specializes it below them
        # (`fewer_rows` the two launches' alone), its 32 x 32 tile on an sm120's fed mma.sync and an sm86's
        # mma.sync "cannot relayout" (et-sm120fused). The dQ launch streams through a producer
        # (`runs_backward`), which a family whose consumers feed their own rings has not: the one launch
        # there, queries first.
        if config.fused and min(config.block_q, config.block_k) < dkv_family.role_rows:
            return False
        dq_family = found.get(config.family)
        if not config.fused and dq_family is not None and split_queries(dq_family):
            return False
        keys_first = config.dkv_apart
        if config.fused and (ba.arguments["query"].shape[-1] != ba.arguments["value"].shape[-1] or (
                keys_first and not (split_queries(dkv_family)
                                    and (config.block_q // config.dkv.consumers) % dkv_family.rows == 0))
                or (config.staged_logits and not keys_first)):
            return False
        dtype = ba.arguments["query"].dtype
        # The one launch stages its weights and score cotangents for its products (`swizzling.staged`): fp16's
        # as two parts at the walk's running scale (`attention_mosaic_fused._sixteen_step`), queries first,
        # each consumer its own keys or its share of a block they share (`attention_mosaic_coop`'s, every
        # share's scale joined): row 3886, its two launches 4.1 times the bf16 step on an H100.
        sixteen = jnp.dtype(dtype) == jnp.dtype(jnp.float16) and not config.queries_first
        if ((config.fused and (not fuses(self.accurate, hooked) or sixteen))
                or not super().legal(ba, config, target)):
            return False
        family = bundle_of(config, dtype)
        # Deterministic mode takes no one launch: its programs add into dQ (and grouped dK and dV) across
        # programs, whose fixed order would wait on programs an ordinary launch need not hold resident
        # (row 4732: a ticket spins on a predecessor another stream's kernel keeps off the
        # multiprocessors). Its two launches' sums are each program's own, a key head's query heads'
        # summed in their order in XLA.
        if config.fused and deterministic_ops_requested():
            return False
        if (self.score_mod is not None and family is not None
                and min(config.block_q, config.block_k) < family.role_rows):
            # A score hook's tiles at a role's rows (`derived`'s fewer rows a consumer): a hooked call's tiles
            # of fewer fail to lower.
            return False
        if config.shared_keys:
            # Every consumer's share of the keys whole rows of a swizzle the exchange stages them at
            # (`swizzling.swizzle_for`), its share of the columns whole swizzled rows
            # (`attention_mosaic_coop`).
            share = _width(ba.arguments["query"].shape[-1], family, dtype) // config.consumers
            narrowest = family.narrowest_columns(dtype)
            return (config.fused and config.queries_first and config.consumers > 1 and config.columns == 1
                    and (config.block_k // config.consumers) % narrowest == 0
                    and share % family.column_unit(dtype) == 0)
        # A width split among programs adds its dQ once, in no fused launch, each share whole swizzled rows
        # and, where its family's products read their right operand's columns from a block staged at its
        # whole width (`Family.operands_in_registers`), whole swizzle tiles of that block: an A100's accurate
        # dQ under a score hook split 64 columns in two, its keys' 64-column tile sliced at 32, and failed to
        # lower ("slice size (32) divisible by the tile size (64)", FS-A100CHECK1d, acchook3).
        widths = [_width(ba.arguments[name].shape[-1], family, dtype) for name in ("query", "value")]
        def unit(width: int) -> int:
            if not family.operands_in_registers:
                return family.column_unit(dtype)
            tile = swizzling.swizzle_columns(dtype, swizzling.swizzle_for(dtype, width))
            return max(family.column_unit(dtype), tile)

        units = [unit(width) for width in widths]
        return config.columns == 1 or (not config.fused and all(
            (width / config.columns) % unit == 0 for width, unit in zip(widths, units, strict=True)))

    def derived(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> BackwardTiles | None:
        """Its tiles from the facts (`_derived`): each dK/dV consumer a unit of keys, the query tile both
        launches walk the largest their registers and the block hold, two stages, one consumer, whose step
        issues its products back to back (TRAINC-h100: 6.88 against 3.72 ms causal, 9.68 against 7.56 full,
        at 64 x 64 with two consumers and one). The fast form of a hookless call (or one whose hook takes no
        parameters, a softcap, or under an initial state, sinks) takes one launch where one holds the whole
        width, its dK/dV program queries first adding each tile's dQ atomically (FUSEDFORM-h100: 1.05 to 1.4
        times as fast at widths 64 and 128 over 1k to 8k tokens; ET-A05AB3-h100: 5.20 against 6.19 ms the two
        keys-first launches at Qwen3-8B causal, 1.19 against 1.53 at width 64;
        check 1 on drawn inputs and finite over every stability family it takes, FUSEDSTAB-h100; 384 MB of
        temporaries against 258 at llama-d128 2 x 4096, its dQ fp32), a call over key spans, windows,
        documents or runs as a plain one (ET-CROSS-h100 and ET-SPAN4-h100, the attention launches a training
        step: Qwen3-8B's heads over four documents 0.67 to 0.72 times the two launches' from 1 x 1024 to
        8192, under a causal window of 4096 at 8192 0.65; width 64's 8 heads over 2 0.56 to 0.88, 16 over
        16 under the window 0.54; check 1 over documents and a window at 1 x 2048; of every call measured,
        row 3171's toy one alone the slower, 1 x 1024 over three short documents and a window of 256,
        1.38 times: 20 us a step); otherwise (deterministic mode, `deterministic_ops_requested`; the
        accurate form, a hook with parameters, a width no fused launch holds) two launches, the dQ launch
        then the keys-first dK/dV one, in a fixed order."""
        family = self._family(ba, target)
        if family is None:
            return None
        query, key = ba.arguments["query"], ba.arguments["key"]

        def fits(config: BackwardTiles) -> bool:
            return self.legal(ba, config, target)

        derive = functools.partial(_derived, least_held=_role_rows)

        # The dK/dV launch on the layout whose consumers feed their rings where one is, its two consumers
        # splitting each slot's queries (`split_queries`); else on the dQ launch's.
        fed = [layout for layout in self._layouts(target.generation, query.dtype) if split_queries(layout)]
        # The dK/dV launch's consumers a backward's at most (`BACKWARD_CONSUMERS`), whatever its layout's.
        dkv = ({"dkv_family": fed[0].name, "dkv_consumers": min(fed[0].consumers, BACKWARD_CONSUMERS)}
               if fed else {})
        # The one launch, legal where the form, the hooks, the widths, the registers and the mode allow
        # (`legal`), the first that holds (design §7: legality steps down, nothing is ranked). Queries first
        # on the layout whose consumers feed their rings, each consumer its own keys over the query stream
        # it feeds (FlashAttention 3's CTA), at the layout's consumers, whom no producer's warpgroup shares
        # the registers with (ET-FFED3-h100, the fused launch: 2.40 against 2.71 ms causal and 4.32 against
        # 5.54 full at Qwen3-8B's widths; at width 64, 64 x 64 on two consumers 0.47 ms against 128 x 64 on
        # one, 0.59): another layout than the dQ launch's, whose own fed layout the next launches name, and
        # at two consumers or more, one's softmax under the other's products, which one consumer has not.
        for layout in (layout for layout in fed if layout.name != family.name):
            def fed_fused(rows: int, walked: int, consumers: int, layout=layout) -> BackwardTiles:
                return BackwardTiles(walked, rows, 2, consumers, fused=True, family=layout.name)

            config = derive(layout, query.dtype, key.shape[1], query.shape[1], fed_fused, fits,
                            overlapped=True)
            if config is not None and config.consumers > 1:
                return config
        # Else keys first on the dQ launch's layout, its dK/dV on the fed one where there is one, the logits'
        # queries a stream of their own, then staged a step, then on the dQ launch's layout alone.
        for shared in [*({**dkv, "staged_logits": staged} for staged in (False, True) if dkv), {}]:
            def fused(rows: int, walked: int, consumers: int, shared=shared) -> BackwardTiles:
                return BackwardTiles(walked, rows, 2, consumers, fused=True, family=family.name, **shared)

            if config := derive(family, query.dtype, key.shape[1], query.shape[1], fused, fits,
                                  overlapped=False):
                return config
        # Where no consumer's registers hold its own keys' whole dK and dV sums, the consumers share
        # one key block, each its share of the columns (FlashAttention 3's width 256: ET-COOP1-h100,
        # 0.68 ms against the two launches' 1.30 at Qwen3-8B's heads 256 wide, causal; where the sums
        # fit, sharing cost 9% at width 128 and 46% at 64, ET-COOP2-h100).
        for layout in fed:
            def sharing(rows: int, walked: int, consumers: int, layout=layout) -> BackwardTiles:
                return BackwardTiles(walked, rows, 2, consumers, fused=True, family=layout.name,
                                     shared_keys=True)

            if config := derive(layout, query.dtype, key.shape[1], query.shape[1], sharing, fits,
                                  overlapped=True):
                return config
        # The fewest dK/dV programs a key block's widths split among whose share of the sums its consumer's
        # registers hold (`BackwardTiles.columns`: a width of 256's whole dK and dV sums pass them).
        # Its logits' queries a stream of their own, else where that passes the block staged a step
        # (`BackwardTiles.staged_logits`). Where no tile of a role's rows holds the block, fewer rows a
        # consumer on a family whose warps arrange themselves by them (`fewer_rows`, `bundle_of`), each
        # launch's tiles that many rows a side (an A100's heads 256 wide: every 64-row tile's two stages of
        # queries and output cotangents beside its resident keys and values past its 166,912 bytes, the
        # two launches at 32 x 32 150,560, et-bwd256sm80).
        # Fewer rows a consumer for a call with no score hook: a hook's parameters and pullback tiles laid out
        # at a role's rows, sliced at fewer, fail to lower (an A100's accurate dQ under a temperature at 32 x
        # 32: "slice size (32) ... divisible by the tile size (64)", acchook3; FS-A100CHECK1d's row-indexed).
        # Whole columns at fewer rows before a column split at the role's: a split recomputes each step's
        # score products once a share, where fewer rows keep the work and halve the tile (an A100's
        # bf16 backward 128 wide at 32 x 32 whole took 0.82 to 0.88 of 64 x 64 in two shares, 1 x 8192
        # causal 36.1 against 41.7 ms; M-ET-29b, W91-a100).
        # Where its family's products complete before the next issues (mma.sync's, `products_overlap`), the
        # largest tile a multiprocessor holds two programs of first, one program's warps issuing under the
        # other's products (an A100's step at Qwen3-8B's heads, causal 2 x 8192: 32 x 32 at two programs
        # 56.65 ms against 64 x 32 at one 69.06, 2 x 2048 over 128 heads 17.43 against 20.33, M-ET-56).
        # The two launches' dQ launch streams through a producer: on the call's bundle, else on the first of
        # its layouts a producer fills (consumer Blackwell's, whose mma.sync bundle feeds its consumers'
        # rings first, `bundles.MmaProduced`).
        dq = next((layout for layout in (family, *self._layouts(target.generation, query.dtype))
                   if not split_queries(layout)), None)
        if dq is None:
            return None
        residing = [] if dq.products_overlap else [
            lambda config: fits(config) and self.resident(ba, config, target) >= 2]
        for wanted in (*residing, fits):
            for columns in (1, 2, 4):
                for rows in (None, *(fewer_rows(dq) if self.score_mod is None else ())):
                    walks = extents if rows is None else functools.partial(_sides, rows)
                    for staged in (False, True):
                        def tiles(rows: int, walked: int, consumers: int, columns=columns, staged=staged
                                  ) -> BackwardTiles:
                            return BackwardTiles(walked, rows, 2, consumers, family=dq.name,
                                                 columns=columns, staged_logits=staged, **dkv)

                        if config := derive(dq, query.dtype, key.shape[1], query.shape[1], tiles, wanted,
                                              overlapped=False, walks=walks, rows=rows):
                            return config
        return None

    def staged_bytes(self, ba: tokamax.BoundArguments, config: BackwardTiles,
                     target: generation.GpuTarget) -> int:
        """The larger of its launches' programs' shared memory (`_dq_allocations`, `_dkv_allocations`,
        `resources.shared_bytes`); a fused backward's dK/dV program's alone."""
        family = bundle_of(config, ba.arguments["query"].dtype)
        query, key, value = (ba.arguments[name] for name in ("query", "key", "value"))
        dtype = query.dtype
        width, value_width = (_width(array.shape[-1], family, dtype) for array in (query, value))
        padded_q, padded_k = _padded(query.shape[1], key.shape[1], config)
        # Each launch on its own layout (`BackwardTiles.dkv`).
        dkv_family = bundle_of(config.dkv, dtype)
        q_stages = min(config.stages, padded_q // config.block_q)
        if config.queries_first:
            # The fused launch reduces across no warps but a score hook's and fp16's running scale's
            # (`reducing`), its statistics and D its operands and its sums its products' (an A100's at
            # 64 x 64 reserves no reduction scratch, its lowering's 114,688 bytes the model's without it,
            # et-fused80).
            # The fused launch's dK and dV in the operands' dtype where it stores them so (`stores_typed`).
            reduced = reducing(family, config.consumers, self.score_reduces, scaled=staged_parts(dtype) > 1)
            allocations = (_coop_allocations if config.shared_keys
                           else functools.partial(_fused_allocations, reduced=reduced))
            typed = {} if config.shared_keys else {"typed": stores_typed(
                family, config, width, dtype, q_stages, grouped=query.shape[2] != key.shape[2],
                reduced=reduced, rounded=self.rounded)}
            launches = [(family, config, allocations(family, config, width, dtype, q_stages, **typed),
                         config.shared_keys)]
        else:
            launches = [(dkv_family, config.dkv, _dkv_allocations(
                dkv_family, config.dkv, width, value_width, dtype, stages=q_stages, columns=config.columns,
                tiled=Kind.TILE in self.score_kinds), True)]
        if not config.fused:
            launches.append((family, config, _dq_allocations(family, config, width, value_width, dtype,
                                                             min(config.stages, padded_k // config.block_k)),
                             True))
        # The fused launch's warpgroups reserve fp16's running scale's reduction scratch too, as it launches
        # (`reducing`'s `scaled`: `_fused_launch`'s and `_coop_launch`'s).
        scaled = config.fused and staged_parts(dtype) > 1
        return max(resources.shared_bytes(layout, launch_scope(layout, allocations, reducing(
            layout, tiles.consumers, self.score_reduces, scaled=scaled), reduces=reduces))
            for layout, tiles, allocations, reduces in launches)

    def registers(self, ba: tokamax.BoundArguments, config: BackwardTiles) -> int | None:
        # None on a family that budgets none (`Family.budgets_registers`), which holds every config.
        if not bundle_of(config, ba.arguments["query"].dtype).budgets_registers:
            return None
        dq, dkv = self._launch_registers(ba, config)
        return dkv if config.fused else max(dq, dkv)

    def _launch_registers(self, ba: tokamax.BoundArguments, config: BackwardTiles) -> tuple[int, int]:
        """Each launch's registers a consumer holds (`backward_registers`): the dQ launch's at `config` (none
        runs where `fused`), the dK/dV launch's at its own tiles (`BackwardTiles.dkv`), its consumers
        splitting a slot's queries where it runs keys first on a layout whose consumers feed their rings
        (`split_queries`), not the fused launch queries first on that layout (`_fused_launch`)."""
        arguments = ba.arguments
        registered = self.registered(arguments["query"], arguments["value"], arguments["score_params"],
                                     arguments["drop"], config)
        return registered.at(config, dq=True), registered.at(config.dkv, dq=False)

    def registered(self, query, value, score_params, drop, config: BackwardTiles) -> BackwardRegisters:
        """The call's facts its launches' registers read at `config` (`BackwardRegisters`)."""
        params = tuple(zip(self.score_kinds, (kept_dtype(param.dtype) for param in score_params),
                           strict=True))
        return BackwardRegisters(query.dtype, query.shape[-1], value.shape[-1], self.accurate, params,
                                 self.score_slopes, drop is not None, config.queries_first, self.score_reads)

    def _held(self, ba: tokamax.BoundArguments, config: BackwardTiles) -> bool:
        # Each launch's registers within its own layout's budget (`Family.registers`); every config where
        # the family budgets none (`Family.budgets_registers`).
        if not bundle_of(config, ba.arguments["query"].dtype).budgets_registers:
            return True
        dq, dkv = self._launch_registers(ba, config)
        held = dkv <= self.register_budget(config.dkv)
        return held and (config.fused or dq <= self.register_budget(config))


    def refusal(self, ba: tokamax.BoundArguments, family: mosaic.Family) -> Refusal | None:
        query, value = ba.arguments["query"], ba.arguments["value"]
        hooks = Hooks(self.score_mod, tuple(map(Param, ba.arguments["score_params"], self.score_kinds)),
                      ba.arguments["initial"], drop=ba.arguments["drop"])
        if split_queries(family) and not fuses(self.accurate, bool(self.score_kinds)):
            # A device whose backward bundle's consumers feed their own rings (a B200's, consumer
            # Blackwell's) runs the one launch alone, which this form does not take (`fuses`).
            form = "accurate form" if self.accurate else "hook"
            return Refusal('family', f"the backward's {form} takes two launches, whose dQ launch streams "
                           f"through a producer {family.name} has not")
        return refused(query.dtype, query.shape[3], value.shape[3], hooks, backward=True, family=family)


def backward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None,
             dout: jax.Array, *, residuals: tuple[jax.Array, jax.Array, jax.Array, Split],
             scale: float | None = None, natural: bool = NATURAL, accurate: bool = ACCURATE,
             interpret: bool = False, rounded: bool = True) -> Gradients:
    """The template's backward on this mainloop (`MosaicBackward`, `gradients`) from a training forward's
    `residuals` (its maxima, denominators, anchor keys and split sums): D about each query's anchor from the
    split sums (`gradient.split_delta`), a dQ launch and a dK/dV launch, and `dq_finish`'s refinement between
    them, in the form `accurate` names; or the fast form's one launch (`BackwardTiles.fused`,
    `_launch_backward`), the derived default of a call whose score hook takes no parameters
    (`MosaicBackward.derived`). Every
    call form of bf16 operands, under every score hook and the initial state. The output hook is no part of
    it: a differentiated call applies `output_mod` past the kernel, where its own derivative runs, and passes
    `dout` before it. `interpret` runs it on the CPU, as a GPU of the family `generation.assumed` names."""
    op, args = _backward_call(query, key, value, spans, hooks, dout, residuals, scale=scale, natural=natural,
                              accurate=accurate, rounded=rounded)
    return op(*args, interpret=interpret)


def backward_declined(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans,
                      hooks: Hooks | None, dout: jax.Array, *,
                      residuals: tuple[jax.Array, jax.Array, jax.Array, Split], scale: float | None = None,
                      natural: bool = NATURAL, accurate: bool = ACCURATE) -> Refusal | None:
    """Why `backward` runs no call of these arrays compiled on this device, or None
    (`MosaicBackward.declined`)."""
    op, args = _backward_call(query, key, value, spans, hooks, dout, residuals, scale=scale, natural=natural,
                              accurate=accurate)
    return op.declined(op.bind(*args, interpret=False), generation.gpu_target())


def _backward_call(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None,
                   dout: jax.Array, residuals: tuple, *, scale: float | None, natural: bool,
                   accurate: bool, rounded: bool = True) -> tuple[MosaicBackward, tuple]:
    """`backward`'s op and its arguments, a dropping call's draw among them (`_forward_call`), its spans an
    interval a query (`KeySpans.per_query`)."""
    hooks, spans = Hooks() if hooks is None else hooks, spans.per_query()
    op = MosaicBackward(score_mod=hooks.score_mod,
                        score_kinds=tuple(param.kind for param in hooks.score_params),
                        score_dtypes=tuple(kept_dtype(param.value.dtype) for param in hooks.score_params),
                        score_reduces=hook_reduces(hooks), form=spans.form, keys=spans.keys,
                        scale=logit_scale(query.shape[3], scale), natural=natural,
                        accurate=accurate, rounded=rounded, score_reads=score_places(hooks),
                        score_slopes=(varying_slopes(hooks, DQ_GRADS), varying_slopes(hooks, DKV_GRADS)))
    return op, (query, key, value, dout, spans.start, spans.end, spans.query_ids, spans.key_ids,
                tuple(param.value for param in hooks.score_params), hooks.initial, residuals, hooks.drop)
