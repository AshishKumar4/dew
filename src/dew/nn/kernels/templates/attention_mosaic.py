"""The attention template's Mosaic GPU mainloop as Template implementations (`MosaicForward`,
`MosaicBackward`; design §6 and §7): the forward, its training form and the backward over every call form,
each launch's one config derived from the instruction family's and the device's facts and the call's extents
(`derived`), legal where its programs stage within the device's block and a consumer's registers hold its live
set (`legal`): no table, no sweep, no space ranked. A GPU nobody measured of a family with a bundle here
(`bundle`) takes its own facts' config. The launches are `attention_mosaic_forward`'s and
`attention_mosaic_backward`'s; what they share, `attention_mosaic_program`'s."""

import dataclasses
import functools
from collections.abc import Callable, Hashable, Sequence
from typing import ClassVar, Literal, overload

import jax
import jax.numpy as jnp
import tokamax

from dew._devices import deterministic_ops_requested
from dew.nn.kernels import generation
from dew.nn.kernels.selection import Refusal
from dew.nn.kernels.templates import attention_algebra as algebra, cost, mosaic, product, resources
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
    certified_walk,
    guarded_columns,
    guarded_tiles,
)
from dew.nn.kernels.templates.attention_mosaic_fused import _fused_allocations, stores_typed
from dew.nn.kernels.templates.attention_mosaic_program import (
    BackwardTiles,
    Tiles,
    _block,
    _padded,
    _products,
    _width,
    bundle,
    bundle_of,
    extents,
    launch_scope,
    layouts,
    query_rows,
    query_swizzle,
    reducing,
    refused,
    split_queries,
    stagings,
    values_deferred,
    varying_slopes,
    walked_extents,
    whole_tiles,
)
from dew.nn.kernels.templates.base import Template, Unfitted


def _column_blocks_of(family: mosaic.Family, dtype, value_width: int) -> list[int | None]:
    """The value columns a forward launch walks at once, widest first: the whole (None), then halved while a
    block stays whole swizzled columns (`Family.column_unit`), each block a launch (`Tiles.columns`), for a
    call no tile holds whole (an fp32 call's parts at width 128 on an H100: 64 x 64 staged 283,744 bytes of
    its 232,448, one consumer, et-fp32derive)."""
    whole, unit = _width(value_width, family, dtype), family.column_unit(dtype)
    blocks = [whole // 2 ** halved for halved in range(1, whole.bit_length())
              if whole % 2 ** halved == 0 and (whole // 2 ** halved) % unit == 0]
    return [None, *blocks]


def _derived[Config: (Tiles, BackwardTiles)](family: mosaic.Family, dtype, held: int, walked: int,
                                                tiles: Callable[[int, int, int], Config],
                                                fits: Callable[[Config], bool], *,
                                                overlapped: bool, walks: Callable = extents) -> Config | None:
    """A launch's config from its family's facts and the call's extents (design §7), no space ranked: its
    consumers the family's (`Family.consumers`) where its step `overlapped` (a softmax between its products,
    which runs under its partner's products as the consumers take turns, FlashAttention-3's), else one (a
    step issuing its products back to back, which turns only make wait); fewer where the `held` rows a
    consumer accumulates (a forward's queries, a dK/dV launch's keys) fill fewer of the family's units; each
    consumer's rows the family's unit (`_extents`' least: the rows its products cover, `Family.role_rows`, in
    whole swizzled rows); the tile it walks the largest of its extents (`walks`: a backward's the family's
    unit, `extents`, its walked tiles' weights and cotangents feeding products from registers, which at fewer
    rows than the unit cross warps (fs-backward-specialized: mma.sync at 32-row shares, no relayout lowers);
    a forward's `walked_extents`, a product's depth in whole swizzled rows) within the `walked` extent whose
    live set a consumer's registers hold and
    whose two stages the block holds (`fits`), two the ring's busy depth, a slot loading while one is read;
    where its family's products read operands from registers, a tile its transposed streams staged at the
    family's own swizzle do not fit fits them staged narrowest (`Tiles.narrow`), a wide stage's bandwidth
    traded for the larger tile; fewer consumers where no tile of the family's count holds the block (a GPU
    of a smaller one). `tiles(rows, walked, consumers)` builds the config. None where no tile fits."""
    rows = query_rows(family, dtype, held)
    most = family.consumers if overlapped else 1
    for consumers in range(max(1, min(most, -(-held // rows))), 0, -1):
        for extent in walks(family, dtype, walked):
            for narrow in stagings(family):
                config = dataclasses.replace(tiles(rows, extent, consumers), narrow=narrow)
                if fits(config):
                    return config
    return None


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

    def padded(self, ba: tokamax.BoundArguments, config: Config) -> float | None:
        """The share of the score grid, each head's queries by its keys, that the launch's blocks pad
        past the call's (`_padded`, `tuning.KernelOp.padded`): its query and key blocks' tails."""
        queries, keys = ba.arguments["query"].shape[1], _keys(ba)
        padded_queries, padded_keys = _padded(queries, keys, config)
        return 1 - queries * keys / (padded_queries * padded_keys)

    def register_budget(self, config: Config) -> int:
        """A consumer thread's share of the registers (`Family.registers`): the family's roles share the
        file (`mosaic.roles`)."""
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
        family = bundle(target.generation, query.dtype, query.shape[1])
        if family is None:
            return Refusal('family', f"the Mosaic mainloop has no bundle for {name or target.generation} yet")
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
        found = {family.name: family for family in layouts(target.generation, query.dtype, query.shape[1])}
        family = found.get(config.family)
        # A forward's and a dQ launch's layout streams through a producer (`Family.stream`); a layout whose
        # consumers feed their own rings (`split_queries`) runs the dK/dV launches alone, the keys-first one
        # and the fused one, each consumer its own keys over the stream they walk whole. The forward on it
        # has no use the producer's layout does not serve, and deadlocked (row 2599): it is no config.
        walks_whole = not isinstance(config, Tiles) and config.queries_first
        # A decode step's 16-row tile on mma.sync's fed layout (an H100's, `few_rows`): one consumer feeding
        # its own rings, no turns between consumers, whose dropped turns hung the two-consumer fed forward.
        alone = (family is not None and isinstance(config, Tiles) and config.consumers == 1
                 and config.block_q < family.role_rows)
        return (family is not None and (walks_whole or alone or not split_queries(family))
                and whole_tiles(family, config) and self.refusal(ba, family) is None
                and super().legal(ba, config, target))

    def _family(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> mosaic.Family | None:
        """The target's family's bundle where the launch takes the call there, else None."""
        query = ba.arguments["query"]
        family = bundle(target.generation, query.dtype, query.shape[1])
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
                     and change.get("stages", 2) >= 2 and change.get("consumers", 1) <= family.consumers)

    def _get_heuristics_config(self, ba: tokamax.BoundArguments) -> Config:
        target = generation.gpu_target()
        query = ba.arguments["query"]
        family = bundle(target.generation, query.dtype, query.shape[1])
        if family is not None and (reason := self.refusal(ba, family)):
            raise NotImplementedError(reason)
        config = self.derived(ba, target)
        if config is None:
            raise Unfitted(type(self).__name__, None, generation.shared_budget(target), target.generation)
        return config

    def refusal(self, ba: tokamax.BoundArguments, family: mosaic.Family) -> Refusal | None:
        """Why the launch takes no call of `ba` on `family`, or None (`refused`)."""
        raise NotImplementedError


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
    merges: bool = False
    """Whether a split launch merges its slices itself, the last slice's program each query block's, the
    initial state's among them (`attention_mosaic_forward._Merge`): its outputs the rows' whole, no merge in
    XLA after it. Legal on a family that waits across programs (`Family.fixes_up`) where every program of
    its grid is resident at once (`resident`), as its waits on the other slices need."""
    parts: int = 1
    """The bf16 parts each of an fp32 call's operands enters as (`product.EXACT_PARTS`, `product.formed`),
    stacked along the batch axis, part `p`'s rows from `p·B`: every product its parts' (`product.terms`),
    each 16-bit (attention.md 4b). One for a 16-bit call."""

    def _fwd(self, query, key, value, start, end, query_ids, key_ids, score_params, output_params, initial,
             drop=None, table=None, sources=None, scales=None, *, residuals: Kept, interpret: bool,
             config: Tiles, return_residuals: bool = False):
        hooks = Hooks(self.score_mod, tuple(map(Param, score_params, self.score_kinds)), initial,
                      self.output_mod, tuple(map(Param, output_params, self.output_kinds)), drop)
        if reason := refused(query.dtype, query.shape[3], value.shape[3], hooks, backward=False,
                             family=bundle_of(config, query.dtype)):
            raise NotImplementedError(reason)
        spans = self.spans(start, end, query_ids, key_ids)
        form = {"scale": self.scale, "natural": self.natural, "tiles": config, "interpret": interpret}
        match residuals:
            case Kept.SPLIT:
                if scales is not None:
                    raise NotImplementedError("a quantized cache's keys are read by inference alone")
                return _trained_forward(query, key, value, spans, hooks, **form), None
            case Kept.NONE | Kept.OUTPUT:
                # The inference walk, its rows beside the output where it keeps them.
                return _plain_forward(query, key, value, spans, hooks, rows=residuals is Kept.OUTPUT,
                                      out_dtype=self.out_dtype, masked=self.masked, splits=self.splits,
                                      table=table, sources=sources, subtracted=self.subtracted,
                                      merges=self.merges, scales=scales, parts=self.parts,
                                      **form), None

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
        if config.natural_form or self._unrewalked(ba.arguments["query"]):
            return True
        relaunch = config.relaunch
        return (config.block_k % relaunch.block_k == 0 and relaunch.consumers <= config.consumers
                and self._launch_legal(ba, relaunch, target))

    def _launch_legal(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget) -> bool:
        """`_Mainloop.legal`'s for one launch at `config`, and on a plain launch whose sums can pass fp32's
        largest (`algebra.headroom`), a guarded key tile its consumers' registers hold beside it
        (`guarded_tiles`), which the launch walks where the plain one overflowed."""
        if not super().legal(ba, config, target):
            return False
        query, value = ba.arguments["query"], ba.arguments["value"]
        # A quantized call's codes converted where the device converts them (`mosaic.converts`: an FP8 cache
        # on sm_89 on), its call the gathered keys' path elsewhere.
        codes = _codes_of(ba.arguments["key"], ba.arguments.get("scales"))
        # Its values' converted tile one buffer a consumer, which no deferred P·V may still read (`_Codes`).
        if codes is not None and (config.deferred or not mosaic.converts(codes)):
            return False
        family = bundle_of(config, query.dtype)
        # A P·V a step late (pinned: the derivation takes none, `derived`) where its walk takes one.
        dropped = ba.arguments.get("drop") is not None
        if config.deferred and not values_deferred(family, self.form, self.splits, query.dtype, dropped,
                                                   self.parts):
            return False
        # Its key tiles (the launch's, its natural rewalk's) split among the consumers that copy them
        # (`Transport.shares_rows`: two mma.sync consumers' cp.async in whole 64-row parts, so no 32-key tile
        # of two consumers, fs-qualified6).
        if not (family.shares_rows(config.block_k, config.consumers)
                and family.shares_rows(config.relaunch_k or config.block_k,
                                       config.relaunch_consumers or config.consumers)):
            return False
        if self.merges and self.splits > 1:
            # On a family that waits across programs (`Family.fixes_up`), every program of the grid resident
            # at once: the last slice's waits on the others' states. Its kernel holds that residency
            # (`_launched`'s `resident`); a guarded rewalk merges nothing (`_inference_walk`).
            programs = (query.shape[0] * query.shape[2] * self.splits
                        * -(-query.shape[1] // (config.consumers * config.block_q)))
            if (not family.fixes_up or not target.multiprocessors
                    or programs > target.multiprocessors * self.resident(ba, config, target)):
                return False
        # A paged call's key tiles (the launch's, its natural rewalk's) whole in its pages or its pages whole
        # in them (`_pages_hold`).
        tiles = (config.block_k, config.relaunch_k or config.block_k)
        page = _page(ba)
        if page is not None and not all(_pages_hold(page, tile, family) for tile in tiles):
            return False
        computed = _computed(ba)
        if ba.arguments["residuals"] is Kept.SPLIT or not algebra.headroom(computed):
            return True
        columns = guarded_columns(family, value.shape[3], computed, parts=self.parts)
        width = _width(query.shape[3], family, query.dtype)
        return guarded_tiles(family, config, columns, computed, width, parts=self.parts) is not None

    def resident(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget) -> int:
        """The programs of its launch at `config` a multiprocessor holds at once: its family's residency at
        its registers (`Family.programs`) and its shared memory's."""
        family = bundle_of(config, ba.arguments["query"].dtype)
        resident = family.programs(config.consumers, self.registers(ba, config))
        if target.shared_limit:
            resident = min(resident, max(1, target.shared_limit // self.staged_bytes(ba, config, target)))
        return resident

    def derived(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> Tiles | None:
        """Its tiles from the facts (`_derived`): each consumer a unit of queries, the key tile the largest
        its registers and the block hold, two stages; the family's consumers taking turns, a step's softmax
        running under the other's products, the training forward's too (its anchored walk's step is the
        plain one's, TWOPASS4-h100: 1.05 ms at 64 x 64 on two consumers against 1.41 at 64 x 128 on one,
        causal llama-d128 2 x 4096 on an H100 PCIe). Its family the first of the device's bundles, ranked
        by the share of their rows its queries fill (`layouts`: a decode step's few rows mma.sync's before
        wgmma's, row 4042), that takes the call and derives a config."""
        query = ba.arguments["query"]
        config, chosen = None, None
        for family in layouts(target.generation, query.dtype, query.shape[1]):
            if self.refusal(ba, family) is not None:
                continue

            for columns in _column_blocks_of(family, query.dtype, ba.arguments["value"].shape[3]):
                def tiles(rows: int, walked: int, consumers: int, family=family, columns=columns) -> Tiles:
                    return Tiles(rows, walked, 2, consumers, family.name, columns=columns)

                config = _derived(family, query.dtype, query.shape[1], _keys(ba), tiles,
                                  lambda config: self._launch_legal(ba, config, target), overlapped=True,
                                  walks=walked_extents)
                if config is not None:
                    break
            if config is not None:
                chosen = family
                break
        if config is None or chosen is None:
            return None
        # No P·V a step late (`values_deferred`, `Tiles.deferred`, reachable pinned): measured a loss on an
        # H100 against the walk that issues each tile's P·V in turn, Qwen3-8B's causal forward 1.23x and
        # full 1.11x the time, an 8k causal one 1.27x, the training step 1.05x (M-ET-1, W5-gh200-5).
        return self._relaunched(ba, config, target)

    def streaming(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget) -> Tiles:
        """`config` for a call whose walk the device's memory bounds (`streams`), its key tile the least its
        ring still streams at: a program pays its ring's first `stages` tiles before its first product, and
        a launch of more programs than the device runs at once pays it a program, so the least tile pays
        least, where its `stages` slots hold the bytes a program's share of the device's stream keeps in
        flight (`GpuTarget.memory_rate` over its multiprocessors, for the part's measured `Measured.fixed`):
        a decode step of 128 rows over 4096 keys took 0.70 ms at 64-key tiles against 0.73 at 128 on an
        H100, and of 32 over 32768 1.35 against 1.38, its 128-wide K and V 32 KB a stage, two of them past
        the 45 KB a program keeps in flight (ET-STAGES2-h100).
        Its natural rewalk's tiles derived again (`_relaunched`); `config` itself where no smaller tile
        holds, or the part's stream is unmeasured."""
        query, value = ba.arguments["query"], ba.arguments["value"]
        rate, fixed = target.memory_rate, target.fixed_seconds
        if not (streams(query, target) and rate and fixed and target.multiprocessors):
            return config
        family = bundle_of(config, query.dtype)
        itemsize = jnp.dtype(query.dtype).itemsize
        per_key = sum(_width(array.shape[3], family, query.dtype) for array in (query, value)) * itemsize
        in_flight = rate / target.multiprocessors * fixed
        for extent in reversed(walked_extents(family, query.dtype, config.block_k)):
            candidate = dataclasses.replace(config, block_k=extent)
            if config.stages * extent * per_key >= in_flight and self._launch_legal(ba, candidate, target):
                return self._relaunched(ba, candidate, target) or config
        return config

    def _relaunched(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget
                    ) -> Tiles | None:
        """`config` with its natural rewalk's tiles, a fold's (`Tiles.relaunch`): the largest key tile and the
        most consumers that hold its own registers, each dividing the config's (et-fwdhooks3: at the fold's
        tiles it spilled 2 KB at width 64); a call that keeps its logits natural rewalks nothing. None where
        no tile holds the rewalk."""
        config = dataclasses.replace(config, relaunch_k=None, relaunch_consumers=None, relaunch_columns=None)
        if self._unrewalked(ba.arguments["query"]):
            return config
        query, value = ba.arguments["query"], ba.arguments["value"]
        family = bundle_of(config, query.dtype)
        # Its value columns whole first, then halved while a block of them stays a whole column unit: a
        # launch a block (row 2526: at width 256 no tile held the anchored rewalk's whole sums).
        whole = _width(value.shape[-1], family, query.dtype)
        unit = family.column_unit(query.dtype)
        widths = [whole // 2 ** halved for halved in range(whole.bit_length())
                  if whole % 2 ** halved == 0 and (whole // 2 ** halved) % unit == 0]
        relaunches = (dataclasses.replace(config, block_k=block, consumers=consumers, natural_form=True,
                                          columns=None if columns == whole else columns)
                      for columns in widths
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

    def _unrewalked(self, query: jax.Array) -> bool:
        """Whether the call of `query` walks no natural rewalk: its logits natural (`_natural`), or a fold no
        logit's reach passes (`algebra.Forward.reaches`: a subtracted fold's raw products), at the scale its
        launch takes (`attention_mosaic_forward._prepared`)."""
        if self._natural(query):
            return True
        power, scale = prescale(query.dtype, query.shape[3], self.scale)
        taken = power if self.form != "spans" else 1.0
        return not algebra.Forward(self.form, self.keys, ids=False, scale=scale * taken,
                                   score_mod=self.score_mod, natural=self.natural, prescale=power / taken,
                                   subtracted=self.subtracted).reaches

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
        allocations = _forward_allocations(family, config, width, value_width, query.dtype,
                                           ba.arguments.get("residuals") is Kept.SPLIT,
                                           min(config.stages, steps), out_dtype=out_dtype,
                                           codes=_codes_of(ba.arguments["key"], ba.arguments.get("scales")),
                                           parts=self.parts)
        # Its guarded launch reduces its keys' magnitudes across warps (`reducing`'s `guarded`).
        guarded = algebra.headroom(_computed(ba)) and ba.arguments.get("residuals") is not Kept.SPLIT
        return resources.shared_bytes(family, launch_scope(family, allocations, reducing(
            family, config.consumers, self.score_mod, self.output_mod, guarded=guarded)))

    def registers(self, ba: tokamax.BoundArguments, config: Tiles, *, peaks: bool = False,
                  columns: int | None = None) -> int:
        """A consumer's registers at `config`: its walk's (`_forward_registers`) over the values' columns
        its launch walks (`columns`, a block of them; the config's own by default), or its peak walk's
        (`peaks`, `_peaks_registers`), which a training call takes first."""
        family = bundle_of(config, ba.arguments["query"].dtype)
        query, value = ba.arguments["query"], ba.arguments["value"]
        natural = config.natural_form or self._natural(ba.arguments["query"])
        width = _width(query.shape[-1], family, query.dtype)
        params = tuple(zip(self.score_kinds,
                           (kept_dtype(param.dtype) for param in ba.arguments["score_params"]), strict=True))
        if peaks:
            return _peaks_registers(family, config, query.dtype, params=params, width=width)
        value_width = columns or config.columns or _width(value.shape[-1], family, query.dtype)
        residuals = ba.arguments.get("residuals")
        # The plain inference walk's weights carried a step where it issues their P·V late
        # (`Tiles.deferred`).
        deferred = config.deferred and residuals is not Kept.SPLIT
        return _forward_registers(family, config, value_width, query.dtype, residuals is Kept.SPLIT,
                                  natural=natural, params=params, width=width, deferred=deferred,
                                  parts=self.parts)


    def refusal(self, ba: tokamax.BoundArguments, family: mosaic.Family) -> Refusal | None:
        query, value = ba.arguments["query"], ba.arguments["value"]
        hooks = Hooks(self.score_mod, tuple(map(Param, ba.arguments["score_params"], self.score_kinds)),
                      ba.arguments["initial"], self.output_mod,
                      tuple(map(Param, ba.arguments["output_params"], self.output_kinds)),
                      ba.arguments["drop"])
        if self.parts > 1 and (ba.arguments["table"] is not None or ba.arguments["sources"] is not None
                               or ba.arguments["residuals"] is Kept.SPLIT or hooks.drop is not None):
            return Refusal('dtype', "the Mosaic mainloop's fp32 parts walk an inference call over dense "
                           "operands, no dropout")
        if self.parts > 1 and family.feeds_itself:
            # Its parts a copy each into a slot's rows (`mosaic.Pieces`), which a producer's loads take and
            # a consumer's own pipeline does not (`mosaic_sm80.Mma.streams`).
            return Refusal('dtype', f"the Mosaic mainloop's fp32 parts load in pieces, which {family.name}'s "
                           "pipeline does not")
        return refused(query.dtype, query.shape[3], value.shape[3], hooks, backward=False, family=family)


def _hook_live(family: mosaic.Family, block_q: int, block_k: int, params: tuple, *, grads: str | None = None,
               keys_first: bool = False) -> tuple[list, int]:
    """A step's score-hook parameters (`params`, each its `Kind` and dtype), live beside its logits: a
    TILE's `[bq, bk]` tile, a KEY's vector in the accumulator's column layout and a QUERY's in its row
    layout (`Family.column_registers`, `row_registers`); a HEAD's scalar no tile; each in the dtype its kernel
    loads it (`kept_dtype`). With `grads`, a backward launch's gradients of them too, fp32, where it carries
    them: a KEY's across its walk in the dK/dV launch (`grads="keys"`), a QUERY's in the dQ launch
    (`grads="queries"`, `DQ_GRADS`); a TILE's gradient is the step's score cotangents through the hook's
    slope, formed in their place and stored, no tile beside them (an A100's dK/dV under a tile bias at
    64 x 64, width 64: 255 registers unspilled and none spilled at a cap of 247, where its tile beside them
    priced 280). A `keys_first` step's
    tile lays its keys along its rows, so a KEY's vector is in the row layout and a QUERY's in the column.
    Returns the step's tiles and the vectors' registers."""
    keys, queries = ((family.row_registers, family.column_registers) if keys_first
                     else (family.column_registers, family.row_registers))
    step, vectors = [], 0
    for kind, dtype in params:
        if kind is Kind.TILE:
            step.append((block_q, block_k, dtype))
        elif kind is Kind.KEY:
            vectors += keys(block_k, dtype) + (keys(block_k, jnp.float32) if grads == "keys" else 0)
        elif kind is Kind.QUERY:
            vectors += queries(block_q, dtype) + (queries(block_q, jnp.float32) if grads == "queries" else 0)
    return step, vectors


PREDICATE_REGISTERS = 7
"""The predicate registers a GPU thread holds (P0 to P6): a step's predicates past them ptxas saves to
general registers (P2R) until the selects that read them (`_anchor_registers`)."""


def _anchor_registers(family: mosaic.Family, block_q: int, block_k: int) -> int:
    """The registers a queries-first backward step's anchor select holds (`at_anchor`, `gradient.grad_scores`)
    across the step, from the compares that form it to the selects that read it: its predicates packed a
    bit an entry; a row pass's of them, a role's rows, pending, all but those its predicate registers hold
    (`PREDICATE_REGISTERS`); and the places it compares, the keys' column vector and the anchor keys' row
    vector, int32. The SASS of dQ under ALiBi at 64 x 64, width 64, on an A100: 32 `key != anchor`
    compares a thread packed into one word, 25 saved by P2R, restored by ISETP and FSEL, 43 words at its
    need (200); the accurate form's at 64 x 64 and width 128 on an H100 1, 24 and 18, its need 255; dQ at
    128 x 64, width 64, under sinks 2, 27 and 20. Priced beside every phase: beside its products alone, the
    accurate form's dQ under a temperature took 64 x 64 at width 64, where it spills 232 bytes
    (ET-PINBWD), its hook's slopes live across the select."""
    entries = block_q * block_k // resources.THREADS
    pending = max(0, family.role_rows * block_k // resources.THREADS - PREDICATE_REGISTERS)
    places = family.column_registers(block_k, jnp.int32) + family.row_registers(block_q, jnp.int32)
    return -(-entries // 32) + pending + places


def _forward_registers(family: mosaic.Family, tiles: Tiles, value_width: int, dtype, residuals: bool, *,
                       natural: bool = False, params: tuple = (), width: int | None = None,
                       deferred: bool = False, parts: int = 1) -> int:
    """The registers a forward consumer holds at `tiles` (`resources.live_registers`): its weighted values'
    sums `[bq, Dv]` fp32 carried, and a step's logits fp32 beside its weights in the operands' dtype. A
    training forward (`_trained`, at its anchor known) holds the other keys' weights beside them too, in the
    operands' dtype, and the key indices int32 its anchor test compares, a thread's one per column its
    fragment holds, which its fragment's two rows share (an MMA accumulator's layout: half the tile's words;
    TWOPASS5-h100: uncounted, 64 x 128 on two consumers derived and spilled, 6.6 ms against 64 x 64's 1.06);
    its peak walk holds less. The `natural` form's exponential (`algebra.natural_exp`) holds at its peak the
    scaled logits' rounded product and its residual (`two_product`), two fp32 tiles, beside the product's
    bit split, an int32 tile, the logits themselves read entry by entry into them (ET-PINSOFT: width 256
    under a softcap at 64 x 64, ptxas 255 unspilled where three fp32 tiles priced 260; the IR's breadth-first
    live set holds the third, ET-HOOKBREAK); a training walk's then its power's integral part built in bits,
    two int32 tiles beside its weights (ET-HOOKBREAK: a natural training walk 174 at width 64, its fold's
    110 and 64 int32). A score hook's parameters are live beside the logits it reads (`_hook_live`), a phase
    of their own: the hook adds them into its logits, dead past it (ET-PINFWD: key-bias and tile-bias at
    64 x 64 on two consumers, width 64, ptxas 168 unspilled where they priced beside the weights 180 and
    196). Where its family reads operands from registers, its products' fragments beside their tiles
    (`_products`): the logits' product of `width`-deep queries and keys, and the values' product of the
    weights. A key tile whose indices do not fold (`Family.folds_indices`) holds the places its masks
    compare, the queries' and the keys', an int32 tile each. A walk that issues a tile's P·V a step late
    (`deferred`, `values_deferred`) carries the tile's weights across the next step, its logits read while
    that product runs. An fp32 call's `parts` (`MosaicForward.parts`) issue its logits' products' terms
    into one accumulator and form its weights' parts one at a time, its remainder fp32 and the part
    beside the weights (`attention_mosaic_forward._parted_products`)."""
    block_q, block_k = tiles.block_q, tiles.block_k
    hooked, vectors = _hook_live(family, block_q, block_k, params)
    logits, weights = (block_q, block_k, jnp.float32), (block_q, block_k, dtype)
    bits = (block_q, block_k, jnp.int32)
    step = [logits, weights, *([weights] if residuals else [])]
    if parts > 1:
        # The part taken from its weights' remainder, which takes the weights' place (`_parted_products`).
        step += [weights]
    if residuals:
        step.append((block_q // 2, block_k, jnp.int32))
    if not family.folds_indices(block_k):
        # Its masks' query and key places, each entry's formed apart.
        step += [bits, bits]
    phases = [step]
    if natural:
        phases.append([logits] * 2 + [bits])
        if residuals:
            phases.append([logits, weights, weights, bits, bits])
    depth = value_width if width is None else width
    products = _products(family, dtype, (block_q, depth, block_k, [logits], False, True),
                         (block_q, block_k, value_width, [weights], True, False), narrow=tiles.narrow,
                         lhs_swizzle=query_swizzle(family))
    carried = [(block_q, value_width, jnp.float32), *([weights] if deferred else [])]
    scored = resources.live_registers(family, carried, [logits, *hooked]) + vectors
    # A training walk's per-query rows (`TRAINING_ROWS`), in the accumulator's row layout.
    rows = TRAINING_ROWS * family.row_registers(block_q, jnp.float32) if residuals else 0
    return max(resources.live_registers(family, carried, *phases, *products), scored) + rows


def _peaks_registers(family: mosaic.Family, tiles: Tiles, dtype, *, params: tuple, width: int) -> int:
    """The registers a peak walk's consumer holds (`attention_mosaic_forward._launch`'s `peaks`, a training
    call's first walk): it reads the keys alone for each row's largest logit, so it carries a row's maximum
    and holds a step's logits fp32 and its score hook's parameters (`_hook_live`), its masks' places where
    the keys' indices do not fold, and its logits' product's fragments where its family reads operands from
    registers; no weighted values, weights or exponentials (FS-1971-LEAST-ATTN: the forward's walk priced it
    238 where ptxas's least spill-free cap was 112)."""
    block_q, block_k = tiles.block_q, tiles.block_k
    hooked, vectors = _hook_live(family, block_q, block_k, params)
    logits, bits = (block_q, block_k, jnp.float32), (block_q, block_k, jnp.int32)
    step = [logits, *([] if family.folds_indices(block_k) else [bits, bits])]
    products = _products(family, dtype, (block_q, width, block_k, [logits], False, True), narrow=tiles.narrow,
                         lhs_swizzle=query_swizzle(family))
    carried = [(block_q, 1, jnp.float32)]
    scored = resources.live_registers(family, carried, [logits, *hooked]) + vectors
    return max(resources.live_registers(family, carried, step, *products), scored)


def _pages_hold(page: int, tile: int, family: mosaic.Family) -> bool:
    """Whether a paged call's key tiles of `tile` keys lie in its pages of `page`: whole tiles a page, each
    read where it lies (`attention_mosaic_forward._paged_views`), or whole pages a tile, each its own copy
    into a slot's whole swizzled rows (`mosaic.Pieces`, `mosaic.TILE_ROWS`) where `family` loads a step's
    block in pieces (`mosaic.Transport.copies_in_pieces`)."""
    if page % tile == 0:
        return True
    return tile % page == 0 and page % mosaic.TILE_ROWS == 0 and family.copies_in_pieces


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
            sources: jax.Array | None = None, subtracted: bool = False, merges: bool = False,
            scales: tuple[jax.Array, jax.Array] | None = None) -> jax.Array:
    ...


@overload
def forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
            *, scale: float | None = None, natural: bool = NATURAL,
            residuals: Literal[Kept.SPLIT, Kept.OUTPUT], interpret: bool = False,
            out_dtype=None, masked: bool = False, splits: int = 1, table: jax.Array | None = None,
            sources: jax.Array | None = None, subtracted: bool = False, merges: bool = False,
            scales: tuple[jax.Array, jax.Array] | None = None
            ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, Split]:
    ...


def forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
            *, scale: float | None = None, natural: bool = NATURAL, residuals: Kept = Kept.NONE,
            interpret: bool = False, out_dtype=None, masked: bool = False, splits: int = 1,
            table: jax.Array | None = None, sources: jax.Array | None = None, subtracted: bool = False,
            merges: bool = False, scales: tuple[jax.Array, jax.Array] | None = None):
    """The template's forward on this mainloop (`MosaicForward`): `query` `[B, Q, H, D]` over `key`
    `[B, K, Hk, D]` and `value` `[B, K, Hk, Dv]`, and the keys `spans` leaves each query under `hooks`, as
    `dew.nn.kernels.templates.attention.reference` computes it; with `residuals` `Kept.SPLIT` the training
    forward's, its output with the backward's residuals as every mainloop's training forward gives them
    (`attention_portable.attend(residuals=True)`): the maxima and denominators `[B, H, Q]`, each query's
    anchor key `[B, H, Q]` int32 and the split sums (`Split`); `Kept.OUTPUT` the same residuals of the
    inference walk, which keeps its output for the backward's D in place of an anchor (`output_split`,
    `takes_output_residuals`). `natural` keeps the logits natural
    (`algebra.Forward.base2`); `interpret` runs it on the CPU (Mosaic GPU's interpret mode), as a GPU of the
    family `generation.assumed` names; `out_dtype` the inference walk's output's (`MosaicForward.out_dtype`).
    A paged call's `key` and `value` are its pools `[Hk, P, page, D]` and `table` its page table `[B, pages
    a row]` int32: row `b`'s key `s` at offset `s % page` of page `table[b, s // page]`. With `sources`
    `[B]` int32 (a call of rows of keys, inference), batch row `b` reads key row `sources[b]` of `key` and
    `value` `[R, K, Hk, D]`: a serving step's work list over its cache's rows, no row copied. `subtracted`
    an inference walk's fold of raw products (`MosaicForward.subtracted`). With `splits` and `merges`, the
    launch merges its slices (`MosaicForward.merges`), `hooks`' initial state among them: the rows' output
    `[B, Q, H, Dv]` in `out_dtype`, and with `Kept.OUTPUT` their finished maxima and denominators `[B, H, Q]`.
    With `scales` (an inference walk's), `key` and `value` are quantized codes, int8 or an FP8, and `scales`
    the keys' and the values' `[B, Hk, K]` fp32, one a key and key head in its row's key order: the call reads
    the keys and values they dequantize to, each converted to the query's dtype in its program (`_Codes`).
    """
    op, args = _forward_call(query, key, value, spans, hooks, scale=scale, natural=natural,
                             out_dtype=out_dtype, masked=masked, splits=splits, table=table, sources=sources,
                             subtracted=subtracted, merges=merges, scales=scales)
    return op(*args, residuals=residuals, interpret=interpret)


def forward_declined(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans,
                     hooks: Hooks | None = None, *, scale: float | None = None, natural: bool = NATURAL,
                     residuals: Kept = Kept.NONE) -> Refusal | None:
    """Why `forward` runs no call of these arrays compiled on this device, or None
    (`MosaicForward.declined`)."""
    op, args = _forward_call(query, key, value, spans, hooks, scale=scale, natural=natural)
    return op.declined(op.bind(*args, residuals=residuals, interpret=False), generation.gpu_target())


def certified(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
              *, scale: float | None = None, natural: bool = NATURAL, interpret: bool = False):
    """The forward's certify form at the tiles its call derives (`MosaicForward.derived`), its walk the plain
    one (no P·V a step late): `certified_walk`'s output and its state after each key tile. Off in
    production; tests/attention_witness.py holds each state to one step's enclosure of the one before. A call
    no config holds raises the reason the forward declines it (`MosaicForward.declined`: an unqualified
    family's, or the device's block and registers)."""
    op, args = _forward_call(query, key, value, spans, hooks, scale=scale, natural=natural)
    ba = op.bind(*args, residuals=Kept.NONE, interpret=interpret)
    target = generation.gpu_target()
    config = op.derived(ba, target)
    if config is None:
        raise ValueError(f"the Mosaic forward declines the call: {op.declined(ba, target)}")
    return certified_walk(query, key, value, spans, hooks or Hooks(), scale=op.scale, natural=op.natural,
                          tiles=config, interpret=interpret)


def _forward_call(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None, *,
                  scale: float | None, natural: bool, out_dtype=None, masked: bool = False,
                  splits: int = 1, table: jax.Array | None = None, sources: jax.Array | None = None,
                  subtracted: bool = False, merges: bool = False,
                  scales: tuple[jax.Array, jax.Array] | None = None) -> tuple[MosaicForward, tuple]:
    """`forward`'s op and its arguments, a dropping call's draw among them, which its legality rule refuses
    (`refused`)."""
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
                       output_width=power_of_two(value.shape[3]), form=spans.form, keys=spans.keys,
                       scale=logit_scale(query.shape[3], scale), natural=natural,
                       out_dtype=None if out_dtype is None else jnp.dtype(out_dtype).name, masked=masked,
                       splits=splits, subtracted=subtracted, per_span=spans.per_span,
                       merges=merges and splits > 1, parts=parts)
    return op, (query, key, value, spans.start, spans.end, spans.query_ids, spans.key_ids,
                tuple(param.value for param in hooks.score_params),
                tuple(param.value for param in hooks.output_params), hooks.initial, hooks.drop,
                table, sources, scales)


DQ_ROWS = 7
"""The per-query rows a dQ step holds across its walk (`_backward_registers`)."""

TRAINING_ROWS = 5
"""The per-query rows a training forward's walk holds at its peak (`_forward_registers`): the maximum, the
denominator and the anchor it carries, and its anchor reduction's shuffled and loaded rows (the corrected IR
live set, ET-HOOKBREAK at the held term: 3 carried rows and 2 at the peak's shuffle beside its tiles, 174 and
176 at 64 x 64 and width 64 where the model without them held 164 and 166)."""
DKV_STATISTICS = 5
"""The per-query statistics a dK/dV step reads, the maximum, the denominator, the anchor, D and the anchor
key (`attention_mosaic_scores._scores`): row vectors a fused launch's queries first holds
(`_backward_registers`); column vectors keys first, one at a time."""


def _dq_registers(family: mosaic.Family, config: BackwardTiles, width: int, value_width: int, dtype,
                  products: list, held: list, vectors: int, *, accurate: bool, slopes: int) -> int:
    """`_backward_registers`' dQ consumer's: its dQ sums (and the accurate form's keyed sums) carried, its
    per-query rows, a step's two products, then its score cotangents, one part of them at a time, its
    hook's parameters beside its pullback (`held`) and its products' fragments where they read registers."""
    block_q, block_k = config.block_q, config.block_k
    logits = (block_q, block_k, jnp.float32)
    # dQ's anchor select priced as its SASS holds it (`_anchor_registers`), within the family's held
    # scalars where they are a measured whole (`Family.scalars_whole`); its hook's parameters and
    # gradients beside its pullback, not its products: they stream in and out of its entries (dQ under
    # a tile bias at 64 x 64, width 128: S's and dP's accumulators the only float tiles at its peak).
    anchor = _anchor_registers(family, block_q, block_k) if not family.scalars_whole else 0
    products = products[:2]
    pullback = products[1:] + [(block_q, block_k, jnp.float32)] * slopes + held
    carried = [(block_q, width // config.columns, jnp.float32)] * (2 if accurate else 1)
    cotangents = [(block_q, block_k, jnp.float32), (block_q, block_k, dtype)]
    # Its per-query rows in the accumulator's row layout (`Family.row_registers`), live across the walk:
    # the maximum, the prior total, the anchor, D, the anchor key and the residual and mass sums
    # (DQROWPROOF64/128: seven rows, D64 at 128 queries past the budget).
    rows = DQ_ROWS * family.row_registers(block_q, jnp.float32)
    # Its products' fragments where they read registers (`_products`): the logits', the output
    # cotangents' beside them, and dQ's of the score cotangents in registers.
    fragments = _products(family, dtype, (block_q, width, block_k, [logits], False, True),
                          (block_q, value_width, block_k, [logits] * 2, False, True),
                          (block_q, block_k, width // config.columns, cotangents[1:], True, False),
                          narrow=config.narrow)
    if accurate and family.operands_in_registers:
        # The accurate form's parts (`algebra.formed_parts`) issued against one right operand, the keys,
        # whose fragments its family loads once and holds across them, two of its depth's loads ahead,
        # beside the cotangents, the keyed sums' weights fp32 and the part (an A100's dQ at 64 x 64:
        # one LDSM's registers read by four HMMAs over three parts, 218 to 221 live at its peak).
        right = (2 * family.fragment_depth(block_k), width // config.columns * family.rhs_copies, dtype)
        fragments.append([cotangents[0], (block_q, block_k, jnp.float32), cotangents[1], right])
    if jnp.dtype(dtype) == jnp.dtype(jnp.float16):
        # fp16's parts scaled (`product.Scaled`): each part's product its own accumulator, restored and
        # added into dQ's (`operands.Issuer.summed`), beside the cotangents and both parts, formed first.
        fragments.append([cotangents[0], cotangents[1], cotangents[1],
                          (block_q, width // config.columns, jnp.float32)])
    phases = resources.live_registers(family, carried, products, pullback, cotangents, *fragments)
    return phases + anchor + rows + vectors


def _backward_registers(family: mosaic.Family, config: BackwardTiles, width: int, value_width: int, dtype, *,
                        accurate: bool, dq: bool, split: bool = False, params: tuple = (),
                        slopes: int = 0) -> int:
    """The registers a backward launch's consumer holds at `config` (`resources.live_registers`): a dQ
    consumer's dQ sums `[bq, D]` fp32 carried (and the accurate form's keyed sums) and its per-query rows,
    beside a step's logits and output cotangents' products fp32, then its score cotangents fp32 and one part
    of them in the operands' dtype at a time (`algebra.formed_parts`; ptxas held the accurate form's three at
    width 128 in 255 registers, unspilled, where all three would be 262); a dK/dV consumer's dK and dV sums
    `[bk, D]`, `[bk, Dv]` fp32 carried, beside a step's two products (a fused launch's queries first, its dQ
    tile `[bq, D]` fp32 too) and its per-query statistics, each of a consumer's share of a slot's queries
    where its consumers `split` them (`split_queries`), else of the slot's; every sum its share of the
    `columns`. A
    queries-first step's masked tile holds its predicates beside its products (`algebra.seen` of its places'
    int tiles, `_coords`, a byte an entry), from its logits' mask to its cotangents' (ptxas: the fused
    launch at 128 x 64 and width 64 spills 60 bytes, its model 252 without them; 64 x 64 at width 128 holds
    255 unspilled, 242); a keys-first step's places are vectors broadcast at each use. A score hook's
    parameters and their gradients the launch forms are live beside its products (`_hook_live`), and its
    pullback's `slopes` tiles fp32 that vary (`varying_slopes`: its output and its slopes along its
    differentiated inputs) beside its output cotangents' product once the logits' is spent, a phase of their
    own (dQ under a temperature at 128 x 64, width 64, spilled 4316 bytes, its SASS's spills their float
    chains, FMUL, FADD and FSEL, the products' and dQ's accumulators live beside them)."""
    block_q, block_k = config.block_q, config.block_k
    logits = (block_q, block_k, jnp.float32)
    held, vectors = _hook_live(family, block_q, block_k, params, grads="queries" if dq else "keys")
    products = [logits] * 2 + [(block_q, block_k, jnp.int8)] + held
    pullback = products[1:] + [logits] * slopes
    if dq:
        return _dq_registers(family, config, width, value_width, dtype, products, held, vectors,
                             accurate=accurate, slopes=slopes)
    if config.shared_keys:
        # Shared keys (`attention_mosaic_coop`): a consumer's share of the head's columns of dK and dV
        # carried, beside a step's two products over its share of the keys and its share of the dQ tile,
        # its statistics row vectors.
        share = config.consumers
        carried = [(block_k, width // share, jnp.float32), (block_k, value_width // share, jnp.float32)]
        products = [(block_q, block_k // share, jnp.float32)] * 2 + [(block_q, block_k // share, jnp.int8)]
        statistics = DKV_STATISTICS * family.row_registers(block_q, jnp.float32)
        # A score hook's pullback (its slopes, `gradient.pulled`) beside the output cotangents' product.
        pullback = products[1:] + [(block_q, block_k // share, jnp.float32)] * slopes
        return resources.live_registers(family, carried, products, pullback,
                                        [(block_q, width // share, jnp.float32)]) + statistics
    carried = [(block_k, width // config.columns, jnp.float32), (block_k, value_width // config.columns,
                                                                 jnp.float32)]
    if config.fused and not split and not config.shared_dq:
        # Queries first (`attention_mosaic_fused._fused_launch`), on either layout: its statistics row
        # vectors.
        statistics = DKV_STATISTICS * family.row_registers(block_q, jnp.float32)
        dq_tile = [(block_q, width, jnp.float32)]
        # Its products' fragments where they read registers (`_products`): the logits' and the output
        # cotangents', then dV's, dK's and dQ's of the weights and score cotangents it stages.
        fragments = _products(family, dtype, (block_q, width, block_k, [logits], False, True),
                              (block_q, value_width, block_k, [logits] * 2, False, True),
                              (block_k, block_q, value_width, [logits], False, False),
                              (block_k, block_q, width, [logits], False, False),
                              (block_q, block_k, width, dq_tile, False, False))
        # A score hook's pullback (its slopes, `gradient.pulled`) beside the output cotangents' product.
        return resources.live_registers(family, carried, products, pullback, dq_tile, *fragments) + statistics
    queries = block_q // config.consumers if split else block_q
    held, vectors = _hook_live(family, queries, block_k, params, grads="keys", keys_first=True)
    step = [(queries, block_k, jnp.float32)] * 2 + held
    pullback = step[1:] + [(queries, block_k, jnp.float32)] * slopes
    # The keys-first step's statistics in its tiles' column layout (`Family.column_registers`), each read
    # where it is used, from global memory or a shared dQ's stream (`STATISTICS`), one's column vector live
    # at a time (an A100's keys-first launch at 64 x 64, width 128: 17 to 18 words at its peak, where all
    # five's are 80); a fused launch's dQ tile of its share `[queries, D]` fp32 beside its score
    # cotangents, a shared dQ's its share of the columns (`BackwardTiles.shared_dq`).
    statistics = family.column_registers(queries, jnp.float32)
    dq_width = width // config.consumers if config.shared_dq else width
    added = [[step[0], (queries, dq_width, jnp.float32)]] if config.fused else []
    # Its products' fragments where they read registers (`_products`), keys first: the logits' and the
    # output cotangents' of the keys against the queries, then dV's and dK's of the weights and the score
    # cotangents in registers.
    weights = (block_k, queries, dtype)
    columns = config.columns
    fragments = _products(family, dtype, (block_k, width // columns, queries, step[:1], False, True),
                          (block_k, value_width // columns, queries, step, False, True),
                          (block_k, queries, value_width // columns, [weights], True, False),
                          (block_k, queries, width // columns, [weights], True, False), narrow=config.narrow)
    if accurate and family.operands_in_registers:
        # The accurate form's dK parts issued against one right operand, the queries, held at its whole depth
        # across them, beside the cotangents fp32, dV's weights and the part (an A100's dK/dV at 64 x 64:
        # one LDSM's registers read by HMMAs of all three parts, 64 words of them at its peak; one part:
        # ptxas's least 240 fell to 192).
        right = (queries, width // columns * family.rhs_copies, dtype)
        fragments.append([step[0], weights, weights, right])
    if jnp.dtype(dtype) == jnp.dtype(jnp.float16):
        # fp16's dV and dK parts scaled (`product.Scaled`): each product its own accumulator, restored and
        # added into dV's or dK's (`operands.Issuer.summed`), beside the cotangents and both parts, formed
        # first.
        restore = (block_k, max(width, value_width) // columns, jnp.float32)
        fragments.append([step[0], weights, weights, restore])
    phases = resources.live_registers(family, carried, step, pullback, *added, *fragments)
    return phases + statistics + vectors


@dataclasses.dataclass(frozen=True)
class MosaicBackward(_Mainloop[BackwardTiles]):
    """The template's backward on this mainloop (`gradients`) over the call's own arrays (`backward` gives
    them): its queries, keys, values and output cotangent, its spans' arrays, its score hook's parameters, its
    initial state (`Initial`, or None) and a training forward's residuals (`forward`'s). Returns its
    gradients (`Gradients`). Its config comes from the family's facts and the call's extents (`derived`: one
    launch for a fast-form call whose score hook takes no parameters, deterministic mode's a full or causal
    one's, two for the accurate form and a call whose hook does),
    legal within the device's block and a consumer's registers (`legal`)."""

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
                                   family=family, interpret=interpret, drop=drop)
        return gradients(launch, query, key, value, spans, hooks, dout, residuals,
                         block=_block(config), scale=self.scale, rounded=self.rounded), None

    def legal(self, ba: tokamax.BoundArguments, config: BackwardTiles, target: generation.GpuTarget) -> bool:
        # The one launch adds each tile's dQ under the first pass's D: the fast form's, an undropped call's
        # of equal head widths with no initial state, its dK/dV program queries first on the dQ launch's
        # layout or keys first on one whose consumers share a key block (`split_queries`), each consumer's
        # share of a slot's queries a warpgroup MMA's rows (dQ's product's M), or keys first each consumer
        # its own keys over the whole slot where they share dQ (`BackwardTiles.shared_dq`). A score hook's
        # only queries first, and one of no parameters, whose gradients no launch writes (Gemma 2's softcap,
        # FlashAttention 3's; at width 256 its consumers sharing a key block, `attention_mosaic_coop`).
        hooked = (bool(self.score_kinds) or ba.arguments["initial"] is not None
                  or ba.arguments["drop"] is not None)
        if config.fused and self.score_mod is not None and not config.queries_first:
            return False
        found = {family.name: family for family in layouts(target.generation, ba.arguments["query"].dtype)}
        dkv_family = found.get(config.dkv.family)
        if dkv_family is None or not whole_tiles(dkv_family, config.dkv):
            return False
        keys_first = config.dkv_apart
        if config.fused and (ba.arguments["query"].shape[-1] != ba.arguments["value"].shape[-1] or (
                keys_first and not ((split_queries(dkv_family)
                                     and (config.block_q // config.dkv.consumers) % dkv_family.rows == 0)
                                    or (config.shared_dq and config.block_q % dkv_family.rows == 0)))
                or (config.staged_logits and not keys_first)):
            return False
        dtype = ba.arguments["query"].dtype
        # The one launch stages its weights and score cotangents for its products (`mosaic.staged`), which
        # fp16's scaled parts (`product.Scaled`) take no power of two back through: bf16 alone.
        sixteen = jnp.dtype(dtype) == jnp.dtype(jnp.float16)
        if (config.fused and (self.accurate or hooked or sixteen)) or not super().legal(ba, config, target):
            return False
        family = bundle_of(config, dtype)
        # Deterministic mode's one launch orders its adds by turns (`Family.ticketed`): each query tile's dQ
        # adds come from its key tiles counted from the first, which a full or causal call's walk keeps (a
        # program's consumers walk their first key tile's readers, which every earlier key tile reads too);
        # queries first, each consumer its own keys, on a family that waits across programs.
        if config.fused and deterministic_ops_requested() and not (
                self.form in ("full", "causal") and family is not None and family.fixes_up
                and config.queries_first and not (config.shared_keys or config.shared_dq)):
            return False
        if config.producer_dq and not config.shared_dq:
            # The producer adds a shared dQ's staged shares alone (`mosaic.Outbox`).
            return False
        if config.shared_dq:
            # Keys first, each consumer its own keys (no layout whose consumers feed their rings), each its
            # share of dQ's columns whole swizzled rows (`_fused_adds`).
            dkv = config.dkv
            share = _width(ba.arguments["query"].shape[-1], family, dtype) // dkv.consumers
            return (config.fused and not config.queries_first and dkv.consumers > 1 and config.columns == 1
                    and not split_queries(bundle_of(dkv, dtype))
                    and share % family.column_unit(dtype) == 0)
        if config.shared_keys:
            # Every consumer's share of the keys whole rows of a swizzle the exchange stages them at
            # (`mosaic.swizzle_for`), its share of the columns whole swizzled rows (`attention_mosaic_coop`).
            share = _width(ba.arguments["query"].shape[-1], family, dtype) // config.consumers
            narrowest = family.narrowest_columns(dtype)
            return (config.fused and config.queries_first and config.consumers > 1 and config.columns == 1
                    and (config.block_k // config.consumers) % narrowest == 0
                    and share % family.column_unit(dtype) == 0)
        # A width split among programs adds its dQ once, in no fused launch, each share whole swizzled rows.
        shares = [_width(ba.arguments[name].shape[-1], family, dtype) / config.columns
                  for name in ("query", "value")]
        return config.columns == 1 or (not config.fused and all(
            share % family.column_unit(dtype) == 0 for share in shares))

    def derived(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> BackwardTiles | None:
        """Its tiles from the facts (`_derived`): each dK/dV consumer a unit of keys, the query tile both
        launches walk the largest their registers and the block hold, two stages, one consumer, whose step
        issues its products back to back (TRAINC-h100: 6.88 against 3.72 ms causal, 9.68 against 7.56 full,
        at 64 x 64 with two consumers and one). The fast form of a hookless call (or one whose hook takes no
        parameters: a softcap) takes one launch where one holds the whole width, its dK/dV program queries
        first adding each tile's dQ atomically (FUSEDFORM-h100: 1.05 to 1.4 times as fast at widths 64 and
        128 over 1k to 8k tokens; ET-A05AB3-h100: 5.20 against 6.19 ms the two keys-first launches at
        Qwen3-8B causal, 1.19 against 1.53 at width 64;
        check 1 on drawn inputs and finite over every stability family it takes, FUSEDSTAB-h100; 384 MB of
        temporaries against 258 at llama-d128 2 x 4096, its dQ fp32), a call over key spans, windows,
        documents or runs as a plain one (ET-CROSS-h100 and ET-SPAN4-h100, the attention launches a training
        step: Qwen3-8B's heads over four documents 0.67 to 0.72 times the two launches' from 1 x 1024 to
        8192, under a causal window of 4096 at 8192 0.65; width 64's 8 heads over 2 0.56 to 0.88, 16 over
        16 under the window 0.54; check 1 over documents and a window at 1 x 2048; of every call measured,
        row 3171's toy one alone the slower, 1 x 1024 over three short documents and a window of 256,
        1.38 times: 20 us a step); in deterministic mode (`deterministic_ops_requested`) only a full or
        causal call's, queries first on a layout whose consumers feed their rings, every add across programs
        in a fixed order (`Family.ticketed`); otherwise (the accurate form, a hooked or dropping call, a
        width no fused launch holds) two launches, the dQ launch then the keys-first dK/dV one, in a fixed
        order."""
        family = self._family(ba, target)
        if family is None:
            return None
        query, key = ba.arguments["query"], ba.arguments["key"]

        def fits(config: BackwardTiles) -> bool:
            return self.legal(ba, config, target)

        # The dK/dV launch on the layout whose consumers feed their rings where one is, its two consumers
        # splitting each slot's queries (`split_queries`); else on the dQ launch's.
        fed = [layout for layout in layouts(target.generation, query.dtype) if split_queries(layout)]
        dkv = {"dkv_family": fed[0].name, "dkv_consumers": fed[0].consumers} if fed else {}
        deterministic = deterministic_ops_requested()
        found = []
        if not deterministic:
            # Legal where the form, the hooks, the widths and the registers allow (`legal`): of the one
            # launch keys first and queries first, the larger query tile (`_derived`'s order).
            for shared in [*({**dkv, "staged_logits": staged} for staged in (False, True) if dkv), {}]:
                def fused(rows: int, walked: int, consumers: int, shared=shared) -> BackwardTiles:
                    return BackwardTiles(walked, rows, 2, consumers, fused=True, family=family.name, **shared)

                if config := _derived(family, query.dtype, key.shape[1], query.shape[1], fused, fits,
                                      overlapped=False):
                    found.append(config)
        # Queries first on the layout whose consumers feed their rings, each consumer its own keys over
        # the query stream it feeds (FlashAttention 3's CTA), at the layout's consumers, whom no
        # producer's warpgroup shares the registers with (ET-FFED3-h100, the fused launch: 2.40 against
        # 2.71 ms causal and 4.32 against 5.54 full at Qwen3-8B's widths, 0.47 against 0.59 at width 64).
        # Deterministic mode's one launch, every add across programs in its turn (`Family.ticketed`,
        # FlashAttention 3's deterministic backward); else the two launches below, each program's sums its
        # own.
        for layout in fed:
            def fed_fused(rows: int, walked: int, consumers: int, layout=layout) -> BackwardTiles:
                return BackwardTiles(walked, rows, 2, consumers, fused=True, family=layout.name)

            if config := _derived(layout, query.dtype, key.shape[1], query.shape[1], fed_fused, fits,
                                  overlapped=True):
                found.append(config)
        if found:
            # The most keys and queries a program's step takes, its consumers' keys over its query tile;
            # on a tie the more consumers, one's softmax under the other's products (ET-FFED3-h100 at
            # width 64: 64 x 64 on two consumers 0.47 ms against 128 x 64 on one, 0.59).
            return max(found, key=lambda config: (config.block_q * config.block_k * config.consumers,
                                                  config.consumers))
        # Where no consumer's registers hold its own keys' whole dK and dV sums, the consumers share
        # one key block, each its share of the columns (FlashAttention 3's width 256: ET-COOP1-h100,
        # 0.68 ms against the two launches' 1.30 at Qwen3-8B's heads 256 wide, causal; where the sums
        # fit, sharing cost 9% at width 128 and 46% at 64, ET-COOP2-h100).
        for layout in fed if not deterministic else ():
            def sharing(rows: int, walked: int, consumers: int, layout=layout) -> BackwardTiles:
                return BackwardTiles(walked, rows, 2, consumers, fused=True, family=layout.name,
                                     shared_keys=True)

            if config := _derived(layout, query.dtype, key.shape[1], query.shape[1], sharing, fits,
                                  overlapped=True):
                return config
        # The fewest dK/dV programs a key block's widths split among whose share of the sums its consumer's
        # registers hold (`BackwardTiles.columns`: a width of 256's whole dK and dV sums pass them).
        # Its logits' queries a stream of their own, else where that passes the block staged a step
        # (`BackwardTiles.staged_logits`).
        for columns in (1, 2, 4):
            for staged in (False, True):
                def tiles(rows: int, walked: int, consumers: int, columns=columns, staged=staged
                          ) -> BackwardTiles:
                    return BackwardTiles(walked, rows, 2, consumers, family=family.name, columns=columns,
                                         staged_logits=staged, **dkv)

                if config := _derived(family, query.dtype, key.shape[1], query.shape[1], tiles, fits,
                                      overlapped=False):
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
            allocations = _coop_allocations if config.shared_keys else _fused_allocations
            # The fused launch reduces across no warps, its statistics and D its operands and its sums its
            # products' (an A100's at 64 x 64 reserves no reduction scratch, its lowering's 114,688 bytes the
            # model's without it, et-fused80).
            # The fused launch's dK and dV in the operands' dtype where it stores them so (`stores_typed`).
            typed = {} if config.shared_keys else {"typed": stores_typed(
                family, config, width, dtype, q_stages, grouped=query.shape[2] != key.shape[2])}
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
        return max(resources.shared_bytes(layout, launch_scope(layout, allocations, reducing(
            layout, tiles.consumers, self.score_mod), reduces=reduces))
            for layout, tiles, allocations, reduces in launches)

    def registers(self, ba: tokamax.BoundArguments, config: BackwardTiles) -> int:
        dq, dkv = self._launch_registers(ba, config)
        return dkv if config.fused else max(dq, dkv)

    def _launch_registers(self, ba: tokamax.BoundArguments, config: BackwardTiles) -> tuple[int, int]:
        """Each launch's registers a consumer holds (`_backward_registers`): the dQ launch's at `config` (none
        runs where `fused`), the dK/dV launch's at its own tiles (`BackwardTiles.dkv`), its consumers
        splitting a slot's queries where it runs keys first on a layout whose consumers feed their rings
        (`split_queries`), not the fused launch queries first on that layout (`_fused_launch`)."""
        query, value = ba.arguments["query"], ba.arguments["value"]
        counts = []
        for tiles, dq in ((config, True), (config.dkv, False)):
            family = bundle_of(tiles, query.dtype)
            width, value_width = (_width(array.shape[-1], family, query.dtype) for array in (query, value))
            split = split_queries(family) and not config.queries_first
            params = tuple(zip(self.score_kinds, (kept_dtype(param.dtype)
                                                  for param in ba.arguments["score_params"]), strict=True))
            counts.append(_backward_registers(family, tiles, width, value_width, query.dtype,
                                              accurate=self.accurate, dq=dq, split=split, params=params,
                                              slopes=self.score_slopes[0 if dq else 1]))
        return counts[0], counts[1]

    def _held(self, ba: tokamax.BoundArguments, config: BackwardTiles) -> bool:
        # Each launch's registers within its own layout's budget (`Family.registers`).
        dq, dkv = self._launch_registers(ba, config)
        held = dkv <= self.register_budget(config.dkv)
        return held and (config.fused or dq <= self.register_budget(config))


    def refusal(self, ba: tokamax.BoundArguments, family: mosaic.Family) -> Refusal | None:
        query, value = ba.arguments["query"], ba.arguments["value"]
        hooks = Hooks(self.score_mod, tuple(map(Param, ba.arguments["score_params"], self.score_kinds)),
                      ba.arguments["initial"], drop=ba.arguments["drop"])
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
    """`backward`'s op and its arguments, a dropping call's draw among them (`_forward_call`)."""
    hooks = Hooks() if hooks is None else hooks
    op = MosaicBackward(score_mod=hooks.score_mod,
                        score_kinds=tuple(param.kind for param in hooks.score_params),
                        score_dtypes=tuple(kept_dtype(param.value.dtype) for param in hooks.score_params),
                        form=spans.form, keys=spans.keys,
                        scale=logit_scale(query.shape[3], scale), natural=natural,
                        accurate=accurate, rounded=rounded,
                        score_slopes=(varying_slopes(hooks, DQ_GRADS), varying_slopes(hooks, DKV_GRADS)))
    return op, (query, key, value, dout, spans.start, spans.end, spans.query_ids, spans.key_ids,
                tuple(param.value for param in hooks.score_params), hooks.initial, residuals, hooks.drop)
