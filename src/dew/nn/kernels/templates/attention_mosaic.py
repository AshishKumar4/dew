"""The attention template's Mosaic GPU mainloop as Template implementations (`MosaicForward`,
`MosaicBackward`; design §6 and §7): the forward, its training form and the backward over every call form,
each launch's one config derived from the instruction family's and the device's facts and the call's extents
(`derived`), legal where its programs stage within the device's block and a consumer's registers hold its live
set (`legal`): no table, no sweep, no space ranked. A GPU nobody measured of a family with a bundle here
(`bundle`) takes its own facts' config. The launches are `attention_mosaic_forward`'s and
`attention_mosaic_backward`'s; what they share, `attention_mosaic_program`'s."""

import dataclasses
import functools
from collections.abc import Callable, Sequence
from typing import ClassVar, Literal, overload

import jax
import jax.numpy as jnp
import tokamax

from dew.devices import deterministic_ops_requested
from dew.nn.kernels import generation
from dew.nn.kernels.templates import attention_algebra as algebra, mosaic, resources
from dew.nn.kernels.templates.attention import (
    ACCURATE,
    NATURAL,
    TRACED_TILE,
    AttentionLaunch,
    Gradients,
    Hooks,
    KeySpans,
    Kind,
    Param,
    Split,
    gradients,
    kept_dtype,
    power_of_two,
)
from dew.nn.kernels.templates.attention_mosaic_backward import (
    _dkv_allocations,
    _dq_allocations,
    _fused_allocations,
    _launch_backward,
    split_queries,
)
from dew.nn.kernels.templates.attention_mosaic_forward import (
    _forward_allocations,
    _plain_forward,
    _trained_forward,
    guarded_columns,
    guarded_tiles,
)
from dew.nn.kernels.templates.attention_mosaic_program import (
    BackwardTiles,
    Tiles,
    _block,
    _padded,
    _width,
    bundle,
    bundle_of,
    extents,
    layouts,
    reducing,
    refused,
    whole_tiles,
)
from dew.nn.kernels.templates.base import Template, Unfitted


def _derived[Config: (Tiles, BackwardTiles)](family: mosaic.Family, dtype, held: int, walked: int,
                                                tiles: Callable[[int, int, int], Config],
                                                fits: Callable[[Config], bool], *,
                                                overlapped: bool) -> Config | None:
    """A launch's config from its family's facts and the call's extents (design §7), no space ranked: its
    consumers the family's (`Family.consumers`) where its step `overlapped` (a softmax between its products,
    which runs under its partner's products as the consumers take turns, FlashAttention-3's), else one (a
    step issuing its products back to back, which turns only make wait); fewer where the `held` rows a
    consumer accumulates (a forward's queries, a dK/dV launch's keys) fill fewer of the family's units; each
    consumer's rows the family's unit (`_extents`' least: the rows its products cover, `Family.role_rows`, in
    whole swizzled rows); the tile
    it walks the largest unit multiple of the `walked` extent whose live set a consumer's registers hold and
    whose two stages the block holds (`fits`), two the ring's busy depth, a slot loading while one is read;
    fewer consumers where no tile of the family's count holds the block (a GPU of a smaller one).
    `tiles(rows, walked, consumers)` builds the config. None where no tile fits."""
    rows = extents(family, dtype, held)[-1]
    most = family.consumers if overlapped else 1
    for consumers in range(max(1, min(most, -(-held // rows))), 0, -1):
        for extent in extents(family, dtype, walked):
            config = tiles(rows, extent, consumers)
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

    def spans(self, start, end, query_ids, key_ids) -> KeySpans:
        """The call's spans from its arrays and the op's static form and key count."""
        return KeySpans(start, end, query_ids, key_ids, self.keys, self.form)

    def register_budget(self, config: Config) -> int:
        """A consumer thread's share of the registers (`Family.registers`): the family's roles share the
        file (`mosaic.roles`)."""
        # A consumer's budget is its family's, whatever the operands' dtype (`mosaic.Family.registers`).
        return bundle_of(config, jnp.bfloat16).registers(config.consumers)

    def eligible(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> bool:
        return self.declined(ba, target) is None

    def declined(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> str | None:
        """Why the launch runs no call of `ba` on `target`, or None: its family has no bundle here, the call
        is one it does not take (`refusal`, each reason with its ledger row), it is interpreted in a GPU
        process (`grouped_matmul._Mosaic.eligible`: the interpreter's callbacks are ordered effects, which a
        GPU process's lowering cannot carry), or no config fits the device (`derived`)."""
        name = generation.family(target.generation)
        family = bundle(target.generation, ba.arguments["query"].dtype)
        if family is None:
            return f"the Mosaic mainloop has no bundle for {name or target.generation} yet"
        if (reason := self.refusal(ba, family)) is not None:
            return reason
        if ba.arguments["interpret"] and jax.default_backend() != "cpu":
            return "an interpreted Mosaic launch runs in a CPU process alone"
        if self.derived(ba, target) is None:
            return f"no config of the Mosaic mainloop fits {target.generation}'s block and registers"
        return None

    def legal(self, ba: tokamax.BoundArguments, config: Config, target: generation.GpuTarget) -> bool:
        """Whether a launch of `ba` runs at `config` on `target`: the target's family's bundle (`bundle`),
        tiles its program takes (`whole_tiles`), a call the launch takes (`refusal`), and the block and
        registers `Template.legal` holds it to. A pinned or stored config is held to it as a derived one."""
        found = {family.name: family for family in layouts(target.generation, ba.arguments["query"].dtype)}
        family = found.get(config.family)
        # A forward's and a dQ launch's layout streams through a producer (`Family.stream`); a layout whose
        # consumers feed their own rings (`split_queries`) runs the keys-first dK/dV launch alone.
        return (family is not None and not split_queries(family) and whole_tiles(family, config)
                and self.refusal(ba, family) is None and super().legal(ba, config, target))

    def _family(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> mosaic.Family | None:
        """The target's family's bundle where the launch takes the call there, else None."""
        family = bundle(target.generation, ba.arguments["query"].dtype)
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
        reach = max(walked * 2, query.shape[1], ba.arguments["key"].shape[1])
        sizes = [size for size in extents(family, query.dtype, reach) if size in (walked // 2, walked * 2)]
        changes = [*({self.WALKED: size} for size in sizes),
                   {"stages": config.stages - 1}, {"stages": config.stages + 1},
                   {"consumers": config.consumers - 1}, {"consumers": config.consumers + 1}]
        return tuple(dataclasses.replace(config, **change) for change in changes
                     if all(value > 0 for value in change.values())
                     and change.get("stages", 2) >= 2 and change.get("consumers", 1) <= family.consumers)

    def _get_heuristics_config(self, ba: tokamax.BoundArguments) -> Config:
        target = generation.gpu_target()
        family = bundle(target.generation, ba.arguments["query"].dtype)
        if family is not None and (reason := self.refusal(ba, family)):
            raise NotImplementedError(reason)
        config = self.derived(ba, target)
        if config is None:
            raise Unfitted(type(self).__name__, None, generation.shared_budget(target), target.generation)
        return config

    def refusal(self, ba: tokamax.BoundArguments, family: mosaic.Family) -> str | None:
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

    def _fwd(self, query, key, value, start, end, query_ids, key_ids, score_params, output_params, initial,
             drop=None, *, residuals: bool, interpret: bool, config: Tiles, return_residuals: bool = False):
        hooks = Hooks(self.score_mod, tuple(map(Param, score_params, self.score_kinds)), initial,
                      self.output_mod, tuple(map(Param, output_params, self.output_kinds)), drop)
        if reason := refused(query.dtype, query.shape[3], value.shape[3], hooks, backward=False,
                             family=bundle_of(config, query.dtype)):
            raise NotImplementedError(reason)
        run = _trained_forward if residuals else _plain_forward
        return run(query, key, value, self.spans(start, end, query_ids, key_ids), hooks, scale=self.scale,
                   natural=self.natural, tiles=config, interpret=interpret), None

    def legal(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget) -> bool:
        """`_Mainloop.legal`'s, and on a plain launch whose sums can pass fp32's largest (`algebra.headroom`),
        a guarded key tile its consumers' registers hold beside it (`guarded_tiles`), which the launch walks
        where the plain one overflowed."""
        if not super().legal(ba, config, target):
            return False
        query, value = ba.arguments["query"], ba.arguments["value"]
        if ba.arguments["residuals"] or not algebra.headroom(value.dtype):
            return True
        family = bundle_of(config, query.dtype)
        columns = guarded_columns(family, value.shape[3], value.dtype)
        return guarded_tiles(family, config, columns, value.dtype) is not None

    def derived(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> Tiles | None:
        """Its tiles from the facts (`_derived`): each consumer a unit of queries, the key tile the largest
        its registers and the block hold, two stages; the family's consumers taking turns, a step's softmax
        running under the other's products, the training forward's too (its anchored walk's step is the
        plain one's, TWOPASS4-h100: 1.05 ms at 64 x 64 on two consumers against 1.41 at 64 x 128 on one,
        causal llama-d128 2 x 4096 on an H100 PCIe)."""
        family = self._family(ba, target)
        if family is None:
            return None
        query, key = ba.arguments["query"], ba.arguments["key"]

        def tiles(rows: int, walked: int, consumers: int) -> Tiles:
            return Tiles(rows, walked, 2, consumers, family.name)

        return _derived(family, query.dtype, query.shape[1], key.shape[1], tiles,
                        lambda config: self.legal(ba, config, target), overlapped=True)

    def staged_bytes(self, ba: tokamax.BoundArguments, config: Tiles, target: generation.GpuTarget) -> int:
        """The program's shared memory (`_forward_allocations`, `resources.shared_bytes`), at the call's
        widths as the launch stages them (`_width`)."""
        family = bundle_of(config, ba.arguments["query"].dtype)
        query, key, value = (ba.arguments[name] for name in ("query", "key", "value"))
        width, value_width = (_width(array.shape[-1], family, query.dtype) for array in (query, value))
        steps = _padded(query.shape[1], key.shape[1], config)[1] // config.block_k
        allocations = _forward_allocations(family, config, width, value_width, query.dtype,
                                           bool(ba.arguments.get("residuals")), min(config.stages, steps))
        return resources.shared_bytes(family, _scope(family, allocations, reducing(
            family, config.consumers, self.score_mod, self.output_mod)))

    def registers(self, ba: tokamax.BoundArguments, config: Tiles) -> int:
        family = bundle_of(config, ba.arguments["query"].dtype)
        query, value = ba.arguments["query"], ba.arguments["value"]
        return _forward_registers(family, config, _width(value.shape[-1], family, query.dtype), query.dtype,
                                  bool(ba.arguments.get("residuals")))


    def refusal(self, ba: tokamax.BoundArguments, family: mosaic.Family) -> str | None:
        query, value = ba.arguments["query"], ba.arguments["value"]
        hooks = Hooks(self.score_mod, tuple(map(Param, ba.arguments["score_params"], self.score_kinds)),
                      ba.arguments["initial"], self.output_mod,
                      tuple(map(Param, ba.arguments["output_params"], self.output_kinds)),
                      ba.arguments["drop"])
        return refused(query.dtype, query.shape[3], value.shape[3], hooks, backward=False, family=family)


def _forward_registers(family: mosaic.Family, tiles: Tiles, value_width: int, dtype, residuals: bool) -> int:
    """The registers a forward consumer holds at `tiles` (`resources.live_registers`): its weighted values'
    sums `[bq, Dv]` fp32 carried, and a step's logits fp32 beside its weights in the operands' dtype. A
    training forward (`_trained`, at its anchor known) holds the other keys' weights beside them too, in the
    operands' dtype, and the key indices int32 its anchor test compares, a thread's one per column its
    fragment holds, which its fragment's two rows share (an MMA accumulator's layout: half the tile's words;
    TWOPASS5-h100: uncounted, 64 x 128 on two consumers derived and spilled, 6.6 ms against 64 x 64's 1.06);
    its peak walk holds less."""
    block_q, block_k = tiles.block_q, tiles.block_k
    step = [(block_q, block_k, jnp.float32)] + [(block_q, block_k, dtype)] * (2 if residuals else 1)
    if residuals:
        step.append((block_q // 2, block_k, jnp.int32))
    return resources.live_registers(family, [(block_q, value_width, jnp.float32)], step)


@overload
def forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
            *, scale: float | None = None, natural: bool = NATURAL, residuals: Literal[False] = False,
            interpret: bool = False) -> jax.Array:
    ...


@overload
def forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
            *, scale: float | None = None, natural: bool = NATURAL, residuals: Literal[True],
            interpret: bool = False) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, Split]:
    ...


def forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
            *, scale: float | None = None, natural: bool = NATURAL, residuals: bool = False,
            interpret: bool = False):
    """The template's forward on this mainloop (`MosaicForward`): `query` `[B, Q, H, D]` over `key`
    `[B, K, Hk, D]` and `value` `[B, K, Hk, Dv]`, and the keys `spans` leaves each query under `hooks`, as
    `dew.nn.kernels.templates.attention.reference` computes it; with `residuals` the training forward's,
    its output with the backward's residuals as every mainloop's training forward gives them
    (`attention_portable.attend(residuals=True)`): the maxima and denominators `[B, H, Q]`, each query's
    anchor key `[B, H, Q]` int32 and the split sums (`Split`). `natural` keeps the logits natural
    (`algebra.Forward.base2`); `interpret` runs it on the CPU (Mosaic GPU's interpret mode), as a GPU of the
    family `generation.assumed` names."""
    op, args = _forward_call(query, key, value, spans, hooks, scale=scale, natural=natural)
    return op(*args, residuals=residuals, interpret=interpret)


def forward_declined(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans,
                     hooks: Hooks | None = None, *, scale: float | None = None, natural: bool = NATURAL,
                     residuals: bool = False) -> str | None:
    """Why `forward` runs no call of these arrays compiled on this device, or None
    (`MosaicForward.declined`)."""
    op, args = _forward_call(query, key, value, spans, hooks, scale=scale, natural=natural)
    return op.declined(op.bind(*args, residuals=residuals, interpret=False), generation.gpu_target())


def forward_takes(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans,
                  hooks: Hooks | None = None, *, scale: float | None = None, natural: bool = NATURAL,
                  residuals: bool = False) -> bool:
    """Whether `forward` runs the call compiled on this device (`forward_declined`): its family has a
    bundle here, the call is one this mainloop takes (`refused`) and a config fits the device."""
    return forward_declined(query, key, value, spans, hooks, scale=scale, natural=natural,
                            residuals=residuals) is None


def _forward_call(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None, *,
                  scale: float | None, natural: bool) -> tuple[MosaicForward, tuple]:
    """`forward`'s op and its arguments, a dropping call's draw among them, which its legality rule refuses
    (`refused`)."""
    hooks = Hooks() if hooks is None else hooks
    op = MosaicForward(score_mod=hooks.score_mod, output_mod=hooks.output_mod,
                       score_kinds=tuple(param.kind for param in hooks.score_params),
                       output_kinds=tuple(param.kind for param in hooks.output_params),
                       score_dtypes=tuple(kept_dtype(param.value.dtype) for param in hooks.score_params),
                       output_dtypes=tuple(kept_dtype(param.value.dtype) for param in hooks.output_params),
                       output_width=power_of_two(value.shape[3]), form=spans.form, keys=spans.keys,
                       scale=query.shape[3] ** -0.5 if scale is None else scale, natural=natural)
    return op, (query, key, value, spans.start, spans.end, spans.query_ids, spans.key_ids,
                tuple(param.value for param in hooks.score_params),
                tuple(param.value for param in hooks.output_params), hooks.initial, hooks.drop)


def _scope(family: mosaic.Family, allocations: tuple, reducing_warpgroups: int) -> resources.Scope:
    """A launch's `allocations` (`(scoped, streams)`, `Family.scoped`'s) by lifetime (`Family.lifetimes`),
    with its `reducing_warpgroups`' scratch (`resources.Scope`)."""
    scoped, streams = allocations
    held, walks = family.lifetimes(streams)
    return resources.Scope(program=(scoped, *held), walks=walks, reducing_warpgroups=reducing_warpgroups)


DQ_ROWS = 7
"""The per-query rows a dQ step holds across its walk (`_backward_registers`)."""
DKV_STATISTICS = 5
"""The per-query statistics a dK/dV step holds, the maximum, the denominator, the anchor, D and the anchor
key (`attention_mosaic_backward._scores`): column vectors keys first, row vectors in a fused launch's queries
first (`_backward_registers`)."""


def _backward_registers(family: mosaic.Family, config: BackwardTiles, width: int, value_width: int, dtype, *,
                        accurate: bool, dq: bool) -> int:
    """The registers a backward launch's consumer holds at `config` (`resources.live_registers`): a dQ
    consumer's dQ sums `[bq, D]` fp32 carried (and the accurate form's keyed sums) and its per-query rows,
    beside a step's logits and output cotangents' products fp32, then its score cotangents fp32 and one part
    of them in the operands' dtype at a time (`algebra.formed_parts`; ptxas held the accurate form's three at
    width 128 in 255 registers, unspilled, where all three would be 262); a dK/dV consumer's dK and dV sums
    `[bk, D]`, `[bk, Dv]` fp32 carried, beside a step's two products (a fused launch's queries first, its dQ
    tile `[bq, D]` fp32 too) and its per-query statistics; every sum its share of the `columns`. A
    queries-first step's masked tile holds its predicates beside its products (`algebra.seen` of its places'
    int tiles, `_coords`, a byte an entry), from its logits' mask to its cotangents' (ptxas: the fused
    launch at 128 x 64 and width 64 spills 60 bytes, its model 252 without them; 64 x 64 at width 128 holds
    255 unspilled, 242); a keys-first step's places are vectors broadcast at each use."""
    block_q, block_k = config.block_q, config.block_k
    products = [(block_q, block_k, jnp.float32)] * 2 + [(block_q, block_k, jnp.int8)]
    if dq:
        carried = [(block_q, width // config.columns, jnp.float32)] * (2 if accurate else 1)
        cotangents = [(block_q, block_k, jnp.float32), (block_q, block_k, dtype)]
        # Its per-query rows in the accumulator's row layout (`Family.row_registers`), live across the walk:
        # the maximum, the prior total, the anchor, D, the anchor key and the residual and mass sums
        # (DQROWPROOF64/128: seven rows, D64 at 128 queries past the budget).
        rows = DQ_ROWS * family.row_registers(block_q, jnp.float32)
        return resources.live_registers(family, carried, products, cotangents) + rows
    carried = [(block_k, width // config.columns, jnp.float32), (block_k, value_width // config.columns,
                                                                 jnp.float32)]
    if config.fused:
        # Queries first (`attention_mosaic_backward._fused_launch`): its statistics row vectors.
        statistics = DKV_STATISTICS * family.row_registers(block_q, jnp.float32)
        dq_tile = [(block_q, width, jnp.float32)]
        return resources.live_registers(family, carried, products, dq_tile) + statistics
    # A consumer's share of a slot's queries where the consumers share its keys (`split_queries`).
    queries = block_q // config.consumers if split_queries(family) else block_q
    step = [(queries, block_k, jnp.float32)] * 2
    # The keys-first step's statistics in its tiles' column layout (`Family.column_registers`).
    statistics = DKV_STATISTICS * family.column_registers(queries, jnp.float32)
    return resources.live_registers(family, carried, step) + statistics


@dataclasses.dataclass(frozen=True)
class MosaicBackward(_Mainloop[BackwardTiles]):
    """The template's backward on this mainloop (`gradients`) over the call's own arrays (`backward` gives
    them): its queries, keys, values and output cotangent, its spans' arrays, its score hook's parameters, its
    initial state (`Initial`, or None) and a training forward's residuals (`forward`'s). Returns its
    gradients (`Gradients`). Its config comes from the family's facts and the call's extents (`derived`: one
    launch for a hookless fast-form call, two for the accurate form, a hooked call and deterministic mode),
    legal within the device's block and a consumer's registers (`legal`)."""

    HOOKS = ("score_mod",)
    DIFFERENTIATED = True
    SPANS = 4
    """Its arrays' spans after the queries, keys, values and the output's cotangent (`_fwd`)."""
    config_cls = BackwardTiles
    WALKED = "block_q"
    accurate: bool = ACCURATE
    rounded: bool = True

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
        padded = _padded(queries, key.shape[1], config)[0] > queries
        # The form and geometry; the launch takes the scale and the query's power `gradients` gives it.
        spec = algebra.Forward(spans.form, spans.keys, ids=query_ids is not None, scale=self.scale,
                               score_mod=self.score_mod, natural=self.natural, initial=initial is not None,
                               queries=queries if padded else None)
        launch = functools.partial(_launch_backward, spec=spec, accurate=self.accurate, tiles=config,
                                   family=family, interpret=interpret, drop=drop)
        return gradients(launch, query, key, value, spans, hooks, dout, residuals,
                         block=_block(config), scale=self.scale, rounded=self.rounded), None

    def legal(self, ba: tokamax.BoundArguments, config: BackwardTiles, target: generation.GpuTarget) -> bool:
        # The one launch adds each tile's dQ under the first pass's D: the fast form's, a hookless and
        # undropped call's of equal head widths, its dK/dV program queries first on the dQ launch's layout.
        hooked = (self.score_mod is not None or ba.arguments["initial"] is not None
                  or ba.arguments["drop"] is not None)
        if config.fused and (config.dkv != dataclasses.replace(config, dkv_family=None, dkv_consumers=None)
                             or config.staged_logits
                             or ba.arguments["query"].shape[-1] != ba.arguments["value"].shape[-1]):
            return False
        found = {family.name: family for family in layouts(target.generation, ba.arguments["query"].dtype)}
        dkv_family = found.get(config.dkv.family)
        if dkv_family is None or not whole_tiles(dkv_family, config.dkv):
            return False
        if (config.fused and (self.accurate or hooked)) or not super().legal(ba, config, target):
            return False
        # A width split among programs adds its dQ once, in no fused launch, each share whole swizzled rows.
        dtype = ba.arguments["query"].dtype
        family = bundle_of(config, dtype)
        shares = [_width(ba.arguments[name].shape[-1], family, dtype) / config.columns
                  for name in ("query", "value")]
        return config.columns == 1 or (not config.fused and all(
            share % family.column_unit(dtype) == 0 for share in shares))

    def derived(self, ba: tokamax.BoundArguments, target: generation.GpuTarget) -> BackwardTiles | None:
        """Its tiles from the facts (`_derived`): each dK/dV consumer a unit of keys, the query tile both
        launches walk the largest their registers and the block hold, two stages, one consumer, whose step
        issues its products back to back (TRAINC-h100: 6.88 against 3.72 ms causal, 9.68 against 7.56 full,
        at 64 x 64 with two consumers and one). The fast form of a hookless call takes one launch where one
        holds the whole width, its dK/dV program queries first adding each tile's dQ atomically
        (FUSEDFORM-h100: 1.05 to 1.4 times as fast at widths 64 and 128 over 1k to 8k tokens; ET-A05AB3-h100:
        5.20 against 6.19 ms the two keys-first launches at Qwen3-8B causal, 1.19 against 1.53 at width 64;
        check 1 on drawn inputs and finite over every stability family it takes, FUSEDSTAB-h100; 384 MB of
        temporaries against 258 at llama-d128 2 x 4096, its dQ fp32); otherwise (the accurate form, a hooked
        or dropping call, deterministic mode, `deterministic_ops_requested`, a width no fused launch holds)
        two launches, the dQ launch then the keys-first dK/dV one, in a fixed order."""
        family = self._family(ba, target)
        if family is None:
            return None
        query, key = ba.arguments["query"], ba.arguments["key"]

        def fits(config: BackwardTiles) -> bool:
            return self.legal(ba, config, target)

        if not deterministic_ops_requested():
            def fused(rows: int, walked: int, consumers: int) -> BackwardTiles:
                return BackwardTiles(walked, rows, 2, consumers, fused=True, family=family.name)

            # Legal where the form, the hooks, the widths and the registers allow (`legal`).
            if config := _derived(family, query.dtype, key.shape[1], query.shape[1], fused, fits,
                                  overlapped=False):
                return config
        # The fewest dK/dV programs a key block's widths split among whose share of the sums its consumer's
        # registers hold (`BackwardTiles.columns`: a width of 256's whole dK and dV sums pass them).
        # The dK/dV launch on the layout whose consumers feed their rings where one is, its two consumers
        # splitting each slot's queries (`split_queries`); else on the dQ launch's.
        # Its logits' queries a stream of their own, else where that passes the block staged a step
        # (`BackwardTiles.staged_logits`).
        fed = [layout for layout in layouts(target.generation, query.dtype) if split_queries(layout)]
        dkv = {"dkv_family": fed[0].name, "dkv_consumers": fed[0].consumers} if fed else {}
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
        if config.fused:
            launches = [(family, config, _fused_allocations(family, config, width, dtype, q_stages))]
        else:
            launches = [(dkv_family, config.dkv, _dkv_allocations(
                dkv_family, config.dkv, width, value_width, dtype, stages=q_stages, columns=config.columns))]
        if not config.fused:
            launches.append((family, config, _dq_allocations(family, config, width, value_width, dtype,
                                                             min(config.stages, padded_k // config.block_k))))
        return max(resources.shared_bytes(layout, _scope(layout, allocations, reducing(
            layout, tiles.consumers, self.score_mod))) for layout, tiles, allocations in launches)

    def registers(self, ba: tokamax.BoundArguments, config: BackwardTiles) -> int:
        dq, dkv = self._launch_registers(ba, config)
        return dkv if config.fused else max(dq, dkv)

    def _launch_registers(self, ba: tokamax.BoundArguments, config: BackwardTiles) -> tuple[int, int]:
        """Each launch's registers a consumer holds (`_backward_registers`): the dQ launch's at `config` (none
        runs where `fused`), the dK/dV launch's at its own tiles (`BackwardTiles.dkv`)."""
        query, value = ba.arguments["query"], ba.arguments["value"]
        counts = []
        for tiles, dq in ((config, True), (config.dkv, False)):
            family = bundle_of(tiles, query.dtype)
            width, value_width = (_width(array.shape[-1], family, query.dtype) for array in (query, value))
            counts.append(_backward_registers(family, tiles, width, value_width, query.dtype,
                                              accurate=self.accurate, dq=dq))
        return counts[0], counts[1]

    def _held(self, ba: tokamax.BoundArguments, config: BackwardTiles) -> bool:
        # Each launch's registers within its own layout's budget (`Family.registers`).
        dq, dkv = self._launch_registers(ba, config)
        held = dkv <= self.register_budget(config.dkv)
        return held and (config.fused or dq <= self.register_budget(config))


    def refusal(self, ba: tokamax.BoundArguments, family: mosaic.Family) -> str | None:
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
    split sums (`algebra.split_delta`), a dQ launch and a dK/dV launch, and `dq_finish`'s refinement between
    them, in the form `accurate` names; or the fast form's one launch (`BackwardTiles.fused`,
    `_launch_backward`), a hookless call's derived default outside deterministic mode
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
                      natural: bool = NATURAL, accurate: bool = ACCURATE) -> str | None:
    """Why `backward` runs no call of these arrays compiled on this device, or None
    (`MosaicBackward.declined`)."""
    op, args = _backward_call(query, key, value, spans, hooks, dout, residuals, scale=scale, natural=natural,
                              accurate=accurate)
    return op.declined(op.bind(*args, interpret=False), generation.gpu_target())


def backward_takes(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None,
                  dout: jax.Array, *, residuals: tuple[jax.Array, jax.Array, jax.Array, Split],
                  scale: float | None = None, natural: bool = NATURAL, accurate: bool = ACCURATE) -> bool:
    """Whether `backward` runs the call compiled on this device (`backward_declined`): its family has a
    bundle here, the call is one this mainloop takes (`refused`) and a config fits the device."""
    return backward_declined(query, key, value, spans, hooks, dout, residuals=residuals, scale=scale,
                             natural=natural, accurate=accurate) is None


def _backward_call(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None,
                   dout: jax.Array, residuals: tuple, *, scale: float | None, natural: bool,
                   accurate: bool, rounded: bool = True) -> tuple[MosaicBackward, tuple]:
    """`backward`'s op and its arguments, a dropping call's draw among them (`_forward_call`)."""
    hooks = Hooks() if hooks is None else hooks
    op = MosaicBackward(score_mod=hooks.score_mod,
                        score_kinds=tuple(param.kind for param in hooks.score_params),
                        score_dtypes=tuple(kept_dtype(param.value.dtype) for param in hooks.score_params),
                        form=spans.form, keys=spans.keys,
                        scale=query.shape[3] ** -0.5 if scale is None else scale, natural=natural,
                        accurate=accurate, rounded=rounded)
    return op, (query, key, value, dout, spans.start, spans.end, spans.query_ids, spans.key_ids,
                tuple(param.value for param in hooks.score_params), hooks.initial, residuals, hooks.drop)
