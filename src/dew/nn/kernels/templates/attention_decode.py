"""The attention template's decode: a few query positions a row over that row's keys, through the template's
own forward over a work list the step builds (kernel-templates.md §5 and §6; theme A06).

A decode step reads, for every row, many keys for a few queries. The template's forward walks a query
tile's keys with the algebra's online softmax; two builders shape the step for it, and a merge ends it:

- `packed`: a key head's group of query heads become its rows, one (query, head-in-group) pair a row, so
  a step of one query position fills a tile with the group and reads each key tile once for all of it.
  The same builder takes a prefill call: a call's rows are its rows whatever their count (§5: GQA packing
  is structure, one builder for prefill and decode).
- `plan`: each query tile's keys cut into contiguous slices, each slice a program of its own on the
  template forward's grid (`MosaicForward.splits`), as many as the GPU holds at once (the derived
  config's resident programs a multiprocessor, from its registers and its shared memory, times the
  multiprocessors; no fitted table), so one row's long cache fills the GPU. No query or span is copied.
- the merge: the slices' softmax states combined with the call's initial state (a sink, `Initial`), in
  a fixed order, deterministic: one launch over the slices where the walk wrote them, the law's own path
  with an initial state (`merged.merged_split`).
  A merge in the launch, by the last slice's program to arrive, was never faster in kernel time at any
  split on an A100 or a GH200, and the step's wall time is its launches' either way (M-ET-46).

No arithmetic of its own: the walk, the hooks, the guards and the merge are the template forward's and its
algebra's.
"""

import dataclasses
import functools
import math
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Literal, NamedTuple, overload

import jax
import jax.numpy as jnp
from jax.core import ShapedArray
from jax.extend import core as jaxpr_core

from dew.nn import programs
from dew.nn.kernels import generation, tuning
from dew.nn.kernels.templates import attention_algebra as algebra, attention_mosaic, cost, emission, merged
from dew.nn.kernels.templates.attention import (
    Hooks,
    Initial,
    Kept,
    KeySpans,
    Kind,
    Param,
    grouped_rows,
    prologues_taken,
    spans as key_spans,
    ungrouped_rows,
)
from dew.nn.kernels.templates.attention_mosaic_forward import NATURAL_NAME, folded_form
from dew.nn.kernels.templates.cost import Work


def packable(hooks: Hooks | None) -> bool:
    """Whether a call's query heads can become its rows (`packed`): no score hook reading a query's place or
    score parameter (a packed row's place is its pair's, not its query's), and no dropout; a softcap (the
    scaled logits alone), sinks (an initial state a head), and output hooks admitted by
    `merged.output_hook_refusal` pack (row 5385)."""
    if hooks is None:
        return True
    if hooks.score_params or hooks.drop is not None:
        return False
    if merged.output_hook_refusal(hooks.output_mod, hooks.output_params) is not None:
        return False
    return hooks.score_mod is None or not _reads_places(hooks.score_mod)


def _packed_output_params(params: tuple[Param, ...], *, heads: int, kv_heads: int) -> tuple[Param, ...]:
    """Pack `output_params` (`Kind.OUTPUT`, or ungrouped `Kind.HEAD`) onto grouped decode rows
    `[B, T·G, KV, ...]` for in-kernel single-slice forward fusion (`Kind.QUERY` and grouped `Kind.HEAD` defer
    to `merged_finished` to preserve `[Q_tile, 1]` and scalar `()` shapes, row 5385)."""
    group = heads // kv_heads
    packed_list: list[Param] = []
    for param in params:
        if param.kind is Kind.HEAD:
            assert group == 1, "grouped single-slice calls defer Kind.HEAD hooks to preserve scalar shape"
            packed_list.append(param)
        else:
            assert param.kind is Kind.OUTPUT, "single-slice forward packs Kind.OUTPUT and ungrouped Kind.HEAD"
            packed_list.append(Param(grouped_rows(param.value, kv_heads), Kind.OUTPUT))
    return tuple(packed_list)


def serves(query: jax.Array, key: jax.Array, hooks: Hooks | None, *, natural: bool,
           target: generation.GpuTarget) -> bool:
    """Whether a call's inference runs as this work list (`attend`) rather than the template's forward: its
    hooks pack (`packable`), its key heads divide its query heads, its form the fold's (the work list folds,
    its natural rewalk its own), and its packed rows a key head too few for the device's products to bound
    its walk (`attention_mosaic.streams`): a decode step's, a speculative verify block's. Its walk derives a
    config wherever the template's forward takes the call (`plan`). A verify block of 4 to 16 drafts a row
    through the forward took 1.1 to 5.9 ms where the work list took 30 us to 1.36 ms on an H100, its query
    heads read the keys once a key head (ET-VERIFY-h100)."""
    heads, kv_heads = query.shape[2], key.shape[2]
    if natural or heads % kv_heads or not packable(hooks):
        return False
    batch, positions, _, width = query.shape
    rows = jax.ShapeDtypeStruct((batch, positions * (heads // kv_heads), kv_heads, width), query.dtype)
    return attention_mosaic.streams(rows, target)

def _reads_places(score_mod) -> bool:
    """Whether `score_mod(s, q_pos, k_pos)` reads its query's place: its traced program uses the argument."""
    scores = jax.ShapeDtypeStruct((8, 8), jnp.float32)
    places = jax.ShapeDtypeStruct((8, 8), jnp.int32)
    traced = jax.make_jaxpr(score_mod)(scores, places, places)
    used = {var for eqn in traced.jaxpr.eqns for var in eqn.invars if isinstance(var, jaxpr_core.Var)}
    used |= {var for var in traced.jaxpr.outvars if isinstance(var, jaxpr_core.Var)}
    return traced.jaxpr.invars[1] in used


def packed(query: jax.Array, spans: KeySpans, kv_heads: int, initial: Initial | None = None
           ) -> tuple[jax.Array, KeySpans, Initial | None]:
    """`query` `[B, T, H, D]` as `[B, T·G, KV, D]` over its `kv_heads` key heads (`grouped_rows`), and `spans`
    `[B, T]` and `initial` `[B, H, T]` alike: each packed row sees its query's keys and starts from its head's
    state; `ungrouped_rows` takes an output back. A full call's rows a full call's, its form kept, so its
    launches price it as one (row 8981: packed as spans of a `whole` flag beside them, its price had a second
    owner); a call of one run's position's interval holds its G rows (`KeySpans.per_span`), read by position
    in the program; a call of several runs or documents an interval a row."""
    batch, positions, heads, _ = query.shape
    group = heads // kv_heads
    rows = grouped_rows(query, kv_heads)
    if spans.form == "full":
        # A full call's rows see every key still: a full call of G rows a position, its launches priced so.
        start, end = (jnp.repeat(x, group, axis=1) for x in (spans.start, spans.end))
        spans = KeySpans(start, end, None, spans.key_ids, spans.keys, "full")
    elif spans.query_ids is None and spans.start.ndim == 2:
        spans = KeySpans(spans.start, spans.end, None, spans.key_ids, spans.keys, "spans",
                         spans.per_span * group)
    else:
        start, end = (jnp.repeat(x, group, axis=1) for x in (spans.start, spans.end))
        query_ids = None if spans.query_ids is None else jnp.repeat(spans.query_ids, group, axis=1)
        spans = KeySpans(start, end, query_ids, spans.key_ids, spans.keys, "spans")
    if initial is not None:
        def heads_rows(x):
            # [B, H, T, ...] as [B, KV, T·G, ...].
            shaped = x.reshape(batch, kv_heads, group, positions, *x.shape[3:])
            shaped = jnp.swapaxes(shaped, 2, 3)
            return shaped.reshape(batch, kv_heads, positions * group, *x.shape[3:])

        initial = dataclasses.replace(initial, m=heads_rows(initial.m), total=heads_rows(initial.total),
                                      acc=heads_rows(initial.acc))
    return rows, spans, initial


@dataclasses.dataclass(frozen=True)
class Plan:
    """A decode walk's work list as the derivation shapes it: the forward's config (`attention_mosaic.Tiles`)
    and the slices each query tile's keys are cut into (`MosaicForward.splits`), the config the launch's own
    op's at those slices, and whether its launch decides its scaled walk itself (`MosaicForward.deciding`),
    the step's one walk then."""

    config: attention_mosaic.Tiles
    slices: int
    decides: bool = False


class Walk(NamedTuple):
    """A decode step's walk as the forward's op names it (`attention_mosaic.forward`'s `scaled` and
    `bounded`)."""

    scaled: bool = False
    bounded: bool = False


WALKS: Mapping[str, Walk] = MappingProxyType({"plain": Walk(), "scaled": Walk(scaled=True),
                                              "bounded": Walk(bounded=True)})
"""The walks a decode step launches (`attend`): a call whose rows can pass fp32's largest the scaled walk and
the unscaled one, which its values' bounds choose between, any other the plain one."""


def launched(query: jax.Array, value: jax.Array,
             scales: tuple[jax.Array, jax.Array] | None) -> tuple[str, ...]:
    """The walks (`WALKS`) a decode step of `query` over `value` (codes where `scales`) launches: the scaled
    and the bounded ones where its rows' sums can pass fp32's largest (`algebra.headroom`, of the dtype its
    walk reads its values in), else the plain one."""
    if algebra.headroom(query.dtype if scales is not None else value.dtype):
        return ("scaled", "bounded")
    return ("plain",)


def plan(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None, *,
         scale: float | None, target: generation.GpuTarget, table: jax.Array | None = None,
         sources: jax.Array | None = None, scales: tuple[jax.Array, jax.Array] | None = None,
         walk: str = "plain", in_key: bool = False, defer_output: bool = False) -> Plan | None:
    """The work list of a packed call (`packed`) under its `hooks` (its initial state among them) on `target`,
    its `walk` (`WALKS`): the forward's derived config at the key tile its memory-bound walk streams at least
    (`MosaicForward.streaming`) and its slices; None where no launch of it derives. Each config is derived
    for the op its launch runs (its walk, its output's dtype and residuals, its slices), so it reads that
    op's refusals (`report.refuse_launch`): a scaled launch ptxas refused steps down at its own config, never
    at the plain op's, which the pin would hold it to (row 5420).
    One where the programs already fill the multiprocessors (a slice's merge would cost and its keys gain
    nothing). Else the count of slices whose step the cost model prices least, walk and merge
    (`_priced_slices`), each slice at least twice its ring's stages of key tiles: a shorter slice's walk
    spends its prologue's loads, `stages` tiles, without as many steps of products to hide them
    (ET-DECODE2-h100: 32 slices of one key tile each, at batch 1 over 4096 keys, took 22 us where cuDNN's
    whole step took 13); where it prices none, as many as keep the device's stream in flight
    (`MosaicForward.streamed`, at most those its residency holds). Each slice writes its fp32 state, which
    the step merges with the initial state (`merged.merged_split`), at a config derived for them."""
    chosen = WALKS[walk]
    single_hooks = (None if hooks is None else dataclasses.replace(
        hooks, initial=None, output_mod=None, output_params=())) if defer_output else hooks
    op, ba = _stream_op(query, key, value, spans, single_hooks, scale=scale, table=table, sources=sources,
                        out_dtype=jnp.float32 if defer_output else None, scales=scales, chosen=chosen,
                        in_key=in_key)
    # A config its caller chooses (a pin held for the forward, or an opted-in tuning run's picks) is retiled
    # as its streaming walk retiles it, never priced over.
    chosen_by_caller = op.pinned(ba) is not None or tuning.reads_picks()
    # The one-slice launch's config as derived (`remaining`, its refusals read) at the tile its walk
    # streams at, certified only where the plan takes it (`single`): a split plan compiles no config of a
    # launch it never runs (row 12697: a hooked one-row step certified the one-slice walk's three configs
    # before it split, 70 of its 168 s on a compile-only B200).
    derived = None if chosen_by_caller else op.remaining(ba)
    if chosen_by_caller or derived is None:
        config = _certified_stream(op, ba, target)
    else:
        config = op.streaming(ba, derived, target)
    if config is None:
        return None

    def planned(config, op, ba, slices: int) -> Plan:
        return Plan(config, slices, op.deciding(ba, config, target=target))

    def single() -> Plan | None:
        certified = config if chosen_by_caller or derived is None else _certified_stream(op, ba, target)
        return None if certified is None else planned(certified, op, ba, 1)

    processors = target.multiprocessors or 1
    tiles = math.ceil(query.shape[1] / (config.consumers * config.block_q))
    # Its launch's programs: a float32 call's operands as their parts stack along the rows
    # (`MosaicForward.parts`), as many programs again each (a merging launch's every one resident).
    programs = op.parts * query.shape[0] * query.shape[2] * tiles
    if programs >= processors:
        return single()

    walked = None if hooks is None else dataclasses.replace(
        hooks, initial=None, output_mod=None, output_params=())
    streamed = functools.partial(_streamed, query, key, value, spans, walked, scale=scale, target=target,
                                 table=table, sources=sources, out_dtype=jnp.float32, scales=scales,
                                 chosen=chosen, in_key=in_key)
    # Each slice at least twice its ring's stages of key tiles: a shorter slice's walk spends its prologue's
    # loads without as many steps of products to hide them (ET-DECODE2-h100).
    longest = math.ceil(spans.keys / config.block_k) // (2 * min(config.stages, 2))
    # No more slices than four programs a multiprocessor take: past them a slice's programs wait a wave.
    most = min(longest, max(2, 4 * processors // programs))
    counts = [1 << power for power in range(1, most.bit_length()) if 1 << power <= most]
    # Priced over every key a row holds (`spans.keys`): a call's own intervals (a cache's lengths, its
    # documents) its launches count no work of, their extents its data's (a decode step's every launch
    # priced none, its plan the heuristic's alone: M-NA-27-gh200 at fb6a6daa2).
    every = spans if spans.form == "full" else key_spans(query.shape[1], spans.keys, query.shape[0])
    priced = None if chosen_by_caller else _priced_slices(
        query, key, value, every, walked, config, counts, scale=scale, target=target, table=table,
        sources=sources, scales=scales, chosen=chosen, in_key=in_key,
        initial=hooks is not None and hooks.initial is not None,
        single_launch=(single_hooks, jnp.float32 if defer_output else None))
    if priced is not None:
        slices, tile = priced
        if slices == 1:
            return single()
        # The priced tile completed and legal at the call's own bound arguments (`_launchable`: its rules, a
        # recorded refusal among them), certified, and the certified config taken there too; else its
        # streaming walk's own config at those slices, certified (`_streamed`).
        sliced_op, sliced_ba = _bound(query, key, value, spans, walked, scale=scale, table=table,
                                      sources=sources, scales=scales, splits=slices, chosen=chosen,
                                      in_key=in_key)
        launched = sliced_op._launchable(sliced_ba, tile, target)
        if launched is not None:
            certified = sliced_op._certified(sliced_ba, launched)
            if (isinstance(certified, attention_mosaic.Tiles)
                    and sliced_op.takes(sliced_ba, certified, target)
                    and sliced_op.pinnable(sliced_ba, certified)):
                return planned(certified, sliced_op, sliced_ba, slices)
    else:
        # The most slices whose programs keep the device's stream in flight at once (`MosaicForward.streamed`,
        # at most those it holds), where the cost model prices no step of them: sized at its stateless walk's
        # derived tile, uncertified as the priced path's are, a pin's or a pick's certified.
        sizing_op, sizing_ba = _bound(query, key, value, spans, walked, scale=scale, table=table,
                                      sources=sources, scales=scales, splits=1, chosen=chosen, in_key=in_key)
        sizing = None if chosen_by_caller else sizing_op.remaining(sizing_ba)
        found = (streamed() if sizing is None
                 else (sizing_op.streaming(sizing_ba, sizing, target), sizing_op, sizing_ba))
        slices = 1 if found is None else max(1, min(
            processors * found[1].streamed(found[2], found[0], target) // programs, longest))
    # The sliced launch's own op, at its slices, certified (`_certified_stream`).
    found = None if slices == 1 else streamed(splits=slices)
    return single() if found is None else planned(*found, slices)


def _priced_slices(query, key, value, spans: KeySpans, hooks: Hooks | None, single: attention_mosaic.Tiles,
                   counts: list[int], *, scale: float | None, target: generation.GpuTarget, table, sources,
                   scales, chosen: Walk, in_key: bool, initial: bool, single_launch: tuple
                   ) -> tuple[int, attention_mosaic.Tiles] | None:
    """The slice count of `counts` (one beside them, at the derived `single` tile as `single_launch`'s hooks
    and written dtype run it, the plan's one-slice launch) and key tile whose step
    the cost model prices least over `spans`: its walk (`MosaicForward.price_config`, its programs' warps
    hiding each other's issue intervals) and, past one slice, the merge of its slices (`merged.launch_work`,
    its passes' waited loads: the state form's merge, a proxy of a finished step's, its cast, its heads'
    layout and its output hook aside). Each further count prices the tiles its derived config's walk may
    stream at (`MosaicForward.streamings`), uncertified here; the caller certifies the chosen one at its own
    launch. None where the model prices some step none. A GH200's B8 over 32768 keys at width 128 walked
    346.0 us on 2 slices of K128 and 309.3 on 4 of K64 at two programs a multiprocessor (M-NA-21-gh200),
    which a walk priced at its device bytes alone ranked level and its programs' issue intervals rank right
    (FS-5274-LAT1)."""
    bound = functools.partial(_bound, query, key, value, spans, hooks, scale=scale, table=table,
                              sources=sources, scales=scales, chosen=chosen, in_key=in_key)
    # The single tile priced as its one-slice launch runs (`single_launch`: its hooks, its initial state
    # among them, and the dtype it writes, `plan`'s), only where it launches at these spans too.
    single_hooks, single_out = single_launch
    op, ba = _stream_op(query, key, value, spans, single_hooks, scale=scale, table=table, sources=sources,
                        out_dtype=single_out, scales=scales, chosen=chosen, in_key=in_key)
    alone = None if op._launchable(ba, single, target) is None else op.price_config(ba, single, target,
                                                                                     issued=True)
    if alone is None:
        return None
    prices = {(1, single): alone}
    for count in counts:
        sliced_op, sliced_ba = bound(splits=count)
        derived = sliced_op.derived(sliced_ba, target)
        if derived is None:
            continue
        tiles = sliced_op.streamings(sliced_ba, derived, target)
        for tile in tiles:
            walk = sliced_op.price_config(sliced_ba, tile, target, issued=True)
            if walk is None:
                return None
            prices[count, tile] = walk + _merge_seconds(query, value, spans, tile, count, target=target,
                                                        initial=initial)
    # The least price, then the fewer slices, then the smaller key tile.
    return min(prices, key=lambda chosen_plan: (prices[chosen_plan], chosen_plan[0], chosen_plan[1].block_k))


def _bound(query, key, value, spans: KeySpans, hooks: Hooks | None, *, scale: float | None, table, sources,
           scales, splits: int, chosen: Walk, in_key: bool) -> tuple:
    """A split walk's op over `splits` slices, its fp32 state written, and its bound arguments, deriving
    none (`_stream_op`)."""
    return _stream_op(query, key, value, spans, hooks, scale=scale, table=table, sources=sources,
                      out_dtype=jnp.float32, scales=scales, splits=splits, chosen=chosen, in_key=in_key)


def _merge_seconds(query, value, spans: KeySpans, config: attention_mosaic.Tiles, slices: int, *,
                   target: generation.GpuTarget, initial: bool) -> float:
    """The seconds of the merge of a split walk's `slices` (`merged.launch_work`): the walk's outputs as its
    launch writes them, `[B, S, Qp, H, Dvp]` fp32 and their maxima and denominators `[2, B, H, S, Qp]`
    (`attention_mosaic_forward.SplitOutput`)."""
    family = attention_mosaic.bundle_of(config, query.dtype, target=target)
    rows = attention_mosaic._padded(query.shape[1], spans.keys, config)[0]
    width = attention_mosaic._width(value.shape[-1], family, query.dtype)
    batch, heads = query.shape[0], query.shape[2]
    padded = jax.ShapeDtypeStruct((batch, slices, rows, heads, width), jnp.float32)
    stacked = jax.ShapeDtypeStruct((2, batch, heads, slices, rows), jnp.float32)
    spec = algebra.finished_form(_spec(query, spans, Hooks(), scale=None))
    work = merged.launch_work(spec, padded, stacked, query.shape[1], value.shape[-1], target,
                              has_initial=initial)
    return cost.seconds(work, target)


def _streamed(query, key, value, spans: KeySpans, hooks: Hooks | None, *, scale: float | None,
              target: generation.GpuTarget, table, sources, out_dtype, scales=None, splits: int = 1,
              chosen: Walk = WALKS["plain"], in_key: bool = False):
    """The forward's config for a packed call's launch of the walk `chosen` names (`WALKS`) writing
    `out_dtype` (the query's where None) over `splits` slices, certified at the key tile its walk streams at
    (`_certified_stream`), with its op and bound arguments (`_stream_op`); None where none derives or
    certification refuses every config."""
    op, ba = _stream_op(query, key, value, spans, hooks, scale=scale, table=table, sources=sources,
                        out_dtype=out_dtype, scales=scales, splits=splits, chosen=chosen, in_key=in_key)
    config = _certified_stream(op, ba, target)
    return None if config is None else (config, op, ba)


def _stream_op(query, key, value, spans: KeySpans, hooks: Hooks | None, *, scale: float | None, table,
               sources, out_dtype, scales=None, splits: int = 1, chosen: Walk = WALKS["plain"],
               in_key: bool = False) -> tuple:
    """The op a packed call's launch of the walk `chosen` names runs, writing `out_dtype` (the query's where
    None) over `splits` slices (`in_key` its values' stream or the key's slot), and its bound arguments,
    deriving none."""
    masked = hooks is None or hooks.output_mod is None
    op, args = attention_mosaic._forward_call(query, key, value, spans, hooks, scale=scale, natural=False,
                                              out_dtype=out_dtype, masked=masked, splits=splits, table=table,
                                              sources=sources, subtracted=True, scales=scales,
                                              scaled=chosen.scaled, bounded=chosen.bounded, in_key=in_key)
    return op, op.bind(*args, residuals=Kept.OUTPUT if out_dtype is not None else Kept.NONE, interpret=False)


def _certified_stream(op: attention_mosaic.MosaicForward, ba, target: generation.GpuTarget
                      ) -> attention_mosaic.Tiles | None:
    """The config a decode launch of `ba` runs, which its plan pins: the forward's own as certification holds
    it (`taken`: a pin, a tuned pick, else the first derived config whose kernels hold), at the key tile a
    decode step's few rows a key head stream at (`MosaicForward.streaming`), that tile certified in turn
    (`KernelOp._certified`): one that spills is refused, which `streaming`'s legality reads, so the launch
    offers its next tile, a held rung of it taken first, else `taken`'s config itself, uncertified here as a
    pick is. None where certification refuses every config. A pinned raw proposal compiled uncertified (row
    7259)."""
    config = op.taken(ba)
    if not isinstance(config, attention_mosaic.Tiles):
        return None

    # `op.streaming` steps down to `config` itself as its terminal offer: a pin or tuned pick is a measured
    # choice `_certified` leaves to its program's compile (`report.settled`), a derived one `taken` certified.
    first = op.streaming(ba, config, target)
    if first == config:
        return config
    found = op._certified(ba, first, offered=lambda: op.streaming(ba, config, target), decided=config)
    return found if isinstance(found, attention_mosaic.Tiles) else None


def _spec(query: jax.Array, spans: KeySpans, hooks: Hooks, *, scale: float | None) -> algebra.Forward:
    """The call's form as the template's forward builds it (`attention_mosaic_forward.folded_form`), its fold
    of raw products (`algebra.Forward.subtracted`): the units its finished rows' maxima are in
    (`algebra.finished_form`), which its merge takes."""
    spec, _ = folded_form(spans.form, spans.keys, ids=spans.query_ids is not None, dtype=query.dtype,
                          width=query.shape[3], scale=scale, natural=False, score_mod=hooks.score_mod,
                          subtracted=True, initial=hooks.initial is not None)
    return algebra.finished_form(spec)


def planned(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks, *,
            scale: float | None = None, table: jax.Array | None = None, sources: jax.Array | None = None,
            scales: tuple[jax.Array, jax.Array] | None = None, in_key: bool = False
            ) -> tuple[Mapping[str, Plan] | None, jax.Array, KeySpans, Hooks]:
    """A decode step's work list as `attend` takes the step (its operands as `attend`'s): its query rows
    packed with their spans and initial state (`packed`), the hooks that read them, and the plan of each walk
    it launches (`launched`, `plan`), None where some walk derives none on the GPU. `attend` launches them; a
    cache's selector asks of the same step's operands whether they derive (`dew.nn.kv_cache._template_tiles`):
    one planning of one call."""
    kv_heads = key.shape[2] if table is None else key.shape[0]
    rows, row_spans, initial = packed(query, spans, kv_heads, hooks.initial)
    group = query.shape[2] // kv_heads
    defer_merge = hooks.output_mod is not None and any(
        (param.kind is Kind.HEAD and group > 1) or param.kind is Kind.QUERY
        for param in hooks.output_params)
    if defer_merge:
        whole = dataclasses.replace(hooks, initial=initial, output_mod=None, output_params=())
    else:
        packed_out = _packed_output_params(hooks.output_params, heads=query.shape[2], kv_heads=kv_heads)
        whole = dataclasses.replace(hooks, initial=initial, output_params=packed_out)
    target = generation.gpu_target()
    planned_walk = functools.partial(plan, rows, key, value, row_spans, whole, scale=scale, target=target,
                                     table=table, sources=sources, scales=scales, in_key=in_key,
                                     defer_output=defer_merge)
    # values' bounds choose between (`WALKS`), the one fallback of a launch whose scaled walk's registers its
    # unscaled walk's config does not hold (`MosaicForward.decided`).
    plain = planned_walk(walk="plain")
    if plain is not None and plain.decides:
        return {"plain": plain}, rows, row_spans, whole
    work = {}
    for walk in launched(query, value, scales):
        found = plain if walk == "plain" else planned_walk(walk=walk)
        if found is None:
            return None, rows, row_spans, whole
        work[walk] = found
    return work, rows, row_spans, whole


def _planned_step(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans,
                  hooks: Hooks | None, *, scale: float | None, interpret: bool, table: jax.Array | None,
                  sources: jax.Array | None, scales: tuple[jax.Array, jax.Array] | None,
                  bounds: jax.Array | None, finished: bool,
                  in_key: bool
                  ) -> tuple[Callable[[], jax.Array | Initial], Callable[[], jax.Array | Initial]]:
    """`attend`'s step planned once (`planned`): `(run, common)`, where `run` executes the full step and
    `common` executes its unscaled/bounded common arm for pricing (`priced_step`)."""
    hooks = Hooks() if hooks is None else hooks
    if hooks.key_mod is not None and (table is not None or scales is not None):
        raise ValueError("a paged or quantized cache holds its keys under their prologue, taken as they were "
                         "cached: a call over its pages or codes takes no key prologue")
    in_key = attention_mosaic.reads_in_key(in_key, key, value, hooks, table=table, scales=scales)
    query, key, hooks = prologues_taken(query, key, hooks)
    if not finished and hooks.output_mod is not None:
        raise ValueError("an unfinished decode step returns a partial state and takes no output hook")
    if not packable(hooks):
        raise NotImplementedError("the template's decode packs no query-placed score hook, score parameter, "
                                  "unsupported output parameter or dropout (`packable`)")
    heads = query.shape[2]
    work, rows, row_spans, whole = planned(query, key, value, spans, hooks, scale=scale, table=table,
                                           sources=sources, scales=scales, in_key=in_key)
    initial = whole.initial
    if work is None:
        raise NotImplementedError(f"the template's decode derives no config on {generation.gpu_target()}")
    launch = functools.partial(attention_mosaic.forward, scale=scale, natural=False, interpret=interpret,
                               table=table, sources=sources, subtracted=True, scales=scales, in_key=in_key)

    def walked(walk: str) -> jax.Array | Initial:
        # The step's walk (`WALKS`) at its own plan: scaled or unscaled as it names, else a cache call's own
        # (`MosaicForward.cached`).
        chosen, at = WALKS[walk], work[walk]
        defer_merge = hooks.output_mod is not None and whole.output_mod is None
        if at.slices == 1 and finished and not defer_merge:
            # A row's walk from the initial state, finished in the query's dtype with any output hook fused
            # in-kernel, no merge past it (its fp32 rows and their merge took 45 of B128's 820 us at 4096
            # keys, ET-DECODE6-h100; row 5385).
            with tuning.pinned({attention_mosaic.MosaicForward: at.config}):
                out = launch(rows, key, value, row_spans, whole, residuals=Kept.NONE, scaled=chosen.scaled,
                             bounded=chosen.bounded, masked=hooks.output_mod is None)
            return ungrouped_rows(out, heads)
        stateless = dataclasses.replace(hooks, initial=None, output_mod=None, output_params=())
        with tuning.pinned({attention_mosaic.MosaicForward: at.config}):
            walked_out = launch(rows, key, value, row_spans, stateless, residuals=Kept.OUTPUT,
                                out_dtype=jnp.float32, splits=at.slices, scaled=chosen.scaled,
                                bounded=chosen.bounded, masked=True)
        spec = _spec(rows, row_spans, stateless, scale=scale)
        queries, width = rows.shape[1], value.shape[-1]
        if at.slices == 1:
            # One slice's outputs `[B, Q, KV, Dv]` and rows `[B, KV, Q]`, a slice axis of one.
            assert not isinstance(walked_out, attention_mosaic.SplitOutput)
            out, maximum, total = walked_out[:3]
            out, maximum, total = out[:, None], maximum[:, :, None], total[:, :, None]
            if finished and defer_merge:
                return merged.merged_finished(
                    spec, out, jnp.stack([maximum, total]), queries=queries, width=width,
                    out_dtype=query.dtype, out_heads=heads, initial=initial,
                    output_mod=hooks.output_mod, output_params=hooks.output_params)
            state = algebra.merged_state(spec, out, maximum, total, initial)
        else:
            # The slices merged where the launch wrote them (`merged_finished` / `merged_split`), fusing any
            # `output_mod` directly into the native merge store (row 5385).
            assert isinstance(walked_out, attention_mosaic.SplitOutput)
            if finished:
                return merged.merged_finished(spec, walked_out.padded, walked_out.stacked, queries=queries,
                                              width=width, out_dtype=query.dtype, out_heads=heads,
                                              initial=initial, output_mod=hooks.output_mod,
                                              output_params=hooks.output_params)
            state = merged.merged_split(spec, walked_out.padded, walked_out.stacked, queries=queries,
                                        width=width, initial=initial)
        if finished:
            return ungrouped_rows(algebra.finish(spec, state, initial=True, normalized=True)[0],
                                  heads).astype(query.dtype)
        return _unpacked_state(algebra.resumable(spec, state), heads)

    def common() -> jax.Array | Initial:
        return walked("plain" if "plain" in work else "bounded")

    def run() -> jax.Array | Initial:
        if "plain" in work:
            return walked("plain")
        init_acc = None if initial is None else initial.acc
        if bounds is None and table is None and scales is None and sources is None:
            may = algebra.may_scale(value, keys_axis=1, initial=init_acc)
        else:
            largest = _largest(value, table, sources, scales) if bounds is None else _held(
                bounds, table, sources)
            may = algebra.bounded_past(largest, keys_axis=-1, initial=init_acc)
        return jax.lax.cond(may, lambda: walked("scaled"), lambda: walked("bounded"))

    return run, common


def priced_step(target: generation.GpuTarget, *, rows: int, queries: int, keys: int, heads: int,
                kv_heads: int, width: int, dtype, page_size: int | None = None) -> Work | None:
    """A decode step's work as `attend` launches it on `target` (a server's choice between a dense and a paged
    cache, row 7420): `rows` rows of `queries` query positions, `heads` query heads over `kv_heads` key heads
    `width` wide in `dtype`, over `keys` keys of a dense cache or, with `page_size`, of a paged pool's pages
    that long. Each Mosaic launch's work as its emission counts it (`emission.work_of`), the launches in
    sequence (their spans summed), and a split step's native `merged_padded` launch from `merged.launch_work`
    (leaving unobserved XLA merge work unpriced); None where some launch's work its emission does not
    count."""
    query = jax.ShapeDtypeStruct((rows, queries, heads, width), dtype)
    spans = key_spans(queries, keys, rows)
    pages = None if page_size is None else -(-keys // page_size)
    key = (jax.ShapeDtypeStruct((rows, keys, kv_heads, width), dtype) if pages is None
           else jax.ShapeDtypeStruct((kv_heads, rows * pages, page_size, width), dtype))
    table = None if pages is None else jax.ShapeDtypeStruct((rows, pages), jnp.int32)

    def step(q, k, v, t):
        # Price the common arm alone (never summing exclusive scaled/bounded fallback arms, row 8813 F2).
        _, common = _planned_step(q, k, v, spans, None, scale=None, interpret=False, table=t, sources=None,
                                  scales=None, bounds=None, finished=True, in_key=False)
        return common()

    with generation.assumed(target):
        traced = jax.make_jaxpr(step)(query, key, key, table)
    # A step whose walks derive no config refuses as it traces (`_planned_step`).
    kernels = (equation for equation in programs.equations(traced.jaxpr)
               if equation.primitive in programs.NATIVE_KERNELS
               and NATURAL_NAME not in equation.params["name"])
    found = [emission.work_of(kernel) for kernel in kernels]
    launches = [each for each in found if each is not None]
    if not launches or len(launches) < len(found):
        # A launch whose work its emission does not count leaves the step's price unknown, never 0 (the base
        # review of 56a716c95, F3: a full step's spans launch priced none, and a step of no split Work()).
        return None
    merges: list[Work] = []
    for equation in programs.equations(traced.jaxpr):
        if equation.primitive is not merged.merged_padded_p:
            continue
        padded_aval, stacked_aval = equation.invars[0].aval, equation.invars[1].aval
        if (isinstance(padded_aval, ShapedArray) and isinstance(stacked_aval, ShapedArray)
                and stacked_aval.dtype == jnp.float32):
            merges.append(merged.launch_work(
                equation.params["units"], jax.ShapeDtypeStruct(padded_aval.shape, padded_aval.dtype),
                jax.ShapeDtypeStruct(stacked_aval.shape, stacked_aval.dtype),
                equation.params["queries"], equation.params["width"], target,
                has_initial=len(equation.invars) > 2, raw_scale=equation.params.get("raw_scale"),
                out_dtype=equation.params.get("out_dtype"), out_heads=equation.params.get("out_heads")))
    all_launches = launches + merges
    return dataclasses.replace(sum(all_launches, Work()), span=sum(each.span for each in all_launches))

@overload
def attend(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None, *,
           scale: float | None = None, interpret: bool = False, table: jax.Array | None = None,
           sources: jax.Array | None = None, scales: tuple[jax.Array, jax.Array] | None = None,
           bounds: jax.Array | None = None, finished: Literal[True] = True, in_key: bool = False
           ) -> jax.Array: ...


@overload
def attend(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None, *,
           scale: float | None = None, interpret: bool = False, table: jax.Array | None = None,
           sources: jax.Array | None = None, scales: tuple[jax.Array, jax.Array] | None = None,
           bounds: jax.Array | None = None, finished: Literal[False], in_key: bool = False) -> Initial: ...


def attend(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None, *,
           scale: float | None = None, interpret: bool = False, table: jax.Array | None = None,
           sources: jax.Array | None = None, scales: tuple[jax.Array, jax.Array] | None = None,
           bounds: jax.Array | None = None, finished: bool = True, in_key: bool = False
           ) -> jax.Array | Initial:
    """A decode step's attention through the template: `query` `[B, T, H, D]` over `key` `[B, K, KV, D]` and
    `value` `[B, K, KV, Dv]` (`[R, K, KV, D]` with `sources`, batch row `b` reading key row `sources[b]`;
    or a paged cache's pools `[KV, P, page, D]` and their page table, `table`: each
    row's keys where it maps them), the keys `spans` `[B, T]` leaves each query under `hooks` (`packable`), as
    `attention.reference` computes it, `[B, T, H, Dv]` in the query's dtype: packed (`packed`), its keys
    sliced over the derived work list (`plan`) on the template's forward (`attention_mosaic.forward`'s
    `splits`), the slices merged with the initial state from their fp32 states (`merged.merged_split`),
    and unpacked. Its logits folded where the algebra folds them, the rows the
    fold's reach rewalks walked again natural over those rows alone, with no branch read on the host
    (`MosaicForward.masked`, row 1978). A quantized cache's keys and values are its codes (int8 or an FP8)
    and `scales` their keys' and values' `[B, KV, K]` fp32 in each row's key order (a dense cache's
    head-major scales, a paged pool's gathered by its table), dequantized in the walk
    (`attention_mosaic.forward`'s `scales`, ledger 3624). `bounds` holds each key's largest value magnitude,
    head-major as the scales are (`[R, KV, K]`, `[B, KV, K]` with no `sources`, a pool's `[KV, P, page]`): a
    cache's, which states it as it writes each key, so the call launches the walk its rows need, unscaled
    unless some row can pass fp32's largest (`algebra.bounded_past`, as a training call's `may_scale`),
    reading no value to choose it; given none, the call takes each key's from the values it reads, one pass
    over them (a quantized cache's codes times their scales), where the scaled walk a cache's call takes
    unchosen (`MosaicForward.cached`) took 2.25 times the unscaled kernel's time (a bf16 step at 8 rows over
    32768 keys, 64 wide, on an H100: 522 against 232 us, M-ET-61c and M-ET-49b). The prologues are taken
    as XLA's program before the launch; a paged or quantized cache's keys none (its pages and codes hold
    them under their prologue, taken as they were cached). A caller whose dense keys already hold their
    prologue passes none. Not `finished`, the step's rows' state in place of their output, the softmax over
    its keys and its initial state left unnormalized (`Initial`, `[B, H, T]` and `[B, H, T, Dv]` fp32, its
    maximum natural, `algebra.resumable`): another call over other keys that takes it as its initial state
    (`Hooks.initial`) merges the two (kernel-templates.md §A rows 19 and 63: a window's walk merged with a
    sparse part's)."""
    return _planned_step(query, key, value, spans, hooks, scale=scale, interpret=interpret, table=table,
                         sources=sources, scales=scales, bounds=bounds, finished=finished, in_key=in_key)[0]()


def _unpacked_state(state: algebra.Partial, heads: int) -> Initial:
    """A packed step's rows' state (`algebra.resumable`'s, rows `[B, T·G, KV]`) as its queries' `Initial`,
    `[B, H, T]` and `[B, H, T, Dv]` (`ungrouped_rows`)."""
    acc = jnp.swapaxes(ungrouped_rows(state.acc, heads), 1, 2)
    m, total = (jnp.swapaxes(ungrouped_rows(rows[..., None], heads)[..., 0], 1, 2)
                for rows in (state.m, state.total))
    return Initial(m, total, acc)


def _held(leaf: jax.Array, table: jax.Array | None, sources: jax.Array | None) -> jax.Array:
    """Each call row's keys' entries of a head-major per-key `leaf` (`[R, KV, K]`, `[B, KV, K]`, or a pool's
    `[KV, P, page]`), `[B, KV, K]`, as the row reads its keys: by its source row, or its table's pages. An
    unwritten key's bound is 0, and a key past a row's span counts too, which only raises its bound."""
    if table is not None:
        return jnp.moveaxis(leaf[:, table].reshape(leaf.shape[0], table.shape[0], -1), 0, 1)
    return leaf if sources is None else leaf[sources]


def _largest(value: jax.Array, table: jax.Array | None, sources: jax.Array | None,
             scales: tuple[jax.Array, jax.Array] | None) -> jax.Array:
    """Each call row's keys' largest value magnitudes, `[B, KV, K]`, from the values it reads (`attend`'s
    `value`, by its source rows or its table's pages): a quantized cache's codes' times their scales, which
    a dense cache holds by its storage rows and a paged call in its rows' key order (`attend`'s `scales`)."""
    magnitudes = jnp.max(jnp.abs(value.astype(jnp.float32)), axis=-1)
    # A dense cache's `[R, K, KV]` head-major as its leaves are; a pool's `[KV, P, page]` is.
    largest = _held(magnitudes if table is not None else jnp.moveaxis(magnitudes, 1, 2), table, sources)
    if scales is None:
        return largest
    value_scale = scales[1] if table is not None else _held(scales[1], None, sources)
    return largest * jnp.abs(value_scale.astype(jnp.float32))
