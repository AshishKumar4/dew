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
from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal, NamedTuple, overload

import jax
import jax.numpy as jnp
from jax.extend import core as jaxpr_core

from dew.nn import programs
from dew.nn.kernels import generation, tuning
from dew.nn.kernels.templates import attention_algebra as algebra, attention_mosaic, emission, merged
from dew.nn.kernels.templates.attention import (
    Hooks,
    Initial,
    Kept,
    KeySpans,
    grouped_rows,
    prologues_taken,
    spans as key_spans,
    ungrouped_rows,
)
from dew.nn.kernels.templates.attention_mosaic_forward import folded_form
from dew.nn.kernels.templates.cost import Work


def packable(hooks: Hooks | None) -> bool:
    """Whether a call's query heads can become its rows (`packed`): no score hook reading a query's place or
    any parameter (a packed row's place is its pair's, not its query's), no output hook and no dropout; a
    softcap (the scaled logits alone) and sinks (an initial state a head) pack."""
    if hooks is None:
        return True
    if hooks.score_params or hooks.output_mod is not None or hooks.drop is not None:
        return False
    return hooks.score_mod is None or not _reads_places(hooks.score_mod)


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
         walk: str = "plain", in_key: bool = False) -> Plan | None:
    """The work list of a packed call (`packed`) under its `hooks` (its initial state among them) on `target`,
    its `walk` (`WALKS`): the forward's derived config at the key tile its memory-bound walk streams at least
    (`MosaicForward.streaming`) and its slices; None where no launch of it derives. Each config is derived
    for the op its launch runs (its walk, its output's dtype and residuals, its slices), so it reads that
    op's refusals (`report.refuse_launch`): a scaled launch ptxas refused steps down at its own config, never
    at the plain op's, which the pin would hold it to (row 5420).
    One where the programs already fill the multiprocessors (a slice's merge would cost and its keys gain
    nothing). Else as many as keep the device's stream in flight (`MosaicForward.streamed`, at most those its
    residency holds, `MosaicForward.resident`), but each slice at least twice its ring's stages of key tiles:
    a shorter slice's walk spends its prologue's loads, `stages` tiles, without as many steps of products to
    hide them (ET-DECODE2-h100: 32 slices of one key tile each, at batch 1 over 4096 keys, took 22 us where
    cuDNN's whole step took 13). Each slice writes its fp32 state, which the step merges with the initial
    state (`merged.merged_split`), at a config derived for them."""
    chosen = WALKS[walk]
    found = _streamed(query, key, value, spans, hooks, scale=scale, target=target, table=table,
                      sources=sources, out_dtype=None, scales=scales, chosen=chosen, in_key=in_key)
    if found is None:
        return None
    config, op, ba = found

    def planned(config, op, ba, slices: int) -> Plan:
        return Plan(config, slices, op.deciding(ba, config))

    processors = target.multiprocessors or 1
    tiles = math.ceil(query.shape[1] / (config.consumers * config.block_q))
    # Its launch's programs: a float32 call's operands as their parts stack along the rows
    # (`MosaicForward.parts`), as many programs again each (a merging launch's every one resident).
    programs = op.parts * query.shape[0] * query.shape[2] * tiles
    if programs >= processors:
        return planned(config, op, ba, 1)

    def sliced(config, op, ba) -> int:
        # The most slices whose programs keep the device's stream in flight at once (`MosaicForward.streamed`,
        # at most those it holds), none past them: one program past those it holds waits a whole walk for a
        # slot (ceil(132 / 8) = 17 slices of one row's 8 key heads is 136 programs, 4 of them a second wave as
        # long as the first), and one past those that stream it only lengthens the merge.
        longest = math.ceil(spans.keys / config.block_k) // (2 * config.stages)
        return max(1, min(processors * op.streamed(ba, config, target) // programs, longest))

    if sliced(config, op, ba) == 1:
        return planned(config, op, ba, 1)
    walked = None if hooks is None else dataclasses.replace(hooks, initial=None)
    streamed = functools.partial(_streamed, query, key, value, spans, walked, scale=scale, target=target,
                                 table=table, sources=sources, out_dtype=jnp.float32, scales=scales,
                                 chosen=chosen, in_key=in_key)
    found = streamed()
    slices = 1 if found is None else sliced(*found)
    # The sliced launch's own op, at its slices.
    found = None if slices == 1 else streamed(splits=slices)
    return planned(config, op, ba, 1) if found is None else planned(*found, slices)


def _streamed(query, key, value, spans: KeySpans, hooks: Hooks | None, *, scale: float | None,
              target: generation.GpuTarget, table, sources, out_dtype, scales=None, splits: int = 1,
              chosen: Walk = WALKS["plain"], in_key: bool = False):
    """The forward's config for a packed call's launch of the walk `chosen` names (`WALKS`) writing
    `out_dtype` (the query's where None) over `splits` slices, certified at the key tile its walk streams at
    (`_certified_stream`), with its op and bound arguments, the op its launch runs (`in_key` its values'
    stream or the key's slot); None where none derives or certification refuses every config."""
    op, args = attention_mosaic._forward_call(query, key, value, spans, hooks, scale=scale, natural=False,
                                              out_dtype=out_dtype, masked=True, splits=splits, table=table,
                                              sources=sources, subtracted=True, scales=scales,
                                              scaled=chosen.scaled, bounded=chosen.bounded, in_key=in_key)
    ba = op.bind(*args, residuals=Kept.OUTPUT if out_dtype is not None else Kept.NONE, interpret=False)
    config = _certified_stream(op, ba, target)
    return None if config is None else (config, op, ba)


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

    def streamed() -> attention_mosaic.Tiles | None:
        # Its next streamed tile, none past the taken config itself: a pin or tuned pick is a measured choice
        # `_certified` leaves to its program's compile (`report.settled`), a derived one `taken` certified.
        tile = op.streaming(ba, config, target)
        return None if tile == config else tile

    first = streamed()
    found = None if first is None else op._certified(ba, first, offered=streamed)
    return found if isinstance(found, attention_mosaic.Tiles) else config


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
    whole = dataclasses.replace(hooks, initial=initial)
    target = generation.gpu_target()
    planned_walk = functools.partial(plan, rows, key, value, row_spans, whole, scale=scale, target=target,
                                     table=table, sources=sources, scales=scales, in_key=in_key)
    # The plain walk first: where its launch decides a scaled second walk itself (`Plan.decides`), it is the
    # step's one walk, whatever its rows' sums can reach; else the scaled and the bounded walks, which its
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


def priced_step(target: generation.GpuTarget, *, rows: int, queries: int, keys: int, heads: int,
                kv_heads: int, width: int, dtype, page_size: int | None = None) -> Work | None:
    """A decode step's work as `attend` launches it on `target` (a server's choice between a dense and a paged
    cache, row 7420): `rows` rows of `queries` query positions, `heads` query heads over `kv_heads` key heads
    `width` wide in `dtype`, over `keys` keys of a dense cache or, with `page_size`, of a paged pool's pages
    that long. Each Mosaic launch's work as its emission counts it (`emission.work_of`), the launches in
    sequence (their spans summed), and a split step's native `merged_padded` launch from
    `merged.launch_work` (leaving unobserved XLA merge work unpriced); None where some launch's work its
    emission does not count."""
    query = jax.ShapeDtypeStruct((rows, queries, heads, width), dtype)
    spans = key_spans(queries, keys, rows)
    pages = None if page_size is None else -(-keys // page_size)
    key = (jax.ShapeDtypeStruct((rows, keys, kv_heads, width), dtype) if pages is None
           else jax.ShapeDtypeStruct((kv_heads, rows * pages, page_size, width), dtype))
    table = None if pages is None else jax.ShapeDtypeStruct((rows, pages), jnp.int32)

    def step(q, k, v, t):
        return attend(q, k, v, spans, None, scale=None, interpret=False, table=t, sources=None,
                      scales=None, bounds=None, finished=True, in_key=False)

    with generation.assumed(target):
        traced = jax.make_jaxpr(step)(query, key, key, table)
    # A step whose walks derive no config refuses as it traces (`attend`).
    kernels = (equation for equation in programs.equations(traced.jaxpr)
               if equation.primitive in programs.NATIVE_KERNELS)
    found = [emission.work_of(kernel) for kernel in kernels]
    launches = [work for work in found if work is not None]
    if not launches or len(launches) < len(found):
        # A launch whose work its emission does not count leaves the step's price unknown, never 0 (the base
        # review of 56a716c95, F3: a full step's spans launch priced none, and a step of no split Work()).
        return None
    merges: list[Work] = []
    for equation in programs.equations(traced.jaxpr):
        if equation.primitive is not merged.merged_padded_p:
            continue
        padded_aval, stacked_aval = equation.invars[0].aval, equation.invars[1].aval
        if (isinstance(padded_aval, jax.core.ShapedArray) and isinstance(stacked_aval, jax.core.ShapedArray)
                and stacked_aval.dtype == jnp.float32):
            merges.append(merged.launch_work(
                equation.params["spec"], jax.ShapeDtypeStruct(padded_aval.shape, padded_aval.dtype),
                jax.ShapeDtypeStruct(stacked_aval.shape, stacked_aval.dtype),
                equation.params["queries"], equation.params["width"], target))
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
    hooks = Hooks() if hooks is None else hooks
    if hooks.key_mod is not None and (table is not None or scales is not None):
        raise ValueError("a paged or quantized cache holds its keys under their prologue, taken as they were "
                         "cached: a call over its pages or codes takes no key prologue")
    in_key = attention_mosaic.reads_in_key(in_key, key, value, hooks, table=table, scales=scales)
    query, key, hooks = prologues_taken(query, key, hooks)
    if not packable(hooks):
        raise NotImplementedError("the template's decode packs no query-placed hook, parameter, output "
                                  "hook or dropout (`packable`)")
    heads = query.shape[2]
    work, rows, row_spans, whole = planned(query, key, value, spans, hooks, scale=scale, table=table,
                                           sources=sources, scales=scales, in_key=in_key)
    initial = whole.initial
    if work is None:
        raise NotImplementedError(f"the template's decode derives no config on {generation.gpu_target()}")
    launch = functools.partial(attention_mosaic.forward, scale=scale, natural=False, interpret=interpret,
                               masked=True, table=table, sources=sources, subtracted=True, scales=scales,
                               in_key=in_key)

    def walked(walk: str) -> jax.Array | Initial:
        # The step's walk (`WALKS`) at its own plan: scaled or unscaled as it names, else a cache call's own
        # (`MosaicForward.cached`).
        chosen, at = WALKS[walk], work[walk]
        if at.slices == 1 and finished:
            # A row's walk from the initial state, finished in the query's dtype, no merge past it (its fp32
            # rows and their merge took 45 of B128's 820 us at 4096 keys, ET-DECODE6-h100).
            with tuning.pinned({attention_mosaic.MosaicForward: at.config}):
                out = launch(rows, key, value, row_spans, whole, residuals=Kept.NONE, scaled=chosen.scaled,
                             bounded=chosen.bounded)
            return ungrouped_rows(out, heads)
        stateless = dataclasses.replace(hooks, initial=None)
        with tuning.pinned({attention_mosaic.MosaicForward: at.config}):
            walked_out = launch(rows, key, value, row_spans, stateless, residuals=Kept.OUTPUT,
                                out_dtype=jnp.float32, splits=at.slices, scaled=chosen.scaled,
                                bounded=chosen.bounded)
        spec = _spec(rows, row_spans, stateless, scale=scale)
        queries, width = rows.shape[1], value.shape[-1]
        if at.slices == 1:
            # One slice's outputs `[B, Q, KV, Dv]` and rows `[B, KV, Q]`, a slice axis of one.
            out, maximum, total = walked_out[:3]
            out, maximum, total = out[:, None], maximum[:, :, None], total[:, :, None]
            state = algebra.merged_state(spec, out, maximum, total, initial)
        else:
            # The slices merged where the launch wrote them (`merged.merged_split`).
            state = merged.merged_split(spec, walked_out[0], walked_out[1], queries=queries, width=width,
                                        initial=initial)
        if finished:
            merged_rows = algebra.finish(spec, state, initial=True, normalized=True)[0]
            return ungrouped_rows(merged_rows, heads).astype(query.dtype)
        return _unpacked_state(algebra.resumable(spec, state), heads)

    if "plain" in work:
        return walked("plain")
    largest = _largest(value, table, sources, scales) if bounds is None else _held(bounds, table, sources)
    may = algebra.bounded_past(largest, keys_axis=-1, initial=None if initial is None else initial.acc)
    return jax.lax.cond(may, lambda: walked("scaled"), lambda: walked("bounded"))


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
