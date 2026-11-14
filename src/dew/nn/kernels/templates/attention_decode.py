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
  a fixed order, deterministic: the algebra's fold over their fp32 states in XLA (`algebra.merged_slices`).
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
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.extend import core as jaxpr_core

from dew.nn.kernels import generation, tuning
from dew.nn.kernels.templates import attention_algebra as algebra, attention_mosaic
from dew.nn.kernels.templates.attention import (
    Hooks,
    Initial,
    Kept,
    KeySpans,
    grouped_rows,
    prescale,
    prologues_taken,
    ungrouped_rows,
)


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
    state; `ungrouped_rows` takes an output back. A position's interval holds its G rows
    (`KeySpans.per_span`), read by position in the program; a call of several runs or documents an interval
    a row."""
    batch, positions, heads, _ = query.shape
    group = heads // kv_heads
    rows = grouped_rows(query, kv_heads)
    if spans.query_ids is None and spans.start.ndim == 2:
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

        initial = Initial(heads_rows(initial.m), heads_rows(initial.total), heads_rows(initial.acc))
    return rows, spans, initial


@dataclasses.dataclass(frozen=True)
class Plan:
    """A decode walk's work list as the derivation shapes it: the forward's config (`attention_mosaic.Tiles`)
    and the slices each query tile's keys are cut into (`MosaicForward.splits`), the config the launch's own
    op's at those slices."""

    config: attention_mosaic.Tiles
    slices: int


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
         walk: str = "plain") -> Plan | None:
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
    cuDNN's whole step took 13). Each slice writes its fp32 state, which XLA merges with the initial state
    (`algebra.merged_slices`), at a config derived for them."""
    chosen = WALKS[walk]
    found = _streamed(query, key, value, spans, hooks, scale=scale, target=target, table=table,
                      sources=sources, out_dtype=None, scales=scales, chosen=chosen)
    if found is None:
        return None
    config, op, ba = found
    processors = target.multiprocessors or 1
    tiles = math.ceil(query.shape[1] / (config.consumers * config.block_q))
    # Its launch's programs: a float32 call's operands as their parts stack along the rows
    # (`MosaicForward.parts`), as many programs again each (a merging launch's every one resident).
    programs = op.parts * query.shape[0] * query.shape[2] * tiles
    if programs >= processors:
        return Plan(config, 1)

    def sliced(config, op, ba) -> int:
        # The most slices whose programs keep the device's stream in flight at once (`MosaicForward.streamed`,
        # at most those it holds), none past them: one program past those it holds waits a whole walk for a
        # slot (ceil(132 / 8) = 17 slices of one row's 8 key heads is 136 programs, 4 of them a second wave as
        # long as the first), and one past those that stream it only lengthens the merge.
        longest = math.ceil(spans.keys / config.block_k) // (2 * config.stages)
        return max(1, min(processors * op.streamed(ba, config, target) // programs, longest))

    if sliced(config, op, ba) == 1:
        return Plan(config, 1)
    walked = None if hooks is None else dataclasses.replace(hooks, initial=None)
    streamed = functools.partial(_streamed, query, key, value, spans, walked, scale=scale, target=target,
                                 table=table, sources=sources, out_dtype=jnp.float32, scales=scales,
                                 chosen=chosen)
    found = streamed()
    slices = 1 if found is None else sliced(*found)
    # The sliced launch's own op, at its slices.
    found = None if slices == 1 else streamed(splits=slices)
    return Plan(config, 1) if found is None else Plan(found[0], slices)


def _streamed(query, key, value, spans: KeySpans, hooks: Hooks | None, *, scale: float | None,
              target: generation.GpuTarget, table, sources, out_dtype, scales=None, splits: int = 1,
              chosen: Walk = WALKS["plain"]):
    """The forward's derived config for a packed call's launch of the walk `chosen` names (`WALKS`) writing
    `out_dtype` (the query's where None) over `splits` slices, at the key tile its walk streams at
    (`MosaicForward.streaming`), with its op and bound arguments; None where none derives."""
    op, args = attention_mosaic._forward_call(query, key, value, spans, hooks, scale=scale, natural=False,
                                              out_dtype=out_dtype, masked=True, splits=splits, table=table,
                                              sources=sources, subtracted=True, scales=scales,
                                              scaled=chosen.scaled, bounded=chosen.bounded)
    ba = op.bind(*args, residuals=Kept.OUTPUT if out_dtype is not None else Kept.NONE, interpret=False)
    config = op.derived(ba, target)
    # A decode step's few rows a key head stream its keys (`MosaicForward.streaming`).
    return None if config is None else (op.streaming(ba, config, target), op, ba)


def _spec(query: jax.Array, spans: KeySpans, hooks: Hooks, *, scale: float | None) -> algebra.Forward:
    """The call's form as the template's forward takes it (`attention_mosaic_forward._prepared`), its fold of
    raw products (`algebra.Forward.subtracted`): the units its finished rows' maxima are in
    (`algebra.finished_form`), which its merge takes."""
    power, factor = prescale(query.dtype, query.shape[3], scale)
    spec = algebra.Forward(spans.form, spans.keys, ids=spans.query_ids is not None, scale=factor,
                           score_mod=hooks.score_mod, natural=False, initial=hooks.initial is not None,
                           prescale=power, subtracted=True)
    return algebra.finished_form(spec)


def planned(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks, *,
            scale: float | None = None, table: jax.Array | None = None, sources: jax.Array | None = None,
            scales: tuple[jax.Array, jax.Array] | None = None
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
    work = {}
    for walk in launched(query, value, scales):
        found = plan(rows, key, value, row_spans, whole, scale=scale, target=target, table=table,
                     sources=sources, scales=scales, walk=walk)
        if found is None:
            return None, rows, row_spans, whole
        work[walk] = found
    return work, rows, row_spans, whole


def attend(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None, *,
           scale: float | None = None, interpret: bool = False, table: jax.Array | None = None,
           sources: jax.Array | None = None, scales: tuple[jax.Array, jax.Array] | None = None,
           bounds: jax.Array | None = None) -> jax.Array:
    """A decode step's attention through the template: `query` `[B, T, H, D]` over `key` `[B, K, KV, D]` and
    `value` `[B, K, KV, Dv]` (`[R, K, KV, D]` with `sources`, batch row `b` reading key row `sources[b]`;
    or a paged cache's pools `[KV, P, page, D]` and their page table, `table`: each
    row's keys where it maps them), the keys `spans` `[B, T]` leaves each query under `hooks` (`packable`), as
    `attention.reference` computes it, `[B, T, H, Dv]` in the query's dtype: packed (`packed`), its keys
    sliced over the derived work list (`plan`) on the template's forward (`attention_mosaic.forward`'s
    `splits`), the slices merged with the initial state in XLA from their fp32 states
    (`algebra.merged_slices`), and unpacked. Its logits folded where the algebra folds them, the rows the
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
    prologue passes none."""
    hooks = Hooks() if hooks is None else hooks
    if hooks.key_mod is not None and (table is not None or scales is not None):
        raise ValueError("a paged or quantized cache holds its keys under their prologue, taken as they were "
                         "cached: a call over its pages or codes takes no key prologue")
    query, key, hooks = prologues_taken(query, key, hooks)
    if not packable(hooks):
        raise NotImplementedError("the template's decode packs no query-placed hook, parameter, output "
                                  "hook or dropout (`packable`)")
    heads = query.shape[2]
    work, rows, row_spans, whole = planned(query, key, value, spans, hooks, scale=scale, table=table,
                                           sources=sources, scales=scales)
    initial = whole.initial
    if work is None:
        raise NotImplementedError(f"the template's decode derives no config on {generation.gpu_target()}")
    launch = functools.partial(attention_mosaic.forward, scale=scale, natural=False, interpret=interpret,
                               masked=True, table=table, sources=sources, subtracted=True, scales=scales)

    def walked(walk: str) -> jax.Array:
        # The step's walk (`WALKS`) at its own plan: scaled or unscaled as it names, else a cache call's own
        # (`MosaicForward.cached`).
        chosen, at = WALKS[walk], work[walk]
        if at.slices == 1:
            # A row's walk from the initial state, finished in the query's dtype, no merge past it (its fp32
            # rows and their merge took 45 of B128's 820 us at 4096 keys, ET-DECODE6-h100).
            with tuning.pinned({attention_mosaic.MosaicForward: at.config}):
                out = launch(rows, key, value, row_spans, whole, residuals=Kept.NONE, scaled=chosen.scaled,
                             bounded=chosen.bounded)
            return ungrouped_rows(out, heads)
        stateless = dataclasses.replace(hooks, initial=None)
        with tuning.pinned({attention_mosaic.MosaicForward: at.config}):
            out, maximum, total = launch(
                rows, key, value, row_spans, stateless, residuals=Kept.OUTPUT, out_dtype=jnp.float32,
                splits=at.slices, scaled=chosen.scaled, bounded=chosen.bounded)[:3]
        merged_rows = algebra.merged_slices(_spec(rows, row_spans, stateless, scale=scale), out, maximum,
                                            total, initial)
        return ungrouped_rows(merged_rows, heads).astype(query.dtype)

    if "plain" in work:
        return walked("plain")
    largest = _largest(value, table, sources, scales) if bounds is None else _held(bounds, table, sources)
    may = algebra.bounded_past(largest, keys_axis=-1, initial=None if initial is None else initial.acc)
    return jax.lax.cond(may, lambda: walked("scaled"), lambda: walked("bounded"))


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
