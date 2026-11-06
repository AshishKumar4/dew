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
  a fixed order, deterministic: in the launch by the last slice's program where the family waits across
  programs (`MosaicForward.merges`, `Family.fixed_up`), else by the algebra's fold over their fp32 states in
  XLA (`algebra.merged_slices`).

No arithmetic of its own: the walk, the hooks, the guards and the merge are the template forward's and its
algebra's.
"""

import dataclasses
import math

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
    ungrouped_rows,
)
from dew.nn.kernels.templates.attention_mosaic_program import bundle_of


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
    its walk (`attention_mosaic.streams`): a decode step's, a speculative verify block's. A verify block of 4
    to 16 drafts a row through the forward took 1.1 to 5.9 ms where the work list took 30 us to 1.36 ms on an
    H100, its query heads read the keys once a key head (ET-VERIFY-h100)."""
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
        spans = KeySpans(spans.start, spans.end, None, spans.key_ids, spans.keys, "spans", group)
    else:
        spans = KeySpans(*(None if x is None else jnp.repeat(x, group, axis=1)
                           for x in (spans.start, spans.end, spans.query_ids)),
                         spans.key_ids, spans.keys, "spans")
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
    """A decode step's work list as the derivation shapes it: the forward's config (`attention_mosaic.Tiles`),
    the slices each query tile's keys are cut into (`MosaicForward.splits`) and whether the launch merges them
    itself (`MosaicForward.merges`), its output the rows' whole in the query's dtype."""

    config: attention_mosaic.Tiles
    slices: int
    merges: bool


def plan(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None, *,
         scale: float | None, target: generation.GpuTarget, table: jax.Array | None = None,
         sources: jax.Array | None = None) -> Plan | None:
    """The work list of a packed call (`packed`) under its `hooks` (its initial state among them) on `target`:
    the forward's derived config at the key tile its memory-bound walk streams at least
    (`MosaicForward.streaming`) and its slices; None where no launch of it derives.
    One where the programs already fill the multiprocessors (a slice's merge would cost and its keys gain
    nothing). Else as many as its programs' residency holds at once (`MosaicForward.resident`: its
    registers', its shared memory's), but each slice at least twice its ring's stages of key tiles: a shorter
    slice's walk spends its prologue's loads, `stages` tiles, without as many steps of products to hide them
    (ET-DECODE2-h100: 32 slices of one key tile each, at batch 1 over 4096 keys, took 22 us where cuDNN's
    whole step took 13). Where the family waits across programs (`Family.fixes_up`), the launch merges the
    slices in the last one's program, the initial state with them (`MosaicForward.merges`); elsewhere each
    slice writes its fp32 state, which XLA merges (`algebra.merged_slices`), at a config derived for them."""
    found = _streamed(query, key, value, spans, hooks, scale=scale, target=target, table=table,
                      sources=sources, out_dtype=None)
    if found is None:
        return None
    config, op, ba = found
    processors = target.multiprocessors or 1
    tiles = math.ceil(query.shape[1] / (config.consumers * config.block_q))
    programs = query.shape[0] * query.shape[2] * tiles
    if programs >= processors:
        return Plan(config, 1, merges=False)
    if not bundle_of(config, query.dtype).fixes_up:
        walked = None if hooks is None else dataclasses.replace(hooks, initial=None)
        found = _streamed(query, key, value, spans, walked, scale=scale, target=target, table=table,
                          sources=sources, out_dtype=jnp.float32)
        if found is None:
            return Plan(config, 1, merges=False)
        config, op, ba = found
    longest = math.ceil(spans.keys / config.block_k) // (2 * config.stages)
    # The most slices whose programs the device holds at once, none past it: one program past them waits a
    # whole walk for a slot (ceil(132 / 8) = 17 slices of one row's 8 key heads is 136 programs, 4 of them a
    # second wave as long as the first), and a merging launch's every program resident.
    slices = max(1, min(processors * op.resident(ba, config, target) // programs, longest))
    return Plan(config, slices, merges=slices > 1 and bundle_of(config, query.dtype).fixes_up)


def _streamed(query, key, value, spans: KeySpans, hooks: Hooks | None, *, scale: float | None,
              target: generation.GpuTarget, table, sources, out_dtype):
    """The forward's derived config for a packed call writing `out_dtype` (the query's where None) at the
    key tile its walk streams at (`MosaicForward.streaming`), with its op and bound arguments; None where
    none derives."""
    op, args = attention_mosaic._forward_call(query, key, value, spans, hooks, scale=scale, natural=False,
                                              out_dtype=out_dtype, masked=True, table=table,
                                              sources=sources, subtracted=True)
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


def attend(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None, *,
           scale: float | None = None, interpret: bool = False, table: jax.Array | None = None,
           sources: jax.Array | None = None) -> jax.Array:
    """A decode step's attention through the template: `query` `[B, T, H, D]` over `key` `[B, K, KV, D]` and
    `value` `[B, K, KV, Dv]` (`[R, K, KV, D]` with `sources`, batch row `b` reading key row `sources[b]`;
    or a paged cache's pools `[KV, P, page, D]` and their page table, `table`: each
    row's keys where it maps them), the keys `spans` `[B, T]` leaves each query under `hooks` (`packable`), as
    `attention.reference` computes it, `[B, T, H, Dv]` in the query's dtype: packed (`packed`), its keys
    sliced over the derived work list (`plan`) on the template's forward (`attention_mosaic.forward`'s
    `splits`), the slices merged with the initial state in the launch where its family waits across programs
    (`merges`), else in XLA from their fp32 states (`algebra.merged_slices`), and unpacked. Its logits folded
    where the algebra folds them, the rows the fold's reach or the headroom guard rewalks walked again
    natural over those rows alone, with no branch read on the host (`MosaicForward.masked`, row 1978)."""
    hooks = Hooks() if hooks is None else hooks
    if not packable(hooks):
        raise NotImplementedError("the template's decode packs no query-placed hook, parameter, output "
                                  "hook or dropout (`packable`)")
    heads, kv_heads = query.shape[2], (key.shape[2] if table is None else key.shape[0])
    rows, row_spans, initial = packed(query, spans, kv_heads, hooks.initial)
    whole = dataclasses.replace(hooks, initial=initial)
    target = generation.gpu_target()
    work = plan(rows, key, value, row_spans, whole, scale=scale, target=target, table=table, sources=sources)
    if work is None:
        raise NotImplementedError(f"the template's decode derives no config on {target}")
    form = {"scale": scale, "natural": False, "interpret": interpret, "masked": True, "table": table,
            "sources": sources, "subtracted": True}
    if work.slices == 1 or work.merges:
        # A row's walk, or its slices merged in the launch, from the initial state, finished in the query's
        # dtype, no merge past it (its fp32 rows and their merge took 45 of B128's 820 us at 4096 keys,
        # ET-DECODE6-h100; XLA's merge of the slices, four fusions, ~6.9 us of B1's step, ET-DPROF-h100).
        with tuning.pinned({attention_mosaic.MosaicForward: work.config}):
            out = attention_mosaic.forward(rows, key, value, row_spans, whole, residuals=Kept.NONE,
                                           splits=work.slices, merges=work.merges, **form)
        return ungrouped_rows(out, heads)
    walked = dataclasses.replace(hooks, initial=None)
    with tuning.pinned({attention_mosaic.MosaicForward: work.config}):
        out, maximum, total = attention_mosaic.forward(
            rows, key, value, row_spans, walked, residuals=Kept.OUTPUT, out_dtype=jnp.float32,
            splits=work.slices, **form)[:3]
    merged_rows = algebra.merged_slices(_spec(rows, row_spans, walked, scale=scale), out, maximum, total,
                                        initial)
    return ungrouped_rows(merged_rows, heads).astype(query.dtype)
