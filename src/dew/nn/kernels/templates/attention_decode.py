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
- `merged`: the slices' softmax states, each its output normalized in fp32, its maximum and its
  denominator, combined with the call's initial state (a sink, `Initial`) as one more by the algebra's
  fold over states (`algebra.merged`, `finish`), one sum a row, deterministic.

No arithmetic of its own past the merge: the walk, the hooks and the guards are the template forward's.
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
    state; `ungrouped_rows` takes an output back."""
    batch, positions, heads, _ = query.shape
    group = heads // kv_heads
    rows = grouped_rows(query, kv_heads)

    def repeated(x):
        return None if x is None else jnp.repeat(x, group, axis=1)

    spans = KeySpans(jnp.repeat(spans.start, group, axis=1), jnp.repeat(spans.end, group, axis=1),
                     repeated(spans.query_ids), spans.key_ids, spans.keys, "spans")
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
    """A decode step's work list as the derivation shapes it: the forward's config (`attention_mosaic.Tiles`)
    and the slices each query tile's keys are cut into (`MosaicForward.splits`)."""

    config: attention_mosaic.Tiles
    slices: int


def plan(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None, *,
         scale: float | None, target: generation.GpuTarget, table: jax.Array | None = None,
         sources: jax.Array | None = None) -> Plan | None:
    """The work list of a packed call (`packed`) on `target`: the forward's derived config at the key tile
    its memory-bound walk streams at least (`MosaicForward.streaming`) and its slices; None where no launch
    of it derives.
    One where the programs already fill the multiprocessors (a slice's merge would cost and its keys gain
    nothing), or where no split launch derives (its slices' fp32 outputs staged beside the query's: a
    256-wide bf16 step's 64-key tile needed 234,576 bytes of an H100's 232,448, dd-d256), whose one slice
    a row finishes in the query's dtype through the query's stage. Else as many as its programs'
    residency holds at once (`Family.programs`, its registers; its shared memory's), but each slice at
    least twice its ring's stages of key tiles: a shorter slice's walk spends its prologue's loads, `stages`
    tiles, without as many steps of products to hide them (ET-DECODE2-h100: 32 slices of one key tile each,
    at batch 1 over 4096 keys, took 22 us where cuDNN's whole step took 13)."""
    split = _streamed(query, key, value, spans, hooks, scale=scale, target=target, table=table,
                      sources=sources, out_dtype=jnp.float32)
    if split is None:
        whole = _streamed(query, key, value, spans, hooks, scale=scale, target=target, table=table,
                          sources=sources, out_dtype=None)
        return None if whole is None else Plan(whole[0], 1)
    config, op, ba = split
    processors = target.multiprocessors or 1
    tiles = math.ceil(query.shape[1] / (config.consumers * config.block_q))
    programs = query.shape[0] * query.shape[2] * tiles
    if programs >= processors:
        return Plan(config, 1)
    family = bundle_of(config, query.dtype)
    resident = family.programs(config.consumers, op.registers(ba, config))
    if target.shared_limit:
        resident = min(resident, max(1, target.shared_limit // op.staged_bytes(ba, config, target)))
    longest = math.ceil(spans.keys / config.block_k) // (2 * config.stages)
    # The most slices whose programs the device holds at once, none past it: one program past them waits a
    # whole walk for a slot (ceil(132 / 8) = 17 slices of one row's 8 key heads is 136 programs, 4 of them a
    # second wave as long as the first).
    return Plan(config, max(1, min(processors * resident // programs, longest)))


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


def merged(spec: algebra.Forward, out: jax.Array, maximum: jax.Array, total: jax.Array,
           initial: Initial | None = None) -> jax.Array:
    """The slices' outputs `out` `[B, S, Q, KV, Dv]` (fp32, each normalized), maxima (in natural `spec`'s
    units, `algebra.finished_rows`', a fold's `algebra.unfolded`) and denominators `[B, KV, S, Q]`, and the
    call's `initial` state `[B, KV, Q]` (its sums `acc` unnormalized) merged by the algebra's fold over
    states (`algebra.merged`, then `finish`): the rows' output `[B, Q, KV, Dv]` fp32. Each state enters as
    its normalized output and its mass, its denominator folded into its maximum (`m + log(total)`), so no
    output multiplied back by its denominator passes fp32's largest; a state of no mass adds nothing, and a
    row of none is zero, as the reference's."""
    if not (spec.natural or spec.raw):
        raise ValueError("the template's decode merges natural states or a raw fold's")
    m, mass = (jnp.transpose(array, (0, 2, 3, 1)) for array in (maximum, total))
    if initial is not None:
        held = initial.total > 0
        own = jnp.where(held[..., None], initial.acc / jnp.where(held, initial.total, 1.0)[..., None], 0.0)
        out = jnp.concatenate([out, jnp.swapaxes(own, 1, 2)[:, None]], axis=1)
        # The initial maximum natural, the slices' in `spec`'s units (raw products where it keeps them).
        start = initial.m / spec.scale if spec.raw else initial.m
        m = jnp.concatenate([m, jnp.swapaxes(start, 1, 2)[:, None]], axis=1)
        mass = jnp.concatenate([mass, jnp.swapaxes(initial.total, 1, 2)[:, None]], axis=1)
    seen = mass > 0
    units = spec.scale if spec.raw else 1.0
    folded = jnp.where(seen, m + jnp.log(jnp.where(seen, mass, 1.0)) / units, -jnp.inf)
    state = algebra.merged(spec, algebra.Partial(out, folded, seen.astype(jnp.float32)), axis=1)
    return algebra.finish(spec, state, initial=True)[0]


def _spec(query: jax.Array, spans: KeySpans, hooks: Hooks, *, scale: float | None) -> algebra.Forward:
    """The call's form as the template's forward takes it (`attention_mosaic_forward._prepared`), its fold of
    raw products (`algebra.Forward.subtracted`): the units its finished rows' maxima are in
    (`algebra.finished_rows`: raw products where its fold keeps them, else natural, a score hook's fold's
    rewalk's, `algebra.unfolded`), which its merge takes."""
    power, factor = prescale(query.dtype, query.shape[3], scale)
    spec = algebra.Forward(spans.form, spans.keys, ids=spans.query_ids is not None, scale=factor,
                           score_mod=hooks.score_mod, natural=False, initial=hooks.initial is not None,
                           prescale=power, subtracted=True)
    return spec if spec.natural or spec.raw else algebra.unfolded(spec)


def attend(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None, *,
           scale: float | None = None, interpret: bool = False, table: jax.Array | None = None,
           sources: jax.Array | None = None) -> jax.Array:
    """A decode step's attention through the template: `query` `[B, T, H, D]` over `key` `[B, K, KV, D]` and
    `value` `[B, K, KV, Dv]` (`[R, K, KV, D]` with `sources`, batch row `b` reading key row `sources[b]`;
    or a paged cache's pools `[KV, P, page, D]` and their page table, `table`: each
    row's keys where it maps them), the keys `spans` `[B, T]` leaves each query under `hooks` (`packable`), as
    `attention.reference` computes it, `[B, T, H, Dv]` in the query's dtype: packed (`packed`), its keys
    sliced over the derived work list (`plan`) on the template's forward in fp32 with its rows
    (`attention_mosaic.forward`'s `splits`), merged with the initial state (`merged`) and unpacked; a row of
    one slice finished by its own walk from the initial state, in the query's dtype. Its
    logits folded where the algebra folds them, the rows the fold's reach or the headroom guard rewalks
    walked again natural in one more launch over those rows alone, with no branch read on the host
    (`MosaicForward.masked`, row 1978)."""
    hooks = Hooks() if hooks is None else hooks
    if not packable(hooks):
        raise NotImplementedError("the template's decode packs no query-placed hook, parameter, output "
                                  "hook or dropout (`packable`)")
    heads, kv_heads = query.shape[2], (key.shape[2] if table is None else key.shape[0])
    rows, row_spans, initial = packed(query, spans, kv_heads, hooks.initial)
    walked = dataclasses.replace(hooks, initial=None)
    target = generation.gpu_target()
    work = plan(rows, key, value, row_spans, walked, scale=scale, target=target, table=table, sources=sources)
    if work is None:
        raise NotImplementedError(f"the template's forward derives no decode step's config on {target}")
    if work.slices == 1:
        # One slice a row: its walk merges the initial state itself (`algebra.merge`) and finishes its rows in
        # the query's dtype, no merge past it (its fp32 rows and their merge took 45 of B128's 820 us at
        # 4096 keys, ET-DECODE6-h100).
        whole = dataclasses.replace(hooks, initial=initial)
        found = _streamed(rows, key, value, row_spans, whole, scale=scale, target=target, table=table,
                          sources=sources, out_dtype=None)
        if found is None:
            raise NotImplementedError(f"the template's forward derives no decode step's config on {target}")
        work = Plan(found[0], 1)
        with tuning.pinned({attention_mosaic.MosaicForward: work.config}):
            out = attention_mosaic.forward(rows, key, value, row_spans, whole, scale=scale, natural=False,
                                           residuals=Kept.NONE, interpret=interpret, masked=True, table=table,
                                           sources=sources, subtracted=True)
        return ungrouped_rows(out, heads)
    with tuning.pinned({attention_mosaic.MosaicForward: work.config}):
        out, maximum, total = attention_mosaic.forward(
            rows, key, value, row_spans, walked, scale=scale, natural=False, residuals=Kept.OUTPUT,
            interpret=interpret, out_dtype=jnp.float32, masked=True, splits=work.slices, table=table,
            sources=sources, subtracted=True)[:3]
    merged_rows = merged(_spec(rows, row_spans, walked, scale=scale), out, maximum, total, initial)
    return ungrouped_rows(merged_rows, heads).astype(query.dtype)
