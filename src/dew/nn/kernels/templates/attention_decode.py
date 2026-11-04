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
from dew.nn.kernels.templates import attention_algebra as algebra, attention_mosaic, resources
from dew.nn.kernels.templates.attention import Hooks, Initial, Kept, KeySpans, prescale
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
    """`query` `[B, T, H, D]` as `[B, T·G, KV, D]` over its `kv_heads` key heads, `G = H // KV`, row `t·G + g`
    query head `k·G + g` of position `t`, and `spans` `[B, T]` and `initial` `[B, H, T]` alike: each packed
    row sees its query's keys and starts from its head's state."""
    batch, positions, heads, width = query.shape
    group = heads // kv_heads
    rows = query.reshape(batch, positions, kv_heads, group, width).transpose(0, 1, 3, 2, 4)
    rows = rows.reshape(batch, positions * group, kv_heads, width)

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


def unpacked(out: jax.Array, heads: int) -> jax.Array:
    """`packed`'s inverse on an output `[B, T·G, KV, Dv]`: `[B, T, H, Dv]`."""
    batch, rows, kv_heads, width = out.shape
    group = heads // kv_heads
    shaped = out.reshape(batch, rows // group, group, kv_heads, width).transpose(0, 1, 3, 2, 4)
    return shaped.reshape(batch, rows // group, heads, width)


@dataclasses.dataclass(frozen=True)
class Plan:
    """A decode step's work list as the derivation shapes it: the forward's config (`attention_mosaic.Tiles`)
    and the slices each query tile's keys are cut into (`MosaicForward.splits`)."""

    config: attention_mosaic.Tiles
    slices: int


def plan(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None, *,
         scale: float | None, target: generation.GpuTarget, table: jax.Array | None = None,
         sources: jax.Array | None = None) -> Plan:
    """The work list of a packed call (`packed`) on `target`: the forward's derived config at the key tile
    its memory-bound walk streams at least (`MosaicForward.streaming`) and its slices.
    One where the programs already fill the multiprocessors (a slice's merge would cost and its keys gain
    nothing). Else as many as its programs' residency holds at once (`resources.resident_programs`, its
    registers; its shared memory's), but each slice at least twice its ring's stages of key tiles: a shorter
    slice's walk spends its prologue's loads, `stages` tiles, without as many steps of products to hide them
    (ET-DECODE2-h100: 32 slices of one key tile each, at batch 1 over 4096 keys, took 22 us where cuDNN's
    whole step took 13)."""
    op, args = attention_mosaic._forward_call(query, key, value, spans, hooks, scale=scale, natural=False,
                                              out_dtype=jnp.float32, masked=True, table=table,
                                              sources=sources)
    ba = op.bind(*args, residuals=Kept.OUTPUT, interpret=False)
    config = op.derived(ba, target)
    if config is None:
        raise NotImplementedError(f"the template's forward derives no decode step's config on {target}")
    # A decode step's few rows a key head stream its keys (`MosaicForward.streaming`).
    config = op.streaming(ba, config, target)
    processors = target.multiprocessors or 1
    tiles = math.ceil(query.shape[1] / (config.consumers * config.block_q))
    programs = query.shape[0] * query.shape[2] * tiles
    if programs >= processors:
        return Plan(config, 1)
    family = bundle_of(config, query.dtype)
    resident = resources.resident_programs(family, config.consumers, op.registers(ba, config), target)
    if target.shared_limit:
        resident = min(resident, max(1, target.shared_limit // op.staged_bytes(ba, config, target)))
    longest = math.ceil(spans.keys / config.block_k) // (2 * config.stages)
    # The most slices whose programs the device holds at once, none past it: one program past them waits a
    # whole walk for a slot (ceil(132 / 8) = 17 slices of one row's 8 key heads is 136 programs, 4 of them a
    # second wave as long as the first).
    return Plan(config, max(1, min(processors * resident // programs, longest)))


def merged(spec: algebra.Forward, out: jax.Array, maximum: jax.Array, total: jax.Array,
           initial: Initial | None = None) -> jax.Array:
    """The slices' outputs `out` `[B, S, Q, KV, Dv]` (fp32, each normalized), maxima (in natural `spec`'s
    units, `algebra.finished_rows`', a fold's `algebra.unfolded`) and denominators `[B, KV, S, Q]`, and the
    call's `initial` state `[B, KV, Q]` (its sums `acc` unnormalized) merged by the algebra's fold over
    states (`algebra.merged`, then `finish`): the rows' output `[B, Q, KV, Dv]` fp32. Each state enters as
    its normalized output and its mass, its denominator folded into its maximum (`m + log(total)`), so no
    output multiplied back by its denominator passes fp32's largest; a state of no mass adds nothing, and a
    row of none is zero, as the reference's."""
    if not spec.natural:
        raise ValueError("the template's decode merges natural states")
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
    """The natural form of the call as the template's forward takes it (`attention_mosaic_forward._prepared`):
    the units its finished rows' maxima are in (`algebra.finished_rows`: natural from the fold and from the
    fast form a score hook keeps, its rewalk's, `algebra.unfolded`), which its merge takes."""
    power, factor = prescale(query.dtype, query.shape[3], scale)
    spec = algebra.Forward(spans.form, spans.keys, ids=spans.query_ids is not None, scale=factor,
                           score_mod=hooks.score_mod, natural=False, initial=hooks.initial is not None,
                           prescale=power)
    return spec if spec.natural else algebra.unfolded(spec)


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
    if work.slices == 1:
        # One slice a row: its walk merges the initial state itself (`algebra.merge`) and finishes its rows in
        # the query's dtype, no merge past it (its fp32 rows and their merge took 45 of B128's 820 us at
        # 4096 keys, ET-DECODE6-h100).
        whole = dataclasses.replace(hooks, initial=initial)
        work = plan(rows, key, value, row_spans, whole, scale=scale, target=target, table=table,
                    sources=sources)
        with tuning.pinned({attention_mosaic.MosaicForward: work.config}):
            out = attention_mosaic.forward(rows, key, value, row_spans, whole, scale=scale, natural=False,
                                           residuals=Kept.NONE, interpret=interpret, masked=True, table=table,
                                           sources=sources)
        return unpacked(out, heads)
    with tuning.pinned({attention_mosaic.MosaicForward: work.config}):
        out, maximum, total = attention_mosaic.forward(
            rows, key, value, row_spans, walked, scale=scale, natural=False, residuals=Kept.OUTPUT,
            interpret=interpret, out_dtype=jnp.float32, masked=True, splits=work.slices, table=table,
            sources=sources)[:3]
    merged_rows = merged(_spec(rows, row_spans, walked, scale=scale), out, maximum, total, initial)
    return unpacked(merged_rows, heads).astype(query.dtype)
