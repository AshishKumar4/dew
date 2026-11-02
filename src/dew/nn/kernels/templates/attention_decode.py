"""The attention template's decode: a few query positions a row over that row's keys, through the template's
own forward over a work list the step builds (kernel-templates.md §5 and §6; theme A06).

A decode step reads, for every row, many keys for a few queries. The template's forward walks a query
tile's keys with the algebra's online softmax; two builders shape the step for it, and a merge ends it:

- `packed`: a key head's group of query heads become its rows, one (query, head-in-group) pair a row, so
  a step of one query position fills a tile with the group and reads each key tile once for all of it.
  The same builder takes a prefill call: a call's rows are its rows whatever their count (§5: GQA packing
  is structure, one builder for prefill and decode).
- `sliced`: each row's keys cut into contiguous slices, each slice the rows' copy over its own keys and
  its own work item, as many items as the GPU holds at once (`slices`: the derived config's resident
  programs a multiprocessor, from its registers and its shared memory, times the multiprocessors; no
  fitted table), so one row's long cache fills the GPU.
- `merged`: the slices' softmax states, each its output normalized in fp32, its maximum and its
  denominator, combined in slice order (the algebra's merge in natural units: each slice's output weighed
  by its mass at the largest maximum), one sum a row, deterministic.

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
    """A decode step's work list as the derivation shapes it: the forward's config (`attention_mosaic.Tiles`),
    the rows a work item holds (a program's, `consumers · block_q`, so no program's consumers walk two
    items' keys) and the slices each row's keys are cut into."""

    config: attention_mosaic.Tiles
    rows: int
    slices: int


def plan(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None, *,
         scale: float | None, natural: bool, target: generation.GpuTarget) -> Plan:
    """The work list of a packed call (`packed`) on `target`: the forward's derived config for one slice's
    rows, and slices opened until the items fill the multiprocessors at the config's residency
    (`resources.resident_programs`, its registers; its shared memory's), each at least a key tile."""
    op, args = attention_mosaic._forward_call(query, key, value, spans, hooks, scale=scale, natural=natural,
                                              out_dtype=jnp.float32, masked=True)
    ba = op.bind(*args, residuals=Kept.OUTPUT, interpret=False)
    config = op.derived(ba, target)
    if config is None:
        raise NotImplementedError(f"the template's forward derives no config for the decode step on {target}")
    family = bundle_of(config, query.dtype)
    resident = resources.resident_programs(family, config.consumers, op.registers(ba, config), target)
    if target.shared_limit:
        resident = min(resident, max(1, target.shared_limit // op.staged_bytes(ba, config, target)))
    rows = config.consumers * config.block_q
    items = query.shape[0] * query.shape[2] * math.ceil(query.shape[1] / rows)
    wanted = math.ceil((target.multiprocessors or 1) * resident / items)
    return Plan(config, rows, max(1, min(wanted, math.ceil(key.shape[1] / config.block_k))))


def sliced(query: jax.Array, spans: KeySpans, initial: Initial | None, plan: Plan
           ) -> tuple[jax.Array, KeySpans, Initial | None]:
    """A packed call's work list (`Plan`): its rows padded to whole items and copied once a slice, each copy
    seeing its slice of its row's keys, contiguous and as long as the others but for rounding, its first
    a key tile's boundary (so no two slices read one tile); the initial state the first slice's alone (a
    sink counted once), the others' empty."""
    rows = query.shape[1]
    padded = math.ceil(rows / plan.rows) * plan.rows
    query = jnp.pad(query, ((0, 0), (0, padded - rows), (0, 0), (0, 0)))
    query = jnp.tile(query, (1, plan.slices, 1, 1))
    tile = plan.config.block_k
    start, end = spans.start, jnp.maximum(spans.end, spans.start)
    length = end - start

    def bound(s):
        if s == 0:
            return start
        if s == plan.slices:
            return end
        cut = start + (length * s) // plan.slices
        return jnp.clip((cut // tile) * tile, start, end)

    def rows_of(x, fill):
        return jnp.pad(x, ((0, 0), (0, padded - rows)), constant_values=fill)

    starts = jnp.concatenate([rows_of(bound(s), 0) for s in range(plan.slices)], axis=1)
    ends = jnp.concatenate([rows_of(bound(s + 1), 0) for s in range(plan.slices)], axis=1)
    ids = None if spans.query_ids is None else jnp.tile(rows_of(spans.query_ids, 0), (1, plan.slices))
    spans = KeySpans(starts, ends, ids, spans.key_ids, spans.keys, "spans")
    if initial is not None:
        def first(x, fill):
            widths = ((0, 0), (0, 0), (0, padded - rows)) + ((0, 0),) * (x.ndim - 3)
            own = jnp.pad(x, widths, constant_values=fill)
            rest = jnp.full((*own.shape[:2], own.shape[2] * (plan.slices - 1), *own.shape[3:]), fill,
                            own.dtype)
            return jnp.concatenate([own, rest], axis=2)

        initial = Initial(first(initial.m, -jnp.inf), first(initial.total, 0.0), first(initial.acc, 0.0))
    return query, spans, initial


def merged(out: jax.Array, maximum: jax.Array, total: jax.Array, slices: int, rows: int) -> jax.Array:
    """The slices' outputs `out` `[B, S·Rp, KV, Dv]` (fp32, each normalized), maxima and denominators
    `[B, KV, S·Rp]` (natural) merged into the rows' output `[B, R, KV, Dv]` fp32: each slice's output
    weighed by its mass at the rows' largest maximum, summed in slice order, over the summed mass; a slice
    of no key (no mass) adds nothing, and a row no slice of which saw a key is zero, as the reference's."""
    batch, items, kv_heads, width = out.shape
    padded = items // slices
    out = out.reshape(batch, slices, padded, kv_heads, width)[:, :, :rows]
    maximum = maximum.reshape(batch, kv_heads, slices, padded)[..., :rows]
    total = total.reshape(batch, kv_heads, slices, padded)[..., :rows]
    seen = total > 0
    maximum = jnp.where(seen, maximum, -jnp.inf)
    top = jnp.max(maximum, axis=2, keepdims=True)
    weight = jnp.where(seen, total * jnp.exp(maximum - jnp.where(jnp.isfinite(top), top, 0.0)), 0.0)
    weight = jnp.transpose(weight, (0, 2, 3, 1))[..., None]
    summed, mass = jnp.sum(weight * out, axis=1), jnp.sum(weight, axis=1)
    return jnp.where(mass > 0, summed / jnp.where(mass > 0, mass, 1.0), 0.0)


def _natural_units(query: jax.Array, spans: KeySpans, hooks: Hooks, *, scale: float | None,
                   natural: bool) -> float:
    """The factor taking the forward's residual maxima to natural logits (`algebra.finished_rows`): the
    accurate form's raw products' scale (`Forward.raw`), else 1."""
    power, factor = prescale(query.dtype, query.shape[3], scale)
    spec = algebra.Forward(spans.form, spans.keys, ids=spans.query_ids is not None, scale=factor,
                           score_mod=hooks.score_mod, natural=natural, initial=hooks.initial is not None,
                           prescale=power)
    return spec.scale if spec.raw else 1.0


def attend(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None, *,
           scale: float | None = None, interpret: bool = False) -> jax.Array:
    """A decode step's attention through the template: `query` `[B, T, H, D]` over `key` `[B, K, KV, D]` and
    `value` `[B, K, KV, Dv]`, the keys `spans` `[B, T]` leaves each query under `hooks` (`packable`), as
    `attention.reference` computes it, `[B, T, H, Dv]` in the query's dtype: packed (`packed`), cut into
    the derived work list (`plan`, `sliced`), walked by the template's forward in fp32 with its rows
    (`attention_mosaic.forward`), merged (`merged`) and unpacked. Its logits natural, so the guard runs as a
    launch over its live rows with no branch read on the host (`MosaicForward.masked`, row 1978)."""
    natural = True
    hooks = Hooks() if hooks is None else hooks
    if not packable(hooks):
        raise NotImplementedError("the template's decode packs no query-placed hook, parameter, output "
                                  "hook or dropout (`packable`)")
    heads, kv_heads = query.shape[2], key.shape[2]
    rows, row_spans, initial = packed(query, spans, kv_heads, hooks.initial)
    row_hooks = dataclasses.replace(hooks, initial=initial)
    work = plan(rows, key, value, row_spans, row_hooks, scale=scale, natural=natural,
                target=generation.gpu_target())
    items, item_spans, item_initial = sliced(rows, row_spans, initial, work)
    with tuning.pinned({attention_mosaic.MosaicForward: work.config}):
        out, maximum, total, *_ = attention_mosaic.forward(
            items, key, value, item_spans, dataclasses.replace(hooks, initial=item_initial), scale=scale,
            natural=natural, residuals=Kept.OUTPUT, interpret=interpret, out_dtype=jnp.float32, masked=True)
    units = _natural_units(rows, row_spans, row_hooks, scale=scale, natural=natural)
    out = merged(out, maximum * units, total, work.slices, rows.shape[1])
    return unpacked(out, heads).astype(query.dtype)

