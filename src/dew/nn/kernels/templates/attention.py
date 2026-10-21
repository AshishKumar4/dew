"""The attention template's shared core: which keys each query reads (`KeySpans`), the tiles a launch
visits in either direction (`BlockMap`), the hooks it traces (`Hooks`), and its arithmetic in XLA ops
(`reference`), the truth and portable arm every implementation is held to.

Visibility is one input for every call form Dew has (`spans`): each query's interval of keys
`[start, end)`, and where the call packs documents, the queries' and keys' document ids, a key visible
to a query exactly when it lies in the query's interval and, with ids, carries the query's id
(`visible`). Causality, windows on one or both sides, chunks, documents of the call's own or of another
sequence (aligned to their last keys, `dew.nn.documents.CrossDocuments`), key lengths and key starts
all narrow the interval. Every implementation reads the same predicate, so they mean the same keys.

A hook is a plain JAX callable the kernel's body traces, pointwise in the score tile:
`score_mod(s, q_pos, k_pos, *params)` on the scaled logits (softcap, ALiBi, a bias, a forget-gate
decay), with each parameter loaded as its kind says (`Param`); the softmax starts from an initial state
(`Initial`: sinks, a resumed partial); `output_mod(o, *params)` on the normalized output tile. Absent,
a hook adds no op.
"""

import dataclasses
import enum
from collections.abc import Callable

import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class KeySpans:
    """Each query's visible keys: the interval `[start, end)` `[B, Q]` of key indices, int32, empty
    where `end <= start`, and where the call packs documents, the queries' ids `[B, Q]` and the keys'
    `[B, K]`, 0 as padding, a key visible only to queries of its own id (`visible`). `keys` is the
    key count, static."""

    start: jax.Array
    end: jax.Array
    query_ids: jax.Array | None
    key_ids: jax.Array | None
    keys: int = dataclasses.field(metadata={"static": True})


def visible(spans: KeySpans, row: jax.Array | int, queries: jax.Array, keys: jax.Array) -> jax.Array:
    """Whether each of `keys` (key indices, broadcasting) is visible to each of `queries` (query
    indices) in batch row `row`: inside the query's interval and, with ids, of its document."""
    start, end = spans.start[row][queries], spans.end[row][queries]
    seen = (keys >= start) & (keys < end)
    if spans.query_ids is not None and spans.key_ids is not None:
        seen &= spans.query_ids[row][queries] == spans.key_ids[row][keys]
    return seen


def dense(spans: KeySpans) -> jax.Array:
    """`visible` over every query and key: `[B, Q, K]` boolean."""
    queries = jnp.arange(spans.start.shape[1])[:, None]
    keys = jnp.arange(spans.keys)[None, :]
    return jax.vmap(lambda row: visible(spans, row, queries, keys))(jnp.arange(spans.start.shape[0]))


def _document_extent(query_ids: jax.Array, key_ids: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The first key index and one past the last carrying each query's id, `[B, Q]` each, the keys' ids
    in any order; an id no key carries gives an empty extent at 0."""
    def row(queries, keys):
        order = jnp.argsort(keys, stable=True)
        ordered = keys[order]
        first = jnp.searchsorted(ordered, queries, side="left")
        last = jnp.searchsorted(ordered, queries, side="right")
        present = last > first
        # A stable sort keeps equal ids in index order, so a document's first and last positions in
        # the sorted keys are its smallest and largest key index; an index past the end clamps.
        lowest = order[jnp.minimum(first, keys.shape[0] - 1)]
        highest = order[jnp.maximum(last - 1, 0)] + 1
        return jnp.where(present, lowest, 0), jnp.where(present, highest, 0)

    return jax.vmap(row)(query_ids, key_ids)


def spans(queries: int, keys: int, batch: int, *, causal: bool = False, sliding_window: int | None = None,
          chunk: int | None = None, segment_ids=None, lengths: jax.Array | None = None,
          starts: jax.Array | None = None) -> KeySpans:
    """The `KeySpans` of a call of `queries` queries over `keys` keys in `batch` rows, in the terms
    `dew.nn.attention.scaled_dot_product_attention` takes them: `causal` keeps query i to keys 0..i (top
    left aligned, as jax's `is_causal`); `sliding_window=w` keeps the w - 1 keys before a query and it, or
    without causality those within w - 1 of it (`window_sides`); `chunk=c` keeps a query to the keys of
    its own chunk of c (`chunk_mask`, Llama 4's, under causality those before it); `segment_ids`, one `[B,
    S]` array or a `CrossDocuments`, keeps each query to its document's keys, under causality or a window
    a cross call's queries aligned to their document's last keys (`aligned_places`); `lengths` `[B]` end
    each row's keys and `starts` `[B]` begin them."""
    from dew.nn.documents import CrossDocuments, aligned_places, document_ids

    place = jnp.broadcast_to(jnp.arange(queries, dtype=jnp.int32)[None, :], (batch, queries))
    start = jnp.zeros((batch, queries), jnp.int32)
    end = jnp.full((batch, queries), keys, jnp.int32)
    query_ids = key_ids = None
    if segment_ids is not None:
        query_ids, key_ids = (ids.astype(jnp.int32) for ids in document_ids(segment_ids))
        first, last = _document_extent(query_ids, key_ids)
        padding = query_ids == 0
        start = jnp.where(padding, 0, jnp.maximum(start, first))
        end = jnp.where(padding, 0, jnp.minimum(end, last))
        if isinstance(segment_ids, CrossDocuments) and (causal or sliding_window is not None):
            # A cross call's query sits at its rank among its document's keys, counted from the
            # document's first key (`cross_document_mask`).
            place = first + aligned_places(segment_ids).astype(jnp.int32)
    if causal:
        end = jnp.minimum(end, place + 1)
    if sliding_window is not None:
        start = jnp.maximum(start, place - sliding_window + 1)
        if not causal:
            end = jnp.minimum(end, place + sliding_window)
    if chunk is not None:
        start = jnp.maximum(start, (place // chunk) * chunk)
        end = jnp.minimum(end, (place // chunk + 1) * chunk)
    if lengths is not None:
        end = jnp.minimum(end, lengths.astype(jnp.int32)[:, None])
    if starts is not None:
        start = jnp.maximum(start, starts.astype(jnp.int32)[:, None])
    return KeySpans(start, end, query_ids, key_ids, keys)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BlockMap:
    """The tiles a launch over `block_q` queries by `block_k` keys visits, both directions, from one
    `KeySpans`: query block i reads key blocks `[lo[b, i], hi[b, i])`, of which `[full_lo, full_hi)` are
    visible in every entry for its every query (the kernel skips the predicate there, where the ids
    agree: `query_doc[b, i] == key_doc[b, j] > 0`, or the call has none); key block j is read by query
    blocks `[q_lo[b, j], q_hi[b, j])`, each of which still checks `lo <= j < hi`. Every visible entry lies
    in a visited tile; a visited tile may hold none (an envelope over a block's spans)."""

    lo: jax.Array
    hi: jax.Array
    full_lo: jax.Array
    full_hi: jax.Array
    q_lo: jax.Array
    q_hi: jax.Array
    query_doc: jax.Array | None
    key_doc: jax.Array | None
    block_q: int = dataclasses.field(metadata={"static": True})
    block_k: int = dataclasses.field(metadata={"static": True})


def _blocked(values: jax.Array, block: int, fill) -> jax.Array:
    """`values` `[B, N]` padded with `fill` to whole blocks, `[B, N // block, block]`."""
    pad = -values.shape[1] % block
    return jnp.pad(values, ((0, 0), (0, pad)), constant_values=fill).reshape(values.shape[0], -1, block)


def _uniform_doc(ids: jax.Array | None, block: int) -> jax.Array | None:
    """Each block's document id where all its entries carry one (and it is not padding), else -1."""
    if ids is None:
        return None
    blocks = _blocked(ids, block, -1)
    first = blocks[..., :1]
    return jnp.where(jnp.all(blocks == first, axis=-1) & (first[..., 0] > 0), first[..., 0], -1)


def block_map(spans: KeySpans, block_q: int, block_k: int) -> BlockMap:
    """The `BlockMap` of `spans` at `block_q` by `block_k` tiles."""
    keys = spans.keys
    key_blocks = -(-keys // block_k)
    start, end = _blocked(spans.start, block_q, 0), _blocked(spans.end, block_q, 0)
    nonempty = end > start
    big = jnp.iinfo(jnp.int32).max
    first = jnp.min(jnp.where(nonempty, start, big), axis=-1)
    last = jnp.max(jnp.where(nonempty, end, 0), axis=-1)
    any_rows = jnp.any(nonempty, axis=-1)
    lo = jnp.where(any_rows, first // block_k, 0)
    hi = jnp.where(any_rows, -(-last // block_k), 0)
    # A tile is whole where every query of the block sees all its keys: the block's latest start and
    # earliest end bound it, and a query that sees nothing leaves no tile whole.
    all_rows = jnp.all(nonempty, axis=-1)
    whole_lo = -(-jnp.max(start, axis=-1) // block_k)
    whole_hi = jnp.minimum(jnp.min(end, axis=-1), keys) // block_k
    full_lo = jnp.where(all_rows, jnp.clip(whole_lo, lo, hi), lo)
    full_hi = jnp.where(all_rows & (whole_hi > whole_lo), jnp.clip(whole_hi, full_lo, hi), full_lo)
    # The query blocks reading each key block: those whose range holds it.
    j = jnp.arange(key_blocks)[None, None, :]
    reads = (lo[:, :, None] <= j) & (j < hi[:, :, None])
    i = jnp.arange(lo.shape[1])[None, :, None]
    q_lo = jnp.min(jnp.where(reads, i, big), axis=1)
    q_hi = jnp.max(jnp.where(reads, i + 1, 0), axis=1)
    q_lo = jnp.where(q_hi > 0, q_lo, 0)
    return BlockMap(lo, hi, full_lo, full_hi, q_lo, q_hi, _uniform_doc(spans.query_ids, block_q),
                    _uniform_doc(spans.key_ids, block_k), block_q, block_k)


class Kind(enum.Enum):
    """How a hook parameter is laid out and loaded per tile."""

    HEAD = "head"
    """`[H]`: one value a query head, the head's scalar."""
    QUERY = "query"
    """`[B, H, Q]`: one value a query, the tile's queries' `[block_q]`."""
    KEY = "key"
    """`[B, H, K]`: one value a key, the tile's keys' `[block_k]`."""
    TILE = "tile"
    """`[B, H, Q, K]`: one value an entry, the tile's `[block_q, block_k]` (a dense bias)."""
    OUTPUT = "output"
    """`[B, Q, H, Dv]`: one value an output entry, the output tile's `[block_q, Dv]` (a gate)."""


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Param:
    """A hook's parameter: its array and how a tile reads it (`Kind`)."""

    value: jax.Array
    kind: Kind = dataclasses.field(metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Initial:
    """Where each query's online softmax starts: its running maximum `m` `[B, H, Q]` (natural log), its
    unnormalized denominator `total` `[B, H, Q]` and numerator `acc` `[B, H, Q, Dv]`, fp32. Sinks are `m`
    the head's sink, `total` one and `acc` zero (`sinks`); a resumed partial is its own state."""

    m: jax.Array
    total: jax.Array
    acc: jax.Array


def sinks(values: jax.Array, batch: int, queries: int, width: int) -> Initial:
    """The `Initial` of one value-free logit a query head, `values` `[H]`."""
    m = jnp.broadcast_to(values.astype(jnp.float32)[None, :, None], (batch, values.shape[0], queries))
    return Initial(m, jnp.ones_like(m), jnp.zeros((*m.shape, width), jnp.float32))


@dataclasses.dataclass(frozen=True)
class Hooks:
    """The template's hooks, each absent where None: `score_mod(s, q_pos, k_pos, *params)` on the scaled
    logits `s` `[..., q, k]` with `q_pos` `[q, 1]` and `k_pos` `[1, k]` the entries' query and key
    indices, pointwise in the entries; `initial` the softmax's start; `output_mod(o, *params)` on the
    normalized output `[..., q, Dv]`."""

    score_mod: Callable | None = None
    score_params: tuple[Param, ...] = ()
    initial: Initial | None = None
    output_mod: Callable | None = None
    output_params: tuple[Param, ...] = ()


def _per_head(param: Param, row: jax.Array, head: jax.Array) -> jax.Array:
    """`param` as the reference reads it for one batch row and query head, over every query and key."""
    value = param.value
    match param.kind:
        case Kind.HEAD:
            return value[head]
        case Kind.QUERY:
            return value[row, head][:, None]
        case Kind.KEY:
            return value[row, head][None, :]
        case Kind.TILE:
            return value[row, head]
        case Kind.OUTPUT:
            return value[row, :, head]


def reference(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
              *, scale: float | None = None, residuals: bool = False):
    """The template's arithmetic in XLA ops: `query` `[B, Q, H, D]` over `key` and `value` `[B, K, Hk, D]`
    and `[B, K, Hk, Dv]`, a query head reading the key head it shares with `H // Hk` others, the keys
    `spans` leaves it. As the kernels compute it: products of the inputs accumulated in fp32 (fp64 for
    fp64 inputs), the scale and `score_mod` on the fp32 logits, the softmax from `hooks.initial` in fp32,
    the probabilities rounded to the values' dtype for their product, the output normalized in fp32 and
    `output_mod` applied before it rounds to the query's dtype. A query that sees no key and starts from
    no state gives zeros. With `residuals`, also each query's running maximum `m` and denominator `total`
    `[B, H, Q]` (natural log, unnormalized), fp32, a query that sees no key and starts from no state
    holding 0 for both, so every residual is finite. No hooks where `hooks` is None."""
    hooks = Hooks() if hooks is None else hooks
    batch, queries, heads, width = query.shape
    kv_heads = key.shape[2]
    wide = jnp.promote_types(query.dtype, jnp.float32)
    scale = width ** -0.5 if scale is None else scale
    seen = dense(spans)

    def one(row, head):
        shared = head // (heads // kv_heads)
        q, k, v = query[row, :, head], key[row, :, shared], value[row, :, shared]
        s = jnp.einsum("qd,kd->qk", q, k, preferred_element_type=wide) * jnp.asarray(scale, wide)
        if hooks.score_mod is not None:
            q_pos = jnp.arange(queries)[:, None]
            k_pos = jnp.arange(spans.keys)[None, :]
            s = hooks.score_mod(s, q_pos, k_pos, *(_per_head(p, row, head) for p in hooks.score_params))
        s = jnp.where(seen[row], s, -jnp.inf)
        if hooks.initial is None:
            m0 = jnp.full((queries,), -jnp.inf, wide)
            total0 = jnp.zeros((queries,), wide)
            acc0 = jnp.zeros((queries, value.shape[-1]), wide)
        else:
            m0 = hooks.initial.m[row, head].astype(wide)
            total0 = hooks.initial.total[row, head].astype(wide)
            acc0 = hooks.initial.acc[row, head].astype(wide)
        m = jnp.maximum(m0, jnp.max(s, axis=-1))
        safe = jnp.where(jnp.isfinite(m), m, 0)
        p = jnp.exp(s - safe[:, None])
        carried = jnp.where(jnp.isfinite(m0), jnp.exp(m0 - safe), 0)
        total = total0 * carried + jnp.sum(p, axis=-1)
        acc = acc0 * carried[:, None] + jnp.einsum("qk,kd->qd", p.astype(value.dtype), v,
                                                    preferred_element_type=wide)
        o = jnp.where(total[:, None] > 0, acc / jnp.where(total > 0, total, 1)[:, None], 0)
        if hooks.output_mod is not None:
            o = hooks.output_mod(o, *(_per_head(p, row, head) for p in hooks.output_params))
        return o.astype(query.dtype), safe, total

    rows = jnp.arange(batch)
    out, m, total = jax.vmap(jax.vmap(one, in_axes=(None, 0)), in_axes=(0, None))(rows, jnp.arange(heads))
    out = jnp.swapaxes(out, 1, 2)
    return (out, m.astype(jnp.float32), total.astype(jnp.float32)) if residuals else out
