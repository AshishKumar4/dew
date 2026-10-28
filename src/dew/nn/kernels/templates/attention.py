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

Precision, every implementation's contract: the products of the inputs accumulate in fp32, the logits'
scale, hooks and softmax run in fp32, and the probabilities round to the values' dtype for each product
they enter, the output's `P V` and the backward's `P^T dO`, as jax's own attention does
(`jax.nn.dot_product_attention`'s xla path, jax/_src/nn/functions.py: `probs =
softmax(...).astype(key.dtype)`) and FlashAttention 2's forward (csrc/flash_attn/src/flash_fwd_kernel.h:
"Convert acc_s from fp32 to fp16/bf16" before the P V product); the output normalizes in fp32 before it
rounds to the query's dtype (`reference`).
"""

import dataclasses
import enum
import functools
import math
import operator
from collections.abc import Callable
from typing import ClassVar, NamedTuple

import jax
import jax.numpy as jnp

from dew._hashing import fmix32
from dew.nn.kernels import tuning
from dew.nn.precision import at_least_fp32, float32_einsum


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class KeySpans:
    """Each query's visible keys: the interval `[start, end)` `[B, Q]` of key indices, int32, empty
    where `end <= start`, or R of them `[B, Q, R]`, disjoint and sorted, whose union a query sees (a
    call's `runs`, `spans`), and where the call packs documents, the queries' ids `[B, Q]` and the keys'
    `[B, K]`, 0 as padding, a key visible only to queries of its own id (`visible`). `keys` is the
    key count, static, and `form` what the spans are, static, so a kernel resolves at trace time what
    a plain call needs: 'full', every query sees every key; 'causal', query i sees keys 0 to i (top
    left aligned); 'spans', anything else, which the intervals and ids say."""

    start: jax.Array
    end: jax.Array
    query_ids: jax.Array | None
    key_ids: jax.Array | None
    keys: int = dataclasses.field(metadata={"static": True})
    form: str = dataclasses.field(default="spans", metadata={"static": True})


def within(keys: jax.Array, start: jax.Array, end: jax.Array,
           ids: tuple[jax.Array, jax.Array] | None = None) -> jax.Array:
    """Whether each key index of `keys` lies in its query's interval `start` to `end` and, with `ids`
    (the queries' and the keys', broadcasting alike), is of its document: the keys a call's spans let a
    query see, which the reference (`visible`) and every mainloop (`attention_algebra.modify`, the
    backward's tiles) mask by. Runs, an interval axis more on `start` and `end` than on `keys`, see their
    union, taken run by run (their count is static), which Pallas Triton lowers where a reduction over the
    axis (`reduce_or`) it does not."""
    if start.ndim > keys.ndim:
        held = [(keys >= start[..., run]) & (keys < end[..., run]) for run in range(start.shape[-1])]
        seen = functools.reduce(operator.or_, held)
    else:
        seen = (keys >= start) & (keys < end)
    return seen if ids is None else seen & (ids[0] == ids[1])


def visible(spans: KeySpans, row: jax.Array | int, queries: jax.Array, keys: jax.Array) -> jax.Array:
    """Whether each of `keys` (key indices, broadcasting) is visible to each of `queries` (query
    indices) in batch row `row` (`within`)."""
    ids = (None if spans.query_ids is None or spans.key_ids is None
           else (spans.query_ids[row][queries], spans.key_ids[row][keys]))
    return within(keys, spans.start[row][queries], spans.end[row][queries], ids)


def dense(spans: KeySpans) -> jax.Array:
    """`visible` over every query and key: `[B, Q, K]` boolean."""
    queries = jnp.arange(spans.start.shape[1])[:, None]
    keys = jnp.arange(spans.keys)[None, :]
    return jax.vmap(lambda row: visible(spans, row, queries, keys))(jnp.arange(spans.start.shape[0]))


def reads(spans: KeySpans) -> jax.Array:
    """Whether each query sees a key, `[B, Q]`: one of its intervals holds one, of its document where the
    call has ids, counted among that document's keys in index order however the documents lie (`_runs`)."""
    start, end = spans.start, spans.end
    runs = start.ndim == 3
    if spans.query_ids is None or spans.key_ids is None:
        held = end > start
        return jnp.any(held, axis=-1) if runs else held
    batch, queries = start.shape[:2]
    order, lo, hi = _runs(spans.query_ids, spans.key_ids, batch, queries, spans.keys)
    if runs:
        count = start.shape[2]
        lo, hi = (jnp.repeat(bound, count, axis=1) for bound in (lo, hi))
        start, end = start.reshape(batch, -1), end.reshape(batch, -1)
    held = _searched(order, lo, hi, end, "left") > _searched(order, lo, hi, start, "left")
    return jnp.any(held.reshape(batch, queries, -1), axis=-1) if runs else held


def _runs(query_ids: jax.Array | None, key_ids: jax.Array | None, batch: int, queries: int,
          keys: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Each row's key indices in document order `[B, K]`, and where each query's document runs in it,
    `[lo, hi)` `[B, Q]` int32: a stable sort of the keys' ids (in any order), so a document's keys keep
    their index order; without ids, every key in index order, one run."""
    if key_ids is None:
        order = jnp.broadcast_to(jnp.arange(keys, dtype=jnp.int32)[None, :], (batch, keys))
        return order, jnp.zeros((batch, queries), jnp.int32), jnp.full((batch, queries), keys, jnp.int32)

    def row(wanted, held):
        order = jnp.argsort(held, stable=True)
        ordered = held[order]
        return (order.astype(jnp.int32), jnp.searchsorted(ordered, wanted, side="left").astype(jnp.int32),
                jnp.searchsorted(ordered, wanted, side="right").astype(jnp.int32))

    return jax.vmap(row)(query_ids, key_ids)


def _indexed(order: jax.Array, at: jax.Array, fill: int) -> jax.Array:
    """`order[b, at[b, q]]` `[B, Q]`, `fill` where `order` has no entries."""
    if order.shape[1] == 0:
        return jnp.full(at.shape, fill, jnp.int32)
    return jnp.take_along_axis(order, jnp.clip(at, 0, order.shape[1] - 1), axis=1)


def _searched(values: jax.Array, lo: jax.Array, hi: jax.Array, wanted: jax.Array, side: str) -> jax.Array:
    """`jnp.searchsorted` of each query's `wanted` `[B, Q]` within its own run `[lo, hi)` of `values`
    `[B, K]`, ascending there: the first position whose value is not below it (`side='left'`) or above
    it (`'right'`), `hi` where none."""
    def step(_, bounds):
        low, high = bounds
        middle = low + (high - low) // 2
        value = _indexed(values, middle, 0)
        right = (value < wanted) if side == "left" else (value <= wanted)
        live = low < high
        return jnp.where(live & right, middle + 1, low), jnp.where(live & ~right, middle, high)

    steps = max(values.shape[1], 1).bit_length()
    return jax.lax.fori_loop(0, steps, step, (lo, hi))[0]


def spans(queries: int, keys: int, batch: int, *, causal: bool = False, sliding_window: int | None = None,
          chunk: int | None = None, positions: tuple[jax.Array, jax.Array] | None = None, segment_ids=None,
          lengths: jax.Array | None = None, starts: jax.Array | None = None,
          block_groups: jax.Array | None = None, image_groups: jax.Array | None = None,
          valid: jax.Array | None = None, runs: jax.Array | None = None) -> KeySpans:
    """The `KeySpans` of a call of `queries` queries over `keys` keys in `batch` rows, in the terms
    `dew.nn.attention.scaled_dot_product_attention` takes them: `causal` keeps query i to keys 0..i (top
    left aligned, as jax's `is_causal`); `sliding_window=w` keeps the keys `window_sides` gives, the w - 1
    before a query and it, or without causality those within w - 1 of it; `chunk=c` keeps a query to the
    keys of its own chunk of c places (`chunk_mask`, Llama 4's, under causality those before it), the
    queries' `[B, Q]` and keys' `[B, K]` places `positions`, which never decrease along a document's
    keys, as Dew's coordinates count them, by default their indices, over which a window also counts where
    they are given (the keys within w - 1 places, padding inside a row skipped: a key's place there is the
    least at or after it in its document, so a padding slot that took the document before's place keeps
    the real keys' windows exact); `segment_ids`, one `[B, S]`
    array or a `CrossDocuments` (its keys in ascending id order, as it takes them), keeps each query to its
    document's keys, under causality or a window a cross call's queries aligned to their document's last
    keys (`aligned_places`); `lengths` `[B]` end each row's keys and `starts` `[B]` begin them.
    `block_groups` and `image_groups` `[B, K]` number the keys' blocks, runs of contiguous keys, -1 outside
    any: under causality a query whose own key (its index, or a cross call's aligned place: a cache call's
    slot) is in a block also sees the block's later keys, within its document. A block of `block_groups`
    is read whole whatever the window, transformers' `block_sequence_ids` (`(causal and window) or block`);
    an image of `image_groups` within the window, which bounds the keys behind the query (Gemma 3's and
    4's, `window and (causal or image)`). A block holds the query's key and is one run, so the query's keys
    stay one interval, the union of both kinds' among them. `valid` `[B, K]` marks the real keys, any
    others hidden from every query as the document ids hide them: an invalid key takes id 0, the padding
    id no query matches, after every place a cross call's alignment reads is taken from the ids as given,
    and a call of no documents takes 1 for its queries and its valid keys. `runs` `[B, Q, R, 2]` are each
    query's key intervals `[lo, hi)`, R of them, static, whose union it may see, intersected with all the
    rest: a window over `positions` is searched within each run alone, whose key places never decrease
    (its builder's contract: a layout whose places restart, as a decision row's options do, splits its
    runs there), so the spans keep R intervals a query `[B, Q, R]`. Lengths and starts are int32 token
    coordinates; places and ids keep their own dtypes."""
    from dew.nn.documents import CrossDocuments, aligned_places, document_ids
    from dew.nn.visibility import block_ends, block_starts, window_sides

    index = jnp.broadcast_to(jnp.arange(queries, dtype=jnp.int32)[None, :], (batch, queries))
    place = index
    start = jnp.zeros((batch, queries), jnp.int32)
    end = jnp.full((batch, queries), keys, jnp.int32)
    # Each query's document's first key and one past its last, which bound a block's keys; 0 for padding.
    floor, bound = start, end
    query_ids, key_ids = (None, None) if segment_ids is None else document_ids(segment_ids)
    order, run_lo, run_hi = _runs(query_ids, key_ids, batch, queries, keys)
    present = run_hi > run_lo
    if segment_ids is not None:
        # A document's first and last keys in document order are its smallest and largest key index.
        first = jnp.where(present, _indexed(order, run_lo, 0), 0)
        last = jnp.where(present, _indexed(order, run_hi - 1, -1) + 1, 0)
        padding = query_ids == 0
        start = jnp.where(padding, 0, jnp.maximum(start, first))
        end = jnp.where(padding, 0, jnp.minimum(end, last))
        floor, bound = start, end
        if isinstance(segment_ids, CrossDocuments) and (causal or sliding_window is not None) and keys > 0:
            # A cross call's query sits at its rank among its document's keys, counted from the
            # document's first key (`cross_document_mask`).
            place = first + aligned_places(segment_ids).astype(jnp.int32)
    if causal:
        end = jnp.minimum(end, place + 1)
    # The query's own key, which its block holds: a query placed before its document's keys has none.
    own = (place >= start) & (place < keys)
    if sliding_window is not None and positions is not None and runs is None:
        # The query's document's keys within the window's places run together in document order, since
        # their places never decrease along it, and their first and last key indices bound the span.
        before, after = window_sides(causal, sliding_window)
        query_places, key_places = _widened(positions)
        ordered = _least_after(jnp.take_along_axis(key_places, order, axis=1),
                               None if key_ids is None else jnp.take_along_axis(key_ids, order, axis=1))
        start, end = _narrowed(start, end, order, run_lo, run_hi, ordered, _moved(query_places, -before),
                               _moved(query_places, after))
    elif sliding_window is not None and positions is None:
        # No two places of the call lie further apart than its queries and keys, so a wider window
        # keeps what that one does, and its bounds stay within int32.
        before, after = window_sides(causal, min(sliding_window, queries + keys + 1))
        start = jnp.maximum(start, place - before)
        end = jnp.minimum(end, place + after + 1)
    for groups, whole in ((block_groups, True), (image_groups, False)):
        if groups is None or not causal or keys == 0:
            continue
        # The block of the query's own key runs on to its end, and a whole block from its start, within
        # the query's document.
        groups = jnp.asarray(groups)
        at = jnp.clip(place, 0, keys - 1)
        inside = own & (jnp.take_along_axis(groups, at, axis=1) >= 0)
        ends = jnp.take_along_axis(block_ends(groups), at, axis=1)
        end = jnp.where(inside, jnp.maximum(end, jnp.minimum(ends, bound)), end)
        if whole:
            begins = jnp.take_along_axis(block_starts(groups), at, axis=1)
            start = jnp.where(inside, jnp.minimum(start, jnp.maximum(begins, floor)), start)
    # Each narrowing by places a run takes alone: its keys' values in index order, never decreasing within a
    # run, and each query's least and greatest it keeps (`_within_runs`).
    by_places = []
    if chunk is not None:
        query_places, key_places = (
            (index, jnp.broadcast_to(jnp.arange(keys, dtype=jnp.int32)[None, :], (batch, keys)))
            if positions is None else positions)
        wanted = _chunk_of(query_places, chunk)
        by_places.append((_chunk_of(key_places, chunk), wanted, wanted))
        if runs is None:
            # The query's document's keys of its chunk run together in document order, since their places
            # never decrease along it, and their first and last key indices bound the span.
            numbers = jnp.take_along_axis(by_places[-1][0], order, axis=1)
            start, end = _narrowed(start, end, order, run_lo, run_hi, numbers, wanted, wanted)
    if lengths is not None:
        end = jnp.minimum(end, lengths.astype(jnp.int32)[:, None])
    if starts is not None:
        start = jnp.maximum(start, starts.astype(jnp.int32)[:, None])
    if runs is not None:
        if sliding_window is not None and positions is not None:
            before, after = window_sides(causal, sliding_window)
            query_places, key_places = _widened(positions)
            by_places.append((key_places, _moved(query_places, -before), _moved(query_places, after)))
        start, end = _within_runs(jnp.asarray(runs, jnp.int32), start, end, keys, by_places)
    blocks = True if causal and (block_groups is not None or image_groups is not None) else None
    narrowed = (sliding_window, chunk, segment_ids, lengths, starts, blocks, valid, runs)
    form = "spans" if any(narrowing is not None for narrowing in narrowed) else "causal" if causal else "full"
    if valid is not None:
        query_ids, key_ids = _valid_ids(query_ids, key_ids, jnp.asarray(valid, bool), batch, queries)
    return KeySpans(start, end, query_ids, key_ids, keys, form)


def _narrowed(start: jax.Array, end: jax.Array, order: jax.Array, run_lo: jax.Array, run_hi: jax.Array,
              values: jax.Array, low: jax.Array, high: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Each query's `[start, end)` narrowed to its document's keys whose `values` (`[B, K]` in document
    order, `order`, never decreasing along a document's run `[run_lo, run_hi)`) lie in `[low, high]`: they
    run together, and their first and last key indices bound the span, which is empty where none do. The
    first key at or above `low` bounds the start whether or not any lies within, so a key a block or image
    later adds past the end stays within the lower bound (past every key where none lies at or above it)."""
    inside_lo = _searched(values, run_lo, run_hi, low, "left")
    inside_hi = _searched(values, run_lo, run_hi, high, "right")
    held = inside_hi > inside_lo
    return (jnp.maximum(start, jnp.where(inside_lo < run_hi, _indexed(order, inside_lo, 0), values.shape[1])),
            jnp.where(held, jnp.minimum(end, _indexed(order, inside_hi - 1, -1) + 1), 0))


def _chunk_of(places, chunk: int) -> jax.Array:
    """Each place's chunk number in the places' own dtype; a chunk wider than it holds every place of one
    sign."""
    places = jnp.asarray(places)
    if chunk > jnp.iinfo(places.dtype).max:
        return jnp.where(places >= 0, 0, -1).astype(places.dtype)
    return places // chunk


def _within_runs(runs: jax.Array, start: jax.Array, end: jax.Array, keys: int,
                 by_places: list[tuple[jax.Array, jax.Array, jax.Array]]) -> tuple[jax.Array, jax.Array]:
    """Each query's `runs` `[B, Q, R, 2]` within its interval `[start, end)` `[B, Q]`, `[B, Q, R]` each, and
    each run's keys within each narrowing by places (`by_places`: its keys' values `[B, K]`, a window's
    places or a chunk's numbers, and each query's least and greatest it keeps `[B, Q]`), searched in the run
    alone, along which a run's places never decrease where the call's may restart between runs."""
    lo = jnp.maximum(jnp.clip(runs[..., 0], 0, keys), start[..., None])
    hi = jnp.maximum(jnp.minimum(jnp.clip(runs[..., 1], 0, keys), end[..., None]), lo)
    batch, _, count = lo.shape
    for values, least, greatest in by_places:
        flat_lo, flat_hi = lo.reshape(batch, -1), hi.reshape(batch, -1)
        first = _searched(values, flat_lo, flat_hi, jnp.repeat(least, count, axis=1), "left")
        last = _searched(values, flat_lo, flat_hi, jnp.repeat(greatest, count, axis=1), "right")
        lo, hi = first.reshape(lo.shape), jnp.maximum(last, first).reshape(hi.shape)
    return lo, hi


def _valid_ids(query_ids: jax.Array | None, key_ids: jax.Array | None, valid: jax.Array, batch: int,
               queries: int) -> tuple[jax.Array, jax.Array]:
    """The queries' and keys' ids that hide the keys `valid` `[B, K]` does not mark: a key's own id where it
    is valid, else 0, which no query matches; without documents every query and valid key 1."""
    if query_ids is None or key_ids is None:
        return jnp.ones((batch, queries), jnp.int32), valid.astype(jnp.int32)
    return query_ids, jnp.where(valid, key_ids, jnp.zeros((), key_ids.dtype))


def _least_after(places: jax.Array, ids: jax.Array | None) -> jax.Array:
    """Each of `places` `[B, K]`, in document order, as the least place at or after it in its document's run
    of equal `ids` (the whole row without them): the places themselves where they never decrease along a
    document, and never decreasing however a padding slot was placed."""
    if places.shape[1] == 0:
        return places
    # A run's last key, where the reversed scan starts again.
    rows, keys = places.shape
    changes = jnp.zeros((rows, keys - 1), bool) if ids is None else ids[:, 1:] != ids[:, :-1]
    ends = jnp.concatenate([changes, jnp.ones((rows, 1), bool)], axis=1)

    def least(left, right):
        return jnp.where(right[1], right[0], jnp.minimum(left[0], right[0])), left[1] | right[1]

    return jax.lax.associative_scan(least, (places, ends), reverse=True, axis=1)[0]


def _widened(positions: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
    """The queries' and keys' places in one integer dtype that holds both, in which a window's bounds move
    (`_moved`): a bound past the narrower one's range is still a place the other may hold. A signed and an
    unsigned dtype that no enabled integer dtype holds both of (int64 beside uint64; int32 beside uint32
    without x64), whose promotion would leave integers or narrow one, are refused for one dtype."""
    query_places, key_places = (jnp.asarray(places) for places in positions)
    ranges = [jnp.iinfo(places.dtype) for places in (query_places, key_places)]
    least, largest = min(int(bounds.min) for bounds in ranges), max(int(bounds.max) for bounds in ranges)
    for wide in map(jnp.dtype, ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64")):
        holds = jnp.iinfo(wide)
        enabled = jax.dtypes.canonicalize_dtype(wide) == wide
        if enabled and int(holds.min) <= least and largest <= int(holds.max):
            return query_places.astype(wide), key_places.astype(wide)
    raise ValueError(f"query places of {query_places.dtype} and key places of {key_places.dtype} share no "
                     "integer dtype that holds both: pass them in one")


def _moved(places: jax.Array, by: int) -> jax.Array:
    """`places` moved by `by`, exactly, in their own integer dtype, held at its least and largest rather
    than wrapping: a window's bound past either end of the dtype keeps every place on that side. The move
    runs on the places' distances from the dtype's least, unsigned, which hold any shift a window makes."""
    limits = jnp.iinfo(places.dtype)
    span = int(limits.max) - int(limits.min)
    if by == 0:
        return places
    if abs(by) > span:
        return jnp.full(places.shape, limits.min if by < 0 else limits.max, places.dtype)
    word = jnp.dtype(f"uint{8 * places.dtype.itemsize}")

    def unsigned(value: int) -> jax.Array:
        return jnp.asarray(value, word)

    flip = unsigned(-int(limits.min))
    # A signed place's distance from the least is its two's complement word with the sign bit flipped.
    distance = jax.lax.bitcast_convert_type(places, word) ^ flip
    if by < 0:
        moved = jnp.where(distance < unsigned(-by), unsigned(0), distance - unsigned(-by))
    else:
        moved = jnp.where(distance > unsigned(span - by), unsigned(span), distance + unsigned(by))
    return jax.lax.bitcast_convert_type(moved ^ flip, places.dtype)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BlockMap:
    """The tiles a launch over `block_q` queries by `block_k` keys visits, both directions, from one
    `KeySpans`: query block i reads key blocks `[lo[b, i], hi[b, i])`, of which `[full_lo, full_hi)` are
    visible in every entry for its every query (the kernel skips the predicate there, where the ids
    agree: `query_doc[b, i] == key_doc[b, j] != 0`, or the call has none); key block j is read by query
    blocks `[q_lo[b, j], q_hi[b, j])`, each of which still checks `lo <= j < hi`. Every visible entry lies
    in a visited tile; a visited tile may hold none (an envelope over a block's spans). Over runs
    (`KeySpans`), query block i reads the union of its runs' tile envelopes, `ranges[b, i]` `[R, 2]`
    sorted, disjoint and nonempty first, `(0, 0)` after, which `[lo, hi)` envelops; `[full_lo, full_hi)`
    are the widest run's tiles every query of the block sees whole. A key block is read where some range
    holds it."""

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
    ranges: jax.Array | None = None


def _blocked(values: jax.Array, block: int, fill) -> jax.Array:
    """`values` `[B, N, ...]` padded with `fill` to whole blocks, `[B, N // block, block, ...]`."""
    pad = -values.shape[1] % block
    widths = ((0, 0), (0, pad), *((0, 0),) * (values.ndim - 2))
    padded = jnp.pad(values, widths, constant_values=fill)
    return padded.reshape(values.shape[0], -1, block, *values.shape[2:])


def _uniform_doc(ids: jax.Array | None, block: int) -> jax.Array | None:
    """Each block's document id where all its entries carry that one, else 0, the padding id, which no
    whole tile carries; a block running past the ids is padded with it."""
    if ids is None:
        return None
    blocks = _blocked(ids, block, 0)
    first = blocks[..., :1]
    return jnp.where(jnp.all(blocks == first, axis=-1), first[..., 0], 0)


def block_map(spans: KeySpans, block_q: int, block_k: int) -> BlockMap:
    """The `BlockMap` of `spans` at `block_q` by `block_k` tiles. One interval a query is a call of one
    run: each run's tile envelope over a block's queries and its tiles every query of the block sees whole,
    the envelopes merged into their sorted union where a call has several (`_merged`), the widest run's
    whole tiles the block's, and each key block read by the query blocks some range of which holds it."""
    runs = spans.start.ndim == 3
    keys = spans.keys
    key_blocks = -(-keys // block_k)
    big = jnp.iinfo(jnp.int32).max
    start, end = (spans.start, spans.end) if runs else (spans.start[..., None], spans.end[..., None])
    start, end = _blocked(start, block_q, 0), _blocked(end, block_q, 0)
    nonempty = end > start
    held = jnp.any(nonempty, axis=2)
    # Each run's envelope over the block's queries, `(0, 0)` where none of them reads it.
    lo_r = jnp.where(held, jnp.min(jnp.where(nonempty, start, big), axis=2) // block_k, 0)
    hi_r = jnp.where(held, -(-jnp.max(jnp.where(nonempty, end, 0), axis=2) // block_k), 0)
    ranges = _merged(lo_r, hi_r, held, key_blocks) if runs else jnp.stack([lo_r, hi_r], axis=-1)
    lo = jnp.where(jnp.any(ranges[..., 1] > ranges[..., 0], axis=-1), ranges[..., 0, 0], 0)
    hi = jnp.max(ranges[..., 1], axis=-1)
    # A run's tile is whole where every query of the block sees all its keys through it: the block's latest
    # start and earliest end bound them, and a query that misses the run leaves none whole.
    whole_lo = -(-jnp.max(start, axis=2) // block_k)
    whole_hi = jnp.minimum(jnp.min(end, axis=2), keys) // block_k
    whole = jnp.all(nonempty, axis=2) & (whole_hi > whole_lo)
    widest = jnp.argmax(jnp.where(whole, whole_hi - whole_lo, -1), axis=-1)[..., None]
    some = jnp.any(whole, axis=-1)
    full_lo = jnp.where(some, jnp.take_along_axis(whole_lo, widest, axis=-1)[..., 0], lo)
    full_hi = jnp.where(some, jnp.take_along_axis(whole_hi, widest, axis=-1)[..., 0], full_lo)
    # The query blocks reading each key block: those some range of which holds it.
    j = jnp.arange(key_blocks)[None, None, :, None]
    reads = jnp.any((ranges[:, :, None, :, 0] <= j) & (j < ranges[:, :, None, :, 1]), axis=-1)
    i = jnp.arange(lo.shape[1])[None, :, None]
    q_lo = jnp.min(jnp.where(reads, i, big), axis=1, initial=big)
    q_hi = jnp.max(jnp.where(reads, i + 1, 0), axis=1, initial=0)
    q_lo = jnp.where(q_hi > 0, q_lo, 0)
    return BlockMap(lo, hi, full_lo, full_hi, q_lo, q_hi, _uniform_doc(spans.query_ids, block_q),
                    _uniform_doc(spans.key_ids, block_k), block_q, block_k, ranges if runs else None)


def _merged(lo: jax.Array, hi: jax.Array, held: jax.Array, key_blocks: int) -> jax.Array:
    """Each query block's run envelopes `[lo, hi)` `[B, Qb, R]` (`held` where a query reads the run) merged
    into their union, `[B, Qb, R, 2]`: sorted, disjoint and nonempty first, `(0, 0)` after. A later run's
    envelope may start inside an earlier one's, so they are sorted first; a range opens where an envelope
    starts past every one before it, and runs to its group's last end."""
    big = jnp.iinfo(jnp.int32).max
    lo, hi = jnp.where(held, lo, key_blocks), jnp.where(held, hi, key_blocks)
    order = jnp.argsort(lo, axis=-1, stable=True)
    lo_s, hi_s = (jnp.take_along_axis(bound, order, axis=-1) for bound in (lo, hi))
    covered = jax.lax.cummax(hi_s, axis=hi_s.ndim - 1)
    before = jnp.concatenate([jnp.full_like(covered[..., :1], -1), covered[..., :-1]], axis=-1)
    opens = lo_s > before
    group = jnp.cumsum(opens, axis=-1)
    same = group[..., :, None] == group[..., None, :]
    closes = jnp.max(jnp.where(same, hi_s[..., None, :], 0), axis=-1)
    kept = opens & (closes > lo_s)
    rank = jnp.argsort(jnp.where(kept, lo_s, big), axis=-1, stable=True)
    return jnp.stack([jnp.take_along_axis(jnp.where(kept, bound, 0), rank, axis=-1)
                      for bound in (lo_s, closes)], axis=-1)


def ranges_walk(ranges: jax.Array) -> tuple[jax.Array, Callable[[jax.Array], jax.Array]]:
    """A query block's walk over its tile `ranges` `[R, 2]` (`BlockMap.ranges`): its step count, and each
    step's key tile, the ranges' tiles in turn, by a select over the ranges every mainloop lowers."""
    lo, hi = ranges[..., 0], ranges[..., 1]
    lengths = hi - lo
    offsets = jnp.cumsum(lengths, axis=-1) - lengths

    def tile(step: jax.Array) -> jax.Array:
        inside = (offsets <= step) & (step < offsets + lengths)
        return jnp.sum(jnp.where(inside, lo + step - offsets, 0))

    return jnp.sum(lengths), tile


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
    unnormalized denominator `total` `[B, H, Q]` and numerator `acc` `[B, H, Q, Dv]`, fp32 or wider.
    Sinks are `m` the head's sink, `total` one and `acc` zero (`sinks`); a resumed partial is its own
    state."""

    m: jax.Array
    total: jax.Array
    acc: jax.Array


def sinks(values: jax.Array, batch: int, queries: int, width: int) -> Initial:
    """The `Initial` of one value-free logit a query head, `values` `[H]`, in fp32 or their own wider
    dtype (`at_least_fp32`)."""
    wide = at_least_fp32(values.dtype)
    m = jnp.broadcast_to(values.astype(wide)[None, :, None], (batch, values.shape[0], queries))
    return Initial(m, jnp.ones_like(m), jnp.zeros((*m.shape, width), wide))


def capped(softcap: float, logits: jax.Array, q_pos: jax.Array, k_pos: jax.Array) -> jax.Array:
    """Gemma 2's logit cap, `softcap * tanh(logits / softcap)`, on the scaled logits as `reference_attention`
    takes it. It is a score hook as `functools.partial(capped, softcap)`, which reads no position."""
    del q_pos, k_pos
    return softcap * jnp.tanh(logits / softcap)


DROP_DOMAIN = 0x6A09E667
"""The keep draw's domain tag: the fraction bits of sqrt(2), so the draw's first word is one no other
`fmix32` user mixes first, and its stream never coincides with theirs."""


@dataclasses.dataclass(frozen=True)
class Draw:
    """One program's keep draw for probability dropout (`Coords.drop`): the seed's two words, the global
    batch row and query head the program holds, and the rate. Called on query and key indices, which
    broadcast, it gives each entry's multiplier: `1 / (1 - rate)` where the draw keeps the entry, 0 where
    it drops it, as a float32 array. The seed's words, the domain tag, the row, the head and the query mix
    into one word, one `fmix32` each, and each key mixes into it by one more, so every mainloop and the
    reference run the same ops and their masks agree bit for bit at the same coordinates."""

    seed: tuple[jax.Array, jax.Array]
    row: jax.Array
    head: jax.Array
    rate: float
    queries: jax.Array | None = None
    keys: jax.Array | None = None

    @property
    def bound(self) -> float:
        """The multiplier a kept entry takes, `1 / (1 - rate)`, the largest an entry's weight takes."""
        return 1.0 / (1.0 - self.rate)

    def keep(self, queries: jax.Array, keys: jax.Array) -> jax.Array:
        """Whether the draw keeps the entries at `queries` and `keys` (indices that broadcast): the keep bits
        every mainloop and the reference share, each taking its own scale for a kept entry. Where the draw
        holds the call's global `queries` and `keys` (a sequence exchange's slice, `DropPlaces`), the indices
        are local and read their global places there."""
        # A padded tile's places past a table read its fill, and an empty table (a slice of no queries or
        # keys) is read at no place: no real entry draws there.
        if self.queries is not None and self.queries.shape[0]:
            queries = jnp.take(self.queries, queries)
        if self.keys is not None and self.keys.shape[0]:
            keys = jnp.take(self.keys, keys)
        word = fmix32(self.seed[0] ^ jnp.uint32(DROP_DOMAIN))
        for part in (self.seed[1], self.row, self.head, queries):
            word = fmix32(word ^ jnp.asarray(part).astype(jnp.uint32))
        bits = fmix32(word ^ jnp.asarray(keys).astype(jnp.uint32))
        return bits >= jnp.uint32(int(self.rate * 2 ** 32))

    def __call__(self, queries: jax.Array, keys: jax.Array) -> jax.Array:
        """The kernels' multiplier of the entries at `queries` and `keys`: `bound` where kept, in fp32, the
        kernels' precision (a reference takes its own scale from `keep`)."""
        return jnp.where(self.keep(queries, keys), jnp.float32(self.bound), jnp.float32(0.0))


def dropout_seed(dropout_rng) -> jax.Array:
    """The two uint32 words a dropout key draws a call's keep mask from (`Draw`)."""
    return jax.random.bits(dropout_rng, (2,), jnp.uint32)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DropPlaces:
    """The global places a slice of a call draws its dropout at, where a sequence exchange holds a slice of
    its rows, heads, queries or keys (`dew.nn.attention.sequence_parallel_attention`): each local batch
    row's `rows` `[B]`, query head's `heads` `[H]`, query's `queries` `[Q]` and key's `keys` `[K]` global
    index, so every device drops the weights one device holding the whole call drops."""

    rows: jax.Array
    heads: jax.Array
    queries: jax.Array
    keys: jax.Array


def dropout_keep(seed: jax.Array, rate: float, shape, places: DropPlaces | None = None) -> jax.Array:
    """Which weights of `shape` `[..., H, Q, K]` probability dropout keeps, true where the draw keeps one.
    Each weight takes its batch row (the leading dims, flattened), head, query and key, the coordinates the
    attention templates draw at (`Draw`), their global places where a sequence exchange holds a slice of
    the call (`places`, its leading dims one); a caller scales a kept weight by `1 / (1 - rate)` in its
    own dtype."""
    *rows, heads, queries, keys = shape
    if places is None:
        places = DropPlaces(jnp.arange(math.prod(rows)), jnp.arange(heads), jnp.arange(queries),
                            jnp.arange(keys))
    draw = Draw((seed[0], seed[1]), places.rows.reshape(*rows, 1, 1, 1), places.heads.reshape(heads, 1, 1),
                rate)
    return draw.keep(places.queries[:, None], places.keys[None, :])


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Dropout:
    """Attention's probability dropout as a weights hook (`Hooks.drop`): each softmax weight at a query
    and key is kept where its draw keeps it and multiplied by `1 / (1 - rate)`, while the softmax's
    denominator takes the undropped weights, so the output is the softmax over the values `z v`. `seed`
    holds the call's two uint32 words, `rows` `[B]` and `heads` `[H]` each local batch row's and query
    head's global index (a shard's own positions are not, `_per_shard`), `queries` `[Q]` and `keys` `[K]`
    each local query's and key's where a sequence exchange holds a slice of them (`DropPlaces`; local are
    global where None), and `rate` is static."""

    seed: jax.Array
    rows: jax.Array
    heads: jax.Array
    rate: float = dataclasses.field(metadata={"static": True})
    queries: jax.Array | None = None
    keys: jax.Array | None = None

    def draw(self, row, head) -> Draw:
        """The draw of the local batch row `row` and query head `head`, indices into `rows` and `heads`, at
        the call's global `queries` and `keys` where a sequence exchange holds a slice of them."""
        return Draw((self.seed[0], self.seed[1]), self.rows[row], self.heads[head], self.rate, self.queries,
                    self.keys)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Hooks:
    """The template's hooks, each absent where None: `score_mod(s, q_pos, k_pos, *params)` on the scaled
    logits `s` `[..., q, k]` with `q_pos` `[q, 1]` and `k_pos` `[1, k]` the entries' query and key
    indices, pointwise in the entries; `initial` the softmax's start; `output_mod(o, *params)` on the
    normalized output `[..., q, Dv]`, pointwise in its entries too (a gate, a scale): a kernel applies it
    to an output tile whose head width pads to a power of two. A pytree whose callables are static and
    whose parameters and state are its leaves, so a compiled call takes new values of them. `drop` is
    probability dropout's weights hook (`Dropout`), absent where None."""

    score_mod: Callable | None = dataclasses.field(default=None, metadata={"static": True})
    score_params: tuple[Param, ...] = ()
    initial: Initial | None = None
    output_mod: Callable | None = dataclasses.field(default=None, metadata={"static": True})
    output_params: tuple[Param, ...] = ()
    drop: Dropout | None = None


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


def prescale(dtype, width: int, scale: float) -> tuple[float, float]:
    """The power of two a call's queries take before their products with the keys, and the scale left after
    it: where the products of `dtype` operands `width` wide can pass fp32's largest (bf16's and fp32's),
    `scale`'s power of two at or below its magnitude, so no product the reference keeps finite overflows
    here (it scales the query before its products, `dew.nn.attention`): Q 2^127 and K 4 over a width of 16,
    a product of 2^129 where the reference's is 2^127; else 1 and `scale` (fp16's products never pass it). A
    power of two scales exactly, so the logits, the weights and every gradient but the query's are the
    unscaled call's bit for bit, and the query's takes the power back (`gradients`)."""
    largest = float(jnp.finfo(dtype).max)
    if scale == 0 or largest * largest * width < float(jnp.finfo(jnp.float32).max):
        return 1.0, scale
    power = 2.0 ** math.floor(math.log2(abs(scale)))
    return power, scale / power


def prescaled(query: jax.Array, scale: float) -> tuple[jax.Array, float, float]:
    """`query` taken by its call's power of two before its products (`prescale`), the scale left after it,
    and the power."""
    power, left = prescale(query.dtype, query.shape[-1], scale)
    return (query if power == 1.0 else query * jnp.asarray(power, query.dtype)), left, power


def scores(query: jax.Array, key: jax.Array, spans: KeySpans, hooks: Hooks | None = None, *,
           scale: float | None = None) -> jax.Array:
    """The scores `[B, H, Q, K]` the template's softmax takes, as `reference` computes them: the products
    of `query` `[B, Q, H, D]` and `key` `[B, K, Hk, D]` accumulated in fp32 (fp64 for fp64 inputs), the
    scale, then `hooks.score_mod`; -inf where `spans` hides the key."""
    hooks = Hooks() if hooks is None else hooks
    batch, queries, heads, width = query.shape
    kv_heads = key.shape[2]
    wide = jnp.promote_types(query.dtype, jnp.float32)
    query, scale, _ = prescaled(query, width ** -0.5 if scale is None else scale)
    seen = dense(spans)
    grouped = query.reshape(batch, queries, kv_heads, heads // kv_heads, width)
    products = float32_einsum("bqhgd,bkhd->bhgqk", grouped, key,
                              preferred_element_type=wide).reshape(batch, heads, queries, spans.keys)
    products = products * jnp.asarray(scale, wide)

    def one(row, head):
        s = products[row, head]
        if hooks.score_mod is not None:
            q_pos = jnp.arange(queries)[:, None]
            k_pos = jnp.arange(spans.keys)[None, :]
            s = hooks.score_mod(s, q_pos, k_pos, *(_per_head(p, row, head) for p in hooks.score_params))
        return jnp.where(seen[row], s, -jnp.inf)

    return jax.vmap(jax.vmap(one, in_axes=(None, 0)), in_axes=(0, None))(jnp.arange(batch), jnp.arange(heads))


def reference(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
              *, scale: float | None = None, residuals: bool = False):
    """The template's arithmetic in XLA ops: `query` `[B, Q, H, D]` over `key` and `value` `[B, K, Hk, D]` and
    `[B, K, Hk, Dv]`, a query head reading the key head it shares with `H // Hk` others, the keys `spans`
    leaves it. As the kernels compute it: products of the inputs accumulated in fp32 (fp64 for fp64
    inputs) as Dew's every product is (`dew.nn.precision.float32_einsum`, which a CPU under Dew's tests
    sums in float64 and rounds once, as the reference path's), the scale and `score_mod` on the fp32
    logits, the softmax from `hooks.initial` in fp32 (or the wider of the inputs' and the state's dtypes),
    the probabilities rounded to the values' dtype for their product, the output normalized in fp32 and
    `output_mod` applied before it rounds to the query's dtype. A query that sees no key and starts from
    no state gives zeros. With `residuals`, also each query's running maximum `m` and denominator `total`
    `[B, H, Q]` (natural log, unnormalized), in the softmax's dtype, a query that sees no key and starts
    from no state holding 0 for both, so every residual is finite; a softmax resumed from them (`Initial`)
    starts such a query from `m = -inf`. No hooks where `hooks` is None."""
    hooks = Hooks() if hooks is None else hooks
    batch, queries, heads, _ = query.shape
    kv_heads = key.shape[2]
    wide = jnp.promote_types(query.dtype, jnp.float32)
    positions = jnp.arange(queries)[:, None], jnp.arange(spans.keys)[None, :]
    if hooks.initial is not None:
        for state in (hooks.initial.m, hooks.initial.total, hooks.initial.acc):
            wide = jnp.promote_types(wide, state.dtype)
    logits = scores(query, key, spans, hooks, scale=scale).astype(wide)

    if hooks.initial is None:
        m0 = jnp.full((batch, heads, queries), -jnp.inf, wide)
        total0 = jnp.zeros((batch, heads, queries), wide)
        acc0 = jnp.zeros((batch, heads, queries, value.shape[-1]), wide)
    else:
        m0 = hooks.initial.m.astype(wide)
        total0 = hooks.initial.total.astype(wide)
        acc0 = hooks.initial.acc.astype(wide)
    m = jnp.maximum(m0, jnp.max(logits, axis=-1, initial=-jnp.inf))
    safe = jnp.where(jnp.isfinite(m), m, 0)
    p = jnp.exp(logits - safe[..., None])
    carried = jnp.where(jnp.isfinite(m0), jnp.exp(m0 - safe), 0)
    total = total0 * carried + jnp.sum(p, axis=-1)

    def valued(weights: jax.Array) -> jax.Array:
        # The weights narrowed to the values' dtype times the values, a query group's over its key head.
        grouped = weights.astype(value.dtype).reshape(batch, kv_heads, heads // kv_heads, queries, spans.keys)
        return float32_einsum("bhgqk,bkhd->bhgqd", grouped, value,
                              preferred_element_type=wide).reshape(batch, heads, queries, value.shape[-1])

    drop = hooks.drop
    if drop is None:
        acc = acc0 * carried[..., None] + valued(p)
        normalized = jnp.where(total[..., None] > 0, acc / jnp.where(total > 0, total, 1)[..., None], 0)
    else:
        # Normalized before the draw's multiplier narrows them, as the kernels take them: a kept weight's z
        # times its normalized weight stays in the values' range where z times the unnormalized one may not
        # (fp16 at a rate of 1 - 2^-16). The state's share is normalized the same way.
        inverse = jnp.where(total > 0, 1 / jnp.where(total > 0, total, 1), 0)

        def multiplier(row, head):
            # The keep bits are the draw's, shared with the kernels; a kept entry takes its own dtype's
            # 1 / (1 - rate), rounded once from its double, so a float64 reference is exact at its rate.
            draw = drop.draw(row, head)
            return jnp.where(draw.keep(*positions), jnp.asarray(draw.bound, p.dtype), 0)

        drawn = jax.vmap(jax.vmap(multiplier, in_axes=(None, 0)), in_axes=(0, None))(jnp.arange(batch),
                                                                                    jnp.arange(heads))
        normalized = acc0 * (carried * inverse)[..., None] + valued((p * inverse[..., None]) * drawn)

    out = output_hooked(normalized, hooks.output_mod, hooks.output_params)
    out = jnp.swapaxes(out.astype(query.dtype), 1, 2)
    return (out, safe, total) if residuals else out


def output_hooked(out: jax.Array, output_mod: Callable | None, output_params: tuple[Param, ...]) -> jax.Array:
    """The normalized output `out` `[B, H, Q, Dv]` under `output_mod`, each batch row's and head's `[Q, Dv]`
    with its parameters as the reference reads them (`_per_head`); `out` itself where there is no hook."""
    if output_mod is None:
        return out

    def one(row, head):
        return output_mod(out[row, head], *(_per_head(param, row, head) for param in output_params))

    batch, heads = out.shape[:2]
    return jax.vmap(jax.vmap(one, in_axes=(None, 0)), in_axes=(0, None))(jnp.arange(batch), jnp.arange(heads))


LOG2E = 1.4426950408889634
"""log2(e): a natural exponent times it is a base-2 one."""


NATURAL = False
"""Whether attention keeps its logits natural unless a call says otherwise
(`attention_algebra.Forward.base2`): False, the base-2 fold, the fast form, until the natural form's cost
closes; every kernel's default reads it, so the default switches here."""

ACCURATE = False
"""Whether attention's backward takes its accurate products unless a call says otherwise: False, the fast
form, the template's own D about each query's anchor (`attention_algebra.split_delta`, not
FlashAttention's rowsum(dO * O)) with each bf16 score cotangent narrowed for one product and dQ at that
first D, as FlashAttention's arithmetic takes them; True, the cotangents' products to fp32's accuracy and
dQ at the refined D (`attention_algebra.cotangent_product`, `dq_finish`). The template's accurate form is
this with natural logits (`natural=True, accurate=True`, a forward and backward of the one form), which the
stability suite holds to check 1 at extreme inputs, where the default holds to finiteness (the owner's
ruling, 2026-10-23). Every backward's default reads it."""


def kernel_dot(a: jax.Array, b: jax.Array, interpret: bool) -> jax.Array:
    """`a @ b` accumulated in fp32 as a template's Triton kernels issue it: bf16's and fp16's own product,
    IEEE fp32 for fp32 operands; interpreted, the template's CPU product (`interpreted_dot`)."""
    if interpret:
        return interpreted_dot(a, b)
    from jax.experimental.pallas import triton as plgpu

    return plgpu.dot(a, b, precision=jax.lax.Precision.HIGHEST).astype(jnp.float32)


def interpreted_dot(a: jax.Array, b: jax.Array) -> jax.Array:
    """`a @ b`, fp32, as every CPU arm of the attention template issues it (the plain-JAX mainloop, the
    interpreted kernels): the product its operands declare at the precision policy's precision
    (`dew.nn.precision.float32_products`), issued off a GPU by the declaration's owner
    (`product.interpreted`), so an fp32 product accumulates in float64, within that sum's rounding bound
    whatever the tile."""
    from dew.nn.kernels.templates import product
    from dew.nn.precision import float32_products, products_dtype

    dtype = products_dtype(a, b)
    declared = product.declared(dtype, float32_products(dtype), outputs=(jnp.float32,))
    return product.interpreted(declared, a, b, (((a.ndim - 1,), (0,)), ((), ())))


def power_of_two(width: int) -> int:
    """The power of two from 16 that holds `width`: a head width a kernel tiles."""
    return max(16, 1 << (width - 1).bit_length())


def whole(count: int, block: int) -> int:
    """`count` rounded up to whole `block`s."""
    return -(-count // block) * block


def _padded(param: Param, queries: int, keys: int, value_width: int) -> jax.Array:
    """A hook parameter at the padded call's shape, in fp32 as the kernels compute (its gradient returns
    in its own dtype): zeros past its queries, keys and value width, and a head vector at a power of
    two."""
    value = param.value
    if jnp.issubdtype(value.dtype, jnp.floating):
        value = value.astype(jnp.float32)
    match param.kind:
        case Kind.HEAD:
            return jnp.pad(value, (0, power_of_two(value.shape[0]) - value.shape[0]))
        case Kind.QUERY:
            return jnp.pad(value, ((0, 0), (0, 0), (0, queries - value.shape[2])))
        case Kind.KEY:
            return jnp.pad(value, ((0, 0), (0, 0), (0, keys - value.shape[2])))
        case Kind.TILE:
            return jnp.pad(value, ((0, 0), (0, 0), (0, queries - value.shape[2]), (0, keys - value.shape[3])))
        case Kind.OUTPUT:
            return jnp.pad(value, ((0, 0), (0, queries - value.shape[1]), (0, 0),
                                   (0, value_width - value.shape[3])))


@dataclasses.dataclass(frozen=True)
class PaddedCall:
    """A call at the launches' padded shapes (`padded_call`): its queries and keys at whole blocks of
    a backend's largest tile (at least one, so a call of no keys still has a tile to trace), its head
    widths at powers of two, its spans' rows and its score and output hooks' parameters alike; the
    padding is unread and sliced off."""

    query: jax.Array
    key: jax.Array
    value: jax.Array
    start: jax.Array
    end: jax.Array
    query_ids: jax.Array | None
    key_ids: jax.Array | None
    score_params: tuple[jax.Array, ...]
    output_params: tuple[jax.Array, ...]
    queries: int
    keys: int

    def rows(self, array: jax.Array, fill: float = 0.0) -> jax.Array:
        """A `[B, H, Q, ...]` row statistic at the padded queries."""
        pad = [(0, 0), (0, 0), (0, self.query.shape[1] - array.shape[2]), *[(0, 0)] * (array.ndim - 3)]
        return jnp.pad(array, pad, constant_values=fill)


def padded_call(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks,
                block: int) -> PaddedCall:
    """A call's one padding for every mainloop of the template (`PaddedCall`): its queries and keys at
    whole `block`s (a backend's largest tile), its head widths at powers of two."""
    queries, width = query.shape[1], query.shape[3]
    keys, value_width = key.shape[1], value.shape[-1]
    padded_q, padded_k = whole(max(queries, 1), block), whole(max(keys, 1), block)
    padded_d, padded_dv = power_of_two(width), power_of_two(value_width)

    def pad(array, rows, columns):
        return jnp.pad(array, ((0, 0), (0, rows - array.shape[1]), (0, 0), (0, columns - array.shape[3])))

    def pad_rows(array: jax.Array, rows: int) -> jax.Array:
        return jnp.pad(array, ((0, 0), (0, rows - array.shape[1]), *((0, 0),) * (array.ndim - 2)))

    def pad_ids(array: jax.Array | None, rows: int) -> jax.Array | None:
        return None if array is None else pad_rows(array, rows)

    return PaddedCall(pad(query, padded_q, padded_d), pad(key, padded_k, padded_d),
                   pad(value, padded_k, padded_dv), pad_rows(spans.start, padded_q),
                   pad_rows(spans.end, padded_q), pad_ids(spans.query_ids, padded_q),
                   pad_ids(spans.key_ids, padded_k),
                   tuple(_padded(param, padded_q, padded_k, padded_dv) for param in hooks.score_params),
                   tuple(_padded(param, padded_q, padded_k, padded_dv) for param in hooks.output_params),
                   queries, keys)


def kept_dtype(dtype) -> str:
    """A hook parameter's dtype in the kernels: fp32 for a floating one (`padded_call`), else its own."""
    return "float32" if jnp.issubdtype(dtype, jnp.floating) else jnp.dtype(dtype).name


def floating_params(params) -> list[int]:
    """The places of a hook's floating parameters, the ones it differentiates along."""
    return [index for index, param in enumerate(params) if jnp.issubdtype(param.dtype, jnp.floating)]


def differentiable(score_mod: Callable, q_pos, k_pos, params, floating: list[int]) -> Callable:
    """`score_mod` at the tile's coordinates as a function of its logits and its floating parameters
    (`floating`) alone, the others held: what `jax.vjp` takes."""
    def hooked(s, *wide):
        values = list(params)
        for index, value in zip(floating, wide, strict=True):
            values[index] = value
        return score_mod(s, q_pos, k_pos, *values)

    return hooked


TRACED_TILE = 16
"""The tile a hook is traced at for its picks' key (`hook_program`)."""


def hook_program(name: str, hook, kinds: tuple[Kind, ...], dtypes: tuple[str, ...], *,
                 differentiated: bool = False, width: int = TRACED_TILE) -> str | None:
    """What a score or output hook computes, as its launch's picks key it (`tuning.program`): a score
    hook traced at a `TRACED_TILE` square fp32 tile, an output hook at `TRACED_TILE` queries of the
    call's padded value `width`, the entries' indices and each parameter as its kind lays it out there,
    in its dtype in the kernels (`dtypes`); `differentiated`, its pullback too, so a hook whose
    derivative is its own rule (`jax.custom_vjp`) keys by that rule. None or an already traced key as
    it is."""
    if hook is None or isinstance(hook, str):
        return hook
    tile = TRACED_TILE
    layouts = {Kind.HEAD: (), Kind.QUERY: (tile, 1), Kind.KEY: (1, tile), Kind.TILE: (tile, tile),
               Kind.OUTPUT: (tile, width)}
    params = [jax.ShapeDtypeStruct(layouts[kind], dtype) for kind, dtype in zip(kinds, dtypes, strict=True)]
    square = jax.ShapeDtypeStruct((tile, tile), jnp.float32)
    if name == "output_mod":
        return tuning.program(hook, jax.ShapeDtypeStruct((tile, width), jnp.float32), *params)
    positions = (jax.ShapeDtypeStruct((tile, 1), jnp.int32), jax.ShapeDtypeStruct((1, tile), jnp.int32))
    if not differentiated:
        return tuning.program(hook, square, *positions, *params)
    floating = floating_params(params)

    def pulled(s, q_pos, k_pos, *values):
        hooked = differentiable(hook, q_pos, k_pos, values, floating)
        out, pull = jax.vjp(hooked, s, *(values[index] for index in floating))
        return out, pull(jnp.ones_like(out))

    return tuning.program(pulled, square, *positions, *params)


class AttentionLaunch(tuning.KernelOp):
    """What every launch of the template shares as the tuner sees it (ahead of the op's own bases): its hooks
    keyed by what each computes (`tuning.KernelOp.traced`, `hook_program`), a score hook at its parameters'
    kinds and dtypes (`score_kinds`, `score_dtypes`) and with its pullback where the launch differentiates
    it (`DIFFERENTIATED`, a backward's), a forward's output hook at its parameters' and the padded value
    width (`output_kinds`, `output_dtypes`, `output_width`); and its timing operands' spans whole
    (`tuning.KernelOp.operands`, `spanned`), the spans' starts at its arrays' place `SPANS`."""

    DIFFERENTIATED: ClassVar[bool] = False
    SPANS: ClassVar[int] = 3
    """The place of the spans' starts among the launch's array arguments, their ends the next, then the
    queries' and keys' ids where the call has them (a forward's after its queries, keys and values)."""
    score_kinds: tuple[Kind, ...] = ()
    score_dtypes: tuple[str, ...] = ()
    output_kinds: tuple[Kind, ...] = ()
    output_dtypes: tuple[str, ...] = ()
    output_width: int = TRACED_TILE

    def traced(self, name: str, value: Callable | str | None) -> str | None:
        if name == "output_mod":
            return hook_program(name, value, self.output_kinds, self.output_dtypes, width=self.output_width)
        return hook_program(name, value, self.score_kinds, self.score_dtypes,
                            differentiated=self.DIFFERENTIATED)

    def operands(self, specs: list[jax.ShapeDtypeStruct], seed: int) -> list[jax.Array]:
        return spanned(specs, super().operands(specs, seed), self.SPANS)


def spanned(specs: list[jax.ShapeDtypeStruct], arrays: list[jax.Array], first: int) -> list[jax.Array]:
    """A launch's timing operands with its spans whole, so a timed launch visits every tile a dense call
    does: its spans' starts and ends at `specs[first]` and the next (`[B, Q]`, or a call's runs `[B, Q, R]`),
    each query's interval, or its first run, every key (`specs[1]` being the keys) and its other runs
    empty past them; and its queries' and keys' ids where the call has them, the two integer arrays after
    them, one document. Every other operand keeps its draw."""
    keys = specs[1].shape[1]
    start, end = specs[first], specs[first + 1]
    runs = start.shape[2:]
    opened = (jnp.arange(math.prod(runs)) == 0).reshape(runs)
    arrays[first] = jnp.broadcast_to(jnp.where(opened, 0, keys), start.shape).astype(start.dtype)
    arrays[first + 1] = jnp.full(end.shape, keys, end.dtype)
    ids = [index for index in (first + 2, first + 3)
           if index < len(specs) and jnp.issubdtype(specs[index].dtype, jnp.integer)]
    if len(ids) == 2:
        for index in ids:
            arrays[index] = jnp.ones(specs[index].shape, specs[index].dtype)
    return arrays


class Split(NamedTuple):
    """A training forward's split sums (`attention_algebra.training_step`), residuals the backward
    reads in place of its first pass: the weighted values of every key but each query's anchor key,
    with the initial state's, `others` `[B, H, Q, Dv]`, the anchor's own weight `own` and the other
    keys' weights' sum `others_total` `[B, H, Q]`, fp32, all at the forward's maximum and at each row's
    `2^-shift` `[B, H, Q]` (`attention_algebra.room`)."""

    others: jax.Array
    own: jax.Array
    others_total: jax.Array
    shift: jax.Array


DQ_GRADS = (Kind.HEAD, Kind.QUERY)
"""The parameter kinds whose gradients the dQ program writes: a head's partial sum over its queries and
its queries' own."""

DKV_GRADS = (Kind.KEY, Kind.TILE)
"""The parameter kinds whose gradients the dK/dV program writes, under the refined D: its keys' and its
tiles' entries."""


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Gradients:
    """The backward's gradients: of the queries, keys and values, of each score hook parameter (the
    order `Hooks.score_params` gives them), and of the softmax's initial state where it has one."""

    query: jax.Array
    key: jax.Array
    value: jax.Array
    score_params: tuple[jax.Array, ...]
    initial: Initial | None


def gradients(launch: Callable, query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans,
              hooks: Hooks | None, dout: jax.Array,
              residuals: tuple[jax.Array, jax.Array, jax.Array, Split], *, block: int,
              scale: float | None = None, rounded: bool = True) -> Gradients:
    """The template's backward of its forward under the output's cotangent `dout` `[B, Q, H, Dv]`, from the
    forward's `residuals` (each query's maximum and denominator `[B, H, Q]`, its anchor key, the first
    holding the keys' maximum, `[B, H, Q]` int32, and its split sums, `Split`), through a mainloop's
    `launch` (the Triton one, `attention_triton.backward`; the plain-JAX one,
    `attention_portable.backward`): the call padded to whole `block`s (`padded_call`), `launch(query, key,
    value, dout, peak0, total0, initial_term, maximum, key_index, anchor_value, others, own, others_total,
    total, start, end, query_ids, key_ids, score_params, *, score_mod, score_kinds, score_dtypes, scale)`
    giving dQ, the softmax's rows (`D - c`, `c`, `m`, `T`, the keys' mass), each query's anchor key as it
    chose it, the dQ program's hook gradients (`DQ_GRADS`), then dK, dV and the dK/dV program's
    (`DKV_GRADS`); each gradient fp32, then rounded to its input's dtype. D is taken two ways, about 0 and
    about the anchor key's `v_c · dO`, each query the nearer, from the forward's split sums
    (`attention_algebra.split_delta`) or a first pass of the dQ launch's own over the keys at the
    forward's maximum (`off_anchor_delta`): the anchor key's `dP - c` enters as exactly 0; the main pass
    then refines D once from its own sums for the dK/dV launch and the head, query and initial state's
    gradients (`attention_algebra.refinement`). The output hook is no part of it: a differentiated call
    applies `output_mod` past the kernel, where its own derivative runs, and passes `dout` before it.
    Deterministic where `launch` writes every gradient element from one program (the two-launch schedules: a
    head parameter's partial sums added in a fixed order and a key head's query heads summed in XLA); the
    Mosaic one-launch schedule adds dQ's tiles atomically, in their arrival order."""
    hooks = Hooks() if hooks is None else hooks
    if hooks.output_mod is not None:
        raise ValueError("the backward takes the output before the output hook (`backward`'s docstring)")
    batch, queries, heads, width = query.shape
    keys, kv_heads, value_width = key.shape[1], key.shape[2], value.shape[-1]
    group = heads // kv_heads
    # The query's products taken as the forward took them (`prescale`), its gradient the power back.
    query, scale, power = prescaled(query, width ** -0.5 if scale is None else scale)
    # One cotangent for every part: the output's, in the operands' dtype as the reference takes it.
    dout = dout.astype(query.dtype)
    cotangent = jnp.swapaxes(dout.astype(jnp.float32), 1, 2)
    state = hooks.initial
    if state is None:
        nothing = jnp.zeros((batch, heads, queries), jnp.float32)
        starts = nothing - jnp.inf, nothing, nothing
    else:
        # The softmax's initial state: its maximum, denominator and its values' mean's product with the
        # cotangent, in fp32 (`attention_algebra.off_anchor_start`): `acc0 · dO` itself overflows where the
        # state's sums are near fp32's largest; its mean does not.
        total0 = state.total.astype(jnp.float32)
        over = jnp.where(total0 > 0, 1.0 / jnp.where(total0 > 0, total0, 1.0), 0.0)
        mean0 = state.acc.astype(jnp.float32) * over[..., None]
        starts = state.m.astype(jnp.float32), total0, jnp.sum(mean0 * cotangent, axis=-1)
    call = padded_call(query, key, value, spans, hooks, block)
    padded_dout = jnp.pad(dout, ((0, 0), (0, call.query.shape[1] - queries), (0, 0),
                                 (0, call.value.shape[-1] - value_width)))
    operands = (call.query, call.key, call.value, padded_dout)
    rows = (call.start, call.end, call.query_ids, call.key_ids, call.score_params)
    kinds = tuple(param.kind for param in hooks.score_params)
    dtypes = tuple(kept_dtype(param.value.dtype) for param in hooks.score_params)
    maxima, totals, anchors, split = residuals
    key_index = call.rows(anchors.astype(jnp.int32))
    widened = ((0, 0),) * 3 + ((0, call.value.shape[-1] - value_width),)
    others = jnp.pad(call.rows(split.others.astype(jnp.float32)), widened)
    own, others_total = call.rows(split.own), call.rows(split.others_total)
    shift = call.rows(split.shift.astype(jnp.float32))
    # The anchor key's value row for each query, `[B, Qp, H, Dvp]`, a key head's for its query heads.
    anchor_value = jnp.take_along_axis(jnp.repeat(call.value, group, axis=2),
                                       jnp.swapaxes(key_index, 1, 2)[..., None], axis=1)
    maximum = call.rows(maxima.astype(jnp.float32))
    outputs = launch(*operands, *(call.rows(start) for start in starts), maximum, key_index, anchor_value,
                     shift, others, own, others_total, call.rows(totals.astype(jnp.float32)), *rows,
                     score_mod=hooks.score_mod, score_kinds=kinds, score_dtypes=dtypes, scale=scale)
    dq, softmax = outputs[0], outputs[1:6]
    grads_out = list(outputs[7:])
    dq_grads = [grads_out.pop(0) for kind in kinds if kind in DQ_GRADS]
    dk, dv, *key_grads = grads_out
    delta, anchor, peak, total, mass = (array[:, :, :queries] for array in softmax)
    delta = anchor + delta
    padded_k, padded_d, padded_dv = call.key.shape[1], call.key.shape[-1], call.value.shape[-1]
    dk = dk.reshape(batch, padded_k, kv_heads, group, padded_d).sum(axis=3)
    dv = dv.reshape(batch, padded_k, kv_heads, group, padded_dv).sum(axis=3)
    grads = []
    for param in hooks.score_params:
        match param.kind:
            case Kind.HEAD:
                grads.append(dq_grads.pop(0).sum(axis=(0, 1))[:param.value.shape[0], 0])
            case Kind.QUERY:
                grads.append(dq_grads.pop(0)[:, :, :queries])
            case Kind.TILE:
                grads.append(key_grads.pop(0)[:, :, :queries, :keys])
            case Kind.KEY:
                grads.append(key_grads.pop(0)[:, :, :keys])
            case Kind.OUTPUT:
                raise ValueError("a score hook takes no output parameter")
    initial = None
    if state is not None:
        # The initial state as one more softmax term, its weight exp(m0 - m) / T and value acc0 / total0:
        # each gradient divides by T first and takes the factor exp(m0 - m) last, so a state that weighs
        # little beside the keys (a subnormal factor) is rounded once there, as the reference rounds it.
        # Bounded as the launch's own carried factor is (`attention_algebra.below`): the forward's
        # maximum is the state's too, whatever its rounding.
        carried = jnp.exp(jnp.minimum(starts[0] - peak, 0.0))
        factor = jnp.where((total > 0) & jnp.isfinite(starts[0]), carried, 0.0)
        inverse = jnp.where(total > 0, 1.0 / jnp.where(total > 0, total, 1.0), 0.0)
        # Its maximum's cotangent, that term's score cotangent, taken whole as the reference's derivative
        # takes it; exactly 0 for a query that read only its state, whose output its maximum leaves unmoved.
        # Its weight before any product, as the launch takes it (`attention_algebra.refinement`).
        moved = (starts[2] - delta) * ((starts[1] * factor) * inverse)
        initial = Initial(m=jnp.where(mass > 0, moved, 0.0).astype(state.m.dtype),
                          total=(-(delta * inverse) * factor).astype(state.total.dtype),
                          acc=((cotangent * inverse[..., None]) * factor[..., None]).astype(state.acc.dtype))
    # Unrounded (`rounded` False), dK and dV stay in at least fp32 for a caller that sums them across
    # shards first and rounds once (`attention_route.attend`'s `operand_dtype`).
    key_dtype, value_dtype = ((key.dtype, value.dtype) if rounded
                              else (at_least_fp32(key.dtype), at_least_fp32(value.dtype)))
    dq = dq if power == 1.0 else dq * power
    return Gradients(dq[:, :queries, :, :width].astype(query.dtype),
                     dk[:, :keys, :, :width].astype(key_dtype),
                     dv[:, :keys, :, :value_width].astype(value_dtype),
                     tuple(grad.astype(param.value.dtype) for grad, param in zip(grads, hooks.score_params,
                                                                                 strict=True)),
                     initial)
