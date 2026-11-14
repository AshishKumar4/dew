"""Packed documents in attention: which keys each query's document holds, as
the mask a dense kernel reads or the offsets of a fused kernel's packed
sequences. A call's own documents are one `[B, S]` array of segment ids,
another sequence's keys numbered apart from its queries a `CrossDocuments`.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from dew.batch import document_firsts

from .scatter import gather_positions
from .visibility import visible


class CrossDocuments(NamedTuple):
    """The packed documents of a call whose keys are not its queries
    (cross-attention): `query` `[B, Q]` and `key` `[B, K]` ids, each query
    reading the keys of its own document, 0 as padding, which reads nothing
    and is read by nothing; ids are nonnegative. FlashAttention runs each
    row's documents as its packed sequences, which needs the keys in
    ascending id order, padding after them; the queries it orders itself.

    Under causality or a window, a document's queries are its last keys, in
    order: the last query reads up to the document's last key, the one before
    it one key less (FlashAttention's alignment of a sequence's queries to the
    bottom right of its keys). A cache call is one: its tokens were appended
    after the keys its row held (`cross_document_mask`)."""

    query: jax.Array
    key: jax.Array


def document_ids(segment_ids) -> tuple[jax.Array, jax.Array]:
    """The query's and the key's ids of `segment_ids`, one `[B, S]` array of a
    call's own documents or a `CrossDocuments`, the two in one dtype (their
    promotion), so an id names one document on both sides however it is
    ordered."""
    if isinstance(segment_ids, CrossDocuments):
        queries, keys = jnp.asarray(segment_ids.query), jnp.asarray(segment_ids.key)
        common = jnp.promote_types(queries.dtype, keys.dtype)
        return queries.astype(common), keys.astype(common)
    return jnp.asarray(segment_ids), jnp.asarray(segment_ids)


def document_mask(segment_ids) -> jax.Array:
    """Keep each packed document to itself: `[B, Q, K]` boolean.

    A query sees a key when they carry the same segment id (`document_ids`).
    Segment 0 is padding, which sees nothing and is seen by nothing. This is
    the mask every kernel but splash and the pallas flash kernel reads for
    `segment_ids`; those two compare the ids per block.
    """
    return _spanned(segment_ids)


def with_documents(mask, segment_ids):
    """`mask` narrowed to each packed document's own keys, `[B, 1, Q, K]`."""
    inside = document_mask(segment_ids)[:, None]
    return inside if mask is None else jnp.logical_and(mask, inside)


def within_documents(values: jax.Array, begins: jax.Array) -> jax.Array:
    """Each token's sum of `values` `[B, S, ...]` over its document up to it, `begins` `[B, S]` marking each
    document's first token: a sum that restarts at each, one associative scan. A later document's sums never
    pass through an earlier one's, so a difference of two within a document keeps the document's own digits
    (a running sum across documents took them from the row's prefix)."""
    def combined(first: tuple, second: tuple) -> tuple:
        (opened, summed), (opens, adds) = first, second
        restarts = opens.reshape(*opens.shape, *(1,) * (adds.ndim - opens.ndim))
        return opened | opens, jnp.where(restarts, adds, summed + adds)

    return jax.lax.associative_scan(combined, (begins, values), axis=1)[1]


def document_offsets(segment_ids, max_segments: int) -> jax.Array:
    """FlashAttention's cumulative offsets, `[B * (max_segments + 1) + 1]`, of
    packed rows `segment_ids` `[B, S]` whose documents run in ascending id
    order with padding (0) after them: each row's documents 1 to
    `max_segments` in turn, then its padding, a document the row lacks an
    empty sequence. The queries and the keys of `CrossDocuments` so give one
    sequence each for every document, in the same order."""
    ids = jnp.arange(1, max_segments + 2) % (max_segments + 1)
    counts = jnp.sum(segment_ids[:, None, :] == ids[None, :, None], axis=-1, dtype=jnp.int32)
    return jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(counts.reshape(-1), dtype=jnp.int32)])


def own_runs(segment_ids: jax.Array, runs: int) -> tuple[jax.Array, jax.Array]:
    """Each row's runs of one id in a call's own `segment_ids` `[B, S]`, its documents and the padding after
    them, whatever their ids: whether each token begins one `[B, S]`, and each run's first token `[B, runs]`
    (`document_firsts`: `S` past a row's last), at most `runs` a row. The packed sequences FlashAttention and
    cuDNN run a call's own documents as (`attention.flash_attention`, `packed_runs`)."""
    rows = segment_ids.shape[0]
    changes = segment_ids[:, 1:] != segment_ids[:, :-1]
    begins = jnp.concatenate([jnp.ones((rows, 1), bool), changes], axis=1)
    return begins, document_firsts(begins, runs)


def packed_runs(segment_ids: jax.Array, runs: int) -> tuple[jax.Array, jax.Array]:
    """A call's own documents' runs (`own_runs`) as cuDNN's packed sequences take them
    (`fused_attention_stablehlo.dot_product_attention`'s `q_offsets` and `q_seqlen`): each run's first token
    `[B, runs + 1]` and its length `[B, runs]`, in order, -1 past a row's last."""
    length = segment_ids.shape[1]
    firsts = own_runs(segment_ids, runs)[1]
    ends = jnp.concatenate([firsts[:, 1:], jnp.full((firsts.shape[0], 1), length, firsts.dtype)], axis=1)
    held = firsts < length
    offsets = jnp.pad(jnp.where(held, firsts, -1), ((0, 0), (0, 1)), constant_values=-1)
    return offsets, jnp.where(held, ends - firsts, -1)


def _padding_last(ids: jax.Array) -> jax.Array:
    """`ids` ordered with padding (0) after every document: each id less one as an unsigned word of its
    own width, so padding wraps to the largest and no two documents' ids meet."""
    ids = jnp.asarray(ids)
    word = jnp.dtype(f"uint{8 * ids.dtype.itemsize}")
    return ids.astype(word) - jnp.asarray(1, word)


def _searched(ordered: jax.Array, wanted: jax.Array, side: str) -> jax.Array:
    """Each of `wanted` `[B, N]`'s position in its row of `ordered` `[B, M]` (`jnp.searchsorted`), its binary
    search unrolled into one program: the default's loop ran each of its log2(M) trips as a launch of its
    own, a GPU's loop iteration each (M-DD-10, W63-gh200: a speculative verify of 4 queries over 4096 keys
    284 us against its 22 us attention)."""
    return jax.vmap(lambda row, ids: jnp.searchsorted(row, ids, side=side, method="scan_unrolled"))(ordered,
                                                                                                    wanted)


def query_order(ids: jax.Array) -> jax.Array:
    """The stable permutation `[B, Q]` that puts each row's queries `ids` in document order, padding
    last: the order FlashAttention's packed sequences take them in, and the one their ranks come
    from (`aligned_places`)."""
    return jnp.argsort(_padding_last(ids), axis=-1, stable=True)


def held_keys(segment_ids: CrossDocuments) -> jax.Array:
    """How many keys each query's document holds in its row, `[B, Q]`, searched in the keys'
    ascending order (`CrossDocuments`), which needs no sort."""
    queries, keys = document_ids(segment_ids)
    ordered, wanted = _padding_last(keys), _padding_last(queries)
    return _searched(ordered, wanted, "right") - _searched(ordered, wanted, "left")


def holds_keys(segment_ids: CrossDocuments) -> jax.Array:
    """Whether each query's document holds any key in its row, `[B, Q]`, whatever order the keys stand
    in: the keys sorted first (`held_keys` reads them in ascending order), a padding query (0) holding
    none."""
    queries, keys = document_ids(segment_ids)
    ordered, wanted = jnp.sort(_padding_last(keys), axis=-1), _padding_last(queries)
    return (queries != 0) & (_searched(ordered, wanted, "right") > _searched(ordered, wanted, "left"))


def aligned_places(segment_ids: CrossDocuments) -> jax.Array:
    """Each query's place among its document's keys under causality or a window, `[B, Q]`: a document's
    query of rank r among its Q queries sits at rank K - Q + r among its K keys (`CrossDocuments`)."""
    queries, _ = document_ids(segment_ids)
    order = query_order(queries)
    ordered = jnp.take_along_axis(_padding_last(queries), order, axis=-1)
    first = _searched(ordered, ordered, "left")
    count = _searched(ordered, ordered, "right") - first
    rows = jnp.arange(queries.shape[0])[:, None]
    # A query's rank among its document's queries and their count, put back in the queries' order.
    placed = jnp.zeros_like(order).at[rows, order].set(jnp.arange(queries.shape[-1])[None, :] - first - count)
    return held_keys(segment_ids) + placed


def cross_document_mask(segment_ids: CrossDocuments, *, causal: bool,
                        sliding_window: int | None) -> jax.Array:
    """`[B, 1, Q, K]`: each query's document's keys (`document_mask`), under causality and a window
    aligned to the document's last key (`aligned_places`); causality keeps the keys at or before a
    query's place, a window of w the w - 1 before it, or without causality those within w - 1 of it
    (`dew.nn.visibility.window_sides`). A key's rank in its document is its distance from the
    document's first key, the keys running in ascending id order."""
    # Elementwise over the keys' ranks, not the spans' sorted search, which takes no batch-sharded ids
    # under explicit sharding (test_parallelism_matrix's data4_explicit).
    _, keys = document_ids(segment_ids)
    ordered = _padding_last(keys)
    key_rank = jnp.arange(keys.shape[-1])[None, :] - _searched(ordered, ordered, "left")
    keep = document_mask(segment_ids) & visible(aligned_places(segment_ids)[:, :, None], key_rank[:, None, :],
                                                 causal=causal, window=sliding_window)
    return keep[:, None]


def _spanned(segment_ids) -> jax.Array:
    """The keys each query of a call of documents `segment_ids` sees by its documents alone, `[B, Q, K]`:
    every key in reach, the ids keeping each query to its own, padding none, as spans (`template.dense`)."""
    from dew.nn.kernels.templates import attention as template

    queries, keys = document_ids(segment_ids)
    end = jnp.where(queries == 0, 0, keys.shape[1]).astype(jnp.int32)
    return template.dense(template.KeySpans(jnp.zeros_like(end), end, queries, keys, keys.shape[1], "spans"))


class DocumentWindows(NamedTuple):
    """A packed row's compressed windows document by document (`document_windows`): each document's
    complete windows of `rate` of its real tokens from its first, as the document alone pools them,
    one entry after another, `S // rate` entries a row whatever its documents hold."""

    sources: jax.Array
    """`[B, T, rate]`: the row's index of each entry's tokens."""
    first: jax.Array
    """`[B, T]`: the entry is its document's first window, which overlaps no earlier one."""
    live: jax.Array
    """`[B, T]`: the entry is a document's complete window, not a slot past the row's last."""
    document: jax.Array
    """`[B, T]`: the entry's document."""
    window: jax.Array
    """`[B, T]`: the entry's index among its document's windows."""
    query_document: jax.Array
    """`[B, S]`: each token's document."""
    query_rank: jax.Array
    """`[B, S]`: each token's rank among its document's real tokens."""

    def gathered(self, x):
        """`x` `[B, S, F]` at each entry's tokens, `[B, T, rate, F]`."""
        batch, entries, rate = self.sources.shape
        return gather_positions(x, self.sources.reshape(batch, entries * rate)).reshape(
            batch, entries, rate, x.shape[-1])

    def origins(self, places):
        """Each entry's rotary origin `[B, T]`: its window's first token's place (`Coordinates.places`),
        where its document's queries that read it rotate."""
        held = jnp.broadcast_to(jnp.asarray(places), self.query_rank.shape)
        return jnp.take_along_axis(held, self.sources[:, :, 0], axis=1)

    def visible(self, rate: int):
        """Which entries each query may attend, `[B, S, T]`: its own document's whose window closed at
        or before it, by its rank there, as `entries_visible` reads a document alone."""
        closed = self.window[:, None, :] < (self.query_rank[:, :, None] + 1) // rate
        return self.live[:, None, :] & (self.document[:, None, :] == self.query_document[:, :, None]) & closed


def document_windows(segments, valid, rate: int) -> DocumentWindows:
    """The compressed windows of a row of `segments` `[B, S]`, its real tokens where `valid` marks them,
    each document's pooled from its own first real token on (`DocumentWindows`): the row's real tokens
    in document order, each document's in its own, each window opening at a multiple of `rate` among
    its document's and closed by a real token of it `rate - 1` later; a document's partial last window
    pools nothing, as the stateless path drops a row's."""
    segments = jnp.asarray(segments)
    batch, length = segments.shape
    real = jnp.ones(segments.shape, bool) if valid is None else jnp.asarray(valid, bool)
    held = jnp.where(real, segments, jnp.iinfo(segments.dtype).max)
    order = jnp.argsort(held, axis=1, stable=True)
    held, kept = (jnp.take_along_axis(field, order, axis=1) for field in (held, real))
    slots = jnp.arange(length)
    previous = jnp.concatenate([jnp.full((batch, 1), -1, held.dtype), held[:, :-1]], axis=1)
    rank = slots - jax.lax.cummax(jnp.where(kept & (held != previous), slots, 0), axis=1)
    last = jnp.minimum(slots + rate - 1, length - 1)
    closes = (slots + rate - 1 < length) & kept[:, last] & (held[:, last] == held)
    opens = kept & (rank % rate == 0) & closes
    # The row's openings in order, an entry each; slots past its last are no window (`live`).
    picked = jnp.argsort(~opens, axis=1, stable=True)[:, :length // rate]
    spans = jnp.minimum(picked[:, :, None] + jnp.arange(rate), length - 1)
    sources = jnp.take_along_axis(order, spans.reshape(batch, -1), axis=1).reshape(spans.shape)
    opened = jnp.take_along_axis(rank, picked, axis=1)
    # Each token's rank back at its place in the row.
    ranks = jnp.zeros_like(rank).at[jnp.arange(batch)[:, None], order].set(rank)
    return DocumentWindows(sources, opened == 0, jnp.take_along_axis(opens, picked, axis=1),
                           jnp.take_along_axis(held, picked, axis=1), opened // rate, segments, ranks)


def document_means(values: jax.Array, segment_ids, valid) -> jax.Array:
    """Each token's mean of `values` `[B, S, ...]` over its document's real tokens (`segment_ids`, those
    `valid` marks), at least float32 wide: what a query that reads every key of its document alike reads,
    as the document alone reads its row's."""
    batch, length = values.shape[:2]
    real = jnp.asarray(valid, bool)
    held = jnp.where(real, jnp.asarray(segment_ids), jnp.iinfo(jnp.int32).max)
    order = jnp.argsort(held, axis=1, stable=True)
    held, kept = (jnp.take_along_axis(field, order, axis=1) for field in (held, real))
    widened = values.astype(jnp.promote_types(values.dtype, jnp.float32))
    ordered = jnp.take_along_axis(widened, order.reshape(batch, length, *(1,) * (values.ndim - 2)), axis=1)
    weight = kept.reshape(batch, length, *(1,) * (values.ndim - 2))
    place = jnp.arange(length)
    begins = (place == 0) | (held != jnp.roll(held, 1, axis=1))
    sums = within_documents(jnp.where(weight, ordered, 0), begins)
    counts = within_documents(kept.astype(widened.dtype), begins)
    # Each document's whole sum, at its last token, read back at each of its tokens.
    ends = jnp.roll(begins, -1, axis=1).at[:, -1].set(True)
    last = jax.lax.cummin(jnp.where(ends, place, length - 1), axis=1, reverse=True)
    whole = jnp.take_along_axis(sums, last.reshape(batch, length, *(1,) * (values.ndim - 2)), axis=1)
    count = jnp.take_along_axis(counts, last, axis=1).reshape(weight.shape)
    means = whole / jnp.maximum(count, 1)
    return jnp.zeros_like(means).at[jnp.arange(batch)[:, None], order].set(means)
