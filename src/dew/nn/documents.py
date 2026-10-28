"""Packed documents in attention: which keys each query's document holds, as
the mask a dense kernel reads or the offsets of a fused kernel's packed
sequences. A call's own documents are one `[B, S]` array of segment ids,
another sequence's keys numbered apart from its queries a `CrossDocuments`.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp



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
    return _spanned(segment_ids, causal=False, sliding_window=None)


def with_documents(mask, segment_ids):
    """`mask` narrowed to each packed document's own keys, `[B, 1, Q, K]`."""
    inside = document_mask(segment_ids)[:, None]
    return inside if mask is None else jnp.logical_and(mask, inside)


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


def _padding_last(ids: jax.Array) -> jax.Array:
    """`ids` ordered with padding (0) after every document: each id less one as an unsigned word of its
    own width, so padding wraps to the largest and no two documents' ids meet."""
    ids = jnp.asarray(ids)
    word = jnp.dtype(f"uint{8 * ids.dtype.itemsize}")
    return ids.astype(word) - jnp.asarray(1, word)


def _searched(ordered: jax.Array, wanted: jax.Array, side: str) -> jax.Array:
    """Each of `wanted` `[B, N]`'s position in its row of `ordered` `[B, M]` (`jnp.searchsorted`)."""
    return jax.vmap(lambda row, ids: jnp.searchsorted(row, ids, side=side))(ordered, wanted)


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
    return _spanned(segment_ids, causal=causal, sliding_window=sliding_window)[:, None]


def _spanned(segment_ids, *, causal: bool, sliding_window: int | None) -> jax.Array:
    """The keys each query of a call of documents `segment_ids` sees, `[B, Q, K]`: the call's spans
    (`template.spans`) as a mask."""
    from dew.nn.kernels.templates import attention as template

    queries, keys = document_ids(segment_ids)
    return template.dense(template.spans(queries.shape[1], keys.shape[1], queries.shape[0], causal=causal,
                                         sliding_window=sliding_window, segment_ids=segment_ids))
