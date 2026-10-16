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
    and is read by nothing. FlashAttention runs each row's documents as its
    packed sequences, which needs them in ascending id order in the queries
    and in the keys alike, padding after them."""

    query: jax.Array
    key: jax.Array


def document_ids(segment_ids) -> tuple[jax.Array, jax.Array]:
    """The query's and the key's ids of `segment_ids`, one `[B, S]` array of a
    call's own documents or a `CrossDocuments`."""
    if isinstance(segment_ids, CrossDocuments):
        return jnp.asarray(segment_ids.query), jnp.asarray(segment_ids.key)
    return jnp.asarray(segment_ids), jnp.asarray(segment_ids)


def document_mask(segment_ids) -> jax.Array:
    """Keep each packed document to itself: `[B, Q, K]` boolean.

    A query sees a key when they carry the same segment id (`document_ids`).
    Segment 0 is padding, which sees nothing and is seen by nothing. This is
    the mask every kernel but splash and the pallas flash kernel reads for
    `segment_ids`; those two compare the ids per block.
    """
    queries, keys = document_ids(segment_ids)
    return (queries[:, :, None] == keys[:, None, :]) & (queries[:, :, None] != 0)


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
