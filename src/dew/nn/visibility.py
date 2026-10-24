"""Which keys a query sees, by their coordinates: causality and a symmetric window (`visible`,
`windowed`, whose extent `window_sides` counts), chunks (`same_chunk`) and blocks (`same_group`,
`block_starts`, `block_ends`), which Dew's attention masks and banded layouts compare positions by. The
coordinates are whatever the caller's layout counts, slots, places in a document or banded rows, so a
caller keeps its layout and builds no global mask to ask. An asymmetric window (Gemma's audio blocks)
states its own sides."""

import jax
import jax.numpy as jnp


def windowed(query, key, window: int, *, causal: bool):
    """Whether each key coordinate lies within the window of each query coordinate (broadcast): a
    causal window bounds the keys behind the query, keeping it and the w - 1 before it, and a two-sided
    one those within w - 1 of it on either side (`window_sides`). It bounds distance alone; causality
    is `visible`'s."""
    distance = query - key
    return (distance < window) if causal else (jnp.abs(distance) < window)


def visible(query, key, *, causal: bool, window: int | None = None):
    """Whether a key at coordinate `key` is visible to a query at `query` (broadcast): causality keeps
    the keys at or before the query, and a window of w keeps those `windowed` keeps. The coordinates
    are whatever the caller's layout counts: slots, places in a document, banded rows."""
    shapes = (jnp.shape(query), jnp.shape(key))
    keep = (key <= query) if causal else jnp.ones(jnp.broadcast_shapes(*shapes), bool)
    return keep if window is None else keep & windowed(query, key, window, causal=causal)


def same_chunk(query, key, chunk_size: int):
    """Whether a key falls in the query's chunk of `chunk_size` (broadcast), both counted from the
    sequence start: transformers' `chunked_overlay`, `kv // chunk == q // chunk`."""
    return query // chunk_size == key // chunk_size


def same_group(query_groups, key_groups):
    """Whether a key is in the query's block (broadcast): both carry one group id, -1 being none, as an
    image's tokens (`AttentionMetadata.image_groups`) or a prompt's blocks (`block_groups`) do."""
    return (query_groups == key_groups) & (query_groups >= 0)


def block_starts(groups):
    """Each token's block's first token, `[B, S]` int32 of `groups` `[B, S]`: where the run of equal ids
    that holds it begins (a block is such a run, -1 tokens runs of their own)."""
    groups = jnp.asarray(groups)
    batch, length = groups.shape
    index = jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32), (batch, length))
    opens = jnp.concatenate([jnp.ones((batch, 1), bool), groups[:, 1:] != groups[:, :-1]], axis=1)
    return jax.lax.cummax(jnp.where(opens, index, 0), axis=1)


def block_ends(groups):
    """Each token's block's end, `[B, S]` int32 of `groups` `[B, S]`: one past the last token of the run of
    equal ids that holds it (`block_starts`)."""
    groups = jnp.asarray(groups)
    batch, length = groups.shape
    index = jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32), (batch, length))
    # A run's last token: the next one's id differs, or the row ends there.
    last = jnp.concatenate([groups[:, 1:] != groups[:, :-1], jnp.ones((batch, 1), bool)], axis=1)
    return jax.lax.cummin(jnp.where(last, index, length), axis=1, reverse=True) + 1


def window_sides(causal: bool, sliding_window: int) -> tuple[int, int]:
    """The keys a window of `sliding_window` keeps before and after each query,
    counted as `jax.nn.dot_product_attention`'s `local_window_size` counts them.

    A causal window keeps the query and the w - 1 keys before it. A
    bidirectional one keeps the keys within w - 1 positions on either side,
    |q - k| < w: ModernBERT's |q - k| <= local_attention // 2, which is w =
    local_attention // 2 + 1 (masking_utils.py:141-158, transformers 5.16.1).
    A model layer passes one only where its kind asks for it
    (`LayerKind.bidirectional_window`).
    """
    return sliding_window - 1, 0 if causal else sliding_window - 1
