"""Which keys a query sees, by their coordinates: causality and a symmetric window (`visible`,
`windowed`, whose extent `window_sides` counts) and chunks (`same_chunk`), which Dew's attention masks
and banded layouts compare positions by. The coordinates are whatever the caller's layout counts,
slots, places in a document or banded rows, so a caller keeps its layout and builds no global mask to
ask. An asymmetric window (Gemma's audio blocks) states its own sides."""

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
    the keys at or before the query, and a window of w keeps those `windowed` keeps. The coordinates are whatever
    the caller's layout counts: slots, places in a document, banded rows."""
    shape = None if causal else jnp.broadcast_shapes(jnp.shape(query), jnp.shape(key))
    keep = (key <= query) if shape is None else jnp.ones(shape, bool)
    return keep if window is None else keep & windowed(query, key, window, causal=causal)


def same_chunk(query, key, chunk_size: int):
    """Whether a key falls in the query's chunk of `chunk_size` (broadcast), both counted from the
    sequence start: transformers' `chunked_overlay`, `kv // chunk == q // chunk`."""
    return query // chunk_size == key // chunk_size


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
