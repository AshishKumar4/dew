"""Which keys a query sees, by their coordinates: the one predicate every attention mask and banded
layout compares positions by (`visible`), with its window (`windowed`) and its chunks (`same_chunk`).
The coordinates are whatever the caller's layout counts, slots, places in a document or banded rows,
so a caller keeps its layout and builds no global mask to ask."""

import jax.numpy as jnp


def windowed(query, key, window: int, *, causal: bool):
    """Whether each key coordinate lies within the window of each query coordinate (broadcast): a
    causal window bounds the keys behind the query, keeping it and the w - 1 before it, and a two-sided
    one those within w - 1 of it on either side (`window_sides`). It bounds distance alone; causality
    is `visible`'s."""
    distance = query - key
    return (distance < window) if causal else (jnp.abs(distance) < window)


def visible(query, key, *, causal: bool, window: int | None = None):
    """Whether a key at coordinate `key` is visible to a query at `query` (broadcast), the one
    predicate every mask and banded layout compares coordinates by: causality keeps the keys at or
    before the query, and a window of w keeps those `windowed` keeps. The coordinates are whatever
    the caller's layout counts: slots, places in a document, banded rows."""
    shape = jnp.broadcast_shapes(jnp.shape(query), jnp.shape(key))
    keep = (key <= query) if causal else jnp.ones(shape, bool)
    return keep if window is None else keep & windowed(query, key, window, causal=causal)


def same_chunk(query, key, chunk_size: int):
    """Whether a key falls in the query's chunk of `chunk_size` (broadcast), both counted from the
    sequence start: transformers' `chunked_overlay`, `kv // chunk == q // chunk`."""
    return query // chunk_size == key // chunk_size
