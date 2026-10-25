"""Which keys a query sees, by their coordinates: causality and a symmetric window (`visible`,
`windowed`, whose extent `window_sides` counts), chunks (`same_chunk`) and blocks (`same_group`), which
Dew's attention masks and banded layouts compare positions by. The coordinates are whatever the
caller's layout counts, slots, places in a document or banded rows, so a caller keeps its layout and
builds no global mask to ask. An asymmetric window (Gemma's audio blocks) states its own sides."""

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np


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


def prefix_blocks(segments, places, prefix: int, *, xp=jnp):
    """The block each token of a row is under a prefix run of `prefix` tokens: the first `prefix` places of
    each document, numbered by the document's segment id, -1 for the rest and for padding (segment 0). The
    `[B, S]` `segments` number a row's documents from 1, and `places` is each token's place in its document.
    The data stage `Blocks` and a served prompt (`prompt_blocks`) read this one rule."""
    return xp.where((segments > 0) & (places < prefix), segments, -1).astype(xp.int32)


def prompt_blocks(valid, prefix: int | Literal["prompt"], *, xp=jnp):
    """The blocks a served prompt's real tokens `valid` `[B, S]` make under a prefix run, as a training row's
    do (`prefix_blocks`): the first `prefix` tokens under an int, the whole prompt under "prompt". Each row is
    one document of its own. It reads no host value, so a trace makes the blocks the host does; a caller
    refuses a row with fewer than `prefix` real tokens on the count it holds, since the trained block would
    cover tokens the prompt lacks."""
    valid = xp.asarray(valid, bool)
    if prefix == "prompt":
        return xp.where(valid, 1, -1).astype(xp.int32)
    return prefix_blocks(valid.astype(xp.int32), xp.cumsum(valid, axis=1, dtype=xp.int32) - 1, prefix, xp=xp)


def served_blocks(real, prefix: int | Literal["prompt"]):
    """The blocks a server makes of the prompts whose real tokens are `real` `[B, S]` under a prefix run
    (`prompt_blocks`), counted on the host, which every serving path reads (`generate`'s request check and
    `Server`'s admission): a prompt shorter than an int `prefix` is refused, since its trained block would
    cover tokens it lacks."""
    real = np.asarray(real, bool)
    if prefix != "prompt" and (real.sum(axis=1) < prefix).any():
        raise ValueError(f"a prompt under prefix {prefix} needs {prefix} real tokens in every row")
    return prompt_blocks(real, prefix, xp=np)


def block_starts(groups):
    """Each token's block's first token, `[B, S]` int32 of `groups` `[B, S]`: where the run of equal ids
    that holds it begins (a block is such a run, -1 tokens runs of their own)."""
    groups = jnp.asarray(groups)
    batch, length = groups.shape
    index = jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32), (batch, length))
    opens = jnp.concatenate([jnp.ones((batch, 1), bool), groups[:, 1:] != groups[:, :-1]], axis=1)
    return jax.lax.cummax(jnp.where(opens, index, 0), axis=1)


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
