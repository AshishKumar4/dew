"""Which keys a query sees, by their coordinates: causality and a symmetric window (`visible`,
`windowed`, whose extent `window_sides` counts), chunks (`same_chunk`) and blocks (`same_group`,
`block_starts`, `block_ends`), which Dew's attention masks and banded layouts compare positions by. The
coordinates are whatever the caller's layout counts, slots, places in a document or banded rows, so a
caller keeps its layout and builds no global mask to ask. An asymmetric window (Gemma's audio blocks)
states its own sides."""

from typing import Literal, overload

import jax
import jax.numpy as jnp
import numpy as np

from dew.batch import BLOCKS_FIELD, coordinates, document_starts, prefix_max, prefix_min, prefix_sum


def windowed(query, key, window: int, *, causal: bool):
    """Whether each key coordinate lies within the window of each query coordinate (broadcast), the keys
    `window_sides` keeps before and after it: a causal window bounds the keys behind the query, keeping
    it and the w - 1 before it, and a two-sided one those within w - 1 of it on either side. It bounds
    distance alone; causality is `visible`'s. The scalar form of the template's spans' window
    (`dew.nn.kernels.templates.attention.spans`), for coordinates no span holds: a stated mask's places,
    which need not run in order, and a banded layout's rows."""
    before, after = window_sides(causal, window)
    # NumPy coordinates stay NumPy's, so a program that traces knows a static window as it traces.
    xp = jnp if any(isinstance(x, jax.Array) for x in (query, key)) else np
    query, key = xp.asarray(query), xp.asarray(key)
    ahead = query >= key
    within_before = _distance(query, key, xp) <= before
    if causal:
        return ~ahead | within_before
    return xp.where(ahead, within_before, _distance(key, query, xp) <= after)


def _distance(far, near, xp):
    """`far - near` where `far >= near`, exact for every integer pair: integers' difference taken in the
    unsigned word of their common width, which holds it where the signed one wraps."""
    common = xp.result_type(far, near)
    if not np.issubdtype(common, np.integer):
        return far - near
    unsigned = np.dtype(f"uint{np.dtype(common).itemsize * 8}")
    if xp is np:
        return far.astype(common).view(unsigned) - near.astype(common).view(unsigned)
    far, near = (jax.lax.bitcast_convert_type(x.astype(common), unsigned) for x in (far, near))
    return far - near


def visible(query, key, *, causal: bool, window: int | None = None):
    """Whether a key at coordinate `key` is visible to a query at `query` (broadcast): causality keeps
    the keys at or before the query, and a window of w keeps those `windowed` keeps. The coordinates
    are whatever the caller's layout counts: slots, places in a document, banded rows."""
    shapes = (jnp.shape(query), jnp.shape(key))
    keep = (key <= query) if causal else jnp.ones(jnp.broadcast_shapes(*shapes), bool)
    return keep if window is None else keep & windowed(query, key, window, causal=causal)


@overload
def chunk_of(places: jax.Array, chunk: int) -> jax.Array: ...


@overload
def chunk_of(places: np.ndarray | int, chunk: int) -> np.ndarray: ...


def chunk_of(places: jax.Array | np.ndarray | int, chunk: int) -> jax.Array | np.ndarray:
    """Each place's chunk number of `chunk` places, in the places' own dtype: `place // chunk`, a chunk wider
    than the dtype holds every place of one sign. The template's spans (`dew.nn.kernels.templates.attention
    .spans`) and `same_chunk` count chunks by it; NumPy places stay NumPy's, as `windowed`'s do."""
    xp = jnp if isinstance(places, jax.Array) else np
    places = xp.asarray(places)
    if chunk > xp.iinfo(places.dtype).max:
        return xp.where(places >= 0, 0, -1).astype(places.dtype)
    return places // chunk


def same_chunk(query, key, chunk_size: int):
    """Whether a key falls in the query's chunk of `chunk_size` (broadcast), both counted from the
    sequence start (`chunk_of`): transformers' `chunked_overlay`, `kv // chunk == q // chunk`. The scalar
    form of the template's spans' chunks, for coordinates no span holds."""
    return chunk_of(query, chunk_size) == chunk_of(key, chunk_size)


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


def role_blocks(segments, roles, assistant: int, *, starts=None, xp=jnp):
    """Each document's tokens before its first `assistant` token as one block, numbered by the document's
    segment id, -1 for the rest and for padding (segment 0): a chat row's prompt, which a prefix run of
    "prompt" reads whole. `segments` and `roles` are `[B, S]`. A document opens at each of `starts`, where a
    caller's documents run on across holes (`dew.batch.document_starts`), else wherever `segments`
    changes. The data stage `Blocks` and the loss of a batch without blocks (`LMObjective`) read this rule."""
    places = xp.broadcast_to(xp.arange(segments.shape[-1], dtype=xp.int32), segments.shape)
    opens = starts if starts is not None else xp.concatenate(
        [xp.ones_like(segments[..., :1], bool), segments[..., 1:] != segments[..., :-1]], axis=-1)
    last = segments.ndim - 1
    start = prefix_max(xp.where(opens, places, 0), last, xp=xp)
    spoken = prefix_max(xp.where((segments > 0) & (roles == assistant), places, -1), last, xp=xp) >= start
    return xp.where((segments > 0) & ~spoken, segments, -1).astype(xp.int32)


def prefix_groups(documents, real, prefix: int | Literal["prompt"], *, positions=None, roles=None,
                  assistant: int = -1, xp=jnp):
    """The blocks a prefix run of `prefix` reads in rows `[B, S]`: each document's first `prefix` places
    (`prefix_blocks`), or under "prompt" its tokens before its first `assistant` token by the rows'
    `roles` (`role_blocks`). `documents` number each row's documents and `real` marks its real tokens: a
    document runs on across a hole in its validity, its starts and its places, where no `positions` give
    them, counted over its real tokens as the decoder's coordinates count them (`document_starts`,
    `coordinates`). The data stage `Blocks` and the loss of a batch without blocks (`LMObjective`) read
    this one rule."""
    documents = xp.asarray(documents).astype(xp.int32)
    segments = xp.where(real, documents, 0).astype(xp.int32)
    if prefix == "prompt":
        starts = document_starts(documents, real, xp=xp)
        return role_blocks(segments, xp.asarray(roles), assistant, starts=starts, xp=xp)
    if positions is None:
        places = coordinates(segments.shape, segments=documents, valid=real, xp=xp).positions
    else:
        places = xp.broadcast_to(xp.asarray(positions), segments.shape)
    return prefix_blocks(segments, places, prefix, xp=xp)


def prompt_blocks(valid, prefix: int | Literal["prompt"], *, xp=jnp):
    """The blocks a served prompt's real tokens `valid` `[B, S]` make under a prefix run, as a training row's
    do (`prefix_blocks`): the first `prefix` tokens under an int, the whole prompt under "prompt". Each row is
    one document of its own. It reads no host value, so a trace makes the blocks the host does; a caller
    refuses a row with fewer than `prefix` real tokens on the count it holds, since the trained block would
    cover tokens the prompt lacks."""
    valid = xp.asarray(valid, bool)
    if prefix == "prompt":
        return xp.where(valid, 1, -1).astype(xp.int32)
    return prefix_blocks(valid.astype(xp.int32), prefix_sum(valid, dtype=np.int32, xp=xp) - 1, prefix, xp=xp)


def served_blocks(real, prefix: int | Literal["prompt"]):
    """The blocks a server makes of the prompts whose real tokens are `real` `[B, S]` under a prefix run
    (`prompt_blocks`), counted on the host, which every serving path reads (`generate`'s request check and
    `Server`'s admission): a prompt shorter than an int `prefix` is refused, since its trained block would
    cover tokens it lacks."""
    real = np.asarray(real, bool)
    if prefix != "prompt" and (real.sum(axis=1) < prefix).any():
        raise ValueError(f"a prompt under prefix {prefix} needs {prefix} real tokens in every row")
    return prompt_blocks(real, prefix, xp=np)


def with_served_blocks(fields: dict, real, prefix: int | Literal["prompt"] | None) -> dict:
    """`fields` with the blocks a prefix run makes of the prompts whose real tokens are `real`
    (`served_blocks`) where they carry none: a request's own blocks stand, as a batch's do in training.
    Every serving path takes its blocks here (`generate`, `Server`)."""
    if prefix is None or BLOCKS_FIELD in fields:
        return fields
    return {**fields, BLOCKS_FIELD: served_blocks(real, prefix)}


def block_starts(groups):
    """Each token's block's first token, `[B, S]` int32 of `groups` `[B, S]`: where the run of equal ids
    that holds it begins (a block is such a run, -1 tokens runs of their own)."""
    groups = jnp.asarray(groups)
    batch, length = groups.shape
    index = jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32), (batch, length))
    opens = jnp.concatenate([jnp.ones((batch, 1), bool), groups[:, 1:] != groups[:, :-1]], axis=1)
    return prefix_max(jnp.where(opens, index, 0))


def block_ends(groups):
    """Each token's block's end, `[B, S]` int32 of `groups` `[B, S]`: one past the last token of the run of
    equal ids that holds it (`block_starts`)."""
    groups = jnp.asarray(groups)
    batch, length = groups.shape
    index = jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32), (batch, length))
    # A run's last token: the next one's id differs, or the row ends there.
    last = jnp.concatenate([groups[:, 1:] != groups[:, :-1], jnp.ones((batch, 1), bool)], axis=1)
    return prefix_min(jnp.where(last, index, length), reverse=True) + 1


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
