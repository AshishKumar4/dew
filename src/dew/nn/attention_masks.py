"""The masks and biases an attention call reads, from which keys each query sees (`dew.nn.visibility`):
causality, a window, chunks, key lengths and starts (`visibility_mask`, `key_spans`), ALiBi's bias,
and grouped key heads repeated to the query's (`repeat_kv_heads`).
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

from .documents import CrossDocuments, document_ids
from .kernels.templates import attention as template
from .precision import at_least_fp32, float32_einsum, operand_precision
from .visibility import same_chunk, windowed


def repeat_kv_heads(x, num_heads: int):
    """Repeat grouped key/value heads out to the query heads: [B, S, K, D] -> [B, S, N, D].

    Query head n reads key/value head n // (N // K), the grouping
    jax.nn.dot_product_attention uses internally, so the same param tree runs
    on the kernels that group heads themselves and on the ones that need the
    keys materialized.
    """
    kv_heads = x.shape[-2]
    if kv_heads == num_heads:
        return x
    if num_heads % kv_heads:
        raise ValueError(
            f"grouped-query attention needs the query heads ({num_heads}) to be "
            f"a multiple of the key/value heads ({kv_heads}).")
    return jnp.repeat(x, num_heads // kv_heads, axis=-2)


def causal_attention_mask(query_positions, kv_len: int, sliding_window=None, *, key_valid=None):
    """Boolean [B, 1, T, S] mask from shared [T] or per-row [B, T] slots.

    A negative query slot denotes an invalid token. Cache validity excludes
    unfilled slots; a window retains the query and its preceding W-1 keys.
    """
    positions = jnp.atleast_2d(jnp.asarray(query_positions, jnp.int32))
    seen = visibility_mask(positions.shape[1], kv_len, causal=True, sliding_window=sliding_window,
                           query_slots=positions)
    # The filled slots ANDed after the spans, which then search no keys' ids: a batch-sharded cache's
    # validity takes no gather under explicit sharding (test_parallelism_matrix's data4_explicit).
    return seen if key_valid is None else seen & jnp.asarray(key_valid, bool)[:, None, None, :]


def alibi_bias(key_positions, num_heads: int, *, dtype) -> jax.Array:
    """BLOOM/Falcon's per-head linear key-position bias, `[B, H, 1, K]`.

    The slopes follow transformers' `build_alibi_tensor`, including the odd
    powers appended for a non-power-of-two head count. Dropping the query's
    position subtracts one constant from every visible logit of that query,
    which leaves softmax unchanged.
    """
    power = 2 ** math.floor(math.log2(num_heads))
    base = np.float32(2 ** (-(2 ** -(math.log2(power) - 3))))
    slopes = np.power(base, np.arange(1, power + 1, dtype=np.float32))
    if power != num_heads:
        extra = np.float32(2 ** (-(2 ** -(math.log2(2 * power) - 3))))
        slopes = np.concatenate((slopes, np.power(
            extra, np.arange(1, 2 * (num_heads - power), 2, dtype=np.float32))))
    positions = jnp.atleast_2d(key_positions)
    return (jnp.asarray(slopes)[None, :, None, None]
            * positions[:, None, None, :]).astype(dtype)


def structural_mask(query_positions, kv_len: int, causal: bool,
                    sliding_window: int | None) -> jax.Array | None:
    """The `[B, 1, T, S]` keys that causality and a window leave each query,
    from shared `[T]` or per-row `[B, T]` positions, or None when neither
    applies. A negative query position sees nothing."""
    if not causal and sliding_window is None:
        return None
    positions = jnp.atleast_2d(jnp.asarray(query_positions, jnp.int32))
    return visibility_mask(positions.shape[1], kv_len, causal, sliding_window, query_slots=positions)


def chunk_mask(query_positions, key_positions, chunk_size: int):
    """Boolean `[.., 1, T, S]` keeping keys in the query's chunk, both absolute.

    transformers' `chunked_overlay`: `kv // chunk == q // chunk`, positions
    counted from the sequence start, so a packed document's positions place
    its chunks from its own first token.
    """
    same = same_chunk(jnp.asarray(query_positions)[..., :, None], jnp.asarray(key_positions)[..., None, :],
                      chunk_size)
    return same[..., None, :, :] if same.ndim == 3 else same[None, None]


def combined_attention_mask(query_length: int, key_length: int, causal: bool,
                            sliding_window: int | None,
                            mask: jax.Array | None) -> jax.Array | None:
    """Fold causality and a sliding window into `mask`.

    A causal flag keeps the keys at or before each query's row; a window
    narrows that to the most recent keys, or to the nearest on either side
    without the flag (`window_sides`). Both read the row index, the way the
    fused kernels take them as flags. Unset stays unset, so a caller that
    distinguishes no mask from an all-true one keeps doing so.
    """
    if not causal and sliding_window is None:
        return mask
    return visibility_mask(query_length, key_length, causal, sliding_window, mask=mask)


def key_spans(query_length: int, key_length: int, causal: bool, sliding_window: int | None,
              segment_ids=None, key_value_seq_lengths=None, key_value_seq_starts=None, *, block_groups=None,
              image_groups=None, key_valid=None, key_runs=None, places=None, chunk: int | None = None,
              query_slots=None, rows: int | None = None) -> template.KeySpans:
    """The keys each query of a call sees as the template's spans (`template.spans`), from the
    coordinates `scaled_dot_product_attention` takes: causality, a window, a chunk of `chunk` places
    (`chunk_mask`), documents (`CrossDocuments` aligned to their last keys), key lengths and starts, block
    and image groups, and the keys' validity `key_valid` `[B, K]`, and each query's key runs `key_runs`
    `[B, Q, R, 2]`, whose union it may see (`template.spans`'s `runs`). `places` `[B, S]` are a call's own
    tokens' places a window or a chunk counts by (`counted_places`), by default their indices, or a pair,
    its queries' `[B, Q]` and its keys' `[B, K]`. `query_slots` `[B, Q]` are the queries' own keys in a
    cache call, -1 for padding (`template.spans`' `slots`). `rows` is the call's batch, by default the
    coordinates' own, or one where none carries it, which every row shares."""
    pair = places if isinstance(places, tuple) else (places, places)
    if rows is None:
        # The batch the coordinates broadcast to: a row's lengths and starts `[B]`, every other one's
        # leading axis where it has a row axis.
        rowed = [None if segment_ids is None else document_ids(segment_ids)[0], block_groups, image_groups,
                 key_valid, key_runs, pair[0], query_slots]
        batches = [(jnp.shape(x)[0],) for x in (key_value_seq_lengths, key_value_seq_starts) if x is not None]
        batches += [(jnp.shape(x)[0],) for x in rowed if x is not None and jnp.ndim(x) >= 2]
        rows = int(np.broadcast_shapes(*batches, (1,))[0])
    lengths, starts = (None if x is None else jnp.asarray(x, jnp.int32)
                       for x in (key_value_seq_lengths, key_value_seq_starts))
    positions = None
    if pair[0] is not None and pair[1] is not None:
        positions = (jnp.broadcast_to(jnp.atleast_2d(pair[0]), (rows, query_length)),
                     jnp.broadcast_to(jnp.atleast_2d(pair[1]), (rows, key_length)))
    valid = None if key_valid is None else jnp.broadcast_to(key_valid, (rows, key_length))
    slots = None if query_slots is None else jnp.broadcast_to(query_slots, (rows, query_length))
    return template.spans(query_length, key_length, rows, causal=causal, sliding_window=sliding_window,
                          chunk=chunk, positions=positions, segment_ids=segment_ids, lengths=lengths,
                          starts=starts, block_groups=block_groups, image_groups=image_groups, valid=valid,
                          runs=key_runs, slots=slots)


def visibility_mask(query_length: int, key_length: int, causal: bool, sliding_window: int | None, mask=None,
                    segment_ids=None, key_value_seq_lengths=None, key_value_seq_starts=None, *,
                    block_groups=None, image_groups=None, key_valid=None, key_runs=None,
                    places=None, chunk: int | None = None, query_slots=None) -> jax.Array:
    """The keys each query sees, `[B or 1, 1, Q, K]`: the call's spans (`key_spans`) as a mask
    (`template.dense`), narrowed by `mask`. It is the mask a call's coordinates mean on the paths
    whose kernel does not read them all."""
    seen = template.dense(key_spans(query_length, key_length, causal, sliding_window, segment_ids,
                                    key_value_seq_lengths, key_value_seq_starts, block_groups=block_groups,
                                    image_groups=image_groups, key_valid=key_valid, key_runs=key_runs,
                                    places=places, chunk=chunk, query_slots=query_slots))[:, None]
    return seen if mask is None else jnp.logical_and(seen, mask)


def visible_rows(query_length: int, key_length: int, causal: bool, sliding_window: int | None,
                 mask=None, segment_ids=None, key_value_seq_lengths=None,
                 key_value_seq_starts=None, *, block_groups=None, image_groups=None, key_valid=None,
                 key_runs=None, places=None) -> jax.Array | None:
    """Whether each query row sees a key, `[B or 1, Q, H or 1, 1]` over the
    output, or None where every row does.

    A row that sees no key reads nothing, so attention gives it exactly zero
    on every implementation (`scaled_dot_product_attention`): a softmax over
    no keys has no value, and a kernel left to it returns the mean of the
    values or, in Splash's backward, gradients from a uniform weight over
    masked keys. The call's spans decide a row (`template.reads`); an
    explicit mask is reduced over its keys, folded with them
    (`visibility_mask`). Without a mask, key bounds, key validity, counted
    places or another sequence's documents, a row sees its own key where the keys reach it,
    which the program knows as it traces, unless it is a packed row's padding.
    """
    if (mask is None and key_value_seq_lengths is None and key_value_seq_starts is None and key_valid is None
            and key_runs is None and places is None and not isinstance(segment_ids, CrossDocuments)):
        # The keys row i may see run from 0 to the last one; causality keeps 0 in reach, and a
        # window ends w - 1 keys before the row (`windowed`). A call's own documents keep each row's
        # own key, and block and image groups only add keys beside it.
        rows = np.arange(query_length)[None]
        last = np.full((1, 1), key_length - 1)
        reach = (last >= 0) if sliding_window is None else windowed(rows, last, sliding_window, causal=True)
        if segment_ids is None:
            return None if reach.all() else jnp.asarray(reach)[:, :, None, None]
        return (jnp.asarray(reach) & (jnp.asarray(segment_ids) != 0))[:, :, None, None]
    if mask is None:
        spans = key_spans(query_length, key_length, causal, sliding_window, segment_ids,
                          key_value_seq_lengths, key_value_seq_starts, block_groups=block_groups,
                          image_groups=image_groups, key_valid=key_valid, key_runs=key_runs, places=places)
        return template.reads(spans)[:, :, None, None]
    seen = jnp.any(visibility_mask(query_length, key_length, causal, sliding_window, mask, segment_ids,
                                   key_value_seq_lengths, key_value_seq_starts, block_groups=block_groups,
                                   image_groups=image_groups, key_valid=key_valid, key_runs=key_runs,
                                   places=places), -1)
    return jnp.swapaxes(seen, -1, -2)[..., None]


def with_key_lengths(mask: jax.Array | None, lengths: jax.Array, key_length: int) -> jax.Array:
    """`mask` narrowed to each row's first `lengths[b]` keys, `[B, 1, 1, K]`
    where there was no mask. This is the mask a `key_value_seq_lengths`
    call means on the paths whose kernel takes no lengths."""
    return visibility_mask(1, key_length, causal=False, sliding_window=None, mask=mask,
                           key_value_seq_lengths=lengths)


def with_key_starts(mask: jax.Array | None, starts: jax.Array | None, key_length: int) -> jax.Array | None:
    """`mask` narrowed to each row's keys from `starts[b]`, `[B, 1, 1, K]` where there was no mask:
    the mask a `key_value_seq_starts` call means on the paths whose kernel takes no starts. None
    starts leave `mask` as it is."""
    if starts is None:
        return mask
    return visibility_mask(1, key_length, causal=False, sliding_window=None, mask=mask,
                           key_value_seq_starts=starts)


def max_attention_logits(query: jax.Array, key: jax.Array, *, causal: bool = False,
                         sliding_window: int | None = None,
                         mask: jax.Array | None = None,
                         bias: jax.Array | None = None) -> jax.Array:
    """Return the largest pre-softmax logit per query head: `[batch, heads]`, at least fp32
    (`at_least_fp32` of the operands'), so a float64 model's maxima are float64.

    The logits are the scaled dot products the kernels softmax. Masked
    positions read -inf, so causality and packing never trip the maximum.
    Grouped key heads repeat out to the query heads first, the grouping the
    kernels run. A logit softcap and attention sinks stay out: the clip that
    reads this bounds the raw query-key growth, and a softcapped model bounds
    it already.
    """
    heads = query.shape[-2]
    key = repeat_kv_heads(key, heads)
    scale = 1.0 / math.sqrt(query.shape[-1])
    wide = at_least_fp32(jnp.result_type(query.dtype, key.dtype))
    logits = float32_einsum('...qhd,...khd->...hqk',
                            query.astype(wide) * jnp.asarray(scale, wide),
                            key.astype(wide), precision=operand_precision(wide))
    if bias is not None:
        logits = logits + bias.astype(wide)
    combined = combined_attention_mask(
        query.shape[-3], key.shape[-3], causal, sliding_window, mask)
    if combined is not None:
        logits = jnp.where(combined, logits, -jnp.inf)
    return jnp.max(logits, axis=(-2, -1))
