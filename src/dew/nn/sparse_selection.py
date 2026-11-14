"""DeepSeek sparse attention's selection and its execution in the latent space.

A selection is `[B, S, K]` key indices per query, -1 in a slot that names
no key, which is how GLM's reference builds its top-k mask
(`build_attention_mask_from_topk`, modeling_glm5_next.py:1218-1256). The
V3.2 indexer (`dew.nn.mla`), GLM's k-pool indexer (`dew.nn.dsa_kpool`) and
DeepSeek V4's compressed-entry indexer (`dew.nn.deepseek_v4`) all score
their keys through `index_scores` and select through `top_k_selection`,
`selection_mask` turns it into the `[B, S, T]` mask the dense kernels
read, and `sparse_latent_attention` attends it without that mask.
"""

import functools

import jax
import jax.numpy as jnp

from dew.nn.attention import AttentionImpl, scaled_dot_product_attention
from dew.nn.inputs import sequence_blocks
from dew.nn.precision import accumulated, at_least_fp32, float32_einsum
from dew.nn.scatter import DROPPED


def index_agreements(query, keys, precision=None):
    """Each lightning indexer head's agreement of each query with each key, `q_h . k` before its relu,
    at least in fp32 (`at_least_fp32`): queries `[B, S, H, D]` against keys `[B, T, D]` every query
    shares or `[B, S, T, D]` per query, as `[B, S, H, T]`, under `precision`. A key every head
    disagrees with scores zero (`index_scores`), however near zero its agreements sit."""
    wide = at_least_fp32(query.dtype)
    shared = 'bshd,btd->bsht' if keys.ndim == 3 else 'bshd,bstd->bsht'
    return float32_einsum(shared, query.astype(wide), keys.astype(wide), precision=precision)


def index_scores(agreements, weights, width: int, precision=None):
    """The lightning indexer's score of each query against each key,
    `sum_h w_h relu(q_h . k) / sqrt(head_dim) / sqrt(n_heads)`, at least in
    fp32, from its heads' `agreements` `[B, S, H, T]` (`index_agreements`) of
    queries `width` wide, weighted by `weights` `[B, S, H]`, as `[B, S, T]`;
    the head sum takes `precision`. The V3.2 indexer scores its keys, GLM's
    k-pool its pooled keys and DeepSeek V4 its compressed entries through it."""
    heads, wide = agreements.shape[-2], agreements.dtype
    scores = jnp.maximum(agreements, 0) * width ** -0.5
    # The head sum as a [1, H] by [H, T] product per query: written as the
    # einsum over the scores first, it ran a third slower forward and backward
    # on an H100 at the V3.2 indexer's 64 heads over 4096 keys.
    weighted = (weights.astype(wide) * heads ** -0.5)[..., None, :]
    return accumulated(functools.partial(jnp.matmul, precision=precision),
                       weighted, scores).squeeze(-2)


def top_k_selection(scores, keep, top_k: int, *, forced=None):
    """The `top_k` highest-scoring keys of each query among those `keep`
    allows: `[..., K]` indices with `K = min(top_k, T)`, -1 in a slot no
    allowed key fills, then each of `forced` `[..., F]` (-1 for none), taken
    whatever its score and `keep`, which the top k passes over.

    Exactly `top_k` keys where at least that many are allowed, and every
    allowed key where fewer are, so a sequence the top-k covers attends as
    the dense mixer does. Equal scores choose the lower key index, because
    `jax.lax.top_k` is stable; torch's `topk` promises no tie order, so a
    selection a tie decides may differ from the reference's. `keep` is the
    attention mask over `scores` `[..., T]` (a leading axis of one
    broadcasts), and a key it forbids is never selected whatever the tie
    rule. A block selection takes blocks for keys (`block_maxima`,
    `dew.nn.block_sparse`).
    """
    total = scores.shape[-1]
    keep = jnp.broadcast_to(keep, scores.shape)
    if forced is not None:
        keep = keep & ~jnp.any(jnp.arange(total)[:, None] == forced[..., None, :], axis=-1)
    ranked = jnp.where(keep, scores, jnp.finfo(at_least_fp32(scores.dtype)).min)
    chosen = jax.lax.top_k(ranked, min(top_k, total), is_stable=True)[1]
    chosen = jnp.where(jnp.take_along_axis(keep, chosen, axis=-1), chosen, -1)
    return chosen if forced is None else jnp.concatenate([chosen, forced.astype(chosen.dtype)], axis=-1)


def selection_mask(indices, total: int):
    """`[B, S, K]` token indices, -1 for none, as the `[B, S, total]` bool mask
    the attention takes: a key is visible to a query iff one of the query's
    indices names it, so out-of-range entries drop out."""
    batch, length, _ = indices.shape
    # A negative index would wrap around in jnp, and one at or past `total`
    # names no key; both go to `DROPPED`, which the scatter drops.
    slots = jnp.where((indices >= 0) & (indices < total), indices, DROPPED)
    return jnp.zeros((batch, length, total), jnp.bool_).at[
        jnp.arange(batch)[:, None, None], jnp.arange(length)[None, :, None], slots
    ].set(True, mode='drop')


def top_k_keys(scores, keep, top_k: int):
    """`top_k_selection` as the `[B, S, T]` mask the dense kernels read."""
    return selection_mask(top_k_selection(scores, keep, top_k), scores.shape[-1])


SPARSE_QUERY_BLOCK = 128
"""Queries `sparse_latent_attention` attends at once. A block holds its
queries' selected latents, `[B, block, K, kv_lora_rank + rope]`, so this
bounds the gather at 151M values per batch row for V3.2's K of 2048."""


def sparse_latent_attention(query_nope, query_rot, latent, rot, key_weight, value_weight,
                            indices, *, scale: float, precision=None,
                            block: int = SPARSE_QUERY_BLOCK):
    """Attend each query over the `K` keys it selected, in the latent space.

    DeepSeek sparse attention's execution (arXiv 2512.02556, section 2.1;
    FlashMLA's sparse decoding kernel): the keys are never expanded per
    head. The nope query absorbs the key half of `kv_b_proj`, so a logit is
    the absorbed query against the key's latent plus the rope query against
    its shared rope head, and the probabilities weight the selected latents
    before the value half expands the result. It is the same softmax over
    the same keys as the dense attention under the selection mask, in
    `O(S * K)` rather than `O(S * T)`.

    `query_nope` `[B, S, H, n]` and `query_rot` `[B, S, H, p]` are the
    rotated, pre-scaled query halves; `latent` `[B, T, r]` and `rot`
    `[B, T, p]` the normed latent and the rotated rope head; `key_weight`
    `[r, H, n]` and `value_weight` `[r, H, v]` the two halves of
    `kv_b_proj`, which carries no bias in any reference. `indices` is the
    selection, -1 where a slot names no key. `scale` is the kernel's
    `1 / sqrt(n + p)`. Returns `[B, S, H, v]`.

    Queries run in blocks of `block` under `jax.checkpoint`, so neither the
    forward nor the backward pass holds more than one block's gathered
    latents. A query whose selection names no key (padding) averages its
    slots where the dense kernel averages every key; both are outputs
    nothing reads.
    """
    batch, length, heads, _ = query_nope.shape
    absorbed = float32_einsum('bshn,rhn->bshr', query_nope, key_weight.astype(query_nope.dtype),
                          precision=precision)
    blocks = -(-length // block)

    def blocked(x):
        return sequence_blocks(x, block, blocks, leading=True)

    def gathered(table, where):
        return jax.vmap(lambda rows, at: rows[at])(table, where)

    @jax.checkpoint
    def attend(pieces):
        q_latent, q_rot, at = pieces
        allowed, at = at >= 0, jnp.maximum(at, 0)
        keys, rots = gathered(latent, at), gathered(rot, at)
        wide = at_least_fp32(q_latent.dtype)
        logits = (float32_einsum('bqhr,bqkr->bqhk', q_latent, keys, precision=precision,
                             preferred_element_type=wide)
                  + float32_einsum('bqhp,bqkp->bqhk', q_rot, rots, precision=precision,
                               preferred_element_type=wide)) * scale
        logits = jnp.where(allowed[:, :, None, :], logits, jnp.finfo(wide).min)
        weights = jax.nn.softmax(logits, axis=-1).astype(keys.dtype)
        return float32_einsum('bqhk,bqkr->bqhr', weights, keys, precision=precision)

    context = jax.lax.map(attend, tuple(blocked(x) for x in (absorbed, query_rot, indices)))
    context = jnp.moveaxis(context, 0, 1).reshape(batch, blocks * block, heads, -1)[:, :length]
    return float32_einsum('bshr,rhv->bshv', context, value_weight.astype(context.dtype),
                      precision=precision)


def block_maxima(scores, visible, block_size: int):
    """Each block of `block_size` entries' best score among those `visible`
    allows (every one where None), `[..., ceil(T / block_size)]`, -inf for a
    block with none: the
    Hierarchical Sparse Indexer's first level's pooling (`candidate_pool`),
    and MiniMax's indexer's onto its key blocks
    (`dew.nn.block_sparse.pooled_selection`)."""
    total = scores.shape[-1]
    count = -(-total // block_size)
    ranked = scores if visible is None else jnp.where(visible, scores, -jnp.inf)
    ranked = jnp.pad(ranked, [(0, 0)] * (ranked.ndim - 1) + [(0, count * block_size - total)],
                     constant_values=-jnp.inf)
    return jnp.max(ranked.reshape(*ranked.shape[:-1], count, block_size), axis=-1)


def candidate_pool(scores, visible, blocks: int, block_size: int):
    """The Hierarchical Sparse Indexer's first level (v41:583-610): the
    `blocks` blocks of `block_size` entries with the highest best score
    (`block_maxima`), the block holding the query's newest entry always
    among them (`top_k_selection`'s `forced`), as the entries they hold,
    `[B, S, min(blocks, ceil(T / block_size)) * block_size]` in ascending
    order, -1 for a pick past the reachable blocks or an entry past the
    last."""
    total = scores.shape[-1]
    best = block_maxima(scores, visible, block_size)
    newest = (jnp.sum(visible, axis=-1) - 1) // block_size
    chosen = top_k_selection(best, best > -jnp.inf, min(blocks, best.shape[-1]) - 1, forced=newest[..., None])
    chosen = jnp.repeat(jnp.sort(chosen, axis=-1), block_size, axis=-1)
    entries = chosen * block_size + jnp.tile(jnp.arange(block_size), chosen.shape[-1] // block_size)
    return jnp.where((chosen >= 0) & (entries < total), entries, -1)


def selected_runs(selected, chunks, *, slots=None, causal: bool = True, own: bool = True):
    """Key runs `[B, Q, S + own, 2]` of a selection `selected` `[B, Q, S]`
    (each query's chunks, -1 for none) over keys numbered into `chunks`
    `[B, K]` or `[K]` (each key's, its chunk's keys consecutive, -1 for a
    key in none: blocks of a row, `arange(K) // block`, or of each document
    from its first place, `dew.nn.inputs.numbered_blocks`): each selected
    chunk's keys, under causality up to the query's own key (`slots` `[B, Q]`,
    each query's key, its index where None; -1 for a query with none), and
    with `own` the query's own chunk up to it; sorted, duplicates and empty
    chunks an empty run, as `scaled_dot_product_attention`'s `key_runs`
    takes them. A selection takes no gradient."""
    rows, queries, _ = selected.shape
    keys = chunks.shape[-1]
    chunks = jnp.broadcast_to(chunks, (rows, keys))
    slots = jnp.broadcast_to(jnp.arange(queries) if slots is None else slots, (rows, queries))
    # Each chunk's first key and the key past its last, `[B, K]` (no chunk outnumbers its keys).
    index, numbered = jnp.arange(keys), jnp.where(chunks >= 0, chunks, keys)
    first = jax.vmap(lambda ids: jax.ops.segment_min(index, ids, num_segments=keys))(numbered)
    past = jax.vmap(lambda ids: jax.ops.segment_max(index, ids, num_segments=keys))(numbered) + 1
    taken = selected.astype(jnp.int32)
    if own:
        mine = jnp.take_along_axis(chunks, jnp.maximum(slots, 0), axis=-1)
        taken = jnp.concatenate([taken, jnp.where(slots >= 0, mine, -1)[..., None]], axis=-1)
    taken = jnp.sort(jnp.where(taken < 0, jnp.iinfo(jnp.int32).max, taken), axis=-1)
    repeated = jnp.concatenate([jnp.zeros_like(taken[..., :1], bool), taken[..., 1:] == taken[..., :-1]],
                               axis=-1)
    kept = (taken != jnp.iinfo(jnp.int32).max) & ~repeated
    at = jnp.where(kept, taken, 0).reshape(rows, -1)
    start, end = (jnp.take_along_axis(table, at, axis=-1).reshape(taken.shape) for table in (first, past))
    if causal:
        end = jnp.minimum(end, slots[..., None] + 1)
    held = kept & (end > start)
    # The held runs first, in order, the empty after them.
    order = jnp.argsort(jnp.where(held, start, jnp.iinfo(jnp.int32).max), axis=-1, stable=True)
    start, end, held = (jnp.take_along_axis(each, order, axis=-1) for each in (start, end, held))
    runs = jnp.stack([jnp.where(held, start, 0), jnp.where(held, end, 0)], axis=-1)
    return jax.lax.stop_gradient(runs)


def selected_attention(query, key, value, selected, chunks, *, slots=None, causal: bool = True,
                       own: bool = True, implementation: AttentionImpl = 'auto', **options):
    """Attention whose each query reads the chunks its selection names
    (`selected` `[B, R, Q, S]`, a selection a GQA group's or a query head's,
    `selected_runs` over `chunks`), each of the `R` selections a row of the
    call's batch (the key runs are a row's): `query` `[B, Q, R * h, D]`,
    `key` and `value` `[B, K, G, D]` with `R` a multiple of `G` (a head's
    selection reads its group's keys), out `[B, Q, R * h, Dv]`, through the
    template's indexed form (`scaled_dot_product_attention`'s `key_runs`).
    `options` are the call's other arguments. A call whose queries are its
    keys (`slots` None) states its causality to the kernel too; a cache
    call's runs alone end each query's keys."""
    rows, queries, heads, _ = query.shape
    groups = selected.shape[1]
    per = heads // groups
    shared = groups // key.shape[2]

    def batched(array: jax.Array, held: int) -> jax.Array:
        # `[B, S, R * held, D]` as `[B * R, S, held, D]`, a selection a row.
        length, depth = array.shape[1], array.shape[-1]
        return array.reshape(rows, length, groups, held, depth).transpose(0, 2, 1, 3, 4).reshape(
            rows * groups, length, held, depth)

    def rowed(array):
        return None if array is None else jnp.repeat(jnp.broadcast_to(array, (rows, array.shape[-1])),
                                                     groups, axis=0)

    runs = selected_runs(selected.reshape(rows * groups, queries, -1), rowed(chunks), slots=rowed(slots),
                         causal=causal, own=own)
    key, value = (jnp.repeat(each, shared, axis=2) for each in (key, value))
    out = scaled_dot_product_attention(batched(query, per), batched(key, 1), batched(value, 1),
                                       causal=causal and slots is None, key_runs=runs,
                                       implementation=implementation, **options)
    out = out.reshape(rows, groups, queries, per, -1).transpose(0, 2, 1, 3, 4)
    return out.reshape(rows, queries, heads, -1)
