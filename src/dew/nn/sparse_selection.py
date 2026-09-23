"""DeepSeek sparse attention's selection and its execution in the latent space.

A selection is `[B, S, K]` key indices per query, -1 in a slot that names
no key, which is how GLM's reference builds its top-k mask
(`build_attention_mask_from_topk`, modeling_glm5_next.py:1218-1256). The
V3.2 indexer (`dew.nn.mla`), GLM's k-pool indexer (`dew.nn.dsa_kpool`) and
DeepSeek V4's compressed-entry indexer (`dew.nn.deepseek_v4`) all produce
one, `selection_mask` turns it into the `[B, S, T]` mask the dense kernels
read, and `sparse_latent_attention` attends it without that mask.
"""

import jax
import jax.numpy as jnp


def top_k_selection(scores, keep, top_k: int):
    """The `top_k` highest-scoring keys of each query among those `keep`
    allows: `[B, S, K]` indices with `K = min(top_k, T)`, -1 in a slot no
    allowed key fills.

    Exactly `top_k` keys where at least that many are allowed, and every
    allowed key where fewer are, so a sequence the top-k covers attends as
    the dense mixer does. Equal scores choose the lower key index, because
    `jax.lax.top_k` is stable; torch's `topk` promises no tie order, so a
    selection a tie decides may differ from the reference's. `keep` is the
    `[B, S, T]` attention mask (a leading axis of one broadcasts), and a key
    it forbids is never selected whatever the tie rule.
    """
    batch, length, total = scores.shape
    keep = jnp.broadcast_to(keep, (batch, length, total))
    ranked = jnp.where(keep, scores, jnp.finfo(jnp.float32).min)
    chosen = jax.lax.top_k(ranked, min(top_k, total), is_stable=True)[1]
    return jnp.where(jnp.take_along_axis(keep, chosen, axis=-1), chosen, -1)


def selection_mask(indices, total: int):
    """`[B, S, K]` token indices, -1 for none, as the `[B, S, total]` bool mask
    the attention takes: a key is visible to a query iff one of the query's
    indices names it, so out-of-range entries drop out."""
    batch, length, _ = indices.shape
    # A negative index would wrap around in jnp; sending it past the end
    # lets the scatter drop it, as it drops any index at or past `total`.
    slots = jnp.where(indices >= 0, indices, total)
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
    absorbed = jnp.einsum('bshn,rhn->bshr', query_nope, key_weight.astype(query_nope.dtype),
                          precision=precision)
    blocks = -(-length // block)

    def blocked(x):
        x = jnp.pad(x, [(0, 0), (0, blocks * block - length)] + [(0, 0)] * (x.ndim - 2))
        return jnp.moveaxis(x.reshape(batch, blocks, block, *x.shape[2:]), 1, 0)

    def gathered(table, where):
        return jax.vmap(lambda rows, at: rows[at])(table, where)

    @jax.checkpoint
    def attend(pieces):
        q_latent, q_rot, at = pieces
        allowed, at = at >= 0, jnp.maximum(at, 0)
        keys, rots = gathered(latent, at), gathered(rot, at)
        logits = (jnp.einsum('bqhr,bqkr->bqhk', q_latent, keys, precision=precision,
                             preferred_element_type=jnp.float32)
                  + jnp.einsum('bqhp,bqkp->bqhk', q_rot, rots, precision=precision,
                               preferred_element_type=jnp.float32)) * scale
        logits = jnp.where(allowed[:, :, None, :], logits, jnp.finfo(jnp.float32).min)
        weights = jax.nn.softmax(logits, axis=-1).astype(keys.dtype)
        return jnp.einsum('bqhk,bqkr->bqhr', weights, keys, precision=precision)

    context = jax.lax.map(attend, tuple(blocked(x) for x in (absorbed, query_rot, indices)))
    context = jnp.moveaxis(context, 0, 1).reshape(batch, blocks * block, heads, -1)[:, :length]
    return jnp.einsum('bshr,rhv->bshv', context, value_weight.astype(context.dtype),
                      precision=precision)


def candidate_pool(scores, visible, blocks: int, block_size: int):
    """The Hierarchical Sparse Indexer's first level (v41:583-610): the
    `blocks` blocks of `block_size` entries with the highest best score, the
    block holding the query's newest entry always among them, as the
    entries they hold, `[B, S, blocks * block_size]` in ascending order, -1
    for a pick past the reachable blocks or an entry past the last."""
    batch, length, total = scores.shape
    count = -(-total // block_size)
    ranked = jnp.where(visible, scores, -jnp.inf)
    ranked = jnp.pad(ranked, ((0, 0), (0, 0), (0, count * block_size - total)),
                     constant_values=-jnp.inf)
    best = jnp.max(ranked.reshape(batch, length, count, block_size), axis=-1)
    newest = (jnp.sum(visible, axis=-1) - 1) // block_size
    best = jnp.where(jnp.arange(count) == newest[..., None], jnp.inf, best)
    values, chosen = jax.lax.top_k(best, min(blocks, count))
    chosen = jnp.repeat(jnp.sort(jnp.where(values > -jnp.inf, chosen, -1), axis=-1), block_size, axis=-1)
    entries = chosen * block_size + jnp.tile(jnp.arange(block_size), chosen.shape[-1] // block_size)
    return jnp.where((chosen >= 0) & (entries < total), entries, -1)
