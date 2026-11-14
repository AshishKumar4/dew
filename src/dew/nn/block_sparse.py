"""Block-selected attention: each query reads the keys of the key blocks it selects, the attention template's
indexed form (docs/design/kernel-templates.md §5; its §A rows 36, 37, 59, 60 and 64).

A selection is each query's key blocks `[B, Q, S]` (-1 where it selects none), from a scoring pass the
model runs in XLA (`top_blocks`): MoBA's gate over block-mean keys (`moba_selection`), NSA's compressed
branch's weights (`nsa_selection`), MiniMax's learned indexer max-pooled to blocks (`pooled_selection`),
InfLLM-V2's two-window pooling (`infllm_selection`) or a caller's block mask (`masked_selection`).
`selected_runs` turns it into the key runs `scaled_dot_product_attention` takes (`key_runs`), each selected
block one run, which the template's mainloops walk as the selected tiles alone
(`attention_mosaic_program.indexed_walk`): work a query block's selected blocks, not its keys. A selection
takes no gradient (the sources' indices are detached)."""

import jax
import jax.numpy as jnp
import numpy as np

from dew.nn.attention import AttentionImpl, scaled_dot_product_attention


def selected_runs(selected: jax.Array, block: int, keys: int, *, causal: bool = True,
                  own: bool = True) -> jax.Array:
    """Key runs `[B, Q, S + own, 2]` of a selection `selected` `[B, Q, S]` (each query's key blocks of `block`
    keys over `keys`, -1 for none): each selected block's keys `[b * block, (b + 1) * block)`, under causality
    up to the query's own key, and with `own` the query's own block up to it; sorted, duplicates and empty
    blocks an empty run, as `scaled_dot_product_attention`'s `key_runs` takes them."""
    rows, queries, _ = selected.shape
    query = jnp.arange(queries, dtype=jnp.int32)[None, :, None]
    blocks = selected.astype(jnp.int32)
    if own:
        blocks = jnp.concatenate([blocks, jnp.broadcast_to(query // block, (rows, queries, 1))], axis=-1)
    blocks = jnp.sort(jnp.where(blocks < 0, jnp.iinfo(jnp.int32).max, blocks), axis=-1)
    repeated = jnp.concatenate([jnp.zeros_like(blocks[..., :1], bool), blocks[..., 1:] == blocks[..., :-1]],
                               axis=-1)
    kept = (blocks != jnp.iinfo(jnp.int32).max) & ~repeated
    start = jnp.minimum(blocks * block, keys)
    end = jnp.minimum(start + block, keys)
    if causal:
        end = jnp.minimum(end, query + 1)
    held = kept & (end > start)
    # The held runs first, in order, the empty after them.
    order = jnp.argsort(jnp.where(held, start, jnp.iinfo(jnp.int32).max), axis=-1, stable=True)
    start, end, held = (jnp.take_along_axis(each, order, axis=-1) for each in (start, end, held))
    runs = jnp.stack([jnp.where(held, start, 0), jnp.where(held, end, 0)], axis=-1)
    return jax.lax.stop_gradient(runs)


def top_blocks(scores: jax.Array, top_k: int, *, forced: jax.Array | None = None) -> jax.Array:
    """Each query's `top_k` key blocks of `scores` `[B, Q, Kb]` (-inf where a block is not a candidate), and
    `forced` `[B, Q, F]` blocks taken whatever their score (-1 for none), as a selection `[B, Q, top_k + F]`,
    -1 where a query has fewer candidates (`selected_runs`)."""
    scores = jax.lax.stop_gradient(scores)
    if forced is not None:
        # A forced block is no candidate of the top k, which takes others beside it.
        taken = jnp.any(jnp.arange(scores.shape[-1])[None, None, :, None] == forced[..., None, :], axis=-1)
        scores = jnp.where(taken, -jnp.inf, scores)
    values, indices = jax.lax.top_k(scores, min(top_k, scores.shape[-1]))
    chosen = jnp.where(values > -jnp.inf, indices, -1).astype(jnp.int32)
    return chosen if forced is None else jnp.concatenate([chosen, forced.astype(jnp.int32)], axis=-1)


def selected_attention(query: jax.Array, key: jax.Array, value: jax.Array, selected: jax.Array, *, block: int,
                       causal: bool = True, own: bool = True,
                       implementation: AttentionImpl = "auto") -> jax.Array:
    """Attention whose each query reads its GQA group's selected key blocks (`selected` `[B, G, Q, S]`, a key
    head's group's, `selected_runs`), the groups in the call's batch (the key runs are a row's): `query`
    `[B, Q, G * h, D]`, `key` and `value` `[B, K, G, D]`, out `[B, Q, G * h, Dv]`."""
    rows, queries, heads, _ = query.shape
    groups = key.shape[2]
    per = heads // groups

    def batched(array: jax.Array, held: int) -> jax.Array:
        # `[B, S, G * held, D]` as `[B * G, S, held, D]`, a group a row.
        length, depth = array.shape[1], array.shape[-1]
        return array.reshape(rows, length, groups, held, depth).transpose(0, 2, 1, 3, 4).reshape(
            rows * groups, length, held, depth)

    runs = selected_runs(selected.reshape(rows * groups, queries, -1), block, key.shape[1], causal=causal,
                         own=own)
    out = scaled_dot_product_attention(batched(query, per), batched(key, 1), batched(value, 1),
                                       causal=causal, key_runs=runs, implementation=implementation)
    out = out.reshape(rows, groups, queries, per, -1).transpose(0, 2, 1, 3, 4)
    return out.reshape(rows, queries, heads, -1)


def _earlier_blocks(queries: int, blocks: int, block: int) -> jax.Array:
    """`[Q, Kb]`: whether key block j lies wholly before query i's own block (a causal selection's
    candidates)."""
    own = jnp.arange(queries)[:, None] // block
    return jnp.arange(blocks)[None, :] < own


def moba_selection(query: jax.Array, key: jax.Array, *, block: int, top_k: int) -> jax.Array:
    """MoBA's selection (arXiv 2502.13189): each query head's gate against each earlier key block's mean key,
    the `top_k` highest, beside its own block (`selected_runs`' `own`). `query` `[B, Q, H, D]`, `key`
    `[B, K, H, D]` (heads matched); a selection `[B, H, Q, top_k]`, a head's own."""
    rows, keys, heads, width = key.shape
    blocks = -(-keys // block)
    padded = jnp.pad(key, ((0, 0), (0, blocks * block - keys), (0, 0), (0, 0)))
    counts = jnp.minimum(block, keys - jnp.arange(blocks) * block).astype(jnp.float32)
    means = (padded.reshape(rows, blocks, block, heads, width).astype(jnp.float32).sum(2)
             / counts[:, None, None])
    gate = jnp.einsum("bqhd,bkhd->bhqk", query.astype(jnp.float32), means)
    gate = jnp.where(_earlier_blocks(query.shape[1], blocks, block)[None, None], gate, -jnp.inf)
    return top_blocks(gate, top_k)


def moba_attention(query: jax.Array, key: jax.Array, value: jax.Array, *, block: int, top_k: int,
                   implementation: AttentionImpl = "auto") -> jax.Array:
    """MoBA's causal attention: each query head reads its own block up to it and the `top_k` earlier blocks
    its gate selects (`moba_selection`), the template's indexed form a head a row (`selected_attention`).
    `[B, S, H, D]` in, `[B, S, H, Dv]` out; keys and values with the queries' heads."""
    selected = moba_selection(query, key, block=block, top_k=top_k)
    return selected_attention(query, key, value, selected, block=block, implementation=implementation)


def nsa_selection(compressed_weights: jax.Array, *, stride: int, length: int, block: int, top_k: int,
                  queries: int, initial: int = 1, local: int = 2) -> jax.Array:
    """NSA's selection (arXiv 2502.11089 §3.3.2): a GQA group's compressed branch's attention weights
    `[B, G, h, Q, C]` (its `h` heads' softmax over `C` compressed keys, each of `length` keys every `stride`,
    eq. 8) summed over the group's heads (eq. 10), spread onto key blocks of `block` by eq. 9 (block j's
    score the weights of compressed keys `(block / stride) j + m + n`, `m < block / stride`, `n < length /
    stride`; `stride` divides both), the `top_k` highest of the earlier blocks beside the first `initial`
    and the `local` ending at the query's own, forced (the paper's n counts them): a selection `[B, G, Q,
    top_k + initial + local]`."""
    assert block % stride == 0 and length % stride == 0, "NSA's eq. 9 takes a stride dividing both blocks"
    weights = jax.lax.stop_gradient(compressed_weights).sum(axis=2)
    compressed = weights.shape[-1]
    blocks = -(-queries // block)
    # Eq. 9's spread: how many (m, n) pairs take compressed key c to block j.
    offsets = (np.arange(block // stride)[:, None] + np.arange(length // stride)[None, :]).ravel()
    spread = np.zeros((compressed, blocks), np.float32)
    for j in range(blocks):
        for c in (block // stride) * j + offsets:
            if c < compressed:
                spread[c, j] += 1
    scores = jnp.einsum("bgqc,ck->bgqk", weights, jnp.asarray(spread, weights.dtype))
    scores = jnp.where(_earlier_blocks(queries, blocks, block)[None, None], scores, -jnp.inf)
    own = jnp.arange(queries)[:, None] // block
    forced = jnp.concatenate([jnp.broadcast_to(jnp.arange(initial)[None], (queries, initial)),
                              own - jnp.arange(local)[None, ::-1]], axis=-1)
    forced = jnp.where(forced >= 0, forced, -1)
    return top_blocks(scores, top_k, forced=jnp.broadcast_to(forced, (*scores.shape[:2], *forced.shape)))


def nsa_attention(query: jax.Array, key: jax.Array, value: jax.Array, compressed_key: jax.Array,
                  compressed_value: jax.Array, gates: jax.Array, *, stride: int, length: int, block: int,
                  top_k: int, window: int, window_key: jax.Array | None = None,
                  window_value: jax.Array | None = None, initial: int = 1, local: int = 2,
                  implementation: AttentionImpl = "auto") -> jax.Array:
    """NSA's attention (arXiv 2502.11089 §3), three branches a GQA group, gated: the compressed branch over
    the model's compressed keys and values `[B, C, G, D]` (compressed key c the keys `[c * stride, c * stride
    + length)`, eq. 7, read by a query that reaches its last), the selected branch over `key` and `value`'s
    blocks its weights select (`nsa_selection`, `selected_attention`) and the sliding branch over a `window`
    of `window_key` and `window_value` (each branch its own keys and values, §3.3.3; the selected branch's
    where not given), combined by `gates` `[B, Q, H, 3]` in XLA (the gate combine a pointwise op,
    kernel-templates.md §A row 36). `query` `[B, Q, G * h, D]`, `key` and `value` `[B, K, G, D]`."""
    rows, queries, heads, width = query.shape
    groups, compressed = key.shape[2], compressed_key.shape[1]
    per = heads // groups
    # The compressed branch's weights (eq. 8), which the selection reads, over the compressed keys whose
    # last key a query reaches.
    reach = jnp.arange(compressed)[None, :] * stride + length <= jnp.arange(queries)[:, None] + 1
    grouped = query.reshape(rows, queries, groups, per, width).astype(jnp.float32)
    logits = jnp.einsum("bqghd,bcgd->bghqc", grouped, compressed_key.astype(jnp.float32)) * width ** -0.5
    logits = jnp.where(reach[None, None, None], logits, -jnp.inf)
    weights = jnp.where(reach[None, None, None], jax.nn.softmax(logits, axis=-1), 0.0)
    compressed_out = jnp.einsum("bghqc,bcgd->bqghd", weights, compressed_value.astype(jnp.float32))
    selected = nsa_selection(weights, stride=stride, length=length, block=block, top_k=top_k, queries=queries,
                             initial=initial, local=local)
    selected_out = selected_attention(query, key, value, selected, block=block, own=False,
                                      implementation=implementation)
    sliding_out = scaled_dot_product_attention(
        query, key if window_key is None else window_key, value if window_value is None else window_value,
        causal=True, sliding_window=window, implementation=implementation)
    branches = jnp.stack([compressed_out.reshape(rows, queries, heads, -1).astype(selected_out.dtype),
                          selected_out, sliding_out], axis=-1)
    return jnp.sum(branches * gates[..., None, :].astype(branches.dtype), axis=-1)


def pooled_selection(index_scores: jax.Array, *, block: int, top_k: int, local: int = 1) -> jax.Array:
    """MiniMax Sparse Attention's selection (kernel-templates.md §A row 59): a learned token indexer's scores
    `[B, G, Q, K]` (a GQA group's, causal: -inf past a query) max-pooled to key blocks of `block`, the `top_k`
    highest of the earlier blocks beside the `local` ending at the query's own, forced: `[B, G, Q, top_k +
    local]`."""
    rows, groups, queries, keys = index_scores.shape
    blocks = -(-keys // block)
    padded = jnp.pad(index_scores, ((0, 0), (0, 0), (0, 0), (0, blocks * block - keys)),
                     constant_values=-jnp.inf)
    pooled = padded.reshape(rows, groups, queries, blocks, block).max(-1)
    pooled = jnp.where(_earlier_blocks(queries, blocks, block)[None, None], pooled, -jnp.inf)
    own = jnp.arange(queries)[:, None] // block
    forced = own - jnp.arange(local)[None, ::-1]
    forced = jnp.where(forced >= 0, forced, -1)
    return top_blocks(pooled, top_k, forced=jnp.broadcast_to(forced, (rows, groups, *forced.shape)))


def infllm_selection(query: jax.Array, key: jax.Array, *, block: int, top_k: int = 64, kernel: int = 32,
                     stride: int = 16, initial: int = 1, local: int = 1) -> jax.Array:
    """InfLLM-V2's selection (arXiv 2509.24663 §3.3): keys mean-pooled by a window of `kernel` every `stride`
    into compressed keys (eq. 4), each head's softmax of its query against the compressed keys whose window
    it reaches whole, at the attention's scale `D ** -0.5` (eq. 5, causal), summed over a GQA group's heads
    (eq. 6), max-pooled onto key blocks of `block` over `block / stride` compressed keys every `block /
    stride` and `kernel / stride - 1` more (eq. 7: 5 every 4 at the paper's 64, 32, 16), the `top_k`
    highest of the earlier blocks beside the first `initial` and the `local` ending at the query's own,
    forced (eqs. 2, 3). `query` `[B, Q, G, h, D]` (G groups of h heads), `key` `[B, K, G, D]`; `[B, G, Q,
    top_k + initial + local]`. Its pooling is `reduce_window`'s, each window a key's or a score's once."""
    assert block % stride == 0 and kernel % stride == 0, "InfLLM-V2's pooling takes a stride dividing both"
    rows, keys, groups, width = key.shape
    queries = query.shape[1]
    blocks = -(-keys // block)
    own = jnp.arange(queries)[:, None] // block
    forced = jnp.concatenate([jnp.broadcast_to(jnp.arange(initial)[None], (queries, initial)),
                              own - jnp.arange(local)[None, ::-1]], axis=-1)
    forced = jnp.broadcast_to(jnp.where(forced >= 0, forced, -1), (rows, groups, queries, initial + local))
    windows = (keys - kernel) // stride + 1 if keys >= kernel else 0
    if windows == 0:
        # No window holds `kernel` keys: the forced blocks alone.
        return top_blocks(jnp.full((rows, groups, queries, blocks), -jnp.inf), top_k, forced=forced)
    summed = jax.lax.reduce_window(key.astype(jnp.float32), 0.0, jax.lax.add, (1, kernel, 1, 1),
                                   (1, stride, 1, 1), "VALID")
    pooled = summed / kernel
    logits = jnp.einsum("bqghd,bwgd->bghqw", query.astype(jnp.float32), pooled) * width ** -0.5
    # A query reaches a window whose last key is at or before its own.
    reach = jnp.arange(windows)[None, :] * stride + kernel <= jnp.arange(queries)[:, None] + 1
    logits = jnp.where(reach, logits, -jnp.inf)
    weights = jnp.where(reach, jax.nn.softmax(logits, axis=-1), 0.0).sum(axis=2)
    step = block // stride
    span = step + kernel // stride - 1
    padded = jnp.pad(weights, ((0, 0), (0, 0), (0, 0), (0, max(0, (blocks - 1) * step + span - windows))))
    scores = jax.lax.reduce_window(padded, -jnp.inf, jax.lax.max, (1, 1, 1, span), (1, 1, 1, step),
                                   "VALID")[..., :blocks]
    scores = jnp.where(_earlier_blocks(queries, blocks, block)[None, None], scores, -jnp.inf)
    return top_blocks(scores, top_k, forced=forced)


def masked_selection(mask: jax.Array, block_q: int, *, queries: int, most: int) -> jax.Array:
    """A caller's block mask `[B, Qb, Kb]` (FlagAttention's Hy3 block-mask prefill, kernel-templates.md §A
    row 64) as a selection `[B, Q, most]`: each query its query block's selected key blocks in order, at
    most `most` (static: the mask's density bound, the scheduler's bucket), -1 after."""
    blocks = mask.shape[-1]
    order = jnp.where(mask, jnp.arange(blocks)[None, None], blocks)
    chosen = jnp.sort(order, axis=-1)[..., :most]
    chosen = jnp.where(chosen < blocks, chosen, -1).astype(jnp.int32)
    return jnp.repeat(chosen, block_q, axis=1)[:, :queries]
