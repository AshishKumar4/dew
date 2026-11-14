"""Block-selected attention (`dew.nn.block_sparse`): each selection against its definition written out in
numpy, the runs a selection gives against the keys it reads, and the attention over a selection on par with
the reference over its dense mask (check 1)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from attention_oracle import qkv
from reference_error import assert_as_exact, with_gradients

from dew.nn import block_sparse
from dew.nn.attention import scaled_dot_product_attention

SEEDS = 8
"""The draws each check-1 site here takes (`reference_error.seeds_for`, sized: 8 each)."""


def _selected_mask(selected: np.ndarray, block: int, keys: int, *, own: bool) -> np.ndarray:
    """`[B, Q, K]`: each query's selected blocks' keys up to its own, and its own block's, written out."""
    rows, queries, _ = selected.shape
    mask = np.zeros((rows, queries, keys), bool)
    for row in range(rows):
        for query in range(queries):
            chosen = {int(b) for b in selected[row, query] if b >= 0}
            if own:
                chosen.add(query // block)
            for b in chosen:
                mask[row, query, b * block:min((b + 1) * block, keys, query + 1)] = True
    return mask


def _drawn(rows: int, queries: int, heads: int, groups: int, width: int):
    """A seed's bf16 queries `[B, Q, H, D]`, keys and values of `groups` heads, and output cotangents."""
    def draw(seed):
        query, key, value = qkv((rows, queries, heads, width), jnp.bfloat16, seed=seed,
                                kv_shape=(rows, queries, groups, width))
        dout = jax.random.normal(jax.random.PRNGKey(seed + 1000), (rows, queries, heads, width), jnp.bfloat16)
        return query, key, value, dout

    return draw


@pytest.mark.parametrize("own", [True, False], ids=["own", "selected-alone"])
def test_a_selections_runs_hold_its_selected_blocks_keys_up_to_each_query(own):
    """A selection's runs (`selected_runs`, duplicates and padding among its blocks) let each query see the
    keys of its selected blocks, and its own block where `own`, up to its own key, and no other (the
    template's visibility over them against the definition, bitwise)."""
    rows, queries, block = 2, 96, 16
    rng = np.random.default_rng(0)
    selected = rng.integers(-1, queries // block, size=(rows, queries, 4))
    selected[:, :, 3] = selected[:, :, 2]
    runs = np.asarray(block_sparse.selected_runs(jnp.asarray(selected), block, queries, own=own))
    index = jnp.arange(queries)[None, None, :, None]
    seen = np.asarray(jnp.any((index >= runs[:, :, None, :, 0]) & (index < runs[:, :, None, :, 1]), axis=-1))
    np.testing.assert_array_equal(seen, _selected_mask(selected, block, queries, own=own))


def test_moba_selects_each_heads_top_earlier_blocks_by_its_gate():
    """MoBA's selection (`moba_selection`): each query head's `top_k` earlier key blocks by its query's
    product with the block's mean key, as written out over numpy (the same blocks)."""
    rows, queries, heads, width, block, top_k = 1, 64, 2, 8, 8, 3
    query, key, _ = (np.asarray(each, np.float32) for each in qkv((rows, queries, heads, width), jnp.float32))
    selected = np.asarray(block_sparse.moba_selection(jnp.asarray(query), jnp.asarray(key), block=block,
                                                      top_k=top_k))
    means = key.reshape(rows, queries // block, block, heads, width).mean(2)
    for head in range(heads):
        for index in range(queries):
            earlier = index // block
            gate = query[0, index, head] @ means[0, :earlier, head].T
            expected = set(np.argsort(-gate)[:top_k].tolist())
            assert {int(b) for b in selected[0, head, index] if b >= 0} == expected


def test_a_masked_selection_takes_each_query_blocks_selected_key_blocks():
    """A block mask's selection (`masked_selection`): each query its query block's selected key blocks, in
    order, at most `most`, -1 after."""
    mask = np.zeros((1, 3, 5), bool)
    mask[0, 0, [0]] = mask[0, 1, [1, 3]] = mask[0, 2, [0, 2, 4]] = True
    selected = np.asarray(block_sparse.masked_selection(jnp.asarray(mask), 4, queries=12, most=3))
    expected = np.repeat(np.asarray([[[0, -1, -1], [1, 3, -1], [0, 2, 4]]]), 4, axis=1)
    np.testing.assert_array_equal(selected, expected)


def test_attention_over_a_groups_selection_holds_check_1_against_its_dense_mask():
    """`selected_attention` over a selection a GQA group (two key heads, four query heads each, its groups
    taken into the call's batch): the output and every gradient on par with the reference attention over
    the selection's dense mask (check 1)."""
    rows, queries, heads, groups, width, block = 1, 128, 8, 2, 32, 16
    rng = np.random.default_rng(1)
    selected = jnp.asarray(rng.integers(-1, queries // block, size=(rows, groups, queries, 3)), jnp.int32)
    mask = np.stack([_selected_mask(np.asarray(selected[:, group]), block, queries, own=True)
                     for group in range(groups)], axis=1)
    dense = jnp.asarray(np.repeat(mask, heads // groups, axis=1))

    draw = _drawn(rows, queries, heads, groups, width)

    def ours(query, key, value):
        return block_sparse.selected_attention(query, key, value, selected, block=block)

    def reference(query, key, value):
        return scaled_dot_product_attention(query, key, value, mask=dense, implementation="xla")

    assert_as_exact(jax.jit(with_gradients(ours)), jax.jit(with_gradients(reference)), draw,
                    "block-selected attention over its groups' selections", seeds=SEEDS)


def _top(scores: np.ndarray, top_k: int, forced: set[int]) -> set[int]:
    """The `top_k` highest finite scores' indices beside `forced`, written out."""
    candidates = [index for index in np.argsort(-scores)
                  if np.isfinite(scores[index]) and index not in forced]
    return set(candidates[:top_k]) | forced


def test_nsa_selects_each_groups_top_blocks_by_its_compressed_weights():
    """NSA's selection (`nsa_selection`): a group's compressed weights (each compressed key `length` keys
    every `stride`) summed over its heads (eq. 10), spread onto key blocks by eq. 9, the `top_k` earlier
    blocks beside the first and the two ending at the query's own, as written out."""
    rows, groups, per, queries, stride, length, block, top_k = 1, 2, 2, 64, 4, 8, 16, 1
    compressed = (queries - length) // stride + 1
    weights = np.random.default_rng(2).random((rows, groups, per, queries, compressed)).astype(np.float32)
    selected = np.asarray(block_sparse.nsa_selection(jnp.asarray(weights), stride=stride, length=length,
                                                     block=block, top_k=top_k, queries=queries))
    for group in range(groups):
        for query in range(queries):
            summed = weights[0, group, :, query].sum(0)
            own = query // block
            scores = np.full(queries // block, -np.inf)
            for j in range(own):
                scores[j] = sum(summed[c] for m in range(block // stride) for n in range(length // stride)
                                if (c := (block // stride) * j + m + n) < compressed)
            forced = {0, *(b for b in (own - 1, own) if b >= 0)}
            assert {int(b) for b in selected[0, group, query] if b >= 0} == _top(scores, top_k, forced)


def test_a_pooled_selection_takes_each_groups_top_blocks_by_its_indexers_block_maxima():
    """MiniMax's selection (`pooled_selection`): a group's indexer scores max-pooled onto key blocks, the
    `top_k` earlier blocks beside the query's own, as written out."""
    rows, groups, queries, block, top_k = 1, 2, 48, 8, 2
    scores = np.random.default_rng(3).standard_normal((rows, groups, queries, queries)).astype(np.float32)
    selected = np.asarray(block_sparse.pooled_selection(jnp.asarray(scores), block=block, top_k=top_k))
    for group in range(groups):
        for query in range(queries):
            own = query // block
            pooled = np.asarray([scores[0, group, query, b * block:(b + 1) * block].max()
                                 if b < own else -np.inf for b in range(queries // block)])
            assert {int(b) for b in selected[0, group, query] if b >= 0} == _top(pooled, top_k, {own})


@pytest.mark.parametrize("keys", [64, 16], ids=["windows", "fewer-keys-than-a-window"])
def test_infllm_selects_each_groups_top_blocks_by_its_pooled_keys(keys):
    """InfLLM-V2's selection (`infllm_selection`): keys mean-pooled by windows of `kernel` every `stride`,
    each head's softmax over the windows its query reaches whole (no future key moves a score), summed over
    its group's heads, max-pooled onto blocks by eq. 7, the `top_k` earlier blocks beside the first and the
    query's own, as written out; with fewer keys than a window, the forced blocks alone."""
    rows, groups, per, width, block, top_k, kernel, stride = 1, 2, 2, 8, 16, 1, 8, 4
    rng = np.random.default_rng(4)
    query = rng.standard_normal((rows, keys, groups, per, width)).astype(np.float32)
    key = rng.standard_normal((rows, keys, groups, width)).astype(np.float32)
    selected = np.asarray(block_sparse.infllm_selection(jnp.asarray(query), jnp.asarray(key), block=block,
                                                        top_k=top_k, kernel=kernel, stride=stride))
    starts = list(range(0, keys - kernel + 1, stride))
    step, span = block // stride, block // stride + kernel // stride - 1
    for group in range(groups):
        pooled = np.stack([key[0, s:s + kernel, group].mean(0) for s in starts]) if starts else None
        for index in range(keys):
            own = index // block
            reached = [w for w, s in enumerate(starts) if s + kernel <= index + 1]
            weights = np.zeros(len(starts))
            for head in range(per):
                if reached:
                    logits = pooled[reached] @ query[0, index, group, head] * width ** -0.5
                    shifted = np.exp(logits - logits.max())
                    weights[reached] += shifted / shifted.sum()
            scores = np.asarray([max(weights[step * j:step * j + span], default=-np.inf)
                                 if j < own else -np.inf for j in range(-(-keys // block))])
            assert {int(b) for b in selected[0, group, index] if b >= 0} == _top(scores, top_k, {0, own})


def test_nsa_attention_holds_check_1_against_its_branches_written_out():
    """NSA's attention (`nsa_attention`): the compressed branch (each compressed key `length` keys every
    `stride`, read once a query reaches its last), the selected blocks its weights pick (through the
    template's indexed form) and the sliding window over its own keys and values, gated, on par with the
    branches written out over dense masks (check 1), the output and every gradient."""
    rows, queries, heads, groups, width, stride, length, block, window = 1, 128, 4, 2, 32, 8, 16, 16, 32
    compressed = (queries - length) // stride + 1

    draw = _drawn(rows, queries, heads, groups, width)

    streams = jax.random.split(jax.random.PRNGKey(7), 5)
    compressed_key, compressed_value = (jax.random.normal(stream, (rows, compressed, groups, width),
                                                          jnp.bfloat16) for stream in streams[:2])
    window_key, window_value = (jax.random.normal(stream, (rows, queries, groups, width), jnp.bfloat16)
                                for stream in streams[3:])
    gates = jax.nn.sigmoid(jax.random.normal(streams[2], (rows, queries, heads, 3)))
    settings = {"stride": stride, "length": length, "block": block, "top_k": 2, "window": window}

    def ours(query, key, value):
        return block_sparse.nsa_attention(query, key, value, compressed_key, compressed_value, gates,
                                          window_key=window_key, window_value=window_value, **settings)

    def compressed_weights(query, dtype):
        # The compressed branch's weights over the compressed keys each query reaches, in `dtype`.
        reach = jnp.arange(compressed)[None, :] * stride + length <= jnp.arange(queries)[:, None] + 1
        grouped = query.reshape(rows, queries, groups, heads // groups, width).astype(dtype)
        logits = jnp.einsum("bqghd,bcgd->bghqc", grouped, compressed_key.astype(dtype)) * width ** -0.5
        return jnp.where(reach, jax.nn.softmax(jnp.where(reach, logits, -jnp.inf), axis=-1), 0.0)

    def reference(query, key, value):
        per = heads // groups
        # The branches in the operands' widest type, the selection from the run's fp32 weights (discrete,
        # detached: the truth selects as the run does).
        wide = jnp.promote_types(query.dtype, jnp.float32)
        cmp = jnp.einsum("bghqc,bcgd->bqghd", compressed_weights(query, wide), compressed_value.astype(wide))
        selected = block_sparse.nsa_selection(compressed_weights(query.astype(jnp.float32), jnp.float32),
                                              stride=stride, length=length, block=block, top_k=2,
                                              queries=queries)
        index = jnp.arange(queries)
        dense = jnp.any((index[None, None, None, None, :] // block == selected[..., None])
                        & (selected[..., None] >= 0), axis=-2) & (index[None, :] <= index[:, None])
        held = jnp.repeat(dense, per, axis=1)
        chosen = scaled_dot_product_attention(query, key, value, mask=held, implementation="xla")
        sliding = scaled_dot_product_attention(query, window_key, window_value, causal=True,
                                               sliding_window=window, implementation="xla")
        branches = jnp.stack([cmp.reshape(rows, queries, heads, -1).astype(chosen.dtype), chosen, sliding],
                             axis=-1)
        return jnp.sum(branches * gates[..., None, :].astype(branches.dtype), axis=-1)

    assert_as_exact(jax.jit(with_gradients(ours)), jax.jit(with_gradients(reference)), draw,
                    "NSA attention's three branches gated", seeds=SEEDS)
