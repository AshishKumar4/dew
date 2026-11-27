"""Block-selected attention (`dew.nn.block_sparse`): each selection against its definition written out in
numpy, the runs a selection gives against the keys it reads, and the attention over a selection on par with
the reference over its dense mask (check 1)."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import selection_margins
from attention_oracle import qkv
from flax import linen as nn
from reference_error import (
    assert_as_exact,
    assert_computes_the_oracle,
    assert_fp32_reduction_bound,
    with_gradients,
)
from test_mixers import VOCAB, tiny
from test_text_rollout_contract import assert_generation_scores_as_full_forwards

from dew.batch import coordinates
from dew.nn import block_sparse, sparse_selection
from dew.nn.attention import scaled_dot_product_attention
from dew.nn.inspection import INDEXER_COLLECTION
from dew.nn.mla import SELECTIONS, indexer_target
from dew.registry import mixers

SEEDS = 8
"""The draws each check-1 site here takes (`reference_error.seeds_for`, sized: 8 each)."""

DECODED_SEEDS = {"block-selected decoding raw": 8, "block-selected decoding policy": 8}
"""The cached generation's check-1 sites' seeds (`reference_error.seeds_for`, sized on armada)."""


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


def _own_runs(rows: int, queries: int, block: int) -> jax.Array:
    """`[B, Q, 2]`: each query's own block of `block` along its row, up to it."""
    index = jnp.arange(queries, dtype=jnp.int32)
    return jnp.broadcast_to(jnp.stack([index // block * block, index + 1], axis=-1), (rows, queries, 2))


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
    bounds = sparse_selection.block_runs(queries, block)
    runs = np.asarray(sparse_selection.selected_runs(jnp.asarray(selected), bounds,
                                                     own=_own_runs(rows, queries, block) if own else None))
    index = jnp.arange(queries)[None, None, :, None]
    seen = np.asarray(jnp.any((index >= runs[:, :, None, :, 0]) & (index < runs[:, :, None, :, 1]), axis=-1))
    np.testing.assert_array_equal(seen, _selected_mask(selected, block, queries, own=own))


def test_moba_selects_each_heads_top_earlier_blocks_by_its_gate():
    """MoBA's selection (`moba_gate`, `moba_selection`): each query head's `top_k - 1` earlier key blocks by
    its query's product with the block's mean key, its group's, beside its own block, which moba_naive.py
    forces into its top k (`MobaChunks.own`, up to the query), written out over numpy (the same blocks)."""
    rows, queries, heads, groups, width, block, top_k = 1, 64, 4, 2, 8, 8, 3
    drawn = qkv((rows, queries, heads, width), jnp.float32, kv_shape=(rows, queries, groups, width))
    query, key, _ = (np.asarray(each, np.float32) for each in drawn)
    chunks = block_sparse.moba_chunks(queries, block)
    selected = np.asarray(block_sparse.moba_selection(
        *block_sparse.moba_gate(jnp.asarray(query), jnp.asarray(key), chunks, block), top_k))
    windows = np.asarray(block_sparse.window_runs(chunks.windows))
    np.testing.assert_array_equal(np.asarray(chunks.own)[0], np.asarray(_own_runs(1, queries, block))[0])
    means = key.reshape(rows, queries // block, block, groups, width).mean(2)
    for head in range(heads):
        for index in range(queries):
            gate = query[0, index, head] @ means[0, :index // block, head // (heads // groups)].T
            expected = {(b * block, (b + 1) * block) for b in np.argsort(-gate)[:top_k - 1].tolist()}
            assert {tuple(windows[0, b]) for b in selected[0, head, index] if b >= 0} == expected


def test_a_holed_rows_queries_read_none_of_its_hidden_keys():
    """A decoder of block-selected attention over a row whose validity hides a key inside a chunk (a hole
    at slot 2 of a chunk of four, `coordinates`' `valid`), whose run spans it: its real queries' logits
    bitwise as they were whatever token the hidden slot holds (train 856's review F1)."""
    model = tiny(mixer=mixers.build("block_sparse_attention", block=4, top_k=2), dtype="float32")
    tokens = jax.random.randint(jax.random.key(0), (1, 12), 0, VOCAB)
    valid = jnp.ones((1, 12), bool).at[0, 2].set(False)
    variables = model.init(jax.random.key(1), tokens)
    forward = jax.jit(lambda row: model.apply(variables, row, coordinates=coordinates(valid=valid)))
    changed = tokens.at[0, 2].set((tokens[0, 2] + 5) % VOCAB)
    real = np.asarray(valid[0])
    np.testing.assert_array_equal(np.asarray(forward(tokens))[0, real], np.asarray(forward(changed))[0, real])


def test_a_masked_selection_takes_each_query_blocks_selected_key_blocks():
    """A block mask's selection (`masked_selection`): each query its query block's selected key blocks, in
    order, at most `most`, -1 after."""
    mask = np.zeros((1, 3, 5), bool)
    mask[0, 0, [0]] = mask[0, 1, [1, 3]] = mask[0, 2, [0, 2, 4]] = True
    selected = np.asarray(block_sparse.masked_selection(jnp.asarray(mask), 4, queries=12, most=3))
    expected = np.repeat(np.asarray([[[0, -1, -1], [1, 3, -1], [0, 2, 4]]]), 4, axis=1)
    np.testing.assert_array_equal(selected, expected)


def test_a_selection_of_no_blocks_attends_to_no_key():
    """A block mask selecting none at a budget of none (`masked_selection`, `most=0`: its selection an empty
    axis) attends to no key: its output zero, and the query's, keys' and values' gradients zero, where the
    selection's empty run axis reached the kernel's reduction over runs and raised."""
    rows, queries, heads, width, block = 1, 16, 2, 16, 4
    mask = jnp.zeros((rows, queries // block, queries // block), bool)
    selected = block_sparse.masked_selection(mask, block, queries=queries, most=0)[:, None]
    shapes = ((rows, queries, heads, width), (rows, queries, 1, width), (rows, queries, 1, width))
    query, key, value = (jnp.asarray(np.random.default_rng(seed).standard_normal(shape), jnp.float32)
                         for seed, shape in enumerate(shapes))

    @jax.jit
    def attended(query, key, value):
        out, pull = jax.vjp(lambda *operands: sparse_selection.selected_attention(
            *operands, selected, sparse_selection.block_runs(queries, block)), query, key, value)
        return out, pull(jnp.ones_like(out))

    out, gradients = attended(query, key, value)
    assert out.shape == (rows, queries, heads, width)
    for array in (out, *gradients):
        np.testing.assert_array_equal(np.asarray(array), np.zeros(array.shape, array.dtype))


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
        return sparse_selection.selected_attention(query, key, value, selected,
                                                   sparse_selection.block_runs(queries, block),
                                                   own=_own_runs(rows, queries, block))

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
    flat = jnp.asarray(query.reshape(rows, keys, groups * per, width))
    selected = np.asarray(block_sparse.infllm_selection(flat, jnp.asarray(key), block=block, top_k=top_k,
                                                        kernel=kernel, stride=stride))
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


def test_infllm_attention_attends_densely_below_its_cut_and_over_its_selection_past_it():
    """InfLLM-V2's attention (`infllm_attention`): a call of fewer keys than its dense cut is the plain causal
    call's output, bit for bit; one of more, each query reading its selected blocks alone
    (`infllm_selection`'s), has its output and every gradient on par with the reference attention over that
    selection's dense mask (check 1)."""
    rows, queries, heads, groups, width = 1, 256, 8, 2, 32
    shape = {"block": 16, "top_k": 3, "kernel": 8, "stride": 4}
    draw = _drawn(rows, queries, heads, groups, width)
    query, key, value, _ = draw(0)
    short = block_sparse.infllm_attention(query, key, value, **shape)
    np.testing.assert_array_equal(np.asarray(short),
                                  np.asarray(scaled_dot_product_attention(query, key, value, causal=True)))

    def ours(query, key, value):
        return block_sparse.infllm_attention(query, key, value, dense_below=queries, **shape)

    def reference(query, key, value):
        # Each query's selected blocks' keys up to its own, a group's mask its query heads'.
        selected = jax.lax.stop_gradient(block_sparse.infllm_selection(query, key, **shape))
        place = jnp.arange(queries)
        chosen = jnp.any(selected[..., None] == (place // shape["block"]), axis=-2)
        dense = jnp.repeat(chosen & (place[None, :] <= place[:, None]), heads // groups, axis=1)
        return scaled_dot_product_attention(query, key, value, mask=dense, implementation="xla")

    assert_as_exact(jax.jit(with_gradients(ours)), jax.jit(with_gradients(reference)), draw,
                    "InfLLM-V2's attention past its dense cut", seeds=SEEDS)


def test_msas_index_branch_scores_selects_and_trains_as_its_equations_write_them():
    """MSA's index branch (arXiv 2606.13392 eqs. 5-7, 9-11), each against its equation written out in numpy:
    the scores of each group's index query against the one shared index key (scaled, -inf past the query),
    the selection of each group's top blocks with the query's own forced, and the KL loss of the indexer's
    softmax from its group's mean head softmax over the visible selected tokens. Its loss's gradient
    reaches the index projections alone: none flows to the hidden states (eq. 11's detach)."""
    rows, length, embed, groups, width, block, top_k, heads = 2, 64, 24, 2, 8, 8, 3, 4
    rng = np.random.default_rng(0x59)
    hidden = jnp.asarray(rng.standard_normal((rows, length, embed)), jnp.float32)
    indexer = block_sparse.MsaIndexer(groups=groups, width=width)
    params = nn.unbox(indexer.init(jax.random.key(0), hidden))
    scores = np.asarray(indexer.apply(params, hidden))
    wq = np.asarray(params["params"]["wq"]["kernel"], np.float64)
    wk = np.asarray(params["params"]["wk"]["kernel"], np.float64)
    query = (np.asarray(hidden, np.float64) @ wq).reshape(rows, length, groups, width)
    key = np.asarray(hidden, np.float64) @ wk
    expected = np.einsum("bngd,bmd->bgnm", query, key) / np.sqrt(width)
    causal = np.tril(np.ones((length, length), bool))
    # Each score's products' magnitudes through both projections' (their own reductions' rounding carried).
    absolute = np.abs(np.asarray(hidden, np.float64))
    magnitudes = np.einsum("bngd,bmd->bgnm", (absolute @ np.abs(wq)).reshape(query.shape),
                           absolute @ np.abs(wk)) / np.sqrt(width)
    assert_fp32_reduction_bound(np.where(causal, scores, 0.0), np.where(causal, expected, 0.0),
                                np.where(causal, magnitudes, 0.0), embed + width + 1)
    assert np.all(np.isneginf(scores[..., ~causal]))

    for initial, local, forcing in ((0, 1, "MSA's own block"), (1, 2, "M3's first and two local")):
        selected = np.asarray(block_sparse.msa_selection(jnp.asarray(scores), block=block, top_k=top_k + 2,
                                                         initial=initial, local=local))
        assert selected.shape == (rows, groups, length, top_k + 2), forcing
        for row, group, at in np.ndindex(rows, groups, length):
            own = at // block
            # The forced blocks it reaches, each once, ranked ahead of the rest within its one top.
            forced = {b for b in set(range(initial)) | set(range(own - local + 1, own + 1)) if 0 <= b <= own}
            maxima = np.array([scores[row, group, at, b * block:min((b + 1) * block, at + 1)].max()
                               if b * block <= at else -np.inf for b in range(length // block)])
            maxima[list(forced)] = -np.inf
            slots = [int(b) for b in selected[row, group, at] if b >= 0]
            assert len(slots) == len(set(slots)), (forcing, row, group, at)
            assert set(slots) == _top(maxima, top_k + 2 - len(forced), forced), (forcing, row, group, at)
    selected = np.asarray(block_sparse.msa_selection(jnp.asarray(scores), block=block, top_k=top_k))

    main = 16
    query = jnp.asarray(rng.standard_normal((rows, length, heads * groups, main)), jnp.float32)
    key = jnp.asarray(rng.standard_normal((rows, length, groups, main)), jnp.float32)
    scale = main ** -0.5
    with jax.enable_x64(new_val=True):
        kls = np.asarray(block_sparse.msa_query_kls(
            jnp.asarray(scores, jnp.float64), jnp.asarray(query, jnp.float64), jnp.asarray(key, jnp.float64),
            jnp.asarray(selected), block=block, scale=scale))
    expected = np.zeros((rows, length))
    for row, group, at in np.ndindex(rows, groups, length):
        blocks = {int(b) for b in selected[row, group, at] if b >= 0}
        tokens = [j for j in range(at + 1) if j // block in blocks]
        heads_of = np.asarray(query, np.float64)[row, at, group * heads:(group + 1) * heads] @ np.asarray(
            key, np.float64)[row, tokens, group].T * scale
        summed = (np.exp(heads_of - heads_of.max(-1, keepdims=True))
                  / np.exp(heads_of - heads_of.max(-1, keepdims=True)).sum(-1, keepdims=True)).sum(0)
        teacher = summed / summed.sum()
        student = scores[row, group, at, tokens].astype(np.float64)
        student = student - student.max() - np.log(np.exp(student - student.max()).sum())
        expected[row, at] += float(np.sum(teacher * (np.log(teacher) - student))) / groups
    # The loss run in float64 is the oracle's, to float64 rounding over a query's softmaxes and its sum.
    assert_computes_the_oracle(kls, expected, "MSA's indexer loss per query",
                               roundings=heads * length * main)

    def trained(params, hidden):
        return block_sparse.msa_query_kls(indexer.apply(params, hidden), query, key, jnp.asarray(selected),
                                          block=block, scale=scale).sum()

    by_params, by_hidden = jax.grad(trained, argnums=(0, 1))(params, hidden)
    assert np.any(np.asarray(by_params["params"]["wq"]["kernel"]) != 0)
    np.testing.assert_array_equal(np.asarray(by_hidden), 0.0)


def test_msas_indexer_loss_holds_a_token_its_branch_scores_far_below_the_attention_or_at_fp32s_range():
    """MSA's indexer loss (`msa_query_kls`) over a query's two selected tokens the attention weighs alike:
    where its index branch scores one 100 below the other, its KL is 50 - log 2 (the float64 run, as a
    float64 oracle writes it) and its gradient on the scores exactly (1/2, -1/2) in fp32, where an epsilon
    inside the logarithm (MaxText's, `mla.indexer_kl`'s) capped the loss near 8.5 and stilled the low
    score's gradient; scored 64 apart at 2^30, its exact KL rounded to fp32; scored fp32's largest and
    least, its KL fp32's largest (its exact value rounds there), not an infinity, over one group and as
    the mean of two, and as the mean of one group's KL past fp32's range and another's 0; a teacher's least
    normal mass at a gap of 2^126, its term held within fp32's rounding; and over one
    selected token beside a key it does not select, nothing, the selected token's score fp32's least."""
    query = jnp.zeros((1, 2, 1, 4), jnp.float32)
    key = jnp.zeros((1, 2, 1, 4), jnp.float32)
    selected = jnp.zeros((1, 1, 2, 1), jnp.int32)
    largest = np.finfo(np.float32).max

    def loss(scores, query=query, key=key):
        return block_sparse.msa_query_kls(scores, query, key, selected, block=2, scale=0.5)

    low = [[[[0.0, -np.inf], [0.0, -100.0]]]]
    with jax.enable_x64(new_val=True):
        wide = (jnp.asarray(low, jnp.float64), query.astype(jnp.float64), key.astype(jnp.float64))
        kls = np.asarray(loss(*wide))
    expected = np.asarray([[0.0, 50.0 - np.log(2.0) + np.log1p(np.exp(-100.0))]])
    assert_computes_the_oracle(kls, expected, "MSA's indexer loss at a token scored 100 below", roundings=8)
    gradient = np.asarray(jax.grad(lambda s: loss(s)[0, 1])(jnp.asarray(low, jnp.float32)))[0, 0, 1]
    np.testing.assert_array_equal(gradient, np.asarray([0.5, -0.5], np.float32))

    # A gap of 64 at 2^30, which a weighted mean of the scores rounds away (2^30 - 32 a tie, to 2^30): the
    # exact KL, 32 - log 2 + log(1 + e^-64), rounded to fp32 (a quarter of its unit from it).
    near = jnp.asarray([[[[0.0, -np.inf], [2.0**30, 2.0**30 - 64]]]], jnp.float32)
    np.testing.assert_array_equal(np.asarray(loss(near))[0, 1], np.float32(32 - np.log(2.0)))
    # At fp32's largest and least, over one group and over two, each group's KL fp32's largest.
    for groups in (1, 2):
        def grouped(scores, groups=groups):
            return block_sparse.msa_query_kls(
                scores, jnp.zeros((1, 2, groups, 4), jnp.float32), jnp.zeros((1, 2, groups, 4), jnp.float32),
                jnp.zeros((1, groups, 2, 1), jnp.int32), block=2, scale=0.5)

        extremes = jnp.asarray([[[[0.0, -np.inf], [largest, -largest]]] * groups], jnp.float32)
        np.testing.assert_array_equal(np.asarray(grouped(extremes))[0],
                                      np.asarray([0.0, largest], np.float32))
    # One group's KL past fp32's range (2 F, its teacher all on the token scored -F) beside a group's 0: their
    # mean F, at its share before the group's own rounds.
    point = jnp.zeros((1, 2, 2, 1), jnp.float32).at[0, 1, 0, 0].set(1.0)
    keys = jnp.zeros((1, 2, 2, 1), jnp.float32).at[0, 0, 0, 0].set(-128.0)
    apart = jnp.asarray([[[[0.0, -np.inf], [largest, -largest]], [[0.0, -np.inf], [0.0, 0.0]]]], jnp.float32)
    beside = block_sparse.msa_query_kls(apart, point, keys, jnp.zeros((1, 2, 2, 1), jnp.int32), block=2,
                                        scale=1.0)
    np.testing.assert_array_equal(np.asarray(beside)[0], np.asarray([0.0, largest], np.float32))
    # A teacher's least normal mass (e^-87 of a group's two keys) at a gap of 2^126 beside a group's 0: its
    # term (~1.4, half of it the mean's) held, not flushed where a share of the mass, subnormal, took it.
    rare = jnp.zeros((1, 2, 2, 1), jnp.float32).at[0, 1, 0, 0].set(1.0)
    far = jnp.zeros((1, 2, 2, 1), jnp.float32).at[0, 1, 0, 0].set(-87.0)
    gapped = jnp.asarray([[[[0.0, -np.inf], [0.0, -2.0**126]], [[0.0, -np.inf], [0.0, 0.0]]]], jnp.float32)
    found = block_sparse.msa_query_kls(gapped, rare, far, jnp.zeros((1, 2, 2, 1), jnp.int32), block=2,
                                       scale=1.0)
    mass = np.exp(-87.0) / (1 + np.exp(-87.0))
    terms = np.asarray([(1 - mass) * np.log1p(-mass), mass * np.log(mass), mass * 2.0**126]) / 2
    assert_fp32_reduction_bound(np.asarray(found)[0, 1], terms.sum(), np.abs(terms).sum(), len(terms) + 2)
    # A teacher's mass of e^-110, below fp32's least subnormal, at the same gap: its term 1.44e-10, a normal
    # fp32, met with the gap in their logarithms, where the mass rounded to 0 first held nothing (train 1345's
    # F2).
    farther = jnp.zeros((1, 2, 2, 1), jnp.float32).at[0, 1, 0, 0].set(-110.0)
    found = block_sparse.msa_query_kls(gapped, rare, farther, jnp.zeros((1, 2, 2, 1), jnp.int32), block=2,
                                       scale=1.0)
    mass = np.exp(-110.0) / (1 + np.exp(-110.0))
    terms = np.asarray([(1 - mass) * np.log1p(-mass), mass * np.log(mass), mass * 2.0**126]) / 2
    assert_fp32_reduction_bound(np.asarray(found)[0, 1], terms.sum(), np.abs(terms).sum(), len(terms) + 2)
    # The same mass against scores fp32's largest and least, a gap past fp32's range: its term 2 F e^-110,
    # its weight and its score's shares met in their logarithms.
    spanned = jnp.asarray([[[[0.0, -np.inf], [largest, -largest]], [[0.0, -np.inf], [0.0, 0.0]]]],
                          jnp.float32)
    found = block_sparse.msa_query_kls(spanned, rare, farther, jnp.zeros((1, 2, 2, 1), jnp.int32), block=2,
                                       scale=1.0)
    terms = np.asarray([(1 - mass) * np.log1p(-mass), mass * np.log(mass), mass * 2.0 * float(largest)]) / 2
    assert_fp32_reduction_bound(np.asarray(found)[0, 1], terms.sum(), np.abs(terms).sum(), len(terms) + 2)
    # The first query selects its own token alone (the second is past it), scored fp32's least.
    hidden = jnp.asarray([[[[-largest, 0.0], [0.0, 0.0]]]], jnp.float32)
    assert np.asarray(loss(hidden))[0, 0] == 0.0


def test_msas_indexer_loss_at_a_teachers_tiny_mass_its_invalid_logit_and_an_infinitely_low_score():
    """MSA's indexer loss (`msa_query_kls`) over two selected tokens whose teacher logits are 0 and -88 (a
    mass below fp32's least normal): its branch scoring them the same, KL 0 within fp32's rounding of its
    terms (88 e^-88, normal), the entropy's tiny term formed as the gap's, where a flushed weight dropped
    it; scored 0 and -inf against logits 0 and -110, its KL +inf and the low score's gradient the weight's
    (0 in fp32), not the infinity's own slope, and scored -2^126 under a large cotangent, the weight's
    product normal; a teacher logit of NaN, its KL and gradient NaN, not a finite loss over the
    rest."""
    query = jnp.ones((1, 2, 1, 1), jnp.float32)
    selected = jnp.zeros((1, 1, 2, 1), jnp.int32)

    def loss(scores, key):
        return block_sparse.msa_query_kls(scores, query, key, selected, block=2, scale=1.0)[0, 1]

    key = jnp.asarray([[[[0.0]], [[-88.0]]]], jnp.float32)
    equal = jnp.asarray([[[[0.0, -np.inf], [0.0, -88.0]]]], jnp.float32)
    assert_fp32_reduction_bound(np.asarray(loss(equal, key)), 0.0, 2 * 88 * np.exp(-88.0), 4)
    # Teacher logits 0 and -110: scored 0 and -inf, the low score's gradient the weight's, -e^-110, 0 in
    # fp32, the high score's 1 less its weight's 1, 0; scored 0 and -2^126 under a cotangent of 2^126, the
    # low score's -2^126 e^-110 (1.44e-10, normal), formed between the weight's halves.
    farther = jnp.asarray([[[[0.0]], [[-110.0]]]], jnp.float32)
    infinite = jnp.asarray([[[[0.0, -np.inf], [0.0, -np.inf]]]], jnp.float32)
    assert float(loss(infinite, farther)) == np.inf
    gradient = np.asarray(jax.grad(lambda s: loss(s, farther))(infinite))[0, 0, 1]
    np.testing.assert_array_equal(gradient, np.zeros(2, np.float32))
    far = jnp.asarray([[[[0.0, -np.inf], [0.0, -(2.0**126)]]]], jnp.float32)
    gradient = np.asarray(jax.grad(lambda s: loss(s, farther) * 2.0**126)(far))[0, 0, 1, 1]
    expected = -(2.0**126) * np.exp(-110.0) / (1 + np.exp(-110.0))
    assert_fp32_reduction_bound(gradient, expected, abs(expected), 3)
    # Teacher logits 0 and -176 against scores 0 and fp32's least: the term fp32's largest e^-176, 1.25e-38,
    # normal, though the weight's half e^-88 is not (a flushing backend's 0).
    farthest = jnp.asarray([[[[0.0]], [[-176.0]]]], jnp.float32)
    largest = float(np.finfo(np.float32).max)
    spanned = jnp.asarray([[[[0.0, -np.inf], [0.0, -largest]]]], jnp.float32)
    mass = np.exp(-176.0)
    terms = np.asarray([mass * np.log(mass), mass * largest])
    assert_fp32_reduction_bound(np.asarray(loss(spanned, farthest)), terms.sum(), np.abs(terms).sum(), 4)
    # Teacher logits 0 and -176.5 against scores fp32's largest and least, a gap past fp32's range: the
    # term 2 F e^-176.5 normal though its halved gap's is not; and teacher 0 and -110 under a cotangent of
    # 2^127, the low score's cotangent -2^127 e^-110 normal, the factor of the halved gap inside the chain.
    farther_still = jnp.asarray([[[[0.0]], [[-176.5]]]], jnp.float32)
    spanning = jnp.asarray([[[[0.0, -np.inf], [largest, -largest]]]], jnp.float32)
    mass = np.exp(-176.5)
    terms = np.asarray([mass * np.log(mass), mass * 2.0 * largest])
    found = np.asarray(loss(spanning, farther_still))
    assert_fp32_reduction_bound(found, terms.sum(), np.abs(terms).sum(), 4)
    gradient = np.asarray(jax.grad(lambda s: loss(s, farther) * 2.0**127)(spanning))[0, 0, 1, 1]
    expected = -(2.0**127) * np.exp(-110.0) / (1 + np.exp(-110.0))
    assert_fp32_reduction_bound(gradient, expected, abs(expected), 4)
    invalid = jnp.asarray([[[[0.0]], [[np.nan]]]], jnp.float32)
    uniform = jnp.zeros((1, 1, 2, 2), jnp.float32)
    assert np.isnan(float(loss(uniform, invalid)))
    assert np.isnan(np.asarray(jax.grad(lambda s: loss(s, invalid))(uniform))[0, 0, 1]).all()


def test_infllm_v2s_selection_spends_no_pick_on_a_block_of_hidden_keys():
    """InfLLM-V2's selection (`infllm_selection`) over five blocks of 16 keys, the second block's every key
    hidden (`key_valid`), its real blocks' pooled logits negative: the last query's one optional pick is a
    real block, never the hidden one, whose windows score nothing."""
    rows, keys, width = 1, 80, 8
    query = -jnp.ones((rows, keys, 1, width), jnp.float32)
    key = jnp.ones((rows, keys, 1, width), jnp.float32)
    valid = jnp.ones((rows, keys), bool).at[:, 16:32].set(False)
    picks = block_sparse.infllm_selection(query, key, block=16, top_k=1, kernel=16, stride=16, initial=1,
                                          local=1, key_valid=valid)
    assert int(picks[0, 0, -1, 0]) in (2, 3)


def test_msas_teacher_takes_the_attentions_precision_not_the_ambient_one():
    """MSA's indexer loss (`msa_query_kls`) at the layer's precision, `highest`, under an ambient bf16
    product preset (`BF16_BF16_F32`, whose operands Dew rounds to bf16 on any backend): its teacher's logits 4096 and 4097 (a query of 1 against keys 4096 and 4097) stay apart, so a
    uniform student's KL is the float64 run's, where the ambient preset rounded both to 4096 and gave 0
    (row 13706)."""
    query = jnp.ones((1, 2, 1, 1), jnp.float32)
    key = jnp.asarray([[[[4096.0]], [[4097.0]]]], jnp.float32)
    selected = jnp.zeros((1, 1, 2, 1), jnp.int32)
    uniform = jnp.zeros((1, 1, 2, 2), jnp.float32)
    with jax.default_matmul_precision("BF16_BF16_F32"):
        found = np.asarray(block_sparse.msa_query_kls(uniform, query, key, selected, block=2, scale=1.0,
                                                      precision="highest"))[0, 1]
    target = np.exp([4096.0 - 4097.0, 0.0])
    target /= target.sum()
    expected = float(np.sum(target * np.log(target)) + np.log(2.0))
    assert_fp32_reduction_bound(found, expected, 2 * np.log(2.0), 4)


def test_the_indexer_teacher_gives_a_hidden_key_no_share_of_a_heads_mass():
    """The indexer's teacher (`mla.indexer_target`, MSA's and the lightning indexers'): each head's softmax
    over the keys its call allows alone. Two fp32 heads over two allowed keys and a hidden one, the first
    head's allowed logits fp32's least both, the second's 0 and -128: the heads' masses 1.5 and 0.5 within
    fp32's rounding, the hidden key none, where a hidden key given fp32's least logit tied the first head's
    and took a third of it (4/3 and 1/3, train 1345's F1); a row that allows no key, each head's uniform
    over every key, the references' masked softmax's."""
    least = float(np.finfo(np.float32).min)
    query = jnp.ones((1, 2, 2, 1), jnp.float32)
    key = jnp.asarray([[[[least], [0.0]], [[least], [-128.0]], [[least], [0.0]]]], jnp.float32)
    keep = jnp.asarray([[[True, True, False], [False, False, False]]])
    target = np.asarray(indexer_target(query, key, keep, 1.0))
    for found, exact in ((target[0, 0, 0], 1.5), (target[0, 0, 1], 0.5)):
        assert_fp32_reduction_bound(found, exact, exact, 8)
    np.testing.assert_array_equal(target[0, 0, 2], 0.0)
    for found in target[0, 1]:
        assert_fp32_reduction_bound(found, 2 / 3, 2 / 3, 8)


def test_msas_selection_keeps_its_slots_and_forces_each_block_once_within_reach():
    """MSA's selection (`msa_selection`), its raw slots: over one block of keys, the query's own once and
    the rest empty (-1) whatever its `top_k`; M3's forcing (the first block and the two ending at its own)
    for a query of the first block, its own once, not the first and its own as two (row 13501); a top of
    one with its own forced alone, its own; with two first blocks forced, a query of the first block its
    first alone, the second not yet past it; no block past the keys' last (16 queries over 8 keys); and its
    own forced ahead of blocks scored +inf."""
    scores = jnp.where(jnp.tril(jnp.ones((16, 16), bool)), 0.0, -jnp.inf)[None, None]
    one = np.asarray(block_sparse.msa_selection(scores, block=16, top_k=16))
    np.testing.assert_array_equal(one[0, 0], np.array([[0] + [-1] * 15] * 16))
    m3 = np.asarray(block_sparse.msa_selection(scores, block=8, top_k=4, initial=1, local=2))
    np.testing.assert_array_equal(m3[0, 0, 0], [0, -1, -1, -1])
    np.testing.assert_array_equal(m3[0, 0, 8], [0, 1, -1, -1])
    alone = np.asarray(block_sparse.msa_selection(scores, block=8, top_k=1, initial=0, local=1))
    np.testing.assert_array_equal(alone[0, 0, :, 0], [0] * 8 + [1] * 8)
    two = np.asarray(block_sparse.msa_selection(scores, block=8, top_k=4, initial=2, local=1))
    np.testing.assert_array_equal(two[0, 0, 0], [0, -1, -1, -1])
    rectangular = np.asarray(block_sparse.msa_selection(scores[..., :8], block=8, top_k=4, local=1))
    np.testing.assert_array_equal(rectangular[0, 0, 8], [0, -1, -1, -1])
    # The forced bit ahead of any score: blocks 0 and 1 scored +inf, its own (2) finite, a top of two.
    infinite = jnp.where(jnp.tril(jnp.ones((24, 24), bool)),
                         jnp.where(jnp.arange(24) < 16, jnp.inf, 0.0), -jnp.inf)[None, None]
    tied = np.asarray(block_sparse.msa_selection(infinite, block=8, top_k=2, initial=0, local=1))
    np.testing.assert_array_equal(tied[0, 0, 16], [2, 0])


def test_infllm_attention_past_its_cut_over_every_block_is_on_par_with_the_dense_call():
    """InfLLM-V2's attention past its dense cut whose selection takes every block a query sees (its top as
    many as the blocks) reads every key the dense causal call reads: its output and gradients are on par
    with that call's (check 1), the two branches one function at the cut. An operand laid by the call's
    places (query slots, key runs) is no argument of either branch."""
    rows, queries, heads, groups, width = 1, 128, 8, 2, 32
    shape = {"block": 16, "top_k": queries // 16, "kernel": 8, "stride": 4}
    draw = _drawn(rows, queries, heads, groups, width)

    def selected(query, key, value):
        return block_sparse.infllm_attention(query, key, value, dense_below=queries, **shape)

    def dense(query, key, value):
        return scaled_dot_product_attention(query, key, value, causal=True, implementation="xla")

    assert_as_exact(jax.jit(with_gradients(selected)), jax.jit(with_gradients(dense)), draw,
                    "InfLLM-V2's attention over every block", seeds=SEEDS)
    query, key, value, _ = draw(0)
    laid = {"query_slots": jnp.zeros((rows, queries), jnp.int32),
            "key_runs": jnp.zeros((rows, queries, 1, 2), jnp.int32)}
    for dense_below, (name, operand) in itertools.product((queries + 1, queries), laid.items()):
        with pytest.raises(TypeError):
            block_sparse.infllm_attention(query, key, value, dense_below=dense_below, **{name: operand},
                                          **shape)


def test_infllm_attention_takes_each_rows_cut_at_its_real_keys():
    """InfLLM-V2's attention over right-padded rows (`key_valid`) past its cut by the call's 80 keys: a row of
    60 real keys, below the cut of 64, attends densely as its 60 tokens do alone, and a row of 80 by its
    selection as its tokens do unpadded, its output and gradients on par with theirs (check 1), each row's
    real queries; a head's sinks join both branches (train 1345's F3 and F4)."""
    rows, queries, heads, groups, width = 2, 80, 4, 2, 32
    shape = {"block": 16, "top_k": 1, "kernel": 8, "stride": 4, "dense_below": 64}
    real = 60
    valid = jnp.arange(queries)[None, :] < jnp.asarray([[real], [queries]])
    sinks = jnp.asarray([0.5, -1.0, 2.0, 0.0], jnp.float32)
    draw = _drawn(rows, queries, heads, groups, width)

    def drawn(seed: int):
        # Each row's own output's cotangent: the first's over its real queries.
        query, key, value, dout = draw(seed)
        return query, key, value, (dout[0, :real], dout[1])

    def ours(query, key, value):
        out = block_sparse.infllm_attention(query, key, value, key_valid=valid, sinks=sinks, **shape)
        return out[0, :real], out[1]

    def alone(query, key, value):
        short = block_sparse.infllm_attention(query[:1, :real], key[:1, :real], value[:1, :real], sinks=sinks,
                                              implementation="xla", **shape)
        long = block_sparse.infllm_attention(query[1:], key[1:], value[1:], sinks=sinks,
                                             implementation="xla", **shape)
        return short[0], long[0]

    assert_as_exact(jax.jit(with_gradients(ours)), jax.jit(with_gradients(alone)), drawn,
                    "InfLLM-V2's cut at each row's real keys", seeds=SEEDS)


@pytest.mark.parametrize("selection", ["moba", "infllm"])
@pytest.mark.parametrize("sinks", ["attention_sinks", "projected_sinks"])
def test_a_block_sparse_decoders_sinks_reach_its_softmax(selection, sinks):
    """A decoder of block-selected attention with sinks (a head's learned, or a query's own projected from
    its input) reads them in its softmax: its logits move with them (train 1345's F4: they were dropped)."""
    options = {"projected_sinks": True} if sinks == "projected_sinks" else {}
    mixer = mixers.build("block_sparse_attention", selection=selection, block=16, top_k=1, dense_below=16,
                         **options)
    model = tiny(mixer=mixer, dtype="float32", **({"attention_sinks": True} if not options else {}))
    tokens = jax.random.randint(jax.random.key(0), (1, 32), 0, VOCAB)
    params = model.init(jax.random.key(1), tokens)
    moved = jax.tree_util.tree_map_with_path(
        lambda path, leaf: leaf + 3.0 if "sink" in jax.tree_util.keystr(path) else leaf, params)
    assert not np.array_equal(np.asarray(model.apply(params, tokens)), np.asarray(model.apply(moved, tokens)))


@pytest.mark.parametrize("sinks", ["attention_sinks", "projected_sinks"])
def test_an_msa_decoder_refuses_sinks(sinks):
    """An MSA decoder with sinks is refused at build: its indexer's loss targets its heads' plain softmax
    (`msa_query_kls`), which the sinks' would part from."""
    options = {"projected_sinks": True} if sinks == "projected_sinks" else {}
    mixer = mixers.build("block_sparse_attention", selection="msa", block=4, top_k=1, index_width=8,
                         **options)
    with pytest.raises(ValueError):
        tiny(mixer=mixer, **({"attention_sinks": True} if not options else {})).init(
            jax.random.key(1), jnp.zeros((1, 16), jnp.int32))


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


def test_an_infllm_decoder_below_its_cut_is_the_dense_decoder():
    """A decoder of block-selected attention by InfLLM-V2's selection (`block_sparse_attention`,
    `selection="infllm"`) over a row shorter than the selection's dense cut (8192 keys) attends densely:
    its logits are the same model's under causal attention's mixer, its parameters the same, bit for bit
    (`infllm_attention`'s dense branch is the plain causal call). Past its cut the mixer runs its selected
    branch, which reads fewer keys than the dense decoder; a block no 16 places divide it refuses. A
    right-padded row (`training.md`'s) it takes, its real queries' logits bitwise as they were whatever token
    its padding holds (train 1345's F3: it was refused)."""
    infllm = tiny(mixer=mixers.build("block_sparse_attention", selection="infllm", block=16, top_k=2),
                  dtype="float32")
    dense = tiny(mixer=mixers.build("attention"), dtype="float32")
    tokens = jax.random.randint(jax.random.key(0), (2, 64), 0, VOCAB)
    params = dense.init(jax.random.key(1), tokens)
    np.testing.assert_array_equal(np.asarray(infllm.apply(params, tokens)),
                                  np.asarray(dense.apply(params, tokens)))
    # Past its cut (64 keys over a cut of 32), each query reads its selected blocks alone: the first, its
    # own and its top one of those between, not the dense decoder's every key.
    sparse = tiny(mixer=mixers.build("block_sparse_attention", selection="infllm", block=16, top_k=1,
                                     dense_below=32), dtype="float32")
    selected = np.asarray(sparse.apply(params, tokens))
    assert not np.array_equal(selected, np.asarray(dense.apply(params, tokens)))
    with pytest.raises(ValueError):
        tiny(mixer=mixers.build("block_sparse_attention", selection="infllm", block=4)).init(
            jax.random.key(1), tokens)
    padded = jnp.ones(tokens.shape, bool).at[:, -1].set(False)
    forward = jax.jit(lambda row: sparse.apply(params, row, coordinates=coordinates(valid=padded,
                                                                                    right_padded=True)))
    changed = tokens.at[:, -1].set((tokens[:, -1] + 5) % VOCAB)
    np.testing.assert_array_equal(np.asarray(forward(tokens))[:, :-1], np.asarray(forward(changed))[:, :-1])


def test_an_msa_decoder_selects_by_its_index_branch_and_trains_it_by_its_own_loss_alone():
    """A decoder of block-selected attention by MSA's selection (`block_sparse_attention`,
    `selection="msa"`): its index branch's parameters are the model's (`msa_indexer`), it sows its index
    scores and its selection (`SELECTIONS`) and, where an objective trains an indexer, its loss per query
    (`msa_query_kls`, `INDEXER_COLLECTION`; an indexed mixer), and the main loss's
    gradient leaves the index branch untouched: its input is read detached, the selection taking none. Its
    selection is sown as every selection is (`mla.sow_selection`'s record), which the selection gate reads
    (`selection_margins`): an MSA decoder's config selects and its margin is finite."""
    model = tiny(mixer=mixers.build("block_sparse_attention", selection="msa", block=4, top_k=2,
                                    index_width=8), dtype="float32")
    tokens = jax.random.randint(jax.random.key(0), (1, 16), 0, VOCAB)
    params = model.init(jax.random.key(1), tokens)

    def named(tree, part):
        return [leaf for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]
                if part in "/".join(str(key) for key in path)]

    assert named(params, "msa_indexer")
    _, sown = model.apply(params, tokens, mutable=[SELECTIONS, INDEXER_COLLECTION])
    assert named(sown, "scores") and named(sown, "visible") and named(sown, "budget")
    # The gate asks the model's own config, its mixer typed, and of a hybrid's layer kind alike.
    config = {"mixer": model.mixer}
    assert selection_margins.selects(config)
    assert selection_margins.selects({"mixer": None, "kinds": {"sparse": {"mixer": model.mixer}}})
    captured = selection_margins.capture(model)(params, tokens)
    assert np.isfinite(selection_margins.margin(config, captured, "msa"))
    # Its indexer's loss per query, each layer's, where an objective training an indexer opens its
    # collection.
    kls = named(sown, "kl")
    assert kls and all(kl.shape == (1, 16) for kl in kls) and model.indexed_mixers
    grads = jax.grad(lambda p: jnp.sum(model.apply(p, tokens)))(params)
    for leaf in named(grads, "msa_indexer"):
        np.testing.assert_array_equal(np.asarray(leaf), 0.0)


def test_an_msa_decoder_of_a_top_of_one_selects_each_querys_own_block_alone():
    """A decoder of block-selected attention by MSA's selection at a top of one (`top_k=1`, row 13527): it
    builds and runs, sowing a budget of no block past its own, and its selection's one slot is each query's
    own block (`msa_selection`'s forcing) whatever the scores."""
    model = tiny(mixer=mixers.build("block_sparse_attention", selection="msa", block=4, top_k=1,
                                    index_width=8), dtype="float32")
    tokens = jax.random.randint(jax.random.key(0), (1, 16), 0, VOCAB)
    params = model.init(jax.random.key(1), tokens)
    logits, sown = model.apply(params, tokens, mutable=[SELECTIONS])
    assert np.all(np.isfinite(np.asarray(logits)))
    budgets = [leaf for path, leaf in jax.tree_util.tree_flatten_with_path(sown)[0]
               if "budget" in "/".join(str(key) for key in path)]
    assert budgets and all(int(budget) == 0 for budget in budgets)
    scores = jax.random.normal(jax.random.key(2), (1, 2, 16, 16), jnp.float32)
    scores = jnp.where(jnp.arange(16)[None, :] <= jnp.arange(16)[:, None], scores, -jnp.inf)
    first = np.asarray(block_sparse.msa_selection(scores, block=4, top_k=1))[..., 0]
    np.testing.assert_array_equal(first, np.broadcast_to(np.arange(16) // 4, first.shape))


def test_an_msa_decoder_refuses_a_logit_softcap():
    """An MSA decoder under a logit softcap is refused at build: its indexer's loss targets its heads' plain
    softmax (`msa_query_kls`), which the softcapped attention it would train toward parts from."""
    with pytest.raises(ValueError):
        tiny(mixer=mixers.build("block_sparse_attention", selection="msa", block=4, top_k=1, index_width=8),
             attn_logit_softcap=30.0).init(jax.random.key(1), jnp.zeros((1, 16), jnp.int32))


@pytest.mark.parametrize(("selection", "refused"), [("moba", False), ("infllm", True), ("msa", True)])
def test_generation_admits_a_block_sparse_decoder_whose_selection_reads_a_cache(selection, refused):
    """Generation admits a decoder of block-selected attention whose selection reads a cache, MoBA's
    (`generation_refusal` None), and refuses InfLLM-V2's and MSA's, which read a row's own tokens uncached
    (row 13530)."""
    model = tiny(mixer=mixers.build("block_sparse_attention", selection=selection, block=16, top_k=2),
                 dtype="float32")
    assert (model.generation_refusal() is not None) == refused


def test_a_cached_generation_selects_the_blocks_its_full_forward_selects():
    """A decoder of block-selected attention (`block_sparse_attention`, MoBA's) at blocks of two places, a
    query's own and one earlier: a left-padded batch drawn through the cache, each step selecting over the
    cache's keys at the places it keeps, scores each drawn token on par with a fresh full forward over the
    row's unpadded tokens (`assert_generation_scores_as_full_forwards`), over prompts and draws of up to
    seven tokens, four blocks."""
    model = tiny(mixer=mixers.build("block_sparse_attention", block=2, top_k=2), dtype="float32")
    assert_generation_scores_as_full_forwards(model, "block-selected decoding", seeds=DECODED_SEEDS)
