"""Block-selected attention (`dew.nn.block_sparse`): each selection against its definition written out in
numpy, the runs a selection gives against the keys it reads, and the attention over a selection on par with
the reference over its dense mask (check 1)."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
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
            forced = set(range(initial)) | {b for b in range(own - local + 1, own + 1) if b >= 0}
            maxima = np.array([scores[row, group, at, b * block:min((b + 1) * block, at + 1)].max()
                               if b * block <= at else -np.inf for b in range(length // block)])
            maxima[list(forced)] = -np.inf
            picked = {int(b) for b in selected[row, group, at] if b >= 0}
            assert picked == _top(maxima, top_k + 2 - initial - local, forced), (forcing, row, group, at)
    selected = np.asarray(block_sparse.msa_selection(jnp.asarray(scores), block=block, top_k=top_k))

    logits = jnp.asarray(rng.standard_normal((rows, heads * groups, length, length)), jnp.float32)
    with jax.enable_x64(new_val=True):
        loss = np.asarray(block_sparse.msa_indexer_kl(jnp.asarray(scores, jnp.float64),
                                                      jnp.asarray(logits, jnp.float64), jnp.asarray(selected),
                                                      block=block))
    total = 0.0
    for row, group, at in np.ndindex(rows, groups, length):
        blocks = {int(b) for b in selected[row, group, at] if b >= 0}
        tokens = [j for j in range(at + 1) if j // block in blocks]
        heads_of = np.asarray(logits, np.float64)[row, group * heads:(group + 1) * heads, at][:, tokens]
        teacher = (np.exp(heads_of - heads_of.max(-1, keepdims=True))
                   / np.exp(heads_of - heads_of.max(-1, keepdims=True)).sum(-1, keepdims=True)).mean(0)
        student = scores[row, group, at, tokens].astype(np.float64)
        student = student - student.max() - np.log(np.exp(student - student.max()).sum())
        total += float(np.sum(teacher * (np.log(teacher) - student)))
    # The loss run in float64 is the oracle's, to float64 rounding over a row's softmaxes and the sum.
    assert_computes_the_oracle(loss, np.asarray(total / (rows * groups * length)), "MSA's indexer loss",
                               roundings=rows * groups * length * length)

    def trained(params, hidden):
        return block_sparse.msa_indexer_kl(indexer.apply(params, hidden), logits, jnp.asarray(selected),
                                           block=block)

    by_params, by_hidden = jax.grad(trained, argnums=(0, 1))(params, hidden)
    assert np.any(np.asarray(by_params["params"]["wq"]["kernel"]) != 0)
    np.testing.assert_array_equal(np.asarray(by_hidden), 0.0)


def test_msas_selection_keeps_its_slots_and_forces_no_block_past_the_query():
    """MSA's selection (`msa_selection`) holds its `top_k` slots whatever its blocks: over one block of keys,
    the query's own and 15 empty (-1); and forces no block a query does not reach: with the first two blocks
    forced, a query of the first block takes the first alone, the second not yet past it; and none past the
    keys' last block, a query past every key (16 queries over 8 keys) forcing no block the keys hold not."""
    scores = jnp.where(jnp.tril(jnp.ones((16, 16), bool)), 0.0, -jnp.inf)[None, None]
    one = np.asarray(block_sparse.msa_selection(scores, block=16, top_k=16))
    assert one.shape == (1, 1, 16, 16)
    assert all({int(b) for b in one[0, 0, at] if b >= 0} == {0} for at in range(16))
    two = np.asarray(block_sparse.msa_selection(scores, block=8, top_k=4, initial=2, local=1))
    assert {int(b) for b in two[0, 0, 0] if b >= 0} == {0}
    assert {int(b) for b in two[0, 0, 8] if b >= 0} == {0, 1}
    rectangular = np.asarray(block_sparse.msa_selection(scores[..., :8], block=8, top_k=4, local=1))
    assert {int(b) for b in rectangular[0, 0, 8] if b >= 0} == {0}


def test_infllm_attention_past_its_cut_over_every_block_is_on_par_with_the_dense_call():
    """InfLLM-V2's attention past its dense cut whose selection takes every block a query sees (its top as
    many as the blocks) reads every key the dense causal call reads: its output and gradients are on par
    with that call's (check 1), the two branches one function at the cut. An operand laid by the call's
    heads, rows or places (a sink, query slots, key runs) is no argument of either branch."""
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
    laid = {"sinks": jnp.zeros((heads,)), "query_slots": jnp.zeros((rows, queries), jnp.int32),
            "key_runs": jnp.zeros((rows, queries, 1, 2), jnp.int32)}
    for dense_below, (name, operand) in itertools.product((queries + 1, queries), laid.items()):
        with pytest.raises(TypeError):
            block_sparse.infllm_attention(query, key, value, dense_below=dense_below, **{name: operand},
                                          **shape)


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
    (`infllm_attention`'s dense branch is the plain causal call)."""
    infllm = tiny(mixer=mixers.build("block_sparse_attention", selection="infllm", block=4, top_k=2),
                  dtype="float32")
    dense = tiny(mixer=mixers.build("attention"), dtype="float32")
    tokens = jax.random.randint(jax.random.key(0), (2, 16), 0, VOCAB)
    params = dense.init(jax.random.key(1), tokens)
    np.testing.assert_array_equal(np.asarray(infllm.apply(params, tokens)),
                                  np.asarray(dense.apply(params, tokens)))


def test_a_cached_generation_selects_the_blocks_its_full_forward_selects():
    """A decoder of block-selected attention (`block_sparse_attention`, MoBA's) at blocks of two places, a
    query's own and one earlier: a left-padded batch drawn through the cache, each step selecting over the
    cache's keys at the places it keeps, scores each drawn token on par with a fresh full forward over the
    row's unpadded tokens (`assert_generation_scores_as_full_forwards`), over prompts and draws of up to
    seven tokens, four blocks."""
    model = tiny(mixer=mixers.build("block_sparse_attention", block=2, top_k=2), dtype="float32")
    assert_generation_scores_as_full_forwards(model, "block-selected decoding", seeds=DECODED_SEEDS)
