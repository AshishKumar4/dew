"""K-pool sparse attention against transformers 5.16.1's `Glm5NextTextAttention`.

The reference block runs live here on random weights (tools/hf_reference.py's
`scatter_weights`), a tiny NoPE config with pools of two keys and a budget of
two pools, over eleven tokens so the tail alternates between empty and one
token, one row of the batch left-padded through the attention mask.
Everything runs at fp32 on CPU; the bound is 1e-4 scaled
(max|ours - theirs| / max|theirs|) on each comparison.

The indexer's scores are relu'd sums, so two pools tie at exactly zero
often; the reference's `topk` and `jax.lax.top_k` break such a tie
differently, which is not a port error. `boundary_gap` reads the reference's
own scores and asserts that no valid query's selection is decided by a tie
at the weights and inputs below, so a failure here is a difference in the
math and not in a tie-break.

Observed on CPU, all scaled: output 2.2e-07 (1.6e-07 with biased
projections), the input gradient 1.9e-07 and the parameter gradients
4.4e-07 at worst, decode 1.6e-07 unpadded and 2.2e-07 with a padded row;
the dense softmax over the same causal keys sits 0.35 from the sparse
output. The selection margin at these weights and inputs is 1.5e-02.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextAttention

from dew.nn.dsa_kpool import KPoolSparseAttention, KPoolSparseAttentionMixer
from dew.nn.inputs import AttentionMetadata
from dew.nn.mixers import MixerContext, mixers
from tools.hf_reference import scatter_weights

BOUND = 1e-4
B, S, E, H = 2, 11, 32, 4
PADDED = 3
"""Leading slots of the second row the attention mask marks as padding."""

SETTINGS = dict(
    q_lora_rank=8, kv_lora_rank=8, qk_nope_head_dim=8, v_head_dim=8,
    index_n_heads=2, index_head_dim=8, index_topk=4, index_kpool=2,
    index_kpool_always_select_tail=True)


def reference_block(attention_bias: bool = False) -> Glm5NextTextAttention:
    # The module's SETTINGS, spelled out: the typed config takes no dict.
    config = Glm5NextTextConfig(
        vocab_size=64, hidden_size=E, intermediate_size=32, moe_intermediate_size=8, num_hidden_layers=1,
        num_attention_heads=H, num_key_value_heads=H, n_routed_experts=4, num_experts_per_tok=2,
        q_lora_rank=8, kv_lora_rank=8, qk_nope_head_dim=8, qk_rope_head_dim=0, v_head_dim=8,
        index_n_heads=2, index_head_dim=8, index_topk=4, index_kpool=2, index_kpool_always_select_tail=True,
        layer_types=["deepseek_sparse_attention"], indexer_types=["full"], mlp_layer_types=["dense"],
        rms_norm_eps=1e-5, attention_bias=attention_bias)
    config._attn_implementation = "eager"
    block = Glm5NextTextAttention(config, 0)
    scatter_weights(block)
    return block.eval()


def module(max_seq_len: int = 16, attention_bias: bool = False, **overrides) -> KPoolSparseAttention:
    return KPoolSparseAttention(
        emb_features=E, num_heads=H, max_seq_len=max_seq_len, norm_eps=1e-5, scale_after_cast=True,
        attention_bias=attention_bias, **{**SETTINGS, **overrides})


def dew_leaf(name: str) -> tuple[str, bool]:
    """A torch parameter name as the module's leaf path and whether the
    tensor transposes: `[out, in]` linear weights become kernels, norm
    weights scales, biases and the compression tables stay as they are."""
    parts = name.split('.')
    if parts[-1] == 'weight' and parts[-2] in ('q_a_layernorm', 'kv_a_layernorm', 'k_norm'):
        return '.'.join(parts[:-1] + ['scale']), False
    if parts[-1] == 'weight':
        return '.'.join(parts[:-1] + ['kernel']), True
    return name, False


def translated(block: Glm5NextTextAttention) -> dict:
    """The block's tensors as the module's parameter tree."""
    tree: dict = {}
    for name, tensor in block.state_dict().items():
        leaf, transpose = dew_leaf(name)
        array = tensor.detach().numpy()
        parts = leaf.split('.')
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = jnp.asarray(array.T if transpose else array)
    return {'params': tree}


def inputs(seed: int = 0):
    hidden = np.random.RandomState(seed).randn(B, S, E).astype(np.float32)
    valid = np.ones((B, S), bool)
    valid[1, :PADDED] = False
    return hidden, valid


def scaled(actual, wanted) -> float:
    return float(np.max(np.abs(np.asarray(actual, np.float64) - wanted)) / np.max(np.abs(wanted)))


def boundary_gap(block: Glm5NextTextAttention, hidden: torch.Tensor, valid: torch.Tensor) -> float:
    """The smallest margin, over the valid queries, between the last selected
    pool's reference score and the best rejected one (modeling_glm5_next.py:
    797-847 replayed); a zero margin means the selection is a tie-break."""
    indexer, q_a_proj, q_a_layernorm = block.indexer, block.q_a_proj, block.q_a_layernorm
    assert indexer is not None and q_a_proj is not None and q_a_layernorm is not None
    with torch.no_grad():
        q_resid = q_a_layernorm(q_a_proj(hidden))
        query = indexer.wq_b(q_resid).view(B, S, -1, indexer.head_dim)
        packed = torch.cat([indexer.k_norm(indexer.wk(hidden)),
                            F.linear(hidden, indexer.index_kpool_compress_gate),
                            valid.to(hidden.dtype)[..., None]], dim=-1)
        pool_keys, pool_indices, pool_valid = indexer.get_pooled_states(packed)
        scores = F.relu(torch.matmul(query.float(), pool_keys.transpose(-1, -2).float().unsqueeze(1))
                        * indexer.softmax_scale)
        weights = indexer.weights_proj(hidden).float() * indexer.n_heads ** -0.5
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)
        slots = torch.arange(S)
        visible = (slots[None, None, :] <= slots[None, :, None]) & valid[:, None, :]
        pool_end = pool_indices[..., -1].clamp(0, S - 1)
        candidates = visible.gather(-1, pool_end[:, None, :].expand(B, S, -1)) & pool_valid[:, None]
        ranked = index_scores.masked_fill(~candidates, float('-inf')).sort(-1, descending=True).values
        select_k = indexer.index_topk // indexer.index_kpool
        rejected = ranked[..., select_k]
        margin = torch.where(torch.isfinite(rejected), ranked[..., select_k - 1] - rejected,
                             torch.ones_like(rejected))
        return float(margin[valid].min())


def reference_output(block, hidden, valid):
    with torch.no_grad():
        return block(torch.from_numpy(hidden), torch.from_numpy(valid))[0].numpy()


def test_the_block_matches_the_reference_and_is_sparse():
    """Output parity on the valid rows, with the left-padded row pooling from
    its first real key; and the selection matters: the same weights under a
    budget that admits every pool (a dense causal softmax) land elsewhere."""
    block = reference_block()
    hidden, valid = inputs()
    assert boundary_gap(block, torch.from_numpy(hidden), torch.from_numpy(valid)) > 1e-3
    wanted = reference_output(block, hidden, valid)
    variables = translated(block)
    metadata = AttentionMetadata(valid=jnp.asarray(valid))

    ours = np.asarray(module().apply(variables, jnp.asarray(hidden), attention_metadata=metadata))
    assert scaled(ours[valid], wanted[valid]) < BOUND
    dense = np.asarray(module(index_topk=64).apply(variables, jnp.asarray(hidden), attention_metadata=metadata))
    assert scaled(dense[valid], wanted[valid]) > 1e-2


def test_the_block_matches_the_reference_with_biased_projections():
    """`attention_bias` puts a bias on q_a_proj, kv_a_proj_with_mqa and o_proj
    and nowhere else (modeling_glm5_next.py:1097-1127)."""
    block = reference_block(attention_bias=True)
    hidden, valid = inputs()
    assert boundary_gap(block, torch.from_numpy(hidden), torch.from_numpy(valid)) > 1e-3
    wanted = reference_output(block, hidden, valid)
    variables = translated(block)
    assert set(variables['params']['q_a_proj']) == {'kernel', 'bias'}
    assert set(variables['params']['q_b_proj']) == {'kernel'}

    ours = module(attention_bias=True).apply(
        variables, jnp.asarray(hidden), attention_metadata=AttentionMetadata(valid=jnp.asarray(valid)))
    assert scaled(np.asarray(ours)[valid], wanted[valid]) < BOUND


def test_the_gradients_match_torch_autograd():
    """One readout of the valid rows, differentiated into the input and every
    parameter: the attention's projections and norms carry the reference's
    gradients and the indexer's weights none, its selection being detached
    on both sides."""
    block = reference_block()
    hidden, valid = inputs()
    cotangent = np.random.RandomState(1).randn(B, S, E).astype(np.float32) * valid[..., None]
    x = torch.from_numpy(hidden).requires_grad_(True)
    block(x, torch.from_numpy(valid))[0].mul(torch.from_numpy(cotangent)).sum().backward()
    theirs = {name: None if p.grad is None else p.grad.numpy() for name, p in block.named_parameters()}
    assert x.grad is not None

    variables = translated(block)
    metadata = AttentionMetadata(valid=jnp.asarray(valid))

    def loss(x, variables):
        out = module().apply(variables, x, attention_metadata=metadata)
        assert isinstance(out, jax.Array)
        return jnp.sum(out * jnp.asarray(cotangent))

    grad_x, grads = jax.grad(loss, argnums=(0, 1))(jnp.asarray(hidden), variables)
    assert scaled(grad_x, x.grad.numpy()) < BOUND
    ours = {'.'.join(str(key.key) for key in path): leaf
            for path, leaf in jax.tree_util.tree_flatten_with_path(grads['params'])[0]}
    compared = 0
    for name, wanted in theirs.items():
        leaf, transpose = dew_leaf(name)
        if name.startswith('indexer.'):
            assert wanted is None, name
            assert float(jnp.max(jnp.abs(ours[leaf]))) == 0.0, name
            continue
        assert wanted is not None, name
        assert scaled(ours[leaf], wanted.T if transpose else wanted) < BOUND, name
        compared += 1
    assert compared == 7


def decode_steps(block_module, variables, hidden, valid, prefill: int):
    """Prefill `prefill` tokens, then one token per step, the sampler's protocol."""
    x = jnp.asarray(hidden)
    row_valid = None if valid is None else jnp.asarray(valid)

    def metadata(start, stop):
        return None if row_valid is None else AttentionMetadata(valid=row_valid[:, start:stop])

    _, allocated = block_module.apply(variables, x[:, :1], decode=True, mutable=["cache"])
    assert not bool(jnp.any(allocated["cache"]["cache_valid"]))
    out, state = block_module.apply({**variables, **allocated}, x[:, :prefill], decode=True,
                                    attention_metadata=metadata(0, prefill), mutable=["cache"])
    steps = [out]
    for position in range(prefill, S):
        out, state = block_module.apply({**variables, **state}, x[:, position:position + 1], decode=True,
                                        attention_metadata=metadata(position, position + 1), mutable=["cache"])
        steps.append(out)
    return np.asarray(jnp.concatenate(steps, axis=1))


def test_prefill_then_token_steps_reproduce_the_parallel_block():
    """Five tokens, then one at a time through the fixed-capacity cache: each
    step's pools are read off the cache slots, the tail is empty after an
    even count of keys and one token after an odd one, so it changes at
    every step; the steps equal the block over the whole sequence."""
    block = reference_block()
    hidden, _ = inputs()
    variables = translated(block)
    wanted = np.asarray(module().apply(variables, jnp.asarray(hidden)))
    assert scaled(wanted, reference_output(block, hidden, np.ones((B, S), bool))) < BOUND

    assert scaled(decode_steps(module(), variables, hidden, None, prefill=5), wanted) < BOUND


def test_a_padded_prefill_decodes_like_the_padded_parallel_block():
    """The cache holds the real tokens compactly where the parallel block
    pools from the row's first real key: the same pools, so the valid rows
    agree; the padded queries' steps write nothing."""
    block = reference_block()
    hidden, valid = inputs()
    variables = translated(block)
    wanted = reference_output(block, hidden, valid)

    ours = decode_steps(module(), variables, hidden, valid, prefill=5)
    assert scaled(ours[valid], wanted[valid]) < BOUND


def test_the_kind_builds_from_the_configs_fields_and_is_nope():
    record = {"kind": "kpool_sparse_attention", **SETTINGS}
    mixer = mixers.from_record(record)
    assert isinstance(mixer, KPoolSparseAttentionMixer)
    assert mixers["kpool_sparse_attention"] is KPoolSparseAttentionMixer
    assert (mixer.index_topk, mixer.index_kpool, mixer.q_lora_rank) == (4, 2, 8)

    ctx = MixerContext(emb_features=E, num_heads=H, num_kv_heads=H, head_dim=8, max_seq_len=16,
                       norm_eps=1e-5, scale_after_cast=True)
    built = mixer.build(ctx)(name='self_attn')
    assert isinstance(built, KPoolSparseAttention)
    assert (built.max_seq_len, built.norm_eps, built.scale_after_cast) == (16, 1e-5, True)

    with pytest.raises(ValueError, match="qk_rope_head_dim"):
        mixers.from_record({**record, "qk_rope_head_dim": 8})
    with pytest.raises(ValueError, match="causal"):
        mixer.build(MixerContext(emb_features=E, num_heads=H, num_kv_heads=H, head_dim=8,
                                 max_seq_len=16, causal=False))
    with pytest.raises(ValueError, match="kv_shared"):
        mixer.build(MixerContext(emb_features=E, num_heads=H, num_kv_heads=H, head_dim=8,
                                 max_seq_len=16, kv_shared=True))


def test_a_budget_the_pool_does_not_divide_is_refused():
    with pytest.raises(ValueError, match="divisible"):
        module(index_topk=5).init(jax.random.key(0), jnp.zeros((1, 4, E), jnp.float32))


def selection_step(block, variables, cache, x, valid, phase):
    return jax.jit(lambda state, values, active: block.apply(
        {**variables, "cache": state}, values, decode=True,
        attention_metadata=AttentionMetadata(valid=active),
        prediction_phase=phase, mutable=["cache"]))(cache, x, valid)


def test_prediction_selection_seed_miss_hit_and_fixed_tail():
    block = module()
    variables = translated(reference_block())
    hidden, valid = inputs()
    x, active = jnp.asarray(hidden), jnp.asarray(valid)
    _, allocated = block.apply(variables, x[:, :1], decode=True,
                               prediction_phase="extend", mutable=["cache"])
    np.testing.assert_array_equal(allocated["cache"]["selection_position"], [-1, -1])
    _, extended = selection_step(block, variables, allocated["cache"], x[:, :5], active[:, :5], "extend")
    seed = extended["cache"]
    np.testing.assert_array_equal(seed["selection_position"], [4, 1])
    assert 4 in np.asarray(seed["selection_indices"])[0]
    mixed = {**seed, "selection_position": seed["selection_position"].at[1].set(-1)}
    _, drafted = selection_step(block, variables, mixed, x[:, 5:6], active[:, 5:6], "draft")
    np.testing.assert_array_equal(drafted["cache"]["selection_indices"][0], seed["selection_indices"][0])
    np.testing.assert_array_equal(drafted["cache"]["selection_position"], [4, 2])
    assert 2 in np.asarray(drafted["cache"]["selection_indices"])[1]
    _, next_draft = selection_step(block, variables, drafted["cache"], x[:, 6:7], active[:, 6:7], "draft")
    np.testing.assert_array_equal(next_draft["cache"]["selection_indices"], drafted["cache"]["selection_indices"])
    assert 6 not in np.asarray(next_draft["cache"]["selection_indices"])[0]
    ordinary, cleared = selection_step(block, variables, next_draft["cache"], x[:, 7:8], active[:, 7:8], "ordinary")
    canonical = {name: value for name, value in next_draft["cache"].items()
                 if name not in ("selection_indices", "selection_position")}
    expected, _ = selection_step(block, variables, canonical, x[:, 7:8], active[:, 7:8], "ordinary")
    np.testing.assert_allclose(ordinary, expected, atol=BOUND, rtol=0)
    np.testing.assert_array_equal(cleared["cache"]["selection_position"], [-1, -1])


def test_prediction_selection_empty_seed_inactive_rows_and_reordering():
    from dew.nn.backbones.causal_transformer import gather_cache_rows

    block = module(index_kpool_always_select_tail=False)
    variables = translated(reference_block())
    hidden, _ = inputs()
    x = jnp.asarray(hidden)
    valid = jnp.asarray([[True], [False]])
    _, allocated = block.apply(variables, x[:, :1], decode=True,
                               prediction_phase="extend", mutable=["cache"])
    _, extended = selection_step(block, variables, allocated["cache"], x[:, :1], valid, "extend")
    np.testing.assert_array_equal(extended["cache"]["selection_position"], [0, -1])
    np.testing.assert_array_equal(extended["cache"]["selection_indices"], -jnp.ones((2, 4), jnp.int32))
    out, drafted = selection_step(block, variables, extended["cache"], x[:, 1:2], valid, "draft")
    np.testing.assert_array_equal(drafted["cache"]["selection_indices"][0], [-1, -1, -1, -1])
    for name, value in extended["cache"].items():
        np.testing.assert_array_equal(drafted["cache"][name][1], value[1])
    rows = jnp.asarray([1, 0, 0])
    reordered, _ = selection_step(block, variables, gather_cache_rows(extended["cache"], rows),
                                  x[rows, 1:2], valid[rows], "draft")
    np.testing.assert_allclose(reordered, out[rows], atol=BOUND, rtol=0)


def test_prediction_selection_rewind_recomputes_only_invalidated_row():
    block = module()
    variables = translated(reference_block())
    hidden, _ = inputs()
    x, valid = jnp.asarray(hidden), jnp.ones((2, 5), bool)
    _, allocated = block.apply(variables, x[:, :1], decode=True,
                               prediction_phase="extend", mutable=["cache"])
    _, extended = selection_step(block, variables, allocated["cache"], x[:, :5], valid, "extend")
    cache = {**extended["cache"], "cache_index": jnp.asarray([3, 5], jnp.int32)}
    _, drafted = selection_step(block, variables, cache, x[:, 5:6], valid[:, :1], "draft")
    np.testing.assert_array_equal(drafted["cache"]["selection_position"], [3, 4])
    np.testing.assert_array_equal(drafted["cache"]["selection_indices"][1], cache["selection_indices"][1])
    assert 4 not in np.asarray(drafted["cache"]["selection_indices"])[0]
    fresh = {**cache, "selection_position": jnp.full((2,), -1, jnp.int32)}
    _, recomputed = selection_step(block, variables, fresh, x[:, 5:6], valid[:, :1], "draft")
    np.testing.assert_array_equal(drafted["cache"]["selection_indices"][0],
                                  recomputed["cache"]["selection_indices"][0])
    inactive_rewound = {
        **drafted["cache"],
        "cache_index": drafted["cache"]["cache_index"].at[1].set(3),
        "cache_valid": drafted["cache"]["cache_valid"].at[1].set(jnp.arange(block.max_seq_len) < 3),
    }
    _, held = selection_step(block, variables, inactive_rewound, x[:, 6:7],
                              jnp.asarray([[True], [False]]), "draft")
    for name, value in inactive_rewound.items():
        np.testing.assert_array_equal(held["cache"][name][1], value[1])
    actual, reactivated = selection_step(block, variables, held["cache"], x[:, 7:8], valid[:, :1], "draft")
    np.testing.assert_array_equal(reactivated["cache"]["selection_position"], [3, 3])
    fresh = {**held["cache"], "selection_position": held["cache"]["selection_position"].at[1].set(-1)}
    expected, recomputed = selection_step(block, variables, fresh, x[:, 7:8], valid[:, :1], "draft")
    np.testing.assert_allclose(actual[1], expected[1], atol=BOUND, rtol=0)
    np.testing.assert_array_equal(reactivated["cache"]["selection_indices"][1],
                                  recomputed["cache"]["selection_indices"][1])


def test_the_pool_selection_breaks_a_tie_at_the_lower_token_index():
    """Pools that score equally go to the lower token indices, and a pool
    the query cannot see stays unselected however the tie falls: every pool
    scores exactly zero with the head weights zeroed, so the tie rule alone
    decides the twelve-token selection."""
    from dew.nn.dsa_kpool import KPoolIndexer

    indexer = KPoolIndexer(emb_features=E, n_heads=2, head_dim=4, top_k=6, kpool=3,
                           always_select_tail=False)
    total = 12
    x = jax.random.normal(jax.random.key(0), (1, total, E), jnp.float32)
    packed = jnp.concatenate(
        [jax.random.normal(jax.random.key(1), (1, total, 8), jnp.float32),
         jnp.ones((1, total, 1), jnp.float32)], axis=-1)
    visible = jnp.tril(jnp.ones((total, total), jnp.bool_))[None]
    params = dict(indexer.init(jax.random.key(2), x, x, packed, visible,
                               method="select_indices")["params"])
    params["weights_proj"] = {"kernel": jnp.zeros_like(params["weights_proj"]["kernel"])}
    tokens = indexer.apply({"params": params}, x, x, packed, visible, method="select_indices")

    np.testing.assert_array_equal(tokens[0, -1], [0, 1, 2, 3, 4, 5])
    # The query at index 4 sees one whole pool, so the second chosen pool is
    # no candidate and contributes empty slots instead of later tokens.
    np.testing.assert_array_equal(tokens[0, 4], [0, 1, 2, -1, -1, -1])
