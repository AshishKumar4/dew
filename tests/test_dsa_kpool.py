"""K-pool sparse attention against transformers 5.16.1's `Glm5NextTextAttention`.

The reference block runs live here on random weights (tools/hf_reference.py's
`scatter_weights`), a tiny NoPE config with pools of two keys and a budget of
two pools, over eleven tokens so the tail alternates between empty and one
token, one row of the batch left-padded through the attention mask. Dew's block,
its gradients and its decode are on par with the reference's fp32 from the
reference run in float64 over drawn inputs (check 1, tests/reference_error.py),
on the valid rows; a decode step that recomputes what a cached selection
holds is the same program on the same values, to the bit, and a different
program for the same result (a reordered batch, a cache that never held a
selection) on par by check 1.

The indexer's scores are relu'd sums, so two pools tie at exactly zero
often; the reference's `topk` and `jax.lax.top_k` break such a tie
differently, which is not a port error, and a selection decided within
rounding could pick other pools in float64. A draw (`drawn`) keeps only
inputs at which the reference's own scores decide every valid query's
selection by more than MARGIN (`boundary_gap`), so a failure here is a
difference in the math and not in a tie-break.
"""

import contextlib
import copy
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from flax import linen as nn
from reference_error import assert_as_exact, assert_moves, torch_with_gradients, widened, with_gradients
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextAttention

from dew.nn.dsa_kpool import KPoolSparseAttention, KPoolSparseAttentionMixer
from dew.nn.inputs import AttentionMetadata, MixerCall, coordinates
from dew.nn.mixers import MixerContext
from dew.registry import mixers
from tools.hf_reference import scatter_weights

B, S, E, H = 2, 11, 32, 4
PADDED = 3
"""Leading slots of the second row the attention mask marks as padding."""

SETTINGS = {
    "q_lora_rank": 8, "kv_lora_rank": 8, "qk_nope_head_dim": 8, "v_head_dim": 8,
    "index_n_heads": 2, "index_head_dim": 8, "index_topk": 4, "index_kpool": 2,
    "index_kpool_always_select_tail": True}


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
        emb_features=E, num_heads=H, max_seq_len=max_seq_len, qk_rope_head_dim=0, norm_eps=1e-5,
        scale_after_cast=True,
        attention_bias=attention_bias, **{**SETTINGS, **overrides})


def dew_leaf(name: str) -> tuple[str, bool]:
    """A torch parameter name as the module's leaf path and whether the
    tensor transposes: `[out, in]` linear weights become kernels, norm
    weights scales, biases and the compression tables stay as they are."""
    parts = name.split('.')
    if parts[-1] == 'weight' and parts[-2] in ('q_a_layernorm', 'kv_a_layernorm', 'k_norm'):
        return '.'.join([*parts[:-1], 'scale']), False
    if parts[-1] == 'weight':
        return '.'.join([*parts[:-1], 'kernel']), True
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


MARGIN = 1e-3
"""The least selection margin a draw admits (`boundary_gap`): four orders above the scores' fp32
rounding, so float64 and fp32 select the same pools."""

SEEDS = {'output': 8, 'biased': 8, 'dense': 8, 'gradients': 64, 'parallel': 8, 'decode': 8,
         'padded decode': 8, 'hole': 8, 'gated layer': 8, 'ordinary step': 32, 'reordered draft': 32}
"""Check 1's declared seeds for each site of the file, sized by `seeds_for` (on armada)."""


def drawn(block, padded: bool = True, cotangent: bool = False):
    """Check 1's draw at `block`'s weights: a seed's hidden states and validity (`inputs`, all valid
    unless `padded`), and with `cotangent` a readout of the valid rows drawn apart from them, from the
    first of the seed's candidates whose every valid query's selection the reference decides by more
    than MARGIN. The candidates are inputs' seeds past 0, which the file's fixed cases read."""
    def draw(seed):
        for attempt in range(64):
            index = 1 + 64 * seed + attempt
            hidden, valid = inputs(index)
            valid = valid if padded else np.ones_like(valid)
            if boundary_gap(block, torch.from_numpy(hidden), torch.from_numpy(valid)) > MARGIN:
                if not cotangent:
                    return hidden, valid
                readout = np.random.default_rng([0x5EAD, index]).standard_normal((B, S, E), np.float32)
                readout = readout * valid[..., None]
                return hidden, valid, readout
        raise AssertionError(f"no candidate of seed {seed} has its selection decided by {MARGIN}")
    return draw


def on_valid_rows(out, valid):
    """`out` on the valid rows and zero on the padded ones, which neither side defines."""
    return np.where(np.asarray(valid)[..., None], np.asarray(out), 0)


def call(valid):
    return MixerCall(AttentionMetadata(coordinates=coordinates(valid=valid)))


def dew_block(variables, **overrides):
    """Dew's block over `variables` as a function of a draw: its output on the valid rows."""
    run = jax.jit(lambda hidden, valid: module(**overrides).apply(variables, hidden, call(valid)))
    return lambda hidden, valid: {'output': on_valid_rows(run(hidden, valid), valid)}


def reference_block_output(block):
    """The reference block, fp32, as a function of a draw: its output on the valid rows."""
    return lambda hidden, valid: {'output': on_valid_rows(reference_output(block, hidden, valid), valid)}


def float64_block(block):
    """The reference block run in float64, every float32 pin widened (`float64`), as a function of a
    widened draw: the truth check 1 measures both sides from."""
    from tools.torch_precision import float64

    wide = copy.deepcopy(block).double()

    def run(hidden, valid):
        with float64(), torch.no_grad():
            out = wide(torch.from_numpy(np.asarray(hidden)), torch.from_numpy(np.asarray(valid)))[0]
        return {'output': on_valid_rows(out.numpy(), valid)}
    return run


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
    """Output on par on the valid rows (check 1), with the left-padded row
    pooling from its first real key; and the selection matters: the same
    weights under a budget that admits every pool (a dense causal softmax)
    move the output past rounding (`assert_moves`)."""
    block = reference_block()
    variables = translated(block)
    draw, truth = drawn(block), float64_block(block)
    assert_as_exact(dew_block(variables), reference_block_output(block), draw, 'k-pool block',
                    seeds=SEEDS['output'], truth_fn=truth)
    assert_moves(dew_block(variables), dew_block(variables, index_topk=64), draw, 'dense budget',
                 seeds=SEEDS['dense'], truth_fn=truth)


def test_the_block_matches_the_reference_with_biased_projections():
    """`attention_bias` puts a bias on q_a_proj, kv_a_proj_with_mqa and o_proj
    and nowhere else (modeling_glm5_next.py:1097-1127); the biased block's
    output is on par with the reference's (check 1)."""
    block = reference_block(attention_bias=True)
    variables = translated(block)
    assert set(variables['params']['q_a_proj']) == {'kernel', 'bias'}
    assert set(variables['params']['q_b_proj']) == {'kernel'}
    assert_as_exact(dew_block(variables, attention_bias=True), reference_block_output(block), drawn(block),
                    'biased k-pool block', seeds=SEEDS['biased'], truth_fn=float64_block(block))


def torch_gradients(block):
    """The gradients of a readout of `block`'s output into its input and every attention parameter, as
    a function of a draw, by name (the input's 'x'); in float64, every float32 pin widened, where the
    block is."""
    from tools.torch_precision import float64

    wide = next(block.parameters()).dtype == torch.float64
    named = {name: parameter.detach().numpy() for name, parameter in block.named_parameters()
             if not name.startswith('indexer.')}

    def forward(x, params, valid):
        with float64() if wide else contextlib.nullcontext():
            return torch.func.functional_call(block, params, (x, valid))[0]

    def run(hidden, valid, readout):
        _, grad_x, grads, _ = torch_with_gradients(forward)(np.asarray(hidden), named, np.asarray(valid),
                                                           np.asarray(readout))
        return {'x': grad_x, **{name: grads[name] for name in named}}
    return run


def dew_gradients(variables, names):
    """Dew's gradients of the same readout, under the reference's parameter `names`, each laid out as
    the reference's."""
    @jax.jit
    def pulled(x, variables, valid, readout):
        _, grad_x, grads = with_gradients(lambda x, variables: module().apply(variables, x, call(valid)))(
            x, variables, readout)
        return grad_x, grads

    def run(hidden, valid, readout):
        grad_x, grads = pulled(hidden, variables, valid, readout)
        ours = {'.'.join(str(key.key) for key in path): leaf
                for path, leaf in jax.tree_util.tree_flatten_with_path(grads['params'])[0]}
        found = {'x': np.asarray(grad_x)}
        for name in names:
            leaf, transpose = dew_leaf(name)
            found[name] = np.asarray(ours[leaf]).T if transpose else np.asarray(ours[leaf])
        return found
    return run


def test_the_gradients_match_torch_autograd():
    """One readout of the valid rows, differentiated into the input and every
    parameter: the attention's projections and norms carry gradients on par
    with the reference's (check 1) and the indexer's weights none, its
    selection being detached on both sides."""
    block = reference_block()
    variables = translated(block)
    draw = drawn(block, cotangent=True)
    names = [name for name, _ in block.named_parameters() if not name.startswith('indexer.')]
    assert len(names) == 7
    assert_as_exact(dew_gradients(variables, names), torch_gradients(block), draw, 'k-pool gradients',
                    seeds=SEEDS['gradients'], truth_fn=torch_gradients(copy.deepcopy(block).double()))

    hidden, valid, readout = draw(0)
    x = torch.from_numpy(hidden).requires_grad_(requires_grad=True)
    block(x, torch.from_numpy(valid))[0].mul(torch.from_numpy(readout)).sum().backward()
    indexer = [name for name, _ in block.named_parameters() if name.startswith('indexer.')]
    assert indexer and all(dict(block.named_parameters())[name].grad is None for name in indexer)
    for name, gradient in dew_gradients(variables, indexer)(hidden, valid, readout).items():
        if name != 'x':
            assert not np.any(gradient), name


def test_decode_cache_preserves_the_expanded_projection_storage_dtype():
    """A promoted KV projection stores its actual keys and values, even when the preceding
    latent norm computes in a narrower dtype."""
    with jax.enable_x64(new_val=True):
        hidden, _ = inputs()
        block = module()
        variables = block.init(jax.random.key(91), jnp.asarray(hidden))
        projection = variables["params"]["kv_b_proj"]
        projection["kernel"] = projection["kernel"].astype(jnp.float64)
        _, state = block.apply(variables, jnp.asarray(hidden[:, :1]), MixerCall(decode=True),
                               mutable=["cache"])
        assert state["cache"]["cached_key"].dtype == jnp.float64
        assert state["cache"]["cached_value"].dtype == jnp.float64


def decode_steps(block_module, variables, hidden, valid, prefill: int):
    """Prefill `prefill` tokens, then one token per step, the sampler's protocol."""
    x = jnp.asarray(hidden)
    row_valid = None if valid is None else jnp.asarray(valid)

    def metadata(start, stop):
        if row_valid is None:
            return None
        return AttentionMetadata(coordinates=coordinates(valid=row_valid[:, start:stop]))

    _, allocated = block_module.apply(variables, x[:, :1], MixerCall(decode=True), mutable=["cache"])
    assert not bool(jnp.any(allocated["cache"]["cache_index"]))
    out, state = block_module.apply({**variables, **allocated}, x[:, :prefill],
                                    MixerCall(metadata(0, prefill), decode=True), mutable=["cache"])
    steps = [out]
    for position in range(prefill, S):
        out, state = block_module.apply(
            {**variables, **state},
            x[:, position : position + 1], MixerCall(metadata(position, position + 1), decode=True),
            mutable=["cache"],
        )
        steps.append(out)
    return np.asarray(jnp.concatenate(steps, axis=1))


def stepped(variables, rows: bool = True):
    """Dew's prefill of five tokens and then one at a time (`decode_steps`) as a function of a draw:
    its output on the valid rows, the steps told the row's validity where `rows`."""
    return lambda hidden, valid: {'output': on_valid_rows(
        decode_steps(module(), variables, hidden, valid if rows else None, prefill=5), valid)}


def test_prefill_then_token_steps_reproduce_the_parallel_block():
    """Five tokens, then one at a time through the fixed-capacity cache: each
    step's pools are read off the cache slots, the tail is empty after an
    even count of keys and one token after an odd one, so it changes at
    every step; the steps are on par with the block over the whole sequence,
    and it with the reference (check 1)."""
    block = reference_block()
    variables = translated(block)
    draw, truth = drawn(block, padded=False), float64_block(block)
    assert_as_exact(dew_block(variables), reference_block_output(block), draw, 'unpadded k-pool block',
                    seeds=SEEDS['parallel'], truth_fn=truth)
    assert_as_exact(stepped(variables, rows=False), dew_block(variables), draw, 'k-pool decode',
                    seeds=SEEDS['decode'], truth_fn=truth)


def test_a_padded_prefill_decodes_like_the_padded_parallel_block():
    """The cache holds the real tokens compactly where the parallel block
    pools from the row's first real key: the same pools, so the valid rows
    are on par (check 1); the padded queries' steps write nothing."""
    block = reference_block()
    assert_as_exact(stepped(translated(block)), reference_block_output(block), drawn(block),
                    'padded k-pool decode', seeds=SEEDS['padded decode'], truth_fn=float64_block(block))


def test_a_row_with_a_hole_pools_as_its_cache_does():
    """A validity hole inside a row: the parallel block pools and tails the row's real keys in the
    order its cache compacts them (`valid_order`), which the reference does not, so its real queries
    are on par with the prefill-then-steps ones (check 1), as a left-padded row's are; the truth is
    Dew's block in float64."""
    variables = translated(reference_block())

    def draw(seed):
        hidden, valid = inputs(1 + seed)
        valid[0, 3] = False
        return hidden, valid

    def truth(hidden, valid):
        wide = module(dtype=jnp.float64).apply(widened(variables), hidden, call(valid))
        return {'output': on_valid_rows(wide, valid)}

    assert_as_exact(stepped(variables), dew_block(variables), draw, 'k-pool hole', seeds=SEEDS['hole'],
                    truth_fn=truth)


def test_the_kind_builds_from_the_configs_fields_and_is_nope():
    record = {"class": "kpool_sparse_attention", "fields": {**SETTINGS}}
    mixer = mixers.from_record(record)
    assert isinstance(mixer, KPoolSparseAttentionMixer)
    assert mixers["kpool_sparse_attention"] is KPoolSparseAttentionMixer
    assert (mixer.index_topk, mixer.index_kpool, mixer.q_lora_rank) == (4, 2, 8)

    ctx = MixerContext(emb_features=E, num_heads=H, num_kv_heads=H, head_dim=8, max_seq_len=16,
                       norm_eps=1e-5, scale_after_cast=True)
    built = mixer.build(ctx)(name='self_attn')
    assert isinstance(built, KPoolSparseAttention)
    assert (built.max_seq_len, built.norm_eps, built.scale_after_cast) == (16, 1e-5, True)

    with pytest.raises(ValueError):
        mixers.from_record({**record, "qk_rope_head_dim": 8})
    with pytest.raises(ValueError):
        mixer.build(MixerContext(emb_features=E, num_heads=H, num_kv_heads=H, head_dim=8,
                                 max_seq_len=16, causal=False))
    with pytest.raises(ValueError):
        mixer.build(MixerContext(emb_features=E, num_heads=H, num_kv_heads=H, head_dim=8,
                                 max_seq_len=16, kv_shared=True))


def test_a_gated_layer_gates_its_heads_context_as_latent_attention_does():
    """K-pool attention takes the latent attention's output gate, a layer's
    `output_gate` as its kind builds it: the gated layer's output is on par
    (check 1) with o_proj over the ungated heads' context times
    sigmoid(g_proj(x)), each projection a plain Dense of the layer's kernels,
    its truth that composition in float64. The heads' context is the ungated
    layer's output through an identity o_proj."""
    gated, plain = (KPoolSparseAttentionMixer(**SETTINGS).build(MixerContext(
        emb_features=E, num_heads=H, num_kv_heads=H, head_dim=8, max_seq_len=16, norm_eps=1e-5,
        scale_after_cast=True, output_gate=gate))(name="self_attn") for gate in (True, False))
    params = gated.init(jax.random.key(0), inputs()[0])["params"]
    width = H * SETTINGS["v_head_dim"]
    assert width == E, "an identity o_proj reads the heads' context back"

    def dense(kernel, x):
        return nn.Dense(kernel.shape[1], use_bias=False).apply({"params": {"kernel": kernel}}, x)

    def composed(params, hidden):
        ungated = {name: leaf for name, leaf in params.items() if name != "g_proj"}
        context = plain.apply({"params": {**ungated, "o_proj": {"kernel": jnp.eye(width)}}}, hidden)
        gate = jax.nn.sigmoid(dense(params["g_proj"]["kernel"], hidden))
        return dense(params["o_proj"]["kernel"], context * gate)

    assert_as_exact(jax.jit(partial(gated.apply, {"params": params})), jax.jit(partial(composed, params)),
                    lambda seed: inputs(1 + seed)[:1], 'gated layer', seeds=SEEDS['gated layer'],
                    truth_fn=partial(composed, widened(params)))


def test_a_budget_the_pool_does_not_divide_is_refused():
    with pytest.raises(ValueError):
        module(index_topk=5).init(jax.random.key(0), jnp.zeros((1, 4, E), jnp.float32))


def selection_step(block, variables, cache, x, valid, phase):
    return jax.jit(lambda state, values, active: block.apply(
        {**variables, "cache": state}, values,
        MixerCall(AttentionMetadata(coordinates=coordinates(valid=active)), decode=True),
        prediction_phase=phase, mutable=["cache"]))(cache, x, valid)


def drafts(block, variables, hidden, valid):
    """The caches of a prediction chain on `hidden` and `valid`: allocated, extended over five tokens
    (the seed), drafted at the sixth with the second row's seed cleared, and drafted again at the
    seventh."""
    x, active = jnp.asarray(hidden), jnp.asarray(valid)
    _, allocated = block.apply(variables, x[:, :1], MixerCall(decode=True),
                               prediction_phase="extend", mutable=["cache"])
    _, extended = selection_step(block, variables, allocated["cache"], x[:, :5], active[:, :5], "extend")
    seed = extended["cache"]
    mixed = {**seed, "selection_position": seed["selection_position"].at[1].set(-1)}
    _, drafted = selection_step(block, variables, mixed, x[:, 5:6], active[:, 5:6], "draft")
    _, next_draft = selection_step(block, variables, drafted["cache"], x[:, 6:7], active[:, 6:7], "draft")
    return allocated["cache"], seed, drafted["cache"], next_draft["cache"]


def test_prediction_selection_seed_miss_hit_and_fixed_tail():
    """The seed's selection holds through its drafts and the fixed tail; an ordinary step clears it and
    is on par (check 1) with the step on a cache that never held one, a program of its own; the truth
    is that step in float64."""
    block = module()
    variables = translated(reference_block())
    chain = jax.jit(partial(drafts, block, variables))
    hidden, valid = inputs()
    allocated, seed, drafted, next_draft = chain(hidden, valid)
    np.testing.assert_array_equal(allocated["selection_position"], [-1, -1])
    np.testing.assert_array_equal(seed["selection_position"], [4, 1])
    assert 4 in np.asarray(seed["selection_indices"])[0]
    np.testing.assert_array_equal(drafted["selection_indices"][0], seed["selection_indices"][0])
    np.testing.assert_array_equal(drafted["selection_position"], [4, 2])
    assert 2 in np.asarray(drafted["selection_indices"])[1]
    np.testing.assert_array_equal(next_draft["selection_indices"], drafted["selection_indices"])
    assert 6 not in np.asarray(next_draft["selection_indices"])[0]
    _, cleared = selection_step(block, variables, next_draft, hidden[:, 7:8], valid[:, 7:8], "ordinary")
    np.testing.assert_array_equal(cleared["cache"]["selection_position"], [-1, -1])

    def ordinary(dtype, *, canonical):
        kpool = module(dtype=dtype)
        weights = variables if dtype == jnp.float32 else widened(variables)

        def run(cache, x, active):
            if canonical:
                cache = {name: value for name, value in cache.items()
                         if name not in ("selection_indices", "selection_position")}
            return selection_step(kpool, weights, cache, jnp.asarray(x, dtype), active, "ordinary")[0]

        return jax.jit(run)

    def draw(seed):
        hidden, valid = inputs(1 + seed)
        return chain(hidden, valid)[-1], hidden[:, 7:8], valid[:, 7:8]

    assert_as_exact(ordinary(jnp.float32, canonical=False), ordinary(jnp.float32, canonical=True), draw,
                    'ordinary step', seeds=SEEDS['ordinary step'],
                    truth_fn=ordinary(jnp.float64, canonical=True))


def test_prediction_selection_empty_seed_inactive_rows_and_reordering():
    from dew.nn.kv_cache import gather_cache_rows

    block = module(index_kpool_always_select_tail=False)
    variables = translated(reference_block())
    hidden, _ = inputs()
    x = jnp.asarray(hidden)
    valid = jnp.asarray([[True], [False]])
    _, allocated = block.apply(variables, x[:, :1], MixerCall(decode=True),
                               prediction_phase="extend", mutable=["cache"])
    _, extended = selection_step(block, variables, allocated["cache"], x[:, :1], valid, "extend")
    np.testing.assert_array_equal(extended["cache"]["selection_position"], [0, -1])
    np.testing.assert_array_equal(extended["cache"]["selection_indices"], -jnp.ones((2, 4), jnp.int32))
    _, drafted = selection_step(block, variables, extended["cache"], x[:, 1:2], valid, "draft")
    np.testing.assert_array_equal(drafted["cache"]["selection_indices"][0], [-1, -1, -1, -1])
    for name, value in extended["cache"].items():
        np.testing.assert_array_equal(drafted["cache"][name][1], value[1])

    # Drafting the cache's rows gathered (one row twice) is on par with gathering the batch's draft
    # (check 1), the inactive row defined by neither; the truth is the latter in float64.
    rows = jnp.asarray([1, 0, 0])

    def drafting(dtype, *, gathered):
        kpool = module(index_kpool_always_select_tail=False, dtype=dtype)
        weights = variables if dtype == jnp.float32 else widened(variables)

        def run(hidden):
            x = jnp.asarray(hidden, dtype)
            _, allocated = kpool.apply(weights, x[:, :1], MixerCall(decode=True), prediction_phase="extend",
                                      mutable=["cache"])
            cache = selection_step(kpool, weights, allocated["cache"], x[:, :1], valid, "extend")[1]["cache"]
            if gathered:
                cache, x, active = gather_cache_rows(cache, rows), x[rows], valid[rows]
                out, _ = selection_step(kpool, weights, cache, x[:, 1:2], active, "draft")
            else:
                out = selection_step(kpool, weights, cache, x[:, 1:2], valid, "draft")[0][rows]
            return jnp.where(valid[rows][..., None], out, 0)

        return jax.jit(run)

    assert_as_exact(drafting(jnp.float32, gathered=True), drafting(jnp.float32, gathered=False),
                    lambda seed: inputs(1 + seed)[:1], 'reordered draft', seeds=SEEDS['reordered draft'],
                    truth_fn=drafting(jnp.float64, gathered=False))


def test_prediction_selection_rewind_recomputes_only_invalidated_row():
    block = module()
    variables = translated(reference_block())
    hidden, _ = inputs()
    x, valid = jnp.asarray(hidden), jnp.ones((2, 5), bool)
    _, allocated = block.apply(variables, x[:, :1], MixerCall(decode=True),
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
    }
    _, held = selection_step(block, variables, inactive_rewound, x[:, 6:7],
                              jnp.asarray([[True], [False]]), "draft")
    for name, value in inactive_rewound.items():
        np.testing.assert_array_equal(held["cache"][name][1], value[1])
    actual, reactivated = selection_step(block, variables, held["cache"], x[:, 7:8], valid[:, :1], "draft")
    np.testing.assert_array_equal(reactivated["cache"]["selection_position"], [3, 3])
    fresh = {**held["cache"], "selection_position": held["cache"]["selection_position"].at[1].set(-1)}
    expected, recomputed = selection_step(block, variables, fresh, x[:, 7:8], valid[:, :1], "draft")
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(reactivated["cache"]["selection_indices"][1],
                                  recomputed["cache"]["selection_indices"][1])
