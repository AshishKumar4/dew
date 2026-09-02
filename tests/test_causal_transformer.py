"""The causal transformer decoder: causality, the KV cache, and the kernel paths.

Two properties carry the whole language model: a position never sees the
future, and decoding one token at a time against the KV cache gives the same
logits as one forward pass over the finished sequence. Everything else here
guards the config surface the HF decoders need (grouped-query heads, sliding
layers, the Gemma flags) and the param tree the interop map renames.
"""

import jax
import jax.numpy as jnp
import pytest

from dew.nn.attention import NormalAttention, scaled_dot_product_attention
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.registry import MODEL_REGISTRY, apply_precision_policy, build_model

VOCAB = 37
SEQ = 12


def tiny(**overrides):
    config = dict(vocab_size=VOCAB, emb_features=32, num_layers=2, num_heads=4,
                  mlp_ratio=2, max_seq_len=16)
    return CausalTransformer(**{**config, **overrides})


def tokens(rng, batch=2, length=SEQ):
    return jax.random.randint(rng, (batch, length), 0, VOCAB)


def decode_logits(model, params, prompt, rest):
    """Prefill `prompt`, then feed `rest` one token at a time: [B, 1 + len(rest), V]."""
    cache = model.apply(params, prompt.shape[0], method=CausalTransformer.init_cache,
                        mutable=['cache'])[1]['cache']
    logits, mutated = model.apply({**params, 'cache': cache}, prompt,
                                  decode=True, mutable=['cache'])
    steps = [logits[:, -1]]
    cache = mutated['cache']
    for position in range(rest.shape[1]):
        logits, mutated = model.apply({**params, 'cache': cache},
                                      rest[:, position:position + 1],
                                      decode=True, mutable=['cache'])
        cache = mutated['cache']
        steps.append(logits[:, -1])
    return jnp.stack(steps, axis=1)


def test_forward_returns_fp32_logits_per_token(rng):
    model = tiny()
    ids = tokens(rng)
    params = model.init(rng, ids)
    logits = model.apply(params, ids)
    assert logits.shape == (ids.shape[0], SEQ, VOCAB)
    assert logits.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(logits))


def test_bf16_compute_keeps_params_and_logits_fp32(rng):
    """bf16 is a compute dtype: the params and the head the loss reads stay fp32."""
    model = tiny(dtype=jnp.bfloat16)
    ids = tokens(rng)
    params = model.init(rng, ids)
    demoted = {jax.tree_util.keystr(path): str(leaf.dtype)
               for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]
               if leaf.dtype != jnp.float32}
    assert not demoted
    assert model.apply(params, ids).dtype == jnp.float32


def test_logits_ignore_every_later_token(rng):
    """Causality, stated as the property a trainer depends on: rewriting the
    tail of a sequence cannot move the logits before it."""
    model = tiny()
    ids = tokens(rng)
    params = model.init(rng, ids)
    baseline = model.apply(params, ids)

    cut = 5
    rewritten = ids.at[:, cut + 1:].set((ids[:, cut + 1:] + 7) % VOCAB)
    changed = model.apply(params, rewritten)
    assert jnp.array_equal(baseline[:, :cut + 1], changed[:, :cut + 1])
    # and the tail did move, so the test is not passing on a dead model
    assert not jnp.allclose(baseline[:, cut + 1:], changed[:, cut + 1:])


@pytest.mark.parametrize("attention_impl", [None, 'xla'])
def test_decode_cache_matches_the_full_sequence(rng, attention_impl):
    """The prefill plus single-token steps must reproduce the full-sequence
    logits position by position, on the reference kernel and on the fused one
    (where the cache mask travels as a mask argument, not a causal flag)."""
    model = tiny(attention_impl=attention_impl)
    ids = tokens(rng)
    params = model.init(rng, ids)
    full = model.apply(params, ids)

    prompt, rest = ids[:, :4], ids[:, 4:]
    incremental = decode_logits(model, params, prompt, rest)
    assert incremental.shape == (ids.shape[0], SEQ - 3, VOCAB)
    assert jnp.allclose(full[:, 3:], incremental, atol=1e-4)


def test_reference_and_xla_kernels_agree_when_causal(rng):
    model = tiny()
    ids = tokens(rng)
    params = model.init(rng, ids)
    reference = model.apply(params, ids)
    fused = model.clone(attention_impl='xla').apply(params, ids)
    assert jnp.allclose(reference, fused, atol=1e-4)


def test_sliding_window_agrees_across_kernels(rng):
    """The banded mask on the reference path and jax's local_window_size have to
    mean the same window, or a run changes behaviour when it changes machine."""
    query, key, value = (jax.random.normal(k, (2, 8, 4, 16))
                         for k in jax.random.split(rng, 3))
    reference = scaled_dot_product_attention(query, key, value, causal=True,
                                             sliding_window=3)
    fused = scaled_dot_product_attention(query, key, value, causal=True,
                                         sliding_window=3, implementation='xla')
    assert jnp.allclose(reference, fused, atol=1e-5)


def test_registry_builds_the_backbone_and_takes_the_precision_policy():
    assert MODEL_REGISTRY['causal_transformer'] is CausalTransformer
    config = apply_precision_policy(
        'causal_transformer', {'vocab_size': VOCAB, 'emb_features': 32,
                               'num_layers': 2, 'num_heads': 4, 'max_seq_len': 16},
        dtype='bfloat16', attention_impl='reference')
    model = build_model('causal_transformer', config)
    assert isinstance(model, CausalTransformer)
    assert model.dtype is jnp.bfloat16
    assert model.attention_impl is None

    ids = jnp.zeros((1, 4), jnp.int32)
    params = model.init(jax.random.PRNGKey(0), ids)
    assert model.apply(params, ids).dtype == jnp.float32


def test_param_tree_mirrors_the_hf_decoder_layout(rng):
    """The interop map has to be a rename, so the tree is a fixed contract."""
    model = tiny()
    params = model.init(rng, tokens(rng))['params']
    paths = sorted('.'.join(str(entry.key) for entry in path)
                   for path, _ in jax.tree_util.tree_flatten_with_path(params)[0])
    layer = ['input_layernorm.scale',
             'mlp.down_proj.kernel', 'mlp.gate_proj.kernel', 'mlp.up_proj.kernel',
             'post_attention_layernorm.scale',
             'self_attn.k_norm.scale', 'self_attn.k_proj.kernel',
             'self_attn.o_proj.kernel', 'self_attn.q_norm.scale',
             'self_attn.q_proj.kernel', 'self_attn.v_proj.kernel']
    assert paths == sorted(
        ['embed_tokens.embedding', 'norm.scale']
        + [f'layers_{index}.{leaf}' for index in (0, 1) for leaf in layer])
    # tie_embeddings=True is the reason there is no lm_head to rename
    assert 'lm_head' not in params


def test_untied_head_adds_lm_head_and_nothing_else(rng):
    model = tiny(tie_embeddings=False)
    params = model.init(rng, tokens(rng))['params']
    assert set(params) == {'embed_tokens', 'layers_0', 'layers_1', 'norm', 'lm_head'}
    assert params['lm_head']['kernel'].shape == (32, VOCAB)


def test_attention_bias_adds_the_qkvo_biases(rng):
    model = tiny(attention_bias=True)
    attention = model.init(rng, tokens(rng))['params']['layers_0']['self_attn']
    assert all('bias' in attention[proj]
               for proj in ('q_proj', 'k_proj', 'v_proj', 'o_proj'))
    assert 'bias' not in model.init(
        rng, tokens(rng))['params']['layers_0']['mlp']['gate_proj']


def test_grouped_query_heads_match_repeated_kv_projections(rng):
    """Grouped heads must read the kv head the fused kernels read: a GQA model
    equals an all-heads model whose kv kernels are the grouped ones repeated."""
    grouped = tiny(num_kv_heads=2)
    ids = tokens(rng)
    params = grouped.init(rng, ids)

    def widen(kernel):
        """One kv kernel per grouped head -> one per query head."""
        features, head_dim = kernel.shape[0], grouped.features_per_head
        return jnp.repeat(kernel.reshape(features, -1, head_dim),
                          grouped.num_heads // grouped.kv_heads,
                          axis=1).reshape(features, -1)

    expanded = {'params': {
        name: child if not name.startswith('layers_') else {
            **child,
            'self_attn': {
                **child['self_attn'],
                'k_proj': {'kernel': widen(child['self_attn']['k_proj']['kernel'])},
                'v_proj': {'kernel': widen(child['self_attn']['v_proj']['kernel'])}}}
        for name, child in params['params'].items()}}

    plain = tiny()
    assert jnp.allclose(grouped.apply(params, ids), plain.apply(expanded, ids), atol=1e-5)


def test_sliding_attention_forgets_past_the_window(rng):
    """Two layers of a window of 3 see 5 tokens back, and nothing before that."""
    model = tiny(layer_types=('sliding_attention',) * 2, sliding_window=3)
    ids = tokens(rng)
    params = model.init(rng, ids)
    baseline = model.apply(params, ids)

    flipped = 3
    changed = model.apply(params, ids.at[:, flipped].set((ids[:, flipped] + 5) % VOCAB))
    moved = jnp.abs(baseline - changed).max(axis=(0, 2)) > 1e-5
    assert [int(index) for index in jnp.where(moved)[0]] == list(
        range(flipped, flipped + 5))


def test_sliding_attention_decode_matches_the_full_sequence(rng):
    model = tiny(layer_types=('full_attention', 'sliding_attention'), sliding_window=4)
    ids = tokens(rng)
    params = model.init(rng, ids)
    full = model.apply(params, ids)
    incremental = decode_logits(model, params, ids[:, :6], ids[:, 6:])
    assert jnp.allclose(full[:, 5:], incremental, atol=1e-4)


def test_gemma_flags_scale_the_embeddings_and_cap_the_logits(rng):
    """embedding_scale, the (1 + w) norms, geglu and the tanh softcap are the
    Gemma switches; with the cap on, no logit can leave (-cap, cap)."""
    cap = 5.0
    model = tiny(embedding_scale=True, scale_offset=True, mlp='geglu',
                 final_logit_softcap=cap, num_kv_heads=2, head_dim=16,
                 rope_theta=1e6, rope_local_theta=1e4)
    ids = tokens(rng)
    params = model.init(rng, ids)
    logits = model.apply(params, ids)
    assert jnp.all(jnp.abs(logits) < cap)
    # zero-initialised (1 + w) scales are the identity, so nothing is dead
    assert jnp.all(params['params']['norm']['scale'] == 0.0)
    assert jnp.all(jnp.isfinite(logits))


def test_gemma_zero_qk_norm_weights_are_identity(rng):
    ids = tokens(rng)
    qwen = tiny(num_layers=1, scale_offset=False)
    gemma = tiny(num_layers=1, scale_offset=True)
    qwen_params = qwen.init(rng, ids)
    gemma_params = gemma.init(rng, ids)

    for name in ("q_norm", "k_norm"):
        qwen_scale = qwen_params["params"]["layers_0"]["self_attn"][name]["scale"]
        gemma_scale = gemma_params["params"]["layers_0"]["self_attn"][name]["scale"]
        assert jnp.all(qwen_scale == 1.0)
        gemma_params["params"]["layers_0"]["self_attn"][name]["scale"] = (
            jnp.zeros_like(gemma_scale))

    assert jnp.allclose(gemma.apply(gemma_params, ids), qwen.apply(qwen_params, ids),
                        atol=1e-6)


def test_local_rope_only_moves_the_sliding_layers(rng):
    """rope_local_theta is Gemma3's second rope base: it must reach the sliding
    layers and leave the full-attention ones alone."""
    ids = tokens(rng)
    shared = dict(layer_types=('full_attention', 'sliding_attention'),
                  sliding_window=4)
    model = tiny(**shared)
    params = model.init(rng, ids)
    same_theta = tiny(**shared, rope_local_theta=10000.0)
    assert jnp.allclose(model.apply(params, ids), same_theta.apply(params, ids))
    other_theta = tiny(**shared, rope_local_theta=1e6)
    assert not jnp.allclose(model.apply(params, ids), other_theta.apply(params, ids))


def param_paths(params):
    return {'.'.join(str(entry.key) for entry in path)
            for path, _ in jax.tree_util.tree_flatten_with_path(params)[0]}


def test_sandwich_norms_add_exactly_the_two_output_norms(rng):
    """Gemma's second pair of norms is additive: the pre-norms keep their names
    and their roles, so a checkpoint without them loads into the same tree."""
    ids = tokens(rng)
    plain = tiny().init(rng, ids)['params']
    sandwiched = tiny(sandwich_norms=True).init(rng, ids)['params']

    assert param_paths(sandwiched) - param_paths(plain) == {
        f'layers_{index}.{norm}.scale' for index in (0, 1)
        for norm in ('attention_output_norm', 'mlp_output_norm')}
    assert not param_paths(plain) - param_paths(sandwiched)
    assert sandwiched['layers_0']['attention_output_norm']['scale'].shape == (32,)


def test_sandwich_norms_normalize_what_the_residual_adds(rng):
    """The two norms sit on the sublayer outputs, which makes each residual
    contribution scale-free: a ten times larger o_proj and down_proj leave the
    logits where they were, and without the norms they move them."""
    ids = tokens(rng)
    model = tiny(sandwich_norms=True)
    params = model.init(rng, ids)

    amplified = ('o_proj', 'down_proj')
    louder = jax.tree_util.tree_map_with_path(
        lambda path, leaf: leaf * 10.0 if path[-2].key in amplified else leaf, params)

    def gap(model):
        return float(jnp.max(jnp.abs(model.apply(params, ids) - model.apply(louder, ids))))

    # exact in real arithmetic, fp32 rounding through the norm is the residue
    assert gap(model) < 1e-3
    assert gap(tiny()) > 0.1


def test_attention_scale_defaults_to_the_head_dim_scale(rng):
    """None is 1/sqrt(head_dim), the scale every kernel applies itself: asking
    for that number explicitly must not move a bit, and Gemma's
    query_pre_attn_scalar must move the logits."""
    ids = tokens(rng)
    model = tiny(head_dim=16)
    params = model.init(rng, ids)

    explicit = tiny(head_dim=16, attention_scale=16 ** -0.5)
    assert jnp.array_equal(model.apply(params, ids), explicit.apply(params, ids))

    # query_pre_attn_scalar 16 on head_dim 16 heads, as Gemma3 sets it
    gemma = tiny(head_dim=16, attention_scale=16 ** -0.5 * 2)
    assert not jnp.allclose(model.apply(params, ids), gemma.apply(params, ids))


def test_dropout_trains_with_an_rng_and_is_off_by_default(rng):
    model = tiny(dropout_rate=0.5)
    ids = tokens(rng)
    params = model.init(rng, ids)
    quiet = model.apply(params, ids)
    assert jnp.array_equal(quiet, model.apply(params, ids))
    noisy = model.apply(params, ids, train=True, rngs={'dropout': jax.random.PRNGKey(1)})
    assert not jnp.allclose(quiet, noisy)


def test_normal_attention_param_tree_survives_causal_and_decode(rng):
    """The diffusion attention gains the flags without gaining parameters, so a
    checkpoint moves between a bidirectional trainer and a decoding sampler."""
    x = jax.random.normal(rng, (2, 6, 16))
    plain = NormalAttention(query_dim=16, heads=2, dim_head=8)
    causal = NormalAttention(query_dim=16, heads=2, dim_head=8, causal=True, max_seq_len=8)
    shapes = jax.tree_util.tree_map(jnp.shape, plain.init(rng, x)['params'])
    assert jax.tree_util.tree_map(jnp.shape, causal.init(rng, x)['params']) == shapes

    decoding = causal.init(rng, x[:, :1], decode=True)
    assert jax.tree_util.tree_map(jnp.shape, decoding['params']) == shapes
    assert set(decoding['cache']) == {'cached_key', 'cached_value', 'cache_index'}
    assert decoding['cache']['cached_key'].shape == (2, 8, 2, 8)


def test_normal_attention_decode_matches_a_causal_forward(rng):
    attention = NormalAttention(query_dim=16, heads=2, dim_head=8, causal=True,
                                max_seq_len=8, use_bias=False)
    x = jax.random.normal(rng, (2, 6, 16))
    params = {'params': attention.init(rng, x)['params']}
    full = attention.apply(params, x)

    cache = attention.apply(params, x[:, :1], decode=True, mutable=['cache'])[1]['cache']
    out, mutated = attention.apply({**params, 'cache': cache}, x[:, :3],
                                   decode=True, mutable=['cache'])
    assert jnp.allclose(out, full[:, :3], atol=1e-5)
    cache = mutated['cache']
    for position in range(3, 6):
        out, mutated = attention.apply({**params, 'cache': cache},
                                       x[:, position:position + 1],
                                       decode=True, mutable=['cache'])
        cache = mutated['cache']
        assert jnp.allclose(out[:, 0], full[:, position], atol=1e-5)


def test_decoding_without_a_cache_length_is_refused(rng):
    attention = NormalAttention(query_dim=16, heads=2, dim_head=8, causal=True)
    with pytest.raises(ValueError, match="max_seq_len"):
        attention.init(rng, jax.random.normal(rng, (2, 4, 16)), decode=True)


def test_a_prompt_longer_than_the_cache_is_refused(rng):
    model = tiny(max_seq_len=8)
    ids = tokens(rng, length=12)
    params = model.init(rng, ids)
    cache = model.apply(params, 2, method=CausalTransformer.init_cache,
                        mutable=['cache'])[1]['cache']
    with pytest.raises(ValueError, match="do not fit"):
        model.apply({**params, 'cache': cache}, ids, decode=True, mutable=['cache'])


@pytest.mark.parametrize("config, message", [
    ({'head_dim': 7}, "even"),
    ({'num_kv_heads': 3}, "multiple"),
    ({'layer_types': ('full_attention',)}, "entries"),
    ({'layer_types': ('full_attention', 'linear_attention')}, "unknown layer types"),
    ({'layer_types': ('sliding_attention',) * 2}, "sliding_window"),
    ({'mlp': 'relu'}, "swiglu"),
])
def test_rejected_configs(rng, config, message):
    with pytest.raises(ValueError, match=message):
        tiny(**config).init(rng, tokens(rng))
