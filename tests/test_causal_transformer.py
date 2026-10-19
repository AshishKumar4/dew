"""The causal transformer decoder: causality, the KV cache, and the kernel paths.

Two properties carry the whole language model: a position never sees the
future, and decoding one token at a time against the KV cache gives the same
logits as one forward pass over the finished sequence. Everything else here
guards the config surface the HF decoders need (grouped-query heads, sliding
layers, the Gemma flags) and the param tree the interop map renames.
"""

import functools
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from model_support import TINY_DECODER
from reference_error import assert_as_exact_as_the_reference, widened

from dew.nn.attention import scaled_dot_product_attention
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.hyper_connections import HyperConnections
from dew.nn.mixers import AttentionMixer

VOCAB = 37
SEQ = 12


def tiny(**overrides):
    return CausalTransformer(**{"vocab_size": VOCAB, **TINY_DECODER, **overrides})


def tokens(rng, batch=2, length=SEQ):
    return jax.random.randint(rng, (batch, length), 0, VOCAB)


@pytest.mark.parametrize("fields,causal,message", [
    ({"head_dim": 3}, True, "head dim"),
    ({"window": 0}, True, "window"),
    ({"chunk": 0}, True, "chunk"),
    ({"window": 2, "chunk": 2}, True, "one or the other"),
    ({"chunk": 2}, False, "not causal"),
    ({"window": 2, "bidirectional_window": True}, True, "not causal"),
    ({"bidirectional_window": True}, False, "needs a window"),
    ({"num_kv_heads": 3}, True, "multiple"),
    ({"num_kv_heads": 0}, True, "multiple"),
])
def test_layer_kinds_refuse_invalid_rotary_mask_and_grouped_head_geometry(fields, causal, message):
    with pytest.raises(ValueError):
        model = tiny(num_layers=1, layer_types=("special",), kinds={"special": fields}, causal=causal)
        model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
@pytest.mark.parametrize("field", ["embedding_dropout_rate", "attention_dropout_rate", "dropout_rate"])
def test_decoder_dropout_is_training_only(rng, field):
    base = tiny(dropout_rate=0, attention_impl="reference")
    ids = tokens(rng, length=4)
    params = base.init(rng, ids)
    active = tiny(attention_impl="reference", **{"dropout_rate": 0, field: .5})
    first = active.apply(params, ids, train=True, rngs={"dropout": jax.random.key(10)})
    second = active.apply(params, ids, train=True, rngs={"dropout": jax.random.key(11)})
    assert not np.array_equal(first, second)
    expected = np.asarray(base.apply(params, ids, train=False))
    for key in (10, 11):
        evaluated = np.asarray(active.apply(params, ids, train=False, rngs={"dropout": jax.random.key(key)}))
        assert evaluated.tobytes() == expected.tobytes()
    np.testing.assert_array_equal(active.apply(params, ids), expected)

    if field == "embedding_dropout_rate":
        class Reference(nn.Module):
            @nn.compact
            def __call__(self, x, train):
                return nn.Dropout(.5, name="embedding_dropout")(x, deterministic=not train)

        embeddings = active.apply(params, ids, method=CausalTransformer.token_embeddings)
        reference = Reference().apply({}, embeddings, train=True, rngs={"dropout": rng})
        drawn = active.apply(params, embeddings, method=lambda module, values:
                             module.embedding_dropout(values, deterministic=False), rngs={"dropout": rng})
        np.testing.assert_array_equal(drawn, reference)
        evaluated = active.apply(params, embeddings, method=lambda module, values:
                                 module.embedding_dropout(values, deterministic=True))
        np.testing.assert_array_equal(evaluated, embeddings)


@pytest.mark.parametrize("geometry", [{"layer_types": ("local", "local"), "kinds": {"local": {"window": 2}}},
                                     {"layer_types": ("local", "local"), "kinds": {"local": {"chunk": 2}}},
                                     {"num_kv_heads": 2}])
def test_attention_dropout_reaches_local_and_grouped_heads(rng, geometry):
    model = tiny(attention_dropout_rate=.5, attention_impl="reference", **geometry)
    ids = tokens(rng, length=4)
    params = model.init(rng, ids)
    one = model.apply(params, ids, train=True, rngs={"dropout": jax.random.key(1)})
    two = model.apply(params, ids, train=True, rngs={"dropout": jax.random.key(2)})
    assert not jnp.array_equal(one, two)
    baseline = tiny(attention_impl="reference", **geometry).apply(params, ids)
    evaluated = model.apply(params, ids)
    assert np.asarray(evaluated).tobytes() == np.asarray(baseline).tobytes()


def test_old_decoder_record_keeps_zero_dropout_and_bitwise_outputs():
    from flax.traverse_util import unflatten_dict
    from jax import export

    from dew.config import ModelConfig

    config = {"vocab_size": 17, "emb_features": 8, "num_layers": 1, "num_heads": 2,
                  "mlp_features": 16, "max_seq_len": 8}
    model = ModelConfig("causal_transformer", {**config, "dtype": "float32",
                                              "precision": "highest", "attention_impl": "reference"}).build()
    assert model.embedding_dropout_rate == model.attention_dropout_rate == 0
    directory = Path(__file__).with_name("fixtures")
    held = np.load(directory / "decoder-default-dropout.npz")
    try:
        before = export.deserialize((directory / "decoder-default-dropout.jaxexport").read_bytes())
    except Exception as error:
        pytest.fail(f"cannot deserialize the pre-dropout forward/gradient export: {error}")
    with jax.default_device(jax.devices("cpu")[0]):
        ids = jnp.asarray(held["ids"])
        params = unflatten_dict({tuple(name.split("/")[1:]): jnp.asarray(leaf)
                                 for name, leaf in held.items() if name.startswith("params/")})

        def forward_and_gradient(params, ids):
            forward = model.apply(params, ids, train=True)
            gradient = jax.grad(lambda p: jnp.sum(model.apply(p, ids, train=True)))(params)
            return forward, gradient

        expected = before.call(params, ids)
        actual = jax.jit(forward_and_gradient)(params, ids)
        assert jax.tree.structure(actual) == jax.tree.structure(expected)
        for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            assert np.asarray(left).tobytes() == np.asarray(right).tobytes()


def test_attention_probability_dropout_matches_flax_on_the_same_draw(rng):
    query = jnp.zeros((1, 4, 2, 4), jnp.float32)
    value = jnp.broadcast_to(jnp.eye(4)[None, :, None, :], (1, 4, 2, 4))
    mask = jnp.tril(jnp.ones((4, 4), bool))[None, None]
    expected = nn.dot_product_attention(query, query, value, mask=mask, dropout_rate=.5,
                                        dropout_rng=rng, broadcast_dropout=False, deterministic=False)
    actual = scaled_dot_product_attention(query, query, value, causal=True, implementation="reference",
                                          dropout_rate=.5, dropout_rng=rng, deterministic=False)
    assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()
    probabilities = np.asarray(actual)[0]
    for row in range(4):
        kept = np.float32((1 / (row + 1)) / (1 - .5))
        visible = probabilities[row, :, :row + 1]
        assert np.all((visible == 0) | (visible == kept))
        assert not np.any(probabilities[row, :, row + 1:]), "future probabilities stay zero"
        keep_counts = np.count_nonzero(visible, axis=-1)
        np.testing.assert_array_equal(keep_counts, np.sum(visible == kept, axis=-1))
    assert jnp.any(actual == 0)
    different = scaled_dot_product_attention(query, query, value, causal=True, implementation="reference",
        dropout_rate=.5, dropout_rng=jax.random.key(7), deterministic=False)
    assert not jnp.array_equal(actual, different)
    evaluated = scaled_dot_product_attention(query, query, value, causal=True, implementation="reference",
                                              dropout_rate=.5, deterministic=True)
    unchanged = scaled_dot_product_attention(query, query, value, causal=True, implementation="reference")
    assert np.asarray(evaluated).tobytes() == np.asarray(unchanged).tobytes()
    for implementation in ("auto", "xla"):
        routed = scaled_dot_product_attention(query, query, value, causal=True, implementation=implementation,
                                               dropout_rate=.5, dropout_rng=rng, deterministic=False)
        assert np.asarray(routed).tobytes() == np.asarray(expected).tobytes()


@pytest.mark.parametrize("implementation", ["cudnn", "tpu"])
def test_attention_dropout_refuses_kernels_that_cannot_drop_probabilities(rng, implementation):
    query = jnp.ones((1, 4, 2, 4), jnp.float32)
    with pytest.raises(ValueError):
        scaled_dot_product_attention(query, query, query, implementation=implementation,
                                      dropout_rate=.5, dropout_rng=rng, deterministic=False)


def test_attention_dropout_preserves_visibility_masks(rng):
    query = jnp.zeros((1, 4, 2, 4), jnp.float32)
    value = jnp.broadcast_to(jnp.eye(4)[None, :, None, :], (1, 4, 2, 4))
    documents = jnp.asarray([[1, 1, 2, 2]])
    visible = (documents[:, :, None] == documents[:, None, :])[:, None]
    visible = visible & jnp.tril(jnp.ones((4, 4), bool))[None, None]
    expected = nn.dot_product_attention(query, query, value, mask=visible, dropout_rate=.5,
                                        dropout_rng=rng, broadcast_dropout=False, deterministic=False)
    actual = scaled_dot_product_attention(query, query, value, causal=True, segment_ids=documents,
        implementation="auto", dropout_rate=.5, dropout_rng=rng, deterministic=False)
    assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()
    assert not np.any(np.asarray(actual)[0, 2:, :, :2]), "documents never see an earlier document"


def test_attention_dropout_refuses_a_mixer_without_probabilities(rng):
    with pytest.raises(ValueError):
        tiny(attention_dropout_rate=.5, mixer={"class": "mamba2", "fields": {}}).init(
            rng, tokens(rng, length=4))


@pytest.mark.parametrize("field", ["embedding_dropout_rate", "attention_dropout_rate"])
@pytest.mark.parametrize("rate", [-.1, 1.1])
def test_decoder_dropout_rates_must_be_probabilities(rng, field, rate):
    with pytest.raises(ValueError):
        tiny(**{field: rate}).init(rng, tokens(rng, length=4))


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


@pytest.mark.parametrize("attention_impl", ['reference', 'xla'])
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


def test_a_softcapped_call_is_the_capped_softmax_on_every_path_it_takes(rng, monkeypatch):
    """Gemma 2's tanh on the logits, against the equation on a hand-computed
    case: reference and xla agree with it, and 'auto' resolves a softcapped
    call to that same math even where cudnn would otherwise run (bf16 inputs,
    an 8-wide head, a gpu backend), while cudnn raises a ValueError naming
    the implementation. A cap of 2 on logits of order 10 changes the
    weights by a wide margin; an uncapped path fails the comparison."""
    query, key, value = (jax.random.normal(k, (1, 6, 2, 8)) * 3
                         for k in jax.random.split(rng, 3))
    scaled = np.einsum('bqhd,bkhd->bhqk', query, key) / np.sqrt(8)
    capped = 2.0 * np.tanh(scaled / 2.0)
    causal = np.tril(np.ones((6, 6), bool))
    weights = jax.nn.softmax(np.where(causal, capped, -np.inf), axis=-1)
    expected = np.einsum('bhqk,bkhd->bqhd', weights, value)
    uncapped = scaled_dot_product_attention(query, key, value, causal=True)
    assert np.max(np.abs(np.asarray(uncapped) - expected)) > 0.1

    reference = scaled_dot_product_attention(query, key, value, causal=True, softcap=2.0)
    fused = scaled_dot_product_attention(query, key, value, causal=True, softcap=2.0,
                                         implementation='xla')
    assert np.allclose(reference, expected, atol=1e-5)
    assert np.allclose(fused, expected, atol=1e-5)

    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    halves = (query.astype(jnp.bfloat16), key.astype(jnp.bfloat16), value.astype(jnp.bfloat16))
    routed = scaled_dot_product_attention(*halves, causal=True, softcap=2.0,
                                          implementation='auto')
    assert np.allclose(routed.astype(jnp.float32), expected, atol=5e-2)
    with pytest.raises(ValueError):
        scaled_dot_product_attention(*halves, causal=True, softcap=2.0, implementation='cudnn')


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
    model = tiny(layer_types=('sliding_attention',) * 2,
                 kinds={'sliding_attention': {'window': 3}})
    ids = tokens(rng)
    params = model.init(rng, ids)
    baseline = model.apply(params, ids)

    flipped = 3
    changed = model.apply(params, ids.at[:, flipped].set((ids[:, flipped] + 5) % VOCAB))
    moved = jnp.abs(baseline - changed).max(axis=(0, 2)) > 1e-5
    assert [int(index) for index in jnp.where(moved)[0]] == list(
        range(flipped, flipped + 5))


def bidirectional_reach(rng, implementation, path, kind):
    """Which positions move when token 6 of a two-layer bidirectional
    sliding stack changes, through the kernel's window, the validity mask or
    packed documents (one document, so the reach is the same)."""
    model = tiny(layer_types=('sliding_attention',) * 2, causal=False, attention_impl=implementation,
                 kinds={'sliding_attention': kind})
    ids = tokens(rng)
    params = model.init(rng, ids)
    inputs = {"plain": {}, "valid": {"attention_mask": jnp.ones(ids.shape, bool)},
              "packed": {"segment_ids": jnp.ones(ids.shape, jnp.int32),
                         "positions": jnp.broadcast_to(jnp.arange(SEQ), ids.shape)}}[path]
    baseline = model.apply(params, ids, **inputs)
    changed = model.apply(params, ids.at[:, 6].set((ids[:, 6] + 5) % VOCAB), **inputs)
    moved = jnp.abs(baseline - changed).max(axis=(0, 2)) > 1e-5
    return [int(index) for index in jnp.where(moved)[0]]


@pytest.mark.parametrize("implementation", ["reference", "xla"])
@pytest.mark.parametrize("path", ["plain", "valid", "packed"])
def test_a_two_sided_window_reaches_both_sides(rng, implementation, path):
    """Two layers of a two-sided window of 3 see 2 keys either side each, so
    a token moves the 4 on either side of it and nothing further, on every
    path. ModernBERT's window is |q - k| < w (`window_sides`)."""
    reach = bidirectional_reach(rng, implementation, path, {'window': 3, 'bidirectional_window': True})
    assert reach == list(range(2, 11))


@pytest.mark.parametrize("implementation", ["reference", "xla"])
@pytest.mark.parametrize("path", ["plain", "valid", "packed"])
def test_a_bidirectional_layer_without_a_two_sided_window_reads_its_whole_row(rng, implementation, path):
    """DiffusionGemma's decoder attends every canvas key whatever its sliding
    layers' window (modeling_diffusion_gemma.py:1399-1401), so a
    bidirectional layer whose kind does not keep its window on both sides
    reads the whole row on every path."""
    assert bidirectional_reach(rng, implementation, path, {'window': 3}) == list(range(SEQ))


def test_sliding_attention_decode_matches_the_full_sequence(rng):
    model = tiny(layer_types=('full_attention', 'sliding_attention'),
                 kinds={'sliding_attention': {'window': 4}})
    ids = tokens(rng)
    params = model.init(rng, ids)
    full = model.apply(params, ids)
    incremental = decode_logits(model, params, ids[:, :6], ids[:, 6:])
    assert jnp.allclose(full[:, 5:], incremental, atol=1e-4)


def hybrid(**overrides):
    """Qwen3.5's stack at toy size: three gated-delta-net layers, one gated
    full-attention layer with a partial rotary."""
    return tiny(num_layers=4, num_kv_heads=2, head_dim=8, output_gate=True,
                partial_rotary_factor=0.5,
                layer_types=('linear_attention',) * 3 + ('full_attention',),
                kinds={'linear_attention': {'mixer': {
                    'class': 'gated_delta_net', 'fields': {'linear_num_key_heads': 2,
                    'linear_num_value_heads': 4, 'linear_key_head_dim': 6,
                    'linear_value_head_dim': 8, 'linear_conv_kernel_dim': 4}}}},
                **overrides)


def test_a_hybrid_stack_decodes_as_it_scores_in_parallel(rng):
    """The test that catches a wrong recurrent state: a prefill of four
    tokens and eight single-token steps, every layer's state riding the
    flax cache collection (the delta net's recurrent memory and conv tail,
    the attention's KV cache), against the same twelve tokens scored at
    once. Largest observed logit difference 5.1e-06 on logits of magnitude
    3.1, every argmax equal. A delta net that stops writing its memory back
    (`recurrent.value = final` dropped) moves the logits by 3.8e+00."""
    model = hybrid()
    ids = tokens(rng)
    params = model.init(rng, ids)
    full = model.apply(params, ids)

    cache = model.apply(params, ids.shape[0], method=CausalTransformer.init_cache,
                        mutable=['cache'])[1]['cache']

    assert all(not jnp.any(leaf) for leaf in jax.tree.leaves(cache))

    incremental = decode_logits(model, params, ids[:, :4], ids[:, 4:])

    difference = float(jnp.abs(full[:, 3:] - incremental).max())
    assert difference < 1e-4, f"max |logit difference| {difference:.3e}"
    assert jnp.array_equal(full[:, 3:].argmax(-1), incremental.argmax(-1))

    _, mutated = model.apply({**params, "cache": cache}, ids[:, :4], decode=True, mutable=["cache"])
    kept, _ = model.apply({**params, "cache": mutated["cache"]}, ids[:, 4:5], decode=True, mutable=["cache"])
    zeroed = jax.tree_util.tree_map_with_path(
        lambda path, leaf: jnp.zeros_like(leaf) if "conv_state" in jax.tree_util.keystr(path) else leaf,
        mutated["cache"])
    reset, _ = model.apply({**params, "cache": zeroed}, ids[:, 4:5], decode=True, mutable=["cache"])
    assert not jnp.allclose(kept, reset, atol=1e-4)


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


def test_a_kinds_rope_base_only_moves_that_kinds_layers(rng):
    """A kind's own rope base is Gemma3's second one: it must reach the
    layers the pattern names and leave the others alone."""
    ids = tokens(rng)
    pattern = ('full_attention', 'sliding_attention')
    model = tiny(layer_types=pattern, kinds={'sliding_attention': {'window': 4}})
    params = model.init(rng, ids)
    same_theta = tiny(layer_types=pattern,
                      kinds={'sliding_attention': {'window': 4, 'rope_theta': 10000.0}})
    assert jnp.allclose(model.apply(params, ids), same_theta.apply(params, ids))
    other_theta = tiny(layer_types=pattern,
                       kinds={'sliding_attention': {'window': 4, 'rope_theta': 1e6}})
    assert not jnp.allclose(model.apply(params, ids), other_theta.apply(params, ids))

    pattern = ("sliding_attention", "sliding_attention")
    both = tiny(rope_theta=111.0, layer_types=pattern,
                kinds={"sliding_attention": {"window": 4, "rope_theta": 1e6}})
    precedence_params = both.init(rng, ids)
    kind_only = tiny(layer_types=pattern, kinds={"sliding_attention": {"window": 4, "rope_theta": 1e6}})
    model_only = tiny(rope_theta=1e6, layer_types=pattern, kinds={"sliding_attention": {"window": 4}})
    assert jnp.allclose(both.apply(precedence_params, ids), kind_only.apply(precedence_params, ids))
    assert jnp.allclose(both.apply(precedence_params, ids), model_only.apply(precedence_params, ids))


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


def test_a_post_norm_block_is_residual_plus_normed_sublayer_output(rng):
    """OLMo 3's block (input norms off, output norms on): x + norm(attn(x)), then
    x + norm(mlp(x)), with no input norms (modeling_olmo3.py:249-266). The
    block's output is checked against that equation computed from its own
    sublayers, and against the pre-norm block on the same weights, which
    normalises the input instead and lands elsewhere."""
    from dew.nn.backbones.causal_transformer import RMSNorm
    from dew.nn.backbones.decoder_block import BlockWiring, DecoderBlock
    from dew.nn.mixers import AttentionMixer, MixerContext
    from dew.nn.mlp import GatedMLP

    features, ids = 32, tokens(rng)
    x = jax.random.normal(rng, (*ids.shape, features)) * 3
    mixer = AttentionMixer().build(MixerContext(
        emb_features=features, num_heads=4, num_kv_heads=4, head_dim=8, max_seq_len=SEQ))
    feedforward = functools.partial(GatedMLP, hidden_features=64, out_features=features)
    block = DecoderBlock(mixer=mixer, feedforward=feedforward, emb_features=features,
                         wiring=BlockWiring(pre_norms=False, output_norms=True))
    params = block.init(rng, x)
    assert set(params['params']) == {'self_attn', 'mlp', 'attention_output_norm', 'mlp_output_norm'}

    def norm(name, value):
        return RMSNorm(epsilon=block.norm_eps).apply({'params': params['params'][name]}, value)

    attended = mixer(name='self_attn').apply({'params': params['params']['self_attn']}, x)
    mid = x + norm('attention_output_norm', attended)
    fed = feedforward(name='mlp').apply({'params': params['params']['mlp']}, mid)
    expected = mid + norm('mlp_output_norm', fed)
    assert jnp.allclose(block.apply(params, x), expected, atol=1e-5)

    pre = DecoderBlock(mixer=mixer, feedforward=feedforward, emb_features=features,
                       wiring=BlockWiring(output_norms=True))
    with_input_norms = pre.init(rng, x)
    with_input_norms['params'].update(params['params'])
    assert not jnp.allclose(pre.apply(with_input_norms, x), expected, atol=1e-2)


def test_the_embedding_scale_is_not_rounded_to_the_activation_dtype(rng):
    """Gemma casts embed_scale to the embedding weight dtype
    (modeling_gemma3.py:117). Dew's nn.Embed holds fp32 parameters and
    returns the compute dtype, so under the bf16 policy a run uses the two
    dtypes differ. At hidden 1152 the factor is 33.94112549695428, not
    bf16(33.941) = 34.0, which is 1.7e-03 of every embedding.

    Folding the factor into the table gives the value the module has to
    produce. The table rounds to bf16 because the lookup rounds it, the fp32
    factor multiplies that, and the product rounds once, so the residual
    stream stays in the activation dtype; an fp32 product would carry the
    whole stack in fp32 and land elsewhere. The head is untied so the fold
    only moves the input side. The fp32 Gemma fixture parity test cannot see
    any of this, since an fp32 policy rounds the factor to itself, and
    gemma3-tiny is hidden 64, where the factor is 8.0 in either dtype.
    """
    features, ids = 1152, tokens(rng, length=4)
    shared = {"emb_features": features, "num_heads": 8, "num_layers": 1,
                  "tie_embeddings": False, "dtype": jnp.bfloat16}
    scaled = tiny(embedding_scale=True, **shared)
    params = scaled.init(rng, ids)
    assert params['params']['embed_tokens']['embedding'].dtype == jnp.float32

    def fold(factor):
        return jax.tree_util.tree_map_with_path(
            lambda path, leaf: leaf.astype(jnp.bfloat16) * factor
            if path[-2].key == 'embed_tokens' else leaf, params)

    assert jnp.array_equal(
        scaled.apply(params, ids),
        tiny(**shared).apply(fold(jnp.float32(math.sqrt(features))), ids))
    assert not jnp.array_equal(
        scaled.apply(params, ids),
        tiny(**shared).apply(fold(jnp.bfloat16(34.0)), ids))


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


def test_the_attention_scale_is_not_rounded_to_the_activation_dtype(rng):
    """transformers hands query_pre_attn_scalar ** -0.5 to the attention call
    as a float (modeling_gemma3.py:318, 376), and the scale itself is unrounded.

    Gemma 3 27B asks for scalar 168 on head_dim 128, where the ratio to the
    kernel's own 1/sqrt(head_dim) is 0.872872 and bf16 holds it as 0.871094.
    A bf16 run that rounds the ratio first cannot tell that scale from the one
    whose ratio is exactly 0.871094, and scales every logit 0.2% low.
    """
    ids = tokens(rng)
    shared = {"head_dim": 128, "num_layers": 1, "dtype": jnp.bfloat16}
    exact = tiny(attention_scale=168 ** -0.5, **shared)
    params = exact.init(rng, ids)
    rounded = tiny(attention_scale=float(jnp.bfloat16(168 ** -0.5 * math.sqrt(128)))
                   / math.sqrt(128), **shared)

    assert not jnp.array_equal(exact.apply(params, ids), rounded.apply(params, ids))
    # None asks for the kernel's own scale, so no factor touches the query
    assert jnp.array_equal(tiny(**shared).apply(params, ids),
                           tiny(attention_scale=128 ** -0.5, **shared).apply(params, ids))


def test_a_prompt_longer_than_the_cache_is_refused(rng):
    model = tiny(max_seq_len=8)
    ids = tokens(rng, length=12)
    params = model.init(rng, ids)
    cache = model.apply(params, 2, method=CausalTransformer.init_cache,
                        mutable=['cache'])[1]['cache']
    with pytest.raises(ValueError):
        model.apply({**params, 'cache': cache}, ids, decode=True, mutable=['cache'])


@pytest.mark.parametrize("config, message", [
    ({'head_dim': 7}, "even"),
    ({'num_kv_heads': 3}, "multiple"),
    ({'layer_types': ('full_attention',)}, "entries"),
    ({'kinds': {'linear_attention': {'window': 2}}}, "name no layer"),
    ({'kinds': {'full_attention': {'window': 0}}}, "window"),
    ({'kinds': {'full_attention': {'head_dim': 7}}}, "even"),
    ({'use_double_wide_mlp': True}, "kv_shared_layers"),
    ({'per_layer_input_dim': 0}, "None is a model without them"),
    ({'mlp': 'unknown'}, "swiglu"),
])
def test_rejected_configs(rng, config, message):
    with pytest.raises(ValueError):
        tiny(**config).init(rng, tokens(rng))


# --- packed batches -------------------------------------------------------

def packed_pair(rng, first=6, second=6):
    """A two-document row, with the segment ids and positions grain emits."""
    ids = tokens(rng, length=first + second)
    segment_ids = jnp.asarray([[1] * first + [2] * second] * ids.shape[0])
    positions = jnp.asarray(
        [list(range(first)) + list(range(second))] * ids.shape[0])
    return ids, segment_ids, positions


def test_positions_default_to_the_row_index(rng):
    """Omitting positions means the row index, so an unpacked run scores the
    same whether the caller spells them out or not."""
    model = tiny()
    ids = tokens(rng)
    params = model.init(rng, ids)

    row_index = jnp.tile(jnp.arange(ids.shape[1]), (ids.shape[0], 1))
    assert jnp.array_equal(model.apply(params, ids),
                           model.apply(params, ids, positions=row_index))


def test_packed_attention_stays_causal_inside_a_document(rng):
    """The segment mask must sit on top of causality, not replace it."""
    model = tiny()
    ids, segment_ids, positions = packed_pair(rng)
    params = model.init(rng, ids)
    baseline = model.apply(params, ids, positions=positions, segment_ids=segment_ids)

    cut = 4
    rewritten = ids.at[:, cut:6].set((ids[:, cut:6] + 5) % VOCAB)
    changed = model.apply(params, rewritten, positions=positions,
                          segment_ids=segment_ids)
    assert jnp.array_equal(baseline[:, :cut], changed[:, :cut])
    assert not jnp.allclose(baseline[:, cut:6], changed[:, cut:6])


def test_a_packed_document_reads_like_the_document_alone(rng):
    """The second document's logits cannot depend on sitting after the first:
    same tokens, same per-document positions, same output."""
    model = tiny()
    ids, segment_ids, positions = packed_pair(rng)
    params = model.init(rng, ids)
    packed = model.apply(params, ids, positions=positions, segment_ids=segment_ids)

    alone = model.apply(params, ids[:, 6:])
    assert jnp.max(jnp.abs(packed[:, 6:] - alone)) < 1e-5


def test_a_packed_row_refuses_a_decode_cache(rng):
    """A cache stores no document ids, so a packed row appended to one would attend across its
    documents: the model refuses it, and runs the same row and positions into a cache without
    its segments."""
    model = tiny()
    ids, segment_ids, positions = packed_pair(rng)
    params = model.init(rng, ids)
    model.apply(params, ids, decode=True, positions=positions, mutable=["cache"])
    with pytest.raises(ValueError):
        model.apply(params, ids, decode=True, positions=positions, segment_ids=segment_ids,
                    mutable=["cache"])


def test_padding_in_a_packed_row_reaches_no_query(rng):
    model = tiny()
    ids = tokens(rng, length=8)
    segment_ids = jnp.asarray([[1] * 5 + [0] * 3] * ids.shape[0])
    positions = jnp.asarray([list(range(5)) + [0] * 3] * ids.shape[0])
    params = model.init(rng, ids)
    baseline = model.apply(params, ids, positions=positions, segment_ids=segment_ids)

    # Rewriting the padded tail cannot move a real token's logits, and the
    # padded rows themselves stay finite: no query divides by an empty
    # softmax.
    padded = ids.at[:, 5:].set((ids[:, 5:] + 11) % VOCAB)
    changed = model.apply(params, padded, positions=positions,
                          segment_ids=segment_ids)
    assert jnp.array_equal(baseline[:, :5], changed[:, :5])
    assert jnp.all(jnp.isfinite(changed))


def test_a_segment_masked_batch_leaves_the_cudnn_kernel(rng, without_deterministic_ops):
    """cuDNN takes causality as a flag and turns any mask into a materialized
    bias, so packed batches ride the xla kernel instead. Pinning cudnn here is
    what proves the routing: the model computes in fp32, which cudnn refuses,
    so an unpacked batch is refused while a packed one runs."""
    # The param tree does not depend on the kernel, so the tree comes from a
    # twin whose init is allowed to run: initialising the cudnn model itself
    # would trip the same refusal before the test could make its point.
    model = tiny(attention_impl='cudnn')
    ids, segment_ids, positions = packed_pair(rng)
    params = tiny().init(rng, ids)

    with pytest.raises(ValueError):
        model.apply(params, ids)

    # The fallback is the xla kernel with the packed mask, not a run with the
    # mask dropped, so the logits are the ones the xla model computes.
    logits = model.apply(params, ids, positions=positions, segment_ids=segment_ids)
    assert jnp.array_equal(logits, tiny(attention_impl='xla').apply(
        params, ids, positions=positions, segment_ids=segment_ids))


def test_metadata_that_restricts_no_visibility_keeps_the_fused_kernel(
        rng, without_deterministic_ops):
    """Rotary positions rotate q and k and narrow nobody's view, so a batch
    that carries them keeps causality as a kernel flag. Pinning cudnn is what
    proves the routing, as in the packed case above: cudnn refuses the fp32
    model, so a call that reaches it is refused, and a call a mask sent to xla
    runs.

    Validity is the other half of the contract. An array is opaque at trace
    time, so an all-true one still builds the mask and still rides xla."""
    model = tiny(attention_impl='cudnn')
    ids = tokens(rng)
    params = tiny().init(rng, ids)
    rotary = jnp.tile(jnp.arange(ids.shape[1]), (ids.shape[0], 1))

    with pytest.raises(ValueError):
        model.apply(params, ids, rotary_positions=rotary)

    valid = jnp.ones(ids.shape, bool)
    assert jnp.array_equal(model.apply(params, ids, attention_mask=valid),
                           tiny(attention_impl='xla').apply(params, ids, attention_mask=valid))


def test_metadata_that_restricts_no_visibility_scores_like_no_metadata(rng):
    """The rotary positions of an unpacked row are its row indices, so a batch
    that spells them out has to score exactly as a batch that omits them. Both
    sides run the same kernel with the same flags; the gradient graphs differ
    by one gather, so XLA may fuse their reductions differently and the
    gradients are compared at the file's fp32 bound rather than bitwise."""
    model = tiny(attention_impl='xla')
    ids = tokens(rng)
    params = model.init(rng, ids)
    rotary = jnp.tile(jnp.arange(ids.shape[1]), (ids.shape[0], 1))

    def scored(**metadata):
        return jax.grad(lambda tree: jnp.sum(model.apply(tree, ids, **metadata) ** 2))(params)

    assert jnp.array_equal(model.apply(params, ids),
                           model.apply(params, ids, rotary_positions=rotary))
    plain, spelled = scored(), scored(rotary_positions=rotary)
    for left, right in zip(jax.tree.leaves(plain), jax.tree.leaves(spelled), strict=True):
        assert jnp.max(jnp.abs(left - right)) < 1e-4 * max(1.0, float(jnp.max(jnp.abs(left))))


def test_a_padded_slot_reaches_no_query(rng):
    """Validity still excludes what it excludes. The padding has to sit where
    causality does not already hide it, so this row is padded on the left and
    holed in the middle: rewriting those slots cannot move a real token's
    logits, and no row comes back non-finite.

    A rule that read an opaque validity array as all-true would pass a
    right-padded row and fail here, which is why the padding is not on the
    right.
    """
    model = tiny()
    ids = tokens(rng, length=8)
    padded = [False] * 2 + [True] * 2 + [False] + [True] * 3
    valid = jnp.asarray([padded] * ids.shape[0])
    slots = jnp.asarray([index for index, real in enumerate(padded) if not real])
    real = jnp.asarray([index for index, real in enumerate(padded) if real])
    params = model.init(rng, ids)
    baseline = model.apply(params, ids, attention_mask=valid)

    changed = model.apply(params, ids.at[:, slots].set((ids[:, slots] + 11) % VOCAB),
                          attention_mask=valid)
    assert jnp.array_equal(baseline[:, real], changed[:, real])
    assert jnp.all(jnp.isfinite(changed))


@pytest.mark.parametrize("overrides", [
    {"attention_impl": 'xla'},
    {"num_kv_heads": 2, "attention_impl": 'xla'},
])
def test_packed_kernels_agree_with_the_reference(rng, overrides):
    """The reference kernel applies the segment mask itself and xla applies it
    inside jax.nn.dot_product_attention, on the same weights."""
    kernel = tiny(**overrides)
    reference = tiny(**{**overrides, "attention_impl": "reference"})
    ids, segment_ids, positions = packed_pair(rng)
    params = reference.init(rng, ids)

    expected = reference.apply(params, ids, positions=positions,
                               segment_ids=segment_ids)
    actual = kernel.apply(params, ids, positions=positions, segment_ids=segment_ids)
    # Largest difference observed on CPU: 1.5e-06.
    assert jnp.max(jnp.abs(expected - actual)) < 1e-4


def test_a_sliding_layer_packs_without_widening_its_window(rng):
    """A packed row folds the window into the same mask, so a sliding layer
    still forgets past it: two layers of a window of 3 reach 5 tokens back
    inside the document, and the boundary stops the reach early."""
    model = tiny(layer_types=('sliding_attention',) * 2,
                 kinds={'sliding_attention': {'window': 3}})
    ids, segment_ids, positions = packed_pair(rng)
    params = model.init(rng, ids)
    baseline = model.apply(params, ids, positions=positions, segment_ids=segment_ids)

    flipped = 1
    changed = model.apply(
        params, ids.at[:, flipped].set((ids[:, flipped] + 5) % VOCAB),
        positions=positions, segment_ids=segment_ids)
    moved = jnp.abs(baseline - changed).max(axis=(0, 2)) > 1e-5
    # Five tokens of reach, but the second document starts at 6, so the token
    # at 1 moves 1..5 and stops there, short of 6.
    assert [int(index) for index in jnp.where(moved)[0]] == list(
        range(flipped, flipped + 5))


def test_the_tied_head_rounds_its_bf16_product_to_bf16_logits_under_bf16_compute(rng):
    """Under bf16 compute the head multiplies the bf16 states by the table
    rounded to bf16, accumulates in fp32 and rounds the logits to bf16, as
    torch autocast's bf16 logits are: each logit is a bf16 value within a
    bf16 rounding (2^-8 relative) of the fp32 sum of the rounded operands,
    plus that sum's own fp32 rounding. At "highest" the head multiplies the
    fp32 table, and its logits are that fp32 product."""
    model = tiny(dtype=jnp.bfloat16)
    ids = tokens(rng)
    params = model.init(rng, ids)
    logits = model.apply(params, ids)
    hidden = model.apply(params, ids, method=CausalTransformer.hidden_states)
    table = params["params"]["embed_tokens"]["embedding"]
    exact = jnp.einsum("...d,vd->...v", hidden.astype(jnp.float32),
                       table.astype(jnp.bfloat16).astype(jnp.float32),
                       precision=jax.lax.Precision.HIGHEST)
    np.testing.assert_array_equal(
        np.asarray(logits), np.asarray(logits.astype(jnp.bfloat16).astype(jnp.float32))
    )
    assert np.all(np.abs(np.asarray(logits - exact)) <= 2 ** -8 * np.abs(np.asarray(exact)) + 1e-6)
    assert not np.allclose(np.asarray(logits), np.asarray(exact), atol=1e-6)

    strict = tiny(dtype=jnp.bfloat16, precision="highest")
    logits = strict.apply(params, ids)
    hidden = strict.apply(params, ids, method=CausalTransformer.hidden_states)
    fp32 = jnp.einsum("...d,vd->...v", hidden.astype(jnp.float32), table,
                      precision=jax.lax.Precision.HIGHEST)
    np.testing.assert_allclose(np.asarray(logits), np.asarray(fp32), atol=1e-6)


def test_the_rmsnorm_cast_order_is_a_field_that_bf16_tells_apart(rng):
    """Gemma scales in fp32 and casts the product; Llama and Qwen3 cast the
    normalized activations, then scale (modeling_qwen3.py:61-64). The
    two agree at fp32 and differ under bf16, and the HF translation picks per
    family."""
    from dew.interop.hf_decoders import translate_config
    from dew.nn.backbones.causal_transformer import RMSNorm

    x = jax.random.normal(rng, (2, 4, 32), jnp.bfloat16) * 3
    variables = {"params": {"scale": jax.random.uniform(rng, (32,), minval=0.5, maxval=1.5)}}
    gemma = RMSNorm(scale_after_cast=False).apply(variables, x)
    llama = RMSNorm(scale_after_cast=True).apply(variables, x)
    assert gemma.dtype == llama.dtype == jnp.bfloat16
    assert not jnp.array_equal(gemma, llama), "bf16 cannot tell the two orders apart"
    fp32 = x.astype(jnp.float32)
    np.testing.assert_allclose(np.asarray(RMSNorm(scale_after_cast=False).apply(variables, fp32)),
                               np.asarray(RMSNorm(scale_after_cast=True).apply(variables, fp32)),
                               rtol=1e-6)

    base = {"model_type": "llama", "hidden_size": 32, "num_hidden_layers": 1,
            "num_attention_heads": 4, "intermediate_size": 64, "vocab_size": 64,
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "hidden_act": "silu"}
    assert translate_config(base)["scale_after_cast"] is True
    assert translate_config({**base, "model_type": "qwen3", "head_dim": 8})["scale_after_cast"] is True
    gemma_config = {
        **base,
        "model_type": "gemma3_text",
        "head_dim": 8,
        "hidden_activation": "gelu_pytorch_tanh",
        "query_pre_attn_scalar": 8,
        "sliding_window": 4,
    }
    assert translate_config(gemma_config)["scale_after_cast"] is False


def test_a_cast_then_scale_norm_rounds_under_jit_and_keeps_an_fp32_weight_gradient():
    """Qwen3RMSNorm rounds the normalized activations to bf16, then multiplies
    by its weight: in fp32 for an fp32 master, and the next layer reads the
    product in bf16. Under jit XLA drops a narrowing cast that a widening one
    follows, and a bf16 product would reduce the weight's gradient in bf16
    (1e-2 relative). The oracle is NumPy: the fp32 normalization, rounded to
    bf16, times the fp32 weight in fp32, rounded to bf16; the gradient of
    sum(out * c) is sum over rows of c * round(y) in float64, which an fp32
    reduction over 64 rows meets to about 1e-7."""
    from dew.nn.attention import RMSNorm

    x = (jax.random.normal(jax.random.key(0), (64, 512)) * 3).astype(jnp.bfloat16)
    weight = 1.0 + 0.1 * jax.random.normal(jax.random.key(1), (512,), jnp.float32)
    # The cotangent of a bf16 output is bf16, so c holds bf16 values.
    c = jax.random.normal(jax.random.key(2), (64, 512)).astype(jnp.bfloat16).astype(jnp.float32)
    norm = RMSNorm(epsilon=1e-6, scale_after_cast=True, dtype=jnp.bfloat16)

    x32 = np.asarray(x, np.float32)
    inverse = np.float32(1) / np.sqrt(np.mean(np.square(x32), -1, keepdims=True) + np.float32(1e-6))
    rounded = (x32 * inverse).astype(jnp.bfloat16).astype(np.float32)
    expected = (rounded * np.asarray(weight)).astype(jnp.bfloat16)
    expected_grad = np.sum(np.asarray(c, np.float64) * rounded, axis=0)

    def loss(weight):
        return jnp.sum(norm.apply({"params": {"scale": weight}}, x).astype(jnp.float32) * c)

    out = jax.jit(norm.apply)({"params": {"scale": weight}}, x)
    assert out.dtype == jnp.bfloat16
    np.testing.assert_array_equal(np.asarray(out), expected)
    grad = np.asarray(jax.jit(jax.grad(loss))(weight), np.float64)
    assert np.max(np.abs(grad - expected_grad)) <= 1e-5 * np.max(np.abs(expected_grad))


@pytest.mark.mesh(devices=4)
@pytest.mark.parametrize("mixture", [None, {"experts": 8, "top_k": 2, "expert_features": 32}])
def test_a_bf16_decoder_scores_the_same_on_one_device_and_split_over_four(mixture):
    """The rows are independent, so splitting the batch over devices changes
    no token's arithmetic. With XLA free to skip roundings, what fused on one
    device and on four differed: the hidden states of a 4-layer bf16 decoder
    matched in 13% of entries, an 8-expert MoE's top-2 routes differed in
    0.3% of layer 0's choices and 2.8% of layer 3's, and its loss moved
    2.6e-4. Every rounding kept, they are bitwise the same."""
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    config = {"vocab_size": 512, "emb_features": 64, "num_layers": 4, "num_heads": 8, "num_kv_heads": 4,
                  "head_dim": 8, "mlp_features": 128, "max_seq_len": 33, "dtype": jnp.bfloat16}
    if mixture is not None:
        config["mixture"] = {**mixture, "layers": (0, 1, 2, 3)}
    model = CausalTransformer(**config)
    ids = jax.random.randint(jax.random.key(0), (16, 32), 0, 512)
    params = model.init(jax.random.key(1), ids)

    def hidden(params, ids):
        return model.apply(params, ids, method=CausalTransformer.hidden_states)

    one = jax.jit(hidden)(params, ids)
    mesh = Mesh(np.asarray(jax.devices()[:4]), ("data",))
    split = jax.jit(hidden)(params, jax.device_put(ids, NamedSharding(mesh, PartitionSpec("data"))))

    np.testing.assert_array_equal(np.asarray(one), np.asarray(split))


@pytest.mark.parametrize("activation", ["swiglu", "geglu", "geglu_exact"])
def test_an_expert_gated_product_rounds_its_projections_and_product_under_jit(activation):
    """An expert's gate and up come out of `expert_projection`, whose fp32
    sum XLA can carry into the activation past the sum's bf16 cast. One
    expert takes the fixture's gate g and up u through projections that sum
    each with an eighth of its bf16 ulp, which rounds back to g and u only
    if the projection's rounding survives, and a down projection that hands
    the product out unchanged: the MLP's output and its input's gradient
    must be Torch's product and gradients of the same gate and up."""
    from dew.nn.moe import ExpertMLP

    fixture = np.load(Path(__file__).parent / "fixtures" / "gated_product" / "bf16.npz")
    tokens, width = 16, 128
    gate, up, cotangent, output, d_gate, d_up = (
        np.asarray(fixture[f"{activation}/{key}"], np.float32).reshape(tokens, width)
        for key in ("gate", "up", "cotangent", "output", "d_gate", "d_up"))

    def eighth_ulp(values):
        magnitude = np.abs(values)
        exponent = np.floor(np.log2(np.where(magnitude > 0, magnitude, 1.0)))
        return np.where(magnitude > 0, np.sign(values) * 2.0 ** (exponent - 10), 0.0).astype(np.float32)

    eye, zero = np.eye(width, dtype=np.float32), np.zeros((width, width), np.float32)
    kernels = {"gate_proj": np.vstack([eye, eye, zero, zero]), "up_proj": np.vstack([zero, zero, eye, eye]),
               "down_proj": np.hstack([eye, zero, zero, zero])}
    variables = {"params": {name: {"kernel": jnp.asarray(np.stack([kernel] * 2))}
                            for name, kernel in kernels.items()}}
    model = ExpertMLP(2, width, 4 * width, activation=activation, dtype=jnp.bfloat16)
    x = jnp.asarray(np.hstack([gate, eighth_ulp(gate), up, eighth_ulp(up)]), jnp.bfloat16)
    weights, indices = jnp.ones((tokens, 1), jnp.float32), jnp.zeros((tokens, 1), jnp.int32)
    slot_cotangent = jnp.asarray(np.hstack([cotangent] + [np.zeros_like(cotangent)] * 3), jnp.bfloat16)

    def forward_and_backward(x, cotangent):
        out, pull = jax.vjp(lambda x: model.apply(variables, x, weights, indices), x)
        return out, pull(cotangent)[0]

    out, dx = jax.jit(forward_and_backward)(x, slot_cotangent)
    np.testing.assert_array_equal(np.asarray(out, np.float32),
                                  np.hstack([output] + [np.zeros_like(output)] * 3), err_msg="output")
    np.testing.assert_array_equal(np.asarray(dx, np.float32), np.hstack([d_gate, d_gate, d_up, d_up]),
                                  err_msg="input gradient")


def test_exclusive_self_attention_removes_the_own_value_direction_per_query_head():
    """The float64 projection y - (<y, v> / <v, v>) v with each value head
    repeated over its query group, forward and backward; the result is
    orthogonal to the token's own value, and a zero value leaves y alone
    where lm-engine's unguarded division is NaN."""
    from jax.test_util import check_grads

    from dew.nn.mixers.attention import exclusive_self_attention
    with jax.enable_x64(new_val=True):
        keys = jax.random.split(jax.random.key(0), 2)
        y = jax.random.normal(keys[0], (2, 5, 4, 8), jnp.float64)
        v = jax.random.normal(keys[1], (2, 5, 2, 8), jnp.float64).at[1, 3, 1].set(0.0)

        def oracle(y, v):
            v = np.repeat(np.asarray(v), 2, axis=-2)
            norm = np.sum(v * v, -1, keepdims=True)
            safe = np.where(norm > 0, norm, 1)
            return (
                np.asarray(y) - np.where(norm > 0, np.sum(np.asarray(y) * v, -1, keepdims=True) / safe, 0) * v
            )

        out = exclusive_self_attention(y, v)
        np.testing.assert_allclose(out, oracle(y, v), rtol=1e-13, atol=1e-13)
        own = jnp.repeat(v, 2, axis=-2)
        np.testing.assert_allclose(jnp.sum(out * own, -1), 0.0, atol=1e-12)
        np.testing.assert_array_equal(out[1, 3, 2:], y[1, 3, 2:])
        # Backward against finite differences, away from the zero value,
        # where the projection jumps and no derivative exists.
        smooth = jax.random.normal(keys[1], (2, 5, 2, 8), jnp.float64)
        check_grads(exclusive_self_attention, (y, smooth), order=1, modes=["rev"])


def nope_model(**overrides) -> CausalTransformer:
    return CausalTransformer(vocab_size=32, emb_features=32, num_layers=1, num_heads=4,
                             num_kv_heads=2, max_seq_len=16, qk_norm=False,
                             mixer=AttentionMixer(nope=True), **overrides)


def test_a_nope_layer_reads_no_positions_and_keeps_its_logit_scale():
    """Without rotation the positions a caller hands in change nothing, and
    attention_scale still scales the logits as it does with rope."""
    tokens = jax.random.randint(jax.random.key(1), (2, 12), 0, 32)
    model = nope_model()
    variables = model.init(jax.random.key(0), tokens)
    plain = model.apply(variables, tokens)
    shifted = model.apply(variables, tokens, positions=jnp.arange(12) + 100)
    np.testing.assert_array_equal(plain, shifted)
    roped = CausalTransformer(vocab_size=32, emb_features=32, num_layers=1, num_heads=4,
                              num_kv_heads=2, max_seq_len=16, qk_norm=False)
    assert not np.allclose(roped.apply(variables, tokens), plain)
    scaled = nope_model(attention_scale=1.0).apply(variables, tokens)
    assert not np.allclose(scaled, plain)


def test_an_xsa_nope_mup_model_decodes_what_its_full_forward_scores(rng):
    """Prefill and token-by-token decode read the same logits as one forward
    for a Rigel-style layer: XSA subtracts the new token's own value while
    decoding, no rotation reads the cache slot, and the multipliers and the
    logit division apply on both paths."""
    model = tiny(qk_norm=False, mixer=AttentionMixer(nope=True, exclusive_self_attention=True),
                 embedding_multiplier=12.0, residual_multiplier=0.22, logits_scaling=4.0,
                 initializer_range=0.1, depth_scaled_init=True)
    ids = tokens(rng, length=10)
    params = model.init(rng, ids)
    full = model.apply(params, ids)
    stepped = decode_logits(model, params, ids[:, :4], ids[:, 4:])
    np.testing.assert_allclose(stepped, full[:, 3:], rtol=1e-4, atol=1e-5)


def test_nope_and_xsa_are_the_attention_mixers_own_switches():
    """Only the grouped-query mixer rotates by rope and subtracts its own
    value, so the switches live on it and a mixer that cannot honour them
    cannot be handed them."""
    from dew.nn.backbones.layer_plan import LayerKind
    ids = tokens(jax.random.key(0), length=8)
    record = {"class": "attention", "fields": {"nope": True, "exclusive_self_attention": True}}
    kinded = tiny(qk_norm=False, layer_types=("a", "a"), kinds={"a": LayerKind(mixer=record)})
    params = kinded.init(jax.random.key(1), ids)
    plain = tiny(qk_norm=False).apply(params, ids)
    assert not np.allclose(kinded.apply(params, ids), plain)


@pytest.mark.parametrize("extra", [{"laurel_rank": 8}, {"per_layer_input_dim": 4},
                                   {"altup": {"num_inputs": 2}},
                                   {"hyper_connections": HyperConnections(hc_mult=2)}])
def test_root_multipliers_scale_around_blocks_that_keep_their_own(extra):
    """`embedding_multiplier` scales the embeddings before the first layer and
    `logits_scaling` divides the final states, outside every block. With
    LAuReL, AltUp, per-layer inputs or mHC's stream stack, a model with both
    runs as the model without them over its embedding table times the
    multiplier, its logits divided by the scaling: bitwise, both being powers
    of two, which round nothing."""
    ids = tokens(jax.random.key(0), length=8)
    plain = tiny(tie_embeddings=False, **extra)
    variables = plain.init(jax.random.key(1), ids)
    table = ("params", "embed_tokens", "embedding")
    multiplied = jax.tree_util.tree_map_with_path(
        lambda path, leaf: leaf * 2 if tuple(key.key for key in path) == table else leaf, variables)
    scaled = tiny(tie_embeddings=False, embedding_multiplier=2.0, logits_scaling=4.0, **extra)
    np.testing.assert_array_equal(np.asarray(scaled.apply(variables, ids)),
                                  np.asarray(plain.apply(multiplied, ids)) / 4)


def test_a_decode_step_reads_each_key_head_once_for_its_group(rng):
    """A decode step's grouped query heads reach the attention as their key
    head's query positions, [rows, group, kv_heads, width], so a kernel reads
    each key head once (cuDNN ran each query head on its own and padded the
    lone query to two: 0.154 against 0.146 ms a layer of Qwen3-0.6B at 64
    rows on an RTX 4080). The steps decode what the full pass computes."""
    model = tiny(num_heads=4, num_kv_heads=2, head_dim=8)
    ids = tokens(rng)
    params = model.init(rng, ids)
    full = model.apply(params, ids)
    prompt, rest = ids[:, :4], ids[:, 4:]
    incremental = decode_logits(model, params, prompt, rest)
    np.testing.assert_allclose(np.asarray(incremental), np.asarray(full[:, 3:]), atol=1e-5)


@pytest.mark.gpu(reason="needs a GPU")
def test_a_plain_decode_step_attends_through_cudnn_where_it_runs(rng, without_deterministic_ops):
    """A decode step's keys are the cache's filled slots, a count per row, so
    they reach cuDNN as its padding lengths rather than as a materialized mask:
    that mask sent Qwen3-0.6B's decode to xla's two dense dots over the whole
    cache on an RTX 4080 (1.00 against cuDNN's 0.30 ms a layer at 128 rows),
    and a small model to cuDNN's bias variant, which reads it per head. The
    steps still decode what the full pass computes."""
    model = tiny(dtype=jnp.bfloat16, num_kv_heads=2, head_dim=32)
    ids = tokens(rng)
    params = model.init(rng, ids)
    full = model.apply(params, ids)
    prompt, rest = ids[:, :4], ids[:, 4:]
    incremental = decode_logits(model, params, prompt, rest)
    np.testing.assert_allclose(np.asarray(incremental, np.float32),
                               np.asarray(full[:, 3:], np.float32), atol=0.05)



@pytest.mark.gpu(reason="needs a GPU")
def test_a_padded_prefill_attends_through_cudnn_where_it_runs(rng, without_deterministic_ops):
    """A cache prefill of left-padded prompts builds its cursor mask, which
    sends a training call to xla (`kernel_for_materialized_mask`); a prefill
    runs forward only, so cuDNN takes that mask as its bias (xla's dense dots
    took 6.2 against 4.0 ms over Qwen3-0.6B's 28 layers for 8 prompts of 256
    on an RTX 4080). The prefill still scores what the full pass does."""
    model = tiny(dtype=jnp.bfloat16, num_kv_heads=2, head_dim=32)
    ids = tokens(rng)
    params = model.init(rng, ids)
    valid = jnp.ones(ids.shape, bool).at[0, :3].set(False)
    full = model.apply(params, ids, attention_mask=valid)
    cache = model.apply(params, ids.shape[0], method=CausalTransformer.init_cache,
                        mutable=['cache'])[1]['cache']

    def prefill(model):
        return jax.jit(lambda cache, ids, valid: model.apply(
            {**params, 'cache': cache}, ids, attention_mask=valid, decode=True, mutable=['cache'])[0])

    np.testing.assert_allclose(np.asarray(prefill(model)(cache, ids, valid), np.float32)[valid],
                               np.asarray(full, np.float32)[valid], atol=0.05)



@pytest.mark.parametrize("causal", [True, False])
def test_right_padded_rows_attend_causally_with_no_mask(rng, causal):
    """Rows declared right-padded reach the attention kernel as an unpadded
    call does, causal with no mask, which keeps 'auto' on its fused kernel
    (a mask sends it to xla's dense dots, `_restricts_visibility`); their
    real tokens' states are as exact as the masked call's, both from the
    masked call's in float64. A bidirectional layer's real queries read the
    padding after them, so it keeps the mask."""
    model = tiny(causal=causal)
    ids = tokens(rng)
    params = model.init(rng, ids)
    valid = jnp.arange(SEQ)[None, :] < jnp.asarray([[7], [SEQ]])

    def states(model, params, **declared):
        states = model.apply(params, ids, attention_mask=valid, method=CausalTransformer.hidden_states,
                             **declared)
        return np.asarray(states, np.float64)[np.asarray(valid)]

    masked = states(model, params)
    declared = states(model, params, right_padded=True)
    with jax.enable_x64(new_val=True):
        truth = states(model.clone(dtype=jnp.float64), widened(params))
    assert_as_exact_as_the_reference(declared, masked, truth, "right-padded states")


def _rotate_half_reference(x, freqs_cos, freqs_sin, scale=None):
    """`dew.nn.rope`'s rotate-half as it was through 2026-10-06, and as the HF
    decoders write it, `x cos + rotate_half(x) sin`, at least fp32."""
    cos = jnp.concatenate([freqs_cos, freqs_cos], axis=-1)
    sin = jnp.concatenate([freqs_sin, freqs_sin], axis=-1)
    cos, sin = (part[:, :, None, :] if part.ndim == 3 else part[None, :, None, :] for part in (cos, sin))
    wide = x.astype(jnp.promote_types(x.dtype, jnp.float32))
    wide, passed = wide[..., :cos.shape[-1]], wide[..., cos.shape[-1]:]
    x1, x2 = jnp.split(wide, 2, axis=-1)
    out = jnp.concatenate([wide * cos + jnp.concatenate([-x2, x1], axis=-1) * sin, passed], axis=-1)
    return (out if scale is None else out * scale).astype(x.dtype)


def test_a_per_axis_rotary_without_its_axes_is_refused():
    """Turning each axis in its own halves needs the axes; without them the
    layer would rotate by its positions and drop the per-axis layout."""
    from dew.nn.mixer_base import MixerContext
    from dew.nn.mixers.attention import CausalSelfAttention

    layer = CausalSelfAttention(MixerContext(emb_features=8, num_heads=2, num_kv_heads=2, head_dim=4,
                                             max_seq_len=4, causal=False), rotary_per_axis=True)
    with pytest.raises(ValueError):
        layer.init(jax.random.key(0), jnp.zeros((1, 4, 8)))


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("pairs, scale, packed", [(64, None, False), (64, 0.0884, False), (24, None, True)],
                         ids=["full", "scaled", "partial-packed"])
def test_the_rotary_rotates_the_halves_as_rotate_half_does(dtype, pairs, scale, packed):
    """The halves rotate as `x1 cos - x2 sin` and `x2 cos + x1 sin`, without
    rotate_half's rotated copy. The arithmetic is the same, its rounding
    order not quite, so the forward and the input gradient are each within
    tests/reference_error.py's rule of rotate_half's against float64: full
    and partial rotary, a folded scale, and a packed batch's own angles."""
    from reference_error import assert_as_exact_as_the_reference

    from dew.nn.rope import rotate

    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(2, 96, 4, 128)), dtype)
    angles = rng.normal(size=(2, 96, pairs) if packed else (96, pairs)) * 3
    cos, sin = (jnp.asarray(f(angles), jnp.float32) for f in (np.cos, np.sin))
    cotangent = jnp.asarray(rng.normal(size=x.shape), dtype)

    def forward_and_gradient(rotary, x, cos, sin):
        out, pullback = jax.vjp(lambda x: rotary(x, cos, sin, scale=scale), x)
        return out, pullback(cotangent.astype(x.dtype))[0]

    with jax.enable_x64():
        wide = (np.asarray(part, np.float64) for part in (x, cos, sin))
        truth = forward_and_gradient(_rotate_half_reference, *wide)
    new = forward_and_gradient(rotate, x, cos, sin)
    old = forward_and_gradient(_rotate_half_reference, x, cos, sin)
    for label, mine, theirs, want in zip(("forward", "gradient"), new, old, truth, strict=True):
        assert mine.dtype == theirs.dtype == dtype
        assert_as_exact_as_the_reference(np.asarray(mine, np.float64), np.asarray(theirs, np.float64),
                                         np.asarray(want), label)


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
