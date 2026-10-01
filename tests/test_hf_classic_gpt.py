"""Classic decoder ports against transformers 5.16.1, with nonzero biases."""

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew import models
from dew.interop import load_pretrained
from dew.interop.hf_decoders import translate_config
from dew.nn.backbones.causal_transformer import CausalTransformer

FIXTURES = Path(__file__).parent / 'fixtures' / 'hf'


def test_gpt2_config_builds_the_reference_block():
    config = translate_config({
        "model_type": "gpt2", "vocab_size": 32, "n_embd": 16,
        "n_layer": 2, "n_head": 2, "n_positions": 32,
        "activation_function": "gelu_new", "layer_norm_epsilon": 1e-5,
        "task_specific_params": {"text-generation": {"do_sample": True, "max_length": 50}},
    })
    model = models.build("causal_transformer", **config)
    variables = model.init(jax.random.key(0), jnp.asarray([[2, 5, 8]], jnp.int32))
    params = variables["params"]
    assert "bias" in params["norm"]
    assert "gate_proj" not in params["layers_0"]["mlp"]
    assert params["embed_positions"]["embedding"].shape == (32, 16)
    altered = jax.tree.map(lambda leaf: leaf.copy(), variables)
    altered["params"]["embed_positions"]["embedding"] += jnp.arange(16)
    before = model.apply(variables, jnp.asarray([[2, 5, 8]], jnp.int32))
    after = model.apply(altered, jnp.asarray([[2, 5, 8]], jnp.int32))
    assert np.max(np.abs(before - after)) > 1e-2


def test_gpt2_same_weight_logits_and_cached_generation():
    """The biased two-projection block matches fp32 eager transformers.

    Observed max error 2.38e-7 on CPU against the RTX 4080 reference at
    highest precision. The 1e-4 bound is the existing dense-family fp32 bound. Scattered
    LayerNorm scales, biases and learned positions make dropping any of
    these paths a numerical failure.
    """
    directory = FIXTURES / 'gpt2-tiny'
    loaded = load_pretrained(directory, dtype='float32', attention_impl='reference')
    model = loaded.model.clone(precision=jax.lax.Precision.HIGHEST)
    ids = np.load(directory / 'input_ids.npy')
    expected = np.load(directory / 'logits.npy')
    actual = np.asarray(model.apply(loaded.variables, jnp.asarray(ids)))
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=0)
    np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))
    from tools.classic_gpt_reference import greedy

    generated = greedy(dataclasses.replace(loaded, model=model), ids[:1, :4], 6)
    np.testing.assert_array_equal(generated, np.load(directory / 'generated.npy'))


@pytest.mark.parametrize('term', ['norm_bias', 'mlp_bias', 'learned_positions'])
def test_gpt2_reference_detects_a_dropped_classic_term(term):
    loaded = load_pretrained(FIXTURES / 'gpt2-tiny', dtype='float32', attention_impl='reference')
    variables = jax.tree.map(jnp.asarray, loaded.variables)
    params = variables['params']
    if term == 'norm_bias':
        params['norm']['bias'] = jnp.zeros_like(params['norm']['bias'])
        for index in range(loaded.model.num_layers):
            for name in ('input_layernorm', 'post_attention_layernorm'):
                params[f'layers_{index}'][name]['bias'] *= 0
    elif term == 'mlp_bias':
        for index in range(loaded.model.num_layers):
            for name in ('up_proj', 'down_proj'):
                params[f'layers_{index}']['mlp'][name]['bias'] *= 0
    else:
        params['embed_positions']['embedding'] *= 0
    ids = np.load(FIXTURES / 'gpt2-tiny' / 'input_ids.npy')
    expected = np.load(FIXTURES / 'gpt2-tiny' / 'logits.npy')
    wrong = np.asarray(loaded.model.apply(variables, jnp.asarray(ids)))
    assert float(np.max(np.abs(wrong - expected))) > 1e-2


def test_gpt2_export_is_read_by_transformers(tmp_path):
    from transformers import AutoModelForCausalLM
    import torch

    loaded = load_pretrained(FIXTURES / 'gpt2-tiny', dtype='float32', attention_impl='reference')
    loaded.save(tmp_path)
    reference = AutoModelForCausalLM.from_pretrained(tmp_path, attn_implementation='eager').float().eval()
    ids = np.load(FIXTURES / 'gpt2-tiny' / 'input_ids.npy')
    with torch.no_grad():
        expected = reference(torch.tensor(ids), use_cache=False).logits.numpy()
    actual = np.asarray(loaded.model.apply(loaded.variables, jnp.asarray(ids)))
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=0)
    np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))
    reloaded = load_pretrained(tmp_path, dtype='float32', attention_impl='reference')
    np.testing.assert_array_equal(
        np.asarray(reloaded.model.apply(reloaded.variables, jnp.asarray(ids))), actual)


def test_original_bare_gpt2_weights_and_causal_buffers_preserve_logits():
    from dew.interop.hf_decoders import translate_weights
    from dew.interop.sources import load_shards

    directory = FIXTURES / 'gpt2-tiny'
    loaded = load_pretrained(directory, dtype='float32', attention_impl='reference')
    tensors = {name.removeprefix('transformer.'): value for name, value in load_shards(directory).items()}
    for index in range(loaded.model.num_layers):
        tensors[f'h.{index}.attn.bias'] = np.tril(np.ones((1, 1, 64, 64), np.uint8))
        tensors[f'h.{index}.attn.masked_bias'] = np.asarray(-1e4, np.float32)
    config = translate_config(loaded.config)
    variables = translate_weights(tensors, config, 'gpt2')
    ids = jnp.asarray(np.load(directory / 'input_ids.npy'))
    np.testing.assert_array_equal(loaded.model.apply(variables, ids), loaded.model.apply(loaded.variables, ids))
    tensors['h.0.attn.bias'][0, 0, 0, 1] = 1
    with pytest.raises(ValueError, match='causal mask'):
        translate_weights(tensors, config, 'gpt2')


@pytest.mark.network
def test_released_gpt2_small_logits_and_generation_at_highest_fp32_precision(tmp_path):
    from tools.classic_gpt_reference import check_checkpoint

    check_checkpoint('openai-community/gpt2', tmp_path / 'parity.json')


@pytest.mark.parametrize('bias', [False, True])
def test_exact_nanogpt_architecture_trains_through_dew(bias):
    """An ungated exact-GELU GPT block goes through the public trainer."""
    import optax

    from dew import Trainer
    from dew.data import Dataset
    from dew.objectives.lm import LMObjective
    from dew.training.distributed import MeshSpec

    model = CausalTransformer(vocab_size=32, emb_features=16, num_layers=2, num_heads=2,
                              max_seq_len=8, position_embedding='learned', mlp='gelu_exact',
                              mlp_bias=bias, norm_type='layer', norm_bias=bias,
                              attention_bias=bias, qk_norm=False, attention_impl='reference',
                              dropout_rate=.2, embedding_dropout_rate=.2, attention_dropout_rate=.2)
    rows = np.asarray([[2, 4, 8, 3, 2, 4, 8, 3, 2], [7, 3, 9, 5, 7, 3, 9, 5, 7]], np.int32)
    variables = model.init(jax.random.key(0), jnp.asarray(rows[:, :-1]))
    objective = LMObjective(model, seq_len=8, ema_decay=None, pretrained=variables)
    data = Dataset(train=lambda partition: iter({'text': rows} for _ in range(3)),
                   val=None, records=6, batch=2)
    trainer = Trainer(objective, optax.adam(1e-3), key=jax.random.key(0), mesh=MeshSpec(data=1))
    before = np.asarray(model.apply(variables, jnp.asarray(rows[:, :-1])))
    state = trainer.fit(data, steps=3)
    after = np.asarray(model.apply(state.params, jnp.asarray(rows[:, :-1])))
    def loss(logits):
        return float(optax.softmax_cross_entropy_with_integer_labels(logits, jnp.asarray(rows[:, 1:])).mean())
    assert np.isfinite(loss(after)) and loss(after) < loss(before)


def test_learned_positions_honor_explicit_positions():
    model = CausalTransformer(vocab_size=16, emb_features=8, num_layers=1, num_heads=2,
                              max_seq_len=8, position_embedding='learned', qk_norm=False)
    ids = jnp.asarray([[1, 2, 3], [1, 2, 3]], jnp.int32)
    variables = model.init(jax.random.key(0), ids)
    positions = jnp.asarray([[0, 1, 2], [3, 4, 5]], jnp.int32)
    actual = model.apply(variables, ids, positions=positions)
    assert float(jnp.max(jnp.abs(actual[0] - actual[1]))) > 1e-2


@pytest.mark.parametrize('scan_layers', [False, True])
def test_learned_position_cache_advances_each_rows_unpadded_positions(scan_layers):
    model = CausalTransformer(vocab_size=16, emb_features=8, num_layers=2, num_heads=2,
                              max_seq_len=8, position_embedding='learned', qk_norm=False,
                              attention_impl='reference', scan_layers=scan_layers)
    ids = jnp.asarray([[1, 2, 3, 4], [0, 0, 5, 6]], jnp.int32)
    valid = jnp.asarray([[True, True, True, True], [False, False, True, True]])
    variables = model.init(jax.random.key(0), ids)
    _, cache = model.apply(variables, ids, decode=True, attention_mask=valid, mutable=['cache'])
    _, cache = model.apply({**variables, **cache}, ids, decode=True,
                           attention_mask=valid, mutable=['cache'])
    token = jnp.asarray([[7], [8]], jnp.int32)
    cached, _ = model.apply({**variables, **cache}, token, decode=True, mutable=['cache'])
    for index, prompt in enumerate((ids[0], ids[1, 2:])):
        joined = jnp.concatenate((prompt, token[index]), axis=0)[None]
        full = model.apply(variables, joined)
        np.testing.assert_allclose(cached[index, -1], full[0, -1], atol=1e-5, rtol=0)
