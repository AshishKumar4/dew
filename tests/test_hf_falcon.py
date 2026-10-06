"""Falcon's fused MQA/MHA and one-norm parallel residual against transformers 5.16.1."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference

from dew.interop import Pretrained, PretrainedDecoder
from dew.interop.hf_decoders import translate_config
from dew.interop.sources import load_shards

ROOT = Path(__file__).parent / 'fixtures' / 'hf'


@pytest.mark.parametrize('name', ['falcon-tiny', 'falcon-mha-tiny'])
def test_falcon_padded_logits_and_cached_generation_match_transformers(name):
    from tools.classic_gpt_reference import greedy

    directory = ROOT / name
    loaded = Pretrained.load(directory, dtype='float32', attention_impl='reference')
    ids = np.load(directory / 'padded_ids.npy')
    mask = np.load(directory / 'attention_mask.npy')
    actual = np.asarray(loaded.model.apply(loaded.variables, ids, attention_mask=mask))
    expected, truth = (np.load(directory / f'padded_logits{suffix}.npy') for suffix in ('', '_f64'))
    assert_as_exact_as_the_reference(actual[mask], expected[mask], truth[mask], f'{name} padded logits')
    np.testing.assert_array_equal(actual[mask].argmax(-1), expected[mask].argmax(-1))
    original = np.load(directory / 'input_ids.npy')
    np.testing.assert_array_equal(greedy(loaded, original[:, :4], 6), np.load(directory / 'generated.npy'))


@pytest.mark.parametrize('name', ['falcon-tiny', 'falcon-mha-tiny'])
def test_falcon_export_preserves_fused_weights_and_reads_in_transformers(name, tmp_path):
    import torch
    from transformers import AutoModelForCausalLM

    directory = ROOT / name
    loaded = Pretrained.load(directory, dtype='float32', attention_impl='reference')
    PretrainedDecoder.from_model(loaded.model, loaded.variables).save(tmp_path)
    original, exported = load_shards(directory), load_shards(tmp_path)
    assert set(original) == set(exported)
    for tensor in original:
        np.testing.assert_array_equal(original[tensor], exported[tensor])
    reference = AutoModelForCausalLM.from_pretrained(tmp_path, attn_implementation='eager').float().eval()
    ids = np.load(directory / 'input_ids.npy')
    with torch.no_grad():
        expected = reference(torch.from_numpy(ids).long(), use_cache=False).logits.numpy()
    actual = loaded.model.apply(loaded.variables, ids)
    assert_as_exact_as_the_reference(actual, expected, np.load(directory / 'logits_f64.npy'), 'Falcon export')
    restored = Pretrained.load(tmp_path, dtype='float32', attention_impl='reference')
    np.testing.assert_array_equal(restored.model.apply(restored.variables, ids), actual)


def test_falcon_source_export_writes_updated_fused_projections(tmp_path):
    from dataclasses import replace

    loaded = Pretrained.load(ROOT / 'falcon-tiny', dtype='float32', attention_impl='reference')
    variables = jax.tree.map(jnp.asarray, loaded.variables)
    variables['params']['layers_0']['self_attn']['q_proj']['kernel'] *= 2
    replace(loaded, variables=variables).save(tmp_path)
    restored = Pretrained.load(tmp_path, dtype='float32', attention_impl='reference')
    ids = np.load(ROOT / 'falcon-tiny' / 'input_ids.npy')
    expected = loaded.model.apply(variables, ids)
    np.testing.assert_array_equal(restored.model.apply(restored.variables, ids), expected)


def test_falcon_reference_catches_sequential_residuals():
    directory = ROOT / 'falcon-tiny'
    loaded = Pretrained.load(directory, dtype='float32', attention_impl='reference')
    variables = jax.tree.map(jnp.asarray, loaded.variables)
    for index in range(loaded.model.num_layers):
        layer = variables['params'][f'layers_{index}']
        layer['post_attention_layernorm'] = layer['input_layernorm']
    wrong = loaded.model.clone(parallel_residual=False, shared_parallel_norm=False).apply(
        variables, np.load(directory / 'input_ids.npy'))
    with pytest.raises(AssertionError, match=r'allowed 2\.0'):
        assert_as_exact_as_the_reference(wrong, np.load(directory / 'logits.npy'),
                                         np.load(directory / 'logits_f64.npy'), 'Falcon sequential residual')


def test_falcon_released_config_retains_multi_query_and_parallel_layernorm():
    model = translate_config(json.loads((ROOT / 'falcon-7b' / 'config.json').read_text())).value
    assert (model.emb_features, model.num_heads, model.num_layers) == (4544, 71, 32)
    assert model.kv_heads == 1 and model.parallel_residual and model.shared_parallel_norm
    assert model.norm_bias and model.mlp == 'gelu_exact' and not model.attention_bias


def test_falcon_block_has_finite_gradients_and_updates():
    import optax

    directory = ROOT / 'falcon-tiny'
    loaded = Pretrained.load(directory, dtype='float32', attention_impl='reference')
    ids = jnp.asarray(np.load(directory / 'input_ids.npy'))

    def loss(params):
        return optax.softmax_cross_entropy_with_integer_labels(
            loaded.model.apply({'params': params}, ids[:, :-1]), ids[:, 1:]).mean()

    value, gradient = jax.value_and_grad(loss)(loaded.variables['params'])
    changed = jax.tree.map(lambda parameter, grad: parameter - 1e-3 * grad,
                           loaded.variables['params'], gradient)
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(gradient))
    assert float(loss(changed)) < float(value)


@pytest.mark.network
def test_falcon_released_config_is_pinned():
    from huggingface_hub import hf_hub_download

    source = json.loads((ROOT / 'falcon-7b' / 'source.json').read_text())
    filename = hf_hub_download(source['repo'], 'config.json', revision=source['revision'])
    expected = json.loads((ROOT / 'falcon-7b' / 'config.json').read_text())
    assert json.loads(Path(filename).read_text()) == expected
