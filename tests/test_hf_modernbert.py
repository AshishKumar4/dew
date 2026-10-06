"""ModernBERT against transformers 5.16.1 under the float64 rounding rule."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference

from dew.interop import Pretrained
from dew.interop.hf_decoders import translate_config
from dew.interop.sources import load_shards
from dew.nn.backbones.causal_transformer import CausalTransformer

FIXTURES = Path(__file__).parent / 'fixtures' / 'hf'
DIRECTORY = FIXTURES / 'modernbert-tiny'
RELEASES = ('modernbert-base', 'laya-encoder')


def _fixture(name: str) -> np.ndarray:
    return np.load(DIRECTORY / f'{name}.npy')


def test_modernbert_masked_lm_logits_match_the_reference():
    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    logits = np.asarray(loaded.model.apply(loaded.variables, _fixture('input_ids')))
    assert_as_exact_as_the_reference(logits, _fixture('logits'), _fixture('logits_f64'), 'ModernBERT logits')
    np.testing.assert_array_equal(logits.argmax(-1), _fixture('logits').argmax(-1))


@pytest.mark.parametrize('implementation', ['reference', 'xla'])
def test_modernbert_states_match_the_encoder_through_every_window_path(implementation):
    """The symmetric window reaches the kernel's flag, the validity mask and
    packed documents alike: plain rows, right-padded rows under an
    attention mask, and two documents packed in one row, each against
    ModernBertModel."""
    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl=implementation)

    def states(ids, **kwargs):
        return np.asarray(loaded.model.apply(loaded.variables, jnp.asarray(ids),
                                             method=CausalTransformer.hidden_states, **kwargs))

    assert_as_exact_as_the_reference(states(_fixture('input_ids')), _fixture('hidden'),
                                     _fixture('hidden_f64'), 'ModernBERT states')
    mask = _fixture('attention_mask')
    padded = states(_fixture('padded_ids'), attention_mask=mask)
    assert_as_exact_as_the_reference(padded[mask], _fixture('padded_hidden')[mask],
                                     _fixture('padded_hidden_f64')[mask], 'ModernBERT padded states')
    documents = _fixture('segment_ids')
    positions = np.where(documents == 1, np.arange(documents.shape[1]), np.arange(documents.shape[1]) - 5)
    packed = states(_fixture('input_ids'), segment_ids=documents, positions=positions)
    assert_as_exact_as_the_reference(packed, _fixture('packed_hidden'), _fixture('packed_hidden_f64'),
                                     'ModernBERT packed states')


@pytest.mark.skipif(jax.default_backend() != 'gpu', reason='the bf16 kernels it selects are the GPU ones')
def test_modernbert_states_in_bf16_on_the_gpu_kernels(without_deterministic_ops):
    """'auto' runs the global layers on cuDNN and keeps the local layers'
    two-sided window off it, which cuDNN refuses; the bf16 states are held to
    the reference's own bf16 run."""
    loaded = Pretrained.load(DIRECTORY, dtype='bfloat16', attention_impl='auto')
    states = loaded.model.apply(loaded.variables, _fixture('input_ids'),
                                method=CausalTransformer.hidden_states)
    assert_as_exact_as_the_reference(np.asarray(states, np.float32), _fixture('hidden_bf16'),
                                     _fixture('hidden_f64'), 'ModernBERT bf16 states')


def test_the_parity_rule_catches_a_window_on_one_side(monkeypatch):
    """The causal-shaped window a bidirectional layer took before the fix."""
    import dew.nn.attention as attention

    monkeypatch.setattr(attention, 'window_sides', lambda causal, window: (window - 1, 0))
    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='xla')
    states = loaded.model.apply(loaded.variables, _fixture('input_ids'),
                                method=CausalTransformer.hidden_states)
    with pytest.raises(AssertionError, match=r'allowed 2\.0'):
        assert_as_exact_as_the_reference(states, _fixture('hidden'), _fixture('hidden_f64'),
                                         'ModernBERT one-sided window')


def test_modernbert_export_is_same_weight_transformers_and_a_bitwise_reload(tmp_path):
    import torch
    from transformers import AutoModelForMaskedLM

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    loaded.save(tmp_path)
    original, exported = load_shards(DIRECTORY), load_shards(tmp_path)
    assert set(original) == set(exported)
    for name in original:
        np.testing.assert_array_equal(original[name], exported[name])
    reference = AutoModelForMaskedLM.from_pretrained(tmp_path, attn_implementation='eager').float().eval()
    ids = _fixture('input_ids')
    with torch.no_grad():
        expected = reference(torch.from_numpy(ids).long()).logits.numpy()
    actual = np.asarray(loaded.model.apply(loaded.variables, ids))
    assert_as_exact_as_the_reference(actual, expected, _fixture('logits_f64'), 'ModernBERT exported logits')
    restored = Pretrained.load(tmp_path, dtype='float32', attention_impl='reference')
    np.testing.assert_array_equal(restored.model.apply(restored.variables, ids), actual)


def test_modernbert_export_keeps_the_original_keys_older_readers_take():
    """llama.cpp's converter (b11445) reads the norm epsilon from `layer_norm_eps`
    alone and the window pattern from `global_attn_every_n_layers`, which
    transformers 5 no longer writes."""
    from dew.interop.pretrained import PretrainedDecoder

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    config = PretrainedDecoder.from_model(loaded.model, loaded.variables).config
    assert config['layer_norm_eps'] == config['norm_eps'] == 3e-05
    assert config['global_attn_every_n_layers'] == 3


def test_modernbert_trains_through_its_unnormed_first_block_and_head():
    import optax

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    ids = jnp.asarray(_fixture('input_ids'))

    def loss(params):
        logits = loaded.model.apply({'params': params}, ids)
        return optax.softmax_cross_entropy_with_integer_labels(logits, ids).mean()

    value, gradient = jax.value_and_grad(loss)(loaded.variables['params'])
    assert 'input_layernorm' not in loaded.variables['params']['layers_0']
    assert all(np.abs(leaf).sum() > 0 for leaf in jax.tree.leaves(gradient))
    changed = jax.tree.map(lambda parameter, grad: parameter - 1e-3 * grad,
                           loaded.variables['params'], gradient)
    assert float(loss(changed)) < float(value)


def test_a_prediction_head_has_no_bare_head_matrix():
    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    assert loaded.model.apply(loaded.variables, method='output_table') is None


def test_the_released_configs_read_as_modernbert():
    base = translate_config(json.loads((FIXTURES / 'modernbert-base' / 'config.json').read_text())).value
    assert (base.emb_features, base.num_heads, base.num_layers, base.hidden_features) == (768, 12, 22, 1152)
    assert base.per_layer_types[:4] == ('full_attention', 'sliding_attention', 'sliding_attention',
                                        'full_attention')
    assert base.kind_of('sliding_attention').window == 65
    assert (base.rope_theta, base.kind_of('sliding_attention').rope_theta) == (160000.0, 10000.0)
    assert not base.causal and base.embedding_norm and not base.first_attention_norm
    assert (base.mlp, base.head_transform, base.head_bias) == ('geglu_exact', 'gelu_exact', True)
    # Laya's encoder states the transformers 5 spelling of the same model.
    laya = translate_config(json.loads((FIXTURES / 'laya-encoder' / 'config.json').read_text())).value
    assert (laya.emb_features, laya.num_layers, laya.hidden_features) == (1024, 28, 2624)
    assert laya.kind_of('sliding_attention').window == 65
    assert (laya.rope_theta, laya.kind_of('sliding_attention').rope_theta) == (160000.0, 10000.0)
    assert laya.per_layer_types == ('full_attention', 'sliding_attention', 'sliding_attention') * 9 + (
        'full_attention',)


@pytest.mark.network
@pytest.mark.parametrize('name', RELEASES)
def test_the_released_modernbert_configs_are_pinned(name):
    from huggingface_hub import hf_hub_download

    directory = FIXTURES / name
    source = json.loads((directory / 'source.json').read_text())
    filename = hf_hub_download(source['repo'], source.get('filename', 'config.json'),
                               revision=source['revision'])
    downloaded = json.loads(Path(filename).read_text())
    assert downloaded == json.loads((directory / 'config.json').read_text())
