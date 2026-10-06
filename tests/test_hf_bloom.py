"""BLOOM against transformers 5.16.1 under the float64 rounding rule."""

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

DIRECTORY = Path(__file__).parent / 'fixtures' / 'hf' / 'bloom-tiny'
RELEASE = Path(__file__).parent / 'fixtures' / 'hf' / 'bloom-560m'


def test_bloom_padded_logits_and_cache_match_the_reference():
    from tools.classic_gpt_reference import greedy

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference', max_seq_len=32)
    ids = np.load(DIRECTORY / 'padded_ids.npy')
    mask = np.load(DIRECTORY / 'attention_mask.npy')
    expected = np.load(DIRECTORY / 'padded_logits.npy')
    truth = np.load(DIRECTORY / 'padded_logits_f64.npy')
    actual = np.asarray(loaded.model.apply(loaded.variables, jnp.asarray(ids), attention_mask=mask))
    assert_as_exact_as_the_reference(actual[mask], expected[mask], truth[mask], 'BLOOM padded logits')
    np.testing.assert_array_equal(actual[mask].argmax(-1), expected[mask].argmax(-1))
    _, cache = loaded.model.apply(loaded.variables, ids, attention_mask=mask,
                                  decode=True, mutable=['cache'])
    cached, cache = loaded.model.apply({**loaded.variables, **cache}, ids, attention_mask=mask,
                                       decode=True, mutable=['cache'])
    assert_as_exact_as_the_reference(np.asarray(cached)[mask], expected[mask], truth[mask],
                                     'BLOOM cached padded logits')
    token = jnp.asarray([[5], [7]], jnp.int32)
    decoded, _ = loaded.model.apply({**loaded.variables, **cache}, token, decode=True, mutable=['cache'])
    for row in range(2):
        joined = np.concatenate((ids[row, mask[row]], np.asarray(token[row])))[None]
        full = loaded.model.apply(loaded.variables, joined)
        np.testing.assert_array_equal(np.asarray(decoded[row, -1]).argmax(), np.asarray(full[0, -1]).argmax())
    original = np.load(DIRECTORY / 'input_ids.npy')
    np.testing.assert_array_equal(greedy(loaded, original[:, :4], 6), np.load(DIRECTORY / 'generated.npy'))


def test_bloom_export_is_same_weight_transformers_and_a_bitwise_reload(tmp_path):
    import torch
    from transformers import AutoModelForCausalLM

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    loaded.save(tmp_path)
    original, exported = load_shards(DIRECTORY), load_shards(tmp_path)
    assert set(original) == set(exported)
    for name in original:
        np.testing.assert_array_equal(original[name], exported[name])
    reference = AutoModelForCausalLM.from_pretrained(tmp_path, attn_implementation='eager').float().eval()
    ids = np.load(DIRECTORY / 'input_ids.npy')
    with torch.no_grad():
        expected = reference(torch.from_numpy(ids).long(), use_cache=False).logits.numpy()
    actual = np.asarray(loaded.model.apply(loaded.variables, ids))
    assert_as_exact_as_the_reference(actual, expected, np.load(DIRECTORY / 'logits_f64.npy'),
                                     'BLOOM exported logits')
    restored = Pretrained.load(tmp_path, dtype='float32', attention_impl='reference')
    np.testing.assert_array_equal(restored.model.apply(restored.variables, ids), actual)


def test_bloom_reference_catches_a_dropped_alibi(monkeypatch):
    import dew.nn.mixers.attention as attention

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    bias = attention.alibi_bias
    monkeypatch.setattr(attention, 'alibi_bias',
                        lambda *args, **kwargs: jnp.zeros_like(bias(*args, **kwargs)))
    wrong = loaded.model.apply(loaded.variables, np.load(DIRECTORY / 'input_ids.npy'))
    with pytest.raises(AssertionError, match=r'allowed 2\.0'):
        assert_as_exact_as_the_reference(wrong, np.load(DIRECTORY / 'logits.npy'),
                                         np.load(DIRECTORY / 'logits_f64.npy'), 'BLOOM dropped ALiBi')


def test_bloom_trains_through_its_embedding_norm_and_biased_block():
    import optax

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    ids = jnp.asarray(np.load(DIRECTORY / 'input_ids.npy'))

    def loss(params):
        logits = loaded.model.apply({'params': params}, ids[:, :-1])
        return optax.softmax_cross_entropy_with_integer_labels(logits, ids[:, 1:]).mean()

    value, gradient = jax.value_and_grad(loss)(loaded.variables['params'])
    changed = jax.tree.map(lambda parameter, grad: parameter - 1e-3 * grad,
                           loaded.variables['params'], gradient)
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(gradient))
    assert float(loss(changed)) < float(value)


@pytest.mark.parametrize('name', ['bloom-560m', 'bloomz-560m'])
def test_bloom_released_config_retains_the_block_fields(name):
    release = DIRECTORY.parent / name
    model = translate_config(json.loads((release / 'config.json').read_text())).value
    assert (model.emb_features, model.num_heads, model.num_layers) == (1024, 16, 24)
    assert model.position_embedding == 'alibi' and model.embedding_norm
    assert model.norm_type == 'layer' and model.norm_bias and model.mlp_bias
    assert model.mlp == 'gelu' and model.tie_embeddings


@pytest.mark.network
@pytest.mark.parametrize('name', ['bloom-560m', 'bloomz-560m'])
def test_bloom_released_config_is_pinned(name):
    from huggingface_hub import hf_hub_download

    release = DIRECTORY.parent / name
    source = json.loads((release / 'source.json').read_text())
    filename = hf_hub_download(source['repo'], 'config.json', revision=source['revision'])
    downloaded = json.loads(Path(filename).read_text())
    assert downloaded == json.loads((release / 'config.json').read_text())
    assert translate_config(downloaded) == translate_config(json.loads((release / 'config.json').read_text()))
