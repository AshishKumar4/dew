"""GPT-Neo's alternating windows and unscaled attention against transformers 5.16.1."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference

from dew.interop import Pretrained
from dew.interop.hf_decoders import translate_config
from dew.nn.backbones.layer_plan import LayerKind

ROOT = Path(__file__).parent / 'fixtures' / 'hf'
DIRECTORY = ROOT / 'gpt-neo-tiny'


def test_gpt_neo_padded_logits_and_cached_generation_match_transformers():
    from tools.classic_gpt_reference import greedy

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    ids = np.load(DIRECTORY / 'padded_ids.npy')
    mask = np.load(DIRECTORY / 'attention_mask.npy')
    actual = np.asarray(loaded.model.apply(loaded.variables, ids, attention_mask=mask))
    expected, truth = (np.load(DIRECTORY / f'padded_logits{suffix}.npy') for suffix in ('', '_f64'))
    assert_as_exact_as_the_reference(actual[mask], expected[mask], truth[mask], 'GPT-Neo padded logits')
    np.testing.assert_array_equal(actual[mask].argmax(-1), expected[mask].argmax(-1))
    original = np.load(DIRECTORY / 'input_ids.npy')
    np.testing.assert_array_equal(greedy(loaded, original[:, :4], 6), np.load(DIRECTORY / 'generated.npy'))


@pytest.mark.parametrize('term', ['window', 'scale'])
def test_gpt_neo_reference_catches_a_wrong_window_or_scale(term):
    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    wrong = loaded.model.clone(**({'kinds': {'sliding_attention': LayerKind(window=48)}}
                                  if term == 'window' else {'attention_scale': None}))
    logits = wrong.apply(loaded.variables, np.load(DIRECTORY / 'input_ids.npy'))
    with pytest.raises(AssertionError, match=r'allowed 2\.0'):
        assert_as_exact_as_the_reference(logits, np.load(DIRECTORY / 'logits.npy'),
                                         np.load(DIRECTORY / 'logits_f64.npy'), f'GPT-Neo wrong {term}')


def test_gpt_neo_export_reads_in_transformers_and_reloads_bitwise(tmp_path):
    import torch
    from transformers import AutoModelForCausalLM

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    loaded.save(tmp_path)
    reference = AutoModelForCausalLM.from_pretrained(tmp_path, attn_implementation='eager').float().eval()
    ids = np.load(DIRECTORY / 'input_ids.npy')
    with torch.no_grad():
        expected = reference(torch.from_numpy(ids).long(), use_cache=False).logits.numpy()
    actual = loaded.model.apply(loaded.variables, ids)
    assert_as_exact_as_the_reference(actual, expected, np.load(DIRECTORY / 'logits_f64.npy'),
                                     'GPT-Neo export')
    restored = Pretrained.load(tmp_path, dtype='float32', attention_impl='reference')
    np.testing.assert_array_equal(restored.model.apply(restored.variables, ids), actual)


def test_gpt_neo_biased_block_has_finite_gradients_and_updates():
    import optax

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    ids = jnp.asarray(np.load(DIRECTORY / 'input_ids.npy'))

    def loss(params):
        return optax.softmax_cross_entropy_with_integer_labels(
            loaded.model.apply({'params': params}, ids[:, :-1]), ids[:, 1:]).mean()

    value, gradient = jax.value_and_grad(loss)(loaded.variables['params'])
    changed = jax.tree.map(lambda parameter, grad: parameter - 1e-3 * grad,
                           loaded.variables['params'], gradient)
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(gradient))
    assert float(loss(changed)) < float(value)


def test_gpt_neo_released_config_retains_local_global_attention_and_scale():
    model = translate_config(json.loads((ROOT / 'gpt-neo-125m' / 'config.json').read_text())).value
    assert (model.emb_features, model.num_heads, model.num_layers) == (768, 12, 12)
    assert model.attention_scale == 1.0 and model.o_proj_bias and not model.attention_bias
    assert model.per_layer_types == ('full_attention', 'sliding_attention') * 6
    assert model.kind_of('sliding_attention').window == 256


@pytest.mark.network
def test_gpt_neo_released_config_is_pinned():
    from huggingface_hub import hf_hub_download

    source = json.loads((ROOT / 'gpt-neo-125m' / 'source.json').read_text())
    filename = hf_hub_download(source['repo'], 'config.json', revision=source['revision'])
    expected = json.loads((ROOT / 'gpt-neo-125m' / 'config.json').read_text())
    assert json.loads(Path(filename).read_text()) == expected
