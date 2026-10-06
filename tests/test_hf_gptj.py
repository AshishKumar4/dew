"""GPT-J's interleaved partial rotary stored in the shared rotate-half layout."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference

from dew.interop import Pretrained, PretrainedDecoder
from dew.interop.hf_decoders import translate_config, translate_weights
from dew.interop.sources import load_shards

ROOT = Path(__file__).parent / 'fixtures' / 'hf'
DIRECTORY = ROOT / 'gptj-tiny'


def test_gptj_padded_logits_and_cached_generation_match_transformers():
    from tools.classic_gpt_reference import greedy

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    ids = np.load(DIRECTORY / 'padded_ids.npy')
    mask = np.load(DIRECTORY / 'attention_mask.npy')
    actual = np.asarray(loaded.model.apply(loaded.variables, ids, attention_mask=mask))
    expected, truth = (np.load(DIRECTORY / f'padded_logits{suffix}.npy') for suffix in ('', '_f64'))
    assert_as_exact_as_the_reference(actual[mask], expected[mask], truth[mask], 'GPT-J padded logits')
    np.testing.assert_array_equal(actual[mask].argmax(-1), expected[mask].argmax(-1))
    original = np.load(DIRECTORY / 'input_ids.npy')
    np.testing.assert_array_equal(greedy(loaded, original[:, :4], 6), np.load(DIRECTORY / 'generated.npy'))


def test_gptj_export_preserves_interleaved_weights_and_reads_in_transformers(tmp_path):
    import torch
    from transformers import AutoModelForCausalLM

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    PretrainedDecoder.from_model(loaded.model, loaded.variables).save(tmp_path)
    original, exported = load_shards(DIRECTORY), load_shards(tmp_path)
    assert set(original) == set(exported)
    for name in original:
        np.testing.assert_array_equal(original[name], exported[name])
    reference = AutoModelForCausalLM.from_pretrained(tmp_path, attn_implementation='eager').float().eval()
    ids = np.load(DIRECTORY / 'input_ids.npy')
    with torch.no_grad():
        expected = reference(torch.from_numpy(ids).long(), use_cache=False).logits.numpy()
    actual = loaded.model.apply(loaded.variables, ids)
    assert_as_exact_as_the_reference(actual, expected, np.load(DIRECTORY / 'logits_f64.npy'), 'GPT-J export')
    restored = Pretrained.load(tmp_path, dtype='float32', attention_impl='reference')
    np.testing.assert_array_equal(restored.model.apply(restored.variables, ids), actual)


def test_gptj_reference_catches_interleaved_pairs_read_as_rotate_half():
    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    config = translate_config(loaded.config)
    tensors = load_shards(DIRECTORY)
    variables = jax.tree.map(jnp.asarray, loaded.variables)
    for name, tensor in tensors.items():
        if name.endswith(('.attn.q_proj.weight', '.attn.k_proj.weight')):
            parts = name.split('.')
            variables['params'][f'layers_{parts[2]}']['self_attn'][parts[4]]['kernel'] = jnp.asarray(tensor.T)
    wrong = loaded.model.apply(variables, np.load(DIRECTORY / 'input_ids.npy'))
    with pytest.raises(AssertionError, match=r'allowed 2\.0'):
        assert_as_exact_as_the_reference(wrong, np.load(DIRECTORY / 'logits.npy'),
                                         np.load(DIRECTORY / 'logits_f64.npy'), 'GPT-J wrong pairing')
    np.testing.assert_array_equal(
        translate_weights(tensors, config, 'gptj')['params']['head_bias'],
        loaded.variables['params']['head_bias'])


def test_gptj_affine_head_scores_through_the_public_objective():
    from dew.objectives.lm import LMObjective

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    ids = jnp.asarray(np.load(DIRECTORY / 'input_ids.npy'))
    objective = LMObjective(loaded.model, seq_len=ids.shape[1] - 1, variables=loaded.variables,
                            head_tile=(3, 17), ema_decay=None)
    scores = objective.token_scores(loaded.variables, ids)
    assert_as_exact_as_the_reference(scores.losses, np.load(DIRECTORY / 'losses.npy'),
                                     np.load(DIRECTORY / 'losses_f64.npy'), 'GPT-J affine-head likelihoods')


def test_gptj_affine_head_trains_through_the_public_objective():
    from dew.objectives.lm import LMObjective

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    ids = jnp.asarray(np.load(DIRECTORY / 'input_ids.npy'))
    objective = LMObjective(loaded.model, seq_len=ids.shape[1] - 1, variables=loaded.variables,
                            head_tile=(3, 17), ema_decay=None)

    def loss(params):
        return objective.token_scores({'params': params}, ids).losses.mean()

    value, gradient = jax.value_and_grad(loss)(loaded.variables['params'])
    assert np.linalg.norm(gradient['head_bias']) > 0
    changed = jax.tree.map(lambda parameter, grad: parameter - 1e-3 * grad,
                           loaded.variables['params'], gradient)
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(gradient))
    assert float(loss(changed)) < float(value)


def test_gptj_released_config_retains_parallel_residual_and_partial_rotary():
    model = translate_config(json.loads((ROOT / 'gpt-j-6b' / 'config.json').read_text())).value
    assert (model.emb_features, model.num_heads, model.num_layers) == (4096, 16, 28)
    assert model.parallel_residual and model.shared_parallel_norm and model.head_bias
    assert model.partial_rotary_factor == .25 and model.partial_rotary_type == 'default'


@pytest.mark.network
def test_gptj_released_config_is_pinned():
    from huggingface_hub import hf_hub_download

    source = json.loads((ROOT / 'gpt-j-6b' / 'source.json').read_text())
    filename = hf_hub_download(source['repo'], 'config.json', revision=source['revision'])
    expected = json.loads((ROOT / 'gpt-j-6b' / 'config.json').read_text())
    assert json.loads(Path(filename).read_text()) == expected
