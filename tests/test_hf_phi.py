"""Phi's parallel biased decoder and affine training head against transformers 5.16.1."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference

from dew.interop import Pretrained, PretrainedDecoder
from dew.interop.hf_decoders import translate_config

ROOT = Path(__file__).parent / 'fixtures' / 'hf'
DIRECTORY = ROOT / 'phi-tiny'


def test_phi_padded_logits_and_cached_generation_match_transformers():
    from tools.classic_gpt_reference import greedy

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    ids = np.load(DIRECTORY / 'padded_ids.npy')
    mask = np.load(DIRECTORY / 'attention_mask.npy')
    actual = np.asarray(loaded.model.apply(loaded.variables, ids, attention_mask=mask))
    expected, truth = (np.load(DIRECTORY / f'padded_logits{suffix}.npy') for suffix in ('', '_f64'))
    assert_as_exact_as_the_reference(actual[mask], expected[mask], truth[mask], 'Phi padded logits')
    np.testing.assert_array_equal(actual[mask].argmax(-1), expected[mask].argmax(-1))
    original = np.load(DIRECTORY / 'input_ids.npy')
    np.testing.assert_array_equal(greedy(loaded, original[:, :4], 6), np.load(DIRECTORY / 'generated.npy'))


def test_phi_affine_head_scores_through_the_public_objective():
    from dew.objectives.lm import LMObjective

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    ids = jnp.asarray(np.load(DIRECTORY / 'input_ids.npy'))
    objective = LMObjective(loaded.model, seq_len=ids.shape[1] - 1, variables=loaded.variables,
                            head_chunks=3, head_tile=(3, 17), ema_decay=None)
    logits = loaded.model.apply(loaded.variables, ids[:, :-1])
    scores = objective.token_scores(loaded.variables, ids)
    assert_as_exact_as_the_reference(scores.losses, np.load(DIRECTORY / 'losses.npy'),
                                     np.load(DIRECTORY / 'losses_f64.npy'), 'Phi affine-head likelihoods')
    np.testing.assert_array_equal(scores.correct, (logits.argmax(-1) == ids[:, 1:]).astype(jnp.float32))


def test_phi_affine_head_trains_through_the_public_objective():
    from dew.objectives.lm import LMObjective

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    ids = jnp.asarray(np.load(DIRECTORY / 'input_ids.npy'))
    objective = LMObjective(loaded.model, seq_len=ids.shape[1] - 1, variables=loaded.variables,
                            head_chunks=3, head_tile=(3, 17), ema_decay=None)

    def loss(params):
        return objective.token_scores({'params': params}, ids).losses.mean()

    value, gradient = jax.value_and_grad(loss)(loaded.variables['params'])
    assert np.linalg.norm(gradient['head_bias']) > 0
    changed = jax.tree.map(lambda parameter, grad: parameter - 1e-3 * grad,
                           loaded.variables['params'], gradient)
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(gradient))
    assert float(loss(changed)) < float(value)


def test_phi_export_reads_in_transformers_and_reloads_bitwise(tmp_path):
    import torch
    from transformers import AutoModelForCausalLM

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    PretrainedDecoder.from_model(loaded.model, loaded.variables).save(tmp_path)
    reference = AutoModelForCausalLM.from_pretrained(tmp_path, attn_implementation='eager').float().eval()
    ids = np.load(DIRECTORY / 'input_ids.npy')
    with torch.no_grad():
        expected = reference(torch.from_numpy(ids).long(), use_cache=False).logits.numpy()
    actual = loaded.model.apply(loaded.variables, ids)
    assert_as_exact_as_the_reference(actual, expected, np.load(DIRECTORY / 'logits_f64.npy'), 'Phi export')
    restored = Pretrained.load(tmp_path, dtype='float32', attention_impl='reference')
    np.testing.assert_array_equal(restored.model.apply(restored.variables, ids), actual)


@pytest.mark.parametrize('term', ['head_bias', 'parallel', 'partial_rotary'])
def test_phi_reference_catches_a_dropped_term(term):
    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    variables = jax.tree.map(jnp.asarray, loaded.variables)
    model = loaded.model
    if term == 'head_bias':
        variables['params']['head_bias'] *= 0
    elif term == 'partial_rotary':
        model = model.clone(partial_rotary_factor=1.)
    else:
        # Sharing one norm with a sequential residual changes the MLP's
        # input; duplicating its weights builds that plausible wrong block.
        model = model.clone(parallel_residual=False, shared_parallel_norm=False)
        for index in range(model.num_layers):
            layer = variables['params'][f'layers_{index}']
            layer['post_attention_layernorm'] = layer['input_layernorm']
    wrong = model.apply(variables, np.load(DIRECTORY / 'input_ids.npy'))
    with pytest.raises(AssertionError, match=r'allowed 2\.0'):
        assert_as_exact_as_the_reference(wrong, np.load(DIRECTORY / 'logits.npy'),
                                         np.load(DIRECTORY / 'logits_f64.npy'), f'Phi dropped {term}')


def test_phi_released_config_retains_partial_rotary_and_parallel_biases():
    model = translate_config(json.loads((ROOT / 'phi-2' / 'config.json').read_text())).value
    assert (model.emb_features, model.num_heads, model.num_layers) == (2560, 32, 32)
    assert model.partial_rotary_factor == .4 and model.partial_rotary_type == 'default'
    assert model.parallel_residual and model.shared_parallel_norm and model.head_bias
    assert model.norm_bias and model.mlp_bias and model.attention_bias


def test_phi_vocabulary_bias_storage_stays_fp32_with_bf16_parameters():
    from dew.interop.sources import load_shards

    loaded = Pretrained.load(DIRECTORY, dtype='bfloat16', param_dtype='bfloat16',
                             attention_impl='reference')
    assert loaded.variables['params']['head_bias'].dtype == jnp.float32
    np.testing.assert_array_equal(loaded.variables['params']['head_bias'],
                                  load_shards(DIRECTORY)['lm_head.bias'])


@pytest.mark.network
def test_phi_released_config_is_pinned():
    from huggingface_hub import hf_hub_download

    source = json.loads((ROOT / 'phi-2' / 'source.json').read_text())
    filename = hf_hub_download(source['repo'], 'config.json', revision=source['revision'])
    assert json.loads(Path(filename).read_text()) == json.loads((ROOT / 'phi-2' / 'config.json').read_text())
