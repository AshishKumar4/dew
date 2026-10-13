"""GPT-J's interleaved partial rotary stored in the shared rotate-half layout."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from reference_error import assert_as_exact_as_the_reference

from dew.interop import Pretrained, PretrainedDecoder
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

    original_ids = np.load(DIRECTORY / 'input_ids.npy')
    original_logits = np.asarray(loaded.model.apply(loaded.variables, jnp.asarray(original_ids, jnp.int32)))
    original_reference = np.load(DIRECTORY / 'logits.npy')
    assert_as_exact_as_the_reference(original_logits, original_reference,
                                     np.load(DIRECTORY / 'logits_f64.npy'), 'unpadded logits')
    np.testing.assert_array_equal(original_logits.argmax(-1), original_reference.argmax(-1))


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

    assert restored.model == loaded.model
    assert jax.tree.structure(restored.variables) == jax.tree.structure(loaded.variables)
    for original, restored_leaf in zip(jax.tree.leaves(loaded.variables),
                                        jax.tree.leaves(restored.variables), strict=True):
        np.testing.assert_array_equal(np.asarray(restored_leaf), np.asarray(original))


def test_gptj_affine_head_trains_through_the_public_objective():
    from dew.objectives.lm import LMObjective

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    ids = jnp.asarray(np.load(DIRECTORY / 'input_ids.npy'))
    objective = LMObjective(loaded.model, seq_len=ids.shape[1] - 1, variables=loaded.variables,
                            head_tile=(3, 17), ema_decay=None)

    def loss(params):
        scores = objective.token_scores({'params': params}, ids)
        return scores.losses.mean(), scores.losses

    (value, losses), gradient = jax.value_and_grad(loss, has_aux=True)(loaded.variables['params'])
    assert_as_exact_as_the_reference(losses, np.load(DIRECTORY / 'losses.npy'),
                                     np.load(DIRECTORY / 'losses_f64.npy'), 'GPT-J affine-head likelihoods')
    assert np.linalg.norm(gradient['head_bias']) > 0
    changed = jax.tree.map(lambda parameter, grad: parameter - 1e-3 * grad,
                           loaded.variables['params'], gradient)
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(gradient))
    assert float(loss(changed)[0]) < float(value)
