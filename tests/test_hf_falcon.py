"""Falcon's fused MQA/MHA and one-norm parallel residual against transformers 5.16.1."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference

from dew.interop import Pretrained

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

    original_ids = np.load(directory / 'input_ids.npy')
    original_logits = np.asarray(loaded.model.apply(loaded.variables, jnp.asarray(original_ids, jnp.int32)))
    original_reference = np.load(directory / 'logits.npy')
    assert_as_exact_as_the_reference(original_logits, original_reference,
                                     np.load(directory / 'logits_f64.npy'), 'unpadded logits')
    np.testing.assert_array_equal(original_logits.argmax(-1), original_reference.argmax(-1))


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
