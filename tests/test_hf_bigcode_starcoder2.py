"""GPTBigCode's fused multi-query and multi-head c_attn and Starcoder2's
windowed grouped-query block against transformers 5.16.1."""

import json
from pathlib import Path

import jax
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference

from dew.interop import Pretrained, PretrainedDecoder
from dew.interop.sources import load_shards

ROOT = Path(__file__).parent / 'fixtures' / 'hf'
FIXTURES = ['gpt-bigcode-tiny', 'gpt-bigcode-mha-tiny', 'starcoder2-tiny']


def load(name):
    return Pretrained.load(ROOT / name, dtype='float32', attention_impl='reference')


@pytest.mark.parametrize('name', FIXTURES)
def test_padded_logits_and_cached_generation_match_transformers(name):
    from tools.classic_gpt_reference import greedy

    directory = ROOT / name
    loaded = load(name)
    ids = np.load(directory / 'padded_ids.npy')
    mask = np.load(directory / 'attention_mask.npy')
    actual = np.asarray(loaded.model.apply(loaded.variables, ids, attention_mask=mask))
    expected, truth = (np.load(directory / f'padded_logits{suffix}.npy') for suffix in ('', '_f64'))
    assert_as_exact_as_the_reference(actual[mask], expected[mask], truth[mask], f'{name} padded logits')
    np.testing.assert_array_equal(actual[mask].argmax(-1), expected[mask].argmax(-1))
    original = np.load(directory / 'input_ids.npy')
    np.testing.assert_array_equal(greedy(loaded, original[:, :4], 6), np.load(directory / 'generated.npy'))


@pytest.mark.parametrize('name, exported_type', [('gpt-bigcode-tiny', 'gpt_bigcode'),
                                                 ('gpt-bigcode-mha-tiny', 'gpt2'),
                                                 ('starcoder2-tiny', 'starcoder2')])
def test_export_reads_in_transformers_and_reloads_bit_for_bit(name, exported_type, tmp_path):
    """A multi-query GPTBigCode and Starcoder2 write their own tensors back
    unchanged; a multi-head GPTBigCode computes GPT-2's model, so it saves
    as GPT-2. Each export reads in transformers to the same logits."""
    import torch
    from transformers import AutoModelForCausalLM

    directory = ROOT / name
    loaded = load(name)
    PretrainedDecoder.from_model(loaded.model, loaded.variables).save(tmp_path)
    assert json.loads((tmp_path / 'config.json').read_text())['model_type'] == exported_type
    if exported_type != 'gpt2':
        original, exported = load_shards(directory), load_shards(tmp_path)
        assert set(original) == set(exported)
        for tensor in original:
            np.testing.assert_array_equal(original[tensor], exported[tensor])
    reference = AutoModelForCausalLM.from_pretrained(tmp_path, attn_implementation='eager').float().eval()
    ids = np.load(directory / 'input_ids.npy')
    with torch.no_grad():
        expected = reference(torch.from_numpy(ids).long(), use_cache=False).logits.numpy()
    actual = loaded.model.apply(loaded.variables, ids)
    truth = np.load(directory / 'logits_f64.npy')
    assert_as_exact_as_the_reference(actual, expected, truth, f'{name} export')
    restored = Pretrained.load(tmp_path, dtype='float32', attention_impl='reference')
    np.testing.assert_array_equal(restored.model.apply(restored.variables, ids), actual)

    assert restored.model == loaded.model
    assert jax.tree.structure(restored.variables) == jax.tree.structure(loaded.variables)
    for original, restored_leaf in zip(jax.tree.leaves(loaded.variables),
                                        jax.tree.leaves(restored.variables), strict=True):
        np.testing.assert_array_equal(np.asarray(restored_leaf), np.asarray(original))
