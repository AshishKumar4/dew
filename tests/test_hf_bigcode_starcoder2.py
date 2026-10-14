"""GPTBigCode's fused multi-query and multi-head c_attn and Starcoder2's
windowed grouped-query block against transformers 5.16.1."""

from pathlib import Path

import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference

from dew.interop import Pretrained

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
