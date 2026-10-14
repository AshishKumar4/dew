"""MiMo-V2-Flash's sliding layers with doubled key and value heads and sinks,
its narrower scaled values and its grouped sigmoid routing, against
transformers 5.16.1: tools/hf_reference.py writes tests/fixtures/hf/mimo-v2-flash-tiny."""

from pathlib import Path

import numpy as np
from reference_error import assert_as_exact_as_the_reference

from dew.interop import Pretrained

ROOT = Path(__file__).parent / 'fixtures' / 'hf'
TINY = ROOT / 'mimo-v2-flash-tiny'


def load():
    return Pretrained.load(TINY, dtype='float32', attention_impl='reference')


def test_padded_logits_and_cached_generation_match_transformers():
    from tools.classic_gpt_reference import greedy

    loaded = load()
    ids = np.load(TINY / 'padded_ids.npy')
    mask = np.load(TINY / 'attention_mask.npy')
    actual = np.asarray(loaded.model.apply(loaded.variables, ids, attention_mask=mask))
    expected, truth = (np.load(TINY / f'padded_logits{suffix}.npy') for suffix in ('', '_f64'))
    assert_as_exact_as_the_reference(actual[mask], expected[mask], truth[mask], 'MiMo-V2-Flash padded logits')
    np.testing.assert_array_equal(actual[mask].argmax(-1), expected[mask].argmax(-1))
    original = np.load(TINY / 'input_ids.npy')
    np.testing.assert_array_equal(greedy(loaded, original[:, :4], 6), np.load(TINY / 'generated.npy'))
