"""BLOOM against transformers 5.16.1 under the float64 rounding rule."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from reference_error import assert_as_exact_as_the_reference

from dew.interop import Pretrained

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

    original_ids = np.load(DIRECTORY / 'input_ids.npy')
    original_logits = np.asarray(loaded.model.apply(loaded.variables, jnp.asarray(original_ids, jnp.int32)))
    original_reference = np.load(DIRECTORY / 'logits.npy')
    assert_as_exact_as_the_reference(original_logits, original_reference,
                                     np.load(DIRECTORY / 'logits_f64.npy'), 'unpadded logits')
    np.testing.assert_array_equal(original_logits.argmax(-1), original_reference.argmax(-1))
