"""BLOOM against transformers 5.16.1 under the float64 rounding rule."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from reference_error import assert_as_exact_as_the_reference

from dew.interop import Pretrained

DIRECTORY = Path(__file__).parent / 'fixtures' / 'hf' / 'bloom-tiny'
RELEASE = Path(__file__).parent / 'fixtures' / 'hf' / 'bloom-560m'


def test_bloom_padded_rows_prefill_and_decode_through_the_cache_as_the_reference_reads_them():
    """Beyond the shared flow (tests/test_hf_reference_decoders.py): the padded rows prefilled into
    the cache give the reference's logits, and one token decoded after them gives the argmax a full
    forward over each row's real tokens gives."""
    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference', max_seq_len=32)
    ids = np.load(DIRECTORY / 'padded_ids.npy')
    mask = np.load(DIRECTORY / 'attention_mask.npy')
    expected = np.load(DIRECTORY / 'padded_logits.npy')
    truth = np.load(DIRECTORY / 'padded_logits_f64.npy')
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
