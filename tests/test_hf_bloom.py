"""BLOOM's embedding norm, ALiBi attention and cached generation parity."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from dew.interop import load_pretrained
from dew.nn.mixers.attention import AttentionMixer

DIRECTORY = Path(__file__).parent / 'fixtures' / 'hf' / 'bloom-tiny'


def test_bloom_logits_and_cached_generation_match_transformers():
    from tools.classic_gpt_reference import greedy

    loaded = load_pretrained(DIRECTORY, dtype='float32', attention_impl='reference')
    ids = np.load(DIRECTORY / 'input_ids.npy')
    expected = np.load(DIRECTORY / 'logits.npy')
    actual = np.asarray(loaded.model.apply(loaded.variables, jnp.asarray(ids)))
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=0)
    np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))
    np.testing.assert_array_equal(greedy(loaded, ids[:1, :4], 6), np.load(DIRECTORY / 'generated.npy'))
    wrong = loaded.model.clone(mixer=AttentionMixer(nope=True))
    assert float(np.max(np.abs(np.asarray(wrong.apply(loaded.variables, ids)) - expected))) > 1e-3


def test_bloom_padded_cached_rows_use_compact_alibi_positions():
    loaded = load_pretrained(DIRECTORY, dtype='float32', attention_impl='reference')
    model = loaded.model
    ids = jnp.asarray([[2, 3, 4, 5], [0, 0, 6, 7]], jnp.int32)
    valid = ids != 0
    _, cache = model.apply(loaded.variables, ids, attention_mask=valid, decode=True, mutable=['cache'])
    _, cache = model.apply({**loaded.variables, **cache}, ids, attention_mask=valid, decode=True, mutable=['cache'])
    following = jnp.asarray([[8], [9]], jnp.int32)
    cached, _ = model.apply({**loaded.variables, **cache}, following, decode=True, mutable=['cache'])
    for row, prompt in enumerate((ids[0], ids[1, 2:])):
        full = model.apply(loaded.variables, jnp.concatenate((prompt, following[row]))[None])
        np.testing.assert_allclose(cached[row, -1], full[0, -1], atol=1e-5, rtol=0)


def test_bloom_export_preserves_reference_layout(tmp_path):
    from dew.interop.sources import load_shards

    loaded = load_pretrained(DIRECTORY, dtype='float32', attention_impl='reference')
    loaded.save(tmp_path)
    actual, expected = load_shards(tmp_path), load_shards(DIRECTORY)
    assert set(actual) == set(expected)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])
