"""GPT-NeoX's parallel residual, partial rotary and fused qkv checkpoint."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from dew.interop import Pretrained
from dew.sampling import Sampling, generate

DIRECTORY = Path(__file__).parent / 'fixtures' / 'hf' / 'gpt-neox-tiny'


def test_gpt_neox_logits_and_cached_greedy_generation_match_transformers():
    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    ids = np.load(DIRECTORY / 'input_ids.npy')
    expected = np.load(DIRECTORY / 'logits.npy')
    actual = np.asarray(loaded.model.apply(loaded.variables, jnp.asarray(ids)))
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=0)
    np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))
    generated = generate(loaded.model, loaded.variables, ids[:1, :4], 6, sampling=Sampling(temperature=0))
    np.testing.assert_array_equal(np.asarray(generated.tokens), np.load(DIRECTORY / 'generated.npy'))
    sequential = loaded.model.clone(parallel_residual=False)
    wrong = np.asarray(sequential.apply(loaded.variables, jnp.asarray(ids)))
    assert float(np.max(np.abs(wrong - expected))) > 1e-2



def test_legacy_neox_buffers_are_validated_in_their_stored_precision():
    from dew.interop.hf_decoders import translate_config, translate_weights
    from dew.interop.sources import load_shards
    from dew.nn.rope import inverse_frequencies

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    config = translate_config(loaded.config)
    tensors = dict(load_shards(DIRECTORY))
    prefix = 'gpt_neox.layers.0.attention.'
    tensors[prefix + 'bias'] = np.tril(np.ones((1, 1, 64, 64), bool))
    tensors[prefix + 'masked_bias'] = np.asarray(-np.inf, np.float16)
    tensors[prefix + "rotary_emb.inv_freq"] = inverse_frequencies(10000.0, 4, dtype=np.float32).astype(
        np.float16
    )
    actual = translate_weights(tensors, config, 'gpt_neox')
    ids = jnp.asarray(np.load(DIRECTORY / 'input_ids.npy'))
    np.testing.assert_array_equal(loaded.model.apply(actual, ids), loaded.model.apply(loaded.variables, ids))
    tensors[prefix + 'rotary_emb.inv_freq'][0] = .5
    with pytest.raises(ValueError):
        translate_weights(tensors, config, 'gpt_neox')


@pytest.mark.network
def test_released_pythia70m_logits_and_cached_greedy_generation(tmp_path):
    from tools.classic_gpt_reference import check_checkpoint

    check_checkpoint('EleutherAI/pythia-70m', tmp_path / 'measurement.json',
                     revision='a39f36b100fe8a5377810d56c3f4789b9c53ac42')
