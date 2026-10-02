"""OPT's learned-position offset and pre-norm block against transformers."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from dew.interop import Pretrained
from dew.interop.hf_decoders import translate_config

DIRECTORY = Path(__file__).parent / 'fixtures' / 'hf' / 'opt-tiny'


def test_opt_config_holds_reserved_position_rows():
    config = translate_config({'model_type': 'opt', 'vocab_size': 32, 'hidden_size': 16,
                               'ffn_dim': 32, 'num_hidden_layers': 2, 'num_attention_heads': 2,
                               'max_position_embeddings': 32})
    assert config['position_embedding_size'] == 34
    assert config['position_embedding_offset'] == 2


def test_opt_same_weight_logits_and_cached_greedy_generation():
    from tools.classic_gpt_reference import greedy

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    ids = np.load(DIRECTORY / 'input_ids.npy')
    actual = np.asarray(loaded.model.apply(loaded.variables, jnp.asarray(ids)))
    expected = np.load(DIRECTORY / 'logits.npy')
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=0)
    np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))
    np.testing.assert_array_equal(greedy(loaded, ids[:1, :4], 6), np.load(DIRECTORY / 'generated.npy'))
    wrong = loaded.model.clone(position_embedding_offset=0)
    assert float(np.max(np.abs(np.asarray(wrong.apply(loaded.variables, ids)) - expected))) > 1e-2


def test_opt_export_uses_the_reference_checkpoint_names(tmp_path):
    from dew.interop.sources import load_shards

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    loaded.save(tmp_path)
    actual = load_shards(tmp_path)
    expected = load_shards(DIRECTORY)
    assert set(actual) == set(expected)
    for name in actual:
        np.testing.assert_array_equal(actual[name], expected[name])
    reloaded = Pretrained.load(tmp_path, dtype='float32', attention_impl='reference')
    ids = jnp.asarray(np.load(DIRECTORY / 'input_ids.npy'))
    np.testing.assert_array_equal(reloaded.model.apply(reloaded.variables, ids),
                                  loaded.model.apply(loaded.variables, ids))


@pytest.mark.parametrize('unsupported', [{'do_layer_norm_before': False},
                                        {'word_embed_proj_dim': 8},
                                        {'layer_norm_elementwise_affine': False}])
def test_opt_refuses_a_block_with_unrepresented_operations(unsupported):
    with pytest.raises(ValueError):
        translate_config({'model_type': 'opt', 'vocab_size': 32, 'hidden_size': 16,
                          'ffn_dim': 32, 'num_hidden_layers': 2, 'num_attention_heads': 2,
                          'max_position_embeddings': 32, **unsupported})


@pytest.mark.network
def test_released_opt125m_logits_and_cached_generation_after_safe_repacking(tmp_path):
    from tools.classic_gpt_reference import check_checkpoint

    check_checkpoint('facebook/opt-125m', tmp_path / 'parity.json',
                     revision='27dcfa74d334bc871f3234de431e71c6eeba5dd6',
                     safetensors_directory=str(tmp_path / 'safetensors'))
