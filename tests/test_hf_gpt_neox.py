"""GPT-NeoX's parallel residual, partial rotary and fused qkv checkpoint."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from dew.interop import Pretrained

DIRECTORY = Path(__file__).parent / 'fixtures' / 'hf' / 'gpt-neox-tiny'


def test_gpt_neox_logits_and_cached_greedy_generation_match_transformers():
    from tools.classic_gpt_reference import greedy

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    ids = np.load(DIRECTORY / 'input_ids.npy')
    expected = np.load(DIRECTORY / 'logits.npy')
    actual = np.asarray(loaded.model.apply(loaded.variables, jnp.asarray(ids)))
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=0)
    np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))
    np.testing.assert_array_equal(greedy(loaded, ids[:1, :4], 6), np.load(DIRECTORY / 'generated.npy'))
    sequential = loaded.model.clone(parallel_residual=False)
    wrong = np.asarray(sequential.apply(loaded.variables, jnp.asarray(ids)))
    assert float(np.max(np.abs(wrong - expected))) > 1e-2


def test_gpt_neox_fused_qkv_is_grouped_by_head():
    from dew.interop.hf_decoders import translate_config, translate_weights
    from dew.interop.sources import load_shards

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    tensors = dict(load_shards(DIRECTORY))
    for index in range(loaded.model.num_layers):
        for suffix in ('weight', 'bias'):
            name = f'gpt_neox.layers.{index}.attention.query_key_value.{suffix}'
            tensor = tensors[name]
            tensors[name] = tensor.reshape(4, 3, 8, *tensor.shape[1:]).transpose(
                (1, 0, 2, *range(3, tensor.ndim + 2))).reshape(tensor.shape)
    variables = translate_weights(tensors, translate_config(loaded.config), 'gpt_neox')
    ids = np.load(DIRECTORY / 'input_ids.npy')
    wrong = np.asarray(loaded.model.apply(variables, jnp.asarray(ids)))
    expected = np.load(DIRECTORY / 'logits.npy')
    assert float(np.max(np.abs(wrong - expected))) > 1e-2


def test_gpt_neox_export_preserves_fused_head_layout(tmp_path):
    from dew.interop.sources import load_shards

    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    loaded.save(tmp_path)
    exported, original = load_shards(tmp_path), load_shards(DIRECTORY)
    assert set(exported) == set(original)
    for name in original:
        np.testing.assert_array_equal(exported[name], original[name])
    reloaded = Pretrained.load(tmp_path, dtype='float32', attention_impl='reference')
    ids = jnp.asarray(np.load(DIRECTORY / 'input_ids.npy'))
    np.testing.assert_array_equal(reloaded.model.apply(reloaded.variables, ids),
                                  loaded.model.apply(loaded.variables, ids))


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
    with pytest.raises(ValueError, match='rotary frequencies'):
        translate_weights(tensors, config, 'gpt_neox')


def test_plain_exact_gelu_rounds_the_negative_tail_like_torch():
    from flax import linen as nn

    from dew.nn.backbones.decoder_block import GatedMLP

    mlp = GatedMLP(1, 1, activation='gelu_exact')
    variables = {'params': {'up_proj': {'kernel': jnp.ones((1, 1), jnp.float32)},
                            'down_proj': {'kernel': jnp.ones((1, 1), jnp.float32)}}}
    negative = jnp.asarray([[[-6.]]], jnp.float32)
    # In Torch's fp32 1 + erf formula, erf(-6/sqrt(2)) rounds to -1.
    # Flax's erfc form retains a nonzero tail, amplified by trained layers.
    np.testing.assert_array_equal(mlp.apply(variables, negative), jnp.zeros_like(negative))
    assert float(jnp.abs(nn.gelu(negative, approximate=False)).max()) > 0


@pytest.mark.network
def test_released_pythia70m_logits_and_cached_greedy_generation(tmp_path):
    from tools.classic_gpt_reference import check_checkpoint

    check_checkpoint('EleutherAI/pythia-70m', tmp_path / 'measurement.json',
                     revision='a39f36b100fe8a5377810d56c3f4789b9c53ac42')
