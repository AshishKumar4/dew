"""GPT-NeoX's parallel residual, partial rotary and fused qkv checkpoint."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from interop_support import drawn_ids, float64_logits
from reference_error import assert_as_exact, declared_seeds, indexed_recording, recording

from dew.interop import Pretrained
from dew.sampling import Sampling, generate

DIRECTORY = Path(__file__).parent / 'fixtures' / 'hf' / 'gpt-neox-tiny'


def test_gpt_neox_logits_and_cached_greedy_generation_match_transformers():
    """Dew's logits as exact as transformers' fp32 eager run, both from float64, over the recorded
    seeds' token grids (tools/gpt_neox_reference.py recording): the truth Dew's model run in float64,
    held at every seed to the recorded transformers truth's sketch; its cached greedy generation the
    reference's token for token; and the sequential residual another computation."""
    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    stored = recording(DIRECTORY / 'recorded_logits.npz')
    seeds = declared_seeds(DIRECTORY / 'recorded_logits.npz')['logits']
    vocab = json.loads((DIRECTORY / 'config.json').read_text())['vocab_size']
    draw, reference, truth = indexed_recording(stored, 'logits', seeds, truth_fn=float64_logits(loaded),
                                               inputs_of=lambda seed: drawn_ids(vocab, seed))
    forward = jax.jit(lambda inputs: {'logits': loaded.model.apply(loaded.variables, inputs['input_ids'])})
    assert_as_exact(lambda index, inputs: forward(inputs), reference, draw, 'gpt-neox logits', seeds=seeds,
                    truth_fn=truth)
    ids = np.load(DIRECTORY / 'input_ids.npy')
    actual = np.asarray(loaded.model.apply(loaded.variables, jnp.asarray(ids)))
    np.testing.assert_array_equal(actual.argmax(-1), np.load(DIRECTORY / 'logits.npy').argmax(-1))
    generated = generate(loaded.model, loaded.variables, ids[:1, :4], 6, key=0,
                         sampling=Sampling(temperature=0))
    np.testing.assert_array_equal(np.asarray(generated.tokens), np.load(DIRECTORY / 'generated.npy'))
    sequential = loaded.model.clone(parallel_residual=False)
    assert not np.array_equal(np.asarray(sequential.apply(loaded.variables, jnp.asarray(ids))), actual)



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
