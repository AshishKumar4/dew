"""Liquid's LFM3 text decoder (`lfm3`) against its own remote code at LiquidAI/LFM3-1.7B@40ebe0a:
tools/remote_code_reference.py writes tests/fixtures/hf/lfm3-tiny, its six layers gated short
convolutions between full and sliding attention layers."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model_support import decode_steps
from reference_error import assert_as_exact_as_the_reference
from safetensors.numpy import load_file

from dew.interop import Pretrained

TINY = Path(__file__).parent / 'fixtures' / 'hf' / 'lfm3-tiny'


def fixture(name: str) -> np.ndarray:
    return np.load(TINY / f'{name}.npy')


def test_the_text_decoder_computes_the_remote_codes_logits():
    """Gated short convolutions of seven taps, attention with per-head q/k norms and a sigmoid gate a
    head, the full layers turning half of each head at 1e6 and the sliding ones all of it at 1e4 over
    a window of four, sandwich norms and a tied head: the release's logits, on par with its own
    against float64 (check 1)."""
    loaded = Pretrained.load(TINY, dtype='float32', attention_impl='reference')
    ours = np.asarray(loaded.model.apply(loaded.variables, fixture('input_ids')))
    assert_as_exact_as_the_reference(ours, fixture('logits'), fixture('logits_f64'), 'LFM3 logits')


def test_a_decode_through_the_cache_computes_the_remote_codes_logits():
    """A prefill of four tokens, then one token a step: each convolution's last six inputs and each
    attention layer's keys through the cache, the sliding window over the cached keys; every step's
    logits on par with the release's (check 1)."""
    loaded = Pretrained.load(TINY, dtype='float32', attention_impl='reference')
    ids = jnp.asarray(fixture('input_ids'), jnp.int32)
    made, step = decode_steps(loaded.model)
    logits, cache = step(loaded.variables, made(loaded.variables, ids.shape[0]), ids[:, :4])
    steps = [logits[:, -1]]
    for position in range(4, ids.shape[1]):
        logits, cache = step(loaded.variables, cache, ids[:, position:position + 1])
        steps.append(logits[:, -1])
    ours = np.asarray(jnp.stack(steps, axis=1))
    assert_as_exact_as_the_reference(ours, fixture('logits')[:, 3:], fixture('logits_f64')[:, 3:],
                                     'LFM3 decoded logits')


def test_a_saved_decoder_writes_the_releases_tensors_back(tmp_path):
    """A load and a save write every tensor the release stores under its own name, bit for bit."""
    loaded = Pretrained.load(TINY, dtype='float32', attention_impl='reference')
    loaded.save(tmp_path / 'saved')
    stored = load_file(TINY / 'model.safetensors')
    written = load_file(tmp_path / 'saved' / 'model.safetensors')
    assert sorted(written) == sorted(stored)
    for name, tensor in stored.items():
        np.testing.assert_array_equal(written[name], tensor, err_msg=name)


@pytest.mark.parametrize('sandwich', [True, False], ids=['sandwich', 'plain'])
def test_a_native_decoder_saves_as_lfm3_and_reloads_the_same_model(tmp_path, sandwich):
    """A model built in Dew (`PretrainedDecoder.from_model`) is an LFM3 by its convolution kind, ahead of
    the families that match its gated attention or its sliding layers, and saves a config and tensors
    that load back to it, its logits bit for bit, with sandwich norms or without and with its full and
    sliding layers at their own bases."""
    from dew.interop import PretrainedDecoder

    loaded = Pretrained.load(TINY, dtype='float32', attention_impl='reference')
    model, variables = loaded.model, loaded.variables
    if not sandwich:
        model = model.clone(sandwich_norms=False)
        variables = model.init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32))
    ids = jnp.asarray(fixture('input_ids'), jnp.int32)
    expected = np.asarray(model.apply(variables, ids))
    PretrainedDecoder.from_model(model, variables).save(tmp_path / 'saved')
    assert json.loads((tmp_path / 'saved' / 'config.json').read_text())['model_type'] == 'lfm3'
    restored = Pretrained.load(tmp_path / 'saved', dtype='float32', attention_impl='reference')
    np.testing.assert_array_equal(np.asarray(restored.model.apply(restored.variables, ids)), expected)


def test_a_config_that_omits_its_pattern_and_rotary_reads_lfm3s_own():
    """A config that states no layer_types, rope_parameters or norm_eps reads Lfm3Config's: full
    attention at every fourth layer from the third, sliding from the fifth, convolutions between; the
    full layers turning half of each head at 1e6 and the sliding ones all of theirs at 1e4; an epsilon
    of 1e-5. One naming another activation than silu is refused."""
    from dew.interop.hf_decoders import translate_config

    config = json.loads((TINY / 'config.json').read_text())
    for field in ('layer_types', 'rope_parameters', 'norm_eps'):
        config.pop(field)
    fields = translate_config({**config, 'num_hidden_layers': 9})
    assert fields['layer_types'] == ('conv', 'conv', 'full_attention', 'conv', 'sliding_attention', 'conv',
                                     'full_attention', 'conv', 'sliding_attention')
    assert fields['rope_theta'] == 1e6 and fields['norm_eps'] == 1e-5
    assert fields['kinds']['full_attention']['partial_rotary_factor'] == 0.5
    assert fields['kinds']['sliding_attention']['rope_theta'] == 1e4
    with pytest.raises(ValueError, match='hidden_act'):
        translate_config({**config, 'hidden_act': 'gelu'})
    # A stated rotary names a record for each attention type its layers hold, as Lfm3Config checks.
    for stated in ({}, {'full_attention': None, 'sliding_attention': {'rope_theta': 1e4}},
                   {'full_attention': {'rope_theta': 1e6}}):
        with pytest.raises(ValueError, match='rope_parameters'):
            translate_config({**config, 'num_hidden_layers': 9, 'rope_parameters': stated})
