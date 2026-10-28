"""Liquid's LFM3 text decoder (`lfm3`) against its own remote code at LiquidAI/LFM3-1.7B@40ebe0a:
tools/remote_code_reference.py writes tests/fixtures/hf/lfm3-tiny, its six layers gated short
convolutions between full and sliding attention layers."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
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
