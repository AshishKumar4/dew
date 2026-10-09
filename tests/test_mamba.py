"""Mamba's selective scan and a `MambaForCausalLM` checkpoint against
transformers 5.16.1: tools/mamba_reference.py writes tests/fixtures/hf/mamba-tiny."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference

from dew.interop import Pretrained, PretrainedDecoder
from dew.interop.hf_decoders import translate_config
from dew.interop.sources import load_shards
from dew.nn.mixers.mamba import Mamba, MambaMixer, selective_scan

ROOT = Path(__file__).parent / 'fixtures' / 'hf'
TINY = ROOT / 'mamba-tiny'


def load():
    return Pretrained.load(TINY, dtype='float32', attention_impl='reference')


def operands(length=11, channels=16, state=4, seed=0):
    keys = jax.random.split(jax.random.key(seed), 6)
    return (jax.random.normal(keys[0], (2, length, channels)),
            jax.nn.softplus(jax.random.normal(keys[1], (2, length, channels))),
            -jnp.exp(jax.random.normal(keys[2], (channels, state))),
            jax.random.normal(keys[3], (2, length, state)), jax.random.normal(keys[4], (2, length, state)),
            jax.random.normal(keys[5], (channels,)))


def stepwise(x, dt, A, B, C, D, state):
    """`mamba_selective_state_update` (modeling_mamba.py:129-172), one token at a time."""
    outputs = []
    for t in range(x.shape[1]):
        state = jnp.exp(dt[:, t, :, None] * A) * state + (dt[:, t] * x[:, t])[..., None] * B[:, t, None, :]
        outputs.append(jnp.einsum('bdn,bn->bd', state, C[:, t]) + x[:, t] * D)
    return jnp.stack(outputs, 1), state


@pytest.mark.parametrize('chunk', [1, 4, 64])
def test_the_chunked_scan_is_the_recurrence_from_any_state(chunk):
    """Each chunk size, one that divides the sequence and one that pads it,
    computes the token-by-token recurrence and leaves its state."""
    x, dt, A, B, C, D = operands()
    held = jax.random.normal(jax.random.key(7), (2, 16, 4))
    out, final = selective_scan(x, dt, A, B, C, D, state=held, chunk_size=chunk)
    expected, last = stepwise(x, dt, A, B, C, D, held)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(final, last, rtol=1e-5, atol=1e-5)


def test_a_document_start_drops_the_state_entering_it():
    x, dt, A, B, C, D = operands()
    starts = jnp.zeros((2, 11), bool).at[:, 5].set(True)
    out, _ = selective_scan(x, dt, A, B, C, D, starts=starts, chunk_size=4)
    second, _ = selective_scan(x[:, 5:], dt[:, 5:], A, B[:, 5:], C[:, 5:], D, chunk_size=4)
    np.testing.assert_allclose(out[:, 5:], second, rtol=1e-5, atol=1e-5)


def test_padded_logits_and_cached_generation_match_transformers():
    from tools.classic_gpt_reference import greedy

    loaded = load()
    ids = np.load(TINY / 'padded_ids.npy')
    mask = np.load(TINY / 'attention_mask.npy')
    actual = np.asarray(loaded.model.apply(loaded.variables, ids, attention_mask=mask))
    expected, truth = (np.load(TINY / f'padded_logits{suffix}.npy') for suffix in ('', '_f64'))
    assert_as_exact_as_the_reference(actual[mask], expected[mask], truth[mask], 'mamba padded logits')
    np.testing.assert_array_equal(actual[mask].argmax(-1), expected[mask].argmax(-1))
    original = np.load(TINY / 'input_ids.npy')
    np.testing.assert_array_equal(greedy(loaded, original[:, :4], 6), np.load(TINY / 'generated.npy'))


def test_logits_match_transformers():
    loaded = load()
    actual = loaded.model.apply(loaded.variables, np.load(TINY / 'input_ids.npy'))
    assert_as_exact_as_the_reference(actual, np.load(TINY / 'logits.npy'), np.load(TINY / 'logits_f64.npy'),
                                     'mamba logits')


def test_the_reference_catches_a_shared_decay():
    """A is a decay per channel and state dimension; one decay per channel,
    Mamba-2's scalar, computes another model."""
    loaded = load()
    params = jax.tree.map(jnp.asarray, loaded.variables)['params']
    for index in range(loaded.model.num_layers):
        log = params[f'layers_{index}']['self_attn']['A_log']
        shared = jnp.broadcast_to(log.mean(-1, keepdims=True), log.shape)
        params[f'layers_{index}']['self_attn']['A_log'] = shared
    wrong = loaded.model.apply({'params': params}, np.load(TINY / 'input_ids.npy'))
    with pytest.raises(AssertionError, match=r'allowed 2\.0'):
        assert_as_exact_as_the_reference(wrong, np.load(TINY / 'logits.npy'),
                                         np.load(TINY / 'logits_f64.npy'), 'mamba with a shared decay')


def test_the_layer_decodes_as_it_prefills():
    mixer = Mamba(emb_features=8, intermediate_size=16, state_size=4, time_step_rank=2, chunk_size=4)
    x = jax.random.normal(jax.random.key(0), (2, 9, 8))
    variables = mixer.init(jax.random.key(1), x)
    whole = mixer.apply(variables, x)
    _, cache = mixer.apply(variables, x[:, :1], decode=True, mutable=['cache'])
    steps = []
    for t in range(9):
        out, cache = mixer.apply({**variables, **cache}, x[:, t:t + 1], decode=True, mutable=['cache'])
        steps.append(out)
    np.testing.assert_allclose(jnp.concatenate(steps, 1), whole, rtol=1e-5, atol=1e-5)
    assert cache['cache']['ssm_state'].shape == (2, 16, 4)


def test_export_reads_in_transformers_and_reloads_bit_for_bit(tmp_path):
    import torch
    from transformers import AutoModelForCausalLM

    loaded = load()
    PretrainedDecoder.from_model(loaded.model, loaded.variables).save(tmp_path)
    assert json.loads((tmp_path / 'config.json').read_text())['model_type'] == 'mamba'
    original, exported = load_shards(TINY), load_shards(tmp_path)
    assert set(original) == set(exported)
    for tensor in original:
        np.testing.assert_array_equal(original[tensor], exported[tensor])
    reference = AutoModelForCausalLM.from_pretrained(tmp_path).float().eval()
    ids = np.load(TINY / 'input_ids.npy')
    with torch.no_grad():
        expected = reference(torch.from_numpy(ids).long(), use_cache=False).logits.numpy()
    actual = loaded.model.apply(loaded.variables, ids)
    assert_as_exact_as_the_reference(actual, expected, np.load(TINY / 'logits_f64.npy'), 'mamba export')
    restored = Pretrained.load(tmp_path, dtype='float32', attention_impl='reference')
    np.testing.assert_array_equal(restored.model.apply(restored.variables, ids), actual)


def test_the_released_config_translates_its_computation():
    model = translate_config(json.loads((ROOT / 'mamba-130m-hf' / 'config.json').read_text())).value
    assert model.mixer == MambaMixer(intermediate_size=1536, state_size=16, time_step_rank=48, conv_kernel=4)
    assert (model.emb_features, model.num_layers, model.vocab_size) == (768, 24, 50280)
    assert model.mlp_features == 0 and model.tie_embeddings


@pytest.mark.network
def test_the_released_config_is_pinned():
    from huggingface_hub import hf_hub_download

    source = json.loads((ROOT / 'mamba-130m-hf' / 'source.json').read_text())
    filename = hf_hub_download(source['repo'], 'config.json', revision=source['revision'])
    pinned = json.loads((ROOT / 'mamba-130m-hf' / 'config.json').read_text())
    assert json.loads(Path(filename).read_text()) == pinned
