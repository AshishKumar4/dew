"""Mamba's selective scan and a `MambaForCausalLM` checkpoint against
transformers 5.16.1: tools/mamba_reference.py writes tests/fixtures/hf/mamba-tiny."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model_support import as_stepped, decode_through, float64_twin
from reference_error import assert_as_exact

from dew.interop import Pretrained
from dew.nn.mixers.mamba import Mamba, selective_scan

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




DECODE_SEEDS = 8
"""The layer's cached decode's declared check-1 seeds against its parallel forward at each step's width,
sized by `seeds_for` on armada."""


def test_the_layer_decodes_as_it_prefills():
    """Over drawn layers and inputs, the layer's single-token steps through its cache on par with its
    parallel forward at each step's width (`as_stepped`) from the float64 layer's forward (check 1,
    ledger 2739), its cache holding a state per channel and state dimension."""
    mixer = Mamba(emb_features=8, intermediate_size=16, state_size=4, time_step_rank=2, chunk_size=4)

    def draw(seed):
        key, inputs = jax.random.split(jax.random.key(seed))
        x = jax.random.normal(inputs, (2, 9, 8))
        return mixer.init(key, x), x

    wide = float64_twin(mixer)
    assert_as_exact(lambda variables, x: decode_through(mixer, variables, x, prefill=1)[0],
                    lambda variables, x: as_stepped(lambda held: mixer.apply(variables, held), x, 1),
                    draw, "mamba decode", seeds=DECODE_SEEDS, truth_fn=wide.apply)
    variables, x = draw(0)
    assert decode_through(mixer, variables, x, prefill=1)[2]['ssm_state'].shape == (2, 16, 4)



@pytest.mark.network
def test_state_spaces_mamba_130m_computes_the_references_logits():
    """state-spaces/mamba-130m-hf in fp32 on CPU, in Dew and in
    transformers 5.16.1, on two prompts cut to a common length."""
    import torch
    import transformers
    source = json.loads((ROOT / 'mamba-130m-hf' / 'source.json').read_text())
    tokenizer = transformers.AutoTokenizer.from_pretrained(source['repo'], revision=source['revision'])
    prompts = ["The capital of France is Paris, and the capital of Germany is",
               "def fibonacci(n):\n    if n < 2:\n        return n\n    return"]
    rows = [tokenizer(prompt)["input_ids"] for prompt in prompts]
    ids = np.asarray([row[:min(map(len, rows))] for row in rows], np.int32)
    reference = transformers.MambaForCausalLM.from_pretrained(
        source['repo'], revision=source['revision'], dtype=torch.float32).eval()
    with torch.no_grad():
        expected = reference(torch.from_numpy(ids.astype(np.int64))).logits.numpy()
    loaded = Pretrained.load(source['repo'], revision=source['revision'], dtype='float32')
    logits = np.asarray(loaded.model.apply(loaded.variables, ids), np.float32)
    assert float(np.max(np.abs(logits - expected))) < 1e-3
    assert np.array_equal(logits.argmax(-1), expected.argmax(-1))
