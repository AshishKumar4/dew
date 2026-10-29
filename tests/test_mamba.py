"""Mamba's selective scan and a `MambaForCausalLM` checkpoint against
transformers 5.16.1: tools/mamba_reference.py writes tests/fixtures/hf/mamba-tiny."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model_support import decode_through
from sequence_split_support import SPLIT_ROWS, assert_split_as_whole

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




def test_the_layer_decodes_as_it_prefills():
    mixer = Mamba(emb_features=8, intermediate_size=16, state_size=4, time_step_rank=2, chunk_size=4)
    x = jax.random.normal(jax.random.key(0), (2, 9, 8))
    variables = mixer.init(jax.random.key(1), x)
    whole = mixer.apply(variables, x)
    decoded, _, cache = decode_through(mixer, variables, x, prefill=1)
    np.testing.assert_allclose(decoded, whole, rtol=1e-5, atol=1e-5)
    assert cache['ssm_state'].shape == (2, 16, 4)



SPLIT_SEEDS = {"all_valid": 64, "right_padded": 64, "left_padded": 64, "gaps": 64, "packed": 64,
               "empty_shard": 64, "empty_shard_packed": 64}
"""Each row's declared draws, sized by `seeds_for` on armada paired with the unsplit layer."""


@pytest.mark.mesh
@pytest.mark.parametrize("row", sorted(SPLIT_ROWS))
def test_a_layer_split_over_the_sequence_runs_padded_and_packed_rows_as_whole_ones(row):
    """Under a sequence axis of 4, the layer runs rows with padding and packed documents as `held_scan`
    runs them on one device (`assert_split_as_whole`): each shard's conv reads the real tokens before it
    from the earlier shards and its scan enters with the state they leave (ledger 1710)."""
    mixer = Mamba(emb_features=8, intermediate_size=16, state_size=4, time_step_rank=2, chunk_size=4)
    variables = mixer.init(jax.random.key(1), jnp.zeros((1, 32, 8)))
    assert_split_as_whole(mixer, variables, row, SPLIT_SEEDS[row], f"mamba split over the sequence, {row}",
                          width=8)


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
