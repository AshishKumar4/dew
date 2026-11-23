"""A diagonal linear recurrence (`recurrence_chunk.diagonal_scan`: HGRN's, fla's `chunk_hgrn`), `h_t = exp(g_t)
h_{t-1} + x_t` a channel, held to the token recurrence (row 1453)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact, with_gradients

from dew.nn.kernels.recurrence_chunk import diagonal_scan

ROWS, CHANNELS = 2, 128


def token_by_token(x, g, state):
    """The recurrence a token at a time, as fla's naive HGRN writes it (fla/ops/hgrn/naive.py): each
    token's state `exp(g_t) h_{t-1} + x_t` its output, the last the final state."""
    def step(h, token):
        x_t, g_t = token
        h = jnp.exp(g_t) * h + x_t
        return h, h

    final, out = jax.lax.scan(step, state, (jnp.moveaxis(x, 1, 0), jnp.moveaxis(g, 1, 0)))
    return jnp.moveaxis(out, 0, 1), final


def draws(length: int, dtype):
    """A seed's inputs, their outputs' cotangents last (`with_gradients`'): writes normal in `dtype`, log
    forget gates `log sigmoid(n + 2)` (HGRN's forget gate a sigmoid, most near one), a state entering,
    fp32."""
    def draw(seed):
        rng = np.random.default_rng([1453, seed, length])
        x = rng.standard_normal((ROWS, length, CHANNELS))
        g = -np.logaddexp(0.0, -(rng.standard_normal((ROWS, length, CHANNELS)) + 2.0))
        state = rng.standard_normal((ROWS, CHANNELS))
        cotangents = (rng.standard_normal(x.shape).astype(dtype),
                      rng.standard_normal(state.shape).astype(np.float32))
        return x.astype(dtype), g.astype(np.float32), state.astype(np.float32), cotangents
    return draw


@pytest.mark.parametrize('length', [64, 150], ids=['a chunk', 'off the chunk grid'])
@pytest.mark.parametrize('dtype', [np.float32, jnp.bfloat16], ids=['fp32', 'bf16'])
def test_a_diagonal_scan_is_as_exact_as_its_token_recurrence(length, dtype):
    """The diagonal recurrence on the portable math: its output, final state and the gradients of its
    writes, gates and entering state on par from float64 with the token recurrence at the call's dtype
    (check 1), the truth the token recurrence in float64."""
    def chunked(x, g, state):
        return diagonal_scan(x, g, state)

    def stepped(x, g, state):
        out, final = token_by_token(x.astype(jnp.float32), g, state)
        return out.astype(x.dtype), final

    assert_as_exact(jax.jit(with_gradients(chunked)), jax.jit(with_gradients(stepped)), draws(length, dtype),
                    f"a diagonal scan, {length} tokens in {jnp.dtype(dtype).name}", seeds=16,
                    truth_fn=jax.jit(with_gradients(token_by_token)))
