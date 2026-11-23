"""A diagonal linear recurrence (`diagonal_scan.diagonal_scan`: HGRN's, fla's `chunk_hgrn`),
`h_t = exp(g_t) h_{t-1} + x_t` a channel, held to the token recurrence on the portable math and on the
template's kernels (`templates.recurrence.diagonal`, interpreted off a GPU) (row 1453)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import interprets
from reference_error import assert_as_exact

from dew.nn.kernels import generation
from dew.nn.kernels.diagonal_scan import diagonal_scan

ROWS, CHANNELS = 2, 256

INTERPRETED = interprets()


def token_by_token(x, g, state, starts):
    """The recurrence a token at a time, as fla's naive HGRN writes it (fla/ops/hgrn/naive.py): each
    token's state `exp(g_t) h_{t-1} + x_t` its output, the last the final state; a document's first token
    (`starts`) from a zero state."""
    def step(h, token):
        x_t, g_t, first = token
        h = jnp.where(first[:, None], 0.0, jnp.exp(g_t)) * h + x_t
        return h, h

    final, out = jax.lax.scan(step, state, (jnp.moveaxis(x, 1, 0), jnp.moveaxis(g, 1, 0), starts.T))
    return jnp.moveaxis(out, 0, 1), final


def draws(length: int, dtype, packed: bool):
    """A seed's inputs, their outputs' cotangents last (`with_gradients`'): writes normal in `dtype`, log
    forget gates `log sigmoid(n + 2)` (HGRN's forget gate a sigmoid, most near one), a state entering,
    fp32, and each row's documents' first tokens, a row's first among them where `packed`."""
    def draw(seed):
        rng = np.random.default_rng([1453, seed, length])
        x = rng.standard_normal((ROWS, length, CHANNELS))
        g = -np.logaddexp(0.0, -(rng.standard_normal((ROWS, length, CHANNELS)) + 2.0))
        state = rng.standard_normal((ROWS, CHANNELS))
        starts = np.zeros((ROWS, length), bool)
        if packed:
            starts[:, 0] = [True, False]
            starts[:, rng.integers(1, length, size=3)] = True
        cotangents = (rng.standard_normal(x.shape).astype(dtype),
                      rng.standard_normal(state.shape).astype(np.float32))
        return x.astype(dtype), g.astype(np.float32), state.astype(np.float32), starts, cotangents
    return draw


def with_gradients(fn):
    """`fn(x, g, state, starts)` as `run(x, g, state, starts, cotangents)`, which returns its output and final
    state and the gradients of `x`, `g` and `state` under the cotangents (`starts` is no input of a
    gradient)."""
    def run(x, g, state, starts, cotangents):
        out, pull = jax.vjp(lambda x, g, state: fn(x, g, state, starts), x, g, state)
        return out, *pull(cotangents)

    return run


ROUTES = {"portable": "xla", "kernels": "mosaic_gpu"}


@pytest.mark.parametrize('packed', [False, True], ids=['one document', 'packed'])
@pytest.mark.parametrize('length', [9, 64, 150], ids=['a turn', 'a span', 'three spans'])
@pytest.mark.parametrize('dtype', [np.float32, jnp.bfloat16], ids=['fp32', 'bf16'])
@pytest.mark.parametrize('route', list(ROUTES))
def test_a_diagonal_scan_is_as_exact_as_its_token_recurrence(route, dtype, length, packed):
    """The diagonal recurrence on each route, the kernels' walks over one span and over several (64 tokens a
    span at an interpreted H100's facts, `diagonal.walk`): its output, final state and the gradients of its
    writes, gates and entering state on par from float64 with the token recurrence at the call's dtype
    (check 1), the truth the token recurrence in float64."""
    def routed(x, g, state, starts):
        # Off a GPU the kernels interpreted at an H100's facts (`generation.assumed`'s `interpreted`).
        with (generation.assumed("sm90" if INTERPRETED else None, interpreted=INTERPRETED),
              generation.chosen(diagonal_scan=ROUTES[route])):
            return diagonal_scan(x, g, state, starts=starts)

    def stepped(x, g, state, starts):
        out, final = token_by_token(x.astype(jnp.float32), g, state, starts)
        return out.astype(x.dtype), final

    assert_as_exact(jax.jit(with_gradients(routed)), jax.jit(with_gradients(stepped)),
                    draws(length, dtype, packed),
                    f"a diagonal scan on the {route}, {length} tokens in {jnp.dtype(dtype).name}", seeds=16,
                    truth_fn=jax.jit(with_gradients(token_by_token)))
