"""A diagonal linear recurrence (`dew.nn.kernels.diagonal_scan`: HGRN's, fla's `chunk_hgrn`),
`h_t = exp(g_t) h_{t-1} + x_t` a channel, held to the token recurrence on the portable math and on the
template's kernels (`templates.recurrence.diagonal`, interpreted off a GPU) (row 1453)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import interprets
from reference_error import FEWEST_SEEDS, assert_as_exact

from dew.nn.kernels import diagonal_scan, generation

ROWS, CHANNELS = 2, 256

INTERPRETED = interprets()

ROUTES = {"portable": "xla", "kernels": "mosaic_gpu"}


def token_by_token(x, g, state, starts):
    """The recurrence a token at a time, as fla's naive HGRN writes it (fla/ops/hgrn/naive.py): each
    token's state `exp(g_t) h_{t-1} + x_t` its output, the last the final state; a document's first token
    (`starts`) its own write alone."""
    def step(h, token):
        x_t, g_t, first = token
        h = jnp.where(first[:, None], x_t, jnp.exp(g_t) * h + x_t)
        return h, h

    final, out = jax.lax.scan(step, state, (jnp.moveaxis(x, 1, 0), jnp.moveaxis(g, 1, 0), starts.T))
    return jnp.moveaxis(out, 0, 1), final


def differentiated(fn):
    """`fn(x, g, state, starts)` as `run(x, g, state, starts, cotangents)`, which returns its output and final
    state and the gradients of `x`, `g` and `state` under the cotangents (`starts` is no input of a
    gradient)."""
    def run(x, g, state, starts, cotangents):
        out, pull = jax.vjp(lambda x, g, state: fn(x, g, state, starts), x, g, state)
        return out, *pull(cotangents)

    return run


def routed(route: str):
    """`diagonal_scan` on `route`'s kernel, the kernels interpreted off a GPU at an H100's facts
    (`generation.assumed`'s `interpreted`: 64 tokens a span)."""
    def scan(x, g, state, starts):
        with (generation.assumed("sm90" if INTERPRETED else None, interpreted=INTERPRETED),
              generation.chosen(diagonal_scan=ROUTES[route])):
            return diagonal_scan(x, g, state, starts=starts)

    return scan


def stepped(x, g, state, starts):
    """The token recurrence at the call's dtype: fp32 states, the outputs rounded to `x`'s dtype."""
    out, final = token_by_token(x.astype(jnp.float32), g, state, starts)
    return out.astype(x.dtype), final


def draws(length: int, dtype, packed: bool):
    """A seed's inputs, their outputs' cotangents last (`differentiated`'s): writes normal in `dtype`, log
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


@pytest.mark.parametrize('packed', [False, True], ids=['one document', 'packed'])
@pytest.mark.parametrize('length', [9, 64, 150], ids=['a turn', 'a span', 'three spans'])
@pytest.mark.parametrize('dtype', [np.float32, jnp.bfloat16], ids=['fp32', 'bf16'])
@pytest.mark.parametrize('route', list(ROUTES))
def test_a_diagonal_scan_is_as_exact_as_its_token_recurrence(route, dtype, length, packed):
    """The diagonal recurrence on each route, the kernels' walks over one span and over several: its output,
    final state and the gradients of its writes, gates and entering state on par from float64 with the
    token recurrence at the call's dtype (check 1), the truth the token recurrence in float64."""
    assert_as_exact(jax.jit(differentiated(routed(route))), jax.jit(differentiated(stepped)),
                    draws(length, dtype, packed),
                    f"a diagonal scan on the {route}, {length} tokens in {jnp.dtype(dtype).name}", seeds=16,
                    truth_fn=jax.jit(differentiated(token_by_token)))


def test_a_spans_decay_below_float32s_range_carries_a_normal_state():
    """ReviewRecurrenceBatch's span underflow: a state of 2^100 decayed by `g = -2` over the first span's 64
    tokens, then held, is 2^-84.7, normal, on every later span: the kernels carry the first span's decay,
    e^-128, past float32's least normal 2^-126 (`diagonal._decayed`) and are on par with the token
    recurrence from float64 (check 1), where a span's decay held as one float32 flushed to zero and the
    later spans entered from nothing."""
    def draw(seed):
        rng = np.random.default_rng([1453, 1, seed])
        g = np.zeros((ROWS, 300, CHANNELS))
        g[:, :64] = -2.0 + 0.01 * rng.standard_normal((ROWS, 64, CHANNELS))
        state = 2.0 ** 100 * (1.0 + 0.1 * rng.standard_normal((ROWS, CHANNELS)))
        cotangents = (rng.standard_normal(g.shape).astype(np.float32),
                      rng.standard_normal(state.shape).astype(np.float32))
        return (np.zeros(g.shape, np.float32), g.astype(np.float32), state.astype(np.float32),
                np.zeros((ROWS, 300), bool), cotangents)

    assert_as_exact(jax.jit(differentiated(routed("kernels"))), jax.jit(differentiated(stepped)), draw,
                    "a span's decay below float32's range", seeds=FEWEST_SEEDS,
                    truth_fn=jax.jit(differentiated(token_by_token)))


def test_a_gates_gradient_forms_its_product_as_reverse_mode_does():
    """ReviewRecurrenceBatch's adjoint underflow: a token's gate gradient `(q_t h_{t-1}) a_t` with `q_t` near
    2^-100, `h_{t-1}` near 2^100 and `a_t = e^-64`, normal, on par with the token recurrence's reverse mode
    from float64 (check 1), which forms `q_t h_{t-1}` first; formed `(q_t a_t) h_{t-1}`, its first product,
    2^-192, flushed to zero."""
    def draw(seed):
        rng = np.random.default_rng([1453, 2, seed])
        g = np.zeros((ROWS, 9, CHANNELS))
        g[:, 1] = -64.0 + 0.01 * rng.standard_normal((ROWS, CHANNELS))
        dy = np.zeros(g.shape)
        dy[:, 1] = 2.0 ** -100 * (1.0 + 0.1 * rng.standard_normal((ROWS, CHANNELS)))
        state = 2.0 ** 100 * (1.0 + 0.1 * rng.standard_normal((ROWS, CHANNELS)))
        return (np.zeros(g.shape, np.float32), g.astype(np.float32), state.astype(np.float32),
                np.zeros((ROWS, 9), bool), (dy.astype(np.float32), np.zeros((ROWS, CHANNELS), np.float32)))

    assert_as_exact(jax.jit(differentiated(routed("kernels"))), jax.jit(differentiated(stepped)), draw,
                    "a gate's gradient past a decay of e^-64", seeds=FEWEST_SEEDS,
                    truth_fn=jax.jit(differentiated(token_by_token)))


def test_a_document_after_an_overflowed_one_reads_none_of_it():
    """ReviewRecurrenceBatch's reset leak: a row's first document overflows float32 (two writes of 2^127)
    at the first span's end, and the next starts at the second span's first token with a write of one,
    every gate zero: the next document's outputs and final state are one, and its writes' and gates'
    gradients under unit cotangents its tokens' counts to the row's end, each exact and finite, where a
    reset multiplying the overflowed state by a zero decay made them NaN."""
    x = np.zeros((ROWS, 150, CHANNELS), np.float32)
    x[:, 62:64] = 2.0 ** 127
    x[:, 64] = 1.0
    starts = np.zeros((ROWS, 150), bool)
    starts[:, 64] = True
    g, state = np.zeros(x.shape, np.float32), np.zeros((ROWS, CHANNELS), np.float32)
    cotangents = (np.ones(x.shape, np.float32), np.zeros(state.shape, np.float32))
    (out, final), dx, dg, _ = jax.jit(differentiated(routed("kernels")))(x, g, state, starts, cotangents)
    np.testing.assert_array_equal(np.asarray(out[:, 64:]), 1.0)
    np.testing.assert_array_equal(np.asarray(final), 1.0)
    later = np.broadcast_to((150 - np.arange(64, 150, dtype=np.float32))[None, :, None], (ROWS, 86, CHANNELS))
    np.testing.assert_array_equal(np.asarray(dx[:, 64:]), later)
    np.testing.assert_array_equal(np.asarray(dg[:, 65:]), later[:, 1:])
    np.testing.assert_array_equal(np.asarray(dg[:, 64]), 0.0)


@pytest.mark.parametrize('route', list(ROUTES))
def test_a_diagonal_scan_of_no_tokens_hands_its_state_through(route):
    """A call of no tokens walks none: no output, its final state the one entering, whose gradient is the
    final state's cotangent."""
    x, g = np.zeros((ROWS, 0, CHANNELS), np.float32), np.zeros((ROWS, 0, CHANNELS), np.float32)
    state = np.arange(ROWS * CHANNELS, dtype=np.float32).reshape(ROWS, CHANNELS)
    cotangents = (x, state + 1.0)
    (out, final), _, _, dstate = differentiated(routed(route))(x, g, state, np.zeros((ROWS, 0), bool),
                                                               cotangents)
    assert out.shape == x.shape
    np.testing.assert_array_equal(np.asarray(final), state)
    np.testing.assert_array_equal(np.asarray(dstate), state + 1.0)
