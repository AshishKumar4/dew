"""RWKV4's WKV (`dew.nn.kernels.wkv4`, fla's fused_recurrent_rwkv4) on the portable math, one associative
scan of its spans' carries, and on the recurrence template's channel walk (`templates.recurrence.wkv4`): its
output, final carry and gradients on par with fla's recurrence a token at a time from float64, one sequence
and packed documents (check 1)."""


import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import interprets
from reference_error import assert_as_exact
from test_recurrence_chunk import assert_packed_on_par, hosted

from dew.nn.kernels import generation
from dew.nn.kernels.rwkv4 import recurrent_wkv4, wkv4

ROWS, LENGTH, CHANNELS = 2, 70, 16
DOCUMENTS = ((0, 23, 40), (9, 48))
"""Each row's packed documents' first tokens: the first row's carry dropped at token 0, the second's
entering its first document."""
SEEDS = {False: 256, True: 512}
"""Each site's declared seeds (one sequence, packed), sized by `seeds_for` on armada."""


def draw(seed: int):
    """The WKV's fp32 operands from `seed` (`hosted`) and a cotangent of its output and final carry: a
    channel's decay and bonus, keys spread over a few units (the running maximum's range), values, and a
    carry entering whose denominator is positive and numerator its values' mean's scale."""
    normal = np.random.default_rng([seed, 1]).standard_normal
    time_decay, time_first = normal((CHANNELS,)), normal((CHANNELS,))
    key, value = 2.0 * normal((ROWS, LENGTH, CHANNELS)), normal((ROWS, LENGTH, CHANNELS))
    beta = np.exp(normal((ROWS, CHANNELS)))
    state = np.stack([beta * normal((ROWS, CHANNELS)), beta, normal((ROWS, CHANNELS))], axis=1)
    cotangent = (normal((ROWS, LENGTH, CHANNELS)), normal((ROWS, 3, CHANNELS)))
    return *(hosted(t) for t in (time_decay, time_first, key, value, state)), jax.tree.map(hosted, cotangent)


@pytest.mark.parametrize('packed', (False, True))
def test_the_scan_is_as_exact_as_the_recurrence_a_token_at_a_time(packed):
    """The scan's output, final carry and every operand's gradient on par with fla's recurrence a token at a
    time from float64, one sequence and packed documents (check 1): each span's carry rescaled to the larger
    running maximum as it joins, a document's first token reading an empty past."""
    # Packed, the first row's carry entering is dropped at its token 0, so its gradient is exactly zero in
    # both arms: held to the reference's residue there, zero, no float64 terms traced (the token loop's terms
    # took 11.5 GB and were killed, jsf-wkv4t1); one sequence holds no zero row, so none are traced either.
    assert_packed_on_par(wkv4, recurrent_wkv4, draw, 'wkv4', packed=packed, documents=DOCUMENTS,
                         length=LENGTH, seeds=SEEDS[packed], terms=False if packed else None)


@pytest.mark.parametrize("wkv", [wkv4, recurrent_wkv4], ids=["scan", "recurrence"])
def test_a_first_token_reads_its_own_value_whatever_its_key(wkv):
    """A sequence's first token reads an empty past beside itself at its bonus, so its output is its own value
    exactly whatever its key: a key of -2e38, below any finite maximum an empty past could stand at, among
    them, its gradients finite."""
    channel = np.zeros((1,), np.float32)
    key, value = np.full((1, 1, 1), -2e38, np.float32), np.ones((1, 1, 1), np.float32)
    out, _ = wkv(channel, channel, key, value)
    np.testing.assert_array_equal(np.asarray(out), value)
    grads = jax.grad(lambda *operands: wkv(*operands)[0].sum(), argnums=(0, 1, 2, 3))(channel, channel, key,
                                                                                        value)
    assert all(np.isfinite(np.asarray(grad)).all() for grad in grads)


INTERPRETED = interprets()

ROUTES = {"portable": "xla", "kernels": "mosaic_gpu"}

WIDE = 128
"""The channels the kernels' cells take: a warpgroup's, a channel a thread (`templates.recurrence.wkv4`)."""


def routed(route: str):
    """`wkv4` on `route`'s kernel, the kernels interpreted off a GPU at an H100's facts
    (`generation.assumed`'s `interpreted`: 64 tokens a span)."""
    def run(time_decay, time_first, key, value, state, starts):
        with (generation.assumed("sm90" if INTERPRETED else None, interpreted=INTERPRETED),
              generation.chosen(wkv4=ROUTES[route])):
            return wkv4(time_decay, time_first, key, value, state, starts)

    return run


def differentiated(fn):
    """`fn(time_decay, time_first, key, value, state, starts)` as `run(..., starts, cotangents)`, which
    returns its output and final carry and the gradients of every operand but `starts` under the
    cotangents."""
    def run(time_decay, time_first, key, value, state, starts, cotangents):
        operands = (time_decay, time_first, key, value, state)
        out, pull = jax.vjp(lambda *given: fn(*given, starts), *operands)
        return out, *pull(cotangents)

    return run


def wide_draws(length: int, dtype, packed: bool):
    """A seed's operands at the kernels' width (`WIDE`), `draw`'s spread, keys and values in `dtype`, each
    row's documents' first tokens where `packed` (a row's first among them), and the cotangents last."""
    def wide(seed):
        normal = np.random.default_rng([1455, seed, length]).standard_normal
        time_decay, time_first = normal((WIDE,)), normal((WIDE,))
        key, value = 2.0 * normal((ROWS, length, WIDE)), normal((ROWS, length, WIDE))
        beta = np.exp(normal((ROWS, WIDE)))
        state = np.stack([beta * normal((ROWS, WIDE)), beta, normal((ROWS, WIDE))], axis=1)
        starts = np.zeros((ROWS, length), bool)
        if packed:
            starts[:, 0] = [True, False]
            starts[:, np.random.default_rng([1455, seed]).integers(1, length, size=3)] = True
        cotangents = (normal(key.shape).astype(dtype), normal(state.shape).astype(np.float32))
        return (time_decay.astype(np.float32), time_first.astype(np.float32), key.astype(dtype),
                value.astype(dtype), state.astype(np.float32), starts, cotangents)
    return wide


@pytest.mark.parametrize('packed', [False, True], ids=['one document', 'packed'])
@pytest.mark.parametrize('length', [9, 64, 150], ids=['a turn', 'a span', 'three spans'])
@pytest.mark.parametrize('dtype', [np.float32, jnp.bfloat16], ids=['fp32', 'bf16'])
def test_the_kernels_are_as_exact_as_the_recurrence_a_token_at_a_time(dtype, length, packed):
    """The WKV on the template's channel walk (`templates.recurrence.wkv4`, interpreted off a GPU) over one
    span and over several: its output, final carry and every operand's gradient on par with fla's recurrence
    a token at a time at the call's dtype from float64 (check 1), the backward's in the numerator's and
    denominator's own terms with the running maximum's path."""
    assert_as_exact(jax.jit(differentiated(routed("kernels"))), jax.jit(differentiated(recurrent_wkv4)),
                    wide_draws(length, dtype, packed),
                    f"wkv4 on the kernels, {length} tokens in {jnp.dtype(dtype).name}", seeds=16)


@pytest.mark.parametrize('route', list(ROUTES))
def test_a_wkv_of_no_tokens_hands_its_carry_through(route):
    """A call of no tokens walks none: no output, its final carry the one entering, whose gradient is the
    final carry's cotangent."""
    channel = np.zeros((WIDE,), np.float32)
    key = np.zeros((ROWS, 0, WIDE), np.float32)
    state = np.arange(ROWS * 3 * WIDE, dtype=np.float32).reshape(ROWS, 3, WIDE)
    cotangents = (key, state + 1.0)
    (out, final), *_, dstate = differentiated(routed(route))(channel, channel, key, key, state,
                                                             np.zeros((ROWS, 0), bool), cotangents)
    assert out.shape == key.shape
    np.testing.assert_array_equal(np.asarray(final), state)
    np.testing.assert_array_equal(np.asarray(dstate), state + 1.0)
