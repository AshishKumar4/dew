"""RWKV4's WKV (`dew.nn.kernels.wkv4`, fla's fused_recurrent_rwkv4) on the portable math, one associative
scan of its spans' carries, and on the recurrence template's channel walk (`templates.recurrence.wkv4`): its
output, final carry and gradients on par with fla's recurrence a token at a time from float64, one sequence
and packed documents (check 1)."""


import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import interprets
from jax.sharding import Mesh, PartitionSpec as P
from reference_error import assert_as_exact
from test_recurrence_chunk import assert_packed_on_par, hosted

from dew.nn.kernels import generation
from dew.nn.kernels.rwkv4 import recurrent_wkv4, wkv4
from dew.nn.kernels.selection import KernelRefused

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


def test_the_kernels_take_keys_and_values_in_the_dtype_they_run_them_in():
    """ReviewRecurrenceBatch's selector dtype: bf16 keys beside fp64 values under x64 run in their common
    dtype, fp64, which the kernels do not take: named, the call is refused, where a selector reading the keys'
    dtype alone took the kernels and ran values of 2^200 in fp32, infinite."""
    channel = np.zeros((WIDE,), np.float64)
    with jax.enable_x64(new_val=True):
        key = jnp.zeros((ROWS, 9, WIDE), jnp.bfloat16)
        value = jnp.full((ROWS, 9, WIDE), 2.0 ** 200, jnp.float64)
        with pytest.raises(KernelRefused):
            routed("kernels")(channel, channel, key, value, None, None)


def test_a_document_at_the_last_token_cuts_the_final_maximums_cotangent():
    """ReviewRecurrenceBatch's cut: a document starting at the last token, of value 2^100, under a final
    numerator's cotangent of 2^100: taken net of the maximum's shift, the last key's gradient (the final
    maximum's own) is finite, as is every other, where the maximum's cotangent formed apart was 2^200 - 2^200
    and the cut multiplying it by zero made the earlier ones NaN."""
    rng = np.random.default_rng([1455, 3])
    channel = rng.standard_normal((WIDE,)).astype(np.float32)
    key = rng.standard_normal((ROWS, 150, WIDE)).astype(np.float32)
    value = rng.standard_normal((ROWS, 150, WIDE)).astype(np.float32)
    value[:, -1] = 2.0 ** 100
    starts = np.zeros((ROWS, 150), bool)
    starts[:, -1] = True
    dfinal = np.zeros((ROWS, 3, WIDE), np.float32)
    dfinal[:, 0] = 2.0 ** 100
    state = np.stack([np.zeros((ROWS, WIDE)), np.ones((ROWS, WIDE)), np.zeros((ROWS, WIDE))], 1)
    state = state.astype(np.float32)
    _, dtime_decay, dtime_first, dkey, dvalue, dstate = jax.jit(differentiated(routed("kernels")))(
        channel, channel, key, value, state, starts, (np.ones(key.shape, np.float32), dfinal))
    for grad in (dtime_decay, dtime_first, dkey, dvalue, dstate):
        assert np.isfinite(np.asarray(grad)).all()


TIES = {"the last key": (129,), "three keys across a span's end": (63, 64, 65)}
"""Tokens whose keys tie the running maximum aged to them, each case's: the last token's ties the entering
maximum, or three tokens' each tie the one before them, across the first span's end."""


@pytest.mark.parametrize('ties', list(TIES))
@pytest.mark.parametrize('route', list(ROUTES))
def test_a_tie_of_the_running_maximum_goes_to_the_later_key_on_every_route(route, ties):
    """ReviewRecurrenceBatch's ties: a carry entering at a maximum of 0, a decay of -1 a token over 130
    tokens (three spans), every key -1000 but those that tie the maximum aged to them (`TIES`): the final
    maximum's unit cotangent goes to the last tied key whole, then ages to the call's end, a decay gradient
    of one a token after it, and none to the entering maximum or an earlier tied key, alike token by token and
    by the portable scan's tree (`rwkv4.joined`), where halves at each tie split three tied keys apart."""
    tied = TIES[ties]
    time_decay = np.zeros((WIDE,), np.float32)
    key = np.full((ROWS, 130, WIDE), -1000.0, np.float32)
    for token in tied:
        key[:, token] = -(token + 1.0)
    state = np.stack([np.zeros((ROWS, WIDE)), np.ones((ROWS, WIDE)), np.zeros((ROWS, WIDE))], 1)
    state = state.astype(np.float32)
    dfinal = np.zeros(state.shape, np.float32)
    dfinal[:, 2] = 1.0
    _, dtime_decay, _, dkey, _, dstate = differentiated(routed(route))(
        time_decay, time_decay, key, np.zeros(key.shape, np.float32), state, np.zeros((ROWS, 130), bool),
        (np.zeros(key.shape, np.float32), dfinal))
    last = tied[-1]
    expected = np.zeros(key.shape, np.float32)
    expected[:, last] = 1.0
    np.testing.assert_array_equal(np.asarray(dkey), expected)
    np.testing.assert_array_equal(np.asarray(dstate[:, 2]), 0.0)
    # The decay is -exp(time_decay), -1: its gradient is minus the decay's, one a token aged past the key.
    np.testing.assert_array_equal(np.asarray(dtime_decay), -(129.0 - last) * ROWS)


def test_a_tokens_gradients_take_their_large_terms_weighted_first():
    """ReviewRecurrenceBatch's output overflow: a carry entering at `(-3 2^126, 1, 0)`, a key of 0, a value of
    3 2^126 and a bonus of -10 at the first token, every later key -1000 and value 0, an output cotangent of
    2^-10 at the first token: its value less its output, 6 2^126, and its numerator less value times
    denominator are past float32's largest, while each weighted by the cotangent is normal: on the kernels
    every gradient is finite (`_token_back` weighs each first)."""
    time_decay = np.zeros((WIDE,), np.float32)
    time_first = np.full((WIDE,), -10.0, np.float32)
    key = np.full((ROWS, 130, WIDE), -1000.0, np.float32)
    key[:, 0] = 0.0
    value = np.zeros(key.shape, np.float32)
    value[:, 0] = 3.0 * 2.0 ** 126
    state = np.stack([np.full((ROWS, WIDE), -3.0 * 2.0 ** 126), np.ones((ROWS, WIDE)),
                      np.zeros((ROWS, WIDE))], 1).astype(np.float32)
    dy = np.zeros(key.shape, np.float32)
    dy[:, 0] = 2.0 ** -10
    cotangents = (dy, np.zeros(state.shape, np.float32))
    operands = (time_decay, time_first, key, value, state, np.zeros((ROWS, 130), bool), cotangents)
    for grad in differentiated(routed("kernels"))(*operands)[1:]:
        assert np.isfinite(np.asarray(grad)).all()


def _carry(alpha, beta, eps):
    """A carry entering every row, `[ROWS, 3, WIDE]` fp32, each value given a channel."""
    return np.stack([np.full((ROWS, WIDE), value) for value in (alpha, beta, eps)], 1).astype(np.float32)


def _kernel_gradients(time_decay, key, value, state, starts, dy, dfinal, time_first=0.0):
    """The kernels' gradients of every operand under `dy` and `dfinal`, the decay and bonus each one value."""
    channel = np.full((WIDE,), 1.0, np.float32)
    run = differentiated(routed("kernels"))
    return run(channel * time_decay, channel * time_first, key, value, state, starts,
               (dy.astype(np.float32), dfinal.astype(np.float32)))[1:]


def test_a_reset_tokens_key_takes_the_maximums_cotangent_whole():
    """ReviewRecurrenceBatch's net key gradient: one token, a document's first, its key, bonus 0 and value
    2^100, no output cotangent and a final carry's of `(1, 0, 1)`: the key sets the maximum, so its gradient
    is the maximum's cotangent, 1, where the numerator's share added (2^100) and taken back rounded it to
    0."""
    key, value = np.zeros((ROWS, 1, WIDE), np.float32), np.full((ROWS, 1, WIDE), 2.0 ** 100, np.float32)
    dfinal = _carry(1.0, 0.0, 1.0)
    starts = np.ones((ROWS, 1), bool)
    _, _, dkey, _, _ = _kernel_gradients(0.0, key, value, _carry(0.0, 1.0, 0.0), starts, np.zeros(key.shape),
                                         dfinal)
    np.testing.assert_array_equal(np.asarray(dkey), 1.0)


def test_a_carrys_cotangent_follows_the_frame_the_forward_rounded():
    """ReviewRecurrenceBatch's rounded frame: a carry entering at `(1, 1, 3 2^29)`, keys, values and bonus 0
    and a decay of -4 over 150 tokens (three spans): each step's aged maximum rounds back to 3 2^29, so the
    carry's numerator stays 1 and a unit cotangent of the final one is the entering one's, exactly, where a
    span's map back taken from `e^(n w)` dropped it to e^-256."""
    key = np.zeros((ROWS, 150, WIDE), np.float32)
    dfinal = _carry(1.0, 0.0, 0.0)
    *_, dstate = _kernel_gradients(np.log(4.0), key, key, _carry(1.0, 1.0, 3.0 * 2.0 ** 29),
                                   np.zeros((ROWS, 150), bool), np.zeros(key.shape), dfinal)
    np.testing.assert_array_equal(np.asarray(dstate[:, 0]), 1.0)


def test_a_spans_own_cotangent_past_float32s_range_composes_finite():
    """ReviewRecurrenceBatch's span constant: a carry entering at `(0, 1, 0)`, keys of -1000, values and
    bonus 0 and a decay of -e^-1000 (zero), output cotangents of 2^127 at the last two tokens and a final
    numerator's of -2^127 over three spans: the numerator's cotangent is 0 past the last token and 2^127
    before it, so the entering carry's is `(2^127, 0, 0)` exactly, where the last span's own constant from
    none, 2^128, was infinite before it met the final carry's."""
    key = np.full((ROWS, 150, WIDE), -1000.0, np.float32)
    dy = np.zeros(key.shape)
    dy[:, -2:] = 2.0 ** 127
    *_, dstate = _kernel_gradients(-1000.0, key, np.zeros(key.shape, np.float32), _carry(0.0, 1.0, 0.0),
                                   np.zeros((ROWS, 150), bool), dy, _carry(-(2.0 ** 127), 0.0, 0.0))
    np.testing.assert_array_equal(np.asarray(dstate), _carry(2.0 ** 127, 0.0, 0.0))


def test_a_document_cuts_an_infinite_cotangent_from_every_token_before_it():
    """ReviewRecurrenceBatch's severed map: a final carry's cotangent of `(inf, 0, 0)` and a document starting
    at token 100 of 150 (in the second span): every gradient of the tokens before it and of the entering carry
    is exactly 0, where a span's map back multiplied the cut's zero by the infinite cotangent, NaN."""
    rng = np.random.default_rng([1455, 4])
    key = rng.standard_normal((ROWS, 150, WIDE)).astype(np.float32)
    value = rng.standard_normal((ROWS, 150, WIDE)).astype(np.float32)
    starts = np.zeros((ROWS, 150), bool)
    starts[:, 100] = True
    _, _, dkey, dvalue, dstate = _kernel_gradients(0.0, key, value, _carry(0.5, 1.0, 0.0), starts,
                                                   np.zeros(key.shape), _carry(np.inf, 0.0, 0.0))
    for grad in (dkey[:, :100], dvalue[:, :100], dstate):
        np.testing.assert_array_equal(np.asarray(grad), 0.0)


def test_a_reset_tokens_key_takes_an_infinite_numerators_cotangent_from_none_of_its_past():
    """ReviewRecurrenceBatch's zero factor: one token, a document's first, its key, bonus 0 and value 1, no
    output cotangent and a final carry's of `(inf, 0, 1)`: the past's share at a document's first token is
    none, so the key's gradient is the maximum's cotangent, 1, where the infinite cotangent times the past's
    zero factor made it NaN."""
    key, value = np.zeros((ROWS, 1, WIDE), np.float32), np.ones((ROWS, 1, WIDE), np.float32)
    starts = np.ones((ROWS, 1), bool)
    _, _, dkey, _, _ = _kernel_gradients(0.0, key, value, _carry(0.0, 1.0, 0.0), starts, np.zeros(key.shape),
                                         _carry(np.inf, 0.0, 1.0))
    np.testing.assert_array_equal(np.asarray(dkey), 1.0)


def test_a_rows_parameter_gradient_sums_its_spans_apart():
    """ReviewRecurrenceBatch's parameter sums: two-token documents of values -2 and 2 (keys and bonus 0, so
    a document's second token's bonus gradient is its output cotangent), output cotangents of -2^127 and
    2^127 at two second tokens of the second span and 2^127 at one of the third, in the first row alone: the
    bonus's gradient is 2^127, each span's partial summed apart (0 and 2^127), where summed through the row
    from its end it passed 2^128 before the -2^127, NaN."""
    key = np.zeros((ROWS, 150, WIDE), np.float32)
    value = np.where(np.arange(150) % 2 == 0, -2.0, 2.0)[None, :, None] * np.ones(key.shape, np.float32)
    starts = np.broadcast_to(np.arange(150) % 2 == 0, (ROWS, 150))
    dy = np.zeros(key.shape)
    dy[0, 65], dy[0, 67], dy[0, 129] = -(2.0 ** 127), 2.0 ** 127, 2.0 ** 127
    _, dtime_first, _, _, _ = _kernel_gradients(0.0, key, value.astype(np.float32), _carry(0.0, 1.0, 0.0),
                                                starts, dy, _carry(0.0, 0.0, 0.0))
    np.testing.assert_array_equal(np.asarray(dtime_first), 2.0 ** 127)


@pytest.mark.mesh(devices=2)
def test_the_kernels_refuse_a_checked_shard_map_whose_values_alone_vary():
    """ReviewRecurrenceBatch's selector variation: inside a `shard_map` that checks varying axes, keys whole
    beside values split by row vary the call, so the kernels named refuse it (a Pallas call states no varying
    type for its results), where a selector reading the keys' variation alone took them."""
    channel = np.zeros((WIDE,), np.float32)

    def run(key, value):
        return routed("kernels")(channel, channel, key, value, None, None)[0]

    mapped = jax.shard_map(run, mesh=Mesh(np.asarray(jax.devices()[:2]), ("data",)),
                           in_specs=(P(), P("data")), out_specs=P("data"), check_vma=True)
    with pytest.raises(KernelRefused):
        jax.jit(mapped)(np.zeros((1, 9, WIDE), np.float32), np.zeros((2, 9, WIDE), np.float32))
