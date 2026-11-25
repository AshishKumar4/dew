"""The decays a chunk of a linear recurrence reads (`dew.algebra.scan`'s `_Decays`): the decays themselves,
their gradient and their forward derivatives, first and second, each on par with each span's own sums
(row 5858): a pair's exponent summed over its span alone, where a difference of prefix sums carries the
rounding of the whole accumulated magnitude."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact

from dew.algebra import scan

TOKENS, KEYS = 64, 16
"""A drawn chunk's tokens and keys, each key's log decays drawn in [-3, -1]."""

SPANNED = jnp.asarray((np.arange(TOKENS)[None, :, None] < np.arange(TOKENS)[None, None, :])
                      & (np.arange(TOKENS)[None, None, :] <= np.arange(TOKENS)[:, None, None]))
"""Each pair's span, `SPANNED[i, j, s]`: `j < s <= i`, the tokens whose decays the decay from `j` to `i`
multiplies."""

BELOW = jnp.asarray(np.arange(TOKENS)[:, None] >= np.arange(TOKENS)[None, :])[..., None]
"""The pairs a chunk's decays hold, `i >= j`; every other is zero."""


def between(g):
    """A chunk's decays between positions, `[T, T, K]` of its log decays `g` `[T, K]` (`_Decays.between`)."""
    return scan._decays_of(g, base2=False).between


def spanned(values):
    """Each pair's span's own sum of `values` `[T, K]`, `[T, T, K]` (`SPANNED`), formed over the span's mask
    and no difference of prefixes."""
    return jnp.einsum("ijs,sk->ijk", SPANNED.astype(values.dtype), values)


def spans_decays(g):
    """Each pair's decay from its span's own sum of log decays, `exp(spanned(g))` on and below the diagonal
    (`BELOW`), zero above."""
    return jnp.where(BELOW, jnp.exp(spanned(g)), 0.0)


def drawing(stream: int, *shapes):
    """`draw(seed)`: a chunk's log decays `[T, K]` in [-3, -1] and drawn normals of `shapes`, in float32."""
    def draw(seed):
        rng = np.random.default_rng([stream, seed])
        return tuple(jnp.asarray(x, jnp.float32) for x in (
            rng.uniform(-3, -1, (TOKENS, KEYS)), *(rng.standard_normal(shape) for shape in shapes)))
    return draw


DECAY_SEEDS = 8
"""The decays' declared draws against each span's own sum (check 1), sized by `seeds_for` on armada."""


def test_a_decay_deep_in_a_chunk_is_on_par_with_its_own_spans_sum():
    """Each pairwise decay is on par with the exponential of its own span's sum of log decays
    (`spans_decays`) from it in float64 (check 1), however far into the chunk the pair sits, and every decay
    above the diagonal is zero. A difference of plain cumulative sums carries the rounding of the whole
    accumulated magnitude instead: at log decays near -2 over 64 positions, about 128 u on a decay between
    neighbours, which the KDA gradient amplified to twice the reference's distance from float64
    (tests/test_kimi_linear.py)."""
    draw = drawing(0x585B)
    assert not np.asarray(jax.jit(between)(*draw(0)))[~np.asarray(BELOW)[..., 0]].any()
    assert_as_exact(between, spans_decays, draw, "the decays", seeds=DECAY_SEEDS)


GRADIENT_SEEDS = 8
"""The decays' gradient's declared draws against the spanning pairs' own sums (check 1), sized by `seeds_for`
on armada."""


def test_the_decay_gradient_is_on_par_with_the_pairs_spanning_each_position():
    """The decays' gradient at a position `p`, `sum_{i >= p > j} cotangent[i, j] decay[i, j]`, is on par
    with that sum of the pairs spanning `p` alone (each decay its span's own, `spans_decays`) from it in
    float64 (check 1). The prefix difference's gradient summed every pair after `p` into it with opposite
    signs, which cancelled."""
    def gradient(g, cotangent):
        return jax.vjp(between, g)[1](cotangent)[0]

    def pairs(g, cotangent):
        # The pairs spanning `p` are those whose span holds it (`SPANNED[i, j, p]`).
        return jnp.einsum("ijp,ijk->pk", SPANNED.astype(g.dtype), cotangent * spans_decays(g))

    assert_as_exact(gradient, pairs, drawing(0x585C, (TOKENS, TOKENS, KEYS)), "the decays' gradient",
                    seeds=GRADIENT_SEEDS)


FIRST_TANGENT_SEEDS = 8
"""The decays' first forward derivative's declared draws against the explicit spans' (check 1)."""


def test_the_decays_take_their_tangent_from_each_spans_own_sums():
    """A chunk's decays between positions (`_Decays.between`), each key's over 64 tokens of log decays in
    [-3, -1], differentiated forward along a drawn direction `u`: `between * U`, `U` the direction summed
    over each pair's span, `j < s <= i`. Dew's forward derivative itself, no pairing summed after it, is on
    par with that product of each span's own sum (over the span's mask) from it in float64 (check 1, two
    programs of Dew's)."""
    def tangent(g, u):
        return jax.jvp(between, (g,), (u,))[1]

    def spans(g, u):
        return between(g) * spanned(u)

    assert_as_exact(tangent, spans, drawing(0x585A, (TOKENS, KEYS)), "the decays' tangent",
                    seeds=FIRST_TANGENT_SEEDS)


SECOND_TANGENT_SEEDS = 8
"""The decays' mixed second forward derivative's declared draws against the explicit spans' (check 1), sized
by `seeds_for` on armada."""


def test_the_decays_take_their_second_tangent_from_each_spans_own_sums():
    """A chunk's decays between positions (`_Decays.between`), each key's over 64 tokens of log decays in
    [-3, -1], differentiated forward twice along two drawn directions `u` and `w`: `between * U * W`, where
    `U` and `W` sum each direction over a pair's span, `j < s <= i`. Dew's second forward derivative is on
    par with that product of each span's own sum (formed over the span's mask, no difference of prefixes)
    from it in float64 (check 1, two programs of Dew's)."""
    def second(g, u, w):
        return jax.jvp(lambda g: jax.jvp(between, (g,), (u,))[1], (g,), (w,))[1]

    def spans(g, u, w):
        return between(g) * spanned(u) * spanned(w)

    assert_as_exact(second, spans, drawing(0x5859, (TOKENS, KEYS), (TOKENS, KEYS)),
                    "the decays' second tangent", seeds=SECOND_TANGENT_SEEDS)


PAIRING_SEEDS = 8
"""The decays' pairings' declared draws against the plain decays' (check 1), sized by `seeds_for` on
armada."""


def plain_between(g):
    """`between` as its declaration states it, with no compensation: `exp(G_s - G_t)` of the plain prefix
    sums `G` of a chunk's log decays `g` `[T, K]`, on and below the diagonal."""
    prefix = jnp.cumsum(g, axis=0)
    return jnp.where(BELOW, jnp.exp(jnp.where(BELOW, prefix[:, None] - prefix[None, :], 0.0)), 0.0)


def test_the_decay_takes_the_same_tangent_forward_as_backward():
    """Forward mode takes the decay gradient's tangent: a drawn tangent's forward-mode decays paired with a
    drawn cotangent, and the cotangent's gradient paired with the tangent, each on par with JAX's own
    derivatives of the plain decays (`plain_between`) from them in float64 (check 1), each batch row's and
    key dimension's pairing a value of its own (each walks its own decays). The two pairings are not held
    to each other: the forward one sums 64 x 64 products in float32 where the reverse one sums 64, and
    measured 2.4 times its error for that alone; the tangent itself is on par with each span's own sums
    (`test_the_decays_take_their_tangent_from_each_spans_own_sums`)."""
    rows = 4

    def draw(seed):
        rng = np.random.default_rng([0x7A9E, seed])
        return tuple(jnp.asarray(x, jnp.float32) for x in (
            rng.uniform(-3, -1, (rows, TOKENS, KEYS)), rng.standard_normal((rows, TOKENS, KEYS)),
            rng.standard_normal((rows, TOKENS, TOKENS, KEYS))))

    def paired(decays):
        decays = jax.vmap(decays)

        def run(g, tangent, cotangent):
            forward = jnp.sum(jax.jvp(decays, (g,), (tangent,))[1] * cotangent, axis=(1, 2))
            backward = jnp.sum(tangent * jax.vjp(decays, g)[1](cotangent)[0], axis=1)
            return {"forward": forward, "backward": backward}
        return run

    assert_as_exact(paired(between), paired(plain_between), draw, "the decays' tangent paired",
                    seeds=PAIRING_SEEDS)


def factored_draw(seed: int):
    """A per-key spread's operands of two sub-chunks (`recurrence_algebra.SUB`) and its exponents: a reader
    at the second's first token and a write two tokens earlier, in the first, of keys 2^100 to 2^126, under
    integer base-2 log decays of -75 to -89 a key at the two tokens between them, summing to -150 or below,
    so the pair's whole decay underflows float32 where its factors to and from the reference do not; its
    decays exact powers of two of the exponents' differences (`np.ldexp`, no `scan.between`), a future
    pair's negative zero, which a live span's underflow is not (`scan._crossed`)."""
    from dew.nn.kernels.recurrence_algebra import SUB

    rng = np.random.default_rng([9549, seed])
    chunk, width = 2 * SUB, 2
    readers, values = np.zeros((chunk, 1), np.float32), np.zeros((chunk, 1), np.float32)
    keys, exponents = np.zeros((chunk, width), np.float32), np.zeros((chunk, width), np.int32)
    readers[SUB], values[SUB - 2] = 1.0, 1.0
    keys[SUB - 2] = np.ldexp(np.float32(1.0), rng.integers(100, 127, width))
    exponents[SUB - 1:SUB + 1] = -rng.integers(75, 90, (2, width))
    total = np.cumsum(exponents, axis=0)
    lower = np.tri(chunk, dtype=bool)[..., None]
    decays = np.ldexp(np.float32(1.0), total[:, None] - total[None, :])
    between = np.where(lower, decays, -0.0).astype(np.float32)
    return readers, values, keys, between, exponents.astype(np.float32)


def token_spread(readers, values, keys, between, exponents):
    """The spread a token at a time, its state a key's: each token decays it by its own exponent's power of
    two (`jnp.ldexp`, exact) and writes its key times its value, and its reader reads it; the decays `between`
    unread, an oracle of their own."""
    del between

    def step(state, token):
        reader, value, key, exponent = token
        state = jnp.ldexp(state, exponent.astype(jnp.int32)) + key * value
        return state, reader * state

    return jax.lax.scan(step, jnp.zeros(keys.shape[1:], keys.dtype), (readers, values, keys, exponents))[1]


def test_a_per_key_spreads_factored_pair_keeps_a_result_its_whole_decay_loses():
    """`scan.spread` under a decay a key dimension, a pair across sub-chunks whose whole decay underflows
    float32 where its factors to and from the reference do not (`factored_draw`): its read through the
    factors held by check 1 against the spread a token at a time (`token_spread`), from float64, where a
    select on the whole decay read zero (head review of 4bc3f5016 F1; row 9549)."""
    def spread(readers, values, keys, between, exponents):
        del exponents
        return scan.spread(readers, values, keys, between)

    assert_as_exact(jax.jit(spread), jax.jit(token_spread), factored_draw, "a per-key spread's factored pair",
                    seeds=8)


def lone_pair(reader: float, value: float, decays, *, width: int = 2, later: int = 2, after: bool = False):
    """A per-key spread's operands of `later + 1` tokens whose one pair is the last's reader and the first's
    write, its score `reader value`, every other score zero: unit keys, the pair's decay `decays` a key, its
    factors across a sub-chunk's reference consistent with it (`recurrence_algebra.SUB`: `decays` up to the
    reference and one after, or where `after` one up to it and `decays` after), every other pair's zero,
    exact as given (no `scan.between`): `(readers, values, keys, between)`."""
    from dew.nn.kernels.recurrence_algebra import SUB

    readers, values = np.zeros((later + 1, 1), np.float32), np.zeros((later + 1, 1), np.float32)
    readers[later], values[0] = reader, value
    between = np.zeros((later + 1, later + 1, width), np.float32)
    between[later, 0] = decays
    if later >= SUB:
        reference = later // SUB * SUB - 1
        between[reference, 0], between[later, reference] = (1.0, decays) if after else (decays, 1.0)
    return readers, values, np.ones((later + 1, width), np.float32), between


@pytest.mark.parametrize("later", [2, 18], ids=["own sub-chunk", "an earlier sub-chunk"])
@pytest.mark.parametrize("read", ["overflowed", "a NaN"])
def test_a_per_key_spreads_read_through_an_underflowed_decay_is_zero(read, later):
    """`scan.spread` under a decay a key dimension, one pair of its last token's reader and its first's
    write, within a sub-chunk or across one, its decay zero at every key: a read of 2^70 times 2^70, which
    overflows, zero at every key, the rule of row 9547 (base review of ce6ad9298 F1), and a NaN read NaN."""
    reader = np.nan if read == "a NaN" else 2.0 ** 70
    found = np.asarray(scan.spread(*lone_pair(reader, 2.0 ** 70, 0.0, later=later)))
    if read == "overflowed":
        np.testing.assert_array_equal(found, np.zeros_like(found))
    else:
        assert np.all(np.isnan(found[later])), found


@pytest.mark.parametrize(("across", "after"), [(False, False), (True, False), (True, True)],
                         ids=["own sub-chunk", "an earlier sub-chunk, zero up to its reference",
                              "an earlier sub-chunk, zero after its reference"])
def test_a_per_key_spreads_overflowed_read_is_zero_at_each_key_its_decay_underflows(across, after):
    """`scan.spread` under a decay a key dimension, one pair of its last token's reader and its first's write
    whose decay is zero at its first key alone, within a sub-chunk or across one, the zero its factor's up to
    the reference or after it: a read of 2^70 times 2^70, which overflows, zero at that key and infinite at
    the other, the recurrence's own (row 9631: across a sub-chunk, NaN at the zero key)."""
    from dew.nn.kernels.recurrence_algebra import SUB

    later = SUB + 2 if across else 2
    found = np.asarray(scan.spread(*lone_pair(2.0 ** 70, 2.0 ** 70, (0.0, 1.0), later=later, after=after)))
    expected = np.zeros_like(found)
    expected[later, 1] = np.inf
    np.testing.assert_array_equal(found, expected)


def test_a_per_key_spreads_pairs_across_a_sub_chunk_are_decayed_before_they_are_summed():
    """`scan.spread` under a decay a key dimension, two writes of unit keys before a sub-chunk's reference
    (`recurrence_algebra.SUB`) read by a reader after it, each read 2^127, the keys' decay 1/4 past the
    reference: the reader's output 2^126 at each key exactly, the recurrence's (its every term normal), each
    pair's read decayed before the pairs are summed, where summed first they passed float32's largest (row
    9029's head review)."""
    from dew.nn.kernels.recurrence_algebra import SUB

    reader, reference = SUB + 1, SUB - 1
    readers, values = np.zeros((reader + 1, 1), np.float32), np.zeros((reader + 1, 1), np.float32)
    readers[reader], values[:2] = 2.0 ** 64, 2.0 ** 63
    between = np.zeros((reader + 1, reader + 1, 2), np.float32)
    between[reader, :2], between[reference, :2], between[reader, reference] = 0.25, 1.0, 0.25
    found = np.asarray(scan.spread(readers, values, np.ones((reader + 1, 2), np.float32), between))
    expected = np.zeros_like(found)
    expected[reader] = 2.0 ** 126
    np.testing.assert_array_equal(found, expected)


@pytest.mark.parametrize("batched", [False, True], ids=["one spread", "a batch of two"])
def test_a_per_key_spreads_overflowed_read_through_a_vanished_decay_has_zero_gradients(batched):
    """`scan.spread` under a decay a key dimension, one pair across a sub-chunk whose read of 2^70 times 2^70
    overflows and whose decay is zero at every key: its keys' and decays' gradients zero, as its output is,
    one spread or a batch of two (`jax.vmap`), where no infinity meets a zero factor (row 9631)."""
    from dew.nn.kernels.recurrence_algebra import SUB

    readers, values, keys, between = (jnp.asarray(x) for x in lone_pair(2.0 ** 70, 2.0 ** 70, 0.0,
                                                                          later=SUB + 2))

    def total(keys, between):
        return jnp.sum(scan.spread(readers, values, keys, between))

    pulled = jax.grad(total, argnums=(0, 1))
    if batched:
        pulled = jax.vmap(pulled)
        keys, between = (jnp.stack([x, x]) for x in (keys, between))
    for gradient in pulled(keys, between):
        np.testing.assert_array_equal(np.asarray(gradient), np.zeros(gradient.shape, np.float32))


def test_a_per_key_spreads_finite_read_through_a_small_decay_is_its_product():
    """`scan.spread` under a decay a key dimension, one pair of a sub-chunk: a read 2^-80 through a decay
    2^-80 of a key 2^80, exactly 2^-80, the key scaling the read before its decay (head review of
    04b2ec480 F1: the decay taken first rounded 2^-160 to zero)."""
    readers, values, keys, between = lone_pair(2.0 ** -80, 1.0, 2.0 ** -80, later=1)
    keys[0] = 2.0 ** 80
    found = np.asarray(scan.spread(readers, values, keys, between))
    np.testing.assert_array_equal(found[1], np.full(2, 2.0 ** -80, np.float32))


def test_a_per_key_spreads_masked_key_carries_no_derivative():
    """`scan.spread` under a decay a key dimension, a sub-chunk's own pair whose read of 2^70 times 2^70
    overflows through decays of zero at its first key and one at its second: the first key's output zero,
    its tangent along the decays zero and its decay's gradient zero under a cotangent on that key alone,
    where the rejected product's transpose formed infinity times zero (head review of 04b2ec480 F3)."""
    readers, values, keys, between = lone_pair(2.0 ** 70, 2.0 ** 70, (0.0, 1.0))

    def first_key(between):
        return scan.spread(readers, values, keys, between)[:, 0]

    out, tangent = jax.jvp(first_key, (between,), (np.ones_like(between),))
    (gradient,) = jax.vjp(first_key, between)[1](np.ones(3, np.float32))
    np.testing.assert_array_equal(np.asarray(out), np.zeros(3, np.float32))
    np.testing.assert_array_equal(np.asarray(tangent)[2], np.float32(0.0))
    np.testing.assert_array_equal(np.asarray(gradient)[..., 0], np.zeros(gradient.shape[:2], np.float32))


def test_a_per_key_paired_faint_product_keeps_value_and_gradient_within_and_across_sub_chunks():
    """`scan._paired` keeps a faint key (`2^-100`) paired with a large read (`2^100`) through a small decay
    (`2^-60`) both within a sub-chunk (`twice_scaled`) and across `SUB` (`_channel_shift`), preserves mixed
    sub-chunks without overflow/underflow, and holds `faint_read_call` on portable math (row 12888)."""
    from test_recurrence_template_stability import faint_read_call

    from dew.nn.kernels.recurrence_algebra import SUB

    # 1. Diagonal (within sub-chunk) and off-diagonal (across SUB) faint pair value and reverse gradient.
    for later in (1, SUB + 1):
        length = later + 1
        left = jnp.zeros((length, 2), jnp.float32).at[later, 0].set(np.float32(2.0 ** 100))
        right = jnp.zeros((length, 2), jnp.float32).at[0, 0].set(np.float32(2.0 ** -100))
        between = np.zeros((length, length, 2), np.float32)
        between[later, 0, 0] = np.float32(2.0 ** -60)
        if later >= SUB:
            ref = (later // SUB) * SUB - 1
            between[ref, 0, 0] = np.float32(2.0 ** -60)
            between[later, ref, 0] = np.float32(1.0)
        between_arr = jnp.asarray(between)
        out, pullback = jax.vjp(lambda l_, r_, b_=between_arr: scan._paired(l_, r_, b_), left, right)
        assert float(out[later, 0]) == float(np.float32(2.0 ** -60))
        cot = jnp.zeros_like(out).at[later, 0].set(np.float32(-(2.0 ** 60)))
        d_left, d_right = pullback(cot)
        assert float(d_right[0, 0]) == float(np.float32(-(2.0 ** 100)))
        assert float(d_left[later, 0]) == float(np.float32(-(2.0 ** -100)))

    # 2. Mixed-magnitude adversarial sub-chunk (`[2^100, 2^-100]` vs `1.0`) across `SUB`.
    n_mix = SUB + 2
    left_mix = jnp.zeros((n_mix, 2), jnp.float32).at[SUB, 0].set(np.float32(2.0 ** 100)).at[
        SUB + 1, 0].set(np.float32(2.0 ** -100))
    right_mix = jnp.zeros((n_mix, 2), jnp.float32).at[0, 0].set(np.float32(1.0))
    between_mix = jnp.ones((n_mix, n_mix, 2), jnp.float32)
    out_mix = scan._paired(left_mix, right_mix, between_mix)
    assert float(out_mix[SUB, 0]) == float(np.float32(2.0 ** 100))
    assert float(out_mix[SUB + 1, 0]) == float(np.float32(2.0 ** -100))

    # 3. Full `faint_read_call` witness on portable math (`xla`): output[1, col 0] == -1 and dk0[0] == -2^100.
    (q, k, v, r), run = faint_read_call(64, 32)
    out_faint = jax.jit(lambda k_: run("xla", q, k_, v, r))(k)
    assert float(out_faint[0, 1, 0, 0]) == -1.0
    dk_faint = jax.jit(jax.grad(lambda k_: run("xla", q, k_, v, r)[0, 1, 0, 0].astype(jnp.float32)))(k)
    assert float(dk_faint[0, 0, 0, 0]) == float(np.float32(-(2.0 ** 100)))


def test_a_balanced_pair_keeps_a_normal_factor_after_decay():
    from dew.nn.kernels.recurrence_algebra import SUB

    left, right = (np.zeros((SUB + 2, 2), np.float32) for _ in range(2))
    left[SUB:, 0] = [2.0 ** -80, 2.0 ** 80]
    right[:2, 0] = [2.0 ** -20, 2.0 ** 120]
    between = np.zeros((SUB + 2, SUB + 2, 2), np.float32)
    between[SUB:, SUB - 1, 0] = 1.0
    between[SUB - 1:, 0, 0] = 2.0 ** -106
    between[SUB - 1:, 1, 0] = 1.0
    found = scan._paired(jnp.asarray(left), jnp.asarray(right), jnp.asarray(between))
    assert float(found[SUB + 1, 0]) == 2.0 ** -46


def test_a_decayed_dot_retains_finite_read_cotangents():
    from dew.nn.kernels.recurrence_algebra import SUB

    left, right = (np.zeros((SUB + 1, 2), np.float32) for _ in range(2))
    left[SUB, 0], right[0, 0] = 2.0 ** 80, 1.0
    decay = np.ones((SUB + 1, SUB + 1, 2), np.float32)
    _, pulled = jax.vjp(lambda reads: scan._paired(reads, right, decay)[SUB, 0], jnp.asarray(left))
    (gradient,) = pulled(jnp.float32(2.0 ** 100))
    expected = np.zeros_like(left)
    expected[SUB, 0] = 2.0 ** 100
    np.testing.assert_array_equal(gradient, expected)
