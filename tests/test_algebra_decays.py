"""The decays a chunk of a linear recurrence reads (`dew.algebra.scan`'s `_Decays`): the decays themselves,
their gradient and their forward derivatives, first and second, each on par with each span's own sums
(row 5858): a pair's exponent summed over its span alone, where a difference of prefix sums carries the
rounding of the whole accumulated magnitude."""

import jax
import jax.numpy as jnp
import numpy as np
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
    integer base-2 log decays of -60 to -89 a key at the two tokens between them, so the pair's whole decay
    underflows float32 where its factors to and from the reference do not; its decays exact powers of two
    of the exponents' differences (`np.ldexp`, no `scan.between`)."""
    from dew.nn.kernels.recurrence_algebra import SUB

    rng = np.random.default_rng([9549, seed])
    chunk, width = 2 * SUB, 2
    readers, values = np.zeros((chunk, 1), np.float32), np.zeros((chunk, 1), np.float32)
    keys, exponents = np.zeros((chunk, width), np.float32), np.zeros((chunk, width), np.int32)
    readers[SUB], values[SUB - 2] = 1.0, 1.0
    keys[SUB - 2] = np.ldexp(np.float32(1.0), rng.integers(100, 127, width))
    exponents[SUB - 1:SUB + 1] = -rng.integers(60, 90, (2, width))
    total = np.cumsum(exponents, axis=0)
    lower = np.tri(chunk, dtype=bool)[..., None]
    decays = np.ldexp(np.float32(1.0), total[:, None] - total[None, :])
    between = np.where(lower, decays, 0.0).astype(np.float32)
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
