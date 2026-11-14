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
    spanning = jnp.asarray((np.arange(TOKENS)[None, :] >= np.arange(TOKENS)[:, None])[:, :, None]
                           & (np.arange(TOKENS)[:, None] > np.arange(TOKENS)[None, :])[:, None, :])

    def gradient(g, cotangent):
        return jax.vjp(between, g)[1](cotangent)[0]

    def pairs(g, cotangent):
        return jnp.einsum("pij,ijk->pk", spanning.astype(g.dtype), cotangent * spans_decays(g))

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
