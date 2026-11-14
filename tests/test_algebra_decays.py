"""The decays a chunk of a linear recurrence reads (`dew.algebra.scan`'s `_Decays`): their forward
derivatives, first and second, each on par with the product of each span's own sums (row 5858)."""

import jax
import jax.numpy as jnp
import numpy as np
from reference_error import assert_as_exact

from dew.algebra import scan

FIRST_TANGENT_SEEDS = 8
"""The decays' first forward derivative's declared draws against the explicit spans' (check 1)."""


def test_the_decays_take_their_tangent_from_each_spans_own_sums():
    """A chunk's decays between positions (`_Decays.between`), each key's over 64 tokens of log decays in
    [-3, -1], differentiated forward along a drawn direction `u`: `between * U`, `U` the direction summed
    over each pair's span, `j < s <= i`. Dew's forward derivative itself, no pairing summed after it, is on
    par with that product of each span's own sum (over the span's mask) from it in float64 (check 1, two
    programs of Dew's)."""
    tokens, keys = 64, 16
    within = jnp.asarray(np.arange(tokens)[None, :, None] < np.arange(tokens)[None, None, :])
    spanned = within & jnp.asarray(np.arange(tokens)[None, None, :] <= np.arange(tokens)[:, None, None])

    def between(g):
        return scan._decays_of(g, base2=False).between

    def draw(seed):
        rng = np.random.default_rng([0x585A, seed])
        return tuple(jnp.asarray(x, jnp.float32) for x in (
            rng.uniform(-3, -1, (tokens, keys)), rng.standard_normal((tokens, keys))))

    def tangent(g, u):
        return jax.jvp(between, (g,), (u,))[1]

    def spans(g, u):
        return between(g) * jnp.einsum("ijs,sk->ijk", spanned.astype(u.dtype), u)

    assert_as_exact(tangent, spans, draw, "the decays' tangent", seeds=FIRST_TANGENT_SEEDS)


SECOND_TANGENT_SEEDS = 8
"""The decays' mixed second forward derivative's declared draws against the explicit spans' (check 1), sized
by `seeds_for` on armada."""


def test_the_decays_take_their_second_tangent_from_each_spans_own_sums():
    """A chunk's decays between positions (`_Decays.between`), each key's over 64 tokens of log decays in
    [-3, -1], differentiated forward twice along two drawn directions `u` and `w`: `between * U * W`, where
    `U` and `W` sum each direction over a pair's span, `j < s <= i`. Dew's second forward derivative is on
    par with that product of each span's own sum (formed over the span's mask, no difference of prefixes)
    from it in float64 (check 1, two programs of Dew's)."""
    tokens, keys = 64, 16
    within = jnp.asarray(np.arange(tokens)[None, :, None] < np.arange(tokens)[None, None, :])
    spanned = within & jnp.asarray(np.arange(tokens)[None, None, :] <= np.arange(tokens)[:, None, None])

    def between(g):
        return scan._decays_of(g, base2=False).between

    def draw(seed):
        rng = np.random.default_rng([0x5859, seed])
        return tuple(jnp.asarray(x, jnp.float32) for x in (
            rng.uniform(-3, -1, (tokens, keys)), rng.standard_normal((tokens, keys)),
            rng.standard_normal((tokens, keys))))

    def second(g, u, w):
        return jax.jvp(lambda g: jax.jvp(between, (g,), (u,))[1], (g,), (w,))[1]

    def spans(g, u, w):
        def span(direction):
            return jnp.einsum("ijs,sk->ijk", spanned.astype(direction.dtype), direction)
        return between(g) * span(u) * span(w)

    assert_as_exact(second, spans, draw, "the decays' second tangent", seeds=SECOND_TANGENT_SEEDS)
