"""The draw from a distribution of scores, refused where a row holds none.

`wellformed` says which rows of scores hold a distribution, and `draw`
draws from them, checking (`checkify.debug_check`) that every row it draws for
holds one: a NaN, a `+inf` beside a finite score, or a row with nothing finite
makes the softmax undefined and the draw an arbitrary token. A device loop
cannot raise, so Dew's programs that draw (text generation, masked and canvas
diffusion) run under `checkify.checkify` with the user checks and throw the
error on the host; a caller's own program that draws without that is not
refused, and the check costs it nothing. Text sampling, masked diffusion and
canvas diffusion all draw through here, and none of them imports another's
strategy to do it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import checkify


def wellformed(scores: jax.Array) -> jax.Array:
    """Which rows hold a distribution: every score finite or `-inf`, one finite.

    A NaN or a `+inf` beside a finite score makes the softmax undefined and
    the draw arbitrary; a row with nothing finite has no distribution at all.
    """
    return (jnp.all(jnp.isfinite(scores) | jnp.isneginf(scores), axis=-1)
            & jnp.any(jnp.isfinite(scores), axis=-1))


def draw(key: jax.Array, scores: jax.Array, where: jax.Array | None = None,
         message: str = "a row drawn from holds no distribution") -> jax.Array:
    """One draw from each row of `scores` `[..., V]` under `key`, as `jax.random.categorical` draws
    it, int32. Every row `where` marks (each row when None) must hold a distribution
    (`wellformed`), checked with `message` where the caller's program is checked; a row it leaves
    out is drawn and not checked, its draw the caller's to discard."""
    held = wellformed(scores)
    checkify.debug_check(jnp.all(held if where is None else held | ~where), message)
    return jax.random.categorical(key, scores, axis=-1).astype(jnp.int32)


def row_draw(keys: jax.Array, scores: jax.Array, where: jax.Array | None = None,
             message: str = "a row drawn from holds no distribution") -> jax.Array:
    """`draw` with row `i` of `scores` `[B, ..., V]` drawn under `keys[i]` alone, as the row
    drawn by itself would be."""
    held = wellformed(scores)
    checkify.debug_check(jnp.all(held if where is None else held | ~where), message)
    return jax.vmap(jax.random.categorical)(keys, scores).astype(jnp.int32)
