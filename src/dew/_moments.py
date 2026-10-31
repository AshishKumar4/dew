"""An optimizer moment's bias correction formed in double (`bias_corrected`), which `dew.optim`'s Adam and
`dew.laprop`'s LaProp both divide by."""

import math

import jax
import jax.numpy as jnp
import optax


def bias_corrected(moment, decay, count):
    """`optax.tree.bias_correction`, with the correction of a Python `decay` in (0, 1) formed as
    -expm1(count log decay), its log taken in double before the trace.

    Optax forms 1 - decay**count inside a jit, where a Python decay is already rounded to float32:
    0.999 becomes 0.99900001, so the correction at count 1 is 1.3e-5 off (2.2e-4 at 0.9999), where
    torch forms it in double. A decay given as an array, a schedule's (`scheduled_solver`), takes
    optax's own. google-deepmind/optax#1827 makes this optax's: once a release carries it, delete this and
    `dew.optim.scale_by_adam` (optax's own then), and have `dew.laprop.scale_by_laprop`, which optax has no
    form of, divide by `optax.tree.bias_correction`.
    """
    if isinstance(decay, (int, float)) and 0 < decay < 1:
        correction = -jnp.expm1(count * math.log(decay))
        return jax.tree.map(lambda leaf: leaf / correction.astype(leaf.dtype), moment)
    return optax.tree.bias_correction(moment, decay, count)
