"""Dev-time component cases for the written classifier-free guidance surrogate."""

import jax
import jax.numpy as jnp
import numpy as np

from dew.algebra.generate import Cases, ComponentForm, SurrogateBodies, SurrogateCase
from dew.sampling import guidance


def _flat_tangent(c, u, scale, dc, du, ds, value):
    return guidance._guided_derivative((c, u, scale), (dc, du, ds), value)


def _value_truth(c, u, scale):
    return jnp.where(scale == 1, c, jnp.where(scale == 0, u, (1 - scale) * u + scale * c))


def _tangent_truth(c, u, scale, dc, du, ds, value):
    return (1 - scale) * du + scale * dc + (c - u) * ds


def _draw(seed, operands):
    rng = np.random.default_rng(seed)
    c, u, scale, *directions = (rng.standard_normal(arg.shape).astype(arg.dtype) for arg in operands)
    scale[:] = np.clip(scale, -0.75, 1.75)
    scale[:3, 0] = [0, 1, 0.25]
    u[0, 0], c[1, 0] = -0.0, -0.0
    if seed % 2:
        c[2, :4], u[2, :4] = 2.0 ** 127, -2.0 ** 127
        if directions:
            directions[2][2, 0] = 0.25
            # Equal large directions can give a finite tangent even outside [0, 1].
            scale[3, 0] = 4
            directions[0][3, :4] = directions[1][3, :4] = np.float32(1e38)
            directions[2][3, 0] = 0
    return c, u, scale, *directions


def _cases(*, tangent):
    operands = []
    for rows, width in ((16, 64), (32, 128)):
        tile = jax.ShapeDtypeStruct((rows, width), jnp.float32)
        scale = jax.ShapeDtypeStruct((rows, 1), jnp.float32)
        operands.append((tile, tile, scale, tile, tile, scale, tile) if tangent else (tile, tile, scale))
    return Cases(tuple(operands), _draw, seeds=8, policies=("default",))


GUIDED = SurrogateBodies(guidance._guided_value, guidance._guided_derivative, (),
                         symbolic_zeros=False, cases=(
    ((), SurrogateCase("guided", ComponentForm(guidance._guided_value, _value_truth, _cases(tangent=False)),
                       {(): ComponentForm(_flat_tangent, _tangent_truth, _cases(tangent=True))},
                       output_tree=(jax.tree.structure(0),))),
)).declared(__name__)
