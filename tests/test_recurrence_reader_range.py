"""Packed delta-rule derivatives at the exponent-edge draw of ledger 8428."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from contract_support import normal_float32, orders

from dew.nn.kernels import recurrence_chunk


@pytest.mark.parametrize("order", ("jvp", "vjp", "jvp of grad", "grad of grad"))
def test_packed_delta_derivatives_keep_representable_values(order):
    """A reset's first read keeps its representable derivative at every held order."""
    rows, length, heads, width = 1, 37, 2, 8
    scale = 2.0 ** 63
    starts_key = jax.random.split(jax.random.key(1861), 8)[5]
    starts = jax.random.bernoulli(starts_key, 0.1, (rows, length))
    keys = jax.random.split(jax.random.fold_in(jax.random.key(1861), 0), 8)
    query, key, value = (jax.random.normal(k, (rows, length, heads, width)) * scale for k in keys[:3])
    g = -jax.nn.softplus(jax.random.normal(keys[3], (rows, length, heads, 1)))
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (rows, length, heads)))
    primals = (query, key, value, g, beta)
    direction = tuple(jax.random.normal(keys[6], x.shape) * (scale if at < 3 else 1.0)
                      for at, x in enumerate(primals))
    cotangent = jax.random.normal(keys[7], query.shape)

    def rule(query, key, value, g, beta):
        return recurrence_chunk.xla_mainloop(
            recurrence_chunk.Tiles(16), query, key, value, g[..., 0], beta, None, starts,
            scale=width ** -0.5, delta=True, dtype=query.dtype)[0]

    narrow = jax.tree.map(np.asarray, jax.jit(orders(rule, primals, direction, cotangent)[order])())
    with jax.enable_x64(True):
        wide = jax.tree.map(lambda x: x.astype(jnp.float64), (primals, direction, cotangent))
        truth = jax.tree.map(np.asarray, jax.jit(orders(rule, *wide)[order])())
    for index, (made, exact) in enumerate(zip(jax.tree.leaves(narrow), jax.tree.leaves(truth), strict=True)):
        held = normal_float32(exact) & np.isfinite(exact)
        assert np.all(np.isfinite(made[held])), (order, index, np.argwhere(held & ~np.isfinite(made)))
