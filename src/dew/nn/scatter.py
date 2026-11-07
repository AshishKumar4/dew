"""Indexing along a sequence: a row's entries at its own positions (`gather_positions`), and the index a
scatter with `mode='drop'` gives the entries it must not write (`DROPPED`).

JAX drops an index at or past its axis, on every backend: XLA's deterministic GPU scatter
(`--xla_gpu_deterministic_ops`, the CUDA test lane's) too since openxla/xla#49498 (in the XLA jaxlib
0.12.0.1 builds from; ledger 951). An index of `DROPPED` lies past any axis a scatter indexes, a position
never written whatever the axis's size.
"""

import jax
import jax.numpy as jnp

__all__ = [
    "DROPPED",
    "gather_positions",
]

DROPPED = 2**30


def gather_positions(values: jax.Array, positions: jax.Array) -> jax.Array:
    """`values` `[B, L, ...]` at each row's own `positions` `[B, K]`: `[B, K, ...]`, the entry at
    `values[b, positions[b, k]]` (a token's state at its cue, the tokens a router keeps, a sequence in
    another order)."""
    return jnp.take_along_axis(values, positions.reshape(*positions.shape, *(1,) * (values.ndim - 2)), axis=1)
