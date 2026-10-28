"""Indexing along a sequence: a row's entries at its own positions (`gather_positions`), and the index a
scatter with `mode='drop'` gives the entries it must not write (`DROPPED`).

JAX drops an index at or past its axis. XLA's deterministic scatter on GPU
(`--xla_gpu_deterministic_ops`, which the CUDA test lane sets) linearizes a
multi-dimensional index row-major and drops only a position past the whole
index space, so a column index equal to its axis size lands on the next row's
first element and can overwrite that row's own write there (jax 0.11.2, XLA
at jaxlib 0.11.2; openxla/xla#49380, fixed by openxla/xla#49498, which no jaxlib release
carries yet: ledger 951). A dropped index lands only where its row-major position is
another update's of the same scatter, and corrupts that one: a set may take the dropped
value and a float add adds it, while with no other update there it is dropped (measured
on an A100 with jaxlib 0.11.2: tests/test_scatter.py fails with the axis size). So it
reaches a scatter over rows that can write the next row's first slot, as sampling's
counts and the decision head's score layout can; the KV cache's window writes and
sparse selection's three-index mask measured unreached. An index of `DROPPED` lies past
the whole space of any scatter indexing fewer than 2**30 positions, so it is dropped on
every backend. The paged pool's write into heads it leaves implicit is a second case,
which `DROPPED` alone does not cover: `KVStore.write_tokens` maps it over its group.
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
