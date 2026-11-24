"""A forked call's walk (`recurrence_forked.forked_steps`, `Chunk.loads`) at the documents' bound its table
gives: the walk the same whatever the bound past its row's documents (row 7952)."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_recurrence_chunk import (
    ADDITIVE,
    CHUNK,
    HOOKS,
    chunk_on,
    entered_draw,
    entering_documents,
    transitioned,
)


@pytest.mark.parametrize('hooks', [HOOKS[4], ADDITIVE], ids=['the delta rule', 'an additive erase'])
def test_a_forked_walk_is_the_same_whatever_its_documents_bound(hooks):
    """A forked call (`Chunk.loads`) on the portable math with its table's documents bound at its own count
    and at its row's every token, the documents past its row taking no slot, storing none and writing no
    entry: its output and last memory bit for bit the same, the walk (`recurrence_forked.forked_steps`)
    stopping at the row's last segment at either bound (row 7952: a scan over the bound evaluated the chunk
    forms at every place past them, a 2048-token row of six documents 2080 windows against 38)."""
    operands = entering_documents(transitioned(hooks, entered_draw(hooks, 0, entries=True)[0], 0))
    extra = operands.query.shape[1] - operands.loads.shape[1]

    def bounded(x):
        """`x` `[B, D, ...]` with `extra` documents past its row, zero."""
        return jnp.pad(x, ((0, 0), (0, extra)) + ((0, 0),) * (x.ndim - 2))

    padded = dataclasses.replace(operands, **{name: bounded(getattr(operands, name)) for name in (
        'loads', 'stores', 'entry_key', 'entry_value', 'entry_weight')})
    run = jax.jit(functools.partial(chunk_on, 'xla', hooks, chunk_size=CHUNK))
    for found, expected in zip(jax.tree.leaves(run(padded)), jax.tree.leaves(run(operands)), strict=True):
        np.testing.assert_array_equal(np.asarray(found), np.asarray(expected), strict=True)
