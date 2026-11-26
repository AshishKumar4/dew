"""A forked call's walk (`recurrence_forked.forked_steps`, `Chunk.loads`) at the documents' bound its table
gives: the walk the same whatever the bound past its row's documents (row 7952); and a forked call that reads
the memory entering its token or weighs its own write (RWKV6's, the delta rule's read before its update),
each document from its table slot's memory, on par with each document alone (row 6267)."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_recurrence_chunk import (
    ADDITIVE,
    CHUNK,
    FORMED,
    HOOKS,
    assert_forked_on_par,
    chunk_on,
    entered_draw,
    entering_documents,
    narrowed_draw,
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


ENTERING_READOUTS = [FORMED[9], FORMED[10], FORMED[11]]
"""The readouts a document's entering memory reaches at its first token: RWKV6's, a decay a key dimension
read before its update with its own write weighed (`Hooks.reads` 'before', `Hooks.own_weight`), the delta
rule's read before its update, and a write gate's own write weighed in a readout after it."""


@pytest.mark.check1_shards(seeds=8, per_task=1)
@pytest.mark.parametrize('hooks', [FORMED[10]], ids=str)
def test_a_forked_read_before_through_the_template_is_as_exact_as_each_document_alone(hooks):
    """A forked call whose readout reaches each document's entering memory at its first token, the delta
    rule's read before its update (of `ENTERING_READOUTS` the one the template's fast backward takes: a
    decay a key dimension without the delta rule it refuses), in 16 bits through the template's route
    (`chunk_on('mosaic_gpu')`), against each document alone through it, interpreted off a GPU (check 1; the
    portable math's walk of the same calls is
    `test_documents_entering_memories_of_their_own_are_as_exact_as_each_alone`'s): today its kernels hand
    such a readout over to the portable math (`recurrence_chunk.READOUT_HANDOFF`), so this holds the handoff
    of a forked call; it holds the template's own walks once they read each document's entering memory
    beside their table (row 6267)."""
    assert_forked_on_par('mosaic_gpu', hooks, functools.partial(narrowed_draw, hooks),
                         f'mosaic_gpu {hooks} a forked call\'s documents, against each alone', seeds=8,
                         alone=True)
