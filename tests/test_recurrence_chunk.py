"""The recurrence template's chunked op (`dew.nn.kernels.recurrence_chunk.chunk`) on its portable math:
every hook combination it takes, a sequence over several chunks and a ragged last one, packed documents
too, held to the same recurrence run a token at a time (the decode step's reference,
`recurrence.step_reference`) under bible §7's gate, both from float64."""

import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact

from dew.nn.kernels import recurrence
from dew.nn.kernels.recurrence_chunk import Chunk, chunk

ROWS, LENGTH, KEY_HEADS, VALUE_HEADS, KEY_WIDTH, VALUE_WIDTH = 2, 70, 1, 2, 16, 8
CHUNK = 16
"""Tokens a chunk: 70 tokens are four whole chunks and a ragged six."""

HOOKS = [recurrence.Hooks(decay=decay, delta=delta, write_gate=not delta, skip=skip,
                          scale=KEY_WIDTH ** -0.5 if delta else None)
         for decay, delta, skip in itertools.product(('none', 'scalar', 'vector'), (True, False), (False, True))]
"""The decay kinds, the delta rule or a write gate alone (R = 0, Mamba2's SSD and the gated linear
attentions), with and without the skip."""

PACKED = (False, True)
"""A row of one sequence, and of packed documents."""

SEEDS = {(hooks, packed): 32 for hooks in HOOKS for packed in PACKED}
"""Each site's declared seeds."""


def draw(hooks: recurrence.Hooks, packed: bool, seed: int) -> tuple[Chunk]:
    """A sequence's operands for `hooks` from `seed`, its keys' and values' dimensions each permuted
    by an order the seed draws (an exact symmetry of every head's recurrence): fresh queries, keys
    and values, a log decay a head or a key dimension in (-1, 0), a gate in (0, 1), a skip, and with
    `packed` documents starting at tokens 0, 23 and 40 of the first row and 9 of the second."""
    keys = jax.random.split(jax.random.key(seed), 8)
    rng = np.random.default_rng(seed)
    key_order, value_order = rng.permutation(KEY_WIDTH), rng.permutation(VALUE_WIDTH)
    query = jax.random.normal(keys[0], (ROWS, LENGTH, KEY_HEADS, KEY_WIDTH))
    key = jax.random.normal(keys[1], (ROWS, LENGTH, KEY_HEADS, KEY_WIDTH))
    key = key / jnp.linalg.norm(key, axis=-1, keepdims=True)
    value = jax.random.normal(keys[2], (ROWS, LENGTH, VALUE_HEADS, VALUE_WIDTH))
    shape = (ROWS, LENGTH, VALUE_HEADS, KEY_WIDTH) if hooks.decay == 'vector' else (ROWS, LENGTH, VALUE_HEADS)
    log_decay = -jax.nn.softplus(jax.random.normal(keys[3], shape) - 1.0)
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (ROWS, LENGTH, VALUE_HEADS)))
    skip = jax.random.normal(keys[5], (VALUE_HEADS,))
    query, key = (jnp.take(x, key_order, axis=-1) for x in (query, key))
    value = jnp.take(value, value_order, axis=-1)
    if hooks.decay == 'vector':
        log_decay = jnp.take(log_decay, key_order, axis=-1)
    starts = None
    if packed:
        starts = jnp.zeros((ROWS, LENGTH), bool).at[0, jnp.array([0, 23, 40])].set(True).at[1, 9].set(True)
    return (Chunk(query, key, value, log_decay=None if hooks.decay == 'none' else log_decay,
                  beta=beta if hooks.gated_write else None, skip=skip if hooks.skip else None,
                  starts=starts),)


def token_by_token(hooks: recurrence.Hooks, operands: Chunk):
    """The sequence a token at a time through the decode step's reference, a packed document's first
    token starting from a zero memory: `(output, final memory)`, the output in the operands' dtype."""
    group = VALUE_HEADS // KEY_HEADS
    query, key = (jnp.repeat(x, group, axis=2) for x in (operands.query, operands.key))
    work = jnp.promote_types(operands.value.dtype, jnp.float32)
    state = jnp.zeros((ROWS, VALUE_HEADS, KEY_WIDTH, VALUE_WIDTH), work)
    starts = jnp.zeros((ROWS, LENGTH), bool) if operands.starts is None else operands.starts

    def one_token(state, t):
        state = jnp.where(starts[:, t, None, None, None], 0.0, state)
        token = recurrence.Step(
            state, query=query[:, t], key=key[:, t], value=operands.value[:, t],
            log_decay=None if operands.log_decay is None else operands.log_decay[:, t],
            beta=None if operands.beta is None else operands.beta[:, t], skip=operands.skip)
        out, _, state = recurrence.step_reference(hooks, token)
        return state, out

    state, out = jax.lax.scan(one_token, state, jnp.arange(LENGTH))
    return jnp.moveaxis(out, 0, 1), state


@pytest.mark.parametrize('packed', PACKED, ids=['one sequence', 'packed'])
@pytest.mark.parametrize('hooks', HOOKS, ids=str)
def test_the_chunked_op_is_as_exact_as_the_recurrence_token_by_token(hooks, packed):
    """`chunk`'s portable math in `CHUNK`-token chunks against the same recurrence a token at a time,
    output and final memory, over draws of fresh operands under orders of the keys' and values'
    dimensions, both from float64."""
    dew = jax.jit(functools.partial(chunk, hooks, chunk_size=CHUNK, ported=False))
    reference = jax.jit(functools.partial(token_by_token, hooks))
    assert_as_exact(dew, reference, functools.partial(draw, hooks, packed), f'{hooks} packed={packed}',
                    seeds=SEEDS[hooks, packed])
