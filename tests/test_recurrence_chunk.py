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
         for decay, delta, skip in itertools.product(('none', 'scalar', 'vector'), (True, False),
                                                     (False, True))]
"""The decay kinds, the delta rule or a write gate alone (R = 0, Mamba2's SSD and the gated linear
attentions), with and without the skip."""

FORMED = [
    recurrence.Hooks(decay='scalar', gates=recurrence.Gates(), l2norm=True),
    recurrence.Hooks(decay='scalar', gates=recurrence.Gates(write='step', limit=(0.0, 0.5)), delta=False,
                     skip=True),
    recurrence.Hooks(decay='vector', l2norm=True, output=recurrence.OutputNorm(1e-6)),
    recurrence.Hooks(decay='scalar', scale=KEY_WIDTH ** -0.5,
                     output=recurrence.OutputNorm(1e-5, 'sigmoid')),
    recurrence.Hooks(decay='scalar', rated=True, delta=False, skip=True),
]
"""The prologue's and epilogue's hooks: the gated delta rule's gates and l2 norm, Mamba2's step gates
clipped with the skip, the gated RMS norm after the l2 norm and after a plain scale, and Mamba2's SSD
as `chunk_ssd` hands it over (a rated decay, the step its write gate)."""

PACKED = (False, True)
"""A row of one sequence, and of packed documents."""

SEEDS = dict(zip(((hooks, packed) for hooks in HOOKS + FORMED for packed in PACKED), (
    8, 8, 8, 8, 8, 8, 8, 8, 16, 16, 64, 64, 16, 16, 16, 16, 32, 32, 64, 64, 8, 8, 8, 8,
    128, 64, 128, 128, 8, 8, 16, 16, 64, 64), strict=True))
"""Each site's declared seeds (hooks in `HOOKS` then `FORMED` order, one sequence then packed), sized
independently by `seeds_for` on armada (2026-10-21). The vector-decay delta rule's two take 32 where its
control resolved at 8: the chunked form's own spread left the pooled ratio 1.315 inconclusive over 8
seeds ([1.046, 1.702])."""


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
    gates = {}
    if hooks.gates is not None:
        # Pre-activations: the gate and beta raw, `log_a` and `bias` a head's.
        heads = jax.random.split(keys[6], 3)
        gates = {'gate': jax.random.normal(heads[0], (ROWS, LENGTH, VALUE_HEADS)),
                 'log_a': 0.5 * jax.random.normal(heads[1], (VALUE_HEADS,)),
                 'bias': jax.random.normal(heads[2], (VALUE_HEADS,))}
        beta = jax.random.normal(keys[4], (ROWS, LENGTH, VALUE_HEADS))
    if hooks.rated:
        # The step through softplus is the write gate; the rate is `A = -exp(A_log)`, a head's.
        beta = jax.nn.softplus(jax.random.normal(keys[4], (ROWS, LENGTH, VALUE_HEADS)) - 1.0)
        gates = {'rate': -jnp.exp(0.5 * jax.random.normal(keys[6], (VALUE_HEADS,)))}
    epilogue = {}
    if hooks.output is not None:
        norms = jax.random.split(keys[7], 2)
        epilogue = {'z': jnp.take(jax.random.normal(norms[0], (ROWS, LENGTH, VALUE_HEADS, VALUE_WIDTH)),
                                  value_order, axis=-1),
                    'weight': jnp.take(1.0 + 0.1 * jax.random.normal(norms[1], (VALUE_WIDTH,)), value_order)}
    query, key = (jnp.take(x, key_order, axis=-1) for x in (query, key))
    value = jnp.take(value, value_order, axis=-1)
    if hooks.decay == 'vector':
        log_decay = jnp.take(log_decay, key_order, axis=-1)
    starts = None
    if packed:
        starts = jnp.zeros((ROWS, LENGTH), bool).at[0, jnp.array([0, 23, 40])].set(True).at[1, 9].set(True)
    reads_beta = hooks.gated_write if hooks.gates is None else hooks.gates.write == 'sigmoid'
    plain = hooks.decay != 'none' and hooks.gates is None and not hooks.rated
    return (Chunk(query, key, value, log_decay=log_decay if plain else None,
                  beta=beta if reads_beta else None, skip=skip if hooks.skip else None, starts=starts,
                  **gates, **epilogue),)


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
        def at(x):
            return None if x is None else x[:, t]

        token = recurrence.Step(
            state, query=query[:, t], key=key[:, t], value=operands.value[:, t],
            log_decay=at(operands.log_decay), gate=at(operands.gate), log_a=operands.log_a,
            bias=operands.bias, beta=at(operands.beta), skip=operands.skip, z=at(operands.z),
            weight=operands.weight, rate=operands.rate)
        out, _, state = recurrence.step_reference(hooks, token)
        return state, out

    state, out = jax.lax.scan(one_token, state, jnp.arange(LENGTH))
    return jnp.moveaxis(out, 0, 1), state


@pytest.mark.parametrize('packed', PACKED, ids=['one sequence', 'packed'])
@pytest.mark.parametrize('hooks', HOOKS + FORMED, ids=str)
def test_the_chunked_op_is_as_exact_as_the_recurrence_token_by_token(hooks, packed):
    """`chunk`'s portable math in `CHUNK`-token chunks against the same recurrence a token at a time,
    output and final memory, over draws of fresh operands under orders of the keys' and values'
    dimensions, both from float64."""
    dew = jax.jit(functools.partial(chunk, hooks, chunk_size=CHUNK, ported=False))
    reference = jax.jit(functools.partial(token_by_token, hooks))
    assert_as_exact(dew, reference, functools.partial(draw, hooks, packed), f'{hooks} packed={packed}',
                    seeds=SEEDS[hooks, packed])


def test_the_output_rounds_once_after_the_epilogue():
    """The chunked op rounds its output once, after the skip and the norm, as the decode step's reference
    does: a bf16 token whose readout `0.1` the skip `-0.10009765625` nearly cancels leaves the residual
    the reference leaves, where a readout rounded first cancels to zero; an explicit `dtype` holds with
    and without an epilogue, and bf16 operands under the l2 norm come back in bf16."""
    ones = jnp.ones((1, 1, 1, 8), jnp.bfloat16)
    hooks = recurrence.Hooks(decay='none', delta=False, write_gate=True, skip=True)
    operands = Chunk(ones, ones, ones, beta=jnp.full((1, 1, 1), 0.1, jnp.float32),
                     skip=jnp.full((1,), -0.10009765625, jnp.float32))
    found, _ = chunk(hooks, operands, ported=False, dtype=jnp.float32)
    want, _, _ = recurrence.step_reference(hooks, recurrence.Step(
        jnp.zeros((1, 1, 8, 8), jnp.float32), query=ones[:, 0], key=ones[:, 0], value=ones[:, 0],
        beta=operands.beta[:, 0], skip=operands.skip), dtype=jnp.float32)
    np.testing.assert_array_equal(np.asarray(found[:, 0]), np.asarray(want))
    assert np.all(np.asarray(found) != 0.0)
    plain = recurrence.Hooks(decay='scalar', scale=0.25)
    wide = Chunk(*(x.astype(jnp.float32) for x in (ones, ones, ones)), log_decay=jnp.zeros((1, 1, 1)),
                 beta=jnp.ones((1, 1, 1)))
    assert chunk(plain, wide, ported=False, dtype=jnp.bfloat16)[0].dtype == jnp.bfloat16
    normed = recurrence.Hooks(decay='scalar', l2norm=True)
    assert chunk(normed, Chunk(ones, ones, ones, log_decay=jnp.zeros((1, 1, 1)), beta=jnp.ones((1, 1, 1))),
                 ported=False)[0].dtype == jnp.bfloat16


def test_a_rated_decay_is_its_log_decay_bit_for_bit():
    """A rated decay (`rate` times the write gate) steps a decode token and a chunked sequence bit for bit
    as the same log decay given whole, on the launch and on the portable math."""
    rng = np.random.default_rng(0)
    rows, heads, width = 2, 2, 16
    state = jnp.asarray(rng.standard_normal((rows, heads, width, width)), jnp.float32)
    token = {name: jnp.asarray(rng.standard_normal((rows, heads, width)), jnp.float32)
             for name in ('query', 'key', 'value')}
    beta = jnp.asarray(rng.uniform(0.0, 0.5, (rows, heads)), jnp.float32)
    rate = -jnp.asarray(rng.uniform(0.1, 2.0, (heads,)), jnp.float32)
    rated = recurrence.Hooks(decay='scalar', rated=True, delta=False, skip=True)
    whole = recurrence.Hooks(decay='scalar', delta=False, write_gate=True, skip=True)
    skip = jnp.asarray(rng.standard_normal((heads,)), jnp.float32)
    for step in (recurrence.step, recurrence.step_reference):
        found = step(rated, recurrence.Step(state, **token, beta=beta, rate=rate, skip=skip))
        want = step(whole, recurrence.Step(state, **token, beta=beta, log_decay=rate[None, :] * beta,
                                           skip=skip))
        for mine, theirs in zip(found[::2], want[::2], strict=True):
            np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs))
    sequence = {name: jnp.asarray(rng.standard_normal((rows, LENGTH, heads, width)), jnp.float32)
                for name in ('query', 'key', 'value')}
    steps = jnp.asarray(rng.uniform(0.0, 0.5, (rows, LENGTH, heads)), jnp.float32)
    found = chunk(rated, Chunk(**sequence, beta=steps, rate=rate, skip=skip), chunk_size=CHUNK, ported=False)
    want = chunk(whole, Chunk(**sequence, beta=steps, log_decay=rate[None, None, :] * steps, skip=skip),
                 chunk_size=CHUNK, ported=False)
    for mine, theirs in zip(found, want, strict=True):
        np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs))
