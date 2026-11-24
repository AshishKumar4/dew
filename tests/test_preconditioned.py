"""The preconditioned delta rules (`dew.nn.kernels.preconditioned`, rows 1458 and 1459: fla's PGDN and PKDA)
on the chunked op: output, final state, final ATK moment and every operand's gradient on par with fla's
recurrence a token at a time (`naive_recurrent_precond_gated_delta_rule`, its PKDA twin) from float64 (check
1), one sequence and packed documents, a decay a head's and a key dimension's, value heads sharing their key
head's moment, in fp32. The centers held at fla's default: their gradient, a key head's sum over every token
and key, measured 1.4 to 1.6 times the fp32 token walk's distance in a typical draw (inconclusive over 80,
where its pooled absolute RMS was 0.85 of the walk's), and a 16-bit call's fla's own roundings, both are
held to fla's chunked release (row 1458's recorded arm), not to a walk no release runs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact, with_gradients
from test_recurrence_chunk import assert_packed_on_par, hosted, sigmoid, softplus

from dew.nn.kernels.preconditioned import CENTER, preconditioned_delta_rule, preconditioned_step
from dew.nn.precision import l2norm

ROWS, LENGTH, KEY_HEADS, VALUE_HEADS, KEYS, VALUES, CHUNK = 2, 40, 2, 4, 16, 8, 16
"""Two whole chunks of 16 tokens and a part one, each pair of value heads sharing a key head's moment."""
DOCUMENTS = ((0, 11, 30), (5, 17))
"""Each row's packed documents' first tokens."""
SEEDS = 16
"""Each site's declared seeds."""
CENTERS = np.full((KEY_HEADS,), CENTER, np.float32)
"""The key heads' centers, fla's default, held (the module's account)."""


def tokenwise(query, key, value, log_decay, beta, atk_decay, atk_beta, state, atk_state, starts=None):
    """fla's recurrence a token at a time (naive_recurrent_precond_gated_delta_rule, its PKDA twin) in at
    least fp32, the queries and keys l2-normalized (`precision.l2norm`, as Dew's), each key head's moment its
    value heads', a packed document's first token entering both states at zero: the output in the values'
    dtype and `(final state, final moment)`."""
    work = jnp.promote_types(jnp.result_type(query, key, value), jnp.float32)
    group = value.shape[2] // key.shape[2]
    query, key = (jnp.repeat(l2norm(x).astype(work), group, axis=2) for x in (query, key))
    atk_decay, atk_beta = (jnp.repeat(jnp.asarray(x, work), group, axis=2) for x in (atk_decay, atk_beta))
    centers = jnp.repeat(jnp.asarray(CENTERS, work), group)[:, None]
    first = jnp.zeros(query.shape[:2], bool) if starts is None else jnp.asarray(starts)
    per_key = log_decay.ndim == 4

    def one(carried, token):
        memory, moment = carried
        q, k, v, g, b, ga, ba, fresh = token
        memory, moment = (jnp.where(fresh.reshape(-1, *[1] * (x.ndim - 1)), 0.0, x)
                          for x in (memory, moment))
        moment = jnp.exp(ga)[..., None] * moment + ba[..., None] * k * k
        r = jnp.log(moment + 1e-6) - centers
        written = k * jnp.exp(-np.log(1.5) * r / (1.0 + jnp.abs(r)))
        memory = memory * jnp.exp(g)[..., :, None] if per_key else memory * jnp.exp(g)[..., None, None]
        corrected = v - jnp.einsum('bhk,bhkv->bhv', k, memory)
        memory = memory + jnp.einsum('bhk,bhv->bhkv', b[..., None] * written, corrected)
        return (memory, moment), jnp.einsum('bhk,bhkv->bhv', q * KEYS ** -0.5, memory)

    steps = (*(jnp.moveaxis(jnp.asarray(x, work), 1, 0)
               for x in (query, key, value, log_decay, beta, atk_decay, atk_beta)), first.T)
    entering = (jnp.asarray(state, work), jnp.repeat(jnp.asarray(atk_state, work), group, axis=1))
    (memory, moment), out = jax.lax.scan(one, entering, steps)
    return jnp.moveaxis(out, 0, 1).astype(value.dtype), (memory, moment[:, ::group])


def chunked(query, key, value, log_decay, beta, atk_decay, atk_beta, state, atk_state, starts=None):
    """The chunked rule on `tokenwise`'s operands."""
    return preconditioned_delta_rule(query, key, value, log_decay, beta, atk_decay, atk_beta, state,
                                     atk_state, starts, center=jnp.asarray(CENTERS), chunk_size=CHUNK)


def operands(per_key: bool):
    """A seed's fp32 operands as fla's layers form them (`hosted`): the log decays `log sigmoid` of normals
    (the memory's near one; a key dimension's each where `per_key`), ATK's decays near one and its gates a
    sigmoid's, the write gates a sigmoid's, entering states and moments (a moment nonnegative), and the
    cotangents of the output and both final states."""
    def at(seed):
        normal = np.random.default_rng([1458, per_key, seed]).standard_normal
        gates = (ROWS, LENGTH, VALUE_HEADS)
        keyed = (ROWS, LENGTH, KEY_HEADS)
        q, k = (normal((*keyed, KEYS)) for _ in range(2))
        v = normal((ROWS, LENGTH, VALUE_HEADS, VALUES))
        log_decay = -softplus(-(normal((*gates, KEYS) if per_key else gates) + 3.0))
        atk_decay = -softplus(-(normal(keyed) + 4.0))
        state = normal((ROWS, VALUE_HEADS, KEYS, VALUES))
        atk_state = np.abs(normal((ROWS, KEY_HEADS, KEYS))) * 0.1
        finals = (hosted(normal(state.shape)), hosted(normal(atk_state.shape)))
        cotangent = (hosted(normal(v.shape)), finals)
        gated = (log_decay, sigmoid(normal(gates)), atk_decay, sigmoid(normal(keyed)))
        drawn = (q, k, v, *gated, state, atk_state)
        return (*(hosted(x) for x in drawn), cotangent)
    return at


@pytest.mark.parametrize(('per_key', 'packed'), [(False, False), (False, True), (True, False), (True, True)],
                         ids=['pgdn', 'pgdn packed', 'pkda', 'pkda packed'])
def test_the_chunked_rule_is_as_exact_as_flas_recurrence_a_token_at_a_time(per_key, packed):
    """The chunked preconditioned rule's output, final state and ATK moment and the gradients of every
    operand and both entering states on par with fla's recurrence a token at a time from float64 (check 1),
    a decay a head's (PGDN) or a key dimension's (PKDA), one sequence or packed documents, each document's
    first token reading neither state before it, the centers held (the module's account)."""
    label = f"preconditioned {'kda' if per_key else 'gdn'}"
    assert_packed_on_par(chunked, tokenwise, operands(per_key), label, packed=packed, documents=DOCUMENTS,
                         length=LENGTH, seeds=SEEDS)


def stepped(query, key, value, log_decay, beta, atk_decay, atk_beta, state, atk_state):
    """The decode step (`preconditioned_step`) a token at a time over the sequence."""
    def one(carried, token):
        out, carried = preconditioned_step(*token, *carried, center=jnp.asarray(CENTERS))
        return carried, out

    tokens = tuple(jnp.moveaxis(x, 1, 0) for x in (query, key, value, log_decay, beta, atk_decay, atk_beta))
    final, out = jax.lax.scan(one, (state, atk_state), tokens)
    return jnp.moveaxis(out, 0, 1), final


@pytest.mark.parametrize('per_key', [False, True], ids=['pgdn', 'pkda'])
def test_the_decode_step_is_as_exact_as_flas_recurrence_a_token_at_a_time(per_key):
    """The decode step's outputs, final state and ATK moment and the gradients of every operand and both
    entering states, a token at a time, on par with fla's recurrence's from float64 (check 1): its delta rule
    corrected along the read of its keys (`recurrence.Hooks.erase` 'read', `recurrence.update`'s reader)."""
    assert_as_exact(jax.jit(with_gradients(stepped)), jax.jit(with_gradients(tokenwise)), operands(per_key),
                    f"preconditioned {'kda' if per_key else 'gdn'} decode steps", seeds=SEEDS)
