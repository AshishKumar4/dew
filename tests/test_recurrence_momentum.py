"""A coupled momentum on the chunked op and the decode step (`recurrence.Hooks.coupled` 'momentum', row 1454:
Momentum DeltaNet, fla's momentum_delta_rule): its output, final memory and momentum and every operand's
gradient on par with the recurrence a token at a time from float64 (check 1), one sequence and packed
documents, its erase key the keys' or its own, fp32 and bf16."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact, with_gradients
from test_recurrence_chunk import assert_packed_on_par, hosted, sigmoid, softplus, unit

from dew.nn.kernels import recurrence
from dew.nn.kernels.recurrence_chunk import Chunk, chunk

ROWS, LENGTH, HEADS, KEYS, VALUES, CHUNK = 2, 40, 2, 16, 8, 16
"""Two whole chunks of 16 tokens and a part one, keys wider than values."""
DOCUMENTS = ((0, 11, 30), (5, 17))
"""Each row's packed documents' first tokens: the first row's entering pair cut at its first token, the
second's read until its sixth."""
SEEDS = 16
"""Each site's declared seeds."""

TIED = recurrence.Hooks(decay='scalar', coupled='momentum', write_bound=1.0, scale=KEYS ** -0.5)
UNTIED = recurrence.Hooks(decay='scalar', coupled='momentum', write_bound=1.0, scale=KEYS ** -0.5,
                          erase='untied')


def tokenwise(query, key, value, log_decay, beta, momentum_decay, erase, state, starts=None):
    """The recurrence a token at a time as fla's reference states it (recurrent_momentum_delta_rule_ref),
    in at least fp32, a packed document's first token entering a zero pair: the output in the values' dtype
    and the final pair `[B, H, 2, Dk, Dv]`."""
    work = jnp.promote_types(jnp.result_type(query, key, value), jnp.float32)
    first = jnp.zeros(query.shape[:2], bool) if starts is None else jnp.asarray(starts)

    def one(pair, token):
        q, k, v, a, b, mu, p, fresh = token
        memory, momentum = (jnp.where(fresh[:, None, None, None], 0.0, x) for x in pair)
        error = jnp.einsum('bhk,bhkv->bhv', p, memory) - v
        momentum = jnp.exp(mu)[..., None, None] * momentum + k[..., :, None] * error[..., None, :]
        memory = jnp.exp(a)[..., None, None] * memory - b[..., None, None] * momentum
        return (memory, momentum), jnp.einsum('bhk,bhkv->bhv', q * KEYS ** -0.5, memory)

    steps = (*(jnp.moveaxis(jnp.asarray(x, work), 1, 0)
               for x in (query, key, value, log_decay, beta, momentum_decay, erase)), first.T)
    (memory, momentum), out = jax.lax.scan(one, (state[:, :, 0].astype(work), state[:, :, 1].astype(work)),
                                           steps)
    return jnp.moveaxis(out, 0, 1).astype(value.dtype), jnp.stack([memory, momentum], axis=2)


def chunked(hooks: recurrence.Hooks):
    """The chunked op under `hooks`, `tokenwise`'s operands, the erase keys its own where untied."""
    def run(query, key, value, log_decay, beta, momentum_decay, erase, state, starts=None):
        own = erase if hooks.erase == 'untied' else None
        return chunk(hooks, Chunk(query, key, value, log_decay=log_decay, beta=beta,
                                  momentum_decay=momentum_decay, erase_key=own, state=state, starts=starts),
                     chunk_size=CHUNK)
    return run


def tied(fn):
    """`fn` with the keys as the erase keys, an operand less."""
    def run(query, key, value, log_decay, beta, momentum_decay, state, starts=None):
        return fn(query, key, value, log_decay, beta, momentum_decay, key, state, starts=starts)
    return run


def operands(dtype, untied: bool):
    """A seed's operands as fla's layer forms them, queries, keys, values and erase keys in `dtype`
    (`hosted`): unit queries, keys and erase keys, the log decays `log sigmoid` of normals (the memory's
    near one, the momentum's held at -2 or above, fla's `min_log_mu`), the write gate a sigmoid's; an
    entering pair, and the cotangents of the output and the final pair."""
    def at(seed):
        normal = np.random.default_rng([1454, seed]).standard_normal
        gates = (ROWS, LENGTH, HEADS)
        q, k, p = (unit(normal((ROWS, LENGTH, HEADS, KEYS))) for _ in range(3))
        v = normal((ROWS, LENGTH, HEADS, VALUES))
        log_decay = -softplus(-(normal(gates) + 3.0))
        momentum_decay = np.maximum(-softplus(-(normal(gates) + 1.0)), -2.0)
        beta = sigmoid(normal(gates))
        state = hosted(normal((ROWS, HEADS, 2, KEYS, VALUES)))
        cotangent = (hosted(normal(v.shape), dtype), hosted(normal(state.shape)))
        erase = (hosted(p, dtype),) if untied else ()
        return (hosted(q, dtype), hosted(k, dtype), hosted(v, dtype), hosted(log_decay), hosted(beta),
                hosted(momentum_decay), *erase, state, cotangent)
    return at


@pytest.mark.parametrize(('untied', 'packed', 'dtype'), [
    (False, False, jnp.float32), (False, True, jnp.float32), (True, False, jnp.float32),
    (True, True, jnp.float32), (False, True, jnp.bfloat16)],
    ids=['one sequence', 'packed', 'own erase keys', 'own erase keys packed', 'packed bf16'])
def test_the_chunked_momentum_is_as_exact_as_the_recurrence_a_token_at_a_time(untied, packed, dtype):
    """The chunked op's output, final memory and momentum and the gradients of every operand and the
    entering pair on par with the recurrence a token at a time from float64 (check 1), the erase keys the
    keys or their own, one sequence and packed documents, each document's first token reading neither state
    before it."""
    dew, reference = chunked(UNTIED if untied else TIED), tokenwise
    if not untied:
        dew, reference = tied(dew), tied(reference)
    label = f"chunked momentum, {'own' if untied else 'tied'} erase keys, {jnp.dtype(dtype).name}"
    assert_packed_on_par(dew, reference, operands(dtype, untied), label, packed=packed, documents=DOCUMENTS,
                         length=LENGTH, seeds=SEEDS)


def stepped(query, key, value, log_decay, beta, momentum_decay, state):
    """The decode step (`recurrence.step`) a token at a time over the sequence: its outputs and final pair."""
    def one(state, token):
        q, k, v, a, b, mu = token
        out, _, state = recurrence.step(TIED, recurrence.Step(state, query=q, key=k, value=v, log_decay=a,
                                                              beta=b, momentum_decay=mu))
        return state, out

    steps = tuple(jnp.moveaxis(x, 1, 0) for x in (query, key, value, log_decay, beta, momentum_decay))
    final, out = jax.lax.scan(one, state, steps)
    return jnp.moveaxis(out, 0, 1), final


def test_the_decode_step_is_as_exact_as_the_recurrence_a_token_at_a_time():
    """The decode step's output and final pair, and the gradients of every operand and the entering pair, a
    token at a time on par with the recurrence's from float64 (check 1)."""
    assert_as_exact(jax.jit(with_gradients(stepped)), jax.jit(with_gradients(tied(tokenwise))),
                    operands(jnp.float32, untied=False), "decode momentum steps", seeds=SEEDS)


def test_a_coupled_momentum_refuses_a_forked_call_and_an_unbounded_gate():
    """A coupled momentum's hooks declare its write gate's bound, which its chunks' sums are bounded by, and
    its chunked op walks no forked documents: each refused, named."""
    with pytest.raises(ValueError):
        recurrence.Hooks(decay='scalar', coupled='momentum', scale=KEYS ** -0.5)
    q, k, v, log_decay, beta, mu, state, _ = operands(jnp.float32, untied=False)(0)
    loads = jnp.zeros((ROWS, 1), jnp.int32)
    with pytest.raises(ValueError):
        starts = jnp.zeros((ROWS, LENGTH), bool).at[:, 0].set(True)
        chunk(TIED, Chunk(q, k, v, log_decay=log_decay, beta=beta, momentum_decay=mu, state=state,
                          starts=starts, loads=loads, stores=loads), chunk_size=CHUNK)


def test_a_query_reads_a_memory_near_float32s_largest_decayed_first():
    """ReviewMomentum1454's overflow: one token reading a memory of 2^127 through a decay of 1/2, its query
    (1/2 each of four keys) and its unit key and erase key along the first, its value 2^126 and no write
    gate: the chunked op's and the decode step's output is the token recurrence's 2^127 and their final
    memory 2^126, each finite, where the query's read of the entering memory before its decay passed float32's
    largest."""
    hooks = recurrence.Hooks(decay='scalar', coupled='momentum', write_bound=1.0, scale=1.0)
    query = jnp.full((1, 1, 1, 4), 0.5, jnp.float32)
    key = jnp.zeros((1, 1, 1, 4), jnp.float32).at[..., 0].set(1.0)
    value = jnp.full((1, 1, 1, 1), 2.0 ** 126, jnp.float32)
    log_decay, beta, momentum_decay = (jnp.full((1, 1, 1), x, jnp.float32) for x in (np.log(0.5), 0.0, 0.0))
    state = jnp.zeros((1, 1, 2, 4, 1), jnp.float32).at[:, :, 0].set(2.0 ** 127)
    out, final = chunk(hooks, Chunk(query, key, value, log_decay=log_decay, beta=beta,
                                    momentum_decay=momentum_decay, state=state), chunk_size=CHUNK)
    token = {name: x[:, 0] for name, x in (('query', query), ('key', key), ('value', value),
                                            ('log_decay', log_decay), ('beta', beta),
                                            ('momentum_decay', momentum_decay))}
    stepped_out, _, stepped_final = recurrence.step(hooks, recurrence.Step(state, **token))
    for read, memory in ((out[:, 0], final), (stepped_out, stepped_final)):
        np.testing.assert_array_equal(np.asarray(read), 2.0 ** 127)
        np.testing.assert_array_equal(np.asarray(memory[:, :, 0]), 2.0 ** 126)
