"""A coupled momentum on the chunked op and the decode step (`recurrence.Hooks.coupled` 'momentum', row 1454:
Momentum DeltaNet, fla's momentum_delta_rule): its output, final memory and momentum and every operand's
gradient on par with the recurrence a token at a time from float64 (check 1), one sequence and packed
documents, its erase key the keys' or its own, fp32 and bf16."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec as P
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


def test_a_momentum_cancelling_past_float32s_range_carries_finite():
    """ReviewMomentum1454's cancellation: two tokens of queries, keys and erase keys along the first key,
    decays and write gates of one, values of 2^127 along the first value, the memory and the momentum
    entering 2^127 there: the chunked op's outputs and final pair are the decode step's, finite, where the
    entering momentum weighed by the tokens' summed write gates, -2^128, overflowed before the memory's term
    joined it."""
    hooks = recurrence.Hooks(decay='scalar', coupled='momentum', write_bound=1.0, scale=1.0)
    along = jnp.zeros((1, 2, 1, 2), jnp.float32).at[..., 0].set(1.0)
    value = jnp.zeros((1, 2, 1, 2), jnp.float32).at[..., 0].set(2.0 ** 127)
    zero, one = jnp.zeros((1, 2, 1), jnp.float32), jnp.ones((1, 2, 1), jnp.float32)
    state = jnp.zeros((1, 1, 2, 2, 2), jnp.float32).at[:, :, :, 0, 0].set(2.0 ** 127)
    out, final = chunk(hooks, Chunk(along, along, value, log_decay=zero, beta=one, momentum_decay=zero,
                                    state=state), chunk_size=CHUNK)

    def one_token(state, token):
        q, v, a, b, mu = token
        read, _, state = recurrence.step(hooks, recurrence.Step(state, query=q, key=q, value=v, log_decay=a,
                                                                beta=b, momentum_decay=mu))
        return state, read

    tokens = tuple(jnp.moveaxis(x, 1, 0) for x in (along, value, zero, one, zero))
    stepped_final, stepped = jax.lax.scan(one_token, state, tokens)
    for got, want in ((out, jnp.moveaxis(stepped, 0, 1)), (final, stepped_final)):
        assert np.isfinite(np.asarray(want)).all()
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_a_coupled_momentum_runs_in_a_shard_map_that_checks_varying_axes():
    """The chunked momentum's rows split over a checked `shard_map`'s axis: its outputs, final pair and every
    operand's gradient finite, each carry typed as the operands reaching it vary, the chunks' decayed sums'
    among them (ReviewMomentum1454: seeded replicated, their first step's carry varied with the gates)."""
    *given, cotangent = operands(jnp.float32, untied=False)(0)
    rows = P('data')

    def rule(query, key, value, log_decay, beta, momentum_decay, state):
        given = Chunk(query, key, value, log_decay=log_decay, beta=beta, momentum_decay=momentum_decay,
                      state=state)
        return chunk(TIED, given, chunk_size=CHUNK)

    mapped = jax.shard_map(rule, mesh=Mesh(np.asarray(jax.devices()[:2]), ('data',)), in_specs=rows,
                           out_specs=rows, check_vma=True)
    found = jax.jit(mapped)(*given)
    pulled = jax.jit(lambda *operands: jax.vjp(mapped, *operands)[1](cotangent))(*given)
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves((found, pulled)))


def test_a_chunks_writes_past_float32s_range_cancel_as_the_token_walks():
    """ReviewMomentum1454's local write (row 12693): two tokens of queries and keys along the first key,
    erase keys along the second, the memory's decays and the write gates one, the momentum's 1/8 then one,
    the memory entering 2^127 along the first key and the momentum none, values -2^127 then 0: the chunked
    op's outputs and final pair are the decode step's, finite, where its first write weighed by its sum to the
    chunk's end, -2 2^127, overflowed before the memory entering joined it (the entering momentum's factors,
    -1/8 and -1/4, which a chunk-wide divisor read, divided by nothing)."""
    hooks = recurrence.Hooks(decay='scalar', coupled='momentum', erase='untied', write_bound=1.0, scale=1.0)
    along = jnp.zeros((1, 2, 1, 2), jnp.float32).at[..., 0].set(1.0)
    erase = jnp.zeros((1, 2, 1, 2), jnp.float32).at[..., 1].set(1.0)
    value = jnp.asarray([-(2.0 ** 127), 0.0], jnp.float32).reshape(1, 2, 1, 1)
    zero, one = jnp.zeros((1, 2, 1), jnp.float32), jnp.ones((1, 2, 1), jnp.float32)
    state = jnp.zeros((1, 1, 2, 2, 1), jnp.float32).at[:, :, 0, 0, 0].set(2.0 ** 127)
    momentum_decay = jnp.asarray([np.log(0.125), 0.0], jnp.float32).reshape(1, 2, 1)
    given = Chunk(along, along, value, log_decay=zero, beta=one, momentum_decay=momentum_decay, erase_key=erase,
                  state=state)
    out, final = chunk(hooks, given, chunk_size=CHUNK)

    def one_token(state, token):
        q, p, v, a, b, mu = token
        read, _, state = recurrence.step(hooks, recurrence.Step(state, query=q, key=q, value=v, log_decay=a,
                                                                beta=b, momentum_decay=mu, erase_key=p))
        return state, read

    tokens = tuple(jnp.moveaxis(x, 1, 0) for x in (along, erase, value, zero, one, momentum_decay))
    stepped_final, stepped = jax.lax.scan(one_token, state, tokens)
    for got, want in ((out, jnp.moveaxis(stepped, 0, 1)), (final, stepped_final)):
        assert np.isfinite(np.asarray(want)).all()
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def faint(seed):
    """A chunk of 16 tokens whose last decays the memory by about exp(-86), every other's one, its keys
    along the first key and its erase keys along the second, no values and write gates of one, the memory
    entering 2^20 normals off the second key's row and the momentum none: the memory is the entering one
    decayed, its reads normal, its momentum's factors reaching -15 before the last token."""
    normal = np.random.default_rng([12693, seed]).standard_normal
    shape = (ROWS, CHUNK, HEADS)
    query = unit(normal((*shape, KEYS)))
    key, erase = np.zeros((*shape, KEYS)), np.zeros((*shape, KEYS))
    key[..., 0], erase[..., 1] = 1.0, 1.0
    log_decay = np.zeros(shape)
    log_decay[:, -1] = -86.0 - 0.25 * np.abs(normal((ROWS, HEADS)))
    state = np.zeros((ROWS, HEADS, 2, KEYS, VALUES))
    state[:, :, 0] = 2.0 ** 20 * normal((ROWS, HEADS, KEYS, VALUES))
    state[:, :, 0, 1] = 0.0
    keyed = (hosted(x, jnp.float32) for x in (query, key, np.zeros((*shape, VALUES))))
    gates = (hosted(x) for x in (log_decay, np.ones(shape), np.zeros(shape)))
    return (*keyed, *gates, hosted(erase, jnp.float32), hosted(state))


def test_a_decay_past_float32s_least_normal_keeps_the_memorys_normal_reads():
    """ReviewMomentum1454's flush (row 12693): a chunk's last token decaying the memory by about exp(-86)
    beside momentum factors reaching -15: the chunked op's outputs and final pair on par with the recurrence a
    token at a time from float64 (check 1), the last token's read about 2^-106, where a chunk-wide divisor of
    16 brought the decayed query below float32's least normal and flushed it."""
    assert_as_exact(jax.jit(chunked(UNTIED)), jax.jit(tokenwise), faint, "a faint decay's chunked reads",
                    seeds=SEEDS)
