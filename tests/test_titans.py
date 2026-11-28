"""Titans' linear memory (`dew.nn.kernels.titans`) and TTT-Linear (`dew.nn.kernels.ttt`) a mini-batch at a
time, as Dew's in-context memory computes them (each token stepping by its own rate, the residual around its
query): each one's output, final states and every operand's gradient on par with a token recurrence from
float64 (check 1), the recurrence this file's own (its inner gradient the inner loss's by autodiff, no part
of either module's)."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec as P
from reference_error import assert_as_exact, with_gradients
from test_recurrence_chunk import hosted, sigmoid, unit

from dew.nn.kernels.titans import titans
from dew.nn.kernels.ttt import ttt_linear

ROWS, LENGTH, HEADS, WIDTH, CHUNK = 2, 40, 2, 16, 16
"""Two whole mini-batches of 16 tokens and a part one."""
SEEDS = 16
"""Each site's declared seeds."""


def _layer_normed(x, eps: float):
    centered = x - jnp.mean(x, axis=-1, keepdims=True)
    return jnp.divide(centered, jnp.sqrt(jnp.mean(jnp.square(centered), axis=-1, keepdims=True) + eps))


def _half_gradient(k, v, readout, weight, bias, eps: float):
    """Half the inner loss `|k + (LN(readout) w + b) - v|^2`'s gradient at `readout`, by autodiff."""
    def loss(at):
        return 0.5 * jnp.sum(((k + (_layer_normed(at, eps) * weight + bias)) - v) ** 2)

    return jax.grad(loss)(readout)


def _read_out(q, out, weight, bias, eps: float):
    """Each query plus its readout's LayerNorm under the affine: the memory's residual around its input."""
    return q + (_layer_normed(out, eps) * weight + bias)


def _tokens_first(work, *xs):
    """Each of `xs` `[B, T, ...]` in `work`, its tokens first."""
    return tuple(jnp.moveaxis(jnp.asarray(x, work), 1, 0) for x in xs)


def titans_by_token(query, key, value, weight, bias, theta, alpha, eta, state=None, *, chunk: int,
                    eps: float = 1e-6):
    """Titans a token at a time as fla's reference walks it, each token's gradient at the memory entering its
    mini-batch (`_half_gradient`), in at least fp32 of its operands."""
    dtype = jnp.result_type(query, key, value)
    work = jnp.promote_types(dtype, jnp.float32)
    tokens = _tokens_first(work, query, key, value, theta, alpha, eta)
    weight, bias = (jnp.asarray(x, work) for x in (weight, bias))
    rows, _, heads, width = jnp.shape(key)
    memory, momentum = ((jnp.zeros((rows, heads, width, width), work),) * 2 if state is None
                        else (jnp.asarray(s, work) for s in state))

    def step(carried, at):
        memory, momentum, anchor = carried
        t, q, k, v, theta, alpha, eta = at
        anchor = jnp.where(t % chunk == 0, memory, anchor)
        u = _half_gradient(k, v, jnp.einsum('bhk,bhkv->bhv', k, anchor), weight, bias, eps)
        momentum = (eta[..., None, None] * momentum
                    - 2.0 * theta[..., None, None] * k[..., :, None] * u[..., None, :])
        memory = (1.0 - alpha)[..., None, None] * memory + momentum
        return (memory, momentum, anchor), jnp.einsum('bhk,bhkv->bhv', q, memory)

    steps = (jnp.arange(tokens[0].shape[0]), *tokens)
    (memory, momentum, _), outs = jax.lax.scan(step, (memory, momentum, memory), steps)
    queries = jnp.moveaxis(tokens[0], 0, 1)
    return _read_out(queries, jnp.moveaxis(outs, 0, 1), weight, bias, eps).astype(dtype), (memory, momentum)


def ttt_by_token(query, key, value, weight, bias, eta, state=None, *, chunk: int, scale=None,
                 eps: float = 1e-6):
    """TTT-Linear a token at a time as Dew's in-context memory steps it: each mini-batch's entering memory and
    bias held, each token's gradient at them (`_half_gradient`), each token stepping the state by its own
    rate times its own gradient and reading the state it leaves, in at least fp32 of its operands."""
    dtype = jnp.result_type(query, key, value)
    work = jnp.promote_types(dtype, jnp.float32)
    q, k, v, eta = _tokens_first(work, query, key, value, eta)
    weight, bias = (jnp.asarray(x, work) for x in (weight, bias))
    rows, _, heads, width = jnp.shape(key)
    q = q * (width ** -0.5 if scale is None else scale)
    memory, offset = ((jnp.zeros((rows, heads, width, width), work), jnp.zeros((rows, heads, 1, width), work))
                      if state is None else (jnp.asarray(s, work) for s in state))

    def step(carried, at):
        memory, offset, entering, entering_offset = carried
        t, q, k, v, eta = at
        start = t % chunk == 0
        entering = jnp.where(start, memory, entering)
        entering_offset = jnp.where(start, offset, entering_offset)
        readout = jnp.einsum('bhk,bhkv->bhv', k, entering) + entering_offset[:, :, 0]
        u = _half_gradient(k, v, readout, weight, bias, eps)
        rate = eta[..., None, None]
        memory = memory - rate * k[..., :, None] * u[..., None, :]
        offset = offset - rate * u[..., None, :]
        out = jnp.einsum('bhk,bhkv->bhv', q, memory) + offset[:, :, 0]
        return (memory, offset, entering, entering_offset), out

    carried = (memory, offset, memory, offset)
    (memory, offset, *_), outs = jax.lax.scan(step, carried, (jnp.arange(q.shape[0]), q, k, v, eta))
    return _read_out(jnp.moveaxis(q, 0, 1), jnp.moveaxis(outs, 0, 1), weight, bias,
                     eps).astype(dtype), (memory, offset)


def operands(form: str, dtype):
    """A seed's operands of `form`, queries, keys and values in `dtype` (`hosted`): unit queries and keys,
    the inner loss's affine near the identity, a learning rate below a tenth (Titans': its rate, forgetting a
    sigmoid's below one half and momentum one's above; TTT's: a rate a token), entering states (Titans'
    memory and momentum, TTT's memory and bias; zero at an even seed, so its first mini-batch's readouts are
    constant rows), and the cotangents of the output and both final states."""
    def at(seed):
        normal = np.random.default_rng([1449 if form == 'titans' else 1448, seed]).standard_normal
        gates = (ROWS, LENGTH, HEADS)
        q, k = (unit(normal((ROWS, LENGTH, HEADS, WIDTH))) for _ in range(2))
        v = normal((ROWS, LENGTH, HEADS, WIDTH))
        weight, bias = 1.0 + 0.1 * normal((HEADS, WIDTH)), 0.1 * normal((HEADS, WIDTH))
        rates = (0.1 * sigmoid(normal(gates)),)
        shapes = ((ROWS, HEADS, WIDTH, WIDTH), (ROWS, HEADS, WIDTH if form == 'titans' else 1, WIDTH))
        if form == 'titans':
            rates += (sigmoid(normal(gates) - 3.0), sigmoid(normal(gates) + 2.0))
        states = tuple(hosted((0.1 if seed % 2 else 0.0) * normal(shape)) for shape in shapes)
        cotangent = (hosted(normal(v.shape), dtype), tuple(hosted(normal(s.shape)) for s in states))
        gated = (hosted(x) for x in (weight, bias, *rates))
        return (hosted(q, dtype), hosted(k, dtype), hosted(v, dtype), *gated, states, cotangent)
    return at


FORMS = {'titans': (titans, titans_by_token), 'ttt': (ttt_linear, ttt_by_token)}


@pytest.mark.parametrize('form', sorted(FORMS))
@pytest.mark.parametrize('dtype', [jnp.float32, jnp.bfloat16], ids=['fp32', 'bf16'])
def test_the_mini_batches_are_as_exact_as_the_recurrence_a_token_at_a_time(form, dtype):
    """The mini-batched memory's output, final states and the gradients of every operand, the inner loss's
    affine and both entering states on par with this file's token recurrence from float64 (check 1), each
    token's inner gradient at the memory entering its mini-batch, the last mini-batch part of one: Titans'
    memory and momentum, and TTT-Linear's memory and bias, each output read at its own rate and the state
    leaving at the mini-batch's last token's."""
    chunked, stepped = (jax.jit(with_gradients(functools.partial(fn, chunk=CHUNK))) for fn in FORMS[form])
    assert_as_exact(chunked, stepped, operands(form, dtype), f'{form}, {jnp.dtype(dtype).name}', seeds=SEEDS)


def _pair(*rows):
    """`[1, len(rows), 1, len(rows[0])]` fp32 from `rows`."""
    return jnp.asarray(rows, jnp.float32).reshape(1, len(rows), 1, len(rows[0]))


@pytest.mark.parametrize('form', sorted(FORMS))
def test_a_live_or_later_overflowed_score_leaves_the_reads_finite(form):
    """Two-token fp32 mini-batches from zero states under an identity affine, unit scale, values their keys:
    queries and keys all `(2^70, 0)` (live scores 2^140) and the first query and second key alone so (a later
    score): every output finite and the token recurrence's (zero), no query meeting a key, where a score of
    2^140 met a zero update, `inf * 0`, NaN (train 1403's review; a head review of 94d6dee89)."""
    far, zero = (2.0 ** 70, 0.0), (0.0, 0.0)
    affine = (jnp.ones((1, 2)), jnp.zeros((1, 2)))
    rates = (jnp.full((1, 2, 1), 0.1),) + ((jnp.full((1, 2, 1), 0.5),) * 2 if form == 'titans' else ())
    settings = {'chunk': 2} | ({} if form == 'titans' else {'scale': 1.0})
    for query, key in ((_pair(far, far), _pair(far, far)), (_pair(far, zero), _pair(zero, far))):
        found, recurrent = (fn(query, key, key, *affine, *rates, **settings)[0] for fn in FORMS[form])
        assert np.isfinite(np.asarray(found)).all(), found
        np.testing.assert_array_equal(np.asarray(found), np.asarray(recurrent))


def test_titans_memory_forgotten_whole_reads_zero():
    """Titans under `alpha` 1 (the memory forgotten whole) from an entering memory of 2^70 at a query of
    `(2^70, 0)`, no key, value or momentum: its readout and final memory zero, so its output the query
    itself (the residual around it), the forgetting applied to the state before any read, where `0 * (q M)`
    with `q M` = 2^140 read NaN (train 1403's review)."""
    entering = jnp.zeros((1, 1, 2, 2)).at[0, 0, 0, 0].set(2.0 ** 70)
    zero = jnp.zeros((1, 1, 1, 2), jnp.float32)
    query = _pair((2.0 ** 70, 0.0))
    out, (memory, _) = titans(query, zero, zero, jnp.ones((1, 2)), jnp.zeros((1, 2)),
                              *(jnp.full((1, 1, 1), x) for x in (0.1, 1.0, 0.5)),
                              state=(entering, jnp.zeros_like(entering)), chunk=1)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(query))
    np.testing.assert_array_equal(np.asarray(memory), np.zeros((1, 1, 2, 2), np.float32))


def test_ttts_padded_token_leaves_a_real_ones_output_and_state():
    """TTT-Linear over one real token in a mini-batch of two, its entering bias `[-A, -A, -A, 15 2^124]` (`A`
    2^127) and its memory's first row the bias negated, a unit query, key and value `e0`, zero affine bias:
    the real token's readout zero, its update none, its output its query (the residual around it) and the
    state leaving as it entered, the padded token's readout (the bias itself) zero before the LayerNorm sees
    it, where its overflowed variance's `0 * inf` reached the real token's output and the bias (train 1403's
    review)."""
    a = 2.0 ** 127
    offset = jnp.asarray([-a, -a, -a, 15 * 2.0 ** 124], jnp.float32).reshape(1, 1, 1, 4)
    memory = jnp.zeros((1, 1, 4, 4), jnp.float32).at[0, 0, 0].set(-offset[0, 0, 0])
    unit_e0 = jnp.zeros((1, 1, 1, 4), jnp.float32).at[..., 0].set(1.0)
    affine = (jnp.ones((1, 4)), jnp.zeros((1, 4)))
    out, (left, left_offset) = ttt_linear(unit_e0, unit_e0, unit_e0, *affine, jnp.full((1, 1, 1), 0.1),
                                          state=(memory, offset), chunk=2, scale=1.0)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(unit_e0))
    np.testing.assert_array_equal(np.asarray(left), np.asarray(memory))
    np.testing.assert_array_equal(np.asarray(left_offset), np.asarray(offset))


def test_titans_rate_meets_its_key_before_an_inner_gradient_near_the_largest():
    """Titans over one fp32 token, width 2, a zero query and entering state, its key and value `(1/4, 0)`,
    `theta` 3/4, no forgetting or momentum, `eps` 1, weights `2^66` and biases `(3 2^60, -3 2^60)`, its inner
    gradient `(3 2^126, -3 2^126)`: its momentum and memory `-(3/8, 0)` outer that, `(-9 2^123, 9 2^123)`
    in the first row and zero in the second, as the token recurrence's `2 theta k u`, where the rate weighed
    onto the gradient first overflowed and the second row's zero key wrote NaN (train 1430's review)."""
    k = jnp.asarray([0.25, 0.0], jnp.float32).reshape(1, 1, 1, 2)
    gates = tuple(jnp.full((1, 1, 1), x, jnp.float32) for x in (0.75, 0.0, 0.0))
    affine = (jnp.full((1, 2), 2.0 ** 66, jnp.float32), jnp.asarray([[3 * 2.0 ** 60, -3 * 2.0 ** 60]]))
    _, (memory, momentum) = titans(jnp.zeros_like(k), k, k, *affine, *gates, chunk=1, eps=1.0)
    expected = np.zeros((1, 1, 2, 2), np.float32)
    expected[0, 0, 0] = [-9 * 2.0 ** 123, 9 * 2.0 ** 123]
    for found in (memory, momentum):
        np.testing.assert_array_equal(np.asarray(found), expected)


def test_titans_inner_gradient_at_a_small_eps_is_the_token_recurrences():
    """Titans over one fp32 token, width 2, a zero query and entering state, its key `(1, 0)` and value `(0,
    1)`, a unit weight and zero bias, `theta` 1/2, no forgetting or momentum, `eps` 2^-100: its memory's and
    momentum's first row `(-2^50, 2^50)`, the LayerNorm's pullback at a zero readout `(1, -1) / sqrt(eps)`
    as the token recurrence's, where `rsqrt`'s derivative passed the range first (`rsqrt(eps) / eps`,
    2^150) and its product with the zero readout was NaN (a head review)."""
    k = jnp.asarray([1.0, 0.0], jnp.float32).reshape(1, 1, 1, 2)
    v = jnp.asarray([0.0, 1.0], jnp.float32).reshape(1, 1, 1, 2)
    gates = tuple(jnp.full((1, 1, 1), x, jnp.float32) for x in (0.5, 0.0, 0.0))
    affine = (jnp.ones((1, 2), jnp.float32), jnp.zeros((1, 2), jnp.float32))
    _, (memory, momentum) = titans(jnp.zeros_like(k), k, v, *affine, *gates, chunk=1, eps=2.0 ** -100)
    expected = np.zeros((1, 1, 2, 2), np.float32)
    expected[0, 0, 0] = [-(2.0 ** 50), 2.0 ** 50]
    for found in (memory, momentum):
        np.testing.assert_array_equal(np.asarray(found), expected)


def test_titans_rate_meets_its_key_before_a_zero_gradient_beside_a_key_near_the_largest():
    """Titans over one fp32 token, width 2, its key and value `(2^127, 0)`, `theta` 1, no forgetting or
    momentum, a unit weight, zero bias and state, `eps` 1: its memory and momentum zero, the LayerNorm's
    reconstruction exact (its inner gradient zero), the rate times the key `2^127` and the 2 after the
    product, where `2 theta k` overflowed and its zero gradient wrote NaN (train 1455's review)."""
    k = jnp.asarray([2.0 ** 127, 0.0], jnp.float32).reshape(1, 1, 1, 2)
    gates = tuple(jnp.full((1, 1, 1), x, jnp.float32) for x in (1.0, 0.0, 0.0))
    affine = (jnp.ones((1, 2), jnp.float32), jnp.zeros((1, 2), jnp.float32))
    _, (memory, momentum) = titans(jnp.zeros_like(k), k, k, *affine, *gates, chunk=1, eps=1.0)
    for found in (memory, momentum):
        np.testing.assert_array_equal(np.asarray(found), np.zeros((1, 1, 2, 2), np.float32))


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize('form', sorted(FORMS))
def test_queries_alone_split_in_a_checked_shard_map_leave_the_states_replicated(form):
    """A mini-batched memory in a checked `shard_map` over two devices, every operand replicated but its
    queries, each device's own (`q` and `2 q`): its final states replicated (`out_specs` `P()`), those of one
    query on both devices exactly, no state reading a query, where the queries' padding mask typed the states
    split (train 1455's review). Its values are the oracle test's."""
    *given, _ = operands(form, jnp.float32)(1)
    (q, k, v), affine, gates, states = tuple(given[:3]), tuple(given[3:5]), tuple(given[5:-1]), given[-1]
    call = functools.partial(FORMS[form][0], chunk=CHUNK)

    def local(queries):
        return call(queries[0], k, v, *affine, *gates, state=states)[1]

    mesh = Mesh(np.asarray(jax.devices()[:2]), ('data',))
    mapped = jax.shard_map(local, mesh=mesh, in_specs=P('data'), out_specs=P(), check_vma=True)
    found = jax.jit(mapped)(jnp.stack([q, 2 * q]))
    # The same program with one query on both devices: its states the split call's, bit for bit.
    expected = jax.jit(mapped)(jnp.stack([q, q]))
    for got, want in zip(jax.tree.leaves(found), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_titans_rates_product_pulls_a_cotangent_near_the_largest_back_in_range():
    """The gradient by the value of `2^127` times the first entry of Titans' final memory over one fp32 token,
    its key and value `(1/4, 0)`, `theta` 1, no forgetting or momentum, a unit weight, zero bias and state,
    `eps` 1: `(2^125, -2^125)` exactly, the write `2 theta k u` one product whose transpose meets the rated
    key `1/2` before any doubling, where the cotangent doubled first overflowed (a head review)."""
    k = jnp.asarray([0.25, 0.0], jnp.float32).reshape(1, 1, 1, 2)
    gates = tuple(jnp.full((1, 1, 1), x, jnp.float32) for x in (1.0, 0.0, 0.0))
    affine = (jnp.ones((1, 2), jnp.float32), jnp.zeros((1, 2), jnp.float32))

    def first(v):
        _, (memory, _) = titans(jnp.zeros_like(k), k, v, *affine, *gates, chunk=1, eps=1.0)
        return 2.0 ** 127 * memory[0, 0, 0, 0]

    found = jax.grad(first)(k)
    np.testing.assert_array_equal(np.asarray(found).reshape(-1), [2.0 ** 125, -(2.0 ** 125)])


@pytest.mark.mesh(devices=2)
def test_titans_replicated_key_beside_split_rates_takes_both_devices_gradients():
    """The gradient by a replicated key `(1, 0)` of the two devices' first memory entries summed, Titans over
    one fp32 token in a checked `shard_map`, its rate `1/2` on one device and `1` on the other, its value `(0,
    1)`, a zero query, state and bias, a unit weight, no forgetting or momentum, `eps` 1: `(-4.5, 1.5)`
    exactly, each device's `theta (-3, 1)` summed, the write's transpose summed over the axis its key does not
    vary over, where it returned a split cotangent to the replicated key (a head review)."""
    v = jnp.asarray([0.0, 1.0], jnp.float32).reshape(1, 1, 1, 2)
    affine = (jnp.ones((1, 2), jnp.float32), jnp.zeros((1, 2), jnp.float32))
    zero = jnp.zeros((1, 1, 1), jnp.float32)
    rates = jnp.asarray([0.5, 1.0], jnp.float32).reshape(2, 1, 1, 1)
    mesh = Mesh(np.asarray(jax.devices()[:2]), ('data',))

    def total(k):
        def local(theta):
            _, (memory, _) = titans(jnp.zeros_like(k), k, v, *affine, theta[0], zero, zero, chunk=1, eps=1.0)
            return jax.lax.psum(memory[0, 0, 0, 0], 'data')
        return jax.shard_map(local, mesh=mesh, in_specs=P('data'), out_specs=P(), check_vma=True)(rates)

    found = jax.jit(jax.grad(total))(jnp.asarray([1.0, 0.0], jnp.float32).reshape(1, 1, 1, 2))
    np.testing.assert_array_equal(np.asarray(found).reshape(-1), [-4.5, 1.5])


def test_titans_rate_near_the_largest_beside_a_zero_gradient_writes_nothing():
    """Titans over one fp32 token, width 2, its key and value `(1/4, 0)`, `theta` 2^127, no forgetting or
    momentum, a unit weight, zero bias and state, `eps` 1: its memory and momentum zero, the LayerNorm's
    reconstruction exact (its inner gradient zero), the 2 one factor of the write's one product, where the
    rate doubled first overflowed and its zero gradient wrote NaN (train 1461's review)."""
    k = jnp.asarray([0.25, 0.0], jnp.float32).reshape(1, 1, 1, 2)
    gates = tuple(jnp.full((1, 1, 1), x, jnp.float32) for x in (2.0 ** 127, 0.0, 0.0))
    affine = (jnp.ones((1, 2), jnp.float32), jnp.zeros((1, 2), jnp.float32))
    _, (memory, momentum) = titans(jnp.zeros_like(k), k, k, *affine, *gates, chunk=1, eps=1.0)
    for found in (memory, momentum):
        np.testing.assert_array_equal(np.asarray(found), np.zeros((1, 1, 2, 2), np.float32))


def test_ttts_rate_meets_its_keys_before_their_updates_sum():
    """TTT-Linear over two fp32 tokens of one mini-batch, width 2, a zero query, value and entering state, a
    unit weight and zero bias, `eps` 1, its keys `(2^70, 0)` and zero, both rates `2^-20`: its final memory's
    first row `(-2^119, 2^119)` and bias `(-2^49, 2^49)` exactly, the first token's update `(2^69, -2^69)`
    rated as it meets its key, where its unrated product `2^139` overflowed (train 1470's review)."""
    keys = _pair((2.0 ** 70, 0.0), (0.0, 0.0))
    zeros = jnp.zeros_like(keys)
    affine = (jnp.ones((1, 2), jnp.float32), jnp.zeros((1, 2), jnp.float32))
    _, (memory, offset) = ttt_linear(zeros, keys, zeros, *affine, jnp.full((1, 2, 1), 2.0 ** -20), chunk=2,
                                     eps=1.0)
    np.testing.assert_array_equal(np.asarray(memory), [[[[-(2.0 ** 119), 2.0 ** 119], [0.0, 0.0]]]])
    np.testing.assert_array_equal(np.asarray(offset), [[[[-(2.0 ** 49), 2.0 ** 49]]]])


def test_a_later_tokens_overflowing_update_leaves_an_earlier_read_as_it_is():
    """TTT-Linear over two fp32 tokens of one mini-batch, width 2, a zero query, key and entering state, the
    second token's value `(2^127, 0)`, unit rates, an affine weight of 4 and zero bias, `eps` 1: the first
    token's output zero, read from the state its own step leaves, the second token's non-finite update (its
    LayerNorm pullback's seed `-2^127 4`) none of it, where a prefix holding it behind a zero factor was NaN
    (a head review)."""
    zeros = _pair((0.0, 0.0), (0.0, 0.0))
    affine = (jnp.full((1, 2), 4.0, jnp.float32), jnp.zeros((1, 2), jnp.float32))
    out, _ = ttt_linear(zeros, zeros, _pair((0.0, 0.0), (2.0 ** 127, 0.0)), *affine, jnp.ones((1, 2, 1)),
                        chunk=2, eps=1.0)
    np.testing.assert_array_equal(np.asarray(out)[0, 0], np.zeros((1, 2), np.float32))


@pytest.mark.parametrize('memory', ['titans', 'ttt'])
def test_a_calls_absent_tail_leaves_it_the_call_on_its_real_tokens(memory):
    """Titans' or TTT-Linear's fp32 call over five tokens of two heads in mini-batches of two, its last two
    absent (`valid`) and holding `2^120` in every operand: its outputs at its real tokens and its final state
    bit for bit the call over its three real tokens alone, an absent token's operands zeroed before any
    product (an in-context memory's compacted rows' tails, rows 1448/1449's consumer)."""
    rng = np.random.default_rng(1448)

    def normal(*shape):
        return jnp.asarray(rng.standard_normal(shape) / 4, jnp.float32)

    q, k, v = (normal(1, 5, 2, 4) for _ in range(3))
    gates = [jnp.asarray(rng.uniform(0.05, 0.3, (1, 5, 2)), jnp.float32)
             for _ in range(3 if memory == 'titans' else 1)]
    affine = (1 + normal(2, 4), normal(2, 4))
    run = functools.partial(titans if memory == 'titans' else ttt_linear, chunk=2)

    def junk(x):
        return x.at[:, 3:].set(2.0 ** 120)

    found, found_state = run(*map(junk, (q, k, v)), *affine, *map(junk, gates),
                             valid=jnp.asarray([[True, True, True, False, False]]))
    expected, expected_state = run(q[:, :3], k[:, :3], v[:, :3], *affine, *(gate[:, :3] for gate in gates))
    np.testing.assert_array_equal(np.asarray(found)[:, :3], np.asarray(expected))
    for state, alone in zip(found_state, expected_state, strict=True):
        np.testing.assert_array_equal(np.asarray(state), np.asarray(alone))


def test_titans_write_coefficient_is_one_factor_of_its_product():
    """Titans' write at `coefficient` 1 and its rate `theta` is bit for bit its write at the default 2 and
    half the rate (an in-context memory's half-squared loss), each one factor of the write's exact
    product."""
    query, key, value, weight, bias, theta, alpha, eta, state, _ = operands('titans', jnp.float32)(0)
    run = functools.partial(titans, query, key, value, weight, bias, chunk=CHUNK, state=state)
    halved = run(theta / 2, alpha, eta)
    unit = run(theta, alpha, eta, coefficient=1.0)
    for found, expected in zip(jax.tree.leaves(unit), jax.tree.leaves(halved), strict=True):
        np.testing.assert_array_equal(np.asarray(found), np.asarray(expected))


@pytest.mark.parametrize('form', sorted(FORMS))
def test_a_memorys_mini_batches_are_recomputed_where_the_rung_recomputes_the_chunks(form):
    """Each mini-batched memory's residuals for its gradient (`jax.ad_checkpoint.saved_residuals`) fewer bytes
    where the run's rung recomputes the memory's chunks (`residuals.memory_chunks`, as `dew.nn.memory`'s),
    each mini-batch recomputed from the state it entered with (XLA fuses the recomputation its own way, so
    its gradients round apart from the stored run's)."""
    from jax._src.ad_checkpoint import saved_residuals

    from dew.nn.residuals import MEMORY_RESIDUALS, recomputing

    *given, states, _ = operands(form, jnp.float32)(1)
    memory = functools.partial(FORMS[form][0], chunk=CHUNK)

    def total(*arguments):
        return sum(jnp.sum(x) for x in jax.tree.leaves(memory(*arguments)))

    def saved():
        return sum(aval.size * aval.dtype.itemsize for aval, _ in saved_residuals(total, *given, states))

    stored_bytes = saved()
    with recomputing((MEMORY_RESIDUALS[1],)):
        recomputed_bytes = saved()
    assert recomputed_bytes < stored_bytes, (recomputed_bytes, stored_bytes)


def test_a_memorys_residual_survives_an_affine_its_bias_cancels():
    """Titans over one fp32 token, its query `e0` reading a memory whose first row is `(1, -1)` (its readout
    `(1, -1)`, normed to itself at `eps` 2^-30), no key, value or rates, an affine weight of 2^25 and bias
    `(-2^25, 2^25)`: its output `e0` exactly, the affine whole before the query's residual
    (`ResidualMemory`'s `q + (LN w + c)`), where `(q + LN w) + c` lost the query to the bias's cancellation (a
    head review)."""
    query = _pair((1.0, 0.0))
    zero = jnp.zeros_like(query)
    entering = jnp.zeros((1, 1, 2, 2), jnp.float32).at[0, 0, 0].set(jnp.asarray([1.0, -1.0]))
    # Past fp32's 24 bits from the query's 1, within the inner gradient's range (its error times the weight).
    large = 2.0 ** 25
    affine = (jnp.full((1, 2), large, jnp.float32), jnp.asarray([[-large, large]], jnp.float32))
    out, _ = titans(query, zero, zero, *affine, *(jnp.zeros((1, 1, 1)) for _ in range(3)),
                    state=(entering, jnp.zeros_like(entering)), chunk=1, eps=2.0 ** -30)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(query))


def test_a_ttt_tokens_absent_slot_leaves_its_live_gradients_the_one_slot_calls():
    """TTT-Linear over two tokens, the second absent (`valid`) with an infinite query, beside the entering
    bias `[-A, -A, -A, 15 2^124]` (`A` 2^127) of `test_ttts_padded_token_leaves_a_real_ones_output_and_state`:
    the live output's gradients by the entering states and the query scale finite and the one-slot call's bit
    for bit, the absent query zeroed before its scale and its readout (the bias kept) zeroed before the
    LayerNorm, where either met its zero cotangent as NaN (a head review)."""
    a = 2.0 ** 127
    offset = jnp.asarray([-a, -a, -a, 15 * 2.0 ** 124], jnp.float32).reshape(1, 1, 1, 4)
    memory = jnp.zeros((1, 1, 4, 4), jnp.float32).at[0, 0, 0].set(-offset[0, 0, 0])
    unit_e0 = jnp.zeros((1, 1, 1, 4), jnp.float32).at[..., 0].set(1.0)
    affine = (jnp.ones((1, 4)), jnp.zeros((1, 4)))

    def live(memory, offset, scale, query, key, value, rate, valid):
        out, _ = ttt_linear(query, key, value, *affine, rate, state=(memory, offset), chunk=2, scale=scale,
                            valid=valid)
        return jnp.sum(out[:, :1])

    two = (jnp.concatenate([unit_e0, jnp.full_like(unit_e0, np.inf)], axis=1),
           *(jnp.concatenate([unit_e0, unit_e0], axis=1) for _ in range(2)), jnp.full((1, 2, 1), 0.1),
           jnp.asarray([[True, False]]))
    one = (unit_e0, unit_e0, unit_e0, jnp.full((1, 1, 1), 0.1), None)
    gradient = jax.jit(jax.grad(live, argnums=(0, 1, 2)))
    found = gradient(memory, offset, jnp.float32(1.0), *two)
    expected = gradient(memory, offset, jnp.float32(1.0), *one)
    for leaf, alone in zip(found, expected, strict=True):
        np.testing.assert_array_equal(np.asarray(leaf), np.asarray(alone))


def test_ttts_inner_error_keeps_a_residual_its_target_would_round_away():
    """TTT-Linear over one fp32 token, its entering state zero, the inner loss's affine weight one and bias
    `(-2^100, 0)`, `eps` 1, its key `(2^100, 1)`, value `(1, 1)` and rate 2^-100: its memory's first row
    `(1/2, -1/2)` exactly, the error `(k + (LN(k W) w + c)) - v` the memory's own (`(-1, 0)`), where against
    `v - k` the target rounded to `(-2^100, 0)` and erased the update (a head review)."""
    key, value = _pair((2.0 ** 100, 1.0)), _pair((1.0, 1.0))
    affine = (jnp.ones((1, 2), jnp.float32), jnp.asarray([[-(2.0 ** 100), 0.0]], jnp.float32))
    _, (memory, _) = ttt_linear(jnp.zeros_like(key), key, value, *affine, jnp.full((1, 1, 1), 2.0 ** -100),
                                chunk=1, scale=1.0, eps=1.0)
    np.testing.assert_array_equal(np.asarray(memory)[0, 0, 0], np.asarray([0.5, -0.5], np.float32))


def test_titans_partial_mini_batch_leaves_the_state_its_last_token_does():
    """Titans over one token in a mini-batch of two, its key, value and affine bias zero (no write), its
    momentum entering zero, under `alpha` 1/2: its memory leaving half the entering one and its momentum
    zero, mini-batched and a token at a time alike, its padded token changing nothing (fla's reference,
    giving that slot the real token's gates, keeps a quarter; `titans.titans`)."""
    entering = jnp.asarray([[1.0, 2.0], [3.0, 4.0]]).reshape(1, 1, 2, 2)
    zero = jnp.zeros((1, 1, 1, 2), jnp.float32)
    gates = (jnp.full((1, 1, 1), 0.3), jnp.full((1, 1, 1), 0.5), jnp.full((1, 1, 1), 0.5))
    for fn in FORMS['titans']:
        _, (memory, momentum) = fn(zero, zero, zero, jnp.ones((1, 2)), jnp.zeros((1, 2)), *gates,
                                   state=(entering, jnp.zeros_like(entering)), chunk=2)
        np.testing.assert_array_equal(np.asarray(memory), np.asarray(entering) / 2)
        np.testing.assert_array_equal(np.asarray(momentum), np.zeros((1, 1, 2, 2), np.float32))


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize('replicated', [(), ('query',)], ids=['every token split', 'the query replicated'])
@pytest.mark.parametrize('form', sorted(FORMS))
def test_the_mini_batches_run_on_an_explicit_mesh_split_by_rows(form, replicated):
    """Each form under `jit` over two devices of an Explicit mesh, its rows split across them (every token's,
    or the keys', values' and rates' with the query replicated), its entering states implicit (zeros;
    `[B, T, H, D]` `[2, 2, 1, 2]`, chunk 2): its outputs split by rows, and they, its final states and every
    operand's gradient finite, its operands and scans' seeds placed alike, the keys' rows and heads first
    (`titans.placed_alike`), where a replicated zero seed, or a replicated query, beside batch-split keys was
    refused (train 1409's review and a head review)."""
    from jax.sharding import AxisType, NamedSharding

    rng = np.random.default_rng([1409, 1])
    q, k = (unit(rng.standard_normal((2, 2, 1, 2))) for _ in range(2))
    v = rng.standard_normal((2, 2, 1, 2))
    gates = (0.1 * sigmoid(rng.standard_normal((2, 2, 1))),) + (
        (sigmoid(rng.standard_normal((2, 2, 1))), sigmoid(rng.standard_normal((2, 2, 1)))) if form == 'titans'
        else ())
    mesh = jax.make_mesh((2,), ('data',), axis_types=(AxisType.Explicit,), devices=jax.devices()[:2])
    rows, whole = NamedSharding(mesh, P('data')), NamedSharding(mesh, P())
    fn = functools.partial(FORMS[form][0], chunk=2)

    def total(tokens, gates, affine):
        out, state = fn(*tokens, *affine, *gates)
        return jnp.sum(out) + sum(jnp.sum(x) for x in state)

    with jax.set_mesh(mesh):
        tokens = [jax.device_put(hosted(x), whole if name in replicated else rows)
                  for name, x in zip(('query', 'key', 'value'), (q, k, v), strict=True)]
        gates = [jax.device_put(hosted(x), rows) for x in gates]
        affine = [jax.device_put(hosted(x), whole) for x in (np.ones((1, 2)), np.zeros((1, 2)))]
        out, state = jax.jit(fn)(*tokens, *affine, *gates)
        gradients = jax.jit(jax.grad(total, argnums=(0, 1, 2)))(tokens, gates, affine)
    assert out.sharding.spec[0] == 'data', out.sharding
    assert all(np.isfinite(np.asarray(x)).all() for x in (out, *state, *jax.tree.leaves(gradients)))


@pytest.mark.mesh(devices=2)
def test_ttt_runs_in_a_checked_shard_map_with_its_scale_alone_varying():
    """TTT-Linear in a checked `shard_map` over two devices, every operand replicated and its query scale
    alone each device's own (1/4 and 1/2): its outputs finite, and its final states replicated (`out_specs`
    `P()`), those of one scale on both devices exactly, the scale reading no state (train 1455's review). Its
    values are the oracle test's."""
    *given, _ = operands('ttt', jnp.float32)(1)
    tokens, affine, gates, states = tuple(given[:3]), tuple(given[3:5]), tuple(given[5:-1]), given[-1]

    def local(scale):
        out, state = ttt_linear(*tokens, *affine, *gates, state=states, chunk=CHUNK, scale=scale[0])
        return out[None], state

    mesh = Mesh(np.asarray(jax.devices()[:2]), ('data',))
    mapped = jax.shard_map(local, mesh=mesh, in_specs=P('data'), out_specs=(P('data'), P()), check_vma=True)
    out, found = jax.jit(mapped)(jnp.asarray([0.25, 0.5], jnp.float32))
    # The same program with one scale on both devices: its states the split call's, bit for bit.
    _, expected = jax.jit(mapped)(jnp.asarray([0.25, 0.25], jnp.float32))
    # Each device's readout the query's shape, in its dtype.
    assert (out.shape, out.dtype) == ((2, *jnp.shape(tokens[0])), jnp.result_type(tokens[0]))
    assert np.isfinite(np.asarray(out)).all()
    for got, want in zip(jax.tree.leaves(found), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize('form', sorted(FORMS))
def test_the_mini_batches_run_on_an_explicit_mesh_with_the_affine_alone_split_by_heads(form):
    """Each form under `jit` over two devices of an Explicit mesh, two heads, the inner loss's affine alone
    split by heads across them (the tokens, rates and entering states replicated): its outputs and final
    states the unsplit call's shapes and dtypes and finite, its output placed by heads, every operand placed
    with the affine's heads (`titans.placed_alike`), where replicated states beside the head-split writes
    the affine placed were refused (train 1430's review). Its values are the oracle test's (a head-split
    program's fusions round apart from the unsplit one's, a unit in the last place)."""
    from jax.sharding import AxisType, NamedSharding

    *given, _ = operands(form, jnp.float32)(1)
    tokens, affine, gates, states = tuple(given[:3]), tuple(given[3:5]), tuple(given[5:-1]), given[-1]
    fn = functools.partial(FORMS[form][0], chunk=CHUNK)
    expected = fn(*tokens, *affine, *gates, state=states)
    mesh = jax.make_mesh((2,), ('heads',), axis_types=(AxisType.Explicit,), devices=jax.devices()[:2])
    whole, heads = NamedSharding(mesh, P()), NamedSharding(mesh, P('heads'))
    with jax.set_mesh(mesh):
        placed = [jax.device_put(x, whole) for x in (*tokens, *gates)]
        split = [jax.device_put(x, heads) for x in affine]
        found = jax.jit(fn)(*placed[:3], *split, *placed[3:], state=jax.device_put(states, whole))
    assert found[0].sharding.spec[2] == 'heads', found[0].sharding
    assert ([(x.shape, x.dtype) for x in jax.tree.leaves(found)]
            == [(x.shape, x.dtype) for x in jax.tree.leaves(expected)])
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(found))


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize('split', ['rows', 'rows, a replicated state', 'affine', 'a state leaf', 'the query',
                                   'the mask'])
@pytest.mark.parametrize('form', sorted(FORMS))
def test_the_mini_batches_run_in_a_checked_shard_map_whatever_varies(form, split):
    """Each form in a checked `shard_map` over two devices, each device's operands its own where they vary
    over the axis (its rows; its affine; its entering state's second leaf) and the same where they do not
    (the affine, the tokens and gates, the entering states, implicit zeros or replicated): every output and
    final state the device's call's shape and dtype and finite, and every operand's gradient finite, its
    scans' carries typed as every operand varies (`sharding.varying_like`), where an invariant carry was
    refused (train 1403's review), and one typed as the tokens and gates alone beside an affine or a state
    leaf that varies (a head review), or beside a token mask that varies alone (one device's tokens live, the
    other's absent: a head review). Its values are the oracle test's. An odd seed's, whose entering states are
    not zero (a zero leaf doubled is no leaf that varies)."""
    *given, _ = operands(form, jnp.float32)(1)
    tokens, affine, gates, states = tuple(given[:3]), tuple(given[3:5]), tuple(given[5:-1]), given[-1]

    def on(device):
        """The device's operands, `(tokens, affine, gates, state, valid)`."""
        if split.startswith('rows'):
            rows = slice(device, device + 1)
            state = None if split == 'rows' else tuple(s[:1] for s in states)
            return tuple(x[rows] for x in tokens), affine, tuple(x[rows] for x in gates), state, None
        if split == 'affine':
            return tokens, tuple(x * (1 + device) for x in affine), gates, None, None
        if split == 'the query':
            return (tokens[0] * (1 + device), *tokens[1:]), affine, gates, None, None
        if split == 'the mask':
            return tokens, affine, gates, None, np.full(tokens[0].shape[:2], device == 0)
        return tokens, affine, gates, (states[0], states[1] * (1 + device)), None

    each = [on(device) for device in range(2)]
    leaves, structure = jax.tree.flatten(each[0])
    varies = [not np.array_equal(a, b) for a, b in zip(leaves, jax.tree.leaves(each[1]), strict=True)]
    args = [jnp.stack(pair) if vary else pair[0]
            for pair, vary in zip(zip(*(jax.tree.leaves(e) for e in each), strict=True), varies, strict=True)]

    def call(tokens, affine, gates, state, valid):
        return FORMS[form][0](*tokens, *affine, *gates, state=state, chunk=CHUNK, valid=valid)

    def local(*leaves):
        mine = [x[0] if vary else x for x, vary in zip(leaves, varies, strict=True)]
        return jax.tree.map(lambda x: x[None], call(*jax.tree.unflatten(structure, mine)))

    mesh = Mesh(np.asarray(jax.devices()[:2]), ('data',))
    specs = tuple(P('data') if vary else P() for vary in varies)
    mapped = jax.shard_map(local, mesh=mesh, in_specs=specs, out_specs=P('data'), check_vma=True)
    found = jax.jit(mapped)(*args)
    shapes = jax.eval_shape(call, *each[0])
    assert ([(x.shape, x.dtype) for x in jax.tree.leaves(found)]
            == [((2, *x.shape), x.dtype) for x in jax.tree.leaves(shapes)])
    if split == 'the mask':
        # The idle device's states leave as they entered (zeros).
        for state in jax.tree.leaves(found[1]):
            np.testing.assert_array_equal(np.asarray(state)[1], np.zeros(state.shape[1:], state.dtype))

    def total(*leaves):
        return sum(jnp.sum(x) for x in jax.tree.leaves(mapped(*leaves)))

    # Every float operand's gradient (a token mask has none).
    floats = tuple(index for index, x in enumerate(args) if jnp.issubdtype(jnp.result_type(x), jnp.inexact))
    gradients = jax.jit(jax.grad(total, argnums=floats))(*args)
    assert all(np.isfinite(np.asarray(x)).all() for x in (*jax.tree.leaves(found), *gradients))
