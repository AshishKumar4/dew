"""The preconditioned delta rules (`dew.nn.kernels.preconditioned`, rows 1458 and 1459: fla's PGDN and PKDA)
on the chunked op and the decode step: output, final state, final ATK moment and every operand's gradient,
the key heads' centers among them, on par with fla's recurrence a token at a time
(`naive_recurrent_precond_gated_delta_rule`, its PKDA twin) from float64 (check 1), one sequence and packed
documents, a decay a head's and a key dimension's, value heads sharing their key head's moment, in fp32; and
ATK's walk on the recurrence template beside it, packed over spans; and the squash's derivative where it
saturates. A 16-bit call carries fla's own roundings, which no walk runs: it is held to fla's chunked release
(row 12696's recorded arm)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import interprets
from reference_error import assert_as_exact, with_gradients
from test_recurrence_chunk import assert_packed_on_par, hosted, marked, sigmoid, softplus

from dew.nn.kernels import generation
from dew.nn.kernels.preconditioned import (
    CENTER,
    preconditioned_delta_rule,
    preconditioned_keys,
    preconditioned_step,
)
from dew.nn.precision import l2norm

SHAPE = (2, 40, 8, 16, 16, 8)
"""Rows, tokens, key heads, value heads, keys and values: two whole chunks of 16 tokens and a part one,
each pair of value heads sharing a key head's moment; eight key heads, so a draw's centers' gradient, each a
key head's sum over every token and key, holds eight (two held its pooled ratio's interval past 1.5 at 80
draws)."""
CHUNK = 16
WALKED = (2, 160, 8, 8, 16, 8)
"""ATK's walk on the template (`diagonal.diagonal_scan` 'mosaic_gpu'): its 128 channels a program's
(eight key heads of 16 keys), 160 tokens three spans of an interpreted walk's 64."""
DOCUMENTS = ((0, 11, 30), (5, 17))
"""Each row's packed documents' first tokens."""
WALKED_DOCUMENTS = ((0, 50, 130), (20, 64, 100))
"""Each row's packed documents' first tokens over the walk's spans: one starting at a span's first token,
others within spans, a document crossing two."""
SEEDS = 16
"""Each site's declared seeds."""
TEMPLATED = (2, 80, 2, 4, 32, 32)
"""PGDN on the recurrence template (its keys erasing along reads of their own, `chunk_forward.forward`'s
`reads`): a whole chunk of its 64 tokens and a part one, two value heads a key head's."""
TEMPLATED_DOCUMENTS = ((0, 30, 70), (5, 64))
"""Each row's packed documents' first tokens over the template's chunks: one starting at a chunk's first
token, others within chunks, a document crossing two."""
INTERPRETED = interprets()


def tokenwise(query, key, value, log_decay, beta, atk_decay, atk_beta, state, atk_state, center, starts=None):
    """fla's recurrence a token at a time (naive_recurrent_precond_gated_delta_rule, its PKDA twin) in at
    least fp32, the queries and keys l2-normalized (`precision.l2norm`, as Dew's) and the queries scaled by
    their width, each key head's moment and center its value heads', a packed document's first token
    entering both states at zero: the output in the values' dtype and `(final state, final moment)`."""
    work = jnp.promote_types(jnp.result_type(query, key, value), jnp.float32)
    group = value.shape[2] // key.shape[2]
    query, key = (jnp.repeat(l2norm(x).astype(work), group, axis=2) for x in (query, key))
    atk_decay, atk_beta = (jnp.repeat(jnp.asarray(x, work), group, axis=2) for x in (atk_decay, atk_beta))
    centers = jnp.repeat(jnp.asarray(center, work), group)[:, None]
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
        return (memory, moment), jnp.einsum('bhk,bhkv->bhv', q * q.shape[-1] ** -0.5, memory)

    steps = (*(jnp.moveaxis(jnp.asarray(x, work), 1, 0)
               for x in (query, key, value, log_decay, beta, atk_decay, atk_beta)), first.T)
    entering = (jnp.asarray(state, work), jnp.repeat(jnp.asarray(atk_state, work), group, axis=1))
    (memory, moment), out = jax.lax.scan(one, entering, steps)
    return jnp.moveaxis(out, 0, 1).astype(value.dtype), (memory, moment[:, ::group])


def chunked(query, key, value, log_decay, beta, atk_decay, atk_beta, state, atk_state, center, starts=None):
    """The chunked rule on `tokenwise`'s operands."""
    return preconditioned_delta_rule(query, key, value, log_decay, beta, atk_decay, atk_beta, state,
                                     atk_state, starts, center=center, chunk_size=CHUNK)


def walked(*operands, starts=None):
    """`chunked` with ATK's walk on the recurrence template (`diagonal_scan` 'mosaic_gpu'), interpreted off a
    GPU at an H100's facts."""
    with (generation.assumed("sm90" if INTERPRETED else None, interpreted=INTERPRETED),
          generation.chosen(diagonal_scan="mosaic_gpu")):
        return chunked(*operands, starts=starts)


def templated(*operands, starts=None, kernel: str = "mosaic_gpu"):
    """`chunked` with the delta rule on `kernel` (the recurrence template's where unnamed), interpreted off a
    GPU at an H100's facts."""
    with (generation.assumed("sm90" if INTERPRETED else None, interpreted=INTERPRETED),
          generation.chosen(gated_delta_rule=kernel)):
        return chunked(*operands, starts=starts)


def operands(per_key: bool, shape: tuple[int, ...] = SHAPE):
    """A seed's fp32 operands as fla's layers form them (`hosted`), of `shape` (`SHAPE`'s order): the log
    decays `log sigmoid` of normals (the memory's near one; a key dimension's each where `per_key`), ATK's
    decays near one and its gates a sigmoid's, the write gates a sigmoid's, entering states and moments (a
    moment nonnegative), the key heads' centers about fla's default, and the cotangents of the output and
    both final states."""
    rows, length, key_heads, value_heads, keys, values = shape

    def at(seed):
        normal = np.random.default_rng([1458, per_key, length, seed]).standard_normal
        gates = (rows, length, value_heads)
        keyed = (rows, length, key_heads)
        q, k = (normal((*keyed, keys)) for _ in range(2))
        v = normal((rows, length, value_heads, values))
        log_decay = -softplus(-(normal((*gates, keys) if per_key else gates) + 3.0))
        atk_decay = -softplus(-(normal(keyed) + 4.0))
        state = normal((rows, value_heads, keys, values))
        atk_state = np.abs(normal((rows, key_heads, keys))) * 0.1
        center = CENTER + 0.1 * normal((key_heads,))
        finals = (hosted(normal(state.shape)), hosted(normal(atk_state.shape)))
        cotangent = (hosted(normal(v.shape)), finals)
        gated = (log_decay, sigmoid(normal(gates)), atk_decay, sigmoid(normal(keyed)))
        drawn = (q, k, v, *gated, state, atk_state, center)
        return (*(hosted(x) for x in drawn), cotangent)
    return at


@pytest.mark.parametrize(('per_key', 'packed'), [(False, False), (False, True), (True, False), (True, True)],
                         ids=['pgdn', 'pgdn packed', 'pkda', 'pkda packed'])
def test_the_chunked_rule_is_as_exact_as_flas_recurrence_a_token_at_a_time(per_key, packed):
    """The chunked preconditioned rule's output, final state and ATK moment and the gradients of every
    operand, both entering states and the centers on par with fla's recurrence a token at a time from
    float64 (check 1), a decay a head's (PGDN) or a key dimension's (PKDA), one sequence or packed
    documents, each document's first token reading neither state before it."""
    label = f"preconditioned {'kda' if per_key else 'gdn'}"
    assert_packed_on_par(chunked, tokenwise, operands(per_key), label, packed=packed, documents=DOCUMENTS,
                         length=SHAPE[1], seeds=SEEDS)


@pytest.mark.parametrize('packed', [False, True], ids=['one sequence', 'packed'])
def test_pgdn_on_the_template_is_as_exact_as_flas_recurrence_a_token_at_a_time(packed):
    """PGDN's delta rule on the recurrence template, its keys erasing along reads of their own (the system's
    rows and the weights the normalized keys', its writes the preconditioned keys'): the output, final state
    and moment and every operand's gradient on par with fla's recurrence a token at a time from float64
    (check 1), over one sequence or packed documents crossing its chunks."""
    assert_packed_on_par(templated, tokenwise, operands(per_key=False, shape=TEMPLATED),
                         "preconditioned gdn on the template", packed=packed, documents=TEMPLATED_DOCUMENTS,
                         length=TEMPLATED[1], seeds=SEEDS)


@pytest.mark.parametrize(('per_key', 'kernel', 'taken'), [(False, 'mosaic_gpu', 'mosaic_gpu'),
                                                         (False, 'fla', 'xla'), (True, 'mosaic_gpu', 'xla')],
                         ids=['pgdn template', 'pgdn port', 'pkda template'])
def test_a_read_erase_takes_the_template_under_a_decay_a_head_alone(per_key, kernel, taken):
    """The delta rule's selector gives PGDN's training call the template it names (its keys' erase along
    reads of their own, `chunk_forward.forward`'s `reads`), where fla's port, which takes no reads, and
    PKDA's decay a key dimension, which the template's read erase does not take yet, hand over to the
    portable math: each traced route its kernel's."""
    drawn = operands(per_key, TEMPLATED)(0)[:-1]

    def loss(*given):
        return templated(*given, kernel=kernel)[0].astype(jnp.float32).sum()

    with generation.routes() as found:
        jax.eval_shape(jax.grad(loss, argnums=(0, 1, 2)), *drawn)
    assert {route.kernel for route in found if route.choice == generation.Choice.GATED_DELTA_RULE} == {taken}


def test_atks_walk_on_the_template_is_as_exact_over_packed_spans():
    """ATK's moment walked by the recurrence template (128 channels, three spans of the interpreted walk's
    64 tokens) under packed documents crossing its spans: the chunked rule's output, final state and moment
    and every gradient, the centers' among them, on par with fla's recurrence a token at a time (check 1)."""
    starts = marked(WALKED_DOCUMENTS, WALKED[1])
    assert_as_exact(jax.jit(with_gradients(lambda *x: walked(*x, starts=starts))),
                    jax.jit(with_gradients(lambda *x: tokenwise(*x, starts=starts))),
                    operands(per_key=False, shape=WALKED), "preconditioned gdn on ATK's template walk",
                    seeds=SEEDS)


def stepped(query, key, value, log_decay, beta, atk_decay, atk_beta, state, atk_state, center):
    """The decode step (`preconditioned_step`) a token at a time over the sequence."""
    def one(carried, token):
        out, carried = preconditioned_step(*token, *carried, center=center)
        return carried, out

    tokens = tuple(jnp.moveaxis(x, 1, 0) for x in (query, key, value, log_decay, beta, atk_decay, atk_beta))
    final, out = jax.lax.scan(one, (state, atk_state), tokens)
    return jnp.moveaxis(out, 0, 1), final


@pytest.mark.parametrize('per_key', [False, True], ids=['pgdn', 'pkda'])
def test_the_decode_step_is_as_exact_as_flas_recurrence_a_token_at_a_time(per_key):
    """The decode step's outputs, final state and ATK moment and the gradients of every operand, both
    entering states and the centers, a token at a time, on par with fla's recurrence's from float64 (check
    1): its delta rule corrected along the read of its keys (`recurrence.Hooks.erase` 'read',
    `recurrence.update`'s reader)."""
    assert_as_exact(jax.jit(with_gradients(stepped)), jax.jit(with_gradients(tokenwise)), operands(per_key),
                    f"preconditioned {'kda' if per_key else 'gdn'} decode steps", seeds=SEEDS)


def test_the_squash_saturated_keeps_the_centers_and_moments_gradients():
    """ATK's squash saturated, a center of -2^25 (where `1 + |r|` rounds to `|r|`): the preconditioned keys'
    gradients in the centers and in the moment entering nonzero and of float64's signs, where the quotient
    rule's `1 + |r| - |r|` cancelled them to zero (fla's ATK backward's `dr / (1 + |r|)^2`, 2^-50)."""
    rows, length, heads, keys = 1, 4, 2, 8
    normal = np.random.default_rng(1459).standard_normal
    key = jnp.asarray(l2norm(jnp.asarray(normal((rows, length, heads, keys)), jnp.float32)))
    decay = jnp.full((rows, length, heads), -0.1, jnp.float32)
    beta = jnp.full((rows, length, heads), 0.5, jnp.float32)
    state = jnp.full((rows, heads, keys), 0.25, jnp.float32)
    center = jnp.full((heads,), -(2.0 ** 25), jnp.float32)
    cotangent = jnp.asarray(normal((rows, length, heads, keys)), jnp.float32)

    def pulled(center, state):
        return jnp.sum(preconditioned_keys(key, decay, beta, state, center=center)[0] * cotangent)

    found = [np.asarray(x) for x in jax.grad(pulled, argnums=(0, 1))(center, state)]
    with jax.enable_x64(new_val=True):
        wide = (jnp.asarray(x, jnp.float64) for x in (key, decay, beta, state, center, cotangent))
        key64, decay64, beta64, state64, center64, cotangent64 = wide

        def truth(center, state):
            written = preconditioned_keys(key64, decay64, beta64, state, center=center)[0]
            return jnp.sum(written * cotangent64)

        expected = [np.asarray(x) for x in jax.grad(truth, argnums=(0, 1))(center64, state64)]
    for got, want in zip(found, expected, strict=True):
        assert np.all(got != 0)
        np.testing.assert_array_equal(np.sign(got), np.sign(want))


def test_the_squash_saturated_keeps_its_second_derivative():
    """The squash's second derivative where it saturates, `r = 2^25` (`1 + |r|` rounding to `|r|`): forward
    over reverse of `(s(r) - 1)^2 / 2` nonzero and of float64's sign, about `3 2^-100`, where the squash's
    derivative rule returned its primal as a quotient, which the second derivative took through the
    quotient rule again and cancelled to zero (ReviewRecurrenceBatch)."""
    from dew.nn.kernels.preconditioned import _squash

    def loss(r):
        return 0.5 * (_squash(r) - 1.0) ** 2

    def second(r):
        return jax.jvp(jax.grad(loss), (r,), (jnp.ones_like(r),))[1]

    found = float(second(jnp.float32(2.0 ** 25)))
    with jax.enable_x64(new_val=True):
        expected = float(second(jnp.float64(2.0 ** 25)))
    assert found != 0 and np.sign(found) == np.sign(expected)


def test_atks_keys_over_no_tokens_keep_the_moment_and_no_gradient():
    """ATK's keys over a sequence of no tokens: no write keys, the moment leaving the one entering, and the
    centers' and the entering moment's gradients zero and one, where a sum over the tokens' first axis read
    an element of none (ReviewRecurrenceBatch)."""
    key = jnp.zeros((1, 0, 2, 8), jnp.float32)
    gates = jnp.zeros((1, 0, 2), jnp.float32)
    state = jnp.full((1, 2, 8), 0.25, jnp.float32)
    center = jnp.full((2,), CENTER, jnp.float32)

    def leaving(center, state):
        written, moment = preconditioned_keys(key, gates, gates, state, center=center)
        return jnp.sum(written) + jnp.sum(moment)

    (d_center, d_state) = jax.grad(leaving, argnums=(0, 1))(center, state)
    np.testing.assert_array_equal(np.asarray(preconditioned_keys(key, gates, gates, state, center=center)[1]),
                                  np.asarray(state))
    np.testing.assert_array_equal(np.asarray(d_center), 0.0)
    np.testing.assert_array_equal(np.asarray(d_state), 1.0)


@pytest.mark.mesh(devices=2)
def test_the_rule_runs_in_a_shard_map_that_checks_varying_axes():
    """The chunked rule's rows split over a checked `shard_map`'s axis, its centers replicated: its output,
    final states and every operand's gradient finite, and the centers' gradient of the unsplit rule's
    signs, the sum of each shard's (ReviewRecurrenceBatch: a pullback returning a shard's own centers'
    cotangent to a replicated center is refused)."""
    from jax.sharding import Mesh, PartitionSpec

    *given, cotangent = operands(per_key=False)(0)
    rows = PartitionSpec('data')
    specs = (*(rows,) * (len(given) - 1), PartitionSpec())
    mesh = Mesh(np.asarray(jax.devices()[:2]), ('data',))
    mapped = jax.shard_map(chunked, mesh=mesh, in_specs=specs, out_specs=rows, check_vma=True)

    def pulled(rule):
        return jax.jit(lambda *x: jax.vjp(rule, *x)[1](cotangent))(*given)

    found, split, whole = jax.jit(mapped)(*given), pulled(mapped), pulled(chunked)
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves((found, split)))
    np.testing.assert_array_equal(np.sign(np.asarray(split[-1])), np.sign(np.asarray(whole[-1])))
