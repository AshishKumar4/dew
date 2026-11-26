"""The preconditioned delta rules (`dew.nn.kernels.preconditioned`, rows 1458 and 1459: fla's PGDN and PKDA)
on the chunked op and the decode step: output, final state, final ATK moment and every operand's gradient,
the key heads' centers among them, on par with fla's recurrence a token at a time
(`naive_recurrent_precond_gated_delta_rule`, its PKDA twin) from float64 (check 1), one sequence and packed
documents, a decay a head's and a key dimension's, value heads sharing their key head's moment, in fp32; and
ATK's walk on the recurrence template beside it, packed over spans; and the squash's derivative where it
saturates. A 16-bit call carries fla's own roundings, which no walk runs: it is held to fla's chunked release
(row 12696's recorded arm)."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import INTERPRETS, interprets
from reference_error import assert_as_exact, with_gradients
from test_recurrence_chunk import assert_packed_on_par, hosted, marked, sigmoid, softplus

from dew.nn.kernels import generation
from dew.nn.kernels.preconditioned import (
    CENTER,
    preconditioned_delta_rule,
    preconditioned_keys,
    preconditioned_step,
)
from dew.nn.kernels.recurrence import Hooks
from dew.nn.kernels.recurrence_chunk import Chunk, chunk
from dew.nn.kernels.templates.recurrence import chunk_forward, mainloop_backward, template
from dew.nn.kernels.templates.recurrence.mainloop import Tiles
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
"""The delta rule erasing along reads of its own on the recurrence template (`chunk_forward.forward`'s
`reads`): a whole chunk of its 64 tokens and a part one, two value heads a key head's."""
NARROW = Tiles(32, 1, chunk=64)
"""The template's tiles a read erase's launches take, interpreted."""
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


def read_erased(query, key, value, log_decay, beta, state, reads):
    """The delta rule erasing along reads of its own (`recurrence.Hooks.erase` 'read') a token at a time in
    the operands' dtype, `S' = (I - beta k b^T) exp(g) S + beta k v^T`, read `q^T S' Dk^-1/2`, its decay a
    head's or a key dimension's (`log_decay` `[B, S, HV]` or `[B, S, HV, Dk]`), each key head's queries, keys
    and reads its value heads': `(output, final memory)`."""
    group = value.shape[2] // key.shape[2]
    query, key, reads = (jnp.repeat(x, group, axis=2) for x in (query, key, reads))

    def one(memory, token):
        q, k, v, g, b, r = token
        memory = memory * (jnp.exp(g)[..., None] if g.ndim == 3 else jnp.exp(g)[..., None, None])
        corrected = v - jnp.einsum('bhk,bhkv->bhv', r, memory)
        memory = memory + jnp.einsum('bhk,bhv->bhkv', b[..., None] * k, corrected)
        return memory, jnp.einsum('bhk,bhkv->bhv', q * q.shape[-1] ** -0.5, memory)

    steps = tuple(jnp.moveaxis(x, 1, 0) for x in (query, key, value, log_decay, beta, reads))
    final, out = jax.lax.scan(one, state, steps)
    return jnp.moveaxis(out, 0, 1), final


def read_erased_with_gradients(*operands):
    """`read_erased`'s output and final memory and every operand's gradient under the drawn cotangents
    (`operands`' last two)."""
    *given, d_out, d_final = operands
    found, pull = jax.vjp(read_erased, *given)
    return found, pull((d_out, d_final))


def on_the_mainloop(query, key, value, log_decay, beta, state, reads, d_out, d_final):
    """`read_erased_with_gradients` on the template's mainloop (`chunk_forward.kept_forward`'s `reads` and
    its fast backward's, `mainloop_backward.kept_backward`), its accurate fp32 forms, interpreted on an
    A100's bundle (`mma.sync`): an fp32 call's reference under the accurate forms is the recurrence a token
    at a time in fp32."""
    target = generation.gpu_target('sm80')
    common = {'tiles': NARROW, 'family': template.bundle(target, jnp.float32), 'interpret': True}
    with generation.assumed(target, interpreted=True):
        out, final, (inverse, kept) = chunk_forward.kept_forward(query, key, value, log_decay, beta, state,
                                                                 reads=reads, **common)
        grads, _ = mainloop_backward.kept_backward(query, key, value, log_decay, beta, state, d_out, d_final,
                                                   inverse=inverse, kept=kept, reads=reads, **common)
    dq, dk, dv, dg, dbeta, dh0, *_, d_reads = grads
    return (out, final), (dq, dk, dv, dg, dbeta, dh0, d_reads)


def read_erased_forward(query, key, value, log_decay, beta, state, reads):
    """`read_erased_on_the_template`'s output and final memory, its forward alone, under a decay a head
    (`log_decay` `[B, S, HV]`) or a key dimension (`[B, S, HV, Dk]`, its decay the head's at every key)."""
    return read_erased_on_the_template(query, key, value, log_decay, beta, state, reads)


def with_key_decays(query, key, value, log_decay, beta, state, reads):
    """`read_erased_forward` under a decay a key dimension, each key's the head's (`log_decay` `[B, S, HV]`
    spread over the keys)."""
    keyed = jnp.broadcast_to(log_decay[..., None], (*log_decay.shape, key.shape[-1]))
    return read_erased_forward(query, key, value, keyed, beta, state, reads)


def sixteen_bit_forward(seed):
    """`sixteen_bit_read_erase`'s draw of a seed, its cotangents gone."""
    return sixteen_bit_read_erase(seed)[:-1]


def but_state(fn):
    """`with_gradients(fn)` for `fn(query, key, value, log_decay, beta, state, reads)`, its entering memory
    closed over: every input's gradient but its."""
    def run(query, key, value, log_decay, beta, state, reads, cotangent):
        out, pull = jax.vjp(lambda q, k, v, g, b, r: fn(q, k, v, g, b, state, r), query, key, value,
                            log_decay, beta, reads)
        return out, *pull(cotangent)
    return run


def along_its_keys(query, key, value, log_decay, beta, state):
    """`read_erased_forward` reading along its own keys: the keys' gradient their reads' and their own
    summed by the derivative."""
    return read_erased_forward(query, key, value, log_decay, beta, state, key)


def kimis(query, key, value, log_decay, beta, state):
    """Kimi's delta attention (the delta rule erasing along its keys) under a decay a key dimension on the
    template, as `read_erased_on_the_template` runs the read erase."""
    fast = dataclasses.replace(generation.numerics(), forms='fast')
    with (generation.assumed("sm90" if INTERPRETED else None, interpreted=INTERPRETED),
          generation.run_precision(fast), generation.chosen(gated_delta_rule="mosaic_gpu")):
        return chunk(Hooks(decay='vector', scale=query.shape[-1] ** -0.5),
                     Chunk(query, key, value, log_decay=log_decay, beta=beta, state=state))


def recurrence_along_its_keys(query, key, value, log_decay, beta, state):
    """`read_erased` reading along its own keys: Kimi's delta attention a token at a time."""
    return read_erased(query, key, value, log_decay, beta, state, key)


def key_decayed_along_its_keys(seed):
    """`sixteen_bit_read_erase`'s draw of a seed with each key its own log decay, a sigmoid's log about the
    head's, its reads gone (the call reads along its keys, normalized: its reads declared unit)."""
    q, k, v, g, beta, state, reads, cotangents = sixteen_bit_read_erase(seed)
    k = reads
    normal = np.random.default_rng([1459, 4, seed]).standard_normal
    keyed = hosted(-softplus(-(normal((*g.shape, k.shape[-1])) + 3.0)))
    return q, k, v, keyed, beta, state, cotangents


def read_erase_operands(seed):
    """A seed's fp32 operands of `TEMPLATED`'s shape, as PGDN gives its delta rule: its reads unit keys, its
    keys the reads scaled a channel at a time within ATK's squash's range (2/3 to 3/2), log decays a
    sigmoid's log, write gates a sigmoid's, an entering memory, and the cotangents."""
    rows, length, key_heads, value_heads, keys, values = TEMPLATED
    normal = np.random.default_rng([1458, 2, seed]).standard_normal
    reads = normal((rows, length, key_heads, keys))
    reads = reads / np.sqrt(np.sum(reads ** 2, axis=-1, keepdims=True))
    written = reads * np.exp(np.log(1.5) * np.tanh(normal(reads.shape)))
    gates = (rows, length, value_heads)
    drawn = (normal((rows, length, key_heads, keys)), written, normal((rows, length, value_heads, values)),
             -softplus(-(normal(gates) + 3.0)), sigmoid(normal(gates)),
             normal((rows, value_heads, keys, values)), reads, normal((rows, length, value_heads, values)),
             normal((rows, value_heads, keys, values)))
    return tuple(hosted(x) for x in drawn)


def templated(*operands, kernel: str = "mosaic_gpu", target: str = "sm90"):
    """`chunked` with the delta rule on `kernel` (the recurrence template's where unnamed) under the run's
    fast forms, where an fp32 call's products round as fla's do (`recurrence_chunk.fast_rounding`), at
    `target`'s facts (an H100's)."""
    fast = dataclasses.replace(generation.numerics(), forms='fast')
    with (generation.assumed(target if INTERPRETED else None, interpreted=INTERPRETED),
          generation.run_precision(fast), generation.chosen(gated_delta_rule=kernel)):
        return chunked(*operands)


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


def test_a_read_erase_on_the_mainloop_is_as_exact_as_its_recurrence_a_token_at_a_time():
    """The delta rule erasing along reads of its own on the template's mainloop (PGDN's: the system's rows
    and the weights the reads', its writes the keys'), its accurate fp32 forms: the output, the final
    memory and every operand's gradient, the reads' among them, on par with the recurrence a token at a time
    in fp32 from float64 (check 1)."""
    assert_as_exact(jax.jit(on_the_mainloop), jax.jit(read_erased_with_gradients), read_erase_operands,
                    "a read erase on the mainloop", seeds=SEEDS)


def read_erased_on_the_template(query, key, value, log_decay, beta, state, reads, *, group: int = 1):
    """The chunked op's delta rule erasing along `reads` (`recurrence.Hooks.erase` 'read') on the recurrence
    template under the run's fast forms at an H100's facts, its queries, keys and reads repeated `group`
    times over their heads first (each value head a key head of its own where `group` is the call's)."""
    query, key, reads = (jnp.repeat(x, group, axis=2) for x in (query, key, reads))
    # Its reads unit keys (`read_erase_operands`'), declared so: the template takes no other read erase.
    hooks = Hooks(decay='vector' if log_decay.ndim == 4 else 'scalar', erase='read', unit_reads=True,
                  scale=query.shape[-1] ** -0.5)
    fast = dataclasses.replace(generation.numerics(), forms='fast')
    with (generation.assumed("sm90" if INTERPRETED else None, interpreted=INTERPRETED),
          generation.run_precision(fast), generation.chosen(gated_delta_rule="mosaic_gpu")):
        return chunk(hooks, Chunk(query, key, value, log_decay=log_decay, beta=beta, erase_read=reads,
                                  state=state))


def sixteen_bit_read_erase(seed):
    """`read_erase_operands`' draw of a seed as a 16-bit call takes it: its queries, keys, values, reads and
    the output's cotangent bf16."""
    q, k, v, g, beta, state, reads, d_out, d_final = read_erase_operands(seed)
    q, k, v, reads, d_out = (x.astype(jnp.bfloat16) for x in (q, k, v, reads, d_out))
    return q, k, v, g, beta, state, reads, (d_out, d_final)


def test_a_grouped_read_erase_on_the_template_is_as_exact_as_its_heads_apart():
    """A 16-bit delta rule erasing along reads of its own through the chunked op on the template, its keys
    and reads a key head's over two value heads each (`TEMPLATED`), on par from float64 with the same call
    whose keys and reads are repeated a value head's each before it (check 1, the truth its recurrence a
    token at a time): its output, final memory and every operand's gradient, the key heads' summed over
    their value heads by the repeat's own derivative. A value head reading another key head's keys or
    reads, or a gradient summed onto the wrong key head, moves its leaves far past the repeated call's,
    whose arithmetic is the grouped call's (head review of 410aee86f)."""
    group = TEMPLATED[3] // TEMPLATED[2]
    assert_as_exact(jax.jit(with_gradients(read_erased_on_the_template)),
                    jax.jit(with_gradients(functools.partial(read_erased_on_the_template, group=group))),
                    sixteen_bit_read_erase, "a grouped read erase on the template", seeds=SEEDS,
                    truth_fn=jax.jit(with_gradients(read_erased)))


@pytest.mark.parametrize(('kernel', 'taken'), [('mosaic_gpu', 'mosaic_gpu'), ('fla', 'xla')],
                         ids=['template', 'port'])
def test_a_read_erase_under_a_decay_a_head_trains_on_the_template(kernel, taken):
    """The delta rule's selector gives PGDN's 16-bit training call the template it names (its keys' erase
    along reads of their own, `chunk_forward.forward`'s `reads`), where fla's port, which takes no reads,
    hands over to the portable math: the kernels its traced routes name, beside the portable tangent's."""
    drawn = operands(per_key=False, shape=TEMPLATED)(0)[:-1]
    # A 16-bit call's queries, keys and values, as a layer's projections give them.
    drawn = (*(x.astype(jnp.bfloat16) for x in drawn[:3]), *drawn[3:])

    def loss(*given):
        return templated(*given, kernel=kernel)[0].astype(jnp.float32).sum()

    with generation.routes() as found:
        jax.eval_shape(jax.grad(loss, argnums=(0, 1, 2)), *drawn)
    # Beside a kernel's routes, its derivative's portable tangent's (`_per_product`).
    kernels = {route.kernel for route in found if route.choice == generation.Choice.GATED_DELTA_RULE}
    assert kernels == {taken, 'xla'}, kernels


def test_a_read_erase_along_its_keys_under_distinct_key_decays_is_kimis_delta_attention():
    """The delta rule erasing along reads of its own under a decay a key dimension, each key its own decay
    (16-bit), reading along its own keys, is Kimi's delta attention: on the template
    its forward (the system's rows from the reads' stream, its weights from the reads' launch and its
    decayed keys from the keys', `chunk_forward._key_weighted`) the same, bit for bit, as the template's
    forward of the delta rule erasing along its keys, which check 1 holds to fla's recorded sites
    (`test_recurrence_mosaic`): a read's decay taken from another key's, or the reads' path decaying them
    otherwise than the keys', moves the output past it (head review of 675cd1ecc). A swap of the system's
    rows and columns, the same here, `test_a_read_erase_under_a_decay_a_key_dimension_on_the_template_is_its
    _head_decays` sees."""
    q, k, v, g, beta, state, _ = sixteen_bit_forward(0)
    # Each key its own log decay, a sigmoid's log about the head's.
    normal = np.random.default_rng([1459, 3]).standard_normal
    keyed = hosted(-softplus(-(normal((*g.shape, k.shape[-1])) + 3.0)))
    read = read_erased_forward(q, k, v, keyed, beta, state, k)
    fast = dataclasses.replace(generation.numerics(), forms='fast')
    with (generation.assumed("sm90" if INTERPRETED else None, interpreted=INTERPRETED),
          generation.run_precision(fast), generation.chosen(gated_delta_rule="mosaic_gpu")):
        tied = chunk(Hooks(decay='vector', scale=q.shape[-1] ** -0.5),
                     Chunk(q, k, v, log_decay=keyed, beta=beta, state=state))
    for found, expected in zip(read, tied, strict=True):
        np.testing.assert_array_equal(np.asarray(found, np.float32), np.asarray(expected, np.float32))


@INTERPRETS
@pytest.mark.parametrize('target', ['sm90', 'sm86'])
def test_a_key_decays_read_erase_trains_on_the_template(target):
    """PKDA's 16-bit training call naming the template, traced through the public gradient, the template's
    on an H100 and on an sm86 (99 KB a block) alike: the kernels its traced routes name, beside the portable
    tangent's. Its key memory launch landing its inverse's gradient through its key block's landing, its
    reads hold whole value rows on a 99 KB part (row 12890: a `[C, C]` landing of its own walked them, its
    memory sums carried, and spilled; the part refused)."""
    drawn = operands(per_key=True, shape=TEMPLATED)(0)[:-1]
    drawn = (*(x.astype(jnp.bfloat16) for x in drawn[:3]), *drawn[3:])

    def loss(*given):
        return templated(*given, target=target)[0].astype(jnp.float32).sum()

    gradient = functools.partial(jax.eval_shape, jax.grad(loss, argnums=(0, 1, 2)), *drawn)
    with generation.routes() as found:
        gradient()
    assert {route.kernel for route in found if route.choice == generation.Choice.GATED_DELTA_RULE} == {
        'mosaic_gpu', 'xla'}


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


KEY_DECAY_CASES = {
    'head decays': (but_state(with_key_decays), but_state(read_erased_forward), but_state(read_erased),
                    sixteen_bit_read_erase),
    'along its keys': (with_gradients(along_its_keys), with_gradients(kimis),
                       with_gradients(recurrence_along_its_keys), key_decayed_along_its_keys),
}
"""The template's read erase under a decay a key dimension (PKDA's), its reference and truth (its
recurrence a token at a time in float64) and draw: each key's decay the head's against the same call under
that decay a head, every leaf but the entering memory's gradient; each key its own decay, reading along its
keys, against Kimi's delta attention, every leaf."""


@pytest.mark.parametrize('case', KEY_DECAY_CASES)
def test_a_key_decays_read_erase_on_the_template_is_as_exact_as_its_reference(case):
    """The delta rule erasing along reads of its own under a decay a key dimension (PKDA's) on the template,
    16-bit (the system's rows the reads' in its own pairs and the intra walk, its weights the reads' and its
    decayed keys the keys' apart, `chunk_forward._paired` and `_key_weighted`; its backward's the same,
    `key_backward.gradients`' reads), its output, final memory and operands' gradients on par from float64
    with a reference (check 1): 'head decays', each key's decay the head's, against the same call under that
    decay a head, whose read erase check 1 holds to the recurrence at the accurate forms: a system whose rows
    and columns trade the reads and the keys, weights formed from the keys, or a gradient landing on the
    other, lands far past it; every leaf but the entering memory's gradient, which a decay a key dimension's
    reverse walk forms 1.87 times as far from float64 as a decay a head's on Kimi's delta attention itself
    against the gated delta net (scratch jsf-beta2, no read erase: check 1's leaf `[6]`). 'along its keys',
    each key its own decay, reading along its keys, against Kimi's delta attention on the template, every
    leaf, the keys' gradient its reads' and its own summed, which check 1 holds to fla's recorded sites
    (`test_recurrence_mosaic_gradient`): a read's decay taken from another key's, or a term lost between the
    reads' gradient and the keys', moves it far past. fla's recorded PKDA sites hold it once M-JS-266b's
    fixture lands (rows 1459, 12696)."""
    dew, reference, truth, draw = KEY_DECAY_CASES[case]
    assert_as_exact(jax.jit(dew), jax.jit(reference), draw,
                    f"a key decay's read erase on the template: {case}", seeds=SEEDS, truth_fn=jax.jit(truth))
