"""The recurrence template's chunked op (`dew.nn.kernels.recurrence_chunk.chunk`) on its portable math:
every hook combination it takes, a sequence over several chunks and a ragged last one, packed documents
too, held to the same recurrence run a token at a time (the decode step's reference,
`recurrence.step_reference`) under the accuracy gate (tests/reference_error.py), both from float64."""

import dataclasses
import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P
from reference_error import (
    FEWEST_SEEDS,
    assert_as_exact,
    assert_fp32_reduction_bound,
    chain_roundings,
    widened,
)

from dew.nn.kernels import generation, recurrence, recurrence_algebra as algebra, recurrence_chunk
from dew.nn.kernels.recurrence_chunk import Chunk, chunk
from dew.nn.precision import float32_einsum

ROWS, LENGTH, KEY_HEADS, VALUE_HEADS, KEY_WIDTH, VALUE_WIDTH = 2, 70, 1, 2, 16, 8
CHUNK = 16
"""Tokens a chunk: 70 tokens are four whole chunks and a ragged six."""

NORMED_WIDTH = 32
"""The values' width under the epilogue's norm: a row of 8 can have its RMS come near zero, where the
norm magnifies every rounding before it, and the reference compared with itself then does not
resolve (Mamba2's SSD with its norm over packed documents, at 512 seeds)."""

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
    recurrence.Hooks(decay='scalar', rated=True, delta=False, skip=True, output=recurrence.OutputNorm(1e-5)),
    recurrence.Hooks(decay='scalar', rated=True, l2norm=True),
    recurrence.Hooks(decay='vector', delta=False, write_gate=True, skip=True,
                     output=recurrence.OutputNorm(1e-6, 'sigmoid')),
    recurrence.Hooks(decay='scalar', delta=False),
]
"""The prologue's and epilogue's hooks: the gated delta rule's gates and l2 norm, Mamba2's step gates
clipped with the skip, the gated RMS norm after the l2 norm and after a plain scale, Mamba2's SSD as
`chunk_ssd` hands it over (a rated decay, the step its write gate), with its gated norm, a rated decay
under the delta rule and the l2 norm, the norm after the skip, and a decay without any write gate."""

COMPOSED_SEEDS = {(HOOKS[4], False): 16, (HOOKS[4], True): 16, (HOOKS[8], False): 8, (HOOKS[8], True): 8,
                  (HOOKS[6], False): 16, (HOOKS[6], True): 16}
"""The composed-chunks sites' declared seeds (the scalar and the vector decay with the delta rule, the
scalar without; one sequence and packed), sized by `seeds_for` on armada (2026-10-21)."""

PACKED = (False, True)


def chunk_on(kernel: str, hooks, *args, **kwargs):
    """`chunk` on `kernel`, which `generation.chosen` names for the delta rule's selector, and for the rated
    scan's the portable math where `kernel` is it ('xla')."""
    named = {'gated_delta_rule': kernel, **({'ssd': kernel} if kernel == 'xla' else {})}
    with generation.chosen(**named):
        return chunk(hooks, *args, **kwargs)
"""A row of one sequence, and of packed documents."""

SEEDS = dict(zip(((hooks, packed) for hooks in HOOKS + FORMED for packed in PACKED), (
    8, 8, 8, 8, 8, 8, 8, 8, 16, 16, 64, 64, 16, 16, 16, 16, 8, 8, 64, 64, 8, 8, 8, 8,
    64, 64, 128, 128, 16, 8, 16, 16, 64, 64, 32, 32, 32, 32, 8, 8, 16, 16), strict=True))
"""Each site's declared seeds (hooks in `HOOKS` then `FORMED` order, one sequence then packed), sized
independently by `seeds_for` on armada (2026-10-21), on the draw with a memory entering and a
document starting on a chunk's first token."""


def draw(hooks: recurrence.Hooks, packed: bool, seed: int, *, length: int = LENGTH,
         value_heads: int = VALUE_HEADS) -> tuple[Chunk]:
    """A sequence's operands for `hooks` from `seed`, its keys' and values' dimensions each permuted
    by an order the seed draws (an exact symmetry of every head's recurrence): fresh queries, keys
    and values (`NORMED_WIDTH` wide under the norm), a log decay a head or a key dimension in (-1, 0),
    a gate in (0, 1), a skip, a memory entering, `length` tokens over `value_heads` value heads, and with
    `packed` documents starting at tokens 0, 23 and 40 of the first row and 9 and 48 of the second, 48 a
    chunk's first token."""
    keys = jax.random.split(jax.random.key(seed), 8)
    rng = np.random.default_rng(seed)
    width = VALUE_WIDTH if hooks.output is None else NORMED_WIDTH
    key_order, value_order = rng.permutation(KEY_WIDTH), rng.permutation(width)
    query = jax.random.normal(keys[0], (ROWS, length, KEY_HEADS, KEY_WIDTH))
    key = jax.random.normal(keys[1], (ROWS, length, KEY_HEADS, KEY_WIDTH))
    key = key / jnp.linalg.norm(key, axis=-1, keepdims=True)
    value = jax.random.normal(keys[2], (ROWS, length, value_heads, width))
    shape = (ROWS, length, value_heads, KEY_WIDTH) if hooks.decay == 'vector' else (ROWS, length, value_heads)
    log_decay = -jax.nn.softplus(jax.random.normal(keys[3], shape) - 1.0)
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (ROWS, length, value_heads)))
    skip = jax.random.normal(keys[5], (value_heads,))
    gates = {}
    if hooks.gates is not None:
        # Pre-activations: the gate and beta raw, `log_a` and `bias` a head's.
        heads = jax.random.split(keys[6], 3)
        gates = {'gate': jax.random.normal(heads[0], (ROWS, length, value_heads)),
                 'log_a': 0.5 * jax.random.normal(heads[1], (value_heads,)),
                 'bias': jax.random.normal(heads[2], (value_heads,))}
        beta = jax.random.normal(keys[4], (ROWS, length, value_heads))
    if hooks.rated:
        # The step through softplus is the write gate; the rate is `A = -exp(A_log)`, a head's.
        beta = jax.nn.softplus(jax.random.normal(keys[4], (ROWS, length, value_heads)) - 1.0)
        gates = {'rate': -jnp.exp(0.5 * jax.random.normal(keys[6], (value_heads,)))}
    epilogue = {}
    if hooks.output is not None:
        norms = jax.random.split(keys[7], 2)
        epilogue = {'z': jnp.take(jax.random.normal(norms[0], (ROWS, length, value_heads, width)),
                                  value_order, axis=-1),
                    'weight': jnp.take(1.0 + 0.1 * jax.random.normal(norms[1], (width,)), value_order)}
    query, key = (jnp.take(x, key_order, axis=-1) for x in (query, key))
    value = jnp.take(value, value_order, axis=-1)
    if hooks.decay == 'vector':
        log_decay = jnp.take(log_decay, key_order, axis=-1)
    state = 0.3 * jax.random.normal(jax.random.fold_in(jax.random.key(seed), 8),
                                    (ROWS, value_heads, KEY_WIDTH, width))
    state = jnp.take(jnp.take(state, key_order, axis=2), value_order, axis=3)
    starts = None
    if packed:
        starts = (jnp.zeros((ROWS, length), bool).at[0, jnp.array([0, 23, 40])].set(True)
                  .at[1, jnp.array([9, 48])].set(True))
    reads_beta = hooks.gated_write if hooks.gates is None else hooks.gates.write == 'sigmoid'
    plain = hooks.decay != 'none' and hooks.gates is None and not hooks.rated
    return (Chunk(query, key, value, log_decay=log_decay if plain else None,
                  beta=beta if reads_beta else None, skip=skip if hooks.skip else None, state=state,
                  starts=starts, **gates, **epilogue),)


def token_by_token(hooks: recurrence.Hooks, operands: Chunk):
    """The sequence a token at a time through the decode step's reference, a packed document's first
    token starting from a zero memory: `(output, final memory)`, the output in the operands' dtype. Host
    operands are taken as JAX arrays: a float64 truth's first call runs on the gate's widened host arrays,
    which the scan's traced token can't index."""
    operands = jax.tree.map(jnp.asarray, operands)
    group = operands.value.shape[2] // operands.query.shape[2]
    query, key = (jnp.repeat(x, group, axis=2) for x in (operands.query, operands.key))
    work = jnp.promote_types(operands.value.dtype, jnp.float32)
    state = operands.state.astype(work)
    rows, length = operands.value.shape[:2]
    starts = jnp.zeros((rows, length), bool) if operands.starts is None else jnp.asarray(operands.starts)

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

    state, out = jax.lax.scan(one_token, state, jnp.arange(length))
    return jnp.moveaxis(out, 0, 1), state


@pytest.mark.parametrize('packed', PACKED, ids=['one sequence', 'packed'])
@pytest.mark.parametrize('hooks', HOOKS + FORMED, ids=str)
def test_the_chunked_op_is_as_exact_as_the_recurrence_token_by_token(hooks, packed):
    """`chunk`'s portable math in `CHUNK`-token chunks against the same recurrence a token at a time,
    output and final memory, over draws of fresh operands under orders of the keys' and values'
    dimensions, both from float64."""
    dew = jax.jit(functools.partial(chunk_on, 'xla', hooks, chunk_size=CHUNK))
    reference = jax.jit(functools.partial(token_by_token, hooks))
    assert_as_exact(dew, reference, functools.partial(draw, hooks, packed), f'{hooks} packed={packed}',
                    seeds=SEEDS[hooks, packed])


ROUNDED_ONCE = [recurrence.Hooks(decay='none', delta=False, write_gate=True, skip=True),
                recurrence.Hooks(decay='scalar', scale=0.1, skip=True),
                recurrence.Hooks(decay='scalar', scale=0.1, skip=True, output=recurrence.OutputNorm(1e-6))]
"""The rounding sites' hooks: a write gate without a decay, which only the portable math takes, and the delta
rule with a decay, with and without the norm, which the port takes too."""


@pytest.mark.parametrize('hooks, ported', [
    pytest.param(hooks, ported, id=f'{hooks}-{"port" if ported else "portable"}')
    for hooks in ROUNDED_ONCE for ported in ((False, True) if hooks.delta else (False,))])
def test_the_output_rounds_once_after_the_epilogue(hooks, ported):
    """The chunked op rounds its output once, after the skip and the norm, as the decode step's reference
    does, on the portable math and on the port (the delta rule with a decay, its kernels interpreted off
    a GPU): a bf16 token whose readout `0.1` (a unit query and key, the write gate or the scale 0.1)
    the skip `-0.10009765625` nearly cancels leaves the residual, within fp32's rounding over its chain of
    the float64 value (the oracle rule for a fixed input), and the norm reads it, where a readout rounded
    to bf16 first cancels to zero."""
    unit = jnp.zeros((1, 1, 1, 8), jnp.bfloat16).at[..., 0].set(1.0)
    beta = jnp.full((1, 1, 1), 1.0 if hooks.delta else 0.1, jnp.float32)
    operands = Chunk(unit, unit, jnp.ones((1, 1, 1, 8), jnp.bfloat16),
                     log_decay=jnp.zeros((1, 1, 1)) if hooks.decay != 'none' else None, beta=beta,
                     skip=jnp.full((1,), -0.10009765625, jnp.float32),
                     z=jnp.ones((1, 1, 1, 8)) if hooks.output else None,
                     weight=jnp.ones((8,)) if hooks.output else None)
    def op(operands):
        return chunk_on('fla' if ported else 'xla', hooks, operands, dtype=jnp.float32)[0]

    def reference(operands, beta, dtype, hooks=hooks):
        """The decode step's reference over the token under `hooks`, in `dtype`."""
        return recurrence.step_reference(hooks, recurrence.Step(
            jnp.zeros((1, 1, 8, 8), dtype), query=operands.query[:, 0], key=operands.key[:, 0],
            value=operands.value[:, 0],
            log_decay=None if operands.log_decay is None else operands.log_decay[:, 0], beta=beta[:, 0],
            skip=operands.skip, z=None if operands.z is None else operands.z[:, 0],
            weight=operands.weight), dtype=dtype)[0]

    found = op(operands)
    with jax.enable_x64(new_val=True):
        wide = widened((operands, beta))
        truth = np.asarray(reference(*wide, jnp.float64))
        residual = np.asarray(reference(*wide, jnp.float64, hooks=dataclasses.replace(hooks, output=None)))
    # The oracle rule for a fixed input: within fp32's rounding over the op's chain of roundings of its two
    # terms' magnitude (the readout and the skip, each about 0.1), times the norm's gain over the residual
    # where it reads it; a readout rounded to bf16 first misses by the whole residual.
    gain = float(np.max(np.abs(truth)) / np.max(np.abs(residual)))
    terms = 2 * gain * float(np.max(np.abs(np.asarray(operands.skip, np.float64))))
    assert_fp32_reduction_bound(found[:, 0], truth, np.full(truth.shape, terms),
                                chain_roundings(jax.make_jaxpr(op)(operands)))
    assert np.all(np.asarray(found) != 0.0)


def test_the_output_takes_the_dtype_asked_for():
    """An explicit `dtype` holds with and without an epilogue, and bf16 operands under the l2 norm, which
    widens them, come back in bf16."""
    ones = jnp.ones((1, 1, 1, 8), jnp.bfloat16)
    plain = recurrence.Hooks(decay='scalar', scale=0.25)
    wide = Chunk(*(x.astype(jnp.float32) for x in (ones, ones, ones)), log_decay=jnp.zeros((1, 1, 1)),
                 beta=jnp.ones((1, 1, 1)))
    assert chunk_on('xla', plain, wide, dtype=jnp.bfloat16)[0].dtype == jnp.bfloat16
    normed = recurrence.Hooks(decay='scalar', l2norm=True)
    assert chunk_on('xla', normed, Chunk(ones, ones, ones, log_decay=jnp.zeros((1, 1, 1)),
                                         beta=jnp.ones((1, 1, 1))))[0].dtype == jnp.bfloat16


RATED = recurrence.Hooks(decay='scalar', rated=True, delta=False, skip=True)
WHOLE = recurrence.Hooks(decay='scalar', delta=False, write_gate=True, skip=True)
"""A rated decay (`rate` times the write gate) and the same log decay given whole."""

RATED_SEEDS = 8
"""The rated-decay site's declared seeds, sized paired by `seeds_for` on armada."""


def rated_draw(seed: int) -> tuple:
    """A decode token over a memory and a chunked sequence, each with its write gate in [0, 0.5), a rate a
    head in [-2, -0.1) and a skip: `(token, sequence)`, each a dict of the step's or chunk's operands."""
    rng = np.random.default_rng(seed)
    rows, heads, width = 2, 2, 16
    rate = -jnp.asarray(rng.uniform(0.1, 2.0, (heads,)), jnp.float32)
    skip = jnp.asarray(rng.standard_normal((heads,)), jnp.float32)
    token = {name: jnp.asarray(rng.standard_normal((rows, heads, width)), jnp.float32)
             for name in ('query', 'key', 'value')}
    token |= {'state': jnp.asarray(rng.standard_normal((rows, heads, width, width)), jnp.float32),
              'beta': jnp.asarray(rng.uniform(0.0, 0.5, (rows, heads)), jnp.float32), 'rate': rate,
              'skip': skip}
    sequence = {name: jnp.asarray(rng.standard_normal((rows, LENGTH, heads, width)), jnp.float32)
                for name in ('query', 'key', 'value')}
    steps = jnp.asarray(rng.uniform(0.0, 0.5, (rows, LENGTH, heads)), jnp.float32)
    sequence |= {'beta': steps, 'rate': rate, 'skip': skip}
    return token, sequence


def decayed(hooks, token: dict, sequence: dict) -> tuple:
    """`hooks`' decode token on the launch and on its reference, and its chunked sequence on the portable
    math, `rated_draw`'s operands read as `hooks` reads them: a rated decay's rate, or that rate times the
    write gate given whole as the log decay."""
    def operands(given: dict, leading) -> dict:
        given = dict(given)
        rate = given.pop('rate')
        return given | ({'rate': rate} if hooks.rated else {'log_decay': rate[leading] * given['beta']})

    stepped = [step(hooks, recurrence.Step(**operands(token, (None, slice(None)))))[::2]
               for step in (recurrence.step, recurrence.step_reference)]
    chunked = chunk_on('xla', hooks, Chunk(**operands(sequence, (None, None, slice(None)))), chunk_size=CHUNK)
    return (*stepped, chunked)


def test_a_rated_decay_is_its_log_decay():
    """A rated decay (`rate` times the write gate) steps a decode token, on the launch and on its reference,
    and a chunked sequence on the portable math as the same log decay given whole: on par with the whole
    decay's from float64 over the site's seeds (check 1)."""
    assert_as_exact(functools.partial(decayed, RATED), functools.partial(decayed, WHOLE), rated_draw,
                    'a rated decay against its log decay', seeds=RATED_SEEDS)


def test_the_operands_are_checked_as_the_decode_steps():
    """A hook's missing operand is named, and a decay of the other kind's rank refused, rather than read
    as another computation."""
    ones = jnp.ones((1, 4, 1, 8))
    with pytest.raises(ValueError):
        chunk(recurrence.Hooks(decay='scalar'), Chunk(ones, ones, ones, log_decay=jnp.zeros((1, 4, 1))))
    with pytest.raises(ValueError):
        chunk(recurrence.Hooks(decay='vector'), Chunk(ones, ones, ones, log_decay=jnp.zeros((1, 4, 1)),
                                                      beta=jnp.ones((1, 4, 1))))


def test_aligned_unit_keys_erase_the_decay_gradient():
    """In one dimension k=beta=1 writes S'=v, independently of the decay. The WY inverse/readout
    paths cancelled only after rounding, leaving a spurious 4.74e-8 gradient in this exact zero."""
    unit = jnp.ones((1, 4, 1, 1), jnp.float32)
    value = jnp.array([0.314159, -0.271828, 0.577216, 1.414214], jnp.float32).reshape(unit.shape)
    decay = jnp.array([-0.17, -0.23, -0.31, -0.41], jnp.float32).reshape(unit.shape)
    beta = jnp.ones((1, 4, 1), jnp.float32)
    gradient = jax.jit(jax.grad(lambda g: jnp.sum(recurrence_chunk.chunk_delta_rule(
        unit, unit, value, g, beta, chunk_size=4, scale=1.0)[0])))(decay)
    np.testing.assert_array_equal(np.asarray(gradient), np.zeros_like(np.asarray(gradient)))


@pytest.mark.parametrize("length, chunk_size", [(3, 1), (3, 4), (9, 4), (10, 4), (12, 4), (9, 16)])
@pytest.mark.parametrize("packed", PACKED, ids=["one sequence", "packed"])
@pytest.mark.parametrize("delta", [False, True], ids=["write", "delta"])
@pytest.mark.parametrize("channels", [1, 2], ids=["scalar decay", "vector decay"])
def test_the_peeled_write_keeps_ragged_tokens_and_document_resets(
        length, chunk_size, packed, delta, channels):
    """The peeled token is the last real one, including an empty prefix and starts before or on it.
    Dyadic operands make the writes and outputs exact, so a misplaced reset or token has no tolerance."""
    key = np.zeros((2, length, 1, 2), np.float32)
    key[..., 0] = 1.0
    value = np.broadcast_to(np.arange(1, length + 1, dtype=np.float32)[None, :, None, None] / 4,
                             (2, length, 1, 1)).copy()
    g = np.zeros((2, length, 1, channels), np.float32)
    beta = np.full((2, length, 1), 0.5, np.float32)
    initial = np.full((2, 1, 2, 1), 2.0, np.float32)
    begins = np.zeros((2, length), bool)
    if packed:
        for row, positions in enumerate(((0, chunk_size - 1, chunk_size + 1, length - 1),
                                          (chunk_size - 2, 2 * chunk_size - 1))):
            begins[row, [i for i in positions if 0 <= i < length]] = True
    starts = jnp.asarray(begins) if packed else None
    expected_output = np.zeros_like(value)
    expected_final = initial.copy()
    for row in range(2):
        for t in range(length):
            if begins[row, t]:
                expected_final[row] = 0.0
            predicted = expected_final[row, 0, 0, 0] if delta else 0.0
            expected_final[row, 0, 0, 0] += 0.5 * (value[row, t, 0, 0] - predicted)
            expected_output[row, t, 0, 0] = expected_final[row, 0, 0, 0]
    run = jax.jit(functools.partial(recurrence_chunk.chunk_delta_rule, chunk_size=chunk_size,
                                    starts=starts, delta=delta, scale=1.0))
    output, final = run(*(jnp.asarray(x) for x in (key, key, value, g, beta, initial)))
    np.testing.assert_array_equal(np.asarray(output), expected_output)
    np.testing.assert_array_equal(np.asarray(final), expected_final)

    steps = recurrence_chunk.chunked(*(jnp.asarray(x) for x in (key, key, value, g, beta)),
                                      chunk_size, starts=starts, delta=delta, scale=1.0)
    _, _, kept, last = recurrence_chunk._peeled_chunk_inputs(
        steps, jnp.asarray(value), jnp.asarray(g), jnp.asarray(beta), starts, chunk_size)
    positions = np.minimum((np.arange(steps.key.shape[2]) + 1) * chunk_size, length) - 1
    np.testing.assert_array_equal(np.asarray(last[1]), np.moveaxis(value[:, positions], 2, 1))
    np.testing.assert_array_equal(np.asarray(last[-1]), begins[:, positions, None].transpose(0, 2, 1))
    expected_kept = np.array([[not begins[row, i * chunk_size:end].any()
                               for i, end in enumerate(positions)] for row in range(2)])[:, None]
    actual_kept = np.broadcast_to(np.asarray(kept[..., 0, 0]), expected_kept.shape)
    np.testing.assert_array_equal(actual_kept, expected_kept)


@pytest.mark.parametrize("packed", PACKED, ids=["one sequence", "packed"])
@pytest.mark.parametrize("delta", [False, True], ids=["write", "delta"])
def test_the_peeled_transition_carries_the_peeled_memory_and_its_adjoint(packed, delta):
    """The custom VJP's boundary transition is the peeled carry's own linear part; corrected values
    keep the whole chunk's WY readout, even when its ragged last chunk has an empty prefix. Exact data:
    dyadic operands every product and sum of which fp32 holds, each side equal to the same in float64."""
    key = jnp.broadcast_to(jnp.array([1.0, 0.0]), (1, 5, 1, 2))
    value = jnp.arange(1, 6, dtype=jnp.float32).reshape(1, 5, 1, 1) / 4
    g, beta = jnp.zeros((1, 5, 1, 2)), jnp.full((1, 5, 1), 0.5)
    starts = jnp.array([[False, False, True, False, True]]) if packed else None
    steps = recurrence_chunk.chunked(key, key, value, g, beta, 4, starts=starts, delta=delta, scale=1.0)
    grown, last, kept, tail = recurrence_chunk._peeled_chunk_inputs(steps, value, g, beta, starts, 4)
    step = steps.key, steps.w, steps.u, grown, last, kept, *tail
    entered = jnp.full((1, 1, 2, 2, 1), 2.0)
    carried, (_, corrected) = algebra.peeled_carry(entered, step, delta=delta)
    linear, written = algebra.peeled_transition(
        steps.key, steps.w, steps.u, grown, last, kept, tail, delta=delta)
    gradient = jax.jit(jax.grad(
        lambda state: algebra.peeled_carry(state, step, delta=delta)[0].sum()))(entered)
    with jax.enable_x64(new_val=True):
        wide = [np.asarray(x, np.float64) for x in (linear, entered, written, steps.u, steps.w)]
    exact = {'carried': wide[0] @ wide[1] + wide[2], 'corrected': wide[3] - wide[4] @ wide[1],
             'gradient': np.swapaxes(wide[0], -1, -2) @ np.ones(np.shape(carried))}
    for name, found in (('carried', carried), ('corrected', corrected), ('gradient', gradient)):
        np.testing.assert_array_equal(np.asarray(found, np.float64), exact[name], err_msg=name)


@pytest.mark.parametrize('shards', [3, 4])
@pytest.mark.parametrize('dense', [True, False], ids=['dense', 'decay'])
def test_each_shard_enters_with_the_spans_before_it_applied_in_turn(dense, shards):
    """The memory each shard of a sequence enters its span with across a mesh axis (`recurrence_algebra
    .entering`, a Kogge-Stone prefix of the shards' spans) is the spans before it applied in turn to the
    initial memory (`Span.applied`), on par from float64 over the fewest draws: a dense transfer (the delta
    rule's) and a head's decay (Mamba-2's), over a power of two of shards and a count that is not."""
    batch, heads, width, columns = 2, 3, 4, 5

    def drawn(seed):
        keys = jax.random.split(jax.random.key(seed), 3)
        transfer = (jax.random.normal(keys[0], (shards, batch, heads, width, width)) / width if dense
                    else -jax.random.uniform(keys[0], (shards, batch, heads)))
        return (transfer, jax.random.normal(keys[1], (shards, batch, heads, width, columns)),
                jax.random.normal(keys[2], (batch, heads, width, columns)))

    mesh = jax.make_mesh((shards,), ('shards',), devices=jax.devices()[:shards])

    def entered(transfer, write, initial):
        def shard(transfer, write, initial):
            return algebra.entering(algebra.Span(transfer[0], write[0]), 'shards', initial)[None]

        return jax.shard_map(shard, mesh=mesh, in_specs=(P('shards'), P('shards'), P()),
                             out_specs=P('shards'))(transfer, write, initial)

    def in_turn(transfer, write, initial):
        memories = [initial]
        for index in range(shards - 1):
            memories.append(algebra.Span(transfer[index], write[index]).applied(memories[-1]))
        return jnp.stack(memories)

    assert_as_exact(jax.jit(entered), jax.jit(in_turn), drawn, f'{shards} shards entering, dense {dense}',
                    seeds=FEWEST_SEEDS)


@pytest.mark.parametrize('packed', PACKED, ids=['one sequence', 'packed'])
@pytest.mark.parametrize('hooks', [HOOKS[4], HOOKS[8], HOOKS[6]], ids=str)
def test_composed_chunks_leave_the_memory_the_recurrence_leaves(hooks, packed):
    """The chunks' transitions (`recurrence_algebra.transition` over production's `chunked` operands)
    composed as an associative scan runs them (`compose`) leave the final memory the recurrence leaves a
    token at a time (`token_by_token`, the decode step's reference), on par from float64 over the site's
    draws: the scalar and the vector decay with the delta rule and the scalar without, over one sequence
    and over packed documents (one starting on a chunk's first token, the memory entering dropped)."""
    group = VALUE_HEADS // KEY_HEADS

    def composed(operands):
        query, key = (jnp.repeat(x, group, axis=2) for x in (operands.query, operands.key))
        g = operands.log_decay if hooks.decay == 'vector' else operands.log_decay[..., None]
        steps = recurrence_chunk.chunked(query, key, operands.value, g, operands.beta, CHUNK,
                                         starts=operands.starts, delta=hooks.delta)
        t, w = jax.lax.associative_scan(algebra.compose, algebra.transition(
            steps.key, steps.w, steps.u, steps.grown, steps.last, steps.kept), axis=2)
        entered = float32_einsum('bhkj,bhjv->bhkv', t[:, :, -1], operands.state.astype(t.dtype))
        return entered + w[:, :, -1]

    def recurrent(operands):
        return token_by_token(hooks, operands)[1]

    assert_as_exact(jax.jit(composed), jax.jit(recurrent), functools.partial(draw, hooks, packed),
                    f'{hooks} composed', seeds=COMPOSED_SEEDS[hooks, packed])


PORTABLE_SEEDS = 8
"""The portable rule's differentiation sites' declared draws (check 1), sized by `seeds_for` (paired, on
armada): the remat rung's gradients, the public seam's tangents, two forwards and a TPU's scan under
forward mode."""


def test_a_call_runs_its_products_kernels_apart():
    """`_by_product`: a call's value is its forward kernel's, and its gradient its training kernel's own,
    the two kernels' forward and pullback where a gradient is taken."""
    def forward(x):
        return 2.0 * x

    def training(x):
        return 3.0 * x

    x = jnp.arange(3.0)
    np.testing.assert_array_equal(recurrence_chunk._by_product(forward, training, x), 2.0 * x)
    value, grad = jax.value_and_grad(lambda x: recurrence_chunk._by_product(forward, training, x).sum())(x)
    assert value == 3.0 * x.sum()
    np.testing.assert_array_equal(grad, jnp.full(3, 3.0))


def portable_draw(seed: int, rows: int, packed: bool) -> tuple:
    """A small delta-rule call's operands `(query, key, value, g, beta)` (a decay a head), its document
    starts where `packed`, and a direction the shape of each operand: the portable rule's
    differentiation tests' draw."""
    keys = jax.random.split(jax.random.key(seed), 7)
    length, heads, width = 37, 2, 8
    query, key, value = (jax.random.normal(k, (rows, length, heads, width)) for k in keys[:3])
    g = -jax.nn.softplus(jax.random.normal(keys[3], (rows, length, heads, 1)))
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (rows, length, heads)))
    starts = jax.random.bernoulli(keys[5], 0.1, (rows, length)) if packed else None
    operands = (query, key, value, g, beta)
    return operands, starts, tuple(jax.random.normal(keys[6], x.shape) for x in operands)


@pytest.mark.parametrize("packed", PACKED, ids=["one sequence", "packed"])
def test_the_portable_rule_recomputes_its_memory_where_the_rung_names_it(packed):
    """Under a remat rung recomputing the delta rule's memory (`recurrence_chunk.recomputing`,
    'delta_memory'), the portable rule (`chunk_delta_rule`, a CPU's route) keeps its operands alone and
    computes its memory again in its backward, as the port and the template drop theirs: its residuals
    hold fewer bytes than the kept rule's, and its gradients, another program's, are on par with the kept
    rule's over the draws (check 1), truth the kept rule's in float64."""
    def loss(*operands, starts=None):
        out, final = recurrence_chunk.chunk_delta_rule(*operands, chunk_size=CHUNK, starts=starts)
        return jnp.sum(out ** 2) + jnp.sum(final)

    def recomputed(*operands, starts=None):
        with recurrence_chunk.recomputing(("delta_memory",)):
            return loss(*operands, starts=starts)

    operands, starts, _ = portable_draw(7, 2, packed)

    def residual_bytes(fn) -> int:
        return sum(leaf.nbytes for leaf in jax.tree.leaves(jax.vjp(functools.partial(fn, starts=starts),
                                                                   *operands)[1]))

    assert residual_bytes(recomputed) < residual_bytes(loss)

    def draw(seed):
        operands, starts, _ = portable_draw(seed, 2, packed)
        return (*operands, starts)

    def gradient(fn):
        return lambda *drawn: jax.grad(functools.partial(fn, starts=drawn[5]), argnums=range(5))(*drawn[:5])

    assert_as_exact(jax.jit(gradient(recomputed)), jax.jit(gradient(loss)), draw, "recomputed memory",
                    seeds=PORTABLE_SEEDS, truth_fn=gradient(loss))


@pytest.mark.parametrize("packed", PACKED, ids=["one sequence", "packed"])
def test_the_portable_rule_takes_forward_mode_at_its_public_seam(packed):
    """`chunk_delta_rule` (a CPU's route, the portable math) under a plain `jax.jvp` and `jax.linearize`,
    with no forward-mode context: its value and tangents on par with the chunked math's own forward mode
    over the draws (check 1), truth the math's in float64, and its gradient still its reverse rule's
    (`_chunk_delta_backward`), finite, the custom VJP's that refused a plain `jax.jvp` before (row 1834)."""
    operands, starts, directions = portable_draw(11, 1, packed)
    rows, _, heads, width = operands[0].shape

    def rule(starts):
        return lambda *operands: recurrence_chunk.chunk_delta_rule(*operands, chunk_size=CHUNK, starts=starts)

    def math(starts):
        def run(*operands):
            state, delta = jnp.zeros((rows, heads, width, width), operands[0].dtype), True
            return recurrence_chunk._chunk_delta_rule(*operands, state, CHUNK, starts, delta, width ** -0.5,
                                                      jnp.dtype(operands[0].dtype))
        return run

    def draw(seed):
        operands, starts, directions = portable_draw(seed, 1, packed)
        return (*operands, *directions, starts)

    def forward(fn):
        def run(*drawn):
            primal, tangent, starts = drawn[:5], drawn[5:10], drawn[10]
            value, along = jax.jvp(fn(starts), primal, tangent)
            return value, along, jax.linearize(fn(starts), *primal)[1](*tangent)
        return run

    assert_as_exact(jax.jit(forward(rule)), jax.jit(forward(math)), draw, "public seam's forward mode",
                    seeds=PORTABLE_SEEDS, truth_fn=forward(math))
    found = jax.jit(lambda *a: jax.jvp(rule(starts), a, directions))(*operands)
    cotangent = jax.tree.map(jnp.ones_like, found[0])
    def scored(*operands):
        return sum(jnp.vdot(x, c) for x, c in zip(rule(starts)(*operands), cotangent, strict=True))

    gradient = jax.grad(scored, argnums=range(5))(*operands)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in gradient)


@pytest.mark.parametrize("packed", PACKED, ids=["one sequence", "packed"])
def test_the_portable_rule_takes_nested_differentiation(packed):
    """`chunk_delta_rule` under forward over reverse (`jax.jvp` of `jax.grad`), reverse over forward
    (`jax.grad` of `jax.jvp`) and two forwards (`jax.jvp` of `jax.jvp`), which raised before (rows 1861: its
    saved bool resets took a bool tangent, not float0's; 1862: its pullback gave six cotangents of seven
    operands): each runs and is finite (the public seam's forward mode is the chunked math's by check 1,
    `test_the_portable_rule_takes_forward_mode_at_its_public_seam`)."""
    (query, key, value, g, beta), starts, (direction, *_) = portable_draw(13, 1, packed)

    def rule(query):
        return recurrence_chunk.chunk_delta_rule(query, key, value, g, beta, chunk_size=CHUNK, starts=starts)

    def loss(fn, query):
        out, final = fn(query)
        return jnp.sum(out ** 2) + jnp.sum(final ** 2)

    def tangent(fn, query):
        return jax.jvp(functools.partial(loss, fn), (query,), (direction,))[1]

    def nested(fn):
        forward_over_reverse = jax.jvp(jax.grad(functools.partial(loss, fn)), (query,), (direction,))[1]
        reverse_over_forward = jax.grad(functools.partial(tangent, fn))(query)
        twice = jax.jvp(functools.partial(tangent, fn), (query,), (direction,))[1]
        return forward_over_reverse, reverse_over_forward, twice

    found = jax.jit(lambda: nested(rule))()
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in found)


def test_a_rated_scan_traced_for_a_jvp_on_a_tpu_gives_its_tangent():
    """A rated scan the TPU's Pallas scan tiles takes in reverse mode alone, its custom VJP giving no tangent
    (row 1869, audit K2): Mamba-2's SSD scan (`mamba2.SSD` without its skip, whose own custom VJP refuses a
    jvp, row 1933) traced for `jax.jvp` (`forward_mode_attention`) on an assumed v6e gives its tangent, on par
    with the scan's on this CPU over the draws (check 1), truth its float64 run."""
    import dataclasses

    from dew.nn.attention import forward_mode_attention
    from dew.nn.mixers.mamba2 import SSD

    hooks = dataclasses.replace(SSD, skip=False)

    def draw(seed):
        keys = jax.random.split(jax.random.key(seed), 6)
        x, direction = (jax.random.normal(key, (1, 64, 4, 16)) for key in keys[:2])
        B, C = (jax.random.normal(key, (1, 64, 2, 16)) for key in keys[2:4])
        dt = jax.nn.softplus(jax.random.normal(keys[4], (1, 64, 4)))
        A = -jnp.exp(jax.random.normal(keys[5], (4,)))
        return x, direction, B, C, dt, A

    def tangent(target=None):
        def run(x, direction, B, C, dt, A):
            def scan(x):
                return chunk(hooks, Chunk(C, B, x, beta=dt, rate=A), chunk_size=CHUNK, dtype=x.dtype)[0]
            with generation.assumed(target), forward_mode_attention():
                return jax.jvp(scan, (x,), (direction,))[1]
        return run

    assert_as_exact(jax.jit(tangent("v6e")), jax.jit(tangent()), draw, "a TPU's scan under jvp",
                    seeds=PORTABLE_SEEDS, truth_fn=tangent())
