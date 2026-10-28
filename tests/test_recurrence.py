"""The linear recurrence's decode step (`dew.nn.kernels.recurrence.step`): every combination of its
hooks against its XLA form in float64, at its launch's config (`recurrence.COLUMNS`' column blocks, or a
whole head where a hook reads one), and the continuation contract a
decode token keeps (an idle row untouched, a bf16 state rounded as it is held)."""

import dataclasses
import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact
from tuned import each_config, taken

from dew.nn.kernels import recurrence, tuning
from dew.nn.precision import FP32_SUMS

ROWS, KEY_HEADS, VALUE_HEADS, KEY_WIDTH, VALUE_WIDTH, TAPS = 16, 2, 4, 16, 64, 4
"""The draws' shape: values two of the step launch's column blocks wide (`recurrence.COLUMNS`), so check 1
reaches a head's later blocks' programs, as a gated delta net's 128-wide values take four (template numerics
review 2026-10-26, P3-6)."""
"""A decode token with grouped heads, idle rows and a four-tap history."""

NORMS = (None, recurrence.OutputNorm(1e-6, 'silu'), recurrence.OutputNorm(1e-6, 'sigmoid'))
HOOKS = [recurrence.Hooks(conv=conv, l2norm=norm, decay='scalar' if decay == 'gates' else decay,
                          gates=recurrence.Gates() if decay == 'gates' else None, delta=delta,
                          write_gate=write, skip=skip, output=output)
         for conv, norm, decay, delta, write, skip, output in itertools.product(
             (None, recurrence.Conv('silu')), (False, True), ('none', 'scalar', 'vector', 'gates'),
             (False, True), (False, True), (False, True), NORMS)
         # The delta rule and the gates read a write gate whatever `write_gate` says.
         if (decay != 'gates' or delta) and not (write and (delta or decay == 'gates'))]

SEEDS = dict(zip(HOOKS, (
    8, 16, 8, 8, 32, 8, 8, 16, 8, 8, 8, 8, 8, 16, 8, 8, 16, 16, 8, 16,
    8, 8, 8, 8, 8, 32, 8, 8, 8, 16, 8, 16, 16, 8, 32, 16, 16, 16, 8, 8,
    16, 8, 8, 16, 8, 8, 16, 16, 8, 32, 8, 8, 16, 16, 16, 32, 16, 16, 16, 16,
    8, 16, 16, 64, 32, 8, 8, 8, 8, 64, 8, 8, 8, 16, 8, 64, 32, 16, 8, 16,
    16, 32, 8, 16, 8, 16, 8, 64, 16, 16, 8, 32, 16, 64, 32, 32, 8, 32, 16, 64,
    32, 32, 8, 16, 8, 64, 16, 8, 8, 16, 8, 64, 64, 16, 8, 16, 16, 64, 64, 16,
    64, 16, 8, 32, 64, 8, 128, 16, 16, 32, 32, 16, 512, 32, 32, 256, 32, 32, 128, 16,
    16, 64, 32, 16, 32, 8, 8, 32, 32, 16, 128, 32, 32, 128, 32, 32, 64, 16, 8, 64,
    32, 16, 16, 32, 8, 16, 16, 16, 128, 32, 16, 128, 16, 16, 128, 64, 32, 128, 32, 64,
    8, 16, 8, 64, 32, 16, 8, 8, 8, 64, 32, 16, 8, 16, 16, 64, 32, 16, 8, 16,
    16, 64, 64, 32, 8, 8, 8, 128, 32, 16, 8, 8, 8, 128, 32, 16, 8, 16, 8, 128,
    32, 8, 8, 16, 8, 128, 16, 8, 8, 16, 8, 64, 16, 8, 8, 16, 8, 64, 32, 8,
), strict=True))
"""Each hook combination's declared seeds, sized independently by `seeds_for` on armada (2026-10-20).
The reference compared with itself resolves the factor in every leaf; the count holds at every config.
"""

FORMED = [
    recurrence.Hooks(decay='scalar', gates=recurrence.Gates(write='step', limit=(0.0, 0.5)), delta=False,
                     skip=True),
    recurrence.Hooks(decay='scalar', gates=recurrence.Gates(limit=(0.0, 0.5))),
    recurrence.Hooks(decay='scalar', scale=0.25),
    recurrence.Hooks(decay='vector', l2norm=True, scale=0.25),
    recurrence.Hooks(decay='scalar', rated=True, delta=False, skip=True, output=NORMS[1]),
]
"""The launch's other prologue values: Mamba2's step gates clipped with the skip, the gated delta rule's
gates with a clipped step, a scale without the l2 norm and one under it, and a rated decay with the
skip and the norm (`recurrence_chunk.chunk`'s SSD)."""

FORMED_SEEDS = dict(zip(FORMED, (8, 16, 8, 8, 16), strict=True))
"""The formed sites' seeds, sized independently by `seeds_for` on armada (2026-10-21)."""

BF16_SEEDS = {HOOKS[0]: 8, HOOKS[-1]: 16}
"""The bf16-state sites' seeds, sized by `seeds_for` on armada (2026-10-20), including every output."""

ONE_TAP_SEEDS = {jnp.float32: 256, jnp.bfloat16: 256}
"""Each one-tap operand dtype's seeds, sized by `seeds_for` on armada (2026-10-20) on the whole tree."""


def operands(hooks: recurrence.Hooks, seed: int = 0, state_dtype=jnp.float32) -> recurrence.Step:
    """A decode token's operands for `hooks`, every third row from the second idle."""
    rng = np.random.default_rng(seed)

    def normal(*shape, scale=1.0):
        return jnp.asarray(rng.standard_normal(shape) * scale, jnp.float32)

    channels = 2 * KEY_HEADS * KEY_WIDTH + VALUE_HEADS * VALUE_WIDTH
    given = {'state': normal(ROWS, VALUE_HEADS, KEY_WIDTH, VALUE_WIDTH, scale=0.3).astype(state_dtype),
             'active': jnp.asarray(np.arange(ROWS) % 3 != 1)}
    if hooks.conv is not None:
        given |= {'token': normal(ROWS, channels), 'held': normal(ROWS, TAPS - 1, channels),
                  'taps': normal(channels, TAPS, scale=0.5)}
    else:
        given |= {'query': normal(ROWS, KEY_HEADS, KEY_WIDTH), 'key': normal(ROWS, KEY_HEADS, KEY_WIDTH) / 4,
                  'value': normal(ROWS, VALUE_HEADS, VALUE_WIDTH)}
    if hooks.gates is not None:
        given['gate'] = normal(ROWS, VALUE_HEADS)
        if hooks.gates.write == 'sigmoid':
            given['beta'] = normal(ROWS, VALUE_HEADS)
        given |= {'log_a': normal(VALUE_HEADS, scale=0.3), 'bias': normal(VALUE_HEADS, scale=0.3)}
    elif hooks.rated:
        # Mamba2's: the step through softplus is the write gate, the rate `A = -exp(A_log)` a head's.
        given |= {'beta': jax.nn.softplus(normal(ROWS, VALUE_HEADS) - 1.0),
                  'rate': -jnp.exp(normal(VALUE_HEADS, scale=0.5))}
    else:
        if hooks.decay == 'scalar':
            given['log_decay'] = -jnp.abs(normal(ROWS, VALUE_HEADS, scale=0.2))
        elif hooks.decay == 'vector':
            given['log_decay'] = -jnp.abs(normal(ROWS, VALUE_HEADS, KEY_WIDTH, scale=0.2))
        if hooks.gated_write:
            given['beta'] = jax.nn.sigmoid(normal(ROWS, VALUE_HEADS))
    if hooks.skip:
        given['skip'] = normal(VALUE_HEADS)
    if hooks.output is not None:
        given |= {'z': normal(ROWS, VALUE_HEADS, VALUE_WIDTH), 'weight': normal(VALUE_WIDTH)}
    return recurrence.Step(**given)


def orbit(given: recurrence.Step, seed: int) -> recurrence.Step:
    """Permute every head's key and value dimensions alike, an exact symmetry of a decode token.

    The state moves on both axes, the per-key decay with the keys, the output gate and norm weight
    with the values, and the conv's channels with the projections they hold. Every arm runs on this
    layout, so its outputs stay in it.
    """
    rng = np.random.default_rng(seed)
    key_width, value_width = given.state.shape[-2:]
    keys, values = rng.permutation(key_width), rng.permutation(value_width)
    moved = {'state': given.state[..., keys, :][..., values]}
    for name in ('query', 'key'):
        if (array := getattr(given, name)) is not None:
            moved[name] = array[..., keys]
    for name in ('value', 'z', 'weight'):
        if (array := getattr(given, name)) is not None:
            moved[name] = array[..., values]
    if given.log_decay is not None and given.log_decay.ndim == 3:
        moved['log_decay'] = given.log_decay[..., keys]
    if given.token is not None:
        value_heads = given.state.shape[1]
        key_heads = (given.token.shape[-1] - value_heads * value_width) // (2 * key_width)
        key_channels = np.arange(key_heads * key_width).reshape(key_heads, key_width)[:, keys].ravel()
        value_channels = np.arange(value_heads * value_width).reshape(value_heads, value_width)
        channels = np.concatenate((key_channels, key_heads * key_width + key_channels,
                                   2 * key_heads * key_width + value_channels[:, values].ravel()))
        moved |= {'token': given.token[..., channels], 'held': given.held[..., channels],
                  'taps': given.taps[channels]}
    return dataclasses.replace(given, **moved)


def draw(hooks: recurrence.Hooks, seed: int, state_dtype=jnp.float32) -> tuple[recurrence.Step]:
    """Fresh operands and an independent exact orbit, deterministically keyed by the draw's seed."""
    operand_seed, orbit_seed = (int(value) for value in np.random.SeedSequence(seed).generate_state(2))
    return (orbit(operands(hooks, operand_seed, state_dtype), orbit_seed),)


def one_tap_operands(dtype, seed: int) -> recurrence.Step:
    """A one-tap token, preserving the two dtypes' consecutive draws at the original seed."""
    channels = 2 * 16 + 16
    rng = np.random.default_rng(seed)
    operands = {}
    for operand_dtype in (jnp.float32, jnp.bfloat16):
        operands[operand_dtype] = recurrence.Step(
            jnp.asarray(rng.standard_normal((ROWS, 1, 16, 16)) * 0.3, jnp.float32),
            token=jnp.asarray(rng.standard_normal((ROWS, channels)), operand_dtype),
            held=jnp.zeros((ROWS, 0, channels), operand_dtype),
            taps=jnp.asarray(rng.standard_normal((channels, 1)), operand_dtype))
    return operands[dtype]


def one_tap_draw(dtype, seed: int) -> tuple[recurrence.Step]:
    """A fresh one-tap token and an independent key/value orbit, deterministic for each seed."""
    operand_seed, orbit_seed = (int(value) for value in np.random.SeedSequence(seed).generate_state(2))
    return (orbit(one_tap_operands(dtype, operand_seed), orbit_seed),)


def summed(hooks: recurrence.Hooks):
    """The XLA form's step with its fp32 products summed in fp32 (`FP32_SUMS`), as the GPU the
    interpreted kernel stands for sums them, where a CPU test process sums them in float64
    (`precision.accumulated`): the reference arm of a launch's check, the plain form's float64 run its
    truth."""

    @jax.jit
    def run(operands):
        with jax.default_matmul_precision(FP32_SUMS):
            return recurrence.step_reference(hooks, operands)
    return run


@pytest.mark.parametrize('hooks', HOOKS + FORMED, ids=str)
def test_every_hook_combination_steps_as_its_float64_form(hooks):
    """A decode token through the launch, at its config, within tests/reference_error.py's
    rule of the XLA form's fp32 run against its float64 run: its output, its state and, under a conv,
    its moved history, which is the history's own bits (`summed`'s fp32 run). The idle row's output is
    zeros and its state and history are its inputs' bits."""
    given = operands(hooks)
    reference_fn = jax.jit(functools.partial(recurrence.step_reference, hooks))
    reference = reference_fn(given)

    configs = each_config(recurrence.StepLaunch, functools.partial(recurrence.step, hooks), given)
    # A column-split launch runs more than one block a head, its later programs' columns among them.
    assert all(config.block == 0 or VALUE_WIDTH // config.block > 1 for config in configs), configs
    for config in configs:
        with tuning.pinned({recurrence.StepLaunch: config}):
            step = jax.jit(functools.partial(recurrence.step, hooks))
            assert_as_exact(step, summed(hooks), functools.partial(draw, hooks), f'{hooks} {config}',
                            seeds={**SEEDS, **FORMED_SEEDS}[hooks], truth_fn=reference_fn)
            out, history, state = step(given)
        np.testing.assert_array_equal(np.asarray(out)[1], 0.0)
        np.testing.assert_array_equal(np.asarray(state)[1], np.asarray(given.state)[1])
        if hooks.conv is not None:
            np.testing.assert_array_equal(np.asarray(history), np.asarray(reference[1]))
            np.testing.assert_array_equal(np.asarray(history)[1], np.asarray(given.held)[1])


def test_a_column_block_pin_is_refused_where_a_hook_reads_a_whole_head():
    """A hook reading a head's whole state (a conv, the epilogue's norm) takes a whole-head launch: a pin
    of a block of value columns is refused as the step traces, a whole-head pin taken."""
    hooks = next(form for form in HOOKS + FORMED if form.conv is not None or form.output is not None)
    given = operands(hooks)
    step = functools.partial(recurrence.step, hooks)
    with tuning.pinned({recurrence.StepLaunch: recurrence.Launch(8, 4)}), pytest.raises(tuning.Unpinnable):
        jax.jit(step).trace(given)
    with tuning.pinned({recurrence.StepLaunch: recurrence.Launch(0, 4)}):
        assert taken(recurrence.StepLaunch, step, given) == [recurrence.Launch(0, 4)]


@pytest.mark.parametrize('hooks', [HOOKS[0], HOOKS[-1]], ids=str)
def test_a_bf16_state_steps_as_its_float64_form(hooks):
    """A bf16 state is read and stepped in fp32 and written once, rounded stochastically by
    `round_to_bf16` from the token's key and value: within the rule of the XLA form's own bf16 run
    against the float64 step, and the idle row's state its input's bits."""
    given = operands(hooks, seed=1, state_dtype=jnp.bfloat16)
    _, _, state = recurrence.step(hooks, given)
    assert state.dtype == jnp.bfloat16
    assert_as_exact(jax.jit(functools.partial(recurrence.step, hooks)), summed(hooks),
                    functools.partial(draw, hooks, state_dtype=jnp.bfloat16), f'bf16 state {hooks}',
                    seeds=BF16_SEEDS[hooks],
                    truth_fn=jax.jit(functools.partial(recurrence.step_reference, hooks)))
    np.testing.assert_array_equal(np.asarray(state)[1], np.asarray(given.state)[1])


def test_a_write_gate_without_the_delta_rule_scales_what_the_token_writes():
    """Without the delta rule a write gate is an input gate: a zero gate writes nothing, so a zero
    state stays zero and the output is zero, in the launch and in its XLA form; and a one-tap conv's
    moved history is empty in both, its step the same as an unconvolved one's on the scaled token."""
    hooks = recurrence.Hooks(decay='none', delta=False, write_gate=True)
    ones = jnp.ones((2, 1, 16))
    given = recurrence.Step(jnp.zeros((2, 1, 16, 16)), query=ones, key=ones, value=ones,
                            beta=jnp.zeros((2, 1)))
    for step in (recurrence.step, recurrence.step_reference):
        out, _, state = step(hooks, given)
        np.testing.assert_array_equal(np.asarray(out), 0.0)
        np.testing.assert_array_equal(np.asarray(state), 0.0)
    one_tap = recurrence.Hooks(conv=recurrence.Conv(None), decay='none', delta=False)
    channels = 2 * 16 + 16
    for dtype in (jnp.float32, jnp.bfloat16):
        given = one_tap_operands(dtype, 3)
        for step in (recurrence.step, recurrence.step_reference):
            assert step(one_tap, given)[1].shape == (ROWS, 0, channels)
        assert_as_exact(jax.jit(functools.partial(recurrence.step, one_tap)), summed(one_tap),
                        functools.partial(one_tap_draw, dtype), f'one-tap {jnp.dtype(dtype).name}',
                        seeds=ONE_TAP_SEEDS[dtype],
                        truth_fn=jax.jit(functools.partial(recurrence.step_reference, one_tap)))


SKIP_SEEDS = 8
"""The skip's forward-mode site's declared seeds, sized paired by `seeds_for` on armada."""

SKIPPED = recurrence.Hooks(decay='scalar', rated=True, delta=False, skip=True)


def skipped(skip, value):
    """The skip `D v` alone (`recurrence.skipped` over a zero output)."""
    return recurrence.skipped(SKIPPED, jnp.zeros_like(value), value, {'skip': skip}, jnp.float32)


def skip_product(skip, value):
    """`D v` as a plain product."""
    return skip[:, None] * value


def skip_draw(seed: int) -> tuple:
    """A skip `[3]` and values `[2, 7, 3, 5]`, and a direction in each, from the seed's key."""
    keys = jax.random.split(jax.random.key(seed), 4)
    return (jax.random.normal(keys[0], (3,)), jax.random.normal(keys[1], (2, 7, 3, 5)),
            jax.random.normal(keys[2], (3,)), jax.random.normal(keys[3], (2, 7, 3, 5)))


def forward_mode(fn):
    """`fn`'s output and tangent under `jax.jvp`, and its tangent under `jax.linearize`, along a direction."""
    def run(skip, value, d_skip, d_value):
        output, tangent = jax.jvp(fn, (skip, value), (d_skip, d_value))
        return output, tangent, jax.linearize(fn, skip, value)[1](d_skip, d_value)

    return run


def test_the_skip_takes_forward_and_reverse_mode():
    """The skip `D v` (`recurrence.skipped`) under `jax.jvp` and `jax.linearize` takes the product's own
    tangent, on par with the plain product's from float64 over the site's seeds (check 1), and under
    `jax.grad` its reverse rule's gradient in `D`, the product's summed over every token and value (a
    decode step's reference reaches it: a custom VJP alone refused `jax.jvp`)."""
    assert_as_exact(forward_mode(skipped), forward_mode(skip_product), skip_draw, "the skip's forward mode",
                    seeds=SKIP_SEEDS)
    skip, value, *_ = skip_draw(0)
    # Its reverse rule's gradient held to float64 by test_recurrence_stability's rated sites.
    gradient = jax.jit(jax.grad(lambda d: jnp.sum(skipped(d, value) * value)))(skip)
    assert gradient.shape == skip.shape and bool(jnp.all(jnp.isfinite(gradient)))


@pytest.mark.parametrize('delta', [False, True], ids=['no delta rule', 'the delta rule'])
def test_a_step_reads_its_updated_state(delta):
    """A token's output is `q^T S'` of the updated state (`recurrence.update`), with the delta rule and
    without: finite where the one-pass form's `(q . k) (write v)` overflowed, an fp32 key and query of 2^80
    on one state channel against a write of 2^-80 and a value of 1 from a zero state with no decay, whose
    updated state is 1 there and its readout exactly 2^80 (rows 1859 and 1962)."""
    channels, width = 32, 4
    query = key = jnp.zeros((1, channels), jnp.float32).at[0, 0].set(2.0 ** 80)
    value = jnp.ones((1, width), jnp.float32)
    state = jnp.zeros((1, channels, width), jnp.float32)
    decay, write = jnp.ones((1, channels), jnp.float32), jnp.full((1, 1), 2.0 ** -80, jnp.float32)
    out, updated = recurrence.update(state, query, key, value, decay, write, delta=delta)
    np.testing.assert_array_equal(updated, state.at[0, 0].set(1.0))
    np.testing.assert_array_equal(out, jnp.full((1, width), 2.0 ** 80, jnp.float32))
