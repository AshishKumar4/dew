"""The linear recurrence's decode step (`dew.nn.kernels.recurrence.step`): every combination of its
hooks against its XLA form in float64, at every config of its launch, and the continuation contract a
decode token keeps (an idle row untouched, a bf16 state rounded as it is held)."""

import dataclasses
import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference, widened
from tuned import each_config

from dew.nn.kernels import recurrence, tuning

ROWS, KEY_HEADS, VALUE_HEADS, KEY_WIDTH, VALUE_WIDTH, TAPS = 16, 2, 4, 16, 32, 4
"""Each row a draw of the operands, and a step's error measured over all of them: three rows' swung
as far as 2.65x of the reference's over 12 seeds, while the twelve together sat within 1.34x."""

NORMS = (None, recurrence.OutputNorm(1e-6, 'silu'), recurrence.OutputNorm(1e-6, 'sigmoid'))
HOOKS = [recurrence.Hooks(conv=conv, l2norm=norm, decay='scalar' if decay == 'gates' else decay,
                          gates=recurrence.Gates() if decay == 'gates' else None, delta=delta,
                          write_gate=write, skip=skip, output=output)
         for conv, norm, decay, delta, write, skip, output in itertools.product(
             (None, recurrence.Conv('silu')), (False, True), ('none', 'scalar', 'vector', 'gates'),
             (False, True), (False, True), (False, True), NORMS)
         # The delta rule and the gates read a write gate whatever `write_gate` says.
         if (decay != 'gates' or delta) and not (write and (delta or decay == 'gates'))]


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
        given |= {'gate': normal(ROWS, VALUE_HEADS), 'beta': normal(ROWS, VALUE_HEADS),
                  'log_a': normal(VALUE_HEADS, scale=0.3), 'bias': normal(VALUE_HEADS, scale=0.3)}
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


@pytest.mark.parametrize('hooks', HOOKS, ids=str)
def test_every_hook_combination_steps_as_its_float64_form(hooks):
    """A decode token through the launch, at each config of its space, within tests/reference_error.py's
    rule of the XLA form's fp32 run against its float64 run: its output, its state and, under a conv,
    its moved history, which is the history's own bits. The idle row's output is zeros and its state
    and history are its inputs' bits."""
    given = operands(hooks)
    reference = recurrence.step_reference(hooks, given)
    with jax.enable_x64(new_val=True):
        wide = recurrence.Step(**{name: None if value is None else widened(value)
                                  for name, value in vars(given).items()})
        truth = [None if x is None else np.asarray(x) for x in recurrence.step_reference(hooks, wide)]
    for config in each_config(recurrence.StepLaunch, functools.partial(recurrence.step, hooks), given):
        with tuning.pinned({recurrence.StepLaunch: config}):
            out, history, state = jax.jit(recurrence.step, static_argnums=0)(hooks, given)
        for name, mine, theirs, exact in (('output', out, reference[0], truth[0]),
                                          ('state', state, reference[2], truth[2])):
            assert_as_exact_as_the_reference(np.asarray(mine), np.asarray(theirs), exact,
                                             f'{config} {name}')
        np.testing.assert_array_equal(np.asarray(out)[1], 0.0)
        np.testing.assert_array_equal(np.asarray(state)[1], np.asarray(given.state)[1])
        if hooks.conv is not None:
            np.testing.assert_array_equal(np.asarray(history), np.asarray(reference[1]))
            np.testing.assert_array_equal(np.asarray(history)[1], np.asarray(given.held)[1])


@pytest.mark.parametrize('hooks', [HOOKS[0], HOOKS[-1]], ids=str)
def test_a_bf16_state_steps_as_its_float64_form(hooks):
    """A bf16 state is read and stepped in fp32 and written once, rounded stochastically by
    `round_to_bf16` from the token's key and value: within the rule of the XLA form's own bf16 run
    against the float64 step, and the idle row's state its input's bits."""
    given = operands(hooks, seed=1, state_dtype=jnp.bfloat16)
    _, _, state = recurrence.step(hooks, given)
    assert state.dtype == jnp.bfloat16
    with jax.enable_x64(new_val=True):
        wide = recurrence.Step(**{name: None if value is None else widened(value)
                                  for name, value in vars(given).items()})
        exact = np.asarray(recurrence.step_reference(hooks, wide)[2])
    reference = np.asarray(recurrence.step_reference(hooks, given)[2], np.float32)
    assert_as_exact_as_the_reference(np.asarray(state, np.float32), reference, exact, 'the bf16 state')
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
    rng = np.random.default_rng(3)
    for dtype in (jnp.float32, jnp.bfloat16):
        given = recurrence.Step(jnp.asarray(rng.standard_normal((ROWS, 1, 16, 16)) * 0.3, jnp.float32),
                                token=jnp.asarray(rng.standard_normal((ROWS, channels)), dtype),
                                held=jnp.zeros((ROWS, 0, channels), dtype),
                                taps=jnp.asarray(rng.standard_normal((channels, 1)), dtype))
        for step in (recurrence.step, recurrence.step_reference):
            assert step(one_tap, given)[1].shape == (ROWS, 0, channels)
        with jax.enable_x64(new_val=True):
            wide = recurrence.Step(**{name: None if value is None else widened(value)
                                      for name, value in vars(given).items()})
            exact = recurrence.step_reference(one_tap, wide)
        found, reference = recurrence.step(one_tap, given), recurrence.step_reference(one_tap, given)
        for index, name in ((0, 'output'), (2, 'state')):
            mine, theirs = (np.asarray(x[index], np.float32) for x in (found, reference))
            assert_as_exact_as_the_reference(mine, theirs, np.asarray(exact[index]),
                                             f'the one-tap {jnp.dtype(dtype).name} {name}')
