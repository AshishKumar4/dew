"""The linear recurrence's decode step (`dew.nn.kernels.recurrence.step`): every combination of its
hooks against its XLA form in float64, at every config of its launch, held where it rounds (its
graph, `recurrence.step_graph`), and the continuation contract a decode token keeps (an idle row
untouched, a bf16 state rounded as it is held)."""

import dataclasses
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference, assert_graph_as_exact, widened

from dew.nn.kernels import recurrence, tuning

ROWS, KEY_HEADS, VALUE_HEADS, KEY_WIDTH, VALUE_WIDTH, TAPS = 16, 2, 4, 16, 32, 4
"""Each row a draw of the operands, and a step's error measured over all of them: three rows' swung
as far as 2.65x of the reference's over 12 seeds, while the twelve together sat within 1.34x."""

NORMS = (None, recurrence.OutputNorm(1e-6, 'silu'), recurrence.OutputNorm(1e-6, 'sigmoid'))
HOOKS = [recurrence.Hooks(conv=conv, l2norm=norm, decay='scalar' if decay == 'gates' else decay,
                          gates=recurrence.Gates() if decay == 'gates' else None, delta=delta, skip=skip,
                          output=output)
         for conv, norm, decay, delta, skip, output in itertools.product(
             (None, recurrence.Conv('silu')), (False, True), ('none', 'scalar', 'vector', 'gates'),
             (False, True), (False, True), NORMS)
         if decay != 'gates' or delta]


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
        if hooks.delta:
            given['beta'] = jax.nn.sigmoid(normal(ROWS, VALUE_HEADS))
    if hooks.skip:
        given['skip'] = normal(VALUE_HEADS)
    if hooks.output is not None:
        given |= {'z': normal(ROWS, VALUE_HEADS, VALUE_WIDTH), 'weight': normal(VALUE_WIDTH)}
    return recurrence.Step(**given)


def configs(hooks: recurrence.Hooks) -> list[recurrence.Launch]:
    """The launch's space for these hooks at the test's widths."""
    if recurrence._whole(hooks):
        return [recurrence.Launch(0, warps) for warps in (4, 8, 16)]
    return [recurrence.Launch(0, 8), *(recurrence.Launch(block, 4) for block in (16, 32))]


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
    for config in configs(hooks):
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


def orbit(hooks: recurrence.Hooks, inputs: dict):
    """Orders that permute every head's key dimensions alike and its value dimensions alike, exact
    symmetries of a decode token, each with the map taking its output's value dimensions and its
    state's both axes back."""
    def draw(order: int):
        rng = np.random.default_rng(order)
        keys = np.arange(KEY_WIDTH) if order == 0 else rng.permutation(KEY_WIDTH)
        values = np.arange(VALUE_WIDTH) if order == 0 else rng.permutation(VALUE_WIDTH)
        moved = {name: np.asarray(array) for name, array in inputs.items()}

        def take(name, units, axis):
            moved[name] = np.take(moved[name], units, axis=axis)

        if hooks.conv is not None:
            channels = np.concatenate([*(head * KEY_WIDTH + keys for head in range(2 * KEY_HEADS)),
                                       *(2 * KEY_HEADS * KEY_WIDTH + head * VALUE_WIDTH + values
                                         for head in range(VALUE_HEADS))])
            for name, axis in (('token', -1), ('held', -1), ('taps', 0)):
                take(name, channels, axis)
        else:
            for name, units in (('query', keys), ('key', keys), ('value', values)):
                take(name, units, -1)
        if hooks.decay == 'vector' and hooks.gates is None:
            take('log_decay', keys, -1)
        take('state', keys, 2)
        take('state', values, 3)
        if hooks.output is not None:
            take('z', values, -1)
            take('weight', values, -1)

        def back(outputs):
            state = np.take(outputs['next_state'], np.argsort(keys), axis=2)
            return {'output': np.take(outputs['output'], np.argsort(values), axis=-1),
                    'next_state': np.take(state, np.argsort(values), axis=3)}
        return moved, back
    return draw


@pytest.mark.parametrize('hooks', HOOKS, ids=str)
def test_every_hook_combination_rounds_where_its_reference_does(hooks):
    """A decode token held where it rounds (bible §7's gate, tests/reference_error.py
    `assert_graph_as_exact`), at each config of its launch's space: the launch equal to its stages run
    one by one, bit for bit; each stage on common parents, the XLA reference's and the launch's own,
    against the reference's stage; and its output and state over orders of the key and value
    dimensions against the reference's."""
    given = dataclasses.replace(operands(hooks), active=None)
    inputs = {name: getattr(given, name) for name in recurrence._names(hooks)}
    reference = recurrence.reference_graph(hooks, given)
    truth = recurrence.reference_graph(hooks, given, float64=True)
    for config in configs(hooks):
        assert_graph_as_exact(recurrence.step_graph(hooks, given, config=config), reference, truth, inputs,
                              orbit(hooks, inputs), str(config))
