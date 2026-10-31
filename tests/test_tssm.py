"""TransDreamer's dynamics (`dew.objectives.rl.tssm.TSSM`): a state reads its episode up to itself and
nothing after, bit for bit; a step at a time over the cache computes the window's states, and
imagination from each step of a window computes the states of the window up to it extended by the
imagined step, each within float64 rounding of the other."""

import jax
import jax.numpy as jnp
import numpy as np
from reference_error import assert_computes_the_oracle, chain_roundings

from dew.objectives.rl.tssm import TSSM

ROWS, LENGTH, WIDTH = 3, 7, 16


def tiny(dtype=jnp.float32) -> TSSM:
    return TSSM(action_size=2, stoch=3, classes=4, width=WIDTH, layers=2, heads=2, dtype=dtype)


def variables(model: TSSM):
    return model.init(jax.random.key(0), jnp.zeros((1, 2, 5)), jnp.zeros((1, 2, 2)), jnp.zeros((1, 2), bool))


def window(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A window's inputs `[3, 7, width]` and where episodes start: row 1 at step 3, row 2 at step 0."""
    first = np.zeros((ROWS, LENGTH), bool)
    first[1, 3] = first[2, 0] = True
    return np.random.default_rng([2202, seed]).normal(size=(ROWS, LENGTH, WIDTH)), first


def test_a_state_reads_its_episode_up_to_itself_and_nothing_after():
    """Changing a window's inputs from step t on leaves every state before t as it was, bit for bit, and
    changes the states from t; changing the steps before an episode's start leaves that episode's states
    as they were."""
    model = tiny()
    held = variables(model)
    x, first = window(0)
    states = jax.jit(lambda x: model.apply(held, x, jnp.asarray(first), method=TSSM.states))
    base = np.asarray(states(x))
    for t in range(LENGTH):
        changed = x.copy()
        changed[:, t:] += 1.0
        moved = np.asarray(states(changed))
        np.testing.assert_array_equal(moved[:, :t], base[:, :t])
        assert not np.array_equal(moved[:, t], base[:, t])
    changed = x.copy()
    changed[1, :3] += 1.0
    np.testing.assert_array_equal(np.asarray(states(changed))[1, 3:], base[1, 3:])


def test_a_step_at_a_time_over_its_cache_computes_the_windows_states():
    """Acting's path, a step at a time from an empty cache, its episodes restarting where they start,
    computes the states training's parallel pass does over the window."""
    with jax.enable_x64(new_val=True):
        model = tiny(jnp.float64)
        held = variables(model)
        x, first = window(1)
        parallel = np.asarray(model.apply(held, x, first, method=TSSM.states))

        def stepped(x):
            cache = model.apply(held, ROWS, LENGTH, method=TSSM.empty_cache)
            found = []
            for t in range(LENGTH):
                h, cache = model.apply(held, x[:, t], jnp.asarray(first[:, t]), cache, method=TSSM.step)
                found.append(h)
            return jnp.stack(found, 1)

        roundings = chain_roundings(jax.make_jaxpr(stepped)(x))
        assert_computes_the_oracle(np.asarray(stepped(x)), parallel, "a TSSM's states a step at a time",
                                   roundings=roundings)


def test_imagination_from_each_step_reads_the_window_up_to_it():
    """Imagination's first step from each step k of a window, over the window's cache, computes the
    state the parallel pass computes over the window up to k extended by the imagined step's input."""
    with jax.enable_x64(new_val=True):
        model = tiny(jnp.float64)
        held = variables(model)
        x, first = window(2)
        following = np.random.default_rng([2202, 3]).normal(size=(ROWS * LENGTH, WIDTH))

        def imagined(following):
            cache = model.apply(held, x, first, 2, method=TSSM.window_cache)
            return model.apply(held, following, jnp.zeros(ROWS * LENGTH, bool), cache, method=TSSM.step)[0]

        expected = []
        for row in range(ROWS):
            for k in range(LENGTH):
                extended = np.concatenate([x[row, :k + 1], following[row * LENGTH + k][None]])[None]
                starts = np.concatenate([first[row, :k + 1], [False]])[None]
                expected.append(np.asarray(model.apply(held, extended, starts, method=TSSM.states))[0, -1])
        roundings = chain_roundings(jax.make_jaxpr(imagined)(following))
        assert_computes_the_oracle(np.asarray(imagined(following)), np.stack(expected),
                                   "imagination from a window's steps", roundings=roundings)
