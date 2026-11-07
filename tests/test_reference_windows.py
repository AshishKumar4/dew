"""Check 1 over windows of stacked draws (`over_windows`, `windows`) and an arm declared to vary from run to
run around them (`run_to_run`): a site's own draws stacked and every arm taking them, each window one draw's
program, and the arms' policies kept through the loop."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_windows_stack_a_sites_own_draws_and_its_arms_take_them_all():
    """`windows(draw, count)`'s seed `s` is the site's draws of seeds `s * count` on, stacked, and an arm
    over them, jitted or on the host, gives each window's own output, stacked, as host arrays."""
    from reference_error import over_windows, windows

    def draw(seed):
        rng = np.random.default_rng(seed)
        return rng.normal(size=(3, 4)).astype(np.float32), np.int32(seed)

    def arm(x, seed):
        return {"sum": jnp.sum(x, axis=-1) + seed, "x": x}

    stacked = windows(draw, 5)(2)
    for index in range(5):
        for got, want in zip(jax.tree.leaves(stacked), jax.tree.leaves(draw(10 + index)), strict=True):
            np.testing.assert_array_equal(got[index], want)
    host = over_windows(lambda x, seed: jax.tree.map(np.asarray, arm(x, seed)))
    for wrapped in (over_windows(jax.jit(arm)), host):
        found = wrapped(*stacked)
        assert all(type(leaf) is np.ndarray for leaf in jax.tree.leaves(found))
        for index in range(5):
            wanted = jax.tree.leaves(arm(*draw(10 + index)))
            for got, want in zip(jax.tree.leaves(found), wanted, strict=True):
                np.testing.assert_array_equal(got[index], np.asarray(want))


def test_windows_keep_a_host_draws_dtypes():
    """A host array of a window keeps its dtype, float16 and int64 alike, as one draw holds it."""
    from reference_error import windows

    stacked = windows(lambda seed: (np.array([1 / 3], np.float16), np.int64(2**40 + seed)), 2)(0)
    assert stacked[0].dtype == np.float16 and stacked[1].dtype == np.int64
    assert stacked[1][1] == 2**40 + 1


def test_windows_keep_a_weak_scalars_weak_type():
    """A weak JAX scalar stays weak, so a float16 arm computes in float16 over windows as over one draw."""
    from reference_error import over_windows, windows

    draw = windows(lambda seed: (np.array([1 / 3], np.float16), jnp.asarray(1.5)), 2)
    stacked = draw(0)
    assert jax.typeof(stacked[1]).weak_type
    found = over_windows(jax.jit(lambda x, scale: x * scale))(*stacked)
    assert found.dtype == np.float16
    np.testing.assert_array_equal(found[0], np.asarray(jnp.asarray(stacked[0][0]) * 1.5))


def test_a_looped_host_arm_keeps_its_outputs_dtypes():
    """A NumPy reference's float64 outputs stay float64."""
    from reference_error import over_windows

    found = over_windows(lambda x: np.asarray(x, np.float64) * 1.1)(np.ones((2, 3), np.float16))
    assert found.dtype == np.float64


def test_no_windows_are_refused():
    """A seed of no windows, or an arm given none, is no draw."""
    from reference_error import over_windows, windows

    with pytest.raises(ValueError):
        windows(lambda seed: (np.zeros(2),), 0)
    with pytest.raises(ValueError):
        over_windows(jax.jit(lambda x: x))(np.zeros((0, 2), np.float32))


def test_each_window_runs_one_draws_program_compiled_once():
    """Each window calls the program one draw compiles, traced once for all of them over host and JAX
    leaves alike, and never a batched one: an arm whose batching rule computes otherwise gives each
    window one draw's output."""
    from reference_error import over_windows, windows

    traces = []

    @jax.custom_batching.custom_vmap
    def arm(x, y):
        traces.append((x.shape, y.shape))
        return x + y

    @arm.def_vmap
    def batched(axis_size, in_batched, x, y):
        return x + y + 1, in_batched[0]

    def draw(seed):
        return np.full(2, seed, np.float32), jnp.full(2, seed, jnp.float32)

    found = over_windows(jax.jit(arm))(*windows(draw, 5)(1))
    np.testing.assert_array_equal(found, np.repeat(2 * np.arange(5, 10, dtype=np.float32)[:, None], 2, 1))
    assert traces == [((2,), (2,))]


def test_a_without_ynn_arm_runs_over_windows():
    """A `without_ynn` arm, whose compiler options refuse to be nested on the CPU lane, runs over windows
    as one draw's program."""
    from reference_error import over_windows, without_ynn

    x = np.ones((3, 2), np.float32)
    np.testing.assert_array_equal(over_windows(without_ynn(lambda x: x * 2))(x), x * 2)


def test_the_gate_compiles_two_dew_arms_without_ynnpack_and_one_beside_a_given_arm_as_users_run_it():
    """Check 1 compiles its arms itself (`_arms`): two Dew paths each `without_ynn`, one function as
    both arms twice (its reference unrewritten), a window loop's arm within it, an arm compiled so already
    as it is; beside an arm
    run `as_given` (torch, NumPy, a program of its own), which it never traces, the Dew arm with the
    fusions, as users run it. A NumPy arm passes `as_given`, and the verdict holds."""
    from reference_error import _arms, _Windowed, as_given, assert_as_exact, over_windows, without_ynn

    def dew(x):
        return jnp.asarray(x) * 2

    def written(x):
        return x + x

    mine, theirs = _arms(dew, written)
    assert mine.without_ynn and theirs.without_ynn
    np.testing.assert_array_equal(mine(np.ones(3, np.float32)), 2 * np.ones(3, np.float32))
    # One function as both arms compiles twice: the reference runs its programs unrewritten.
    same, again = _arms(dew, dew)
    assert same is not again and same.without_ynn and again.without_ynn
    compiled = without_ynn(dew)
    assert _arms(compiled, written)[0] is compiled
    windowed, _ = _arms(over_windows(dew), written)
    assert isinstance(windowed, _Windowed) and windowed.fn.without_ynn

    host = as_given(lambda x: np.asarray(x) + np.asarray(x))
    mine, theirs = _arms(dew, host)
    assert theirs.wrapped is host and not getattr(mine, "without_ynn", False)
    np.testing.assert_array_equal(mine(np.ones(3, np.float32)), 2 * np.ones(3, np.float32))
    def draw(seed):
        return (np.random.default_rng(seed).standard_normal(8).astype(np.float32),)

    assert_as_exact(dew, host, draw, "a NumPy arm as given", seeds=8)


def test_a_declaration_around_a_window_loop_runs_as_the_loop_inside_it():
    """An arm declared to vary run to run around a window loop (`run_to_run(over_windows(f))`) runs as
    the loop of the declared arm does: a given NumPy reference within it stays given, and the Dew arm is
    compiled, not called eagerly on the stacked draw and refused as a NumPy arm (row 4586)."""
    from reference_error import as_given, assert_as_exact, over_windows, run_to_run, windows

    def dew(x):
        return jnp.asarray(x) * 2

    def numpy(x):
        return np.asarray(x, np.float64) * 2

    def draw(seed):
        return (np.random.default_rng([4586, seed]).standard_normal(8).astype(np.float32),)

    assert_as_exact(run_to_run(over_windows(dew), "atomic adds"),
                    run_to_run(over_windows(as_given(numpy)), "atomic adds"), windows(draw, 3),
                    "a declared window loop", seeds=8)


def test_a_numpy_arm_is_refused_unless_it_is_given_and_windows_keep_their_arms_policy():
    """An arm the gate would compile that returns NumPy values is refused (a NumPy reference compiled
    would compute in JAX); passed `as_given` it runs as given. A window loop of a given arm is
    classified by that arm, so its Dew partner compiles with the fusions as it does without windows.
    A paired sizing of one function against itself stays paired, its draws from SIZING_FROM on."""
    from reference_error import SIZING_FROM, _arms, as_given, assert_as_exact, over_windows, seeds_for

    def dew(x):
        return jnp.asarray(x) * 2

    def numpy(x):
        return np.asarray(x) * 2

    def draw(seed):
        return (np.random.default_rng(seed).standard_normal(8).astype(np.float32),)

    with pytest.raises(TypeError):
        assert_as_exact(dew, numpy, draw, "a NumPy arm unmarked", seeds=8)
    assert_as_exact(dew, as_given(numpy), draw, "a NumPy arm as given", seeds=8)
    windowed, _ = _arms(over_windows(dew), over_windows(as_given(numpy)))
    assert not getattr(windowed.fn, "without_ynn", False)
    seen = []

    def drawn(seed):
        seen.append(seed)
        return draw(seed)

    seeds_for(dew, drawn, "paired against itself", dew_fn=dew)
    assert min(seen[1:]) >= SIZING_FROM
