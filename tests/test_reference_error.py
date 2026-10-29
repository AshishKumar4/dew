"""The float64 accuracy rule (tests/reference_error.py) on comparisons whose answer is known: what it
must refuse, and the rounding it must admit."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from reference_error import (
    FACTOR,
    PARITY,
    REDRAW,
    GateFailure,
    Inconclusive,
    as_given,
    assert_as_exact,
    assert_as_exact_as_the_reference,
    assert_leaves_as_exact,
    assert_moves,
    assert_same_artifact,
    declared_seeds,
    distance,
    indexed_recording,
    input_digest,
    null_rate_bound,
    null_runs,
    seeds_for,
    widened,
)


@pytest.mark.skipif(jax.default_backend() != "cpu", reason="WITHOUT_YNN is XLA:CPU's")
def test_without_ynn_computes_what_xla_cpu_computes_with_every_ynnpack_fusion_off(tmp_path):
    """Each of tests/ynn_programs.py's programs under `without_ynn` gives, bit for bit, what the same
    program gives jitted in a process whose XLA:CPU runs with every YNNPACK fusion off."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    import ynn_programs
    from reference_error import without_ynn

    out = tmp_path / "theirs.npz"
    flags = os.environ.get("XLA_FLAGS", "") + " --xla_cpu_experimental_ynn_fusion_type="
    subprocess.run([sys.executable, str(Path(ynn_programs.__file__)), str(out)], check=True,
                   env={**os.environ, "XLA_FLAGS": flags})
    with np.load(out) as theirs:
        for name, value in ynn_programs.outputs(without_ynn).items():
            np.testing.assert_array_equal(np.asarray(value), theirs[name], err_msg=name)


@pytest.mark.skipif(jax.default_backend() != "cpu", reason="WITHOUT_YNN is XLA:CPU's")
def test_without_ynn_takes_a_callers_compiler_options_over_its_own():
    """A caller who asks for YNNPACK's dot fusion gets the dot the default XLA:CPU computes, bit for
    bit."""
    from reference_error import without_ynn
    from ynn_programs import w, x

    dot_fusion = {"xla_cpu_experimental_ynn_fusion_type": "LIBRARY_FUSION_TYPE_DOT"}
    np.testing.assert_array_equal(without_ynn(jnp.dot, compiler_options=dot_fusion)(x, w),
                                  jax.jit(jnp.dot)(x, w))


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
    both arms once, a window loop's arm within it, an arm compiled so already as it is; beside an arm
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
    same, again = _arms(dew, dew)
    assert same is again and same.without_ynn
    compiled = without_ynn(dew)
    assert _arms(compiled, written)[0] is compiled
    windowed, _ = _arms(over_windows(dew), written)
    assert isinstance(windowed, _Windowed) and windowed.fn.without_ynn

    host = as_given(lambda x: np.asarray(x) + np.asarray(x))
    mine, theirs = _arms(dew, host)
    assert theirs is host and not getattr(mine, "without_ynn", False)
    np.testing.assert_array_equal(mine(np.ones(3, np.float32)), 2 * np.ones(3, np.float32))
    def draw(seed):
        return (np.random.default_rng(seed).standard_normal(8).astype(np.float32),)

    assert_as_exact(dew, host, draw, "a NumPy arm as given", seeds=8)


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


def test_further_references_judge_one_dew_arm_each_by_its_own_check_once_a_draw():
    """`also`: one Dew arm against several references on the same draws, each judged by its own check 1
    under its own label, Dew and the truth evaluated once a draw for them all. A Dew arm on par with one
    reference and three times another's error fails against that one alone; a site with every
    reference on par passes; beside a plain Dew arm, a further reference of another kind than the
    reference's is refused, as is a site with several and no named truth."""
    calls = {"dew": 0, "truth": 0}

    def draw(seed):
        rng = np.random.default_rng(seed)
        return rng.standard_normal(4096), *(1e-7 * rng.standard_normal(4096) for _ in range(3))

    def arm(scale, noise, counted=None):
        @as_given
        def run(x, mine, theirs, other):
            if counted:
                calls[counted] += 1
            return {"value": x + scale * {"mine": mine, "theirs": theirs, "other": other}[noise]}
        return run

    @as_given
    def truth(x, *_):
        calls["truth"] += 1
        return {"value": x}

    dew = arm(1.0, "mine", "dew")
    assert_as_exact(dew, arm(1.0, "theirs"), draw, "two references", seeds=16, truth_fn=truth,
                    also={"another": arm(1.0, "other")})
    assert calls == {"dew": 16, "truth": 16}, calls
    with pytest.raises(GateFailure) as failed:
        assert_as_exact(arm(3.0, "mine"), arm(3.0, "theirs"), draw, "one far", seeds=16, truth_fn=truth,
                        also={"another": arm(1.0, "other")})
    assert {finding.leaf for finding in failed.value.findings} == {"one far, against another['value']"}
    with pytest.raises(ValueError):
        assert_as_exact(dew, arm(1.0, "theirs"), draw, "no truth", seeds=16,
                        also={"another": arm(1.0, "other")})

    def plain(x, *_):
        return {"value": x}

    with pytest.raises(ValueError):
        assert_as_exact(plain, plain, draw, "two kinds", seeds=16, truth_fn=truth,
                        also={"given": arm(1.0, "other")})


def test_a_library_arm_holds_dew_to_the_larger_of_its_error_and_the_references():
    """With the owner's library arm (`library_fn`), Dew is held within PARITY of the larger of the
    reference's error and the library arm's, seed by seed, on the leaves the exception covers: a Dew
    1.7 times the reference's error passes beside a library arm 1.8 times it, fails without it, fails
    at 3 times (past the library arm's own by 1.67), and fails on a leaf the exception does not cover;
    a library arm without its ledger row or its leaves is refused, as is one in NumPy the gate would
    compile. Null runs judge as the site does."""
    from reference_error import null_runs

    def draw(seed):
        rng = np.random.default_rng(seed)
        return rng.standard_normal(4096), *(1e-7 * rng.standard_normal(4096) for _ in range(3))

    def arm(scale, noise):
        @as_given
        def run(x, mine, theirs, library):
            value = x + scale * {"mine": mine, "theirs": theirs, "library": library}[noise]
            return {"covered": value, "other": value}
        return run

    @as_given
    def truth(x, *_):
        return {"covered": x, "other": x}

    reference, library, both = arm(1.0, "theirs"), arm(1.8, "library"), ["['covered']", "['other']"]
    assert_as_exact(arm(1.7, "mine"), reference, draw, "library arm", seeds=16, truth_fn=truth,
                    library_fn=library, library_row=1543, library_leaves=both)
    with pytest.raises(GateFailure):
        assert_as_exact(arm(1.7, "mine"), reference, draw, "no library arm", seeds=16, truth_fn=truth)
    with pytest.raises(GateFailure):
        assert_as_exact(arm(3.0, "mine"), reference, draw, "past the library", seeds=16, truth_fn=truth,
                        library_fn=library, library_row=1543, library_leaves=both)
    with pytest.raises(GateFailure) as failed:
        assert_as_exact(arm(1.7, "mine"), reference, draw, "one leaf", seeds=16, truth_fn=truth,
                        library_fn=library, library_row=1543, library_leaves=["['covered']"])
    assert {finding.leaf for finding in failed.value.findings} == {"one leaf['other']"}
    for unnamed in ({"library_leaves": both}, {"library_row": 1543}, {"library_row": 1543,
                                                                      "library_leaves": ["['third']"]}):
        with pytest.raises(ValueError):
            assert_as_exact(arm(1.7, "mine"), reference, draw, "unrecorded", seeds=16, truth_fn=truth,
                            library_fn=library, **unnamed)
    with pytest.raises(TypeError):
        assert_as_exact(arm(1.7, "mine"), reference, draw, "a NumPy library arm", seeds=16, truth_fn=truth,
                        library_fn=lambda x, *_: {"covered": np.full(np.shape(x), 1 / 3),
                                                  "other": np.full(np.shape(x), 1 / 3)},
                        library_row=1543, library_leaves=both)
    verdicts = null_runs(reference, draw, "library arm", seeds=16, runs=2, truth_fn=truth,
                         dew_fn=arm(1.7, "mine"), paired=True, library_fn=library, library_leaves=both)
    assert [(verdict, stage) for verdict, stage, _ in verdicts] == [("pass", 1), ("pass", 1)], verdicts


def test_a_reference_compiled_with_options_of_its_own_runs_as_given():
    """A reference compiled with compiler options of its own (a trainer step's, `as_a_step`) runs as
    given: the gate would nest it in its own jit, which a jit with options of its own refuses (row
    1800)."""
    from reference_error import _arms

    own = jax.jit(lambda x: {"y": x * 2}, compiler_options={"xla_cpu_enable_fast_math": False})
    def doubled(x):
        return {"y": jnp.asarray(x) + jnp.asarray(x)}

    mine, theirs = _arms(doubled, as_given(own))
    assert theirs.as_given and not getattr(mine, "without_ynn", False)
    # Unmarked, the gate nests it in its own jit, which a jit with options of its own refuses.
    _, nested = _arms(doubled, own)
    with pytest.raises(ValueError):
        nested(np.ones(4, np.float32))
    assert_as_exact(doubled, as_given(own),
                    lambda seed: (np.random.default_rng(seed).standard_normal(64).astype(np.float32),),
                    "a reference with options", seeds=8)


def test_a_float64_truth_runs_as_one_compiled_program():
    """The gate compiles a JAX truth (`_truth_program`): its first call runs as given and every later
    one is the compiled program's, whose values are the eager truth's own. A NumPy truth (its first
    call returns host values) runs as given, and so do a truth `as_given`, one compiled already and a
    window loop's by its own; a check over a compiled truth judges as over the eager one."""
    from reference_error import _Truth, _truth_program, _Windowed, as_given, over_windows

    calls = []

    def truth(x, w):
        calls.append(isinstance(x, jax.core.Tracer))
        return {"y": jnp.dot(x, w, precision="highest")}

    with jax.enable_x64(new_val=True):
        program = _truth_program(truth)
        x, w = (np.asarray(value, np.float64) for value in _draw(0))
        eager = program(x, w)
        compiled = program(x, w)
        assert calls[0] is False and True in calls
        np.testing.assert_allclose(compiled["y"], eager["y"], rtol=1e-14)
        numpy = _truth_program(lambda x, w: {"y": np.asarray(x) @ np.asarray(w)})
        numpy(x, w)
        assert numpy.run is numpy.fn
        # A truth that widens its inputs on the host and returns JAX values fails at its first call.
        hosted = _truth_program(lambda x, w: truth(np.asarray(x, np.float64), np.asarray(w, np.float64)))
        with pytest.raises(jax.errors.TracerArrayConversionError):
            hosted(x, w)
    given = as_given(truth)
    assert _truth_program(given) is given and _truth_program(_product) is _product
    assert _truth_program(None) is None
    windowed = _truth_program(over_windows(truth))
    assert isinstance(windowed, _Windowed) and isinstance(windowed.fn, _Truth)
    assert_as_exact(_reordered, _product, _draw, "a compiled truth", seeds=8, truth_fn=truth)


def test_an_arm_of_dew_programs_compiles_them_as_the_gate_compiles_its_other_arm(monkeypatch):
    """Beside another Dew arm, an arm of Dew's own programs (`dew_programs`) runs with Dew's compile
    seam holding WITHOUT_YNN open (`compiled_with`), so a program it compiles through the seam
    (`dew.nn.kernels.options.program`) carries the option, and beside an arm `as_given` it runs as it
    is; the seam adds nothing once the arm returns, and a run's own XLA flag for an option wins."""
    from reference_error import WITHOUT_YNN, _arms, as_given, dew_programs

    from dew.nn.kernels import options

    seen = []

    @dew_programs
    def programs(x):
        seen.append(options.compiler_options())
        return options.program(lambda x: x * 2)(x)

    def written(x):
        return x + x

    mine, _ = _arms(programs, written)
    np.testing.assert_array_equal(mine(np.ones(3, np.float32)), 2 * np.ones(3, np.float32))
    expected = WITHOUT_YNN if jax.default_backend() == "cpu" else {}
    assert seen == [expected] and options.compiler_options() == {}
    mine, _ = _arms(programs, as_given(lambda x: np.asarray(x) * 2))
    mine(np.ones(3, np.float32))
    assert seen[-1] == {}
    with options.compiled_with({"xla_cpu_experimental_ynn_fusion_type": ""}):
        assert options.compiler_options() == {"xla_cpu_experimental_ynn_fusion_type": ""}
        monkeypatch.setenv("XLA_FLAGS", "--xla_cpu_experimental_ynn_fusion_type=dot")
        assert options.compiler_options() == {}
    assert options.compiler_options() == {}


def test_a_recordings_measured_errors_size_its_site_as_seeds_for_would():
    """`seeds_resolving` over the reference's errors measured apart (a recording's merged shards) gives
    `seeds_for`'s count for the reference compared with itself on the same draws, and refuses a
    recording of too few seeds."""
    from reference_error import _drawn, seeds_for, seeds_resolving

    count = seeds_for(_product, _draw, "product")
    draws = _drawn(_product, _product, _draw, range(4 * count))
    assert seeds_resolving(draws.theirs, draws.names, "product") == count
    with pytest.raises(AssertionError):
        seeds_resolving(draws.theirs[:count], draws.names, "product", start=count)
    # Past its ceiling it refuses, as seeds_for does, however many seeds are recorded.
    with pytest.raises(AssertionError):
        seeds_resolving(draws.theirs, draws.names, "product", most=count // 2)


def test_a_torch_arm_pulls_back_the_cotangent_it_is_given():
    """`torch_with_gradients` gives, bit for bit, a torch function's output and the gradients torch's
    autograd pulls back from the cotangent passed last, cast to the output's dtype, at float32 and at
    float64; a module's parameters are inputs through `functional_call`; an input the output does not
    reach has a zero gradient."""
    from reference_error import torch_with_gradients

    rng = np.random.default_rng(0)
    for dtype in (torch.float32, torch.float64):
        layer = torch.nn.Linear(3, 4).to(dtype)
        x, unused = (rng.normal(size=shape) for shape in ((5, 3), (2,)))
        params = {name: value.detach().numpy() for name, value in layer.named_parameters()}
        x, unused = x.astype(params["weight"].dtype), unused.astype(params["weight"].dtype)
        # A float64 cotangent, cast to the output's dtype.
        cotangent = rng.normal(size=(5, 4))

        def forward(x, params, unused, layer=layer):
            return torch.tanh(torch.func.functional_call(layer, params, (x,)))

        out, grad_x, grads, grad_unused = torch_with_gradients(forward)(x, params, unused, cotangent)
        tensors = [torch.from_numpy(x).requires_grad_(), *(torch.from_numpy(v).requires_grad_()
                                                           for v in params.values())]
        expected = torch.tanh(torch.func.functional_call(layer, dict(zip(params, tensors[1:], strict=True)),
                                                         (tensors[0],)))
        wanted = torch.autograd.grad(expected, tensors, torch.from_numpy(cotangent).to(dtype))
        assert out.dtype == grad_x.dtype == params["weight"].dtype
        np.testing.assert_array_equal(out, expected.detach().numpy())
        for got, want in zip((grad_x, *(grads[name] for name in params)), wanted, strict=True):
            np.testing.assert_array_equal(got, want.numpy())
        np.testing.assert_array_equal(grad_unused, np.zeros_like(unused))


def test_a_torch_arm_pulls_a_scalar_loss_back_from_its_scalar_one():
    """A scalar input takes a scalar gradient, its shape and dtype kept."""
    from reference_error import torch_with_gradients

    loss, gradient = torch_with_gradients(lambda a: 3 * a * a)(np.float32(2), np.float32(1))
    assert loss.dtype == gradient.dtype == np.float32 and np.shape(gradient) == ()
    np.testing.assert_array_equal(loss, np.float32(12))
    np.testing.assert_array_equal(gradient, np.float32(12))


def test_a_torch_arm_pulls_nothing_back_from_a_constant():
    """A constant output pulls nothing back, alone, or beside a reached one whose own cotangent it then
    takes, by its place in the tree."""
    from reference_error import torch_with_gradients

    x = np.array([2.0, 3.0], np.float32)
    _, gradient = torch_with_gradients(lambda x: torch.ones_like(x))(x, np.ones(2, np.float32))
    assert gradient.dtype == np.float32
    np.testing.assert_array_equal(gradient, np.zeros(2, np.float32))
    _, gradient = torch_with_gradients(lambda x: (torch.ones_like(x), x * 2))(
        x, (np.full(2, 7, np.float32), np.full(2, 100, np.float32)))
    np.testing.assert_array_equal(gradient, np.full(2, 200, np.float32))


def test_a_torch_arms_integer_input_has_no_gradient():
    """An integer input takes no gradient, where a float one would."""
    from reference_error import torch_with_gradients

    _, gradient = torch_with_gradients(lambda i: i.float() * 2)(np.array([1, 2], np.int32),
                                                                 np.ones(2, np.float32))
    assert gradient is None


def test_a_torch_arm_refuses_a_cotangent_of_another_tree():
    """A cotangent of another tree than the output's, a dict for a tuple or a list for one, is refused,
    where pairing by order would pull back the wrong leaf."""
    from reference_error import torch_with_gradients

    pulled = torch_with_gradients(lambda v: (v * 1, v * 2))
    x = np.ones(2, np.float32)
    for cotangent in ({"z": np.ones(2, np.float32), "a": np.ones(2, np.float32)}, [x, x]):
        with pytest.raises(ValueError):
            pulled(x, cotangent)


def test_a_torch_arm_pairs_each_cotangent_with_its_own_output():
    """Each cotangent pulls back through its own output, the first through the first."""
    from reference_error import torch_with_gradients

    _, gradient = torch_with_gradients(lambda v: (v * 1, v * 2))(
        np.ones(2, np.float32), (np.full(2, 10, np.float32), np.full(2, 100, np.float32)))
    assert gradient.dtype == np.float32
    np.testing.assert_array_equal(gradient, np.full(2, 210, np.float32))


def test_a_torch_arms_narrow_floats_go_in_through_float32():
    """A bfloat16 input and cotangent, as JAX holds them on the host, and a float8 input reach the arm
    in torch's format of their name, through the float32 that holds them exactly, and come out in
    float32."""
    from reference_error import torch_with_gradients

    def doubled(x):
        assert x.dtype == torch.bfloat16
        return x * 2

    halves = np.array([2.0, 3.0], jnp.bfloat16)
    out, gradient = torch_with_gradients(doubled)(halves, np.ones(2, jnp.bfloat16))
    assert out.dtype == gradient.dtype == np.float32
    np.testing.assert_array_equal(out, np.array([4.0, 6.0], np.float32))
    np.testing.assert_array_equal(gradient, np.full(2, 2, np.float32))
    # Torch computes no float8 on the CPU, so a float8 input is widened inside, as a quantized arm's is.

    def widened_inside(x):
        assert x.dtype == torch.float8_e4m3fn
        return x.float() * 2

    out, gradient = torch_with_gradients(widened_inside)(np.array([2.0, 3.0], jnp.float8_e4m3fn),
                                                         np.ones(2, np.float32))
    assert gradient.dtype == np.float32
    np.testing.assert_array_equal(gradient, np.full(2, 2, np.float32))


def test_a_torch_arms_narrow_integer_input_has_no_gradient():
    """An int4 input, which NumPy holds as ml_dtypes' too, is no float: it enters as int32, its values
    kept, and takes no gradient."""
    from reference_error import torch_with_gradients

    out, gradient = torch_with_gradients(lambda i: i.float() * 2)(np.array([1, -2], jnp.int4),
                                                                   np.ones(2, np.float32))
    assert gradient is None
    np.testing.assert_array_equal(out, np.array([2.0, -4.0], np.float32))


def test_a_torch_arms_bfloat16_output_comes_out_through_float32():
    """A bfloat16 output of float32 inputs comes back as the float32 that holds it, its values exactly."""
    from reference_error import torch_with_gradients

    x = np.arange(4, dtype=np.float32) / 3
    out, gradient = torch_with_gradients(lambda x: (x.bfloat16() * 3).bfloat16())(x, np.ones(4, np.float32))
    assert out.dtype == gradient.dtype == np.float32
    np.testing.assert_array_equal(out, (torch.from_numpy(x).bfloat16() * 3).float().numpy())
    np.testing.assert_array_equal(gradient, np.full(4, 3, np.float32))


def test_a_small_row_beside_large_ones_is_held_to_its_own_size():
    """Dew doubles a row of 5s beside rows of 1e8 that the reference rounds by 8. Every
    tensor-wide distance passes it: its RMS error, 3.5, sits under twice the reference's, 5.7.
    Held to its own size, the row is 100% wrong against the reference's 8e-8."""
    truth = np.array([[1e8, 1e8], [5.0, 5.0]])
    reference = np.array([[1e8 + 8, 1e8 + 8], [5.0, 5.0]])
    dew = np.array([[1e8, 1e8], [10.0, 10.0]])
    assert distance(dew, truth) <= FACTOR * distance(reference, truth)
    with pytest.raises(AssertionError):
        assert_as_exact_as_the_reference(dew, reference, truth, "small row")
    with pytest.raises(AssertionError):
        assert_leaves_as_exact({"rows": dew}, {"rows": reference}, {"rows": truth})


def test_a_row_that_cancels_is_held_to_what_it_sums():
    """A query that keeps one key has a zero gradient: its terms, about 1 each, cancel. Both
    sides round them by about a float32 unit, which over the row's near-zero float64 value is a
    ratio near 1e10 and over its terms (`float64_terms`) one unit, so Dew rounding as the
    reference does passes. Dew ten times further in that row alone moves the RMS over 512 rows
    by 9%, which the tensor-wide rule admits; the row, held to its terms, does not pass."""
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(512, 64))
    truth[7] = 1e-17
    terms = np.abs(truth)
    terms[7] = 1.0
    unit = float(np.finfo(np.float32).eps)

    def rounded(cancelling=1.0):
        noise = unit * terms * rng.choice([-1.0, 1.0], size=truth.shape)
        noise[7] *= cancelling
        return truth + noise

    reference, further = rounded(), rounded(10.0)
    assert_leaves_as_exact({"dq": rounded()}, {"dq": reference}, {"dq": truth}, terms={"dq": terms})
    assert distance(further, truth) <= FACTOR * distance(reference, truth)
    with pytest.raises(AssertionError):
        assert_leaves_as_exact({"dq": further}, {"dq": reference}, {"dq": truth}, terms={"dq": terms})


def test_a_row_of_float64_zero_is_held_to_the_references_own_row():
    """A sum that cancels exactly before anything its terms see, as a query that keeps one key
    has a zero gradient through the flash formula's delta, has no size of its own: its distance
    is held to the reference's on that same row. Dew at the reference's residue passes; Dew with a
    residue where the reference computes the float64 zero exactly does not, though the tensor-wide
    RMS admits it."""
    rng = np.random.default_rng(1)
    truth = rng.normal(size=(64, 16))
    truth[0] = 0.0
    unit = float(np.finfo(np.float32).eps)
    reference = truth + unit * rng.normal(size=truth.shape)
    dew = truth + unit * rng.normal(size=truth.shape)
    assert_as_exact_as_the_reference(dew, reference, truth, "zero row")
    reference[0], dew[0] = 0.0, unit
    assert distance(dew, truth) <= FACTOR * distance(reference, truth)
    with pytest.raises(AssertionError):
        assert_as_exact_as_the_reference(dew, reference, truth, "zero row")


def _draw(seed):
    """A product's operands, fresh for each seed and the same on every run."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(64, 256)).astype(np.float32), rng.normal(size=(256, 32)).astype(np.float32)


@jax.jit
def _product(x, w):
    return {"y": jnp.dot(x, w, precision="highest")}


@jax.jit
def _reordered(x, w):
    """The same sum in another order: its terms reversed, so it rounds at other partial sums."""
    return {"y": jnp.dot(x[:, ::-1], w[::-1], precision="highest")}


@jax.jit
def _lost_term(x, w):
    return {"y": jnp.dot(x[:, :-1], w[:-1], precision="highest")}


@jax.jit
def _bf16_operands(x, w):
    return {"y": jnp.dot(x.astype(jnp.bfloat16), w.astype(jnp.bfloat16),
                         preferred_element_type=jnp.float32)}


@jax.jit
def _broadcast(x, w):
    """The product with a singleton axis before it: the reference's values, not its shape."""
    return {"y": jnp.dot(x, w, precision="highest")[None]}


@jax.jit
def _renamed(x, w):
    return {"z": jnp.dot(x, w, precision="highest")}


def test_check_one_holds_each_leaf_to_its_truths_path_and_shape():
    """A result with a singleton axis added, whose values broadcast against the truth, or with its key
    renamed, whose values pair with the truth's by position, is not the reference's output: check 1
    fails it before any arithmetic. So does a reference whose shape is not its truth's."""
    for dew_fn in (_broadcast, _renamed):
        with pytest.raises(AssertionError):
            assert_as_exact(dew_fn, _product, _draw, "another tree", seeds=8)
    with pytest.raises(AssertionError):
        assert_as_exact(_product, _broadcast, _draw, "another reference", seeds=8, truth_fn=_product)


def test_check_one_passes_an_equally_exact_sum_over_the_seeds_its_own_reference_resolves():
    """The product summed in reverse rounds as exactly as the reference, by other partial sums: over the
    seeds the reference compared with itself needs (`seeds_for`), its pooled RMS error sits within
    PARITY of the reference's, every run alike."""
    seeds = seeds_for(_product, _draw, "product")
    assert_as_exact(_reordered, _product, _draw, "reordered", seeds=seeds)
    assert_as_exact(_reordered, _product, _draw, "reordered", seeds=seeds)


def test_a_committed_fixture_is_its_references_output_within_twice_a_fresh_runs_error():
    """An artifact check, not a parity (`assert_same_artifact`): a committed fixture whose largest error
    from float64 is up to twice a fresh run's is the reference's output; one further is not, even where
    only one element moves, which an RMS would let through, nor one not finite, or held to a fresh run
    that is not."""
    f64 = np.linspace(-1, 1, 101)
    fresh = f64 + 1e-7
    assert_same_artifact(fresh, f64 - 2e-7, f64, "twice as far")
    with pytest.raises(AssertionError):
        assert_same_artifact(fresh, f64 + 3e-7, f64, "three times as far")
    corrupted = f64.copy()
    corrupted[50] += 5e-7
    assert np.sqrt(np.mean(np.square(corrupted - f64))) < 1e-7
    with pytest.raises(AssertionError):
        assert_same_artifact(fresh, corrupted, f64, "one element moved")
    # Neither a NaN nor an infinity on either side is an evaluation of the reference.
    for committed, recomputed in ((np.full_like(f64, np.nan), fresh), (f64 + 1e-7, np.full_like(f64, np.inf)),
                                  (np.full_like(f64, np.inf), np.full_like(f64, np.inf))):
        with pytest.raises(AssertionError):
            assert_same_artifact(recomputed, committed, f64, "not finite")


def test_a_runs_learning_rates_part_from_the_base_at_any_float32_difference(monkeypatch):
    """compare.py's conditions read each run's learning rates as float32s, which must be the base run's
    exactly: a float64 schedule that rounds to the base's float32 matches it, and a value at the next
    float32 parts, which a relative 1e-6 let through."""
    from tool_support import REPO_ROOT, load

    monkeypatch.syspath_prepend(str(REPO_ROOT / "tools" / "reference_runs"))
    compare = load("reference_runs/compare")
    config = dict.fromkeys(compare.CONDITIONS, 1)
    base = np.float32([1e-3, 5e-3, 1e-2])
    rounding = base.astype(np.float64) + 1e-12
    next_up = np.nextafter(base, np.float32(1)).astype(np.float64)
    records = {"base": {"config": config, "lr": base.tolist()}}
    assert not compare.conditions({**records, "run": {"config": config, "lr": rounding.tolist()}})
    [found] = compare.conditions({**records, "run": {"config": config, "lr": next_up.tolist()}})
    assert found.startswith("lr of run parts from base at 3 of 3 steps, first at step 0")


@pytest.mark.parametrize("wrong", [_lost_term, _bf16_operands], ids=["lost term", "bf16 operands"])
def test_check_one_fails_a_sum_that_differs_rather_than_rounds(wrong):
    """A lost term, and operands rounded to bf16, put the pooled ratio far past PARITY, and drawing
    REDRAW times as many seeds again does not bring it back."""
    with pytest.raises(AssertionError) as raised:
        assert_as_exact(wrong, _product, _draw, "wrong", seeds=seeds_for(_product, _draw, "product"))
    assert raised.type is GateFailure


def test_a_failing_site_raises_its_findings_as_data():
    """A lost term fails check 1's pooled ratio: its finding names the leaf, the rule and a measure past
    PARITY, and the failure's message is its findings', one a line."""
    with pytest.raises(GateFailure) as raised:
        assert_as_exact(_lost_term, _product, _draw, "lost", seeds=seeds_for(_product, _draw, "product"))
    assert str(raised.value) == "\n".join(map(str, raised.value.findings))
    pooled = [finding for finding in raised.value.findings if finding.rule == "pooled"]
    assert pooled and all(finding.leaf.startswith("lost") and finding.measure > math.log(PARITY)
                          for finding in pooled)


def test_check_one_is_inconclusive_where_its_seeds_cannot_resolve_the_factor():
    """Errors set seed by seed, alike on every host: Dew's squared error is 4 on even seeds and 0.01 on
    odd ones, the reference's 1 on each, so the pooled ratio (1.42) is under PARITY but its interval
    straddles it and is wider than it over 8 seeds and over the second stage's 40 alike: inconclusive,
    never a pass. Fewer seeds than FEWEST_SEEDS are inconclusive whatever they read."""
    def errors(seed):
        return (np.zeros(64, np.float32), np.full(64, 2.0 if seed % 2 == 0 else 0.1, np.float32),
                np.ones(64, np.float32))

    # NumPy arms, as given: the error law is set on the host, seed by seed.
    @as_given
    def dew(x, mine, theirs):
        return {"y": x + mine}

    @as_given
    def reference(x, mine, theirs):
        return {"y": x + theirs}

    def truth(x, mine, theirs):
        return {"y": x}

    with pytest.raises(Inconclusive):
        assert_as_exact(dew, reference, errors, "straddling", seeds=8, truth_fn=truth)
    with pytest.raises(Inconclusive):
        assert_as_exact(_reordered, _product, _draw, "two seeds", seeds=2)
    # And seeds_for never declares fewer, whatever it is asked to start from.
    assert seeds_for(_product, _draw, "product", start=1) >= 8


def _set_errors(even: float, odd: float):
    """A draw whose Dew's squared error is `even` on even seeds and `odd` on odd ones, the reference's 1
    on each: a site's error law set seed by seed, alike on every host."""
    def errors(seed):
        return (np.zeros(64, np.float32), np.full(64, np.sqrt(even if seed % 2 == 0 else odd), np.float32),
                np.ones(64, np.float32))

    # NumPy arms, as given: the error law is set on the host, seed by seed.
    @as_given
    def dew(x, mine, theirs):
        return {"y": x + mine}

    @as_given
    def reference(x, mine, theirs):
        return {"y": x + theirs}

    def truth(x, mine, theirs):
        return {"y": x}

    return dew, reference, errors, truth


def test_check_one_decides_at_its_second_stage_what_its_first_leaves_inconclusive(tmp_path, monkeypatch):
    """Dew's squared error 2.6 on even seeds and 0.1 on odd ones, the reference's 1: over 8 seeds the
    pooled ratio's interval straddles PARITY and is wider than it; the second stage's 40 resolve it
    under PARITY, so the site passes, and its stage is logged, as every judgement's is."""
    import json

    from reference_error import STAGES_LOG, _interval, _pooled, _status

    dew, reference, errors, truth = _set_errors(2.6, 0.1)
    first = np.array([2.6 if seed % 2 == 0 else 0.1 for seed in range(8)])
    assert _status(_pooled(first, np.ones(8)), *_interval(first, np.ones(8))) == "inconclusive"
    log = tmp_path / "stages.jsonl"
    monkeypatch.setenv(STAGES_LOG, str(log))
    assert_as_exact(dew, reference, errors, "second stage", seeds=8, truth_fn=truth)
    assert_as_exact(_reordered, _product, _draw, "first stage", seeds=8)
    logged = [json.loads(line) for line in log.read_text().splitlines()]
    assert [(entry["label"], entry["stage"], entry["verdict"]) for entry in logged] == [
        ("second stage", 2, "pass"), ("first stage", 1, "pass")]


def test_a_leafs_interval_is_widened_to_hold_its_ratio():
    """A percentile interval that leaves out its own ratio is widened to hold it before any verdict: a
    ratio over PARITY under an interval wholly at or under it is no pass, and a ratio that is not a
    number is inconclusive."""
    from reference_error import _status

    assert _status(1.6, 1.0, 1.4) != "pass"
    assert _status(0.9, 1.1, 1.4) == "pass"
    assert _status(1.7, 1.6, 1.8) == "over"
    assert _status(float("nan"), 1.0, 1.2) == "inconclusive"
    # The rule before the two stages read the interval as it was.
    assert _status(1.6, 1.0, 1.4, contained=False) == "over"
    assert _status(1.2, 0.5, 1.6, contained=False) == "inconclusive"


def test_check_one_holds_dew_exact_where_the_reference_is_exact_on_every_seed():
    """Integer-valued operands make the reference's product exact on every seed: Dew must be too."""
    def integral(seed):
        rng = np.random.default_rng(seed)
        return (rng.integers(-4, 5, (16, 8)).astype(np.float32),
                rng.integers(-4, 5, (8, 4)).astype(np.float32))

    shifted = jax.jit(lambda x, w: {"y": jnp.dot(x, w) + 2.0 ** -20})
    exact = jax.jit(lambda x, w: {"y": jnp.dot(x, w)})
    with pytest.raises(AssertionError) as raised:
        assert_as_exact(shifted, exact, integral, "shifted", seeds=8)
    assert raised.type is GateFailure


@pytest.mark.parametrize("rank", [1, 2])
def test_check_one_fails_a_value_that_is_not_finite(rank):
    """A NaN in Dew's leaf fails, whatever the pooled errors would read, a vector's (which has no rows
    to catch it) as a matrix's."""
    def shaped(fn):
        return jax.jit(lambda x, w: {"y": fn(x, w)["y"].reshape(-1) if rank == 1 else fn(x, w)["y"]})

    poisoned = shaped(lambda x, w: {"y": jnp.dot(x, w, precision="highest").at[0, 0].set(jnp.nan)})
    with pytest.raises(AssertionError) as raised:
        assert_as_exact(poisoned, shaped(_product), _draw, "poisoned", seeds=8)
    assert raised.type is GateFailure


def test_check_one_holds_a_row_of_float64_zero_to_the_references_own():
    """A row the float64 truth holds at exactly zero, the reference too, is no further from zero in Dew
    than ROW_FACTOR times the reference on that row: a corruption too small to move the pooled error
    still fails there."""
    def zero_row(offset):
        return jax.jit(lambda x, w: {"y": jnp.concatenate(
            [jnp.dot(x, w, precision="highest"), jnp.full((1, w.shape[1]), offset, jnp.float32)])})

    with pytest.raises(AssertionError) as raised:
        assert_as_exact(zero_row(1e-12), zero_row(0.0), _draw, "zero row", seeds=8)
    assert raised.type is GateFailure


def _tiny_row(dew_ulps):
    """A product's rows and one row of inputs scaled by 2^-60, about 1e-18 of the product: exact in float32
    and float64 alike, so the reference holds it exactly and float64 to its own precision. Dew moves it
    by `dew_ulps` float32 units relative."""
    unit = float(np.finfo(np.float32).eps)

    def run(ulps):
        def product(x, w):
            y = jnp.dot(x, w, precision="highest")
            tiny = x[:1, :w.shape[1]] * 2.0 ** -60
            if ulps:
                tiny = (tiny.astype(jnp.float64) * (1 + ulps * unit)).astype(tiny.dtype)
            return {"y": jnp.concatenate([y, tiny])}
        return product

    return jax.jit(run(dew_ulps)), jax.jit(run(0))


def test_a_small_row_the_reference_measures_is_judged_at_its_own_size():
    """A row 1e-18 of its leaf, which float64 holds to its own precision, lies below RESOLUTION of the
    leaf; the reference carries it exactly, so it is no row of float64 zero: Dew four float32 units off
    there is ordinary rounding judged at the row's own size, as at every row, and passes; four thousand
    units off fails the worst-row rule. The row is held against the reference's worst row over the leaf,
    not its own error on the row, which is exact: the zero rows' rule failed Dew at four units, the
    worst-row rule admits Dew's row up to ROW_FACTOR times the leaf's worst."""
    dew, reference = _tiny_row(4)
    assert_as_exact(dew, reference, _draw, "tiny row", seeds=8)
    further, reference = _tiny_row(4000)
    with pytest.raises(AssertionError) as raised:
        assert_as_exact(further, reference, _draw, "tiny row", seeds=8)
    assert raised.type is GateFailure


def _tiny_row_recording(live):
    """A recording's draw and evaluation of `live` (`_tiny_row`'s reference): each seed's inputs by name,
    and the reference's float32 outputs beside their float64 truth."""
    def draw(seed):
        x, w = _draw(seed)
        return {"x": x, "w": w}

    def evaluate(inputs):
        with jax.enable_x64(new_val=False):
            reference = jax.tree.map(np.asarray, live(inputs["x"], inputs["w"]))
        truth = jax.tree.map(np.asarray, live(*(np.asarray(inputs[n], np.float64) for n in ("x", "w"))))
        return reference, truth
    return draw, evaluate


def test_a_summarized_recording_keeps_the_zero_rows_rule_for_its_small_rows():
    """A recording that keeps its reference's evidence and its truth's sketch alone (`record_case`'s
    summary) keeps no row's own size, and its sketch holds the truth at its leaf's scale, not a tiny
    row's: its rows below RESOLUTION keep the zero rows' rule, so Dew four float32 units off on the row
    the reference holds exactly fails there, where the live reference judges the row at its own size."""
    from tool_support import load

    recorder = load("recorded_references")
    dew, live = _tiny_row(4)
    draw, evaluate = _tiny_row_recording(live)

    def as_inputs(fn):
        return lambda inputs: fn(inputs["x"], inputs["w"])

    with jax.enable_x64(new_val=True):
        recorded = recorder.record_case("site", draw, evaluate, as_inputs(live), summary=True)
    seeds = int(recorded["site.seeds"])
    assert_as_exact(dew, live, _draw, "live", seeds=seeds)
    drawn, reference, truth = indexed_recording(recorded, "site", seeds, inputs_of=draw,
                                                truth_fn=as_inputs(live))
    with pytest.raises(AssertionError) as raised:
        assert_as_exact(lambda index, inputs: as_inputs(dew)(inputs), reference, drawn, "summarized",
                        seeds=seeds, truth_fn=truth)
    assert raised.type is GateFailure


def test_a_recording_made_in_shards_is_the_recording_one_run_makes():
    """A summarized recording made in shards of seeds (`record_shard`), each drawn as one run draws it,
    and joined (`merge_shards`) is the recording `record_case`'s summary makes of the same reference:
    the same declared count, evidence and sketches, whatever order the shards come in. Shards that
    leave a seed out are refused."""
    from tool_support import load

    recorder = load("recorded_references")
    _, live = _tiny_row(4)
    draw, evaluate = _tiny_row_recording(live)

    with jax.enable_x64(new_val=True):
        whole = recorder.record_case("site", draw, evaluate, lambda inputs: live(inputs["x"], inputs["w"]),
                                     summary=True)
        kept = len(whole["site.input_digest"])
        cut = kept // 3
        shards = [recorder.record_shard("site", draw, evaluate, seeds)
                  for seeds in (range(cut, 2 * kept), range(cut))]
        merged = recorder.merge_shards("site", shards)
        assert merged.keys() == whole.keys()
        for key, value in whole.items():
            np.testing.assert_array_equal(merged[key], value, err_msg=key)
        with pytest.raises(ValueError):
            recorder.merge_shards("site", [shards[0], recorder.record_shard("site", draw, evaluate,
                                                                            range(cut - 1))])


def test_a_recording_with_pilot_shards_sizes_paired_on_them_as_the_live_draws_do():
    """A summarized recording made in shards with a run of pilots from SIZING_FROM (`merge_shards`) keeps
    them apart from its verdict's seeds, and its reference carries them: a paired sizing of it takes those
    pilots alone and declares what the live reference's paired sizing declares on the same draws, its
    lookups never past them. Pilots that leave a seed out of their run, or whose leaves are not the
    verdict's shapes, are refused, and so are a seed past them and a pilot drawn again otherwise."""
    from reference_error import SIZING_FROM
    from tool_support import load

    recorder = load("recorded_references")
    draw, evaluate = _tiny_row_recording(_product)

    def product(inputs):
        return _product(inputs["x"], inputs["w"])

    # The product summed in the other order: Dew's own rounding, apart from the reference's on each draw.
    reordered = jax.jit(lambda x, w: {"y": jnp.dot(x[:, ::-1], w[::-1], precision="highest")})
    live = seeds_for(_product, _draw, "live", dew_fn=reordered)
    pilots = range(SIZING_FROM, SIZING_FROM + 2 * live)

    with jax.enable_x64(new_val=True):
        whole = recorder.record_case("site", draw, evaluate, product, summary=True)
        kept = len(whole["site.input_digest"])
        cut = SIZING_FROM + live // 2
        shards = [recorder.record_shard("site", draw, evaluate, seeds)
                  for seeds in (range(cut, pilots.stop), range(kept), range(SIZING_FROM, cut))]
        merged = recorder.merge_shards("site", shards)
        for key, value in whole.items():
            np.testing.assert_array_equal(merged[key], value, err_msg=key)
        assert len(merged["site.pilot.input_digest"]) == len(pilots)
        with pytest.raises(ValueError):
            recorder.merge_shards("site", shards[:2])
        # One pilot shard's leaf apart from the others', and every pilot's apart from the verdict's.
        for at in ((0,), (0, 2)):
            narrow = [dict(shard, **{"site.shape.y": np.asarray([1, 1], np.int64)}) if index in at else shard
                      for index, shard in enumerate(shards)]
            with pytest.raises(ValueError):
                recorder.merge_shards("site", narrow)
        # Pilots of another leaf than the verdict's.
        renamed = [{key.removesuffix(".y") + ".z" if key.endswith(".y") else key: value
                    for key, value in shard.items()} for shard in shards[::2]]
        with pytest.raises(ValueError):
            recorder.merge_shards("site", [shards[1], *renamed])
        straddling = recorder.record_shard("site", draw, evaluate, range(SIZING_FROM - 1, SIZING_FROM + 1))
        with pytest.raises(ValueError):
            recorder.merge_shards("site", [*shards, straddling])
        # Pilots joined onto a recording made whole before them are the pilots merged with its shards.
        joined = recorder.with_pilots("site", whole, shards[::2])
        assert joined.keys() == merged.keys()
        for key, value in merged.items():
            np.testing.assert_array_equal(joined[key], value, err_msg=key)
        with pytest.raises(ValueError):
            recorder.with_pilots("site", joined, shards[::2])

        seeds = int(merged["site.seeds"])
        drawn, reference, truth = indexed_recording(merged, "site", seeds, inputs_of=draw, truth_fn=product)
    assert reference.pilots == pilots
    asked = []

    def recorded(index, inputs):
        asked.append(int(index))
        return reference(index, inputs)
    recorded.recorded, recorded.pilots = True, reference.pilots

    def dew(index, inputs):
        return reordered(inputs["x"], inputs["w"])

    assert seeds_for(recorded, drawn, "recorded", dew_fn=dew, truth_fn=truth) == live
    assert min(asked) >= SIZING_FROM and max(asked) < pilots.stop
    with pytest.raises(AssertionError):
        drawn(pilots.stop)
    with pytest.raises(AssertionError):
        indexed_recording({**merged, "site.pilot.input_digest": merged["site.pilot.input_digest"][::-1]},
                          "site", seeds, inputs_of=draw, truth_fn=product)[0](SIZING_FROM)


def test_a_recording_read_for_some_of_its_leaves_judges_them_alone():
    """A summarized recording read for some of its leaves (`indexed_recording(..., leaves=)`) holds those
    leaves' evidence and sketches, its pilots' too, and every entry the recording shares: its reference
    gives those leaves alone, at a verdict seed and a pilot's, and check 1 judges them over it; a leaf the
    recording lacks is refused."""
    from reference_error import SIZING_FROM
    from tool_support import load

    recorder = load("recorded_references")

    @jax.jit
    def both(x, w):
        y = jnp.dot(x, w, precision="highest")
        return {"y": y, "z": 2 * y}

    def one(inputs):
        return {"y": both(inputs["x"], inputs["w"])["y"]}

    draw, evaluate = _tiny_row_recording(both)
    with jax.enable_x64(new_val=True):
        whole = recorder.record_case("site", draw, evaluate, lambda inputs: both(inputs["x"], inputs["w"]),
                                     summary=True)
        recorded = recorder.with_pilots("site", whole, [recorder.record_shard(
            "site", draw, evaluate, range(SIZING_FROM, SIZING_FROM + 16))])
        seeds = int(recorded["site.seeds"])
        drawn, reference, truth = indexed_recording(recorded, "site", seeds, inputs_of=draw, truth_fn=one,
                                                    leaves=("y",))
        for seed in (0, SIZING_FROM):
            assert set(reference(*drawn(seed))) == {"y"}
        with pytest.raises(ValueError):
            indexed_recording(recorded, "site", seeds, inputs_of=draw, truth_fn=one, leaves=("y", "w"))
    assert_as_exact(lambda index, inputs: one(inputs), reference, drawn, "leaves", seeds=seeds,
                    truth_fn=truth)


def test_a_recorded_truth_over_windows_counts_one_window_and_checks_every_one():
    """A summarized recording of windows read back through `indexed_recording` with a truth over windows
    (`over_windows`): its rounding count traces one window, as one draw's program, so the truth is read
    where a loop that stacks its windows on the host cannot be traced whole, and every window's truth is
    checked against the recording, a change confined to the last window refused."""
    from reference_error import _one_draws_program, chain_roundings, over_windows, windows
    from tool_support import load

    recorder = load("recorded_references")

    def product(inputs):
        return {"y": jnp.dot(inputs["x"], inputs["w"], precision="highest")}

    draw = windows(lambda seed: dict(zip(("x", "w"), _draw(seed), strict=True)), 4)

    def evaluate(inputs):
        with jax.enable_x64(new_val=False):
            reference = over_windows(product)(inputs)
        return reference, over_windows(product)(widened(inputs))

    one = dict(zip(("x", "w"), _draw(0), strict=True))
    program, arguments = _one_draws_program(over_windows(product), (draw(0),))
    assert program is product
    counted = chain_roundings(jax.make_jaxpr(program)(*arguments))
    assert counted == chain_roundings(jax.make_jaxpr(product)(one))
    last = draw(0)["x"][3, 0, 0]

    def moved_last(inputs):
        # The last window's truth moved by a millionth, every other window's as it is.
        return {"y": product(inputs)["y"] * jnp.where(inputs["x"][0, 0] == last, 1 + 1e-6, 1.0)}

    with jax.enable_x64(new_val=True):
        recorded = recorder.record_case("site", draw, evaluate, jax.vmap(product), summary=True)
        seeds = int(recorded["site.seeds"])
        for truth_fn, holds in ((product, True), (moved_last, False)):
            _, _, truth = indexed_recording(recorded, "site", seeds, inputs_of=draw,
                                            truth_fn=over_windows(truth_fn))
            if holds:
                truth(0, widened(draw(0)))
            else:
                with pytest.raises(AssertionError):
                    truth(0, widened(draw(0)))


def _recorded_pairs(dew, reference, count: int, missing=()):
    """`dew` and `reference` recorded at `count` pilot seeds from SIZING_FROM on but the `missing` ones,
    each looked up by its seed, with a truth of zero, and the seeds the reference was asked for."""
    from reference_error import SIZING_FROM

    pilots = range(SIZING_FROM, SIZING_FROM + count)
    mine = {seed: dew(seed) for seed in pilots}
    theirs = {seed: reference(seed) for seed in pilots if seed not in missing}
    asked = []

    def recorded(seed):
        asked.append(int(seed))
        return theirs[int(seed)]

    @as_given
    def recorded_dew(seed):
        return mine[int(seed)]

    def truth(seed):
        return {"y": np.zeros(64, np.float64)}

    recorded.recorded = True
    return recorded, recorded_dew, truth, asked


def test_a_recorded_reference_with_pilots_sizes_paired_on_them_alone():
    """A recorded reference sizes paired only on pilot draws it holds (`recorded_pilots`, a view of its
    own), from SIZING_FROM on, never past them: an error law that never resolves exhausts the 24 pilots
    held at its last doubling (16), and no lookup reaches past them. One without pilots is refused, a
    view leaves its reference as it was, a live reference cannot hold pilots, a declaration of pilots
    the recording lacks is refused at once, and a start past the pilots is refused, not unbound."""
    from reference_error import SIZING_FROM, recorded_pilots

    def draw(seed):
        return (np.int64(seed),)

    def straddling(seed):
        return {"y": np.full(64, 2.0 if seed % 2 == 0 else 0.1, np.float32)}

    def unit(seed):
        return {"y": np.ones(64, np.float32)}

    reference, dew, truth, asked = _recorded_pairs(straddling, unit, 24)
    with pytest.raises(ValueError):
        seeds_for(reference, draw, "no pilots", dew_fn=dew, truth_fn=truth)
    held = recorded_pilots(reference, draw, 24)
    assert not hasattr(reference, "pilots") and held.pilots == range(SIZING_FROM, SIZING_FROM + 24)
    with pytest.raises(AssertionError):
        seeds_for(held, draw, "never resolving", dew_fn=dew, truth_fn=truth)
    assert max(asked) < SIZING_FROM + 24
    with pytest.raises(KeyError):
        recorded_pilots(reference, draw, 25)
    holed, _, _, _ = _recorded_pairs(unit, unit, 24, missing={SIZING_FROM + 5})
    with pytest.raises(KeyError):
        recorded_pilots(holed, draw, 24)
    # Arms that agree exactly resolve at once, at the ceiling itself: half the pilots, the other half
    # its confirming block.
    same, same_dew, zero, _ = _recorded_pairs(unit, unit, 48)
    assert seeds_for(recorded_pilots(same, draw, 48), draw, "at the ceiling", start=24, dew_fn=same_dew,
                     truth_fn=zero) == 24
    with pytest.raises(AssertionError):
        seeds_for(held, draw, "start past the pilots", start=32, dew_fn=dew, truth_fn=truth)
    with pytest.raises(ValueError):
        recorded_pilots(_product, draw, 8)


def test_a_recorded_reference_is_drawn_by_its_index_and_covers_the_redraw(tmp_path):
    """A reference recorded over (1 + REDRAW)·K seeds: each draw reads its seed's inputs, the reference
    and truth read their recorded outputs at it, and check 1 judges Dew against them; a recording too
    short for the redraw is refused before anything is judged."""
    rng = np.random.default_rng(9)
    count, seeds = 40, 8
    x = rng.normal(size=(count, 32)).astype(np.float32)
    # x·x - 1, rounded twice at fp32 alike by NumPy and XLA (XLA's tanh is about 1.6 times libm's error
    # on a CPU, so a transcendental would compare the two libraries, not the recording).
    truth = x.astype(np.float64) ** 2 - 1
    path = tmp_path / "recorded.npz"
    np.savez(path, **{"site.input.x": x, "site.reference.y": x * x - np.float32(1),
                      "site.truth.y": truth, "site.seeds": np.int64(seeds),
                      "site.redraw": np.int64(REDRAW)})
    assert declared_seeds(path) == {"site": seeds}
    with np.load(path) as stored:
        draw, reference, truth_fn = indexed_recording(dict(stored), "site", seeds)
        squared = jax.jit(lambda index, inputs: {"y": inputs["x"] * inputs["x"] - 1})
        assert_as_exact(squared, reference, draw, "recorded", seeds=seeds, truth_fn=truth_fn)
        with pytest.raises(AssertionError):
            indexed_recording(dict(stored), "site", 16)
        for narrow in (np.float32, jnp.bfloat16):
            with pytest.raises(AssertionError):
                indexed_recording({**stored, "site.truth.y": truth.astype(narrow)}, "site", seeds)


def _squared_case(first: int):
    """A recording test's draw, 64 normal fp32 values a seed from `first` on, and its evaluation: the
    reference's `x x - 1` in fp32 beside its float64 truth."""
    def draw(seed):
        return {"x": np.random.default_rng(first + seed).normal(size=(64,)).astype(np.float32)}

    def evaluate(inputs):
        x = inputs["x"]
        return {"y": x * x - np.float32(1)}, {"y": x.astype(np.float64) ** 2 - 1}

    return draw, evaluate


@pytest.mark.parametrize("proxy", [True, False])
def test_a_recording_declares_the_count_its_recorded_reference_resolves(tmp_path, proxy):
    """tools/recorded_references.py records a reference over the seeds its own recording resolves, and
    their redraw; check 1 then judges Dew against the recording by its index."""
    from tool_support import load

    recorder = load("recorded_references")

    draw, evaluate = _squared_case(100)

    def squared(inputs):
        return {"y": inputs["x"] * inputs["x"] - 1}

    recorded = recorder.record_case("site", draw, evaluate, squared if proxy else None)
    seeds = int(recorded["site.seeds"])
    assert seeds >= 8 and len(recorded["site.input.x"]) == 5 * seeds
    path = tmp_path / "recorded.npz"
    np.savez(path, **recorded)
    assert declared_seeds(path) == {"site": seeds}
    with np.load(path) as stored:
        draw_recorded, reference, truth = indexed_recording(dict(stored), "site", seeds)
        assert_as_exact(jax.jit(lambda index, inputs: squared(inputs)), reference, draw_recorded, "recorded",
                        seeds=seeds, truth_fn=truth)
        with pytest.raises(AssertionError):
            assert_as_exact(lambda index, inputs: {"y": jnp.asarray(inputs["x"]) ** 2}, reference,
                            draw_recorded, "recorded with its subtraction missing", seeds=seeds,
                            truth_fn=truth)
    if not proxy:
        with pytest.raises(ValueError):
            recorder.record_case("site", draw, evaluate, None, truth_proxy=squared)


def test_check_one_holds_an_empty_leaf_exact_by_construction():
    """A tree with an empty leaf beside a real one (a one-tap convolution's history): the empty leaf adds
    no error and resolves nothing, so the tree is judged by its real leaf; an empty leaf whose shape or
    dtype is not the reference's fails."""
    def tree(fn):
        return jax.jit(lambda x, w: {"y": fn(x, w)["y"], "history": jnp.zeros((16, 0, 48), jnp.float32)})

    assert_as_exact(tree(_reordered), tree(_product), _draw, "with an empty leaf",
                    seeds=seeds_for(tree(_product), _draw, "with an empty leaf"))
    narrower = jax.jit(lambda x, w: {"y": _reordered(x, w)["y"],
                                     "history": jnp.zeros((16, 0, 48), jnp.bfloat16)})
    with pytest.raises(AssertionError):
        assert_as_exact(narrower, tree(_product), _draw, "a narrower empty leaf", seeds=8)

    def fuller(x, w):
        return {"y": _product(x, w)["y"], "history": jnp.zeros((16, 1, 48), x.dtype)}

    with pytest.raises(AssertionError):
        assert_as_exact(tree(_reordered), tree(_product), _draw, "a truth that is not empty", seeds=8,
                        truth_fn=fuller)


def test_a_recording_keeps_the_confirmed_count_where_the_proxy_planned_more():
    """A proxy whose one element needs more seeds than the recorded reference's 64 do: the recording
    keeps (1 + REDRAW) times the count the recorded reference confirms, not what the proxy planned."""
    from tool_support import load

    recorder = load("recorded_references")

    draw, evaluate = _squared_case(200)

    def one_element(inputs):
        return {"y": jnp.sum(inputs["x"] * inputs["x"])[None]}

    planned = seeds_for(jax.jit(lambda inputs: one_element(inputs)), lambda seed: (draw(seed),), "proxy")
    recorded = recorder.record_case("site", draw, evaluate, one_element)
    seeds = int(recorded["site.seeds"])
    assert planned > seeds and len(recorded["site.input.x"]) == 5 * seeds


def test_an_input_digest_tells_apart_a_scalar_from_its_vector_and_two_formats_of_one_width():
    """Inputs drawn again in another shape or format with the same bytes are not the recorded ones."""
    assert input_digest({"x": np.float32(1)}) != input_digest({"x": np.ones(1, np.float32)})
    one = np.array([1.0, 2.0], jnp.float8_e4m3fn)
    assert input_digest({"x": one}) != input_digest({"x": one.view(jnp.float8_e4m3fnuz)})
    strided = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]
    assert input_digest({"x": strided}) == input_digest({"x": np.ascontiguousarray(strided)})


def test_a_recording_of_outputs_alone_draws_its_inputs_again(tmp_path):
    """A generator's seeded draw, called again by the test (`inputs_of`), stands in for recorded inputs:
    the recording holds the outputs alone and is judged as one that holds both. A draw that changed since
    the recording is refused by its digest, as is `inputs_of` on a recording with no digest."""
    from tool_support import load

    recorder = load("recorded_references")

    draw, evaluate = _squared_case(300)

    def squared(inputs):
        return {"y": inputs["x"] * inputs["x"] - 1}

    recorded = recorder.record_case("site", draw, evaluate, squared, store_inputs=False)
    assert not any(name.startswith("site.input.") for name in recorded)
    seeds = int(recorded["site.seeds"])
    with pytest.raises(AssertionError):
        indexed_recording(recorded, "site", seeds)
    drawn, reference, truth = indexed_recording(recorded, "site", seeds, inputs_of=draw)
    assert_as_exact(jax.jit(lambda index, inputs: squared(inputs)), reference, drawn, "drawn again",
                    seeds=seeds, truth_fn=truth)
    changed, _, _ = indexed_recording(recorded, "site", seeds, inputs_of=lambda seed: draw(seed + 1))
    with pytest.raises(AssertionError):
        changed(0)
    undigested = {name: value for name, value in recorded.items() if name != "site.input_digest"}
    with pytest.raises(AssertionError):
        indexed_recording(undigested, "site", seeds, inputs_of=draw)


def test_null_runs_judge_the_reference_against_itself_on_draws_no_other_run_takes():
    """Each null run is check 1 on the reference against itself, its two sides on disjoint draws and
    every run on draws of its own; an equally exact pair passes, and the red rate's bound is the
    one-sided Clopper-Pearson one, not one over the runs."""
    taken = []

    def draw(seed):
        taken.append(seed)
        return _draw(seed)

    verdicts = null_runs(_product, draw, "null", seeds=8, runs=3)
    assert [verdict for verdict, _, _ in verdicts] == ["pass"] * 3

    # Draws of one truth (an orbit's) pair their rows of float64 zero: Dew's residue on a row the
    # reference holds at zero is no pass for drawing the reference elsewhere.
    def one_truth(seed):
        return np.ones(4, np.float32), np.float32(seed % 2)

    def exact(x, s):
        return {"y": jnp.stack([x * (1 + 1e-7), 0 * x])}

    def residue(x, s):
        return {"y": jnp.stack([x * (1 + 1e-7), 1e-12 * s * x])}

    def truth(x, s):
        return {"y": jnp.stack([x, 0 * x])}

    with pytest.raises(AssertionError):
        assert_as_exact(residue, exact, one_truth, "orbit", seeds=8, truth_fn=truth)
    [(verdict, _, _)] = null_runs(exact, one_truth, "orbit", seeds=8, runs=1, truth_fn=truth, dew_fn=residue)
    assert verdict == "over"
    assert len(taken) == len(set(taken))
    assert null_rate_bound(0, 1400) == pytest.approx(1 - 0.001 ** (1 / 1400), rel=1e-9)
    assert null_rate_bound(0, 200) > 0.03
    assert null_rate_bound(3, 3) == 1.0


def test_a_recording_of_the_references_evidence_alone_holds_the_recomputed_truth_to_its_sketch():
    """A recording that keeps the reference's evidence (`leaf_summary`) and its truth's sketch, no
    outputs: check 1 judges Dew against it with the test's own float64 path as truth, held at every seed
    to the recorded sketch, so a float64 path that computes one element a millionth off is refused, and
    a recording without that path is refused outright."""
    from tool_support import load

    recorder = load("recorded_references")

    def draw(seed):
        rng = np.random.default_rng(400 + seed)
        return {"x": rng.normal(size=(16, 8)).astype(np.float32),
                "w": rng.normal(size=(8, 32)).astype(np.float32)}

    def evaluate(inputs):
        x, w = inputs["x"], inputs["w"]
        return ({"y": np.dot(x, w)},
                {"y": np.dot(x.astype(np.float64), w.astype(np.float64))})

    def product(inputs):
        return {"y": jnp.dot(inputs["x"], inputs["w"], precision="highest")}

    def wide(inputs):
        return {"y": jnp.dot(jnp.asarray(inputs["x"], jnp.float64), jnp.asarray(inputs["w"], jnp.float64))}

    recorded = recorder.record_case("site", draw, evaluate, product, summary=True)
    assert not any(name.startswith(("site.reference.", "site.truth.", "site.input.")) for name in recorded)
    seeds = int(recorded["site.seeds"])
    drawn, reference, truth = indexed_recording(recorded, "site", seeds, inputs_of=draw, truth_fn=wide)
    assert_as_exact(jax.jit(lambda index, inputs: product(inputs)), reference, drawn, "evidence", seeds=seeds,
                    truth_fn=truth)

    def off(inputs):
        y = wide(inputs)["y"]
        return {"y": y.at[3, 5].multiply(1 + 1e-6)}

    _, _, wrong = indexed_recording(recorded, "site", seeds, inputs_of=draw, truth_fn=off)
    with pytest.raises(AssertionError), jax.enable_x64(new_val=True):
        wrong(*widened(drawn(0)))
    _, _, overflowing = indexed_recording(recorded, "site", seeds, inputs_of=draw,
                                          truth_fn=lambda inputs: {"y": wide(inputs)["y"] * 1e300})
    with pytest.raises(AssertionError), jax.enable_x64(new_val=True):
        overflowing(*widened(drawn(0)))
    assert recorded["site.sketch.y"].shape == (len(recorded["site.mse.y"]), 32)
    with pytest.raises(AssertionError):
        indexed_recording(recorded, "site", seeds, inputs_of=draw)


def test_a_truths_sketch_rounds_alike_in_any_order_and_refuses_an_overflowing_or_empty_truth_apart():
    """The sketch of one truth agrees with itself summed in another order within `sketch_rounding`, each
    block of SKETCH_BLOCK summed from its other end as another host may; a
    truth recomputed far past float64's range is refused, not passed on an infinite bound; and an empty
    leaf's evidence is exact, its reference's dtype kept."""
    from reference_error import SKETCH, _sign_bits, leaf_summary, sketch_rounding, truth_sketch

    rng = np.random.default_rng(7)
    truth = rng.normal(size=4096)
    signs = ((_sign_bits(truth.size, 3)[:, None] >> np.arange(SKETCH, dtype=np.uint64)) & 1) * 2.0 - 1
    # Another host: each block summed one by one from its end, the blocks' sums summed exactly.
    blocks = np.flip((truth[:, None] * signs).reshape(-1, 64, SKETCH), axis=1).cumsum(axis=1)[:, -1]
    other_order = np.asarray([math.fsum(blocks[:, index].tolist()) for index in range(SKETCH)])
    allowed = 2 * sketch_rounding(float(np.sum(np.abs(truth))))
    assert np.all(np.abs(truth_sketch(truth, 3) - other_order) <= allowed)
    empty = leaf_summary(jnp.zeros((16, 0), jnp.bfloat16), np.zeros((16, 0)))
    assert (empty.mse, empty.worst, empty.finite, empty.dtype) == (0.0, 0.0, True, "bfloat16")
    with pytest.raises(AssertionError):
        leaf_summary(np.zeros(0), np.ones(3))
    # Two entries swapped are seen at a seed unless their signs agree in all SKETCH places.
    swapped = truth.copy()
    swapped[[0, 1]] = swapped[[1, 0]]
    bits = _sign_bits(truth.size, 3)
    assert (bits[0] == bits[1]) or np.any(truth_sketch(swapped, 3) != truth_sketch(truth, 3))


def test_the_procedure_before_the_two_stages_draws_again_as_it_did():
    """Compared against, the procedure before check 1's two stages (`_staged(two_stage=False)`) draws
    the second set wherever a leaf's ratio is over PARITY, a value not finite included, and only there:
    an inconclusive leaf is final at its declared seeds."""
    from reference_error import NO_ZERO_ROWS, _Draws, _staged

    def law(dew):
        def drawn(start, stop):
            count = stop - start
            return _Draws(["leaf"], np.full((count, 1), dew), np.ones((count, 1)), np.zeros((count, 1, 2)),
                          [[NO_ZERO_ROWS]] * count, np.full((count, 1), np.isfinite(dew)), (0,))
        return drawn

    with np.errstate(invalid="ignore"):
        assert _staged(law(np.inf), 8, "infinite", two_stage=False)[2] == 2
        assert _staged(law(np.inf), 8, "infinite")[2] == 2
    assert _staged(law(1.0), 8, "equal", two_stage=False)[2] == 1


def test_a_change_moves_the_result_only_past_rounding():
    """`assert_moves`: a product scaled by 1 + 2^-10 moves it far past its rounding and passes; the same
    product summed in another order, a change rounding accounts for, does not; a leaf the original holds
    exactly moves on any departure."""
    def scaled(x, w):
        return {"y": jnp.dot(x, w, precision="highest") * (1 + 2.0 ** -10)}

    assert_moves(_product, jax.jit(scaled), _draw, "scaled", seeds=8)
    with pytest.raises(AssertionError):
        assert_moves(_product, _reordered, _draw, "reordered", seeds=8)

    def zero(x, w):
        return {"y": jnp.zeros(3, jnp.float32)}

    def off_zero(x, w):
        return {"y": jnp.full(3, 1e-30, jnp.float32)}

    assert_moves(jax.jit(zero), jax.jit(off_zero), _draw, "off zero", seeds=8)


def test_a_change_moves_by_its_intervals_lower_end_and_one_leaf_is_enough():
    """Errors set seed by seed: a changed RMS error of 1.51 times the original's on every seed (its
    interval that one value) moves; 1.49 does not, and is no move however its seeds are drawn; with two
    leaves, one moved is enough; a leaf not finite moves nothing."""
    for factor, moves in ((1.51, True), (1.49, False)):
        changed, original, errors, truth = _set_errors(factor ** 2, factor ** 2)
        if moves:
            assert_moves(original, changed, errors, "constant", seeds=8, truth_fn=truth)
        else:
            with pytest.raises(AssertionError):
                assert_moves(original, changed, errors, "constant", seeds=8, truth_fn=truth)

    dew, reference, errors, truth = _set_errors(4.0, 4.0)

    @as_given
    def two(x, mine, theirs):
        return {"moved": dew(x, mine, theirs)["y"], "kept": reference(x, mine, theirs)["y"]}

    @as_given
    def two_original(x, mine, theirs):
        return {"moved": reference(x, mine, theirs)["y"], "kept": reference(x, mine, theirs)["y"]}

    def two_truth(x, mine, theirs):
        return {"moved": x, "kept": x}

    assert_moves(two_original, two, errors, "one leaf", seeds=8, truth_fn=two_truth)

    @as_given
    def infinite(x, mine, theirs):
        return {"y": x + np.inf}

    with pytest.raises(AssertionError):
        assert_moves(reference, infinite, errors, "infinite", seeds=8, truth_fn=truth)


def test_a_row_of_float64_zero_is_judged_over_the_runs_seeds():
    """A row of float64 zero is held to the reference's residue on it pooled over the run's seeds
    (`pooled_zero_rows`): one seed's residue far past the reference's own, where the other seeds' are
    alike, passes; Dew ten times the reference's residue on every seed fails, and so does a residue
    where the reference holds the row at exactly zero on every seed."""
    from reference_error import ROW_FACTOR, pooled_zero_rows

    rows = np.asarray([3, 7])

    def seed(mine, theirs):
        return rows, np.asarray(mine, np.float64), np.asarray(theirs, np.float64)

    alike = [seed([1.0, 2.0], [1.0, 2.0]) for _ in range(7)]
    assert pooled_zero_rows([*alike, seed([1.0, 2.0 * ROW_FACTOR * 1.5], [1.0, 2.0])]) == []
    tenfold = [seed([10.0, 2.0], [1.0, 2.0]) for _ in range(8)]
    assert [row for row, _, _ in pooled_zero_rows(tenfold)] == [3]
    exact = [seed([0.0, 1e-30], [0.0, 0.0]) for _ in range(8)]
    assert [row for row, _, _ in pooled_zero_rows(exact)] == [7]
    # A row zero on some seeds alone pools over those.
    assert pooled_zero_rows([seed([1.0, 1.0], [1.0, 1.0]), (np.asarray([3]), np.asarray([5.0]),
                                                             np.asarray([1.0]))]) == []


def test_a_truth_sketch_is_its_signed_sums_bit_for_bit():
    """`truth_sketch` is its definition, bit for bit: each entry times its seed's SKETCH signs (bit k of
    its word, +1 where set), summed in float64 in blocks of SKETCH_BLOCK in order and the blocks summed
    exactly, at sizes off and on a block's edge and past one chunk of 2^16 entries."""
    import math

    from reference_error import SKETCH, SKETCH_BLOCK, _sign_bits, truth_sketch

    for size in (1, SKETCH_BLOCK - 1, 5 * SKETCH_BLOCK, (1 << 16) + 77):
        values = np.random.default_rng(size).standard_normal(size) * 10.0 ** np.random.default_rng(
            size + 1).uniform(-30, 30, size)
        bits = _sign_bits(size, seed=3)
        signs = np.where((bits[:, None] >> np.arange(SKETCH, dtype=np.uint64)) & np.uint64(1), 1.0, -1.0)
        products = values[:, None] * signs
        blocks = []
        for start in range(0, size, 1 << 16):
            chunk = products[start:start + (1 << 16)]
            padded = np.concatenate([chunk, np.zeros((-len(chunk) % SKETCH_BLOCK, SKETCH))])
            blocks.append(padded.reshape(-1, SKETCH_BLOCK, SKETCH).sum(axis=1))
        expected = [math.fsum(column) for column in np.concatenate(blocks).T.tolist()]
        assert truth_sketch(values, seed=3).tolist() == expected


def test_a_leaf_summary_reads_the_same_with_its_truths_rows_read_once():
    """`leaf_summary` over a truth's rows read once (`_TruthRows.of`) is the summary that reads them
    itself, field by field, on a leaf with rows sized, of float64 zero and exactly zero."""
    import dataclasses

    from reference_error import _TruthRows, leaf_summary

    truth = np.random.default_rng(0).standard_normal((6, 5))
    truth[2] = 1e-30
    truth[4] = 0.0
    arm = (truth + 1e-7 * np.random.default_rng(1).standard_normal(truth.shape)).astype(np.float32)
    alone, shared = leaf_summary(arm, truth), leaf_summary(arm, truth, _TruthRows.of(truth))
    for field in dataclasses.fields(alone):
        one, other = getattr(alone, field.name), getattr(shared, field.name)
        assert np.array_equal(one, other) if isinstance(one, np.ndarray) else one == other, field.name



def _bits(value):
    """`value` as its type and bits: a float held to its bits (a NaN's and a zero's sign included), an
    array to its dtype, shape and bytes, a summary field by field."""
    import dataclasses

    from reference_error import RecordedLeaf

    if isinstance(value, RecordedLeaf):
        return [(field.name, _bits(getattr(value, field.name))) for field in dataclasses.fields(value)]
    if isinstance(value, (tuple, list)):
        return [_bits(item) for item in value]
    if value is None or isinstance(value, (bool, str)):
        return type(value).__name__, value
    array = np.asarray(value)
    return type(value).__name__, array.dtype.str, array.shape, array.tobytes()


def _evidence_leaf_by_leaf(dew, reference, truth, their_truth) -> list[tuple]:
    """`_evidence`'s rows as it read them before it read a draw at once: each arm's `leaf_summary` of
    each leaf alone, paired (`_paired`) where the truths are the same values, an empty leaf exact and
    held to the reference's dtype."""
    from reference_error import LOG_ZERO, NO_ZERO_ROWS, RecordedLeaf, _paired, leaf_summary

    row = []
    for (path, leaf), their_leaf, true, their_true in zip(
            jax.tree_util.tree_leaves_with_path(dew), jax.tree.leaves(reference), jax.tree.leaves(truth),
            jax.tree.leaves(their_truth), strict=True):
        recorded = isinstance(their_leaf, RecordedLeaf)
        their = their_leaf if recorded else leaf_summary(their_leaf, their_true)
        if not np.size(leaf):
            kept = str(jnp.dtype(leaf.dtype))
            assert kept == their.dtype, (jax.tree_util.keystr(path), kept, their.dtype)
            row.append((0.0, 0.0, LOG_ZERO, LOG_ZERO, NO_ZERO_ROWS, True))
            continue
        paired = recorded or their_truth is truth or np.array_equal(np.asarray(true), np.asarray(their_true))
        row.append(_paired(jax.tree_util.keystr(path), leaf_summary(leaf, true), their, paired=paired))
    return row



class _RandomLeaves:
    """Random leaves of a draw for check 1's evidence: shapes of every kind it reads, each truth with rows
    of float64 zero, values not finite and scales near float64's range, and arms in four dtypes, laid out
    C-ordered, F-ordered and strided."""

    WIDTHS = (1, 2, 3, 7, 8, 9, 16, 33, 129)

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def shape(self) -> tuple:
        """A scalar's, a vector's, an empty leaf's, rows past NumPy's 8,192-entry buffer, or rows."""
        rng, kind = self.rng, self.rng.integers(12)
        if kind == 0:
            return ()
        if kind == 1:
            return (int(rng.integers(1, 40)),)
        if kind == 2:
            return [(0,), (3, 0), (0, 5)][rng.integers(3)]
        if kind == 3:
            return [(70, 129), (2, 9000)][rng.integers(2)]
        if kind <= 5:
            return int(rng.integers(1, 6)), int(rng.integers(1, 5)), int(rng.choice(self.WIDTHS))
        return int(rng.integers(1, 12)), int(rng.choice(self.WIDTHS))

    def laid_out(self, array):
        """`array` as it is, or F-ordered, or every other entry of a row twice as wide."""
        kind = self.rng.integers(5)
        if kind == 0 and array.ndim >= 2:
            return np.asfortranarray(array)
        if kind == 1 and array.ndim >= 1:
            wide = np.zeros((*array.shape[:-1], 2 * array.shape[-1]), array.dtype)
            wide[..., ::2] = array
            return wide[..., ::2]
        return array

    def truth(self, shape):
        """A float64 truth of `shape`: a few rows zero or 1e-30 of the rest, one in 20 a value not finite."""
        rng = self.rng
        scale = 10.0 ** (rng.uniform(-3, 3) if rng.random() < 0.8 else rng.choice([-300, -160, 154, 300]))
        truth = rng.standard_normal(shape) * scale
        if truth.ndim >= 2 and truth.size:
            rows = truth.reshape(-1, shape[-1])
            for row in rng.choice(len(rows), min(len(rows), int(rng.integers(0, 3))), replace=False):
                rows[row] *= rng.choice([0.0, 1e-30])
        if truth.size and rng.random() < 0.05:
            truth.reshape(-1)[rng.integers(truth.size)] = rng.choice([np.inf, -np.inf, np.nan])
        return truth

    def arm(self, truth, dtype):
        """An arm of `truth` in `dtype`: exact in float64, or off it relatively and at times absolutely
        too (so a row of zero has a residue), at times exactly zero on a row or not finite at a value."""
        rng = self.rng
        if dtype is np.float64 and rng.random() < 0.3:
            return self.laid_out(truth.copy())
        arm = truth * (1 + 2.0 ** -20 * rng.standard_normal(truth.shape))
        if rng.random() < 0.5:
            largest = np.max(np.abs(truth), initial=0.0, where=np.isfinite(truth))
            arm = arm + 1e-9 * largest * rng.standard_normal(truth.shape)
        arm = np.asarray(arm.astype(dtype))
        if arm.ndim >= 2 and arm.size and rng.random() < 0.3:
            arm.reshape(-1, arm.shape[-1])[rng.integers(arm.size // arm.shape[-1])] = 0
        if arm.size and rng.random() < 0.05:
            arm.reshape(-1)[rng.integers(arm.size)] = rng.choice([np.inf, np.nan])
        return self.laid_out(arm)

    def draw(self) -> tuple[dict, dict, dict, dict]:
        """One draw's Dew, reference, truth and the reference's truth: one in five references recorded,
        and three in ten held to truths of their own, each the same object, the same values in another
        array, or another draw's."""
        rng = self.rng
        pool = [self.shape() for _ in range(rng.integers(2, 8))]
        names = [f"leaf{index:02d}" for index in range(rng.integers(4, 24))]
        shapes = {name: pool[rng.integers(len(pool))] for name in names}
        dtypes = {name: (np.float16, jnp.bfloat16, np.float32, np.float64)[rng.integers(4)] for name in names}
        truth = {name: self.truth(shapes[name]) for name in names}
        truth = {name: self.laid_out(value) if rng.random() < 0.1 else value for name, value in truth.items()}
        their_truth = truth
        if rng.random() < 0.3:
            their_truth = {name: [truth[name], self.laid_out(np.array(truth[name])),
                                  self.truth(shapes[name])][rng.integers(3)] for name in names}
        dew = {name: self.arm(truth[name], dtypes[name]) for name in names}
        if dtypes[names[0]] is not np.float64 and rng.random() < 0.5:
            dew[names[0]] = jnp.asarray(dew[names[0]])
        reference = {name: self.arm(their_truth[name], dtypes[name]) for name in names}
        if rng.random() < 0.2:
            import dataclasses

            from reference_error import leaf_summary

            reference = {name: dataclasses.replace(leaf_summary(reference[name], their_truth[name]),
                                                   zero_sizes=None, worst_log=None) for name in names}
        return dew, reference, truth, their_truth


def test_a_draws_evidence_read_at_once_is_each_leafs_read_alone_bit_for_bit():
    """A draw's truths' rows read at once (`_LaidTruths`) are `_TruthRows`' of each leaf alone (each row's
    own RMS, its size, the leaf's finiteness), its summaries (`_summaries`) `leaf_summary`'s, and
    `_evidence`'s rows the rows it paired from those (`_evidence_leaf_by_leaf`), bit for bit, over 160
    random draws (`_RandomLeaves`): scalars, vectors, empty leaves and rows of widths 1 to 9,000, one
    shape repeated (a run laid as one matrix), in float16, bfloat16, float32 and float64, C-ordered,
    F-ordered and strided, with rows of float64 zero, an arm exactly zero on a row, values not finite or
    near float64's range, a recorded reference, and a reference held to truths of its own."""
    from reference_error import _evidence, _LaidTruths, _summaries, _TruthRows, leaf_summary

    def outcome(fn, *arguments):
        try:
            return _bits(fn(*arguments))
        except AssertionError as error:
            return "raises", str(error)

    random, read, leaves = _RandomLeaves(1848), 0, 0
    with np.errstate(all="ignore"):
        for draw in range(160):
            trees = random.draw()
            assert outcome(_evidence, *trees) == outcome(_evidence_leaf_by_leaf, *trees), draw
            dew, _, truth, their_truth = trees
            for values in (list(truth.values()), list(their_truth.values())):
                held = _LaidTruths.of(values)
                ends = np.cumsum([0] + [size // width if width else 0
                                        for leaves, size, width in held.runs for _ in range(leaves)])
                for index, at in enumerate(held.laid):
                    alone = _TruthRows.of(np.asarray(values[at], np.float64))
                    rows = slice(ends[index], ends[index + 1])
                    assert _bits([held.own[rows], held.sizes[rows], bool(held.finite[index])]) == _bits(
                        [alone.own, alone.sizes, alone.finite]), (draw, at)
            arms, truths = list(jax.device_get(dew).values()), list(truth.values())
            arms[::3] = [None] * len(arms[::3])
            held = _LaidTruths.of(truths)
            for name, arm, true, summary in zip(dew, arms, truths, _summaries(arms, held), strict=True):
                assert _bits(summary) == _bits(None if arm is None else leaf_summary(arm, true)), (draw, name)
            leaves += len(arms)
            read += sum(arms[at] is not None and np.asarray(arms[at], np.float64).flags.c_contiguous
                        for at in held.laid)
    # Most leaves are read at once, not alone.
    assert read > leaves / 3, (read, leaves)
