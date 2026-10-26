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

    def dew(x, mine, theirs):
        return {"y": x + mine}

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

    def dew(x, mine, theirs):
        return {"y": x + mine}

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


def test_a_window_loops_roundings_are_one_windows():
    """A recorded truth's rounding count (`indexed_recording`) traces a window loop (`over_windows`) on
    its first window, as one draw's program: each window is a draw of its own, and a loop that stacks
    its windows on the host cannot be traced whole."""
    from reference_error import _one_draws_program, chain_roundings, over_windows, windows

    def product(x, w):
        return {"y": jnp.dot(x, w, precision="highest")}

    program, arguments = _one_draws_program(over_windows(product), windows(_draw, 4)(0))
    assert program is product
    assert chain_roundings(jax.make_jaxpr(program)(*arguments)) == chain_roundings(
        jax.make_jaxpr(product)(*_draw(0)))
    assert _one_draws_program(product, _draw(0))[0] is product


def test_a_recorded_reference_with_pilots_sizes_paired_on_them_alone():
    """A recorded reference sizes paired only on pilot draws its recording holds (`recorded_pilots`),
    from SIZING_FROM on, never past them; one without pilots is refused, and so is a live reference
    marked with pilots."""
    from reference_error import SIZING_FROM, recorded_pilots

    pilots = range(SIZING_FROM, SIZING_FROM + 32)
    theirs = {seed: _product(*_draw(seed)) for seed in pilots}
    mine = {seed: _reordered(*_draw(seed)) for seed in pilots}
    truths = {seed: {"y": np.asarray(_draw(seed)[0], np.float64) @ np.asarray(_draw(seed)[1], np.float64)}
              for seed in pilots}

    def reference(seed):
        return theirs[int(seed)]

    def dew(seed):
        return mine[int(seed)]

    def truth(seed):
        return truths[int(seed)]

    def draw(seed):
        return (np.int64(seed),)

    reference.recorded = True
    with pytest.raises(ValueError):
        seeds_for(reference, draw, "no pilots", dew_fn=dew, truth_fn=truth)
    assert seeds_for(recorded_pilots(reference, 32), draw, "pilots", dew_fn=dew, truth_fn=truth) <= 32
    with pytest.raises(ValueError):
        recorded_pilots(_product, 32)


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


def test_a_recording_declares_the_count_its_recorded_reference_resolves(tmp_path):
    """tools/recorded_references.py records a reference over the seeds its own recording resolves, and
    their redraw; check 1 then judges Dew against the recording by its index."""
    from tool_support import load

    recorder = load("recorded_references")

    def draw(seed):
        return {"x": np.random.default_rng(100 + seed).normal(size=(64,)).astype(np.float32)}

    def evaluate(inputs):
        x = inputs["x"]
        return {"y": x * x - np.float32(1)}, {"y": x.astype(np.float64) ** 2 - 1}

    def squared(inputs):
        return {"y": inputs["x"] * inputs["x"] - 1}

    recorded = recorder.record_case("site", draw, evaluate, squared)
    seeds = int(recorded["site.seeds"])
    assert seeds >= 8 and len(recorded["site.input.x"]) == 5 * seeds
    path = tmp_path / "recorded.npz"
    np.savez(path, **recorded)
    assert declared_seeds(path) == {"site": seeds}
    with np.load(path) as stored:
        draw_recorded, reference, truth = indexed_recording(dict(stored), "site", seeds)
        assert_as_exact(jax.jit(lambda index, inputs: squared(inputs)), reference, draw_recorded, "recorded",
                        seeds=seeds, truth_fn=truth)


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

    def draw(seed):
        return {"x": np.random.default_rng(200 + seed).normal(size=(64,)).astype(np.float32)}

    def evaluate(inputs):
        x = inputs["x"]
        return {"y": x * x - np.float32(1)}, {"y": x.astype(np.float64) ** 2 - 1}

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

    def draw(seed):
        return {"x": np.random.default_rng(300 + seed).normal(size=(64,)).astype(np.float32)}

    def evaluate(inputs):
        x = inputs["x"]
        return {"y": x * x - np.float32(1)}, {"y": x.astype(np.float64) ** 2 - 1}

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

    def two(x, mine, theirs):
        return {"moved": dew(x, mine, theirs)["y"], "kept": reference(x, mine, theirs)["y"]}

    def two_original(x, mine, theirs):
        return {"moved": reference(x, mine, theirs)["y"], "kept": reference(x, mine, theirs)["y"]}

    def two_truth(x, mine, theirs):
        return {"moved": x, "kept": x}

    assert_moves(two_original, two, errors, "one leaf", seeds=8, truth_fn=two_truth)

    def infinite(x, mine, theirs):
        return {"y": x + jnp.inf}

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
