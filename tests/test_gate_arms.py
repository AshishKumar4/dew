"""Check 1's arms beside Dew's and its reference (`assert_as_exact`'s `library_fn` and `also`): each is held
as the reference is."""

import contextlib
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from arm_forms import without_ynn
from contract_support import normal_drawn
from reference_error import (
    GateFailure,
    Inconclusive,
    UnpairedDraw,
    ahead_arm,
    as_given,
    assert_as_exact,
    seeds_for,
)
from test_reference_error import exact_rows, noised_draw, stacked_arm, stacked_examples
from test_reference_error_written import _cancelling, _combined
from truths_ahead import UNPREPARED, compiled_truths

from dew.nn.kernels import gemm, generation, tuning


def noised(scale: float, noise: str, own: float = 0.0):
    """An arm of `noised_draw`'s: the value plus `scale` times the draw's noise named `noise` ("mine",
    "theirs", "other"), and `own` times a noise of the arm's own, another at each call."""
    calls = itertools.count()

    @as_given
    def run(x, mine, theirs, other):
        drawn = {"mine": mine, "theirs": theirs, "other": other}[noise]
        return {"value": x + scale * drawn + own * np.random.default_rng(next(calls)).standard_normal(4096)}
    return run


@as_given
def noiseless(x, *_):
    """`noised`'s truth: the value itself."""
    return {"value": x}


@pytest.mark.parametrize("arm", ["library_fn", "also"])
def test_check_1_refuses_a_library_arm_and_a_further_reference_of_more_than_their_inputs(arm):
    """A library arm and a further reference enter a site's verdict as its reference does, so each is run
    again on the first draw's inputs and refused where it gives another value (`UnpairedDraw`): one adding a
    noise of its own, a millionth of the draw's, at every call. The same arms without that noise pass. The
    replay asked this of Dew's arm and of the reference alone, and judged Dew against the others' draws no
    seed pairs."""
    given = {"library_fn": {"library_row": 1543, "library_leaves": ["['value']"]}, "also": {}}[arm]

    def beside(own: float) -> dict:
        other = noised(1.0, "other", own)
        return {arm: other if arm == "library_fn" else {"another": other}, **given}

    assert_as_exact(noised(1.0, "mine"), noised(1.0, "theirs"), noised_draw, "arms of their inputs", seeds=16,
                    truth_fn=noiseless, **beside(0.0))
    with pytest.raises(UnpairedDraw):
        assert_as_exact(noised(1.0, "mine"), noised(1.0, "theirs"), noised_draw, "an arm of its own noise",
                        seeds=16, truth_fn=noiseless, **beside(1e-13))


def test_a_library_arm_leaves_the_leaves_it_does_not_cover_judged_by_their_examples():
    """At a site whose leaves stack independent examples (`examples`), a leaf a library arm's allowance does
    not cover is judged as it is without the arm: by its examples' paired ratios where its pooled ratio
    cannot size it. Beside a library arm that covers another leaf, an equally exact Dew passes at the count
    the site sizes, and a Dew three times off on a fifth of each draw's examples of the uncovered leaf is
    refused, decided. Under the allowance the examples' evidence was dropped for every leaf, the uncovered
    among them, and the site, that leaf's pooled ratio unresolved, was left undecided."""
    def beside(stacked, arm: int):
        """`stacked`'s leaf of stacked examples beside an even leaf, each value's own rounding of one size,
        arm `arm`'s own."""
        @as_given
        def run(x, scale, sizes):
            rng = np.random.default_rng([arm, 7, int(np.asarray(x).view(np.uint32)[0, 0])])
            return {**stacked(x, scale, sizes),
                    "even": np.asarray(x, np.float64) + 1e-7 * rng.standard_normal(x.shape)}
        return run

    @as_given
    def truth(x, scale, sizes):
        return {**exact_rows(x, scale, sizes), "even": np.asarray(x, np.float64)}

    reference, library = beside(stacked_arm(1), 1), beside(stacked_arm(1), 2)
    seeds = seeds_for(stacked_arm(1), stacked_examples, "stacked", truth_fn=exact_rows, dew_fn=stacked_arm(0),
                      examples=True)
    allowed = {"library_fn": library, "library_row": 1543, "library_leaves": ["['even']"]}
    assert_as_exact(beside(stacked_arm(0), 0), reference, stacked_examples, "stacked beside a library arm",
                    seeds=seeds, truth_fn=truth, examples=True, **allowed)
    with pytest.raises(GateFailure) as failed:
        assert_as_exact(beside(stacked_arm(0, 3.0, every=5), 0), reference, stacked_examples,
                        "a fifth three times off", seeds=seeds, truth_fn=truth, examples=True, **allowed)
    assert not isinstance(failed.value, Inconclusive)


def test_a_prepared_sites_arms_report_the_routes_their_unprepared_run_reports(tmp_path):
    """A site whose arms its prepare compiled ahead (`truths_ahead.compiled_truths`, a GPU tier part's)
    reports, as its run calls each arm's program, the routes the arm's trace reported where its prepare traced
    it: those the site run unprepared reports. A run calling its executables traced nothing and reported no
    route, so a matrix cell asserting its routes failed on its GPU (row 8374)."""
    draw = normal_drawn(0x8374, (8, 16), (16, 4))

    def dew(rows, matrix):
        return gemm.gemm(rows, matrix, out_dtype=jnp.float32)

    truth = as_given(lambda rows, matrix: np.asarray(rows, np.float64) @ np.asarray(matrix, np.float64))

    def reported(**store) -> set:
        with contextlib.ExitStack() as stack:
            if store:
                stack.enter_context(compiled_truths(tmp_path, stamp="here", **store))
            found = stack.enter_context(generation.routes())
            assert_as_exact(jax.jit(dew), jax.jit(jnp.dot), draw, "a routed product", seeds=16,
                            truth_fn=truth)
        return {(route.choice, route.kernel) for route in found}

    unprepared = reported()
    reported(prepare=True)
    assert unprepared and reported(prepare=False) == unprepared, unprepared


def test_a_prepared_sites_reference_as_written_is_prepared_and_fails_dews_arm_in_its_run(tmp_path):
    """The combination XLA's fusions rewrite (`test_reference_error_written._combined`, row 6909), one
    function as both arms, its site prepared into a store (`truths_ahead.compiled_truths`, a GPU tier part's)
    and run from it: its reference compiled as written (`arm_forms._written_twin`) is prepared beside its arms
    compiled alike (`arm_store.armed`), and the run, calling that program, fails Dew's arm against it as a
    site run unprepared does, nothing of the site run unprepared."""
    label = "a prepared combination XLA's fusions rewrite"
    with compiled_truths(tmp_path, prepare=True, stamp="here"):
        assert_as_exact(_combined, _combined, _cancelling, label, seeds=8)
    with compiled_truths(tmp_path, prepare=False, stamp="here"), pytest.raises(GateFailure):
        assert_as_exact(_combined, _combined, _cancelling, label, seeds=8)
    assert not (tmp_path / UNPREPARED).exists()


def test_an_arm_of_the_cpu_lanes_options_is_built_by_where_its_calls_are_placed():
    """`arm_forms.without_ynn` takes XLA:CPU's options (WITHOUT_YNN) where the context places its calls on a
    CPU, and none where it places them on a GPU, whatever the process's default backend: a GPU tier part's
    prepare, a CPU process placing on its compile-only GPU, builds the arm its run on the GPU builds, which
    the gate's statement names alike (`truths_ahead.arm_role`); keyed on the default backend, the prepare's
    arm held YNNPACK's options and its run's none, and the run found no program (MissingTruth, the review of
    row 8075's conversion)."""
    from arm_forms import without_ynn
    from truths_ahead import arm_role

    def scaled(x):
        return x * 2.0

    plain = arm_role("dew", jax.jit(scaled), "a scaled arm")
    with jax.default_device("gpu"):
        on_a_gpu = arm_role("dew", without_ynn(scaled), "a scaled arm")
    with jax.default_device("cpu"):
        on_a_cpu = arm_role("dew", without_ynn(scaled), "a scaled arm")
    assert on_a_gpu == plain != on_a_cpu


def _summed(x, w):
    """A loss whose gradient an arm takes (the library-function arm witness's)."""
    return jnp.sum(x @ w)


def _scaled_sum(x, w):
    """Another loss (the library-function arm witness's)."""
    return jnp.sum(x @ w) * 2.0


def test_an_arm_of_a_librarys_function_over_the_users_is_named_by_the_users_function():
    """An arm that is a library's function over the user's (`jax.grad`'s, its closure the loss) is named by
    what it holds (`truths_ahead.arm_role`): the containers it reads walked through the user's code and the
    library's closures, not the library's module state (its registries, whose entries no statement holds,
    rows 10064 on: every gradient arm ran unprepared), so one loss's arm is named alike twice and another
    loss's apart."""
    from truths_ahead import arm_role

    one = arm_role("dew", jax.jit(jax.grad(_summed, argnums=(0, 1))), "a gradient")
    assert one is not None
    assert arm_role("dew", jax.jit(jax.grad(_summed, argnums=(0, 1))), "a gradient again") == one
    assert arm_role("dew", jax.jit(jax.grad(_scaled_sum, argnums=(0, 1))), "another gradient") != one


def test_the_arms_arithmetic_is_that_of_where_their_calls_are_placed():
    """Check 1 runs its arms in the arithmetic of where the context places their calls
    (`arm_arithmetic.summed_as_run`): placed on the CPU lane, its fp32 sums (`FP32_SUMS`); placed on a GPU,
    whatever the process's default backend, as a GPU tier part's prepare places them on its compile-only GPU,
    the GPU's own, no preset (row 8580: keyed on the default backend, that prepare traced a cuBLASLt FP8
    arm under the CPU's preset, which compiled it without its default fast accumulation)."""
    from arm_arithmetic import summed_as_run

    from dew.nn.kernels.generation import compile_only_gpus
    from dew.nn.precision import FP32_SUMS

    def preset():
        with summed_as_run():
            return jax.config.jax_default_matmul_precision

    with jax.default_device(jax.devices("cpu")[0]):
        on_a_cpu = preset()
    with jax.default_device(compile_only_gpus("sm90", 1)[0]):
        on_a_gpu = preset()
    assert on_a_cpu == FP32_SUMS and on_a_gpu == jax.config.jax_default_matmul_precision


def test_an_arm_ahead_returns_the_launches_of_the_program_it_ran(tmp_path):
    """`ahead_arm(fn, label, launched=True)` outside a part returns its outputs and the launches the program
    it ran took, at every call: its one lowering's, each the launch a trace of the function takes, the second
    call's too, whose program a plain jit would take from its cache and trace no launch for a collector open
    around it (row 9550: a witness retraced a program of its own to see them); a jit of its own compiler
    options (`without_ynn`'s) lowered as itself, never nested in another; and prepared into a store and run
    from it, a plain function's and a whole jit's, its prepare's lowering's launches once, its run's alike (a
    relowering past a spill, a GPU's, keeps its own: `_arm_program`'s `again`, `_Whole.lower`)."""
    sizes = np.asarray([128, 128], np.int32)
    rng = np.random.default_rng(9550)
    x = rng.standard_normal((256, 128)).astype(np.float32)
    dy = rng.standard_normal((256, 64)).astype(np.float32)

    def gradient(x, dy):
        return gemm.gemm_t(x.astype(jnp.bfloat16), dy.astype(jnp.bfloat16), jnp.asarray(sizes),
                           out_dtype=jnp.float32, precision="highest", interpret_on_cpu=True)

    with generation.assumed("sm90"), generation.chosen(grouped_matmul="triton"):
        made = [ahead_arm(arm, f"a kernel gradient and its launches, {at}", launched=True)(x, dy)
                for at, arm in enumerate((gradient, gradient, without_ynn(gradient)))]
        with tuning.seen() as traced:
            jax.jit(gradient).trace(x, dy)
        # Prepared and run from its store: the launches its prepare's lowering took, its run's alike.
        stored = []
        for prepare in (True, False):
            with compiled_truths(tmp_path, prepare=prepare, stamp="launched"):
                stored.extend(ahead_arm(arm, f"a stored kernel gradient, {at}", launched=True)(x, dy)[1]
                              for at, arm in enumerate((gradient, without_ynn(gradient))))
    expected = [type(op) for op, _ in traced]
    assert expected and all([type(op) for op, _ in launches] == expected
                            for launches in (*(found for _, found in made), *stored))
    np.testing.assert_array_equal(np.asarray(made[0][0]), np.asarray(made[1][0]))


def test_an_arm_whose_launch_takes_a_local_hook_runs_from_its_store(tmp_path):
    """An arm whose launch takes a hook no module holds (a lambda epilogue, which the launch its prepare's
    trace took holds, `tuning.Seen.held`) prepares into its store and runs from it, computing what it does
    outside a part, nothing unprepared: its held record keeps the hook by its name (`truths_ahead.HeldName`),
    which its run never calls (rows 9584, 9600: a held launch refused the store, its arms compiled nothing
    ahead)."""
    from truths_ahead import UNPREPARED

    sizes = np.asarray([128, 128], np.int32)
    rng = np.random.default_rng(9584)
    x = rng.standard_normal((256, 128)).astype(np.float32)
    w = rng.standard_normal((2, 128, 64)).astype(np.float32)

    def doubled(x, w):
        return gemm.gemm(x.astype(jnp.bfloat16), w.astype(jnp.bfloat16), jnp.asarray(sizes),
                         epilogue=lambda acc: acc * 2, out_dtype=jnp.float32, interpret_on_cpu=True)

    with generation.assumed("sm80"), generation.chosen(grouped_matmul="triton"):
        expected = np.asarray(ahead_arm(doubled, "a doubled product")(x, w))
        with compiled_truths(tmp_path, prepare=True, stamp="hooked"):
            ahead_arm(doubled, "a doubled product, stored")(x, w)
        with compiled_truths(tmp_path, prepare=False, stamp="hooked"):
            found = np.asarray(ahead_arm(doubled, "a doubled product, stored")(x, w))
    assert not (tmp_path / UNPREPARED).exists(), (tmp_path / UNPREPARED).read_text()
    np.testing.assert_array_equal(found, expected)


def test_an_arm_ahead_called_again_at_its_program_in_a_prepare_reuses_it_and_refuses_another(tmp_path):
    """An arm with no site called twice in one prepare at one program (two draws of one shape): the second
    call gives the first's zeros and launches, compiling nothing again, and a call of its label at other
    inputs' shapes, or placed on the host, is refused; its run executes each call's own inputs (row 9550's
    witness's cache)."""
    def doubled(x):
        return x * 2

    draws = [np.arange(8, dtype=np.float32) + offset for offset in (0, 100)]
    with compiled_truths(tmp_path, prepare=True, stamp="again") as store:
        arm = ahead_arm(doubled, "one program, two draws")
        prepared = [arm(x) for x in draws]
        with pytest.raises(ValueError):
            arm(np.arange(4, dtype=np.float32))
        # The same function and label placed on the host: another program (head review).
        with pytest.raises(ValueError):
            ahead_arm(doubled, "one program, two draws", host=True)(draws[0])
    assert len(store.arms) == 1 and all(not np.any(made) for made in prepared)
    with compiled_truths(tmp_path, prepare=False, stamp="again"):
        ran = [ahead_arm(doubled, "one program, two draws")(x) for x in draws]
    for x, made in zip(draws, ran, strict=True):
        np.testing.assert_array_equal(np.asarray(made), x * 2)



class _Marked:
    """A test's markers as pytest's item reads them (`get_closest_marker`), the conftest's hook's only read
    of it (the prepare skip witnesses'): each name's mark, its reason given."""

    def __init__(self, *names: str):
        self.names = names

    def get_closest_marker(self, name: str):
        return getattr(pytest.mark, name)(reason="the witness's").mark if name in self.names else None


def test_a_gpu_parts_prepare_skips_what_its_run_skips_by_backend(monkeypatch):
    """A GPU tier part's prepare (`tier_wire.preparing`) skips a test its GPU run skips by its backend
    marker, a TPU's or the CPU lane's, and prepares one marked for a GPU or unmarked (tests/conftest.py's
    `pytest_runtest_setup`): prepared where its run skips it, a test computed there and was filed
    unprepared (row 10169's cells, the TPU's ways of test_algebra_scaled)."""
    import lane_environment
    import conftest
    import tier_wire

    monkeypatch.setenv(tier_wire.PREPARE, "1")
    monkeypatch.delenv(lane_environment.USERS_ARITHMETIC, raising=False)
    for backend in ("tpu", "cpu"):
        with pytest.raises(pytest.skip.Exception):
            conftest.pytest_runtest_setup(_Marked(backend))
    conftest.pytest_runtest_setup(_Marked("gpu"))
    conftest.pytest_runtest_setup(_Marked())


def test_a_gpu_parts_prepare_skips_a_test_declared_to_compile_at_run(monkeypatch):
    """A behavioral GPU test with no check-1 site that declares its programs compiled at its run
    (`compiles_at_run`) is skipped by its part's prepare, which has nothing of it to compile ahead, and run
    outside one (tests/conftest.py's `pytest_runtest_setup`): its first computation in a prepare was filed
    as a check-1 site's prepare missed (rows 8310, 8097)."""
    import lane_environment
    import conftest
    import tier_wire

    monkeypatch.delenv(lane_environment.USERS_ARITHMETIC, raising=False)
    monkeypatch.setenv(tier_wire.PREPARE, "1")
    with pytest.raises(pytest.skip.Exception):
        conftest.pytest_runtest_setup(_Marked("gpu", "compiles_at_run"))
    monkeypatch.delenv(tier_wire.PREPARE)
    conftest.pytest_runtest_setup(_Marked("compiles_at_run"))
