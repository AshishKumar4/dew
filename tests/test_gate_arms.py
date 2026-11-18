"""Check 1's arms beside Dew's and its reference (`assert_as_exact`'s `library_fn` and `also`): each is held
as the reference is."""

import contextlib
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from contract_support import normal_drawn
from reference_error import GateFailure, Inconclusive, UnpairedDraw, as_given, assert_as_exact, seeds_for
from test_reference_error import exact_rows, noised_draw, stacked_arm, stacked_examples
from truths_ahead import compiled_truths

from dew.nn.kernels import gemm, generation


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
