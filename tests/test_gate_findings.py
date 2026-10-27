"""The gate's findings as data (tests/reference_error.py `Finding`, `GateFailure`): each names its leaf,
its rule and its measure, larger worse, and a test the gate fails reports them as the JUnit property
`gate_findings` (tests/conftest.py), what a promotion matches a candidate's failures to its parent's by."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest
from reference_error import (
    GateFailure,
    as_given,
    assert_as_exact,
    assert_as_exact_as_the_reference,
    row_errors,
)


def test_the_one_draw_rule_measures_its_worst_zero_row_and_names_a_value_not_finite():
    """A row five times the reference's error (over ROW_FACTOR, under the pooled factor), and two rows of
    float64 zero that Dew misses by 5 and by 100 times the reference's own: the zero rows' measure is
    the worst row's log ratio, not the first's. A value that is not finite is that rule's finding,
    measured 1, not a ratio NaN would leave unorderable."""
    truth = np.concatenate([np.linspace(1.0, 2.0, 64).reshape(32, 2), np.zeros((2, 2))])
    reference = truth + np.concatenate([np.full((32, 2), 1e-6), np.full((2, 2), 1e-9)])
    dew = reference.copy()
    dew[0] = truth[0] + 5e-6
    dew[32], dew[33] = 5e-9, 1e-7
    with pytest.raises(GateFailure) as raised:
        assert_as_exact_as_the_reference(dew, reference, truth, "rows")
    rules = {finding.rule: finding.measure for finding in raised.value.findings}
    _, mine = row_errors(dew, truth)
    _, theirs = row_errors(reference, truth)
    first, worst = (math.log(mine[row] / theirs[row]) for row in (32, 33))
    assert "row" in rules and rules["zero rows"] > (first + worst) / 2
    dew[1, 0] = np.nan
    with pytest.raises(GateFailure) as raised:
        assert_as_exact_as_the_reference(dew, reference, truth, "nan")
    [finding] = raised.value.findings
    assert (finding.rule, finding.measure) == ("not finite", 1)


def test_the_one_draw_rule_reports_every_rule_a_result_fails_at_once():
    """A result over the pooled factor and, at one row, over the row factor reports both findings, so a
    change that eases the pooled ratio while breaking a row is no smaller failure than its parent's."""
    truth = np.ones((32, 2))
    reference = truth + 2.0 ** -10
    dew = reference.copy()
    dew[0] = truth[0] + 12 * 2.0 ** -10
    with pytest.raises(GateFailure) as raised:
        assert_as_exact_as_the_reference(dew, reference, truth, "both")
    assert {finding.rule for finding in raised.value.findings} == {"pooled", "row"}


def test_an_exact_reference_measures_dew_by_its_own_distance_and_two_exact_arms_measure_nothing():
    """Where the reference equals float64, a Dew that does not is measured by its own distance from it,
    a farther one larger, so a worse candidate is no match for its parent; where both equal it the site
    measures nothing, a finding of its own rule."""
    truth = np.linspace(1.0, 2.0, 64)
    with pytest.raises(GateFailure) as raised:
        assert_as_exact_as_the_reference(truth.copy(), truth.copy(), truth, "exact")
    assert [finding.rule for finding in raised.value.findings] == ["no measure"]
    measures = []
    for off in (1e-9, 1e-6):
        with pytest.raises(GateFailure) as raised:
            assert_as_exact_as_the_reference(truth + off, truth.copy(), truth, "exact")
        [finding] = raised.value.findings
        assert finding.rule == "exact reference"
        measures.append(finding.measure)
    assert measures[0] < measures[1]


def test_a_zero_reference_row_measures_dews_error_on_it_finitely_and_a_worse_one_larger():
    """Rows of float64 zero the reference keeps exactly, where Dew errs by 8 and then by 16: each is a
    finding of finite measure (a ratio over a reference error of zero, divided, would be infinite both
    times and order nothing), the larger error the larger measure."""
    truth = np.array([[1.0, 1.0], [0.0, 0.0]])
    reference = np.array([[17.0, 17.0], [0.0, 0.0]])
    measures = []
    for off in (8.0, 16.0):
        with pytest.raises(GateFailure) as raised:
            assert_as_exact_as_the_reference(np.array([[17.0, 17.0], [off, off]]), reference, truth, "zero")
        [zero] = [finding for finding in raised.value.findings if finding.rule == "zero rows"]
        assert math.isfinite(zero.measure)
        measures.append(zero.measure)
    assert measures[0] < measures[1]


def test_a_subnormal_row_error_and_an_overflowing_sum_still_measure_a_worse_candidate_larger():
    """A row error of 2^-1030 and one of 2^-1029 over a reference row error of zero (the row normalized by
    a size of 2^500) measure apart, the larger larger, where flooring them at the least normal would
    measure both alike; and a check-1 site whose per-seed squared errors are finite but sum past
    float64's range measures its pooled ratio finitely, a farther Dew larger, and so does one whose
    reference is exactly float64 (its measure Dew's own distance)."""
    truth = np.array([[2.0 ** 500, 0.0], [0.0, 0.0]])
    reference = np.array([[2.0 ** 500, 0.0], [2.0 ** -520, 2.0 ** -520]])
    measures = []
    for off in (2.0 ** -530, 2.0 ** -529):
        dew = reference.copy()
        dew[0, 1] = off
        with pytest.raises(GateFailure) as raised:
            assert_as_exact_as_the_reference(dew, reference, truth, "subnormal")
        [row] = [finding for finding in raised.value.findings if finding.rule == "row"]
        measures.append(row.measure)
    assert measures[0] < measures[1]

    def draw(seed):
        return (np.zeros(1),)

    def reference_fn(x):
        return {"y": x + 1.0}

    def truth_fn(x):
        return {"y": x}

    exact = []
    for far in (1e154, 1.2e154):
        with pytest.raises(GateFailure) as raised:
            # NumPy arms, as given: a Dew arm compiled would compute in float32 and overflow.
            assert_as_exact(as_given(lambda x, far=far: {"y": x + far}), as_given(lambda x: {"y": x}), draw,
                            "exact far", seeds=8, truth_fn=truth_fn)
        [finding] = [finding for finding in raised.value.findings if finding.rule == "exact reference"]
        assert math.isfinite(finding.measure)
        exact.append(finding.measure)
    assert exact[0] < exact[1]

    measures = []
    for far in (1e154, 1.2e154):
        with pytest.raises(GateFailure) as raised:
            assert_as_exact(as_given(lambda x, far=far: {"y": x + far}), as_given(reference_fn), draw, "far",
                            seeds=8, truth_fn=truth_fn)
        [pooled] = [finding for finding in raised.value.findings
                    if finding.rule in ("pooled", "inconclusive")]
        assert math.isfinite(pooled.measure)
        measures.append(pooled.measure)
    assert measures[0] < measures[1]


def three_times(label):
    """The one-draw rule on Dew three times the reference's error from float64, which it fails."""
    truth = np.linspace(1.0, 2.0, 64)
    reference = truth + 1e-3 * np.cos(np.arange(64))
    assert_as_exact_as_the_reference(truth + 3 * (reference - truth), reference, truth, label)


def test_a_test_the_gate_fails_reports_its_findings_as_a_junit_property(tmp_path):
    """A test failing the one-draw rule at three times the reference's error (`three_times`), in its call
    and, apart, in its teardown: each reports its finding (leaf, rule, measure, phase) as the JUnit
    property `gate_findings`, the measure the finding's own."""
    site = tmp_path / "test_site.py"
    site.write_text(
        "import pytest\n"
        "from test_gate_findings import three_times\n\n\n"
        "@pytest.fixture\n"
        "def torn():\n"
        "    yield\n"
        "    three_times('teardown')\n\n\n"
        "def test_call():\n"
        "    three_times('call')\n\n\n"
        "def test_teardown(torn):\n"
        "    pass\n")
    report = tmp_path / "report.xml"
    tests = Path(__file__).resolve().parent
    path = os.pathsep.join([str(tests), os.environ.get("PYTHONPATH", "")])
    done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "conftest", "-p", "no:cacheprovider",
                           "-o", "junit_family=xunit1", f"--junitxml={report}", str(site)], cwd=tmp_path,
                          env={**os.environ, "PYTHONPATH": path})
    assert done.returncode == 1
    found = {}
    for case in ElementTree.parse(report).iter("testcase"):
        for node in case.iter("property"):
            if node.get("name") == "gate_findings":
                found[case.get("name")] = json.loads(node.get("value"))
    assert [(item["leaf"], item["rule"], item["phase"]) for item in found["test_call"]] == [
        ("call", "pooled", "call")]
    assert [(item["leaf"], item["rule"], item["phase"]) for item in found["test_teardown"]] == [
        ("teardown", "pooled", "teardown")]
    with pytest.raises(GateFailure) as raised:
        three_times("call")
    assert found["test_call"][0]["measure"] == raised.value.findings[0].measure


def test_a_measure_is_finite_where_the_evidence_is_finite_even_past_float64s_largest_quotient():
    """Row 1628: a sized row's error over its size and the zero rows' pooled squares overflowed from
    finite evidence (an error of 1e154 over a row of 1e-160), and the promotion matcher refuses a
    non-finite measure. Each now comes from logs or a scaled sum: the measure is finite, and the row still
    fails."""
    from reference_error import pooled_zero_rows, rows_over

    truth = np.asarray([[1e-160, 1e-160], [1.0, 1.0]])
    reference = np.asarray([[0.0, 0.0], [1.0, 1.0]])
    dew = np.asarray([[1e154, 1e154], [1.0, 1.0]])
    found = rows_over(dew, reference, truth)
    assert found and all(np.isfinite(measure) for _, measure, _ in found), found
    rows = np.asarray([0]), np.asarray([1e155]), np.asarray([1e-10])
    pooled = pooled_zero_rows([rows, rows])
    assert pooled and all(np.isfinite(value) for row in pooled for value in row[1:]), pooled


def test_a_worse_row_past_float64s_largest_quotient_measures_larger():
    """A row's error over its size is judged and measured by its log (`_log_over`), so two quotients
    past float64's largest, 1e310 and 1e311, keep their order: a candidate ten times worse is not
    matched to its parent's failure."""
    from reference_error import rows_over

    truth = np.asarray([[1e-160, 1e-160], [1e-148, 1e-148]])
    reference = truth + np.asarray([[1e140, 1e140], [1e152, 1e152]])

    def measure(error):
        dew = truth + np.asarray([[error, error], [1e152, 1e152]])
        return {rule: value for rule, value, _ in rows_over(dew, reference, truth)}["row"]

    assert np.isfinite(measure(1e150)) and measure(1e151) > measure(1e150) + 2


def test_a_reference_exactly_zero_on_a_residue_row_does_not_measure_it():
    """A row float64 sizes zero whose truth is float64's residue (a sum that cancels, 1.6e-40 beside a leaf
    of 1) is a row of float64 zero where the reference is exactly zero on it: its error is the row's
    size by the same arithmetic, so Dew's ordinary rounding there (1e-15) is held by the zero rows' rule
    and not read over the residue's size (ledger 1831); a reference that carries the row still
    measures it."""
    from reference_error import _paired, leaf_summary

    # Residue rows whose squares fall below float64's normal range, where a size and an error taken by
    # two arithmetics part by more than a rounding.
    residue = [[1.6e-40, -1.6e-40, 1.6e-40], [3e-160, -2e-160, 1e-160], [7e-161, 5e-161, -3e-161],
               [1.1e-159, -9e-160, 4e-161], [2.3e-160, 2.9e-160, -1.7e-160]]
    truth = np.asarray([[1.0, -1.0, 0.5], *residue])
    reference = np.where(np.arange(len(truth))[:, None] == 0, truth, 0.0)
    dew = np.where(np.arange(len(truth))[:, None] == 0, truth, 1e-15)
    mine = leaf_summary(dew, truth)
    _, _, worst, _, (rows, _, _), _ = _paired("leaf", mine, leaf_summary(reference, truth))
    assert rows.tolist() == list(range(1, len(truth))) and worst < 0
    close = np.where(np.arange(len(truth))[:, None] == 1, [1.6e-40, -1.6e-40, 1.5e-40], reference)
    _, _, worst, _, (rows, _, _), _ = _paired("leaf", mine, leaf_summary(close, truth))
    assert 1 not in rows.tolist() and worst > 50


def _sinkhorn_row(x):
    """A leaf whose first row is ordinary and whose second is eps times a normalization's residual,
    the column sum of x over its own sum less one: zero in exact arithmetic, a rounding in any."""
    import jax.numpy as jnp

    p = x / jnp.sum(x)
    residual = 1e-6 * (jnp.sum(p) - 1)
    return jnp.stack([x / 3, jnp.full_like(x, residual) + 1e-30])


def test_the_truths_spread_is_its_own_roundings_moved():
    """`_truth_spread` moves every rounding the truth makes by float64's unit roundoff (Monte Carlo
    arithmetic): a normalization's residual, which float64 holds only to its own rounding, moves by its
    whole size, and a value computed exactly ((x - 1) + t at x = 1, t = 2^-100) moves by a unit roundoff
    of itself, however its inputs would move it (ledger 1856's review)."""
    import jax
    import jax.numpy as jnp
    from reference_error import ROW_FACTOR, _truth_spread

    with jax.enable_x64(new_val=True):
        x = jnp.asarray(np.random.default_rng(0).uniform(0.5, 1.5, 64))
        truth = _sinkhorn_row(x)
        spread = _truth_spread(_sinkhorn_row, (x,), truth)
        assert np.max(spread[1]) > np.max(np.abs(truth[1])) / ROW_FACTOR ** 2

        def exact(x, t):
            return jnp.stack([x / 3, (x - 1) + t])
        ones, tiny = jnp.ones(3), jnp.full(3, 2.0 ** -100)
        spread = _truth_spread(exact, (ones, tiny), exact(ones, tiny))
        assert np.max(spread[1]) <= 2.0 ** -100 * float(np.finfo(np.float64).eps)


def test_an_exact_tiny_row_keeps_its_bound_where_rows_are_resolved():
    """With `resolve_rows`, a tiny value float64 computes exactly ((x - 1) + t at x = 1, t = 2^-100)
    stays measured: a reference half the row off and a Dew four times it off on one seed of eight fails
    the worst-row rule (8 times the reference's, past ROW_FACTOR), as it does without the option; a
    Dew 1.6 times the reference's error everywhere on a product fails too."""
    import jax
    import jax.numpy as jnp
    from reference_error import GateFailure, as_given, assert_as_exact

    def draw(seed):
        return np.ones(3, np.float32), np.full(3, 2.0 ** -100, np.float32), np.int32(seed)

    def truth(x, t, seed):
        return {"y": jnp.stack([x / 3, (x - 1) + t])}

    def arm(*, far):
        @as_given
        def run(x, t, seed):
            off = np.float32(4.0 if far and int(seed) % 8 == 0 else 0.5)
            first = (np.asarray(x, np.float32) / np.float32(3))
            return {"y": np.stack([first, np.asarray(t, np.float32) * (1 + off)])}
        return run

    for resolve in (False, True):
        with pytest.raises(GateFailure):
            assert_as_exact(arm(far=True), arm(far=False), draw, "an exact tiny row", seeds=8, truth_fn=truth,
                            resolve_rows=resolve)

    def product_draw(seed):
        rng = np.random.default_rng(seed)
        return tuple(rng.standard_normal(shape).astype(np.float32) for shape in ((16, 32), (32, 8)))

    def product(x, w):
        return {"y": jnp.dot(x, w, precision="highest")}

    @as_given
    def farther(x, w):
        exact = np.asarray(x, np.float64) @ np.asarray(w, np.float64)
        return {"y": exact + 1.6 * (np.asarray(jax.jit(product)(x, w)["y"], np.float64) - exact)}

    assert_as_exact(jax.jit(product), jax.jit(product), product_draw, "resolved", seeds=8, truth_fn=product,
                    resolve_rows=True)
    with pytest.raises(GateFailure):
        assert_as_exact(farther, jax.jit(product), product_draw, "resolved", seeds=8, truth_fn=product,
                        resolve_rows=True)
    with pytest.raises(ValueError):
        assert_as_exact(farther, jax.jit(product), product_draw, "as given", seeds=8,
                        truth_fn=as_given(product), resolve_rows=True)

