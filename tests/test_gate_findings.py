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
