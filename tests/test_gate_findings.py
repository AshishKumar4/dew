"""The gate's findings as data (tests/reference_error.py `Finding`, `GateFailure`): each names its leaf,
its rule and its measure, larger worse, and a test the gate fails reports them as the JUnit property
`gate_findings` (tests/conftest.py), what a promotion matches a candidate's failures to its parent's by."""

import json
import os
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest
from reference_error import GateFailure, assert_as_exact_as_the_reference, row_errors


def test_the_one_draw_rule_measures_its_worst_zero_row_and_names_a_value_not_finite():
    """A row five times the reference's error (over ROW_FACTOR, under the pooled factor), and two rows of
    float64 zero that Dew misses by 5 and by 100 times the reference's own: the zero rows' measure is
    the worst row's, not the first's. A value that is not finite is that rule's finding, measured 1,
    not a ratio NaN would leave unorderable."""
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
    first, worst = (float(mine[row] / theirs[row]) for row in (32, 33))
    assert "row" in rules and rules["zero rows"] == worst > first
    dew[1, 0] = np.nan
    with pytest.raises(GateFailure) as raised:
        assert_as_exact_as_the_reference(dew, reference, truth, "nan")
    [finding] = raised.value.findings
    assert (finding.rule, finding.measure) == ("not finite", 1)


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
