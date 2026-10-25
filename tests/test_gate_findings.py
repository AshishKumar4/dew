"""The gate's findings as data (tests/reference_error.py `Finding`, `GateFailure`): each names its leaf,
its rule and its measure, larger worse, and a test the gate fails reports them as the JUnit property
`gate_findings` (tests/conftest.py), what a promotion matches a candidate's failures to its parent's by."""

import json
import os
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import PARITY, GateFailure, assert_as_exact, assert_as_exact_as_the_reference, seeds_for


def _draw(seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(64, 256)).astype(np.float32), rng.normal(size=(256, 32)).astype(np.float32)


@jax.jit
def _product(x, w):
    return {"y": jnp.dot(x, w, precision="highest")}


@jax.jit
def _lost_term(x, w):
    return {"y": jnp.dot(x[:, :-1], w[:-1], precision="highest")}


def test_a_failing_site_raises_its_findings_as_data():
    """A lost term fails check 1's pooled ratio: its finding names the leaf, the rule and a measure past
    PARITY, and the failure's message is its findings', one a line."""
    with pytest.raises(GateFailure) as raised:
        assert_as_exact(_lost_term, _product, _draw, "lost", seeds=seeds_for(_product, _draw, "product"))
    assert str(raised.value) == "\n".join(map(str, raised.value.findings))
    pooled = [finding for finding in raised.value.findings if finding.rule == "pooled"]
    assert pooled and all(finding.leaf.startswith("lost") and finding.measure > PARITY for finding in pooled)


def test_the_one_draw_rule_measures_its_worst_zero_row_and_names_a_value_not_finite():
    """A row five times the reference's error (over ROW_FACTOR, under the pooled factor), and two rows of
    float64 zero that Dew misses by 5 and by 100 times the reference's own: the zero rows' measure is
    the worst, 100, not the first row's; the message keeps the rule's own wording, the label once, the
    rows' findings on one line. A value that is not finite is that rule's finding, measured 1, not a
    ratio NaN would leave unorderable."""
    truth = np.concatenate([np.linspace(1.0, 2.0, 64).reshape(32, 2), np.zeros((2, 2))])
    reference = truth + np.concatenate([np.full((32, 2), 1e-6), np.full((2, 2), 1e-9)])
    dew = reference.copy()
    dew[0] = truth[0] + 5e-6
    dew[32], dew[33] = 5e-9, 1e-7
    with pytest.raises(GateFailure) as raised:
        assert_as_exact_as_the_reference(dew, reference, truth, "rows")
    rules = {finding.rule: finding.measure for finding in raised.value.findings}
    assert rules["row"] == pytest.approx(5.0) and rules["zero rows"] == pytest.approx(100.0)
    assert str(raised.value).startswith("rows: ") and "\n" not in str(raised.value)
    dew[1, 0] = np.nan
    with pytest.raises(GateFailure) as raised:
        assert_as_exact_as_the_reference(dew, reference, truth, "nan")
    [finding] = raised.value.findings
    assert (finding.rule, finding.measure) == ("not finite", 1)


def test_a_test_the_gate_fails_reports_its_findings_as_a_junit_property(tmp_path):
    """A test failing the one-draw rule at three times the reference's error, in its call and, apart, in
    its teardown: each reports its finding (leaf, rule, measure, phase) as the JUnit property
    `gate_findings`."""
    site = tmp_path / "test_site.py"
    site.write_text(
        "import numpy as np\n"
        "import pytest\n"
        "from reference_error import assert_as_exact_as_the_reference\n\n\n"
        "def three_times(label):\n"
        "    truth = np.linspace(1.0, 2.0, 64)\n"
        "    reference = truth + 1e-3 * np.cos(np.arange(64))\n"
        "    assert_as_exact_as_the_reference(truth + 3 * (reference - truth), reference, truth, label)\n\n\n"
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
    assert found["test_call"][0]["measure"] == pytest.approx(3.0)
