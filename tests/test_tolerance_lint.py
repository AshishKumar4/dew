"""SLOP020: every accuracy comparison in tests/ and tools/ goes through the gate's one assertion, and
the inventory of files that still compare floats within a tolerance only shrinks."""

import json
import subprocess
import textwrap

import pytest
from tool_support import load


@pytest.fixture(scope="module")
def lint():
    return load("lint_slop")


COMPARISONS = {
    "numpy's tolerance": ("import numpy as np\nnp.testing.assert_allclose(a, b, rtol=1e-5)", 1),
    "an alias of it": ("from numpy.testing import assert_allclose as check\ncheck(a, b)", 1),
    "pytest's approx": ("import pytest\nassert a == pytest.approx(b)", 1),
    "math's isclose by its default": ("import math as m\nassert m.isclose(a, b)", 1),
    "torch's assert_close": ("import torch\ntorch.testing.assert_close(a, b)", 1),
    "unittest's assertAlmostEqual": ("self.assertAlmostEqual(a, b)", 1),
    "dew.accuracy's within, qualified": ("import dew.accuracy\nassert dew.accuracy.within(e, b)", 1),
    "a wrapper given a tolerance": ("close(a, b, atol=1e-6)", 1),
    "a hand-written tolerance": ("import numpy as np\nassert np.max(np.abs(a - b)) <= 1e-6", 1),
    "an error bound to a name first": ("import numpy as np\ngap = np.abs(a - b)\nworst = gap.max()\n"
                                       "assert worst < bound", 1),
    "an error a helper of the file returns": ("def gaps(a, b):\n    return abs(a - b) / abs(b)\n"
                                              "def test():\n    assert gaps(x, y) < 1e-6", 1),
    "a helper that may return an error": ("def gap(a, b):\n    if a == b:\n        return 0.0\n"
                                          "    return abs(a - b)\n"
                                          "def test():\n    assert gap(actual, expected) < 1e-6", 1),
    "an ordering assertion of an error": ("import numpy as np\n"
                                          "np.testing.assert_array_less(abs(actual - expected), bound)", 1),
    "a residual held near zero": ("assert abs(score) < 1e-3", 1),
    "a magnitude given by keyword": ("import torch\n"
                                     "assert torch.abs(input=actual - expected).max() < 1e-6", 1),
    "a distance against a bound": ("from reference_error import distance\n"
                                   "assert distance(a, truth) <= 2 * distance(b, truth)", 1),
    "a difference against a float bound": ("assert actual - expected < 1e-6", 1),
    "a float within an additive slack": ("assert honest <= expected(rule) + 1e-6", 1),
    "a named tolerance": ("TOLERANCE = 1e-5", 1),
    "a chain whose second relation is a tolerance": (
        "assert 0 <= abs(actual - truth) <= 2 * abs(reference - truth)", 1),
    "a tolerance call inside a call that reads a clock": (
        "import numpy as np, time\nprint(np.testing.assert_allclose(actual, expected), time.monotonic())", 1),
    "a tolerance call whose message names a duration": (
        "import numpy as np\nnp.testing.assert_allclose(actual, expected, err_msg=duration_message)", 1),
    "a model output named like a time": ("import numpy as np\nms = model.apply(params, inputs)\n"
                                         "np.testing.assert_allclose(ms, expected)", 1),
    "a time's name rebound to a model output": ("import numpy as np, time\nvalue = time.monotonic()\n"
                                                "value = model.apply(params, inputs)\n"
                                                "np.testing.assert_allclose(value, expected)", 1),
    "a model's output computed from a duration": ("import numpy as np\n"
                                                  "low = pipe(tone(220, data.audio_seconds))\n"
                                                  "assert np.abs(low - high).max() > 1e-3", 1),
    "a packed tolerance": ("import numpy as np\n"
                           "np.testing.assert_allclose(a, b, rtol=0, **{'atol': 1e-3})", 1),
    "a count's error, which is still a magnitude": ("assert abs(len(rows) - 68) <= 1", 1),
    "a difference and its bound bound to names first": ("gap = actual - expected\nlimit = 1e-6\n"
                                                         "assert gap < limit", 1),
    "a mean squared difference": ("import numpy as np\nassert np.mean((actual - expected) ** 2) < 1e-6", 1),
    "a clock compared by keyword operands": (
        "import numpy as np, time\n"
        "np.testing.assert_allclose(actual=time.monotonic(), desired=time.monotonic(), rtol=0, atol=0.1)", 0),
    "approx's rel is no compared value": ("import pytest\n"
                                          "assert loss == pytest.approx(expected, wall_seconds)", 1),
    "a clock sample written into a mapping of losses": (
        "import numpy as np, time\ndef check(metrics):\n    metrics['clock_sample'] = time.monotonic()\n"
        "    np.testing.assert_allclose(metrics['loss'], expected)", 1),
    "a model output under a time's key": ("import numpy as np\nrecord = {'ms': model.apply(params, inputs)}\n"
                                          "np.testing.assert_allclose(record['ms'], expected)", 1),
    "a mapping of times a loss is added to": (
        "import numpy as np, time\ntimes = {'a_seconds': time.monotonic()}\ntimes.update(loss=model_loss)\n"
        "np.testing.assert_allclose(times['loss'], expected)", 1),
    "a model output captured by a pattern": ("import numpy as np\nmatch (model.apply(params, inputs),):\n"
                                             "    case (ms,):\n"
                                             "        np.testing.assert_allclose(ms, expected)", 1),
    "a clock's default that is a model's output": (
        "import numpy as np, time\n"
        "value = max((time.monotonic() for _ in []), default=model_value)\n"
        "np.testing.assert_allclose(value, expected)", 1),
    "a time against a written expectation": (
        "import numpy as np\nnp.testing.assert_allclose(elapsed_seconds, 3.0, rtol=0.25)", 0),
    "exact equality": ("import numpy as np\nnp.testing.assert_array_equal(a, b)\nassert a == b", 0),
    "a tolerance call told to be exact": ("import torch\n"
                                          "torch.testing.assert_close(a, b, atol=0, rtol=0)", 0),
    "pytest's approx given only a zero abs": ("import pytest\nassert a == pytest.approx(b, abs=0)", 0),
    "both of numpy's tolerances zero": ("import numpy as np\nnp.testing.assert_allclose(a, b, rtol=0)", 0),
    "a count bounded on both sides": ("assert 67 <= len(rows) <= 69", 0),
    "a difference of counts within whole bounds": ("assert -1 <= left - right <= 1", 0),
    "index arithmetic": ("while len(ended) < bridge.MAX_CELLS - 1:\n    pass\n"
                         "hung = output[stack:] if 0 <= stack < len(output) - 4000 else ''", 0),
    "a difference against a named tolerance": ("assert actual - expected < tolerance", 1),
    "a magnitude that is nonzero, or zero": ("import numpy as np\nassert np.abs(a - b).max() > 0\n"
                                             "assert np.linalg.norm(gradient) <= 0.0", 0),
    "a loss that falls": ("assert loss - first_loss < 0", 0),
    "a time against its budget": ("assert abs(elapsed - expected_seconds) < 0.5", 0),
    "a time compared by approx": ("import pytest\nassert wall_seconds == pytest.approx(3.0, rel=0.25)", 0),
    "a time read by its key, then compared": ("import pytest\n"
                                              "short, long = (row[\"seconds\"] for row in rows)\n"
                                              "assert long / short == pytest.approx(3.0, rel=0.25)", 0),
    "times read through a mapping": ("import pytest\n"
                                     "rows = {row[\"name\"]: row[\"seconds\"] for row in report}\n"
                                     "short, long = rows[\"a\"], rows[\"b\"]\n"
                                     "assert long / short == pytest.approx(3.0, rel=0.25)", 0),
    "a profiler's milliseconds": ("import pytest\n"
                                  "assert timeline[\"device_busy_ms_per_step\"] == pytest.approx(30)", 0),
    "a method that happens to be named within or approx": ("assert method.within(family)\n"
                                                           "assert rows.approx(3)", 0),
    "an ordering of two values, no error taken": ("assert loss < first_loss", 0),
}


def _module(lint, tmp_path, relative, body, inventory=None):
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body))
    if inventory is not None:
        (tmp_path / "tools").mkdir(exist_ok=True)
        (tmp_path / lint.INVENTORY).write_text(json.dumps(inventory))
    [module] = list(lint.collect([relative], tmp_path, "dew"))
    return module


@pytest.mark.parametrize("case", list(COMPARISONS))
def test_a_float_compared_within_a_tolerance_outside_the_gate_is_reported(lint, case, tmp_path):
    body, count = COMPARISONS[case]
    module = _module(lint, tmp_path, "tests/test_case.py", body)
    assert len(list(lint.tolerances(module))) == count


def test_the_gate_compares_floats_and_scripts_outside_tests_and_tools_are_not_read(lint, tmp_path):
    body = "import numpy as np\nnp.testing.assert_allclose(a, b, rtol=1e-5)\n"
    assert not list(lint.tolerances(_module(lint, tmp_path, "tests/reference_error.py", body)))
    assert not list(lint.tolerances(_module(lint, tmp_path, "examples/run.py", body)))
    assert list(lint.tolerances(_module(lint, tmp_path, "tools/report.py", body)))


def test_the_inventory_holds_a_file_at_its_count_until_its_date_and_only_shrinks(lint, tmp_path):
    """A file the inventory holds at its count and before its dates passes; one that grew reports
    every comparison; one that shrank reports once, until the count comes down; past its own date or
    the inventory's, every comparison is reported, whatever its count. Each reading sees the
    inventory as it is then, in one process."""
    body = "import numpy as np, pytest\nnp.testing.assert_allclose(a, b)\nassert a == pytest.approx(b)\n"

    def findings(count, by="2999-01-01", empty_by="2999-01-01"):
        entry = {"tests/test_case.py": {"comparisons": count, "lane": "Kernels", "by": by}}
        inventory = {"empty_by": empty_by, "files": entry}
        return list(lint.tolerances(_module(lint, tmp_path, "tests/test_case.py", body, inventory)))

    assert findings(2) == []
    assert len(findings(1)) == 2
    [lower] = findings(3)
    assert lower.line == 1
    assert len(findings(2, by="2000-01-01")) == 2
    assert len(findings(3, by="2000-01-01")) == 2
    assert len(findings(2, empty_by="2000-01-01")) == 2
    empty = {"empty_by": "2999-01-01", "files": {"tests/test_case.py": {"comparisons": 2, "lane": "Kernels",
                                                                           "by": "2000-01-01"}}}
    [remove] = list(lint.tolerances(_module(lint, tmp_path, "tests/test_case.py", "x = 1\n", empty)))
    assert "remove its entry" in remove.message


def _commit(repo, inventory, message, files=None):
    for name, body in (files or {}).items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(body)
    (repo / "tools").mkdir(exist_ok=True)
    (repo / "tools/tolerance_inventory.json").write_text(json.dumps(inventory))
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", message], cwd=repo,
                   check=True)


def _inventory(counts: dict, by="2026-11-20", empty_by="2026-11-27"):
    return {"empty_by": empty_by,
            "files": {name: {"comparisons": count, "lane": "Kernels", "by": by}
                      for name, count in counts.items()}}


def test_no_commit_loosens_the_inventory_its_parents_kept(lint, tmp_path):
    """A commit may lower a count and carry an entry across a rename; one that raises a count, adds a
    file, or merges back a looser inventory than its branch kept is reported, merges included."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    _commit(repo, _inventory({"tests/test_a.py": 10}), "made", {"tests/test_a.py": "a = 1\n"})
    _commit(repo, _inventory({"tests/test_a.py": 8}), "lowered")
    subprocess.run(["git", "mv", "tests/test_a.py", "tests/test_b.py"], cwd=repo, check=True)
    _commit(repo, _inventory({"tests/test_b.py": 8}), "renamed")
    assert list(lint.ratchet(repo)) == []
    subprocess.run(["git", "checkout", "-q", "-b", "branch"], cwd=repo, check=True)
    _commit(repo, _inventory({"tests/test_b.py": 5}), "branch lowers")
    subprocess.run(["git", "checkout", "-q", "main"], cwd=repo, check=True)
    _commit(repo, _inventory({"tests/test_b.py": 8}), "main touches", {"tests/other.py": "x = 1\n"})
    subprocess.run(["git", "checkout", "-q", "branch"], cwd=repo, check=True)
    # The merge takes main's inventory whole, the branch's lower count undone: TREESAME to main.
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "merge", "-q", "--no-commit", "main"],
                   cwd=repo, check=True)
    _commit(repo, _inventory({"tests/test_b.py": 8}), "merge keeps main's 8")
    assert any("raises tests/test_b.py" in finding.message for finding in lint.ratchet(repo))
    # Removing the inventory, then making it again looser, is reported at the removal.
    subprocess.run(["git", "rm", "-q", "tools/tolerance_inventory.json"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "removed"], cwd=repo,
                   check=True)
    _commit(repo, _inventory({"tests/test_b.py": 20}), "made again")
    assert any("removes the inventory" in finding.message for finding in lint.ratchet(repo))


def test_the_ratchet_holds_a_branch_to_its_base_and_refuses_a_shallow_history(lint, tmp_path):
    """A branch whose inventory is looser than its base's merge-base is reported against that base;
    a shallow clone cannot be held, and says so rather than passing."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    _commit(repo, _inventory({"tests/test_a.py": 4}), "made", {"tests/test_a.py": "a = 1\n"})
    _commit(repo, _inventory({"tests/test_a.py": 3}), "lowered")
    subprocess.run(["git", "checkout", "-q", "-b", "head", "HEAD~1"], cwd=repo, check=True)
    _commit(repo, _inventory({"tests/test_a.py": 4}, by="2026-11-21"), "a later date")
    assert any("against main's" in finding.message for finding in lint.ratchet(repo, "main"))
    # Two readings in one process, a commit between them, see each HEAD as it is.
    subprocess.run(["git", "checkout", "-q", "main"], cwd=repo, check=True)
    assert list(lint.ratchet(repo, "main")) == []
    _commit(repo, _inventory({"tests/test_a.py": 2}), "lowered again")
    assert list(lint.ratchet(repo, "main")) == []
    shallow = tmp_path / "shallow"
    subprocess.run(["git", "clone", "-q", "--depth", "1", f"file://{repo}", str(shallow)], check=True)
    [refused] = list(lint.ratchet(shallow))
    assert "shallow" in refused.message
