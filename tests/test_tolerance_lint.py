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
    "a model's output computed from a duration": ("import numpy as np\n"
                                                  "low = pipe(tone(220, data.audio_seconds))\n"
                                                  "assert np.abs(low - high).max() > 1e-3", 1),
    "a packed tolerance": ("import numpy as np\n"
                           "np.testing.assert_allclose(a, b, rtol=0, **{'atol': 1e-3})", 1),
    "a count's distance, made of explicit whole numbers": ("assert abs(len(rows) - 68) <= 1\n"
                                                          "assert abs(int(a.mean()) - int(b.mean())) > 1", 0),
    "a count's distance of unknown provenance": ("assert abs(count - 68) <= 1", 1),
    "a difference and its bound bound to names first": ("gap = actual - expected\nlimit = 1e-6\n"
                                                         "assert gap < limit", 1),
    "a mean squared difference": ("import numpy as np\nassert np.mean((actual - expected) ** 2) < 1e-6", 1),
    "a difference a helper returns": ("def gap(a, b):\n    return a - b\n"
                                      "def test():\n    assert gap(actual, expected) < 1e-6", 1),
    "a bound a helper returns": ("def limit():\n    return 1e-6\n"
                                 "def test():\n    assert actual - expected < limit()", 1),
    "a time within a tolerance": ("import pytest\nassert wall_seconds == pytest.approx(3.0, rel=0.25)", 1),
    "a mask of positions by a magnitude": (
        "import numpy as np\nrows, cols = np.arange(q_len)[:, None], np.arange(k_len)[None, :]\n"
        "keep = np.abs(cols - rows) < window", 1),
    "a mask of positions as a plain ordering": ("keep = (cols - rows < window) & (rows - cols < window)", 0),
    "a mask of a fractional range": ("import numpy as np\nrows = np.arange(0, 1, 0.1)\n"
                                     "keep = np.abs(rows - 1) < window", 1),
    "a mask of positions changed in place": ("import numpy as np\nrows = np.arange(n)\nrows[0] = 0.5\n"
                                             "keep = np.abs(rows - 1) < window", 1),
    # A band around a float target, in an assertion (`_band`), each way round and as a conjunction.
    "a time bounded on both sides": ("assert 2.25 <= long_seconds / short_seconds <= 3.75", 1),
    "a band written the other way": ("assert 3.75 >= ratio >= 2.25", 1),
    "a band about zero": ("assert -1e-6 <= residual <= 1e-6", 1),
    "a band through names": ("low, high = 2.25, 3.75\nassert low <= ratio <= high", 1),
    "a target less and plus a fraction of it": ("assert c - c / 4 <= x <= c + c / 4", 1),
    "a target scaled both ways": ("assert 0.75 * c <= x <= 1.25 * c", 1),
    "a band as a conjunction": ("assert x >= 2.25 and x <= 3.75", 1),
    "a band of masks": ("import numpy as np\nassert np.all((x > 2.25) & (x < 3.75))", 1),
    "a statistic's band": ("assert 0.3 < revealed.mean() < 0.7", 1),
    "a ratio's band above one": ("assert 1.0 <= ratio <= 1.05", 1),
    "a band beside an error in one assertion": (
        "assert abs(a - b) <= 1e-6 and 2.25 <= ratio and ratio <= 3.75", 2),
    "two bands in one assertion": ("assert 2.25 <= r and r <= 3.75 and 0.3 < s and s < 0.7", 2),
    "a share's domain": ("assert all(1 / 8 <= load <= 1.0 for load in loads)", 0),
    "an error bounded the other way round": ("import numpy as np\nassert 1e-6 >= np.max(np.abs(a - b))", 1),
    "a sign and a bound": ("assert 0 <= share <= 0.5", 0),
    "a cosine's domain": ("assert -1.0 <= float(alignment) <= 1.0", 0),
    "a cosine near one": ("assert 0.99 <= cosine <= 1.0", 0),
    "the sleeps' floor and the task's wall time": ("assert 1.2 < short + long <= wall", 0),
    "a duration's floor and timeout": ("import time\nbegan = time.monotonic()\n"
                                       "assert 4.0 <= time.monotonic() - began < 30", 0),
    "a count's band of unknown provenance": ("assert 3 * batches <= tallies['error'] <= 4 * batches", 0),
    "a schedule's window in a helper": ("scale = 2 if 0.1 <= index / 4 <= 0.8 else 1", 0),
    "two orderings of two values": ("assert a < 2.5 and b > 1.5", 0),
    "a timeout": ("assert elapsed < 5.0", 0),
    "a magnitude against an exact zero, written as a float": (
        "import numpy as np\nassert np.abs(actual - expected).max() <= 0.0", 0),
    # Against zero an ordering asks a sign; a margin folded in out of sight, in a helper's parameter, is a
    # gap SLOP020 leaves to review (ledger, P2).
    "a margin folded into a helper's result, against zero": (
        "def excess(a, b, allowance):\n    return abs(a - b) - allowance\n"
        "def test():\n    assert excess(a, b, 1e-6) <= 0", 0),
    "a margin folded in, against zero": ("assert abs(a - b) - 1e-6 <= 0", 1),
    "a duration against a timeout": ("import time\nbegan = time.monotonic()\n"
                                     "waited = time.monotonic() - began\nassert waited < 1.5", 0),
    "a schedule read against a clock": ("import time\nbegan = time.perf_counter()\n"
                                        "now = time.perf_counter() - began\nready = arrivals[0] <= now", 0),
    "a helper's magnitude that moved": ("import jax.numpy as jnp\ndef total(tree):\n"
                                        "    return sum(jnp.abs(leaf).sum() for leaf in tree)\n"
                                        "def test():\n    assert total(gradients) > 0", 0),
    "a sum of magnitudes that moved": ("import jax.numpy as jnp\n"
                                       "assert float(sum(jnp.abs(leaf).sum() for leaf in leaves)) > 0", 0),
    "a ratio bound": ("assert np.mean(trained[-5:]) < 0.5 * np.mean(trained[:5])", 0),
    "a bound written out": ("assert starts.max() <= 2.0 - 0.4 - 0.1", 0),
    "a relative margin": ("assert actual <= expected * (1 + 1e-6)", 1),
    "a field of a record beside an error": ("row = {'name': name}\nrow['gap'] = abs(a - b)\n"
                                            "row['flops_ratio'] = compiled / reference\n"
                                            "late = row['flops_ratio'] > row['flops_bound']", 0),
    "a record's error field": ("row = {'name': name}\nrow['gap'] = abs(a - b)\nassert row['gap'] < 1e-6", 1),
    "a helper's count beside its error, by place": ("def movement(a, b, flops):\n"
                                                    "    return abs(a - b), flops\n"
                                                    "def test():\n    moved, flops = movement(a, b, 3)\n"
                                                    "    _, wider = movement(a, b, 6)\n"
                                                    "    assert wider > flops", 0),
    "a helper's error by place": ("def movement(a, b, flops):\n    return abs(a - b), flops\n"
                                  "def test():\n    moved, flops = movement(a, b, 3)\n"
                                  "    assert moved < 1e-6", 1),
    "a magnitude against a domain threshold": ("import numpy as np\n"
                                               "clamped = int((np.abs(moments[..., 4:]) > 20).sum())\n"
                                               "bounded = np.abs(activations).max() < 100.0", 0),
    "a squared count": ("assert 2 * width ** 2 + flops > budget", 0),
    # The places of tuple-returning helpers only grow, so mutual recursion reaches its least fixed point.
    "helpers returning tuples through one another": (
        "def f(x):\n    if x <= 0:\n        return (0, 0)\n    return g(abs(x - 1))[0], 0\n"
        "def g(x):\n    if x <= 0:\n        return (0, 0)\n    return h(x - 1)[0], 0\n"
        "def h(x):\n    if x <= 0:\n        return (0, 0)\n    return f(x - 1)[0], 0\n"
        "def test():\n    assert f(a)[0] < 1e-6", 0),
    "token counts a helper of the file returns": (
        "def delivered(batches):\n    targets = [0, 0]\n    for source, count in batches:\n"
        "        targets[int(source)] += int(count)\n    return targets\n"
        "def test():\n    targets = delivered(batches)\n    assert abs(targets[0] - targets[1]) <= 9", 0),
    "a count of matches": ("assert abs(sum(value // 1000 == 2 for value in phase) - 68) <= 1", 0),
    "a squared difference": ("import numpy as np\nassert np.mean((actual - expected) ** 2) < 1e-6", 1),
    "a margin bound to a name": ("margin = 1e-6\nassert actual <= expected + margin", 1),
    "a margin a helper returns": ("def margin():\n    return 1e-6\n"
                                  "def test():\n    assert actual <= expected + margin()", 1),
    "a margin of a whole unit": ("assert actual <= expected + 1.0", 1),
    "a zero margin": ("assert actual <= expected + 0.0", 0),
    "a count of errors": ("import numpy as np\nerrors = np.abs(actual - expected)\ncount = len(errors)\n"
                          "assert count <= 100", 0),
    "a count beside an error in a mapping": ("report = {'error': abs(a - b), 'count': len(rows)}\n"
                                             "assert report['count'] <= 100", 0),
    "an error read from a mapping": ("report = {'error': abs(a - b), 'count': len(rows)}\n"
                                     "assert report['error'] <= 1e-6", 1),
    "an error taken in a loop": ("gaps = [abs(actual - expected)]\nfor gap in gaps:\n"
                                 "    assert gap < 1e-6", 1),
    "a bound a default gives": ("def limit(value=1e-6):\n    return value\n"
                                "def test():\n    assert actual - expected < limit()", 1),
    "exact equality": ("import numpy as np\nnp.testing.assert_array_equal(a, b)\nassert a == b", 0),
    "a tolerance call told to be exact": ("import torch\n"
                                          "torch.testing.assert_close(a, b, atol=0, rtol=0)", 0),
    "pytest's approx given only a zero abs": ("import pytest\nassert a == pytest.approx(b, abs=0)", 0),
    "both of numpy's tolerances zero": ("import numpy as np\nnp.testing.assert_allclose(a, b, rtol=0)", 0),
    "a count bounded on both sides": ("assert 67 <= len(rows) <= 69", 0),
    "a difference of counts within whole bounds": ("left, right = len(a), len(b)\n"
                                                    "assert -1 <= left - right <= 1", 0),
    "index arithmetic": ("while len(ended) < bridge.MAX_CELLS - 1:\n    pass\n"
                         "hung = output[stack:] if 0 <= stack < len(output) - 4000 else ''", 0),
    "a difference against a named tolerance": ("assert actual - expected < tolerance", 1),
    "a magnitude that is nonzero, or zero": ("import numpy as np\nassert np.abs(a - b).max() > 0\n"
                                             "assert np.linalg.norm(gradient) <= 0.0", 0),
    "a loss that falls": ("assert loss - first_loss < 0", 0),
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


def test_counts_a_helper_beside_the_file_returns_are_whole_where_its_own_file_proves_it(lint, tmp_path):
    """A test importing its support's helper by a bare module name: the helper's token counts are whole
    as its own file makes them, and a helper there that may return a float is not, nor one its file
    binds twice."""
    _module(lint, tmp_path, "tests/counting_support.py", """\
        def delivered(batches):
            targets = [0, 0]
            for source, count in batches:
                targets[int(source)] += int(count)
            return targets


        def shares(batches):
            left, right = delivered(batches)
            return left / 2, right / 2


        def rebound(batches):
            return delivered(batches)


        def rebound(batches):
            return [0.25, 0.75]
        """)
    body = """\
        from counting_support import delivered, rebound, shares


        def test():
            left, right = delivered(batches)
            assert abs(left - right) <= 12 + 12
            half, other = shares(batches)
            assert abs(half - other) <= 12
            one, two = rebound(batches)
            assert abs(one - two) <= 12
        """
    found = list(lint.tolerances(_module(lint, tmp_path, "tests/test_case.py", body)))
    assert [finding.line for finding in found] == [8, 10]


def test_an_imported_helper_that_returns_an_error_is_compared_as_a_local_one_is(lint, tmp_path):
    """A test importing its support's difference helper (`largest_difference`) and holding its result
    within a tolerance compares an error, as the same helper defined in the test does, and a tuple's
    place is read as the helper's own file places it: the gap is an error, the mean is not."""
    helper = """\
        import numpy as np


        def largest_difference(a, b):
            return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


        def mean_and_gap(a, b):
            return float(np.mean(a)), largest_difference(a, b)
        """
    _module(lint, tmp_path, "tests/model_support.py", helper)
    test = """\


        def test():
            assert largest_difference(a, b) < 1e-5
            mean, gap = mean_and_gap(a, b)
            assert gap < 1e-5
            assert mean < 1e-5
        """
    imported = """\
        from model_support import largest_difference, mean_and_gap
        """ + test
    for relative, body in (("tests/test_imported.py", imported), ("tests/test_local.py", helper + test)):
        found = list(lint.tolerances(_module(lint, tmp_path, relative, body)))
        assert len(found) == 2, relative


def test_the_gate_compares_floats_and_scripts_outside_tests_and_tools_are_not_read(lint, tmp_path):
    body = "import numpy as np\nnp.testing.assert_allclose(a, b, rtol=1e-5)\n"
    assert not list(lint.tolerances(_module(lint, tmp_path, "tests/reference_error.py", body)))
    assert not list(lint.tolerances(_module(lint, tmp_path, "examples/run.py", body)))
    assert list(lint.tolerances(_module(lint, tmp_path, "tools/report.py", body)))


def test_the_inventory_holds_a_file_at_its_count_and_only_shrinks(lint, tmp_path):
    """A file the inventory holds at its count passes; one that grew reports every comparison; one that
    shrank reports once, until the count comes down, and one with none left is told to remove its
    entry. Each reading sees the inventory as it is then, in one process."""
    body = "import numpy as np, pytest\nnp.testing.assert_allclose(a, b)\nassert a == pytest.approx(b)\n"

    def findings(count, body=body):
        inventory = {"files": {"tests/test_case.py": {"comparisons": count, "lane": "Kernels"}}}
        return list(lint.tolerances(_module(lint, tmp_path, "tests/test_case.py", body, inventory)))

    assert findings(2) == []
    assert len(findings(1)) == 2
    [lower] = findings(3)
    assert lower.line == 1
    [remove] = findings(2, "x = 1\n")
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


def _inventory(counts: dict):
    return {"files": {name: {"comparisons": count, "lane": "Kernels"} for name, count in counts.items()}}


def test_no_commit_loosens_the_inventory_its_parents_kept(lint, tmp_path):
    """A commit may lower a count and carry an entry across a rename; one that raises a count or adds a
    file is reported, and a merge that keeps a count past every parent's. A merge keeping one parent's
    count is held at that parent's own commit; a file left above its count is the inventory's to bring
    down (`tolerances`)."""
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
    # A merge keeping main's 8 keeps one parent's count, judged where main kept it.
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "merge", "-q", "--no-commit", "main"],
                   cwd=repo, check=True)
    _commit(repo, _inventory({"tests/test_b.py": 8}), "merge keeps main's 8")
    assert list(lint.ratchet(repo)) == []
    # One past both parents' is the merge's own.
    subprocess.run(["git", "checkout", "-q", "main"], cwd=repo, check=True)
    _commit(repo, _inventory({"tests/test_b.py": 8}), "main again", {"tests/other.py": "x = 2\n"})
    subprocess.run(["git", "checkout", "-q", "branch"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "merge", "-q", "--no-commit", "main"],
                   cwd=repo, check=True)
    _commit(repo, _inventory({"tests/test_b.py": 9}), "merge raises past both")
    assert any("raises tests/test_b.py" in finding.message for finding in lint.ratchet(repo))
    # Removing the inventory, then making it again looser, is reported at the removal.
    subprocess.run(["git", "rm", "-q", "tools/tolerance_inventory.json"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "removed"], cwd=repo,
                   check=True)
    _commit(repo, _inventory({"tests/test_b.py": 20}), "made again")
    assert any("removes the inventory" in finding.message for finding in lint.ratchet(repo))


def test_a_change_of_the_rules_reading_grows_the_inventory_only_for_files_it_leaves_unchanged(lint, tmp_path):
    """A commit that changes SLOP020's reading (`RULE`) may add or raise an entry for a file it leaves
    unchanged, against its parent and its base alike; a file it also changes, or one it adds, may not
    grow."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    _commit(repo, _inventory({"tests/test_a.py": 1}), "made",
            {"tests/test_a.py": "a = 1\n", "tests/test_b.py": "b = 1\n", lint.RULE: "reading = 1\n"})
    subprocess.run(["git", "checkout", "-q", "-b", "head"], cwd=repo, check=True)
    _commit(repo, _inventory({"tests/test_a.py": 2, "tests/test_b.py": 1}), "the rule reads more",
            {lint.RULE: "reading = 2\n"})
    assert list(lint.ratchet(repo, "main")) == []
    # A later edit of a file whose count the rule raised keeps that count against the base.
    _commit(repo, _inventory({"tests/test_a.py": 2, "tests/test_b.py": 1}), "a docstring",
            {"tests/test_a.py": "a = 1  # one\n"})
    assert list(lint.ratchet(repo, "main")) == []
    _commit(repo, _inventory({"tests/test_a.py": 3, "tests/test_b.py": 1, "tests/test_c.py": 1}),
            "the rule and the files change", {lint.RULE: "reading = 3\n", "tests/test_a.py": "a = 2\n",
                                               "tests/test_c.py": "c = 1\n"})
    messages = [finding.message for finding in lint.ratchet(repo, "main")]
    assert any("raises tests/test_a.py" in message for message in messages)
    assert any("adds tests/test_c.py" in message for message in messages)
    assert not any("tests/test_b.py" in message for message in messages)


def test_a_merge_bringing_a_count_its_other_parent_raised_by_rereading_is_no_loosening(lint, tmp_path):
    """Main raises a file's count by changing the rule's reading while a lane edits that file; the lane
    merging main keeps main's count, past the lane's but no parent's, and is no loosening, against its
    parents or against main."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    _commit(repo, _inventory({"tests/test_a.py": 7}), "made",
            {"tests/test_a.py": "a = 1\n", lint.RULE: "r = 1\n"})
    subprocess.run(["git", "checkout", "-q", "-b", "lane"], cwd=repo, check=True)
    _commit(repo, _inventory({"tests/test_a.py": 7}), "the lane edits the file",
            {"tests/test_a.py": "a = 2\n"})
    subprocess.run(["git", "checkout", "-q", "main"], cwd=repo, check=True)
    _commit(repo, _inventory({"tests/test_a.py": 8}), "the rule reads more", {lint.RULE: "r = 2\n"})
    subprocess.run(["git", "checkout", "-q", "lane"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "merge", "-q", "--no-commit", "main"],
                   cwd=repo, check=True)
    _commit(repo, _inventory({"tests/test_a.py": 8}), "the lane merges main")
    assert list(lint.ratchet(repo, "main")) == []


def _merged(repo, branch, into, inventory, message, files=None):
    """`branch` merged into `into` without committing, never fast-forward, then committed with
    `inventory` and `files`."""
    subprocess.run(["git", "checkout", "-q", into], cwd=repo, check=True)
    merge = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "merge", "-q", "--no-ff", "--no-commit"]
    subprocess.run([*merge, branch], cwd=repo, check=False, capture_output=True)
    _commit(repo, inventory, message, files)


def test_a_merge_keeps_an_entry_its_parents_renamed_apart_only_once(lint, tmp_path):
    """Two parents rename one inventoried file to two names; a merge keeping both holds its allowance
    twice, though each name keeps its own parent's count."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    _commit(repo, _inventory({"tests/test_a.py": 2}), "made",
            {"tests/test_a.py": "a = 1  # two comparisons\n"})
    subprocess.run(["git", "checkout", "-q", "-b", "lane"], cwd=repo, check=True)
    subprocess.run(["git", "mv", "tests/test_a.py", "tests/test_b.py"], cwd=repo, check=True)
    _commit(repo, _inventory({"tests/test_b.py": 2}), "the lane renames it")
    subprocess.run(["git", "checkout", "-q", "main"], cwd=repo, check=True)
    subprocess.run(["git", "mv", "tests/test_a.py", "tests/test_c.py"], cwd=repo, check=True)
    _commit(repo, _inventory({"tests/test_c.py": 2}), "main renames it")
    # Each side edits its copy past git's notice of the rename before the merge, converting one
    # comparison, so its inventory changes with it.
    for branch, name in (("lane", "test_b.py"), ("main", "test_c.py")):
        subprocess.run(["git", "checkout", "-q", branch], cwd=repo, check=True)
        _commit(repo, _inventory({f"tests/{name}": 1}), "edited",
                {f"tests/{name}": f"a = 1  # one comparison\n{branch * 40}\n" * 8})
    _merged(repo, "main", "lane", _inventory({"tests/test_b.py": 1, "tests/test_c.py": 1}),
            "the merge keeps both")
    assert any("renamed it apart" in finding.message for finding in lint.ratchet(repo))
    # Kept under a third name the merge itself gives, the copy is the same entry still.
    subprocess.run(["git", "reset", "-q", "--hard", "HEAD~1"], cwd=repo, check=True)
    _merged(repo, "main", "lane", _inventory({"tests/test_d.py": 1, "tests/test_c.py": 1}),
            "the merge renames one copy")
    subprocess.run(["git", "mv", "-f", "tests/test_b.py", "tests/test_d.py"], cwd=repo, check=True)
    amend = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--amend", "--no-edit"]
    subprocess.run(amend, cwd=repo, check=True)
    assert any("renamed it apart" in finding.message for finding in lint.ratchet(repo))


def test_only_the_heads_own_rereadings_authorize_a_count_against_its_base(lint, tmp_path):
    """A rule change before the base raised a file's count, which main later brought down; a stale lane
    from before that merged back, keeping the old count, is held against main's lower one."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    _commit(repo, _inventory({"tests/test_a.py": 1}), "made",
            {"tests/test_a.py": "a = 1\n", lint.RULE: "r = 1\n"})
    _commit(repo, _inventory({"tests/test_a.py": 3}), "the rule reads more", {lint.RULE: "r = 2\n"})
    subprocess.run(["git", "checkout", "-q", "-b", "lane"], cwd=repo, check=True)
    _commit(repo, _inventory({"tests/test_a.py": 3}), "the lane works", {"tests/other.py": "o = 1\n"})
    subprocess.run(["git", "checkout", "-q", "main"], cwd=repo, check=True)
    _commit(repo, _inventory({"tests/test_a.py": 1}), "main brings it down", {"tests/test_a.py": "a = 2\n"})
    _merged(repo, "lane", "main", _inventory({"tests/test_a.py": 1}), "main merges the lane")
    subprocess.run(["git", "checkout", "-q", "lane"], cwd=repo, check=True)
    _merged(repo, "main", "lane", _inventory({"tests/test_a.py": 3}), "the lane keeps its 3",
            {"tests/test_a.py": "a = 1\n"})
    assert any("test_a.py's count" in finding.message for finding in lint.ratchet(repo, "main"))


def test_the_ratchet_holds_a_branch_to_its_base_and_refuses_a_shallow_history(lint, tmp_path):
    """A branch whose inventory is looser than its base's merge-base is reported against that base;
    a shallow clone cannot be held, and says so rather than passing."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    _commit(repo, _inventory({"tests/test_a.py": 4}), "made", {"tests/test_a.py": "a = 1\n"})
    _commit(repo, _inventory({"tests/test_a.py": 3}), "lowered")
    subprocess.run(["git", "checkout", "-q", "-b", "head", "HEAD~1"], cwd=repo, check=True)
    _commit(repo, _inventory({"tests/test_a.py": 5}), "a raised count")
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
