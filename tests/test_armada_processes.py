"""A CI task's processes (tools/armada/ci.py's `task`): every reuse found before any process starts, a file
whose process ended red set aside by no resumed attempt, and a report a killed pytest cut short graded as
none."""

import json

import pytest
from armada_support import commit, interpreter, lifetime_task, loaded, written


@pytest.fixture
def ci(tmp_path, monkeypatch):
    return loaded(tmp_path, monkeypatch)


def test_a_tasks_reuse_is_found_before_any_of_its_files_changes_the_checkout(ci, tmp_path, monkeypatch,
                                                                            cache_worker):
    """A task finds every row it reuses before its first file runs: a file that leaves the checkout as an
    earlier green run of a later file read it makes no reuse of that row, which the commit's own checkout
    keys (row 3924's review: file A writing a fixture back to `good`, B's run on `good` was reused though the
    commit holds `bad`)."""
    (tmp_path / "data").mkdir()
    (tmp_path / "data/state.txt").write_text("good")
    lifetime_task(ci, tmp_path, monkeypatch, cache_worker,
                  {"test_b.py": "def test_b():\n    assert open('data/state.txt').read() == 'good'\n"}, {})
    (tmp_path / "data/state.txt").write_text("bad")
    written(tmp_path, {"test_a.py": "def test_a():\n    open('data/state.txt', 'w').write('good')\n"})
    ci.task("3.12", ["tests/test_a.py", "tests/test_b.py"], "", tmp_path / "verdict.json", 600.0)
    rows = {row["name"]: row for row in json.loads((tmp_path / "verdict.json").read_text())["rows"]}
    assert "cached" not in rows["3.12:tests/test_b.py"]


def test_a_resumed_task_runs_whole_a_file_whose_process_ended_red_after_its_tests_passed(ci, tmp_path,
                                                                                         monkeypatch):
    """A file whose tests passed in a process that then ended red (its exit's crash) keeps none of them for
    the next attempt of a recalled task, which runs the file whole again rather than grade it green on the
    passes of a process that failed (row 3924's review)."""
    written(tmp_path, {
        "test_crashing.py": "import atexit, os\nfrom pathlib import Path\n"
                            "atexit.register(lambda: None if Path('healed').exists() else os._exit(3))\n"
                            "def test_passes():\n    with Path('ran').open('a') as ran: ran.write('ran ')\n",
        "test_other.py": "def test_other():\n    pass\n"})
    interpreter(tmp_path, "3.12")
    tests = ["tests/test_crashing.py", "tests/test_other.py"]
    out = tmp_path / "verdict.json"
    monkeypatch.setenv("ARMADA_ARTIFACTS", str(tmp_path / "first"))
    ci.task("3.12", tests, "", out, 600.0, cache="off")
    rows = {row["name"]: row["exitCode"] for row in json.loads(out.read_text())["rows"]}
    assert rows == {"3.12:tests/test_crashing.py": 1, "3.12:tests/test_other.py": 0}
    (tmp_path / "healed").write_text("")
    (tmp_path / "ran").unlink()
    monkeypatch.setenv("ARMADA_RESUME", str(tmp_path / "first"))
    monkeypatch.setenv("ARMADA_ARTIFACTS", str(tmp_path / "second"))
    ci.task("3.12", tests, "", out, 600.0, cache="off")
    assert (tmp_path / "ran").read_text() == "ran "
    rows = {row["name"]: row["exitCode"] for row in json.loads(out.read_text())["rows"]}
    assert rows == {"3.12:tests/test_crashing.py": 0, "3.12:tests/test_other.py": 0}


@pytest.mark.parametrize("failing", [False, True])
def test_a_dry_prepares_tests_past_its_recycle_bound_run_on_in_fresh_processes_as_one_row(
        ci, tmp_path, monkeypatch, failing):
    """A dry prepare's pytest that holds more than its bound (`tested`'s `recycle_at`, recycle.py) ends after
    the test running, and the tests left run in a fresh process, each test once, so what a process compiled
    is returned at its end: here a bound every process passes, each test its own process. The row is one,
    every test's case in it, and red where a test of any of its processes failed."""
    middle = "    assert False\n" if failing else "    pass\n"
    written(tmp_path, {"test_heavy.py": "import os\n"
                                        "def ran():\n"
                                        "    with open('pids', 'a') as pids:\n"
                                        "        pids.write(f'{os.getpid()} ')\n"
                                        "def test_a():\n    ran()\n"
                                        f"def test_b():\n    ran()\n{middle}"
                                        "def test_c():\n    ran()\n"})
    interpreter(tmp_path, "3.12")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ARMADA_ARTIFACTS", str(tmp_path / "artifacts"))
    out = tmp_path / "verdict.json"
    ci.tested("3.12", ["tests/test_heavy.py"], "", out, 600.0, "off", 5.0, "", {}, recorded=False,
              recycle_at=1)
    [row] = json.loads(out.read_text())["rows"]
    assert len(set((tmp_path / "pids").read_text().split())) == 3
    assert sorted(row["tests"]) == [f"tests/test_heavy.py::test_{name}" for name in "abc"]
    assert row["exitCode"] == (1 if failing else 0)


def test_a_report_its_killed_pytest_cut_short_greens_nothing_and_the_tasks_other_files_are_graded(
        ci, tmp_path, monkeypatch):
    """A file whose pytest left its JUnit report and its collection's record cut short (killed as it wrote
    them) is graded as a run that wrote none, a red of the file whole, and the task's other files are graded
    each by its own process."""
    (tmp_path / "tests/cut").mkdir(parents=True)
    cutting = ("import os\n"
               "def pytest_unconfigure(config):\n"
               "    for cut in (config.option.xmlpath, os.environ['DEW_COLLECTED']):\n"
               "        with open(cut, 'r+') as kept:\n"
               "            kept.truncate(5)\n")
    written(tmp_path, {"cut/conftest.py": cutting,
                       "cut/test_cut.py": "def test_cut():\n    pass\n",
                       "test_whole.py": "def test_whole():\n    pass\n"})
    interpreter(tmp_path, "3.12")
    monkeypatch.setenv("ARMADA_ARTIFACTS", str(tmp_path / "artifacts"))
    out = tmp_path / "verdict.json"
    ci.task("3.12", ["tests/cut/test_cut.py", "tests/test_whole.py"], "", out, 600.0, cache="off")
    rows = {row["name"]: row["exitCode"] for row in json.loads(out.read_text())["rows"]}
    assert rows == {"3.12:tests/cut/test_cut.py": 1, "3.12:tests/test_whole.py": 0}
    # Each process sealed as graded, the cut one red though it exited 0: a recall reruns its file whole
    # (row 3944's review).
    ended = [json.loads(line) for line in (tmp_path / "artifacts/progress.jsonl").read_text().splitlines()]
    assert [line["code"] for line in ended if "ended" in line] == [1, 0]


def test_each_file_of_a_task_runs_on_the_commits_checkout_whatever_an_earlier_file_wrote(ci, tmp_path):
    """A file that rewrites a later file of its task (here its source), or makes entries beside it (a
    conftest.py that skips it, a link to a directory holding a package it imports, a package it removes
    again whose directory stays), leaves nothing for it: the next process runs on the checkout its commit
    holds, so a file failing there is red and one passing there green, never the other on a checkout the
    commit does not hold (row 3932's, 3944's and 3953's reviews)."""
    skipping = "import pytest\\ndef pytest_runtest_setup(item):\\n    pytest.skip()\\n"
    passing = "def test_b():\\n    pass\\n"
    # A package outside the checkout, which a link the earlier file makes would put on the later's path.
    outside = tmp_path.parent / f"{tmp_path.name}-helper"
    outside.mkdir()
    (outside / "__init__.py").write_text("PRESENT = True\n")
    (tmp_path / "tests/sub").mkdir(parents=True)
    written(tmp_path, {"test_a.py": "import os\nfrom pathlib import Path\n"
                                    "def test_a():\n"
                                    f"    Path('tests/sub/test_b.py').write_text('{passing}')\n"
                                    f"    Path('tests/sub/conftest.py').write_text('{skipping}')\n"
                                    f"    os.symlink({str(outside)!r}, 'tests/helper')\n"
                                    "    Path('tests/optional').mkdir()\n"
                                    "    Path('tests/optional/__init__.py').write_text('')\n",
                       "sub/test_b.py": "def test_b():\n    assert False\n",
                       "test_c.py": "import importlib.util\ndef test_c():\n"
                                    "    assert importlib.util.find_spec('helper') is None\n"
                                    "    assert importlib.util.find_spec('optional') is None\n"})
    interpreter(tmp_path, "3.12")
    commit(tmp_path, "the commit")
    out = tmp_path / "verdict.json"
    ci.task("3.12", ["tests/test_a.py", "tests/sub/test_b.py", "tests/test_c.py"], "", out, 600.0,
            cache="off")
    rows = {row["name"]: row["exitCode"] for row in json.loads(out.read_text())["rows"]}
    assert rows == {"3.12:tests/test_a.py": 0, "3.12:tests/sub/test_b.py": 1, "3.12:tests/test_c.py": 0}
