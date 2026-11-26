"""A CI task's processes (tools/armada/ci.py's `task`): every reuse found before any process starts, a file
whose process ended red set aside by no resumed attempt, and a report a killed pytest cut short graded as
none."""

import json
import subprocess
import sys

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


def recycled_row(ci, tmp_path, monkeypatch, source: str, hook: str = "", recycle_at: int = 1,
                 **kwargs) -> dict:
    """The row a dry prepare's task gives tests/test_recycled.py of `source` (a conftest `hook` in its
    continuations, where DEW_CONTINUES_ROW is set), under a recycle bound every process passes (or
    `recycle_at`)."""
    written(tmp_path, {"test_recycled.py": source})
    if hook:
        conftest = tmp_path / "tests" / "conftest.py"
        conftest.write_text((conftest.read_text() if conftest.is_file() else "")
                            + "\nimport os\n\n\ndef pytest_collection_modifyitems(items):\n"
                              f"    if os.environ.get('DEW_CONTINUES_ROW'):\n        {hook}\n")
    interpreter(tmp_path, "3.12")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ARMADA_ARTIFACTS", str(tmp_path / "artifacts"))
    out = tmp_path / "verdict.json"
    split = kwargs.pop("split", "")
    ci.tested("3.12", ["tests/test_recycled.py"], split, out, 600.0, "off", 5.0, "", {},
              recycle_at=recycle_at, **{"recorded": False, **kwargs})
    [row] = json.loads(out.read_text())["rows"]
    return row


@pytest.mark.parametrize("failing", [False, True])
def test_a_dry_prepares_tests_past_its_recycle_bound_run_on_in_fresh_processes_as_one_row(
        ci, tmp_path, monkeypatch, failing):
    """A dry prepare's pytest that holds more than its bound (`tested`'s `recycle_at`, recycle.py) ends after
    the test running, and the tests left run in a fresh process, each test once, so what a process compiled
    is returned at its end: here a bound every process passes, each test its own process. The row is one,
    every test's case in it, and red where a test of any of its processes failed; recorded as a blind watch's
    (`run_keys.record`'s `opened` None), keyed by none, as what each process read is no one record's, its red
    still beginning its next epoch and its green still writing its baseline."""
    middle = "    assert False\n" if failing else "    pass\n"
    recorded = []
    monkeypatch.setattr(ci.run_keys, "record", lambda *args: recorded.append(args) or set())
    row = recycled_row(ci, tmp_path, monkeypatch,
                       "import os\ndef ran():\n    with open('pids', 'a') as pids:\n"
                       "        pids.write(f'{os.getpid()} ')\n"
                       f"def test_a():\n    ran()\ndef test_b():\n    ran()\n{middle}"
                       "def test_c():\n    ran()\n",
                       recorded=True)
    # Recorded as a blind watch's (no inputs, so no key), its red's epoch and its green's baseline kept.
    [(_, [recorded_row], _, _, _, opened, *_)] = recorded
    assert opened is None and recorded_row["name"] == row["name"]
    assert len(set((tmp_path / "pids").read_text().split())) == 3
    assert sorted(row["tests"]) == [f"tests/test_recycled.py::test_{name}" for name in "abc"]
    assert row["exitCode"] == (1 if failing else 0)


def test_a_test_marked_fresh_process_runs_in_a_process_of_its_own_under_a_bound_none_passes(
        ci, tmp_path, monkeypatch):
    """A test marked `fresh_process` (recycle.py's `FRESH`) runs in a process of its own: its pytest ends
    after the test before it and after it, the tests left running on in fresh processes, under a resident
    bound no process passes; every test once, in one green row."""
    monkeypatch.setattr(ci.run_keys, "record", lambda *args: set())
    row = recycled_row(ci, tmp_path, monkeypatch,
                       "import os\nimport pytest\ndef ran(name):\n    with open('pids', 'a') as pids:\n"
                       "        pids.write(f'{name}:{os.getpid()} ')\n"
                       "def test_a():\n    ran('a')\ndef test_b():\n    ran('b')\n"
                       "@pytest.mark.fresh_process\ndef test_c():\n    ran('c')\n"
                       "def test_d():\n    ran('d')\ndef test_e():\n    ran('e')\n",
                       recycle_at=1 << 50)
    pids = dict(entry.split(":") for entry in (tmp_path / "pids").read_text().split())
    assert pids["a"] == pids["b"] and pids["d"] == pids["e"]
    assert len({pids["b"], pids["c"], pids["d"]}) == 3
    assert row["exitCode"] == 0 and len(row["tests"]) == 5


def test_a_plain_task_runs_its_tests_past_its_recycle_bound_in_fresh_processes(ci, tmp_path, monkeypatch):
    """A plain task (`task`, no dry prepare's) runs its pytest under `PLAIN_RECYCLE_AT`, so a process holding
    more than it after a test ends and the tests left run in a fresh one, its row one and green (row 13063:
    a group's compiles held 10 GiB of a 10 GiB container in one process, each green alone)."""
    monkeypatch.setattr(ci, "PLAIN_RECYCLE_AT", 1)
    monkeypatch.setattr(ci.run_keys, "record", lambda *args: set())
    # Outside the checkout, which each later process finds as the commit holds it (`restored`).
    pids = tmp_path.parent / f"{tmp_path.name}-pids"
    written(tmp_path, {"test_plain.py": f"import os\ndef ran():\n    with open({str(pids)!r}, 'a') as pids:\n"
                                        "        pids.write(f'{os.getpid()} ')\n"
                                        "def test_a():\n    ran()\ndef test_b():\n    ran()\n"})
    commit(tmp_path, "the commit")
    interpreter(tmp_path, "3.12")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ARMADA_ARTIFACTS", str(tmp_path / "artifacts"))
    out = tmp_path / "verdict.json"
    ci.task("3.12", ["tests/test_plain.py"], "", out, 600.0, "off")
    [row] = json.loads(out.read_text())["rows"]
    assert row["exitCode"] == 0, row["output"]
    assert len(set(pids.read_text().split())) == 2


def test_a_resumed_groups_test_its_recalled_attempt_passed_is_no_test_its_recycled_processes_missed(
        ci, tmp_path, monkeypatch):
    """A split group's attempt resumed after a recall (its ARMADA_RESUME holding a test it passed, set aside
    by `progress.resumed`) and run past a recycle is green: the test set aside ran, in the recalled attempt,
    and is none its processes failed to report (`graded_across`)."""
    resume = tmp_path.parent / f"{tmp_path.name}-resume"
    resume.mkdir()
    (resume / "progress.jsonl").write_text(
        json.dumps({"node": "tests/test_recycled.py::test_a", "seconds": 0.01, "pid": 1}) + "\n")
    monkeypatch.setenv("ARMADA_RESUME", str(resume))
    monkeypatch.setattr(ci.run_keys, "record", lambda *args: set())
    row = recycled_row(ci, tmp_path, monkeypatch,
                       "def test_a():\n    raise AssertionError('set aside, never run')\n"
                       "def test_b():\n    pass\ndef test_c():\n    pass\n",
                       recorded=True, split="1/1")
    assert row["exitCode"] == 0, row["output"]


def test_a_fresh_process_test_runs_alone_after_its_predecessor_was_set_aside_by_a_resume(
        ci, tmp_path, monkeypatch):
    """A split group resumed after a recall whose attempt passed the test before a `fresh_process` one (set
    aside, `progress.resumed`) still runs the marked test in a process of its own, apart from the test before
    it as the group now stands, under a bound no process passes."""
    resume = tmp_path.parent / f"{tmp_path.name}-resume"
    resume.mkdir()
    (resume / "progress.jsonl").write_text(
        json.dumps({"node": "tests/test_recycled.py::test_b", "seconds": 0.01, "pid": 1}) + "\n")
    monkeypatch.setenv("ARMADA_RESUME", str(resume))
    monkeypatch.setattr(ci.run_keys, "record", lambda *args: set())
    pids = tmp_path.parent / f"{tmp_path.name}-pids"
    row = recycled_row(ci, tmp_path, monkeypatch,
                       "import os\nimport pytest\n"
                       f"def ran(name):\n    with open({str(pids)!r}, 'a') as out:\n"
                       "        out.write(f'{name}:{os.getpid()} ')\n"
                       "def test_a():\n    ran('a')\ndef test_b():\n    ran('b')\n"
                       "@pytest.mark.fresh_process\ndef test_c():\n    ran('c')\n",
                       recorded=True, split="1/1", recycle_at=1 << 50)
    ran = dict(entry.split(":") for entry in pids.read_text().split())
    assert row["exitCode"] == 0, row["output"]
    assert "b" not in ran and ran["a"] != ran["c"]


def test_a_recycled_rows_later_process_runs_on_the_commits_checkout(ci, tmp_path, monkeypatch):
    """A dry prepare's later process of one row (past its recycle bound) runs on the commit's checkout, not on
    what the process before it left: a tracked file it changed checked out again and a file it made removed
    (`restored`), so the row's green, which publishes its baseline, never ran on a checkout the commit does
    not hold (row 13000)."""
    (tmp_path / "data").mkdir()
    (tmp_path / "data/state.txt").write_text("good")
    written(tmp_path, {"test_recycled.py": (
        "from pathlib import Path\ndef test_a():\n    Path('data/state.txt').write_text('bad')\n"
        "    Path('left.txt').write_text('x')\ndef test_b():\n"
        "    assert Path('data/state.txt').read_text() == 'good'\n"
        "    assert not Path('left.txt').exists()\n")})
    commit(tmp_path, "the commit")
    interpreter(tmp_path, "3.12")
    monkeypatch.setattr(ci.run_keys, "record", lambda *args: set())
    monkeypatch.setenv("ARMADA_ARTIFACTS", str(tmp_path / "artifacts"))
    out = tmp_path / "verdict.json"
    ci.tested("3.12", ["tests/test_recycled.py"], "", out, 600.0, "off", 5.0, "", {}, recycle_at=1,
              recorded=True)
    [row] = json.loads(out.read_text())["rows"]
    assert row["exitCode"] == 0, row["output"]


def test_a_restored_modules_cached_bytecode_goes_with_its_rewrite(ci, tmp_path):
    """A module a process rewrote and imported, checked out again (`restored`), leaves none of the bytecode
    that process cached for it: its rewrite's size and second may match the restored source's, so the next
    process would run the rewrite (row 13000's review)."""
    (tmp_path / "tests/helper.py").write_text("WORD = 'good'\n")
    commit(tmp_path, "the commit")
    (tmp_path / "tests/helper.py").write_text("WORD = 'evil'\n")
    subprocess.run([sys.executable, "-c", "import helper"], cwd=tmp_path / "tests", check=True)
    assert list((tmp_path / "tests/__pycache__").glob("helper.*.pyc"))
    ci.restored({"tests/helper.py"}, ci.untracked(), tmp_path / "verdict.json")
    assert (tmp_path / "tests/helper.py").read_text() == "WORD = 'good'\n"
    assert not list((tmp_path / "tests/__pycache__").glob("helper.*.pyc"))


def test_a_restore_keeps_the_runners_files_beside_the_verdict_and_no_other_of_its_name(ci, tmp_path):
    """A checkout restored after a process (`restored`) keeps the task's verdict, each process's numbered one,
    and the runner's own files beside each (`SIDECARS`, the recorder's children), and removes any other entry
    the process made, one named as the verdict but for its suffix among them: a later process importing a
    `<out>.py` would run a module the commit does not hold (row 13000's review)."""
    written(tmp_path, {"test_kept.py": ""})
    commit(tmp_path, "the commit")
    before = ci.untracked()
    out = tmp_path / "tests/result"
    kept = [out.with_name(f"result.0.{name}") for name in ci.SIDECARS] + [tmp_path / "tests/result.0"]
    for path in [*kept, out.with_name("result.py"), out.with_name("result.0.py")]:
        path.write_text("")
    (tmp_path / "tests/result.0.lifetime.log.children").mkdir()
    (tmp_path / "tests/result.0.lifetime.log.children/1.log").write_text("")
    ci.restored(set(), before, out)
    assert all(path.exists() for path in kept)
    assert (tmp_path / "tests/result.0.lifetime.log.children/1.log").exists()
    assert not out.with_name("result.py").exists() and not out.with_name("result.0.py").exists()


@pytest.mark.parametrize(("source", "hook"), [
    # A crash at its exit, after its report: graded as it ended, nothing run past it.
    ("import atexit, os\ndef test_a():\n    atexit.register(lambda: os._exit(3))\n"
     "def test_b():\n    pass\n", ""),
    # Its last process exits 0 with no report of its own (its JUnit file removed as it exits).
    ("import atexit, os, sys\ndef test_a():\n    pass\ndef test_b():\n"
     "    report = next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--junitxml='))\n"
     "    atexit.register(lambda: os.remove(report))\n", ""),
    # Its last process's collection drops the tests left unreported (a valid report of no test, exit 5).
    ("def test_a():\n    pass\ndef test_b():\n    pass\n", "items.clear()"),
    # Its last process's collection drops test_c alone, test_b reported.
    ("def test_a():\n    pass\ndef test_b():\n    pass\ndef test_c():\n    pass\n",
     "items[:] = [item for item in items if not item.name.endswith('_c')]"),
    # A later import no longer defines test_c (test_a wrote the file its definition reads).
    ("import pathlib\ndef test_a():\n    pathlib.Path('ran-a').touch()\ndef test_b():\n    pass\n"
     "if not pathlib.Path('ran-a').exists():\n    def test_c():\n        pass\n", ""),
])
def test_a_recycled_row_whose_processes_did_not_report_every_test_its_first_selected_is_red(
        ci, tmp_path, monkeypatch, source, hook):
    """A dry prepare's row run past a recycle is red unless every test its first process selected was reported
    by one of its processes, its last by its own report: a process that crashed after its report is never run
    past, an earlier process's cases are no evidence of its last's, and a later collection that drops or no
    longer defines a test left never stands for it."""
    assert recycled_row(ci, tmp_path, monkeypatch, source, hook)["exitCode"] != 0


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
