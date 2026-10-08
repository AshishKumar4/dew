"""Dew's CI plan and task on armada (tools/armada/ci.py), on small test files written here."""

import importlib.util
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def ci(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("armada_ci", ROOT / "tools/armada/ci.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tests").mkdir()
    return module


def test_the_plan_runs_every_file_on_the_floor_and_the_newest_python_on_its_own(ci, tmp_path, monkeypatch):
    """On 3.12 files are packed near the target by armada's medians, else
    the recorded durations; a file nothing has timed runs alone; a file
    heavier than the target splits into groups of itself. 3.14 runs its own
    files (NEWEST_FILES) the same way and imports every other one in a few
    collect-only tasks. Every row is named once, test_gen_api never."""
    for name in ("test_heavy", "test_light", "test_new", "test_measured", "test_gen_api", "test_config"):
        (tmp_path / "tests" / f"{name}.py").write_text("")
    (tmp_path / "tests/test_durations.json").write_text(json.dumps(
        {"tests/test_heavy.py::test_a": 150.0, "tests/test_heavy.py::test_b": 100.0,
         "tests/test_light.py::test_c": 30.0, "tests/test_measured.py::test_d": 500.0,
         "tests/test_config.py::test_e": 20.0}))
    monkeypatch.setattr(ci, "NEWEST_FILES", ("tests/test_config.py", "tests/test_light.py"))
    entries = ci.plan(120.0, {"files": {"3.12:tests/test_measured.py": 40.0,
                                        "3.12:tests/test_light.py": 30.0}})
    floor = [(entry["tests"], entry["split"]) for entry in entries if entry["python"] == "3.12"]
    assert sorted(floor) == [("tests/test_config.py tests/test_light.py tests/test_measured.py", ""),
                             ("tests/test_heavy.py", "1/2"), ("tests/test_heavy.py", "2/2"),
                             ("tests/test_new.py", "")]
    newest = [entry for entry in entries if entry["python"] == "3.14"]
    assert [(entry["tests"], entry["split"]) for entry in newest if entry["split"] != ci.COLLECT] == [
        ("tests/test_config.py tests/test_light.py", "")]
    collected = sorted(name for entry in newest if entry["split"] == ci.COLLECT
                       for name in entry["tests"].split())
    assert collected == ["tests/test_heavy.py", "tests/test_measured.py", "tests/test_new.py"]
    rows = [name for entry in entries for name in entry["rows"]]
    assert len(rows) == len(set(rows)) == 11
    assert "3.14:tests/test_heavy.py#collect" in rows


def test_a_collect_task_rows_each_file_red_where_it_would_not_import(ci, tmp_path):
    """The newest Python's collect-only task turns a file's row red where
    the file does not import or skips on a missing import, and keeps one
    that imports green, without running any test."""
    files = {
        "test_fine.py": "def test_never_run(): raise SystemExit('ran')\n",
        "test_broken.py": "import no_such_module_anywhere\n",
        "test_missing.py": "import pytest\nmissing = pytest.importorskip('no_such_module_anywhere')\n"
                           "def test_never(): pass\n",
    }
    for name, source in files.items():
        (tmp_path / "tests" / name).write_text(source)
    venv = tmp_path / ".venv-3.14/bin"
    venv.mkdir(parents=True)
    (venv / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (venv / "python").chmod(0o755)
    out = tmp_path / "verdict.json"
    ci.collect("3.14", [f"tests/{name}" for name in files], out, 120.0)
    rows = {row["name"]: row for row in json.loads(out.read_text())["rows"]}
    assert {name: row["exitCode"] for name, row in rows.items()} == {
        "3.14:tests/test_fine.py#collect": 0, "3.14:tests/test_broken.py#collect": 1,
        "3.14:tests/test_missing.py#collect": 1}
    assert "no_such_module_anywhere" in rows["3.14:tests/test_broken.py#collect"]["output"]
    assert "could not import" in rows["3.14:tests/test_missing.py#collect"]["output"]


def test_a_task_rows_each_file_red_where_ci_would_fail(ci, tmp_path):
    """A failed test, a skip on a missing import and a file that does not
    collect each turn their own file's row red; a passing file and an
    ordinary skip stay green, timed."""
    files = {
        "test_pass.py": "import pytest\ndef test_ok(): pass\n"
                        "def test_skip(): pytest.skip('a GPU test on a CPU run')\n",
        "test_fail.py": "def test_ok(): pass\ndef test_wrong(): assert 2 == 3\n",
        "test_missing.py": "import pytest\nmissing = pytest.importorskip('no_such_module_anywhere')\n"
                           "def test_never(): pass\n",
        "test_broken.py": "import no_such_module_anywhere\n",
    }
    for name, source in files.items():
        (tmp_path / "tests" / name).write_text(source)
    venv = tmp_path / ".venv-3.12/bin"
    venv.mkdir(parents=True)
    (venv / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (venv / "python").chmod(0o755)
    out = tmp_path / "verdict.json"
    ci.task("3.12", [f"tests/{name}" for name in files], "", out, 600.0)
    rows = {row["name"]: row for row in json.loads(out.read_text())["rows"]}
    assert {name: row["exitCode"] for name, row in rows.items()} == {
        "3.12:tests/test_pass.py": 0, "3.12:tests/test_fail.py": 1, "3.12:tests/test_missing.py": 1,
        "3.12:tests/test_broken.py": 1}
    assert "test_wrong" in rows["3.12:tests/test_fail.py"]["output"]
    assert "SKIPPED on a missing import" in rows["3.12:tests/test_missing.py"]["output"]
    assert rows["3.12:tests/test_pass.py"]["timings"].keys() == {"3.12:tests/test_pass.py"}
    assert rows["3.12:tests/test_pass.py"]["tests"].keys() == {"tests/test_pass.py::test_ok",
                                                               "tests/test_pass.py::test_skip"}
    assert ci.node_of(ET.Element("testcase", classname="tests.test_split.TestB", name="test_b")) == \
        "tests/test_split.py::TestB::test_b"


def test_the_plan_weighs_a_file_by_armadas_median_and_the_rest_by_armadas_pace(ci):
    """A file armada timed weighs its median; a file only GitHub's durations
    record weighs them scaled by how much slower armada ran the files
    measured both ways."""
    Path("tests/test_durations.json").write_text(json.dumps(
        {"tests/test_paced.py::test_b": 20.0, "tests/test_measured.py::test_c": 10.0,
         "tests/test_other.py::test_d": 5.0}))
    timings = {"files": {"3.12:tests/test_measured.py": 30.0, "3.12:tests/test_other.py": 15.0,
                         "3.14:tests/test_measured.py": 45.0}}
    files = ["tests/test_measured.py", "tests/test_other.py", "tests/test_paced.py"]
    assert ci.weights(files, timings, "3.14") == {"tests/test_measured.py": 45.0, "tests/test_other.py": 22.5,
                                                  "tests/test_paced.py": 90.0}
    assert ci.weights(files, timings) == {"tests/test_measured.py": 30.0, "tests/test_other.py": 15.0,
                                          "tests/test_paced.py": 60.0}


def test_a_files_groups_are_consecutive_runs_as_light_as_their_count_lets_them_be(ci):
    """Every test lands in exactly one run, the runs keep the tests' order,
    and the heaviest is the lightest any cut into that many runs has: 5 1 1 1
    5 1 in three runs is 5 1 | 1 1 | 5 1. A test heavier than the rest runs
    alone and leaves the runs beside it no heavier than they must be. More
    runs than tests leaves the extra runs empty."""
    seconds = [5.0, 1.0, 1.0, 1.0, 5.0, 1.0]
    runs = ci.groups(seconds, 3)
    assert [index for run in runs for index in run] == list(range(6))
    assert max(sum(seconds[index] for index in run) for run in runs) == 6.0
    assert ci.groups([1.0, 1.0, 9.0, 1.0, 1.0, 1.0, 1.0], 3) == [range(2), range(2, 3), range(3, 7)]
    assert ci.groups([2.0, 3.0], 4) == [range(1), range(1, 2), range(2, 2), range(2, 2)]


def test_a_heavy_file_runs_as_the_fewest_groups_that_hold_each_to_the_target(ci):
    """A file's groups hold its recorded tests' time scaled to its weight, as
    many as it takes to keep each under the target, a heavier test alone,
    beside which the rest still split evenly. A file with no recorded tests
    runs as equal counts of them."""
    assert ci.split([150.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0], 420.0, 100.0) == [300.0, 60.0, 60.0]
    assert ci.split([40.0, 40.0, 40.0, 40.0], 160.0, 100.0) == [80.0, 80.0]
    assert ci.split([], 250.0, 100.0) == [250.0 / 3] * 3


def test_a_split_files_groups_run_every_test_once(ci, tmp_path):
    """Each task of a split file runs its own group (split.py) of the same
    cut, so the groups together run every test, none twice, a test the
    durations lack included. Each group times the file at what it spent,
    which armada sums over the groups."""
    (tmp_path / "tests/test_split.py").write_text(
        "import pytest\n@pytest.mark.parametrize('n', range(6))\ndef test_n(n): pass\n"
        "def test_unrecorded(): pass\n")
    Path("tests/test_durations.json").write_text(json.dumps(
        {f"tests/test_split.py::test_n[{n}]": seconds for n, seconds in enumerate([5, 1, 1, 1, 5, 1])}))
    venv = tmp_path / ".venv-3.12/bin"
    venv.mkdir(parents=True)
    (venv / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (venv / "python").chmod(0o755)
    ran = []
    for group in ("1/3", "2/3", "3/3"):
        out = tmp_path / f"verdict-{group.replace('/', '-')}.json"
        ci.task("3.12", ["tests/test_split.py"], group, out, 120.0)
        [verdict] = json.loads(out.read_text())["rows"]
        assert verdict["exitCode"] == 0, verdict["output"]
        assert verdict["timings"] == {"3.12:tests/test_split.py": verdict["seconds"]}
        ran.append(sorted(verdict["tests"]))
    assert all(ran)
    every = sorted(node for nodes in ran for node in nodes)
    assert every == sorted([f"tests/test_split.py::test_n[{n}]" for n in range(6)]
                           + ["tests/test_split.py::test_unrecorded"])


def test_durations_are_each_tests_least_time_over_green_runs_floors(ci):
    """`durations` records each test's least time over green verdicts' floor
    rows, so a slow container or run moves nothing, and refuses a red run or
    rows that carry no times."""
    def verdict(x, y):
        return {"rows": [
            {"name": "3.12:tests/test_a.py#1/2", "exitCode": 0, "tests": {"tests/test_a.py::test_x": x}},
            {"name": "3.12:tests/test_a.py#2/2", "exitCode": 0, "tests": {"tests/test_a.py::test_y": y}},
            {"name": "3.14:tests/test_a.py", "exitCode": 0, "tests": {"tests/test_a.py::test_x": 90.0}}]}

    assert ci.durations([verdict(1.2346, 2.0)]) == {
        "tests/test_a.py::test_x": 1.235, "tests/test_a.py::test_y": 2.0}
    assert ci.durations([verdict(1.5, 2.0), verdict(30.0, 60.0), verdict(1.0, 2.5)]) == {
        "tests/test_a.py::test_x": 1.0, "tests/test_a.py::test_y": 2.0}
    red = {"name": "3.12:tests/test_b.py", "exitCode": 1, "tests": {}}
    with pytest.raises(SystemExit, match="red"):
        ci.durations([verdict(1.0, 2.0), {"rows": [*verdict(1.0, 2.0)["rows"], red]}])
    with pytest.raises(SystemExit, match="no test times"):
        ci.durations([{"rows": [{"name": "3.12:tests/test_b.py", "exitCode": 0}]}])


def test_a_tasks_rows_share_its_wall_time_by_their_tests_times(ci, tmp_path):
    """The plan weighs a file by what a task spends on it, its import and
    collection included, so a task's rows add up to its wall time, split in
    proportion to their tests' own times."""
    (tmp_path / "tests").mkdir(exist_ok=True)
    (tmp_path / "tests/test_short.py").write_text("import time\ndef test_a(): time.sleep(0.3)\n")
    (tmp_path / "tests/test_long.py").write_text("import time\ndef test_b(): time.sleep(0.9)\n")
    venv = tmp_path / ".venv-3.12/bin"
    venv.mkdir(parents=True)
    (venv / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (venv / "python").chmod(0o755)
    out = tmp_path / "verdict.json"
    began = time.monotonic()
    ci.task("3.12", ["tests/test_short.py", "tests/test_long.py"], "", out, 60.0)
    wall = time.monotonic() - began
    rows = {row["name"]: row["seconds"] for row in json.loads(out.read_text())["rows"]}
    short, long = rows["3.12:tests/test_short.py"], rows["3.12:tests/test_long.py"]
    assert 1.2 < short + long <= wall
    assert long / short == pytest.approx(3.0, rel=0.25)


def test_a_task_past_its_deadline_is_interrupted_and_every_row_it_holds_is_red(ci, tmp_path):
    """A hang is graded: the task interrupts pytest at its deadline, short of
    armada's own timeout, kills its process group where the interrupt is not
    heard, and writes a red row for each of its files, the one that finished
    first included."""
    (tmp_path / "tests/test_quick.py").write_text("def test_ok(): pass\n")
    # A hang that ignores ^C, beside a child that holds pytest's output open:
    # neither may keep the task from writing its verdict.
    (tmp_path / "tests/test_hangs.py").write_text(
        "import signal, subprocess, sys, time\n"
        "def test_forever():\n"
        "    subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
        "    signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        "    time.sleep(600)\n")
    venv = tmp_path / ".venv-3.12/bin"
    venv.mkdir(parents=True)
    (venv / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (venv / "python").chmod(0o755)
    out = tmp_path / "verdict.json"
    began = time.monotonic()
    ci.task("3.12", ["tests/test_quick.py", "tests/test_hangs.py"], "", out, 3.0, grace=2.0)
    assert time.monotonic() - began < 60
    rows = {row["name"]: row for row in json.loads(out.read_text())["rows"]}
    assert {name: row["exitCode"] for name, row in rows.items()} == {
        "3.12:tests/test_quick.py": 1, "3.12:tests/test_hangs.py": 1}
    assert "ran past its deadline of 3 s" in rows["3.12:tests/test_hangs.py"]["output"]
