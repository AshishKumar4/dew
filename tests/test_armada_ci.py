"""Dew's CI plan and task on armada (tools/armada/ci.py), on small test files written here."""

import importlib.util
import json
import sys
import time
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


def test_the_plan_packs_every_file_once_for_each_python(ci, tmp_path):
    """Files are packed near the target by armada's medians, else the
    recorded durations; a file nothing has timed runs alone; a file heavier
    than the target splits into groups of itself; every row is named once,
    test_gen_api never."""
    for name in ("test_heavy", "test_light", "test_new", "test_measured", "test_gen_api"):
        (tmp_path / "tests" / f"{name}.py").write_text("")
    (tmp_path / "tests/test_durations.json").write_text(json.dumps(
        {"tests/test_heavy.py::test_a": 150.0, "tests/test_heavy.py::test_b": 100.0,
         "tests/test_light.py::test_c": 30.0, "tests/test_measured.py::test_d": 500.0}))
    entries = ci.plan(120.0, {"files": {"tests/test_measured.py": 40.0, "tests/test_light.py": 30.0}})
    for python in ci.PYTHONS:
        own = [(entry["tests"], entry["split"]) for entry in entries if entry["python"] == python]
        assert sorted(own) == [("tests/test_heavy.py", "1/3"), ("tests/test_heavy.py", "2/3"),
                               ("tests/test_heavy.py", "3/3"),
                               ("tests/test_light.py tests/test_measured.py", ""), ("tests/test_new.py", "")]
    rows = [name for entry in entries for name in entry["rows"]]
    assert len(rows) == len(set(rows)) == 12
    assert "3.14:tests/test_heavy.py#2/3" in rows


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
    assert rows["3.12:tests/test_pass.py"]["timings"].keys() == {"tests/test_pass.py"}


def test_the_plan_weighs_a_split_file_by_its_groups_and_the_rest_by_armadas_pace(ci):
    """A file armada last ran in a complete set of groups weighs their sum;
    a file only GitHub's durations record weighs them scaled by how much
    slower armada ran the files measured both ways."""
    Path("tests/test_durations.json").write_text(json.dumps(
        {"tests/test_split.py::test_a": 10.0, "tests/test_paced.py::test_b": 20.0,
         "tests/test_measured.py::test_c": 10.0, "tests/test_other.py::test_d": 5.0}))
    timings = {"files": {"tests/test_measured.py": 30.0, "tests/test_other.py": 15.0},
               "rows": {"3.12:tests/test_split.py#1/2": 200.0, "3.12:tests/test_split.py#2/2": 150.0,
                        "3.12:tests/test_split.py#1/3": 90.0}}
    files = ["tests/test_measured.py", "tests/test_other.py", "tests/test_paced.py", "tests/test_split.py"]
    assert ci.weights(files, timings, "3.12") == {"tests/test_measured.py": 30.0, "tests/test_other.py": 15.0,
                                                  "tests/test_paced.py": 60.0, "tests/test_split.py": 350.0}
    assert ci.weights(files, timings, "3.14")["tests/test_split.py"] == 30.0


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
