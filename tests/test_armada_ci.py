"""Dew's CI plan and task on armada (tools/armada/ci.py), on small test files written here."""

import importlib.util
import json
import re
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
    # CI's own environment names the shared test cache; these runs of the tools must not read or write it.
    for name in ("DEW_XLA_CACHE_URL", "DEW_XLA_CACHE_TOKEN", "DEW_TEST_CACHE"):
        monkeypatch.delenv(name, raising=False)
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
    assert rows["3.12:tests/test_pass.py"]["timings"].keys() == {"3.12:tests/test_pass.py"}
    assert rows["3.12:tests/test_pass.py"]["tests"].keys() == {"tests/test_pass.py::test_ok",
                                                               "tests/test_pass.py::test_skip"}
    assert ci.node_of(ET.Element("testcase", classname="tests.test_split.TestB", name="test_b")) == \
        "tests/test_split.py::TestB::test_b"
    # A process that dies after its report is written leaves every row it ran red, green cases included.
    (tmp_path / "tests/test_dies.py").write_text("import atexit, os\natexit.register(lambda: os._exit(3))\n"
                                                 "def test_ok(): pass\n")
    ci.task("3.12", ["tests/test_pass.py", "tests/test_dies.py"], "", out, 600.0)
    assert {row["name"]: row["exitCode"] for row in json.loads(out.read_text())["rows"]} == {
        "3.12:tests/test_pass.py": 1, "3.12:tests/test_dies.py": 1}


def test_the_gpu_tier_is_every_file_whose_tests_read_the_backend_or_are_marked_for_a_gpu(
        ci, tmp_path, monkeypatch, cache_worker):
    """A task keeps, beside each green file's key, where its run read the
    backend and its GPU marks (executed.py), and the GPU tier is each file
    with either, each whose inputs changed since, and each without a record
    the Worker returns. A read is the file's whose test calls it, by any
    name, every file's of a task where a session fixture made it, and a
    test module's under whose body it ran, though another file imported it
    first, beside others. The `sys` and `platform` modules are no read. A
    file that started a process has no key, so it is in, and no task reuses
    its row."""
    url, stored, _ = cache_worker
    monkeypatch.setenv("DEW_XLA_CACHE_URL", url)
    monkeypatch.setenv("DEW_XLA_CACHE_TOKEN", "token")
    device = "class Device:\n    platform = 'gpu'\n"
    files = {
        "devices.py": device + "def placed(device): return device.platform\n",
        "conftest.py": "import pytest\nfrom devices import Device, placed\n"
                       "@pytest.fixture(scope='session')\ndef where(): return placed(Device())\n",
        "test_routes.py": device + "def chosen(device): return device.platform == 'gpu'\n"
                                   "def test_routes(): assert chosen(Device())\n",
        "test_passed.py": "def chosen(default_backend): return default_backend()\n"
                          "def test_passed(): assert chosen(lambda: 'gpu') == 'gpu'\n",
        "test_one.py": "def test_one(where): assert where == 'gpu'\n",
        "test_two.py": "def test_two(request): assert request.getfixturevalue('where') == 'gpu'\n",
        "test_importer.py": "import test_imported, test_imported_too\n"
                            "def test_imports(): assert test_imported.ON == test_imported_too.ON == 'gpu'\n",
        "test_imported.py": "from devices import Device, placed\nON = placed(Device())\n"
                            "def test_on(): assert ON == 'gpu'\n",
        "test_imported_too.py": "from devices import Device, placed\nON = placed(Device())\n"
                                "def test_on(): assert ON == 'gpu'\n",
        "test_marked.py": "import pytest\n@pytest.mark.gpu\ndef test_on_a_gpu(): pass\n",
        "test_machine.py": "import platform, sys\n"
                           "def test_machine(): assert platform.python_version() and sys.platform\n",
        "plain.py": "def helper(): return 1\n",
        "test_plain.py": "def test_plain():\n    from plain import helper\n    assert helper() == 1\n",
        "test_lost.py": "def test_lost(): pass\n",
        "test_spawns.py": "import subprocess, sys\n"
                          "def test_child(): subprocess.run([sys.executable, '-c', 'pass'], check=True)\n",
        "test_new.py": "def test_unrun(): pass\n",
    }
    for name, source in files.items():
        (tmp_path / "tests" / name).write_text(source)
    venv = tmp_path / ".venv-3.12/bin"
    venv.mkdir(parents=True)
    (venv / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (venv / "python").chmod(0o755)
    out = tmp_path / "verdict.json"
    # The session fixture's files in a task of their own, where it reads for every file.
    fixtures = ["tests/test_one.py", "tests/test_two.py"]
    ran = [f"tests/{name}" for name in files if name.startswith("test_") and name != "test_new.py"]
    for tests in ([name for name in ran if name not in fixtures], fixtures):
        ci.task("3.12", tests, "", out, 600.0)
        assert {row["name"]: row["exitCode"] for row in json.loads(out.read_text())["rows"]} == {
            f"3.12:{name}": 0 for name in tests}
    (tmp_path / "tests/plain.py").write_text("def helper(): return 2 - 1\n")
    # A group of test_lost.py's the Worker lists and does not return.
    stored[ci.cache_url("3.12", "3.12:tests/test_lost.py#2/2", ci.cpu_class())[len(url):]] = None
    assert ci.gpu_tier("3.12") == {
        "tests/test_imported.py": ["tests/devices.py:placed"],
        "tests/test_imported_too.py": ["tests/devices.py:placed"],
        "tests/test_importer.py": ["tests/devices.py:placed"],
        "tests/test_lost.py": ["unrecorded"],
        "tests/test_marked.py": ["mark:gpu"],
        "tests/test_new.py": ["unrecorded"],
        "tests/test_one.py": ["tests/devices.py:placed"],
        "tests/test_passed.py": ["tests/test_passed.py:chosen"],
        "tests/test_plain.py": ["changed"],
        "tests/test_routes.py": ["tests/test_routes.py:chosen"],
        "tests/test_spawns.py": ["changed"],
        "tests/test_two.py": ["tests/devices.py:placed"]}
    common = ci.shared("3.12")
    assert [ci.reused("3.12", f"tests/{name}", "", common) is not None
            for name in ("test_machine.py", "test_spawns.py")] == [True, False]


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
    """`durations` records each test of the newest verdict at its least time
    over green verdicts' floor rows, so a slow container or run moves nothing
    and a test since removed is dropped, and refuses a red run or rows that
    carry no times."""
    def verdict(x, y):
        return {"rows": [
            {"name": "3.12:tests/test_a.py#1/2", "exitCode": 0, "tests": {"tests/test_a.py::test_x": x}},
            {"name": "3.12:tests/test_a.py#2/2", "exitCode": 0, "tests": {"tests/test_a.py::test_y": y}},
            {"name": "3.14:tests/test_a.py", "exitCode": 0, "tests": {"tests/test_a.py::test_x": 90.0}}]}

    assert ci.durations([verdict(1.2346, 2.0)]) == {
        "tests/test_a.py::test_x": 1.235, "tests/test_a.py::test_y": 2.0}
    assert ci.durations([verdict(1.5, 2.0), verdict(30.0, 60.0), verdict(1.0, 2.5)]) == {
        "tests/test_a.py::test_x": 1.0, "tests/test_a.py::test_y": 2.0}
    removed = {"rows": [{"name": "3.12:tests/test_a.py", "exitCode": 0,
                         "tests": {"tests/test_a.py::test_x": 0.5, "tests/test_a.py::test_gone": 3.0}}]}
    assert ci.durations([verdict(1.0, 2.0), removed]) == {
        "tests/test_a.py::test_x": 0.5, "tests/test_a.py::test_y": 2.0}
    red = {"name": "3.12:tests/test_b.py", "exitCode": 1, "tests": {}}
    with pytest.raises(SystemExit):
        ci.durations([verdict(1.0, 2.0), {"rows": [*verdict(1.0, 2.0)["rows"], red]}])
    with pytest.raises(SystemExit):
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


def test_a_task_reuses_a_green_row_only_on_the_cpu_class_and_inputs_it_ran_on(ci, tmp_path, monkeypatch):
    """A file that ran green is reused by a later task on the same CPU class while its inputs are
    unchanged; a task on another class, after its inputs change, or planned with DEW_TEST_CACHE=off
    (the daily full run) runs it again. Each task runs as armada starts it: .armada.json's command
    filled from its plan entry."""
    (tmp_path / "tests/test_counted.py").write_text(
        "from pathlib import Path\n"
        "def test_ok():\n"
        "    with Path('runs').open('a') as runs: runs.write('x')\n")
    venv = tmp_path / ".venv-3.12/bin"
    venv.mkdir(parents=True)
    (venv / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (venv / "python").chmod(0o755)
    worker: dict[str, bytes] = {}

    def cache_request(url, token, data=None):
        if data is not None:
            worker[url] = data
        return worker.get(url) if data is None else None

    monkeypatch.setattr(ci, "cache_request", cache_request)
    monkeypatch.setenv("DEW_XLA_CACHE_URL", "https://cache.invalid")
    monkeypatch.setenv("DEW_XLA_CACHE_TOKEN", "token")
    out = tmp_path / "verdict.json"
    command = json.loads((ROOT / ".armada.json").read_text())["task"]["command"]

    def run(cpu, policy="on"):
        monkeypatch.setattr(ci, "cpu_class", lambda: cpu)
        monkeypatch.setenv("DEW_TEST_CACHE", policy)
        [entry] = [entry for entry in ci.plan(300.0, {}) if entry["python"] == "3.12"]
        values = {**entry, "out": out}
        filled = [re.sub(r"\{(\w+)\}", lambda field: str(values[field[1]]), word) for word in command]
        monkeypatch.setattr(sys, "argv", ["ci.py", *filled[2:]])
        # A task's environment is .armada.json's, which the plan's DEW_TEST_CACHE is not part of.
        monkeypatch.delenv("DEW_TEST_CACHE")
        assert ci.main() == 0
        [only] = json.loads(out.read_text())["rows"]
        assert only["exitCode"] == 0
        return len((tmp_path / "runs").read_text())

    assert run("GenuineIntel-6-143") == 1
    assert run("GenuineIntel-6-143") == 1
    assert run("AuthenticAMD-25-1") == 2
    assert run("AuthenticAMD-25-1") == 2
    assert run("AuthenticAMD-25-1", policy="off") == 3
    (tmp_path / "tests/test_counted.py").write_text(
        (tmp_path / "tests/test_counted.py").read_text() + "def test_more(): pass\n")
    assert run("GenuineIntel-6-143") == 4


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
