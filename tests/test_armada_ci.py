"""Dew's CI plan and task on armada (tools/armada/ci.py), on small test files written here."""

import itertools
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

import pytest
from tool_support import git, load

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def ci(tmp_path, monkeypatch):
    module = load("ci", "tools/armada")
    monkeypatch.chdir(tmp_path)
    # CI's own environment names the shared test cache; these runs of the tools must not read or write it.
    for name in ("DEW_XLA_CACHE_URL", "DEW_XLA_CACHE_TOKEN", "DEW_XLA_CACHE_READ_TOKEN", "DEW_TEST_CACHE"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "tests").mkdir()
    return module


def interpreter(root: Path, version: str) -> None:
    """The checkout's `.venv-<version>` python, as the CI image has it: here, this interpreter."""
    venv = root / f".venv-{version}/bin"
    venv.mkdir(parents=True)
    (venv / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (venv / "python").chmod(0o755)


def written(root: Path, files: dict[str, str]) -> None:
    for name, source in files.items():
        (root / "tests" / name).write_text(source)


def cached(monkeypatch, url: str) -> None:
    """The shared test cache at `url`, as CI's environment names it."""
    monkeypatch.setenv("DEW_XLA_CACHE_URL", url)
    monkeypatch.setenv("DEW_XLA_CACHE_TOKEN", "token")


def gpu_driver(root: Path, monkeypatch) -> None:
    """nvidia-smi as the driver answers it for an 80 GiB GPU, first on PATH."""
    tools = root / "bin"
    tools.mkdir()
    (tools / "nvidia-smi").write_text("#!/bin/sh\necho 81920\n")
    (tools / "nvidia-smi").chmod(0o755)
    monkeypatch.setenv("PATH", f"{tools}:{os.environ['PATH']}")


def armada_task(ci, monkeypatch, out: Path, policy: str = "on") -> list[dict]:
    """The 3.12 tasks the plan makes under DEW_TEST_CACHE=`policy`, each run as armada starts it:
    .armada.json's command filled from its plan entry, in .armada.json's environment, which the plan's
    DEW_TEST_CACHE is not part of. Their rows, in the plan's order."""
    command = json.loads((ROOT / ".armada.json").read_text())["task"]["command"]
    monkeypatch.setenv("DEW_TEST_CACHE", policy)
    entries = [entry for entry in ci.plan(300.0, {}) if entry["python"] == "3.12"]
    monkeypatch.delenv("DEW_TEST_CACHE")
    rows = []
    for entry in entries:
        values = {**entry, "out": out}
        filled = [re.sub(r"\{(\w+)\}", lambda field, values=values: str(values[field[1]]), word)
                  for word in command]
        monkeypatch.setattr(sys, "argv", ["ci.py", *filled[2:]])
        assert ci.main() == 0
        rows += json.loads(out.read_text())["rows"]
    return rows


def test_the_plan_runs_every_file_on_the_floor_and_the_newest_python_on_its_own(ci, tmp_path, monkeypatch):
    """On 3.12 files are packed near the target by armada's medians, else
    the recorded durations; a file nothing has timed runs alone; a file
    heavier than the target splits into groups of itself. On a pushed run
    3.14 runs its own files (NEWEST_FILES) the same way and imports every
    other one in a few collect-only tasks; the daily run (DEW_TEST_CACHE off)
    runs every file on both, with nothing only collected. Every row is named
    once, test_gen_api never."""
    for name in ("test_heavy", "test_light", "test_new", "test_measured", "test_gen_api", "test_config"):
        (tmp_path / "tests" / f"{name}.py").write_text("")
    (tmp_path / "tests/test_durations.json").write_text(json.dumps(
        {"tests/test_heavy.py::test_a": 150.0, "tests/test_heavy.py::test_b": 100.0,
         "tests/test_light.py::test_c": 30.0, "tests/test_measured.py::test_d": 25.0,
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

    monkeypatch.setenv("DEW_TEST_CACHE", "off")
    daily = ci.plan(120.0, {"files": {"3.12:tests/test_measured.py": 40.0, "3.12:tests/test_light.py": 30.0}})
    for python in ("3.12", "3.14"):
        ran = [(entry["tests"], entry["split"]) for entry in daily if entry["python"] == python]
        assert {name for tests, _ in ran for name in tests.split()} == {
            "tests/test_config.py", "tests/test_light.py", "tests/test_measured.py", "tests/test_heavy.py",
            "tests/test_new.py"}
    assert all(entry["split"] != ci.COLLECT and entry["cache"] == "off" for entry in daily)


def test_a_python_no_run_has_timed_weighs_a_file_by_another_pythons_time(ci, tmp_path, monkeypatch):
    """A file 3.14 has no measured time for (3.14 runs every file only on the daily run) weighs, and
    splits, by 3.12's measured time of it, not by its tests' least recorded times, so armada's
    longest-first dispatch does not start it last (train 143: 3.14 tasks of 500-800 s, weighed near
    the target, started 550-760 s into the run)."""
    (tmp_path / "tests/test_heavy.py").write_text("")
    (tmp_path / "tests/test_durations.json").write_text(json.dumps(
        {"tests/test_heavy.py::test_a": 150.0, "tests/test_heavy.py::test_b": 100.0}))
    monkeypatch.setenv("DEW_TEST_CACHE", "off")
    entries = ci.plan(240.0, {"files": {"3.12:tests/test_heavy.py": 800.0}})
    weighed = {python: sum(entry["weight"] for entry in entries if entry["python"] == python)
               for python in ("3.12", "3.14")}
    assert weighed["3.14"] == weighed["3.12"] >= 800.0, weighed
    cuts = {python: sorted(entry["split"] for entry in entries if entry["python"] == python)
            for python in ("3.12", "3.14")}
    assert cuts["3.14"] == cuts["3.12"] and len(cuts["3.12"]) > 1, cuts


def test_a_collect_task_rows_each_file_red_where_it_would_not_import(ci, tmp_path):
    """The newest Python's collect-only task turns a file's row red where
    the file does not import or skips on a missing import, and keeps one
    that imports green, without running any test; a red row says it is a
    collection's, failing no test of its own."""
    files = {
        "test_fine.py": "def test_never_run(): raise SystemExit('ran')\n",
        "test_broken.py": "import no_such_module_anywhere\n",
        "test_missing.py": "import pytest\nmissing = pytest.importorskip('no_such_module_anywhere')\n"
                           "def test_never(): pass\n",
    }
    written(tmp_path, files)
    interpreter(tmp_path, "3.14")
    out = tmp_path / "verdict.json"
    ci.collect("3.14", [f"tests/{name}" for name in files], out, 120.0)
    rows = {row["name"]: row for row in json.loads(out.read_text())["rows"]}
    assert {name: row["exitCode"] for name, row in rows.items()} == {
        "3.14:tests/test_fine.py#collect": 0, "3.14:tests/test_broken.py#collect": 1,
        "3.14:tests/test_missing.py#collect": 1}
    assert {name: (row["kind"], row["failed"]) for name, row in rows.items() if row["exitCode"]} == {
        "3.14:tests/test_broken.py#collect": ("collect", []),
        "3.14:tests/test_missing.py#collect": ("collect", [])}


def test_a_targeted_pytest_selects_the_tests_a_task_would_and_its_own_arguments_override(ci, tmp_path):
    """`ci.py pytest`, which a lane's targeted run uses (tmp/run_tests.py), runs the tests a CI task would,
    with the task's own marker selection: a network test is deselected; a `-m` of the caller's own, after
    the selection, chooses instead. Its exit status is pytest's."""
    written(tmp_path, {"test_marked.py": "import pytest\n\ndef test_offline():\n    pass\n\n\n"
                                         "@pytest.mark.network\ndef test_online():\n    pass\n"})
    interpreter(tmp_path, "3.12")

    def ran(*arguments: str) -> tuple[int, list[str]]:
        report = tmp_path / "report.xml"
        status = ci.targeted("3.12", ["tests/test_marked.py", *arguments], report)
        return status, [case.get("name") for case in ET.parse(report).iter("testcase")]

    assert ran() == (0, ["test_offline"])
    assert ran("-m", "network") == (0, ["test_online"])
    assert ran("-k", "nothing_matches")[0] == 5


def test_a_task_rows_each_file_red_where_ci_would_fail(ci, tmp_path):
    """A failed test, a skip on a missing import and a file that does not
    collect each turn their own file's row red; a passing file and an
    ordinary skip stay green, timed. A red row names the tests that failed
    where they alone turned it red, and the whole file otherwise."""
    files = {
        "test_pass.py": "import pytest\ndef test_ok(): pass\n"
                        "def test_skip(): pytest.skip('a GPU test on a CPU run')\n",
        "test_fail.py": "def test_ok(): pass\ndef test_wrong(): assert 2 == 3\n",
        "test_missing.py": "import pytest\nmissing = pytest.importorskip('no_such_module_anywhere')\n"
                           "def test_never(): pass\n",
        "test_broken.py": "import no_such_module_anywhere\n",
    }
    written(tmp_path, files)
    interpreter(tmp_path, "3.12")
    out = tmp_path / "verdict.json"
    ci.task("3.12", [f"tests/{name}" for name in files], "", out, 600.0)
    rows = {row["name"]: row for row in json.loads(out.read_text())["rows"]}
    assert {name: row["exitCode"] for name, row in rows.items()} == {
        "3.12:tests/test_pass.py": 0, "3.12:tests/test_fail.py": 1, "3.12:tests/test_missing.py": 1,
        "3.12:tests/test_broken.py": 1}
    assert {name: (row["kind"], row["failed"]) for name, row in rows.items() if row["exitCode"]} == {
        "3.12:tests/test_fail.py": ("nodes", ["tests/test_fail.py::test_wrong"]),
        "3.12:tests/test_missing.py": ("whole", []), "3.12:tests/test_broken.py": ("whole", [])}
    assert rows["3.12:tests/test_pass.py"]["timings"].keys() == {"3.12:tests/test_pass.py"}
    assert rows["3.12:tests/test_pass.py"]["tests"].keys() == {"tests/test_pass.py::test_ok",
                                                               "tests/test_pass.py::test_skip"}
    assert ci.node_of(ET.Element("testcase", classname="tests.test_split.TestB", name="test_b")) == \
        "tests/test_split.py::TestB::test_b"
    # A process that dies after its report is written leaves every row it ran red, green cases included.
    (tmp_path / "tests/test_dies.py").write_text("import atexit, os\natexit.register(lambda: os._exit(3))\n"
                                                 "def test_ok(): pass\n")
    ci.task("3.12", ["tests/test_pass.py", "tests/test_dies.py"], "", out, 600.0)
    assert {row["name"]: (row["exitCode"], row["kind"]) for row in json.loads(out.read_text())["rows"]} == {
        "3.12:tests/test_pass.py": (1, "whole"), "3.12:tests/test_dies.py": (1, "whole")}


def test_a_tasks_tests_run_with_no_timed_stack_dump_armed(ci, tmp_path):
    """A task's pytest arms no timed dump of every thread's stack, which killed a multithreaded test mid-dump
    (exit 245, a SIGSEGV, Main 2026-10-25): the test process CI starts carries no dump timer. A test running
    past the old ten minutes through a task is green with no dump (armada receipt in the head's review)."""
    (tmp_path / "tests/test_timer.py").write_text(
        "def test_no_dump_timer(request):\n"
        "    assert request.config.getini('faulthandler_timeout') == 0\n")
    interpreter(tmp_path, "3.12")
    out = tmp_path / "verdict.json"
    ci.task("3.12", ["tests/test_timer.py"], "", out, 600.0)
    [row] = json.loads(out.read_text())["rows"]
    assert row["exitCode"] == 0, row["output"]


def test_a_file_pytest_reports_nothing_of_is_red_unless_its_collection_shows_every_test_deselected(
        ci, tmp_path):
    """Every requested file is accounted for by its tests' rows, or by its collection: one whose every
    test the selection deselected (`-m "not network"`) is green, though another file of its task fails. A
    file whose tests vanish from the run with no word (a hook dropping them unreported) is red, and so is
    every file of a run that wrote no report at all, though pytest exits 0: CI never goes green on tests
    that did not run."""
    files = {
        "test_pass.py": "def test_ok(): pass\n",
        "test_fails.py": "def test_no(): assert False\n",
        "test_online.py": "import pytest\n@pytest.mark.network\ndef test_fetch(): pass\n",
        "test_dropped.py": "def test_one(): pass\ndef test_two(): pass\n",
    }
    written(tmp_path, files)
    (tmp_path / "tests" / "conftest.py").write_text(
        "def pytest_collection_modifyitems(items):\n"
        "    items[:] = [item for item in items if 'test_dropped' not in item.nodeid]\n")
    interpreter(tmp_path, "3.12")
    out = tmp_path / "verdict.json"
    ci.task("3.12", [f"tests/{name}" for name in files], "", out, 600.0)
    rows = {row["name"]: row for row in json.loads(out.read_text())["rows"]}
    assert {name: row["exitCode"] for name, row in rows.items()} == {
        "3.12:tests/test_pass.py": 0, "3.12:tests/test_fails.py": 1, "3.12:tests/test_online.py": 0,
        "3.12:tests/test_dropped.py": 1}
    assert rows["3.12:tests/test_dropped.py"].get("kind") == "whole"
    # A run that writes no report: pytest exits 0, and no file of it is green.
    (tmp_path / "tests" / "conftest.py").write_text(
        "def pytest_configure(config):\n    config.option.xmlpath = None\n")
    ci.task("3.12", ["tests/test_pass.py", "tests/test_online.py"], "", out, 600.0)
    assert {row["name"]: row["exitCode"] for row in json.loads(out.read_text())["rows"]} == {
        "3.12:tests/test_pass.py": 1, "3.12:tests/test_online.py": 1}


def install_compiler(directory: Path) -> None:
    """A kernel's compiler under `directory`, named as the CUDA toolkit installs it (ptxas)."""
    directory.mkdir()
    (directory / "ptxas").write_text("#!/bin/sh\necho 'ptxas: NVIDIA (R) Ptx assembler'\n")
    (directory / "ptxas").chmod(0o755)


def test_the_gpu_tier_is_every_file_whose_tests_read_the_backend_or_are_marked_for_a_gpu(
        ci, tmp_path, monkeypatch, cache_worker, capsys):
    """A task keeps, beside each green file's key, where its run read the
    backend and its GPU marks (executed.py), and the GPU tier is each file
    with either, each whose inputs changed since, and each without a record
    the Worker returns. A read is the file's whose test calls it, by any
    name, every file's of a task where a session fixture made it, and a
    test module's under whose body it ran, though another file imported it
    first, beside others. The `sys` and `platform` modules are no read. A
    file that started a process a record does not follow
    (test_a_python_child_joins_its_files_record), a shell's or any
    multiprocessing context's, has no key, so it is in, and no task reuses
    its row, whatever names its program (a descriptor, for `os.execve`); one
    that asked a native tool about the host or compiled a kernel with ptxas
    (`followed.TOOLS`) keeps its key. The command line
    spreads the tier over its shards, each file once, and per batch names
    the files the checkout marks for a GPU."""
    url, stored, _ = cache_worker
    cached(monkeypatch, url)
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
        "test_tool.py": "import platform, subprocess\n"
                        "def test_tool():\n"
                        "    subprocess.run(['git', '--version'], check=True, capture_output=True)\n"
                        "    subprocess.run(['tests/compiler/ptxas', '--version'], check=True)\n"
                        "    assert platform.platform()\n",
        "test_shell.py": "import os, subprocess, sys\n"
                         "def test_shell(): subprocess.run('git --version', shell=True, check=True)\n"
                         "def test_by_descriptor():\n"
                         "    child = os.fork()\n"
                         "    if child == 0:\n"
                         "        try:\n"
                         "            program = os.open(sys.executable, os.O_RDONLY)\n"
                         "            os.execve(program, ['python', '-c', 'pass'], {})\n"
                         "        finally:\n"
                         "            os._exit(1)\n"
                         "    assert os.waitstatus_to_exitcode(os.waitpid(child, 0)[1]) == 0\n",
        **{f"test_{method}.py": "import multiprocessing, os\n"
                                "def test_child():\n"
                                f"    context = multiprocessing.get_context('{method}')\n"
                                "    child = context.Process(target=os.getpid)\n"
                                "    child.start()\n    child.join()\n    assert child.exitcode == 0\n"
           for method in ("spawn", "forkserver", "fork")},
        "test_new.py": "def test_unrun(): pass\n",
    }
    written(tmp_path, files)
    install_compiler(tmp_path / "tests/compiler")
    interpreter(tmp_path, "3.12")
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
    stored["/" + ci.record_key("3.12", "3.12:tests/test_lost.py#2/2", ci.cpu_class())] = None
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
        "tests/test_fork.py": ["changed"],
        "tests/test_forkserver.py": ["changed"],
        "tests/test_shell.py": ["changed"],
        "tests/test_routes.py": ["tests/test_routes.py:chosen"],
        "tests/test_spawn.py": ["changed"],
        "tests/test_two.py": ["tests/devices.py:placed"]}
    common = ci.shared("3.12")
    assert [ci.reused("3.12", f"tests/{name}", "", common) is not None
            for name in ("test_machine.py", "test_tool.py", "test_spawn.py", "test_forkserver.py",
                         "test_fork.py", "test_shell.py")] == [True, True, False, False, False, False]
    monkeypatch.setattr(sys, "argv", ["ci.py", "gpu-tier", "--shards=2"])
    capsys.readouterr()
    assert ci.main() == 0
    shards = [line.split() for line in capsys.readouterr().out.splitlines()]
    assert len(shards) == 2 and all(shards)
    assert sorted(name for names in shards for name in names) == sorted(ci.gpu_tier("3.12"))
    # A file gains a kernel's test after its record: per batch the checkout's marks choose, not records.
    (tmp_path / "tests/test_plain.py").write_text(
        "import pytest\n@pytest.mark.kernel\ndef test_lowers(): pass\n")
    monkeypatch.setattr(sys, "argv", ["ci.py", "gpu-tier", "--marked"])
    assert ci.main() == 0
    assert capsys.readouterr().out.split() == ["tests/test_marked.py", "tests/test_plain.py"]


def test_a_python_child_joins_its_files_record(ci, tmp_path, monkeypatch, cache_worker):
    """A Python child a test file starts with its environment records itself
    (tools/armada/followed.py), once and silently though its site-packages are
    processed again: its reads of the backend join the file's evidence, and
    the files it executed and the pages it read the file's key, which holds
    until one changes. A child given an environment of its own, or killed
    before it wrote its record, leaves its file without a key."""
    url, _, _ = cache_worker
    cached(monkeypatch, url)
    started = "import os, subprocess, sys\ndef test_child():\n    "
    files = {
        "devices.py": "class Device:\n    platform = 'gpu'\ndef placed(device): return device.platform\n",
        "plain.py": "def helper(): return 1\n",
        "child.py": "import site, sysconfig\n"
                    "site.addsitedir(sysconfig.get_paths()['purelib'])\n"
                    "import devices, plain\n"
                    "assert devices.placed(devices.Device()) == 'gpu' and plain.helper()\n"
                    "assert open('notes/page.md').read()\n",
        "test_child.py": started + "done = subprocess.run([sys.executable, 'tests/child.py'], check=True,\n"
                                   "                      capture_output=True, text=True)\n"
                                   "    assert not done.stderr, done.stderr\n",
        "test_quiet.py": started + "subprocess.run([sys.executable, '-c', 'pass'], check=True)\n",
        "test_killed.py": started + "child = subprocess.Popen([sys.executable, '-c', 'input()'], stdin=-1)\n"
                                    "    child.kill()\n    child.wait()\n",
        "test_own_environment.py": started + "subprocess.run([sys.executable, '-c', 'pass'], check=True,\n"
                                             "                   env={'PATH': os.environ['PATH']})\n",
    }
    written(tmp_path, files)
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes/page.md").write_text("page\n")
    interpreter(tmp_path, "3.12")
    tests = [f"tests/{name}" for name in files if name.startswith("test_")]
    ci.task("3.12", tests, "", tmp_path / "verdict.json", 600.0)

    def reused() -> list[bool]:
        common = ci.shared("3.12")
        return [ci.reused("3.12", name, "", common) is not None for name in tests]

    assert reused() == [True, True, False, False]
    (tmp_path / "notes/page.md").write_text("page again\n")
    assert reused() == [False, True, False, False]
    (tmp_path / "notes/page.md").write_text("page\n")
    assert reused() == [True, True, False, False]
    (tmp_path / "tests/plain.py").write_text("def helper(): return 2 - 1\n")
    assert reused() == [False, True, False, False]
    assert ci.gpu_tier("3.12") == {"tests/test_child.py": ["tests/devices.py:placed"],
                                   "tests/test_killed.py": ["changed"],
                                   "tests/test_own_environment.py": ["changed"]}


def fetcher(root: Path, *, due: bool = False) -> Path:
    """A stand-in tools/fetched_fixtures.py under `root`: its fetch records that it ran, and with `due`
    refuses as the real one does a pin whose upload is due."""
    (root / "tools").mkdir(exist_ok=True)
    (root / "tools/fetched_fixtures.py").write_text(
        "import sys, time\nfrom pathlib import Path\n"
        "assert sys.argv[1:] == ['fetch']\n"
        "Path('fetched').write_text(str(time.monotonic()))\n"
        + ("raise SystemExit('x.npz: pinned with no revision: its upload is due')\n" if due else ""))
    return root / "fetched"


def test_a_gpu_lane_fetches_the_pinned_fixtures_before_any_file_and_stops_where_one_is_due(
        ci, tmp_path, monkeypatch, cache_worker):
    """The tests read pinned fixtures offline (CLOSED_HUB), so the lane fetches them first, while the Hub is
    reachable; a pin whose upload is due stops the lane before any file runs."""
    url, _, _ = cache_worker
    cached(monkeypatch, url)
    monkeypatch.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
    gpu_driver(tmp_path, monkeypatch)
    interpreter(tmp_path, "3.12")
    (tmp_path / "tests/test_reads.py").write_text(
        "import time\nfrom pathlib import Path\n"
        "def test_reads(): Path('read').write_text(str(time.monotonic()))\n")
    out = tmp_path / "verdict.json"
    fetched = fetcher(tmp_path)
    assert ci.gpu_lane("3.12", ["tests/test_reads.py"], out, cache="on", processes=1, deadline=600.0) == 0
    assert float(fetched.read_text()) < float((tmp_path / "read").read_text())
    fetched.unlink()
    (tmp_path / "read").unlink()
    fetcher(tmp_path, due=True)
    with pytest.raises(subprocess.CalledProcessError):
        ci.gpu_lane("3.12", ["tests/test_reads.py"], out, cache="on", processes=1, deadline=600.0)
    assert fetched.is_file() and not (tmp_path / "read").exists()


def test_a_network_lane_runs_the_network_marked_tests_with_the_hub_open_and_records_them_apart(
        ci, tmp_path, monkeypatch, cache_worker):
    """The network lane runs each file's network-marked tests alone, which every other task sets aside,
    with the Hub open, and records them on a class of their own: the same file's CUDA lane, Hub closed,
    runs its other tests and is no reuse of the network lane's run, and a second network lane on the same
    inputs reuses its own (Main, 2026-10-25: no CI run reached the published checkpoints' parity tests)."""
    url, _, _ = cache_worker
    gpu_lane_at(tmp_path, monkeypatch, url, 0)
    interpreter(tmp_path, "3.12")
    fetcher(tmp_path)
    (tmp_path / "tests/test_hub.py").write_text(
        "import os, pytest\nfrom pathlib import Path\n"
        "@pytest.mark.network\ndef test_reads_the_hub():\n"
        "    assert not {'HF_HUB_OFFLINE', 'HF_ENDPOINT'} & set(os.environ), 'the Hub is closed'\n"
        "    Path('hub').write_text('ran')\n"
        "def test_offline():\n"
        "    assert os.environ['HF_HUB_OFFLINE'] == '1'\n"
        "    Path('offline').write_text('ran')\n")
    out = tmp_path / "verdict.json"
    hub, offline = tmp_path / "hub", tmp_path / "offline"

    def lane(**network: bool) -> list[tuple[str, int]]:
        hub.unlink(missing_ok=True)
        offline.unlink(missing_ok=True)
        assert ci.gpu_lane("3.12", ["tests/test_hub.py"], out, cache="on", processes=1, deadline=600.0,
                           **network) == 0
        return [(row["name"], row["exitCode"]) for row in json.loads(out.read_text())["rows"]]

    assert lane(network=True) == [("3.12:tests/test_hub.py", 0)]
    assert hub.is_file() and not offline.exists()
    assert lane() == [("3.12:tests/test_hub.py", 0)]
    assert offline.is_file() and not hub.exists(), "the CUDA lane's run is its own"
    assert lane(network=True) == [("3.12:tests/test_hub.py", 0)]
    assert not hub.exists(), "the network lane reuses its own green run on the same inputs"


def test_a_gpu_lane_names_its_files_but_with_network(ci, monkeypatch):
    """`gpu-lane` without --tests is the network lane's alone, which runs every file holding a network-marked
    test: a CUDA lane names its files, and one that names none is refused, not run over the network files'
    offline tests (train 137's review)."""
    monkeypatch.setattr(sys, "argv", ["ci.py", "gpu-lane", "--out=rows.json", "--cache=on", "--processes=1"])
    with pytest.raises(SystemExit) as refused:
        ci.main()
    assert refused.value.code == 2


LANE_TEST = ("import os, subprocess, sys, time\n"
             "from pathlib import Path\n"
             "def test_runs():\n"
             "    began = time.monotonic()\n"
             "    time.sleep(1)\n"
             "    share = os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']\n"
             "    grown = os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE', 'true')\n"
             "    threads = os.environ.get('OMP_NUM_THREADS', 'all')\n"
             "    Path(f'threads-{Path(__file__).stem}').write_text(threads)\n"
             "    ran = Path(f'ran-{Path(__file__).stem}-{os.getpid()}')\n"
             "    ran.write_text(f'{share} {began} {time.monotonic()} {grown}')\n")
"""A GPU lane's test file standing in: it records the share it was given, when it ran and whether its pool
grows."""


def gpu_lane_at(root: Path, monkeypatch, url: str, files: int) -> list[str]:
    """A GPU lane's machine at `root`: the test cache at `url`, half of one 80 GiB GPU, its session growing
    every job's pool as a GPU node's does (dewops' settings), and `files` test files of `LANE_TEST`. Their
    names."""
    cached(monkeypatch, url)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    gpu_driver(root, monkeypatch)
    names = [f"tests/test_file_{index}.py" for index in range(files)]
    for name in names:
        (root / name).write_text(LANE_TEST)
    return names


def lane_runs(root: Path, stem: str) -> list[list[str]]:
    """Each run of `stem`'s `LANE_TEST`, oldest first: its share, start, end and growth."""
    return [path.read_text().split() for path in sorted(root.glob(f"ran-{stem}-*"),
                                                        key=lambda path: path.stat().st_mtime)]


def overlapping(runs: list[list[str]]) -> int:
    """The most of `runs` (`lane_runs`) that ran at one moment."""
    spans = [(float(began), float(ended)) for _, began, ended, _ in runs]
    return max(sum(began <= moment < ended for began, ended in spans) for moment, _ in spans)


def test_a_gpu_lane_runs_each_file_in_a_process_of_its_own_within_its_memory_share(
        ci, tmp_path, monkeypatch, cache_worker):
    """The GPU lane runs each file once, in a process of its own. A file no
    run has measured runs alone with the lane's whole share, and so does one
    that started a process it did not follow, its pool growing as it uses it
    so that child has the rest of the share; files with a measured peak run
    side by side, each with its peak and headroom, and one whose child took
    its share's fraction holds a share more beside them (`gpu_needs`). A red
    row stays red: nothing is run again. The rows join one verdict, each
    recorded as a task's are, and a later lane reuses the green ones."""
    url, stored, _ = cache_worker
    names, source = gpu_lane_at(tmp_path, monkeypatch, url, 5), LANE_TEST
    # One that fails on any share, and one that starts a process.
    (tmp_path / "tests/test_red.py").write_text(source + "    assert share == 'all of it'\n")
    (tmp_path / "tests/test_spawner.py").write_text(
        source + "    subprocess.run([sys.executable, '-c', 'pass'], check=True)\n")
    every = [*names, "tests/test_red.py", "tests/test_spawner.py"]
    interpreter(tmp_path, "3.12")
    fetcher(tmp_path)

    def ran(stem: str) -> list[list[str]]:
        return lane_runs(tmp_path, stem)

    out = tmp_path / "verdict.json"
    assert ci.gpu_lane("3.12", every, out, cache="on", processes=4, deadline=600.0) == 0
    rows = json.loads(out.read_text())["rows"]
    assert [(row["name"], row["exitCode"]) for row in rows] == [
        (f"3.12:{name}", int(name == "tests/test_red.py")) for name in every]
    first = [run for name in every for run in ran(Path(name).stem)]
    assert [share for share, *_ in first] == ["0.5"] * len(every), "unmeasured: each alone, the whole share"
    # Each file's torch and OpenMP take its share of this process's CPUs, as four files may run at once.
    threads = str(max(1, len(os.sched_getaffinity(0)) // 4))
    assert {(tmp_path / f"threads-{Path(name).stem}").read_text() for name in every} == {threads}
    # Alone with the whole share, a file grows its pool as it uses it: a child it did not follow, another
    # environment's Python, has what the file leaves of the share.
    assert [grown for *_, grown in first] == ["false"] * len(every)
    assert overlapping(first) == 1
    assert len(ran("test_red")) == 1, "a red row stays red: its file is not run again"
    # Measured peaks of 4 GiB: each file then holds 6 GiB of the 40, and four run at once. Records without
    # a key, as a run on these inputs leaves to be run again, under the last run's and these inputs'.
    # The spawner's child took its fraction, so it holds two shares: 12 GiB beside the others' 24.
    common = ci.lane_inputs("3.12")
    for name, inputs in itertools.product([*names, "tests/test_spawner.py"], (None, common)):
        at = "/" + ci.record_key("3.12", ci.row("3.12", name, ""), ci.device_class(), inputs)
        stored[at] = json.dumps({"key": None, "depends": [name], "row": {}, "reads": [], "peak": 4 * 2 ** 30,
                                 "shares": int(name == "tests/test_spawner.py"), "pools": 0}).encode()
    assert ci.gpu_lane("3.12", every, out, cache="on", processes=4, deadline=600.0) == 0
    rows = json.loads(out.read_text())["rows"]
    assert [row["exitCode"] for row in rows] == [int(name == "tests/test_red.py") for name in every]
    measured = [ran(Path(name).stem)[-1] for name in names]
    assert [share for share, *_ in measured] == [str(6 * 2 ** 30 / (81920 * 2 ** 20))] * len(names)
    assert [grown for *_, grown in measured] == ["true"] * len(names), "a measured share is taken whole"
    assert 2 <= overlapping(measured) <= 4
    assert [share for share, *_ in ran("test_spawner")] == ["0.5", measured[0][0]]
    assert overlapping([*measured, ran("test_spawner")[-1]]) >= 2, "a followed spawner packs with the rest"
    gib, budget = 2 ** 30, 40 * 2 ** 30
    assert ci.gpu_needs({"depends": None, "peak": 4 * gib}, budget) == (budget, budget)
    held = {"depends": [], "peak": 4 * gib, "shares": 2, "pools": gib}
    assert ci.gpu_needs(held, budget) == (6 * gib, 19 * gib), "6 GiB, twice more for its children, and 1 GiB"
    assert ci.gpu_lane("3.12", [], out, cache="on", processes=4, deadline=600.0) == 0
    assert json.loads(out.read_text()) == {"rows": []}, "an empty lane runs nothing"
    assert len(ran("test_red")) == 2, "a red row is never reused"
    assert ci.gpu_lane("3.12", names, out, cache="on", processes=4, deadline=600.0) == 0
    assert all(len(ran(Path(name).stem)) == 2 for name in names), "the third lane reused the green rows"
    # Their records, of these inputs as the lane keys them, read no backend: none is in the GPU tier.
    assert not set(names) & set(ci.gpu_tier("3.12"))


def test_a_first_gpu_lane_on_a_class_borrows_another_classs_measured_shares(
        ci, tmp_path, monkeypatch, cache_worker):
    """A lane on a class no run has measured borrows each file's share from another GPU class's last
    measured peak (the largest any class holds), so its files run side by side as a measured lane's do,
    not each alone. A file red under a borrowed share because its memory ran out runs once more, alone,
    with the lane's whole share, and its row says so (interference) with that first failure; one whose
    first failure is its own (an assertion, whatever its message names) stays red, run once. The lane
    records this class's own measurement, so the next lane on it reads its own, not the borrowed one. A
    file no class measured runs alone, as every unmeasured one does where the Worker does not list its
    records."""
    url, stored, _ = cache_worker
    names, source = gpu_lane_at(tmp_path, monkeypatch, url, 4), LANE_TEST
    # Out of memory on any share but the whole: run again alone, its second row green, the first failure kept
    # in its row. And one whose first failure is its own (an assertion): red, run once, whatever share.
    (tmp_path / "tests/test_tight.py").write_text(
        source + "    if share != '0.5':\n"
                 "        raise RuntimeError('RESOURCE_EXHAUSTED: Out of memory allocating 9.0GiB')\n")
    (tmp_path / "tests/test_wrong.py").write_text(
        source + "    assert share == '0.5', 'an accuracy miss, not RESOURCE_EXHAUSTED: out of memory'\n")
    # A block past the GPU's shared memory is no device memory a larger share gives: red, run once. And one
    # out of memory whose borrowed measurement already reached the whole share: red, run once, as it ran.
    (tmp_path / "tests/test_shared.py").write_text(
        source + "    raise RuntimeError('RESOURCE_EXHAUSTED: Shared memory size limit exceeded: 300000')\n")
    (tmp_path / "tests/test_whole.py").write_text(
        source + "    raise RuntimeError('RESOURCE_EXHAUSTED: Out of memory allocating 90.0GiB')\n")
    (tmp_path / "tests/test_new.py").write_text(source)
    every = [*names, "tests/test_tight.py", "tests/test_wrong.py", "tests/test_shared.py",
             "tests/test_whole.py", "tests/test_new.py"]
    interpreter(tmp_path, "3.12")
    fetcher(tmp_path)
    # Another class measured 4 GiB for each file but the new one; a third, smaller peaks for some.
    for other, peak in (("NVIDIA-H100-80GB-HBM3-x86", 4), ("NVIDIA-A10-x86", 2)):
        for name in [*names, "tests/test_tight.py", "tests/test_wrong.py", "tests/test_shared.py",
                     "tests/test_whole.py"]:
            at = "/" + ci.record_key("3.12", ci.row("3.12", name, ""), other)
            whole = 64 if name == "tests/test_whole.py" else peak
            stored[at] = json.dumps({"key": None, "depends": [name], "row": {}, "reads": [],
                                     "peak": whole * 2 ** 30, "shares": 0, "pools": 0}).encode()

    def ran(stem: str) -> list[list[str]]:
        return lane_runs(tmp_path, stem)

    out = tmp_path / "verdict.json"
    assert ci.gpu_lane("3.12", every, out, cache="on", processes=4, deadline=600.0) == 0
    rows = {row["name"]: row for row in json.loads(out.read_text())["rows"]}
    borrowed = str(6 * 2 ** 30 / (81920 * 2 ** 20))
    first = [ran(Path(name).stem)[0] for name in names]
    assert [share for share, *_ in first] == [borrowed] * len(names), "the largest class's 4 GiB and headroom"
    assert overlapping(first) >= 2, "borrowed shares pack as measured ones do"
    assert [share for share, *_ in ran("test_new")] == ["0.5"], "measured on no class: alone, the whole share"
    assert [share for share, *_ in ran("test_tight")] == [borrowed, "0.5"]
    tight = rows["3.12:tests/test_tight.py"]
    assert tight["exitCode"] == 0 and "borrowed from NVIDIA-H100-80GB-HBM3-x86" in tight["interference"]
    assert "RESOURCE_EXHAUSTED" in tight["interference"]
    wrong = rows["3.12:tests/test_wrong.py"]
    assert wrong["exitCode"] == 1 and "an accuracy miss" in wrong["output"] and "interference" not in wrong
    assert [share for share, *_ in ran("test_wrong")] == [borrowed], "its own failure: red, run once"
    for stem in ("test_shared", "test_whole"):
        found = rows[f"3.12:tests/{stem}.py"]
        assert found["exitCode"] == 1 and len(ran(stem)) == 1, f"{stem}: red, run once"
        assert "interference" not in found
    assert ran("test_whole")[0][0] == "0.5", "its borrowed measurement already reached the whole share"
    assert all("interference" not in row for name, row in rows.items() if name != "3.12:tests/test_tight.py")
    # A Worker that does not list its records: nothing is borrowed, and an unmeasured file runs alone.
    (tmp_path / "tests/test_listless.py").write_text(source)
    for other in ("NVIDIA-H100-80GB-HBM3-x86",):
        at = "/" + ci.record_key("3.12", ci.row("3.12", "tests/test_listless.py", ""), other)
        stored[at] = json.dumps({"key": None, "depends": [], "row": {}, "reads": [], "peak": 2 ** 30,
                                 "shares": 0, "pools": 0}).encode()
    listing = ci.r2.Store.keys

    def refused(self, prefix):
        raise ci.r2.Unavailable("listing refused")
        yield

    monkeypatch.setattr(ci.r2.Store, "keys", refused)
    assert ci.gpu_lane("3.12", ["tests/test_listless.py"], out, cache="off", processes=4, deadline=600.0) == 0
    assert [share for share, *_ in ran("test_listless")] == ["0.5"]
    monkeypatch.setattr(ci.r2.Store, "keys", listing)
    # This class's own records now: the next lane reads them, the tight file alone as its own last run was.
    assert ci.gpu_lane("3.12", ["tests/test_tight.py"], out, cache="off", processes=4, deadline=600.0) == 0
    assert [share for share, *_ in ran("test_tight")][-1] != borrowed
    assert "interference" not in json.loads(out.read_text())["rows"][0]


def test_a_files_children_hold_what_they_held_at_once_and_each_start_must_leave_its_record(tmp_path):
    """The children's records join the file that started them (`followed.gathered`): their files,
    and the GPU they held, the most a child that took the parent's fraction used, those children as
    shares and the bytes of those that named their own, counted only while they lived at once. A
    file is followed only where every start it and its children saw left a record naming it and none
    started a process it did not follow; one whose child marked its start and left no record, killed
    before it wrote, is not."""
    followed = load("followed", "tools/armada")
    gib = 2 ** 30

    def child(index, file, began, ended, pool=None, inherited=True, expected=0, unfollowed=False, peak=None):
        (tmp_path / f"{index}.started").write_text(file)
        (tmp_path / f"{index}.json").write_text(json.dumps({
            "file": file, "files": [f"src/{index}.py"], "reads": [], "expected": expected, "peak": peak,
            "unfollowed": unfollowed, "began": began, "ended": ended, "pool": pool, "inherited": inherited}))

    child(1, "tests/a.py", 0, 10, pool=6 * gib, peak=5 * gib)
    child(2, "tests/a.py", 5, 15, pool=6 * gib, expected=1, peak=gib)
    child(3, "tests/a.py", 20, 30, pool=3 * gib, inherited=False)
    child(4, "tests/a.py", 25, 35, pool=2 * gib, inherited=False)
    child(5, "tests/a.py", 40, 41)
    child(6, "tests/b.py", 0, 1)
    child(7, "tests/c.py", 0, 1)
    (tmp_path / "8.started").write_text("tests/c.py")
    child(9, "tests/d.py", 0, 1, unfollowed=True)
    found = followed.gathered(tmp_path, {"tests/a.py": 4, "tests/b.py": 2, "tests/c.py": 2, "tests/d.py": 1})
    assert found["tests/a.py"] == {"files": {f"src/{index}.py" for index in range(1, 6)}, "reads": set(),
                                   "followed": True, "peak": 5 * gib, "shares": 2, "pools": 5 * gib}
    assert [found[name]["followed"] for name in ("tests/b.py", "tests/c.py", "tests/d.py")] == [False] * 3


def test_coverage_combines_each_tasks_branches_into_one_report_a_module_a_line(ci, tmp_path):
    """`coverage items` runs the floor's tasks with the test cache off, each writing its data among
    its artifacts; `coverage report` combines them, so a branch one task took and its other branch
    a Python process another task's test started took make the module whole, and a module no test
    reached leads the table."""
    configured = (ROOT / "pyproject.toml").read_text()
    run = configured[configured.index("[tool.coverage.run]"):]
    run = run[:run.index("\n[", 1)]
    (tmp_path / "pyproject.toml").write_text(f'[tool.pytest.ini_options]\npythonpath = ["src"]\n{run}\n')
    package = tmp_path / "src/dew"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "sign.py").write_text("def sign(x):\n    if x < 0:\n        return -1\n    return 1\n")
    (package / "unused.py").write_text("def never():\n    return 0\n")
    (tmp_path / "tests/test_negative.py").write_text("from dew.sign import sign\n"
                                                     "def test_negative(): assert sign(-2) == -1\n")
    (tmp_path / "tests/test_positive.py").write_text(
        "import os, subprocess, sys\n"
        "def test_positive():\n"
        "    subprocess.run([sys.executable, '-c', 'from dew.sign import sign; assert sign(2) == 1'],\n"
        "                   check=True, env={**os.environ, 'PYTHONPATH': 'src'})\n")
    items = ci.coverage_items(300.0, {})
    assert items and all("--python=3.12 " in item and "--cache=off" in item
                         and f"--coverage={ci.ARTIFACTS}" in item for item in items)
    assert sorted(name for item in items for name in re.search(r"--tests='([^']*)'", item)[1].split()) == [
        "tests/test_negative.py", "tests/test_positive.py"]
    interpreter(tmp_path, "3.12")
    # armada keeps each task's artifacts under its index; a directory of another name is no task.
    for index, name in enumerate(("negative", "positive")):
        (tmp_path / "runs" / str(index)).mkdir(parents=True)
        ci.task("3.12", [f"tests/test_{name}.py"], "", tmp_path / "runs" / str(index) / "rows.json", 600.0,
                cache="off", coverage=tmp_path / "runs" / str(index))
    (tmp_path / "runs/logs").mkdir()
    assert ci.coverage_report(tmp_path / "runs") == 0
    files = {Path(name).name: measured for name, measured in
             json.loads((tmp_path / "runs/coverage.json").read_text())["files"].items()}
    assert files["sign.py"]["summary"]["percent_covered"] == 100.0
    least = (tmp_path / "runs/coverage.md").read_text().splitlines()[2].split(" | ")
    assert Path(least[0].removeprefix("| ")).name == "unused.py" and least[-1] == "0.0% |"
    # A task interrupted at its deadline leaves its later tests' coverage out: the pass is refused.
    (tmp_path / "tests/test_slow.py").write_text("import time\ndef test_slow(): time.sleep(30)\n")
    (tmp_path / "runs/2").mkdir()
    ci.task("3.12", ["tests/test_slow.py"], "", tmp_path / "runs/2/rows.json", 1.0, cache="off", grace=5.0,
            coverage=tmp_path / "runs/2")
    assert ci.coverage_report(tmp_path / "runs") == 1
    [interrupted] = json.loads((tmp_path / "runs/2/rows.json").read_text())["rows"]
    assert (interrupted["kind"], interrupted["failed"]) == ("deadline", [])


def test_a_coverage_task_has_a_deadline_for_tracing(ci, tmp_path):
    """Branch tracing slows a task several times over: each coverage item runs by COVERAGE_DEADLINE,
    not a CI task's."""
    (tmp_path / "tests/test_one.py").write_text("def test_one(): pass\n")
    items = ci.coverage_items(300.0, {})
    assert items and all(f"--deadline={ci.COVERAGE_DEADLINE:.0f}" in item for item in items)


def test_a_file_whose_recorded_tests_outweigh_its_median_runs_split_by_them(ci, tmp_path):
    """Train 18's census: main's runs timed the file at 6 s with one test; a head recorded the 1,576
    tests it added (here 40 of 30 s). Each record is a test's least time, so their sum past the median
    may mean those runs predate the file's tests, and is a floor on its time: it runs split by its
    records, alone, not packed with the light files by the median it had before."""
    for name in ("test_census", "test_light", "test_other"):
        (tmp_path / "tests" / f"{name}.py").write_text("")
    (tmp_path / "tests/test_durations.json").write_text(json.dumps(
        {**{f"tests/test_census.py::test_cell[{n}]": 30.0 for n in range(40)},
         "tests/test_light.py::test_a": 20.0, "tests/test_other.py::test_b": 20.0}))
    timings = {"files": {"3.12:tests/test_census.py": 6.0, "3.12:tests/test_light.py": 20.0,
                         "3.12:tests/test_other.py": 20.0},
               "rows": {"3.12:tests/test_census.py": 6.0}}
    floor = sorted((entry["tests"], entry["split"], entry["weight"]) for entry in ci.plan(240.0, timings)
                   if entry["python"] == "3.12")
    assert floor == [*((("tests/test_census.py", f"{group}/5", 240.0)) for group in range(1, 6)),
                     ("tests/test_light.py tests/test_other.py", "", 40.0)]


def test_a_split_file_pays_its_setup_once_a_group_and_is_not_cut_finer_for_it(ci):
    """Twenty recorded tests of 10 s that armada measured whole at 400 s paid 200 s besides their
    tests, which every group pays again: at a 150 s target that leaves a group half the target of
    tests, so 3 groups of 270, 270 and 260 s, not slivers each paying it. Measured at 2 groups,
    300 s, each paid 50 s: its setup is not counted again for the groups it ran as, and it is
    measured by its rows at that cut, not by a median that remembers other cuts."""
    tests = [10.0] * 20
    assert ci.split(tests, 400.0, 150.0, cut=1) == [270.0, 270.0, 260.0]
    assert ci.split(tests, 200.0 + 2 * 50.0, 150.0, cut=2) == [150.0, 150.0]
    # A cut whose heaviest group, setup included, holds within half again the target is kept, so
    # its cached programs are: 4 groups of 100 s, and 20 groups of 50 s (an over-cut paying 40 s
    # each), stay as they ran; 2 groups at 130 s of setup each (230 s, past 225) are cut again.
    assert ci.split(tests, 400.0, 150.0, cut=4) == [100.0] * 4
    assert ci.split(tests, 200.0 + 20 * 40.0, 150.0, cut=20) == [50.0] * 20
    assert ci.split(tests, 200.0 + 2 * 130.0, 150.0, cut=2) == [200.0, 200.0, 190.0]
    # The file's median remembers runs at other cuts; its rows are timed at the cut they name. Here
    # four runs at 4 groups (400 s) and the last at 2 (300 s, its rows): measured at 2, it stays at 2.
    rows = {"3.12:tests/test_split.py#1/2": 150.0, "3.12:tests/test_split.py#2/2": 150.0,
            "3.12:tests/test_split.py#3/4": 100.0, "3.14:tests/test_split.py#1/4": 10.0,
            "3.12:tests/test_split.py#collect": 1.0, "3.12:tests/test_whole.py": 30.0}
    timings = {"rows": rows, "files": {"3.12:tests/test_split.py": 400.0}}
    assert ci.measured_cut(timings, "3.12", "tests/test_split.py") == (300.0, 2)
    seconds, count = ci.measured_cut(timings, "3.12", "tests/test_split.py")
    assert ci.split(tests, seconds, 150.0, cut=count) == [150.0, 150.0]
    assert ci.measured_cut(timings, "3.12", "tests/test_whole.py") == (30.0, 1)
    assert ci.measured_cut(timings, "3.12", "tests/test_unseen.py") is None


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
    interpreter(tmp_path, "3.12")
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


def test_the_durations_fold_rewrites_only_a_file_its_record_no_longer_cuts(ci, tmp_path, monkeypatch):
    """`durations` folds main's CI verdict into the record a file at a time, each whole, from its floor rows:
    a file whose cut no longer fits as its tests ran takes what they took, though no test's node id changed
    and its total barely moved (one test grown from 10 s to 180 s of 100; ten of 20 grown from 30 s to 39 s,
    each within half again), so the next plan cuts it to fit; a file whose cut fits keeps its entries byte for
    byte, as does one whose heaviest group is a test past the budget alone, which no cut shortens, though
    its time moves; a file with a red group keeps its own; a file the checkout no longer has loses them; the
    newest Python's times move nothing."""
    (tmp_path / ".armada.json").write_text(json.dumps({"target": 240}))
    record, rows = {}, []

    def file(name: str, recorded: list[float], ran: list[list[float]], red: int | None = None) -> None:
        (tmp_path / f"tests/{name}.py").write_text("")
        nodes = [f"tests/{name}.py::test_{index:03}" for index in range(len(recorded))]
        record.update(zip(nodes, recorded, strict=True))
        start = 0
        for group, times in enumerate(ran, start=1):
            split = f"#{group}/{len(ran)}" if len(ran) > 1 else ""
            tests = dict(zip(nodes[start:start + len(times)], times, strict=True))
            start += len(times)
            rows.append({"name": f"3.12:tests/{name}.py{split}", "exitCode": int(group == red),
                         "seconds": sum(times), "tests": tests})
            rows.append({"name": f"3.14:tests/{name}.py{split}", "exitCode": 0, "seconds": 999.0,
                         "tests": dict.fromkeys(tests, 999.0)})

    file("test_grown", [10.0] * 100, [[180.0] + [10.0] * 19] + [[10.0] * 20] * 4)
    file("test_drift", [30.0] * 20, [[39.0] * 10, [30.0] * 10])
    file("test_steady", [100.0] * 4, [[101.0, 99.0], [104.5, 98.0]])
    file("test_straggler", [381.593, 20.0, 20.0], [[382.0], [20.5, 19.0]])
    file("test_partial", [150.0] * 4, [[400.0, 400.0], [150.0, 150.0]], red=2)
    file("test_empty", [], [[]])  # every test deselected, its collection past the target
    rows[-2]["seconds"] = 300.0
    record["tests/test_gone.py::test_x"] = 5.0
    rows.append({"name": "3.12:tests/test_gone.py", "exitCode": 0, "seconds": 5.0,
                 "tests": {"tests/test_gone.py::test_x": 5.0, "tests/test_gone.py::test_y": 1.0}})
    (tmp_path / "tests/test_durations.json").write_text(json.dumps(record, indent=0) + "\n")
    verdict = tmp_path / "main.json"
    verdict.write_text(json.dumps({"rows": rows}))
    monkeypatch.setattr(sys, "argv", ["ci.py", "durations", str(verdict)])
    assert ci.main() == 0
    folded = json.loads((tmp_path / "tests/test_durations.json").read_text())
    observed = {node: seconds for row in rows if row["name"].startswith("3.12:") and not row["exitCode"]
                for node, seconds in row["tests"].items()}
    for name in ("test_grown", "test_drift"):
        assert {node: s for node, s in folded.items() if f"/{name}.py" in node} == {
            node: s for node, s in observed.items() if f"/{name}.py" in node}
    for name in ("test_steady", "test_straggler", "test_partial"):
        assert {node: s for node, s in folded.items() if f"/{name}.py" in node} == {
            node: s for node, s in record.items() if f"/{name}.py" in node}
    assert not [node for node in folded if node.startswith("tests/test_gone.py")]


def test_a_tasks_wall_time_is_apportioned_by_its_files_tests_times(ci):
    """A task's wall time split among its files in proportion to their tests'
    own times, and evenly over its rows where no test recorded one: recorded
    times whose shares are exact in binary."""
    assert ci.apportioned({"tests/test_a.py": 0.25, "tests/test_b.py": 0.75}, 2.0, 2) == {
        "tests/test_a.py": 0.5, "tests/test_b.py": 1.5}
    assert ci.apportioned({"tests/test_a.py": 0.0, "tests/test_b.py": 0.0}, 3.0, 2) == {
        "tests/test_a.py": 1.5, "tests/test_b.py": 1.5}
    # A file named twice is two of the task's rows.
    assert ci.apportioned({"tests/test_a.py": 0.0, "tests/test_b.py": 0.0}, 3.0, 3) == {
        "tests/test_a.py": 1.0, "tests/test_b.py": 1.0}


def test_a_tasks_rows_share_its_wall_time_by_their_tests_times(ci, tmp_path):
    """The plan weighs a file by what a task spends on it, its import and
    collection included, so a task's rows add up to its wall time, split by
    their tests' own times (`apportioned`): at least what the tests slept, at
    most the task's wall time, and the longer sleep the larger row."""
    (tmp_path / "tests").mkdir(exist_ok=True)
    (tmp_path / "tests/test_short.py").write_text("import time\ndef test_a(): time.sleep(0.3)\n")
    (tmp_path / "tests/test_long.py").write_text("import time\ndef test_b(): time.sleep(0.9)\n")
    interpreter(tmp_path, "3.12")
    out = tmp_path / "verdict.json"
    began = time.monotonic()
    ci.task("3.12", ["tests/test_short.py", "tests/test_long.py"], "", out, 60.0)
    wall = time.monotonic() - began
    rows = {row["name"]: row["seconds"] for row in json.loads(out.read_text())["rows"]}
    short, long = rows["3.12:tests/test_short.py"], rows["3.12:tests/test_long.py"]
    assert 1.2 < short + long <= wall
    assert short < long


def test_a_task_reuses_a_green_row_only_on_the_cpu_class_and_inputs_it_ran_on(ci, tmp_path, monkeypatch,
                                                                                  cache_worker):
    """A file that ran green is reused by a later task on the same CPU class while its inputs are
    unchanged; a task on another class, after its inputs change, or planned with DEW_TEST_CACHE=off
    (the daily full run) runs it again, and once it has run red on those inputs, so does every task
    after, though the red run died before it wrote a report. A commit's retry reuses its green rows,
    and none after a red run on its inputs, though a run of another commit recorded the file since.
    Each task runs as armada starts it: .armada.json's command filled from its plan entry."""
    (tmp_path / "tests/test_counted.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "def test_ok():\n"
        "    with Path('runs').open('a') as runs: runs.write('x')\n"
        "    assert not os.environ.get('FAIL')\n"
        "    if os.environ.get('EXIT'): os._exit(3)\n")
    interpreter(tmp_path, "3.12")
    url, _, _ = cache_worker
    cached(monkeypatch, url)
    out = tmp_path / "verdict.json"

    def run(cpu, policy="on"):
        monkeypatch.setattr(ci, "cpu_class", lambda: cpu)
        [only] = armada_task(ci, monkeypatch, out, policy)
        assert (only["exitCode"] == 0) == (not os.environ.get("FAIL") and not os.environ.get("EXIT"))
        return len((tmp_path / "runs").read_text())

    assert run("GenuineIntel-6-143") == 1
    assert run("GenuineIntel-6-143") == 1
    assert run("AuthenticAMD-25-1") == 2
    assert run("AuthenticAMD-25-1") == 2
    assert run("AuthenticAMD-25-1", policy="off") == 3
    (tmp_path / "tests/test_counted.py").write_text(
        (tmp_path / "tests/test_counted.py").read_text() + "def test_more(): pass\n")
    assert run("GenuineIntel-6-143") == 4
    # A flaky red on the inputs a green run recorded: the green row is not reused after it.
    monkeypatch.setenv("FAIL", "1")
    assert run("GenuineIntel-6-143", policy="off") == 5
    monkeypatch.delenv("FAIL")
    assert run("GenuineIntel-6-143") == 6
    assert run("GenuineIntel-6-143") == 6
    # A red run that died before it wrote its report.
    monkeypatch.setenv("EXIT", "1")
    assert run("GenuineIntel-6-143", policy="off") == 7
    monkeypatch.delenv("EXIT")
    assert run("GenuineIntel-6-143") == 8
    # Another commit's run between a commit's run and its retry: a file every test process reads
    # (`SHARED`) differs between them, so each has a record of its own.
    shared = tmp_path / "constraints.txt"
    shared.write_text("")
    assert run("GenuineIntel-6-143") == 9
    shared.write_text("a-package==1\n")
    assert run("GenuineIntel-6-143") == 10
    shared.write_text("")
    assert run("GenuineIntel-6-143") == 10
    monkeypatch.setenv("FAIL", "1")
    assert run("GenuineIntel-6-143", policy="off") == 11
    monkeypatch.delenv("FAIL")
    shared.write_text("a-package==1\n")
    assert run("GenuineIntel-6-143") == 11
    shared.write_text("")
    assert run("GenuineIntel-6-143") == 12



def cached_checkout(ci, root: Path, monkeypatch, cache_worker) -> Path:
    """Checkout `root` run by 3.12 tasks on one CPU class against the shared test cache, and where their
    verdict is written: beside the checkout's entries, not among them (the checkout's own directory is an
    input)."""
    interpreter(root, "3.12")
    url, _, _ = cache_worker
    monkeypatch.setattr(ci, "cpu_class", lambda: "GenuineIntel-6-143")
    cached(monkeypatch, url)
    (root / ".verdict").mkdir()
    return root / ".verdict/verdict.json"


def cached_runs(ci, root: Path, monkeypatch, cache_worker) -> Callable[[], int]:
    """The checkout's one 3.12 task, run green on one CPU class against the shared test cache as armada
    starts it (`armada_task`) under the cache's policy (`cached_checkout`): each call, how many times its
    test has run, which it counts in `runs`."""
    verdict = cached_checkout(ci, root, monkeypatch, cache_worker)

    def run(policy: str = "on") -> int:
        rows = armada_task(ci, monkeypatch, verdict, policy)
        assert [row["exitCode"] for row in rows] == [0]
        return len((root / "runs").read_text())

    return run


@pytest.mark.parametrize(("changed", "policy"), [
    ({"src/pkg/mod.py": "VALUE = 2\n"}, "on"), ({"docs/page.md": "page\n"}, "on"),
    ({"tutorials/notes.ipynb": "{}\n"}, "on"), ({"tools/tool.py": "VALUE = 2\n"}, "on"),
    ({"pyproject.toml": "[project]\nname = 'changed'\n"}, "off"), ({"locks/cpu.txt": "jax==9\n"}, "off"),
    ({"tools/armada/session.py": "# the session\n"}, "off"),
    ({"tools/armada/executed.py": "# the recorder\n"}, "off"),
    ({".github/workflows/ci.yml": "on: []\n"}, "on"),
    ({"Makefile": "all:\n"}, "off"),
    ({"CHANGELOG.md": "# Changes\n\n- an entry\n"}, "on"),
    ({".armada.json": '{"environment": {"base": "debian"}, "env": {"MODE": "1"}, "pool": 400}\n'}, "on"),
    ({".armada.json": '{"environment": {"base": "debian"}, "env": {"MODE": "2"}, "pool": 160}\n'}, "off"),
], ids=["a source module", "a docs page", "a notebook", "a tool", "the project", "a lock",
        "the test session", "the recorder", "a workflow", "a file no record places", "the changelog",
        "the pool", "the environment's variables"])
def test_the_policy_reruns_what_a_change_reaches_and_everything_where_it_cannot_place_it(
        ci, tmp_path, changed, policy):
    """CI's test cache policy for a change from main: each file's own inputs decide ('on') where the records
    place what changed, a tool and a workflow among them, and every file runs ('off') for a change to what
    every test process runs or reads, or to a file the records do not place. .armada.json's pool decides
    where tasks run, not what a test sees (owner, 2026-10-27: a pool change reran every file), and a workflow
    is an input of the files that read it alone (Main, 2026-10-28: 5 of 10 trains ran with the cache off, and
    0.75% of rows were reused). The recorder (executed.py, followed.py, watched.py) stays every file's input:
    an old record may lack a dependency a new recorder finds."""
    armada = '{"environment": {"base": "debian"}, "env": {"MODE": "1"}, "pool": 160}\n'
    for name, text in {"src/pkg/mod.py": "VALUE = 1\n", "pyproject.toml": "[project]\nname = 'x'\n",
                       "locks/cpu.txt": "jax==1\n", ".armada.json": armada}.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "base")
    for name, text in changed.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "change")
    assert ci.policy("HEAD~1") == policy


def test_a_fold_that_moves_a_test_between_groups_runs_each_test_once_in_its_new_group(
        ci, tmp_path, monkeypatch, cache_worker):
    """A fold that moves a test from one group to another, the file's cut count unchanged, reuses neither
    group's green row: the next run of the file's groups, cut by the split plugin from the folded record, runs
    every test once, the moved one in its new group."""
    (tmp_path / ".armada.json").write_text(json.dumps({"target": 300, "environment": {}}))
    (tmp_path / "tests/test_moved.py").write_text(
        "from pathlib import Path\n"
        + "".join(f"def test_{index}():\n"
                  f"    with Path('runs').open('a') as runs: runs.write('{index}')\n" for index in range(3)))
    nodes = [f"tests/test_moved.py::test_{index}" for index in range(3)]
    (tmp_path / "tests/test_durations.json").write_text(json.dumps(dict(zip(nodes, [300.0, 100.0, 100.0],
                                                                           strict=True))))
    verdict = cached_checkout(ci, tmp_path, monkeypatch, cache_worker)
    base = commit(tmp_path, "base")

    def run() -> dict[str, list[str]]:
        rows = armada_task(ci, monkeypatch, verdict, ci.policy(base))
        assert all(row["exitCode"] == 0 for row in rows)
        return {row["name"]: sorted(row["tests"]) for row in rows}

    assert run() == {"3.12:tests/test_moved.py#1/2": nodes[:1], "3.12:tests/test_moved.py#2/2": nodes[1:]}
    main_ci = tmp_path / ".verdict/main.json"
    main_ci.write_text(json.dumps({"rows": [
        {"name": "3.12:tests/test_moved.py#1/2", "exitCode": 0, "seconds": 100.0, "tests": {nodes[0]: 100.0}},
        {"name": "3.12:tests/test_moved.py#2/2", "exitCode": 0, "seconds": 600.0,
         "tests": {nodes[1]: 100.0, nodes[2]: 500.0}}]}))
    monkeypatch.setattr(sys, "argv", ["ci.py", "durations", str(main_ci)])
    assert ci.main() == 0
    commit(tmp_path, "main's CI folded")
    (tmp_path / "runs").write_text("")
    assert run() == {"3.12:tests/test_moved.py#1/2": nodes[:2], "3.12:tests/test_moved.py#2/2": nodes[2:]}
    assert sorted((tmp_path / "runs").read_text()) == ["0", "1", "2"]


def test_a_durations_change_reruns_only_the_groups_it_moves(ci, tmp_path, monkeypatch, cache_worker):
    """The durations record is no file's input but a split file's groups', which are cut from that file's
    own entries: a change to another file's entries reuses every row, and so does one to the split file's
    seconds that leaves each of its tests in its group, as a group's key holds the tests it runs, not their
    timings; one that moves a test to another group reruns the groups (a refresh reran the whole suite while
    the record was every file's input; Main, 2026-10-28: each train's fold of main's durations re-keyed every
    split file, row 2000)."""
    count = {"heavy": 3, "light": 2}
    for name, tests in count.items():
        (tmp_path / f"tests/test_{name}.py").write_text(
            "from pathlib import Path\n"
            + "".join(f"def test_{name}_{index}():\n"
                      f"    with Path('runs-{name}').open('a') as runs: runs.write('x')\n"
                      for index in range(tests)))
    heavy = {f"tests/test_heavy.py::test_heavy_{index}": seconds
             for index, seconds in enumerate((300.0, 1.0, 1.0))}
    record = {**heavy, "tests/test_light.py::test_light_0": 1.0, "tests/test_light.py::test_light_1": 1.0}
    (tmp_path / "tests/test_durations.json").write_text(json.dumps(record))
    verdict = cached_checkout(ci, tmp_path, monkeypatch, cache_worker)
    base = commit(tmp_path, "base")

    def run() -> dict[str, int]:
        rows = armada_task(ci, monkeypatch, verdict, ci.policy(base))
        assert sorted(row["name"] for row in rows if not row["exitCode"]) == [
            "3.12:tests/test_heavy.py#1/2", "3.12:tests/test_heavy.py#2/2", "3.12:tests/test_light.py"]
        return {name: len((tmp_path / f"runs-{name}").read_text()) for name in ("heavy", "light")}

    assert run() == {"heavy": 3, "light": 2}
    (tmp_path / "tests/test_durations.json").write_text(
        json.dumps({**record, "tests/test_light.py::test_light_0": 2.0}))
    commit(tmp_path, "the light file's entry moves")
    assert run() == {"heavy": 3, "light": 2}
    # The heavy file's seconds change, each test kept in its group: the first alone, the other two together.
    (tmp_path / "tests/test_durations.json").write_text(
        json.dumps({**record, "tests/test_heavy.py::test_heavy_0": 301.0}))
    commit(tmp_path, "the heavy file's entries move, its groups as they were")
    assert run() == {"heavy": 3, "light": 2}
    # Its last test turns heavy: the cut moves, and both groups run again.
    (tmp_path / "tests/test_durations.json").write_text(json.dumps(
        {**record, "tests/test_heavy.py::test_heavy_0": 1.0, "tests/test_heavy.py::test_heavy_2": 300.0}))
    commit(tmp_path, "the heavy file's cut moves")
    assert run() == {"heavy": 6, "light": 2}


def green_runs(ci, monkeypatch, root: Path, verdict, policy: str, names: tuple[str, ...]) -> list[int]:
    """One armada task of the checkout at `root` under `policy`, every row green: how many times each of
    `names`' tests has run so far (its runs-NAME file, a mark per run)."""
    rows = armada_task(ci, monkeypatch, verdict, policy)
    assert [row["exitCode"] for row in rows] == [0] * len(names)
    return [len((root / f"runs-{name}").read_text()) for name in names]


def commit(root: Path, message: str) -> str:
    """Checkout `root`'s changes committed, and the commit's sha: every file at the first, which makes the
    repository, and the tracked and staged changes after, so what a run leaves in the checkout stays out."""
    first = not (root / ".git").exists()
    if first:
        git(root, "init", "-q")
        git(root, "add", "-A")
    git(root, "commit", "-qm" if first else "-qam", message)
    return git(root, "rev-parse", "HEAD").strip()


def test_a_watch_the_kernel_cannot_hold_whole_is_blind_and_keys_nothing(tmp_path):
    """A tree with more directories than inotify will watch (fs.inotify.max_user_watches; here a user
    namespace's own limit, lowered to 2) leaves the open watch blind: closing it gives no opened files, so
    no record keeps a key, and nothing raises (Main, 2026-10-28: a failed watch left the reader unset, and
    its close crashed)."""
    for name in ("a", "b", "c"):
        (tmp_path / "tests" / name).mkdir(parents=True)
    script = (
        "import sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from pathlib import Path\n"
        "Path('/proc/sys/user/max_inotify_watches').write_text('2')\n"
        "import watched\n"
        "print(watched.Opens(sys.argv[2]).closed())\n")
    ran = subprocess.run(["unshare", "--user", "--map-root-user", sys.executable, "-c", script,
                          str(ROOT / "tools/armada"), str(tmp_path)],
                         capture_output=True, text=True, timeout=60)
    assert (ran.returncode, ran.stdout.strip()) == (0, "None"), ran.stderr


def test_a_changed_workflow_reruns_only_the_files_that_read_it(ci, tmp_path, monkeypatch, cache_worker):
    """A change under .github/ reruns the files whose runs read the changed file, which the records place as
    any of the checkout's files, and reuses every other file's green row, under the policy CI computes for
    the change from main (Main, 2026-10-28: a workflow change turned the cache off for the whole suite)."""
    (tmp_path / ".github/workflows").mkdir(parents=True)
    (tmp_path / ".github/workflows/ci.yml").write_text("name: CI\n")
    (tmp_path / "tests/test_reader.py").write_text(
        "from pathlib import Path\n"
        "def test_reader():\n"
        "    with Path('runs-reader').open('a') as runs: runs.write('x')\n"
        "    assert Path('.github/workflows/ci.yml').read_text().startswith('name')\n")
    (tmp_path / "tests/test_other.py").write_text(
        "from pathlib import Path\n"
        "def test_other():\n"
        "    with Path('runs-other').open('a') as runs: runs.write('x')\n")
    verdict = cached_checkout(ci, tmp_path, monkeypatch, cache_worker)
    base = commit(tmp_path, "base")

    def run(policy: str | None = None) -> list[int]:
        return green_runs(ci, monkeypatch, tmp_path, verdict, policy or ci.policy(base), ("reader", "other"))

    assert run() == [1, 1]
    (tmp_path / ".github/workflows/ci.yml").write_text("name: CI, again\n")
    commit(tmp_path, "the workflow changes")
    assert ci.policy(base) == "on"
    assert run() == [2, 1]
    # A change to the recorder keys every file anew, the cache on though it is: a record it took before may
    # lack a dependency it now finds, so no old record is reused.
    (tmp_path / "tools/armada").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tools/armada/executed.py").write_text("# the recorder, as it records now\n")
    git(tmp_path, "add", "tools/armada/executed.py")
    commit(tmp_path, "the recorder changes")
    assert ci.policy(base) == "off"
    assert run("on") == [3, 2]


def test_a_run_that_read_its_limits_runs_again_where_they_change(ci, tmp_path, monkeypatch, cache_worker):
    """A test whose run read its cgroup's CPU or memory limit (dew.data's `Host.current` sizes loading by
    them) keys those limits: on a container of other limits its row is not reused though nothing else
    changed, and a file that read none is reused (row 1998; Main, 2026-10-28: a row green on 4 vCPUs was
    reused on 8)."""
    cgroup = tmp_path.parent / f"{tmp_path.name}-cgroup"
    cgroup.mkdir()
    (cgroup / "cpu.max").write_text("400000 100000\n")
    (cgroup / "memory.max").write_text("17179869184\n")
    monkeypatch.setenv("DEW_CGROUP_ROOT", str(cgroup))
    (tmp_path / "tests/test_sized.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "def test_sized():\n"
        "    with Path('runs-sized').open('a') as runs: runs.write('x')\n"
        "    assert Path(os.environ['DEW_CGROUP_ROOT'], 'cpu.max').read_text().split()[1] == '100000'\n")
    (tmp_path / "tests/test_plain.py").write_text(
        "from pathlib import Path\n"
        "def test_plain():\n"
        "    with Path('runs-plain').open('a') as runs: runs.write('x')\n")
    verdict = cached_checkout(ci, tmp_path, monkeypatch, cache_worker)
    base = commit(tmp_path, "base")

    def run() -> list[int]:
        return green_runs(ci, monkeypatch, tmp_path, verdict, ci.policy(base), ("plain", "sized"))

    assert run() == [1, 1]
    assert run() == [1, 1]
    # The same checkout on a container of 8 vCPUs, not 4.
    (cgroup / "cpu.max").write_text("800000 100000\n")
    assert run() == [1, 2]


def test_a_changed_tool_runs_again_only_the_files_that_read_it(ci, tmp_path, monkeypatch, cache_worker):
    """A change under tools/ reruns the files whose runs executed or read the changed tool, which their keys
    hash, and reuses every other file's green row, though it read another tool: the policy CI computes for
    the change from main is the one the task runs under (train 115 reran its whole suite for five tools no
    pulled head had touched)."""
    (tmp_path / "tools").mkdir()
    for name in ("first", "second"):
        (tmp_path / f"tools/{name}.py").write_text("VALUE = 1\n")
        (tmp_path / f"tests/test_{name}.py").write_text(
            "import runpy\n"
            "from pathlib import Path\n"
            f"def test_{name}():\n"
            f"    with Path('runs-{name}').open('a') as runs: runs.write('x')\n"
            f"    assert runpy.run_path('tools/{name}.py')['VALUE'] >= 1\n")
    verdict = cached_checkout(ci, tmp_path, monkeypatch, cache_worker)
    base = commit(tmp_path, "base")

    def run() -> list[int]:
        return green_runs(ci, monkeypatch, tmp_path, verdict, ci.policy(base), ("first", "second"))

    assert run() == [1, 1]
    (tmp_path / "tools/first.py").write_text("VALUE = 2\n")
    commit(tmp_path, "the first tool changes")
    assert run() == [2, 1]
    # A package added beside a tool a test imports by its name, from tools/ on sys.path, shadows it: each file
    # that executed a tool there runs again, and the one that imports it by name sees the package.
    (tmp_path / "tests/test_named.py").write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "def test_named():\n"
        "    with Path('runs-named').open('a') as runs: runs.write('x')\n"
        "    sys.path.insert(0, 'tools')\n"
        "    import named\n"
        "    assert named.VALUE == 1\n")
    (tmp_path / "tools/named.py").write_text("VALUE = 1\n")
    git(tmp_path, "add", "tests/test_named.py", "tools/named.py")
    commit(tmp_path, "a test imports a tool by its name")
    rows = armada_task(ci, monkeypatch, verdict, ci.policy(base))
    assert [row["exitCode"] for row in rows] == [0, 0, 0]
    (tmp_path / "tools/named").mkdir()
    (tmp_path / "tools/named/__init__.py").write_text("VALUE = 2\n")
    git(tmp_path, "add", "tools/named/__init__.py")
    commit(tmp_path, "a package shadows the tool")
    assert ci.policy(base) == "on"
    rows = armada_task(ci, monkeypatch, verdict, ci.policy(base))
    assert {row["name"]: row["exitCode"] for row in rows}["3.12:tests/test_named.py"] == 1


def test_a_reused_row_runs_again_only_when_what_its_test_process_runs_or_reads_changes(
        ci, tmp_path, monkeypatch, cache_worker):
    """A file's green row is kept across a change its tests never meet, the plan's own code (ci.py, which
    runs outside every test process), another test module, or a fixture it never opens (its openers'
    input alone); it runs again once what its test process runs or reads changes: a plugin loaded into
    it, a Python file added that can change what an import finds (a conftest.py, a top-level package
    under src/), or the process group it runs as."""
    (tmp_path / "tests/test_kept.py").write_text(
        "from pathlib import Path\n"
        "def test_kept():\n"
        "    with Path('runs').open('a') as runs: runs.write('x')\n")
    armada = tmp_path / "tools/armada"
    armada.mkdir(parents=True)
    for tool in (ROOT / "tools/armada").glob("*.py"):
        (armada / tool.name).write_text(tool.read_text())
    run = cached_runs(ci, tmp_path, monkeypatch, cache_worker)
    changes = [(None, "", 1), ("tools/armada/ci.py", "# the plan's own code\n", 1),
               ("tools/armada/executed.py", "# a plugin loaded into it\n", 2),
               ("tests/fixtures/data.json", "{}\n", 2), ("tests/conftest.py", "VALUE = 1\n", 3),
               ("src/package/__init__.py", "", 4), ("tests/more/test_added.py", "VALUE = 2\n", 4),
               ("tools/armada/child_groups.py", "# its process group\n", 5)]
    for path, text, runs in changes:
        if path:
            (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
            with (tmp_path / path).open("a") as file:
                file.write(text)
        assert run() == runs, path


def test_a_fixture_reruns_its_readers_alone_though_a_native_reader_reads_it(
        ci, tmp_path, monkeypatch, cache_worker):
    """A fixture is the input of the files that open it, whatever opens it: a tokenizer's JSON that Rust's
    `Tokenizer.from_file` reads, which raises no audit event, runs its test again once changed (tests/
    test_text_data.py reads one so), and no other file; a fixture no test opens runs none (owner,
    2026-10-27: one fixture re-recorded reran all 1,300 rows)."""
    tokenizers = pytest.importorskip("tokenizers")

    def tokenizer(vocabulary: dict[str, int]) -> None:
        model = tokenizers.models.WordLevel({**vocabulary, "[UNK]": len(vocabulary)}, unk_token="[UNK]")
        tokenizers.Tokenizer(model).save(str(tmp_path / "tests/fixtures/tok/tokenizer.json"))

    (tmp_path / "tests/fixtures/tok").mkdir(parents=True)
    tokenizer({"a": 0})
    (tmp_path / "tests/fixtures/unread.json").write_text("{}\n")
    written(tmp_path, {
        "test_a_native.py": "from pathlib import Path\nfrom tokenizers import Tokenizer\n"
                            "def test_native():\n"
                            "    with Path('runs-native').open('a') as runs: runs.write('x')\n"
                            "    loaded = Tokenizer.from_file('tests/fixtures/tok/tokenizer.json')\n"
                            "    assert loaded.get_vocab_size()\n",
        "test_b_other.py": "from pathlib import Path\n"
                           "def test_other():\n"
                           "    with Path('runs-other').open('a') as runs: runs.write('x')\n"})
    run = counted_runs(ci, tmp_path, monkeypatch, cache_worker, ("native", "other"))

    assert run() == [1, 1]
    assert run() == [1, 1]
    tokenizer({"a": 0, "b": 1})
    assert run() == [2, 1]
    (tmp_path / "tests/fixtures/unread.json").write_text("[]\n")
    assert run() == [2, 1]


def counted_runs(ci, root: Path, monkeypatch, cache_worker,
                 names: tuple[str, ...]) -> Callable[[], list[int]]:
    """The checkout's 3.12 tasks, run green on one CPU class against the shared test cache as armada starts
    them (`cached_checkout`): each call, how many times each test file named in `names` has run, which it
    counts in runs-<name>."""
    verdict = cached_checkout(ci, root, monkeypatch, cache_worker)

    def run() -> list[int]:
        rows = armada_task(ci, monkeypatch, verdict)
        assert [row["exitCode"] for row in rows] == [0] * len(names)
        return [len((root / f"runs-{name}").read_text()) for name in names]

    return run


def packed(root: Path, names: list[str]) -> None:
    """The durations record timing each of `names` (tests/<name>) at a second, so the plan packs them into one
    task, as CI packs small files."""
    (root / "tests/test_durations.json").write_text(json.dumps(
        {f"tests/{name}::{name[len('test_'):-len('.py')]}": 1.0 for name in names}))


def test_a_reader_files_share_in_one_process_is_every_such_files_input(
        ci, tmp_path, monkeypatch, cache_worker):
    """A data file a reader opens is the input of every test file its pytest process ran, not only the
    one that opened it: a session fixture's lazy reader, or a process-lived cache (dew.data.text's
    load_tokenizer), opened under one file serves the next without reopening it, so a change to the data
    reruns both (train 239's review, row 1987)."""
    (tmp_path / "tests/fixtures").mkdir(parents=True)
    (tmp_path / "tests/fixtures/shared.json").write_text("[1]\n")
    (tmp_path / "tests/conftest.py").write_text(
        "import functools, pytest\nfrom pathlib import Path\n"
        "@pytest.fixture(scope='session')\ndef vocabulary():\n"
        "    return functools.cache(lambda: Path('tests/fixtures/shared.json').read_text())\n")
    body = ("from pathlib import Path\ndef test_{0}(vocabulary):\n"
            "    with Path('runs-{0}').open('a') as runs: runs.write('x')\n    assert vocabulary()\n")
    written(tmp_path, {"test_a.py": body.format("a"), "test_b.py": body.format("b")})
    packed(tmp_path, ["test_a.py", "test_b.py"])
    run = counted_runs(ci, tmp_path, monkeypatch, cache_worker, ("a", "b"))

    assert run() == [1, 1]
    assert run() == [1, 1]
    (tmp_path / "tests/fixtures/shared.json").write_text("[2]\n")
    assert run() == [2, 2]


def test_an_open_after_the_tests_ended_is_the_records_input(ci, tmp_path, monkeypatch, cache_worker):
    """A data file opened after the last test ended, by a later session-finish hook, at exit or by a
    thread, is the task's input as well: the task's supervisor reads the watch once pytest and its process
    group stopped, so a change to it reruns the files (train 239's review, row 1988)."""
    (tmp_path / "tests/fixtures").mkdir(parents=True)
    (tmp_path / "tests/fixtures/verdict.txt").write_text("ok\n")
    (tmp_path / "tests/conftest.py").write_text(
        "import pytest\nfrom pathlib import Path\n"
        "@pytest.hookimpl(trylast=True)\ndef pytest_sessionfinish(session):\n"
        "    Path('tests/fixtures/verdict.txt').read_text()\n")
    written(tmp_path, {"test_late.py": "from pathlib import Path\ndef test_late():\n"
                                       "    with Path('runs').open('a') as runs: runs.write('x')\n"})
    run = cached_runs(ci, tmp_path, monkeypatch, cache_worker)
    assert [run(), run()] == [1, 1]
    (tmp_path / "tests/fixtures/verdict.txt").write_text("no\n")
    assert run() == 2


def test_a_watched_directory_lost_mid_run_keeps_no_row(ci, tmp_path, monkeypatch, cache_worker):
    """A directory under tests/ or src/ deleted (or moved, or unmounted) while the task runs takes its
    watch with it, and what is opened in it, or in one made again in its place, is not seen: the task's
    records keep no key, so its rows run every time (train 239's review, row 1989)."""
    (tmp_path / "tests/fixtures/scratch").mkdir(parents=True)
    written(tmp_path, {"test_lost.py": "import shutil\nfrom pathlib import Path\ndef test_lost():\n"
                                       "    with Path('runs').open('a') as runs: runs.write('x')\n"
                                       "    shutil.rmtree('tests/fixtures/scratch')\n"
                                       "    Path('tests/fixtures/scratch').mkdir()\n"})
    run = cached_runs(ci, tmp_path, monkeypatch, cache_worker)
    assert [run(), run()] == [1, 2]


def test_a_pool_change_reruns_no_file_and_the_environments_variables_every_file(
        ci, tmp_path, monkeypatch, cache_worker):
    """.armada.json's pool decides where tasks run, so a change to it reruns no file; a change to the
    variables its environment sets, which every test process sees, reruns every file (owner, 2026-10-27:
    a pool change reran all 1,300 rows)."""
    armada = {"environment": {"base": "debian"}, "env": {"MODE": "1"}, "pool": 160}
    (tmp_path / ".armada.json").write_text(json.dumps(armada))
    (tmp_path / "tests/test_kept.py").write_text(
        "from pathlib import Path\n"
        "def test_kept():\n"
        "    with Path('runs').open('a') as runs: runs.write('x')\n")
    run = cached_runs(ci, tmp_path, monkeypatch, cache_worker)
    changes = [(None, 1), ({**armada, "pool": 400}, 1),
               ({**armada, "pool": 400, "env": {"MODE": "2"}}, 2)]
    for config, runs in changes:
        if config:
            (tmp_path / ".armada.json").write_text(json.dumps(config))
        assert run() == runs, config


def test_a_reused_row_runs_again_when_a_page_its_tests_read_or_a_directory_they_list_changes(
        ci, tmp_path, monkeypatch, cache_worker):
    """A test's inputs outside tests/ and src/ are its file's inputs too (executed.py): a documentation
    page its test reads, and the pages of a directory it lists, so a green row is reused while they
    stay, and its file runs again once the page it trains from changes or a page is added. So are a
    directory `os.walk` lists, the checkout's own directory listed, and a page opened by its name in
    a directory's descriptor. A file it writes is no input, and nor is what the import system or
    pytest's collection lists."""
    (tmp_path / "docs/guides").mkdir(parents=True)
    (tmp_path / "docs/model.md").write_text("width 4\n")
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes/plan.md").write_text("plan\n")
    (tmp_path / "tests/test_pages.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "def test_pages():\n"
        "    with Path('runs').open('a') as runs: runs.write('x')\n"
        "    assert Path('docs/model.md').read_text().startswith('width')\n"
        "    assert sorted(path.name for path in Path('docs').glob('*.md'))\n"
        "    assert [folder for folder, _, _ in os.walk('docs')]\n"
        "    assert 'docs' in os.listdir('.')\n"
        "    folder = os.open('notes', os.O_RDONLY)\n"
        "    try:\n"
        "        os.close(os.open('plan.md', os.O_RDONLY, dir_fd=folder))\n"
        "    finally:\n"
        "        os.close(folder)\n")
    run = cached_runs(ci, tmp_path, monkeypatch, cache_worker)
    # Each page changed, added or listed anew runs the file again, and the run after reuses that row.
    # Python bytecode a run leaves beside a page (a directory, `None` here) is no change to the pages.
    changes = [(None, "", 1), ("docs/model.md", "width 8\n", 2), ("docs/serving.md", "serve\n", 3),
               ("docs/__pycache__", None, 3), ("docs/guides/serving.md", "serve\n", 4),
               ("LICENSE", "license\n", 5), ("notes/plan.md", "plan again\n", 6)]
    for page, text, runs in changes:
        if page and text is None:
            (tmp_path / page).mkdir()
        elif page:
            (tmp_path / page).write_text(text)
        assert run() == runs
        assert run() == runs

def test_a_row_runs_again_only_where_a_module_comes_or_goes_beside_one_its_tests_executed(
        ci, tmp_path, monkeypatch, cache_worker):
    """A train that loses a head loses the test modules and modules the head added, and its other
    files keep their rows (the land queue's recompose after a pull): a test module, or a module in a
    package no test imported, added or taken away runs no other file again. A module added beside
    one a file's tests executed under src/, which an import there may now find, runs that file
    again, and only that one, and so does its removal, and so does an __init__.py added to, or taken
    from, a namespace package on the way to one; a top-level package added under src/, which may
    shadow any import, runs every file."""
    # src/pkg/space is a namespace package on the way to pkg.space.sub.used: its import executes no file.
    for name, source in {"src/pkg/__init__.py": "", "src/pkg/space/sub/__init__.py": "",
                         "src/pkg/space/sub/used.py": "def value(): return 1\n",
                         "src/other/__init__.py": "", "src/other/lone.py": "def value(): return 2\n",
                         "src/idle/__init__.py": ""}.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(source)
    written(tmp_path, {
        # Imported as it runs, not as it is collected, which every file of the task shares.
        f"test_{name}.py": "import sys\nsys.path.insert(0, 'src')\n"
                           f"def test_{name}():\n"
                           f"    from {package} import value\n"
                           f"    with open('runs-{name}', 'a') as runs: runs.write('x')\n"
                           "    assert value()\n"
        for name, package in (("uses", "pkg.space.sub.used"), ("lone", "other.lone"))})
    interpreter(tmp_path, "3.12")
    url, _, _ = cache_worker
    monkeypatch.setattr(ci, "cpu_class", lambda: "GenuineIntel-6-143")
    cached(monkeypatch, url)
    out = tmp_path / "verdict.json"

    def run():
        ci.task("3.12", ["tests/test_uses.py", "tests/test_lone.py"], "", out, 600.0)
        assert [row["exitCode"] for row in json.loads(out.read_text())["rows"]] == [0, 0]
        return [len((tmp_path / f"runs-{name}").read_text()) for name in ("uses", "lone")]

    changes = [(None, None, [1, 1]), ("tests/test_added.py", "def test_added(): pass\n", [1, 1]),
               ("src/idle/new.py", "def value(): return 3\n", [1, 1]),
               ("src/other/extra.py", "def value(): return 4\n", [1, 2]),
               ("src/other/extra.py", None, [1, 3]), ("tests/test_added.py", None, [1, 3]),
               ("src/pkg/space/__init__.py", "", [2, 3]), ("src/pkg/space/__init__.py", None, [3, 3]),
               ("src/top/__init__.py", "", [4, 4])]
    for path, source, runs in changes:
        if path and source is None:
            (tmp_path / path).unlink()
        elif path:
            (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / path).write_text(source)
        assert run() == runs, path
        assert run() == runs, path


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
    interpreter(tmp_path, "3.12")
    out = tmp_path / "verdict.json"
    began = time.monotonic()
    ci.task("3.12", ["tests/test_quick.py", "tests/test_hangs.py"], "", out, 3.0, grace=2.0)
    # The deadline, the interrupt's grace, one settle budget and the reader's join, with room to spare.
    assert time.monotonic() - began < 3.0 + 2.0 + ci.SETTLE + 10 + 15
    rows = {row["name"]: row for row in json.loads(out.read_text())["rows"]}
    assert {name: row["exitCode"] for name, row in rows.items()} == {
        "3.12:tests/test_quick.py": 1, "3.12:tests/test_hangs.py": 1}


def test_stopping_a_group_spends_one_settle_budget_and_reports_a_leader_that_will_not_reap(monkeypatch):
    """A group is stopped within one settle budget after SIGKILL, the leader's reaping and the other
    members' ending together, and a leader still unreapable then is reported among the members that
    outlived it, never raised: CI writes its verdict either way."""
    groups = load("child_groups", "tools/armada")

    class Stuck:
        pid = 4242

        def poll(self):
            return None

        def wait(self, timeout=None):
            time.sleep(timeout)
            raise subprocess.TimeoutExpired("stuck", timeout)

    monkeypatch.setattr(groups, "signal_group", lambda leader, number: None)
    monkeypatch.setattr(groups, "members", lambda group: [group])
    began = time.monotonic()
    assert groups.stop(Stuck(), settle=0.5) == [4242]
    assert time.monotonic() - began < 0.9, "the leader's wait and the members' share one budget"


def test_ci_gangs_run_every_multihost_case_in_exactly_one_gang(ci, tmp_path):
    """`ci.py gangs` reads the models from test_multihost.py: each gang's -k selects a part of the
    file's collected cases, the parts are disjoint and cover every case, and a model added to
    MODELS makes a gang of its own."""
    source = ROOT / "tests/test_multihost.py"
    items = ci.gangs(source)

    def collected(*select: str) -> set[str]:
        # As a gang's rank collects it: the file skips itself outside a gang.
        done = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", str(source), *select],
                              cwd=ROOT, capture_output=True, text=True, check=True,
                              env={**os.environ, "ARMADA_WORLD": "4", "ARMADA_RANK": "0"})
        return {line for line in done.stdout.splitlines() if "::" in line}

    every = collected()
    parts = [collected("-k", item["select"]) for item in items]
    assert all(item["gang"] == 4 for item in items) and all(parts)
    assert sum(len(part) for part in parts) == len(every) and set().union(*parts) == every
    grown = tmp_path / "test_multihost.py"
    grown.write_text(source.read_text().replace('MODELS = ("dense", "moe")',
                                                'MODELS = ("dense", "moe", "hybrid")'))
    assert [item["select"] for item in ci.gangs(grown)] == [
        "layouts and dense", "layouts and moe", "layouts and hybrid",
        "not (layouts and dense) and not (layouts and moe) and not (layouts and hybrid)"]


def planned_checkout(ci, root: Path, monkeypatch, cache_worker,
                     pinned: str) -> tuple[Path, Callable[[], list[str]]]:
    """Three 3.12 test files, each weighing about a task of its own, run green on CPU class
    GenuineIntel-6-143 against the shared test cache (`cached_checkout`), in a checkout whose .armada.json
    pins its tasks to `pinned` (vendor-family-model), with the cache's read token: where their verdict is
    written, and the plan's 3.12 entries' files, each entry's sorted."""
    verdict = cached_checkout(ci, root, monkeypatch, cache_worker)
    vendor, family, model = pinned.split("-")
    config = json.loads((ROOT / ".armada.json").read_text())
    config["task"]["cpu"] = {"vendor": vendor, "family": int(family), "model": int(model)}
    (root / ".armada.json").write_text(json.dumps(config))
    names = ("a", "b", "c")
    written(root, {f"test_{name}.py": "from pathlib import Path\n"
                                     f"def test_{name}():\n"
                                     f"    with Path('runs-{name}').open('a') as runs: runs.write('x')\n"
                   for name in names})
    (root / "tests/test_durations.json").write_text(json.dumps(
        {f"tests/test_{name}.py::test_{name}": 250.0 for name in names}))
    monkeypatch.setenv("DEW_XLA_CACHE_READ_TOKEN", "token")

    def planned() -> list[str]:
        return sorted(" ".join(sorted(entry["tests"].split())) for entry in ci.plan(300.0, {})
                      if entry["python"] == "3.12")

    return verdict, planned


def test_a_run_plans_only_the_files_whose_inputs_changed_and_its_verdict_names_every_file(
        ci, tmp_path, monkeypatch, cache_worker):
    """The plan reads the test cache with its read token: the files whose rows ran green on exactly these
    inputs, on the class .armada.json pins the tasks to, share one task, which looks each row up again in
    its container, and only a changed file plans a task of its own, so containers scale with the change;
    the verdict still names every file, the unchanged ones' rows reused (Main, 2026-10-28: a lane run with
    most files unchanged booted a container a task, each exiting after a lookup)."""
    verdict, planned = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "GenuineIntel-6-143")
    assert planned() == ["tests/test_a.py", "tests/test_b.py", "tests/test_c.py"]
    rows = armada_task(ci, monkeypatch, verdict)
    assert len(rows) == 3 and all(row["exitCode"] == 0 for row in rows)
    (tmp_path / "tests/test_c.py").write_text((tmp_path / "tests/test_c.py").read_text() + "\n# changed\n")
    assert planned() == ["tests/test_a.py tests/test_b.py", "tests/test_c.py"]
    rows = armada_task(ci, monkeypatch, verdict)
    assert sorted(row["name"] for row in rows) == ["3.12:tests/test_a.py", "3.12:tests/test_b.py",
                                                   "3.12:tests/test_c.py"]
    assert [len((tmp_path / f"runs-{name}").read_text()) for name in ("a", "b", "c")] == [1, 1, 2]


def test_a_plan_without_build_output_or_fetched_payloads_finds_what_a_task_recorded_with_them(
        ci, tmp_path, monkeypatch, cache_worker):
    """The plan's checkout, clean, keys rows as the task's checkout does, though the task's holds an editable
    install's metadata under src/ (setuptools' dewml.egg-info) and the fetched fixtures' payloads, which the
    planner never downloads: a build's output is no input, and a fetched payload is keyed by the digest its
    manifest pins (Main, 2026-10-28: a one-file change planned 493 tasks, as the planner's src/ listing and
    walk lacked them)."""
    verdict, planned = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "GenuineIntel-6-143")
    payload = tmp_path / "tests/fixtures/model.npz"
    payload.parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests/fixtures/model.npz.fetched.json").write_text(json.dumps({"sha256": "ab" * 32}))
    built = tmp_path / "src/dewml.egg-info"
    built.mkdir(parents=True)
    (built / "PKG-INFO").write_text("Name: dewml\n")
    payload.write_bytes(b"payload")
    armada_task(ci, monkeypatch, verdict)
    payload.unlink()
    (built / "PKG-INFO").unlink()
    built.rmdir()
    (tmp_path / "tests/test_c.py").write_text((tmp_path / "tests/test_c.py").read_text() + "\n# changed\n")
    assert planned() == ["tests/test_a.py tests/test_b.py", "tests/test_c.py"]


def test_a_plan_reuses_nothing_for_tasks_pinned_to_another_class(ci, tmp_path, monkeypatch, cache_worker):
    """The plan looks records up under the class .armada.json pins the tasks to, never its own host's: rows
    recorded on another class plan every file a task of its own, as before."""
    verdict, planned = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "AuthenticAMD-25-17")
    armada_task(ci, monkeypatch, verdict)
    (tmp_path / "tests/test_c.py").write_text((tmp_path / "tests/test_c.py").read_text() + "\n# changed\n")
    assert planned() == ["tests/test_a.py", "tests/test_b.py", "tests/test_c.py"]


def test_whole_suite_tests_run_on_a_whole_suite_run_alone(ci, tmp_path, monkeypatch):
    """A test marked whole_suite runs where nothing is reused (the cache's policy off: the daily run, a
    pushed full-* branch, a train's change to what every test reads) and is set aside where the policy is on
    (a train's or a lane's run of what changed), as a whole-suite cell needs no rerun for a change it does
    not read (owner, 2026-10-27, via OrganicMouse: G03's parallelism matrix marks 11 such cells)."""
    interpreter(tmp_path, "3.12")
    written(tmp_path, {"test_matrix.py": "import pytest\nfrom pathlib import Path\n"
                       "@pytest.mark.whole_suite\ndef test_cell():\n"
                       "    with Path('runs-cell').open('a') as runs: runs.write('x')\n"
                       "def test_contract():\n"
                       "    with Path('runs-contract').open('a') as runs: runs.write('x')\n"})
    (tmp_path / ".verdict").mkdir()
    out = tmp_path / ".verdict/verdict.json"

    def ran(policy: str) -> list[int]:
        rows = armada_task(ci, monkeypatch, out, policy)
        assert [row["exitCode"] for row in rows] == [0]
        return [len(path.read_text()) if path.exists() else 0
                for path in (tmp_path / "runs-cell", tmp_path / "runs-contract")]

    assert ran("on") == [0, 1]
    assert ran("off") == [1, 2]



def proven_step(tmp_path: Path, ref: str, runs: str, statuses: str) -> str:
    """The run .github/actions/proven's step gives, its own script run in bash for a push of `ref`, with a
    GitHub CLI that answers the successful-runs query with `runs` and the commit's statuses query with
    `statuses`, each as its --jq filter gives it."""
    import yaml

    action = yaml.safe_load((ROOT / ".github/actions/proven/action.yml").read_text())
    [step] = action["runs"]["steps"]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "gh").write_text('#!/bin/sh\ncase "$2" in\n  */statuses*) cat "$STATUSES" ;;\n'
                                '  *) cat "$RUNS" ;;\nesac\n')
    (bin_dir / "gh").chmod(0o755)
    (tmp_path / "runs").write_text(runs)
    (tmp_path / "statuses").write_text(statuses)
    output = tmp_path / "output"
    subprocess.run(["bash", "-c", step["run"]], check=True, capture_output=True, text=True,
                   env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
                        "RUNS": str(tmp_path / "runs"), "STATUSES": str(tmp_path / "statuses"),
                        "GITHUB_OUTPUT": str(output),
                        "GITHUB_REPOSITORY": "AshishKumar4/dew", "GITHUB_SHA": "a" * 40, "GITHUB_REF": ref,
                        "GITHUB_SERVER_URL": "https://github.com", "WORKFLOW": "ci.yml"})
    return output.read_text().strip().removeprefix("run=")


def test_a_tip_the_land_queue_graded_green_is_proven_for_mains_push_alone(tmp_path):
    """Main's push run proves nothing again at a tip the land queue graded green, which it posts before main
    moves (a commit status, context land-queue): a train whose CI was red on a row its regrade passed has no
    successful run at its tip, and main's run re-ran the suite for 11 minutes and held the next train's test
    job (Main, 2026-10-27). Only main's push reads it: a lane's push of a commit carrying one runs as ever."""
    status = "4242\n"
    assert proven_step(tmp_path / "main", "refs/heads/main", "", status) == "4242"
    assert proven_step(tmp_path / "lane", "refs/heads/fix/lane", "", status) == ""
    # A successful run proves the commit for any push, as before.
    assert proven_step(tmp_path / "run", "refs/heads/fix/lane", "77\n", "") == "77"


def suite_step(tmp_path: Path, ref: str, proven: str, policy: str, verdict: int) -> list[list[str]]:
    """The armada commands ci.yml's suite step gives, its own script run in bash for a push of `ref` with
    the proven job's output `proven` and the cache's `policy`, beside an armada whose stored verdict at the
    commit exits `verdict` (0 green, 1 red, 2 none): each call's words, a call a list."""
    import yaml

    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    [step] = [step for step in workflow["jobs"]["test"]["steps"]
              if step.get("name") == "Run the suite and the multi-host gangs on armada"]
    script = step["run"].replace("${{ github.sha }}", "a" * 40).replace("${{ github.ref_name }}", ref)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    calls = tmp_path / "calls"
    (bin_dir / "armada").write_text(f'#!/bin/sh\necho "$*" >> {calls}\n'
                                    f'[ "$1" = verdict ] && exit {verdict}\nexit 0\n')
    (bin_dir / "armada").chmod(0o755)
    subprocess.run(["bash", "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=120,
                   env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "PROVEN": proven,
                        "DEW_TEST_CACHE": policy, "GITHUB_REF_NAME": ref, "BACKGROUND_POOL": "",
                        "RUNNER_TEMP": str(tmp_path), "ARMADA_URL": "http://127.0.0.1:9", "ARMADA_TOKEN": ""})
    return [line.split() for line in calls.read_text().splitlines()] if calls.exists() else []


def commands(calls: list[list[str]]) -> list[str]:
    """Each armada call's command (`suite_step`)."""
    return [call[0] for call in calls]


def test_a_proven_push_runs_no_suite_though_its_stored_verdict_is_red(tmp_path):
    """Main's push of a tip the land queue graded green, its own CI run red on a row the regrade passed,
    carries the queue's proof: the suite step submits no suite, whatever the stored verdict or the cache's
    policy (a train that changed what every test reads runs main's push with the cache off); the background
    whole suite (ci/full-whole-*), whose commit main's push proved, runs it as ever, and an unproven push
    with a red verdict too (train 237's review, P1: the proof skipped the gangs alone)."""
    def ran(ref: str, proven: str, policy: str, verdict: int) -> bool:
        calls = suite_step(tmp_path / ref.replace("/", "-") / policy, ref, proven, policy, verdict)
        return "run" in commands(calls)

    assert not ran("main", "4242", "on", 1) and not ran("main", "4242", "off", 1)
    assert ran("ci/full-whole-1", "4242", "off", 0) and ran("fix/lane", "", "on", 1)


def test_the_lint_job_runs_every_gate_on_armada_against_the_merge_base(tmp_path):
    """CI's lint job runs each gate of tools/check_gates on an armada container of its own, given the commit's
    merge-base with main as BASE, so the job lasts as long as its slowest gate rather than their sum on the
    runner (Main, 2026-10-28: the lint job set trains' wall at 35-41 min); a train's gates start at its
    priority."""
    import yaml

    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    [step] = [step for step in workflow["jobs"]["lint"]["steps"]
              if step.get("name") == "Every cheap gate (tools/check.sh), each on armada"]
    gates = [line.split("\t")[0] for line in (ROOT / "tools/check_gates").read_text().splitlines()
             if line and not line.startswith("#")]
    # A checkout whose main moved on past its merge-base, as CI's (fetch-depth 0) holds it.
    checkout = tmp_path / "checkout"
    (checkout / "tools").mkdir(parents=True)
    (checkout / "tools/check_gates").write_text((ROOT / "tools/check_gates").read_text())
    git(checkout, "init", "-q", "-b", "main")
    git(checkout, "add", "-A")
    git(checkout, "commit", "-qm", "base")
    base = git(checkout, "rev-parse", "HEAD").strip()
    git(checkout, "checkout", "-qb", "head")
    git(checkout, "commit", "-q", "--allow-empty", "-m", "a head")
    git(checkout, "checkout", "-q", "main")
    git(checkout, "commit", "-q", "--allow-empty", "-m", "main moves on")
    git(checkout, "update-ref", "refs/remotes/origin/main", "main")
    git(checkout, "checkout", "-q", "head")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "armada").write_text(f'#!/bin/sh\necho "$*" > {tmp_path}/words\ncat > {tmp_path}/items\n')
    (bin_dir / "armada").chmod(0o755)
    for ref, priority in (("fix/lane", []), ("ci/batch-abc-1", ["--priority=10"]),
                          ("ci/stage-abc-1", ["--priority=5"])):
        script = step["run"].replace("${{ github.sha }}", "a" * 40).replace("${{ github.ref_name }}", ref)
        environment = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "GITHUB_REF_NAME": ref}
        ran = subprocess.run(["bash", "-c", script], cwd=checkout, capture_output=True, text=True, timeout=60,
                             env=environment)
        assert ran.returncode == 0, ran.stderr
        assert (tmp_path / "items").read_text().split() == gates
        words = (tmp_path / "words").read_text().split()
        assert words[:2] == ["map", f"--commit={'a' * 40}"] and words[-2:] == [base, "{item}"]
        assert [word for word in words if word.startswith("--priority")] == priority


def test_base_pushes_and_the_background_suite_run_at_a_lanes_whole_suite_priority(tmp_path):
    """A push to base/kernels, a base/<producer> promote and the background whole suite run their suite
    at a lane's whole-suite priority, -10 as verdict.sh's, so they never take containers from a train (10)
    or a lane's precheck or targeted run (0); a train's runs at 10, and a lane's branch at the default
    (Main, 2026-10-28: base pushes ran at 0, outside the lanes' lease ledger)."""
    def priority(ref: str) -> list[str]:
        calls = suite_step(tmp_path / ref.replace("/", "-"), ref, "", "off", 2)
        [run] = [call for call in calls if call[0] == "run"]
        return [word for word in run if word.startswith("--priority")]

    for ref in ("base/kernels", "ci/base-abc-1", "ci/full-whole-1"):
        assert priority(ref) == ["--priority=-10"], ref
    assert priority("ci/batch-abc-1") == ["--priority=10"]
    # A train staged on the testing train's tip runs beside it, below it (owner and Main, 2026-10-28).
    assert priority("ci/stage-abc-1") == ["--priority=5"]
    assert priority("fix/lane") == []
