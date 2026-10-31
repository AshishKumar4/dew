"""Dew's CI plan and task on armada (tools/armada/ci.py), on small test files written here."""

import functools
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest
from armada_support import cached, commit, interpreter, loaded, written
from tool_support import git, load

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def ci(tmp_path, monkeypatch):
    return loaded(tmp_path, monkeypatch)


def gpu_driver(root: Path, monkeypatch) -> None:
    """nvidia-smi as the driver answers it for an 80 GiB GPU, first on PATH."""
    tools = root / "bin"
    tools.mkdir()
    (tools / "nvidia-smi").write_text("#!/bin/sh\necho 81920\n")
    (tools / "nvidia-smi").chmod(0o755)
    monkeypatch.setenv("PATH", f"{tools}:{os.environ['PATH']}")


def armada_task(ci, monkeypatch, out: Path, policy: str = "on", target: float = 300.0) -> list[dict]:
    """The 3.12 tasks the plan makes under DEW_TEST_CACHE=`policy` for tasks of `target` seconds, each run as
    armada starts it: .armada.json's command filled from its plan entry, in .armada.json's environment, which
    the plan's DEW_TEST_CACHE is not part of. Their rows, in the plan's order."""
    command = json.loads((ROOT / ".armada.json").read_text())["task"]["command"]
    monkeypatch.setenv("DEW_TEST_CACHE", policy)
    entries = [entry for entry in ci.plan(target, {}) if entry["python"] == "3.12"]
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
    # A collection runs no test, so its rows count no site moved onto YNNPACK's fusions.
    assert all(row["ynn_switched"] == [] for row in rows.values())


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
    (exit 245, a SIGSEGV, 2026-10-25): the test process CI starts carries no dump timer. A test running
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
    # The session fixture's files in a task of their own, where it reads for every file, and the file whose
    # helper changes below in one of its own, as what a process ran before a file is that file's input too.
    fixtures, alone = ["tests/test_one.py", "tests/test_two.py"], ["tests/test_plain.py"]
    ran = [f"tests/{name}" for name in files if name.startswith("test_") and name != "test_new.py"]
    for tests in ([name for name in ran if name not in fixtures + alone], fixtures, alone):
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
    # A day's tier is the checkout's gpu-marked files, whatever the records say a file reads (ledger 2452);
    # --marked names it for the caller that still passes it (2714); --evidence prints each beside its reads.
    tier = functools.partial(named_tier, ci, monkeypatch, capsys)
    assert tier("--shards=2") == tier("--marked") == ["tests/test_marked.py"]
    assert [line.split("\t")[:2] for line in tier("--evidence", lines=True)] == [
        ["tests/test_marked.py", "1 marked"]]
    # A file gains a GPU test after its record: the checkout's marks choose, not records; a kernel's test is
    # the CPU's, interpreted.
    (tmp_path / "tests/test_plain.py").write_text(
        "import pytest\n@pytest.mark.gpu\ndef test_lowers(): pass\n")
    (tmp_path / "tests/test_new.py").write_text(
        "import pytest\n@pytest.mark.kernel\ndef test_interpreted(): pass\n")
    assert tier() == ["tests/test_marked.py", "tests/test_plain.py"]


def named_tier(ci, monkeypatch, capsys, *flags: str, lines: bool = False) -> list:
    """What `ci.py gpu-tier` with `flags` prints: its files, or its lines where `lines`."""
    monkeypatch.setattr(sys, "argv", ["ci.py", "gpu-tier", *flags])
    capsys.readouterr()
    assert ci.main() == 0
    out = capsys.readouterr().out
    return out.splitlines() if lines else out.split()


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
    inputs reuses its own (the maintainer, 2026-10-25: no CI run reached the published checkpoints' parity
    tests)."""
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



def test_every_row_carries_its_cases_reports_and_a_reused_row_keeps_them(ci, tmp_path, monkeypatch,
                                                                         cache_worker):
    """Each row a task runs carries the sites its file's cases moved onto YNNPACK's fusions (`ynn_switched`),
    and its GPU flows' counts by case (`cells`), from the JUnit properties tests/conftest.py writes, green
    rows included, empty where its cases report none; a green row a later task reuses keeps them as it ran."""
    sites = [{"label": "a site over one key", "refused": "Dot with multiple contracting dimensions"}]
    counts = {"passed": 2, "failed": 0, "skipped": 1, "device": "cpu"}
    written(tmp_path, {
        "test_gpu_band.py": "import json\nfrom pathlib import Path\n"
                            "def test_the_band_runs(record_property):\n"
                            "    with Path('runs').open('a') as runs: runs.write('x')\n"
                            f"    record_property('ynn_switched', json.dumps({sites!r}))\n"
                            f"    record_property('cells', json.dumps({counts!r}))\n"
                            "def test_still(): pass\n",
        "test_plain.py": "def test_plain(): pass\n"})
    out = cached_checkout(ci, tmp_path, monkeypatch, cache_worker)

    def fields() -> dict[str, tuple]:
        return {row["name"]: (row["ynn_switched"], row["cells"]) for row in armada_task(ci, monkeypatch, out)}

    node = "tests/test_gpu_band.py::test_the_band_runs"
    expected = {"3.12:tests/test_gpu_band.py": ([{"node": node, **sites[0]}], {node: counts}),
                "3.12:tests/test_plain.py": ([], {})}
    assert fields() == expected
    assert fields() == expected
    assert (tmp_path / "runs").read_text() == "x", "the second task reused the rows"


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
    is an input of the files that read it alone (the maintainer, 2026-10-28: 5 of 10 trains ran with the cache
    off, and 0.75% of rows were reused). The recorder (executed.py, followed.py, watched.py) stays every
    file's input: an old record may lack a dependency a new recorder finds."""
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
    main_ci.write_text(json.dumps({"sha": base, "part": "all", "rows": [
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
    the record was every file's input; the maintainer, 2026-10-28: each train's fold of main's durations
    re-keyed every split file, row 2000)."""
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


def test_a_watch_the_kernel_cannot_hold_whole_is_blind_and_keys_nothing(tmp_path):
    """A tree with more directories than inotify will watch (fs.inotify.max_user_watches; here a user
    namespace's own limit, lowered to 2) leaves the open watch blind: closing it gives no opened files, so
    no record keeps a key, and nothing raises (the maintainer, 2026-10-28: a failed watch left the reader
    unset, and its close crashed)."""
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
    the change from main (the maintainer, 2026-10-28: a workflow change turned the cache off for the whole
    suite)."""
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
    changed, and a file that read none is reused (row 1998; the maintainer, 2026-10-28: a row green on 4 vCPUs
    was reused on 8)."""
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


def test_a_fixture_reruns_only_the_files_that_opened_listed_or_probed_it(ci, tmp_path, monkeypatch,
                                                                         cache_worker):
    """A fixture's arrival reruns the files whose runs opened it, listed its directory, or asked whether it
    exists (`Path.is_file`, `os.path.exists`: recorded as a read, in a test, at collection or in a child it
    started), and no other: an existence probe is a read the record keeps, so a fixture's name is no other
    file's input (the maintainer, 2026-10-29: train 277 reused 2 of 1967 rows once a head added fixtures;
    train 253's review: a test asserting a fixture's absence must not be reused once it comes)."""
    fixtures = tmp_path / "tests/fixtures"
    (fixtures / "model").mkdir(parents=True)
    (fixtures / "model/config.json").write_text("{}\n")
    (fixtures / "other").mkdir()
    (fixtures / "other/data.json").write_text("{}\n")
    probe = "import os; os.path.exists('tests/fixtures/model/tokenizer.json')"
    sources = {
        "opens": "    Path('tests/fixtures/model/config.json').read_text()\n",
        "probes": "    Path('tests/fixtures/model/weights.bin').is_file()\n",
        "collects": "",
        "child": "    import subprocess, sys\n"
                 f"    subprocess.run([sys.executable, '-c', {probe!r}], check=True)\n",
        "lists": "    [entry.name for entry in Path('tests/fixtures').iterdir()]\n",
        "other": "    Path('tests/fixtures/other/data.json').read_text()\n"}
    for name, body in sources.items():
        found = "[name for name in ('weights.bin',) if Path('tests/fixtures/model', name).exists()]"
        cases = f"{found} or ['none']" if name == "collects" else "['one']"
        (tmp_path / f"tests/test_{name}.py").write_text(
            f"import pytest\nfrom pathlib import Path\n@pytest.mark.parametrize('case', {cases})\n"
            f"def test_{name}(case):\n    with Path('runs-{name}').open('a') as runs: runs.write('x')\n"
            + body)
    run = counted_runs(ci, tmp_path, monkeypatch, cache_worker, tuple(sources))
    commit(tmp_path, "base")
    assert run() == [1] * 6
    assert run() == [1] * 6
    (fixtures / "model/weights.bin").write_text("w\n")
    (fixtures / "model/tokenizer.json").write_text("{}\n")
    git(tmp_path, "add", "tests/fixtures")
    commit(tmp_path, "fixtures arrive beside a model's config")
    # Its probers rerun, at their test, their collection and their child; opener, lister and other do not.
    assert run() == [1, 2, 2, 2, 1, 1]
    (fixtures / "later").mkdir()
    (fixtures / "later/config.json").write_text("{}\n")
    git(tmp_path, "add", "tests/fixtures")
    commit(tmp_path, "a model's fixtures arrive")
    # A new model's directory reruns what listed the directory it arrived in, and nothing else.
    assert run() == [1, 2, 2, 2, 2, 1]


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
    runs outside every test process), another test module, or a change to a fixture it never opens (its
    openers' input alone); it runs again once what its test process runs or reads changes: a plugin loaded
    into it, a Python file added that can change what an import finds (a conftest.py, a top-level package
    under src/), or the process group it runs as."""
    (tmp_path / "tests/test_kept.py").write_text(
        "from pathlib import Path\n"
        "def test_kept():\n"
        "    with Path('runs').open('a') as runs: runs.write('x')\n")
    armada = tmp_path / "tools/armada"
    armada.mkdir(parents=True)
    for tool in (ROOT / "tools/armada").glob("*.py"):
        (armada / tool.name).write_text(tool.read_text())
    (tmp_path / "tests/fixtures").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests/fixtures/data.json").write_text("{}\n")
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
    """A data file a reader opens or probes, and code it runs, is the input of every test file its pytest
    process ran after it, not only the one that ran it: a session fixture's lazy reader, a process-lived
    cache (dew.data.text's load_tokenizer, which asks after a tokenizer's optional files), or a module
    imported inside a test, run under one file serves the next without running again, so a change to the
    data, an optional file's arrival or a change to the module reruns both (train 239's review, row 1987;
    the CI bundle's review; row 2228: 256 test files hold ~2,300 imports inside tests)."""
    (tmp_path / "tests/fixtures").mkdir(parents=True)
    (tmp_path / "tests/fixtures/shared.json").write_text("[1]\n")
    (tmp_path / "tests/lazy_values.py").write_text("VALUE = 1\n")
    (tmp_path / "tests/conftest.py").write_text(
        "import functools, pytest, sys\nfrom pathlib import Path\nsys.path.insert(0, 'tests')\n"
        "@pytest.fixture(scope='session')\ndef vocabulary():\n"
        "    return functools.cache(lambda: Path('tests/fixtures/shared.json').read_text()\n"
        "                           + str(Path('tests/fixtures/template.jinja').exists()))\n")
    body = ("from pathlib import Path\ndef test_{0}(vocabulary):\n"
            "    with Path('runs-{0}').open('a') as runs: runs.write('x')\n    assert vocabulary()\n"
            "    import lazy_values\n    assert lazy_values.VALUE\n")
    written(tmp_path, {"test_a.py": body.format("a"), "test_b.py": body.format("b")})
    packed(tmp_path, ["test_a.py", "test_b.py"])
    run = counted_runs(ci, tmp_path, monkeypatch, cache_worker, ("a", "b"))

    assert run() == [1, 1]
    assert run() == [1, 1]
    (tmp_path / "tests/fixtures/shared.json").write_text("[2]\n")
    assert run() == [2, 2]
    (tmp_path / "tests/fixtures/template.jinja").write_text("{{ x }}\n")
    assert run() == [3, 3]
    (tmp_path / "tests/lazy_values.py").write_text("VALUE = 2\n")
    assert run() == [4, 4]


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

def test_a_tool_that_comes_or_goes_runs_no_file_again_that_imported_none(tmp_path, monkeypatch, cache_worker):
    """A tool added under tools/, or taken away, runs no file again whose tests imported no tool: the harness
    its test process runs, tools/armada's recorder and plugins, is every row's input through its own files
    (`SHARED`), not through tools/'s entries, which only a test that imports a tool by its name reads (the
    maintainer, 2026-10-29: train 296 added tools/fla_float64.py and reused 1 of 1795 rows)."""
    # The harness as a checkout holds it, under the checkout's root, beside the project it reads.
    shutil.copytree(ROOT / "tools/armada", tmp_path / "tools/armada",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    module = load("ci", str(tmp_path / "tools/armada"))
    monkeypatch.chdir(tmp_path)
    for name in ("DEW_XLA_CACHE_URL", "DEW_XLA_CACHE_TOKEN", "DEW_XLA_CACHE_READ_TOKEN", "DEW_TEST_CACHE"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "tests").mkdir()
    written(tmp_path, {"test_a.py": "def test_a():\n    with open('runs', 'a') as runs: runs.write('x')\n"})
    run = cached_runs(module, tmp_path, monkeypatch, cache_worker)

    assert run() == 1
    assert run() == 1
    (tmp_path / "tools/new_tool.py").write_text("x = 1\n")
    assert run() == 1
    (tmp_path / "tools/new_tool.py").unlink()
    assert run() == 1


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
                     *pinned: str) -> tuple[Path, Callable[[], list[str]]]:
    """Three 3.12 test files, each weighing about a task of its own, run green on CPU class
    GenuineIntel-6-143 against the shared test cache (`cached_checkout`), in a checkout whose .armada.json
    pins its tasks to `pinned` (vendor-family-model, one or several), with the cache's read token: where
    their verdict is written, and the plan's 3.12 entries' files, each entry's sorted."""
    verdict = cached_checkout(ci, root, monkeypatch, cache_worker)
    classes = [{"vendor": vendor, "family": int(family), "model": int(model)}
               for vendor, family, model in (each.split("-") for each in pinned)]
    config = json.loads((ROOT / ".armada.json").read_text())
    config["task"]["cpu"] = classes[0] if len(classes) == 1 else classes
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
        return sorted(" ".join(sorted(ci.kept_of(entry["tests"].split())[0])) for entry in ci.plan(300.0, {})
                      if entry["python"] == "3.12")

    return verdict, planned


def test_a_run_plans_only_the_files_whose_inputs_changed_and_its_verdict_names_every_file(
        ci, tmp_path, monkeypatch, cache_worker):
    """The plan reads the test cache with its read token: the files whose rows ran green on exactly these
    inputs, on the class .armada.json pins the tasks to, share one task, which looks each row up again in
    its container, and only a changed file plans a task of its own, so containers scale with the change;
    the verdict still names every file, the unchanged ones' rows reused (the maintainer, 2026-10-28: a lane
    run with most files unchanged booted a container a task, each exiting after a lookup)."""
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


SCRIPT = str(ROOT / "tools/armada/ci.py")


def ran_by_script(ci, root: Path, cache_worker, names: tuple[str, ...], module: str) -> dict[str, str]:
    """Test files `names` that import src/shapes.py (`module` its text), each weighing about a task, run
    green by tools/armada/ci.py as a script, as one 3.12 task against the shared test cache, in a checkout
    whose .armada.json pins its tasks to this host's class: the environment a plan of theirs runs in, which
    names the cache and its tokens."""
    interpreter(root, "3.12")
    held = ci.cpu_class().split("-")
    config = json.loads((ROOT / ".armada.json").read_text())
    config["task"]["cpu"] = {"vendor": held[0], "family": int(held[1]), "model": int(held[2])}
    (root / ".armada.json").write_text(json.dumps(config))
    (root / "src").mkdir()
    (root / "src/shapes.py").write_text(module)
    written(root, {f"test_{name}.py": "import sys\nsys.path.insert(0, 'src')\nfrom shapes import area\n"
                                      f"def test_{name}():\n    assert area(2) == 2\n" for name in names})
    (root / "tests/test_durations.json").write_text(json.dumps(
        {f"tests/test_{name}.py::test_{name}": 250.0 for name in names}))
    (root / ".verdict").mkdir()
    url, _, _ = cache_worker
    env = {**os.environ, "DEW_XLA_CACHE_URL": url, "DEW_XLA_CACHE_TOKEN": "token",
           "DEW_XLA_CACHE_READ_TOKEN": "token"}
    verdict = root / ".verdict/verdict.json"
    tests = "--tests=" + " ".join(f"tests/test_{name}.py" for name in names)
    subprocess.run([sys.executable, SCRIPT, "task", "--python=3.12", tests, f"--out={verdict}", "--cache=on"],
                   cwd=root, env=env, check=True)
    assert [row["exitCode"] for row in json.loads(verdict.read_text())["rows"]] == [0] * len(names)
    return env


def test_a_plan_run_by_another_python_than_its_tasks_keeps_the_files_they_ran_green(
        ci, tmp_path, cache_worker):
    """The plan and a task each run tools/armada/ci.py as a script under whatever Python runs it where it runs
    (a planner's machine, a GitHub runner, a task's image), and key a file's run alike: the files a task ran
    green under one Python, a plan run by another keeps, in one task (2026-10-30: plans under 3.14 and 3.12
    found none of the 3.13 tasks' records fresh, as each Python compiles a module's units to its own digests,
    and every file took a task of its own)."""
    # A module whose units each Python compiles to its own digests: a documented, annotated function.
    env = ran_by_script(ci, tmp_path, cache_worker, ("a", "b"),
                        'def area(width: int, height: int = 1) -> int:\n    """An area."""\n'
                        '    return width * height\n')

    def version(python: str) -> str:
        return subprocess.run([python, "-c", "import sys; print(sys.version_info[:2])"], capture_output=True,
                              text=True, check=True).stdout

    # The task under this Python; the plan under another the image holds: a venv of another tested Python, or
    # the image's own.
    other = next(python for python in (*(str(ROOT / f".venv-{each}/bin/python") for each in ci.PYTHONS),
                                       shutil.which("python3") or "")
                 if Path(python).is_file() and version(python) != version(sys.executable))
    timings = tmp_path / "timings.json"
    planned = subprocess.run([other, SCRIPT, "plan", "--target=300", f"--timings={timings}"], cwd=tmp_path,
                             env=env, capture_output=True, text=True, check=True).stdout
    entries = [entry for entry in json.loads(planned)["include"] if entry["python"] == "3.12"]
    assert [sorted(ci.kept_of(entry["tests"].split())[1]) for entry in entries] == [
        [("tests/test_a.py", ""), ("tests/test_b.py", "")]]


def test_a_plan_reads_each_input_once_however_many_rows_ran_it(ci, tmp_path, cache_worker):
    """The plan keys every row it looks up from one reading of the checkout: a module three files' runs
    executed is read once, as the plan runs in CI, under the Python its keys are hashed by (2026-10-31:
    train 368's plan read each of a row's ~3,000 inputs again for each of 1,682 rows, 9.5 minutes before its
    first task)."""
    names = ("a", "b", "c")
    env = ran_by_script(ci, tmp_path, cache_worker, names,
                        "def area(width, height=1):\n    return width * height\n")
    # The plan as CI's runs it, under the Python its keys are hashed by, each file it opens counted.
    opened = tmp_path / ".verdict/opened.json"
    counted = ("import json, runpy, sys\nopened = []\n"
               "sys.addaudithook(lambda event, args: event == 'open' and opened.append(str(args[0])))\n"
               f"sys.argv = [{SCRIPT!r}, 'plan', '--target=300', '--timings={tmp_path / 'timings.json'}']\n"
               f"try:\n    runpy.run_path({SCRIPT!r}, run_name='__main__')\n"
               f"finally:\n    open({str(opened)!r}, 'w').write(json.dumps(opened))\n")
    planned = subprocess.run(["uv", "run", "--no-project", "--python", ci.HASHED_BY, "python", "-c", counted],
                             cwd=tmp_path, env=env, capture_output=True, text=True)
    entries = [entry for entry in json.loads(planned.stdout)["include"] if entry["python"] == "3.12"]
    assert [sorted(ci.kept_of(entry["tests"].split())[1]) for entry in entries] == [
        [(f"tests/test_{name}.py", "") for name in names]]
    assert json.loads(opened.read_text()).count("src/shapes.py") == 1


def test_a_plan_without_build_output_or_fetched_payloads_finds_what_a_task_recorded_with_them(
        ci, tmp_path, monkeypatch, cache_worker):
    """The plan's checkout, clean, keys rows as the task's checkout does, though the task's holds an editable
    install's metadata under src/ (setuptools' dewml.egg-info) and the fetched fixtures' payloads, which the
    planner never downloads: a build's output is no input, and a fetched payload is keyed by the digest its
    manifest pins (the maintainer, 2026-10-28: a one-file change planned 493 tasks, as the planner's src/
    listing and walk lacked them)."""
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


def test_a_row_that_read_its_installs_metadata_is_reused_by_a_planner_without_it(
        ci, tmp_path, monkeypatch, cache_worker):
    """A run that read its editable install's metadata (setuptools' <name>.egg-info, which importlib.metadata
    reads for a version, and which a task's environment writes into its checkout) keys no row on it: the
    environment's build output is the locks' (`shared`), and a planner, whose checkout has none, reuses the
    row (train 300: every record held src/dewml.egg-info's files, and its plan reused none of 501)."""
    verdict, planned = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "GenuineIntel-6-143")
    metadata = tmp_path / "src/dewml.egg-info/PKG-INFO"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("Version: 1\n")
    (tmp_path / "tests/test_a.py").write_text(
        "from pathlib import Path\ndef test_a():\n"
        "    assert Path('src/dewml.egg-info/PKG-INFO').read_text()\n"
        "    with Path('runs-a').open('a') as runs: runs.write('x')\n")
    armada_task(ci, monkeypatch, verdict)
    shutil.rmtree(metadata.parent)
    assert planned() == ["tests/test_a.py tests/test_b.py tests/test_c.py"]


def test_a_plan_reuses_nothing_for_tasks_pinned_to_another_class(ci, tmp_path, monkeypatch, cache_worker):
    """The plan looks records up under the class .armada.json pins the tasks to, never its own host's: rows
    recorded on another class plan every file a task of its own, as before."""
    verdict, planned = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "AuthenticAMD-25-17")
    armada_task(ci, monkeypatch, verdict)
    (tmp_path / "tests/test_c.py").write_text((tmp_path / "tests/test_c.py").read_text() + "\n# changed\n")
    assert planned() == ["tests/test_a.py", "tests/test_b.py", "tests/test_c.py"]


def test_a_plan_pinned_to_several_classes_reuses_a_row_green_on_any_of_them(
        ci, tmp_path, monkeypatch, cache_worker):
    """Tasks pinned to several CPU classes (.armada.json's task.cpu a list) land on any of them, so the plan
    reuses a file whose row ran green on these inputs on any pinned class, and plans the rest: rows split by
    class, the plan reading each pinned class's (the maintainer, 2026-10-29: the suite widened to 25/17 and
    26/17, the fleet's own classes, as one class starved lanes' verdicts)."""
    verdict, planned = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "AuthenticAMD-25-17",
                                        "GenuineIntel-6-143")
    armada_task(ci, monkeypatch, verdict)
    (tmp_path / "tests/test_c.py").write_text((tmp_path / "tests/test_c.py").read_text() + "\n# changed\n")
    assert planned() == ["tests/test_a.py tests/test_b.py", "tests/test_c.py"]


def test_a_row_run_green_on_these_inputs_is_reused_though_another_commits_run_of_its_file_came_after(
        ci, tmp_path, monkeypatch, cache_worker):
    """A file's row run green on exactly a checkout's inputs is reused there whatever runs of the file on
    other inputs came after it, as lanes' checks and trains of other commits run it all day: a lane's one-line
    change over a green verdict reruns only what that line reaches (owner, 2026-10-29: a verdict found 272 of
    its own 504 green files reusable an hour later, the others' records replaced by other commits' runs)."""
    verdict, planned = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "GenuineIntel-6-143")
    armada_task(ci, monkeypatch, verdict)
    test_c = tmp_path / "tests/test_c.py"
    first = test_c.read_text()
    # Another commit's run of test_c, on other inputs, after this one's.
    test_c.write_text(first + "\n# another commit\n")
    armada_task(ci, monkeypatch, verdict)
    test_c.write_text(first)
    assert planned() == ["tests/test_a.py tests/test_b.py tests/test_c.py"]
    rows = armada_task(ci, monkeypatch, verdict)
    assert len(rows) == 3 and all(row["exitCode"] == 0 for row in rows)
    assert [len((tmp_path / f"runs-{name}").read_text()) for name in "abc"] == [1, 1, 2]


def test_a_plan_keeps_each_green_group_of_a_heavy_file_and_a_task_runs_a_group_whose_green_run_is_gone(
        ci, tmp_path, monkeypatch, cache_worker):
    """A file heavier than the target runs as groups, each a row the plan looks up as it does a whole file's:
    once every group ran green on these inputs, one task holds them with the light files and reuses each
    row, running nothing; a kept group whose green run its task cannot read, it runs (2026-10-30: the plan
    looked up whole files alone, and a docs-only change's 1,034 green groups of heavy files each booted a
    container to read its record)."""
    verdict, _ = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "GenuineIntel-6-143")
    written(tmp_path, {"test_c.py": "from pathlib import Path\n" + "".join(
        f"def test_{name}():\n    with Path('runs-{name}').open('a') as runs: runs.write('x')\n"
        for name in "cd")})
    (tmp_path / "tests/test_durations.json").write_text(json.dumps(
        {**{f"tests/test_{name}.py::test_{name}": 250.0 for name in "ab"},
         "tests/test_c.py::test_c": 250.0, "tests/test_c.py::test_d": 250.0}))

    def planned() -> list[list[str]]:
        return sorted(sorted(entry["rows"]) for entry in ci.plan(300.0, {}) if entry["python"] == "3.12")

    heavy = ["3.12:tests/test_c.py#1/2", "3.12:tests/test_c.py#2/2"]
    assert planned() == [["3.12:tests/test_a.py"], ["3.12:tests/test_b.py"], heavy[:1], heavy[1:]]
    assert all(row["exitCode"] == 0 for row in armada_task(ci, monkeypatch, verdict))
    assert planned() == [["3.12:tests/test_a.py", "3.12:tests/test_b.py", *heavy]]
    rows = armada_task(ci, monkeypatch, verdict)
    assert sorted(row["name"] for row in rows) == ["3.12:tests/test_a.py", "3.12:tests/test_b.py", *heavy]
    assert [len((tmp_path / f"runs-{name}").read_text()) for name in "abcd"] == [1, 1, 1, 1]
    # A kept group whose green run its task cannot read, its file changed since the plan: the task runs that
    # group alone.
    test_c = tmp_path / "tests/test_c.py"
    test_c.write_text(test_c.read_text() + "\n# changed after the plan\n")
    monkeypatch.setattr(sys, "argv", ["ci.py", "task", "--python=3.12", "--tests=tests/test_c.py#2/2@"
                                      "GenuineIntel-6-143:0000", f"--out={verdict}", "--cache=on"])
    assert ci.main() == 0
    ran = json.loads(verdict.read_text())["rows"]
    assert [(row["name"], row["exitCode"]) for row in ran] == [(heavy[1], 0)]
    assert [len((tmp_path / f"runs-{name}").read_text()) for name in "cd"] == [1, 2]


def test_a_plan_spreads_its_kept_rows_over_tasks_of_the_targets_lookups(ci, tmp_path, monkeypatch,
                                                                        cache_worker):
    """The plan's kept rows fill tasks of at most the target's lookups, a lookup a row in seconds, each of
    which reuses its rows, so no kept task outlasts the rest of the run and none carries more tokens than a
    process starts with (2026-10-31: main de315f9a05's kept rows were one task, whose 149,515-byte --tests
    argument started no task, and its run was not graded)."""
    verdict, _ = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "GenuineIntel-6-143")
    assert all(row["exitCode"] == 0 for row in armada_task(ci, monkeypatch, verdict))
    names = ["3.12:tests/test_a.py", "3.12:tests/test_b.py", "3.12:tests/test_c.py"]
    entries = [entry for entry in ci.plan(2.0, {}) if entry["python"] == "3.12"]
    assert [(entry["rows"], entry["weight"]) for entry in entries] == [(names[:2], 2.0), (names[2:], 1.0)]
    rows = armada_task(ci, monkeypatch, verdict, target=2.0)
    assert [(row["name"], row["exitCode"]) for row in rows] == [(name, 0) for name in names]
    assert [len((tmp_path / f"runs-{name}").read_text()) for name in "abc"] == [1, 1, 1]


def test_a_kept_tasks_entry_stays_within_what_a_process_starts_with_however_high_the_target(
        ci, tmp_path, monkeypatch, cache_worker):
    """However many lookups the target allows, a kept task's entry, which armada hands its task whole as one
    environment string (ARMADA_ITEM), and its tokens, one argument (--tests), stay within the 131,071 bytes
    Linux starts a process with, and every kept row is reused (2026-10-31: main de315f9a05's kept rows were
    one task, whose 149,515-byte argument started none)."""
    verdict, _ = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "GenuineIntel-6-143")
    # Files of long names (beside a fetched fixture's manifest name, within a file system's 255 bytes), enough
    # that their kept rows' tokens and names pass what a process starts with together.
    names = [f"tests/test_{index:03d}{'x' * 200}.py" for index in range(300)]
    written(tmp_path, {Path(name).name: f"def test_{index}():\n    pass\n"
                       for index, name in enumerate(names)})
    durations = tmp_path / "tests/test_durations.json"
    durations.write_text(json.dumps({**json.loads(durations.read_text()),
                                     **{f"{name}::test_{index}": 1.0 for index, name in enumerate(names)}}))
    assert all(row["exitCode"] == 0 for row in armada_task(ci, monkeypatch, verdict, target=100_000.0))
    entries = [entry for entry in ci.plan(100_000.0, {}) if entry["python"] == "3.12"]
    every = sorted(f"3.12:{name}" for name in [*names, *(f"tests/test_{letter}.py" for letter in "abc")])
    assert sorted(row for entry in entries for row in entry["rows"]) == every
    assert all(ci.KEPT_AT in token for entry in entries for token in entry["tests"].split())
    assert max(len(f"ARMADA_ITEM={json.dumps(entry)}".encode()) for entry in entries) <= 131_071
    assert max(len(f"--tests={entry['tests']}".encode()) for entry in entries) <= 131_071
    rows = armada_task(ci, monkeypatch, verdict, target=100_000.0)
    assert sorted((row["name"], row["exitCode"]) for row in rows) == [(name, 0) for name in every]
    assert [len((tmp_path / f"runs-{name}").read_text()) for name in "abc"] == [1, 1, 1]


def test_a_kept_task_whose_deadline_runs_out_names_its_unread_groups_red_and_runs_none_past_it(
        ci, tmp_path, monkeypatch, cache_worker):
    """A kept task that must run a kept group whose green run it cannot read, past its deadline, reads and
    runs no group after it: each is a red row its task's deadline cut off, and the task writes every row
    (review: each later group was still read and run, a fresh grace and settle each, past the deadline)."""
    verdict, _ = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "GenuineIntel-6-143")
    _, _, seen = cache_worker
    names = "cde"
    written(tmp_path, {"test_c.py": "import time\nfrom pathlib import Path\n" + "".join(
        f"def test_{name}():\n    Path('runs-{name}').write_text('x')\n    time.sleep(3)\n"
        for name in names)})
    (tmp_path / "tests/test_durations.json").write_text(json.dumps(
        {f"tests/test_c.py::test_{name}": 250.0 for name in names}))
    groups = " ".join(f"tests/test_c.py#{index}/3@GenuineIntel-6-143:0000" for index in (1, 2, 3))
    monkeypatch.setattr(sys, "argv", ["ci.py", "task", "--python=3.12", f"--tests={groups}",
                                      f"--out={verdict}", "--cache=on", "--deadline=1"])
    seen.clear()
    assert ci.main() == 0
    rows = json.loads(verdict.read_text())["rows"]
    assert [(row["name"], row["exitCode"], row["kind"]) for row in rows] == [
        (f"3.12:tests/test_c.py#{index}/3", 1, "deadline") for index in (1, 2, 3)]
    assert [(tmp_path / f"runs-{name}").exists() for name in names] == [True, False, False]
    # The groups past the deadline were never looked up.
    assert [any(f"tests/test_c.py/{index}-of-3" in path for path in seen) for index in (1, 2, 3)] == [
        True, False, False]


def test_a_kept_task_reuses_its_rows_after_another_commit_changes_the_recorded_dependencies(
        ci, tmp_path, monkeypatch, cache_worker):
    """A plan's kept task reuses the green rows the plan found though another commit's run, of other inputs
    (a file importing a helper this commit's never did), replaced their files' last records before the task
    started: the plan hands the task what it found, never a record another run may move (2026-10-30: a
    verdict's kept task found 23 of its 259 files, ran the rest itself and timed out)."""
    verdict, planned = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "GenuineIntel-6-143")
    (tmp_path / "tests/later_dependency.py").write_text("VALUE = 1\n")
    monkeypatch.setenv("PYTEST_ADDOPTS", "-p no:cacheprovider")
    armada_task(ci, monkeypatch, verdict)
    common = ci.shared("3.12")
    at = ci.record_key("3.12", "3.12:tests/test_a.py", "GenuineIntel-6-143", common)
    before = ci.recorded(at, "token")
    assert planned() == ["tests/test_a.py tests/test_b.py tests/test_c.py"]
    entries = [entry for entry in ci.plan(300.0, {}) if entry["python"] == "3.12"]
    assert len(entries) == 1
    test_a = tmp_path / "tests/test_a.py"
    original = test_a.read_text()
    # Another commit runs between this plan and its task, importing a helper this commit never
    # imported. It shares the common hash, so its run replaces the slot this plan read.
    test_a.write_text("import later_dependency\n" + original)
    armada_task(ci, monkeypatch, verdict)
    test_a.write_text(original)
    assert ci.recorded(at, "token")["key"] != before["key"]
    command = json.loads((ROOT / ".armada.json").read_text())["task"]["command"]
    values = {**entries[0], "out": verdict}
    filled = [re.sub(r"\{(\w+)\}", lambda field: str(values[field[1]]), word) for word in command]
    monkeypatch.setattr(sys, "argv", ["ci.py", *filled[2:]])
    assert ci.main() == 0
    rows = json.loads(verdict.read_text())["rows"]
    assert len(rows) == 3 and all(row["exitCode"] == 0 for row in rows)
    assert [len((tmp_path / f"runs-{name}").read_text()) for name in "abc"] == [2, 1, 1]


def test_a_kept_task_reuses_each_files_own_row_where_files_that_ran_in_one_process_share_a_key(
        ci, tmp_path, monkeypatch, cache_worker):
    """Files one task ran in one process share what it executed before them, so their keys are one: a kept
    task still reuses each file's own green row, and names every file once (2026-10-30: a kept task reported
    one file's row for six files, and a verdict was not graded)."""
    verdict, planned = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "GenuineIntel-6-143")
    (tmp_path / "tests/test_durations.json").write_text(json.dumps(
        {f"tests/test_{name}.py::test_{name}": 50.0 for name in "abc"}))
    armada_task(ci, monkeypatch, verdict)
    assert planned() == ["tests/test_a.py tests/test_b.py tests/test_c.py"]
    rows = armada_task(ci, monkeypatch, verdict)
    assert sorted(row["name"] for row in rows) == ["3.12:tests/test_a.py", "3.12:tests/test_b.py",
                                                   "3.12:tests/test_c.py"]
    assert [len((tmp_path / f"runs-{name}").read_text()) for name in "abc"] == [1, 1, 1]


def test_a_helper_added_under_tests_reruns_only_the_runs_that_imported_a_module_of_its_name(
        ci, tmp_path, monkeypatch, cache_worker):
    """A helper module added under tests/ reruns no file whose run imported no module of its name, as it
    shadows nothing they import; one named as a module they imported, which tests/ leading sys.path would
    shadow, reruns each of them (owner, 2026-10-29: train 315's one new helper reran all 2490 tasks)."""
    verdict, planned = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "GenuineIntel-6-143")
    armada_task(ci, monkeypatch, verdict)
    (tmp_path / "tests/quantized_basis.py").write_text("BASIS = 1\n")
    assert planned() == ["tests/test_a.py tests/test_b.py tests/test_c.py"]
    (tmp_path / "tests/pathlib.py").write_text("")
    assert planned() == ["tests/test_a.py", "tests/test_b.py", "tests/test_c.py"]


def test_a_helper_named_as_a_module_only_a_child_process_imported_reruns_its_file(
        ci, tmp_path, monkeypatch, cache_worker):
    """A module only a test's child process imports (a tool run as a script) is that file's input as one its
    own process imports is: a helper of its name added under tests/ reruns the file (row 2502: the recorder
    named the pytest process's imports alone, so such a helper kept the file's green key)."""
    verdict, planned = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "GenuineIntel-6-143")
    written(tmp_path, {"test_c.py": "import subprocess, sys\nfrom pathlib import Path\n"
                                    "def test_c():\n"
                                    "    with Path('runs-c').open('a') as runs: runs.write('x')\n"
                                    "    subprocess.run([sys.executable, '-c', 'import wave'],\n"
                                    "                   check=True)\n"})
    armada_task(ci, monkeypatch, verdict)
    assert planned() == ["tests/test_a.py tests/test_b.py tests/test_c.py"]
    (tmp_path / "tests/wave.py").write_text("")
    assert planned() == ["tests/test_a.py tests/test_b.py", "tests/test_c.py"]


def test_a_task_on_another_pinned_class_reuses_the_rows_its_plan_kept(
        ci, tmp_path, monkeypatch, cache_worker):
    """The files a plan keeps for reuse, their rows green on any class the tasks are pinned to, are reused by
    the task that holds them on whichever pinned class it lands on, as the plan counted them: a task that
    looked up its own class alone ran every kept file itself, in one task, past its deadline (2026-10-29: a
    verdict's kept task of 182 files on 26/17 ran 76 of them into its 1680 s deadline, the rows green on
    25/17)."""
    verdict, planned = planned_checkout(ci, tmp_path, monkeypatch, cache_worker, "AuthenticAMD-25-17",
                                        "GenuineIntel-6-143")
    armada_task(ci, monkeypatch, verdict)
    (tmp_path / "tests/test_c.py").write_text((tmp_path / "tests/test_c.py").read_text() + "\n# changed\n")
    assert planned() == ["tests/test_a.py tests/test_b.py", "tests/test_c.py"]
    monkeypatch.setattr(ci, "cpu_class", lambda: "AuthenticAMD-25-17")
    rows = armada_task(ci, monkeypatch, verdict)
    assert len(rows) == 3 and all(row["exitCode"] == 0 for row in rows)
    assert [len((tmp_path / f"runs-{name}").read_text()) for name in "abc"] == [1, 1, 2]


def test_whole_suite_tests_run_on_a_whole_suite_run_alone(ci, tmp_path, monkeypatch):
    """A test marked whole_suite runs on the whole suite alone (DEW_WHOLE_SUITE, every fifth publish's
    ci/full-whole-* run), and is set aside on every other run, a train's whatever the cache's policy: a
    policy-off train runs every file, but not a whole-suite cell (the maintainer, 2026-10-28, via Models:
    train 250's policy-off suite ran G03's GRPO cells, 826-1,430 s each, to its deadline)."""
    interpreter(tmp_path, "3.12")
    written(tmp_path, {"test_matrix.py": "import pytest\nfrom pathlib import Path\n"
                       "@pytest.mark.whole_suite\ndef test_cell():\n"
                       "    with Path('runs-cell').open('a') as runs: runs.write('x')\n"
                       "def test_contract():\n"
                       "    with Path('runs-contract').open('a') as runs: runs.write('x')\n"})
    (tmp_path / ".verdict").mkdir()
    out = tmp_path / ".verdict/verdict.json"

    def ran(policy: str, whole: str = "") -> list[int]:
        monkeypatch.setenv("DEW_WHOLE_SUITE", whole)
        rows = armada_task(ci, monkeypatch, out, policy)
        assert [row["exitCode"] for row in rows] == [0]
        return [len(path.read_text()) if path.exists() else 0
                for path in (tmp_path / "runs-cell", tmp_path / "runs-contract")]

    assert ran("on") == [0, 1]
    assert ran("off") == [0, 2]
    assert ran("off", "1") == [1, 3]



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
    job (the maintainer, 2026-10-27). Only main's push reads it: a lane's push of a commit carrying one runs
    as ever."""
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
    runner (the maintainer, 2026-10-28: the lint job set trains' wall at 35-41 min); a train's gates start at
    its priority."""
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


def test_a_cold_suite_is_planned_with_its_cold_cost_and_given_a_long_deadline(ci, tmp_path, monkeypatch):
    """A suite the test cache's policy turns off (a lock, JAX's build or the recorder changed) runs every
    file cold: each task pays the compiles and imports no warm run did, so its weight carries COLD_SECONDS
    beside its estimate, which sizes armada's pool to the fleet rather than to the longest task, and its
    deadline is 5400 s, not 2700 (the maintainer, 2026-10-28: train 254's cold suite, 2,169 tasks planned on
    242 containers, about 470 s a task against an estimated 120, left 780 tasks unstarted at its 2,698 s
    deadline). A warm suite keeps both."""
    for name in ("test_a", "test_b"):
        (tmp_path / "tests" / f"{name}.py").write_text("")
    (tmp_path / "tests/test_durations.json").write_text(json.dumps(
        {"tests/test_a.py::test_x": 30.0, "tests/test_b.py::test_y": 30.0}))
    timings = {"files": {"3.12:tests/test_a.py": 30.0, "3.12:tests/test_b.py": 30.0}}
    monkeypatch.setenv("DEW_TEST_CACHE", "on")
    warm = {entry["tests"]: entry["weight"] for entry in ci.plan(120.0, timings) if entry["python"] == "3.12"}
    monkeypatch.setenv("DEW_TEST_CACHE", "off")
    cold = {entry["tests"]: entry["weight"] for entry in ci.plan(120.0, timings) if entry["python"] == "3.12"}
    assert cold == {tests: weight + ci.COLD_SECONDS for tests, weight in warm.items()}

    def deadline(policy: str) -> list[str]:
        calls = suite_step(tmp_path / f"cold-{policy}", "ci/batch-abc-1", "", policy, 2)
        [run] = [call for call in calls if call[0] == "run"]
        return [word for word in run if word.startswith("--deadline")]

    assert deadline("off") == ["--deadline=5400"] and deadline("on") == ["--deadline=2700"]


def run_options(tmp_path: Path, ref: str, policy: str, option: str) -> list[str]:
    """The armada run's `option` words CI's suite step gives a push of `ref` under `policy` (`suite_step`)."""
    calls = suite_step(tmp_path / ref.replace("/", "-"), ref, "", policy, 2)
    [run] = [call for call in calls if call[0] == "run"]
    return [word for word in run if word.startswith(option)]


def test_a_staged_trains_suite_asks_no_more_than_its_lease(tmp_path):
    """A train staged on the testing train's tip runs its suite on the pool its branch names
    (ci/stage-...-p<N>, the share of its lease the queue's promote.sh leaves the suite), never the whole
    pool, so it asks armada no more than it leased (the maintainer, 2026-10-28); a train on main asks the
    whole pool."""
    assert run_options(tmp_path, "ci/stage-abc-1-p120", "on", "--pool") == ["--pool=120"]
    assert run_options(tmp_path, "ci/batch-abc-1", "on", "--pool") == []


def test_base_pushes_and_the_background_suite_run_at_a_lanes_whole_suite_priority(tmp_path):
    """A push to base/kernels, a base/<producer> promote and the background whole suite run their suite
    at a lane's whole-suite priority, -10 as verdict.sh's, so they never take containers from a train (10)
    or a lane's precheck or targeted run (0); a train's runs at 10, and a lane's branch at the default
    (the maintainer, 2026-10-28: base pushes ran at 0, outside the lanes' lease ledger)."""
    def priority(ref: str) -> list[str]:
        return run_options(tmp_path, ref, "off", "--priority")

    for ref in ("base/kernels", "ci/base-abc-1", "ci/full-whole-1"):
        assert priority(ref) == ["--priority=-10"], ref
    assert priority("ci/batch-abc-1") == ["--priority=10"]
    # A train staged on the testing train's tip runs beside it, below it (owner and the maintainer,
    # 2026-10-28).
    assert priority("ci/stage-abc-1") == ["--priority=5"]
    assert priority("fix/lane") == []


def test_a_test_past_the_group_target_is_flagged_and_a_new_one_fails(ci, tmp_path, monkeypatch, capsys):
    """A test whose own seconds in a task's run pass the group target (.armada.json's `target`) is flagged
    from the verdict's rows, as no cut can bring its group within the target; a whole_suite cell is flagged
    too, marked as running on the whole suite alone, however pytest comes to mark it (its own mark, its
    class's, a parameter set's, a mark held in a name), which the run records in its row (bible §7: every
    test past the target is reported); and with main's durations record, a test it lacks fails the check, a
    cell newly marked whole_suite too, while one main holds already passes (owner, 2026-10-28: a new slow test
    fails its lane's check; row 2073's cell ran 1,540 s)."""
    interpreter(tmp_path, "3.12")
    written(tmp_path, {"test_slow.py": "import time\nimport pytest\ncell = pytest.mark.whole_suite\n"
                       "def test_old(): time.sleep(0.4)\n"
                       "def test_new(): time.sleep(0.6)\n"
                       "def test_quick(): pass\n"
                       "@pytest.mark.whole_suite\ndef test_cell(): time.sleep(0.4)\n"
                       "@cell\ndef test_named(): time.sleep(0.4)\n"
                       "class TestHeld:\n    pytestmark = cell\n    def test_inside(self): time.sleep(0.4)\n"
                       "@pytest.mark.parametrize('n', [1, pytest.param(2, marks=cell, id='big')])\n"
                       "def test_sized(n): time.sleep(0.3 * n)\n"})
    (tmp_path / ".verdict").mkdir()
    out = tmp_path / ".verdict/verdict.json"
    # The whole suite's run, which runs the cells too.
    monkeypatch.setenv("DEW_WHOLE_SUITE", "1")
    verdict = tmp_path / "verdict.json"
    verdict.write_text(json.dumps({"rows": armada_task(ci, monkeypatch, out, "off")}))
    (tmp_path / ".armada.json").write_text(json.dumps({"target": 0.2}))
    base = tmp_path / "base.json"
    held = ("test_old", "test_sized[1]", "test_cell", "test_named", "TestHeld::test_inside")
    base.write_text(json.dumps({f"tests/test_slow.py::{name}": 0.4 for name in held}))

    def slow(*options: str) -> tuple[int, dict[str, bool]]:
        monkeypatch.setattr(sys, "argv", ["ci.py", "slow", str(verdict), *options])
        code = ci.main()
        printed = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
        return code, {item["test"].partition("::")[2]: item.get("whole_suite", False) for item in printed}

    assert slow() == (0, {"test_new": False, "test_old": False, "test_sized[1]": False, "test_cell": True,
                          "test_named": True, "TestHeld::test_inside": True, "test_sized[big]": True})
    assert slow("--new", str(base)) == (1, {"test_new": False, "test_sized[big]": True})


def test_a_train_runs_row_names_the_whole_suite_cells_it_set_aside(ci, tmp_path, monkeypatch):
    """A train's run sets a file's whole_suite cells aside, yet its row names them, as pytest marks them: main
    then holds them though its train CI never times one, so a lane's whole-suite run of a cell main has is no
    new slow test (verdict.sh), while one main has not stays new (2026-10-30: row 2116's two cells failed
    every whole-suite run)."""
    interpreter(tmp_path, "3.12")
    written(tmp_path, {"test_cells.py": "import pytest\ncell = pytest.mark.whole_suite\n"
                       "def test_plain(): pass\n"
                       "@cell\ndef test_cell(): pass\n"
                       "@pytest.mark.parametrize('n', [1, pytest.param(2, marks=cell, id='big')])\n"
                       "def test_sized(n): pass\n"})
    (tmp_path / ".verdict").mkdir()
    rows = armada_task(ci, monkeypatch, tmp_path / ".verdict/verdict.json", "off")
    assert [(sorted(row["tests"]), row.get("whole_suite")) for row in rows] == [
        (["tests/test_cells.py::test_plain", "tests/test_cells.py::test_sized[1]"],
         ["tests/test_cells.py::test_cell", "tests/test_cells.py::test_sized[big]"])]


def a_wheel(root: Path, name: str, version: str) -> Path:
    """A pure-Python wheel of `name` at `version` whose module says its version, as an index serves one."""
    wheel = root / f"{name}-{version}-py3-none-any.whl"
    info = f"{name}-{version}.dist-info"
    files = {f"{name}/__init__.py": f"VERSION = {version!r}\n",
             f"{info}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
             f"{info}/WHEEL": "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\n"
                              "Tag: py3-none-any\n"}
    with zipfile.ZipFile(wheel, "w") as archive:
        for path, text in files.items():
            archive.writestr(path, text)
        recorded = [*files, f"{info}/RECORD"]
        archive.writestr(f"{info}/RECORD", "".join(f"{path},,\n" for path in recorded))
    return wheel


def test_a_release_check_marked_whole_suite_or_network_too_runs_in_its_release_task_alone(
        ci, tmp_path, monkeypatch, cache_worker):
    """A release_env test marked whole_suite or network too runs in its release task, never set aside there
    as the whole suite's or the network lane's cells, and the network lane, which lacks its requirement,
    never runs it (the bundle's review: such a check was planned, deselected by its own task, and reported
    green)."""
    url, _, _ = cache_worker
    gpu_lane_at(tmp_path, monkeypatch, url, 0)
    interpreter(tmp_path, "3.12")
    fetcher(tmp_path)
    pinned = a_wheel(tmp_path, "pinned", "2.0")
    written(tmp_path, {"test_release.py": "import pytest\nfrom pathlib import Path\n" + "".join(
        f"@pytest.mark.release_env({str(pinned)!r})\n@pytest.mark.{also}\n"
        f"def test_{also}_too():\n    import pinned\n    Path('ran-{also}').write_text(pinned.VERSION)\n"
        for also in ("whole_suite", "network"))})
    monkeypatch.setenv("UV_OFFLINE", "1")
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "uv-cache"))
    out = tmp_path / "verdict.json"
    assert ci.gpu_lane("3.12", ["tests/test_release.py"], out, cache="off", processes=1, deadline=600.0,
                       network=True) == 0
    assert not (tmp_path / "ran-network").exists(), "the network lane ran a release check"
    monkeypatch.setattr(sys, "argv", ["ci.py", "release", "--python=3.12", "--tests=tests/test_release.py",
                                      f"--requirement={pinned}", f"--out={tmp_path / 'release.json'}"])
    assert ci.main() == 0
    assert [(tmp_path / f"ran-{also}").read_text() for also in ("whole_suite", "network")] == ["2.0", "2.0"]


@pytest.mark.parametrize("mark", ["release_env()", "release_env(requirement='pinned==2.0')",
                                  "release_env('pinned==2.0', 'other==1.0')", "release_env(2)"])
def test_a_release_mark_without_one_requirement_fails_its_files_collection(ci, tmp_path, monkeypatch, mark):
    """A release_env mark that names no one requirement string (none, a keyword's, two, or not a string)
    fails its file's collection, naming its test: every suite task sets a marked test aside, and no release
    task would plan it, so it would run nowhere (the maintainer's read of 6b48c32ee)."""
    interpreter(tmp_path, "3.12")
    written(tmp_path, {"test_release.py": f"import pytest\n@pytest.mark.{mark}\n"
                       "def test_under_the_release():\n    pass\n"
                       "def test_plain():\n    pass\n"})
    (tmp_path / ".verdict").mkdir()
    [row] = armada_task(ci, monkeypatch, tmp_path / ".verdict/verdict.json", "off")
    assert row["exitCode"] != 0, row["output"]
    assert "tests/test_release.py::test_under_the_release" in row["output"] and "release_env" in row["output"]


def test_a_release_check_runs_in_the_whole_suites_second_job_under_its_requirement(ci, tmp_path, monkeypatch,
                                                                                   capsys):
    """A test marked release_env(<requirement>) checks a release's own code under a package pinned apart from
    the suite's (2173: LFM3's needs transformers>=5.18.0, MiMo-V2's <=5.1.0): every suite task sets it aside,
    as the suite's environment lacks its requirement, and its file's row names it as pytest resolved its mark;
    from a whole suite's rows, its second job plans a task each requirement, whose pytest runs with the
    requirement over the suite's environment (uv's overlay, torch and jax the suite's), under a row of its
    own beside its file's, and exits 1 on a red one."""
    interpreter(tmp_path, "3.12")
    pinned = a_wheel(tmp_path, "pinned", "2.0")
    written(tmp_path, {"test_release.py": "import pytest\n"
                       "@pytest.mark.release_env(" + repr(str(pinned)) + ")\n"
                       "def test_under_the_release():\n    import pinned\n"
                       "    assert pinned.VERSION == '2.0'\n"
                       "def test_plain():\n    pass\n"})
    monkeypatch.setenv("UV_OFFLINE", "1")
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "uv-cache"))
    monkeypatch.setenv("DEW_WHOLE_SUITE", "1")
    (tmp_path / ".verdict").mkdir()
    rows = armada_task(ci, monkeypatch, tmp_path / ".verdict/verdict.json", "off")
    assert [(found["name"], found["exitCode"], found.get("release_env")) for found in rows] == [
        ("3.12:tests/test_release.py", 0, {str(pinned): ["tests/test_release.py::test_under_the_release"]})]
    verdict = tmp_path / "verdict.json"
    verdict.write_text(json.dumps({"rows": rows}))
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["ci.py", "release-plan", str(verdict)])
    assert ci.main() == 0
    [item] = json.loads(capsys.readouterr().out)
    assert item == {"release": str(pinned), "python": "3.12", "tests": "tests/test_release.py"}
    out = tmp_path / ".verdict/release.json"
    monkeypatch.setattr(sys, "argv", ["ci.py", "release", f"--python={item['python']}",
                                      f"--tests={item['tests']}", f"--requirement={item['release']}",
                                      f"--out={out}"])
    assert ci.main() == 0
    assert [(found["name"], found["exitCode"]) for found in json.loads(out.read_text())["rows"]] == [
        (f"3.12:tests/test_release.py#release:{pinned}", 0)]
