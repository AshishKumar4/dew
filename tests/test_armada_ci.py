"""Dew's CI plan and task on armada (tools/armada/ci.py), on small test files written here."""

import itertools
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from tool_support import load

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def ci(tmp_path, monkeypatch):
    module = load("ci", "tools/armada")
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
    for name, source in files.items():
        (tmp_path / "tests" / name).write_text(source)
    install_compiler(tmp_path / "tests/compiler")
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
    monkeypatch.setenv("DEW_XLA_CACHE_URL", url)
    monkeypatch.setenv("DEW_XLA_CACHE_TOKEN", "token")
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
    for name, source in files.items():
        (tmp_path / "tests" / name).write_text(source)
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes/page.md").write_text("page\n")
    venv = tmp_path / ".venv-3.12/bin"
    venv.mkdir(parents=True)
    (venv / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (venv / "python").chmod(0o755)
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


def test_a_gpu_lane_runs_each_file_in_a_process_of_its_own_within_its_memory_share(
        ci, tmp_path, monkeypatch, cache_worker):
    """The GPU lane runs each file once, in a process of its own. A file no
    run has measured runs alone with the lane's whole share, and so does one
    that started a process it did not follow; files with a measured peak run
    side by side, each with its peak and headroom, and one whose child took
    its share's fraction holds a share more beside them (`gpu_needs`). A red
    row stays red: nothing is run again. The rows join one verdict, each
    recorded as a task's are, and a later lane reuses the green ones."""
    url, stored, _ = cache_worker
    monkeypatch.setenv("DEW_XLA_CACHE_URL", url)
    monkeypatch.setenv("DEW_XLA_CACHE_TOKEN", "token")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
    # nvidia-smi as the driver answers it for an 80 GiB GPU.
    tools = tmp_path / "bin"
    tools.mkdir()
    (tools / "nvidia-smi").write_text("#!/bin/sh\necho 81920\n")
    (tools / "nvidia-smi").chmod(0o755)
    monkeypatch.setenv("PATH", f"{tools}:{os.environ['PATH']}")
    source = ("import os, subprocess, sys, time\n"
              "from pathlib import Path\n"
              "def test_runs():\n"
              "    began = time.monotonic()\n"
              "    time.sleep(1)\n"
              "    share = os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']\n"
              "    ran = Path(f'ran-{Path(__file__).stem}-{os.getpid()}')\n"
              "    ran.write_text(f'{share} {began} {time.monotonic()}')\n")
    names = [f"tests/test_file_{index}.py" for index in range(5)]
    for name in names:
        (tmp_path / name).write_text(source)
    # One that fails on any share, and one that starts a process.
    (tmp_path / "tests/test_red.py").write_text(source + "    assert share == 'all of it'\n")
    (tmp_path / "tests/test_spawner.py").write_text(
        source + "    subprocess.run([sys.executable, '-c', 'pass'], check=True)\n")
    every = [*names, "tests/test_red.py", "tests/test_spawner.py"]
    venv = tmp_path / ".venv-3.12/bin"
    venv.mkdir(parents=True)
    (venv / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (venv / "python").chmod(0o755)

    def ran(stem: str) -> list[list[str]]:
        return [path.read_text().split() for path in sorted(tmp_path.glob(f"ran-{stem}-*"),
                                                            key=lambda path: path.stat().st_mtime)]

    def overlapping(runs: list[list[str]]) -> int:
        spans = [(float(began), float(ended)) for _, began, ended in runs]
        return max(sum(began <= moment < ended for began, ended in spans) for moment, _ in spans)

    out = tmp_path / "verdict.json"
    assert ci.gpu_lane("3.12", every, out, cache=True, processes=4, deadline=600.0) == 0
    rows = json.loads(out.read_text())["rows"]
    assert [(row["name"], row["exitCode"]) for row in rows] == [
        (f"3.12:{name}", int(name == "tests/test_red.py")) for name in every]
    first = [run for name in every for run in ran(Path(name).stem)]
    assert [share for share, *_ in first] == ["0.5"] * len(every), "unmeasured: each alone, the whole share"
    assert overlapping(first) == 1
    assert len(ran("test_red")) == 1, "a red row stays red: its file is not run again"
    # Measured peaks of 4 GiB: each file then holds 6 GiB of the 40, and four run at once. Records without
    # a key, as a run on these inputs leaves to be run again, under the last run's and these inputs'.
    # The spawner's child took its fraction, so it holds two shares: 12 GiB beside the others' 24.
    common = ci.shared("3.12")
    for name, inputs in itertools.product([*names, "tests/test_spawner.py"], (None, common)):
        at = ci.cache_url("3.12", ci.row("3.12", name, ""), ci.device_class(), inputs)[len(url):]
        stored[at] = json.dumps({"key": None, "depends": [name], "row": {}, "reads": [], "peak": 4 * 2 ** 30,
                                 "shares": int(name == "tests/test_spawner.py"), "pools": 0}).encode()
    assert ci.gpu_lane("3.12", every, out, cache=True, processes=4, deadline=600.0) == 0
    rows = json.loads(out.read_text())["rows"]
    assert [row["exitCode"] for row in rows] == [int(name == "tests/test_red.py") for name in every]
    measured = [ran(Path(name).stem)[-1] for name in names]
    assert [share for share, *_ in measured] == [str(6 * 2 ** 30 / (81920 * 2 ** 20))] * len(names)
    assert 2 <= overlapping(measured) <= 4
    assert [share for share, *_ in ran("test_spawner")] == ["0.5", measured[0][0]]
    assert overlapping([*measured, ran("test_spawner")[-1]]) >= 2, "a followed spawner packs with the rest"
    gib, budget = 2 ** 30, 40 * 2 ** 30
    assert ci.gpu_needs({"depends": None, "peak": 4 * gib}, budget) == (budget, budget)
    held = {"depends": [], "peak": 4 * gib, "shares": 2, "pools": gib}
    assert ci.gpu_needs(held, budget) == (6 * gib, 19 * gib), "6 GiB, twice more for its children, and 1 GiB"
    assert ci.gpu_lane("3.12", [], out, cache=True, processes=4, deadline=600.0) == 0
    assert json.loads(out.read_text()) == {"rows": []}, "an empty lane runs nothing"
    assert len(ran("test_red")) == 2, "a red row is never reused"
    assert ci.gpu_lane("3.12", names, out, cache=True, processes=4, deadline=600.0) == 0
    assert all(len(ran(Path(name).stem)) == 2 for name in names), "the third lane reused the green rows"


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
    venv = tmp_path / ".venv-3.12/bin"
    venv.mkdir(parents=True)
    (venv / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (venv / "python").chmod(0o755)
    for name in ("negative", "positive"):
        (tmp_path / "runs" / name).mkdir(parents=True)
        ci.task("3.12", [f"tests/test_{name}.py"], "", tmp_path / "runs" / name / "rows.json", 600.0,
                cache=False, coverage=tmp_path / "runs" / name)
    assert ci.coverage_report(tmp_path / "runs") == 0
    files = {Path(name).name: measured for name, measured in
             json.loads((tmp_path / "runs/coverage.json").read_text())["files"].items()}
    assert files["sign.py"]["summary"]["percent_covered"] == 100.0
    least = (tmp_path / "runs/coverage.md").read_text().splitlines()[2].split(" | ")
    assert Path(least[0].removeprefix("| ")).name == "unused.py" and least[-1] == "0.0% |"


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
    venv = tmp_path / ".venv-3.12/bin"
    venv.mkdir(parents=True)
    (venv / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (venv / "python").chmod(0o755)
    worker: dict[str, bytes] = {}

    def cache_request(url, token, data=None):
        if data is not None:
            worker[url] = data
            return b""
        return worker.get(url)

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
    # Another commit's run between a commit's run and its retry: a fixture's change is every file's input.
    fixture = tmp_path / "tests/fixture.json"
    fixture.write_text("{}")
    assert run("GenuineIntel-6-143") == 9
    fixture.write_text('{"changed": true}')
    assert run("GenuineIntel-6-143") == 10
    fixture.write_text("{}")
    assert run("GenuineIntel-6-143") == 10
    monkeypatch.setenv("FAIL", "1")
    assert run("GenuineIntel-6-143", policy="off") == 11
    monkeypatch.delenv("FAIL")
    fixture.write_text('{"changed": true}')
    assert run("GenuineIntel-6-143") == 11
    fixture.write_text("{}")
    assert run("GenuineIntel-6-143") == 12



def test_a_reused_row_runs_again_when_a_page_its_tests_read_or_a_directory_they_list_changes(
        ci, tmp_path, monkeypatch):
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
    venv = tmp_path / ".venv-3.12/bin"
    venv.mkdir(parents=True)
    (venv / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (venv / "python").chmod(0o755)
    worker: dict[str, bytes] = {}

    def cache_request(url, token, data=None):
        if data is not None:
            worker[url] = data
            return b""
        return worker.get(url)

    monkeypatch.setattr(ci, "cache_request", cache_request)
    monkeypatch.setattr(ci, "cpu_class", lambda: "GenuineIntel-6-143")
    monkeypatch.setenv("DEW_XLA_CACHE_URL", "https://cache.invalid")
    monkeypatch.setenv("DEW_XLA_CACHE_TOKEN", "token")
    # Beside the checkout's entries, not among them: the checkout's own directory is an input.
    (tmp_path / ".verdict").mkdir()
    out = tmp_path / ".verdict/verdict.json"
    command = json.loads((ROOT / ".armada.json").read_text())["task"]["command"]

    def run():
        monkeypatch.setenv("DEW_TEST_CACHE", "on")
        [entry] = [entry for entry in ci.plan(300.0, {}) if entry["python"] == "3.12"]
        values = {**entry, "out": out}
        filled = [re.sub(r"\{(\w+)\}", lambda field: str(values[field[1]]), word) for word in command]
        monkeypatch.setattr(sys, "argv", ["ci.py", *filled[2:]])
        monkeypatch.delenv("DEW_TEST_CACHE")
        assert ci.main() == 0
        [only] = json.loads(out.read_text())["rows"]
        assert only["exitCode"] == 0
        return len((tmp_path / "runs").read_text())

    assert run() == 1
    assert run() == 1
    (tmp_path / "docs/model.md").write_text("width 8\n")
    assert run() == 2
    assert run() == 2
    (tmp_path / "docs/serving.md").write_text("serve\n")
    assert run() == 3
    assert run() == 3
    # Python bytecode a run leaves beside a page is no change to the directory's pages.
    (tmp_path / "docs/__pycache__").mkdir()
    assert run() == 3
    (tmp_path / "docs/guides/serving.md").write_text("serve\n")
    assert run() == 4
    (tmp_path / "LICENSE").write_text("license\n")
    assert run() == 5
    assert run() == 5
    (tmp_path / "notes/plan.md").write_text("plan again\n")
    assert run() == 6
    assert run() == 6

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


def test_ci_gangs_run_every_multihost_case_in_exactly_one_gang(ci, tmp_path):
    """`ci.py gangs` reads the models from test_multihost.py: each gang's -k selects a part of the
    file's collected cases, the parts are disjoint and cover every case, and a model added to
    MODELS makes a gang of its own."""
    source = ROOT / "tests/test_multihost.py"
    items = ci.gangs(source)

    def collected(*select: str) -> set[str]:
        # As a gang's rank collects it: the file skips itself outside a gang.
        done = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p",
                               "no:cacheprovider", str(source), *select],
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
