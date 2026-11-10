"""A CI task's deadline (tools/armada/ci.py's `tested`): a task past it interrupts its pytest, names every row
it holds red, and says where every thread of its pytest stood."""

import base64
import hashlib
import json
import time
import zipfile
from pathlib import Path

import pytest
from armada_support import interpreter, loaded, written


@pytest.fixture
def ci(tmp_path, monkeypatch):
    return loaded(tmp_path, monkeypatch)


def test_a_task_past_its_deadline_is_interrupted_its_hung_file_red_and_the_rest_run_again(ci, tmp_path):
    """A hang is graded: the task interrupts pytest at its deadline, short of
    armada's own timeout, kills its process group where the interrupt is not
    heard, and writes the file it was spent in red, the deadline's; a file
    pytest left no report of, before the hang or after it, is not graded but
    said stale, which the queue runs again, never a red of its tests (Main,
    2026-11-09: train 745 filed six files packed beside a hung one as reds)."""
    (tmp_path / "tests/test_quick.py").write_text("def test_ok(): pass\n")
    (tmp_path / "tests/test_zlast.py").write_text("def test_after(): pass\n")
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
    ci.task("3.12", ["tests/test_quick.py", "tests/test_hangs.py", "tests/test_zlast.py"], "", out, 3.0,
            grace=2.0)
    # The deadline, the interrupt's grace, one settle budget and the reader's join, with room to spare.
    assert time.monotonic() - began < 3.0 + 2.0 + ci.SETTLE + 10 + 15
    rows = {row["name"]: row for row in json.loads(out.read_text())["rows"]}
    assert {name: (row["exitCode"], row["kind"]) for name, row in rows.items()} == {
        "3.12:tests/test_quick.py": (1, ci.STALE), "3.12:tests/test_hangs.py": (1, "deadline"),
        "3.12:tests/test_zlast.py": (1, ci.STALE)}
    assert "spent in tests/test_hangs.py" in rows["3.12:tests/test_zlast.py"]["output"]


STALLS = {
    "in the interpreter": ("import time\n\n\ndef stalled_reading():\n    time.sleep(60)\n\n\n"
                           "stalled_reading()\n", "stalled_reading"),
    # A mutex its own thread locks again waits in the kernel for good, and no signal wakes it: through
    # ctypes.CDLL without the GIL, as an XLA compile or a wait on a device holds a thread; through PyDLL
    # holding it, which no thread of the process can then take (row 4828).
    "in native code": ("import ctypes\n\nlibc = ctypes.CDLL(None)\nheld = ctypes.create_string_buffer(64)\n"
                       "libc.pthread_mutex_lock(held)\n\n\n"
                       "def stalled_natively():\n    libc.pthread_mutex_lock(held)\n\n\n"
                       "stalled_natively()\n", "stalled_natively"),
    "in native code holding the GIL": (
        "import ctypes\n\nlibc = ctypes.PyDLL(None)\nheld = ctypes.create_string_buffer(64)\n"
        "libc.pthread_mutex_lock(held)\n\n\n"
        "def stalled_holding_the_gil():\n    libc.pthread_mutex_lock(held)\n\n\n"
        "stalled_holding_the_gil()\n", "stalled_holding_the_gil")}
"""A stall at collection, by where its thread waits, and the function it waits in."""


def wheel(root: Path) -> Path:
    """A wheel of one empty package, a release check's requirement that uv installs from its file alone."""
    path = root / "unused-1.0-py3-none-any.whl"
    files = {"unused/__init__.py": b"",
             "unused-1.0.dist-info/METADATA": b"Metadata-Version: 2.1\nName: unused\nVersion: 1.0\n",
             "unused-1.0.dist-info/WHEEL": b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\n"
                                           b"Tag: py3-none-any\n"}
    hashed = {name: base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
              for name, data in files.items()}
    digests = [f"{name},sha256={hashed[name]},{len(data)}" for name, data in files.items()]
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        archive.writestr("unused-1.0.dist-info/RECORD",
                         "\n".join([*digests, "unused-1.0.dist-info/RECORD,,"]) + "\n")
    return path


@pytest.mark.parametrize("launched", [False, True], ids=["pytest", "a release check's"])
@pytest.mark.parametrize("stall", STALLS)
def test_a_task_past_its_deadline_says_where_every_thread_of_its_pytest_was(ci, tmp_path, monkeypatch, stall,
                                                                           launched):
    """A task interrupted at its deadline reports its pytest's every thread as it stood then, its Python
    frames and the native ones under them, a stall before any test ran included, which pytest's own report
    of an interrupt cannot give (2026-11-03: a group stalled 28 minutes in a read with no deadline during
    collection, its row saying only that no tests ran): a thread stalled in the interpreter, or in native
    code holding the GIL or not (an XLA compile, a wait on a device, a collective), read from outside the
    process. A release check's pytest runs under `uv run`, which stands between the task and pytest: the
    stacks are still pytest's own. The row names the stall's whole record among its evidence in the task's
    artifacts (`stalls.kept`), its stacks and the host's cgroup quota beside them (rows 5452, 5454)."""
    source, stalled = STALLS[stall]
    artifacts = tmp_path / "artifacts"
    monkeypatch.setenv("ARMADA_ARTIFACTS", str(artifacts))
    interpreter(tmp_path, "3.12")
    written(tmp_path, {"test_stalls.py": f"{source}\n\ndef test_never():\n    pass\n"})
    ci.task("3.12", ["tests/test_stalls.py"], "", tmp_path / "rows.json", 10.0 if launched else 3.0,
            cache="off", grace=5.0, release=str(wheel(tmp_path)) if launched else "")
    [interrupted] = json.loads((tmp_path / "rows.json").read_text())["rows"]
    assert interrupted["kind"] == "deadline"
    # py-spy's frame of the stalled function, by its file and line.
    assert f"{stalled} (test_stalls.py:" in interrupted["output"]
    [kept] = interrupted["artifacts"]
    record = artifacts / kept
    assert f"{stalled} (test_stalls.py:" in record.read_text() and "cpu.max" in record.read_text()


def test_each_stall_of_a_tasks_runs_of_the_same_rows_keeps_its_own_record(ci, tmp_path):
    """A task's two runs of the same rows past their deadlines (a rerun after a deadline) each keep their own
    stall record (`stalls.kept`), neither the other's."""
    first, second = (ci.stalls.kept(tmp_path, "3.12-same", f"stacks of run {run}", time.time(), "host")
                     for run in (1, 2))
    assert first != second
    assert first.read_text().startswith("stacks of run 1")
    assert second.read_text().startswith("stacks of run 2")


def test_a_task_within_its_deadline_writes_no_stacks(ci, tmp_path):
    """A run that ends within its deadline is never dumped, however close to it it ends: the stacks are the
    deadline's alone, never a timer's before it, which could kill a healthy multithreaded run mid-dump and
    read it red as a whole-file failure (row 3738). A test failing at 7 s of a 10 s deadline keeps its own
    failure in its row, and no thread's stack."""
    interpreter(tmp_path, "3.12")
    written(tmp_path, {"test_close.py": "import time\n\n\ndef test_close():\n    time.sleep(7)\n"
                                        "    assert False, 'its own failure'\n"})
    ci.task("3.12", ["tests/test_close.py"], "", tmp_path / "rows.json", 10.0, cache="off")
    [row] = json.loads((tmp_path / "rows.json").read_text())["rows"]
    assert row["kind"] == "nodes" and "its own failure" in row["output"]
    assert "(test_close.py:" not in row["output"]


def test_a_task_past_its_deadline_times_the_file_it_stopped_in_at_the_least_it_needs(ci, tmp_path):
    """Dew's base CI run 42382371599 packed tests/test_stacked_blocks.py, which takes over 1,744 s, with four
    short files into a task of a 1,680 s deadline, and the task was red on it run after run: the plan weighed
    the file at its old green time, as its row held what its task's wall time gave a file in proportion to its
    tests' times, none of which the interrupted test had. Past its deadline a task times the file its tests
    were running in at its own tests' seconds and the wall time no file's tests account for, at least what the
    file needs; a file that finished, at its tests' own seconds, graded by them; and a file that never
    started, at none: its row says `unobserved`, for armada to weigh no plan by, names its file at 0 in its
    timings, so armada leaves out of its file timings a file it shares with other rows, and is stale, run
    again, never a red of its tests (Main, 2026-11-09)."""
    deadline = 8.0
    written(tmp_path, {"test_first.py": "import time\n\n\ndef test_first():\n    time.sleep(0.5)\n",
                       "test_stuck.py": "import time\n\n\ndef test_stuck():\n    time.sleep(600)\n",
                       "test_never.py": "def test_never():\n    pass\n"})
    interpreter(tmp_path, "3.12")
    out = tmp_path / "verdict.json"
    began = time.monotonic()
    ci.task("3.12", ["tests/test_first.py", "tests/test_stuck.py", "tests/test_never.py"], "", out, deadline,
            cache="off", grace=5.0)
    wall = time.monotonic() - began
    rows = {row["name"]: row for row in json.loads(out.read_text())["rows"]}
    first, stuck, never = (rows[f"3.12:tests/test_{name}.py"] for name in ("first", "stuck", "never"))
    slept = first["tests"]["tests/test_first.py::test_first"]
    assert (first["exitCode"], stuck["kind"], never["kind"]) == (0, "deadline", ci.STALE)
    assert slept >= 0.5 and first["seconds"] == slept
    # The file the deadline stopped holds the wall time the finished file's own did not.
    assert stuck["seconds"] + slept >= deadline
    assert stuck["timings"] == {"3.12:tests/test_stuck.py": stuck["seconds"]}
    # The rows claim no more than the task took, and a file that never started none of it.
    assert sum(row["seconds"] for row in rows.values()) <= wall
    assert (never["unobserved"], never["seconds"], never["timings"]) == (
        True, 0.0, {"3.12:tests/test_never.py": 0.0})
    assert "unobserved" not in first and "unobserved" not in stuck


def test_a_pytest_that_never_began_runs_again_and_one_a_signal_ended_is_red_with_its_signal(ci, tmp_path):
    """A task whose pytest never began (its interpreter would not start: the environment, the launch, an
    install before pytest) has no verdict of any test: its row is said stale and run again, never a red of
    its tests. One whose pytest began and a signal ended, its report never written, is its file's red whole,
    the signal kept on its row for whoever it routes to (Main, 2026-11-09: a write the runner could not make
    before pytest began filed 43 GPU tier parts' files as reds)."""
    interpreter(tmp_path, "3.12")
    python = tmp_path / ".venv-3.12/bin/python"
    kept = python.read_text()
    python.write_text("#!/bin/sh\nexit 3\n")
    written(tmp_path, {"test_any.py": "def test_ok():\n    pass\n"})
    ci.task("3.12", ["tests/test_any.py"], "", tmp_path / "rows.json", 30.0, cache="off")
    [unstarted] = json.loads((tmp_path / "rows.json").read_text())["rows"]
    assert (unstarted["exitCode"], unstarted["kind"]) == (1, ci.STALE)
    python.write_text(kept)
    written(tmp_path, {"test_any.py": "import os\nimport signal\n\n\ndef test_dies():\n"
                                      "    os.kill(os.getpid(), signal.SIGKILL)\n"})
    ci.task("3.12", ["tests/test_any.py"], "", tmp_path / "killed.json", 30.0, cache="off")
    [killed] = json.loads((tmp_path / "killed.json").read_text())["rows"]
    assert (killed["exitCode"], killed["kind"], killed.get("signal")) == (1, "whole", 9)


def test_a_task_says_each_red_row_last_where_armada_keeps_its_tail(ci, tmp_path, capsys):
    """A task's red rows end what it says, each its name, why it is red and what pytest said last: armada
    keeps a task's log tail beside its exit, and a map task's verdict reaches no report but its judge's red
    row, which names that tail (Main, 2026-11-09: a map task "exited 1 after 437 s", its row's "pytest
    exited -9" kept in its output alone)."""
    interpreter(tmp_path, "3.12")
    written(tmp_path, {"test_any.py": "import os\nimport signal\n\n\ndef test_dies():\n"
                                      "    os.kill(os.getpid(), signal.SIGKILL)\n"})
    capsys.readouterr()
    ci.task("3.12", ["tests/test_any.py"], "", tmp_path / "killed.json", 30.0, cache="off")
    said = capsys.readouterr().out.rstrip().splitlines()
    red = said.index("RED 3.12:tests/test_any.py")
    assert said[red + 1].startswith("  | pytest was killed by signal 9 with no report of tests/test_any.py")
    # A failure whose report runs long: why its row is red is said, before what pytest said last.
    written(tmp_path, {"test_any.py": "def test_fails():\n"
                                      "    print('\\n'.join(f'line {at}' for at in range(40)))\n"
                                      "    assert False, 'the sum is wrong'\n"})
    ci.task("3.12", ["tests/test_any.py"], "", tmp_path / "failed.json", 30.0, cache="off")
    said = capsys.readouterr().out.rstrip().splitlines()
    red = said.index("RED 3.12:tests/test_any.py")
    assert "the sum is wrong" in said[red + 1] and "  | ..." in said[red:]
