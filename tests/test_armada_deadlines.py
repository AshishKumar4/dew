"""A CI task's deadline (tools/armada/ci.py's `tested`): a task past it interrupts its pytest, names every row
it holds red, and says where every thread of its pytest stood."""

import json
import time

import pytest
from armada_support import interpreter, loaded, written


@pytest.fixture
def ci(tmp_path, monkeypatch):
    return loaded(tmp_path, monkeypatch)


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


def test_a_task_past_its_deadline_says_where_every_thread_of_its_pytest_was(ci, tmp_path):
    """A task interrupted at its deadline reports its pytest's every thread as it stood then, a stall before
    any test ran included, which pytest's own report of an interrupt cannot give (2026-11-03: a group stalled
    28 minutes in a read with no deadline during collection, its row saying only that no tests ran)."""
    interpreter(tmp_path, "3.12")
    written(tmp_path, {"test_stalls.py": "import time\n\n\ndef stalled_reading():\n    time.sleep(60)\n\n\n"
                                         "stalled_reading()\n\n\ndef test_never():\n    pass\n"})
    ci.task("3.12", ["tests/test_stalls.py"], "", tmp_path / "rows.json", 3.0, cache="off", grace=5.0)
    [interrupted] = json.loads((tmp_path / "rows.json").read_text())["rows"]
    assert interrupted["kind"] == "deadline"
    # faulthandler's dump: each thread's frames, most recent call first, the stalled one's among them.
    assert "(most recent call first)" in interrupted["output"] and "stalled_reading" in interrupted["output"]
