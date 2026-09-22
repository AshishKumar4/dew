"""`dew launch`: the pool it starts, the commands it writes, and how it stops.

Each test runs the launcher as its own process against small Python
programs on this machine, so what is checked is what a shell sees: the exit
code, the output, and which processes are left alive.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}


def test_training_does_not_import_the_command_line():
    """The pool's variable names are shared with `prepare_process` through a
    module with no imports, so the training layer never loads the CLI."""
    probe = ("import sys, dew.training.runtime\n"
             "print(sorted(name for name in sys.modules if name.startswith('dew.cli')))\n")
    done = subprocess.run([sys.executable, "-c", probe], env=ENV, capture_output=True,
                          text=True, check=True)
    assert done.stdout.strip() == "[]"


def launcher(*arguments: str) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-m", "dew.cli.main", "launch", *arguments],
                            cwd=REPO_ROOT, env=ENV, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, start_new_session=True)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def started_ranks(tmp_path: Path, count: int, seconds: float = 60) -> list[int]:
    """The pids the ranks wrote, once all `count` have written one."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        pids = [path.read_text() for path in tmp_path.glob("rank*")]
        if len(pids) == count and all(pids):
            return [int(pid) for pid in pids]
        time.sleep(0.1)
    raise AssertionError(f"{count} ranks did not start")


# Each rank records its pid and then waits the way one blocked in a
# collective does.
WAITING_RANK = ("import os, pathlib, sys, time\n"
                "pathlib.Path(sys.argv[1], 'rank' + os.environ['DEW_PROCESS_ID'])"
                ".write_text(str(os.getpid()))\n"
                "time.sleep(600)\n")


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP, signal.SIGINT])
def test_a_signalled_launch_stops_its_pool_and_reports_the_signal(tmp_path, signum):
    """A scheduler's SIGTERM, a closed terminal's SIGHUP and a Ctrl-C all
    stop every rank, which sit in sessions of their own that no terminal
    signal reaches, and the launch exits 128 plus the signal."""
    launch = launcher("--processes-per-host", "2", "--port", "1", "--",
                      sys.executable, "-c", WAITING_RANK, str(tmp_path))
    ranks = started_ranks(tmp_path, 2)
    launch.send_signal(signum)
    assert launch.wait(timeout=60) == 128 + signum
    deadline = time.monotonic() + 10
    while any(alive(pid) for pid in ranks) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not any(alive(pid) for pid in ranks)


def test_a_rank_printing_bytes_that_are_not_utf8_still_finishes():
    """One undecodable byte, then more output than a pipe holds: the relay
    keeps reading, so the rank never blocks on a full pipe, and the byte
    arrives as a replacement character."""
    program = ("import sys\n"
               "sys.stdout.buffer.write(b'\\xff\\xfe\\n')\n"
               "sys.stdout.buffer.write(b'x' * 300_000 + b'\\n')\n")
    launch = launcher("--port", "1", "--", sys.executable, "-c", program)
    try:
        output, _ = launch.communicate(timeout=60)
    finally:
        launch.kill()
    assert launch.returncode == 0
    assert "[0] \ufffd\ufffd" in output.decode()
