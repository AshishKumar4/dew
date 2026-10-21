"""A command's processes owned as one group: started as the leader of a session and process group of their
own (`start`), signalled as a group (`signal_group`) and stopped (`stop`): an interrupt first where asked,
so a test runner reports what it ran, then SIGKILL to the whole group, then a wait until none of the
group's processes still runs, as SIGKILL lands asynchronously and a descendant can outlive its leader's
reaping. Dew's tests (tests/process_support.py) and CI (tools/armada/ci.py) own their children through it.
dew-ops keeps a reaper of its own beside it (dewops/process.py), which also follows descendants that leave
the group, as a subreaper, on hosts without a Dew checkout. The standard library alone.
"""

import contextlib
import os
import signal
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path


def start(command: Sequence[str], **options) -> subprocess.Popen:
    """`command` as the leader of a session and process group of its own, `options` as Popen's."""
    return subprocess.Popen(command, start_new_session=True, **options)


def signal_group(leader: int, number: signal.Signals) -> None:
    """`number` to every process left in the group `leader` led."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(leader, number)


def members(group: int) -> list[int]:
    """The live (not zombie) processes of process group `group`."""
    found = []
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if fields[0] != "Z" and int(fields[2]) == group:
            found.append(int(stat.parent.name))
    return found


def stop(process: subprocess.Popen, *, interrupt: float | None = None, settle: float = 60.0) -> list[int]:
    """`process`'s group stopped: where `interrupt` is given and the leader still runs, SIGINT and up to
    `interrupt` seconds for it to end; then SIGKILL to the group and up to `settle` seconds, one budget,
    for the leader to be reaped and every other member to end. The members still running then (none,
    normally; a leader the kill has not reached, in uninterruptible sleep, among them), never raised, so
    a caller can still report what ran."""
    if interrupt is not None and process.poll() is None:
        signal_group(process.pid, signal.SIGINT)
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(interrupt)
    signal_group(process.pid, signal.SIGKILL)
    deadline = time.monotonic() + settle
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(settle)
    while (running := members(process.pid)) and time.monotonic() < deadline:
        time.sleep(0.01)
    return running
