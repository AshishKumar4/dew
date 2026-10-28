"""A command's processes owned as one group: started as the leader of a session and process group of their
own (`start`), signalled as a group (`signal_group`) and stopped (`stop`): an interrupt first where asked,
so a test runner reports what it ran, then SIGKILL to the whole group, then a wait until none of the
group's processes still runs, as SIGKILL lands asynchronously and a descendant can outlive its leader's
reaping. Dew's tests (tests/process_support.py) and CI (tools/armada/ci.py) own their children through it.
dew-ops keeps a reaper of its own beside it (dewops/process.py), which also follows descendants that leave
the group, as a subreaper, on hosts without a Dew checkout. The standard library alone.

A group outlives no owner: each has a lifeline (`LIFELINE`), a process in a session of its own that holds
the read end of a pipe whose write end only the owner holds, and kills the group once that pipe closes, as
it does when the owner dies however it dies (a SIGKILL, a SIGTERM that unwinds no `with`), or stops the
group (`stop`). The group stays in a session of its own, so a rank's cleanup signals its own (Main,
2026-10-28: a worker outlived its killed owner 43 hours at 195% CPU on the VM).
"""

import contextlib
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

LIFELINE = """import os, signal, sys
owner = int(sys.argv[1])
named = os.read(owner, 32).strip()
if named:
    os.read(owner, 1)
    try:
        os.killpg(int(named), signal.SIGKILL)
    except ProcessLookupError:
        pass
"""
"""A group's lifeline: it reads the group's leader from its owner pipe, then waits for the pipe to close, the
owner gone or the group stopped, and kills the group; an owner gone before it named one leaves none."""
LIFELINES: dict[int, tuple[int, subprocess.Popen]] = {}
"""Each group's lifeline, by its leader: the owner pipe's write end, and the lifeline's process."""


def start(command: Sequence[str], **options) -> subprocess.Popen:
    """`command` as the leader of a session and process group of its own, `options` as Popen's, with its
    lifeline (`LIFELINE`) started first, so no instant leaves the group without one."""
    read, write = os.pipe2(os.O_CLOEXEC)
    lifeline = subprocess.Popen([sys.executable, "-c", LIFELINE, str(read)], pass_fds=(read,),
                                start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
    os.close(read)
    try:
        process = subprocess.Popen(command, start_new_session=True, **options)
    except BaseException:
        os.close(write)
        lifeline.wait()
        raise
    os.write(write, f"{process.pid}\n".encode())
    LIFELINES[process.pid] = (write, lifeline)
    return process


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
    # Its lifeline, told the group is stopped: it kills what is left and ends.
    if held := LIFELINES.pop(process.pid, None):
        os.close(held[0])
        held[1].wait()
    return running
