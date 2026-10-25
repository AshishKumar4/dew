"""Running every step of a teardown, whichever fail, and raising one error that names them all, and
the lifetime of a process group: a command started as the leader of a session of its own, whose
group ends with it on every exit."""

import contextlib
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

POLL_SECONDS = 0.05
"""How often a group's end asks whether a process is left in it."""
SETTLE_SECONDS = 10.0
"""How long a group's end waits for its SIGKILL to land on every process left in it."""


def attempted(steps: Sequence[tuple[str, Callable[[], None]]], *,
              primary: BaseException | None = None) -> None:
    """Run every step, even after one fails. Each failure is noted on `primary`, the error its
    caller is already raising, which stays the one raised; without one the first failure is raised
    with the later ones noted on it."""
    error = primary
    for label, step in steps:
        try:
            step()
        except BaseException as failure:
            if error is None:
                error = failure
            else:
                error.add_note(f"{label} failed: {failure!r}")
    if primary is None and error is not None:
        raise error


UNREAPED = sys.platform != "darwin" or sys.version_info >= (3, 13)
"""Whether Python reads a child's exit without reaping it (`os.waitid`), which it does not on macOS
before Python 3.13."""


def exit_code(process: subprocess.Popen) -> int | None:
    """The code `process` exited with, as Popen reports it (-N for death by signal N), or None while
    it runs, read without reaping it: a leader not yet reaped keeps its pid, and so its group's id,
    from every other process until `end_group` reaps it. Where Python cannot (`UNREAPED`), the read
    reaps it, after which its group is no longer signalled."""
    if process.returncode is not None:
        return process.returncode
    if not UNREAPED:
        return process.poll()
    exited = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    if exited is None:
        return None
    return -exited.si_status if exited.si_code in (os.CLD_KILLED, os.CLD_DUMPED) else exited.si_status


def signal_group(process: subprocess.Popen, signum: int) -> bool:
    """Send `signum` to the process group `process` leads, whether a process of it was there to take
    it. Its id is the leader's pid, which no other process takes while the leader is unreaped
    (`exit_code`), so a leader already reaped (its `returncode` set) is not signalled."""
    if process.returncode is not None:
        return False
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        return False
    return True


def groups_ended(processes: Sequence[subprocess.Popen], grace: float, forced: threading.Event | None = None
                 ) -> None:
    """Wait up to `grace` seconds, or until `forced` is set, for every leader of `processes` to exit,
    left unreaped (`exit_code`)."""
    deadline = time.monotonic() + grace
    while (forced is None or not forced.is_set()) and time.monotonic() < deadline:
        if not [process for process in processes if exit_code(process) is None]:
            return
        time.sleep(POLL_SECONDS)


def members(process: subprocess.Popen) -> list[int]:
    """The live (not zombie) processes left in the group `process` leads, as Linux's /proc lists
    them; none where there is no /proc to read."""
    found = []
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            # Bytes: a process's name is any bytes it set, which the fields after it never hold.
            fields = stat.read_bytes().rsplit(b")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if fields[0] != b"Z" and int(fields[2]) == process.pid:
            found.append(int(stat.parent.name))
    return found


def end_group(process: subprocess.Popen, grace: float = 0.0, interrupt: int = signal.SIGTERM) -> int:
    """End the group `process` leads: with a `grace`, `interrupt` to every process in it and up to
    `grace` seconds for the leader to exit; then SIGKILL to every process left in the group, a child
    the leader left running among them, sent before the leader is reaped, while the group's id is
    still its own; then the leader reaped. Returns its exit code. A process that left the group on
    purpose (a session of its own) is out of reach."""
    if grace > 0 and exit_code(process) is None and signal_group(process, interrupt):
        groups_ended((process,), grace)
    if not signal_group(process, signal.SIGKILL):
        return process.wait()
    # The kill lands asynchronously; the group is held, its leader unreaped, until it has.
    deadline = time.monotonic() + SETTLE_SECONDS
    while (left := members(process)) and time.monotonic() < deadline:
        time.sleep(POLL_SECONDS)
    code = process.wait()
    if left:
        raise TimeoutError(f"processes {left} of the group {process.pid} led outlived its SIGKILL by "
                           f"{SETTLE_SECONDS:.0f} s")
    return code


@contextlib.contextmanager
def process_group(command: Sequence[str], *, grace: float = 0.0, interrupt: int = signal.SIGTERM,
                  **options) -> Iterator[subprocess.Popen]:
    """`command` started as the leader of a session and process group of its own (`subprocess.Popen`
    with `options`), whose group ends with the block on every exit, a success's too (`end_group`),
    and whose pipes close after it."""
    process = subprocess.Popen(command, start_new_session=True, **options)

    def end() -> None:
        end_group(process, grace, interrupt)

    try:
        yield process
    finally:
        streams = [stream for stream in (process.stdin, process.stdout, process.stderr) if stream is not None]
        attempted([("end the process group", end),
                   *((f"close {stream!r}", stream.close) for stream in streams)], primary=sys.exception())
