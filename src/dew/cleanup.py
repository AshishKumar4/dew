"""Running every step of a teardown, whichever fail, and raising one error that names them all, and
the lifetime of a process group: a command started as the leader of a session of its own, whose
group ends with it on every exit."""

import contextlib
import os
import signal
import subprocess
import time
from collections.abc import Callable, Iterator, Sequence

POLL_SECONDS = 0.05
"""How often a group's end asks whether a process is left in it."""


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


def signal_group(process: subprocess.Popen, signum: int) -> bool:
    """Send `signum` to the process group `process` leads, whether a process of it was there to take
    it. Its id is the leader's pid, which Linux gives no other process while the group holds one."""
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        return False
    return True


def group_held(process: subprocess.Popen) -> bool:
    """Whether the group `process` leads still holds a process: the leader, or one it started, which
    its exit leaves running on in the group (a data worker, a server). The leader's own exit is
    reaped first, which a group of exited processes alone no longer counts."""
    process.poll()
    return signal_group(process, 0)


def end_group(process: subprocess.Popen, grace: float = 0.0, interrupt: int = signal.SIGTERM) -> int:
    """End the group `process` leads: with a `grace`, `interrupt` to every process in it and up to
    `grace` seconds for them to exit; then SIGKILL to what remains, and the leader reaped. Returns its
    exit code. A process that left the group on purpose (a session of its own) is out of reach."""
    if grace > 0 and group_held(process) and signal_group(process, interrupt):
        deadline = time.monotonic() + grace
        while group_held(process) and time.monotonic() < deadline:
            time.sleep(POLL_SECONDS)
    signal_group(process, signal.SIGKILL)
    return process.wait()


@contextlib.contextmanager
def process_group(command: Sequence[str], *, grace: float = 0.0, interrupt: int = signal.SIGTERM,
                  **options) -> Iterator[subprocess.Popen]:
    """`command` started as the leader of a session and process group of its own (`subprocess.Popen`
    with `options`), whose group ends with the block on every exit, a success's too (`end_group`),
    and whose pipes close after it."""
    process = subprocess.Popen(command, start_new_session=True, **options)
    try:
        yield process
    finally:
        streams = [stream for stream in (process.stdin, process.stdout, process.stderr) if stream is not None]
        attempted([("end the process group", lambda: end_group(process, grace, interrupt)),
                   *((f"close {stream!r}", stream.close) for stream in streams)])
