"""Running every step of a teardown, whichever fail, and raising one error that names them all, and
the lifetime of a process group: a command started as the leader of a session of its own, whose
group ends with it on every exit.

A group `start`ed has a lifeline (`lifeline`) as well: this module run in a session of its own, which
holds the read end of a pipe whose write end only the owner holds and kills the group once that pipe
closes, as it does when the owner dies however it dies (a SIGKILL, a SIGTERM that unwinds no `with`)
or stops the group (`stop`). A group that ends on its own ends its lifeline, which then kills nothing:
its id is free for another process. The group stays in a session of its own, so a rank's cleanup
signals its own (the maintainer, 2026-10-28: a worker outlived its killed owner 43 hours at 195% CPU on
the VM). The standard library alone: the CI harness loads this file by path (tools/armada/child_groups.py),
as importing `dew` sets the process's XLA flags."""

import contextlib
import os
import select
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager
from pathlib import Path

POLL_SECONDS = 0.05
"""How often a group's end asks whether a process is left in it."""
SETTLE_SECONDS = 10.0
"""How long a group's end waits for its SIGKILL to land on every process left in it."""

_LIFELINES: dict[int, tuple[int, subprocess.Popen]] = {}
"""Each `start`ed group's lifeline, by its leader: the owner pipe's write end, and its process."""
_HELD = threading.Lock()
"""`_LIFELINES`'s, as CI starts and stops groups from a pool of threads (ci.py's gpu-lane): each entry is
taken out under it by the one caller that closes its pipe, so no pipe is closed twice, nor another's that
took its number."""


def attempted(steps: Sequence[tuple[str, Callable[[], None]]], *,
              primary: BaseException | None = None) -> None:
    """Run every step, even after one fails. Each failure is noted on `primary`, the error its
    caller is already raising, which stays the one raised; without one the first failure is raised,
    noted with its own step, with the later ones noted on it."""
    error = primary
    for label, step in steps:
        try:
            step()
        except BaseException as failure:
            if error is None:
                error = failure
                error.add_note(f"{label} failed")
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


def _killpg(group: int, signum: int) -> bool:
    """Send `signum` to the process group `group`, whether a process of it was there to take it."""
    try:
        os.killpg(group, signum)
    except ProcessLookupError:
        return False
    return True


def signal_group(process: subprocess.Popen, signum: int) -> bool:
    """Send `signum` to the process group `process` leads, whether a process of it was there to take
    it. Its id is the leader's pid, which no other process takes while the leader is unreaped
    (`exit_code`), so a leader already reaped (its `returncode` set) is not signalled."""
    return process.returncode is None and _killpg(process.pid, signum)


def groups_ended(processes: Sequence[subprocess.Popen], grace: float, forced: threading.Event | None = None
                 ) -> None:
    """Wait up to `grace` seconds, or until `forced` is set, for every leader of `processes` to exit,
    left unreaped (`exit_code`)."""
    deadline = time.monotonic() + grace
    while (forced is None or not forced.is_set()) and time.monotonic() < deadline:
        if not [process for process in processes if exit_code(process) is None]:
            return
        time.sleep(POLL_SECONDS)


def members(group: int) -> list[int]:
    """The live (not zombie) processes left in the process group `group`, as Linux's /proc lists
    them; none where there is no /proc to read."""
    found = []
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            # Bytes: a process's name is any bytes it set, which the fields after it never hold.
            fields = stat.read_bytes().rsplit(b")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if fields[0] != b"Z" and int(fields[2]) == group:
            found.append(int(stat.parent.name))
    return found


GATE = ("import os, signal, sys\n"
        "held, failed = int(sys.argv[1]), int(sys.argv[2])\n"
        "os.set_inheritable(failed, False)\n"
        "if sys.argv[3]:\n"
        "    for name in ('SIGPIPE', 'SIGXFZ', 'SIGXFSZ'):\n"
        "        if hasattr(signal, name):\n"
        "            signal.signal(getattr(signal, name), signal.SIG_DFL)\n"
        "if os.read(held, 1):\n"
        "    os.close(held)\n"
        "    try:\n"
        "        os.execvp(sys.argv[4], sys.argv[5:])\n"
        "    except OSError as error:\n"
        "        os.write(failed, str(error.errno).encode())\n"
        "os._exit(127)\n")
"""What a `start`ed group's leader runs before its command, as the same process: it waits for its owner's
release, given once the lifeline holds its id, then executes the command, or writes the error its
execution raised; an owner gone before it released the command (its pipe closed with nothing written)
leaves it to exit having run nothing. It restores the signals Python ignores (SIGPIPE, SIGXFSZ) to their
defaults first, as Popen's `restore_signals` does for the process it executes, so the command meets them
as Popen leaves them, a pipe's writer ending by its SIGPIPE."""


def start(command: Sequence[str], **options) -> subprocess.Popen:
    """`command` as the leader of a session and process group of its own, `options` as Popen's, with its
    lifeline (`lifeline`) started first and holding the group's id before the command runs (`GATE`), so no
    instant leaves the group's processes without one: an owner killed between the group's start and its
    lifeline's knowing it took the group with it, where the command had run beside a lifeline that never
    learned it (row 4395). Returns once the command is executing, raising the error its execution raised
    (a missing program's `FileNotFoundError`) as Popen does. The lifelines of groups that ended on their own
    are let go first, their pipes closed. A start interrupted before its command runs, or whose command
    cannot execute, stops the group it started and closes its pipes: no caller holds it."""
    with _HELD:
        ended = [_LIFELINES.pop(leader) for leader, (_, held) in list(_LIFELINES.items())
                 if held.poll() is not None]
    for written, _ in ended:
        os.close(written)
    read, write = os.pipe2(os.O_CLOEXEC)
    # Isolated (-I): run as a script, this file's own directory would otherwise lead sys.path, where dew's
    # io.py and logging.py stand in for the standard library's.
    try:
        line = subprocess.Popen([sys.executable, "-I", __file__, str(read)], pass_fds=(read,),
                                start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
    except BaseException:
        # A lifeline started, or not, before the interruption ends with its owner pipe's close.
        os.close(write)
        raise
    finally:
        os.close(read)
    held, release = os.pipe2(os.O_CLOEXEC)
    told, failed = os.pipe2(os.O_CLOEXEC)
    program = options.pop("executable", command[0])
    restored = "1" if options.pop("restore_signals", True) else ""
    passed = (*options.pop("pass_fds", ()), held, failed)
    try:
        process = subprocess.Popen([sys.executable, "-I", "-S", "-c", GATE, str(held), str(failed), restored,
                                    program, *command], start_new_session=True, pass_fds=passed, **options)
    except BaseException:
        for end in (write, release, told):
            os.close(end)
        line.wait()
        raise
    finally:
        os.close(held)
        os.close(failed)
    try:
        with _HELD:
            _LIFELINES[process.pid] = (write, line)
        process.args = command
        with open(release, "wb", buffering=0) as releasing, open(told, "rb") as said:
            os.write(write, f"{process.pid}\n".encode())
            releasing.write(b"1")
            # Closed by the command's execution, or written with its error.
            number = said.read()
        if number:
            raise OSError(int(number), os.strerror(int(number)), program)
    except BaseException:
        stop(process)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        raise
    return process


def stop(process: subprocess.Popen, grace: float = 0.0, interrupt: int = signal.SIGTERM,
         settle: float = SETTLE_SECONDS) -> list[int]:
    """Stop the group `process` leads: with a `grace`, `interrupt` to every process in it and up to `grace`
    seconds for the leader to exit, left unreaped; then SIGKILL to the group, while its id is still the
    leader's own, and, where the group was `start`ed, its lifeline told the group is stopped, which kills
    what is left by the group's id: a leader reaped before is no bar, which CI's is (it reaps pytest before
    it stops what pytest left). Then up to `settle` seconds, one budget, for every process of it to end, and
    the leader reaped once it has exited. Returns the processes still running then (none, normally; a
    leader the kill has not reached, in uninterruptible sleep, among them), never raised, so a caller can
    still report what ran."""
    if grace > 0 and exit_code(process) is None and signal_group(process, interrupt):
        groups_ended((process,), grace)
    killed = signal_group(process, signal.SIGKILL)
    with _HELD:
        held = _LIFELINES.pop(process.pid, None)
    if held:
        os.close(held[0])
        held[1].wait()
        killed = True
    left: list[int] = []
    if killed:
        deadline = time.monotonic() + settle
        while (left := members(process.pid)) and time.monotonic() < deadline:
            time.sleep(POLL_SECONDS)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(0)
    return left


def _outlived(process: subprocess.Popen, left: list[int], settle: float) -> None:
    """Raise where processes of the group `process` led outlived its SIGKILL for `settle` seconds."""
    if left:
        raise TimeoutError(f"processes {left} of the group {process.pid} led outlived its SIGKILL by "
                           f"{settle:.0f} s")


def end_group(process: subprocess.Popen, grace: float = 0.0, interrupt: int = signal.SIGTERM) -> int:
    """End the group `process` leads: with a `grace`, `interrupt` to every process in it and up to
    `grace` seconds for the leader to exit; then SIGKILL to every process left in the group, a child
    the leader left running among them, sent before the leader is reaped, while the group's id is
    still its own; then the leader reaped. Returns its exit code. A process that left the group on
    purpose (a session of its own) is out of reach."""
    left = stop(process, grace, interrupt)
    code = process.wait()
    _outlived(process, left, SETTLE_SECONDS)
    return code


@contextlib.contextmanager
def _ending(started: Callable[[], subprocess.Popen], end: Callable[[subprocess.Popen], None]
            ) -> Iterator[subprocess.Popen]:
    """The process `started` returns, `end`ed with the block on every exit, a success's too, and its pipes
    closed after it, each step attempted whichever fail (`attempted`)."""
    process = started()
    try:
        yield process
    finally:
        streams = [stream for stream in (process.stdin, process.stdout, process.stderr) if stream is not None]
        attempted([("end the process group", lambda: end(process)),
                   *((f"close {stream!r}", stream.close) for stream in streams)], primary=sys.exception())


def process_group(command: Sequence[str], *, grace: float = 0.0, interrupt: int = signal.SIGTERM,
                  **options) -> AbstractContextManager[subprocess.Popen]:
    """`command` started as the leader of a session and process group of its own (`subprocess.Popen`
    with `options`), whose group ends with the block on every exit, a success's too (`end_group`),
    and whose pipes close after it."""
    def ended(process: subprocess.Popen) -> None:
        end_group(process, grace, interrupt)

    return _ending(lambda: subprocess.Popen(command, start_new_session=True, **options), ended)


def owned_group(command: Sequence[str], *, grace: float = 0.0, interrupt: int = signal.SIGTERM,
                settle: float = SETTLE_SECONDS, **options) -> AbstractContextManager[subprocess.Popen]:
    """`command` `start`ed (`options` as Popen's), whose group ends with the block on every exit, a
    success's too, and whose pipes close after it: `stop`ped with the `grace`, `interrupt` and `settle`
    given, an error naming the processes that outlived it. The group outlives no owner, as `start`'s
    does not: the one for a long process whose owner may be killed, a worker or a server on a GPU."""
    def ended(process: subprocess.Popen) -> None:
        _outlived(process, stop(process, grace, interrupt, settle), settle)

    return _ending(lambda: start(command, **options), ended)


def lifeline(owner: int) -> None:
    """A group's lifeline, reading its owner pipe `owner`: the group's leader, written as the group starts,
    then the pipe's close, the owner gone or the group stopped, upon which it kills the group. A group whose
    leader ends with no process of it left ends the lifeline with no kill, as the group's id may then name
    another's; one whose descendants outlive its leader is still killed with its owner. An owner gone before
    it named a group leaves none to kill."""
    named = os.read(owner, 32).strip()
    if not named:
        return
    group = int(named)
    try:
        watched = [owner, os.pidfd_open(group)]
    except ProcessLookupError:
        watched = [owner]
        if not members(group):
            return
    # Readable as the owner's pipe closes, or as the leader ends.
    while owner not in select.select(watched, [], [])[0]:
        if not members(group):
            return
        watched = [owner]
    _killpg(group, signal.SIGKILL)


if __name__ == "__main__":
    lifeline(int(sys.argv[1]))
