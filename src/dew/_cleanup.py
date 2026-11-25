"""Running every step of a teardown, whichever fail, and raising one error that names them all, and
the lifetime of a process group: a command started as the leader of a session of its own, whose
group ends with it on every exit.

A group `start`ed has a lifeline (`lifeline`) as well: this module run in a session of its own, which
holds the read end of a pipe whose write end only the owner holds and kills the group once that pipe
closes, as it does when the owner dies however it dies (a SIGKILL, a SIGTERM that unwinds no `with`)
or stops the group (`stop`). A group that ends on its own, its leader's or the last process its leader
left, ends its lifeline, which then kills nothing: its id is free for another process. The group stays in
a session of its own, so a rank's cleanup signals its own (the maintainer, 2026-10-28: a worker outlived its
killed owner 43 hours at 195% CPU on the VM). The standard library alone: the CI harness loads this file by
path (tools/armada/child_groups.py), as importing `dew` sets the process's XLA flags."""

import _thread
import contextlib
import functools
import os
import select
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
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
        "for name in filter(None, sys.argv[3].split(',')):\n"
        "    signal.signal(getattr(signal, name), signal.SIG_DFL)\n"
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
leaves it to exit having run nothing. It sets the signals it is given (`_restored`) to their defaults
first, as Popen leaves them in the process it executes, so the command meets them so, a pipe's writer
ending by its SIGPIPE."""


def start(command: Sequence[str], **options) -> subprocess.Popen:
    """`command` as the leader of a session and process group of its own, `options` as Popen's, with its
    lifeline (`lifeline`) started first and holding the group's id before the command runs (`GATE`), so no
    instant leaves the group's processes without one: an owner killed between the group's start and its
    lifeline's knowing it took the group with it, where the command had run beside a lifeline that never
    learned it (row 4395). Returns once the command is executing, raising the error its execution raised
    (a missing program's `FileNotFoundError`) as Popen does. The lifelines of groups that ended on their own
    are let go first, their pipes closed. A start interrupted, or whose command cannot execute, stops the
    group it started, lets its lifeline go and closes every descriptor it opened: no caller holds them.

    The start runs on a thread of its own (`_Starting`, started by `_thread` and waited for on a lock alone,
    `_Done`), which no Python signal handler runs on: one that
    raises (a Ctrl-C, a deadline's alarm) runs on the main thread, whichever thread the signal reached,
    and landing between a descriptor's creation and its holder's holding it, there or inside Popen's
    own, would leak it. An interruption reaches the caller wherever it is (launching the thread, waiting
    for it, returning), which then abandons the start (`_Starting.abandon`): a group the start has handed
    over is undone on a thread of its own, one it has yet to hand over is undone by the start itself,
    and the caller waits for either before it raises; a second interruption while it waits leaves the
    undoing to finish on its own. A start abandoned before it began starts nothing."""
    starting = _Starting(command, options)
    try:
        _thread.start_new_thread(_run, (starting.run,))
        starting.done.wait()
        return starting.result()
    except BaseException:
        process, begun = starting.abandon()
        if process is not None:
            done = _Done()
            _thread.start_new_thread(_run, (functools.partial(_undo_and_signal, process, done),))
            done.wait()
        elif begun:
            starting.done.wait()
        raise


def _run(work: Callable[[], None]) -> None:
    """`work()` on a thread `_thread` started, under the trace and profile hooks a `threading` thread takes
    (`threading.settrace`, `threading.setprofile`)."""
    sys.settrace(threading.gettrace())
    sys.setprofile(threading.getprofile())
    work()


class _Done:
    """One worker's end, which one caller waits for (`wait`) on a lock alone: a caller's interruption
    (an exception a signal handler raises) leaves it as it was wherever it lands, where `threading`'s
    waits (`Thread.start`, `Event.wait`) reacquire a condition's lock as they unwind and an interruption
    landing there made them raise "release unlocked lock" in its place (row 7619). `finished` is set
    before the lock is released, so a wait whose acquire had succeeded as it was interrupted never
    acquires again."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.lock.acquire()
        self.finished = False

    def set(self) -> None:
        self.finished = True
        self.lock.release()

    def wait(self) -> None:
        if not self.finished:
            self.lock.acquire()


def _undo_and_signal(process: subprocess.Popen, done: _Done) -> None:
    """`process`'s group undone (`_undone`), then `done` set, however the undoing ends."""
    try:
        _undone(process)
    finally:
        done.set()


class _Starting:
    """One `start` on a thread of its own (`run`): its group, handed to its caller (`result`), or the error
    that ended it, raised on the caller's thread, and, where its caller abandoned it (`abandon`), its
    undoing, by whichever of the two holds the group then: the hand-over is one step under `lock`. `done`
    is set once the start has handed its group over, raised, or undone what it started."""

    def __init__(self, command: Sequence[str], options: Mapping[str, object]) -> None:
        self.command, self.options = command, options
        self.lock = threading.Lock()
        self.done = _Done()
        self.begun = self.abandoned = False
        self.process: subprocess.Popen | None = None
        self.error: BaseException | None = None

    def run(self) -> None:
        with self.lock:
            if self.abandoned:
                return
            self.begun = True
        try:
            process = _started(self.command, **self.options)
        except BaseException as error:  # the start's own error, raised on its caller's thread (`result`)
            self.error = error
            self.done.set()
            return
        with self.lock:
            abandoned = self.abandoned
            if not abandoned:
                self.process = process
        if abandoned:
            _undone(process)
        self.done.set()

    def result(self) -> subprocess.Popen:
        if self.error is not None:
            raise self.error
        assert self.process is not None
        return self.process

    def abandon(self) -> tuple[subprocess.Popen | None, bool]:
        """Its caller's no more: the group it has handed over, for its caller to undo, and whether it has
        begun, so that its caller waits for it to undo what it starts."""
        with self.lock:
            self.abandoned = True
            process, self.process = self.process, None
            return process, self.begun


def _undone(process: subprocess.Popen) -> None:
    """The group `process` leads stopped, its lifeline let go and its pipes closed: a start its caller
    abandoned (`_Starting`)."""
    stop([process])
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            stream.close()


def _started(command: Sequence[str], **options) -> subprocess.Popen:
    """`start`'s group, started on the thread `_Starting` runs it on."""
    with _HELD:
        ended = [_LIFELINES.pop(leader) for leader, (_, held) in list(_LIFELINES.items())
                 if held.poll() is not None]
    for written, _ in ended:
        os.close(written)
    program = options.pop("executable", command[0])
    defaults = ",".join(_restored(options.pop("restore_signals", True)))
    passed = options.pop("pass_fds", ())
    # The descriptors this start holds, each closed once it is done with, or handed to `_LIFELINES` (the
    # owner pipe's write end): what is left here when it raises is closed then.
    owned: list[int] = []
    # The lifeline, until `_LIFELINES` holds it beside its pipe.
    unheld: subprocess.Popen | None = None
    process = None
    try:
        owned.extend(os.pipe2(os.O_CLOEXEC))
        read, write = owned
        # Isolated (-I): run as a script, this file's own directory would otherwise lead sys.path, where
        # dew's io.py and logging.py stand in for the standard library's.
        line = subprocess.Popen([sys.executable, "-I", __file__, str(read)], pass_fds=(read,),
                                start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        unheld = line
        _close(owned, read)
        owned.extend(os.pipe2(os.O_CLOEXEC))
        owned.extend(os.pipe2(os.O_CLOEXEC))
        held, release, told, failed = owned[1:]
        process = subprocess.Popen([sys.executable, "-I", "-S", "-c", GATE, str(held), str(failed), defaults,
                                    program, *command], start_new_session=True,
                                   pass_fds=(*passed, held, failed), **options)
        _close(owned, held)
        _close(owned, failed)
        process.args = command
        os.write(write, f"{process.pid}\n".encode())
        with _HELD:
            _LIFELINES[process.pid] = (write, line)
            owned.remove(write)
            unheld = None
        os.write(release, b"1")
        # Closed by the command's execution, or written with its error.
        number = b"".join(iter(lambda: os.read(told, 64), b""))
        if number:
            raise OSError(int(number), os.strerror(int(number)), program)
    except BaseException:
        if process is not None:
            # Its lifeline too, where the table holds it.
            _undone(process)
        for end in owned:
            os.close(end)
        owned.clear()
        if unheld is not None:
            unheld.wait()
        raise
    finally:
        for end in owned:
            os.close(end)
    return process


SIGNALS = ("SIGPIPE", "SIGXFZ", "SIGXFSZ")
"""The signals Popen's `restore_signals` sets to their defaults in the process it executes: those Python
ignores, which a Python process, the gate among them, would otherwise pass on ignored."""


def _restored(restore: bool) -> list[str]:
    """The `SIGNALS` the gate sets to their defaults before its command: every one with `restore`, as
    Popen's; without it, those this process does not ignore, which a command Popen executes inherits at
    their defaults (a handler does not outlive an exec) where the gate, a Python process, ignores them."""
    return [name for name in SIGNALS if hasattr(signal, name)
            and (restore or signal.getsignal(getattr(signal, name)) != signal.SIG_IGN)]


def _close(owned: list[int], end: int) -> None:
    """Close descriptor `end`, which `owned` then no longer holds."""
    owned.remove(end)
    os.close(end)


def stop(processes: Sequence[subprocess.Popen], grace: float = 0.0, interrupt: int = signal.SIGTERM,
         settle: float = SETTLE_SECONDS) -> list[int]:
    """Stop the groups `processes` lead, together, in their order: with a `grace`, `interrupt` to every
    process in each and up to `grace` seconds for their leaders to exit, left unreaped; then SIGKILL to
    every group, each while its id is still its leader's own, and, where a group was `start`ed, its lifeline
    told the group is stopped, which kills what is left by the group's id: a leader reaped before is no
    bar, which CI's is (it reaps pytest before it stops what pytest left). Only then up to `settle`
    seconds, one budget, for every process of them to end, and each leader reaped once it has exited: a
    group killed after another's settling outlived it, and a pool's worker that saw its peer gone ended
    on its own before its kill (row 7678). Returns the processes still running then (none, normally; a
    leader the kill has not reached, in uninterruptible sleep, among them), never raised, so a caller can
    still report what ran."""
    if grace > 0:
        interrupted = [process for process in processes
                       if exit_code(process) is None and signal_group(process, interrupt)]
        if interrupted:
            groups_ended(interrupted, grace)
    killed = {process.pid: signal_group(process, signal.SIGKILL) for process in processes}
    for process in processes:
        with _HELD:
            held = _LIFELINES.pop(process.pid, None)
        if held:
            os.close(held[0])
            held[1].wait()
            killed[process.pid] = True
    deadline = time.monotonic() + settle
    # Each leader reaped within the budget: a leader of many threads reads as a zombie, no member, while its
    # other threads end, and only then can it be reaped.
    for process in processes:
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(max(deadline - time.monotonic(), 0.0) if killed[process.pid] else 0)

    def left() -> list[int]:
        return [pid for process in processes if killed[process.pid] for pid in members(process.pid)]

    running = left()
    while running and time.monotonic() < deadline:
        time.sleep(POLL_SECONDS)
        running = left()
    return running


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
    left = stop([process], grace, interrupt)
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


def owned_group(command: Sequence[str], *, grace: float = 0.0, interrupt: int = signal.SIGTERM,
                settle: float = SETTLE_SECONDS, **options) -> AbstractContextManager[subprocess.Popen]:
    """`command` `start`ed (`options` as Popen's), whose group ends with the block on every exit, a
    success's too, and whose pipes close after it: `stop`ped with the `grace`, `interrupt` and `settle`
    given, an error naming the processes that outlived it. The group outlives no owner, as `start`'s
    does not: the one for a long process whose owner may be killed, a worker or a server on a GPU."""
    def ended(process: subprocess.Popen) -> None:
        _outlived(process, stop([process], grace, interrupt, settle), settle)

    return _ending(lambda: start(command, **options), ended)


RESCAN_SECONDS = 1.0
"""How often a lifeline asks which processes its group holds where it can watch none of them itself (a Python
without `os.pidfd_open`)."""


PIDFD: Callable[[int], int] | None = getattr(os, "pidfd_open", None)
"""Opens a descriptor of a process, readable when it ends (Linux's); None on a system or a Python build
without it, where a lifeline reads its group again each `RESCAN_SECONDS` instead."""


def _grouped(pid: int, group: int) -> bool:
    """Whether process `pid` is there and of process group `group`."""
    try:
        return os.getpgid(pid) == group
    except ProcessLookupError:
        return False


def _opened(opener: Callable[[int], int], pid: int) -> list[int]:
    """A descriptor of process `pid`, readable when it ends, by `opener` (`PIDFD`); none where the process is
    gone."""
    try:
        return [opener(pid)]
    except ProcessLookupError:
        return []


def _watched(pid: int, group: int) -> int | None:
    """A descriptor of process `pid`, readable when it ends, where it is a process of `group`: asked after the
    descriptor is open, so the answer is of the process the descriptor holds. None where it is gone, is
    another group's (its number taken again), or no such descriptor can be opened here (`PIDFD`)."""
    if PIDFD is None:
        return None
    for found in _opened(PIDFD, pid):
        if _grouped(pid, group):
            return found
        os.close(found)
    return None


def lifeline(owner: int) -> None:
    """A group's lifeline, reading its owner pipe `owner`: the group's leader, written as the group starts,
    then the pipe's close, the owner gone or the group stopped, upon which it kills the group. It watches
    every process of the group by a descriptor of that process (`_watched`), reading the group again as each
    ends, so it knows the moment the group holds none: it then ends with no kill, as the group's id may name
    another's from then on, however long its owner lives (it watched the leader alone, and after a leader
    that left a process it watched nothing of the group: that process's end went unseen, and its owner's end
    killed whatever group held the id by then). One whose processes outlive its leader is still killed with
    its owner: by its id, while a process it watches is alive, which keeps the id the group's. An owner gone
    before it named a group leaves none to kill."""
    named = os.read(owner, 32).strip()
    if not named:
        return
    group = int(named)
    held: dict[int, int] = {}

    def read() -> list[int]:
        """The group's processes now, each one new to `held` watched; those it could not watch."""
        unwatched = []
        for pid in members(group):
            if pid not in held:
                found = _watched(pid, group)
                if found is None:
                    unwatched.append(pid)
                else:
                    held[pid] = found
        return unwatched

    while True:
        unwatched = read()
        if not held and not unwatched:
            return
        # Readable as the owner's pipe closes, or as a process of the group ends; where one cannot be watched
        # (no `os.pidfd_open`), the group is read again each `RESCAN_SECONDS`.
        ready = select.select([owner, *held.values()], [], [], RESCAN_SECONDS if unwatched else None)[0]
        if owner in ready:
            break
        for pid in [pid for pid, found in held.items() if found in ready]:
            os.close(held.pop(pid))
    unwatched = read()
    alive = [found for found in held.values() if not select.select([found], [], [], 0)[0]]
    if alive or (unwatched and members(group)):
        _killpg(group, signal.SIGKILL)


if __name__ == "__main__":
    lifeline(int(sys.argv[1]))
