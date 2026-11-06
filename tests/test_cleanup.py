"""dew._cleanup: its teardown, every step run and each failure saying which step it was, and its process
groups, real processes, each the leader of a session of its own, ended as the library and the CI harness end
them."""

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from process_support import running
from test_process_support import gone_within

from dew._cleanup import attempted, end_group, owned_group, process_group, start, stop

LEAVES_A_CHILD = (
    "import os, time\n"
    "child = os.fork()\n"
    "if child == 0:\n"
    "    time.sleep(600)\n"
    "print(child, flush=True)\n"
    "{leader}\n")
"""A leader whose child outlives it; the leader then ends at once or waits."""


def leader_with_child(end: str) -> str:
    return LEAVES_A_CHILD.format(leader=end)


def child_of(process: subprocess.Popen) -> int:
    return int(process.stdout.readline())


@contextlib.contextmanager
def started(program: str, **options):
    """`program` as a group of its own that nothing owns, ended by the test whatever it asserts."""
    process = subprocess.Popen([sys.executable, "-c", program], start_new_session=True,
                               stdout=subprocess.PIPE, text=True, **options)
    try:
        yield process
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        process.stdout.close()


def test_ending_a_group_ends_the_child_its_leader_left_and_gives_the_leaders_code():
    """A leader still running, and a child it started, are both gone once the group is ended, the leader's
    exit status that of its SIGKILL: no process of the group is left to hold its pipes or its memory."""
    with started(leader_with_child("time.sleep(600)")) as leader:
        child = child_of(leader)
        assert end_group(leader) == -signal.SIGKILL
        assert gone_within(child, 30)


@pytest.mark.parametrize(("handled", "code"), [(True, 5), (False, -signal.SIGKILL)])
def test_a_leader_is_given_its_grace_to_end_by_its_own_hand_before_the_group_is_killed(handled, code):
    """With a grace, the interrupt reaches the leader first: one that handles it and exits by its own
    status keeps that status, and one that ignores it is killed when the grace is over."""
    action = "lambda number, frame: sys.exit(5)" if handled else "signal.SIG_IGN"
    program = ("import signal, sys, time\n"
               f"signal.signal(signal.SIGTERM, {action})\n"
               "print('handled', flush=True)\n"
               "time.sleep(600)\n")
    with started(program) as leader:
        leader.stdout.readline()
        assert end_group(leader, grace=30 if handled else 1) == code


def test_a_group_whose_leader_was_reaped_is_signalled_only_where_its_owner_started_it():
    """CI reaps a leader before it stops the group, and the processes the leader left are its to stop,
    gone when the stop returns, since a group `start`ed holds its id by its lifeline; a group nothing holds
    is not signalled once its leader is reaped, as its id may be another process's by then."""
    with started(leader_with_child("")) as unowned:
        stranded = child_of(unowned)
        unowned.wait()
        assert stop(unowned) == []
        assert running(stranded)
        os.kill(stranded, signal.SIGKILL)
    owned = start([sys.executable, "-c", leader_with_child("")], stdout=subprocess.PIPE, text=True)
    try:
        held = child_of(owned)
        owned.wait()
        assert stop(owned) == []
        assert not running(held)
    finally:
        owned.stdout.close()


def test_a_group_whose_owner_is_killed_before_its_lifeline_knows_it_ends_with_its_owner(tmp_path):
    """An owner killed between its group's start and its lifeline's reading the group's id (the window an
    owner busy elsewhere, its other thread holding the interpreter, keeps open while its command already
    runs) takes the group with it: the command runs only once its lifeline holds the group (row 4395),
    where the group had outlived its owner in its session of its own. The command here kills the owner
    itself, as soon as it runs."""
    recorded = tmp_path / "pid"
    command = ["sh", "-c", f'echo $$ > {recorded}; kill -9 "$PPID"; exec sleep 600']
    program = ("import random, sys, threading, time\n"
               "from dew._cleanup import start\n"
               "numbers = [random.random() for _ in range(2_000_000)]\n"
               "def busy():\n"
               "    while True:\n"
               "        sorted(numbers)\n"
               "threading.Thread(target=busy, daemon=True).start()\n"
               f"start({command!r})\n"
               "time.sleep(600)\n")
    owner = subprocess.Popen([sys.executable, "-c", program], start_new_session=True)
    try:
        assert owner.wait(timeout=120) == -signal.SIGKILL
        command_pid = int(recorded.read_text())
        try:
            assert gone_within(command_pid, 30)
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.kill(command_pid, signal.SIGKILL)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(owner.pid, signal.SIGKILL)
        owner.wait()


def test_a_started_command_that_cannot_execute_raises_its_error_as_popen_does_and_leaves_no_group():
    """A program `start` cannot execute raises from `start` the error Popen raises for it, naming it."""
    with pytest.raises(FileNotFoundError) as raised:
        start(["a-program-no-path-holds"], stdout=subprocess.PIPE)
    assert raised.value.filename == "a-program-no-path-holds"


@pytest.mark.parametrize(("disposition", "restore"),
                         [("SIG_IGN", True), ("SIG_IGN", False), ("SIG_DFL", False)])
def test_a_started_command_meets_sigpipe_as_popen_leaves_it(disposition, restore):
    """The command `start` executes behind its gate, a Python process that ignores SIGPIPE, meets SIGPIPE
    as Popen leaves it: at its default where Popen restores it (`restore_signals`), and without that as its
    owner holds it, ignored where it is ignored and at its default where it is: one that sends itself
    SIGPIPE ends by it, or runs on to its own exit. Its owner here a process of its own, which sets the
    disposition."""
    program = ("import signal, subprocess, sys\n"
               "from dew._cleanup import start, stop\n"
               f"signal.signal(signal.SIGPIPE, signal.{disposition})\n"
               "command = ['sh', '-c', 'kill -PIPE $$; exit 3']\n"
               f"plain = subprocess.run(command, restore_signals={restore}).returncode\n"
               f"started = start(command, restore_signals={restore})\n"
               "print(plain, started.wait(timeout=60))\n"
               "stop(started)\n")
    done = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=120)
    ended = -signal.SIGPIPE if restore or disposition == "SIG_DFL" else 3
    assert done.stdout == f"{ended} {ended}\n", done.stdout + done.stderr


class Interrupted(Exception):
    """What a caller's deadline, or a Ctrl-C, raises in a start under way."""


def lifelines() -> int:
    """How many lifelines this process has started that still run (zombies, which have no command line,
    aside)."""
    found = 0
    for entry in Path("/proc").iterdir():
        with contextlib.suppress(OSError, IndexError):
            parent = int((entry / "stat").read_text().rpartition(")")[2].split()[1])
            found += parent == os.getpid() and b"_cleanup.py" in (entry / "cmdline").read_bytes()
    return found


def test_a_start_interrupted_before_its_command_runs_leaves_nothing_behind(tmp_path):
    """An interruption that reaches `start` while its group starts (in its lifeline's or its gate's Popen,
    or while its gate is released and executes the command) leaves nothing of it: its command, once it
    runs, does not outlive the start that raised, though its owner, the test, runs on, and no lifeline of
    it nor descriptor it opened is left. Alarms a millisecond apart land across the start's steps. The
    command writes its id whole (a write beside it renamed into place), so a command stopped as it wrote
    leaves none to read, never a part."""
    armed = {"on": False}

    def interrupt(number: int, frame: object) -> None:
        if armed["on"]:
            raise Interrupted

    previous = signal.signal(signal.SIGALRM, interrupt)
    interrupted, survivors = 0, []
    descriptors, held = len(os.listdir("/proc/self/fd")), lifelines()
    try:
        for step in range(40):
            recorded = tmp_path / f"{step}.pid"
            armed["on"] = True
            signal.setitimer(signal.ITIMER_REAL, 0.001 * (step + 1))
            try:
                started = start(["sh", "-c", f"echo $$ > {recorded}.part && mv {recorded}.part {recorded}; "
                                              "exec sleep 600"])
                armed["on"] = False
            except Interrupted:
                interrupted += 1
            else:
                stop(started)
                continue
            finally:
                armed["on"] = False
                signal.setitimer(signal.ITIMER_REAL, 0)
            time.sleep(0.5)
            if recorded.exists() and not gone_within(pid := int(recorded.read_text()), 5):
                survivors.append(pid)
                os.kill(pid, signal.SIGKILL)
    finally:
        signal.signal(signal.SIGALRM, previous)
    # A lifeline let go ends on its own pipe's close, a moment after.
    deadline = time.monotonic() + 30
    while lifelines() > held and time.monotonic() < deadline:
        time.sleep(0.1)
    assert interrupted and not survivors
    assert (len(os.listdir("/proc/self/fd")), lifelines()) == (descriptors, held)


@pytest.mark.parametrize(("opened", "nth"), [("pipe2", 1), ("pipe2", 2), ("pipe2", 3), ("pipe", 1),
                                             ("pipe", 2)])
def test_a_start_interrupted_as_a_descriptor_opens_leaves_it_closed(opened, nth):
    """An interruption that lands the moment one of `start`'s descriptors opens (its own pipes, `os.pipe2`,
    or the one Popen opens to hear its child's execution, `os.pipe`), before anything holds it, closes it
    and leaves no lifeline: the start lets it land once every descriptor it opened is held. A signal
    raised as the call returns, its handler raising, is how one lands there (a Ctrl-C, a deadline's alarm);
    a millisecond's alarm reached it rarely, leaking the pair (two trains' cleanup reds)."""
    calls = {"seen": 0}

    def interrupt(number: int, frame: object) -> None:
        raise Interrupted

    def profile(frame: object, event: str, called: object) -> None:
        if event == "c_return" and called is getattr(os, opened):
            calls["seen"] += 1
            if calls["seen"] == nth:
                signal.raise_signal(signal.SIGALRM)

    previous = signal.signal(signal.SIGALRM, interrupt)
    descriptors, held = len(os.listdir("/proc/self/fd")), lifelines()
    try:
        sys.setprofile(profile)
        with pytest.raises(Interrupted):
            try:
                stop(start(["sleep", "600"]))
            finally:
                sys.setprofile(None)
    finally:
        signal.signal(signal.SIGALRM, previous)
    assert calls["seen"] >= nth
    # A lifeline let go ends on its own pipe's close, a moment after.
    deadline = time.monotonic() + 30
    while lifelines() > held and time.monotonic() < deadline:
        time.sleep(0.1)
    assert (len(os.listdir("/proc/self/fd")), lifelines()) == (descriptors, held)


def test_threads_starting_their_first_groups_at_once_in_a_process_without_dew_all_start_them(tmp_path):
    """CI's harness never imports dew (its XLA flags would reach every test process), so its first group's
    start loads dew._cleanup by its path (tools/armada/child_groups.py), from as many threads as start
    groups at once (ci.py's gpu-lane): every one gets the module whole and starts its group."""
    loader = Path(__file__).resolve().parents[1] / "tools" / "armada" / "child_groups.py"
    program = ("import importlib.util, sys, threading\n"
               f"spec = importlib.util.spec_from_file_location('child_groups', {str(loader)!r})\n"
               "groups = importlib.util.module_from_spec(spec)\n"
               "spec.loader.exec_module(groups)\n"
               "barrier, started, failed = threading.Barrier(16), [], []\n"
               "def one():\n"
               "    barrier.wait()\n"
               "    try:\n"
               "        started.append(groups.start([sys.executable, '-c', 'pass']))\n"
               "    except Exception as error:\n"
               "        failed.append(repr(error))\n"
               "threads = [threading.Thread(target=one) for _ in range(16)]\n"
               "for thread in threads:\n"
               "    thread.start()\n"
               "for thread in threads:\n"
               "    thread.join()\n"
               "left = [pid for process in started for pid in groups.stop(process)]\n"
               "print(len(started), failed, left)\n")
    for _ in range(5):
        done = subprocess.run([sys.executable, "-I", "-c", program], capture_output=True, text=True,
                              timeout=300, cwd=tmp_path)
        assert done.stdout == "16 [] []\n", done.stdout + done.stderr


class Own(Exception):
    """A block's own error, which no cleanup failure may replace."""


def test_a_group_owned_for_a_block_ends_with_it_and_a_failing_close_is_a_note_on_the_blocks_own_error():
    """A group `owned_group` starts ends with its block, an error among the ways it ends; a pipe that fails
    to close (a request the ended process never read, left unflushed in it) is a note on the block's own
    error, which stays the one raised, and the rest of the cleanup is done."""
    own = Own("the block's own")
    command = [sys.executable, "-c", leader_with_child("time.sleep(600)")]
    with pytest.raises(Own) as raised, owned_group(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as leader:
        child = child_of(leader)
        leader.stdin.write("a request nothing reads")
        raise own
    assert raised.value is own and len(own.__notes__) == 1
    assert leader.returncode is not None and leader.stdout.closed
    assert gone_within(child, 30)


def test_a_process_group_ends_with_its_block_and_closes_its_pipes():
    """A `process_group` ends the group, the child its leader left among it, as its block ends."""
    with process_group([sys.executable, "-c", leader_with_child("time.sleep(600)")],
                               stdout=subprocess.PIPE, text=True) as leader:
        child = child_of(leader)
    assert leader.returncode is not None and leader.stdout.closed
    assert gone_within(child, 30)


def test_every_step_runs_and_each_failure_carries_its_own_step():
    """Two failing steps around one that passes: all three run, the first failure is raised, and it
    carries each failed step's label, its own and the later one's; the passing step's label is on none."""
    ran = []

    def failing(name):
        def step():
            ran.append(name)
            raise OSError(name)

        return step

    with pytest.raises(OSError) as raised:
        attempted([("closing the rollout", failing("rollout")),
                   ("closing the log", lambda: ran.append("log")),
                   ("closing the tracker", failing("tracker"))])
    assert ran == ["rollout", "log", "tracker"]
    notes = " ".join(raised.value.__notes__)
    assert "closing the rollout" in notes and "closing the tracker" in notes
    assert "closing the log" not in notes


def test_a_run_unwinding_with_its_own_failure_keeps_it_and_notes_its_rollouts_failed_close():
    """A run unwinding with its own failure whose rollout fails to close (`dew.config._closed`) raises
    nothing of the close's: its own failure stays the one raised, noted with the close that failed, its
    step and its error (row 4276)."""
    from dew.config import _closed

    class Rollout:
        def close(self) -> None:
            raise OSError("the engine")

    primary = RuntimeError("the run's own")
    _closed(None, None, primary, Rollout())
    notes = " ".join(primary.__notes__)
    assert "closing the rollout failed" in notes and "OSError('the engine')" in notes
