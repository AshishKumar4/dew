"""dew._cleanup: its teardown, every step run and each failure saying which step it was, and its process
groups, real processes, each the leader of a session of its own, ended as the library and the CI harness end
them."""

import contextlib
import os
import signal
import subprocess
import sys

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
