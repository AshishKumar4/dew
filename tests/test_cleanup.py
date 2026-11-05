"""dew.cleanup's process groups: real processes, each the leader of a session of its own, ended as the
library and the CI harness end them."""

import contextlib
import os
import signal
import subprocess
import sys

import pytest
from process_support import running
from test_process_support import gone_within

from dew import cleanup

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
        assert cleanup.end_group(leader) == -signal.SIGKILL
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
        assert cleanup.end_group(leader, grace=30 if handled else 1) == code


def test_a_group_whose_leader_was_reaped_is_signalled_only_where_its_owner_started_it():
    """CI reaps a leader before it stops the group, and the processes the leader left are its to stop,
    gone when the stop returns, since a group `start`ed holds its id by its lifeline; a group nothing holds
    is not signalled once its leader is reaped, as its id may be another process's by then."""
    with started(leader_with_child("")) as unowned:
        stranded = child_of(unowned)
        unowned.wait()
        assert cleanup.stop(unowned) == []
        assert running(stranded)
        os.kill(stranded, signal.SIGKILL)
    owned = cleanup.start([sys.executable, "-c", leader_with_child("")], stdout=subprocess.PIPE, text=True)
    try:
        held = child_of(owned)
        owned.wait()
        assert cleanup.stop(owned) == []
        assert not running(held)
    finally:
        owned.stdout.close()


class Own(Exception):
    """A block's own error, which no cleanup failure may replace."""


def test_a_group_owned_for_a_block_ends_with_it_and_a_failing_close_is_a_note_on_the_blocks_own_error():
    """A group `owned_group` starts ends with its block, an error among the ways it ends; a pipe that fails
    to close (a request the ended process never read, left unflushed in it) is a note on the block's own
    error, which stays the one raised, and the rest of the cleanup is done."""
    own = Own("the block's own")
    command = [sys.executable, "-c", leader_with_child("time.sleep(600)")]
    with pytest.raises(Own) as raised, cleanup.owned_group(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as leader:
        child = child_of(leader)
        leader.stdin.write("a request nothing reads")
        raise own
    assert raised.value is own and len(own.__notes__) == 1
    assert leader.returncode is not None and leader.stdout.closed
    assert gone_within(child, 30)


def test_a_process_group_ends_with_its_block_and_closes_its_pipes():
    """A `process_group` ends the group, the child its leader left among it, as its block ends."""
    with cleanup.process_group([sys.executable, "-c", leader_with_child("time.sleep(600)")],
                               stdout=subprocess.PIPE, text=True) as leader:
        child = child_of(leader)
    assert leader.returncode is not None and leader.stdout.closed
    assert gone_within(child, 30)
