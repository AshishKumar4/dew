"""dew._cleanup's teardown: every step runs, and each failure says which step it was."""

import pytest

from dew._cleanup import attempted


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
