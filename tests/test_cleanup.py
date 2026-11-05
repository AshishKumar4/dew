"""dew.cleanup's teardown: every step runs, and each failure says which step it was."""

import pytest

from dew.cleanup import attempted


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
