"""A tracker that keeps everything a run reports, for a test to read back, and one that preempts
the run it reports."""

import os
import signal


class RecordingTracker:
    """Every `log` as `(step, scalars)` and every `artifact` as `(step, value)`, in order."""

    def __init__(self):
        self.scalars: list[tuple[int, dict]] = []
        self.artifacts: list[tuple[int, object]] = []

    def log(self, scalars, step):
        self.scalars.append((step, dict(scalars)))

    def artifact(self, value, step):
        self.artifacts.append((step, value))


class Preempting(RecordingTracker):
    """Sends this process the scheduler's SIGTERM, once, when step `at` is
    logged. The loop reads a step's log once the next step is dispatched, so
    the notice stops the run after step `at` + 1."""

    def __init__(self, at):
        super().__init__()
        self.at = at

    def log(self, scalars, step):
        super().log(scalars, step)
        if step == self.at:
            self.at = None
            os.kill(os.getpid(), signal.SIGTERM)
