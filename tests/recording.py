"""A tracker that keeps everything a run reports, for a test to read back, and an observer that
preempts the run it observes."""

import dataclasses
import os
import signal
from collections.abc import Callable

from dew.state import TrainState
from dew.training.observe import Observe


class RecordingTracker:
    """Every `log` as `(step, scalars)` and every `artifact` as `(step, value)`, in order."""

    def __init__(self):
        self.scalars: list[tuple[int, dict]] = []
        self.artifacts: list[tuple[int, object]] = []

    def log(self, scalars, step):
        self.scalars.append((step, dict(scalars)))

    def artifact(self, value, step):
        self.artifacts.append((step, value))


@dataclasses.dataclass
class Preempting(Observe):
    """An observer that sends this process the scheduler's SIGTERM, once, after step `at`, as the loop
    observes it (`Observe.at`), and observes as `Observe` does at its own cadence (`every`, `call`;
    unset, never). The loop reads a notice as the next step ends, so it stops the run after step
    `at` + 1, at the same step every run, as a tracker's log, written off the loop, would not."""

    every: int = 10 ** 9
    call: Callable[[TrainState], object] = lambda state: None
    at_step: int | None = None

    def at(self, step, state):
        if step == self.at_step:
            self.at_step = None
            os.kill(os.getpid(), signal.SIGTERM)
        super().at(step, state)


def preempting(at: int, **observed) -> Preempting:
    """`Preempting` after step `at`."""
    return Preempting(at_step=at, **observed)
