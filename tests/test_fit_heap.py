"""A fit's Python heap: the collection that walks what a start made runs where the step is compiled, never
inside the loop (row 6739)."""

import gc
import sys

from affine_run import dataset
from recording import RecordingTracker
from test_trainer import make_trainer

from dew.telemetry.records import StepCompiled
from dew.training import Observe


class Reported(RecordingTracker):
    """A tracker noting, in `events`, each step's report as compiled."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, int]] = []

    def artifact(self, value, step):
        if isinstance(value, StepCompiled):
            self.events.append(("compiled", step))
        super().artifact(value, step)


def _fit_events(trainer, tracker: Reported, steps: int) -> list[tuple[str, int]]:
    """A fit's own events, in order: each collection as the collector starts it (`gc.callbacks`), with its
    generation, the step's report as compiled, and each step."""
    tracker.events = events = []

    def collecting(phase, info):
        if phase == "start":
            events.append(("collection", info["generation"]))

    gc.callbacks.append(collecting)
    try:
        trainer.fit(dataset(), steps=steps, log_every=1,
                    observe=Observe(1, lambda state: events.append(("step", 0))))
    finally:
        gc.callbacks.remove(collecting)
    return events


def test_no_full_collection_runs_among_a_fits_steps_and_a_held_steps_fit_forces_none():
    """No full collection runs among a fit's steps (row 6739: one walking what the start made ran inside a 4B
    fit's loop, 0.75 to 0.9 s): a fresh trainer's first fit, whose compiles grew the heap by more than a
    quarter since the last full collection, collects before its step is reported compiled. A second fit of
    the same trainer, whose step the process holds and whose heap grew little, forces none, and none of the
    collector's own runs among its steps either, the collector enabled throughout (rows 8511, 8545: the
    growth read against the last full collection, a plan's compile and an earlier fit's counted)."""
    tracker = Reported()
    trainer = make_trainer(tracker=tracker)
    first = _fit_events(trainer, tracker, 12)
    compiled = first.index(("compiled", 0))
    steps = [index for index, event in enumerate(first) if event[0] == "step"]
    assert len(steps) == 12 and steps[0] > compiled
    assert ("collection", 2) in first[:compiled], first[:compiled]
    assert ("collection", 2) not in first[compiled:], first[compiled:]
    again = _fit_events(trainer, tracker, 3)
    assert ("collection", 2) not in again, again
    # Objects made between fits, more than a quarter of the heap the last full collection left, are pending:
    # a fit whose step the process holds collects them before its steps, not among them (row 8511).
    gc.disable()  # Made pending, none collected while made.
    try:
        made = [[index] for index in range(sys.getallocatedblocks() // 2)]
    finally:
        gc.enable()
    grown = _fit_events(trainer, tracker, 3)
    first_step = grown.index(("step", 0))
    assert ("collection", 2) in grown[:first_step], grown
    assert ("collection", 2) not in grown[first_step:], grown
    del made
