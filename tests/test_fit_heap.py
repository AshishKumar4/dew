"""A fit's Python heap: the collection that walks what a start made runs where the step is compiled, never
inside the loop (row 6739)."""

import gc

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
    """No full collection runs among a fit's steps after its step is compiled (row 6739: one walking what the
    start made ran inside a 4B fit's loop, 0.75 to 0.9 s): where its compile made a quarter of the heap the
    fit collects there, before its steps. A second fit of the same trainer, whose step the process holds,
    forces none at all, the collector left to its own thresholds, where every fit walked the whole heap at
    its compile (a warm 3-step fit's 0.48 s of its 0.72)."""
    tracker = Reported()
    trainer = make_trainer(tracker=tracker)
    first = _fit_events(trainer, tracker, 12)
    compiled = first.index(("compiled", 0))
    steps = [index for index, event in enumerate(first) if event[0] == "step"]
    assert len(steps) == 12 and steps[0] > compiled
    assert ("collection", 2) not in first[compiled:], first[compiled:]
    gc.disable()
    try:
        again = _fit_events(trainer, tracker, 3)
    finally:
        gc.enable()
    assert not [event for event in again if event[0] == "collection"], again
