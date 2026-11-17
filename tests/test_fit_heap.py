"""A fit's Python heap: the collection that walks what a start made runs where the step is compiled, never
inside the loop (row 6739)."""

import gc

from affine_run import dataset
from recording import RecordingTracker
from test_trainer import make_trainer

from dew.telemetry.records import StepCompiled
from dew.training import Observe


def test_a_fits_full_collection_after_its_start_runs_where_its_step_is_compiled():
    """A fit collects Python's heap where its step is compiled, so the collection that walks what the
    start made (its traces, its lowered programs) runs before the step is reported compiled and none runs
    among its steps after it, where it ran at whichever step the start's objects first outnumbered a
    quarter of the heap (row 6739: 0.75 to 0.9 s inside a 4B fit's loop). Read from the fit's own events:
    each collection as the collector starts it (`gc.callbacks`), the step's report and each step."""
    events: list[tuple[str, int]] = []

    def collecting(phase, info):
        if phase == "start":
            events.append(("collection", info["generation"]))

    class Reported(RecordingTracker):
        def artifact(self, value, step):
            if isinstance(value, StepCompiled):
                events.append(("compiled", step))
            super().artifact(value, step)

    gc.callbacks.append(collecting)
    try:
        make_trainer(tracker=Reported()).fit(dataset(), steps=12, log_every=1,
                                             observe=Observe(1, lambda state: events.append(("step", 0))))
    finally:
        gc.callbacks.remove(collecting)
    compiled = events.index(("compiled", 0))
    before = [event for event in events[:compiled] if event[0] == "collection"]
    assert before and before[-1] == ("collection", 2), events[:compiled]
    steps = [index for index, event in enumerate(events) if event[0] == "step"]
    assert len(steps) == 12 and steps[0] > compiled
    assert ("collection", 2) not in events[compiled:], events[compiled:]
