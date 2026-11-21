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


def pending() -> list[list[int]]:
    """Objects made pending, more than a quarter of the heap the last full collection left, which the
    collector is kept from collecting while they are made: a fit after them meets what a start's compiles
    make, whatever this process held before (row 8771: a fit whose step the shared XLA cache served made
    too few for its start to collect in a task beside other tests' heaps)."""
    gc.collect()
    gc.disable()
    try:
        return [[index] for index in range(sys.getallocatedblocks() // 2)]
    finally:
        gc.enable()


def test_no_full_collection_runs_among_a_fits_steps_and_a_held_steps_fit_forces_none():
    """No full collection runs among a fit's steps (row 6739: one walking what the start made ran inside a 4B
    fit's loop, 0.75 to 0.9 s): a fresh trainer's first fit, its heap grown by more than a quarter since the
    last full collection (`pending`, as a start's compiles grow it), collects before its step is reported
    compiled. A second fit of the same trainer, whose step the process holds and whose heap grew little,
    forces none, and none of the collector's own runs among its steps either; one after objects made between
    fits collects them before its steps, not among them, the collector enabled throughout (rows 8511, 8545:
    the growth read against the last full collection, a plan's compile and an earlier fit's counted)."""
    tracker = Reported()
    trainer = make_trainer(tracker=tracker)
    made = pending()
    first = _fit_events(trainer, tracker, 12)
    del made
    compiled = first.index(("compiled", 0))
    steps = [index for index, event in enumerate(first) if event[0] == "step"]
    assert len(steps) == 12 and steps[0] > compiled
    assert ("collection", 2) in first[:compiled], first[:compiled]
    assert ("collection", 2) not in first[compiled:], first[compiled:]
    again = _fit_events(trainer, tracker, 3)
    assert ("collection", 2) not in again, again
    made = pending()
    grown = _fit_events(trainer, tracker, 3)
    first_step = grown.index(("step", 0))
    assert ("collection", 2) in grown[:first_step], grown
    assert ("collection", 2) not in grown[first_step:], grown
    del made


def test_a_compile_mid_fit_collects_before_the_steps_after_it():
    """A ramped fit compiles its step again where its ramp's full batch begins: objects made among the first
    stage's steps, more than a quarter of the heap the last full collection left, are collected at that
    compile, before the steps after it, none of the collector's own full collections among those either
    (row 8760: only a fit's first compile collected, so they waited for the collector among the steps)."""
    import numpy as np

    from dew.data import Loading, Memory
    from dew.data.compile import compile
    from dew.data.dataset import Ramp, ramped
    from dew.data.pipeline import Pipeline

    rng = np.random.default_rng(0)
    x = rng.normal(size=(128, 3)).astype(np.float32)
    # Its rows a multiple of the devices' at every stage: eight, then sixteen.
    loader = ramped(compile(Pipeline(Memory({"x": x, "y": 2 * x[:, :2]})).repeat().batch(16),
                            loading=Loading(workers=0)), Ramp(start=8, increment=8, samples=16))
    tracker = Reported()
    trainer = make_trainer(tracker=tracker)
    tracker.events = events = []
    made: list[list[int]] = []

    def stepped(state):
        events.append(("step", 0))
        if not made:
            made.extend(pending())

    def collecting(phase, info):
        if phase == "start":
            events.append(("collection", info["generation"]))

    gc.callbacks.append(collecting)
    try:
        trainer.fit(loader, steps=4, log_every=1, observe=Observe(1, stepped))
    finally:
        gc.callbacks.remove(collecting)
    compiled = [index for index, event in enumerate(events) if event[0] == "compiled"]
    assert len(compiled) == 2, events
    first_step = events.index(("step", 0))
    assert ("collection", 2) in events[first_step:compiled[1]], events
    assert ("collection", 2) not in events[compiled[1]:], events
