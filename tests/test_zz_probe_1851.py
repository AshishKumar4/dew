"""Probe (not for landing): where the slow decision tests of row 1851 spend their time."""

import statistics
import time

import jax
import numpy as np
import pytest


def per_call(label, fn, *args):
    jax.block_until_ready(fn(*args))
    seconds = []
    for _ in range(5):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        seconds.append(time.perf_counter() - start)
    print("PROBE", label, f"{statistics.median(seconds):.3f}")


@pytest.mark.parametrize("hypotheses", [6, 4])
def test_option_slots(hypotheses, monkeypatch):
    import test_decision_option_slots as slots
    from reference_error import _arms, widened

    from dew.decision import Choice, Example

    def ragged():
        clauses = Choice("Which clause?", [f"c{index}" for index in range(40)])
        held = {f"h{index}": Choice(f"Hypothesis {index}?", ["yes", "no", "unsaid"]) for index in range(hypotheses)}
        intents = Choice("Which intent?", [f"i{index}" for index in range(24)])
        return [Example("a clause on fees", {"clause": clauses}, {"clause": "c7"}),
                Example("a contract", held, dict.fromkeys(held, "unsaid")),
                Example("book a flight", {"intent": intents}, {"intent": "i3"})]

    calls = {}

    def recorded(dew_fn, reference_fn, draw, label, *, seeds, truth_fn=None, orders=None):
        calls.update(dew=dew_fn, reference=reference_fn, draw=draw, truth=truth_fn)
        raise StopIteration

    monkeypatch.setattr(slots, "assert_as_exact", recorded)
    monkeypatch.setattr(slots, "ragged", ragged)
    with pytest.raises((StopIteration, AssertionError)):
        slots.test_a_ragged_batch_trains_on_compacted_options_as_on_the_grid()
    if not calls:
        print("PROBE slots", hypotheses, "refused by the test's own asserts")
        return
    mine, theirs = _arms(calls["dew"], calls["reference"])
    drawn = calls["draw"](0)
    print("PROBE slots shapes", hypotheses, [np.shape(leaf) for leaf in jax.tree.leaves(drawn[1:])])
    per_call(f"slots {hypotheses} draw", calls["draw"], 1)
    per_call(f"slots {hypotheses} compact", mine, *drawn)
    per_call(f"slots {hypotheses} grid", theirs, *drawn)
    with jax.enable_x64(True):
        wide = widened(drawn)
        per_call(f"slots {hypotheses} truth", calls["truth"], *wide)


@pytest.mark.parametrize("rows", [2, 1])
def test_packing(rows, monkeypatch):
    import test_decision_packing as packing

    calls = {}

    def recorded(dew_fn, reference_fn, draw, label, *, seeds, truth_fn=None, orders=None):
        calls.update(dew=dew_fn, reference=reference_fn, draw=draw, truth=truth_fn)

    original = packing.windows
    monkeypatch.setattr(packing, "windows", lambda objective, count: original(objective, rows))
    monkeypatch.setattr(packing, "assert_as_exact", recorded)
    objective = packing.objective_for("marker")
    try:
        packing.assert_rows_decided_alone(objective, (objective.model.backbone.clone(dtype=np.float64),
                                                      objective.model.head.clone(dtype=np.float64)), 8)
    except AssertionError as error:
        print("PROBE packing", rows, "refused:", str(error)[:200])
        return
    drawn = calls["draw"](0)
    per_call(f"packing {rows} packed", calls["dew"], *drawn)
    per_call(f"packing {rows} alone", calls["reference"], *drawn)
    per_call(f"packing {rows} truth", calls["truth"], *drawn)
    per_call(f"packing {rows} draw", calls["draw"], 1)
