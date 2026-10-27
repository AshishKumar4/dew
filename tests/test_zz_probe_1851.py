"""Probe (not for landing): size and time the cut decision sites of row 1851."""

import statistics
import time

import jax
import jax.numpy as jnp
import pytest
from reference_error import _arms, seeds_for


def per_call(label, fn, *args):
    jax.block_until_ready(fn(*args))
    seconds = []
    for _ in range(5):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        seconds.append(time.perf_counter() - start)
    print("PROBE", label, f"{statistics.median(seconds):.3f}")


def recorder(calls):
    def recorded(dew_fn, reference_fn, draw, label, *, seeds, truth_fn=None, orders=None):
        calls.update(dew=dew_fn, reference=reference_fn, draw=draw, truth=truth_fn)
        raise StopIteration
    return recorded


def test_packing_marker_one_window(monkeypatch):
    import test_decision_packing as packing

    calls = {}
    monkeypatch.setattr(packing, "assert_as_exact", recorder(calls))
    objective = packing.objective_for("marker")
    with pytest.raises(StopIteration):
        packing.assert_rows_decided_alone(objective, (objective.model.backbone.clone(dtype=jnp.float64),
                                                      objective.model.head.clone(dtype=jnp.float64)), 8, 1)
    start = time.perf_counter()
    seeds = seeds_for(calls["reference"], calls["draw"], "marker one window", truth_fn=calls["truth"],
                      dew_fn=calls["dew"])
    print("PROBE packing marker one window seeds", seeds, f"{time.perf_counter() - start:.0f}")


@pytest.mark.parametrize("counts", [(24, 6, 12), (32, 6, 16)])
def test_option_slots(counts, monkeypatch):
    import test_decision_option_slots as slots

    clauses, hypotheses, intents = counts
    monkeypatch.setattr(slots, "CLAUSES", clauses)
    monkeypatch.setattr(slots, "HYPOTHESES", hypotheses)
    monkeypatch.setattr(slots, "INTENTS", intents)
    calls = {}
    monkeypatch.setattr(slots, "assert_as_exact", recorder(calls))
    with pytest.raises(StopIteration):
        slots.test_a_ragged_batch_trains_on_compacted_options_as_on_the_grid()
    mine, theirs = _arms(calls["dew"], calls["reference"])
    drawn = calls["draw"](0)
    print("PROBE slots shapes", counts, [jnp.shape(leaf) for leaf in jax.tree.leaves(drawn[1:])])
    per_call(f"slots {counts} draw", calls["draw"], 1)
    per_call(f"slots {counts} compact", mine, *drawn)
    per_call(f"slots {counts} grid", theirs, *drawn)
    start = time.perf_counter()
    seeds = seeds_for(calls["reference"], calls["draw"], f"slots {counts}", truth_fn=calls["truth"],
                      dew_fn=calls["dew"])
    print("PROBE slots seeds", counts, seeds, f"{time.perf_counter() - start:.0f}")
