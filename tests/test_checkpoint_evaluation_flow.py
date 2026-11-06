"""Checkpoint-cadenced scoring through the real evaluation transfers and Orbax save.

This is a CPU lifecycle check, not a reproduction of an accelerator stream stall.
"""

import jax
import numpy as np
import optax
import pytest
from affine_run import Features, Spread, dataset, val_batches

from dew import _coordination
from dew.checkpoints import Checkpoints
from dew.training import EvalSuite, Trainer
from dew.training import evaluation


class ReportedFeatures(Features):
    """The same compiled combined loss/artifact route used by decision validation."""

    def _validation_scores(self, variables, batch, step):
        return self.loss(variables, batch, step)[0], self.evaluate(variables, batch, step)


@pytest.mark.parametrize("cadence", [None, "20%"], ids=["final-only", "checkpoint-ranking"])
def test_checkpoint_ranking_materializes_all_suites_before_saving(tmp_path, monkeypatch, cadence):
    """A checkpoint requests ranking even with eval_every=None; every suite's compiled
    report is materialized and scored before that step's save starts. Earlier saves use
    the real asynchronous checkpointer, and training continues with its donated state.
    """
    events = []
    active = []
    readbacks = []
    pinned = []
    original_evaluate = Trainer._evaluate
    original_host = evaluation.collective_host
    original_pinned = _coordination.pinned_get
    original_save = Checkpoints.save

    def evaluate(self, state, shardings, reader, metrics, preview, **options):
        context = (int(state.step), options.get("split", "val"), options.get("warm", False))
        active.append(context)
        events.append(("evaluate", *context))
        try:
            return original_evaluate(self, state, shardings, reader, metrics, preview, **options)
        finally:
            events.append(("evaluated", *context))
            active.pop()

    def home(value, *, phase, **options):
        scoring = phase.startswith("scoring batch ")
        if scoring:
            assert active
            assert any(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(value))
        hosted = original_host(value, phase=phase, **options)
        if scoring:
            assert not any(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(hosted))
            assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(hosted))
            readbacks.append((*active[-1], phase))
        return hosted

    def pinned_get(leaves):
        if active:
            pinned.append(active[-1])
        return original_pinned(leaves)

    def save(self, step, *arguments, **options):
        assert not active
        events.append(("save", step))
        return original_save(self, step, *arguments, **options)

    monkeypatch.setattr(Trainer, "_evaluate", evaluate)
    monkeypatch.setattr(evaluation, "collective_host", home)
    monkeypatch.setattr(_coordination, "pinned_get", pinned_get)
    monkeypatch.setattr(Checkpoints, "save", save)
    seen = []
    metric = Spread(seen)
    source = dataset(val=val_batches(3))
    suites = {
        name: EvalSuite(source.val, (metric,), None)
        for name in ("first", "second")
    }
    checkpoints = Checkpoints(str(tmp_path / "run"), keep=2)
    run = Trainer(ReportedFeatures(), optax.sgd(0.05), key=0, checkpoints=checkpoints)
    state = run.fit(source, steps=10, log_every=2, eval_every=None,
                    checkpoint_every=cadence, validation=suites, preview=False)
    checkpoints.wait()
    assert int(state.step) == checkpoints.latest == 10
    expected = [10] if cadence is None else [2, 4, 6, 8, 10]
    assert [event[1] for event in events if event[0] == "save"] == expected
    assert [(event[1], event[2]) for event in events
            if event[0] == "evaluated" and not event[3]] == [
        (step, split) for step in expected for split in suites
    ]
    assert [(step, split) for step, split, warm, _ in readbacks if not warm] == [
        (step, split) for step in expected for split in suites for _ in range(3)
    ]
    assert all((step, split, warm) in pinned for step, split, warm, _ in readbacks)
    for step in expected:
        saved = events.index(("save", step))
        assert all(events.index(("evaluated", step, split, False)) < saved for split in suites)
    # No save of this run precedes the first checkpoint-triggered evaluation.
    first = events.index(("evaluate", expected[0], "first", False))
    assert not any(event[0] == "save" for event in events[:first])
    assert seen
    assert all(np.isfinite(value) for entry in checkpoints.kept() for value in entry.metrics.values())
