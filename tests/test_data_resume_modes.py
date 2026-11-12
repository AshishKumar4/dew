"""How a resume treats its data (`TrainerConfig.data_resume`, docs/design/data.md, resuming data): relaxed,
the default, reads on where the saved plan makes the same rows and starts the data again where it differs,
named in the run's records, its state resuming either way; exact refuses the difference."""

import numpy as np
import pytest
from recording import RecordingTracker
from test_run_budgets import BATCH, PEAK, Pulled, records

from dew.data import Memory
from dew.data.pipeline import Pipeline
from dew.optim import OptimConfig
from dew.telemetry.records import DataRestarted, FitStarted
from dew.training import Checkpoints, Trainer


def _fit(directory, seed: int, steps: int, *, data_resume: str = "relaxed"):
    """`Pulled` trained over a shuffle of seed `seed` of one corpus to `steps`, checkpointed in `directory`
    at every step, and what its run reported."""
    (x,) = records(0)
    chain = Pipeline(Memory({"x": x})).shuffle(seed=seed).repeat().batch(BATCH)
    tracker = RecordingTracker()
    trainer = Trainer(Pulled(), OptimConfig(learning_rate=PEAK), key=0,
                      checkpoints=Checkpoints(str(directory)), tracker=tracker, data_resume=data_resume)
    state = trainer.fit(chain, steps=steps, log_every=1, checkpoint_every=1)
    return state, tracker


def _reported(tracker, kind) -> list:
    return [value for _, value in tracker.artifacts if isinstance(value, kind)]


def test_a_relaxed_resume_whose_data_changed_starts_the_data_again_and_resumes_its_state(tmp_path):
    """A run resumed relaxed over its data shuffled by another seed starts its data again, the run's record
    naming the order's seed (`DataRestarted`), while its step and optimizer resume: it trains on from the
    step it saved, from the weights it saved, so it ends elsewhere than a fresh run of the new data."""
    _fit(tmp_path, seed=1, steps=2)
    resumed, tracker = _fit(tmp_path, seed=2, steps=4)
    (started,) = _reported(tracker, FitStarted)
    (restarted,) = _reported(tracker, DataRestarted)
    assert started.start_step == 2 and int(resumed.step) == 4
    assert restarted.step == 2 and any(name.startswith("order.") for name in restarted.differs)
    fresh, _ = _fit(tmp_path / "fresh", seed=2, steps=4)
    assert not np.array_equal(resumed.variables["params"]["w"], fresh.variables["params"]["w"])


def test_an_exact_resume_whose_data_changed_is_refused(tmp_path):
    """The same resume, exact, is refused before it trains a step."""
    _fit(tmp_path, seed=1, steps=2)
    with pytest.raises(ValueError):
        _fit(tmp_path, seed=2, steps=4, data_resume="exact")


def test_a_resume_whose_data_is_unchanged_reads_on_and_records_no_restart(tmp_path):
    """A relaxed resume of unchanged data reads on from its saved position, as an exact one does: the same
    weight as a run trained to the end without stopping, and no restart reported."""
    _fit(tmp_path, seed=1, steps=2)
    resumed, tracker = _fit(tmp_path, seed=1, steps=4)
    whole, _ = _fit(tmp_path / "whole", seed=1, steps=4)
    assert not _reported(tracker, DataRestarted)
    np.testing.assert_array_equal(resumed.variables["params"]["w"], whole.variables["params"]["w"])


def test_a_run_refuses_a_loader_compiled_to_resume_otherwise():
    """A loader compiled already resumes as it was compiled to (`compile(..., resume=)`), so a run that
    resumes its data otherwise is refused before it trains, and one of the same mode trains."""
    from dew.data.compile import compile

    (x,) = records(0)
    chain = Pipeline(Memory({"x": x})).shuffle(seed=1).repeat().batch(BATCH)
    for data_resume, compiled in (("relaxed", "exact"), ("exact", "relaxed")):
        trainer = Trainer(Pulled(), OptimConfig(learning_rate=PEAK), key=0, data_resume=data_resume)
        with pytest.raises(ValueError):
            trainer.fit(compile(chain, resume=compiled), steps=1)
    trainer = Trainer(Pulled(), OptimConfig(learning_rate=PEAK), key=0, data_resume="exact")
    assert int(trainer.fit(compile(chain, resume="exact"), steps=1).step) == 1


def test_a_position_read_by_other_rules_names_every_one():
    """A position whose plan read by two rules this Dew reads otherwise is refused naming both
    (`PlanChanged.differs`), which a relaxed run's restart records."""
    import json

    from dew.data.compile import compile
    from dew.data.dataset import DataPartition, Loading
    from dew.data.pipeline import mix
    from dew.data.plan import PlanChanged

    def corpus(name, tag):
        return Pipeline(Memory({"x": np.full((16, 1), tag, np.float32)}, shard_rows=4), name=name).repeat()

    chain = mix([corpus("a", 1), corpus("b", 2)], seed=4).shuffle(seed=3, window=4).batch(2)
    loader = compile(chain, read="streamed", loading=Loading(workers=0))
    stream = loader.train(DataPartition())
    next(stream)
    decoded = json.loads(stream.get_state())
    stream.close()
    decoded["dew_data_plan"]["rules"].update(mix="another-schedule", shuffle="another-key")
    resumed = loader.train(DataPartition())
    try:
        with pytest.raises(PlanChanged) as refused:
            resumed.set_state(json.dumps(decoded).encode())
    finally:
        resumed.close()
    assert {"rules.mix", "rules.shuffle"} <= set(refused.value.differs)


def _phased(seed: int):
    """Two phases of 8-row batches: corpus `a` for three steps, then a mix of `b` and `c` shuffled by
    `seed`, read on; each record's id names its corpus."""
    from dew.data.pipeline import mix, phases

    def corpus(tag: int):
        x = np.random.default_rng(tag).normal(size=(32, 4)).astype(np.float32)
        return Pipeline(Memory({"x": x, "id": np.arange(tag * 100, tag * 100 + 32, dtype=np.int32)}),
                        name=str(tag))

    later = mix([corpus(2).repeat(), corpus(3).repeat()], seed=seed)
    return phases((corpus(1).repeat().batch(8), 3), (later.batch(8), None))


def _phased_fit(directory, seed: int, steps: int):
    tracker = RecordingTracker()
    trainer = Trainer(Pulled(), OptimConfig(learning_rate=PEAK), key=0,
                      checkpoints=Checkpoints(str(directory)), tracker=tracker)
    return trainer.fit(_phased(seed), steps=steps, log_every=1, checkpoint_every=1), tracker


def test_a_phased_run_resumed_relaxed_after_a_changed_mix_opens_its_phase_at_its_start(tmp_path):
    """A phased run saved two steps into its second phase, resumed relaxed with that phase's mix seeded
    otherwise, starts its data again where its progress stands: its first phase kept as completed at the
    records it ended at, its second phase's new order opened at its own start, every boundary on the run's
    record count (the position's `origin`), and its state resumed; the position it saves reads on there."""
    import json

    from dew.data.compile import compile
    from dew.data.dataset import DataPartition

    _phased_fit(tmp_path, seed=1, steps=5)
    resumed, tracker = _phased_fit(tmp_path, seed=2, steps=6)
    (restarted,) = _reported(tracker, DataRestarted)
    assert restarted.step == 5 and int(resumed.step) == 6
    with Checkpoints(str(tmp_path)) as checkpoints:
        saved = checkpoints.position(6, DataPartition())
    held = json.loads(saved)["dew_global_position"]
    # Three steps of the first phase, two of the second before the restart, one after it.
    assert (held["records"], held["completed"][0][1], held["origin"]) == (48, 24, 16)
    fresh = compile(_phased(2)).train(DataPartition())
    try:
        ids = [np.asarray(next(fresh)["id"]) for _ in range(5)]
    finally:
        fresh.close()
    stream = compile(_phased(2)).train(DataPartition())
    try:
        stream.set_state(saved)
        # Its second phase's new order from its start: the step after the restart read its first batch.
        assert np.array_equal(np.asarray(next(stream)["id"]), ids[4])
    finally:
        stream.close()


def test_a_run_bounded_in_examples_resumed_relaxed_after_its_data_changed_ends_at_its_bound(tmp_path):
    """A run bounded in examples that counts no steps ahead (`clocked_fit`), paused halfway and resumed
    relaxed over other records, starts its data again and ends where its own count of consumed examples
    reaches its bound."""
    from test_run_budgets import CLOCKED_RECORDS, clocked_fit, clocked_records

    from dew.training import Pause

    paused, _ = clocked_fit(clocked_records(0)[0], 1, directory=tmp_path, stop=Pause(0.5))
    assert 0 < int(paused.consumed) < CLOCKED_RECORDS
    ended, tracker = clocked_fit(clocked_records(1)[0], 1, directory=tmp_path)
    assert _reported(tracker, DataRestarted)
    assert int(ended.consumed) == CLOCKED_RECORDS
