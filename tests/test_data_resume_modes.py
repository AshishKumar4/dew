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
    assert not np.allclose(resumed.variables["params"]["w"], fresh.variables["params"]["w"])


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
