"""The diagnostic must exercise the real Trainer's unchanged profiler window."""
import jax
import numpy as np
import optax

from affine_run import Regression, batches, dataset
from profiler_sequence_support import ReplayTracker, TrainerProbe

from dew.training import MeshSpec, Trainer
from dew.training.profiling import ProfileWindow


def test_the_real_loop_captures_steps_41_to_48_and_all_suites_at_43(tmp_path):
    events = []

    def event(phase, **fields):
        events.append({"phase": phase, **fields})

    class ProbedTrainer(TrainerProbe, Trainer):
        pass

    trainer = ProbedTrainer(
        Regression(), optax.sgd(1e-3), key=57831, accumulation=2, split=2,
        mesh=MeshSpec().build([jax.devices("cpu")[0]]),
        profile=ProfileWindow(str(tmp_path / "trace"), 8, warmup=40, reports=False),
        tracker=ReplayTracker(event))
    trainer.replay_event = event
    batch = next(batches())
    validation = {f"suite{i:03d}": lambda partition: iter((batch,)) for i in range(101)}
    result = trainer.fit(dataset(), steps=48, eval_every=43, log_every=10,
                         validation=validation, checkpoint_final=False)
    assert int(result.step) == 48
    captured_steps = [entry["step"] for entry in events
                      if entry["phase"] == "between_steps" and entry["capture"]]
    assert captured_steps == list(range(41, 49))
    evaluations = [entry for entry in events if entry["phase"] == "evaluation"]
    at_43 = [entry for entry in evaluations if entry["step"] == 43]
    at_48 = [entry for entry in evaluations if entry["step"] == 48]
    assert len(at_43) == len(at_48) == 101
    assert all(entry["capture"] for entry in at_43)
    assert all(not entry["capture"] for entry in at_48)
    assert all(np.isfinite(list(entry["scores"].values())).all() for entry in evaluations)
