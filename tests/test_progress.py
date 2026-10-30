"""The training step's position, a declared input a module whose computation changes over a run reads
(`dew.nn.progress`, ledgers 2455, 2479 and 2537): a run's training loss passes it to a model that reads
it; a call that passes none, a validation's among them, reads the run's end; a compiled forward keys on
it as on any argument; an objective that passes none refuses a model that reads it."""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from affine_run import Replaying
from recording import RecordingTracker

from dew.data.dataset import Loader
from dew.inputs import Field, InputSpec
from dew.nn.progress import END, Progress
from dew.objectives.base import Aux
from dew.objectives.supervised import Supervised
from dew.training import Trainer


class Annealed(nn.Module):
    """A map whose output is offset by the updates its run committed before the step, or by -1 past
    the run's training: an anneal's schedule, as a module reads it from its call's `progress`."""

    @nn.nowrap
    def reads_progress(self) -> bool:
        return True

    @nn.compact
    def __call__(self, x, progress: Progress = END):
        weight = self.param("weight", nn.initializers.zeros, (x.shape[-1], 1))
        offset = -1.0 if progress.updates is None else progress.updates.astype(jnp.float32)
        return x @ weight + offset


def objective() -> Supervised:
    return Supervised(Annealed(), lambda outputs, batch: jnp.mean(outputs, axis=-1),
                      inputs=InputSpec(Field("x", (2,))))


def data(val: bool = False) -> Loader:
    zeros = {"x": np.zeros((8, 2), np.float32)}
    return Loader(train=lambda partition: Replaying(lambda step: zeros),
                  val=(lambda partition: iter((zeros,))) if val else None, batch=8, records=None)


def test_a_run_s_loss_reads_the_updates_committed_before_each_step_and_its_end_outside_it():
    """A module reading the step's progress in a run reads the updates the run committed before each
    step: its loss, the offset alone on zero inputs, is 0, 1, 2 over three steps; applied outside a
    run, as an evaluation or a server applies it, it reads the run's end."""
    tracker = RecordingTracker()
    trained = Trainer(objective(), optax.sgd(0.0), key=0, tracker=tracker)
    state = trained.fit(data(), steps=3, log_every=1)
    losses = [scalars["train/loss"] for _, scalars in tracker.scalars if "train/loss" in scalars]
    np.testing.assert_array_equal(losses, [0.0, 1.0, 2.0])
    applied = Annealed().apply(jax.device_get(state.variables), jnp.zeros((1, 2)))
    np.testing.assert_array_equal(applied, [[-1.0]])


def test_a_run_s_validation_scores_the_model_at_the_run_s_end():
    """A validation within a run scores the model the run would hand on, its schedules ended, as an
    evaluation after it does: the offset alone on zero inputs, -1, at each pass, while the training loss
    reads the updates committed before each step."""
    tracker = RecordingTracker()
    Trainer(objective(), optax.sgd(0.0), key=0, tracker=tracker).fit(data(val=True), steps=4, log_every=1,
                                                                    eval_every=2)
    validated = [scalars["val/loss"] for _, scalars in tracker.scalars if "val/loss" in scalars]
    np.testing.assert_array_equal(validated, [-1.0, -1.0])
    losses = [scalars["train/loss"] for _, scalars in tracker.scalars if "train/loss" in scalars]
    np.testing.assert_array_equal(losses, [0.0, 1.0, 2.0, 3.0])


def test_a_compiled_forward_keys_on_the_progress_it_is_passed():
    """The progress is the call's input, so one forward compiled once reads each value it is passed and
    the run's end where it is passed none, in whichever order: no value it first traced with is held,
    as a value read from ambient state outside the compile's key was (ledger 2537)."""
    variables = Annealed().init(jax.random.key(0), jnp.zeros((1, 2)))
    forward = jax.jit(lambda variables, x, progress=END: Annealed().apply(variables, x, progress))
    x = jnp.zeros((1, 2))
    np.testing.assert_array_equal(forward(variables, x), [[-1.0]])
    np.testing.assert_array_equal(forward(variables, x, Progress(jnp.asarray(3, jnp.int32))), [[3.0]])
    np.testing.assert_array_equal(forward(variables, x, Progress(jnp.asarray(5, jnp.int32))), [[5.0]])
    np.testing.assert_array_equal(forward(variables, x), [[-1.0]])


def test_an_objective_that_passes_no_progress_refuses_a_model_that_reads_it():
    """An objective whose loss passes its model no progress (`Objective.passes_progress`) refuses a model
    that reads it, which would otherwise train at the run's end; a custom training step passes it as any
    other input, and the model reads what the step passes."""
    class Unstepped(Supervised):
        passes_progress = False

    with pytest.raises(ValueError):
        Trainer(Unstepped(Annealed(), lambda outputs, batch: jnp.mean(outputs, axis=-1),
                          inputs=InputSpec(Field("x", (2,)))), optax.sgd(0.0), key=0).fit(data(), steps=1)

    def make_step(objective, optimizer):
        def step(state, batch):
            outputs = Annealed().apply(state.variables, batch["x"], Progress(state.updates))
            return (state.replace(microstep=state.microstep + 1, updates=state.updates + 1),
                    jnp.mean(outputs), Aux({}))
        return step

    tracker = RecordingTracker()
    Trainer(objective(), optax.sgd(0.0), key=0, step=make_step, tracker=tracker).fit(data(), steps=2,
                                                                                    log_every=1)
    losses = [scalars["train/loss"] for _, scalars in tracker.scalars if "train/loss" in scalars]
    np.testing.assert_array_equal(losses, [0.0, 1.0])
