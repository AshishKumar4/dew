"""The training step's position, which a module whose computation changes over a run reads
(`dew.nn.progress`, ledger 2455): a run's loss supplies it, in training and validation; outside a run
it is the run's end; a training step that supplies none refuses a module that reads it."""

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
from dew.nn.progress import progress
from dew.objectives.base import Aux
from dew.objectives.supervised import Supervised
from dew.training import Trainer


class Annealed(nn.Module):
    """A map whose output is offset by the updates its run committed before the step, or by -1 past
    the run's training: an anneal's schedule, as a module reads it."""

    @nn.compact
    def __call__(self, x):
        weight = self.param("weight", nn.initializers.zeros, (x.shape[-1], 1))
        updates = progress().updates
        offset = -1.0 if updates is None else updates.astype(jnp.float32)
        return x @ weight + offset


def objective() -> Supervised:
    return Supervised(Annealed(), lambda outputs, batch: jnp.mean(outputs, axis=-1),
                      inputs=InputSpec(Field("x", (2,))))


def data() -> Loader:
    return Loader(train=lambda partition: Replaying(lambda step: {"x": np.zeros((8, 2), np.float32)}),
                  val=None, batch=8, records=None)


def test_a_run_s_loss_reads_the_updates_committed_before_each_step_and_its_end_outside_it():
    """A module reading the step's progress in a run reads the updates the run committed before each
    step: its loss, the offset alone on zero inputs, is 0, 1, 2 over three steps; applied outside a
    run, as an evaluation or a server applies it, it reads the run's end."""
    tracker = RecordingTracker()
    trained = Trainer(objective(), optax.sgd(0.0), key=0, tracker=tracker)
    state = trained.fit(data(), steps=3, log_every=1)
    losses = [scalars["train/loss"] for _, scalars in tracker.scalars if "train/loss" in scalars]
    np.testing.assert_allclose(losses, [0.0, 1.0, 2.0])
    applied = Annealed().apply(jax.device_get(state.variables), jnp.zeros((1, 2)))
    np.testing.assert_array_equal(applied, [[-1.0]])


def test_a_custom_training_step_refuses_a_module_that_reads_the_progress_it_does_not_supply():
    """A training step the trainer traces that does not run an objective's loss (a custom step) supplies
    no progress, and a module that reads it there is refused, not handed a stand-in; the same step over
    a module that reads none trains."""
    def stepping(model):
        def make_step(objective, optimizer):
            def step(state, batch):
                loss = jnp.mean(model.apply(state.variables, batch["x"]))
                return state.replace(microstep=state.microstep + 1, updates=state.updates + 1), loss, Aux({})
            return step
        return make_step

    with pytest.raises(ValueError):
        Trainer(objective(), optax.sgd(0.0), key=0, step=stepping(Annealed())).fit(data(), steps=1)
    plain = Supervised(nn.Dense(1), lambda outputs, batch: jnp.mean(outputs, axis=-1),
                       inputs=InputSpec(Field("x", (2,))))
    trained = Trainer(plain, optax.sgd(0.0), key=0, step=stepping(nn.Dense(1))).fit(data(), steps=1)
    assert int(trained.step) == 1
