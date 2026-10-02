"""The fit boundary handles research objectives without redundant return/state plumbing."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from test_trainer import RecordingTracker

from dew.artifacts import Representations
from dew.data import Dataset
from dew.eval.common import Mean
from dew.inputs import Field, InputSpec
from dew.objectives.base import Aux, Objective, Ratio, Step
from dew.training import Trainer


class ScalarRegression(Objective):
    inputs = InputSpec(Field('x', (2,)))

    def __init__(self, auxiliary=False):
        self.auxiliary = auxiliary

    def init(self, key, variables=None):
        return {'params': {'weight': jnp.zeros(2)}}

    def loss(self, variables, batch, step):
        error = batch['x'] @ variables['params']['weight'] - 1
        stats = Ratio(jnp.sum(error ** 2), jnp.asarray(error.size, jnp.float32))
        return (stats, Aux({})) if self.auxiliary else stats


def data(width=2):
    import grain.python as grain
    return Dataset.from_grain(grain.MapDataset.source(
        [{'x': np.ones(width, np.float32)}] * 32), batch=8)


def trainer(auxiliary=False, tracker=None):
    return Trainer(ScalarRegression(auxiliary), optax.sgd(.1), key=0, tracker=tracker)


def test_stats_only_loss_matches_auxiliary_loss_and_direct_differentiation():
    plain, paired = trainer(), trainer(auxiliary=True)
    state = plain.fit(data(), steps=2, log_every=2)
    reference = paired.fit(data(), steps=2, log_every=2)
    jax.tree.map(np.testing.assert_array_equal, state.params, reference.params)
    variables = plain.objective.init(jax.random.key(0))
    step = Step(jnp.asarray(0), jax.random.key(0), None)
    assert float(plain.objective.scalar_loss(variables, {'x': jnp.ones((8, 2))}, step)[0]) == 1


def test_supplied_state_continues_instead_of_initializing_again():
    direct = trainer(auxiliary=True).fit(data(), steps=4, log_every=4)
    running = trainer(auxiliary=True)
    middle = running.fit(data(), steps=2, log_every=2)
    resumed = running.fit(data(), state=middle, steps=4, log_every=4)
    assert int(resumed.step) == 4
    jax.tree.map(np.testing.assert_array_equal, direct.params, resumed.params)


def test_first_actual_batch_is_checked_before_compilation():
    with pytest.raises(ValueError, match=r"declares shape.*first training batch"):
        trainer(auxiliary=True).fit(data(width=3), steps=1)


def test_metrics_without_a_cadence_are_refused_before_initialization():
    class Uninitialized(ScalarRegression):
        def init(self, key, variables=None):
            raise AssertionError('should not initialize')

    metric = Mean(lambda artifact, batch: jnp.ones(8), 'score', 'lower', Representations)
    with pytest.raises(ValueError, match='metrics need eval_every'):
        Trainer(Uninitialized(), optax.sgd(.1), key=0).fit(data(), steps=1, metrics=(metric,))


def test_gradient_norm_and_injected_learning_rate_are_logged():
    tracker = RecordingTracker()
    optimizer = optax.inject_hyperparams(optax.sgd)(learning_rate=.1)
    Trainer(ScalarRegression(auxiliary=True), optimizer, key=0, tracker=tracker).fit(data(), steps=1, log_every=1)
    recorded = next(values for _, values in tracker.scalars if 'train/grad_norm' in values)
    assert recorded['train/grad_norm'] == pytest.approx(np.sqrt(8))
    assert recorded['train/learning_rate'] == pytest.approx(.1)
