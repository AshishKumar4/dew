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

    def __init__(self, auxiliary=False, statistics='ratio'):
        self.auxiliary = auxiliary
        self.statistics = statistics

    def init(self, key, variables=None):
        return {'params': {'weight': jnp.zeros(2)}}

    def loss(self, variables, batch, step):
        error = batch['x'] @ variables['params']['weight'] - 1
        stats = Ratio(jnp.sum(error ** 2), jnp.asarray(error.size, jnp.float32))
        if self.statistics == 'scalar':
            stats = stats.total / stats.mass
        elif self.statistics == 'composite':
            stats = (stats.total, stats.mass)
        return (stats, Aux({})) if self.auxiliary else stats

    def reduce_loss(self, stats):
        if self.statistics == 'composite':
            total, mass = stats
            return total / mass, mass > 0
        return super().reduce_loss(stats)


def data(width=2):
    import grain.python as grain
    return Dataset.from_grain(grain.MapDataset.source(
        [{'x': np.ones(width, np.float32)}] * 32), batch=8)


def trainer(auxiliary=False, tracker=None, statistics='ratio'):
    return Trainer(ScalarRegression(auxiliary, statistics), optax.sgd(.1), key=0, tracker=tracker)


@pytest.mark.parametrize('statistics', ('ratio', 'scalar', 'composite'))
def test_stats_only_loss_matches_auxiliary_loss_and_direct_differentiation(statistics):
    plain, paired = trainer(statistics=statistics), trainer(auxiliary=True, statistics=statistics)
    state = plain.fit(data(), steps=2, log_every=2)
    reference = paired.fit(data(), steps=2, log_every=2)
    jax.tree.map(np.testing.assert_array_equal, state.params, reference.params)
    variables = plain.objective.init(jax.random.key(0))
    step = Step(jnp.asarray(0), jax.random.key(0), None)
    batch = {'x': jnp.ones((8, 2))}
    assert float(plain.objective.scalar_loss(variables, batch, step)[0]) == 1
    gradient = jax.grad(lambda values: plain.objective.scalar_loss(values, batch, step)[0])(variables)
    np.testing.assert_array_equal(gradient['params']['weight'], jnp.full(2, -2.))


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
    running = Trainer(ScalarRegression(auxiliary=True), optimizer, key=0, tracker=tracker)
    running.fit(data(), steps=1, log_every=1)
    recorded = next(values for _, values in tracker.scalars if 'train/grad_norm' in values)
    assert recorded['train/grad_norm'] == pytest.approx(np.sqrt(8))
    assert recorded['train/learning_rate'] == pytest.approx(.1)


def test_gradient_norm_runs_only_at_reports_and_preserves_resume_cadence(monkeypatch):
    observed = []
    tree_norm = optax.tree.norm

    def measured_norm(gradient):
        norm = tree_norm(gradient)
        jax.debug.callback(lambda value: observed.append(float(value)), norm)
        return norm

    monkeypatch.setattr(optax.tree, 'norm', measured_norm)
    tracker = RecordingTracker()
    running = trainer(auxiliary=True, tracker=tracker)
    state = running.fit(data(), steps=4, log_every=3)
    jax.effects_barrier()
    assert len(observed) == 1
    running.fit(data(), state=state, steps=8, log_every=3)
    jax.effects_barrier()
    assert len(observed) == 2
    reports = [(step, values['train/grad_norm']) for step, values in tracker.scalars
               if 'train/grad_norm' in values]
    assert [step for step, _ in reports] == [3, 6]
    assert all(norm > 0 for _, norm in reports)


def test_nonreporting_steps_do_not_even_trace_a_gradient_norm(monkeypatch):
    def unexpected_norm(gradient):
        raise AssertionError('a normal step must not trace a norm')

    monkeypatch.setattr(optax.tree, 'norm', unexpected_norm)
    state = trainer(auxiliary=True).fit(data(), steps=2, log_every=100)
    assert int(state.step) == 2
