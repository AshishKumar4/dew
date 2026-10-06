"""The fit boundary handles research objectives without redundant return/state plumbing."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from recording import RecordingTracker

from dew.artifacts import Representations
from dew.data import DataPartition, Dataset
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
    jax.tree.map(np.testing.assert_array_equal, state.variables, reference.variables)
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
    jax.tree.map(np.testing.assert_array_equal, direct.variables, resumed.variables)


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


def test_injected_learning_rate_is_logged():
    tracker = RecordingTracker()
    optimizer = optax.inject_hyperparams(optax.sgd)(learning_rate=.1)
    running = Trainer(ScalarRegression(auxiliary=True), optimizer, key=0, tracker=tracker)
    running.fit(data(), steps=1, log_every=1)
    recorded = next(values for _, values in tracker.scalars if 'train/learning_rate' in values)
    assert recorded['train/learning_rate'] == pytest.approx(.1)


def test_the_documented_norm_recorder_holds_the_gradient_norm_and_shares_the_clips_reduction():
    """docs/key-concepts.md's `record_norm` keeps the step's gradient norm in
    the optimizer state, and beside `clip_by_global_norm` the compiled step
    reduces no more than the clip alone does: the two norms are one."""
    import re

    from test_run_artifact import documented_block

    scope: dict = {}
    exec(documented_block("docs/key-concepts.md", "To track it, chain a transformation"), scope)
    record_norm = scope["record_norm"]
    # A first step from zero weights over rows of ones has gradient [-2, -2].
    recorded = Trainer(ScalarRegression(), optax.chain(record_norm, optax.sgd(.1)), key=0).fit(
        data(), steps=1)
    assert float(recorded.opt_state[0]) == pytest.approx(np.sqrt(8))

    def reductions(optimizer):
        running = Trainer(ScalarRegression(), optimizer, key=0)
        state, _, _ = running.place()
        batch = next(iter(data().train(DataPartition())))
        running.compile(state, batch)
        return len(re.findall(r"= \S+ reduce\(", running.executable.as_text()))

    clip = optax.clip_by_global_norm(1.0)
    shared = reductions(optax.chain(record_norm, clip, optax.sgd(.1)))
    assert shared == reductions(optax.chain(clip, optax.sgd(.1)))
