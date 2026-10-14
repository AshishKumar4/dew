"""The fit boundary handles research objectives without redundant return/state plumbing."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.data import Memory
from dew.data.compile import compile
from dew.data.pipeline import Pipeline
from dew.inputs import Field, InputSpec
from dew.objectives.base import Aux, Objective, Ratio, Step
from dew.training import Checkpoints, Trainer


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
    return compile(Pipeline(Memory(grain.MapDataset.source(
        [{'x': np.ones(width, np.float32)}] * 32))).repeat().batch(8), read="indexed")


def trainer(auxiliary=False, tracker=None, statistics='ratio', **settings):
    return Trainer(ScalarRegression(auxiliary, statistics), optax.sgd(.1), key=0, tracker=tracker, **settings)


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


def test_a_supplied_state_is_the_runs_initial_state_and_its_checkpoint_the_continuation(tmp_path):
    """A run handed another run's state starts from it rather than a fresh
    initialization: on a stream that reads the same batch throughout, two steps
    from a two-step state equal four from scratch. A checkpoint is the one
    continuation: beside a checkpoint at step 2, a run handed the initial state
    trains from step 0, and the same run handed none resumes at step 2."""
    direct = trainer(auxiliary=True).fit(data(), steps=4, log_every=4)
    running = trainer(auxiliary=True)
    middle = running.fit(data(), steps=2, log_every=2)
    resumed = running.fit(data(), state=middle, steps=4, log_every=4)
    assert int(resumed.step) == 4
    jax.tree.map(np.testing.assert_array_equal, direct.variables, resumed.variables)

    def checkpointed():
        return trainer(auxiliary=True, checkpoints=Checkpoints(str(tmp_path / "run")))

    checkpointed().fit(data(), steps=2, log_every=2)
    beside = checkpointed()
    restarted = beside.fit(data(), state=beside.initial_state(), steps=1, log_every=1, checkpoint_final=False)
    assert int(restarted.step) == 1
    once = trainer(auxiliary=True).fit(data(), steps=1, log_every=1)
    jax.tree.map(np.testing.assert_array_equal, once.variables, restarted.variables)
    continued = checkpointed().fit(data(), steps=4, log_every=4)
    assert int(continued.step) == 4
    jax.tree.map(np.testing.assert_array_equal, direct.variables, continued.variables)
