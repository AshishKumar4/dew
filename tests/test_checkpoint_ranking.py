"""Evaluation ranks precisely the weights saved, with independent best trackers.

Restoring the ranked training state is Dew's lifecycle; Orbax oracles storage and selection.
"""
import dataclasses
import datetime
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from affine_run import Counting, Data, val_batches

from dew.artifacts import TokenScores
from dew.checkpoints import Checkpoints, Keep, Ranking
from dew.objectives.base import Aux, Objective, Shown
from dew.training import Trainer
from dew.training.trainer import Best, Plateau


class Overfit(Objective):
    def init(self, key, variables=None):
        return {'params': {'w': jnp.zeros(())}}

    def loss(self, variables, batch, step):
        weight = variables['params']['w']
        # Train to 1; validation prefers the halfway weights.
        target = jnp.where(batch['validation'][0], 0.5, 1.0)
        return (weight - target) ** 2, Aux({})

    def evaluate(self, params, batch, step):
        weight = params['params']['w']
        return TokenScores(jnp.broadcast_to(weight, (4, 1)), jnp.ones((4, 1)),
                           correct=jnp.zeros((4, 1), dtype=bool))


@dataclasses.dataclass
class Value:
    name: str
    shown: Shown
    reads = TokenScores

    def __call__(self, artifact, batch):
        return float(np.mean(artifact.losses))

    def merge(self, left, right):
        return (left + right) / 2

    def finalize(self, value):
        return value


def data():
    class Train(Counting):
        def __next__(self):
            batch = super().__next__()
            return {**batch, 'validation': np.zeros(len(batch['x']), bool)}
    def train():
        return Train()
    def validation():
        for batch in val_batches(1)():
            yield {**batch, 'validation': np.ones(len(batch['x']), bool)}
    return Data(train=train, val=validation)


def trainer(path, keep=1):
    return Trainer(
        Overfit(), optax.sgd(0.1), key=jax.random.key(0), checkpoints=Checkpoints(str(path), keep=keep)
    )


def test_default_retains_the_weights_with_lowest_validation_loss(tmp_path):
    run = trainer(tmp_path / 'run')
    result = run.fit(data(), steps=12, log_every=1, eval_every=1, checkpoint_every=6)
    run.checkpoints.wait()
    assert int(result.step) == 12
    assert run.checkpoints.best == 3
    assert run.checkpoints.latest == 12
    scores = {entry.step: entry.metrics for entry in run.checkpoints.kept()}
    assert set(scores) == {3, 12}
    assert scores[3]['val/loss'] < scores[12]['val/loss']
    assert scores[3]['train/loss'] > scores[12]['train/loss']
    restored, _ = run.checkpoints.restore({'variables': result.variables}, step='best')
    assert float(restored['variables']['params']['w']) == pytest.approx(0.488)


def test_multiple_directions_and_threshold(tmp_path):
    metric = Value('accuracy', Shown(better='higher'))
    run = trainer(tmp_path / 'run')
    run.fit(data(), steps=6, log_every=1, eval_every=1, checkpoint_every=6,
            metrics=[metric], best=[Best(metric, top=2), Best(lambda m: abs(m[metric] - .5), threshold=.1)])
    run.checkpoints.wait()
    assert run.checkpoints.resolve('best:val/accuracy') == 6
    assert run.checkpoints.resolve('best:aggregate:1') == 3
    assert {entry.step for entry in run.checkpoints.kept()} == {3, 5, 6}


def test_a_terminal_winner_is_saved_without_the_unconditional_final_policy(tmp_path):
    metric = Value('value', Shown(better='higher'))
    run = trainer(tmp_path / 'run')
    state = run.fit(data(), steps=2, log_every=1, eval_every=1, metrics=[metric],
                    best=metric, checkpoint_final=False, restore_best=True)
    assert run.checkpoints.best == 2
    assert int(state.step) == 2
    assert float(state.variables['params']['w']) == pytest.approx(.36)


def test_missing_and_undeclared_metrics_are_refused_before_training(tmp_path):
    metric = Value('accuracy', Shown())
    with pytest.raises(ValueError):
        trainer(tmp_path / 'a').fit(data(), steps=2, best=metric)
    with pytest.raises(ValueError):
        trainer(tmp_path / 'b').fit(data(), steps=2, metrics=[metric], best=metric)


def test_ranked_loss_and_periodic_retention(tmp_path):
    run = trainer(tmp_path / 'run', keep=Keep(latest=1, every=2))
    state = run.fit(Data(train=data()._train), steps=1, log_every=1)
    checkpoints = run.checkpoints
    for step, loss in [(2, .1), (3, .9), (4, .7), (5, .8)]:
        checkpoints.save(step, state.replace(step=jnp.int32(step)), None, ranking=Ranking('train/loss', loss))
        checkpoints.wait()
    assert checkpoints.best == 2
    assert {entry.step for entry in checkpoints.kept()} == {2, 4, 5}


def test_plateau_restores_patience_and_stops_after_eligible_evaluations(tmp_path):
    metric = Value('value', Shown(better='lower'))
    first = trainer(tmp_path / 'run')
    first.fit(data(), steps=2, log_every=1, eval_every=1, checkpoint_every=1,
              metrics=[metric], stop=Plateau(metric, evals=3))
    second = trainer(tmp_path / 'run')
    state = second.fit(data(), steps=10, log_every=1, eval_every=1, checkpoint_every=1,
                       metrics=[metric], stop=Plateau(metric, evals=3))
    assert int(state.step) == 4
    assert second.checkpoints.latest == 4
    assert second.checkpoints.control(4)['plateau:val/value']['bad'] == 3


@pytest.mark.parametrize("checkpoint_final", [True, False])
def test_terminal_local_resume_keeps_the_final_plateau_evaluation(tmp_path, checkpoint_final):
    metric = Value('value', Shown(better='lower'))

    def local():
        run = trainer(tmp_path / 'run')
        run.checkpoints = Checkpoints(str(tmp_path / 'run'), local_directory=str(tmp_path / 'local'),
                                      local_every=2)
        return run

    first = local()
    first.fit(data(), steps=2, log_every=1, eval_every=1, checkpoint_every=1,
              checkpoint_final=checkpoint_final, metrics=[metric], stop=Plateau(metric, evals=3))
    assert first.checkpoints.control(2)['plateau:val/value']['bad'] == 1
    second = local()
    state = second.fit(data(), steps=10, log_every=1, eval_every=1, checkpoint_every=1,
                       checkpoint_final=checkpoint_final, metrics=[metric], stop=Plateau(metric, evals=3))
    assert int(state.step) == 4


def test_multi_split_metric_keys_and_ambiguous_selection(tmp_path):
    metric = Value('value', Shown(better='higher'))
    readers = {'first': data().val, 'second': data().val}
    with pytest.raises(ValueError):
        trainer(tmp_path / 'bad').fit(data(), steps=2, metrics=[metric], best=metric, validation=readers)
    run = trainer(tmp_path / 'run')
    run.fit(data(), steps=4, log_every=1, eval_every=1, checkpoint_every=4, metrics=[metric],
            validation=readers, best=Best(lambda m: abs(m['first', metric] - .5), top=1))
    assert run.checkpoints.best == 3
    record = run.checkpoints.kept()[0]
    assert {'first/value', 'second/value'} <= set(record.metrics)


def test_weights_only_best_is_smaller_and_not_a_resume_state(tmp_path):
    metric = Value('value', Shown(better='lower'))
    run = trainer(tmp_path / 'run')
    run.fit(data(), steps=4, log_every=1, eval_every=1, checkpoint_every=4,
            metrics=[metric], best=Best(metric, weights_only=True))
    assert run.checkpoints.best == 1
    assert run.checkpoints.latest == 4
    assert set(run.checkpoints.stored('best')) == {'variables', 'ema'}
    saved, _ = run.checkpoints.restore(None, 'best')
    assert set(saved) == {'variables', 'ema'}
    with pytest.raises(ValueError):
        run.checkpoints.restore(run.initial_state(), 'best')
    with pytest.raises(ValueError):
        trainer(tmp_path / 'bad').fit(data(), steps=2, metrics=[metric],
                                    best=Best(metric, weights_only=True), restore_best=True)


def test_restore_best_returns_the_whole_earlier_state(tmp_path):
    run = trainer(tmp_path / 'run')
    state = run.fit(data(), steps=8, log_every=1, eval_every=1, checkpoint_every=4, restore_best=True)
    assert int(state.step) == 3
    assert int(state.updates) == 3
    assert float(state.variables['params']['w']) == pytest.approx(.488)


@pytest.mark.parametrize("aggregate", [False, True])
def test_training_values_are_selected_through_the_objective(tmp_path, aggregate):
    run = trainer(tmp_path / 'run')
    stream = Data(train=data()._train) if aggregate else data()
    best = Best(lambda metrics: metrics[run.objective.loss]) if aggregate else run.objective.loss
    run.fit(stream, steps=4, log_every=1, checkpoint_every=4 if aggregate else 1, best=best)
    assert run.checkpoints.best == 4
    if not aggregate:
        assert run.checkpoints.kept()[0].ranked_by == 'train/loss'


def test_time_cadence_and_recorded_duration(tmp_path):
    from dew.config import TrainerConfig
    from dew.data import Dataset

    assert TrainerConfig(checkpoint_every="30m").checkpoint_interval(
        Dataset(lambda p: iter(()), None, records=4, batch=2)
    ) == datetime.timedelta(minutes=30)
    with pytest.raises(ValueError):
        TrainerConfig(checkpoint_every="0m").checkpoint_interval(
            Dataset(lambda p: iter(()), None, records=4, batch=2)
        )
    with pytest.raises(TypeError):
        TrainerConfig(best=Best(lambda m: 0))
    run = trainer(tmp_path / 'run', keep=Keep(latest=5))
    run.fit(data(), steps=3, log_every=1, checkpoint_every=datetime.timedelta(microseconds=1))
    assert [entry.step for entry in run.checkpoints.kept()] == [1, 2, 3]


def test_keep_predicate_and_interval_union_with_latest(tmp_path):
    run = trainer(tmp_path / 'run', keep=Keep(latest=1, interval=datetime.timedelta(days=1),
                                            where=lambda c: c.metrics.get('special', 0) > 0))
    state = run.fit(data(), steps=1, log_every=1)
    for step in range(2, 5):
        run.checkpoints.save(step, state.replace(step=jnp.int32(step)), None,
                             metrics={'special': float(step == 2)}, ranking=Ranking('score', step))
        run.checkpoints.wait()
    assert {entry.step for entry in run.checkpoints.kept()} == {1, 2, 4}


def test_named_config_selector_roundtrips_and_refuses_callable():
    from dew.config import RunConfig, TrainerConfig
    config = RunConfig(trainer=TrainerConfig(best=Best('accuracy', top=2), checkpoint_every='15m'))
    assert RunConfig.from_dict(config.to_dict()) == config
    best = config.trainer.best_policies()[0]
    assert best.metric == 'accuracy' and best.top == 2
    with pytest.raises(TypeError):
        TrainerConfig(best=lambda m: 1.)


def test_validation_loss_reduces_additive_statistics_over_uneven_batches():
    from dew.objectives.base import Ratio
    from dew.training import Evaluation

    class Weighted(Overfit):
        def loss(self, variables, batch, step):
            values = (variables['params']['w'] - batch['target']) ** 2
            return Ratio(jnp.sum(values), jnp.asarray(values.size, jnp.float32)), Aux({})

    objective = Weighted()
    variables = objective.init(jax.random.key(0))
    batches = [{'target': np.zeros(16, np.float32)}, {'target': np.ones(8, np.float32)}]
    result = Evaluation.run(objective, variables, lambda partition: iter(batches), key=jax.random.key(0),
                            loss=True)
    assert result.scores['val/loss'] == pytest.approx(1 / 3)


def test_validation_loss_uses_exactly_the_ema_weights_of_the_evaluated_state():
    from dew.training import Evaluation
    objective = Overfit()
    live = {'params': {'w': jnp.asarray(.9)}}
    averaged = {'params': {'w': jnp.asarray(.5)}}
    result = Evaluation.run(objective, live, data().val, key=jax.random.key(0), averaged=averaged, step=7,
                            loss=True)
    assert result.step == 7
    assert result.scores['val/loss'] == 0.
    direct = Evaluation.run(objective, averaged, data().val, key=jax.random.key(0), step=7, loss=True)
    assert direct.scores == result.scores


def test_record_metrics_and_keep_predicate_accept_unhashable_metric_objects(tmp_path):
    metric = Value('value', Shown(better='lower'))
    run = trainer(tmp_path / 'run', keep=Keep(latest=1, where=lambda c: c.metrics[metric] < .4))
    run.fit(data(), steps=6, log_every=1, eval_every=1, checkpoint_every=1,
            metrics=[metric], best=Best(metric, mode='max'))
    retained = run.checkpoints.kept()
    assert {entry.step for entry in retained} == {1, 2, 6}
    for entry in retained:
        assert entry.metrics[metric] == entry.metrics['val/value']
    class WithCe(Overfit):
        shown: ClassVar = {'ce': Shown(better='lower')}
    objective = WithCe()
    assert 'ce' in dir(objective.scalars)
    assert objective.scalars.ce.owner is objective
    with pytest.raises(AttributeError):
        _ = objective.scalars.unknown_report


def test_local_resume_keeps_plateau_patience(tmp_path):
    metric = Value('value', Shown(better='lower'))
    plain = trainer(tmp_path / 'plain')
    state = plain.fit(data(), steps=2, log_every=1)
    path, local = str(tmp_path / 'run'), str(tmp_path / 'local')
    control = {'plateau:val/value': {'best': .2, 'bad': 1, 'step': 2,
                                    'rule': {'mode': 'min', 'evals': 3, 'min_delta': 0.}}}
    checkpoints = Checkpoints(path, local_directory=local, local_every=1)
    checkpoints.save_local(2, state, None, control=control)
    checkpoints.wait()
    resumed = Trainer(Overfit(), optax.sgd(.1), key=jax.random.key(0),
                       checkpoints=Checkpoints(path, local_directory=local, local_every=1))
    assert resumed.checkpoints.control(2) == control
    result = resumed.fit(data(), steps=10, log_every=1, eval_every=1, checkpoint_every=1,
                          metrics=[metric], stop=Plateau(metric, evals=3))
    assert int(result.step) == 4



def test_recorded_retention_and_cadence_roundtrip_without_losing_microseconds():
    from dew.config import RunConfig, TrainerConfig
    from dew.records import duration, recorded_duration
    spacing = datetime.timedelta(days=999999999, microseconds=1)
    assert duration(recorded_duration(spacing)) == spacing
    config = RunConfig(trainer=TrainerConfig(keep=Keep(latest=2, every=100, interval='1h'),
                                             checkpoint_every=datetime.timedelta(minutes=30),
                                             best=Best('fid', top=3)))
    assert config.to_dict()['trainer']['checkpoint_every'] == '30m'
    assert RunConfig.from_dict(config.to_dict()) == config
    with pytest.raises(TypeError):
        TrainerConfig(keep=Keep(where=lambda c: True))
