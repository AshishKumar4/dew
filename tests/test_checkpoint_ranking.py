"""Evaluation ranks precisely the weights saved, with independent best trackers.

Restoring the ranked training state is Dew's lifecycle; Orbax oracles storage and selection.
"""
import dataclasses
import datetime
import itertools
from collections.abc import Iterator
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from affine_run import Counting, dataset, val_batches

from dew.artifacts import TokenScores
from dew.checkpoints import Checkpoints, Keep, Metrics, Ranking
from dew.data.dataset import DataPartition
from dew.objectives.base import Aux, Objective, Shown
from dew.telemetry.records import FitEnded
from dew.training import Trainer
from dew.training.evaluation import MetricScores
from dew.training.trainer import Best, Plateau


class Overfit(Objective):
    def fresh_variables(self, key, held):
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


@dataclasses.dataclass
class Scripted:
    """A lower-is-better metric that scores each evaluation it finalizes `scores`' next value, whatever
    it reads: a warming pass, which finalizes nothing, takes none."""
    name: str
    scores: Iterator[float]
    shown: Shown = dataclasses.field(default_factory=lambda: Shown(better='lower'))
    reads = TokenScores

    def __call__(self, artifact, batch):
        return 0.0

    def merge(self, left, right):
        return left

    def finalize(self, value):
        return next(self.scores)


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
    return dataset(train=train, val=validation)


def trainer(path, keep=1):
    return Trainer(
        Overfit(), optax.sgd(0.1), key=jax.random.key(0), checkpoints=Checkpoints(str(path), keep=keep)
    )


def test_a_score_object_names_one_recorded_score_live_and_in_a_checkpoint():
    """`best=` and `stop=` read a run's live scores, and a predicate a checkpoint's kept ones, by one
    rule: a metric is its validation split's, `(split, metric)` that split's, a training scalar the
    training split's and a recorded name itself. A metric with no validation score, recorded under
    another split alone, is refused in both until its split is named, never resolved by guessing
    which split was meant."""
    objective = Overfit()
    metric, other = Value('value', Shown(better='lower')), Value('other', Shown(better='lower'))
    scores = {'val/value': 1.0, 'test/value': 2.0, 'test/other': 3.0, 'train/loss': 4.0}
    live, kept = MetricScores({'val': [metric], 'test': [metric, other]}, scores, objective), Metrics(scores)
    for read in (live, kept):
        assert read[metric] == 1.0 and read[('test', metric)] == 2.0 and read[('test', other)] == 3.0
        assert read['val/value'] == 1.0 and read['test/other'] == 3.0
        assert read[objective.scalars.loss] == 4.0
        with pytest.raises(KeyError):
            read[other]
    with pytest.raises(KeyError):
        live[Overfit().scalars.loss]


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


def test_an_aggregate_score_that_fails_fails_its_run(tmp_path):
    """An aggregate score whose own code fails, here by reading a score no split records, fails
    the run that asked for it, where it once ranked nothing and the run went on without a best."""
    metric = Value('accuracy', Shown(better='higher'))
    run = trainer(tmp_path / 'run')
    with pytest.raises(KeyError):
        run.fit(data(), steps=2, log_every=1, eval_every=1, checkpoint_every=2, metrics=[metric],
                best=Best(lambda scores: scores['val/acuracy']))


def test_an_aggregate_is_asked_only_where_an_evaluation_scored(tmp_path):
    """A run that evaluates every other step asks an aggregate, here one reading a score by its
    recorded name, only at the steps an evaluation scored, so it always reads a score, and it ranks
    the checkpoints taken there."""
    metric = Value('accuracy', Shown(better='higher'))
    seen = []

    def accuracy(scores):
        seen.append(scores['val/accuracy'])
        return seen[-1]

    run = trainer(tmp_path / 'run')
    run.fit(data(), steps=4, log_every=1, eval_every=2, checkpoint_every=2, metrics=[metric],
            best=Best(accuracy, mode='max'))
    run.checkpoints.wait()
    assert seen and np.isfinite(seen).all()
    assert run.checkpoints.resolve('best:aggregate:0') == 4


def test_an_aggregate_reads_a_split_on_its_own_cadence_as_nan_where_it_was_not_scored(tmp_path):
    """A split evaluated on a cadence of its own is a score the fit declares that the evaluations
    between its own do not record: an aggregate reading it by `(split, metric)` reads NaN there,
    which ranks nothing, and ranks the steps it was scored at. The value rises as training does,
    so the lowest is each split's first scored step."""
    from dew.training import EvalSuite

    metric = Value('accuracy', Shown(better='higher'))
    run = trainer(tmp_path / 'run')
    run.fit(data(), steps=4, log_every=1, eval_every=1, metrics=[metric],
            validation={'often': data().val, 'rare': EvalSuite(data().val, (metric,), 2)},
            best=[Best(lambda scores: scores[('often', metric)], mode='min'),
                  Best(lambda scores: scores[('rare', metric)], mode='min')])
    run.checkpoints.wait()
    assert run.checkpoints.resolve('best:aggregate:0') == 1
    assert run.checkpoints.resolve('best:aggregate:1') == 2


def test_an_aggregate_reads_a_training_scalar_whose_name_has_parts(tmp_path):
    """A training scalar may name itself in parts, as a critic's `critic/loss`: an aggregate reads
    it by the scalar the objective declares, under its whole name, and ranks by it."""
    class Critic(Overfit):
        shown: ClassVar = {'critic/loss': Shown(better='lower')}

        def loss(self, variables, batch, step):
            statistic, _ = super().loss(variables, batch, step)
            return statistic, Aux({'critic/loss': statistic})

    objective = Critic()
    run = Trainer(objective, optax.sgd(0.1), key=jax.random.key(0),
                  checkpoints=Checkpoints(str(tmp_path / 'run'), keep=1))
    run.fit(dataclasses.replace(data(), val=None), steps=4, log_every=1, checkpoint_every=2,
            best=Best(lambda scores: scores[objective.scalars['critic/loss']]))
    run.checkpoints.wait()
    assert run.checkpoints.resolve('best:aggregate:0') == 4


@pytest.mark.parametrize('by_name', [False, True], ids=['object', 'name'])
def test_multiple_directions_and_threshold(tmp_path, by_name):
    """An aggregate score reads a metric by its object or by the name its score is recorded under."""
    metric = Value('accuracy', Shown(better='higher'))
    key = 'val/accuracy' if by_name else metric
    run = trainer(tmp_path / 'run')
    run.fit(data(), steps=6, log_every=1, eval_every=1, checkpoint_every=6,
            metrics=[metric], best=[Best(metric, top=2), Best(lambda m: abs(m[key] - .5), threshold=.1)])
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


@pytest.mark.parametrize('spelling', ['object', 'name', 'recorded'])
def test_each_suites_best_policy_ranks_that_suites_instance_of_a_metric_two_suites_hold(tmp_path, spelling):
    """Two suites may each score their own instance of one metric, each declaring its own direction: a
    policy for each suite, by the instance, by its name and split, or by its recorded name, ranks that
    suite's scores in that suite's instance's direction, though the other's has the same name and the
    opposite one. The value rises as training does, so the best of a higher-is-better value is the
    last step and of a lower-is-better one the first."""
    from dew.training import EvalSuite

    often, rare = Value('value', Shown(better='higher')), Value('value', Shown(better='lower'))
    best = {'object': [Best(often, split='often'), Best(rare, split='rare')],
            'name': [Best('value', split='often'), Best('value', split='rare')],
            'recorded': ['often/value', 'rare/value']}[spelling]
    run = trainer(tmp_path / 'run')
    run.fit(data(), steps=3, log_every=1, eval_every=1, metrics=[often], best=best,
            validation={'often': data().val, 'rare': EvalSuite(data().val, (rare,), 1)})
    run.checkpoints.wait()
    assert run.checkpoints.resolve('best:often/value') == 3
    assert run.checkpoints.resolve('best:rare/value') == 1


def leaf_values(leaf) -> np.ndarray:
    """A state leaf's values, a random key's as its key data."""
    return np.asarray(jax.random.key_data(leaf) if jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key) else leaf)


def test_restore_best_without_checkpoints_returns_the_best_state_it_held(tmp_path):
    """A trainer that writes no checkpoints keeps its best state in host memory as it evaluates, so
    restore_best returns the state a checkpointed run restores as its best, every leaf of it: by the
    validation loss, an earlier state than the last, and by a best policy in its own direction."""
    metric = Value('value', Shown(better='higher'))
    for name, policy in (('loss', {}), ('value', {'metrics': [metric], 'best': metric})):
        saved = trainer(tmp_path / name).fit(data(), steps=8, log_every=1, eval_every=1, checkpoint_every=4,
                                             restore_best=True, **policy)
        unsaved = Trainer(Overfit(), optax.sgd(0.1), key=jax.random.key(0)).fit(
            data(), steps=8, log_every=1, eval_every=1, restore_best=True, **policy)
        assert int(unsaved.step) == int(saved.step) == (3 if name == 'loss' else 8)
        for left, right in zip(jax.tree.leaves(unsaved), jax.tree.leaves(saved), strict=True):
            np.testing.assert_array_equal(leaf_values(left), leaf_values(right))


def test_restore_best_without_checkpoints_ranks_by_the_training_loss_without_validation(tmp_path):
    """Without validation a run ranks its states by the training loss, as a checkpointed run's best
    does, and returns the state that ranked best though a later step ended the run."""
    stream = dataclasses.replace(data(), val=None)
    saved = trainer(tmp_path / 'run').fit(stream, steps=8, log_every=1, eval_every=1, checkpoint_every=1,
                                         restore_best=True)
    unsaved = Trainer(Overfit(), optax.sgd(0.1), key=jax.random.key(0)).fit(
        stream, steps=8, log_every=1, eval_every=1, restore_best=True)
    assert int(unsaved.step) == int(saved.step)
    for left, right in zip(jax.tree.leaves(unsaved), jax.tree.leaves(saved), strict=True):
        np.testing.assert_array_equal(leaf_values(left), leaf_values(right))


def test_a_held_best_no_evaluation_ranks_ends_the_run_failed():
    """Without checkpoints, a restore_best whose every evaluation its threshold leaves unranked has
    no state to return: the trackers hear a failed run, as a checkpointed one's do, not the final
    state handed back as the best."""
    metric = Value('value', Shown(better='lower'))
    tracker = Outcomes()
    run = Trainer(Overfit(), optax.sgd(0.1), key=jax.random.key(0), tracker=tracker)
    with pytest.raises(ValueError):
        run.fit(data(), steps=2, log_every=1, eval_every=1, metrics=[metric],
                best=Best(metric, threshold=-1.0), restore_best=True)
    assert tracker.ended == ['failed']


def test_a_scheduled_validation_records_its_loss_whatever_the_best_ranks_by():
    """A validation scheduled with no metric and a best that ranks by the training loss still
    reads its batches: it records the objective's validation loss, and restore_best returns the
    state the training loss ranked best, as a checkpointed run's best does."""
    tracker = Outcomes()
    run = Trainer(Overfit(), optax.sgd(0.1), key=jax.random.key(0), tracker=tracker)
    run.fit(data(), steps=4, log_every=1, eval_every=1, best=run.objective.loss, restore_best=True)
    assert tracker.ended == ['completed']
    assert any('val/loss' in scalars for scalars in tracker.logged)

def test_ranked_loss_and_periodic_retention(tmp_path):
    run = trainer(tmp_path / 'run', keep=Keep(latest=1, every=2))
    state = run.fit(dataclasses.replace(data(), val=None), steps=1, log_every=1)
    checkpoints = run.checkpoints
    for step, loss in [(2, .1), (3, .9), (4, .7), (5, .8)]:
        checkpoints.save(step, state.replace(step=jnp.int32(step)), None, ranking=Ranking('train/loss', loss))
        checkpoints.wait()
    assert checkpoints.best == 2
    assert {entry.step for entry in checkpoints.kept()} == {2, 4, 5}


def test_ranked_ties_keep_the_older_steps(tmp_path):
    """A ranking's best `top` steps are Orbax's `BestN` over the steps it scored: of tied scores the
    older steps are kept, and a step the ranking never scored is not among them. The run's own first
    step ranks by its higher training loss."""
    run = trainer(tmp_path / 'run', keep=Keep(latest=1))
    state = run.fit(dataclasses.replace(data(), val=None), steps=1, log_every=1)
    checkpoints = run.checkpoints
    for step, loss in [(2, .5), (3, .5), (4, None), (5, .5), (6, .7), (7, .9)]:
        ranking = None if loss is None else Ranking('train/loss', loss, 'min', top=2)
        checkpoints.save(step, state.replace(step=jnp.int32(step)), None, ranking=ranking)
        checkpoints.wait()
    assert {entry.step for entry in checkpoints.kept()} == {2, 3, 7}


def test_plateau_counts_an_improvement_as_flax_early_stopping_does(tmp_path):
    """An evaluation improves on the best where Flax's `EarlyStopping` says it does, by more than
    `min_delta` (`best - value > min_delta`): 1 - 2**-24 improves on 1 by 2**-24, more than a
    `min_delta` of 2**-24 - 2**-60, though `1 - min_delta` rounds to it in float64. So a plateau of one
    evaluation stops at the first evaluation that does not improve, the third, not the second."""
    delta, value = 2.0 ** -24 - 2.0 ** -60, 1.0 - 2.0 ** -24
    assert 1.0 - delta == value
    metric = Scripted('value', itertools.chain([1.0], itertools.repeat(value)))
    run = trainer(tmp_path / 'run')
    state = run.fit(data(), steps=10, log_every=1, eval_every=1, checkpoint_every=1, metrics=[metric],
                    stop=Plateau(metric, evals=1, min_delta=delta))
    assert int(state.step) == 3
    held = run.checkpoints.control(3)['plateau:val/value']
    assert (held['best'], held['bad']) == (value, 1)


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
    model = {'step', 'microstep', 'updates', 'variables', 'ema'}
    assert set(run.checkpoints.stored('best')) == model
    saved, _ = run.checkpoints.restore(None, 'best')
    assert set(saved) == model
    with pytest.raises(ValueError):
        run.checkpoints.restore(run.initial_state(), 'best')


def test_restore_best_reads_a_weights_only_best_as_the_model_it_holds(tmp_path):
    """A weights-only best holds the model state of its step, which is what restore_best returns,
    so a run whose best is one returns that step's counters and variables."""
    metric = Value('value', Shown(better='lower'))
    run = trainer(tmp_path / 'run')
    state = run.fit(data(), steps=4, log_every=1, eval_every=1, checkpoint_every=4, metrics=[metric],
                    best=Best(metric, weights_only=True), restore_best=True)
    saved, _ = run.checkpoints.restore(None, 'best')
    assert int(state.step) == int(saved['step']) == run.checkpoints.best == 1
    np.testing.assert_array_equal(np.asarray(state.variables['params']['w']),
                                  np.asarray(saved['variables']['params']['w']))


def test_restore_best_returns_the_best_steps_model_and_the_final_optimizer_state(tmp_path):
    """restore_best returns the model the run ranked best, the counters and variables of its best
    step, and the final step's optimizer state, in both its paths: it is the best model, not a
    point to resume from, and a run without checkpoints holds no optimizer state for it."""
    def fit(path, **options):
        checkpoints = None if path is None else Checkpoints(str(path))
        return Trainer(Overfit(), optax.adam(0.1), key=jax.random.key(0), checkpoints=checkpoints).fit(
            data(), steps=8, log_every=1, eval_every=1, checkpoint_every=4 if path else None, **options)

    models = []
    for path in (tmp_path / 'run', None):
        # A checkpointed run fit again to the step it reached trains nothing and returns its best.
        final, best = fit(path), fit(path, restore_best=True)
        assert int(best.step) == int(best.updates) < int(final.step) == 8
        assert float(best.variables['params']['w']) != float(final.variables['params']['w'])
        for kept, last in zip(jax.tree.leaves(best.opt_state), jax.tree.leaves(final.opt_state), strict=True):
            np.testing.assert_array_equal(np.asarray(kept), np.asarray(last))
        models.append((int(best.step), float(best.variables['params']['w'])))
    assert models[0] == models[1] and models[0][0] == Checkpoints(str(tmp_path / 'run')).best


def test_a_finished_run_fit_again_returns_its_best_state(tmp_path):
    """A second fit to the step the run already reached trains nothing and
    still returns the best state restore_best asks for, as the first did."""
    trainer(tmp_path / 'run').fit(data(), steps=8, log_every=1, eval_every=1, checkpoint_every=4)
    again = trainer(tmp_path / 'run')
    state = again.fit(data(), steps=8, log_every=1, eval_every=1, checkpoint_every=4, restore_best=True)
    assert again.checkpoints.latest == 8 and again.checkpoints.best == 3
    assert int(state.step) == 3
    assert float(state.variables['params']['w']) == pytest.approx(.488)


class Outcomes:
    """A tracker that keeps the run's reported end and the scalars it logged."""

    def __init__(self):
        self.ended, self.logged = [], []

    def log(self, scalars, step):
        self.logged.append(dict(scalars))

    def artifact(self, value, step):
        if isinstance(value, FitEnded):
            self.ended.append(value.status)

    def close(self):
        pass


def test_a_best_state_that_cannot_be_restored_ends_the_run_failed(tmp_path):
    """restore_best selects the state inside the run's lifecycle: when no
    step qualifies, the trackers hear a failed run, not a completed one."""
    metric = Value('value', Shown(better='lower'))
    tracker = Outcomes()
    run = Trainer(Overfit(), optax.sgd(0.1), key=jax.random.key(0),
                  checkpoints=Checkpoints(str(tmp_path / 'run')), tracker=tracker)
    with pytest.raises(FileNotFoundError):
        run.fit(data(), steps=2, log_every=1, eval_every=1, metrics=[metric],
                best=Best(metric, threshold=-1.0), checkpoint_final=False, restore_best=True)
    assert tracker.ended == ['failed']


@pytest.mark.parametrize("aggregate", [False, True])
def test_training_values_are_selected_through_the_objective(tmp_path, aggregate):
    run = trainer(tmp_path / 'run')
    stream = dataclasses.replace(data(), val=None) if aggregate else data()
    best = Best(lambda metrics: metrics[run.objective.loss]) if aggregate else run.objective.loss
    run.fit(stream, steps=4, log_every=1, checkpoint_every=4 if aggregate else 1, best=best)
    assert run.checkpoints.best == 4
    if not aggregate:
        assert run.checkpoints.kept()[0].ranked_by == 'train/loss'


def test_time_cadence_and_recorded_duration(tmp_path):
    from dew.config import TrainerConfig
    from dew.data.dataset import Loader

    assert TrainerConfig(checkpoint_every="30m").checkpoint_interval(
        Loader(lambda p: iter(()), None, records=4, batch=2)
    ) == datetime.timedelta(minutes=30)
    with pytest.raises(ValueError):
        TrainerConfig(checkpoint_every="0m").checkpoint_interval(
            Loader(lambda p: iter(()), None, records=4, batch=2)
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
    """The predicate reads a step's metrics by their objects and sees the
    record `kept` lists for that step, its ranking included."""
    metric = Value('value', Shown(better='lower'))
    seen = {}

    def where(kept):
        seen[kept.step] = kept
        return kept.metrics[metric] < .4

    run = trainer(tmp_path / 'run', keep=Keep(latest=1, where=where))
    run.fit(data(), steps=6, log_every=1, eval_every=1, checkpoint_every=1,
            metrics=[metric], best=Best(metric, mode='max'))
    retained = run.checkpoints.kept()
    assert {entry.step for entry in retained} == {1, 2, 6}
    for entry in retained:
        assert entry.metrics[metric] == entry.metrics['val/value']
        assert entry.ranked_by is not None
        assert dataclasses.replace(seen[entry.step], metrics=entry.metrics) == entry
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
    plain.checkpoints.wait()
    # The data position the run saved with its state, which a resume past step 0 needs.
    _, position = plain.checkpoints.restore(step=2, share=DataPartition())
    path, local = str(tmp_path / 'run'), str(tmp_path / 'local')
    control = {'plateau:val/value': {'best': .2, 'bad': 1, 'step': 2,
                                    'rule': {'mode': 'min', 'evals': 3, 'min_delta': 0.}}}
    checkpoints = Checkpoints(path, local_directory=local, local_every=1)
    checkpoints.save_local(2, state, position, share=DataPartition(), control=control)
    checkpoints.wait()
    resumed = Trainer(Overfit(), optax.sgd(.1), key=jax.random.key(0),
                       checkpoints=Checkpoints(path, local_directory=local, local_every=1))
    assert resumed.checkpoints.control(2) == control
    result = resumed.fit(data(), steps=10, log_every=1, eval_every=1, checkpoint_every=1,
                          metrics=[metric], stop=Plateau(metric, evals=3))
    assert int(result.step) == 4


def test_tile_head_invalidates_validation_trace_for_the_same_batch_shape():
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm import LMObjective
    from dew.training import Evaluation

    class TracedLM(LMObjective):
        traces = 0
        def loss(self, variables, batch, step):
            self.traces += 1
            return super().loss(variables, batch, step)
    model = CausalTransformer(vocab_size=16, emb_features=16, num_layers=1, num_heads=2,
                              mlp_features=32, max_seq_len=16, dtype='float32', attention_impl='xla')
    objective = TracedLM(model, seq_len=8, ema_decay=None)
    variables = objective.init(jax.random.key(0))
    batch = {'text': np.tile(np.arange(9, dtype=np.int32), (8, 1))}
    def reader(partition):
        return iter([batch])
    first = Evaluation.run(objective, variables, reader, key=jax.random.key(0), loss=True)
    old_program = objective._validation_loss
    assert objective.traces == 1
    assert objective.tile_head((4, 4)) is not None
    second = Evaluation.run(objective, variables, reader, key=jax.random.key(0), loss=True)
    assert objective.traces == 2
    assert objective._validation_loss is not old_program
    assert np.isfinite(first.scores['val/loss']) and np.isfinite(second.scores['val/loss'])
    # Replacing the immutable model (as the fit ladder does) also invalidates
    # the program, even when every variable and input shape stays the same.
    objective.model = objective.model.clone(remat=None)
    Evaluation.run(objective, variables, reader, key=jax.random.key(0), loss=True)
    assert objective.traces == 3
    # So does the context the model reads as it traces (`trace_context`): the
    # layout's rules and the pipeline's microbatch count, and returning to one
    # finds its program again.
    from flax import linen as nn

    from dew.nn.sharding import DEFAULT_RULES, pipeline_microbatches
    with nn.logical_axis_rules((("vocab", None), *DEFAULT_RULES)):
        Evaluation.run(objective, variables, reader, key=jax.random.key(0), loss=True)
    with pipeline_microbatches(2):
        Evaluation.run(objective, variables, reader, key=jax.random.key(0), loss=True)
    Evaluation.run(objective, variables, reader, key=jax.random.key(0), loss=True)
    assert objective.traces == 5


def test_a_refused_save_leaves_the_step_the_retention_policy_reads(tmp_path):
    """Orbax refuses a second save of a step it holds; the retention policy
    then still reads that step's record as it was saved, and keeps it."""
    checkpoints = Checkpoints(str(tmp_path), keep=Keep(latest=1, where=lambda kept: kept.ranked_by is None))
    state = {"v": jnp.zeros((4,))}
    checkpoints.save(1, state)
    checkpoints.wait()
    assert [kept.ranked_by for kept in checkpoints.kept()] == [None]
    with pytest.raises(ValueError):
        checkpoints.save(1, state, metrics={"loss": 1.0}, ranking=Ranking("loss", 1.0))
    checkpoints.save(2, state)
    checkpoints.wait()
    assert [kept.step for kept in checkpoints.kept()] == [1, 2]
