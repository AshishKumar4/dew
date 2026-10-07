"""Evaluation ranks precisely the weights saved, with independent best trackers."""
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


def test_missing_and_undeclared_metrics_are_refused_before_training(tmp_path):
    metric = Value('accuracy', Shown())
    with pytest.raises(ValueError, match='passed in metrics'):
        trainer(tmp_path / 'a').fit(data(), steps=2, best=metric)
    with pytest.raises(ValueError, match='declared direction'):
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


def test_multi_split_metric_keys_and_ambiguous_selection(tmp_path):
    metric = Value('value', Shown(better='higher'))
    readers = {'first': data().val, 'second': data().val}
    with pytest.raises(ValueError, match='several validation splits'):
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
    with pytest.raises(ValueError, match='inference-only'):
        run.checkpoints.restore(run.initial_state(), 'best')
    with pytest.raises(ValueError, match='requires full'):
        trainer(tmp_path / 'bad').fit(data(), steps=2, metrics=[metric],
                                    best=Best(metric, weights_only=True), restore_best=True)


def test_restore_best_returns_the_whole_earlier_state(tmp_path):
    run = trainer(tmp_path / 'run')
    state = run.fit(data(), steps=8, log_every=1, eval_every=1, checkpoint_every=4, restore_best=True)
    assert int(state.step) == 3
    assert int(state.updates) == 3
    assert float(state.variables['params']['w']) == pytest.approx(.488)


def test_training_values_are_selected_through_the_objective(tmp_path):
    run = trainer(tmp_path / 'run')
    run.fit(data(), steps=4, log_every=1, checkpoint_every=1, best=run.objective.loss)
    assert run.checkpoints.best == 4
    assert run.checkpoints.kept()[0].ranked_by == 'train/loss'


def test_time_cadence_and_recorded_duration(tmp_path):
    from dew.config import TrainerConfig
    from dew.data import Dataset

    assert TrainerConfig(checkpoint_every="30m").checkpoint_interval(
        Dataset(lambda p: iter(()), None, records=4, batch=2)
    ) == datetime.timedelta(minutes=30)
    with pytest.raises(ValueError, match='positive'):
        TrainerConfig(checkpoint_every="0m").checkpoint_interval(
            Dataset(lambda p: iter(()), None, records=4, batch=2)
        )
    with pytest.raises(TypeError, match='code-only'):
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


def test_missing_scores_do_not_rank_and_declared_max_is_inferred(tmp_path):
    run = trainer(tmp_path / 'run')
    state = run.fit(data(), steps=1, log_every=1)
    checkpoint = Checkpoints(str(tmp_path / 'isolated'), keep=1)
    checkpoint.save(1, state, None, metrics={'train/loss': .1})
    checkpoint.wait()
    assert checkpoint.best is None
    assert not checkpoint.would_keep(Ranking('score', np.nan))


def test_named_config_selector_roundtrips_and_refuses_callable():
    from dew.config import RunConfig, TrainerConfig
    config = RunConfig(trainer=TrainerConfig(best=Best('accuracy', top=2), checkpoint_every='15m'))
    assert RunConfig.from_dict(config.to_dict()) == config
    best = config.trainer.best_policies()[0]
    assert best.metric == 'accuracy' and best.top == 2
    with pytest.raises(TypeError, match='code-only'):
        TrainerConfig(best=lambda m: 1.)


def test_unranked_first_tracker_does_not_select_a_different_trackers_best(tmp_path):
    run = trainer(tmp_path / 'run')
    state = run.fit(Data(train=data()._train), steps=1, log_every=1)
    checkpoints = Checkpoints(str(tmp_path / 'isolated'))
    checkpoints.save(1, state, None, ranking=[Ranking('fid', float('nan')), Ranking('clip', 1., mode='max')])
    checkpoints.wait()
    assert checkpoints.best is None
    assert checkpoints.resolve('best:clip') == 1


def test_training_selector_aggregate_uses_bound_objective_loss(tmp_path):
    run = trainer(tmp_path / 'run')
    run.fit(Data(train=data()._train), steps=4, log_every=1, eval_every=1, checkpoint_every=4,
            best=Best(lambda m: m[run.objective.loss]))
    assert run.checkpoints.best == 4


def test_aggregate_never_reuses_missing_metrics_and_preserves_first_tracker(tmp_path):
    first = Value('first', Shown(better='lower'))
    second = Value('second', Shown(better='higher'))
    run = trainer(tmp_path / 'run')
    from dew.training.trainer import _FitPlan
    plan = _FitPlan(data(), 1, 1, 1, 1, None, [first, second], preview=False,
                    best=(Best(lambda m: m[first] - m[second]), Best(second, mode='max', split='val')),
                    validation=True)
    ranks = run._ranking(plan, {'val/second': .5})
    assert np.isnan(ranks[0].value)
    assert ranks[1].value == .5


def test_off_cadence_evaluations_only_write_winners_once(tmp_path):
    run = trainer(tmp_path / 'run')
    written = []
    save = run.checkpoints.save
    def record(*args, **kwargs):
        written.append(args[0])
        return save(*args, **kwargs)
    run.checkpoints.save = record
    run.fit(data(), steps=8, log_every=1, eval_every=1, checkpoint_every=5)
    assert written == [1, 2, 3, 5, 8]
    assert len(set(written)) == len(written)


def test_aggregate_does_not_accept_strings_for_object_owned_metrics(tmp_path):
    metric = Value('value', Shown(better='higher'))
    run = trainer(tmp_path / 'run')
    run.fit(data(), steps=2, log_every=1, eval_every=1, checkpoint_every=2,
            metrics=[metric], best=Best(lambda m: m['val/value']))
    assert run.checkpoints.best is None


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
    with pytest.raises(AttributeError, match='does not declare'):
        _ = objective.scalars.unknown_report


@pytest.mark.parametrize('given', ['fid', '{"metric":"fid","top":3}',
                                  '[{"metric":"fid"},{"metric":"clip_score","top":2}]'])
def test_named_policy_cli_is_one_argument(given):
    import tyro

    from dew.config import TrainerConfig
    config = tyro.cli(TrainerConfig, args=['--best', given])
    assert config.best is not None


def test_direct_image_metrics_declare_their_ranking_directions():
    from dew.eval import FID, CLIPScore
    assert FID().shown.better == 'lower'
    assert CLIPScore().shown.better == 'higher'


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
    assert resumed.checkpoints.control(4)['stop_reason'] == 'validation plateau'


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
    with pytest.raises(TypeError, match='code-only'):
        TrainerConfig(keep=Keep(where=lambda c: True))


def test_committed_metadata_is_cached_and_deleted_steps_leave_the_cache(tmp_path, monkeypatch):
    run = trainer(tmp_path / 'run')
    state = run.fit(Data(train=data()._train), steps=1, log_every=1)
    checkpoints = Checkpoints(str(tmp_path / 'isolated'), keep=1)
    checkpoints.save(1, state, None, ranking=Ranking('value', 1.))
    checkpoints.wait()
    persistent = checkpoints._open()
    original = persistent.metadata
    reads = []
    def metadata(step):
        reads.append(step)
        return original(step)
    monkeypatch.setattr(persistent, 'metadata', metadata)
    assert checkpoints.would_keep(Ranking('value', .5))
    first_reads = list(reads)
    assert first_reads
    assert checkpoints.would_keep(Ranking('value', .4))
    assert reads == first_reads
    # Better step 2 replaces step 1 under both latest and best policies.
    checkpoints.save(2, state.replace(step=jnp.int32(2)), None, ranking=Ranking('value', .5))
    checkpoints.wait()
    assert [checkpoint.step for checkpoint in checkpoints.kept()] == [2]
    assert 1 not in checkpoints._step_cache
    records = checkpoints.kept()
    records[0].rankings['value']['top'] = 99
    assert checkpoints.kept()[0].rankings['value']['top'] == 1


def test_validation_loss_reuses_compilation_without_retaining_dead_objectives():
    import gc
    import weakref

    from dew.training import Evaluation

    class Traced(Overfit):
        traces = 0
        def loss(self, variables, batch, step):
            self.traces += 1
            return super().loss(variables, batch, step)
    objective = Traced()
    variables = objective.init(jax.random.key(0))
    for step in (1, 2):
        report = Evaluation.run(objective, variables, data().val, key=jax.random.key(0), step=step, loss=True)
        assert report.scores['val/loss'] == .25
    assert objective.traces == 1
    owner = weakref.ref(objective)
    del objective
    gc.collect()
    assert owner() is None


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
