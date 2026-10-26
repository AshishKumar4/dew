"""The sweep: a trial per point through the ordinary train entry point, a
ledger an interrupted sweep continues from, and the three search backends.

The trials train the same small regression the trainer tests use, through
`RunConfig.train`, so what is asserted is what a swept run really produced:
the record each trial trained from, the ledger, and what reached the
caller's tracker.
"""

import json

from affine_run import TimingRegression as Regression, batches

from dew.config import RunConfig, TrainerConfig
from dew.config.sweep import GridSearch
from dew.data.dataset import Loader
from dew.training import LocalTracker

SPACE = {'optim.learning_rate': [0.1, 0.01], 'optim.clip_grads': [0.0, 1.0]}


def config(tmp_path, name='sweep'):
    return RunConfig(trainer=TrainerConfig(
        name=name, checkpoint_dir=str(tmp_path / 'runs'), steps=2,
        eval_every=None, checkpoint_every=None))


def train(run: RunConfig) -> float:
    """The loss the trial's own run ends on, through the normal entry point."""
    objective = Regression()
    assert run.trainer.name is not None
    state = run.train(objective, Loader(lambda partition: batches(), None, None, 8), name=run.trainer.name)
    loss, _ = objective.loss(state.variables, next(batches()), state.step)
    return float(loss)


def journal(directory, name):
    return [json.loads(line) for line in (directory / name).read_text().splitlines()]


def test_four_trials_train_their_own_points_and_reach_the_callers_tracker(tmp_path):
    with LocalTracker(tmp_path / 'sweep') as tracker:
        trials = config(tmp_path).sweep(SPACE, train=train, trials=4,
                                        ledger=tmp_path / 'ledger.json', tracker=tracker, search=GridSearch())

    assert [trial.name for trial in trials] == [f'sweep/trial-{index}' for index in range(4)]
    # Every point of the grid was trained, each under its own run.
    assert ({tuple(sorted(trial.overrides.items())) for trial in trials}
            == {(('optim.clip_grads', clip), ('optim.learning_rate', rate))
                for rate in (0.1, 0.01) for clip in (0.0, 1.0)})
    trained = [json.loads((tmp_path / 'runs' / trial.name / 'run.json').read_text())['fields']
               for trial in trials]
    assert [run['optim']['learning_rate'] for run in trained] == [0.1, 0.1, 0.01, 0.01]
    assert [run['optim']['clip_grads'] for run in trained] == [0.0, 1.0, 0.0, 1.0]
    # The swept rate reached the optimizer, not just the record.
    assert trials[0].value != trials[2].value

    reported = journal(tmp_path / 'sweep', 'records.jsonl')
    assert [row['type'] for row in reported] == ['TrialFinished'] * 4
    assert [row['value']['name'] for row in reported] == [trial.name for trial in trials]
    assert ([row['scalars']['sweep/value'] for row in journal(tmp_path / 'sweep', 'scalars.jsonl')]
            == [trial.value for trial in trials])
    assert json.loads((tmp_path / 'ledger.json').read_text())['trials'][2]['index'] == 2



def test_optunas_first_trials_draw_apart_and_a_resumed_sweep_draws_the_same():
    """TPE draws its first ten trials at random, and the finished trials a study is given are added
    without drawing: each trial's point comes from the sweep's seed and its number, so the first
    trials of a large space are not one repeated point, and the same ledger asks the same point."""
    from dew.config.sweep import OptunaSearch
    from dew.telemetry.records import TrialFinished

    space = {"optim.learning_rate": [10.0 ** -exponent for exponent in range(1, 41)]}
    search = OptunaSearch(seed=3)
    finished: list[TrialFinished] = []
    for index in range(5):
        point = search(space, finished)
        assert search(space, finished) == point
        finished.append(TrialFinished(index=index, name=f"trial-{index}", overrides=point,
                                      value=float(index)))
    assert len({trial.overrides["optim.learning_rate"] for trial in finished}) > 1


def test_a_grid_of_a_million_points_names_each_trial_as_its_product_orders_it():
    """A grid search's next point is the trial's place in the space's cartesian product, its last
    field varying fastest, as `itertools.product` walks it; a grid of a million points names a late
    trial without building the points before it, and refuses a trial past its last."""
    import itertools

    import pytest

    from dew.telemetry.records import TrialFinished

    space = {'a': list(range(100)), 'b': list(range(100)), 'c': list(range(100))}
    search = GridSearch()

    def done(count):
        return [TrialFinished(index, str(index), {}, 0.0) for index in range(count)]

    walked = list(itertools.product(*SPACE.values()))
    assert [search(SPACE, done(index)) for index in range(4)] == [
        dict(zip(SPACE, point, strict=True)) for point in walked]
    assert search(space, done(123_456)) == {'a': 12, 'b': 34, 'c': 56}
    with pytest.raises(ValueError):
        search(SPACE, done(4))
