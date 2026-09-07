"""The sweep: a trial per point through the ordinary train entry point, a
ledger an interrupted sweep continues from, and the three search backends.

The trials train the same small regression the trainer tests use, through
`RunConfig.train`, so what is asserted is what a swept run really produced:
the record each trial trained from, the ledger, and what reached the
caller's tracker.
"""

import json

import pytest

from dew.config import RunConfig, TrainerConfig
from dew.config.sweep import grid_search, optuna_search, override, random_search, sweep
from dew.data import Dataset
from dew.telemetry.records import TrialFinished
from dew.training import LocalTracker
from test_instrumentation import Regression, batches

SPACE = {'optim.learning_rate': [0.1, 0.01], 'optim.clip_grads': [0.0, 1.0]}


def config(tmp_path, name='sweep'):
    return RunConfig(trainer=TrainerConfig(
        name=name, checkpoint_dir=str(tmp_path / 'runs'), steps=2, batch_size=8,
        eval_every=None, checkpoint_every=None))


def train(run: RunConfig) -> float:
    """The loss the trial's own run ends on, through the normal entry point."""
    objective = Regression()
    assert run.trainer.name is not None
    state = run.train(objective, Dataset(batches, None, None, 8), name=run.trainer.name)
    loss, _ = objective.loss(state.params, next(batches()), state.step)
    return float(loss)


def journal(directory, name):
    return [json.loads(line) for line in (directory / name).read_text().splitlines()]


def test_four_trials_train_their_own_points_and_reach_the_callers_tracker(tmp_path):
    with LocalTracker(tmp_path / 'sweep') as tracker:
        trials = sweep(config(tmp_path), SPACE, train=train, trials=4,
                       ledger=tmp_path / 'ledger.json', tracker=tracker, search=grid_search)

    assert [trial.name for trial in trials] == [f'sweep/trial-{index}' for index in range(4)]
    # Every point of the grid was trained, each under its own run.
    assert ({tuple(sorted(trial.overrides.items())) for trial in trials}
            == {(('optim.clip_grads', clip), ('optim.learning_rate', rate))
                for rate in (0.1, 0.01) for clip in (0.0, 1.0)})
    trained = [json.loads((tmp_path / 'runs' / trial.name / 'run.json').read_text())
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


def test_an_interrupted_sweep_continues_at_the_trial_it_stopped_on(tmp_path):
    trained = []

    def counted(run: RunConfig) -> float:
        trained.append(run.trainer.name)
        if len(trained) == 3:
            raise KeyboardInterrupt('the operator stopped the sweep')
        return float(len(trained))

    arguments = dict(train=counted, trials=4, ledger=tmp_path / 'ledger.json',
                     search=grid_search)
    with LocalTracker(tmp_path / 'sweep') as tracker:
        with pytest.raises(KeyboardInterrupt):
            sweep(config(tmp_path), SPACE, tracker=tracker, **arguments)
        assert trained == ['sweep/trial-0', 'sweep/trial-1', 'sweep/trial-2']
        assert len(json.loads((tmp_path / 'ledger.json').read_text())['trials']) == 2

        trained.clear()
        trials = sweep(config(tmp_path), SPACE, tracker=tracker, **arguments)

    assert trained == ['sweep/trial-2', 'sweep/trial-3']
    assert [trial.index for trial in trials] == [0, 1, 2, 3]
    # The finished trials kept the values and points of their own run.
    assert [trial.value for trial in trials] == [1.0, 2.0, 1.0, 2.0]
    assert [trial.overrides['optim.learning_rate'] for trial in trials] == [0.1, 0.1, 0.01, 0.01]


def test_a_ledger_of_another_space_is_refused(tmp_path):
    with LocalTracker(tmp_path / 'sweep') as tracker:
        arguments = dict(train=lambda run: 1.0, trials=1, ledger=tmp_path / 'ledger.json',
                         tracker=tracker)
        sweep(config(tmp_path), SPACE, **arguments)
        with pytest.raises(ValueError, match='not this space'):
            sweep(config(tmp_path), {'optim.learning_rate': [0.5]}, **arguments)


def test_a_path_the_run_record_does_not_declare_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown fields \\['learn_rate'\\]"):
        override(config(tmp_path), {'optim.learn_rate': 0.1})
    with pytest.raises(KeyError, match='names no group'):
        override(config(tmp_path), {'optimizer.learning_rate': 0.1})
    assert override(config(tmp_path), {'optim.learning_rate': 0.1}).optim.learning_rate == 0.1


def test_random_search_repeats_a_trial_numbers_draw_and_moves_over_the_space():
    def point(index):
        return random_search(SPACE, [TrialFinished(number, f'sweep/trial-{number}', {}, 1.0)
                                     for number in range(index)], 3)

    assert point(1) == point(1)
    drawn = [point(index) for index in range(6)]
    assert all(row[path] in values for row in drawn for path, values in SPACE.items())
    assert len({tuple(sorted(row.items())) for row in drawn}) > 1


def test_the_optuna_backend_asks_within_the_space_and_takes_the_ledgers_trials(tmp_path):
    pytest.importorskip('optuna')
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with LocalTracker(tmp_path / 'sweep') as tracker:
        trials = sweep(config(tmp_path), SPACE, train=lambda run: run.optim.learning_rate,
                       trials=3, ledger=tmp_path / 'ledger.json', tracker=tracker,
                       search=optuna_search)
    assert [trial.index for trial in trials] == [0, 1, 2]
    assert all(trial.overrides[path] in values
               for trial in trials for path, values in SPACE.items())
    # The ask after two told trials is the study's, not a redraw of trial one.
    assert optuna_search(SPACE, trials[:2], 0) == optuna_search(SPACE, trials[:2], 0)


def test_a_sweep_without_a_run_name_is_refused(tmp_path):
    with LocalTracker(tmp_path / 'sweep') as tracker:
        with pytest.raises(ValueError, match='trainer.name'):
            sweep(config(tmp_path, name=None), SPACE, train=lambda run: 1.0, trials=1,
                  ledger=tmp_path / 'ledger.json', tracker=tracker)


