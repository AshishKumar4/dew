"""Training telemetry describes the admitted optimizer work and its checkout."""

import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from recording import RecordingTracker

import dew.config
from dew.config import OptimConfig, RunConfig, TrainerConfig, Wandb
from dew.data import Dataset
from dew.objectives import Aux, Objective
from dew.training import Trainer, WandbTracker
from dew.training.optim import Cosine


class Squared(Objective):
    """A two-coordinate gradient with unit direction (0.6, 0.8)."""

    def init(self, key, variables=None):
        return {'params': {'weight': jnp.zeros(2)}}

    def loss(self, variables, batch, step):
        prediction = jnp.vdot(variables['params']['weight'], jnp.array([.6, .8]))
        return jnp.mean((prediction - batch['target']) ** 2), Aux({'probe': prediction})


class ComplexSquared(Objective):
    """A real loss whose optimizer input has complex coordinates, as an SSM's can."""

    def init(self, key, variables=None):
        return {'params': {'weight': jnp.zeros(2, jnp.complex64)}}

    def loss(self, variables, batch, step):
        error = variables['params']['weight'] - batch['target']
        return jnp.mean(jnp.sum(jnp.abs(error) ** 2, axis=-1)), Aux({})


def targets(values):
    return Dataset(lambda partition: iter({'target': np.full(8, value, np.float32)}
                                          for value in values), None, None, 8)


@pytest.mark.parametrize('cadence,step,streak', [(1, 7, 5), (2, 8, 6)])
def test_a_non_finite_run_stops_at_the_step_whose_losses_it_read(cadence, step, streak):
    """The losses go non-finite from step 3 on. A log is read once the next
    step is dispatched, and the stop names the step the log was taken at,
    with the streak its losses held, and every earlier log is reported."""
    sink = RecordingTracker()
    trainer = Trainer(Squared(), optax.sgd(.1), key=0, tracker=sink)
    with pytest.raises(RuntimeError, match=f"{streak} consecutive steps ending near step {step},"):
        trainer.fit(targets([1., 1.] + [float('nan')] * 8), steps=10, log_every=cadence)
    logged = [row for row, scalars in sink.scalars if 'train/loss' in scalars]
    assert logged == list(range(cadence, step, cadence))


def test_each_checkpoint_follows_the_log_of_its_step(tmp_path):
    """A log is read a step late, but never past a checkpoint: each save,
    persistent or local, comes after its step's row, and the rows are the
    ones a run logging every step reports at those steps."""
    from dew.telemetry.records import CheckpointRequested
    from dew.training import Checkpoints

    class Ordered(RecordingTracker):
        def __init__(self):
            super().__init__()
            self.events = []

        def log(self, scalars, step):
            super().log(scalars, step)
            if 'train/loss' in scalars:
                self.events.append(('log', step))

        def artifact(self, value, step):
            super().artifact(value, step)
            if isinstance(value, CheckpointRequested):
                self.events.append(('local' if value.local else 'checkpoint', step))

    def rows(log_every, directory=None):
        sink = Ordered()
        checkpoints = None if directory is None else Checkpoints(
            str(directory / 'run'), local_directory=str(directory / 'local'), local_every=3)
        Trainer(Squared(), optax.sgd(.1), key=0, tracker=sink, checkpoints=checkpoints).fit(
            targets([1., 3., 2., 6., 1., 3., 2., 5.]), steps=8, log_every=log_every,
            checkpoint_every=None if directory is None else 2)
        return sink

    ordered, every = rows(2, tmp_path), rows(1)
    saves = [event for event in ordered.events if event[0] != 'log']
    assert {kind for kind, _ in saves} == {'checkpoint', 'local'}
    for save in saves:
        logged = [step for kind, step in ordered.events[:ordered.events.index(save)] if kind == 'log']
        assert logged and logged[-1] == save[1] // 2 * 2
    losses = {step: scalars['train/loss'] for step, scalars in every.scalars if 'train/loss' in scalars}
    assert {step: scalars['train/loss'] for step, scalars in ordered.scalars if 'train/loss' in scalars} == {
        step: losses[step] for step in (2, 4, 6, 8)}


@pytest.mark.parametrize('cadence,expected', [
    (1, {2: 4., 4: 7.2, 6: 1.76}), (3, {3: 4., 6: 4.48}), (6, {6: 4.32}),
])
def test_grad_norm_averages_committed_windows_only(cadence, expected):
    sink = RecordingTracker()
    trainer = Trainer(Squared(), optax.sgd(.1), key=0, tracker=sink, accumulation=2)
    trainer.fit(targets([1., 3., 2., 6., 1., 3.]), steps=6, log_every=cadence)
    rows = {step: values['train/grad_norm'] for step, values in sink.scalars
            if 'train/grad_norm' in values}
    assert rows == pytest.approx(expected)


def test_rejected_attempts_do_not_contribute_to_grad_norm():
    sink = RecordingTracker()
    trainer = Trainer(Squared(), optax.sgd(.1), key=0, tracker=sink, accumulation=2, dynamic_scale=True)
    trainer.fit(targets([1., float('nan'), 3., 2.]), steps=4, log_every=1)
    rows = {step: values['train/grad_norm'] for step, values in sink.scalars
            if 'train/grad_norm' in values}
    assert rows == pytest.approx({3: 4.})


@pytest.mark.parametrize('accumulation', [1, 2])
def test_complex_gradients_have_a_real_committed_norm(accumulation):
    batch = {'target': np.broadcast_to(np.array([1 + 2j, 3 + 4j], np.complex64), (8, 2))}
    data = Dataset(lambda partition: iter([batch] * accumulation), None, None, 8)
    sink = RecordingTracker()
    trainer = Trainer(ComplexSquared(), optax.sgd(.1), key=0, tracker=sink, accumulation=accumulation)
    trainer.fit(data, steps=accumulation, log_every=1)
    rows = {step: scalars['train/grad_norm'] for step, scalars in sink.scalars
            if 'train/grad_norm' in scalars}
    assert rows == pytest.approx({accumulation: np.sqrt(120.)})


def test_run_record_names_the_package_checkout_and_hardware(tmp_path, monkeypatch):
    import wandb
    from test_publish import TrackedRun

    run = TrackedRun()
    monkeypatch.setattr(wandb, 'init', lambda **_: run)
    config = RunConfig(trainer=TrainerConfig(
        name='telemetry', steps=2, batch_size=8, checkpoint_dir=str(tmp_path), log_every=1,
        eval_every=None, checkpoint_every=None, wandb=Wandb('dew', 'ashishkumar4', offline=True)))
    root = Path(dew.config.__file__).resolve().parents[3]
    expected_sha = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    monkeypatch.chdir(tmp_path)
    config.train(Squared(), targets([1., 3.]), name='telemetry')
    record = run.configuration['run_record']
    assert record['config'] == config.recorded(Squared()).to_dict()
    assert record['dew_sha'] == expected_sha
    assert record['device_kind'] == jax.devices()[0].device_kind
    assert record['devices'] == jax.device_count()
    assert record['processes'] == jax.process_count()
    assert not run.logged


def test_wandb_counts_each_new_step_shape_once(monkeypatch):
    import wandb
    from test_publish import TrackedRun

    run = TrackedRun()
    monkeypatch.setattr(wandb, 'init', lambda **_: run)
    with WandbTracker('dew', 'shapes', offline=True) as sink:
        trainer = Trainer(Squared(), optax.sgd(.1), key=0, tracker=sink)
        data = Dataset(lambda partition: iter(
            {'target': np.ones((8, columns), np.float32)} for columns in (1, 2, 1, 2)), None, None, 8)
        trainer.fit(data, steps=4, log_every=1)
    rows = [row for row in run.history if 'train/loss' in row]
    assert [row['train/compiles'] for row in rows] == [1, 2, 2, 2]
    assert rows[1]['train/compile_s'] > 0
    assert sum(row.get('reporting/type') == 'StepCompiled' for row in run.history) == 2


@pytest.mark.parametrize('every', [1, 2])
def test_configured_cosine_rate_is_the_last_applied_update(every, monkeypatch):
    import wandb
    from test_publish import TrackedRun

    run = TrackedRun()
    monkeypatch.setattr(wandb, 'init', lambda **_: run)
    schedule = Cosine(peak=.1, init=.01, end=.005, warmup_steps=1, every=every)
    with WandbTracker('dew', 'scheduled', offline=True) as sink:
        trainer = Trainer(Squared(), OptimConfig(schedule=schedule), key=0, tracker=sink, accumulation=2)
        trainer.fit(targets([1.] * 16), steps=16, log_every=1)
    oracle = optax.warmup_cosine_decay_schedule(
        init_value=.01, peak_value=.1, warmup_steps=1, decay_steps=8 // every, end_value=.005)
    rows = [row for row in run.history if 'train/loss' in row]
    assert len(rows) == 16
    for row in rows:
        update = max(row['train/step'] // 2 - 1, 0)
        assert row['train/learning_rate'] == pytest.approx(float(oracle(update // every)))
