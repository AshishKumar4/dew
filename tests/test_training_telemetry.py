"""Training telemetry describes the admitted optimizer work and its checkout."""

import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

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
from dew.telemetry.instrumentation import StepFlops
from dew.training import Trainer, WandbTracker
from dew.training.trainer import _Interval


class Squared(Objective):
    """A two-coordinate gradient with unit direction (0.6, 0.8)."""

    def init(self, key, variables=None):
        return {'params': {'weight': jnp.zeros(2)}}

    def loss(self, variables, batch, step):
        prediction = jnp.vdot(variables['params']['weight'], jnp.array([.6, .8]))
        return jnp.mean((prediction - batch['target']) ** 2), Aux({'probe': prediction})


def targets(values):
    return Dataset(lambda partition: iter({'target': np.full(8, value, np.float32)}
                                          for value in values), None, None, 8)


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
    assert record['config']['trainer'] == config.to_dict()['trainer']
    assert record['config']['optim'] == config.to_dict()['optim']
    assert record['dew_sha'] == expected_sha
    assert record['device_kind'] == jax.devices()[0].device_kind
    assert record['devices'] == jax.device_count()
    assert record['processes'] == jax.process_count()
    assert not run.logged


def test_checkout_revision_is_none_outside_a_checkout(tmp_path, monkeypatch):
    from dew.telemetry import records

    monkeypatch.setattr(records, '__file__', str(tmp_path / 'src' / 'dew' / 'telemetry' / 'records.py'))
    assert records.checkout_revision() is None


@pytest.mark.parametrize('peaks,expected', [([2 * 1024 ** 3, 5 * 1024 ** 3], 5.), ([None, None], None)])
def test_wandb_tick_receives_memory_cache_and_existing_throughput(tmp_path, monkeypatch, peaks, expected):
    import wandb
    from test_publish import TrackedRun

    run = TrackedRun()
    monkeypatch.setattr(wandb, 'init', lambda **_: run)
    sink = WandbTracker('dew', 'tick', offline=True)
    trainer = Trainer(Squared(), OptimConfig(learning_rate=.1), key=0, tracker=sink)
    trainer.fit(targets([1., 3.]), steps=2, log_every=1)
    first = [row for row in run.history if 'train/loss' in row]
    assert all(row['train/compiles'] == 1 for row in first)
    assert first[1]['train/cache_hits'] >= first[0]['train/cache_hits']
    assert first[1]['train/cache_misses'] >= first[0]['train/cache_misses']
    state = trainer.initial_state()
    devices = [SimpleNamespace(memory_stats=lambda peak=peak: None if peak is None
                               else {'peak_bytes_in_use': peak}) for peak in peaks]
    monkeypatch.setattr(jax, 'local_devices', lambda: devices)
    monkeypatch.setattr(jax, 'devices', lambda: [SimpleNamespace(device_kind='NVIDIA A100-SXM4-80GB')])
    monkeypatch.setattr(jax, 'device_count', lambda: 1)
    interval = _Interval(time.time() - 2, last_saved=None, since_log=2, samples=16,
                         flops=StepFlops(model=312e12, hardware=624e12), waited=.02, compiling=.1, compiles=1)
    loss = jnp.asarray(1.)
    jax.monitoring.record_event('/jax/compilation_cache/cache_hits')
    jax.monitoring.record_event('/jax/compilation_cache/cache_misses')
    jax.monitoring.record_event('/jax/compilation_cache/cache_misses')
    trainer._log_interval(3, loss, {}, jnp.asarray(a=True), state, interval)
    row = run.history[-1]
    names = {'train/loss', 'train/learning_rate', 'train/step_time_ms', 'train/data_wait_ms',
             'train/compile_s', 'train/mfu', 'train/hfu', 'train/compiles', 'train/cache_hits',
             'train/cache_misses'}
    assert names <= row.keys()
    assert row['train/cache_hits'] == first[-1]['train/cache_hits'] + 1
    assert row['train/cache_misses'] == first[-1]['train/cache_misses'] + 2
    assert row['train/hfu'] == pytest.approx(2 * row['train/mfu'])
    assert row['train/compiles'] == 1
    if expected is None:
        assert 'train/peak_memory_gib' not in row
    else:
        assert row['train/peak_memory_gib'] == expected
    sink.close()


def test_cache_listener_is_registered_once_across_trainers():
    from dew.telemetry.instrumentation import compilation_cache_counts

    trainers = [Trainer(Squared(), optax.sgd(.1), key=0) for _ in range(3)]
    before = compilation_cache_counts()
    jax.monitoring.record_event('/jax/compilation_cache/cache_hits')
    jax.monitoring.record_event('/jax/compilation_cache/cache_misses')
    after = compilation_cache_counts()
    assert len(trainers) == 3
    assert after['train/cache_hits'] == before['train/cache_hits'] + 1
    assert after['train/cache_misses'] == before['train/cache_misses'] + 1
