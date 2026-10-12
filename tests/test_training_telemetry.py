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
from dew.data.dataset import Loader
from dew.objectives import Aux, Objective
from dew.telemetry.instrumentation import StepFlops, compilation_cache_counts, count_compilation_cache
from dew.training import Trainer, WandbTracker
from dew.training.interval import Interval
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
    return Loader(lambda partition: iter({'target': np.full(8, value, np.float32)}
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


@pytest.mark.parametrize('accumulation', [1, 2])
def test_complex_gradients_have_a_real_committed_norm(accumulation):
    batch = {'target': np.broadcast_to(np.array([1 + 2j, 3 + 4j], np.complex64), (8, 2))}
    data = Loader(lambda partition: iter([batch] * accumulation), None, None, 8)
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
    assert all(row['train/learning_rate'] == pytest.approx(.1) for row in first)
    assert first[1]['train/cache_hits'] >= first[0]['train/cache_hits']
    assert first[1]['train/cache_misses'] >= first[0]['train/cache_misses']
    state = trainer.initial_state()
    devices = [SimpleNamespace(memory_stats=lambda peak=peak: None if peak is None
                               else {'peak_bytes_in_use': peak}) for peak in peaks]
    monkeypatch.setattr(jax, 'local_devices', lambda: devices)
    monkeypatch.setattr(jax, 'devices', lambda: [SimpleNamespace(device_kind='NVIDIA A100-SXM4-80GB')])
    monkeypatch.setattr(jax, 'device_count', lambda: 1)
    interval = Interval(time.time() - 2, last_saved=None, since_log=2, samples=16,
                         flops=StepFlops(model=312e12, hardware=624e12), waited=.02, compiling=.1, compiles=1)
    loss = jnp.asarray(1.)
    accepted = jnp.asarray(a=True)
    count_compilation_cache()
    before = compilation_cache_counts()
    jax.monitoring.record_event('/jax/compilation_cache/cache_hits')
    jax.monitoring.record_event('/jax/compilation_cache/cache_misses')
    jax.monitoring.record_event('/jax/compilation_cache/cache_misses')
    trainer._log_interval(3, loss, {}, accepted, state, interval, statistics=dict)
    row = run.history[-1]
    names = {'train/loss', 'train/learning_rate', 'train/step_time_ms', 'train/data_wait_ms',
             'train/compile_s', 'train/hlo_flops_util', 'train/hfu', 'train/compiles', 'train/cache_hits',
             'train/cache_misses'}
    assert names <= row.keys()
    assert row['train/cache_hits'] == before['train/cache_hits'] + 1
    assert row['train/cache_misses'] == before['train/cache_misses'] + 2
    assert row['train/hfu'] == pytest.approx(2 * row['train/hlo_flops_util'])
    assert row['train/compiles'] == 1
    if expected is None:
        assert 'train/peak_memory_gib' not in row
    else:
        assert row['train/peak_memory_gib'] == expected
    sink.close()


def test_wandb_counts_each_new_step_shape_once(monkeypatch):
    import wandb
    from test_publish import TrackedRun

    run = TrackedRun()
    monkeypatch.setattr(wandb, 'init', lambda **_: run)
    with WandbTracker('dew', 'shapes', offline=True) as sink:
        trainer = Trainer(Squared(), optax.sgd(.1), key=0, tracker=sink)
        data = Loader(lambda partition: iter(
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


def test_cache_counting_starts_once_however_often_it_is_asked():
    for _ in range(3):
        count_compilation_cache()
    before = compilation_cache_counts()
    jax.monitoring.record_event('/jax/compilation_cache/cache_hits')
    jax.monitoring.record_event('/jax/compilation_cache/cache_misses')
    after = compilation_cache_counts()
    assert after['train/cache_hits'] == before['train/cache_hits'] + 1
    assert after['train/cache_misses'] == before['train/cache_misses'] + 1
