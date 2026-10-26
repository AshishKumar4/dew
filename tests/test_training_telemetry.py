"""Training telemetry describes the admitted optimizer work and its checkout."""

import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from recording import RecordingTracker
from reference_error import as_given, assert_as_exact

import dew.config
from dew.config import RunConfig, TrainerConfig
from dew.data.dataset import Loader
from dew.objectives import Aux, Objective
from dew.optim import Cosine, OptimConfig
from dew.training import Trainer, Wandb, WandbTracker


class Squared(Objective):
    """A two-coordinate gradient with unit direction (0.6, 0.8)."""

    def fresh_variables(self, key, held):
        return {'params': {'weight': jnp.zeros(2)}}

    def loss(self, variables, batch, step):
        prediction = jnp.vdot(variables['params']['weight'], jnp.array([.6, .8]))
        return jnp.mean((prediction - batch['target']) ** 2), Aux({'probe': prediction})


class ComplexSquared(Objective):
    """A real loss whose optimizer input has `width` complex coordinates, as an SSM's can."""

    def __init__(self, width=2):
        self.width = width

    def fresh_variables(self, key, held):
        return {'params': {'weight': jnp.zeros(self.width, jnp.complex64)}}

    def loss(self, variables, batch, step):
        error = variables['params']['weight'] - batch['target']
        return jnp.mean(jnp.sum(jnp.abs(error) ** 2, axis=-1)), Aux({})


def targets(values):
    return Loader(lambda partition: iter({'target': np.full(8, value, np.float32)}
                                          for value in values), None, None, 8)


class Targets:
    """`targets`' stream with its position, which a checkpoint saves."""

    def __init__(self, values):
        self.values, self.index = values, 0

    def __iter__(self):
        return self

    def __next__(self):
        self.index += 1
        return {'target': np.full(8, self.values[self.index - 1], np.float32)}

    def get_state(self):
        return json.dumps(self.index).encode()

    def set_state(self, state):
        self.index = json.loads(state)


@pytest.mark.parametrize('cadence,step', [(1, 7), (2, 8)])
def test_a_non_finite_run_stops_at_the_step_whose_losses_it_read(cadence, step):
    """The losses go non-finite from step 3 on. A log is read once the next
    step is dispatched, and the run stops at the log whose losses held the
    fifth non-finite one, every earlier log reported; the same run over
    finite losses trains to its end."""
    Trainer(Squared(), optax.sgd(.1), key=0).fit(targets([1.] * 10), steps=10, log_every=cadence)
    sink = RecordingTracker()
    trainer = Trainer(Squared(), optax.sgd(.1), key=0, tracker=sink)
    with pytest.raises(RuntimeError):
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
        data = Loader(lambda partition: Targets([1., 3., 2., 6., 1., 3., 2., 5.]), None, None, 8)
        Trainer(Squared(), optax.sgd(.1), key=0, tracker=sink, checkpoints=checkpoints).fit(
            data, steps=8, log_every=log_every,
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


def test_a_run_ending_between_logs_reports_its_last_interval():
    """A run whose last step falls between logging steps logs that step as a logging step: three
    steps at log_every=100 report step 3, and five at log_every=2 report steps 2, 4 and 5, each
    with the loss, objective metrics and committed gradient norm the run sends at its logs."""
    values = [1., 3., 2., 6., 1.]
    every = RecordingTracker()
    Trainer(Squared(), optax.sgd(.1), key=0, tracker=every).fit(targets(values), steps=5, log_every=1)
    losses = {step: scalars['train/loss'] for step, scalars in every.scalars if 'train/loss' in scalars}
    for steps, cadence, logged in ((3, 100, [3]), (5, 2, [2, 4, 5])):
        sink = RecordingTracker()
        trainer = Trainer(Squared(), optax.sgd(.1), key=0, tracker=sink)
        trainer.fit(targets(values), steps=steps, log_every=cadence)
        rows = {step: scalars for step, scalars in sink.scalars if 'train/loss' in scalars}
        assert sorted(rows) == logged
        assert {step: row['train/loss'] for step, row in rows.items()} == pytest.approx(
            {step: losses[step] for step in logged})
        assert all({'train/probe', 'train/grad_norm'} <= row.keys() for row in rows.values())


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
    # The rejection drops its window, the first target's with it: the one update is the last two's.
    assert rows == pytest.approx({4: 5.})


class Repeated:
    """One batch over and over, with its position, which a checkpoint saves."""

    def __init__(self, batch):
        self.batch, self.index = batch, 0

    def __iter__(self):
        return self

    def __next__(self):
        self.index += 1
        return self.batch

    def get_state(self):
        return json.dumps(self.index).encode()

    def set_state(self, state):
        self.index = json.loads(state)


COMPLEX_SEEDS = 8
"""Check 1's declared seeds for the complex descent, sized by `seeds_for` (2026-10-21, on armada)."""


COMPLEX_WIDTH = 64
"""The drawn targets' complex coordinates: enough for a few seeds to resolve the factor."""


def complex_target(seed: int) -> tuple[np.ndarray]:
    """A target of COMPLEX_WIDTH complex coordinates, each part drawn from a unit normal."""
    parts = np.random.default_rng(seed).normal(size=(2, COMPLEX_WIDTH))
    return ((parts[0] + 1j * parts[1]).astype(np.complex64),)


def parts(value):
    """A complex array as its real and imaginary parts, stacked: what check 1 compares."""
    value = np.asarray(value)
    return np.stack([value.real, value.imag])


@pytest.mark.parametrize('accumulation', [1, 2])
def test_complex_weights_descend_and_resume_where_the_whole_run_lands(tmp_path, accumulation):
    """A real loss of complex weights descends. JAX's gradient of a real function of a complex
    weight is the conjugate of its steepest ascent, so the optimizer takes its conjugate: SGD at
    0.1 on |w - t|^2 moves w by -0.2 (w - t) each update, w_k = t (1 - 0.8^k). The committed norm is
    the gradient's Euclidean norm, 2|t| at w = 0, and a run resumed from its checkpoint after two
    updates lands where the whole run does."""
    from dew.training import Checkpoints

    target = np.array([1 + 2j, 3 + 4j], np.complex64)
    untracked = Trainer(ComplexSquared(COMPLEX_WIDTH), optax.sgd(.1), key=0, accumulation=accumulation)

    def fit(steps, target=target, directory=None, sink=None):
        trainer = untracked if directory is None and sink is None else Trainer(
            ComplexSquared(), optax.sgd(.1), key=0, tracker=sink, accumulation=accumulation,
            checkpoints=None if directory is None else Checkpoints(str(directory)))
        batch = {'target': np.broadcast_to(target, (8, len(target)))}
        data = Loader(lambda partition: Repeated(batch), None, None, 8)
        return trainer.fit(data, steps=steps * accumulation, log_every=accumulation,
                           checkpoint_every=None if directory is None else accumulation)

    sink = RecordingTracker()
    weight = np.asarray(fit(4, sink=sink).variables['params']['weight'])

    # The same four updates in Optax's own loop over the same batch, complex64 as the trainer's, and
    # the closed form in complex128, which both are held to over drawn targets (check 1), real and
    # imaginary parts alike.
    @jax.jit
    def optax_descent(target):
        batch = {'target': jnp.broadcast_to(target, (8, COMPLEX_WIDTH))}

        def update(carry, _):
            weight, state = carry
            loss = ComplexSquared().loss
            grads = jax.grad(lambda w: loss({'params': {'weight': w}}, batch, None)[0])(weight)
            updates, state = optax.sgd(.1).update(grads, state, weight)
            return (optax.apply_updates(weight, updates), state), None

        start = jnp.zeros(COMPLEX_WIDTH, jnp.complex64)
        (weight, _), _ = jax.lax.scan(update, (start, optax.sgd(.1).init(start)), None, length=4)
        return weight

    assert_as_exact(as_given(lambda target: parts(fit(4, target).variables['params']['weight'])),
                    as_given(lambda target: parts(optax_descent(target))), complex_target, "complex descent",
                    seeds=COMPLEX_SEEDS,
                    truth_fn=lambda target: parts(np.asarray(target, np.complex128) * (1 - 0.8 ** 4)))
    norms = {step: scalars['train/grad_norm'] for step, scalars in sink.scalars
             if 'train/grad_norm' in scalars}
    assert norms[accumulation] == pytest.approx(np.sqrt(120.))
    fit(2, directory=tmp_path)
    resumed = fit(4, directory=tmp_path).variables['params']['weight']
    np.testing.assert_array_equal(np.asarray(resumed), weight)


def test_run_record_names_the_package_checkout_and_hardware(tmp_path, monkeypatch):
    import wandb
    from test_publish import TrackedRun

    run = TrackedRun()
    monkeypatch.setattr(wandb, 'init', lambda **_: run)
    config = RunConfig(trainer=TrainerConfig(
        name='telemetry', steps=2, checkpoint_dir=str(tmp_path), log_every=1,
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


@pytest.mark.parametrize("shown", [None, "arm 3, stage 2"])
def test_a_runs_wandb_name_is_its_own_and_its_checkpoints_keep_the_runs(tmp_path, monkeypatch, shown):
    """A W&B run shows the name the tracker is given, and the run's own name where it is given none;
    its checkpoints are kept under the run's name either way."""
    import wandb
    from test_publish import TrackedRun

    opened = {}

    def init(**given):
        opened.update(given)
        return TrackedRun()

    monkeypatch.setattr(wandb, 'init', init)
    config = RunConfig(trainer=TrainerConfig(
        name='kept', steps=2, checkpoint_dir=str(tmp_path), log_every=1, eval_every=None,
        checkpoint_every=None, wandb=Wandb('dew', offline=True, name=shown)))
    monkeypatch.chdir(tmp_path)
    config.train(Squared(), targets([1., 3.]), name='kept')
    assert opened['name'] == ('kept' if shown is None else shown)
    assert (tmp_path / 'kept').is_dir()


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
    schedule = Cosine(peak_value=.1, init_value=.01, end_value=.005, warmup_steps=1, every=every)
    with WandbTracker('dew', 'scheduled', offline=True) as sink:
        trainer = Trainer(Squared(), OptimConfig(learning_rate=schedule), key=0, tracker=sink, accumulation=2)
        trainer.fit(targets([1.] * 16), steps=16, log_every=1)
    oracle = optax.warmup_cosine_decay_schedule(
        init_value=.01, peak_value=.1, warmup_steps=1, decay_steps=8 // every, end_value=.005)
    rows = [row for row in run.history if 'train/loss' in row]
    assert len(rows) == 16
    for row in rows:
        update = max(row['train/step'] // 2 - 1, 0)
        assert row['train/learning_rate'] == pytest.approx(float(oracle(update // every)))
