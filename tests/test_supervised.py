"""`Supervised` trains any Flax model on a loss over its outputs, and a run
names it, its loss and its metrics in a record that builds it again;
`--set` changes one field of a run, read as that field's type reads it."""

import json
import os
import subprocess
import sys
from functools import wraps

import jax
import jax.numpy as jnp
import numpy as np
import optax
from affine_run import Affine, Replaying, dataset
from flax import linen as nn
from learning_gate import assert_learns

from dew.batch import VALID_ROWS
from dew.config import ObjectiveConfig, RunConfig, TrainerConfig
from dew.inputs import Field, InputSpec
from dew.objectives.base import Step
from dew.objectives.supervised import Accuracy, CrossEntropy, Supervised
from dew.registry import ModelConfig
from dew.training import Trainer

INPUTS = InputSpec(Field("x", (3,)))
STEP = Step(step=jnp.asarray(0), key=jax.random.key(1), ema=None)


def squared(outputs, batch):
    """Each target entry's squared error."""
    return (outputs - batch["y"]) ** 2


def batch(rows=8, seed=0):
    x = np.random.default_rng(seed).normal(size=(rows, 3)).astype(np.float32)
    return {"x": x, "y": 2 * x[:, :2], "label": (x[:, 0] > 0).astype(np.int32)}


RUNS = 8
"""The independent training runs a learning claim here is judged over (`assert_learns`)."""


def learned(objective, optimizer, loader_of, held_out_of, steps: int, margin: float) -> np.ndarray:
    """Each of RUNS independent runs' contrast for "training lowers the held-out loss by `margin`": the
    loss before over `margin` less the loss after, on the run's own held-out batch, each run from its
    own initialization key and data."""
    contrasts = []
    for run in range(RUNS):
        trainer = Trainer(objective, optimizer, key=jax.random.key(run))
        start = trainer.initial_state()
        state = trainer.fit(loader_of(run), steps=steps, log_every=steps)
        held_out = held_out_of(run)
        before, _ = objective.scalar_loss(start.variables, held_out, STEP)
        after, _ = objective.scalar_loss(state.variables, held_out, STEP)
        contrasts.append(float(before) / margin - float(after))
    return np.asarray(contrasts)


def test_a_plain_flax_model_trains_on_a_loss_function():
    """Over independent runs, each from its own key on its own data, sixty steps of Adam cut the affine
    model's held-out squared error to under a quarter (`assert_learns`)."""
    objective = Supervised(Affine(), squared, inputs=INPUTS)
    drop = learned(objective, optax.adam(0.1),
                   lambda run: dataset(train=lambda run=run: Replaying(lambda step: batch(seed=[run, step])),
                                       batch=8),
                   lambda run: batch(seed=[run, 10_000]), steps=60, margin=4)
    assert_learns(lambda runs: {"training cuts the held-out loss to a quarter": drop[runs]},
                  "a plain flax model", units=RUNS)


def finite_inputs(call):
    """Supply the wrapped model's finite-input mask from its sample."""
    @wraps(call)
    def apply(self, x):
        return call(self, x, jnp.isfinite(x))
    return apply


class MaskedAffine(nn.Module):
    """An affine map whose call decorator supplies its mask."""

    @nn.compact
    @finite_inputs
    def __call__(self, x, mask):
        return nn.Dense(2)(jnp.where(mask, x, 0).astype(jnp.float32))


def test_a_decorated_flax_call_trains_through_its_actual_calling_convention():
    model = MaskedAffine()
    sample = batch()
    variables = model.init(jax.random.key(0), sample["x"])
    assert model.apply(variables, sample["x"]).shape == sample["y"].shape
    objective = Supervised(model, squared, inputs=INPUTS)
    trainer = Trainer(objective, optax.sgd(.05), key=jax.random.key(0))
    initial = trainer.initial_state()
    before = float(objective.scalar_loss(initial.variables, sample, STEP)[0])
    trained, _, _, _, accepted, _ = trainer.compile(initial, sample)(initial, sample)
    assert bool(accepted) and int(trained.updates) == 1
    assert float(objective.scalar_loss(trained.variables, sample, STEP)[0]) < before


def test_the_loss_is_the_mean_over_the_real_rows_and_metrics_report_by_name():
    """A repeated evaluation row counts for nothing, so the loss over a padded
    batch is the loss over its real rows; each metric reports its own mean
    under its function's or its class's name."""
    objective = Supervised(Affine(), CrossEntropy(), (Accuracy(), squared), inputs=INPUTS)
    variables = objective.init(jax.random.key(0))
    real = batch(6)
    padded = {name: np.concatenate([value, value[:2]]) for name, value in real.items()}
    padded[VALID_ROWS] = np.arange(8) < 6
    alone, aux = objective.scalar_loss(variables, real, STEP)
    np.testing.assert_allclose(objective.scalar_loss(variables, padded, STEP)[0], alone, rtol=1e-6)
    outputs = Affine().apply(variables, real["x"])
    np.testing.assert_allclose(aux.metrics["accuracy"],
                               np.mean(np.argmax(outputs, -1) == real["label"]), rtol=1e-6)
    np.testing.assert_allclose(aux.metrics["squared"], np.mean(squared(outputs, real)), rtol=1e-6)


def supervised_run(loss) -> RunConfig:
    return RunConfig(model=ModelConfig.from_model(Affine()), trainer=TrainerConfig(steps=1),
                     objective=ObjectiveConfig("supervised", {"loss": loss, "metrics": (Accuracy(),),
                                                              "inputs": INPUTS}))


def test_the_run_record_builds_the_same_objective_again():
    run = supervised_run(CrossEntropy(labels="label"))
    record = json.loads(json.dumps(run.to_dict()))
    rebuilt = RunConfig.from_dict(record)
    original, again = (config.objective.build(model=config.model.build()) for config in (run, rebuilt))
    variables = original.init(jax.random.key(0))
    np.testing.assert_array_equal(again.scalar_loss(variables, batch(), STEP)[0],
                                  original.scalar_loss(variables, batch(), STEP)[0])
    assert record["objective"]["fields"]["loss"] == {
        "class": "dew.objectives.supervised:CrossEntropy", "fields": {"labels": "label", "output": []}}


def test_an_autoencoder_with_two_outputs_trains_and_its_record_trains_it_again_in_a_new_process(tmp_path):
    """A model returning its reconstruction and its latent trains through
    `dew train` on a loss reading both; a new process given only the run's
    `run.json` rebuilds the same run and trains the same loss, step for step."""
    from process_support import dew_train, logged_losses
    from test_examples import REPO_ROOT, single_device

    (tmp_path / "tmp").mkdir()
    environment = {**single_device(), "TMPDIR": str(tmp_path / "tmp"),
                   "PYTHONPATH": os.pathsep.join([str(REPO_ROOT / "src"), str(REPO_ROOT / "tests")])}

    def train(*arguments):
        dew_train(tmp_path, environment, *arguments)

    def losses(directory):
        return logged_losses(directory / "autoencoder_experiment")

    train(str(REPO_ROOT / "tests" / "autoencoder_experiment.py"),
          "--set", f"trainer.checkpoint_dir={tmp_path / 'first'}")
    record = tmp_path / "first" / "autoencoder_experiment" / "run.json"
    loss = json.loads(record.read_text())["fields"]["objective"]["fields"]["loss"]
    assert loss == {"function": "autoencoder_experiment:evidence_bound"}
    trained = losses(tmp_path / "first")
    # That the run learns is held over independent runs in process
    # (`test_the_autoencoders_evidence_bound_falls_over_independent_runs`).
    assert len(trained) == 40

    train(str(record), "--trust", "autoencoder_experiment",
          "--set", f"trainer.checkpoint_dir={tmp_path / 'again'}")
    assert losses(tmp_path / "again") == trained


def test_the_autoencoders_evidence_bound_falls_over_independent_runs():
    """The two-output experiment's objective (tests/autoencoder_experiment.py) trained forty steps at
    its learning rate halves its held-out evidence bound over independent runs (`assert_learns`), each
    run from its own key on its own points of the experiment's plane."""
    from autoencoder_experiment import FEATURES, Autoencoder, evidence_bound

    plane = np.random.default_rng(0).normal(size=(2, FEATURES)) / np.sqrt(2)

    def points(seed, rows=32):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(rows, 2)) @ plane + rng.normal(0, 0.05, (rows, FEATURES))
        return {"x": x.astype(np.float32)}

    objective = Supervised(Autoencoder(), evidence_bound, inputs=InputSpec(Field("x", (FEATURES,))))
    drop = learned(objective, optax.adam(1e-2),
                   lambda run: dataset(train=lambda run=run: Replaying(lambda step: points([run, step])),
                                       batch=32),
                   lambda run: points([run, 10_000], rows=64), steps=40, margin=2)
    assert_learns(lambda runs: {"training halves the held-out evidence bound": drop[runs]},
                  "the autoencoder experiment", units=RUNS)


class Normed(nn.Module):
    """A layer over a BatchNorm that normalizes by the batch it reads."""

    @nn.compact
    def __call__(self, x):
        return nn.Dense(2)(nn.BatchNorm(use_running_average=False, momentum=0.5)(x.astype(jnp.float32)))


RESTORED = """
import json, sys
import jax, numpy as np
from dew.checkpoints import Checkpoints
held = Checkpoints(sys.argv[1]).variables(ema=None, step=None, mesh=None, layout=None, param_dtype=None)
print(json.dumps({jax.tree_util.keystr(path): np.asarray(leaf).tobytes().hex()
                  for path, leaf in jax.tree_util.tree_leaves_with_path(held["batch_stats"])}))
"""


def test_batch_statistics_update_as_the_model_trains_and_reload_from_its_checkpoint(tmp_path):
    """A BatchNorm's running statistics move with every step, and a new
    process restores the ones the run ended on, bit for bit."""
    from test_examples import REPO_ROOT

    from dew.checkpoints import Checkpoints

    checkpoints = Checkpoints(str(tmp_path / "run"))
    trainer = Trainer(Supervised(Normed(), squared, inputs=INPUTS), optax.adam(0.05),
                      key=jax.random.key(0), checkpoints=checkpoints)
    # Copied to the host: a step may donate the buffers it starts from.
    start = jax.tree.map(np.asarray, trainer.initial_state().variables["batch_stats"])
    state = trainer.fit(dataset(), steps=4, log_every=4, checkpoint_every=4)
    checkpoints.wait()
    trained = state.variables["batch_stats"]
    for began, ended in zip(jax.tree.leaves(start), jax.tree.leaves(trained), strict=True):
        assert not np.array_equal(np.asarray(began), np.asarray(ended))
    restored = subprocess.run([sys.executable, "-c", RESTORED, str(tmp_path / "run")],
                              capture_output=True, text=True, timeout=300, check=False,
                              env={**os.environ, "JAX_PLATFORMS": "cpu",
                                   "PYTHONPATH": str(REPO_ROOT / "src")})
    assert restored.returncode == 0, restored.stderr[-2000:]
    assert json.loads(restored.stdout.splitlines()[-1]) == {
        jax.tree_util.keystr(path): np.asarray(leaf).tobytes().hex()
        for path, leaf in jax.tree_util.tree_leaves_with_path(trained)}
