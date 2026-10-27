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
import pytest
from affine_run import BATCH, Affine, Counting, dataset
from flax import linen as nn
from recording import RecordingTracker

from dew.batch import VALID_ROWS
from dew.config import ObjectiveConfig, RunConfig, TrainerConfig
from dew.data.dataset import Loader
from dew.data.pipeline import DeviceStep
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


def batch(rows=8):
    x = np.random.default_rng(0).normal(size=(rows, 3)).astype(np.float32)
    return {"x": x, "y": 2 * x[:, :2], "label": (x[:, 0] > 0).astype(np.int32)}


def test_a_plain_flax_model_trains_on_a_loss_function():
    objective = Supervised(Affine(), squared, inputs=INPUTS)
    trainer = Trainer(objective, optax.adam(0.1), key=jax.random.key(0))
    start = trainer.initial_state()
    state = trainer.fit(dataset(), steps=60, log_every=60)
    before, _ = objective.scalar_loss(start.variables, batch(), STEP)
    after, _ = objective.scalar_loss(state.variables, batch(), STEP)
    assert after < before / 4


def test_a_model_named_by_its_input_field_alone_takes_the_shape_its_data_gives():
    """`Supervised(model, loss, inputs="x")` names the field alone: the run reads its first batch once,
    before the model initializes, and the objective declares the shape that batch holds, as an explicit
    spec would, and trains as the run given that spec does. A state placed before any run has read the
    data is refused."""
    declared = Trainer(Supervised(Affine(), squared, inputs=INPUTS), optax.sgd(0.1), key=0).fit(
        dataset(), steps=3)
    named = Supervised(Affine(), squared, inputs="x")
    with pytest.raises(ValueError):
        Trainer(named, optax.sgd(0.1), key=0).place()
    state = Trainer(named, optax.sgd(0.1), key=0).fit(dataset(), steps=3)
    assert named.inputs == INPUTS
    for leaf, expected in zip(jax.tree.leaves(state.variables), jax.tree.leaves(declared.variables),
                              strict=True):
        np.testing.assert_array_equal(np.asarray(leaf), np.asarray(expected))


class Cropped(DeviceStep):
    """A device step that keeps the first two of each row's three features."""

    def __call__(self, batch):
        return {**batch, "x": batch["x"][:, :2]}


def test_a_named_input_takes_the_shape_the_loaders_device_steps_give_the_model():
    """A loader whose device steps crop each row's three features to two gives the model rows of two: an
    input named alone is declared at those, as the model reads them after the steps, not as the reader
    delivers them, and the run trains as the run given an explicit spec of two does."""
    def data() -> Loader:
        return Loader(lambda partition: Counting(), None, None, BATCH, device=(Cropped(),))

    two = InputSpec(Field("x", (2,)))
    declared = Trainer(Supervised(Affine(), squared, inputs=two), optax.sgd(0.1), key=0).fit(data(), steps=3)
    named = Supervised(Affine(), squared, inputs="x")
    state = Trainer(named, optax.sgd(0.1), key=0).fit(data(), steps=3)
    assert named.inputs == two
    for leaf, expected in zip(jax.tree.leaves(state.variables), jax.tree.leaves(declared.variables),
                              strict=True):
        np.testing.assert_array_equal(np.asarray(leaf), np.asarray(expected))


def affine_work(batch):
    """Training `Affine` on a batch: a row's forward is 3 x 2 multiply-adds, two FLOPs each, and its
    backward twice that."""
    return 3 * 2 * 3 * 2 * len(batch["x"])


def test_a_model_dew_holds_no_count_for_reports_the_work_its_objective_declares():
    """`Supervised(work=)` declares a batch's analytical training FLOPs for a model Dew holds no count
    for: the run logs it as each step's train/flops, the numerator train/mfu reads where the device's
    peak is known; without it the run logs no count, never the compiled program's."""
    def logged(work) -> list[float | None]:
        tracker = RecordingTracker()
        Trainer(Supervised(Affine(), squared, inputs=INPUTS, work=work), optax.sgd(0.1), key=0,
                tracker=tracker).fit(dataset(), steps=2, log_every=1)
        return [scalars.get("train/flops") for _, scalars in tracker.scalars if "train/loss" in scalars]

    assert logged(affine_work) == [affine_work(batch(BATCH))] * 2
    assert logged(None) == [None, None]


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
    assert len(trained) == 40 and np.mean(trained[-5:]) < 0.5 * np.mean(trained[:5])

    train(str(record), "--trust", "autoencoder_experiment",
          "--set", f"trainer.checkpoint_dir={tmp_path / 'again'}")
    assert losses(tmp_path / "again") == trained


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
