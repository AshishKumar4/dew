"""`Supervised` trains any Flax model on a loss over its outputs, and a run
names it, its loss and its metrics in a record that builds it again;
`--set` changes one field of a run, read as that field's type reads it."""

import json
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from affine_run import Affine, Data
from flax import linen as nn

from dew.config import ModelConfig, ObjectiveConfig, RunConfig, TrainerConfig
from dew.inputs import Field, InputSpec
from dew.objectives.base import VALID_ROWS, Step
from dew.objectives.supervised import Accuracy, CrossEntropy, Supervised
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
    state = trainer.fit(Data(), steps=60, log_every=60)
    before, _ = objective.scalar_loss(start.variables, batch(), STEP)
    after, _ = objective.scalar_loss(state.variables, batch(), STEP)
    assert after < before / 4


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


def test_a_loss_that_returns_its_own_mean_is_refused():
    objective = Supervised(Affine(), lambda outputs, batch: jnp.mean(squared(outputs, batch)), inputs=INPUTS)
    with pytest.raises(ValueError, match="one loss per example"):
        objective.scalar_loss(objective.init(jax.random.key(0)), batch(), STEP)


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


def test_a_lambda_loss_is_refused_where_the_run_states_it():
    with pytest.raises(ValueError, match="define it at module level"):
        supervised_run(lambda outputs, batch: squared(outputs, batch))


def test_a_run_stating_another_objectives_arguments_refuses_to_train():
    run = RunConfig(objective=ObjectiveConfig("lm", {"ema_decay": 0.9}), trainer=TrainerConfig(batch_size=8))
    with pytest.raises(ValueError, match=r"states the arguments of .*LMObjective"):
        run.train(Supervised(Affine(), squared, inputs=INPUTS), Data(), name="mismatched")


def test_set_reads_each_value_as_its_field_reads_a_flag():
    run = supervised_run(CrossEntropy()).assigned([
        "trainer.steps=2000", "optim.learning_rate=1e-3", "data.image_size=96", "trainer.eval_every=None",
        'objective.loss={"class": "dew.objectives.supervised:CrossEntropy", "fields": {"labels": "y"}}'])
    assert run.trainer.steps == 2000 and run.optim.learning_rate == 0.001
    assert run.data.image_size == 96 and run.trainer.eval_every is None
    assert run.objective.build(model=run.model.build()).criterion == CrossEntropy(labels="y")

    lm = RunConfig(model=ModelConfig("causal_transformer", {"vocab_size": 8}),
                   objective=ObjectiveConfig("lm"))
    lm = lm.assigned(["model.num_layers=12", "model.dtype=bfloat16", "objective.ema_decay=0.99"])
    assert lm.model.fields == {"vocab_size": 8, "num_layers": 12, "dtype": "bfloat16"}
    assert lm.objective.fields == {"ema_decay": 0.99}


def test_set_refuses_a_path_the_run_does_not_declare_and_a_value_its_type_does_not_read():
    run = supervised_run(CrossEntropy())
    with pytest.raises(KeyError, match="names no field"):
        run.assigned(["trainer.stepz=3"])
    with pytest.raises(KeyError, match="names no argument"):
        run.assigned(["objective.los=3"])
    with pytest.raises(ValueError, match="sets no value"):
        run.assigned(["trainer.steps"])
    with pytest.raises(SystemExit):
        run.assigned(["trainer.steps=many"])


def test_a_model_whose_call_needs_more_than_the_sample_is_refused_by_name():
    """Supervised feeds the model one input, so a call that also needs a time
    (a denoiser's) is refused naming what it needs, before anything traces."""
    class Timed(nn.Module):
        @nn.compact
        def __call__(self, sample, time, *, train=False):
            return nn.Dense(1)(sample) * time[:, None]

    with pytest.raises(TypeError, match="sample alone, and Timed's call also needs time"):
        Supervised(Timed(), squared, inputs=INPUTS)


class Heads(nn.Module):
    """Logits beside the features they are read from."""

    @nn.compact
    def __call__(self, x):
        features = nn.Dense(4)(x)
        return {"logits": nn.Dense(2)(features), "features": features}


def test_criteria_read_the_output_their_path_selects_from_a_model_with_several():
    objective = Supervised(Heads(), CrossEntropy(output=("logits",)), (Accuracy(output=("logits",)),),
                           inputs=INPUTS)
    variables = objective.init(jax.random.key(0))
    _, aux = objective.scalar_loss(variables, batch(), STEP)
    logits = Heads().apply(variables, batch()["x"])["logits"]
    np.testing.assert_allclose(aux.metrics["accuracy"], np.mean(np.argmax(logits, -1) == batch()["label"]),
                               rtol=1e-6)
    with pytest.raises(TypeError, match="selects a dict"):
        Supervised(Heads(), CrossEntropy(), inputs=INPUTS).scalar_loss(variables, batch(), STEP)


def test_an_autoencoder_with_two_outputs_trains_and_its_record_trains_it_again_in_a_new_process(tmp_path):
    """A model returning its reconstruction and its latent trains through
    `dew train` on a loss reading both; a new process given only the run's
    `run.json` rebuilds the same run and trains the same loss, step for step."""
    from test_examples import REPO_ROOT, single_device

    (tmp_path / "tmp").mkdir()
    environment = {**single_device(), "TMPDIR": str(tmp_path / "tmp"),
                   "PYTHONPATH": os.pathsep.join([str(REPO_ROOT / "src"), str(REPO_ROOT / "tests")])}

    def train(*arguments):
        finished = subprocess.run([sys.executable, "-m", "dew.cli.main", "train", *arguments],
                                  cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=900)
        assert finished.returncode == 0, (
            f"--- stdout ---\n{finished.stdout}\n--- stderr ---\n{finished.stderr}")

    def losses(directory):
        rows = (directory / "autoencoder_experiment" / "tracking" / "scalars.jsonl").read_text().splitlines()
        scalars = [json.loads(row)["scalars"] for row in rows]
        return [row["train/loss"] for row in scalars if "train/loss" in row]

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
    state = trainer.fit(Data(), steps=4, log_every=4, checkpoint_every=4)
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


class Moded(nn.Module):
    """A layer whose BatchNorm and Dropout read the mode its call is given."""

    @nn.compact
    def __call__(self, x, train=False):
        x = nn.BatchNorm(use_running_average=not train, momentum=0.5)(x.astype(jnp.float32))
        return nn.Dropout(0.5)(x, deterministic=not train)


def test_a_model_with_a_training_mode_trains_in_it_and_is_scored_out_of_it():
    """With `mode`, the step calls the model in training mode, so its BatchNorm
    moves toward the batch and its Dropout draws; a validation pass calls it in
    evaluation mode, on the running statistics, the same whatever the key, and
    writes nothing. Without it the model keeps its call's default."""
    moded = Supervised(Moded(), squared, inputs=INPUTS, mode="train")
    variables = moded.init(jax.random.key(0))
    batch = {"x": jnp.full((4, 3), 5.0)}
    _, aux = moded.loss(variables, batch, Step(jnp.int32(0), jax.random.key(1), None))
    np.testing.assert_allclose(aux.variables["batch_stats"]["BatchNorm_0"]["mean"], 2.5)
    first, second = (moded.validation_loss(variables, batch, Step(jnp.int32(0), jax.random.key(seed), None))
                     for seed in (1, 2))
    assert first[0].mean()[0] == second[0].mean()[0] and first[1].variables is None
    trained, _ = moded.loss(variables, batch, Step(jnp.int32(0), jax.random.key(1), None))
    assert trained.mean()[0] != first[0].mean()[0]
    _, default = Supervised(Moded(), squared, inputs=INPUTS).loss(
        variables, batch, Step(jnp.int32(0), jax.random.key(1), None))
    np.testing.assert_array_equal(default.variables["batch_stats"]["BatchNorm_0"]["mean"], 0.0)
    with pytest.raises(TypeError, match="does not take"):
        Supervised(Moded(), squared, inputs=INPUTS, mode="training")
