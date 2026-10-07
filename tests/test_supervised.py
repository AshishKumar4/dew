"""`Supervised` trains any Flax model on a loss over its outputs, and a run
names it, its loss and its metrics in a record that builds it again;
`--set` changes one field of a run, read as that field's type reads it."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from affine_run import Affine, Data
from flax import linen as nn

from dew.config import ModelConfig, ObjectiveConfig, RunConfig, TrainerConfig
from dew.config.sweep import assigned
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
        "class": "dew.objectives.supervised:CrossEntropy", "fields": {"labels": "label"}}


def test_a_lambda_loss_is_refused_where_the_run_is_saved(tmp_path):
    run = supervised_run(lambda outputs, batch: squared(outputs, batch))
    with pytest.raises(ValueError, match="define it at module level"):
        run.save(str(tmp_path))


def test_a_run_stating_another_objectives_arguments_refuses_to_train():
    run = RunConfig(objective=ObjectiveConfig("lm", {"ema_decay": 0.9}), trainer=TrainerConfig(batch_size=8))
    with pytest.raises(ValueError, match=r"states the arguments of .*LMObjective"):
        run.train(Supervised(Affine(), squared, inputs=INPUTS), Data(), name="mismatched")


def test_set_reads_each_value_as_its_field_reads_a_flag():
    run = assigned(supervised_run(CrossEntropy()), [
        "trainer.steps=2000", "optim.learning_rate=1e-3", "data.image_size=96", "trainer.eval_every=None",
        'objective.loss={"class": "dew.objectives.supervised:CrossEntropy", "fields": {"labels": "y"}}'])
    assert run.trainer.steps == 2000 and run.optim.learning_rate == 0.001
    assert run.data.image_size == 96 and run.trainer.eval_every is None
    assert run.objective.build(model=run.model.build()).criterion == CrossEntropy(labels="y")

    lm = RunConfig(model=ModelConfig("causal_transformer", {"vocab_size": 8}),
                   objective=ObjectiveConfig("lm"))
    lm = assigned(lm, ["model.num_layers=12", "model.dtype=bfloat16", "objective.ema_decay=0.99"])
    assert lm.model.fields == {"vocab_size": 8, "num_layers": 12, "dtype": "bfloat16"}
    assert lm.objective.fields == {"ema_decay": 0.99}


def test_set_refuses_a_path_the_run_does_not_declare_and_a_value_its_type_does_not_read():
    run = supervised_run(CrossEntropy())
    with pytest.raises(KeyError, match="names no field"):
        assigned(run, ["trainer.stepz=3"])
    with pytest.raises(KeyError, match="names no argument"):
        assigned(run, ["objective.los=3"])
    with pytest.raises(ValueError, match="sets no value"):
        assigned(run, ["trainer.steps"])
    with pytest.raises(SystemExit):
        assigned(run, ["trainer.steps=many"])


def test_a_model_whose_call_needs_more_than_the_sample_is_refused_by_name():
    """Supervised feeds the model one input, so a call that also needs a time
    (a denoiser's) is refused naming what it needs, before anything traces."""
    class Timed(nn.Module):
        @nn.compact
        def __call__(self, sample, time, *, train=False):
            return nn.Dense(1)(sample) * time[:, None]

    with pytest.raises(TypeError, match="sample alone, and Timed's call also needs time"):
        Supervised(Timed(), squared, inputs=INPUTS)
