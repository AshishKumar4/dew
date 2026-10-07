"""Shortcut models against kvfrans/shortcut-models' `get_targets` and the
loss its `update` takes (`tools/shortcut_reference.py`), and the run config
around them."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from diffusion_stubs import label_table
from flax import linen as nn
from reference_error import assert_as_exact_as_the_reference

from dew.diffusion import presets
from dew.inputs import Condition, Field, InputSpec
from dew.objectives.base import Step
from dew.objectives.diffusion.few_step import ShortcutObjective

CASE = np.load(Path(__file__).resolve().parent / "fixtures" / "shortcut" / "targets.npz")
SETTINGS = json.loads(str(CASE["settings"]))


class Tiny(nn.Module):
    """`tools/shortcut_reference.py`'s velocity in Dew's convention: Dew's
    sigma is 1 - t and its velocity noise minus data, the reference's
    negated; the model times are sigma and the step times 1000, a step of
    2^-level being the reference's level; and the class is the condition's
    second token, whose table entry is the class."""

    interval = True

    @nn.compact
    def __call__(self, x, time, textcontext, duration=None, train=False):
        def column(value):
            return value.reshape(-1, 1, 1, 1)

        # The objective initializes the model without a duration.
        duration = 1000 * jnp.ones_like(time) if duration is None else duration
        t, level, y = 1 - time / 1000, jnp.round(jnp.log2(1000 / duration)), textcontext.hidden[:, 1, 0]
        weights = self.param("weights", nn.initializers.zeros, (4, *x.shape[1:]))
        return -(jnp.tanh(x) * weights[0] + jnp.sin(2 * column(t)) * x * weights[1]
                 + column(level) * jnp.cos(x) * weights[2] + column(y) * weights[3])


def test_the_loss_and_its_gradient_are_the_references(monkeypatch):
    """`ShortcutObjective.loss` on the reference's own draws: the
    self-consistency rows' levels and dyadic times, their targets two half
    steps of the EMA weights, the flow-matching rows on the finest grid with
    their condition dropped, the path whose noise keeps 1e-5 at the data
    end, and the mean squared error. The loss within 1e-6 of the reference's
    float64 run and the gradient in the network's 192 weights held to it by
    the float64 rule."""
    count = SETTINGS["batch_size"]
    rows = count // SETTINGS["bootstrap_every"]
    table = label_table(range(SETTINGS["num_classes"]), null=SETTINGS["num_classes"])
    inputs = InputSpec(Field("image", CASE["pixels"].shape[1:]), {"textcontext": Condition(table)})
    task = ShortcutObjective(Tiny(), presets.Shortcut()(), inputs, sections=SETTINGS["denoise_timesteps"],
                             bootstrap_every=SETTINGS["bootstrap_every"],
                             unconditional_prob=SETTINGS["class_dropout_prob"])
    # The reference draws its self-consistency rows and its flow rows apart
    # and keeps the first of the flow draws, and its flow rows take the
    # batch's first images again, as its self-consistency rows do; here each
    # row holds the image and draws the reference gave it.
    images = np.concatenate([np.arange(rows), np.arange(count - rows)])
    drawn = (jnp.concatenate([CASE["bootstrap_times"], CASE["flow_times"][:count - rows]]),
             jnp.concatenate([CASE["bootstrap_noise"], CASE["flow_noise"][:count - rows]]),
             # Self-consistency rows keep their condition whatever they draw.
             jnp.concatenate([jnp.ones(rows, bool), CASE["dropout"][:count - rows]]))
    monkeypatch.setattr(ShortcutObjective, "_draws", lambda self, key, count, grid, shape: drawn)
    variables = task.init(jax.random.PRNGKey(0))
    teacher = {**variables, "params": {"weights": jnp.asarray(CASE["teacher"])}}
    batch = {"image": CASE["pixels"][images],
             **inputs.tokenize([str(int(label)) for label in CASE["classes"][images]])}
    step = Step(step=jnp.asarray(0), key=jax.random.PRNGKey(1), ema=teacher)

    def loss(weights):
        return task.scalar_loss({**variables, "params": {"weights": weights}}, batch, step)[0]

    value, gradient = jax.value_and_grad(loss)(jnp.asarray(CASE["weights"]))
    np.testing.assert_allclose(float(value), float(CASE["loss_f64"]), rtol=1e-6)
    assert_as_exact_as_the_reference(gradient, CASE["grad"], CASE["grad_f64"], "gradient")


def test_a_run_config_trains_a_shortcut_model_on_its_own_targets():
    from diffusion_stubs import batch_for

    from dew.config import ModelConfig, ObjectiveConfig
    from dew.data import TFDSImages
    from dew.objectives.diffusion import DiffusionRunConfig, TextCondition
    from dew.sampling import Euler
    from dew.training import Trainer

    config = DiffusionRunConfig(
        model=ModelConfig(
            "simple_dit",
            {"patch_size": 2, "emb_features": 16, "num_layers": 1, "num_heads": 2,
             "dtype": "float32", "attention_impl": "xla"},
        ),
        data=TFDSImages(image_size=4),
        preset=presets.Shortcut(),
        val_metrics=(),
        text=TextCondition(encoder="char_table", checkpoint="char_table"),
        objective=ObjectiveConfig("shortcut",
                                  {"sections": 4, "bootstrap_every": 2, "solver": Euler(), "steps": 3}),
    )
    task = config.build()
    assert isinstance(task, ShortcutObjective) and task.model.interval
    trainer = Trainer(task, optax.adam(1e-2), key=jax.random.PRNGKey(3))
    state = trainer.initial_state()
    batch = batch_for(task, 4)
    step = trainer.compile(state, batch)
    for _ in range(3):
        state, *_ = step(state, batch)
    images = task.pipeline(state)(["a red bird"], key=9).host().images
    assert np.all(np.isfinite(images))
