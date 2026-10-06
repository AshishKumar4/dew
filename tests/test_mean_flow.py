"""MeanFlow against Gsunshine/meanflow's own `forward`
(`tools/meanflow_reference.py`), and the objective and interval sampling
built around it."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from reference_error import assert_as_exact_as_the_reference

from dew.diffusion import presets
from dew.inputs import CharTable, Condition, Field, InputSpec
from dew.nn.backbones import SimpleDiT
from dew.objectives.base import Step
from dew.objectives.diffusion.few_step import MeanFlowObjective, MeanFlowTraining
from dew.sampling import Euler, sample

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "meanflow"
CLASSES = 10


class Tiny(nn.Module):
    """`tools/meanflow_reference.py`'s average velocity u(x, t, h, y) in
    Dew's calling convention: the model times are the flow time and the
    duration times 1000, and the class is the condition's second token,
    whose table entry is the class, the blank prompt's padding reading the
    reference's null class."""

    @nn.compact
    def __call__(self, x, time, textcontext, duration=None, train=False):
        def column(value):
            return value.reshape(-1, 1, 1, 1)

        # The objective initializes the model without a duration.
        duration = jnp.zeros_like(time) if duration is None else duration
        t, h, y = time / 1000, duration / 1000, textcontext.hidden[:, 1, 0]
        weights = self.param("weights", nn.initializers.zeros, (4, *x.shape[1:]))
        return (jnp.tanh(x) * weights[0] + jnp.sin(3 * column(t)) * x * weights[1]
                + column(h) * jnp.cos(x) * weights[2] + column(y) * weights[3])


def labelled() -> CharTable:
    """Character tables of one feature: the digit k reads k, and the padding
    id 0 reads the null class."""
    table = CharTable.from_pretrained(tokens=2, features=1)
    entries = np.zeros((table.vocab, 1), np.float32)
    entries[0] = CLASSES
    for digit in range(CLASSES):
        entries[table.tokenize([str(digit)])["input_ids"][0, 1]] = digit
    return CharTable.from_pretrained(tokens=2, features=1, params={"table": jnp.asarray(entries)})


@pytest.mark.parametrize("power", ["0", "1"])
def test_the_loss_and_its_gradient_are_the_references(power, monkeypatch):
    """`MeanFlowObjective.loss` on Gsunshine/meanflow's `forward`'s own
    draws, two times, the noise and the dropout uniforms: the interval with
    its instantaneous rows, the guided velocity inside and outside the
    guidance interval, the condition dropped on the first rows as many as the
    uniforms under the rate count, the JVP target and the adaptive weight.
    The loss within 1e-6 of the reference's float64 run and the gradient in
    the network's 192 weights held to it by the float64 rule."""
    case = np.load(FIXTURES / f"loss_p{power}.npz")
    settings = json.loads(str(case["settings"]))
    assert settings["num_classes"] == CLASSES
    inputs = InputSpec(Field("image", case["pixels"].shape[1:]), {"textcontext": Condition(labelled())})
    task = MeanFlowObjective(
        Tiny(), presets.MeanFlow()(), inputs, MeanFlowTraining(
            instantaneous=settings["data_proportion"], omega=settings["omega"], kappa=settings["kappa"],
            guidance_interval=(settings["t_start"], settings["t_end"]), norm_p=settings["norm_p"],
            norm_eps=settings["norm_eps"]),
        ema_decay=None, unconditional_prob=settings["class_dropout_prob"])
    drawn = tuple(jnp.asarray(case[name]) for name in ("later", "earlier", "noise", "uniform"))
    monkeypatch.setattr(MeanFlowObjective, "_draws", lambda self, key, count, shape: drawn)
    variables = task.init(jax.random.PRNGKey(0))
    batch = {"image": case["pixels"], **inputs.tokenize([str(int(label)) for label in case["classes"]])}
    step = Step(step=jnp.asarray(0), key=jax.random.PRNGKey(1), ema=None)

    def loss(weights):
        return task.scalar_loss({**variables, "params": {"weights": weights}}, batch, step)[0]

    value, gradient = jax.value_and_grad(loss)(jnp.asarray(case["weights"]))
    np.testing.assert_allclose(float(value), float(case["loss_f64"]), rtol=1e-6)
    assert_as_exact_as_the_reference(gradient, case["grad"], case["grad_f64"], f"gradient at norm_p {power}")


def objective(**fields):
    model = SimpleDiT(patch_size=2, emb_features=16, num_layers=1, num_heads=2, mlp_ratio=1, interval=True)
    inputs = InputSpec(
        Field("image", (4, 4, 3)), {"textcontext": Condition(CharTable.from_pretrained("char_table"))}
    )
    return MeanFlowObjective(model, presets.MeanFlow()(), inputs, MeanFlowTraining(**fields), ema_decay=None)


def test_the_objective_trains_the_duration_and_one_step_samples_the_interval():
    """The loss reaches the duration embedding, and a one-step walk is the
    average velocity over the whole of [0, 1] from the prior."""
    task = objective(omega=2.0, kappa=0.2)
    params = task.init(jax.random.PRNGKey(0))
    # adaLN-Zero's gates and output start at zero, where no conditioning
    # reaches the output; drawn weights let it.
    leaves, tree = jax.tree.flatten(params["params"])
    keys = jax.random.split(jax.random.PRNGKey(4), len(leaves))
    params = {**params, "params": jax.tree.unflatten(tree, [
        0.3 * jax.random.normal(key, leaf.shape) for key, leaf in zip(keys, leaves, strict=True)])}
    batch = {"image": np.asarray(jax.random.randint(jax.random.PRNGKey(1), (4, 4, 4, 3), 0, 256), np.uint8),
             **task.inputs.tokenize(["a", "b", "c", "d"])}
    step = Step(step=jnp.asarray(0), key=jax.random.PRNGKey(2), ema=None)
    grads = jax.grad(lambda tree: task.scalar_loss({**params, "params": tree}, batch, step)[0])(
        params["params"]
    )
    duration = grads["conditioning"]["duration_embed"]
    assert float(sum(jnp.abs(leaf).sum() for leaf in jax.tree.leaves(duration))) > 0

    given, _ = task._conditions(params, batch, jax.random.PRNGKey(0), dropout=False)
    denoise = task.denoiser(params, given, None)
    x_T = jax.random.normal(jax.random.PRNGKey(3), (4, 4, 4, 3))
    walked = sample(denoise, x_T, 2, solver=Euler(), key=jax.random.PRNGKey(0), final_denoise=False)
    schedule = task.process.schedule
    u = task.model.apply(task.model_variables(params), x_T, schedule.model_time(jnp.ones((4,))), **given,
                         duration=schedule.model_time(jnp.ones((4,))) - schedule.model_time(jnp.zeros((4,))))
    np.testing.assert_allclose(np.asarray(walked), np.asarray(x_T - u), rtol=1e-5, atol=1e-6)


def test_meanflow_refuses_an_instantaneous_process():
    model = SimpleDiT(patch_size=2, emb_features=16, num_layers=1, num_heads=2, mlp_ratio=1)
    with pytest.raises(ValueError, match=r"presets\.MeanFlow"):
        MeanFlowObjective(model, presets.Flow()(), InputSpec(Field("image", (4, 4, 3))), MeanFlowTraining())


def test_a_run_config_trains_meanflow_and_its_saved_task_samples_in_one_step(tmp_path):
    import optax
    from diffusion_stubs import batch_for

    from dew.checkpoints import Checkpoints
    from dew.config import ModelConfig, TrainerConfig
    from dew.data import TFDSImages
    from dew.objectives.diffusion import DiffusionRunConfig, MeanFlowTraining, TextCondition
    from dew.sampling import TextToImage
    from dew.training import Trainer

    config = DiffusionRunConfig(
        model=ModelConfig(
            "simple_dit",
            {"patch_size": 2, "emb_features": 16, "num_layers": 1, "num_heads": 2},
            dtype="float32",
            attention_impl="xla",
        ),
        data=TFDSImages(image_size=4),
        preset=presets.MeanFlow(),
        solver=Euler(),
        guidance=None,
        sampling_steps=2,
        ema_decay=None,
        val_metrics=(),
        trainer=TrainerConfig(checkpoint_dir=str(tmp_path)),
        text=TextCondition(encoder="char_table", checkpoint="char_table"),
        mode=MeanFlowTraining(omega=2.0, kappa=0.5),
    )
    task = config.build()
    assert isinstance(task, MeanFlowObjective) and task.model.interval
    trainer = Trainer(task, optax.adam(1e-2), key=jax.random.PRNGKey(3))
    state = trainer.initial_state()
    batch = batch_for(task, 4)
    state, *_ = trainer.compile(state, batch)(state, batch)
    run = tmp_path / "run"
    checkpoints = Checkpoints(str(run))
    checkpoints.save(1, state, None, artifact=task.inference_record())
    checkpoints.wait()
    config.save(str(run))
    expected = task.pipeline(state, ema=False)(["a red bird"], key=9).host().images
    np.testing.assert_array_equal(
        TextToImage.from_run(str(run))(["a red bird"], key=9).host().images, expected
    )


def test_a_meanflow_run_samples_unguided():
    from dew.objectives.diffusion import DiffusionRunConfig, MeanFlowTraining

    with pytest.raises(ValueError, match="set guidance None"):
        DiffusionRunConfig(preset=presets.MeanFlow(), mode=MeanFlowTraining())


@pytest.mark.parametrize("extra", [{"uncertainty": 8}])
def test_meanflow_refuses_the_denoising_losss_extras(extra):
    model = SimpleDiT(patch_size=2, emb_features=16, num_layers=1, num_heads=2, mlp_ratio=1, interval=True)
    with pytest.raises(ValueError, match="own loss"):
        MeanFlowObjective(model, presets.MeanFlow()(), InputSpec(Field("image", (4, 4, 3))),
                          MeanFlowTraining(), **extra)


def test_a_meanflow_run_config_builds_a_smooth_time_embedding_unless_it_names_one():
    from dew.config import ModelConfig
    from dew.data import TFDSImages
    from dew.objectives.diffusion import DiffusionRunConfig, MeanFlowTraining

    def built(config):
        return DiffusionRunConfig(model=ModelConfig("simple_dit", config), data=TFDSImages(image_size=8),
                                  preset=presets.MeanFlow(), solver=Euler(), guidance=None, text=None,
                                  val_metrics=(), mode=MeanFlowTraining()).build().model

    assert built({"patch_size": 2}).time_scale == 0.002 and built({"patch_size": 2}).interval
    assert built({"patch_size": 2, "time_scale": 16}).time_scale == 16
