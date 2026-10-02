"""MeanFlow against Gsunshine/meanflow's own `forward`
(`tools/meanflow_reference.py`), and the objective and interval sampling
built around it."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.diffusion import presets
from dew.inputs import CharTable, Condition, Field, InputSpec
from dew.nn.backbones import SimpleDiT
from dew.objectives.base import Step, scalar_loss
from dew.objectives.diffusion.few_step import (
    MeanFlowObjective,
    adaptive_loss,
    guided_velocity,
    intervals,
    mean_flow_target,
)
from dew.sampling import Euler, sample

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "meanflow"


def tiny(x, t, h, y):
    """`tools/meanflow_reference.py`'s closed-form average velocity."""
    def column(value):
        return value.reshape(-1, 1, 1, 1)

    return (jnp.tanh(x) * 0.6 + jnp.sin(3 * column(t)) * x * 0.3
            + column(h) * jnp.cos(x) * 0.2 + column(y.astype(jnp.float32)) * 0.05)


@pytest.mark.parametrize("power", ["0", "1"])
def test_the_loss_is_the_references(power):
    """On the reference's own draws: the guided velocity inside and outside
    the guidance interval, and the loss over the dropped condition, the JVP
    target and the adaptive weight."""
    case = np.load(FIXTURES / f"loss_p{power}.npz")
    settings = json.loads(str(case["settings"]))
    t, r = jnp.asarray(case["t"]).ravel(), jnp.asarray(case["r"]).ravel()
    z, v = jnp.asarray(case["z"]), jnp.asarray(case["v"])
    classes = jnp.asarray(case["classes"])
    inside = ((t >= settings["t_start"]) & (t <= settings["t_end"])).reshape(-1, 1, 1, 1)
    null = jnp.full_like(classes, settings["num_classes"])
    guided = guided_velocity(v, tiny(z, t, 0 * t, null), tiny(z, t, 0 * t, classes),
                             jnp.where(inside, settings["omega"], 1.0), jnp.where(inside, settings["kappa"], 0.0))
    # Both sides run the same float32 JAX operations in the same order, up
    # to the association of the three-term sum: a few ulps.
    np.testing.assert_allclose(np.asarray(guided), case["guided"], rtol=1e-5, atol=1e-6)

    labels = jnp.asarray(case["labels"])
    u, target = mean_flow_target(lambda z, t, r: tiny(z, t, t - r, labels), z, t, r, jnp.asarray(case["dropped"]))
    loss = jnp.mean(adaptive_loss(u, target, settings["norm_p"], settings["norm_eps"]))
    np.testing.assert_allclose(float(loss), float(case["loss"]), rtol=1e-5)


def test_the_first_fraction_of_rows_is_instantaneous():
    t, r = intervals(jnp.asarray([0.2, 0.9, 0.5, 0.1]), jnp.asarray([0.6, 0.3, 0.4, 0.7]), 0.5)
    np.testing.assert_array_equal(np.asarray(t), np.float32([0.6, 0.9, 0.5, 0.7]))
    np.testing.assert_array_equal(np.asarray(r), np.float32([0.6, 0.9, 0.4, 0.1]))


def objective(**fields):
    model = SimpleDiT(patch_size=2, emb_features=16, num_layers=1, num_heads=2, mlp_ratio=1, interval=True)
    inputs = InputSpec(Field("image", (4, 4, 3)), {"textcontext": Condition(CharTable.from_pretrained("char_table"))})
    return MeanFlowObjective(model, presets.MeanFlow()(), inputs, ema_decay=None, **fields)


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
    grads = jax.grad(lambda tree: scalar_loss(task, {**params, "params": tree}, batch, step)[0])(params["params"])
    duration = grads["conditioning"]["duration_embed"]
    assert float(sum(jnp.abs(leaf).sum() for leaf in jax.tree.leaves(duration))) > 0

    given, _ = task._conditions(params, batch, jax.random.PRNGKey(0), dropout=False)
    denoise = task.denoiser(params, given, None)
    x_T = jax.random.normal(jax.random.PRNGKey(3), (4, 4, 4, 3))
    walked = sample(denoise, x_T, 2, solver=Euler(), key=jax.random.PRNGKey(0), final_denoise=False)
    schedule = task.process.schedule
    u = task.model.apply(task.trainable(params), x_T, schedule.model_time(jnp.ones((4,))), **given,
                         duration=schedule.model_time(jnp.ones((4,))) - schedule.model_time(jnp.zeros((4,))))
    np.testing.assert_allclose(np.asarray(walked), np.asarray(x_T - u), rtol=1e-5, atol=1e-6)


def test_meanflow_refuses_an_instantaneous_process():
    model = SimpleDiT(patch_size=2, emb_features=16, num_layers=1, num_heads=2, mlp_ratio=1)
    with pytest.raises(ValueError, match=r"presets\.MeanFlow"):
        MeanFlowObjective(model, presets.Flow()(), InputSpec(Field("image", (4, 4, 3))))


def test_a_run_config_trains_meanflow_and_its_saved_task_samples_in_one_step(tmp_path):
    import optax
    from test_diffusion_run_sources import batch_for

    from dew.checkpoints import Checkpoints
    from dew.config import ModelConfig, TrainerConfig
    from dew.data import OxfordFlowers
    from dew.objectives.diffusion import DiffusionRunConfig, MeanFlowTraining, TextCondition
    from dew.sampling import TextToImage
    from dew.training import Trainer

    config = DiffusionRunConfig(
        model=ModelConfig("simple_dit", {"patch_size": 2, "emb_features": 16, "num_layers": 1, "num_heads": 2},
                          dtype="float32", attention_impl="xla"),
        data=OxfordFlowers(image_size=4), preset=presets.MeanFlow(), sampler=Euler(), guidance=None,
        sampling_steps=2, ema_decay=None, val_metrics=(), trainer=TrainerConfig(checkpoint_dir=str(tmp_path)),
        text=TextCondition(encoder="char_table", checkpoint="char_table"),
        mean_flow=MeanFlowTraining(omega=2.0, kappa=0.5))
    task = config.build()
    assert isinstance(task, MeanFlowObjective) and task.model.interval
    trainer = Trainer(task, optax.adam(1e-2), key=jax.random.PRNGKey(3))
    state = trainer.initial_state()
    batch = batch_for(task, 4)
    state, *_ = trainer.compile(state, batch)(state, batch)
    run = tmp_path / "run"
    checkpoints = Checkpoints(str(run))
    checkpoints.save(1, state, None)
    checkpoints.wait()
    config.save(str(run))
    expected = task.pipeline(state, ema=False)(["a red bird"], key=9).host().images
    np.testing.assert_array_equal(TextToImage.from_run(str(run))(["a red bird"], key=9).host().images, expected)


def test_a_meanflow_run_samples_unguided():
    from dew.objectives.diffusion import DiffusionRunConfig, MeanFlowTraining

    with pytest.raises(ValueError, match="set guidance None"):
        DiffusionRunConfig(preset=presets.MeanFlow(), mean_flow=MeanFlowTraining())


@pytest.mark.parametrize("extra", [{"uncertainty": 8}])
def test_meanflow_refuses_the_denoising_losss_extras(extra):
    model = SimpleDiT(patch_size=2, emb_features=16, num_layers=1, num_heads=2, mlp_ratio=1, interval=True)
    with pytest.raises(ValueError, match="own loss"):
        MeanFlowObjective(model, presets.MeanFlow()(), InputSpec(Field("image", (4, 4, 3))), **extra)


def test_the_time_embeddings_take_the_models_time_scale():
    """`time_scale` sets the Fourier frequencies of the time and the duration
    embeddings, the smoothness a loss differentiating in time needs."""
    model = SimpleDiT(patch_size=2, emb_features=16, num_layers=1, num_heads=2, mlp_ratio=1, interval=True,
                             time_scale=0.002)
    variables = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 4, 4, 3)), jnp.ones((1,)))
    table = np.random.RandomState(42).normal(size=(8,)).astype(np.float32) * np.float32(0.002)
    for name in ("time_embed", "duration_embed"):
        frequencies = variables["constants"]["conditioning"][name]["layers_0"]["frequencies"]
        np.testing.assert_allclose(np.asarray(frequencies), table, rtol=1e-6)


def test_a_meanflow_run_config_builds_a_smooth_time_embedding_unless_it_names_one():
    from dew.config import ModelConfig
    from dew.data import OxfordFlowers
    from dew.objectives.diffusion import DiffusionRunConfig, MeanFlowTraining

    def built(config):
        return DiffusionRunConfig(model=ModelConfig("simple_dit", config), data=OxfordFlowers(image_size=8),
                                  preset=presets.MeanFlow(), sampler=Euler(), guidance=None, text=None,
                                  val_metrics=(), mean_flow=MeanFlowTraining()).build().model

    assert built({"patch_size": 2}).time_scale == 0.002
    assert built({"patch_size": 2, "time_scale": 16}).time_scale == 16
