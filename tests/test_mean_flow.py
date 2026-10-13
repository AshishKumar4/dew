"""MeanFlow against Gsunshine/meanflow's own `forward`
(`tools/meanflow_reference.py`), and the objective and interval sampling
built around it."""


import jax
import jax.numpy as jnp
import numpy as np

from dew.diffusion import presets
from dew.inputs import CharTable, Condition, Field, InputSpec
from dew.nn.backbones import SimpleDiT
from dew.objectives.base import Step
from dew.objectives.diffusion.few_step import MeanFlowObjective
from dew.sampling import Euler, sample

CLASSES = 10


def objective(**fields):
    model = SimpleDiT(patch_size=2, emb_features=16, num_layers=1, num_heads=2, mlp_ratio=1, interval=True)
    inputs = InputSpec(
        Field("image", (4, 4, 3)), {"textcontext": Condition(CharTable.from_pretrained("char_table"))}
    )
    return MeanFlowObjective(model, presets.MeanFlow()(), inputs, **fields, ema_decay=None)


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


def test_a_run_config_trains_meanflow_and_its_saved_task_samples_in_one_step(tmp_path):
    import optax
    from diffusion_stubs import batch_for

    from dew.checkpoints import Checkpoints
    from dew.config import ModelConfig, ObjectiveConfig, TrainerConfig
    from dew.config.data import ImageData
    from dew.objectives.diffusion import DiffusionRunConfig, TextCondition
    from dew.sampling import TextToImage
    from dew.training import Trainer

    config = DiffusionRunConfig(
        model=ModelConfig(
            "simple_dit",
            {"patch_size": 2, "emb_features": 16, "num_layers": 1, "num_heads": 2,
             "dtype": "float32", "attention_impl": "xla"},
        ),
        data=ImageData(source="tfds://oxford_flowers102", image_size=4),
        preset=presets.MeanFlow(),
        val_metrics=(),
        trainer=TrainerConfig(checkpoint_dir=str(tmp_path)),
        text=TextCondition(encoder="char_table", checkpoint="char_table"),
        objective=ObjectiveConfig("mean_flow",
                                  {"omega": 2.0, "kappa": 0.5, "solver": Euler(), "steps": 2,
                                   "ema_decay": None}),
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
