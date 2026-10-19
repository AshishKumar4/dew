"""MeanFlow against Gsunshine/meanflow's own `forward`
(`tools/meanflow_reference.py`), and the objective and interval sampling
built around it."""

import functools
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference, assert_leaves_as_exact, widened

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

    given, _ = task._conditions(params, batch, jax.random.PRNGKey(0))
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


def meanflow_step(implementation: str, dtype, params=None):
    """One MeanFlow step of a one-layer DiT at `dtype` through the attention `implementation`,
    every row on an interval (r < t), so every target takes the JVP: its loss and its gradient in
    every parameter, from `params` (`moved_parameters` where None) and the same batch and step key
    each time, and the parameters it started from."""
    model = SimpleDiT(patch_size=2, emb_features=64, num_layers=1, num_heads=2, mlp_ratio=1,
                      interval=True, attention_impl=implementation, dtype=dtype)
    inputs = InputSpec(Field("image", (16, 16, 3)),
                       {"textcontext": Condition(CharTable.from_pretrained("char_table"))})
    task = MeanFlowObjective(model, presets.MeanFlow()(), inputs, ema_decay=None, instantaneous=0.0)
    params = moved_parameters(task.init(jax.random.PRNGKey(0))) if params is None else params
    pixels = jax.random.randint(jax.random.PRNGKey(1), (8, 16, 16, 3), 0, 256)
    batch = {"image": np.asarray(pixels, np.uint8), **task.inputs.tokenize(["a", "b", "c", "d"] * 2)}
    step = Step(step=jnp.asarray(0), key=jax.random.PRNGKey(2), ema=None)

    def loss_of(tree):
        return task.scalar_loss({**params, "params": tree}, batch, step)[0]

    loss, grads = jax.jit(jax.value_and_grad(loss_of))(params["params"])
    widest = functools.partial(np.asarray, dtype=np.float64)
    return widest(loss), jax.tree.map(widest, grads), params


def moved_parameters(variables):
    """`variables` with every parameter moved by a seeded normal of 0.05: a DiT starts with its
    output zero, so its velocity's time derivative, the JVP MeanFlow's target takes, is zero there,
    and away from it the target and every attention derivative are not."""
    leaves, tree = jax.tree.flatten(variables["params"])
    keys = jax.random.split(jax.random.PRNGKey(3), len(leaves))
    moved = [leaf + 0.05 * jax.random.normal(key, leaf.shape, leaf.dtype)
             for leaf, key in zip(leaves, keys, strict=True)]
    return {**variables, "params": jax.tree.unflatten(tree, moved)}


def as_exact_as_xla(kernel: str) -> None:
    """A MeanFlow step in bfloat16 through the fused attention `kernel`, its loss and every
    parameter's gradient, as near float64 as the same step through XLA's attention in bfloat16:
    the truth is that step widened, the model at float64 from the same float32 parameters, and
    the float64 rule holds each entry (tests/reference_error.py). Every row's target takes the
    JVP, which the fused kernel's call traces from the reference path (`forward_mode_attention`),
    so each gradient holds that tangent."""
    fused_loss, fused, params = meanflow_step(kernel, "bfloat16")
    xla_loss, xla, _ = meanflow_step("xla", "bfloat16")
    with jax.enable_x64(new_val=True), jax.default_device(jax.devices("cpu")[0]):
        truth_loss, truth, _ = meanflow_step("xla", "float64", widened(params))
    assert_as_exact_as_the_reference(fused_loss, xla_loss, truth_loss, f"{kernel} loss")
    assert_leaves_as_exact(fused, xla, truth)


@pytest.mark.gpu(reason="a fused attention kernel runs on a cuda device")
def test_meanflow_trains_through_a_fused_attention_kernel():
    """A fused attention kernel defines only its reverse-mode derivative, so MeanFlow's target JVP
    traces for forward mode (`forward_mode_attention`): a step through Triton's kernel, which runs
    under the cuda lane's --xla_gpu_deterministic_ops, is as exact as the xla kernel's
    (`as_exact_as_xla`), and so is one through cuDNN's, which XLA cannot run under that flag
    (openxla/xla#46500), in a process of its own whose XLA starts without it."""
    as_exact_as_xla("triton")
    flags = [flag for flag in os.environ.get("XLA_FLAGS", "").split()
             if not flag.startswith("--xla_gpu_deterministic_ops")]
    script = "import test_mean_flow; test_mean_flow.as_exact_as_xla('cudnn')"
    # This checkout's Dew, ahead of any the environment installs.
    source = str(Path(__file__).resolve().parents[1] / "src")
    path = os.pathsep.join(filter(None, (source, os.environ.get("PYTHONPATH"))))
    done = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parent,
                          capture_output=True, text=True, timeout=900,
                          env={**os.environ, "XLA_FLAGS": " ".join(flags), "PYTHONPATH": path})
    assert done.returncode == 0, done.stderr[-4000:]
