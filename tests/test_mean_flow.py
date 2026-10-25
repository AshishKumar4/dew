"""MeanFlow against Gsunshine/meanflow's own `forward`
(`tools/meanflow_reference.py`), and the objective and interval sampling
built around it."""

import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact, widened

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
    schedule = task.process.schedule
    variables = task.model_variables(params)

    def draw(seed):
        return (jax.random.normal(jax.random.PRNGKey(seed), (4, 4, 4, 3)),)

    def walked(x_T):
        return sample(denoise, x_T, 2, solver=Euler(), key=jax.random.PRNGKey(0), final_denoise=False)

    def interval(model, held, x_T, conditions):
        start, end = (schedule.model_time(jnp.full((4,), time, x_T.dtype)) for time in (1, 0))
        return x_T - model.apply(held, x_T, start, **conditions, duration=start - end)

    def truth(x_T):
        with jax.enable_x64(new_val=True):
            return interval(task.model.clone(dtype=jnp.float64), *widened((variables, x_T, given)))

    # The walk is one Euler step from t=1 to t=0 at the average velocity u over the interval, x_T - u:
    # on par with u taken by the model itself (check 1), the model in float64 the truth.
    assert_as_exact(jax.jit(walked), jax.jit(lambda x_T: interval(task.model, variables, x_T, given)), draw,
                    "one-step walk", seeds=WALK_SEEDS, truth_fn=truth)


WALK_SEEDS = 8
"""Check 1's declared seeds for the one-step walk, sized by `seeds_for`."""


def test_a_run_config_trains_meanflow_and_its_saved_task_samples_in_one_step(tmp_path):
    from diffusion_stubs import batch_for
    from diffusion_support import images_of, saved_after_one_step

    from dew.config import ModelConfig, ObjectiveConfig, TrainerConfig
    from dew.config.diffusion import DiffusionRunConfig
    from dew.inputs import TextCondition
    from dew.sampling import TextToImage

    config = DiffusionRunConfig(
        model=ModelConfig(
            "simple_dit",
            {"patch_size": 2, "emb_features": 16, "num_layers": 1, "num_heads": 2,
             "dtype": "float32", "attention_impl": "xla"},
        ),
        data=images_of("tfds://oxford_flowers102", 4),
        preset=presets.MeanFlow(),
        val_metrics=(),
        trainer=TrainerConfig(checkpoint_dir=str(tmp_path)),
        text=TextCondition(encoder="char_table", checkpoint="char_table"),
        objective=ObjectiveConfig("mean_flow",
                                  {"omega": 2.0, "kappa": 0.5, "solver": Euler(), "steps": 2,
                                   "ema_decay": None}),
    )
    run = tmp_path / "run"
    task, state, _ = saved_after_one_step(config, lambda task: batch_for(task, 4), run, 1e-2)
    assert isinstance(task, MeanFlowObjective) and task.model.interval
    expected = task.pipeline(state, ema=False)(["a red bird"], key=9).host().images
    np.testing.assert_array_equal(
        TextToImage.from_run(str(run))(["a red bird"], key=9).host().images, expected
    )


def agrees_with_xla(kernel: str) -> None:
    """A MeanFlow step through the fused attention `kernel` runs, and its loss and gradient agree
    with the xla kernel's within bf16's rounding of the step's scale."""
    def run(implementation):
        model = SimpleDiT(patch_size=2, emb_features=64, num_layers=1, num_heads=2, mlp_ratio=1,
                          interval=True, attention_impl=implementation, dtype="bfloat16")
        inputs = InputSpec(Field("image", (16, 16, 3)),
                           {"textcontext": Condition(CharTable.from_pretrained("char_table"))})
        task = MeanFlowObjective(model, presets.MeanFlow()(), inputs, ema_decay=None)
        params = task.init(jax.random.PRNGKey(0))
        pixels = jax.random.randint(jax.random.PRNGKey(1), (8, 16, 16, 3), 0, 256)
        batch = {"image": np.asarray(pixels, np.uint8), **task.inputs.tokenize(["a", "b", "c", "d"] * 2)}
        step = Step(step=jnp.asarray(0), key=jax.random.PRNGKey(2), ema=None)
        def loss_of(tree):
            return task.scalar_loss({**params, "params": tree}, batch, step)[0]

        loss, grads = jax.jit(jax.value_and_grad(loss_of))(params["params"])
        return float(loss), jax.tree.leaves(grads)

    (fused, fused_grads), (xla, xla_grads) = run(kernel), run("xla")
    assert fused == pytest.approx(xla, rel=2 ** -7)
    for left, right in zip(fused_grads, xla_grads, strict=True):
        scale = float(jnp.max(jnp.abs(right))) + 1e-12
        assert float(jnp.max(jnp.abs(left - right))) <= 2 * 2 ** -7 * scale


@pytest.mark.gpu(reason="a fused attention kernel runs on a cuda device")
def test_meanflow_trains_through_a_fused_attention_kernel():
    """A fused attention kernel defines only its reverse-mode derivative, so MeanFlow's target JVP
    traces for forward mode (`forward_mode_attention`): a step through Triton's kernel, which runs
    under the cuda lane's --xla_gpu_deterministic_ops, agrees with the xla kernel's
    (`agrees_with_xla`), and so does one through cuDNN's, which XLA cannot run under that flag
    (openxla/xla#46500), in a process of its own whose XLA starts without it."""
    agrees_with_xla("triton")
    flags = [flag for flag in os.environ.get("XLA_FLAGS", "").split()
             if not flag.startswith("--xla_gpu_deterministic_ops")]
    script = "import test_mean_flow; test_mean_flow.agrees_with_xla('cudnn')"
    # This checkout's Dew, ahead of any the environment installs.
    source = str(Path(__file__).resolve().parents[1] / "src")
    path = os.pathsep.join(filter(None, (source, os.environ.get("PYTHONPATH"))))
    done = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parent,
                          capture_output=True, text=True, timeout=900,
                          env={**os.environ, "XLA_FLAGS": " ".join(flags), "PYTHONPATH": path})
    assert done.returncode == 0, done.stderr[-4000:]
