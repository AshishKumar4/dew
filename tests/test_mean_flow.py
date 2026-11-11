"""MeanFlow against Gsunshine/meanflow's own `forward`
(`tools/meanflow_reference.py`), and the objective and interval sampling
built around it."""


import jax
import jax.numpy as jnp
import numpy as np
from float64_replay import float64_replay
from reference_error import as_given, assert_as_exact, widened

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

    @as_given
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

    from dew.config import ObjectiveConfig, TrainerConfig
    from dew.config.diffusion import DiffusionRunConfig
    from dew.inference import TextToImage
    from dew.inputs import TextCondition
    from dew.registry import ModelConfig

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


MEANFLOW_SEEDS = {"triton": 128, "cudnn": 8}
"""Check 1's declared seeds for a MeanFlow step through each fused attention kernel, sized paired on the
GPUs the site runs on (`seeds_for` with the fused step as `dew_fn`, its confirming block; `meanflow_case`),
the largest of an A100's and a GH200's."""


def meanflow_case(kernel: str):
    """The site of a MeanFlow step through the fused attention `kernel`: `(dew, reference, truth, draw)`,
    the step through `kernel` and through xla, both in bf16, and the xla step's program replayed in float64
    (`float64_replay`), each the step's loss and every parameter's gradient as a function of a draw's
    weights, images and the step's raw key, which draws the step's times and noise.

    A draw's weights are the model's own initializers' at the draw's key, each matrix that initializes to
    zero drawn as a lecun-normal matrix of its shape: adaLN-Zero's modulation and output start at zero,
    where attention reaches neither the output nor any gradient (ledger 789). Every leaf drawn at a flat
    0.3 instead made the output's derivative in time 2e7 to 1.3e8 (its Fourier time features, 2 pi 16 a
    unit time, through gains a trained or initialized model has not), where one float32 rounding of the
    time moves the output by the gradient's own size: float32 determined no digit of the gradients, and
    check 1 judged no kernel (row 5469). At its initializers' scale the derivative is 1.3e5 to 2e5, a
    float32 step 0.4 to 0.8% from the truth in its worst leaf and a bf16 step 1.2 to 1.9%."""
    def over(implementation: str, dtype: str) -> MeanFlowObjective:
        model = SimpleDiT(patch_size=2, emb_features=64, num_layers=1, num_heads=2, mlp_ratio=1,
                          interval=True, attention_impl=implementation, dtype=dtype)
        inputs = InputSpec(Field("image", (16, 16, 3)),
                           {"textcontext": Condition(CharTable.from_pretrained("char_table"))})
        return MeanFlowObjective(model, presets.MeanFlow()(), inputs, ema_decay=None)

    fused, xla = over(kernel, "bfloat16"), over("xla", "bfloat16")
    template = xla.init(jax.random.PRNGKey(0))
    tokens = xla.inputs.tokenize(["a", "b", "c", "d"] * 4)

    def draw(seed):
        keys = jax.random.split(jax.random.PRNGKey(seed), 4)
        leaves, tree = jax.tree.flatten(xla.init(keys[2])["params"])
        zeroed = jax.random.split(keys[3], len(leaves))
        params = jax.tree.unflatten(tree, [
            jax.random.normal(key, leaf.shape, leaf.dtype) / np.sqrt(np.prod(leaf.shape[:-1]))
            if leaf.ndim >= 2 and not np.any(np.asarray(leaf)) else leaf
            for key, leaf in zip(zeroed, leaves, strict=True)])
        pixels = np.asarray(jax.random.randint(keys[0], (16, 16, 16, 3), 0, 256), np.uint8)
        return params, pixels, np.asarray(keys[1])

    def stepped(task):
        def run(params, pixels, key):
            batch = {"image": pixels, **tokens}
            step = Step(step=jnp.asarray(0), key=jnp.asarray(key), ema=None)

            def loss_of(tree):
                return task.scalar_loss({**template, "params": tree}, batch, step)[0]

            return jax.value_and_grad(loss_of)(params)
        return run

    return stepped(fused), stepped(xla), float64_replay(stepped(xla), compiled=True), draw


def agrees_with_xla(kernel: str) -> None:
    """A MeanFlow step through the fused attention `kernel` runs, and its loss and every parameter's
    gradient are on par with the xla kernel's (check 1) from the float64 truth (`meanflow_case`)."""
    dew, reference, truth, draw = meanflow_case(kernel)
    assert_as_exact(dew, reference, draw, f"MeanFlow step on {kernel}", seeds=MEANFLOW_SEEDS[kernel],
                    truth_fn=truth)


def meanflow_trains_through_a_fused_attention_kernel():
    """A fused attention kernel defines only its reverse-mode derivative, so MeanFlow's target JVP
    traces for forward mode (`forward_mode_attention`): a step through Triton's kernel and one through
    cuDNN's each agree with the xla kernel's (`agrees_with_xla`), under the cuda lane's
    --xla_gpu_deterministic_ops as the lane runs."""
    agrees_with_xla("triton")
    agrees_with_xla("cudnn")
