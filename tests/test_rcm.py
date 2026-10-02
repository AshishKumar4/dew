"""rCM's losses against NVlabs/rcm's own methods (`tools/rcm_reference.py`),
and forward-mode attention for its tangent."""

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.nn.attention import forward_mode_attention, scaled_dot_product_attention
from dew.objectives.diffusion.consistency import (
    backward_simulation,
    consistency_loss,
    critic_loss,
    discrete_consistency_loss,
    distribution_matching_loss,
    guided,
    trig_prediction,
)

CASE = np.load(Path(__file__).resolve().parent / "fixtures" / "rcm" / "losses.npz")
CONFIG = json.loads(str(CASE["config"]))
X0 = jnp.asarray(CASE["x0"], jnp.float32)
LABEL = jnp.asarray(CASE["label"], jnp.float32)


def network(name: str, label=LABEL):
    """`tools/rcm_reference.py`'s closed-form velocity at rf time."""
    strength = CONFIG[f"strength_{name}"]

    def velocity(x, rf):
        time = rf.reshape(-1, 1, 1, 1, 1)
        return (
            jnp.tanh(x) * strength
            + jnp.sin(2 * time) * x * 0.3
            + (0.1 + time) * label.reshape(-1, 1, 1, 1, 1)
        )

    return velocity


def teacher(x, t):
    clean, F = trig_prediction(network("teacher"), x, t)
    clean_u, F_u = trig_prediction(network("teacher", 0 * LABEL), x, t)
    scale = CONFIG["teacher_guidance"]
    return guided(clean_u, clean, scale), guided(F_u, F, scale)


# The reference runs in float64 and Dew in float32; each loss is a sum over
# 32 entries of O(1) terms, so a relative 1e-4 is float32 rounding, and a
# wrong term (a missed warmup, tangent or normalization) moves it by far more.
RTOL = 1e-4


@pytest.mark.parametrize("iteration", [3, 12])
def test_the_consistency_loss_is_rcms(iteration):
    """sCM's step through its warmup (ratio 0.3) and past it."""
    t = jnp.asarray(CASE[f"scm{iteration}.time"][:, 0], jnp.float32)
    noise = jnp.asarray(CASE[f"scm{iteration}.noise"], jnp.float32)
    cos, sin = jnp.cos(t).reshape(-1, 1, 1, 1, 1), jnp.sin(t).reshape(-1, 1, 1, 1, 1)
    x = X0 * cos + noise * sin
    ratio = min(1.0, iteration / CONFIG["tangent_warmup"])
    loss = consistency_loss(lambda x, t: trig_prediction(network("student"), x, t)[1], x, t, teacher(x, t)[1],
                            ratio, CONFIG["loss_scale"])
    np.testing.assert_allclose(np.asarray(loss), CASE[f"scm{iteration}.loss"], rtol=RTOL)


def test_the_discrete_consistency_loss_is_rcms():
    """rCM's dCM step: two teacher Euler steps on an 8-point grid at shift 5."""
    student = network("student")
    loss = discrete_consistency_loss(
        lambda x, t: trig_prediction(student, x, t)[0], lambda x, t: teacher(x, t)[1], X0,
        jnp.asarray(CASE["dcm.noise"], jnp.float32),
        # The recorded uniforms, over [0, 1 - skip / steps) as the step scales them.
        jnp.asarray(CASE["dcm.u"][:, 0], jnp.float32) * (1 - CONFIG["dcm_skipping_interval_steps"]
                                                        / CONFIG["dcm_total_steps"]),
        CONFIG["dcm_total_steps"], CONFIG["dcm_skipping_interval_steps"], CONFIG["dcm_timestep_shift"],
        CONFIG["loss_scale"])
    np.testing.assert_allclose(np.asarray(loss), CASE["dcm.loss"], rtol=RTOL)


def generated():
    """The student's four-step sample on the reference's draws: three
    simulation times, the initial noise and three noisings."""
    times = jnp.asarray(CASE["dmd.times"][:3, :, 0], jnp.float32)
    draws = jnp.asarray(CASE["dmd.draws"], jnp.float32)
    student = network("student")
    return backward_simulation(lambda x, t: trig_prediction(student, x, t)[0], draws[0], times, draws[1:4])


def test_the_distribution_matching_loss_is_rcms():
    sample = generated()
    t = jnp.asarray(CASE["dmd.times"][3, :, 0], jnp.float32)
    noise = jnp.asarray(CASE["dmd.draws"][4], jnp.float32)
    x = jnp.cos(t).reshape(-1, 1, 1, 1, 1) * sample + jnp.sin(t).reshape(-1, 1, 1, 1, 1) * noise
    fake, _ = trig_prediction(network("fake_score"), x, t)
    loss = distribution_matching_loss(sample, fake, teacher(x, t)[0], CONFIG["loss_scale_dmd"])
    np.testing.assert_allclose(np.asarray(loss), CASE["dmd.loss"], rtol=RTOL)


def test_the_critic_loss_is_rcms():
    times = jnp.asarray(CASE["critic.times"][:3, :, 0], jnp.float32)
    draws = jnp.asarray(CASE["critic.draws"], jnp.float32)
    student = network("student")
    sample = backward_simulation(lambda x, t: trig_prediction(student, x, t)[0], draws[0], times, draws[1:4])
    t = jnp.asarray(CASE["critic.times"][3, :, 0], jnp.float32)
    x = jnp.cos(t).reshape(-1, 1, 1, 1, 1) * sample + jnp.sin(t).reshape(-1, 1, 1, 1, 1) * draws[4]
    fake, _ = trig_prediction(network("fake_score"), x, t)
    np.testing.assert_allclose(np.asarray(critic_loss(sample, fake, t)), CASE["critic.loss"], rtol=RTOL)


def test_a_step_past_the_drawn_count_leaves_the_sample_where_it_is():
    """With the last noising not live, the walk is the three-step one."""
    times = jnp.asarray(CASE["dmd.times"][:3, :, 0], jnp.float32)
    draws = jnp.asarray(CASE["dmd.draws"], jnp.float32)
    student = network("student")

    def clean(x, t):
        return trig_prediction(student, x, t)[0]
    masked = backward_simulation(clean, draws[0], times, draws[1:4], jnp.asarray([True, True, False]))
    np.testing.assert_array_equal(np.asarray(masked),
                                  np.asarray(backward_simulation(clean, draws[0], times[:2], draws[1:3])))
    assert float(jnp.pi / 2) == pytest.approx(math.pi / 2)


def attention(implementation):
    q, k, v = (jax.random.normal(jax.random.PRNGKey(i), (2, 16, 2, 8)) for i in range(3))
    tangents = tuple(jax.random.normal(jax.random.PRNGKey(10 + i), q.shape) for i in range(3))
    with forward_mode_attention():
        return jax.jvp(lambda q, k, v: scaled_dot_product_attention(q, k, v, implementation=implementation),
                       (q, k, v), tangents)


def test_forward_mode_attention_leaves_a_jvp_capable_kernel_as_it_is():
    out, tangent = attention("xla")
    reference_out, reference_tangent = attention("reference")
    np.testing.assert_allclose(np.asarray(out), np.asarray(reference_out), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(tangent), np.asarray(reference_tangent), rtol=1e-5, atol=1e-6)


def test_a_run_config_distills_a_saved_flow_run_and_alternates_student_and_critic(tmp_path):
    """The teacher is a saved run; the student and fake score start from
    it. Past the warmup one step in two trains the student and the other
    the fake score, each leaving the other's gradient zero, and the saved
    student's task samples as the objective's own."""
    import dataclasses

    import optax
    from test_diffusion_run_sources import batch_for

    from dew.checkpoints import Checkpoints
    from dew.config import ModelConfig, TrainerConfig
    from dew.data import OxfordFlowers
    from dew.objectives.base import Step
    from dew.objectives.diffusion import (
        ConsistencyDistillation,
        ConsistencyDistillationObjective,
        DiffusionRunConfig,
        TextCondition,
    )
    from dew.objectives.diffusion.objective import FAKE_SCORE, TEACHER
    from dew.registry import presets, samplers
    from dew.sampling import TextToImage
    from dew.training import Trainer

    teacher_run = DiffusionRunConfig(
        model=ModelConfig("simple_dit", {"patch_size": 2, "emb_features": 16, "num_layers": 1, "num_heads": 2,
                                         "time_scale": 0.002},
                          dtype="float32", attention_impl="xla"),
        data=OxfordFlowers(image_size=4), preset=presets.Flow(), sampler=samplers.Euler(), guidance=None,
        sampling_steps=2, ema_decay=None, val_metrics=(), trainer=TrainerConfig(checkpoint_dir=str(tmp_path)),
        text=TextCondition(encoder="char_table", checkpoint="char_table"))
    teacher = teacher_run.build()
    trainer = Trainer(teacher, optax.adam(1e-2), key=jax.random.PRNGKey(3))
    state = trainer.initial_state()
    batch = batch_for(teacher, 4)
    state, *_ = trainer.compile(state, batch)(state, batch)
    for directory in ("teacher", "student"):
        (tmp_path / directory).mkdir()
    checkpoints = Checkpoints(str(tmp_path / "teacher"))
    checkpoints.save(1, state, None)
    checkpoints.wait()
    teacher_run.save(str(tmp_path / "teacher"))

    fast = tmp_path / "fast"
    fast.mkdir()
    dataclasses.replace(
        teacher_run,
        model=dataclasses.replace(
            teacher_run.model,
            config={key: value for key, value in teacher_run.model.config.items() if key != "time_scale"},
        ),
    ).save(str(fast))
    with pytest.raises(ValueError, match="time_scale=16"):
        dataclasses.replace(teacher_run, distill=ConsistencyDistillation(teacher=str(fast))).build()

    config = dataclasses.replace(teacher_run, distill=ConsistencyDistillation(
        teacher=str(tmp_path / "teacher"), teacher_guidance=2.0, tangent_warmup=1, student_update_freq=2,
        max_simulation_steps=2), sampler=samplers.Consistency(), sampling_steps=3)
    task = config.build()
    assert isinstance(task, ConsistencyDistillationObjective)
    params = task.init(jax.random.PRNGKey(0))
    for got, want in zip(jax.tree.leaves(params[TEACHER]), jax.tree.leaves(task.trainable(state.params)),
                         strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    # The student reads the teacher's Fourier table, built at the teacher's time scale.
    table = params["constants"]["conditioning"]["time_embed"]["layers_0"]["frequencies"]
    np.testing.assert_array_equal(np.asarray(table), np.asarray(
        state.params["constants"]["conditioning"]["time_embed"]["layers_0"]["frequencies"]))

    def gradients(step):
        def loss(tree):
            return task.loss({**params, "params": tree}, batch,
                             Step(jnp.asarray(step), jax.random.PRNGKey(4), None))[0].total
        return jax.grad(loss)(params["params"])

    def size(tree):
        return float(sum(jnp.abs(leaf).sum() for leaf in jax.tree.leaves(tree)))

    student, critic = gradients(3), gradients(2)
    assert size({k: v for k, v in student.items() if k != FAKE_SCORE}) > 0 and size(student[FAKE_SCORE]) == 0
    assert size({k: v for k, v in critic.items() if k != FAKE_SCORE}) == 0 and size(critic[FAKE_SCORE]) > 0

    trainer = Trainer(task, optax.adam(1e-3), key=jax.random.PRNGKey(5))
    distilled = trainer.initial_state()
    distilled, *_ = trainer.compile(distilled, batch)(distilled, batch)
    checkpoints = Checkpoints(str(tmp_path / "student"))
    checkpoints.save(1, distilled, None)
    checkpoints.wait()
    config.save(str(tmp_path / "student"))
    expected = task.pipeline(distilled, ema=False)(["a red bird"], key=9).host().images
    np.testing.assert_array_equal(TextToImage.from_run(str(tmp_path / "student"))(["a red bird"], key=9)
                                  .host().images, expected)


def test_a_reverse_only_kernel_takes_forward_mode_through_its_reference():
    """The mechanism `forward_mode_attention` wraps cuDNN and TPU calls in,
    on a double of such a kernel: the reference path under a custom_vjp,
    which refuses forward mode as the fused kernels do. Wrapped, its value
    is the kernel's own and its tangent the reference's, bit for bit."""
    from dew.nn.attention import forward_differentiable

    def reference(q, k, v):
        return scaled_dot_product_attention(q, k, v, implementation="reference")

    @jax.custom_vjp
    def reverse_only(q, k, v):
        return reference(q, k, v)

    reverse_only.defvjp(lambda q, k, v: (reference(q, k, v), (q, k, v)),
                        lambda residuals, cotangent: jax.vjp(reference, *residuals)[1](cotangent))
    q, k, v = (jax.random.normal(jax.random.PRNGKey(i), (2, 16, 2, 8)) for i in range(3))
    tangents = tuple(jax.random.normal(jax.random.PRNGKey(10 + i), q.shape) for i in range(3))
    with pytest.raises(TypeError, match="forward-mode"):
        jax.jvp(reverse_only, (q, k, v), tangents)
    out, tangent = jax.jvp(lambda q, k, v: forward_differentiable(reverse_only, reference, q, k, v),
                           (q, k, v), tangents)
    _, reference_tangent = jax.jvp(reference, (q, k, v), tangents)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(reverse_only(q, k, v)))
    np.testing.assert_array_equal(np.asarray(tangent), np.asarray(reference_tangent))
