"""rCM's losses against NVlabs/rcm's own methods (`tools/rcm_reference.py`),
and forward-mode attention for its tangent."""

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn
from reference_error import assert_as_exact_as_the_reference, distance

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
    student's task samples as the objective's own. A LoRA student distills
    the same teacher, whose model and the fake score's are the run's
    without the adapter."""
    import dataclasses

    import optax
    from diffusion_stubs import batch_for

    from dew.checkpoints import Checkpoints
    from dew.config import ModelConfig, ObjectiveConfig, TrainerConfig
    from dew.data import TFDSImages
    from dew.diffusion.presets import Flow
    from dew.lora import LoRA
    from dew.objectives.base import FROZEN, Step
    from dew.objectives.diffusion import ConsistencyDistillationObjective, DiffusionRunConfig, TextCondition
    from dew.objectives.diffusion.objective import FAKE_SCORE, TEACHER
    from dew.sampling import Consistency, Euler, TextToImage
    from dew.training import Trainer

    teacher_run = DiffusionRunConfig(
        model=ModelConfig("simple_dit", {"patch_size": 2, "emb_features": 16, "num_layers": 1, "num_heads": 2,
                                         "time_scale": 0.002, "dtype": "float32", "attention_impl": "xla"}),
        data=TFDSImages(image_size=4), preset=Flow(), val_metrics=(),
        trainer=TrainerConfig(checkpoint_dir=str(tmp_path)),
        text=TextCondition(encoder="char_table", checkpoint="char_table"),
        objective=ObjectiveConfig("diffusion",
                                  {"solver": Euler(), "guidance": None, "steps": 2, "ema_decay": None}))
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

    def distilling(**fields):
        return dataclasses.replace(teacher_run, objective=ObjectiveConfig("rcm", {
            "teacher_run": str(tmp_path / "teacher"), "solver": Consistency(), "steps": 3, "ema_decay": None,
            **fields}))

    # The student starts from the teacher's time features, so a model whose
    # features turn fast is refused, and a slower one is as smooth.
    for time_scale, refused in ((16, True), (0.001, False)):
        model = dataclasses.replace(teacher_run.model, fields={**teacher_run.model.fields,
                                                                "time_scale": time_scale})
        if refused:
            with pytest.raises(ValueError, match="time_scale=16"):
                dataclasses.replace(distilling(), model=model).build()
        else:
            dataclasses.replace(distilling(), model=model).build()

    config = distilling(teacher_guidance=2.0, tangent_warmup=1, student_update_freq=2, max_simulation_steps=2)
    task = config.build()
    assert isinstance(task, ConsistencyDistillationObjective)
    params = task.init(jax.random.PRNGKey(0))
    expected = task.model_variables(state.variables)
    for got, want in zip(jax.tree.leaves(params[TEACHER]), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    # The student reads the teacher's Fourier table, built at the teacher's time scale.
    table = params["constants"]["conditioning"]["time_embed"]["layers_0"]["frequencies"]
    np.testing.assert_array_equal(np.asarray(table), np.asarray(
        state.variables["constants"]["conditioning"]["time_embed"]["layers_0"]["frequencies"]))

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

    # A LoRA student starts as the teacher: B is zero, up to its branch's rounding.
    adapted = dataclasses.replace(config, lora=LoRA(rank=2, modules=("ada_proj", "final_proj"))).build()
    assert [program.trained for program in adapted.program_key()] == [True, False, True]
    start = adapted.init(jax.random.PRNGKey(0))
    assert {path[-1].key for path, _ in jax.tree_util.tree_flatten_with_path(start["params"])[0]
            if path[0].key != FAKE_SCORE} == {"lora_A", "lora_B"}
    for step in (Step(jnp.asarray(at), jax.random.PRNGKey(4), None) for at in (3, 2)):
        np.testing.assert_allclose(*(objective.loss(tree, batch, step)[0].total
                                     for objective, tree in ((adapted, start), (task, params))), rtol=1e-5)
    trainer = Trainer(adapted, optax.adam(1e-3), key=jax.random.PRNGKey(5))
    tuned = trainer.initial_state()
    tuned, *_ = trainer.compile(tuned, batch)(tuned, batch)
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree.leaves(tuned.variables[FROZEN]))

    trainer = Trainer(task, optax.adam(1e-3), key=jax.random.PRNGKey(5))
    distilled = trainer.initial_state()
    distilled, *_ = trainer.compile(distilled, batch)(distilled, batch)
    checkpoints = Checkpoints(str(tmp_path / "student"))
    checkpoints.save(1, distilled, None, artifact=task.inference_record())
    checkpoints.wait()
    config.save(str(tmp_path / "student"))
    expected = task.pipeline(distilled, ema=False)(["a red bird"], key=9).host().images
    np.testing.assert_array_equal(TextToImage.from_run(str(tmp_path / "student"))(["a red bird"], key=9)
                                  .host().images, expected)


TRAINING = np.load(Path(__file__).resolve().parent / "fixtures" / "rcm" / "training.npz")


class Weighted(nn.Module):
    """`tools/rcm_reference.py`'s trained velocity network, four terms with
    a weight per entry, at Dew's model time (rf time times 1000), its label
    the condition's second token's table entry."""

    @nn.compact
    def __call__(self, x, time, textcontext):
        time = time.reshape(-1, 1, 1, 1, 1) / 1000
        label = textcontext.hidden[:, 1, 0].reshape(-1, 1, 1, 1, 1)
        w = self.param("weights", nn.initializers.zeros, (4, *x.shape[1:]))
        return (jnp.tanh(x) * w[0] + jnp.sin(2 * time) * x * w[1] + (0.1 + time) * label * w[2]
                + jnp.cos(x) * time * w[3])


def test_a_pretrained_run_distills_the_model_it_loads_without_a_teacher_run(tmp_path):
    """`--pretrained <pipeline> --objective rcm` with no teacher run: the
    teacher is the model the run loads, its weights bitwise the source's,
    and the student starts as it."""
    import tarfile
    from pathlib import Path

    from test_diffusion_run_sources import precision

    from dew.config import ObjectiveConfig
    from dew.data import TFDSImages
    from dew.diffusion.presets import Flow
    from dew.interop.pretrained import load_diffusion_source
    from dew.objectives.diffusion import DiffusionRunConfig
    from dew.objectives.diffusion.objective import TEACHER, model_part

    with tarfile.open(Path(__file__).resolve().parent / "fixtures" / "flux_source.tar.xz") as archive:
        archive.extractall(tmp_path, filter="data")
    pipeline = str(tmp_path / "pipeline")
    objective = DiffusionRunConfig(pretrained=pipeline, preset=Flow(), model=precision(),
                                   data=TFDSImages(image_size=16), val_metrics=(),
                                   objective=ObjectiveConfig("rcm", {"ema_decay": None})).build()
    source = load_diffusion_source(pipeline, dtype="float32", attention_impl="xla", size=(16, 16))
    expected = model_part(source.variables)
    held = objective.init(jax.random.PRNGKey(0))
    for tree in (objective.teacher_variables, held[TEACHER]):
        assert jax.tree.structure(tree) == jax.tree.structure(expected)
        for got, want in zip(jax.tree.leaves(tree), jax.tree.leaves(expected), strict=True):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def discrete_fields(prefix: str) -> dict:
    """rCM's dCM settings as the objective's arguments, none for sCM."""
    if not prefix:
        return {}
    settings = json.loads(str(TRAINING["dcm"]))
    return {"consistency": "discrete", "discrete_steps": settings["dcm_total_steps"],
            "discrete_skip": settings["dcm_skipping_interval_steps"],
            "discrete_shift": settings["dcm_timestep_shift"]}


def distilled(monkeypatch, optimizer, prefix=""):
    """`ConsistencyDistillationObjective` over the fixture's network, labels
    and settings, trained by `Trainer` with `optimizer` for the fixture's
    iterations on the reference's draws: the objective and its final state.
    `prefix` "dcm/" trains rCM's discrete-time consistency on its draws."""
    from diffusion_stubs import label_table

    from dew.diffusion import presets
    from dew.inputs import Condition, Field, InputSpec
    from dew.objectives.diffusion import ConsistencyDistillationObjective
    from dew.objectives.diffusion.consistency import _Draws
    from dew.objectives.diffusion.objective import TEACHER
    from dew.training import Trainer
    from dew.training.posthoc import power_decay

    config = json.loads(str(TRAINING["config"]))
    pixels, labels = TRAINING["pixels"], TRAINING["label"]
    names = [str(row) for row in range(pixels.shape[0])]
    table = label_table(labels)
    inputs = InputSpec(Field("image", pixels.shape[1:]), {"textcontext": Condition(table)})
    (mean_g, std_g), (mean_d, std_d) = config["times"]["G"], config["times"]["D"]
    teacher = {"params": {"weights": jnp.asarray(TRAINING["teacher"])}}
    task = ConsistencyDistillationObjective(
        Weighted(), presets.Flow()(), inputs, consistency_weight=config["loss_scale"],
        dmd_weight=config["loss_scale_dmd"], teacher_guidance=config["teacher_guidance"],
        tangent_warmup=config["tangent_warmup"], student_update_freq=config["student_update_freq"],
        max_simulation_steps=config["max_simulation_steps_fake"], student_times=(mean_g, std_g),
        critic_times=(mean_d, std_d), **discrete_fields(prefix), variables={TEACHER: teacher},
        ema_decay=power_decay(config["ema_rate"]))
    drawn = {name: jnp.asarray(TRAINING[f"{prefix}draws/{name}"]) for name in _Draws._fields}
    monkeypatch.setattr(ConsistencyDistillationObjective, "_draws", lambda self, step, count, shape: _Draws(
        **{name: value[step.step] for name, value in drawn.items()}))
    trainer = Trainer(task, optimizer, key=jax.random.PRNGKey(0))
    state = trainer.initial_state()
    batch = {"image": pixels, **inputs.tokenize(names)}
    step = trainer.compile(state, batch)
    for _ in range(config["iterations"]):
        state, *_ = step(state, batch)
    return task, state


@pytest.mark.parametrize("prefix", ["", "dcm/"], ids=["scm", "dcm"])
def test_training_steps_the_student_and_the_fake_score_as_rcms_loop_does(monkeypatch, prefix):
    """Ten updates of rCM's own loop (`ImaginaireTrainer_Distill.training_step`
    over the model's closures, `tools/rcm_reference.py`): a warmup of three
    student updates on sCM alone, then the student, sCM and DMD2, on one
    update in three and the fake score on the other two, each network with
    its own Adam and the power EMA (rate 0.1) on the student's updates at
    the student's own count. On the reference's draws, `Trainer` over
    `ConsistencyDistillationObjective` lands every network and the EMA where
    rCM's float64 run does, held by the float64 rule.

    rCM's loop backpropagates the rows' summed loss and Dew the mean, so
    Dew's Adam takes rCM's epsilon over the eight rows; the steps are the
    same. Adam's decays are 0.5 and 0.75, exact with their powers in binary,
    since optax rounds its bias corrections in float32 where torch keeps
    float64. `dcm` runs the same loop on rCM's discrete-time consistency
    (`_student_dcm_step`): two teacher Euler steps apart on an 8-point grid
    at shift 5."""
    from dew.objectives.diffusion.objective import FAKE_SCORE
    from dew.training import Trainer

    config = json.loads(str(TRAINING["config"]))
    rows = TRAINING["pixels"].shape[0]
    b1, b2 = config["betas"]
    task, state = distilled(monkeypatch, optax.adam(config["learning_rate"], b1=b1, b2=b2,
                                                    eps=config["epsilon"] / rows), prefix)
    params = state.variables["params"]
    for name, got in (("student", params["weights"]), ("fake_score", params[FAKE_SCORE]["weights"]),
                      ("ema", state.ema["params"]["weights"])):
        key = f"{prefix}{name}/weights"
        assert_as_exact_as_the_reference(got, TRAINING[key], TRAINING[f"{key}_f64"], key)
    with pytest.raises(ValueError, match="accumulation=1"):
        Trainer(task, optax.adam(1e-3), key=0, accumulation=2)


def test_training_at_rcms_published_optimizer_is_rcms_up_to_optax_rounding(monkeypatch):
    """The same loop at rCM's published AdamW (lr 1e-4, betas 0.9 and
    0.99, weight decay 0.1). optax rounds Adam's bias corrections 1 - b^k in
    float32, and each Adam step is at most the learning rate (Kingma & Ba,
    section 2.1, as (1 - b1) = sqrt(1 - b2) here), so step k of a network
    lands up to lr (|e1_k| + |e2_k| / 2 + |e1_k e2_k|) from torch's, e_k the
    relative error of optax's float32 correction, computed here as optax
    computes it. The student's distance from rCM's float64 run is held to
    twice the reference's float32 one plus the sum of those over its steps."""
    published = json.loads(str(TRAINING["published"]))
    rows = TRAINING["pixels"].shape[0]
    (b1, b2), lr = published["betas"], published["lr"]
    _, state = distilled(monkeypatch, optax.adamw(lr, b1=b1, b2=b2, eps=published["eps"] / rows,
                                                  weight_decay=published["weight_decay"]))

    def error(decay, step):
        rounded = 1 - jnp.asarray(decay, jnp.float32) ** jnp.asarray(step, jnp.int32)
        exact = 1 - decay ** step
        return abs(float(rounded) - exact) / exact

    steps = range(1, int(TRAINING["published/student/count"]) + 1)
    slack = sum(lr * (error(b1, k) + error(b2, k) / 2 + error(b1, k) * error(b2, k)) for k in steps)
    student = np.asarray(state.variables["params"]["weights"], np.float64)
    truth = TRAINING["published/student/weights_f64"]
    theirs = distance(TRAINING["published/student/weights"], truth)
    assert distance(student, truth) <= 2 * theirs + slack, (distance(student, truth), theirs, slack)


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
