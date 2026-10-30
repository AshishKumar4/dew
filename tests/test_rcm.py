"""rCM's losses against NVlabs/rcm's own methods (`tools/rcm_reference.py`),
and forward-mode attention for its tangent."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
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


@pytest.mark.parametrize("time_scale", [.001, 16.])
def test_continuous_consistency_trains_a_teacher_with_its_own_time_scale(time_scale):
    """The student differentiates the loaded Fourier table and takes a finite sCM update."""
    from dew.diffusion.presets import Flow
    from dew.inputs import Field, InputSpec
    from dew.nn.backbones import SimpleDiT
    from dew.objectives.base import TEACHER
    from dew.objectives.diffusion import ConsistencyDistillationObjective, DiffusionObjective
    from dew.sampling import Euler
    from dew.training import Trainer

    model = SimpleDiT(patch_size=2, emb_features=16, num_layers=1, num_heads=2,
                       time_scale=time_scale, attention_impl="xla")
    inputs = InputSpec(Field("image", (4, 4, 3)))
    teacher = DiffusionObjective(model, Flow(), inputs, guidance=None, solver=Euler(), ema_decay=None)
    weights = teacher.init(jax.random.key(1))
    objective = ConsistencyDistillationObjective(
        model, Flow(), inputs, guidance=None, ema_decay=None, dmd_weight=0., consistency_weight=1.,
        variables={TEACHER: teacher.model_variables(weights)})
    trainer = Trainer(objective, optax.sgd(1e-3), key=jax.random.key(2))
    initial = trainer.initial_state()
    before = jax.tree.map(np.asarray, initial.variables)
    batch = {"image": np.random.default_rng(3).integers(0, 256, (8, 4, 4, 3), dtype=np.uint8)}
    trained, _, _, _, accepted, _ = trainer.compile(initial, batch)(initial, batch)
    assert bool(accepted) and int(trained.updates) == 1
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(trained.variables))
    assert any(not np.array_equal(start, end) for start, end in zip(
        jax.tree.leaves(before["params"]), jax.tree.leaves(trained.variables["params"]), strict=True))
    for start, end in zip(jax.tree.leaves(before[TEACHER]),
                          jax.tree.leaves(trained.variables[TEACHER]), strict=True):
        np.testing.assert_array_equal(start, end)


def test_a_run_config_distills_a_saved_flow_run_and_alternates_student_and_critic(tmp_path):
    """The teacher is a saved run; the student and fake score start from
    it. Past the warmup one step in two trains the student and the other
    the fake score, each leaving the other's gradient zero, and the saved
    student's task samples as the objective's own. A LoRA student distills
    the same teacher, whose model and the fake score's are the run's
    without the adapter."""
    import dataclasses

    from diffusion_stubs import batch_for
    from diffusion_support import images_of

    from dew.batch import FROZEN
    from dew.checkpoints import Checkpoints
    from dew.config import ObjectiveConfig, TrainerConfig
    from dew.config.diffusion import DiffusionRunConfig
    from dew.diffusion.presets import Flow
    from dew.inference import TextToImage
    from dew.inputs import TextCondition
    from dew.lora import LoRA
    from dew.objectives.base import TEACHER, Step
    from dew.objectives.diffusion import ConsistencyDistillationObjective
    from dew.objectives.diffusion.objective import FAKE_SCORE
    from dew.registry import ModelConfig
    from dew.sampling import Consistency, Euler
    from dew.training import Trainer

    teacher_run = DiffusionRunConfig(
        model=ModelConfig("simple_dit", {"patch_size": 2, "emb_features": 16, "num_layers": 1, "num_heads": 2,
                                         "time_scale": 0.002, "dtype": "float32", "attention_impl": "xla"}),
        data=images_of("tfds://oxford_flowers102", 4), preset=Flow(), val_metrics=(),
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
        objective = ObjectiveConfig("rcm", {"solver": Consistency(), "steps": 3, "ema_decay": None, **fields})
        return dataclasses.replace(teacher_run, teacher_run=str(tmp_path / "teacher"), objective=objective)

    # The checkpoint's Fourier table is authoritative, whatever scale the model record names.
    for time_scale in (16, 0.001):
        model = dataclasses.replace(teacher_run.model, fields={**teacher_run.model.fields,
                                                                "time_scale": time_scale})
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

    @jax.jit
    def gradients(step):
        def loss(tree):
            return task.loss({**params, "params": tree}, batch,
                             Step(step, jax.random.PRNGKey(4), None))[0].total
        return jax.grad(loss)(params["params"])

    def size(tree):
        return float(sum(jnp.abs(leaf).sum() for leaf in jax.tree.leaves(tree)))

    student, critic = gradients(jnp.asarray(3)), gradients(jnp.asarray(2))
    assert size({k: v for k, v in student.items() if k != FAKE_SCORE}) > 0 and size(student[FAKE_SCORE]) == 0
    assert size({k: v for k, v in critic.items() if k != FAKE_SCORE}) == 0 and size(critic[FAKE_SCORE]) > 0

    # A LoRA student starts as the teacher: B is zero, up to its branch's rounding.
    adapted = dataclasses.replace(config, lora=LoRA(rank=2, modules=("ada_proj", "final_proj"))).build()
    assert [program.trained for program in adapted.program_key()] == [True, False, True]
    start = adapted.init(jax.random.PRNGKey(0))
    assert {path[-1].key for path, _ in jax.tree_util.tree_flatten_with_path(start["params"])[0]
            if path[0].key != FAKE_SCORE} == {"lora_A", "lora_B"}
    losses = [jax.jit(lambda tree, step, objective=objective: objective.loss(tree, batch, Step(
        step, jax.random.PRNGKey(4), None))[0].total) for objective in (adapted, task)]
    for at in (3, 2):
        np.testing.assert_allclose(*(loss(tree, jnp.asarray(at)) for loss, tree in zip(
            losses, (start, params), strict=True)), rtol=1e-5)
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


def test_an_accumulated_distillation_alternates_its_networks_update_by_update(tmp_path):
    """An update of rCM takes two microbatches (`accumulation=2`), each of its update's phase: the
    loss reads the committed updates (`Step.updates`), so a critic update's second microbatch leaves
    the student's gradient zero, and four updates past a one-update warmup move the student, the
    student, the fake score and the student, each alone."""
    import dataclasses

    from diffusion_stubs import batch_for
    from diffusion_support import images_of

    from dew.checkpoints import Checkpoints
    from dew.config import ObjectiveConfig, TrainerConfig
    from dew.config.diffusion import DiffusionRunConfig
    from dew.diffusion.presets import Flow
    from dew.inputs import TextCondition
    from dew.objectives.base import Step
    from dew.objectives.diffusion.objective import FAKE_SCORE
    from dew.registry import ModelConfig
    from dew.sampling import Consistency, Euler
    from dew.training import Trainer

    teacher_run = DiffusionRunConfig(
        model=ModelConfig("simple_dit", {"patch_size": 2, "emb_features": 16, "num_layers": 1, "num_heads": 2,
                                         "dtype": "float32", "attention_impl": "xla"}),
        data=images_of("tfds://oxford_flowers102", 4), preset=Flow(), val_metrics=(),
        trainer=TrainerConfig(checkpoint_dir=str(tmp_path)),
        text=TextCondition(encoder="char_table", checkpoint="char_table"),
        objective=ObjectiveConfig("diffusion",
                                  {"solver": Euler(), "guidance": None, "steps": 2, "ema_decay": None}))
    teacher = teacher_run.build()
    trainer = Trainer(teacher, optax.adam(1e-2), key=jax.random.PRNGKey(3))
    state = trainer.initial_state()
    batch = batch_for(teacher, 4)
    state, *_ = trainer.compile(state, batch)(state, batch)
    (tmp_path / "teacher").mkdir()
    checkpoints = Checkpoints(str(tmp_path / "teacher"))
    checkpoints.save(1, state, None)
    checkpoints.wait()
    teacher_run.save(str(tmp_path / "teacher"))
    task = dataclasses.replace(teacher_run, teacher_run=str(tmp_path / "teacher"), objective=ObjectiveConfig(
        "rcm", {"solver": Consistency(), "steps": 3, "ema_decay": None, "tangent_warmup": 1,
                "student_update_freq": 2, "max_simulation_steps": 2})).build()

    trainer = Trainer(task, optax.sgd(1e-2), key=jax.random.PRNGKey(5), accumulation=2)
    distilled, _, _ = trainer.place()
    step = trainer.compile(distilled, batch)
    moved = []
    for _ in range(4):
        before = jax.device_get(distilled.variables["params"])
        for _ in range(2):
            distilled, *_ = step(distilled, batch)
        after = jax.device_get(distilled.variables["params"])
        moved.append(sorted({"critic" if name == FAKE_SCORE else "student" for name in after
                             if any(not np.array_equal(old, new) for old, new in
                                    zip(jax.tree.leaves(before[name]), jax.tree.leaves(after[name]),
                                        strict=True))}))
    assert int(distilled.updates) == 4
    assert moved == [["student"], ["student"], ["critic"], ["student"]]

    params = task.init(jax.random.PRNGKey(0))
    critic = jax.grad(lambda tree: task.loss({**params, "params": tree}, batch, Step(
        jnp.asarray(5), jax.random.PRNGKey(4), None, committed=jnp.asarray(2)))[0].total)(params["params"])
    assert all(float(jnp.abs(leaf).sum()) == 0 for name, tree in critic.items() if name != FAKE_SCORE
               for leaf in jax.tree.leaves(tree))


def test_a_pretrained_run_distills_the_model_it_loads_without_a_teacher_run(tmp_path):
    """`--pretrained <pipeline> --objective rcm` with no teacher run: the
    teacher is the model the run loads, its weights bitwise the source's,
    and the student starts as it. The run samples by what the same objective
    built over the pipeline directly samples by: rCM's own few steps, solver
    and unguided draw, on the pipeline's grid, and its published task draws
    the direct one's images, bit for bit, from the same weights and key."""
    import tarfile
    from pathlib import Path

    from diffusion_support import images_of
    from test_diffusion_run_sources import precision

    from dew.config import ObjectiveConfig
    from dew.config.diffusion import DiffusionRunConfig
    from dew.diffusion.presets import Flow
    from dew.inference.images import ImageSettings
    from dew.interop.pretrained import load_diffusion_source
    from dew.objectives.base import TEACHER
    from dew.objectives.diffusion import ConsistencyDistillationObjective
    from dew.objectives.diffusion.objective import model_part
    from dew.registry import to_record

    with tarfile.open(Path(__file__).resolve().parent / "fixtures" / "flux_source.tar.xz") as archive:
        archive.extractall(tmp_path, filter="data")
    pipeline = str(tmp_path / "pipeline")
    objective = DiffusionRunConfig(pretrained=pipeline, preset=Flow(), model=precision(),
                                   data=images_of("tfds://oxford_flowers102", 16),
                                       val_metrics=(),
                                   objective=ObjectiveConfig("rcm", {"ema_decay": None})).build()
    source = load_diffusion_source(pipeline, dtype="float32", attention_impl="xla", size=(16, 16))
    direct = ConsistencyDistillationObjective(source, objective.process, None, ema_decay=None)
    assert (to_record(objective.task_settings(), ImageSettings)
            == to_record(direct.task_settings(), ImageSettings))
    assert objective.steps == direct.steps == 3 and objective.guidance is None
    assert objective.grid is not None and objective.solver != source.schedule.solver
    expected = model_part(source.variables)
    held = objective.init(jax.random.PRNGKey(0))
    for tree in (objective.teacher_variables, held[TEACHER]):
        assert jax.tree.structure(tree) == jax.tree.structure(expected)
        for got, want in zip(jax.tree.leaves(tree), jax.tree.leaves(expected), strict=True):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    prompt = ["a red bird"]
    published = objective.build_task(held)(prompt, key=9).host().images
    alone = direct.build_task(direct.init(jax.random.PRNGKey(0)))(prompt, key=9).host().images
    np.testing.assert_array_equal(published, alone)


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
    with pytest.raises(TypeError):
        jax.jvp(reverse_only, (q, k, v), tangents)
    out, tangent = jax.jvp(lambda q, k, v: forward_differentiable(reverse_only, reference, q, k, v),
                           (q, k, v), tangents)
    _, reference_tangent = jax.jvp(reference, (q, k, v), tangents)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(reverse_only(q, k, v)))
    np.testing.assert_array_equal(np.asarray(tangent), np.asarray(reference_tangent))
