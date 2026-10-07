"""LADD with ADD's distillation term: StyleGAN-T's head, one step against
the papers' equations on StyleGAN-T's heads (`tools/ladd_reference.py`),
each side's gradient from its own loss, and a run config over a saved
teacher."""

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from diffusion_stubs import batch_for
from flax import linen as nn
from reference_error import assert_as_exact_as_the_reference

import dew
from dew.checkpoints import Checkpoints
from dew.config import ModelConfig, TrainerConfig
from dew.data import TFDSImages
from dew.diffusion.presets import Flow
from dew.nn.backbones import SimpleDiT
from dew.objectives.base import Step
from dew.objectives.diffusion import (
    AdversarialDistillation,
    AdversarialDistillationObjective,
    DiffusionRunConfig,
    TextCondition,
)
from dew.objectives.diffusion.adversarial import Head
from dew.objectives.diffusion.objective import DISCRIMINATOR, SPECTRAL, TEACHER
from dew.sampling import Consistency, Euler, TextToImage
from dew.training import Trainer

HEAD = np.load(Path(__file__).resolve().parent / "fixtures" / "stylegan_t" / "head.npz")


def test_a_head_is_stylegan_ts_at_grid_height_one():
    """StyleGAN-T's DiscHead in training mode (tools/stylegan_t_reference.py)
    on 16 sequences of 12 tokens: LADD's 2D head over a grid of height one,
    with a (1, 9) kernel, from the same weights and spectral-norm vectors.
    Both run in float64, so the gap is a few roundings of O(1) logits."""
    with jax.enable_x64(new_val=True):
        def conv(name):
            return {"kernel": jnp.asarray(HEAD[f"{name}.kernel"]), "bias": jnp.asarray(HEAD[f"{name}.bias"])}

        def norm(name):
            return {"weight": jnp.asarray(HEAD[f"{name}.weight"]), "bias": jnp.asarray(HEAD[f"{name}.bias"])}

        condition_width = HEAD["c"].shape[-1]
        params = {"block_0": {"conv": conv("main.0.0"), "norm": norm("main.0.1")},
                  "block_1": {"conv": conv("main.1.fn.0"), "norm": norm("main.1.fn.1")},
                  "cls": conv("cls"),
                  "cmapper_weight": jnp.asarray(HEAD["cmapper.weight"]) * np.sqrt(condition_width),
                  "cmapper_bias": jnp.asarray(HEAD["cmapper.bias"])}
        spectral = {"block_0": {"conv": {"u": jnp.asarray(HEAD["main.0.0.u"])}},
                    "block_1": {"conv": {"u": jnp.asarray(HEAD["main.1.fn.0.u"])}},
                    "cls": {"u": jnp.asarray(HEAD["cls.u"])}}
        logits, updated = Head(kernel_size=(1, 9)).apply(
            {"params": params, SPECTRAL: spectral}, jnp.asarray(HEAD["x"]), jnp.asarray(HEAD["c"]),
            update=True, batch={}, mutable=[SPECTRAL])
        np.testing.assert_allclose(np.asarray(logits), HEAD["logits"], rtol=1e-10, atol=1e-12)
        assert not np.allclose(np.asarray(updated[SPECTRAL]["cls"]["u"]), HEAD["cls.u"])


STEP = np.load(Path(__file__).resolve().parent / "fixtures" / "ladd" / "step.npz")


class Layer(nn.Module):
    @nn.compact
    def __call__(self, tokens):
        return jnp.tanh(tokens @ self.param("kernel", nn.initializers.zeros, (tokens.shape[-1],) * 2))


class Tokens(nn.Module):
    """`tools/ladd_reference.py`'s stand-in velocity network at Dew's model
    time (t times 1000): the sample as one token plus the time, two layers
    whose tokens the heads read, and a head back."""

    width: int

    @nn.compact
    def __call__(self, x, time):
        b = x.shape[0]
        flat = x.reshape(b, 1, -1)
        tokens = (flat @ self.param("embed", nn.initializers.zeros, (flat.shape[-1], self.width))
                  + (time / 1000).reshape(-1, 1, 1)
                  * self.param("time", nn.initializers.zeros, (self.width,)))
        layer_b = Layer(name="layer_b")(Layer(name="layer_a")(tokens))
        head = self.param("head", nn.initializers.zeros, (self.width, flat.shape[-1]))
        return (layer_b @ head).reshape(x.shape)


def unflattened(prefix: str) -> dict:
    """The fixture's arrays under `prefix/` as a nested tree."""
    tree: dict = {}
    for key in STEP.files:
        if key.startswith(prefix + "/"):
            *path, leaf = key[len(prefix) + 1:].split("/")
            node = tree
            for name in path:
                node = node.setdefault(name, {})
            node[leaf] = jnp.asarray(STEP[key], jnp.float32)
    return tree


def test_one_step_is_the_papers_equations_on_stylegan_ts_heads():
    """One LADD step against `tools/ladd_reference.py`'s oracle, which runs
    StyleGAN-T's `DiscHead` and DiT's timestep embedding as published inside
    the papers' equations, on the draws `AdversarialDistillationObjective.loss`
    makes: the student's x_0 at its drawn time, the logit-normal renoising,
    the teacher's tokens after both layers, the hinge losses meaned over
    every head's logits, R1 at its real input, ADD's (1 - s)-weighted
    distillation, and the heads' spectral norms at Dew's documented cadence
    (the real pass iterates and keeps its `u`; the held, fake and R1 passes
    iterate from it and keep nothing). The loss within 1e-6 of the oracle's
    float64 run; every gradient, the student's through the fake pass and the
    distillation and the heads' through the hinge and R1, and the kept
    `u`s held to it by the float64 rule. The residual block's kernel is 1
    here, as one token is all a stand-in's grid holds;
    test_a_head_is_stylegan_ts_at_grid_height_one holds the kernel of 9."""
    import json

    from dew.inputs import Field, InputSpec

    settings = json.loads(str(STEP["settings"]))
    pixels = STEP["pixels"]
    layers = {name: {"kernel": jnp.asarray(STEP[f"teacher/{name}"], jnp.float32)}
              for name in ("layer_a", "layer_b")}
    teacher = {"params": unflattened("teacher") | layers}
    model = Tokens(settings["width"])
    task = AdversarialDistillationObjective(
        model, Flow()(), InputSpec(Field("image", pixels.shape[1:])),
        AdversarialDistillation(feature_layers=settings["layers"], student_times=settings["student_times"],
                                renoise_times=tuple(settings["renoise_times"]),
                                distillation_weight=settings["distillation_weight"],
                                r1_weight=settings["r1_weight"], cmap_dim=settings["cmap_dim"],
                                kernel_size=(1, 1)),
        teacher=model, teacher_variables=teacher, time_features=settings["time_features"], ema_decay=None)
    variables = task.init(jax.random.PRNGKey(0))
    student = unflattened("student")
    for name in ("layer_a", "layer_b"):
        student[name] = {"kernel": student[name]}
    params = {**student, DISCRIMINATOR: unflattened("heads")}
    variables = {**variables, SPECTRAL: unflattened("initial")}
    step = Step(step=jnp.asarray(0), key=jax.random.key(settings["key"]), ema=None)

    def loss(params):
        return task.scalar_loss({**variables, "params": params}, {"image": pixels}, step)

    (value, aux), gradient = jax.value_and_grad(loss, has_aux=True)(params)
    np.testing.assert_allclose(float(value), float(STEP["loss_f64"]), rtol=1e-6)
    student = {name: gradient[name]["kernel"] if name.startswith("layer") else gradient[name]
               for name in ("embed", "time", "head", "layer_a", "layer_b")}
    # Each tree is held whole, its leaves in one vector: a few entries' root
    # mean square is no estimate of rounding, and the two biases a batch norm
    # follows have a gradient of exactly zero, rounding alone in both runs.
    for label, tree, prefix in (("student", student, "grad/student"),
                                ("head 0", gradient[DISCRIMINATOR]["head_0"], "grad/heads/head_0"),
                                ("head 1", gradient[DISCRIMINATOR]["head_1"], "grad/heads/head_1"),
                                ("spectral u", aux.variables[SPECTRAL], "spectral")):
        leaves = jax.tree_util.tree_flatten_with_path(tree)[0]
        keys = ["/".join([prefix, *(entry.key for entry in path)]) for path, _ in leaves]
        assert_as_exact_as_the_reference(
            np.concatenate([np.ravel(leaf) for _, leaf in leaves]),
            np.concatenate([np.ravel(STEP[key]) for key in keys]),
            np.concatenate([np.ravel(STEP[f"{key}_f64"]) for key in keys]), label)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("ladd")
    teacher = DiffusionRunConfig(
        model=ModelConfig("simple_dit",
                          {"patch_size": 2, "emb_features": 16, "num_layers": 2, "num_heads": 2,
                           "dtype": "float32", "attention_impl": "xla"}),
        data=TFDSImages(image_size=4), preset=Flow(), solver=Euler(), guidance=None,
        sampling_steps=2, ema_decay=None, val_metrics=(), trainer=TrainerConfig(checkpoint_dir=str(root)),
        text=TextCondition(encoder="char_table", checkpoint="char_table"))
    objective = teacher.build()
    trainer = Trainer(objective, optax.adam(1e-2), key=jax.random.PRNGKey(0))
    state = trainer.initial_state()
    batch = batch_for(objective, 4)
    state, *_ = trainer.compile(state, batch)(state, batch)
    checkpoints = Checkpoints(str(root / "teacher"))
    checkpoints.save(1, state, None, artifact=objective.inference_record())
    checkpoints.wait()
    teacher.save(str(root / "teacher"))
    student = dataclasses.replace(teacher, solver=Consistency(), mode=AdversarialDistillation(
        teacher=str(root / "teacher"), feature_layers=("dit_block_0", "dit_block_1"), cmap_dim=8,
        kernel_size=(3, 3)))
    return student, batch


def test_each_side_trains_on_its_own_loss(runs):
    """The heads' gradient is the discriminator loss's alone and the
    student's the generator and distillation losses' alone; the teacher is
    not a parameter."""
    config, batch = runs
    task = config.build()
    assert isinstance(task, AdversarialDistillationObjective)
    params = task.init(jax.random.PRNGKey(1))
    assert TEACHER not in params["params"]
    step = Step(step=jnp.asarray(0), key=jax.random.PRNGKey(2), ema=None)

    def part(name):
        def loss(tree):
            total, aux = task.loss({**params, "params": tree}, batch, step)
            if name is None:
                return total.total
            gamma = task.distillation.r1_weight
            return sum(aux.metrics[key] * (gamma if key == "r1" else 1) for key in name) * total.mass
        return jax.grad(loss)(params["params"])

    everything = part(None)
    critic, student = part(("discriminator", "r1")), part(("generator", "distillation"))
    for got, want in zip(jax.tree.leaves(everything[DISCRIMINATOR]), jax.tree.leaves(critic[DISCRIMINATOR]),
                         strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-7)
    for name in everything:
        if name != DISCRIMINATOR:
            for got, want in zip(jax.tree.leaves(everything[name]), jax.tree.leaves(student[name]),
                                 strict=True):
                np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-7)
    assert float(sum(jnp.abs(leaf).sum() for leaf in jax.tree.leaves(critic[DISCRIMINATOR]))) > 0


def test_a_saved_student_samples_in_one_step(runs, tmp_path):
    config, batch = runs
    task = config.build()
    trainer = Trainer(task, optax.adam(1e-3), key=jax.random.PRNGKey(3))
    state = trainer.initial_state()
    state, *_ = trainer.compile(state, batch)(state, batch)
    checkpoints = Checkpoints(str(tmp_path / "student"))
    checkpoints.save(1, state, None, artifact=task.inference_record())
    checkpoints.wait()
    config.save(str(tmp_path / "student"))
    restored = TextToImage.from_run(str(tmp_path / "student"))
    assert TEACHER not in restored.variables and DISCRIMINATOR not in restored.variables["params"]
    expected = task.pipeline(state, ema=False)(["a red bird"], key=9).host().images
    np.testing.assert_array_equal(restored(["a red bird"], key=9).host().images, expected)
    # The run records the "ladd" objective, whose saved task it inherits.
    front = dew.pipeline(str(tmp_path / "student"))
    assert isinstance(front, TextToImage)
    np.testing.assert_array_equal(front(["a red bird"], key=9).host().images, expected)


def test_a_lora_student_distills_beside_its_whole_teacher():
    """The teacher runs through its own model, so a LoRA student over its
    weights distills: a step trains its factors and the heads, and the
    weights it froze stay the teacher's."""
    from dew.inputs import Field, InputSpec
    from dew.lora import LoRA
    from dew.objectives.base import FROZEN

    model = SimpleDiT(patch_size=2, emb_features=16, num_layers=1, num_heads=2)
    teacher = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 4, 4, 3)), jnp.ones((1,)))
    held = jax.tree.map(np.asarray, teacher["params"])  # the step donates these buffers
    adapter = LoRA(rank=2, modules=("ada_proj", "final_proj")).apply(model, teacher, key=1)
    task = AdversarialDistillationObjective(
        adapter.model, Flow()(), InputSpec(Field("image", (4, 4, 3))),
        AdversarialDistillation(feature_layers=("dit_block_0",), cmap_dim=4), teacher=model,
        teacher_variables=teacher, ema_decay=None, variables={**adapter.variables, "encoders": {}})
    assert [program.trained for program in task.program_key()] == [True, False]
    batch = {"image": np.asarray(jax.random.randint(jax.random.PRNGKey(1), (8, 4, 4, 3), 0, 256), np.uint8)}
    trainer = Trainer(task, optax.adam(1e-2), key=jax.random.PRNGKey(4))
    state = trainer.initial_state()
    state, *_ = trainer.compile(state, batch)(state, batch)
    assert {path[-1].key for path, _ in jax.tree_util.tree_flatten_with_path(state.variables["params"])[0]
            if path[0].key != DISCRIMINATOR} == {"lora_A", "lora_B"}
    for got, want in zip(jax.tree.leaves(state.variables[FROZEN]), jax.tree.leaves(held), strict=True):
        np.testing.assert_array_equal(np.asarray(got), want)
    assert np.any(np.asarray(state.variables["params"]["output"]["final_proj"]["lora_B"]) != 0)
