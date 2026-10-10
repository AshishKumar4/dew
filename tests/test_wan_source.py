"""Native Wan 2.1 against the actual Diffusers 0.34.0 objects it reconstructs.

`tools/diffusers_wan_reference.py` builds tiny `WanTransformer3DModel`
instances with every parameter moved off its initialization, saves each with
its real config and safetensors, and calls each as `WanPipeline` calls it,
recording the output and, against a fixed cotangent, the gradients of the
latents, the prompt states and every parameter: once in float32 and once in
float64, the truth both float32 runs are measured from. It also saves one
tiny `WanPipeline` over the published configs and walks the unmodified call,
guided at its default scale.

The native transformer takes channels-last latents and Dew's model time,
which is the timestep the source's pipeline passes, and returns the flow the
source returns. Dew in float32 is held to tests/reference_error.py's rule,
within twice the reference's own RMS distance from float64. Observed on CPU
(Dew's RMS error over the reference's): published forward 1.01, gradients
1.01; variant forward 0.96, gradients 0.99; in bfloat16, 1.26 and 0.99
against the source's own bfloat16 run. The truth is float64 throughout:
Dew run in float64 with a float64 rotary table lands 3e-16 (RMS) from it.

The pipeline walk composes the UMT5 encoder, the transformer, UniPC and the
VAE as the source composes them; like the other pipeline walks it is held to
the largest difference over the largest value, 1e-5. Observed: prompt states
3.2e-7, final latents 2.8e-6, decoded frames 2.7e-6.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from interop_support import assert_layout_tensors, extract_fixture, fixture_arrays
from reference_error import assert_as_exact_as_the_reference

from dew.diffusion.process import DenoisingCondition
from dew.interop.diffusion import component_tensors, translate_wan_weights, wan_fields
from dew.nn.backbones.wan import WanTransformer
from dew.objectives.diffusion import DiffusionObjective

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "tests/fixtures/hf/wan-source"
CASES = ("published", "variant")


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return extract_fixture(ROOT / "tests/fixtures/wan_transformer.tar.xz",
                           tmp_path_factory.mktemp("wan-transformer"))


@pytest.fixture(scope="module")
def arrays(source):
    return fixture_arrays(source / "wan_transformer.npz")


def channels_last(value):
    """`[B, C, F, H, W]` to `[B, F, H, W, C]`."""
    return np.moveaxis(np.asarray(value), 1, -1)


def load(source, name):
    config = json.loads((source / name / "transformer" / "config.json").read_text())
    model = WanTransformer(**wan_fields(config, dtype="float32", attention_impl="xla"))
    params, layouts = translate_wan_weights(component_tensors(source / name, "transformer"))
    return model, params, layouts


@pytest.mark.parametrize("name", CASES)
def test_every_published_tensor_maps_and_exports_bit_identical(source, name):
    _, params, layouts = load(source, name)
    tensors = component_tensors(source / name, "transformer")
    assert_layout_tensors(layouts, {"params": params}, tensors, "transformer/")


@pytest.mark.parametrize("name", CASES)
def test_the_flow_and_its_gradients_are_as_exact_as_the_source(source, arrays, name):
    """The flow, and the gradients of `sum(flow * probe)` with respect to the
    latents, the prompt states and every parameter, each within twice the
    float32 source's RMS distance from its float64 run."""
    model, params, layouts = load(source, name)
    latents = jnp.asarray(channels_last(arrays[f"{name}.latents"]))
    times = jnp.asarray(arrays[f"{name}.times"])
    context = jnp.asarray(arrays[f"{name}.context"])
    probe = channels_last(arrays[f"{name}.probe"])

    def flow(params, latents, context):
        return model.apply({"params": params}, latents, times, DenoisingCondition(context))

    output = flow(params, latents, context)
    grad_params, grad_latents, grad_context = jax.grad(
        lambda *args: jnp.sum(flow(*args) * probe), argnums=(0, 1, 2))(params, latents, context)
    assert_as_exact_as_the_reference(np.moveaxis(np.asarray(output), -1, 1), arrays[f"{name}.fp32.output"],
                                     arrays[f"{name}.fp64.output"], f"{name} flow")

    def source_order(precision: str) -> np.ndarray:
        leaves = [arrays[f"{name}.{precision}.grad_latents"], arrays[f"{name}.{precision}.grad_context"]]
        leaves += [arrays[f"{name}.{precision}.grad_param.{layout.name.removeprefix('transformer/')}"]
                   for layout in layouts]
        return np.concatenate([np.ravel(leaf) for leaf in leaves])

    native = [np.moveaxis(np.asarray(grad_latents), -1, 1), np.asarray(grad_context)]
    native += [layout.export({"params": grad_params}) for layout in layouts]
    native_order = np.concatenate([np.ravel(leaf) for leaf in native])
    assert_as_exact_as_the_reference(native_order, source_order("fp32"), source_order("fp64"),
                                     f"{name} gradients")


@pytest.mark.parametrize("name", CASES)
def test_a_bfloat16_flow_is_as_exact_as_the_bfloat16_source(source, arrays, name):
    """In bfloat16, Dew's flow is no further from float64 than the source's
    own bfloat16 run, whose time embedder stays in float32: sinusoids taken
    in bfloat16 land ten to twenty times further."""
    config = json.loads((source / name / "transformer" / "config.json").read_text())
    model = WanTransformer(**wan_fields(config, dtype="bfloat16", attention_impl="xla"))
    params, _ = translate_wan_weights(component_tensors(source / name, "transformer"))
    latents = jnp.asarray(channels_last(arrays[f"{name}.latents"]), jnp.bfloat16)
    context = DenoisingCondition(jnp.asarray(arrays[f"{name}.context"], jnp.bfloat16))
    flow = model.apply({"params": params}, latents, jnp.asarray(arrays[f"{name}.times"]), context)
    assert_as_exact_as_the_reference(np.moveaxis(np.asarray(flow, np.float32), -1, 1),
                                     arrays[f"{name}.bf16.output"], arrays[f"{name}.fp64.output"],
                                     f"{name} bfloat16 flow")


@pytest.mark.parametrize("change", [
    {"image_dim": 1280}, {"added_kv_proj_dim": 5120}, {"pos_embed_seq_len": 514}, {"qk_norm": "rms_norm"},
    {"patch_size": [2, 2]}, {"attention_head_dim": 13},
])
def test_an_unsupported_config_is_refused(source, change):
    config = json.loads((source / "published" / "transformer" / "config.json").read_text())
    wan_fields(config)
    with pytest.raises(ValueError):
        wan_fields({**config, **change})


def test_every_tensor_of_the_published_checkpoint_has_a_native_place():
    """Wan2.1-T2V-1.3B's own config builds the native tree, and every tensor
    its shard index names lands on a leaf of it."""
    config = json.loads((PUBLISHED / "transformer" / "config.json").read_text())
    names = json.loads((PUBLISHED / "transformer" / "diffusion_pytorch_model.safetensors.index.json")
                       .read_text())["weight_map"]
    model = WanTransformer(**wan_fields(config))
    shapes = jax.eval_shape(lambda: model.init(
        jax.random.PRNGKey(0), jnp.zeros((1, 1, 2, 2, config["in_channels"])), jnp.zeros((1,)),
        DenoisingCondition(jnp.zeros((1, 1, config["text_dim"])))))
    leaves = {tuple(key.key for key in path) for path, _ in jax.tree_util.tree_leaves_with_path(shapes)}
    tree, _ = translate_wan_weights({name: np.zeros((1, 1) if name.endswith("weight") else (1,), np.float32)
                                     for name in names})
    placed = {tuple(key.key for key in path) for path, _ in jax.tree_util.tree_leaves_with_path(tree)}
    assert {("params", *path) for path in placed} == leaves


@pytest.fixture(scope="module")
def walk(tmp_path_factory):
    directory = tmp_path_factory.mktemp("wan-pipeline")
    extract_fixture(ROOT / "tests/fixtures/wan_pipeline.tar.xz", directory)
    arrays = fixture_arrays(directory / "wan_pipeline.npz")
    return directory, json.loads((directory / "wan_pipeline.json").read_text()), arrays


@pytest.fixture(scope="module")
def pipeline(walk):
    from dew.interop.pretrained import Pretrained

    return Pretrained.load(str(walk[0] / "pipeline"), dtype="float32", attention_impl="xla")


@pytest.fixture(scope="module")
def streamed(walk):
    """The pipeline loaded onto a mesh, its weights streamed as recipes and
    placed a leaf at a time (tests/test_pipeline_streaming.py)."""
    from dew.interop.pretrained import Pretrained
    from dew.training import MeshSpec

    return Pretrained.load(str(walk[0] / "pipeline"), dtype="float32", attention_impl="xla", mesh=MeshSpec())


def assert_walked_as_the_source(walked, arrays, rows: int) -> None:
    """Each row's final latent and decoded frames by tests/reference_error.py's
    rule, against the source pipeline's float32 walk and its float64 one.

    Observed: latents 1.58 to 1.80, frames 0.95 to 1.61. The walk is the
    recorded 4 steps of the same host-table UniPC the replay below steps,
    which at 4 steps measures 0.85, so the margin is the transformer's.
    Teacher-forced on the walk's own inputs its forwards run 1.1 to 1.66:
    every stage up to the cross-attention's inputs at most 1.05, the
    cross-attention 1.93. XLA's float32 attention over Wan's 512 text keys
    rounds 2.07 times as far from float64 as torch's SDPA does on the same
    call (its math and flash backends both measure 1.0)."""
    frames = np.clip(np.asarray(walked.images) / 2 + 0.5, 0.0, 1.0)
    for row in range(rows):
        latent = (channels_last(arrays[name][None])[0] for name in (f"latents.{row}", f"latents_f64.{row}"))
        assert_as_exact_as_the_reference(np.asarray(walked.latents)[row], *latent, f"latent {row}")
        assert_as_exact_as_the_reference(frames[row], arrays[f"frames.{row}"], arrays[f"frames_f64.{row}"],
                                         f"frames {row}")


def test_prompt_encoding_matches_the_source_pipeline(pipeline, walk):
    """The conditioner reads what `encode_prompt` reads: the prompt cleaned
    (its curly quotes, the entities ftfy leaves beside a `<` for the source's
    two unescapes, its runs of whitespace), the UMT5 states of its tokens and
    zeros after them, the whole 512 positions."""
    _, record, arrays = walk
    encoder = pipeline.inputs.conditions["conditioning"].encoder
    prompts = [*record["prompts"], ""]
    params = pipeline.variables["encoders"]["conditioning"]
    context = encoder.encode(params, encoder.tokenize(prompts)).context
    for row in range(len(prompts)):
        expected = arrays[f"context.{row}"]
        assert context[row].shape == expected.shape == (512, record["transformer"]["text_dim"])
        np.testing.assert_array_equal(np.asarray(context[row]) == 0, expected == 0)
        assert_as_exact_as_the_reference(np.asarray(context[row]), expected, arrays[f"context_f64.{row}"],
                                         f"prompt {row}")


@pytest.mark.parametrize("load", ["pipeline", "streamed"])
def test_pipeline_walk_matches_the_source(load, walk, request):
    """`Pretrained.load().text_to_image()` samples a clip with the source's
    own policy, 50 steps guided at 5.0 over its UniPC grid, and at the
    recorded step count ends on the source's latent and decodes its frames,
    loaded whole or streamed onto a mesh."""
    pipeline = request.getfixturevalue(load)
    _, record, arrays = walk
    task = pipeline.text_to_image()
    assert task.steps == 50 and task.guidance is not None and task.guidance.scale == record["guidance"]
    assert pipeline.inputs.sample.shape == (record["frames"], record["height"], record["width"], 3)
    walked = task(task.prepare(record["prompts"], initial=channels_last(arrays["x_T"]), key=0,
                               steps=record["steps"]), key=jax.random.PRNGKey(0)).host()
    assert_walked_as_the_source(walked, arrays, len(record["prompts"]))


def test_a_text_free_walk_from_the_sources_own_encodings_matches_the_source(walk):
    """A pipeline loaded without its text encoder (`load_diffusion_source(
    text=False)`) samples from encoded conditions alone: handed the UMT5
    states the source's own `encode_prompt` wrote for each prompt and for
    its empty negative, it walks from the recorded latents to the source's
    latent and decodes its frames, as the prompted walk above does. The
    encodings are Diffusers' and transformers', not Dew's tower's."""
    from dew.interop.pretrained import load_diffusion_source

    _, record, arrays = walk
    source = load_diffusion_source(str(walk[0] / "pipeline"), dtype="float32", attention_impl="xla",
                                   text=False)
    assert "conditioning" not in source.variables.get("encoders", {})
    task = source.text_to_image()
    rows = len(record["prompts"])
    given = DenoisingCondition(jnp.asarray(np.stack([arrays[f"context.{row}"] for row in range(rows)])))
    negative = DenoisingCondition(jnp.asarray(arrays[f"context.{rows}"][None]))
    prepared = task.prepare(conditions={"conditioning": given}, unconditional={"conditioning": negative},
                            initial=channels_last(arrays["x_T"]), key=0, steps=record["steps"])
    assert_walked_as_the_source(task(prepared, key=jax.random.PRNGKey(0)).host(), arrays, rows)


class Replay(nn.Module):
    """A model that answers each grid point's model time with the output the
    source's scheduler was stepped on there, whatever the sample."""

    timesteps: tuple[int, ...]
    outputs: jax.Array

    def __call__(self, x, time):
        return self.outputs[jnp.argmin(jnp.abs(jnp.asarray(self.timesteps) - time[0]))]


@pytest.mark.parametrize("steps", [10, 50])
def test_the_scheduler_steps_as_the_source_on_the_same_model_outputs(pipeline, walk, steps):
    """The grid and solver `text_to_image()` walks, stepped on the outputs the
    source's own UniPC scheduler was stepped on: the same timesteps, and a
    sample after every step as close to the scheduler's float64 run as its
    float32 run is. No sample feeds back into a model, so this isolates the
    flow-shifted sigmas, the timesteps and the bh2 update arithmetic.
    Observed (Dew's RMS distance from float64 over the source's): 1.01 at
    10 steps, 1.86 at 50; the sigma and alpha tables are bit-identical to the
    source's float32 ones. UniPC of order 1, or bh1, lands 1e4 to 1e5 times
    further."""
    from diffusion_support import walk as step_by_step

    _, _, arrays = walk
    process, times = pipeline.task.grid(steps)
    timesteps = arrays[f"replay.{steps}.fp32.timesteps"]
    np.testing.assert_array_equal(np.asarray(process.sampler_schedule.model_time(times[:-1])), timesteps)
    model = Replay(tuple(int(time) for time in timesteps), jnp.asarray(arrays[f"replay.{steps}.outputs"]))
    latents = step_by_step(pipeline.schedule.solver, process, model,
                           jnp.asarray(arrays[f"replay.{steps}.x_T"]), times)
    assert_as_exact_as_the_reference(np.asarray(latents), arrays[f"replay.{steps}.fp32.latents"],
                                     arrays[f"replay.{steps}.fp64.latents"], f"{steps}-step replay")


def test_a_clip_the_vae_cannot_decode_whole_is_refused(walk):
    from dew.interop.pretrained import load_diffusion_source

    with pytest.raises(ValueError, match="1 \\+ 4k frames"):
        load_diffusion_source(str(walk[0] / "pipeline"), dtype="float32", size=(8, 32, 48))
    shorter = load_diffusion_source(str(walk[0] / "pipeline"), dtype="float32", size=(5, 16, 32))
    assert shorter.inputs.sample.shape == (5, 16, 32, 3)


def clip_batch(objective, prompts):
    """One clip per device, a fixed ramp of pixels, and the prompts in turn."""
    rows, (frames, height, width, channels) = jax.device_count(), objective.inputs.sample.shape
    pixels = np.arange(frames * height * width * channels, dtype=np.int64) % 251
    return {"video": np.tile(pixels.astype(np.uint8).reshape(1, frames, height, width, channels),
                             (rows, 1, 1, 1, 1)),
            **objective.inputs.tokenize([prompts[row % len(prompts)] for row in range(rows)])}


def test_a_trained_wan_step_exports_and_reloads(pipeline, walk, tmp_path):
    """A denoising step over the whole source: the VAE encodes the clip, UMT5
    the prompt, and the transformer takes the gradient. The text encoder and
    the VAE are state, so every one of their leaves survives the step, and
    the export reloads, its clip geometry with it, to the same forward."""
    import optax

    from dew.interop.pretrained import Pretrained
    from dew.objectives import Step
    from dew.training import Trainer

    objective = DiffusionObjective(pipeline, unconditional_prob=0.0, ema_decay=None, steps=2)
    batch = clip_batch(objective, walk[1]["prompts"])
    trainer = Trainer(objective, optax.sgd(1e-2), key=jax.random.PRNGKey(3))
    initial = trainer.initial_state()
    fixed = Step(jnp.asarray(0), jax.random.PRNGKey(5), None)

    def value(params) -> float:
        loss, _ = objective.loss(params, batch, fixed)
        return float(loss.total / loss.mass)

    before = value(initial.variables)
    state, _, _, _, accepted = trainer.compile(initial, batch)(initial, batch)
    assert bool(accepted)
    assert value(state.variables) < before
    assert not np.allclose(state.variables["params"]["proj_out"]["kernel"],
                           initial.variables["params"]["proj_out"]["kernel"])
    for held in ("encoders", "autoencoder"):
        for got, want in zip(jax.tree.leaves(state.variables[held]), jax.tree.leaves(initial.variables[held]),
                             strict=True):
            np.testing.assert_array_equal(got, want)

    export = tmp_path / "export"
    pipeline.save(export, variables=state.variables)
    reloaded = Pretrained.load(str(export), dtype="float32", attention_impl="xla")
    assert reloaded.inputs.sample.shape == pipeline.inputs.sample.shape
    assert value(reloaded.variables) == value(state.variables)

    # Diffusers' own classes read the export: every component cleanly, the
    # pipeline whole, and its transformer recomputes the trained flow.
    from reference_error import assert_as_exact_as_the_reference

    from tools import diffusers_consumer as consumer

    consumer.assert_components_load(export)
    config = json.loads((export / "transformer" / "config.json").read_text())
    rng = np.random.default_rng(7)
    latents = rng.standard_normal((1, 2, 4, 6, config["in_channels"])).astype(np.float32)
    context = rng.standard_normal((1, 5, config["text_dim"])).astype(np.float32)
    times = np.asarray([500.0], np.float32)
    trained = pipeline.model.apply({"params": state.variables["params"]}, jnp.asarray(latents),
                                   jnp.asarray(times), DenoisingCondition(jnp.asarray(context)))

    def call(model, tensor):
        output = model(hidden_states=tensor(np.moveaxis(latents, -1, 1)), timestep=tensor(times),
                       encoder_hidden_states=tensor(context)).sample
        return channels_last(output.numpy())

    assert_as_exact_as_the_reference(np.asarray(trained), consumer.denoiser_prediction(export, call),
                                     consumer.denoiser_prediction(export, call, wide=True), "trained Wan")


def test_a_wan_lora_starts_as_the_source_and_trains_its_factors_alone(pipeline, walk):
    """An adapter binds the transformer's self- and cross-attention projections,
    never UMT5's; with B at zero it samples exactly what the source does, and
    a step moves every B while the base, the text encoder and the VAE stay
    bitwise where they were."""
    import optax

    from dew.lora import LoRA
    from dew.objectives.base import FROZEN
    from dew.training import Trainer

    prompts = walk[1]["prompts"]
    tuned = pipeline.adapt(LoRA(rank=2, modules=("attn1.to_q", "attn1.to_v", "attn2.to_k", "attn2.to_out.0")),
                           key=0)
    assert {path[1] for path in tuned.adapter.targets} == {f"blocks_{index}" for index in range(2)}
    assert all(path[0] == "params" for path in tuned.adapter.targets)

    def sampled(task):
        return np.asarray(task(prompts, steps=2, key=1).host().images)

    np.testing.assert_array_equal(sampled(tuned.text_to_image()), sampled(pipeline.text_to_image()))
    objective = DiffusionObjective(tuned, ema_decay=None, unconditional_prob=0.0)
    batch = clip_batch(objective, prompts)
    trainer = Trainer(objective, optax.sgd(1e-1), key=jax.random.key(3))
    initial = trainer.initial_state()
    state, _, _, _, accepted = trainer.compile(initial, batch)(initial, batch)
    assert bool(accepted)
    factors = jax.tree_util.tree_leaves_with_path(state.variables["params"])
    assert factors and all(path[-1].key in ("lora_A", "lora_B") for path, _ in factors)
    assert all(bool(jnp.any(leaf)) for path, leaf in factors if path[-1].key == "lora_B")
    for collection in (FROZEN, "encoders", "autoencoder"):
        for before, after in zip(jax.tree.leaves(initial.variables[collection]),
                                 jax.tree.leaves(state.variables[collection]), strict=True):
            np.testing.assert_array_equal(np.asarray(before), np.asarray(after), err_msg=collection)
