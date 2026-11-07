"""Native Wan 2.1 against the actual Diffusers objects it reconstructs, at the release the test extra pins.

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

import contextlib
import functools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tool_support
from flax import linen as nn
from interop_support import (
    assert_transformer_exports,
    channels_last,
    extract_fixture,
    fixture_arrays,
    source_transformer,
)
from reference_error import as_given, assert_as_exact
from reference_runs import assert_arms_as_recorded

from dew.diffusion.process import DenoisingCondition
from dew.interop.diffusion import component_tensors, read_transformer, wan_fields, wan_path
from dew.nn.backbones.wan import WanTransformer
from dew.objectives.diffusion import DiffusionObjective

ROOT = Path(__file__).resolve().parents[1]
CASES = ("published", "variant")


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return extract_fixture(ROOT / "tests/fixtures/wan_transformer.tar.xz",
                           tmp_path_factory.mktemp("wan-transformer"))


@pytest.fixture(scope="module")
def arrays(source):
    return fixture_arrays(source / "wan_transformer.npz")


def load(source, name):
    return source_transformer(source, name, WanTransformer, wan_fields, wan_path)


@pytest.mark.parametrize("name", CASES)
def test_every_published_tensor_maps_and_exports_bit_identical(source, name):
    _, params, layouts = load(source, name)
    assert_transformer_exports(source, name, layouts, params)


SEEDS = 8
"""Every site's declared seeds for check 1, sized by `seeds_for` paired as each check confirmed it."""


def transformer_draw(arrays, name):
    """`draw(seed)` of case `name`: latents, prompt states and a probe of the fixture's shapes, drawn, at the
    fixture's times."""
    shapes = {key: arrays[f"{name}.{key}"].shape for key in ("latents", "context", "probe")}
    times = np.asarray(arrays[f"{name}.times"], np.float32)

    def draw(seed):
        generator = np.random.default_rng([337, seed])
        drawn = {key: generator.standard_normal(shape).astype(np.float32) for key, shape in shapes.items()}
        return ({**drawn, "times": times},)
    return draw


WAN_SEEDS = {"published": 8, "variant": 8, "bf16/published": 8, "bf16/variant": 8}
"""Each recorded site's declared seeds, sized by `seeds_for` paired as its recording confirmed them."""


@functools.cache
def extracted() -> Path:
    """The transformer fixture's archive extracted once, for the recorded sites' arms."""
    import tempfile

    return extract_fixture(ROOT / "tests/fixtures/wan_transformer.tar.xz", Path(tempfile.mkdtemp()))


def flowing(name: str, *, gradients: bool):
    """`dew(dtype, inputs)` of case `name`: Dew's flow at `dtype`, channels first, and with `gradients` those
    of `sum(flow * probe)` with respect to the latents, the prompt states and every parameter by its
    published name, sorted, concatenated, as tools/wan_worker.py gives them."""
    source = extracted()
    config = json.loads((source / name / "transformer" / "config.json").read_text())
    params, layouts = read_transformer(component_tensors(source / name, "transformer"), wan_path)
    tensors = sorted((layout.name.removeprefix("transformer/"), layout) for layout in layouts)

    def dew(dtype, inputs) -> dict:
        model = WanTransformer(**wan_fields(config, dtype=jnp.dtype(dtype).name, attention_impl="xla"))
        held = jax.tree.map(lambda leaf: jnp.asarray(leaf, dtype if dtype == jnp.float64 else leaf.dtype),
                            params)
        times = jnp.asarray(inputs["times"], jnp.float64 if dtype == jnp.float64 else jnp.float32)

        def flow(held, latents, context):
            output = model.apply({"params": held}, jnp.moveaxis(latents, 1, -1), times,
                                 DenoisingCondition(context))
            return jnp.moveaxis(output, -1, 1)

        latents, context = (jnp.asarray(inputs[key], dtype) for key in ("latents", "context"))
        output = flow(held, latents, context)
        if not gradients:
            return {"output": output.astype(jnp.float32 if dtype != jnp.float64 else dtype)}
        probe = jnp.asarray(inputs["probe"], dtype)
        grads = jax.grad(lambda *args: jnp.sum(flow(*args) * probe), argnums=(0, 1, 2))(
            held, latents, context)
        native = [grads[1], grads[2], *(layout.export({"params": grads[0]}) for _, layout in tensors)]
        flat = jnp.concatenate([jnp.ravel(jnp.asarray(leaf)) for leaf in native])
        return {"output": output, "gradients": flat}
    return dew


def pipeline_arms(arms: str):
    """tools/venv_reference.py's `wan-<case>` or `wan-bf16-<case>` site (`arms` the case, or `bf16/<case>`):
    its recording, draw, Dew run and leaves."""
    half = arms.startswith("bf16/")
    name = arms.removeprefix("bf16/")
    stem = f"{name}-bf16" if half else name
    arrays = fixture_arrays(extracted() / "wan_transformer.npz")
    return (ROOT / "tests/fixtures/wan" / f"{stem}.npz", lambda seed: transformer_draw(arrays, name)(seed)[0],
            flowing(name, gradients=not half), ("output",) if half else ("output", "gradients"))


@pytest.mark.parametrize("name", CASES)
def test_the_flow_and_its_gradients_are_as_exact_as_the_source(name):
    """The flow, and the gradients of `sum(flow * probe)` with respect to the latents, the prompt states and
    every parameter (by its published name), each on par with Diffusers' float32 walk over drawn inputs
    (check 1), its float64 twin the truth."""
    assert_arms_as_recorded(pipeline_arms(name), f"wan-{name}", WAN_SEEDS[name], f"{name} flow and gradients")


@pytest.mark.parametrize("name", CASES)
def test_a_bfloat16_flow_is_as_exact_as_the_bfloat16_source(name):
    """In bfloat16, Dew's flow on par with the source's own bfloat16 run, whose time embedder stays in
    float32, over drawn inputs (check 1), its float64 twin the truth: sinusoids taken in bfloat16 land ten to
    twenty times further."""
    assert_arms_as_recorded(pipeline_arms(f"bf16/{name}"), f"wan-bf16-{name}", WAN_SEEDS[f"bf16/{name}"],
                            f"{name} bfloat16 flow", narrow=jnp.bfloat16)


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


@pytest.fixture(scope="module")
def sources(walk):
    """The saved pipeline Diffusers loads in float32, and read back in float64 inside `float64` with its
    UniPC tables widened (the recorder's `float64_scheduler`), with the context that widens a call into it."""
    import torch
    from diffusers import WanPipeline
    from diffusers.schedulers import scheduling_unipc_multistep

    from tools.torch_precision import float64

    with pytest.MonkeyPatch.context() as patched:
        patched.syspath_prepend(str(ROOT / "tools"))
        tool = tool_support.load("diffusers_wan_reference")

    @contextlib.contextmanager
    def widening():
        with float64(), tool.float64_scheduler(scheduling_unipc_multistep):
            yield

    def read(dtype):
        return WanPipeline.from_pretrained(walk[0] / "pipeline", torch_dtype=dtype, local_files_only=True)

    narrow = read(torch.float32)
    with widening():
        wide = read(torch.float64)
    for pipe in (narrow, wide):
        pipe.set_progress_bar_config(disable=True)
    return narrow, wide, widening


def source_arms(sources, run):
    """`run(pipe, dtype, *inputs)` as given over the float32 pipeline, and over the float64 one widened."""
    import torch

    narrow, wide, widening = sources

    @as_given
    def reference(*inputs):
        with torch.no_grad():
            return run(narrow, torch.float32, *inputs)

    @as_given
    def truth(*inputs):
        with widening(), torch.no_grad():
            return run(wide, torch.float64, *inputs)
    return reference, truth


WORDS = ("a", "red", "cat", "on", "mat", "tiny", "boat", "sea", "blue", "small", "dog", "tree")
"""The words a drawn prompt is made of."""


def prompts_of(record, first, second) -> list[str]:
    """A draw's two prompts: the first its words, the second its words before the record's second prompt,
    whose curly quotes, entities beside a `<` and runs of whitespace the source cleans."""
    return [" ".join(WORDS[index] for index in first),
            " ".join(WORDS[index] for index in second) + record["prompts"][1]]


def walk_draw(record):
    """`draw(seed)`: the two prompts' word indices and the starting latents `[2, C, T, H, W]`."""
    shape = (2, 4, (record["frames"] - 1) // 4 + 1, record["height"] // 8, record["width"] // 8)

    def draw(seed):
        generator = np.random.default_rng([353, seed])
        return (generator.integers(0, len(WORDS), 4).astype(np.int32),
                generator.integers(0, len(WORDS), 2).astype(np.int32),
                generator.standard_normal(shape).astype(np.float32))
    return draw


def encoded_by(pipe, prompt: str) -> np.ndarray:
    """The UMT5 states the source's `encode_prompt` writes for `prompt`, 512 positions as the call passes
    (its default is 226)."""
    import torch

    states, _ = pipe.encode_prompt(prompt, do_classifier_free_guidance=False, max_sequence_length=512,
                                   device=torch.device("cpu"))
    return states[0].numpy()


def source_walked(record):
    """The source's own call on a draw, row by row: the latent it ends on, channels last, and the frames it
    decodes."""
    import torch

    def run(pipe, dtype, first, second, latents):
        rows = {"latents": [], "frames": []}
        for row, prompt in enumerate(prompts_of(record, first, second)):
            call = {"prompt": prompt, "height": record["height"], "width": record["width"],
                    "num_frames": record["frames"], "num_inference_steps": record["steps"],
                    "guidance_scale": record["guidance"]}
            start = torch.from_numpy(latents[row:row + 1]).to(dtype)
            walked = pipe(**call, latents=start.clone(), output_type="latent").frames
            rows["latents"].append(channels_last(walked.numpy())[0])
            rows["frames"].append(pipe(**call, latents=start.clone(), output_type="np").frames[0])
        return {key: np.stack(value) for key, value in rows.items()}
    return run


def dew_walked(task, prepare):
    """Dew's walk of a draw from `prepare(first, second, initial)`: its final latents and decoded frames."""
    @as_given
    def run(first, second, latents):
        walked = task(prepare(first, second, channels_last(latents)), key=jax.random.PRNGKey(0)).host()
        return {"latents": np.asarray(walked.latents),
                "frames": np.clip(np.asarray(walked.images) / 2 + 0.5, 0.0, 1.0)}
    return run


def assert_walked_as_the_source(walked, sources, record, label: str) -> None:
    """Dew's walk (`dew_walked`) on par with the source pipeline's own call over drawn prompts and starting
    latents (check 1), the whole pipeline's float64 run the truth. XLA's float32 attention over Wan's 512
    text keys rounds 2.07 times as far from float64 as torch's SDPA does on the same call."""
    reference, truth = source_arms(sources, source_walked(record))
    assert_as_exact(walked, reference, walk_draw(record), label, seeds=SEEDS, truth_fn=truth)


def test_prompt_encoding_matches_the_source_pipeline(pipeline, walk, sources):
    """The conditioner reads what `encode_prompt` reads: the prompt cleaned (its curly quotes, the entities
    ftfy leaves beside a `<` for the source's two unescapes, its runs of whitespace), the UMT5 states of its
    tokens and zeros after them, the whole 512 positions; on par with it over drawn prompts (check 1), its
    float64 run the truth."""
    _, record, _ = walk
    encoder = pipeline.inputs.conditions["conditioning"].encoder
    params = pipeline.variables["encoders"]["conditioning"]

    @as_given
    def encoded(first, second):
        rows = [*prompts_of(record, first, second), ""]
        return np.asarray(encoder.encode(params, encoder.tokenize(rows)).context)

    def encode_prompt(pipe, dtype, first, second):
        return np.stack([encoded_by(pipe, prompt) for prompt in [*prompts_of(record, first, second), ""]])

    def draw(seed):
        return walk_draw(record)(seed)[:2]

    reference, truth = source_arms(sources, encode_prompt)
    found, expected = encoded(*draw(0)), reference(*draw(0))
    assert found.shape == expected.shape == (3, 512, record["transformer"]["text_dim"])
    np.testing.assert_array_equal(found == 0, expected == 0)
    assert_as_exact(encoded, reference, draw, "Wan prompt encoding", seeds=SEEDS, truth_fn=truth)


@pytest.mark.parametrize("load", ["pipeline", "streamed"])
def test_pipeline_walk_matches_the_source(load, walk, sources, request):
    """`Pretrained.load().text_to_image()` samples a clip with the source's own policy, 50 steps guided at
    5.0 over its UniPC grid, and at the recorded step count ends on the source's latent and decodes its
    frames, loaded whole or streamed onto a mesh."""
    pipeline = request.getfixturevalue(load)
    _, record, _ = walk
    task = pipeline.text_to_image()
    assert task.steps == 50 and task.guidance is not None and task.guidance.scale == record["guidance"]
    assert pipeline.inputs.sample.shape == (record["frames"], record["height"], record["width"], 3)

    def prepare(first, second, initial):
        return task.prepare(prompts_of(record, first, second), initial=initial, key=0, steps=record["steps"])

    assert_walked_as_the_source(dew_walked(task, prepare), sources, record, f"{load} Wan walk")


def test_a_text_free_walk_from_the_sources_own_encodings_matches_the_source(walk, sources):
    """A pipeline loaded without its text encoder (`load_diffusion_source(text=False)`) samples from encoded
    conditions alone: handed the UMT5 states the source's own `encode_prompt` writes for each prompt and for
    its empty negative, it walks to the source's latent and decodes its frames, as the prompted walk above
    does. The encodings are Diffusers' and transformers', not Dew's tower's."""
    import torch

    from dew.interop.pretrained import load_diffusion_source

    _, record, _ = walk
    source = load_diffusion_source(str(walk[0] / "pipeline"), dtype="float32", attention_impl="xla",
                                   text=False)
    assert "conditioning" not in source.variables.get("encoders", {})
    task = source.text_to_image()
    narrow = sources[0]

    def prepare(first, second, initial):
        with torch.no_grad():
            states = [encoded_by(narrow, prompt) for prompt in [*prompts_of(record, first, second), ""]]
        given = DenoisingCondition(jnp.asarray(np.stack(states[:-1])))
        negative = DenoisingCondition(jnp.asarray(states[-1][None]))
        return task.prepare(conditions={"conditioning": given}, unconditional={"conditioning": negative},
                            initial=initial, key=0, steps=record["steps"])

    assert_walked_as_the_source(dew_walked(task, prepare), sources, record, "text-free Wan walk")


class Replay(nn.Module):
    """A model that answers each grid point's model time with the output the
    source's scheduler was stepped on there, whatever the sample."""

    timesteps: tuple[int, ...]
    outputs: jax.Array

    def __call__(self, x, time):
        return self.outputs[jnp.argmin(jnp.abs(jnp.asarray(self.timesteps) - time[0]))]


@pytest.mark.parametrize("steps", [10, 50])
def test_the_scheduler_steps_as_the_source_on_the_same_model_outputs(pipeline, walk, sources, steps):
    """The grid and solver `text_to_image()` walks, stepped on drawn model outputs as the source's own UniPC
    scheduler steps on them: the same timesteps, and every step's sample on par with the scheduler's float32
    run over the draws (check 1), its float64 run (its tables widened) the truth. No sample feeds back into a
    model, so this isolates the flow-shifted sigmas, the timesteps and the bh2 update arithmetic. UniPC of
    order 1, or bh1, lands 1e4 to 1e5 times further."""
    import torch
    from diffusers import UniPCMultistepScheduler
    from diffusion_support import walk as step_by_step

    config = json.loads((walk[0] / "pipeline" / "scheduler" / "scheduler_config.json").read_text())
    process, times = pipeline.grid(steps)
    source = UniPCMultistepScheduler.from_config(config)
    source.set_timesteps(steps)
    timesteps = source.timesteps.numpy()
    np.testing.assert_array_equal(np.asarray(process.sampler_schedule.model_time(times[:-1])), timesteps)
    shape = (2, 4, 3, 4, 6)

    def draw(seed):
        generator = np.random.default_rng([359, seed])
        return (generator.standard_normal(shape).astype(np.float32),
                generator.standard_normal((steps, *shape)).astype(np.float32))

    @as_given
    def stepped(x_T, outputs):
        model = Replay(tuple(int(time) for time in timesteps), jnp.asarray(outputs))
        return np.asarray(step_by_step(pipeline.solver, process, model, jnp.asarray(x_T), times))

    def scheduled(pipe, dtype, x_T, outputs):
        scheduler = UniPCMultistepScheduler.from_config(config)
        scheduler.set_timesteps(steps)
        x, latents = torch.from_numpy(x_T).to(dtype), []
        for index, time in enumerate(scheduler.timesteps):
            x = scheduler.step(torch.from_numpy(outputs[index]).to(dtype), time, x).prev_sample
            latents.append(x)
        return torch.stack(latents).numpy()

    reference, truth = source_arms(sources, scheduled)
    assert_as_exact(stepped, reference, draw, f"{steps}-step replay", seeds=SEEDS, truth_fn=truth)


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
    from diffusion_support import trained_step

    from dew.interop.pretrained import Pretrained

    objective = DiffusionObjective(pipeline, unconditional_prob=0.0, ema_decay=None, steps=2)
    _, state = trained_step(objective, clip_batch(objective, walk[1]["prompts"]))

    export = tmp_path / "export"
    pipeline.save(export, variables=state.variables)
    reloaded = Pretrained.load(str(export), dtype="float32", attention_impl="xla")
    assert reloaded.inputs.sample.shape == pipeline.inputs.sample.shape
    jax.tree.map(np.testing.assert_array_equal, reloaded.variables, state.variables)

    # Diffusers' own classes read the export: every component cleanly, the
    # pipeline whole, and its transformer recomputes the trained flow.
    from diffusion_support import assert_export_recomputes

    from tools import diffusers_consumer as consumer

    consumer.assert_components_load(export)
    config = json.loads((export / "transformer" / "config.json").read_text())
    times = np.asarray([500.0], np.float32)
    forward = jax.jit(lambda latents, context: pipeline.model.apply(
        {"params": state.variables["params"]}, latents, jnp.asarray(times), DenoisingCondition(context)))

    def draw(seed):
        """A latent video and a prompt's context of the transformer's widths, their values drawn."""
        rng = np.random.default_rng([7, seed])
        return (rng.standard_normal((1, 2, 4, 6, config["in_channels"])).astype(np.float32),
                rng.standard_normal((1, 5, config["text_dim"])).astype(np.float32))

    def call(model, tensor, latents, context):
        output = model(hidden_states=tensor(np.moveaxis(latents, -1, 1)), timestep=tensor(times),
                       encoder_hidden_states=tensor(context)).sample
        return channels_last(output.numpy())

    assert_export_recomputes(forward, lambda wide: consumer.denoiser(export, call, wide=wide), draw,
                             "trained Wan")


def test_a_wan_lora_starts_as_the_source_and_trains_its_factors_alone(pipeline, walk, sources):
    """An adapter binds the transformer's self- and cross-attention projections,
    never UMT5's; with B at zero it follows the source within its float64 rule, and
    a step moves every B while the base, the text encoder and the VAE stay
    bitwise where they were."""
    import optax

    from dew.batch import FROZEN
    from dew.lora import LoRA
    from dew.training import Trainer

    _, record, _ = walk
    prompts = record["prompts"]
    tuned = pipeline.adapt(LoRA(rank=2, modules=("attn1.to_q", "attn1.to_v", "attn2.to_k", "attn2.to_out.0")),
                           key=0)
    assert {path[1] for path in tuned.adapter.targets} == {f"blocks_{index}" for index in range(2)}
    assert all(path[0] == "params" for path in tuned.adapter.targets)

    def walked(task):
        def prepare(first, second, initial):
            return task.prepare(prompts_of(record, first, second), initial=initial, key=0,
                                steps=record["steps"])
        return dew_walked(task, prepare)

    # With B at zero the adapted walk is on par with the plain one over drawn prompts and latents (check 1),
    # the source pipeline's float64 walk the truth.
    _, truth = source_arms(sources, source_walked(record))
    assert_as_exact(walked(tuned.text_to_image()), walked(pipeline.text_to_image()), walk_draw(record),
                    "zero-adapter Wan walk", seeds=SEEDS, truth_fn=truth)
    objective = DiffusionObjective(tuned, ema_decay=None, unconditional_prob=0.0)
    batch = clip_batch(objective, prompts)
    trainer = Trainer(objective, optax.sgd(1e-1), key=jax.random.key(3))
    initial = trainer.initial_state()
    # The step donates the state it is given, so what it is compared with is read first; the
    # source it started from still reads as loaded after it.
    held, loaded = jax.device_get(initial), jax.device_get(tuned.variables)
    state, _, _, _, accepted, _ = trainer.compile(initial, batch)(initial, batch)
    assert bool(accepted)
    for leaf, before in zip(jax.tree.leaves(tuned.variables), jax.tree.leaves(loaded), strict=True):
        np.testing.assert_array_equal(np.asarray(leaf), before)
    initial = held
    factors = jax.tree_util.tree_leaves_with_path(state.variables["params"])
    assert factors and all(path[-1].key in ("lora_A", "lora_B") for path, _ in factors)
    assert all(bool(jnp.any(leaf)) for path, leaf in factors if path[-1].key == "lora_B")
    for collection in (FROZEN, "encoders", "autoencoder"):
        for before, after in zip(jax.tree.leaves(initial.variables[collection]),
                                 jax.tree.leaves(state.variables[collection]), strict=True):
            np.testing.assert_array_equal(np.asarray(before), np.asarray(after), err_msg=collection)
