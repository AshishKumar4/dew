"""Native SD3 against the actual Diffusers 0.34.0 objects it reconstructs.

`tools/diffusers_sd3_reference.py` builds tiny `SD3Transformer2DModel`
instances, saves each with its real config and safetensors, and records the
forward, the vector-Jacobian products of the latent, the text tokens and the
pooled vector, and the gradient of every parameter. The same tool records the
actual `FlowMatchEulerDiscreteScheduler` grids, including the mu a source
pipeline computes for a resolution and the sigma seed a Flux pipeline hands
it, with the Euler trajectory and its input gradient.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tool_support
from arm_arithmetic import float64_held
from arm_forms import dew_path
from flax import linen as nn
from interop_support import extract_fixture, scaled_gap as relative_gap
from model_support import attending
from reference_error import FEWEST_SEEDS, as_given, assert_as_exact, torch_with_gradients, with_gradients

from dew.diffusion.schedules.source import SourceSchedule
from dew.interop.diffusion import component_tensors, sd3_fields, translate_sd3_weights
from dew.nn.backbones.sd3 import SD3Transformer
from dew.nn.inputs import DenoisingCondition
from dew.nn.kernels import generation
from dew.sampling import sample

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return extract_fixture(ROOT / "tests/fixtures/sd3_source.tar.xz", tmp_path_factory.mktemp("sd3-source"))


def transformer_case(directory: Path, arrays, name: str):
    """The native model, its loaded variables and the case's inputs."""
    config = json.loads(str(arrays[f"{name}.config"]))
    model = SD3Transformer(**sd3_fields(config))
    params, buffers, layouts = translate_sd3_weights(
        component_tensors(directory / name, "transformer"))
    latent = jnp.asarray(arrays[f"{name}.latent"]).transpose(0, 2, 3, 1)
    condition = DenoisingCondition(jnp.asarray(arrays[f"{name}.context"]),
                                   jnp.asarray(arrays[f"{name}.pooled"]))
    times = jnp.asarray(arrays[f"{name}.times"])

    def forward(params, latent, context, pooled):
        return model.apply({"params": params, "buffers": buffers}, latent, times,
                           DenoisingCondition(context, pooled))

    return forward, params, buffers, layouts, latent, condition


CASES = ("plain", "qknorm", "dual", "rect_cropped", "deep")

CASE_SEEDS = {"plain": 64, "qknorm": 128, "dual": 128, "rect_cropped": 64, "deep": 256}
"""Each transformer's declared check-1 seeds, sized paired by `seeds_for` on armada (11-14)."""


@attending("xla")
@pytest.mark.parametrize("name", CASES)
def test_native_sd3_matches_the_source_forward_and_every_gradient(name, source):
    """One published transformer's own tensors, read through the native model, against the source's
    `SD3Transformer2DModel` loaded from the same files: the output and the latent's, the text tokens', the
    pooled vector's and every parameter's gradients (by its published name) under a drawn probe, over
    drawn latents, tokens and pooled vectors at the case's timesteps, on par (check 1), the source's
    float64 twin the truth (ledger 705). The variants are the ones whose wiring differs: plain, `qk_norm`,
    SD3.5's dual attention with its ninefold modulation, a rectangular latent whose patch grid is smaller
    than the position buffer, and a deeper stack."""
    import torch
    from diffusers import SD3Transformer2DModel

    from _dew_torch.precision import float64, float64_twin

    with np.load(source / "sd3_transformer.npz") as arrays:
        forward, params, buffers, layouts, _, _ = transformer_case(source, arrays, name)
        fixture = {key: np.asarray(arrays[f"{name}.{key}"], np.float32)
                   for key in ("latent", "context", "pooled", "times", "probe")}
        # The stored buffer is the source's, perturbed away from its own
        # sin/cos initializer, so a model that rebuilt it would not match.
        np.testing.assert_array_equal(np.asarray(buffers["pos_embed"]), arrays[f"{name}.position_buffer"])
    narrow = SD3Transformer2DModel.from_pretrained(source / name / "transformer",
                                                   local_files_only=True).eval()
    wide = float64_twin(narrow)
    # Every parameter's layout; the position buffer's (held above) has no gradient.
    buffers = {key for key, _ in narrow.named_buffers()}
    tensors = {layout.name.removeprefix("transformer/"): layout for layout in layouts
               if layout.name.removeprefix("transformer/") not in buffers}
    assert set(tensors) == {key for key, _ in narrow.named_parameters()}
    weights = {id(module): {key: value.detach().numpy() for key, value in module.named_parameters()}
               for module in (narrow, wide)}

    def draw(seed):
        generator = np.random.default_rng([711, seed])
        drawn = {key: generator.standard_normal(fixture[key].shape).astype(np.float32)
                 for key in ("latent", "context", "pooled", "probe")}
        return ({**drawn, "times": fixture["times"]},)

    @jax.jit
    def walk(latent, context, pooled, params, probe):
        def channels_first(latent, context, pooled, params):
            out = forward(params, jnp.transpose(latent, (0, 2, 3, 1)), context, pooled)
            return jnp.transpose(out, (0, 3, 1, 2))
        return with_gradients(channels_first)(latent, context, pooled, params, probe)

    @as_given
    def native(inputs):
        out, grad_latent, grad_context, grad_pooled, grad_params = walk(
            inputs["latent"], inputs["context"], inputs["pooled"], params, inputs["probe"])
        return out, grad_latent, grad_context, grad_pooled, {
            tensor: np.asarray(layout.export({"params": grad_params})) for tensor, layout in tensors.items()}

    def reference_walk(module, drawn):
        dtype = torch.from_numpy(np.asarray(drawn["latent"])).dtype
        timestep = torch.tensor(np.asarray(drawn["times"]), dtype=dtype)

        def source_forward(latent, context, pooled, params):
            return torch.func.functional_call(module, params, (), {
                "hidden_states": latent, "encoder_hidden_states": context, "pooled_projections": pooled,
                "timestep": timestep, "return_dict": False})[0]

        return torch_with_gradients(source_forward)(drawn["latent"], drawn["context"], drawn["pooled"],
                                                    weights[id(module)], drawn["probe"])

    @as_given
    def truth(inputs):
        with float64():
            return reference_walk(wide, inputs)

    assert_as_exact(native, as_given(lambda inputs: reference_walk(narrow, inputs)), draw, f"SD3 {name}",
                    seeds=CASE_SEEDS[name], truth_fn=truth)


class Velocity(nn.Module):
    """The closed-form velocity of the optimal transport path for Gaussian
    data, which is the model both sides of the flow comparison run."""

    std: float

    @nn.compact
    def __call__(self, x, temb):
        sigma = temb.reshape((-1,) + (1,) * (x.ndim - 1)) / 1000.0
        variance = (1 - sigma) ** 2 * self.std ** 2 + sigma ** 2
        return x * sigma / variance - x * (1 - sigma) * self.std ** 2 / variance


def flow_reference(source):
    with np.load(source / "sd3_flow.npz") as arrays:
        return {key: arrays[key] for key in arrays.files}


FLOW_CASES = ("flow.plain", "flow.shift3", "flow.dynamic", "flow.linear_dynamic",
              "flow.terminal", "flow.karras", "flow.exponential")


FLOW_SEEDS = 32
"""Each flow schedule site's declared check-1 seeds."""
FLOW_SHAPE = (4, 3, 8)
FLOW_TOKENS = (256, 4096)
"""The latent token counts a flow seed draws between: SD3's 256 patches at 512 pixels to Flux's 4096 packed
positions at 1024."""


def flow_draw(seed: int) -> tuple[np.ndarray, ...]:
    """A flow seed's inputs: x_T, the cotangent its last latent is pulled back by, the latent token count
    a resolution's shift reads (`FLOW_TOKENS`) and the sigmas' origin, 0 the scheduler's own and 1 the
    `linspace(1, 1/N, N)` a Flux pipeline hands it."""
    generator = np.random.default_rng([0xF10, seed])
    return (generator.standard_normal(FLOW_SHAPE).astype(np.float32),
            generator.standard_normal(FLOW_SHAPE).astype(np.float32),
            np.int32(generator.integers(FLOW_TOKENS[0], FLOW_TOKENS[1] + 1)), np.int32(generator.integers(2)))


def source_flow(config: dict, steps: int, std: float, dtype, x_T, cotangent, tokens, origin):
    """`FlowMatchEulerDiscreteScheduler` under `config` at `steps` (the unmodified library's at float32, its
    independent oracle's at float64, tools/float64_schedulers.py), its sigmas set as SD3's pipeline (`mu`
    alone, from `calculate_shift` at `tokens`, where the file shifts dynamically) or a Flux pipeline
    (`origin` 1: its linspace beside that mu) sets them, and the recording's Euler walk over the closed-form
    velocity at `dtype` (tools/scheduler_cases.py's `flow_arguments`, `flow_walk`): the sigmas, the
    model times, every latent and the last one's VJP. The velocity reads the timestep the scheduler hands
    its model, as a transformer reads it and as Dew's model reads the model time it is handed."""
    import torch
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

    cases = tool_support.load("scheduler_cases")
    library = FlowMatchEulerDiscreteScheduler.from_config(config)
    assert isinstance(library, FlowMatchEulerDiscreteScheduler)  # `from_config`'s stub widens it.
    scheduler = (tool_support.scheduler_oracles().oracle_of(library) if dtype == torch.float64
                 else library)
    arguments = cases.flow_arguments(config, steps, int(tokens), bool(int(origin)))
    scheduler.set_timesteps(steps, **arguments)
    with torch.enable_grad():
        latent = torch.from_numpy(np.asarray(x_T)).to(dtype).requires_grad_()
        def level(index, time):
            return torch.as_tensor(time).to(dtype) / 1000.0

        latents = cases.flow_walk(scheduler, latent, level, std=std)
        pulled = torch.from_numpy(np.asarray(cotangent)).to(dtype)
        (gradient,) = torch.autograd.grad((latents[-1] * pulled).sum(), latent)
    return {"sigmas": scheduler.sigmas, "times": scheduler.timesteps,
            **{f"after {index}": latent.detach() for index, latent in enumerate(latents)}, "grad": gradient}


@pytest.mark.parametrize("steps", [4, 7])
@pytest.mark.parametrize("name", FLOW_CASES)
def test_native_flow_schedule_matches_the_source_grids_and_trajectory(name, steps, source):
    """The actual `FlowMatchEulerDiscreteScheduler` of each published file, at a drawn resolution and
    either sigma origin.

    `scheduler` is the seed SD3's pipeline leaves to the class, whose
    constructor has already applied the static shift once, so the shift lands
    twice; `linspace` is the `linspace(1, 1/N, N)` a Flux pipeline hands it
    with the mu its `calculate_shift` returns. The sigmas, the model times,
    every latent of the Euler walk and its input gradient are on par with the
    source's own float32 walk over drawn starts, cotangents, token counts and
    origins (check 1), the class's independent float64 oracle
    (tools/float64_schedulers.py) the truth.
    """
    import torch

    from _dew_torch.precision import float64

    config = json.loads(str(flow_reference(source)[f"{name}.config"]))
    std = json.loads((source / "sd3_transformer.json").read_text())["flow"]["data_std"]
    schedule = SourceSchedule.from_config(config)
    model = Velocity(std)
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, *FLOW_SHAPE[1:])), jnp.ones((1,)))

    @as_given
    def native(x_T, cotangent, tokens, origin):
        process, times = schedule.sampling(steps, tokens=int(tokens),
                                           origin="linspace" if origin else "scheduler")
        denoise = process.denoiser(model, params, {})

        def run(value):
            return walk_latents(denoise, schedule.solver, value, times)

        latents, pullback = jax.vjp(run, jnp.asarray(x_T))
        (gradient,) = pullback(jnp.zeros_like(latents).at[-1].set(jnp.asarray(cotangent, latents.dtype)))
        return {"sigmas": process.sampler_schedule.sigmas(times),
                "times": process.sampler_schedule.model_time(times[:-1]),
                **{f"after {index}": latent for index, latent in enumerate(latents)}, "grad": gradient}

    def reference(dtype):
        def walked(*inputs):
            return jax.tree.map(lambda value: value.numpy(), source_flow(config, steps, std, dtype, *inputs))
        return walked

    @as_given
    def truth(*inputs):
        with float64():
            return reference(torch.float64)(*inputs)

    assert_as_exact(native, as_given(reference(torch.float32)), flow_draw, f"{name} at {steps} steps",
                    seeds=FLOW_SEEDS, truth_fn=truth)


def walk_latents(denoise, solver, x_T, times):
    """Each interval's latent of `sample`'s walk over `times` from `x_T`, without the closing denoise."""
    _, latents = sample(denoise, x_T, solver=solver, key=jax.random.PRNGKey(0), times=times,
                        final_denoise=False, record=lambda x, state, denoised: x)
    return latents


@attending("xla")
@pytest.mark.mesh
def test_native_sd3_agrees_across_a_sequence_sharded_mesh(source):
    """The joint attention under a real mesh whose sequence axis is above one.

    The model's attention is `scaled_dot_product_attention`, which routes
    itself through `sequence_parallel_attention` when that axis is above one,
    so this is the actual sharded seam rather than a declaration about it. The
    case is deliberately awkward: a CLIP-length 77 context beside a
    rectangular 12x6 latent, so the joint sequence is 95 queries over 2
    shards, and the loss and the parameter gradient are on par with the same
    walk on a mesh that keeps sequences whole over drawn contexts and probes
    (check 1).
    """
    from dew.training import Layout, MeshSpec

    with np.load(source / "sd3_transformer.npz") as arrays:
        config = json.loads(str(arrays["rect_cropped.config"]))
        latent = jnp.asarray(arrays["rect_cropped.latent"]).transpose(0, 2, 3, 1)
        pooled = jnp.asarray(arrays["rect_cropped.pooled"])
        times = jnp.asarray(arrays["rect_cropped.times"])
    model = SD3Transformer(**sd3_fields(config))
    params, buffers, _ = translate_sd3_weights(
        component_tensors(source / "rect_cropped", "transformer"))
    rows = latent.shape[0]

    def draw(seed):
        generator = np.random.default_rng([0x5D3, seed])
        return (generator.standard_normal((rows, 77, config["joint_attention_dim"])).astype(np.float32),
                generator.standard_normal(latent.shape).astype(np.float32))

    def loss(variables, context, probe):
        output = model.apply({"params": variables, "buffers": buffers}, latent, times,
                             DenoisingCondition(context, pooled))
        return jnp.sum(output * probe)

    def placed(spec):
        """The loss and its parameter gradient on `spec`'s mesh, the parameters placed as training places
        them."""
        mesh = spec.build()
        placement = Layout().shardings(mesh, {"params": params})["params"]
        step = jax.jit(jax.value_and_grad(loss))

        def run(context, probe):
            with jax.set_mesh(mesh):
                return step(jax.device_put(params, placement), context, probe)

        return run

    # Two programs of Dew's, the sequence split against it whole: held so, and not bit for bit.
    split, whole = (as_given(placed(spec)) for spec in (MeshSpec(fsdp=2, sequence=2), MeshSpec(fsdp=4)))
    wide = model.clone(dtype=jnp.float64)
    held = float64_held({"params": params, "buffers": buffers,
                         "inputs": (latent, times, pooled)})

    # A float64 program of JAX's, traced and compiled by the check as its arms are, its float64 read off it.
    def truth(context, probe):
        def wide_loss(variables, context):
            output = wide.apply({"params": variables, "buffers": held["buffers"]}, held["inputs"][0],
                                held["inputs"][1], DenoisingCondition(context, held["inputs"][2]))
            return jnp.sum(output * probe)

        return jax.value_and_grad(wide_loss)(held["params"], context)

    assert_as_exact(split, dew_path(whole), draw, "SD3 on a sequence-sharded mesh", seeds=FEWEST_SEEDS,
                    truth_fn=truth)


# The tiny pipeline the reference tool saves and walks: two CLIP towers, a T5
# tower and the transformer, in a real published directory layout.
PIPELINE_CASES = ("pipeline", "pipeline_no_t5")


@pytest.fixture(scope="module")
def pipeline_record(source):
    return json.loads((source / "sd3_transformer.json").read_text())["pipeline"]


@attending("xla")
@pytest.mark.parametrize("case", PIPELINE_CASES)
def test_published_prompt_encoding_matches_the_source_pipeline(source, pipeline_record, case):
    """The conditioner composes what `encode_prompt` composes.

    Every text slot carries different words, so a native encoder that routed
    `prompt_2` to T5 or dropped the third slot would not land here: the CLIP
    towers' penultimate states pad out to the T5 width, the T5 states follow
    them along the sequence, and the pooled vector is both projections. The
    second case ships no third encoder, whose segment is the zero block the
    source writes at its CLIP window rather than at the sequence it asked for.
    """
    from dew.interop.pretrained import Pretrained

    arrays = np.load(source / "sd3_pipeline.npz")
    loaded = Pretrained.load(str(source / case), dtype="float32")
    encoder = loaded.inputs.conditions["conditioning"].encoder
    params = loaded.variables["encoders"]["conditioning"]
    for prefix, rows in (("", pipeline_record["prompts"]), ("negative_", pipeline_record["negatives"])):
        condition = encoder.encode(params, encoder.tokenize(rows))
        expected = arrays[f"{case}.{prefix}context"]
        assert condition.context.shape == expected.shape
        # Observed 2.5e-6 with the T5 segment and 6e-7 without it, which is
        # float32 reassociation across the towers' matmuls.
        assert relative_gap(condition.context, expected) < 1e-5
        assert relative_gap(condition.pooled, arrays[f"{case}.{prefix}pooled"]) < 1e-6
    # Crossing the slots is what the distinct words rule out.
    crossed = encoder.encode(params, encoder.tokenize(
        [{"text": row["second"], "second": row["text"], "third": row["third"]}
         for row in pipeline_record["prompts"]]))
    assert relative_gap(crossed.context, arrays[f"{case}.context"]) > 1e-3


@attending("xla")
@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("case", PIPELINE_CASES)
def test_published_pipeline_walk_matches_the_source(source, pipeline_record, case, streamed):
    """`Pretrained.load().text_to_image()` reproduces the source's own call.

    The published directory decides everything: the transformer, the wide
    latent VAE with its shift and scale, the flow schedule's shifted sigmas,
    the guidance the call applies to the raw velocity, and the geometry the
    grid is bound to. The walk starts from the source's own latents so only
    the trajectory is under test. Loaded whole or streamed onto a mesh
    (tests/test_pipeline_streaming.py).
    """
    from dew.interop.pretrained import Pretrained
    from dew.sampling.guidance import CFG
    from dew.training import MeshSpec

    arrays = np.load(source / "sd3_pipeline.npz")
    loaded = Pretrained.load(str(source / case), dtype="float32",
                             **({"mesh": MeshSpec()} if streamed else {}))
    task = loaded.text_to_image()
    prepared = task.prepare(pipeline_record["prompts"], unconditional=pipeline_record["negatives"],
                            initial=arrays[f"{case}.x_T"], steps=pipeline_record["steps"], key=0)
    # The flow walk is an Euler integration with no noise draw; the key is
    # the call's contract, not a source of difference.
    walked = task(prepared, guidance=CFG(pipeline_record["guidance"]),
                  key=jax.random.PRNGKey(0)).host()
    assert relative_gap(walked.latents, arrays[f"{case}.latents"]) < 2e-5
    images = np.clip(np.asarray(walked.images) / 2 + 0.5, 0.0, 1.0)
    assert relative_gap(images, arrays[f"{case}.images"]) < 2e-5


@attending("xla")
@pytest.mark.parametrize("case", PIPELINE_CASES)
def test_omitted_call_policy_takes_the_published_pipelines_own(source, pipeline_record, case):
    """A call that names no policy walks what the source's own call walks.

    The published pipeline class carries the step count and the guidance scale
    its `__call__` applies, so a native task built from a checkpoint that
    declares that class takes them: not the fifty steps and 7.5 of the older
    UNet pipelines, and not anything a test passes in.
    """
    from dew.interop.pretrained import Pretrained

    arrays = np.load(source / "sd3_pipeline.npz")
    loaded = Pretrained.load(str(source / case), dtype="float32")
    task = loaded.text_to_image()
    assert task.steps == pipeline_record["default_steps"]
    assert task.guidance.scale == pipeline_record["default_guidance"]
    prepared = task.prepare(pipeline_record["prompts"], unconditional=pipeline_record["negatives"],
                            initial=arrays[f"{case}.x_T"], key=0)
    walked = task(prepared, key=jax.random.PRNGKey(0)).host()
    assert relative_gap(walked.latents, arrays[f"{case}.default_latents"]) < 2e-5


@attending("xla")
def test_a_trained_step_keeps_the_frozen_buffer_and_exports_for_the_source(source, tmp_path):
    """A real coupled-loss step over the published source, then a resume and
    an export.

    The objective's own loss runs the whole source: the VAE encodes the
    pixels, both CLIP towers and the T5 tower encode the prompt, and the
    transformer takes the step. The optimizer sees only `params`, so the
    stored position buffer is state rather than a weight: it survives the
    step and the resume unchanged, and the export writes it back where the
    source keeps it, so a reload recomputes the trained model exactly.
    """
    from diffusion_support import image_batch, source_objective, trained_step

    from dew.checkpoints import Checkpoints
    from dew.interop.pretrained import Pretrained

    loaded, objective = source_objective(source)
    batch = image_batch(loaded, [{"text": "a red cat", "second": "a blue dog", "third": "a green bird"}])
    initial, state = trained_step(objective, batch, Checkpoints(str(tmp_path / "run")))
    # The step moved weights and left the buffer alone.
    buffer = initial.variables["buffers"]["pos_embed"]
    np.testing.assert_array_equal(buffer, loaded.variables["buffers"]["pos_embed"])
    np.testing.assert_array_equal(state.variables["buffers"]["pos_embed"], buffer)

    export = tmp_path / "export"
    loaded.save(export, variables=state.variables)
    again = Pretrained.load(str(export), dtype="float32")
    np.testing.assert_array_equal(again.variables["buffers"]["pos_embed"], buffer)
    latent = jnp.asarray(np.load(source / "sd3_pipeline.npz")["pipeline.x_T"][:1])
    condition = DenoisingCondition(jnp.ones((1, 4, 32), jnp.float32),
                                   jnp.ones((1, 10), jnp.float32))
    # The reload builds the trained model, so one compiled forward reads both trees.
    assert again.model == loaded.model
    forward = jax.jit(lambda tree, drawn: loaded.model.apply(tree, drawn, jnp.asarray([0.5]), condition))
    trained_tree = {"params": state.variables["params"], "buffers": state.variables["buffers"]}
    trained = forward(trained_tree, latent)
    reloaded = forward({"params": again.variables["params"], "buffers": again.variables["buffers"]}, latent)
    np.testing.assert_array_equal(reloaded, trained)

    # Diffusers' own classes read the export: every component cleanly, the
    # pipeline whole, and its transformer recomputes the trained forward.
    from diffusion_support import assert_export_recomputes

    from tools import diffusers_consumer as consumer

    consumer.assert_components_load(export)

    def draw(seed):
        """A latent of the pipeline's shape, its values drawn."""
        return (np.random.default_rng([4441, seed]).standard_normal(latent.shape).astype(np.float32),)

    def call(model, tensor, drawn):
        output = model(hidden_states=tensor(drawn).permute(0, 3, 1, 2),
                       encoder_hidden_states=tensor(condition.context),
                       pooled_projections=tensor(condition.pooled), timestep=tensor([0.5])).sample
        return output.permute(0, 2, 3, 1).numpy()

    assert_export_recomputes(lambda drawn: forward(trained_tree, drawn),
                             lambda wide: consumer.denoiser(export, call, wide=wide), draw,
                             "trained SD3")


@pytest.mark.parametrize("kernel", ["reference", None], ids=["reference", "auto"])
def test_the_published_t5_tower_over_its_whole_window_is_on_par_with_the_reference(source, kernel):
    """The pipeline's T5 tower in float32 over its prompts' whole 256-token window, every key attended as
    the pipeline's encode attends them (no mask): its last hidden state on par with transformers' float32
    encoder from its float64 one over drawn token ids (check 1), on the reference route and on the one its
    selector takes (`auto`, a CPU's xla where a process computes as a user's does). One fp32 dot over the
    256 keys of its attention's probability-value product rounded 2.2 times as far from float64 as torch's
    CPU attention, which sums keys in blocks (`attention.weighted_values`, row 13655); tests/
    test_users_process.py runs both cells in a user's arithmetic, where the reference route's blocked sum
    is the one it takes (train 1373's F5: 1.555 times with the one dot there)."""
    import torch
    from transformers import T5EncoderModel

    from _dew_torch.precision import float64, float64_twin
    from dew.inputs import T5Text

    directory = source / "pipeline" / "text_encoder_3"
    encoder = T5Text.from_pretrained(str(directory), dtype=jnp.float32)
    vocabulary = json.loads((directory / "config.json").read_text())["vocab_size"]
    narrow = T5EncoderModel.from_pretrained(directory, dtype=torch.float32).eval()
    wide = float64_twin(narrow)

    def draw(seed):
        return (np.random.default_rng([0x13655, seed]).integers(2, vocabulary, (2, 256)).astype(np.int32),)

    def transformers(model):
        def run(ids):
            with torch.no_grad():
                states = model(input_ids=torch.from_numpy(np.asarray(ids, np.int64))).last_hidden_state
            return {"last_hidden_state": states.numpy()}
        return run

    @as_given
    def truth(ids):
        with float64():
            return transformers(wide)(ids)

    @as_given
    def ours(ids):
        tokens = {"input_ids": np.asarray(ids), "attention_mask": np.ones(np.shape(ids), np.int32)}
        return {"last_hidden_state": np.asarray(encoder.encode(encoder.params, tokens).hidden, np.float32)}

    with generation.chosen(**({} if kernel is None else {"attention": kernel})):
        assert_as_exact(ours, as_given(transformers(narrow)), draw, "SD3's T5 tower over its window", seeds=8,
                        truth_fn=truth)
