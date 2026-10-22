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
from flax import linen as nn
from interop_support import extract_fixture, scaled_gap as relative_gap

from dew.diffusion.process import DenoisingCondition
from dew.diffusion.schedules.source import SourceSchedule
from dew.interop.diffusion import component_tensors, sd3_fields, translate_sd3_weights
from dew.nn.backbones.sd3 import SD3Transformer
from dew.sampling import sample

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return extract_fixture(ROOT / "tests/fixtures/sd3_source.tar.xz", tmp_path_factory.mktemp("sd3-source"))


def transformer_case(directory: Path, arrays, name: str):
    """The native model, its loaded variables and the case's inputs."""
    config = json.loads(str(arrays[f"{name}.config"]))
    model = SD3Transformer(**sd3_fields(config, attention_impl="xla"))
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


@pytest.mark.parametrize("name", CASES)
def test_native_sd3_matches_the_source_forward_and_every_gradient(name, source):
    """One published transformer's own tensors, read through the native model.

    The variants are the ones whose wiring differs: plain, `qk_norm`, SD3.5's
    dual attention with its ninefold modulation, a rectangular latent whose
    patch grid is smaller than the position buffer, and a deeper stack. The
    output and the latent, token and pooled gradients land within 6e-7 of the
    source's own scale, and every parameter gradient within 4e-5: the
    timestep embedder's first weight is the sinusoidal feature itself, where a
    float32 `exp` differing by an ulp is multiplied by a timestep near a
    thousand, and the rest sit under 2e-6.
    """
    with np.load(source / "sd3_transformer.npz") as arrays:
        forward, params, buffers, layouts, latent, condition = transformer_case(
            source, arrays, name)
        probe = jnp.asarray(arrays[f"{name}.probe"]).transpose(0, 2, 3, 1)
        output = jax.jit(forward)(params, latent, condition.context, condition.pooled)
        assert relative_gap(output.transpose(0, 3, 1, 2), arrays[f"{name}.output"]) < 1e-5
        # The stored buffer is the source's, perturbed away from its own
        # sin/cos initializer, so a model that rebuilt it would not match.
        assert relative_gap(buffers["pos_embed"], arrays[f"{name}.position_buffer"]) == 0.0
        gradients = jax.jit(
            jax.grad(
                lambda p, noisy_latent, c, q: jnp.sum(forward(p, noisy_latent, c, q) * probe),
                argnums=(0, 1, 2, 3),
            )
        )(params, latent, condition.context, condition.pooled)
        assert relative_gap(gradients[1].transpose(0, 3, 1, 2), arrays[f"{name}.grad_latent"]) < 1e-5
        assert relative_gap(gradients[2], arrays[f"{name}.grad_context"]) < 1e-5
        assert relative_gap(gradients[3], arrays[f"{name}.grad_pooled"]) < 1e-5
        layout = {entry.name: entry for entry in layouts}
        prefix = f"{name}.grad_param."
        recorded = [key for key in arrays.files if key.startswith(prefix)]
        assert recorded, name
        for key in recorded:
            value = np.asarray(layout["transformer/" + key[len(prefix):]].export({"params": gradients[0]}))
            assert relative_gap(value, arrays[key]) < 1e-4, key


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


@pytest.mark.parametrize("name", FLOW_CASES)
def test_native_flow_schedule_matches_the_source_grids_and_trajectory(name, source):
    """The actual `FlowMatchEulerDiscreteScheduler` at two step counts, three
    resolutions and both sigma origins.

    `scheduler` is the seed SD3's pipeline leaves to the class, whose
    constructor has already applied the static shift once, so the shift lands
    twice; `linspace` is the `linspace(1, 1/N, N)` a Flux pipeline hands it
    with the mu its `calculate_shift` returns. The sigmas, the model times,
    every latent of the Euler walk and its input gradient land within 3e-7 of
    the source's own scale.
    """
    arrays = flow_reference(source)
    meta = json.loads((source / "sd3_transformer.json").read_text())["flow"]
    schedule = SourceSchedule.from_config(json.loads(str(arrays[f"{name}.config"])))
    model = Velocity(meta["data_std"])
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 3, 4)), jnp.ones((1,)))
    for steps in meta["steps"]:
        for tokens in meta["tokens"]:
            for origin in ("scheduler", "linspace"):
                tag = f"{name}.{steps}.{tokens}.{origin}"
                process, times = schedule.sampling(steps, tokens=tokens, origin=origin)
                assert relative_gap(process.sampler_schedule.sigmas(times),
                                    arrays[f"{tag}.sigmas"]) < 1e-5
                assert relative_gap(process.sampler_schedule.model_time(times[:-1]),
                                    arrays[f"{tag}.times"]) < 1e-5
                denoise = process.denoiser(model, params, {})
                def run(value, *, denoise=denoise, times=times):
                    return sample(denoise, value, solver=schedule.solver, key=jax.random.PRNGKey(0),
                                  times=times, final_denoise=False)
                x_T = jnp.asarray(arrays[f"{tag}.x_T"])
                assert relative_gap(run(x_T), arrays[f"{tag}.latents"][-1]) < 1e-4
                (gradient,) = jax.vjp(run, x_T)[1](jnp.asarray(arrays[f"{tag}.cotangent"]))
                assert relative_gap(gradient, arrays[f"{tag}.grad"]) < 1e-4


@pytest.mark.mesh
def test_native_sd3_agrees_across_a_sequence_sharded_mesh(source):
    """The joint attention under a real mesh whose sequence axis is above one.

    The model's attention is `scaled_dot_product_attention`, which routes
    itself through `sequence_parallel_attention` when that axis is above one,
    so this is the actual sharded seam rather than a declaration about it. The
    case is deliberately awkward: a CLIP-length 77 context beside a
    rectangular 12x6 latent, so the joint sequence is 95 queries over 2
    shards, and the prediction and the parameter gradient are compared against
    the same walk on a mesh that keeps sequences whole.
    """
    from dew.training import Layout, MeshSpec

    with np.load(source / "sd3_transformer.npz") as arrays:
        config = json.loads(str(arrays["rect_cropped.config"]))
        latent = jnp.asarray(arrays["rect_cropped.latent"]).transpose(0, 2, 3, 1)
        pooled = jnp.asarray(arrays["rect_cropped.pooled"])
        times = jnp.asarray(arrays["rect_cropped.times"])
    model = SD3Transformer(**sd3_fields(config, attention_impl="xla"))
    params, buffers, _ = translate_sd3_weights(
        component_tensors(source / "rect_cropped", "transformer"))
    rows = latent.shape[0]
    context = jax.random.normal(jax.random.PRNGKey(5), (rows, 77, config["joint_attention_dim"]))
    probe = jax.random.normal(jax.random.PRNGKey(6), latent.shape)

    def loss(variables):
        output = model.apply({"params": variables, "buffers": buffers}, latent, times,
                             DenoisingCondition(context, pooled))
        return jnp.sum(output * probe)

    def run(spec):
        mesh = spec.build()
        placement = Layout().shardings(mesh, {"params": params})["params"]
        with jax.set_mesh(mesh):
            placed = jax.device_put(params, placement)
            return jax.jit(jax.value_and_grad(loss))(placed)

    whole, split = run(MeshSpec(fsdp=4)), run(MeshSpec(fsdp=2, sequence=2))
    assert relative_gap(split[0], whole[0]) < 1e-5
    gaps = [relative_gap(a, b) for a, b in zip(jax.tree.leaves(split[1]),
                                               jax.tree.leaves(whole[1]), strict=True)]
    assert max(gaps) < 1e-5, max(gaps)


# The tiny pipeline the reference tool saves and walks: two CLIP towers, a T5
# tower and the transformer, in a real published directory layout.
PIPELINE_CASES = ("pipeline", "pipeline_no_t5")


@pytest.fixture(scope="module")
def pipeline_record(source):
    return json.loads((source / "sd3_transformer.json").read_text())["pipeline"]


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
    loaded = Pretrained.load(str(source / case), dtype="float32", attention_impl="xla")
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
    loaded = Pretrained.load(str(source / case), dtype="float32", attention_impl="xla",
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
    loaded = Pretrained.load(str(source / case), dtype="float32", attention_impl="xla")
    task = loaded.text_to_image()
    assert task.steps == pipeline_record["default_steps"]
    assert task.guidance.scale == pipeline_record["default_guidance"]
    prepared = task.prepare(pipeline_record["prompts"], unconditional=pipeline_record["negatives"],
                            initial=arrays[f"{case}.x_T"], key=0)
    walked = task(prepared, key=jax.random.PRNGKey(0)).host()
    assert relative_gap(walked.latents, arrays[f"{case}.default_latents"]) < 2e-5


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
    again = Pretrained.load(str(export), dtype="float32", attention_impl="xla")
    np.testing.assert_array_equal(again.variables["buffers"]["pos_embed"], buffer)
    latent = jnp.asarray(np.load(source / "sd3_pipeline.npz")["pipeline.x_T"][:1])
    condition = DenoisingCondition(jnp.ones((1, 4, 32), jnp.float32),
                                   jnp.ones((1, 10), jnp.float32))
    trained = loaded.model.apply({"params": state.variables["params"],
                                  "buffers": state.variables["buffers"]},
                                 latent, jnp.asarray([0.5]), condition)
    reloaded = again.model.apply({"params": again.variables["params"],
                                  "buffers": again.variables["buffers"]},
                                 latent, jnp.asarray([0.5]), condition)
    np.testing.assert_array_equal(reloaded, trained)

    # Diffusers' own classes read the export: every component cleanly, the
    # pipeline whole, and its transformer recomputes the trained forward.
    from reference_error import assert_as_exact_as_the_reference

    from tools import diffusers_consumer as consumer

    consumer.assert_components_load(export)

    def call(model, tensor):
        output = model(hidden_states=tensor(latent).permute(0, 3, 1, 2),
                       encoder_hidden_states=tensor(condition.context),
                       pooled_projections=tensor(condition.pooled), timestep=tensor([0.5])).sample
        return output.permute(0, 2, 3, 1).numpy()

    assert_as_exact_as_the_reference(np.asarray(trained), consumer.denoiser_prediction(export, call),
                                     consumer.denoiser_prediction(export, call, wide=True), "trained SD3")
