"""Native FLUX.2 against the actual Diffusers 0.40.0 objects it reconstructs.

`tools/diffusers_flux2_reference.py` builds tiny `Flux2Transformer2DModel`
instances with every parameter moved off its initialization, saves each with
its real config and safetensors, and runs each as `Flux2Pipeline` runs it -
one token per latent position, the four-axis ids the pipeline lays out, the
timestep it has divided by the training count and the distilled guidance -
recording the forward and the gradients of the latent, the text states and
every parameter against a fixed cotangent. It walks two tiny
`AutoencoderKLFlux2`s the same way: the pipeline's encode of a reference
image and its decode of a result, through the 2x2 fold and the batch norm's
statistics.

The pipeline flattens its latent row-major, one token per position, so these
tests reshape between that and NHWC with numpy's own reshape.
"""

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from interop_support import extract_fixture, fixture_arrays

from dew.diffusion.process import DenoisingCondition
from dew.interop.diffusion import component_tensors, flux2_fields, translate_flux2_weights
from dew.nn.autoencoders.flux2 import load_flux2_vae
from dew.nn.backbones.flux2 import Flux2Transformer
from dew.nn.scan_orders import pixel_shuffle

ROOT = Path(__file__).resolve().parents[1]
CASES = ("dev", "rect", "unguided", "narrow")
FORWARD = 1e-5
GRADIENT = 1e-4


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return extract_fixture(ROOT / "tests/fixtures/flux2_source.tar.xz",
                           tmp_path_factory.mktemp("flux2-source"))


@pytest.fixture(scope="module")
def record(source):
    return json.loads((source / "flux2_transformer.json").read_text())


@pytest.fixture(scope="module")
def arrays(source):
    return fixture_arrays(source / "flux2_transformer.npz")


def relative_gap(actual, expected) -> float:
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    return float(np.abs(actual - expected).max() / max(1.0, float(np.abs(expected).max())))


def load(source, name):
    config = json.loads((source / name / "transformer" / "config.json").read_text())
    model = Flux2Transformer(**flux2_fields(config, dtype="float32", attention_impl="xla"))
    params, layouts = translate_flux2_weights(component_tensors(source / name, "transformer"))
    return model, params, layouts


def walk(name, arrays, record):
    rows, columns = record["cases"][name]["grid"]
    latent = arrays[f"{name}.latent"].reshape(-1, rows, columns, arrays[f"{name}.latent"].shape[-1])
    guidance = arrays[f"{name}.guidance"]
    condition = DenoisingCondition(jnp.asarray(arrays[f"{name}.context"]),
                                   guidance=jnp.asarray(guidance) if guidance.size else None)
    return jnp.asarray(latent), jnp.asarray(arrays[f"{name}.times"]), condition, (rows, columns)


@pytest.mark.parametrize("name", CASES)
def test_every_published_tensor_maps_and_exports_bit_identical(source, name):
    _, params, layouts = load(source, name)
    tensors = component_tensors(source / name, "transformer")
    assert {layout.name for layout in layouts} == {f"transformer/{key}" for key in tensors}
    for layout in layouts:
        written = layout.export({"params": params})
        assert written.tobytes() == tensors[layout.name.removeprefix("transformer/")].tobytes(), layout.name


@pytest.mark.parametrize("name", CASES)
def test_the_forward_and_every_gradient_match_the_source(source, arrays, record, name):
    model, params, layouts = load(source, name)
    latent, times, condition, (rows, columns) = walk(name, arrays, record)
    probe = arrays[f"{name}.probe"].reshape(latent.shape[0], rows, columns, -1)

    def objective(params, latent, context):
        conditioned = DenoisingCondition(context, guidance=condition.guidance)
        return jnp.sum(model.apply({"params": params}, latent, times, conditioned) * probe)

    output = model.apply({"params": params}, latent, times, condition)
    forward = relative_gap(output.reshape(arrays[f"{name}.output"].shape), arrays[f"{name}.output"])
    grad_params, grad_latent, grad_context = jax.grad(objective, argnums=(0, 1, 2))(
        params, latent, condition.context)
    gaps = {"latent": relative_gap(grad_latent.reshape(arrays[f"{name}.grad_latent"].shape),
                                   arrays[f"{name}.grad_latent"]),
            "context": relative_gap(grad_context, arrays[f"{name}.grad_context"])}
    gaps.update({layout.name: relative_gap(layout.export({"params": grad_params}),
                                           arrays[f"{name}.grad_param.{layout.name.removeprefix('transformer/')}"])
                 for layout in layouts})
    worst = max(gaps, key=gaps.get)
    print(f"{name}: forward {forward:.3g}; worst gradient {worst} {gaps[worst]:.3g}")
    assert forward < FORWARD
    assert gaps[worst] < GRADIENT, worst


def test_a_guided_checkpoint_refuses_a_call_without_guidance(source, arrays, record):
    model, params, _ = load(source, "dev")
    latent, times, condition, _ = walk("dev", arrays, record)
    with pytest.raises(ValueError, match="embeds its guidance"):
        model.apply({"params": params}, latent, times, DenoisingCondition(condition.context))


@pytest.mark.parametrize(
    "change", [{"patch_size": 2}, {"axes_dims_rope": [8, 8]}, {"axes_dims_rope": [4, 4, 4, 3]}]
)
def test_an_unsupported_config_is_refused(source, change):
    config = json.loads((source / "dev" / "transformer" / "config.json").read_text())
    flux2_fields(config)
    with pytest.raises(ValueError):
        flux2_fields({**config, **change})


def nhwc(array):
    return np.asarray(array).transpose(0, 2, 3, 1)


@pytest.mark.parametrize("name", ("vae", "small_decoder"))
def test_the_autoencoder_folds_and_normalizes_as_the_pipeline_does(source, arrays, name):
    autoencoder, params, layouts, _ = load_flux2_vae(source / name)
    tensors = component_tensors(source / name, "vae")
    assert {layout.name for layout in layouts} == {
        f"vae/{key}" for key in tensors if not key.startswith("bn.")
    }
    image = nhwc(arrays[f"{name}.image"])
    latent = autoencoder.encode(params, image)
    assert latent.shape == (2, *autoencoder.latent_shape(image.shape[1:])) == (2, 4, 6, 16)
    probe_latent = nhwc(arrays[f"{name}.probe_latent"])
    grad_image = jax.grad(lambda image: jnp.sum(autoencoder.encode(params, image) * probe_latent))(image)
    code = nhwc(arrays[f"{name}.code"])
    pixels = autoencoder.decode(params, code)
    probe = nhwc(arrays[f"{name}.probe"])
    grad_code = jax.grad(lambda code: jnp.sum(autoencoder.decode(params, code) * probe))(code)
    gaps = {"latent": relative_gap(latent, nhwc(arrays[f"{name}.latent"])),
            "pixels": relative_gap(pixels, nhwc(arrays[f"{name}.pixels"])),
            "grad_image": relative_gap(grad_image, nhwc(arrays[f"{name}.grad_image"])),
            "grad_code": relative_gap(grad_code, nhwc(arrays[f"{name}.grad_code"]))}
    print(f"{name} gaps {gaps}")
    assert max(gaps["latent"], gaps["pixels"]) < FORWARD, gaps
    assert max(gaps["grad_image"], gaps["grad_code"]) < GRADIENT, gaps


@pytest.mark.network
def test_the_published_autoencoder_matches_the_source(monkeypatch):
    """FLUX.2's VAE as `FLUX.2-klein-4B` publishes it (the one FLUX.2 [dev]
    ships), downloaded, on a smooth 128x192 image, against the source run in
    float64 (`flux2_published.npz`): the pipeline's folded, normalized latent
    and a corner of its decode, each to within twice the source's own
    float32 distance from that result, or 1e-5."""
    import importlib.util

    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location(
        "diffusers_flux2_reference", ROOT / "tools/diffusers_flux2_reference.py"
    )
    tool = importlib.util.module_from_spec(spec)
    # Its dataclasses resolve their module through sys.modules.
    sys.modules[spec.name] = tool
    spec.loader.exec_module(tool)
    reference = np.load(ROOT / "tests/fixtures/flux2_published.npz")
    repo, revision = tool.PUBLISHED_VAE
    autoencoder, params, _, _ = load_flux2_vae(repo, revision=revision)
    latent = autoencoder.encode(params, nhwc(tool.smooth_image()))
    pixels = autoencoder.decode(params, nhwc(reference["vae.latent"]))[:, :tool.CROP, :tool.CROP]
    assert latent.shape == (1, 8, 12, 128)
    gaps = {"latent": relative_gap(latent, nhwc(reference["vae.latent"])),
            "pixels": relative_gap(pixels, nhwc(reference["vae.pixels"]))}
    bounds = {key: max(FORWARD, 2 * float(reference[f"vae.{key}.float32_gap"])) for key in gaps}
    print(f"published VAE gaps {gaps}, bounds {bounds}")
    assert all(gaps[key] < bounds[key] for key in gaps), (gaps, bounds)


@pytest.fixture(scope="module")
def klein(source):
    from dew.interop.pretrained import Pretrained

    return Pretrained.load(str(source / "pipeline"), dtype="float32", attention_impl="xla")


def test_klein_prompt_encoding_matches_the_source_pipeline(klein, arrays, record):
    """The conditioner stacks what `encode_prompt` stacks: the chat template
    with thinking off, every row padded on the right to 512 tokens, and the
    outputs of layers 9, 18 and 27 side by side, pads included."""
    encoder = klein.inputs.conditions["conditioning"].encoder
    params = klein.variables["encoders"]["conditioning"]
    prompts = [*record["pipeline"]["prompts"], ""]
    condition = encoder.encode(params, encoder.tokenize(prompts))
    assert condition.context.shape == (3, 512, record["pipeline"]["config"]["joint_attention_dim"])
    assert condition.guidance is None
    for row in range(len(prompts)):
        assert relative_gap(condition.context[row], arrays[f"pipeline.context.{row}"][0]) < FORWARD, row
    with pytest.raises(ValueError, match="token budget"):
        encoder.tokenize(["x" * 513])


def test_klein_pipeline_walk_matches_the_source(klein, arrays, record):
    """`Pretrained.load().text_to_image()` reproduces the source's own call:
    its defaults (50 steps, two branches guided at 4.0 against the empty
    prompt), and at the recorded step count the sigmas it lays out shifted
    by its empirical mu, the latent it ends on and the image it decodes."""
    pipeline = record["pipeline"]
    task = klein.text_to_image()
    assert task.steps == 50 and task.guidance is not None and task.guidance.scale == 4.0
    initial = arrays["pipeline.x_T"].transpose(0, 2, 3, 1)
    walked = task(task.prepare(pipeline["prompts"], initial=initial, key=0, steps=pipeline["steps"]),
                  key=jax.random.PRNGKey(0)).host()
    images = np.clip(np.asarray(walked.images) / 2 + 0.5, 0.0, 1.0)
    autoencoder = klein.autoencoder
    raw = pixel_shuffle(np.asarray(walked.latents) / autoencoder.latent_scale + autoencoder.latent_shift)
    for row in range(len(pipeline["prompts"])):
        assert relative_gap(raw[row], nhwc(arrays[f"pipeline.latents.{row}"])[0]) < FORWARD, row
        assert relative_gap(images[row], arrays[f"pipeline.images.{row}"][0]) < FORWARD, row


def test_a_step_distilled_klein_samples_unguided(source):
    """The published [klein] marks itself step-distilled, and its call then
    ignores its guidance scale."""
    from dew.interop.pipeline_assembly import _call_policy, pipeline_denoiser

    denoiser = pipeline_denoiser(source / "pipeline", dtype="float32", attention_impl="xla")
    index = json.loads((source / "pipeline" / "model_index.json").read_text())
    assert _call_policy(index, denoiser).guided
    assert not _call_policy({**index, "is_distilled": True}, denoiser).guided


def test_the_empirical_shift_is_the_pipelines():
    from dew.diffusion.schedules.source import empirical_mu

    # compute_empirical_mu at diffusers 0.40.0, evaluated by hand: the 10- and
    # 200-step lines at 1024 tokens, and the 200-step line past 4300.
    assert empirical_mu(1024, 200) == pytest.approx(0.00016927 * 1024 + 0.45666666)
    assert empirical_mu(1024, 10) == pytest.approx(8.73809524e-05 * 1024 + 1.89833333)
    assert empirical_mu(5000, 4) == pytest.approx(0.00016927 * 5000 + 0.45666666)


def test_a_mistral3_encoder_reads_as_flux2_dev_reads_it(source, arrays, record):
    """FLUX.2 [dev]'s conditioning over a tiny Mistral-3 encoder with Mistral
    Small 3.1's chat template: the system and user turns, 512 tokens padded
    on the right, the hidden states after layers 10, 20 and 30 stacked, and
    the guidance [dev] embeds. The vision tower is carried, not run, and
    exports as it came."""
    from dew.interop.pipeline_assembly import _hidden_states_conditioning

    directory = source / "mistral3"
    encoder, layouts, _ = _hidden_states_conditioning(
        directory, {}, jnp.float32, (16, 16), pipeline="flux2", tokens=512, guidance=4.0,
        param_dtype="float32", attention_impl="xla")
    assert (encoder.template, encoder.layers) == ("mistral3", (10, 20, 30))
    condition = encoder.encode(encoder.params, encoder.tokenize(record["pipeline"]["prompts"]))
    assert relative_gap(condition.context, arrays["mistral3.context"]) < FORWARD
    np.testing.assert_array_equal(np.asarray(condition.guidance), [4.0, 4.0])
    tensors = component_tensors(directory, "text_encoder")
    assert {layout.name for layout in layouts} == {f"text_encoder/{name}" for name in tensors}
    assert any("vision_tower." in name for name in tensors)
    for layout in layouts:
        written = layout.export({"encoders": {"conditioning": encoder.params}})
        assert written.tobytes() == tensors[layout.name.removeprefix("text_encoder/")].tobytes(), layout.name
