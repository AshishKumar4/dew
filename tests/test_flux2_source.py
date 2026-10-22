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
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tool_support
from interop_support import assert_transformer_exports, extract_fixture, fixture_arrays, source_transformer
from reference_error import assert_as_exact, assert_as_exact_as_the_reference

from dew.diffusion.process import DenoisingCondition
from dew.interop.diffusion import component_tensors, flux2_fields, flux2_path
from dew.nn.autoencoders.flux2 import load_flux2_vae
from dew.nn.backbones.flux2 import Flux2Transformer
from dew.nn.scan_orders import pixel_shuffle

ROOT = Path(__file__).resolve().parents[1]
CASES = ("dev", "rect", "unguided", "narrow")
CASE_SEEDS = {"dev": 256, "rect": 256, "unguided": 64, "narrow": 256}
"""Check 1's declared seeds for each case, sized by `seeds_for` on an A100 node's CPU, where the
reference runs."""
ROWS = 4
"""Each draw holds the case's batch ROWS times over, so a parameter's gradient sums over more rows and
one draw's error varies less: at the fixture's batch the dev and rect cases did not resolve within 512
seeds."""


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


def load(source, name):
    return source_transformer(source, name, Flux2Transformer, flux2_fields, flux2_path)


def flux2_walk(model, params, grid):
    """A compiled native forward and every gradient on a draw: the latent and the probe are the
    source's packed rows, folded to the grid and back."""
    rows, columns = grid

    @jax.jit
    def walk(inputs):
        batch = inputs["latent"].shape[0]
        latent = inputs["latent"].reshape(batch, rows, columns, -1)
        probe = inputs["probe"].reshape(batch, rows, columns, -1)
        guidance = inputs["guidance"] if inputs["guidance"].size else None

        def objective(params, latent, context):
            output = model.apply({"params": params}, latent, inputs["times"],
                                 DenoisingCondition(context, guidance=guidance))
            return jnp.sum(output * probe), output

        (_, output), gradients = jax.value_and_grad(objective, argnums=(0, 1, 2), has_aux=True)(
            params, latent, inputs["context"])
        return {"output": output.reshape(batch, rows * columns, -1),
                "grad_latent": gradients[1].reshape(batch, rows * columns, -1),
                "grad_context": gradients[2], "grad_params": gradients[0]}

    return walk


@pytest.mark.parametrize("name", CASES)
def test_every_published_tensor_maps_and_exports_bit_identical(source, name):
    _, params, layouts = load(source, name)
    assert_transformer_exports(source, name, layouts, params)


@pytest.mark.parametrize("name", CASES)
def test_the_forward_and_every_gradient_match_the_source(source, arrays, record, name):
    """The forward, the latent's and the context's gradients and every parameter's, each as near
    float64 as Diffusers' own float32 walk (`assert_as_exact`), over the fixture's draw and fresh
    ones: the latent, the context and the probe drawn again, each embedded guidance moved by a unit
    normal."""
    import torch
    from diffusers import Flux2Transformer2DModel

    from tools.diffusers_wan_reference import float64

    model, params, layouts = load(source, name)
    rows, columns = record["cases"][name]["grid"]
    walk = flux2_walk(model, params, (rows, columns))
    tensors = {layout.name.removeprefix("transformer/"): layout for layout in layouts}
    original = {key: np.concatenate([np.array(arrays[f"{name}.{key}"])] * ROWS)
                for key in ("latent", "context", "times", "guidance", "probe")}

    def draw(seed):
        """The fixture's inputs, its batch ROWS times over, at seed 0, else fresh normal draws of
        those shapes."""
        if seed == 0:
            return (original,)
        generator = torch.Generator().manual_seed(29 + seed)
        drawn = {key: torch.randn(original[key].shape, generator=generator).numpy()
                 for key in ("latent", "context", "probe")}
        guidance = original["guidance"] + torch.randn(original["guidance"].shape, generator=generator).numpy()
        return ({**drawn, "times": original["times"], "guidance": guidance.astype(np.float32)},)

    checkpoint = source / name / "transformer"
    narrow_model = Flux2Transformer2DModel.from_pretrained(
        checkpoint, torch_dtype=torch.float32, local_files_only=True).eval()
    with float64():
        wide_model = Flux2Transformer2DModel.from_pretrained(
            checkpoint, torch_dtype=torch.float64, local_files_only=True).eval()
    text_ids, image_ids = tool_support.load("diffusers_flux2_reference").ids(rows, columns)

    def reference_walk(model, dtype, drawn):
        inputs = [torch.tensor(drawn[key], dtype=dtype).requires_grad_() for key in ("latent", "context")]
        named = list(model.named_parameters())
        guidance = torch.tensor(drawn["guidance"], dtype=dtype) if drawn["guidance"].size else None
        output = model(hidden_states=inputs[0], encoder_hidden_states=inputs[1],
                       timestep=torch.tensor(drawn["times"], dtype=dtype) / 1000, guidance=guidance,
                       img_ids=image_ids, txt_ids=text_ids, return_dict=False)[0]
        gradients = torch.autograd.grad((output * torch.tensor(drawn["probe"], dtype=dtype)).sum(),
                                        inputs + [value for _, value in named])
        return {"output": output.detach().numpy(), "grad_latent": gradients[0].numpy(),
                "grad_context": gradients[1].numpy(),
                **{f"grad_param.{key}": gradient.numpy()
                   for (key, _), gradient in zip(named, gradients[2:], strict=True)}}

    def truth(inputs):
        with float64():
            return reference_walk(wide_model, torch.float64, inputs)

    def native(inputs):
        values = walk(inputs)
        gradients = values.pop("grad_params")
        return {**values, **{f"grad_param.{tensor}": layout.export({"params": gradients})
                             for tensor, layout in tensors.items()}}

    assert_as_exact(native, lambda inputs: reference_walk(narrow_model, torch.float32, inputs), draw,
                    f"Flux.2 {name}", seeds=CASE_SEEDS[name], truth_fn=truth)


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
    for key, value in (("latent", latent), ("pixels", pixels), ("grad_image", grad_image),
                       ("grad_code", grad_code)):
        assert_as_exact_as_the_reference(value, nhwc(arrays[f"{name}.{key}"]),
                                         nhwc(arrays[f"{name}.fp64.{key}"]), f"{name} {key}")


@pytest.mark.network
def test_the_published_autoencoder_matches_the_source(monkeypatch):
    """FLUX.2's VAE as `FLUX.2-klein-4B` publishes it (the one FLUX.2 [dev]
    ships), downloaded, on a smooth 128x192 image, against the source run in
    float32 and float64 (`flux2_published.npz`): the pipeline's folded,
    normalized latent and a corner of its decode, each as near float64 as the
    source's float32."""
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    tool = tool_support.load("diffusers_flux2_reference")
    reference = np.load(ROOT / "tests/fixtures/flux2_published.npz")
    repo, revision = tool.PUBLISHED_VAE
    autoencoder, params, _, _ = load_flux2_vae(repo, revision=revision)
    latent = autoencoder.encode(params, nhwc(tool.smooth_image()))
    pixels = autoencoder.decode(params, nhwc(reference["vae.latent.fp64"]))[:, :tool.CROP, :tool.CROP]
    assert latent.shape == (1, 8, 12, 128)
    for key, value in (("latent", latent), ("pixels", pixels)):
        assert_as_exact_as_the_reference(value, nhwc(reference[f"vae.{key}"]),
                                         nhwc(reference[f"vae.{key}.fp64"]), f"published {key}")


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
        assert_as_exact_as_the_reference(condition.context[row], arrays[f"pipeline.context.{row}"][0],
                                         arrays[f"pipeline.fp64.context.{row}"][0], f"prompt {row}")
    with pytest.raises(ValueError):
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
        assert_as_exact_as_the_reference(raw[row], nhwc(arrays[f"pipeline.latents.{row}"])[0],
                                         nhwc(arrays[f"pipeline.fp64.latents.{row}"])[0], f"latents {row}")
        assert_as_exact_as_the_reference(images[row], arrays[f"pipeline.images.{row}"][0],
                                         arrays[f"pipeline.fp64.images.{row}"][0], f"images {row}")


def test_a_mistral3_encoder_reads_as_flux2_dev_reads_it(source, arrays, record):
    """FLUX.2 [dev]'s conditioning over a tiny Mistral-3 encoder with Mistral
    Small 3.1's chat template: the system and user turns, 512 tokens padded
    on the right, the hidden states after layers 10, 20 and 30 stacked, and
    the guidance [dev] embeds. The vision tower and projector are carried as
    stored, not held as variables: the variables are the leaves the language
    model's layouts bind, and those layouts with the carried tensors export
    every stored tensor as it came."""
    from flax.traverse_util import flatten_dict

    from dew.interop.pipeline_assembly import _hidden_states_conditioning

    directory = source / "mistral3"
    encoder, layouts, retained, _ = _hidden_states_conditioning(
        directory, {}, jnp.float32, pipeline="flux2", tokens=512, guidance=4.0,
        param_dtype="float32", attention_impl="xla")
    assert (encoder.template, encoder.layers) == ("mistral3", (10, 20, 30))
    condition = encoder.encode(encoder.params, encoder.tokenize(record["pipeline"]["prompts"]))
    assert_as_exact_as_the_reference(condition.context, arrays["mistral3.context"],
                                     arrays["mistral3.fp64.context"], "mistral3 context")
    np.testing.assert_array_equal(np.asarray(condition.guidance), [4.0, 4.0])
    tensors = component_tensors(directory, "text_encoder")
    assert set(flatten_dict(encoder.params)) == {layout.paths[0][2:] for layout in layouts}
    assert {f"text_encoder/{name}" for name in tensors} == {layout.name for layout in layouts} | set(retained)
    assert any("vision_tower." in name for name in retained)
    assert any("multi_modal_projector." in name for name in retained)
    variables = {"encoders": {"conditioning": encoder.params}}
    written = {layout.name: layout.export(variables) for layout in layouts}
    for name, tensor in {**written, **retained}.items():
        stored = tensors[name.removeprefix("text_encoder/")]
        assert (tensor.dtype, tensor.tobytes()) == (stored.dtype, stored.tobytes()), name


def test_a_mistral3_encoder_storing_a_tensor_twice_reads_once_and_writes_both(source, record, tmp_path):
    """A checkpoint that stores its embedding under both of Mistral-3's
    spellings (transformers 4.50's `language_model.model.` and 4.52's
    `model.language_model.`) reads the equal copies once, conditions as the
    checkpoint with one copy does, and writes each name back as it came."""
    import shutil

    from safetensors.numpy import save_file

    from dew.interop.pipeline_assembly import _hidden_states_conditioning

    directory = tmp_path / "mistral3"
    shutil.copytree(source / "mistral3", directory)
    tensors = component_tensors(directory, "text_encoder")
    name = next(name for name in tensors if name.endswith("embed_tokens.weight"))
    spellings = ("model.language_model.", "language_model.model.")
    alias = next(other + name.removeprefix(spelling) for spelling, other in (spellings, spellings[::-1])
                 if name.startswith(spelling))
    save_file({**tensors, alias: tensors[name]}, str(directory / "text_encoder" / "model.safetensors"))

    def conditioned(where):
        encoder, layouts, *_ = _hidden_states_conditioning(
            where, {}, jnp.float32, pipeline="flux2", tokens=512, guidance=4.0, param_dtype="float32",
            attention_impl="xla")
        return encoder, layouts

    (encoder, layouts), (single, _) = conditioned(directory), conditioned(source / "mistral3")
    prompts = encoder.tokenize(record["pipeline"]["prompts"])
    np.testing.assert_array_equal(np.asarray(encoder.encode(encoder.params, prompts).context),
                                  np.asarray(single.encode(single.params, prompts).context))
    variables = {"encoders": {"conditioning": encoder.params}}
    written = {layout.name: layout.export(variables) for layout in layouts}
    for each in (name, alias):
        assert written[f"text_encoder/{each}"].tobytes() == tensors[name].tobytes(), each
