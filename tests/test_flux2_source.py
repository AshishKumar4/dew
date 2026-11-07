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

import contextlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tool_support
from interop_support import assert_transformer_exports, extract_fixture, fixture_arrays, source_transformer
from reference_error import as_given, assert_as_exact, torch_with_gradients, with_gradients

from dew.diffusion.process import DenoisingCondition
from dew.interop.diffusion import component_tensors, flux2_fields, flux2_path
from dew.nn.autoencoders.flux2 import load_flux2_vae
from dew.nn.backbones.flux2 import Flux2Transformer
from dew.nn.scan_orders import pixel_shuffle

ROOT = Path(__file__).resolve().parents[1]
CASES = ("dev", "rect", "unguided", "narrow")
CASE_SEEDS = {"dev": 8, "rect": 8, "unguided": 8, "narrow": 8}
"""Each transformer's declared seeds for check 1 (`assert_as_exact`), sized paired by `seeds_for` with
Dew's arm, which shares the reference's float32 inputs (2026-10-27, on armada): the reference compared
with itself does not resolve within 512 seeds for three of them, its errors heavy-tailed across draws."""


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
    assert_transformer_exports(source, name, layouts, params)


def flux2_arms(source, arrays, record, name):
    """The case's draw, Dew's compiled walk and Diffusers' `Flux2Transformer2DModel` walks in float32
    and float64: each the forward, then the gradients of the latent, the text states and every
    parameter (by its published name) under the draw's probe, as `Flux2Pipeline` runs the model."""
    import torch
    from diffusers import Flux2Transformer2DModel

    from tools.diffusers_reference_helpers import flux2_ids
    from tools.torch_precision import float64, float64_twin

    model, params, layouts = load(source, name)
    rows, columns = record["cases"][name]["grid"]
    tensors = {layout.name.removeprefix("transformer/"): layout for layout in layouts}
    fixture = {key: np.asarray(arrays[f"{name}.{key}"], np.float32)
               for key in ("latent", "context", "times", "guidance", "probe")}

    def draw(seed):
        """The fixture's inputs at seed 0, else normal draws of their shapes at `seed`, each guidance
        value perturbed by a unit-normal draw."""
        if seed == 0:
            return (fixture,)
        generator = np.random.default_rng(31 + seed)
        drawn = {key: generator.standard_normal(fixture[key].shape).astype(np.float32)
                 for key in ("latent", "context", "probe")}
        shift = generator.standard_normal(fixture["guidance"].shape).astype(np.float32)
        guidance = fixture["guidance"] + shift
        return ({**drawn, "times": fixture["times"], "guidance": guidance},)

    guided = fixture["guidance"].size > 0

    @jax.jit
    def walk(latent, context, times, guidance, params, probe):
        """The forward and its gradients, one program for every draw of the case."""
        def forward(latent, context, params):
            batch = latent.shape[0]
            image = latent.reshape(batch, rows, columns, -1)
            out = model.apply({"params": params}, image, times,
                              DenoisingCondition(context, guidance=guidance if guided else None))
            return out.reshape(batch, rows * columns, -1)

        return with_gradients(forward)(latent, context, params, probe)

    @as_given
    def native(inputs):
        out, grad_latent, grad_context, grad_params = walk(
            inputs["latent"], inputs["context"], inputs["times"], inputs["guidance"], params, inputs["probe"])
        return out, grad_latent, grad_context, {
            tensor: np.asarray(layout.export({"params": grad_params})) for tensor, layout in tensors.items()}

    checkpoint = source / name / "transformer"
    narrow = Flux2Transformer2DModel.from_pretrained(checkpoint, torch_dtype=torch.float32,
                                                     local_files_only=True).eval()
    # The float64 truth's model, built as every twin is (`float64_twin`): its config's class at float64,
    # the published weights loaded into it strictly.
    wide = float64_twin(narrow)
    weights = {id(module): {key: value.detach().numpy() for key, value in module.named_parameters()}
               for module in (narrow, wide)}
    # The pipeline's four-axis ids, as the fixture's walk laid them out for its text and latent.
    text_ids, image_ids = flux2_ids(rows, columns, fixture["context"].shape[1])

    def reference_walk(module, drawn):
        dtype = torch.from_numpy(np.asarray(drawn["latent"])).dtype
        times = torch.tensor(drawn["times"], dtype=dtype)
        guidance = torch.tensor(drawn["guidance"], dtype=dtype) if drawn["guidance"].size else None

        def forward(latent, context, params):
            return torch.func.functional_call(module, params, (), {
                "hidden_states": latent, "encoder_hidden_states": context, "timestep": times / 1000,
                "guidance": guidance, "img_ids": image_ids, "txt_ids": text_ids, "return_dict": False})[0]

        out, grad_latent, grad_context, grad_params = torch_with_gradients(forward)(
            drawn["latent"], drawn["context"], weights[id(module)], drawn["probe"])
        return out, grad_latent, grad_context, {tensor: grad_params[tensor] for tensor in tensors}

    @as_given
    def truth(inputs):
        with float64():
            return reference_walk(wide, inputs)

    return draw, native, as_given(lambda inputs: reference_walk(narrow, inputs)), truth


@pytest.mark.parametrize("name", CASES)
def test_the_forward_and_every_gradient_match_the_source(source, arrays, record, name):
    """The forward, the latent's and the text states' gradients and every parameter's, each as near
    float64 as Diffusers' float32 walk of the published model (check 1), over the variants whose
    wiring differs: distilled guidance, a rectangular grid, no guidance embedder, a narrower model."""
    draw, native, reference, truth = flux2_arms(source, arrays, record, name)
    assert_as_exact(native, reference, draw, f"FLUX.2 {name}", seeds=CASE_SEEDS[name], truth_fn=truth)


def nhwc(array):
    return np.asarray(array).transpose(0, 2, 3, 1)


PIPELINE_SEEDS = 8
"""The autoencoder's, encoders' and pipeline's declared seeds for check 1, sized by `seeds_for` paired as
each check confirmed them."""


def torch_arms(run):
    """`run(dtype, *inputs)` as given in float32, and in float64 inside `float64` (each a reference arm), its
    tensors as NumPy arrays."""
    import torch

    from tools.torch_precision import float64

    def arm(dtype):
        @as_given
        def given(*inputs):
            with float64() if dtype == torch.float64 else contextlib.nullcontext(), torch.no_grad():
                found = run(dtype, *inputs)
                return jax.tree.map(
                    lambda out: out.detach().numpy() if isinstance(out, torch.Tensor) else out, found)
        return given
    return arm(torch.float32), arm(torch.float64)


def vae_walk(model, dtype, inputs):
    """`model` walked as `Flux2Pipeline` walks it at `dtype`: `_encode_vae_image` (posterior mode, 2x2 fold,
    batch-norm normalization) of the draw's image and the call's decode of its code (the inverse, then
    `vae.decode`), with the gradients of the draw's probes against the pixels and the code."""
    import torch
    from diffusers import Flux2Pipeline

    tensors = {key: torch.from_numpy(np.asarray(value)).to(dtype) for key, value in inputs.items()}
    statistics = (model.bn.running_mean.view(1, -1, 1, 1),
                  torch.sqrt(model.bn.running_var.view(1, -1, 1, 1) + model.config.batch_norm_eps))
    with torch.enable_grad():
        pixels_in = tensors["image"].requires_grad_()
        latent = (Flux2Pipeline._patchify_latents(model.encode(pixels_in).latent_dist.mode())
                  - statistics[0]) / statistics[1]
        (grad_image,) = torch.autograd.grad((latent * tensors["probe_latent"]).sum(), [pixels_in])
        code_in = tensors["code"].requires_grad_()
        unfolded = Flux2Pipeline._unpatchify_latents(code_in * statistics[1] + statistics[0])
        pixels = model.decode(unfolded).sample
        (grad_code,) = torch.autograd.grad((pixels * tensors["probe"]).sum(), [code_in])
    outputs = {"latent": latent, "pixels": pixels, "grad_image": grad_image, "grad_code": grad_code}
    return {key: nhwc(value.detach()) for key, value in outputs.items()}


@pytest.mark.parametrize("name", ("vae", "small_decoder"))
def test_the_autoencoder_folds_and_normalizes_as_the_pipeline_does(source, name):
    """The pipeline's folded, normalized latent of an image and its decode of a code, and the gradients of
    probes against both, on par with Diffusers' `AutoencoderKLFlux2` walked as `Flux2Pipeline` walks it, over
    drawn images, codes and probes (check 1), its float64 twin the truth."""
    import torch
    from diffusers import AutoencoderKLFlux2

    from tools.torch_precision import float64_twin

    autoencoder, params, layouts, _ = load_flux2_vae(source / name)
    tensors = component_tensors(source / name, "vae")
    assert {layout.name for layout in layouts} == {
        f"vae/{key}" for key in tensors if not key.startswith("bn.")
    }
    narrow = AutoencoderKLFlux2.from_pretrained(source / name / "vae", local_files_only=True).eval()
    models = {torch.float32: narrow, torch.float64: float64_twin(narrow)}

    def draw(seed):
        generator = np.random.default_rng([241, seed])
        return ({"image": generator.uniform(-1, 1, (2, 3, 16, 24)).astype(np.float32),
                 "probe_latent": generator.standard_normal((2, 16, 4, 6)).astype(np.float32),
                 "code": generator.standard_normal((2, 16, 4, 6)).astype(np.float32),
                 "probe": generator.standard_normal((2, 3, 16, 24)).astype(np.float32)},)

    def walked(inputs):
        image, probe_latent, code, probe = (jnp.transpose(inputs[key], (0, 2, 3, 1))
                                            for key in ("image", "probe_latent", "code", "probe"))
        latent = autoencoder.encode(params, image)
        grad_image = jax.grad(lambda image: jnp.sum(autoencoder.encode(params, image) * probe_latent))(image)
        pixels = autoencoder.decode(params, code)
        grad_code = jax.grad(lambda code: jnp.sum(autoencoder.decode(params, code) * probe))(code)
        return {"latent": latent, "pixels": pixels, "grad_image": grad_image, "grad_code": grad_code}

    reference, truth = torch_arms(lambda dtype, inputs: vae_walk(models[dtype], dtype, inputs))
    assert_as_exact(jax.jit(walked), reference, draw, f"{name} walk", seeds=PIPELINE_SEEDS, truth_fn=truth)


@pytest.mark.network
def test_the_published_autoencoder_matches_the_source(monkeypatch):
    """FLUX.2's VAE as `FLUX.2-klein-4B` publishes it (the one FLUX.2 [dev] ships), downloaded, over drawn
    smooth 128x192 images: the pipeline's folded, normalized latent and a corner of its decode on par with
    the source run in float32 on the device Dew runs on, its matmuls and convolutions in fp32 (no TF32)
    (check 1), the truth the source's float64 twin on the CPU. On an A100 cuDNN's fp32 convolutions put the
    source's latent 2.3 times as far from float64 as the CPU's arithmetic does, Dew's alike (ledger 1703),
    so a GPU run is judged against the source on its GPU."""
    import torch

    from tools.torch_precision import float64, float64_twin

    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    tool = tool_support.load("diffusers_flux2_reference")
    repo, revision = tool.PUBLISHED_VAE
    autoencoder, params, _, _ = load_flux2_vae(repo, revision=revision)
    device = {"gpu": "cuda", "cpu": "cpu"}[jax.devices()[0].platform]
    if device == "cuda" and not torch.cuda.is_available():
        pytest.fail("the source runs on the GPU Dew runs on, and this torch has no CUDA")
    source = tool.published_vae()
    wide = float64_twin(source)
    source = source.to(device)

    def draw(seed):
        smooth = tool.smooth_image()
        noise = np.random.default_rng([231, seed]).standard_normal(smooth.shape) * 0.05
        return (np.clip(smooth + noise, -1, 1).astype(np.float32),)

    @as_given
    def walked(image):
        latent = autoencoder.encode(params, nhwc(image))
        pixels = autoencoder.decode(params, latent)[:, :tool.CROP, :tool.CROP]
        return {"latent": np.asarray(latent), "pixels": np.asarray(pixels)}

    def source_walk(model, dtype, image):
        with torch.no_grad():
            mean = model.bn.running_mean.view(1, -1, 1, 1)
            std = torch.sqrt(model.bn.running_var.view(1, -1, 1, 1) + model.config.batch_norm_eps)
            folded = Flux2Pipeline._patchify_latents(model.encode(
                torch.from_numpy(image).to(next(model.parameters()).device, dtype)).latent_dist.mode())
            pixels = model.decode(Flux2Pipeline._unpatchify_latents(folded)).sample
            return {"latent": nhwc(((folded - mean) / std).cpu()), "pixels": nhwc(pixels.cpu())[:, :tool.CROP,
                                                                                                :tool.CROP]}

    from diffusers import Flux2Pipeline

    held = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        @as_given
        def truth(image):
            with float64():
                return source_walk(wide, torch.float64, image)

        assert_as_exact(walked, as_given(lambda image: source_walk(source, torch.float32, image)), draw,
                        "published FLUX.2 VAE", seeds=PIPELINE_SEEDS, truth_fn=truth)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = held


@pytest.fixture(scope="module")
def klein(source):
    from dew.interop.pretrained import Pretrained

    return Pretrained.load(str(source / "pipeline"), dtype="float32", attention_impl="xla")


WORDS = ("a", "red", "green", "blue", "cat", "bird", "on", "mat", "tiny", "photo", "of", "tree")
"""The words a drawn prompt is made of."""


def prompt_draw(seed: int) -> tuple[np.ndarray]:
    """Two prompts as word indices, the first five words long and the second two."""
    generator = np.random.default_rng([247, seed])
    return (generator.integers(0, len(WORDS), 5).astype(np.int32),
            generator.integers(0, len(WORDS), 2).astype(np.int32))


def words(*rows) -> list[str]:
    return [" ".join(WORDS[index] for index in row) for row in rows]


@pytest.fixture(scope="module")
def klein_pipelines(source):
    """The saved [klein] pipeline Diffusers loads in float32, and its float64 twin (each module's
    `float64_twin`) with the run that widens it: `walk(pipe, torch.float64)` inside `float64`, the
    flow-matching scheduler's own tables widened around every call into it (`float64_scheduler`)."""
    import copy

    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.schedulers import scheduling_flow_match_euler_discrete

    from tools.torch_precision import float64, float64_twin

    with pytest.MonkeyPatch.context() as patched:
        patched.syspath_prepend(str(ROOT / "tools"))
        tool = tool_support.load("diffusers_wan_reference")

    def widened_walk(walk, model):
        with float64(), tool.float64_scheduler(scheduling_flow_match_euler_discrete):
            return walk(model, torch.float64)
    narrow = Flux2KleinPipeline.from_pretrained(source / "pipeline", torch_dtype=torch.float32,
                                                local_files_only=True)
    narrow.set_progress_bar_config(disable=True)
    wide = copy.deepcopy(narrow)
    for name, component in narrow.components.items():
        if isinstance(component, torch.nn.Module):
            setattr(wide, name, float64_twin(component))
    return narrow, wide, widened_walk


def klein_arms(pipelines, run):
    """`run(pipe, dtype, *inputs)` as given over the float32 pipeline, and over its float64 twin widened,
    its tensors as NumPy arrays."""
    import torch

    narrow, wide, widened_walk = pipelines

    def numpy(tree):
        return jax.tree.map(lambda out: out.numpy() if isinstance(out, torch.Tensor) else out, tree)

    @as_given
    def reference(*inputs):
        with torch.no_grad():
            return numpy(run(narrow, torch.float32, *inputs))

    @as_given
    def truth(*inputs):
        with torch.no_grad():
            return numpy(widened_walk(lambda pipe, dtype: run(pipe, dtype, *inputs), wide))
    return reference, truth


def test_klein_prompt_encoding_matches_the_source_pipeline(klein, klein_pipelines, record):
    """The conditioner stacks what `encode_prompt` stacks: the chat template with thinking off, every row
    padded on the right to 512 tokens, and the outputs of layers 9, 18 and 27 side by side, pads included;
    on par with it over drawn prompts and the empty one (check 1), its float64 twin the truth."""
    import torch

    encoder = klein.inputs.conditions["conditioning"].encoder
    params = klein.variables["encoders"]["conditioning"]

    def encoded(*rows):
        return encoder.encode(params, encoder.tokenize([*words(*rows), ""])).context

    def encode_prompt(pipe, dtype, *rows):
        return torch.cat([pipe.encode_prompt(prompt=prompt, device=torch.device("cpu"))[0]
                          for prompt in [*words(*rows), ""]])

    condition = encoder.encode(params, encoder.tokenize(["", ""]))
    assert condition.context.shape == (2, 512, record["pipeline"]["config"]["joint_attention_dim"])
    assert condition.guidance is None
    reference, truth = klein_arms(klein_pipelines, encode_prompt)
    assert_as_exact(as_given(encoded), reference, prompt_draw, "[klein] prompt encoding",
                    seeds=PIPELINE_SEEDS, truth_fn=truth)
    with pytest.raises(ValueError):
        encoder.tokenize(["x" * 513])


def test_klein_pipeline_walk_matches_the_source(klein, klein_pipelines, record):
    """`Pretrained.load().text_to_image()` reproduces the source's own call: its defaults (50 steps, two
    branches guided at 4.0 against the empty prompt), and at the recorded step count the sigmas it lays out
    shifted by its empirical mu, the latent it ends on and the image it decodes, on par with it over drawn
    prompts and starting latents (check 1), its float64 twin the truth."""
    import torch

    pipeline = record["pipeline"]
    task = klein.text_to_image()
    assert task.steps == 50 and task.guidance is not None and task.guidance.scale == 4.0
    autoencoder = klein.autoencoder
    height, width, steps = 16, 24, pipeline["steps"]

    def draw(seed):
        latents = np.random.default_rng([251, seed]).standard_normal((2, 16, height // 4, width // 4))
        return (*prompt_draw(seed), latents.astype(np.float32))

    @as_given
    def walked(first, second, latents):
        initial = latents.transpose(0, 2, 3, 1)
        walk = task(task.prepare(words(first, second), initial=initial, key=0, steps=steps),
                    key=jax.random.PRNGKey(0)).host()
        raw = pixel_shuffle(np.asarray(walk.latents) / autoencoder.latent_scale + autoencoder.latent_shift)
        return {"latents": raw, "images": np.clip(np.asarray(walk.images) / 2 + 0.5, 0.0, 1.0)}

    def called(pipe, dtype, first, second, latents):
        rows = {"latents": [], "images": []}
        for row, prompt in enumerate(words(first, second)):
            call = {"prompt": prompt, "height": height, "width": width, "num_inference_steps": steps}
            start = torch.from_numpy(latents[row:row + 1]).to(dtype)
            rows["latents"].append(nhwc(pipe(**call, latents=start.clone(), output_type="latent").images)[0])
            rows["images"].append(pipe(**call, latents=start.clone(), output_type="np").images[0])
        return {key: np.stack(value) for key, value in rows.items()}

    reference, truth = klein_arms(klein_pipelines, called)
    assert_as_exact(walked, reference, draw, "[klein] pipeline walk", seeds=PIPELINE_SEEDS, truth_fn=truth)


def assert_mistral3_reads_as_flux2_dev(directory, encoder) -> None:
    """The encoder's prompt states on par with `Flux2Pipeline._get_mistral_3_small_prompt_embeds` over the
    tiny Mistral-3 and its tokenizer, over drawn prompts (check 1), its float64 twin the truth."""
    import torch
    from diffusers import Flux2Pipeline
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    from tools.torch_precision import float64_twin

    tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer", local_files_only=True)
    narrow = AutoModelForImageTextToText.from_pretrained(directory / "text_encoder",
                                                         local_files_only=True).eval()
    models = {torch.float32: narrow, torch.float64: float64_twin(narrow)}

    def embeds(dtype, *rows):
        return Flux2Pipeline._get_mistral_3_small_prompt_embeds(models[dtype], tokenizer, words(*rows),
                                                                dtype=dtype, device=torch.device("cpu"))

    reference, truth = torch_arms(embeds)
    def encoded(*rows):
        return encoder.encode(encoder.params, encoder.tokenize(words(*rows))).context

    assert_as_exact(as_given(encoded), reference, prompt_draw, "mistral3 context", seeds=PIPELINE_SEEDS,
                    truth_fn=truth)


def test_a_mistral3_encoder_reads_as_flux2_dev_reads_it(source):
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
    condition = encoder.encode(encoder.params, encoder.tokenize(words(*prompt_draw(0))))
    np.testing.assert_array_equal(np.asarray(condition.guidance), [4.0, 4.0])
    assert_mistral3_reads_as_flux2_dev(directory, encoder)
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
