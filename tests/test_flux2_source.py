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
from interop_support import (
    assert_transformer_exports,
    drawn_prompts,
    extract_fixture,
    fixture_arrays,
    source_transformer,
)
from reference_error import assert_as_exact, torch_with_gradients, with_gradients

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
    """A compiled native forward on a draw, and the gradients of the latent, the context and every
    parameter under the draw's `probe`, its cotangent (`with_gradients`): the latent and the probe
    are the source's packed rows, folded to the grid and back."""
    rows, columns = grid

    @jax.jit
    def walk(inputs):
        batch = inputs["latent"].shape[0]
        guidance = inputs["guidance"] if inputs["guidance"].size else None

        def forward(latent, context, params):
            folded = latent.reshape(batch, rows, columns, -1)
            output = model.apply({"params": params}, folded, inputs["times"],
                                 DenoisingCondition(context, guidance=guidance))
            return output.reshape(batch, rows * columns, -1)

        return with_gradients(forward)(inputs["latent"], inputs["context"], params, inputs["probe"])

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

    weights = {model: {key: value.detach().numpy() for key, value in model.named_parameters()}
               for model in (narrow_model, wide_model)}

    def reference_walk(model, drawn):
        """The source's forward in the draw's dtype, and the same gradients under the same cotangent
        as `flux2_walk`'s (`torch_with_gradients`)."""
        dtype = torch.from_numpy(np.asarray(drawn["latent"])).dtype
        guidance = torch.tensor(drawn["guidance"], dtype=dtype) if drawn["guidance"].size else None

        def forward(latent, context, params):
            return torch.func.functional_call(model, params, (), {
                "hidden_states": latent, "encoder_hidden_states": context,
                "timestep": torch.tensor(drawn["times"], dtype=dtype) / 1000, "guidance": guidance,
                "img_ids": image_ids, "txt_ids": text_ids, "return_dict": False})[0]

        return torch_with_gradients(forward)(drawn["latent"], drawn["context"], weights[model],
                                             drawn["probe"])

    def truth(inputs):
        with float64():
            return reference_walk(wide_model, inputs)

    def native(inputs):
        *values, gradients = walk(inputs)
        return (*values, {tensor: layout.export({"params": gradients}) for tensor, layout in tensors.items()})

    assert_as_exact(native, lambda inputs: reference_walk(narrow_model, inputs), draw,
                    f"Flux.2 {name}", seeds=CASE_SEEDS[name], truth_fn=truth)


def nhwc(array):
    """A host array, channels first, channels last."""
    return np.asarray(array).transpose(0, 2, 3, 1)


VAE_SEEDS = {"vae": 8, "small_decoder": 16, "published": 16}
"""Check 1's declared seeds for each autoencoder, sized by `seeds_for` (the published one on a GPU, its
download a network test's)."""


def source_autoencoder_walk(model, device=None):
    """The source's walk of its autoencoder as `Flux2Pipeline` walks it, in the draw's dtype: its
    folded, normalized latent of an image and that latent's gradient under a cotangent, and its
    decode of a latent code (the inverse normalization, the unfold, `vae.decode`) and the code's
    gradient under a cotangent (`torch_with_gradients`, on `device`, the model's), channels first."""
    import torch
    from diffusers import Flux2Pipeline

    def statistics(dtype):
        return (model.bn.running_mean.view(1, -1, 1, 1).to(dtype),
                torch.sqrt(model.bn.running_var.view(1, -1, 1, 1) + model.config.batch_norm_eps).to(dtype))

    def encode(pixels):
        mean, scale = statistics(pixels.dtype)
        return (Flux2Pipeline._patchify_latents(model.encode(pixels).latent_dist.mode()) - mean) / scale

    def decode(code):
        mean, scale = statistics(code.dtype)
        return model.decode(Flux2Pipeline._unpatchify_latents(code * scale + mean)).sample

    def walk(image, probe_latent, code, probe):
        latent, grad_image = torch_with_gradients(encode, device=device)(image, probe_latent)
        pixels, grad_code = torch_with_gradients(decode, device=device)(code, probe)
        return {"latent": latent, "grad_image": grad_image, "pixels": pixels, "grad_code": grad_code}
    return walk


def autoencoder_check(autoencoder, params, narrow, wide, draw, label, seeds):
    """Dew's autoencoder walk, channels last, against the source's (`source_autoencoder_walk`) over
    `draw`'s images, codes and cotangents, its float64 walk the truth (check 1)."""
    from tools.diffusers_wan_reference import float64

    def traced_nhwc(array):
        return jnp.transpose(array, (0, 2, 3, 1))

    @jax.jit
    def native(image, probe_latent, code, probe):
        latent, grad_image = with_gradients(lambda image: autoencoder.encode(params, image))(
            traced_nhwc(image), traced_nhwc(probe_latent))
        pixels, grad_code = with_gradients(lambda code: autoencoder.decode(params, code))(
            traced_nhwc(code), traced_nhwc(probe))
        return {"latent": latent, "grad_image": grad_image, "pixels": pixels, "grad_code": grad_code}

    def channels_last(walk):
        return lambda *drawn: {key: nhwc(value) for key, value in walk(*drawn).items()}

    def truth(*drawn):
        with float64():
            return channels_last(wide)(*drawn)

    assert_as_exact(native, channels_last(narrow), draw, label, seeds=seeds, truth_fn=truth)


@pytest.mark.parametrize("name", ("vae", "small_decoder"))
def test_the_autoencoder_folds_and_normalizes_as_the_pipeline_does(source, arrays, name):
    """The autoencoder's folded, normalized latent and its decode, and their gradients, on par with
    the source's own float32 walk over drawn images, codes and cotangents (check 1); every stored
    tensor but the batch norm's statistics binds a leaf."""
    import copy

    import torch
    from diffusers import AutoencoderKLFlux2

    from tools.diffusers_wan_reference import float64

    autoencoder, params, layouts, _ = load_flux2_vae(source / name)
    tensors = component_tensors(source / name, "vae")
    assert {layout.name for layout in layouts} == {
        f"vae/{key}" for key in tensors if not key.startswith("bn.")
    }
    model = AutoencoderKLFlux2.from_pretrained(source / name / "vae", torch_dtype=torch.float32,
                                               local_files_only=True).eval()
    with float64():
        wide = copy.deepcopy(model).double()
    shapes = {key: np.shape(arrays[f"{name}.{key}"]) for key in ("image", "probe_latent", "code", "probe")}
    assert autoencoder.latent_shape(shapes["image"][2:] + shapes["image"][1:2]) == (4, 6, 16)

    def draw(seed):
        generator = torch.Generator().manual_seed(41 + seed)
        image = torch.rand(shapes["image"], generator=generator) * 2 - 1
        return (image.numpy(), *(torch.randn(shapes[key], generator=generator).numpy()
                                 for key in ("probe_latent", "code", "probe")))

    walks = (source_autoencoder_walk(model), source_autoencoder_walk(wide))
    autoencoder_check(autoencoder, params, *walks, draw,
                      f"{name} autoencoder", VAE_SEEDS[name])


PUBLISHED_CROPS = 8
"""The noisy crops of the published autoencoder's draw, each its own noise, code and cotangent: its code
gradient reads a 4x6 code a crop, whose group norms read too few positions for one crop's code gradient
to resolve 1.5 within 64 seeds (the reference compared with itself, on an A100); a draw of eight
averages eight times the positions for far less than eight times a draw's cost."""


@pytest.mark.network
def test_the_published_autoencoder_matches_the_source(monkeypatch):
    """FLUX.2's VAE as `FLUX.2-klein-4B` publishes it (the one FLUX.2 [dev]
    ships), downloaded, against the source loaded beside it in float32 and
    float64: over drawn noisy versions of a 64x96 crop of a smooth image, codes
    and cotangents, the pipeline's folded, normalized latent, its whole decode
    and both their input gradients on par with the source's float32 (check 1,
    `autoencoder_check`). The crop keeps every published weight in the walk at
    a quarter of the full image's cost, which the float64 truth's sizing over
    seeds needs; at 32x48 the decoder's group norms read too few positions a
    group for its code gradient to resolve. The source runs on a CUDA device where there is one, its
    float32 in IEEE float32 (no TF32), its float64 truth in float64."""
    import copy

    import torch
    from diffusers import AutoencoderKLFlux2

    from tools.diffusers_wan_reference import float64

    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    tool = tool_support.load("diffusers_flux2_reference")
    repo, revision = tool.PUBLISHED_VAE
    autoencoder, params, _, _ = load_flux2_vae(repo, revision=revision)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    model = AutoencoderKLFlux2.from_pretrained(repo, subfolder="vae", revision=revision).eval().to(device)
    with float64():
        wide = copy.deepcopy(model).double()
    smooth = np.repeat(tool.smooth_image()[..., :64, :96], PUBLISHED_CROPS, axis=0)
    folded = (PUBLISHED_CROPS, 128, 4, 6)
    assert autoencoder.encode(params, nhwc(smooth)).shape == (PUBLISHED_CROPS, 4, 6, 128)

    def draw(seed):
        """PUBLISHED_CROPS copies of the smooth crop, each with fresh noise, and normal codes and
        cotangents."""
        rng = np.random.default_rng([19, seed])
        image = np.clip(smooth + 0.05 * rng.standard_normal(smooth.shape), -1, 1).astype(np.float32)
        return (image, *(rng.standard_normal(shape).astype(np.float32)
                         for shape in (folded, folded, smooth.shape)))

    walks = (source_autoencoder_walk(model, device), source_autoencoder_walk(wide, device))
    autoencoder_check(autoencoder, params, *walks, draw,
                      "published autoencoder", VAE_SEEDS["published"])


@pytest.fixture(scope="module")
def klein(source):
    from dew.interop.pretrained import Pretrained

    return Pretrained.load(str(source / "pipeline"), dtype="float32", attention_impl="xla")


PIPELINE_SEEDS = {"prompts": 8, "walk": 8, "mistral3": 8}
"""Check 1's declared seeds for the pipeline's prompt encoding, its walk and FLUX.2 [dev]'s Mistral-3
encoding, sized by `seeds_for` on armada."""
def klein_pipelines(source):
    """Diffusers' own `Flux2KleinPipeline` the fixture saved, in float32 and widened to float64."""
    import copy

    import torch
    from diffusers import Flux2KleinPipeline

    from tools.diffusers_wan_reference import float64

    narrow = Flux2KleinPipeline.from_pretrained(source / "pipeline", torch_dtype=torch.float32,
                                                local_files_only=True)
    narrow.set_progress_bar_config(disable=True)
    with float64():
        wide = copy.deepcopy(narrow).to(torch.float64)
    return narrow, wide


def widened(run):
    """`run()` with the source's float32 pins widened, as its float64 truth runs."""
    from tools.diffusers_wan_reference import float64

    with float64():
        return run()


def test_klein_prompt_encoding_matches_the_source_pipeline(klein, source, record):
    """The conditioner stacks what `encode_prompt` stacks: the chat template
    with thinking off, every row padded on the right to 512 tokens, and the
    outputs of layers 9, 18 and 27 side by side, pads included."""
    import torch

    encoder = klein.inputs.conditions["conditioning"].encoder
    params = klein.variables["encoders"]["conditioning"]
    narrow, wide = klein_pipelines(source)
    condition = encoder.encode(params, encoder.tokenize([*record["pipeline"]["prompts"], ""]))
    assert condition.context.shape == (3, 512, record["pipeline"]["config"]["joint_attention_dim"])
    assert condition.guidance is None

    def prompts_of(seed) -> tuple[str, ...]:
        return (*drawn_prompts(11, seed), "")

    def source_encoded(pipe, prompts):
        with torch.no_grad():
            return np.concatenate([pipe.encode_prompt(prompt=prompt, device=torch.device("cpu"))[0].numpy()
                                   for prompt in prompts])

    # The arms draw their prompts from the seed, a leaf the gate widens as it is.
    assert_as_exact(lambda seed: encoder.encode(params, encoder.tokenize(list(prompts_of(seed)))).context,
                    lambda seed: source_encoded(narrow, prompts_of(seed)), lambda seed: (np.int32(seed),),
                    "klein prompts", seeds=PIPELINE_SEEDS["prompts"],
                    truth_fn=lambda seed: widened(lambda: source_encoded(wide, prompts_of(seed))))
    with pytest.raises(ValueError):
        encoder.tokenize(["x" * 513])


def test_klein_pipeline_walk_matches_the_source(klein, source, record):
    """`Pretrained.load().text_to_image()` reproduces the source's own call:
    its defaults (50 steps, two branches guided at 4.0 against the empty
    prompt), and at the recorded step count the sigmas it lays out shifted
    by its empirical mu, the latent it ends on and the image it decodes, on
    par with the source's own float32 call over drawn starting latents (check 1)."""
    import torch

    pipeline = record["pipeline"]
    task = klein.text_to_image()
    assert task.steps == 50 and task.guidance is not None and task.guidance.scale == 4.0
    narrow, wide = klein_pipelines(source)
    height, width, steps = pipeline["height"], pipeline["width"], pipeline["steps"]
    shape = (len(pipeline["prompts"]), pipeline["config"]["in_channels"], height // 4, width // 4)
    autoencoder = klein.autoencoder

    def draw(seed):
        return (torch.randn(shape, generator=torch.Generator().manual_seed(59 + seed)).numpy(),)

    def native(x_T):
        prepared = task.prepare(pipeline["prompts"], initial=x_T.transpose(0, 2, 3, 1), key=0, steps=steps)
        walked = task(prepared, key=jax.random.PRNGKey(0)).host()
        raw = pixel_shuffle(np.asarray(walked.latents) / autoencoder.latent_scale + autoencoder.latent_shift)
        return {"latents": np.asarray(raw), "images": np.clip(np.asarray(walked.images) / 2 + 0.5, 0.0, 1.0)}

    def source_walk(pipe, x_T):
        latents, images = [], []
        with torch.no_grad():
            for row, prompt in enumerate(pipeline["prompts"]):
                call = {"prompt": prompt, "height": height, "width": width, "num_inference_steps": steps}
                start = torch.from_numpy(np.asarray(x_T[row:row + 1]))
                walked = pipe(**call, latents=start.clone(), output_type="latent").images.numpy()
                latents.append(nhwc(walked)[0])
                images.append(pipe(**call, latents=start.clone(), output_type="np").images[0])
        return {"latents": np.stack(latents), "images": np.stack(images)}

    assert_as_exact(native, lambda x_T: source_walk(narrow, x_T), draw, "klein walk",
                    seeds=PIPELINE_SEEDS["walk"],
                    truth_fn=lambda x_T: widened(lambda: source_walk(wide, x_T)))


def test_a_mistral3_encoder_reads_as_flux2_dev_reads_it(source, record):
    """FLUX.2 [dev]'s conditioning over a tiny Mistral-3 encoder with Mistral
    Small 3.1's chat template: the system and user turns, 512 tokens padded
    on the right, the hidden states after layers 10, 20 and 30 stacked, and
    the guidance [dev] embeds. The vision tower and projector are carried as
    stored, not held as variables: the variables are the leaves the language
    model's layouts bind, and those layouts with the carried tensors export
    every stored tensor as it came."""
    import copy

    import torch
    from diffusers import Flux2Pipeline
    from flax.traverse_util import flatten_dict
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    from dew.interop.pipeline_assembly import _hidden_states_conditioning
    from tools.diffusers_wan_reference import float64

    directory = source / "mistral3"
    encoder, layouts, retained, _ = _hidden_states_conditioning(
        directory, {}, jnp.float32, pipeline="flux2", tokens=512, guidance=4.0,
        param_dtype="float32", attention_impl="xla")
    assert (encoder.template, encoder.layers) == ("mistral3", (10, 20, 30))
    condition = encoder.encode(encoder.params, encoder.tokenize(record["pipeline"]["prompts"]))
    np.testing.assert_array_equal(np.asarray(condition.guidance), [4.0, 4.0])
    model = AutoModelForImageTextToText.from_pretrained(directory / "text_encoder", torch_dtype=torch.float32,
                                                        local_files_only=True).eval()
    with float64():
        wide = copy.deepcopy(model).double()
    tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer")

    def prompts_of(seed) -> tuple[str, ...]:
        return drawn_prompts(13, seed)

    def source_encoded(model, prompts, dtype):
        with torch.no_grad():
            return Flux2Pipeline._get_mistral_3_small_prompt_embeds(
                model, tokenizer, list(prompts), dtype=dtype, device=torch.device("cpu")).numpy()

    # The arms draw their prompts from the seed, a leaf the gate widens as it is.
    assert_as_exact(
        lambda seed: encoder.encode(encoder.params, encoder.tokenize(list(prompts_of(seed)))).context,
        lambda seed: source_encoded(model, prompts_of(seed), torch.float32), lambda seed: (np.int32(seed),),
        "mistral3 context", seeds=PIPELINE_SEEDS["mistral3"],
        truth_fn=lambda seed: widened(lambda: source_encoded(wide, prompts_of(seed), torch.float64)))
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
