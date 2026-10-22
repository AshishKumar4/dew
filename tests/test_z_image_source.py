"""Native Z-Image against the actual Diffusers 0.40.0 objects it reconstructs.

`tools/diffusers_z_image_reference.py` builds tiny `ZImageTransformer2DModel`
instances with every parameter moved off its initialization, saves each with
its real config and safetensors, and calls each as `ZImagePipeline` calls it
- a list of latents, the time 1 - sigma, a list of prompt states of each
prompt's own length - recording the output and the gradients of the latents,
the prompt states and every parameter against a fixed cotangent, in float32
and in float64. It also saves one tiny `ZImagePipeline` over the published
configs and walks the unmodified call, guided at its default scale, the same
two ways. Each native tensor is held to the source's own float32 distance
from float64 (tests/reference_error.py).

The native transformer takes Dew's model time, sigma times a thousand, and
returns the flow, the negated source output; its prompt states arrive padded
to a budget with a mask. These tests convert with the arithmetic written out
here.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from interop_support import assert_transformer_exports, extract_fixture, fixture_arrays, source_transformer
from reference_error import assert_as_exact, assert_as_exact_as_the_reference

from dew.diffusion.process import DenoisingCondition
from dew.interop.diffusion import z_image_fields, z_image_path
from dew.nn.backbones.z_image import ZImageTransformer

ROOT = Path(__file__).resolve().parents[1]
CASES = ("padded", "exact", "deep")
CASE_SEEDS = {"padded": 16, "exact": 16, "deep": 16}
ROWS = 4
HEIGHT = 3
BUDGET = 64


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return extract_fixture(ROOT / "tests/fixtures/z_image_source.tar.xz",
                           tmp_path_factory.mktemp("z-image-source"))


@pytest.fixture(scope="module")
def record(source):
    return json.loads((source / "z_image.json").read_text())


@pytest.fixture(scope="module")
def arrays(source):
    return fixture_arrays(source / "z_image.npz")


def nhwc(latents):
    """`[B, C, 1, H, W]` to `[B, H, W, C]`."""
    return np.asarray(latents)[:, :, 0].transpose(0, 2, 3, 1)


def load(source, name):
    return source_transformer(source, name, ZImageTransformer, z_image_fields, z_image_path)


def z_image_walk(model, params, lengths):
    """A compiled native flow and every gradient on a draw: the source's latents channels last, the
    model time from the source's 1 - sigma, and each prompt padded to the budget and masked."""
    mask = jnp.asarray(np.arange(BUDGET)[None] < np.asarray(lengths)[:, None])

    @jax.jit
    def walk(latents, times, context, probe):
        time = 1000.0 * (1.0 - times)

        def objective(params, latents, context):
            flow = model.apply({"params": params}, latents, time, DenoisingCondition(context, mask=mask))
            return jnp.sum(flow * probe), flow

        (_, flow), gradients = jax.value_and_grad(objective, argnums=(0, 1, 2), has_aux=True)(
            params, latents, context)
        return flow, gradients

    return walk


@pytest.mark.parametrize("name", CASES)
def test_every_published_tensor_maps_and_exports_bit_identical(source, name):
    _, params, layouts = load(source, name)
    assert_transformer_exports(source, name, layouts, params)


@pytest.mark.parametrize("name", CASES)
def test_the_flow_and_its_gradients_match_the_source(source, arrays, record, name):
    """The native flow is the negated source output, its gradients the negated source gradients,
    every parameter's among them, whatever the rows' prompt lengths, each as near float64 as
    Diffusers' own float32 walk (`assert_as_exact`) over the fixture's draw and fresh ones; a
    prompt's padding takes no gradient."""
    import torch
    from diffusers import ZImageTransformer2DModel

    from tools.diffusers_wan_reference import float64

    model, params, layouts = load(source, name)
    # Each draw holds the case's rows ROWS times over and its images HEIGHT times as tall, so a
    # parameter's gradient sums over more of them and one draw's error varies less: the first
    # layer's norm gradients did not resolve within 512 seeds at the fixture's size. Three times
    # the height keeps each case's padding: the padded case's 72 patches pad to 96, the exact
    # case's 96 fill three blocks of 32, and the deep case's 45 pad to 64.
    cased = len(record["cases"][name]["lengths"])
    lengths = tuple(record["cases"][name]["lengths"]) * ROWS
    walk = z_image_walk(model, params, lengths)
    tensors = {layout.name.removeprefix("transformer/"): layout for layout in layouts}
    assert {key.removeprefix(f"{name}.grad_param.") for key in arrays
            if key.startswith(f"{name}.grad_param.")} == set(tensors)
    captions = [f"caption.{row}" for row in range(len(lengths))]
    original = {key: np.concatenate([arrays[f"{name}.{key}"]] * ROWS)
                for key in ("latents", "times", "probe")}
    original |= {key: np.tile(original[key], (1, 1, 1, HEIGHT, 1)) for key in ("latents", "probe")}
    original |= {caption: np.array(arrays[f"{name}.caption.{row % cased}"])
                 for row, caption in enumerate(captions)}

    def draw(seed):
        """The fixture's inputs, its rows ROWS times over and its images tiled HEIGHT times down, at
        seed 0, else fresh normal draws of those shapes."""
        if seed == 0:
            return (original,)
        generator = torch.Generator().manual_seed(31 + seed)
        return ({**{key: torch.randn(original[key].shape, generator=generator).numpy()
                    for key in ("latents", "probe", *captions)}, "times": original["times"]},)

    checkpoint = source / name / "transformer"
    narrow_model = ZImageTransformer2DModel.from_pretrained(
        checkpoint, torch_dtype=torch.float32, local_files_only=True).eval()
    with float64():
        wide_model = ZImageTransformer2DModel.from_pretrained(
            checkpoint, torch_dtype=torch.float64, local_files_only=True).eval()

    def reference_walk(model, dtype, drawn):
        inputs = [torch.tensor(drawn[key], dtype=dtype).requires_grad_() for key in ("latents", *captions)]
        named = list(model.named_parameters())
        output = torch.stack(model(list(inputs[0].unbind(0)), torch.tensor(drawn["times"], dtype=dtype),
                                   inputs[1:], return_dict=False)[0])
        gradients = torch.autograd.grad((output * torch.tensor(drawn["probe"], dtype=dtype)).sum(),
                                        inputs + [value for _, value in named])
        return {"output": nhwc(output.detach().numpy()), "grad_latents": nhwc(gradients[0].numpy()),
                **{f"grad_{caption}": gradient.numpy()
                   for caption, gradient in zip(captions, gradients[1:1 + len(captions)], strict=True)},
                **{f"grad_param.{key}": gradient.numpy()
                   for (key, _), gradient in zip(named, gradients[1 + len(captions):], strict=True)}}

    def truth(inputs):
        with float64():
            return reference_walk(wide_model, torch.float64, inputs)

    def native(inputs):
        width = inputs[captions[0]].shape[-1]
        context = np.zeros((len(lengths), BUDGET, width), np.float32)
        for row, length in enumerate(lengths):
            context[row, :length] = inputs[f"caption.{row}"]
        flow, (grad_params, grad_latents, grad_context) = walk(
            jnp.asarray(nhwc(inputs["latents"])), jnp.asarray(inputs["times"], jnp.float32),
            jnp.asarray(context), jnp.asarray(nhwc(inputs["probe"])))
        grad_context = np.asarray(grad_context)
        assert not any(grad_context[row, length:].any() for row, length in enumerate(lengths))
        return {"output": -np.asarray(flow), "grad_latents": -np.asarray(grad_latents),
                **{f"grad_caption.{row}": -grad_context[row, :length] for row, length in enumerate(lengths)},
                **{f"grad_param.{tensor}": -np.asarray(layout.export({"params": grad_params}))
                   for tensor, layout in tensors.items()}}

    assert_as_exact(native, lambda inputs: reference_walk(narrow_model, torch.float32, inputs), draw,
                    f"Z-Image {name}", seeds=CASE_SEEDS[name], truth_fn=truth)


@pytest.fixture(scope="module")
def pipeline(source):
    from dew.interop.pretrained import Pretrained

    return Pretrained.load(str(source / "pipeline"), dtype="float32", attention_impl="xla")


@pytest.fixture(scope="module")
def streamed(source):
    """The pipeline loaded onto a mesh, its weights streamed as recipes and
    placed a leaf at a time (tests/test_pipeline_streaming.py)."""
    from dew.interop.pretrained import Pretrained
    from dew.training import MeshSpec

    return Pretrained.load(str(source / "pipeline"), dtype="float32", attention_impl="xla", mesh=MeshSpec())


def test_prompt_encoding_matches_the_source_pipeline(pipeline, arrays, record):
    """The conditioner reads what `_encode_prompt` reads: the chat template
    with thinking on, the output of the encoder's second-to-last layer, and
    the real tokens, which the mask marks."""
    encoder = pipeline.inputs.conditions["conditioning"].encoder
    params = pipeline.variables["encoders"]["conditioning"]
    prompts = [*record["pipeline"]["prompts"], ""]
    condition = encoder.encode(params, encoder.tokenize(prompts))
    for row in range(len(prompts)):
        expected = arrays[f"pipeline.context.{row}"]
        length = expected.shape[0]
        np.testing.assert_array_equal(np.asarray(condition.mask[row]), np.arange(encoder.tokens) < length)
        assert_as_exact_as_the_reference(condition.context[row, :length], expected,
                                         arrays[f"pipeline.fp64.context.{row}"], f"prompt {row}")


@pytest.mark.parametrize("load", ["pipeline", "streamed"])
def test_pipeline_walk_matches_the_source(load, arrays, record, request):
    """`Pretrained.load().text_to_image()` reproduces the source's call: its
    50 default steps guided at 5.0 its way, which is Dew's 6.0, and at the
    recorded step count the latent it ends on and the image it decodes,
    loaded whole or streamed onto a mesh."""
    pipeline = request.getfixturevalue(load)
    recorded = record["pipeline"]
    task = pipeline.text_to_image()
    assert task.steps == 50 and task.guidance is not None and task.guidance.scale == recorded["guidance"] + 1
    initial = arrays["pipeline.x_T"].transpose(0, 2, 3, 1)
    walked = task(task.prepare(recorded["prompts"], initial=initial, key=0, steps=recorded["steps"]),
                  key=jax.random.PRNGKey(0)).host()
    images = np.clip(np.asarray(walked.images) / 2 + 0.5, 0.0, 1.0)
    latents = np.asarray(walked.latents)
    for row in range(len(recorded["prompts"])):
        expected, truth = (arrays[f"pipeline.{kind}latents.{row}"].transpose(0, 2, 3, 1)[0]
                           for kind in ("", "fp64."))
        assert_as_exact_as_the_reference(latents[row], expected, truth, f"latents {row}")
        assert_as_exact_as_the_reference(images[row], arrays[f"pipeline.images.{row}"][0],
                                         arrays[f"pipeline.fp64.images.{row}"][0], f"images {row}")
