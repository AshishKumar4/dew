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
from interop_support import assert_layout_tensors, extract_fixture, fixture_arrays
from reference_error import (
    assert_as_exact,
    assert_as_exact_as_the_reference,
    declared_seeds,
    indexed_recording,
)

from dew.diffusion.process import DenoisingCondition
from dew.interop.diffusion import component_tensors, read_transformer, z_image_fields, z_image_path
from dew.nn.backbones.z_image import ZImageTransformer

ROOT = Path(__file__).resolve().parents[1]
CASES = ("padded", "exact", "deep")
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


def load(source, name, *, dtype="float32"):
    config = json.loads((source / name / "transformer" / "config.json").read_text())
    model = ZImageTransformer(**z_image_fields(config, dtype=dtype, attention_impl="xla"))
    params, layouts = read_transformer(component_tensors(source / name, "transformer"), z_image_path)
    return model, params, layouts


def transformer_walk(source, name, lengths, *, dtype="float32"):
    """The native flow and gradients, plus the padded prompt gradient.

    The recorder plans seeds on the same walk, inferring dtype for its widened proxy.
    """
    model, params, layouts = load(source, name, dtype=dtype)
    mask = np.arange(BUDGET)[None] < np.asarray(lengths)[:, None]

    @jax.jit
    def walk(inputs):
        latents = inputs["latents"][:, :, 0].transpose(0, 2, 3, 1)
        probe = inputs["probe"][:, :, 0].transpose(0, 2, 3, 1)
        context = jnp.zeros((len(lengths), BUDGET, inputs["caption.0"].shape[-1]), latents.dtype)
        for row, length in enumerate(lengths):
            context = context.at[row, :length].set(inputs[f"caption.{row}"])
        time = 1000.0 * (1.0 - inputs["times"])

        def objective(params, latents, context):
            output = model.apply({"params": params}, latents, time, DenoisingCondition(context, mask=mask))
            return jnp.sum(output * probe), output

        (_, output), gradients = jax.value_and_grad(objective, argnums=(0, 1, 2), has_aux=True)(
            jax.tree.map(lambda value: jnp.asarray(value, latents.dtype), params), latents, context)
        return {"output": -output, "grad_latents": -gradients[1],
                **{f"grad_caption.{row}": -gradients[2][row, :length]
                   for row, length in enumerate(lengths)},
                "grad_params": gradients[0], "grad_context": gradients[2]}

    return walk, layouts


@pytest.mark.parametrize("name", CASES)
def test_every_published_tensor_maps_and_exports_bit_identical(source, name):
    _, params, layouts = load(source, name)
    tensors = component_tensors(source / name, "transformer")
    assert_layout_tensors(layouts, {"params": params}, tensors, "transformer/")


@pytest.mark.parametrize("name", CASES)
def test_the_flow_and_its_gradients_match_the_source(source, arrays, record, name):
    """The native flow is the negated source output, its gradients the
    negated source gradients, every parameter's among them, whatever the
    rows' prompt lengths: each as near float64 as the source's float32."""
    lengths = record["cases"][name]["lengths"]
    walk, layouts = transformer_walk(source, name, lengths)
    tensors = {layout.name.removeprefix("transformer/"): layout for layout in layouts}
    assert {key.removeprefix(f"{name}.grad_param.") for key in arrays
            if key.startswith(f"{name}.grad_param.")} == set(tensors)
    seeds = declared_seeds(source / "z_image.npz")[name]
    draw, reference, truth = indexed_recording(arrays, name, seeds)

    def native(index, inputs):
        values = walk(inputs)
        gradients = values.pop("grad_params")
        grad_context = values.pop("grad_context")
        for row, length in enumerate(lengths):
            assert not np.asarray(grad_context[row, length:]).any()
        return {**values, **{f"grad_param.{tensor}": -layout.export({"params": gradients})
                            for tensor, layout in tensors.items()}}

    assert_as_exact(native, reference, draw, f"Z-Image {name}", seeds=seeds, truth_fn=truth)


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
