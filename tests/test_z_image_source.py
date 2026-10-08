"""Native Z-Image against the actual Diffusers 0.40.0 objects it reconstructs.

`tools/diffusers_z_image_reference.py` builds tiny `ZImageTransformer2DModel`
instances with every parameter moved off its initialization, saves each with
its real config and safetensors, and calls each as `ZImagePipeline` calls it
- a list of latents, the time 1 - sigma, a list of prompt states of each
prompt's own length - recording the output and the gradients of the latents
and the prompt states, and for one case of every parameter, against a fixed
cotangent. It also saves one tiny `ZImagePipeline` over the published
configs and walks the unmodified call, guided at its default scale.

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
from interop_support import extract_fixture, fixture_arrays

from dew.diffusion.process import DenoisingCondition
from dew.interop.diffusion import component_tensors, translate_z_image_weights, z_image_fields
from dew.nn.backbones.z_image import ZImageTransformer

ROOT = Path(__file__).resolve().parents[1]
CASES = ("padded", "exact", "deep")
BUDGET = 64
FORWARD = 1e-5
GRADIENT = 1e-4


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


def relative_gap(actual, expected) -> float:
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    return float(np.abs(actual - expected).max() / max(1.0, float(np.abs(expected).max())))


def nhwc(latents):
    """`[B, C, 1, H, W]` to `[B, H, W, C]`."""
    return np.asarray(latents)[:, :, 0].transpose(0, 2, 3, 1)


def load(source, name):
    config = json.loads((source / name / "transformer" / "config.json").read_text())
    model = ZImageTransformer(**z_image_fields(config, dtype="float32", attention_impl="xla"))
    params, layouts = translate_z_image_weights(component_tensors(source / name, "transformer"))
    return model, params, layouts


def inputs(name, arrays, record):
    """The native call's inputs: the latents channels last, the model time
    from the source's 1 - sigma, and each prompt padded to the budget."""
    lengths = record["cases"][name]["lengths"]
    width = arrays[f"{name}.caption.0"].shape[-1]
    context = np.zeros((len(lengths), BUDGET, width), np.float32)
    for row, length in enumerate(lengths):
        context[row, :length] = arrays[f"{name}.caption.{row}"]
    mask = np.arange(BUDGET)[None] < np.asarray(lengths)[:, None]
    time = 1000.0 * (1.0 - arrays[f"{name}.times"])
    return jnp.asarray(nhwc(arrays[f"{name}.latents"])), jnp.asarray(time, jnp.float32), context, mask


@pytest.mark.parametrize("name", CASES)
def test_every_published_tensor_maps_and_exports_bit_identical(source, name):
    _, params, layouts = load(source, name)
    tensors = component_tensors(source / name, "transformer")
    assert {layout.name for layout in layouts} == {f"transformer/{key}" for key in tensors}
    for layout in layouts:
        written = layout.export({"params": params})
        assert written.tobytes() == tensors[layout.name.removeprefix("transformer/")].tobytes(), layout.name


@pytest.mark.parametrize("name", CASES)
def test_the_flow_and_its_gradients_match_the_source(source, arrays, record, name):
    """The native flow is the negated source output, its gradients the
    negated source gradients, whatever the rows' prompt lengths."""
    model, params, layouts = load(source, name)
    latents, time, context, mask = inputs(name, arrays, record)
    probe = nhwc(arrays[f"{name}.probe"])

    def objective(params, latents, context):
        return jnp.sum(
            model.apply({"params": params}, latents, time, DenoisingCondition(context, mask=mask)) * probe
        )

    flow = model.apply({"params": params}, latents, time, DenoisingCondition(jnp.asarray(context), mask=mask))
    gaps = {"forward": relative_gap(-flow, nhwc(arrays[f"{name}.output"]))}
    grad_params, grad_latents, grad_context = jax.grad(objective, argnums=(0, 1, 2))(params, latents,
                                                                                     jnp.asarray(context))
    gaps["latents"] = relative_gap(-grad_latents, nhwc(arrays[f"{name}.grad_latents"]))
    for row, length in enumerate(record["cases"][name]["lengths"]):
        gaps[f"prompt {row}"] = relative_gap(
            -grad_context[row, :length], arrays[f"{name}.grad_caption.{row}"]
        )
        assert not np.asarray(grad_context[row, length:]).any()
    for layout in layouts:
        key = f"{name}.grad_param.{layout.name.removeprefix('transformer/')}"
        if key in arrays:
            gaps[layout.name] = relative_gap(-layout.export({"params": grad_params}), arrays[key])
    worst = max(gaps, key=gaps.get)
    print(f"{name}: forward {gaps['forward']:.3g}; worst {worst} {gaps[worst]:.3g}")
    assert gaps["forward"] < FORWARD
    assert gaps[worst] < GRADIENT, worst


@pytest.mark.parametrize("change", [
    {"all_patch_size": [1]}, {"n_kv_heads": 1}, {"qk_norm": False}, {"siglip_feat_dim": 16},
    {"axes_dims": [8, 8, 8]},
])
def test_an_unsupported_config_is_refused(source, change):
    config = json.loads((source / "padded" / "transformer" / "config.json").read_text())
    z_image_fields(config)
    with pytest.raises(ValueError):
        z_image_fields({**config, **change})


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
        assert relative_gap(condition.context[row, :length], expected) < FORWARD, row


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
        expected = arrays[f"pipeline.latents.{row}"].transpose(0, 2, 3, 1)[0]
        assert relative_gap(latents[row], expected) < FORWARD, row
        assert relative_gap(images[row], arrays[f"pipeline.images.{row}"][0]) < FORWARD, row
