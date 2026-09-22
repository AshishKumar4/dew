"""Native Qwen-Image 2.1 against the actual Diffusers objects it reconstructs.

`tools/diffusers_qwen_image_reference.py` builds tiny
`QwenImage21Transformer2DModel` instances at Diffusers 6256aa76, saves each
with its real config and safetensors, and runs each the way
`QwenImage21Pipeline` runs it for text-to-image, recording the prediction at
the image's tokens and the gradients of the latent, the prompt states and
every parameter against a fixed cotangent. A row padded behind a longer
prompt is recorded as the source's call for its own prompt alone. It also
saves one tiny pipeline - Qwen3-VL text encoder, 2.1 VAE, transformer,
published scheduler config, processor - and walks it once per prompt.

The pipeline packs its latent as a plain row-major flatten of the grid, so
these tests reshape between that and NHWC with numpy's own reshape.
"""

import json
import tarfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.diffusion.process import DenoisingCondition
from dew.interop.diffusion import (component_tensors, qwen_image_fields,
                                   translate_qwen_image_weights)
from dew.nn.backbones.qwen_image import QwenImageTransformer, image_grid

ROOT = Path(__file__).resolve().parents[1]
CASES = ("square", "rect", "odd", "padded", "acausal")


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    directory = tmp_path_factory.mktemp("qwen-image-source")
    with tarfile.open(ROOT / "tests/fixtures/qwen_image_source.tar.xz") as archive:
        archive.extractall(directory, filter="data")
    return directory


@pytest.fixture(scope="module")
def record(source):
    return json.loads((source / "qwen_image.json").read_text())


@pytest.fixture(scope="module")
def arrays(source):
    with np.load(source / "qwen_image.npz") as loaded:
        return dict(loaded)


def relative_gap(actual, expected) -> float:
    actual, expected = np.asarray(actual, np.float32), np.asarray(expected, np.float32)
    return float(np.abs(actual - expected).max() / max(1.0, float(np.abs(expected).max())))


def nhwc(packed: np.ndarray, rows: int, columns: int) -> np.ndarray:
    """`QwenImage21Pipeline._pack_latents` inverted: one token per position, row-major."""
    return packed.reshape(packed.shape[0], rows, columns, packed.shape[-1])


def transformer_case(source, arrays, name: str):
    directory = source / name
    config = json.loads((directory / "transformer" / "config.json").read_text())
    model = QwenImageTransformer(**qwen_image_fields(config, attention_impl="xla"))
    params, layouts = translate_qwen_image_weights(component_tensors(directory, "transformer"))
    return model, params, layouts


def qwen_walk(source, arrays, record, name: str):
    """The native forward and every gradient of one case."""
    model, params, layouts = transformer_case(source, arrays, name)
    rows, columns = record["cases"][name]["grid"]
    latent = jnp.asarray(nhwc(arrays[f"{name}.packed"], rows, columns))
    context = jnp.asarray(arrays[f"{name}.context"])
    mask = jnp.asarray(arrays[f"{name}.mask"])
    times = jnp.asarray(arrays[f"{name}.times"])
    probe = jnp.asarray(nhwc(arrays[f"{name}.probe"], rows, columns))

    def forward(params, latent, context):
        return model.apply({"params": params}, latent, times, DenoisingCondition(context, mask=mask))

    output, pullback = jax.vjp(forward, params, latent, context)
    return output, pullback(probe), {entry.name: entry for entry in layouts}


@pytest.mark.parametrize("name", CASES)
def test_native_qwen_image_matches_the_source_forward_and_every_gradient(
        name, source, arrays, record):
    """The source at the suite's fixed bounds: 1e-5 scaled error on the
    prediction, 1e-4 on every gradient, every parameter included."""
    rows, columns = record["cases"][name]["grid"]
    output, (grad_params, grad_latent, grad_context), layouts = qwen_walk(
        source, arrays, record, name)
    assert relative_gap(output, nhwc(arrays[f"{name}.output"], rows, columns)) < 1e-5
    assert relative_gap(grad_latent, nhwc(arrays[f"{name}.grad_packed"], rows, columns)) < 1e-4
    assert relative_gap(grad_context, arrays[f"{name}.grad_context"]) < 1e-4
    gaps = {}
    prefix = f"{name}.grad_param."
    for key in (key for key in arrays if key.startswith(prefix)):
        tensor = key.removeprefix(prefix)
        gaps[tensor] = relative_gap(layouts[f"transformer/{tensor}"].export({"params": grad_params}),
                                    arrays[key])
    assert len(gaps) == len(layouts) == 27
    worst = max(gaps.items(), key=lambda item: item[1])
    assert worst[1] < 1e-4, worst


def test_a_padded_row_is_its_own_prompt_alone(source, arrays, record):
    """Padding a row behind a longer prompt changes nothing it reads: the
    padded keys are excluded and its image sits after its own text. The
    padding values are masked, so filling them with anything leaves the
    prediction where it was."""
    model, params, _ = transformer_case(source, arrays, "padded")
    rows, columns = record["cases"]["padded"]["grid"]
    latent = jnp.asarray(nhwc(arrays["padded.packed"], rows, columns))
    context, mask = arrays["padded.context"], arrays["padded.mask"]
    times = jnp.asarray(arrays["padded.times"])
    filled = np.where(mask[..., None], context, 7.0)
    run = lambda values: model.apply({"params": params}, latent, times,  # noqa: E731
                                     DenoisingCondition(jnp.asarray(values), mask=jnp.asarray(mask)))
    np.testing.assert_allclose(run(filled), run(context), atol=1e-6)
    # Without the mask the short row reads its padding as prompt.
    unmasked = model.apply({"params": params}, latent, times, DenoisingCondition(jnp.asarray(context)))
    assert relative_gap(unmasked[1], nhwc(arrays["padded.output"], rows, columns)[1]) > 1e-3


def test_every_declared_qwen_image_tensor_is_mapped(source):
    """Every stored tensor lands in the tree and writes back bit for bit; a
    name the class does not declare, a bias included, is refused."""
    tensors = component_tensors(source / "square", "transformer")
    params, layouts = translate_qwen_image_weights(tensors)
    assert {entry.name for entry in layouts} == {f"transformer/{name}" for name in tensors}
    for entry in layouts:
        np.testing.assert_array_equal(entry.export({"params": params}),
                                      tensors[entry.name.removeprefix("transformer/")])
    weight = next(iter(tensors.values()))
    for foreign in ("img_in.bias", "transformer_blocks.0.attn.add_q_proj.weight",
                    "transformer_blocks.0.norm1.linear.weight", "pos_embed.freqs"):
        with pytest.raises(ValueError, match="unknown tensor name"):
            translate_qwen_image_weights({foreign: weight})


def test_unsupported_qwen_image_geometry_is_refused(source):
    config = json.loads((source / "square" / "transformer" / "config.json").read_text())
    with pytest.raises(ValueError, match="patch_size must be 1"):
        qwen_image_fields({**config, "patch_size": 2})
    with pytest.raises(ValueError, match="must cover"):
        qwen_image_fields({**config, "axes_dims_rope": [4, 4, 6]})


def test_the_image_grid_is_centred_as_the_source_lays_it_out():
    """`QwenImage21Rope`'s image indices: an even side runs -n/2..n/2-1 and
    an odd one is one short on the positive side."""
    heights, widths = image_grid(3, 4)
    np.testing.assert_array_equal(heights, [-2] * 4 + [-1] * 4 + [0] * 4)
    np.testing.assert_array_equal(widths, [-2, -1, 0, 1] * 3)
