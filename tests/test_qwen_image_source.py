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

import functools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from interop_support import extract_fixture, fixture_arrays
from reference_error import assert_as_exact_as_the_reference, assert_leaves_as_exact, float64_terms, widened

from dew.diffusion.process import DenoisingCondition
from dew.interop.diffusion import component_tensors, qwen_image_fields, qwen_image_path, read_transformer
from dew.nn.backbones.qwen_image import QwenImageTransformer

ROOT = Path(__file__).resolve().parents[1]
CASES = ("square", "rect", "odd", "padded", "acausal", "eps")


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return extract_fixture(ROOT / "tests/fixtures/qwen_image_source.tar.xz",
                           tmp_path_factory.mktemp("qwen-image-source"))


@pytest.fixture(scope="module")
def record(source):
    return json.loads((source / "qwen_image.json").read_text())


@pytest.fixture(scope="module")
def arrays(source):
    return fixture_arrays(source / "qwen_image.npz")


def nhwc(packed: np.ndarray, rows: int, columns: int) -> np.ndarray:
    """`QwenImage21Pipeline._pack_latents` inverted: one token per position, row-major."""
    return packed.reshape(packed.shape[0], rows, columns, packed.shape[-1])


def transformer_case(source, arrays, name: str):
    directory = source / name
    config = json.loads((directory / "transformer" / "config.json").read_text())
    model = QwenImageTransformer(**qwen_image_fields(config, attention_impl="xla"))
    params, layouts = read_transformer(component_tensors(directory, "transformer"), qwen_image_path)
    return model, params, layouts


def qwen_walk(source, arrays, record, name: str):
    """The native forward and every gradient of one case, and the terms each
    entry's final sums add, from Dew's own path in float64 (`float64_terms`)."""
    model, params, layouts = transformer_case(source, arrays, name)
    rows, columns = record["cases"][name]["grid"]
    latent = jnp.asarray(nhwc(arrays[f"{name}.packed"], rows, columns))
    context = jnp.asarray(arrays[f"{name}.context"])
    mask = jnp.asarray(arrays[f"{name}.mask"])
    times = jnp.asarray(arrays[f"{name}.times"])
    probe = jnp.asarray(nhwc(arrays[f"{name}.probe"], rows, columns))

    def walk(model, params, latent, context):
        def forward(params, latent, context):
            return model.apply({"params": params}, latent, times, DenoisingCondition(context, mask=mask))

        output, pullback = jax.vjp(forward, params, latent, context)
        return output, pullback(probe.astype(output.dtype))

    output, gradients = walk(model, params, latent, context)
    with jax.enable_x64(new_val=True), jax.default_device(jax.devices("cpu")[0]):
        _, terms = float64_terms(functools.partial(walk, model.clone(dtype=jnp.float64)),
                                 *widened((params, latent, context)))
    return output, gradients, terms, {entry.name: entry for entry in layouts}


@pytest.mark.parametrize("name", CASES)
def test_native_qwen_image_matches_the_source_forward_and_every_gradient(
        name, source, arrays, record):
    """The prediction, the latent's and the prompt's gradients and every
    parameter's, each as near float64 as the source's float32, an entry
    whose sum cancels (the queries' and keys' path through the softmax) held
    to its terms' rounding."""
    rows, columns = record["cases"][name]["grid"]
    output, (grad_params, grad_latent, grad_context), terms, layouts = qwen_walk(
        source, arrays, record, name)
    output_terms, (param_terms, latent_terms, context_terms) = terms
    for key, value, size, layout in (("output", output, output_terms, True),
                                     ("grad_packed", grad_latent, latent_terms, True),
                                     ("grad_context", grad_context, context_terms, False)):
        expected, truth = (arrays[f"{name}.{kind}{key}"] for kind in ("", "fp64."))
        if layout:
            expected, truth = nhwc(expected, rows, columns), nhwc(truth, rows, columns)
        assert_as_exact_as_the_reference(value, expected, truth, f"{name} {key}", terms=size)
    prefix = f"{name}.grad_param."
    tensors = [key.removeprefix(prefix) for key in arrays if key.startswith(prefix)]
    assert len(tensors) == len(layouts) == 27

    def exported(tree):
        return {tensor: np.asarray(layouts[f"transformer/{tensor}"].export({"params": tree}), np.float64)
                for tensor in tensors}

    assert_leaves_as_exact(exported(grad_params), {tensor: arrays[prefix + tensor] for tensor in tensors},
                           {tensor: arrays[f"{name}.fp64.grad_param.{tensor}"] for tensor in tensors},
                           terms=exported(param_terms))


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
    def run(values):
        return model.apply({"params": params}, latent, times,
                                         DenoisingCondition(jnp.asarray(values), mask=jnp.asarray(mask)))
    np.testing.assert_allclose(run(filled), run(context), atol=1e-6)
    # Without the mask the short row reads its padding as prompt.
    unmasked = model.apply({"params": params}, latent, times, DenoisingCondition(jnp.asarray(context)))
    with pytest.raises(AssertionError):
        assert_as_exact_as_the_reference(unmasked[1], nhwc(arrays["padded.output"], rows, columns)[1],
                                         nhwc(arrays["padded.fp64.output"], rows, columns)[1], "unmasked")


@pytest.mark.gpu(reason="needs a cuda device")
def test_cudnn_attends_a_padded_batch_as_xla_does(without_deterministic_ops):
    """The fused kernel runs both of the block's attention calls over a
    padded prompt, forward and backward, and computes what xla computes.

    bf16 and a head width of 64 are what cuDNN takes; the fixture cases'
    width of 12 sends 'auto' to xla. The text is odd-length, so both calls
    also take the kernel's own odd-length padding. cuDNN once refused the
    key-padding mask this model passed (a [B, 1, 1, K] bool), which only a
    run through the kernel shows. The bound is two bf16 ulps (2**-7 each)
    of each array's scale: the two kernels round the same products in a
    different order, and the block carries that into the output and the
    gradients. One block, because the cuda lane's deterministic ops crash
    an executable holding two identical cuDNN backward calls
    (openxla/xla#46500); two blocks outside that lane: 1.1 ulps on an
    RTX 4080."""
    model = QwenImageTransformer(in_channels=4, out_channels=4, num_layers=1, heads=2, head_dim=64,
                                 context_in_dim=16, axes_dims_rope=(16, 24, 24), dtype=jnp.bfloat16,
                                 attention_impl="cudnn")
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    latent = jax.random.normal(keys[0], (2, 4, 6, 4))
    context = jax.random.normal(keys[1], (2, 25, 16))
    condition = DenoisingCondition(context, mask=jnp.arange(25)[None] < jnp.asarray([[25], [13]]))
    times = jnp.asarray([731.0, 42.0])
    params = model.init(keys[2], latent, times, condition)["params"]
    probe = jax.random.normal(keys[3], (2, 4, 6, 4))

    def pullback(implementation):
        def forward(params, latent, context):
            return model.clone(attention_impl=implementation).apply(
                {"params": params}, latent, times, DenoisingCondition(context, mask=condition.mask))
        output, vjp = jax.vjp(forward, params, latent, context)
        return [output, *jax.tree.leaves(vjp(probe.astype(output.dtype)))]

    fused, reference = jax.jit(lambda: pullback("cudnn"))(), jax.jit(lambda: pullback("xla"))()
    for got, want in zip(fused, reference, strict=True):
        got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
        assert np.abs(got - want).max() <= 2 ** -6 * np.abs(want).max()


def test_every_declared_qwen_image_tensor_is_mapped(source):
    """Every stored tensor lands in the tree and writes back bit for bit; a
    name the class does not declare, a bias included, is refused."""
    tensors = component_tensors(source / "square", "transformer")
    params, layouts = read_transformer(tensors, qwen_image_path)
    assert {entry.name for entry in layouts} == {f"transformer/{name}" for name in tensors}
    for entry in layouts:
        np.testing.assert_array_equal(entry.export({"params": params}),
                                      tensors[entry.name.removeprefix("transformer/")])
    weight = next(iter(tensors.values()))
    for foreign in ("img_in.bias", "transformer_blocks.0.attn.add_q_proj.weight",
                    "transformer_blocks.0.norm1.linear.weight", "pos_embed.freqs"):
        with pytest.raises(ValueError):
            read_transformer({foreign: weight}, qwen_image_path)


@pytest.fixture(scope="module")
def loaded(source):
    from dew.interop.pretrained import Pretrained

    return Pretrained.load(str(source / "pipeline"), dtype="float32", attention_impl="xla")


def test_published_qwen_image_prompt_encoding_matches_the_source_pipeline(loaded, arrays, record):
    """The conditioner composes what `encode_prompt` composes: the template,
    the last layer before the final norm, the system turn dropped. Each row
    is padded on the right to the budget, its real tokens are the source's
    states for that prompt alone, and its padding is masked and zero."""
    encoder = loaded.inputs.conditions["conditioning"].encoder
    params = loaded.variables["encoders"]["conditioning"]
    prompts = record["pipeline"]["prompts"]
    condition = encoder.encode(params, encoder.tokenize(prompts))
    assert condition.context.shape == (2, encoder.tokens, record["pipeline"]["config"]["context_in_dim"])
    for row in range(len(prompts)):
        expected = arrays[f"pipeline.context.{row}"][0]
        length = expected.shape[0]
        np.testing.assert_array_equal(np.asarray(condition.mask[row]),
                                      np.arange(encoder.tokens) < length)
        assert_as_exact_as_the_reference(condition.context[row, :length], expected,
                                         arrays[f"pipeline.fp64.context.{row}"][0], f"prompt {row}")
        assert not np.asarray(condition.context[row, length:]).any()
    assert encoder.captions(encoder.tokenize(prompts)) == tuple(prompts)
    # The pipeline encodes an empty prompt as one space.
    np.testing.assert_array_equal(encoder.tokenize([""])["input_ids"],
                                  encoder.tokenize([" "])["input_ids"])
    with pytest.raises(ValueError):
        encoder.tokenize(["x" * (encoder.tokens + 1)])


def test_the_prompt_states_are_the_decoders_own_last_layer_output():
    """The conditioner embeds through the decoder's own path, so a decoder
    that scales its embeddings, by Gemma's sqrt(d) or a muP multiplier,
    gives the conditioner the last layer's output of its own forward."""
    from dew.inputs.diffusion import _residual_states
    from dew.nn.backbones.causal_transformer import (
        INTERMEDIATES,
        CausalTransformer,
        layer_output,
        layer_outputs,
    )

    model = CausalTransformer(vocab_size=37, emb_features=32, num_layers=2, num_heads=4, mlp_features=64,
                              max_seq_len=16, embedding_scale=True, embedding_multiplier=12.0)
    ids = jax.random.randint(jax.random.key(0), (2, 12), 0, 37)
    params = model.init(jax.random.key(1), ids)
    _, state = model.apply(params, ids, capture_intermediates=layer_outputs, mutable=[INTERMEDIATES])
    np.testing.assert_array_equal(model.apply(params, ids, method=_residual_states),
                                  layer_output(state[INTERMEDIATES], 1))


def test_published_qwen_image_pipeline_walk_matches_the_source(loaded, arrays, record):
    """`Pretrained.load().text_to_image()` reproduces the source's own call:
    its 40 default steps, no guidance, the sigmas its call lays out shifted
    by the mu of this latent's token count, and the RGBA decode. Both rows
    walk in one batch, the shorter prompt padded, and each lands on the
    source's call for its prompt alone. Each prompt also walks alone, the
    call one image makes, which XLA compiles differently (see the one-row
    grid walk in test_samplers.py)."""
    pipeline = record["pipeline"]
    task = loaded.text_to_image()
    assert task.steps == pipeline["default_steps"] == 40
    assert task.guidance is None and pipeline["true_cfg"] == 1.0
    rows, columns = pipeline["height"] // 16, pipeline["width"] // 16
    initial = nhwc(arrays["pipeline.x_T"], rows, columns)
    walked = task(task.prepare(pipeline["prompts"], initial=initial, key=0),
                  key=jax.random.PRNGKey(0)).host()
    images = np.clip(np.asarray(walked.images) / 2 + 0.5, 0.0, 1.0)
    for row, prompt in enumerate(pipeline["prompts"]):
        expected, truth = (nhwc(arrays[f"pipeline.{kind}latents.{row}"], rows, columns)[0]
                           for kind in ("", "fp64."))
        assert_as_exact_as_the_reference(np.asarray(walked.latents)[row], expected, truth, f"latents {row}")
        assert_as_exact_as_the_reference(images[row], arrays[f"pipeline.images.{row}"][0],
                                         arrays[f"pipeline.fp64.images.{row}"][0], f"images {row}")
        alone = task(task.prepare([prompt], initial=initial[row:row + 1], key=0),
                     key=jax.random.PRNGKey(0)).host()
        assert_as_exact_as_the_reference(np.asarray(alone.latents)[0], expected, truth,
                                         f"latents {row} alone")


def test_a_trained_qwen_image_step_exports_and_reloads(source, loaded, arrays, record, tmp_path):
    """A real flow-matching step over the published source, then a resume and
    an export.

    The objective runs the whole source: the VAE encodes the RGBA pixels,
    the Qwen3-VL language model encodes the prompts, and the transformer
    takes the gradient. The encoder and the autoencoder are state, so every
    leaf survives the step. The encoder's state is the leaves its language
    model's layouts bind; the vision tower, which the prompt never reads, is
    carried as stored beside it, and both go back out byte for byte. The
    export reloads to the trained forward.
    """
    from diffusion_support import image_batch, trained_step
    from flax.traverse_util import flatten_dict

    from dew.checkpoints import Checkpoints
    from dew.inputs.diffusion import QwenImageConditioner
    from dew.interop.diffusion import component_tensors
    from dew.interop.pretrained import Pretrained
    from dew.objectives.diffusion import DiffusionObjective

    assert loaded.inputs.sample.shape[-1] == 4
    objective = DiffusionObjective(loaded.model, loaded.process, loaded.inputs,
                                   autoencoder=loaded.autoencoder, variables=loaded.variables,
                                   unconditional_prob=0.0, ema_decay=None, steps=2)
    prompts = record["pipeline"]["prompts"]
    batch = image_batch(loaded, prompts)
    _, state = trained_step(objective, batch, Checkpoints(str(tmp_path / "run")))
    assert set(flatten_dict(state.variables["encoders"]["conditioning"])) == {
        layout.paths[0][2:] for layout in loaded.weight_layouts if layout.name.startswith("text_encoder/")}
    visual = {f"text_encoder/{name}" for name in component_tensors(source / "pipeline", "text_encoder")
              if name.startswith("model.visual.")}
    assert visual and set(loaded.retained_tensors) == visual

    export = tmp_path / "export"
    loaded.save(export, variables=state.variables)
    for component in ("text_encoder", "vae"):
        published, written = (component_tensors(source / "pipeline", component),
                              component_tensors(export, component))
        assert written.keys() == published.keys()
        for name, tensor in published.items():
            np.testing.assert_array_equal(written[name], tensor)
    again = Pretrained.load(str(export), dtype="float32", attention_impl="xla")
    rebuilt = QwenImageConditioner.from_pretrained(
        str(export), dtype="float32", **{key: value for key, value in
                                          again.inputs.conditions["conditioning"].encoder.to_json().items()
                                          if key not in ("checkpoint", "dtype")},
        params=again.variables["encoders"]["conditioning"])
    tokens = rebuilt.tokenize(prompts)
    condition = rebuilt.encode(again.variables["encoders"]["conditioning"], tokens)
    grid = (record["pipeline"]["height"] // 16, record["pipeline"]["width"] // 16)
    latent = jnp.asarray(nhwc(arrays["pipeline.x_T"], *grid))
    times = jnp.asarray([500.0, 100.0])
    trained = loaded.model.apply({"params": state.variables["params"]}, latent, times, condition)
    reloaded = again.model.apply({"params": again.variables["params"]}, latent, times, condition)
    np.testing.assert_array_equal(reloaded, trained)


def test_diffusers_reads_a_qwen_image_export(tmp_path):
    """Qwen-Image 2.1's classes are Diffusers 6256aa76's, which this
    environment does not install, so their reading of an export is recorded
    (tools/qwen_image_export_reference.py): the committed pipeline with its
    transformer perturbed by a seeded draw, saved by Dew, then loaded whole by
    `QwenImage21Pipeline` with every component's loading report empty, and its
    transformer run on fixed inputs in float32 and float64. The same export
    written here is the one that was read, file for file by SHA-256, and Dew's
    forward on the perturbed weights holds the float64 rule against it."""
    from reference_error import assert_as_exact_as_the_reference

    from tools.qwen_image_export_reference import GRID, digests, perturbed_export

    loaded, variables = perturbed_export(tmp_path / "export", tmp_path / "scratch")
    with np.load(ROOT / "tests/fixtures/qwen_image_export.npz") as stored:
        recorded = dict(stored)
    meta = json.loads(recorded.pop("meta").tobytes())
    assert digests(tmp_path / "export") == meta["digests"], (
        "Dew's export changed: rerun both halves of tools/qwen_image_export_reference.py, "
        "`consume` in the qwen-image reference environment, so Diffusers reads the new files")
    latent = jnp.asarray(nhwc(recorded["packed"], *GRID))
    context = jnp.asarray(recorded["context"])
    condition = DenoisingCondition(context, mask=jnp.ones(context.shape[:2], bool))
    times = jnp.asarray(recorded["times"])
    ours = loaded.model.apply({"params": variables["params"]}, latent, times, condition)
    assert_as_exact_as_the_reference(np.asarray(ours), nhwc(recorded["fp32.output"], *GRID),
                                     nhwc(recorded["fp64.output"], *GRID), "exported Qwen-Image 2.1")
