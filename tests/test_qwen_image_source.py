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
from dew.interop.diffusion import component_tensors, qwen_image_fields, translate_qwen_image_weights
from dew.nn.backbones.qwen_image import QwenImageTransformer

ROOT = Path(__file__).resolve().parents[1]
RELEASED = ROOT / "tests/fixtures/hf/qwen-image-2.1-source"
CASES = ("square", "rect", "odd", "padded", "acausal", "eps")


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


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="needs a cuda device")
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


def test_the_released_configs_and_weight_maps_translate():
    """Qwen/Qwen-Image-2.1 at 790c9263 as published: the transformer and VAE
    configs build their modules, and every tensor name the transformer and
    the Qwen3-VL encoder store, the vision tower's included, maps to a
    parameter path of its own."""
    from dew.interop import hf_decoders
    from dew.interop.diffusion import _qwen_image_path
    from dew.interop.pretrained import _qwen_text_path, _qwen_vl_text_config
    from dew.nn.autoencoders.qwen_image import QwenImageVAE, qwen_image_vae_fields

    def read(name):
        return json.loads((RELEASED / name).read_text())

    fields = qwen_image_fields(read("transformer/config.json"))
    assert (fields["num_layers"], fields["heads"], fields["head_dim"], fields["context_in_dim"],
            fields["axes_dims_rope"]) == (32, 32, 128, 4096, (16, 56, 56))
    names = read("transformer/diffusion_pytorch_model.safetensors.index.json")["weight_map"]
    assert len({_qwen_image_path(name) for name in names}) == len(names) == 297
    text = _qwen_text_path(hf_decoders.translate_config(_qwen_vl_text_config(read("text_encoder/config.json"))))
    names = read("text_encoder/model.safetensors.index.json")["weight_map"]
    paths = [text(name) for name in names]
    assert None not in paths and len(set(paths)) == len(names) == 750
    assert sum(path[0] == "visual" for path in paths) == 351
    vae = QwenImageVAE(**qwen_image_vae_fields(read("vae/config.json")))
    assert (vae.downscale_factor, vae.latent_channels, vae.image_channels) == (16, 64, 4)


@pytest.fixture(scope="module")
def loaded(source):
    from dew.interop.pretrained import load_pretrained

    return load_pretrained(str(source / "pipeline"), dtype="float32", attention_impl="xla")


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
        assert relative_gap(condition.context[row, :length], expected) < 1e-5
        assert not np.asarray(condition.context[row, length:]).any()
    assert encoder.captions(encoder.tokenize(prompts)) == tuple(prompts)
    # The pipeline encodes an empty prompt as one space.
    np.testing.assert_array_equal(encoder.tokenize([""])["input_ids"],
                                  encoder.tokenize([" "])["input_ids"])
    with pytest.raises(ValueError, match="token budget"):
        encoder.tokenize(["x" * (encoder.tokens + 1)])


def test_published_qwen_image_pipeline_walk_matches_the_source(loaded, arrays, record):
    """`load_pretrained().text_to_image()` reproduces the source's own call:
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
    walked = task(task.prepare(pipeline["prompts"], initial=initial, seed=0),
                  key=jax.random.PRNGKey(0)).host()
    images = np.clip(np.asarray(walked.images) / 2 + 0.5, 0.0, 1.0)
    for row, prompt in enumerate(pipeline["prompts"]):
        expected = nhwc(arrays[f"pipeline.latents.{row}"], rows, columns)[0]
        assert relative_gap(np.asarray(walked.latents)[row], expected) < 2e-5
        assert relative_gap(images[row], arrays[f"pipeline.images.{row}"][0]) < 2e-5
        alone = task(task.prepare([prompt], initial=initial[row:row + 1], seed=0),
                     key=jax.random.PRNGKey(0)).host()
        assert relative_gap(np.asarray(alone.latents)[0], expected) < 2e-5


def test_a_trained_qwen_image_step_exports_and_reloads(source, loaded, arrays, record, tmp_path):
    """A real flow-matching step over the published source, then a resume and
    an export.

    The objective runs the whole source: the VAE encodes the RGBA pixels,
    the Qwen3-VL language model encodes the prompts, and the transformer
    takes the gradient. The encoder and the autoencoder are state, so every
    leaf survives the step and their tensors - the vision tower the prompt
    never reads included - go back out byte for byte; the export reloads to
    the trained forward.
    """
    import optax

    from dew.checkpoints import Checkpoints
    from dew.inputs.diffusion import QwenImageConditioner
    from dew.interop.diffusion import component_tensors
    from dew.interop.pretrained import load_pretrained
    from dew.objectives import Step
    from dew.objectives.diffusion import DiffusionObjective
    from dew.training import Trainer

    height, width, channels = loaded.inputs.sample.shape
    assert channels == 4
    objective = DiffusionObjective(loaded.model, loaded.process, loaded.inputs,
                                   autoencoder=loaded.autoencoder, pretrained=loaded.variables,
                                   unconditional_prob=0.0, ema_decay=None, steps=2)
    rows = jax.device_count()
    pixels = np.tile(np.arange(height * width * channels, dtype=np.uint8).reshape(
        1, height, width, channels), (rows, 1, 1, 1))
    prompts = record["pipeline"]["prompts"]
    batch = {"image": pixels,
             **loaded.inputs.tokenize([prompts[row % len(prompts)] for row in range(rows)])}
    checkpoints = Checkpoints(str(tmp_path / "run"))
    trainer = Trainer(objective, optax.sgd(1e-2), key=jax.random.PRNGKey(3),
                      checkpoints=checkpoints)
    initial = trainer.initial_state()
    fixed = Step(jnp.asarray(0), jax.random.PRNGKey(5), None)

    def value(params) -> float:
        loss, _ = objective.loss(params, batch, fixed)
        return float(loss.total / loss.mass)

    before = value(initial.params)
    state, _, _, _, accepted = trainer.compile(initial, batch)(initial, batch)
    assert bool(accepted)
    assert value(state.params) < before
    assert not np.allclose(state.params["params"]["proj_out"]["kernel"],
                           initial.params["params"]["proj_out"]["kernel"])
    for held in ("encoders", "autoencoder"):
        for got, want in zip(jax.tree.leaves(state.params[held]),
                             jax.tree.leaves(initial.params[held]), strict=True):
            np.testing.assert_array_equal(got, want)

    checkpoints.save(1, state, None, {})
    checkpoints.wait()
    restored, _, _ = trainer.place()
    for got, want in zip(jax.tree.leaves(restored), jax.tree.leaves(state), strict=True):
        np.testing.assert_array_equal(got, want)

    export = tmp_path / "export"
    loaded.save(export, variables=state.params)
    for component in ("text_encoder", "vae"):
        published, written = (component_tensors(source / "pipeline", component),
                              component_tensors(export, component))
        assert written.keys() == published.keys()
        for name, tensor in published.items():
            np.testing.assert_array_equal(written[name], tensor)
    again = load_pretrained(str(export), dtype="float32", attention_impl="xla")
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
    trained = loaded.model.apply({"params": state.params["params"]}, latent, times, condition)
    reloaded = again.model.apply({"params": again.variables["params"]}, latent, times, condition)
    np.testing.assert_array_equal(reloaded, trained)
