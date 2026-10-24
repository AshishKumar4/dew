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
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from interop_support import extract_fixture, fixture_arrays, scaled_gap as relative_gap

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
    def run(values):
        return model.apply({"params": params}, latent, times,
                                         DenoisingCondition(jnp.asarray(values), mask=jnp.asarray(mask)))
    np.testing.assert_allclose(run(filled), run(context), atol=1e-6)
    # Without the mask the short row reads its padding as prompt.
    unmasked = model.apply({"params": params}, latent, times, DenoisingCondition(jnp.asarray(context)))
    assert relative_gap(unmasked[1], nhwc(arrays["padded.output"], rows, columns)[1]) > 1e-3


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
        assert relative_gap(condition.context[row, :length], expected) < 1e-5
        assert not np.asarray(condition.context[row, length:]).any()
    assert encoder.captions(encoder.tokenize(prompts)) == tuple(prompts)
    # The pipeline encodes an empty prompt as one space.
    np.testing.assert_array_equal(encoder.tokenize([""])["input_ids"],
                                  encoder.tokenize([" "])["input_ids"])
    with pytest.raises(ValueError):
        encoder.tokenize(["x" * (encoder.tokens + 1)])


def test_the_prompt_states_are_the_decoders_own_last_layer_output():
    """The conditioner reads the decoder's layers through its own forward (`LayerStates`), so a
    decoder that scales its embeddings, by Gemma's sqrt(d) or a muP multiplier, gives the
    conditioner the last layer's output: the decoder's layers over its own scaled embeddings."""
    from dew.inputs.diffusion import _layer_states
    from dew.nn.backbones.causal_transformer import CausalTransformer

    model = CausalTransformer(vocab_size=37, emb_features=32, num_layers=2, num_heads=4, mlp_features=64,
                              max_seq_len=16, embedding_scale=True, embedding_multiplier=12.0)
    ids = jax.random.randint(jax.random.key(0), (2, 12), 0, 37)
    params = model.init(jax.random.key(1), ids)

    def layers(decoder, ids):
        embedded = decoder.scaled_embeddings(decoder.token_embeddings(ids))
        return decoder.stack(embedded, train=False, decode=False, per_layer_input=None)

    (last,) = _layer_states(model, params, ids, (-1,))
    np.testing.assert_array_equal(last, model.apply(params, ids, method=layers))


def test_a_looped_layer_is_read_at_its_last_pass():
    """A layer a loop runs twice is read at its last pass, which the final norm reads: the looped
    stack's last layer, normed, is the model's hidden states, and its first pass is not."""
    from dew.inputs.diffusion import _layer_states
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.backbones.decoder_stack import Loop

    model = CausalTransformer(vocab_size=37, emb_features=32, num_layers=2, num_heads=4, mlp_features=64,
                              max_seq_len=16, loop=Loop(steps=2))
    ids = jax.random.randint(jax.random.key(0), (2, 12), 0, 37)
    params = model.init(jax.random.key(1), ids)
    (last,) = _layer_states(model, params, ids, (-1,))
    hidden = model.apply(params, ids, method="hidden_states")
    normed = model.apply(params, last, method=lambda decoder, x: decoder.norm(x))
    np.testing.assert_array_equal(normed, hidden)
    _, kept = model.apply(params, ids, method="hidden_states", capture_intermediates=model.capture_layers,
                          mutable=list(model.layer_collections))
    first = kept["intermediates"]["layers_1"]["__call__"][0]
    assert not np.array_equal(model.apply(params, first, method=lambda decoder, x: decoder.norm(x)), hidden)


def test_a_hidden_states_conditioner_reads_a_looped_tower_at_its_last_pass():
    """FLUX.2's and Z-Image's conditioning (`HiddenStatesConditioner`) reads its tower's last layer
    as the forward left it: a layer a loop runs twice gives its last pass, which the final norm
    turns into the tower's hidden states, not its first."""
    from dew.inputs.diffusion import HiddenStatesConditioner
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.backbones.decoder_stack import Loop

    model = CausalTransformer(vocab_size=37, emb_features=32, num_layers=2, num_heads=4, mlp_features=64,
                              max_seq_len=16, loop=Loop(steps=2))
    ids = jax.random.randint(jax.random.key(0), (2, 12), 0, 37)
    params = model.init(jax.random.key(1), ids)
    conditioner = HiddenStatesConditioner(model, None, params, "tower", template="qwen3", layers=(2,))
    mask = np.ones(ids.shape, bool)
    condition = conditioner.encode({"text_encoder": params}, {"input_ids": ids, "attention_mask": mask})
    positions = jnp.broadcast_to(jnp.arange(ids.shape[1]), ids.shape)
    hidden = model.apply(params, ids, method="hidden_states", attention_mask=mask, positions=positions)
    normed = model.apply(params, condition.context, method=lambda decoder, x: decoder.norm(x))
    np.testing.assert_array_equal(normed, hidden)


@pytest.mark.parametrize("copies", [None, 2])
def test_every_layer_reads_as_the_layer_returned_it_with_or_without_residual_copies(copies):
    """The last layer's state is the stack's own output and a middle one's the layer's own, each in
    the layout the layer returned it: with AltUp's residual copies, all of them, before the model
    collapses them for its final norm; the last layer's is what its capture kept too."""
    from model_support import decoder

    from dew.inputs.diffusion import _layer_states
    from dew.nn.gemma3n import AltUp

    # Two layers, so the first is read from its capture and the last from the stack's output.
    model = decoder().clone(num_layers=2, **({} if copies is None else {"altup": AltUp(num_inputs=copies)}))
    ids = jax.random.randint(jax.random.key(0), (2, 6), 0, model.vocab_size)
    params = model.init(jax.random.key(1), ids)
    first, last = _layer_states(model, params, ids, (0, -1))
    _, kept = model.apply(params, ids, method="hidden_states", capture_intermediates=model.capture_layers,
                          mutable=list(model.layer_collections))
    captured = kept["intermediates"]
    np.testing.assert_array_equal(first, captured["layers_0"]["__call__"][-1])
    np.testing.assert_array_equal(last, captured[f"layers_{model.num_layers - 1}"]["__call__"][-1])
    assert first.shape == last.shape == ((2, 6, model.emb_features) if copies is None
                                         else (copies, 2, 6, model.emb_features))


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
        expected = nhwc(arrays[f"pipeline.latents.{row}"], rows, columns)[0]
        assert relative_gap(np.asarray(walked.latents)[row], expected) < 2e-5
        assert relative_gap(images[row], arrays[f"pipeline.images.{row}"][0]) < 2e-5
        alone = task(task.prepare([prompt], initial=initial[row:row + 1], key=0),
                     key=jax.random.PRNGKey(0)).host()
        assert relative_gap(np.asarray(alone.latents)[0], expected) < 2e-5


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
                                          again.inputs.conditions["conditioning"].encoder.recorded_fields().items()
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
