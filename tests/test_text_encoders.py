"""The vendored CLIP towers against transformers' own.

The claim is parity, not plausibility: the same weights and the same token ids
through the reference implementation and through dew have to produce the same
hidden states and the same pooled row, and the same pixels the same projected
image embedding. tools/clip_reference.py writes the fixtures under torch and
transformers, including a tiny random-weight checkpoint whose outputs are
committed, so the comparison runs in CI without a download.

Tolerances and the differences actually observed, fp32 on CPU:

- tiny checkpoint: max |hidden state difference| 1.4e-06, max |pooled
  difference| 9.5e-07, tolerance 4e-6, on hidden states reaching 2.8. Through
  the projection heads, max |text embedding difference| 7.2e-07 on values
  reaching 2.5 and max |image embedding difference| 4.8e-07 on values
  reaching 2.8, the pixel values bit-identical.
- clip-vit-large-patch14: max |hidden state difference| 1.9e-04 (mean 1.0e-06,
  median 6.0e-07), max |pooled difference| 4.8e-06, tolerance 1e-3, on hidden
  states reaching 33.1. Through the heads, max |text embedding difference|
  3.8e-06 on values reaching 14.2 and max |image embedding difference|
  1.2e-05 (mean 1.1e-06) on values reaching 6.6.

That last residue is fp32 rounding in CLIP's outlier channels, not a different
computation, and the reference carries more of it than the gap between the two:
transformers' own fp32 output differs from its fp64 output by 2.8e-04 on the
same prompts, and run in fp64 on both sides the two agree to 2.8e-13
(8.4e-15 relative). tools/clip_reference.py --fp64 writes that reference and
its docstring carries the dew side of the comparison.

What rounds differently at all is one rearrangement: dew's attention kernel
divides the query by sqrt(head_dim) before the logits, where the reference
multiplies the logits after (modeling_clip.py, eager_attention_forward).
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference

from dew.inputs import CLIPText, Condition, Field, InputSpec, T5Text
from dew.nn.text_encoders import CLIPModel, translate_config

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "clip"
TINY = FIXTURES / "tiny"
REAL = FIXTURES / "large-patch14"
TOLERANCE = 4e-6
"""Under three times the tiny checkpoint's largest residue, and five times under
what a final norm off by one part in 2^17 moves its states by."""
# The real checkpoint's outlier channels reach 33, where an fp32 sum carries
# more error than this whole test asks for.
REAL_TOLERANCE = 1e-3


def reference(directory):
    return np.load(directory / "reference.npz")


def prompts(directory):
    return json.loads((directory / "prompts.json").read_text())["prompts"]


def fixture_config(directory):
    return json.loads((directory / "config.json").read_text())


def synthetic_images(directory):
    """The uint8 images the fixture was run on, from the recipe it records."""
    recipe = json.loads((directory / "prompts.json").read_text())["images"]
    return np.random.RandomState(recipe["seed"]).randint(
        0, 256, tuple(recipe["shape"]), dtype=np.uint8)


def image_processor(directory_or_repo):
    from transformers import CLIPImageProcessorPil
    return CLIPImageProcessorPil.from_pretrained(str(directory_or_repo))


def largest_difference(actual, expected) -> float:
    return float(np.max(np.abs(np.asarray(actual, np.float32) - expected)))


def hidden(encoder, ids, mask) -> np.ndarray:
    """The encoder's last hidden state for `ids` under `mask`."""
    tokens = {"input_ids": ids, "attention_mask": mask}
    return np.asarray(encoder.encode(encoder.params, tokens).hidden, np.float32)


def test_translating_inside_a_module_keeps_its_native_parameter_names():
    import jax
    import jax.numpy as jnp
    from flax import linen as nn

    config = fixture_config(TINY)

    class Encoded(nn.Module):
        @nn.compact
        def __call__(self, input_ids):
            return translate_config(config).value(input_ids).last_hidden_state

    input_ids = jnp.zeros((1, 2), jnp.int32)
    module = Encoded()
    variables = module.init(jax.random.key(9), input_ids)
    assert set(variables["params"]) == {"CLIPTextTransformer_0"}
    direct = translate_config(config).value
    expected = direct.apply({"params": variables["params"]["CLIPTextTransformer_0"]}, input_ids)
    np.testing.assert_array_equal(module.apply(variables, input_ids), expected.last_hidden_state)


def test_tiny_checkpoint_matches_the_reference():
    """Both towers of a CLIP checkpoint in one file, the text one loaded out of
    it, run on the reference's own token ids."""
    expected = reference(TINY)
    encoder = CLIPText.from_pretrained(str(TINY))

    outputs = encoder.transformer.apply(encoder.params, expected["input_ids"], expected["attention_mask"])

    hidden = float(np.max(np.abs(np.asarray(outputs.last_hidden_state, np.float32)
                                 - expected["last_hidden_state"])))
    pooled = float(np.max(np.abs(np.asarray(outputs.pooler_output, np.float32)
                                 - expected["pooler_output"])))
    assert hidden < TOLERANCE, f"max |hidden state difference| {hidden:.3e}"
    assert pooled < TOLERANCE, f"max |pooled difference| {pooled:.3e}"

    ids = expected["input_ids"]
    rows = np.arange(ids.shape[0])
    eos = np.argmax(ids == encoder.transformer.eos_token_id, axis=-1)
    np.testing.assert_array_equal(np.asarray(outputs.pooler_output, np.float32),
                                  np.asarray(outputs.last_hidden_state, np.float32)[rows, eos])
    assert (np.argmax(ids, axis=-1) != eos).any()


def test_the_encoder_loads_and_encodes():
    """`CLIPText.from_pretrained` loads the vendored tower and the checkpoint's
    tokenizer; `encode` under the encoder's own params returns the
    reference's hidden states with the tokenizer's mask."""
    expected = reference(TINY)
    encoder = CLIPText.from_pretrained(str(TINY))

    tokens = encoder.tokenize(prompts(TINY))
    context = encoder.encode(encoder.params, tokens)

    assert np.array_equal(tokens["input_ids"], expected["input_ids"])
    assert np.array_equal(np.asarray(context.mask), expected["attention_mask"])
    difference = float(np.max(np.abs(np.asarray(context.hidden, np.float32)
                                     - expected["last_hidden_state"])))
    assert difference < TOLERANCE, f"max |hidden state difference| {difference:.3e}"
    assert encoder.captions(tokens)[0] == "a red bird"


def test_the_json_fields_rebuild_an_encoder_that_agrees():
    """A run's record stores the spec as JSON, and inference rebuilds the
    encoder from it, so the round-trip has to come back as the same encoder,
    equal fields and equal outputs."""
    spec = InputSpec(Field("image", (8, 8, 3)),
                     {"textcontext": Condition(CLIPText.from_pretrained(str(TINY)))})
    data = spec.to_json()


    restored = InputSpec.from_json(data)
    assert restored.sample == spec.sample
    encoder, rebuilt = spec.conditions["textcontext"].encoder, restored.conditions["textcontext"].encoder
    tokens = encoder.tokenize(prompts(TINY))
    assert np.array_equal(np.asarray(rebuilt.encode(rebuilt.params, tokens).hidden),
                          np.asarray(encoder.encode(encoder.params, tokens).hidden))





def test_padding_is_masked_where_the_reference_masks_it():
    """A padded slot must reach no real token. Right padding plus causality
    leaves the real rows alone whatever the padding holds, and the mask is what
    keeps the padded rows themselves off the padding: dropping it moves those
    rows by 4.4, which the parity fixture would catch."""
    expected = reference(TINY)
    ids, mask = expected["input_ids"], expected["attention_mask"]
    encoder = CLIPText.from_pretrained(str(TINY))
    real = mask.astype(bool)

    filled = np.where(real, ids, 7).astype(np.int32)
    baseline = hidden(encoder, ids, mask)
    refilled = hidden(encoder, filled, mask)
    unmasked = hidden(encoder, ids, np.ones_like(mask))

    assert np.array_equal(baseline[real], refilled[real])
    assert np.array_equal(baseline[real], unmasked[real])
    assert np.max(np.abs(baseline[~real] - unmasked[~real])) > 1.0


def test_attention_stays_causal():
    """CLIP's text tower is causal, so a token can only move the rows at and
    after its own position."""
    expected = reference(TINY)
    ids, mask = expected["input_ids"], expected["attention_mask"]
    encoder = CLIPText.from_pretrained(str(TINY))
    position = 3

    changed = ids.copy()
    changed[:, position] = 5
    baseline = hidden(encoder, ids, mask)
    moved = hidden(encoder, changed, mask)

    assert np.array_equal(baseline[:, :position], moved[:, :position])
    assert np.max(np.abs(baseline[:, position] - moved[:, position])) > 0.1


def test_legacy_eos_token_id_pools_the_argmax_row():
    """openai's own configs carry eos_token_id 2, and transformers keeps
    pooling those at the argmax of the input ids (PR #24773). The real
    checkpoint takes this branch, so it has to be the config's choice."""
    expected = reference(TINY)
    ids, mask = expected["input_ids"], expected["attention_mask"]
    encoder = CLIPText.from_pretrained(str(TINY))
    legacy = encoder.transformer.clone(eos_token_id=2)

    outputs = legacy.apply(encoder.params, ids, mask)
    hidden = np.asarray(outputs.last_hidden_state, np.float32)
    rows = np.arange(ids.shape[0])

    assert np.array_equal(np.asarray(outputs.pooler_output, np.float32),
                          hidden[rows, np.argmax(ids, axis=-1)])


def test_tiny_checkpoint_text_projection_matches_the_reference():
    """`get_text_features` is the pooled row through `text_projection`, which
    the metrics score with; the head is loaded out of the same file."""
    expected = reference(TINY)
    model = CLIPModel.from_pretrained(str(TINY))

    embeds = model.get_text_features(expected["input_ids"], expected["attention_mask"])

    difference = largest_difference(embeds, expected["text_embeds"])
    assert difference < TOLERANCE, f"max |text embedding difference| {difference:.3e}"


def test_tiny_checkpoint_image_embeddings_match_the_reference():
    """The vision tower on the reference's own pixel values, through
    `visual_projection`. The images are taller than wide, so the processor's
    resize and centre crop both run, and its pixel values have to be the
    reference's exactly since it is the reference's own processor."""
    expected = reference(TINY)
    model = CLIPModel.from_pretrained(str(TINY))

    pixel_values = image_processor(TINY)(
        images=synthetic_images(TINY), return_tensors="np")["pixel_values"]
    assert np.array_equal(pixel_values, expected["pixel_values"])
    embeds = model.get_image_features(pixel_values)

    difference = largest_difference(embeds, expected["image_embeds"])
    assert difference < TOLERANCE, f"max |image embedding difference| {difference:.3e}"

    baseline = model.module.apply(
        model.variables, pixel_values, method=lambda clip, pixels: clip.vision_model(pixels))
    moved = model.module.apply(model.variables, np.flip(pixel_values, axis=-1),
                               method=lambda clip, pixels: clip.vision_model(pixels))
    assert np.max(np.abs(np.asarray(baseline.pooler_output) - np.asarray(moved.pooler_output))) > 0.1


SAFETY_CROP = {"size": {"shortest_edge": 224}, "crop_size": {"height": 224, "width": 224}}
SMALL_CROP = {"crop_size": {"height": 8, "width": 8}, "do_resize": False}


@pytest.mark.parametrize("config, height, width", [
    (SAFETY_CROP, 440, 440), (SAFETY_CROP, 320, 360), (SAFETY_CROP, 360, 320),
    (SMALL_CROP, 5, 6), (SMALL_CROP, 7, 9)])
def test_clip_image_transform_matches_the_reference_processor(config, height, width):
    """The safety checker's preprocessing, against the PIL processor it
    follows (`CLIPImageTransform` says why that one). The shorter side
    resizes to exactly `shortest_edge` and the longer to
    int(size * long / short), so an image whose side the size does not
    divide still fills the crop. An image smaller than the crop sits in it
    with an odd pad row above it and an odd pad column to its left."""
    from transformers import CLIPImageProcessorPil

    from dew.inputs.diffusion import CLIPImageTransform

    pixels = np.random.RandomState(0).randint(0, 256, (1, height, width, 3), dtype=np.uint8)

    expected = CLIPImageProcessorPil(**config)(list(pixels), return_tensors="np")["pixel_values"]
    actual = np.asarray(CLIPImageTransform.from_config(config)(pixels))

    np.testing.assert_allclose(actual, expected, atol=1e-6)


@pytest.mark.network
def test_the_real_checkpoint_matches_the_reference():
    """The flagship: twelve layers of clip-vit-large-patch14, fp32, from the
    checkpoint's own safetensors, against what transformers computes for the
    same prompts."""
    expected = reference(REAL)
    repo = json.loads((REAL / "prompts.json").read_text())["repo"]
    encoder = CLIPText.from_pretrained(repo)

    tokens = encoder.tokenize(prompts(REAL))
    assert np.array_equal(np.asarray(tokens["input_ids"], np.int32),
                          expected["input_ids"])
    outputs = encoder.transformer.apply(encoder.params, tokens["input_ids"],
                                        tokens["attention_mask"])

    hidden = float(np.max(np.abs(np.asarray(outputs.last_hidden_state, np.float32)
                                 - expected["last_hidden_state"])))
    pooled = float(np.max(np.abs(np.asarray(outputs.pooler_output, np.float32)
                                 - expected["pooler_output"])))
    assert hidden < REAL_TOLERANCE, f"max |hidden state difference| {hidden:.3e}"
    assert pooled < REAL_TOLERANCE, f"max |pooled difference| {pooled:.3e}"


@pytest.mark.network
def test_the_real_checkpoint_embeddings_match_the_reference():
    """Both heads of clip-vit-large-patch14, the twenty-four vision layers
    behind one of them, against what transformers computes for the same
    prompts and the same synthetic images."""
    expected = reference(REAL)
    repo = json.loads((REAL / "prompts.json").read_text())["repo"]
    model = CLIPModel.from_pretrained(repo)

    pixel_values = image_processor(repo)(
        images=synthetic_images(REAL), return_tensors="np")["pixel_values"]
    image_embeds = model.get_image_features(pixel_values)
    text_embeds = model.get_text_features(expected["input_ids"], expected["attention_mask"])

    image = largest_difference(image_embeds, expected["image_embeds"])
    text = largest_difference(text_embeds, expected["text_embeds"])
    assert image < REAL_TOLERANCE, f"max |image embedding difference| {image:.3e}"
    assert text < REAL_TOLERANCE, f"max |text embedding difference| {text:.3e}"


def test_a_t5_tower_computes_in_its_compute_dtype_as_the_reference_does():
    """FP32 masters under `dtype=bfloat16`: the token and relative-bias tables
    are read in bfloat16, as transformers' T5 reads its own in the dtype the
    model runs in, so the residual stream every block adds to is bfloat16
    (the tables read at their FP32 storage had kept it FP32). The last hidden
    state is as exact as transformers' bfloat16 encoder from its float64 one
    (tests/reference_error.py's rule); observed 6.6e-3 and 6.8e-3 RMS."""
    import jax
    import jax.numpy as jnp
    import torch
    from transformers import T5EncoderModel

    directory = Path(__file__).resolve().parent / "fixtures/t5/tiny"
    expected = reference(directory)
    ids, mask = expected["input_ids"], expected["attention_mask"]
    encoder = T5Text.from_pretrained(str(directory), dtype=jnp.bfloat16)
    _, state = encoder.transformer.apply(encoder.params, jnp.asarray(ids), jnp.asarray(mask),
                                         capture_intermediates=True, mutable=["intermediates"])
    sites = {name: site for name, site in state["intermediates"].items() if name != "__call__"}
    assert {"embed_tokens", "relative_attention_bias_0", "layers_0", "layers_1"} <= sites.keys()
    for name, site in sites.items():
        for output in jax.tree.leaves(site["__call__"]):
            assert output.dtype == jnp.bfloat16, name

    def transformers(dtype):
        model = T5EncoderModel.from_pretrained(directory, dtype=dtype).eval()
        with torch.no_grad():
            return model(input_ids=torch.from_numpy(ids.astype(np.int64)),
                         attention_mask=torch.from_numpy(mask.astype(np.int64))).last_hidden_state.float().numpy()

    real = mask.astype(bool)
    assert_as_exact_as_the_reference(hidden(encoder, ids, mask)[real], transformers(torch.bfloat16)[real],
                                     transformers(torch.float64)[real], "T5 bfloat16 hidden states")


@pytest.mark.parametrize("kind", ["clip-text", "clip", "t5"])
def test_public_text_encoder_storage_is_separate_from_compute(kind, tmp_path):
    """`dtype` sets the compute and `param_dtype` the stored weights, each
    leaf the FP32 master cast; supplied variables are bound as they are, from
    a directory holding no weights, and encode as the loaded ones do."""
    import jax
    import jax.numpy as jnp
    from flax.core import freeze

    load = {"clip-text": CLIPText.from_pretrained, "clip": CLIPModel.from_pretrained,
            "t5": T5Text.from_pretrained}[kind]
    directory = Path(__file__).resolve().parent / "fixtures/t5/tiny" if kind == "t5" else TINY
    masters = load(str(directory), dtype=jnp.bfloat16)
    native = load(str(directory), dtype=jnp.float32, param_dtype="bfloat16")
    if isinstance(native, CLIPModel):
        assert isinstance(masters, CLIPModel) and native.module.dtype == jnp.float32
        stored = (masters.variables, native.variables)
    else:
        assert isinstance(masters, (CLIPText, T5Text)) and native.transformer.dtype == jnp.float32
        stored = (masters.params, native.params)
    assert jax.tree.structure(stored[0]) == jax.tree.structure(stored[1])
    for before, after in zip(jax.tree.leaves(stored[0]), jax.tree.leaves(stored[1]), strict=True):
        assert isinstance(before, jax.Array) and isinstance(after, jax.Array)
        assert np.asarray(before).dtype == np.float32
        assert after.dtype == jnp.bfloat16
        np.testing.assert_array_equal(after, np.asarray(before).astype(jnp.bfloat16))
    if isinstance(native, (CLIPText, T5Text)):
        metadata = tmp_path / "metadata"
        metadata.mkdir()
        for file in directory.iterdir():
            if file.suffix != ".safetensors":
                shutil.copy(file, metadata)
        supplied = freeze({**jax.tree.map(np.asarray, native.params),
                           "constants": {"step": np.asarray(0, np.int32)}})
        rebound = type(native).from_pretrained(str(metadata), dtype=jnp.float32, params=supplied)
        assert rebound.params is supplied
        expected = reference(directory)
        np.testing.assert_array_equal(hidden(rebound, expected["input_ids"], expected["attention_mask"]),
                                      hidden(native, expected["input_ids"], expected["attention_mask"]))
