"""The vendored CLIP towers against transformers' own.

The claim is parity, not plausibility: the same weights and the same token ids
through the reference implementation and through dew have to produce the same
hidden states and the same pooled row, and the same pixels the same projected
image embedding. tools/clip_reference.py writes the fixtures under torch and
transformers, including a tiny random-weight checkpoint whose outputs are
committed, so the comparison runs in CI without a download.

Tolerances and the differences actually observed, fp32 on CPU:

- tiny checkpoint: max |hidden state difference| 1.4e-06, max |pooled
  difference| 9.5e-07, tolerance 1e-4, on hidden states reaching 2.8. Through
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
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from dew.inputs import CLIPText, Condition, InputSpec, Field
from dew.nn.text_encoders import (
    CLIPModel, CLIPTextModel, CLIPTextTransformer, translate_clip_config,
    translate_clip_weights, translate_config, translate_vision_config, translate_weights,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "clip"
TINY = FIXTURES / "tiny"
REAL = FIXTURES / "large-patch14"
TOLERANCE = 1e-4
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


def test_tiny_checkpoint_matches_the_reference():
    """Both towers of a CLIP checkpoint in one file, the text one loaded out of
    it, run on the reference's own token ids."""
    expected = reference(TINY)
    model = CLIPTextModel.from_pretrained(str(TINY))

    outputs = model(expected["input_ids"], expected["attention_mask"])

    hidden = float(np.max(np.abs(np.asarray(outputs.last_hidden_state, np.float32)
                                 - expected["last_hidden_state"])))
    pooled = float(np.max(np.abs(np.asarray(outputs.pooler_output, np.float32)
                                 - expected["pooler_output"])))
    assert hidden < TOLERANCE, f"max |hidden state difference| {hidden:.3e}"
    assert pooled < TOLERANCE, f"max |pooled difference| {pooled:.3e}"


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


def test_the_manifest_fields_rebuild_an_encoder_that_agrees():
    """A run manifest stores the spec as JSON, and inference rebuilds the
    encoder from it, so the round-trip has to come back as the same encoder
    and not just the same fields."""
    spec = InputSpec(Field("image", (8, 8, 3)),
                     {"textcontext": Condition(CLIPText.from_pretrained(str(TINY)))})
    data = spec.to_json()
    assert data["conditions"]["textcontext"] == {
        "encoder": {"name": "clip_text", "fields": {"checkpoint": str(TINY), "dtype": None}},
        "field": "text", "unconditional": ""}
    assert data["sample"] == {"key": "image", "shape": [8, 8, 3]}

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
    model = CLIPTextModel.from_pretrained(str(TINY))
    real = mask.astype(bool)

    filled = np.where(real, ids, 7).astype(np.int32)
    baseline = np.asarray(model(ids, mask).last_hidden_state, np.float32)
    refilled = np.asarray(model(filled, mask).last_hidden_state, np.float32)
    unmasked = np.asarray(model(ids, np.ones_like(mask)).last_hidden_state, np.float32)

    assert np.array_equal(baseline[real], refilled[real])
    assert np.array_equal(baseline[real], unmasked[real])
    assert np.max(np.abs(baseline[~real] - unmasked[~real])) > 1.0


def test_attention_stays_causal():
    """CLIP's text tower is causal, so a token can only move the rows at and
    after its own position."""
    expected = reference(TINY)
    ids, mask = expected["input_ids"], expected["attention_mask"]
    model = CLIPTextModel.from_pretrained(str(TINY))
    position = 3

    changed = ids.copy()
    changed[:, position] = 5
    baseline = np.asarray(model(ids, mask).last_hidden_state, np.float32)
    moved = np.asarray(model(changed, mask).last_hidden_state, np.float32)

    assert np.array_equal(baseline[:, :position], moved[:, :position])
    assert np.max(np.abs(baseline[:, position] - moved[:, position])) > 0.1


def test_pooled_row_follows_the_eos_token():
    """The reference pools the row of the first eos token unless the config
    carries the legacy eos_token_id 2, when it pools the argmax of the ids
    instead. The tiny fixture's eos is 1 and its characters are larger, so the
    two rules disagree and this pins the one the config asks for."""
    expected = reference(TINY)
    ids, mask = expected["input_ids"], expected["attention_mask"]
    model = CLIPTextModel.from_pretrained(str(TINY))
    assert model.config["eos_token_id"] == 1

    outputs = model(ids, mask)
    hidden = np.asarray(outputs.last_hidden_state, np.float32)
    rows = np.arange(ids.shape[0])
    eos = np.argmax(ids == 1, axis=-1)

    assert np.array_equal(np.asarray(outputs.pooler_output, np.float32),
                          hidden[rows, eos])
    argmax = np.argmax(ids, axis=-1)
    assert (argmax != eos).any(), "the fixture cannot tell the two rules apart"


def test_legacy_eos_token_id_pools_the_argmax_row():
    """openai's own configs carry eos_token_id 2, and transformers keeps
    pooling those at the argmax of the input ids (PR #24773). The real
    checkpoint takes this branch, so it has to be the config's choice."""
    expected = reference(TINY)
    ids, mask = expected["input_ids"], expected["attention_mask"]
    loaded = CLIPTextModel.from_pretrained(str(TINY))
    legacy = CLIPTextTransformer(**{**loaded.config, "eos_token_id": 2})

    outputs = legacy.apply(loaded.variables, ids, mask)
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


def test_vision_attention_reaches_every_patch():
    """The vision tower is not causal: the class row, which is pooled, attends
    to every patch, so changing the pixels moves the pooled row. With the text
    tower's causal mask reused here, position 0 would see nothing but itself
    and the pooled row would not depend on the image at all."""
    expected = reference(TINY)
    model = CLIPModel.from_pretrained(str(TINY))
    pixel_values = expected["pixel_values"]

    baseline = model.module.apply(
        model.variables, pixel_values, method=lambda clip, pixels: clip.vision_model(pixels))
    moved = model.module.apply(
        model.variables, np.flip(pixel_values, axis=-1),
        method=lambda clip, pixels: clip.vision_model(pixels))

    assert np.max(np.abs(np.asarray(baseline.pooler_output)
                         - np.asarray(moved.pooler_output))) > 0.1


def test_pixel_values_of_another_size_are_refused():
    """A checkpoint's position table covers one grid of patches; the reference
    refuses any other image size rather than interpolating on its own."""
    model = CLIPModel.from_pretrained(str(TINY))

    with pytest.raises(ValueError, match="3x8x8"):
        model.get_image_features(np.zeros((1, 3, 16, 16), np.float32))


def test_the_real_config_translates_to_the_tower_it_describes():
    """openai/clip-vit-large-patch14 nests the text fields under text_config and
    carries the whole transformers 4.16 config dump around them."""
    config = translate_config(fixture_config(REAL))

    assert config == {
        "vocab_size": 49408, "hidden_size": 768, "intermediate_size": 3072,
        "num_layers": 12, "num_heads": 12, "max_position_embeddings": 77,
        "layer_norm_eps": 1e-5, "eos_token_id": 2,
    }


def test_a_text_only_config_translates_the_same_way():
    """A checkpoint of the tower alone carries the CLIPTextConfig at the root,
    which has to describe the same tower as the nested copy."""
    nested = fixture_config(TINY)
    assert translate_config(nested) == translate_config(nested["text_config"])


def test_the_real_config_translates_to_the_towers_it_describes():
    """A full config carries both towers and the width of the heads; the
    vision fields come from vision_config the way the text ones come from
    text_config, and a CLIPVisionConfig on its own reads the same."""
    config = translate_clip_config(fixture_config(REAL))

    assert config == {
        "text": translate_config(fixture_config(REAL)),
        "vision": {
            "hidden_size": 1024, "intermediate_size": 4096, "num_layers": 24,
            "num_heads": 16, "image_size": 224, "patch_size": 14, "num_channels": 3,
            "layer_norm_eps": 1e-5,
        },
        "projection_dim": 768,
    }
    nested = fixture_config(TINY)
    assert translate_vision_config(nested) == translate_vision_config(nested["vision_config"])


def test_a_config_asking_for_another_activation_is_refused():
    """quick-GELU is the only activation here, and a checkpoint trained with
    GELU would otherwise load into a model that computes something else."""
    config = fixture_config(TINY)
    config["text_config"]["hidden_act"] = "gelu"

    with pytest.raises(ValueError, match="quick-GELU"):
        translate_config(config)


def test_an_unfamiliar_tensor_name_is_refused():
    """Skipping a name nobody mapped is how a checkpoint loads with half its
    weights. The text tower's loader skips the vision tower and the projection
    heads by name; the full model's loader maps them and skips only the
    buffers and the logit scale."""
    with pytest.raises(ValueError, match="text_model.encoder.layers.0.self_attn.qkv"):
        translate_weights({"text_model.encoder.layers.0.self_attn.qkv.weight":
                           np.zeros((3, 2), np.float32)})
    with pytest.raises(ValueError, match="vision_model.embeddings.patch_embedding.bias"):
        translate_clip_weights({"vision_model.embeddings.patch_embedding.bias":
                                np.zeros((2,), np.float32)})
    with pytest.raises(ValueError, match="logit_bias"):
        translate_clip_weights({"logit_bias": np.zeros((), np.float32)})

    beside_the_text_tower = {
        "vision_model.embeddings.class_embedding": np.zeros((2,), np.float32),
        "visual_projection.weight": np.zeros((2, 2), np.float32),
        "text_projection.weight": np.zeros((2, 2), np.float32),
        "logit_scale": np.zeros((), np.float32),
        "text_model.embeddings.position_ids": np.zeros((1, 77), np.int64),
        "vision_model.embeddings.position_ids": np.zeros((1, 5), np.int64),
    }
    assert translate_weights(beside_the_text_tower) == {}
    whole = translate_clip_weights(beside_the_text_tower)
    assert {name: leaf.shape for name, leaf in whole.items() if leaf.__class__ is np.ndarray} == {}
    assert sorted(whole) == ["text_projection", "vision_model", "visual_projection"]
    assert sorted(whole["vision_model"]) == ["class_embedding"]


def test_conv_kernels_land_in_linen_layout():
    """torch Conv2d holds [out, in, kh, kw] and nn.Conv [kh, kw, in, out]; a
    kernel handed over untransposed would convolve the wrong axes and only
    fail parity, so the layout is pinned by value."""
    kernel = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    tree = translate_clip_weights({"vision_model.embeddings.patch_embedding.weight": kernel})

    landed = tree["vision_model"]["patch_embedding"]["kernel"]
    assert landed.shape == (4, 5, 3, 2)
    assert landed[1, 2, 0, 1] == kernel[1, 0, 1, 2]


def test_a_checkpoint_that_does_not_fit_its_config_is_refused(tmp_path):
    """A config claiming more layers than the file holds would otherwise load a
    model whose last layer is still random."""
    for name in ("model.safetensors", "tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(TINY / name, tmp_path / name)
    config = fixture_config(TINY)
    config["text_config"]["num_hidden_layers"] = 3
    (tmp_path / "config.json").write_text(json.dumps(config))

    with pytest.raises(ValueError, match="missing"):
        CLIPTextModel.from_pretrained(str(tmp_path))


def test_a_full_checkpoint_that_does_not_fit_its_config_is_refused(tmp_path):
    """The same refusal for the vision tower: a config claiming a second
    vision layer the file does not hold."""
    for name in ("model.safetensors", "preprocessor_config.json"):
        shutil.copyfile(TINY / name, tmp_path / name)
    config = fixture_config(TINY)
    config["vision_config"]["num_hidden_layers"] = 2
    (tmp_path / "config.json").write_text(json.dumps(config))

    with pytest.raises(ValueError, match="vision_model.layers_1"):
        CLIPModel.from_pretrained(str(tmp_path))


def real_checkpoint_is_available() -> bool:
    if os.environ.get("DEW_NETWORK_TESTS") == "1":
        return True
    from huggingface_hub import try_to_load_from_cache
    repo = json.loads((REAL / "prompts.json").read_text())["repo"]
    return isinstance(try_to_load_from_cache(repo, "model.safetensors"), str)


@pytest.mark.network
@pytest.mark.skipif(not real_checkpoint_is_available(),
                    reason="openai/clip-vit-large-patch14 is neither cached nor "
                           "DEW_NETWORK_TESTS=1")
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
@pytest.mark.skipif(not real_checkpoint_is_available(),
                    reason="openai/clip-vit-large-patch14 is neither cached nor "
                           "DEW_NETWORK_TESTS=1")
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
