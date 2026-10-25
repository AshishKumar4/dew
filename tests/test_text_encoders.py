"""The vendored CLIP towers against transformers' own.

The claim is parity, not plausibility: the same weights and the same token ids
through the reference implementation and through dew have to produce the same
hidden states and the same pooled row, and the same pixels the same projected
image embedding. Every output is on par with transformers' own float32 output
from its float64 one over the same weights (`transformers_clip`), over drawn
prompts and images (check 1, tests/reference_error.py). tools/clip_reference.py
writes the fixtures under torch and transformers, including a tiny
random-weight checkpoint whose token ids, pixels and float32 outputs are
committed; the float32 and float64 reference outputs are computed here, so the
comparison runs in CI without a download.

What rounds differently at all is one rearrangement: dew's attention kernel
divides the query by sqrt(head_dim) before the logits, where the reference
multiplies the logits after (modeling_clip.py, eager_attention_forward).
"""

import functools
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from reference_error import as_given, assert_as_exact

from dew.inputs import CLIPText, Condition, Field, InputSpec, T5Text
from dew.nn.text_encoders import CLIPModel, translate_config

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "clip"
TINY = FIXTURES / "tiny"
REAL = FIXTURES / "large-patch14"


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


SEEDS = {'tiny text tower': 8, 'tiny encode': 8, 'tiny text projection': 8, 'tiny image embeddings': 8,
         'quick_gelu image embeddings': 8, 'gelu image embeddings': 8,
         'gelu_pytorch_tanh image embeddings': 8, 'real text tower': 16, 'real embeddings': 8,
         'T5 bfloat16 hidden states': 8, 'image transform': 8}
"""Check 1's declared seeds for each site of the file, sized by `seeds_for` (on armada)."""
WORDS = ("a", "red", "bird", "two", "cats", "on", "mat", "painted", "x", "blue", "sky", "of", "the", "dog")
"""The words a drawn prompt is made of."""


def drawn_prompts(seed: int, rows: int = 4) -> list[str]:
    """`rows` prompts of up to eleven drawn words, the empty one among the draws."""
    rng = np.random.default_rng([0xC11, seed])
    return [" ".join(rng.choice(WORDS, rng.integers(0, 12))) for _ in range(rows)]


def text_draw(encoder):
    """Check 1's draw of drawn prompts' token ids and masks, as `encoder` (anything with a
    `tokenize`) tokenizes them."""
    def draw(seed):
        tokens = encoder.tokenize(drawn_prompts(seed))
        return ({name: np.asarray(tokens[name], np.int32) for name in ("input_ids", "attention_mask")},)
    return draw


def image_draw(directory_or_repo, shape):
    """Check 1's draw of uint8 images of `shape` through `directory_or_repo`'s own processor."""
    processor = image_processor(directory_or_repo)

    def draw(seed):
        images = np.random.RandomState(seed).randint(0, 256, shape, dtype=np.uint8)
        return ({"pixel_values": processor(images=images, return_tensors="np")["pixel_values"]},)
    return draw


@functools.cache
def reference_clip(source, dtype):
    """transformers' CLIPModel over `source`'s weights at `dtype`, float64 built under `float64`."""
    from transformers import CLIPModel as ReferenceCLIP

    with precision(dtype):
        return ReferenceCLIP.from_pretrained(str(source), attn_implementation="eager").to(dtype).eval()


def precision(dtype):
    """transformers at `dtype`: under `float64` for float64, every float32 pin of it widened."""
    import contextlib

    import torch

    from tools.torch_precision import float64

    return float64() if dtype == torch.float64 else contextlib.nullcontext()


def transformers_clip(source, inputs, dtype, names):
    """transformers' CLIPModel over `source`'s weights at `dtype`, float64 built and run under
    `float64`: of its text tower's states and pooled row and its text embeddings for `inputs`' token
    ids and mask, and its image embeddings for their pixel values, the outputs `names`, at `dtype`."""
    import torch

    model = reference_clip(source, dtype)
    outputs = {}
    with precision(dtype), torch.no_grad():
        if {"last_hidden_state", "pooler_output", "text_embeds"} & set(names):
            ids, mask = (torch.from_numpy(np.asarray(inputs[name], np.int64))
                         for name in ("input_ids", "attention_mask"))
            text = model.text_model(input_ids=ids, attention_mask=mask)
            outputs.update(last_hidden_state=text.last_hidden_state, pooler_output=text.pooler_output)
            # `get_text_features`' projection of the pooled row, without running the tower again.
            outputs["text_embeds"] = model.text_projection(text.pooler_output)
        if "image_embeds" in names:
            pixels = torch.from_numpy(np.asarray(inputs["pixel_values"])).to(dtype)
            outputs["image_embeds"] = model.get_image_features(pixel_values=pixels).pooler_output
    return {name: outputs[name].numpy() for name in names}


def text_tower(encoder):
    """`encoder`'s text transformer on a draw's token ids and mask: its states and its pooled row."""
    apply = jax.jit(encoder.transformer.apply)

    def run(inputs):
        outputs = apply(encoder.params, inputs["input_ids"], inputs["attention_mask"])
        return {"last_hidden_state": outputs.last_hidden_state, "pooler_output": outputs.pooler_output}
    return run


def assert_as_exact_as_transformers(ours, names, source, draw, label: str) -> None:
    """Check 1: each of `ours(inputs)`' outputs, `names`, on par with transformers' float32 one from its
    float64 one over `draw`'s inputs."""
    import torch

    def theirs(dtype):
        return lambda inputs: transformers_clip(source, inputs, dtype, names)

    assert_as_exact(as_given(lambda inputs: {name: np.asarray(value)
                                             for name, value in ours(inputs).items()}),
                    theirs(torch.float32), draw, label, seeds=SEEDS[label], truth_fn=theirs(torch.float64))


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
    it, run on drawn prompts' token ids; on the reference's own, its pooled
    row is the end of text's state, not the largest id's."""
    expected = reference(TINY)
    encoder = CLIPText.from_pretrained(str(TINY))
    assert_as_exact_as_transformers(text_tower(encoder), ("last_hidden_state", "pooler_output"), TINY,
                                    text_draw(encoder), "tiny text tower")
    outputs = encoder.transformer.apply(encoder.params, expected["input_ids"], expected["attention_mask"])

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
    assert encoder.captions(tokens)[0] == "a red bird"

    def encoded(inputs):
        return {"last_hidden_state": encoder.encode(encoder.params, inputs).hidden}

    assert_as_exact_as_transformers(encoded, ("last_hidden_state",), TINY, text_draw(encoder), "tiny encode")


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
    assert not np.array_equal(baseline[~real], unmasked[~real])


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
    assert not (baseline[:, position] == moved[:, position]).all(axis=-1).any()


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
    model = CLIPModel.from_pretrained(str(TINY))

    def projected(inputs):
        return {"text_embeds": model.get_text_features(inputs["input_ids"], inputs["attention_mask"])}

    assert_as_exact_as_transformers(projected, ("text_embeds",), TINY,
                                    text_draw(CLIPText.from_pretrained(str(TINY))), "tiny text projection")


def test_tiny_checkpoint_image_embeddings_match_the_reference():
    """The vision tower on drawn images' pixel values, through
    `visual_projection`. The images are taller than wide, so the processor's
    resize and centre crop both run, and on the reference's own images its
    pixel values have to be the reference's exactly since it is the
    reference's own processor."""
    expected = reference(TINY)
    model = CLIPModel.from_pretrained(str(TINY))

    pixel_values = image_processor(TINY)(
        images=synthetic_images(TINY), return_tensors="np")["pixel_values"]
    assert np.array_equal(pixel_values, expected["pixel_values"])
    shape = tuple(json.loads((TINY / "prompts.json").read_text())["images"]["shape"])

    def embedded(inputs):
        return {"image_embeds": model.get_image_features(inputs["pixel_values"])}

    assert_as_exact_as_transformers(embedded, ("image_embeds",), TINY, image_draw(TINY, shape),
                                    "tiny image embeddings")

    baseline = model.module.apply(
        model.variables, pixel_values, method=lambda clip, pixels: clip.vision_model(pixels))
    moved = model.module.apply(model.variables, np.flip(pixel_values, axis=-1),
                               method=lambda clip, pixels: clip.vision_model(pixels))
    assert not (np.asarray(baseline.pooler_output) == np.asarray(moved.pooler_output)).all(axis=-1).any()


@pytest.mark.parametrize("hidden_act", ["quick_gelu", "gelu", "gelu_pytorch_tanh"])
def test_a_vision_tower_runs_the_activation_its_config_names(hidden_act, tmp_path):
    """The vision tower's MLP runs the `hidden_act` its config names, as the text tower's does
    (`tower_activation`): the tiny checkpoint's image embeddings under each activation are on par
    with transformers' own float32 ones from its float64 ones over drawn images (check 1)."""
    copy = tmp_path / "tiny"
    shutil.copytree(TINY, copy)
    config = fixture_config(copy)
    config["vision_config"]["hidden_act"] = hidden_act
    (copy / "config.json").write_text(json.dumps(config))
    model = CLIPModel.from_pretrained(str(copy))
    shape = tuple(json.loads((TINY / "prompts.json").read_text())["images"]["shape"])

    def embedded(inputs):
        return {"image_embeds": model.get_image_features(inputs["pixel_values"])}

    assert_as_exact_as_transformers(embedded, ("image_embeds",), copy, image_draw(TINY, shape),
                                    f"{hidden_act} image embeddings")


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

    processor, transform = CLIPImageProcessorPil(**config), CLIPImageTransform.from_config(config)
    resized = CLIPImageProcessorPil(**config, do_rescale=False, do_normalize=False)

    def draw(seed):
        return (np.random.RandomState(seed).randint(0, 256, (1, height, width, 3), dtype=np.uint8),)

    def normalized(pixels):
        """The processor's resized and cropped integers, rescaled and normalized in float64."""
        crop = np.asarray(resized(list(np.asarray(pixels, np.uint8)), return_tensors="np")["pixel_values"],
                          np.float64)
        mean, std = (np.asarray(value, np.float64)[:, None, None] for value in (processor.image_mean,
                                                                                processor.image_std))
        return (crop * processor.rescale_factor - mean) / std

    assert_as_exact(as_given(lambda pixels: np.asarray(transform(pixels))),
                    lambda pixels: processor(list(pixels), return_tensors="np")["pixel_values"], draw,
                    f"image transform {height}x{width}", seeds=SEEDS['image transform'], truth_fn=normalized)


@pytest.mark.network
def test_the_real_checkpoint_matches_the_reference():
    """The flagship: twelve layers of clip-vit-large-patch14, fp32, from the
    checkpoint's own safetensors, against what transformers computes for the
    same prompts. Its outlier channels reach 33, where an fp32 sum carries
    errors of 2.8e-4: transformers' own float32 states lie that far from its
    float64 ones, and the rule holds Dew to them."""
    expected = reference(REAL)
    repo = json.loads((REAL / "prompts.json").read_text())["repo"]
    encoder = CLIPText.from_pretrained(repo)

    tokens = encoder.tokenize(prompts(REAL))
    assert np.array_equal(np.asarray(tokens["input_ids"], np.int32),
                          expected["input_ids"])
    assert_as_exact_as_transformers(text_tower(encoder), ("last_hidden_state", "pooler_output"), repo,
                                    text_draw(encoder), "real text tower")


@pytest.mark.network
def test_the_real_checkpoint_embeddings_match_the_reference():
    """Both heads of clip-vit-large-patch14, the twenty-four vision layers
    behind one of them, against what transformers computes for the same
    drawn prompts and drawn images (check 1)."""
    repo = json.loads((REAL / "prompts.json").read_text())["repo"]
    model = CLIPModel.from_pretrained(repo)

    shape = tuple(json.loads((REAL / "prompts.json").read_text())["images"]["shape"])
    draw = image_draw(repo, shape)

    def embedded(inputs):
        return {"image_embeds": model.get_image_features(inputs["pixel_values"]),
                "text_embeds": model.get_text_features(inputs["input_ids"], inputs["attention_mask"])}

    from transformers import CLIPTokenizer

    # The checkpoint's tokenizer alone, padded and truncated as `CLIPText.tokenize` does.
    tokenizer = CLIPTokenizer.from_pretrained(repo)
    prompts_of = text_draw(SimpleNamespace(tokenize=lambda texts: tokenizer(
        texts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True,
        return_tensors="np")))

    def drawn(seed):
        # Drawn images and drawn prompts, each its own draw of the seed.
        return ({**draw(seed)[0], **prompts_of(seed)[0]},)

    assert_as_exact_as_transformers(embedded, ("image_embeds", "text_embeds"), repo, drawn, "real embeddings")


def test_a_t5_tower_computes_in_its_compute_dtype_as_the_reference_does():
    """FP32 masters under `dtype=bfloat16`: the token and relative-bias tables
    are read in bfloat16, as transformers' T5 reads its own in the dtype the
    model runs in, so the residual stream every block adds to is bfloat16, and
    every site the forward calls returns bfloat16.
    Every position of the last hidden state the conditioner hands on, padded
    rows included (the mask holds keys out, not rows), is on par with
    transformers' bfloat16 encoder from its float64 one built and run under
    `float64`, over drawn prompts (check 1)."""
    import jax.numpy as jnp
    import torch
    from transformers import T5EncoderModel


    directory = Path(__file__).resolve().parent / "fixtures/t5/tiny"
    expected = reference(directory)
    ids, mask = expected["input_ids"], expected["attention_mask"]
    encoder = T5Text.from_pretrained(str(directory), dtype=jnp.bfloat16)
    # Every site the forward calls returns bfloat16: the tables read at their FP32 storage had kept
    # the residual stream FP32, which only beats the bfloat16 reference.
    _, state = encoder.transformer.apply(encoder.params, jnp.asarray(ids), jnp.asarray(mask),
                                         capture_intermediates=True, mutable=["intermediates"])
    sites = {name: site for name, site in state["intermediates"].items() if name != "__call__"}
    assert {"embed_tokens", "relative_attention_bias_0", "layers_0", "layers_1"} <= sites.keys()
    for name, site in sites.items():
        for output in jax.tree.leaves(site["__call__"]):
            assert output.dtype == jnp.bfloat16, name

    def transformers(dtype):
        with precision(dtype):
            model = T5EncoderModel.from_pretrained(directory, dtype=dtype).eval()

        def run(inputs):
            ids, mask = (torch.from_numpy(np.asarray(inputs[name], np.int64))
                         for name in ("input_ids", "attention_mask"))
            with precision(dtype), torch.no_grad():
                states = model(input_ids=ids, attention_mask=mask)
            states = states.last_hidden_state
            # bfloat16's values held in float32, which holds them exactly.
            return {"last_hidden_state": (states.float() if dtype == torch.bfloat16 else states).numpy()}
        return run

    @as_given
    def ours(inputs):
        return {"last_hidden_state": hidden(encoder, inputs["input_ids"], inputs["attention_mask"])}

    assert_as_exact(ours, transformers(torch.bfloat16), text_draw(encoder), "T5 bfloat16 hidden states",
                    seeds=SEEDS['T5 bfloat16 hidden states'], truth_fn=transformers(torch.float64))


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
