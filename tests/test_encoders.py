"""The conditioning seam: an encoder's value reaches the model under one keyword.

The base class stays modality-agnostic: text and audio differ in both
tokenization and embedding, and adding a modality must not require touching
anything shared. The pooling test pins what the text mask is for: a padded
row moves nothing.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.inputs import Audio, CLIPText, Condition, ConditionEncoder, InputSpec, Field, unit_range
from dew.nn.dit import ConditioningEmbed, TextContext, masked_mean
from dew.registry import encoders

CLIP_TINY = Path(__file__).resolve().parent / "fixtures" / "clip" / "tiny"


def test_registry_exposes_both_modalities():
    assert encoders["clip_text"] is CLIPText and encoders.CLIPText is CLIPText
    assert encoders["audio"] is Audio


def test_base_class_is_modality_agnostic():
    """tokenize/encode must be abstract: a base class that assumes
    input_ids/attention_mask cannot serve audio."""
    for name in ("tokenize", "encode", "from_pretrained", "to_json"):
        assert name in ConditionEncoder.__abstractmethods__


@pytest.mark.parametrize("feature_key", ["input_values", "input_features"])
def test_audio_encoder_passes_extractor_keys_through(feature_key):
    """wav2vec2/HuBERT emit input_values, Whisper/AST emit input_features.
    The encoder must forward whatever the extractor produced, so swapping the
    audio model is a config change and nothing else."""
    seen = {}

    class StubExtractor:
        def __call__(self, audio, sampling_rate=None, padding=None, return_tensors=None):
            seen["sampling_rate"] = sampling_rate
            return {feature_key: np.zeros((1, 4), dtype=np.float32),
                    "attention_mask": np.ones((1, 4), dtype=np.int32)}

    def apply(params, features):
        seen["forwarded"] = sorted(features)
        return params["scale"] * jnp.ones((1, 4, 8), jnp.float32)

    encoder = Audio(checkpoint="stub", extractor=StubExtractor(), apply=apply,
                    params={"scale": jnp.asarray(2.0)}, sampling_rate=16000)
    embeddings = encoder.encode(encoder.params, encoder.tokenize(np.zeros(16000, np.float32)))

    assert seen["sampling_rate"] == 16000
    assert seen["forwarded"] == sorted([feature_key, "attention_mask"])
    assert embeddings.shape == (1, 4, 8) and float(embeddings[0, 0, 0]) == 2.0
    assert encoder.captions(None) == ()
    assert encoder.to_json() == {"checkpoint": "stub", "sampling_rate": 16000}


def test_the_audio_loader_says_what_it_cannot_do():
    """transformers 5 removed FlaxAutoModel, and a torch model would fail on
    the numpy arrays tokenize produces. The loader refuses up front, names the
    model, and says what would make it work; a run record that names an audio
    encoder takes the same path, so a logged config cannot half-build one."""
    with pytest.raises(NotImplementedError, match="vendor"):
        Audio.from_pretrained("facebook/wav2vec2-base-960h")
    with pytest.raises(NotImplementedError, match="facebook/wav2vec2-base-960h"):
        Condition.from_json({"encoder": {"name": "audio", "fields": {
            "checkpoint": "facebook/wav2vec2-base-960h", "sampling_rate": 16000}},
            "field": "audio", "unconditional": 0.0})


def test_unit_range_is_the_one_pixel_convention():
    pixels = jnp.array([0, 127, 128, 255], jnp.uint8)
    assert jnp.allclose(unit_range(pixels), jnp.array([-1.0, -0.5 / 127.5, 0.5 / 127.5, 1.0]))


def test_field_shape_is_a_tuple_of_ints():
    assert Field("image", [8, 8, 3]).shape == (8, 8, 3)
    assert InputSpec(Field("image", (8, 8, 3))).conditions == {}


############################################################################################################
# Text pooling reads the mask (T23)
############################################################################################################

def test_masked_mean_ignores_padded_rows():
    hidden = jnp.arange(2 * 4 * 3, dtype=jnp.float32).reshape(2, 4, 3)
    mask = jnp.array([[1, 1, 0, 0], [1, 1, 1, 1]])
    pooled = masked_mean(hidden, mask)
    assert jnp.allclose(pooled[0], jnp.mean(hidden[0, :2], axis=0))
    assert jnp.allclose(pooled[1], jnp.mean(hidden[1], axis=0))


def test_an_empty_mask_row_contributes_nothing(rng):
    """A row with no real tokens pools to exactly zero, the same vector the
    model gets with no text at all, rather than 0/0 = NaN."""
    embed = ConditioningEmbed(emb_features=16, mlp_ratio=1)
    hidden = jax.random.normal(rng, (2, 6, 8))
    mask = jnp.array([[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]])
    temb = jnp.array([10.0, 20.0])
    params = embed.init(rng, temb, TextContext(hidden, mask))
    pooled = embed.apply(params, temb, TextContext(hidden, mask))
    assert jnp.all(jnp.isfinite(pooled))
    assert jnp.allclose(pooled[0], embed.apply(params, temb, None)[0], atol=1e-6)


def test_padded_rows_do_not_move_the_conditioning_vector():
    """The tokenizer pads every prompt to 77 slots; what the padded rows hold
    must not reach the adaLN vector. Before the mask, they were averaged in
    and moved it."""
    embed = ConditioningEmbed(emb_features=16, mlp_ratio=1)
    key = jax.random.PRNGKey(0)
    hidden = jax.random.normal(jax.random.fold_in(key, 1), (2, 6, 8))
    mask = jnp.array([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]])
    temb = jnp.array([10.0, 20.0])
    params = embed.init(key, temb, TextContext(hidden, mask))

    baseline = embed.apply(params, temb, TextContext(hidden, mask))
    garbage = hidden.at[0, 3:].set(hidden[0, 3:] + 50.0)
    moved = embed.apply(params, temb, TextContext(garbage, mask))
    assert jnp.allclose(moved, baseline, atol=1e-5)

    # the same rows do count once the mask says they are real
    full = jnp.ones_like(mask)
    assert not jnp.allclose(embed.apply(params, temb, TextContext(garbage, full)),
                            embed.apply(params, temb, TextContext(hidden, full)), atol=1e-2)


def test_the_text_encoder_hands_the_model_its_mask():
    """The CLIP encoder's TextContext carries the tokenizer's mask, so a short
    prompt's padding is excluded from the pooling by construction."""
    encoder = CLIPText.from_pretrained(str(CLIP_TINY))
    tokens = encoder.tokenize(["a red bird", ""])
    context = encoder.encode(encoder.params, tokens)
    assert context.hidden.shape[:2] == context.mask.shape == (2, 77)
    # BOS, four word pieces, EOS for the prompt; BOS and EOS for the empty one
    assert int(context.mask[1].sum()) == 2
    assert int(context.mask[0].sum()) > 2
