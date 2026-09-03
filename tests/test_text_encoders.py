"""The vendored CLIP text tower against transformers' own.

The claim is parity, not plausibility: the same weights and the same token ids
through the reference implementation and through dew have to produce the same
hidden states and the same pooled row. tools/clip_reference.py writes the
fixtures under torch and transformers, including a tiny random-weight
checkpoint whose outputs are committed, so the comparison runs in CI without a
download.

Tolerances and the differences actually observed, fp32 on CPU:

- tiny checkpoint: max |hidden state difference| 1.4e-06, max |pooled
  difference| 9.5e-07, tolerance 1e-4, on hidden states reaching 2.8.
- clip-vit-large-patch14: max |hidden state difference| 1.9e-04 (mean 1.0e-06,
  median 6.0e-07), max |pooled difference| 4.8e-06, tolerance 1e-3, on hidden
  states reaching 33.1.

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

from dew.inputs.encoders import CLIPTextEncoder
from dew.nn.text_encoders import (
    CLIPTextModel, CLIPTextTransformer, translate_config, translate_weights,
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


def test_from_modelname_loads_and_encodes():
    """The path that was dead: from_modelname on the jax backend built its model
    with FlaxCLIPTextModel, which transformers 5 removed, so every call raised
    ImportError. It loads the vendored tower now, tokenizes with AutoTokenizer,
    and the embeddings it returns are the reference's."""
    expected = reference(TINY)
    encoder = CLIPTextEncoder.from_modelname(modelname=str(TINY), backend="jax")

    tokens = encoder.tokenize(prompts(TINY))
    embeddings = np.asarray(encoder.encode_from_tokens(tokens), np.float32)

    assert np.array_equal(np.asarray(tokens["input_ids"], np.int32),
                          expected["input_ids"])
    difference = float(np.max(np.abs(embeddings - expected["last_hidden_state"])))
    assert difference < TOLERANCE, f"max |hidden state difference| {difference:.3e}"


def test_serialized_config_rebuilds_an_encoder_that_agrees():
    """Sampling restores an encoder from the run's config, so a round-trip has
    to come back as the same encoder, not just the same fields."""
    encoder = CLIPTextEncoder.from_modelname(modelname=str(TINY), backend="jax")
    config = encoder.serialize()
    assert config == {"modelname": str(TINY), "backend": "jax"}

    restored = CLIPTextEncoder.deserialize(config)
    texts = prompts(TINY)
    assert np.array_equal(np.asarray(restored(texts)), np.asarray(encoder(texts)))


def test_a_backend_that_cannot_run_is_refused():
    """The torch branch this replaced loaded a model that then raised on the
    first call: `tokenize` returns numpy arrays and transformers'
    `CLIPTextModel.forward` calls `input_ids.size()` on them. Refusing at
    construction says so, instead of failing a batch later."""
    with pytest.raises(ValueError, match="'jax' is the only one"):
        CLIPTextEncoder.from_modelname(modelname=str(TINY), backend="torch")


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


def test_a_config_asking_for_another_activation_is_refused():
    """quick-GELU is the only activation here, and a checkpoint trained with
    GELU would otherwise load into a model that computes something else."""
    config = fixture_config(TINY)
    config["text_config"]["hidden_act"] = "gelu"

    with pytest.raises(ValueError, match="quick-GELU"):
        translate_config(config)


def test_an_unfamiliar_tensor_name_is_refused():
    """Skipping a name nobody mapped is how a checkpoint loads with half its
    weights; the vision tower and the projection heads are skipped by name."""
    with pytest.raises(ValueError, match="text_model.encoder.layers.0.self_attn.qkv"):
        translate_weights({"text_model.encoder.layers.0.self_attn.qkv.weight":
                           np.zeros((3, 2), np.float32)})

    skipped = translate_weights({
        "vision_model.embeddings.class_embedding": np.zeros((2,), np.float32),
        "visual_projection.weight": np.zeros((2, 2), np.float32),
        "text_projection.weight": np.zeros((2, 2), np.float32),
        "logit_scale": np.zeros((), np.float32),
        "text_model.embeddings.position_ids": np.zeros((1, 77), np.int64),
    })
    assert skipped == {}


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
    encoder = CLIPTextEncoder.from_modelname(modelname=repo, backend="jax")

    tokens = encoder.tokenize(prompts(REAL))
    assert np.array_equal(np.asarray(tokens["input_ids"], np.int32),
                          expected["input_ids"])
    outputs = encoder.model(tokens["input_ids"], tokens["attention_mask"])

    hidden = float(np.max(np.abs(np.asarray(outputs.last_hidden_state, np.float32)
                                 - expected["last_hidden_state"])))
    pooled = float(np.max(np.abs(np.asarray(outputs.pooler_output, np.float32)
                                 - expected["pooler_output"])))
    assert hidden < REAL_TOLERANCE, f"max |hidden state difference| {hidden:.3e}"
    assert pooled < REAL_TOLERANCE, f"max |pooled difference| {pooled:.3e}"
