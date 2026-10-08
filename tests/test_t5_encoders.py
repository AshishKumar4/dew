"""The vendored T5 encoder tower against transformers' own.

The claim is parity: the same weights and the same token ids through the
reference T5EncoderModel and through dew have to produce the same last hidden
states. tools/t5_reference.py writes the fixtures under torch and
transformers, including a tiny random-weight checkpoint whose outputs are
committed, so the comparison runs in CI without a download.

Tolerances and the differences actually observed, fp32 on CPU:

- tiny checkpoint (gated-gelu, the v1.1 feed-forward): max |hidden state
  difference| 1.55e-06 (mean 2.37e-07, median 2.09e-07), tolerance 1e-4, on
  hidden states reaching 3.6. The two rearrangements against the reference
  cost nothing measurable: the query carries sqrt(head_dim) to cancel the
  kernel's 1/sqrt(head_dim), and the gate is `jax.nn.gelu(approximate=True)`,
  transformers' tanh `gelu_new`, not the erf one (4.7e-4 apart at |x| = 11,
  measured against `ACT2FN["gelu_new"]`).
- tiny UMT5 (three layers, a bias table each): max |hidden state difference|
  1.61e-06 (mean 2.49e-07, median 2.38e-07), tolerance 1e-4, on hidden states
  reaching 3.3.
- t5-small (plain relu): max |hidden state difference| 4.8e-07 (mean 8.2e-08,
  median 6.7e-08), tolerance 1e-3, on hidden states reaching 3.3.
"""

import json
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.inputs import T5Text

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "t5"
TINY = FIXTURES / "tiny"
TINY_UMT5 = FIXTURES / "tiny-umt5"
TOLERANCE = 1e-4


def reference(directory):
    return np.load(directory / "reference.npz")


def prompts(directory):
    return json.loads((directory / "prompts.json").read_text())["prompts"]


def largest_difference(actual, expected) -> float:
    return float(np.max(np.abs(np.asarray(actual, np.float32) - expected)))


def test_tiny_checkpoint_matches_the_reference():
    """The encoder read out of a full T5 checkpoint file, run on the
    reference's own token ids."""
    expected = reference(TINY)
    encoder = T5Text.from_pretrained(str(TINY))

    tokens = {"input_ids": expected["input_ids"],
              "attention_mask": expected["attention_mask"]}
    context = encoder.encode(encoder.params, tokens)

    difference = largest_difference(context.hidden, expected["last_hidden_state"])
    assert difference < TOLERANCE, f"max |hidden state difference| {difference:.3e}"
    assert np.array_equal(np.asarray(context.mask), expected["attention_mask"])


def test_a_umt5_checkpoint_matches_the_reference():
    """UMT5, Wan 2.1's text encoder: the same tower with a relative bias
    table in every layer, run on the reference's own ids and padding."""
    expected = reference(TINY_UMT5)
    encoder = T5Text.from_pretrained(str(TINY_UMT5))
    tokens = {"input_ids": expected["input_ids"], "attention_mask": expected["attention_mask"]}

    difference = largest_difference(encoder.encode(encoder.params, tokens).hidden,
                                    expected["last_hidden_state"])
    assert difference < TOLERANCE, f"max |hidden state difference| {difference:.3e}"


def test_each_umt5_layer_reads_its_own_bias():
    """T5's sharing, layer 0's table read by every layer, is not UMT5's: with
    the later layers' tables replaced by layer 0's, the outputs leave the
    reference."""
    expected = reference(TINY_UMT5)
    encoder = T5Text.from_pretrained(str(TINY_UMT5))
    tokens = {"input_ids": expected["input_ids"], "attention_mask": expected["attention_mask"]}
    tables = encoder.params["params"]
    shared = dict(tables)
    for name in ("relative_attention_bias_1", "relative_attention_bias_2"):
        shared[name] = tables["relative_attention_bias_0"]

    difference = largest_difference(encoder.encode({"params": shared}, tokens).hidden,
                                    expected["last_hidden_state"])
    assert difference > TOLERANCE, f"layer 0's table in every layer still matches: {difference:.3e}"


def test_the_encoder_tokenizes_and_captions():
    """The committed tokenizer turns the fixture prompts into the committed
    ids, and the ids back into the prompts."""
    encoder = T5Text.from_pretrained(str(TINY))
    expected = reference(TINY)

    tokens = encoder.tokenize(prompts(TINY))
    width = expected["input_ids"].shape[1]
    assert np.array_equal(tokens["input_ids"][:, :width], expected["input_ids"])
    assert np.array_equal(tokens["attention_mask"][:, :width], expected["attention_mask"])
    assert list(encoder.captions(tokens)) == prompts(TINY)


def test_the_json_fields_rebuild_an_encoder_that_agrees():
    """A run's record stores the encoder as its fields; rebuilding from them
    gives the same embeddings."""
    encoder = T5Text.from_pretrained(str(TINY))
    fields = encoder.to_json()


    rebuilt = T5Text.from_pretrained(**fields)
    tokens = encoder.tokenize(prompts(TINY)[:2])
    assert np.array_equal(np.asarray(rebuilt.encode(rebuilt.params, tokens).hidden),
                          np.asarray(encoder.encode(encoder.params, tokens).hidden))


@pytest.mark.parametrize("path", [
    ("embed_tokens", "embedding"),
    ("relative_attention_bias_0", "embedding"),
    ("layers_0", "self_attn", "q_proj", "kernel"),
    ("layers_0", "self_attn", "o_proj", "kernel"),
    ("layers_0", "input_layernorm", "scale"),
    ("layers_0", "mlp", "gate_proj", "kernel"),
    ("layers_0", "mlp", "up_proj", "kernel"),
    ("layers_0", "mlp", "down_proj", "kernel"),
    ("layers_1", "post_attention_layernorm", "scale"),
    ("final_layer_norm", "scale"),
], ids=lambda path: ".".join(path))
def test_every_translated_leaf_is_load_bearing(path):
    """One mutation per module of the tower: zeroing any leaf the translator
    places must move the outputs past the tolerance, or the parity test above
    proves nothing about that module. The relative bias table, the gate and
    both norms are each a leaf a name map can misplace without a shape error."""
    encoder = T5Text.from_pretrained(str(TINY))
    expected = reference(TINY)
    tokens = {"input_ids": expected["input_ids"],
              "attention_mask": expected["attention_mask"]}

    broken = jax.tree.map(lambda leaf: leaf, encoder.params)
    node = broken["params"]
    for entry in path[:-1]:
        node = node[entry]
    node[path[-1]] = jnp.zeros_like(node[path[-1]])

    difference = largest_difference(
        encoder.encode(broken, tokens).hidden, expected["last_hidden_state"])
    assert difference > TOLERANCE, f"zeroed {'.'.join(path)} still matches: {difference:.3e}"


def test_the_name_map_takes_the_encoder_and_refuses_the_rest():
    """The encoder's tied embedding is this tower whether a file stores it as
    shared.weight or encoder.embed_tokens.weight; the decoder and the lm_head
    translate to nothing. A name the map cannot explain raises ValueError, so
    a renamed upstream layout fails whole."""
    from dew.nn.text_encoders import translate_t5_weights

    input_embedding = np.arange(8, dtype=np.float32).reshape(4, 2)
    encoded = translate_t5_weights({
        "encoder.embed_tokens.weight": input_embedding,
        "decoder.block.0.layer.0.SelfAttention.q.weight": np.zeros((2, 2), np.float32),
        "lm_head.weight": np.zeros((4, 2), np.float32),
    })
    assert set(encoded) == {"embed_tokens"}
    np.testing.assert_array_equal(encoded["embed_tokens"]["embedding"], input_embedding)

    beside_the_embedding = {
        "shared.weight": input_embedding,
        "decoder.block.0.layer.1.DenseReluDense.wi.weight": np.zeros((2, 2), np.float32),
        "lm_head.weight": np.zeros((4, 2), np.float32),
    }
    encoded = translate_t5_weights(beside_the_embedding)
    assert set(encoded) == {"embed_tokens"}
    np.testing.assert_array_equal(encoded["embed_tokens"]["embedding"], input_embedding)

    for name in ("encoder.block.0.layer.0.SelfAttention.qkv.weight", "encoder.pooler.weight"):
        with pytest.raises(ValueError, match=re.escape(name)):
            translate_t5_weights({"shared.weight": input_embedding,
                                  name: np.zeros((2, 2), np.float32)})


@pytest.mark.network
def test_the_real_checkpoint_matches_the_reference():
    """t5-small's encoder, fp32, from the checkpoint's own safetensors,
    against what transformers computes for the same prompts. This is the v1.0
    relu feed-forward; the tiny fixture covers gated-gelu."""
    import torch
    from transformers import AutoTokenizer, T5EncoderModel

    repo = "google-t5/t5-small"
    tokenizer = AutoTokenizer.from_pretrained(repo)
    texts = ["a red bird", "", "two cats on a mat, painted"]
    tokens = tokenizer(texts, padding=True, return_tensors="pt")
    reference_model = T5EncoderModel.from_pretrained(repo, dtype=torch.float32)
    reference_model.eval()
    with torch.no_grad():
        expected = reference_model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"]).last_hidden_state.numpy()

    encoder = T5Text.from_pretrained(repo)
    got = encoder.encode(encoder.params, {
        "input_ids": tokens["input_ids"].numpy().astype(np.int32),
        "attention_mask": tokens["attention_mask"].numpy().astype(np.int32)}).hidden

    difference = largest_difference(got, expected)
    assert difference < 1e-3, f"max |hidden state difference| {difference:.3e}"
