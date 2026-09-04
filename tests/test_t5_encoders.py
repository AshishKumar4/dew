"""The vendored T5 encoder tower against transformers' own.

The claim is parity: the same weights and the same token ids through the
reference T5EncoderModel and through dew have to produce the same last hidden
states. tools/t5_reference.py writes the fixtures under torch and
transformers, including a tiny random-weight checkpoint whose outputs are
committed, so the comparison runs in CI without a download.

Tolerances and the differences actually observed, fp32 on CPU:

- tiny checkpoint (gated-gelu, the v1.1 feed-forward): max |hidden state
  difference| 1.4e-06 (mean 2.2e-07, median 1.8e-07), tolerance 1e-4, on
  hidden states reaching 3.6. The two rearrangements against the reference
  cost nothing measurable: the query carries sqrt(head_dim) to cancel the
  kernel's 1/sqrt(head_dim), and the gate is `jax.nn.gelu(approximate=True)`,
  transformers' tanh `gelu_new`, not the erf one (4.7e-4 apart at |x| = 11,
  measured against `ACT2FN["gelu_new"]`).
- t5-small (plain relu): max |hidden state difference| 4.8e-07 (mean 8.2e-08,
  median 6.7e-08), tolerance 1e-3, on hidden states reaching 3.3.
"""

import json
from pathlib import Path

import jax

import numpy as np
import pytest

from dew.inputs import Condition, Field, InputSpec, T5Text

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "t5"
TINY = FIXTURES / "tiny"
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
    assert set(fields) == {"checkpoint", "dtype", "max_length"}

    rebuilt = T5Text.from_pretrained(**fields)
    tokens = encoder.tokenize(prompts(TINY)[:2])
    assert np.array_equal(np.asarray(rebuilt.encode(rebuilt.params, tokens).hidden),
                          np.asarray(encoder.encode(encoder.params, tokens).hidden))


def test_an_encoder_without_relative_bias_fails_parity():
    """The relative position table is load-bearing: zeroing it must move the
    outputs past the tolerance, or the parity test above proves nothing."""
    import jax.numpy as jnp

    encoder = T5Text.from_pretrained(str(TINY))
    expected = reference(TINY)
    tokens = {"input_ids": expected["input_ids"],
              "attention_mask": expected["attention_mask"]}
    params = jax.tree.map(
        lambda leaf: jnp.zeros_like(leaf)
        if leaf.shape == encoder.params["params"]["layers_0"]["self_attn"]["rel_bias"]["embedding"].shape
        else leaf,
        encoder.params)
    difference = largest_difference(
        encoder.encode(params, tokens).hidden, expected["last_hidden_state"])
    assert difference > TOLERANCE, f"zeroed relative bias still matches: {difference:.3e}"


def test_an_encoder_without_its_gate_fails_parity():
    """The gated feed-forward's second projection is load-bearing too."""
    import jax
    import jax.numpy as jnp

    encoder = T5Text.from_pretrained(str(TINY))
    expected = reference(TINY)
    tokens = {"input_ids": expected["input_ids"],
              "attention_mask": expected["attention_mask"]}

    def zero_gate(tree):
        leaves, treedef = jax.tree.flatten(tree)
        return treedef.unflatten([
            jnp.zeros_like(leaf) if leaf.ndim == 2 and leaf.shape[1] == 128 else leaf
            for leaf in leaves])

    difference = largest_difference(
        encoder.encode(zero_gate(encoder.params), tokens).hidden,
        expected["last_hidden_state"])
    assert difference > TOLERANCE, f"zeroed gate still matches: {difference:.3e}"


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
