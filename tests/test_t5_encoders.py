"""The vendored T5 encoder tower against transformers' own.

The claim is parity: the same weights and the same token ids through the
reference T5EncoderModel and through dew give last hidden states as exact as
the reference's (check 1, tests/reference_error.py), each measured from the
reference in float64. tools/t5_reference.py writes the fixtures under torch and
transformers, a tiny random-weight checkpoint among them, so the comparison
runs in CI without a download. Two rearrangements against the reference: the
query carries sqrt(head_dim) to cancel the kernel's 1/sqrt(head_dim), and the
gate is `jax.nn.gelu(approximate=True)`, transformers' tanh `gelu_new`, not
the erf one.
"""

import copy
import json
from pathlib import Path

import jax
import numpy as np
import pytest
from reference_error import assert_as_exact

from dew.inputs import T5Text

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "t5"
TINY = FIXTURES / "tiny"
TINY_UMT5 = FIXTURES / "tiny-umt5"

SEEDS = {"tiny T5 encoder": 8, "tiny UMT5 encoder": 8, "t5-small encoder": 8}
"""Check 1's declared seeds for each site of the file, sized by `seeds_for` (on armada)."""


def reference(directory):
    return np.load(directory / "reference.npz")


def prompts(directory):
    return json.loads((directory / "prompts.json").read_text())["prompts"]


def assert_encodes_as_exact(encoder: T5Text, model, label: str) -> None:
    """Check 1 of `encoder`'s last hidden states against transformers' `model`, an encoder of the same
    weights, its truth `model` in float64 with every float32 pin widened. A draw is eight rows of 32
    token ids drawn over the vocabulary, each real up to a length drawn from one to all, padded by
    id 0."""
    import torch

    from tools.diffusers_wan_reference import float64

    model = model.float().eval()
    with float64():
        wide = copy.deepcopy(model).double()
    vocab = model.config.vocab_size

    def draw(seed):
        rng = np.random.default_rng(seed)
        lengths = rng.integers(1, 33, 8)
        mask = (np.arange(32)[None, :] < lengths[:, None]).astype(np.int32)
        return np.where(mask, rng.integers(1, vocab, (8, 32)), 0).astype(np.int32), mask

    @jax.jit
    def encoded(ids, mask):
        return encoder.encode(encoder.params, {"input_ids": ids, "attention_mask": mask}).hidden

    def run(module, ids, mask):
        with torch.no_grad():
            return module(input_ids=torch.from_numpy(np.asarray(ids)).long(),
                          attention_mask=torch.from_numpy(np.asarray(mask)).long()).last_hidden_state.numpy()

    def truth(ids, mask):
        with float64():
            return run(wide, ids, mask)

    assert_as_exact(encoded, lambda ids, mask: run(model, ids, mask), draw, label, seeds=SEEDS[label],
                    truth_fn=truth)


def test_tiny_checkpoint_matches_the_reference():
    """The encoder read out of a full T5 checkpoint file (gated-gelu, the v1.1 feed-forward), as exact
    as transformers' encoder of the same file, which the committed fixture's own ids and mask also
    pass through."""
    from transformers import T5EncoderModel

    expected = reference(TINY)
    encoder = T5Text.from_pretrained(str(TINY))
    context = encoder.encode(encoder.params, {"input_ids": expected["input_ids"],
                                              "attention_mask": expected["attention_mask"]})
    assert np.array_equal(np.asarray(context.mask), expected["attention_mask"])
    assert_encodes_as_exact(encoder, T5EncoderModel.from_pretrained(str(TINY)), "tiny T5 encoder")


def test_a_umt5_checkpoint_matches_the_reference():
    """UMT5, Wan 2.1's text encoder: the same tower with a relative bias
    table in every layer, as exact as transformers' UMT5 encoder of the same file."""
    from transformers import UMT5EncoderModel

    encoder = T5Text.from_pretrained(str(TINY_UMT5))
    assert_encodes_as_exact(encoder, UMT5EncoderModel.from_pretrained(str(TINY_UMT5)), "tiny UMT5 encoder")


def test_training_drops_units_and_one_key_drops_the_same_ones():
    """The config's dropout_rate reaches the tower, and with `train` the
    encoder drops units: two dropout keys give different states, one key the
    same states twice, and evaluation is the checkpoint's own tower's, bit for
    bit, whose parity with the reference check 1 holds."""
    from dew.nn.text_encoders import translate_t5_config

    config = json.loads((TINY / "config.json").read_text())
    tower = translate_t5_config({**config, "dropout_rate": 0.25}).value
    assert tower.dropout_rate == 0.25
    encoder = T5Text.from_pretrained(str(TINY))
    variables = encoder.params
    expected = reference(TINY)

    def encoded(**kwargs):
        return np.asarray(tower.apply(variables, expected["input_ids"], expected["attention_mask"], **kwargs))

    dropped = encoded(train=True, rngs={"dropout": jax.random.key(1)})
    np.testing.assert_array_equal(dropped, encoded(train=True, rngs={"dropout": jax.random.key(1)}))
    assert not np.array_equal(dropped, encoded(train=True, rngs={"dropout": jax.random.key(2)}))
    np.testing.assert_array_equal(
        encoded(), encoder.transformer.apply(variables, expected["input_ids"], expected["attention_mask"]))


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


@pytest.mark.network
def test_the_real_checkpoint_matches_the_reference():
    """t5-small's encoder, fp32, from the checkpoint's own safetensors, as
    exact as transformers' encoder of the same checkpoint. This is the v1.0
    relu feed-forward; the tiny fixture covers gated-gelu."""
    import torch
    from transformers import T5EncoderModel

    repo = "google-t5/t5-small"
    reference_model = T5EncoderModel.from_pretrained(repo, dtype=torch.float32)
    assert_encodes_as_exact(T5Text.from_pretrained(repo), reference_model, "t5-small encoder")
