"""The vendored T5 encoder tower against transformers' own.

The claim is parity: the same weights and the same token ids through the
reference T5EncoderModel and through dew are on par from float64 (check 1).
tools/t5_reference.py writes the tiny checkpoints and records transformers'
encoders on them over check 1's seeds (`seeds`), so the comparison runs in CI
without torch or a download; t5-small, live, runs where the network does.
The two rearrangements against the reference: the query carries
sqrt(head_dim) to cancel the kernel's 1/sqrt(head_dim), and the gate is
`jax.nn.gelu(approximate=True)`, transformers' tanh `gelu_new`, not the erf
one.
"""

import json
from pathlib import Path

import jax
import numpy as np
import pytest
from reference_error import assert_as_exact, declared_seeds, indexed_recording, recording

from dew.inputs import T5Text

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "t5"
TINY = FIXTURES / "tiny"
TINY_UMT5 = FIXTURES / "tiny-umt5"


def reference(directory):
    return np.load(directory / "reference.npz")


def prompts(directory):
    return json.loads((directory / "prompts.json").read_text())["prompts"]


def t5_draw(seed: int, vocabulary: int) -> dict[str, np.ndarray]:
    """A recording's draw at `seed` (tools/t5_reference.py), which its test draws again: two rows of 24
    token ids, each a drawn length of ids ended by the end token and padded after it."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(1, 25, 2)
    ids = np.zeros((2, 24), np.int32)
    for row, length in enumerate(lengths):
        ids[row, :length - 1] = rng.integers(3, vocabulary, length - 1)
        ids[row, length - 1] = 1
    return {"input_ids": ids, "attention_mask": (np.arange(24) < lengths[:, None]).astype(np.int32)}


def assert_recorded_encoder(directory, name):
    """The checkpoint's encoder as exact as transformers' fp32 one over its recording's seeds (check
    1), both measured from transformers' run in float64, held through Dew's own float64 encoder to the
    recorded truth's sketch (`indexed_recording`)."""
    path = directory.parent / f"{directory.name}-seeds.npz"
    seeds = declared_seeds(path)[name]
    encoder = T5Text.from_pretrained(str(directory))
    with jax.enable_x64(new_val=True):
        wide = T5Text.from_pretrained(str(directory), dtype="float64", param_dtype="float64")
    wide_encode = jax.jit(lambda inputs: {"last_hidden_state": wide.encode(wide.params, inputs).hidden})
    draw, reference, truth = indexed_recording(
        recording(path), name, seeds, inputs_of=lambda seed: t5_draw(seed, encoder.tokenizer.vocab_size),
        truth_fn=wide_encode)
    encode = jax.jit(lambda inputs: {"last_hidden_state": encoder.encode(encoder.params, inputs).hidden})
    assert_as_exact(lambda index, inputs: encode(inputs), reference, draw, f"{name} last hidden states",
                    seeds=seeds, truth_fn=truth)


def test_tiny_checkpoint_matches_the_reference():
    """The encoder read out of a full T5 checkpoint file, on par with
    transformers' over the recorded seeds; the committed prompts' ids and mask
    are the reference's own."""
    expected = reference(TINY)
    encoder = T5Text.from_pretrained(str(TINY))
    tokens = {"input_ids": expected["input_ids"], "attention_mask": expected["attention_mask"]}
    assert np.array_equal(np.asarray(encoder.encode(encoder.params, tokens).mask), expected["attention_mask"])
    assert_recorded_encoder(TINY, "t5")


def test_a_umt5_checkpoint_matches_the_reference():
    """UMT5, Wan 2.1's text encoder: the same tower with a relative bias
    table in every layer, on par with transformers' over the recorded seeds."""
    assert_recorded_encoder(TINY_UMT5, "umt5")


def test_training_drops_units_and_one_key_drops_the_same_ones():
    """The config's dropout_rate reaches the tower, and with `train` the
    encoder drops units: two dropout keys give different states, one key the
    same states twice, and evaluation is the encoder's own forward."""
    from dew.nn.text_encoders import translate_t5_config

    config = json.loads((TINY / "config.json").read_text())
    tower = translate_t5_config({**config, "dropout_rate": 0.25}).value
    assert tower.dropout_rate == 0.25
    variables = T5Text.from_pretrained(str(TINY)).params
    expected = reference(TINY)

    def encoded(**kwargs):
        return np.asarray(tower.apply(variables, expected["input_ids"], expected["attention_mask"], **kwargs))

    dropped = encoded(train=True, rngs={"dropout": jax.random.key(1)})
    np.testing.assert_array_equal(dropped, encoded(train=True, rngs={"dropout": jax.random.key(1)}))
    assert not np.array_equal(dropped, encoded(train=True, rngs={"dropout": jax.random.key(2)}))
    # Evaluation drops nothing: it is the encoder's own forward, the one the recorded seeds hold.
    encoder = T5Text.from_pretrained(str(TINY))
    tokens = {"input_ids": expected["input_ids"], "attention_mask": expected["attention_mask"]}
    np.testing.assert_array_equal(encoded(), np.asarray(encoder.encode(encoder.params, tokens).hidden))


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


T5_SMALL_SEEDS = 8
"""t5-small's live site's declared draws (`seeds_for`, on its own reference)."""


@pytest.mark.network
def test_the_real_checkpoint_matches_the_reference():
    """t5-small's encoder, fp32, from the checkpoint's own safetensors, on par
    with transformers' fp32 over drawn ids (check 1), both measured from
    transformers' run in float64, built and run inside `torch_precision.float64`.
    This is the v1.0 relu feed-forward; the tiny fixture covers gated-gelu."""
    import torch
    from transformers import T5EncoderModel

    from tools.torch_precision import float64, float64_twin

    repo = "google-t5/t5-small"
    reference_model = T5EncoderModel.from_pretrained(repo, dtype=torch.float32).eval()
    wide = float64_twin(reference_model, lambda: T5EncoderModel.from_pretrained(repo, dtype=torch.float64))

    def ran(model, ids, mask):
        with torch.no_grad():
            ids, mask = torch.from_numpy(np.asarray(ids)).long(), torch.from_numpy(np.asarray(mask)).long()
            return {"last_hidden_state": model(input_ids=ids, attention_mask=mask).last_hidden_state.numpy()}

    def wide_ran(ids, mask):
        with float64():
            return ran(wide, ids, mask)

    encoder = T5Text.from_pretrained(repo)
    encode = jax.jit(lambda ids, mask: {"last_hidden_state": encoder.encode(
        encoder.params, {"input_ids": ids, "attention_mask": mask}).hidden})

    def draw(seed):
        drawn = t5_draw(seed, reference_model.config.vocab_size)
        return drawn["input_ids"], drawn["attention_mask"]

    assert_as_exact(encode, lambda ids, mask: ran(reference_model, ids, mask), draw,
                    "t5-small last hidden states", seeds=T5_SMALL_SEEDS, truth_fn=wide_ran)
