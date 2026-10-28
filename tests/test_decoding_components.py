"""Transforms and criteria against their reference implementations.

The oracles are the Transformers 5.16.1 processors and criteria in
`transformers/generation/logits_process.py` and `stopping_criteria.py`, run on
CPU in float32, one call per row over that row's unpadded token ids. Dew reads
a padded buffer with a validity mask instead, so every case here pads its rows
on different sides: a transform that read the buffer instead of the row's own
history disagrees immediately.

Frequency and presence penalties have no Transformers processor. Their oracle
is vLLM's own `apply_penalties` (`vllm/model_executor/layers/utils.py` at
v0.30.0), run as published by tools/vllm_penalty_reference.py.
"""

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from reference_error import assert_as_exact_as_the_reference
from token_support import byte_level_tokenizer
from transformers.generation.logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    MinPLogitsWarper,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SequenceBiasLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopHLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
)

from dew.sampling import decoding
from dew.sampling.decoding import StepState

VOCAB = 17
PROMPT = 5
BUDGET = 4


def rows(prompts, drawn, sides=("left", "right")):
    """A `StepState` whose rows hold their own unpadded history.

    Prompts pad on alternating sides and the draw slots above `step` stay
    invalid, so nothing but the row's real tokens may reach a transform.
    """
    count = len(prompts)
    tokens = np.zeros((count, PROMPT + BUDGET), np.int32)
    valid = np.zeros((count, PROMPT + BUDGET), bool)
    step = np.zeros(count, np.int32)
    for row, (prompt, drew) in enumerate(zip(prompts, drawn, strict=True)):
        start = PROMPT - len(prompt) if sides[row % len(sides)] == "left" else 0
        tokens[row, start:start + len(prompt)] = prompt
        valid[row, start:start + len(prompt)] = True
        tokens[row, PROMPT:PROMPT + len(drew)] = drew
        valid[row, PROMPT:PROMPT + len(drew)] = True
        step[row] = len(drew)
        # Slots the row has not drawn yet hold a token outside the history, so
        # a transform reading the buffer instead of the mask would penalize it.
        tokens[row, PROMPT + len(drew):] = VOCAB - 1 - row
    return StepState(tokens=jnp.asarray(tokens), valid=jnp.asarray(valid),
                     step=jnp.asarray(step), active=jnp.ones(count, bool),
                     keys=jax.random.split(jax.random.key(0), count), prompt_width=PROMPT)


def histories(prompts, drawn):
    return [list(prompt) + list(drew) for prompt, drew in zip(prompts, drawn, strict=True)]


def logits_of(count, seed=0):
    return jnp.asarray(np.asarray(jax.random.normal(jax.random.key(seed), (count, VOCAB)),
                                  np.float32))


def reference(processor, ids, logits):
    """One Transformers call per row, over that row's unpadded ids."""
    out = []
    for row, history in enumerate(ids):
        scores = torch.tensor(np.asarray(logits)[row:row + 1].copy())
        result = processor(torch.tensor([history], dtype=torch.long), scores)
        out.append(result.detach().numpy()[0])
    return np.stack(out)


def applied(native, state, logits):
    """The transform through the same composition `generate` compiles."""
    return np.asarray(jax.jit(decoding.chain((native,)))(state, logits))


def agrees(native, processor, prompts, drawn, seed=0, tolerance=1e-7):
    logits = logits_of(len(prompts), seed)
    got = applied(native, rows(prompts, drawn), logits)
    want = reference(processor, histories(prompts, drawn), logits)
    kept = np.isfinite(want)
    np.testing.assert_array_equal(np.isfinite(got), kept)
    largest = float(np.max(np.abs(np.where(kept, got, 0.0) - np.where(kept, want, 0.0))))
    assert largest <= tolerance, f"largest difference {largest:g} above {tolerance:g}"
    return largest


PROMPTS = [[3, 3, 9, 1], [7, 2, 5, 2, 7]]
DRAWN = [[9, 1, 3], [2]]


@pytest.mark.parametrize("value", [0.5, 1.3])
def test_temperature_matches_the_reference_warper(value):
    assert agrees(decoding.Temperature(value), TemperatureLogitsWarper(value),
                  PROMPTS, DRAWN, tolerance=1e-6) < 1e-6


@pytest.mark.parametrize("k", [1, 3, VOCAB + 5])
def test_top_k_matches_the_reference_warper(k):
    agrees(decoding.TopK(k), TopKLogitsWarper(k), PROMPTS, DRAWN)


@pytest.mark.parametrize("p", [0.1, 0.6, 0.95])
def test_top_p_matches_the_reference_warper(p):
    agrees(decoding.TopP(p), TopPLogitsWarper(p), PROMPTS, DRAWN)


@pytest.mark.parametrize("p", [0.05, 0.4, 0.9])
def test_min_p_matches_the_reference_warper(p):
    agrees(decoding.MinP(p), MinPLogitsWarper(p), PROMPTS, DRAWN)


@pytest.mark.parametrize("mass", [0.3, 0.9])
def test_typical_matches_the_reference_warper(mass):
    assert agrees(decoding.Typical(mass), TypicalLogitsWarper(mass), PROMPTS, DRAWN,
                  tolerance=1e-6) < 1e-6


@pytest.mark.parametrize("epsilon", [0.02, 0.09])
def test_epsilon_cutoff_matches_the_reference_warper(epsilon):
    agrees(decoding.EpsilonCutoff(epsilon), EpsilonLogitsWarper(epsilon), PROMPTS, DRAWN)


@pytest.mark.parametrize("epsilon", [0.02, 0.3])
def test_eta_cutoff_matches_the_reference_warper(epsilon):
    agrees(decoding.EtaCutoff(epsilon), EtaLogitsWarper(epsilon), PROMPTS, DRAWN)


@pytest.mark.parametrize("h", [0.2, 0.75])
def test_top_h_matches_the_reference_warper(h):
    agrees(decoding.TopH(h), TopHLogitsWarper(h), PROMPTS, DRAWN)


def test_renormalize_matches_the_reference_normalization():
    assert agrees(decoding.Renormalize(), LogitNormalization(), PROMPTS, DRAWN,
                  tolerance=1e-6) < 1e-6


def test_remove_invalid_values_matches_the_reference_processor():
    logits = jnp.asarray(np.array([[float("nan"), 1.0, float("inf")] + [0.5] * (VOCAB - 3),
                                   [-float("inf"), 2.0, 0.0] + [0.1] * (VOCAB - 3)], np.float32))
    state = rows(PROMPTS, DRAWN)
    got = (applied(decoding.RemoveInvalidValues(), state, logits))
    want = reference(InfNanRemoveLogitsProcessor(), histories(PROMPTS, DRAWN), logits)
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("penalty", [0.6, 1.8])
def test_repetition_penalty_reads_each_rows_own_history(penalty):
    """The reference penalizes every id in `input_ids`, which here is the
    row's prompt and its draws, and nothing the padding holds."""
    agrees(decoding.RepetitionPenalty(penalty), RepetitionPenaltyLogitsProcessor(penalty),
           PROMPTS, DRAWN)


@pytest.mark.parametrize("penalty", [0.7, 1.5])
def test_prompt_repetition_penalty_reads_only_the_prompt(penalty):
    logits = logits_of(len(PROMPTS))
    state = rows(PROMPTS, DRAWN)
    got = applied(decoding.PromptRepetitionPenalty(penalty), state, logits)
    want = np.stack([
        EncoderRepetitionPenaltyLogitsProcessor(
            penalty, torch.tensor([prompt], dtype=torch.long))(
                torch.tensor([prompt], dtype=torch.long),
                torch.tensor(np.asarray(logits)[row:row + 1].copy())).numpy()[0]
        for row, prompt in enumerate(PROMPTS)])
    # Each score one float32 product or quotient by the inverted penalty, the reference's own operation
    # on the same values, or left as it is: the same numbers bit for bit.
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("size", [2, 3])
def test_no_repeat_ngram_matches_the_reference_processor(size):
    prompts = [[3, 9, 1, 3, 9], [4, 4, 4, 4]]
    drawn = [[1, 3, 9], [4, 4]]
    agrees(decoding.NoRepeatNGram(size), NoRepeatNGramLogitsProcessor(size), prompts, drawn)


@pytest.mark.parametrize("size", [2, 3])
def test_prompt_no_repeat_ngram_matches_the_reference_encoder_processor(size):
    prompts = [[3, 9, 1, 3, 9], [4, 6, 4, 6]]
    drawn = [[1, 3, 9], [4, 6]]
    logits = logits_of(len(prompts))
    state = rows(prompts, drawn)
    got = applied(decoding.PromptNoRepeatNGram(size), state, logits)
    want = np.stack([
        EncoderNoRepeatNGramLogitsProcessor(size, torch.tensor([prompt], dtype=torch.long))(
            torch.tensor([history], dtype=torch.long),
            torch.tensor(np.asarray(logits)[row:row + 1].copy())).numpy()[0]
        for row, (prompt, history) in enumerate(zip(prompts, histories(prompts, drawn), strict=True))])
    np.testing.assert_array_equal(np.isfinite(got), np.isfinite(want))
    # Every score not banned is the logit itself, in both.
    kept = np.isfinite(want)
    np.testing.assert_array_equal(np.where(kept, got, 0.0), np.where(kept, want, 0.0))


def test_sequence_bias_matches_the_reference_over_mixed_lengths():
    entries = [([9], 2.5), ([1, 3], -1.5), ([2, 5, 2], 3.0), ([3, 3, 9, 1, 3], 1.0)]
    prompts = [[3, 3, 9, 1], [7, 2, 5, 2, 7]]
    drawn = [[3], [5, 2]]
    agrees(decoding.sequence_bias(entries),
           SequenceBiasLogitsProcessor([[list(ids), value] for ids, value in entries]),
           prompts, drawn, tolerance=1e-6)


@pytest.mark.parametrize("eos", [(4,), [4], range(4, 5), np.array([4]), np.int64(4)],
                         ids=["tuple", "list", "range", "array", "numpy-int"])
def test_bad_words_ban_sequences_and_keep_the_eos_reachable(eos):
    """The EOS ids are any sequence of ids or one id, a tokenizer's arrays among them."""
    words = [[9], [1, 3], [4]]
    prompts = [[3, 3, 9, 1], [7, 2, 5, 2, 7]]
    drawn = [[1], [5]]
    agrees(decoding.bad_words(words, eos_token_ids=eos),
           NoBadWordsLogitsProcessor(words, eos_token_id=torch.tensor([4])), prompts, drawn)


def test_empty_sequence_bias_leaves_every_logit_unchanged():
    state, logits = rows(PROMPTS, DRAWN), logits_of(len(PROMPTS))
    np.testing.assert_array_equal(applied(decoding.sequence_bias([]), state, logits), logits)


@pytest.mark.parametrize("length", [4, 8])
def test_min_length_counts_the_whole_unpadded_sequence(length):
    agrees(decoding.MinLength(length, jnp.asarray([2, 6], jnp.int32)),
           MinLengthLogitsProcessor(length, torch.tensor([2, 6])), PROMPTS, DRAWN)


@pytest.mark.parametrize("count", [1, 3])
def test_min_new_tokens_counts_only_the_drawn_tokens(count):
    logits = logits_of(len(PROMPTS))
    state = rows(PROMPTS, DRAWN)
    got = applied(decoding.MinNewTokens(count, jnp.asarray([2, 6], jnp.int32)), state, logits)
    want = np.stack([
        MinNewTokensLengthLogitsProcessor(len(prompt), count, torch.tensor([2, 6]))(
            torch.tensor([history], dtype=torch.long),
            torch.tensor(np.asarray(logits)[row:row + 1].copy())).numpy()[0]
        for row, (prompt, history) in enumerate(zip(PROMPTS, histories(PROMPTS, DRAWN), strict=True))])
    np.testing.assert_array_equal(np.isfinite(got), np.isfinite(want))


def test_forced_bos_only_fires_on_a_single_token_sequence():
    prompts = [[3], [7, 2]]
    drawn = [[], []]
    agrees(decoding.ForcedBOS(5), ForcedBOSTokenLogitsProcessor(5), prompts, drawn)


def test_forced_eos_fires_one_step_before_the_declared_length():
    prompts = [[3, 3, 9], [7, 2]]
    drawn = [[1, 4], [5]]
    agrees(decoding.ForcedEOS(6, max_length=6), ForcedEOSTokenLogitsProcessor(6, torch.tensor([6])),
           prompts, drawn)


def test_suppress_tokens_matches_the_reference_processor():
    agrees(decoding.SuppressTokens(jnp.asarray([2, 6, 11], jnp.int32)),
           SuppressTokensLogitsProcessor([2, 6, 11]), PROMPTS, DRAWN)


def test_begin_suppress_tokens_fires_at_the_first_drawn_token():
    prompts = [[3, 3, 9, 1], [7, 2, 5, 2, 7]]
    for step, drawn in enumerate(([[], []], [[1], [5]])):
        logits = logits_of(len(prompts))
        state = rows(prompts, drawn)
        got = applied(decoding.BeginSuppressTokens(jnp.asarray([2, 6], jnp.int32)), state, logits)
        want = np.stack([
            SuppressTokensAtBeginLogitsProcessor([2, 6], len(prompt))(
                torch.tensor([history], dtype=torch.long),
                torch.tensor(np.asarray(logits)[row:row + 1].copy())).numpy()[0]
            for row, (prompt, history) in enumerate(zip(prompts, histories(prompts, drawn), strict=True))])
        np.testing.assert_array_equal(np.isfinite(got), np.isfinite(want))
        assert np.isfinite(got[:, 2]).all() == (step == 1)


def test_begin_suppress_tokens_after_a_forced_bos_fires_where_each_rows_bos_ran():
    """With a forced BOS, transformers suppresses one token later only for a
    one-token prompt, the rows where the BOS is forced: a batch of a
    one-token prompt and a longer one suppresses the first at its second
    drawn token and the second at its first, as each row does alone."""
    prompts = [[3], [7, 2, 5]]
    for drawn in ([[], []], [[4], [5]]):
        logits = logits_of(len(prompts))
        got = applied(decoding.BeginSuppressTokens(jnp.asarray([2, 6], jnp.int32), forced_bos=True),
                      rows(prompts, drawn), logits)
        want = np.stack([
            SuppressTokensAtBeginLogitsProcessor([2, 6], len(prompt) + (len(prompt) == 1))(
                torch.tensor([history], dtype=torch.long),
                torch.tensor(np.asarray(logits)[row:row + 1].copy())).numpy()[0]
            for row, (prompt, history) in enumerate(zip(prompts, histories(prompts, drawn), strict=True))])
        np.testing.assert_array_equal(np.isfinite(got), np.isfinite(want))


def test_exponential_decay_grows_the_eos_score_after_its_start():
    prompts = [[3, 3, 9, 1], [7, 2, 5, 2, 7]]
    drawn = [[1, 3, 9], [2]]
    logits = logits_of(len(prompts))
    state = rows(prompts, drawn)
    native = decoding.ExponentialDecayLengthPenalty(1, 1.4, jnp.asarray([6], jnp.int32))
    got = applied(native, state, logits)
    want = np.stack([
        ExponentialDecayLengthPenalty((1, 1.4), torch.tensor([6]), len(prompt))(
            torch.tensor([history], dtype=torch.long),
            torch.tensor(np.asarray(logits)[row:row + 1].copy())).numpy()[0]
        for row, (prompt, history) in enumerate(zip(prompts, histories(prompts, drawn), strict=True))])
    np.testing.assert_allclose(got, want, atol=1e-5, rtol=0)
    # The first row is three tokens past the start and the second one is at it,
    # so a penalty that ignored the count would not separate them.
    assert got[0, 6] > logits[0, 6] and got[1, 6] == logits[1, 6]


PENALTIES = np.load(Path(__file__).resolve().parent / "fixtures" / "decoding" / "penalties.npz")


@pytest.mark.parametrize("case", range(len(PENALTIES["penalties"])))
def test_frequency_and_presence_penalties_are_vllms(case):
    """vLLM's own `apply_penalties` (v0.30.0, tools/vllm_penalty_reference.py)
    on four histories: the frequency penalty times each token's count among
    the generated tokens and the presence penalty times whether it was
    generated, the prompt counting for neither, both signs. Dew's two
    transforms, chained in vLLM's order, held to its float64 run by the
    float64 rule."""
    frequency, presence = (float(value) for value in PENALTIES["penalties"][case])
    prompts, drawn = ([[int(token) for token in row if token != VOCAB] for row in PENALTIES[name]]
                      for name in ("prompt", "output"))
    native = decoding.chain((decoding.FrequencyPenalty(frequency), decoding.PresencePenalty(presence)))
    got = np.asarray(jax.jit(native)(rows(prompts, drawn), jnp.asarray(PENALTIES["logits"])))
    assert_as_exact_as_the_reference(got, PENALTIES[f"case_{case}"], PENALTIES[f"case_{case}_f64"],
                                     f"penalties {frequency}, {presence}")


def stop_string_tokenizer(tmp_path):
    """A tokenizer whose decoder concatenates pieces, so a stop string can be
    spelled across token boundaries the way `StopStringCriteria` describes."""
    from tokenizers import Tokenizer, decoders, models
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    pieces = [
        "st",
        "op",
        "sto",
        "pper",
        "las",
        "topper",
        "s",
        "to",
        "pped",
        "stop",
        "at",
        "opera",
        "tion",
        "x",
        "yy",
        *list("abcdef"),
    ]
    vocabulary = {"<unk>": 0}
    vocabulary.update({piece: index for index, piece in enumerate(pieces, start=1)})
    backend = Tokenizer(models.BPE(vocabulary, [], unk_token="<unk>"))
    backend.decoder = decoders.Fuse()
    PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>").save_pretrained(
        tmp_path / "stops")
    return AutoTokenizer.from_pretrained(tmp_path / "stops", local_files_only=True)


class PublicVocabulary:
    """A tokenizer exposing only the public vocabulary operations the stop compiler reads."""

    def __init__(self, source):
        self.source = source

    def get_vocab(self):
        return self.source.get_vocab()

    def convert_tokens_to_string(self, tokens):
        return self.source.convert_tokens_to_string(tokens)

    def convert_ids_to_tokens(self, index):
        return self.source.convert_ids_to_tokens(index)

    def __call__(self, text, *, add_special_tokens):
        return self.source(text, add_special_tokens=add_special_tokens)


@pytest.mark.parametrize("surface", ["tokenizer", "wrapped", "public"])
def test_stop_strings_match_the_reference_criterion_over_random_rows(tmp_path, surface):
    """Both sides read the same vocabulary. Overhangs on either end, strings
    spelled across several tokens and several stop strings at once all have to
    agree, and a string finished before the last token must not stop the row.
    """
    from transformers.generation.stopping_criteria import StopStringCriteria

    tokenizer = stop_string_tokenizer(tmp_path)
    size = len(tokenizer)
    strings = ["stop", "operation"]
    source = tokenizer
    if surface == "wrapped":
        source = SimpleNamespace(reference=SimpleNamespace(tokenizer=SimpleNamespace(
            reference=SimpleNamespace(tokenizer=tokenizer))))
    elif surface == "public":
        source = PublicVocabulary(tokenizer)
    native = decoding.stop_strings(source, strings, size)
    oracle = StopStringCriteria(tokenizer, strings)
    checked = jax.jit(decoding.criterion((native,)))

    spelled = [["st", "op"], ["stop"], ["st", "opera"], ["sto", "pper"], ["las", "topper"],
               ["s", "to", "pped"], ["stop", "at"], ["st", "op", "at"], ["st", "opera", "tion"],
               ["x", "yy", "st", "op"], ["x", "opera", "tion"], ["yy", "x"]]
    sequences = [[tokenizer.convert_tokens_to_ids(piece) for piece in row] for row in spelled]
    generator = np.random.default_rng(0)
    ids = [index for index in range(1, size) if index != tokenizer.unk_token_id]
    sequences += [list(generator.choice(ids, size=int(generator.integers(1, 6))))
                  for _ in range(40)]

    for row in sequences:
        width = len(row)
        state = StepState(tokens=jnp.asarray([[0] * (8 - width) + list(row)], jnp.int32),
                          valid=jnp.asarray([[False] * (8 - width) + [True] * width]),
                          step=jnp.asarray([width], jnp.int32), active=jnp.ones(1, bool),
                          keys=jax.random.split(jax.random.key(0), 1), prompt_width=0)
        got = bool(checked(state, jnp.asarray([row[-1]], jnp.int32))[0])
        want = bool(oracle(torch.tensor([row], dtype=torch.long), None)[0])
        assert got == want, (tokenizer.convert_ids_to_tokens(row), got, want)


def test_stop_strings_refuse_a_cycle_of_tokenizer_wrappers():
    first = SimpleNamespace()
    second = SimpleNamespace(tokenizer=first)
    first.reference = second
    with pytest.raises(TypeError):
        decoding.stop_strings(first, "stop")


def test_stop_strings_refuse_a_vocabulary_that_cannot_spell_them(tmp_path):
    tokenizer = stop_string_tokenizer(tmp_path)
    with pytest.raises(ValueError):
        decoding.stop_strings(tokenizer, "zzzz", len(tokenizer))


def test_a_stop_string_split_across_byte_tokens_still_ends_the_row(tmp_path):
    """A byte-level vocabulary spells 'é' as two tokens, neither of which is
    valid text on its own. Reading the vocabulary as text loses them, so the
    match runs over the bytes each piece contributes, as the reference does.
    """
    from transformers.generation.stopping_criteria import StopStringCriteria

    from dew.sampling.vocabulary import matching_mode

    tokenizer = byte_level_tokenizer(tmp_path / "bytes")
    assert matching_mode(tokenizer) == "byte_level"
    pair = tokenizer.encode("é")


    native = decoding.stop_strings(tokenizer, ["é", "stop"], 256)
    oracle = StopStringCriteria(tokenizer, ["é", "stop"])
    checked = jax.jit(decoding.criterion((native,)))
    rows = [pair, [ord("a"), *pair], [pair[0]], [pair[1]], tokenizer.encode("stop"),
            tokenizer.encode("laststop"), tokenizer.encode("stopat"), tokenizer.encode("héllo")]
    for row in rows:
        width = len(row)
        state = StepState(tokens=jnp.asarray([[0] * (12 - width) + list(row)], jnp.int32),
                          valid=jnp.asarray([[False] * (12 - width) + [True] * width]),
                          step=jnp.asarray([width], jnp.int32), active=jnp.ones(1, bool),
                          keys=jax.random.split(jax.random.key(0), 1), prompt_width=0)
        got = bool(checked(state, jnp.asarray([row[-1]], jnp.int32))[0])
        want = bool(oracle(torch.tensor([row], dtype=torch.long), None)[0])
        assert got == want, (row, got, want)
