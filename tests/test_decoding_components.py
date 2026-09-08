"""Transforms and criteria against their reference implementations.

The oracles are the Transformers 5.16.1 processors and criteria in
`transformers/generation/logits_process.py` and `stopping_criteria.py`, run on
CPU in float32, one call per row over that row's unpadded token ids. Dew reads
a padded buffer with a validity mask instead, so every case here pads its rows
on different sides: a transform that read the buffer instead of the row's own
history disagrees immediately.

Frequency and presence penalties have no Transformers processor. Their oracle
is vLLM's formula in `vllm/model_executor/layers/utils.py`, applied here as
plain numpy.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from transformers.generation.logits_process import (
    EncoderNoRepeatNGramLogitsProcessor, EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper, EtaLogitsWarper, ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor, ForcedEOSTokenLogitsProcessor, InfNanRemoveLogitsProcessor,
    LogitNormalization, MinLengthLogitsProcessor, MinNewTokensLengthLogitsProcessor,
    MinPLogitsWarper, NoBadWordsLogitsProcessor, NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor, SequenceBiasLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor, SuppressTokensLogitsProcessor, TemperatureLogitsWarper,
    TopHLogitsWarper, TopKLogitsWarper, TopPLogitsWarper, TypicalLogitsWarper,
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
    for row, (prompt, drew) in enumerate(zip(prompts, drawn)):
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
    return [list(prompt) + list(drew) for prompt, drew in zip(prompts, drawn)]


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
    np.testing.assert_allclose(got, want, atol=1e-6, rtol=0)


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
        for row, (prompt, history) in enumerate(zip(prompts, histories(prompts, drawn)))])
    np.testing.assert_array_equal(np.isfinite(got), np.isfinite(want))
    np.testing.assert_allclose(np.where(np.isfinite(want), got, 0.0),
                               np.where(np.isfinite(want), want, 0.0), atol=1e-6, rtol=0)


def test_sequence_bias_matches_the_reference_over_mixed_lengths():
    entries = [([9], 2.5), ([1, 3], -1.5), ([2, 5, 2], 3.0), ([3, 3, 9, 1, 3], 1.0)]
    prompts = [[3, 3, 9, 1], [7, 2, 5, 2, 7]]
    drawn = [[3], [5, 2]]
    agrees(decoding.sequence_bias(entries),
           SequenceBiasLogitsProcessor([[list(ids), value] for ids, value in entries]),
           prompts, drawn, tolerance=1e-6)


def test_bad_words_ban_sequences_and_keep_the_eos_reachable():
    words = [[9], [1, 3], [4]]
    prompts = [[3, 3, 9, 1], [7, 2, 5, 2, 7]]
    drawn = [[1], [5]]
    agrees(decoding.bad_words(words, eos_id=(4,)),
           NoBadWordsLogitsProcessor(words, eos_token_id=torch.tensor([4])), prompts, drawn)


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
        for row, (prompt, history) in enumerate(zip(PROMPTS, histories(PROMPTS, DRAWN)))])
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
            for row, (prompt, history) in enumerate(zip(prompts, histories(prompts, drawn)))])
        np.testing.assert_array_equal(np.isfinite(got), np.isfinite(want))
        assert np.isfinite(got[:, 2]).all() == (step == 1)


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
        for row, (prompt, history) in enumerate(zip(prompts, histories(prompts, drawn)))])
    np.testing.assert_allclose(got, want, atol=1e-5, rtol=0)
    # The first row is three tokens past the start and the second one is at it,
    # so a penalty that ignored the count would not separate them.
    assert got[0, 6] > logits[0, 6] and got[1, 6] == pytest.approx(float(logits[1, 6]), abs=1e-6)


def test_frequency_and_presence_penalties_follow_the_vllm_formula():
    """vllm/model_executor/layers/utils.py subtracts the penalty times the
    count of each generated token, and the presence penalty times its mask.
    Prompt tokens are not counted, which is what separates the two rows."""
    prompts = [[3, 3, 9, 1], [7, 2, 5, 2, 7]]
    drawn = [[9, 9, 1], [2]]
    logits = logits_of(len(prompts))
    state = rows(prompts, drawn)
    counts = np.zeros((2, VOCAB), np.float32)
    for row, drew in enumerate(drawn):
        for token in drew:
            counts[row, token] += 1
    frequency = applied(decoding.FrequencyPenalty(0.7), state, logits)
    presence = applied(decoding.PresencePenalty(0.7), state, logits)
    np.testing.assert_allclose(frequency, np.asarray(logits) - 0.7 * counts, atol=1e-6, rtol=0)
    np.testing.assert_allclose(presence, np.asarray(logits) - 0.7 * (counts > 0), atol=1e-6, rtol=0)
    assert frequency[0, 9] < presence[0, 9]


def test_a_history_transform_ignores_the_slots_a_row_has_not_drawn():
    """The buffer above `step` holds a token on purpose. A transform that read
    the buffer instead of the validity mask would penalize it."""
    state = rows([[3, 3, 9, 1]], [[9]])
    logits = logits_of(1)
    penalized = applied(decoding.RepetitionPenalty(2.0), state, logits)
    future = int(np.asarray(state.tokens)[0, PROMPT + 1])
    assert future not in (3, 9, 1)
    assert penalized[0, future] == pytest.approx(float(logits[0, future]))
    assert penalized[0, 9] != pytest.approx(float(logits[0, 9]))


def stop_string_tokenizer(tmp_path):
    """A tokenizer whose decoder concatenates pieces, so a stop string can be
    spelled across token boundaries the way `StopStringCriteria` describes."""
    from tokenizers import Tokenizer, decoders, models
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    pieces = ["st", "op", "sto", "pper", "las", "topper", "s", "to", "pped", "stop",
              "at", "opera", "tion", "x", "yy"] + list("abcdef")
    vocabulary = {"<unk>": 0}
    vocabulary.update({piece: index for index, piece in enumerate(pieces, start=1)})
    backend = Tokenizer(models.BPE(vocabulary, [], unk_token="<unk>"))
    backend.decoder = decoders.Fuse()
    PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>").save_pretrained(
        tmp_path / "stops")
    return AutoTokenizer.from_pretrained(tmp_path / "stops", local_files_only=True)


def test_stop_strings_match_the_reference_criterion_over_random_rows(tmp_path):
    """Both sides read the same vocabulary. Overhangs on either end, strings
    spelled across several tokens and several stop strings at once all have to
    agree, and a string finished before the last token must not stop the row.
    """
    from transformers.generation.stopping_criteria import StopStringCriteria

    tokenizer = stop_string_tokenizer(tmp_path)
    size = len(tokenizer)
    strings = ["stop", "operation"]
    native = decoding.stop_strings(tokenizer, strings, size)
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


def test_stop_strings_refuse_a_vocabulary_that_cannot_spell_them(tmp_path):
    tokenizer = stop_string_tokenizer(tmp_path)
    with pytest.raises(ValueError, match="no token in the vocabulary"):
        decoding.stop_strings(tokenizer, "zzzz", len(tokenizer))


def test_end_of_sequence_and_length_criteria_read_what_they_name():
    state = rows(PROMPTS, DRAWN)
    eos = decoding.EndOfSequence(jnp.asarray([4, 6], jnp.int32))
    np.testing.assert_array_equal(np.asarray(eos(state, jnp.asarray([6, 5], jnp.int32))),
                                  [True, False])
    # Row zero drew three tokens over a four-token prompt, row one drew one
    # over five, so a criterion counting the wrong thing separates them.
    np.testing.assert_array_equal(
        np.asarray(decoding.MaxNewTokens(3)(state, jnp.zeros(2, jnp.int32))), [True, False])
    np.testing.assert_array_equal(
        np.asarray(decoding.MaxLength(7)(state, jnp.zeros(2, jnp.int32))), [True, False])


def byte_level_tokenizer(tmp_path):
    """A byte-level BPE over all 256 bytes, so a code point splits in two."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    alphabet = {byte: char for char, byte in decoding.byte_alphabet().items()}
    backend = Tokenizer(models.BPE({alphabet[byte]: byte for byte in range(256)}, [],
                                   unk_token=None))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    PreTrainedTokenizerFast(tokenizer_object=backend).save_pretrained(tmp_path / "bytes")
    return AutoTokenizer.from_pretrained(tmp_path / "bytes", local_files_only=True)


def test_a_stop_string_split_across_byte_tokens_still_ends_the_row(tmp_path):
    """A byte-level vocabulary spells 'é' as two tokens, neither of which is
    valid text on its own. Reading the vocabulary as text loses them, so the
    match runs over the bytes each piece contributes, as the reference does.
    """
    from transformers.generation.stopping_criteria import StopStringCriteria

    tokenizer = byte_level_tokenizer(tmp_path)
    assert decoding.matching_mode(tokenizer) == "byte_level"
    pair = tokenizer.encode("é")
    assert len(pair) == 2 and tokenizer.decode(pair) == "é"

    native = decoding.stop_strings(tokenizer, ["é", "stop"], 256)
    oracle = StopStringCriteria(tokenizer, ["é", "stop"])
    checked = jax.jit(decoding.criterion((native,)))
    rows = [pair, [ord("a")] + pair, [pair[0]], [pair[1]], tokenizer.encode("stop"),
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
