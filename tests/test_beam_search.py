"""Beam search against transformers' own and against searches written apart
from the device loop.

transformers' `_beam_search` (tools/beam_worker.py, 5.16.1) on the
llama-tiny and qwen3-tiny checkpoints over drawn prompts: three prompts,
width three, three returned beams, six new tokens, an EOS that ends some
beams inside the budget, under four settings of the length penalty and
early stopping. Each draw's every ranking is decided by the selection margin
in Dew's float64 search (`Beam.margins`), so a rounding cannot choose its
beams.

One more oracle: over a two-token budget with the width set to the vocabulary the
search is exhaustive, so the returned hypothesis has to be the best-scoring
sequence out of every one that exists, under whatever length normalization is
asked for.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model_support import initialized_decoder
from reference_runs import assert_arms_as_recorded

from dew.interop import Pretrained
from dew.nn.inputs import ModelInputs
from dew.sampling import Beam, Sampling, decoding, generate

VOCAB = 13


@pytest.fixture(scope="module")
def model():
    return initialized_decoder(length=3)


def next_log_probs(model, params, rows):
    return np.asarray(jax.nn.log_softmax(
        model.apply(params, jnp.asarray(np.asarray(rows), jnp.int32))[:, -1].astype(jnp.float32)))


def every_sequence(model, params, prompt, budget, eos_ids=()):
    """Every sequence the model can produce, with its cumulative log probability."""
    live, done = [([], 0.0)], []
    for position in range(budget):
        scores = next_log_probs(model, params, [list(prompt) + seq for seq, _ in live])
        following = []
        for index, (seq, total) in enumerate(live):
            for token in range(VOCAB):
                grown, score = [*seq, token], float(total + scores[index, token])
                if token in eos_ids or position + 1 == budget:
                    done.append((grown, score, token in eos_ids))
                else:
                    following.append((grown, score))
        live = following
    return done


def searched(model, params, prompt, budget, width, penalty=1.0, early=False, eos=None, n=1):
    return generate(model, params, jnp.asarray([prompt], jnp.int32), budget,
                    key=jax.random.key(0), sampling=Sampling(eos_token_ids=eos, pad_token_id=0), n=n,
                    strategy=Beam(width=width, length_penalty=penalty, early_stopping=early,
                                  stop_ids=0 if eos is None else 1))


PROMPT = [1, 2, 3]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"


BEAM_SEEDS = {"llama-tiny": 32, "qwen3-tiny": 32}
"""Each checkpoint's declared seeds, sized by `seeds_for` paired over margin-cleared draws (`beam_draw`)."""
PROMPTS, LENGTH, NEW_TOKENS, WIDTH = 3, 6, 6, 3
SETTINGS = ((1.0, False), (0.0, True), (2.0, "never"), (-1.0, False))
"""(length_penalty, early_stopping) pairs, tools/beam_reference.py's: the three early-stopping rules, and a
penalty that rewards, ignores and punishes length."""


def _searched(pretrained, ids, eos: int, pad: int, penalty, early, *, margins: bool = False):
    return generate(pretrained.model, pretrained.variables, jnp.asarray(ids, jnp.int32), NEW_TOKENS,
                    key=jax.random.key(0), sampling=Sampling(eos_token_ids=eos, pad_token_id=pad), n=WIDTH,
                    strategy=Beam(width=WIDTH, length_penalty=penalty, early_stopping=early, stop_ids=1,
                                  margins=margins))


def _scored(found, penalty) -> np.ndarray:
    """Each returned beam's score: its summed log probability over its generated length raised to the
    penalty, as transformers' `sequences_scores`."""
    lengths = np.asarray(found.lengths)
    valid = np.arange(NEW_TOKENS)[None] < lengths[:, None]
    summed = np.sum(np.where(valid, np.asarray(found.raw_log_probs, np.float64), 0.0), -1)
    return (summed / lengths ** penalty).astype(np.float32)


def _special(name: str) -> tuple[int, int]:
    with np.load(FIXTURES / name / "beam.npz") as stored:
        return int(stored["eos"]), int(stored["pad"])


def beam_draw(name: str):
    """`draw(seed)` of checkpoint `name`: PROMPTS unpadded prompts of LENGTH tokens over its vocabulary but
    its EOS and pad, redrawn until Dew's float64 search decides every ranking of every setting by the
    selection margin (`Beam.margins`, `selection_margins.MARGIN`), so float32 searches return its beams."""
    from selection_margins import MARGIN, clearing

    vocabulary = json.loads((FIXTURES / name / "config.json").read_text())["vocab_size"]
    eos, pad = _special(name)
    tokens = np.asarray([token for token in range(vocabulary) if token not in (eos, pad)])
    with jax.enable_x64(new_val=True):
        truth = Pretrained.load(str(FIXTURES / name), dtype="float64", attention_impl="reference")

    def clears(drawn: dict) -> bool:
        # In float64 in any process: a recorder's draws run where JAX's float64 is off.
        with jax.enable_x64(new_val=True):
            for penalty, early in SETTINGS:
                found = _searched(truth, drawn["ids"], eos, pad, penalty, early, margins=True)
                margins = np.asarray(found.margins)
                if np.any(margins[np.isfinite(margins)] <= MARGIN):
                    return False
        return True

    def draw(seed: int) -> dict:
        rows = np.random.default_rng([53, seed]).choice(tokens, (PROMPTS, LENGTH))
        return {"ids": rows.astype(np.int32)}

    return lambda seed: clearing(draw, clears)(seed)


def beam_scores(name: str):
    """`dew(dtype, inputs)` of checkpoint `name`: its float32 search's returned beams' scores under each
    setting (`case_<i>`)."""
    pretrained = Pretrained.load(str(FIXTURES / name), dtype="float32", attention_impl="reference")
    eos, pad = _special(name)

    def dew(_, inputs) -> dict:
        return {f"case_{index}": _scored(_searched(pretrained, inputs["ids"], eos, pad, penalty, early),
                                        penalty) for index, (penalty, early) in enumerate(SETTINGS)}

    return dew


def pipeline_arms(name: str):
    """tools/venv_reference.py's `beam-<name>` site: its recording, draw, Dew run and leaves."""
    return (FIXTURES / name / "beam_recording.npz", beam_draw(name), beam_scores(name),
            tuple(f"case_{index}" for index in range(len(SETTINGS))))


@pytest.mark.parametrize("name", ["llama-tiny", "qwen3-tiny"])
def test_beams_are_transformers_beam_search(name):
    """Each returned beam's score (its summed log probability over its generated length raised to the
    penalty), Dew's float32 search on par with transformers' over drawn prompts whose every ranking the
    float64 search decides by the selection margin (check 1, the truth transformers' float64 search):
    under each of four settings of the penalty and early stopping, beams ending on EOS and beams running
    to the budget alike."""
    assert_arms_as_recorded(pipeline_arms(name), f"beam-{name}", BEAM_SEEDS[name], f"{name} beam scores",
                            host=True)


@pytest.mark.parametrize("penalty", [0.0])
def test_an_exhaustive_width_returns_the_best_scoring_sequence(model, penalty):
    """With the width set to the vocabulary and a two-token budget the search
    sees every sequence, so the hypothesis it returns is the best one under
    the normalization asked for, and a different penalty moves it."""
    module, params = model
    found = searched(module, params, PROMPT, 2, VOCAB, penalty=penalty)
    best = max(every_sequence(module, params, PROMPT, 2),
               key=lambda entry: entry[1] / 2 ** penalty)
    np.testing.assert_array_equal(np.asarray(found.tokens)[0, 3:], best[0])
    assert int(found.lengths[0]) == 2 and not bool(found.terminated[0])


@pytest.mark.parametrize("penalty", [0.0, 2.0])
def test_a_completed_hypothesis_competes_on_its_own_length(model, penalty):
    """An EOS can end a beam early. The completed set normalizes by the length
    each hypothesis actually reached, so a short completion wins under a
    penalty below one and loses under one above it."""
    module, params = model
    eos = int(np.argmax(next_log_probs(module, params, [PROMPT])[0]))
    found = searched(module, params, PROMPT, 2, VOCAB, penalty=penalty, eos=eos)
    everything = every_sequence(module, params, PROMPT, 2, eos_ids=(eos,))
    best = max(everything, key=lambda entry: entry[1] / len(entry[0]) ** penalty)
    length = int(found.lengths[0])
    np.testing.assert_array_equal(np.asarray(found.tokens)[0, 3:3 + length], best[0])
    assert length == len(best[0]) and bool(found.terminated[0]) == best[2]


def test_the_returned_rows_are_prompt_major_and_carry_no_behaviour_probability(model):
    """`n` is the return count, not the width. The rows come back prompt by
    prompt, a selected path has no draw behind it so its behaviour likelihood
    is zero, and the raw likelihoods are the model's own for its tokens."""
    module, params = model
    prompts = jnp.asarray([[1, 2, 3], [7, 8, 9]], jnp.int32)
    found = generate(module, params, prompts, 3, key=jax.random.key(0),
                     sampling=Sampling(pad_token_id=0), strategy=Beam(width=4), n=2)
    assert found.tokens.shape == (4, 6) and found.rows == 4
    np.testing.assert_array_equal(np.asarray(found.tokens)[:, :3],
                                  np.repeat(np.asarray(prompts), 2, axis=0))
    np.testing.assert_array_equal(np.asarray(found.behavior_log_probs), 0.0)
    for row in range(4):
        walked = list(np.asarray(prompts)[row // 2])
        for step in range(3):
            scores = next_log_probs(module, params, [walked])[0]
            token = int(np.asarray(found.tokens)[row, 3 + step])
            np.testing.assert_allclose(found.raw_log_probs[row, step], scores[token],
                                       atol=3e-6, rtol=0)
            walked.append(token)
    # The two hypotheses of one prompt are different sequences.
    assert not np.array_equal(np.asarray(found.tokens)[0], np.asarray(found.tokens)[1])


def test_asking_for_more_hypotheses_than_the_width_is_refused(model):
    module, params = model
    with pytest.raises(ValueError):
        generate(module, params, jnp.asarray([[1, 2, 3]], jnp.int32), 2, key=jax.random.key(0),
                 strategy=Beam(width=2), n=3)


@pytest.mark.parametrize(("penalty", "early"), [(1.0, False), (0.5, True), (2.0, "never")])
def test_a_search_reports_its_margins_without_changing_its_beams(model, penalty, early):
    """A search asked for its margins (`Beam.margins`) returns the beams, lengths, flags and likelihoods it
    returns unasked, which reports none; under a model that scores every token alike each ranking it makes
    ties, a margin of zero at every step it searches."""
    module, params = model

    def search(variables, *, margins):
        return generate(module, variables, jnp.asarray([PROMPT], jnp.int32), 3, key=jax.random.key(0),
                        sampling=Sampling(eos_token_ids=4, pad_token_id=0), n=2,
                        strategy=Beam(width=2, length_penalty=penalty, early_stopping=early, stop_ids=1,
                                      margins=margins))

    asked, unasked = search(params, margins=True), search(params, margins=False)
    assert unasked.margins is None
    for field in ("tokens", "lengths", "terminated", "raw_log_probs"):
        np.testing.assert_array_equal(np.asarray(getattr(asked, field)), np.asarray(getattr(unasked, field)))
    alike = np.asarray(search(jax.tree.map(jnp.zeros_like, params), margins=True).margins)
    assert np.all(alike[:, 0] == 0) and np.all((alike == 0) | np.isposinf(alike))


@pytest.mark.parametrize("name", ["llama-tiny", "qwen3-tiny"])
def test_a_renormalized_biased_search_is_transformers(name):
    """A sequence bias followed by a renormalization is what a source with
    both controls compiles to, and transformers appends the renormalization
    even for a search: its search with `sequence_bias` and
    `renormalize_logits` returns the beams, lengths and EOS flags Dew's
    returns under the same chain. The fixture checks that the bias and the
    renormalization each move a beam."""
    directory = FIXTURES / name
    pretrained = Pretrained.load(str(directory), dtype="float32", attention_impl="reference")
    with np.load(directory / "generate.npz") as stored:
        inputs = ModelInputs(jnp.asarray(stored["prompt"], jnp.int32),
                             {"attention_mask": jnp.asarray(stored["mask"], bool)})
        width = stored["prompt"].shape[1]
    with np.load(directory / "beam.npz") as stored:
        reference = {key: stored[key] for key in stored.files}
    entries = [(list(tokens), bias) for tokens, bias in json.loads(str(reference["shaped_bias"]))]
    penalty, early = json.loads(str(reference["shaped_setting"]))
    found = generate(pretrained.model, pretrained.variables, inputs, 6, key=jax.random.key(0),
                     sampling=Sampling(eos_token_ids=int(reference["eos"]),
                                       pad_token_id=int(reference["pad"])),
                     logits=(decoding.sequence_bias(entries), decoding.Renormalize()), n=3,
                     strategy=Beam(width=3, length_penalty=penalty, early_stopping=early, stop_ids=1))
    want = reference["shaped_tokens"]
    np.testing.assert_array_equal(np.asarray(found.tokens)[:, width:], want)
    ended = want == int(reference["eos"])
    np.testing.assert_array_equal(np.asarray(found.lengths), np.where(ended.any(-1), ended.argmax(-1) + 1, 6))
    np.testing.assert_array_equal(np.asarray(found.terminated), ended.any(-1))
