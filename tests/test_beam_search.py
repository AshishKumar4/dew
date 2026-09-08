"""Beam search against searches written independently of the device loop.

Two oracles. Over a two-token budget with the width set to the vocabulary the
search is exhaustive, so the returned hypothesis has to be the best-scoring
sequence out of every one that exists, under whatever length normalization is
asked for. Over a longer budget with a narrow width the oracle is a plain host
beam search written from `_beam_search`'s rules in Transformers 5.16.1, run on
full forward passes with no cache, which also checks that branching the cache
rows keeps each beam's own prefix.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.sampling import Beam, Sampling, generate
from test_text_rollout_contract import decoder

VOCAB = 13
DEAD = -1.0e9


@pytest.fixture(scope="module")
def model():
    module = decoder()
    return module, module.init(jax.random.key(0), jnp.ones((1, 3), jnp.int32))


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
                grown, score = seq + [token], float(total + scores[index, token])
                if token in eos_ids or position + 1 == budget:
                    done.append((grown, score, token in eos_ids))
                else:
                    following.append((grown, score))
        live = following
    return done


def host_beam(model, params, prompt, budget, width, eos_ids, penalty, early=False):
    """`_beam_search`'s bookkeeping on the host, over uncached forward passes."""
    keep = max(2, 1 + len(eos_ids)) * width
    running, finished, open_ = [([], 0.0)], [], True
    for position in range(budget):
        scores = next_log_probs(model, params, [list(prompt) + seq for seq, _ in running])
        candidates = sorted(
            ((seq + [token], float(total + scores[index, token]), token)
             for index, (seq, total) in enumerate(running) for token in range(VOCAB)),
            key=lambda entry: -entry[1])[:keep]
        hits = [entry[2] in eos_ids or position + 1 == budget for entry in candidates]
        recording = open_ and not (early is True and len(finished) >= width)
        for slot, (entry, hit) in enumerate(zip(candidates, hits)):
            if recording and slot < width and hit:
                finished.append((entry[0], entry[1] / (position + 1) ** penalty,
                                 entry[2] in eos_ids, position + 1))
        finished = sorted(finished, key=lambda entry: -entry[1])[:width]
        running = [(entry[0], entry[1]) for entry, hit in zip(candidates, hits) if not hit][:width]
        if not running:
            break
        reach = budget if (early == "never" and penalty > 0) else position + 1
        worst = min(entry[1] for entry in finished) if len(finished) >= width else DEAD
        open_ = open_ and running[0][1] / reach ** penalty > worst
    return finished


def searched(model, params, prompt, budget, width, penalty=1.0, early=False, eos=None, n=1):
    return generate(model, params, jnp.asarray([prompt], jnp.int32), budget,
                    key=jax.random.key(0), sampling=Sampling(eos_id=eos, pad_id=0), n=n,
                    strategy=Beam(width=width, length_penalty=penalty, early_stopping=early,
                                  stop_ids=0 if eos is None else 1))


PROMPT = [1, 2, 3]


@pytest.mark.parametrize("penalty", [0.0, 1.0, 2.0])
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


@pytest.mark.parametrize("early", [False, True, "never"])
@pytest.mark.parametrize("penalty", [0.0, 1.0])
def test_a_narrow_search_matches_a_host_search_with_the_same_rules(model, early, penalty):
    """Four steps at width three, with an EOS that fires: every returned token,
    length and termination flag has to match a beam search written on the host
    over uncached forwards, which only agrees if the cache rows follow the
    beams they were selected from."""
    module, params = model
    eos = int(np.argsort(next_log_probs(module, params, [PROMPT])[0])[-3])
    found = searched(module, params, PROMPT, 4, 3, penalty=penalty, early=early, eos=eos, n=3)
    expected = host_beam(module, params, PROMPT, 4, 3, (eos,), penalty, early)
    assert len(expected) == 3
    for row, (tokens, _, terminated, length) in enumerate(expected):
        assert int(found.lengths[row]) == length
        np.testing.assert_array_equal(np.asarray(found.tokens)[row, 3:3 + length], tokens)
        assert bool(found.terminated[row]) == terminated


def test_the_returned_rows_are_prompt_major_and_carry_no_behaviour_probability(model):
    """`n` is the return count, not the width. The rows come back prompt by
    prompt, a selected path has no draw behind it so its behaviour likelihood
    is zero, and the raw likelihoods are the model's own for its tokens."""
    module, params = model
    prompts = jnp.asarray([[1, 2, 3], [7, 8, 9]], jnp.int32)
    found = generate(module, params, prompts, 3, key=jax.random.key(0),
                     sampling=Sampling(pad_id=0), strategy=Beam(width=4), n=2)
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
    with pytest.raises(ValueError, match="at most its width"):
        generate(module, params, jnp.asarray([[1, 2, 3]], jnp.int32), 2, key=jax.random.key(0),
                 strategy=Beam(width=2), n=3)
