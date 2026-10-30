"""Speculative decoding: the emitted law, and the cache the block leaves behind.

The acceptance and the token after it are transformers' `_speculative_sampling`
on fixed logits and the same uniforms (tools/speculative_reference.py). The
law is checked against fixed distributions rather than a model, so nothing
but the acceptance rule, the residual and the bonus decide the answer. The
cache is checked against greedy sampling on real models: with a zero
temperature the target's point mass wins every rejection, so a block has to
emit exactly the greedy walk, which it can only do if the state it replays
after a rejection is the state the accepted prefix would have left.
"""


import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model_support import decoder

from dew.sampling import Sampling, Speculative, decoding, generate

VOCAB = 13
def predictor(kind="attention", **overrides):
    """A tiny decoder that also carries prediction depths."""
    return decoder(kind).clone(**{"max_seq_len": 16, "num_nextn_predict_layers": 1, **overrides})


@pytest.mark.parametrize("kind", ["attention", "mla", "recurrent"])
def test_speculation_reproduces_the_greedy_walk_on_a_real_model(kind):
    """At zero temperature the target is a point mass, so every rejected draft
    is replaced by the target's own token and the whole run has to equal the
    greedy walk. It only can if the accepted prefix is replayed into the
    cache: a recurrent mixer keeps a running summary that no cursor rewinds,
    and reusing the draft's state would show up on the next token."""
    model = predictor(kind)
    prompts = jnp.asarray([[1, 2, 3], [7, 8, 9]], jnp.int32)
    params = model.init(jax.random.key(0), prompts)
    drawn = generate(model, params, prompts, 6, key=jax.random.key(1),
                     sampling=Sampling(temperature=0), strategy=Speculative(block=3))

    walked = np.asarray(prompts)
    for _ in range(6):
        scores = np.asarray(model.apply(params, jnp.asarray(walked))[:, -1])
        walked = np.concatenate([walked, scores.argmax(-1)[:, None].astype(np.int32)], axis=1)
    np.testing.assert_array_equal(np.asarray(drawn.tokens), walked)
    np.testing.assert_array_equal(np.asarray(drawn.lengths), 6)
    np.testing.assert_array_equal(np.asarray(drawn.behavior_log_probs), 0.0)
    for step in range(6):
        scores = jax.nn.log_softmax(model.apply(params, jnp.asarray(walked[:, :3 + step]))[:, -1])
        chosen = walked[:, 3 + step]
        np.testing.assert_allclose(drawn.raw_log_probs[:, step],
                                   np.asarray(scores)[np.arange(2), chosen], atol=3e-6, rtol=0)


def longrope_predictor(kind="attention"):
    """`predictor` whose rotary switches tables past its sixth position (LongRoPE), its cached keys
    stale beyond it (`cache_rebuild_position`)."""
    from dew.nn.rope import LongRopeScaling

    return predictor(kind, rope_scaling=LongRopeScaling(short_factor=(1.0, 1.0, 1.0, 1.0),
                                                        long_factor=(2.0, 3.0, 4.0, 5.0),
                                                        original_max_position_embeddings=6))


@pytest.mark.parametrize("kind", ["attention", "recurrent"])
def test_speculation_across_a_longrope_rebuild_reproduces_the_greedy_walk(kind):
    """A LongRoPE crossing rebuilds every cache the decode keeps (`decode_ops`), the prediction depth's
    with the trunk's, so speculating across it at zero temperature emits the greedy walk, each row's
    crossing inside a block."""
    model = longrope_predictor(kind)
    prompts = jnp.asarray([[1, 2, 3], [7, 8, 9]], jnp.int32)
    params = model.init(jax.random.key(0), prompts)
    walked, speculated = (generate(model, params, prompts, 6, key=jax.random.key(1),
                                   sampling=Sampling(temperature=0), strategy=strategy)
                          for strategy in (None, Speculative(block=3)))
    np.testing.assert_array_equal(np.asarray(speculated.tokens), np.asarray(walked.tokens))
    np.testing.assert_array_equal(np.asarray(speculated.lengths), 6)


def test_a_longrope_crossing_reseeds_the_prediction_depth_as_a_prefill_of_its_history_does():
    """A verified stretch that crosses the table change leaves the decoder state a prefill of the whole
    history leaves, bit for bit: the trunk's cache rebuilt at the long factors, the prediction depth's
    cache reseeded over the history, and the last state and logits; the rows are marked rebuilt."""
    from dew.nn.inputs import ModelInputs
    from dew.sampling.text import decode_ops, prefill_state

    model = longrope_predictor()
    prompts = jnp.asarray([[1, 2, 3], [7, 8, 9]], jnp.int32)
    params = model.init(jax.random.key(0), prompts)
    ops = decode_ops(model, params, 0, 1)
    state, _ = prefill_state(model, params, ModelInputs(prompts), ops)
    stretch = jnp.asarray([[4, 5, 6, 2], [3, 1, 2, 5]], jnp.int32)
    crossed, _, _ = ops.verify(state, stretch, jnp.ones(stretch.shape, bool))
    assert np.asarray(crossed.rebuilt).all()
    history = jnp.zeros((2, model.max_seq_len), jnp.int32).at[:, :7].set(
        jnp.concatenate([prompts, stretch], axis=1))
    real = jnp.arange(model.max_seq_len)[None, :] < 7
    fresh, _ = prefill_state(model, params, ModelInputs(history, {"attention_mask": real}), ops)
    jax.tree.map(np.testing.assert_array_equal, dataclasses.replace(crossed, rebuilt=None),
                 dataclasses.replace(fresh, rebuilt=None))


def test_an_eos_inside_a_block_ends_the_row_on_the_token_that_drew_it():
    """A block commits its tokens in order, so an EOS drawn at the second of
    three candidates is emitted with its likelihoods and every later slot of
    that row stays padding."""
    model = predictor()
    prompt = jnp.asarray([[1, 2, 3]], jnp.int32)
    params = model.init(jax.random.key(0), prompt)
    eos = 4

    def script(state, logits):
        """Force the EOS at the second drawn token, leaving the first alone."""
        forced = jnp.where(jnp.arange(VOCAB)[None, :] == eos, 0.0, -jnp.inf)
        return jnp.where((state.step == 1)[:, None], forced, logits)

    drawn = generate(model, params, prompt, 6, key=jax.random.key(1),
                     sampling=Sampling(temperature=0, eos_token_ids=eos, pad_token_id=0),
                     logits=(script, decoding.Greedy()), strategy=Speculative(block=3))
    assert int(drawn.lengths[0]) == 2 and bool(drawn.terminated[0])
    assert int(np.asarray(drawn.tokens)[0, 4]) == eos
    np.testing.assert_array_equal(np.asarray(drawn.tokens)[0, 5:], 0)
    np.testing.assert_array_equal(np.asarray(drawn.behavior_log_probs)[0, 2:], 0.0)
    np.testing.assert_array_equal(np.asarray(drawn.raw_log_probs)[0, 2:], 0.0)
    # The EOS is an emitted action, so it carries the model's own likelihood.
    scores = jax.nn.log_softmax(model.apply(
        params, jnp.asarray(np.asarray(drawn.tokens)[:, :4]))[:, -1])
    np.testing.assert_allclose(drawn.raw_log_probs[0, 1], scores[0, eos], atol=3e-6, rtol=0)
