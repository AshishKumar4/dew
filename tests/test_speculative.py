"""Speculative decoding: the emitted law, and the cache the block leaves behind.

The law is checked against fixed distributions rather than a model, so nothing
but the acceptance rule, the residual and the bonus decide the answer. The
cache is checked against greedy sampling on real models: with a zero
temperature the target's point mass wins every rejection, so a block has to
emit exactly the greedy walk, which it can only do if the state it replays
after a rejection is the state the accepted prefix would have left.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import checkify

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.sampling import Sampling, Speculative, generate
from dew.sampling import decoding
from dew.sampling.decoding import StepState, chain, criterion
from dew.sampling.strategies import DecodeOps, DecoderState
from test_text_rollout_contract import decoder

VOCAB = 13
HIDDEN = 2


def predictor(kind="attention", **overrides):
    """A tiny decoder that also carries prediction depths."""
    fields = {"num_nextn_predict_layers": 1, **overrides}
    return CausalTransformer(vocab_size=VOCAB, emb_features=16, num_layers=1, num_heads=2,
                             head_dim=8, mlp_features=32, max_seq_len=16, dtype="float32",
                             mixer=decoder(kind).mixer, **fields)


def scripted(target, drafts, rows):
    """Decode operations whose target and draft distributions are fixed.

    `drafts` holds one distribution per chained draft step, so a block can be
    made to agree on its first candidates and disagree later.
    """
    calls = {"index": 0}

    def verify(state, tokens, valid):
        width = tokens.shape[1]
        return (state, jnp.broadcast_to(target, (rows, width, VOCAB)),
                jnp.zeros((rows, width, HIDDEN), jnp.float32))

    def propose(state, states, tokens, embeds, valid, positions, depth):
        width = valid.shape[1]
        scores = drafts[calls["index"] % len(drafts)]
        calls["index"] += width
        return (state, jnp.broadcast_to(scores, (rows, width, VOCAB)),
                jnp.zeros((rows, width, HIDDEN), jnp.float32))

    return DecodeOps(advance=lambda state, token, active: state,
                     reindex=lambda state, rows_: state, verify=verify, propose=propose,
                     embed=lambda tokens: jnp.zeros((rows, tokens.shape[1], HIDDEN), jnp.float32),
                     depths=1)


def run(target, drafts, rows, budget, block, seed=0, stopping=(), transforms=()):
    """One speculative run over fixed distributions."""
    return run_with(Speculative(block=block), target, drafts, rows, budget, seed, stopping,
                    transforms)


def run_with(plan, target, drafts, rows, budget, seed=0, stopping=(), transforms=()):
    """One run of a given speculative plan over fixed distributions."""
    state = DecoderState(cache={}, logits=jnp.broadcast_to(target, (rows, VOCAB)),
                         positions=None, hidden=jnp.zeros((rows, HIDDEN), jnp.float32),
                         drafts=(jnp.zeros((rows, HIDDEN), jnp.float32),))
    start = StepState(tokens=jnp.zeros((rows, 1 + budget), jnp.int32),
                      valid=jnp.concatenate([jnp.ones((rows, 1), bool),
                                             jnp.zeros((rows, budget), bool)], axis=1),
                      step=jnp.zeros(rows, jnp.int32), active=jnp.ones(rows, bool),
                      keys=jax.random.split(jax.random.key(seed), rows), prompt_width=1)
    ops = scripted(target, drafts, rows)

    def body(carried, opening):
        return plan(carried, opening, ops, chain(transforms), criterion(stopping), budget, 1)

    failure, drawn = checkify.checkify(body, errors=checkify.user_checks)(state, start)
    failure.throw()
    return drawn


def point(token):
    return jnp.where(jnp.arange(VOCAB) == token, 0.0, -jnp.inf)


def spread(weights):
    full = np.full(VOCAB, -np.inf, np.float32)
    for token, weight in weights.items():
        full[token] = np.log(weight)
    return jnp.asarray(full)


def test_the_emitted_tokens_follow_the_target_distribution():
    """The point of the law: whatever the draft proposes, the tokens that come
    out are distributed as the target. Four thousand rows over four positions
    give sixteen thousand samples, whose standard error at a probability of a
    quarter is 0.0034, so 0.015 is a four-sigma band."""
    target = spread({0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1})
    drafts = [spread({0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4})]
    drawn = run(target, drafts, 4096, 4, 2)
    tokens = np.asarray(drawn.tokens)[np.asarray(drawn.valid)]
    counts = np.bincount(tokens, minlength=VOCAB)[:4] / tokens.size
    largest = float(np.max(np.abs(counts - np.array([0.4, 0.3, 0.2, 0.1]))))
    assert largest < 0.015, f"largest deviation {largest:g}"
    # The draft is the reverse distribution, so emitting it would be obvious.
    assert counts[0] > counts[3]


def test_a_rejected_first_candidate_emits_the_residual_of_the_target():
    """The target is a point mass the draft never proposes, so the second
    candidate is always rejected and the replacement is the normalized
    positive part of `p - q`, which here is the target itself."""
    drawn = run(point(5), [point(7)], 4, 6, 2)
    tokens, valid = np.asarray(drawn.tokens), np.asarray(drawn.valid)
    np.testing.assert_array_equal(valid.sum(axis=1), 6)
    # Every emitted token is the target's, never the draft's.
    np.testing.assert_array_equal(tokens[valid], 5)
    np.testing.assert_allclose(np.asarray(drawn.behavior_log_probs)[valid], 0.0, atol=1e-6)


def test_a_rejection_in_the_middle_keeps_the_accepted_prefix():
    """The first chained draft agrees with the target and the second does not,
    so the block emits the accepted candidate, then the replacement, and
    nothing the rejected draft proposed."""
    drawn = run(point(5), [point(5), point(7)], 4, 3, 3)
    tokens, valid = np.asarray(drawn.tokens), np.asarray(drawn.valid)
    np.testing.assert_array_equal(valid.sum(axis=1), 3)
    np.testing.assert_array_equal(tokens[valid], 5)


def test_a_matching_draft_is_accepted_and_the_block_draws_a_bonus():
    """With `q` equal to `p` nothing is rejected, so a block of `g` candidates
    emits `g + 1` tokens and the budget is reached in fewer blocks."""
    for block in (2, 4):
        drawn = run(point(5), [point(5)] * (block - 1), 2, block + 1, block)
        np.testing.assert_array_equal(np.asarray(drawn.valid).sum(axis=1), block + 1)
        np.testing.assert_array_equal(np.asarray(drawn.tokens)[np.asarray(drawn.valid)], 5)


def test_a_draft_outside_the_target_support_never_survives():
    """A candidate the target cannot produce is rejected with probability one,
    and the residual still covers the target's own support."""
    target = spread({1: 0.5, 2: 0.5})
    drawn = run(target, [point(9)], 2048, 2, 2)
    tokens = np.asarray(drawn.tokens)[np.asarray(drawn.valid)]
    assert set(np.unique(tokens).tolist()) == {1, 2}
    share = float(np.mean(tokens == 1))
    assert abs(share - 0.5) < 0.02, f"share {share:g}"


def test_a_criterion_inside_a_block_truncates_it():
    """A stopping criterion is applied after every committed token, so a block
    that produces one stops there instead of emitting the rest."""
    stop = jax.tree_util.Partial(lambda state, tokens: state.step >= 2)
    drawn = run(point(5), [point(5)] * 3, 2, 8, 4)
    assert int(np.asarray(drawn.valid).sum(axis=1)[0]) == 8
    stopped = run(point(5), [point(5)] * 3, 2, 8, 4, stopping=(stop,))
    np.testing.assert_array_equal(np.asarray(stopped.valid).sum(axis=1), 2)
    np.testing.assert_array_equal(np.asarray(stopped.terminated), True)


def test_a_stop_on_the_last_candidate_of_a_whole_block_draws_no_bonus():
    """A block whose every candidate is accepted draws a bonus token from the
    target unless a criterion ended the row on one of them. The last candidate
    gets that check like the others: a chain that leaves nothing to draw
    after the stop is never asked for a bonus, exactly as `Sample` would not
    ask for a third token."""
    stop = jax.tree_util.Partial(lambda state, tokens: state.step >= 2)
    nothing_after = jax.tree_util.Partial(
        lambda state, logits: jnp.where((state.step < 2)[:, None], logits, -jnp.inf))
    drawn = run(point(1), [point(1)] * 3, 2, 4, 2, stopping=(stop,), transforms=(nothing_after,))
    np.testing.assert_array_equal(np.asarray(drawn.valid).sum(axis=1), 2)
    np.testing.assert_array_equal(np.asarray(drawn.terminated), True)
    np.testing.assert_array_equal(np.asarray(drawn.tokens)[:, :2], 1)


def test_the_budget_bounds_the_last_block():
    """A block that would emit past the budget stops at it, and the emitted
    count is exactly the budget even when it is not a multiple of the block."""
    for budget in (3, 5, 7):
        drawn = run(point(5), [point(5)] * 3, 2, budget, 4)
        np.testing.assert_array_equal(np.asarray(drawn.valid).sum(axis=1), budget)
        np.testing.assert_array_equal(np.asarray(drawn.valid)[:, :budget], True)


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
                     sampling=Sampling(temperature=0, eos_id=eos, pad_id=0),
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


def test_a_rejected_prefix_leaves_the_prediction_cache_teacher_forced():
    """The cache a block leaves has to describe the tokens that were emitted,
    not the ones the draft proposed. Writing an emitted history the draft
    never suggested and then drafting from it has to match a teacher-forced
    pass over that realized sequence, which comparing greedy target tokens
    cannot show, because the target picks the same token whatever the
    proposer did."""
    from dew.nn.inputs import ModelInputs
    from dew.sampling.strategies import reseed
    from dew.sampling.text import _operations, _prefill

    model = predictor()
    prompt = jnp.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], jnp.int32)
    emitted = jnp.asarray([[11, 3], [2, 9]], jnp.int32)
    realized = jnp.concatenate([prompt, emitted], axis=1)
    width = realized.shape[1]
    ops = _operations(model, params_of(model, prompt), 0, 1)
    params = params_of(model, prompt)

    ahead = model.apply(params, realized, method=model.hidden_states)
    reference = model.apply(params, ahead[:, :-1], realized[:, 1:], depth=0,
                            positions=jnp.broadcast_to(jnp.arange(1, width)[None, :], (2, width - 1)),
                            method=model.mtp_step)[0]

    state, _ = _prefill(model, params, ModelInputs(prompt), ops)
    state, _, _ = reseed(
        ops, state, (state.hidden,), ahead[:, prompt.shape[1]:width],
        model.apply(params, emitted, method=model.token_embeddings),
        jnp.ones((2, 2), bool),
        jnp.broadcast_to(jnp.arange(prompt.shape[1], width)[None, :], (2, 2)),
        jnp.ones(2, jnp.int32), prior_tokens=jnp.full(2, prompt.shape[1], jnp.int32))
    _, cached, _ = ops.propose(state, ahead[:, -1:], jnp.asarray([[7], [7]], jnp.int32), None,
                               jnp.ones((2, 1), bool), jnp.full((2, 1), width, jnp.int32), 0)

    grown = jnp.concatenate([realized, jnp.asarray([[7], [7]], jnp.int32)], axis=1)
    after = model.apply(params, grown, method=model.hidden_states)
    plain = model.apply(params, after[:, :-1], grown[:, 1:], depth=0,
                        positions=jnp.broadcast_to(jnp.arange(1, width + 1)[None, :], (2, width)),
                        method=model.mtp_step)[0]
    largest = float(np.max(np.abs(np.asarray(cached)[:, 0] - np.asarray(plain)[:, -1])))
    assert largest < 3e-5, f"largest difference {largest:g}"
    assert reference.shape[1] == width - 1


def params_of(model, prompt):
    return model.init(jax.random.key(0), prompt)


def test_a_model_without_prediction_depths_is_refused():
    model = decoder()
    prompt = jnp.asarray([[1, 2, 3]], jnp.int32)
    params = model.init(jax.random.key(0), prompt)
    with pytest.raises(ValueError, match="no prediction depths|declares none"):
        generate(model, params, prompt, 2, key=jax.random.key(0), strategy=Speculative(block=2))


def test_a_draft_that_loses_confidence_ends_the_block_on_a_target_draw():
    """Below the confidence threshold the proposer simply stops offering, so
    the block ends on an ordinary target draw. Treating that as a rejection
    would ask for the positive part of `p - q` where the two are equal, which
    is nothing at all."""
    target = spread({3: 0.5, 4: 0.5})
    open_ = run(target, [target] * 3, 512, 5, 4, seed=2)
    np.testing.assert_array_equal(np.asarray(open_.valid).sum(axis=1), 5)
    assert set(np.unique(np.asarray(open_.tokens)[np.asarray(open_.valid)]).tolist()) == {3, 4}
    # A threshold no draft of this distribution can meet truncates every block
    # after its free target draw, and the emitted law is unchanged.
    guarded = run_with(Speculative(block=4, confidence=0.9), target, [target] * 3, 512, 5, seed=2)
    np.testing.assert_array_equal(np.asarray(guarded.valid).sum(axis=1), 5)
    share = float(np.mean(np.asarray(guarded.tokens)[np.asarray(guarded.valid)] == 3))
    assert abs(share - 0.5) < 0.03, f"share {share:g}"


def test_the_prediction_cache_matches_a_teacher_forced_reference():
    """The prompt seeds the prediction cache, and a cached draft step then has
    to produce what an uncached pass over the same pairing produces. A cache
    left empty, paired with the wrong hidden state, or written at the wrong
    position all show up here and nowhere in the emitted tokens, because the
    proposal does not change the law."""
    from dew.nn.inputs import ModelInputs
    from dew.sampling.strategies import reseed
    from dew.sampling.text import _operations, _prefill

    model = predictor()
    prompt = jnp.asarray([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], jnp.int32)
    params = model.init(jax.random.key(0), prompt)
    width = prompt.shape[1]
    states = model.apply(params, prompt, method=model.hidden_states)
    reference = model.apply(params, states[:, :-1], prompt[:, 1:], depth=0,
                            positions=jnp.broadcast_to(jnp.arange(1, width)[None, :],
                                                       (2, width - 1)),
                            method=model.mtp_step)[0]

    ops = _operations(model, params, 0, 1)
    seeded, _ = _prefill(model, params, ModelInputs(prompt[:, :-1]), ops)
    _, cached, _ = ops.propose(seeded, states[:, width - 2:width - 1], prompt[:, width - 1:width],
                               None, jnp.ones((2, 1), bool),
                               jnp.full((2, 1), width - 1, jnp.int32), 0)
    largest = float(np.max(np.abs(np.asarray(cached)[:, 0] - np.asarray(reference)[:, -1])))
    assert largest < 2e-5, f"largest difference {largest:g}"

    # Writing one more accepted token through the same seam keeps agreeing,
    # which is what a block does after it decides its accepted prefix.
    grown = jnp.concatenate([prompt, jnp.asarray([[2], [3]], jnp.int32)], axis=1)
    ahead = model.apply(params, grown, method=model.hidden_states)
    full, _ = _prefill(model, params, ModelInputs(prompt), ops)
    full = reseed(ops, full, (states[:, width - 1],), states[:, width - 1:width],
                  model.apply(params, grown[:, width:width + 1], method=model.token_embeddings),
                  jnp.ones((2, 1), bool), jnp.full((2, 1), width, jnp.int32),
                  jnp.zeros(2, jnp.int32), prior_tokens=jnp.full(2, width, jnp.int32))[0]
    _, after, _ = ops.propose(full, ahead[:, width:width + 1], jnp.asarray([[5], [5]], jnp.int32),
                              None, jnp.ones((2, 1), bool),
                              jnp.full((2, 1), width + 1, jnp.int32), 0)
    plain = model.apply(
        params, jnp.concatenate([ahead, ahead[:, -1:]], axis=1)[:, :-1],
        jnp.concatenate([grown[:, 1:], jnp.asarray([[5], [5]], jnp.int32)], axis=1), depth=0,
        positions=jnp.broadcast_to(jnp.arange(1, width + 2)[None, :], (2, width + 1)),
        method=model.mtp_step)[0]
    largest = float(np.max(np.abs(np.asarray(after)[:, 0] - np.asarray(plain)[:, -1])))
    assert largest < 2e-5, f"largest difference {largest:g}"


def test_a_second_prediction_depth_is_seeded_the_way_the_model_trains_it():
    """`mtp_hidden_states` chains a depth onto the one before it, shifted one
    token further on, and that is the history a checkpoint's second depth was
    trained behind. Seeding every depth from the target's own states instead
    would leave the second one drafting from a pairing it never saw."""
    from dew.nn.inputs import ModelInputs
    from dew.sampling.strategies import reseed
    from dew.sampling.text import _operations, _prefill

    model = predictor(num_nextn_predict_layers=2)
    prompt = jnp.asarray([[1, 2, 3, 4, 5, 6], [6, 7, 8, 9, 10, 11]], jnp.int32)
    params = model.init(jax.random.key(0), prompt)
    width = prompt.shape[1]
    states = model.apply(params, prompt, method=model.hidden_states)
    trained = model.apply(params, states, prompt, method=model.mtp_hidden_states)

    ops = _operations(model, params, 0, 2)
    empty, _ = _prefill(model, params, ModelInputs(prompt[:, :1]), ops)
    _, produced, _ = reseed(
        ops, empty, (states[:, 0], None), states[:, 1:],
        model.apply(params, prompt[:, 1:], method=model.token_embeddings),
        jnp.ones((2, width - 1), bool),
        jnp.broadcast_to(jnp.arange(1, width)[None, :], (2, width - 1)),
        jnp.full(2, width - 2, jnp.int32), prior_tokens=jnp.ones(2, jnp.int32))

    assert len(produced) == 2 == len(trained)
    for depth, (cached, reference) in enumerate(zip(produced, trained)):
        # Depth d starts d positions in, as the training pass shifts it.
        kept = np.asarray(cached)[:, depth:]
        assert kept.shape == np.asarray(reference).shape
        largest = float(np.max(np.abs(kept - np.asarray(reference))))
        assert largest < 3e-5, f"depth {depth} differs by {largest:g}"


def test_explicit_prompt_coordinates_reach_the_prediction_cache():
    """A caller that supplies its own rotary coordinates gets a prediction
    cache written at those coordinates, not at a count of tokens."""
    from dew.nn.inputs import ModelInputs
    from dew.sampling.text import _operations, _prefill

    model = predictor()
    prompt = jnp.asarray([[1, 2, 3, 4, 5]], jnp.int32)
    coordinates = jnp.asarray([[2, 4, 7, 11, 13]], jnp.int32)
    params = model.init(jax.random.key(0), prompt)
    ops = _operations(model, params, 0, 1)
    states = model.apply(params, prompt, positions=coordinates, method=model.hidden_states)
    reference = model.apply(params, states[:, :-1], prompt[:, 1:], depth=0,
                            positions=coordinates[:, 1:], method=model.mtp_step)[0]

    seeded, _ = _prefill(model, params,
                         ModelInputs(prompt[:, :-1], {"positions": coordinates[:, :-1]}), ops)
    _, cached, _ = ops.propose(seeded, states[:, -2:-1], prompt[:, -1:], None,
                               jnp.ones((1, 1), bool), coordinates[:, -1:], 0)
    largest = float(np.max(np.abs(np.asarray(cached)[:, 0] - np.asarray(reference)[:, -1])))
    assert largest < 3e-5, f"largest difference {largest:g}"


def test_a_draft_after_an_advance_is_proposed_at_the_advanced_coordinate():
    """The target and the drafts read one next coordinate. Once the target
    has advanced a token, a draft chained from it sits one coordinate on,
    which a teacher-forced pass at the same coordinates has to reproduce. A
    coordinate the advance left behind would draft every later block one
    place back and nothing in the emitted tokens would show it."""
    from dew.nn.inputs import ModelInputs
    from dew.sampling.strategies import _coordinates, reseed
    from dew.sampling.text import _operations, _prefill

    model = predictor()
    grown = jnp.asarray([[1, 2, 3, 4, 5]], jnp.int32)
    coordinates = jnp.asarray([[10, 11, 12, 13, 14]], jnp.int32)
    params = model.init(jax.random.key(0), grown)
    ops = _operations(model, params, 0, 1)
    states = jnp.asarray(model.apply(params, grown, positions=coordinates,
                                     method=model.hidden_states))
    reference = model.apply(params, states[:, :-1], grown[:, 1:], depth=0,
                            positions=coordinates[:, 1:], method=model.mtp_step)[0]

    prompted, _ = _prefill(model, params,
                           ModelInputs(grown[:, :3], {"positions": coordinates[:, :3]}), ops)
    advanced = ops.advance(prompted, grown[:, 3], jnp.ones(1, bool))
    step = StepState(tokens=jnp.pad(grown[:, :3], ((0, 0), (0, 2))),
                     valid=jnp.asarray([[True, True, True, False, False]]),
                     step=jnp.zeros(1, jnp.int32), active=jnp.ones(1, bool),
                     keys=jax.random.split(jax.random.key(0), 1), prompt_width=3)
    step = step.commit(grown[:, 3], jnp.ones(1, bool))
    base = _coordinates(advanced, step, jnp.arange(2)[None, :])
    # The block that emitted token four writes its own entry behind the draft.
    advanced, _, _ = reseed(ops, advanced, (prompted.hidden,), states[:, 3:4],
                            jnp.asarray(model.apply(params, grown[:, 3:4],
                                                    method=model.token_embeddings)),
                            jnp.ones((1, 1), bool), coordinates[:, 3:4], jnp.zeros(1, jnp.int32),
                            prior_tokens=jnp.full(1, 3, jnp.int32))
    assert ops.propose is not None
    _, proposed, _ = ops.propose(advanced, states[:, 3:4], grown[:, 4:5], None,
                                 jnp.ones((1, 1), bool), base[:, :1], 0)
    np.testing.assert_allclose(proposed[:, 0], reference[:, -1], atol=3e-6, rtol=0)


def test_beam_search_refuses_a_chain_that_leaves_a_live_beam_undefined():
    """A search reads the same distributions a draw does, so a chain that
    removes every token has to raise there too rather than ranking `-inf`."""
    from dew.sampling import Beam

    model = decoder()
    prompt = jnp.asarray([[1, 2, 3]], jnp.int32)
    params = model.init(jax.random.key(0), prompt)
    banned = decoding.SuppressTokens(jnp.arange(VOCAB, dtype=jnp.int32))
    with pytest.raises(Exception, match="without a distribution"):
        generate(model, params, prompt, 4, key=jax.random.key(0), logits=(banned,),
                 strategy=Beam(width=2))
    kept = decoding.SuppressTokens(jnp.asarray([token for token in range(VOCAB) if token != 5],
                                               jnp.int32))
    found = generate(model, params, prompt, 4, key=jax.random.key(0), logits=(kept,),
                     sampling=Sampling(pad_id=0), strategy=Beam(width=2))
    np.testing.assert_array_equal(np.asarray(found.tokens)[0, 3:], 5)


def test_a_media_prompt_seeds_the_depths_with_its_prepared_embeddings():
    """An image slot's embedding comes from the vision tower, not from the
    token id sitting in that slot. A depth seeded from ids would read a
    placeholder where the picture is, and the prefill already prepared the
    real embeddings, so they reach the depths without running the tower
    again."""
    from dew.nn.inputs import ModelInputs
    from dew.nn.multimodal import MultimodalTransformer
    from dew.nn.vision import GemmaProjector, SiglipVision
    from dew.sampling.text import _operations, _prefill

    language = predictor()
    model = MultimodalTransformer(
        language, SiglipVision(hidden_size=16, intermediate_size=32, num_layers=1, num_heads=2,
                               image_size=8, patch_size=4),
        GemmaProjector(vision_width=16, text_width=16, patches_per_side=2, tokens_per_side=1),
        family="gemma3", image_token_id=1)
    prompt = jnp.asarray([[2, 1, 3, 4]], jnp.int32)
    indices = jnp.asarray([[-1, 0, -1, -1]], jnp.int32)
    pixels = jax.random.normal(jax.random.key(1), (1, 1, 3, 8, 8))
    params = model.init(jax.random.key(0), prompt, image_indices=indices,
                        conditioning={"pixel_values": pixels})
    width = prompt.shape[1]

    states = model.apply(params, prompt, image_indices=indices,
                         conditioning={"pixel_values": pixels}, method=model.hidden_states)
    reference = model.apply(params, states, prompt, image_indices=indices,
                            conditioning={"pixel_values": pixels},
                            method=model.mtp_hidden_states)[0]

    ops = _operations(model, params, 0, 1)
    seeded, _ = _prefill(model, params,
                         ModelInputs(prompt[:, :-1], {"image_indices": indices[:, :-1]},
                                     {"pixel_values": pixels}), ops)
    _, _, produced = ops.propose(seeded, states[:, -2:-1], prompt[:, -1:], None,
                                 jnp.ones((1, 1), bool), jnp.full((1, 1), width - 1, jnp.int32), 0)
    largest = float(np.max(np.abs(np.asarray(produced)[:, 0] - np.asarray(reference)[:, -1])))
    assert largest < 3e-5, f"largest difference {largest:g}"


def test_multi_axis_rotary_coordinates_reach_the_depths():
    """A processor that emits three rotary axes per token gives the depths
    coordinates, not a count of tokens: the prompt seed carries the axes it
    was given and a drawn token continues from the coordinate the model's
    cache reached, which is where the reference puts it."""
    from dew.nn.inputs import ModelInputs
    from dew.nn.mixers.attention import AttentionMixer
    from dew.sampling.text import _operations, _prefill

    model = CausalTransformer(
        vocab_size=VOCAB, emb_features=16, num_layers=1, num_heads=2, head_dim=8,
        mlp_features=32, max_seq_len=16, dtype="float32", num_nextn_predict_layers=1,
        partial_rotary_type="default", mixer=AttentionMixer(mrope_section=(1, 1, 1)))
    tokens = jnp.asarray([[1, 2, 3, 4]], jnp.int32)
    rotary = jnp.asarray([[[0, 0, 0], [1, 4, 1], [1, 4, 2], [5, 5, 5]]], jnp.int32)
    params = model.init(jax.random.key(0), tokens, rotary_positions=rotary)

    states = model.apply(params, tokens, rotary_positions=rotary, method=model.hidden_states)
    reference = model.apply(params, states, tokens, rotary_positions=rotary,
                            method=model.mtp_hidden_states)[0]

    ops = _operations(model, params, 0, 1)
    seeded, _ = _prefill(model, params,
                         ModelInputs(tokens[:, :-1], {"rotary_positions": rotary[:, :-1]}), ops)
    _, _, produced = ops.propose(seeded, states[:, -2:-1], tokens[:, -1:], None,
                                 jnp.ones((1, 1), bool), jnp.full((1, 1), 5, jnp.int32), 0)
    largest = float(np.max(np.abs(np.asarray(produced)[:, 0] - np.asarray(reference)[:, -1])))
    assert largest < 3e-5, f"largest difference {largest:g}"
    # The continuation coordinate is the model's own, not the token count.
    assert int(np.asarray(seeded.positions)[0]) == 5


def test_a_padded_prompt_seeds_the_depths_at_its_logical_coordinates():
    """A left-padded prompt's third real token sits at coordinate two, not at
    the physical slot the padding pushed it to. Seeding from slots would put
    the whole prediction history one place along."""
    from dew.nn.inputs import ModelInputs
    from dew.sampling.text import _operations, _prefill

    model = predictor()
    padded = jnp.asarray([[0, 1, 2, 3]], jnp.int32)
    mask = jnp.asarray([[False, True, True, True]])
    plain = jnp.asarray([[1, 2, 3]], jnp.int32)
    params = model.init(jax.random.key(0), plain)

    grown = jnp.concatenate([plain, jnp.asarray([[4]], jnp.int32)], axis=1)
    states = model.apply(params, grown, method=model.hidden_states)
    reference = model.apply(params, states[:, :-1], grown[:, 1:], depth=0,
                            positions=jnp.asarray([[1, 2, 3]], jnp.int32),
                            method=model.mtp_step)[0]

    ops = _operations(model, params, 0, 1)
    seeded, _ = _prefill(model, params, ModelInputs(padded, {"attention_mask": mask}), ops)
    _, cached, _ = ops.propose(seeded, states[:, 2:3], grown[:, 3:4], None,
                               jnp.ones((1, 1), bool), jnp.asarray([[3]], jnp.int32), 0)
    largest = float(np.max(np.abs(np.asarray(cached)[:, 0] - np.asarray(reference)[:, -1])))
    assert largest < 3e-5, f"largest difference {largest:g}"


def test_a_cold_prompt_holds_its_second_depth_back_until_it_has_a_predecessor():
    """A depth's first entry needs that many tokens behind it. A one-token
    prompt gives the second depth no predecessor at all, so inventing one from
    the target's state writes an entry the training pass never has."""
    from dew.nn.inputs import ModelInputs
    from dew.sampling.strategies import reseed
    from dew.sampling.text import _operations, _prefill

    model = predictor(num_nextn_predict_layers=2)
    cold = jnp.asarray([[1]], jnp.int32)
    emitted = jnp.asarray([[2, 3]], jnp.int32)
    whole = jnp.concatenate([cold, emitted], axis=1)
    params = model.init(jax.random.key(0), whole)
    states = model.apply(params, whole, method=model.hidden_states)
    trained = model.apply(params, states, whole, method=model.mtp_hidden_states)

    ops = _operations(model, params, 0, 2)
    state, _ = _prefill(model, params, ModelInputs(cold), ops)
    _, produced, _ = reseed(ops, state, (state.hidden,) + tuple(state.drafts[1:]),
                            states[:, 1:], model.apply(params, emitted,
                                                       method=model.token_embeddings),
                            jnp.ones((1, 2), bool), jnp.asarray([[1, 2]], jnp.int32),
                            jnp.ones(1, jnp.int32), prior_tokens=jnp.ones(1, jnp.int32))
    # Depth two's only entry is at coordinate two, and it is the trained one.
    largest = float(np.max(np.abs(np.asarray(produced[1])[:, 1] - np.asarray(trained[1])[:, -1])))
    assert largest < 3e-5, f"largest difference {largest:g}"


@pytest.mark.parametrize("prefix, coordinates", [(1, "offset"), (2, "ordinary"), (4, "repeated")])
def test_prediction_depths_resume_from_real_history_not_rotary_coordinates(prefix, coordinates):
    """Cached proposal logits match training across depth-readiness boundaries.

    Rotary offsets and repeated image coordinates do not create or remove
    predecessors. A depth that just became valid must also reach the next
    depth across a block boundary.
    """
    from dew.nn.inputs import ModelInputs
    from dew.sampling.strategies import reseed
    from dew.sampling.text import _operations, _prefill

    model = predictor(num_nextn_predict_layers=2)
    whole = jnp.arange(1, prefix + 4, dtype=jnp.int32)[None, :]
    positions = (jnp.zeros_like(whole) if coordinates == "repeated" else
                 jnp.arange(whole.shape[1], dtype=jnp.int32)[None, :]
                 + (20 if coordinates == "offset" else 0))
    params = model.init(jax.random.key(13), whole)
    hidden = model.apply(params, whole, positions=positions, method=model.hidden_states)
    reference = model.apply(params, hidden, whole, positions=positions, method=model.mtp_logits)[1]
    ops = _operations(model, params, 0, 2)
    state, _ = _prefill(model, params,
                         ModelInputs(whole[:, :prefix], {"positions": positions[:, :prefix]}), ops)
    emitted = whole[:, prefix:-1]
    state, _, carried = reseed(
        ops, state, (state.hidden,) + state.drafts[1:], hidden[:, prefix:-1],
        ops.embed(emitted), jnp.ones(emitted.shape, bool), positions[:, prefix:-1],
        jnp.asarray([emitted.shape[1] - 1], jnp.int32),
        prior_tokens=jnp.asarray([prefix], jnp.int32))
    _, proposed, _ = ops.propose(state, carried[1][:, None], whole[:, -1:], None,
                                 jnp.ones((1, 1), bool), positions[:, -1:], 1)
    np.testing.assert_allclose(proposed[:, 0], reference[:, -1], atol=3e-6, rtol=0)
