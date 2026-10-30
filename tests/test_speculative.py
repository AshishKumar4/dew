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
from pathlib import Path

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


DSPARK = Path(__file__).parent / "fixtures" / "hf" / "deepseek-v4-dspark-tiny"


def longrope_drafter():
    """The tiny DeepSeek V4-Flash-0731 (`DSPARK`) with its DSpark block drafter over a trunk of plain
    attention whose rotary switches tables past its sixth position (LongRoPE), the drafter's own sliding
    V4 kind rotating under YaRN, which V4's attention takes where it refuses LongRoPE; its like layers
    scanned, so each compiles once."""
    from dew.interop import Pretrained
    from dew.nn.backbones.layer_plan import LayerKind
    from dew.nn.mixers.attention import AttentionMixer
    from dew.nn.rope import LongRopeScaling, YarnScaling

    model = Pretrained.load(DSPARK, weights=False, dtype="float32", attention_impl="reference").model
    model = model.clone(scan_layers=True)
    drafter = model.dspark.layer_type
    head = model.head_dim or model.emb_features // model.num_heads
    pairs = int(head * (model.partial_rotary_factor or 1.0)) // 2
    return model.clone(
        layer_types=("trunk",) * model.num_layers,
        kinds={"trunk": LayerKind(mixer=AttentionMixer()),
               drafter: dataclasses.replace(model.kinds[drafter], rope_scaling=YarnScaling(
                   factor=4.0, original_max_position_embeddings=8))},
        rope_scaling=LongRopeScaling(short_factor=(1.0,) * pairs,
                                     long_factor=tuple(2.0 + pair for pair in range(pairs)),
                                     original_max_position_embeddings=6))


LONGROPE_DRAFTING = {"depth": (longrope_predictor, 1), "drafter": (longrope_drafter, 0)}
"""The decoders that draft across a LongRoPE change, each with the prediction depths a drafting
decode runs (`decode_ops`): a prediction depth's, and a block drafter's, which runs none."""


@pytest.mark.parametrize("kind", ["attention", "recurrent", "dspark"])
def test_speculation_across_a_longrope_rebuild_reproduces_the_greedy_walk(kind):
    """A LongRoPE crossing rebuilds every cache the decode keeps (`decode_ops`), a prediction depth's or
    a block drafter's with the trunk's, so speculating across it at zero temperature emits the greedy
    walk, each row's crossing inside a block."""
    model = longrope_drafter() if kind == "dspark" else longrope_predictor(kind)
    prompts = jnp.asarray([[1, 2, 3], [7, 8, 9]], jnp.int32)
    params = model.init(jax.random.key(0), prompts)
    walked, speculated = (generate(model, params, prompts, 6, key=jax.random.key(1),
                                   sampling=Sampling(temperature=0), strategy=strategy)
                          for strategy in (None, Speculative(block=3)))
    np.testing.assert_array_equal(np.asarray(speculated.tokens), np.asarray(walked.tokens))
    np.testing.assert_array_equal(np.asarray(speculated.lengths), 6)


def stretch_verified(model, params, depths):
    """The state and logits a stretch verified from a prompt's prefill leaves (`DecodeOps.verify`), the
    decode running `depths` prediction depths, compiled once."""
    from dew.nn.inputs import ModelInputs
    from dew.sampling.text import decode_ops, prefill_state

    ops = decode_ops(model, params, 0, depths)

    @jax.jit
    def verified(prompts, stretch):
        state, _ = prefill_state(model, params, ModelInputs(prompts), ops)
        return ops.verify(state, stretch, jnp.ones(stretch.shape, bool))[:2]

    return verified


CROSSING_SEEDS = {"depth": 8, "drafter": 8}
"""The crossing checks' declared draws of prompts and stretches (check 1), each decoder's the most
either check took, sized by `seeds_for` paired on armada."""


@pytest.mark.parametrize("drafting", LONGROPE_DRAFTING)
def test_a_block_across_a_longrope_change_answers_each_token_as_the_walk_does(drafting):
    """A stretch verified from before LongRoPE's change to past it answers each of its tokens as the
    walk a token at a time does (`decode_ops`): the one that reaches the change at the short table over
    the cache as it stands, those past it from the history rebuilt at the long, each token's logits as
    near the walk's float64 run as the walk's own (check 1, over drawn prompts and stretches)."""
    from reference_error import as_given, assert_as_exact, widened

    from dew.nn.inputs import ModelInputs
    from dew.sampling.text import decode_ops, prefill_state

    build, depths = LONGROPE_DRAFTING[drafting]
    model = build()
    params = model.init(jax.random.key(0), jnp.ones((2, 5), jnp.int32))

    def walk(model, params):
        """The prompt's prefill, then the stretch a token at a time, each token's logits."""
        ops = decode_ops(model, params, 0, depths)

        @jax.jit
        def walked(prompts, stretch):
            state, _ = prefill_state(model, params, ModelInputs(prompts), ops)
            logits = []
            for at in range(stretch.shape[1]):
                state = ops.advance(state, stretch[:, at], jnp.ones(stretch.shape[0], bool))
                logits.append(state.logits)
            return jnp.stack(logits, axis=1)

        return lambda prompts, stretch: np.asarray(walked(jnp.asarray(prompts), jnp.asarray(stretch)))

    verified = stretch_verified(model, params, depths)

    with jax.enable_x64(new_val=True):
        wide = walk(model.clone(dtype=jnp.float64), widened(params))

    def truth(prompts, stretch):
        with jax.enable_x64(new_val=True):
            return wide(prompts, stretch)

    def draw(seed):
        # Five prompt tokens, so the stretch's first reaches the change at the sixth and the rest pass it.
        rng = np.random.default_rng([0x1899, 1, seed])
        return tuple(rng.integers(1, model.vocab_size, (2, width)).astype(np.int32) for width in (5, 3))

    assert_as_exact(as_given(lambda prompts, stretch: np.asarray(verified(prompts, stretch)[1])),
                    as_given(walk(model, params)), draw, f"a {drafting}'s block across a LongRoPE change",
                    seeds=CROSSING_SEEDS[drafting], truth_fn=truth)


@pytest.mark.parametrize("drafting", LONGROPE_DRAFTING)
def test_a_longrope_crossing_reseeds_the_drafter_as_a_prefill_of_its_history_does(drafting):
    """A verified stretch that crosses the table change leaves the decoder state a prefill of the whole
    history leaves: the trunk's cache rebuilt at the long factors, the prediction depth's cache or the
    block drafter's windows reseeded over the history, and the last state and logits, each as near the
    prefill's float64 run as the prefill's own (check 1, over drawn prompts and stretches: the two
    compile apart, so their roundings differ), its rows marked rebuilt."""
    from reference_error import as_given, assert_as_exact, widened

    from dew.nn.inputs import ModelInputs
    from dew.sampling.strategies import DecoderState
    from dew.sampling.text import decode_ops, prefill_state

    build, depths = LONGROPE_DRAFTING[drafting]
    model = build()
    params = model.init(jax.random.key(0), jnp.ones((2, 3), jnp.int32))
    width = model.max_seq_len

    def floats(state: DecoderState):
        leaves = jax.tree_util.tree_leaves_with_path(dataclasses.replace(state, rebuilt=None))
        return {jax.tree_util.keystr(path): np.asarray(leaf) for path, leaf in leaves
                if jnp.issubdtype(leaf.dtype, jnp.floating)}

    def prefill(model, params):
        """The prefill of a prompt and its stretch as one history, compiled once."""
        ops = decode_ops(model, params, 0, depths)

        @jax.jit
        def run(prompts, stretch):
            history = jnp.zeros((2, width), jnp.int32).at[:, :7].set(jnp.concatenate([prompts, stretch], 1))
            real = jnp.broadcast_to(jnp.arange(width)[None, :] < 7, history.shape)
            return prefill_state(model, params, ModelInputs(history, {"attention_mask": real}), ops)[0]

        return lambda prompts, stretch: floats(run(jnp.asarray(prompts), jnp.asarray(stretch)))

    verified = stretch_verified(model, params, depths)

    @as_given
    def crossed(prompts, stretch):
        after = verified(jnp.asarray(prompts), jnp.asarray(stretch))[0]
        assert np.asarray(after.rebuilt).all()
        return floats(after)

    with jax.enable_x64(new_val=True):
        wide = prefill(model.clone(dtype=jnp.float64), widened(params))

    def truth(prompts, stretch):
        with jax.enable_x64(new_val=True):
            return wide(prompts, stretch)

    def draw(seed):
        rng = np.random.default_rng([0x1899, seed])
        return tuple(rng.integers(1, model.vocab_size, (2, width)).astype(np.int32) for width in (3, 4))

    label = f"a LongRoPE crossing's {drafting} reseed"
    assert_as_exact(crossed, as_given(prefill(model, params)), draw, label, seeds=CROSSING_SEEDS[drafting],
                    truth_fn=truth)


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
