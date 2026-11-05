"""Speculative decoding: the emitted law, and the cache the block leaves behind.

The law is checked on a real model's own draws: a block's tokens, whatever its
draft proposed, are drawn as the target alone draws them, the acceptance, the
residual after a rejection and the bonus together, held to the target's own
chances by the sampling gate (tests/sampling_gate.py). The cache is checked
against greedy sampling on real models: with a zero temperature the target's
point mass wins every rejection, so a block has to emit exactly the greedy
walk, which it can only do if the state it replays after a rejection is the
state the accepted prefix would have left.
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


@pytest.mark.parametrize("kind", ["attention", "mla", "recurrent", "parallel"])
def test_speculation_reproduces_the_greedy_walk_on_a_real_model(kind):
    """At zero temperature the target is a point mass, so every rejected draft
    is replaced by the target's own token and the whole run has to equal the
    greedy walk. It only can if the accepted prefix is replayed into the
    cache: a recurrent mixer keeps a running summary that no cursor rewinds,
    and reusing the draft's state would show up on the next token. A parallel
    hybrid (Falcon-H1's Mamba-2 beside attention, its prediction depth's mixer
    the same) drafts and verifies through both branches' caches (ledger 3575)."""
    model = predictor(kind)
    prompts = jnp.asarray([[1, 2, 3], [7, 8, 9]], jnp.int32)
    params = jax.jit(model.init)(jax.random.key(0), prompts)
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
    V4 kind rotating under YaRN, which V4's attention takes where it refuses LongRoPE."""
    from dew.interop import Pretrained
    from dew.nn.backbones.layer_plan import LayerKind
    from dew.nn.mixers.attention import AttentionMixer
    from dew.nn.rope import LongRopeScaling, YarnScaling

    model = Pretrained.load(DSPARK, weights=False, dtype="float32", attention_impl="reference").model
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


def greedy_walk(model, params, prompts, steps: int) -> np.ndarray:
    """The greedy walk's `steps` tokens after each of `prompts`: the prompt's prefill, then a token at a
    time, each its last logits' argmax, through the decode the decode loops run (`decode_ops`), one
    compiled step a token."""
    from dew.nn.inputs import ModelInputs
    from dew.sampling.text import decode_ops, prefill_state

    ops = decode_ops(model, params, 0, 0)

    def step(state, _):
        token = jnp.argmax(state.logits, axis=-1).astype(prompts.dtype)
        return ops.advance(state, token, jnp.ones(token.shape[0], bool)), token

    @jax.jit
    def walk(prompts):
        state, _ = prefill_state(model, params, ModelInputs(prompts), ops)
        return jnp.moveaxis(jax.lax.scan(step, state, None, length=steps)[1], 0, 1)

    return np.asarray(walk(prompts))


LONGROPE_CASES = [(np.asarray([[1, 2, 3], [7, 8, 9]], np.int32), 6),
                  (np.asarray([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], np.int32), 2)]
"""Two rows' prompts that cross LongRoPE's change past their third token, and two that start at it, each
with its budget."""


@pytest.mark.parametrize("case", range(len(LONGROPE_CASES)))
@pytest.mark.parametrize("kind", ["attention", "recurrent", "dspark"])
def test_speculation_across_a_longrope_rebuild_reproduces_the_greedy_walk(kind, case):
    """A LongRoPE crossing rebuilds every cache the decode keeps (`decode_ops`), a prediction depth's or
    a block drafter's with the trunk's, so speculating across it at zero temperature emits the greedy
    walk (`greedy_walk`). A row whose candidates would cross keeps none and walks its block's one token
    (`strategies.block_step`), crossing in the replay of what the block emits; a prompt that starts so
    walks a token a block and still fills its budget. Each case a test of its own, its programs its own
    (ledger 3892: DSpark's two took 168 s in one test)."""
    model, params = longrope_speculator(kind)
    held, steps = LONGROPE_CASES[case]
    prompts = jnp.asarray(held)
    speculated = generate(model, params, prompts, steps, key=jax.random.key(1),
                          sampling=Sampling(temperature=0), strategy=Speculative(block=3))
    np.testing.assert_array_equal(np.asarray(speculated.tokens)[:, prompts.shape[1]:],
                                  greedy_walk(model, params, prompts, steps))
    np.testing.assert_array_equal(np.asarray(speculated.lengths), steps)


def longrope_speculator(kind: str):
    """`kind`'s model whose rotary changes tables past its sixth position, and its variables."""
    model = longrope_drafter() if kind == "dspark" else longrope_predictor(kind)
    return model, jax.jit(model.init)(jax.random.key(0), jnp.asarray(LONGROPE_CASES[0][0]))


@pytest.mark.parametrize("kind", ["attention", "recurrent", "dspark"])
def test_a_speculative_block_holds_a_longrope_rebuild_once(kind):
    """A block's program holds the rebuild, a whole prefill of the history, once: its candidates' verify
    compiles none, its replay the one (ledger 3703: the candidates' verify held a second, 27 s of the
    DSpark program's compile on CPU). The speculative loop traced over the decode's own operations
    (`DecodeOps`), each verify's program counted, run as traced."""
    from dew.nn.inputs import ModelInputs
    from dew.sampling.text import decode_ops, prediction_depths, prefill_state

    model, params = longrope_speculator(kind)
    ops = decode_ops(model, params, 0, prediction_depths(model))
    verified, conds = ops.verify, []
    assert verified is not None

    def verify(state, tokens, valid, **options):
        program, shape = jax.make_jaxpr(lambda *operands: verified(*operands, **options),
                                        return_shape=True)(state, tokens, valid)
        conds.append(sum(equation.primitive.name == "cond" for equation in program.eqns))
        outputs = jax.core.eval_jaxpr(program.jaxpr, program.consts, *jax.tree.leaves((state, tokens, valid)))
        return jax.tree.unflatten(jax.tree.structure(shape), outputs)

    held, steps = LONGROPE_CASES[0]
    prompts = jnp.asarray(held)
    rows, width = prompts.shape

    def speculate(prompts):
        state, real = prefill_state(model, params, ModelInputs(prompts), ops)
        start = decoding.StepState(jnp.concatenate([prompts, jnp.zeros((rows, steps), jnp.int32)], axis=1),
                                   jnp.arange(width + steps)[None, :] < width, jnp.zeros(rows, jnp.int32),
                                   real, jax.random.split(jax.random.key(1), rows), prompt_width=width)
        return Speculative(block=3)(state, start, dataclasses.replace(ops, verify=verify),
                                    decoding.chain((decoding.Greedy(),)),
                                    decoding.finish(decoding.Finishing.of(())), steps, 1)

    jax.make_jaxpr(speculate)(prompts)
    # A block verifies its candidates, then replays what it emits, which alone holds the rebuild's cond.
    candidates, replay = conds
    assert replay == candidates + 1


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
    params = jax.jit(model.init)(jax.random.key(0), jnp.ones((2, 3), jnp.int32))

    def walk(model, params):
        """The prompt's prefill, then the stretch a token at a time, each token's logits."""
        ops = decode_ops(model, params, 0, depths)

        def advanced(state, token):
            state = ops.advance(state, token, jnp.ones(token.shape[0], bool))
            return state, state.logits

        @jax.jit
        def walked(prompts, stretch):
            # One token a step of a loop, as the decode's own loop carries its state: one program of a
            # step, whatever the stretch.
            state, _ = prefill_state(model, params, ModelInputs(prompts), ops)
            return jnp.moveaxis(jax.lax.scan(advanced, state, stretch.T)[1], 0, 1)

        return lambda prompts, stretch: np.asarray(walked(jnp.asarray(prompts), jnp.asarray(stretch)))

    verified = stretch_verified(model, params, depths)

    with jax.enable_x64(new_val=True):
        wide = walk(model.clone(dtype=jnp.float64), widened(params))

    @as_given
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
    params = jax.jit(model.init)(jax.random.key(0), jnp.ones((2, 3), jnp.int32))
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

    @as_given
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
    params = jax.jit(model.init)(jax.random.key(0), prompt)
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


LAW_ROWS, LAW_SEEDS, LAW_TEMPERATURE = 512, 8, 4.0
"""The law's draws: each seed a call of `LAW_ROWS` copies of one prompt, every row drawn apart, at a
temperature that spreads the tiny decoder's chances over every token, so each pair is drawn often enough
to be counted (`sampling_gate.FEWEST_EXPECTED`)."""


def test_a_blocks_tokens_are_drawn_as_the_target_draws_them():
    """Speculation changes how a token is drawn, never its law: a block's first two tokens, the target's
    own draw and the one its draft proposed after it (accepted at the ratio of the two distributions or
    replaced from their difference), are drawn as the target alone draws them, the pair at the chance the
    target's float64 run gives the first token times the second's given it, at the sampling's temperature,
    over every pair of a 13-token vocabulary (the sampling gate). The draft is the model's own prediction
    depth, untrained, so it proposes off the target's law and the acceptance has to undo it."""
    from reference_error import widened
    from sampling_gate import assert_draws_as_declared

    model = predictor()
    prompt = jnp.asarray([[1, 2, 3]], jnp.int32)
    params = jax.jit(model.init)(jax.random.key(0), prompt)
    vocab = model.vocab_size
    with jax.enable_x64(new_val=True):
        wide, wide_params = model.clone(dtype=jnp.float64), widened(params)
        first = jax.nn.softmax(wide.apply(wide_params, prompt)[0, -1] / LAW_TEMPERATURE)
        pairs = jnp.concatenate([jnp.broadcast_to(prompt, (vocab, 3)), jnp.arange(vocab)[:, None]], axis=1)
        second = jax.nn.softmax(wide.apply(wide_params, pairs)[:, -1] / LAW_TEMPERATURE, axis=-1)
        chances = np.asarray(first[:, None] * second).reshape(-1)

    rows = jnp.broadcast_to(prompt, (LAW_ROWS, 3))

    def counts(seed):
        drawn = generate(model, params, rows, 2, key=jax.random.key(seed),
                         sampling=Sampling(temperature=LAW_TEMPERATURE), strategy=Speculative(block=3))
        tokens = np.asarray(drawn.tokens)[:, 3:5]
        return np.bincount(tokens[:, 0] * vocab + tokens[:, 1], minlength=vocab * vocab)

    assert_draws_as_declared(counts, chances, "a speculative block's first two tokens", seeds=LAW_SEEDS)
