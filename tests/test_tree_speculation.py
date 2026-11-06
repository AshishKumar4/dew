"""Tree speculative decoding (ledger 4527): a tree verified in one pass, its acceptance and its draws."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import checkify
from model_support import decoder
from reference_error import as_given, assert_as_exact, widened

from dew.nn.inputs import ModelInputs
from dew.sampling.text import decode_ops, prefill_state
from dew.sampling.trees import TreeBlock, TreeShape

SHAPE = TreeShape.widths(2, 2, 1)
"""Two children of the root, two of each, one of each of those: 2 + 4 + 4 nodes, three deep."""
SEEDS = 8
"""The tree verify's declared draws of prompts and trees (check 1)."""
KINDS = ["attention", "windowed"]
"""The models a tree runs on: attention over every key, and attention windowed to its last three keys,
which a tree's prefix reaches past."""


def _model(kind: str, max_seq_len: int):
    """The tiny decoder of `kind` (`KINDS`) holding `max_seq_len` tokens a row."""
    from dew.nn.backbones.layer_plan import LayerKind

    model = decoder("attention").clone(max_seq_len=max_seq_len)
    if kind == "windowed":
        return model.clone(layer_types=("windowed",), kinds={"windowed": LayerKind(window=3)})
    return model


@pytest.mark.parametrize("kind", KINDS)
def test_a_trees_nodes_verified_in_one_pass_score_as_their_paths_walked(kind):
    """Each node of a tree verified in one pass (`TreeBlock`: every node over the row's prefix and its own
    ancestors, at its depth's place) scores the token after it as the walk over its path from the root does,
    a token at a time (`decode_ops`): each node's logits as near the walk's float64 run as the walk's own
    (check 1, over drawn prompts and trees)."""
    model = _model(kind, 24)
    params = jax.jit(model.init)(jax.random.key(0), jnp.ones((2, 3), jnp.int32))
    paths = [SHAPE.ancestry(node) for node in range(SHAPE.nodes)]

    def walk(model, params):
        """The prompt's prefill, then each node's path from the root a token at a time, the last token's
        logits for each node `[rows, 1 + N, V]` (the root's first)."""
        ops = decode_ops(model, params, 0, 0)

        @jax.jit
        def walked(prompts, fed):
            state, _ = prefill_state(model, params, ModelInputs(prompts), ops)
            every = jnp.ones(prompts.shape[0], bool)
            root = ops.advance(state, fed[:, 0], every)
            scored = [root.logits]
            for path in paths:
                held = root
                for node in path:
                    held = ops.advance(held, fed[:, 1 + node], every)
                scored.append(held.logits)
            return jnp.stack(scored, axis=1)

        return lambda prompts, fed: np.asarray(walked(jnp.asarray(prompts), jnp.asarray(fed)))

    ops = decode_ops(model, params, 0, 0)

    @jax.jit
    def verified(prompts, fed):
        state, _ = prefill_state(model, params, ModelInputs(prompts), ops)
        # Every prompt token is real: each row's next slot and place its prompt's length.
        total = jnp.full(prompts.shape[0], prompts.shape[1], jnp.int32)
        block = TreeBlock(SHAPE, total, total)
        return ops.verify(state, fed, jnp.ones(fed.shape, bool), rebuilds=False, tree=block)[1]

    with jax.enable_x64(new_val=True):
        wide = walk(model.clone(dtype=jnp.float64), widened(params))

    @as_given
    def truth(prompts, fed):
        with jax.enable_x64(new_val=True):
            return wide(prompts, fed)

    def draw(seed):
        rng = np.random.default_rng([0x4527, seed])
        return (rng.integers(1, model.vocab_size, (2, 4)).astype(np.int32),
                rng.integers(1, model.vocab_size, (2, 1 + SHAPE.nodes)).astype(np.int32))

    ours = as_given(lambda prompts, fed: np.asarray(verified(jnp.asarray(prompts), jnp.asarray(fed))))
    assert_as_exact(ours,
                    as_given(walk(model, params)), draw, f"a {kind} model's tree verified in one pass",
                    seeds=SEEDS, truth_fn=truth)


def test_a_trees_runs_read_each_nodes_prefix_and_ancestors():
    """A tree block's key runs (`TreeBlock.runs`) hold each fed token's prefix and root, then its ancestors
    and itself a slot each: their union is exactly the keys its path reads."""
    shape = TreeShape((-1, -1, 0, 1, 2))
    cursor = jnp.asarray([3, 0], jnp.int32)
    runs = np.asarray(TreeBlock(shape, cursor, cursor).runs())
    for row, at in enumerate((3, 0)):
        for fed in range(shape.nodes + 1):
            seen = set()
            for start, end in runs[row, fed]:
                seen.update(range(start, end))
            path = () if fed == 0 else shape.ancestry(fed - 1)
            assert seen == set(range(at + 1)) | {at + 1 + node for node in path}, (row, fed, runs[row, fed])


def _tree_generated(model, params, prompts, budget: int, sampling, shape: TreeShape, drafting, key):
    """`generate`'s draws under `TreeSpeculative(shape)`, its tree drafted by `drafting(ops)` (a
    `DecodeOps.draft_tree`), as `generate` runs a strategy: the prompt's prefill, then the strategy's loop."""
    from dew.sampling import TreeSpeculative, decoding
    from dew.sampling.decoding import StepState
    from dew.sampling.text import resolve

    transforms, stopping, strategy = resolve(sampling, None, None, TreeSpeculative(shape))
    ops = decode_ops(model, params, 0, 0)
    ops = dataclasses.replace(ops, draft_tree=drafting(ops))

    def run(prompts, key):
        state, real = prefill_state(model, params, ModelInputs(prompts), ops)
        rows, width = prompts.shape
        start = StepState(tokens=jnp.concatenate([prompts, jnp.zeros((rows, budget), jnp.int32)], axis=1),
                          valid=jnp.concatenate([jnp.ones((rows, width), bool),
                                                 jnp.zeros((rows, budget), bool)], axis=1),
                          step=jnp.zeros(rows, jnp.int32), active=real, keys=jax.random.split(key, rows),
                          prompt_width=width, budget=jnp.full(rows, budget, jnp.int32))
        return strategy(state, start, ops, decoding.chain(transforms), decoding.finish(stopping), budget, 1)

    # Its device checks discharged into a returned error, as `generate` discharges them.
    error, drawn = jax.jit(checkify.checkify(run, errors=checkify.user_checks))(jnp.asarray(prompts), key)
    error.throw()
    return drawn


def _ranked(ranks):
    """A tree drafter (`DecodeOps.draft_tree`) picking each node's children as the target ranks the tokens
    after their parent's path: the `j`-th child at rank `ranks[depth - 1][j]` (0 the most probable), each a
    pick (one-hot `q`); the target's own walk over each path, so its picks are the target's."""
    from dew.sampling.trees import TreeDraft

    def drafting(ops):
        def draft(state, root, shape, keys):
            del keys
            every = jnp.ones(root.shape[0], bool)
            held = [ops.advance(state, root, every)]
            tokens = []
            for node, parent in enumerate(shape.parents):
                order = jnp.argsort(-held[parent + 1].logits, axis=-1)
                index = shape.children[parent + 1].index(node)
                token = order[:, ranks[shape.depths[node] - 1][index]].astype(jnp.int32)
                tokens.append(token)
                held.append(ops.advance(held[parent + 1], token, every))
            return TreeDraft(jnp.stack(tokens, axis=1), None)

        return draft

    return drafting


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("stop", ["budget", "eos"])
def test_tree_speculation_draws_as_greedy_decoding_does(kind, stop):
    """At zero temperature the target is a point mass, so a tree's walk keeps exactly the path of greedy
    picks it holds and emits the target's next pick after it: the draws equal plain greedy decoding's
    (`generate` with `Sample`), lengths and terminations too. The drafter ranks its first child second and
    its second first at the first two depths, and offers only the second-ranked at the third, so the walk
    rejects a child, accepts its sibling, and ends on a rejection with the target's own token. Under `eos`
    the first row's third greedy token ends it, so the rows finish at different steps."""
    from dew.sampling import Sample, Sampling, generate

    model = _model(kind, 40)
    prompts = np.asarray([[1, 2, 3], [7, 8, 9]], np.int32)
    params = jax.jit(model.init)(jax.random.key(0), jnp.asarray(prompts))
    sampling = Sampling(temperature=0)
    if stop == "eos":
        walked = generate(model, params, jnp.asarray(prompts), 8, key=jax.random.key(1), sampling=sampling,
                          strategy=Sample())
        sampling = Sampling(temperature=0, eos_token_ids=int(np.asarray(walked.tokens)[0, 5]), pad_token_id=0)
    plain = generate(model, params, jnp.asarray(prompts), 8, key=jax.random.key(1), sampling=sampling,
                     strategy=Sample())
    drawn = _tree_generated(model, params, prompts, 8, sampling, SHAPE, _ranked([(1, 0), (1, 0), (1,)]),
                            jax.random.key(1))
    tokens = np.where(np.asarray(drawn.valid), np.asarray(drawn.tokens), 0)
    np.testing.assert_array_equal(tokens, np.asarray(plain.tokens)[:, 3:])
    np.testing.assert_array_equal(np.sum(np.asarray(drawn.valid), axis=1), np.asarray(plain.lengths))
    np.testing.assert_array_equal(np.asarray(drawn.terminated), np.asarray(plain.terminated))
    if stop == "eos":
        # The rows finish at different steps, the first on its EOS.
        lengths = np.sum(np.asarray(drawn.valid), axis=1)
        assert lengths[0] != lengths[1], lengths


def test_a_tree_past_its_rows_cache_walks_as_greedy_decoding_does():
    """A row whose cache holds no room for the whole tree (13 prompt tokens of 16 slots, a tree of 8
    nodes) draws its tokens as the plain walk does, one a block, and they equal greedy decoding's: no node
    is scored past the cache without its own key."""
    from dew.sampling import Sample, Sampling, generate

    model = _model("attention", 16)
    prompts = np.asarray([[*range(1, 13), 1], [*range(12, 0, -1), 2]], np.int32)
    params = jax.jit(model.init)(jax.random.key(0), jnp.asarray(prompts))
    sampling = Sampling(temperature=0)
    plain = generate(model, params, jnp.asarray(prompts), 3, key=jax.random.key(1), sampling=sampling,
                     strategy=Sample())
    drawn = _tree_generated(model, params, prompts, 3, sampling, TreeShape.widths(4, 1),
                            _ranked([(1, 2, 3, 0), (0,)]), jax.random.key(1))
    np.testing.assert_array_equal(np.asarray(drawn.tokens), np.asarray(plain.tokens)[:, 13:])


def test_generate_refuses_a_tree_for_a_model_without_a_tree_drafter():
    """`generate` under `TreeSpeculative` drafts with the model's tree drafter, and a model with none is
    refused rather than decoded another way."""
    from dew.sampling import Sampling, TreeSpeculative, generate

    model = _model("attention", 16)
    prompts = jnp.asarray([[1, 2, 3]], jnp.int32)
    params = jax.jit(model.init)(jax.random.key(0), prompts)
    with pytest.raises(ValueError):
        generate(model, params, prompts, 4, key=jax.random.key(1), sampling=Sampling(temperature=0),
                 strategy=TreeSpeculative(SHAPE))


def _sampled(sharpness: float):
    """A tree drafter drawing each node's children independently from the target's distribution after
    their parent's path sharpened by `sharpness` (off the target's law at any other value than 1), each
    with that distribution as its `q`."""
    from dew.sampling.trees import TreeDraft

    def drafting(ops):
        def draft(state, root, shape, keys):
            every = jnp.ones(root.shape[0], bool)
            held = [ops.advance(state, root, every)]
            tokens, log_q = [], []
            for node, parent in enumerate(shape.parents):
                logq = jax.nn.log_softmax(held[parent + 1].logits.astype(jnp.float32) * sharpness, axis=-1)
                token = jax.vmap(lambda key, row, node=node: jax.random.categorical(
                    jax.random.fold_in(key, node), row))(keys, logq).astype(jnp.int32)
                tokens.append(token)
                log_q.append(logq)
                held.append(ops.advance(held[parent + 1], token, every))
            return TreeDraft(jnp.stack(tokens, axis=1), jnp.stack(log_q, axis=1))

        return draft

    return drafting


@pytest.mark.parametrize("drafter", ["picked", "sampled"])
def test_a_trees_first_two_tokens_are_drawn_as_the_target_draws_them(drafter):
    """A tree changes how a token is drawn, never its law: a block's root and the token after it (a child
    the walk accepts by recursive rejection, or the residual's draw) are drawn as the target alone draws
    them, the pair at the chance the target's float64 run gives the first token times the second's given
    it, over every pair of the vocabulary (the sampling gate). The picked drafter offers the target's
    second- and third-ranked tokens (one-hot `q`), the sampled one draws children from a sharpened
    distribution (its own `q`), both off the target's law."""
    from reference_error import widened
    from sampling_gate import assert_draws_as_declared
    from test_speculative import LAW_ROWS, LAW_SEEDS, LAW_TEMPERATURE

    from dew.sampling import Sampling

    model = _model("attention", 16)
    prompt = jnp.asarray([[1, 2, 3]], jnp.int32)
    params = jax.jit(model.init)(jax.random.key(0), prompt)
    vocab = model.vocab_size
    with jax.enable_x64(new_val=True):
        wide, wide_params = model.clone(dtype=jnp.float64), widened(params)
        first = jax.nn.softmax(wide.apply(wide_params, prompt)[0, -1] / LAW_TEMPERATURE)
        pairs = jnp.concatenate([jnp.broadcast_to(prompt, (vocab, 3)), jnp.arange(vocab)[:, None]], axis=1)
        second = jax.nn.softmax(wide.apply(wide_params, pairs)[:, -1] / LAW_TEMPERATURE, axis=-1)
        chances = np.asarray(first[:, None] * second).reshape(-1)

    shape = TreeShape.widths(2, 1)
    drafting = _ranked([(1, 2), (0,)]) if drafter == "picked" else _sampled(3.0)
    rows = np.broadcast_to(np.asarray(prompt), (LAW_ROWS, 3))

    def counts(seed):
        drawn = _tree_generated(model, params, rows, 2, Sampling(temperature=LAW_TEMPERATURE), shape,
                                drafting, jax.random.key(seed))
        tokens = np.asarray(drawn.tokens)
        return np.bincount(tokens[:, 0] * vocab + tokens[:, 1], minlength=vocab * vocab)

    assert_draws_as_declared(counts, chances, f"a tree's first two tokens, {drafter} children",
                             seeds=LAW_SEEDS)


@pytest.mark.parametrize("case", ["rejected", "past-the-cache"])
def test_a_tree_block_carries_the_targets_state_at_each_rows_last_token(case):
    """A tree block leaves each row's target state (`DecoderState.hidden`, what a drafter reads next) at its
    last emitted token, as the walk a token at a time over the same tokens leaves it, check 1 against the
    walk's float64 run; a row the block left idle keeps its state. `rejected`: the walk rejects a child
    and accepts its sibling at the first two depths, then rejects the third's, four tokens emitted;
    `past-the-cache`: the cache holds no room for the tree and the row walks its one token."""
    from dew.sampling import Sampling, TreeSpeculative, decoding
    from dew.sampling.decoding import StepState
    from dew.sampling.text import resolve

    capacity, width = (40, 3) if case == "rejected" else (16, 13)
    model = _model("attention", capacity)
    params = jax.jit(model.init)(jax.random.key(0), jnp.ones((2, width), jnp.int32))
    emitted = 4 if case == "rejected" else 1

    def stepped(model, params):
        transforms, stopping, plan = resolve(Sampling(temperature=0), None, None, TreeSpeculative(SHAPE))
        ops = decode_ops(model, params, 0, 0)
        ops = dataclasses.replace(ops, draft_tree=_ranked([(1, 0), (1, 0), (1,)])(ops))

        @jax.jit
        def block(prompts):
            state, _ = prefill_state(model, params, ModelInputs(prompts), ops)
            rows = prompts.shape[0]
            # The second row idle this block.
            start = StepState(tokens=jnp.concatenate([prompts, jnp.zeros((rows, 8), jnp.int32)], axis=1),
                              valid=jnp.concatenate([jnp.ones((rows, width), bool),
                                                     jnp.zeros((rows, 8), bool)], axis=1),
                              step=jnp.zeros(rows, jnp.int32), active=jnp.asarray([True, False]),
                              keys=jax.random.split(jax.random.key(1), rows), prompt_width=width,
                              budget=jnp.full(rows, 8, jnp.int32))
            done = plan.stepped(state, start, jnp.zeros(rows, bool), ops, decoding.chain(transforms),
                                decoding.finish(stopping), 8)
            return done.state.hidden

        return block

    def walk(model, params):
        ops = decode_ops(model, params, 0, 0)

        @jax.jit
        def walked(prompts):
            state, _ = prefill_state(model, params, ModelInputs(prompts), ops)
            held = state
            for _ in range(emitted):
                held = ops.advance(held, jnp.argmax(held.logits, axis=-1).astype(jnp.int32),
                                   jnp.asarray([True, False]))
            return jnp.stack([held.hidden[0], state.hidden[1]])

        return walked

    with jax.enable_x64(new_val=True):
        wide = walk(model.clone(dtype=jnp.float64), widened(params))

    @as_given
    def truth(prompts):
        with jax.enable_x64(new_val=True):
            return np.asarray(wide(jnp.asarray(prompts)))

    def draw(seed):
        rng = np.random.default_rng([0x4527, 7, seed])
        return (rng.integers(1, model.vocab_size, (2, width)).astype(np.int32),)

    ours, reference = stepped(model, params), walk(model, params)
    error = checkify.checkify(ours, errors=checkify.user_checks)
    assert_as_exact(as_given(lambda prompts: np.asarray(error(jnp.asarray(prompts))[1])),
                    as_given(lambda prompts: np.asarray(reference(jnp.asarray(prompts)))), draw,
                    f"a tree block's carried state, {case}", seeds=SEEDS, truth_fn=truth)
