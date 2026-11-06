"""Tree speculative decoding (ledger 4527): a tree verified in one pass, its acceptance and its draws."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model_support import decoder
from reference_error import as_given, assert_as_exact, widened

from dew.nn.inputs import ModelInputs
from dew.sampling.text import decode_ops, prefill_state
from dew.sampling.trees import TreeBlock, TreeShape

SHAPE = TreeShape.widths(2, 2, 1)
"""Two children of the root, two of each, one of each of those: 2 + 4 + 4 nodes, three deep."""
SEEDS = 8
"""The tree verify's declared draws of prompts and trees (check 1)."""


@pytest.mark.parametrize("kind", ["attention"])
def test_a_trees_nodes_verified_in_one_pass_score_as_their_paths_walked(kind):
    """Each node of a tree verified in one pass (`TreeBlock`: every node over the row's prefix and its own
    ancestors, at its depth's place) scores the token after it as the walk over its path from the root does,
    a token at a time (`decode_ops`): each node's logits as near the walk's float64 run as the walk's own
    (check 1, over drawn prompts and trees)."""
    model = decoder(kind).clone(max_seq_len=24)
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
    import dataclasses

    from dew.sampling import TreeSpeculative, decoding
    from dew.sampling.decoding import StepState
    from dew.sampling.text import resolve

    transforms, stopping, strategy = resolve(sampling, None, None, TreeSpeculative(shape))
    ops = decode_ops(model, params, 0, 0)
    ops = dataclasses.replace(ops, draft_tree=drafting(ops))

    @jax.jit
    def run(prompts, key):
        state, real = prefill_state(model, params, ModelInputs(prompts), ops)
        rows, width = prompts.shape
        start = StepState(tokens=jnp.concatenate([prompts, jnp.zeros((rows, budget), jnp.int32)], axis=1),
                          valid=jnp.concatenate([jnp.ones((rows, width), bool),
                                                 jnp.zeros((rows, budget), bool)], axis=1),
                          step=jnp.zeros(rows, jnp.int32), active=real, keys=jax.random.split(key, rows),
                          prompt_width=width, budget=jnp.full(rows, budget, jnp.int32))
        return strategy(state, start, ops, decoding.chain(transforms), decoding.finish(stopping), budget, 1)

    return run(jnp.asarray(prompts), key)


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


@pytest.mark.parametrize("kind", ["attention"])
def test_tree_speculation_reproduces_the_greedy_walk(kind):
    """At zero temperature the target is a point mass, so a tree's walk keeps exactly the path of greedy
    picks it holds and emits the target's next pick after it: the whole run equals the greedy walk. The
    drafter ranks its first child second and its second first at the first two depths, and offers only the
    second-ranked at the third, so the walk rejects a child, accepts its sibling, and ends on a rejection
    with the target's own token."""
    from test_speculative import greedy_walk

    from dew.sampling import Sampling

    model = decoder(kind).clone(max_seq_len=40)
    prompts = np.asarray([[1, 2, 3], [7, 8, 9]], np.int32)
    params = jax.jit(model.init)(jax.random.key(0), jnp.asarray(prompts))
    drawn = _tree_generated(model, params, prompts, 8, Sampling(temperature=0), SHAPE,
                            _ranked([(1, 0), (1, 0), (1,)]), jax.random.key(1))
    walked = greedy_walk(model, params, jnp.asarray(prompts), 8)
    np.testing.assert_array_equal(np.asarray(drawn.tokens), walked)
    np.testing.assert_array_equal(np.asarray(drawn.valid), True)


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
                token = jax.vmap(lambda key, row: jax.random.categorical(jax.random.fold_in(key, node), row))(
                    keys, logq).astype(jnp.int32)
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

    model = decoder("attention").clone(max_seq_len=16)
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
