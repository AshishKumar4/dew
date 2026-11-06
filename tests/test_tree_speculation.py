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


@pytest.mark.parametrize("kind", ["attention", "mla"])
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
        state, step = prefill_state(model, params, ModelInputs(prompts), ops)
        total = step.total()
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

    assert_as_exact(as_given(lambda prompts, fed: np.asarray(verified(jnp.asarray(prompts), jnp.asarray(fed)))),
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
