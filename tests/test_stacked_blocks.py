"""A decoder's own feed-forward blocks stepped in context (dew.objectives.lm.stacking) over rows whose
attention reads blocks whole, as a prefix run's does: each block read in one call by the chunk it starts in,
before the cache holds it. The literal loop, the decoders and the draws are test_level_stacking's."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact
from test_level_stacking import (
    BLOCKED,
    BLOCKED_SCANNED,
    BLOCKED_TRUTH,
    LENGTH,
    ROWS,
    STACKING_SEEDS,
    VOCAB,
    banked,
    blocked_draw,
    chunk_loop,
    draw,
    stacked,
    token_losses,
)

from dew.objectives.lm import LMObjective
from dew.objectives.lm.adaptation import ChunkAdaptation


def test_a_rows_blocks_at_rate_zero_score_as_its_forward_reads_them():
    """At rate zero a stacked pass over rows whose attention reads blocks whole (a prefix, and a later run,
    within one chunk or across bounds) scores each target as the decoder's own forward over the whole row
    and its blocks does (check 1): every block is read whole, by the chunk it starts in."""
    adaptation = ChunkAdaptation(window=3, rate=0.0, layers=(0, 1))

    def plain(model, *, truth):
        return jax.jit(lambda variables, tokens, blocks: token_losses(
            model.apply(variables, tokens[:, :-1], block_groups=blocks), tokens[:, 1:], truth=truth))

    assert_as_exact(jax.jit(stacked(adaptation, BLOCKED)), plain(BLOCKED, truth=False), blocked_draw,
                    "the stacked pass over blocks at rate zero", seeds=STACKING_SEEDS["zero"],
                    truth_fn=plain(BLOCKED_TRUTH, truth=True))


@pytest.mark.parametrize(("model", "store"), [(BLOCKED, None), (BLOCKED_SCANNED, banked)],
                         ids=["layers", "banked"])
def test_a_rows_blocks_step_as_the_literal_chunk_loop_reads_them(model, store):
    """A decoder's blocks step in context over rows whose attention reads blocks whole, each target's loss on
    par with the literal chunk loop's (`chunk_loop`, check 1), whose chunk holds a block's every token from
    the chunk its first token is in: a block's chunk reads it whole before the cache holds it, at that
    chunk's levels, and a chunk a block covers whole steps no level by any token of its own."""
    adaptation = ChunkAdaptation(window=(3, 6), rate=(0.1, 0.05), layers=(0, 1))

    def literal(reference, *, truth):
        return jax.jit(lambda variables, tokens, blocks: chunk_loop(reference, variables, tokens, adaptation,
                                                                    truth=truth, blocks=blocks))

    assert_as_exact(jax.jit(stacked(adaptation, model, store)), literal(BLOCKED, truth=False), blocked_draw,
                    "stacked blocks over a row's blocks", seeds=STACKING_SEEDS["blocks"],
                    truth_fn=literal(BLOCKED_TRUTH, truth=True))


def test_a_prefix_run_steps_its_blocks_in_context():
    """A prefix run (`LMObjective(prefix=...)`) whose decoder steps its own blocks in context scores at rate
    zero as the prefix run without them (check 1), its prefix read whole across the chunks it spans, and
    trains through their steps: its loss finite and its blocks' starts moved."""
    from dew.objectives.base import Step

    stepping = ChunkAdaptation(window=3, rate=0.0, layers=(0, 1))
    objective = LMObjective(BLOCKED, LENGTH, prefix=5, adaptation=stepping)
    plain = LMObjective(BLOCKED, LENGTH, prefix=5)

    def scored(scorer):
        def losses(variables, tokens):
            step = Step(jnp.asarray(0), jax.random.key(0), None)
            _, _, prediction = scorer.predict(variables, {"text": tokens}, step, train=False, logits=False)
            return prediction.losses * prediction.weights
        return jax.jit(losses)

    def truth(variables, tokens):
        blocks = jnp.where(jnp.arange(LENGTH) < 5, 1, -1)[None].repeat(ROWS, axis=0)
        logits = BLOCKED_TRUTH.apply(variables, tokens[:, :-1], block_groups=blocks)
        return token_losses(logits, tokens[:, 1:], truth=True) * (jnp.arange(1, LENGTH + 1) >= 5)

    assert_as_exact(scored(objective), scored(plain), lambda seed: draw(seed, model=BLOCKED),
                    "a prefix run's stacked pass at rate zero",
                    seeds=STACKING_SEEDS["zero"], truth_fn=jax.jit(truth))
    trained = LMObjective(BLOCKED, LENGTH, prefix=5, adaptation=ChunkAdaptation(window=(3, 6), rate=0.05,
                                                                               layers=(0, 1)))
    variables = trained.init(jax.random.key(0))
    tokens = jnp.asarray(np.random.default_rng(0).integers(0, VOCAB, (ROWS, LENGTH + 1)), jnp.int32)

    def loss(params):
        statistics, _ = trained.loss(params, {"text": tokens}, Step(jnp.asarray(0), jax.random.key(0), None))
        return statistics.mean()[0]

    value, gradient = jax.value_and_grad(loss)(variables)
    assert np.isfinite(float(value))
    blocks = gradient["params"]["layers_0"]["mlp"]
    assert any(float(jnp.abs(leaf).max()) > 0 for leaf in jax.tree.leaves(blocks))
