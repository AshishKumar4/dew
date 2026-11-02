"""A decoder's own feed-forward blocks stepped in context (dew.objectives.lm.stacking) over rows whose
attention reads blocks whole, as a prefix run's does: each block read in one call by the chunk it starts in,
before the cache holds it. The literal loop, the decoders and the draws are test_level_stacking's."""

import jax
import pytest
from reference_error import assert_as_exact
from test_level_stacking import (
    BLOCKED,
    BLOCKED_SCANNED,
    BLOCKED_TRUTH,
    STACKING_SEEDS,
    banked,
    blocked_draw,
    chunk_loop,
    stacked,
    token_losses,
)

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
