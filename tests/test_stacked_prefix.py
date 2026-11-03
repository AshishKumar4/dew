"""A prefix run (`LMObjective(prefix=...)`) whose decoder steps its own feed-forward blocks in context
(dew.objectives.lm.stacking): its prefix read whole across the chunks it spans. The decoder reading blocks and
the draws are test_level_stacking's."""

import jax
import jax.numpy as jnp
import numpy as np
from reference_error import assert_as_exact
from test_level_stacking import (
    BLOCKED,
    BLOCKED_TRUTH,
    LENGTH,
    ROWS,
    STACKING_SEEDS,
    VOCAB,
    draw,
    token_losses,
)

from dew.objectives.lm import LMObjective
from dew.objectives.lm.adaptation import ChunkAdaptation


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
