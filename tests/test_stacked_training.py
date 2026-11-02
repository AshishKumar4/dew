"""A decoder's own feed-forward blocks stepped in context (dew.objectives.lm.stacking) where its store banks
its layers (each block read and written at its row of the bank, or at its position of a periodic run's,
`StackView.stored_layer`), and an objective that steps them training through the steps: the literal loop,
the decoder and the draws are test_level_stacking's."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact
from test_level_stacking import (
    LENGTH,
    ROWS,
    SCANNED,
    STACKED,
    STACKING_SEEDS,
    TRUTH,
    VOCAB,
    banked,
    chunk_loop,
    draw,
    stacked,
)

from dew.nn.backbones.decoder_stack import StackView
from dew.objectives.lm import LMObjective
from dew.objectives.lm.adaptation import ChunkAdaptation


@pytest.mark.parametrize(("layers", "windows"), [((1,), (3,)), ((0, 1), (3, 6)), ((1, 0), (6, 3))],
                         ids=["one-layer", "faster-first", "slower-first"])
def test_a_banked_stores_blocks_step_as_the_literal_chunk_loop(layers, windows):
    """A store that banks the decoder's two like layers as one run (`banked`, as a store built bank by bank
    holds them) steps its blocks in context at their own windows and rates as the literal chunk loop does
    (`chunk_loop`, check 1): each block read and written at its row of the bank (`StackView.stored_layer`),
    the bank's other rows as stored."""
    adaptation = ChunkAdaptation(window=windows, rate=(0.1, 0.05)[:len(layers)], layers=layers)

    def literal(reference, *, truth):
        return jax.jit(lambda variables, tokens: chunk_loop(reference, variables, tokens, adaptation,
                                                            truth=truth))

    assert_as_exact(jax.jit(stacked(adaptation, SCANNED, banked)), literal(STACKED, truth=False), draw,
                    f"banked stacked blocks of layers {layers} at windows {windows}",
                    seeds=STACKING_SEEDS[layers, windows], truth_fn=literal(TRUTH, truth=True))


def test_a_training_step_continues_pretraining_through_the_blocks_steps():
    """A training step's loss (`LMObjective.loss`, its summed head and z-loss) of a decoder stepping its
    own blocks in context is finite and moves the blocks' starts, as continued pre-training trains them."""
    from dew.objectives.base import Step

    objective = LMObjective(STACKED, LENGTH, adaptation=ChunkAdaptation(window=(3, 6), rate=0.05,
                                                                        layers=(0, 1)), z_loss=1e-4)
    variables = objective.init(jax.random.key(0))
    tokens = jnp.asarray(np.random.default_rng(0).integers(0, VOCAB, (ROWS, LENGTH + 1)), jnp.int32)

    def loss(params):
        step = Step(jnp.asarray(0), jax.random.key(0), None)
        statistics, _ = objective.loss(params, {"text": tokens}, step)
        return statistics.mean()[0]

    value, gradient = jax.value_and_grad(loss)(variables)
    assert np.isfinite(float(value))
    blocks = gradient["params"]["layers_0"]["mlp"]
    assert any(float(jnp.abs(leaf).max()) > 0 for leaf in jax.tree.leaves(blocks))


def test_a_periodic_runs_layer_is_read_and_written_at_its_position():
    """A store a periodic run's positions stack (`StackView.stack` over a period of two) reads each layer's
    block at its position's repeat (`StackView.stored_layer`), and a block written there
    (`with_stored_layer`) is that layer's alone once the run is unstacked, every other layer as stored."""
    view = StackView(((0, 4),), periods=(2,))
    layers = {f"layers_{index}": {"mlp": {"kernel": jnp.full((2, 3), index, jnp.float32)},
                                  "norm": {"scale": jnp.full(3, -index, jnp.float32)}} for index in range(4)}
    stored = view.stack({"params": layers})["params"]
    for index in range(4):
        np.testing.assert_array_equal(view.stored_layer(stored, index, "mlp")["kernel"],
                                      layers[f"layers_{index}"]["mlp"]["kernel"])
    written = view.with_stored_layer(stored, 3, "mlp", {"kernel": jnp.full((2, 3), 9.0)})
    unstacked = view.unstack({"params": written})["params"]
    for index in range(4):
        kernel = 9.0 if index == 3 else index
        np.testing.assert_array_equal(unstacked[f"layers_{index}"]["mlp"]["kernel"], jnp.full((2, 3), kernel))
        np.testing.assert_array_equal(unstacked[f"layers_{index}"]["norm"]["scale"],
                                      layers[f"layers_{index}"]["norm"]["scale"])
