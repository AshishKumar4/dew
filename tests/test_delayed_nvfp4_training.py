"""A delayed-scaling NVFP4 run's amax histories as its training steps move them (row 12721): each projection
the recipe takes reads its history and records its rows' largest magnitude a call; the trainer takes the
largest across a step's calls and microbatches and rolls it into the history once per committed update."""

import jax
import numpy as np
import optax

from dew.checkpoints import Checkpoints
from dew.data import Memory
from dew.data.pipeline import Pipeline
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.dense import NVFP4_HISTORY
from dew.nn.kernels import generation
from dew.nn.precision import DTYPE_PRESETS, LowPrecision, Numerics
from dew.objectives.lm import LMObjective
from dew.training import Trainer

SEQ = 16
DELAYED = Numerics(DTYPE_PRESETS["bf16-mixed"], low_precision=LowPrecision(recipe="nvfp4",
                                                                           tensor_scaling="delayed"))
"""A run's numerics in the delayed-scaling NVFP4 recipe, which takes the feed-forward's projections."""


def delayed_lm(split: int, checkpoints: Checkpoints | None = None) -> Trainer:
    """A two-layer LM whose feed-forward projections the delayed recipe takes, initialized under it (so its
    variables hold their histories), its batch of eight rows read in `split` microbatches."""
    model = CausalTransformer(vocab_size=64, emb_features=128, num_layers=2, num_heads=2, mlp_features=256,
                              max_seq_len=SEQ)
    return Trainer(LMObjective(model, seq_len=SEQ), optax.adamw(1e-3), key=0,
                   split=split, checkpoints=checkpoints)


def data() -> Pipeline:
    """Rows of eight, each step's its own."""
    text = (np.arange(64 * (SEQ + 1)).reshape(64, SEQ + 1) * 7 % 64).astype(np.int32)
    return Pipeline(Memory({"text": text})).batch(8)


def histories(state) -> list[np.ndarray]:
    """Each projection's amax history the state holds, in tree order."""
    return [np.asarray(leaf) for leaf in jax.tree.leaves(state.variables[NVFP4_HISTORY])]


def test_a_split_step_rolls_the_largest_its_microbatches_recorded_into_each_history_once_an_update():
    """A step read whole and the same step read in two microbatches roll the same largest magnitude into
    each history, newest first (the window's largest is the largest of its microbatches', so the history
    does not depend on the split), once an update: after one step each history holds one recorded value and
    zeros, and after the next, the first step's value moved one place on, behind the second's."""
    with generation.run_precision(DELAYED):
        whole, halves = delayed_lm(split=1), delayed_lm(split=2)
        started = histories(whole.place()[0])
        first = [whole.fit(data(), steps=1), halves.fit(data(), steps=1)]
        second = delayed_lm(split=2).fit(data(), steps=2)
    assert started and all(not history.any() for history in started), started
    for one, other in zip(histories(first[0]), histories(first[1]), strict=True):
        np.testing.assert_array_equal(one, other)
        assert one[0] > 0 and not one[1:].any(), one
    for after, before in zip(histories(second), histories(first[1]), strict=True):
        np.testing.assert_array_equal(after[1], before[0])
        assert after[0] > 0 and not after[2:].any(), after


def test_a_resumed_run_restores_its_amax_histories(tmp_path):
    """A delayed-scaling NVFP4 run's histories are state its checkpoint keeps: a run resumed from it reads
    them as the run that saved them held them, and its next step rolls them on."""
    with generation.run_precision(DELAYED):
        saved = delayed_lm(1, Checkpoints(str(tmp_path))).fit(data(), steps=2, checkpoint_every=1)
        fresh = delayed_lm(1, Checkpoints(str(tmp_path)))
        restored = fresh.place()[0]
        resumed = fresh.fit(data(), steps=3, checkpoint_every=1)
    for one, other in zip(histories(saved), histories(restored), strict=True):
        np.testing.assert_array_equal(one, other)
    for after, before in zip(histories(resumed), histories(saved), strict=True):
        np.testing.assert_array_equal(after[1:3], before[:2])
        assert after[0] > 0, after
