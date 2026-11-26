"""Support for the delayed-scaling NVFP4 run witnesses (row 12721): an LM whose feed-forward projections the
recipe takes, its rows, and the histories its state holds."""

import jax
import numpy as np
import optax

from dew.checkpoints import Checkpoints
from dew.data import Memory
from dew.data.pipeline import Pipeline
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.dense import NVFP4_HISTORY
from dew.nn.precision import DTYPE_PRESETS, LowPrecision, Numerics
from dew.objectives.lm import LMObjective
from dew.training import Trainer

SEQ = 16
DELAYED = Numerics(DTYPE_PRESETS["bf16-mixed"], low_precision=LowPrecision(recipe="nvfp4",
                                                                           tensor_scaling="delayed"))
"""A run's numerics in the delayed-scaling NVFP4 recipe, which takes the feed-forward's projections."""


class Referenced(LMObjective):
    """An LM whose average is a frozen reference, as a preference or GRPO objective's is
    (`Objective.ema_is_reference`)."""

    ema_is_reference = True


def delayed_lm(split: int, checkpoints: Checkpoints | None = None, ema_decay: float | None = None, *,
               optimizer: optax.GradientTransformation | None = None,
               objective: type[LMObjective] = LMObjective) -> Trainer:
    """A two-layer LM whose feed-forward projections the delayed recipe takes, initialized under it (so its
    variables hold their histories), its batch of sixteen rows read in `split` microbatches, keeping an
    average of its weights at `ema_decay` where given, by `optimizer` (AdamW where none is given) and as
    `objective` (an `LMObjective` class)."""
    model = CausalTransformer(vocab_size=64, emb_features=128, num_layers=2, num_heads=2, mlp_features=256,
                              max_seq_len=SEQ)
    return Trainer(objective(model, seq_len=SEQ, ema_decay=ema_decay), optimizer or optax.adamw(1e-3),
                   key=0, split=split, checkpoints=checkpoints)


def data(rows: int = 16) -> Pipeline:
    """Batches of `rows`, sixteen by default, each step's its own: every microbatch of a split in two whole
    rows on each of a CPU mesh's eight data shards."""
    text = (np.arange(128 * (SEQ + 1)).reshape(128, SEQ + 1) * 7 % 64).astype(np.int32)
    return Pipeline(Memory({"text": text})).batch(rows)


def histories(state) -> list[np.ndarray]:
    """Each projection's amax history the state holds, in tree order."""
    return [np.asarray(leaf) for leaf in jax.tree.leaves(state.variables[NVFP4_HISTORY])]


def rolled_once(state) -> None:
    """Every amax history `state` holds took a recorded largest magnitude newest: each projection the step
    ran was recorded at its history's path, none left at its start."""
    held = histories(state)
    assert held and all(history[..., 0].min() > 0 and not history[..., 1:].any() for history in held), held
