"""Real-token work rides the batch and the step's existing metric read.

The prefetch worker counts host rows before placement. One scalar per row
then follows the batch's row sharding, so summing it in the step gives the
global count even on a multi-process mesh. No tokens move back to the host
and no new device_get waits in the step loop. The log's existing counters
sum these scalars along with losses across the reporting interval.
"""

import jax
import jax.numpy as jnp
import numpy as np

from dew.batch import CLOSES_WINDOW, EXAMPLES_CONSUMED, VALID_ROWS, Batch
from dew.data.dataset import rows_of, take_batch
from dew.objectives.base import Objective
from dew.state import TokenCount

TRAINING_FLOPS = "__dew_training_flops"
TOKENS = "__dew_real_tokens"


def prepare_counts[Loss, Effects](objective: Objective[Loss, Effects], batch: Batch, *,
                                 count_model: bool) -> Batch:
    """Count the source's host rows, preserving its checkpoint and model inputs."""
    if not count_model and not objective.counts_tokens:
        return batch
    if any(name in batch for name in (TRAINING_FLOPS, TOKENS)):
        raise ValueError("the input batch contains reserved model-work telemetry columns")
    rows = rows_of(batch)
    counted = dict(batch)
    # A repeat that fills the batch (`VALID_ROWS`, `dew.data.dataset.FilledBatch`) is no work the run does
    # for its signal: it scores no target and counts no model work, a batch of repeats alone none.
    valid = batch.get(VALID_ROWS)
    real = np.ones(rows, bool) if valid is None else np.asarray(valid).reshape(rows) > 0
    if objective.counts_tokens:
        tokens = np.asarray(objective.scored_tokens(batch), np.uint64).reshape(rows)
        counted[TOKENS] = TokenCount.words(np.where(real, tokens, 0))
    if count_model:
        # Partition the assembled scalar over the batch's real rows. A shared conditioning row is counted
        # once, not once per example.
        held = batch if real.all() else take_batch(batch, np.flatnonzero(real))
        training = objective.training_flops(None, held) if real.any() else 0.0
        counted[TRAINING_FLOPS] = np.where(real, training / max(int(real.sum()), 1), 0).astype(np.float32)
    return counted


def uncounted(batch: Batch) -> Batch:
    """`batch` with its work unknown: NaN where counted work would be, so a
    step keeps its program and the global sum reports none (`step_counts`)."""
    rows = rows_of(batch)
    return {**batch, TRAINING_FLOPS: np.full((rows,), np.nan, np.float32)}


def step_counts(batch: Batch) -> dict[str, jax.Array]:
    """Global device-side sums, returned with the step's other scalar metrics."""
    work = ({TRAINING_FLOPS: jnp.sum(batch[TRAINING_FLOPS], dtype=jnp.float32)}
            if TRAINING_FLOPS in batch else {})
    if TOKENS in batch:
        work[TOKENS] = TokenCount.sum(batch[TOKENS])
    return work


def batch_tokens(batch: Batch) -> jax.Array:
    """The real tokens of `batch`'s rows (`TOKENS`), as two uint32 words; none where it carries no count."""
    return TokenCount.sum(batch[TOKENS]) if TOKENS in batch else jnp.zeros((2,), jnp.uint32)


def batch_consumed(batch: Batch) -> jax.Array:
    """The source examples `batch`'s rows consume (`EXAMPLES_CONSUMED`), summed over the whole batch
    wherever its rows lie, int32; none where it carries no count."""
    if EXAMPLES_CONSUMED not in batch:
        return jnp.zeros((), jnp.int32)
    return jnp.sum(jnp.asarray(batch[EXAMPLES_CONSUMED], jnp.int32), dtype=jnp.int32)


def model_batch(batch: Batch) -> Batch:
    """The objective sees only its own inputs, not the trainer's accounting columns."""
    return {name: value for name, value in batch.items()
            if name not in (TRAINING_FLOPS, TOKENS, EXAMPLES_CONSUMED, CLOSES_WINDOW)}
