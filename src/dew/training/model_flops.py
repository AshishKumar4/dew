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

from dew.batch import EXAMPLES_CONSUMED, VALID_ROWS, Batch
from dew.data.dataset import rows_of
from dew.objectives.base import Objective

TRAINING_FLOPS = "__dew_training_flops"
TOKENS = "__dew_real_tokens"


class _TokenCount:
    """Unsigned 64-bit counts as low/high uint32 words, without enabling float64 in JAX."""

    @staticmethod
    def words(values) -> np.ndarray:
        values = np.asarray(values, np.uint64)
        return np.stack((values.astype(np.uint32), (values >> np.uint64(32)).astype(np.uint32)), axis=-1)

    @staticmethod
    def _plus(left, right):
        low = left[0] + right[0]
        return low, left[1] + right[1] + (low < left[0]).astype(jnp.uint32)

    @classmethod
    def add(cls, left, right) -> jax.Array:
        return jnp.stack(cls._plus(left, right))

    @classmethod
    def sum(cls, rows: jax.Array) -> jax.Array:
        return jnp.stack(jax.lax.reduce((rows[:, 0], rows[:, 1]),
                                       (np.uint32(0), np.uint32(0)), cls._plus, (0,)))

    @staticmethod
    def host(words) -> int:
        words = np.asarray(words)
        return int(words[0]) + (int(words[1]) << 32)


def prepare_counts[Loss, Effects](objective: Objective[Loss, Effects], batch: Batch, *,
                                 count_model: bool) -> Batch:
    """Count the source's host rows, preserving its checkpoint and model inputs."""
    if not count_model and not objective.counts_tokens:
        return batch
    if any(name in batch for name in (TRAINING_FLOPS, TOKENS)):
        raise ValueError("the input batch contains reserved model-work telemetry columns")
    rows = rows_of(batch)
    counted = dict(batch)
    if objective.counts_tokens:
        # The targets each row's loss scores, none on a repeat that fills the batch (`VALID_ROWS`).
        tokens = np.asarray(objective.scored_tokens(batch), np.uint64).reshape(rows)
        valid = batch.get(VALID_ROWS)
        counted[TOKENS] = _TokenCount.words(tokens if valid is None else
                                            np.where(np.asarray(valid).reshape(rows) > 0, tokens, 0))
    if count_model:
        # Partition the assembled scalar over the batch's row shards. A
        # shared conditioning row is counted once, not once per example.
        training = objective.training_flops(None, batch)
        counted[TRAINING_FLOPS] = np.full((rows,), training / rows if rows else 0, np.float32)
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
        work[TOKENS] = _TokenCount.sum(batch[TOKENS])
    return work


def batch_tokens(batch: Batch) -> jax.Array:
    """The real tokens of `batch`'s rows (`TOKENS`), as two uint32 words; none where it carries no count."""
    return _TokenCount.sum(batch[TOKENS]) if TOKENS in batch else jnp.zeros((2,), jnp.uint32)


def batch_consumed(batch: Batch) -> jax.Array:
    """The source examples `batch`'s rows consume (`EXAMPLES_CONSUMED`), summed over the whole batch
    wherever its rows lie, int32; none where it carries no count."""
    if EXAMPLES_CONSUMED not in batch:
        return jnp.zeros((), jnp.int32)
    return jnp.sum(jnp.asarray(batch[EXAMPLES_CONSUMED], jnp.int32))


def model_batch(batch: Batch) -> Batch:
    """The objective sees only its own inputs, not the trainer's accounting columns."""
    return {name: value for name, value in batch.items()
            if name not in (TRAINING_FLOPS, TOKENS, EXAMPLES_CONSUMED)}
