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

from dew.data.dataset import rows_of
from dew.objectives.base import Batch, Objective

FORWARD_FLOPS = "__dew_forward_flops"
TRAINING_FLOPS = "__dew_training_flops"
TOKENS = "__dew_real_tokens"


def prepare_counts[Loss, Effects](objective: Objective[Loss, Effects], batch: Batch, *,
                                 count_model: bool) -> Batch:
    """Count the source's host rows, preserving its checkpoint and model inputs."""
    if not count_model and not objective.counts_tokens:
        return batch
    if any(name in batch for name in (FORWARD_FLOPS, TRAINING_FLOPS, TOKENS)):
        raise ValueError("the input batch contains reserved model-work telemetry columns")
    rows = rows_of(batch)
    tokens = []
    for index in range(rows):
        row = jax.tree.map(lambda leaf, index=index: leaf[index:index + 1]
                           if np.ndim(leaf) and np.shape(leaf)[0] == rows else leaf, batch)
        if objective.counts_tokens:
            tokens.append(float(np.sum(objective.real_lengths(row), dtype=np.float64)))
    # The step's scalar telemetry accumulates in fp32, just like its loss;
    # widening before counting kept large integer products from overflowing.
    counted = dict(batch)
    if objective.counts_tokens:
        counted[TOKENS] = np.asarray(tokens, np.float32)
    if count_model:
        # Partition each assembled scalar over the batch's row shards. A
        # shared conditioning row is counted once, not once per example.
        forward = objective.model_flops(None, batch)
        training = objective.training_flops(None, batch)
        counted[FORWARD_FLOPS] = np.full((rows,), forward / rows if rows else 0, np.float32)
        counted[TRAINING_FLOPS] = np.full((rows,), training / rows if rows else 0, np.float32)
    return counted


def uncounted(batch: Batch) -> Batch:
    """`batch` with its work unknown: NaN where counted work would be, so a
    step keeps its program and the global sum reports none (`step_counts`)."""
    rows = rows_of(batch)
    return {**batch, FORWARD_FLOPS: np.full((rows,), np.nan, np.float32),
            TRAINING_FLOPS: np.full((rows,), np.nan, np.float32)}


def step_counts(batch: Batch) -> dict[str, jax.Array]:
    """Global device-side sums, returned with the step's other scalar metrics."""
    return {name: jnp.sum(batch[name], dtype=jnp.float32)
            for name in (FORWARD_FLOPS, TRAINING_FLOPS, TOKENS) if name in batch}


def model_batch(batch: Batch) -> Batch:
    """The objective sees only its own inputs, not the trainer's accounting columns."""
    return {name: value for name, value in batch.items()
            if name not in (FORWARD_FLOPS, TRAINING_FLOPS, TOKENS)}
