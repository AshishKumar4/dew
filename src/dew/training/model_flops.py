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
TOKENS = "__dew_real_tokens"


def prepare_counts[Loss, Effects](objective: Objective[Loss, Effects], batch: Batch, *,
                                 count_model: bool) -> Batch:
    """Count the source's host rows, preserving its checkpoint and model inputs."""
    if not count_model and not objective.counts_tokens:
        return batch
    if FORWARD_FLOPS in batch or TOKENS in batch:
        raise ValueError("the input batch contains reserved model-work telemetry columns")
    rows = rows_of(batch)
    forward, tokens = [], []
    for index in range(rows):
        row = jax.tree.map(lambda leaf, index=index: leaf[index:index + 1]
                           if np.ndim(leaf) and np.shape(leaf)[0] == rows else leaf, batch)
        if objective.counts_tokens:
            tokens.append(float(np.sum(objective.real_lengths(row), dtype=np.float64)))
        if count_model:
            forward.append(objective.model_flops(None, row))
    # The step's scalar telemetry accumulates in fp32, just like its loss;
    # widening before counting kept large integer products from overflowing.
    counted = dict(batch)
    if objective.counts_tokens:
        counted[TOKENS] = np.asarray(tokens, np.float32)
    if count_model:
        counted[FORWARD_FLOPS] = np.asarray(forward, np.float32)
    return counted


def step_counts(batch: Batch) -> dict[str, jax.Array]:
    """Global device-side sums, returned with the step's other scalar metrics."""
    return {name: jnp.sum(batch[name], dtype=jnp.float32)
            for name in (FORWARD_FLOPS, TOKENS) if name in batch}
