"""Perplexity, as an evaluation metric over a language model's artifacts.

The objective already computed the cross entropy the trainer needs for its
loss curve, so the metric reads it back out of the validation artifacts rather
than running the model a second time. The validation loop averages a metric
over the epoch's batches, which makes this the mean of the per-batch
perplexities rather than the exponential of the mean cross entropy; the two
agree closely once the batches are the same length, which packed token files
guarantee.
"""

import jax.numpy as jnp

from .common import EvaluationMetric


def get_perplexity_metric(name: str = "perplexity") -> EvaluationMetric:
    """Perplexity of the teacher-forced predictions, lower is better."""
    return EvaluationMetric(
        function=lambda artifacts, batch: float(jnp.exp(artifacts["ce"])),
        name=name,
        higher_is_better=False,
    )
