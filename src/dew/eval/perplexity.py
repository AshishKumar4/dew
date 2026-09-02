"""Perplexity over a validation pass.

The objective supplies one teacher-forced cross entropy per batch. The metric
averages those cross-entropies before exponentiating them.
"""

import numpy as np

from .common import EvaluationMetric


def get_perplexity_metric(name: str = "perplexity") -> EvaluationMetric:
    """Perplexity of the teacher-forced predictions, lower is better."""
    return EvaluationMetric(
        function=lambda artifacts, batch: float(artifacts["ce"]),
        name=name,
        higher_is_better=False,
        reducer=lambda values: float(np.exp(np.mean(values))),
    )
