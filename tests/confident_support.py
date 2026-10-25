"""Confident-label rows and a label's loss derivative, for the categorical and head tests.

A label's loss derivative is Barron, Gnanapragasam and Tschannen's eq. 7 (arXiv:2610.31642): each
other class's probability, and the label's own entry the other classes' summed and negated, so every
row sums to 0. The tests hold each derivative to it from float64, a confident label's included.
"""

import numpy as np


def confident(margin: float, seed: int, label: int = 0, *, width: int = 7,
              count: int = 3) -> tuple[np.ndarray]:
    """`count` rows of `width` float32 logits whose `label` column leads the other classes' log
    partition by `margin` (arXiv:2610.31642, eq. 2)."""
    others = np.random.default_rng(seed).standard_normal((count, width - 1))
    peak = others.max(axis=-1, keepdims=True)
    lead = margin + peak + np.log(np.exp(others - peak).sum(axis=-1, keepdims=True))
    return (np.concatenate([others[:, :label], lead, others[:, label:]], axis=-1).astype(np.float32),)


def label_derivative(rows: np.ndarray, label: int = 0) -> np.ndarray:
    """The label loss's derivative by eq. 7 in NumPy, in the rows' own dtype, for the label at column
    `label`: each other class's `exp(z_j - z_label - softplus(-m))`, and the label's entry their sum's
    negative."""
    rest = np.delete(rows, label, axis=-1) - rows[:, label:label + 1]
    peak = rest.max(axis=-1, keepdims=True)
    margin = -(peak + np.log(np.exp(rest - peak).sum(axis=-1, keepdims=True)))
    others = np.exp(rest - np.logaddexp(0, -margin))
    summed = -others.sum(axis=-1, keepdims=True)
    return np.concatenate([others[:, :label], summed, others[:, label:]], axis=-1)
