"""The float64 gate's arithmetic on a kernel's graph (src/dew/nn/kernels/graph.py): each element's
distance from float64 with the node's sentinel and NaN rules, and the permutation of a node's summed
axis that is its orbit (tests/reference_ensemble.py `assert_graph_as_exact`).

No asserted test relies on the gate's attention key-sum nodes (a denominator or an accumulator summed
over keys) until ledger row 758 closes: equally exact key sums are flagged at every stage on some
workers (2 of 768 elements at 48 keys on an x86-64-v3 worker, none on x86-64-v4), the rate and its
root cause not yet measured per worker ISA."""

from collections.abc import Mapping

import numpy as np


def errors(value, truth, sentinel: float | None) -> np.ndarray:
    """|value - truth| in float64, zero where both hold the node's `sentinel` and NaN (a failure)
    wherever either is not finite otherwise."""
    value, truth = np.asarray(value, np.float64), np.asarray(truth, np.float64)
    held = np.zeros(value.shape, bool) if sentinel is None else (value == sentinel) & (truth == sentinel)
    finite = np.isfinite(value) & np.isfinite(truth)
    distance = np.abs(np.where(finite, value, 0.0) - np.where(finite, truth, 0.0))
    return np.where(held, 0.0, np.where(finite, distance, np.nan))


def permuted(values: Mapping[str, object], axes: Mapping[str, int], order: np.ndarray) -> dict:
    """`values` with each named parent's summed axis in `order`, the others as they are."""
    return {name: (np.take(np.asarray(value), order, axis=axes[name]) if name in axes else value)
            for name, value in values.items()}
