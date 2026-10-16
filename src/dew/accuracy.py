"""Distances and bounds between a computation and its float64 truth: the arithmetic the
accuracy rule (tests/reference_error.py), the reference runs' comparison
(tools/reference_runs/compare.py) and the kernel benchmarks share. Each keeps its own
acceptance policy, its factor and what it compares; only the arithmetic is one."""

import numpy as np


def rms(values) -> float:
    """The root mean square of `values`, in float64; NaN for none."""
    values = np.asarray(values, np.float64)
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else float("nan")


def distance(value, truth) -> float:
    """The root-mean-square difference over every entry, in float64."""
    return rms(np.asarray(value, np.float64) - np.asarray(truth, np.float64))


def within(error, bound) -> bool:
    """Whether every `error` is at most its `bound`, both finite. Written as the
    passing condition, so a NaN or infinite error never passes, nor does an
    infinite allowance from a reference that diverged."""
    error, bound = np.asarray(error, np.float64), np.asarray(bound, np.float64)
    return bool(np.all(np.isfinite(error) & np.isfinite(bound) & (error <= bound)))


def reduction_bound(magnitudes, terms, dtype=np.float32) -> np.ndarray:
    """How far two `dtype` reductions of the same products can differ: 2 gamma_n
    times `magnitudes`, the elementwise sum of the products' absolute values,
    where `terms` counts the products and any bias term in each reduction and
    gamma_n is n u / (1 - n u), u the unit roundoff (2^-24 for float32)."""
    roundoffs = np.asarray(terms, np.float64) * np.finfo(dtype).eps / 2
    return 2 * roundoffs / (1 - roundoffs) * np.asarray(magnitudes, np.float64)
