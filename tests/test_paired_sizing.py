"""Check 1's paired sizing (tests/reference_error.py `seeds_for` with `dew_fn`): arms that share their
upstream rounding, each one's error heavy-tailed across draws, resolve paired where the reference
compared with itself does not, and a fault still fails at the count they declare."""

import numpy as np
import pytest
from reference_error import assert_as_exact, seeds_for


def _draw(seed):
    """Sums' terms scaled per draw by a heavy-tailed factor (a Pareto of index one, plus one)."""
    rng = np.random.default_rng([0x5E, seed])
    return ((rng.pareto(1.0) + 1) * rng.standard_normal((32, 128))).astype(np.float32),


def _summed(order):
    """The rows' float32 sums with their terms in `order`'s permutation, one rounding a term."""
    def summed(x):
        terms = np.asarray(x, np.float32)[:, np.random.default_rng(order).permutation(x.shape[1])]
        return {"sum": np.cumsum(terms, axis=1, dtype=np.float32)[:, -1]}
    return summed


def _truth(x):
    return {"sum": np.sum(np.asarray(x, np.float64), axis=1)}


def test_a_shared_heavy_tail_resolves_paired_and_not_against_itself():
    reference, twin = _summed(0), _summed(1)
    seeds = seeds_for(reference, _draw, "heavy sums", truth_fn=_truth, dew_fn=twin)
    with pytest.raises(AssertionError) as raised:
        seeds_for(reference, _draw, "heavy sums", truth_fn=_truth, most=seeds)
    assert raised.type is AssertionError
    assert_as_exact(twin, reference, _draw, "heavy sums", seeds=seeds, truth_fn=_truth)


def test_a_fault_fails_at_the_paired_count():
    """A sum in float32 whose every term is first rounded to bfloat16's precision fails at the count the
    equally exact pair declares."""
    import ml_dtypes

    reference, twin = _summed(0), _summed(1)
    seeds = seeds_for(reference, _draw, "heavy sums", truth_fn=_truth, dew_fn=twin)

    def coarse(x):
        return twin(np.asarray(x).astype(ml_dtypes.bfloat16).astype(np.float32))

    with pytest.raises(AssertionError) as raised:
        assert_as_exact(coarse, reference, _draw, "heavy sums", seeds=seeds, truth_fn=_truth)
    assert raised.type is AssertionError
