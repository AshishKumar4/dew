"""A leaf's own resolution, with cancellation reserved for declared invariants."""

import jax
import numpy as np
import pytest
from reference_error import (
    assert_as_exact_as_the_reference,
    assert_fp32_reduction_bound,
    assert_leaves_as_exact,
    assert_rounds_where_the_reference_does,
)


def test_deleting_a_small_representable_gradient_fails_beside_a_large_leaf():
    truth = {"large": np.ones(4, np.float64), "small": np.full(4, 1e-20, np.float64)}
    reference = jax.tree.map(lambda leaf: leaf.astype(np.float32), truth)
    assert_leaves_as_exact(reference, reference, truth, reductions=4)
    dew = {**reference, "small": np.zeros(4, np.float32)}
    with pytest.raises(AssertionError, match=r"\['small'\]: 1.000e-20 from float64"):
        assert_leaves_as_exact(dew, reference, truth, reductions=4)


def test_a_declared_invariant_accepts_float32_cancellation_noise():
    truth = {"bias": np.asarray([np.finfo(np.float64).eps / 4])}
    reference = {"bias": np.zeros(1, np.float32)}
    dew = {"bias": np.asarray([np.finfo(np.float32).eps], np.float32)}
    assert_leaves_as_exact(dew, reference, truth, reductions=4, invariants={"['bias']": 1.0})
    with pytest.raises(AssertionError, match=r"\['bias'\]: .*from float64"):
        assert_leaves_as_exact(dew, reference, truth, reductions=4)


def test_a_nonzero_leaf_cannot_be_declared_invariant():
    truth = {"small": np.full(4, 1e-20, np.float64)}
    reference = jax.tree.map(lambda leaf: leaf.astype(np.float32), truth)
    with pytest.raises(AssertionError, match=r"\['small'\]: declared invariant has float64 RMS"):
        assert_leaves_as_exact(reference, reference, truth, reductions=4, invariants={"['small']": 1e-20})


def test_an_absent_invariant_path_fails_visibly():
    tree = {"bias": np.zeros(1, np.float32)}
    with pytest.raises(AssertionError, match="declared invariant leaves absent from the tree"):
        assert_leaves_as_exact(tree, tree, tree, invariants={"['bais']": 1.0})


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_a_non_finite_error_or_allowance_never_passes(bad):
    """Each check is its passing condition, so NaN, which compares false either way, fails."""
    truth = np.ones(4, np.float64)
    reference = (truth + 1e-7).astype(np.float32)
    broken = reference.copy()
    broken[1] = bad
    with pytest.raises(AssertionError, match=r"\['leaf'\]"):
        assert_leaves_as_exact({"leaf": broken}, {"leaf": reference}, {"leaf": truth})
    with pytest.raises(AssertionError, match=r"\['leaf'\]"):
        assert_leaves_as_exact({"leaf": reference}, {"leaf": broken}, {"leaf": truth})
    with pytest.raises(AssertionError):
        assert_as_exact_as_the_reference(broken, reference, truth, "leaf")
    with pytest.raises(AssertionError):
        assert_rounds_where_the_reference_does(broken, reference, truth, "leaf")
    with pytest.raises(AssertionError):
        assert_fp32_reduction_bound(broken, reference, np.ones(4), terms=4)
