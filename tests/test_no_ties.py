"""The accurate form's ties claim (docs/design/testing.md, fast and accurate forms: it keeps apart every pair
of values the reference keeps apart), as the gate holds it (`reference_error.assert_merges_no_ties`): which
pairs it holds, and which the reference keeps apart."""

import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import as_given, assert_merges_no_ties

U = 2.0 ** -23
"""float32's ulp at one."""


def _held(mine, theirs, exact, rounded=None):
    """`assert_merges_no_ties` of fixed arms: Dew's values `mine` (float32), the reference's `theirs`, the
    float64 truth `exact` and, where given, the truth with the format's own roundings `rounded`, over one
    draw."""
    def draw(seed):
        return (np.zeros(1, np.float32),)

    assert_merges_no_ties(as_given(lambda _: np.asarray(mine, np.float32)),
                          as_given(lambda _: np.asarray(theirs, np.float64)),
                          lambda _: jnp.asarray(exact, jnp.float64), draw, "fixed arms", seeds=1,
                          rounded_fn=None if rounded is None else lambda _: jnp.asarray(rounded, jnp.float64))


def test_equal_entries_of_two_rows_are_no_tie():
    """Two batches' entries Dew returns equal by chance, their truths four ulps apart, which the reference
    keeps apart: no pair the claim holds, each row its own along its positions (row 6302: collisions across
    a leaf's batches, heads and channels, flattened, failed three accurate cases). One row's merges them."""
    _held([[1.0], [1.0]], [[1.0], [1.0 + 4 * U]], [[1.0], [1.0 + 4 * U]])
    with pytest.raises(AssertionError):
        _held([[1.0, 1.0]], [[1.0, 1.0 + 4 * U]], [[1.0, 1.0 + 4 * U]])


def test_a_pair_the_reference_holds_apart_by_its_own_rounding_is_kept_apart_by_none():
    """Two positions whose truths are 1.5 ulps apart, the reference's 2.1 (each 0.3 ulps off): two entries at
    the reference's accuracy may meet there, so Dew's tie merges nothing the reference keeps apart (row 6302:
    keys an ulp apart, whose float64 logits a correctly rounded fp32 logit merges, the reference's drift
    alone holding them apart). Truths four ulps apart, Dew's tie merges them."""
    _held([[1.0, 1.0]], [[1.0 - 0.3 * U, 1.0 + 1.8 * U]], [[1.0, 1.0 + 1.5 * U]])
    with pytest.raises(AssertionError):
        _held([[1.0, 1.0]], [[1.0 - 0.1 * U, 1.0 + 4.1 * U]], [[1.0, 1.0 + 4 * U]])


def test_a_pair_a_correctly_rounded_computation_merges_is_kept_apart_by_none():
    """Two positions whose truths are four ulps apart, the reference's 0.3 off each: kept apart by the
    reference, but their truths with the format's own roundings (a site's `rounded_fn`: an attention's
    logits correctly rounded) are one value, a tie any correctly rounded computation makes, so Dew's tie
    merges nothing (row 6302: keys an ulp apart whose float64 logits round to one fp32 logit, the
    reference's 0.55 ulps off splitting them). Rounded truths four ulps apart, it merges them."""
    _held([[1.0, 1.0]], [[1.0 + 0.3 * U, 1.0 + 3.7 * U]], [[1.0, 1.0 + 4 * U]], [[1.0, 1.0]])
    with pytest.raises(AssertionError):
        _held([[1.0, 1.0]], [[1.0 + 0.3 * U, 1.0 + 3.7 * U]], [[1.0, 1.0 + 4 * U]], [[1.0, 1.0 + 4 * U]])
