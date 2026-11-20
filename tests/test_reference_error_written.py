"""Check 1's accuracy claim held as written (`reference_error._written_twin`, docs/design/testing.md's check
1): Dew's arm against its reference compiled alike and against the same reference compiled as written,
without XLA's rewrites and fusions, so a rewrite both arms carry fails (row 6909)."""

import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import GateFailure, assert_as_exact


def _cancelling(seed):
    """Three outputs, +-2^127 (1 + u) and 3 2^-126 (1 + u'), of equal masses."""
    rng = np.random.default_rng(seed)
    large = np.float32(2.0 ** 127 * (1 + rng.uniform()))
    tiny = np.float32(3.0 * 2.0 ** -126 * (1 + rng.uniform()))
    return np.asarray([large, -large, tiny], np.float32), np.ones(3, np.float32)


def _combined(outputs, totals):
    """The outputs' convex combination by their masses."""
    return jnp.sum(totals / jnp.sum(totals) * outputs)[None, None]


def test_a_rewrite_both_arms_would_carry_fails_dews_arm_against_its_reference_as_written():
    """The convex combination of three equal masses whose outputs are +-2^127 (1 + u) and 3 2^-126 (1 + u'),
    one function as both arms: compiled with every XLA pass, as Dew's arm is, its fused sum takes each
    product unrounded beside its rounded opposite and leaves some 3.6e30 where the truth is 1.2e-38. On
    par with the reference compiled alike, both wrong (row 6909), Dew's arm fails against the reference
    compiled as written (`_written_twin`), which sums them as written, near the truth, by far more than
    PARITY to the power WRITTEN."""
    with pytest.raises(AssertionError) as raised:
        assert_as_exact(_combined, _combined, _cancelling, "a combination XLA's fusions rewrite", seeds=8)
    assert raised.type is GateFailure




def test_a_twins_products_are_rounded_as_written_its_fusions_kept():
    """`a * b + c` of `c` the product rounded and negated: compiled with every pass, a fusion sums the product
    unrounded beside `c` (XLA:CPU's fused multiply-add) and leaves its rounding error; compiled as a twin is,
    its fusions kept and its products held (`arm_arithmetic.products_held`), it sums the rounded product,
    zero at every element, as NumPy's float32 does."""
    import jax
    from arm_arithmetic import as_written, products_held

    rng = np.random.default_rng(6909)
    a, b = (rng.standard_normal(4096).astype(np.float32) for _ in range(2))
    c = -(a * b)

    def summed(a, b, c):
        return a * b + c

    assert np.count_nonzero(np.asarray(jax.jit(summed)(a, b, c))) > 0
    twin = jax.jit(products_held(summed), compiler_options=as_written())
    np.testing.assert_array_equal(np.asarray(twin(a, b, c)), np.zeros(4096, np.float32))


def test_a_reference_compiled_without_ynnpacks_fusions_is_held_as_written_too():
    """The combination as its reference compiled by its site without YNNPACK's fusions (`without_ynn`), a
    policy and not a claim: written apart from Dew, its own function is compiled as written too
    (`_written_twin`), and Dew's arm fails against it as against the plain function."""
    from reference_error import without_ynn

    with pytest.raises(GateFailure):
        assert_as_exact(_combined, without_ynn(_combined), _cancelling, "a combination XLA's fusions rewrite",
                        seeds=8)


def test_a_written_comparisons_bound_decides_its_own_second_stage():
    """A ratio between PARITY and its square (Dew's squared errors twice the reference's, a stable 2 in
    RMS ratio terms by its log) passes the comparison as written at its first stage, its factors squared
    (`needs_second_stage`'s and `_staged`'s `power`), where the plain comparison's bound draws the second
    stage: the written comparison's own bound decides its stage, never the plain one's (head review)."""
    from dew.parity import NO_ZERO_ROWS, PARITY, _Draws, _staged, needs_second_stage

    ratio = (PARITY + PARITY ** 2) / 2

    def drawn(start, stop):
        count = stop - start
        # Each seed's squared errors' logs: Dew's at `ratio` squared over the reference's one.
        return _Draws(["leaf"], np.full((count, 1), 2 * np.log(ratio)), np.zeros((count, 1)),
                      np.zeros((count, 1, 2)), [[NO_ZERO_ROWS]] * count, np.ones((count, 1), bool), (0,))

    assert needs_second_stage(drawn(0, 64), "plain") and not needs_second_stage(drawn(0, 64), "written",
                                                                                 power=2)
    assert _staged(drawn, 64, "written", power=2)[2] == 1

