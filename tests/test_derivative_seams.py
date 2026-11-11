"""The derivative contract's own validation (tests/contract_support.py's `assert_contract`). Every custom
derivative of Dew's declares the contract that holds it at its seam (`dew.nn.derivatives.contracted`, which
tools/lint_slop.py's SLOP028 checks), each one of tests/test_contracts_*.py."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from contract_support import LessFinite, assert_contract


def test_the_contract_refuses_a_formula_less_finite_than_the_plain_one():
    """The rule's validation: a formula overflowing where the plain one does not (|x| as the root of its
    square, past 2**64) is refused, at its primal and its derivatives; the plain formula against itself
    holds, at the same edge, where float32's own arithmetic holds every value."""
    x = (jnp.asarray([2.0**70, -3.0, 2.0**-70], jnp.float32),)
    direction = (jnp.ones((3,), jnp.float32),)
    cotangent = jnp.ones((3,), jnp.float32)
    assert_contract(jnp.abs, x, direction, cotangent, "the plain formula", plain_fn=jnp.abs)
    with pytest.raises(LessFinite):
        assert_contract(lambda x: jnp.sqrt(x * x), x, direction, cotangent, "a squared root",
                        plain_fn=jnp.abs)


def test_the_contract_refuses_a_bfloat16_formula_less_finite_than_the_plain_one():
    """The same refusal where the formula's values are bfloat16 (row 4499): the squared root overflows
    bfloat16's range where |x| does not, and its leaves are held as a float32 formula's are."""
    x = (jnp.asarray([2.0**70, -3.0, 2.0**-70], jnp.bfloat16),)
    direction = (jnp.ones((3,), jnp.bfloat16),)
    cotangent = jnp.ones((3,), jnp.bfloat16)
    with pytest.raises(LessFinite):
        assert_contract(lambda x: jnp.sqrt(x * x), x, direction, cotangent, "a bf16 squared root",
                        plain_fn=jnp.abs)


def test_the_contract_holds_a_sum_only_where_no_order_of_it_can_overflow():
    """A sum is held finite where the magnitudes of its terms sum inside float32's range, whatever their
    order: a form that sums before the scale that brings it back overflows where the plain formula, scaling
    each term first, does not, and is refused (row 1768's class). Where float32 overflows the terms in one
    order and not another (terms near the largest that cancel), no order is held: a form that sums the
    alternate terms first, which overflows where the plain formula's pairs cancel, holds the contract (row
    5426, a blocked sum beside a chain)."""
    direction, cotangent = (jnp.ones((8,), jnp.float32),), jnp.ones((), jnp.float32)

    def scaled_first(x):
        return jnp.sum(x * jnp.float32(2.0**-10))

    alike = (jnp.full((8,), 2.0**126, jnp.float32),)
    assert_contract(scaled_first, alike, direction, cotangent, "terms scaled first", plain_fn=scaled_first)
    with pytest.raises(LessFinite):
        assert_contract(lambda x: jnp.sum(x) * jnp.float32(2.0**-10), alike, direction, cotangent,
                        "a sum scaled after", plain_fn=scaled_first)
    cancelling = (jnp.asarray([2.0**127, -(2.0**127)] * 4, jnp.float32),)
    assert_contract(lambda x: jnp.sum(x[::2]) + jnp.sum(x[1::2]), cancelling, direction, cotangent,
                    "alternate terms first", plain_fn=lambda x: jnp.sum(jnp.sum(x.reshape(4, 2), axis=1)))


def test_the_contract_holds_terms_that_cancel_inside_the_range_in_every_order():
    """Of 2^126 twice and its negative twice every partial sum, in every order, stays within 2^127, inside
    float32, though their magnitudes sum to 2^128. The position is held: the plain sum is finite there, and
    a form that scales its terms up before it sums them and back down after overflows and is refused.
    Judged by the magnitudes' sum the position was dropped, and that form passed (row 5426)."""
    x = (jnp.asarray([2.0**126, 2.0**126, -(2.0**126), -(2.0**126)], jnp.float32),)
    direction, cotangent = (jnp.ones((4,), jnp.float32),), jnp.ones((), jnp.float32)
    assert_contract(jnp.sum, x, direction, cotangent, "the plain sum", plain_fn=jnp.sum)
    with pytest.raises(LessFinite):
        assert_contract(lambda x: jnp.sum(x * 4) / 4, x, direction, cotangent, "terms scaled up first",
                        plain_fn=jnp.sum)


def test_the_contract_reads_a_contraction_behind_a_mask_by_its_own_terms():
    """A plain formula that selects each row's product by a mask sums its groups last, the contraction
    behind that sum: the position is held by the contraction's own terms. Where they cancel past float32's
    range, a form that sums them in another order than the formula's and overflows is held to neither (a
    grouped GEMM's second derivative near float32's largest, row 5426)."""
    groups = np.asarray([0, 1])
    rows = (jnp.asarray([[2.0**64, -(2.0**64)] * 4] * 2, jnp.float32),)
    weights = jnp.full((2, 8, 1), 2.0**63, jnp.float32)
    direction, cotangent = (jnp.zeros((2, 8), jnp.float32),), jnp.ones((2, 1), jnp.float32)

    def masked(rows):
        pairs = jnp.einsum("rpk,gpkn->rgpn", rows.reshape(2, 4, 2), weights.reshape(2, 4, 2, 1))
        return jnp.einsum("rg,rgn->rn", jax.nn.one_hot(groups, 2, dtype=rows.dtype), pairs.sum(axis=2))

    def alternate_first(rows):
        chosen = weights[groups]
        return (jnp.einsum("rk,rkn->rn", rows[:, ::2], chosen[:, ::2])
                + jnp.einsum("rk,rkn->rn", rows[:, 1::2], chosen[:, 1::2]))

    assert_contract(alternate_first, rows, direction, cotangent, "alternate terms of a selected product",
                    truth_fn=masked, plain_fn=masked)
