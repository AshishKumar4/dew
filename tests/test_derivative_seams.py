"""The derivative contract's own validation (tests/contract_support.py's `assert_contract`). Every custom
derivative of Dew's declares the contract that holds it at its seam (`dew.nn.derivatives.contracted`, which
tools/lint_slop.py's SLOP028 checks), each one of tests/test_contracts_*.py."""

import jax.numpy as jnp
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
