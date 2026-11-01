"""nGPT's join and normed hidden reversed in two passes over their rows (ledger 1308): `ngpt.stepped` and
`ngpt.normed_hidden` keep each unit norm's sum of squares from the forward and take every inner product
the reverse reads in one pass, where JAX's own transpose of the formula takes each norm apart. Each
gradient is held on par with JAX's own gradient of the formula (check 1, truth in float64), in float32
and in the bf16 a model trains its stream in; the seams' every-order contracts are
tests/test_contracts_norms.py's."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact
from test_contracts_norms import HIDDEN, STEP, plain_hidden, plain_stepped

from dew.nn import ngpt
from dew.nn.activations import gated_product

SEEDS = {"join": 16, "hidden": 16}
"""Declared seeds for check 1."""

ROWS, WIDTH = (2, 8), 64


def drawn(seed: int, dtype, shapes) -> tuple:
    """Rows of `shapes` drawn from `seed` as a stream holds them, in `dtype`, a unit cotangent's row each,
    and gains near the initializer's scale."""
    rng = np.random.default_rng(seed)
    return tuple(jnp.asarray(rng.standard_normal(shape) * (0.1 if len(shape) == 1 else 1.0), dtype)
                 for shape in shapes)


def gradients(fn):
    """The gradient of `fn`'s output against a cotangent, in every operand."""
    def pulled(*operands_and_cotangent):
        *operands, cotangent = operands_and_cotangent
        _, pull = jax.vjp(fn, *operands)
        return pull(cotangent.astype(fn(*operands).dtype))
    return pulled


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["float32", "bf16"])
def test_ngpts_join_reverses_as_its_formula_does(dtype):
    """The join's gradient in the residual, the branch and the rate (`ngpt.stepped`) is on par with JAX's
    gradient of the formula (`plain_stepped`) over the draws."""
    def draw(seed):
        residual, branch, cotangent = drawn(seed, dtype, ((*ROWS, WIDTH),) * 3)
        (alpha,) = drawn(seed + 1000, jnp.float32, ((WIDTH,),))
        return residual, branch, alpha, cotangent

    def joined(residual, branch, alpha):
        return ngpt.stepped(residual, branch, alpha, STEP)

    assert_as_exact(jax.jit(gradients(joined)), jax.jit(gradients(plain_stepped)), draw,
                    f"nGPT's join in {jnp.dtype(dtype).name}", seeds=SEEDS["join"])


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["float32", "bf16"])
def test_ngpts_normed_hidden_reverses_as_its_formula_does(dtype):
    """The normed hidden's gradient in the gate, the up projection and both gains (`ngpt.normed_hidden`)
    is on par with JAX's gradient of the formula (`plain_hidden`) over the draws."""
    product = gated_product("swiglu", implementation="xla")

    def draw(seed):
        gate, up, cotangent = drawn(seed, dtype, ((*ROWS, WIDTH),) * 3)
        gate_gain, up_gain = drawn(seed + 1000, jnp.float32, ((WIDTH,), (WIDTH,)))
        return gate, up, gate_gain, up_gain, cotangent

    def hidden(gate, up, gate_gain, up_gain):
        return ngpt.normed_hidden(gate, up, gate_gain, up_gain, HIDDEN, product)

    assert_as_exact(jax.jit(gradients(hidden)), jax.jit(gradients(plain_hidden)), draw,
                    f"nGPT's normed hidden in {jnp.dtype(dtype).name}", seeds=SEEDS["hidden"])
