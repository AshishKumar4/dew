"""Projections joined per call (`dew.nn.joined`): one GEMM in place of a
layer's own projection calls, where a generation's kernel row measured it
faster."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference, equations, float64_terms

from dew.nn.kernels import generation
from dew.nn.linear import GatedDeltaNet
from dew.nn.mlp import GatedMLP


def row_joins(choice: str, *, joins: bool = True):
    """Choose joined projections for `choice` where `joins`, else separate projections."""
    return generation.chosen(**{choice: "joined" if joins else "apart"})


def dots(function, *args) -> int:
    """The dot products `function`'s jaxpr runs, its sub-jaxprs' included."""
    graph = jax.make_jaxpr(lambda *operands: function(*operands))(*args)
    return sum(equation.primitive.name == "dot_general" for equation in equations(graph))


def assert_joined_as_exact(choice: str, run, arguments, wide_run):
    """`run(*arguments)` with `choice` joined, every leaf within tests/reference_error.py's rule of
    the run with it apart, both measured from `wide_run`, the apart run in float64, each entry sized
    by its float64 terms: a weight's gradient sums every token's product, which cancel."""
    with row_joins(choice, joins=False):
        apart = jax.jit(lambda *operands: run(*operands))(*arguments)
    with row_joins(choice):
        together = jax.jit(lambda *operands: run(*operands))(*arguments)
    with row_joins(choice, joins=False), jax.enable_x64(new_val=True):
        truth, terms = float64_terms(wide_run, *jax.tree.map(
            lambda leaf: jnp.asarray(np.asarray(leaf, np.float64)), arguments))
    for (path, mine), theirs, want, size in zip(jax.tree_util.tree_leaves_with_path(together),
                                                jax.tree.leaves(apart), jax.tree.leaves(truth),
                                                jax.tree.leaves(terms), strict=True):
        assert mine.dtype == theirs.dtype
        assert_as_exact_as_the_reference(mine, theirs, want, jax.tree_util.keystr(path), terms=size)


def gradients(apply):
    """`apply`'s output and the gradients of its arguments under a cotangent, the last argument."""
    def run(*arguments):
        out, pull = jax.vjp(apply, *arguments[:-1])
        return out, *pull(arguments[-1])

    return run


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
def test_gate_and_up_joined_are_as_exact_as_apart_from_float64(dtype, bias):
    """A GatedMLP whose row joins gate and up computes both in one GEMM, two
    dots where it ran three, and its output and the gradients of its input
    and every parameter lie within tests/reference_error.py's rule of the
    layer's with them apart, both measured from float64."""
    x = jax.random.normal(jax.random.key(3), (2, 33, 64)).astype(dtype)
    cotangent = jax.random.normal(jax.random.key(5), (2, 33, 64)).astype(dtype)
    layer = GatedMLP(hidden_features=96, out_features=64, use_bias=bias, dtype=dtype)
    # Drawn biases too, which init leaves at zero.
    params = jax.tree.map(lambda leaf: leaf + 0.1 * jax.random.normal(jax.random.key(6), leaf.shape),
                          layer.init(jax.random.key(4), x))
    with row_joins("gate_up", joins=False):
        assert dots(layer.apply, params, x) == 3
    with row_joins("gate_up"):
        assert dots(layer.apply, params, x) == 2
    assert_joined_as_exact("gate_up", gradients(layer.apply), (params, x, cotangent),
                           gradients(layer.clone(dtype=jnp.float64).apply))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_a_delta_nets_gates_joined_are_as_exact_as_apart_from_float64(dtype):
    """A gated delta net whose row joins its b and a projections ('delta_gates') runs them as one
    GEMM, one dot fewer, and its output and the gradients of its input and every parameter lie within
    tests/reference_error.py's rule of the layer's with them apart, both measured from float64."""
    x = jax.random.normal(jax.random.key(7), (2, 40, 64)).astype(dtype)
    cotangent = jax.random.normal(jax.random.key(8), (2, 40, 64)).astype(dtype)
    layer = GatedDeltaNet(emb_features=64, num_k_heads=2, num_v_heads=4, head_k_dim=16, head_v_dim=16,
                          dtype=dtype)
    params = layer.init(jax.random.key(9), x)
    with row_joins("delta_gates", joins=False):
        apart = dots(layer.apply, params, x)
    with row_joins("delta_gates"):
        assert dots(layer.apply, params, x) == apart - 1
    assert_joined_as_exact("delta_gates", gradients(layer.apply), (params, x, cotangent),
                           gradients(layer.clone(dtype=jnp.float64).apply))
