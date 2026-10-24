"""Projections joined per call (`dew.nn.joined`): one GEMM in place of a
layer's own projection calls, where a generation's kernel row measured it
faster."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from reference_error import assert_as_exact_as_the_reference, equations, float64_terms, with_gradients

from dew.nn.dense import Dense
from dew.nn.joined import joined_dense
from dew.nn.kernels import generation
from dew.nn.linear import GatedDeltaNet
from dew.nn.mlp import GatedMLP


def row_joins(choice: str, *, joins: bool = True):
    """Choose joined projections for `choice` where `joins`, else separate projections."""
    return generation.chosen(**{choice: "joined" if joins else "apart"})


def dots(function, *args) -> int:
    """The dot products an accelerator's run of `function` executes, its sub-jaxprs' included: of each
    platform choice, the model's own products, where a CPU sums them in float64 and spreads a bias
    by a product (`dew.nn.precision`, tests/test_cpu_accumulation.py)."""
    graph = jax.make_jaxpr(lambda *operands: function(*operands))(*args)
    return sum(equation.primitive.name == "dot_general" for equation in equations(graph, platform="cuda"))


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
    assert_joined_as_exact("gate_up", with_gradients(layer.apply), (params, x, cotangent),
                           with_gradients(layer.clone(dtype=jnp.float64).apply))


def test_joined_projections_keep_the_dtype_their_promotion_gives_a_wider_bias():
    """Joined, two projections whose promotion leaves the bias at fp32 add it to their bf16 products in
    the fp32 the addition promotes to, as each projection apart adds its own: the stacked outputs
    carry each projection's own dtype, within tests/reference_error.py's rule of the projections'
    apart against float64."""
    def bias_kept(inputs, kernel, bias, dtype=None):
        return inputs.astype(jnp.bfloat16), kernel.astype(jnp.bfloat16), bias

    class Pair(nn.Module):
        wide: bool = False

        @nn.compact
        def __call__(self, x):
            promotion = {} if self.wide else {"promote_dtype": bias_kept}
            projections = [Dense(96, dtype=jnp.float64 if self.wide else jnp.bfloat16, name=name, **promotion)
                           for name in ("gate", "up")]
            if self.wide:
                return jnp.stack([projection(x) for projection in projections], axis=-2)
            apart = jnp.stack([projection(x) for projection in projections], axis=-2)
            return apart, joined_dense(x, projections, axis=-2)

    x = jax.random.normal(jax.random.key(10), (2, 33, 64)).astype(jnp.bfloat16)
    params = jax.tree.map(lambda leaf: leaf + 0.1 * jax.random.normal(jax.random.key(12), leaf.shape),
                          Pair().init(jax.random.key(13), x))
    apart, joined = Pair().apply(params, x)
    assert joined.dtype == apart.dtype == jnp.float32
    with jax.enable_x64(new_val=True):
        truth = Pair(wide=True).apply(jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64), params),
                                      jnp.asarray(x, jnp.float64))
    assert_as_exact_as_the_reference(np.asarray(joined), np.asarray(apart), np.asarray(truth), "joined")


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
    assert_joined_as_exact("delta_gates", with_gradients(layer.apply), (params, x, cotangent),
                           with_gradients(layer.clone(dtype=jnp.float64).apply))
