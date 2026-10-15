"""Projections joined per call (`dew.nn.joined`): one GEMM in place of a
layer's own projection calls, where a generation's kernel row measured it
faster."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference

from dew.nn import joined
from dew.nn.linear import GatedDeltaNet
from dew.nn.mlp import GatedMLP


def row_joins(monkeypatch, choice: str, *, joins: bool = True):
    """The device's target row names `choice` 'joined' where `joins`, as the A100's row names
    'gate_up' (`measured_kernel`)."""
    monkeypatch.setattr(joined, "measured_kernel",
                        lambda name, default: "joined" if joins and name == choice else default)


def dots(function, *args) -> int:
    return sum(equation.primitive.name == "dot_general" for equation in jax.make_jaxpr(function)(*args).eqns)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
def test_gate_and_up_joined_are_as_exact_as_apart_from_float64(monkeypatch, dtype, bias):
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

    def run(layer, params, x, cotangent):
        out, pull = jax.vjp(layer.apply, params, x)
        return out, *pull(cotangent)

    row_joins(monkeypatch, "gate_up", joins=False)
    assert dots(layer.apply, params, x) == 3
    apart = jax.jit(lambda *args: run(layer, *args))(params, x, cotangent)
    row_joins(monkeypatch, "gate_up")
    assert dots(layer.apply, params, x) == 2
    together = jax.jit(lambda *args: run(layer, *args))(params, x, cotangent)
    row_joins(monkeypatch, "gate_up", joins=False)
    with jax.enable_x64(new_val=True):
        wide = jax.tree.map(lambda leaf: jnp.asarray(np.asarray(leaf, np.float64)), (params, x, cotangent))
        truth = run(layer.clone(dtype=jnp.float64), *wide)
    for (path, mine), theirs, want in zip(jax.tree_util.tree_leaves_with_path(together),
                                          jax.tree.leaves(apart), jax.tree.leaves(truth), strict=True):
        assert mine.dtype == theirs.dtype
        assert_as_exact_as_the_reference(mine, theirs, want, jax.tree_util.keystr(path))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_a_delta_nets_gates_joined_are_as_exact_as_apart_from_float64(monkeypatch, dtype):
    """A gated delta net whose row joins its b and a projections ('delta_gates') runs them as one
    GEMM, one dot fewer, and its output and the gradients of its input and every parameter lie within
    tests/reference_error.py's rule of the layer's with them apart, both measured from float64."""
    x = jax.random.normal(jax.random.key(7), (2, 40, 64)).astype(dtype)
    cotangent = jax.random.normal(jax.random.key(8), (2, 40, 64)).astype(dtype)
    layer = GatedDeltaNet(emb_features=64, num_k_heads=2, num_v_heads=4, head_k_dim=16, head_v_dim=16,
                          dtype=dtype)
    params = layer.init(jax.random.key(9), x)

    def run(layer, params, x, cotangent):
        out, pull = jax.vjp(layer.apply, params, x)
        return out, *pull(cotangent)

    row_joins(monkeypatch, "delta_gates", joins=False)
    apart_dots = dots(layer.apply, params, x)
    apart = jax.jit(lambda *args: run(layer, *args))(params, x, cotangent)
    row_joins(monkeypatch, "delta_gates")
    assert dots(layer.apply, params, x) == apart_dots - 1
    together = jax.jit(lambda *args: run(layer, *args))(params, x, cotangent)
    row_joins(monkeypatch, "delta_gates", joins=False)
    with jax.enable_x64(new_val=True):
        wide = jax.tree.map(lambda leaf: jnp.asarray(np.asarray(leaf, np.float64)), (params, x, cotangent))
        truth = run(layer.clone(dtype=jnp.float64), *wide)
    for (path, mine), theirs, want in zip(jax.tree_util.tree_leaves_with_path(together),
                                          jax.tree.leaves(apart), jax.tree.leaves(truth), strict=True):
        assert mine.dtype == theirs.dtype
        assert_as_exact_as_the_reference(mine, theirs, want, jax.tree_util.keystr(path))
