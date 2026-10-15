"""Projections joined per call (`dew.nn.joined`): one GEMM in place of a
layer's own projection calls, where a generation's kernel row measured it
faster."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference, equations

from dew.nn import joined
from dew.nn.mlp import GatedMLP


def row_joins(monkeypatch, choice: str, *, joins: bool = True):
    """The device's target row names `choice` 'joined' where `joins`, as the A100's row names
    'gate_up' (`measured_kernel`)."""
    monkeypatch.setattr(joined, "measured_kernel",
                        lambda name, default: "joined" if joins and name == choice else default)


def dots(function, *args) -> int:
    """The dot products `function`'s jaxpr runs, its sub-jaxprs' included."""
    graph = jax.make_jaxpr(function)(*args)
    return sum(equation.primitive.name == "dot_general" for equation in equations(graph))


def assert_joined_as_exact(monkeypatch, choice: str, run, arguments, wide_run):
    """`run(*arguments)` with `choice` joined, every leaf within tests/reference_error.py's rule of
    the run with it apart, both measured from `wide_run`, the apart run in float64."""
    row_joins(monkeypatch, choice, joins=False)
    apart = jax.jit(run)(*arguments)
    row_joins(monkeypatch, choice)
    together = jax.jit(run)(*arguments)
    row_joins(monkeypatch, choice, joins=False)
    with jax.enable_x64(new_val=True):
        truth = wide_run(*jax.tree.map(lambda leaf: jnp.asarray(np.asarray(leaf, np.float64)), arguments))
    for (path, mine), theirs, want in zip(jax.tree_util.tree_leaves_with_path(together),
                                          jax.tree.leaves(apart), jax.tree.leaves(truth), strict=True):
        assert mine.dtype == theirs.dtype
        assert_as_exact_as_the_reference(mine, theirs, want, jax.tree_util.keystr(path))


def gradients(apply):
    """`apply`'s output and the gradients of its arguments under a cotangent, the last argument."""
    def run(*arguments):
        out, pull = jax.vjp(apply, *arguments[:-1])
        return out, *pull(arguments[-1])

    return run


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
    row_joins(monkeypatch, "gate_up", joins=False)
    assert dots(layer.apply, params, x) == 3
    row_joins(monkeypatch, "gate_up")
    assert dots(layer.apply, params, x) == 2
    assert_joined_as_exact(monkeypatch, "gate_up", gradients(layer.apply), (params, x, cotangent),
                           gradients(layer.clone(dtype=jnp.float64).apply))
