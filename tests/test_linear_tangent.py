"""Which program each mode of differentiation runs through `generation.linear_tangent`: forward mode the
portable arithmetic's tangent, reverse mode the backward kernel, however a jit, a shard_map, a scan, a
remat or a vmap stages the call."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec

from dew.nn.kernels import generation


def tripled(x, index):
    """Three times `x`, whose backward kernel returns four times the cotangent: a gradient says by its
    value which program ran, 4 the backward kernel, 3 the portable arithmetic's pullback. `index`, an
    integer operand, takes float0's cotangent."""
    return generation.linear_tangent(
        lambda x, index: (3 * x, ()),
        lambda saved, operands, cotangent: (4 * cotangent, np.zeros(operands[1].shape, jax.dtypes.float0)),
        lambda x, index: 3 * x)(x, index)


def summed(x, index):
    return tripled(x, index).sum()


def reversed_through(staging):
    """The gradient of `summed` at every element, reached through `staging`."""
    mesh = Mesh(np.array(jax.devices()), ("rows",))
    rows = PartitionSpec("rows")
    if staging == "vjp":
        return jax.grad(summed)
    if staging == "jit":
        return jax.jit(jax.grad(summed))
    if staging == "shard_map":
        # shard_map transposes the jvp of its body rather than taking a vjp.
        return jax.jit(jax.grad(lambda x, index: jax.shard_map(
            tripled, mesh=mesh, in_specs=(rows, rows), out_specs=rows)(x, index).sum()))
    if staging == "scan":
        def scanned(x, index):
            return jax.lax.scan(lambda carry, row: (carry, tripled(*row)), None,
                                (x.reshape(2, -1), index.reshape(2, -1)))[1].sum()

        return jax.grad(scanned)
    if staging == "remat":
        return jax.grad(jax.checkpoint(summed))
    if staging == "linearize":
        def transposed(x, index):
            linear = jax.linearize(lambda x: tripled(x, index), x)[1]
            return jax.linear_transpose(linear, x)(jnp.ones_like(x))[0]

        return transposed
    return lambda x, index: jax.vmap(jax.grad(summed))(x.reshape(2, -1), index.reshape(2, -1)).reshape(-1)


@pytest.mark.parametrize("staging", ["vjp", "jit", "shard_map", "scan", "remat", "linearize", "vmap"])
def test_reverse_mode_runs_the_backward_kernel_however_the_call_is_staged(staging):
    """A vjp, or a jvp that a shard_map, a scan or `jax.linear_transpose` then transposes: each runs the
    backward kernel (`generation.linear_map`)."""
    x = jnp.arange(8.0 * jax.device_count())
    index = jnp.arange(x.size)
    np.testing.assert_array_equal(np.asarray(reversed_through(staging)(x, index)), np.full(x.shape, 4.0))


@pytest.mark.parametrize("staging", ["jvp", "jit", "linearize"])
def test_forward_mode_takes_the_portable_tangent_however_the_call_is_staged(staging):
    """`jax.jvp` and `jax.linearize`, staged by a jit or not, evaluate the portable arithmetic's tangent."""
    x, index = jnp.arange(8.0), jnp.arange(8)
    direction = jnp.ones_like(x)

    def tangent(x):
        if staging == "linearize":
            return jax.linearize(lambda x: tripled(x, index), x)[1](direction)
        return jax.jvp(lambda x: tripled(x, index), (x,), (direction,))[1]

    run = jax.jit(tangent) if staging == "jit" else tangent
    np.testing.assert_array_equal(np.asarray(run(x)), np.full(x.shape, 3.0))
