"""The custom-kernel contract's own seams (tests/sharded.py)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from arm_arithmetic import summed_as_run
from jax.sharding import AxisType, PartitionSpec
from sharded import placed_on, split_executable

from dew.nn import precision


@pytest.mark.mesh(devices=2)
@pytest.mark.skipif(jax.default_backend() != "cpu", reason="the suite's wide sums are a CPU's")
def test_the_split_executable_runs_each_shard_as_the_gate_runs_its_arms():
    """The split arm's executable sums each shard's products as the gate's scope sums them (fp32,
    `summed_as_run`), as the one-device arm traced in that scope does, not as the suite's wide CPU
    arithmetic would have, frozen into an executable compiled outside it (ledger 4393): each shard's rows
    are bit for bit one device's product over those rows in the scope, and that product differs from the
    wide sum, which this case tells apart."""
    mesh = jax.make_mesh((2,), ("data",), axis_types=(AxisType.Auto,), devices=jax.devices()[:2])
    rng = np.random.default_rng(0x4393)
    x = jnp.asarray(rng.standard_normal((8, 4096)), jnp.float32)
    w = jnp.asarray(rng.standard_normal((4096, 16)), jnp.float32)
    specs = (PartitionSpec("data"), PartitionSpec())

    def product(x, w):
        return precision.accumulated(jnp.dot, x, w)

    compiled = split_executable(jax.jit(product), (x, w), specs, mesh)
    with jax.set_mesh(mesh):
        split = np.asarray(compiled(*placed_on((x, w), specs, mesh)))
    with summed_as_run():
        narrow = np.concatenate([np.asarray(jax.jit(product)(rows, w)) for rows in (x[:4], x[4:])])
    wide = np.concatenate([np.asarray(jax.jit(product)(rows, w)) for rows in (x[:4], x[4:])])
    assert not np.array_equal(narrow, wide)
    np.testing.assert_array_equal(split, narrow)
