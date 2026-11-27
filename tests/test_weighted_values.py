"""The probability-value product attention's reference route takes (`dew.nn.attention.weighted_values`): a
CPU's long fp32 sum in blocks of keys joined once, the one dot kept where its sum is one float64 sum or a
mesh splits the keys (row 13655)."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, NamedSharding, PartitionSpec

from dew.nn.attention import VALUE_BLOCK, weighted_values
from dew.nn.precision import sums_in_float64

EQUATION = '...hqk,...khd->...qhd'


def test_a_sum_across_its_blocks_cancels_exactly_where_the_one_dot_is_one_float64_sum():
    """Under Dew's wide CPU arithmetic, where the one dot is one float64 sum rounded once
    (`sums_in_float64`), a product whose keys cancel across a block's edge keeps what they leave: 1 and 2^-24
    in the first block and -1 past it leave 2^-24, which blocks rounded to float32 apart would lose."""
    keys = VALUE_BLOCK + 1
    weights = np.zeros((1, 1, 1, keys), np.float32)
    weights[..., [0, 1, keys - 1]] = 1.0
    value = np.zeros((1, keys, 1, 1), np.float32)
    value[0, [0, 1, keys - 1], 0, 0] = (1.0, 2.0 ** -24, -1.0)
    assert sums_in_float64(jnp.asarray(weights), jnp.asarray(value)), "the suite runs wide CPU arithmetic"
    out = jax.jit(lambda w, v: weighted_values(EQUATION, w, v))(weights, value)
    assert float(np.asarray(out).ravel()[0]) == 2.0 ** -24


def test_keys_a_mesh_splits_take_the_one_dot():
    """Keys an Explicit mesh axis splits three ways, 66 of them, which no 64-key block divides: the product
    traces as the one dot, split as its operands are."""
    devices = np.asarray(jax.devices()[:3])
    mesh = jax.sharding.Mesh(devices, ("keys",), axis_types=(AxisType.Explicit,))
    weights = jax.ShapeDtypeStruct((1, 2, 5, 66), jnp.float32,
                                   sharding=NamedSharding(mesh, PartitionSpec(None, None, None, "keys")))
    value = jax.ShapeDtypeStruct((1, 66, 2, 4), jnp.float32,
                                 sharding=NamedSharding(mesh, PartitionSpec(None, "keys", None, None)))
    with jax.sharding.use_mesh(mesh):
        traced = jax.jit(lambda w, v: weighted_values(EQUATION, w, v)).trace(weights, value)
    assert traced.out_info.shape == (1, 5, 2, 4)
