"""The probability-value product attention's reference route takes (`dew.nn.attention.weighted_values`): a
CPU's long fp32 sum in blocks of keys joined once, the one dot kept where its sum is one float64 sum or a
mesh splits the keys (row 13655)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, NamedSharding, PartitionSpec

from dew.nn.attention import VALUE_BLOCK, weighted_values
from dew.nn.precision import sums_in_float64

EQUATION = '...hqk,...khd->...qhd'


@pytest.mark.cpu
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


@pytest.mark.mesh(devices=3)
def test_keys_a_mesh_splits_take_the_one_dot():
    """Keys an Explicit mesh axis splits three ways, 66 of them, which no 64-key block divides, summed in fp32
    (`FP32_SUMS`, where a CPU would block them): the product traces as the one dot, split as its value is."""
    from dew.nn.precision import FP32_SUMS

    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:3]), ("keys",), axis_types=(AxisType.Explicit,))
    weights = jax.ShapeDtypeStruct((1, 2, 5, 66), jnp.float32, sharding=NamedSharding(mesh, PartitionSpec()))
    value = jax.ShapeDtypeStruct((1, 66, 2, 4), jnp.float32,
                                 sharding=NamedSharding(mesh, PartitionSpec(None, "keys", None, None)))
    with jax.set_mesh(mesh), jax.default_matmul_precision(FP32_SUMS):
        traced = jax.jit(lambda w, v: weighted_values(EQUATION, w, v)).trace(weights, value)
    assert traced.out_info.shape == (1, 5, 2, 4)


@pytest.mark.parametrize("default", [
    pytest.param("cpu", marks=pytest.mark.cpu),
    pytest.param("gpu", marks=[pytest.mark.gpu, pytest.mark.compiles_at_run(
        reason="an eager call, its programs compiled where it reaches them")])])
def test_an_eager_product_on_cpu_operands_sums_where_they_lie(default, monkeypatch):
    """Called eagerly on operands committed to the CPU, in a process computing as a user's does (fp32 sums in
    fp32), the product takes the CPU's blocked sum whatever the default device, as its jitted call there
    does (audit 2026-11-27 P07: under a GPU default it took the accelerator's one dot)."""
    from dew.nn.precision import WIDE_CPU_ARITHMETIC, float32_einsum

    monkeypatch.delenv(WIDE_CPU_ARITHMETIC, raising=False)
    assert jax.default_backend() == default
    cpu = jax.devices("cpu")[0]
    rng = np.random.default_rng(13998)
    weights = jax.device_put(rng.random((1, 2, 4, 16 * VALUE_BLOCK), np.float32), cpu)
    value = jax.device_put(rng.standard_normal((1, 16 * VALUE_BLOCK, 2, 8)).astype(np.float32), cpu)
    blocked = np.asarray(jax.jit(lambda w, v: weighted_values(EQUATION, w, v))(weights, value))
    whole = np.asarray(jax.jit(lambda w, v: float32_einsum(EQUATION, w, v))(weights, value))
    assert not np.array_equal(blocked, whole), "the blocked and the one-dot sums tell the routes apart"
    np.testing.assert_array_equal(np.asarray(weighted_values(EQUATION, weights, value)), blocked)
