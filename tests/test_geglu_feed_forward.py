"""diffusers' GEGLU feed-forward as a packed `GatedMLP` (`geglu_feed_forward`): its one projection
holds the up half first, as the checkpoint stores it, and computes what the unpacked MLP computes from
those halves, bit for bit."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import with_gradients

from dew.nn.mlp import GatedMLP, geglu_feed_forward


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("approximate", [True, False], ids=["tanh", "erf"])
def test_the_packed_up_first_projection_is_the_unpacked_mlp_bit_for_bit(approximate, dtype):
    """The packed projection's first half is the up projection and its second the gate: the output
    and every gradient are the unpacked GatedMLP's on those halves, to the bit."""
    dim = 8
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((2, 5, dim)), dtype)
    cotangent = jnp.asarray(rng.standard_normal((2, 5, dim)), dtype)
    packed = geglu_feed_forward(dim, approximate=approximate, dtype=dtype)
    params = jax.tree.map(lambda leaf: jnp.asarray(rng.standard_normal(leaf.shape) * 0.3, leaf.dtype),
                          packed.init(jax.random.key(0), x)["params"])
    kernels, biases = (jnp.split(params["gate_up_proj"][leaf], 2, axis=-1) for leaf in ("kernel", "bias"))
    apart = {"up_proj": {"kernel": kernels[0], "bias": biases[0]},
             "gate_proj": {"kernel": kernels[1], "bias": biases[1]}, "down_proj": params["down_proj"]}
    unpacked = GatedMLP(hidden_features=4 * dim, out_features=dim,
                        activation="geglu" if approximate else "geglu_exact", use_bias=True, dtype=dtype)

    out, d_params, d_x = with_gradients(lambda p, x: packed.apply({"params": p}, x))(params, x, cotangent)
    their_out, their_params, their_x = with_gradients(lambda p, x: unpacked.apply({"params": p}, x))(
        apart, x, cotangent)
    np.testing.assert_array_equal(np.asarray(out, np.float32), np.asarray(their_out, np.float32))
    np.testing.assert_array_equal(np.asarray(d_x, np.float32), np.asarray(their_x, np.float32))
    for leaf in ("kernel", "bias"):
        joined = jnp.concatenate([their_params["up_proj"][leaf], their_params["gate_proj"][leaf]], axis=-1)
        np.testing.assert_array_equal(np.asarray(d_params["gate_up_proj"][leaf], np.float32),
                                      np.asarray(joined, np.float32))
    for leaf in ("kernel", "bias"):
        np.testing.assert_array_equal(np.asarray(d_params["down_proj"][leaf], np.float32),
                                      np.asarray(their_params["down_proj"][leaf], np.float32))


def test_a_packed_order_names_the_gate_and_up_projections():
    """A packed order other than the two projections' is refused."""
    mlp = GatedMLP(hidden_features=8, out_features=4, activation="geglu", packed=True,
                   packed_order=("gate_proj", "down_proj"))
    with pytest.raises(ValueError):
        mlp.init(jax.random.key(0), jnp.zeros((1, 4)))
