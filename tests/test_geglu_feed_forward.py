"""diffusers' GEGLU feed-forward as a packed `GatedMLP` (`geglu_feed_forward`): its one projection
holds the up half first, as the checkpoint stores it, and computes what diffusers' `FeedForward`
computes from it, bit for bit."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import with_gradients

from dew.diffusion.process import DenoisingCondition
from dew.nn.activations import gated_product
from dew.nn.dense import Dense
from dew.nn.mlp import GatedMLP, geglu_feed_forward
from dew.nn.precision import float32_dot_general
from dew.nn.sharding import declarations, declared_axes, parameter_path


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("approximate", [True, False], ids=["tanh", "erf"])
def test_the_packed_feed_forward_is_diffusers_geglu_bit_for_bit(approximate, dtype):
    """diffusers' `GEGLU`: one projection whose first half is the hidden state and second the gate,
    `hidden * gelu(gate)`, then the output map. The packed GatedMLP's output and every gradient are
    that computation's, to the bit."""
    dim = 8
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((2, 5, dim)), dtype)
    cotangent = jnp.asarray(rng.standard_normal((2, 5, dim)), dtype)
    packed = geglu_feed_forward(dim, approximate=approximate, dtype=dtype)
    params = jax.tree.map(lambda leaf: jnp.asarray(rng.standard_normal(leaf.shape) * 0.3, leaf.dtype),
                          packed.init(jax.random.key(0), x)["params"])
    dense = {width: Dense(width, dtype=dtype, dot_general=float32_dot_general) for width in (8 * dim, dim)}

    def diffusers(params, x):
        hidden, gate = jnp.split(dense[8 * dim].apply({"params": params["gate_up_proj"]}, x), 2, axis=-1)
        activated = gated_product("geglu" if approximate else "geglu_exact")(gate, hidden)
        return dense[dim].apply({"params": params["down_proj"]}, activated)

    ours = jax.jit(with_gradients(lambda p, x: packed.apply({"params": p}, x)))(params, x, cotangent)
    theirs = jax.jit(with_gradients(diffusers))(params, x, cotangent)
    for mine, reference in zip(jax.tree.leaves(ours), jax.tree.leaves(theirs), strict=True):
        np.testing.assert_array_equal(np.asarray(mine, np.float32), np.asarray(reference, np.float32))


def test_a_unets_feed_forward_declares_its_kernels_axes():
    """Every leaf of a UNet's feed-forward, the packed projection's included, carries its declared
    axes, the hidden width `mlp`, so a tensor axis splits it as it splits any MLP."""
    from dew.registry import models

    model = models.build("unet_2d_condition", stages=[{"features": 32, "heads": 2}], blocks_per_level=1)
    params = jax.eval_shape(lambda: model.init(
        jax.random.key(0), jnp.zeros((1, 8, 8, 4)), jnp.ones((1,)),
        conditioning=DenoisingCondition(jnp.zeros((1, 3, 768)))))["params"]
    declared = declarations(params)
    leaves = {parameter_path(path): leaf for path, leaf in jax.tree_util.tree_leaves_with_path(params)
              if "feed_forward" in parameter_path(path)}
    assert {names[-2] for names in leaves} == {"gate_up_proj", "down_proj"}
    for names, leaf in leaves.items():
        axes = declared_axes(names, len(leaf.shape), declared)
        assert axes is not None and "mlp" in axes, names


def test_a_packed_order_names_the_gate_and_up_projections():
    """A packed order other than the two projections' is refused."""
    mlp = GatedMLP(hidden_features=8, out_features=4, activation="geglu", packed=True,
                   packed_order=("gate_proj", "down_proj"))
    with pytest.raises(ValueError):
        mlp.init(jax.random.key(0), jnp.zeros((1, 4)))
