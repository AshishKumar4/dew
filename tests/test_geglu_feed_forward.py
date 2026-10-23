"""diffusers' GEGLU feed-forward as a packed `GatedMLP` (`geglu_feed_forward`): its one projection
holds the up half first, as the checkpoint stores it, and computes what diffusers' `FeedForward`
computes from it, on par with that computation from its float64 run (check 1)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact, with_gradients

from dew.diffusion.process import DenoisingCondition
from dew.nn.activations import gated_product
from dew.nn.dense import Dense
from dew.nn.mlp import GatedMLP, geglu_feed_forward
from dew.nn.precision import float32_dot_general
from dew.nn.sharding import declarations, declared_axes, parameter_path

GEGLU_SEEDS = {"tanh float32": 16, "tanh bfloat16": 16, "erf float32": 16, "erf bfloat16": 16}
"""Check 1's declared seeds for each form (to size)."""


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["float32", "bfloat16"])
@pytest.mark.parametrize("approximate", [True, False], ids=["tanh", "erf"])
def test_the_packed_feed_forward_computes_diffusers_geglu(approximate, dtype):
    """diffusers' `GEGLU`: one projection whose first half is the hidden state and second the gate,
    `hidden * gelu(gate)`, then the output map. The packed GatedMLP's output and every gradient are on
    par with that computation's in the same dtype (check 1), from its float64 run, over drawn states,
    parameters and cotangents."""
    dim = 8
    shape = (4, 16, dim)
    packed = geglu_feed_forward(dim, approximate=approximate, dtype=dtype)
    shapes = jax.eval_shape(packed.init, jax.random.key(0), jnp.zeros(shape, dtype))["params"]

    def draw(seed):
        rng = np.random.default_rng(seed)
        params = jax.tree.map(lambda leaf: np.asarray(rng.standard_normal(leaf.shape) * 0.3, leaf.dtype),
                              shapes)
        x, cotangent = (np.asarray(rng.standard_normal(shape), dtype) for _ in range(2))
        return params, x, cotangent

    def diffusers(dtype):
        dense = {width: Dense(width, dtype=dtype, dot_general=float32_dot_general)
                 for width in (8 * dim, dim)}

        def forward(params, x):
            hidden, gate = jnp.split(dense[8 * dim].apply({"params": params["gate_up_proj"]}, x), 2, axis=-1)
            activated = gated_product("geglu" if approximate else "geglu_exact")(gate, hidden)
            return dense[dim].apply({"params": params["down_proj"]}, activated)
        return jax.jit(with_gradients(forward))

    label = f"{'tanh' if approximate else 'erf'} {jnp.dtype(dtype).name}"
    assert_as_exact(jax.jit(with_gradients(lambda params, x: packed.apply({"params": params}, x))),
                    diffusers(dtype), draw, label, seeds=GEGLU_SEEDS[label], truth_fn=diffusers(None))


def test_a_unets_feed_forward_declares_its_kernels_axes():
    """Every leaf of a UNet's feed-forward, the packed projection's included, carries declared axes:
    the packed projection's kernel `embed` to `mlp` and its bias `mlp`, so a tensor axis splits it as
    it splits any MLP."""
    from dew.registry import models

    model = models.build("unet_2d_condition", stages=[{"features": 32, "heads": 2}], blocks_per_level=1)
    params = jax.eval_shape(lambda: model.init(
        jax.random.key(0), jnp.zeros((1, 8, 8, 4)), jnp.ones((1,)),
        conditioning=DenoisingCondition(jnp.zeros((1, 3, 768)))))["params"]
    declared = declarations(params)
    leaves = [(path, leaf) for path, leaf in jax.tree_util.tree_leaves_with_path(params)
              if "feed_forward" in parameter_path(path)]
    assert {parameter_path(path)[-2] for path, _ in leaves} == {"gate_up_proj", "down_proj"}
    expected = {"kernel": ("embed", "mlp"), "bias": ("mlp",)}
    for path, leaf in leaves:
        axes = declared_axes(path, len(leaf.shape), declared)
        names = parameter_path(path)
        assert axes is not None, names
        if names[-2] == "gate_up_proj":
            assert axes == expected[names[-1]], names


def test_a_packed_order_names_the_gate_and_up_projections():
    """A packed order other than the two projections' is refused."""
    mlp = GatedMLP(hidden_features=8, out_features=4, activation="geglu", packed=True,
                   packed_order=("gate_proj", "down_proj"))
    with pytest.raises(ValueError):
        mlp.init(jax.random.key(0), jnp.zeros((1, 4)))
