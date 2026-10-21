"""Scratch sizing of the fla gate sites' declared seeds (`seeds_for`): temporary, removed once
the counts are declared in test_fla_gdn.py and test_fla_kda.py. Each test fails with its count so
it lands in the run's log."""

import jax
import jax.numpy as jnp
import pytest
from delta_rule_support import recurrence, through
from reference_error import seeds_for
from test_fla_kda import fla_math, operands

from dew.nn.kernels import generation
from dew.nn.linear import GatedDeltaNet


def _run(rule):
    return lambda q, k, v, g, beta, h0, do, dht: through(rule, do, dht)(q, k, v, g, beta, h0)


@pytest.mark.parametrize("key_heads, value_heads, width, columns, length", [
    (1, 2, 64, 64, 128),
    (2, 2, 64, 64, 100),
    (1, 1, 32, 96, 70),
    (1, 1, 128, 128, 64),
])
def test_size_kda(key_heads, value_heads, width, columns, length):
    def draw(seed):
        return operands(1, length, key_heads, value_heads, width, columns, jnp.float32, seed=seed)

    count = seeds_for(jax.jit(_run(fla_math)), draw,
                      f"fla_kda {(key_heads, value_heads, width, columns, length)}",
                      truth_fn=jax.jit(_run(recurrence)))
    pytest.fail(f"SIZE|kda-{key_heads}-{value_heads}-{width}-{columns}-{length}|{count}")


@generation.chosen(gated_delta_rule="fla")
def test_size_gdn_remat():
    layer = GatedDeltaNet(emb_features=64, num_k_heads=1, num_v_heads=2, head_k_dim=64, head_v_dim=64)
    x = jax.random.normal(jax.random.key(71), (1, 64, 64))
    params = layer.init(jax.random.key(73), x)

    def loss(params, x):
        return jnp.sum(layer.apply(params, x) ** 2)

    def draw(seed):
        return params, jax.random.normal(jax.random.key(seed), x.shape)

    wide = layer.clone(dtype=jnp.float64)

    def truth(params, x):
        return jax.grad(lambda params: jnp.sum(wide.apply(params, x) ** 2))(params)

    count = seeds_for(jax.jit(jax.grad(loss)), draw, "remat policy gradient", truth_fn=jax.jit(truth))
    pytest.fail(f"SIZE|gdn-remat|{count}")
