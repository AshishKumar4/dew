"""Attention's probability dropout on a mesh: a dropping call on a data and head mesh, and under axis
contention, keeps the entries one device's draw keeps, the draw taking each entry's global batch row and
query head through `_per_shard`.
"""


import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from test_attention_dropout import RATE

from dew.nn.attention import scaled_dot_product_attention
from dew.nn.kernels.templates.attention import dropout_keep, dropout_seed
from dew.mesh import DEFAULT_RULES


@pytest.mark.mesh(devices=8)
def test_a_dropping_call_on_a_data_and_head_mesh_keeps_the_entries_one_device_keeps():
    """A dropping template call on a mesh of two rows by four tensor shards draws each entry at its global
    batch row and query head, so its kept entries are one device's draw bit for bit."""
    mesh = Mesh(np.asarray(jax.devices()[:8]).reshape(2, 4), ("data", "tensor"))
    rng = jax.random.PRNGKey(7)
    query = jnp.zeros((2, 16, 8, 8), jnp.float32)
    key = jnp.zeros((2, 16, 4, 8), jnp.float32)
    value = jnp.broadcast_to(jnp.eye(16, dtype=jnp.float32)[None, :, None, :], (2, 16, 4, 16))
    rows_and_heads = PartitionSpec("data", None, "tensor")
    with jax.set_mesh(mesh), nn.logical_axis_rules(DEFAULT_RULES):
        placed = [jax.device_put(array, NamedSharding(mesh, rows_and_heads)) for array in (query, key, value)]
        out = scaled_dot_product_attention(*placed, implementation="template", dropout_rate=RATE,
                                           dropout_rng=rng, deterministic=False)
    mask = np.asarray(dropout_keep(dropout_seed(rng), RATE, (2, 8, 16, 16)))
    np.testing.assert_array_equal(np.asarray(out) > 0, mask.transpose(0, 2, 1, 3))


CONTENTION = (("activation_batch", "expert"), ("activation_heads", ("tensor", "expert")), *DEFAULT_RULES)
"""Rules that give the rows the expert axis first, so the query heads split over the tensor axis alone, where
a head rule on its own splits them over both: the axis contention `_per_shard`'s offsets must follow."""


@pytest.mark.mesh(devices=8)
def test_a_dropping_call_under_axis_contention_keeps_the_entries_one_device_keeps():
    """A dropping template call on a tensor-by-expert mesh (2 by 4) under `CONTENTION`, without sinks: each
    entry's draw takes its global batch row and query head from the query's own split, so the kept entries
    are one device's draw bit for bit."""
    mesh = Mesh(np.asarray(jax.devices()[:8]).reshape(2, 4), ("tensor", "expert"))
    rng = jax.random.PRNGKey(7)
    query = jnp.zeros((4, 16, 8, 8), jnp.float32)
    key = jnp.zeros((4, 16, 8, 8), jnp.float32)
    value = jnp.broadcast_to(jnp.eye(16, dtype=jnp.float32)[None, :, None, :], (4, 16, 8, 16))
    rows_and_heads = PartitionSpec("expert", None, "tensor")
    with jax.set_mesh(mesh), nn.logical_axis_rules(CONTENTION):
        placed = [jax.device_put(array, NamedSharding(mesh, rows_and_heads)) for array in (query, key, value)]
        out = scaled_dot_product_attention(*placed, implementation="template", dropout_rate=RATE,
                                           dropout_rng=rng, deterministic=False)
    mask = np.asarray(dropout_keep(dropout_seed(rng), RATE, (4, 8, 16, 16)))
    np.testing.assert_array_equal(np.asarray(out) > 0, mask.transpose(0, 2, 1, 3))


