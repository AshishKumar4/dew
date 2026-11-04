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

from dew.mesh import DEFAULT_RULES
from dew.nn.attention import scaled_dot_product_attention
from dew.nn.kernels.templates.attention import dropout_keep, dropout_seed

CONTENTION = (("activation_batch", "expert"), ("activation_heads", ("tensor", "expert")), *DEFAULT_RULES)
"""Rules that give the rows the expert axis first, so the query heads split over the tensor axis alone, where
a head rule on its own splits them over both: the axis contention `_per_shard`'s offsets must follow."""


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("axes, rules, rows_axis, batch, kv_heads", [
    (("data", "tensor"), DEFAULT_RULES, "data", 2, 4),
    (("tensor", "expert"), CONTENTION, "expert", 4, 8),
], ids=["a data and head mesh", "axis contention"])
def test_a_dropping_call_on_a_mesh_keeps_the_entries_one_device_keeps(axes, rules, rows_axis, batch,
                                                                       kv_heads):
    """A dropping template call on a mesh of 2 by 4, its rows split over `rows_axis` and its query heads over
    the tensor axis as `rules` place them (a data and head mesh, or `CONTENTION`'s, without sinks): each
    entry's draw takes its global batch row and query head from the query's own split, so the kept entries
    are one device's draw bit for bit."""
    mesh = Mesh(np.asarray(jax.devices()[:8]).reshape(2, 4), axes)
    rng = jax.random.PRNGKey(7)
    query = jnp.zeros((batch, 16, 8, 8), jnp.float32)
    key = jnp.zeros((batch, 16, kv_heads, 8), jnp.float32)
    value = jnp.broadcast_to(jnp.eye(16, dtype=jnp.float32)[None, :, None, :], (batch, 16, kv_heads, 16))
    rows_and_heads = PartitionSpec(rows_axis, None, "tensor")
    with jax.set_mesh(mesh), nn.logical_axis_rules(rules):
        placed = [jax.device_put(array, NamedSharding(mesh, rows_and_heads)) for array in (query, key, value)]
        out = scaled_dot_product_attention(*placed, implementation="template", dropout_rate=RATE,
                                           dropout_rng=rng, deterministic=False)
    mask = np.asarray(dropout_keep(dropout_seed(rng), RATE, (batch, 8, 16, 16)))
    np.testing.assert_array_equal(np.asarray(out) > 0, mask.transpose(0, 2, 1, 3))
