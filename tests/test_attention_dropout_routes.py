"""Attention's probability dropout on the Triton mainloop and on a mesh: the interpreted Triton forward and
backward are on par with the reference path on the same draw (check 1), and a dropping call on a data and
head mesh keeps the entries one device's draw keeps, since the draw takes each entry's global batch row and
query head through `_per_shard`.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from reference_error import assert_as_exact
from test_attention_dropout import RATE, grads_of, hooked, reference_arm, route_draw
from test_attention_template import CASES, SEEDS, arguments, geometry

from dew.nn.attention import dropout_draw, dropout_seed, scaled_dot_product_attention
from dew.nn.kernels.templates import attention as template, attention_triton
from dew.nn.sharding import DEFAULT_RULES

TRITON_CASES = ("causal-window", "causal-documents-window")
"""The span forms the interpreted Triton check runs: a window, and documents under a window."""


def triton_arm(case: dict):
    """`case`'s Triton forward and backward, interpreted, dropping from the seed words, as a function of
    `(query, key, value, words, dout)` returning the output and the query, key and value gradients."""
    queries, keys = geometry(case)
    spans = template.spans(queries, keys, 2, **arguments(case))

    def run(query, key, value, words, dout):
        hooks = hooked(words, 2, query.shape[2])
        out, peak, total, anchors, split = attention_triton.forward(
            query, key, value, spans, hooks, residuals=True, interpret=True)
        grads = attention_triton.backward(query, key, value, spans, hooks, dout,
                                          residuals=(peak, total, anchors, split), interpret=True)
        return out, grads.query, grads.key, grads.value

    return run


@pytest.mark.parametrize("name", TRITON_CASES)
def test_the_interpreted_triton_mainloop_drops_on_par_with_the_reference_path(name):
    """The Triton forward and backward, interpreted, on the same draw as the reference path, both measured
    from float64 (check 1)."""
    case = CASES[name]
    queries, keys = geometry(case)
    assert_as_exact(grads_of(triton_arm(case)), grads_of(reference_arm(case)),
                    route_draw(queries, keys, jnp.float32), f"interpreted triton dropping {name}",
                    seeds=SEEDS)


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
    mask = np.asarray(dropout_draw(dropout_seed(rng), RATE, (2, 8, 16, 16))) > 0
    np.testing.assert_array_equal(np.asarray(out) > 0, mask.transpose(0, 2, 1, 3))
