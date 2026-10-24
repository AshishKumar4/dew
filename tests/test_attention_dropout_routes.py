"""Attention's probability dropout on the Triton mainloop, and on a mesh: the interpreted Triton forward and
backward are on par with the reference path on the same draw (check 1), over causality, documents, and
sinks with a softcap, the draw's four query heads grouped over two key heads; and a dropping call on a data
and head mesh keeps the entries one device's draw keeps, the draw taking each entry's global batch row and
query head through `_per_shard`.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from reference_error import assert_as_exact
from test_attention_dropout import RATE, dropout_of, grads_of, keyed_draw, reference_arm, seed_words
from test_attention_template import CASES, ROWS, SEEDS, arguments, geometry
from test_attention_triton import hooks_for

from dew.nn.attention import dropout_draw, dropout_seed, scaled_dot_product_attention
from dew.nn.kernels.templates import attention as template, attention_triton
from dew.nn.sharding import DEFAULT_RULES

FORMS = (
    ("causal-window", frozenset()),
    ("causal-documents-window", frozenset()),
    ("causal-window", frozenset({"softcap", "sinks"})),
)
"""The forms the interpreted Triton check drops over: causality under a window, documents under a window,
and a causal window with sinks and a softcap."""


def triton_arm(name: str, names: frozenset):
    """The interpreted Triton forward and backward of `name`'s spans with the score hooks `names`, dropping
    from the call's key, as `(query, key, value, rng, dout)` to the output and the query, key and value
    gradients."""
    case = CASES[name]
    queries, keys = geometry(case)
    spans = template.spans(queries, keys, ROWS, **arguments(case))

    def run(query, key, value, rng, dout):
        base = hooks_for(names, queries, value.shape[-1])
        hooks = dataclasses.replace(base, drop=dropout_of(dropout_seed(rng), ROWS, query.shape[2]))
        out = attention_triton.forward(query, key, value, spans, hooks, interpret=True)
        grads = attention_triton.backward(query, key, value, spans, hooks, dout, interpret=True)
        return out, grads.query, grads.key, grads.value

    return run


@pytest.mark.parametrize("dtype", attention_triton.DTYPES, ids=lambda dtype: jnp.dtype(dtype).name)
@pytest.mark.parametrize("form", FORMS, ids=lambda form: "+".join([form[0], *sorted(form[1])]))
def test_the_interpreted_triton_mainloop_drops_on_par_with_the_reference_path(form, dtype):
    """The Triton forward and backward, interpreted, on the same draw as the reference path: the output and
    the query, key and value gradients, both measured from float64 (check 1). The sinks are the call's
    constants here, as in the Triton on-par tests, so their gradient is not measured by this check."""
    name, names = form
    case = CASES[name]
    queries, keys = geometry(case)
    assert_as_exact(jax.jit(triton_arm(name, names)), jax.jit(grads_of(reference_arm(case, names))),
                    keyed_draw(queries, keys, dtype),
                    f"interpreted triton dropping {name} {sorted(names)} {jnp.dtype(dtype).name}",
                    seeds=SEEDS)


def test_the_interpreted_triton_mainloop_keeps_the_reference_mask():
    """The Triton forward, interpreted, keeps the entries the reference's draw keeps, bit for bit: uniform
    weights and identity values make each output entry its own draw times `1 / keys`, so its nonzero entries
    are the kept ones, against `dropout_draw`'s mask over the same coordinates (the XLA draw)."""
    words = seed_words(3)
    query = jnp.zeros((2, 8, 4, 8), jnp.float32)
    key = jnp.zeros((2, 16, 2, 8), jnp.float32)
    value = jnp.broadcast_to(jnp.eye(16, dtype=jnp.float32)[None, :, None, :], (2, 16, 2, 16))
    spans = template.spans(8, 16, 2)
    out = attention_triton.forward(query, key, value, spans, template.Hooks(drop=dropout_of(words, 2, 4)),
                                   interpret=True)
    mask = np.asarray(dropout_draw(words, RATE, (2, 4, 8, 16))) > 0
    np.testing.assert_array_equal(np.asarray(out) > 0, mask.transpose(0, 2, 1, 3))


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
