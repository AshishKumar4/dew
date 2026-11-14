"""Attention variants composed of the template's calls (docs/design/kernel-templates.md §A rows 52, 53): the
Forgetting Transformer's (`dew.nn.forgetting_attention`) and differential attention
(`dew.nn.differential_attention`), each on par with its definition written out (check 1), its output and
every gradient, the forget gates' included."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from attention_oracle import qkv
from flax import linen as nn
from jax.sharding import Mesh, PartitionSpec
from reference_error import assert_as_exact, with_gradients
from sharded import assert_runs_per_shard

from dew.mesh import DEFAULT_RULES
from dew.nn.attention import scaled_dot_product_attention
from dew.nn.differential_attention import differential_attention
from dew.nn.forgetting_attention import forgetting_attention

SEEDS = 8
"""The draws each forgetting site takes (`reference_error.seeds_for`, sized: 8 each)."""

DIFFERENTIAL_SEEDS = 128
"""The draws the differential site takes (`reference_error.seeds_for`, sized: its two softmaxes' outputs
cancel in their difference, so a draw's error reads against less of a truth)."""

MESH_SEEDS = 8
"""The draws the mesh site takes (`reference_error.seeds_for`, sized)."""

ROWS, LENGTH, HEADS, KV_HEADS, WIDTH = 2, 96, 4, 2, 32
MESH_TENSOR, MESH_HEADS = 4, 8


def _cotangent(seed: int, shape: tuple) -> jax.Array:
    """A seed's bf16 output cotangent of `shape`."""
    return jax.random.normal(jax.random.PRNGKey(seed + 1000), shape, jnp.bfloat16)


def _gated(heads: int, kv_heads: int):
    """A seed's bf16 queries `[B, S, heads, D]`, keys and values of `kv_heads`, fp32 forget logits
    `[B, S, heads]` (a gate near 0.9), and output cotangent."""
    def draw(seed):
        query, key, value = qkv((ROWS, LENGTH, heads, WIDTH), jnp.bfloat16, seed=seed,
                                kv_shape=(ROWS, LENGTH, kv_heads, WIDTH))
        gates = jax.random.normal(jax.random.PRNGKey(seed + 500), (ROWS, LENGTH, heads), jnp.float32) + 2.0
        return query, key, value, gates, _cotangent(seed, (ROWS, LENGTH, heads, WIDTH))

    return draw


def _forgetting(segment_ids=None):
    """FoX's attention on the template, and its logits written out as a dense bias on the reference path: each
    plus its query's cumulative log forget gate and minus its key's."""
    def ours(query, key, value, gates):
        return forgetting_attention(query, key, value, gates, segment_ids=segment_ids,
                                    implementation="template")

    def reference(query, key, value, gates):
        prefix = jnp.cumsum(jax.nn.log_sigmoid(gates), axis=1).transpose(0, 2, 1)
        bias = prefix[:, :, :, None] - prefix[:, :, None, :]
        return scaled_dot_product_attention(query, key, value, causal=True, bias=bias,
                                            segment_ids=segment_ids, implementation="reference")

    return ours, reference


@pytest.mark.parametrize("packed", [False, True], ids=["one-document", "packed"])
def test_forgetting_attention_holds_check_1_against_its_logits_written_out(packed):
    """FoX's causal attention on the template (`forgetting_attention`, its gates' prefixes a separable bias):
    each logit plus its query's cumulative log forget gate and minus its key's, written out here as a dense
    bias over the reference path (row 52), packed documents apart where `packed`: the output and the
    queries', keys', values' and forget logits' gradients on par (check 1)."""
    segment_ids = (jnp.repeat(jnp.asarray([[1] * 40 + [2] * 56]), ROWS, axis=0) if packed else None)
    ours, reference = _forgetting(segment_ids)
    assert_as_exact(jax.jit(with_gradients(ours)), jax.jit(with_gradients(reference)),
                    _gated(HEADS, KV_HEADS), f"forgetting attention {'packed' if packed else 'one document'}",
                    seeds=SEEDS)


@pytest.mark.mesh(devices=8)
def test_forgetting_attention_runs_per_shard_on_a_data_and_head_mesh():
    """FoX on the template on a mesh of two rows by four heads (`template_attention`'s per-shard call, its
    gates' terms split with their rows and heads): it runs per shard (`sharded.assert_runs_per_shard`), its
    output and its query, key, value and forget-logit gradients on par with one device's (check 1) from
    float64 on its logits written out, and the compiled step moves no operand between devices."""
    mesh = Mesh(np.asarray(jax.devices()[:ROWS * MESH_TENSOR]).reshape(ROWS, MESH_TENSOR), ("data", "tensor"))
    rows_and_heads, rows = PartitionSpec("data", None, "tensor"), PartitionSpec("data")
    ours, reference = _forgetting()
    with nn.logical_axis_rules(DEFAULT_RULES):
        assert_runs_per_shard(with_gradients(ours), _gated(MESH_HEADS, KV_HEADS),
                              [rows_and_heads, rows, rows, rows_and_heads, rows_and_heads], mesh,
                              "forgetting attention over rows and heads", seeds=MESH_SEEDS,
                              truth_fn=with_gradients(reference))


def test_differential_attention_holds_check_1_against_its_two_softmaxes_written_out():
    """Differential attention (`differential_attention`): the first softmax's output less a head's `λ` times
    the second's, over one value, against each written out on the reference path (row 53): the output and
    the queries', keys', value's and `λ`'s gradients on par (check 1)."""
    def draw(seed):
        first = qkv((ROWS, LENGTH, HEADS, WIDTH), jnp.bfloat16, seed=seed,
                    kv_shape=(ROWS, LENGTH, KV_HEADS, WIDTH))
        second = qkv((ROWS, LENGTH, HEADS, WIDTH), jnp.bfloat16, seed=seed + 100,
                     kv_shape=(ROWS, LENGTH, KV_HEADS, WIDTH))
        lambdas = jax.random.uniform(jax.random.PRNGKey(seed + 700), (HEADS,), jnp.float32, 0.1, 0.9)
        return (first[0], first[1], second[0], second[1], first[2], lambdas,
                _cotangent(seed, (ROWS, LENGTH, HEADS, WIDTH)))

    def ours(q1, k1, q2, k2, value, lambdas):
        return differential_attention((q1, q2), (k1, k2), value, lambdas)

    def reference(q1, k1, q2, k2, value, lambdas):
        first, second = (scaled_dot_product_attention(query, key, value, causal=True,
                                                      implementation="reference")
                         for query, key in ((q1, k1), (q2, k2)))
        return first - lambdas[None, None, :, None].astype(second.dtype) * second

    assert_as_exact(jax.jit(with_gradients(ours)), jax.jit(with_gradients(reference)), draw,
                    "differential attention", seeds=DIFFERENTIAL_SEEDS)
