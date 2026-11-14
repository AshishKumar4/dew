"""Attention variants composed of the template's calls (docs/design/kernel-templates.md §A rows 52, 53): the
Forgetting Transformer's (`dew.nn.forgetting_attention`) and differential attention
(`dew.nn.differential_attention`), each on par with its definition written out (check 1), its output and
every gradient, the forget gates' included."""

import jax
import jax.numpy as jnp
import pytest
from attention_oracle import qkv
from reference_error import assert_as_exact

from dew.nn.attention import scaled_dot_product_attention
from dew.nn.differential_attention import differential_attention
from dew.nn.forgetting_attention import forget_prefix, forgetting_attention

SEEDS = 8
"""The draws each forgetting site takes (`reference_error.seeds_for`, sized: 8 each)."""

DIFFERENTIAL_SEEDS = 128
"""The draws the differential site takes (`reference_error.seeds_for`, sized: its two softmaxes' outputs
cancel in their difference, so a draw's error reads against less of a truth)."""

ROWS, LENGTH, HEADS, KV_HEADS, WIDTH = 2, 96, 4, 2, 32


def _cotangent(seed: int, shape: tuple) -> jax.Array:
    """A seed's bf16 output cotangent of `shape`."""
    return jax.random.normal(jax.random.PRNGKey(seed + 1000), shape, jnp.bfloat16)


def _pulled(fn):
    """`fn`'s output and its inputs' gradients under the drawn cotangent (the draw's last operand)."""
    def pulled(*operands):
        out, pull = jax.vjp(fn, *operands[:-1])
        return (out, *pull(operands[-1]))

    return pulled


@pytest.mark.parametrize("packed", [False, True], ids=["one-document", "packed"])
def test_forgetting_attention_holds_check_1_against_its_logits_written_out(packed):
    """FoX's causal attention (`forgetting_attention`): each logit plus its query's cumulative log forget gate
    and minus its key's, as a dense bias over the reference path (row 52), packed documents apart where
    `packed`: the output and the queries', keys', values' and forget logits' gradients on par (check 1)."""
    segment_ids = (jnp.repeat(jnp.asarray([[1] * 40 + [2] * 56]), ROWS, axis=0) if packed else None)

    def draw(seed):
        query, key, value = qkv((ROWS, LENGTH, HEADS, WIDTH), jnp.bfloat16, seed=seed,
                                kv_shape=(ROWS, LENGTH, KV_HEADS, WIDTH))
        gates = jax.random.normal(jax.random.PRNGKey(seed + 500), (ROWS, LENGTH, HEADS), jnp.float32) + 2.0
        return query, key, value, gates, _cotangent(seed, (ROWS, LENGTH, HEADS, WIDTH))

    def ours(query, key, value, gates):
        return forgetting_attention(query, key, value, gates, segment_ids=segment_ids)

    def reference(query, key, value, gates):
        prefix = forget_prefix(gates)
        bias = prefix[:, :, :, None] - prefix[:, :, None, :]
        return scaled_dot_product_attention(query, key, value, causal=True, bias=bias,
                                            segment_ids=segment_ids, implementation="reference")

    assert_as_exact(jax.jit(_pulled(ours)), jax.jit(_pulled(reference)), draw,
                    f"forgetting attention {'packed' if packed else 'one document'}", seeds=SEEDS)


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

    assert_as_exact(jax.jit(_pulled(ours)), jax.jit(_pulled(reference)), draw, "differential attention",
                    seeds=DIFFERENTIAL_SEEDS)
