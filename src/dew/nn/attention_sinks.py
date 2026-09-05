"""GPT OSS attention's learned, value-free softmax entry."""

import math

import jax
import jax.numpy as jnp
from flax.linen.dtypes import promote_dtype
from flax.typing import Dtype, PrecisionLike


def attention_with_sinks(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    sinks: jax.Array,
    *,
    mask: jax.Array | None = None,
    bias: jax.Array | None = None,
    dtype: Dtype | None = None,
    precision: PrecisionLike = None,
    force_fp32_for_softmax: bool = True,
) -> jax.Array:
    """Attend over [batch, sequence, heads, width] with one sink per query head.

    JAX's attention primitive cannot put a learned logit in the denominator
    without adding a value. Keeping the sink in logit space also preserves
    the eager reference's softmax order when a sink dominates the real keys.
    Grouped einsums avoid materializing repeated keys and values for GQA.
    """
    query, key, value = promote_dtype(query, key, value, dtype=dtype)
    batch, length, heads, width = query.shape
    kv_heads = key.shape[-2]
    if heads % kv_heads:
        raise ValueError("attention sinks require query heads divisible by key/value heads")
    if sinks.shape != (heads,):
        raise ValueError(f"sinks must have one logit per query head, got {sinks.shape}")
    groups = heads // kv_heads
    queries = query.reshape(batch, length, kv_heads, groups, width)
    logits = jnp.einsum("btkgd,bskd->bkgts", queries, key, precision=precision)
    logits = logits.reshape(batch, heads, length, key.shape[1]) / math.sqrt(width)
    if bias is not None:
        logits = logits + bias
    if mask is not None:
        logits = jnp.where(mask, logits, jnp.finfo(logits.dtype).min)
    sink_logits = jnp.broadcast_to(sinks[None, :, None, None], (*logits.shape[:-1], 1))
    combined = jnp.concatenate((logits, sink_logits), axis=-1)
    if force_fp32_for_softmax:
        combined = combined.astype(jnp.float32)
    combined = combined - jnp.max(combined, axis=-1, keepdims=True)
    scores = jax.nn.softmax(combined, axis=-1)[..., :-1].astype(value.dtype)
    scores = scores.reshape(batch, kv_heads, groups, length, key.shape[1])
    output = jnp.einsum("bkgts,bskd->btkgd", scores, value, precision=precision)
    return output.reshape(batch, length, heads, value.shape[-1])
