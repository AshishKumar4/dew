"""Attention sinks: a learned, value-free logit a head in the softmax's denominator
(GPT OSS, MiMo-V2, DeepSeek V4)."""

import math

import jax
import jax.numpy as jnp
from flax.linen.dtypes import promote_dtype
from flax.typing import Dtype, PrecisionLike

from .precision import at_least_fp32


def sink_softmax(logits: jax.Array, sinks: jax.Array, dtype: Dtype | None = None) -> jax.Array:
    """The softmax of `logits` `[..., heads, queries, keys]` over the keys with
    each head's logit of `sinks` `[heads]` in its denominator and no value, so
    a row sums to less than one. The softmax runs at `dtype`, the caller's
    precision policy, else at the type the logits and sinks promote to.

    The sink stays in logit space beside the keys, as the eager reference
    concatenates it, so the order of the softmax holds when a sink dominates
    the real keys.
    """
    sink = jnp.broadcast_to(sinks[:, None, None], (*logits.shape[:-1], 1))
    combined = jnp.concatenate((logits, sink), axis=-1)
    if dtype is not None:
        combined = combined.astype(dtype)
    return jax.nn.softmax(combined, axis=-1)[..., :-1]


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
    without adding a value, so the scores are `sink_softmax`'s. Grouped
    einsums avoid materializing repeated keys and values for GQA.
    """
    query, key, value = promote_dtype(query, key, value, dtype=dtype)
    batch, length, heads, width = query.shape
    kv_heads = key.shape[-2]
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
    wide = jnp.promote_types(logits.dtype, sinks.dtype)
    scores = sink_softmax(logits, sinks, at_least_fp32(wide) if force_fp32_for_softmax else None)
    scores = scores.astype(value.dtype).reshape(batch, kv_heads, groups, length, key.shape[1])
    output = jnp.einsum("bkgts,bskd->btkgd", scores, value, precision=precision)
    return output.reshape(batch, length, heads, value.shape[-1])
