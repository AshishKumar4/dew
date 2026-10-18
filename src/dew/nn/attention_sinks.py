"""Attention sinks: a learned, value-free logit a head in the softmax's denominator
(GPT OSS, MiMo-V2, DeepSeek V4)."""

import jax
import jax.numpy as jnp
from flax.typing import Dtype


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
