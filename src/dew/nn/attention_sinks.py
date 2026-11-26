"""Attention sinks: a learned, value-free logit a head in the softmax's denominator
(GPT OSS, MiMo-V2, DeepSeek V4)."""

import jax
import jax.numpy as jnp
from flax.typing import Dtype

from dew.nn.derivatives import portable_formed


def sink_softmax(logits: jax.Array, sinks: jax.Array, dtype: Dtype | None = None) -> jax.Array:
    """The softmax of `logits` `[..., heads, queries, keys]` over the keys with
    a sink's logit in its denominator and no value, so a row sums to less than
    one: each head's of `sinks` `[heads]`, or each query's own `[batch, queries,
    heads]` (`sink_column`). The softmax runs at `dtype`, the caller's
    precision policy, else at the type the logits and sinks promote to.

    The sink stays in logit space beside the keys, as the eager reference
    concatenates it, so the order of the softmax holds when a sink dominates
    the real keys. Its reverse mode returns the logits' cotangent only with the
    sinks' gradient formed (`derivatives.with_gradients_formed`): a reduction
    over every query's sink column, which XLA otherwise put off to the step's
    end, each layer's `[heads, queries, keys + 1]` holding to it (a DeepSeek V4
    decoder's temporaries 499 MB from two blocks to four on two H100s, row 6027).
    Its values and every derivative are XLA's of `_weights`, the same arithmetic;
    its forward saves nothing, so its backward recomputes the weights (one more
    exp and sum over `[heads, queries, keys + 1]`) in place of the probabilities
    autodiff kept.
    """
    def portable(logits, sinks):
        return _weights(logits, sinks, dtype)

    return portable_formed(portable)(logits, sinks)


def _weights(logits: jax.Array, sinks: jax.Array, dtype: Dtype | None) -> jax.Array:
    """`sink_softmax`'s weights, the softmax over the keys and each row's sink."""
    sink = sink_column(sinks, logits.shape)
    combined = jnp.concatenate((logits, sink), axis=-1)
    if dtype is not None:
        combined = combined.astype(dtype)
    return jax.nn.softmax(combined, axis=-1)[..., :-1]


def sink_column(sinks: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    """`sinks` as the column `[..., heads, queries, 1]` beside logits of `shape` `[..., heads, queries,
    keys]`: a head's `[heads]` for every row and query, or each query's own `[batch, queries, heads]`."""
    column = sinks[:, None, None] if sinks.ndim == 1 else jnp.swapaxes(sinks, -1, -2)[..., None]
    return jnp.broadcast_to(column, (*shape[:-1], 1))
