"""The log probability a categorical over the last axis gives one token.

It lives beside `objectives.base` rather than in `objectives.lm` with the
chunked loss, because `objectives.lm` imports `sampling.text`, and sampling
is one of this function's callers.
"""

import jax
import jax.numpy as jnp


def token_log_probs(logits: jax.Array, tokens: jax.Array) -> jax.Array:
    """`log_softmax(logits)[..., tokens]`, reduced in fp32 whatever `logits` holds.

    `tokens` is `logits.shape[:-1]` of ids and the result has that shape. A
    bf16 log partition over a vocabulary carries bf16's 8-bit mantissa into
    every score, which is error in the second decimal of a log probability
    near -10; fp32 carries the logits exactly and rounds only the reduction.
    """
    normalized = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    return jnp.take_along_axis(normalized, tokens[..., None], axis=-1)[..., 0]
