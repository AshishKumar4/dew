"""A decode step's attention over the few query positions it holds: jax.nn's xla attention reading a
cache in its own layout (`folded_attention`), which `dew.nn.attention` dispatches where cudnn does not
run (`folds`), over a query whose heads are grouped as their key head's positions
(`dew.nn.kernels.decode_attention.grouped_query`)."""

import math

import jax
import jax.numpy as jnp

from .kernels import generation
from .precision import at_least_fp32, float32_einsum, operand_precision

FOLDED_PAIRS = 16
"""Most query positions times key heads `folded_attention` takes."""


def folds(query, key, bias, mask, causal, sliding_window) -> bool:
    """Whether an xla call goes to `folded_attention`: on a GPU, a call of
    few query positions over keys of the query's heads, masked at most by a
    mask every head shares (key lengths are one), such as a decode step's
    grouped query (`decode_attention.grouped_query`) where cudnn does not run."""
    return (generation.backend() == 'gpu' and query.shape[-2] == key.shape[-2]
            and query.shape[-3] * key.shape[-2] <= FOLDED_PAIRS
            and (mask is None or (mask.ndim == 4 and mask.shape[1] == 1))
            and bias is None and not causal and sliding_window is None)


def folded_attention(query, key, value, mask=None):
    """jax.nn's xla attention for query `[B, T, N, D]` over keys and values
    `[B, S, N, D]` of the same heads, with every query head against every
    (key, head) pair, under a boolean `mask` `[B, 1, T | 1, S]` every head
    shares (key lengths come as the mask they mean, `with_key_lengths`).

    XLA's GPU products take each head's keys apart, so jax.nn's attention
    transposed the whole cache, keys and values, at every decode step where
    cudnn does not run: a tenth of Qwen3.5-0.8B's decode program at 128
    rows, its 256-wide heads. Here both products read the keys and values
    as `[B, S * N, D]`, their own layout, and the cross-head products are
    thrown away: N times the multiplications, of a step bound by reading
    the cache. The arithmetic is jax.nn's otherwise (fp32 logits of the
    operands, the scale after, an fp32 softmax of logits masked to -0.7
    times the largest fp32, probabilities in the value's dtype). Six layers
    of Qwen3.5's decode took 1.21 against 2.48 ms at 128 rows and 0.33
    against 0.47 at 32 on an RTX 4080 (docs/performance.md).
    """
    batch, positions, heads, width = query.shape
    keys = key.shape[-3]
    wide = at_least_fp32(query.dtype)
    logits = float32_einsum('bxd,byd->bxy', query.reshape(batch, positions * heads, width),
                            key.reshape(batch, keys * heads, width),
                            precision=operand_precision(query.dtype), preferred_element_type=wide)
    logits = jnp.diagonal(logits.reshape(batch, positions, heads, keys, heads), axis1=2, axis2=4)
    logits = logits * jnp.asarray(1 / math.sqrt(width), wide)  # [B, T, S, N]
    if mask is not None:
        valid = jnp.broadcast_to(mask[:, 0], (batch, positions, keys))[..., None]
        logits = jnp.where(valid, logits, jnp.asarray(-0.7 * jnp.finfo(jnp.float32).max, wide))
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=2).astype(value.dtype)
    # Each query head's probabilities over its own head's keys, zeros elsewhere.
    spread = jnp.moveaxis(probs, 3, 2)[..., None] * jnp.eye(heads, dtype=probs.dtype)[:, None, :]
    out = float32_einsum('bxy,byd->bxd', spread.reshape(batch, positions * heads, keys * heads),
                         value.reshape(batch, keys * heads, value.shape[-1]))
    return out.reshape(batch, positions, heads, value.shape[-1])
