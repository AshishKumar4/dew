"""Cross entropy that never holds the whole logits tensor.

The vocabulary projection is the largest tensor in a language-model step. At
the small benchmark preset it is 8,192 tokens by 50,304 float32 logits, 1.57
GiB, and the loss walks it several times: the maximum, the sum of
exponentials, the target gather, the top-1 prediction, then the softmax again
in the backward pass. Splitting the vocabulary into chunks turns that into
`chunks` tiles produced, reduced and dropped one at a time. All the loss keeps
per token is the running logsumexp, the target logit and the best logit seen
so far, four float32 numbers instead of a row of 50,304.

The arithmetic is the arithmetic of one pass. logsumexp over a concatenation
is `logaddexp` of the parts' logsumexps, the target logit lives in exactly one
chunk, and a maximum over parts taken in vocabulary order with a strict
comparison keeps the lowest index among equals, which is what `jnp.argmax`
returns for the whole row. The tile matmuls accumulate in float32 and the
reductions run in float32, so the numbers are the ones the full pass produced.

The tiles are kept for the backward pass rather than recomputed there. That is
measured, not assumed: on one RTX 4080 at this shape, keeping them costs 0.32
GiB and runs the head forward and backward in 49.71 ms, while `jax.checkpoint`
around the tile saves the 0.32 GiB and costs 67.19 ms, because recomputing
every tile is a fourth pass of the head matmul. Storing four tiles is still
1.08 GiB below the full-vocabulary path, which had to hold the logits and
their softmax gradient at once. MaxText's vocabulary tiling recomputes instead,
behind a hand-written `custom_vjp` over a `lax.scan`, because its tiles carry
sharding constraints across a much larger vocabulary and many devices
(https://github.com/AI-Hypercomputer/maxtext, maxtext/utils/vocabulary_tiling.py).
The numbers above are what a single 16 GiB card says instead; the reproduction
is in docs/research/lm-head.md.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax.typing import PrecisionLike


def vocabulary_chunks(vocab_size: int, chunks: int) -> Tuple[Tuple[int, int], ...]:
    """`chunks` half-open column ranges covering `range(vocab_size)`.

    The last chunk is the short one when the count does not divide the
    vocabulary. Uneven tiles are free here: the loop over them is a Python
    loop, so each tile is its own matmul and nothing has to be padded to a
    common width and masked back out.
    """
    if chunks < 1:
        raise ValueError(f"a vocabulary needs at least one chunk, got {chunks}")
    if chunks > vocab_size:
        raise ValueError(
            f"{chunks} chunks for a vocabulary of {vocab_size} leaves empty tiles")
    width = -(-vocab_size // chunks)
    bounds = tuple((start, min(start + width, vocab_size))
                   for start in range(0, vocab_size, width))
    if len(bounds) != chunks:
        # ceil division can cover the vocabulary early (13 columns in 4 chunks
        # is 4 + 4 + 4 + 1, not 4 + 4 + 4 + 4 + 1), so the caller's count is
        # the number of tiles, not a hint.
        raise ValueError(
            f"{vocab_size} columns do not split into {chunks} chunks of {width}")
    return bounds


def _chunk_terms(hidden, head_chunk, targets, start: int, stop: int,
                 softcap: Optional[float], precision: PrecisionLike):
    """One tile's logsumexp, target logit, best logit and its column."""
    logits = jnp.einsum('td,dv->tv', hidden, head_chunk,
                        precision=precision, preferred_element_type=jnp.float32)
    if softcap is not None:
        cap = jnp.asarray(softcap, jnp.float32)
        logits = cap * jnp.tanh(logits / cap)

    inside = (targets >= start) & (targets < stop)
    column = jnp.clip(targets - start, 0, stop - start - 1)
    picked = jnp.take_along_axis(logits, column[:, None], axis=-1)[:, 0]
    return (jax.nn.logsumexp(logits, axis=-1),
            jnp.where(inside, picked, 0.0),
            jnp.max(logits, axis=-1),
            jnp.argmax(logits, axis=-1) + start)


def chunked_cross_entropy(hidden, head_weight, targets, chunks: int, *,
                          softcap: Optional[float] = None,
                          precision: PrecisionLike = None):
    """Per-token cross entropy of `hidden @ head_weight` and its top-1 column.

    `hidden` is `[..., features]` states in any compute dtype, `head_weight`
    the `[features, vocab]` float32 head, `targets` the `[...]` int32 ids.
    Returns the per-token losses and the argmax prediction, both shaped like
    `targets`: the caller owns the weighting and the mean, which is where a
    padding id has to be honored.

    `softcap` is the backbone's `final_logit_softcap`. It is elementwise, so
    capping a tile and capping the row agree.
    """
    features = head_weight.shape[0]
    if hidden.shape[-1] != features:
        raise ValueError(
            f"hidden states are {hidden.shape[-1]} wide and the head is {features}")
    if hidden.shape[:-1] != targets.shape:
        raise ValueError(
            f"{targets.shape} targets for {hidden.shape[:-1]} hidden states")

    bounds = vocabulary_chunks(head_weight.shape[1], chunks)
    # fp32 states, as the full-vocabulary head did before the loss chunked it.
    flat = hidden.astype(jnp.float32).reshape(-1, features)
    flat_targets = targets.reshape(-1)

    total = jnp.full(flat_targets.shape, -jnp.inf, jnp.float32)
    target_logit = jnp.zeros(flat_targets.shape, jnp.float32)
    best = jnp.full(flat_targets.shape, -jnp.inf, jnp.float32)
    predicted = jnp.zeros(flat_targets.shape, jnp.int32)

    for start, stop in bounds:
        chunk_lse, picked, chunk_best, chunk_column = _chunk_terms(
            flat, head_weight[:, start:stop], flat_targets, start, stop,
            softcap, precision)
        total = jnp.logaddexp(total, chunk_lse)
        target_logit = target_logit + picked
        # Strictly greater, and the chunks run in vocabulary order, so among
        # equal logits the lowest column wins: jnp.argmax's rule on the row.
        better = chunk_best > best
        best = jnp.where(better, chunk_best, best)
        predicted = jnp.where(better, chunk_column, predicted)

    return ((total - target_logit).reshape(targets.shape),
            predicted.reshape(targets.shape))
