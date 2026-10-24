"""The categorical distribution over a class axis: the one owner of its log-likelihood in Dew.

A model's logits over a vocabulary, a classifier's over its classes, a decision's over a question's
options and a sampler's over its next token are each a categorical distribution over the last axis.
Every log-likelihood Dew takes of one composes on this module: the normalized log probabilities
(`log_probs`), a label's cross entropy (`cross_entropy`), a soft target's (`soft_cross_entropy`), a
divergence from a reference distribution (`kl_divergence`), the entropy (`entropy`), and, for a head
that reads its vocabulary in blocks (`dew.objectives.lm.chunked`), the running partition it combines
block by block (`rescaled`), a label's cross entropy from it (`target_loss`) and the log partition of
each segment of a ragged row (`segment_log_partition`), a support's.

Each computes in at least float32 (`dew.nn.precision.at_least_fp32`): a bf16 log partition over a
vocabulary carries bf16's 8-bit mantissa into every score, error in the second decimal of a log
probability near -10, while float64 logits stay float64. `where` masks the classes a row does not
hold, as a decision's padded option slots: they take no probability, and a class of no probability adds
no term to a sum over the classes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from dew.nn.precision import at_least_fp32

__all__ = ["cross_entropy", "entropy", "kl_divergence", "log_probs", "probabilities", "rescaled",
           "segment_log_partition", "soft_cross_entropy", "target_loss"]


def widened(logits: jax.Array, where: jax.Array | None = None) -> jax.Array:
    """`logits` at the precision a categorical is computed in, at least float32, with the classes
    `where` excludes at -inf."""
    logits = logits.astype(at_least_fp32(logits.dtype))
    return logits if where is None else jnp.where(where, logits, -jnp.inf)


def log_probs(logits: jax.Array, where: jax.Array | None = None) -> jax.Array:
    """The normalized log probabilities over the last axis (`jax.nn.log_softmax`)."""
    return jax.nn.log_softmax(widened(logits, where), axis=-1)


def probabilities(logits: jax.Array, where: jax.Array | None = None) -> jax.Array:
    """The probabilities over the last axis (`jax.nn.softmax`)."""
    return jax.nn.softmax(widened(logits, where), axis=-1)


def cross_entropy(logits: jax.Array, labels: jax.Array, where: jax.Array | None = None) -> jax.Array:
    """`-log p[labels]` over the last axis, `labels` of `logits.shape[:-1]`: the log partition less the
    label's logit, as Optax takes it, which reads the label's logit without building the normalized
    row."""
    return optax.softmax_cross_entropy_with_integer_labels(widened(logits, where), labels)


def soft_cross_entropy(logits: jax.Array, targets: jax.Array, where: jax.Array | None = None) -> jax.Array:
    """`-sum_k t_k log p_k` over the last axis against the distribution `targets`, a class of no target
    mass contributing nothing, whatever its log probability."""
    return -jnp.sum(jnp.where(targets > 0, targets * log_probs(logits, where), 0.0), axis=-1)


def kl_divergence(logits: jax.Array, reference: jax.Array, where: jax.Array | None = None) -> jax.Array:
    """`KL(q || p)` over the last axis, `p` the categorical of `logits` and `q` that of the `reference`
    logits (Optax's `kl_divergence` of the two), the classes `where` excludes adding no term."""
    log_p, q = log_probs(logits, where), probabilities(reference, where)
    if where is not None:
        # An excluded class enters Optax's sum as log 1 against a log probability of 0, so neither its
        # value nor its gradient meets 0 * inf.
        log_p, q = jnp.where(where, log_p, 0.0), jnp.where(where, q, 1.0)
    return optax.kl_divergence(log_p, q, where=where)


def entropy(logits: jax.Array, where: jax.Array | None = None) -> jax.Array:
    """`-sum_k p_k log p_k` over the last axis, a class of no probability (a -inf logit, or one `where`
    excludes) adding no term, in value or gradient."""
    log_p = log_probs(logits, where)
    held = jnp.where(jnp.isneginf(log_p), 0.0, log_p)
    return -(jnp.exp(log_p) * held).sum(axis=-1)


def _wide(*operands: jax.Array) -> tuple[jax.Array, ...]:
    """`operands` in their common dtype at least float32."""
    dtype = at_least_fp32(jnp.result_type(*operands))
    return tuple(jnp.asarray(operand).astype(dtype) for operand in operands)


def rescaled(maximum: jax.Array, total: jax.Array, larger: jax.Array) -> jax.Array:
    """A sum of exponentials moved from its own maximum to the larger one, as a blockwise partition
    combines its blocks; unchanged where they agree, so two sums whose maximum is -inf add rather than
    turn NaN."""
    maximum, total, larger = _wide(maximum, total, larger)
    return jnp.where(maximum == larger, total, total * jnp.exp(maximum - larger))


def target_loss(peak: jax.Array, mass: jax.Array, target: jax.Array) -> jax.Array:
    """A label's cross entropy from its row's maximum `peak`, the sum of exponentials scaled to it
    `mass`, and its label's logit: log(mass) - (target - peak), as `cross_entropy` takes it, so neither
    term is rounded at the logits' magnitude before the subtraction that cancels it does."""
    peak, mass, target = _wide(peak, mass, target)
    return jnp.log(mass) - (target - peak)


def segment_log_partition(logits: jax.Array, segments: jax.Array, count: int) -> tuple[jax.Array, jax.Array]:
    """The log partition `[count]` of each segment of a ragged row of logits `[C]`, `segments` naming
    each entry's (`count` for an entry of none), and whether each segment holds an entry: its maximum,
    held off the gradient as `jax.nn.logsumexp` holds it, plus the log of its exponentials' sum scaled to
    it, a segment of no entry's log partition 0. An entry's log probability within its segment is its
    logit less its segment's log partition."""
    logits = widened(logits)
    real = segments < count
    peak = jax.ops.segment_max(jnp.where(real, jax.lax.stop_gradient(logits), -jnp.inf), segments,
                               num_segments=count + 1)
    present = jnp.isfinite(peak)
    peak = jnp.where(present, peak, 0.0)
    # An entry of no segment exponentiates -inf, so no overflowed entry reaches the backward.
    mass = jax.ops.segment_sum(jnp.exp(jnp.where(real, logits - peak[segments], -jnp.inf)), segments,
                               num_segments=count + 1)[:count]
    return jnp.log(jnp.where(present[:count], mass, 1.0)) + peak[:count], present[:count]
