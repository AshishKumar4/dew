"""The categorical distribution over a class axis: the one owner of its log-likelihood in Dew.

A model's logits over a vocabulary, a classifier's over its classes, a decision's over a question's
options and a sampler's over its next token are each a categorical distribution over the last axis.
Every log-likelihood Dew takes of one composes on this module: the normalized log probabilities
(`log_probs`), a label's cross entropy (`cross_entropy`), a soft target's (`soft_cross_entropy`), a
divergence from a reference distribution (`kl_divergence`), the entropy (`entropy`), and, for a head
that reads its vocabulary in blocks (`dew.objectives.lm.chunked`), the running partition it combines
block by block (`rescaled`) and a label's cross entropy from it (`target_loss`).

Each computes in at least float32 (`dew.nn.precision.at_least_fp32`): a bf16 log partition over a
vocabulary carries bf16's 8-bit mantissa into every score, error in the second decimal of a log
probability near -10, while float64 logits stay float64. `where` masks the classes a row does not
hold, as a decision's padded option slots: they take no probability.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from dew.nn.precision import at_least_fp32


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


def cross_entropy(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """`-log p[labels]` over the last axis, `labels` of `logits.shape[:-1]`: the log partition less the
    label's logit, as Optax takes it, which reads the label's logit without building the normalized
    row."""
    return optax.softmax_cross_entropy_with_integer_labels(widened(logits), labels)


def soft_cross_entropy(logits: jax.Array, targets: jax.Array, where: jax.Array | None = None) -> jax.Array:
    """`-sum_k t_k log p_k` over the last axis against the distribution `targets`, a class of no target
    mass contributing nothing, whatever its log probability."""
    return -jnp.sum(jnp.where(targets > 0, targets * log_probs(logits, where), 0.0), axis=-1)


def kl_divergence(logits: jax.Array, reference: jax.Array) -> jax.Array:
    """`KL(q || p)` over the last axis, `p` the categorical of `logits` and `q` that of the `reference`
    logits (Optax's `kl_divergence` of the two)."""
    return optax.kl_divergence(log_probs(logits), probabilities(reference))


def entropy(logits: jax.Array) -> jax.Array:
    """`-sum_k p_k log p_k` over the last axis."""
    log_p = log_probs(logits)
    return -(jnp.exp(log_p) * log_p).sum(axis=-1)


def rescaled(maximum: jax.Array, total: jax.Array, larger: jax.Array) -> jax.Array:
    """A sum of exponentials moved from its own maximum to the larger one, as a blockwise partition
    combines its blocks; unchanged where they agree, so two sums whose maximum is -inf add rather than
    turn NaN."""
    return jnp.where(maximum == larger, total, total * jnp.exp(maximum - larger))


def target_loss(peak: jax.Array, mass: jax.Array, target: jax.Array) -> jax.Array:
    """A label's cross entropy from its row's maximum `peak`, the sum of exponentials scaled to it
    `mass`, and its label's logit: log(mass) - (target - peak), as `cross_entropy` takes it, so neither
    term is rounded at the logits' magnitude before the subtraction that cancels it does."""
    return jnp.log(mass) - (target - peak)
