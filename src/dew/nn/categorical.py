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

import functools

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
    row. Its derivative is Barron, Gnanapragasam and Tschannen's (arXiv:2610.31642, eq. 7): each other
    class's probability, and at the label minus their sum, so the row's derivative sums to zero. The
    label's `p - 1` collapses to 0 once its margin over the other classes passes the dtype's precision
    (eq. 4; 16.6 in float32), while the others' probabilities stay until they underflow."""
    return _label_loss(widened(logits, where), labels)


@jax.custom_jvp
def _label_loss(logits: jax.Array, labels: jax.Array) -> jax.Array:
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels)


@_label_loss.defjvp
def _label_loss_tangent(primals, tangents):
    logits, labels = primals
    label = jax.nn.one_hot(labels, logits.shape[-1], dtype=logits.dtype)
    others = jnp.where(label > 0, 0.0, jax.nn.softmax(logits, axis=-1))
    derivative = others - label * others.sum(axis=-1, keepdims=True)
    return _label_loss(logits, labels), jnp.sum(derivative * tangents[0], axis=-1)


def soft_cross_entropy(logits: jax.Array, targets: jax.Array, where: jax.Array | None = None) -> jax.Array:
    """`-sum_k t_k log p_k` over the last axis against the distribution `targets`, a class of no target
    mass contributing nothing, whatever its log probability."""
    return -jnp.sum(jnp.where(targets > 0, targets * log_probs(logits, where), 0.0), axis=-1)


def kl_divergence(logits: jax.Array, reference: jax.Array, where: jax.Array | None = None) -> jax.Array:
    """`KL(q || p)` over the last axis, `p` the categorical of `logits` and `q` that of the `reference`
    logits (Optax's `kl_divergence` of the two), the classes `where` excludes adding no term.

    Its derivative in `logits`, `p - q`, is taken from the two rows' difference in log space, as Barron,
    Gnanapragasam and Tschannen take it (arXiv:2610.31642, eq. 9), not by subtracting two softmaxes
    (eq. 8), which near each other keep only their own rounding: logits equal to the reference's have
    derivative exactly zero, and near them it keeps the difference's precision. A row where a class is
    -inf in one and not the other takes `p - q`; the derivative in `reference` is autodiff's."""
    held = jnp.ones(logits.shape, bool) if where is None else where
    return _divergence(widened(logits, where), widened(reference, where), held, where is not None)


def _divergence_value(logits: jax.Array, reference: jax.Array, where: jax.Array, masked: bool) -> jax.Array:
    log_p, q = jax.nn.log_softmax(logits, axis=-1), jax.nn.softmax(reference, axis=-1)
    if not masked:
        return optax.kl_divergence(log_p, q)
    # An excluded class enters Optax's sum as log 1 against a log probability of 0, so neither its value
    # nor its gradient meets 0 * inf.
    return optax.kl_divergence(jnp.where(where, log_p, 0.0), jnp.where(where, q, 1.0), where=where)


@functools.partial(jax.custom_jvp, nondiff_argnums=(3,))
def _divergence(logits: jax.Array, reference: jax.Array, where: jax.Array, masked: bool) -> jax.Array:
    return _divergence_value(logits, reference, where, masked)


@_divergence.defjvp
def _divergence_tangent(masked, primals, tangents):
    logits, reference, where = primals
    d_logits, d_reference, _ = tangents
    value, moved = jax.jvp(lambda reference: _divergence_value(logits, reference, where, masked),
                           (reference,), (d_reference,))
    return value, moved + jnp.sum(_divergence_derivative(logits, reference) * d_logits, axis=-1)


def _divergence_derivative(logits: jax.Array, reference: jax.Array) -> jax.Array:
    """`p - q` over the last axis, `p` the softmax of `logits`, `q` of `reference`, from their difference
    `delta` (arXiv:2610.31642, eq. 9, its `p` here `q` and its `q` here `p`): `d`, `delta` less its mean
    under `q`; `s = log(p / q)`, `d` less `log1p` of `q`'s mean of `expm1(d)`; `u = q expm1(s)`; and
    `u` less `p` times its sum, which makes the derivative sum to zero."""
    log_q = jax.nn.log_softmax(reference, axis=-1)
    q, p = jnp.exp(log_q), jax.nn.softmax(logits, axis=-1)
    both = jnp.isfinite(logits) & jnp.isfinite(reference)
    alike = jnp.all(both | (jnp.isneginf(logits) & jnp.isneginf(reference)), axis=-1, keepdims=True)
    delta = jnp.where(both, logits - reference, 0.0)
    d = delta - jnp.sum(q * delta, axis=-1, keepdims=True)
    s = d - jnp.log1p(jnp.sum(_scaled_expm1(q, log_q, d), axis=-1, keepdims=True))
    u = _scaled_expm1(q, log_q, s)
    return jnp.where(alike, u - p * jnp.sum(u, axis=-1, keepdims=True), p - q)


def _scaled_expm1(q: jax.Array, log_q: jax.Array, x: jax.Array) -> jax.Array:
    """`q expm1(x)`: through `expm1` where `x` is small, whose precision is the point, and as
    `exp(log q + x) - q` where it is large, so a class whose `q` underflowed meets no `0 * inf`. Each
    branch reads `x` only where it is taken, so neither's derivative meets the other's overflow."""
    small = x < 1
    return jnp.where(small, q * jnp.expm1(jnp.where(small, x, 0.0)),
                     jnp.exp(log_q + jnp.where(small, 0.0, x)) - q)


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
