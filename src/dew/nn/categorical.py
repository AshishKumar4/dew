"""The categorical distribution over a class axis: the one owner of its log-likelihood in Dew.

A model's logits over a vocabulary, a classifier's over its classes, a decision's over a question's
options and a sampler's over its next token are each a categorical distribution over the last axis.
Every log-likelihood Dew takes of one composes on this module: the normalized log probabilities
(`log_probs`), a label's cross entropy (`cross_entropy`), a soft target's (`soft_cross_entropy`), a
divergence from a reference distribution (`kl_divergence`), the entropy (`entropy`), the divergence
among a group's members over the classes they share (`group_divergence`), and, for a head
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
from jax.custom_derivatives import SymbolicZero

from dew.nn.precision import at_least_fp32

__all__ = ["cross_entropy", "entropy", "group_divergence", "kl_divergence", "label_derivative", "label_entry",
           "log_probs", "probabilities", "rescaled", "segment_log_partition", "soft_cross_entropy",
           "target_loss"]


def widened(logits: jax.Array, where: jax.Array | None = None) -> jax.Array:
    """`logits` at the precision a categorical is computed in, at least float32, with the classes
    `where` excludes at -inf."""
    logits = logits.astype(at_least_fp32(logits.dtype))
    return logits if where is None else jnp.where(where, logits, -jnp.inf)


def log_probs(logits: jax.Array, where: jax.Array | None = None) -> jax.Array:
    """The normalized log probabilities over the last axis (`jax.nn.log_softmax`)."""
    return jax.nn.log_softmax(widened(logits, where), axis=-1)


def probabilities(logits: jax.Array, where: jax.Array | None = None) -> jax.Array:
    """The probabilities over the last axis (`jax.nn.softmax`). Their derivative is softmax's,
    `p (dz - sum p dz)`, but at the most probable class, which takes the other classes' derivative
    summed, negated, so a row's derivative sums to zero: softmax's own subtracts two near-equal terms
    there once its probability nears one, keeping only their rounding (as a label's `p - 1` does,
    arXiv:2610.31642, eq. 4), while the others' sum keeps their precision, in either mode."""
    return _probabilities(widened(logits, where))


@jax.custom_jvp
def _probabilities(logits: jax.Array) -> jax.Array:
    return jax.nn.softmax(logits, axis=-1)


@_probabilities.defjvp
def _probabilities_tangent(primals, tangents):
    (logits,), (d_logits,) = primals, tangents
    probability = _probabilities(logits)
    tangent = probability * (d_logits - jnp.sum(probability * d_logits, axis=-1, keepdims=True))
    top = jax.nn.one_hot(jnp.argmax(probability, axis=-1), logits.shape[-1], dtype=bool)
    rest = jnp.sum(jnp.where(top, 0.0, tangent), axis=-1, keepdims=True)
    return probability, jnp.where(top, -rest, tangent)


def cross_entropy(logits: jax.Array, labels: jax.Array, where: jax.Array | None = None) -> jax.Array:
    """`-log p[labels]` over the last axis, `labels` of `logits.shape[:-1]`: the log partition less the
    label's logit, each taken from the row's maximum (`target_loss`), which reads the label's logit
    without building the normalized row. Optax's `softmax_cross_entropy_with_integer_labels` subtracts
    the label's logit from the whole log partition, maximum included, which rounds at the logits'
    magnitude rather than the loss's: mamba-tiny's drawn tokens' raw scores sat 1.47 times as far from
    float64 as `log_softmax`'s, and 1.00 times so taken here. Its derivative is Barron, Gnanapragasam and Tschannen's (arXiv:2610.31642, eq. 7): each other
    class's probability, and at the label minus their sum, so the row's derivative sums to zero. The
    label's `p - 1` collapses to 0 once its margin over the other classes passes the dtype's precision
    (eq. 4; 16.6 in float32), while the others' probabilities stay until they underflow."""
    return _label_loss(widened(logits, where), labels)


@jax.custom_jvp
def _label_loss(logits: jax.Array, labels: jax.Array) -> jax.Array:
    # An infinite maximum, a row excluded whole, shifts by 0, as `jax.nn.logsumexp` shifts.
    peak = jnp.max(logits, axis=-1, keepdims=True)
    peak = jnp.where(jnp.isfinite(peak), peak, 0.0)
    label = jnp.take_along_axis(logits, labels[..., None], axis=-1)[..., 0]
    return target_loss(peak[..., 0], jnp.sum(jnp.exp(logits - peak), axis=-1), label)


@_label_loss.defjvp
def _label_loss_tangent(primals, tangents):
    logits, labels = primals
    derivative = label_derivative(jax.nn.softmax(logits, axis=-1), labels)
    return _label_loss(logits, labels), jnp.sum(derivative * tangents[0], axis=-1)


def label_entry(share: jax.Array, others: jax.Array) -> jax.Array:
    """The derivative of `-log p` in the label's own logit, from the label's `share` of its row and
    the `others`' share summed: `share - 1` while the label holds at most half the row, else minus
    the others' sum (arXiv:2610.31642, eq. 7). Past a half `share - 1` cancels, and rounds to 0 once
    the share rounds to 1, where the others' sum keeps it and the row's derivative sums to zero in
    any rounding; below a half `1 - share` is at least a half and `share - 1` rounds once, where the
    others' sum carries each class's share of the partition's rounding, which left an unconfident
    label's entry two to three times as far from float64. Every head's label entry is this one rule
    (`label_derivative`, `dew.objectives.lm.chunked`)."""
    return jnp.where(share > 0.5, -others, share - 1)


def label_derivative(probability: jax.Array, labels: jax.Array) -> jax.Array:
    """The derivative of `-log p[labels]` in its row's logits, from the row's `probability` over the
    last axis: each other class's probability, and at the label `label_entry`'s. A label outside the
    row one-hots to nothing, and its row's derivative is the probabilities."""
    label = jax.nn.one_hot(labels, probability.shape[-1], dtype=bool)
    others = jnp.where(label, 0.0, probability)
    share = jnp.sum(jnp.where(label, probability, 0.0), axis=-1, keepdims=True)
    return jnp.where(label, label_entry(share, jnp.sum(others, axis=-1, keepdims=True)), others)


def soft_cross_entropy(logits: jax.Array, targets: jax.Array, where: jax.Array | None = None) -> jax.Array:
    """`-sum_k t_k log p_k` over the last axis against the distribution `targets`, a class of no target
    mass contributing nothing, whatever its log probability. Its derivative in `logits`, `p T - t` for
    target mass `T`, is `p (T - t) - t (1 - p)`, the most probable class's `1 - p` the other classes'
    probability summed, and the class of the most target mass's `T - t` the other classes' targets
    summed: a label's confident one-hot target keeps it as `cross_entropy` does (arXiv:2610.31642,
    eq. 7), where `p - t` rounds to 0, and targets as confident as the logits keep its sum 0."""
    return _soft_loss(widened(logits, where), targets)


def _soft_value(logits: jax.Array, targets: jax.Array) -> jax.Array:
    return -jnp.sum(jnp.where(targets > 0, targets * jax.nn.log_softmax(logits, axis=-1), 0.0), axis=-1)


@jax.custom_jvp
def _soft_loss(logits: jax.Array, targets: jax.Array) -> jax.Array:
    return _soft_value(logits, targets)


@functools.partial(_soft_loss.defjvp, symbolic_zeros=True)
def _soft_loss_tangent(primals, tangents):
    logits, targets = primals
    d_logits, d_targets = tangents
    value = _soft_value(logits, targets)
    tangent = jnp.zeros_like(value)
    if not isinstance(d_targets, SymbolicZero):
        tangent = jax.jvp(lambda targets: _soft_value(logits, targets), (targets,), (d_targets,))[1]
    if isinstance(d_logits, SymbolicZero):
        return value, tangent
    probability = jax.nn.softmax(logits, axis=-1)
    top = jax.nn.one_hot(jnp.argmax(probability, axis=-1), logits.shape[-1], dtype=bool)
    rest = jnp.where(top, jnp.sum(jnp.where(top, 0.0, probability), axis=-1, keepdims=True), 1 - probability)
    mass = jnp.where(targets > 0, targets, 0).astype(probability.dtype)
    # `T - t`, the other classes' target mass: summed directly for the class that holds most of it, whose
    # difference would round, as `rest` is for the most probable class.
    held = jax.nn.one_hot(jnp.argmax(mass, axis=-1), logits.shape[-1], dtype=bool)
    beside = jnp.where(held, jnp.sum(jnp.where(held, 0.0, mass), axis=-1, keepdims=True),
                       mass.sum(axis=-1, keepdims=True) - mass)
    derivative = probability * beside - mass * rest
    return value, tangent + jnp.sum(derivative * d_logits, axis=-1)


def kl_divergence(logits: jax.Array, reference: jax.Array, where: jax.Array | None = None) -> jax.Array:
    """`KL(q || p)` over the last axis, `p` the categorical of `logits` and `q` that of the `reference`
    logits (Optax's `kl_divergence` of the two), the classes `where` excludes adding no term.

    Its derivative in `logits`, `p - q`, is taken from the two rows' difference in log space, as Barron,
    Gnanapragasam and Tschannen take it (arXiv:2610.31642, eq. 9), not by subtracting two softmaxes
    (eq. 8), which near each other keep only their own rounding: logits equal to the reference's have
    derivative exactly zero, and near them it keeps the difference's precision. A row where a class is
    -inf in one and not the other takes `p - q`, a row holding no class 0; the derivative in
    `reference` is autodiff's."""
    return _divergence(widened(logits, where), widened(reference, where), where)


def _divergence_value(logits: jax.Array, reference: jax.Array, where: jax.Array | None) -> jax.Array:
    log_p, q = jax.nn.log_softmax(logits, axis=-1), jax.nn.softmax(reference, axis=-1)
    if where is None:
        return optax.kl_divergence(log_p, q)
    # An excluded class enters Optax's sum as log 1 against a log probability of 0, so neither its value
    # nor its gradient meets 0 * inf.
    return optax.kl_divergence(jnp.where(where, log_p, 0.0), jnp.where(where, q, 1.0), where=where)


@jax.custom_jvp
def _divergence(logits: jax.Array, reference: jax.Array, where: jax.Array | None) -> jax.Array:
    return _divergence_value(logits, reference, where)


@functools.partial(_divergence.defjvp, symbolic_zeros=True)
def _divergence_tangent(primals, tangents):
    logits, reference, where = primals
    d_logits, d_reference, _ = tangents
    value = _divergence_value(logits, reference, where)
    tangent = jnp.zeros_like(value)
    if not isinstance(d_reference, SymbolicZero):
        tangent = jax.jvp(lambda reference: _divergence_value(logits, reference, where), (reference,),
                          (d_reference,))[1]
    if isinstance(d_logits, SymbolicZero):
        return value, tangent
    return value, tangent + jnp.sum(_divergence_derivative(logits, reference) * d_logits, axis=-1)


def _divergence_derivative(logits: jax.Array, reference: jax.Array) -> jax.Array:
    """`p - q` over the last axis, `p` the softmax of `logits`, `q` of `reference`, from their difference
    (arXiv:2610.31642, eq. 9, its `p` here `q` and its `q` here `p`): `delta`, the rows' difference,
    taken of the rows each moved to its own largest held logit where it overflows, which neither softmax
    reads, less its value at `q`'s most probable class; `d`, `delta` less its mean under `q`;
    `s = log(p / q)`, `d` less the log of `q`'s mean of `exp(d)`, through `log1p` of the mean of
    `expm1(d)` where every `d` is within 1 of 0 and as a log-sum-exp where one is not; `u = q expm1(s)`;
    and `u` less `p` times its sum, which makes the derivative sum to zero. A row whose `delta` or `d`
    overflows, its rows at opposite extrema, or whose `s` rounds a class's `log q + s`, its `log p`,
    above 0, holds distributions too far apart for a subtraction of their softmaxes to cancel, and takes
    `p - q`; the centring reads such a `delta` as 0 and holds such an exponent at 0, so no derivative of
    either meets the overflow."""
    empty = jnp.all(jnp.isneginf(logits) & jnp.isneginf(reference), axis=-1, keepdims=True)
    logits, reference = jnp.where(empty, 0.0, logits), jnp.where(empty, 0.0, reference)
    both = jnp.isfinite(logits) & jnp.isfinite(reference)
    alike = jnp.all(both | (jnp.isneginf(logits) & jnp.isneginf(reference)), axis=-1, keepdims=True)

    def moved(row: jax.Array) -> jax.Array:
        top = jnp.max(jnp.where(both, row, -jnp.inf), axis=-1, keepdims=True)
        return jnp.where(both, row - jnp.where(jnp.isfinite(top), top, 0.0), 0.0)

    log_q = jax.nn.log_softmax(reference, axis=-1)
    q, p = jnp.exp(log_q), jax.nn.softmax(logits, axis=-1)
    # Subtracted directly, the difference of near rows is exact; a row whose difference overflows takes
    # it of the rows moved to their maxima, which shifts it by a constant `d` removes.
    direct = jnp.where(both, logits - reference, 0.0)
    overflow = jnp.any(~jnp.isfinite(direct), axis=-1, keepdims=True)
    delta = jnp.where(overflow, moved(logits) - moved(reference), jnp.where(overflow, 0.0, direct))
    # Taken from its value at q's most probable class, the difference's mean under q is a mean of the
    # difference's own differences, which a uniform offset leaves exactly 0.
    delta = delta - jnp.take_along_axis(delta, jnp.argmax(log_q, axis=-1, keepdims=True), axis=-1)
    finite = jnp.all(jnp.isfinite(delta), axis=-1, keepdims=True)
    delta = jnp.where(finite, delta, 0.0)
    mean = jnp.sum(q * delta, axis=-1, keepdims=True)
    # Halved, the difference of two finite values is finite, and it overflows whole where its half
    # reaches half the largest value.
    centred = finite & jnp.all(jnp.abs(delta / 2 - mean / 2) < jnp.finfo(delta.dtype).max / 2, axis=-1,
                               keepdims=True)
    d = jnp.where(centred, delta - mean, 0.0)
    near = jnp.max(jnp.abs(d), axis=-1, keepdims=True) < 1
    small, large = jnp.where(near, d, 0.0), jnp.where(near, 0.0, d)
    normalizer = jnp.where(near, jnp.log1p(jnp.sum(q * jnp.expm1(small), axis=-1, keepdims=True)),
                           jax.nn.logsumexp(log_q + large, axis=-1, keepdims=True))
    s = d - normalizer
    # `log q + s` is `log p`, at most 0; a class past it is one whose difference rounded by more than its
    # whole log probability, which only rows far apart reach, and its row takes `p - q`.
    rounded = jnp.any((s >= 1) & (log_q + s > 0), axis=-1, keepdims=True)
    u = _scaled_expm1(q, log_q, s)
    return jnp.where(alike & centred & ~rounded, u - p * jnp.sum(u, axis=-1, keepdims=True), p - q)


def _scaled_expm1(q: jax.Array, log_q: jax.Array, x: jax.Array) -> jax.Array:
    """`q expm1(x)`: through `expm1` where `x` is small, whose precision is the point, and as
    `exp(log q + x) - q` where it is large, so a class whose `q` underflowed meets no `0 * inf`, its
    exponent, the log of a probability, held at most 0, which rounding may overshoot. Each branch reads
    `x` only where it is taken, so neither's derivative meets the other's overflow."""
    small = x < 1
    return jnp.where(small, q * jnp.expm1(jnp.where(small, x, 0.0)),
                     jnp.exp(jnp.minimum(log_q + jnp.where(small, 0.0, x), 0.0)) - q)


def entropy(logits: jax.Array, where: jax.Array | None = None) -> jax.Array:
    """`-sum_k p_k log p_k` over the last axis, a class of no probability (a -inf logit, or one `where`
    excludes) adding no term, in value or gradient."""
    log_p = log_probs(logits, where)
    held = jnp.where(jnp.isneginf(log_p), 0.0, log_p)
    return -(jnp.exp(log_p) * held).sum(axis=-1)


def group_divergence(logits: jax.Array, groups: jax.Array, classes: jax.Array,
                     where: jax.Array | None = None) -> tuple[jax.Array, jax.Array]:
    """Each member's share of its group's generalized Jensen-Shannon divergence, and its share of the
    group, `[..., M]` arrays whose members sum to each group's divergence and to one a group.

    `logits` `[..., M, K]` are M members' categoricals, each over its own K classes (`where` masks those
    it does not hold); `groups` `[..., M]` names each member's group among the M (-1: none); and
    `classes` `[..., M, K]` each class's shared class, in `[0, K)` (-1: none), as questions that ask
    one thing with their options re-ordered or reworded share their options. A member's answer is its
    probabilities (`probabilities`) over its classes that map to a shared one, summed onto them; a
    group's divergence is the mean of each member's KL divergence from the members' mean answer, zero
    exactly where they agree, at most the log of the member count, computed in at least float32 and
    finite at any finite logits. A shared class of no probability adds no term, in value or gradient;
    the gradient is autodiff's, through the probabilities' own derivative, which a confident member
    keeps. A member in no group, or with no class that maps, has share zero and takes no part."""
    width, count = logits.shape[-1], logits.shape[-2]
    held = jnp.ones(logits.shape, bool) if where is None else jnp.broadcast_to(where, logits.shape)
    mapped = held & (classes >= 0)
    # A member of no mapped class normalizes over nothing: it reads zeros instead, which its answer then
    # takes no part of, so neither its value nor its gradient is NaN.
    any_mapped = jnp.any(mapped, axis=-1, keepdims=True)
    member_probabilities = probabilities(jnp.where(any_mapped, logits, 0.0), mapped | ~any_mapped)
    slots = jnp.where(mapped, classes, width)

    def summed(row_probabilities, row_slots):
        return jax.ops.segment_sum(row_probabilities, row_slots, num_segments=width + 1)[:width]

    rows = (-1, count, width)
    answer = jax.vmap(jax.vmap(summed))(jnp.where(mapped, member_probabilities, 0.0).reshape(rows),
                                        slots.reshape(rows))
    # A member stands in its group where it holds a class that maps: one of none has no answer to share.
    member = ((groups >= 0) & any_mapped[..., 0]).reshape(-1, count)
    ids = jnp.where(member, groups.reshape(-1, count), count)

    def window(answer, ids, member):
        sizes = jax.ops.segment_sum(member.astype(answer.dtype), ids, num_segments=count + 1)[:count]
        total = jax.ops.segment_sum(jnp.where(member[:, None], answer, 0.0), ids, num_segments=count + 1)
        size = sizes[jnp.minimum(ids, count - 1)]
        mean = total[jnp.minimum(ids, count - 1)] / jnp.maximum(size, 1.0)[:, None]
        # A shared class of no probability adds no term, nor does a non-member's, whose mean is no
        # group's: their logarithms read one, not zero, so no derivative meets 0 * inf.
        held = (answer > 0) & member[:, None]
        ratio = jnp.log(jnp.where(held, answer, 1.0)) - jnp.log(jnp.where(held, mean, 1.0))
        each = jnp.sum(jnp.where(held, answer * ratio, 0.0), axis=-1)
        return (jnp.where(member, each / jnp.maximum(size, 1.0), 0.0),
                jnp.where(member, 1.0 / jnp.maximum(size, 1.0), 0.0))

    divergence, shares = jax.vmap(window)(answer, ids, member)
    return divergence.reshape(groups.shape), shares.reshape(groups.shape)


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
