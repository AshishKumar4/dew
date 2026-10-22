"""The attention oracle's float64 truth against an exact evaluation (mpmath at 60 digits): its softmax's
derivative about each row's largest weight (`attention_oracle.anchored_softmax`) is never further from
exact than the plain derivative, on the extreme case where the plain one cancels (logits near 1e4) and
on ordinary draws."""

import jax
import jax.numpy as jnp
import mpmath
import numpy as np
import pytest
from attention_oracle import anchored_softmax

mpmath.mp.dps = 60


def exact_gradient(logits: np.ndarray, cotangent: np.ndarray) -> np.ndarray:
    """The softmax's VJP of one row in 60-digit arithmetic: `p (g - sum p g)`, rounded to float64."""
    terms = [mpmath.mpf(float(x)) for x in logits]
    top = max(terms)
    weights = [mpmath.exp(x - top) for x in terms]
    total = mpmath.fsum(weights)
    p = [w / total for w in weights]
    g = [mpmath.mpf(float(x)) for x in cotangent]
    mean = mpmath.fsum(pi * gi for pi, gi in zip(p, g, strict=True))
    return np.asarray([float(pi * (gi - mean)) for pi, gi in zip(p, g, strict=True)])


def rows(case: str, count: int):
    """`count` rows of 16 logits and cotangents, float64: logits near 1e4 (weights spanning 1 to 1e-22
    and below), or standard-normal ones."""
    streams = jax.random.split(jax.random.key({"extreme": 0, "ordinary": 1}[case]), 2)
    with jax.enable_x64(new_val=True):
        logits = np.asarray(jax.random.normal(streams[0], (count, 16), jnp.float64))
        cotangents = np.asarray(jax.random.normal(streams[1], (count, 16), jnp.float64))
    if case == "extreme":
        logits = logits * 1e4
        # Each row's runner-up 50 below its largest, a weight near 1e-22 beside one near 1.
        logits[:, 1] = logits.max(axis=1) - 50.0 - np.arange(count) / count
    return logits, cotangents


def anchored_bound(logits: np.ndarray, cotangent: np.ndarray) -> np.ndarray:
    """The anchored form's float64 rounding bound, entrywise: with `r = g - g_a` its terms are each weight
    times `|r_i|` and the weights' sum of `|r_j|`, each rounded through at most `n + 4` operations
    (the weights' exponential and normalization, the subtraction, the sum, the products), so
    `|error_i| <= 2 gamma_{n+4} p_i (|r_i| + sum_j p_j |r_j|)` (Higham, Accuracy and Stability of
    Numerical Algorithms, 3.1 and 4.2)."""
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    relative = np.abs(cotangent - cotangent[np.argmax(logits)])
    unit = np.finfo(np.float64).eps / 2
    steps = logits.size + 4
    gamma = steps * unit / (1 - steps * unit)
    return 2 * gamma * weights * (relative + np.sum(weights * relative))


@pytest.mark.parametrize("case", ["extreme", "ordinary"])
def test_the_anchored_truth_is_within_its_bound_and_never_further_than_the_plain_one(case):
    """Each anchored entry is within its float64 rounding bound of exact (`anchored_bound`), and never
    further from exact than the plain form's entry except within that bound; on the extreme rows the
    plain form loses the anchor's gradient, the runner-up's share, past it."""
    logits, cotangents = rows(case, 32)
    with jax.enable_x64(new_val=True):
        def gradient(softmax, logit, cotangent):
            return np.asarray(jax.vjp(softmax, jnp.asarray(logit))[1](jnp.asarray(cotangent))[0])

        plain = np.asarray([gradient(jax.nn.softmax, x, g) for x, g in zip(logits, cotangents, strict=True)])
        anchored = np.asarray([gradient(anchored_softmax, x, g)
                               for x, g in zip(logits, cotangents, strict=True)])
    exact = np.asarray([exact_gradient(x, g) for x, g in zip(logits, cotangents, strict=True)])
    bound = np.asarray([anchored_bound(x, g) for x, g in zip(logits, cotangents, strict=True)])
    anchored_errors, plain_errors = np.abs(anchored - exact), np.abs(plain - exact)
    assert (anchored_errors <= bound).all(), float((anchored_errors / np.maximum(bound, 1e-300)).max())
    assert (anchored_errors <= np.maximum(plain_errors, bound)).all()
    if case == "extreme":
        # Where an anchor's exact gradient is past the bound, a share float64 resolves, the plain form
        # cancels it away and the anchored keeps it.
        anchors = (np.arange(32), np.argmax(logits, axis=1))
        resolved = np.abs(exact[anchors]) > 2 * bound[anchors]
        assert resolved.any()
        assert (plain_errors[anchors][resolved] > bound[anchors][resolved]).all()
