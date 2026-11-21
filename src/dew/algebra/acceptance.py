"""The generator's first acceptance cases (docs/design/algebra.md §1.5): two forms of a derived attention
backward, each equal to its alternative in the reals, whose choice is decided by their error alone
(`dew.algebra.generate`). Each declared here and generated beside this module (`acceptance_derived.py`).

- The probabilities through a saved log-sum-exp, `exp(s - (m + log t))`: `log t` rounded into a value as
  large as the logits (1.6 to 1.7 times the reference's error on mapped dK and dQ), where `exp(s - m) / t`
  is on par, bounded where the total is at least one, as a softmax's is.
- dV's normalization along its contraction, `(e / t)^T dO`: before the contraction or moved onto the
  cotangent, `e^T (dO / t)` (causal queries-first dV at 1.64 times the oracle's error the one way, on par
  the other)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from dew.algebra.generate import Cases, StableForm, declare

ROWS, KEYS, WIDTH = 64, 256, 64
"""A block of queries, its keys and the head's width."""


def lse_folded_probabilities(s, m, t):
    """A softmax's probabilities through its saved log-sum-exp: logits `s`, each row's maximum `m` and
    total `t` (of `exp(s - m)`), laid at the logits' shape."""
    return jnp.exp(s - (m + jnp.log(t)))


def _folded(seed: int, operands) -> tuple:
    """Logits near a thousand, a row's spread of eight, with their rows' maximum and total: a softmax's."""
    rng = np.random.default_rng(seed)
    shape = operands[0].shape
    s = (1000.0 - rng.uniform(0.0, 8.0, shape)).astype(np.float32)
    m = np.broadcast_to(s.max(-1, keepdims=True), shape)
    t = np.broadcast_to(np.exp(s - m).sum(-1, keepdims=True), shape)
    return tuple(jnp.asarray(value, jnp.float32) for value in (s, m, t))


def normalized_values_product(exponentials, totals, cotangent):
    """dV of a causal attention backward, queries first: the unnormalized probabilities' transpose
    (`exponentials`, keys by queries), normalized by each query's total along the contraction, times the
    output's cotangent (queries by width)."""
    return jnp.dot(exponentials / totals[None, :], cotangent)


def _causal(seed: int, operands) -> tuple:
    """A causal attention's exponentials `exp(s - m)` of logits of a spread of eight (each query sees the
    keys up to its own, the rest zero), transposed to keys by queries, each query's total (at least one, its
    maximum's), and a cotangent of unit scale."""
    rng = np.random.default_rng(seed)
    keys, queries = operands[0].shape
    logits = rng.standard_normal((queries, keys)) * 8.0
    seen = np.arange(keys)[None, :] <= np.arange(queries)[:, None]
    logits = np.where(seen, logits, -np.inf)
    exponentials = np.exp(logits - logits.max(-1, keepdims=True))
    cotangent = rng.standard_normal(operands[2].shape)
    return tuple(jnp.asarray(value, jnp.float32)
                 for value in (exponentials.T, exponentials.sum(-1), cotangent))


def _struct(*shape: int) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(shape, jnp.float32)


declare(__name__, "lse_folded_probabilities", StableForm(lse_folded_probabilities, Cases(
    ((_struct(ROWS, KEYS),) * 3,), _folded, seeds=16)))
declare(__name__, "normalized_values_product", StableForm(normalized_values_product, Cases(
    ((_struct(KEYS, KEYS), _struct(KEYS), _struct(KEYS, WIDTH)),), _causal, seeds=16)))
