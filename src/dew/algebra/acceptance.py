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


def _softmax(s: np.ndarray, shape) -> tuple:
    """Logits `s` with their rows' maximum and total (of `exp(s - m)`), laid at the logits' shape."""
    m = np.broadcast_to(s.max(-1, keepdims=True), shape)
    t = np.broadcast_to(np.exp(s.astype(np.float64) - m).sum(-1, keepdims=True), shape)
    return tuple(jnp.asarray(value, jnp.float32) for value in (s, m, t))


def _folded(seed: int, operands) -> tuple:
    """Logits near a thousand, a row's spread of eight, with their rows' maximum and total: a softmax's."""
    rng = np.random.default_rng(seed)
    shape = operands[0].shape
    return _softmax((1000.0 - rng.uniform(0.0, 8.0, shape)).astype(np.float32), shape)


def _folded_spread(spread: float, centre: float = 1000.0):
    """`_folded`'s logits at `centre`, a row's spread `spread`: eighty, a row's exponentials across 35
    decades, its least near float32's least normal (wide ranges); a thousandth, a row's logits a few ulps
    apart (near-equal operands, `s - m` cancelling); or near zero, the total's logarithm as large as a
    logit."""
    def drawn(seed: int, operands) -> tuple:
        rng = np.random.default_rng(seed)
        shape = operands[0].shape
        return _softmax((centre - rng.uniform(0.0, spread, shape)).astype(np.float32), shape)
    return drawn


def _folded_one_key(seed: int, operands) -> tuple:
    """`_folded`'s logits with each row's mass on one key, the rest two hundred below it (their exponentials
    zero): a row of one key."""
    rng = np.random.default_rng(seed)
    shape = operands[0].shape
    s = np.full(shape, 800.0, np.float32)
    s[np.arange(shape[0]), rng.integers(0, shape[1], shape[0])] = 1000.0
    return _softmax(s, shape)


def normalized_values_product(exponentials, totals, cotangent):
    """dV of a causal attention backward, queries first: the unnormalized probabilities' transpose
    (`exponentials`, keys by queries), normalized by each query's total along the contraction, times the
    output's cotangent (queries by width)."""
    return jnp.dot(exponentials / totals[None, :], cotangent)


def _causal_drawn(spread: float = 8.0, rows: tuple[float, float] = (0.0, 0.0), paired: bool = False,
                  one_key: bool = False):
    """A causal attention's exponentials `exp(s - m)` of logits of a row's spread `spread` (each query sees
    the keys up to its own, the rest zero; `one_key`: each query's mass on one key it sees), transposed to
    keys by queries, each query's total (at least one, its maximum's), and a cotangent of unit scale, each
    query's row scaled by `10^u` for `u` uniform in `rows` and, `paired`, each odd query's the negation of
    the one before (a contraction over queries cancelling)."""
    def drawn(seed: int, operands) -> tuple:
        rng = np.random.default_rng(seed)
        keys, queries = operands[0].shape
        logits = rng.standard_normal((queries, keys)) * spread
        seen = np.arange(keys)[None, :] <= np.arange(queries)[:, None]
        if one_key:
            chosen = rng.integers(0, np.arange(queries) + 1)
            logits = np.where(np.arange(keys)[None, :] == chosen[:, None], 0.0, -200.0)
        logits = np.where(seen, logits, -np.inf)
        exponentials = np.exp(logits - logits.max(-1, keepdims=True))
        cotangent = rng.standard_normal(operands[2].shape) * 10.0 ** rng.uniform(*rows, (queries, 1))
        if paired:
            cotangent[1::2] = -cotangent[0::2][:len(cotangent[1::2])]
        return tuple(jnp.asarray(value, jnp.float32)
                     for value in (exponentials.T, exponentials.sum(-1), cotangent))
    return drawn


_causal = _causal_drawn()
"""A causal attention's exponentials of logits of a spread of eight, their totals, and a unit cotangent."""


def _struct(*shape: int) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(shape, jnp.float32)


declare(__name__, "lse_folded_probabilities", StableForm(lse_folded_probabilities, Cases(
    ((_struct(ROWS, KEYS),) * 3,), _folded, seeds=16, families={
        "wide ranges": _folded_spread(80.0), "near-equal operands": _folded_spread(1e-3),
        "logits near zero": _folded_spread(8.0, centre=0.0), "one-key rows": _folded_one_key})))
declare(__name__, "normalized_values_product", StableForm(normalized_values_product, Cases(
    ((_struct(KEYS, KEYS), _struct(KEYS), _struct(KEYS, WIDTH)),), _causal, seeds=16, families={
        "wide ranges": _causal_drawn(rows=(-3.0, 3.0)), "near-equal operands": _causal_drawn(spread=1e-3),
        "cancellation": _causal_drawn(spread=1e-3, paired=True),
        "one-key rows": _causal_drawn(one_key=True)})))
