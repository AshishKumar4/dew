"""A coupled pair's chunk forms (docs/design/algebra.md §6.9, `scan.Scan` of a pair state): two states
`(h, m)`, keys by values each, updated token by token under the map `linear_scan.coupled_parts` reads from the
declaration's own update, its second state's input reading the first as the token enters it,

    m_t = mu_t m_{t-1} + B_t (U_t + C_t^T h_{t-1})^T,    h_t = alpha_t h_{t-1} + lam_t m_t,
    o_t = Q_t^T h_t + R_t,

`alpha = exp(first)`, `mu = exp(second)` and `lam` scalars: a 2x2 triangular transition on the pair beside the
input's rank one (Momentum DeltaNet's, fla's momentum_delta_rule: `lam = -beta`, `C` its erase key). Over a
chunk from the pair `(h_0, m_0)` entering it, the inputs `w_t = U_t + C_t^T h_{t-1}` solve the unit
lower-triangular system

    (I - L) W = U + diag(A_{t-1}) C h_0 + diag(b_{t-1}) C m_0,    L_ts = g(s, t - 1) C_t . B_s  (s < t),
    o = diag(A) Q h_0 + diag(b) Q m_0 + (g o Q B^T) W + R,
    h_C = A_C h_0 + b_C m_0 + (B o g(., C))^T W,    m_C = m(0, C) m_0 + (B o m(., C))^T W,

with `A_t` the alphas' product to `t`, `m(s, t)` the mus' over `(s, t]`, `g(s, t) = sum_{s <= r <= t}
a(r, t) lam_r m(s, r)` and `b_t` its sum from the entering momentum (`_decayed_sums`). A 2x2 product has no
logarithm, so its sums are running products a token at a time (`_decayed_sums`), not differences of a
compensated prefix as the single state's decays are; each decay meets its query or key before their product
with a state (a query read of the state before its decay's weight overflowed where the token walk's read is
finite: ReviewMomentum1454). The chunk's adjoint is its token walk's (`replay`), differentiated, as the single
state's is. A packed document's first token (`Scan.reset`) enters a zero pair: its decays and its correction
zero, the walk's pair selected to zero.

The declaration names ranges for the operands its decays and `lam` read (`Scan.ranges`), which bound the
chunk's sums (`_bounded`); a pair whose decays or `lam` no declared range bounds, whose sums pass float32's
range at its chunk, whose decays are a key's own, or whose readout reads otherwise than after its token is
refused, named, as every other pair's update is (`linear_scan.coupled_parts`). Its walks do not merge, and a
document's window or a walk's own hooks it takes none of."""

from __future__ import annotations

import collections
import functools
import math
from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.linalg import solve_triangular

from dew import _cache
from dew.algebra import derive, linear_scan, program, terms
from dew.algebra.forms import Derived, Refused
from dew.algebra.scan import (
    _carry_type,
    _exp,
    _replay_backward,
    _traced,
    _walked_tokens,
    token_terms,
    token_type,
)
from dew.nn.precision import at_least_fp32
from dew.nn.sharding import varying_like


def derive_coupled(declaration, *, saturation: program.Saturation) -> Derived:
    """A pair's chunk forms (the module's account), `scan.derive_scan`'s contract: `init(shapes)` a zero
    decay and pair (`[2, keys, values]`, the first state first), `prepare(chunk)` a chunk's values that read
    no state, `step(state, prepared)` the pair after the chunk, `mapped(prepared, state)` its outputs from
    the pair entering it, `finish(state, operands)` the final pair, `replay(chunk, h)` the chunk walked token
    by token and `backward(chunk, state, d_state, d_outputs)` its derivative; `merge`, a window and a walk's
    hooks refused."""
    names = tuple(name for name in declaration.operands if name != declaration.reset)
    (shape, dtype), second = declaration.state
    if second != (shape, dtype) or len(shape) != 2:
        raise Refused("a pair's states are of one shape (keys by values) and one dtype")
    if declaration.reads != "after" or declaration.group != 1:
        raise Refused("a pair's readout reads the first state after its token's update alone")
    carried = at_least_fp32(jnp.dtype(dtype))
    decomposed: collections.OrderedDict = collections.OrderedDict()

    def affine_of(chunk: Mapping) -> linear_scan.Coupled:
        key = (tuple((tuple(chunk[name].shape), str(chunk[name].dtype)) for name in names), derive.context())

        def made():
            parts = linear_scan.coupled_parts(_traced(declaration, declaration.update, names, chunk),
                                              _traced(declaration, declaration.readout, names, chunk),
                                              saturation)
            _bounded(declaration, names, chunk, parts)
            return parts

        return _cache.holding(decomposed, key, made, derive.FORMS_HELD)

    def tokens(chunk: Mapping) -> tuple:
        along, every = _walked_tokens(declaration, chunk)
        parts = affine_of(chunk)
        return jax.vmap(lambda token: _at_token(declaration, names, parts, {**token, **every}))(along)

    def init(shapes):
        del shapes
        return jnp.zeros((shape[0],), carried), jnp.zeros((2, *shape), carried)

    def prepare(chunk, issue=None, transition: bool = False, window=None):
        if issue is not None or transition or window is not None:
            raise Refused("a pair's chunk takes no mainloop's issuer, transfer or document window here")
        return _prepared(*tokens(chunk))

    def merge(a, b, operands, issue=None):
        del a, b, operands, issue
        raise Refused("a pair's walks are not merged here")

    def finish(state, operands):
        del operands
        return state[1].astype(dtype)

    def replay(chunk, h, walked=None):
        if walked is not None:
            raise Refused("a pair's walk takes no hooks of its own here (a forked call's documents)")
        return _replayed(declaration, names, affine_of(chunk), chunk, h)

    return Derived(init, _stepped, merge, finish, _mapped,
                   {"rounding": "running products", "walked": declaration.walked},
                   functools.partial(_replay_backward, replay, names), prepare, replay=replay)


def _token(parts: linear_scan.Coupled, token: tuple) -> tuple:
    """One token's parts at its values (`linear_scan.Coupled`'s), each in float32 at least: the two
    exponents and `lam`, scalars, the input `B` and `U`, the correction `C`, the reader `Q` (its scale on it)
    and the free readout `R`; refused where an exponent or `lam` is a key's own."""
    shape, along = token_terms(parts, token, states=2)

    def scalar(term):
        value = along(term, None)
        if np.size(value) != 1:
            raise Refused("a pair's decays and its second state's weight are scalars a token, not a key's "
                          "own")
        # A scalar as it is: reshaped, its declared interval no longer bounds its exponential (`_bounded`).
        return value if jnp.ndim(value) == 0 else jnp.reshape(value, ())

    keys, values = shape
    reader = parts.reader_scale * along(parts.reader, keys)
    return (scalar(parts.first), scalar(parts.second), scalar(parts.lam), along(parts.left, keys),
            along(parts.right, values), along(parts.correction, keys), reader, along(parts.free, values))


def _at_token(declaration, names: Sequence[str], parts: linear_scan.Coupled, token: Mapping) -> tuple:
    """One token's decays `alpha` and `mu`, `lam`, `B`, `U`, `C`, `Q` and `R` (`_token`), a reset token's
    decays and correction zero: it reads nothing of the pair entering it."""
    first, second, lam, b, u, c, q, r = _token(parts, tuple(token[name] for name in names))
    alpha, mu = _exp(parts.base2, first), _exp(parts.base2, second)
    if declaration.reset is not None:
        reset = token[declaration.reset]
        alpha, mu, c = jnp.where(reset, 0.0, alpha), jnp.where(reset, 0.0, mu), jnp.where(reset, 0.0, c)
    return alpha, mu, lam, b, u, c, q, r


def _bounded(declaration, names: Sequence[str], chunk: Mapping, parts: linear_scan.Coupled) -> None:
    """Refused unless the declared ranges (`Scan.ranges`, `terms.region`'s magnitudes) bound the decays and
    `lam`, and the chunk's sums they bound stay within float32's range: each `g(s, t)` at most the chunk's
    tokens times `lam`'s bound times the larger decay's to the chunk's length."""
    tokens = [token_type(declaration, chunk, name) for name in names]

    def bounded(*xs):
        # The language's exponentials, whose bound an exponent's declared interval states (`terms.region`).
        first, second, lam = _token(parts, xs)[:3]
        power = jnp.exp2 if parts.base2 else jnp.exp
        return power(first), power(second), lam

    closed = jax.make_jaxpr(bounded)(*tokens)
    # An exponent's interval reaching -inf held at float64's least: the region bounds an exponential of an
    # interval at most 0 by 2 where its input is proven finite, and an exponential of -inf is 0, finite too.
    declared = {index: _finite(declaration.ranges[name]) for index, name in enumerate(names)
                if name in declaration.ranges}
    region = terms.region(closed, declared)
    magnitudes = []
    for output, literal in zip(region.outputs, region.literal_outputs, strict=True):
        if output is None:
            # A literal bounds itself (a unit decay's, a constant weight's).
            largest = float(np.max(np.abs(np.asarray(literal, np.float64))))
            magnitudes.append(math.ceil(math.log2(largest)) if largest > 0 else -126)
        else:
            magnitudes.append(region.magnitudes.get(terms.key(output)))
    if any(magnitude is None for magnitude in magnitudes):
        raise Refused("a pair's decays and its second state's weight need declared ranges (`Scan.ranges`) "
                      "that bound them")
    length = next(iter(chunk[name].shape[declaration.operands[name].index(declaration.walked)]
                       for name in names if declaration.walked in declaration.operands[name]))
    decay, lam = max(magnitudes[:2]), magnitudes[2]
    if math.log2(length) + lam + length * max(decay, 0) >= 127:
        raise Refused(f"a pair's chunk of {length} tokens sums past float32's range under its declared "
                      "ranges")


def _finite(declared: float | tuple[float, float]) -> float | tuple[float, float]:
    """A declared range (`Scan.ranges`) with an interval's infinite ends held at float64's largest."""
    if not isinstance(declared, tuple):
        return declared
    low, high = declared
    largest = float(np.finfo(np.float64).max)
    return max(low, -largest), min(high, largest)


def _decayed_sums(alpha, mu, lam):
    """A chunk's sums from its `alpha`, `mu` and `lam` `[C]`, a token at a time, each a running product:
    `g` `[C, C + 1]`, row t `g(s, t)` at column `s + 1` for token `s` (none past `t`) and `b_t` at column
    zero; the last token's `m(., C)` `[C + 1]` (`m(0, C)` at column zero), and `A` `[C]`. `m(s, t) = mu_t
    m(s, t - 1)`, one at `s = t`; `g(s, t) = alpha_t g(s, t - 1) + lam_t m(s, t)`."""
    tokens = alpha.shape[-1]
    own = jnp.eye(tokens + 1, dtype=alpha.dtype)[1:]

    def token(carried, step):
        g, m, kept = carried
        a, u, scale, mine = step
        m = m * u + mine
        g = a * g + scale * m
        kept = kept * a
        return (g, m, kept), (g, kept)

    # The carries typed as the tokens vary (a checked `shard_map`'s manual axes), as a step's outputs vary.
    entering = varying_like(jnp.zeros((tokens + 1,), alpha.dtype), alpha, mu, lam)
    carried = (entering, entering.at[0].set(1.0), varying_like(jnp.ones((), alpha.dtype), alpha, mu, lam))
    (_, last, _), (g, kept) = lax.scan(token, carried, (alpha, mu, lam, own))
    return g, last, kept


def _prepared(alpha, mu, lam, b, u, c, q, r) -> tuple:
    """A chunk's values that read no state (the module's account), each term the entering pair reaches
    divided by the chunk's `scale`, a power of two at least its largest decayed sum and one (`_scale`): the
    solve's three solutions (against the inputs, the entering first state's and its momentum's), the
    scores, the queries decayed to each token from the entering first state and momentum, the free readouts,
    the keys weighed to the chunk's end, the entering pair's three factors there, and the scale."""
    g, last, kept = _decayed_sums(alpha, mu, lam)
    before = jnp.concatenate([jnp.zeros_like(g[:1]), g[:-1]])
    kept_before = jnp.concatenate([jnp.ones_like(kept[:1]), kept[:-1]])
    system = jnp.eye(g.shape[0], dtype=g.dtype) - before[:, 1:] * (c @ b.T)
    scale = _scale(kept, g[:, 0])
    sides = jnp.concatenate([u, kept_before[:, None] * c, before[:, :1] * c], axis=-1) / scale
    solved = solve_triangular(system, sides, lower=True, unit_diagonal=True)
    width = c.shape[-1]
    inputs, by_first, by_second = jnp.split(solved, [u.shape[-1], u.shape[-1] + width], axis=-1)
    scores = g[:, 1:] * (q @ b.T)
    ends = g[-1], last
    entering = kept[-1] / scale, ends[0][0] / scale, ends[1][0] / scale
    return (inputs, by_first, by_second, scores, kept[:, None] * q / scale, g[:, :1] * q / scale, r,
            b * ends[0][1:, None], b * ends[1][1:, None], *entering, scale)


def _scale(kept, entering):
    """A power of two at least one and every decayed sum the entering pair is weighed by in a chunk (`A_t`,
    `b_t`): each term the pair reaches divided by it before its products, the chunk's sums with it after
    them, so a term past float32's range where their sum is not cancels before it is rounded (2^127 entering
    both states, decays of one and write gates of one: `b_2 M_0` is -2^128 where the memory leaving is
    -2^127, ReviewMomentum1454); exact either way, a power of two."""
    largest = jnp.maximum(jnp.max(jnp.abs(kept)), jnp.max(jnp.abs(entering)))
    return jnp.exp2(jnp.ceil(jnp.log2(jnp.maximum(largest, 1.0))))


def _inputs(prepared: tuple, pair):
    """A chunk's inputs `W` from the pair entering it, divided by the chunk's scale (`_prepared`)."""
    inputs, by_first, by_second = prepared[:3]
    return inputs + by_first @ pair[0] + by_second @ pair[1]


def _stepped(state, prepared: tuple, issue=None):
    """The pair after a prepared chunk (`derive_coupled`'s `step`), its walk's decay as it was."""
    del issue
    decay, pair = state
    w = _inputs(prepared, pair)
    *_, first_keys, second_keys, kept, entering_first, entering_second, scale = prepared
    first = (kept * pair[0] + entering_first * pair[1] + first_keys.T @ w) * scale
    second = (entering_second * pair[1] + second_keys.T @ w) * scale
    return decay, jnp.stack([first, second])


def _mapped(prepared: tuple, state, issue=None):
    """A chunk's outputs from the pair entering it (`derive_coupled`'s `mapped`)."""
    del issue
    pair, scale = state[1], prepared[-1]
    scores, first_queries, second_queries, free = prepared[3:7]
    read = first_queries @ pair[0] + second_queries @ pair[1] + scores @ _inputs(prepared, pair)
    return read * scale + free


def _replayed(declaration, names: Sequence[str], parts: linear_scan.Coupled, chunk: Mapping, pair) -> tuple:
    """`chunk` walked token by token from the pair entering it (`_at_token`'s parts): `(outputs, pair)`, each
    token's output and the pair leaving the chunk, a reset token's entering pair selected to zero."""
    along, every = _walked_tokens(declaration, chunk)
    count = next(iter(along.values())).shape[0]
    along.update({name: jnp.broadcast_to(value, (count, *value.shape)) for name, value in every.items()})

    def token(carried, step):
        alpha, mu, lam, b, u, c, q, r = _at_token(declaration, names, parts, step)
        first, second = carried
        if declaration.reset is not None:
            first, second = (jnp.where(step[declaration.reset], 0.0, x) for x in (first, second))
        w = u + jnp.sum(c[:, None] * first, axis=0)
        second = mu * second + b[:, None] * w[None, :]
        first = alpha * first + lam * second
        return (first, second), jnp.sum(q[:, None] * first, axis=0) + r

    # The pair carried in the type its update's parts widen it to (`scan._carry_type`, the single state's
    # walk's): a float32 pair over float64 tokens (x64's) widens once.
    one = {name: jax.ShapeDtypeStruct(x.shape[1:], x.dtype, weak_type=jax.typeof(x).weak_type)
           for name, x in along.items()}
    def carry_typed(step):
        # The update's parts (its readout's none): its decays, weight, input and correction.
        return jnp.zeros((), _carry_type(pair, *_at_token(declaration, names, parts, step)[:6]))

    pair = pair.astype(jax.eval_shape(carry_typed, one).dtype)
    (first, second), outputs = lax.scan(token, (pair[0], pair[1]), along)
    return outputs, jnp.stack([first, second])
