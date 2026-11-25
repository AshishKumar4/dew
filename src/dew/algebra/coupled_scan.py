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
logarithm, so its sums are running products a token at a time, not differences of a compensated prefix as
the single state's decays are, each a scaled value (`dew.algebra.scaled`) held at its own range: a decay
past float32's least normal is kept, and no sum is divided by another's (`_exponential`). Each term a sum
weighs is brought into it once, the memory's and the outputs' terms summed in scaled arithmetic and landed
once (`_sum`, `_summed`): a term past float32's range where the result is not cancels before it is rounded,
as the token walk's running state does, and a read of the entering pair (`_read`) a key at a time, weighed
after; each result the float computation's where it is its own (`_landed`). The solve is one forward
substitution in scaled arithmetic (`_substituted`), its system's weights and sides never landed, its step's
(`Derived.step`'s `kept`), which the outputs read again. The chunk's adjoint is its token walk's (`replay`),
differentiated, as the single state's is, its faint decays never landed alone (`_decayed`). A packed
document's first token (`Scan.reset`) enters a zero pair: its decays and its correction zero, the walk's
pair selected to zero.

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
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from dew import _cache
from dew.algebra import derive, linear_scan, program, scaled, terms
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
from dew.nn.float_bits import exact_power2
from dew.nn.precision import at_least_fp32, power2
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
        return jax.vmap(lambda token: _chunk_parts(declaration, names, parts, {**token, **every}))(along)

    def init(shapes):
        del shapes
        return jnp.zeros((shape[0],), carried), jnp.zeros((2, *shape), carried)

    def prepare(chunk, issue=None, transition: bool = False, window=None):
        if issue is not None or transition or window is not None:
            raise Refused("a pair's chunk takes no mainloop's issuer, transfer or document window here")
        return _prepared(*tokens(chunk), base2=affine_of(chunk).base2)

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


def _chunk_parts(declaration, names: Sequence[str], parts: linear_scan.Coupled, token: Mapping) -> tuple:
    """One token's exponents, `lam`, `B`, `U`, `C`, `Q` and `R` (`_token`) and whether a document starts at
    it (`cut`), its correction zero there: it reads nothing of the pair entering it."""
    first, second, lam, b, u, c, q, r = _token(parts, tuple(token[name] for name in names))
    cut = (jnp.zeros((), jnp.bool_) if declaration.reset is None
           else jnp.asarray(token[declaration.reset], jnp.bool_))
    return first, second, lam, b, u, jnp.where(cut, 0.0, c), q, r, cut


def _at_token(declaration, names: Sequence[str], parts: linear_scan.Coupled, token: Mapping) -> tuple:
    """One token's decays `alpha` and `mu` as floats, `lam`, `B`, `U`, `C`, `Q` and `R` (`_chunk_parts`), a
    reset token's decays zero too: the types the token walk carries its pair in (`_replayed`)."""
    first, second, lam, b, u, c, q, r, cut = _chunk_parts(declaration, names, parts, token)
    alpha, mu = (jnp.where(cut, 0.0, _exp(parts.base2, x)) for x in (first, second))
    return alpha, mu, lam, b, u, c, q, r


def _decayed(state, exponent, base2: bool, cut) -> tuple:
    """`exp(exponent) state` (`2^exponent` where `base2`) as a factor and a term, `factor state + term`,
    zero where `cut`: the scan's own decay (`scan._exp`) the factor where it is a normal float, the token
    walk's rounding, its expression the walk's; beneath float32's least normal the factor zero and the term
    the remainder's exponential times the state scaled by the base-2 exponent's nearest integer power of
    two first (`precision.power2`), so the product and its derivative in the exponent stay what the decay of
    a large state leaves (exp(-120) over 2^100, ReviewMomentum1454), as the chunk's forms keep them
    (`_exponential`), never the decay landed alone, nor a cotangent scaled beneath the least normal before
    the state it meets (power2's pullback scales its cotangent first)."""
    decay = _exp(base2, exponent)
    # A faint decay's exponent above the least any finite state outlives (beneath it, an exact zero's -inf
    # or float32's least, the decay of the largest state underflows, read as none): neither arm reads -inf
    # less its own nearest integer, NaN, nor differentiates through it (ReviewMomentum1454).
    limits = np.finfo(decay.dtype)
    outlived = -(limits.maxexp - limits.minexp + limits.nmant + 2) * (1.0 if base2 else math.log(2.0))
    faint = (jnp.abs(decay) < limits.tiny) & (exponent > outlived) & ~cut
    exponent = jnp.where(faint, exponent, 0.0)
    if base2:
        high, low = exponent, jnp.zeros_like(exponent)
    else:
        high, low = scaled.base2(exponent)
    whole = lax.stop_gradient(jnp.round(high))
    fraction = (high - whole) + low
    split = jnp.exp(fraction * np.asarray(math.log(2), fraction.dtype)) * power2(state, whole)
    return jnp.where(cut | faint, 0.0, decay), jnp.where(faint, split, 0.0)


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
    return min(max(low, -largest), largest), min(max(high, -largest), largest)


FLOOR = -(1 << 16)
"""The least power of two a chunk's decayed sum is held at (`_floored`): one below it is zero, as each
term it weighs (a float32's or a float64's, at most `2^1024`) lands at zero beneath it."""

SUMS_REACH = -FLOOR + 1
"""The bound a decayed sum's exponent is held within (`scaled.Scaled.reach`): `FLOOR` below, the declared
ranges' bound on the sums above (`_bounded`, under float32's range)."""


def _exponential(exponent, base2: bool, cut) -> scaled.Scaled:
    """`exp(exponent)` (`2^exponent` where `base2`) as a scaled value, zero where `cut` (a document's first
    token): the scan's own decay (`scan._exp`, fp32's rounding) where it is a normal float, exactly as a
    scaled value; else its power of two the base-2 exponent's nearest integer and its significand the
    remainder's exponential (`windows`' factors' form), zero below `FLOOR`: a decay past float32's least
    normal kept, and an ordinary one the token walk's own."""
    natural = _exp(base2, exponent)
    ordinary = jnp.isfinite(natural) & (natural >= np.finfo(natural.dtype).tiny)
    if base2:
        high, low = exponent, jnp.zeros_like(exponent)
    else:
        high, low = scaled.base2(exponent)
    held = (high > FLOOR) & ~cut
    high, low = jnp.where(held, high, 0.0), jnp.where(held, low, 0.0)
    whole = jnp.floor(high)
    whole = whole + (high - whole >= 0.5).astype(high.dtype)
    fraction = (high - whole) + low
    significand = jnp.where(held, jnp.exp(fraction * np.asarray(math.log(2), fraction.dtype)), 0.0)
    split = scaled.normalized(significand, whole.astype(jnp.int32), SUMS_REACH)
    exact = scaled.scaled(jnp.where(ordinary, natural, 0.0))
    return scaled.where(ordinary & ~cut, exact, split)


def _floored(value: scaled.Scaled) -> scaled.Scaled:
    """A decayed sum held within `SUMS_REACH`: zero below `FLOOR` (`FLOOR`'s account), its reach restated."""
    kept = value.exponent >= FLOOR
    return scaled.Scaled(jnp.where(kept, value.significand, 0.0), jnp.where(kept, value.exponent, 0),
                         SUMS_REACH)


ORDINARY = 2.0 ** -90
"""The least magnitude a float result is taken at for its scaled twin (`_taken`): every term it flushed,
under float32's least normal, at most 2^-118 together over 256 terms, beneath its rounding."""


def _taken(running, twin: scaled.Scaled):
    """Where a float result is its own (`_ordinary`, `_landed`): finite, and ordinary (`ORDINARY`) or zero
    where its scaled twin is; no term of it then overflowed, and those it flushed are beneath its rounding."""
    zero = (running == 0) & (twin.significand == 0)
    return jnp.isfinite(running) & ((jnp.abs(running) >= ORDINARY) | zero)


def _ordinary(running, twin: scaled.Scaled, faint) -> scaled.Scaled:
    """A decayed sum the float running product's (its FMA-contracted rounding, as the token walk's) where it
    is its own (`_taken`) and no decay of the chunk is `faint` (beneath float32's least normal, which the
    float's products would have read as zero); else its scaled twin's, past the float's range."""
    taken = _taken(running, twin) & ~faint
    exact = scaled.scaled(jnp.where(taken, running, 0.0))
    return scaled.Scaled(jnp.where(taken, exact.significand, twin.significand),
                         jnp.where(taken, exact.exponent, twin.exponent), SUMS_REACH)


def _landed(running, twin: scaled.Scaled, flushed=None):
    """A result the float computation's (matrix products' rounding, as the token walk's) where it is its own
    (`_taken`) and no decayed sum's product with a value it went on to multiply flushed (`flushed`,
    `_weighed`); else its scaled twin's landed once, past the float's range."""
    taken = _taken(running, twin)
    if flushed is not None:
        taken = taken & ~flushed
    return jnp.where(taken, running, scaled.landed(twin))


def _weighed(values, sums: scaled.Scaled):
    """`values` weighed by their decayed sums (`scaled.applied`: no intermediate leaves the range its result
    does not), and where a product flushed beneath float32's least normal though neither factor is zero, as
    0 or 1 (`_reached`)."""
    weighed = scaled.applied(values, sums)
    flushed = (values != 0) & (sums.significand != 0) & (jnp.abs(weighed) < np.finfo(weighed.dtype).tiny)
    return weighed, flushed.astype(weighed.dtype)


def _reached(product) -> jax.Array:
    """Which entries a matrix product of flushed products' indicators (`_weighed`) and nonzero values'
    reached (`_landed`'s `flushed`)."""
    return product > 0


def _decayed_sums(alpha: scaled.Scaled, mu: scaled.Scaled, lam) -> tuple:
    """A chunk's sums from its decays `alpha` and `mu` (scaled values) and weights `lam` `[C]`, a token at a
    time, each a running product of scaled values (`dew.algebra.scaled`: a 2x2 product's sums reach past
    float32's range where the terms they weigh do not) beside its float twin, the float's where it is
    ordinary (`_ordinary`): `g` `[C, C + 1]`, row t `g(s, t)` at column `s + 1` for token `s` (none past `t`)
    and `b_t` at column zero; the last token's `m(., C)` `[C + 1]` (`m(0, C)` at column zero), and `A` `[C]`.
    `m(s, t) = mu_t m(s, t - 1)`, one at `s = t`; `g(s, t) = alpha_t g(s, t - 1) + lam_t m(s, t)`."""
    tokens = lam.shape[-1]
    # The sums in the widest of their factors' types (a float64 decay beside a float32 gate's, x64's).
    lam = lam.astype(jnp.result_type(alpha.significand, mu.significand, lam))
    own = jnp.eye(tokens + 1, dtype=jnp.bool_)[1:]
    weights = scaled.scaled(lam)
    floats = (scaled.landed(alpha).astype(lam.dtype), scaled.landed(mu).astype(lam.dtype), lam)
    tiny = np.finfo(lam.dtype).tiny
    faint = jnp.any(jnp.stack([(jnp.abs(running) < tiny) & (twin.significand != 0)
                               for running, twin in ((floats[0], alpha), (floats[1], mu))]))

    def token(carried, step):
        scaled_parts, float_parts = carried
        g, m, kept = (scaled.Scaled(significand, exponent, SUMS_REACH)
                      for significand, exponent in scaled_parts)
        g_f, m_f, kept_f = float_parts
        a, u, weight, mine, a_f, u_f, lam_f = step
        one = scaled.one_like(m.significand)
        m = _floored(scaled.where(mine, one, scaled.mul(m, u)))
        g = _floored(scaled.add(scaled.mul(a, g), scaled.mul(weight, m)))
        kept = _floored(scaled.mul(kept, a))
        m_f = jnp.where(mine, 1.0, m_f * u_f)
        g_f = a_f * g_f + lam_f * m_f
        kept_f = kept_f * a_f
        held = tuple((x.significand, x.exponent) for x in (g, m, kept))
        return (held, (g_f, m_f, kept_f)), (held[0], held[2], g_f, kept_f)

    # The carries typed as the tokens vary (a checked `shard_map`'s manual axes), as a step's outputs vary.
    def seeded(*parts):
        return tuple(varying_like(x, alpha.significand, mu.significand, lam) for x in parts)

    zeros, exponents = jnp.zeros((tokens + 1,), lam.dtype), jnp.zeros((tokens + 1,), jnp.int32)
    first = zeros.at[0].set(1.0)
    carried = ((seeded(zeros, exponents), seeded(first, exponents),
                seeded(jnp.ones((), lam.dtype), jnp.zeros((), jnp.int32))),
               seeded(zeros, first, jnp.ones((), lam.dtype)))
    ((_, last, _), (_, last_f, _)), (g, kept, g_f, kept_f) = lax.scan(
        token, carried, (alpha, mu, weights, own, *floats))
    return tuple(_ordinary(running, scaled.Scaled(significand, exponent, SUMS_REACH), faint)
                 for running, (significand, exponent) in ((g_f, g), (last_f, last), (kept_f, kept)))


def _shifted(value: scaled.Scaled, first: float) -> scaled.Scaled:
    """`value` a token back along its first axis, the chunk's first token's `first` (zero or one)."""
    def back(part, fill):
        return jnp.concatenate([jnp.full_like(part[:1], fill), part[:-1]])

    return scaled.Scaled(back(value.significand, first), back(value.exponent, 0), value.reach)


def _sliced(value: scaled.Scaled, index) -> scaled.Scaled:
    """`value` at `index`."""
    return scaled.Scaled(value.significand[index], value.exponent[index], value.reach)


LOCAL_REACH = SUMS_REACH + 2048
"""The bound a chunk's scaled sum of weighed products holds its exponent within (`_summed`): a decayed sum's
(`SUMS_REACH`) beside a float64 product's range and the chunk's tokens."""


class Chunked(NamedTuple):
    """A chunk's values that read no state (`_prepared`), its decayed sums scaled values (`_decayed_sums`)
    each brought into its term once, none divided by another's factor: the system's strictly lower part `L`
    (`L_ts = g(s, t - 1) C_t . B_s`) scaled, never landed (`_substituted`), the tokens' inputs `U`
    and corrections `C`, the entering memory's and momentum's factors a token back (`A_{t-1}`, `b_{t-1}`),
    the queries' dots with the keys and their sums `g(s, t)`, the entering memory's factors to each token
    (`A_t`), the queries and the momentum's factors to each token (`b_t`), the free readouts, the keys and
    their sums to the chunk's end by the memory (`g(s, C)`) and the momentum (`m(s, C)`), and the entering
    memory's, momentum's and momentum's own factors there (`A_C`, `b_C`, `m(0, C)`)."""

    lower: scaled.Scaled
    inputs: jax.Array
    corrections: jax.Array
    kept_before: scaled.Scaled
    momentum_before: scaled.Scaled
    dots: jax.Array
    sums: scaled.Scaled
    kept_to: scaled.Scaled
    queries: jax.Array
    momentum: scaled.Scaled
    free: jax.Array
    keys: jax.Array
    memory_sums: scaled.Scaled
    momentum_sums: scaled.Scaled
    kept: scaled.Scaled
    entering_momentum: scaled.Scaled
    momentum_kept: scaled.Scaled


def _prepared(first, second, lam, b, u, c, q, r, cut, *, base2: bool) -> Chunked:
    """A chunk's values that read no state (`Chunked`), from its tokens' parts (`_chunk_parts`)."""
    alpha, mu = _exponential(first, base2, cut), _exponential(second, base2, cut)
    g, last, kept = _decayed_sums(alpha, mu, lam)
    before = _shifted(g, 0.0)
    # Each overlap `C_t . B_s` a key at a time in scaled arithmetic, never rounded alone beneath the least
    # normal before its weight (2^-150 over 2^100, ReviewMomentum1454).
    overlaps = _summed(_outer, (c.T, b.T), (c.shape[0], b.shape[0]))
    lower = scaled.mul(overlaps, _sliced(before, (..., slice(1, None))))
    ends = _sliced(g, -1)
    return Chunked(
        lower=lower, inputs=u, corrections=c, kept_before=_shifted(kept, 1.0),
        momentum_before=_sliced(before, (..., 0)), dots=q @ b.T, sums=_sliced(g, (..., slice(1, None))),
        kept_to=kept, queries=q, momentum=_sliced(g, (..., 0)), free=r, keys=b,
        memory_sums=_sliced(ends, slice(1, None)), momentum_sums=_sliced(last, slice(1, None)),
        kept=_sliced(kept, -1), entering_momentum=_sliced(ends, 0), momentum_kept=_sliced(last, 0))


def _sum(*terms) -> scaled.Scaled:
    """The terms' sum in scaled arithmetic, each a float or a scaled value: a term past the format's range
    where the sum is not cancels before it is rounded (2^127 entering both states under decays and write
    gates of one: `b_2 M_0` is -2^128 where the memory leaving is -2^127, ReviewMomentum1454)."""
    total = None
    for term in terms:
        term = term if isinstance(term, scaled.Scaled) else scaled.scaled(term)
        total = term if total is None else scaled.add(total, term)
    if total is None:
        raise ValueError("a sum of no terms")
    return total


def _total(value: scaled.Scaled) -> scaled.Scaled:
    """`value`'s sum over its first axis in scaled arithmetic at once: each term's significand moved to the
    largest exponent among the nonzero ones (exact, a power of two, at most its format's `shift`, as
    `scaled.add` aligns), the moved significands summed once in float and renormalized."""
    live = value.significand != 0
    top = jnp.max(jnp.where(live, value.exponent, jnp.iinfo(jnp.int32).min), axis=0)
    top = jnp.where(jnp.any(live, axis=0), top, 0)
    form = value.format
    moved = value.significand * exact_power2(jnp.clip(value.exponent - top, -form.shift, 0),
                                             value.significand.dtype)
    widened = value.reach + form.shift + form.mantissa + 3 + math.ceil(math.log2(max(value.shape[0], 2)))
    return scaled.normalized(jnp.sum(moved, axis=0), top, widened)


def _column(value: scaled.Scaled) -> scaled.Scaled:
    """A token's factor `[C]` laid over its row's values `[C, 1]`."""
    return _sliced(value, (..., None))


def _summed(factors, operands: tuple, shape: tuple[int, ...], weights: scaled.Scaled | None = None
            ) -> scaled.Scaled:
    """`sum_s weights_s x_s y_s` (the products alone where `weights` is None), `(x_s, y_s)` the
    `factors(*operands_s)` over the operands' and the weights' first axis, a term at a time in scaled
    arithmetic, each product of the factors' scaled values: a sum of products past the format's range where
    the total is not kept (ReviewMomentum1454: `g(1, 2) w_1` -2^128 where the memory leaving is -2^127),
    and a product beneath its least normal kept for the weight that raises it (`q M` 2^-129 under a weight
    of -16, -2^-125), its carry typed as its operands vary (a checked `shard_map`'s)."""
    def term(total, step):
        *each, significand, exponent = step
        x, y = factors(*each)
        weighed = scaled.mul(scaled.scaled(x), scaled.scaled(y))
        if weights is not None:
            weighed = scaled.mul(weighed, scaled.Scaled(significand, exponent, weights.reach))
        total = scaled.add(scaled.Scaled(total[0], total[1], LOCAL_REACH), weighed)
        return (total.significand, total.exponent), None

    count = operands[0].shape[0]
    held = ((jnp.zeros((count,), operands[0].dtype), jnp.zeros((count,), jnp.int32)) if weights is None
            else (weights.significand, weights.exponent))
    # The sum in the widest of its terms' types (a float64 weight beside float32 factors, x64's).
    wide = jnp.result_type(*operands, held[0])
    zero = tuple(varying_like(jnp.zeros(shape, kind), *operands, *held) for kind in (wide, jnp.int32))
    (significand, exponent), _ = lax.scan(term, zero, (*operands, *held))
    return scaled.Scaled(significand, exponent, LOCAL_REACH)


def _outer(left, right) -> tuple:
    """A column of `left` beside a row of `right`: their outer product's factors (`_summed`)."""
    return left[:, None], right[None, :]


def _read(vectors, state) -> scaled.Scaled:
    """Each token's vector's read of a state, `vectors @ state` `[C, V]`, a key at a time in scaled
    arithmetic (`_summed`), weighed after by a decayed sum: no read overflows before its decay where the
    token walk's of the decayed state does not, and none flushes after it where the read is normal (a query
    of 1/2 each of four keys over a memory of 2^127 under a decay of 1/2; a decay of exp(-86) over a memory
    of 2^20, ReviewMomentum1454)."""
    return _summed(_outer, (vectors.T, state), (vectors.shape[0], state.shape[-1]))


def _substituted(lower: scaled.Scaled, sides: scaled.Scaled):
    """`W` of `(I - L) W = sides`, a token at a time in scaled arithmetic, the chunk's one solve: each row its
    side plus the earlier rows' weights by `L`, both scaled, never landed alone, summed in scaled arithmetic
    (`_total`) and landed once: a side past the format's range that the correction brings back (2^127
    entering the memory and -2^127 the momentum under decays and write gates of one: a side 2^128 where the
    input is 2^127) and a weight beneath its least normal that a large earlier row raises (`L_21 = -2^-155`
    over `w_1 = 2^100`, -2^-55) each kept (ReviewMomentum1454)."""
    def row(w, t):
        weights = scaled.Scaled(lower.significand[t][:, None], lower.exponent[t][:, None], lower.reach)
        products = scaled.mul(weights, scaled.scaled(w))
        side = scaled.Scaled(sides.significand[t], sides.exponent[t], sides.reach)
        return w.at[t].set(scaled.landed(scaled.add(side, _total(products)))), None

    start = varying_like(jnp.zeros(sides.shape, sides.significand.dtype), lower.significand,
                         sides.significand)
    w, _ = lax.scan(row, start, jnp.arange(lower.shape[0]))
    return w


def _inputs(chunk: Chunked, pair):
    """A chunk's inputs `W` from the pair entering it: the system solved once in scaled arithmetic
    (`_substituted`) against each token's input and its correction's reads of the entering memory and
    momentum, each weighed by its factor, the sides never landed."""
    sides = _sum(chunk.inputs, scaled.mul(_read(chunk.corrections, pair[0]), _column(chunk.kept_before)),
                 scaled.mul(_read(chunk.corrections, pair[1]), _column(chunk.momentum_before)))
    return _substituted(chunk.lower, sides)


def _leaving(entering_terms: tuple, keys, w, sums: scaled.Scaled):
    """A state leaving a chunk: its entering terms (`(state, factor)` pairs) and the chunk's writes `B^T W`
    weighed by their sums to its end, the float computation's where it is its own (`_landed`), else its
    twin's in scaled arithmetic."""
    written = _summed(_outer, (keys, w), (keys.shape[-1], w.shape[-1]), sums)
    twin = _sum(*(scaled.mul(scaled.scaled(state), factor) for state, factor in entering_terms), written)
    weighed, flushed = _weighed(w, _column(sums))
    running = sum((scaled.applied(state, factor) for state, factor in entering_terms), keys.T @ weighed)
    return _landed(running, twin, _reached((keys != 0).astype(w.dtype).T @ flushed))


def _stepped(state, chunk: Chunked, issue=None):
    """The pair after a prepared chunk and the chunk's inputs it solved (`derive_coupled`'s `step`, kept
    for `_mapped`), its walk's decay as it was: the memory the entering memory decayed, the entering
    momentum weighed by its factor and the chunk's writes weighed by theirs; the momentum the entering one
    decayed and the chunk's writes (`_leaving`)."""
    del issue
    decay, pair = state
    w = _inputs(chunk, pair)
    memory = _leaving(((pair[0], chunk.kept), (pair[1], chunk.entering_momentum)), chunk.keys, w,
                      chunk.memory_sums)
    momentum = _leaving(((pair[1], chunk.momentum_kept),), chunk.keys, w, chunk.momentum_sums)
    return (decay, jnp.stack([memory, momentum])), w


def _mapped(chunk: Chunked, state, w, issue=None):
    """A chunk's outputs from the pair entering it and the inputs its step solved (`derive_coupled`'s
    `mapped`): each token's query's reads of the entering memory and momentum weighed by their factors, its
    dots with the chunk's keys weighed by their sums against the chunk's inputs (`_summed`) and its free
    readout, the float computation's where it is its own, else its scaled twin's (`_landed`)."""
    del issue
    pair = state[1]
    weights = scaled.Scaled(chunk.sums.significand.T[..., None], chunk.sums.exponent.T[..., None],
                            chunk.sums.reach)
    scored = _summed(_outer, (chunk.dots.T, w), (chunk.dots.shape[0], w.shape[-1]), weights)
    memory = scaled.mul(_read(chunk.queries, pair[0]), _column(chunk.kept_to))
    momentum = scaled.mul(_read(chunk.queries, pair[1]), _column(chunk.momentum))
    scores, flushed = _weighed(chunk.dots, chunk.sums)
    running = (scaled.applied(chunk.queries @ pair[0], _column(chunk.kept_to))
               + scaled.applied(chunk.queries @ pair[1], _column(chunk.momentum)) + scores @ w + chunk.free)
    reached = _reached(flushed @ (w != 0).astype(w.dtype))
    return _landed(running, _sum(memory, momentum, scored, chunk.free), reached)


def _replayed(declaration, names: Sequence[str], parts: linear_scan.Coupled, chunk: Mapping, pair) -> tuple:
    """`chunk` walked token by token from the pair entering it (`_chunk_parts`' parts, each decay applied
    unlanded, `_decayed`): `(outputs, pair)`, each token's output and the pair leaving the chunk, a reset
    token's entering pair selected to zero."""
    along, every = _walked_tokens(declaration, chunk)
    count = next(iter(along.values())).shape[0]
    along.update({name: jnp.broadcast_to(value, (count, *value.shape)) for name, value in every.items()})

    def token(carried, step):
        first_exponent, second_exponent, lam, b, u, c, q, r, cut = _chunk_parts(declaration, names, parts,
                                                                                 step)
        first, second = carried
        if declaration.reset is not None:
            first, second = (jnp.where(step[declaration.reset], 0.0, x) for x in (first, second))
        w = u + jnp.sum(c[:, None] * first, axis=0)
        mu, faint_second = _decayed(second, second_exponent, parts.base2, cut)
        alpha, faint_first = _decayed(first, first_exponent, parts.base2, cut)
        second = mu * second + b[:, None] * w[None, :] + faint_second
        first = alpha * first + lam * second + faint_first
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
