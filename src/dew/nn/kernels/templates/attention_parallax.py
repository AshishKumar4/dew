"""Parallax attention (docs/design/kernel-templates.md, row 54): the softmax of a query `q` by its keys `k`
over the values `v`, each weight `w` taken by `1 + c - t`, where `t = r · k2ᵀ` is a second product, unscaled,
of a second query `r` by second keys `k2`, and `c = Σ w t` its mean under the weights. Its output is the
values' mean less their covariance with the second product:

    out = a - C,    a = Σ w v,    C = Σ w (t - c)(v - a).

Its walk holds each query row's moments at once, never their raw sums (`Moments`: the softmax's maximum and
denominator, `a`, `c` and `C`), a key tile's own (`moments`) merged into the row's by Chan's pairwise law
(`merged`):

    c = s_A c_A + s_B c_B,   a = s_A a_A + s_B a_B,   C = s_A C_A + s_B C_B + s_A s_B (c_A - c_B)(a_A - a_B),

`s_A` and `s_B` the two parts' shares of the merged denominator. Every term is a mean or a centred moment, of
the size of the spread it measures: no `a (1 + c) - Σ w t v` cancels past fp32's range (train 1407's
review, P0-3), no anchor fixed by the first tile reads a later tile's keys at the first's scale (P0-5).

Its ranges, each by a power of two:

- the second products as `t̂ = 2^p t` (`second_powers`): a row's second query taken by the power `2^p` that
  holds every product with the keys it sees within `2^125` before the product forms (a raw `2^100 · 2^40`
  overflowed before any centring; a hidden key bounds nothing); `C` a mantissa and its power
  (`Moments.spread`), so a tile's covariance past fp32 comes back through a merge's share;
- a weight met by a moment before either rounds: a key's `w (t - c)` formed as `(t - c)` decayed by its
  logit's distance below the maximum (`exp_scaled`, every power of two applied at once), so a weight of
  `e^-128` against a second product of `2^126` keeps their product, where the weight alone underflowed
  (P0-4), and a share `s_A` of `e^-256` meets its tile's spread before it rounds (`merged`);
- the values' differences in halves (`(a_A - a_B) / 2`, `v / 2 - a / 2`), and every sum of signed
  weights' moments at weights summing to at most 1 (`moments`' `ũ`), so no partial sum of a covariance
  passes fp32's range unless the covariance does (P0-2's signed weights never enter the softmax's raised
  tiers);
- `c` as a pair `c_hi + c_lo` (`Moments.centre`), so `t - c` holds the products' spread where their mean
  is no fp32 (`2^25` and `2^25 + 4`: `c = 2^25 + 2`).

Its derivative, from the forward's moments of each row (`Moments`) and the output's cotangent `g`, a key
tile at a time over every query (`derivative`), with `τ = t - c` and `δ = g · (v - a)`:

    dS = w (δ (1 - τ) + g · C)          the logits'
    dT = -w δ                           the second products'
    dV = w (1 - τ) g

so `dq = scale Σ dS k`, `dk = scale Σ dSᵀ q`, `dr = Σ dT k2`, `dk2 = Σ dTᵀ r`. Each coefficient is formed
with its power of two (`g · v` at its row's power over the pairs it sees, `g · a` and `g · C = Σ w δ τ` at
their rows' running powers, `_summed_rows`; `g · C - δ τ` subtracted at their common power before `δ` meets
them), then landed with its decay at its contraction's own power (`_landing`: a row's for `dq` and `dr`, a
key's for `dk`, `dk2` and `dV`) before its product with its operand and the scale, and taken back by that
power after, so a term fp32 holds is kept whatever its factors (train 1423's review).

Its range contract (Main's rulings of 2026-11-27 on Parallax P-II): every weight and moment, and every
product of one weight with one cotangent, value, key or query, is formed range-safe, so a term fp32 holds
is kept however far its factors lie apart; within one contraction (`dV = Σ w (1 - τ) g`, `dq`, `dk`, `dr`,
`dk2`), terms whose magnitudes span more than fp32's range relative to the row's largest (an `e^-160`
beside 1 under a cotangent of `2^127`) are held to the template's own attention and plain fp32 softmax
attention, which lose them alike (no exponent tiers; row 14033). The shared dots, `r · k2` and `g · v`, are
taken at one power a row over the pairs it sees (`_row_bounds`): its largest visible dot held below `2^125`
(`HELD`, the headroom threshold), a visible dot more than fp32's range below that largest held to plain
fp32's parity (row 14033); the row's power taken by its second query before its products, so an entry of
`r` more than fp32's range below the row's largest visible dot past `2^125` flushes, as plain fp32's
product overflows there (Main's ruling on train 1444's review, row 14033). Each value channel is taken
to one power a tile over the keys its queries see (`_channel_powers`), the covariance's one `[Q, K] @
[K, Dv]` contraction: a row's seen values more than fp32's range below another row's seen value of the same
channel and key head are held to plain fp32's parity, and so is a row's: two of its live keys' values in
one channel more than fp32's range apart, `2^127` beside `2^-120` (Main's rulings of 2026-11-28, row 14033;
rank-one key and channel scaling would still leave two channels of opposite extremes). Every higher
order differentiates the dense reference (`reference`), whose tails read as the plain formula's (row 14007):
a lifted term meets a tangent or cotangent at its own size, its power taken first (`_shifted`).

It is XLA's alone: no launch of a Pallas mainloop (a GH200's Triton walk spilled at every backward tile and
walked its forward 10 times the template's sdpa, M-NA-58b)."""

import functools
import math
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_derivatives import zero_from_primal

from dew.algebra import means
from dew.algebra.windows import Units
from dew.nn.derivatives import KernelGlue, contracted, surrogate
from dew.nn.kernels import generation
from dew.nn.kernels.templates.attention import REFERENCE_PRECISION, KeySpans, dense, logit_scale, prescaled
from dew.nn.precision import accurate_exp, float32_einsum, ln2_remainder, power2, two_sum

KEY_TILE = 128
"""The keys a walk's step takes (`walk`, `derivative`)."""

HELD = 125
"""The power of two below which a row's products are held: its second products (`second_powers`), its
cotangent's products with values and covariance, and a contraction's landed terms (`derivative`)."""

_ABSENT = -1024
"""The exponent `_exponent` gives a zero, below every product's."""

_SPAN = 280
"""The powers of two fp32 spans, its least subnormal to its largest, and a margin: a term more than this below
another of its sum is none of it (`_channel_powers`)."""

_NONE = -1000
"""A bound over no pair: a row that sees no key, or a pair of a zero."""


class Moments(NamedTuple):
    """A query row's moments over the keys it has met `[B, Hk, G, Q]` (each value `[..., Dv]`), fp32: the
    largest logit `m` (-inf before any key) and the denominator `total` at it, the values' mean `mean`, the
    second products' mean `centre` as a pair `(hi, lo)` summing to it, in the row's units (`t̂`,
    `second_powers`), and their covariance with the values `spread · 2^spread_power`, a channel at a time
    (`[..., Dv]` each: a channel of `2^-100` beside one of `2^127` its own), a mantissa of at most
    1 and its row's power of two, at least 0: a part's covariance past fp32's range carried through a
    merge whose shares bring it back (`merged`), one below 1 held as it is (its power 0, a zero's too, so
    a covariance that cancels bounds nothing after it), its tangent then the covariance's own (train 1444's
    review)."""

    m: jax.Array
    total: jax.Array
    mean: jax.Array
    centre: tuple[jax.Array, jax.Array]
    spread: jax.Array
    spread_power: jax.Array


def _exponent(x: jax.Array) -> jax.Array:
    """The power of two each entry of `x` lies below in magnitude, int32: `|x| < 2^e` (`frexp`'s), `_ABSENT`
    for a zero; read apart from any derivative."""
    _, exponent = jnp.frexp(jax.lax.stop_gradient(x))
    return jnp.where(x == 0, _ABSENT, exponent)


def _reach(below: jax.Array) -> jax.Array:
    """The power of two a decay `e^below` (`below` at most 0, -inf none) lies below, int32, `_NONE` for
    none: held at `2 _NONE` before it is an integer, so a decay of `e^-1488522240` beside a power of two
    sums below every bound, where its `INT32_MIN` wrapped past them (train 1444's review)."""
    reach = jnp.maximum(jnp.floor(jax.lax.stop_gradient(below) * np.log2(np.e)) + 1, 2 * _NONE)
    return jnp.where(jnp.isfinite(below), reach, _NONE).astype(jnp.int32)


def _normalized(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """`x` `[..., D]` as a mantissa of at most 1 in magnitude and its row's power of two (`x = m 2^e`, `e`
    int32 `[...]`, 0 for a row of zeros)."""
    largest = jnp.max(_exponent(x), axis=-1)
    power = jnp.where(largest > _ABSENT, largest, 0)
    return power2(x, -power[..., None]), power


def exp_scaled(z: jax.Array, exponent: jax.Array, power) -> jax.Array:
    """`z e^exponent 2^power` (`exponent` natural units, -inf a zero; `power` integral), rounded once where
    the whole is: the decay split as `2^k e^f` (`k` its whole powers of two, `f` in `[0, ln 2)` by
    `precision.ln2_remainder`), `z e^f` formed, and every power, `k` and `power` together, applied at once
    (`precision.power2`), so no factor's partial product leaves fp32 before the rest brings it back: a
    weight of `e^-128` against `2^126` keeps `2^-58`, and `2^127` decayed by `e^-256` then raised by
    `2^254`, `2^-115`, where applying the power after the decay read 0. Differentiable at every order:
    `f`'s derivative its exponent's."""
    exponent = jnp.broadcast_to(exponent, jnp.shape(z))
    finite = jnp.isfinite(exponent)
    decay = jnp.where(finite, exponent, 0.0)
    whole = jax.lax.stop_gradient(jnp.floor(decay * (1.0 / np.log(2.0))))
    # The fraction 0 past `2^16` whole powers (no fp32 value survives them), as it is within them; and the
    # powers together, decay's and `power`, held within `±400` after they meet, never apart (a decay of
    # -9744 beside a power of 13862 is `2^-195`, where either held alone read infinity).
    fraction = jnp.where(jnp.abs(whole) < 2.0 ** 16, ln2_remainder(decay, whole), 0.0)
    lifted = (z * 0.5) * accurate_exp(fraction)
    powers = jnp.clip(jnp.asarray(power, jnp.float32) + whole + 1, -400.0, 400.0).astype(jnp.int32)
    return jnp.where(finite, power2(lifted, powers), 0.0)


def _pair_bound(left: jax.Array, right: jax.Array) -> jax.Array:
    """The power of two each product `left · rightᵀ` (`[B, Hk, G, Q, D]` by a tile's `[B, T, Hk, D]`) lies
    below, int32 `[B, Hk, G, Q, T]` (`_NONE` for a pair of no nonzero term): its terms' exponents aligned,
    `max_d e(l_d) + e(r_d)` (`_exponent`), and the width's, so the bound is its own pair's whatever the
    rows' and keys' other entries: `r = [2^127, 2^-125]` against `(0, 2^127)` bounds its dot of 4 by 8,
    where mantissas at the row's power flushed `2^-125` and a floor bounded it by `2^131` (train 1444's
    review)."""
    rows = _exponent(left)[..., None, :]
    keys = jnp.moveaxis(_exponent(right), 1, 2)[:, :, None, None]
    largest = jnp.max(rows + keys, axis=-1, initial=2 * _ABSENT)
    width = (left.shape[-1] - 1).bit_length()
    return jnp.where(largest > _ABSENT // 2, largest + width, _NONE)


def _row_bounds(left: jax.Array, right: jax.Array, spans: KeySpans, tile: int) -> jax.Array:
    """The power of two each row's products `left · rightᵀ` (`[B, Hk, G, Q, D]` by `[B, Kp, Hk, D]`, whole
    tiles of `tile`) lie below over the keys it sees (`_pair_bound`), int32 `[B, Hk, G, Q]` (`_NONE` for
    none): a hidden key bounds nothing (its `2^127` took a visible pair's `2^-100 · 2^100` below fp32's
    least normal, train 1423's review). One power a row: a visible pair's dot more than fp32's range below
    the row's largest visible dot past `2^125`, which plain fp32 forms as infinity, is held to plain fp32's
    parity (Main's ruling of 2026-11-27, row 14033)."""
    def step(reached, index):
        start = index * tile
        seen = dense(spans, start + jnp.arange(tile))[:, None, None]
        bound = _pair_bound(left, jax.lax.dynamic_slice_in_dim(right, start, tile, axis=1))
        return jnp.maximum(reached, jnp.max(jnp.where(seen, bound, _NONE), axis=-1)), None

    return _tiled(step, jnp.full(left.shape[:-1], _NONE, jnp.int32), right.shape[1] // tile)[0]


def _tiled(step: Callable, start, count: int):
    """`jax.lax.scan(step, start, jnp.arange(count))` with its first step taken apart, so its carry is typed
    as a step makes it from what it reads (on an Explicit mesh their placement, under a checked `shard_map`
    their manual axes' variance), whichever operand is split: a value split alone, its query replicated,
    failed the scan whose carry was seeded from the query (train 1444's review). The steps' outputs
    stacked, None where a step gives none."""
    carry, first = step(start, 0)
    carry, rest = jax.lax.scan(step, carry, jnp.arange(1, count))
    # Each step's outputs fused into their stacking apart: XLA:CPU's instruction fusion of the first step's
    # outputs into the concatenation left an int32 sentinel's bits in a float output (a dK of -0 read NaN,
    # intermittently, row 14101).
    first, rest = jax.lax.optimization_barrier((first, rest))
    return carry, jax.tree.map(lambda one, more: jnp.concatenate([one[None], more]), first, rest)


def second_powers(bounds: jax.Array) -> jax.Array:
    """Each query row's power of two `2^p` `[B, Hk, G, Q]` int32 (its log2, at most 0) whose second query
    keeps every product with the second keys it sees within `2^HELD`, from their bounds (`_row_bounds`);
    0 for a row whose products cannot pass it."""
    return jnp.clip(HELD - bounds, -300, 0)


def _keyed_largest(value: jax.Array) -> jax.Array:
    """The power of two each key's value lies below, at least 1 (a zero's 1), int32 `[B, Hk, 1, 1, T]`:
    `value` `[B, T, Hk, Dv]` a tile's."""
    return jnp.maximum(1, jnp.moveaxis(jnp.max(_exponent(value), axis=-1), 1, -1))[:, :, None, None]


def moments(logits: jax.Array, seconds: jax.Array, value: jax.Array, powers: jax.Array) -> Moments:
    """A key tile's own moments (`Moments`) of each row `[B, Hk, G, Q]`: its `logits` and second products in
    the row's units `seconds` (`t̂`) `[B, Hk, G, Q, K]`, -inf and 0 where a query does not see a key, over
    `value` `[B, K, Hk, Dv]`, `powers` the rows' `2^p` (`second_powers`). The covariance's terms are
    `ũ (v / 2 - a / 2)`, `ũ = w (t̂ - c) 2^-G` decayed from the logits (`exp_scaled`), `G` the row's power
    that holds `Σ |ũ|` at most 1 against its seen values' largest, so no partial sum passes fp32's range
    where the covariance does not; the covariance then a mantissa and its power (`Moments.spread`).
    Differentiable at every order (the reference's, `reference`)."""
    tile = _centred_tile(logits, seconds)
    seen, below, total, over = tile.seen, tile.below, tile.total, tile.over
    weights = tile.masses / over

    # The values' mean in two tiers: the weights fp32 holds normally as they are, and those below `e^-80`
    # raised by their row's power first (decayed whole, `exp_scaled`) and the power taken off their sum,
    # so a weight of `e^-128` over a value of `2^127` keeps its product, `2^127 e^-128`, where the weight
    # alone underflowed; the raised weights' products with the row's seen values within `2^126`.
    tail = below < -80.0
    width = (logits.shape[-1] - 1).bit_length()
    valued = _keyed_largest(value)
    # Each pair's own: a tail weight's reach beside its own key's value, no other key's value lowering it.
    tail_pairs = jnp.where(seen & tail, _reach(below) + valued, _NONE)
    raised = jnp.clip(126 - width - jnp.max(tail_pairs, axis=-1), 0, 241)[..., None]
    lifted = jnp.where(tail, exp_scaled(jnp.broadcast_to(1.0 / over, below.shape), below, raised), 0.0)
    mean = (_weighed(jnp.where(tail, 0.0, weights), value)
            + _shifted(_weighed, lifted, value, -raised, 0, (0, 0)))
    hi, lo, apart = tile.hi, tile.lo, tile.apart
    # Each value channel taken to its own largest power first (`_channel_powers`), so its values and mean
    # are at most 1 and one channel's `2^127` takes no other's `2^-124` below fp32's least (train 1444's
    # review); the row's power then its largest pair `|w (t̂ - c)|` below `2^(G - log2 K)`, so `Σ |ũ| |v| / 2`
    # and `Σ |ũ| |a| / 2` within `2^126` in every channel.
    reach = jnp.where(seen & (apart != 0), _reach(below) + _exponent(apart), _NONE)
    channels, rows = _channel_powers(value, mean, reach)
    largest = jnp.max(jnp.where(reach > _NONE, reach + 1, _NONE), axis=-1)
    power = jnp.where(largest > _NONE, largest, 0) + width - 125
    # The call's units' power, `2^(G + 1) / 2^p`: a part below 1 (at most `2^-60` of it, `ũ` kept above
    # fp32's least normal for its values' products) taken into `ũ` before its products, so a derivative's
    # cotangent never meets `2^-179` before the `ũ` of `2^180` it scales (the reference's derivative at a
    # weight of `e^-128` against `2^126`); the rest the covariance's power (`Moments.spread_power`).
    units = power + 1 - powers
    before = jnp.maximum(jnp.minimum(units, 0), -60)
    taken = exp_scaled(apart / over, below, (before - power)[..., None])

    # A key no query sees weighs nothing in the covariance: its value 0, never one past its channel's power.
    # Every key some query sees is the covariance's, its derivative's too (a key whose term is 0 at the
    # centre still moves it); a value the channel's power takes past fp32 held at its largest, its term
    # past fp32's span below the others' (`_channel_powers`), so no 0 meets an infinity.
    seen_keys = jnp.moveaxis(jnp.any(seen, axis=(2, 3)), 1, 2)[..., None]
    largest = np.float32(np.finfo(np.float32).max)
    halves = (jnp.clip(power2(jnp.where(seen_keys, value, 0.0), -channels), -largest, largest) * 0.5,
              power2(mean, -rows) * 0.5)
    # Its power from the covariance at the lifted units, the covariance then taken to it whole (`_shifted`).
    lift = (units - before)[..., None]
    spread_power = _spread_power(jax.lax.stop_gradient(_covariance(taken, halves)), lift + rows)
    spread = _shifted(_covariance, taken, halves, lift, rows - spread_power, ((channels, rows), rows))
    return Moments(jnp.max(logits, axis=-1), total, mean, (hi, lo), spread, spread_power)


class _Tile(NamedTuple):
    """A key tile's softmax and centre as its moments and the dense tangent both read them
    (`_centred_tile`): where each query sees each key, the logits below the row's maximum, their
    exponentials (`masses`), the denominator
    (and it held at 1 where none), the second products' centre as a pair `(hi, lo)` and each second
    product's distance from it."""

    seen: jax.Array
    below: jax.Array
    masses: jax.Array
    total: jax.Array
    over: jax.Array
    hi: jax.Array
    lo: jax.Array
    apart: jax.Array


def _centred_tile(logits: jax.Array, seconds: jax.Array) -> _Tile:
    """A tile's `_Tile` from its logits and second products `[B, Hk, G, Q, K]`: the centre's terms each a
    second product decayed with its weight before either rounds (`exp_scaled`), so a tail's `2^200 e^-128`
    is kept where its weight alone underflowed (train 1444's review)."""
    seen = logits > -jnp.inf
    m = jax.lax.stop_gradient(jnp.max(logits, axis=-1))
    below = logits - jnp.where(jnp.isfinite(m), m, 0.0)[..., None]
    masses = jnp.exp(below)
    total = jnp.sum(masses, axis=-1)
    over = jnp.where(total > 0, total, 1.0)[..., None]
    hi = jax.lax.stop_gradient(jnp.sum(exp_scaled(seconds / over, below, 0), axis=-1))
    lo = jnp.sum(exp_scaled((seconds - hi[..., None]) / over, below, 0), axis=-1)
    apart = jnp.where(seen, (seconds - hi[..., None]) - lo[..., None], 0.0)
    return _Tile(seen, below, masses, total, over, hi, lo, apart)


def _weighed(weights: jax.Array, value: jax.Array) -> jax.Array:
    """A row's weights `[B, Hk, G, Q, K]` times a tile's values `[B, K, Hk, Dv]`, summed over the keys. A
    module function, as `_shifted`'s product: a surrogate's held rule is keyed by its static arguments, which
    a closure made each trace named apart only by an identity its garbage hands on (a mean's program
    replayed for a covariance, train 1444's re-review's flake)."""
    return float32_einsum("bhgqk,bkhd->bhgqd", weights, value, precision=REFERENCE_PRECISION,
                          preferred_element_type=value.dtype)


def _covariance(terms: jax.Array, centred: tuple[jax.Array, jax.Array]) -> jax.Array:
    """`Σ ũ (v / 2 - a / 2)`, bilinear in the terms and the halved values and mean `centred` (`moments`)."""
    values, means = centred
    return _weighed(terms, values) - jnp.sum(terms, axis=-1)[..., None] * means


def _channel_powers(value: jax.Array, mean: jax.Array, reach: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Each value channel's power of two over the tile's keys whose covariance terms it can hold (`reach`
    `[B, Hk, G, Q, K]`, each pair's `|w (t̂ - c)|` power, `_NONE` where none; `value` `[B, K, Hk, Dv]`): a
    key's term in a channel `|w (t̂ - c)| |v|` within fp32's span (`_SPAN`) of the channel's largest, a key
    whose every term lies below it bounding nothing (a key at `e^-1000` with a value of `2^127` erases no live
    `2^-120`, train 1444's review; one row's largest beside another's, row 14033's ruling); the power those
    values and the rows' means (`mean` `[B, Hk, G, Q, Dv]`, of a row with a term) lie below (0 for a channel
    of none), as the values take it `[B, 1, Hk, Dv]` and as a row's mean does `[B, Hk, 1, 1, Dv]`."""
    keyed = jnp.moveaxis(jnp.max(reach, axis=(2, 3), initial=_NONE), 1, 2)[..., None]
    exponents = _exponent(value)
    pairs = jnp.where((keyed > _NONE) & (exponents > _ABSENT), keyed + exponents, _NONE)
    visible = pairs >= jnp.max(pairs, axis=1, keepdims=True, initial=_NONE) - _SPAN
    found = jnp.max(jnp.where(visible, exponents, _ABSENT), axis=1, keepdims=True, initial=_ABSENT)
    termed = (jnp.max(reach, axis=-1, initial=_NONE) > _NONE)[..., None]
    centred = jnp.max(jnp.where(termed, _exponent(mean), _ABSENT), axis=(2, 3), initial=_ABSENT)
    found = jnp.maximum(found, centred[:, None])
    found = jnp.where(found > _ABSENT, found, 0)
    return found, jnp.moveaxis(found, 2, 1)[:, :, None]


def _spread_power(covariance: jax.Array, power) -> jax.Array:
    """`Moments.spread_power` of a covariance `covariance · 2^power` (`[..., Dv]`, `power` int32 broadcast to
    it): each channel's power of two where that passes 1, else 0 (a zero's too)."""
    return jnp.where(covariance != 0, jnp.maximum(_exponent(covariance) + power, 0), 0)


def _shifted_value(product: Callable, rows: jax.Array, by, lift, scale, restored) -> jax.Array:
    """`product(rows, by) 2^(lift + scale)`, its product formed before the power (`_shifted`)."""
    del restored
    return power2(product(rows, by), lift + scale)


def _shifted_tangent(product: Callable, primals, tangents, value) -> jax.Array:
    """`_shifted`'s tangent: each direction, the rows' and `by`'s, met with the rows taken toward their own
    size first, by as much of the row's `lift` as keeps their largest within
    `2^125` (all of it where it is negative), the rest of the power after: a lift of `2^60` taken off before
    a direction of `2^125` meets them, and a positive one put back before a direction of `2^-124` does (its
    products `2^-165` otherwise, train 1444's review). Each direction meets `by` with the channels' powers
    `restored` put back (`(by's, the output's)`: `by` taken down by them before the value's product), so a
    pullback takes no channel's power on a cotangent before `by` meets it."""
    del value
    rows, by, lift, scale, (by_powers, out_power) = primals
    drows, dby, _, _, _ = tangents
    largest = jnp.max(_exponent(jax.lax.stop_gradient(rows)), axis=-1, keepdims=True)
    take = jnp.minimum(lift, 125 - largest)
    whole = jax.tree.map(power2, by, by_powers)
    moved = jax.tree.map(power2, dby, by_powers)
    return (power2(product(power2(drows, take), whole), lift - take + scale - out_power)
            + power2(product(power2(rows, take), moved), lift - take + scale - out_power))


_shifted = contracted("test_parallax_holds_every_order")(
    surrogate(_shifted_value, _shifted_tangent, nondiff_argnums=(0,)))
"""`product(rows, by) 2^(lift + scale)` (`_shifted_value`), `product` bilinear in a row's terms `rows`
`[..., K]` lifted for their products and what they meet, `by` (an array or a tuple of them), `lift` the
rows' integral power back `[..., 1]`, `scale` each channel's own `[..., Dv]` (or 0) and `restored` the
channels' powers `by` was taken down by (`_shifted_tangent`): its value the
product's before the power, so a lifted term keeps its product with a small value and terms that cancel
do so before the power; a direction along `by` meets the terms at their
own size (`_shifted_tangent`): equal logits, second products `[1, -1]`, values 0 along `[2^126, -2^126]`
read the covariance's tangent `2^126`, where `2^60`-lifted terms met `2^125` before their `2^-60` and
overflowed (train 1444's review). Its tails' tangents so read as plain fp32's (Main's ruling of
2026-11-27, row 14007)."""


def merged(held: Moments, tile: Moments, powers: jax.Array) -> Moments:
    """Two parts' moments of the same rows merged (Chan's law, module docstring): each part's share of the
    merged denominator, as its fraction and power, decayed by its maximum's distance below the merged one
    and met by every moment it weighs before either rounds (`exp_scaled`); the covariances and the cross
    term `s_A s_B (c_A - c_B)(a_A - a_B)` summed at their common power and carried with it, so a part's
    covariance past fp32's range comes back by its share (`2^254` by `e^-128`); the centre about the larger
    share's (`_merged_centre`); the maximum, denominator and mean the algebra's law's (`means.merged_state`);
    a part of no key, or of no covariance, weighs nothing in the covariance's power."""
    # The softmax's state (maximum, denominator, mean) by the algebra's law of finished means
    # (`means.merged_state`), the anchor its total lands at the merged maximum.
    mean, total, m = means.merged_state(jnp.stack([held.mean, tile.mean]),
                                        jnp.stack([held.total, tile.total]), jnp.stack([held.m, tile.m]),
                                        Units(), 0)
    level = jnp.where(jnp.isfinite(m), m, 0.0)

    def below(part: Moments) -> jax.Array:
        return jnp.where(part.total > 0, part.m - level, 0.0)

    first, second = below(held), below(tile)
    over = jnp.where(total > 0, total, 1.0)
    # Each part's share as its fraction and power of two (a share past 1 where its part's maximum lies
    # below the merged one: 128 at `e^-128`), applied with its decay after the fraction meets the moment.
    (share, own), (other, theirs) = jnp.frexp(held.total / over), jnp.frexp(tile.total / over)

    def column(x):
        return x[..., None]

    (hi, lo), (fraction, exponent) = _merged_centre(held, tile, (share, own, first), (other, theirs, second))
    apart = held.mean * 0.5 - tile.mean * 0.5
    # The three terms' powers, each its decay's reach and its mantissa's bound: the largest their common
    # power, each term taken to it with its decay at once.
    cross_power = own + theirs + exponent + 1 - powers
    # Each channel's own (`Moments.spread`).
    estimates = (jnp.where(_spread_held(held), column(_reach(first) + own) + held.spread_power, _NONE),
                 jnp.where(_spread_held(tile), column(_reach(second) + theirs) + tile.spread_power, _NONE),
                 jnp.where(column((held.total > 0) & (tile.total > 0) & (fraction != 0)),
                           column(_reach(first + second) + cross_power) + _exponent(apart), _NONE))
    common = functools.reduce(jnp.maximum, estimates)
    common = jnp.where(common > _NONE, common, 0)
    summed = (exp_scaled(column(share) * held.spread, column(first), column(own) + held.spread_power - common)
              + exp_scaled(column(other) * tile.spread, column(second),
                           column(theirs) + tile.spread_power - common)
              + exp_scaled(column(share * other * fraction) * apart, column(first + second),
                           column(cross_power) - common))
    spread_power = _spread_power(summed, common)
    return Moments(m, total, mean, (hi, lo), power2(summed, common - spread_power), spread_power)


def _spread_held(part: Moments) -> jax.Array:
    """Whether each channel of a part's rows holds a covariance (a zero's power bounds no merge,
    `Moments.spread`)."""
    return (part.total > 0)[..., None] & (part.spread != 0)


def _merged_centre(held: Moments, tile: Moments, held_share: tuple, tile_share: tuple):
    """Two parts' centres merged (`merged`), each share `(fraction, power, decay)` of the merged
    denominator: `c = c_L + s_S (c_S - c_L)` about the larger share's centre `c_L`, the smaller share `s_S`
    its own (never `1 - s_L` rounded: logits `[0, -20]` over second products `[0, 2^100]` read `c` 0 where
    it is `2.6e21`, train 1444's review), each word's difference weighed apart and the sum's rounding kept,
    so a part of no key leaves the other's pair exactly; and `c_A - c_B` as its fraction and power of two."""
    def scaled(fraction, power, decay):
        return exp_scaled(fraction, decay, power)

    smaller = scaled(*held_share) <= scaled(*tile_share)

    def pick(held_part, tile_part):
        return jnp.where(smaller, held_part, tile_part)

    small_hi, small_lo = pick(held.centre[0], tile.centre[0]), pick(held.centre[1], tile.centre[1])
    large_hi, large_lo = pick(tile.centre[0], held.centre[0]), pick(tile.centre[1], held.centre[1])
    fraction, power, decay = (pick(mine, theirs) for mine, theirs in zip(held_share, tile_share, strict=True))
    high, rounded = two_sum(small_hi, -large_hi)
    hi, dropped = two_sum(large_hi, exp_scaled(fraction * high, decay, power))
    low = rounded + (small_lo - large_lo)
    lo = dropped + large_lo + exp_scaled(fraction * low, decay, power)
    difference, exponent = jnp.frexp(high + low)
    return (hi, lo), (jnp.where(smaller, difference, -difference), exponent)


class _Call(NamedTuple):
    """A call's operands as its walks read them, fp32: the prescaled query `[B, Hk, G, Q, D]` and the second
    query in its rows' units `[B, Hk, G, Q, D2]`, the keys, second keys and values `[B, Kp, Hk, ...]`
    padded to whole tiles (zeros, unseen), the rows' powers `[B, Hk, G, Q]` (`second_powers`), the scale
    left after the query's power, and the spans."""

    query: jax.Array
    second: jax.Array
    key: jax.Array
    second_key: jax.Array
    value: jax.Array
    powers: jax.Array
    scale: float
    spans: KeySpans


def _wide(*operands) -> jnp.dtype:
    """The dtype a call's walks compute in: its operands' with fp32 at least (float64's own for a float64
    truth)."""
    return jnp.result_type(*(x.dtype for x in operands), jnp.float32)


def _grouped(x: jax.Array, kv_heads: int) -> jax.Array:
    """`[B, Q, H, ...]` as `[B, Hk, G, Q, ...]`, a key head's query heads together."""
    batch, queries, heads = x.shape[:3]
    return jnp.moveaxis(x.reshape(batch, queries, kv_heads, heads // kv_heads, *x.shape[3:]), 1, 3)


def _ungrouped(x: jax.Array) -> jax.Array:
    """`[B, Hk, G, Q, ...]` back to `[B, Q, H, ...]`."""
    batch, kv_heads, group, queries = x.shape[:4]
    return jnp.moveaxis(x, 3, 1).reshape(batch, queries, kv_heads * group, *x.shape[4:])


def _padded(x: jax.Array, tile: int, wide) -> jax.Array:
    """Keys `[B, K, Hk, ...]` padded with zeros to whole tiles of `tile`, at least one."""
    pad = -x.shape[1] % tile if x.shape[1] else tile
    return jnp.pad(x.astype(wide), ((0, 0), (0, pad), (0, 0), (0, 0)))


def _called(query, second, key, second_key, value, spans: KeySpans, scale: float, tile: int) -> _Call:
    """A call's operands as its walks read them (`_Call`), its keys padded to whole tiles of `tile`."""
    kv_heads = key.shape[2]
    wide = _wide(query, value)
    prescaled_query, left, _ = prescaled(query, scale)
    raw = _grouped(second.astype(wide), kv_heads)
    second_key = _padded(second_key, tile, wide)
    query, key = _grouped(prescaled_query.astype(wide), kv_heads), _padded(key, tile, wide)
    powers = second_powers(_row_bounds(raw, second_key, spans, tile))
    return _Call(query, power2(raw, powers[..., None]), key, second_key, _padded(value, tile, wide), powers,
                 left, spans)


def _tile(call: _Call, start, width: int) -> tuple[jax.Array, jax.Array, jax.Array, tuple]:
    """Keys `start` to `start + width` of a call: their logits and second products in the rows' units
    `[B, Hk, G, Q, width]`, -inf and 0 where unseen, their values, and their keys and second keys."""
    def at(x):
        return jax.lax.dynamic_slice_in_dim(x, start, width, axis=1)

    key, second_key, value = at(call.key), at(call.second_key), at(call.value)

    def products(left, right):
        return float32_einsum("bhgqd,bkhd->bhgqk", left, right, precision=REFERENCE_PRECISION,
                              preferred_element_type=left.dtype)

    seen = dense(call.spans, start + jnp.arange(width))[:, None, None]
    # A scale of 0 gives logits of 0, never an overflowing product times it (`inf · 0`, train 1444's review).
    scaled = (jnp.zeros((*call.query.shape[:-1], width), call.query.dtype) if call.scale == 0
              else products(call.query, key) * call.scale)
    logits = jnp.where(seen, scaled, -jnp.inf)
    seconds = jnp.where(seen, products(call.second, second_key), 0.0)
    return logits, seconds, value, (key, second_key)


def walk(query, second, key, second_key, value, spans: KeySpans, *, scale: float,
         tile: int = KEY_TILE) -> tuple[jax.Array, Moments]:
    """Parallax's forward (module docstring): its output `[B, Q, H, Dv]` in the query's dtype and each
    row's moments (`Moments`, the derivative's residuals), a key tile of `tile` at a time over every query,
    each tile's moments merged into the rows' (`merged`), the first tile's the walk's start (its carry typed
    as its steps make it, `_tiled`)."""
    call = _called(query, second, key, second_key, value, spans, scale, tile)

    def own(index) -> Moments:
        logits, seconds, values, _ = _tile(call, index * tile, tile)
        return moments(logits, seconds, values, call.powers)

    def step(held: Moments, index) -> tuple[Moments, None]:
        return merged(held, own(index), call.powers), None

    found, _ = jax.lax.scan(step, own(0), jnp.arange(1, call.key.shape[1] // tile))
    return _output(found, query.dtype), found


def _output(found: Moments, dtype) -> jax.Array:
    """`a - C` `[B, Q, H, Dv]` in `dtype`, the two subtracted at their common power and the difference
    taken back by it, so a covariance past fp32 whose difference with the mean fp32 holds reads that
    difference (`a = 2^126`, `C = 2^128`: `-3 · 2^126`, train 1423's review of 72b1ef355)."""
    largest = jnp.maximum(found.spread_power + 1, _exponent(found.mean))
    common = jnp.maximum(0, largest - 126)
    difference = power2(found.mean, -common) - power2(found.spread, found.spread_power - common)
    return _ungrouped(power2(difference, common)).astype(dtype)


def _landing(estimate: jax.Array, operand: jax.Array, axis: int | tuple[int, int]) -> jax.Array:
    """The power of two `2^L` a contraction's terms land at before their products with `operand`'s rows
    (`[B, Hk, G, Q, D]` over keys, or `[B, T, Hk, D]` over queries): `estimate` `[B, Hk, G, Q, T]` each
    term's power of two (`_NONE` none), the largest term's product with its operand's largest entry
    `2^(HELD - log2 n)`, `n` the terms summed; int32, over `axis` of the estimates (-1 a row's keys, (2,
    3) a key's queries)."""
    largest = jnp.max(_exponent(operand), axis=-1)
    if axis == -1:
        paired = estimate + jnp.moveaxis(largest, 1, -1)[:, :, None, None]
        summed = estimate.shape[-1]
    else:
        paired = estimate + largest[..., None]
        summed = estimate.shape[2] * estimate.shape[3]
    live = estimate > _NONE
    # Over no term (a key's of no query, `Q = 0`) `_NONE`.
    reached = jnp.max(jnp.where(live, paired, _NONE), axis=axis, initial=_NONE)
    # No landed term past `2^126` itself, whatever its operand (a zero operand bounds nothing: its terms'
    # `2^400` met it as `inf · 0`).
    largest_term = jnp.max(jnp.where(live, estimate, _NONE), axis=axis, initial=_NONE)
    return jnp.clip(jnp.minimum(HELD - (summed - 1).bit_length() - reached, 126 - largest_term), -250, 250)


def _logit_coefficient(delta, delta_power, fraction, tau_power, against, against_power):
    """The logits' coefficient `δ - δ τ + g · C` (before the weight) as a mantissa and its power of two:
    `g · C - δ τ` at their common power first, so the two cancel exactly before `δ` meets their difference
    at the power of the larger (both landed first read `inf - inf`, and `δ` at their common power flushed:
    equal logits, second products `∓2^127` under a cotangent of 16 read `[-4, 4]` as 0, train 1423's
    review)."""
    moved_tau = delta * fraction
    pair = functools.reduce(jnp.maximum, (delta_power + tau_power + _exponent(moved_tau),
                                          against_power + _exponent(against)))
    pair = jnp.where(pair > _ABSENT // 2, pair, 0)
    difference = (power2(jnp.broadcast_to(against, delta.shape), against_power - pair)
                  - power2(moved_tau, delta_power + tau_power - pair))
    common = jnp.maximum(pair + _exponent(difference), delta_power + _exponent(delta))
    common = jnp.where(common > _ABSENT // 2, common, 0)
    return power2(delta, delta_power - common) + power2(difference, pair - common), common


def _values_coefficient(fraction, tau_power):
    """The values' coefficient `1 - τ` (before the weight) as a mantissa and its power of two."""
    power = jnp.maximum(1, tau_power)
    return power2(jnp.ones_like(fraction), -power) - power2(fraction, tau_power - power), power


def _contracted(spec: str, left: jax.Array, right: jax.Array) -> jax.Array:
    """A contraction of the derivative's landed terms with their operand, at the reference's precision."""
    return float32_einsum(spec, left, right, precision=REFERENCE_PRECISION, preferred_element_type=left.dtype)


def _on_rows(summed, power, coefficient, at, below, estimated, by):
    """A row's running `Σ_k term · by_k` (`dq`'s, `dr`'s): this tile's terms landed at the row's power
    (`_landing`, the least of its running one and this tile's), the carried sum taken to it."""
    fresh = jnp.minimum(power, _landing(estimated, by, -1))
    part = _contracted("bhgqk,bkhd->bhgqd", exp_scaled(coefficient, below, at + fresh[..., None]), by)
    return power2(summed, (fresh - power)[..., None]) + part, fresh


def _on_keys(coefficient, at, below, estimated, by, scale: float = 1.0):
    """A key's `Σ_q term · by_q` (`dk`'s, `dk2`'s, `dV`'s): its terms landed at the key's own power
    (`_landing`), their sum times `scale`'s mantissa while landed, then taken back by that power with
    `scale`'s own power of two (`_scaled`)."""
    power = _landing(estimated, by, (2, 3))
    terms = exp_scaled(coefficient, below, at + power[:, :, None, None])
    return _scaled(_contracted("bhgqk,bhgqd->bkhd", terms, by), scale,
                   -jnp.moveaxis(power, 1, -1)[..., None])


def _scaled(landed: jax.Array, scale: float, power) -> jax.Array:
    """`landed · scale · 2^power`: `scale`'s mantissa met while `landed` holds within fp32, its power of two
    taken with `power` at once, so an explicit scale of `2^100` never overflows a landed sum near `2^120`
    first (train 1423's review of 72b1ef355)."""
    mantissa, exponent = math.frexp(scale)
    return power2(landed * mantissa, power + exponent)


def _summed_rows(terms: Callable, seed: jax.Array, tiles: int) -> tuple[jax.Array, jax.Array]:
    """Each row's `Σ_k z e^below 2^power` over its keys, `terms(index)` a tile's `(z, below, power)`
    `[B, Hk, G, Q, T]`, as a sum and its power of two `(s, P)`, `s 2^P` the whole: each term decayed whole
    (`exp_scaled`) at the row's running power, its largest term near `2^100`, the sum taken to a raised
    power exactly, so a term no factor of which fp32 holds alone keeps its value; `seed` the rows' zeros,
    its walk's carry typed as its steps make it (`_tiled`)."""
    def step(summed, index):
        total, reached = summed
        coefficient, below, power = terms(index)
        bound = jnp.where((coefficient != 0) & jnp.isfinite(below),
                          _reach(below) + power + _exponent(coefficient), _NONE)
        widest = jnp.maximum(reached, jnp.max(bound, axis=-1))
        lifted, before = jnp.clip(100 - widest, -600, 600), jnp.clip(100 - reached, -600, 600)
        part = jnp.sum(exp_scaled(coefficient, below, power + lifted[..., None]), axis=-1)
        return (power2(total, lifted - before) + part, widest), None

    (total, reached), _ = _tiled(step, (seed, jnp.full(seed.shape, _NONE, jnp.int32)), tiles)
    return total, -jnp.clip(100 - reached, -600, 600)


def derivative(query, second, key, second_key, value, spans: KeySpans, held: Moments, dout: jax.Array, *,
               scale: float, tile: int = KEY_TILE) -> tuple[jax.Array, ...]:
    """Parallax's derivative (module docstring) from the forward's moments `held` and the output's cotangent
    `dout`: the cotangents of the query, the second query, the keys, the second keys and the values, a key
    tile at a time over every query. Each term's coefficient formed with its power of two (`δ`'s, `δ τ`'s
    and `g · C`'s summed at their common power, so terms that cancel do before either lands), then landed
    with its decay at its contraction's power (`_landing`: a row's for `dq` and `dr`, a key's for `dk`,
    `dk2` and `dV`), multiplied by its operand and the scale, and taken back by that power."""
    call = _called(query, second, key, second_key, value, spans, scale, tile)
    kv_heads = key.shape[2]
    wide = call.query.dtype
    g = _grouped(dout.astype(wide), kv_heads)
    over = jnp.where(held.total > 0, held.total, 1.0)
    level = jnp.where(jnp.isfinite(held.m), held.m, 0.0)
    queries, seconds_query = _grouped(query.astype(wide), kv_heads), _grouped(second.astype(wide), kv_heads)
    raw_second_key = _padded(second_key, tile, wide)
    # The cotangent's products with the values held at its row's power over the pairs it sees
    # (`_row_bounds`): no operand scaled alone (a cotangent `[2^127, 2^-100]` over values `(0, 2^100)` keeps
    # `g · v = 1`, train 1423's review).
    raise_gv = jnp.maximum(0, _row_bounds(g, call.value, spans, tile) - HELD)
    taken_g = power2(g, -raise_gv[..., None])

    def row(x):
        return x[..., None]

    def products(spec, left, right):
        return float32_einsum(spec, left, right, precision=REFERENCE_PRECISION, preferred_element_type=wide)

    def moving(index):
        # A tile's logits below the rows' maximum, its `g · v` at the held powers, and its keys and values.
        logits, seconds, values, keys = _tile(call, index * tile, tile)
        return logits, seconds, keys, logits - row(level), products("bhgqd,bkhd->bhgqk", taken_g, values)

    def gv(index):
        _, _, _, below, moved = moving(index)
        return moved / row(over), below, jnp.zeros_like(moved, dtype=jnp.int32)

    tiles = call.key.shape[1] // tile
    # `g · a` as the walk's own weights take each `g · v` (`Σ w g · v`, each decayed whole), so a tail
    # weight's `g · v` is kept where the held units flushed it (`_summed_rows`).
    ga, ga_power = _summed_rows(gv, jnp.zeros(over.shape, over.dtype), tiles)

    def paired(logits, seconds, moved):
        # A tile's `τ` as its fraction and power, and `δ = g · v - g · a` at each pair's own power `2^E`
        # above the held units.
        seen = logits > -jnp.inf
        apart = jnp.where(seen, (seconds - row(held.centre[0])) - row(held.centre[1]), 0.0)
        fraction, exponent = jnp.frexp(apart)
        spot = jnp.maximum(_exponent(moved), row(_exponent(ga) + ga_power))
        spot = jnp.where(spot > _ABSENT // 2, spot, 0)
        delta = power2(moved, -spot) - power2(row(ga), row(ga_power) - spot)
        return seen, fraction, exponent - row(call.powers), delta, spot + row(raise_gv)

    def delta_tau(index):
        logits, seconds, _, below, moved = moving(index)
        _, fraction, tau_power, delta, delta_power = paired(logits, seconds, moved)
        return delta * fraction / row(over), below, delta_power + tau_power

    # `g · C = Σ w δ τ`, summed from the very terms the logits' coefficients take, where `g · C` from the
    # forward's covariance rounded apart from them and a row's coefficients summed short of 0.
    against, against_power = _summed_rows(delta_tau, jnp.zeros(over.shape, over.dtype), tiles)

    def step(sums, index):
        (dq, dq_power), (dr, dr_power) = sums
        logits, seconds, (keys, _), below, moved = moving(index)
        raw_keys = jax.lax.dynamic_slice_in_dim(raw_second_key, index * tile, tile, axis=1)
        seen, fraction, tau_power, delta, delta_power = paired(logits, seconds, moved)
        logit, common = _logit_coefficient(delta, delta_power, fraction, tau_power, row(against),
                                           row(against_power))
        # A scale of 0 moves no logit: no query's or key's cotangent, never an overflowing coefficient
        # times 0 (train 1444's review).
        logit = jnp.zeros_like(logit) if scale == 0 else logit / row(over)
        values_coefficient, values_power = _values_coefficient(fraction, tau_power)
        values_coefficient = values_coefficient / row(over)
        second_coefficient = -delta / row(over)
        reach = jnp.where(seen, _reach(below), _NONE)
        terms = ((logit, common), (second_coefficient, delta_power), (values_coefficient, values_power))
        (logit_estimate, second_estimate, values_estimate) = (
            jnp.where(seen & (coefficient != 0), reach + power + _exponent(coefficient), _NONE)
            for coefficient, power in terms)
        # Over the row's keys: `dq` and `dr`, each carried at its row's running power.
        on_rows = (_on_rows(dq, dq_power, logit, common, below, logit_estimate, keys),
                   _on_rows(dr, dr_power, second_coefficient, delta_power, below, second_estimate, raw_keys))
        # Over the key's queries: `dk`, `dk2` and `dV`, each at its key's own power.
        on_keys = (_on_keys(logit, common, below, logit_estimate, queries, scale),
                   _on_keys(second_coefficient, delta_power, below, second_estimate, seconds_query),
                   _on_keys(values_coefficient, values_power, below, values_estimate, g))
        return on_rows, on_keys

    def carried(x):
        return jnp.zeros(x.shape, x.dtype), jnp.full(x.shape[:-1], 250, jnp.int32)

    (dq, dr), per_tile = _tiled(step, (carried(call.query), carried(call.second)), tiles)

    def keyed(x):
        # `[tiles, B, tile, Hk, D]` back to `[B, K, Hk, D]`.
        return jnp.moveaxis(x, 0, 1).reshape(x.shape[1], -1, *x.shape[3:])[:, :key.shape[1]]

    dk, dk2, dv = map(keyed, per_tile)
    return (_ungrouped(_scaled(dq[0], scale, -dq[1][..., None])),
            _ungrouped(power2(dr[0], -dr[1][..., None])), dk, dk2, dv)


def attend(query: jax.Array, second: jax.Array, key: jax.Array, second_key: jax.Array, value: jax.Array,
           spans: KeySpans, *, scale: float | None = None) -> jax.Array:
    """Parallax attention `[B, Q, H, Dv]` (module docstring) of `query` `[B, Q, H, D]` by `key`
    `[B, K, Hk, D]` over `value` `[B, K, Hk, Dv]` within `spans`, its second products `second · second_keyᵀ`
    (`[B, Q, H, D2]`, `[B, K, Hk, D2]`), unscaled: its forward and first derivative a key tile at a time
    (`walk`, `derivative`), every higher order through `reference` (`generation.linear_tangent`). `scale` is
    a Python float, `D ** -0.5` by default."""
    scale = logit_scale(query.shape[-1], scale)

    def primal(query, second, key, second_key, value, spans):
        return walk(query, second, key, second_key, value, spans, scale=scale)[0]

    def forward(query, second, key, second_key, value, spans):
        return walk(query, second, key, second_key, value, spans, scale=scale)

    def backward(saved, operands, dout):
        query, second, key, second_key, value, spans = operands
        grads = derivative(query, second, key, second_key, value, spans, saved, dout, scale=scale)
        primals = (query, second, key, second_key, value)
        return (*(grad.astype(primal.dtype) for grad, primal in zip(grads, primals, strict=True)),
                zero_from_primal(spans))

    def portable(query, second, key, second_key, value, spans):
        return reference(query, second, key, second_key, value, spans, scale=scale)

    rule = contracted("test_parallax_holds_every_order")(
        generation.linear_tangent(forward, backward, portable, kind=KernelGlue(), primal=primal))
    return rule(query, second, key, second_key, value, spans)


def reference(query: jax.Array, second: jax.Array, key: jax.Array, second_key: jax.Array, value: jax.Array,
              spans: KeySpans, *, scale: float | None = None) -> jax.Array:
    """Parallax (`attend`) in XLA ops over every key at once: one tile's moments (`moments`) of the whole
    call, differentiable at every order, the output in the query's dtype; a row that sees no key 0. Its
    derivatives are autodiff's of the moments: a lifted term meets a direction or cotangent at its own size,
    its power taken first (`_shifted`), and its `g · v` forms as the plain formula's does, so at a softmax
    tail under a large cotangent (`e^-128` under `2^127`) a derivative reads 0 or overflows where float64's
    is a normal fp32, as the plain fp32 formula's does, every order finite within its contract
    (`test_parallax_holds_every_order`; Main's ruling of 2026-11-27, row 14007); the walk's first derivative
    (`derivative`) holds them."""
    scale = logit_scale(query.shape[-1], scale)
    keys = max(key.shape[1], 1)
    call = _called(query, second, key, second_key, value, spans, scale, keys)
    logits, seconds, values, _ = _tile(call, 0, keys)
    return _dense(jnp.dtype(query.dtype).name, logits, seconds, values, call.powers)


def _dense_value(dtype: str, logits, seconds, values, powers) -> jax.Array:
    """The dense call's output from its one tile (`reference`): its moments' `a - C` (`_output`)."""
    return _output(moments(logits, seconds, values, powers), dtype)


def _dense_tangent(dtype: str, primals, tangents, value) -> jax.Array:
    """`_dense`'s tangent: the logits' direction the moments' own (their values held), the second
    products' `Σ w (a - v) dt`, the values `Σ w (1 - τ) dv` (`out = Σ w (1 - τ) v`, `Σ w τ = 0`),
    so a mean's and a covariance's tangents meet in it before either rounds: second products `[8, 0]`
    over values 0 along `[2^127, 0]` read plain fp32's `-3 · 2^126`, where the covariance's `2^128` landed
    apart and overflowed (train 1444's review)."""
    del value
    logits, seconds, values, powers = primals
    dlogits, dseconds, dvalues, _ = tangents

    def held(logits):
        return _dense_value(dtype, logits, seconds, values, powers)

    _, moved = jax.jvp(held, (logits,), (dlogits,))
    # The values' coefficients `w (1 - τ)`, each pair's product decayed with its weight (`exp_scaled`), held
    # below `2^126` by their row's power `2^E` where they pass it (none where they do not, so a small
    # direction meets them at their size), their contraction with the direction taken back by it after: a
    # coefficient past fp32 (`w τ` of `2^138`) meets a small direction, or its pullback a small cotangent,
    # before it would overflow.
    tile = _centred_tile(logits, seconds)
    seen, below, over, apart = tile.seen, tile.below, tile.over, tile.apart
    rows = powers[..., None]
    reach = jnp.where(seen & (apart != 0), _reach(below) + _exponent(apart) - rows, _NONE)
    power = jnp.maximum(jnp.max(reach, axis=-1, keepdims=True) - 126, 0)
    coefficients = (exp_scaled(jnp.broadcast_to(1.0 / over, below.shape), below, -power)
                    - exp_scaled(apart / over, below, -rows - power))
    valued = power2(float32_einsum("bhgqk,bkhd->bhgqd", coefficients, dvalues, precision=REFERENCE_PRECISION,
                                   preferred_element_type=dvalues.dtype), power)
    # The second products' direction the output's own contraction `Σ w (a - v) dt` over the values as they
    # are, as plain fp32 forms it, never the moments' values at their channel's power (a value of `2^124`
    # its power took past fp32 and held at its largest moved the tangent below `2^120`, train 1444's
    # review).
    # The direction centred before it meets the values, `w (Σ w dt - dt)`, as plain fp32 forms it: one
    # contraction, never a mean's and a weighted sum's products apart (`2^128` before `3 · 2^126` met it).
    weights = jnp.where(seen, tile.masses / over, 0.0)
    centred = weights * (jnp.sum(weights * dseconds, axis=-1, keepdims=True) - dseconds)
    secondly = _weighed(centred, values)
    return moved + _ungrouped(valued + secondly).astype(dtype)


_dense = contracted("test_parallax_holds_every_order")(
    surrogate(_dense_value, _dense_tangent, nondiff_argnums=(0,)))
"""A dense call's output from its one tile (`_dense_value`), its values' tangent the plain formula's one
contraction (`_dense_tangent`)."""

