"""Check 2's native step witness for the recurrence template (row 2084, link 3): each record its launches
store between them, held inside its launch step's enclosure on the native records that step reads.

The template's launches already store a chunk's inverse less its unit diagonal `a`, its weighted keys `w`
(the solve), the memory it enters `h`, its corrected values `c` (the states), and its output `o` (the
output); `mainloop.kept_forward` returns them (`recorded`). For the gated delta rule with a decay a head, each
chunk and head (`chunk_step`):

1. `a` from the keys, write gates and log decays: the system `tril(beta (k k^T) exp2(g_s - g_t))`, inverted by
   doubling in its merges' staged dtype (`algebra.doubled`), stored as the call holds it;
2. `w`, and `u`'s enclosure, from the stored `a` (the operand its products stage): `x beta [exp2(g)] + a (x
   beta [exp2(g)])` (`algebra.weighted`);
3. `c` from `u`, the stored `w` and the memory entering: `u - w S` (`algebra.corrected`), the memory staged
   as `h` is stored;
4. the next chunk's `h` (the last chunk's final memory) from the carry, `c`'s fp32 values and the keys:
   `S exp2(g_last) + k^T (c exp2(g_last - g))` (`algebra.scalar_write`, `carried`);
5. `o` from the queries, keys, `h` and `c`: `scale ((q h) exp2(g) + tril(q k^T exp2(g_s - g_t)) c)`
   (`algebra.chunk_scores`, `chunk_output`).

Under a decay a key dimension (Kimi delta attention's, a declared floor's, `key_enclosures`) the pairs are the
intra launch's, each pair's decay factored over sub-chunks (`key_pairs`), the weights gated by each key's
own decay, the keys gated to the chunk's end, and each key row of the memory decayed by its own.

Each enclosure is outward interval arithmetic in float64, each end snapped to its op's own grid: products of
16-bit values (or of an fp32 operand's three bf16 parts, six of them a pair, `product.formed`) exact, summed
as the native-dot contract holds a tensor core's sum (gamma_n at a truncating add's unit, of the terms that
may be nonzero); fp32 adds and multiplies rounded to nearest, a product before an add rounded or exact (an
FMA's contraction); casts to nearest; `exp2` within `reach` steps of the fp32 grid of its correctly rounded
value (PTX ISA 9.4 §9.7.3.21: ex2.approx.f32's 2 ulp, the mainloop's `approx_math`); and each log decay's
base-2 prefix within `EXPONENT_BOUND` of its exact value (XLA's compensated scan,
`recurrence_algebra.prefix`), then the kernel's own roundings of its differences."""

import dataclasses
import functools
import math
from typing import NamedTuple

import numpy as np
from certified import allowance
from native_certificates import TRUNCATING, truncating_underflow
from op_certificates import (
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    Format,
    above,
    below,
    exp2_rounded,
    grid_index,
    nearest,
    stepped,
)

from dew.nn.kernels import recurrence_algebra
from dew.nn.kernels.templates import bundles
from dew.nn.kernels.templates.recurrence import mainloop

CHUNK = mainloop.CHUNK
LOG2E = recurrence_algebra.LOG2_E
EXPONENT_BOUND = 2.0 ** -38
"""How far a position's prefix `hi + lo` (`recurrence_algebra.prefix`) lies from its exact `log2(e)
cumsum(g)`, as a share of `log2(e) sum |g|` over the positions it spans: each compensated add rounds two adds
of its remainders, each within 2^-23 of their sum, at most 2^-24 of the running value each; an element of a
64-long `associative_scan` takes at most 2 log2(64) such adds, under 2^-41 in all, and each token's exponent
`g log2(e)` is a Dekker product and its rounded remainder, within 2^-46 of it. Four times their sum."""
FRACTION = 2.0 ** -22
"""What `algebra.raised` rounds off its exponent: the fraction `(whole - floor(whole)) + part`, under 2,
rounded twice to nearest."""
PAIR_FRACTION = 2.0 ** -21
"""What `algebra.decayed` rounds off a pair's exponent: each side's fraction (where the side is under one, its
distance from its floor rounds), their difference and its sum with the low halves', each under 2."""
PAIR_FLOOR = recurrence_algebra.PAIR_FLOOR
"""A pair whose integers differ by less weighs zero (`algebra.PAIR_FLOOR`)."""


class Iv(NamedTuple):
    """Each element's closed interval of values, `[lo, hi]` (float64)."""
    lo: np.ndarray
    hi: np.ndarray


def point(x) -> Iv:
    x = np.asarray(x, np.float64)
    return Iv(x, x)


def held_iv(x) -> Iv:
    """`x` as an interval: an enclosure as it is, a record as its point."""
    return x if isinstance(x, Iv) else point(x)


def _outward(lo, hi) -> Iv:
    return Iv(np.nextafter(lo, -np.inf), np.nextafter(hi, np.inf))


def onto(iv: Iv, fmt: Format) -> Iv:
    """The values of `fmt` within `iv`: what a value known to be on `fmt`'s grid takes."""
    return Iv(above(iv.lo, fmt), below(iv.hi, fmt))


def cast(iv: Iv, fmt: Format) -> Iv:
    """A cast to nearest of a value within `iv` (monotone)."""
    return Iv(nearest(iv.lo, fmt), nearest(iv.hi, fmt))


def cell(x, fmt: Format) -> Iv:
    """The values that round to nearest to each `x` of `fmt`: between its midpoints with its neighbours."""
    x = np.asarray(x, np.float64)
    return Iv((x + stepped(x, fmt, -np.inf)) / 2, (x + stepped(x, fmt, np.inf)) / 2)


def meet(a: Iv, b: Iv) -> Iv:
    return Iv(np.maximum(a.lo, b.lo), np.minimum(a.hi, b.hi))


def hull(a: Iv, b: Iv) -> Iv:
    return Iv(np.minimum(a.lo, b.lo), np.maximum(a.hi, b.hi))


def add(a: Iv, b: Iv) -> Iv:
    return _outward(a.lo + b.lo, a.hi + b.hi)


def neg(a: Iv) -> Iv:
    return Iv(-a.hi, -a.lo)


def mul(a: Iv, b: Iv) -> Iv:
    """The exact products' range, outward but where it is an exact zero."""
    ends = np.stack(np.broadcast_arrays(a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi))
    lo, hi = ends.min(0), ends.max(0)
    return Iv(np.where(lo == 0, lo, np.nextafter(lo, -np.inf)),
              np.where(hi == 0, hi, np.nextafter(hi, np.inf)))


def where(mask, a: Iv, b: Iv) -> Iv:
    return Iv(np.where(mask, a.lo, b.lo), np.where(mask, a.hi, b.hi))


def within(native, iv: Iv) -> np.ndarray:
    """Whether each native value lies in its interval."""
    native = np.asarray(native, np.float64)
    return np.isfinite(native) & (native >= iv.lo) & (native <= iv.hi)


@dataclasses.dataclass(frozen=True)
class Lane:
    """The arithmetic a launch runs in: `exp2`'s reach (the most steps of the fp32 grid its result lies from
    its correctly rounded value) and whether fp32 results below the normal range flush to zero; `block`, its
    family's tile width, over which a product's contraction is issued in column blocks (`mainloop.block_of`);
    and how the call holds its operands and passed values (`held`, `given`, `wide`), which `enclosed` reads
    from its run."""
    reach: int
    flushes: bool
    block: int
    held: int = 1
    """The parts the call holds the values its launches pass between them in (`mainloop.Call.held`): one, a
    16-bit value's own or an fp32 one TF32 operand of; two or three bf16 parts (`product.formed`)."""
    given: int = 1
    """The parts its operands, the queries, keys and values, are given in (`mainloop.Call.given`)."""
    wide: bool = False
    """Whether the call is fp32's, its output stored fp32 and a one-part operand TF32."""


def fp32(iv: Iv, lane: Lane) -> Iv:
    """An fp32 add, subtract or multiply rounded to nearest of an exact value within `iv`, its result zero
    where it is below the normal range on a lane that flushes."""
    found = cast(iv, FLOAT32)
    if not lane.flushes:
        return found
    tiny = 2.0 ** FLOAT32.emin
    reaches = (found.lo < tiny) & (found.hi > -tiny)
    return Iv(np.where(reaches, np.minimum(found.lo, 0.0), found.lo),
              np.where(reaches, np.maximum(found.hi, 0.0), found.hi))


def fused(iv: Iv, lane: Lane) -> Iv:
    """An fp32 product an add after it may contract into an FMA: rounded, or exact."""
    return hull(iv, fp32(iv, lane))


PART_BOUNDS = (1 + 2.0 ** -8, 2.0 ** -8 * (1 + 2.0 ** -8), 2.0 ** -16 * (1 + 2.0 ** -8))
"""Each bf16 part's magnitude over its fp32 value's (`product.bf16_parts`): the first the value rounded,
within half a bf16 unit, 2^-8 of it; each later the remainder rounded, within 2^-8 of the last."""


def staged(iv: Iv, lane: Lane) -> Iv:
    """A register value as the call's launches hold it (`Lane.held`): cast to bf16 (a 16-bit call's), its fp32
    value (three parts summing to it, or one TF32 operand, which its product reads, `dot`), or two parts'
    sum, within 2^-16 of it."""
    if lane.held == 1:
        return iv if lane.wide else cast(iv, BFLOAT16)
    if lane.held == 2:
        reach = 2.0 ** -16 * np.maximum(np.abs(iv.lo), np.abs(iv.hi))
        return _outward(iv.lo - reach, iv.hi + reach)
    return iv


def held_cell(x, lane: Lane) -> Iv:
    """The fp32 values a stored record `x` holds as the call holds it (`staged`): those rounding to it in
    bf16, those its two parts sum within 2^-16 of, or itself."""
    if lane.held == 1 and not lane.wide:
        return cell(x, BFLOAT16)
    if lane.held == 2:
        x = np.asarray(x, np.float64)
        reach = 2.0 ** -16 * np.abs(x) * (1 + 2.0 ** -7)
        return _outward(x - reach, x + reach)
    return point(x)


def out(iv: Iv, lane: Lane) -> Iv:
    """The output as the call stores it: in its operands' dtype, bf16 or fp32 (`Lane.wide`)."""
    return iv if lane.wide else cast(iv, BFLOAT16)


def aligned(iv: Iv, fmt: Format) -> np.ndarray:
    """The largest magnitude a tensor core aligns a product by of an operand within `iv`
    (`native_certificates.aligned`: a subnormal of `fmt` at `fmt`'s least normal)."""
    from native_certificates import aligned as at

    return np.maximum(at(iv.lo, fmt), at(iv.hi, fmt))


def dot(a: Iv, b: Iv, lane: Lane, fmt: Format = BFLOAT16, into: Iv | None = None,
        parts: tuple[int, int] = (1, 1)) -> Iv:
    """`a @ b` (`[M, K] @ [K, N]`) as one product the tensor cores accumulate in fp32: each term's exact range
    (products of 16-bit values of `fmt`), summed within the native-dot contract's bound
    (`native_certificates.native_dot_holds`: gamma_n at a truncating add's unit, n the terms that may be
    nonzero, an add of an exact zero exact), the float64 sums within n 2^-52 of their own. The bound's
    magnitudes are the terms' as the tensor cores align them (`aligned`): a subnormal operand's at `fmt`'s
    least normal, the exponent its field holds. With `into`, the product accumulated into fp32 registers
    holding values within it (a tensor core's accumulate-in-place): one more term of the sum. Operands held in
    `parts` (`Lane.held`, `Lane.given`): a one-part fp32 operand read as TF32 (`operand`); bf16 parts
    multiplied over the pairs `product.terms` issues, each exact, their sum the exact product less the pairs
    dropped (`PART_BOUNDS`' products of them), the terms `product.terms` times as many."""
    from dew.nn.kernels.templates import product

    pa, pb = parts
    if lane.wide and pa == 1:
        a, fmt = operand(a), FLOAT32
    if lane.wide and pb == 1:
        b, fmt = operand(b), FLOAT32
    issued = product.terms(pa, pb)
    dropped = [(i, j) for i in range(pa) for j in range(pb) if (i, j) not in issued]
    share = sum(PART_BOUNDS[i] * PART_BOUNDS[j] for i, j in issued) if pa * pb > 1 else 1.0
    lost = sum(PART_BOUNDS[i] * PART_BOUNDS[j] for i, j in dropped)
    if pa * pb > 1:
        fmt = BFLOAT16
    terms = mul(Iv(a.lo[:, None, :], a.hi[:, None, :]), Iv(b.lo.T[None, :, :], b.hi.T[None, :, :]))
    if into is not None:
        terms = Iv(np.concatenate([terms.lo, into.lo[..., None]], -1),
                   np.concatenate([terms.hi, into.hi[..., None]], -1))
    live = ((terms.lo != 0) | (terms.hi != 0)).sum(-1) * len(issued)
    pairs = aligned(a, fmt)[:, None, :] * aligned(b, fmt).T[None, :, :]
    magnitude = pairs.sum(-1) * share * (1 + 2.0 ** -50)
    if into is not None:
        magnitude = magnitude + np.maximum(np.abs(into.lo), np.abs(into.hi)) * (1 + 2.0 ** -50)
    # The native-dot contract's allowance, and the float64 interval sums' own roundings beside it.
    slack = (allowance(magnitude, live, TRUNCATING["float32"], truncating_underflow(lane.flushes))
             + (live + 1) * 2.0 ** -52 * magnitude + lost * pairs.sum(-1) * (1 + 2.0 ** -50))
    return onto(_outward(terms.lo.sum(-1) - slack, terms.hi.sum(-1) + slack), FLOAT32)


def blocked(a: Iv, b: Iv, lane: Lane, parts: tuple[int, int] = (1, 1)) -> Iv:
    """`a @ b` as the mainloop issues it over its contraction's column blocks, the lane's family's tile
    width (`Lane.block`): each block's product a fresh accumulator (`dot`), summed in fp32 from zero
    (`sum`'s first add exact)."""
    width, block = a.lo.shape[1], lane.block
    total = None
    for start in range(0, width, block):
        part = dot(Iv(a.lo[:, start:start + block], a.hi[:, start:start + block]),
                   Iv(b.lo[start:start + block], b.hi[start:start + block]), lane, parts=parts)
        total = part if total is None else fp32(add(total, part), lane)
    return total


def exp2(x: Iv, lane: Lane) -> Iv:
    """`exp2` of an fp32 value within `x`: within `lane.reach` steps of the fp32 grid of its correctly rounded
    value (`op_certificates.exp2_rounded`, verified), at the least and greatest fp32 operands within `x`
    (round to nearest is monotone); zero where it may fall below the normal range (ex2's flush, or a
    scaling's last product rounding there)."""
    lo = exp2_rounded(above(x.lo, FLOAT32), FLOAT32)
    hi = exp2_rounded(below(x.hi, FLOAT32), FLOAT32)
    for _ in range(lane.reach):
        lo, hi = stepped(lo, FLOAT32, -np.inf), stepped(hi, FLOAT32, np.inf)
    return Iv(np.where(lo < 2.0 ** FLOAT32.emin, 0.0, lo), hi)


@dataclasses.dataclass(frozen=True)
class Exponents:
    """A chunk's positions' base-2 log decays from its start, `log2(e) cumsum(g)` (`exact`, float64 within
    2^-50 of it), and how far the kernel's prefix may lie from each (`bound`, `EXPONENT_BOUND`)."""
    exact: np.ndarray
    bound: np.ndarray

    @classmethod
    def of(cls, g) -> "Exponents":
        """A chunk's exponents of its log decays `g` `[C]` (a head's) or `[C, Dk]` (a key dimension's, each
        column its own prefix: the intra launch's scan of the same exact terms, `algebra.scanned_prefix`,
        its doubling's adds as many as the associative scan's)."""
        g = np.asarray(g, np.float64)
        columns = g.reshape(len(g), -1)
        exact = np.array([[math.fsum(columns[:s + 1, j]) for j in range(columns.shape[1])]
                          for s in range(len(g))]).reshape(g.shape) * LOG2E
        spanned = np.cumsum(np.abs(g), axis=0) * LOG2E * (1 + 2.0 ** -40)
        return cls(exact, EXPONENT_BOUND * spanned + 2.0 ** -50 * np.abs(exact))

    def raised(self, lane: Lane) -> Iv:
        """`algebra.raised(hi, lo)` at each position: the decay from the chunk's start."""
        return exp2(_outward(self.exact - self.bound - FRACTION, self.exact + self.bound + FRACTION), lane)

    def gated(self, lane: Lane) -> Iv:
        """`algebra.scalar_write`'s gate at each position, `raised(last - hi, last_lo - lo)`: the
        difference of the his rounded to nearest (half an fp32 unit of it) and of the los."""
        difference = self.exact[-1] - self.exact
        spread = (self.bound[-1] + self.bound + 2.0 ** -24 * np.abs(difference) * (1 + 2.0 ** -20)
                  + 2.0 ** -46 * (np.abs(self.exact[-1]) + np.abs(self.exact)) + FRACTION)
        return exp2(_outward(difference - spread, difference + spread), lane)

    def pairs(self, lane: Lane) -> Iv:
        """`algebra.decayed(hi, lo, hi, lo)` `[C, C]`: each pair's decay `exp2(E_s - E_t)` (`between`), the
        upper triangle's left to the caller's mask."""
        return between(self.exact[:, None], self.bound[:, None], self.exact[None, :], self.bound[None, :],
                       lane)


def between(later, later_bound, earlier, earlier_bound, lane: Lane) -> Iv:
    """`algebra.between` of two separated prefixes, `exp2(later - earlier)` (broadcasting): each side's
    exponent within its bound of exact, their fractions' difference rounded (`PAIR_FRACTION`), exp2 at the
    lane's reach and scaled by the wholes' difference exactly, zero where the integers' difference may pass
    `PAIR_FLOOR`."""
    difference = later - earlier
    spread = later_bound + earlier_bound + 2.0 ** -46 * (np.abs(later) + np.abs(earlier)) + PAIR_FRACTION
    found = exp2(_outward(difference - spread, np.minimum(difference + spread, 126.0)), lane)
    # The integers' difference lies within two of the exponent's.
    lo = np.where(difference - spread < PAIR_FLOOR + 2, 0.0, found.lo)
    return Iv(lo, np.where(difference + spread < PAIR_FLOOR - 2, 0.0, found.hi))


def _lags() -> np.ndarray:
    index = np.arange(CHUNK)
    return index[:, None] - index[None, :]


def inverse(k, beta, exponents: Exponents, lane: Lane) -> Iv:
    """The solve's stored inverse less its unit diagonal, `[C, C]` (`mainloop._solved`): the system
    `algebra.keyed` (`k k^T` over the key blocks, times the pairs' decays, times the row's write gate), then
    `algebra.doubled` on `-system`, each level's two merges staged in fp16 (a 16-bit call) or as parts."""
    product = blocked(point(k), point(np.asarray(k).T), lane, parts=(lane.given, lane.given))
    decayed = fp32(mul(product, exponents.pairs(lane)), lane)
    system = fp32(mul(decayed, point(np.asarray(beta)[:, None])), lane)
    return solved(where(_lags() > 0, system, point(np.zeros((CHUNK, CHUNK)))), lane)


def solved(system: Iv, lane: Lane) -> Iv:
    """`algebra.solved` of a chunk's strictly lower system `[C, C]`: `algebra.doubled` on `-system`, each
    level's two merges staged in fp16 (a 16-bit call's), less the unit diagonal."""
    lower_a = neg(system)
    rows, cols = np.indices((CHUNK, CHUNK))
    identity = (rows == cols).astype(np.float64)
    first = (rows % 2 == 1) & (cols == rows - 1)
    current = Iv(identity + np.where(first, lower_a.lo, 0.0), identity + np.where(first, lower_a.hi, 0.0))

    # A 16-bit call's merges in fp16; an fp32 call's as it holds its values (`mainloop._merged`).
    def merge(x: Iv) -> Iv:
        return staged(x, lane) if lane.wide else cast(x, FLOAT16)

    def merged(x: Iv, y: Iv) -> Iv:
        if lane.wide:
            return dot(x, y, lane, parts=(lane.held, lane.held))
        return dot(x, y, lane, FLOAT16)

    width = 2
    while width < CHUNK:
        lower = ((rows // width) % 2 == 1) & (cols // width == rows // width - 1)
        power = merged(merge(current), merge(lower_a))
        masked = where(lower, power, point(np.zeros((CHUNK, CHUNK))))
        joined = merged(merge(masked), merge(current))
        current = fp32(add(current, joined), lane)
        width *= 2
    return Iv(current.lo - identity, current.hi - identity)


def weighted(lower, tile, beta, decay: Iv | None, lane: Lane) -> tuple[Iv, Iv]:
    """`algebra.weighted` from the stored inverse `lower` (the operand its product stages): `(fp32 value,
    stored)`, the scaled tile `tile beta [decay]` (each multiply rounded) on the diagonal outside the
    product."""
    scaled = fp32(mul(point(tile), point(np.asarray(beta)[:, None])), lane)
    if decay is not None:
        # A head's decay a row's, `[C]`; a key dimension's its own entry's, `[C, Dk]`.
        scaled = fp32(mul(scaled, decay if decay.lo.ndim == 2 else Iv(decay.lo[:, None], decay.hi[:, None])),
                      lane)
    held = (lane.held, lane.held)
    value = fp32(add(scaled, dot(point(lower), staged(scaled, lane), lane, parts=held)), lane)
    return value, staged(value, lane)


def corrected(values: Iv, weights, entering, lane: Lane) -> Iv:
    """`algebra.corrected`'s fp32 value, `u - w S`: the stored weighted values `values`, the stored weighted
    keys, and the memory entering staged as the states launch stores it (`entering`, its key slices each a
    fresh product summed from zero)."""
    product = blocked(point(weights), point(entering), lane, parts=(lane.held, lane.held))
    return fp32(add(values, neg(product)), lane)


def carried(carry: Iv, values: Iv, k, exponents: Exponents, lane: Lane) -> Iv:
    """The memory leaving the chunk, fp32 (`algebra.scalar_write`, `carried`): the carry decayed to the
    chunk's end plus the keys against the corrected values gated to it, each staged as the call holds it."""
    gate = exponents.gated(lane)
    gated = fp32(mul(values, Iv(gate.lo[:, None], gate.hi[:, None])), lane)
    keys = point(np.asarray(k).T)
    write = dot(keys, staged(gated, lane), lane, parts=(lane.given, lane.held))
    decay = exponents.raised(lane)
    held = Iv(np.full(carry.lo.shape, decay.lo[-1]), np.full(carry.lo.shape, decay.hi[-1]))
    return fp32(add(fused(mul(carry, held), lane), write), lane)


def output(q, k, entering, values, exponents: Exponents, scale: float, lane: Lane) -> Iv:
    """The chunk's fp32 output before its store (`algebra.chunk_scores`, `chunk_output`): its own pairs'
    scores staged as the call holds them, the memory it entered read through the decayed queries, each term
    scaled by fp32's `scale` and their sum rounded once, either product before it maybe contracted."""
    given, held = lane.given, lane.held
    products = blocked(point(q), point(np.asarray(k).T), lane, parts=(given, given))
    scores = fp32(mul(products, exponents.pairs(lane)), lane)
    scores = staged(where(_lags() >= 0, scores, point(np.zeros((CHUNK, CHUNK)))), lane)
    memory = blocked(point(q), held_iv(entering), lane, parts=(given, held))
    decay = exponents.raised(lane)
    memory = fp32(mul(memory, Iv(decay.lo[:, None], decay.hi[:, None])), lane)
    factor = point(np.float32(scale))
    local = dot(scores, held_iv(values), lane, parts=(held, held))
    return fp32(add(fused(mul(memory, factor), lane), fused(mul(local, factor), lane)), lane)


def keys_gated_enclosures(step: "Step", scale: float, lane: Lane,
                          entering: Iv | None = None) -> dict[str, Iv]:
    """`enclosures` of a chunk whose write gate takes the keys' side (`mainloop.Call.gated_keys`, Mamba2's SSD
    under the fast forms): the memory it leaves, fp32 (`algebra.keys_gated_carried`: the carry decayed to the
    chunk's end, plus each key row gated in fp32 by its write gate times its decay to the end and rounded to
    the values' dtype, against the values as given), and its output before its store (`chunk_scores` with
    each key column scaled by its write gate before the scores round, `chunk_output`). The scores' tile is a
    derived form (`algebra.derived_tile`), its products, decays and gates multiplied in either association:
    their enclosure is both's hull. `entering` is the memory its output reads, by default the stored `h`."""
    exponents = Exponents.of(step.g)
    entering = point(step.h) if entering is None else entering
    gate = fp32(mul(exponents.gated(lane), point(step.beta)), lane)
    keys = staged(fp32(mul(point(step.k), Iv(gate.lo[:, None], gate.hi[:, None])), lane), lane)
    write = dot(Iv(keys.lo.T, keys.hi.T), point(step.v), lane, parts=(lane.held, lane.given))
    decay = exponents.raised(lane)
    held = Iv(np.full(step.carry.lo.shape, decay.lo[-1]), np.full(step.carry.lo.shape, decay.hi[-1]))
    leaving = fp32(add(fused(mul(step.carry, held), lane), write), lane)
    products = blocked(point(step.q), point(np.asarray(step.k).T), lane, parts=(lane.given, lane.given))
    pairs = exponents.pairs(lane)
    columns = point(np.asarray(step.beta)[None, :])
    scores = hull(fp32(mul(fp32(mul(products, pairs), lane), columns), lane),
                  fp32(mul(products, fp32(mul(pairs, columns), lane)), lane))
    scores = staged(where(_lags() >= 0, scores, point(np.zeros((CHUNK, CHUNK)))), lane)
    memory = fp32(mul(blocked(point(step.q), entering, lane, parts=(lane.given, lane.held)),
                      Iv(decay.lo[:, None], decay.hi[:, None])), lane)
    factor = point(np.float32(scale))
    local = dot(scores, point(step.v), lane, parts=(lane.held, lane.given))
    output = fp32(add(fused(mul(memory, factor), lane), fused(mul(local, factor), lane)), lane)
    return {"h": leaving, "o": out(output, lane)}


def walk_enclosures(step: "Step", scale: float, lane: Lane, *, write: str) -> dict[str, Iv]:
    """`enclosures` of a chunk of the fused inference walk (`mainloop._scan`, a 16-bit call no gradient
    follows, under a decay a head), whose records are its inverse (the delta rule's, `_solved`'s), its output
    and the memory it leaves: the memory it enters in its registers, staged as the memories' form for its
    products, the walk's corrected values from it (`_scan_corrected`: under the delta rule
    `(I + L) (beta (v - exp2(g) k S))`, the keys' prediction from the memory decayed to each token, its
    difference from the values write-gated and corrected by the stored inverse; the gated values without
    it; the values as given where the keys take the gate), and the output and leaving memory from them."""
    exponents = Exponents.of(step.g)
    entering = staged(step.carry, lane)
    if write == "keys":
        return keys_gated_enclosures(step, scale, lane, entering)
    found = {}
    if write == "delta":
        found["a"] = staged(inverse(step.k, step.beta, exponents, lane), lane)
        predicted = blocked(point(step.k), entering, lane, parts=(lane.given, lane.held))
        decay = exponents.raised(lane)
        decayed = fused(mul(Iv(decay.lo[:, None], decay.hi[:, None]), neg(predicted)), lane)
        difference = fp32(add(point(step.v), decayed), lane)
        scaled = fp32(mul(difference, point(np.asarray(step.beta)[:, None])), lane)
        lower = dot(point(step.a), staged(scaled, lane), lane, parts=(lane.held, lane.held))
        corrected = fp32(add(scaled, lower), lane)
    else:
        corrected = fp32(mul(point(step.v), point(np.asarray(step.beta)[:, None])), lane)
    output_ = output(step.q, step.k, entering, staged(corrected, lane), exponents, scale, lane)
    return {**found, "h": carried(step.carry, corrected, step.k, exponents, lane), "o": out(output_, lane)}


def values_gated_enclosures(step: "Step", scale: float, lane: Lane) -> dict[str, Iv]:
    """`enclosures` of a chunk without the delta rule whose write gate takes the values' side (its values held
    in parts, `Lane.held`: an fp32 call's, or a 16-bit one's under the accurate forms, Lightning's or the
    SSD's): its corrected values, the values times their write gate as stored (`algebra.weighted`'s
    identity inverse, `corrected` with no weighted keys), the memory it leaves (`carried`), and its output."""
    exponents = Exponents.of(step.g)
    written = fp32(mul(point(step.v), point(np.asarray(step.beta)[:, None])), lane)
    values = meet(written, held_cell(step.c, lane))
    return {"c": staged(written, lane), "h": carried(step.carry, values, step.k, exponents, lane),
            "o": out(output(step.q, step.k, step.h, step.c, exponents, scale, lane), lane)}


class Step(NamedTuple):
    """One chunk and head's operands as its launches read them (float64), its records' stored values, and the
    carry entering it (the fp32 values the states launch holds)."""
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    g: np.ndarray
    beta: np.ndarray
    a: np.ndarray
    w: np.ndarray
    h: np.ndarray
    c: np.ndarray
    carry: Iv


def solved_step(step: Step, decay: Iv, lane: Lane) -> tuple[Iv, Iv, Iv]:
    """A step's records past its solve, from its stored inverse: the weighted keys under `decay` (a head's
    `[C]` or a key dimension's `[C, Dk]`) as stored, the corrected values as stored, and their fp32 values
    the write takes (within their enclosure and rounding to the stored `c`)."""
    _, w = weighted(step.a, step.k, step.beta, decay, lane)
    _, u = weighted(step.a, step.v, step.beta, None, lane)
    c = corrected(u, step.w, step.h, lane)
    return w, staged(c, lane), meet(c, held_cell(step.c, lane))


def enclosures(step: Step, scale: float, lane: Lane) -> dict[str, Iv]:
    """Each record's enclosure in its stored values (`a`, `w`, `c`, `o`) and the memory leaving the chunk,
    fp32 (`h`, the next chunk's entering memory before its store's rounding), each from the native records
    its launch reads (the module's docstring)."""
    if np.ndim(step.g) == 2:
        return key_enclosures(step, scale, lane)
    exponents = Exponents.of(step.g)
    w, c, values = solved_step(step, exponents.raised(lane), lane)
    stored = staged(inverse(step.k, step.beta, exponents, lane), lane)
    return {"a": stored, "w": w, "c": c,
            "h": carried(step.carry, values, step.k, exponents, lane),
            "o": out(output(step.q, step.k, step.h, step.c, exponents, scale, lane), lane)}


SUB = recurrence_algebra.SUB
"""The sub-chunk a decay a key dimension's pairs factor over (`algebra.key_block_pairs`)."""


def operand(iv: Iv) -> Iv:
    """An fp32 value as the factored pairs' products take it (`mainloop._pairing_operand`): TF32 (10 bits
    past the leading one, rounded or truncated) or two bf16 parts and their pairs' products of orders under
    two (16 bits, the dropped term under 2^-16 of the product), both within 2^-10 of the value."""
    widen = 2.0 ** -10 * np.maximum(np.abs(iv.lo), np.abs(iv.hi))
    return _outward(iv.lo - widen, iv.hi + widen)


def key_pairs(rows: Iv, k, exponents: Exponents, lane: Lane) -> Iv:
    """`algebra.key_block_pairs` `[C, C]` under a decay a key dimension, the pairs of sub-chunks at or before
    each row's (a declared floor's, `diagonal`): `sum_k r_ik k_jk exp2(E_ik - E_jk)` factored at row `i`'s
    sub-chunk's first token `G_s` and key `j`'s last `E_t`, the rows gated down (`exp2(E_i - G_s)`) and by
    the middle factor (`exp2(G_s - E_t)`), the keys up (`exp2(E_t - E_j)`), each an fp32 product taken as
    its operand form (`operand`), the products' sum over every key block within one native sum's bound of
    its terms (a key block's sum and its adds into the total, at most as many roundings as the width's
    terms). The caller masks the lags."""
    exact, bound = exponents.exact, exponents.bound
    sub_of = np.arange(CHUNK) // SUB
    firsts, lasts = sub_of * SUB, sub_of * SUB + SUB - 1
    down = between(exact, bound, exact[firsts], bound[firsts], lane)
    gated = fp32(mul(rows, down), lane)
    up = operand(fp32(mul(point(k), between(exact[lasts], bound[lasts], exact, bound, lane)), lane))
    found = point(np.zeros((CHUNK, CHUNK)))
    for t in range(CHUNK // SUB):
        reached = sub_of >= t
        middle = between(exact[firsts], bound[firsts], exact[t * SUB + SUB - 1][None, :],
                         bound[t * SUB + SUB - 1][None, :], lane)
        middle = where(reached[:, None], middle, point(np.zeros(middle.lo.shape)))
        left = operand(fp32(mul(gated, middle), lane))
        columns = slice(t * SUB, (t + 1) * SUB)
        product = dot(left, Iv(up.lo[columns].T, up.hi[columns].T), lane, FLOAT32)
        found = Iv(np.concatenate([found.lo[:, :t * SUB], product.lo, found.lo[:, (t + 1) * SUB:]], 1),
                   np.concatenate([found.hi[:, :t * SUB], product.hi, found.hi[:, (t + 1) * SUB:]], 1))
    return found


def key_enclosures(step: Step, scale: float, lane: Lane) -> dict[str, Iv]:
    """`enclosures` under a decay a key dimension (`g` `[C, Dk]`, Kimi delta attention's, a declared floor's
    factored pairs): the intra launch's system (the write-gated keys' pairs, strict, fp32) and scores (the
    queries' pairs, scaled, stored as held), the solve, the weights under each key's decay, the keys gated
    to the chunk's end (`algebra.key_decayed_write`, stored), the memory each key row decays by its own,
    and the output's one sum of the gated queries against the memory and the scores against the values
    (`algebra.key_gated`, `key_decayed_output`)."""
    exponents = Exponents.of(step.g)
    raised = exponents.raised(lane)
    zeros = point(np.zeros((CHUNK, CHUNK)))
    writes = fp32(mul(point(step.k), point(np.asarray(step.beta)[:, None])), lane)
    system = where(_lags() > 0, key_pairs(writes, step.k, exponents, lane), zeros)
    w, c, values = solved_step(step, raised, lane)
    exact, bound = exponents.exact, exponents.bound
    # `range_decay`: the difference of the compensated prefixes, then `raised`'s fraction.
    spread = bound[-1][None, :] + bound + 2.0 ** -46 * (np.abs(exact[-1])[None, :] + np.abs(exact)) + FRACTION
    reach = exp2(_outward(exact[-1][None, :] - exact - spread, exact[-1][None, :] - exact + spread), lane)
    written = staged(fp32(mul(point(step.k), reach), lane), lane)
    write = dot(Iv(written.lo.T, written.hi.T), staged(values, lane), lane)
    held = Iv(np.broadcast_to(raised.lo[-1][:, None], step.carry.lo.shape),
              np.broadcast_to(raised.hi[-1][:, None], step.carry.lo.shape))
    leaving = fp32(add(fused(mul(step.carry, held), lane), write), lane)
    factor = point(np.float32(scale))
    scores = fp32(mul(key_pairs(point(step.q), step.k, exponents, lane), factor), lane)
    # Each key block's scores scaled before their sum: one rounding a block more, at most the width's.
    width = np.asarray(step.k).shape[-1]
    slack = width * 2.0 ** -24 * np.maximum(np.abs(scores.lo), np.abs(scores.hi))
    scores = staged(where(_lags() >= 0, _outward(scores.lo - slack, scores.hi + slack), zeros), lane)
    queries = staged(fp32(mul(fp32(mul(point(step.q), raised), lane), factor), lane), lane)
    left = Iv(np.concatenate([queries.lo, scores.lo], 1), np.concatenate([queries.hi, scores.hi], 1))
    right = point(np.concatenate([step.h, step.c], 0))
    return {"a": staged(solved(system, lane), lane), "w": w, "c": c, "h": leaving,
            "o": out(dot(left, right, lane), lane)}


def stored_memory(iv: Iv, lane: Lane) -> Iv:
    """The memory as the states launch stores it: bf16, or fp32 exactly."""
    return staged(iv, lane)


def exp2_reach(native, low: float = -2.0, high: float = 2.0) -> int:
    """The most steps of the fp32 grid `native(x)`, an fp32 `exp2` of fp32 operands, lies from `2^x` correctly
    rounded (`op_certificates.exp2_rounded`), over the operands a mainloop takes, (`low`, `high`): by
    default the fractions `algebra.raised` and `algebra.decayed` take, (-2, 2); 2^17 of them, densely and
    drawn, the integers there (exact cases) among them. A zero is the flush `exp2` admits (its enclosure is
    zero where its lower end, that many steps down, falls below the normal range), not a step: measured
    over the nonzero results, and a zero where that end is normal refuses the lane."""
    rng = np.random.default_rng(2084)
    whole = np.arange(np.ceil(low), np.floor(high) + 1, dtype=np.float32)
    x = np.concatenate([np.linspace(low, high, 1 << 16, dtype=np.float32), whole,
                        rng.uniform(low, high, (1 << 16) - len(whole)).astype(np.float32)])
    found, exact = np.asarray(native(x), np.float64), exp2_rounded(x, FLOAT32)
    flushed = found == 0
    reach = int(np.max(np.abs(grid_index(found) - grid_index(exact))[~flushed]))
    lowest = exact[flushed]
    for _ in range(reach):
        lowest = stepped(lowest, FLOAT32, -np.inf)
    normal = lowest >= 2.0 ** FLOAT32.emin
    if np.any(normal):
        raise ArithmeticError(f"the lane's exp2 flushes a normal result at {x[flushed][normal][:4]}")
    return reach


RECORDS = ("a", "w", "c", "o", "h", "leaving")
KEYS_GATED_RECORDS = ("o", "h", "leaving")
"""The records a call whose write gate takes the keys' side keeps (`keys_gated_enclosures`): its values are
its corrected values, held as given."""
VALUES_GATED_RECORDS = ("c", "o", "h", "leaving")
"""The records a call without the delta rule whose write gate takes the values' side keeps
(`values_gated_enclosures`)."""
WRITES = {"delta": RECORDS, "keys": KEYS_GATED_RECORDS, "values": VALUES_GATED_RECORDS}
"""Each write's records (`Run.write`)."""
WALKED = {"delta": ("a", "o", "leaving"), "keys": ("o", "leaving"), "values": ("o", "leaving")}
"""Each write's records in the fused inference walk (`walk_enclosures`), which stores no memory a chunk enters
and no weighted keys or corrected values."""
"""The records the witness holds (`enclosed`)."""


@functools.cache
def interpreted_lane() -> Lane:
    """The arithmetic Mosaic GPU's interpreter runs a launch in, a CPU process's XLA, measured once: its
    `exp2`'s reach over the witness's own operands (`exp2_reach`: XLA's CPU exp2, computed as exp(x ln 2), is
    a step off some powers of two, 2^71 among them, which check 2's exact cases admit no lane breaking, so
    its reach is the witness's own measure, not `certified.lane`'s) and whether it flushes (`certified.lane`),
    a 16-bit call."""
    import jax
    import jax.numpy as jnp
    from certified import lane

    return Lane(exp2_reach(lambda x: jax.jit(jnp.exp2)(jnp.asarray(x))), lane().flushes,
                mainloop.block_of(bundles.INTERPRETED))


@dataclasses.dataclass(frozen=True)
class Run:
    """A call's operands as its launches read them and its records, one row: `q`, `k` `[N C, Hk, Dk]`, `v`
    `[N C, H, Dv]`, `g`, `beta` `[N C, H]` (float64 of their dtypes' values), laid a chunk at a time as the
    launches lay them (`laid_chunks`: a document's chunks its own, a chunk's rows past its tokens zero); each
    chunk and head's stored inverse less
    its diagonal `a` `[N, H, C, C]`, weighted keys `w` `[N, H, C, Dk]`, entering memory `h` `[N, H, Dk, Dv]`,
    corrected values `c` and output `o` `[N, H, C, Dv]`, an fp32 call's three parts summed; and the fp32
    memory each chunk leaves, `carries` `[N + 1, H, Dk, Dv]` from the initial memory: the final memory of the
    call over the tokens up to the chunk's last, at the same tiles (`recorded`), the state the states launch
    carries in fp32 and stores rounded as `h`, which `h`'s enclosure holds bit for bit (`enclosed`); and
    `begins` `[N]`, the chunks a packed call's documents past the row's first begin, which enter the zero
    memory such a document starts from in place of the one the chunk before leaves."""
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    g: np.ndarray
    beta: np.ndarray
    a: np.ndarray
    w: np.ndarray
    h: np.ndarray
    c: np.ndarray
    o: np.ndarray
    carries: np.ndarray
    begins: np.ndarray
    scale: float
    """The output's scale (the keys' width's inverse root, a delta rule's; 1, the SSD's)."""
    write: str
    """Its write (`WRITES`): the delta rule's; the write gate on the keys' side (`mainloop.Call.gated_keys`,
    the SSD's under the fast forms), no inverse, weighted keys or corrected values of its own (`a`, `w`, `c`
    zero); or on the values' side without the delta rule, its corrected values the gated values (`a`, `w`
    zero)."""
    held: int
    """The parts its launches hold the values they pass in (`Lane.held`)."""
    given: int
    """The parts its operands are given in (`Lane.given`)."""
    wide: bool
    """Whether it is an fp32 call (`Lane.wide`)."""
    walked: bool
    """Whether it is the fused inference walk's (`walk_enclosures`), its records `WALKED`'s."""

    @property
    def chunks(self) -> int:
        return self.a.shape[0]

    @property
    def heads(self) -> int:
        return self.a.shape[1]

    def entered(self, n: int, head: int) -> np.ndarray:
        """The fp32 memory chunk `n` and value head `head` enter: zero where a document begins it (`begins`),
        else the one the chunk before leaves (`carries`)."""
        return np.zeros_like(self.carries[n, head]) if self.begins[n] else self.carries[n, head]

    def step(self, n: int, head: int, *, carry: Iv | None = None) -> Step:
        """Chunk `n` and value head `head`'s step: its operands, its records and the fp32 memory it enters
        (`entered`), unless given."""
        rows, key = slice(n * CHUNK, (n + 1) * CHUNK), head // (self.heads // self.k.shape[1])
        return Step(self.q[rows, key], self.k[rows, key], self.v[rows, head], self.g[rows, head],
                    self.beta[rows, head], self.a[n, head], self.w[n, head], self.h[n, head], self.c[n, head],
                    point(self.entered(n, head)) if carry is None else carry)


def laid_chunks(length: int, starts, max_segments: int | None) -> np.ndarray:
    """The chunks a call of one row of `length` tokens lays its tokens in, `[N, 2]`: each chunk's first
    token and its count, in the launches' order (`chunked_launch.sequences`, the mainloop's layout): a row's
    own chunks, or each document's (its first token marked in `starts` `[1, T]`, up to `max_segments` a
    row) its own, so a document beginning inside a chunk starts a chunk of its own."""
    from dew.nn.inputs import cu_seqlens
    from dew.nn.kernels import chunked_launch

    if starts is None:
        table = chunked_launch.sequences(1, length, chunk=CHUNK).chunks
    else:
        documents = length if max_segments is None else max_segments
        table = chunked_launch.sequences(1, length, np.asarray(cu_seqlens(starts, documents + 1)),
                                         chunk=CHUNK).chunks
    table = np.asarray(table)[:, :2]
    return table[table[:, 1] > 0]


def recorded(q, k, v, g, beta, h0, *, starts=None, max_segments: int | None = None,
             floor: float | None = None, carried: bool = True, delta: bool = True,
             rounded: bool | None = None, walked: bool = False) -> Run:
    """The records of the gated delta rule's training call on the template, forced (`generation.chosen`) at
    JAX's own default precision as a model trains it: `mainloop.kept_forward` at the tiles the route takes
    (`tuning.seen`), its output and final memory bit for bit the route's training forward's (the certify
    build is the production program), on the device's bundle, interpreted off a GPU. A call no gradient
    follows runs another program, its states and output in one walk that never stores the states it reads
    (`mainloop.forward`), which this witness does not hold. Operands `[1, T, ...]`. With
    `carried`, each chunk's fp32 memory leaving it: the same launches at the same tiles over the chunks up to
    it, whose final memory, rounded as the launch stores `h`, is the next chunk's `h` bit for bit (`seams`,
    each chunk's: end_j == start_{j+1}); without, the initial and the final memories alone. A packed call's
    documents (`starts` `[1, T]`) lay their own chunks (`laid_chunks`), a document beginning inside a chunk
    included, its first chunk entering the zero memory. Without the `delta` rule, the template's op as the
    SSD's route calls it (`recurrence_chunk.template_rule`, scale 1, its output in the operands' dtype),
    `rounded` as the route's fast forms round it unless given (a 16-bit call's keys then take the write
    gate, `mainloop.Call.gated_keys`; unrounded, Lightning's, its values held in parts). An fp32 call's
    operands and passed values are held in parts as the mainloop holds them (`Lane.held`, `Lane.given`),
    each record their sum. `walked`, the inference forward no gradient follows: a 16-bit call's under a decay
    a head runs its fused walk (`mainloop._scan`, `kept_forward` keeping nothing), whose records are its
    inverse, output and final memory (`WALKED`)."""
    import jax
    import jax.numpy as jnp

    from dew.nn.kernels import generation, recurrence_chunk, tuning

    # Under a decay a key dimension the witness holds a declared floor's factored pairs (`key_pairs`): a
    # call without one takes each sub-chunk's own pairs (`algebra.key_diagonal_pairs`), which it does not.
    assert np.ndim(g) == 3 or (floor is not None and recurrence_algebra.factored_range(floor)), \
        "the witness holds a decay a key dimension's factored pairs alone"
    from dew.nn.kernels.templates.recurrence import template

    operands = (q, k, v, g, beta, h0)
    dtype = jnp.result_type(q, k, v)
    scale = k.shape[-1] ** -0.5 if delta else 1.0
    if rounded is None:
        rounded = recurrence_chunk.chunk_rounding(delta, dtype) if delta else recurrence_chunk.fast_rounding()
    family = mainloop.family_of(dtype) or bundles.INTERPRETED
    # The parts its operands and passed values take (`mainloop.prepared`'s `given` and `held`): one of a
    # 16-bit value; an fp32 one's three, or under the fast forms its family's (`mainloop._fast_parts`).
    wide = jnp.dtype(dtype) == jnp.float32
    fast = 1 if family.takes_tf32 and family.transposes_tf32 else mainloop.FAST_PARTS
    given = (fast if rounded else mainloop.PARTS) if wide else 1
    held = given if delta or rounded else mainloop.PARTS
    write = "delta" if delta else ("keys" if held == 1 else "values")

    def route(*x):
        if not delta:
            return recurrence_chunk.template_rule(*x, starts, max_segments=max_segments, scale=scale,
                                                  delta=False, wide=False, rounded=rounded)
        # The draws' sigmoid write gates of unit keys (`recurrence.Hooks.write_bound`), as layers declare.
        return recurrence_chunk.chunk_gated_delta_rule(*x, starts=starts, max_segments=max_segments,
                                                       floor=floor, write_bound=1.0)

    with jax.default_matmul_precision(None), generation.chosen(gated_delta_rule="mosaic_gpu"):
        with tuning.seen() as launches:
            # The training forward: the primal of the route's VJP, as a training step runs it; or the
            # inference forward.
            o, final = jax.jit(route if walked else lambda *x: jax.vjp(route, *x)[0])(*operands)
        [tiles] = [config for (op, _), (config, _) in zip(launches, launches.taken, strict=True)
                   if isinstance(op, template.Mosaic)]

        def kept(*x, starts=starts):
            return jax.jit(lambda *x: mainloop.kept_forward(
                *x, tiles=tiles, starts=starts, max_segments=max_segments, floor=floor, scale=scale,
                delta=delta, rounded=rounded, family=family, keeps=not walked,
                interpret=generation.backend() != "gpu"))(*x)

        kept_o, kept_final, (a, (w, h, c)) = kept(*operands)
        for native, own in ((o, kept_o), (final, kept_final)):
            assert np.array_equal(np.asarray(native).view(np.uint8), np.asarray(own).view(np.uint8)), \
                "the certify build is the production program"
        _, length, heads, value_width = v.shape
        key_width = k.shape[-1]
        table = laid_chunks(length, starts, max_segments)
        chunks = len(table)
        # Each chunk's leaving memory: the final memory of the call over the tokens up to its last, the
        # document it ends then the row's last.
        ends = [kept(*(x[:, :first + count] for x in operands[:5]), h0,
                     starts=None if starts is None else starts[:, :first + count])[1]
                for first, count in table[:-1]] if carried else []
    # The chunks the launches lay out (a packed call's every sequence's, its worst case's), of which the
    # call's come first.
    if walked:
        # The walk stores no memories, weighted keys or corrected values (`WALKED`), each chunk its own.
        assert starts is None and jnp.dtype(dtype).itemsize == 2, "the walk is a 16-bit call's, unpacked"
        laid = chunks
        w = np.zeros((held * laid * CHUNK, heads * key_width))
        h, c = (np.zeros((held * laid * heads * key_width, value_width)),
                np.zeros((held * laid * CHUNK, heads * value_width)))
    laid = np.shape(h)[0] // (held * heads * key_width)
    if not delta:
        # No inverse or weighted keys of its own, nor corrected values where its keys take the gate.
        a, w = (np.zeros((held * laid * CHUNK, heads * width)) for width in (CHUNK, key_width))
        if write == "keys":
            c = np.zeros((held * laid * CHUNK, heads * value_width))

    def summed(x, rows):
        """A record's stacked parts (`mainloop.Operand`) summed, exactly, `[rows, ...]`."""
        return np.asarray(x, np.float64).reshape(held, rows, -1).sum(0)

    def chunked(x, width):
        return summed(x, laid * CHUNK).reshape(laid, CHUNK, heads, width).transpose(0, 2, 1, 3)[:chunks]

    memories = summed(h, laid * heads * key_width).reshape(laid, heads, key_width, value_width)[:chunks]
    carries = [np.asarray(h0, np.float64)[0], *(np.asarray(end, np.float64)[0] for end in ends)]
    unknown = np.full(carries[0].shape, np.nan)
    carries += [unknown] * (chunks - len(carries)) + [np.asarray(kept_final, np.float64)[0]]
    hosted_operands = {name: _laid(value, table)
                       for name, value in hosted(operands, given=given, delta=delta).items()}
    output = _laid(np.asarray(kept_o, np.float64)[0], table).reshape(chunks, CHUNK, heads, value_width)
    # A document past the row's first begins on the zero memory (`chunked_launch.Sequences`).
    begins = np.asarray([first > 0 and (starts is not None and bool(np.asarray(starts)[0, first]))
                         for first, _ in table])
    return Run(**hosted_operands, a=chunked(a, CHUNK), w=chunked(w, key_width), h=memories, scale=scale,
               write=write, held=held, given=given, wide=bool(wide), walked=walked,
               c=chunked(c, value_width), o=output.transpose(0, 2, 1, 3), carries=np.stack(carries),
               begins=begins)


def _laid(x: np.ndarray, table: np.ndarray) -> np.ndarray:
    """A row's `[T, ...]` laid a chunk at a time by `table` (`laid_chunks`), `[N C, ...]`: each chunk's
    tokens, its rows past them zero."""
    found = np.zeros((len(table) * CHUNK, *x.shape[1:]), x.dtype)
    for n, (first, count) in enumerate(table):
        found[n * CHUNK:n * CHUNK + count] = x[first:first + count]
    return found


def hosted(operands: tuple, *, given: int = 1, delta: bool = True) -> dict[str, np.ndarray]:
    """A call's operands `(q, k, v, g, beta, h0)` as its launches read them, one row's, float64 (`Run`'s): the
    queries, keys and values in their common dtype, in its `given` parts summed exactly (`product.formed`; an
    fp32 one-part operand itself, which its products read as TF32), the delta rule's write gate rounded to it
    (without the rule its own fp32), the log decay fp32."""
    import jax.numpy as jnp

    from dew.nn.kernels.templates import product

    q, k, v, g, beta, _ = operands
    dtype = jnp.result_type(q, k, v)

    def host(x):
        held = jnp.asarray(x).astype(dtype).astype(jnp.float32)
        if given > 1:
            return sum(np.asarray(part, np.float64) for part in product.formed(held, given))[0]
        return np.asarray(held, np.float64)[0]

    gate = host(beta) if delta else np.asarray(jnp.asarray(beta, jnp.float32), np.float64)[0]
    return {"q": host(q), "k": host(k), "v": host(v), "g": np.asarray(g, np.float64)[0], "beta": gate}


def enclosed(run: Run, lane: Lane) -> dict[str, Iv]:
    """Each record's enclosure `[N, H, ...]` (`enclosures`), each chunk's step from the fp32 memory it enters:
    `a`, `w`, `c`, `o`; `h`, the memory each chunk enters as stored; and `leaving`, the fp32 memory it
    leaves."""
    writes = {"delta": enclosures, "keys": keys_gated_enclosures, "values": values_gated_enclosures}
    enclose = functools.partial(walk_enclosures, write=run.write) if run.walked else writes[run.write]
    lane = dataclasses.replace(lane, held=run.held, given=run.given, wide=run.wide)
    found: dict[str, list] = {}
    for n in range(run.chunks):
        for head in range(run.heads):
            step = enclose(run.step(n, head), run.scale, lane)
            step["leaving"], step["h"] = step["h"], stored_memory(point(run.entered(n, head)), lane)
            for name, iv in step.items():
                found.setdefault(name, []).append(iv)
    return {name: Iv(*(np.stack([getattr(iv, end) for iv in ivs]).reshape(run.chunks, run.heads,
                                                                           *ivs[0].lo.shape)
                       for end in ("lo", "hi"))) for name, ivs in found.items()}


def records(run: Run) -> dict[str, np.ndarray]:
    """Each record `[N, H, ...]`, as `enclosed` names them."""
    held = {"a": run.a, "w": run.w, "c": run.c, "o": run.o, "h": run.h, "leaving": run.carries[1:]}
    return {name: held[name] for name in (WALKED if run.walked else WRITES)[run.write]}


def witness(run: Run, lane: Lane) -> dict:
    """Whether each record of `run` lies inside its step's enclosure (`enclosed`), by record `[N, H, ...]`."""
    found, recorded_ = enclosed(run, lane), records(run)
    return {name: within(recorded_[name], found[name]) for name in recorded_}


def refusals(faulty: Run, claimed: Run, lane: Lane) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """A fault's refusals, by record: `faulty`'s records held to the enclosures `claimed` gives them (the
    call's own operands and memories, about `faulty`'s records), and where those are separated from the
    enclosures on what the faulty kernel was given (`faulty`'s own, `native_certificates.separated`)."""
    from native_certificates import separated

    truth, fault, found = enclosed(claimed, lane), enclosed(faulty, lane), records(faulty)
    return {name: (within(found[name], truth[name]), separated(truth[name], fault[name])) for name in found}


def rounded_cumsums(g) -> np.ndarray:
    """Log decays `[1, T, H]` whose exact prefix within each chunk is `g`'s cumulative sum in fp32, token by
    token (row 2066's defect, the decay from a difference of rounded cumulative sums): each token's the
    difference of two neighbouring rounded sums, rounded to fp32, so the kernel's compensated prefix carries
    their rounding."""
    shape = np.shape(g)
    chunked = np.asarray(g, np.float32).reshape(shape[0], -1, CHUNK, shape[-1])
    sums = np.cumsum(chunked, axis=2, dtype=np.float32)
    before = np.concatenate([np.zeros_like(sums[:, :, :1]), sums[:, :, :-1]], axis=2)
    return (sums - before).astype(np.float32).reshape(shape)


def shifted(operands: tuple) -> tuple:
    """A call's operands `(q, k, v, g, beta, h0)` with each token one later: a neutral token first (no key,
    value, decay or write), the last dropped, so each chunk's boundary falls one token early."""
    import jax.numpy as jnp

    def later(x):
        return jnp.concatenate([jnp.zeros_like(x[:, :1]), x[:, :-1]], axis=1)

    *tokens, h0 = operands
    return (*map(later, tokens), h0)


REGIMES = ("mild", "slow", "steep", "decay")
"""Each draw's log decays a token (`draw`): Gated DeltaNet's (`-softplus(N(0, 1) - 1)`, as
tests/test_recurrence_mosaic.py draws them), a memory kept across chunks (`-U(0, 0.01)`), one forgotten within
a few tokens (`-U(1, 3)`), and a decay alone (-0.4 a token and no write, `beta = 0`), whose fp32 cumulative
sums within a chunk drift 1.5e-5 from the exact ones by its end (row 2066's defect's), where the decay's
enclosure is 7e-7 of it wide."""


FLOOR = -5.0
"""A decay a key dimension's declared least log decay a token (Kimi delta attention's), under which its pairs
factor over sub-chunks (`algebra.factored_range`)."""


def draw(seed: int, regime: str, *, chunks: int = 3, heads: int = 1, width: int = 64,
         per_key: bool = False, dtype=None):
    """A call's operands `(q, k, v, g, beta, h0)` of one row of `chunks` chunks, in `dtype` (bf16 by default):
    unit queries and keys,
    normal values, the regime's log decays (`REGIMES`), a write gate in (0, 1), an initial memory; `per_key`,
    a decay a key dimension in (`FLOOR`, 0) instead (Kimi delta attention's bounded gate)."""
    import jax.numpy as jnp

    rng = np.random.default_rng([2084, 3, seed, REGIMES.index(regime)])
    length = chunks * CHUNK

    def unit(shape):
        x = rng.normal(size=shape)
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    q, k = unit((1, length, 1, width)), unit((1, length, 1, width))
    v = rng.normal(size=(1, length, heads, width))
    shape = (1, length, heads)
    g = {"mild": lambda: -np.log1p(np.exp(rng.normal(size=shape) - 1.0)),
         "slow": lambda: -0.01 * rng.uniform(size=shape), "steep": lambda: -rng.uniform(1, 3, shape),
         "decay": lambda: np.full(shape, -0.4)}[regime]()
    if per_key:
        g = FLOOR / (1 + np.exp(1.0 - rng.normal(size=(*shape, width))))
    beta = 0.0 if regime == "decay" else 1 / (1 + np.exp(-rng.normal(size=shape)))
    h0 = 0.3 * rng.normal(size=(1, heads, width, width))
    dtype = jnp.bfloat16 if dtype is None else dtype
    return (jnp.asarray(q, dtype), jnp.asarray(k, dtype), jnp.asarray(v, dtype), jnp.asarray(g, jnp.float32),
            jnp.asarray(np.broadcast_to(beta, shape), dtype), jnp.asarray(h0, jnp.float32))


def shifted_refusals(operands: tuple, lane: Lane) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """An off-by-one chunk boundary (`refusals`): the call run on its tokens one later (`shifted`), each
    record held to the step on the call's own chunks' operands."""
    faulty = recorded(*shifted(operands))
    return refusals(faulty, dataclasses.replace(faulty, **hosted(operands)), lane)


def shared_decay_refusals(operands: tuple, lane: Lane) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """A decay a key dimension dropped for its head's mean (a kernel that reads one decay a head,
    `refusals`): the call run on the mean of each token's decays over its key dimensions, each record held
    to the step on the call's own per-key decays."""
    import jax.numpy as jnp

    q, k, v, g, beta, h0 = operands
    shared = jnp.broadcast_to(jnp.mean(g, -1, keepdims=True), g.shape)
    faulty = recorded(q, k, v, shared, beta, h0, floor=FLOOR)
    return refusals(faulty, dataclasses.replace(faulty, **hosted(operands)), lane)


def cumsum_refusals(operands: tuple, lane: Lane) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The decay from a difference of rounded cumulative sums (`refusals`, `rounded_cumsums`): the call run on
    those log decays, each record held to the step on its own."""
    import jax.numpy as jnp

    q, k, v, g, beta, h0 = operands
    faulty = recorded(q, k, v, jnp.asarray(rounded_cumsums(g)), beta, h0)
    return refusals(faulty, dataclasses.replace(faulty, g=hosted(operands)["g"]), lane)


def handed_refusals(operands: tuple, truth: Run, at: int, lane: Lane, *,
                    walked: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """A state rounded to bf16 where fp32 is held: the call over the chunks from `at` on (the fused inference
    walk's where `walked`), from the fp32 memory `truth`'s chunks before it leave rounded to bf16, its first
    chunk's leaving memory held to the step from the memory unrounded (`refusals`)."""
    import jax.numpy as jnp
    import ml_dtypes as dtypes

    carry = truth.carries[at]
    handed = np.asarray(np.asarray(carry, np.float32).astype(dtypes.bfloat16), np.float32)
    tail = recorded(*(x[:, at * CHUNK:] for x in operands[:5]), jnp.asarray(handed)[None], walked=walked)
    claimed = dataclasses.replace(tail, carries=np.concatenate([carry[None], tail.carries[1:]]))
    return refusals(tail, claimed, lane)["leaving"]


def dropped_refusals(operands: tuple, truth: Run, at: int, lane: Lane) -> tuple[np.ndarray, np.ndarray]:
    """A dropped carry: a document starting at chunk `at`'s first token (`starts`), whose stored entering
    memory is the zero a new document starts from, held to the memory the step before it leaves, rounded as
    stored; the fault's interval is that zero."""
    import jax.numpy as jnp
    from native_certificates import separated

    length = operands[0].shape[1]
    starts = jnp.zeros((1, length), bool).at[0, at * CHUNK].set(True)
    faulty = recorded(*operands, starts=starts, max_segments=2, carried=False)
    leaving = enclosed(truth, lane)["leaving"]
    stored = stored_memory(Iv(leaving.lo[at - 1], leaving.hi[at - 1]), lane)
    return within(faulty.h[at], stored), separated(stored, point(np.zeros_like(stored.lo)))


def assert_refusals(found: dict, shares: dict) -> None:
    """A fault's refusals by record (`refusals`): no record held where its intervals separate, and those more
    than `shares[record]` of each record where the fault is claimed to land
    (`native_certificates.assert_refused`)."""
    from native_certificates import assert_refused

    for name, (held, apart) in found.items():
        assert not np.any(held & apart), name
    for name, share in shares.items():
        assert_refused(*found[name], share)


DROPPED_SHARE = 0.95
"""The least share of the memory the first chunk leaves at which a dropped carry is refused (99.99% natively
on an A100 and an H100)."""
HANDED_SHARE = 0.6
"""The least share of the fp32 memory the second chunk leaves at which its entering memory rounded to bf16,
where fp32 is held, is refused (74% natively)."""
CUMSUM_SHARES = {"leaving": 0.5}
"""The least share of each chunk's leaving fp32 memory at which decays from a difference of fp32 cumulative
sums (row 2066's defect) are refused (75% natively; the 16-bit records' grid hides it)."""
SHIFTED_SHARES = {"w": 0.95, "c": 0.95, "o": 0.95}
"""The least shares of the weighted keys, corrected values and outputs at which each chunk's boundary a
token early is refused (99.9% natively)."""


def assert_carry_faults_refused(lane: Lane, heads: int | None = None) -> None:
    """A memory kept across chunks (`REGIMES`' 'slow'): a document starting at the second chunk, whose zero
    memory is refused at `DROPPED_SHARE` of the memory the first chunk leaves; and the second chunk run from
    that memory rounded to bf16 where fp32 is held, at `HANDED_SHARE` of the fp32 memory it leaves."""
    from native_certificates import assert_refused

    operands = draw(1, "slow", chunks=2, **({} if heads is None else {"heads": heads}))
    truth = recorded(*operands)
    assert_refused(*dropped_refusals(operands, truth, 1, lane), DROPPED_SHARE)
    assert_refused(*handed_refusals(operands, truth, 1, lane), HANDED_SHARE)


def assert_step_faults_refused(lane: Lane, heads: int | None = None) -> None:
    """Decays from a difference of fp32 cumulative sums in a decay alone, refused at `CUMSUM_SHARES`; and each
    chunk's boundary a token early, at `SHIFTED_SHARES`; no record held where its intervals separate."""
    sized = {} if heads is None else {"heads": heads}
    assert_refusals(cumsum_refusals(draw(1, "decay", chunks=2, **sized), lane), CUMSUM_SHARES)
    assert_refusals(shifted_refusals(draw(1, "mild", chunks=2, **sized), lane), SHIFTED_SHARES)


KEY_DECAY_SHARES = {"leaving": 0.5}
"""The least share of each chunk's leaving fp32 memory at which a decay a key dimension read as its head's
mean is refused."""


def assert_key_decays_witnessed(lane: Lane, heads: int = 1) -> None:
    """The step witness under a decay a key dimension (Kimi delta attention's, its floor's factored pairs): a
    bf16 call of three chunks, every record inside its step's enclosure, and a kernel reading each token's
    mean decay for all its keys refused (`shared_decay_refusals`, `KEY_DECAY_SHARES`)."""
    operands = draw(0, "mild", heads=heads, per_key=True)
    held = witness(recorded(*operands, floor=FLOOR), lane)
    for name in RECORDS:
        assert np.all(held[name]), (name, np.mean(held[name]))
    assert_refusals(shared_decay_refusals(operands, lane), KEY_DECAY_SHARES)


UNGATED_SHARES = {"leaving": 0.5}
"""The least share of each chunk's leaving fp32 memory at which a write gate on the keys' side read as one is
refused."""


def assert_keys_gated_witnessed(lane: Lane, heads: int = 1) -> None:
    """The step witness where the write gate takes the keys' side (`keys_gated_enclosures`, Mamba2's SSD under
    the fast forms): a bf16 call of three chunks without the delta rule, every record it keeps inside its
    step's enclosure, and a kernel reading every token's write gate as one refused (`UNGATED_SHARES`)."""
    import jax.numpy as jnp

    operands = draw(0, "mild", heads=heads)
    held = witness(recorded(*operands, delta=False), lane)
    for name in KEYS_GATED_RECORDS:
        assert np.all(held[name]), (name, np.mean(held[name]))
    q, k, v, g, beta, h0 = operands
    faulty = recorded(q, k, v, g, jnp.ones_like(beta), h0, delta=False)
    assert_refusals(refusals(faulty, dataclasses.replace(faulty, **hosted(operands)), lane), UNGATED_SHARES)


PARTS_SHARES = {"h": 0.9, "c": 0.9, "o": 0.8}
"""The least share of each record at which an fp32 call whose values are held in parts, run at bf16 instead,
is refused: its stored memories and corrected values (99% of them over two draws, interpreted at 5dd8e1788)
and its outputs (89%); the fp32 memory it leaves carries the call's own rounding as much (41-44%)."""


def assert_parts_witnessed(lane: Lane, heads: int = 1) -> None:
    """The step witness where the launches hold the values they pass in parts (`Lane.held`): an fp32 call of
    three chunks with the delta rule and without it (two parts under the fast forms), and a bf16 call without
    it unrounded (Lightning's, the accurate forms' SSD: its values in three parts), every record each keeps
    inside its step's enclosure; and the fp32 call run in bf16 refused against the fp32 call's enclosures."""
    import jax.numpy as jnp

    wide = draw(0, "mild", heads=heads, dtype=jnp.float32)
    for run in (recorded(*wide), recorded(*wide, delta=False), recorded(*draw(0, "mild", heads=heads),
                                                                        delta=False, rounded=False)):
        held = witness(run, lane)
        for name in WRITES[run.write]:
            assert np.all(held[name]), (run.write, run.held, name, np.mean(held[name]))
    q, k, v, g, beta, h0 = wide
    narrow = recorded(*(x.astype(jnp.bfloat16) for x in (q, k, v)), g, beta.astype(jnp.bfloat16), h0)
    claimed = recorded(*wide)
    held = {"write": claimed.write, "held": claimed.held, "given": claimed.given, "wide": True}
    as_claimed = dataclasses.replace(narrow, **hosted(wide, given=claimed.given), **held)
    assert_refusals(refusals(narrow, as_claimed, lane), PARTS_SHARES)


WALK_HANDED_SHARE = 0.55
"""The least share of the fp32 memory the walk's second chunk leaves at which its entering memory rounded to
bf16 is refused: 0.597 of it interpreted at a8dccec7a (the training call's states launch, `HANDED_SHARE`,
reads the stored memory too, which the walk never stores)."""


def assert_walk_witnessed(lane: Lane, heads: int = 1) -> None:
    """The step witness of the fused inference walk (`walk_enclosures`, `mainloop._scan`): a bf16 call of
    three chunks no gradient follows, with the delta rule and without it (its keys taking the write gate, and
    unrounded), every record each keeps inside its step's enclosure; and the walk run from a memory rounded to
    bf16 where fp32 is held refused (`handed_refusals`, `WALK_HANDED_SHARE`)."""
    operands = draw(0, "mild", heads=heads)
    for run in (recorded(*operands, walked=True), recorded(*operands, delta=False, walked=True),
                recorded(*operands, delta=False, rounded=False, walked=True)):
        held = witness(run, lane)
        for name in WALKED[run.write]:
            assert np.all(held[name]), (run.write, name, np.mean(held[name]))
    from native_certificates import assert_refused

    slow = draw(1, "slow", chunks=2, heads=heads)
    assert_refused(*handed_refusals(slow, recorded(*slow, walked=True), 1, lane, walked=True),
                   WALK_HANDED_SHARE)


class BackStep(NamedTuple):
    """One chunk and value head of a training call's fast backward under the delta rule
    (`mainloop_backward.kept_backward`), its inputs as the launches read them (float64, one value head's):
    the forward's `q`, `k`, `v`, `g`, `beta` and the records the forward kept (the inverse `a`, the weighted
    keys `w`, the corrected values `c` and the memory the chunk entered `h`), the output's gradient `do`, and
    the gradient of the memory the chunk leaves, `grad` (an interval: the next chunk's stored record, or the
    final memory's gradient as given)."""

    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    g: np.ndarray
    beta: np.ndarray
    a: np.ndarray
    w: np.ndarray
    c: np.ndarray
    h: np.ndarray
    do: np.ndarray
    grad: Iv
    dh: np.ndarray
    """The gradient of the memory the chunk leaves as stored (`Kept.dh`), which the query/key launch reads."""
    local: np.ndarray
    dv: np.ndarray
    ds: np.ndarray
    dw: np.ndarray


BACK_RECORDS = ("local", "dv", "entering", "ds", "dq", "dk", "dw", "d_l", "values", "keys")
"""The records the delta rule's fast backward keeps, each a launch's (`Kept`): `_local_back`'s `local`,
`_reverse`'s `dv` and the gradient of the memory each chunk enters (`entering`), `_scores_back`'s `ds`,
`_query_key`'s `dq`, `dk`, `dw`, `_system_back`'s `d_l`, `_values_back`'s `values` and `_keys_back`'s
`keys`."""


def tr(iv: Iv) -> Iv:
    """`iv` transposed."""
    return Iv(iv.lo.T, iv.hi.T)


def back_enclosures(step: BackStep, scale: float, lane: Lane) -> dict[str, Iv]:
    """Each launch's records of a chunk of the delta rule's fast backward (fla's DAG, its rounding:
    `mainloop_backward.gradients`), from the inputs it reads (`BackStep`), each register operand staged as
    the call holds it (`staged`) and each sum of products a `dot`'s:

    - `local`: `A^T do`, `A[t, s] = scale (q_t . k_s) exp2(E_t - E_s)` (`s <= t`), its transpose's product
      `k q^T` decayed and scaled in fp32 and staged (`algebra.local_gradient`);
    - `dv`: `(k grad) exp2(E_last - E) + local` (`algebra.reverse_carried`), `grad` staged;
    - `entering`: `grad exp2(E_last) + scale q^T (do exp2(E)) - w^T dv`, the memory's gradient into the chunk;
    - `ds`: `do c^T` decayed, causal and scaled, in the held form (`algebra.scores_gradient`);
    - `dq`, `dk`, `dw`: `scale (do h^T) exp2(E) + ds k`, `(c grad^T) exp2(E_last - E) + ds^T q`,
      `-dv h^T` (`algebra.memory_gradients`);
    - `d_l`: `-(X^T tril(dX, -1) X^T)` decayed, `dX = dw (k beta exp2 E)^T + dv (v beta)^T`, `X = I + a`
      (`algebra.system_gradient`);
    - `values`: `(X^T dv) beta` (`algebra.values_gradient`);
    - `keys`: `(X^T dw) exp2(E) beta + (d_l k) beta + (beta d_l)^T k` (`algebra.keys_gradient`)."""
    exponents = Exponents.of(step.g)
    raised, gated, pairs = exponents.raised(lane), exponents.gated(lane), exponents.pairs(lane)
    factor = point(np.float32(scale))
    zeros = point(np.zeros((CHUNK, CHUNK)))
    rows = Iv(raised.lo[:, None], raised.hi[:, None])
    ranged = Iv(gated.lo[:, None], gated.hi[:, None])
    q, k, v, do = point(step.q), point(step.k), point(step.v), point(step.do)
    beta = np.asarray(step.beta)[:, None]
    lag = _lags()

    transposed = fp32(mul(blocked(k, tr(q), lane), tr(pairs)), lane)
    scores = staged(where(lag.T >= 0, fp32(mul(transposed, factor), lane), zeros), lane)
    local = dot(scores, do, lane)
    dv = fp32(add(fp32(mul(blocked(k, staged(step.grad, lane), lane), ranged), lane), point(step.local)),
              lane)
    held = Iv(np.full(step.grad.lo.shape, raised.lo[-1]), np.full(step.grad.lo.shape, raised.hi[-1]))
    read = staged(fp32(mul(do, rows), lane), lane)
    reads = dot(tr(q), read, lane)
    entering = fp32(add(fp32(add(fused(mul(step.grad, held), lane), fused(mul(reads, factor), lane)), lane),
                        neg(dot(tr(point(step.w)), point(step.dv), lane))), lane)
    ds = fp32(mul(fp32(mul(blocked(do, tr(point(step.c)), lane), pairs), lane), factor), lane)
    ds = staged(where(lag >= 0, ds, zeros), lane)
    narrow = point(step.ds)
    entered = tr(point(step.h))
    dq = fp32(add(fp32(mul(fp32(mul(blocked(do, entered, lane), rows), lane), factor), lane),
                  dot(narrow, k, lane)), lane)
    dk = fp32(mul(dot(point(step.c), tr(point(step.dh)), lane), ranged), lane)
    dk = fp32(add(dk, dot(tr(narrow), q, lane)), lane)
    dw = neg(dot(point(step.dv), entered, lane))
    lower_t = tr(point(step.a))
    weighting = fp32(mul(point(np.asarray(step.beta)), raised), lane)
    keyed = staged(fp32(mul(k, Iv(weighting.lo[:, None], weighting.hi[:, None])), lane), lane)
    valued = staged(fp32(mul(v, point(beta)), lane), lane)
    d_x = dot(point(step.dv), tr(valued), lane, into=dot(point(step.dw), tr(keyed), lane))
    d_x = where(lag > 0, d_x, zeros)
    d_x = fp32(add(d_x, dot(staged(d_x, lane), lower_t, lane)), lane)
    d_x = fp32(add(d_x, dot(lower_t, staged(d_x, lane), lane)), lane)
    d_l = staged(where(lag > 0, neg(fp32(mul(d_x, pairs), lane)), zeros), lane)
    du = point(step.dv)
    values = fp32(mul(fp32(add(du, dot(lower_t, staged(du, lane), lane)), lane), point(beta)), lane)
    dw_held = point(step.dw)
    through = fp32(mul(fp32(add(dw_held, dot(lower_t, staged(dw_held, lane), lane)), lane), rows), lane)
    combined = fp32(add(through, dot(d_l, k, lane)), lane)
    weighted = staged(fp32(mul(d_l, point(beta)), lane), lane)
    keys = fp32(add(fp32(mul(combined, point(beta)), lane), dot(tr(weighted), k, lane)), lane)
    return {"local": local, "dv": dv, "entering": entering, "ds": ds, "dq": dq, "dk": dk, "dw": dw,
            "d_l": d_l, "values": values, "keys": keys}


class BackRun(NamedTuple):
    """A training call's fast backward under the delta rule as its launches kept it (`recorded_backward`):
    its forward's run (`Run`), the output's and final memory's gradients `do` `[N, H, C, Dv]` and `dfinal`
    `[H, Dk, Dv]`, and each launch's records by name (`BACK_RECORDS`), `[N, H, ...]` float64 (the gradient
    of the memory each chunk leaves `dh`, `[N, H, Dk, Dv]`, beside them), with the dtype each is stored in."""

    run: Run
    do: np.ndarray
    dfinal: np.ndarray
    dh: np.ndarray
    dh0: np.ndarray
    records: dict
    dtypes: dict

    def step(self, n: int, head: int, lane: Lane) -> BackStep:
        """Chunk `n` and value head `head`'s inputs, the gradient of the memory it leaves the final memory's
        as given for the last, else what the next chunk's stored record holds (`held_cell`)."""
        run, records = self.run, self.records
        last = n == run.chunks - 1
        grad = point(self.dfinal[head]) if last else held_cell(self.dh[n, head], lane)
        return BackStep(run.q[n, head], run.k[n, head], run.v[n, head], run.g[n, head], run.beta[n, head],
                        run.a[n, head], run.w[n, head], run.c[n, head], run.h[n, head], self.do[n, head],
                        grad, self.dh[n, head], records["local"][n, head], records["dv"][n, head],
                        records["ds"][n, head], records["dw"][n, head])


def recorded_backward(operands: tuple, seed: int = 0) -> BackRun:
    """The fast backward of a 16-bit call under the delta rule over one row (`draw`'s `operands`), run as a
    training step runs it (the route's VJP), and the same build's `mainloop_backward.kept_backward` at the
    forward's kept values, checked to give the step's gradients bit for bit: each launch's outputs as stored
    (`Kept`), laid a chunk and value head at a time, for output and final memory gradients `seed` draws."""
    import jax
    import jax.numpy as jnp

    from dew.nn.kernels import generation, recurrence_chunk, tuning
    from dew.nn.kernels.templates import bundles
    from dew.nn.kernels.templates.recurrence import mainloop, mainloop_backward, template

    q, k, v, _, _, h0 = operands
    dtype = jnp.result_type(q, k, v)
    run = recorded(*operands)
    rng = np.random.default_rng([2084, seed])
    do = jnp.asarray(rng.standard_normal(v.shape), dtype)
    dfinal = jnp.asarray(rng.standard_normal(h0.shape), jnp.float32)
    scale = k.shape[-1] ** -0.5
    rounded = recurrence_chunk.chunk_rounding(delta=True, dtype=dtype)
    family = mainloop.family_of(dtype) or bundles.INTERPRETED
    interpret = generation.backend() != "gpu"

    def route(*x):
        return recurrence_chunk.chunk_gated_delta_rule(*x, write_bound=1.0)

    with jax.default_matmul_precision(None), generation.chosen(gated_delta_rule="mosaic_gpu"):
        with tuning.seen() as launches:
            native = jax.jit(lambda *x: jax.vjp(route, *x)[1]((do, dfinal)))(*operands)
        [tiles] = [config for (op, _), (config, _) in zip(launches, launches.taken, strict=True)
                   if isinstance(op, template.Mosaic)]

        def kept(*x):
            _, _, (inverse, held) = mainloop.kept_forward(*x, tiles=tiles, scale=scale, rounded=rounded,
                                                         family=family, interpret=interpret)
            return mainloop_backward.kept_backward(*x, do, dfinal, tiles=tiles, scale=scale, rounded=rounded,
                                                   family=family, interpret=interpret, inverse=inverse,
                                                   kept=held)

        grads, saved = jax.jit(kept)(*operands)
    for mine, theirs in zip(grads[:6], native, strict=True):
        assert np.array_equal(np.asarray(mine).view(np.uint8), np.asarray(theirs).view(np.uint8)), \
            "the certify build is the production program"
    _, length, heads, value_width = v.shape
    key_width, chunks = k.shape[-1], run.chunks
    laid = np.shape(saved.dh)[0] // (heads * key_width)

    def chunked(x, width):
        rows = np.asarray(x, np.float64).reshape(-1, laid * CHUNK, heads * width).sum(0)
        return rows.reshape(laid, CHUNK, heads, width).transpose(0, 2, 1, 3)[:chunks]

    named = {"local": (saved.local, value_width), "dv": (saved.dv, value_width), "ds": (saved.ds, CHUNK),
             "dq": (saved.dq, key_width), "dk": (saved.dk, key_width), "dw": (saved.dw, key_width),
             "d_l": (saved.d_l, CHUNK), "values": (saved.values, value_width),
             "keys": (saved.keys, key_width)}
    records = {name: chunked(x, width) for name, (x, width) in named.items()}
    dtypes = {name: jnp.dtype(x.dtype) for name, (x, _) in named.items()}
    dh = np.asarray(saved.dh, np.float64).reshape(laid, heads, key_width, value_width)[:chunks]
    dtypes["entering"] = jnp.dtype(saved.dh.dtype)
    hosted_do = _laid(np.asarray(do, np.float64)[0], laid_chunks(length, None, None))
    return BackRun(run, hosted_do.reshape(chunks, CHUNK, heads, value_width).transpose(0, 2, 1, 3),
                   np.asarray(dfinal, np.float64)[0], dh,
                   np.asarray(saved.dh0, np.float64).reshape(heads, key_width, value_width), records, dtypes)


def back_witnessed(back: BackRun, lane: Lane) -> dict[str, np.ndarray]:
    """Whether each record of `back` lies within its launch's enclosure (`back_enclosures`), as stored (in
    its record's dtype): `[N, H, ...]` each, the gradient each chunk's memory enters with the record before
    it (the initial memory's, `dh0`, for the first)."""
    def stored(iv: Iv, dtype) -> Iv:
        return iv if dtype == np.float32 else cast(iv, {"bfloat16": BFLOAT16, "float16": FLOAT16}[str(dtype)])

    found: dict[str, list] = {}
    for n in range(back.run.chunks):
        for head in range(back.run.heads):
            step = back_enclosures(back.step(n, head, lane), back.run.scale, lane)
            for name in BACK_RECORDS:
                if name == "entering":
                    record = back.dh0[head] if n == 0 else back.dh[n - 1, head]
                    dtype = np.float32 if n == 0 else back.dtypes["entering"]
                else:
                    record, dtype = back.records[name][n, head], back.dtypes[name]
                found.setdefault(name, []).append(within(record, stored(step[name], dtype)))
    return {name: np.stack(held) for name, held in found.items()}


def assert_backward_witnessed(lane: Lane, heads: int = 1) -> None:
    """The step witness of the delta rule's fast backward (`back_enclosures`): a bf16 call of three chunks,
    every record each launch keeps inside its step's enclosure."""
    back = recorded_backward(draw(0, "mild", heads=heads))
    held = back_witnessed(back, lane)
    for name in BACK_RECORDS:
        assert np.all(held[name]), (name, np.mean(held[name]))
