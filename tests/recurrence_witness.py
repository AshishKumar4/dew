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

Each enclosure is outward interval arithmetic in float64, each end snapped to its op's own grid: products of
16-bit values (or of an fp32 operand's three bf16 parts, six of them a pair, `product.formed`) exact, summed
as the native-dot contract holds a tensor core's sum (gamma_n at a truncating add's unit, of the terms that
may be nonzero); fp32 adds and multiplies rounded to nearest, a product before an add rounded or exact
(an FMA's contraction); casts to nearest; `exp2` within `reach` steps of the fp32 grid of its correctly
rounded value (PTX ISA 9.4 §9.7.3.21: ex2.approx.f32's 2 ulp, the mainloop's `approx_math`); and each log
decay's base-2 prefix within `EXPONENT_BOUND` of its exact value (XLA's compensated scan, `mainloop.prefix`),
then the kernel's own roundings of its differences."""

import dataclasses
import math
from typing import NamedTuple

import ml_dtypes
import numpy as np
from op_certificates import BFLOAT16, FLOAT32, Format, rounded

FLOAT16 = Format("float16", 11, -14, 15)
"""fp16, the inverse's merges' staged dtype in a 16-bit call (`mainloop._merges`)."""

DTYPES = {"float32": np.float32, "bfloat16": ml_dtypes.bfloat16, "float16": np.float16}
CHUNK = 64
LOG2E = math.log2(math.e)
EXPONENT_BOUND = 2.0 ** -38
"""How far a position's prefix `hi + lo` (`mainloop.prefix`) lies from its exact `log2(e) cumsum(g)`, as a
share of `log2(e) sum |g|` over the positions it spans: each compensated add rounds two adds of its
remainders, each within 2^-23 of their sum, at most 2^-24 of the running value each; an element of a 64-long
`associative_scan` takes at most 2 log2(64) such adds, under 2^-41 in all, and each token's exponent
`g log2(e)` is a Dekker product and its rounded remainder, within 2^-46 of it. Four times their sum."""
FRACTION = 2.0 ** -22
"""What `algebra.raised` rounds off its exponent: the fraction `(whole - floor(whole)) + part`, under 2,
rounded twice to nearest."""
PAIR_FRACTION = 2.0 ** -21
"""What `algebra.decayed` rounds off a pair's exponent: each side's fraction (where the side is under one, its
distance from its floor rounds), their difference and its sum with the low halves', each under 2."""
PAIR_FLOOR = -124
"""`algebra.PAIR_FLOOR`: a pair whose integers differ by less weighs zero."""


class Iv(NamedTuple):
    """Each element's closed interval of values, `[lo, hi]` (float64)."""
    lo: np.ndarray
    hi: np.ndarray


def point(x) -> Iv:
    x = np.asarray(x, np.float64)
    return Iv(x, x)


def _outward(lo, hi) -> Iv:
    return Iv(np.nextafter(lo, -np.inf), np.nextafter(hi, np.inf))


def nearest(x, fmt: Format) -> np.ndarray:
    x = np.asarray(x, np.float64)
    return rounded(x, np.zeros_like(x), fmt)


def _toward(x, fmt: Format, direction: float) -> np.ndarray:
    """The next value of `fmt` after each of its values `x` toward `direction`."""
    dtype = DTYPES[fmt.name]
    x = np.asarray(x, np.float64).astype(dtype)
    return np.nextafter(x, np.full(x.shape, direction, dtype)).astype(np.float64)


def below(x, fmt: Format) -> np.ndarray:
    """The greatest value of `fmt` at or below each `x`."""
    found = nearest(x, fmt)
    return np.where(found > x, _toward(found, fmt, -np.inf), found)


def above(x, fmt: Format) -> np.ndarray:
    """The least value of `fmt` at or above each `x`."""
    found = nearest(x, fmt)
    return np.where(found < x, _toward(found, fmt, np.inf), found)


def onto(iv: Iv, fmt: Format) -> Iv:
    """The values of `fmt` within `iv`: what a value known to be on `fmt`'s grid takes."""
    return Iv(above(iv.lo, fmt), below(iv.hi, fmt))


def cast(iv: Iv, fmt: Format) -> Iv:
    """A cast to nearest of a value within `iv` (monotone)."""
    return Iv(nearest(iv.lo, fmt), nearest(iv.hi, fmt))


def cell(x, fmt: Format) -> Iv:
    """The values that round to nearest to each `x` of `fmt`: between its midpoints with its neighbours."""
    x = np.asarray(x, np.float64)
    return Iv((x + _toward(x, fmt, -np.inf)) / 2, (x + _toward(x, fmt, np.inf)) / 2)


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
    its correctly rounded value), whether fp32 results below the normal range flush to zero, and whether
    the call's values are fp32 (three bf16 parts) or bf16."""
    reach: int
    flushes: bool
    wide: bool

    @property
    def held(self) -> Format | None:
        """The format the launches store their values in: bf16, or None where fp32 is stored exactly."""
        return None if self.wide else BFLOAT16


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


def staged(iv: Iv, lane: Lane) -> Iv:
    """A register value staged for a product as the call's launches hold it: cast to bf16, or an fp32 value's
    three parts, exactly."""
    return iv if lane.held is None else cast(iv, lane.held)


def aligned(iv: Iv, fmt: Format) -> np.ndarray:
    """The largest magnitude a tensor core aligns a product by of an operand within `iv`
    (`native_certificates.aligned`: a subnormal of `fmt` at `fmt`'s least normal)."""
    from native_certificates import aligned as at

    return np.maximum(at(iv.lo, fmt), at(iv.hi, fmt))


def dot(a: Iv, b: Iv, lane: Lane, fmt: Format = BFLOAT16) -> Iv:
    """`a @ b` (`[M, K] @ [K, N]`) as one product the tensor cores accumulate in fp32: each term's exact range
    (products of 16-bit values of `fmt`, or of three bf16 parts an fp32 operand's, the six whose orders sum
    under three, within 2^-24 of the fp32 values' product), summed within the native-dot contract's bound
    (`native_certificates.native_dot_holds`: gamma_n at a truncating add's unit, n the terms that may be
    nonzero, an add of an exact zero exact), the float64 sums within n 2^-52 of their own. The bound's magnitudes are the
    terms' as the tensor cores align them (`aligned`): a subnormal operand's at `fmt`'s least normal, the
    exponent its field holds."""
    terms = mul(Iv(a.lo[:, None, :], a.hi[:, None, :]), Iv(b.lo.T[None, :, :], b.hi.T[None, :, :]))
    parts = 6 if lane.wide else 1
    live = ((terms.lo != 0) | (terms.hi != 0)).sum(-1) * parts
    magnitude = (aligned(a, fmt)[:, None, :] * aligned(b, fmt).T[None, :, :]).sum(-1) * (1 + 2.0 ** -50)
    unit = 2.0 ** -(FLOAT32.precision - 1)  # a truncating add's
    gammas = np.nextafter(live * unit / (1 - live * unit), np.inf)
    underflow = 2.0 ** FLOAT32.emin if lane.flushes else 2.0 ** (FLOAT32.emin - FLOAT32.precision + 1)
    dropped = 2.0 ** -24 if lane.wide else 0.0
    allowance = (gammas + dropped + (live + 2) * 2.0 ** -52) * magnitude + live * underflow
    return onto(_outward(terms.lo.sum(-1) - allowance, terms.hi.sum(-1) + allowance), FLOAT32)


def blocked(a: Iv, b: Iv, lane: Lane, block: int = CHUNK) -> Iv:
    """`a @ b` as the mainloop issues it over its contraction's `block`-wide column blocks: each block's
    product a fresh accumulator (`dot`), summed in fp32 from zero (`sum`'s first add exact)."""
    width = a.lo.shape[1]
    total = None
    for start in range(0, width, block):
        part = dot(Iv(a.lo[:, start:start + block], a.hi[:, start:start + block]),
                   Iv(b.lo[start:start + block], b.hi[start:start + block]), lane)
        total = part if total is None else fp32(add(total, part), lane)
    return total


def exp2(x: Iv, lane: Lane) -> Iv:
    """`exp2` of a value within `x`: within `lane.reach` steps of the fp32 grid of its correctly rounded
    value, the float64 `exp2` within 2^-50 of it; zero where it may fall below the normal range (ex2's flush,
    or a scaling's last product rounding there)."""
    lo = below(np.exp2(x.lo) * (1 - 2.0 ** -50), FLOAT32)
    hi = above(np.exp2(x.hi) * (1 + 2.0 ** -50), FLOAT32)
    for _ in range(lane.reach):
        lo, hi = _toward(lo, FLOAT32, -np.inf), _toward(hi, FLOAT32, np.inf)
    return Iv(np.where(lo < 2.0 ** FLOAT32.emin, 0.0, lo), hi)


@dataclasses.dataclass(frozen=True)
class Exponents:
    """A chunk's positions' base-2 log decays from its start, `log2(e) cumsum(g)` (`exact`, float64 within
    2^-50 of it), and how far the kernel's prefix may lie from each (`bound`, `EXPONENT_BOUND`)."""
    exact: np.ndarray
    bound: np.ndarray

    @classmethod
    def of(cls, g) -> "Exponents":
        g = np.asarray(g, np.float64)
        exact = np.array([math.fsum(g[:s + 1]) for s in range(len(g))]) * LOG2E
        spanned = np.cumsum(np.abs(g)) * LOG2E * (1 + 2.0 ** -40)
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
        """`algebra.decayed(hi, lo, hi, lo)` `[C, C]`: each pair's decay `exp2(E_s - E_t)`, zero where its
        integers' difference may pass `PAIR_FLOOR`, the upper triangle's left to the caller's mask."""
        difference = self.exact[:, None] - self.exact[None, :]
        spread = (self.bound[:, None] + self.bound[None, :]
                  + 2.0 ** -46 * (np.abs(self.exact[:, None]) + np.abs(self.exact[None, :])) + PAIR_FRACTION)
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
    product = blocked(point(k), point(np.asarray(k).T), lane)
    decayed = fp32(mul(product, exponents.pairs(lane)), lane)
    system = fp32(mul(decayed, point(np.asarray(beta)[:, None])), lane)
    lags = _lags()
    system = where(lags > 0, system, point(np.zeros((CHUNK, CHUNK))))
    lower_a = neg(system)
    rows, cols = np.indices((CHUNK, CHUNK))
    identity = (rows == cols).astype(np.float64)
    first = (rows % 2 == 1) & (cols == rows - 1)
    current = Iv(identity + np.where(first, lower_a.lo, 0.0), identity + np.where(first, lower_a.hi, 0.0))

    def merge(x: Iv) -> Iv:
        return x if lane.wide else cast(x, FLOAT16)

    width = 2
    while width < CHUNK:
        lower = ((rows // width) % 2 == 1) & (cols // width == rows // width - 1)
        staged16 = BFLOAT16 if lane.wide else FLOAT16
        power = dot(merge(current), merge(lower_a), lane, staged16)
        masked = where(lower, power, point(np.zeros((CHUNK, CHUNK))))
        joined = dot(merge(masked), merge(current), lane, staged16)
        current = fp32(add(current, joined), lane)
        width *= 2
    return Iv(current.lo - identity, current.hi - identity)


def weighted(lower, tile, beta, decay: Iv | None, lane: Lane) -> tuple[Iv, Iv]:
    """`algebra.weighted` from the stored inverse `lower` (the operand its product stages): `(fp32 value,
    stored)`, the scaled tile `tile beta [decay]` (each multiply rounded) on the diagonal outside the
    product."""
    scaled = fp32(mul(point(tile), point(np.asarray(beta)[:, None])), lane)
    if decay is not None:
        scaled = fp32(mul(scaled, Iv(decay.lo[:, None], decay.hi[:, None])), lane)
    value = fp32(add(scaled, dot(point(lower), staged(scaled, lane), lane)), lane)
    return value, staged(value, lane)


def corrected(values: Iv, weights, entering, lane: Lane) -> Iv:
    """`algebra.corrected`'s fp32 value, `u - w S`: the stored weighted values `values`, the stored weighted
    keys, and the memory entering staged as the states launch stores it (`entering`, its key slices each a
    fresh product summed from zero)."""
    return fp32(add(values, neg(blocked(point(weights), point(entering), lane))), lane)


def carried(carry: Iv, values: Iv, k, exponents: Exponents, lane: Lane) -> Iv:
    """The memory leaving the chunk, fp32 (`algebra.scalar_write`, `carried`): the carry decayed to the
    chunk's end plus the keys against the corrected values gated to it, each staged as the call holds it; an
    fp32 call's last token's write outside the product (`mainloop._last_write`)."""
    gate = exponents.gated(lane)
    gated = fp32(mul(values, Iv(gate.lo[:, None], gate.hi[:, None])), lane)
    keys = point(np.asarray(k).T)
    if lane.wide:
        earlier = where(np.arange(CHUNK)[:, None] < CHUNK - 1, gated, point(np.zeros(gated.lo.shape)))
        last = mul(Iv(keys.lo[:, -1:], keys.hi[:, -1:]), Iv(gated.lo[-1:], gated.hi[-1:]))
        write = fp32(add(dot(keys, earlier, lane), fused(last, lane)), lane)
    else:
        write = dot(keys, staged(gated, lane), lane)
    decay = exponents.raised(lane)
    held = Iv(np.full(carry.lo.shape, decay.lo[-1]), np.full(carry.lo.shape, decay.hi[-1]))
    return fp32(add(fused(mul(carry, held), lane), write), lane)


def output(q, k, entering, values, exponents: Exponents, scale: float, lane: Lane) -> Iv:
    """The chunk's fp32 output before its store (`algebra.chunk_scores`, `chunk_output`): its own pairs'
    scores staged as the call holds them, the memory it entered read through the decayed queries, each term
    scaled by fp32's `scale` and their sum rounded once, either product before it maybe contracted."""
    scores = fp32(mul(blocked(point(q), point(np.asarray(k).T), lane), exponents.pairs(lane)), lane)
    scores = staged(where(_lags() >= 0, scores, point(np.zeros((CHUNK, CHUNK)))), lane)
    memory = blocked(point(q), point(entering), lane)
    decay = exponents.raised(lane)
    memory = fp32(mul(memory, Iv(decay.lo[:, None], decay.hi[:, None])), lane)
    factor = point(np.float32(scale))
    local = dot(scores, point(values), lane)
    return fp32(add(fused(mul(memory, factor), lane), fused(mul(local, factor), lane)), lane)


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


def enclosures(step: Step, scale: float, lane: Lane) -> dict[str, Iv]:
    """Each record's enclosure in its stored values (`a`, `w`, `c`, `o`) and the memory leaving the chunk,
    fp32 (`h`, the next chunk's entering memory before its store's rounding), each from the native records
    its launch reads (the module's docstring)."""
    exponents = Exponents.of(step.g)
    _, w = weighted(step.a, step.k, step.beta, exponents.raised(lane), lane)
    _, u = weighted(step.a, step.v, step.beta, None, lane)
    c = corrected(u, step.w, step.h, lane)
    # The corrected values the write takes: fp32, within their enclosure and rounding to the stored `c`.
    values = c if lane.held is None else meet(c, cell(step.c, lane.held))
    stored = staged(inverse(step.k, step.beta, exponents, lane), lane)
    return {"a": stored, "w": w, "c": staged(c, lane),
            "h": carried(step.carry, values, step.k, exponents, lane),
            "o": staged(output(step.q, step.k, step.h, step.c, exponents, scale, lane), lane)}


def stored_memory(iv: Iv, lane: Lane) -> Iv:
    """The memory as the states launch stores it: bf16, or fp32 exactly."""
    return staged(iv, lane)


def _index(x) -> np.ndarray:
    """Each fp32 value's place on fp32's grid, in steps from zero."""
    bits = np.abs(np.asarray(x, np.float64)).astype(np.float32).view(np.uint32).astype(np.int64)
    return np.where(np.signbit(np.asarray(x, np.float64)), -bits, bits)


def ulps(iv: Iv, record, fmt: Format) -> np.ndarray:
    """Each enclosure's width in units of `fmt`'s last place at its record (at the least normal's for a zero):
    how far apart two values it admits may lie, on the record's own grid."""
    record = np.abs(np.asarray(record, np.float64))
    _, exponent = np.frexp(np.maximum(record, 2.0 ** fmt.emin))
    return (iv.hi - iv.lo) / np.ldexp(1.0, exponent - fmt.precision)


def exp2_reach(native) -> int:
    """The most steps of the fp32 grid `native(x)`, an fp32 `exp2` of fp32 operands, lies from `2^x` correctly
    rounded (`op_certificates.exp2_rounded`), over the fractions `algebra.raised` and `algebra.decayed` take,
    (-2, 2): 2^17 of them, densely and drawn, the integers there (exact cases) among them."""
    from op_certificates import exp2_rounded

    rng = np.random.default_rng(2084)
    x = np.concatenate([np.linspace(-2, 2, 1 << 16, dtype=np.float32), np.arange(-2, 3, dtype=np.float32),
                        rng.uniform(-2, 2, (1 << 16) - 5).astype(np.float32)])
    return int(np.max(np.abs(_index(np.asarray(native(x), np.float64)) - _index(exp2_rounded(x, FLOAT32)))))


RECORDS = ("a", "w", "c", "o", "h", "leaving")
"""The records the witness holds (`enclosed`)."""


def interpreted_lane() -> Lane:
    """The arithmetic Mosaic GPU's interpreter runs a launch in, a CPU process's XLA: its `exp2`'s reach
    (`exp2_reach`) and whether it flushes (`certified.lane`), a 16-bit call."""
    import jax
    import jax.numpy as jnp
    from certified import lane

    return Lane(exp2_reach(lambda x: jax.jit(jnp.exp2)(jnp.asarray(x))), lane().flushes, wide=False)


@dataclasses.dataclass(frozen=True)
class Run:
    """A call's operands as its launches read them and its records, one row: `q`, `k` `[T, Hk, Dk]`, `v` `[T,
    H, Dv]`, `g`, `beta` `[T, H]` (float64 of their dtypes' values); each chunk and head's stored inverse less
    its diagonal `a` `[N, H, C, C]`, weighted keys `w` `[N, H, C, Dk]`, entering memory `h` `[N, H, Dk, Dv]`,
    corrected values `c` and output `o` `[N, H, C, Dv]`, an fp32 call's three parts summed; and the fp32
    memory each chunk leaves, `carries` `[N + 1, H, Dk, Dv]` from the initial memory: the final memory of the
    call over the chunks before it, at the same tiles (`recorded`), the state the states launch carries in
    fp32 and stores rounded as `h` (`seams`)."""
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
    seams: np.ndarray

    @property
    def chunks(self) -> int:
        return self.a.shape[0]

    @property
    def heads(self) -> int:
        return self.a.shape[1]

    def step(self, n: int, head: int, *, carry: Iv | None = None) -> Step:
        """Chunk `n` and value head `head`'s step: its operands, its records and the fp32 memory it enters
        (`carries`), unless given."""
        rows, key = slice(n * CHUNK, (n + 1) * CHUNK), head // (self.heads // self.k.shape[1])
        return Step(self.q[rows, key], self.k[rows, key], self.v[rows, head], self.g[rows, head],
                    self.beta[rows, head], self.a[n, head], self.w[n, head], self.h[n, head], self.c[n, head],
                    point(self.carries[n, head]) if carry is None else carry)


def recorded(q, k, v, g, beta, h0, *, starts=None, max_segments: int | None = None,
             carried: bool = True) -> Run:
    """The records of the gated delta rule's call on the template, forced (`generation.chosen`) at JAX's own
    default precision as a model runs it: `mainloop.kept_forward` at the tiles the route takes
    (`tuning.seen`), its output and final memory bit for bit the route's own (the certify build is the
    production program), on the device's bundle, interpreted off a GPU. Operands `[1, T, ...]`. With
    `carried`, each chunk's fp32 memory leaving it: the same launches at the same tiles over the chunks up to
    it, whose final memory, rounded as the launch stores `h`, is the next chunk's `h` bit for bit (`seams`,
    each chunk's: end_j == start_{j+1}); without, the initial and the final memories alone."""
    import jax
    import jax.numpy as jnp

    from dew.nn.kernels import generation, recurrence_chunk, tuning
    from dew.nn.kernels.templates.recurrence import mainloop, template

    operands = (q, k, v, g, beta, h0)
    dtype = jnp.result_type(q, k, v)
    with jax.default_matmul_precision(None), generation.chosen(gated_delta_rule="mosaic_gpu"):
        with tuning.seen() as launches:
            o, final = jax.jit(lambda *x: recurrence_chunk.chunk_gated_delta_rule(
                *x, starts=starts, max_segments=max_segments))(*operands)
        [tiles] = [config for (op, _), (config, _) in zip(launches, launches.taken, strict=True)
                   if isinstance(op, template.Mosaic)]

        def kept(*x, starts=starts):
            return jax.jit(lambda *x: mainloop.kept_forward(
                *x, tiles=tiles, starts=starts, max_segments=max_segments, scale=k.shape[-1] ** -0.5,
                family=mainloop.family_of(dtype) or mainloop.INTERPRETED,
                interpret=generation.backend() != "gpu"))(*x)

        kept_o, kept_final, (a, (w, h, c)) = kept(*operands)
        for native, own in ((o, kept_o), (final, kept_final)):
            assert np.array_equal(np.asarray(native).view(np.uint8), np.asarray(own).view(np.uint8)), \
                "the certify build is the production program"
        _, length, heads, value_width = v.shape
        key_width, chunks = k.shape[-1], length // CHUNK
        ends = [kept(*(x[:, :n * CHUNK] for x in operands[:5]), h0,
                     starts=None if starts is None else starts[:, :n * CHUNK])[1]
                for n in range(1, chunks)] if carried else []
    parts = 3 if jnp.dtype(dtype) == jnp.float32 else 1
    # The chunks the launches lay out (a packed call's every sequence's, its worst case's), of which the
    # call's come first where its documents start on the chunks' grid.
    laid = np.shape(a)[0] // parts // CHUNK

    def summed(x, rows):
        return np.asarray(x, np.float64).reshape(parts, rows, -1).sum(0)

    def chunked(x, width):
        return summed(x, laid * CHUNK).reshape(laid, CHUNK, heads, width).transpose(0, 2, 1, 3)[:chunks]

    memories = summed(h, laid * heads * key_width).reshape(laid, heads, key_width, value_width)[:chunks]
    stored = jnp.float32 if parts == 3 else dtype
    seams = np.asarray([np.array_equal(np.asarray(np.asarray(end[0]).astype(stored), np.float64),
                                       memories[n + 1]) for n, end in enumerate(ends)], bool)
    carries = [np.asarray(h0, np.float64)[0], *(np.asarray(end, np.float64)[0] for end in ends)]
    unknown = np.full(carries[0].shape, np.nan)
    carries += [unknown] * (chunks - len(carries)) + [np.asarray(kept_final, np.float64)[0]]
    return Run(**hosted(operands), a=chunked(a, CHUNK), w=chunked(w, key_width), h=memories,
               c=chunked(c, value_width),
               o=np.asarray(kept_o, np.float64)[0].reshape(chunks, CHUNK, heads, value_width).transpose(
                   0, 2, 1, 3),
               carries=np.stack(carries), seams=seams)


def hosted(operands: tuple) -> dict[str, np.ndarray]:
    """A call's operands `(q, k, v, g, beta, h0)` as its launches read them, one row's, float64 (`Run`'s): the
    queries, keys and values in their common dtype (fp32's three parts summed exactly), the write gate rounded
    to it, the log decay fp32."""
    import jax.numpy as jnp

    q, k, v, g, beta, _ = operands
    dtype = jnp.result_type(q, k, v)

    def host(x):
        return np.asarray(jnp.asarray(x).astype(dtype).astype(jnp.float32), np.float64)[0]

    return {"q": host(q), "k": host(k), "v": host(v), "g": np.asarray(g, np.float64)[0], "beta": host(beta)}


def enclosed(run: Run, lane: Lane) -> dict[str, Iv]:
    """Each record's enclosure `[N, H, ...]` (`enclosures`), each chunk's step from the fp32 memory it enters:
    `a`, `w`, `c`, `o`; `h`, the memory each chunk enters as stored; and `leaving`, the fp32 memory it
    leaves."""
    scale = run.k.shape[-1] ** -0.5
    found: dict[str, list] = {}
    for n in range(run.chunks):
        for head in range(run.heads):
            step = enclosures(run.step(n, head), scale, lane)
            step["leaving"], step["h"] = step["h"], stored_memory(point(run.carries[n, head]), lane)
            for name, iv in step.items():
                found.setdefault(name, []).append(iv)
    return {name: Iv(*(np.stack([getattr(iv, end) for iv in ivs]).reshape(run.chunks, run.heads,
                                                                           *ivs[0].lo.shape)
                       for end in ("lo", "hi"))) for name, ivs in found.items()}


def records(run: Run) -> dict[str, np.ndarray]:
    """Each record `[N, H, ...]`, as `enclosed` names them."""
    return {"a": run.a, "w": run.w, "c": run.c, "o": run.o, "h": run.h, "leaving": run.carries[1:]}


def witness(run: Run, lane: Lane) -> dict:
    """Whether each record of `run` lies inside its step's enclosure (`enclosed`), by record `[N, H, ...]`,
    and each enclosure's width in units of the last place of its record's grid, at the record (`ulps`) and
    at its row's largest (`spans`)."""
    found, recorded_ = enclosed(run, lane), records(run)
    held = {name: within(recorded_[name], found[name]) for name in found}

    def grid(name):
        return FLOAT32 if name == "leaving" else lane.held or FLOAT32

    largest = {name: np.max(np.abs(record), -1, keepdims=True) for name, record in recorded_.items()}
    return {**held, "ulps": {name: ulps(found[name], recorded_[name], grid(name)) for name in found},
            "spans": {name: ulps(found[name], largest[name], grid(name)) for name in found}}


def refusals(faulty: Run, claimed: Run, lane: Lane) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """A fault's refusals, by record: `faulty`'s records held to the enclosures `claimed` gives them (the
    call's own operands and memories, about `faulty`'s records), and where those are separated from the
    enclosures on what the faulty kernel was given (`faulty`'s own, `native_certificates.separated`)."""
    from native_certificates import separated

    truth, fault, found = enclosed(claimed, lane), enclosed(faulty, lane), records(faulty)
    return {name: (within(found[name], truth[name]), separated(truth[name], fault[name])) for name in truth}


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


def draw(seed: int, regime: str, *, chunks: int = 3, heads: int = 1, width: int = 64):
    """A bf16 call's operands `(q, k, v, g, beta, h0)` of one row of `chunks` chunks: unit queries and keys,
    normal values, the regime's log decays (`REGIMES`), a write gate in (0, 1), an initial memory."""
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
    beta = 0.0 if regime == "decay" else 1 / (1 + np.exp(-rng.normal(size=shape)))
    h0 = 0.3 * rng.normal(size=(1, heads, width, width))
    return (jnp.asarray(q, jnp.bfloat16), jnp.asarray(k, jnp.bfloat16), jnp.asarray(v, jnp.bfloat16),
            jnp.asarray(g, jnp.float32), jnp.asarray(np.broadcast_to(beta, shape), jnp.bfloat16),
            jnp.asarray(h0, jnp.float32))


def shifted_refusals(operands: tuple, lane: Lane) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """An off-by-one chunk boundary (`refusals`): the call run on its tokens one later (`shifted`), each
    record held to the step on the call's own chunks' operands."""
    faulty = recorded(*shifted(operands))
    return refusals(faulty, dataclasses.replace(faulty, **hosted(operands)), lane)


def cumsum_refusals(operands: tuple, lane: Lane) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The decay from a difference of rounded cumulative sums (`refusals`, `rounded_cumsums`): the call run on
    those log decays, each record held to the step on its own."""
    import jax.numpy as jnp

    q, k, v, g, beta, h0 = operands
    faulty = recorded(q, k, v, jnp.asarray(rounded_cumsums(g)), beta, h0)
    return refusals(faulty, dataclasses.replace(faulty, g=hosted(operands)["g"]), lane)


def handed_refusals(operands: tuple, truth: Run, at: int, lane: Lane) -> tuple[np.ndarray, np.ndarray]:
    """A state rounded to bf16 where fp32 is held: the call over the chunks from `at` on, from the fp32 memory
    `truth`'s chunks before it leave rounded to bf16, its first chunk's leaving memory held to the step from
    the memory unrounded (`refusals`)."""
    import jax.numpy as jnp
    import ml_dtypes as dtypes

    carry = truth.carries[at]
    handed = np.asarray(np.asarray(carry, np.float32).astype(dtypes.bfloat16), np.float32)
    tail = recorded(*(x[:, at * CHUNK:] for x in operands[:5]), jnp.asarray(handed)[None])
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
