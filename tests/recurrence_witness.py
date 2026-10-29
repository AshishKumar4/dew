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
    ends = np.stack(np.broadcast_arrays(a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi))
    return _outward(ends.min(0), ends.max(0))


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


def dot(a: Iv, b: Iv, lane: Lane) -> Iv:
    """`a @ b` (`[M, K] @ [K, N]`) as one product the tensor cores accumulate in fp32: each term's exact range
    (products of 16-bit values, or of three bf16 parts an fp32 operand's, the six whose orders sum under
    three, within 2^-24 of the fp32 values' product), summed within the native-dot contract's bound
    (`certified.native_dot_holds`: gamma_n at a truncating add's unit, n the terms that may be nonzero, an add
    of an exact zero exact), the float64 sums within n 2^-52 of their own."""
    terms = mul(Iv(a.lo[:, None, :], a.hi[:, None, :]), Iv(b.lo.T[None, :, :], b.hi.T[None, :, :]))
    parts = 6 if lane.wide else 1
    live = ((terms.lo != 0) | (terms.hi != 0)).sum(-1) * parts
    magnitude = np.maximum(np.abs(terms.lo), np.abs(terms.hi)).sum(-1)
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
        power = dot(merge(current), merge(lower_a), lane)
        joined = dot(merge(where(lower, power, point(np.zeros((CHUNK, CHUNK))))), merge(current), lane)
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
    (-2, 2), densely, and the integers there, exact cases."""
    from op_certificates import exp2_rounded

    rng = np.random.default_rng(2084)
    x = np.concatenate([np.linspace(-2, 2, 1 << 16, dtype=np.float32), np.arange(-2, 3, dtype=np.float32),
                        rng.uniform(-2, 2, 1 << 16).astype(np.float32)])
    return int(np.max(np.abs(_index(np.asarray(native(x), np.float64)) - _index(exp2_rounded(x, FLOAT32)))))


@dataclasses.dataclass(frozen=True)
class Run:
    """A call's operands as its launches read them and its records, one row: `q`, `k` `[T, Hk, Dk]`, `v` `[T,
    H, Dv]`, `g`, `beta` `[T, H]`, `h0` `[H, Dk, Dv]` (float64 of their dtypes' values), and each chunk and
    head's stored inverse less its diagonal `a` `[N, H, C, C]`, weighted keys `w` `[N, H, C, Dk]`, entering
    memory `h` `[N, H, Dk, Dv]`, corrected values `c` and output `o` `[N, H, C, Dv]`, and the final memory
    `final` `[H, Dk, Dv]`, an fp32 call's three parts summed."""
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    g: np.ndarray
    beta: np.ndarray
    h0: np.ndarray
    a: np.ndarray
    w: np.ndarray
    h: np.ndarray
    c: np.ndarray
    o: np.ndarray
    final: np.ndarray

    @property
    def chunks(self) -> int:
        return self.a.shape[0]

    @property
    def heads(self) -> int:
        return self.a.shape[1]

    def step(self, n: int, head: int, lane: Lane, *, carry: Iv | None = None) -> Step:
        """Chunk `n` and value head `head`'s step: its operands, its records and the carry entering it (the
        first chunk's initial memory exactly; an fp32 call's stored memory; a 16-bit call's the fp32 values
        rounding to its stored memory), unless given."""
        rows, key = slice(n * CHUNK, (n + 1) * CHUNK), head // (self.heads // self.k.shape[1])
        if carry is None:
            carry = (point(self.h0[head]) if n == 0 else point(self.h[n, head]) if lane.held is None
                     else cell(self.h[n, head], lane.held))
        return Step(self.q[rows, key], self.k[rows, key], self.v[rows, head], self.g[rows, head],
                    self.beta[rows, head], self.a[n, head], self.w[n, head], self.h[n, head], self.c[n, head],
                    carry)

    def leaving(self, n: int, head: int) -> np.ndarray:
        """The memory chunk `n` leaves in its record: the next chunk's stored memory, or the final memory."""
        return self.h[n + 1, head] if n + 1 < self.chunks else self.final[head]


def recorded(q, k, v, g, beta, h0, *, starts=None) -> Run:
    """The records of the gated delta rule's call on the template, forced (`generation.chosen`) at JAX's own
    default precision as a model runs it: `mainloop.kept_forward` at the tiles the route takes
    (`tuning.seen`), its output and final memory bit for bit the route's own (the certify build is the
    production program), on the device's bundle, interpreted off a GPU. Operands `[1, T, ...]`."""
    import jax
    import jax.numpy as jnp

    from dew.nn.kernels import generation, recurrence_chunk, tuning
    from dew.nn.kernels.templates.recurrence import mainloop, template

    operands = (q, k, v, g, beta, h0)
    with jax.default_matmul_precision(None), generation.chosen(gated_delta_rule="mosaic_gpu"):
        with tuning.seen() as launches:
            o, final = jax.jit(lambda *x: recurrence_chunk.chunk_gated_delta_rule(*x, starts=starts))(
                *operands)
        [tiles] = [config for (op, _), (config, _) in zip(launches, launches.taken, strict=True)
                   if isinstance(op, template.Mosaic)]
        dtype = jnp.result_type(q, k, v)
        kept = jax.jit(lambda *x: mainloop.kept_forward(
            *x, tiles=tiles, starts=starts, scale=k.shape[-1] ** -0.5,
            family=mainloop.family_of(dtype) or mainloop.INTERPRETED,
            interpret=generation.backend() != "gpu"))(*operands)
    kept_o, kept_final, (a, (w, h, c)) = kept
    for native, own in ((o, kept_o), (final, kept_final)):
        assert np.array_equal(np.asarray(native).view(np.uint8), np.asarray(own).view(np.uint8)), \
            "the certify build is the production program"
    parts = 3 if jnp.dtype(dtype) == jnp.float32 else 1
    _, length, heads, value_width = v.shape
    key_width, chunks = k.shape[-1], length // CHUNK

    def summed(x, rows):
        return np.asarray(x, np.float64).reshape(parts, rows, -1).sum(0)

    def chunked(x, width):
        return summed(x, chunks * CHUNK).reshape(chunks, CHUNK, heads, width).transpose(0, 2, 1, 3)

    def host(x):
        return np.asarray(np.asarray(x).astype(np.float32 if parts == 3 else x.dtype), np.float64)[0]

    return Run(host(q), host(k), host(v), np.asarray(g, np.float64)[0],
               np.asarray(jnp.asarray(beta).astype(dtype).astype(jnp.float32), np.float64)[0],
               np.asarray(h0, np.float64)[0], chunked(a, CHUNK), chunked(w, key_width),
               summed(h, chunks * heads * key_width).reshape(chunks, heads, key_width, value_width),
               chunked(c, value_width),
               np.asarray(kept_o, np.float64)[0].reshape(chunks, CHUNK, heads, value_width).transpose(
                   0, 2, 1, 3),
               np.asarray(kept_final, np.float64)[0])


def witness(run: Run, lane: Lane) -> dict[str, np.ndarray]:
    """Whether each record of `run` lies inside its step's enclosure (`enclosures`), by record `[N, H, ...]`:
    `a`, `w`, `c`, `o`, the entering memory `h` (the first chunk's its initial memory stored), and `leaving`,
    each chunk's leaving memory (the next chunk's stored, or the final memory, fp32), with each enclosure's
    width in its record's units of the last place (`ulps`) under `ulps`."""
    scale = run.k.shape[-1] ** -0.5
    held: dict[str, list] = {name: [] for name in ("a", "w", "c", "o", "leaving")}
    widths: dict[str, list] = {name: [] for name in held}
    for n in range(run.chunks):
        for head in range(run.heads):
            step = run.step(n, head, lane)
            found = enclosures(step, scale, lane)
            last = n + 1 == run.chunks
            found["leaving"] = found["h"] if last else stored_memory(found["h"], lane)
            records = {"a": step.a, "w": step.w, "c": step.c, "o": run.o[n, head],
                       "leaving": run.leaving(n, head)}
            for name, record in records.items():
                held[name].append(within(record, found[name]))
                grid = FLOAT32 if name == "leaving" and last else lane.held or FLOAT32
                widths[name].append(ulps(found[name], record, grid))
    first = within(run.h[0], stored_memory(point(run.h0), lane))
    shaped = {name: np.stack(values).reshape(run.chunks, run.heads, *values[0].shape)
              for name, values in held.items()}
    return {**shaped, "h": first, "ulps": {name: np.stack(values) for name, values in widths.items()}}
