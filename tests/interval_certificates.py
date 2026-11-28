"""Check 2's outward interval arithmetic for a native launch's records: what the templates' step witnesses
(`recurrence_witness`, `attention_witness`) enclose each record in.

Each enclosure is outward interval arithmetic in float64, each end snapped to its op's own grid: products of
16-bit values (or of an fp32 operand's three bf16 parts, six of them a pair, `product.formed`) exact, summed
as the native-dot contract holds a tensor core's sum (gamma_n at a truncating add's unit, of the terms that
may be nonzero, `dot`); fp32 adds and multiplies rounded to nearest (`fp32`), a product before an add rounded
or exact (an FMA's contraction, `fused`); casts to nearest (`cast`); and `exp2` within the lane's reach of the
fp32 grid of its correctly rounded value (`exp2`, `exp2_reach`: PTX ISA 9.4 §9.7.3.21's ex2.approx.f32 at 2
ulp, a mainloop's `approx_math`). A lane (`Lane`) states the arithmetic a launch runs in and how its call
holds the values its launches pass between them."""

import dataclasses
from collections.abc import Callable
from typing import Literal, NamedTuple

import numpy as np
import tier_wire
from certified import allowance, underflow
from native_certificates import TRUNCATING, truncating_underflow
from op_certificates import (
    BFLOAT16,
    FLOAT32,
    TFLOAT32,
    Format,
    above,
    below,
    exp2_rounded,
    exp_rounded,
    grid_index,
    nearest,
    stepped,
    tf32_operand,
)


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


def outward(lo, hi) -> Iv:
    return Iv(np.nextafter(lo, -np.inf), np.nextafter(hi, np.inf))


def onto(iv: Iv, fmt: Format) -> Iv:
    """The values of `fmt` within `iv`: what a value known to be on `fmt`'s grid takes."""
    return Iv(above(iv.lo, fmt), below(iv.hi, fmt))


def cast(iv: Iv, fmt: Format) -> Iv:
    """A cast to nearest of a value within `iv` (monotone)."""
    return Iv(nearest(iv.lo, fmt), nearest(iv.hi, fmt))


def saturated(iv: Iv, fmt: Format) -> Iv:
    """A saturating cast to nearest of a value within `iv`: clipped to `fmt`'s largest finite magnitude, then
    rounded (both monotone), as a bf16 part a store forms (`product.bf16_parts`' saturating parts): a finite
    fp32 value past `fmt`'s overflow midpoint its largest, where `cast` gives infinity."""
    largest = fmt.largest
    return cast(Iv(np.clip(iv.lo, -largest, largest), np.clip(iv.hi, -largest, largest)), fmt)


def cell(x, fmt: Format) -> Iv:
    """The values that round to nearest to each `x` of `fmt`: between its midpoints with its neighbours."""
    x = np.asarray(x, np.float64)
    return Iv((x + stepped(x, fmt, -np.inf)) / 2, (x + stepped(x, fmt, np.inf)) / 2)


def meet(a: Iv, b: Iv) -> Iv:
    return Iv(np.maximum(a.lo, b.lo), np.minimum(a.hi, b.hi))


def hull(a: Iv, b: Iv) -> Iv:
    return Iv(np.minimum(a.lo, b.lo), np.maximum(a.hi, b.hi))


def add(a: Iv, b: Iv) -> Iv:
    return outward(a.lo + b.lo, a.hi + b.hi)


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
    pairing: int = 1
    """The parts a factored pair's fp32 operand takes (`intra.pairing_parts`, `operand`)."""
    unit_pairs: bool = False
    """Whether a sub-chunk's own pairs fold each decay into the key before the row (`algebra.decayed_key`,
    `mainloop.Call.unit_pairs`, a layer's unit queries and keys and write gates at most one:
    `own_pairs.own_bands`, `algebra.key_diagonal_bands`), not the row times the key first (`key_diagonal`)."""
    anchored: bool = False
    """Whether a sub-chunk's own pairs are one anchored product (`anchored.anchoring`, a layer's decay
    bound: `recurrence_witness.key_anchored`), not each pair its own decay (`key_diagonal`)."""
    merged: int = 0
    """The parts the doubling's merges take in fp32 (`mainloop._merged`): zero for fp16, else `merged`."""
    rounded_prefix: bool = False
    """Whether a decay a key dimension's prefix is its rounded sum alone (`mainloop.Call.prefix_limbs`), its
    exponents half an fp32 unit further (`recurrence_witness.Exponents.of`)."""
    clamped: bool = False
    """Whether its log decays are clamped at its floor before their prefix is summed (`mainloop.Call.clamped`,
    `recurrence_witness.Exponents.of`)."""
    exp_reach: int = 0
    """`exp`'s reach: the most steps of the fp32 grid its result lies from its correctly rounded value
    (`exp_reach`), an accurate backward's exponential's (`jnp.exp`, no approximation)."""
    exp_flushes: bool = True
    """Whether that `exp` gives zero where its correctly rounded value is a positive subnormal
    (`exp_flushes`); a lane not measured may."""
    tanh_reach: int = 0
    """`tanh`'s reach: the most steps of the fp32 grid its result lies from its correctly rounded value
    (`tanh_reach`), a capped score's (Gemma 2's cap, `template.Capped`)."""
    tf32: Literal["plain", "rna", "read"] = "read"
    """How a one-part fp32 operand reaches its product (`operand`), the issuer's: 'plain', as itself (an
    interpreted product, a plain fp32 dot: mosaic_sm80.py's and mosaic_sm90.py's interpreted branches);
    'rna', rounded to TF32 to nearest, a tie away from zero, first (`cvt.rna.tf32.f32`,
    `Family.rounds_tf32`: mma.sync's); 'read', read as TF32 from its fp32 register by the instruction, no
    conversion stated (warpgroup MMA's), which encloses either of the others."""
    solve: tuple[int, Literal["plain", "rna", "read"]] | None = None
    """The merges' parts and TF32 conversion (`merged`, `tf32`) a decay a key dimension's solve takes on the
    intra walk's bundle (`intra.sub_chunked`: mma.sync's on every target, an H100's one TF32 part rounded a
    tie away where its other launches take warpgroup MMA's two bf16 parts); the call's where None."""

    def solving(self) -> "Lane":
        """This lane as a decay a key dimension's solve runs in (`solve`)."""
        if self.solve is None:
            return self
        return dataclasses.replace(self, merged=self.solve[0], tf32=self.solve[1])


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


def two_parts_reach(magnitude, lane: Lane):
    """How far two bf16 parts' sum lies from the fp32 value of `magnitude` they split: the second part the
    remainder rounded, within 2^-16 of the value, or within half bf16's least subnormal step where the
    remainder lies below bf16's normals, or, on a lane that flushes, the whole of a remainder or value below
    its normals, fp32's (`certified.underflow`): a value near 1e-40 split into parts dropped 2^-134, whose
    relative reach was none, and Kimi's steep inverse lay one bf16 step outside its enclosure natively on a
    GH200 (row 10007, M-JS-156: 68 of 24576 entries, each 2^-133 out)."""
    return 2.0 ** -16 * magnitude + underflow(BFLOAT16, lane.flushes)


def staged(iv: Iv, lane: Lane) -> Iv:
    """A register value as the call's launches hold it (`Lane.held`): cast to bf16 (a 16-bit call's), its fp32
    value (three parts summing to it, or one TF32 operand, which its product reads, `dot`), or two parts'
    sum (`two_parts_reach`)."""
    if lane.held == 1:
        return iv if lane.wide else cast(iv, BFLOAT16)
    if lane.held == 2:
        reach = two_parts_reach(np.maximum(np.abs(iv.lo), np.abs(iv.hi)), lane)
        return outward(iv.lo - reach, iv.hi + reach)
    return iv


def held_cell(x, lane: Lane) -> Iv:
    """The fp32 values a stored record `x` holds as the call holds it (`staged`): those rounding to it in
    bf16, those its two parts sum within 2^-16 of, or itself."""
    if lane.held == 1 and not lane.wide:
        return cell(x, BFLOAT16)
    if lane.held == 2:
        x = np.asarray(x, np.float64)
        reach = two_parts_reach(np.abs(x) * (1 + 2.0 ** -7), lane)
        return outward(x - reach, x + reach)
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
    return _contracted(Iv(a.lo[:, None, :], a.hi[:, None, :]), Iv(b.lo.T[None, :, :], b.hi.T[None, :, :]),
                       lane, fmt, into, parts)


def row_dot(a: Iv, b: Iv, lane: Lane, fmt: Format = BFLOAT16, parts: tuple[int, int] = (1, 1)) -> Iv:
    """Each row's `sum(a * b)` over its last axis (`[M, K]` each, `[M]`) as `dot` bounds one of its entries,
    the diagonal of `a @ b^T` alone: the terms a row's own."""
    return _contracted(a, b, lane, fmt, None, parts)


def _contracted(a: Iv, b: Iv, lane: Lane, fmt: Format, into: Iv | None, parts: tuple[int, int]) -> Iv:
    """`dot`'s sums over the last axis of `a` and `b` broadcast together (`dot`'s `[M, 1, K]` and `[1, N, K]`,
    `row_dot`'s rows), each `[..., K]` term exact and their sum within the native-dot contract's bound."""
    from dew.nn.kernels.templates import product

    pa, pb = parts
    if lane.wide and pa == 1:
        a, fmt = operand(a, lane), FLOAT32
    if lane.wide and pb == 1:
        b, fmt = operand(b, lane), FLOAT32
    issued = product.terms(pa, pb)
    dropped = [(i, j) for i in range(pa) for j in range(pb) if (i, j) not in issued]
    share = sum(PART_BOUNDS[i] * PART_BOUNDS[j] for i, j in issued) if pa * pb > 1 else 1.0
    lost = sum(PART_BOUNDS[i] * PART_BOUNDS[j] for i, j in dropped)
    if pa * pb > 1:
        fmt = BFLOAT16
    terms = mul(a, b)
    if into is not None:
        terms = Iv(np.concatenate([terms.lo, into.lo[..., None]], -1),
                   np.concatenate([terms.hi, into.hi[..., None]], -1))
    live = ((terms.lo != 0) | (terms.hi != 0)).sum(-1) * len(issued)
    pairs = aligned(a, fmt) * aligned(b, fmt)
    magnitude = pairs.sum(-1) * share * (1 + 2.0 ** -50)
    if into is not None:
        magnitude = magnitude + np.maximum(np.abs(into.lo), np.abs(into.hi)) * (1 + 2.0 ** -50)
    # The native-dot contract's allowance, and the float64 interval sums' own roundings beside it.
    slack = (allowance(magnitude, live, TRUNCATING["float32"], truncating_underflow(lane.flushes))
             + (live + 1) * 2.0 ** -52 * magnitude + lost * pairs.sum(-1) * (1 + 2.0 ** -50))
    return onto(outward(terms.lo.sum(-1) - slack, terms.hi.sum(-1) + slack), FLOAT32)


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
    return _exponential(x, exp2_rounded, lane.reach)


def exp(x: Iv, lane: Lane) -> Iv:
    """`exp` of an fp32 value within `x`: within `lane.exp_reach` steps of the fp32 grid of its correctly
    rounded value (`op_certificates.exp_rounded`, verified), as `exp2` is of its own; zero below the normal
    range only where the lane's `exp` flushes there (`Lane.exp_flushes`), a subnormal weight else kept."""
    return _exponential(x, exp_rounded, lane.exp_reach, flushes=lane.exp_flushes)


def natural_exp(x: Iv, lane: Lane) -> Iv:
    """`attention_algebra.natural_exp`'s `exp` of the exact value within `x`, no fp32 operand (its
    compensated conversion keeps the product's residual): within exp2's reach and one step of its correctly
    rounded value (its joined exponent within a quarter of the weight's ulp), as `exp2` is of its own."""
    return _exponential(x, exp_rounded, lane.reach + 1, operands=False)


def _exponential(x: Iv, rounded: Callable, reach: int, *, operands: bool = True, flushes: bool = True) -> Iv:
    """An exponential's enclosure (`exp2`'s, `exp`'s, `natural_exp`'s): `rounded`, its correct rounding, at
    the ends of `x` (its least and greatest fp32 operands where `operands`), each `reach` steps of the fp32
    grid outward, and zero where its lower end falls below the normal range and it `flushes` there."""
    lo, hi = (above(x.lo, FLOAT32), below(x.hi, FLOAT32)) if operands else (x.lo, x.hi)
    lo, hi = rounded(lo, FLOAT32), rounded(hi, FLOAT32)
    for _ in range(reach):
        lo, hi = stepped(lo, FLOAT32, -np.inf), stepped(hi, FLOAT32, np.inf)
    return Iv(np.where(flushes & (lo < 2.0 ** FLOAT32.emin), 0.0, lo), hi)


def tanh_rounding(x) -> tuple[np.ndarray, np.ndarray]:
    """The fp32 values tanh of each fp32 `x` rounds to nearest within: tanh in float64 (libm's, within a few
    of its last bits) widened by 2^-40 of itself, far past those bits and far inside fp32's, and taken to
    the fp32 grid below and above, so the correctly rounded fp32 tanh is one of the two."""
    exact = np.tanh(np.asarray(x, np.float64))
    slack = np.abs(exact) * 2.0 ** -40 + 2.0 ** -1074
    return below(exact - slack, FLOAT32), above(exact + slack, FLOAT32)


def tanh(x: Iv, lane: Lane) -> Iv:
    """`tanh` of an fp32 value within `x`: within `lane.tanh_reach` steps of the fp32 grid of its correctly
    rounded value (`tanh_rounding`), at the least and greatest fp32 operands within `x` (tanh and rounding to
    nearest are monotone), never past ±1."""
    lo = tanh_rounding(above(x.lo, FLOAT32))[0]
    hi = tanh_rounding(below(x.hi, FLOAT32))[1]
    for _ in range(lane.tanh_reach):
        lo, hi = stepped(lo, FLOAT32, -np.inf), stepped(hi, FLOAT32, np.inf)
    return Iv(np.maximum(lo, -1.0), np.minimum(hi, 1.0))


def _rounded_of(found: np.ndarray, x: np.ndarray, name: str, ranged: Callable,
                exact: np.ndarray | None = None) -> None:
    """Refuse a probe's results (`tanh_reach`'s, `exp2_reach`'s, `exp_reach`'s) that are no rounding of `name`
    at all, whose steps from the correct one are no reach (a NaN's word read as a grid index lies two billion
    steps off, a walk of them a timeout, not a verdict): one outside the function's range (`ranged`), and one
    non-finite where the correctly rounded value `exact` is finite or another non-finite (where no `exact`,
    every non-finite one: tanh's results are all finite)."""
    finite = np.isfinite(found)
    wrong = ~finite if exact is None else (finite != np.isfinite(exact)) | (~finite & (found != exact))
    if (bad := np.flatnonzero(wrong | ~ranged(np.where(finite, found, 0.0)))).size:
        raise ArithmeticError(f"the lane's {name} gives {found[bad[:4]]} at {x[bad[:4]]}, no rounding of "
                              f"{name}")


def tanh_reach(native, low: float = -12.0, high: float = 12.0) -> int:
    """The most steps of the fp32 grid `native(x)`, an fp32 `tanh` of fp32 operands, lies from tanh(x)
    correctly rounded (`tanh_rounding`'s nearer end), over (`low`, `high`), the operands a capped score
    takes (its products times scale over the cap) and past where tanh is ±1 in fp32: 2^17 of them, densely
    and drawn, zero among them. In a GPU tier's prepare, whose arms compute nothing (`exp2_reach`'s), none."""
    rng = np.random.default_rng(1402)
    x = np.concatenate([np.linspace(low, high, 1 << 16, dtype=np.float32), np.zeros(1, np.float32),
                        rng.uniform(low, high, (1 << 16) - 1).astype(np.float32)])
    found = np.asarray(native(x), np.float64)
    if tier_wire.preparing():
        return 0
    _rounded_of(found, x, "tanh", lambda values: np.abs(values) <= 1.0)
    lower, upper = tanh_rounding(x)
    below_by, above_by = grid_index(lower) - grid_index(found), grid_index(found) - grid_index(upper)
    return int(np.max(np.maximum(np.maximum(below_by, above_by), 0)))


def operand(iv: Iv, lane: Lane, parts: int = 1) -> Iv:
    """An fp32 value within `iv` as a product takes it in `parts`: one part as the lane's issuer reaches its
    TF32 product (`Lane.tf32`: itself; `op_certificates.tf32_operand`'s rounding, monotone, at both
    ends; or read as TF32 from the register, both ends rounded outward on its grid, which encloses nearest
    under either tie and truncation); two bf16 parts (`product.formed`) within the third part's bound of it
    (`PART_BOUNDS`); three, the value itself."""
    if parts == 1 and lane.tf32 == "plain":
        return iv
    if parts == 1 and lane.tf32 == "rna":
        return Iv(tf32_operand(iv.lo), tf32_operand(iv.hi))
    if parts == 1 and lane.tf32 == "read":
        return Iv(below(iv.lo, TFLOAT32), above(iv.hi, TFLOAT32))
    if parts == 2:
        widen = PART_BOUNDS[2] * np.maximum(np.abs(iv.lo), np.abs(iv.hi))
        return outward(iv.lo - widen, iv.hi + widen)
    if parts == 3:
        return iv
    raise ValueError(f"no enclosure of an fp32 operand in {parts} parts under conversion {lane.tf32!r}")


def exp2_reach(native, low: float = -2.0, high: float = 2.0) -> int:
    """The most steps of the fp32 grid `native(x)`, an fp32 `exp2` of fp32 operands, lies from `2^x` correctly
    rounded (`op_certificates.exp2_rounded`), over the operands a mainloop takes, (`low`, `high`): by
    default the fractions `algebra.raised` and `algebra.decayed` take, (-2, 2); 2^17 of them, densely and
    drawn, the integers there (exact cases) among them. A zero is the flush `exp2` admits (its enclosure is
    zero where its lower end, that many steps down, falls below the normal range), not a step, and an
    infinity where the correctly rounded value is one (an overflow, from 128 up) is exact: measured over the
    finite nonzero results, and a zero where that end is normal, or an infinity where the rounded value is
    finite, refuses the lane. In a GPU tier's prepare,
    whose arms give zeros and compute nothing (`reference_error.ahead_arm`), `native` is compiled at its
    operands and no reach is read: none, so a cell's later arms are prepared too (row 8497)."""
    return _exponential_reach(native, low, high, exp2_rounded, "exp2", 2084)


def exp_reach(native, low: float = -104.0, high: float = 0.0, *, flushes: bool = True) -> int:
    """`exp2_reach`'s measurement of `native(x)`, an fp32 `exp` of fp32 operands, from `e^x` correctly rounded
    (`op_certificates.exp_rounded`), over (`low`, `high`): by default the natural differences a row's
    weights take, -104 to 0, through fp32's least normal (e^-87.34) and its subnormals to where every result
    rounds to zero (e^-103.97, half the least subnormal). Where it may not flush (`flushes` False: a native
    unapproximated `exp`, which keeps a subnormal weight), a zero for a positive subnormal is refused."""
    return _exponential_reach(native, low, high, exp_rounded, "exp", 2085, flushes=flushes)


def exp_flushes(native, low: float = -104.0, high: float = -87.0) -> bool:
    """Whether `native(x)`, an fp32 `exp` of fp32 operands, gives zero where `e^x` correctly rounded is a
    positive subnormal, over (`low`, `high`), fp32's subnormal results (`Lane.exp_flushes`). In a GPU tier's
    prepare, whose arms compute nothing (`exp2_reach`'s), it may."""
    probed = _probed(native, low, high, exp_rounded, "exp", 2086)
    return probed is None or bool(np.any((probed[1] == 0) & (probed[2] > 0)))


def _probed(native, low: float, high: float, rounded: Callable, name: str, seed: int) -> tuple | None:
    """`native` at fp32 operands over (`low`, `high`), dense, whole and drawn from `seed`, beside `rounded`,
    `name`'s correct rounding, there, each a result no rounding gives refused (`_rounded_of`): the operands,
    the results and the rounded values; none in a GPU tier's prepare, whose arms compute nothing."""
    rng = np.random.default_rng(seed)
    whole = np.arange(np.ceil(low), np.floor(high) + 1, dtype=np.float32)
    x = np.concatenate([np.linspace(low, high, 1 << 16, dtype=np.float32), whole,
                        rng.uniform(low, high, (1 << 16) - len(whole)).astype(np.float32)])
    found, exact = np.asarray(native(x), np.float64), rounded(x, FLOAT32)
    if tier_wire.preparing():
        return None
    _rounded_of(found, x, name, lambda values: values >= 0.0, exact)
    return x, found, exact


def _exponential_reach(native, low: float, high: float, rounded: Callable, name: str, seed: int, *,
                       flushes: bool = True) -> int:
    """`exp2_reach`'s and `exp_reach`'s measurement: `native` against `rounded`, `name`'s correct rounding,
    over (`low`, `high`) drawn from `seed` (`_probed`), a zero for a positive result refused where it may not
    `flushes`."""
    if (probed := _probed(native, low, high, rounded, name, seed)) is None:
        return 0
    x, found, exact = probed
    flushed = found == 0
    if not flushes and (lost := flushed & (exact > 0)).any():
        raise ArithmeticError(f"the lane's {name} flushes a subnormal result at {x[lost][:4]}")
    measured = ~flushed & np.isfinite(found)
    reach = int(np.max(np.abs(grid_index(found) - grid_index(exact))[measured], initial=0))
    # Each flushed result's enclosure's lower end, `reach` places below its exact value on the grid, in places
    # past the least normal value: at or below none, it is normal.
    below_normal = grid_index(2.0 ** FLOAT32.emin) - grid_index(exact[flushed]) + reach
    normal = below_normal <= 0
    if np.any(normal):
        raise ArithmeticError(f"the lane's {name} flushes a normal result at {x[flushed][normal][:4]}")
    return reach
