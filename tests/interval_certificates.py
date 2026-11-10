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
from typing import Literal, NamedTuple

import numpy as np
from certified import allowance
from native_certificates import TRUNCATING, tf32_operand, truncating_underflow
from op_certificates import (
    BFLOAT16,
    FLOAT32,
    TFLOAT32,
    Format,
    above,
    below,
    exp2_rounded,
    grid_index,
    nearest,
    stepped,
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
    """The parts a factored pair's fp32 operand takes (`mainloop.pairing_parts`, `operand`)."""
    tf32: Literal["plain", "rna", "read"] = "read"
    """How a one-part fp32 operand reaches its product (`operand`), the issuer's: 'plain', as itself (an
    interpreted product, a plain fp32 dot: mosaic_sm80.py's and mosaic_sm90.py's interpreted branches);
    'rna', rounded to TF32 to nearest, a tie away from zero, first (`cvt.rna.tf32.f32`,
    `Family.rounds_tf32`: mma.sync's); 'read', read as TF32 from its fp32 register by the instruction, no
    conversion stated (warpgroup MMA's), which encloses either of the others."""


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
        return outward(iv.lo - reach, iv.hi + reach)
    return iv


def held_cell(x, lane: Lane) -> Iv:
    """The fp32 values a stored record `x` holds as the call holds it (`staged`): those rounding to it in
    bf16, those its two parts sum within 2^-16 of, or itself."""
    if lane.held == 1 and not lane.wide:
        return cell(x, BFLOAT16)
    if lane.held == 2:
        x = np.asarray(x, np.float64)
        reach = 2.0 ** -16 * np.abs(x) * (1 + 2.0 ** -7)
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
    lo = exp2_rounded(above(x.lo, FLOAT32), FLOAT32)
    hi = exp2_rounded(below(x.hi, FLOAT32), FLOAT32)
    for _ in range(lane.reach):
        lo, hi = stepped(lo, FLOAT32, -np.inf), stepped(hi, FLOAT32, np.inf)
    return Iv(np.where(lo < 2.0 ** FLOAT32.emin, 0.0, lo), hi)


def operand(iv: Iv, lane: Lane, parts: int = 1) -> Iv:
    """An fp32 value within `iv` as a product takes it in `parts`: one part as the lane's issuer reaches its
    TF32 product (`Lane.tf32`: itself; `native_certificates.tf32_operand`'s rounding, monotone, at both
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
