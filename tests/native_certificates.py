"""Check 2's native certificates for the fused template kernels (ledger row 2084, kernel-templates.md
§9): a native GEMM's output held element by element to its exact products, accumulated in fp32 by the
tensor cores' truncating adds in any order and rounded once into its output (`native_dot_holds`), its
products' magnitudes as the tensor cores align them (`Terms`, `aligned`); TF32 operands as the hardware
reads them (`tf32_operand`); exact sums a depth step at a time (`promoted_step_holds`); and a fault's
refusals derived from its interval's separation from the call's (`separated`, `assert_refused`). Check
2's per-primitive contracts are tests/certified.py's."""

import dataclasses

import numpy as np
from certified import allowance, rounded_on_lane
from op_certificates import (
    FORMATS,
    TFLOAT32,
    Format,
    exact_sum,
    magnitude_sum,
    rounded,
    rounded_away,
    truncated,
    two_sum,
)

TRUNCATING = {name: dataclasses.replace(fmt, name=f"{name}, truncating adds", precision=fmt.precision - 1)
              for name, fmt in FORMATS.items()}
"""Each format's sums as a GPU's tensor cores carry an MMA step's: each add truncated, not rounded, an error
up to a whole unit in the last place, twice round to nearest's (src/dew/nn/kernels/templates/product.py),
the unit roundoff of a format one bit narrower."""


SUBNORMAL_REACH = 4.0
"""Below it a product may come from a subnormal operand of a 16-bit format or of TF32: a subnormal times the
format's largest finite value is under 2^-126 2^128 = 4 (bf16, TF32), and under 2^-14 65504 < 4 (fp16)."""


def aligned(values, fmt: Format) -> np.ndarray:
    """Each operand's magnitude as a tensor core aligns its products by: its own, but a subnormal of `fmt` at
    `fmt`'s least normal, the exponent its field holds (`SUBNORMAL_CASE`, where the products' own magnitudes'
    bound admits 12 fp32 units and the tensor cores' sum lies 752 from the exact)."""
    magnitude = np.abs(np.asarray(values, np.float64))
    least = 2.0 ** fmt.emin
    return np.where((magnitude > 0) & (magnitude < least), least, magnitude)


SUBNORMAL_CASE = ((-(2.0 ** -24), 23 * 2.0 ** -24, 97 * 2.0 ** -24, 8.821487426757812e-05,
                   -0.00101470947265625),
                  (1.0, 0.0023059844970703125, -201 * 2.0 ** -24, -401 * 2.0 ** -24, -55 * 2.0 ** -24),
                  -5.52972779e-08)
"""Five fp16 products, the first of the subnormal 2^-24 and 1, and the sum the tensor cores give them in fp32
on an A100 (the GEMM template's Triton launch) and an H100 (its Mosaic GPU and Triton launches), a recurrence
inverse's merge (row 2084, link 3): 752 fp32 units from their exact sum, -5.52946098e-08, as if each term
were kept to 2^-37, 2^-23 under the subnormal's exponent field's 2^-14 (`aligned`)."""


@dataclasses.dataclass(frozen=True)
class Terms:
    """A native GEMM's exact products (float64, summed along the last axis) and each one's magnitude as the
    tensor cores align it (`aligned`), sliced alike: what its contract reads (`native_dot_ends`)."""
    exact: np.ndarray
    magnitudes: np.ndarray

    @property
    def shape(self) -> tuple:
        return self.exact.shape

    def __getitem__(self, index) -> "Terms":
        return Terms(self.exact[index], self.magnitudes[index])


def terms(build, *operands, fmt: Format) -> Terms:
    """The products `build(*operands)` multiplies out of 16-bit or TF32 operands of `fmt` (a product of them,
    each term one product), and their magnitudes as the tensor cores align them: `build` of each operand's
    aligned magnitude (`aligned`)."""
    return Terms(np.asarray(build(*operands), np.float64),
                 np.asarray(build(*(aligned(operand, fmt) for operand in operands)), np.float64))


def _terms_of(products) -> Terms:
    """`products` as `Terms`: given so, or exact products none of which a subnormal operand may have made
    (none nonzero below `SUBNORMAL_REACH`), whose magnitudes are their own; a subnormal operand must never
    take the tight bound unseen, so any other raises."""
    if isinstance(products, Terms):
        return products
    products = np.asarray(products, np.float64)
    if np.any((products != 0) & (np.abs(products) < SUBNORMAL_REACH)):
        raise ValueError("a product under 4 may come from a subnormal 16-bit operand, which the tensor cores "
                         "align at its format's least normal: pass its operands' `Terms` (`terms`)")
    return Terms(products, np.abs(products))


def native_dot_ends(products, out: Format, flushes: bool) -> tuple[np.ndarray, np.ndarray]:
    """The least and the greatest value of `out` a native GEMM's output takes on its exact `products`
    (`native_dot_holds`; `Terms`, or exact products `_terms_of` admits bare): the nearest `out` values of the
    allowance's two ends; +0 for an empty sum."""
    products = _terms_of(products)
    count = products.shape[-1]
    if count == 0:
        return np.zeros(products.shape[:-1]), np.zeros(products.shape[:-1])
    exact = exact_sum(products.exact, -1)
    accumulator = FORMATS["float32"]
    # A truncating add below the normal range loses up to a whole subnormal unit, not half of one.
    underflow = 2.0 ** accumulator.emin if flushes else 2.0 ** (accumulator.emin - accumulator.precision + 1)
    slack = allowance(magnitude_sum(products.magnitudes, -1), count, TRUNCATING["float32"], underflow)
    zero = np.zeros_like(exact)
    return (rounded(np.nextafter(exact - slack, -np.inf), zero, out),
            rounded(np.nextafter(exact + slack, np.inf), zero, out))


def native_dot_holds(native, products, out: Format, flushes: bool) -> np.ndarray:
    """Whether each element of a native GEMM's output is a value its exact `products` (`Terms`: float64,
    summed along the last axis, with their magnitudes as the tensor cores align them) admit, as check 2
    holds a fused template GEMM natively (row 2084, link 1): accumulated in fp32 in any order by adds that
    truncate (`TRUNCATING`), within γ_n of their aligned magnitudes of the exact sum plus what underflow loses
    past that (as `_terms_hold`), then rounded once to nearest in `out`, the output's format. Rounding is
    monotone, so the output lies between the nearest `out` values of the allowance's two ends."""
    low, high = native_dot_ends(products, out, flushes)
    native = np.asarray(native, np.float64)
    return np.isfinite(native) & (native >= low) & (native <= high)


def separated(ends, others) -> np.ndarray:
    """Whether each element's two intervals (each its least and greatest values, as `native_dot_ends` gives
    them) share no value. A kernel run around a defect computes, at each element, a value within the
    interval of the products it was given (the contract's null); wherever that interval is separated from
    the interval of the call's own products, the contract refuses it, whatever order or tile the kernel
    sums in. So a fault's refusals are derived from its products, not counted from one run."""
    (low, high), (other_low, other_high) = ends, others
    return (high < other_low) | (other_high < low)


def assert_refused(held, apart, share: float) -> None:
    """A fault refused where it lands, as a native contract's validation holds it (row 2084): every element
    whose interval on the call's products is separated from that on the products a kernel run around the
    defect was given (`apart`, `separated`) is refused (`held` false), and those are more than `share` of
    the elements, a share of the draw's products, which no order or tile the kernel legally takes moves."""
    held, apart = np.asarray(held), np.asarray(apart)
    assert not np.any(held & apart) and np.mean(apart) > share, np.mean(apart)


def tf32_operand(values) -> np.ndarray:
    """fp32 operands' values (as float64) as a tensor core multiplies them in a TF32 product (an fp32 product
    at the default precision, src/dew/nn/kernels/templates/product.py): rounded to TF32 to nearest, a tie away
    from zero (PTX's cvt.rna.tf32.f32), as the GEMM template's Triton launches read both operands of the
    forward's, the input gradient's and the kernel gradient's products on an A100 and an H100 (row 2084, link
    2: each of 38,400 values of each operand, 30% of them ties, read so, where 85% were their round to
    nearest even and 35% their truncation). A product of two is exact in float64 (22 bits), as
    `native_dot_holds` takes."""
    return rounded_away(values, TFLOAT32)


def promoted_step_ends(low, high, products, flushes: bool) -> tuple[np.ndarray, np.ndarray]:
    """The least and the greatest fp32 value an IEEE add, rounded to nearest, takes of an accumulator
    between `low` and `high` (fp32 values as float64, the same where it is known) and a depth step's partial,
    an fp32 value its exact `products` admit as a tensor core sums them from zero (`native_dot_ends`):
    rounding to nearest is monotone in both, so the sums' roundings at their ends."""
    accumulator = FORMATS["float32"]
    least, most = native_dot_ends(products, accumulator, flushes)
    return (rounded_on_lane(*two_sum(np.asarray(low, np.float64), least), accumulator, flushes),
            rounded_on_lane(*two_sum(np.asarray(high, np.float64), most), accumulator, flushes))


def truncating_step_ends(before, products, flushes: bool) -> tuple[np.ndarray, np.ndarray]:
    """The least and the greatest fp32 value a tensor core's accumulator `before` (fp32 values as float64)
    takes once a depth step's exact `products` are added into it by the MMA's own adds, which truncate
    (`TRUNCATING`), as a launch without exact sums adds them: the native-dot contract's ends with the
    accumulator one more term (`native_dot_ends`), and, where the accumulator and every product share a sign,
    no further from zero than the exact sum, since each value an add of such terms makes is rounded toward
    zero (Fasi, Higham, Mikaitis and Pranesh, Numerical behavior of NVIDIA tensor cores, PeerJ Computer
    Science 7:e330, 2021: the products aligned and the sum normalized by truncation)."""
    accumulator = FORMATS["float32"]
    before, products = np.asarray(before, np.float64), _terms_of(products)
    summed = Terms(np.concatenate([before[..., None], products.exact], -1),
                   np.concatenate([np.abs(before)[..., None], products.magnitudes], -1))
    low, high = native_dot_ends(summed, accumulator, flushes)
    # The float64 sum is within half its unit of the exact one, so a unit further from zero bounds it.
    total = exact_sum(summed.exact, -1)
    bound = truncated(np.nextafter(total, np.copysign(np.inf, total)), accumulator)
    positive = (before >= 0) & np.all(products.exact >= 0, -1)
    negative = (before <= 0) & np.all(products.exact <= 0, -1)
    return np.where(negative, np.maximum(low, bound), low), np.where(positive, np.minimum(high, bound), high)


def promoted_step_holds(before, after, products, flushes: bool) -> np.ndarray:
    """Whether each native fp32 accumulator after one depth step of exact sums (`after`) is the IEEE sum,
    rounded to nearest, of the accumulator before the step (`before`) and the step's own partial: an fp32
    value the step's exact `products` admit as a tensor core sums them from zero (`native_dot_holds`, into
    fp32). So a product declared above the default precision into an fp32 output adds each depth step's
    partial apart (src/dew/nn/kernels/templates/product.py's `promoted`; row 2084, link 2). Rounding to
    nearest is monotone in the partial, so `after` lies between the sums' roundings at its two ends."""
    least, most = promoted_step_ends(before, before, products, flushes)
    after = np.asarray(after, np.float64)
    return np.isfinite(after) & (after >= least) & (after <= most)


def narrow_partial_ends(first, rest, flushes: bool) -> tuple[np.ndarray, np.ndarray]:
    """The fault interval of a partial rounded to bf16 mid-depth (a defect a native GEMM's certificate
    refuses): the first half's products (`first`, `Terms`) summed and rounded to bf16, then the rest's
    (`rest`) added in fp32. The first half's bf16 value lies between its ends' bf16 values, and the IEEE add
    is monotone in both (`promoted_step_ends`)."""
    halved = native_dot_ends(first, FORMATS["float32"], flushes)
    return promoted_step_ends(*(rounded(end, np.zeros_like(end), FORMATS["bfloat16"]) for end in halved),
                              rest, flushes)


# The least share of each fault's elements whose interval on the call's products is separated from that on
# the products the kernel was given (`assert_refused`), measured natively and in the CPU emulations alike
# (tests/test_native_certificates.py, tests/test_check2.py), each claim below its measurement.
LOST_SHARE = 0.95
"""The least share of the elements a lost block of sixteen rows or depth columns reaches whose interval on
the call's products is separated from that on the products the kernel was given (`assert_refused`): 99.4%
of a forward's bf16 outputs, 99.9% of a kernel gradient's fp32 ones, and 99.98% of the exact-sums step that
loses them, on an A100 and an H100 alike."""
MOVED_SHARE = 0.95
"""The same of the elements a row read under its neighbour group reaches: all of the forward's row, 99.8% of
both groups' kernel gradients, and 99.85% of both groups' first exact-sums steps."""
NARROW_SHARE = 0.65
"""The same of an fp32 forward's elements where a partial sum over half the depth is rounded to bf16 before
the rest is added, on Gaussian draws, whose partials sometimes cancel to a bf16 value's neighbourhood:
70.5%."""
TRUNCATED_SHARE = 0.4
"""The same of a kernel gradient's elements past each group's first depth step, on a falling draw of one sign
(`FALL`), where its default launch adds each step's products by truncating adds and exact sums are
declared: 48.8% on Mosaic GPU's launch and 49.2% on Triton's, about those whose exact sum lies past the
midpoint between two fp32 values (each was refused, and 87% to 99% of all of them were)."""
TF32_SHARES = {True: (0.85, 0.85, 0.75), False: (0.85, 0.85, 0.95)}
"""The same, on Gaussian draws and on draws of one sign, of an fp32 forward's elements where its operands'
bf16 values are multiplied and TF32 is declared, its TF32 values where bf16 is, and its truncated TF32 values
where they are rounded: 92.5%, 92.5% and 79.8%, and 89.3%, 89.3% and all."""
