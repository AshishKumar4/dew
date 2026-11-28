"""fp32 division correctly rounded on every backend (`correctly_rounded_divide`, a primitive of its own,
`correctly_rounded_divide_p`), however far below the normal range (`lowered_divide`), its derivatives
dividing as it does (the denominator's by `binades.over_denominator`), and PyTorch's division by a rate
(`divided`): `dew.nn.precision`'s division, which it re-exports."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend import core
from jax.interpreters import ad, batching, mlir

from dew.nn.binades import over_denominator, stood_in
from dew.nn.derivatives import Protected, bound_span, contracted
from dew.nn.float_bits import exact_power2, float_bits, lowered_rounded, quiet_nan_bits, raw_fields
from dew.nn.programs import UNROUNDED, register_entrywise


def software_divide(numerator: jax.typing.ArrayLike,
                              denominator: jax.typing.ArrayLike) -> jax.Array:
    """The correctly rounded fp32 quotient, IEEE division's, on XLA's approximate
    dividers too: XLA:CPU rewrites a division by a broadcast into a multiplication
    by its reciprocal (openxla/xla#51148), and XLA:GPU's fp32 a / b is off IEEE on
    30 to 50% of quotients (DIVIDE-a100, DIVIDE-h100 receipts), where a quantizer's
    scales and a diffusion schedule's states must round as the reference's host or
    CUDA division does (compressed-tensors' NVFP4, the FP8 block cast, PyTorch's
    schedulers).

    Each operand's significand, its fraction bits under the exponent of one, in
    [1, 2), divides apart from its exponent field (`_rounded_significand`), so
    every intermediate stays normal whatever the operands' range; the fields'
    difference then scales the rounded significand by its exponent bits, exact
    for a normal result. A quotient below the normal range rounds once to the
    subnormal grid: the significand scaled onto that grid rounds to its nearest
    integer, ties to even, the exact residual settling a scaled significand that
    lands on a half, and the integer is the result's bits, which no flush to zero
    reaches. Past the largest float the quotient is infinite. A zero, infinite or
    NaN operand is the division's; a subnormal one, which XLA:CPU reads as zero, is
    normalized from its bits first (`float_bits`), so its quotient is its value's.
    Every step is a bit operation or a float32 operation on normal values, about 140
    operations a quotient where frexp's and nextafter's own special cases took 210
    (row 6312). It is the lowering of a primitive of its own
    (`correctly_rounded_divide`), so no pass sees its error-free steps to reassociate them.
    """
    return _divided(numerator, denominator)


def lowered_divide(numerator: jax.Array, denominator: jax.Array, lowered: jax.Array) -> jax.Array:
    """`numerator / denominator * 2^-lowered` (`lowered` int32, at least 0, broadcast), rounded once to fp32
    however far below its least normal (`_divided`, `correctly_rounded_divide`'s): a quotient a CPU flushes
    (row 4023), its numerator raised by `lowered` from its bits by its caller (`raised_exactly`); a subnormal
    operand normalized from its bits as `correctly_rounded_divide`'s is."""
    return _divided(numerator, denominator, lowered)


def _divided(numerator: jax.typing.ArrayLike, denominator: jax.typing.ArrayLike,
             lowered: jax.typing.ArrayLike = 0) -> jax.Array:
    """`correctly_rounded_divide`'s quotient times `2^-lowered`, rounded once
    (`software_divide`); a special operand's division lowered in integers (`lowered_rounded`)."""
    numerator, denominator = jnp.broadcast_arrays(jnp.asarray(numerator, jnp.float32),
                                                  jnp.asarray(denominator, jnp.float32))
    # Raw fields class a special operand; normalized ones divide a subnormal as its value, which XLA's
    # division flushed (2^-149 / 2^-126 is 2^-23, not 0; the 2026-11-26 audit).
    (top_sign, top_biased, top_raw, top_fraction, top_exponent), (
        bottom_sign, bottom_biased, bottom_raw, bottom_fraction, bottom_exponent) = (
        float_bits(value) for value in (numerator, denominator))
    special = (((top_biased == 0) & (top_raw == 0)) | (top_biased == 255)
               | ((bottom_biased == 0) & (bottom_raw == 0)) | (bottom_biased == 255))
    # A special lane's subnormal a one of its sign (-0 / 2^-149 is -0, not a flushed 0 / 0: `stood_in`).
    plain = stood_in(numerator, top_sign, top_biased, top_raw) / stood_in(
        denominator, bottom_sign, bottom_biased, bottom_raw)
    if not (isinstance(lowered, int) and lowered == 0):
        plain = lowered_rounded(plain, jnp.broadcast_to(jnp.asarray(lowered, jnp.int32), plain.shape))
    top_field, bottom_field = top_exponent + 127, bottom_exponent + 127
    top_fraction, bottom_fraction = top_fraction.astype(jnp.uint32), bottom_fraction.astype(jnp.uint32)
    negative = top_sign != bottom_sign
    # A normal operand's significand, exact in [1, 2); any bits give one in that range, so a special
    # lane's steps stay normal too, and its result is the division's.
    top, bottom = (jax.lax.bitcast_convert_type(fraction | np.uint32(0x3F800000), jnp.float32)
                   for fraction in (top_fraction, bottom_fraction))
    significand, residual = _rounded_significand(top, bottom)
    exponent = top_field - bottom_field - lowered
    # The rounded significand's fields (`raw_fields`), a positive normal's: its biased exponent once scaled,
    # 1 to 254 a normal float, set back in its bits.
    bits, _, significand_field, _ = raw_fields(significand)
    field = significand_field.astype(jnp.int32) + exponent
    normal = jax.lax.bitcast_convert_type(bits + jax.lax.bitcast_convert_type(exponent << 23, jnp.uint32),
                                          jnp.float32)
    # Below the normal range: the significand on the grid of 2^-149, a value under 2^23 whose
    # integer part and half are exact in fp32; within the float32 range scaled by 2^shift.
    shift = jnp.clip(exponent + 149, -126, 127)
    scaled = significand * exact_power2(shift, jnp.float32)
    whole = jnp.floor(scaled)
    fraction = scaled - whole
    odd = (whole.astype(jnp.int32) & 1) == 1
    half_up = (residual > 0) | ((residual == 0) & odd)
    up = (fraction > 0.5) | ((fraction == 0.5) & half_up)
    subnormal = jax.lax.bitcast_convert_type((whole.astype(jnp.int32) + up.astype(jnp.int32)),
                                             jnp.float32)
    magnitude = jnp.where(field >= 255, jnp.inf, jnp.where(field >= 1, normal, subnormal))
    signed = jnp.where(negative, -magnitude, magnitude)
    return jnp.where(special, plain, signed)


def short_divide(numerator: jax.typing.ArrayLike, denominator: jax.typing.ArrayLike) -> jax.Array:
    """`correctly_rounded_divide`'s quotient, bit for bit (a NaN float32's quiet one), where `denominator` is
    a positive normal fp32 of at most four significant bits (an e4m3 value, e2m1's largest): in integer
    arithmetic, about 50 operations where the division takes 140 (row 14083: an NVFP4 coding's block ratios
    and codes, a step's coding as many times as its call sites). The numerator's significand (`raw_fields`, a
    subnormal's its fraction), its leading bit moved to bit 30, over the denominator's four bits as integers:
    a 27- or 28-bit quotient and its remainder, rounded once at the result's place, ties to even, on the
    subnormal grid below the normals, past the largest an infinity. A zero or infinite numerator is its own
    quotient, a NaN float32's quiet one, read off its bits (no float division of the tile). Its integer steps
    carry no derivative: a coding's quotients, whose codes have none."""
    word = jnp.uint32
    numerator, denominator = jnp.broadcast_arrays(jnp.asarray(numerator, jnp.float32),
                                                  jnp.asarray(denominator, jnp.float32))
    bits, sign, biased, fraction = raw_fields(numerator)
    biased = biased.astype(jnp.int32)
    significand = fraction | jnp.where(biased > 0, word(0x800000), word(0))
    lead = jax.lax.clz(significand)
    top = significand << (lead - word(1))
    _, _, divisor_biased, divisor_fraction = raw_fields(denominator)
    divisor = (divisor_fraction >> word(20)) | word(8)
    whole, rest = jax.lax.div(top, divisor), jax.lax.rem(top, divisor)
    # `whole` holds the quotient's 27 or 28 leading bits (`wide`); `field` its biased exponent: the
    # numerator's (a subnormal's as one), less the divisor's and the leading bit's move.
    wide = (whole >= word(1 << 27)).astype(jnp.int32)
    field = jnp.maximum(biased, 1) - divisor_biased.astype(jnp.int32) - lead.astype(jnp.int32) + wide + 134
    # 24 bits kept for a normal result, fewer on the subnormal grid; a shift past the quotient's bits keeps
    # none, its rounding half past them too.
    shift = jnp.minimum(3 + wide + jnp.maximum(1 - field, 0), 31).astype(word)
    kept = whole >> shift
    dropped = whole - (kept << shift)
    half = word(1) << (shift - word(1))
    up = (dropped > half) | ((dropped == half) & ((rest != 0) | ((kept & word(1)) == word(1))))
    # The rounded significand added to the exponent field below it, its carry the rounding's own (into the
    # next binade, from the subnormals to the least normal, from the largest to the infinity).
    rounded = ((jnp.clip(field, 1, 254) - 1).astype(word) << word(23)) + kept + up.astype(word)
    # A zero or infinite numerator is its own quotient by a positive finite divisor, a NaN float32's quiet
    # one, whatever its sign or payload, as `float_bits.rounded_binary` gives it (so a coding's NaN lanes
    # encode alike at either width and in a GEMM epilogue's software division: row 14083's review).
    nan = (biased == 255) & (fraction != 0)
    special = (biased == 255) | ((biased == 0) & (fraction == 0))
    own = jnp.where(nan, word(quiet_nan_bits(jnp.float32)), bits)
    value = jnp.where(special, own, jnp.where(field >= 255, word(0x7F800000), rounded) | sign)
    return jax.lax.bitcast_convert_type(value, jnp.float32)


def _significand_quotient(numerator: jax.Array, denominator: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The correctly rounded fp32 quotient of two significands in [1, 2), and the sign of the exact
    quotient's distance from it (+1 above, -1 below, 0 exact).

    A Dekker TwoProduct gives the exact residual of the approximate
    quotient, split into two fp32 values. One residual correction puts the
    quotient within one ulp; comparing its exact residual with each
    neighbour's half-spacing times the divisor chooses nearest-even. Its
    neighbours are its bits' (a positive normal float's next and previous
    floats), and that threshold product is exact because the half-spacing
    is a power of two, every intermediate normal at these magnitudes. The
    rounded quotient's residual is the corrected one's less the move, one
    spacing times the divisor, exact, and within a factor of two of the
    corrected residual where the quotient moves (Sterbenz), so its sign needs
    no further product. Each residual is an error-free step, so the whole is
    one protected span."""
    from dew.nn.precision import two_product, two_sum

    quotient = numerator / denominator
    product, error = two_product(quotient, denominator, bit_split=True, subtract=True)
    residual = (numerator - product) - error
    quotient = quotient + residual / denominator
    product, error = two_product(quotient, denominator, bit_split=True, subtract=True)
    high, low = two_sum(numerator - product, -error)
    bits = jax.lax.bitcast_convert_type(quotient, jnp.uint32)
    above, below = (jax.lax.bitcast_convert_type(bits + step, jnp.float32) for step in (1, -1))
    upper = (above - quotient) * 0.5 * denominator
    lower = (quotient - below) * 0.5 * denominator
    odd = (bits & 1) != 0
    up = (high > upper) | ((high == upper) & ((low > 0) | ((low == 0) & odd)))
    down = (high < -lower) | ((high == -lower) & ((low < 0) | ((low == 0) & odd)))
    rounded = jnp.where(up, above, jnp.where(down, below, quotient))
    # The exact quotient's side of the rounded one: the rounded quotient's exact residual, the numerator
    # less its product with the divisor, which is zero only where the two are equal.
    side = jnp.where(up, (high - 2 * upper) + low, jnp.where(down, (high + 2 * lower) + low, high))
    return rounded, jnp.sign(side)


_rounded_significand = bound_span(Protected(), _significand_quotient)
"""`_significand_quotient` as one span the algebra pass keeps as it is (`dew.nn.derivatives.Protected`), as
reassociation would cancel the residuals its error-free steps keep."""


correctly_rounded_divide_p = core.Primitive("correctly_rounded_divide")
"""The correctly rounded fp32 division of two arrays (`correctly_rounded_divide`), an operation of its own so
that its derivatives divide as it does. Its operands broadcast as a binary operation's do: the function
gives it two of one shape, and a batching of one of them leaves the other a scalar or of unit axes
(`batching.defbroadcasting`). Its rounding is a selection among neighbours, so its
tangent is the division's, each quotient in it correctly rounded: `g / d` in the numerator and
`-g * (n / d) / d` in the denominator, the order PyTorch's backward divides in, and its transpose in the
numerator `ct / d`. As `jnp.divide`'s, they divided as XLA does: by the divisor's rounded reciprocal where it
is a constant or a broadcast on a CPU (openxla/xla#51148) and off IEEE on 30 to 50% of quotients on a GPU, so
a quantizer's scale or a rate's gradient left the reference's by an ulp on a third of its entries (row
5277). It is entrywise (`dew.nn.programs.register_entrywise`), which the analyses of a program's entries
read."""
correctly_rounded_divide_p.def_impl(software_divide)
# Its result is typed as JAX types its own division's: the operands' broadcast shape, their dtype, and
# their sharding and varying axes, which a mesh's Explicit axis carries in a value's type.
correctly_rounded_divide_p.def_effectful_abstract_eval(jax.lax.div_p.abstract_eval)
mlir.register_lowering(correctly_rounded_divide_p,
                       mlir.lower_fun(software_divide, multiple_results=False))


@functools.cache
def triton_lowerings() -> None:
    """Register `correctly_rounded_divide_p`'s Pallas Triton lowering lazily
    (`dew.nn.kernels.launches.pallas_call`, row 9794) so importing `precision` loads no Pallas Triton
    machinery."""
    from jax._src.pallas.triton import lowering as triton_lowering
    from jax.experimental.pallas import triton as pltriton

    def divided(numerator, denominator):
        numerator, denominator = jnp.broadcast_arrays(numerator, denominator)
        [quotient] = pltriton.elementwise_inline_asm(
            "div.rn.f32 $0, $1, $2;", args=[numerator, denominator], constraints="=r,r,r", pack=1,
            result_shape_dtypes=[jax.ShapeDtypeStruct(numerator.shape, numerator.dtype)])
        return quotient

    triton_lowering.register_lowering(correctly_rounded_divide_p)(
        triton_lowering.lower_fun(divided, multiple_results=False))


batching.defbroadcasting(correctly_rounded_divide_p)
register_entrywise(correctly_rounded_divide_p)
# It rounds the quotient to float32 whatever its operands' dtype: at a wider width it is the quotient.
UNROUNDED[correctly_rounded_divide_p] = jnp.divide


@contracted("test_the_precision_seams_hold_every_order")
def correctly_rounded_divide(numerator: jax.typing.ArrayLike, denominator: jax.typing.ArrayLike) -> jax.Array:
    """The correctly rounded fp32 quotient `numerator / denominator` (`software_divide`), IEEE
    division's on every backend, differentiable at every order, each of its derivatives' divisions
    correctly rounded too (`correctly_rounded_divide_p`)."""
    numerator, denominator = jnp.broadcast_arrays(jnp.asarray(numerator, jnp.float32),
                                                  jnp.asarray(denominator, jnp.float32))
    return correctly_rounded_divide_p.bind(numerator, denominator)


def _divide_transpose(cotangent, numerator, denominator):
    """The division's transpose in its numerator, the one operand it is linear in: the cotangent over the
    denominator."""
    if not ad.is_undefined_primal(numerator) or ad.is_undefined_primal(denominator):
        raise TypeError("a correctly rounded division is linear in its numerator alone")
    if type(cotangent) is ad.Zero:
        return [ad.Zero(numerator.aval), None]
    quotient = correctly_rounded_divide_p.bind(cotangent, denominator)
    # A numerator the division broadcast (a scalar's, or along a unit axis) takes the sum of the quotients
    # it was broadcast to, as a broadcast's transpose sums them (`sharding.unbroadcast`), in their dtype.
    from dew.nn.sharding import unbroadcast

    return [unbroadcast(quotient, numerator.aval.shape), None]


ad.defjvp2(correctly_rounded_divide_p,
           lambda tangent, quotient, numerator, denominator: correctly_rounded_divide(tangent, denominator),
           lambda tangent, quotient, numerator, denominator: -over_denominator(
               tangent, quotient, numerator, denominator))
ad.primitive_transposes[correctly_rounded_divide_p] = _divide_transpose


def divided(numerator, rate):
    """`numerator / rate` as a source written in PyTorch divides a state by a rate, or an array by a
    constant: IEEE division, whatever the backend's own division does (XLA takes a division by a
    constant as a product with its rounded reciprocal, on a CPU and an A100 alike, row 1402). A
    float32, bfloat16 or float16 quotient is `correctly_rounded_divide`'s of the operands widened to
    float32, rounded to its dtype, as PyTorch's float32 opmath gives it, under any compiler flags. A
    float64 one is the plain division."""
    numerator, rate = jnp.asarray(numerator), jnp.asarray(rate)
    dtype = jnp.result_type(numerator, rate)
    if not jnp.issubdtype(dtype, jnp.floating):
        return numerator / rate
    if dtype == jnp.float64:
        return numerator / rate
    return correctly_rounded_divide(numerator.astype(jnp.float32), rate.astype(jnp.float32)).astype(dtype)
