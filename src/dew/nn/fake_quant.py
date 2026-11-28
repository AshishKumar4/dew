"""Fake quantizers of DeepSeek-V4.1's cache (arXiv 2609.19969, section 2.4.4).

Each rounds fp32 values the way the release's tilelang kernels store them
(inference/kernel.py at the revision tools/deepseek_v41_reference.py pins)
and passes the gradient straight through. The rounding is arithmetic on the
fp32 values and bits, never a bare `astype` round trip through a narrow
dtype: XLA GPU's default xla_allow_excess_precision deletes an
f32 -> f8 -> f32 convert pair under jit, and the quantization silently does
not happen. One cast rounds instead, `fp8_codes`' float32 E4M3 on CUDA, held
by an optimization barrier between its two converts, and witnessed bit for
bit against the arithmetic (row 14171).
"""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from flax import linen as nn

from dew.nn.derivatives import Protected, contracted, declared_tangent, surrogate
from dew.nn.float_bits import exact_power2, largest_magnitude, raw_fields
from dew.nn.precision import at_least_fp32, correctly_rounded_divide, divided

E2M1 = np.arange(16, dtype=np.uint8).view(ml_dtypes.float4_e2m1fn).astype(np.float32)
"""Each E2M1 code's value in code order (ml_dtypes' float4_e2m1fn): 0 to 6
by codes 0-7, then their negatives with the sign in bit 3, -0 included. The
FP4 codecs, these quantizers and the fixture generators all read it."""

E2M1_MIDPOINTS = (E2M1[:7] + E2M1[1:8]) / 2
"""The magnitudes halfway between neighbouring E2M1 values, 0.25 up to 5."""

E2M1_TIES_UP = np.arange(7) % 2 == 1
"""Whether a magnitude exactly on each midpoint rounds up, to even: where the
lower neighbour's mantissa bit, its code's bit 0, is 1 (0.5, 1.5, 3)."""

E4M3_MAX = float(jnp.finfo(jnp.float8_e4m3fn).max)
"""E4M3FN's largest finite value, 448, which a block's amax is scaled to."""

AMAX_FLOOR = 1e-4
"""`per_block_cast_to_fp8`'s `clamp(1e-4)` on a block's amax (DeepGEMM, deep_gemm/utils/math.py at
057ca596)."""

type Block = int | tuple[int, int]
"""The rows and columns one FP8 scale covers: DeepSeek's square `block`, or `(rows, cols)`, such as
`(1, cols)` for one scale a row of a `cols`-wide matrix (a frozen FP8 matrix's rows, `dew.quantization`)."""

E2M1_MAX = float(jnp.finfo(jnp.float4_e2m1fn).max)
"""E2M1's largest finite value, 6, which an FP4 block's amax is scaled to."""


def _power_of_two_ceil(value):
    """2 ** ceil(log2(value)) off the fp32 bits, as the kernels' fast_round_scale
    computes it (kernel.py:22-37), the power set by `float_bits.exact_power2`. Unclamped, unlike the
    host codecs' `ceil_to_ue8m0` rule (exponents 1 to 254): only past the normal range do they part, a
    zero giving 0.0 here (exact_power2's sentinel at 2 ** -127) and 2 ** -126 there, a value above 2 **
    127 inf here (its biased exponent 255) and 2 ** 127 there; an infinity inf and a NaN the bits
    0x80000000, -0.0 (`exact_power2` past the normal range)."""
    _, _, biased, fraction = raw_fields(value.astype(jnp.float32))
    exponent = biased.astype(jnp.int32) - 127 + (fraction != 0).astype(jnp.int32)
    return exact_power2(exponent, jnp.float32)


def _round(values, mantissa: int, lowest: int):
    """Round fp32 values to a float format of `mantissa` bits whose smallest
    normal exponent is `lowest`, ties to even, by arithmetic: the quantum is
    the power of two of the value's binade less the mantissa bits, floored
    at the subnormal spacing; a product by its reciprocal (a normal power of
    two too) is the exact quotient, with no division (row 14171), and
    `round` ties to the even multiple, the one whose last mantissa bit is 0."""
    biased = raw_fields(values.astype(jnp.float32))[2].astype(jnp.int32)
    exponent = jnp.maximum(biased - 127, lowest) - mantissa
    return jnp.round(values * exact_power2(-exponent, jnp.float32)) * exact_power2(exponent, jnp.float32)


def _round_e4m3fn(values):
    """Round fp32 values within +-448 to E4M3FN, ties to even (`_round`).
    Not `jax.lax.reduce_precision(x, 4, 3)`, which models IEEE-style e4m3
    with infinities and a largest finite 240, not the FN format whose
    largest finite is 448."""
    return _round(values, 3, -6)


def _e2m1(values):
    """Round fp32 values within +-6 to E2M1, ties to even (`_round`); IEEE-style
    e2m1 would stop at 3 where E2M1 reaches 6, so reduce_precision cannot."""
    return _round(values, 1, 0)


def _e2m1_quotient(values, scale):
    """E2M1 of `values / scale` under an E4M3 scale, ties to even, clamped to
    +-6, without dividing: |values| is compared with each midpoint times the
    scale, a product of at most 7 significant bits and so exact, which
    rounds the exact quotient. For bf16 values that is the rounding of the
    correctly rounded quotient the kernel divides to, since a bf16 value off
    a midpoint times the scale lies further from it than half an fp32 ulp of
    the quotient. A division would not do: XLA divides by a broadcast
    through its reciprocal, and XLA GPU's division is not correctly rounded,
    either of which carries a quotient across a tie (-0.146484375 /
    0.1171875 comes out -1.2500001, rounding to -1.5 where the kernel's
    -1.25 rounds to -1.0)."""
    magnitude = jnp.abs(values)
    index = sum(((magnitude > point * scale) | ((magnitude == point * scale) if up else False))
                .astype(jnp.int32) for point, up in zip(E2M1_MIDPOINTS.tolist(), E2M1_TIES_UP.tolist(),
                                                        strict=True))
    return jnp.copysign(jnp.asarray(E2M1[:8])[index], values)


def _rounded_value(x, rounded):
    """`rounded` in `x`'s dtype, bit for bit, a negative zero included, which the kernels keep."""
    return rounded.astype(x.dtype)


def rounded_tangent(primals, tangents, value):
    """The straight-through quantizer propagates only the written input's tangent."""
    return tangents[0]


straight_through = contracted("test_a_straight_through_rounding_holds_every_order")(
    surrogate(_rounded_value, rounded_tangent))
"""`rounded` forward, bit for bit in `x`'s dtype (`_rounded_value`), and the identity's derivative, `x`'s
tangent, at every order, declared (`dew.nn.derivatives.surrogate`): `x + (rounded - x)` would re-round in
bf16 wherever the quantizer clamped far from `x`, and turn a -0 into +0. The estimator is Dew's choice; the
release is inference code and the paper names none."""


def _peak(axis, x):
    """`x`'s largest magnitude along `axis` by its words (`largest_magnitude`)."""
    return largest_magnitude(x, axis)


def _peak_tangent(axis, primals, tangents):
    (x,), (moved,) = primals, tangents
    # The float maximum's derivative, its tie pullback jnp.max's own (the cotangent shared among tied maxima).
    _, tangent = jax.jvp(lambda x: jnp.max(jnp.abs(x), axis=axis), (x,), (moved,))
    return largest_peak(axis, x), tangent


largest_peak = contracted("test_a_largest_peak_holds_the_float_maximums_derivative_at_every_order")(
    declared_tangent(Protected(), _peak, _peak_tangent, nondiff_argnums=(0,)))
"""`x`'s largest magnitude along `axis` (an int, a tuple of them, or None for every axis): its words'
(`float_bits.largest_magnitude`), exact at every width on every platform, a NaN above every magnitude
where XLA:CPU's YNNPACK float maximum drops one (row 14083), with the float maximum's derivative, declared
(row 14202): its tangent the moved maximum's, shared among tied maxima as `jnp.max`'s. A block scale read
from it (`fp8_block_cast`'s, `algebra.bits.nvfp4_coded`'s tensor scale) keeps that scale's derivative,
where a word maximum's alone is zero."""


def fp8_block_scale(amax):
    """A block's power-of-two FP8 scale from its largest magnitude `amax` (act_quant with scale_fmt ue8m0,
    kernel.py:40-124): amax floored at `AMAX_FLOOR`, the scale the ceiling power of two of amax / 448; the
    one statement a separate pass (`fp8_scaled_blocks`) and a GEMM epilogue's (`quantized`, row 8547) read."""
    return _power_of_two_ceil(jnp.maximum(amax, AMAX_FLOOR) * jnp.float32(1 / E4M3_MAX))


def fp8_scaled_blocks(x, block: int):
    """`x` in fp32 blocks of `block` channels and each block's power-of-two scale (`fp8_block_scale`)."""
    blocks = x.astype(at_least_fp32(x.dtype)).reshape(*x.shape[:-1], -1, block)
    return blocks, fp8_block_scale(largest_magnitude(blocks, -1, keepdims=True))


def fp8_codes(values):
    """`values` (fp32, a block over its scale) rounded to E4M3FN, ties to even, clamped to +-448: an FP8
    block's codes (`fp8_rounded`'s, before its scale). On CUDA, float32 values, clipped first, take XLA's own
    conversion to E4M3FN and back, an optimization barrier between (`lax.platform_dependent`; without it,
    XLA GPU's excess precision deletes the pair), one hardware rounding to nearest even where the
    arithmetic took a dozen operations an element (row 14171: a bank's codes pass 6.6 ms on a GH200), the
    same codes bit for bit (tests/test_gemm_blockwise.py's `fp8_codes_convert_as_the_rounding_does`, a
    native cell of tests/test_gpu_gemm.py: every code, the midpoints between them and their neighbours, the
    subnormal codes' range, past 448); a NaN its canonical NaN there, its own payload elsewhere, which no
    stored code holds. Elsewhere, and at any other width (a float64 truth's), the rounding by arithmetic
    (`_round_e4m3fn`)."""
    clipped = jnp.clip(values, -E4M3_MAX, E4M3_MAX)
    if clipped.dtype != jnp.float32:
        return _round_e4m3fn(clipped)
    return jax.lax.platform_dependent(clipped, cuda=_converted, default=_round_e4m3fn)


def _converted(clipped):
    """`fp8_codes`' CUDA branch: float32 `clipped` to E4M3FN and back, the narrow value held past XLA GPU's
    excess precision, which deletes a bare f32 -> f8 -> f32 pair (row 14171's review: 1.0625 kept, where E4M3
    ties it to 1.0); its derivative zero, the rounding's (`_round_e4m3fn`'s `lax.round`), where the casts'
    own would pass the tangent through (train 1495's review K-F1: `jvp` of `MxFp8.quantized(x).decoded()`
    on CUDA gave ones for zeros). A rounding's straight-through estimate is its caller's (`fp8_rounded`)."""
    return jax.lax.stop_gradient(_held_codes(clipped.astype(jnp.float8_e4m3fn)).astype(jnp.float32))


def _held_codes(codes):
    """E4M3 `codes` through an optimization barrier (`fp8_codes`' CUDA branch), held four to a uint32 word,
    a code's byte at its place's (little-endian, as a bitcast reads the word back), where the last axis holds
    whole words, then read back by bitcasts: the fusion forming them writes words, which XLA's loop emitter
    takes four a thread as it takes codes, so each thread forms sixteen codes and stores sixteen bytes. Codes
    written as codes ran four a thread in 128-thread blocks: 7,340,032 blocks for a 256 x 7168 x 2048 bank's
    natural codes, where its K-major codes' transposing fusion, the same arithmetic at 32 a thread, ran
    917,504 (sm90 compile, probe e2365886982), 5.1 against 3.05 ms on a GH200 (M-NG-75; row 14171). A
    `bitcast_convert_type` to words XLA folds into a bitcast of the codes, the emitter's unchanged (probe
    7ef36fb389); the shifts and ors (`algebra.bits.packed`'s, the one packing of codes into words) keep the
    words its output."""
    from dew.algebra.bits import E4M3, packed  # it imports this module's constants
    from dew.nn.sharding import explicit_spec

    shape = codes.shape
    placed = explicit_spec(codes)
    if not shape or shape[-1] % E4M3.codes_per_word or (placed is not None and placed.spec[-1] is not None):
        # A last axis of no whole words, or one an Explicit axis splits (its shards' words a reshape of the
        # split axis JAX may refuse to place, the head review of 38e98e841), as codes.
        return jax.lax.optimization_barrier(codes)
    held = jax.lax.optimization_barrier(packed(jax.lax.bitcast_convert_type(codes, jnp.uint8), E4M3))
    return jax.lax.bitcast_convert_type(jax.lax.bitcast_convert_type(held, jnp.uint8).reshape(shape),
                                        jnp.float8_e4m3fn)


def fp4_codes(values):
    """`values` (fp32, a block over its scale) rounded to E2M1, ties to even, clamped to +-6: an FP4
    block's codes (`fp4_rounded`'s under a power-of-two scale, before its scale)."""
    return _e2m1(jnp.clip(values, -E2M1_MAX, E2M1_MAX))


def fp8_rounded(x, blocks, scale):
    """`x` as E4M3 of its `blocks` over their `scale` (`fp8_scaled_blocks`), clamped to +-448, times the
    scale."""
    return straight_through(x, (fp8_codes(blocks / scale) * scale).reshape(x.shape))


def fake_quant_fp8(x, block: int):
    """E4M3 per `block` channels under a power-of-two scale (act_quant with
    scale_fmt ue8m0, kernel.py:40-124): amax floored at 1e-4, scale the
    ceiling power of two of amax / 448, value clamped to +-448 and rounded."""
    return fp8_rounded(x, *fp8_scaled_blocks(x, block))


def fp4_scaled_blocks(x, block: int, e4m3_scale: bool):
    """`x` in blocks of `block` channels at its width, fp32 at least (a float64 truth's float64), and each
    block's scale (fp4_act_quant, kernel.py:127-204): with `e4m3_scale` the E4M3 of amax / 6 with amax
    floored at 6 * 2**-9 (the compressed KV), else the ceiling power of two of amax / 6 with amax floored
    at 6 * 2**-126 (the indexer)."""
    wide = at_least_fp32(x.dtype)
    blocks = x.astype(wide).reshape(*x.shape[:-1], -1, block)
    amax = largest_magnitude(blocks, -1, keepdims=True)
    if e4m3_scale:
        # the kernel's cast saturates at E4M3's 448 (cvt.rn.satfinite)
        floored = jnp.maximum(amax, E2M1_MAX * 2 ** -9)
        # amax / 6 as fp32's correctly rounded quotient, as the kernel's division is: a plain `/` by the
        # constant compiles on the CPU to a multiply by RN(1/6), off by an ulp in a quarter of blocks
        # (ledger 1402). At a float64 truth's width, the float64 quotient rounded once to fp32, IEEE's
        # (53 >= 2 x 24 + 2), then rounded to E4M3 at that width, exactly.
        quotient = divided(floored, jnp.asarray(E2M1_MAX, wide)).astype(jnp.float32).astype(wide)
        return blocks, _round_e4m3fn(jnp.minimum(quotient, E4M3_MAX))
    return blocks, _power_of_two_ceil(jnp.maximum(amax, E2M1_MAX * 2 ** -126) * jnp.float32(1 / E2M1_MAX))


def fp4_rounded(x, blocks, scale, e4m3_scale: bool):
    """`x` as E2M1 of its `blocks` over their `scale` (`fp4_scaled_blocks`), clamped to +-6, times the
    scale: an E4M3 scale compared against (`_e2m1_quotient`), a power of two divided by exactly."""
    rounded = _e2m1_quotient(blocks, scale) if e4m3_scale else fp4_codes(blocks / scale)
    return straight_through(x, (rounded * scale).reshape(x.shape))


def fake_quant_fp4(x, block: int, e4m3_scale: bool):
    """E2M1 per `block` channels (fp4_act_quant, kernel.py:127-204): under an
    E4M3 scale amax / 6 with amax floored at 6 * 2**-9 (the compressed KV),
    or under the ceiling power of two of amax / 6 with amax floored at
    6 * 2**-126 (the indexer); the value clamped to +-6 and rounded."""
    return fp4_rounded(x, *fp4_scaled_blocks(x, block, e4m3_scale), e4m3_scale)


class FakeQuantizing(nn.Module):
    """A layer whose activations round through the fake quantizers as its own
    methods, so a caller sees and can stand in for each rounding through
    `nn.intercept_methods`, as `tools/deepseek_v41_numerics.py` does to replay
    a float64 twin's roundings in the given run. Each rounding reads its
    blocks and their scales from a method of their own (`fp8_scaled_blocks`,
    `fp4_scaled_blocks`), so a caller reads what a forward rounded, and how
    near a rounding's boundary it was, through `capture_intermediates`."""

    def fp8_scaled_blocks(self, x, block: int):
        return fp8_scaled_blocks(x, block)

    def fp8(self, x, block: int):
        return fp8_rounded(x, *self.fp8_scaled_blocks(x, block))

    def fp4_scaled_blocks(self, x, block: int, e4m3_scale: bool):
        return fp4_scaled_blocks(x, block, e4m3_scale)

    def fp4(self, x, block: int, e4m3_scale: bool):
        return fp4_rounded(x, *self.fp4_scaled_blocks(x, block, e4m3_scale), e4m3_scale)


def block_sides(block: Block) -> tuple[int, int]:
    """`block` as its rows and columns, refused unless both are positive ints."""
    rows, cols = (block, block) if isinstance(block, int) else block
    if type(rows) is not int or type(cols) is not int or rows < 1 or cols < 1:
        raise ValueError(f"a block covers a positive number of rows and columns, got {block!r}")
    return rows, cols


def fp8_blocks_on_host(weight, block: Block, *, rounded=np.asarray) -> tuple[np.ndarray, np.ndarray]:
    """DeepGEMM's `per_block_cast_to_fp8` on the host: `weight`, a finite [rows, cols] matrix in any float
    dtype on any device (a host float32 copy of it, exact), cast to float8_e4m3fn blocks of `block` and the
    float32 scales that invert them, each block's scale rounded by `rounded` (a ue8m0 checkpoint's rounds
    it up to a power of two, `dew.interop.codecs.quantize_fp8_blocks`). A matrix that is not finite is
    refused by name: E4M3FN encodes no infinity, and one carries its whole block's scale away with it."""
    values = np.asarray(weight, np.float32)
    if values.ndim != 2:
        raise ValueError(f"block-scaled quantization takes a [rows, cols] weight, got shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(
            f"a {values.shape} weight holds {int(np.count_nonzero(~np.isfinite(values)))} value(s) that are "
            f"not finite; E4M3FN encodes no infinity, and one of them carries its whole block's scale away "
            f"with it")
    rows, cols = block_sides(block)
    codes, scale_inv = _cast_blocks(values, rows, cols, xp=np, divide=np.divide, rounded=rounded,
                                    coded=np.asarray, peak=lambda axis, x: np.max(np.abs(x), axis=axis))
    # A contiguous copy: sliced off the padding, the codes are a view whose buffer a writer taking it would
    # read as the padded rows.
    return np.ascontiguousarray(codes), scale_inv


def fp8_block_cast(x, block: Block) -> tuple[jax.Array, jax.Array]:
    """`fp8_blocks_on_host`' cast in a traced program, on a matrix where it computes, to the same bytes: a
    frozen FP8 product's activation rows (`dew.quantization`). No value is refused and no scale rounded to
    ue8m0 here."""
    rows, cols = block_sides(block)
    # XLA divides through a reciprocal; the cast's two divisions round as IEEE's do. The codes round to
    # E4M3 by `fp8_codes` (arithmetic, or on CUDA its barred convert, bit for bit the same), which the cast
    # then holds exactly: XLA GPU's default excess precision deletes a bare f32 -> E4M3 -> f32 convert pair,
    # and XLA:CPU converts a hoisted small loop's narrow floats through float16, rounding twice.
    return _cast_blocks(jnp.asarray(x, jnp.float32), rows, cols, xp=jnp, divide=correctly_rounded_divide,
                        rounded=jnp.asarray, coded=fp8_codes, peak=largest_peak)


def _cast_blocks(values, rows: int, cols: int, *, xp, divide, rounded, coded, peak):
    """The FP8 block cast of a float32 `values` matrix under `xp`, NumPy or jax.numpy, dividing by `divide`,
    rounding each block's scale by `rounded` and each value over it to E4M3 by `coded` before the cast:
    `per_block_cast_to_fp8`'s rule, its 1e-4 amax floor `AMAX_FLOOR`. Each block's maximum is `peak(axis,
    tiles)`'s: on the host NumPy's float one, which keeps a NaN; traced, `largest_peak`, which keeps it on
    every backend and declares the float maximum's derivative, which the scale keeps (row 14202)."""
    height, width = values.shape
    padded = xp.pad(values, ((0, -height % rows), (0, -width % cols)))
    tiles = padded.reshape(padded.shape[0] // rows, rows, padded.shape[1] // cols, cols)
    amax = xp.maximum(peak((1, 3), tiles), xp.float32(AMAX_FLOOR))
    scale_inv = rounded(divide(amax, xp.float32(E4M3_MAX)))
    # One reciprocal per block, as the reference takes it.
    codes = coded(tiles * divide(xp.float32(1.0), scale_inv)[:, None, :, None]).astype(
        ml_dtypes.float8_e4m3fn)
    return codes.reshape(padded.shape)[:height, :width], scale_inv
