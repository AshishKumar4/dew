"""The fused backward's lean score tail on wgmma (`attention_mosaic_scores._tile_scores`' `lean`), lowered in
its accumulators' registers (`plgpu.inline_mgpu` bodies): a tile's normalized weights, an entry's fused
centring, exponential and share (`weighed`), and its bf16 score cotangents narrowed saturating by one
conversion a pair (`saturated_bf16`). M-NA-28-gh200: the fused dK/dV's 2.02B instructions against
FlashAttention 3's 1.08B, its excess on the score tile (283M FMUL against 70M, 136M FMNMX against none,
row 5272)."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.lib.mlir.dialects import vector
from jax.experimental.mosaic.gpu import fragmented_array as mgpu_fragments, utils as mgpu_utils
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend.mlir import ir
from jax.extend.mlir.dialects import arith

from dew.nn.kernels.templates import attention_gradient as gradient
from dew.nn.kernels.templates.attention import LOG2E

WGMMA = plgpu.Layout.WGMMA
"""wgmma's accumulators' layout, a tile's registers' the bodies read and write."""

WEIGHED = ("{ .reg .f32 x; fma.rn.f32 x, $1, $2, $3; mul.rn.f32 x, x, $4; ex2.approx.ftz.f32 x, x; "
           "mul.rn.f32 $0, x, $5; }")
"""`Weighed`'s entry: its logit centred in natural units by one fused multiply-add, in base 2, its
exponential, its share (a multiply that keeps a subnormal weight)."""


@dataclasses.dataclass(frozen=True, eq=False)
class Weighed:
    """`weighed`'s body: each entry `exp(s scale - m) / T` as `exp2((s scale - m) log2(e)) (1/T)`, its
    centring one fused multiply-add, in the logits' natural units (a logit at its maximum 0 whatever their
    size, rounded once), where it took a multiply and a subtraction; its exponential flushes as the parent's
    `ex2.approx.ftz` does, its share a multiply that does not (a weight below fp32's least normal of an
    exponential above it kept). Equal only to itself (`inline_work.body_work`)."""

    scale: float

    def __call__(self, ctx, scores, below, inverse):
        del ctx
        f32 = ir.F32Type.get()
        scale = arith.constant(f32, ir.FloatAttr.get(f32, self.scale))
        log2e = arith.constant(f32, ir.FloatAttr.get(f32, LOG2E))
        registers = np.empty_like(scores.registers)
        for index, register in np.ndenumerate(scores.registers):
            lanes = []
            for lane in range(ir.VectorType(register.type).shape[0]):
                entry, peak, share = (vector.extract(array.registers[index], [], [lane])
                                      for array in (scores, below, inverse))
                lanes.append(mgpu_utils.inline_ptx(WEIGHED, entry, scale, peak, log2e, share,
                                                   result_types=f32))
            registers[index] = vector.from_elements(register.type, lanes)
        return mgpu_fragments.FragmentedArray(_registers=registers, _layout=scores.layout, _is_signed=None)


def weighed(scores: jax.Array, peak: jax.Array, total: jax.Array, scale: float) -> jax.Array:
    """A queries-first tile's normalized weights (`softmax_weights`' second value) from its products or
    hooked logits `scores` (fp32, wgmma's layout) at `scale`, its queries' maxima `peak` and denominators
    `total` (`Weighed`): a query that saw nothing (peak -inf, `safe_peak`'s +inf) weighs 0, a total of 0
    is taken as 1. Unmasked: its caller selects the entries a query sees after it."""
    below = jnp.broadcast_to(-gradient.safe_peak(peak)[:, None], scores.shape)
    inverse = jnp.broadcast_to((1.0 / jnp.where(total == 0, 1.0, total))[:, None], scores.shape)
    inline = plgpu.inline_mgpu(arg_types=(WGMMA, WGMMA, WGMMA),
                               return_type=plgpu.ShapeDtypeStruct(scores.shape, jnp.dtype(jnp.float32),
                                                                  layout=WGMMA))
    return inline(Weighed(scale))(scores, below, inverse)


@dataclasses.dataclass(frozen=True, eq=False)
class SaturatedBf16:
    """`saturated_bf16`'s body: each register's pair of fp32 entries narrowed to bf16 by one
    `cvt.rn.satfinite.bf16x2.f32`, an entry past bf16's largest its largest (an infinity's too), a NaN a
    NaN: `product.bf16_parts`' saturating leading part, its clamp's two FMNMX an entry the conversion's own.
    Equal only to itself (`inline_work.body_work`)."""

    def __call__(self, ctx, tile):
        del ctx
        i32 = ir.IntegerType.get_signless(32)
        pair = ir.VectorType.get((2,), ir.BF16Type.get())
        registers = np.empty_like(tile.registers)
        for index, register in np.ndenumerate(tile.registers):
            if ir.VectorType(register.type).shape != [2]:
                raise ValueError(f"saturated_bf16 narrows fp32 pairs, not {register.type}")
            low, high = (vector.extract(register, [], [lane]) for lane in range(2))
            # The first operand the pair's upper half, the second its lower (PTX's packed conversions).
            packed = mgpu_utils.inline_ptx("cvt.rn.satfinite.bf16x2.f32 $0, $1, $2;", high, low,
                                           result_types=i32)
            registers[index] = mgpu_utils.bitcast(packed, pair)
        return mgpu_fragments.FragmentedArray(_registers=registers, _layout=tile.layout, _is_signed=None)


SATURATED_BF16 = SaturatedBf16()
"""The one `SaturatedBf16` body every `saturated_bf16` inlines."""


def saturated_bf16(tile: jax.Array) -> jax.Array:
    """An fp32 tile in wgmma's accumulator layout narrowed to bf16 saturating at its largest, a conversion a
    pair of entries (`SaturatedBf16`): `product.bf16_parts(tile, 1, saturating=1)`'s one part."""
    inline = plgpu.inline_mgpu(arg_types=(WGMMA,),
                               return_type=plgpu.ShapeDtypeStruct(tile.shape, jnp.dtype(jnp.bfloat16),
                                                                  layout=WGMMA))
    return inline(SATURATED_BF16)(tile)
