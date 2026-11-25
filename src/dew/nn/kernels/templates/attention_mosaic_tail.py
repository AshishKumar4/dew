"""The fused backward's lean score tail on wgmma (`attention_mosaic_scores._tile_scores`' `lean`), lowered in
its accumulators' registers (a `plgpu.inline_mgpu` body): its bf16 score cotangents narrowed saturating by
one conversion a pair (`saturated_bf16`). M-NA-28-gh200: the fused dK/dV's 2.02B instructions against
FlashAttention 3's 1.08B, its excess on the score tile (136M FMNMX against none, row 5272)."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.lib.mlir.dialects import vector
from jax.experimental.mosaic.gpu import fragmented_array as mgpu_fragments, utils as mgpu_utils
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend.mlir import ir

WGMMA = plgpu.Layout.WGMMA
"""wgmma's accumulators' layout, a tile's registers' the body reads and writes."""

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
