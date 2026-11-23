"""BitNet b1.58 ternary weight product (`ternary_weight_product`, row 1464): 2-bit packed ternary weights
(`formats.TernaryWeights`, `{-1, 0, +1}`, 16 codes/uint32) decoded inline in the Mosaic GEMM mainloop
(`MosaicTernaryPackedGmm`) with optional per-token int8 absmax activation quantization and straight-through
estimator (STE) gradients (`fla/modules/fused_bitlinear.py`)."""

from __future__ import annotations

import contextlib
import dataclasses
from typing import ClassVar, NamedTuple

import jax
import jax.numpy as jnp
from jax.core import Tracer

from dew.nn import precision
from dew.nn.kernels import generation, grouped_matmul, selection, xla_products
from dew.nn.kernels.formats import TernaryWeights, fp8_factor, fp8_scale_checked
from dew.nn.kernels.templates.gemm import packed, tiling
from dew.nn.sharding import automatic_axes

TERNARY_WEIGHT_ROW = "ledger 1464"


@dataclasses.dataclass(frozen=True)
class MosaicTernaryPackedGmm(grouped_matmul.MosaicFp8PackedGmm):
    """`MosaicFp8PackedGmm` of 2-bit BitNet b1.58 ternary weights (`formats.TernaryWeights`, rows 1464,
    5387), supporting both staged decode and direct in-register swapped decode at skinny rows."""
    STEPS: ClassVar[tuple[int, ...] | None] = (packed.TERNARY_STEP,)
    WEIGHT_BYTES: ClassVar[float] = 0.25
    DEFAULT_FMT: ClassVar[str] = "ternary"

    def _decode_live(self, ba) -> int:
        del ba
        return 4


class TernaryWeightCase(NamedTuple):
    """Facts deciding whether `ternary_weight_product` runs the inline-decoding Mosaic GEMM mainloop."""
    tile: tiling.Tiles | None
    whole: bool
    interpret: bool


TERNARY_WEIGHT: selection.Op[TernaryWeightCase] = selection.Op(generation.Choice.GROUPED_MATMUL, (
    selection.Kernel("mosaic", (
        selection.Requirement(
            "launch", lambda case: case.tile is not None,
            "no legal Mosaic ternary-packed tile holds this shape on the target", TERNARY_WEIGHT_ROW),
        selection.Requirement(
            "sharding", lambda case: case.whole,
            "automatic mesh axes keep decoded XLA outside per-shard launch", TERNARY_WEIGHT_ROW),
        selection.Requirement(
            "measured", lambda case: case.interpret,
            "no receipt yet has Mosaic ternary-packed beat decoded GEMM (pending M-NG-8)",
            TERNARY_WEIGHT_ROW, kind="measured"),
    )),
    selection.Kernel("xla", (), generic=True),
))


def quantize_activations_int8(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Per-token absmax int8 quantization (`Q_b = 127`) for BitNet b1.58 (`fused_bitlinear`, row 1464).
    Returns `(x_q_bf16, act_scale)` where `x_q_bf16` holds exact integers in `[-128, 127]` as bfloat16
    (every int8 integer is bit-exact in bfloat16) and `act_scale = max(|x|, axis=-1) / 127.0` (`1.0` on
    zero rows so outputs stay zero with finite gradients)."""
    x_f32 = x.astype(jnp.float32)
    absmax = jnp.max(jnp.abs(x_f32), axis=-1)
    act_scale = jnp.where(absmax > 0, precision.divided(absmax, jnp.float32(127.0)), jnp.float32(1.0))
    quantized = jnp.clip(jnp.round(x_f32 / act_scale[:, None]), jnp.float32(-128.0), jnp.float32(127.0))
    return quantized.astype(jnp.bfloat16), act_scale

def ternary_weight_product(lhs: jax.Array, weights: TernaryWeights, group_sizes: jax.Array | None = None, *,
                           quantize_activations: bool = True, out_dtype=None,
                           interpret_on_cpu: bool = False) -> jax.Array:
    """BitNet b1.58 `lhs @ weights` for 2-bit packed ternary `weights` (`formats.TernaryWeights`, row 1464):
    optionally quantizes `lhs` per token to int8 (`quantize_activations=True`, STE gradient w.r.t. `lhs`),
    decodes staged uint32 ternary words inline in the GEMM mainloop on sm80/sm90 into exact bf16
    `{-1, 0, +1}`, accumulates exact integer products in fp32, applies `(act_scale * weights.scale)` once,
    and rounds once to `out_dtype`."""
    from dew.nn.kernels import gemm

    if not isinstance(weights.form, Tracer):
        weights.checked()
    else:
        fp8_scale_checked(jnp.shape(weights.scale), weights.shape)
    dense = group_sizes is None
    if (dense and len(weights.shape) != 2) or (not dense and len(weights.shape) != 3):
        raise ValueError(f"ternary_weight_product weight rank mismatch for dense={dense}: {weights.shape}")
    words = jnp.asarray(weights.words if not dense else weights.words[None])
    sizes = jnp.asarray([lhs.shape[0]], jnp.int32) if dense else group_sizes.astype(jnp.int32)
    if quantize_activations:
        x_q, act_scale = quantize_activations_int8(lhs)
        # Straight-through estimator: forward uses exact int8-quantized bf16 rows; backward passes through.
        scaled_lhs = lhs.astype(jnp.float32) / act_scale[:, None]
        rows = (scaled_lhs + jax.lax.stop_gradient(x_q.astype(jnp.float32) - scaled_lhs)).astype(jnp.bfloat16)
        post_scale = jax.lax.stop_gradient(act_scale)[:, None]
    else:
        rows = lhs.astype(jnp.bfloat16)
        post_scale = jnp.float32(1.0)
    launch = MosaicTernaryPackedGmm()

    def _spec(operand):
        return jax.ShapeDtypeStruct(jnp.shape(operand), jnp.result_type(operand))

    bank_spec = jax.tree.map(_spec, TernaryWeights(words, jnp.ones((), jnp.float32), weights.form))
    bound = launch.bind(_spec(rows), bank_spec, _spec(sizes), (), (), None, None, None,
                        out_dtype="float32", trans_rhs=False, interpret=interpret_on_cpu,
                        dense=dense, exact_sums=False)
    assumed = generation.interpreted_target() if xla_products.interpreted(interpret_on_cpu) else None
    with generation.assumed(assumed) if assumed is not None else contextlib.nullcontext():
        tile = gemm._planned(launch, bound)
        case = TernaryWeightCase(tile, not automatic_axes(), interpret_on_cpu)
        if selection.select(TERNARY_WEIGHT, case).kernel == "mosaic":
            form_bytes = jnp.zeros((8,), jnp.uint8)
            one = jnp.ones((), jnp.float32)
            summed = gemm._packed_weight_product(
                rows, words, sizes, launch, lambda w: TernaryWeights(w, one, form_bytes),
                interpret_on_cpu, dense=dense, contract="test_a_ternary_weight_product_holds_every_order")
        else:
            summed = gemm.gemm(rows, weights.unscaled(jnp.bfloat16), group_sizes, out_dtype=jnp.float32,
                               precision=jax.lax.Precision.DEFAULT, interpret_on_cpu=interpret_on_cpu)
            assert isinstance(summed, jax.Array)
    factor = fp8_factor(jnp.asarray(weights.scale, jnp.float32), sizes, rows.shape[0], weights.shape, dense)
    return (summed * (post_scale * factor)).astype(lhs.dtype if out_dtype is None else out_dtype)
