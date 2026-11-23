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
from dew.nn.derivatives import KernelGlue, contracted
from dew.nn.kernels import generation, grouped_matmul, selection, xla_products
from dew.nn.kernels.formats import TernaryWeights, fp8_factor, fp8_scale_checked
from dew.nn.kernels.templates.gemm import tiling
from dew.nn.sharding import automatic_axes

TERNARY_WEIGHT_ROW = "ledger 1464"


@dataclasses.dataclass(frozen=True)
class MosaicTernaryPackedGmm(grouped_matmul.MosaicWordPackedGmm):
    """`MosaicWordPackedGmm` of 2-bit BitNet b1.58 ternary weights (`formats.TernaryWeights`, row 1464)."""
    DEFAULT_FMT: ClassVar[str] = "ternary"


class TernaryWeightCase(NamedTuple):
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


def _quantize_activations_peak(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Per-token absmax int8 quantization normalized by `peak` BEFORE multiplying by `127.0`, so tiny
    subnormal nonzero rows (e.g. `2**-144`) never divide by an underflowed `peak / 127.0` (row 1464)."""
    if x.shape[-1] == 0:
        return x.astype(jnp.bfloat16), jnp.zeros(x.shape[:-1], jnp.float32)
    x_f32 = x.astype(jnp.float32)
    peak = jnp.max(jnp.abs(x_f32), axis=-1)
    safe_peak = jnp.where(peak > 0, peak, jnp.float32(1.0))
    normalized = x_f32 / safe_peak[:, None]
    quantized = jnp.clip(jnp.round(normalized * jnp.float32(127.0)), jnp.float32(-128.0), jnp.float32(127.0))
    return quantized.astype(jnp.bfloat16), peak


def quantize_activations_int8(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Per-token absmax int8 quantization (`Q_b = 127`) for BitNet b1.58 (`fused_bitlinear`, row 1464).
    Returns `(x_q_bf16, act_scale)` where `x_q_bf16` holds exact integers in `[-128, 127]` as bfloat16
    (every int8 integer is bit-exact in bfloat16) and `act_scale = max(|x|, axis=-1) / 127.0` (`1.0` on
    zero rows)."""
    quantized, peak = _quantize_activations_peak(x)
    act_scale = jnp.where(peak > 0, precision.divided(peak, jnp.float32(127.0)), jnp.float32(1.0))
    return quantized, act_scale


def ternary_weight_product(lhs: jax.Array, weights: TernaryWeights, group_sizes: jax.Array | None = None, *,
                           quantize_activations: bool = True, out_dtype=None,
                           interpret_on_cpu: bool = False) -> jax.Array:
    """BitNet b1.58 `lhs @ weights` for 2-bit packed ternary `weights` (`formats.TernaryWeights`, row 1464):
    optionally quantizes `lhs` per token to int8 (`quantize_activations=True`, STE gradient w.r.t. `lhs`),
    decodes staged uint32 ternary words inline in the GEMM mainloop on sm80/sm90 into exact bf16
    `{-1, 0, +1}`, accumulates exact integer products in fp32, applies `(peak / 127) * weights.scale` via
    `precision.twice_scaled`, and rounds once to `out_dtype`."""
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
    launch = MosaicTernaryPackedGmm()

    def _spec(operand):
        return jax.ShapeDtypeStruct(jnp.shape(operand), jnp.result_type(operand))

    bank_spec = jax.tree.map(_spec, TernaryWeights(words, jnp.ones((), jnp.float32), weights.form))
    bound = launch.bind(jax.ShapeDtypeStruct(lhs.shape, jnp.bfloat16), bank_spec, _spec(sizes), (), (),
                        None, None, None, out_dtype="float32", trans_rhs=False, interpret=interpret_on_cpu,
                        dense=dense, exact_sums=False)
    assumed = generation.interpreted_target() if xla_products.interpreted(interpret_on_cpu) else None
    with generation.assumed(assumed) if assumed is not None else contextlib.nullcontext():
        tile = gemm._planned(launch, bound)
        case = TernaryWeightCase(tile, not automatic_axes(), interpret_on_cpu)
        use_mosaic = selection.select(TERNARY_WEIGHT, case).kernel == "mosaic"

    def run_dot(rows_bf16: jax.Array) -> jax.Array:
        with generation.assumed(assumed) if assumed is not None else contextlib.nullcontext():
            if use_mosaic:
                form_bytes = jnp.zeros((8,), jnp.uint8)
                one = jnp.ones((), jnp.float32)
                rules = gemm._packed_weight_rules(launch, lambda w: TernaryWeights(w, one, form_bytes),
                                                  interpret_on_cpu, dense=dense)
                differentiable = contracted("test_a_ternary_weight_product_holds_every_order")(
                    generation.linear_tangent(*rules, live_cotangents=True, kind=KernelGlue()))
                return differentiable(rows_bf16, words, sizes)
            summed = gemm.gemm(rows_bf16, weights.unscaled(jnp.bfloat16), group_sizes,
                               out_dtype=jnp.float32, precision=jax.lax.Precision.DEFAULT,
                               interpret_on_cpu=interpret_on_cpu)
            assert isinstance(summed, jax.Array)
            return summed

    def factor_of(scale_arr: jax.Array) -> jax.Array:
        return fp8_factor(jnp.asarray(scale_arr, jnp.float32), sizes, lhs.shape[0], weights.shape, dense)

    scale_f32 = jnp.asarray(weights.scale, jnp.float32)
    if not quantize_activations:
        scaled = run_dot(lhs.astype(jnp.bfloat16)) * factor_of(scale_f32)
        return scaled.astype(lhs.dtype if out_dtype is None else out_dtype)

    @jax.custom_jvp
    def ste_scaled(a: jax.Array, s: jax.Array) -> jax.Array:
        x_q, peak = _quantize_activations_peak(a)
        prod = run_dot(x_q) / jnp.float32(127.0)
        return precision.twice_scaled(prod, peak[:, None], factor_of(s))

    @ste_scaled.defjvp
    def _ste_scaled_jvp(primals, tangents):
        a, s = primals
        d_a, d_s = tangents
        x_q, peak = _quantize_activations_peak(a)
        peak_col = jax.lax.stop_gradient(peak)[:, None]
        # Straight-through activation derivative at the scaled product: never carries canceling `peak`
        # into the pullback (preventing `act_scale * factor -> inf` overflow on canceling rows), while
        # delegating to `run_dot`'s `linear_tangent` so VJP transposes through the GEMM backward kernel.
        summed, d_summed = jax.jvp(run_dot, (jax.lax.stop_gradient(x_q),), (d_a.astype(jnp.bfloat16),))
        prod = jax.lax.stop_gradient(summed) / jnp.float32(127.0)
        factor = factor_of(s)
        d_factor = factor_of(d_s)
        y = precision.twice_scaled(prod, peak_col, factor)
        once = prod * peak_col
        tiny = jnp.finfo(prod.dtype).tiny
        kept = jnp.isfinite(once) & ((jnp.abs(once) >= tiny) | (prod == 0))
        dy_s = jnp.where(kept, once * d_factor, (prod * d_factor) * peak_col)
        return y, d_summed * factor + dy_s

    return ste_scaled(lhs, scale_f32).astype(lhs.dtype if out_dtype is None else out_dtype)
