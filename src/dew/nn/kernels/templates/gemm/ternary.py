"""BitNet b1.58 ternary weight product (`ternary_weight_product`, row 1464): 2-bit packed ternary weights
(`formats.TernaryWeights`, `{-1, 0, +1}`, 16 codes/uint32) decoded inline in the Mosaic GEMM mainloop
(`MosaicTernaryPackedGmm`) with optional per-token int8 absmax activation quantization and straight-through
estimator (STE) gradients (`fla/modules/fused_bitlinear.py`)."""

from __future__ import annotations

import dataclasses
import functools
from typing import ClassVar

import jax
import jax.numpy as jnp

from dew.nn import float_bits, precision
from dew.nn.derivatives import KernelGlue, contracted, surrogate
from dew.nn.kernels import gemm, generation, grouped_matmul, selection, xla_products
from dew.nn.kernels.formats import TernaryWeights, scale_factor
from dew.nn.kernels.templates.gemm.factored import factored
from dew.nn.sharding import automatic_axes

TERNARY_WEIGHT_ROW = "ledger 1464"


@dataclasses.dataclass(frozen=True)
class MosaicTernaryPackedGmm(grouped_matmul.MosaicWordPackedGmm):
    """`MosaicWordPackedGmm` of 2-bit BitNet b1.58 ternary weights (`formats.TernaryWeights`, rows 1464,
    5387), supporting both staged decode and direct in-register swapped decode at skinny rows."""
    DEFAULT_FMT: ClassVar[str] = "ternary"


TERNARY_WEIGHT = gemm.word_packed_op("ternary", TERNARY_WEIGHT_ROW, "M-NG-8")


def _quantize_activations_peak(x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Per-token absmax int8 codes of `x` and each row's peak, read from bits (`float_bits.binade_of`,
    `float_bits.raised_exactly`): a row whose entries a nonzero code reads (down to `peak / 2**8`) reach
    below fp32's least normal is raised by `2**32` first (`shift`), so a process whose float arithmetic
    flushes subnormals (XLA's CPU) still codes a `2**-144` row (row 1464). Returns `(x_q_bf16, peak_raised,
    shift)`, `peak = peak_raised * 2**-shift` (`0` on zero rows)."""
    if x.shape[-1] == 0:
        return (x.astype(jnp.bfloat16), jnp.zeros(x.shape[:-1], jnp.float32),
                jnp.zeros(x.shape[:-1], jnp.int32))
    x_f32 = float_bits.widened_exactly(x, jnp.float32)
    row_exp = jnp.max(float_bits.binade_of(x_f32), axis=-1)
    shift = jnp.where((row_exp > -1024) & (row_exp - 8 < jnp.finfo(jnp.float32).minexp), jnp.int32(32),
                      jnp.int32(0))
    raised = float_bits.raised_exactly(x_f32, shift[..., None])
    peak_raised = jnp.max(jnp.abs(raised), axis=-1)
    safe_peak = jnp.where(peak_raised > 0, peak_raised, jnp.float32(1.0))
    normalized = precision.divided(raised, safe_peak[..., None])
    quantized = jnp.clip(jnp.round(normalized * jnp.float32(127.0)), jnp.float32(-128.0), jnp.float32(127.0))
    return quantized.astype(jnp.bfloat16), peak_raised, shift


def quantize_activations_int8(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Per-token absmax int8 quantization (`Q_b = 127`) for BitNet b1.58 (`fused_bitlinear`, row 1464).
    Returns `(x_q_bf16, act_scale)` where `x_q_bf16` holds exact integers in `[-128, 127]` as bfloat16
    (every int8 integer is bit-exact in bfloat16) and `act_scale = max(|x|, axis=-1) / 127.0` (`1.0` on
    zero rows)."""
    quantized, peak_raised, shift = _quantize_activations_peak(x)
    scaled = float_bits.lowered_rounded(precision.divided(peak_raised, jnp.float32(127.0)), shift)
    act_scale = jnp.where(peak_raised > 0, scaled, jnp.float32(1.0))
    return quantized, act_scale


def ternary_weight_product(lhs: jax.Array, weights: TernaryWeights, group_sizes: jax.Array | None = None, *,
                           quantize_activations: bool = True, out_dtype=None,
                           interpret_on_cpu: bool = False) -> jax.Array:
    """BitNet b1.58 `lhs @ weights` for 2-bit packed ternary `weights` (`formats.TernaryWeights`, row 1464):
    optionally quantizes `lhs` per token to int8 (`quantize_activations=True`, STE gradient w.r.t. `lhs`),
    decodes staged uint32 ternary words inline in the GEMM mainloop on sm80/sm90 into exact bf16
    `{-1, 0, +1}`, accumulates exact integer products in fp32, applies `(peak / 127) * weights.scale` via
    `precision.twice_scaled`, and rounds once to `out_dtype`. Its scale is zero or normal
    (`TernaryWeights.checked`: a subnormal one refused, row 13340), so its products are normal wherever their
    values are, a process that flushes a subnormal operand forming them as a GPU does."""
    weights.checked()
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
    with xla_products.interpreted_context(interpret_on_cpu):
        case = gemm.WordPackedCase(functools.partial(gemm._planned, launch, bound), not automatic_axes())
        mosaic = selection.select(TERNARY_WEIGHT, case).kernel == "mosaic"
    if mosaic:
        form_bytes, one = jnp.zeros((8,), jnp.uint8), jnp.ones((), jnp.float32)
        rules = gemm._packed_weight_rules(launch, lambda w: TernaryWeights(w, one, form_bytes),
                                          interpret_on_cpu, dense=dense)
        product = contracted("test_a_ternary_weight_product_holds_every_order")(
            generation.linear_tangent(*rules, live_cotangents=True, kind=KernelGlue()))
        operands = (words, sizes)
    else:
        product = functools.partial(gemm.decoded_product, interpret_on_cpu=interpret_on_cpu)
        operands = (weights.unscaled(jnp.bfloat16), group_sizes)
    run = functools.partial(gemm._interpreted, product, interpret_on_cpu)
    wide = gemm.interpreted_wide(interpret_on_cpu)
    # The codes alone for the rescue's decoded matrix, no scale: a derivative rule closes over no
    # differentiated value.
    codes = (dataclasses.replace(weights, scale=jnp.ones((), jnp.float32)), group_sizes)

    def run_dot(rows_bf16: jax.Array) -> jax.Array:
        return run(rows_bf16, *operands)

    def factor_of(scale_arr: jax.Array) -> jax.Array:
        return scale_factor(jnp.asarray(scale_arr, jnp.float32), sizes, lhs.shape[0], weights.shape, dense)

    scale_f32 = jnp.asarray(weights.scale, jnp.float32)
    if not quantize_activations:
        scaled = factored(run, operands, factor_of(scale_f32), lhs, wide=wide,
                          wide_operands=codes, prepared=gemm.decoded_operands)
        return scaled.astype(lhs.dtype if out_dtype is None else out_dtype)

    def ste_value(a: jax.Array, s: jax.Array) -> tuple[jax.Array, jax.Array]:
        x_q, peak_raised, shift = _quantize_activations_peak(a)
        prod = precision.divided(run_dot(x_q), jnp.float32(127.0))
        # A raised row's output lowered back by its shift, rounded once (`lowered_rounded`), past FTZ.
        raised = precision.twice_scaled(prod, peak_raised[:, None], factor_of(s))
        return float_bits.lowered_rounded(raised, shift[:, None]), prod

    def ste_derivative(primals, tangents, value):
        a, s = primals
        d_a, d_s = tangents
        _, prod = value
        _, peak_raised, shift = _quantize_activations_peak(a)
        peak_col = jax.lax.stop_gradient(peak_raised)[:, None]
        shift_scale = float_bits.exact_power2(-jax.lax.stop_gradient(shift))[:, None]
        prod_held = jax.lax.stop_gradient(prod)
        factor = factor_of(s)
        d_factor = factor_of(d_s)
        dy_a = factored(run, operands, factor, d_a, wide=wide, wide_operands=codes,
                        prepared=gemm.decoded_operands)
        once = prod_held * peak_col
        tiny = jnp.finfo(prod_held.dtype).tiny
        kept = jnp.isfinite(once) & ((jnp.abs(once) >= tiny) | (prod_held == 0))
        dy_s = jnp.where(kept, (once * d_factor) * shift_scale,
                         ((prod_held * d_factor) * peak_col) * shift_scale)
        return dy_a + dy_s, jnp.zeros_like(prod_held)

    ste_scaled = contracted("test_a_ternary_weight_product_holds_every_order")(
        surrogate(ste_value, ste_derivative))
    return ste_scaled(lhs, scale_f32)[0].astype(lhs.dtype if out_dtype is None else out_dtype)
