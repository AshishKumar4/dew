"""BitNet b1.58 ternary weight product (`ternary_weight_product`, row 1464): 2-bit packed ternary weights
(`formats.TernaryWeights`, `{-1, 0, +1}`, 16 codes/uint32) decoded inline in the Mosaic GEMM mainloop
(`MosaicTernaryPackedGmm`) with optional per-token int8 absmax activation quantization and straight-through
estimator (STE) gradients (`fla/modules/fused_bitlinear.py`)."""

from __future__ import annotations

import contextlib
import dataclasses
import functools
from typing import ClassVar, NamedTuple

import jax
import jax.numpy as jnp
from jax.core import Tracer
from jax.custom_derivatives import zero_from_primal

from dew.nn.derivatives import KernelGlue, contracted
from dew.nn.kernels import generation, grouped_matmul, selection, xla_products
from dew.nn.kernels.formats import TernaryWeights, fp8_factor, fp8_scale_checked
from dew.nn.kernels.templates import resources
from dew.nn.kernels.templates.gemm import packed, tiling
from dew.nn.kernels.templates.product import bf16_part_count
from dew.nn.sharding import automatic_axes

TERNARY_WEIGHT_ROW = "ledger 1464"


@dataclasses.dataclass(frozen=True)
class MosaicTernaryPackedGmm(grouped_matmul.MosaicFp8PackedGmm):
    """`MosaicFp8PackedGmm` of 2-bit BitNet b1.58 ternary weights (`formats.TernaryWeights`, rows 1464,
    5387), supporting both staged decode and direct in-register swapped decode at skinny rows."""
    STEPS: ClassVar[tuple[int, ...] | None] = (packed.TERNARY_STEP,)

    def _operand_bytes(self, ba, tile: tiling.Tiles) -> tuple[tuple[float, float], tuple]:
        tokens, kernel = ba.arguments["tokens"], ba.arguments["kernel"]
        rows_b, stage = jnp.dtype(tokens.dtype).itemsize, jnp.dtype(kernel.dtype).itemsize
        if tile.swapped:
            return (0.25, rows_b), ((0, 1), None)
        copies = 1 if packed.shared_decode(tile, interpret=False) else tile.warpgroups
        return (rows_b, 0.25), (None, (stage, copies))

    def _registers(self, ba, config: tiling.Tiles, *, merging: bool, target) -> int | None:
        base = grouped_matmul.MosaicGmm._registers(self, ba, config, merging=merging, target=target)
        if base is None:
            return None
        share = config.block_m // config.warpgroups
        decoded = (2 * share * 16 + share * config.block_k // 2 if ba.arguments["packed_rows"]
                   else 2 * config.block_n * 16)
        return base + decoded // resources.THREADS + 4
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
    act_scale = jnp.where(absmax > 0, absmax / jnp.float32(127.0), jnp.float32(1.0))
    quantized = jnp.clip(jnp.round(x_f32 / act_scale[:, None]), jnp.float32(-128.0), jnp.float32(127.0))
    return quantized.astype(jnp.bfloat16), act_scale


def _ternary_packed(rows: jax.Array, words: jax.Array, sizes: jax.Array,
                    launch: MosaicTernaryPackedGmm, interpret: bool, *,
                    dense: bool) -> jax.Array:
    """Inline-decoding ternary weight product in fp32 (`ternary_weight_product`, row 1464), differentiable
    at every order through `generation.linear_tangent`."""
    from dew.nn.kernels import gemm

    form = TernaryWeights(words, jnp.ones((), jnp.float32), jnp.zeros((8,), jnp.uint8))

    def decoded(packed_words: jax.Array) -> jax.Array:
        matrices = TernaryWeights(packed_words, jnp.ones((), jnp.float32), form.form).unscaled()
        return matrices[0] if dense else matrices

    def kernel(r: jax.Array, w: jax.Array, s: jax.Array, *, interpret: bool) -> jax.Array:
        bank = TernaryWeights(w, jnp.ones((), jnp.float32), form.form)
        return launch(r, bank, s, (), (), None, None, None, out_dtype="float32", trans_rhs=False,
                      interpret=interpret, dense=dense, exact_sums=False)

    def portable(r: jax.Array, w: jax.Array, s: jax.Array) -> jax.Array:
        matrices = decoded(w)
        return (jax.lax.dot(r, matrices, preferred_element_type=jnp.float32) if dense
                else jax.lax.ragged_dot(r, matrices, s, preferred_element_type=jnp.float32))

    def forward(r: jax.Array, w: jax.Array, s: jax.Array):
        ctx = (generation.assumed(generation.interpreted_target())
               if xla_products.interpreted(interpret) else contextlib.nullcontext())
        with ctx:
            return xla_products.dispatched(kernel, portable, interpret, r, w, s), ()

    def backward(saved, operands, cotangent, live):
        del saved
        r, w, s = operands
        matrices = decoded(w)
        parts = gemm._bf16_parts(cotangent, bf16_part_count(jnp.float32))
        ctx = (generation.assumed(generation.interpreted_target())
               if xla_products.interpreted(interpret) else contextlib.nullcontext())
        with ctx:
            d_rows = functools.reduce(jnp.add, [
                gemm.gemm(part, matrices, None if dense else s, transpose_rhs=True, out_dtype=jnp.float32,
                          precision=jax.lax.Precision.DEFAULT, interpret_on_cpu=interpret) for part in parts])
        leaves = (d_rows.astype(r.dtype), zero_from_primal(w), zero_from_primal(s))
        return tuple(leaves[index] for index in live)

    differentiable = contracted("test_a_ternary_weight_product_holds_every_order")(
        generation.linear_tangent(forward, backward, portable, live_cotangents=True, kind=KernelGlue()))
    return differentiable(rows, words, sizes)


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
            summed = _ternary_packed(rows, words, sizes, launch, interpret_on_cpu, dense=dense)
        else:
            summed = gemm.gemm(rows, weights.unscaled(jnp.bfloat16), group_sizes, out_dtype=jnp.float32,
                               precision=jax.lax.Precision.DEFAULT, interpret_on_cpu=interpret_on_cpu)
            assert isinstance(summed, jax.Array)
    factor = fp8_factor(jnp.asarray(weights.scale, jnp.float32), sizes, rows.shape[0], weights.shape, dense)
    return (summed * (post_scale * factor)).astype(lhs.dtype if out_dtype is None else out_dtype)
