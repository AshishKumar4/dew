"""LoRA's adapted product as one GEMM template product over the depth the rank extends (`low_rank`, Appendix C
row 26, ledger 1304), supporting both dense and grouped MoE adapters, and its selector (`LOW_RANK`)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax.typing import PrecisionLike
from jax.custom_derivatives import zero_from_primal

from dew.nn.derivatives import KernelGlue, contracted, declared_derivative, with_gradients_formed
from dew.nn.kernels import generation, selection, xla_products
from dew.nn.kernels.gemm import gemm_t, joined_depths
from dew.nn.precision import EXACT_FLOAT32, at_least_fp32, effective_precision, refuse_recipe
from dew.nn.programs import dot_arithmetic
from dew.nn.sharding import excess_axes, summed_to

LOW_RANK_ROW = 'ledger 1304'
"""The adapted product's fused form (docs/design/kernel-templates.md, Appendix C row 26)."""

LOW_RANK: selection.Op[None] = selection.Op(generation.Choice.LOW_RANK, (
    selection.Kernel('fused', (selection.Requirement(
        'measured', lambda case: False, "no receipt yet has the fused form beat the base product and the "
        "update apart", LOW_RANK_ROW, kind='measured'),)),
    selection.Kernel('xla', (), generic=True)))
"""An adapted projection's selector (`low_rank`'s caller, `dew.lora`): one template product over the depth
the rank extends (`low_rank`) where named (`generation.chosen(low_rank='fused')`), which passes the
measured requirement no receipt has met yet; else its base product and its update apart, the adapter's own
program."""


def low_rank(lhs: jax.Array, kernel: jax.Array, a: jax.Array, b: jax.Array, scale,
             group_sizes: jax.Array | None = None, *, out_dtype, precision: PrecisionLike = None,
             interpret_on_cpu: bool = False) -> jax.Array:
    """`lhs · kernel + scale · (lhs · a) · b`, LoRA's adapted product (Appendix C row 26, strategy (b)), for
    both dense (`group_sizes=None`: `kernel [depth, width]`, `a [depth, rank]`, `b [rank, width]`) and
    grouped MoE (`group_sizes [G]`: `kernel [G, depth, width]`, `a [G, depth, rank]`, `b [G, rank, width]`):
    one template product over the extended depth via `gemm.joined_depths` (zero-copy `chained` walk when
    Mosaic takes it, row 1304), forward and reverse, computing a frozen kernel's gradient only when
    perturbed."""
    refuse_recipe("an adapter's fused product (`gemm.low_rank`)", *kernel.shape[-2:])
    return _low_rank(lhs, kernel, a, b, jnp.asarray(scale, jnp.float32),
                     None if group_sizes is None else group_sizes.astype(jnp.int32), out_dtype, precision,
                     interpret_on_cpu)


def _rank_product(lhs: jax.Array, factor: jax.Array, scale: jax.Array, precision: PrecisionLike,
                  group_sizes: jax.Array | None = None) -> jax.Array:
    """`scale · (lhs · factor)`, summed and kept in at least fp32 (dense or grouped by `group_sizes`): the
    rank-wide operand an adapted product appends to its depth, rounded once to `lhs`'s dtype where it is
    appended (`_extended`)."""
    wide = at_least_fp32(lhs.dtype)
    if group_sizes is None:
        summed = jnp.dot(lhs, factor.astype(lhs.dtype), precision=precision, preferred_element_type=wide)
    else:
        summed = xla_products.xla_ragged_dot(lhs, factor.astype(lhs.dtype), group_sizes, precision=precision,
                                             preferred_element_type=wide)
    return scale.astype(wide) * summed


def _extended(lhs, kernel, a, b, scale, group_sizes, out_dtype, precision, interpret_on_cpu):
    """`low_rank`'s value and its appended operand `t`, in at least fp32: `[lhs | t] · [kernel ; b]`, `t`
    rounded once to `lhs`'s dtype where it joins the depth (`gemm.joined_depths`)."""
    t = _rank_product(lhs, a, scale, precision, group_sizes)
    out = joined_depths(((lhs, kernel), (t.astype(lhs.dtype), b)), group_sizes, out_dtype=out_dtype,
                        precision=precision, interpret_on_cpu=interpret_on_cpu)
    return out, t


def _value(lhs, kernel, a, b, scale, group_sizes, out_dtype, precision, interpret_on_cpu):
    return _extended(lhs, kernel, a, b, scale, group_sizes, out_dtype, precision, interpret_on_cpu)[0]


def _forward(lhs, kernel, a, b, scale, group_sizes, out_dtype, precision, interpret_on_cpu):
    primals = (lhs, kernel, a, b, scale)
    values = (lhs.value, kernel.value, a.value, b.value, scale.value)
    # A dense call's `group_sizes` is None, a tree of no leaves, which reaches the rule as it is.
    sizes = None if group_sizes is None else group_sizes.value
    out, t = _extended(*values, sizes, out_dtype, precision, interpret_on_cpu)
    # Which operands are differentiated, as the residuals' static shapes: a frozen kernel's gradient is
    # never computed.
    perturbed = jnp.zeros(tuple(int(primal.perturbed) for primal in primals), jnp.int8)
    return out, (*values, sizes, t, perturbed)


def _backward(out_dtype, precision, interpret_on_cpu, residuals, cotangent):
    del out_dtype
    lhs, kernel, a, b, scale, group_sizes, t, perturbed = residuals
    wants_lhs, wants_kernel, wants_a, wants_b, wants_scale = map(bool, perturbed.shape)
    # Every product takes the cotangent at its own precision where it is wider than the rows (an fp32
    # output's of fp16 rows), the rows' dtype otherwise, as the plain product's input and kernel gradients
    # do (`gemm._worked`): an fp32 cotangent past fp16's largest rounded to fp16 first would be inf.
    work = jnp.promote_types(cotangent.dtype, lhs.dtype)
    dy = cotangent.astype(work)
    wide = at_least_fp32(lhs.dtype)
    # The cotangent of `t` before its scale, dy · bᵀ (b as the forward multiplied it, in the rows' dtype),
    # summed wide, and after it: the scale taken before any product with it, as the adapter's own program
    # takes it (a product near float32's largest that the scale brings back in range never overflows first).
    matrix = b.astype(lhs.dtype).astype(work)
    # A product of the cotangent's width at a precision keeping it (`_kept`), where that width is fp32's.
    at_work = _kept(precision) if jnp.dtype(work).itemsize >= 4 else precision
    if group_sizes is None:
        through = jnp.dot(dy, matrix.T, precision=at_work, preferred_element_type=wide)
    else:
        through = xla_products.xla_ragged_dot(dy, jnp.swapaxes(matrix, 1, 2), group_sizes,
                                              precision=at_work, preferred_element_type=wide)
    scaled = scale.astype(wide) * through
    d_lhs = d_kernel = d_a = d_b = d_scale = None
    if wants_lhs:
        # The matrices as the forward multiplied them, in the rows' dtype, joined with the cotangents at the
        # cotangent's width; where the scaled update's cotangent (`scaled`, wide) is no normal value of that
        # width but zero (past fp16's largest at 2**16 or below its least normal, bf16's top binade), at
        # fp32, its precision one that keeps fp32 operands (`_kept`): the narrow product is the fast one
        # every call in range takes.
        matrices = (kernel.astype(lhs.dtype), a.astype(lhs.dtype))

        def joined(dtype):
            wide_operands = jnp.dtype(dtype).itemsize >= 4
            return joined_depths(((dy.astype(dtype), matrices[0]), (scaled.astype(dtype), matrices[1])),
                                 group_sizes, transpose_rhs=True, out_dtype=lhs.dtype,
                                 precision=_kept(precision) if wide_operands else precision,
                                 interpret_on_cpu=interpret_on_cpu)

        if jnp.dtype(work).itemsize >= 4:
            d_lhs = joined(work)
        else:
            size, limits = jnp.abs(scaled), jnp.finfo(work)
            held = jnp.all((size == 0) | ((size <= limits.max) & (size >= limits.tiny)))
            d_lhs = jax.lax.cond(held, lambda: joined(work), lambda: joined(wide))

    # A replicated operand's gradient inside a checked `shard_map` is each device's partial over the rows it
    # holds, summed over the axes the rows vary over and the operand does not (`sharding.summed_to`), each
    # partial kept wide until the sum is formed and rounded once to the operand's dtype.
    def formed_in(operand):
        summed = excess_axes(dy, operand) or excess_axes(lhs, operand)
        return jnp.promote_types(wide, operand.dtype) if summed else operand.dtype

    if wants_kernel:
        d_kernel = gemm_t(lhs.astype(work), dy, group_sizes, out_dtype=formed_in(kernel), precision=at_work,
                          interpret_on_cpu=interpret_on_cpu)
        d_kernel = d_kernel[0] if group_sizes is None else d_kernel
    # The factors' gradients from the wide `scaled` (the rank-wide product's sum, unrounded) and from `t` as
    # the forward multiplied it, rounded to the rows' dtype: fp32 operands each, at a precision keeping them.
    kept = _kept(precision)
    if wants_a:
        d_a = (jnp.dot(lhs.T.astype(wide), scaled, precision=kept, preferred_element_type=wide)
               if group_sizes is None
               else gemm_t(lhs.astype(wide), scaled, group_sizes, out_dtype=formed_in(a), precision=kept,
                           interpret_on_cpu=interpret_on_cpu))
    if wants_b:
        # `t` holds the scale already: tᵀ · dy is the factor's gradient whole.
        rank = t.astype(lhs.dtype).astype(wide)
        d_b = (jnp.dot(rank.T, dy.astype(wide), precision=kept, preferred_element_type=wide)
               if group_sizes is None
               else gemm_t(rank, dy.astype(wide), group_sizes, out_dtype=formed_in(b), precision=kept,
                           interpret_on_cpu=interpret_on_cpu))
    if wants_scale:
        d_scale = jnp.sum(through * _rank_product(lhs, a, jnp.ones((), wide), precision, group_sizes))
    gradients = tuple(None if d is None else summed_to(d, operand).astype(operand.dtype) for d, operand in
                      zip((d_kernel, d_a, d_b, d_scale), (kernel, a, b, scale), strict=True))
    d_lhs, gradients = with_gradients_formed(d_lhs, gradients)
    return d_lhs, *gradients, zero_from_primal(group_sizes)


def _kept(precision: PrecisionLike) -> PrecisionLike:
    """`precision` for a product of fp32 operands: as given (a run's configured one where it states none),
    unless the dot algorithm it names keeps fewer of an fp32 operand's digits than fp32's
    (`programs.dot_arithmetic`'s `digits`: an F16, BF16 or TF32 preset, a `jax.lax.DotAlgorithm` of such
    types, or one JAX refuses fp32 operands for), which would round them back to those, then `HIGHEST`."""
    try:
        arithmetic = dot_arithmetic(effective_precision(precision), jnp.float32, jnp.float32)
    except ValueError:
        # JAX's own refusal of fp32 operands for the algorithm (an FP8 preset takes FP8 operands alone): it
        # keeps no fp32 operand whole. A narrow product's fp32 sum, exact.
        return EXACT_FLOAT32
    narrow = arithmetic is not None and arithmetic.digits < jnp.finfo(jnp.float32).nmant + 1
    # A narrow product's fp32 sum, exact: its operands' digits would round them back.
    return EXACT_FLOAT32 if narrow else precision


# The launches' derivative glue (`derivatives.KernelGlue`): an operand no reverse pass reaches is a symbolic
# zero, whose gradient is never formed.
_low_rank = contracted("test_an_adapted_product_holds_every_reverse_order")(
    declared_derivative(KernelGlue(), _value, _forward, _backward, nondiff_argnums=(6, 7, 8),
                        symbolic_zeros=True))
