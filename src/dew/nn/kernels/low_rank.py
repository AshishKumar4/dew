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
from dew.nn.precision import at_least_fp32, refuse_recipe

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
    scale_arr = jnp.asarray(scale, jnp.float32)
    if group_sizes is None:
        return _low_rank_dense(lhs, kernel, a, b, scale_arr, out_dtype, precision, interpret_on_cpu)
    return _low_rank_grouped(lhs, kernel, a, b, scale_arr, group_sizes.astype(jnp.int32), out_dtype,
                             precision, interpret_on_cpu)


def _rank_product(lhs: jax.Array, factor: jax.Array, scale: jax.Array, precision: PrecisionLike,
                  group_sizes: jax.Array | None = None) -> jax.Array:
    """`scale · (lhs · factor)`, summed and kept in at least fp32 (dense or grouped by `group_sizes`)."""
    wide = at_least_fp32(lhs.dtype)
    if group_sizes is None:
        summed = jnp.dot(lhs, factor.astype(lhs.dtype), precision=precision, preferred_element_type=wide)
    else:
        summed = xla_products.xla_ragged_dot(lhs, factor.astype(lhs.dtype), group_sizes, precision=precision,
                                             preferred_element_type=wide)
    return scale.astype(wide) * summed


def _extended(lhs, kernel, a, b, scale, group_sizes, out_dtype, precision, interpret_on_cpu):
    """`low_rank`'s value and its appended operand `t`, in at least fp32: `[lhs | t] · [kernel ; b]`."""
    t = _rank_product(lhs, a, scale, precision, group_sizes)
    out = joined_depths(((lhs, kernel), (t.astype(lhs.dtype), b)), group_sizes, out_dtype=out_dtype,
                        precision=precision, interpret_on_cpu=interpret_on_cpu)
    return out, t


def _dense_value(lhs, kernel, a, b, scale, out_dtype, precision, interpret_on_cpu):
    return _extended(lhs, kernel, a, b, scale, None, out_dtype, precision, interpret_on_cpu)[0]


def _dense_forward(lhs, kernel, a, b, scale, out_dtype, precision, interpret_on_cpu):
    primals = (lhs, kernel, a, b, scale)
    values = (lhs.value, kernel.value, a.value, b.value, scale.value)
    out, t = _extended(*values, None, out_dtype, precision, interpret_on_cpu)
    perturbed = jnp.zeros(tuple(int(primal.perturbed) for primal in primals), jnp.int8)
    return out, (*values, t, perturbed)


def _grouped_value(lhs, kernel, a, b, scale, group_sizes, out_dtype, precision, interpret_on_cpu):
    return _extended(lhs, kernel, a, b, scale, group_sizes, out_dtype, precision, interpret_on_cpu)[0]


def _grouped_forward(lhs, kernel, a, b, scale, group_sizes, out_dtype, precision, interpret_on_cpu):
    primals = (lhs, kernel, a, b, scale)
    values = (lhs.value, kernel.value, a.value, b.value, scale.value)
    sizes = group_sizes.value
    out, t = _extended(*values, sizes, out_dtype, precision, interpret_on_cpu)
    perturbed = jnp.zeros(tuple(int(primal.perturbed) for primal in primals), jnp.int8)
    return out, (*values, sizes, t, perturbed)


def _backward_core(lhs, kernel, a, b, scale, group_sizes, t, perturbed, cotangent, *, precision,
                   interpret_on_cpu):
    wants_lhs, wants_kernel, wants_a, wants_b, wants_scale = map(bool, perturbed.shape)
    work = jnp.promote_types(cotangent.dtype, lhs.dtype)
    dy, wide_dy = cotangent.astype(lhs.dtype), cotangent.astype(work)
    wide = at_least_fp32(lhs.dtype)
    if group_sizes is None:
        through = jnp.dot(wide_dy, b.astype(lhs.dtype).astype(work).T, precision=precision,
                          preferred_element_type=wide)
    else:
        through = xla_products.xla_ragged_dot(
            wide_dy, jnp.swapaxes(b.astype(lhs.dtype).astype(work), 1, 2), group_sizes,
            precision=precision, preferred_element_type=wide)
    scaled = scale.astype(wide) * through
    d_lhs = d_kernel = d_a = d_b = d_scale = None
    if wants_lhs:
        d_lhs = joined_depths(((dy, kernel), (scaled.astype(lhs.dtype), a)), group_sizes, transpose_rhs=True,
                              out_dtype=lhs.dtype, precision=precision, interpret_on_cpu=interpret_on_cpu)
    if wants_kernel:
        grad_k = gemm_t(lhs.astype(work), wide_dy, group_sizes, out_dtype=kernel.dtype, precision=precision,
                        interpret_on_cpu=interpret_on_cpu)
        d_kernel = grad_k[0] if group_sizes is None else grad_k
    if wants_a:
        if group_sizes is None:
            d_a = jnp.dot(lhs.T.astype(wide), scaled, precision=precision,
                          preferred_element_type=wide).astype(a.dtype)
        else:
            d_a = gemm_t(lhs.astype(wide), scaled, group_sizes, out_dtype=a.dtype, precision=precision,
                         interpret_on_cpu=interpret_on_cpu)
    if wants_b:
        if group_sizes is None:
            d_b = jnp.dot(t.T, wide_dy.astype(wide), precision=precision,
                          preferred_element_type=wide).astype(b.dtype)
        else:
            d_b = gemm_t(t.astype(wide), wide_dy.astype(wide), group_sizes, out_dtype=b.dtype,
                         precision=precision, interpret_on_cpu=interpret_on_cpu)
    if wants_scale:
        if group_sizes is None:
            unscaled = jnp.dot(lhs, a.astype(lhs.dtype), precision=precision, preferred_element_type=wide)
        else:
            unscaled = xla_products.xla_ragged_dot(lhs, a.astype(lhs.dtype), group_sizes, precision=precision,
                                                   preferred_element_type=wide)
        d_scale = jnp.sum(through * unscaled).astype(scale.dtype)
    d_lhs, (d_kernel, d_a, d_b, d_scale) = with_gradients_formed(d_lhs, (d_kernel, d_a, d_b, d_scale))
    return d_lhs, d_kernel, d_a, d_b, d_scale


def _dense_backward(out_dtype, precision, interpret_on_cpu, residuals, cotangent):
    del out_dtype
    lhs, kernel, a, b, scale, t, perturbed = residuals
    return _backward_core(lhs, kernel, a, b, scale, None, t, perturbed, cotangent, precision=precision,
                          interpret_on_cpu=interpret_on_cpu)


def _grouped_backward(out_dtype, precision, interpret_on_cpu, residuals, cotangent):
    del out_dtype
    lhs, kernel, a, b, scale, group_sizes, t, perturbed = residuals
    grads = _backward_core(lhs, kernel, a, b, scale, group_sizes, t, perturbed, cotangent,
                           precision=precision, interpret_on_cpu=interpret_on_cpu)
    return (*grads, zero_from_primal(group_sizes))


_low_rank_dense = contracted("test_an_adapted_product_holds_every_reverse_order")(
    declared_derivative(KernelGlue(), _dense_value, _dense_forward, _dense_backward,
                        nondiff_argnums=(5, 6, 7), symbolic_zeros=True))

_low_rank_grouped = contracted("test_an_adapted_product_holds_every_reverse_order")(
    declared_derivative(KernelGlue(), _grouped_value, _grouped_forward, _grouped_backward,
                        nondiff_argnums=(6, 7, 8), symbolic_zeros=True))
