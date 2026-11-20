"""LoRA's adapted product as one GEMM template product over the depth the rank extends (`low_rank`, Appendix C
row 26, ledger 1304), and its selector (`LOW_RANK`)."""


import jax
import jax.numpy as jnp
from flax.typing import PrecisionLike

from dew.nn.derivatives import KernelGlue, contracted, declared_derivative, with_gradients_formed
from dew.nn.kernels import generation, selection
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


def low_rank(lhs: jax.Array, kernel: jax.Array, a: jax.Array, b: jax.Array, scale, *, out_dtype,
             precision: PrecisionLike = None, interpret_on_cpu: bool = False) -> jax.Array:
    """`lhs · kernel + scale · (lhs · a) · b`, LoRA's adapted product (Appendix C row 26, strategy (b)), as
    one template product over the depth the rank extends: `[lhs | t] · [kernel ; b]`, where `t = scale ·
    (lhs · a)` is summed in fp32 and rounded once to `lhs`'s dtype, so the update accumulates into the base
    product's accumulator and no output-sized pass adds it. `lhs` is `[rows, depth]`, `kernel` `[depth,
    width]`, `a` `[depth, rank]` and `b` `[rank, width]`; `scale` a scalar.

    Its reverse mode (first order): the input gradient is one product too, `[dy | u] · [kernel | a]ᵀ` with
    `u = scale · (dy · bᵀ)` rounded once; the factors' gradients are `lhsᵀ · (scale · (dy · bᵀ))` and `tᵀ ·
    dy`, the rank-wide kernel gradients; each operand's only where it is differentiated (a frozen kernel's,
    `lhsᵀ · dy`, never). Each joined product is `gemm.joined_depths`': one `gemm` over the operands joined
    along the depth, a materialized strategy (b). A projection the run's low-precision recipe takes (its
    role and its widths, `refuse_recipe`) is refused: the adapted product does not take the recipe."""
    refuse_recipe("an adapter's fused product (`gemm.low_rank`)", *kernel.shape)
    return _low_rank(lhs, kernel, a, b, jnp.asarray(scale, jnp.float32), out_dtype, precision,
                     interpret_on_cpu)


def _rank_product(lhs: jax.Array, factor: jax.Array, scale: jax.Array, precision: PrecisionLike) -> jax.Array:
    """`scale · (lhs · factor)`, summed and kept in at least fp32: the rank-wide operand an adapted product
    appends to its depth, rounded once to `lhs`'s dtype where it is appended (`_appended`)."""
    wide = at_least_fp32(lhs.dtype)
    summed = jnp.dot(lhs, factor.astype(lhs.dtype), precision=precision, preferred_element_type=wide)
    return scale.astype(wide) * summed


def _extended(lhs, kernel, a, b, scale, out_dtype, precision, interpret_on_cpu):
    """`low_rank`'s value and its appended operand `t`, in at least fp32: `[lhs | t] · [kernel ; b]`, `t`
    rounded once to `lhs`'s dtype where it joins the depth (`gemm.joined_depths`)."""
    t = _rank_product(lhs, a, scale, precision)
    out = joined_depths(((lhs, kernel), (t.astype(lhs.dtype), b)), out_dtype=out_dtype, precision=precision,
                        interpret_on_cpu=interpret_on_cpu)
    return out, t


def _low_rank_value(lhs, kernel, a, b, scale, out_dtype, precision, interpret_on_cpu):
    return _extended(lhs, kernel, a, b, scale, out_dtype, precision, interpret_on_cpu)[0]


def _low_rank_forward(lhs, kernel, a, b, scale, out_dtype, precision, interpret_on_cpu):
    primals = (lhs, kernel, a, b, scale)
    values = (lhs.value, kernel.value, a.value, b.value, scale.value)
    out, t = _extended(*values, out_dtype, precision, interpret_on_cpu)
    # Which operands are differentiated, as the residuals' static shapes: a frozen kernel's gradient is
    # never computed.
    perturbed = jnp.zeros(tuple(int(primal.perturbed) for primal in primals), jnp.int8)
    return out, (*values, t, perturbed)


def _low_rank_backward(out_dtype, precision, interpret_on_cpu, residuals, cotangent):
    del out_dtype
    lhs, kernel, a, b, scale, t, perturbed = residuals
    wants_lhs, wants_kernel, wants_a, wants_b, wants_scale = map(bool, perturbed.shape)
    # The input gradient's product takes the cotangent in the rows' dtype, as the forward's operands; the
    # wide gradients (the kernel's, the factors' and the scale's) take it at its own precision where it is
    # wider (an fp32 output's), the rows widened to it, as the plain product's kernel gradient
    # (`_kernel_gradient`'s `work`).
    work = jnp.promote_types(cotangent.dtype, lhs.dtype)
    dy, wide_dy = cotangent.astype(lhs.dtype), cotangent.astype(work)
    wide = at_least_fp32(lhs.dtype)
    # The cotangent of `t` before its scale, dy · bᵀ (b as the forward multiplied it, in the rows' dtype),
    # summed wide, and after it: the scale taken before any product with it, as the adapter's own program
    # takes it (a product near float32's largest that the scale brings back in range never overflows first).
    through = jnp.dot(wide_dy, b.astype(lhs.dtype).astype(work).T, precision=precision,
                      preferred_element_type=wide)
    scaled = scale.astype(wide) * through
    d_lhs = d_kernel = d_a = d_b = d_scale = None
    if wants_lhs:
        d_lhs = joined_depths(((dy, kernel), (scaled.astype(lhs.dtype), a)), transpose_rhs=True,
                              out_dtype=lhs.dtype, precision=precision, interpret_on_cpu=interpret_on_cpu)
    if wants_kernel:
        d_kernel = gemm_t(lhs.astype(work), wide_dy, None, out_dtype=kernel.dtype, precision=precision,
                          interpret_on_cpu=interpret_on_cpu)[0]
    # The factors' gradients from the wide `scaled` and `t` (the rank-wide products' sums, unrounded), as
    # the adapter's own program differentiates its fp32 low-rank products.
    if wants_a:
        d_a = jnp.dot(lhs.T.astype(wide), scaled, precision=precision,
                      preferred_element_type=wide).astype(a.dtype)
    if wants_b:
        # `t` holds the scale already: tᵀ · dy is the factor's gradient whole.
        d_b = jnp.dot(t.T, wide_dy.astype(wide), precision=precision,
                      preferred_element_type=wide).astype(b.dtype)
    if wants_scale:
        unscaled = jnp.dot(lhs, a.astype(lhs.dtype), precision=precision, preferred_element_type=wide)
        d_scale = jnp.sum(through * unscaled).astype(scale.dtype)
    d_lhs, (d_kernel, d_a, d_b, d_scale) = with_gradients_formed(d_lhs, (d_kernel, d_a, d_b, d_scale))
    return d_lhs, d_kernel, d_a, d_b, d_scale

# The launches' derivative glue (`derivatives.KernelGlue`): an operand no reverse pass reaches is a symbolic
# zero, whose gradient is never formed.
_low_rank = contracted("test_an_adapted_product_holds_every_reverse_order")(
    declared_derivative(KernelGlue(), _low_rank_value, _low_rank_forward, _low_rank_backward,
                        nondiff_argnums=(5, 6, 7), symbolic_zeros=True))
