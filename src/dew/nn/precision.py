"""The precision a bf16 model's fp32 boundaries multiply at.

A model's last matmul before a loss (the patch output head, the vocabulary
head) keeps an fp32 result, since the loss is fp32; `head_product` says why
the vocabulary head still rounds to bf16 values at the default precision.
`preferred_element_type=jnp.float32` alone makes that dot's cotangent fp32,
so the mixed f32 x bf16 backward products run at the fp32 rate. XLA's
`BF16_BF16_F32` dot algorithm names the arithmetic instead (bf16 operands,
fp32 accumulation, whatever the storage), and jax carries `precision` into
the transpose rules, so the backward follows; jax's own xla attention sets it
for this reason (jax-ml/jax#24047). Dew's products at that algorithm type
their operands bf16 instead (`bf16_dot_general`, which the dot family runs
for it): an A100 computes the algorithm wrong where a dot has a vector side
or sits below XLA's GEMM rewrite threshold.

On a CPU, a float32 contraction's sums are XLA's own, one sequential FMA chain an output, on every mesh,
which GSPMD partitions natively (row 5448: CPU is not where Dew competes, and a blocked sum of Dew's own
reached a product whole on one device alone). A GPU's and a TPU's products are their backend's. Dew's test
processes run wide CPU arithmetic instead (`WIDE_CPU_ARITHMETIC`, which tests/lane_environment.py sets): a
CPU sums those products in float64 and rounds once, outside a scope stating `FP32_SUMS`, which sums them
as every process's CPU does.
Convolutions, depthwise ones included, keep XLA's fp32 sum on a CPU: no convolution's check 1
failed there, and XLA:CPU's float64 convolution cost a 3x3 one's training step 44 times (15.5 to
675 ms on a 4x32x32x128 map), its best rewrite 2 to 4. Should a convolution's check 1 resolve above
its bar, the recorded fix is that rewrite: the kernel's shifted windows taken inside the float64
scope and contracted with the taps as one 2D product (stride 1 train 3.9 times, stride 2 1.6)."""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import functools
import math
import os
from collections.abc import Callable, Hashable, Sequence
from typing import Literal, Protocol, Self, overload

import jax
import jax.numpy as jnp
import numpy as np
from flax.typing import Dtype, PrecisionLike
from jax.extend import core
from jax.interpreters import ad, batching, mlir
from jax.typing import DTypeLike
from numpy.lib.array_utils import normalize_axis_tuple

from dew.nn.derivatives import (
    Protected,
    bound_span,
    contracted,
    declared_derivative,
    declared_tangent,
    surrogate,
    with_gradients_formed_at,
    with_tangents_formed,
)
from dew.nn.float_bits import (
    exact_power2,
    lowered_rounded,
    raw_fields,
    split_bits,
    two_factor_reach,
    widened_exactly,
)
from dew.nn.nested import nested
from dew.nn.programs import UNROUNDED, equations, own_preset
from dew.nn.sharding import explicit_spec

__all__ = [
    "ACCURATE",
    "DTYPE_PRESETS",
    "FP8",
    "FP32_SUMS",
    "L2NORM_EPS",
    "MATMUL_PRECISIONS",
    "MXFP8",
    "NUMERICS_PRESETS",
    "NVFP4",
    "PROJECTION_ROLES",
    "ROUNDING_STREAM",
    "WIDE_CPU_ARITHMETIC",
    "Dtypes",
    "LowPrecision",
    "Numerics",
    "RecipeRefused",
    "accumulated",
    "activation_and_slope",
    "added_to",
    "asks_default_precision",
    "at_least_fp32",
    "at_precision",
    "bf16_dot_general",
    "bf16_einsum",
    "bf16_logits",
    "bf16_product",
    "biased",
    "biased_value",
    "capped_logits",
    "compensated_add",
    "compensated_difference",
    "compensated_sum",
    "compensated_total",
    "compensates_loops",
    "configurable_precision",
    "configured_precision",
    "correctly_rounded_divide",
    "divided",
    "effective_precision",
    "exact_float32",
    "exact_products",
    "float32_dot_general",
    "float32_einsum",
    "float32_matmul",
    "float64_sum",
    "fp32_result_dot_general",
    "head_dot_general",
    "head_product",
    "iteration_dtype",
    "l2_squares",
    "l2_unit",
    "l2norm",
    "numerics_of",
    "operand_precision",
    "paired_sum",
    "precision_names",
    "products_dtype",
    "projection_role",
    "recipe_role",
    "refuse_recipe",
    "rms_statistics",
    "rms_weighted",
    "rounded_operand",
    "rounded_to",
    "rounds_products",
    "rounds_to_bf16",
    "running_sum",
    "running_total",
    "scaled",
    "scaled_logits",
    "span_means",
    "spread",
    "sqrt_or_zero",
    "stochastic_bf16",
    "summed_once",
    "summed_pairs",
    "summed_total",
    "twice_scaled",
    "two_product",
    "two_sum",
    "uncontracted",
    "weighted_total",
    "wide_cpu_arithmetic",
    "widened_reduction",
]


def stochastic_bf16(x, noise):
    """fp32 `x` rounded to bf16 up in magnitude with the probability of its dropped low 16 bits: `noise`'s
    low 16 bits (uint32, uniform) added below the kept bits and the sum truncated, which carries into them
    exactly with that probability. It does not saturate: a value past bf16's largest finite rounds to
    infinity with its dropped bits' probability, as rounding to nearest would. NaN stays NaN. The noise is
    its caller's policy: a hash of the state's own bits and the token (`recurrence.round_to_bf16`), of the
    step's seed and the element's index (`dew._moments.stochastic_round_bf16`)."""
    bits = jax.lax.bitcast_convert_type(x, jnp.uint32)
    rounded = jax.lax.bitcast_convert_type((bits + (noise & jnp.uint32(0xFFFF))) & jnp.uint32(0xFFFF0000),
                                           jnp.float32)
    return jnp.where(jnp.isnan(x), x, rounded).astype(jnp.bfloat16)


def at_least_fp32(dtype: DTypeLike | None) -> jnp.dtype:
    """The dtype arithmetic Dew keeps from rounding below float32 runs in:
    float32 for float32, every narrower float and a module's unset dtype
    (None), and `dtype` itself where it is wider. Norm statistics, softmaxes,
    rotary and time-embedding angles, the vocabulary head's product and loss
    reductions read it, so a bfloat16 model reduces in float32 and a float64
    model computes in float64 throughout, not with its float32 twin's
    roundings. An 8-bit float, which promotes implicitly to nothing, reads
    float32 as every narrower float does."""
    if dtype is None or jnp.dtype(dtype).itemsize < 4:
        return jnp.dtype(jnp.float32)
    return jnp.promote_types(dtype, jnp.float32)


def rms_statistics(x: jax.Array, epsilon: float, *, axis=-1, keepdims: bool = True,
                   count: int | None = None) -> tuple[jax.Array, jax.Array]:
    """Mean square and inverse RMS in at least fp32. A padded kernel tile supplies its logical
    `count` after zeroing padding; an ordinary array uses its reduction dimensions. Both means multiply
    by the reciprocal count, as XLA's mean does. Callers own when the inverse, normalized values and
    learned scale round. The count's reciprocal is a Python float, where `jnp.reciprocal` of the count's
    array is an integer power, which a Pallas Triton kernel cannot lower on a folded constant."""
    square = jnp.square(x.astype(at_least_fp32(x.dtype)))
    mean = (jnp.mean(square, axis=axis, keepdims=keepdims) if count is None
            else jnp.sum(square, axis=axis, keepdims=keepdims) * (1.0 / count))
    return mean, jax.lax.rsqrt(mean + epsilon)


def activation_and_slope(gate, activation: str) -> tuple[jax.Array, jax.Array]:
    """An fp32 gate's `act(gate)` and its derivative, for 'silu' or 'sigmoid', its sigmoid and its slope the
    one logistic's arithmetic (`logistic_and_slope`), as a kernel's body takes them; a gate differentiated by
    JAX takes Dew's one logistic itself (`activations.gate_activation`), whose tangent is declared.

    silu's derivative `s + g s s(-g)`, with `s = sigmoid(g)`, cancels to
    zero near g = -1.28; formed as the product rule forms it, `s + g (s s(-g))`,
    its worst element there is half `(g s) s(-g) + s`'s, as XLA's is."""
    sigmoid, slope = logistic_and_slope(gate)
    if activation == 'sigmoid':
        return sigmoid, slope
    return gate * sigmoid, sigmoid + gate * slope


def rms_weighted(x: jax.Array, inverse: jax.Array, *, weight: jax.Array | None = None,
                 gate: jax.Array | None = None,
                 rounded: Callable[[jax.Array], jax.Array] | None = None, weight_first: bool = False
                 ) -> jax.Array:
    """An RMS norm's arithmetic past its statistics: `x` times its rows' `inverse` RMS (`rms_statistics`'s, or
    a kernel's own), times `weight`, times the activated `gate` (Qwen3-Next's gated norm: a kernel's body
    activates it by `activation_and_slope`, a gate JAX differentiates by `activations.GATE_ACTIVATIONS`),
    in that order and in `x`'s dtype, which the caller widens (at least fp32) and rounds once after; with
    `weight_first`, the weight folded into the inverse before the rows (Flax's `nn.RMSNorm`,
    `x * (inverse * weight)`). `rounded` rounds the normalized rows, before any weight
    multiplies them (Llama's and Qwen3's cast first, `RMSNorm.scale_after_cast`; a norm computing in its
    input's dtype), which a weight folded first has none of. The fused norm kernel, the recurrence's gated
    output norm, the portable norm and the portable gated norm compute it alike, each keeping its own
    statistics, memory schedule and derivative."""
    if weight is not None and weight_first:
        if rounded is not None:
            raise ValueError("a weight folded into the inverse leaves no normalized rows to round first")
        y = x * (inverse * weight)
    else:
        y = x * inverse if rounded is None else rounded(x * inverse)
        if weight is not None:
            y = y * weight
    return y if gate is None else y * gate


def weighted_total(x: jax.Array, weights: jax.Array, axis=None) -> tuple[jax.Array, jax.Array]:
    """Weighted total and mass along `axis` (an int, a tuple, or None for every axis), in `x`'s
    dtype, each summed once rounded (`compensated_sum`), so a loss over many tokens rounds as its
    exact sum does. Broadcastable weights mark excluded values with zero; those values are selected
    away before multiplication, so their NaNs and infinities cannot reach the sum. The caller owns
    its denominator policy."""
    weights = jnp.broadcast_to(jnp.asarray(weights, x.dtype), x.shape)
    kept = jnp.where(weights != 0, x, 0)
    axes = tuple(range(x.ndim)) if axis is None else tuple(
        dimension % x.ndim for dimension in ((axis,) if isinstance(axis, int) else axis))
    return compensated_sum(kept * weights, axes), compensated_sum(weights, axes)


def biased(product: jax.Array, bias: jax.Array) -> jax.Array:
    """`product` plus `bias` broadcast over its leading axes, in the dtype their addition promotes
    to, as flax's layer adds them.
    Rounding a 16-bit product and a bias converted inside the step once together, as torch's addmm
    does, wrote an fp32 product and a pass of its own on every biased layer (a DiT-B step 17.5% slower
    on an A100), for a bf16 output 1.37 times torch's rounding, which the parity rule holds already:
    XLA fuses an fp32 product's epilogue into its GEMM only where the bias is a parameter (row 634), so
    a layer whose bias is stored at fp32 adds it to its fp32 product and rounds once
    (`dew.nn.dense._rounds_once`), and a 16-bit bias adds here as flax adds it.

    JAX transposes a broadcast into a sum in the broadcast operand's dtype (jax-ml/jax#30310), so
    a bfloat16 bias added to bfloat16 products sums its gradient over every position in bfloat16,
    where torch's addmm sums its in float32: Qwen 2's projection biases, summed over 8192 tokens,
    came out up to 3.75 times further from float64 than torch's own bfloat16 gradients. Broadcast
    at fp32, the gradient sums at fp32 and rounds once into the bias. A CPU running Dew's wide
    arithmetic (`WIDE_CPU_ARITHMETIC`) sums it in float64, as it sums a float32 product (`spread`).

    Its reverse mode returns the product's cotangent only with the bias's gradient formed (the rule's
    dual, `with_tangents_formed`): XLA's sum of a bias's gradient, a reduction over every row of the
    cotangent, was put off to a step's end, each layer's cotangent held to it (a biased attention decoder's
    temporaries grew 27 MB a layer, tests/test_declared_cells_memory.py)."""
    return _biased(product, bias)


def biased_value(product: jax.Array, bias: jax.Array) -> jax.Array:
    """`biased`'s value: `product` plus `bias` spread at fp32 (`broadcast_summed`) over the product's
    trailing axes, as flax's layers reshape it, so no rank is promoted implicitly; rounded back, it adds as
    flax adds it (XLA folds the round trip, so the product and bias fuse as before). Differentiated as it
    is, its bias's gradient sums at fp32 (`broadcast_summed`'s): a projection's rule over its product and
    bias together reads it so (`dew.nn.kernels.projections.dot_general`'s `bias`)."""
    dtype = jnp.result_type(product, bias)
    leading = product.ndim - bias.ndim
    spread_bias = broadcast_summed(bias, product.shape, tuple(range(leading, product.ndim)), dtype=dtype)
    return product.astype(dtype) + spread_bias


def _biased_tangent(primals, tangents):
    """`biased`'s tangent: the product's and the bias's directions added as the value adds them, both
    through the rule's dual where both move, so reverse mode, its transpose, sums the bias's gradient at fp32
    (`broadcast_summed`) and returns the product's cotangent only with it."""
    from jax.custom_derivatives import SymbolicZero

    from dew._cache import unintercepted

    product, bias = primals
    d_product, d_bias = tangents
    d_product, d_bias = with_tangents_formed(d_product, d_bias)
    value = biased_value(product, bias)
    dtype = value.dtype
    leading = product.ndim - bias.ndim
    terms = ([] if isinstance(d_product, SymbolicZero) else [d_product.astype(dtype)]) + (
        [] if isinstance(d_bias, SymbolicZero)
        # The spread is Dew's own arithmetic, no model's product for a quantization rule to take: its
        # transpose, reverse mode's, has no custom VJP to meet (`unintercepted`).
        else [unintercepted(broadcast_summed)(d_bias, product.shape, tuple(range(leading, product.ndim)),
                                              dtype=dtype)])
    return value, sum(terms[1:], terms[0]) if terms else jnp.zeros_like(value)


_biased = contracted("test_a_biased_sum_holds_every_order")(
    declared_tangent(Protected(), biased_value, _biased_tangent, symbolic_zeros=True))


def broadcast_summed(value: jax.Array, shape: tuple[int, ...], dimensions: tuple[int, ...], *,
                     dtype: DTypeLike | None = None) -> jax.Array:
    """`jax.lax.broadcast_in_dim(value, shape, dimensions)` in `dtype` (the value's where None), whose
    gradient, the sum of a cotangent over every position the value spread over, sums at fp32 or wider
    and rounds once to the value's dtype, as torch's autograd sums a bf16 one: the value widened to fp32
    between exact casts and spread (`spread`), which a CPU running Dew's wide arithmetic sums in
    float64, then cast to `dtype`, so a cotangent wider than the value sums before it rounds. Every
    order JAX forms its transpose at takes it, forward mode's included. A bias's spread (`biased`, in
    the dtype its addition promotes to) and a tier-3 model's broadcasts (`dew.interop.torchax_fallback`)
    are each this (ledger 3038)."""
    wide = at_least_fp32(value.dtype)
    kept = iter(value.shape)
    placed = tuple(next(kept) if axis in dimensions else 1 for axis in range(len(shape)))
    spread_value = spread(jnp.reshape(value.astype(wide), placed), shape)
    return spread_value.astype(value.dtype if dtype is None else dtype)


def spread(value: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    """`value` broadcast to `shape`, whose transpose, the sum of a cotangent over every position the
    value spread over (a bias's gradient), every CPU process takes as products' sums: a float32 value's
    broadcast is outer products of ones and the value (`_blocked_spread`), whose transpose sums the
    positions in blocks, each block's and then the blocks', an error of about `N / B + B` float32
    roundings where XLA:CPU's reduction of the broadcast's transpose sums the N positions in order, and
    a program summing whole and one summing by shard each a product's sum, not a reduction XLA's CPU
    fusions sum one way in one program and another in the other (row 4520); a user's process sums it
    so, as check 1 measures it (ledger 4882: the blocked sum every CPU process's, where it
    was the suite's alone). A CPU running Dew's wide arithmetic (`WIDE_CPU_ARITHMETIC`, its tests) in no
    scope that states fp32 sums (`FP32_SUMS`) takes the one outer product summed in float64 and rounded
    once (`accumulated`). An accelerator, and a quantized model's trace, take the broadcast, whose
    transpose JAX sums in the value's dtype. A value an Explicit axis places along an axis it spreads
    over (never a bias, whose leading axes `biased` adds) is broadcast: the outer product does not keep
    that placement on a dropped size-one axis."""
    axes = tuple(axis for axis, (size, total) in enumerate(zip(value.shape, shape, strict=True))
                 if size == 1 and total != 1)
    placed = explicit_spec(value)
    if (not axes or (placed is not None and any(placed.spec[axis] is not None for axis in axes))
            or not _float32_products(value)):
        return jnp.broadcast_to(value, shape)
    kept = tuple(axis for axis in range(len(shape)) if axis not in axes)

    # Products by ones at float32's own precision, whatever dot algorithm the process states: a narrower
    # one would round the spread values themselves, where a broadcast moves them exactly.
    outer_product = functools.partial(jax.lax.dot_general, dimension_numbers=(((), ()), ((), ())),
                                      precision=jax.lax.Precision.HIGHEST)
    sizes = tuple(shape[axis] for axis in axes)

    def outer(value: jax.Array) -> jax.Array:
        # Only the size-one axes are dropped, so a sharded kept axis keeps its own sharding.
        squeezed = jnp.squeeze(value, axes)
        if _sums_in_float64(value):
            product = accumulated(outer_product, jnp.ones(sizes, value.dtype), squeezed)
        else:
            product = _blocked_spread(squeezed, sizes, outer_product)
        return jnp.transpose(product, tuple(np.argsort((*axes, *kept))))

    return jax.lax.platform_dependent(value, cpu=outer, default=lambda value: jnp.broadcast_to(value, shape))


def _spread_block(count: int) -> int:
    """The positions a blocked spread sums together first (`_blocked_spread`): the power of two nearest the
    square root of `count`, a function of the count alone, so every program and every site sums one count of
    positions in one order; one where there are none."""
    return 2 ** max(0, round(math.log2(math.sqrt(max(count, 1)))))


def _blocked_spread(value: jax.Array, sizes: tuple[int, ...], outer_product) -> jax.Array:
    """`value` spread over `sizes` as two outer products of ones, whose transpose sums a cotangent's `N`
    positions in blocks of `_spread_block(N)` consecutive ones, each block's positions first and then the
    blocks' sums, each an fp32 product sum (`accumulated`): an error of about `(N / B + B)`
    units of float32's roundoff, where one sum of every position, which XLA:CPU accumulates in order, loses
    about `N` of them (a padded flow's bias, row 4520's review) and a reduction XLA's fusions block their own
    way sums the whole and a shard's otherwise (row 4520). Consecutive positions sum together first, as an
    ordered sum does: terms that cancel where they lie together (a question's options' shares of a bias
    under its softmax) cancel before any sum reaches another row's (train 700's test_decision_packing). The
    positions past `N` the blocks pad are the zeros the transpose of the slice that drops them adds to the
    cotangent."""
    count = math.prod(sizes)
    block = _spread_block(count)
    blocks = -(-count // block)
    across = accumulated(outer_product, jnp.ones((blocks,), value.dtype), value)
    within = accumulated(outer_product, jnp.ones((block,), value.dtype), across)
    # Laid out block by block, so the transpose sums a block's consecutive positions (`within`'s) first.
    flat = jnp.swapaxes(within, 0, 1).reshape((blocks * block, *value.shape))[:count]
    return flat.reshape((*sizes, *value.shape))


def precision_names(precision: jax.lax.PrecisionLike) -> frozenset[str]:
    """The canonical names a `jax.lax.PrecisionLike` spells, unordered.

    JAX's alias is None, a string, a `jax.lax.Precision`, a dot algorithm
    preset or a dot algorithm, or a pair of precisions; flax's is the same
    without the algorithms. A string is read as JAX reads
    `jax_default_matmul_precision`: a dot algorithm preset's name is that
    preset ('F32_F32_F32'), any other goes through `jax.lax.Precision`, so
    'tensorfloat32' is 'HIGH' and 'bfloat16' is 'DEFAULT'; a preset keeps its
    own name, and a dot algorithm, which has none, is named by its fields.
    """
    if precision is None:
        return frozenset()
    written = (precision,) if isinstance(
        precision, str | jax.lax.Precision | jax.lax.DotAlgorithmPreset | jax.lax.DotAlgorithm) else precision
    return frozenset(_name(one) for one in written if one is not None)


FP32_PRODUCTS = frozenset({"HIGHEST", "F32_F32_F32"})
"""The precision names that declare fp32 operands' products in IEEE fp32 (`precision_names`): `highest` and
the F32_F32_F32 preset, which `jax_default_matmul_precision` spells 'highest', 'float32' or the preset's
name. On a GPU a program holding such a dot compiles its GEMMs on cuBLAS
(`dew.nn.kernels.templates.report.targeted`, reading the program's own dots, an operation's precision as
a run's), whose sum is the one a float32 reference is computed with: XLA's Triton GEMM sums farther from
float64 (rows 5026/5028/5029). That is the price of 'highest' meeting its reference, a wrong result
weighed against a slower one: a decode-sized float32 forward is slower on cuBLAS, a training step and a long
forward are not (M-IG-13's A100 and GH200 measures)."""


def declares_fp32_products(precision: jax.lax.PrecisionLike) -> bool:
    """Whether `precision` (a run's `jax_default_matmul_precision`, `Numerics.matmul`) declares fp32
    operands' products in IEEE fp32 (`FP32_PRODUCTS`), however it is spelled."""
    return bool(precision_names(precision) & FP32_PRODUCTS)


def _name(precision: str | jax.lax.Precision | jax.lax.DotAlgorithmPreset | jax.lax.DotAlgorithm) -> str:
    """The name of one precision `precision_names` takes, a string read as JAX reads one."""
    if isinstance(precision, jax.lax.DotAlgorithm):
        return repr(precision)
    if not isinstance(precision, str):
        return precision.name
    if precision in jax.lax.DotAlgorithmPreset.__members__:
        return jax.lax.DotAlgorithmPreset[precision].name
    return jax.lax.Precision(precision).name


def asks_default_precision(precision: jax.lax.PrecisionLike, *, configured: bool = False) -> bool:
    """Whether `precision` asks for no more than the default. With
    `configured`, an unset precision reads `jax_default_matmul_precision`,
    the default a product without one gets."""
    return precision_names(effective_precision(precision) if configured else precision) <= {'DEFAULT'}


def effective_precision(precision: jax.lax.PrecisionLike) -> jax.lax.PrecisionLike:
    """The precision a product multiplies at, resolved once: the one it states, else the run's configured
    one (`jax_default_matmul_precision`, a run's `Numerics.matmul`, `generation.run_precision`), which a
    route takes as the call's declaration, honouring it or handing the call to a route that does, never
    replacing it with a backend's own arithmetic."""
    return precision if precision_names(precision) else jax.config.jax_default_matmul_precision


def rounds_products() -> bool:
    """Whether the run's declaration lets a native route round an fp32 product's operands (TF32, or bf16
    parts): its fast forms (`Numerics.forms`) at a configured precision of no more than the default
    (`asks_default_precision`); a run at 'highest' or a dot algorithm keeps them whole whatever its forms."""
    from dew.nn.kernels.generation import numerics

    return numerics().forms == "fast" and asks_default_precision(None, configured=True)


_LN2_HI, _LN2_LO = 0.693145751953125, 1.428606765330187e-06
"""ln 2 in two parts (Cody-Waite), the first's 16 bits times an exponent's at most 8 exact."""
_EXP_TERMS = (1.0 / 5040.0, 1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0)
"""`e^r`'s Taylor sum to `r^7` by Horner, its highest term first: at `|r| <= ln 2 / 2` the next is 5e-9."""


def _exp_value(base2: bool, x) -> jax.Array:
    """`e^x` (`2^x` where `base2`) of an fp32 `x`: `x = n ln 2 + r`, `n` integral and `|r| <= ln 2 / 2` (a
    base-2 `x` split exactly), `e^r` by its Taylor sum (`_EXP_TERMS`), times `2^n` as two powers of two each
    set in the exponent bits, so a result past fp32's largest is infinite and one below its least normal
    rounds into the subnormals; `x` held first where its power passes both (its tangent declared apart)."""
    if base2:
        x = jnp.clip(x, -150.0, 129.0)
        n = jnp.round(x)
        r = (x - n) * math.log(2.0)
    else:
        x = jnp.clip(x, -104.0, 90.0)
        n = jnp.round(x * (1.0 / math.log(2.0)))
        r = (x - n * _LN2_HI) - n * _LN2_LO
    summed = functools.reduce(lambda p, term: p * r + term, _EXP_TERMS[1:], jnp.full_like(r, _EXP_TERMS[0]))
    return power2(summed, n, reach=150)


def power2(x: jax.typing.ArrayLike, k: jax.typing.ArrayLike, reach: float | None = None) -> jax.Array:
    """`x * 2^k` for an integral `k` of any size, exactly wherever `x` and the result are normal values of
    its format (`x`'s, at least float32's, `at_least_fp32`), by exact powers of two of `k`'s sign
    (`exact_power2`), `k` held within `reach` (by default the format's saturating bound, 279 in float32):
    the one owner of scaling by a power of two (`scalar_power2` a Mosaic kernel's scalar `k`). A backend that
    flushes subnormals flushes a subnormal `x` first (`raised_exactly` raises one from its bits).

    Past `two_factor_reach`, `x` takes `2^o 2^(k - 2o) 2^o`, its own reverse: `o` zero where `k` is a normal
    exponent, else `k`'s half held within the normal exponents, at least half of `-minexp` (float32's 63),
    past `nmant`. So on a backend that keeps subnormals a result below the least normal rounds once, at the
    last factor (the product before it is normal wherever the result rounds to no zero), and so does its
    pullback, which applies the same factors in reverse before it sums a broadcast operand's cotangent
    (base review 44e282c4f F1).
    Within it, a caller stating its `reach` takes `k`'s two halves, each built as its product takes it, one
    integer tile live, not two (row 2519's register measurements), a subnormal result rounded at each half."""
    # Held at least in float32: an 8-bit float promotes implicitly to nothing.
    dtype = at_least_fp32(jnp.asarray(x).dtype)
    x = jnp.asarray(x, dtype)
    limits = jnp.finfo(dtype)
    least, largest = limits.minexp, limits.maxexp - 1
    held = (largest + 1) - (least - limits.nmant) + 2 if reach is None else min(
        reach, (largest + 1) - (least - limits.nmant) + 2)
    # A Python integer held before it converts: one past float32's range raised in its host conversion.
    if isinstance(k, int):
        k = min(max(k, -held), held)
    # Held as float32, which holds every exponent within the bound exactly: no narrow integer `k` wraps.
    k = jnp.clip(jnp.asarray(k, jnp.float32), -held, held)
    if held <= two_factor_reach(dtype):
        # The half truncated by its conversion, an integer only at each factor's bias: an int32 half or a
        # float `trunc` spilled the A100's recurrence and attention launches (row 2519).
        half = (k * np.float32(0.5)).astype(jnp.int32)
        return (x * exact_power2(half, dtype)
                * exact_power2((k - half.astype(jnp.float32)).astype(jnp.int32), dtype))
    # The middle `k - 2o` within `nmant + 4` of zero wherever `o` is held: `held` is at most three times
    # `-minexp`.
    outer = jnp.where(jnp.abs(k) <= -least, 0,
                      jnp.clip((k * np.float32(0.5)).astype(jnp.int32), least, -least))
    side = exact_power2(outer, dtype)
    return x * side * exact_power2(k.astype(jnp.int32) - 2 * outer, dtype) * side


def scalar_power2(x: jax.Array, k: jax.Array) -> jax.Array:
    """`x` times `2^k` for a scalar integral `k` from -252 to 254, as two scalar normal powers of two
    (`floor(k/2)` and the rest; `exp2` of an integer, a GPU's exactly): `power2`'s form for a Mosaic GPU
    kernel's scalar `k`, which takes no bitcast of a scalar and no tile of exponent bits (et-c7512)."""
    half = jnp.floor(k * 0.5)
    return x * jnp.exp2(half) * jnp.exp2(k - half)


def _exp_jvp(base2: bool, primals, tangents) -> tuple[jax.Array, jax.Array]:
    """`_exp`'s value and tangent, the value times the direction (times ln 2 where `base2`)."""
    (x,), (tangent,) = primals, tangents
    value = _exp(base2, x)
    return value, value * (math.log(2.0) if base2 else 1.0) * tangent


_exp = contracted("test_accurate_exp_holds_every_order")(
    declared_tangent(Protected(), _exp_value, _exp_jvp, nondiff_argnums=(0,)))


def accurate_exp(x, *, base2: bool = False) -> jax.Array:
    """`e^x` (`2^x` where `base2`) to fp32's rounding in every form, not the run's (`Numerics.forms`): the
    recurrences' decays (a scan's, `A = -exp(A_log)`, the softplus's), where an fp32 default path must meet
    its upstream's bound. The backend's own where it is so (a GPU's and a CPU's default, a GH200's that of
    `jax.lax.AccuracyMode.HIGHEST` bit for bit, M-JS-135); on a device whose exp is not
    (`GpuTarget.accurate_exp`, every TPU's) below float64, Dew's own on the vector unit's adds and
    multiplies (`_exp_value`): a v5e's default fp32 exp is 1.6e-6 from float64 (its exp of a
    reduced argument alike), which held Mamba-2's layer cell at 5.5 times its CPU float32, and at HIGHEST
    took the cell's training step 4.5% longer in its scan's decays (11.58 to 12.10 ms, M-JS-138); Dew's own
    is 2.6e-8 there."""
    from dew.nn.kernels.generation import gpu_target

    x = jnp.asarray(x)
    if gpu_target().accurate_exp or x.dtype == jnp.float64:
        return jnp.exp2(x) if base2 else jnp.exp(x)
    return _exp(base2, x.astype(jnp.float32)).astype(x.dtype)


def logistic_value(x) -> jax.Array:
    """`1 / (1 + e^-x)` in at least float32, its exp accurate (`accurate_exp`) and a true division: the one
    logistic's arithmetic, which `activations.logistic` declares its tangent over and a kernel's body, which
    differentiates nothing, calls as it is (`activation_and_slope`, the recurrence step's and conv's). JAX's
    derivative of it is no number where `e^-x` overflows (below x near -88 in fp32): what JAX differentiates
    takes `activations.logistic`."""
    work = jnp.asarray(x).astype(at_least_fp32(jnp.asarray(x).dtype))
    return 1.0 / (1.0 + accurate_exp(-work))


def logistic_and_slope(x) -> tuple[jax.Array, jax.Array]:
    """The one logistic's value `s(x)` and derivative `s(x) s(-x)` (`logistic_value`), as a kernel's body
    differentiates it by hand, `activations.logistic`'s declared tangent alike: `s (1 - s)` reads the rounded
    `s(x)`, one past x near 17 in fp32, its derivative then zero where float64's is 4.1e-8."""
    return logistic_value(x), logistic_value(x) * logistic_value(-jnp.asarray(x))


def softplus_series(x) -> jax.Array:
    """`jax.nn.softplus(x)`'s value, `log(1 + exp(x))`, as its larger part plus `log1p` of the smaller's
    exponential `y` in (0, 1], the `log1p` by its series in `s = y / (2 + y)`, at most a third, `2 (s + s^3 /
    3 + s^5 / 5 + ...)` to its working dtype's rounding (`_series_terms`): a backend's own `log1p` is no part
    of it, a v5e's 1.4e-4 from float64 (its `log` alike), which held a Mamba-2 layer's output at 2.9e-5 from
    float64 where this holds it at 1.4e-6 (row 7062). Its leading term `2 s` formed as `y / (1 + y / 2)`,
    normal wherever `y` is, the series in `s^2` alone: `s` itself, half of `y` near fp32's least normal
    (`x` near -87), is subnormal, which a target flushing them (a TPU's, a GPU's) made zero, and the softplus
    with it, where `y` is the answer; `s^2` underflowing leaves the series its first term, one. Formed in at
    least float32 and rounded once to `x`'s dtype; its derivative Dew's one logistic
    (`activations.logistic`)."""
    given = jnp.asarray(x)
    x = given.astype(at_least_fp32(given.dtype))
    y = accurate_exp(-jnp.abs(x))
    leading = y / (1.0 + 0.5 * y)
    square = 0.25 * (leading * leading)
    series = jnp.zeros_like(y)
    for odd in range(_series_terms(x.dtype), 0, -2):
        series = series * square + 1.0 / odd
    return (jnp.maximum(x, 0) + leading * series).astype(given.dtype)


def _series_terms(dtype) -> int:
    """The last odd power `softplus`'s series keeps in `dtype`: past it the next term relative to the
    first, `s^(n + 1) / (n + 2)` at `s`'s most, a third, is under `dtype`'s rounding (15 terms to s^29 in
    float64, 7 to s^13 in float32)."""
    rounding, last = float(jnp.finfo(dtype).eps), 1
    while (1 / 3) ** (last + 1) / (last + 2) >= rounding:
        last += 2
    return last


def _softplus_jvp(primals, tangents) -> tuple[jax.Array, jax.Array]:
    """`softplus`'s value and tangent: its derivative Dew's one logistic (`activations.logistic`), in at
    least float32 and rounded once to its input's dtype."""
    from dew.nn.activations import logistic

    (x,), (tangent,) = primals, tangents
    work = jnp.asarray(x).astype(at_least_fp32(jnp.asarray(x).dtype))
    return softplus(x), (logistic(work) * tangent.astype(work.dtype)).astype(tangent.dtype)


softplus = contracted("test_softplus_holds_every_order")(
    declared_tangent(Protected(), softplus_series, _softplus_jvp))
"""Dew's one softplus (`softplus_series`, its derivative the logistic), which every softplus of `src/dew`
calls (tools/lint_slop.py's SLOP030); a kernel's body, which differentiates nothing, calls
`softplus_series`."""


def iteration_dtype() -> jnp.dtype:
    """The dtype an optimizer's matrix iteration (Muon's and M3's Newton-Schulz, `dew.newton_schulz`)
    multiplies in under the run's numerics: in its fast forms, bfloat16 for any 16-bit compute dtype (the
    measured default under bf16-mixed: a Qwen3.5-4B decision step 844 ms against 1287 at TF32 and 4641 at
    IEEE float32 on an H100) and its compute dtype otherwise; at least float32 of it in its accurate ones;
    a float64 run iterates in float64. Never float16, whose range holds neither the iteration's Frobenius
    norm past 256 nor optax's 1e-8 under it, which rounds to zero there, so a zero update (a zero-initialized
    output projection's upstream matrices at the first step) divides zero by zero."""
    from dew.nn.kernels.generation import numerics

    held = numerics()
    compute = jnp.dtype(held.dtypes.compute)
    if held.forms != "fast":
        return jnp.promote_types(compute, jnp.float32)
    return jnp.dtype(jnp.bfloat16) if compute.itemsize == 2 else compute


def exact_products(precision: jax.lax.PrecisionLike, dtype: DTypeLike) -> bool:
    """Whether a route that multiplies `dtype` operands whole with float32 sums (an fp32 FMA, a 16-bit
    operand's exact product, fp32 operands in bf16 parts that hold their 24 bits) computes the declaration
    `effective_precision(precision)`: a precision level, each a floor (DEFAULT, HIGH, HIGHEST), or a dot
    algorithm whose products are whole for these operands (F32_F32_F32, and a 16-bit operand's own
    BF16_BF16_F32 or F16_F16_F32); not an algorithm that narrows them (TF32's, bf16 parts of an fp32
    product) or widens them (F64), which a route hands to XLA, which computes it."""
    whole = {"DEFAULT", "HIGH", "HIGHEST", "F32_F32_F32"}
    if (own := own_preset(dtype)) is not None:
        whole.add(own.name)
    return precision_names(effective_precision(precision)) <= whole


def products_dtype(*operands) -> jnp.dtype | None:
    """The dtype `operands` multiply in, by jax's implicit promotion of the dtypes they hold: a JAX value's
    own (a float64 formed under x64 keeps it where x64 is off, as JAX keeps it) and a host array's as JAX
    reads an array (`jax.dtypes.canonicalize_dtype`: its float64 is float32 there). Operands of one dtype
    multiply in it, and JAX is asked to promote differing ones alone: asked of float64 dtype objects where
    x64 is off, it reads a request for a dtype it does not grant there and warns, though the operands hold
    it. None where an 8-bit float takes part, which jax promotes with nothing and whose product has no
    precision to resolve."""
    dtypes = [jnp.dtype(operand.dtype) if isinstance(operand, jax.Array)
              else jax.dtypes.canonicalize_dtype(operand.dtype) for operand in operands]
    if any(jnp.issubdtype(dtype, jnp.floating) and jnp.finfo(dtype).bits < 16 for dtype in dtypes):
        return None
    return dtypes[0] if len(set(dtypes)) == 1 else jnp.result_type(*dtypes)


def _intercepted() -> bool:
    """Whether this thread traces a quantized model's method (`dew._cache.QUANTIZED_TRACE`, which jit's
    cache keys on), whose sums are Qwix's products' fp32 ones throughout: Dew's wide CPU test arithmetic
    takes none of them in float64 (`_sums_in_float64`, `compensates_loops`), so the quantized model compares
    bitwise with the unquantized one under `FP32_SUMS`."""
    from dew._cache import QUANTIZED_TRACE

    return QUANTIZED_TRACE.value is not None


def products_quantized() -> bool:
    """Whether a quantized model's rule owns the products the module calling forms: in the trace of a
    quantized model's method (`dew._cache.QUANTIZED_TRACE`, which jit's cache keys on), a Qwix rule matching
    `dot_general` at that module (`qwix.pallas.get_current_rule`, Qwix's query for code that runs its own
    products). A module its rules leave alone forms the plain model's products, summed as the plain model
    sums them."""
    from dew._cache import QUANTIZED_TRACE

    if QUANTIZED_TRACE.value is None:
        return False
    from qwix import pallas as qwix_pallas

    return qwix_pallas.get_current_rule("dot_general") is not None


_OWN_DERIVATIVES = (core.primitives.custom_vjp_call_p, core.primitives.custom_jvp_call_p)
"""JAX's primitives for a function that states its own derivative (`jax.custom_vjp`, `jax.custom_jvp`)."""


def _own_backward(jaxpr) -> bool:
    """Whether a product's trace states a derivative of its own (a `jax.custom_vjp`, as Qwix's
    quantized products are when Qwix quantizes a model directly, or a `jax.custom_jvp`), in it or any
    program inside it (`dew.nn.programs.equations`): a product its library owns, which `accumulated`
    neither widens nor transposes, since a transpose would differentiate its primal instead."""
    return any(eqn.primitive in _OWN_DERIVATIVES for eqn in equations(jaxpr))


FP32_SUMS = "F32_F32_F32"
"""JAX's dot algorithm of fp32 operands summed in fp32: as a process's or a scope's default matmul
precision, the statement that its fp32 products sum as an accelerator sums them, which `accumulated`
honors on a CPU."""


WIDE_CPU_ARITHMETIC = "DEW_WIDE_CPU_ARITHMETIC"
"""The environment variable that, set to 1, has a process's CPU programs take the arithmetic Dew's
tests hold a CPU to: float32 products summed in float64 and rounded once (`accumulated`, `spread`), a
loop's running sum of them compensated so it rounds once too (`compensates_loops`), and a float32
reduction taken in float64 and rounded once (`widened_reduction`). Read as each program is traced;
set by the tests' environment (tests/lane_environment.py), never by the library, which without it
takes none of these forms on any platform. Each such site chooses the wide form by the platform the
program is lowered for (`platform_dependent`, or for a loop's carry the platform its code is chosen
for), so a test that lowers for an accelerator keeps the target's."""


def wide_cpu_arithmetic() -> bool:
    """Whether this process runs wide CPU arithmetic (`WIDE_CPU_ARITHMETIC`)."""
    return os.environ.get(WIDE_CPU_ARITHMETIC) == "1"


def _float32_products(*operands) -> bool:
    """Whether these operands multiply in float32 as Dew's own products: float32 arrays, outside a quantized
    model's trace (`_intercepted`)."""
    return (not _intercepted() and all(isinstance(operand, jax.Array | np.ndarray) for operand in operands)
            and products_dtype(*operands) == jnp.dtype(jnp.float32))


def _wide_cpu_products(*operands) -> bool:
    """Whether these operands' products are the ones Dew's wide CPU arithmetic sums (`accumulated`): in a
    process running it (`WIDE_CPU_ARITHMETIC`), float32 products (`_float32_products`)."""
    return wide_cpu_arithmetic() and _float32_products(*operands)


def _sums_in_float64(*operands) -> bool:
    """Whether a CPU sums these operands' products in float64 (`accumulated`): Dew's wide CPU arithmetic's
    products (`_wide_cpu_products`), in no scope that states fp32 sums (`FP32_SUMS`)."""
    return _wide_cpu_products(*operands) and jax.config.jax_default_matmul_precision != FP32_SUMS


def float64_sum(product: Callable[..., jax.Array], *operands, dtype: DTypeLike, counts: int = 0) -> jax.Array:
    """`product(*operands)`, a sum of products, taken in float64 from its operands widened exactly and
    rounded once to `dtype`, under `jax.enable_x64` for its own trace: the sum `accumulated` takes on a
    CPU under Dew's wide arithmetic (`WIDE_CPU_ARITHMETIC`), with no platform of its own to choose.
    Its products run at HIGHEST precision, whatever fp32 algorithm the caller bound, which would
    narrow float64 operands back to float32. Its last `counts` operands (a ragged dot's group sizes) multiply
    nothing and pass through as they are."""
    with jax.enable_x64(new_val=True):
        wide = product(*_widened(operands, counts), precision=jax.lax.Precision.HIGHEST,
                       preferred_element_type=None)
        return wide.astype(dtype)


def _widened(operands, counts: int = 0) -> list[jax.Array]:
    """`operands` as the float64 operands of their float32 product: each converted to the dtype they
    multiply in (`products_dtype`), as the product converts them, then widened exactly, a subnormal from
    its bits where XLA:CPU's cast flushes it (`widened_exactly`, rows 8966, 9780)."""
    multiplied = operands[:len(operands) - counts]
    dtype = products_dtype(*multiplied)
    widened = [widened_exactly(operand.astype(dtype), jnp.float64) for operand in multiplied]
    return widened + list(operands[len(multiplied):])


def widened_reduction(reduction: Callable[..., Sequence[jax.Array]],
                      *operands: jax.Array) -> tuple[jax.Array, ...]:
    """`reduction(*operands)`, a reduction returning arrays: the plain one, but in a process running wide
    CPU arithmetic (`WIDE_CPU_ARITHMETIC`, Dew's tests) a CPU takes it over the float32 operands widened
    exactly to float64, each float64 result rounded once back to float32, as `accumulated` takes a
    product's sums: XLA's CPU compiler runs a reduction with a combine of its own as one loop over the
    row, rounding at each step. A process without the setting, or operands not float32, take the plain
    reduction on every platform, chosen by the platform a program is lowered for (`platform_dependent`).
    The widened path is a protected span of its own (`_protected`), whatever span its caller is in: a
    rewrite that moved its casts or reordered its sum would undo the widening. The plain reduction is the
    caller's math, rewritten as any other."""
    if not wide_cpu_arithmetic() or not any(operand.dtype == jnp.float32 for operand in operands):
        return tuple(reduction(*operands))

    @_protected
    def wide(*operands):
        with jax.enable_x64(new_val=True):
            # Widened from its bits, where XLA:CPU's cast flushes a subnormal (`widened_exactly`).
            reduced = reduction(*(widened_exactly(operand, jnp.float64) if operand.dtype == jnp.float32
                                  else operand for operand in operands))
            return tuple(term.astype(jnp.float32) if term.dtype == jnp.float64 else term for term in reduced)

    return tuple(jax.lax.platform_dependent(*operands, cpu=wide, default=lambda *operands: tuple(
        reduction(*operands))))


def accumulated(product: Callable[..., jax.Array], *operands,
                preferred_element_type: DTypeLike | None = None) -> jax.Array:
    """`product(*operands, preferred_element_type=...)`, a sum of products of `operands` (a dot, an
    einsum, a matmul): the plain product, XLA's own sum on every platform (row 5448), but in a process
    running wide CPU arithmetic (`WIDE_CPU_ARITHMETIC`, Dew's tests), where they multiply in float32
    (`products_dtype`) a CPU sums the products in float64 and rounds the sum once to the dtype the
    product returns.
    Each fp32 product is exact in float64, so the result is within half an fp32 ulp of the float64
    sum at every shape, where XLA's CPU dot chooses its fp32 accumulation by shape, instruction set
    and release (jax 0.12's sums twice numpy's error from float64 at widths where jax 0.11's summed
    0.6 of it). It costs a CPU product about twice the time (2048-cubed 56 to 126 ms on armada's
    EPYC); a GPU or TPU product, and any whose products are 16-bit, are the product itself. float64
    runs on these ops alone, under `jax.enable_x64` for their own trace, which nothing outside
    reads, and their gradients and tangents are float64 sums too. An operand that is not an array (a
    quantized one, Qwix's `QArray`) leaves the product to its own library, whose dequantized sum
    keeps XLA's accumulation.

    A library that swaps JAX's products for its own while a model runs (Qwix, quantizing) owns them:
    the product is its, as flax's own layers' would be. One whose trace calls a backward of its own
    (a `jax.custom_vjp`, Qwix's when it quantizes a model directly) is left to it whole, its forward
    and its backward, read from the trace that gives the product's dtype (`_own_backward`).

    The wide sum is Dew's test setting, not a user option: a process without `WIDE_CPU_ARITHMETIC`
    traces the plain product. Under it, it is real CPU execution's policy alone, chosen by the platform
    the program is lowered for
    (`platform_dependent`), which keys every compiled program: a step a CPU lowers for a GPU target
    (`dew.training.trainer.Trainer.lower_for`, `lowering_platforms=('cuda',)`) takes the target's
    fp32 sum, and a Pallas kernel's body, interpreted on a CPU to stand in for a GPU, never forms its
    products here. A computation that stands for such a target, a reference for a kernel interpreted
    on a CPU, states its fp32 sums with JAX's dot algorithm for them, `with
    jax.default_matmul_precision(FP32_SUMS)`, which jit's cache keys as it keys every precision."""
    if _ragged(product):
        # A ragged dot's group sizes count its rows and multiply nothing (`counts`): its product is of the two
        # others, the sizes passed through as they are.
        *multiplied, _ = operands
        if _sums_in_float64(*multiplied):
            return _summed_wide(product, preferred_element_type, *operands, counts=1)
        return product(*operands, preferred_element_type=preferred_element_type)
    if not _sums_in_float64(*operands):
        return product(*operands, preferred_element_type=preferred_element_type)
    if (key := _contraction(product, preferred_element_type)) is not None:
        return _traced_once(key, _summed_wide)(*operands)
    return _summed_wide(product, preferred_element_type, *operands)


def _ragged(product: Callable) -> bool:
    """Whether `product` is `jax.lax.ragged_dot_general` (bound through `functools.partial`s), whose last
    operand is its group sizes."""
    while isinstance(product, functools.partial) and type(product) is functools.partial:
        product = product.func
    return product is jax.lax.ragged_dot_general


CONTRACTIONS = (jnp.einsum, jnp.matmul, jnp.dot, jnp.tensordot, jnp.vdot, jnp.inner, jax.lax.dot_general)
"""The contractions `accumulated` traces once a call (`_contraction`): JAX's own, whose trace states no
derivative of its own (`_own_backward`) and depends on its operands' types and its arguments alone."""

_JAX_CODE = (jax.lax.dot_general.__code__, jnp.einsum.__code__)
"""The code of the contractions every other one calls through, as JAX defines it: a library that
intercepts JAX's products in place (Qwix's `quantize_model` replaces `jax.lax.dot_general`'s and
`jax.numpy.einsum`'s) changes it (`_quantized`)."""

_REACHED = {jnp.einsum: ("einsum", "dot_general"), jnp.dot: ("dot", "dot_general")}
"""The ops of JAX's a contraction reaches, as Qwix names the ops it intercepts, where more than
`dot_general` (which every other contraction calls through)."""


def _quantized(function: Callable) -> bool:
    """Whether Qwix quantizes the contraction `function` at the module calling it: under its interception of
    JAX's products (`_JAX_CODE` changed), a rule of its matching an op the contraction reaches, as Qwix's own
    query answers it (`qwix.pallas.get_current_rule`, which its providers intercept). Qwix owns the products
    it quantizes; a contraction no rule matches stays float, and Dew's sums govern it, so a quantized model's
    kept-in-float parts compute as the unquantized model's."""
    if (jax.lax.dot_general.__code__, jnp.einsum.__code__) == _JAX_CODE:
        return False
    from qwix import pallas as qwix_pallas

    return any(qwix_pallas.get_current_rule(op) is not None
               for op in _REACHED.get(function, ("dot_general",)))


def _contraction(product: Callable, preferred_element_type) -> tuple | None:
    """`product` as a key of one of JAX's contractions (`CONTRACTIONS`, each as JAX defines it), bound through
    plain `functools.partial`s around it (a subclass may call it otherwise) to arguments with no array among
    them, with the dtype it is asked for: the contraction, the arguments' tree and their leaves; None for any
    other product, and for one Qwix quantizes (`_quantized`), whose product is its own."""
    function, arguments, keywords = product, (), {}
    while isinstance(function, functools.partial) and type(function) is functools.partial:
        arguments, keywords = function.args + arguments, {**function.keywords, **keywords}
        function = function.func
    if not any(function is contraction for contraction in CONTRACTIONS) or _quantized(function):
        return None
    leaves, tree = jax.tree.flatten((arguments, keywords, preferred_element_type))
    if not all(isinstance(leaf, Hashable) and not isinstance(leaf, jax.Array | np.ndarray)
               for leaf in leaves):
        return None
    return function, tree, tuple(leaves)


@functools.lru_cache(maxsize=1024)
def _traced_once(key: tuple, summed: Callable) -> Callable:
    """`summed` (`_summed_wide`) of the contraction `key` names (`_contraction`) as one
    program JAX traces once for its operands' types: a gated delta net layer's portable chunked math sums
    some 64 contractions a call, each traced afresh at about 6.6 ms (the wide sum, its derivative and the
    narrow one each traced again), 130 s of a decision witness's trace under Dew's test arithmetic (row
    3142). Its arithmetic is the product's own, the call a nested program XLA inlines. The most recent 1024
    contractions are kept, a model's working set many times over."""
    function, tree, leaves = key
    arguments, keywords, preferred_element_type = jax.tree.unflatten(tree, leaves)
    product = functools.partial(function, *arguments, **keywords)
    return nested(functools.partial(summed, product, preferred_element_type))


def _summed_wide(product: Callable, preferred_element_type, *operands, counts: int = 0) -> jax.Array:
    """`accumulated`'s product under Dew's wide CPU arithmetic: its float64 sum where it takes the
    platform's choice (`platform_dependent`), the plain product where its result is wider than float32 or
    its library states its derivative; its last `counts` operands multiply nothing (`float64_sum`)."""
    from dew.nn.kernels.generation import linear_tangent

    narrow = functools.partial(product, preferred_element_type=preferred_element_type)
    traced, shape = jax.make_jaxpr(narrow, return_shape=True)(*operands)
    dtype = shape.dtype
    if jnp.promote_types(dtype, jnp.float32) != jnp.float32 or _own_backward(traced.jaxpr):
        # A wider result converts the operands to it, so the products are not float32's; a product
        # with a backward of its own is its library's.
        return narrow(*operands)

    def summed(*operands):
        return float64_sum(product, *operands, dtype=dtype, counts=counts)

    def pulled(saved, operands, cotangent, live):
        """The cotangents of the operands at `live`, each the float64 sum of its products rounded
        once: JAX transposes a product after its trace, outside the scope that widened it, so the
        pullback is formed in that scope here, for the operands a reverse pass differentiates alone."""
        with jax.enable_x64(new_val=True):
            wide = _widened(operands, counts)
            wide_cotangent = widened_exactly(cotangent, jnp.float64)

            def transposed(index):
                # The product is linear in each operand alone, so its transpose there forms the
                # cotangent's products and nothing of the forward.
                def alone(operand):
                    return product(*wide[:index], operand, *wide[index + 1:],
                                   precision=jax.lax.Precision.HIGHEST, preferred_element_type=None)

                (gradient,) = jax.linear_transpose(alone, wide[index])(wide_cotangent)
                return gradient.astype(operands[index].dtype)

            # No barrier here: which operand is a parameter is the caller's to know, and a projection's
            # own reverse mode, over this product, ties its kernel's gradient (`projections.dot_general`).
            return tuple(transposed(index) for index in live)

    # Forward mode differentiates `summed` as it traces, inside its scope; reverse mode runs `pulled`.
    wide = contracted("test_a_wide_cpu_sum_of_products_holds_every_order")(
        linear_tangent(lambda *operands: (summed(*operands), ()), pulled, summed, live_cotangents=True,
                       kind=Protected()))
    return jax.lax.platform_dependent(*operands, cpu=wide, default=narrow)


L2NORM_EPS = 1e-6
"""The epsilon `l2norm` adds under its square root: FLA's, and Dory's `justnorm`'s."""

ROUNDING_STREAM = 0x726F756E
"""The stream stochastic rounding draws from: what a training step folds its key with
(`dew.objectives.base.training_rngs`), and what a served row's rounding key starts from
(`dew.sampling.text.rounding_keys`), apart from dropout's and sampling's."""


def span_means(values: jax.Array, spans: jax.Array) -> jax.Array:
    """The mean of `values` `[B, L, D]` over each `[..., 2]` (start, end) span of its row.

    `spans` is `[B, ...]`; an empty span, a padding slot's, gives zeros. A token no span of its row
    reads is selected away before the product, so padding's NaN or infinity never reaches a mean (the
    rule of `weighted_total`), and the sum is the product's `accumulated` one.
    """
    positions = jnp.arange(values.shape[1])
    inside = (positions >= spans[..., :1]) & (positions < spans[..., 1:])
    counts = jnp.maximum(jnp.sum(inside, axis=-1, keepdims=True), 1)
    flat = inside.reshape(inside.shape[0], -1, inside.shape[-1])
    read = jnp.where(jnp.any(flat, axis=1)[..., None], values, 0)
    summed = accumulated(functools.partial(jnp.einsum, "bnl,bld->bnd"), flat.astype(values.dtype), read)
    summed = summed.reshape(*inside.shape[:-1], values.shape[-1])
    return summed / counts.astype(values.dtype)


def l2norm(x: jax.Array, eps: float = L2NORM_EPS) -> jax.Array:
    """`x` over its L2 length along the last axis, `x * rsqrt(sum(x**2) + eps)` with `eps` added under
    the root, computed in at least fp32 (`at_least_fp32`, a float64 input's in float64) and rounded
    once to `x`'s dtype: the normalization of FLA's `l2norm` (the linear attentions' queries and
    keys, modeling_qwen3_next.py:368-371) and of nGPT's `justnorm` (its residual stream and
    weights)."""
    return l2_unit(*l2_squares(x), x.dtype, eps)


def l2_squares(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """`x` in at least fp32 and its rows' sums of squares: `l2norm`'s reduction, which a reverse that keeps
    them reads (nGPT's join, `dew.nn.ngpt`)."""
    wide = x.astype(at_least_fp32(x.dtype))
    return wide, jnp.sum(jnp.square(wide), axis=-1, keepdims=True)


def l2_normed(x: jax.Array, dtype, eps: float = L2NORM_EPS) -> jax.Array:
    """`l2norm` of `x`'s rows rounded once to `dtype`, as `l2_unit(*l2_squares(x), dtype, eps)` computes it,
    its derivative the unit norm's own, `r (t - y <y, t>)` for the unit row `y = x r` and its transpose
    `r (c - y <y, c>)`: every term at the row's own scale, where JAX's derivative of `rsqrt` reads
    `rsqrt(s) / s`, which flushes to zero at a row of length 2^55 and leaves a radial cotangent its
    `r c` term alone, 2^-55 where the exact gradient is 0 (nGPT's join and hidden norm, row 11940's
    review: their written transposes cancelled it)."""
    return _unit_rows(eps, *l2_squares(x)).astype(dtype)


def _unit_rows_value(eps: float, wide: jax.Array, squares: jax.Array) -> jax.Array:
    """The rows `wide` over their lengths, from their sums of squares (`l2_normed`'s, in its wide dtype)."""
    return wide * jax.lax.rsqrt(squares + eps)


def _unit_rows_tangent(eps: float, primals, tangents):
    wide, squares = primals
    along, _ = tangents
    inverse = _inverse_length(eps, wide, squares)
    unit = wide * inverse

    def projected(v: jax.Array) -> jax.Array:
        return v - unit * jnp.sum(unit * v, axis=-1, keepdims=True)

    # The sums of squares are the rows' own (`l2_squares`): their tangent is the rows' through the norm.
    # r P(t) for P(t) = t - y <y, t>, the unit row y (`1 - eps r^2` long), written as
    # P(r P(t)) + r eps r^2 y <y, t> (P P = P - eps r^2 y y^T, and P its own transpose): every term
    # projected or summed before it is scaled, both forward and in the reverse JAX transposes it into, so a
    # radial tangent or cotangent as large as float32 holds meets a short row's inverse length only as the
    # small part the eps leaves, where either order of one projection scaled one direction's first and
    # overflowed to inf - inf (the API02 review).
    radial = (inverse * (eps * inverse * inverse)) * jnp.sum(unit * along, axis=-1, keepdims=True)
    return unit, projected(inverse * projected(along)) + unit * radial


def _inverse_length_value(eps: float, wide: jax.Array, squares: jax.Array) -> jax.Array:
    del wide
    return jax.lax.rsqrt(squares + eps)


def _inverse_length_tangent(eps: float, primals, tangents):
    wide, squares = primals
    along, _ = tangents
    inverse = _inverse_length(eps, wide, squares)
    # dr = -r^3 <x, dx> = -r (r <y, dx>) for the unit row y = x r, read from the row's own tangent: every
    # factor at the row's scale in both directions, where through its squares' tangent (2 <x, dx>, 2^56 at a
    # row of length 2^55) a second order's reverse met r^3 first and flushed to 0 (2^-166, the API02 review).
    return inverse, -inverse * (inverse * jnp.sum(wide * inverse * along, axis=-1, keepdims=True))


_inverse_length = contracted("test_ngpts_join_and_normed_hidden_hold_every_order")(
    declared_tangent(Protected(), _inverse_length_value, _inverse_length_tangent, nondiff_argnums=(0,)))
"""The rows' inverse lengths `rsqrt(squares + eps)` from the rows `wide` and their sums of squares, its
derivative the rows' own (`_unit_rows`')."""


_unit_rows = contracted("test_ngpts_join_and_normed_hidden_hold_every_order")(
    declared_tangent(Protected(), _unit_rows_value, _unit_rows_tangent, nondiff_argnums=(0,)))
"""`_unit_rows_value` with the unit norm's own tangent, every term at the row's scale (`l2_normed`)."""


def l2_unit(wide: jax.Array, squares: jax.Array, dtype, eps: float = L2NORM_EPS) -> jax.Array:
    """`l2norm` of the rows `wide` from their sums of squares (`l2_squares`), rounded once to `dtype`."""
    return (wide * jax.lax.rsqrt(squares + eps)).astype(dtype)


def float32_dot_general(lhs, rhs, dimension_numbers, precision: jax.lax.PrecisionLike = None,
                        preferred_element_type: DTypeLike | None = None, **kwargs) -> jax.Array:
    """`jax.lax.dot_general` at `precision`, accumulated as Dew accumulates a CPU's float32 products
    (`accumulated`): the product a flax layer forms (`nn.Dense(dot_general=...)`). Unstated, JAX's
    configured precision decides (`jax_default_matmul_precision`), as it does a flax layer's. At the
    `BF16_BF16_F32` algorithm it is `bf16_dot_general`'s product."""
    if _bf16_algorithm(precision):
        return _held(bf16_dot_general(lhs, rhs, dimension_numbers), preferred_element_type, lhs, rhs)
    from .sharding import contracted_alike

    if kwargs.get("out_sharding") is None:
        # A contraction Explicit axes split on both operands lands its sum where JAX asks to be told.
        lhs, rhs, placed = contracted_alike(lhs, rhs, dimension_numbers)
        kwargs = kwargs if placed is None else {**kwargs, "out_sharding": placed}
    return accumulated(functools.partial(jax.lax.dot_general, dimension_numbers=dimension_numbers,
                                         precision=precision, **kwargs),
                       lhs, rhs, preferred_element_type=preferred_element_type)


def float32_einsum(subscripts: str, *operands, precision: jax.lax.PrecisionLike = None,
                   preferred_element_type: DTypeLike | None = None, **kwargs) -> jax.Array:
    """`jnp.einsum` at `precision`, JAX's configured one where unstated, as `float32_dot_general` is a
    layer's, accumulated alike, and `bf16_einsum`'s product at the `BF16_BF16_F32` algorithm."""
    if _bf16_algorithm(precision):
        return _held(bf16_einsum(subscripts, *operands, **kwargs), preferred_element_type, *operands)
    return accumulated(functools.partial(jnp.einsum, subscripts, precision=precision, **kwargs), *operands,
                       preferred_element_type=preferred_element_type)


def _bf16_algorithm(precision: jax.lax.PrecisionLike) -> bool:
    """Whether `precision` names XLA's `BF16_BF16_F32` dot algorithm, which Dew's products run as
    `bf16_dot_general`."""
    return precision_names(precision) == {jax.lax.DotAlgorithmPreset.BF16_BF16_F32.name}


def _held(product: jax.Array, preferred_element_type: DTypeLike | None, *operands) -> jax.Array:
    """A product the dot family formed in fp32 as the dtype it would have returned: the preferred
    one, else its operands' (`products_dtype`)."""
    held = products_dtype(*operands) if preferred_element_type is None else preferred_element_type
    return product.astype(held)


def float32_matmul(lhs, rhs, precision: jax.lax.PrecisionLike = None,
                   preferred_element_type: DTypeLike | None = None) -> jax.Array:
    """`lhs @ rhs` at `precision`, JAX's configured one where unstated, accumulated as
    `float32_dot_general` is, summed and returned in `preferred_element_type` where given."""
    return accumulated(functools.partial(jnp.matmul, precision=precision), lhs, rhs,
                       preferred_element_type=preferred_element_type)


@dataclasses.dataclass(frozen=True)
class Dtypes:
    """A run's dtypes, three as Flax casts them: the parameters' (its layers' `param_dtype`, and the
    optimizer's moments'), the compute's (their `dtype`) and the outputs' (the logits and losses a
    model returns). A source the run keeps frozen is stored as its checkpoint stores it, unless the
    parameters' dtype is narrower (`dew.config.RunConfig.source_keywords`)."""

    params: str = "float32"
    compute: str = "bfloat16"
    """bf16 by default, Lightning's bf16-mixed, where Flax computes in its parameters' float32: a run's
    matmuls run on the tensor cores' bf16 rate, its float32 masters and moments keep the update exact,
    and the float32 products a run needs say so (`exact_float32`); "32-true" computes in float32."""
    output: str = "float32"

    def __post_init__(self):
        for name in ("params", "compute", "output"):
            if not jnp.issubdtype(jnp.dtype(getattr(self, name)), jnp.floating):
                raise ValueError(f"a run's {name} dtype is a floating one, not {getattr(self, name)!r}")


def output_dtype(dtype: Dtype | None) -> jnp.dtype:
    """The dtype a model returns its outputs (its logits) in: its run's `Dtypes.output`
    (`dew.nn.kernels.generation.numerics`, float32 outside a run), never narrower than its compute
    `dtype` (None for Flax's float32), so a float64 truth's stay float64."""
    from dew.nn.kernels.generation import numerics

    return jnp.promote_types(numerics().dtypes.output, jnp.float32 if dtype is None else dtype)


DTYPE_PRESETS = {
    "bf16-mixed": Dtypes(),
    "32-true": Dtypes(compute="float32"),
    "bf16-true": Dtypes(params="bfloat16", output="bfloat16"),
    "16-mixed": Dtypes(compute="float16"),
}
"""The usual dtypes by Lightning's names: float32 parameters computing in bf16 by default, float32
throughout, bf16 throughout, and float16 compute over float32 parameters."""

MATMUL_PRECISIONS = ("default", "high", "highest")
"""JAX's matmul precision levels (`jax_default_matmul_precision`): "default" TF32 for a float32 product
on a GPU and one bf16 pass on a TPU, "highest" IEEE float32."""


PROJECTION_ROLES = ("mixer", "mlp", "experts", "router", "shared_gate")
"""The roles a model's projections play, which a low-precision recipe chooses among (`LowPrecision`): a
token mixer's (attention's queries, keys, values and output, a linear mixer's), a dense feed-forward's, the
routed experts', a router's scores, a shared expert's gate. A projection that plays none of them, a head's
or an embedding's, keeps the run's dtypes under every recipe."""

@dataclasses.dataclass(frozen=True)
class LowPrecision:
    """A low-precision training recipe for a run's projections by role (row 2513): `recipe`, one of
    `dew.nn.kernels.projections.RECIPES` (the one declaration of the recipes: their projections, blocks and
    routes), which a run's numerics check where they take effect (`generation.run_precision`), multiplies
    the projections whose role is among `roles` (`PROJECTION_ROLES`), each where its widths take the
    recipe's blocks; every other product keeps the run's dtypes, the routers, norms, embeddings and heads
    among them, as every published low-precision recipe keeps them. `tensor_scaling`, NVFP4's activations'
    tensor scale: "current" (the default), each tensor's own largest magnitude, exactly; "delayed" (opt-in,
    NVFP4 alone), the largest of the last `amax_history` steps' (Transformer Engine's delayed scaling), which
    the run's state holds and hands each projection (`recipe_projection`'s `tensor_scale`): a scale known
    before the tensor is, so the product forming it may quantize it in its epilogue (row 12721)."""

    recipe: str = "blockwise_fp8"
    roles: tuple[str, ...] = ("mixer", "mlp", "experts")
    tensor_scaling: str = "current"
    amax_history: int = 16

    def __post_init__(self):
        object.__setattr__(self, "roles", tuple(self.roles))
        unknown = sorted(set(self.roles) - set(PROJECTION_ROLES))
        if unknown:
            raise ValueError(f"a recipe's roles are among {PROJECTION_ROLES}, not {unknown}")
        if self.tensor_scaling not in ("current", "delayed"):
            raise ValueError(f"tensor_scaling is 'current' or 'delayed', not {self.tensor_scaling!r}")
        if self.tensor_scaling == "delayed" and self.recipe != "nvfp4":
            raise ValueError(f"delayed tensor scaling is NVFP4's alone, not {self.recipe!r}'s")
        if (not isinstance(self.amax_history, int) or isinstance(self.amax_history, bool)
                or self.amax_history < 1):
            raise ValueError(f"an amax history holds a whole number of steps, at least one, not "
                             f"{self.amax_history!r}")


@contextlib.contextmanager
def projection_role(role: str):
    """Within the block, the projections a model traces play `role` (`PROJECTION_ROLES`), which the run's
    low-precision recipe reads (`recipe_role`): a block sets its mixer's and its feed-forward's, a mixture
    its router's and its experts', the innermost holding. JAX keys its trace caches on it
    (`dew._cache.PROJECTION_ROLE`), so a function jitted under one role traces again under another."""
    from dew import _cache

    if role not in PROJECTION_ROLES:
        raise ValueError(f"a projection's role is one of {PROJECTION_ROLES}, not {role!r}")
    with _cache.PROJECTION_ROLE(role):
        yield


def recipe_role() -> tuple[LowPrecision, str | None] | None:
    """The run's low-precision recipe and the role of the projection being traced (None for a projection of
    no role), or None where the run trains in no recipe (`Numerics.low_precision`)."""
    from dew import _cache
    from dew.nn.kernels.generation import numerics

    recipe = numerics().low_precision
    return None if recipe is None else (recipe, _cache.PROJECTION_ROLE.value)


class RecipeRefused(ValueError):
    """A projection the run's low-precision recipe takes by its role holds another owner's product
    (`refuse_recipe`)."""


def refuse_recipe(owner: str, depth: int, width: int) -> None:
    """Refuse a `depth` x `width` projection the run's low-precision recipe takes (`recipe_refusal`, the one
    decision: its role the recipe's, its widths whole blocks of the recipe's) whose product `owner` holds
    instead (a quantizer's product class, a wrapper's interception, an adapter's fused product), which would
    train it unquantized, silently: the recipe and a product wrapper compose nowhere yet. A projection the
    recipe leaves (a role it keeps, widths of no whole blocks) runs under the wrapper as without a recipe."""
    from dew.nn.kernels.projections import recipe_refusal

    held = recipe_role()
    if held is not None and recipe_refusal(depth, width) is None:
        raise RecipeRefused(f"the run's {held[0].recipe} recipe takes this {depth} x {width} {held[1]} "
                            f"projection, whose product {owner} holds: train it in the recipe or under the "
                            "wrapper, not both")


@dataclasses.dataclass(frozen=True)
class Numerics:
    """A run's numerics, set once and recorded with it, which its model, optimizer and trainer read: its
    dtypes (`Dtypes`, which a model takes as Flax's `param_dtype` and `dtype` and its optimizer as its
    moments'; `DTYPE_PRESETS` names the usual ones by Lightning's names), the precision every product
    that states none multiplies at (JAX's own `jax_default_matmul_precision` level, or a
    `jax.lax.DotAlgorithmPreset`, which a record holds by its name), and each op's form, its kernels'
    fast or accurate arithmetic (`dew.nn.kernels.generation.run_precision`). A precision a call or a
    model states still wins, and determinism is a setting of its own. The default is bf16-mixed at
    JAX's default precision in the fast forms; `ACCURATE` names the scientific one
    (`NUMERICS_PRESETS`)."""

    dtypes: Dtypes = dataclasses.field(default_factory=Dtypes)
    matmul: str | jax.lax.DotAlgorithmPreset = "default"
    forms: Literal["fast", "accurate"] = "fast"
    low_precision: LowPrecision | None = None
    """The recipe the run's projections train in by role (`LowPrecision`), None for none."""
    loss_scaling: Literal["auto", "none", "dynamic"] = "auto"
    """How the loss is scaled before its gradient: "dynamic" by a scale that halves where a gradient
    overflows and grows where none has (`flax.training.dynamic_scale`), "none" not at all, "auto" dynamic
    where the run computes in float16, whose range a gradient overflows, else none (`scales_loss`)."""

    def __post_init__(self):
        if isinstance(self.dtypes, str):
            # A record or a flag names the usual dtypes by Lightning's name (`DTYPE_PRESETS`).
            if self.dtypes not in DTYPE_PRESETS:
                raise ValueError(f"a run's dtypes are a Dtypes record or one of {sorted(DTYPE_PRESETS)}, not "
                                 f"{self.dtypes!r}")
            object.__setattr__(self, "dtypes", DTYPE_PRESETS[self.dtypes])
        if isinstance(self.matmul, str) and self.matmul in jax.lax.DotAlgorithmPreset.__members__:
            # A record holds a preset by its name (`dew.registry.to_record`).
            object.__setattr__(self, "matmul", jax.lax.DotAlgorithmPreset[self.matmul])
        if not isinstance(self.matmul, jax.lax.DotAlgorithmPreset) and self.matmul not in MATMUL_PRECISIONS:
            raise ValueError(f"a run's matmul precision is one of JAX's levels {MATMUL_PRECISIONS} or a "
                             f"jax.lax.DotAlgorithmPreset, not {self.matmul!r}")
        if self.forms not in ("fast", "accurate"):
            raise ValueError(f"a run's forms are fast or accurate, not {self.forms!r}")
        if self.loss_scaling not in ("auto", "none", "dynamic"):
            raise ValueError(f"a run's loss scaling is auto, none or dynamic, not {self.loss_scaling!r}")

    def scales_loss(self, compute: str | None = None) -> bool:
        """Whether the run scales its loss dynamically (`loss_scaling`): stated, or "auto" where it computes
        in float16, `compute` where its model states its own compute dtype, else its dtypes'."""
        if self.loss_scaling == "auto":
            return jnp.dtype(self.dtypes.compute if compute is None else compute) == jnp.float16
        return self.loss_scaling == "dynamic"


ACCURATE = Numerics(DTYPE_PRESETS["32-true"], matmul="highest", forms="accurate")
"""Accurate, scientific work at its cost: float32 throughout, every product that states no precision in
IEEE float32 and each op's accurate form."""

FP8 = Numerics(DTYPE_PRESETS["bf16-mixed"], low_precision=LowPrecision())
"""DeepSeek-V3's FP8 training on Hopper: bf16-mixed, the token mixers', dense feed-forwards' and routed
experts' projections in blockwise FP8 (`LowPrecision`), the routers, norms, embeddings and head in the run's
dtypes."""

MXFP8 = Numerics(DTYPE_PRESETS["bf16-mixed"], low_precision=LowPrecision(recipe="mxfp8"))
"""MXFP8 training (OCP MX): bf16-mixed, the token mixers', dense feed-forwards' and routed experts'
projections in MXFP8 (`LowPrecision`), the routers, norms, embeddings and head in the run's dtypes."""

NVFP4 = Numerics(DTYPE_PRESETS["bf16-mixed"], low_precision=LowPrecision(recipe="nvfp4"))
"""NVFP4 training: bf16-mixed, the token mixers', dense feed-forwards' and routed experts' projections in
NVFP4 (`LowPrecision`), the routers, norms, embeddings and head in the run's dtypes."""

NUMERICS_PRESETS = {"accurate": ACCURATE, "fp8": FP8, "mxfp8": MXFP8, "nvfp4": NVFP4}
"""The named run numerics beside the default (`Numerics()`)."""


def numerics_of(given: Numerics | str) -> Numerics:
    """A run's numerics from what a flag, a record or a load names: a `Numerics` as it is, a preset's
    name (`NUMERICS_PRESETS`, "default" for `Numerics()`), or the usual dtypes' Lightning name
    (`DTYPE_PRESETS`) at the default precision and forms."""
    if isinstance(given, Numerics):
        return given
    if given == "default":
        return Numerics()
    if given in NUMERICS_PRESETS:
        return NUMERICS_PRESETS[given]
    if given in DTYPE_PRESETS:
        return Numerics(DTYPE_PRESETS[given])
    raise ValueError(f"a run's precision is a Numerics record, 'default', one of {sorted(NUMERICS_PRESETS)} "
                     f"or a dtype preset of {sorted(DTYPE_PRESETS)}, not {given!r}")


def exact_float32(precision: jax.lax.PrecisionLike) -> jax.lax.PrecisionLike:
    """The precision a float32 product whose rounding decides what follows takes, a numerically critical
    one, a vocabulary head's (the loss's logits), a router's or the state's of Dew's own recurrence math
    (`recurrence_algebra`'s products, the TPU scan's, `ssd`), each stating its reason where it calls this:
    the one the call states, else IEEE float32 (`Precision.HIGHEST`), stated over JAX's configured
    precision, whose default runs TF32 on a GPU and one bf16 pass on a TPU, where the references multiply
    IEEE float32 (torch's float32 matmul). Every other product follows JAX's configured precision and the
    run's numerics (`dew.nn.kernels.generation.run_precision`), a port of an upstream's kernels among them
    and not this rule's: the gated delta rule's and Kimi delta attention's of fla's and the SSD's of
    mamba_ssm's (`fla_gdn._dot`) multiply an fp32 product as their upstream's Triton dots do at their own
    default, one TF32, under a run's numerics at the default precision (`Numerics()`), and decline a run
    whose configured precision is above the default (`ACCURATE`'s 'highest', `recurrence_chunk.
    _port_precision`), whose call takes the portable math at that precision (the owner, 2026-11-11: "tf32
    default, fp32 if accurate")."""
    return jax.lax.Precision.HIGHEST if precision is None else precision


def configurable_precision(precision: jax.lax.PrecisionLike) -> bool:
    """Whether `jax.default_matmul_precision` can name `precision`: one precision or dot algorithm
    preset, unset or default included; not a pair of precisions or a dot algorithm."""
    return len(precision_names(precision)) <= 1 and not isinstance(precision, jax.lax.DotAlgorithm)


def _configured_name(precision: jax.lax.PrecisionLike) -> str | None:
    """The name `jax.default_matmul_precision` takes `precision` by, the default's 'default' among them;
    None for an unset one. A precision no config names (`configurable_precision`) is refused."""
    if not configurable_precision(precision):
        raise ValueError(f"precision={precision} has no jax_default_matmul_precision name")
    names = precision_names(precision)
    if not names:
        return None
    (name,) = names
    # A level's lower-case name (DotAlgorithmPreset has a DEFAULT of its own, which no config names).
    return name.lower() if name in jax.lax.Precision.__members__ else name


def configured_precision(precision: jax.lax.PrecisionLike, *, exact: bool = False):
    """A context in which a product that takes no precision of its own (jax.nn's xla attention)
    multiplies at `precision`, through `jax.default_matmul_precision`; an unset precision changes nothing,
    and a default one nothing unless `exact`, where it overrides an outer higher one (splash's arms, each
    at exactly its own). A precision no config names (`configurable_precision`) is refused."""
    name = _configured_name(precision)
    kept = name is None or (name == 'default' and not exact)
    return contextlib.nullcontext() if kept else jax.default_matmul_precision(name)


def operand_precision(dtype: Dtype | None, precision: jax.lax.PrecisionLike = None) -> jax.lax.PrecisionLike:
    """The precision a product of `dtype` operands takes: the one stated, JAX's configured one where
    none is, and for bf16 operands at the default precision the algorithm that keeps a bf16 dot bf16
    in both directions. A caller that asked for more than the default keeps what it asked for, and so
    does a run whose configured level asks for more where the call states none (a run's
    `Numerics.matmul`, `generation.run_precision`), and a GPU older than sm80, which rejects the
    algorithm at run time (`dew.nn.kernels.generation.BF16_GPU`). A configured dot algorithm preset is
    no level a run picks but how dots that state none sum (check 1's arms state F32_F32_F32), which the
    algorithm's fp32 sums already are, so it keeps the algorithm."""
    asked = effective_precision(precision)
    presets = set(jax.lax.DotAlgorithmPreset.__members__)
    if not precision_names(precision) and precision_names(asked) & presets:
        asked = None
    if dtype is None or jnp.dtype(dtype) != jnp.bfloat16 or not asks_default_precision(asked):
        return precision
    # Imported here: dew.nn.kernels imports this module.
    from .kernels.generation import bf16_dot_runs
    if not bf16_dot_runs():
        return precision
    return own_preset(dtype)


def fp32_result_dot_general(precision: PrecisionLike = None):
    """A flax layer's `dot_general` for a head whose result stays at least
    fp32.

    The layer has already promoted both operands when this runs - to its own
    `dtype`, or to the activations' where it carries none - so the compute
    dtype is the operands' own. Only the accumulation and the result widen,
    to `at_least_fp32` of it.
    """
    requested = precision

    def dot_general(lhs, rhs, dimension_numbers, precision=None,
                    preferred_element_type=None):
        del precision, preferred_element_type  # this head's policy, not the layer's
        return float32_dot_general(lhs, rhs, dimension_numbers,
                                   precision=operand_precision(lhs.dtype, requested),
                                   preferred_element_type=at_least_fp32(lhs.dtype))

    return dot_general


def scaled(x: jax.Array, factor: float) -> jax.Array:
    """`x * factor` with the factor kept in fp32 and only the product rounded
    to `x`'s dtype, which is what torch's `bf16_tensor * python_float` does
    (fp32 opmath) and what Gemma's `embed_scale` here does. Rounding 0.22 to
    bf16 first would shrink every product by a systematic 0.12%."""
    if factor == 1.0:
        return x
    return (x.astype(at_least_fp32(x.dtype)) * factor).astype(x.dtype)


def rounded_to(x: jax.Array, dtype: Dtype) -> jax.Array:
    """`x` rounded to `dtype`'s precision and held in `x`'s own dtype: where a
    torch computation stores a tensor of `dtype`. Under jit XLA's GPU default
    `xla_allow_excess_precision` deletes a narrowing cast a widening one
    follows; `reduce_precision` keeps the rounding but flushes float16
    subnormals, so float16 rounds by the cast behind an optimization barrier
    (`rounded_operand`). The derivative rounds where the value does, as a torch
    tensor of `dtype`'s gradient does."""
    if jnp.dtype(dtype) == jnp.dtype(jnp.float16):
        return jax.lax.optimization_barrier(x.astype(dtype)).astype(x.dtype)
    bits = jnp.finfo(dtype)
    return jax.lax.reduce_precision(x, exponent_bits=bits.nexp, mantissa_bits=bits.nmant)


def _protected(transform: Callable) -> Callable:
    """`transform` as one span the algebra pass keeps as it is (`dew.nn.derivatives.Protected`), as
    reassociation would cancel the residual an error-free transform keeps (`bound_span`: its arrays the span's
    operands, its constants bound in, so a constant still splits where it is known, `two_product`)."""
    return bound_span(Protected(), transform)


MARKED_BACKENDS = ("cpu", "cuda")
"""The platforms whose compiler rounds an operation the fork's no-contract attribute marks on its own, so a
marked transform's operands need no barrier there (`uncontracted`, chosen as the program lowers for its
platform): XLA:CPU's fusion compiler fences the mark (the fork's xla no_contract_mark patch); XLA:GPU's
native fusion emitters on CUDA contract no float add, subtract or multiply, marked or not, NVPTX fusing only
an operation flagged to contract, which no emitter but the CPU's sets (row 10135: a marked and a plain
`a * b + c` bitwise the separately rounded one on an A100 and a GH200 at post115;
`test_no_contract.a_marked_multiply_add_rounds_each_step` fails should a later XLA contract one). ROCm's and
a TPU's take the barrier."""

TRACED_MARKED = ("cpu",)
"""The platforms a call's code is chosen for (`dew.nn.kernels.generation.backend`) whose traces are marked
(`rounding_alone`): a CPU's, which may lower for a platform of `MARKED_BACKENDS` or another (`uncontracted`'s
platform choice). A GPU's trace keeps the barrier: a raw `pallas_call`'s body traced on a GPU outside Dew's
launches would meet the platform choice, which Pallas's Triton and Mosaic GPU lowerings refuse, and no public
JAX predicate states a Pallas trace yet (row 10138)."""

def _as_they_are(values: tuple) -> tuple:
    """`values` unbarriered: a platform choice's branch where its compiler fences the mark
    (`uncontracted`)."""
    return values


_UNBARRIERED = dict.fromkeys(MARKED_BACKENDS, _as_they_are)
"""`uncontracted`'s branches by the platform a program lowers for, each that fences the mark's."""

_MARKING = contextvars.ContextVar("dew.nn.precision.marking", default=False)
"""Whether the arithmetic traced now is marked to round alone (`rounding_alone`), so the operands it reads
need no barrier (`uncontracted`)."""


def no_contract() -> str | None:
    """The frontend attribute under which a compiler of `MARKED_BACKENDS` rounds a float add, subtract or
    multiply on its own, never contracted into a fused multiply-add (`jax.extend.backend
    .no_contract_attribute`, the fork's from post112); None where the jaxlib marks nothing (a stock jaxlib),
    or where the suite asks the barrier path (`dew._cache.BARRIERED`)."""
    from jax.extend.backend import no_contract_attribute

    from dew import _cache

    return None if _cache.BARRIERED.value else no_contract_attribute()


@contextlib.contextmanager
def rounding_alone():
    """Every float add, subtract and multiply traced in the block marked to round on its own, never
    contracted into a fused multiply-add, where the jaxlib marks so (`no_contract`, `set_xla_metadata`) and
    the trace is for a platform `TRACED_MARKED` names (the platform a call's code is chosen for,
    `dew.nn.kernels.generation.backend`): an error-free transform's arithmetic, each step's rounded value the
    one every later step reads, its operands read without a barrier where the platform the program lowers for
    fences the mark too (`uncontracted`). Elsewhere, and in the body of a kernel Dew launches
    (`dew.nn.kernels.launches.interpreting`), nothing is marked and the transforms hold their operands behind
    a barrier. A raw `pallas_call` outside Dew's launches, traced for a CPU and lowered for CUDA, meets the
    platform choice, which Triton's lowering refuses by name: no public JAX predicate states a Pallas trace
    yet (row 10138)."""
    from dew.nn.kernels.generation import backend
    from dew.nn.kernels.launches import interpreting

    name = no_contract()
    if name is None or backend() not in TRACED_MARKED or interpreting() is not None:
        yield
        return
    from jax.experimental.xla_metadata import set_xla_metadata

    held = _MARKING.set(True)
    try:
        with set_xla_metadata(**{name: True}):
            yield
    finally:
        _MARKING.reset(held)


@_protected
def two_sum(left, right):
    """`left + right` rounded, and exactly what the rounding dropped (Knuth's TwoSum). Each operand is read as
    one value: a compiler that fused the multiply making an operand into the sum (an FMA, XLA's CPU and GPU
    contract one) would add its exact product where the steps after read its rounded one, counting that
    rounding twice; its additions are marked to round alone where the jaxlib marks (`rounding_alone`), its
    operands held behind a barrier elsewhere (`uncontracted`)."""
    with rounding_alone():
        return _two_sum(*uncontracted(left, right))


class _Summed(Protocol):
    """A value whose `+` and `-` with one of its own kind give its kind (an array, a native kernel's word,
    `templates.native.Word`): what an error-free transform's arithmetic reads (`renormalized_sum`)."""

    def __add__(self, other: Self, /) -> Self: ...

    def __sub__(self, other: Self, /) -> Self: ...


def _two_sum[V: _Summed](left: V, right: V) -> tuple[V, V]:
    """`two_sum`'s arithmetic over operands no multiply makes in the same fusion: a compensated pair's
    halves, made by additions alone, and values read as one value each (`uncontracted`)."""
    total = left + right
    part = total - left
    return total, (left - (total - part)) + (right - part)


def uncontracted(*values):
    """`values` each read as the one value it is, never fused into the arithmetic that reads it: an error-free
    transform reads each operand more than once and holds only where every read sees the same rounded value,
    which a compiler's contraction of a multiply into an add (an FMA, XLA's CPU and GPU, Triton's) breaks by
    reading the exact product at one place and the rounded one at another (a two_product's product added in a
    compensated sum, its residual then counted twice). An optimization barrier, which no fusion crosses, on
    the value where it is made, so every reader sees the one rounded value (`two_sum`'s operands,
    `two_product`'s product); a caller whose plain float32 arithmetic leaned on the contractions a barrier
    splits around it takes its sum compensated (`summed_once`, row 7979). Inside a transform whose own
    arithmetic is marked to round alone (`rounding_alone`), the values as they are where the platform the
    program lowers for fences the mark (`MARKED_BACKENDS`, `jax.lax.platform_dependent`): no step reading
    them contracts, and no barrier splits the fusions around them (row 8378); the barrier on any other.

    In a kernel's body (`launches.interpreting` not None), whose platform is its launch's, the barrier, as
    the body's lowering takes it: a compiled Pallas TPU kernel's to its operands as they are, Mosaic keeping
    a kernel's multiply and add apart (the fork's lowering rule from post111, row 7954), Mosaic GPU's and
    Triton's by their own rules, an interpreted body's as XLA's programs do."""
    from dew.nn.kernels.launches import interpreting

    if interpreting() is not None or not _MARKING.get():
        return jax.lax.optimization_barrier(values)
    return jax.lax.platform_dependent(values, default=jax.lax.optimization_barrier, **_UNBARRIERED)


def _odd(dtype) -> bool:
    """Whether a float's significand holds an odd count of bits (`split_bits`' rounded cut)."""
    return (jnp.finfo(dtype).nmant + 1) % 2 == 1


@_protected
def two_product(left, right, *, bit_split: bool = False, subtract: bool = False,
                rounding: Callable[[jax.Array], jax.Array] | None = None) -> tuple[jax.Array, jax.Array]:
    """Dekker's rounded product and its residual, with a Veltkamp significand split.

    Guidance keeps the multiply split. `bit_split` truncates the significands
    instead (`split_bits`: a float's low half of its bits masked, fp32's 12,
    float64's 27, a 16-bit float's), with no splitter to overflow: the multiply
    split's `2^(p/2) + 1` overflowed past 2^115 in fp32 and 2^996 in float64
    (base reviews 9bc4326dd F1, 1fe546a10 F1); near underflow the splits differ.
    `subtract` keeps division's subtractive residual order and its signed zeros.
    Norm interpretation passes `optimization_barrier` as `rounding` to keep each
    product and residual step separate; the other callers allow fusion.
    With each arithmetic step rounded, the residual is exact where intermediates
    neither overflow nor underflow.
    """
    with rounding_alone():
        return _two_product(left, right, bit_split=bit_split, subtract=subtract, rounding=rounding)


def _two_product(left, right, *, bit_split: bool, subtract: bool,
                 rounding: Callable[[jax.Array], jax.Array] | None) -> tuple[jax.Array, jax.Array]:
    """`two_product`'s arithmetic, its steps' roundings each their own (`rounding_alone`, `uncontracted`)."""
    def rounded(value):
        return value if rounding is None else rounding(value)

    def split(value):
        if bit_split and isinstance(value, np.generic | np.ndarray):
            # A constant (a kernel's static factor) splits where it is known, on the host.
            constant = np.asarray(value, np.float32)
            unsigned, mask, _ = split_bits(np.float32)
            high = (constant.view(unsigned) & unsigned(mask)).view(np.float32)
            return jnp.asarray(high), jnp.asarray(constant - high)
        # The value read as one (`uncontracted`): its split reads it twice (its high part from its bits or a
        # scaled copy, then `value - high`), which a multiply that made it, contracted into the subtraction,
        # would read exactly at one and rounded at the other, a spurious low part.
        (value,) = uncontracted(value)
        if bit_split:
            unsigned, mask, low = split_bits(value.dtype)
            mask = np.array(mask, unsigned)
            bits = jax.lax.bitcast_convert_type(value, unsigned)
            if _odd(value.dtype):
                # Half the cut's last bit added to the magnitude's bits: the cut rounds to nearest.
                bits = bits + np.array(1 << (low - 1), unsigned)
            high = jax.lax.bitcast_convert_type(bits & mask, value.dtype)
        else:
            splitter = (1 << 27) + 1 if value.dtype == jnp.float64 else (1 << 12) + 1
            scaled = value * splitter
            high = scaled - (scaled - value)
        return high, rounded(value - high)

    scale = None
    if bit_split and _odd(jnp.result_type(left, right)):
        # A rounded cut of the top binade's last values carries past the largest float, and two cuts
        # rounded up carry a product in the top binade past it ((2 - 2^-52) 2^511 squared, the head inf):
        # such an operand, and the larger of such a product's, is halved first, its product and residual
        # doubled back after (each exact, a power of two; train 1018's review F1).
        top = 2.0 ** (jnp.finfo(jnp.result_type(left, right)).maxexp - 1)
        large = jnp.abs(left * right) >= top
        larger = jnp.abs(left) >= jnp.abs(right)
        halves = [(jnp.abs(left) >= top) | (large & larger), (jnp.abs(right) >= top) | (large & ~larger)]
        left, right = (jnp.where(half, operand * 0.5, operand)
                       for operand, half in zip((left, right), halves, strict=True))
        scale = jnp.where(halves[0], 2.0, 1.0) * jnp.where(halves[1], 2.0, 1.0)
    lh, ll = split(left)
    rh, rl = split(right)
    # The product read as one value (`uncontracted`) by the residual's steps and by its caller alike: a
    # compiler that took its multiply into the step reading it (`product - head`, a caller's `n - product`)
    # would read the exact product there and the rounded one elsewhere, the residual then the product's
    # rounding where a caller adds it to the rounded product.
    (product,) = uncontracted(rounded(left * right))
    head = rounded(lh * rh)
    lost = rounded(product - head if subtract else head - product)
    cross = ((ll, rh), (lh, rl)) if subtract else ((lh, rl), (ll, rh))
    for first, second in cross:
        term = rounded(first * second)
        lost = rounded(lost - term if subtract else lost + term)
    tail = rounded(ll * rl)
    residual = rounded(tail - lost if subtract else lost + tail)
    return (product, residual) if scale is None else (product * scale, residual * scale)


def _correctly_rounded_divide(numerator: jax.typing.ArrayLike,
                              denominator: jax.typing.ArrayLike) -> jax.Array:
    """The correctly rounded fp32 quotient, IEEE division's, on XLA's approximate
    dividers too: XLA:CPU rewrites a division by a broadcast into a multiplication
    by its reciprocal (openxla/xla#51148), and XLA:GPU's fp32 a / b is off IEEE on
    30 to 50% of quotients (DIVIDE-a100, DIVIDE-h100 receipts), where a quantizer's
    scales and a diffusion schedule's states must round as the reference's host or
    CUDA division does (compressed-tensors' NVFP4, the FP8 block cast, PyTorch's
    schedulers).

    Each operand's significand, its fraction bits under the exponent of one, in
    [1, 2), divides apart from its exponent field (`_rounded_significand`), so
    every intermediate stays normal whatever the operands' range; the fields'
    difference then scales the rounded significand by its exponent bits, exact
    for a normal result. A quotient below the normal range rounds once to the
    subnormal grid: the significand scaled onto that grid rounds to its nearest
    integer, ties to even, the exact residual settling a scaled significand that
    lands on a half, and the integer is the result's bits, which no flush to zero
    reaches. Past the largest float the quotient is infinite. A zero, infinite or
    NaN operand, or a subnormal one, which XLA:CPU reads as zero (an exponent field
    of 0 or 255), is the division's. Every step is a bit operation or a float32
    operation on normal values, about 140 operations a quotient where frexp's and
    nextafter's own special cases took 210 (row 6312). It is the lowering of a
    primitive of its own (`correctly_rounded_divide`), so no pass sees its
    error-free steps to reassociate them.
    """
    return _divided(numerator, denominator)


def lowered_divide(numerator: jax.Array, denominator: jax.Array, lowered: jax.Array) -> jax.Array:
    """`numerator / denominator * 2^-lowered` (`lowered` int32, at least 0, broadcast), rounded once to fp32
    however far below its least normal (`_divided`, `correctly_rounded_divide`'s): a quotient a CPU flushes
    (row 4023), its numerator raised by `lowered` from its bits by its caller (`raised_exactly`), so both
    operands are normal; a subnormal operand is the division's (row 6302's finish and an initial state's
    gradient)."""
    return _divided(numerator, denominator, lowered)


def _divided(numerator: jax.typing.ArrayLike, denominator: jax.typing.ArrayLike,
             lowered: jax.typing.ArrayLike = 0) -> jax.Array:
    """`correctly_rounded_divide`'s quotient times `2^-lowered`, rounded once
    (`_correctly_rounded_divide`); a special operand's division lowered in integers (`lowered_rounded`)."""
    numerator, denominator = jnp.broadcast_arrays(jnp.asarray(numerator, jnp.float32),
                                                  jnp.asarray(denominator, jnp.float32))
    plain = numerator / denominator
    if not (isinstance(lowered, int) and lowered == 0):
        plain = lowered_rounded(plain, jnp.broadcast_to(jnp.asarray(lowered, jnp.int32), plain.shape))
    (_, top_sign, top_field, top_fraction), (_, bottom_sign, bottom_field, bottom_fraction) = (
        raw_fields(value) for value in (numerator, denominator))
    top_field, bottom_field = top_field.astype(jnp.int32), bottom_field.astype(jnp.int32)
    special = (top_field == 0) | (top_field == 255) | (bottom_field == 0) | (bottom_field == 255)
    negative = top_sign != bottom_sign
    # A normal operand's significand, exact in [1, 2); any bits give one in that range, so a special
    # lane's steps stay normal too, and its result is the division's.
    top, bottom = (jax.lax.bitcast_convert_type(fraction | np.uint32(0x3F800000), jnp.float32)
                   for fraction in (top_fraction, bottom_fraction))
    significand, residual = _rounded_significand(top, bottom)
    exponent = top_field - bottom_field - lowered
    # The rounded significand's fields (`raw_fields`), a positive normal's: its biased exponent once scaled,
    # 1 to 254 a normal float, set back in its bits.
    bits, _, significand_field, _ = raw_fields(significand)
    field = significand_field.astype(jnp.int32) + exponent
    normal = jax.lax.bitcast_convert_type(bits + jax.lax.bitcast_convert_type(exponent << 23, jnp.uint32),
                                          jnp.float32)
    # Below the normal range: the significand on the grid of 2^-149, a value under 2^23 whose
    # integer part and half are exact in fp32; within the float32 range scaled by 2^shift.
    shift = jnp.clip(exponent + 149, -126, 127)
    scaled = significand * exact_power2(shift, jnp.float32)
    whole = jnp.floor(scaled)
    fraction = scaled - whole
    odd = (whole.astype(jnp.int32) & 1) == 1
    half_up = (residual > 0) | ((residual == 0) & odd)
    up = (fraction > 0.5) | ((fraction == 0.5) & half_up)
    subnormal = jax.lax.bitcast_convert_type((whole.astype(jnp.int32) + up.astype(jnp.int32)),
                                             jnp.float32)
    magnitude = jnp.where(field >= 255, jnp.inf, jnp.where(field >= 1, normal, subnormal))
    signed = jnp.where(negative, -magnitude, magnitude)
    return jnp.where(special, plain, signed)


@_protected
def _rounded_significand(numerator: jax.Array, denominator: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The correctly rounded fp32 quotient of two significands in [1, 2), and the sign of the exact
    quotient's distance from it (+1 above, -1 below, 0 exact).

    A Dekker TwoProduct gives the exact residual of the approximate
    quotient, split into two fp32 values. One residual correction puts the
    quotient within one ulp; comparing its exact residual with each
    neighbour's half-spacing times the divisor chooses nearest-even. Its
    neighbours are its bits' (a positive normal float's next and previous
    floats), and that threshold product is exact because the half-spacing
    is a power of two, every intermediate normal at these magnitudes. The
    rounded quotient's residual is the corrected one's less the move, one
    spacing times the divisor, exact, and within a factor of two of the
    corrected residual where the quotient moves (Sterbenz), so its sign needs
    no further product. Each residual is an error-free step, so the whole is
    one protected span."""
    quotient = numerator / denominator
    product, error = two_product(quotient, denominator, bit_split=True, subtract=True)
    residual = (numerator - product) - error
    quotient = quotient + residual / denominator
    product, error = two_product(quotient, denominator, bit_split=True, subtract=True)
    high, low = two_sum(numerator - product, -error)
    bits = jax.lax.bitcast_convert_type(quotient, jnp.uint32)
    above, below = (jax.lax.bitcast_convert_type(bits + step, jnp.float32) for step in (1, -1))
    upper = (above - quotient) * 0.5 * denominator
    lower = (quotient - below) * 0.5 * denominator
    odd = (bits & 1) != 0
    up = (high > upper) | ((high == upper) & ((low > 0) | ((low == 0) & odd)))
    down = (high < -lower) | ((high == -lower) & ((low < 0) | ((low == 0) & odd)))
    rounded = jnp.where(up, above, jnp.where(down, below, quotient))
    # The exact quotient's side of the rounded one: the rounded quotient's exact residual, the numerator
    # less its product with the divisor, which is zero only where the two are equal.
    side = jnp.where(up, (high - 2 * upper) + low, jnp.where(down, (high + 2 * lower) + low, high))
    return rounded, jnp.sign(side)


correctly_rounded_divide_p = core.Primitive("correctly_rounded_divide")
"""The correctly rounded fp32 division of two arrays (`correctly_rounded_divide`), an operation of its own so
that its derivatives divide as it does. Its operands broadcast as a binary operation's do: the function
gives it two of one shape, and a batching of one of them leaves the other a scalar or of unit axes
(`batching.defbroadcasting`). Its rounding is a selection among neighbours, so its
tangent is the division's, each quotient in it correctly rounded: `g / d` in the numerator and
`-g * (n / d) / d` in the denominator, the order PyTorch's backward divides in, and its transpose in the
numerator `ct / d`. As `jnp.divide`'s, they divided as XLA does: by the divisor's rounded reciprocal where it
is a constant or a broadcast on a CPU (openxla/xla#51148) and off IEEE on 30 to 50% of quotients on a GPU, so
a quantizer's scale or a rate's gradient left the reference's by an ulp on a third of its entries (row
5277). It is entrywise (`Primitive.elementwise`), which the analyses of a program's entries read."""
correctly_rounded_divide_p.def_impl(_correctly_rounded_divide)
# Its result is typed as JAX types its own division's: the operands' broadcast shape, their dtype, and
# their sharding and varying axes, which a mesh's Explicit axis carries in a value's type.
correctly_rounded_divide_p.def_effectful_abstract_eval(jax.lax.div_p.abstract_eval)
mlir.register_lowering(correctly_rounded_divide_p,
                       mlir.lower_fun(_correctly_rounded_divide, multiple_results=False))


@functools.cache
def triton_lowerings() -> None:
    """Register `correctly_rounded_divide_p`'s Pallas Triton lowering lazily
    (`dew.nn.kernels.launches.pallas_call`, row 9794) so importing `precision` loads no Pallas Triton
    machinery."""
    from jax._src.pallas.triton import lowering as triton_lowering
    from jax.experimental.pallas import triton as pltriton

    def divided(numerator, denominator):
        numerator, denominator = jnp.broadcast_arrays(numerator, denominator)
        [quotient] = pltriton.elementwise_inline_asm(
            "div.rn.f32 $0, $1, $2;", args=[numerator, denominator], constraints="=r,r,r", pack=1,
            result_shape_dtypes=[jax.ShapeDtypeStruct(numerator.shape, numerator.dtype)])
        return quotient

    triton_lowering.register_lowering(correctly_rounded_divide_p)(
        triton_lowering.lower_fun(divided, multiple_results=False))


batching.defbroadcasting(correctly_rounded_divide_p)
correctly_rounded_divide_p.elementwise = True
# It rounds the quotient to float32 whatever its operands' dtype: at a wider width it is the quotient.
UNROUNDED[correctly_rounded_divide_p] = jnp.divide


@contracted("test_the_precision_seams_hold_every_order")
def correctly_rounded_divide(numerator: jax.typing.ArrayLike, denominator: jax.typing.ArrayLike) -> jax.Array:
    """The correctly rounded fp32 quotient `numerator / denominator` (`_correctly_rounded_divide`), IEEE
    division's on every backend, differentiable at every order, each of its derivatives' divisions
    correctly rounded too (`correctly_rounded_divide_p`)."""
    numerator, denominator = jnp.broadcast_arrays(jnp.asarray(numerator, jnp.float32),
                                                  jnp.asarray(denominator, jnp.float32))
    return correctly_rounded_divide_p.bind(numerator, denominator)


def _divide_transpose(cotangent, numerator, denominator):
    """The division's transpose in its numerator, the one operand it is linear in: the cotangent over the
    denominator."""
    if not ad.is_undefined_primal(numerator) or ad.is_undefined_primal(denominator):
        raise TypeError("a correctly rounded division is linear in its numerator alone")
    if type(cotangent) is ad.Zero:
        return [ad.Zero(numerator.aval), None]
    quotient = correctly_rounded_divide_p.bind(cotangent, denominator)
    # A numerator the division broadcast (a scalar's, or along a unit axis) takes the sum of the quotients
    # it was broadcast to.
    shape = numerator.aval.shape
    leading = quotient.ndim - len(shape)
    widened = (*range(leading), *(leading + axis for axis, size in enumerate(shape)
                                  if size != quotient.shape[leading + axis]))
    return [jnp.reshape(jnp.sum(quotient, axis=widened), shape) if widened else quotient, None]


ad.defjvp2(correctly_rounded_divide_p,
           lambda tangent, quotient, numerator, denominator: correctly_rounded_divide(tangent, denominator),
           lambda tangent, quotient, numerator, denominator: -tangent * correctly_rounded_divide(
               quotient, denominator))
ad.primitive_transposes[correctly_rounded_divide_p] = _divide_transpose


def divided(numerator, rate):
    """`numerator / rate` as a source written in PyTorch divides a state by a rate, or an array by a
    constant: IEEE division, whatever the backend's own division does (XLA takes a division by a
    constant as a product with its rounded reciprocal, on a CPU and an A100 alike, row 1402). A
    float32, bfloat16 or float16 quotient is `correctly_rounded_divide`'s of the operands widened to
    float32, rounded to its dtype, as PyTorch's float32 opmath gives it, under any compiler flags. A
    float64 one is the plain division."""
    numerator, rate = jnp.asarray(numerator), jnp.asarray(rate)
    dtype = jnp.result_type(numerator, rate)
    if not jnp.issubdtype(dtype, jnp.floating):
        return numerator / rate
    if dtype == jnp.float64:
        return numerator / rate
    return correctly_rounded_divide(numerator.astype(jnp.float32), rate.astype(jnp.float32)).astype(dtype)


def vanishing(dtype, base2: bool = False) -> float:
    """The exponent at and below which a decay (`exp`, or `exp2` with `base2`) is zero in `dtype`, whatever
    exponents join it (each at most zero): below the base's logarithm of its least subnormal (`exp(-105)` and
    `exp2(-151)` are zero in fp32). A prefix summing one with the others loses them (two of -2**127 are
    -inf; one of -2**70 loses a later -1 to its rounding), so a chunk's decays restart past it
    (`dew.algebra.scan`'s decays)."""
    least = math.log(float(jnp.finfo(dtype).smallest_subnormal))
    return math.floor(least / (math.log(2.0) if base2 else 1.0)) - 1


def twice_scaled(product: jax.Array, first: jax.Array, second: jax.Array) -> jax.Array:
    """`product` times two scales, each entry in the order whose intermediate is a normal float32: an
    input's block scale near float32's largest (2**113 for inputs near 2**122) overflows the codes' product
    the kernel's scale (2**-9) brings back in range, and one near the least normal underflows it the other
    way (ledger 2715), as two tensor scales near 2**-76 do multiplied first (an NVFP4 product of operands
    near 2**-65), so no one order holds both ends; the result overflows only where the product does. Its
    tests are comparisons alone (finite: a magnitude below infinity), so a kernel's epilogue runs it as XLA
    does: Mosaic GPU lowers no `is_finite` (`gemm.nvfp4_block_finished`, row 12721)."""
    tiny = jnp.finfo(product.dtype).tiny
    once = product * first
    magnitude = jnp.abs(once)
    # Not `jnp.isfinite`: Mosaic GPU lowers no is_finite, and the fused NVFP4 epilogue runs this in a kernel.
    kept = (magnitude < np.inf) & ((magnitude >= tiny) | (product == 0))
    return jnp.where(kept, once, product * second) * jnp.where(kept, second, first)  # no cotangent unchosen


@_protected
def compensated_add(left, right):
    """Add two double-float values, `(hi, lo)` pairs whose sum is the value:
    `hi` is the sum rounded and `lo` what the rounding dropped (the his'
    TwoSum, the los added to its remainder, then the pair's own TwoSum), so a
    running sum keeps fp32's precision squared and its `hi` is the running
    value rounded once, where a pair left unrenormalised keeps the plain
    running sum in `hi`, every addition's rounding in it.
    XLA reassociates no float arithmetic unless fast math is on; with it,
    `lo` cancels to zero and the sum is the plain one."""
    (left_hi, left_lo), (right_hi, right_lo) = left, right
    with rounding_alone():
        return renormalized_add((*uncontracted(left_hi), left_lo), (*uncontracted(right_hi), right_lo))


@_protected
def renormalized_add(left, right):
    """`compensated_add`'s arithmetic over pairs whose his no multiply makes in the same fusion (each read as
    one value, `uncontracted`, or a compensated reduction's or scan's accumulator, made by additions alone):
    `renormalized_sum`, as one span the algebra pass keeps (`_protected`). The combine of a compensated
    reduction (`summed_pairs`) and a compensated scan (`dew.algebra.scan.compensated`)."""
    return renormalized_sum(left, right)


def renormalized_sum[V: _Summed](left: tuple[V, V], right: tuple[V, V]) -> tuple[V, V]:
    """Two `(hi, lo)` pairs' renormalized sum, over any values `+` and `-` add and subtract (arrays, or a
    Mosaic GPU kernel's words, `templates.native.Word`, `templates.scan.compensated_columns`'): the his'
    TwoSum, then the pair's own over the dropped part and the los in that order, whose operands are
    additions' (`_two_sum`, no barrier of its own: one in a reduction's computation splits the fusions around
    the program, row 7953). A lo is read once, so a multiply contracted into its addition reads it once too.
    Its caller supplies his each rounded alone, no multiply contracted into them (a native kernel's words, or
    pairs made by additions alone); a JAX program whose his a multiply may make takes them as one span the
    algebra pass keeps, through `renormalized_add`, or read as one value each, `compensated_add`."""
    (left_hi, left_lo), (right_hi, right_lo) = left, right
    total, dropped = _two_sum(left_hi, right_hi)
    return _two_sum(total, dropped + left_lo + right_lo)


@_protected
def compensated_difference(later, earlier):
    """`later - earlier` of two double-float values, rounded once: the halves'
    differences summed, so the difference of two near values (a range of a
    running sum) keeps its own precision, where the his' difference alone
    carries their whole magnitude's rounding: a one-token range of a decay's
    running sum near -78 off by 4e-6. Broadcasts as the halves do."""
    return (later[0] - earlier[0]) + (later[1] - earlier[1])


def compensated_sum(values: jax.Array, axes: tuple[int, ...]) -> jax.Array:
    """The sum of `values` over `axes`, rounded once: one XLA reduction whose accumulator is a
    double-float `(hi, lo)` pair combined by `compensated_add`'s arithmetic, its plain sum carried beside
    it for an infinity's fallback (`summed_total`), so it partitions as any reduction does, reads its
    input once, and what the roundings dropped is added back at the end.

    One plain reduction over a loss's every axis rounds as it goes; on 6144 JEPA squares it left the
    sum one float32 step above the correctly rounded one, and the loss a float32 ulp from float64,
    where numpy's float32 mean of the same squares was correctly rounded. A narrower float adds in
    float32 and returns in its own dtype, as `jnp.sum` does; an integer sum is `jnp.sum`'s. The
    derivative is `jnp.sum`'s."""
    if not jnp.issubdtype(values.dtype, jnp.floating):
        return jnp.sum(values, axes)
    return _compensated_sum(values, tuple(axes))


def _compensated(values: jax.Array, axes: tuple[int, ...]) -> jax.Array:
    """The sum of `values` over `axes` its compensated pair rounds once, its tangent the plain sum's: a
    protected span (`_compensated_sum`, `dew.nn.derivatives.declared_tangent`), whose compensation the
    algebra pass must not cancel."""
    wide = values.astype(at_least_fp32(values.dtype))
    return summed_total((wide, jnp.zeros_like(wide)), axes).astype(values.dtype)


def summed_pairs(pair: tuple[jax.Array, jax.Array], axes: Sequence[int]) -> tuple[jax.Array, jax.Array]:
    """Double-float `(hi, lo)` pairs summed over `axes` in one XLA reduction, `(hi, lo)`: each `hi` read as
    one value where it is made (`uncontracted`), or the reduction's additions marked to round alone
    (`rounding_alone`), and the pairs combined by `compensated_add`'s arithmetic with no barrier of its own
    (`renormalized_add`: one in a reduction's computation splits the fusions around the program, row 7953).
    The compensated pair reduction alone: a guidance projection's dot products' (row 8498); a compensated
    sum's (`compensated_sum`) and a kernel's programs' parts' (`norm_swiglu`) carry their plain sum beside it
    in the same reduction (`summed_total`). An axis of one element is no sum and
    is taken out before the reduction: XLA:GPU's simplifier (post117) rewrote a pair reduction over such an
    axis into one bitcast of a single array's shape and refused its own program (a RET_CHECK of
    `ShapeUtil::Compatible`, row 10674)."""
    return _summed(pair, axes, plain=False)


def summed_total(pair: tuple[jax.Array, jax.Array], axes: Sequence[int]) -> jax.Array:
    """Double-float `(hi, lo)` pairs summed over `axes` and read as every compensated sum is
    (`compensated_total`): the pairs (`summed_pairs`) and the plain sum of their `hi`, the total's inf or
    NaN where it reaches one, carried in one XLA reduction, the input read once. A plain sum beside the
    pairs' reduction read it again: XLA fused it into its consumer, a second pass over Nemotron-H 30B's
    SSD scores' gradient (`f32[4096, 4096]`, 69 us of an A100's training step, M-JS-204-a100)."""
    return compensated_total(*_summed(pair, axes, plain=True))


def _summed(pair: tuple[jax.Array, jax.Array], axes: Sequence[int], *, plain: bool) -> tuple:
    """`summed_pairs`' `(hi, lo)`, and where `plain` the plain sum of the pairs' `hi` beside them in the
    same reduction, `((hi, lo), plain)`: the pairs combined by `renormalized_add`, the plain sums
    added."""
    hi, lo = pair
    if hi.shape != lo.shape or hi.dtype != lo.dtype:
        raise ValueError(f"a pair's limbs share one shape and dtype, not {hi.dtype}{list(hi.shape)} and "
                         f"{lo.dtype}{list(lo.shape)}")
    # An axis out of range or named twice refused as numpy refuses it (an AxisError, a ValueError).
    axes = normalize_axis_tuple(tuple(axes), hi.ndim)
    ones = tuple(axis for axis in axes if hi.shape[axis] == 1)
    if ones:
        hi, lo = jnp.squeeze(hi, ones), jnp.squeeze(lo, ones)
    kept = tuple(axis - sum(one < axis for one in ones) for axis in axes if axis not in ones)
    with rounding_alone():
        (hi,) = uncontracted(hi)
        limbs = (hi, lo, hi) if plain else (hi, lo)
        combine = _paired_and_plain if plain else renormalized_add
        if kept:
            summed = jax.lax.reduce(limbs, (jnp.zeros((), hi.dtype),) * len(limbs), combine, kept)
        else:
            # Each accumulator combined once with the zero, as a reduction over its one element is.
            summed = combine(tuple(jnp.zeros_like(limb) for limb in limbs), limbs)
    return ((summed[0], summed[1]), summed[2]) if plain else (summed[0], summed[1])


def _paired_and_plain(left, right):
    """`summed_total`'s combine: two `(hi, lo, plain)` accumulators' pairs by `renormalized_add`, their plain
    sums added."""
    return (*renormalized_add(left[:2], right[:2]), left[2] + right[2])


COMPENSATED_COMBINES: dict[Callable, tuple[str, ...]] = {
    compensated_add: ("pair", "pair"), renormalized_add: ("pair", "pair"),
    _paired_and_plain: ("pair", "pair", "plain")}
"""Each combine Dew's compensated sums hand `jax.lax.reduce`, as a reduction equation holds it (its
`computation`), with its results' parts: a double-float pair's halves (`compensated_add`, and
`renormalized_add`, `summed_pairs`'), and a one-pass sum's pair and the plain sum carried beside it
(`_paired_and_plain`, `summed_total`'s): the one list the gate's rounding bound and its check 2 read them by
(row 11880)."""


def compensates_loops() -> bool:
    """Whether a running sum of float32 products carried across a loop keeps what each addition
    drops (`compensated_add`), so the loop's sum rounds once as one product's does: where a CPU sums
    each product in float64 (`accumulated`), in a process running Dew's wide CPU arithmetic
    (`WIDE_CPU_ARITHMETIC`). A carry's shape is fixed when its loop traces, so this
    reads the platform a call's code is chosen for (`dew.nn.kernels.generation.backend`), as a CPU's
    attention routing does; an accelerator's running sum is the plain one, as its products' sums are
    float32's."""
    from dew.nn.kernels.generation import backend

    return (wide_cpu_arithmetic() and backend() == "cpu"
            and jax.config.jax_default_matmul_precision != FP32_SUMS and not _intercepted())


def running_sum(shape: tuple[int, ...], dtype: DTypeLike) -> jax.Array:
    """An empty running sum of `shape` (`added_to`, `running_total`): `[3, *shape]`, a double-float
    pair and the plain sum beside it, where loops compensate (`compensates_loops`), else `[1, *shape]`,
    the plain sum alone. Its leading axis is the form's, so a loop slices and updates the sum's own
    axes one past theirs."""
    return paired_sum(shape, dtype) if compensates_loops() else jnp.zeros((1, *shape), dtype)


def paired_sum(shape: tuple[int, ...], dtype: DTypeLike) -> jax.Array:
    """An empty running sum of `shape` that is a double-float pair on every backend, the plain sum
    beside it (`added_to`, `running_total`): for partials of a product Dew splits where its reference
    takes the product whole, so the sum's error is the partials' own, whether XLA takes the addition
    into a GEMM's epilogue or cuBLAS splits a product's sum by its shape (row 1725)."""
    return jnp.zeros((3, *shape), dtype)


type Running = jax.Array | tuple[jax.Array, ...]
"""A running sum (`running_sum`), stacked along its leading axis or held as that axis's parts."""


@overload
def added_to(running: jax.Array, term: jax.Array) -> jax.Array: ...


@overload
def added_to(running: tuple[jax.Array, ...], term: jax.Array) -> tuple[jax.Array, ...]: ...


def added_to(running: Running, term: jax.Array) -> Running:
    """`running` (`running_sum`) with `term` added: through `compensated_add` where it is a pair. A running
    sum held as its leading axis's parts (a tuple) stays one, as a kernel whose layouts hold no unit axis
    carries it."""
    held = running if isinstance(running, tuple) else tuple(running)
    term = term.astype(held[0].dtype)
    if len(held) == 1:
        added = (held[0] + term,)
    else:
        added = (*compensated_add((held[0], held[1]), (term, jnp.zeros_like(term))), held[2] + term)
    return added if isinstance(running, tuple) else jnp.stack(added)


def running_total(running: Running) -> jax.Array:
    """A running sum's value (`running_sum`, or its leading axis's parts): a pair's rounded once
    (`compensated_total`), else the sum."""
    held = running if isinstance(running, tuple) else tuple(running)
    if len(held) == 1:
        return held[0]
    return compensated_total((held[0], held[1]), held[2])


def compensated_total(pair: tuple[jax.Array, jax.Array], plain: jax.Array) -> jax.Array:
    """A compensated sum's value, its `(hi, lo)` pair rounded once, or the `plain` sum of the same
    values where that reaches an infinity: +-inf or NaN as `jnp.sum`'s is, where every residual past
    the infinity is inf - inf. `compensated_add` itself stays arithmetic alone, so a tangent's scan
    through it stays linear; a running sum that can overflow keeps its plain sum beside the pair
    and reads its value here."""
    return jnp.where(jnp.isfinite(plain), pair[0] + pair[1], plain)


def _compensated_sum_jvp(axes: tuple[int, ...], primals: tuple[jax.Array],
                         tangents: tuple[jax.Array]) -> tuple[jax.Array, jax.Array]:
    return _compensated_sum(primals[0], axes), jnp.sum(tangents[0], axes)


_compensated_sum = contracted("test_the_precision_seams_hold_every_order")(
    declared_tangent(Protected(), _compensated, _compensated_sum_jvp, nondiff_argnums=(1,)))


def summed_once(values: Sequence[jax.Array], products: Sequence[tuple[jax.Array, jax.Array]] = ()
                ) -> jax.Array:
    """The sum of `values` and of `products` (each a `(left, right)` pair, broadcasting as `*` does), every
    product's rounding (`two_product`) and every addition's (`two_sum`) carried apart in one plain sum and
    added back once at the end (Ogita, Rump and Oishi's Dot2), its value read as a compensated sum's is
    (`compensated_total`): the sum as one rounding of its twice-precise value leaves it, whatever a compiler
    fuses or keeps apart, so no contraction of a multiply into an add (an FMA) it takes or leaves rounds it
    more (row 7953: a barrier in a step's arithmetic split its fusion and lost the contractions its plain
    float32 sum leaned on, row 7979), and a sum reaching an infinity is the plain sum's, as `jnp.sum`'s is
    (row 8453). Its residuals are carried apart, not renormalized into the running pair as `compensated_add`
    renormalizes it: added back as they come, `M + 2^102 + 2^102 - M` (`M` float32's largest) rounded the
    running value past `M` to an infinity, and a residual a later one cancels was rounded away first. Its
    tangent is the plain sum's."""
    return _summed_once(tuple(values), tuple(products))


def _summed_once_value(values, products):
    total, carried = values[0], jnp.zeros_like(values[0])
    for left, right in products:
        product, lost = two_product(left, right, bit_split=True)
        total, dropped = two_sum(total, product)
        carried = carried + (lost + dropped)
    for value in values[1:]:
        total, dropped = two_sum(total, value)
        carried = carried + dropped
    # Each sum's rounded half the plain running sum, which reads an infinity as `jnp.sum` does.
    return compensated_total((total, carried), total)


def _summed_once_tangent(primals, tangents):
    (values, products), (value_tangents, product_tangents) = primals, tangents
    terms = [*value_tangents, *(left_tangent * right + left * right_tangent for (left, right), (
        left_tangent, right_tangent) in zip(products, product_tangents, strict=True))]
    return _summed_once_value(values, products), sum(terms[1:], terms[0])


_summed_once = contracted("test_the_precision_seams_hold_every_order")(
    declared_tangent(Protected(), _summed_once_value, _summed_once_tangent))

def _rounded(x: jax.Array, dtype: Dtype) -> jax.Array:
    """The value `x` takes in `dtype`, held in `x`'s own dtype (`rounded_operand`'s value).

    The rounding is the cast itself, subnormals included. The barrier keeps
    `xla_allow_excess_precision` from deleting the round trip under jit, and
    stays where `dtype` is `x`'s own so XLA cannot carry an operand wider into
    the fusion that reads it. Formats are compared, not widths: float16 and
    bfloat16 each lose bits in the other.
    """
    if jnp.dtype(dtype) == jnp.dtype(x.dtype):
        return jax.lax.optimization_barrier(x)
    return jax.lax.optimization_barrier(x.astype(dtype)).astype(x.dtype)


rounded_operand = contracted("test_the_precision_seams_hold_every_order")(
    surrogate(_rounded, lambda dtype, primals, tangents, value: tangents[0],
              nondiff_argnums=(1,)))
"""The value `x` takes in `dtype`, held in `x`'s own dtype (`_rounded`), with a straight-through
derivative (`dew.nn.derivatives.surrogate`): its tangent is `x`'s, never rounded a second time."""


def rounds_to_bf16(dtype: Dtype | None, precision: PrecisionLike = None) -> bool:
    """Whether an fp32-result product under this compute dtype and precision
    multiplies bf16 operands (`operand_precision`)."""
    return operand_precision(dtype, precision) is jax.lax.DotAlgorithmPreset.BF16_BF16_F32


def bf16_logits(logits: jax.Array) -> jax.Array:
    """fp32 `logits` rounded to bf16 and held in fp32, as `rounded_to` rounds
    them, written as bf16 under CUDA.

    There the cast joins the product's epilogue and each reader widens in its
    own fusion, behind a barrier `xla_allow_excess_precision` cannot see
    through. `reduce_precision` instead stayed a kernel of its own after the
    Triton head, writing and reading the fp32 logits in full when a serving
    step's draw read them three times: 0.16 ms of a step at 128 rows of
    Qwen3-0.6B on an RTX 4080, the same bits (docs/performance.md)."""
    return jax.lax.platform_dependent(
        logits, cuda=lambda x: jax.lax.optimization_barrier(x.astype(jnp.bfloat16)).astype(x.dtype),
        default=lambda x: rounded_to(x, jnp.bfloat16))


def head_dot_general(dtype: Dtype | None, precision: PrecisionLike = None):
    """A flax layer's `dot_general` for a vocabulary head: a result at
    least fp32 whose operands are the compute dtype's values, in both
    directions.

    `dtype` is the model's compute dtype, not the layer's: the layer
    promotes the states to at least fp32. Under bf16 compute at the default
    precision both operands multiply as bf16 and the logits are bf16 values
    (`head_product`); otherwise the product is the layer's own, in the
    operands' dtype, at the precision stated, IEEE float32 where none is (`exact_float32`): its
    logits feed the loss's logsumexp and the sampler, logits tens wide that TF32's part in a thousand
    moves.
    """
    resolved = operand_precision(dtype, precision)
    rounds = rounds_to_bf16(dtype, precision)

    def dot_general(lhs, rhs, dimension_numbers, precision=None,
                    preferred_element_type=None):
        del precision, preferred_element_type  # this head's policy, not the layer's
        if rounds:
            # The bf16 product, exact in fp32 (`bf16_product`).
            return bf16_logits(bf16_dot_general(lhs, rhs, dimension_numbers))
        return accumulated(functools.partial(
            jax.lax.dot_general, dimension_numbers=dimension_numbers,
            precision=exact_float32(resolved)),
            lhs, rhs, preferred_element_type=at_least_fp32(lhs.dtype))

    return dot_general


def scaled_logits(logits: jax.Array, scale: jax.Array | None, bias: jax.Array | None) -> jax.Array:
    """A vocabulary head's per-column terms after its product's rounding, in fp32 or wider: each
    column times its `scale` (nGPT's per-token logit gain), then plus its `bias`."""
    if scale is not None:
        logits = logits * jnp.asarray(scale, logits.dtype)
    return logits if bias is None else logits + jnp.asarray(bias, logits.dtype)


def capped_logits(logits: jax.Array, softcap, temperature: float = 1.0) -> jax.Array:
    """Softcapped logits over a sampling temperature, which an engine applies
    to what the model's head returns."""
    if softcap is not None:
        cap = jnp.asarray(softcap, logits.dtype)
        logits = cap * jnp.tanh(logits / cap)
    return logits / temperature


def head_product(subscripts: str, hidden: jax.Array, head: jax.Array,
                 precision: PrecisionLike = None, *, compute: DTypeLike | None = None) -> jax.Array:
    """`jnp.einsum(subscripts, hidden, head)` as a vocabulary head computes
    it, with accumulation and a result at least fp32.

    The product follows the compute dtype, as torch autocast and MaxText
    (`logits_dot_in_fp32=False`) run it: bf16 states at the default precision
    multiply the head as bf16 and the logits round to bf16, held in fp32, with
    the cotangent rounded once alike. Anything else multiplies fp32 states by
    the head as stored. The rounding cost nothing measurable in 2000-step
    wikitext-103 runs (validation loss within 4.5e-4 and 1.6e-3 of fp32 logits,
    below seed noise) and saves 13% of a 3-layer decoder's step. It does move
    layout parity (a 4 x RTX 3090 bf16 MoE read 5758 of its bound against 0.41
    in fp32), so a run comparing layouts or resuming onto another mesh sets
    `matmul_precision` "highest", whose head is fp32. `compute`, the model's
    compute dtype where the states arrive in another, decides in their place."""
    if rounds_to_bf16(hidden.dtype if compute is None else compute, precision):
        return bf16_product(subscripts, hidden, head)
    wide = at_least_fp32(hidden.dtype)
    # The logits feed the loss's logsumexp and the sampler (`exact_float32`, as `head_dot_general`'s).
    return float32_einsum(subscripts, hidden.astype(wide), head, precision=exact_float32(precision),
                          preferred_element_type=wide)


def bf16_product(subscripts: str, hidden: jax.Array, head: jax.Array) -> jax.Array:
    """`jnp.einsum(subscripts, hidden, head)` multiplied as bf16 and its result rounded to bf16, held
    in fp32: the one product a bf16 head computes (`head_product`) and a tiled head computes over
    each tile, so the two round each logit alike."""
    return bf16_logits(bf16_einsum(subscripts, hidden, head))


def bf16_dot_general(lhs: jax.Array, rhs: jax.Array, dimension_numbers) -> jax.Array:
    """`jax.lax.dot_general(lhs, rhs, dimension_numbers)` multiplied as XLA's `BF16_BF16_F32` dot
    algorithm states the arithmetic: the operands rounded to bf16 (to nearest even), each product
    exact and summed in fp32 to an fp32 result. Backward, the cotangent rounds to bf16 once and each
    operand's gradient is the same product of it and the other operand, held in the operand's dtype;
    forward mode takes the product's own tangent (`linear_tangent`). The one product Dew runs at
    that algorithm: the dot family's (`float32_dot_general`, `float32_einsum`) and a bf16 head's.

    Its operands are bf16 arrays multiplied at the default precision, whatever a scope configures
    (an F16 preset would narrow a bf16 operand to fp16), not wider ones under the algorithm; a CPU,
    which runs no bf16 dot at some layouts, multiplies their values in fp32, exactly. On an
    A100, XLA's dot under the algorithm with a vector side (one row or one column), or below its
    GEMM rewrite threshold (6 x 3 x 8), did not round fp32 operands to nearest even, and products
    of exact bf16 values came out off round-to-nearest-even at their ties; at a vector side bf16
    operands under the algorithm missed too, at the default precision and at the highest. With bf16
    operands and no algorithm every shape was exact (node receipts SF-a100-tie-probe3 and 4). A
    one-token tile of the chunked head and a one-row decode are such dots."""
    from .kernels.generation import linear_tangent

    (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
    numbers = ((tuple(lhs_contracting), tuple(rhs_contracting)), (tuple(lhs_batch), tuple(rhs_batch)))

    def typed(lhs, rhs, numbers):
        from .sharding import contracted_alike

        # A contraction Explicit axes split on both operands, forward and in either transpose, lands its sum
        # where JAX asks to be told (`contracted_alike`, the confirm review of row 12225).
        lhs, rhs, placed = contracted_alike(lhs.astype(jnp.bfloat16), rhs.astype(jnp.bfloat16), numbers)

        def native(lhs, rhs):
            return jax.lax.dot_general(lhs, rhs, numbers, precision=jax.lax.Precision.DEFAULT,
                                       preferred_element_type=jnp.float32, out_sharding=placed)

        def widened(lhs, rhs):
            # XLA:CPU has no bf16 x bf16 = fp32 dot at every layout ("Unsupported element type for
            # DotThunk"); bf16 values multiply exactly in fp32, so the fp32 dot is the same arithmetic.
            return jax.lax.dot_general(lhs.astype(jnp.float32), rhs.astype(jnp.float32), numbers,
                                       precision=jax.lax.Precision.HIGHEST,
                                       preferred_element_type=jnp.float32, out_sharding=placed)

        return jax.lax.platform_dependent(lhs, rhs, cpu=widened, default=native)

    def pulled(saved, operands, cotangent):
        """Each operand's cotangent: the product's transpose to it over the cotangent rounded to bf16. No
        barrier: which operand is a parameter is its caller's to know, a projection's own reverse mode over
        it tying its kernel's gradient (`projections.dot_general`). A vocabulary head's (`head_product`,
        `head_dot_general`) takes none: XLA forms its table's gradient beside the states' with nothing put
        off (a stack under a bf16 head, compile-only H100: its temporaries trained are its held ones plus the
        gradient, 272 against 256 + 16 MiB), and the rule's portable reverse mode (`portable_formed`) would
        recompute the logits through their rounding's barrier, 768 MiB (train 911 F3)."""
        from .sharding import explicit_spec, holds_sum

        del saved
        rounded = cotangent.astype(jnp.bfloat16)

        def transposed(side: int) -> jax.Array:
            if not holds_sum(operands[side]):
                return _transposed(rounded, operands, numbers, side, typed).astype(operands[side].dtype)
            # An operand holding a sum still to take has a cotangent of its own type (unreduced over the
            # sum's axes), which JAX's own transpose of the product forms (the confirm review).
            def along(operand):
                lhs, rhs = (operand, operands[1]) if side == 0 else (operands[0], operand)
                return typed(lhs, rhs, numbers)

            return jax.vjp(along, operands[side])[1](rounded.astype(cotangent.dtype))[0]

        pulled = [transposed(side) for side in (0, 1)]
        # Each operand's cotangent placed as the operand is (a transpose formed against the other operand's
        # own placement lands on it, the confirm review of row 12225's bilateral split).
        # An operand holding a sum still to take keeps the cotangent JAX's rules place (`holds_sum`).
        return tuple(part if holds_sum(operand) or (placed := explicit_spec(operand)) is None
                     else jax.sharding.reshard(part, placed)
                     for part, operand in zip(pulled, operands, strict=True))

    def forward(lhs, rhs):
        return typed(lhs, rhs, numbers)

    return contracted("test_a_bf16_dot_holds_every_order")(
        linear_tangent(lambda lhs, rhs: (forward(lhs, rhs), ()), pulled, forward,
                       kind=Protected()))(lhs, rhs)


def _transposed(cotangent, operands, numbers, side: int, product) -> jax.Array:
    """The transpose of a dot to its operand at `side` (0 the left, 1 the right) over the dot's
    `cotangent`: `product` of the cotangent and the other operand, contracting the other's free axes
    and batching the batch axes, its axes then put in the operand's order. The dot's result holds
    its batch axes, then the left operand's free axes, then the right's, each in axis order."""
    (contracting, batch) = numbers
    mine_contracting, other_contracting = contracting[side], contracting[1 - side]
    mine_batch, other_batch = batch[side], batch[1 - side]
    mine, other = operands[side], operands[1 - side]
    mine_free = [axis for axis in range(mine.ndim) if axis not in (*mine_contracting, *mine_batch)]
    other_free = [axis for axis in range(other.ndim) if axis not in (*other_contracting, *other_batch)]
    first_free = len(mine_batch) + (len(mine_free) if side == 0 else 0)
    other_axes = tuple(range(first_free, first_free + len(other_free)))
    formed = product(cotangent, other, ((other_axes, tuple(other_free)),
                                        (tuple(range(len(mine_batch))), tuple(other_batch))))
    # `formed` holds the batch axes, the cotangent's other free axes (this operand's own) and the other
    # operand's contracting axes in axis order, each the axis of this operand it contracted with.
    order = [*mine_batch, *mine_free,
             *(mine_contracting[other_contracting.index(axis)] for axis in sorted(other_contracting))]
    return jnp.transpose(formed, tuple(int(axis) for axis in np.argsort(order)))


def bf16_einsum(subscripts: str, *operands, **kwargs) -> jax.Array:
    """`jnp.einsum(subscripts, *operands)` in fp32, its every product `bf16_dot_general`'s: einsum plans
    the contraction (its path, broadcasts, and each operand's own sums, diagonals and transposes, which
    form no product and run in fp32) and hands each pairwise product to `bf16_dot_general` as its
    `_dot_general`. `kwargs` are einsum's own (`optimize`, `out_sharding`)."""
    return jnp.einsum(subscripts, *operands, preferred_element_type=jnp.float32, _dot_general=_bf16_product,
                      **kwargs)


def _bf16_product(lhs: jax.Array, rhs: jax.Array, dimension_numbers, precision=None,
                  preferred_element_type: DTypeLike | None = None,
                  out_sharding: jax.sharding.Sharding | None = None) -> jax.Array:
    """One of `jnp.einsum`'s pairwise products (its `_dot_general`) as `bf16_dot_general` forms it, in the
    fp32 `bf16_einsum` asks of every product, placed where einsum asks."""
    product = bf16_dot_general(lhs, rhs, dimension_numbers)
    return product if out_sharding is None else jax.sharding.reshard(product, out_sharding)


def at_precision[Output](precision: jax.lax.PrecisionLike, parameters: tuple[int, ...] = ()
                         ) -> Callable[[Callable[..., Output]], Callable[..., Output]]:
    """`fn` traced, forward and backward, at `precision`, the default included: a product that takes no
    precision of its own (a Pallas dot, jax.nn's xla attention) reads it from the configuration at trace
    time (`jax.default_matmul_precision`), and a custom derivative's backward is traced after the forward
    returns, so it is held under the same setting by hand. An unset precision is the configured one, which
    forward and backward read alike: `fn` itself. A precision no config names (`configurable_precision`)
    is refused.

    A fused attention route under a call's stated precision (`attention._attention_kernel`), and Mosaic's
    TPU kernels at the default: they refuse a 16-bit matmul at HIGHEST ("Bad lhs type"); 16-bit dots are
    exact at any precision, and an fp32 dot takes one bf16 pass on a TPU, so splash's P·V rounds P to
    bf16 as every default-precision run does (`attention_tpu`). The arguments at `parameters` are the
    call's parameters (an attention's bias and sinks), whose gradients its backward forms with the others'
    cotangents (`derivatives.with_gradients_formed_at`)."""
    name = _configured_name(precision)

    def held(fn: Callable[..., Output]) -> Callable[..., Output]:
        if name is None:
            return fn

        def forward(*args):
            with jax.default_matmul_precision(name):
                return jax.vjp(fn, *args)

        def backward(pullback, cotangent):
            with jax.default_matmul_precision(name):
                return with_gradients_formed_at(tuple(pullback(cotangent)), parameters)

        # A protected span: the precision it holds is the kernels' own, which no rewrite may change.
        @contracted("test_a_default_precision_call_holds_every_order_it_takes")
        @functools.partial(declared_derivative, Protected(), forward=forward, backward=backward)
        def run(*args):
            with jax.default_matmul_precision(name):
                return fn(*args)

        return run
    return held


def sqrt_or_zero(square):
    """`sqrt(square)`, and zero with a zero gradient where `square` is zero:
    sqrt's infinite derivative there would turn a masked row's zero upstream
    gradient into NaN for every parameter it touches."""
    positive = square > 0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, square, 1.0)), 0.0)
