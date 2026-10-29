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

Dew's test processes run wide CPU arithmetic (`WIDE_CPU_ARITHMETIC`, which tests/lane_environment.py
sets): a CPU then sums every float32 product of the dot family, dot_general, einsum and matmul, in
float64 and rounds once (`accumulated`), since XLA:CPU chooses its fp32 dot accumulation by shape,
instruction set and release, and jax 0.12's summed twice numpy's error from float64 where 0.11's
summed 0.6 of it. A library process's products are the plain products on every platform.
Convolutions, depthwise ones included, keep XLA's fp32 sum on a CPU: no convolution's check 1
failed there, and XLA:CPU's float64 convolution cost a 3x3 one's training step 44 times (15.5 to
675 ms on a 4x32x32x128 map), its best rewrite 2 to 4. Should a convolution's check 1 resolve above
its bar, the recorded fix is that rewrite: the kernel's shifted windows taken inside the float64
scope and contracted with the taps as one 2D product (stride 1 train 3.9 times, stride 2 1.6)."""

from __future__ import annotations

import contextlib
import functools
import os
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax.typing import Dtype, PrecisionLike
from jax.extend import core
from jax.typing import DTypeLike

__all__ = [
    "FP32_SUMS",
    "L2NORM_EPS",
    "WIDE_CPU_ARITHMETIC",
    "accumulated",
    "activation_and_slope",
    "added_to",
    "asks_default_precision",
    "at_default_precision",
    "at_least_fp32",
    "bf16_dot_general",
    "bf16_einsum",
    "bf16_logits",
    "bf16_product",
    "biased",
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
    "float32_dot_general",
    "float32_einsum",
    "float32_matmul",
    "float32_products",
    "float64_sum",
    "fp32_result_dot_general",
    "head_dot_general",
    "head_product",
    "l2norm",
    "operand_precision",
    "paired_sum",
    "precision_names",
    "products_dtype",
    "rms_statistics",
    "rms_weighted",
    "rounded_operand",
    "rounded_to",
    "rounds_to_bf16",
    "running_sum",
    "running_total",
    "scaled",
    "scaled_logits",
    "span_means",
    "spread",
    "sqrt_or_zero",
    "stochastic_bf16",
    "two_product",
    "two_sum",
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
    step's seed and the element's index (`optim.stochastic_round_bf16`)."""
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
    roundings."""
    return jnp.promote_types(jnp.float32 if dtype is None else dtype, jnp.float32)


def rms_statistics(x: jax.Array, epsilon: float, *, axis=-1, keepdims: bool = True,
                   count: int | None = None) -> tuple[jax.Array, jax.Array]:
    """Mean square and inverse RMS in at least fp32. A padded kernel tile supplies its logical
    `count` after zeroing padding; an ordinary array uses its reduction dimensions. Callers own
    when the inverse, normalized values and learned scale round."""
    square = jnp.square(x.astype(at_least_fp32(x.dtype)))
    mean = (jnp.mean(square, axis=axis, keepdims=keepdims) if count is None
            else jnp.sum(square, axis=axis, keepdims=keepdims) / count)
    return mean, jax.lax.rsqrt(mean + epsilon)


def activation_and_slope(gate, activation: str) -> tuple[jax.Array, jax.Array]:
    """An fp32 gate's `act(gate)` and its derivative, for 'silu' or 'sigmoid'.

    silu's derivative `s + g s (1 - s)`, with `s = sigmoid(g)`, cancels to
    zero near g = -1.28; formed as the product rule forms it, `s + g (s (1 - s))`,
    its worst element there is half `(g s)(1 - s) + s`'s, as XLA's is."""
    sigmoid = jax.nn.sigmoid(gate)
    slope = sigmoid * (1.0 - sigmoid)
    if activation == 'sigmoid':
        return sigmoid, slope
    return gate * sigmoid, sigmoid + gate * slope


def rms_weighted(x: jax.Array, inverse: jax.Array, *, weight: jax.Array | None = None,
                 gate: jax.Array | None = None, activation: str = 'silu',
                 rounded: Callable[[jax.Array], jax.Array] | None = None) -> jax.Array:
    """An RMS norm's arithmetic past its statistics: `x` times its rows' `inverse` RMS (`rms_statistics`'s,
    or a kernel's own), times `weight`, times `activation(gate)` (`activation_and_slope`: Qwen3-Next's gated
    norm), in that order and in `x`'s dtype, which the caller widens (at least fp32) and rounds once after.
    `rounded` rounds the normalized rows before the weight multiplies them (Llama's and Qwen3's cast first,
    `RMSNorm.scale_after_cast`). The fused norm kernel, the recurrence's gated output norm and the portable
    gated norm compute it alike, each keeping its own memory schedule and derivative."""
    y = x * inverse
    if weight is not None:
        y = (y if rounded is None else rounded(y)) * weight
    return y if gate is None else y * activation_and_slope(gate, activation)[0]


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
    Rounding the product and the bias once together, as torch's addmm does, wrote an fp32 product
    and a pass of its own on every biased layer (a DiT-B step 17.5% slower on an A100), for a bf16
    output 1.37 times torch's rounding, which the parity rule holds already.

    JAX transposes a broadcast into a sum in the broadcast operand's dtype (jax-ml/jax#30310), so
    a bfloat16 bias added to bfloat16 products sums its gradient over every position in bfloat16,
    where torch's addmm sums its in float32: Qwen 2's projection biases, summed over 8192 tokens,
    came out up to 3.75 times further from float64 than torch's own bfloat16 gradients. Broadcast
    at fp32, the gradient sums at fp32 and rounds once into the bias. A CPU running Dew's wide
    arithmetic (`WIDE_CPU_ARITHMETIC`) sums it in float64, as it sums a float32 product (`spread`).
    """
    dtype, wide = jnp.result_type(product, bias), at_least_fp32(bias.dtype)
    # The bias takes the product's rank before the sum, as flax's layers reshape it, so no rank is
    # promoted implicitly. Broadcast at fp32 and rounded back, it adds as flax adds it (XLA folds
    # the round trip, so the product and bias fuse as before) and its gradient sums at fp32.
    bias = jnp.reshape(bias.astype(wide), (1,) * (product.ndim - bias.ndim) + bias.shape)
    return product.astype(dtype) + spread(bias, product.shape).astype(dtype)


def spread(value: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    """`value` broadcast to `shape`, whose transpose, the sum of a cotangent over every position the
    value spread over (a bias's gradient), a CPU running wide arithmetic (`WIDE_CPU_ARITHMETIC`, Dew's
    tests) takes in float64 and rounds once, as it sums a float32 product (`accumulated`): there the
    broadcast is the outer product of ones and the value, whose transpose is a product. Elsewhere,
    or where `accumulated` leaves a product its own sum, it is the broadcast, whose transpose JAX
    sums in the value's dtype."""
    axes = tuple(axis for axis, (size, total) in enumerate(zip(value.shape, shape, strict=True))
                 if size == 1 and total != 1)
    if not axes or not _sums_in_float64(value):
        return jnp.broadcast_to(value, shape)
    kept = tuple(axis for axis in range(len(shape)) if axis not in axes)

    def outer(value: jax.Array) -> jax.Array:
        # Only the size-one axes are dropped, so a sharded kept axis keeps its own sharding.
        ones = jnp.ones(tuple(shape[axis] for axis in axes), value.dtype)
        product = accumulated(functools.partial(jax.lax.dot_general, dimension_numbers=(((), ()), ((), ()))),
                              ones, jnp.squeeze(value, axes))
        return jnp.transpose(product, tuple(np.argsort((*axes, *kept))))

    return jax.lax.platform_dependent(value, cpu=outer, default=lambda value: jnp.broadcast_to(value, shape))


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
    names = precision_names(precision)
    if not names and configured:
        names = precision_names(jax.config.jax_default_matmul_precision)
    return names <= {'DEFAULT'}


def float32_products[P: jax.lax.PrecisionLike](dtype: Dtype | None, precision: P = None
                                                ) -> P | jax.lax.Precision:
    """The precision products computed in `dtype` take where none is stated: float32 ones multiply
    float32 products (`Precision.HIGHEST`), as torch's float32 matmuls do by default; JAX's
    default runs them in TF32 on a GPU, 2.9e-4 from float64 against 2.9e-7. A stated precision
    is the caller's own, TF32 (`'tensorfloat32'`) its opt-in; other dtypes keep theirs, and so
    does an unknown one (None), which leaves the products to resolve it from their own dtype."""
    if precision is not None or dtype is None or jnp.dtype(dtype) != jnp.float32:
        return precision
    return jax.lax.Precision.HIGHEST


def products_dtype(*operands) -> jnp.dtype | None:
    """The dtype `operands` multiply in, by jax's implicit promotion; None where an 8-bit float
    takes part, which jax promotes with nothing and whose product has no precision to resolve."""
    dtypes = [jnp.dtype(operand.dtype) for operand in operands]
    if any(jnp.issubdtype(dtype, jnp.floating) and jnp.finfo(dtype).bits < 16 for dtype in dtypes):
        return None
    return jnp.result_type(*dtypes)


def _intercepted() -> bool:
    """Whether this thread traces a quantized model's method, whose products Qwix owns
    (`dew.cache.QUANTIZED_TRACE`, which jit's cache keys on)."""
    from dew.cache import QUANTIZED_TRACE

    return QUANTIZED_TRACE.value is not None


_OWN_DERIVATIVES = (core.primitives.custom_vjp_call_p, core.primitives.custom_jvp_call_p)
"""JAX's primitives for a function that states its own derivative (`jax.custom_vjp`, `jax.custom_jvp`)."""


def _own_backward(jaxpr) -> bool:
    """Whether a product's trace states a derivative of its own (a `jax.custom_vjp`, as Qwix's
    quantized products are when Qwix quantizes a model directly, or a `jax.custom_jvp`), in it or any
    program inside it (`jax.extend.core.subjaxprs`): a product its library owns, which `accumulated`
    neither widens nor transposes, since a transpose would differentiate its primal instead."""
    return (any(eqn.primitive in _OWN_DERIVATIVES for eqn in jaxpr.eqns)
            or any(_own_backward(program) for program in core.subjaxprs(jaxpr)))


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


def _sums_in_float64(*operands) -> bool:
    """Whether a CPU sums these operands' products in float64 (`accumulated`): in a process running wide
    CPU arithmetic (`WIDE_CPU_ARITHMETIC`), float32 arrays, in no scope that states fp32 sums
    (`FP32_SUMS`), outside a quantized model's trace (`_intercepted`)."""
    return (wide_cpu_arithmetic() and jax.config.jax_default_matmul_precision != FP32_SUMS
            and not _intercepted()
            and all(isinstance(operand, jax.Array | np.ndarray) for operand in operands)
            and products_dtype(*operands) == jnp.dtype(jnp.float32))


def float64_sum(product: Callable[..., jax.Array], *operands, dtype: DTypeLike) -> jax.Array:
    """`product(*operands)`, a sum of products, taken in float64 from its operands widened exactly and
    rounded once to `dtype`, under `jax.enable_x64` for its own trace: the sum `accumulated` takes on a
    CPU under Dew's wide arithmetic (`WIDE_CPU_ARITHMETIC`), with no platform of its own to choose.
    Its products run at HIGHEST precision, whatever fp32 algorithm the caller bound, which would
    narrow float64 operands back to float32."""
    with jax.enable_x64(new_val=True):
        wide = product(*_widened(operands), precision=jax.lax.Precision.HIGHEST, preferred_element_type=None)
        return wide.astype(dtype)


def _widened(operands) -> list[jax.Array]:
    """`operands` as the float64 operands of their float32 product: each converted to the dtype they
    multiply in (`products_dtype`), as the product converts them, then widened exactly."""
    dtype = products_dtype(*operands)
    return [operand.astype(dtype).astype(jnp.float64) for operand in operands]


def widened_reduction(reduction: Callable[..., Sequence[jax.Array]],
                      *operands: jax.Array) -> tuple[jax.Array, ...]:
    """`reduction(*operands)`, a reduction returning arrays: the plain one, but in a process running wide
    CPU arithmetic (`WIDE_CPU_ARITHMETIC`, Dew's tests) a CPU takes it over the float32 operands widened
    exactly to float64, each float64 result rounded once back to float32, as `accumulated` takes a
    product's sums: XLA's CPU compiler runs a reduction with a combine of its own as one loop over the
    row, rounding at each step. A process without the setting, or operands not float32, take the plain
    reduction on every platform, chosen by the platform a program is lowered for (`platform_dependent`)."""
    if not wide_cpu_arithmetic() or not any(operand.dtype == jnp.float32 for operand in operands):
        return tuple(reduction(*operands))

    def wide(*operands):
        with jax.enable_x64(new_val=True):
            reduced = reduction(*(operand.astype(jnp.float64) if operand.dtype == jnp.float32 else operand
                                  for operand in operands))
            return tuple(term.astype(jnp.float32) if term.dtype == jnp.float64 else term for term in reduced)

    return tuple(jax.lax.platform_dependent(*operands, cpu=wide, default=lambda *operands: tuple(
        reduction(*operands))))


def accumulated(product: Callable[..., jax.Array], *operands,
                preferred_element_type: DTypeLike | None = None) -> jax.Array:
    """`product(*operands, preferred_element_type=...)`, a sum of products of `operands` (a dot, an
    einsum, a matmul): the plain product, but in a process running wide CPU arithmetic
    (`WIDE_CPU_ARITHMETIC`, Dew's tests), where they multiply in float32 (`products_dtype`) a CPU
    sums the products in float64 and rounds the sum once to the dtype the product returns.
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
    traces the plain product alone. Under it, it is real CPU execution's policy alone, chosen by the
    platform the program is lowered for
    (`platform_dependent`), which keys every compiled program: a step a CPU lowers for a GPU target
    (`dew.training.trainer.Trainer.lower_for`, `lowering_platforms=('cuda',)`) takes the target's
    fp32 sum, and a Pallas kernel's body, interpreted on a CPU to stand in for a GPU, never forms its
    products here. A computation that stands for such a target, a reference for a kernel interpreted
    on a CPU, states its fp32 sums with JAX's dot algorithm for them, `with
    jax.default_matmul_precision(FP32_SUMS)`, which jit's cache keys as it keys every precision."""
    from dew.nn.kernels.generation import linear_tangent

    narrow = functools.partial(product, preferred_element_type=preferred_element_type)
    if not _sums_in_float64(*operands):
        return narrow(*operands)
    traced, shape = jax.make_jaxpr(narrow, return_shape=True)(*operands)
    dtype = shape.dtype
    if jnp.promote_types(dtype, jnp.float32) != jnp.float32 or _own_backward(traced.jaxpr):
        # A wider result converts the operands to it, so the products are not float32's; a product
        # with a backward of its own is its library's.
        return narrow(*operands)

    def summed(*operands):
        return float64_sum(product, *operands, dtype=dtype)

    def pulled(saved, operands, cotangent, live):
        """The cotangents of the operands at `live`, each the float64 sum of its products rounded
        once: JAX transposes a product after its trace, outside the scope that widened it, so the
        pullback is formed in that scope here, for the operands a reverse pass differentiates alone."""
        with jax.enable_x64(new_val=True):
            wide = _widened(operands)
            wide_cotangent = cotangent.astype(jnp.float64)

            def transposed(index):
                # The product is linear in each operand alone, so its transpose there forms the
                # cotangent's products and nothing of the forward.
                def alone(operand):
                    return product(*wide[:index], operand, *wide[index + 1:],
                                   precision=jax.lax.Precision.HIGHEST, preferred_element_type=None)

                (gradient,) = jax.linear_transpose(alone, wide[index])(wide_cotangent)
                return gradient.astype(operands[index].dtype)

            return tuple(transposed(index) for index in live)

    # Forward mode differentiates `summed` as it traces, inside its scope; reverse mode runs `pulled`.
    wide = linear_tangent(lambda *operands: (summed(*operands), ()), pulled, summed, live_cotangents=True)
    return jax.lax.platform_dependent(*operands, cpu=wide, default=narrow)


L2NORM_EPS = 1e-6
"""The epsilon `l2norm` adds under its square root: FLA's, and Dory's `justnorm`'s."""


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
    wide = x.astype(at_least_fp32(x.dtype))
    return (wide * jax.lax.rsqrt(jnp.sum(jnp.square(wide), axis=-1, keepdims=True) + eps)).astype(x.dtype)


def float32_dot_general(lhs, rhs, dimension_numbers, precision: jax.lax.PrecisionLike = None,
                        preferred_element_type: DTypeLike | None = None, **kwargs) -> jax.Array:
    """`jax.lax.dot_general` at the precision its operands' own dtype takes (`float32_products`),
    accumulated as Dew accumulates a CPU's float32 products (`accumulated`): the product a flax
    layer forms (`nn.Dense(dot_general=...)`), resolved where flax has promoted its operands, so a
    layer of any dtype, unset included, multiplies as it computes. At the `BF16_BF16_F32` algorithm
    it is `bf16_dot_general`'s product."""
    precision = float32_products(products_dtype(lhs, rhs), precision)
    if _bf16_algorithm(precision):
        return _held(bf16_dot_general(lhs, rhs, dimension_numbers), preferred_element_type, lhs, rhs)
    return accumulated(functools.partial(jax.lax.dot_general, dimension_numbers=dimension_numbers,
                                         precision=precision, **kwargs),
                       lhs, rhs, preferred_element_type=preferred_element_type)


def float32_einsum(subscripts: str, *operands, precision: jax.lax.PrecisionLike = None,
                   preferred_element_type: DTypeLike | None = None, **kwargs) -> jax.Array:
    """`jnp.einsum` at the precision its operands' own dtype takes (`float32_products`), as
    `float32_dot_general` is a layer's, accumulated alike, and `bf16_einsum`'s product at the
    `BF16_BF16_F32` algorithm."""
    precision = float32_products(products_dtype(*operands), precision)
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


def float32_matmul(lhs, rhs) -> jax.Array:
    """`lhs @ rhs` at the precision its operands' own dtype takes (`float32_products`), accumulated
    as `float32_dot_general` is."""
    return accumulated(functools.partial(jnp.matmul, precision=float32_products(products_dtype(lhs, rhs))),
                       lhs, rhs)


def configurable_precision(precision: jax.lax.PrecisionLike) -> bool:
    """Whether `jax.default_matmul_precision` can name `precision`: one precision or dot algorithm
    preset, unset or default included; not a pair of precisions or a dot algorithm."""
    return len(precision_names(precision)) <= 1 and not isinstance(precision, jax.lax.DotAlgorithm)


def configured_precision(precision: jax.lax.PrecisionLike):
    """A context in which a product that takes no precision of its own (jax.nn's xla attention)
    multiplies at `precision`, through `jax.default_matmul_precision`; an unset or default
    precision changes nothing. A precision no config names (`configurable_precision`) is
    refused."""
    if not configurable_precision(precision):
        raise ValueError(f"precision={precision} has no jax_default_matmul_precision name")
    names = precision_names(precision)
    if names <= {'DEFAULT'}:
        return contextlib.nullcontext()
    (name,) = names
    preset = name in jax.lax.DotAlgorithmPreset.__members__
    return jax.default_matmul_precision(name if preset else name.lower())


def operand_precision(dtype: Dtype | None, precision: jax.lax.PrecisionLike = None) -> jax.lax.PrecisionLike:
    """The precision a product of `dtype` operands takes: float32 products for float32 operands
    where none is stated (`float32_products`), and for bf16 operands at the default precision the
    algorithm that keeps a bf16 dot bf16 in both directions. A caller that asked for more than the
    default keeps what it asked for, and so does a GPU older than sm80, which rejects the algorithm
    at run time (`dew.nn.kernels.generation.BF16_GPU`)."""
    precision = float32_products(dtype, precision)
    if dtype is None or jnp.dtype(dtype) != jnp.bfloat16 or not asks_default_precision(precision):
        return precision
    # Imported here: dew.nn.kernels imports this module.
    from .kernels.generation import bf16_dot_runs
    if not bf16_dot_runs():
        return precision
    return jax.lax.DotAlgorithmPreset.BF16_BF16_F32


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


def two_sum(left, right):
    """`left + right` rounded, and exactly what the rounding dropped (Knuth's TwoSum)."""
    total = left + right
    part = total - left
    return total, (left - (total - part)) + (right - part)


def two_product(left, right, *, bit_split: bool = False, subtract: bool = False,
                rounding: Callable[[jax.Array], jax.Array] | None = None) -> tuple[jax.Array, jax.Array]:
    """Dekker's rounded product and its residual, with a Veltkamp significand split.

    Guidance keeps the multiply split, which also supports float64. `bit_split`
    truncates fp32 significands instead, avoiding splitter overflow in norm
    interpretation and quantizer division; near underflow the splits differ.
    `subtract` keeps division's subtractive residual order and its signed zeros.
    Norm interpretation passes `optimization_barrier` as `rounding` to keep each
    product and residual step separate; the other callers allow fusion.
    With each arithmetic step rounded, the residual is exact where intermediates
    neither overflow nor underflow.
    """
    def rounded(value):
        return value if rounding is None else rounding(value)

    def split(value):
        if bit_split and isinstance(value, np.generic | np.ndarray):
            # A constant (a kernel's static factor) splits where it is known, on the host.
            constant = np.asarray(value, np.float32)
            high = (constant.view(np.uint32) & np.uint32(0xfffff000)).view(np.float32)
            return jnp.asarray(high), jnp.asarray(constant - high)
        if bit_split:
            bits = jax.lax.bitcast_convert_type(value, jnp.uint32) & jnp.uint32(0xfffff000)
            high = jax.lax.bitcast_convert_type(bits, jnp.float32)
        else:
            splitter = (1 << 27) + 1 if value.dtype == jnp.float64 else (1 << 12) + 1
            scaled = value * splitter
            high = scaled - (scaled - value)
        return high, rounded(value - high)

    lh, ll = split(left)
    rh, rl = split(right)
    product = rounded(left * right)
    head = rounded(lh * rh)
    lost = rounded(product - head if subtract else head - product)
    cross = ((ll, rh), (lh, rl)) if subtract else ((lh, rl), (ll, rh))
    for first, second in cross:
        term = rounded(first * second)
        lost = rounded(lost - term if subtract else lost + term)
    tail = rounded(ll * rl)
    return product, rounded(tail - lost if subtract else lost + tail)


@jax.custom_jvp
def correctly_rounded_divide(numerator: jax.typing.ArrayLike, denominator: jax.typing.ArrayLike) -> jax.Array:
    """The correctly rounded fp32 quotient, IEEE division's, on XLA's approximate
    dividers too: XLA:CPU rewrites a division by a broadcast into a multiplication
    by its reciprocal (openxla/xla#51148), and XLA:GPU's fp32 a / b is off IEEE on
    30 to 50% of quotients (DIVIDE-a100, DIVIDE-h100 receipts), where a quantizer's
    scales and a diffusion schedule's states must round as the reference's host or
    CUDA division does (compressed-tensors' NVFP4, the FP8 block cast, PyTorch's
    schedulers).

    Each operand's significand, in [0.5, 1), divides apart from its exponent
    (`_rounded_significand`), so every intermediate stays normal whatever the
    operands' range; the exponents' difference then scales the rounded
    significand by its exponent bits, exact for a normal result. A quotient
    below the normal range rounds once to the subnormal grid: the significand
    scaled onto that grid rounds to its nearest integer, ties to even, the exact
    residual settling a scaled significand that lands on a half, and the
    integer is the result's bits, which no flush to zero reaches. Past the
    largest float the quotient is infinite. A zero, infinite or NaN operand,
    or a subnormal one, which XLA:CPU reads as zero, is the division's. Its
    derivative is the division's, as JAX differentiates `a / b`.
    """
    numerator, denominator = jnp.broadcast_arrays(jnp.asarray(numerator, jnp.float32),
                                                  jnp.asarray(denominator, jnp.float32))
    plain = numerator / denominator
    tiny = jnp.finfo(jnp.float32).tiny
    special = ((jnp.abs(numerator) < tiny) | (jnp.abs(denominator) < tiny)
               | ~jnp.isfinite(numerator) | ~jnp.isfinite(denominator))
    negative = (numerator < 0) != (denominator < 0)
    # Away from the specials both operands are normal: frexp's significands are exact, in [0.5, 1).
    held = jnp.where(special, 1.0, numerator), jnp.where(special, 1.0, denominator)
    (top, top_exponent), (bottom, bottom_exponent) = (jnp.frexp(jnp.abs(value)) for value in held)
    significand, residual = _rounded_significand(top, bottom)
    exponent = top_exponent - bottom_exponent
    bits = jax.lax.bitcast_convert_type(significand, jnp.int32)
    # The rounded significand's biased exponent once scaled: 1 to 254 is a normal float.
    field = ((bits >> 23) & 0xFF) + exponent
    normal = jax.lax.bitcast_convert_type(bits + (exponent << 23), jnp.float32)
    # Below the normal range: the significand on the grid of 2^-149, a value under 2^23 whose
    # integer part and half are exact in fp32; within the float32 range scaled by 2^shift.
    shift = jnp.clip(exponent + 149, -126, 127)
    scaled = significand * jax.lax.bitcast_convert_type((shift + 127) << 23, jnp.float32)
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


def _rounded_significand(numerator: jax.Array, denominator: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The correctly rounded fp32 quotient of two significands in [0.5, 1), and the sign of the exact
    quotient's distance from it (+1 above, -1 below, 0 exact).

    A Dekker TwoProduct gives the exact residual of the approximate
    quotient, split into two fp32 values. One residual correction puts the
    quotient within one ulp; comparing its exact residual with each
    neighbour's half-spacing times the divisor chooses nearest-even.
    That threshold product is exact because the half-spacing is a power
    of two, and every intermediate is normal at these magnitudes."""
    quotient = numerator / denominator
    product, error = two_product(quotient, denominator, bit_split=True, subtract=True)
    residual = (numerator - product) - error
    quotient = quotient + residual / denominator
    product, error = two_product(quotient, denominator, bit_split=True, subtract=True)
    difference = numerator - product
    high = difference - error
    recovered = high - difference
    low = (difference - (high - recovered)) - (error + recovered)
    above, below = jnp.nextafter(quotient, jnp.inf), jnp.nextafter(quotient, -jnp.inf)
    upper = (above - quotient) * 0.5 * denominator
    lower = (quotient - below) * 0.5 * denominator
    odd = (jax.lax.bitcast_convert_type(quotient, jnp.uint32) & 1) != 0
    up = (high > upper) | ((high == upper) & ((low > 0) | ((low == 0) & odd)))
    down = (high < -lower) | ((high == -lower) & ((low < 0) | ((low == 0) & odd)))
    rounded = jnp.where(up, above, jnp.where(down, below, quotient))
    # The exact quotient's side of the rounded one: the sign of its exact residual, the product's
    # difference from the numerator less its error, which is zero only where the two are equal.
    product, error = two_product(rounded, denominator, bit_split=True, subtract=True)
    return rounded, jnp.sign((numerator - product) - error)


@correctly_rounded_divide.defjvp
def _divide_jvp(primals, tangents):
    """The rounding is a selection among neighbours, so the quotient's tangent is the division's."""
    return correctly_rounded_divide(*primals), jax.jvp(jnp.divide, primals, tangents)[1]


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
    total, dropped = two_sum(left_hi, right_hi)
    return two_sum(total, dropped + left_lo + right_lo)


def compensated_difference(later, earlier):
    """`later - earlier` of two double-float values, rounded once: the halves'
    differences summed, so the difference of two near values (a range of a
    running sum) keeps its own precision, where the his' difference alone
    carries their whole magnitude's rounding: a one-token range of a decay's
    running sum near -78 off by 4e-6. Broadcasts as the halves do."""
    return (later[0] - earlier[0]) + (later[1] - earlier[1])


def compensated_sum(values: jax.Array, axes: tuple[int, ...]) -> jax.Array:
    """The sum of `values` over `axes`, rounded once: one XLA reduction whose accumulator is a
    double-float `(hi, lo)` pair combined by `compensated_add`, so it partitions as any reduction
    does, and what the roundings dropped is added back at the end.

    One plain reduction over a loss's every axis rounds as it goes; on 6144 JEPA squares it left the
    sum one float32 step above the correctly rounded one, and the loss a float32 ulp from float64,
    where numpy's float32 mean of the same squares was correctly rounded. A narrower float adds in
    float32 and returns in its own dtype, as `jnp.sum` does; an integer sum is `jnp.sum`'s. The
    derivative is `jnp.sum`'s."""
    if not jnp.issubdtype(values.dtype, jnp.floating):
        return jnp.sum(values, axes)
    return _compensated_sum(values, tuple(axes))


@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def _compensated_sum(values: jax.Array, axes: tuple[int, ...]) -> jax.Array:
    wide = values.astype(at_least_fp32(values.dtype))
    zero = jnp.zeros((), wide.dtype)
    hi, lo = jax.lax.reduce((wide, jnp.zeros_like(wide)), (zero, zero), compensated_add, axes)
    return compensated_total((hi, lo), jnp.sum(wide, axes)).astype(values.dtype)


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


def added_to(running: jax.Array, term: jax.Array) -> jax.Array:
    """`running` (`running_sum`) with `term` added: through `compensated_add` where it is a pair."""
    term = term.astype(running.dtype)
    if running.shape[0] == 1:
        return running + term[None]
    pair = compensated_add((running[0], running[1]), (term, jnp.zeros_like(term)))
    return jnp.stack((*pair, running[2] + term))


def running_total(running: jax.Array) -> jax.Array:
    """A running sum's value (`running_sum`): a pair's rounded once (`compensated_total`), else the sum."""
    if running.shape[0] == 1:
        return running[0]
    return compensated_total((running[0], running[1]), running[2])


def compensated_total(pair: tuple[jax.Array, jax.Array], plain: jax.Array) -> jax.Array:
    """A compensated sum's value, its `(hi, lo)` pair rounded once, or the `plain` sum of the same
    values where that reaches an infinity: +-inf or NaN as `jnp.sum`'s is, where every residual past
    the infinity is inf - inf. `compensated_add` itself stays arithmetic alone, so a tangent's scan
    through it stays linear; a running sum that can overflow keeps its plain sum beside the pair
    and reads its value here."""
    return jnp.where(jnp.isfinite(plain), pair[0] + pair[1], plain)


@_compensated_sum.defjvp
def _compensated_sum_jvp(axes: tuple[int, ...], primals: tuple[jax.Array],
                         tangents: tuple[jax.Array]) -> tuple[jax.Array, jax.Array]:
    return _compensated_sum(primals[0], axes), jnp.sum(tangents[0], axes)

@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def rounded_operand(x: jax.Array, dtype: Dtype) -> jax.Array:
    """The value `x` takes in `dtype`, held in `x`'s own dtype, with a
    straight-through tangent that is never rounded a second time.

    The rounding is the cast itself, subnormals included. The barrier keeps
    `xla_allow_excess_precision` from deleting the round trip under jit, and
    stays where `dtype` is `x`'s own so XLA cannot carry an operand wider into
    the fusion that reads it. Formats are compared, not widths: float16 and
    bfloat16 each lose bits in the other.
    """
    if jnp.dtype(dtype) == jnp.dtype(x.dtype):
        return jax.lax.optimization_barrier(x)
    return jax.lax.optimization_barrier(x.astype(dtype)).astype(x.dtype)


@rounded_operand.defjvp
def _rounded_operand_jvp(dtype: Dtype, primals: tuple[jax.Array],
                         tangents: tuple[jax.Array]) -> tuple[jax.Array, jax.Array]:
    return jnp.asarray(rounded_operand(primals[0], dtype)), tangents[0]


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
    operands' dtype, at the precision that dtype takes (`float32_products`).
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
            precision=float32_products(products_dtype(lhs, rhs), resolved)),
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
    return float32_einsum(subscripts, hidden.astype(wide), head, precision=precision,
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
        def native(lhs, rhs):
            return jax.lax.dot_general(lhs, rhs, numbers, precision=jax.lax.Precision.DEFAULT,
                                       preferred_element_type=jnp.float32)

        def widened(lhs, rhs):
            # XLA:CPU has no bf16 x bf16 = fp32 dot at every layout ("Unsupported element type for
            # DotThunk"); bf16 values multiply exactly in fp32, so the fp32 dot is the same arithmetic.
            return jax.lax.dot_general(lhs.astype(jnp.float32), rhs.astype(jnp.float32), numbers,
                                       precision=jax.lax.Precision.HIGHEST,
                                       preferred_element_type=jnp.float32)

        return jax.lax.platform_dependent(lhs.astype(jnp.bfloat16), rhs.astype(jnp.bfloat16), cpu=widened,
                                          default=native)

    def pulled(saved, operands, cotangent):
        """Each operand's cotangent: the product's transpose to it over the cotangent rounded to bf16."""
        del saved
        rounded = cotangent.astype(jnp.bfloat16)
        return tuple(_transposed(rounded, operands, numbers, side, typed).astype(operands[side].dtype)
                     for side in (0, 1))

    def forward(lhs, rhs):
        return typed(lhs, rhs, numbers)

    return linear_tangent(lambda lhs, rhs: (forward(lhs, rhs), ()), pulled, forward)(lhs, rhs)


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


def at_default_precision[Output](fn: Callable[..., Output]) -> Callable[..., Output]:
    """`fn` traced, forward and backward, under the default matmul precision.

    Mosaic's TPU kernels refuse a 16-bit matmul at HIGHEST ("Bad lhs type"),
    and Pallas dots read the precision from the configuration at trace time.
    16-bit dots are exact at any precision; an fp32 dot takes one bf16 pass on
    a TPU, so splash's P·V rounds P to bf16 as every default-precision run
    does. The kernels take no per-dot precision, so there is no other setting.
    The backward is traced after the forward returns, so it is held under the
    same setting by hand."""
    @jax.custom_vjp
    def run(*args):
        with jax.default_matmul_precision('default'):
            return fn(*args)

    def forward(*args):
        with jax.default_matmul_precision('default'):
            return jax.vjp(fn, *args)

    def backward(pullback, cotangent):
        with jax.default_matmul_precision('default'):
            return pullback(cotangent)

    run.defvjp(forward, backward)
    return run


def sqrt_or_zero(square):
    """`sqrt(square)`, and zero with a zero gradient where `square` is zero:
    sqrt's infinite derivative there would turn a masked row's zero upstream
    gradient into NaN for every parameter it touches."""
    positive = square > 0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, square, 1.0)), 0.0)
