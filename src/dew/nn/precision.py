"""The precision a bf16 model's fp32 boundaries multiply at.

A model's last matmul before a loss (the patch output head, the vocabulary
head) keeps an fp32 result, since the loss is fp32; `head_product` says why
the vocabulary head still rounds to bf16 values at the default precision.
`preferred_element_type=jnp.float32` alone makes that dot's cotangent fp32,
so the mixed f32 x bf16 backward products run at the fp32 rate. XLA's
`BF16_BF16_F32` dot algorithm names the arithmetic instead (bf16 operands,
fp32 accumulation, whatever the storage), and jax carries `precision` into
the transpose rules, so the backward follows; jax's own xla attention sets it
for this reason (jax-ml/jax#24047).
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import jax
import jax.numpy as jnp
from flax.typing import Dtype, PrecisionLike
from jax.typing import DTypeLike

if TYPE_CHECKING:
    from flax.core.scope import Scope
    from flax.typing import VariableDict


def at_least_fp32(dtype: DTypeLike | None) -> jnp.dtype:
    """The dtype arithmetic Dew keeps from rounding below float32 runs in:
    float32 for float32, every narrower float and a module's unset dtype
    (None), and `dtype` itself where it is wider. Norm statistics, softmaxes,
    rotary and time-embedding angles, the vocabulary head's product and loss
    reductions read it, so a bfloat16 model reduces in float32 and a float64
    model computes in float64 throughout, not with its float32 twin's
    roundings."""
    return jnp.promote_types(jnp.float32 if dtype is None else dtype, jnp.float32)


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


def float32_products(dtype: Dtype | None, precision: PrecisionLike = None) -> PrecisionLike:
    """The precision products computed in `dtype` take where none is stated: float32 ones multiply
    float32 products (`Precision.HIGHEST`), as torch's float32 matmuls do by default; JAX's
    default runs them in TF32 on a GPU, 2.9e-4 from float64 against 2.9e-7. A stated precision
    is the caller's own, TF32 (`'tensorfloat32'`) its opt-in; other dtypes keep theirs, and so
    does an unknown one (None), which leaves the products to resolve it from their own dtype."""
    if precision is not None or dtype is None or jnp.dtype(dtype) != jnp.float32:
        return precision
    return jax.lax.Precision.HIGHEST


class Module(Protocol):
    """What `computes_in` reads of a flax module: its dtype field, scope and variables."""

    @property
    def dtype(self) -> Dtype | None: ...
    @property
    def scope(self) -> Scope | None: ...
    @property
    def variables(self) -> VariableDict: ...


class PrecisionModule(Module, Protocol):
    """A flax module that also states a precision, as `Float32Products` reads it."""

    @property
    def precision(self) -> PrecisionLike: ...


def computes_in(module: Module) -> Dtype | None:
    """The dtype `module` computes in: its `dtype`, or, where that is unset, the dtype its floating
    parameters promote to, which flax's layers compute in; None for a module holding no
    parameters yet (one initializing, or unbound)."""
    if module.dtype is not None or module.scope is None:
        return module.dtype
    held = [leaf for leaf in jax.tree.leaves(module.variables.get("params", {}))
            if jnp.issubdtype(leaf.dtype, jnp.floating)]
    return jnp.result_type(*held) if held else None


class Float32Products:
    """A model's precision as its layers take it (`float32_products`), over the dtype it computes
    in (`computes_in`). The module hands `products` to its layers in place of `precision`, which
    stays what the model was built with, so a record rebuilds it and a clone resolves anew. It
    mixes into an `nn.Module`."""

    dtype: Dtype | None
    precision: PrecisionLike

    @functools.cached_property
    def products(self: PrecisionModule) -> PrecisionLike:
        return float32_products(computes_in(self), self.precision)


def configurable_precision(precision: PrecisionLike) -> bool:
    """Whether `jax.default_matmul_precision` can name `precision`: one precision or dot algorithm
    preset, unset or default included; not a pair of precisions or a dot algorithm."""
    return len(precision_names(precision)) <= 1 and not isinstance(precision, jax.lax.DotAlgorithm)


def configured_precision(precision: PrecisionLike):
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


def jax_attention_honors(dtype: DTypeLike, precision: PrecisionLike) -> bool:
    """Whether jax.nn's xla attention over `dtype` queries multiplies at `precision`, under its
    `configured_precision`: float32 queries at any a config can name; a 16-bit query's logits take
    jax.nn's own dot algorithm for its dtype whatever the config says, so the default alone."""
    wide = jnp.dtype(dtype).itemsize >= 4
    return (configurable_precision if wide else asks_default_precision)(precision)


def operand_precision(dtype: Dtype | None, precision: PrecisionLike = None) -> jax.lax.PrecisionLike:
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
        return jax.lax.dot_general(
            lhs, rhs, dimension_numbers,
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


def _two_sum(left, right):
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
    total, dropped = _two_sum(left_hi, right_hi)
    return _two_sum(total, dropped + left_lo + right_lo)


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
    wide = values.astype(jnp.promote_types(values.dtype, jnp.float32))
    zero = jnp.zeros((), wide.dtype)
    hi, lo = jax.lax.reduce((wide, jnp.zeros_like(wide)), (zero, zero), compensated_add, axes)
    # A sum that reaches an infinity is the plain one, +-inf or NaN as `jnp.sum`'s is, where every
    # residual past the infinity is inf - inf.
    plain = jnp.sum(wide, axes)
    return jnp.where(jnp.isfinite(plain), hi + lo, plain).astype(values.dtype)


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


def _algorithm_operand(value: jax.Array) -> jax.Array:
    """`value` as a bf16-algorithm product reads it. GPU and TPU round inside
    the product, so the value passes as it is and no rounded copy is written;
    the CPU backend ignores the algorithm, so there it is rounded first."""
    return jax.lax.platform_dependent(
        value, cpu=lambda value: jnp.asarray(rounded_operand(value, jnp.bfloat16)),
        default=lambda value: value)


def head_dot_general(dtype: Dtype | None, precision: PrecisionLike = None):
    """A flax layer's `dot_general` for a vocabulary head: a result at
    least fp32 whose operands are the compute dtype's values, in both
    directions.

    `dtype` is the model's compute dtype, not the layer's: the layer
    promotes the states to at least fp32. Under bf16 compute at the default
    precision both operands multiply as bf16 and the logits are bf16 values
    (`head_product`); otherwise the product is the layer's own, in the
    operands' dtype.
    """
    resolved = operand_precision(dtype, precision)
    rounds = rounds_to_bf16(dtype, precision)

    def dot_general(lhs, rhs, dimension_numbers, precision=None,
                    preferred_element_type=None):
        del precision, preferred_element_type  # this head's policy, not the layer's
        if rounds:
            rhs = _algorithm_operand(rhs)
        logits = jax.lax.dot_general(lhs, rhs, dimension_numbers, precision=resolved,
                                     preferred_element_type=at_least_fp32(lhs.dtype))
        return bf16_logits(logits) if rounds else logits

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
        states = _algorithm_operand(hidden.astype(jnp.float32))
        logits = jnp.einsum(subscripts, states, _algorithm_operand(head),
                            precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
                            preferred_element_type=jnp.float32)
        return bf16_logits(logits)
    wide = at_least_fp32(hidden.dtype)
    return jnp.einsum(subscripts, hidden.astype(wide), head,
                      precision=precision, preferred_element_type=wide)


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
