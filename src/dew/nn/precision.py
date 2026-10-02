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

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax.typing import Dtype, PrecisionLike
from jax.typing import DTypeLike


def at_least_fp32(dtype: DTypeLike | None) -> jnp.dtype:
    """The dtype arithmetic Dew keeps from rounding below float32 runs in:
    float32 for float32, every narrower float and a module's unset dtype
    (None), and `dtype` itself where it is wider. Norm statistics, softmaxes,
    rotary and time-embedding angles, the vocabulary head's product and loss
    reductions read it, so a bfloat16 model reduces in float32 and a float64
    model computes in float64 throughout, not with its float32 twin's
    roundings."""
    return jnp.promote_types(jnp.float32 if dtype is None else dtype, jnp.float32)


def precision_names(precision: PrecisionLike) -> frozenset[str]:
    """The canonical names a `PrecisionLike` spells, unordered.

    flax's alias is None, a string, a `jax.lax.Precision`, or a pair of either.
    A string goes through `jax.lax.Precision`, so 'tensorfloat32' is 'HIGH' and
    'bfloat16' is 'DEFAULT'; a dot algorithm keeps its own name.
    """
    if precision is None:
        return frozenset()
    written = (precision,) if isinstance(
        precision, str | jax.lax.Precision | jax.lax.DotAlgorithmPreset) else precision
    return frozenset((jax.lax.Precision(one) if isinstance(one, str) else one).name
                     for one in written if one is not None)


def asks_default_precision(precision: PrecisionLike, *, configured: bool = False) -> bool:
    """Whether `precision` asks for no more than the default. With
    `configured`, an unset precision reads `jax_default_matmul_precision`,
    the default a product without one gets."""
    names = precision_names(precision)
    if not names and configured:
        names = precision_names(jax.config.jax_default_matmul_precision)
    return names <= {'DEFAULT'}


def bf16_operand_precision(dtype: Dtype | None,
                           precision: PrecisionLike = None) -> jax.lax.PrecisionLike:
    """The precision that keeps a bf16 dot bf16 in both directions.

    A caller that asked for more than the default precision keeps what it
    asked for, and compute in anything but bf16 is left alone. So is a GPU
    older than sm80, which rejects the algorithm at run time
    (`dew.nn.kernels.generation.BF16_GPU`).
    """
    if not asks_default_precision(precision) or dtype is None:
        return precision
    if jnp.dtype(dtype) != jnp.bfloat16:
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
            precision=bf16_operand_precision(lhs.dtype, requested),
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
    multiplies bf16 operands (`bf16_operand_precision`)."""
    return bf16_operand_precision(dtype, precision) is jax.lax.DotAlgorithmPreset.BF16_BF16_F32


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
    resolved = bf16_operand_precision(dtype, precision)
    rounds = rounds_to_bf16(dtype, precision)

    def dot_general(lhs, rhs, dimension_numbers, precision=None,
                    preferred_element_type=None):
        del precision, preferred_element_type  # this head's policy, not the layer's
        if rounds:
            rhs = _algorithm_operand(rhs)
        logits = jax.lax.dot_general(lhs, rhs, dimension_numbers, precision=resolved,
                                     preferred_element_type=at_least_fp32(lhs.dtype))
        return rounded_to(logits, jnp.bfloat16) if rounds else logits

    return dot_general


def head_product(subscripts: str, hidden: jax.Array, head: jax.Array,
                 precision: PrecisionLike = None) -> jax.Array:
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
    `matmul_precision` "highest", whose head is fp32."""
    if rounds_to_bf16(hidden.dtype, precision):
        logits = jnp.einsum(subscripts, hidden.astype(jnp.float32), _algorithm_operand(head),
                            precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
                            preferred_element_type=jnp.float32)
        return rounded_to(logits, jnp.bfloat16)
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
