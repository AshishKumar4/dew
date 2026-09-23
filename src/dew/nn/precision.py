"""The precision a bf16 model's fp32 boundaries multiply at.

A model's last matmul before a loss - the patch output head, the vocabulary
head - keeps an fp32 result on purpose: the loss is computed in fp32, and a
result rounded to bf16 would carry no more than bf16 tells. Asking for that
result with `preferred_element_type=jnp.float32` leaves the operands in the
compute dtype, which is what a bf16 run wants, but it also makes the
cotangent of that dot fp32, and the mixed f32 x bf16 products of the
backward pass then run at the fp32 rate - twice the forward's matmuls at
half the speed.

XLA's dot algorithm names the arithmetic rather than the storage:
`BF16_BF16_F32` converts its operands to bf16, multiplies them and
accumulates in fp32, whatever dtype the operands are stored as. jax carries
`precision` into its transpose rules, so an algorithm set on the forward dot
reaches every product of its backward. jax's own xla attention sets this
preset for exactly this reason (jax-ml/jax#24047).
"""

import functools

import jax
import jax.numpy as jnp
from flax.typing import Dtype, PrecisionLike


def precision_names(precision: PrecisionLike) -> frozenset[str]:
    """The canonical names a `PrecisionLike` spells, unordered.

    flax's alias is four shapes at once: None, a string, a
    `jax.lax.Precision`, or a pair of either for the two operands. A reader
    has to take whichever shape the caller wrote, so this reads the union
    rather than asking each member what it is. A string goes through
    `jax.lax.Precision` itself, so 'tensorfloat32' is 'HIGH' and 'bfloat16'
    is 'DEFAULT' whichever way a caller spells them; a dot algorithm keeps
    its own name.
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
    asked for, and compute in anything but bf16 is left alone.
    """
    if not asks_default_precision(precision) or dtype is None:
        return precision
    if jnp.dtype(dtype) != jnp.bfloat16:
        return precision
    return jax.lax.DotAlgorithmPreset.BF16_BF16_F32


def fp32_result_dot_general(precision: PrecisionLike = None):
    """A flax layer's `dot_general` for a head whose result stays fp32.

    The layer has already promoted both operands when this runs - to its own
    `dtype`, or to the activations' where it carries none - so the compute
    dtype is the operands' own. Only the accumulation and the result are
    fp32.
    """
    requested = precision

    def dot_general(lhs, rhs, dimension_numbers, precision=None,
                    preferred_element_type=None):
        del precision, preferred_element_type  # this head's policy, not the layer's
        return jax.lax.dot_general(
            lhs, rhs, dimension_numbers,
            precision=bf16_operand_precision(lhs.dtype, requested),
            preferred_element_type=jnp.float32)

    return dot_general


def scaled(x: jax.Array, factor: float) -> jax.Array:
    """`x * factor` with the factor kept in fp32 and only the product rounded
    to `x`'s dtype, which is what torch's `bf16_tensor * python_float` does
    (fp32 opmath) and what Gemma's `embed_scale` here does. Rounding 0.22 to
    bf16 first would shrink every product by a systematic 0.12%."""
    if factor == 1.0:
        return x
    return (x.astype(jnp.promote_types(x.dtype, jnp.float32)) * factor).astype(x.dtype)

@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def rounded_operand(x: jax.Array, dtype: Dtype) -> jax.Array:
    """The value `x` takes in `dtype`, held in `x`'s own dtype, with a
    straight-through tangent: the rounding is the forward's, and a gradient
    through it keeps its dtype and is never rounded a second time."""
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
    """A flax layer's `dot_general` for a bf16 vocabulary head
    (`CausalTransformer.bf16_head`): an fp32 result whose operands are the
    bf16 compute dtype's values, in both directions.

    `dtype` is the model's compute dtype, not the layer's: the layer
    promotes the states to fp32. Under any other compute dtype, or a
    precision above the default, the product is the layer's own fp32 one.
    """
    resolved = bf16_operand_precision(dtype, precision)
    rounds = rounds_to_bf16(dtype, precision)

    def dot_general(lhs, rhs, dimension_numbers, precision=None,
                    preferred_element_type=None):
        del precision, preferred_element_type  # this head's policy, not the layer's
        if rounds:
            rhs = _algorithm_operand(rhs)
        return jax.lax.dot_general(lhs, rhs, dimension_numbers, precision=resolved,
                                   preferred_element_type=jnp.float32)

    return dot_general


def head_product(subscripts: str, hidden: jax.Array, head: jax.Array,
                 precision: PrecisionLike = None, *, bf16: bool = False) -> jax.Array:
    """`jnp.einsum(subscripts, hidden, head)` as a vocabulary head computes
    it, with fp32 accumulation and an fp32 result.

    By default the states are widened to fp32 and multiply the head as
    stored. With `bf16` and bf16 states at the default precision, both
    operands multiply as bf16 (`head_dot_general`'s arithmetic)."""
    if bf16 and rounds_to_bf16(hidden.dtype, precision):
        return jnp.einsum(subscripts, hidden.astype(jnp.float32), _algorithm_operand(head),
                          precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
                          preferred_element_type=jnp.float32)
    return jnp.einsum(subscripts, hidden.astype(jnp.float32), head,
                      precision=precision, preferred_element_type=jnp.float32)
