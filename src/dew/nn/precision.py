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

import jax
import jax.numpy as jnp
from flax.typing import Dtype, PrecisionLike


def precision_names(precision: PrecisionLike) -> frozenset[str]:
    """The names a `PrecisionLike` spells, upper case and unordered.

    flax's alias is four shapes at once: None, a string, a
    `jax.lax.Precision`, or a pair of either for the two operands. A reader
    has to take whichever shape the caller wrote, so this reads the union
    rather than asking each member what it is: the enum carries the name, a
    string is the name, and the pair is both operands'.
    """
    if precision is None:
        return frozenset()
    written = (precision,) if isinstance(precision, str | jax.lax.Precision) else precision
    return frozenset((one.name if isinstance(one, jax.lax.Precision) else one).upper()
                     for one in written if one is not None)


def bf16_operand_precision(dtype: Dtype | None,
                           precision: PrecisionLike = None) -> PrecisionLike:
    """The precision that keeps a bf16 dot bf16 in both directions.

    A caller that asked for more than the default precision keeps what it
    asked for, and compute in anything but bf16 is left alone.
    """
    if precision_names(precision) - {'DEFAULT'} or dtype is None:
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
