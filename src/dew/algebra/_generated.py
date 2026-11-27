"""Case keys and dispatch for checked-in generated modules, without loading the dev-time registry."""

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from jax.core import ShapedArray


def case_of(operands: Sequence) -> tuple:
    """The flat, unplaced array ABI admitted by the generator."""
    return tuple((tuple(value.shape), jnp.dtype(value.dtype).name) for value in operands)


def refused(operands: Sequence) -> str | None:
    """Why `operands` are not the generated ABI, strongly typed unplaced arrays alone: a Python number, a
    weakly typed value or a placed one keys no case as the written math reads it. None where they are."""
    for operand in operands:
        if not (hasattr(operand, "shape") and hasattr(operand, "dtype")):
            return f"an operand of type {type(operand).__name__}, not an array"
        aval = jax.typeof(operand)
        if not isinstance(aval, ShapedArray) or aval.weak_type or not aval.sharding.mesh.empty:
            return f"an operand {aval}, not a strongly typed unplaced array"
    return None


def call(forms: dict[tuple, Callable[..., tuple]], operands: tuple, policy: str, target: str, *,
         single: bool):
    """Run only the checked form for the exact policy, target, shape and dtype key."""
    outputs = forms[policy, target, case_of(operands)](*operands)
    return outputs[0] if single else outputs
