"""Case keys and dispatch for checked-in generated modules, without loading the dev-time registry."""

from collections.abc import Callable, Sequence

import jax.numpy as jnp


def case_of(operands: Sequence) -> tuple:
    """The flat, unplaced array ABI admitted by the generator."""
    return tuple((tuple(value.shape), jnp.dtype(value.dtype).name) for value in operands)


def call(forms: dict[tuple, Callable[..., tuple]], operands: tuple, policy: str, target: str, *, single: bool):
    """Run only the checked form for the exact policy, target, shape and dtype key."""
    outputs = forms[policy, target, case_of(operands)](*operands)
    return outputs[0] if single else outputs
