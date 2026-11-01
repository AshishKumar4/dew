"""Declared derivatives: a function whose derivative is not its value's own.

A surrogate (`surrogate`) evaluates one function and differentiates as a declared rule: a straight-through
rounding, a quantized product differentiated as its float one, an estimate of a gradient no autodiff of
the value gives (node perturbation), or a gradient a learning rule supplies (`Objective.with_gradients`).
Each is a derivative of Dew's choosing, not of the value's, so it is one declaration here, which the seam
registry reads as a surrogate (tests/test_derivative_seams.py) and no rewrite of the program's algebra
replaces with the value's own derivative.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

import jax

__all__ = ["surrogate"]

Value = TypeVar("Value")
"""A surrogate's value, a tree of arrays, and its tangent's type."""


def surrogate(forward: Callable[..., Value], derivative: Callable[..., Value], *,
              nondiff_argnums: Sequence[int] = ()) -> Callable[..., Value]:
    """The function whose value is `forward(*operands)`, exactly, and whose derivative is the declared rule
    `derivative(primals, tangents, value)`: the output's tangent from the differentiable operands, their
    tangents and the value, linear in the tangents, so reverse mode transposes it and a higher order
    differentiates it. The arguments at `nondiff_argnums` (static, as a dtype) take no tangent; each reaches
    both `forward` and, ahead of the primals in the order of their positions, `derivative`, as JAX's
    `custom_jvp` passes them whatever order `nondiff_argnums` names them in.

    The value a derivative reads is the surrogate's own, so a derivative of a derivative meets the
    declared rule again wherever it differentiates the value (a straight-through rounding's tangent is
    never rounded at a second order)."""
    declared = tuple(nondiff_argnums)
    rule = jax.custom_jvp(forward, nondiff_argnums=declared)

    def tangent(*arguments):
        *fixed, primals, tangents = arguments
        operands = list(primals)
        for index, value in zip(sorted(declared), fixed, strict=True):
            operands.insert(index, value)
        value = rule(*operands)
        return value, derivative(*fixed, primals, tangents, value)

    rule.defjvp(tangent)
    return rule
