"""Declared derivatives: a function whose derivative is not its value's own.

A surrogate (`surrogate`) evaluates one function and differentiates as a declared rule: a straight-through
rounding, a quantized product differentiated as its float one, an estimate of a gradient no autodiff of
the value gives (node perturbation), or a gradient a learning rule supplies (`Objective.with_gradients`).
Each is a derivative of Dew's choosing, not of the value's, so it is one declaration here, which the seam
registry reads as a surrogate (tests/test_derivative_seams.py) and no rewrite of the program's algebra
replaces with the value's own derivative.

A marked span (`marked`) is a computation carried as one call of its kind (`Kind`), which every transform
keeps a call of that kind, its tangent and cotangent too (docs/design/algebra.md §6.10). The algebra pass
reads its kind (`dew.algebra.program`): a protected span (`Protected`, an error-free transform's) is kept as
it is, a surrogate's bodies (`Surrogate`) and a kernel derivative's glue (`KernelGlue`) are rewritten within
their call, and a derivative written to restructure algebra (`Algebra`) opens into the region around it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import TypeVar

import jax
from jax.extend.core import create_call_primitive

__all__ = ["Algebra", "KernelGlue", "Kind", "Protected", "Surrogate", "marked", "marked_call_p", "surrogate"]

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


@dataclasses.dataclass(frozen=True)
class Protected:
    """A span the algebra pass keeps as it is: an error-free transform's (`dew.nn.precision`), whose
    residual a reassociation would cancel."""


@dataclasses.dataclass(frozen=True)
class Surrogate:
    """A surrogate's bodies (`surrogate`): rewritten within, never related to autodiff's derivative of its
    forward."""


@dataclasses.dataclass(frozen=True)
class KernelGlue:
    """A kernel derivative's XLA around its launches: rewritten as any other math, its launches opaque."""


@dataclasses.dataclass(frozen=True)
class Algebra:
    """A derivative written only to restructure algebra, which the pass rewrites freely and the engine
    retires once it derives the form itself."""


type Kind = Protected | Surrogate | KernelGlue | Algebra
"""What a marked span is to the algebra pass, read as an object, never a name."""

marked_call_p = create_call_primitive("marked_call")
"""A marked span: a call of its body's jaxpr (`call_jaxpr`) under its `kind`, whose JVP, linearization,
transpose, batching, partial evaluation and checkpointing are JAX's call primitive's, each a marked call of
the same kind; lowered by inlining its body, so it costs nothing compiled, inside a Pallas kernel's body too
(Dew's JAX build lowers a call primitive there as its body)."""


def marked[Result](kind: Kind, function: Callable[..., Result]) -> Callable[..., Result]:
    """`function` as one marked call of `kind` (`marked_call_p`) over its arguments, a tree of arrays: it
    computes what `function` does, and every transform of it keeps a marked call of `kind`."""
    def call(*arguments):
        flat, structure = jax.tree.flatten(arguments)
        closed, shaped = jax.make_jaxpr(lambda *leaves: function(*jax.tree.unflatten(structure, leaves)),
                                        return_shape=True)(*flat)
        outputs = marked_call_p.bind(*flat, call_jaxpr=closed, kind=kind)
        return jax.tree.unflatten(jax.tree.structure(shaped), outputs)
    return call
