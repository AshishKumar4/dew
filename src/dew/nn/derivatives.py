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

A custom derivative is declared with its kind, which makes it one marked call of that kind:
`declared_derivative` (a reverse-mode rule, as `jax.custom_vjp` takes one) and `declared_tangent` (a tangent
rule, as `jax.custom_jvp` takes one), each of any kind, and `surrogate` (a value and its declared derivative).
Each lane moves its sites onto them (rows 4026 to 4030); then the slop check refuses `jax.custom_vjp` and
`jax.custom_jvp` outside this module.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
from collections.abc import Callable, Sequence

import jax
from jax.core import eval_context
from jax.extend import core as jax_core
from jax.extend.core import create_call_primitive

__all__ = [
    "Algebra",
    "KernelGlue",
    "Kind",
    "Protected",
    "Surrogate",
    "bound_span",
    "declared_derivative",
    "declared_tangent",
    "marked",
    "marked_call_p",
    "surrogate",
]


def surrogate[Value](forward: Callable[..., Value], derivative: Callable[..., Value], *,
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
    return bound_span(Surrogate(), rule, static=declared)


def declared_derivative[Value](kind: Kind, primal: Callable[..., Value], forward: Callable,
                               backward: Callable, *, nondiff_argnums: Sequence[int] = (),
                               symbolic_zeros: bool = False) -> Callable[..., Value]:
    """`primal` with a declared reverse-mode rule (`forward` saving its residuals, `backward` taking them and
    the cotangent, as `jax.custom_vjp.defvjp` takes them, with `symbolic_zeros` as it takes it), one marked
    call of `kind` (`bound_span`): a kernel's glue (`KernelGlue`), a rule its numerics need as written
    (`Protected`), a derivative of Dew's choosing (`Surrogate`) or one written to restructure algebra
    (`Algebra`). Forward mode refuses it as a bare `jax.custom_vjp` does."""
    rule = jax.custom_vjp(primal, nondiff_argnums=tuple(nondiff_argnums))
    rule.defvjp(forward, backward, symbolic_zeros=symbolic_zeros)
    return bound_span(kind, rule, static=nondiff_argnums)


def declared_tangent[Value](kind: Kind, primal: Callable[..., Value], tangent: Callable, *,
                            nondiff_argnums: Sequence[int] = (),
                            symbolic_zeros: bool = False) -> Callable[..., Value]:
    """`primal` with a declared tangent rule (`tangent(primals, tangents)` returning the primal's value and
    its tangent, as `jax.custom_jvp.defjvp` takes it, which reverse mode transposes; with `symbolic_zeros`,
    an operand no tangent reaches is a `SymbolicZero` there, as `defjvp` gives it), one marked call of
    `kind` (`bound_span`), as `declared_derivative` takes it."""
    rule = jax.custom_jvp(primal, nondiff_argnums=tuple(nondiff_argnums))
    rule.defjvp(tangent, symbolic_zeros=symbolic_zeros)
    return bound_span(kind, rule, static=nondiff_argnums)


def bound_span[Result](kind: Kind, function: Callable[..., Result], *,
                       static: Sequence[int] = ()) -> Callable[..., Result]:
    """`function` as one marked call of `kind` (`marked`) over its arrays, passed by position or by name: its
    other arguments (a static factor, a dtype, a Python number, an option) and those at its `static` positions
    (a custom derivative's `nondiff_argnums`, arrays among them, which its rule takes as constants) bound in,
    so a constant still splits where it is known; a call of no array, `function` itself."""
    signature = inspect.signature(function)
    fixed = frozenset(static)

    @functools.wraps(function)
    def call(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        operands = [arg for position, arg in enumerate(bound.args) if position not in fixed]
        leaves, structure = jax.tree.flatten((operands, bound.kwargs))
        traced = [index for index, leaf in enumerate(leaves) if isinstance(leaf, jax.Array)]
        if not traced:
            return function(*args, **kwargs)

        def span(*arrays):
            filled = list(leaves)
            for index, array in zip(traced, arrays, strict=True):
                filled[index] = array
            given, named = jax.tree.unflatten(structure, filled)
            positional, rest = [], iter(given)
            for position, arg in enumerate(bound.args):
                positional.append(arg if position in fixed else next(rest))
            return function(*positional, **named)

        return marked(kind, span)(*(leaves[index] for index in traced))
    return call


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


def _evaluating() -> bool:
    """Whether a call here runs eagerly: the current trace is JAX's own evaluation (`jax.core.eval_context`'s)
    and not a trace that records a program (a jit's, a `make_jaxpr`'s, a transform's), whatever the
    call's arguments are."""
    here = jax_core.find_top_trace(())
    with eval_context():
        return here is jax_core.find_top_trace(())


def marked[Result](kind: Kind, function: Callable[..., Result]) -> Callable[..., Result]:
    """`function` as one marked call of `kind` (`marked_call_p`) over its arguments, a tree of arrays: it
    computes what `function` does, and every transform of it keeps a marked call of `kind`. An eager call
    (`_evaluating`), which no program records, is `function`'s own: traced, every such call traced a program
    a step's steady state paid each step (a decision task's eager scores, train 586)."""
    def call(*arguments):
        if _evaluating():
            return function(*arguments)
        flat, structure = jax.tree.flatten(arguments)
        closed, shaped = jax.make_jaxpr(lambda *leaves: function(*jax.tree.unflatten(structure, leaves)),
                                        return_shape=True)(*flat)
        outputs = marked_call_p.bind(*flat, call_jaxpr=closed, kind=kind)
        return jax.tree.unflatten(jax.tree.structure(shaped), outputs)
    return call

