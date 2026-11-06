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

A custom derivative is declared with its kind, which makes its primal and each of its rules one marked call
of that kind within the custom rule:
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
from jax.core import Tracer, eval_context
from jax.custom_derivatives import CustomVJPPrimal, SymbolicZero
from jax.extend import core as jax_core
from jax.extend.core import create_call_primitive

from dew.nn import programs

__all__ = [
    "Algebra",
    "KernelGlue",
    "Kind",
    "Protected",
    "Surrogate",
    "TangentKind",
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
    rule = jax.custom_jvp(bound_span(Surrogate(), forward, static=declared), nondiff_argnums=declared)

    def tangent(*arguments):
        *fixed, primals, tangents = arguments
        operands = list(primals)
        for index, value in zip(sorted(declared), fixed, strict=True):
            operands.insert(index, value)
        value = rule(*operands)
        return value, derivative(*fixed, primals, tangents, value)

    rule.defjvp(bound_span(Surrogate(), tangent, static=range(len(declared))))
    return rule


def declared_derivative[Value](kind: Kind, primal: Callable[..., Value], forward: Callable,
                               backward: Callable, *, nondiff_argnums: Sequence[int] = (),
                               symbolic_zeros: bool = False) -> Callable[..., Value]:
    """`primal` with a declared reverse-mode rule (`forward` saving its residuals, `backward` taking them and
    the cotangent, as `jax.custom_vjp.defvjp` takes them, with `symbolic_zeros` as it takes it), each of the
    three one marked call of `kind` (`bound_span`) within the custom rule: a kernel's glue (`KernelGlue`), a
    rule its numerics need as written (`Protected`), a derivative of Dew's choosing (`Surrogate`) or one
    written to restructure algebra (`Algebra`). The rule outermost, a differentiated call traces its forward
    alone, not its primal too, and each part is traced where it runs, so a value of that trace it closes over
    is its own (`marked`). Forward mode refuses it as a bare `jax.custom_vjp` does."""
    fixed = tuple(nondiff_argnums)
    rule = jax.custom_vjp(bound_span(kind, primal, static=fixed), nondiff_argnums=fixed)
    rule.defvjp(bound_span(kind, forward, static=fixed), bound_span(kind, backward, static=range(len(fixed))),
                symbolic_zeros=symbolic_zeros)
    return rule


def declared_tangent[Value](kind: TangentKind, primal: Callable[..., Value], tangent: Callable, *,
                            nondiff_argnums: Sequence[int] = (),
                            symbolic_zeros: bool = False) -> Callable[..., Value]:
    """`primal` with a declared tangent rule (`tangent(primals, tangents)` returning the primal's value and
    its tangent, as `jax.custom_jvp.defjvp` takes it, which reverse mode transposes; with `symbolic_zeros`,
    an operand no tangent reaches is a `SymbolicZero` there, as `defjvp` gives it), the primal and the rule
    each one marked call of `kind` (`bound_span`) within the custom rule, as `declared_derivative` declares
    its; a surrogate's tangent is `surrogate`'s, whose rule reads the surrogate's own value, so a second
    order meets the rule again (`TangentKind`)."""
    fixed = tuple(nondiff_argnums)
    rule = jax.custom_jvp(bound_span(kind, primal, static=fixed), nondiff_argnums=fixed)
    rule.defjvp(bound_span(kind, tangent, static=range(len(fixed))), symbolic_zeros=symbolic_zeros)
    return rule


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
        # A forward's primal under `symbolic_zeros` (`CustomVJPPrimal`) is its array, its flag bound in.
        held, fill = programs.holes((operands, bound.kwargs),
                                    lambda leaf: isinstance(_array(leaf), jax.Array),
                                    is_leaf=lambda leaf: isinstance(leaf, CustomVJPPrimal))
        if not held:
            return function(*args, **kwargs)

        def span(*arrays):
            given, named = fill([CustomVJPPrimal(array, leaf.perturbed) if isinstance(leaf, CustomVJPPrimal)
                                 else array for leaf, array in zip(held, arrays, strict=True)])
            positional, rest = [], iter(given)
            for position, arg in enumerate(bound.args):
                positional.append(arg if position in fixed else next(rest))
            return function(*positional, **named)

        return marked(kind, span)(*map(_array, held))
    return call


def _array(leaf):
    """A span's operand of `leaf`: a forward's primal's array (`CustomVJPPrimal.value`), else the leaf."""
    return leaf.value if isinstance(leaf, CustomVJPPrimal) else leaf


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

type TangentKind = Protected | KernelGlue | Algebra
"""The kinds a tangent rule is declared as (`declared_tangent`): a surrogate's forward mode is `surrogate`'s
alone, the one route whose rule reads the surrogate's own value."""

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
        # A tangent rule's known zero (`SymbolicZero`) among its outputs is no value of the span: it is
        # returned beside the span's outputs, in its place.
        returned = {}

        def span(*leaves):
            produced, returned["structure"] = jax.tree.flatten(
                function(*jax.tree.unflatten(structure, leaves)),
                is_leaf=lambda leaf: isinstance(leaf, SymbolicZero))
            returned["zeros"] = {index: leaf for index, leaf in enumerate(produced)
                                 if isinstance(leaf, SymbolicZero)}
            return [leaf for leaf in produced if not isinstance(leaf, SymbolicZero)]

        closed = jax.make_jaxpr(span)(*flat)
        # A value of the enclosing trace the function closes over (a kernel's mask, a reference's operand) is
        # one of the call's operands, leading its arguments, as a closed-over value is no constant there.
        held = [isinstance(const, Tracer) for const in closed.consts]
        if any(held):
            lifted = [const for const, is_held in zip(closed.consts, held, strict=True) if is_held]
            closed = closed.replace(
                constvars=[var for var, is_held in zip(closed.constvars, held, strict=True) if not is_held],
                invars=[*(var for var, is_held in zip(closed.constvars, held, strict=True) if is_held),
                        *closed.invars],
                consts=[const for const, is_held in zip(closed.consts, held, strict=True) if not is_held])
            flat = [*lifted, *flat]
        outputs = iter(marked_call_p.bind(*flat, call_jaxpr=closed, kind=kind))
        zeros = returned["zeros"]
        return jax.tree.unflatten(returned["structure"], [
            zeros[index] if index in zeros else next(outputs)
            for index in range(returned["structure"].num_leaves)])
    return call

