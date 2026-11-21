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
from collections.abc import Callable, Collection, Sequence

import jax
import jax.numpy as jnp
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
    "Rare",
    "Surrogate",
    "TangentKind",
    "WalkStep",
    "bound_span",
    "contracted",
    "declared_derivative",
    "declared_tangent",
    "marked",
    "marked_call_p",
    "portable_formed",
    "surrogate",
    "with_gradients_formed",
    "with_gradients_formed_at",
    "with_tangents_formed",
    "with_tangents_formed_at",
]


def with_gradients_formed[Input, Gradients](d_input: Input, gradients: Gradients
                                            ) -> tuple[Input, Gradients]:
    """A reverse mode's input cotangent, returned only with the parameter gradients it forms: `gradients`,
    each formed in the dtype it is stored in, and `d_input` leave through one `jax.lax.optimization_barrier`
    (the primitive `jax.checkpoint` binds a layer's recomputation to its incoming cotangent with), so the
    layer below, which reads `d_input` at once, waits on them and XLA's scheduler puts no parameter gradient,
    nor the cotangents it reads, off to the step's end. Every reverse mode that forms a parameter gradient
    calls it (`gemm`'s, the projection XLA's product takes, the conv's, the SSD skip's, the selective scan's).

    Row 6027: a Qwen3.5-4B four-row step, compile-only on a GH200, held 80.50 GB of temporaries without it
    (XLA's dot of each layer's up and gate kernel gradients put off to the end, every layer's SwiGLU backward
    pair held to it) and 22.36 GB with it, its schedule 55 GB under the best XLA's List, DFS and post-order
    memory schedulers built unordered. Bound at a layer's boundary instead, over its variables' cotangents,
    it held 28.35 GB: there it frees the optimizer's per-parameter update to move into the backward and hold
    its momenta to the update's stacked Newton-Schulz at the step's end, and leaves each gradient in its
    accumulation dtype until then. Its guards are tests/step_memory.py's (the step users run over every
    registered mixer, tests/test_step_memory_*.py), which a reverse mode forming a parameter gradient
    without it fails, and tests/test_declared_cells_memory.py's stack of every kernel family's products.
    Only floating arrays enter the barrier: an integer operand's zero cotangent (`float0`) and an unreached
    one (None) pass through as they are, and where no gradient is a floating array (an unhooked call's, a
    sinkless attention's) there is nothing to form and the cotangents leave as they are, fusing into their
    readers. A projection's rule holds its entry too (`with_cotangent_arrived`, `portable_formed`)."""
    if not any(_floating(leaf) for leaf in jax.tree.leaves(gradients)):
        return d_input, gradients
    return _held(d_input, gradients)


def with_cotangent_arrived[Cotangent, Operands](cotangent: Cotangent, operands: Operands
                                               ) -> tuple[Cotangent, Operands]:
    """The rule's entry (`with_gradients_formed`'s exit beside it): a reverse mode reads its operands only
    once its cotangent has arrived, both through one barrier, so XLA computes nothing of a reverse mode from
    its operands alone (a projection's kernel transposed for its input gradient) ahead of the cotangent,
    where it held one of them a layer to the backward's start (a Mamba-2 decoder's in-projection, 0.8 MB a
    layer: its temporaries' growth from two blocks to four 1.87 MB past what its forward saves, 0.23 MB under
    it with the entry; a four-layer Muon LM's measured need 6.34 MB joined and 4.27 MB parted with it, 7.76 MB
    either way without, so its planner could no longer part its stacks: row 6740's review in train 1113)."""
    return _held(cotangent, operands)


def _held(first, second):
    """`first` and `second` through one `jax.lax.optimization_barrier`, their floating arrays alone."""
    leaves, tree = jax.tree.flatten((first, second))
    floating = [index for index, leaf in enumerate(leaves) if _floating(leaf)]
    held = jax.lax.optimization_barrier(tuple(leaves[index] for index in floating))
    for index, leaf in zip(floating, held, strict=True):
        leaves[index] = leaf
    return jax.tree.unflatten(tree, leaves)


def with_tangents_formed[Input, Parameters](t_input: Input, t_parameters: Parameters
                                            ) -> tuple[Input, Parameters]:
    """The rule's dual (`with_gradients_formed`), for a reverse mode JAX derives by transposing a tangent
    rule (a `declared_tangent`, a custom JVP, a surrogate's tangent): where both the input and a parameter
    move, their live tangents enter the rule's linear part through the same barrier, whose transpose returns
    the input's cotangent only with the parameters' gradients formed; forward mode passes them through
    unchanged. Where either side moves nothing (a frozen parameter's, an input no derivative reaches: no
    floating array, its tangent a symbolic zero) there is nothing to tie and the tangents pass as they are,
    so a tangent rule takes symbolic zeros (`symbolic_zeros=True`): a zero JAX instantiates is a constant
    beside the tangents, which the barrier's transpose cannot take."""
    if not (any(_floating(leaf) for leaf in _leaves(t_input))
            and any(_floating(leaf) for leaf in _leaves(t_parameters))):
        return t_input, t_parameters
    return _held(t_input, t_parameters)


def with_gradients_formed_at(cotangents: tuple, parameters: Collection[int]) -> tuple:
    """A reverse mode's cotangents by operand (`cotangents`), those at `parameters` its parameters' (an
    attention's bias and sinks, a recurrence's skip and rate), through the rule
    (`with_gradients_formed`): the inputs' leave only with the parameters' formed; where no parameter's
    cotangent is a floating array (a sinkless, biasless call's) they leave as they are."""
    inputs, formed = with_gradients_formed(*_apart(cotangents, parameters))
    return _together(inputs, formed, parameters)


def with_tangents_formed_at(tangents: tuple, parameters: Collection[int]) -> tuple:
    """A tangent rule's tangents by operand, those at `parameters` its parameters', through the rule's dual
    (`with_tangents_formed`)."""
    inputs, formed = with_tangents_formed(*_apart(tangents, parameters))
    return _together(inputs, formed, parameters)


def _apart(values: tuple, parameters: Collection[int]) -> tuple[tuple, tuple]:
    """`values` the rest's and the `parameters` positions', in order."""
    return (tuple(value for index, value in enumerate(values) if index not in parameters),
            tuple(value for index, value in enumerate(values) if index in parameters))


def _together(inputs: tuple, formed: tuple, parameters: Collection[int]) -> tuple:
    """`_apart`'s inverse: the values back at their positions."""
    rest, held = iter(inputs), iter(formed)
    return tuple(next(held) if index in parameters else next(rest)
                 for index in range(len(inputs) + len(formed)))


def _leaves(tree) -> list:
    """`tree`'s leaves, a symbolic zero one of them."""
    from jax.custom_derivatives import SymbolicZero

    return jax.tree.leaves(tree, is_leaf=lambda leaf: isinstance(leaf, SymbolicZero))


def _floating(leaf) -> bool:
    """Whether `leaf` is an array of a floating dtype, which the rule's barrier holds: no symbolic zero (a
    tangent no derivative moves, `jax.custom_jvp`'s `SymbolicZero`), which carries no value to hold."""
    from jax.custom_derivatives import SymbolicZero

    dtype = getattr(leaf, "dtype", None)
    return not isinstance(leaf, SymbolicZero) and dtype is not None and jnp.issubdtype(dtype, jnp.inexact)


def portable_formed[Value](portable: Callable[..., Value], /) -> Callable[..., Value]:
    """`portable`, an XLA program whose last operand is a parameter, or a tree of them (a projection's kernel
    and its bias, a norm's weight, an attention's sinks), its value and every derivative XLA's of it
    (`generation.linear_tangent`), its reverse mode the rule's: it reads its operands only once its
    cotangent has arrived (`with_cotangent_arrived`) and returns the inputs' cotangents only with the
    parameters' gradients formed (`with_gradients_formed`), each formed only where a reverse pass
    differentiates it (`live_cotangents`: a frozen operand's never is). Its forward saves nothing; its
    backward recomputes `portable` as its VJP reads it."""
    from dew.nn.kernels.generation import linear_tangent

    def backward(saved, operands, cotangent, live):
        del saved
        count = len(jax.tree.leaves(operands))
        first = count - len(jax.tree.leaves(operands[-1]))
        # `portable` of the live leaves, the others held as they are.
        # Every operand the VJP reads (a frozen kernel the input's cotangent reads transposed among them)
        # read once its cotangent has arrived; `live` chooses the cotangents formed.
        cotangent, operands = with_cotangent_arrived(cotangent, operands)
        moving, fill = programs.holes(operands, [index in live for index in range(count)])
        pulled = jax.vjp(lambda *values: portable(*fill(values)), *moving)[1](cotangent)
        inputs, formed = with_gradients_formed(
            tuple(each for index, each in zip(live, pulled, strict=True) if index < first),
            tuple(each for index, each in zip(live, pulled, strict=True) if index >= first))
        return (*inputs, *formed)

    return contracted("test_a_portable_formed_rule_holds_every_order")(linear_tangent(
        lambda *operands: (portable(*operands), ()), backward, portable, kind=KernelGlue(),
        live_cotangents=True))


def contracted[Function](test: str, /) -> Callable[[Function], Function]:
    """A custom derivative's contract, declared where the derivative is made: `test`, the name of the one
    test function of tests/test_contracts_*.py that holds every order of it finite against its float64
    truth and an independent plain formula. The function it decorates or wraps is returned unchanged;
    tools/lint_slop.py (SLOP028) resolves the name and refuses a custom derivative without one. Its kind
    is its maker's (`Kind`), declared there alone."""
    return lambda function: function


def surrogate[Value](forward: Callable[..., Value], derivative: Callable[..., Value], *,
                     nondiff_argnums: Sequence[int] = (), symbolic_zeros: bool = False
                     ) -> Callable[..., Value]:
    """The function whose value is `forward(*operands)`, exactly, and whose derivative is the declared rule
    `derivative(primals, tangents, value)`: the output's tangent from the differentiable operands, their
    tangents and the value, linear in the tangents, so reverse mode transposes it and a higher order
    differentiates it. The arguments at `nondiff_argnums` (static, as a dtype) take no tangent; each reaches
    both `forward` and, ahead of the primals in the order of their positions, `derivative`, as JAX's
    `custom_jvp` passes them whatever order `nondiff_argnums` names them in.

    The value a derivative reads is the surrogate's own, so a derivative of a derivative meets the
    declared rule again wherever it differentiates the value (a straight-through rounding's tangent is
    never rounded at a second order). With `symbolic_zeros`, an operand no derivative moves takes a
    `SymbolicZero` tangent, as `jax.custom_jvp.defjvp` passes it."""
    declared = tuple(nondiff_argnums)
    rule = contracted("test_a_surrogate_holds_every_order")(
        jax.custom_jvp(bound_span(Surrogate(), forward, static=declared), nondiff_argnums=declared))

    def tangent(*arguments):
        *fixed, primals, tangents = arguments
        operands = list(primals)
        for index, value in zip(sorted(declared), fixed, strict=True):
            operands.insert(index, value)
        value = rule(*operands)
        return value, derivative(*fixed, primals, tangents, value)

    rule.defjvp(bound_span(Surrogate(), tangent, static=range(len(declared))), symbolic_zeros=symbolic_zeros)
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
    rule = contracted("test_a_declared_derivative_holds_every_order")(
        jax.custom_vjp(bound_span(kind, primal, static=fixed), nondiff_argnums=fixed))
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
    order meets the rule again (`TangentKind`). A protected kind that states the operation its primal
    computes (`Protected.computes`) states it of the primal alone: the rule's span is protected stating
    nothing, as partial evaluation splits it into pieces that compute no such one operation."""
    fixed = tuple(nondiff_argnums)
    # A span stating its operation is bound with the rule it is declared with (`Protected.declared`).
    stated = (dataclasses.replace(kind, declared=tangent) if isinstance(kind, Protected) and kind.computes
              else kind)
    rule = contracted("test_a_declared_derivative_holds_every_order")(
        jax.custom_jvp(bound_span(stated, primal, static=fixed), nondiff_argnums=fixed))
    ruled = Protected() if isinstance(kind, Protected) else kind
    rule.defjvp(bound_span(ruled, tangent, static=range(len(fixed))), symbolic_zeros=symbolic_zeros)
    return rule


def custom_rule(value) -> tuple[str, dict] | None:
    """`value`, a function with JAX's own custom derivative, as a trace of it reads it, by name: a custom
    JVP's function, its rule (`defjvps`' a closure over each argument's), the arguments it does not
    differentiate and its zeros' form; a custom VJP's function, its rules (None before `defvjp`, which a call
    refuses) and the options a trace reads. None for any other value. This module names JAX's custom
    derivatives (SLOP026), so the statement of a value (`dew._stated`) reads one of the user's here (row
    10087: an arm holding a test's own rule had no statement and ran unprepared)."""
    if isinstance(value, jax.custom_jvp):
        return "custom jvp", {"function": value.fun, "jvp": value.jvp,
                              "nondiff_argnums": value.nondiff_argnums,
                              "symbolic_zeros": value.symbolic_zeros}
    if isinstance(value, jax.custom_vjp):
        return "custom vjp", {"function": value.fun, "fwd": value.fwd, "bwd": value.bwd,
                              "nondiff_argnums": value.nondiff_argnums,
                              "symbolic_zeros": value.symbolic_zeros, "optimize_remat": value.optimize_remat}
    return None


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
    residual a reassociation would cancel; or, where it names what it `computes`, one operation of Dew's
    in at least float32 and rounded once to its operand's dtype (`dew.nn.precision.logistic_value`, Dew's
    one logistic, `dew.nn.activations.logistic`), which the algebra's normal form reads as that operation
    (`dew.algebra.normal_form`), its arithmetic Dew's wherever it is fused. Such a span declared with a
    derivative of its own (`declared_tangent`) is `declared` with that rule, set there alone, so a reader
    that takes the span for its operation takes its derivative for the operation's only where the rule is
    the one Dew's own declaration of it is declared with (`dew.algebra.recognize`)."""
    computes: Callable | None = None
    declared: Callable | None = None


def stated_operation(equation) -> Callable | None:
    """The operation a protected span states it computes (`Protected.computes`: Dew's one logistic its
    `precision.logistic_value`), None for any other equation: what the algebra's normal form reads such a
    span as, while a declaration traced inline keeps the span itself, Dew's arithmetic."""
    if equation.primitive is not marked_call_p:
        return None
    kind = equation.params["kind"]
    return kind.computes if isinstance(kind, Protected) else None


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


@dataclasses.dataclass(frozen=True)
class WalkStep:
    """A kernel walk's step (`dew.nn.kernels.templates.mosaic.stepped`): its index through a marked call,
    which a count of the kernel's work by emission reads as one step (`templates.emission`), whatever loop
    or unrolled rotation holds it; inlined, it lowers to its index."""


@dataclasses.dataclass(frozen=True)
class Activation:
    """An activation's region (`dew.nn.activations`): `name`'s activation of its operand, or where `gated` a
    gated product's, the activation of its gate times its up projection, its operands gate then up. Its one
    reader is a fusion pass's choice of a matmul's epilogue that computes it; inlined, it lowers to its body,
    Dew's arithmetic as it stands."""
    name: str
    gated: bool


class Rare:
    """A kernel's verdict a rare span of it runs by (a decided walk's second, where its rows' sums passed
    fp32's largest: `templates.after_walk.Decided`), through a marked call, which a count of the kernel's work
    by emission reads as its common value, False (`templates.emission`): its launch states its common walk's
    work, as a rare call's scaled launch states none; inlined, it lowers to its value."""


type Kind = Protected | Surrogate | KernelGlue | Algebra | WalkStep | Activation | Rare
"""What a marked span is, read as an object, never a name: to the algebra pass, or to a kernel's count."""

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

