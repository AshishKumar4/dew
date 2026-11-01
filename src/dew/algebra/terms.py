"""The term language Dew's rewrite holds a program in (docs/design/algebra.md §6.2): one egglog sort, `Term`,
a constructor for each kind of equation of the language, and the conversion of a jaxpr's equations into terms
and of an extracted term back into equations.

A term names its equation's primitive and refers to the primitive's other parameters by an index into the
region's table of them (`Region.params`), so a rule rewrites the math and keeps each parameter as the program
wrote it. Each term carries the number of elements of its value where a rule cannot derive it (a leaf's, a
broadcast's, a reduction's), from which the `elements` fact follows for the rest (`dew.algebra.facts`)."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping

import jax
import numpy as np
from egglog import Expr, StringLike, Vec, f64Like, get_callable_args, get_literal_value, i64, i64Like, method
from jax.extend import core as jcore

BINARY = frozenset({"add", "sub", "mul", "div", "max", "min", "pow", "atan2", "rem", "nextafter"})
UNARY = frozenset({"neg", "exp", "exp2", "log", "log1p", "expm1", "sqrt", "rsqrt", "abs", "sign", "tanh",
                   "logistic", "integer_pow", "square", "sin", "cos", "erf", "erf_inv", "floor", "ceil",
                   "round", "is_finite", "not"})
COMPARISONS = frozenset({"lt", "le", "eq", "ne", "ge", "gt"})
REDUCTIONS = {"reduce_sum": "sum", "reduce_max": "max", "reduce_min": "min"}
SHAPES = frozenset({"broadcast_in_dim", "reshape", "transpose", "squeeze", "expand_dims"})
"""The primitives of the language (§1): elementwise arithmetic and comparisons, casts, the layout changes and
the reductions a rule reads. Every other equation is a leaf of the region."""


class Term(Expr):
    """One value of a region: a leaf (an input, or an output of an equation outside the language), a literal,
    or an equation of the language over other terms."""

    @classmethod
    def leaf(cls, index: i64Like, elements: i64Like) -> Term: ...

    @classmethod
    def literal(cls, value: f64Like, params: i64Like) -> Term: ...

    @method(cost=1)
    def binary(self, name: StringLike, other: Term) -> Term: ...

    @method(cost=1)
    def unary(self, name: StringLike, params: i64Like) -> Term: ...

    @method(cost=1)
    def compare(self, name: StringLike, other: Term) -> Term: ...

    @method(cost=1)
    def select(self, false: Term, true: Term) -> Term: ...

    @method(cost=1)
    def convert(self, params: i64Like) -> Term: ...

    @method(cost=1)
    def shaped(self, name: StringLike, params: i64Like, elements: i64Like) -> Term: ...

    @method(cost=1)
    def reduce(self, monoid: StringLike, axes: Vec[i64], params: i64Like, elements: i64Like) -> Term: ...


@dataclasses.dataclass
class Opaque:
    """An equation outside the language: its inputs' terms (None for a literal, kept in `literals`) and the
    leaves of its outputs."""

    equation: jcore.JaxprEqn
    inputs: list[Term | None]
    literals: list[object]
    outputs: list[int]


@dataclasses.dataclass
class Region:
    """A jaxpr held as terms: each leaf's value in order (`leaves`, the inputs and constants first), the
    equations outside the language in program order (`opaque`), each output's term (`outputs`, None for a
    literal output, kept in `literal_outputs`), and the parameters the terms refer to by index (`params`: a
    primitive and its parameters)."""

    leaves: int
    opaque: list[Opaque]
    outputs: list[Term | None]
    literal_outputs: list[object]
    params: list[tuple[jcore.Primitive, Mapping]]


def protected(equation: jcore.JaxprEqn) -> bool:
    """Whether an equation lies in a region a precision module marks protected (`dew.algebra.PROTECTED`, an
    error-free transform's), which survives autodiff in the name stack its traced equations carry."""
    from dew.algebra import PROTECTED

    return PROTECTED in str(equation.source_info.name_stack)


def _elements(aval) -> int:
    return int(np.prod(getattr(aval, "shape", ()), dtype=np.int64))


def region(closed: jcore.ClosedJaxpr) -> Region:
    """`closed`'s equations as terms: each of the language a constructor over its inputs' terms, each other
    one (and each protected one, `protected`) a leaf for each output, after its inputs' terms."""
    jaxpr = closed.jaxpr
    params: list[tuple[jcore.Primitive, Mapping]] = []
    indexed: dict[tuple[str, str], int] = {}

    def param(primitive: jcore.Primitive, values: Mapping) -> int:
        key = (primitive.name, repr(sorted(values.items(), key=lambda item: item[0])))
        if key not in indexed:
            indexed[key] = len(params)
            params.append((primitive, dict(values)))
        return indexed[key]

    env: dict[jcore.Var, Term] = {}
    leaves = 0
    for var in [*jaxpr.constvars, *jaxpr.invars]:
        env[var] = Term.leaf(leaves, _elements(var.aval))
        leaves += 1

    def read(atom) -> Term:
        if isinstance(atom, jcore.Literal):
            return Term.literal(float(np.asarray(atom.val)), param(jax.lax.convert_element_type_p,
                                                                   {"new_dtype": atom.aval.dtype}))
        return env[atom]

    opaque: list[Opaque] = []
    for equation in jaxpr.eqns:
        term = None if protected(equation) else _term(equation, read, param)
        if term is not None:
            env[equation.outvars[0]] = term
            continue
        outputs = []
        for var in equation.outvars:
            env[var] = Term.leaf(leaves, _elements(var.aval))
            outputs.append(leaves)
            leaves += 1
        opaque.append(Opaque(equation, [None if isinstance(atom, jcore.Literal) else env[atom]
                                        for atom in equation.invars],
                             [atom.val if isinstance(atom, jcore.Literal) else None
                              for atom in equation.invars],
                             outputs))
    outputs = [None if isinstance(atom, jcore.Literal) else env[atom] for atom in jaxpr.outvars]
    literal_outputs = [atom.val if isinstance(atom, jcore.Literal) else None for atom in jaxpr.outvars]
    return Region(leaves, opaque, outputs, literal_outputs, params)


def _term(equation: jcore.JaxprEqn, read: Callable, param: Callable) -> Term | None:
    """`equation` as a term over its inputs' (`read`), or None where it is outside the language."""
    name, primitive = equation.primitive.name, equation.primitive
    if len(equation.outvars) != 1:
        return None
    if name in BINARY and len(equation.invars) == 2 and not equation.params:
        return read(equation.invars[0]).binary(name, read(equation.invars[1]))
    if name in COMPARISONS and len(equation.invars) == 2:
        return read(equation.invars[0]).compare(name, read(equation.invars[1]))
    if name in UNARY and len(equation.invars) == 1:
        return read(equation.invars[0]).unary(name, param(primitive, equation.params))
    if name == "convert_element_type":
        return read(equation.invars[0]).convert(param(primitive, equation.params))
    if name == "select_n" and len(equation.invars) == 3:
        predicate, false, true = (read(atom) for atom in equation.invars)
        return predicate.select(false, true)
    if name in SHAPES and len(equation.invars) == 1:
        return read(equation.invars[0]).shaped(name, param(primitive, equation.params),
                                               _elements(equation.outvars[0].aval))
    if name in REDUCTIONS and len(equation.invars) == 1:
        axes = Vec[i64](*(i64(int(axis)) for axis in equation.params["axes"]))
        return read(equation.invars[0]).reduce(REDUCTIONS[name], axes, param(primitive, equation.params),
                                               _elements(equation.outvars[0].aval))
    return None


PRIMITIVES = {name: getattr(jax.lax, f"{name}_p") for name in BINARY | COMPARISONS
              if hasattr(jax.lax, f"{name}_p")}
PRIMITIVES["select_n"] = jax.lax.select_n_p


def evaluated(region_: Region, chosen: Callable[[Term], Term], values: list) -> list:
    """The values of `region_`'s outputs over its leaves' `values` (inputs and constants first), each term as
    extraction `chosen` its form: each equation outside the language bound in program order once its inputs
    are known, each shared term evaluated once."""
    known: dict[str, object] = {}

    def value(term: Term) -> object:
        key = str(term)
        if key not in known:
            known[key] = _value(term, value, region_.params, values)
        return known[key]

    for opaque in region_.opaque:
        operands = [opaque.literals[index] if term is None else value(chosen(term))
                    for index, term in enumerate(opaque.inputs)]
        results = opaque.equation.primitive.bind(*operands, **opaque.equation.params)
        results = results if opaque.equation.primitive.multiple_results else [results]
        for leaf, result in zip(opaque.outputs, results, strict=True):
            values[leaf] = result
    return [region_.literal_outputs[index] if root is None else value(chosen(root))
            for index, root in enumerate(region_.outputs)]


def _integers(vector) -> tuple[int, ...]:
    return tuple(int(get_literal_value(item)) for item in vector.value)


def _value(term: Term, value: Callable[[Term], object], params: list, leaves: list) -> object:
    """One term's value, its operands' from `value`."""
    if (args := get_callable_args(term, Term.leaf)) is not None:
        return leaves[int(get_literal_value(args[0]))]
    if (args := get_callable_args(term, Term.literal)) is not None:
        _, given = params[int(get_literal_value(args[1]))]
        return np.asarray(float(get_literal_value(args[0])), dtype=given["new_dtype"])
    if (args := get_callable_args(term, Term.binary)) is not None:
        left, name, right = args
        return PRIMITIVES[str(get_literal_value(name))].bind(value(left), value(right))
    if (args := get_callable_args(term, Term.compare)) is not None:
        left, name, right = args
        return PRIMITIVES[str(get_literal_value(name))].bind(value(left), value(right))
    if (args := get_callable_args(term, Term.select)) is not None:
        predicate, false, true = args
        return jax.lax.select_n_p.bind(value(predicate), value(false), value(true))
    # Each other constructor's primitive parameters, by index, after its operand and its other arguments.
    for constructor, operand_count in ((Term.unary, 1), (Term.convert, 0), (Term.shaped, 1),
                                       (Term.reduce, 2)):
        if (args := get_callable_args(term, constructor)) is not None:
            primitive, given = params[int(get_literal_value(args[1 + operand_count]))]
            return primitive.bind(value(args[0]), **given)
    raise ValueError(f"a term of no known constructor: {term}")
