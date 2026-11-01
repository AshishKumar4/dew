"""The term language Dew's rewrite holds a program in (docs/design/algebra.md §6.2): one egglog sort, `Term`,
a constructor for each kind of equation of the language, and the conversion of a jaxpr's equations into terms
and of an extracted term back into equations.

A term names its equation's primitive and refers to the primitive's other parameters by an index into the
region's table of them (`Region.params`), and a literal to its exact value in the region's table of literals,
so a rule rewrites the math and keeps each parameter and constant as the program wrote it. Every term made
from an equation states its shape (`rules.shape`), which rules read before they join or move values."""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Callable, Iterator, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from egglog import Expr, StringLike, Vec, get_callable_args, get_literal_value, i64, i64Like, method
from jax.core import ShapedArray
from jax.extend import core as jcore

from dew.nn.precision import PROTECTED

BINARY = frozenset({"add", "sub", "mul", "div", "max", "min", "pow", "atan2", "rem", "nextafter", "and",
                    "or", "xor"})
UNARY = frozenset({"neg", "exp", "exp2", "log", "log1p", "expm1", "sqrt", "rsqrt", "abs", "sign", "tanh",
                   "logistic", "integer_pow", "square", "sin", "cos", "erf", "erf_inv", "floor", "ceil",
                   "round", "is_finite", "not"})
COMPARISONS = frozenset({"lt", "le", "eq", "ne", "ge", "gt"})
REDUCTIONS = {"reduce_sum": "sum", "reduce_max": "max", "reduce_min": "min"}
SHAPES = frozenset({"broadcast_in_dim", "reshape", "transpose", "squeeze", "expand_dims"})
"""The primitives of the language (§1): elementwise arithmetic and comparisons, casts, the layout changes, the
reductions a rule reads, and products (`dot_general`). Every other equation is a leaf of the region."""


class Term(Expr):
    """One value of a region: a leaf (an input, or an output of an equation outside the language), a literal,
    or an equation of the language over other terms."""

    @classmethod
    def leaf(cls, index: i64Like) -> Term: ...

    @classmethod
    def literal(cls, index: i64Like) -> Term: ...

    @classmethod
    def bundle(cls, roots: Vec[Term]) -> Term: ...

    @method(cost=1)
    def binary(self, name: StringLike, other: Term, params: i64Like) -> Term: ...

    @method(cost=1)
    def compare(self, name: StringLike, other: Term, params: i64Like) -> Term: ...

    @method(cost=1)
    def unary(self, name: StringLike, params: i64Like) -> Term: ...

    @method(cost=1)
    def select(self, false: Term, true: Term) -> Term: ...

    @method(cost=1)
    def convert(self, params: i64Like) -> Term: ...

    @method(cost=1)
    def shaped(self, name: StringLike, params: i64Like) -> Term: ...

    @method(cost=1)
    def reduce(self, monoid: StringLike, axes: Vec[i64], params: i64Like) -> Term: ...

    @method(cost=1)
    def dot(self, other: Term, params: i64Like) -> Term: ...

    # The derivation's (`dew.algebra.derive`, §6.9): a walked reduction's value over the tiles before the one
    # a form reads (`state`, by the reduction's index, one side's of a merge), and a value as a step, a merge,
    # a finish or a carried state reads it (`stepped`, `merged`, `finished`, `held`), each rewritten into the
    # language by the derivation's rules and never extracted itself.

    @classmethod
    def state(cls, index: i64Like, side: i64Like) -> Term: ...

    @method(cost=1)
    def held(self, side: i64Like) -> Term: ...

    @method(cost=1)
    def stepped(self) -> Term: ...

    @method(cost=1)
    def merged(self) -> Term: ...

    @method(cost=1)
    def finished(self) -> Term: ...


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
    """A jaxpr held as terms: the count of its leaves (the inputs and constants first, then the outputs of the
    equations outside the language), those equations in program order (`opaque`), each output's term (None
    for a literal output, kept in `literal_outputs`), the parameters terms refer to by index (`params`: a
    primitive and its parameters), its literals as the program holds them, value and weak type
    (`literals`), each term's shape as its equation states it (`shapes`), the longest chain of the
    language's equations (`depth`), the index of each primitive a rule may write afresh at its defaults
    (`stock`, `stocked`), each term's dtype, a bound on each term's magnitude its equations and constants
    prove (`bounds`, by term text; infinite where none: an operand's, an equation's outside the language),
    and the largest finite value of its narrowest floating dtype (`limit`)."""

    leaves: int
    opaque: list[Opaque]
    outputs: list[Term | None]
    literal_outputs: list
    params: list[tuple[jcore.Primitive, Mapping]]
    literals: list
    shapes: list[tuple[Term, tuple[int, ...]]]
    depth: int
    equations: int
    stock: Mapping[str, int]
    dtypes: Mapping[str, np.dtype]
    bounds: Mapping[str, float]
    limit: float
    extents: dict[str, tuple[int, ...]] = dataclasses.field(init=False)

    def __post_init__(self):
        self.extents = {}
        for term, extent in self.shapes:
            self.extents.setdefault(str(term), extent)

    def shape(self, term: Term) -> tuple[int, ...]:
        """`term`'s shape as its equation states it, a term of the region's: the one lookup of the shapes its
        terms carry (`shapes`, `extents` by text)."""
        return self.extents[str(term)]

    def shaped(self, term: Term, extent: tuple[int, ...]) -> None:
        """`term`, a term a derivation adds (a literal of its own), held at `extent`."""
        self.shapes.append((term, extent))
        self.extents.setdefault(str(term), extent)

    def roots(self) -> list[Term]:
        """The terms extraction chooses forms for: each output and each input of an equation outside the
        language, once each, in program order."""
        seen: dict[str, Term] = {}
        for term in [*(term for opaque in self.opaque for term in opaque.inputs), *self.outputs]:
            if term is not None:
                seen.setdefault(str(term), term)
        return list(seen.values())


def protected(equation: jcore.JaxprEqn) -> bool:
    """Whether an equation lies in a region a precision module marks protected (`PROTECTED`, an error-free
    transform's), which survives autodiff in the name stack its traced equations carry."""
    return PROTECTED in str(equation.source_info.name_stack)


def region(closed: jcore.Jaxpr) -> Region:
    """`closed`'s equations as terms: each of the language a constructor over its inputs' terms, each other
    one (and each protected one, `protected`) a leaf for each output, after its inputs' terms."""
    jaxpr = closed.jaxpr
    params: list[tuple[jcore.Primitive, Mapping]] = []
    indexed: dict[tuple[str, str], int] = {}
    literals: list = []
    shapes: list[tuple[Term, tuple[int, ...]]] = []
    depths: dict[jcore.Var, int] = {}

    def param(primitive: jcore.Primitive, values: Mapping) -> int:
        key = (primitive.name, repr(sorted(values.items(), key=lambda named: named[0])))
        if key not in indexed:
            indexed[key] = len(params)
            params.append((primitive, dict(values)))
        return indexed[key]

    env: dict[jcore.Var, Term] = {}
    dtypes: dict[str, np.dtype] = {}
    bounds: dict[str, float] = {}
    leaves = 0
    held = [_magnitude(value) for value in closed.consts]
    for var, bound in zip([*jaxpr.constvars, *jaxpr.invars], [*held, *([math.inf] * len(jaxpr.invars))],
                          strict=True):
        env[var] = Term.leaf(leaves)
        shapes.append((env[var], _shape(var)))
        _typed(dtypes, bounds, env[var], var, bound)
        leaves += 1

    def read(atom) -> Term:
        if isinstance(atom, jcore.Literal):
            literals.append(atom.val)
            term = Term.literal(len(literals) - 1)
            shapes.append((term, _shape(atom)))
            _typed(dtypes, bounds, term, atom, _magnitude(atom.val))
            return term
        return env[atom]

    def bound_of(atom) -> float:
        return _magnitude(atom.val) if isinstance(atom, jcore.Literal) else bounds[str(env[atom])]

    # Each value a broadcast made, by its source and the axes it laid the source's along, so a broadcast of
    # one is read as one broadcast of the source (exact: a layout alone), whose axes a hoist reads.
    laid: dict[jcore.Var, tuple[jcore.Var, tuple[int, ...]]] = {}
    opaque: list[Opaque] = []
    for equation in jaxpr.eqns:
        if (equation.primitive is jax.lax.broadcast_in_dim_p and not protected(equation)
                and not isinstance(equation.invars[0], jcore.Literal) and equation.invars[0] in laid):
            source, inner = laid[equation.invars[0]]
            outer = tuple(equation.params["broadcast_dimensions"])
            composed = {**equation.params, "broadcast_dimensions": tuple(outer[axis] for axis in inner)}
            out = equation.outvars[0]
            env[out] = read(source).shaped("broadcast_in_dim", param(equation.primitive, composed))
            shapes.append((env[out], _shape(out)))
            _typed(dtypes, bounds, env[out], out, bounds[str(env[source])])
            laid[out] = (source, composed["broadcast_dimensions"])
            depths[out] = depths.get(source, 0) + 1
            continue
        if (equation.primitive is jax.lax.broadcast_in_dim_p and not protected(equation)
                and not isinstance(equation.invars[0], jcore.Literal)):
            laid[equation.outvars[0]] = (equation.invars[0], tuple(equation.params["broadcast_dimensions"]))
        term = None if protected(equation) else _term(equation, read, param)
        if term is not None:
            out = equation.outvars[0]
            env[out] = term
            shapes.append((term, _shape(out)))
            _typed(dtypes, bounds, term, out,
                   _bounded(equation, [bound_of(atom) for atom in equation.invars]))
            depths[out] = 1 + max((depths.get(atom, 0) for atom in equation.invars
                                   if not isinstance(atom, jcore.Literal)), default=0)
            continue
        outputs = []
        for var in equation.outvars:
            env[var] = Term.leaf(leaves)
            shapes.append((env[var], _shape(var)))
            _typed(dtypes, bounds, env[var], var, math.inf)
            outputs.append(leaves)
            leaves += 1
        opaque.append(Opaque(equation, [None if isinstance(atom, jcore.Literal) else env[atom]
                                        for atom in equation.invars],
                             [atom.val if isinstance(atom, jcore.Literal) else None
                              for atom in equation.invars], outputs))
    outputs = [None if isinstance(atom, jcore.Literal) else env[atom] for atom in jaxpr.outvars]
    literal_outputs = [atom.val if isinstance(atom, jcore.Literal) else None for atom in jaxpr.outvars]
    stock = {name: param(primitive, values) for name, (primitive, values) in stocked().items()}
    floats = [jnp.finfo(dtype).max for dtype in dtypes.values() if jnp.issubdtype(dtype, jnp.floating)]
    return Region(leaves, opaque, outputs, literal_outputs, params, literals, shapes,
                  max(depths.values(), default=0), len(jaxpr.eqns), stock, dtypes, bounds,
                  float(min(floats, default=np.finfo(np.float32).max)))


BOUNDED = {"tanh": 1.0, "logistic": 1.0, "sin": 1.0, "cos": 1.0, "erf": 1.0, "sign": 1.0}
"""The language's functions whose magnitude is at most a constant whatever their operand's."""
KEEPING_MAGNITUDE = frozenset({"neg", "abs", "convert_element_type", "reduce_max", "reduce_min", "floor",
                               "ceil", "round"}) | SHAPES
"""The language's equations whose output is no larger in magnitude than their operand's."""


def _bounded(equation: jcore.JaxprEqn, given: list[float]) -> float:
    """A bound on the magnitude of `equation`'s output from its operands' (`given`), by interval arithmetic
    on magnitudes; infinite where its primitive proves none."""
    name = equation.primitive.name
    if name in COMPARISONS or name in ("and", "or", "xor", "not", "is_finite"):
        return 1.0
    if name in BOUNDED:
        return BOUNDED[name]
    if any(math.isinf(bound) for bound in given):
        return math.inf
    if name in ("add", "sub"):
        return given[0] + given[1]
    if name == "mul":
        return given[0] * given[1]
    if name in ("max", "min"):
        return max(given)
    if name == "select_n":
        return max(given[1:])
    if name in KEEPING_MAGNITUDE:
        return given[0]
    if name == "square":
        return given[0] ** 2
    if name == "integer_pow" and equation.params["y"] >= 0:
        return given[0] ** equation.params["y"]
    if name == "reduce_sum":
        shape = _shape(equation.invars[0])
        return given[0] * math.prod(shape[axis] for axis in equation.params["axes"])
    if name == "dot_general":
        (contracted, _), _ = equation.params["dimension_numbers"]
        shape = _shape(equation.invars[0])
        return given[0] * given[1] * math.prod(shape[axis] for axis in contracted)
    return math.inf


def _magnitude(value) -> float:
    """The largest magnitude among a constant's elements, infinite for one of no number (a key's)."""
    array = np.asarray(value)
    if not (np.issubdtype(array.dtype, np.number) or array.dtype == bool) or array.size == 0:
        return math.inf if array.size else 0.0
    largest = float(np.max(np.abs(array.astype(np.float64))))
    return largest if math.isfinite(largest) else math.inf


def _typed(dtypes: dict, bounds: dict, term: Term, atom, bound: float) -> None:
    """`term`'s dtype (`atom`'s) and its magnitude's bound, the tighter where its text names two values."""
    key = str(term)
    if isinstance(atom.aval, ShapedArray):
        dtypes.setdefault(key, np.dtype(atom.aval.dtype))
    bounds[key] = min(bounds.get(key, math.inf), bound)


@functools.cache
def stocked() -> Mapping[str, tuple[jcore.Primitive, Mapping]]:
    """The primitives a stable rule writes where the program held none (`rules.stock`), each at the
    parameters JAX binds it with by default, read from JAX's own trace of them."""
    scalar = jnp.ones((), jnp.float32)
    traced = jax.make_jaxpr(lambda a: (a / a, a - a, a + a, a * a, jnp.maximum(a, a), jnp.exp(a),
                                       jnp.log1p(a), jnp.expm1(a)))(scalar)
    return {equation.primitive.name: (equation.primitive, dict(equation.params))
            for equation in traced.jaxpr.eqns}


def spoken(equation: jcore.JaxprEqn) -> bool:
    """Whether `equation` is of the language (`_term` makes a term of it), outside every protected region."""
    name, arity = equation.primitive.name, len(equation.invars)
    if len(equation.outvars) != 1 or protected(equation):
        return False
    binary = name in BINARY or name in COMPARISONS or name == "dot_general"
    unary = name in UNARY or name in SHAPES or name in REDUCTIONS or name == "convert_element_type"
    return (binary and arity == 2) or (unary and arity == 1) or (name == "select_n" and arity == 3)


def form_value(region_: Region, form: Term, values: list) -> jax.Array:
    """One chosen `form`'s value over `region_`'s leaves' `values` (a region of the language alone, whose
    leaves are its inputs), each shared term evaluated once."""
    known: dict[str, jax.Array] = {}

    def value(term: Term) -> jax.Array:
        key = str(term)
        if key not in known:
            known[key] = value_of(term, value, region_, values)
        return known[key]

    return value(form)


def _shape(atom) -> tuple[int, ...]:
    """An array value's shape; a value of no shape (a token, a key's) has none the rules read."""
    return tuple(atom.aval.shape) if isinstance(atom.aval, ShapedArray) else ()


def _term(equation: jcore.JaxprEqn, read: Callable, param: Callable) -> Term | None:
    """`equation` as a term over its inputs' (`read`), or None where it is outside the language."""
    name, primitive = equation.primitive.name, equation.primitive
    if len(equation.outvars) != 1:
        return None
    if name in BINARY and len(equation.invars) == 2:
        return read(equation.invars[0]).binary(name, read(equation.invars[1]),
                                               param(primitive, equation.params))
    if name in COMPARISONS and len(equation.invars) == 2:
        return read(equation.invars[0]).compare(name, read(equation.invars[1]),
                                                param(primitive, equation.params))
    if name in UNARY and len(equation.invars) == 1:
        return read(equation.invars[0]).unary(name, param(primitive, equation.params))
    if name == "convert_element_type" and len(equation.invars) == 1:
        return read(equation.invars[0]).convert(param(primitive, equation.params))
    if name == "select_n" and len(equation.invars) == 3:
        predicate, false, true = (read(atom) for atom in equation.invars)
        return predicate.select(false, true)
    if name in SHAPES and len(equation.invars) == 1:
        return read(equation.invars[0]).shaped(name, param(primitive, equation.params))
    if name in REDUCTIONS and len(equation.invars) == 1:
        axes = Vec[i64](*(i64(int(axis)) for axis in equation.params["axes"]))  # the reduced axes
        return read(equation.invars[0]).reduce(REDUCTIONS[name], axes, param(primitive, equation.params))
    if name == "dot_general" and len(equation.invars) == 2:
        given = dict(equation.params)
        dtypes = {var.aval.dtype for var in equation.invars if isinstance(var.aval, ShapedArray)}
        if dtypes == {given.get("preferred_element_type")}:
            # A product accumulating in its operands' own dtype is one of no preferred type (`rules.plain`).
            given["preferred_element_type"] = None
        return read(equation.invars[0]).dot(read(equation.invars[1]), param(primitive, given))
    return None


def _index(argument) -> int:
    return int(get_literal_value(argument))


def value_of(term: Term, value: Callable[[Term], jax.Array], region_: Region, leaves: Sequence) -> jax.Array:
    """One term's value, its operands' from `value`."""
    if (args := get_callable_args(term, Term.leaf)) is not None:
        return leaves[_index(args[0])]
    if (args := get_callable_args(term, Term.literal)) is not None:
        return region_.literals[_index(args[0])]
    for constructor in (Term.binary, Term.compare):
        if (args := get_callable_args(term, constructor)) is not None:
            left, _, right, index = args
            primitive, given = region_.params[_index(index)]
            return primitive.bind(value(left), value(right), **given)
    if (args := get_callable_args(term, Term.dot)) is not None:
        left, right, index = args
        primitive, given = region_.params[_index(index)]
        return primitive.bind(value(left), value(right), **given)
    if (args := get_callable_args(term, Term.select)) is not None:
        predicate, false, true = args
        return jax.lax.select_n_p.bind(value(predicate), value(false), value(true))
    # Each other constructor's primitive parameters, by index, last after its operand and its other arguments.
    for constructor in (Term.unary, Term.convert, Term.shaped, Term.reduce):
        if (args := get_callable_args(term, constructor)) is not None:
            primitive, given = region_.params[_index(args[-1])]
            return primitive.bind(value(args[0]), **given)
    raise ValueError(f"a term of no known constructor: {term}")


LANGUAGE = (Term.binary, Term.compare, Term.unary, Term.convert, Term.shaped, Term.reduce, Term.dot,
            Term.select)
"""The language's constructors that read terms, a leaf's and a literal's none."""


def subterms(form: Term) -> Iterator[Term]:
    """Each distinct term `form` reads through the language, itself first."""
    seen, pending = set(), [form]
    while pending:
        term = pending.pop()
        if (key := str(term)) in seen:
            continue
        seen.add(key)
        yield term
        for constructor in LANGUAGE:
            if (args := get_callable_args(term, constructor)) is not None:
                pending.extend(arg for arg in args if isinstance(arg, Term))
