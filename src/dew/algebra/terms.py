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
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from egglog import Expr, StringLike, Vec, get_callable_args, get_literal_value, i64, i64Like, method
from jax._src import core as jax_core
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
    (`stock`, `stocked`), each term's dtype, and each term's magnitude its equations, constants and
    declared ranges prove (`magnitudes`, by term text, `Magnitude`; None where none: an exponential of an
    operand's, a quotient's)."""

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
    dtypes: Mapping[Hashable, np.dtype]
    magnitudes: Mapping[Hashable, Magnitude | None]
    products: dict[int, tuple[np.dtype | None, ...]] = dataclasses.field(default_factory=dict)
    """Each product's parameters' index (`Term.dot`'s), its operands' dtypes (a rule's product its own's)."""
    extents: dict[Hashable, tuple[int, ...]] = dataclasses.field(init=False)

    def __post_init__(self):
        self.extents = {}
        for term, extent in self.shapes:
            self.extents.setdefault(key(term), extent)

    def shape(self, term: Term) -> tuple[int, ...]:
        """`term`'s shape as its equation states it, a term of the region's: the one lookup of the shapes its
        terms carry (`shapes`, `extents` by `key`)."""
        return self.extents[key(term)]

    def shaped(self, term: Term, extent: tuple[int, ...]) -> None:
        """`term`, a term a derivation adds (a literal of its own), held at `extent`."""
        self.shapes.append((term, extent))
        self.extents.setdefault(key(term), extent)

    def roots(self) -> list[Term]:
        """The terms extraction chooses forms for: each output and each input of an equation outside the
        language, once each, in program order."""
        seen: dict[Hashable, Term] = {}
        for term in [*(term for opaque in self.opaque for term in opaque.inputs), *self.outputs]:
            if term is not None:
                seen.setdefault(key(term), term)
        return list(seen.values())


def key(term: Expr) -> Hashable:
    """`term` as a dictionary's key: its declaration, which egglog builds it from, equal wherever two terms
    are (its text is black's formatting of it, about a millisecond a term: a third of a gemma3n training
    step's pass)."""
    return term.__egg_typed_expr__


def protected(equation: jcore.JaxprEqn) -> bool:
    """Whether an equation lies in a region a precision module marks protected (`PROTECTED`, an error-free
    transform's), which survives autodiff in the name stack its traced equations carry."""
    return PROTECTED in str(equation.source_info.name_stack)


Magnitude = int
"""A bound on a value's magnitude (docs/design/algebra.md §6.3): its power of two, `|x| <= 2**log2`, a number
its constants', its declared ranges' and its equations' prove; a value no number bounds has none (None). A
sum's is twice the larger of its terms', a product's their product, a contraction's their product times its
length."""

NOTHING = -(2 ** 20)
"""A zero's magnitude: under every other."""


def region(closed: jcore.Jaxpr, ranges: Mapping[int, float | tuple[float, float]] | None = None) -> Region:
    """`closed`'s equations as terms: each of the language a constructor over its inputs' terms, each other
    one (and each protected one, `protected`) a leaf for each output, after its inputs' terms. Each term's
    magnitude (`Magnitude`) from its constants', its operands' declared `ranges` (by input index, the
    constants apart: a bound on its magnitude, or an interval `(low, high)`, whose `high` at most 0 bounds its
    exponential by 1; an input with none is unbounded) and its equations'."""
    jaxpr = closed.jaxpr
    params: list[tuple[jcore.Primitive, Mapping]] = []
    indexed: dict[tuple[str, str, tuple | None], int] = {}
    literals: list = []
    shapes: list[tuple[Term, tuple[int, ...]]] = []
    depths: dict[jcore.Var, int] = {}

    products: dict[int, tuple[np.dtype | None, ...]] = {}

    def param(primitive: jcore.Primitive, values: Mapping, operands: tuple | None = None) -> int:
        # A product's operands' dtypes key its parameters too, its mode's (`cost.work`).
        key = (primitive.name, repr(sorted(values.items(), key=lambda named: named[0])), operands)
        if key not in indexed:
            indexed[key] = len(params)
            params.append((primitive, dict(values)))
            if operands is not None:
                products[indexed[key]] = operands
        return indexed[key]

    env: dict[jcore.Var, Term] = {}
    dtypes: dict[Hashable, np.dtype] = {}
    magnitudes: dict[Hashable, Magnitude | None] = {}
    leaves = 0
    declared = dict(ranges or {})
    given = [*(_magnitude(value) for value in closed.consts),
             *(_declared(declared.get(index)) for index in range(len(jaxpr.invars)))]
    # The values at most 0 (an operand declared so, and the layouts and conversions of one), whose
    # exponential is at most 1.
    nonpositive = {var for index, var in enumerate(jaxpr.invars)
                   if isinstance(interval := declared.get(index), tuple) and interval[1] <= 0}
    for var, magnitude in zip([*jaxpr.constvars, *jaxpr.invars], given, strict=True):
        env[var] = Term.leaf(leaves)
        shapes.append((env[var], _shape(var)))
        _typed(dtypes, magnitudes, env[var], var, magnitude)
        leaves += 1

    def read(atom) -> Term:
        if isinstance(atom, jcore.Literal):
            literals.append(atom.val)
            term = Term.literal(len(literals) - 1)
            shapes.append((term, _shape(atom)))
            _typed(dtypes, magnitudes, term, atom, _magnitude(atom.val))
            return term
        return env[atom]

    def magnitude_of(atom) -> Magnitude | None:
        return _magnitude(atom.val) if isinstance(atom, jcore.Literal) else magnitudes[key(env[atom])]

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
            _typed(dtypes, magnitudes, env[out], out, magnitudes[key(env[source])])
            if source in nonpositive:
                nonpositive.add(out)
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
            operands = [magnitude_of(atom) for atom in equation.invars]
            name = equation.primitive.name
            source = equation.invars[0]
            held = isinstance(source, jcore.Var) and source in nonpositive
            if name in KEEPING_SIGN and held:
                nonpositive.add(out)
            exponentiated = name in ("exp", "exp2") and held
            _typed(dtypes, magnitudes, term, out, UNIT if exponentiated else _bounded(equation, operands))
            depths[out] = 1 + max((depths.get(atom, 0) for atom in equation.invars
                                   if not isinstance(atom, jcore.Literal)), default=0)
            continue
        outputs = []
        for var in equation.outvars:
            env[var] = Term.leaf(leaves)
            shapes.append((env[var], _shape(var)))
            # An equation outside the language bounds nothing.
            _typed(dtypes, magnitudes, env[var], var, None)
            outputs.append(leaves)
            leaves += 1
        opaque.append(Opaque(equation, [None if isinstance(atom, jcore.Literal) else env[atom]
                                        for atom in equation.invars],
                             [atom.val if isinstance(atom, jcore.Literal) else None
                              for atom in equation.invars], outputs))
    outputs = [None if isinstance(atom, jcore.Literal) else env[atom] for atom in jaxpr.outvars]
    literal_outputs = [atom.val if isinstance(atom, jcore.Literal) else None for atom in jaxpr.outvars]
    stock = {name: param(primitive, values) for name, (primitive, values) in stocked().items()}
    return Region(leaves, opaque, outputs, literal_outputs, params, literals, shapes,
                  max(depths.values(), default=0), len(jaxpr.eqns), stock, dtypes, magnitudes, products)


BOUNDED = frozenset({"tanh", "logistic", "sin", "cos", "erf"})
"""The language's functions whose magnitude is at most 1 over a real operand of any magnitude (over a complex
one, `sin(pi/2 + 89j)` is 2.2e38: unbounded)."""
KEEPING_MAGNITUDE = frozenset({"neg", "abs", "convert_element_type", "reduce_max", "reduce_min"}) | SHAPES
"""The language's equations whose output is no larger in magnitude than their operand's."""
INTEGRAL = frozenset({"floor", "ceil", "round"})
"""The language's equations rounding to an integer: no larger in magnitude than their operand's or 1, an
integer bounding any value it bounds (`ceil(2**-126)` is 1)."""
BITWISE = frozenset({"and", "or", "xor", "not"})
"""The language's bitwise equations: a boolean's at most 1, an integer's any value of its dtype (int32's
`not(-2**31)` is `2**31 - 1`)."""
UNIT = 0
KEEPING_SIGN = frozenset({"convert_element_type", "reduce_max", "reduce_min"}) | SHAPES
"""The language's equations whose output is at most 0 where their operand is."""


def _bounded(equation: jcore.JaxprEqn, operands: list[Magnitude | None]) -> Magnitude | None:
    """A bound on the magnitude of `equation`'s output from its operands' (`operands`, `Magnitude`), None
    where its primitive proves none (an exponential of a value of any magnitude, a quotient)."""
    name = equation.primitive.name
    real = not any(isinstance(atom.aval, ShapedArray) and jnp.issubdtype(atom.aval.dtype, jnp.complexfloating)
                   for atom in equation.invars)
    if name in BITWISE:
        aval = equation.outvars[0].aval
        if not isinstance(aval, ShapedArray):
            return None
        dtype = np.dtype(aval.dtype)
        if dtype == np.bool_:
            return UNIT
        # A signed integer of `bits` bits is at most 2**(bits - 1) in magnitude, an unsigned one 2**bits.
        integers = jnp.iinfo(dtype)
        return integers.bits - (integers.min < 0)
    if name in COMPARISONS or name in ("is_finite", "sign") or (name in BOUNDED and real):
        return UNIT
    if any(bound is None for bound in operands):
        return None
    given: list[int] = [bound for bound in operands if bound is not None]
    if name in ("add", "sub"):
        return max(given) + 1
    if name == "mul":
        return given[0] + given[1]
    if name in ("max", "min"):
        return max(given)
    if name == "select_n":
        return max(given[1:])
    if name in KEEPING_MAGNITUDE:
        return given[0]
    if name in INTEGRAL:
        return max(given[0], UNIT)
    if name == "square":
        return 2 * given[0]
    if name == "integer_pow" and equation.params["y"] >= 1:
        return equation.params["y"] * given[0]
    if name in ("exp", "exp2") and given[0] <= 6:
        # An exponential of a value under 64: under 2**(64 log2(e)).
        return math.ceil(2 ** given[0] * (math.log2(math.e) if name == "exp" else 1.0))
    if name == "reduce_sum":
        shape = _shape(equation.invars[0])
        return given[0] + _log2(math.prod(shape[axis] for axis in equation.params["axes"]))
    if name == "dot_general":
        (contracted, _), _ = equation.params["dimension_numbers"]
        shape = _shape(equation.invars[0])
        return given[0] + given[1] + _log2(math.prod(shape[axis] for axis in contracted))
    return None


def _log2(value: float) -> int:
    """`ceil(log2(value))`, `NOTHING` for 0: a magnitude's exponent bounding `value` from above."""
    return NOTHING if value == 0 else math.ceil(math.log2(value))


def _magnitude(value) -> Magnitude | None:
    """A constant's magnitude, its elements' largest (a complex one's modulus); None for one of no number (a
    key's)."""
    array = np.asarray(value)
    if not (np.issubdtype(array.dtype, np.number) or array.dtype == bool):
        return None
    # In complex128, which holds every integer's absolute value (int32's least's too) and a complex modulus.
    largest = float(np.max(np.abs(array.astype(np.complex128)))) if array.size else 0.0
    return _log2(largest) if math.isfinite(largest) else None


def _declared(declared: float | tuple[float, float] | None) -> Magnitude | None:
    """An input's magnitude from its declared range (an interval's larger end's), None where it declares
    none."""
    if declared is None:
        return None
    largest = max(abs(declared[0]), abs(declared[1])) if isinstance(declared, tuple) else declared
    return _log2(largest) if math.isfinite(largest) else None


def _typed(dtypes: dict, magnitudes: dict, term: Term, atom, magnitude: Magnitude | None) -> None:
    """`term`'s dtype (`atom`'s) and its magnitude, the first where one term names two values."""
    named = key(term)
    if isinstance(atom.aval, ShapedArray):
        dtypes.setdefault(named, np.dtype(atom.aval.dtype))
    magnitudes.setdefault(named, magnitude)


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
    known: dict[Hashable, jax.Array] = {}

    def value(term: Term) -> jax.Array:
        named = key(term)
        if named not in known:
            known[named] = value_of(term, value, region_, values)
        return known[named]

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
        operands = tuple(np.dtype(var.aval.dtype) if isinstance(var.aval, ShapedArray) else None
                         for var in equation.invars)
        return read(equation.invars[0]).dot(read(equation.invars[1]), param(primitive, given, operands))
    return None


def _index(argument) -> int:
    return int(get_literal_value(argument))


def value_of(term: Term, value: Callable[[Term], jax.Array], region_: Region, leaves: Sequence) -> jax.Array:
    """One term's value, its operands' from `value`, each equation of several operands bound over them aligned
    as the lax function it came from aligns them (`jax_core.auto_insert_reshard`): a literal or a constant
    beside a value a checked `shard_map`'s manual axes vary varies as it does."""
    if (args := get_callable_args(term, Term.leaf)) is not None:
        return leaves[_index(args[0])]
    if (args := get_callable_args(term, Term.literal)) is not None:
        return region_.literals[_index(args[0])]
    for constructor in (Term.binary, Term.compare, Term.dot):
        if (args := get_callable_args(term, constructor)) is not None:
            # A product's operands lead its arguments; an operation's name stands between a binary's two.
            left, right, index = args if constructor is Term.dot else (args[0], args[2], args[3])
            primitive, given = region_.params[_index(index)]
            return primitive.bind(*jax_core.auto_insert_reshard(value(left), value(right)), **given)
    if (args := get_callable_args(term, Term.select)) is not None:
        operands = jax_core.auto_insert_reshard(*(value(operand) for operand in args))
        return jax.lax.select_n_p.bind(*operands)
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
        if (named := key(term)) in seen:
            continue
        seen.add(named)
        yield term
        for constructor in LANGUAGE:
            if (args := get_callable_args(term, constructor)) is not None:
                pending.extend(arg for arg in args if isinstance(arg, Term))
