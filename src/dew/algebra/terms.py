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
from typing import overload

import jax
import jax.numpy as jnp
import numpy as np
from egglog import (
    BaseExpr,
    Expr,
    String,
    StringLike,
    Vec,
    f64,
    f64Like,
    get_callable_args,
    get_literal_value,
    i64,
    i64Like,
    method,
)
from egglog.declarations import Declarations, HasDeclarations
from egglog.runtime import RuntimeExpr
from jax._src import core as jax_core
from jax._src.literals import TypedNdArray
from jax.core import ShapedArray
from jax.extend import core as jcore

from dew.nn.precision import PROTECTED

BINARY = frozenset({"add", "sub", "mul", "div", "max", "min", "pow", "atan2", "rem", "nextafter", "and",
                    "or", "xor"})
UNARY = frozenset({"neg", "exp", "exp2", "log", "log2", "log1p", "expm1", "sqrt", "rsqrt", "abs", "sign",
                   "tanh", "logistic", "integer_pow", "square", "sin", "cos", "erf", "erf_inv", "floor",
                   "ceil", "round", "is_finite", "not"})
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


@dataclasses.dataclass(frozen=True)
class Skeleton:
    """A term's structure without its egglog expression: `Term`'s constructors over plain tuples, equal
    exactly where the terms are (`region`'s `forms`). What the pass reads of a window to decide whether a rule
    may fire in it (`rules.reorderable`), at a fraction of its terms' cost: a Dory-4B training step's 1018
    windows, every one refused, spent most of the pass building their terms (FS-OVERHEAD-DORY-ON-P3)."""

    form: tuple

    @classmethod
    def leaf(cls, index: int) -> Skeleton:
        return cls(("leaf", index))

    @classmethod
    def literal(cls, index: int) -> Skeleton:
        return cls(("literal", index))

    def binary(self, name: str, other: Skeleton, params: int) -> Skeleton:
        return Skeleton(("binary", self.form, name, other.form, params))

    def compare(self, name: str, other: Skeleton, params: int) -> Skeleton:
        return Skeleton(("compare", self.form, name, other.form, params))

    def unary(self, name: str, params: int) -> Skeleton:
        return Skeleton(("unary", self.form, name, params))

    def select(self, false: Skeleton, true: Skeleton) -> Skeleton:
        return Skeleton(("select", self.form, false.form, true.form))

    def convert(self, params: int) -> Skeleton:
        return Skeleton(("convert", self.form, params))

    def shaped(self, name: str, params: int) -> Skeleton:
        return Skeleton(("shaped", self.form, name, params))

    def reduce(self, monoid: str, axes: Vec[i64], params: int) -> Skeleton:
        # The reduced axes are among the parameters `params` indexes.
        del axes
        return Skeleton(("reduce", self.form, monoid, params))

    def dot(self, other: Skeleton, params: int) -> Skeleton:
        return Skeleton(("dot", self.form, other.form, params))


def reduce_of(term: Term | Skeleton) -> tuple[str, Hashable] | None:
    """A reduction's monoid and its operand's key (`key`), None where `term` is no reduction: an egglog
    term's or a skeleton's alike."""
    if isinstance(term, Skeleton):
        return (term.form[2], term.form[1]) if term.form[0] == "reduce" else None
    if (args := get_callable_args(term, Term.reduce)) is None:
        return None
    return text(args[1]), key(args[0])


def binary_of(term: Term | Skeleton) -> tuple[str, Hashable, Hashable] | None:
    """A binary term's operation name and its operands' keys (`key`), None where `term` is no binary: an
    egglog term's or a skeleton's alike."""
    if isinstance(term, Skeleton):
        return (term.form[2], term.form[1], term.form[3]) if term.form[0] == "binary" else None
    if (args := get_callable_args(term, Term.binary)) is None:
        return None
    return text(args[1]), key(args[0]), key(args[2])


@dataclasses.dataclass
class Opaque[Node: (Term, Skeleton)]:
    """An equation outside the language: its inputs' terms (None for a literal, kept in `literals`) and the
    leaves of its outputs."""

    equation: jcore.JaxprEqn
    inputs: list[Node | None]
    literals: list[object]
    outputs: list[int]


@dataclasses.dataclass
class Region[Node: (Term, Skeleton)]:
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
    opaque: list[Opaque[Node]]
    outputs: list[Node | None]
    literal_outputs: list
    params: list[tuple[jcore.Primitive, Mapping]]
    literals: list
    shapes: list[tuple[Node, tuple[int, ...]]]
    depth: int
    equations: int
    stock: Mapping[str, int]
    dtypes: Mapping[Hashable, np.dtype]
    magnitudes: Mapping[Hashable, Magnitude | None]
    products: dict[int, tuple[np.dtype | None, ...]] = dataclasses.field(default_factory=dict)
    """Each product's parameters' index (`Term.dot`'s), its operands' dtypes (a rule's product its own's)."""
    literal_keys: frozenset[Hashable] = frozenset()
    """Each literal's term's key (`literals`' terms)."""
    extents: dict[Hashable, tuple[int, ...]] = dataclasses.field(init=False)

    def __post_init__(self):
        self.extents = {}
        for term, extent in self.shapes:
            self.extents.setdefault(key(term), extent)

    def shape(self, term: Node) -> tuple[int, ...]:
        """`term`'s shape as its equation states it, a term of the region's: the one lookup of the shapes its
        terms carry (`shapes`, `extents` by `key`)."""
        return self.extents[key(term)]

    def shaped(self, term: Node, extent: tuple[int, ...]) -> None:
        """`term`, a term a derivation adds (a literal of its own), held at `extent`."""
        self.shapes.append((term, extent))
        self.extents.setdefault(key(term), extent)

    def roots(self) -> list[Node]:
        """The terms extraction chooses forms for: each output and each input of an equation outside the
        language, once each, in program order."""
        seen: dict[Hashable, Node] = {}
        for term in [*(term for opaque in self.opaque for term in opaque.inputs), *self.outputs]:
            if term is not None:
                seen.setdefault(key(term), term)
        return list(seen.values())


def key(term: BaseExpr | Skeleton) -> Hashable:
    """`term` as a dictionary's key: its declaration, which egglog builds it from, equal wherever two terms
    are (its text is black's formatting of it, about a millisecond a term: a third of a gemma3n training
    step's pass); a skeleton's form (`Skeleton`)."""
    if isinstance(term, Skeleton):
        return term.form
    if not isinstance(term, RuntimeExpr):
        raise TypeError(f"{term!r} is no egglog term")
    return term.__egg_typed_expr__


def parameters_key(primitive: jcore.Primitive, values: Mapping) -> tuple[str, str]:
    """An equation's primitive and parameters as a region holds them: two whose parameters print alike share
    one entry of its table (`Region.params`), the first's."""
    return primitive.name, repr(sorted(values.items(), key=lambda named: named[0]))


def declarations(language: object) -> Declarations:
    """A language's declarations (egglog's `HasDeclarations`), which reading at import frees its defining
    frame (`linear_scan._DECLARATIONS`)."""
    if not isinstance(language, HasDeclarations):
        raise TypeError(f"{language!r} declares no egglog language")
    return language.__egg_decls__


def protected(equation: jcore.JaxprEqn) -> bool:
    """Whether an equation lies in a region a precision module marks protected (`PROTECTED`, an error-free
    transform's), which survives autodiff in the name stack its traced equations carry."""
    return PROTECTED in str(equation.source_info.name_stack)


Magnitude = int
"""A bound on a value's magnitude (docs/design/algebra.md §6.3): its power of two, `|x| <= 2**log2`, a number
its constants', its declared ranges' and its equations' prove; a value no number bounds has none (None). A
sum's is twice the larger of its terms', a product's their product, a contraction's their product times its
length. Proved under its arithmetic's contract (`_real_bound`): IEEE rounding to nearest, a native
transcendental given one power past its real range, a contraction's mode one it states the rounding of
(`_contraction_slack`); an equation outside the contract is unbounded, so overflow admission
(`rules.dominated`) never reads a bound no argument holds."""

NOTHING = -(2 ** 20)
"""A zero's magnitude: under every other."""


type Ranges = Mapping[int, float | tuple[float, float]] | None
"""Declared ranges by input index (`region`'s): a bound on an input's magnitude, or an interval."""


@overload
def region(closed: jcore.Jaxpr, ranges: Ranges = None) -> Region[Term]:
    ...


@overload
def region(closed: jcore.Jaxpr, ranges: Ranges = None, *,
           forms: type[Skeleton]) -> Region[Skeleton]:
    ...


def region(closed: jcore.Jaxpr, ranges: Ranges = None, *,
           forms: type[Term] | type[Skeleton] = Term) -> Region:
    """`closed`'s equations as terms: each of the language a constructor over its inputs' terms, each other
    one (and each protected one, `protected`) a leaf for each output, after its inputs' terms, each of `forms`
    (egglog's `Term`, or a window's structure alone, `Skeleton`). Each term's
    magnitude (`Magnitude`) from its constants', its operands' declared `ranges` (by input index, the
    constants apart: a bound on its magnitude, or an interval `(low, high)`, whose `high` at most 0 bounds its
    exponential by 2, one power past the reals' 1 for a native exponential; an input with none is
    unbounded) and its equations'."""
    if forms is Skeleton:
        return _region(closed, ranges, Skeleton)
    return _region(closed, ranges, Term)


def _region[Node: (Term, Skeleton)](closed: jcore.Jaxpr, ranges: Ranges,
                                    forms: type[Node]) -> Region[Node]:
    """`region`'s terms, each of `forms`."""
    jaxpr = closed.jaxpr
    params: list[tuple[jcore.Primitive, Mapping]] = []
    indexed: dict[tuple[str, str, tuple | None], int] = {}
    literals: list = []
    shapes: list[tuple[Node, tuple[int, ...]]] = []
    depths: dict[jcore.Var, int] = {}

    products: dict[int, tuple[np.dtype | None, ...]] = {}

    def param(primitive: jcore.Primitive, values: Mapping, operands: tuple | None = None) -> int:
        # A product's operands' dtypes key its parameters too, its mode's (`cost.work`).
        key = (*parameters_key(primitive, values), operands)
        if key not in indexed:
            indexed[key] = len(params)
            params.append((primitive, dict(values)))
            if operands is not None:
                products[indexed[key]] = operands
        return indexed[key]

    env: dict[jcore.Var, Node] = {}
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
        env[var] = forms.leaf(leaves)
        shapes.append((env[var], _shape(var)))
        _typed(dtypes, magnitudes, env[var], var, magnitude)
        leaves += 1

    def read(atom) -> Node:
        if isinstance(atom, jcore.Literal):
            literals.append(_held(atom.val))
            term = forms.literal(len(literals) - 1)
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
            if name in KEEPING_SIGN and held and (name != "convert_element_type" or _keeps_sign(source, out)):
                nonpositive.add(out)
            # An exponential of a value at most 0 is at most 1 in the reals, at most 2 as computed: a native
            # exponential is not certified under 1 (`ex2.approx`'s relative error, 2**-22.5, is more than an
            # ulp of 1), so it is given one power.
            exponentiated = name in ("exp", "exp2") and held
            _typed(dtypes, magnitudes, term, out, UNIT + 1 if exponentiated else _bounded(equation, operands))
            depths[out] = 1 + max((depths.get(atom, 0) for atom in equation.invars
                                   if not isinstance(atom, jcore.Literal)), default=0)
            continue
        outputs = []
        for var in equation.outvars:
            env[var] = forms.leaf(leaves)
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
                  max(depths.values(), default=0), len(jaxpr.eqns), stock, dtypes, magnitudes, products,
                  frozenset(key(forms.literal(index)) for index in range(len(literals))))


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
"""The language's equations whose output is at most 0 where their operand is, a conversion where it keeps
the sign (`_keeps_sign`)."""


def _keeps_sign(source, out) -> bool:
    """Whether a value of `source`'s dtype at most 0 stays at most 0 as `out`'s: a conversion that keeps
    every nonpositive value's sign (a float's to a float, an integer's to a float, a signed integer's to a
    signed one at least as wide), or any other equation `KEEPING_SIGN` names. A negative value converted to
    a boolean is true, to an unsigned integer positive, to a narrower signed one wraps (int32's -200 is int8's
    56), and a float's to an integer may pass its range."""
    if not all(isinstance(atom.aval, ShapedArray) for atom in (source, out)):
        return False
    given, made = np.dtype(source.aval.dtype), np.dtype(out.aval.dtype)
    if jnp.issubdtype(made, jnp.floating):
        return jnp.issubdtype(given, jnp.floating) or jnp.issubdtype(given, jnp.signedinteger)
    return (jnp.issubdtype(given, jnp.signedinteger) and jnp.issubdtype(made, jnp.signedinteger)
            and made.itemsize >= given.itemsize)


def _integer_range(dtype) -> Magnitude:
    """The power of two bounding every value of the integer `dtype`: a signed one of `bits` bits
    `2**(bits - 1)`, an unsigned one `2**bits`."""
    integers = jnp.iinfo(dtype)
    return integers.bits - (integers.min < 0)


def _bounded(equation: jcore.JaxprEqn, operands: list[Magnitude | None]) -> Magnitude | None:
    """A bound on the magnitude of `equation`'s output from its operands' (`operands`, `Magnitude`), None
    where its primitive proves none (an exponential of a value of any magnitude, a quotient), as its output's
    dtype holds it: a boolean at most 1 whatever made it (a tiny float converted is 1); an integer at most
    its dtype's range (`_integer_range`), within which a sum or a product wraps, and at it where an unsigned
    result may wrap from below zero (`uint32(0) - 1`, a negation, a conversion of int32's -1: `2**32 - 1`)
    and for a bitwise operation (int32's `not(-2**31)` is `2**31 - 1`); a float as the real arithmetic
    bounds it (`_real_bound`)."""
    aval = equation.outvars[0].aval
    if not isinstance(aval, ShapedArray):
        return None
    dtype = np.dtype(aval.dtype)
    if dtype == np.bool_:
        return UNIT
    real = _real_bound(equation, operands)
    if not jnp.issubdtype(dtype, jnp.integer):
        return real
    full = _integer_range(dtype)
    name = equation.primitive.name
    source = equation.invars[0].aval
    from_unsigned = isinstance(source, ShapedArray) and jnp.issubdtype(source.dtype, jnp.unsignedinteger)
    wraps = jnp.issubdtype(dtype, jnp.unsignedinteger) and (
        name in ("neg", "sub") or (name == "convert_element_type" and not from_unsigned))
    if name in BITWISE or wraps:
        return full
    return None if real is None else min(real, full)


def _real_bound(equation: jcore.JaxprEqn, operands: list[Magnitude | None]) -> Magnitude | None:
    """`_bounded`'s bound as the real arithmetic on the operands' bounds proves it (a float's, or an
    integer's before its dtype wraps it), each operand `|x| <= 2**a` (`y`'s `2**b`). Each rule's reason
    over the box of its operands, and its float rounding's slack (a float rounds to nearest, monotone, so a
    result whose exact value is under a representable power of two `2**c` rounds to at most it: the
    rules whose bound is such a power spend no slack):
    - a comparison, `is_finite`, `sign`: at most 1 by its definition;
    - `tanh`, `logistic`, `sin`, `cos`, `erf` of a real operand: at most 1 everywhere in the reals, and at
      most 2 as computed: a native implementation's polynomial is not certified within the real range (an
      ulp past 1 breaks a bound of exactly 1), so it is given one power; unbounded over a complex operand;
    - `add`, `sub`: `|x +- y| <= 2**a + 2**b <= 2**(max(a, b) + 1)`, monotone in each operand;
    - `mul`: `|x y| <= 2**(a + b)`, bilinear, its extreme at the box's corners;
    - `max`, `min`, a selection, `neg`, `abs`, a conversion, a layout change, a maximum or minimum along an
      axis: no larger than an operand (`abs` least at its critical point 0, largest at a corner);
    - `floor`, `ceil`, `round`: at most `max(2**a, 1)` (`ceil(2**-126)` is 1);
    - `square`, `integer_pow` of `y >= 1`: `|x|**y <= 2**(y a)`, monotone in `|x|` (an even power least at
      its critical point 0);
    - `exp`, `exp2` of `|x| <= 2**a <= 64`: at most `e**(2**a)`, `2**(2**a log2 e)`, bounded by the power
      above its floor: one power of slack for XLA's rounding (float16's `exp2(4)` is 16.02);
    - a sum along axes of `n` elements: each term's range summed, `n 2**a <= 2**(a + ceil(log2 n))`
      (separable; each partial sum rounds under the power, a float of under `2**p` terms exactly, more
      rounding back to it);
    - a contraction of `n` terms: `n 2**(a + b)`, a sum of bilinear terms, separable alike, where its mode
      rounds each operand once to nearest (a power of two rounds to itself) and accumulates monotonely; a
      mode that splits its operands into parts (bf16 x3, x6, x9, TF32 x3, `Precision.HIGH`'s bf16 x3 on a
      TPU) sums their products' rounding too, given one power; a mode whose rounding no source states (FP8
      fast accumulation) proves none (`_contraction_slack`).
    Any other equation proves none (a quotient, a logarithm, a power of a float exponent)."""
    name = equation.primitive.name
    real = not any(isinstance(atom.aval, ShapedArray) and jnp.issubdtype(atom.aval.dtype, jnp.complexfloating)
                   for atom in equation.invars)
    if name in BITWISE:
        # An integer's its dtype's range, a boolean's 1 (`_bounded`).
        return None
    if name in COMPARISONS or name in ("is_finite", "sign"):
        return UNIT
    if name in BOUNDED and real:
        # One power past the real range for a native implementation's rounding.
        return UNIT + 1
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
        # An exponential of a value under 64: under 2**(64 log2(e)), a power above the exact one's floor, as
        # XLA's rounds (float16's exp2(4) is 16.02).
        return math.floor(2 ** given[0] * (math.log2(math.e) if name == "exp" else 1.0)) + 1
    if name == "reduce_sum":
        shape = _shape(equation.invars[0])
        return given[0] + _log2(math.prod(shape[axis] for axis in equation.params["axes"]))
    if name == "dot_general":
        slack = _contraction_slack(equation.params.get("precision"))
        if slack is None:
            return None
        (contracted, _), _ = equation.params["dimension_numbers"]
        shape = _shape(equation.invars[0])
        return given[0] + given[1] + _log2(math.prod(shape[axis] for axis in contracted)) + slack
    return None


SPLIT_PRESETS = frozenset({"BF16_BF16_F32_X3", "BF16_BF16_F32_X6", "BF16_BF16_F32_X9", "TF32_TF32_F32_X3"})
"""The dot algorithms that split each operand into parts and sum their products (`_contraction_slack`)."""
SINGLE_PRESETS = frozenset({"DEFAULT", "F16_F16_F16", "F16_F16_F32", "BF16_BF16_BF16", "BF16_BF16_F32",
                            "TF32_TF32_F32", "F32_F32_F32", "F64_F64_F64"})
"""The dot algorithms that round each operand once to nearest and accumulate (`_contraction_slack`)."""


def _contraction_slack(precision) -> int | None:
    """The powers a contraction's bound adds for its mode (`dot_general`'s `precision`): none where it rounds
    each operand once to nearest (`SINGLE_PRESETS`, `Precision.DEFAULT` and `HIGHEST`); one where it splits
    them into parts whose products it sums (`SPLIT_PRESETS`, `Precision.HIGH`: a TPU's bf16 x3); None, no
    bound, where no source states its rounding (an FP8 preset's fast accumulation, an algorithm given by
    its own types)."""
    if precision is None:
        return 0
    if isinstance(precision, jax.lax.DotAlgorithmPreset):
        return 0 if precision.name in SINGLE_PRESETS else 1 if precision.name in SPLIT_PRESETS else None
    if isinstance(precision, tuple) and all(isinstance(part, jax.lax.Precision) for part in precision):
        return 1 if jax.lax.Precision.HIGH in precision else 0
    if isinstance(precision, jax.lax.Precision):
        return 1 if precision == jax.lax.Precision.HIGH else 0
    return None


def _log2(value: float) -> int:
    """`ceil(log2(value))`, `NOTHING` for 0: a magnitude's exponent bounding `value` from above."""
    return NOTHING if value == 0 else math.ceil(math.log2(value))


def _magnitude(value) -> Magnitude | None:
    """A constant's magnitude, its elements' largest (a complex one's modulus); None for one of no number (a
    key's). A number of any dtype, ml_dtypes' (bfloat16, float8) among them, which numpy's type tree omits."""
    array = np.asarray(value)
    if not (jnp.issubdtype(array.dtype, jnp.number) or array.dtype == bool):
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


def _typed(dtypes: dict, magnitudes: dict, term: Term | Skeleton, atom, magnitude: Magnitude | None) -> None:
    """`term`'s dtype (`atom`'s) and its magnitude, the first where one term names two values."""
    named = key(term)
    if isinstance(atom.aval, ShapedArray):
        dtypes.setdefault(named, np.dtype(atom.aval.dtype))
    magnitudes.setdefault(named, magnitude)


@functools.cache
def stocked() -> Mapping[str, tuple[jcore.Primitive, Mapping]]:
    """The primitives a stable rule writes where the program held none (`rules.stock`), each at the
    parameters JAX binds it with by default, read from JAX's own trace of them."""
    # An abstract scalar: the pass reads a jit's trace after it is made, where a process compiling for a
    # device it does not hold (a compile-only GPU) places no array.
    scalar = jax.ShapeDtypeStruct((), jnp.float32)
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


def _held(value: np.ndarray | complex) -> np.ndarray | complex:
    """A literal's value as a region holds it, its own copy and its type (a `TypedNdArray`'s aval, its weak
    type and placement): JAX's literal may view a NumPy scalar a function captured, which its caller may
    change after a region is derived and kept (`program._derived`)."""
    if isinstance(value, TypedNdArray):
        return TypedNdArray(np.array(value), value.aval)
    return np.array(value) if isinstance(value, np.ndarray) else value


def _shape(atom) -> tuple[int, ...]:
    """An array value's shape; a value of no shape (a token, a key's) has none the rules read."""
    return tuple(atom.aval.shape) if isinstance(atom.aval, ShapedArray) else ()


def _term[Node: (Term, Skeleton)](equation: jcore.JaxprEqn, read: Callable[..., Node],
                                   param: Callable) -> Node | None:
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


def _literal(argument: object, kind: type[i64] | type[String] | type[f64]) -> int | str | float:
    """A literal argument's value: a term's argument as `get_callable_args` gives it, which egglog types by
    its constructor's parameter (`i64Like`, `StringLike`), each a literal of `kind` in a term."""
    if isinstance(argument, kind) and (value := get_literal_value(argument)) is not None:
        return value
    raise TypeError(f"{argument!r} is no {kind.__name__} literal")


def integer(argument: i64Like) -> int:
    """An `i64` literal's value (`_literal`): a term's indices."""
    return int(_literal(argument, i64))


def text(argument: StringLike) -> str:
    """A `String` literal's value (`_literal`): a term's operation names."""
    return str(_literal(argument, String))


def real(argument: f64Like) -> float:
    """An `f64` literal's value (`_literal`): a term's float constants."""
    return float(_literal(argument, f64))


def value_of(term: Term, value: Callable[[Term], jax.Array], region_: Region, leaves: Sequence) -> jax.Array:
    """One term's value, its operands' from `value`, each equation of several operands bound over them aligned
    as the lax function it came from aligns them (`jax_core.auto_insert_reshard`): a literal or a constant
    beside a value a checked `shard_map`'s manual axes vary varies as it does."""
    if (args := get_callable_args(term, Term.leaf)) is not None:
        return leaves[integer(args[0])]
    if (args := get_callable_args(term, Term.literal)) is not None:
        return region_.literals[integer(args[0])]

    def bound(left: Term, right: Term, index: i64Like) -> jax.Array:
        primitive, given = region_.params[integer(index)]
        return primitive.bind(*jax_core.auto_insert_reshard(value(left), value(right)), **given)

    if (args := get_callable_args(term, Term.dot)) is not None:
        return bound(args[0], args[1], args[2])
    for constructor in (Term.binary, Term.compare):
        if (args := get_callable_args(term, constructor)) is not None:
            # An operation's name stands between its two operands.
            return bound(args[0], args[2], args[3])
    if (args := get_callable_args(term, Term.select)) is not None:
        operands = jax_core.auto_insert_reshard(*(value(operand) for operand in args))
        return jax.lax.select_n_p.bind(*operands)
    # Each other constructor's primitive parameters, by index, last after its operand and its other arguments.
    for constructor in (Term.unary, Term.convert, Term.shaped, Term.reduce):
        if (args := get_callable_args(term, constructor)) is not None:
            primitive, given = region_.params[integer(args[-1])]
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
