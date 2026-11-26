"""The term language Dew's rewrite holds a program in (docs/design/algebra.md §6.2): one egglog sort, `Term`,
a constructor for each kind of equation of the language, and the conversion of a jaxpr's equations into terms
and of an extracted term back into equations.

A term names its equation's primitive and refers to the primitive's other parameters by an index into the
region's table of them (`Region.params`), and a literal to its exact value in the region's table of literals,
so a rule rewrites the math and keeps each parameter and constant as the program wrote it. Every term made
from an equation states its shape (`rules.shape`), which rules read before they join or move values."""

from __future__ import annotations

import dataclasses
import enum
import functools
import hashlib
import math
import threading
import types
import weakref
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from typing import Literal, NamedTuple, overload

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
from egglog.declarations import (
    Declarations,
    FunctionSignature,
    HasDeclarations,
    TypeOrVarRef,
    TypeRefWithVars,
)
from egglog.runtime import RuntimeExpr
from jax._src import ad_util, core as jax_core
from jax._src.literals import TypedNdArray
from jax.core import ShapedArray
from jax.experimental.layout import Layout
from jax.extend import core as jcore
from jax.typing import DTypeLike

from dew.algebra._proofs import (
    NOTHING,
    NOWHERE,
    UNIT,
    WHOLE,
    Apart,
    Magnitude,
    Proven,
    _apart_of,
    _constant_apart,
    _declared,
    _declared_apart,
    _deep_nonlang,
    _deep_refine,
    _held_apart,
    _host_array,
    _log2,
    _magnitude,
    _NormTrack,
    _reduced,
    _shape,
    _shared_sign,
    _side_matches,
    _SoftmaxTrack,
    _within,
    dtype_lossless_in as dtype_lossless_in,
    typed_key,
)
from dew.nn import programs


@dataclasses.dataclass(frozen=True)
class Operation:
    """One of JAX's operations as the term language speaks it (§1), declared on its primitive object
    (`OPERATIONS`): its `symbol` in the language (the primitive's own name), the term it is written as
    (`kind`: a binary or unary function, a comparison, a layout change, a reduction over its `monoid`, a cast,
    a selection, a product), and what its value proves of its operand's (`_real_bound`, `_bounded`): a
    magnitude at most 1 over a real operand of any magnitude (`bounded`; over a complex one, `sin(pi/2 +
    89j)` is 2.2e38), one no larger than its operand's (`keeps_magnitude`), an integer it rounds to, no
    larger than its operand's or 1 (`integral`: `ceil(2**-126)` is 1), a bitwise result, a boolean's at most
    1 and an integer's any value of its dtype (`bitwise`: int32's `not(-2**31)` is `2**31 - 1`), a value at
    most 0 where its operand is (`keeps_sign`; a conversion where it keeps the sign, `_keeps_sign`). Which
    units an operation's work is the kernels' counter states (`emission.SPECIAL_FUNCTIONS`, `cost.work`)."""

    symbol: str
    kind: str
    monoid: str | None = None
    bounded: bool = False
    keeps_magnitude: bool = False
    integral: bool = False
    bitwise: bool = False
    keeps_sign: bool = False


OPERATIONS: dict[jcore.Primitive, Operation] = {}
"""The language's operations, each by its JAX primitive object: elementwise arithmetic and comparisons, casts,
the layout changes, the reductions a rule reads, and products (`dot_general`); an optimization barrier among
the unaries, whose value is its operand's, which no rule rewrites through (a value rounded once before its
readers, as a declaration's dropped products are). JAX's tangent sum (`add_any`) is an add only
for ordinary, equal-shape floating arrays; other equations outside the language are opaque, including
another library's primitive of the same name."""


def _speaks(kind: str, *primitives: jcore.Primitive, **facts) -> None:
    """Declares `primitives` operations of the language of `kind`, each under its own name."""
    for primitive in primitives:
        OPERATIONS[primitive] = Operation(primitive.name, kind, **facts)


_speaks("binary", jax.lax.add_p, jax.lax.sub_p, jax.lax.mul_p, jax.lax.div_p, jax.lax.max_p, jax.lax.min_p,
        jax.lax.atan2_p, jax.lax.rem_p, jax.lax.nextafter_p)
OPERATIONS[ad_util.add_jaxvals_p] = OPERATIONS[jax.lax.add_p]
_speaks("binary", jax.lax.pow_p)
_speaks("binary", jax.lax.and_p, jax.lax.or_p, jax.lax.xor_p, bitwise=True)
_speaks("unary", jax.lax.sign_p, jax.lax.integer_pow_p, jax.lax.square_p, jax.lax.erf_inv_p,
        jax.lax.is_finite_p, jax.lax.optimization_barrier_p)
_speaks("unary", jax.lax.neg_p, jax.lax.abs_p, keeps_magnitude=True)
_speaks("unary", jax.lax.exp_p, jax.lax.exp2_p, jax.lax.log_p, jax.lax.log2_p, jax.lax.log1p_p,
        jax.lax.expm1_p, jax.lax.sqrt_p, jax.lax.rsqrt_p)
_speaks("unary", jax.lax.tanh_p, jax.lax.logistic_p, jax.lax.sin_p, jax.lax.cos_p, jax.lax.erf_p,
        bounded=True)
_speaks("unary", jax.lax.floor_p, jax.lax.ceil_p, jax.lax.round_p, integral=True)
_speaks("unary", jax.lax.not_p, bitwise=True)
_speaks("comparison", jax.lax.lt_p, jax.lax.le_p, jax.lax.eq_p, jax.lax.ne_p, jax.lax.ge_p, jax.lax.gt_p)
_speaks("shape", jax.lax.broadcast_in_dim_p, jax.lax.reshape_p, jax.lax.transpose_p, jax.lax.squeeze_p,
        keeps_magnitude=True, keeps_sign=True)
_speaks("reduction", jax.lax.reduce_sum_p, monoid="sum")
_speaks("reduction", jax.lax.reduce_max_p, monoid="max", keeps_magnitude=True, keeps_sign=True)
_speaks("reduction", jax.lax.reduce_min_p, monoid="min", keeps_magnitude=True, keeps_sign=True)
_speaks("cast", jax.lax.convert_element_type_p, keeps_magnitude=True, keeps_sign=True)
_speaks("select", jax.lax.select_n_p)
_speaks("clamp", jax.lax.clamp_p)
_speaks("product", jax.lax.dot_general_p)


class Written(NamedTuple):
    """How an operation of a kind is written as a term (`Term`): the constructor it is a call of, the
    operands it reads, and what the term holds of the operation beside them: its symbol, its reduction's
    monoid, or nothing where the kind has one operation."""

    constructor: str
    arity: int
    names: Literal["symbol", "monoid"] | None


WRITTEN = {"binary": Written("binary", 2, "symbol"), "comparison": Written("compare", 2, "symbol"),
           "unary": Written("unary", 1, "symbol"), "shape": Written("shaped", 1, "symbol"),
           "reduction": Written("reduce", 1, "monoid"), "cast": Written("convert", 1, None),
           "select": Written("select", 3, None), "clamp": Written("clamp", 3, None),
           "product": Written("dot", 2, None)}
"""Each kind of operation (`Operation.kind`) as its term, the one statement of the correspondence: an
equation is written as a term by it (`_term`, `spoken_as`) and a term's operation read back by it
(`written_as`, a rule's left side's head)."""


def _symbols(chosen: Callable[[Operation], bool]) -> frozenset[str]:
    """The symbols of the language's operations `chosen` holds of."""
    return frozenset(operation.symbol for operation in OPERATIONS.values() if chosen(operation))


COMPARISONS = _symbols(lambda operation: operation.kind == "comparison")
"""The language's comparisons' symbols, read from the declarations (`OPERATIONS`)."""
RESCALE = "rescale"
"""A binary no program holds: a derived walk's rescale of a carried state from its first operand's anchor to
its second's where the declaration states its own (`derive.Reduction.rescale`), evaluated by that function
(`forms.evaluator`) and priced as an exponential (`cost.work`); no rule rewrites it."""


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
    def clamp(self, operand: Term, upper: Term) -> Term: ...

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


class Skeleton:
    """A term's structure without its egglog expression: `Term`'s constructors over plain tuples (`form`: its
    constructor's name, its operands' skeletons and its other arguments), equal exactly where the terms are
    (`region`'s `forms`). What the pass reads of a window to decide whether a rule may fire in it
    (`rules.reorderable`), and of a whole program to bound its windows' inputs (`dew.algebra.program._held`),
    at a fraction of its terms' cost: a Dory-4B training step's 1018 windows, every one refused, spent most of
    the pass building their terms (FS-OVERHEAD-DORY-ON-P3).

    Hash-consed: a skeleton is made once for its form (`made`, the process's table of the skeletons alive), so
    two equal skeletons are one object, compared by identity and hashed by it, in constant time. A program's
    value read by many others is one skeleton many forms hold, and equal values traced apart (a computation
    traced twice over the same inputs, which JAX does not merge) one skeleton too: compared as plain nested
    tuples, each comparison walked every path through that sharing, 2**depth of them (a LAMB step's whole
    program never finished; train 902 F1)."""

    __slots__ = ("__weakref__", "form")

    def __init__(self, form: tuple) -> None:
        self.form = form

    def __repr__(self) -> str:
        return f"Skeleton{self.form!r}"

    @classmethod
    def made(cls, form: tuple) -> Skeleton:
        """The one skeleton of `form` (its operands' skeletons in it, each already one): the live one where
        the process holds it, else a new one, held while it lives."""
        with _MAKING:
            held = _MADE.get(form)
            if held is None:
                held = _MADE[form] = cls(form)
            return held

    @classmethod
    def leaf(cls, index: int) -> Skeleton:
        return cls.made(("leaf", index))

    @classmethod
    def literal(cls, index: int) -> Skeleton:
        return cls.made(("literal", index))

    def binary(self, name: str, other: Skeleton, params: int) -> Skeleton:
        return Skeleton.made(("binary", self, name, other, params))

    def compare(self, name: str, other: Skeleton, params: int) -> Skeleton:
        return Skeleton.made(("compare", self, name, other, params))

    def unary(self, name: str, params: int) -> Skeleton:
        return Skeleton.made(("unary", self, name, params))

    def select(self, false: Skeleton, true: Skeleton) -> Skeleton:
        return Skeleton.made(("select", self, false, true))

    def clamp(self, operand: Skeleton, upper: Skeleton) -> Skeleton:
        return Skeleton.made(("clamp", self, operand, upper))

    def convert(self, params: int) -> Skeleton:
        return Skeleton.made(("convert", self, params))

    def shaped(self, name: str, params: int) -> Skeleton:
        return Skeleton.made(("shaped", self, name, params))

    def reduce(self, monoid: str, axes: Vec[i64], params: int) -> Skeleton:
        # The reduced axes are among the parameters `params` indexes.
        del axes
        return Skeleton.made(("reduce", self, monoid, params))

    def dot(self, other: Skeleton, params: int) -> Skeleton:
        return Skeleton.made(("dot", self, other, params))


_MADE: weakref.WeakValueDictionary[tuple, Skeleton] = weakref.WeakValueDictionary()
"""Each live skeleton by its form (`Skeleton.made`): a form's operands are skeletons, hashed and compared by
identity, so a form's hash and its comparison take its own arguments alone."""
_MAKING = threading.Lock()


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
    products: dict[int, Product] = dataclasses.field(default_factory=dict)
    """Each product's dtypes (`Product`) by its parameters' index (`Term.dot`'s; a rule's product its
    source's)."""
    literal_keys: frozenset[Hashable] = frozenset()
    """Each literal's term's key (`literals`' terms)."""
    indexed: dict[Hashable, int] = dataclasses.field(default_factory=dict)
    """Each interned entry of `params` by what makes two entries one (`interned`)."""
    held: dict[jcore.Var, Magnitude | None] = dataclasses.field(default_factory=dict)
    """Each value's magnitude (`magnitudes`') by the jaxpr's variable that names it: what a window cut from
    the jaxpr declares of its inputs (`dew.algebra.program._windowed`)."""
    apart: dict[jcore.Var, Apart] = dataclasses.field(default_factory=dict)
    """What the jaxpr proves of each value beside its magnitude (`Apart`: a number, finite, its side and
    floor) by the variable that names it: with `held`, what a window cut from the jaxpr declares of its
    inputs (`Proven`)."""
    floors: dict[Hashable, Apart] = dataclasses.field(default_factory=dict)
    """`apart` of each term held away from zero, by term text: a number whose floor (`Apart.floor`) is
    known, the floor and side a guarded reuse reads of a divisor (`rules.log2_floor`, `rules.positive`)."""
    nonnegative: frozenset[Hashable] = frozenset()
    """Keys of terms proven nonnegative numbers (`Apart.number` and `Apart.sign > 0`)."""
    unplaced_full_fp32: frozenset[Hashable] = frozenset()
    """Keys of full-fp32 dots traced without explicit or manual placement."""
    extents: dict[Hashable, tuple[int, ...]] = dataclasses.field(init=False)

    def __post_init__(self):
        self.extents = {}
        for term, extent in self.shapes:
            self.extents.setdefault(key(term), extent)

    def interned(self, primitive: jcore.Primitive, values: Mapping, product: Product | None = None) -> int:
        """The index of `primitive` at `values` in the region's table (`params`), added where it holds none
        (`_interned`): a rule set's own parameters (a transposed product's plain one) by the rule the
        region's equations were interned by."""
        return _interned(self.params, self.products, self.indexed, primitive, values, product)

    @functools.cached_property
    def made_by(self) -> dict[int, Opaque[Node]]:
        """Each leaf an equation outside the language makes (`opaque`), by its index, that equation."""
        return {leaf: opaque for opaque in self.opaque for leaf in opaque.outputs}

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
    step's pass); a skeleton itself, one object for its form (`Skeleton`)."""
    if isinstance(term, Skeleton):
        return term
    if not isinstance(term, RuntimeExpr):
        raise TypeError(f"{term!r} is no egglog term")
    return term.__egg_typed_expr__


class Product(NamedTuple):
    """A product's dtypes (`Region.products`): its operands' (None where one is no array), which key its
    mode (`cost.work`), and the one it makes as traced, which its operands' dtypes and parameters fix, kept
    apart from its parameters, which the region replays as they were traced (a product of two dtypes'
    unset preferred type stays so, base review 3232beb7c): never another library's promotion of its
    operands' (NumPy's refuses a bfloat16 beside a float16, base review 0b8b7ed98 F1), nor a trace of a
    product made for it (base review 7754e5fde F3)."""

    operands: tuple[np.dtype | None, ...]
    made: np.dtype


def _interned(params: list, products: dict[int, Product], indexed: dict[Hashable, int],
              primitive: jcore.Primitive, values: Mapping, product: Product | None = None) -> int:
    """The index of `primitive` at `values` in a region's table `params`, added where it holds none: two
    entries are one where their primitive and parameters are (`parameters_key`) and, of a product, its
    dtypes too (`Product`), kept in `products` (`Region.products`). The one interning of a region's
    parameters, while it is built (`_region`) and after (`Region.interned`)."""
    entry = (*parameters_key(primitive, values), product)
    if entry not in indexed:
        indexed[entry] = len(params)
        params.append((primitive, dict(values)))
        if product is not None:
            products[indexed[entry]] = product
    return indexed[entry]


def parameters_key(primitive: jcore.Primitive, values: Mapping) -> tuple[jcore.Primitive, Hashable]:
    """An equation's primitive and parameters as a region holds them: two equations of one primitive whose
    parameters are one (`parameter_key`) share one entry of its table (`Region.params`), the first's."""
    if not values:
        return primitive, ()
    if len(values) == 1:
        name, value = next(iter(values.items()))
        return primitive, ((name, parameter_key(value)),)
    return primitive, tuple((name, parameter_key(value)) for name, value in sorted(values.items()))


@functools.lru_cache(maxsize=1 << 16)
def _abstract_key(value: ShapedArray) -> Hashable:
    """An abstract value's key (`parameter_key`'s): its shape, dtype, weak type, placement, memory space,
    layout and manual axes, kept for each abstract value (JAX's own value equality and hash), which every
    window's structure reads of its every equation's outputs: a program's few distinct types, keyed once each,
    where each read built the key anew (2.27 million keys of a Qwen3.5-4B start's 40 passes, row 6782). Its
    placement by its partitions and its mesh's axes and memory, not the kind of device the mesh was made of,
    which a step compiled ahead for a GPU and the run on that GPU share no name of."""
    sharding, manual = value.sharding, value.manual_axis_type
    mesh = sharding.mesh
    placement = (parameter_key(sharding.spec), parameter_key(mesh.axis_sizes),
                 parameter_key(mesh.axis_names), parameter_key(mesh.axis_types),
                 parameter_key(sharding.memory_kind))
    # Its dtype by its name, a PRNG key's extended one's too (`key<fry>`).
    return ("abstract", parameter_key(value.shape), ("dtype", value.dtype.name), value.weak_type,
            placement, parameter_key(value.memory_space), parameter_key(value.layout),
            parameter_key((manual.varying, manual.unreduced, manual.reduced, manual.unreduced_kind)))


def _cycle_guard(value, kind: type, seen: set[int], *, reject_device: bool,
                 build: Callable[[], Hashable]) -> Hashable:
    """Run `build()` with `id(value)` on active ancestor stack `seen`; on a cycle raise `_Unshareable` when
    `reject_device` else key `value` by identity."""
    obj_id = id(value)
    if obj_id in seen:
        if reject_device:
            raise _Unshareable
        return "object", _named(kind), _Identity(value)
    seen.add(obj_id)
    try:
        return build()
    finally:
        seen.discard(obj_id)


def _container_param_key(value, kind: type, recurse: Callable, seen: set[int], *,
                         reject_device: bool) -> Hashable | None:
    """Cycle-safe structural key for Python containers and `functools.partial`, or None for other types."""
    if isinstance(value, functools.partial):
        return _cycle_guard(value, kind, seen, reject_device=reject_device, build=lambda: (
            "partial", recurse(value.func), recurse(value.args), recurse(value.keywords)))
    if isinstance(value, (tuple, list)) and not isinstance(value, jax.sharding.PartitionSpec):
        return _cycle_guard(value, kind, seen, reject_device=reject_device, build=lambda: (
            "sequence", _named(kind), tuple(recurse(part) for part in value)))
    if isinstance(value, (set, frozenset)):
        return _cycle_guard(value, kind, seen, reject_device=reject_device, build=lambda: (
            "set", tuple(sorted((recurse(part) for part in value), key=repr))))
    if isinstance(value, Mapping):
        return _cycle_guard(value, kind, seen, reject_device=reject_device, build=lambda: (
            "mapping", tuple(sorted(((recurse(k), recurse(v)) for k, v in value.items()), key=repr))))
    return None


def parameter_key(value) -> Hashable:
    """Structural parameter key (`parameters_key`, `program.structure`, `tuning.program`): plain data across
    processes; unhashable, cyclic, or identity-equality objects by identity."""
    return _parameter_key(value, None)


def _parameter_key(value, seen: set[int] | None) -> Hashable:
    kind = type(value)
    if value is None or kind in (bool, str, int):
        return kind.__name__, value
    if kind is float:
        return "float", value.hex()
    if kind is ShapedArray:
        return _abstract_key(value)
    if kind in (tuple, list):
        active = set() if seen is None else seen
        return _cycle_guard(value, kind, active, reject_device=False, build=lambda: (
            "sequence", _named(kind), tuple(_parameter_key(part, active) for part in value)))
    if isinstance(value, jcore.Jaxpr):
        digest, opaques = jaxpr_digest(value)
        held = ("object", "subkey", _SubKey(digest, opaques)) if opaques else ()
        consts = tuple(_nested_const_key(const) for const in value.consts) if value.consts else ()
        return "jaxpr", digest, held, consts
    return _parameter_key_other(value, kind, set() if seen is None else seen)


def _array_key(value, kind: type) -> Hashable:
    """Cross-process array digest (`parameter_key`), never transferring non-addressable global shards."""
    if isinstance(value, jax.Array) and not value.is_fully_addressable:
        return "object", _named(kind), _Identity(value)
    _, array = held_bits(value)
    return ("array", str(value.dtype), array.shape, str(array.dtype),
            hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest())


def _parameter_key_other(value, kind: type, seen: set[int]) -> Hashable:
    """`parameter_key` fallback for NumPy scalars/arrays, enums, layouts, shardings, dataclasses, mappings,
    and opaque objects."""
    if isinstance(value, (bool, str)):
        return kind.__name__, value
    if isinstance(value, (int, np.integer)):
        return "int", int(value)
    if isinstance(value, (float, np.floating)):
        return "float", float(value).hex()
    if isinstance(value, complex):
        return "complex", value.real.hex(), value.imag.hex()
    if isinstance(value, np.dtype):
        return "dtype", value.name
    if isinstance(value, enum.Enum):
        return "enum", _named(kind), value.name
    if isinstance(value, (np.ndarray, jax.Array)) and not isinstance(value, jax_core.Tracer):
        return _array_key(value, kind)
    if isinstance(value, ShapedArray):
        return _abstract_key(value)
    if value is Layout.AUTO:
        return "layout", "auto"
    if isinstance(value, Layout):
        return ("layout", _parameter_key(value.major_to_minor, seen), _parameter_key(value.tiling, seen),
                value.sub_byte_element_size_in_bits)
    if isinstance(value, jax.sharding.Mesh):
        devices = () if value.empty else value.device_ids
        return ("mesh", _parameter_key(devices, seen), _parameter_key(value.axis_names, seen),
                _parameter_key(value.axis_types, seen))
    if isinstance(value, jax.sharding.AbstractMesh):
        return ("abstract mesh", _parameter_key(value.axis_sizes, seen),
                _parameter_key(value.axis_names, seen), _parameter_key(value.axis_types, seen),
                _parameter_key(value.abstract_device, seen))
    if isinstance(value, jax.sharding.NamedSharding):
        _, arguments = value.__reduce__()
        return "named sharding", _parameter_key(arguments, seen)
    if isinstance(value, jax.sharding.PartitionSpec):
        return ("partitions", _parameter_key(tuple(value), seen), _parameter_key(value.unreduced, seen),
                _parameter_key(value.reduced, seen), _parameter_key(value.unreduced_kind, seen))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        if kind.__eq__ is object.__eq__:
            return "object", _named(kind), _Identity(value)
        return _cycle_guard(value, kind, seen, reject_device=False, build=lambda: (
            "record", _named(kind), tuple((f.name, _parameter_key(getattr(value, f.name), seen))
                                          for f in dataclasses.fields(value))))
    if (boxed := _container_param_key(value, kind, lambda part: _parameter_key(part, seen), seen,
                                      reject_device=False)) is not None:
        return boxed
    if isinstance(value, types.FunctionType):
        return "function", value.__module__, value.__qualname__
    try:
        hash(value)
    except TypeError:
        return "object", _named(kind), _Identity(value)
    return "object", _named(kind), value


class _SubKey:
    """DAG-memoized structural key for a `jcore.Jaxpr` with O(1) hash and O(V) equality."""

    __slots__ = ("_hash", "digest", "opaques")

    def __init__(self, digest: str, opaques: tuple[Hashable, ...]) -> None:
        self.digest, self.opaques, self._hash = digest, opaques, hash((digest, opaques))

    def __repr__(self) -> str:
        return f"_SubKey({self.digest!r})"

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other) -> bool:
        return (self is other or (isinstance(other, _SubKey) and self._hash == other._hash
                                  and self.digest == other.digest and _subkeys_equal(self, other)))


def _subkey_child(entry) -> _SubKey | None:
    if (isinstance(entry, tuple) and len(entry) == 3 and entry[0] == "object"
            and entry[1] == "subkey" and isinstance(entry[2], _SubKey)):
        return entry[2]
    return None


def _subkeys_equal(left: _SubKey, right: _SubKey) -> bool:
    """Iterative pair-memoized equality of two `_SubKey` DAGs, O(V) even across distinct copies."""
    seen: set[tuple[int, int]] = set()
    stack: list[tuple[_SubKey, _SubKey]] = [(left, right)]
    while stack:
        first, second = stack.pop()
        if first is second:
            continue
        pair = (id(first), id(second))
        if pair in seen:
            continue
        seen.add(pair)
        if (first._hash != second._hash or first.digest != second.digest
                or len(first.opaques) != len(second.opaques)):
            return False
        for lhs, rhs in zip(first.opaques, second.opaques, strict=True):
            if lhs is rhs:
                continue
            lhs_child, rhs_child = _subkey_child(lhs), _subkey_child(rhs)
            if lhs_child is not None and rhs_child is not None:
                stack.append((lhs_child, rhs_child))
            elif lhs != rhs:
                return False
    return True


_JAXPR_DIGESTS: weakref.WeakKeyDictionary[jcore.Jaxpr, tuple[str, tuple[Hashable, ...]]] = (
    weakref.WeakKeyDictionary())
_IN_PROCESS_DIGESTS: weakref.WeakKeyDictionary[jcore.Jaxpr, _SubKey] = weakref.WeakKeyDictionary()
_BODY_KEYS: weakref.WeakKeyDictionary[jcore.Jaxpr, _SubKey | None] = weakref.WeakKeyDictionary()
_PROVENANCE_KEYS: weakref.WeakKeyDictionary[jcore.Jaxpr, str] = weakref.WeakKeyDictionary()


class _Unshareable(Exception):
    """Raised internally by `body_key` when a body holds nested consts, cycles, or device arrays."""


def _nested_const_key(const) -> Hashable:
    if isinstance(const, jax.Array) and not isinstance(const, jax_core.Tracer):
        return "object", _named(type(const)), _Identity(const)
    return parameter_key(const)


def _extended_dtype(dtype) -> Hashable | None:
    if isinstance(dtype, (str, np.dtype)):
        return None
    return ("object", "extended_dtype", dtype) if jnp.issubdtype(dtype, jax.dtypes.extended) else None


def _in_process_digest(jaxpr: jcore.Jaxpr) -> _SubKey:
    inner = jaxpr.jaxpr
    # A const-free body already keyed for whole-body reuse has the same structural digest. Only that
    # direction of reuse is safe: the permissive key can hold device buffers or identity-only parameters.
    cached = _BODY_KEYS.get(inner) or _IN_PROCESS_DIGESTS.get(inner)
    if cached is not None:
        return cached
    sub = _SubKey(*_body_digest(inner, in_process=True, reject_device=False))
    _IN_PROCESS_DIGESTS[inner] = sub
    return sub


def _body_key_of(jaxpr: jcore.Jaxpr) -> _SubKey | None:
    if jaxpr.consts or jaxpr.constvars:
        return None
    inner = jaxpr.jaxpr
    if inner in _BODY_KEYS:
        return _BODY_KEYS[inner]
    try:
        made: _SubKey | None = _SubKey(*_body_digest(inner, in_process=True, reject_device=True))
    except _Unshareable:
        made = None
    _BODY_KEYS[inner] = made
    return made


def _dataclass_param_key(value, kind: type, *, reject_device: bool, seen: set[int]) -> Hashable:
    """In-process structural key for a dataclass preserving class, identity-eq, extra attrs, and cycles."""
    fields = dataclasses.fields(value)
    names = {field.name for field in fields}
    inst = value.__dict__ if any("__dict__" in cls.__dict__ for cls in kind.__mro__) else None
    extra_slots = any(
        not set((s,) if isinstance(s, str) else s).issubset(names | {"__dict__", "__weakref__"})
        for cls in kind.__mro__ if (s := cls.__dict__.get("__slots__")) is not None)
    if kind.__eq__ is object.__eq__ or (inst is not None and not set(inst).issubset(names)) or extra_slots:
        if reject_device:
            raise _Unshareable
        return "object", _named(kind), _Identity(value)
    return _cycle_guard(
        value, kind, seen, reject_device=reject_device,
        build=lambda: ("record", ("object", "class", kind), tuple(
            (f.name, _in_process_param_key(getattr(value, f.name), reject_device=reject_device, seen=seen))
            for f in fields)))


def _in_process_param_key(value, *, reject_device: bool, seen: set[int] | None = None) -> Hashable:
    """In-process structural key for an equation parameter preserving executable callable, primitive, and
    extended-dtype identities while never copying `jax.Array` device buffers to host."""
    kind = type(value)
    if value is None or kind in (bool, str, int):
        return kind.__name__, value
    if kind is float:
        return "float", value.hex()
    if isinstance(value, ShapedArray):
        return _abstract_key(value), _extended_dtype(value.dtype)
    if isinstance(value, jcore.Jaxpr):
        if reject_device:
            if (child := _body_key_of(value)) is None:
                raise _Unshareable
            return "jaxpr", child.digest, ("object", "subkey", child), ()
        child = _in_process_digest(value)
        consts = tuple(_nested_const_key(c) for c in value.consts) if value.consts else ()
        return "jaxpr", child.digest, ("object", "subkey", child), consts
    if isinstance(value, jax.Array) and not isinstance(value, jax_core.Tracer):
        if reject_device:
            raise _Unshareable
        return "object", _named(kind), _Identity(value)
    if isinstance(value, (types.FunctionType, types.BuiltinFunctionType)):
        return "object", _named(kind), value
    active = set() if seen is None else seen
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _dataclass_param_key(value, kind, reject_device=reject_device, seen=active)
    if (boxed := _container_param_key(
            value, kind, lambda part: _in_process_param_key(part, reject_device=reject_device, seen=active),
            active, reject_device=reject_device)) is not None:
        return boxed
    if callable(value) and not isinstance(value, (type, np.dtype, jax.sharding.Mesh)):
        if reject_device:
            raise _Unshareable
        if kind.__eq__ is object.__eq__:
            return "object", _named(kind), _Identity(value)
        try:
            hash(value)
        except TypeError:
            return "object", _named(kind), _Identity(value)
        return "object", _named(kind), value
    return _parameter_key_other(value, kind, active)


def _collect_opaque(key, opaques: list[Hashable]) -> Hashable:
    """Hoist any `("object", cls, held)` entries out of `key` into `opaques`, replacing each with a
    deterministic positional reference so the remaining key hashes cross-process without object addresses."""
    if not isinstance(key, tuple):
        return key
    if len(key) == 3 and key[0] == "object":
        slot = len(opaques)
        opaques.append(key)
        return "object_ref", key[1], slot
    return tuple(_collect_opaque(part, opaques) for part in key)


def _body_digest(jaxpr: jcore.Jaxpr, *, in_process: bool = True,
                 reject_device: bool = False) -> tuple[str, tuple[Hashable, ...]]:
    """Compute `(blake2b_hex, opaques)` for `jaxpr`. With `in_process=True`, executable callables,
    primitive objects, and extended dtypes stay by exact identity in `opaques`; with `reject_device=True`,
    any `jax.Array` literal/parameter or nested `consts` raises `_Unshareable` without touching device
    buffers."""
    hasher = hashlib.blake2b(digest_size=20)
    opaques: list[Hashable] = []
    ctx_keys: dict[int, Hashable] = {}
    position: dict[jcore.Var, int] = {}

    def feed(part: Hashable) -> None:
        hasher.update(repr(part).encode())
        hasher.update(b"\0")

    def aval_key(given) -> Hashable:
        if in_process and isinstance(given, ShapedArray):
            return _collect_opaque((_abstract_key(given), _extended_dtype(given.dtype)), opaques)
        return _collect_opaque(parameter_key(given), opaques)

    def feed_atom(atom) -> None:
        if isinstance(atom, jcore.Literal):
            if isinstance(atom.val, jax.Array) and not isinstance(atom.val, jax_core.Tracer):
                if reject_device:
                    raise _Unshareable
                feed(("lit_device", _collect_opaque(("object", "jax.Array", _Identity(atom.val)), opaques),
                      aval_key(atom.aval)))
                return
            dtype, value = held_bits(atom.val)
            raw = np.ascontiguousarray(value).tobytes()
            ext = _collect_opaque(_extended_dtype(dtype), opaques) if in_process else None
            feed(("lit", str(dtype), ext, value.shape, len(raw), aval_key(atom.aval)))
            hasher.update(raw)
            hasher.update(b"\0")
        else:
            feed(("var", position[atom]))

    def var_key(var: jcore.Var) -> Hashable:
        return isinstance(var, jcore.DropVar), aval_key(var.aval)

    feed(("constvars", tuple(var_key(var) for var in jaxpr.constvars)))
    feed(("invars", tuple(var_key(var) for var in jaxpr.invars)))
    for var in (*jaxpr.constvars, *jaxpr.invars):
        position[var] = len(position)
    feed(("eqns", len(jaxpr.eqns)))
    for equation in jaxpr.eqns:
        ctx = equation.ctx
        if (ctx_key := ctx_keys.get(id(ctx))) is None:
            if in_process:
                # JAX interns `JaxprEqnContext` by all fields (`no_collectives` and `cur_abstract_mesh`
                # included), so keying the interned object preserves every equation-local policy in O(1).
                ctx_key = _collect_opaque(("object", "eqn_ctx", ctx), opaques)
            else:
                ctx_key = _collect_opaque(parameter_key((
                    ctx.compute_type, ctx.threefry_partitionable, ctx.xla_metadata,
                    str(ctx.cur_layout_mode), ctx.remove_size_one_mesh_axis,
                    ctx.no_collectives, ctx.cur_abstract_mesh)), opaques)
            ctx_keys[id(ctx)] = ctx_key
        if equation.params:
            raw_params = (
                tuple((name, _in_process_param_key(part, reject_device=reject_device))
                      for name, part in sorted(equation.params.items()))
                if in_process else parameters_key(equation.primitive, equation.params)[1]
            )
            params_key = _collect_opaque(raw_params, opaques)
        else:
            params_key = ()
        effects_key = _collect_opaque(parameter_key(equation.effects), opaques) if equation.effects else ()
        prim_key = (_collect_opaque(("object", "primitive", equation.primitive), opaques) if in_process
                    else (_named(type(equation.primitive)), equation.primitive.name))
        feed((prim_key, params_key, ctx_key, effects_key, len(equation.invars),
              tuple(var_key(var) for var in equation.outvars)))
        for atom in equation.invars:
            feed_atom(atom)
        for var in equation.outvars:
            position[var] = len(position)
    feed(("outvars", len(jaxpr.outvars)))
    for atom in jaxpr.outvars:
        feed_atom(atom)
    feed(("effects", _collect_opaque(parameter_key(jaxpr.effects), opaques) if jaxpr.effects else ()))
    return hasher.hexdigest(), tuple(opaques)


def jaxpr_digest(jaxpr: jcore.Jaxpr, *, in_process: bool = False) -> tuple[str, tuple[Hashable, ...]]:
    """Structural `(hex_digest, opaques)` of `jaxpr` (constvars/invars avals, equations, parameters,
    contexts, literals, outvars, and effects; excluding `jaxpr.consts` values). Plain data when `in_process`
    is False and `opaques` is empty; with `in_process=True` returns a DAG-memoized `_SubKey` preserving exact
    primitive, callable, equation-context, and extended-dtype identities without touching device buffers."""
    if in_process:
        sub = _in_process_digest(jaxpr)
        return sub.digest, (sub,)
    inner = jaxpr.jaxpr
    if (cached := _JAXPR_DIGESTS.get(inner)) is not None:
        return cached
    digest = _body_digest(inner, in_process=False, reject_device=False)
    _JAXPR_DIGESTS[inner] = digest
    return digest


def body_key(closed: jcore.Jaxpr) -> Hashable | None:
    """Exact DAG-memoized in-process structural key for whole-body rewrite reuse
    (`program._STRUCTURAL_CLOSED`), or None when `closed` (or any nested body) holds constants or concrete
    `jax.Array` buffers: preserves primitive, callback-closure, equation-context, and extended PRNG dtype
    identities while never copying device buffers to host."""
    return _body_key_of(closed)


def _walk_sub_provenance(value, out: list[str], seen: set[int]) -> None:
    """Collect recursive provenance digests for any nested `jcore.Jaxpr` inside equation parameters."""
    if isinstance(value, jcore.Jaxpr):
        out.append(provenance_key(value))
        return
    if isinstance(value, (tuple, list)):
        for part in value:
            _walk_sub_provenance(part, out, seen)
        return
    if isinstance(value, Mapping):
        for _, part in sorted(value.items(), key=lambda entry: repr(entry[0])):
            _walk_sub_provenance(part, out, seen)
        return
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        ident = id(value)
        if ident in seen:
            return
        seen.add(ident)
        try:
            for field in dataclasses.fields(value):
                _walk_sub_provenance(getattr(value, field.name), out, seen)
        finally:
            seen.discard(ident)


def provenance_key(closed: jcore.Jaxpr) -> str:
    """DAG-memoized digest of `closed`'s root/nested `debug_info` and every equation's `source_info`
    (`name_stack` and raw traceback frames), used by `program._STRUCTURAL_CLOSED` so a modified body is never
    substituted across differing equation or nested-body provenance (row 13020)."""
    inner = closed.jaxpr
    if (cached := _PROVENANCE_KEYS.get(inner)) is not None:
        return cached
    hasher = hashlib.blake2b(digest_size=20)

    def feed(part) -> None:
        hasher.update(repr(part).encode())
        hasher.update(b"\0")

    di = inner.debug_info
    if di is None:
        feed(("debug_info", None))
    else:
        result_paths = (di.result_paths if isinstance(di.result_paths, (type(None), tuple))
                        else repr(di.result_paths))
        feed(("debug_info", di.traced_for, di.func_src_info, di.arg_names, result_paths))
    for equation in inner.eqns:
        si = equation.source_info
        tb = si.traceback
        if tb is None:
            frames = None
        else:
            codes, lastis = tb.raw_frames()
            frames = tuple((code.co_filename, code.co_qualname, code.co_firstlineno, lasti)
                           for code, lasti in zip(codes, lastis, strict=True))
        feed(("eqn", si.name_stack.stack, frames))
        if equation.params:
            nested: list[str] = []
            _walk_sub_provenance(equation.params, nested, set())
            if nested:
                feed(("nested", tuple(nested)))
    digest = hasher.hexdigest()
    _PROVENANCE_KEYS[inner] = digest
    return digest


class _Identity:
    """An unhashable parameter as a key: equal to a key of the same object alone, the object held."""

    __slots__ = ("held",)

    def __init__(self, held):
        self.held = held

    def __hash__(self) -> int:
        return id(self.held)

    def __eq__(self, other) -> bool:
        return isinstance(other, _Identity) and other.held is self.held


def opaque(key) -> Hashable:
    """`key` (`parameter_key`'s) with each object no structural key reads as its type's name alone: what a
    key kept beyond this process reads (`dew.nn.kernels.tuning.program`'s digest), the same in every
    process, where the object itself would print its process's address."""
    if isinstance(key, tuple) and len(key) == 3 and key[0] == "object":
        return key[:2]
    if isinstance(key, tuple):
        return tuple(opaque(part) for part in key)
    return key


def _named(kind: type) -> str:
    """A type's import path, which names it in every process."""
    return f"{kind.__module__}.{kind.__qualname__}"


def declarations(language: object) -> Declarations:
    """A language's declarations (egglog's `HasDeclarations`), which reading at import frees its defining
    frame (`linear_scan._DECLARATIONS`)."""
    if not isinstance(language, HasDeclarations):
        raise TypeError(f"{language!r} declares no egglog language")
    return language.__egg_decls__


type Ranges = Mapping[int, float | tuple[float, float] | Proven] | None
"""Declared ranges by input index (`region`'s): a bound on an input's magnitude, an interval, or what a
program around the input proves of it (`Proven`)."""


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
    one (a protected span's marked call among them) a leaf for each output, after its inputs' terms, each of
    `forms` (egglog's `Term`, or a window's structure alone, `Skeleton`). Each term's
    magnitude (`Magnitude`) from its constants', its operands' declared `ranges` (by input index, the
    constants apart: a bound on its magnitude, or an interval `(low, high)`, whose `high` at most 0 bounds its
    exponential by 2, one power past the reals' 1 for a native exponential; an input with none is
    unbounded) and its equations'."""
    if forms is Skeleton:
        return _region(closed, ranges, Skeleton)
    return _region(closed, ranges, Term)


def _merge_proven(out_var: jcore.Var, bound: Magnitude | None, proved: Apart,
                  ext: float | tuple[float, float] | Proven | None) -> tuple[Magnitude | None, Apart]:
    """Join an equation's locally proven `(bound, apart)` with an enclosing program's proven entry `ext`."""
    if ext is None or not isinstance(out_var.aval, ShapedArray):
        return bound, proved
    ext_bound = _declared(ext)
    ext_apart = _held_apart(_declared_apart(ext), out_var.aval.dtype)
    floors = [f for f in (proved.floor, ext_apart.floor) if f is not None]
    floor = max(floors) if floors else None
    sign = proved.sign if proved.sign != 0 else ext_apart.sign
    merged = _held_apart(Apart(
        floor, sign, proved.number or ext_apart.number,
        proved.finite or ext_apart.finite, proved.unit_norm or ext_apart.unit_norm,
        proved.dyadic_scale or ext_apart.dyadic_scale,
    ), out_var.aval.dtype)
    bounds = [b for b in (bound if proved.finite else None,
                          ext_bound if ext_apart.finite else None) if b is not None]
    best = min(bounds) if bounds else None
    return (best if merged.finite else None), merged


def _region[Node: (Term, Skeleton)](closed: jcore.Jaxpr, ranges: Ranges,
                                    forms: type[Node]) -> Region[Node]:
    """`region`'s terms, each of `forms`."""
    jaxpr = closed.jaxpr
    params: list[tuple[jcore.Primitive, Mapping]] = []
    indexed: dict[Hashable, int] = {}
    literals: list = []
    shapes: list[tuple[Node, tuple[int, ...]]] = []
    depths: dict[jcore.Var, int] = {}

    products: dict[int, Product] = {}

    param = functools.partial(_interned, params, products, indexed)

    env: dict[jcore.Var, Node] = {}
    dtypes: dict[Hashable, np.dtype] = {}
    magnitudes: dict[Hashable, Magnitude | None] = {}
    leaves = 0
    declared = dict(ranges or {})
    given = [*(_magnitude(value) for value in closed.consts),
             *(_declared(declared.get(index)) for index in range(len(jaxpr.invars)))]
    # What each value holds beside its magnitude (`Apart`): a floor away from zero and its sign, a
    # constant's and a declared interval's own, each equation's by its operands' (`_apart`). A quotient's
    # bound, a root's and a logarithm's read the floor; an exponential of a value at most 0 its sign.
    apart: dict[jcore.Var, Apart] = {
        var: _held_apart(held, var.aval.dtype) for var, held in zip(
            [*jaxpr.constvars, *jaxpr.invars],
            [*map(_constant_apart, closed.consts),
             *(_declared_apart(declared.get(index)) for index in range(len(jaxpr.invars)))], strict=True)
        if isinstance(var.aval, ShapedArray)}
    for var, magnitude in zip([*jaxpr.constvars, *jaxpr.invars], given, strict=True):
        env[var] = forms.leaf(leaves)
        shapes.append((env[var], _shape(var)))
        _typed(dtypes, magnitudes, env[var], var, magnitude)
        leaves += 1

    literal = _literals(forms, literals, shapes, dtypes, magnitudes)

    def read(atom) -> Node:
        return literal(atom) if isinstance(atom, jcore.Literal) else env[atom]

    def magnitude_of(atom) -> Magnitude | None:
        return _magnitude(atom.val) if isinstance(atom, jcore.Literal) else magnitudes[key(env[atom])]

    def apart_of(atom) -> Apart:
        return _apart_of(atom, apart)

    # Each value a broadcast made, by its source and the axes it laid the source's along, so a broadcast of
    # one is read as one broadcast of the source (exact: a layout alone), whose axes a hoist reads.
    laid: dict[jcore.Var, tuple[jcore.Var, tuple[int, ...]]] = {}
    full_fp32_sites: dict[Hashable, bool] = {}
    softmax = _SoftmaxTrack()
    opaque: list[Opaque] = []
    n_invars = len(jaxpr.invars)

    for eq_idx, equation in enumerate(jaxpr.eqns):
        operation = _in_language(equation)
        composed, facts = _propagate_facts(equation, operation, laid, magnitude_of, apart_of)
        if composed is not None and facts is not None:
            origin, params_ = composed
            out = equation.outvars[0]
            env[out] = read(origin).shaped("broadcast_in_dim", param(equation.primitive, params_))
            shapes.append((env[out], _shape(out)))
            bound, proved = softmax.step(equation, *facts, apart_of)
            bound, apart[out] = _merge_proven(out, bound, proved, declared.get(n_invars + eq_idx))
            _typed(dtypes, magnitudes, env[out], out, bound)
            depths[out] = depths.get(origin, 0) + 1
            continue
        if operation is not None and facts is not None:
            term = _term(equation, operation, read, param)
            out = equation.outvars[0]
            env[out] = term
            shapes.append((term, _shape(out)))
            bound, proved = softmax.step(equation, *facts, apart_of)
            bound, apart[out] = _merge_proven(out, bound, proved, declared.get(n_invars + eq_idx))
            _typed(dtypes, magnitudes, term, out, bound)
            if (equation.primitive is jax.lax.dot_general_p
                    and equation.params.get("precision") == jax.lax.DotAlgorithmPreset.F32_F32_F32):
                named = key(term)
                # If equal terms come from more than one mesh scope, every source must admit the move.
                full_fp32_sites[named] = full_fp32_sites.get(named, True) and _unplaced_dot(equation)
            depths[out] = 1 + max((depths.get(atom, 0) for atom in equation.invars
                                   if not isinstance(atom, jcore.Literal)), default=0)
            continue
        is_stop = equation.primitive is jax.lax.stop_gradient_p and len(equation.outvars) == 1
        outputs = []
        for var in equation.outvars:
            env[var] = forms.leaf(leaves)
            shapes.append((env[var], _shape(var)))
            if is_stop:
                src = equation.invars[0]
                apart[var] = apart_of(src)
                if isinstance(src, jcore.Var):
                    softmax.passthrough(src, var)
                _typed(dtypes, magnitudes, env[var], var, magnitude_of(src))
            else:
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
                  frozenset(key(forms.literal(index)) for index in range(len(literals))), indexed,
                  {var: magnitudes.get(key(node)) for var, node in env.items()},
                  {var: apart.get(var, NOWHERE) for var in env},
                  {key(node): held for var, node in env.items()
                   if (held := apart.get(var, NOWHERE)).number and held.floor is not None},
                  frozenset({
                      *(key(node) for var, node in env.items()
                        if (held := apart.get(var, NOWHERE)).number and held.sign > 0),
                      *(key(forms.literal(index)) for index, value in enumerate(literals)
                        if (held := _constant_apart(value)).number and held.sign > 0),
                  }), frozenset(named for named, allowed in full_fp32_sites.items() if allowed))


def _unplaced_dot(equation: jcore.JaxprEqn) -> bool:
    """A dot seen later outside the tracing context may still hold explicit or manual placement."""
    mesh = equation.ctx.cur_abstract_mesh
    if mesh is not None and (mesh.explicit_axes or mesh.manual_axes):
        return False
    if equation.params.get("out_sharding") is not None:
        return False
    for atom in (*equation.invars, *equation.outvars):
        if isinstance(aval := atom.aval, ShapedArray):
            mesh = aval.sharding.mesh
            if mesh.explicit_axes or mesh.manual_axes:
                return False
    return True


def _prove_equation(equation: jcore.JaxprEqn, name: str, magnitude_of: Callable[..., Magnitude | None],
                    apart_of: Callable[..., Apart]) -> tuple[Magnitude | None, Apart]:
    """The finite magnitude bound and `Apart` proven of `equation`'s single output from its operands'."""
    operands = [magnitude_of(atom) for atom in equation.invars]
    held_apart = [apart_of(atom) for atom in equation.invars]
    # An exponential of a value at most 0 is at most 1 in the reals, at most 2 as computed: a native
    # exponential is not certified under 1 (`ex2.approx`'s relative error, 2**-22.5, is more than an ulp
    # of 1), so it is given one power.
    exponentiated = name in ("exp", "exp2") and held_apart[0].sign < 0
    bound = UNIT + 1 if exponentiated else _bounded(equation, name, operands, held_apart)
    proved = _apart(equation, name, held_apart, operands, bound)
    # A bound is stated of a value proven finite alone, as a rewrite's side condition reads it (row 5780;
    # owner, 2026-11-14): a NaN's tanh is no number, and a product of a value that overflowed reads the
    # bound its real factors have and computes an infinity.
    return (bound if proved.finite else None), proved


def _in_language(equation: jcore.JaxprEqn) -> Operation | None:
    """Read a declared operation only where the equation has its arity, result and supported values."""
    operation = OPERATIONS.get(equation.primitive)
    if (operation is None or len(equation.outvars) != 1
            or len(equation.invars) != WRITTEN[operation.kind].arity):
        return None
    if equation.primitive is ad_util.add_jaxvals_p:
        left, right = equation.invars[0].aval, equation.invars[1].aval
        made = equation.outvars[0].aval
        if not (isinstance(left, ShapedArray) and isinstance(right, ShapedArray)
                and isinstance(made, ShapedArray)):
            return None
        if (made.dtype not in (jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64)
                or left.dtype != made.dtype or right.dtype != made.dtype
                or not jax_core.definitely_equal_shape(left.shape, made.shape)
                or not jax_core.definitely_equal_shape(right.shape, made.shape)):
            return None
        # Rewrites can move an add between ranks, which can change explicit layouts or manual-axis
        # variation. Unlike an ordinary add, accumulation demands identical placed input types.
        mesh = made.sharding.mesh
        if mesh.explicit_axes or mesh.manual_axes:
            return None
    return operation


def _composed_broadcast(
    equation: jcore.JaxprEqn,
    laid: dict[jcore.Var, tuple[jcore.Var, tuple[int, ...]]],
) -> tuple[jcore.Var, dict] | None:
    """Where `equation` broadcasts a variable already broadcast (`laid`), compose both into one broadcast
    from the original source and record it in `laid`; for a first broadcast of a variable, record it in `laid`
    and return None."""
    if equation.primitive is not jax.lax.broadcast_in_dim_p or isinstance(equation.invars[0], jcore.Literal):
        return None
    source = equation.invars[0]
    outer = tuple(equation.params["broadcast_dimensions"])
    out = equation.outvars[0]
    if source in laid:
        origin, inner = laid[source]
        composed = {**equation.params, "broadcast_dimensions": tuple(outer[axis] for axis in inner)}
        laid[out] = (origin, composed["broadcast_dimensions"])
        return origin, composed
    laid[out] = (source, outer)
    return None


def _propagate_facts(
    equation: jcore.JaxprEqn,
    operation: Operation | None,
    laid: dict[jcore.Var, tuple[jcore.Var, tuple[int, ...]]],
    magnitude_of: Callable[..., Magnitude | None],
    apart_of: Callable[..., Apart],
) -> tuple[tuple[jcore.Var, dict] | None, tuple[Magnitude | None, Apart] | None]:
    """The single shared fact-transfer step for `_region` and `proven`: composed broadcasts keep their
    origin's magnitude and `Apart`, other in-language equations propagate via `_prove_equation`, and opaque
    equations return `(None, None)`."""
    composed = _composed_broadcast(equation, laid)
    if composed is not None:
        origin, _ = composed
        return composed, (magnitude_of(origin), apart_of(origin))
    if operation is not None:
        return None, _prove_equation(equation, operation.symbol, magnitude_of, apart_of)
    return None, None


def _proven_walk(jaxpr: jcore.Jaxpr, *, consts: Mapping[jcore.Var, object] | None,
                 targets: Sequence[jcore.Var] | None, deep: bool,
                 init: Mapping[jcore.Var, tuple[Magnitude | None, Apart]] | None = None,
                 track: _NormTrack | None = None,
                 softmax: _SoftmaxTrack | None = None,
                 depth: int = 0) -> tuple[dict[jcore.Var, Magnitude | None], dict[jcore.Var, Apart]]:
    """Core walker for `proven`, shared between plain in-language proofs (`deep=False`) and deep slice
    proofs (`deep=True`, peeling calls, `concatenate`/`slice`/`stop_gradient`, NaN guards, and L2 norms)."""
    needed = programs.read_by(jaxpr, targets) if targets is not None else None
    held: dict[jcore.Var, Magnitude | None] = {}
    apart: dict[jcore.Var, Apart] = {}
    for var in (*jaxpr.constvars, *jaxpr.invars):
        if needed is not None and var not in needed:
            continue
        if init is not None and var in init:
            held[var], apart[var] = init[var]
        elif consts is not None and var in consts and _host_array(consts[var]) is not None:
            held[var] = _magnitude(consts[var])
            apart[var] = (_held_apart(_constant_apart(consts[var]), var.aval.dtype)
                          if isinstance(var.aval, ShapedArray) else NOWHERE)
        else:
            held[var] = None
            if isinstance(var.aval, ShapedArray):
                apart[var] = _held_apart(WHOLE, var.aval.dtype)

    def magnitude_of(atom) -> Magnitude | None:
        return _magnitude(atom.val) if isinstance(atom, jcore.Literal) else held.get(atom)

    def apart_of(atom) -> Apart:
        return _apart_of(atom, apart)

    laid: dict[jcore.Var, tuple[jcore.Var, tuple[int, ...]]] = {}
    made_eq: dict[jcore.Var, jcore.JaxprEqn] = {}
    norm_track = _NormTrack() if (deep and track is None) else track
    softmax_track = _SoftmaxTrack() if softmax is None else softmax
    for equation in jaxpr.eqns:
        if needed is not None and not any(var in needed for var in equation.outvars):
            continue
        if deep and len(equation.outvars) == 1:
            made_eq[equation.outvars[0]] = equation
        _, facts = _propagate_facts(equation, _in_language(equation), laid, magnitude_of, apart_of)
        if facts is not None:
            out = equation.outvars[0]
            bound, proved = softmax_track.step(equation, *facts, apart_of)
            if deep and norm_track is not None:
                bound, proved = _deep_refine(equation, bound, proved, made_eq, magnitude_of, apart_of,
                                             norm_track)
            held[out], apart[out] = bound, proved
            continue
        if equation.primitive is jax.lax.stop_gradient_p and len(equation.outvars) == 1:
            out = equation.outvars[0]
            held[out], apart[out] = magnitude_of(equation.invars[0]), apart_of(equation.invars[0])
            if isinstance(equation.invars[0], jcore.Var):
                softmax_track.passthrough(equation.invars[0], out)
            continue
        if deep and norm_track is not None and _deep_nonlang(
                equation, held, apart, magnitude_of, apart_of, norm_track, softmax_track, depth,
                _proven_walk):
            continue
        for var in equation.outvars:
            held[var] = None
    return held, apart


def proven(jaxpr: jcore.Jaxpr, *, consts: Mapping[jcore.Var, object] | None = None,
           targets: Sequence[jcore.Var] | None = None,
           deep: bool = False) -> tuple[dict[jcore.Var, Magnitude | None], dict[jcore.Var, Apart]]:
    """What `jaxpr` proves of each variable (`held` magnitudes and `apart`), without building terms or
    interning parameters (`dew.algebra.program._held`), sharing `_propagate_facts` with `_region` so both
    produce identical facts when `deep=False`. With `deep=True` and optional `consts`/`targets`, walks only
    the dependency slice of `targets` without touching tracers or device arrays, peeling called bodies and
    tracking `concatenate`, slicing, NaN guards, and unit-norm denominators."""
    return _proven_walk(jaxpr, consts=consts, targets=targets, deep=deep)


BOUNDED = _symbols(lambda operation: operation.bounded)
KEEPING_MAGNITUDE = _symbols(lambda operation: operation.keeps_magnitude)
INTEGRAL = _symbols(lambda operation: operation.integral)
BITWISE = _symbols(lambda operation: operation.bitwise)
KEEPING_SIGN = _symbols(lambda operation: operation.keeps_sign)
"""The symbols whose operations' values prove each fact of their operand's (`Operation`)."""


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


def _apart(equation: jcore.JaxprEqn, name: str, operands: list[Apart], bounds: list[Magnitude | None],
           bound: Magnitude | None) -> Apart:
    """What `equation`'s output holds beside its magnitude (`Apart`) from its operands' (`operands`), their
    magnitudes (`bounds`, each `|x| <= 2**a`) and its own (`bound`), by the real arithmetic over their
    boxes; a float rounds to nearest, monotone, so a result whose exact value is at least a normal power of
    two rounds to no less (`_held_apart` keeps such floors alone). An equation with no rule here, and one of
    an operand nothing is known of, states nothing: its output may be anything.

    A number, and finite. An output is a number where its operands are and its operation makes no NaN of
    them: a sum or a difference unless infinities of unknown or opposite sides meet; a product unless a
    value that may be zero meets one that may be infinite; a quotient unless both may be zero or both
    infinite; a root or a logarithm of a value at least 0; a maximum, a minimum, a selection, a negation, an
    absolute value, a power of an integer, an exponential, a logistic, a layout, a conversion and a maximum
    or minimum along an axis of numbers; a sum along axes of numbers on one side of zero. A tanh, an erf and
    a logistic of a number are finite (an infinity's is its side's limit). It is finite where it is a number
    made of finite operands (a quotient's divisor, and a negative power's operand, held away
    from zero) and its own bound is under its dtype's largest power; a contraction, and a sum along axes of
    either side's values, is a number only so: its partial sums, in whatever order its mode takes them,
    overflow under no bound within the dtype, and past one two of them may be infinities of opposite sides.

    A side of zero and a floor, a number's alone:
    - a sum of values on one side of zero stays there, at least the largest of their floors away, and so
      does a difference of values on either side;
    - a product's sign is its operands' multiplied and its floor their floors' sum; a square's, and an
      integer power's of `y >= 1`, likewise; a power's of `y <= -1` of a finite operand is at least
      `2**(y a)` where `2**(-y a)` is a float (the reciprocal of a power that overflows is zero), and keeps
      an odd power's sign only of an operand held away from zero (a zero's reciprocal takes the zero's own
      sign);
    - a quotient `x / y` of `|x| >= 2**l` and a finite `|y| <= 2**b` is at least `2**(l - b)`, and has a
      sign where its divisor is held away from zero;
    - a maximum is at least an operand that is at least 0, a minimum at most one that is at most 0, and
      each is one of its operands, as a selection is: at least the least of their floors;
    - `exp`, `exp2` of a finite `|x| <= 2**a <= 64`: at least `e**-(2**a)`, a power under the upper bound's
      own reciprocal for a native exponential's rounding (`_real_bound`);
    - `sqrt` of `x >= 2**l` is at least `2**(l // 2)` (correctly rounded); `rsqrt` of a finite
      `0 <= x <= 2**a` at least `2**-ceil(a / 2)`, less a power for a native reciprocal root's rounding, and
      above zero only of an operand above it (a negative zero's is negative infinity);
    - an absolute value, a negation (which turns the sign), a layout change, a conversion between floats, a
      maximum or minimum along an axis of at least one element keep a floor; a sum along axes of at least
      one element of values on one side of zero keeps theirs;
    - a sign of a number is a finite number on its operand's side, 1 away from zero where its operand is
      held away.
    A float is at least 0 where its operation makes it so of a number (a square, an even power, an absolute
    value, an exponential, a root, a logistic). A contraction's sign is not claimed: a mode that splits its
    operands into parts sums parts of either sign."""
    aval = equation.outvars[0].aval
    if not isinstance(aval, ShapedArray):
        return NOWHERE
    if not jnp.issubdtype(aval.dtype, jnp.floating):
        # An integer keeps a sign through the operations that keep one alone: its arithmetic wraps.
        kept = name in KEEPING_SIGN and (
            name != "convert_element_type" or _keeps_sign(equation.invars[0], equation.outvars[0])) and not (
            name.startswith("reduce_") and _reduced(equation) < 1)
        # An integer's bound holds of it where its operands' hold of them: a float that may be infinite
        # converts to a value its bound says nothing of (XLA saturates it), which stays so as a float again.
        proven = all(held.finite for held in operands)
        return _held_apart((operands[0] if kept else WHOLE)._replace(finite=proven), aval.dtype)
    if not operands:
        return NOWHERE
    made = _float_apart(equation, name, operands, bounds, aval.dtype)
    # Finite where its operands make it so and its own bound is within its dtype; a finite value is a number.
    finite = made.finite and _within(bound, aval.dtype)
    dyadic = ((name == "broadcast_in_dim" and operands[0].dyadic_scale)
              or (name == "select_n" and all(held.dyadic_scale for held in operands[1:])))
    made = made._replace(number=made.number or finite, finite=finite, dyadic_scale=dyadic)
    return _held_apart(made, aval.dtype)


def _extremes(name: str, operands: list[Apart], bounds: list[Magnitude | None]) -> Apart:
    """`_apart` of a maximum, a minimum or a clamp, XLA's: the minimum of its upper bound and the maximum of
    its operand and its lower bound (`_extreme`), finite between finite bounds whatever its operand."""
    if name == "clamp":
        lower, operand, upper = operands
        lb, ob, ub = bounds
        clamped = _extreme(-1, [_extreme(1, [operand, lower], [ob, lb]), upper], [None, ub])
        # Finite where it is a number between finite bounds, whatever its operand: an infinity clamps to one.
        return clamped._replace(finite=clamped.number and lower.finite and upper.finite)
    return _extreme(1 if name == "max" else -1, operands, bounds)


def _extreme(side: int, operands: list[Apart], bounds: list[Magnitude | None]) -> Apart:
    """`_apart` of a maximum (`side` 1) or a minimum (-1) of `operands`: on the side of an operand on that
    side (including exact zero `NOTHING`), else of theirs where they share one; at least the largest floor of
    an operand on that side, else the least of all their floors (it is one of them); a number, and finite,
    where they all are."""
    floors = [held.floor for held in operands]
    known = [floor for floor in floors if floor is not None]
    if any(_side_matches(h, b, side) for h, b in zip(operands, bounds, strict=True)):
        sign = side
    elif all(_side_matches(h, b, 1) for h, b in zip(operands, bounds, strict=True)):
        sign = 1
    elif all(_side_matches(h, b, -1) for h, b in zip(operands, bounds, strict=True)):
        sign = -1
    else:
        sign = 0
    beyond = [held.floor for held in operands if held.sign == side and held.floor is not None]
    least = min(known) if len(known) == len(floors) else None
    return Apart(max(beyond) if beyond else least, sign, all(held.number for held in operands),
                 all(held.finite for held in operands))


def _float_apart(equation: jcore.JaxprEqn, name: str, operands: list[Apart],
                 bounds: list[Magnitude | None], dtype: DTypeLike) -> Apart:
    """`_apart` of a float output of `dtype` before its dtype holds it (`_held_apart`), `finite` before its
    own bound is read: whether its operands make it finite."""
    first, floors, signs = operands[0], [held.floor for held in operands], [held.sign for held in operands]
    numbers, finite = all(held.number for held in operands), all(held.finite for held in operands)
    known = [floor for floor in floors if floor is not None]
    every = known if len(known) == len(floors) else None
    if name in ("add", "sub"):
        shifted = operands if name == "add" else [operands[0], operands[1]._replace(sign=-operands[1].sign)]
        sign = _shared_sign(shifted, bounds)
        floor = max((found for found in floors if found is not None), default=None) if sign else None
        return Apart(floor, sign, numbers and (finite or sign != 0), finite)
    if name == "mul":
        return Apart(None if every is None else sum(every), math.prod(signs),
                     numbers and (finite or every is not None), finite)
    if name == "div":
        dividend, divisor = operands
        apart = divisor.floor is not None
        floor = (floors[0] - bounds[1] if floors[0] is not None and divisor.finite and bounds[1] is not None
                 and bounds[1] != NOTHING else None)
        return Apart(floor, signs[0] * signs[1] if apart else 0,
                     numbers and (apart or floors[0] is not None) and (dividend.finite or divisor.finite),
                     dividend.finite and divisor.number and apart)
    if name == "pow":
        # A finite base held above zero to a finite exponent is a positive number, finite where its own
        # bound is within its dtype (`_apart`), the case `_apart_bound` bounds.
        base, exponent = operands
        held = base.positive and base.finite and exponent.finite
        return Apart(None, 1 if base.positive else 0, held, held)
    if name in ("max", "min", "clamp"):
        return _extremes(name, operands, bounds)
    if name == "select_n":
        cases, case_bounds = operands[1:], bounds[1:]
        kept = [held.floor for held in cases if held.floor is not None]
        return Apart(min(kept) if len(kept) == len(cases) else None, _shared_sign(cases, case_bounds),
                     all(held.number for held in cases), all(held.finite for held in cases))
    if name == "neg":
        return first._replace(sign=-first.sign)
    if name == "sign":
        return Apart(0 if first.floor is not None else None, first.sign, first.number, first.number)
    if name == "abs":
        return first._replace(sign=1)
    if name in ("square", "integer_pow"):
        return _power_apart(equation, first, bounds[0], dtype)
    if name in ("exp", "exp2", "sqrt", "rsqrt", "logistic", "log", "log2", "log1p"):
        return _transcendental_apart(name, first, bounds[0])
    if name == "dot_general":
        return Apart(None, 0, number=False, finite=finite)
    if name in BOUNDED:
        # Finite of a finite operand, each of them; a sine of an infinity is no number, but a tanh's and an
        # erf's are their side's 1 (XLA clamps the operand before its polynomial), so of a number they are
        # finite: a product that may overflow, which a window takes as its program proves it, bounds its
        # tanh still (spec alignment, 8993).
        held = first.number if name in ("tanh", "erf") else first.finite
        return Apart(None, 0, held, held)
    if name in INTEGRAL:
        return Apart(None, 0, first.number, first.finite)
    if name in KEEPING_SIGN:
        source = equation.invars[0]
        if name.startswith("reduce_") and _reduced(equation) < 1:
            return NOWHERE
        if name == "convert_element_type" and not _keeps_sign(source, equation.outvars[0]):
            return first._replace(floor=None, sign=0, unit_norm=False)
        floating = isinstance(source.aval, ShapedArray) and jnp.issubdtype(source.aval.dtype, jnp.floating)
        return first._replace(unit_norm=False) if floating else first._replace(floor=None, unit_norm=False)
    if name == "reduce_sum":
        if _reduced(equation) < 1:
            return WHOLE
        return Apart(first.floor if first.sign else None, first.sign, first.number and first.sign != 0,
                     first.finite)
    return NOWHERE


def _transcendental_apart(name: str, first: Apart, bound: Magnitude | None) -> Apart:
    """`Apart` for elementary unary transcendentals and roots (`_float_apart`)."""
    if name in ("exp", "exp2"):
        floor = None
        if first.finite and bound is not None and bound <= 6:
            floor = -(math.floor(2 ** bound * (math.log2(math.e) if name == "exp" else 1.0)) + 2)
        return Apart(floor, 1, first.number, first.finite or (first.number and first.sign < 0))
    if name == "sqrt":
        real, above = first.number and first.sign > 0, first.above
        return Apart(None if above is None else above // 2, 1, real, real and first.finite)
    if name == "rsqrt":
        real = first.number and first.sign > 0
        floor = (-bound) // 2 - 1 if real and first.finite and bound not in (None, NOTHING) else None
        return Apart(floor, 1 if first.positive else 0, real, first.positive)
    if name == "logistic":
        return Apart(None, 1, first.number, first.number)
    if name in ("log", "log2"):
        return Apart(None, 0, first.number and first.sign > 0, first.positive and first.finite)
    nonneg = _side_matches(first, bound, 1)
    return Apart(None, 1 if nonneg else 0, first.number and nonneg, first.finite and nonneg)


def _power_apart(equation: jcore.JaxprEqn, operand: Apart, bound: Magnitude | None,
                 dtype: DTypeLike) -> Apart:
    """A square's or an integer power's `Apart`, of `dtype`, before its dtype holds it (`_float_apart`)."""
    power = 2 if equation.primitive is jax.lax.square_p else equation.params["y"]
    if power == 0:
        # JAX lowers a zeroth power to ones.
        return Apart(0, 1, number=True, finite=True)
    # An even power is at least 0 (a zero's reciprocal squared is positive infinity); an odd one keeps its
    # operand's sign, a negative odd one only of an operand that is no zero.
    sign = 1 if power % 2 == 0 else operand.sign if power > 0 or operand.floor is not None else 0
    if power >= 1:
        return Apart(None if operand.floor is None else power * operand.floor, sign, operand.number,
                     operand.finite)
    # A negative power is the reciprocal of the positive one, which must be a float for its reciprocal to
    # be above zero; of a number it is a number (an infinity of a zero), finite of one held away from zero.
    floor, reach = None, int(jnp.finfo(dtype).maxexp)
    if operand.finite and bound is not None and bound != NOTHING and -power * bound < reach:
        floor = power * bound
    return Apart(floor, sign, operand.number, operand.number and operand.floor is not None)


def _apart_bound(equation: jcore.JaxprEqn, name: str, operands: list[Magnitude | None],
                 apart: list[Apart], dtype: DTypeLike) -> Magnitude | None:
    """`_real_bound`'s bound of an equation of a float `dtype` whose result shrinks with an operand or needs
    one away from zero, each operand's magnitude read only where it is proven finite (`Apart.finite`: an
    infinite dividend's quotient is infinite whatever bounds it, and a value that overflowed in a narrower
    dtype reads small in a wider one) and each held where the equation needs it, None otherwise; a divisor, a
    reciprocal root's operand and a negative power's base are read by their floor alone, which an infinite one
    holds too, its result a zero:
    - a quotient `x / y` of a finite `|x| <= 2**a` and `|y| >= 2**l`: at most `2**(a - l)`, a power the
      rounded quotient is at most (a division by a reciprocal's product rounds under it as well);
    - `sqrt` of a finite `0 <= x <= 2**a`: at most `2**ceil(a / 2)`; `rsqrt` of `x >= 2**l`: at most
      `2**-(l // 2)`, and a power more for a native reciprocal root's rounding;
    - `log`, `log2` of a finite `2**l <= x <= 2**a`: at most `max(|l|, |a|)` in magnitude (a natural
      logarithm's is smaller), and a power more for the native function's rounding;
    - a power `x**y` of a finite `2**l <= x <= 2**a` and a finite `|y| <= 2**b <= 64`: `2**(y log2 x)`, at
      most `2**(2**b max(|l|, |a|))`, and a power more;
    - an integer power of `y <= -1` of `|x| >= 2**l`: at most `2**(y l)`, where `2**(-y l)` is a float (the
      reciprocal of a power that underflows to zero is infinite)."""
    if not jnp.issubdtype(dtype, jnp.floating):
        return None
    upper, held = operands[0], apart[0]
    above = held.above
    if name == "div":
        floor = apart[1].floor
        if upper is None or not held.finite or floor is None:
            return None
        return upper if upper == NOTHING else upper - floor
    if name == "sqrt":
        if upper is None or not held.finite or held.sign <= 0:
            return None
        return upper if upper == NOTHING else -((-upper) // 2)
    if name == "rsqrt":
        return None if above is None else -(above // 2) + 1
    if name in ("log", "log2"):
        if above is None or not held.finite or upper is None or upper == NOTHING:
            return None
        return _log2(max(abs(above), abs(upper), 1)) + 1
    if name == "pow":
        exponent = operands[1]
        if above is None or not (held.finite and apart[1].finite) or upper is None or upper == NOTHING:
            return None
        if exponent is None or exponent > 6:
            return None
        return math.ceil(2.0 ** max(exponent, -64) * max(abs(above), abs(upper))) + 1
    # An integer power of a negative exponent.
    power = equation.params["y"]
    if power == 0:
        return UNIT
    if held.floor is None or -power * held.floor < int(jnp.finfo(dtype).minexp):
        return None
    return power * held.floor


APART_BOUNDED = frozenset({"div", "sqrt", "rsqrt", "log", "log2", "pow"})
"""The operations whose bound `_apart_bound` states, a negative integer power's beside them."""


def _bounded(equation: jcore.JaxprEqn, name: str, operands: list[Magnitude | None],
             apart: list[Apart]) -> Magnitude | None:
    """A bound on the magnitude of `equation`'s output from its operands' (`operands`, `Magnitude`) and what
    they hold away from zero (`apart`), None where its primitive proves none (an exponential of a value of any
    magnitude, a quotient by a value nothing holds apart from zero), as its output's
    dtype holds it: a boolean at most 1 whatever made it (a tiny float converted is 1); an integer at most
    its dtype's range (`_integer_range`), within which a sum or a product wraps, and at it where an unsigned
    result may wrap from below zero (`uint32(0) - 1`, a negation, a conversion of int32's -1: `2**32 - 1`)
    and for a bitwise operation (int32's `not(-2**31)` is `2**31 - 1`); a float as the real arithmetic
    bounds it (`_real_bound`)."""
    aval = equation.outvars[0].aval
    # An extended dtype's value (a typed key) or symbolic `float0` tangent is no number.
    if (not isinstance(aval, ShapedArray) or _extended_dtype(aval.dtype) is not None
            or aval.dtype == jax.dtypes.float0):
        return None
    dtype = np.dtype(aval.dtype)
    if dtype == np.bool_:
        return UNIT
    real = _real_bound(equation, name, operands, apart, dtype)
    if not jnp.issubdtype(dtype, jnp.integer):
        return real
    full = _integer_range(dtype)
    source = equation.invars[0].aval
    from_unsigned = isinstance(source, ShapedArray) and jnp.issubdtype(source.dtype, jnp.unsignedinteger)
    wraps = jnp.issubdtype(dtype, jnp.unsignedinteger) and (
        name in ("neg", "sub") or (name == "convert_element_type" and not from_unsigned))
    if name in BITWISE or wraps:
        return full
    return None if real is None else min(real, full)


def _real_bound(equation: jcore.JaxprEqn, name: str, operands: list[Magnitude | None],
                apart: list[Apart], dtype: DTypeLike) -> Magnitude | None:
    """`_bounded`'s bound of an output of `dtype` as the real arithmetic on the operands' bounds proves it (a
    float's, or an integer's before its dtype wraps it), each operand `|x| <= 2**a` (`y`'s `2**b`). Each
    rule's reason over the box of its operands, and its float rounding's slack (a float rounds to nearest,
    monotone, so a result whose exact value is under a representable power of two `2**c` rounds to at most it:
    the rules whose bound is such a power spend no slack):
    - a comparison, `is_finite`, `sign`: at most 1 by its definition;
    - `tanh`, `logistic`, `sin`, `cos`, `erf` of a real operand: at most 1 everywhere in the reals, and at
      most 2 as computed: a native implementation's polynomial is not certified within the real range (an
      ulp past 1 breaks a bound of exactly 1), so it is given one power; unbounded over a complex operand;
    - `add`, `sub`: `|x +- y| <= 2**a + 2**b <= 2**(max(a, b) + 1)`, monotone in each operand;
    - `mul`: `|x y| <= 2**(a + b)`, bilinear, its extreme at the box's corners;
    - `max`, `min`, a selection, `neg`, `abs`, a conversion, a layout change, a maximum or minimum along an
      axis: no larger than an operand (`abs` least at its critical point 0, largest at a corner); a `min`
      with an operand at least 0, and a `max` with one at most 0, no larger than the other (a clamp's);
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
      fast accumulation) proves none (`contraction_slack`).
    A quotient, a root, a logarithm and a power are bounded where an operand's floor holds them
    (`_apart_bound`). Any other equation proves none."""
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
    if name in APART_BOUNDED or (name == "integer_pow" and equation.params["y"] < 1):
        return _apart_bound(equation, name, operands, apart, dtype) if real else None
    if name == "clamp":
        # Its operand where it lies between its bounds, else a bound: no larger than either bound, whatever
        # its operand is.
        lower, _, upper = operands
        return None if lower is None or upper is None else max(lower, upper)
    if any(bound is None for bound in operands):
        return None
    given: list[int] = [bound for bound in operands if bound is not None]
    if name in ("add", "sub"):
        return max(given) + 1
    if name == "mul":
        return given[0] + given[1]
    if name in ("max", "min"):
        # Either operand, and no more than the other where one is on the side it loses to: a minimum with a
        # value at least 0 is the other operand or less than it, a maximum with one at most 0 likewise.
        losing = 1 if name == "min" else -1
        within = [given[1 - at] for at in range(len(given)) if len(given) == 2 and apart[at].sign == losing]
        return min([max(given), *within])
    if name == "select_n":
        return max(given[1:])
    if name in ("reduce_max", "reduce_min") and _reduced(equation) < 1:
        # A maximum or a minimum of no element is its identity, an infinity.
        return None
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
    if name == "log1p" and _side_matches(apart[0], given[0], 1):
        return given[0] if given[0] == NOTHING else given[0] + 1
    if name == "reduce_sum":
        shape = _shape(equation.invars[0])
        return given[0] + _log2(math.prod(shape[axis] for axis in equation.params["axes"]))
    if name == "dot_general":
        left, right = (atom.aval.dtype for atom in equation.invars if isinstance(atom.aval, ShapedArray))
        slack = contraction_slack(equation.params.get("precision"), left, right)
        if slack is None:
            return None
        (contracted, _), _ = equation.params["dimension_numbers"]
        shape = _shape(equation.invars[0])
        return given[0] + given[1] + _log2(math.prod(shape[axis] for axis in contracted)) + slack
    return None


def product_slack(region: Region, index: int) -> int | None:
    """The powers the product of `region`'s parameters `index` adds to its bound for its mode
    (`contraction_slack` at its operands' dtypes, `Region.products`); None where an operand's dtype is
    unknown or the mode's rounding is unstated."""
    product = region.products.get(index)
    if product is None or len(product.operands) != 2 or None in product.operands:
        return None
    return contraction_slack(region.params[index][1].get("precision"), *product.operands)


def contraction_slack(precision, left, right) -> int | None:
    """The powers a contraction's bound adds for its mode (a `dot_general`'s `precision`, over operands of
    dtypes `left` and `right`): none where it
    rounds each operand once to nearest and accumulates (no precision stated, `Precision.DEFAULT` and
    `HIGHEST`, the default preset, a preset of one primitive operation over operands of 16 bits or wider);
    one where it splits them into parts whose products it sums (`Precision.HIGH`, a TPU's bf16 x3: a preset
    of several primitive operations); None, no bound, where no source states its rounding (an FP8 preset's
    accumulation, an algorithm given by its own types). A preset's facts are JAX's own
    (`programs.dot_arithmetic`). The one reading of a contraction's mode: a traced product's bound
    (`_bounded`) and a rewrite's bound of the product it makes in that mode (`program._products`)."""
    if precision is None:
        return 0
    if isinstance(precision, jax.lax.DotAlgorithmPreset):
        arithmetic = programs.dot_arithmetic(precision, left, right)
        if arithmetic is None:
            return 0
        if arithmetic.imprecise or min(arithmetic.operand_bits) < 16:
            return None
        return 0 if arithmetic.operations == 1 else 1
    if isinstance(precision, tuple) and all(isinstance(part, jax.lax.Precision) for part in precision):
        return 1 if jax.lax.Precision.HIGH in precision else 0
    if isinstance(precision, jax.lax.Precision):
        return 1 if precision == jax.lax.Precision.HIGH else 0
    return None


def _typed(dtypes: dict, magnitudes: dict, term: Term | Skeleton, atom, magnitude: Magnitude | None) -> None:
    """`term`'s dtype (`atom`'s) and its magnitude, the first where one term names two values."""
    named = key(term)
    if isinstance(atom.aval, ShapedArray):
        # An extended dtype (a typed key's) is JAX's own, which NumPy holds no type of.
        dtype = atom.aval.dtype
        dtypes.setdefault(named, dtype if _extended_dtype(dtype) is not None else np.dtype(dtype))
    magnitudes.setdefault(named, magnitude)


WIDENED = (jnp.bfloat16, jnp.float16)
"""The floats narrower than float32 a widened sum adds in float32 (`rules.narrowed`), each one's conversion
back stocked (`stocked`)."""


@functools.cache
def stocked() -> Mapping[str, tuple[jcore.Primitive, Mapping]]:
    """The primitives a stable rule writes where the program held none (`rules.stock`), each at the
    parameters JAX binds it with by default, read from JAX's own trace of them, by its symbol; a conversion
    by its target's name (`convert_float32`: a widened sum's, and each of `WIDENED`'s, its result's)."""
    # An abstract scalar: the pass reads a jit's trace after it is made, where a process compiling for a
    # device it does not hold (a compile-only GPU) places no array.
    scalar = jax.ShapeDtypeStruct((), jnp.float32)
    traced = jax.make_jaxpr(lambda a: (a / a, a - a, a + a, a * a, jnp.maximum(a, a), jnp.exp(a),
                                       jnp.log1p(a), jnp.expm1(a), jax.lax.rsqrt(a),
                                       jax.lax.integer_pow(a, -1),
                                       *(a.astype(narrow).astype(jnp.float32) for narrow in WIDENED)))(scalar)
    found = {}
    for equation in traced.jaxpr.eqns:
        symbol = OPERATIONS[equation.primitive].symbol
        if equation.primitive is jax.lax.convert_element_type_p:
            symbol = f"convert_{np.dtype(equation.params['new_dtype']).name}"
        elif equation.primitive is jax.lax.integer_pow_p and equation.params.get("y") == -1:
            symbol = "reciprocal"
        found.setdefault(symbol, (equation.primitive, dict(equation.params)))
    return found


def written_as(constructor: str, named: str | None = None) -> str | None:
    """The symbol of the operation a term of `constructor` (a `Term` method's name) writes, `_term`'s writing
    read back (`WRITTEN`): the name the term holds (`named`) where its kind's terms hold their symbol, the
    operation of its monoid (`named`) for a reduction, its kind's one operation otherwise; None for a
    constructor that writes no operation (a leaf, a literal, a state)."""
    kind = next((kind for kind, written in WRITTEN.items() if written.constructor == constructor), None)
    if kind is None:
        return None
    if WRITTEN[kind].names == "symbol":
        return named
    chosen = [operation.symbol for operation in OPERATIONS.values()
              if operation.kind == kind and (WRITTEN[kind].names is None or operation.monoid == named)]
    return chosen[0] if len(chosen) == 1 else None


def spoken_as(equation: jcore.JaxprEqn) -> str | None:
    """The symbol `equation` is spoken as in the language (`_term` makes a term of it), None for one outside
    it: a protected span is one marked call (`dew.nn.derivatives.Protected`), no equation of it."""
    operation = _in_language(equation)
    return operation.symbol if operation is not None else None


def form_value(region_: Region, form: Term, values: list) -> jax.Array:
    """One chosen `form`'s value over `region_`'s leaves' `values` (a region of the language alone, whose
    leaves are its inputs), each shared term evaluated once."""
    (made,) = forms_value(region_, [form], values)
    return made


def forms_value(region_: Region, forms: list[Term], values: list) -> list[jax.Array]:
    """Chosen `forms`' values over `region_`'s leaves' `values` together, each term any of them shares
    evaluated once (`dew.algebra.program._splices`: roots priced together are made together)."""
    known: dict[Hashable, jax.Array] = {}

    def value(term: Term) -> jax.Array:
        named = key(term)
        if named not in known:
            known[named] = value_of(term, value, region_, values)
        return known[named]

    return [value(form) for form in forms]


def _held(value: np.ndarray | complex) -> np.ndarray | complex:
    """A literal's value as a region holds it, its own copy and its type (a `TypedNdArray`'s aval, its weak
    type and placement): JAX's literal may view a NumPy scalar a function captured, which its caller may
    change after a region is derived and kept (`program._derived`)."""
    if isinstance(value, TypedNdArray):
        return TypedNdArray(np.array(value), value.aval)
    return np.array(value) if isinstance(value, np.ndarray) else value


def _literals[Node: (Term, Skeleton)](forms: type[Node], literals: list, shapes: list, dtypes: dict,
                                      magnitudes: dict) -> Callable[[jcore.Literal], Node]:
    """A region's reader of its literals into `literals`: each once by its value as the program holds it
    (`literal_key`), so equal constants are one term."""
    index: dict[Hashable, int] = {}

    def literal(atom: jcore.Literal) -> Node:
        named = literal_key(atom)
        if named not in index:
            index[named] = len(literals)
            literals.append(_held(atom.val))
        term = forms.literal(index[named])
        shapes.append((term, _shape(atom)))
        _typed(dtypes, magnitudes, term, atom, _magnitude(atom.val))
        return term

    return literal


def literal_key(atom: jcore.Literal) -> Hashable:
    """A literal's identity in a region: its dtype, shape and bits, so two literals are one term where they
    are the same constant, and never where they round apart. Its weak type is no part of it: an equation's
    types are settled where it is traced, so a weak and a strong literal of one dtype compute alike. A
    constant held as a broadcast (`normal_form.compact`: the recognition's folded masks and fills) is read
    by its values once, with the axes it is broadcast along, never its every entry's bytes."""
    dtype, array = held_bits(atom.val)
    held = compact(array)
    return dtype, array.shape, held.shape, np.ascontiguousarray(held).tobytes()


def compact(value: np.ndarray) -> np.ndarray:
    """`value` with each axis along which every entry holds the same bits one element long: the values it
    holds, each once, in its rank, which with its shape is the value (a broadcast of it). One form whatever
    holds the value, a broadcast's view or every entry written (`dew.algebra.normal_form`'s folded masks and
    fills): what a literal is keyed by (`literal_key`) and folded arithmetic computes over. The axes a view
    repeats are read off its strides; every other is compared, bit for bit, over what is left."""
    value = np.asarray(value)

    def first(held: np.ndarray, axis: int) -> np.ndarray:
        return held[(slice(None),) * axis + (slice(0, 1),)]

    for axis, (stride, size) in enumerate(zip(value.strides, value.shape, strict=True)):
        if stride == 0 and size > 1:
            value = first(value, axis)
    for axis, size in enumerate(value.shape):
        if size > 1:
            bits = np.ascontiguousarray(value).view(f"V{value.dtype.itemsize}")
            if bool(np.all(bits == first(bits, axis))):
                value = first(value, axis)
    return value


def held_bits(value) -> tuple[Hashable, np.ndarray]:
    """A constant's dtype and the array of the bits it holds: a typed PRNG key's are its key data
    (`jax.random.key_data`), its dtype the key type itself, which is its implementation (two implementations
    may print one tag); any other value's its own, its dtype by name."""
    if typed_key(value):
        return value.dtype, np.asarray(jax.random.key_data(value))
    array = np.asarray(value)
    return str(array.dtype), array


def _term[Node: (Term, Skeleton)](equation: jcore.JaxprEqn, operation: Operation,
                                   read: Callable[..., Node], param: Callable) -> Node:
    """An admitted equation over its inputs' terms, written with its operation's constructor."""
    primitive = equation.primitive
    written = WRITTEN[operation.kind]
    first, *others = (read(atom) for atom in equation.invars)
    write = getattr(first, written.constructor)
    if operation.kind in ("select", "clamp"):
        # Its operands in the equation's order: a selection's predicate, then the value where it is false; a
        # clamp's lower bound, its operand, then its upper bound.
        return write(*others)
    if operation.kind == "cast":
        return write(param(primitive, equation.params))
    if operation.kind == "reduction":
        axes = Vec[i64](*(i64(int(axis)) for axis in equation.params["axes"]))  # the reduced axes
        return write(operation.monoid, axes, param(primitive, equation.params))
    if operation.kind == "product":
        given = dict(equation.params)
        dtypes = {var.aval.dtype for var in equation.invars if isinstance(var.aval, ShapedArray)}
        if dtypes == {given.get("preferred_element_type")}:
            # A product accumulating in its operands' own dtype is one of no preferred type (`rules.plain`).
            given["preferred_element_type"] = None
        operands = tuple(np.dtype(var.aval.dtype) if isinstance(var.aval, ShapedArray) else None
                         for var in equation.invars)
        (traced,) = (var.aval for var in equation.outvars if isinstance(var.aval, ShapedArray))
        return write(others[0], param(primitive, given, Product(operands, np.dtype(traced.dtype))))
    return write(operation.symbol, *others, param(primitive, equation.params))


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
        return jax.lax.select_n_p.bind(*_aligned_explicit(*(value(operand) for operand in args)))
    if (args := get_callable_args(term, Term.clamp)) is not None:
        # A clamp binds as itself, its derivative JAX's own (zero at a bound, where a maximum and a
        # minimum would halve it), keeping JAX's placement from its middle operand.
        return jax.lax.clamp_p.bind(*jax_core.auto_insert_reshard(*(value(operand) for operand in args)))
    # Each other constructor's primitive parameters, by index, last after its operand and its other arguments.
    for constructor in (Term.unary, Term.convert, Term.shaped, Term.reduce):
        if (args := get_callable_args(term, constructor)) is not None:
            primitive, given = region_.params[integer(args[-1])]
            if primitive is jax.lax.logistic_p:
                # A logistic, a program's own or Dew's read as it (`normal_form._computed`), computed as Dew's
                # one logistic (`activations.logistic`: its arithmetic `precision.logistic_value`, never a
                # backend's, a v5e's approximate one or Mosaic's under `approx_math`; its tangent declared, so
                # a rewritten program JAX differentiates keeps a number where `e^-x` overflows), rounded once
                # to its operand's dtype.
                from dew.nn.activations import logistic

                operand = value(args[0])
                return logistic(operand).astype(jnp.result_type(operand))
            made = primitive.bind(value(args[0]), **given)
            # A barrier of one value binds as a primitive of many results, its one result its value.
            return made[0] if primitive.multiple_results else made
    raise ValueError(f"a term of no known constructor: {term}")


def _aligned_explicit(*operands: jax.Array) -> tuple[jax.Array, ...]:
    """Align operands of `select_n`: `auto_insert_reshard` plus resharding any replicated operand
    (`explicit_spec is None`) onto the explicit placement shared by split siblings of the same shape, which
    JAX's `select_n` requires under `AxisType.Explicit`."""
    from dew.nn.sharding import explicit_spec

    aligned = tuple(jax_core.auto_insert_reshard(*operands))
    specs = [explicit_spec(operand) if isinstance(operand, jax.Array) else None for operand in aligned]
    found = next(((spec, operand.shape) for operand, spec in zip(aligned, specs, strict=True)
                  if spec is not None), None)
    if found is None:
        return aligned
    split, shape = found
    return tuple(jax.sharding.reshard(operand, split)
                 if isinstance(operand, jax.Array) and spec is None and operand.shape == shape
                 else operand for operand, spec in zip(aligned, specs, strict=True))


def _sort(declared: TypeOrVarRef) -> type[BaseExpr]:
    """The sort an argument is declared of (egglog's type reference): its class by its module and name, a
    parameterized one (`Vec[i64]`) given its parameters."""
    import importlib

    if not isinstance(declared, TypeRefWithVars):
        raise TypeError(f"{declared} is a type variable, no sort a term's argument takes")
    sort = getattr(importlib.import_module(declared.ident.module or "egglog"), declared.ident.name)
    parameters = tuple(map(_sort, declared.args))
    return sort[parameters[0] if len(parameters) == 1 else parameters] if parameters else sort


def _arguments(constructor) -> tuple[type[BaseExpr], ...]:
    """The sorts of the arguments `constructor` takes after the term it is called on, as `Term` declares its
    method (egglog's signature of it)."""
    signature = declarations(Term).get_callable_decl(constructor.__egg_ref__).signature
    if not isinstance(signature, FunctionSignature):
        raise TypeError(f"{constructor} is declared by {signature!r}, no signature of its arguments")
    return tuple(map(_sort, signature.arg_types[1:]))


LANGUAGE: Mapping = {constructor: _arguments(constructor)
                     for constructor in (getattr(Term, written.constructor) for written in WRITTEN.values())}
"""The language's constructors that read terms (a leaf's and a literal's none: each kind's, `WRITTEN`), each
by the sorts of the arguments it takes after the term it is called on, as `Term` declares them: the one
inventory every reader of the language walks (`subterms`, `pattern`)."""


def pattern(constructor, name: str) -> tuple[Term, tuple[Term, ...]]:
    """`constructor` (a `LANGUAGE` entry) over fresh variables named from `name`: the pattern, and the terms
    it reads, the one it is called on first."""
    from egglog import var

    called = var(f"{name}_called", Term)
    kinds = LANGUAGE[constructor]
    arguments = [var(f"{name}_{index}", kind) for index, kind in enumerate(kinds)]
    read = tuple(argument for argument, kind in zip(arguments, kinds, strict=True) if kind is Term)
    return constructor(called, *arguments), (called, *read)


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
