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
from jax._src import core as jax_core
from jax._src.literals import TypedNdArray
from jax.core import ShapedArray
from jax.experimental.layout import Layout
from jax.extend import core as jcore
from jax.typing import DTypeLike

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
readers, as a declaration's dropped products are). Every other equation is a leaf of the region, another
library's primitive of one of these names among them, and the sum JAX's derivatives accumulate with
(`add_any`), which no rule reads."""


def _speaks(kind: str, *primitives: jcore.Primitive, **facts) -> None:
    """Declares `primitives` operations of the language of `kind`, each under its own name."""
    for primitive in primitives:
        OPERATIONS[primitive] = Operation(primitive.name, kind, **facts)


_speaks("binary", jax.lax.add_p, jax.lax.sub_p, jax.lax.mul_p, jax.lax.div_p, jax.lax.max_p, jax.lax.min_p,
        jax.lax.atan2_p, jax.lax.rem_p, jax.lax.nextafter_p)
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


def symbol(primitive: jcore.Primitive) -> str | None:
    """`primitive`'s symbol in the language (`OPERATIONS`), None for one it does not speak."""
    spoken_as = OPERATIONS.get(primitive)
    return None if spoken_as is None else spoken_as.symbol


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
    def single_float32(self) -> bool:
        """Whether every value the region's program traced is fp32 (`dtypes`; an extended dtype and a mask's
        bool aside, `where(y > 0, a / y, 0)`'s): a division of it, traced or a rewrite's, fp32 too
        (`cost.work`); an integer's or another float's program none."""
        return all(np.dtype(dtype) == np.float32 for dtype in self.dtypes.values()
                   if not jnp.issubdtype(dtype, jax.dtypes.extended) and np.dtype(dtype) != np.bool_)

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


def parameter_key(value) -> Hashable:
    """An equation's parameter as what it is, the one structural key the algebra pass (`parameters_key`,
    `program.structure`) and the kernel tuner (`dew.nn.kernels.tuning.program`) read: plain data, the same in
    every process, tagged with its kind. A number by its value whatever its Python or NumPy type (a float by
    its bits, so a NaN is one key and -0.0 another than 0.0); a sequence's parts, a named tuple's with its
    type; a set's and a mapping's parts, ordered by their keys' printed forms (an order, never an identity); a
    dtype by its name; an enumeration by its type and member; a dataclass by its type and fields; a mesh by
    its devices, axes and axis types, an abstract one by its sizes, axes, axis types and device; a sharding by
    its mesh, partitions, memory and logical device order; an abstract value by its shape, dtype, weak type,
    placement (partitions, mesh axes, memory), memory space, layout and the manual axes it varies over,
    reduces or leaves unreduced; a layout by its dimension order, tiling and sub-byte element size; an array
    by its dtype, shape and bytes' digest; a function by its module and name, a partial one with its bound
    arguments. Any other object is its type's name and the object itself, compared by its own equality, never
    by its printed form, which a key kept in another process cannot read (`opaque`); an object that cannot be
    hashed (JAX's `WrappedFun` of a derivative rule, hashing an `Unhashable` tree) by its identity, held alive
    by the key (`_Identity`), so it is equal to itself alone. `value` is any parameter JAX's primitives take,
    which share no type."""
    kind = type(value)
    if value is None or kind is bool or kind is str:
        return kind.__name__, value
    if kind is int:
        return "int", value
    if kind is float:
        return "float", value.hex()
    if kind is tuple:
        return "sequence", "builtins.tuple", tuple(parameter_key(part) for part in value)
    if kind is ShapedArray:
        return _abstract_key(value)
    if kind is list:
        return "sequence", "builtins.list", tuple(parameter_key(part) for part in value)
    if isinstance(value, jcore.Jaxpr):
        digest, opaques = jaxpr_digest(value)
        held = ("object", "subkey", _SubKey(digest, opaques)) if opaques else ()
        consts = tuple(_nested_const_key(const) for const in value.consts) if value.consts else ()
        return "jaxpr", digest, held, consts
    return _parameter_key_other(value, kind)


def _array_key(value, kind: type) -> Hashable:
    """Cross-process array digest (`parameter_key`), never transferring non-addressable global shards."""
    if isinstance(value, jax.Array) and not value.is_fully_addressable:
        return "object", _named(kind), _Identity(value)
    # The key type's printed form, which a key kept in another process can read.
    _, array = held_bits(value)
    return ("array", str(value.dtype), array.shape, str(array.dtype),
            hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest())


def _parameter_key_other(value, kind: type) -> Hashable:
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
        # The layout XLA chooses, one singleton.
        return "layout", "auto"
    if isinstance(value, Layout):
        return ("layout", parameter_key(value.major_to_minor), parameter_key(value.tiling),
                value.sub_byte_element_size_in_bits)
    if isinstance(value, jax.sharding.Mesh):
        # An empty mesh (a jit's context outside any) holds no devices.
        devices = () if value.empty else value.device_ids
        return ("mesh", parameter_key(devices), parameter_key(value.axis_names),
                parameter_key(value.axis_types))
    if isinstance(value, jax.sharding.AbstractMesh):
        return ("abstract mesh", parameter_key(value.axis_sizes), parameter_key(value.axis_names),
                parameter_key(value.axis_types), parameter_key(value.abstract_device))
    if isinstance(value, jax.sharding.NamedSharding):
        # Its mesh, partitions, memory and logical device order, as JAX pickles it, which its equality reads.
        _, arguments = value.__reduce__()
        return "named sharding", parameter_key(arguments)
    if isinstance(value, functools.partial):
        return "partial", parameter_key(value.func), parameter_key(value.args), parameter_key(value.keywords)
    if isinstance(value, jax.sharding.PartitionSpec):
        return ("partitions", parameter_key(tuple(value)), parameter_key(value.unreduced),
                parameter_key(value.reduced), parameter_key(value.unreduced_kind))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = dataclasses.fields(value)
        return "record", _named(kind), tuple((field.name, parameter_key(getattr(value, field.name)))
                                             for field in fields)
    if isinstance(value, (tuple, list)):
        # A named tuple's type names its fields, which its parts take in order.
        return "sequence", _named(kind), tuple(parameter_key(part) for part in value)
    if isinstance(value, (set, frozenset)):
        return "set", tuple(sorted((parameter_key(part) for part in value), key=repr))
    if isinstance(value, Mapping):
        return "mapping", tuple(sorted(((parameter_key(name), parameter_key(part))
                                        for name, part in value.items()), key=repr))
    if isinstance(value, types.FunctionType):
        return "function", value.__module__, value.__qualname__
    try:
        hash(value)
    except TypeError:
        # A function JAX wraps (a derivative rule's `WrappedFun`) hashes its parts, an `Unhashable` tree among
        # them: keyed by its identity.
        return "object", _named(kind), _Identity(value)
    return "object", _named(kind), value


class _SubKey:
    """Memoized structural key for a `jcore.Jaxpr` with O(1) cached hash and DAG-bounded equality
    (`_subkeys_equal`), held by child reference so shared nested-jit DAGs stay O(V) in work and slots."""

    __slots__ = ("_hash", "digest", "opaques")

    def __init__(self, digest: str, opaques: tuple[Hashable, ...]) -> None:
        self.digest = digest
        self.opaques = opaques
        self._hash = hash((digest, opaques))

    def __repr__(self) -> str:
        return f"_SubKey({self.digest!r})"

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if not isinstance(other, _SubKey) or self._hash != other._hash or self.digest != other.digest:
            return False
        return _subkeys_equal(self, other)


def _subkey_child(entry) -> _SubKey | None:
    """Return the child `_SubKey` wrapped by `("object", "subkey", child)`, or None."""
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
"""Cross-process `(blake2b_hex, opaque_objects)` digest per `jcore.Jaxpr`, held while the jaxpr lives."""

_IN_PROCESS_DIGESTS: weakref.WeakKeyDictionary[jcore.Jaxpr, _SubKey] = weakref.WeakKeyDictionary()
"""In-process DAG-memoized `_SubKey` per `jcore.Jaxpr` (`jaxpr_digest(..., in_process=True)`)."""

_BODY_KEYS: weakref.WeakKeyDictionary[jcore.Jaxpr, _SubKey | None] = weakref.WeakKeyDictionary()
"""In-process structural key per const-free `jcore.Jaxpr` (`body_key`), None when unshareable."""


class _Unshareable(Exception):
    """Raised internally by `body_key` when a body holds nested consts or device arrays."""


def _nested_const_key(const) -> Hashable:
    """Key a nested `Jaxpr.consts` entry without transferring device arrays to host."""
    if isinstance(const, jax.Array) and not isinstance(const, jax_core.Tracer):
        return "object", _named(type(const)), _Identity(const)
    return parameter_key(const)


def _extended_dtype(dtype) -> Hashable | None:
    """Return an opaque entry for extended JAX dtypes (such as typed PRNG specs) that share a tag,
    without calling `jnp.issubdtype` on plain `np.dtype` (including `float0`) or string tags."""
    if isinstance(dtype, (str, np.dtype)):
        return None
    return ("object", "extended_dtype", dtype) if jnp.issubdtype(dtype, jax.dtypes.extended) else None


def _in_process_digest(jaxpr: jcore.Jaxpr) -> _SubKey:
    """Return the DAG-memoized `_SubKey` for `jaxpr` (`jaxpr_digest(..., in_process=True)`)."""
    inner = jaxpr.jaxpr
    if (cached := _IN_PROCESS_DIGESTS.get(inner)) is not None:
        return cached
    sub = _SubKey(*_body_digest(inner, in_process=True, reject_device=False))
    _IN_PROCESS_DIGESTS[inner] = sub
    return sub


def _body_key_of(jaxpr: jcore.Jaxpr) -> _SubKey | None:
    """Return the DAG-memoized `_SubKey` for const-free `jaxpr`, or None when unshareable."""
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


def _dataclass_param_key(value, kind: type, *, reject_device: bool) -> Hashable:
    """In-process structural key for a dataclass instance preserving exact class identity and guarding
    non-field attributes (so factory-made callable dataclasses with identical qualnames never collide)."""
    fields = dataclasses.fields(value)
    field_names = {field.name for field in fields}
    inst_dict = value.__dict__ if any("__dict__" in cls.__dict__ for cls in kind.__mro__) else None
    extra_dict = inst_dict is not None and not set(inst_dict).issubset(field_names)
    extra_slots = any(
        not set((slots,) if isinstance(slots, str) else slots).issubset(
            field_names | {"__dict__", "__weakref__"})
        for cls in kind.__mro__ if (slots := cls.__dict__.get("__slots__")) is not None)
    if extra_dict or extra_slots:
        if reject_device:
            raise _Unshareable
        return "object", _named(kind), _Identity(value)
    return "record", ("object", "class", kind), tuple(
        (field.name, _in_process_param_key(getattr(value, field.name), reject_device=reject_device))
        for field in fields)


def _in_process_param_key(value, *, reject_device: bool) -> Hashable:
    """In-process structural key for an equation parameter preserving executable callable, primitive, and
    extended-dtype identities while never copying `jax.Array` device buffers to host."""
    kind = type(value)
    if value is None or kind is bool or kind is str:
        return kind.__name__, value
    if kind is int:
        return "int", value
    if kind is float:
        return "float", value.hex()
    if kind is tuple or kind is list:
        return "sequence", f"builtins.{kind.__name__}", tuple(
            _in_process_param_key(part, reject_device=reject_device) for part in value)
    if isinstance(value, ShapedArray):
        return _abstract_key(value), _extended_dtype(value.dtype)
    if isinstance(value, jcore.Jaxpr):
        if reject_device:
            child = _body_key_of(value)
            if child is None:
                raise _Unshareable
            return "jaxpr", child.digest, ("object", "subkey", child), ()
        child = _in_process_digest(value)
        consts = tuple(_nested_const_key(c) for c in value.consts) if value.consts else ()
        return "jaxpr", child.digest, ("object", "subkey", child), consts
    if isinstance(value, jax.Array) and not isinstance(value, jax_core.Tracer):
        if reject_device:
            raise _Unshareable
        return "object", _named(kind), _Identity(value)
    if isinstance(value, functools.partial):
        return ("partial", _in_process_param_key(value.func, reject_device=reject_device),
                _in_process_param_key(value.args, reject_device=reject_device),
                _in_process_param_key(value.keywords, reject_device=reject_device))
    if isinstance(value, (types.FunctionType, types.BuiltinFunctionType)):
        return "object", _named(kind), value
    if callable(value) and not isinstance(value, (type, np.dtype)):
        if reject_device:
            raise _Unshareable
        return "object", _named(kind), _Identity(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _dataclass_param_key(value, kind, reject_device=reject_device)
    if isinstance(value, (tuple, list)):
        return "sequence", _named(kind), tuple(
            _in_process_param_key(part, reject_device=reject_device) for part in value)
    if isinstance(value, (set, frozenset)):
        return "set", tuple(sorted(
            (_in_process_param_key(part, reject_device=reject_device) for part in value), key=repr))
    if isinstance(value, Mapping):
        return "mapping", tuple(sorted((
            (_in_process_param_key(name, reject_device=reject_device),
             _in_process_param_key(part, reject_device=reject_device))
            for name, part in value.items()), key=repr))
    return _parameter_key_other(value, kind)


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


Magnitude = int
"""A bound on a value's magnitude (docs/design/algebra.md §6.3): its power of two, `|x| <= 2**log2`, a number
its constants', its declared ranges' and its equations' prove of a value proven finite (`Apart.finite`); a
value no number bounds, or that may be a NaN or an infinity, has none (None). A
sum's is twice the larger of its terms', a product's their product, a contraction's their product times its
length. Proved under its arithmetic's contract (`_real_bound`): IEEE rounding to nearest, a native
transcendental given one power past its real range, a contraction's mode one it states the rounding of
(`contraction_slack`); an equation outside the contract is unbounded, so overflow admission
(`rules.dominated`) never reads a bound no argument holds."""

NOTHING = -(2 ** 20)
"""A zero's magnitude: under every other."""


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
    opaque: list[Opaque] = []
    for equation in jaxpr.eqns:
        composed, facts = _propagate_facts(equation, laid, magnitude_of, apart_of)
        if composed is not None and facts is not None:
            origin, params_ = composed
            out = equation.outvars[0]
            env[out] = read(origin).shaped("broadcast_in_dim", param(equation.primitive, params_))
            shapes.append((env[out], _shape(out)))
            bound, apart[out] = facts
            _typed(dtypes, magnitudes, env[out], out, bound)
            depths[out] = depths.get(origin, 0) + 1
            continue
        term = _term(equation, read, param)
        if term is not None and facts is not None:
            out = equation.outvars[0]
            env[out] = term
            shapes.append((term, _shape(out)))
            bound, apart[out] = facts
            _typed(dtypes, magnitudes, term, out, bound)
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
                  }))


def _prove_equation(equation: jcore.JaxprEqn, magnitude_of: Callable[..., Magnitude | None],
                    apart_of: Callable[..., Apart]) -> tuple[Magnitude | None, Apart]:
    """The finite magnitude bound and `Apart` proven of `equation`'s single output from its operands'."""
    operands = [magnitude_of(atom) for atom in equation.invars]
    name = symbol(equation.primitive)
    held_apart = [apart_of(atom) for atom in equation.invars]
    # An exponential of a value at most 0 is at most 1 in the reals, at most 2 as computed: a native
    # exponential is not certified under 1 (`ex2.approx`'s relative error, 2**-22.5, is more than an ulp
    # of 1), so it is given one power.
    exponentiated = name in ("exp", "exp2") and held_apart[0].sign < 0
    bound = UNIT + 1 if exponentiated else _bounded(equation, operands, held_apart)
    proved = _apart(equation, held_apart, operands, bound)
    # A bound is stated of a value proven finite alone, as a rewrite's side condition reads it (row 5780;
    # owner, 2026-11-14): a NaN's tanh is no number, and a product of a value that overflowed reads the
    # bound its real factors have and computes an infinity.
    return (bound if proved.finite else None), proved


def _in_language(equation: jcore.JaxprEqn) -> Operation | None:
    """`equation`'s declared `Operation` where the term language speaks it (one output, matching arity)."""
    operation = OPERATIONS.get(equation.primitive)
    if operation is None or len(equation.outvars) != 1:
        return None
    return operation if len(equation.invars) == WRITTEN[operation.kind].arity else None


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
    if _in_language(equation) is not None:
        return None, _prove_equation(equation, magnitude_of, apart_of)
    return None, None


def _host_array(value) -> np.ndarray | None:
    """Return `np.asarray(value)` for host constants, or None for PRNG keys, tracers, or concrete device
    `jax.Array`s (avoiding device transfers and `TracerArrayConversionError`)."""
    if typed_key(value) or isinstance(value, (jax_core.Tracer, jax.Array)):
        return None
    return np.asarray(value)


def _apart_of(atom, apart: Mapping[jcore.Var, Apart]) -> Apart:
    """`atom`'s proven `Apart` from `apart`, or its constant `Apart` for a literal."""
    if isinstance(atom, jcore.Literal):
        if isinstance(atom.val, (jax_core.Tracer, jax.Array)) and not typed_key(atom.val):
            return _held_apart(WHOLE, atom.aval.dtype) if isinstance(atom.aval, ShapedArray) else NOWHERE
        dtype = atom.val.dtype if typed_key(atom.val) else np.asarray(atom.val).dtype
        return _held_apart(_constant_apart(atom.val), dtype)
    return apart.get(atom, NOWHERE)


def _side_matches(held: Apart, bound: Magnitude | None, side: int) -> bool:
    """Whether a value proven `(bound, held)` lies on `side` (`>= 0` for `1`, `<= 0` for `-1`), treating an
    exact zero (`bound == NOTHING` and `held.number`) as on both sides."""
    return held.sign == side or (held.number and bound == NOTHING)


def _reshape_kept(kept: tuple[int | None, ...], in_shape: tuple[int, ...],
                  out_shape: tuple[int, ...]) -> tuple[int | None, ...] | None:
    """Propagate tracked source axes across a reshape that only inserts or drops size-1 axes."""
    in_non1 = [(axis, size) for axis, size in enumerate(in_shape) if size != 1]
    out_non1 = [(axis, size) for axis, size in enumerate(out_shape) if size != 1]
    if [size for _, size in in_non1] != [size for _, size in out_non1]:
        return None
    mapped: list[int | None] = [None] * len(out_shape)
    for (in_ax, _), (out_ax, _) in zip(in_non1, out_non1, strict=True):
        mapped[out_ax] = kept[in_ax]
    return tuple(mapped)


def _broadcast_kept(kept: tuple[int | None, ...], in_shape: tuple[int, ...],
                    out_shape: tuple[int, ...], dims: Sequence[int]) -> tuple[int | None, ...]:
    """Propagate tracked source axes across `broadcast_in_dim_p`, clearing broadcast size-1 axes to None."""
    mapped: list[int | None] = [None] * len(out_shape)
    for in_ax, out_ax in enumerate(dims):
        if in_shape[in_ax] == out_shape[out_ax]:
            mapped[out_ax] = kept[in_ax]
    return tuple(mapped)


def _aligned_unit_axes(kept: tuple[int | None, ...], rank: int) -> bool:
    """Whether `kept` aligns 1-to-1 with every non-reduced axis of a rank-`rank` tensor and reduces at least
    one axis."""
    return (len(kept) == rank and any(entry is None for entry in kept)
            and all(entry in (axis, None) for axis, entry in enumerate(kept)))


def _lossless_float_cast(source, out) -> bool:
    """Whether `convert_element_type` is a same-width or widening floating cast (exact value preserved)."""
    if not (isinstance(source.aval, ShapedArray) and isinstance(out.aval, ShapedArray)):
        return False
    src_dt, dst_dt = source.aval.dtype, out.aval.dtype
    return (jnp.issubdtype(src_dt, jnp.floating) and jnp.issubdtype(dst_dt, jnp.floating)
            and jnp.dtype(dst_dt).itemsize >= jnp.dtype(src_dt).itemsize)


class _NormTrack:
    """Tracks verified L2-norm denominator provenance (`sum(x^2, axes) + eps` with `eps > 0`) for
    `proven(..., deep=True)` so `x * rsqrt(sum(x^2) + eps)` and `x / sqrt(sum(x^2) + eps)` prove unit norm
    (`bound = UNIT + 1`, finite number) without false positives across mismatched axes or operands."""

    def __init__(self, sq_sum: dict | None = None, inv_norm: dict | None = None,
                 pos_norm: dict | None = None) -> None:
        self.origin: dict[jcore.Var, jcore.Var] = {}
        self.squared: dict[jcore.Var, jcore.Var] = {}
        self.sq_sum: dict[jcore.Var, tuple[jcore.Var, tuple[int | None, ...], bool]] = dict(sq_sum or {})
        self.inv_norm: dict[jcore.Var, tuple[jcore.Var, tuple[int | None, ...]]] = dict(inv_norm or {})
        self.pos_norm: dict[jcore.Var, tuple[jcore.Var, tuple[int | None, ...]]] = dict(pos_norm or {})

    def _canon(self, var: jcore.Var) -> jcore.Var:
        return self.origin.get(var, var)

    def _propagate_layout(self, equation: jcore.JaxprEqn, src: jcore.Var, out: jcore.Var) -> None:
        in_shape, out_shape, prim = _shape(src), _shape(out), equation.primitive

        def move(kept: tuple[int | None, ...]) -> tuple[int | None, ...] | None:
            if prim is jax.lax.broadcast_in_dim_p:
                return _broadcast_kept(kept, in_shape, out_shape, equation.params["broadcast_dimensions"])
            dims = equation.params.get("dimensions")
            if dims is not None and tuple(dims) != tuple(range(len(in_shape))):
                return None
            return _reshape_kept(kept, in_shape, out_shape)

        if (entry := self.sq_sum.get(src)) is not None and (moved := move(entry[1])) is not None:
            self.sq_sum[out] = (entry[0], moved, entry[2])
        for table in (self.inv_norm, self.pos_norm):
            if (entry := table.get(src)) is not None and (moved := move(entry[1])) is not None:
                table[out] = (entry[0], moved)

    def _unit_check(self, scaled: jcore.Var | jcore.Literal, divisor: jcore.Var | jcore.Literal,
                    table: Mapping[jcore.Var, tuple[jcore.Var, tuple[int | None, ...]]],
                    out_shape: tuple[int, ...]) -> tuple[bool, bool]:
        if not (isinstance(scaled, jcore.Var) and isinstance(divisor, jcore.Var)) or divisor not in table:
            return False, False
        src, kept = table[divisor]
        if (self._canon(scaled) is not src or _shape(scaled) != out_shape
                or not _aligned_unit_axes(kept, len(out_shape))):
            return False, False
        return True, bool(kept) and kept[-1] is None

    def _step_cast(self, src: jcore.Var, out: jcore.Var) -> None:
        if not _lossless_float_cast(src, out):
            return
        self.origin[out] = self._canon(src)
        for table in (self.sq_sum, self.inv_norm, self.pos_norm):
            if src in table:
                table[out] = table[src]

    def _step_add_max(self, prim: jcore.Primitive, invars: Sequence, out: jcore.Var,
                      apart_of: Callable[..., Apart]) -> None:
        for i in (0, 1):
            cand, other = invars[i], invars[1 - i]
            if not isinstance(cand, jcore.Var):
                continue
            if (entry := self.sq_sum.get(cand)) is not None and (oa := apart_of(other)).sign > 0:
                self.sq_sum[out] = (entry[0], entry[1], entry[2] or oa.positive)
            if (prim is jax.lax.max_p and (pos := self.pos_norm.get(cand)) is not None
                    and apart_of(other).sign > 0):
                self.pos_norm[out] = pos

    def step(self, equation: jcore.JaxprEqn,
             apart_of: Callable[..., Apart]) -> tuple[Magnitude | None, Apart] | None:
        """Update norm provenance for `equation`, returning unit-norm facts when `equation` divides/scales
        `x` by a verified positive-floored L2 norm of `x`."""
        if len(equation.outvars) != 1 or not isinstance(out_aval := equation.outvars[0].aval, ShapedArray):
            return None
        prim, out, invars = equation.primitive, equation.outvars[0], equation.invars
        if prim is jax.lax.convert_element_type_p and isinstance(invars[0], jcore.Var):
            self._step_cast(invars[0], out)
            return None
        if (prim is jax.lax.square_p
                or (prim is jax.lax.integer_pow_p and equation.params.get("y") == 2)
                or (prim is jax.lax.mul_p and len(invars) == 2 and invars[0] is invars[1])):
            if isinstance(invars[0], jcore.Var):
                self.squared[out] = self._canon(invars[0])
            return None
        if prim is jax.lax.reduce_sum_p and isinstance(invars[0], jcore.Var):
            if (src := self.squared.get(invars[0])) is not None:
                red = frozenset(int(ax) for ax in equation.params["axes"])
                kept = tuple(ax for ax in range(len(_shape(invars[0]))) if ax not in red)
                self.sq_sum[out] = (src, kept, False)
            return None
        if prim in (jax.lax.broadcast_in_dim_p, jax.lax.reshape_p) and isinstance(invars[0], jcore.Var):
            self._propagate_layout(equation, invars[0], out)
            return None
        if prim in (jax.lax.add_p, jax.lax.max_p) and len(invars) == 2:
            self._step_add_max(prim, invars, out, apart_of)
            return None
        if prim in (jax.lax.rsqrt_p, jax.lax.sqrt_p) and isinstance(invars[0], jcore.Var):
            if (entry := self.sq_sum.get(invars[0])) is not None and entry[2]:
                (self.inv_norm if prim is jax.lax.rsqrt_p else self.pos_norm)[out] = (entry[0], entry[1])
            return None
        if prim in (jax.lax.mul_p, jax.lax.div_p) and len(invars) == 2:
            out_shape = tuple(out_aval.shape)
            if prim is jax.lax.mul_p:
                hit, last = self._unit_check(invars[0], invars[1], self.inv_norm, out_shape)
                if not hit:
                    hit, last = self._unit_check(invars[1], invars[0], self.inv_norm, out_shape)
            else:
                hit, last = self._unit_check(invars[0], invars[1], self.pos_norm, out_shape)
            if hit:
                return UNIT + 1, _held_apart(
                    Apart(None, 0, number=True, finite=True, unit_norm=last), out_aval.dtype)
        return None

    def child(self, outer_invars: Sequence, inner_invars: Sequence[jcore.Var]) -> _NormTrack:
        """Map tracked norm entries into a peeled call body when both source and sum enter the call."""
        pairs = [(o, i) for o, i in zip(outer_invars, inner_invars, strict=True) if isinstance(o, jcore.Var)]
        by_origin: dict[jcore.Var, jcore.Var] = {}
        for outer, inner in pairs:
            by_origin.setdefault(self._canon(outer), inner)
        return _NormTrack(*({i: (by_origin[e[0]], *e[1:]) for o, i in pairs
                             if (e := t.get(o)) is not None and e[0] in by_origin}
                            for t in (self.sq_sum, self.inv_norm, self.pos_norm)))


def _same_atom(a, b) -> bool:
    """Literal-safe identity/value match between two Jaxpr atoms (JAX 0.12 `Literal` is unhashable)."""
    if a is b:
        return True
    if isinstance(a, jcore.Literal) and isinstance(b, jcore.Literal):
        ha, hb = _host_array(a.val), _host_array(b.val)
        return ha is not None and hb is not None and ha.dtype == hb.dtype and bool(np.array_equal(ha, hb))
    return False


def _nan_guard_apart(equation: jcore.JaxprEqn, made_eq: Mapping[jcore.Var, jcore.JaxprEqn],
                     magnitude_of: Callable[..., Magnitude | None],
                     apart_of: Callable[..., Apart]) -> Apart | None:
    """Recognize JAX's `logaddexp(u, v)` NaN-propagation guard `select_n(ne(u - v, u - v), safe, u + v)`:
    when one operand of `sub(u, v)` is statically bounded and finite, `u - v` is NaN only if the other is
    NaN, so every non-NaN value takes `safe` (`false_val`)."""
    if equation.primitive is not jax.lax.select_n_p or len(equation.invars) != 3:
        return None
    pred, false_val, true_val = equation.invars
    pred_eq = made_eq.get(pred) if isinstance(pred, jcore.Var) else None
    if pred_eq is None or pred_eq.primitive is not jax.lax.ne_p or pred_eq.invars[0] is not pred_eq.invars[1]:
        return None
    diff = pred_eq.invars[0]
    diff_eq = made_eq.get(diff) if isinstance(diff, jcore.Var) else None
    if diff_eq is None or diff_eq.primitive is not jax.lax.sub_p:
        return None
    left, right = diff_eq.invars
    if not ((apart_of(left).finite and magnitude_of(left) is not None)
            or (apart_of(right).finite and magnitude_of(right) is not None)):
        return None
    if _same_atom(true_val, left) or _same_atom(true_val, right):
        return apart_of(false_val)
    true_eq = made_eq.get(true_val) if isinstance(true_val, jcore.Var) else None
    if true_eq is not None and true_eq.primitive is jax.lax.add_p and len(true_eq.invars) == 2:
        a0, a1 = true_eq.invars
        if ((_same_atom(a0, left) and _same_atom(a1, right))
                or (_same_atom(a0, right) and _same_atom(a1, left))):
            return apart_of(false_val)
    return None


def _shared_sign(operands: Sequence[Apart], bounds: Sequence[Magnitude | None]) -> int:
    """Common sign (`1` if all `>= 0`, `-1` if all `<= 0`, else `0`) honoring exact zero `NOTHING`."""
    if all(_side_matches(h, b, 1) for h, b in zip(operands, bounds, strict=True)):
        return 1
    if all(_side_matches(h, b, -1) for h, b in zip(operands, bounds, strict=True)):
        return -1
    return 0


def _concat_facts(operands: Sequence[Apart], bounds: Sequence[Magnitude | None], dtype,
                  *, keep_unit: bool = False) -> tuple[Magnitude | None, Apart]:
    """Intersect sign, floor, finiteness, unit-norm, and magnitude bounds across `concatenate_p` operands."""
    if not operands:
        return None, NOWHERE
    sign = _shared_sign(operands, bounds)
    floors = [h.floor for h in operands if h.floor is not None]
    floor = min(floors) if len(floors) == len(operands) else None
    known_bounds = [b for b in bounds if b is not None]
    bound = max(known_bounds) if len(known_bounds) == len(bounds) else None
    finite = all(h.finite for h in operands) and (
        not jnp.issubdtype(dtype, jnp.floating) or _within(bound, dtype))
    unit = keep_unit and all(h.unit_norm for h in operands)
    proved = _held_apart(Apart(floor, sign, all(h.number for h in operands), finite, unit_norm=unit), dtype)
    return (bound if proved.finite else None), proved


def _deep_refine(equation: jcore.JaxprEqn, bound: Magnitude | None, proved: Apart,
                 made_eq: Mapping[jcore.Var, jcore.JaxprEqn],
                 magnitude_of: Callable[..., Magnitude | None], apart_of: Callable[..., Apart],
                 norm_track: _NormTrack) -> tuple[Magnitude | None, Apart]:
    """Refine an in-language equation's `(bound, apart)` with unit-norm, NaN-guard, and `log(logistic)`."""
    out = equation.outvars[0]
    if (unit := norm_track.step(equation, apart_of)) is not None:
        bound, proved = unit
    if (equation.primitive is jax.lax.convert_element_type_p and proved.finite
            and apart_of(equation.invars[0]).unit_norm
            and isinstance(out.aval, ShapedArray) and jnp.issubdtype(out.aval.dtype, jnp.floating)
            and jnp.dtype(out.aval.dtype).itemsize >= 2):
        proved = proved._replace(unit_norm=True)
    if equation.primitive is jax.lax.select_n_p:
        guarded = _nan_guard_apart(equation, made_eq, magnitude_of, apart_of)
        if guarded is not None:
            proved = _held_apart(guarded, out.aval.dtype) if isinstance(out.aval, ShapedArray) else guarded
    if (equation.primitive in (jax.lax.log_p, jax.lax.log2_p)
            and isinstance(equation.invars[0], jcore.Var)
            and (src_eq := made_eq.get(equation.invars[0])) is not None
            and src_eq.primitive is jax.lax.logistic_p):
        proved = proved._replace(sign=-1)
    return bound, proved


def _deep_nonlang(equation: jcore.JaxprEqn, held: dict[jcore.Var, Magnitude | None],
                  apart: dict[jcore.Var, Apart], magnitude_of: Callable[..., Magnitude | None],
                  apart_of: Callable[..., Apart], norm_track: _NormTrack, depth: int) -> bool:
    """Handle slice/stop_gradient, `concatenate_p`, and peeled called bodies for `proven(..., deep=True)`."""
    prim = equation.primitive
    if (prim in (jax.lax.slice_p, jax.lax.dynamic_slice_p, jax.lax.stop_gradient_p)
            and len(equation.outvars) == 1):
        out = equation.outvars[0]
        held[out], apart[out] = magnitude_of(equation.invars[0]), apart_of(equation.invars[0])
        return True
    if (prim is jax.lax.concatenate_p and len(equation.outvars) == 1
            and isinstance(out_aval := equation.outvars[0].aval, ShapedArray)):
        out = equation.outvars[0]
        keep_unit = out_aval.ndim > 0 and equation.params.get("dimension") != out_aval.ndim - 1
        held[out], apart[out] = _concat_facts(
            [apart_of(atom) for atom in equation.invars],
            [magnitude_of(atom) for atom in equation.invars], out_aval.dtype, keep_unit=keep_unit)
        return True
    if (depth < 6 and (body := programs.called(equation)) is not None
            and len(body.consts) == len(body.constvars)):
        sub_init = {inner: (magnitude_of(outer), apart_of(outer))
                    for outer, inner in zip(equation.invars, body.invars, strict=True)}
        sub_consts = dict(zip(body.constvars, body.consts, strict=True))
        sub_targets = [atom for atom in body.outvars if isinstance(atom, jcore.Var)]
        sub_held, sub_apart = _proven_walk(
            body.jaxpr, consts=sub_consts, targets=sub_targets, deep=True,
            init=sub_init, track=norm_track.child(equation.invars, body.invars), depth=depth + 1)
        for out_var, body_out in zip(equation.outvars, body.outvars, strict=True):
            if isinstance(body_out, jcore.Literal):
                held[out_var], apart[out_var] = _magnitude(body_out.val), _apart_of(body_out, {})
            else:
                held[out_var] = sub_held.get(body_out)
                apart[out_var] = sub_apart.get(body_out, NOWHERE)
        return True
    return False


def _proven_walk(jaxpr: jcore.Jaxpr, *, consts: Mapping[jcore.Var, object] | None,
                 targets: Sequence[jcore.Var] | None, deep: bool,
                 init: Mapping[jcore.Var, tuple[Magnitude | None, Apart]] | None = None,
                 track: _NormTrack | None = None,
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
    for equation in jaxpr.eqns:
        if needed is not None and not any(var in needed for var in equation.outvars):
            continue
        if deep and len(equation.outvars) == 1:
            made_eq[equation.outvars[0]] = equation
        _, facts = _propagate_facts(equation, laid, magnitude_of, apart_of)
        if facts is not None:
            out = equation.outvars[0]
            bound, proved = facts
            if deep and norm_track is not None:
                bound, proved = _deep_refine(equation, bound, proved, made_eq, magnitude_of, apart_of,
                                             norm_track)
            held[out], apart[out] = bound, proved
            continue
        if deep and norm_track is not None and _deep_nonlang(
                equation, held, apart, magnitude_of, apart_of, norm_track, depth):
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
UNIT = 0


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


class Apart(NamedTuple):
    """What a value is known to hold beside its magnitude, as its dtype holds it: `floor`, a power with
    `2**floor <= |x|` (None where nothing holds it away from zero; a float's alone); `sign`, 1 where
    `x >= 0`, -1 where `x <= 0`, 0 where either may be; `number`, that it is no NaN; and `finite`, that it
    is a finite value its magnitude bound holds of as computed; each proven of it from the program's inputs
    taken as finite reals within their declared ranges.

    A zero of either sign is on both sides: only a floor says a value is none, so a reciprocal's sign is
    read where its operand has one. A sign and a floor are a number's alone, and the bounds read from a
    floor (`_apart_bound`) read an operand's magnitude of a finite one alone. A value that may have
    overflowed is not finite, nor is one an equation makes by keeping or scaling it, a conversion to a
    wider float or to an integer among them, whatever its bound reads there; a value is finite again where
    its operation bounds it whatever its operand is (a boolean, a logistic, a zeroth power) or makes a
    zero of an infinity (a quotient by one, its reciprocal root, its negative power). The bounds that read
    no floor keep their own convention, every value a finite real (`_real_bound`)."""

    floor: int | None = None
    sign: int = 0
    number: bool = False
    finite: bool = False
    unit_norm: bool = False

    @property
    def above(self) -> int | None:
        """Its floor where `x >= 2**floor`, above zero; None where it is not known so."""
        return self.floor if self.sign > 0 else None

    @property
    def positive(self) -> bool:
        """Whether `x >= 2**floor`: above zero."""
        return self.above is not None


NOWHERE = Apart()
"""A value nothing is proven of: one an equation outside the language made, or one with no rule here."""
WHOLE = Apart(number=True, finite=True)
"""A finite number of no known side: an input of no declared interval, an integer or a boolean before its
sign."""


class Proven(NamedTuple):
    """An input's declaration as a program around it proves it (`dew.algebra.program._declared`): its
    magnitude's bound where one holds (a power of two, 0 for nothing; None for none) and what else it holds
    (`Apart`). A window cut from a program takes each value it reads from the program so, not as every
    input is taken, a finite real: a value the program proves no number (a NaN's tanh made outside the
    window) bounds nothing within it (spec alignment, 8993)."""

    bound: float | None
    apart: Apart


def _floor_log2(value: float) -> int:
    """`floor(log2(value))` of a positive float, exactly, by its exponent."""
    return math.frexp(value)[1] - 1


def _held_apart(apart: Apart, dtype) -> Apart:
    """`apart` as `dtype` holds it: a sign and a floor of a number alone; a floor only of a float, and only
    a normal power of it, which a result at or above it rounds to nothing under (rounding to nearest is
    monotone; a subnormal one, or one past the dtype's largest, is dropped); an integer a number whatever
    made it, and a boolean a finite one; a complex value nothing."""
    if jnp.issubdtype(dtype, jnp.integer) or dtype == np.bool_:
        return Apart(None, apart.sign if apart.number else 0, number=True,
                     finite=apart.finite or dtype == np.bool_)
    if not jnp.issubdtype(dtype, jnp.floating):
        return NOWHERE
    if not apart.number:
        return NOWHERE
    floats = jnp.finfo(dtype)
    unit = apart.unit_norm and jnp.dtype(dtype).itemsize >= 2
    floor = (apart.floor if apart.floor is not None and int(floats.minexp) <= apart.floor < int(floats.maxexp)
             else None)
    return apart if (floor == apart.floor and unit == apart.unit_norm) else apart._replace(
        floor=floor, unit_norm=unit)


def _constant_apart(value) -> Apart:
    """A real constant's `Apart`: whether it holds no NaN and no infinity, its sign where its elements share
    one, and a float's least magnitude's power where no element is zero; none of a key's, tracer's, or
    concrete device array's."""
    array = _host_array(value)
    if array is None:
        return NOWHERE
    real = jnp.issubdtype(array.dtype, jnp.floating) or jnp.issubdtype(array.dtype, jnp.integer)
    if not real or not array.size:
        return NOWHERE
    if not jnp.issubdtype(array.dtype, jnp.floating):
        return WHOLE._replace(sign=1 if (array >= 0).all() else -1 if (array <= 0).all() else 0)
    # A float's value is a float64 exactly.
    values = array.astype(np.float64)
    if np.isnan(values).any():
        return NOWHERE
    sign = 1 if (values >= 0).all() else -1 if (values <= 0).all() else 0
    least = float(np.min(np.abs(values)))
    return Apart(_floor_log2(least) if 0 < least < math.inf else None, sign, number=True,
                 finite=bool(np.isfinite(values).all()))


def _declared_apart(declared: float | tuple[float, float] | Proven | None) -> Apart:
    """An input's `Apart`: a finite number, as every input is taken to be (a real, within its declared range
    where it has one); and, of an interval, its end nearer zero where the interval holds none and the side
    of zero it lies on."""
    if declared is None:
        return WHOLE
    if isinstance(declared, Proven):
        return declared.apart
    if not isinstance(declared, tuple):
        return Apart(number=True, finite=math.isfinite(declared))
    low, high = declared
    nearest = low if low > 0 else -high if high < 0 else 0.0
    return Apart(_floor_log2(nearest) if 0 < nearest < math.inf else None,
                 1 if low >= 0 else -1 if high <= 0 else 0, number=True,
                 finite=math.isfinite(low) and math.isfinite(high))


def _within(bound: Magnitude | None, dtype) -> bool:
    """Whether a float of `dtype` bounded by `2**bound` in the reals is finite as computed: its bound is
    known and under the dtype's largest power, which a value under it cannot round past."""
    return bound is not None and (bound == NOTHING or bound < int(jnp.finfo(dtype).maxexp))


def _reduced(equation: jcore.JaxprEqn) -> int:
    """The elements a reduction along axes reads for each it makes."""
    shape = _shape(equation.invars[0])
    return math.prod(shape[axis] for axis in equation.params["axes"])


def _apart(equation: jcore.JaxprEqn, operands: list[Apart], bounds: list[Magnitude | None],
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
        name = symbol(equation.primitive)
        kept = name in KEEPING_SIGN and (
            name != "convert_element_type" or _keeps_sign(equation.invars[0], equation.outvars[0])) and not (
            name.startswith("reduce_") and _reduced(equation) < 1)
        # An integer's bound holds of it where its operands' hold of them: a float that may be infinite
        # converts to a value its bound says nothing of (XLA saturates it), which stays so as a float again.
        proven = all(held.finite for held in operands)
        return _held_apart((operands[0] if kept else WHOLE)._replace(finite=proven), aval.dtype)
    if not operands:
        return NOWHERE
    made = _float_apart(equation, operands, bounds, aval.dtype)
    # Finite where its operands make it so and its own bound is within its dtype; a finite value is a number.
    finite = made.finite and _within(bound, aval.dtype)
    return _held_apart(made._replace(number=made.number or finite, finite=finite), aval.dtype)


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


def _float_apart(equation: jcore.JaxprEqn, operands: list[Apart], bounds: list[Magnitude | None],
                 dtype: DTypeLike) -> Apart:
    """`_apart` of a float output of `dtype` before its dtype holds it (`_held_apart`), `finite` before its
    own bound is read: whether its operands make it finite."""
    name = symbol(equation.primitive)
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
    power = 2 if symbol(equation.primitive) == "square" else equation.params["y"]
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


def _apart_bound(equation: jcore.JaxprEqn, operands: list[Magnitude | None], apart: list[Apart],
                 dtype: DTypeLike) -> Magnitude | None:
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
    name, upper, held = symbol(equation.primitive), operands[0], apart[0]
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


def _bounded(equation: jcore.JaxprEqn, operands: list[Magnitude | None],
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
    real = _real_bound(equation, operands, apart, dtype)
    if not jnp.issubdtype(dtype, jnp.integer):
        return real
    full = _integer_range(dtype)
    name = symbol(equation.primitive)
    source = equation.invars[0].aval
    from_unsigned = isinstance(source, ShapedArray) and jnp.issubdtype(source.dtype, jnp.unsignedinteger)
    wraps = jnp.issubdtype(dtype, jnp.unsignedinteger) and (
        name in ("neg", "sub") or (name == "convert_element_type" and not from_unsigned))
    if name in BITWISE or wraps:
        return full
    return None if real is None else min(real, full)


def _real_bound(equation: jcore.JaxprEqn, operands: list[Magnitude | None], apart: list[Apart],
                dtype: DTypeLike) -> Magnitude | None:
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
    name = symbol(equation.primitive)
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
        return _apart_bound(equation, operands, apart, dtype) if real else None
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


def _log2(value: float) -> int:
    """`ceil(log2(value))`, `NOTHING` for 0: a magnitude's exponent bounding `value` from above, exactly, by
    the float's own exponent (a logarithm rounds, and `log2` of the float under 8 is 3)."""
    if value == 0:
        return NOTHING
    mantissa, exponent = math.frexp(value)
    return exponent - 1 if mantissa == 0.5 else exponent


def _magnitude(value) -> Magnitude | None:
    """A constant's magnitude, its elements' largest (a complex one's modulus); None for one of no number (a
    key's, tracer's, or concrete device array's). A number of any dtype, ml_dtypes' (bfloat16, float8) among
    them, which numpy's type tree omits."""
    array = _host_array(value)
    if array is None:
        return None
    if not (jnp.issubdtype(array.dtype, jnp.number) or array.dtype == bool):
        return None
    # In complex128, which holds every integer's absolute value (int32's least's too) and a complex modulus.
    largest = float(np.max(np.abs(array.astype(np.complex128)))) if array.size else 0.0
    return _log2(largest) if math.isfinite(largest) else None


def _declared(declared: float | tuple[float, float] | Proven | None) -> Magnitude | None:
    """An input's magnitude from its declared range (an interval's larger end's), None where it declares
    none."""
    if isinstance(declared, Proven):
        declared = declared.bound
    if declared is None:
        return None
    largest = max(abs(declared[0]), abs(declared[1])) if isinstance(declared, tuple) else declared
    return _log2(largest) if math.isfinite(largest) else None


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
                                       jnp.log1p(a), jnp.expm1(a),
                                       *(a.astype(narrow).astype(jnp.float32) for narrow in WIDENED)))(scalar)
    found = {}
    for equation in traced.jaxpr.eqns:
        symbol = OPERATIONS[equation.primitive].symbol
        if equation.primitive is jax.lax.convert_element_type_p:
            symbol = f"convert_{np.dtype(equation.params['new_dtype']).name}"
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
    operation = OPERATIONS.get(equation.primitive)
    if operation is None or len(equation.outvars) != 1:
        return None
    return operation.symbol if len(equation.invars) == WRITTEN[operation.kind].arity else None


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


def typed_key(value) -> bool:
    """Whether `value` holds typed PRNG keys (`jax.random.key`), whose bits NumPy cannot read."""
    dtype = getattr(value, "dtype", None)
    return (dtype is not None and not isinstance(dtype, (str, np.dtype))
            and jnp.issubdtype(dtype, jax.dtypes.prng_key))


def held_bits(value) -> tuple[Hashable, np.ndarray]:
    """A constant's dtype and the array of the bits it holds: a typed PRNG key's are its key data
    (`jax.random.key_data`), its dtype the key type itself, which is its implementation (two implementations
    may print one tag); any other value's its own, its dtype by name."""
    if typed_key(value):
        return value.dtype, np.asarray(jax.random.key_data(value))
    array = np.asarray(value)
    return str(array.dtype), array


def _shape(atom) -> tuple[int, ...]:
    """An array value's shape; a value of no shape (a token, a key's) has none the rules read."""
    return tuple(atom.aval.shape) if isinstance(atom.aval, ShapedArray) else ()


def _term[Node: (Term, Skeleton)](equation: jcore.JaxprEqn, read: Callable[..., Node],
                                   param: Callable) -> Node | None:
    """`equation` as a term over its inputs' (`read`), written as its operation's kind is (`WRITTEN`), or
    None where it is outside the language."""
    operation, primitive = _in_language(equation), equation.primitive
    if operation is None:
        return None
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
