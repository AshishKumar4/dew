"""A recurrence's chunk form derived by rules (docs/design/algebra.md §6.9, `linear_scan`): the token update
the declaration states read as an affine map of its state by rules (`parts`), then the token loop over a
chunk rewritten by identities of affine scans into every equal chunk algorithm they reach, and the cheapest
by the work-span model extracted (`derive`).

The token's update and readout enter an e-graph as their jaxprs' algebra (`Token`): the state `h` (keys by
values), the values free of it, broadcast as numpy broadcasts them (a scalar, a vector along the keys or the
values), their products, sums and exponentials, and a vector's contraction with a matrix over the keys
(`read`); a value free of the state whose equation no constructor states is a leaf. The rules are identities
of that algebra: products and sums commute and associate, products distribute over sums (and factor back), a
broadcast distributes over a product or a sum, and a contraction is linear in each operand, a scalar or a
key's weight passing through it. Where the update's class holds a decayed state plus a rank-one input along
`B` whose values `U` may read the decayed state once more (`U + X^T D h`), each part free of the state, it
is affine (`Affine.decayed`, `Affine.corrected`): `h_t = A_t h_{t-1} + B_t U_t^T` with the transition
`A_t = (I - s_t B_t C_t^T) D_t`, `D_t = exp(z_t)` (or `exp2`) a scalar's or each key's, `C = -X` and `s = 1`
where a token corrects the state; and a readout `o_t = Q_t^T h_t + R_t` (`Affine.readout`). Any other update
or readout reaches no such form and is refused.

The affine token update enters a second e-graph as its chunk's token loop, `Chunk.loop`, which runs its
tokens in turn. Each identity holds of every chunk, its side condition a fact of the declaration:

- `wy` (the compact WY form): where tokens correct the state, the loop of corrected transitions is the loop
  of plain ones over the corrected inputs `W`, which solve the unit lower-triangular system
  `(I + diag(s) strictly_lower(gram(C, B))) W = U - diag(s) (C * gamma) h_in`, a correction of token `i`
  reading each earlier token `j`'s write decayed between them;
- `dual` (the state-space dual): the loop of plain transitions is its outputs
  `(Q * gamma) h_in + lower(gram(Q, B)) U + R` and its state
  `gamma_last h_in + sum_t (B_t * exp(G_last - G_t)) U_t^T`, its decays the exponentials of its exponents'
  prefix sums `G` (`gamma = exp(G)`, and each pair's `exp(G_i - G_j)` inside `gram`);
- `solve`: a unit lower-triangular solve is the given values plus the inverse's strictly lower part applied
  (its inverse by doubling, `doubled`), or a forward substitution (`substituted`), the two side by side;
- `linear`: a solve of a difference whose second part is a product with the entering state is the
  difference of two solves, the second a solve of the product's left operand, so a solve reads no state.

Every node's work (`templates.cost.Work`) is priced at the chunk's extents, its span its dependent steps (the
loop's tokens, a substitution's rows), and a candidate costs the work-span model's time
(`dew.algebra.cost.fused`: the larger of its work at its units' rates and its critical path's steps at a
dependent step's latency, `GpuTarget.step_seconds`). The e-graph holds every candidate the identities reach;
each is drawn from it whole (`EGraph.extract_multiple`), priced with its shared nodes counted once and its
critical path summed along its chains, and the cheapest is the chunk's form, which `scan` evaluates.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import operator
from collections.abc import Callable, Iterator

import jax
import jax.numpy as jnp
import numpy as np
from egglog import (
    DagCostModel,
    EGraph,
    Expr,
    StringLike,
    birewrite,
    eq,
    f64,
    f64Like,
    function,
    get_callable_args,
    i64,
    i64Like,
    method,
    relation,
    rewrite,
    rule,
    ruleset,
    set_,
    union,
    vars_,
)
from jax.extend import core as jcore

from dew.algebra import cost, extract, terms
from dew.algebra.cost import Work
from dew.algebra.forms import Program, Refused, aval


class Token(Expr):
    """A value of a token's update or readout over the state (keys by values), as numpy broadcasts it: the
    state, a value of the traced programs free of it, a scalar, a vector along the keys (`column`, a
    `(keys, 1)` value) or along the values (`row`), and the algebra of their equations; and the affine map's
    forms the rules reach (`affine`, `corrected`, `readout`)."""

    @classmethod
    def state(cls) -> Token: ...

    @classmethod
    def value(cls, index: i64Like) -> Token: ...
    """The traced programs' value `index` (`Parts.atoms`), free of the state: an input, a constant, or an
    equation's whose algebra no constructor states."""

    @classmethod
    def literal(cls, value: f64Like) -> Token: ...

    @classmethod
    def zero(cls) -> Token: ...

    @classmethod
    def scalar(cls, value: Token) -> Token: ...
    """A scalar, broadcast to any shape."""

    @classmethod
    def column(cls, vector: Token) -> Token: ...

    @classmethod
    def row(cls, vector: Token) -> Token: ...

    @method(cost=1)
    def times(self, other: Token) -> Token: ...

    @method(cost=1)
    def plus(self, other: Token) -> Token: ...

    @classmethod
    def read(cls, vector: Token, matrix: Token) -> Token: ...
    """`vector^T matrix`, the keys contracted: a vector along the values."""

    @classmethod
    def exp(cls, value: Token) -> Token: ...

    @classmethod
    def exp2(cls, value: Token) -> Token: ...



class Affine(Expr):
    """A token's update or readout as an affine map of the state, over its parts free of it (`form_of`)."""

    @classmethod
    def decayed(cls, base2: i64Like, per_key: i64Like, exponent: Token, left: Token,
                right: Token) -> Affine: ...
    """`D h + left right^T`, `D` the decay `exp(exponent)` (`exp2` where `base2`), a scalar's or each key's
    (`per_key`)."""

    @classmethod
    def corrected(cls, base2: i64Like, per_key: i64Like, exponent: Token, left: Token, right: Token,
                  correction: Token) -> Affine: ...
    """`D h + left (right + correction^T (D h))^T`: an input and the corrections along one vector."""

    @classmethod
    def readout(cls, reader: Token, free: Token) -> Affine: ...
    """`reader^T h + free`."""


@function(cost=2**40)
def form_of(value: Token) -> Affine: ...
"""A value's affine forms, each equal to it: priced past any of them, so a value with one is drawn as it."""


free = relation("free", Token)
"""A value that reads no state."""

_MINUS_ONE = Token.scalar(Token.literal(-1.0))


def _token_rules():
    """The algebra's identities, freeness, and the affine forms (the module's account)."""
    a, b, c, m, x, z, t = vars_("a b c m x z t", Token)
    (index,), (number,) = vars_("index", i64), vars_("number", f64)
    h = Token.state()
    found = [
        birewrite(a.times(b)).to(b.times(a)),
        birewrite(a.times(b).times(c)).to(a.times(b.times(c))),
        birewrite(a.plus(b)).to(b.plus(a)),
        birewrite(a.plus(b).plus(c)).to(a.plus(b.plus(c))),
        birewrite(a.times(b.plus(c))).to(a.times(b).plus(a.times(c))),
        birewrite(Token.read(c, a.plus(b))).to(Token.read(c, a).plus(Token.read(c, b))),
        birewrite(Token.read(a.plus(b), m)).to(Token.read(a, m).plus(Token.read(b, m))),
        birewrite(Token.read(c, Token.scalar(x).times(m))).to(Token.scalar(x).times(Token.read(c, m))),
        birewrite(Token.read(Token.scalar(x).times(c), m)).to(Token.scalar(x).times(Token.read(c, m))),
        birewrite(Token.read(c, Token.column(a).times(m))).to(Token.read(c.times(a), m)),
        birewrite(Token.read(c, Token.row(a).times(m))).to(a.times(Token.read(c, m))),
    ]
    for wrap in (Token.scalar, Token.column, Token.row):
        found += [birewrite(wrap(a.times(b))).to(wrap(a).times(wrap(b))),
                  birewrite(wrap(a.plus(b))).to(wrap(a).plus(wrap(b))),
                  birewrite(Token.exp(wrap(a))).to(wrap(Token.exp(a))),
                  birewrite(Token.exp2(wrap(a))).to(wrap(Token.exp2(a)))]
    found += [birewrite(wrap(Token.scalar(x))).to(Token.scalar(x)) for wrap in (Token.column, Token.row)]
    found += [rule(eq(t).to(leaf)).then(free(t))
              for leaf in (Token.value(index), Token.literal(number), Token.zero())]
    found += [rule(eq(t).to(made(a)), free(a)).then(free(t))
              for made in (Token.scalar, Token.column, Token.row, Token.exp, Token.exp2)]
    found += [rule(eq(t).to(made), free(a), free(b)).then(free(t))
              for made in (a.times(b), a.plus(b), Token.read(a, b))]
    for base2, power in ((0, Token.exp), (1, Token.exp2)):
        for per_key, wrap in ((0, Token.scalar), (1, Token.column)):
            decayed = wrap(power(z)).times(h)
            found += [
                rule(eq(t).to(decayed.plus(Token.column(a).times(Token.row(b)))),
                     free(z), free(a), free(b)).then(
                    union(form_of(t)).with_(Affine.decayed(base2, per_key, z, a, b))),
                rule(eq(t).to(decayed.plus(Token.column(a).times(Token.row(b.plus(Token.read(c, decayed)))))),
                     free(z), free(a), free(b), free(c)).then(
                    union(form_of(t)).with_(Affine.corrected(base2, per_key, z, a, b, c))),
            ]
    found += [rule(eq(t).to(Token.read(a, h)), free(a)).then(
                  union(form_of(t)).with_(Affine.readout(a, Token.zero()))),
              rule(eq(t).to(Token.read(a, h).plus(b)), free(a), free(b)).then(
                  union(form_of(t)).with_(Affine.readout(a, b)))]
    return ruleset(*found, name="token_affine")


@dataclasses.dataclass(frozen=True)
class Parts:
    """A token's update and readout as the affine map the rules reach (`parts`): the decay `exp(exponent)`
    (`exp2` where `base2`), a scalar's or each key's (`per_key`), on the state; the input `left right^T`;
    the corrections' sum `left (correction^T (D h))^T`, None where none corrects; the readout
    `reader^T h + free`. Each a term free of the state over the traced programs' values `atoms` (each its
    program and its atom, at `Token.value`'s place)."""

    update: Program
    readout: Program
    atoms: tuple
    base2: bool
    per_key: bool
    exponent: Token
    left: Token
    right: Token
    correction: Token | None
    reader: Token
    free: Token


def parts(update: Program, readout: Program) -> Parts:
    """The token's update and readout (each a traced program whose first input is the state) as an affine
    map of the state (`Parts`, the module's account); refused where either reaches no affine form, or reads
    the state through an equation whose algebra no constructor states."""
    atoms: list = []
    made, read = _translated(update, atoms), _translated(readout, atoms)
    graph = EGraph()
    graph.register(made, read)
    graph.run(_token_rules().saturate())
    made, read = graph.extract(form_of(made)), graph.extract(form_of(read))
    corrected = get_callable_args(made, Affine.corrected)
    args = corrected if corrected is not None else get_callable_args(made, Affine.decayed)
    if args is None:
        raise Refused("the update is not a decayed state, exp(z) a scalar's or each key's, plus a rank-one "
                      "input and corrections of the decayed state along the input's vector")
    readout_args = get_callable_args(read, Affine.readout)
    if readout_args is None:
        raise Refused("the readout is not the state contracted with a vector over its keys plus a part free "
                      "of it")
    base2, per_key, exponent, left, right, *correction = args
    return Parts(update, readout, tuple(atoms), bool(terms.integer(base2)),
                 bool(terms.integer(per_key)), exponent, left, right,
                 correction[0] if correction else None, *readout_args)


_KEEPING = frozenset({"copy", "copy_p"})
"""Equations that keep a value's elements, dtype and shape."""


def _translated(traced: Program, atoms: list) -> Token:
    """`traced`'s output (its first input the state) as a term (`Token`): each equation its constructor's,
    a value free of the state whose equation has none a leaf (`Token.value`, its place in `atoms`); refused
    where the state is read through such an equation."""
    jaxpr = traced.jaxpr
    held: dict = {jaxpr.invars[0]: Token.state()}
    reads = {jaxpr.invars[0]}

    def leaf(atom, shape: tuple) -> Token:
        atoms.append((traced, atom))
        value = Token.value(len(atoms) - 1)
        return Token.scalar(value) if shape == () else value

    def term(atom) -> Token:
        # A literal is a leaf of its own at each use, read at its dtype (`Parts.atoms`): a float64 1e100 is no
        # float32's.
        if isinstance(atom, jcore.Literal):
            return leaf(atom, np.shape(atom.val))
        if atom not in held:
            held[atom] = leaf(atom, aval(atom).shape)
        return held[atom]

    for equation in jaxpr.eqns:
        reading = any(isinstance(atom, jcore.Var) and atom in reads for atom in equation.invars)
        made = (_algebra(equation, [term(atom) for atom in equation.invars])
                if len(equation.outvars) == 1 else None)
        if made is None and reading:
            raise Refused(f"the state read through {equation.primitive.name}, which no chunk form here holds")
        if made is not None:
            held[equation.outvars[0]] = made
        if reading:
            reads.update(equation.outvars)
    return term(jaxpr.outvars[0])


def _algebra(equation: jcore.JaxprEqn, operands: list[Token]) -> Token | None:
    """One equation's output as its constructor states it over its operands' terms, None where none does."""
    name = equation.primitive.name
    shapes = [np.shape(atom.val) if isinstance(atom, jcore.Literal) else aval(atom).shape
              for atom in equation.invars]
    after = aval(equation.outvars[0]).shape
    if name in _KEEPING:
        return operands[0]
    if name == "convert_element_type":
        # A float widened keeps every value, so its term is its operand's; any other conversion rounds (an
        # integer's truncation), so its output is no term here: a leaf where it reads no state, refused where
        # it does.
        before, after = aval(equation.invars[0]).dtype, jnp.dtype(equation.params["new_dtype"])
        widened = (jnp.issubdtype(before, jnp.floating) and jnp.issubdtype(after, jnp.floating)
                   and jnp.promote_types(before, after) == after)
        return operands[0] if widened else None
    if name in ("mul", "add"):
        return operands[0].times(operands[1]) if name == "mul" else operands[0].plus(operands[1])
    if name == "sub":
        return operands[0].plus(_MINUS_ONE.times(operands[1]))
    if name == "neg":
        return _MINUS_ONE.times(operands[0])
    if name in ("exp", "exp2"):
        return (Token.exp if name == "exp" else Token.exp2)(operands[0])
    if name in ("broadcast_in_dim", "reshape"):
        return _laid(operands[0], shapes[0], after, equation.params.get("broadcast_dimensions"))
    if name == "dot_general":
        (contracted, batch) = equation.params["dimension_numbers"]
        if tuple(map(tuple, batch)) != ((), ()) or tuple(map(tuple, contracted)) != ((0,), (0,)):
            return None
        if [len(shape) for shape in shapes] == [1, 2]:
            return Token.read(operands[0], operands[1])
        if [len(shape) for shape in shapes] == [2, 1]:
            return Token.read(operands[1], operands[0])
    return None


def _laid(operand: Token, before: tuple, after: tuple, dimensions) -> Token | None:
    """A value broadcast (along `dimensions`) or reshaped from `before` to `after`: a scalar as itself, a
    vector to a column of the keys or a row of the values, a value of its own shape or a matrix's unit axes
    broadcast as itself; None for any other."""
    if len(before) == 0 or tuple(before) == tuple(after) or (len(before) == len(after) == 2 and
                                                               tuple(dimensions or ()) == (0, 1)):
        return operand
    if len(before) != 1 or len(after) != 2:
        return None
    if dimensions is None:
        # A reshape of a vector to a matrix: its axis is the one of its extent.
        dimensions = {(before[0], 1): (0,), (1, before[0]): (1,)}.get(tuple(after))
        if dimensions is None or before[0] == 1:
            return None
    if len(dimensions) != 1:
        return None
    return (Token.column, Token.row)[dimensions[0]](operand)


_APPLIED: tuple[tuple[Callable, Callable], ...] = (
    (Token.scalar, lambda value: value),
    (Token.column, lambda value: jnp.reshape(value, (-1, 1)) if jnp.ndim(value) == 1 else value),
    (Token.row, lambda value: jnp.reshape(value, (1, -1)) if jnp.ndim(value) == 1 else value),
    (Token.times, operator.mul),
    (Token.plus, operator.add),
    (Token.exp, jnp.exp),
    (Token.exp2, jnp.exp2),
)
"""Each elementwise or broadcasting constructor of a free value as numpy computes it."""


def _read(vector, matrix, keys: int) -> jax.Array:
    """`vector^T matrix` over the state's `keys`, each operand broadcast along them (a scalar reader reads
    every key alike, a row is every key's)."""
    product = jnp.reshape(vector, (-1, 1)) * matrix
    return jnp.sum(jnp.broadcast_to(product, (keys, jnp.shape(product)[-1])), axis=0)


def value_of(term: Token, read: Callable[[int], jax.Array], keys: int) -> jax.Array:
    """A free term's value over a state of `keys` keys, its leaves `read(index)` (`Token.value`'s)."""
    if (args := get_callable_args(term, Token.value)) is not None:
        return read(terms.integer(args[0]))
    if (args := get_callable_args(term, Token.literal)) is not None:
        return jnp.asarray(terms.real(args[0]), jnp.float32)
    if get_callable_args(term, Token.zero) is not None:
        return jnp.zeros((), jnp.float32)
    if (args := get_callable_args(term, Token.read)) is not None:
        vector, matrix = args
        return _read(value_of(vector, read, keys), value_of(matrix, read, keys), keys)
    for constructor, applied in _APPLIED:
        if (args := get_callable_args(term, constructor)) is not None:
            return applied(*(value_of(arg, read, keys) for arg in args))
    raise Refused("a part of the token's affine map that reads the state")


class Chunk(Expr):
    """One value of a chunk's computation: a token value along the chunk, the entering state, the chunk's
    decays, a loop, or the linear algebra of a chunk form."""

    @classmethod
    def token(cls, name: StringLike) -> Chunk: ...
    """A token value along the chunk (`scan._token`'s): "B", "U", "s", "C", "Q" or "R"."""

    @classmethod
    def entering(cls) -> Chunk: ...
    """The state entering the chunk."""

    @classmethod
    def decay(cls, part: StringLike) -> Chunk: ...
    """The chunk's decays (`scan._Decays`): "entering" `gamma`, "to_last" `exp(G_last - G_t)`, "whole"
    `gamma_last`, "between" each pair's."""

    @classmethod
    def loop(cls, values: Chunk) -> Chunk: ...
    """The token loop the declaration states over the chunk, its inputs' values `values`, each token's
    transition corrected: its outputs and the state it leaves (`pair`)."""

    @classmethod
    def plain(cls, values: Chunk) -> Chunk: ...
    """The token loop with no token correcting the state, its inputs' values `values`."""

    @classmethod
    def pair(cls, outputs: Chunk, state: Chunk) -> Chunk: ...

    @method(cost=1)
    def outputs(self) -> Chunk: ...

    @method(cost=1)
    def state(self) -> Chunk: ...

    @classmethod
    def gram(cls, left: Chunk, right: Chunk, strict: i64Like) -> Chunk: ...
    """Each pair's decayed contraction `sum_k left_ik right_jk between_ijk`, `j < i` where `strict`, else
    `j <= i` (`scan._paired`)."""

    @method(cost=1)
    def rows(self, scale: Chunk) -> Chunk: ...
    """`diag(scale) @ self`."""

    @method(cost=1)
    def times(self, other: Chunk) -> Chunk: ...
    """Elementwise, a token vector times a per-token decay along its keys."""

    @method(cost=1)
    def matmul(self, other: Chunk) -> Chunk: ...

    @method(cost=1)
    def plus(self, other: Chunk) -> Chunk: ...

    @method(cost=1)
    def minus(self, other: Chunk) -> Chunk: ...

    @classmethod
    def decayed(cls, state: Chunk) -> Chunk: ...
    """The entering state decayed across the whole chunk, `gamma_last * h` along its keys."""

    @classmethod
    def written(cls, keys: Chunk, values: Chunk) -> Chunk: ...
    """`sum_t (keys_t * exp(G_last - G_t)) values_t^T`."""

    @classmethod
    def solve(cls, lower: Chunk, given: Chunk) -> Chunk: ...
    """`(I + lower)^-1 given`, `lower` strictly lower-triangular."""

    @classmethod
    def doubled(cls, lower: Chunk) -> Chunk: ...
    """`(I + lower)^-1 - I` by doubling (`scan._lowered`)."""

    @classmethod
    def substituted(cls, lower: Chunk, given: Chunk) -> Chunk: ...
    """`(I + lower)^-1 given` by forward substitution, a row a step."""


def _identities(corrects: bool):
    """The identities of the module's account, `wy` where tokens correct the state."""
    values = Chunk.token("U")
    given, other, lower, left, right = vars_("given other lower left right", Chunk)
    h = Chunk.entering()
    found = []
    if corrects:
        system = Chunk.gram(Chunk.token("C"), Chunk.token("B"), 1).rows(Chunk.token("s"))
        corrected = Chunk.solve(system, values.minus(
            Chunk.token("C").times(Chunk.decay("entering")).rows(Chunk.token("s")).matmul(h)))
        found.append(rewrite(Chunk.loop(values)).to(Chunk.plain(corrected)))
    found += [
        rewrite(Chunk.plain(given)).to(Chunk.pair(
            Chunk.token("Q").times(Chunk.decay("entering")).matmul(h).plus(
                Chunk.gram(Chunk.token("Q"), Chunk.token("B"), 0).matmul(given)).plus(Chunk.token("R")),
            Chunk.decayed(h).plus(Chunk.written(Chunk.token("B"), given)))),
        rewrite(Chunk.pair(left, right).outputs()).to(left),
        rewrite(Chunk.pair(left, right).state()).to(right),
        rewrite(Chunk.solve(lower, given)).to(given.plus(Chunk.doubled(lower).matmul(given))),
        rewrite(Chunk.solve(lower, given)).to(Chunk.substituted(lower, given)),
        rewrite(Chunk.solve(lower, given.minus(other.matmul(h)))).to(
            Chunk.solve(lower, given).minus(Chunk.solve(lower, other).matmul(h))),
    ]
    return ruleset(*found, name="linear_scan")


@dataclasses.dataclass(frozen=True)
class Extents:
    """A chunk's extents: its tokens, the state's keys and values, the decays' keys (1 for a scalar's), and
    whether a product's operands are fp32 (TF32 at the default precision) or narrower."""

    tokens: int
    keys: int
    values: int
    decay_keys: int


@dataclasses.dataclass(frozen=True)
class Form:
    """A chunk's extracted form: its outputs' and its state's terms, and its price."""

    outputs: Chunk
    state: Chunk
    price: cost.Seconds


@function(merge=lambda old, new: old)
def rows_of(value: Chunk) -> i64: ...
"""A value's rows at the chunk's extents (each value a matrix: a vector a column)."""


@function(merge=lambda old, new: old)
def columns_of(value: Chunk) -> i64: ...
"""A value's columns at the chunk's extents."""


def _shapes(extents: Extents):
    """Each value's shape (`rows_of`, `columns_of`) from its operands', as the e-graph grows."""
    a, b, t = vars_("a b t", Chunk)
    r, c = vars_("r c", i64)
    tokens, keys, values = i64(extents.tokens), i64(extents.keys), i64(extents.values)

    def fixed(made, rows, columns):
        return rule(eq(t).to(made)).then(set_(rows_of(t)).to(rows), set_(columns_of(t)).to(columns))

    found = [rule(eq(t).to(a.matmul(b)), eq(r).to(rows_of(a)), eq(c).to(columns_of(b))).then(
        set_(rows_of(t)).to(r), set_(columns_of(t)).to(c))]
    # A value of its first operand's shape: an elementwise one, a row scaling, a solve's of its given values.
    found += [rule(eq(t).to(made), eq(r).to(rows_of(a)), eq(c).to(columns_of(a))).then(
        set_(rows_of(t)).to(r), set_(columns_of(t)).to(c))
        for made in (a.rows(b), a.times(b), a.plus(b), a.minus(b), Chunk.solve(b, a),
                     Chunk.substituted(b, a))]
    found += [fixed(made, tokens, tokens)
              for made in (Chunk.gram(a, b, 0), Chunk.gram(a, b, 1), Chunk.doubled(a))]
    found += [fixed(made, keys, values) for made in (Chunk.decayed(a), Chunk.written(a, b), a.state())]
    found.append(fixed(a.outputs(), tokens, values))
    return ruleset(*found, name="chunk_shapes")


def _leaves(extents: Extents) -> list:
    """The leaves' shapes, stated before saturation."""
    c, k, v, d = extents.tokens, extents.keys, extents.values, extents.decay_keys
    shapes = {Chunk.token("B"): (c, k), Chunk.token("C"): (c, k), Chunk.token("Q"): (c, k),
              Chunk.token("U"): (c, v), Chunk.token("R"): (c, v), Chunk.token("s"): (c, 1),
              Chunk.entering(): (k, v), Chunk.decay("entering"): (c, d), Chunk.decay("to_last"): (c, d),
              Chunk.decay("whole"): (d, 1), Chunk.decay("between"): (c, c)}
    return [fact for value, (rows, columns) in shapes.items()
            for fact in (set_(rows_of(value)).to(i64(rows)), set_(columns_of(value)).to(i64(columns)))]


def derive(extents: Extents, *, corrects: bool, schedule: cost.Schedule, checked: bool = False) -> Form:
    """The chunk's cheapest form (the module's account) under `schedule`: the e-graph saturated under the
    identities, extracted by each node's work (`_work`; a solve, which no node computes until an identity
    rewrites it, costs past every form), and the extracted form priced exactly, its shared nodes once and
    its critical path along its chains (`_priced`)."""
    graph = EGraph()
    values = Chunk.token("U")
    loop = Chunk.loop(values) if corrects else Chunk.plain(values)
    root = Chunk.pair(loop.outputs(), loop.state())
    graph.register(root, *_leaves(extents))
    graph.run((_identities(corrects) | _shapes(extents)).saturate())

    def marginal(egraph: EGraph, node) -> cost.Seconds:
        if get_callable_args(node, Chunk.solve) is not None:
            return cost.Seconds(Work(), schedule, math.inf)
        return cost.Seconds(_work(node, extents, lambda child: _shape_of(egraph, child)), schedule)

    model = DagCostModel(marginal_cost=marginal, identity=cost.Seconds(Work(), schedule))
    extracted, _ = extract.cheapest(graph, root, model, checked=checked)
    # The pair of its two parts, or one algorithm that gives both (the token loop, where the target prices
    # its steps past every other form's work), read by its projections.
    parts = get_callable_args(extracted, Chunk.pair)
    outputs, state = parts if parts is not None else (extracted.outputs(), extracted.state())
    shape_of = functools.partial(_shape_of, graph)
    return Form(outputs, state, cost.Seconds(_priced((outputs, state), extents, shape_of), schedule))


def _shape_of(graph: EGraph, value: Chunk) -> tuple[int, int]:
    """A value's shape from the e-graph's facts (`rows_of`, `columns_of`)."""
    rows, columns = (graph.lookup_function_value(fact(value)) for fact in (rows_of, columns_of))
    return (terms.integer(rows) if rows is not None else 0,
            terms.integer(columns) if columns is not None else 0)


def _priced(roots: tuple[Chunk, ...], extents: Extents, shape_of) -> Work:
    """The candidate's work, each distinct node once (`terms`' keys), and its span, its critical path's
    dependent steps, each node's own along its longest chain of operands."""
    work, spans = Work(), {}
    for node in _nodes(roots):
        own = _work(node, extents, shape_of)
        work = work + dataclasses.replace(own, span=0.0)
        spans[terms.key(node)] = own.span + max((spans[terms.key(child)] for child in _children(node)),
                                                default=0.0)
    return dataclasses.replace(work, span=max(spans[terms.key(root)] for root in roots))


def _nodes(roots: tuple[Chunk, ...]) -> Iterator[Chunk]:
    """Each distinct node under `roots`, children first."""
    seen: set = set()

    def visit(node: Chunk) -> Iterator[Chunk]:
        if terms.key(node) in seen:
            return
        seen.add(terms.key(node))
        for child in _children(node):
            yield from visit(child)
        yield node

    for root in roots:
        yield from visit(root)


def _children(node: Chunk) -> list[Chunk]:
    for constructor in CONSTRUCTORS:
        if (args := get_callable_args(node, constructor)) is not None:
            return [arg for arg in args if isinstance(arg, Chunk)]
    return []


CONSTRUCTORS = (Chunk.token, Chunk.entering, Chunk.decay, Chunk.loop, Chunk.plain, Chunk.pair, Chunk.outputs,
                Chunk.state, Chunk.gram, Chunk.rows, Chunk.times, Chunk.matmul, Chunk.plus, Chunk.minus,
                Chunk.decayed, Chunk.written, Chunk.solve, Chunk.doubled, Chunk.substituted)
"""Every constructor of the chunk's language, as `get_callable_args` matches a node."""


def _work(node: Chunk, extents: Extents, shape_of) -> Work:
    """A node's own work and dependent steps at the chunk's extents: products as fp32 products at the default
    precision (`cost._mode`), elementwise work on the CUDA cores, exponentials on the special-function units,
    a loop's and a substitution's rows a step each."""
    c, k, v, d = extents.tokens, extents.keys, extents.values, extents.decay_keys
    product = cost.product_work
    if get_callable_args(node, Chunk.loop) is not None or get_callable_args(node, Chunk.plain) is not None:
        corrected = get_callable_args(node, Chunk.loop) is not None
        # A token's step: the state decayed, read by its correction (a contraction over keys) and corrected,
        # written, and read out, each over the state's keys and values; its dependent steps a reduction and
        # the update that reads it (two launches a token on XLA's path).
        per_token = k * v * (6.0 if corrected else 3.0) + 2.0 * k * v
        return Work(operations=c * per_token, exponentials=c * d, span=2.0 * c)
    if (args := get_callable_args(node, Chunk.decay)) is not None:
        # The compensated prefix (`scan._decays`), its pairs' exponentials.
        part = terms.text(args[0])
        elements = {"entering": c * d, "to_last": c * d, "whole": d, "between": c * c * d}[part]
        return Work(operations=12.0 * c * d + elements, exponentials=elements)
    if (args := get_callable_args(node, Chunk.gram)) is not None:
        if d == 1:
            return product(c, k, c) + Work(operations=2.0 * c * c)
        return Work(operations=3.0 * c * c * k)
    if get_callable_args(node, Chunk.doubled) is not None:
        # log2(C) levels, each squaring the power and applying it: two products of C by C.
        levels = max(math.ceil(math.log2(max(c, 2))), 1)
        return product(c, c, c, count=2 * levels) + Work(operations=levels * c * c)
    if (args := get_callable_args(node, Chunk.substituted)) is not None:
        # A row a step: its contraction with the rows before it, then its row set (two launches a row).
        columns = shape_of(args[1])[-1]
        return Work(operations=2.0 * c * c * columns, span=2.0 * c)
    if (args := get_callable_args(node, Chunk.matmul)) is not None:
        left, right = shape_of(args[0]), shape_of(args[1])
        return product(left[0], left[-1], right[-1])
    if get_callable_args(node, Chunk.written) is not None:
        return product(k, c, v) + Work(operations=c * k)
    if get_callable_args(node, Chunk.decayed) is not None:
        return Work(operations=k * v)
    for constructor in (Chunk.rows, Chunk.times, Chunk.plus, Chunk.minus):
        if (args := get_callable_args(node, constructor)) is not None:
            return Work(operations=math.prod(shape_of(args[0])))
    return Work()


def reads_state(node: Chunk) -> bool:
    """Whether `node` reads the entering state."""
    return any(get_callable_args(each, constructor) is not None for each in _nodes((node,))
               for constructor in (Chunk.entering, Chunk.loop, Chunk.plain))


def prepared(roots: tuple[Chunk, ...]) -> list[Chunk]:
    """The nodes under `roots` a chunk's walk prepares before its carry, as no equation of the program's
    language computes them: each solve's inverse (`doubled`) and each substitution that reads no state."""
    return [node for node in _nodes(roots)
            if get_callable_args(node, Chunk.doubled) is not None
            or (get_callable_args(node, Chunk.substituted) is not None and not reads_state(node))]


def inverse(node: Chunk) -> bool:
    """Whether `node` is a solve's inverse (`doubled`), which the solve's entries' bound bounds."""
    return get_callable_args(node, Chunk.doubled) is not None


# egglog keeps a language's defining frame, and through it every importer's, until its declarations are
# first read (`dew/__init__` imports egglog on its own frame for its builtins'): read here, at import, they
# hold none of the locals of a call that compiles a process's first program through the algebra pass, which
# imports this module and never derives a scan.
_DECLARATIONS = tuple(map(terms.declarations, (Token, Affine, Chunk)))
