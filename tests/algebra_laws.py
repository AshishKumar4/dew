"""Law certificates of a derived algebra (docs/design/algebra.md §6.9): the laws every mainloop's schedule
relies on, proven over the reals with SymPy for a declaration's derived forms before any form is emitted.

For forms `init`, `step`, `merge` and `finish` (`dew.algebra.forms.Derived`) over states `a`, `b`, `c` and
tiles `t`, `u`:

- `identity`: `merge(init, a) = a = merge(a, init)`;
- `associative`: `merge(merge(a, b), c) = merge(a, merge(b, c))`;
- `consistent`: `step(a, t) = merge(a, step(init, t))`, a tile folded into a state as a walk of it merged;
- `declared`: `finish(step(step(init, t), u)) = f(t ++ u)`, the walk computing the declared function `f`
  over the tiles joined along the walked axis.

Together they prove schedule independence: by `consistent` and `associative` every walk of the same tiles,
in any chunking, merged in any tree (a split walk's fix-up, a split decode's combine), reaches the one state,
and by `declared` its finish is the function; `finish` commuting with any split follows, as it is applied to
equal states. A metamorphic test then only samples what this proves.

The proof is over the reals: each form traced (`jax.make_jaxpr`) and its program evaluated over SymPy
symbols, every element its own symbol, a dtype's largest magnitude read as a finite value below every other
(`LEAST`: the empty state's anchor, whose weights vanish); each comparison the sides hold (a mask's position
against a key's) decided each way in turn (`_cases`), each maximum and minimum by each order of the values it
compares (`_orderings`), and the two sides' difference zero as a rational function of exponentials
(`_zero`). A certificate is a reduced-tile proof: the laws hold of the forms at the small tile they were
proven at. A kernel's tile shares it where its forms' structure (`skeleton`) is that tile's and its extents
are equal and unequal as that tile's are (tests/test_algebra_laws.py), so a broadcast or reshape that confuses
two axes shows the same at both. That a law proven at the small tile holds at a kernel's is not proven: a
form whose reductions treat their elements alike holds its laws at any extent in every case the suite has
seen, but a form that held them at a few elements and broke them at more would share the certificate."""

import dataclasses
import functools
import itertools
import math
from collections.abc import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import sympy
from egglog import declarations
from jax.extend import core as jcore

from dew.algebra import SATURATION, derive, forms
from dew.nn import programs

LAWS = ("identity", "associative", "consistent", "declared")
"""The laws a certificate proves, in the order it proves them."""


@dataclasses.dataclass(frozen=True)
class Certificate:
    """A derived algebra's laws, proven over the reals (`LAWS`), at the tile `shapes` (by operand: its shape
    and dtype), with the side conditions the proof read (`conditions`)."""

    laws: tuple[str, ...]
    shapes: tuple
    conditions: tuple[str, ...]


class Unlawful(ValueError):
    """A derived algebra whose forms break a law: the law (`law`) and the elements that differ."""

    def __init__(self, law: str, detail: str):
        super().__init__(f"the derived forms break {law}: {detail}")
        self.law = law


EMPTY_ROWS = ("a row that sees no key, where the declared function has no value (its softmax 0/0), is no "
              "case of a law: the walk's own value there is its template's")
"""The rows a certificate's laws read nothing of (`_cases`)."""

EMPTY_CONDITION = ("a float's largest magnitude is read as a value past every other (`LEAST`): an anchor at "
                   "a dtype's least weighs its unseen logits zero, as its exponentials underflow")


def reduced(shapes: Mapping, walked: Mapping[str, int]) -> dict:
    """The tile a certificate is proven at: extent 2 along each operand's walked axis (`walked`, by name),
    1 along every other, each dtype as given."""
    return {name: (tuple(2 if axis == walked.get(name) else 1 for axis in range(len(shape))), dtype)
            for name, (shape, dtype) in shapes.items()}


def certify(*, init: Callable, step: Callable, merge: Callable, finish: Callable, declared: Callable,
            shapes: Mapping, walked: Mapping[str, int], mapped: Callable | None = None) -> Certificate:
    """`LAWS` proven for forms `init(shapes)`, `step(state, tiles)`, `merge(a, b, shapes=)` and
    `finish(state, tiles, shapes=)` at `shapes` (`reduced`'s), `declared(**operands)` the function over the
    whole operands, each traced at its operands' own dtypes; `Unlawful` where a law fails. An output along the
    walked axis, which `finish` gives as None, is each tile's own (`mapped(tiles)`, `Derived.mapped`): the
    tiles' joined along it are the declared function's."""
    empty = init(shapes)
    tile_types = {name: jax.ShapeDtypeStruct(shape, dtype) for name, (shape, dtype) in shapes.items()}
    if not empty:
        # A declaration that carries no state walks nothing: each tile's forms are its function's (a tile
        # declaration's, its positions the tile's own), and the walk's laws hold of no state.
        tile = _symbolic_tiles(shapes, "t0")
        _equal("declared", _outputs(_evaluate(mapped, (tile_types,), (tile,))),
               _outputs(_evaluate(lambda given: declared(**given), (tile_types,), (tile,))))
        return Certificate(("declared",), tuple((name, shape, str(dtype)) for name, (shape, dtype)
                                                in shapes.items()), (EMPTY_CONDITION, EMPTY_ROWS))
    state_types = tuple(jax.ShapeDtypeStruct(np.shape(part), part.dtype) for part in empty)
    # Each tile's walked operands its own, its others whole and shared, as a walk's tiles hold them.
    first = _symbolic_tiles(shapes, "t0")
    tiles = [first, {name: first[name] if name not in walked else value
                     for name, value in _symbolic_tiles(shapes, "t1").items()}]
    states = [_symbolic_state(empty, f"s{index}") for index in range(3)]
    whole = {name: first[name] for name in shapes if name not in walked}
    whole_types = {name: tile_types[name] for name in whole}

    def merged(a, b):
        return _evaluate(lambda x, y, given: merge(x, y, given, shapes=shapes),
                         (state_types, state_types, whole_types), (a, b, whole))

    def stepped(state, tile):
        return _evaluate(step, (state_types, tile_types), (state, tile))

    empty_symbols = tuple(_constant(part) for part in empty)
    _equal("identity", merged(empty_symbols, states[0]), states[0])
    _equal("identity", merged(states[0], empty_symbols), states[0])
    _equal("associative", merged(merged(states[0], states[1]), states[2]),
           merged(states[0], merged(states[1], states[2])))
    _equal("consistent", stepped(states[0], tiles[0]), merged(states[0], stepped(empty_symbols, tiles[0])))
    walked_state = stepped(stepped(empty_symbols, tiles[0]), tiles[1])
    joined = {name: np.concatenate([tiles[0][name], tiles[1][name]], axis=walked[name])
              if name in walked else tiles[0][name] for name in shapes}
    joined_types = {name: jax.ShapeDtypeStruct(np.shape(value), shapes[name][1])
                    for name, value in joined.items()}
    # A finish reads no walked operand, its others whole.
    def finished_of(state, given):
        return finish(state, {**dict.fromkeys(walked), **given}, shapes=shapes)

    finished = _outputs(_evaluate(finished_of, (state_types, whole_types), (walked_state, whole)))
    theirs = _outputs(_evaluate(lambda tile: declared(**tile), (joined_types,), (joined,)))
    along = [_outputs(_evaluate(mapped, (tile_types,), (tile,))) for tile in tiles] if mapped else None
    for index, (mine, their) in enumerate(zip(finished, theirs, strict=True)):
        if mine is None and along is not None:
            parts = [np.asarray(found[index], object) for found in along]
            # The walked axis: the one the joined tiles' output is longer along than a tile's.
            axis = next(axis for axis, (one, both)
                        in enumerate(zip(parts[0].shape, np.shape(their), strict=True)) if one != both)
            mine = np.concatenate(parts, axis=axis)
        _equal("declared", mine, their)
    return Certificate(LAWS, tuple((name, shape, str(dtype)) for name, (shape, dtype) in shapes.items()),
                       (EMPTY_CONDITION, EMPTY_ROWS))


def _outputs(value) -> tuple:
    """A form's or a function's outputs as a tuple: one output alone, or each of several."""
    return value if isinstance(value, tuple) else (value,)


def skeletons(declaration, shapes: Mapping) -> tuple[str, ...]:
    """The structure of `declaration`'s forms at a tile of `shapes` (`dew.algebra.derive.Reduction`'s, each
    operand's shape and dtype): each step's, merge's, finish's and output's along the walked axis
    (`skeleton`), one for two tiles exactly where their forms compute alike up to the extents they hold."""
    names = tuple(declaration.operands)
    walked = {name: axes.index(declaration.walked) for name, axes in declaration.operands.items()
              if declaration.walked in axes}
    made = derive._derived(declaration, names, walked, shapes, accurate=False, saturation=SATURATION)
    terms = (*made.steps, *made.merges, *made.finishes, *made.outputs)
    return tuple(skeleton(term, made.region) for term in terms if term is not None)


def certified(declaration, shapes: Mapping) -> Certificate:
    """`declaration`'s derived forms (the fast class's, `dew.algebra.derive.derive`) certified at a tile of
    `shapes` (`certify`)."""
    made = derive.derive(declaration, accurate=False, saturation=SATURATION)
    walked = {name: axes.index(declaration.walked) for name, axes in declaration.operands.items()
              if declaration.walked in axes}
    return certify(init=made.init, step=made.step, merge=made.merge, finish=made.finish,
                   declared=declaration.function, shapes=shapes, walked=walked, mapped=made.mapped)


SHAPES = frozenset({"shape", "new_sizes", "sizes"})
"""The parameters that hold a value's shape, which `skeleton` reads past: a well-typed term's are its
operands' and its consumers' shapes, so two tiles whose extents are equal and unequal alike take them alike.
A window's parameters (a slice's starts, limits and sizes) are no type's: they stay in the skeleton, so a
form sliced at one place shares no certificate with a form sliced at another."""


def skeleton(term, region) -> str:
    """A term's structure in `region` (`terms.Region`): its constructors, operation names, literals and each
    parameter but the shapes it holds (`SHAPES`), so the forms of two tiles of one declaration whose extents
    are equal and unequal alike have one skeleton exactly where they compute alike up to their extents."""
    def parameter(index: int) -> str:
        primitive, params = region.params[index]
        kept = {name: value for name, value in sorted(params.items()) if name not in SHAPES}
        return f"{primitive.name}{kept}"

    made = {leaf: (opaque, place) for opaque in region.opaque for place, leaf in enumerate(opaque.outputs)}

    def defined(leaf: int) -> str:
        # A leaf an equation outside the language computes (a local value: a slice of an operand the walk
        # does not read), as that equation and its inputs: a leaf's index names no computation.
        opaque, place = made[leaf]
        kept = {name: value for name, value in sorted(opaque.equation.params.items()) if name not in SHAPES}
        # Each input its term or, at its own slot, its literal (`terms.Opaque`).
        inputs = [walk(term.__egg_typed_expr__, None, 0) if term is not None else repr(literal)
                  for term, literal in zip(opaque.inputs, opaque.literals, strict=True)]
        return f"{opaque.equation.primitive.name}{kept}({','.join(inputs)})[{place}]"

    def walk(expr, method: str | None, place: int) -> str:
        inner = expr.expr if isinstance(expr, declarations.TypedExprDecl) else expr
        if isinstance(inner, declarations.LitDecl):
            # A product's, a reduction's, a layout's and an operation's parameters, by their index.
            if method in ("dot", "reduce", "shaped", "binary", "compare", "unary", "convert") and isinstance(
                    inner.value, int) and place == _PARAMETER[method]:
                return parameter(inner.value)
            if method == "literal":
                return repr(np.asarray(region.literals[inner.value]).tolist())
            return repr(inner.value)
        if isinstance(inner, declarations.CallDecl):
            # A term's constructor is a method or a class method of its language; a function's and a
            # container's (`Vec[i64](...)`) is named by its identifier.
            named = inner.callable
            methods = (declarations.MethodRef, declarations.ClassMethodRef)
            name = named.method_name if isinstance(named, methods) else named.ident.name
            leaf = _inner_literal(inner.args[0]) if name == "leaf" else None
            if isinstance(leaf, int) and leaf in made:
                return defined(leaf)
            return f"{name}({','.join(walk(arg, name, index) for index, arg in enumerate(inner.args))})"
        return type(inner).__name__

    return walk(term.__egg_typed_expr__, None, 0)


def _inner_literal(expr):
    """A term argument's literal value, None where it is no literal."""
    inner = expr.expr if isinstance(expr, declarations.TypedExprDecl) else expr
    return inner.value if isinstance(inner, declarations.LitDecl) else None


_PARAMETER = {"dot": 2, "reduce": 3, "shaped": 2, "binary": 3, "compare": 3, "unary": 2, "convert": 2}
"""Each term constructor's place of its parameters' index (`terms.Term`)."""


def _symbolic_tiles(shapes: Mapping, prefix: str) -> dict:
    """A tile of fresh real symbols by operand."""
    return {name: _symbols(f"{prefix}_{name}", shape) for name, (shape, _) in shapes.items()}


def _symbolic_state(empty: Sequence, prefix: str) -> tuple:
    """A state of fresh real symbols, shaped as the empty state's parts."""
    return tuple(_symbols(f"{prefix}_{index}", np.shape(part)) for index, part in enumerate(empty))


def _symbols(prefix: str, shape) -> np.ndarray:
    count = math.prod(shape)
    found = sympy.symbols(f"{prefix}_0:{count}", real=True) if count else ()
    return np.asarray(found, dtype=object).reshape(shape)


def _evaluate(function: Callable, types: tuple, arguments: tuple):
    """`function` over SymPy arrays `arguments`: traced at `types` (each argument's shapes and dtypes), each
    choice by platform its default branch (`forms.inlined`: every branch computes one real function), and its
    program evaluated over them (`evaluate`), in its output's structure."""
    leaves, structure = jax.tree.flatten(types)

    def flat(*values):
        return function(*jax.tree.unflatten(structure, values))

    closed = jax.make_jaxpr(forms.inlined(flat))(*leaves)
    found = evaluate(closed.jaxpr, closed.consts, jax.tree.leaves(arguments))
    return jax.tree.unflatten(jax.tree.structure(jax.eval_shape(function, *types)), found)


def evaluate(jaxpr: jcore.Jaxpr, consts, arguments) -> list:
    """`jaxpr`'s outputs over the reals: each array an object array of SymPy expressions, each equation its
    outputs read evaluated."""
    env: dict = {}
    env.update(zip(jaxpr.constvars, (_constant(value) for value in consts), strict=True))
    env.update(zip(jaxpr.invars, arguments, strict=True))

    def read(atom):
        return _constant(atom.val) if isinstance(atom, jcore.Literal) else env[atom]

    # Only the equations the outputs read: an inlined platform choice leaves its index unread.
    needed = forms.read_by(jaxpr, {atom for atom in jaxpr.outvars if isinstance(atom, jcore.Var)})
    for equation in jaxpr.eqns:
        if not any(out in needed for out in equation.outvars):
            continue
        outs = _equation(equation, [read(atom) for atom in equation.invars])
        env.update(zip(equation.outvars, outs, strict=True))
    return [read(atom) for atom in jaxpr.outvars]


LARGEST = frozenset(float(jnp.finfo(dtype).max) for dtype in (jnp.float16, jnp.bfloat16, jnp.float32,
                                                               jnp.float64))
"""The floats' largest values, which the real evaluation reads as `LEAST`'s magnitude (`EMPTY_CONDITION`)."""

LEAST = -sympy.Symbol("largest", positive=True)
"""A float's least value over the reals: finite, below every other value a law compares (`_orderings`), so
`exp(-inf - LEAST)` is 0 as a float's `exp(-inf + 3.4e38)` is, where reading it as -infinity left a NaN."""


def _constant(value) -> np.ndarray:
    """A program's constant over the reals, exactly: a float's value its dyadic rational, so the laws'
    arithmetic stays exact (a 30-digit float's left a scaled exponent's zero test undecided); a float's
    largest magnitude below every other (`LARGEST`)."""
    array = np.asarray(value)

    def real(element):
        number = float(element) if np.issubdtype(array.dtype, np.number) else int(element)
        if not math.isfinite(number):
            return sympy.oo if number > 0 else -sympy.oo
        if abs(number) in LARGEST:
            return -LEAST if number > 0 else LEAST
        if np.issubdtype(array.dtype, np.integer) or array.dtype == bool:
            return sympy.Integer(int(element))
        return sympy.Rational(number)

    return np.vectorize(real, otypes=[object])(array) if array.shape else np.asarray(real(array), object)


_ELEMENTWISE: dict[str, Callable] = {
    "add": lambda x, y: x + y, "sub": lambda x, y: x - y, "mul": lambda x, y: x * y,
    "div": lambda x, y: x / y, "max": sympy.Max, "min": sympy.Min, "pow": lambda x, y: x ** y,
    "neg": lambda x: -x, "exp": sympy.exp, "exp2": lambda x: 2 ** x, "log": sympy.log,
    "log2": lambda x: sympy.log(x) / sympy.log(2), "log1p": lambda x: sympy.log(1 + x),
    "expm1": lambda x: sympy.exp(x) - 1, "abs": sympy.Abs,
    "sign": sympy.sign, "sqrt": sympy.sqrt, "rsqrt": lambda x: 1 / sympy.sqrt(x), "tanh": sympy.tanh,
    "logistic": lambda x: 1 / (1 + sympy.exp(-x)), "square": lambda x: x ** 2,
    "lt": sympy.Lt, "le": sympy.Le, "gt": sympy.Gt, "ge": sympy.Ge, "eq": sympy.Eq, "ne": sympy.Ne,
    "and": sympy.And, "or": sympy.Or, "not": sympy.Not,
    "is_finite": lambda x: sympy.true if x.is_finite else (sympy.false if x.is_infinite else sympy.true),
}

_SAME = frozenset({"convert_element_type", "copy", "copy_p", "optimization_barrier", "stop_gradient",
                   "reduce_precision"})


def _equation(equation, values) -> list:
    name, params = equation.primitive.name, equation.params
    # A call's program, read where every reader reads it (`dew.nn.programs.called`).
    inner = programs.called(equation)
    if inner is not None:
        closed = inner if isinstance(inner, jcore.ClosedJaxpr) else jcore.ClosedJaxpr(inner, ())
        return evaluate(closed.jaxpr, closed.consts, values)
    if name in _SAME:
        return list(values)
    if name == "integer_pow":
        return [np.vectorize(lambda x: x ** params["y"], otypes=[object])(values[0])]
    if name in _ELEMENTWISE:
        operands = np.broadcast_arrays(*values)
        return [np.vectorize(_ELEMENTWISE[name], otypes=[object])(*operands)]
    if name == "select_n":
        predicate, *cases = np.broadcast_arrays(*values)

        def chosen(which, *options):
            if which in (sympy.true, sympy.false) or isinstance(which, (bool, np.bool_)):
                return options[int(bool(which))]
            if isinstance(which, sympy.Integer):
                return options[int(which)]
            return sympy.Piecewise((options[1], which), (options[0], True))

        return [np.vectorize(chosen, otypes=[object])(predicate, *cases)]
    return [_shaped(name, params, values)]


def _shaped(name: str, params: Mapping, values) -> np.ndarray:
    """A layout's or a reduction's equation over object arrays."""
    if name == "iota":
        shape, axis = params["shape"], params["dimension"]
        return np.broadcast_to(np.arange(shape[axis], dtype=object).reshape(
            [-1 if index == axis else 1 for index in range(len(shape))]), shape).astype(object)
    if not values:
        raise TypeError(f"no real evaluation of {name}{dict(params)}, which reads no operand")
    operand = values[0]
    if name == "reduce_sum":
        return np.sum(operand, axis=tuple(params["axes"]), dtype=object)
    if name in ("reduce_max", "reduce_min"):
        pick = sympy.Max if name == "reduce_max" else sympy.Min
        axes = tuple(params["axes"])
        kept = [axis for axis in range(operand.ndim) if axis not in axes]
        rows = np.transpose(operand, (*kept, *axes)).reshape(math.prod(operand.shape[a] for a in kept), -1)
        picked = np.asarray([pick(*row) for row in rows], dtype=object)
        return picked.reshape(tuple(operand.shape[axis] for axis in kept))
    if name == "broadcast_in_dim":
        laid = [1] * len(params["shape"])
        for index, axis in enumerate(params["broadcast_dimensions"]):
            laid[axis] = operand.shape[index]
        return np.broadcast_to(operand.reshape(laid), params["shape"])
    if name == "reshape":
        return operand.reshape(params["new_sizes"])
    if name == "transpose":
        return np.transpose(operand, params["permutation"])
    if name == "squeeze":
        return np.squeeze(operand, axis=tuple(params["dimensions"]))
    if name == "expand_dims":
        return np.expand_dims(operand, tuple(params["dimensions"]))
    if name == "concatenate":
        return np.concatenate(values, axis=params["dimension"])
    if name == "slice":
        return operand[tuple(slice(start, limit, stride) for start, limit, stride in zip(
            params["start_indices"], params["limit_indices"], params["strides"] or (1,) * operand.ndim,
            strict=True))]
    if name == "dot_general":
        return _dot(values[0], values[1], params["dimension_numbers"])
    raise TypeError(f"no real evaluation of {name}")


def _dot(x, y, dimension_numbers) -> np.ndarray:
    (contracted, other_contracted), (batch, other_batch) = dimension_numbers
    free_x = [axis for axis in range(x.ndim) if axis not in (*contracted, *batch)]
    free_y = [axis for axis in range(y.ndim) if axis not in (*other_contracted, *other_batch)]
    left = np.transpose(x, (*batch, *free_x, *contracted))
    right = np.transpose(y, (*other_batch, *other_contracted, *free_y))
    batches = math.prod(x.shape[axis] for axis in batch)
    count = math.prod(x.shape[axis] for axis in contracted)
    rows, columns = math.prod(x.shape[a] for a in free_x), math.prod(y.shape[a] for a in free_y)
    left, right = left.reshape(batches, rows, count), right.reshape(batches, count, columns)
    out = np.empty((batches, rows, columns), object)
    for index in range(batches):
        out[index] = np.dot(left[index], right[index]) if count else sympy.Integer(0)
    return out.reshape((*(x.shape[a] for a in batch), *(x.shape[a] for a in free_x),
                        *(y.shape[a] for a in free_y)))


def _equal(law: str, mine, theirs) -> None:
    """Two structures of the same arrays, each of one shape, each element equal over the reals in every case
    of their comparisons (`_cases`) and every order of their maxima (`_orderings`); `Unlawful` with the first
    that differs: a value's shape is its contract, so the same elements laid out otherwise break the law."""
    left = [np.asarray(leaf, object) for leaf in jax.tree.leaves(mine)]
    right = [np.asarray(leaf, object) for leaf in jax.tree.leaves(theirs)]
    shapes = [part.shape for part in left], [part.shape for part in right]
    if shapes[0] != shapes[1]:
        raise Unlawful(law, f"its sides' shapes differ: {shapes[0]} and {shapes[1]}")
    for one, other in zip(left, right, strict=True):
        for a, b in zip(one.ravel(), other.ravel(), strict=True):
            for decided, mine_case, theirs_case in _cases(sympy.sympify(a), sympy.sympify(b)):
                for ordered in _orderings(mine_case - theirs_case):
                    if not _zero(ordered):
                        residue = sympy.expand(ordered, power_exp=True)
                        raise Unlawful(law, f"where {decided}, {mine_case} against {theirs_case}, their "
                                            f"difference {residue} at an order of their maxima")


def _cases(mine, theirs):
    """Both sides with each comparison they hold (a mask's position against a key's, a sign) decided each way
    in turn: an identity of every assignment is one of every realizable one. An assignment under which the
    right side, the declared or the law's own, has no value is no case of the law (`EMPTY_ROWS`)."""
    relations = sorted((mine - theirs).atoms(sympy.core.relational.Relational), key=str)
    for truths in itertools.product((sympy.true, sympy.false), repeat=len(relations)):
        decided = dict(zip(relations, truths, strict=True))
        mine_case, theirs_case = mine.xreplace(decided), theirs.xreplace(decided)
        if theirs_case.has(sympy.nan, sympy.zoo):
            continue
        yield decided, mine_case, theirs_case


def _scaled_out(expression):
    """`expression` with each maximum and minimum whose arguments share a positive rational factor (their
    contents' greatest common one, a float's least, `LEAST`, apart) taken as that factor times the extremum of
    the arguments over it: a positive scale keeps every order, so a scaled logit's maxima are ordered as its
    raw products' are, not as twice as many values (a positive factor's forward certificate ran past
    900 seconds with every order of both counted: FS-EACH-LAWS1)."""
    def factored(atom):
        scaled = [argument for argument in atom.args if argument != LEAST]
        if not scaled:
            return atom
        factor = functools.reduce(sympy.gcd, [argument.primitive()[0] for argument in scaled])
        if not (factor.is_Rational and factor > 0 and factor != 1):
            return atom
        return factor * type(atom)(*(argument if argument == LEAST else sympy.expand(argument / factor)
                                     for argument in atom.args))

    return expression.replace(lambda term: isinstance(term, (sympy.Max, sympy.Min)), factored)


def _orderings(expression):
    """`expression` with each maximum and minimum it holds decided by each order of the values they compare
    (each over its arguments' positive factor, `_scaled_out`): every maximum its greatest argument by that
    order, every minimum its least, a float's least (`LEAST`) below all, so an identity of every order is
    one of every value of them (two equal values decide alike). An argument that holds a maximum of its own
    is decided again once it is taken."""
    expression = _scaled_out(expression)
    atoms = expression.atoms(sympy.Max, sympy.Min)
    if not atoms:
        yield expression
        return
    compared = sorted({argument for atom in atoms for argument in atom.args} - {LEAST}, key=str)
    for order in itertools.permutations(compared):
        rank = {argument: place for place, argument in enumerate(order)} | {LEAST: -1}
        decided = {atom: (max if isinstance(atom, sympy.Max) else min)(atom.args, key=rank.__getitem__)
                   for atom in atoms}
        yield from _orderings(expression.xreplace(decided))


@functools.lru_cache(maxsize=4096)
def _zero(expression) -> bool:
    """Whether `expression` is zero over the reals. A rational function of exponentials is zero where its
    numerator over one denominator, each exponential split into its terms' (`power_exp`), expands to zero:
    the canonical test, tried first; SymPy's simplification of it, its exponentials joined, where that
    leaves a term."""
    if expression == 0:
        return True
    canonical = sympy.expand(expression, power_exp=True)
    if canonical == 0:
        return True
    numerator, _ = sympy.fraction(sympy.together(canonical))
    if sympy.expand(numerator, power_exp=True) == 0:
        return True
    joined = sympy.powsimp(sympy.expand(expression), combine="exp")
    return sympy.simplify(joined) == 0 or sympy.simplify(sympy.together(sympy.expand_power_exp(joined))) == 0
