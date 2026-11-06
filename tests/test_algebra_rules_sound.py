"""Every rule of the pass's rule sets proven sound, read from the rules as egglog holds them (their
declarations), so no rule escapes and none drifts from its proof (docs/design/algebra.md §6.3):

- a rewrite (`rewrite`, `birewrite`, a rule's `union`) is an identity over the reals: its two sides, each
  term an array of SymPy symbols, equal exactly (SymPy's `simplify` of their difference zero, at each order
  of the symbols a maximum or a minimum compares, `_ordered`), at every
  instantiation the rule's conditions admit over a universe of shapes (scalars, vectors, matrices, a batch
  of matrices) and parameters (plain, transposed and batched products, every broadcast of a lower rank),
  each side of one shape. Its conditions on dtypes (`plain`), on a scale's being a power of two
  (`exact_scale`) and on magnitudes (`log2_bound`, `reach`: bible §6's dominance rule) are rounding's,
  which a real identity reads nothing of; a logarithm's operand is positive, its domain, as is a term a
  condition asks be a positive scale (`positive_scale`). A rule no instantiation admits is vacuous, and
  fails;
- a magnitude a rule states for its new intermediate (`set log2_bound`) is at least the bound the magnitude
  table proves for it (`terms._bounded`, itself proven by tests/test_algebra_bounds.py) from its operands'
  stated powers;
- a shape fact (`set shape`, `matrix`) is the shape the term computes;
- a value marked as reading the carried state (`reads`) is one whose value depends on it."""

import functools
import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sympy
from egglog import EGraph, Vec, declarations as decl, i64, set_
from jax import lax

from dew.algebra import carry, cost, rules, terms

SHAPES = ((), (2,), (3,), (2, 3), (3, 2), (2, 2), (2, 3, 2), (2, 2, 3))
"""The universe of shapes each term of a rule takes."""

DOTS = ({"dimension_numbers": (((1,), (0,)), ((), ()))},
        {"dimension_numbers": (((0,), (0,)), ((), ()))},
        {"dimension_numbers": (((2,), (1,)), ((0,), (0,)))})
"""The products' parameters each product of a rule takes: a plain one, its left operand's first axis
contracted (`x^T @ y`), and a batched one."""

PLAIN = DOTS[0]["dimension_numbers"]


def _broadcasts() -> list[dict]:
    """Every broadcast of a lower rank into a shape of the universe."""
    return [{"shape": target, "broadcast_dimensions": dims} for target in SHAPES
            for rank in range(len(target)) for dims in itertools.combinations(range(len(target)), rank)]


BROADCASTS = _broadcasts()
ELEMENTWISE = ({},)
"""An elementwise equation's parameters: none a real identity reads (its dtype's)."""


class Inadmissible(Exception):
    """An instantiation a rule's conditions or its terms' shapes do not admit."""


RANGE_FACTS = frozenset({"log2_bound", "reach", "log2_of", "sum_length", "contracted_length"})
"""The facts a magnitude condition reads: rounding's, which a real identity reads nothing of."""
ROUNDING_FACTS = frozenset({"plain", "exact_scale"})
"""The facts a dtype or power-of-two condition reads: rounding's too."""


class _Magnitude:
    """A magnitude condition's value (`RANGE_FACTS`), which every instantiation admits."""


MAGNITUDE = _Magnitude()


@functools.cache
def _unary(name: str):
    """The real function a unary equation named `name` computes, over SymPy and over NumPy."""
    symbolic = {"exp": sympy.exp, "log": sympy.log, "log1p": lambda x: sympy.log(1 + x),
                "expm1": lambda x: sympy.exp(x) - 1, "neg": lambda x: -x, "tanh": sympy.tanh,
                "abs": sympy.Abs}[name]
    numeric = {"exp": np.exp, "log": np.log, "log1p": np.log1p, "expm1": np.expm1, "neg": np.negative,
               "tanh": np.tanh, "abs": np.abs}[name]
    return np.vectorize(symbolic, otypes=[object]), numeric


def _binary(name: str, x, y):
    """A binary equation's real value, broadcasting as XLA's elementwise equations over a scalar."""
    if name in ("max", "min"):
        pick = sympy.Max if name == "max" else sympy.Min
        if x.dtype == object or y.dtype == object:
            return np.vectorize(pick, otypes=[object])(x, y)
        return np.maximum(x, y) if name == "max" else np.minimum(x, y)
    if x.shape and y.shape and x.shape != y.shape:
        raise Inadmissible(f"{name} of {x.shape} and {y.shape}")
    # A fused multiply-add's names (`rules.CONTRACTED`) are their primitives' over the reals.
    real = cost.PRIMITIVES.get(name, name)
    return {"add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide}[real](x, y)


def _dot(x, y, params):
    """`lax.dot_general` over object or float arrays: batch axes first, then the left's free, the right's
    free axes, each contracted pair summed."""
    (contracted, other_contracted), (batch, other_batch) = params["dimension_numbers"]
    for operand, axes in ((x, (*contracted, *batch)), (y, (*other_contracted, *other_batch))):
        if any(axis >= operand.ndim for axis in axes):
            raise Inadmissible("a product's axes past its operand's rank")
    if [x.shape[a] for a in contracted] != [y.shape[a] for a in other_contracted] or (
            [x.shape[a] for a in batch] != [y.shape[a] for a in other_batch]):
        raise Inadmissible("a product's axes of other extents")
    free_x = [a for a in range(x.ndim) if a not in (*contracted, *batch)]
    free_y = [a for a in range(y.ndim) if a not in (*other_contracted, *other_batch)]
    left = np.transpose(x, (*batch, *free_x, *contracted))
    right = np.transpose(y, (*other_batch, *other_contracted, *free_y))
    count = math.prod(x.shape[a] for a in contracted)
    batches = math.prod(x.shape[a] for a in batch)
    rows, columns = math.prod(x.shape[a] for a in free_x), math.prod(y.shape[a] for a in free_y)
    left = left.reshape(batches, rows, count)
    right = right.reshape(batches, count, columns)
    out = np.empty((batches, rows, columns), dtype=np.result_type(x, y))
    for b in range(batches):
        out[b] = np.dot(left[b], right[b]) if count else np.zeros((rows, columns), out.dtype)
    return out.reshape((*(x.shape[a] for a in batch), *(x.shape[a] for a in free_x),
                        *(y.shape[a] for a in free_y)))


def _broadcast(x, params):
    """`lax.broadcast_in_dim`."""
    target, dims = params["shape"], params["broadcast_dimensions"]
    if len(dims) != x.ndim or any(x.shape[i] not in (1, target[d]) for i, d in enumerate(dims)):
        raise Inadmissible("a broadcast of another rank or extent")
    laid = [1] * len(target)
    for i, d in enumerate(dims):
        laid[d] = x.shape[i]
    return np.broadcast_to(x.reshape(laid), target)


TERM_METHODS = frozenset({"binary", "compare", "unary", "convert", "dot", "shaped", "reduce", "select"})
"""The methods a term's value is an array of."""


def _inner(expr):
    """`expr` unwrapped of its type."""
    return expr.expr if isinstance(expr, decl.TypedExprDecl) else expr


class Evaluated:
    """A rule's terms' values at one instantiation: its terms' arrays (`terms`), its parameters'
    (`params`), its other variables' (`bound`); `symbolic` evaluates over SymPy, else over NumPy floats."""

    def __init__(self, arrays: dict, params: dict, symbolic: bool):
        self.arrays, self.params, self.bound, self.symbolic = arrays, params, {}, symbolic

    def value(self, expr):
        if isinstance(expr, decl.TypedExprDecl):
            return self.value(expr.expr)
        if isinstance(expr, decl.UnboundVarDecl):
            for held in (self.arrays, self.params, self.bound):
                if expr.name in held:
                    return held[expr.name]
            raise KeyError(expr.name)
        if isinstance(expr, decl.LitDecl):
            return expr.value
        if isinstance(expr, decl.CallDecl):
            return self.call(expr)
        raise TypeError(f"no value of a {type(expr).__name__}")

    def call(self, expr):
        ref, args = expr.callable, expr.args
        if isinstance(ref, decl.MethodRef):
            return self.method(ref.method_name, args)
        name = ref.ident.name if hasattr(ref, "ident") else ref.class_ident.name
        if isinstance(ref, decl.InitRef):
            return tuple(self.value(arg) for arg in args)
        if isinstance(ref, decl.ClassMethodRef):
            return ()
        return self.function(name, args)

    def method(self, name: str, args):
        values = [self.value(arg) for arg in args]
        if any(value is MAGNITUDE for value in values):
            return MAGNITUDE
        found = self._method(name, values)
        # A 0-d array's arithmetic gives a scalar: an array again.
        if isinstance(found, (sympy.Basic, float, int, np.generic)) and name in TERM_METHODS:
            return np.asarray(found, dtype=object if self.symbolic else float)
        return found

    def _method(self, name: str, values: list):
        if name == "binary":
            x, op, y, _ = values
            return _binary(op, x, y)
        if name == "compare":
            x, op, y, _ = values
            if self.symbolic:
                relation = {"lt": sympy.Lt, "gt": sympy.Gt, "eq": sympy.Eq}[op]
                return np.vectorize(lambda a, b: relation(a, b, evaluate=False), otypes=[object])(x, y)
            return {"lt": np.less, "gt": np.greater, "eq": np.equal}[op](x, y)
        if name == "unary":
            x, op, _ = values
            symbolic, numeric = _unary(op)
            return symbolic(x) if np.asarray(x).dtype == object else numeric(x)
        if name == "convert":
            return values[0]
        if name == "dot":
            return _dot(values[0], values[1], values[2])
        if name == "shaped":
            x, kind, params = values
            if kind == "broadcast_in_dim" and "shape" in params:
                return _broadcast(x, params)
            if kind == "transpose" and "permutation" in params:
                if len(params["permutation"]) != x.ndim:
                    raise Inadmissible("a transpose of another rank")
                return np.transpose(x, params["permutation"])
            raise Inadmissible(f"a {kind} of another's parameters")
        if name == "reduce":
            x, monoid, axes, _ = values
            if any(axis >= x.ndim for axis in axes):
                raise Inadmissible("a reduction past its operand's rank")
            if monoid == "sum":
                return np.sum(x, axis=tuple(axes))
            if monoid in ("max", "min"):
                return _extreme(x, axes, monoid, symbolic=self.symbolic)
            raise TypeError(monoid)
        if name == "select":
            predicate, false, true = values
            if self.symbolic:
                return np.vectorize(lambda p, f, t: sympy.Piecewise((t, p), (f, True)), otypes=[object])(
                    predicate, false, true)
            return np.where(predicate, true, false)
        if name in ("__getitem__",):
            return values[0][values[1]]
        if name == "length":
            return len(values[0])
        if name == "remove":
            return tuple(size for index, size in enumerate(values[0]) if index != values[1])
        if name in ("__add__", "max", "__le__"):
            return MAGNITUDE
        raise TypeError(f"no method {name}")

    def function(self, name: str, args):
        if name in RANGE_FACTS:
            return MAGNITUDE
        values = [self.value(arg) for arg in args]
        if name == "shape":
            return tuple(values[0].shape)
        if name == "along":
            params, axis = values
            rank = len(params["shape"])
            return params["broadcast_dimensions"] == tuple(a for a in range(rank) if a != axis)
        if name == "lifted":
            extent, axis = values
            return {"shape": tuple(extent), "broadcast_dimensions": (axis,)}
        if name == "matrix_product":
            return values[0]["dimension_numbers"] == PLAIN
        if name == "matrix":
            return values[0].ndim == 2
        if name in ("plain_of", "swap_of"):
            # Defined of a product of its left operand's first axis contracted alone (`carry._products`).
            if values[0].get("dimension_numbers") != DOTS[1]["dimension_numbers"]:
                raise Inadmissible(f"{name} of another product")
            return {"dimension_numbers": PLAIN} if name == "plain_of" else {"permutation": (1, 0)}
        if name == "stock":
            return values[0]
        if name == "unit":
            return np.shape(values[0]) == () and bool(np.asarray(values[0]) == 1)
        if name == "positive_scale":
            # Its witness is positive (`_instantiations`): a scalar of it the fact's own.
            return np.shape(values[0]) == ()
        if name == "reads":
            return 1
        if name in ROUNDING_FACTS:
            return True
        raise TypeError(f"no function {name}")

    def holds(self, fact) -> bool:
        """Whether `fact`, a condition, holds, binding a variable it names."""
        if isinstance(fact, decl.ExprFactDecl):
            found = self.value(fact.typed_expr)
            return True if found is MAGNITUDE else bool(found)
        left, right = (fact.left, fact.right) if hasattr(fact, "left") else fact.exprs
        for side, other in ((left, right), (right, left)):
            named = _inner(side)
            if isinstance(named, decl.UnboundVarDecl) and not any(
                    named.name in held for held in (self.arrays, self.params, self.bound)):
                self.bound[named.name] = self.value(other)
                return True
        found, other = self.value(left), self.value(right)
        if found is MAGNITUDE or other is MAGNITUDE:
            return True
        if isinstance(found, np.ndarray) or isinstance(other, np.ndarray):
            return np.shape(found) == np.shape(other) and bool(np.all(found == other))
        return found == other


def _declared(ruleset) -> list:
    """`ruleset`'s rules, as egglog declares them."""
    return list(ruleset.__egg_ruleset__.rules)


SETS = {name: entry.ruleset for name, entry in rules.RULES.items()}
SETS.update(facts=rules.FACTS, carried=carry._carried(), carrying=carry._carrying())
RULES = [(name, index, rule) for name, ruleset in SETS.items()
         for index, rule in enumerate(_declared(ruleset))]


def _parts(rule) -> tuple:
    """A rule's matched term, its conditions and its actions: a rewrite's sides as a match and a union."""
    if isinstance(rule, (decl.RewriteDecl, decl.BiRewriteDecl)):
        return rule.lhs, list(rule.conditions), [("union", rule.rhs)]
    matched, conditions = None, []
    for fact in rule.body:
        if matched is None and not isinstance(fact, decl.ExprFactDecl):
            left, right = (fact.left, fact.right) if hasattr(fact, "left") else fact.exprs
            named = left.expr if isinstance(left, decl.TypedExprDecl) else left
            if isinstance(named, decl.UnboundVarDecl) and _is_term(right):
                matched = named.name, right
                continue
        conditions.append(fact)
    actions = []
    for action in rule.head:
        if isinstance(action, decl.UnionDecl):
            actions.append(("union", action.rhs))
        elif isinstance(action, decl.SetDecl):
            actions.append(("set", action))
        else:
            actions.append(("do", action.typed_expr))
    return matched, conditions, actions


def _is_term(expr) -> bool:
    """Whether `expr` is a term's call (a `Term` method)."""
    inner = expr.expr if isinstance(expr, decl.TypedExprDecl) else expr
    return isinstance(inner, decl.CallDecl) and isinstance(inner.callable, decl.MethodRef) and (
        inner.callable.ident.name == "Term")


def _variables(expr, found: dict) -> dict:
    """Each variable `expr` reads, by its role: a term's, an operation's name (by the method naming it), a
    reduction's axes, or a parameter's of a product, a broadcast, a transpose or an elementwise equation."""
    inner = _inner(expr)
    if isinstance(inner, decl.UnboundVarDecl):
        found.setdefault(inner.name, "term")
        return found
    if not isinstance(inner, decl.CallDecl):
        return found
    args = [_inner(arg) for arg in inner.args]
    if isinstance(inner.callable, decl.MethodRef) and inner.callable.ident.name == "Term":
        method = inner.callable.method_name
        role = {"dot": "dot", "binary": "elementwise", "unary": "elementwise", "convert": "elementwise",
                "compare": "elementwise", "reduce": "elementwise"}.get(method)
        if method == "shaped":
            kind = args[1]
            role = ({"broadcast_in_dim": "broadcast", "transpose": "transpose"}[kind.value]
                    if isinstance(kind, decl.LitDecl) else "layout")
        named = {"binary": 1, "compare": 1, "unary": 1, "reduce": 1, "shaped": 1}.get(method)
        for place, arg in enumerate(args):
            if not isinstance(arg, decl.UnboundVarDecl):
                _variables(arg, found)
            elif place == len(args) - 1 and role:
                found[arg.name] = role
            elif place == named:
                found[arg.name] = f"{method} name"
            elif method == "reduce" and place == 2:
                found[arg.name] = "axes"
            else:
                found.setdefault(arg.name, "term")
        return found
    for arg in args:
        _variables(arg, found)
    return found


def _free(conditions, roles: dict, matched: str) -> list[str]:
    """The variables `conditions` read that neither the matched term (`matched`, its name) nor an equality
    binds: an axis a condition names (`along(laid, axis)`), each taken over the universe's axes."""
    bound = set(roles) | {matched}
    for fact in conditions:
        if not isinstance(fact, decl.ExprFactDecl):
            for side in (fact.left, fact.right) if hasattr(fact, "left") else fact.exprs:
                if isinstance(_inner(side), decl.UnboundVarDecl):
                    bound.add(_inner(side).name)
    free: list[str] = []
    for fact in conditions:
        _names(fact.typed_expr if isinstance(fact, decl.ExprFactDecl) else (
            fact.right if hasattr(fact, "right") else fact.exprs[1]), free)
    return sorted({name for name in free if name not in bound})


def _names(expr, found: list) -> None:
    inner = _inner(expr)
    if isinstance(inner, decl.UnboundVarDecl):
        found.append(inner.name)
    elif isinstance(inner, decl.CallDecl):
        for arg in inner.args:
            _names(arg, found)


CANDIDATES = {"dot": DOTS, "broadcast": BROADCASTS, "transpose": ({"permutation": (1, 0)},),
              "elementwise": ELEMENTWISE, "axes": ((0,), (1,), (2,)), "axis": (0, 1, 2),
              "binary name": ("add", "sub", "mul", "max", "min"), "compare name": ("lt", "eq"),
              "unary name": ("exp", "neg", "tanh"), "reduce name": ("sum", "max", "min"),
              "shaped name": ("broadcast_in_dim", "transpose"),
              "layout": (*BROADCASTS, {"permutation": (1, 0)})}


def _instantiations(matched, conditions, symbolic: bool):
    """Every instantiation of a rule's variables over the universe its conditions admit, its matched term
    evaluated: `(evaluated, value)`. A term a condition asks be the literal 1 (`unit`) is 1."""
    name, pattern = matched
    roles = _variables(pattern, {})
    roles.update(dict.fromkeys(_free(conditions, roles, name), "axis"))
    units = _named(conditions, "unit")
    positive = _named(conditions, "positive_scale") | POSITIVE
    term_names = sorted(n for n, role in roles.items() if role == "term" and n not in units)
    param_names = sorted(n for n, role in roles.items() if role != "term")
    for shapes in itertools.product(SHAPES, repeat=len(term_names)):
        for chosen in itertools.product(*(CANDIDATES[roles[n]] for n in param_names)):
            arrays = {n: _operand(n, shape, symbolic, positive=n in positive)
                      for n, shape in zip(term_names, shapes, strict=True)}
            arrays.update({n: np.asarray(1 if symbolic else 1.0, dtype=object if symbolic else float)
                           for n in units})
            evaluated = Evaluated(arrays, dict(zip(param_names, chosen, strict=True)), symbolic)
            try:
                value = evaluated.value(pattern)
                evaluated.bound[name] = value
                if all(evaluated.holds(fact) for fact in conditions):
                    yield evaluated, value
            except (Inadmissible, ValueError, IndexError):
                continue


def _named(conditions, fact: str) -> set[str]:
    """The terms `conditions` ask `fact` of (`unit`: be the literal 1; `positive_scale`: be past zero)."""
    found = set()
    for condition in conditions:
        inner = _inner(condition.typed_expr) if isinstance(condition, decl.ExprFactDecl) else None
        named = getattr(inner.callable, "ident", None) if isinstance(inner, decl.CallDecl) else None
        if named is not None and named.name == fact:
            found.add(_inner(inner.args[0]).name)
    return found


def _operand(name: str, shape: tuple, symbolic: bool, *, positive: bool = False):
    """A term's value: an array of fresh real symbols (positive ones where `positive`), or of distinct floats
    (a fresh shape's check)."""
    count = math.prod(shape)
    if not symbolic:
        return np.arange(1.0, count + 1.0).reshape(shape) + 0.5
    symbols = sympy.symbols(f"{name}_0:{count}", real=True, positive=positive or None) if count else ()
    return np.asarray(symbols, dtype=object).reshape(shape)


POSITIVE = frozenset({"t"})
"""Terms a rule takes the logarithm of: positive, the logarithm's domain."""


REWRITES = [(name, index, rule) for name, index, rule in RULES
            if any(kind == "union" for kind, _ in _parts(rule)[2])]


@pytest.mark.parametrize(("name", "index", "rule"), REWRITES,
                         ids=[f"{name}[{index}]" for name, index, _ in REWRITES])
def test_every_rewrite_is_an_identity_over_the_reals(name, index, rule):
    """The rewrite's two sides equal exactly over the reals, each of one shape, at every instantiation
    its conditions admit over the universe; at least one does."""
    matched, conditions, actions = _parts(rule)
    if isinstance(rule, (decl.RewriteDecl, decl.BiRewriteDecl)):
        matched = ("t", matched)
    proven = 0
    for evaluated, value in _instantiations(matched, conditions, symbolic=True):
        for kind, rhs in actions:
            if kind != "union":
                continue
            other = evaluated.value(rhs)
            assert np.shape(other) == np.shape(value), f"{name}[{index}] changes {np.shape(value)}"
            difference = np.asarray(value - other, dtype=object).ravel()
            assert all(sympy.simplify(sympy.expand(ordered)) == 0 for entry in difference
                       for ordered in _ordered(sympy.sympify(entry))), (
                f"{name}[{index}] differs: {difference[:2]}")
        proven += 1
    assert proven, f"{name}[{index}]: no instantiation over the universe admits it"


def _extreme(x: np.ndarray, axes, monoid: str, *, symbolic: bool) -> np.ndarray:
    """The maximum or minimum of `x` over `axes`, each axis folded in turn (SymPy's `Max`/`Min` over
    symbols)."""
    pick = {"max": (sympy.Max, np.maximum), "min": (sympy.Min, np.minimum)}[monoid][0 if symbolic else 1]
    for axis in sorted(axes, reverse=True):
        parts = np.moveaxis(x, axis, 0)
        x = parts[0]
        for part in parts[1:]:
            x = np.vectorize(pick, otypes=[object])(x, part) if symbolic else pick(x, part)
    return np.asarray(x, dtype=object if symbolic else float)


def _ordered(expression):
    """`expression` at each order of the symbols its maxima and minima compare: each a base and nonnegative
    gaps above it, so SymPy decides every comparison of them (and of their positive multiples), ties among
    them; `expression` alone where it holds none."""
    compared = sorted({symbol for found in expression.atoms(sympy.Max, sympy.Min)
                       for symbol in found.free_symbols if not symbol.is_positive}, key=str)
    if not compared:
        yield expression
        return
    base = sympy.Symbol("base", real=True)
    gaps = sympy.symbols(f"gap_0:{len(compared)}", nonnegative=True)
    for order in itertools.permutations(compared):
        decided = expression.subs({symbol: base + sum(gaps[:place]) for place, symbol in enumerate(order)})
        assert not decided.atoms(sympy.Max, sympy.Min), f"undecided by an order: {decided}"
        yield decided


def _set_name(action) -> str:
    """The function a `set` action writes."""
    return action.call.callable.ident.name


MADE = [(name, index, rule) for name, index, rule in RULES
        if any(kind == "set" and _set_name(action) == "log2_bound" for kind, action in _parts(rule)[2])]


POWERS = (3, 5, 0)
"""The powers of two each term's stated magnitude takes in turn (`log2_bound`)."""


def _jax_value(expr, arrays: dict, params: dict, bound: dict, matched: tuple):
    """A term's value as JAX computes it over `arrays`, for the magnitude table to bound: the matched term
    (`matched`, its name and pattern) by its pattern."""
    inner = _inner(expr)
    if isinstance(inner, decl.UnboundVarDecl):
        if inner.name == matched[0]:
            return _jax_value(matched[1], arrays, params, bound, matched)
        return arrays[inner.name] if inner.name in arrays else bound[inner.name]
    method, args = inner.callable.method_name, inner.args

    def value(arg):
        return _jax_value(arg, arrays, params, bound, matched)

    def param(arg):
        found = arg.expr
        return params[found.name] if found.name in params else bound[found.name]

    if method == "binary":
        op = args[1].expr.value
        return {"add": jnp.add, "sub": jnp.subtract, "mul": jnp.multiply}[op](value(args[0]), value(args[2]))
    if method == "dot":
        return lax.dot_general(value(args[0]), value(args[1]), param(args[2])["dimension_numbers"])
    if method == "reduce":
        reduced = {"sum": jnp.sum, "max": jnp.max, "min": jnp.min}[args[1].expr.value]
        return reduced(value(args[0]), axis=tuple(_axes(args[2], params, bound)))
    if method == "shaped":
        kind, laid = args[1].expr.value, param(args[2])
        if kind == "transpose":
            return jnp.transpose(value(args[0]), laid["permutation"])
        return lax.broadcast_in_dim(value(args[0]), laid["shape"], laid["broadcast_dimensions"])
    raise TypeError(method)


def _axes(arg, params: dict, bound: dict) -> tuple:
    inner = _inner(arg)
    if isinstance(inner, decl.UnboundVarDecl):
        return params[inner.name] if inner.name in params else bound[inner.name]
    return tuple(_inner(part).value for part in inner.args)


@pytest.mark.parametrize(("name", "index", "rule"), MADE,
                         ids=[f"{name}[{index}]" for name, index, _ in MADE])
def test_every_stated_magnitude_is_at_least_the_tables(name, index, rule):
    """A new intermediate's stated power of two is at least the bound the magnitude table proves of it from
    its operands' stated powers, at every instantiation and every assignment of `POWERS`."""
    matched, conditions, actions = _parts(rule)
    checked = 0
    for evaluated, _ in itertools.islice(_instantiations(matched, conditions, symbolic=False), 64):
        names = sorted(evaluated.arrays)
        for powers in itertools.permutations(POWERS * 2, len(names)):
            stated_powers = dict(zip(names, powers, strict=True))
            for kind, action in actions:
                if kind != "set" or _set_name(action) != "log2_bound":
                    continue
                made = action.call.args[0]
                stated = _power(action.rhs, rule, stated_powers, evaluated)
                operands = [jnp.asarray(evaluated.arrays[n], jnp.float32) for n in names]

                def computed(*values, made=made, names=names, evaluated=evaluated):
                    return _jax_value(made, dict(zip(names, values, strict=True)), evaluated.params,
                                      evaluated.bound, matched)

                closed = jax.make_jaxpr(computed)(*operands)
                region = terms.region(closed, {i: 2.0 ** stated_powers[n] for i, n in enumerate(names)})
                table = region.magnitudes.get(terms.key(region.outputs[0]))
                assert table is not None and stated >= table, (
                    f"{name}[{index}] states 2**{stated} where the table proves 2**{table}")
                checked += 1
    assert checked, f"{name}[{index}]: no instantiation over the universe admits it"


def _power(expr, rule, powers: dict, evaluated: Evaluated) -> int:
    """A stated power's value: each term's `log2_bound` its assigned power, `log2_of(n)` `ceil(log2 n)`."""
    inner = expr.expr if isinstance(expr, decl.TypedExprDecl) else expr
    if isinstance(inner, decl.LitDecl):
        return inner.value
    if isinstance(inner, decl.UnboundVarDecl):
        return _power(_binding(rule, inner.name), rule, powers, evaluated)
    ref, args = inner.callable, inner.args
    if isinstance(ref, decl.MethodRef):
        values = [_power(arg, rule, powers, evaluated) for arg in args]
        return {"__add__": sum, "max": max}[ref.method_name](values)
    if ref.ident.name == "log2_bound":
        return powers[args[0].expr.name]
    if ref.ident.name == "log2_of":
        return max(int(evaluated.value(args[0])) - 1, 0).bit_length()
    if ref.ident.name == "sum_length":
        reduced = _inner(args[0])
        operand, axes = evaluated.value(reduced.args[0]), evaluated.value(reduced.args[2])
        return max(math.prod(np.shape(operand)[axis] for axis in axes) - 1, 0).bit_length()
    if ref.ident.name == "contracted_length":
        # The elements each output of the product sums: its left operand's contracted axes.
        product = _inner(args[0])
        (contracting, _), _ = evaluated.value(product.args[2])["dimension_numbers"]
        length = math.prod(np.shape(evaluated.value(product.args[0]))[axis] for axis in contracting)
        return max(length - 1, 0).bit_length()
    raise TypeError(ref.ident.name)


def _binding(rule, name: str):
    """The expression a rule's condition binds `name` to."""
    for fact in rule.body:
        if isinstance(fact, decl.ExprFactDecl):
            continue
        left, right = (fact.left, fact.right) if hasattr(fact, "left") else fact.exprs
        inner = left.expr if isinstance(left, decl.TypedExprDecl) else left
        if isinstance(inner, decl.UnboundVarDecl) and inner.name == name:
            return right
    raise KeyError(name)


SHAPE_FACTS = [(name, index, rule) for name, index, rule in RULES if name == "facts"]


@pytest.mark.parametrize(("name", "index", "rule"), SHAPE_FACTS,
                         ids=[f"{name}[{index}]" for name, index, _ in SHAPE_FACTS])
def test_every_shape_fact_is_the_shape_its_term_computes(name, index, rule):
    """A shape fact's stated shape is the shape its term computes, and a term it calls a matrix has two
    axes, at every instantiation its conditions admit; at least one does."""
    matched, conditions, actions = _parts(rule)
    proven = 0
    if matched is None:
        # The matrix fact, applied: a term the facts give each shape is a matrix exactly where an array of
        # that shape has two axes.
        for shape in SHAPES:
            graph = EGraph()
            term = terms.Term.leaf(i64(0))
            graph.register(set_(rules.shape(term)).to(Vec[i64](*(i64(extent) for extent in shape))))
            graph.run(rules.FACTS.saturate())
            matrix = graph.check_bool(rules.matrix(term))
            assert matrix == (np.ones(shape).ndim == 2), (shape, f"{name}[{index}]")
            proven += 1
    else:
        for evaluated, value in _instantiations(matched, conditions, symbolic=False):
            for kind, action in actions:
                if kind == "set":
                    assert tuple(evaluated.value(action.rhs)) == np.shape(value), f"{name}[{index}]"
            proven += 1
    assert proven, f"{name}[{index}]: no instantiation over the universe admits it"


READS = [(name, index, rule) for name, index, rule in RULES if name == "carrying"]


@pytest.mark.parametrize(("name", "index", "rule"), READS,
                         ids=[f"{name}[{index}]" for name, index, _ in READS])
def test_every_value_marked_reading_the_state_depends_on_it(name, index, rule):
    """A value a rule marks as reading the carried state depends on the child its condition names as
    reading it: its symbols hold that child's, at every instantiation; at least one admits it."""
    matched, conditions, _ = _parts(rule)
    child = next(fact for fact in conditions if not isinstance(fact, decl.ExprFactDecl))
    left = _inner(child.left if hasattr(child, "left") else child.exprs[0])
    reader = _inner(left.args[0]).name
    proven = 0
    for evaluated, value in _instantiations(matched, [], symbolic=True):
        held = {symbol for entry in np.asarray(evaluated.arrays[reader], dtype=object).ravel()
                for symbol in sympy.sympify(entry).free_symbols}
        made = {symbol for entry in np.asarray(value, dtype=object).ravel()
                for symbol in sympy.sympify(entry).free_symbols}
        assert held & made, f"{name}[{index}] marks a value not reading {reader}"
        proven += 1
    assert proven, f"{name}[{index}]: no instantiation over the universe admits it"
