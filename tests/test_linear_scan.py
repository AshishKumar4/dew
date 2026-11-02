"""The rules of a recurrence's derivation (`dew.algebra.linear_scan`) proven sound, read from the rules as
egglog holds them (their declarations), so no rule escapes its proof (docs/design/algebra.md §6.9):

- each identity of a token's algebra (`linear_scan.Token`) holds exactly over the reals at every
  instantiation of its variables over the shapes the language's values take (a scalar, a vector along the
  keys or the values, a column, a row, a matrix): where both sides are values (`linear_scan.value_of`'s),
  of one shape and equal;
- each identity of a chunk's algebra (`linear_scan.Chunk`: `wy`, `dual`, a solve's two forms and its
  linearity) holds exactly over the reals on a chunk of symbolic tokens and decays, scalar and per key, a
  decay a positive symbol, each span's the product of its tokens'.

That each declared update's affine form computes its token scan is held by tests/test_algebra_scan.py."""

import itertools
import math

import numpy as np
import pytest
import sympy
from egglog import declarations as decl

from dew.algebra import linear_scan

KEYS, VALUES, TOKENS = 2, 3, 3
SHAPES = ((), (KEYS,), (VALUES,), (KEYS, 1), (1, VALUES), (KEYS, VALUES))
"""Every shape a token's value takes: a scalar, a vector along the keys or the values, a column, a row, a
matrix of the state's."""


class Inadmissible(Exception):
    """A term no value of the language is."""


def _inner(expr):
    return expr.expr if isinstance(expr, decl.TypedExprDecl) else expr


def _name(call: decl.CallDecl) -> str:
    """A constructor's name: every side of the rules calls a method or a class method of its language."""
    return call.callable.method_name


def _symbols(name: str, shape: tuple, **assumptions) -> np.ndarray:
    count = math.prod(shape)
    made = sympy.symbols(f"{name}_0:{count}", real=True, **assumptions) if count else ()
    return np.asarray(made, dtype=object).reshape(shape)


def _broadcast(operation, x, y):
    try:
        return operation(np.asarray(x, dtype=object), np.asarray(y, dtype=object))
    except ValueError as error:
        raise Inadmissible from error


def _laid(value, extent: int, axis: int):
    """A vector laid along a matrix's `axis` (`Token.column`, `Token.row`), a scalar as itself."""
    value = np.asarray(value, dtype=object)
    if value.shape == ():
        return value
    if value.shape != (extent,):
        raise Inadmissible
    return value.reshape((extent, 1) if axis == 0 else (1, extent))


STATE = _symbols("h", (KEYS, VALUES))
MATRICES = ((), (KEYS, 1), (1, VALUES), (KEYS, VALUES))
"""The shapes a value broadcast over the state's takes."""


def _token(expr, bound: dict):
    """A token term's value over SymPy (`linear_scan.value_of`'s, the state a matrix of symbols)."""
    expr = _inner(expr)
    if isinstance(expr, decl.UnboundVarDecl):
        return bound[expr.name]
    if isinstance(expr, decl.LitDecl):
        return sympy.Rational(expr.value)
    args = [_token(arg, bound) for arg in expr.args]
    name = _name(expr)
    if name == "state":
        return STATE
    if name == "literal":
        return np.asarray(args[0], dtype=object)
    if name == "scalar":
        if np.shape(args[0]) != ():
            raise Inadmissible
        return args[0]
    if name in ("column", "row"):
        return _laid(args[0], KEYS if name == "column" else VALUES, 0 if name == "column" else 1)
    if name in ("times", "plus"):
        return _broadcast(np.multiply if name == "times" else np.add, *args)
    if name in ("exp", "exp2"):
        power = sympy.exp if name == "exp" else (lambda x: sympy.Integer(2) ** x)
        return np.vectorize(power, otypes=[object])(np.asarray(args[0], dtype=object))
    if name == "read":
        # A scalar reader reads every key alike (`value_of`'s: a factor of a key vector's, a rule's reverse).
        vector, matrix = (np.asarray(arg, dtype=object) for arg in args)
        if vector.shape not in ((), (KEYS,)) or matrix.shape not in MATRICES:
            raise Inadmissible
        product = vector.reshape(-1, 1) * matrix
        return np.sum(np.broadcast_to(product, (KEYS, product.shape[-1])), axis=0)
    raise TypeError(f"no value of {name}")


def _variables(expr, found: set) -> set:
    expr = _inner(expr)
    if isinstance(expr, decl.UnboundVarDecl):
        found.add(expr.name)
    elif isinstance(expr, decl.CallDecl):
        for arg in expr.args:
            _variables(arg, found)
    return found


def _declared(ruleset) -> list:
    return list(ruleset.__egg_ruleset__.rules)


IDENTITIES = [(index, rule) for index, rule in enumerate(_declared(linear_scan._token_rules()))
              if isinstance(rule, (decl.RewriteDecl, decl.BiRewriteDecl))]


def _equal(left, right) -> bool:
    difference = np.asarray(left - right, dtype=object).ravel()
    return all(sympy.expand(sympy.powsimp(sympy.expand(entry, power_exp=True))) == 0 for entry in difference)


@pytest.mark.parametrize(("index", "rule"), IDENTITIES, ids=[f"token[{index}]" for index, _ in IDENTITIES])
def test_every_token_identity_holds_over_the_reals(index, rule):
    """At every instantiation of the identity's variables over the language's shapes where both its sides
    are values, they are of one shape and equal; at least one instantiation is."""
    names = sorted(_variables(rule.lhs, set()) | _variables(rule.rhs, set()))
    proven = 0
    for shapes in itertools.product(SHAPES, repeat=len(names)):
        bound = {name: _symbols(name, shape) for name, shape in zip(names, shapes, strict=True)}
        values = []
        for side in (rule.lhs, rule.rhs):
            try:
                values.append(_token(side, bound))
            except Inadmissible:
                values.append(None)
        if any(value is None for value in values):
            continue
        left, right = values
        assert np.shape(left) == np.shape(right), f"token[{index}] at {shapes} changes a shape"
        assert _equal(left, right), f"token[{index}] differs at {shapes}"
        proven += 1
    assert proven, f"token[{index}]: no instantiation makes both sides values"


def _chunk_values(per_key: bool) -> dict:
    """A symbolic chunk: its tokens, its entering state, and its decays (`scan._Decays`), each token's a
    positive symbol (a key's own where `per_key`) and each span's the product of its tokens'."""
    keys = KEYS if per_key else 1
    tokens = {name: _symbols(name, (TOKENS, KEYS)) for name in ("B", "C", "Q")}
    tokens.update({name: _symbols(name, (TOKENS, VALUES)) for name in ("U", "R")})
    tokens["s"] = _symbols("s", (TOKENS,))
    step = _symbols("d", (TOKENS, keys), positive=True)

    def span(first: int, last: int) -> np.ndarray:
        """The decay over tokens `first..last` inclusive (none: 1)."""
        product = np.ones(keys, dtype=object)
        for t in range(first, last + 1):
            product = product * step[t]
        return product

    between = np.asarray([[span(j + 1, i) if j <= i else np.zeros(keys, dtype=object)
                           for j in range(TOKENS)] for i in range(TOKENS)], dtype=object)
    decays = {"entering": np.asarray([span(0, i) for i in range(TOKENS)], dtype=object),
              "to_last": np.asarray([span(i + 1, TOKENS - 1) for i in range(TOKENS)], dtype=object),
              "whole": span(0, TOKENS - 1), "between": between, "step": step}
    return {"tokens": tokens, "decays": decays, "h": _symbols("h", (KEYS, VALUES))}


def _solved(lower: np.ndarray, given: np.ndarray) -> np.ndarray:
    """`(I + lower)^-1 given`, `lower` strictly lower, by forward substitution over SymPy."""
    rows: list = []
    for i in range(given.shape[0]):
        rows.append(given[i] - sum((lower[i, j] * rows[j] for j in range(i)), np.zeros_like(given[i])))
    return np.asarray(rows, dtype=object)


def _looped(chunk: dict, values: np.ndarray, *, corrected: bool) -> tuple:
    """The token loop (`scan._looped`'s): its outputs and the state it leaves."""
    tokens, step, h = chunk["tokens"], chunk["decays"]["step"], chunk["h"]
    outputs = []
    for t in range(TOKENS):
        h = step[t][:, None] * h
        if corrected:
            h = h - tokens["s"][t] * np.outer(tokens["B"][t], tokens["C"][t] @ h)
        h = h + np.outer(tokens["B"][t], values[t])
        outputs.append(tokens["Q"][t] @ h + tokens["R"][t])
    return np.asarray(outputs, dtype=object), h


def _chunk(expr, chunk: dict, bound: dict):
    """A chunk term's value over SymPy (`scan._evaluated`'s, a solve's and its inverse's exact)."""
    expr = _inner(expr)
    if isinstance(expr, decl.UnboundVarDecl):
        return bound[expr.name]
    if isinstance(expr, decl.LitDecl):
        return expr.value
    name, decays = _name(expr), chunk["decays"]
    args = [_chunk(arg, chunk, bound) for arg in expr.args]
    if name == "token":
        return chunk["tokens"][args[0]]
    if name == "entering":
        return chunk["h"]
    if name == "decay":
        return decays[args[0]]
    if name in ("loop", "plain"):
        return _looped(chunk, args[0], corrected=name == "loop")
    if name == "pair":
        return tuple(args)
    if name in ("outputs", "state"):
        return args[0][0 if name == "outputs" else 1]
    if name == "gram":
        left, right, strict = args
        paired = [[np.sum(left[i] * right[j] * decays["between"][i, j]) if j < i or (j == i and not strict)
                   else 0 for j in range(TOKENS)] for i in range(TOKENS)]
        return np.asarray(paired, dtype=object)
    if name == "rows":
        return args[1][:, None] * args[0]
    operations = {"times": np.multiply, "matmul": np.matmul, "plus": np.add, "minus": np.subtract}
    if name in operations:
        return operations[name](*args)
    if name == "decayed":
        return decays["whole"][:, None] * args[0]
    if name == "written":
        return (args[0] * decays["to_last"]).T @ args[1]
    if name in ("solve", "substituted"):
        return _solved(*args)
    if name == "doubled":
        return _solved(args[0], np.eye(TOKENS, dtype=int).astype(object)) - np.eye(TOKENS, dtype=int)
    raise TypeError(f"no value of {name}")


VARIABLES = {"given": (TOKENS, VALUES), "other": (TOKENS, KEYS), "left": (TOKENS, VALUES),
             "right": (KEYS, VALUES)}
"""Each chunk identity's variable's shape: a solve's given values and the left operand of a product with
the entering state, a pair's parts; a solve's `lower` strictly lower."""

CHUNK_IDENTITIES = [(corrects, index, rule) for corrects in (True, False)
                    for index, rule in enumerate(_declared(linear_scan._identities(corrects)))]


@pytest.mark.parametrize("per_key", [False, True], ids=["scalar", "per_key"])
@pytest.mark.parametrize(("corrects", "index", "rule"), CHUNK_IDENTITIES,
                         ids=[f"{'wy' if corrects else 'plain'}[{index}]" for corrects, index, _ in
                              CHUNK_IDENTITIES])
def test_every_chunk_identity_holds_over_the_reals(corrects, index, rule, per_key):
    """The identity's two sides equal exactly over a symbolic chunk of three tokens, its decays a scalar's
    or each key's: every entry of each side's value."""
    chunk = _chunk_values(per_key)
    bound = {name: _symbols(name, shape) for name, shape in VARIABLES.items()}
    bound["lower"] = np.tril(_symbols("lower", (TOKENS, TOKENS)), -1)
    left, right = (_chunk(side, chunk, bound) for side in (rule.lhs, rule.rhs))
    for mine, theirs in zip(left if isinstance(left, tuple) else (left,),
                            right if isinstance(right, tuple) else (right,), strict=True):
        assert np.shape(mine) == np.shape(theirs)
        assert _equal(mine, theirs), f"chunk identity {index} differs"
