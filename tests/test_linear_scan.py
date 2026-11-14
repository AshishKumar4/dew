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
    if name == "applied":
        # The state read along its values (`value_of`'s `_applied`): a scalar vector reads every value alike.
        matrix, vector = (np.asarray(arg, dtype=object) for arg in args)
        if vector.shape not in ((), (VALUES,)) or matrix.shape not in MATRICES:
            raise Inadmissible
        product = matrix * vector.reshape(1, -1)
        return np.sum(np.broadcast_to(product, (KEYS, product.shape[-1])), axis=1)
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


def _chunk_values(per_key: bool, values_read: bool = False) -> dict:
    """A symbolic chunk: its tokens, its entering state, and its decays (`scan._Decays`), each token's a
    positive symbol (a key's own where `per_key`) and each span's the product of its tokens'; its reader and
    free readout along the values and the keys where `values_read`."""
    keys = KEYS if per_key else 1
    read, out = (VALUES, KEYS) if values_read else (KEYS, VALUES)
    tokens = {name: _symbols(name, (TOKENS, KEYS)) for name in ("B", "C")}
    tokens.update(Q=_symbols("Q", (TOKENS, read)), R=_symbols("R", (TOKENS, out)),
                  U=_symbols("U", (TOKENS, VALUES)))
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
    return {"tokens": tokens, "decays": decays, "h": _symbols("h", (KEYS, VALUES)),
            "values_read": values_read}


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
        read = h @ tokens["Q"][t] if chunk["values_read"] else tokens["Q"][t] @ h
        outputs.append(read + tokens["R"][t])
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
    if name == "prefix":
        return None
    if name == "decay":
        return decays[args[0]]
    if name in ("loop", "plain"):
        return _looped(chunk, args[0], corrected=name == "loop")
    if name == "pair":
        return tuple(args)
    if name in ("outputs", "state"):
        return args[0][0 if name == "outputs" else 1]
    if name == "gram":
        left, right, strict, between = args
        paired = [[np.sum(left[i] * right[j] * between[i, j]) if j < i or (j == i and not strict)
                   else 0 for j in range(TOKENS)] for i in range(TOKENS)]
        return np.asarray(paired, dtype=object)
    if name == "spread":
        readers, values, keys, between = args
        return np.asarray([sum((np.dot(readers[i], values[j]) * keys[j] * between[i, j]
                                for j in range(i + 1)), np.zeros(KEYS, dtype=object))
                           for i in range(TOKENS)], dtype=object)
    if name == "applied":
        return args[1] @ args[0].T
    if name == "rows":
        return args[1][:, None] * args[0]
    operations = {"times": np.multiply, "matmul": np.matmul, "plus": np.add, "minus": np.subtract}
    if name in operations:
        return operations[name](*args)
    if name == "decayed":
        return args[1][:, None] * args[0]
    if name == "written":
        return (args[0] * args[2]).T @ args[1]
    if name in ("solve", "substituted"):
        return _solved(*args)
    if name == "doubled":
        return _solved(args[0], np.eye(TOKENS, dtype=int).astype(object)) - np.eye(TOKENS, dtype=int)
    raise TypeError(f"no value of {name}")


VARIABLES = {"given": (TOKENS, VALUES), "other": (TOKENS, KEYS), "left": (TOKENS, VALUES),
             "right": (KEYS, VALUES)}
"""Each chunk identity's variable's shape: a solve's given values and the left operand of a product with
the entering state, a pair's parts; a solve's `lower` strictly lower."""

CHUNK_IDENTITIES = [(corrects, per_key, values_read, index, rule)
                    for corrects in (True, False) for per_key in (False, True)
                    for values_read in (False, True)
                    for index, rule in enumerate(_declared(linear_scan._identities(corrects, not per_key,
                                                                                   values_read)))]


@pytest.mark.parametrize(("corrects", "per_key", "values_read", "index", "rule"), CHUNK_IDENTITIES,
                         ids=[f"{'wy' if corrects else 'plain'}-{'per_key' if per_key else 'scalar'}"
                              f"{'-values' if values_read else ''}[{index}]"
                              for corrects, per_key, values_read, index, _ in CHUNK_IDENTITIES])
def test_every_chunk_identity_holds_over_the_reals(corrects, per_key, values_read, index, rule):
    """The identity's two sides equal exactly over a symbolic chunk of three tokens, its decays a scalar's
    or each key's, its readout along the state's keys or its values, each under the identities the engine
    saturates for them (`linear_scan._identities`): every entry of each side's value."""
    chunk = _chunk_values(per_key, values_read)
    bound = {name: _symbols(name, shape) for name, shape in VARIABLES.items()}
    bound["lower"] = np.tril(_symbols("lower", (TOKENS, TOKENS)), -1)
    left, right = (_chunk(side, chunk, bound) for side in (rule.lhs, rule.rhs))
    for mine, theirs in zip(left if isinstance(left, tuple) else (left,),
                            right if isinstance(right, tuple) else (right,), strict=True):
        assert np.shape(mine) == np.shape(theirs)
        assert _equal(mine, theirs), f"chunk identity {index} differs"


@pytest.mark.parametrize("size", [2, 4, 8])
def test_the_templates_doubling_is_the_systems_inverse(size):
    """The inverse by doubling as the template runs it (`recurrence_algebra.doubled`, which `scan._lowered`
    and every recurrence mainloop take), its program traced and evaluated over the reals
    (`algebra_laws.evaluate`): of a strictly lower `a` whose every entry is its own symbol, `(I - a)^-1`
    exactly, its every level's join among them (sizes 2, 4 and 8: levels of two-, four- and eight-wide
    blocks). With `test_a_doublings_join_is_the_block_inverse_at_every_size`, the doubling is the inverse at
    every chunk size: each level joins blocks by the one identity."""
    import jax
    import jax.numpy as jnp
    from algebra_laws import evaluate

    from dew.nn.kernels import recurrence_algebra

    def doubled(a):
        return recurrence_algebra.doubled(a, lambda name, left, right: left @ right)

    closed = jax.make_jaxpr(doubled)(jax.ShapeDtypeStruct((size, size), jnp.float32))
    a = np.tril(_symbols("a", (size, size)), -1)
    inverse, = evaluate(closed.jaxpr, closed.consts, [a])
    assert _equal(inverse, _solved(-a, np.eye(size, dtype=int).astype(object)))


def test_a_doublings_join_is_the_block_inverse_at_every_size():
    """A doubling level's join, over blocks of any sizes `n` and `m`: where `X1` and `X2` are the inverses
    of a lower block-triangular system's diagonal blocks, `[[X1, 0], [X2 A21 X1, X2]]` is the inverse of
    `[[X1^-1, 0], [-A21, X2^-1]]`, the system `I - A` restricted to two blocks, whose lower block the
    doubling adds as `(X A) X` masked to it (`recurrence_algebra.doubled`'s merge). From the two-wide
    blocks, each level an inverse by this identity, at every size (its sizes symbols, not numbers)."""
    n, m = sympy.symbols("n m", positive=True, integer=True)
    first, second = sympy.MatrixSymbol("X1", n, n), sympy.MatrixSymbol("X2", m, m)
    below = sympy.MatrixSymbol("A21", m, n)
    joined = sympy.BlockMatrix([[first, sympy.ZeroMatrix(n, m)], [second * below * first, second]])
    system = sympy.BlockMatrix([[sympy.Inverse(first), sympy.ZeroMatrix(n, m)],
                                [-below, sympy.Inverse(second)]])
    product = sympy.block_collapse(joined * system)
    identity = (sympy.Identity(n), sympy.ZeroMatrix(n, m), sympy.ZeroMatrix(m, n), sympy.Identity(m))
    assert [sympy.simplify(product.blocks[i, j]) for i in range(2) for j in range(2)] == list(identity)


class _Rows(list):
    """A chunk's per-token value blockwise (`_blocked`): each block's rows, a SymPy matrix expression; `token`
    the raw token it is (None for a computed one), `read` the raw token whose rows read the state `h`
    (`Q h`), and `inner` whether each block's value is the part a token's decay to it scales elementwise
    (`E_b (.) inner`, a readout along the values)."""

    def __init__(self, blocks, token=None, read=None, inner=False):
        super().__init__(blocks)
        self.token, self.read, self.inner = token, read, inner


class _Out(tuple):
    """A readout along the values blockwise: its decayed part (`_Rows`, `inner`) and its free part."""


def _blocked(expr, chunk: dict, bound: dict):
    """A chunk identity's right side over a chunk of one block or two (`chunk["sizes"]`, symbolic), each
    operand a matrix symbol of its block (`_operand`), every decay relative to its block's start: a token's
    rows decayed forward to it from its block's start `F(X)_b` (`X (.) e^{G'}`) or back `K(X)_b`
    (`X (.) e^{-G'}`), a block's own masked pairs `G(L, R)_b` and spread `Sp_b`, each block's whole decay
    `W_b`, and the second block's rows of a two-block chunk led by the first's whole decay (`e^{G_t} =
    e^{G'_t} W_1`): each constructor's meaning (`linear_scan.Chunk`, `_BLOCKWISE`) at symbolic sizes, a pair
    of rows across the split decayed through it (`F(L)_2 W_1 K(R)_1^T`). A plain loop over computed values
    (the WY form's) is the declared dual form over them."""
    expr = _inner(expr)
    if isinstance(expr, decl.UnboundVarDecl):
        return bound[expr.name]
    if isinstance(expr, decl.LitDecl):
        return expr.value
    args = [_blocked(arg, chunk, bound) for arg in expr.args]
    if _name(expr) == "plain":
        return _blocked(chunk["dual"].rhs, chunk, {**bound, "given": args[0]})
    return _BLOCKWISE[_name(expr)](chunk, *args)


def _lead(chunk: dict) -> list:
    """Each block's lead: the first block's whole decay before the second block of a two-block chunk."""
    k = chunk["k"]
    return [sympy.Identity(k)] if len(chunk["blocks"]) == 1 else [sympy.Identity(k), _operand("W", 1, k, k)]


def _token_blocks(chunk: dict, name: str) -> _Rows:
    """A token's rows, each block's own symbol: a strength's a diagonal."""
    sizes, k, v = chunk["sizes"], chunk["k"], chunk["v"]
    if name == "s":
        return _Rows([_operand("S", b, sizes[b], sizes[b]) for b in chunk["blocks"]], token="s")
    read, out = (v, k) if chunk["values_read"] else (k, v)
    width = {"B": k, "C": k, "U": v, "Q": read, "R": out}
    return _Rows([_operand(name, b, sizes[b], width[name]) for b in chunk["blocks"]], token=name)


def _decayed_to(chunk: dict, rows: _Rows, part: str):
    """`rows` decayed to each token (`times` a chunk's `entering` decay): a read along the values elementwise
    after it (the second block's by the first's whole, `E_2 (.) X = E'_2 (.) (X W_1)`, a diagonal's
    columns); a read along the keys of a scalar's decay, which commutes with it (`(X h) e^{G} = (X e^{G})
    h`); a token's rows forward from its block's start, the second block's led by the first's whole."""
    if part != "entering":
        raise TypeError(f"no blockwise product with the {part} decay")
    lead, sizes, k = _lead(chunk), chunk["sizes"], chunk["k"]
    if rows.inner:
        return _Rows([part * lead[i] for i, part in enumerate(rows)], inner=True)
    if rows.read is not None:
        return _Rows([_operand(f"F{rows.read}", b, sizes[b], k) * lead[i] * chunk["h"]
                      for i, b in enumerate(chunk["blocks"])])
    return _Rows([_operand(f"F{rows.token}", b, sizes[b], k) * lead[i]
                  for i, b in enumerate(chunk["blocks"])])


def _product(chunk: dict, left, right):
    """`left @ right`: a block-lower square of pairs over rows, or rows over the state."""
    if isinstance(left, dict):
        return _Rows([sum((left[i, j] * right[j] for j in range(i + 1)), sympy.ZeroMatrix(*right[i].shape))
                      for i in range(len(chunk["blocks"]))])
    return _Rows([part * right for part in left], read=left.token)


def _gram(chunk: dict, left: _Rows, right: _Rows, strict, between):
    """Each block's own masked pairs, and the pairs across the split decayed through it."""
    del between
    sizes, k, blocks = chunk["sizes"], chunk["k"], chunk["blocks"]
    own = {(i, i): _operand(f"G{left.token}{right.token}{strict}", b, sizes[b], sizes[b])
           for i, b in enumerate(blocks)}
    if len(blocks) == 2:
        across = _operand(f"K{right.token}", 1, sizes[1], k).T
        own[1, 0] = _operand(f"F{left.token}", 2, sizes[2], k) * _lead(chunk)[1] * across
    return own


def _spread(chunk: dict, readers: _Rows, given: _Rows, keys: _Rows, between):
    """A readout along the values' local part, its part before the decay to each token: each block's own,
    the second block's with the first's writes read through the split."""
    del between
    sizes, k, blocks = chunk["sizes"], chunk["k"], chunk["blocks"]
    own = [_linear(f"Sp_{b}", given[i], sizes[b], k) for i, b in enumerate(blocks)]
    if len(blocks) == 2:
        written = _operand(f"K{keys.token}", 1, sizes[1], k) * _lead(chunk)[1]
        own[1] = own[1] + readers[1] * given[0].T * written
    return _Rows(own, inner=True)


def _linear(name: str, given, rows, columns):
    """An operator linear in `given` (a block's own spread over it), applied term by term to `given` expanded,
    each term's scalar factor out: one symbol a term, so equal values however they were formed apply
    alike."""
    expanded = _diagonal(sympy.expand(given))
    total = sympy.ZeroMatrix(rows, columns)
    for term in expanded.args if isinstance(expanded, sympy.MatAdd) else (expanded,):
        factor, rest = term.as_coeff_mmul() if isinstance(term, sympy.MatMul) else (1, term)
        total = total + factor * sympy.MatrixSymbol(f"{name}[{rest}]", rows, columns)
    return total


def _diagonal(expr):
    """`expr`, each whole decay's and strength's transpose the matrix itself (each a diagonal)."""
    def diagonal(e) -> bool:
        return isinstance(e, sympy.Transpose) and getattr(e.arg, "name", "")[:2] in ("W_", "S_")

    return expr.replace(diagonal, lambda e: e.arg)


def _scaled(chunk: dict, values, scale: _Rows):
    """Rows (or a square's) scaled by each token's strength."""
    del chunk
    if isinstance(values, dict):
        return {(i, j): scale[i] * part for (i, j), part in values.items()}
    return _Rows([scale[i] * part for i, part in enumerate(values)])


def _sum(chunk: dict, left, right, sign: int = 1):
    """`left + sign right`: rows of one kind, a readout along the values' decayed part beside its free part
    (`_Out`), or the state."""
    del chunk
    if isinstance(left, _Rows) and isinstance(right, _Rows) and left.inner == right.inner:
        return _Rows([a + sign * b for a, b in zip(left, right, strict=True)], inner=left.inner)
    if isinstance(left, _Rows) and isinstance(right, _Rows) and left.inner and sign == 1:
        return _Out((left, right))
    return left + sign * right


def _written(chunk: dict, keys: _Rows, given: _Rows, to_last):
    """The chunk's writes decayed to its last token: the first block's through the second's whole."""
    del to_last
    sizes, k, blocks = chunk["sizes"], chunk["k"], chunk["blocks"]
    if len(blocks) == 1:
        return _whole(chunk) * _operand(f"K{keys.token}", blocks[0], sizes[blocks[0]], k).T * given[0]
    whole = _operand("W", 2, k, k)
    return (whole * _lead(chunk)[1] * _operand(f"K{keys.token}", 1, sizes[1], k).T * given[0]
            + whole * _operand(f"K{keys.token}", 2, sizes[2], k).T * given[1])


def _solve(chunk: dict, system: dict, given: _Rows) -> _Rows:
    """`(I + system)^-1 given` blockwise: a block-lower system solved block by block."""
    solved = []
    for i, b in enumerate(chunk["blocks"]):
        earlier = sum((system[i, j] * solved[j] for j in range(i)), sympy.ZeroMatrix(*given[i].shape))
        solved.append(sympy.Inverse(sympy.Identity(chunk["sizes"][b]) + system[i, i]) * (given[i] - earlier))
    return _Rows(solved)


_BLOCKWISE = {
    "token": _token_blocks, "entering": lambda chunk: chunk["h"], "prefix": lambda chunk: None,
    "decay": lambda chunk, part, prefix: part, "times": _decayed_to,
    "applied": lambda chunk, h, readers: _Rows([part * h.T for part in readers], inner=True),
    "matmul": _product, "gram": _gram, "spread": _spread, "rows": _scaled, "plus": _sum,
    "minus": lambda chunk, left, right: _sum(chunk, left, right, -1),
    "decayed": lambda chunk, h, whole: _whole(chunk) * h, "written": _written, "solve": _solve,
    "pair": lambda chunk, outputs, state: (outputs, state),
    "outputs": lambda chunk, pair: pair[0], "state": lambda chunk, pair: pair[1],
}
"""Each constructor's blockwise meaning (`_blocked`)."""


def _operand(name: str, block: int, rows, columns):
    """A block's operand, a matrix symbol named by what it is and its block."""
    return sympy.MatrixSymbol(f"{name}_{block}", rows, columns)


def _whole(chunk: dict):
    """The chunk's whole decay: its block's, or the second's after the first's (`W_2 W_1`)."""
    k, blocks = chunk["k"], chunk["blocks"]
    return (_operand("W", blocks[0], k, k) if len(blocks) == 1
            else _operand("W", 2, k, k) * _operand("W", 1, k, k))


def _chunked(identities, blocks: tuple[int, ...], h, values_read: bool) -> dict:
    """A chunk of `blocks` (1, 2 or both) of symbolic sizes for `_blocked`, entering the state `h`."""
    m, n, k, v = sympy.symbols("m n k v", positive=True, integer=True)
    dual = next(rule for rule in identities if _name(_inner(rule.lhs)) == "plain")
    return {"blocks": blocks, "sizes": {1: m, 2: n}, "k": k, "v": v, "h": h, "values_read": values_read,
            "dual": dual}


def _agree(chunked, alone) -> bool:
    """Whether a value of the two-block chunk equals its blocks' own in turn, part by part, each whole decay
    and each strength its own transpose (diagonals)."""
    if isinstance(chunked, _Out):
        return _agree(chunked[0], alone[0]) and _agree(chunked[1], alone[1])
    if isinstance(chunked, _Rows):
        return all(_agree(a, b) for a, b in zip(chunked, alone, strict=True))
    difference = _diagonal(sympy.expand(chunked - alone))
    return sympy.simplify(sympy.expand(difference)) == sympy.ZeroMatrix(*difference.shape)


def _in_turn(first, second):
    """Two blocks' own values, the first's rows then the second's (`_agree`'s right side)."""
    if isinstance(first, _Out):
        return _Out((_in_turn(first[0], second[0]), _in_turn(first[1], second[1])))
    return _Rows([*first, *second], inner=first.inner)


CHUNK_FORMS = [(corrects, per_key, values_read) for corrects in (True, False) for per_key in (False, True)
               for values_read in (False, True)]


@pytest.mark.parametrize(("corrects", "per_key", "values_read"), CHUNK_FORMS,
                         ids=[f"{'wy' if corrects else 'plain'}-{'per_key' if per_key else 'scalar'}"
                              f"{'-values' if values_read else ''}"
                              for corrects, per_key, values_read in CHUNK_FORMS])
def test_every_chunk_identitys_right_side_composes_over_two_blocks_of_any_sizes(corrects, per_key,
                                                                                 values_read):
    """Each chunk identity's right side as declared (`linear_scan._identities`: the dual form, and where
    tokens correct the state the WY form), evaluated blockwise at symbolic sizes (`_blocked`), over a chunk
    of `m + n` tokens equals its first `m` tokens' followed by its last `n` tokens' from the state the
    first leave: outputs and state. The loop its left side states composes so by its definition, so with
    the identity at one token (`test_every_chunk_identity_holds_over_the_reals`, at three, holds it at one:
    each output reads its own and earlier tokens alone), every declared identity holds at every chunk size:
    the induction docs/design/algebra.md §6.9 states. A term the declaration added that vanishes at three
    tokens (a power of a strictly lower system) breaks its composition here."""
    identities = _declared(linear_scan._identities(corrects, not per_key, values_read))
    composed = [rule for rule in identities if _name(_inner(rule.lhs)) in ("plain", "loop")]
    assert len(composed) == 1 + corrects
    for rule in composed:
        chunk = _chunked(identities, (1, 2), None, values_read)
        h0 = sympy.MatrixSymbol("h", chunk["k"], chunk["v"])

        def evaluated(blocks: tuple[int, ...], h, side=rule.rhs):
            held = _chunked(identities, blocks, h, values_read)
            given = _Rows([_operand("U", b, held["sizes"][b], held["v"]) for b in blocks], token="U")
            return _blocked(side, held, {"given": given})

        whole = evaluated((1, 2), h0)
        first = evaluated((1,), h0)
        second = evaluated((2,), first[1])
        assert _agree(whole[0], _in_turn(first[0], second[0])), f"{_name(_inner(rule.lhs))}: outputs"
        assert _agree(whole[1], second[1]), f"{_name(_inner(rule.lhs))}: state"
