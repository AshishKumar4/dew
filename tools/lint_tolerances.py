"""SLOP020's reading of a file, for tools/lint_slop.py: each place it compares floats within a
tolerance, which the accuracy gate (tests/reference_error.py) holds alone.

`comparisons` reads a file's tree, given each node's import table (lint_scopes) and its calls'
definitions (lint_slop's `_Calls`); lint_slop counts what it finds against the inventory. A value's kind
(an error, a difference, a margin, whole numbers, a clock's reading) is read by how the file makes it,
through each scope's bindings and the file's functions, never by its name.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Set
from dataclasses import dataclass, field
from typing import Protocol

import lint_scopes as scopes


class Calls(Protocol):
    """The definitions of a file's functions a call reaches, read where it is made."""

    def reached(self, scope: ast.AST, call: ast.Call) -> Iterable[ast.AST]: ...


TOLERANT: dict[str, dict[str, tuple[int | None, float]] | None] = {
    "numpy.testing.assert_allclose": {"rtol": (2, 1e-7), "atol": (3, 0)},
    "numpy.allclose": {"rtol": (2, 1e-5), "atol": (3, 1e-8)},
    "numpy.isclose": {"rtol": (2, 1e-5), "atol": (3, 1e-8)},
    "jax.numpy.allclose": {"rtol": (2, 1e-5), "atol": (3, 1e-8)},
    "jax.numpy.isclose": {"rtol": (2, 1e-5), "atol": (3, 1e-8)},
    "torch.allclose": {"rtol": (2, 1e-5), "atol": (3, 1e-8)},
    "torch.isclose": {"rtol": (2, 1e-5), "atol": (3, 1e-8)},
    "math.isclose": {"rel_tol": (None, 1e-9), "abs_tol": (None, 0)},
    "torch.testing.assert_close": {"rtol": (None, 1.3e-6), "atol": (None, 1e-5)},
    "pytest.approx": {"rel": (1, 1e-6), "abs": (2, 1e-12)},
    "numpy.testing.assert_almost_equal": None, "numpy.testing.assert_array_almost_equal": None,
    "numpy.testing.assert_approx_equal": None, "numpy.testing.assert_array_max_ulp": None,
    "numpy.testing.assert_array_almost_equal_nulp": None, "dew.accuracy.within": None,
    "chex.assert_trees_all_close": None, "chex.assert_trees_all_equal_comparator": None,
}
"""The APIs that compare floats within a tolerance, as their imports resolve them (lint_scopes), however
they are imported: each with its tolerances, their positional places (None: keyword only) and defaults,
where a call can name them all zero and compare exactly (`_exact`), or None where it cannot.
`assert_array_equal` and `==` compare exactly, and stay."""
TOLERANT_METHODS = frozenset({"assertAlmostEqual", "assertNotAlmostEqual"})
"""unittest's, on whatever test case calls them."""
BOUNDED = frozenset({"numpy.testing.assert_array_less"})
"""An ordering assertion, a tolerance when what it orders is an error (`_erring`)."""
MAGNITUDES = frozenset({"abs", "numpy.abs", "numpy.absolute", "numpy.fabs", "jax.numpy.abs",
                        "jax.numpy.absolute", "torch.abs", "math.fabs", "numpy.linalg.norm",
                        "jax.numpy.linalg.norm", "torch.linalg.norm", "dew.accuracy.distance",
                        "reference_error.distance", "numpy.square", "jax.numpy.square", "torch.square"})
"""What a hand-written tolerance takes of an error before comparing it with a bound."""
DISTANCES = frozenset({"dew.accuracy.distance", "reference_error.distance"})
"""Distances between two values: an error, whatever the values are."""
TOLERANCE_KEYWORDS = frozenset({"atol", "rtol"})
TOLERANCE_NAME = re.compile(r"(?:^|_)(?:tolerance|tol|atol|rtol)s?$", re.IGNORECASE)


def _called(node: ast.Call, table: Mapping[str, str]) -> str:
    """A call's API through the scope's imports, or its bare built-in name (`abs`)."""
    called = scopes.resolved(node.func, table)
    if not called and isinstance(node.func, ast.Name):
        return node.func.id
    return called


@dataclass(frozen=True)
class Imported:
    """What its own file's reading proves of a function another file imports (`support_functions`):
    whether every return is whole, whether one may be an error, and, where every return is a tuple,
    the places of it that may hold an error (else None), as `_Reading` knows a function of the file."""
    whole: bool
    erring: bool
    places: frozenset[int] | None


@dataclass
class _Reading:
    """What SLOP020 knows of one file: each node's import table, its calls' definitions, and the ids of
    its functions that may return an error or a difference, and of those that return only a margin or
    only whole numbers; for each function every return of which is a tuple, the places of it that may
    hold an error; and what is known of a function it imports, by its dotted name (`comparisons`)."""
    tables: Mapping[int, Mapping[str, str]]
    calls: Calls
    erring: set[int]
    differencing: set[int]
    margining: set[int]
    counting: set[int] = field(default_factory=set)
    erring_at: dict[int, set[int]] = field(default_factory=dict)
    imported: Callable[[str], Imported | None] = field(default=lambda called: None)


@dataclass
class _Scope:
    """One lexical scope: its nodes, its bindings (each name with every value the scope binds it to, a
    member write binding its base name unless its key is a constant), the values it writes to each
    `name[constant]`, and the names it binds to an error (the least set its bindings make)."""
    node: ast.AST
    nodes: list[ast.AST]
    bindings: list[tuple[list[str], ast.AST]]
    fields: dict[tuple[str, object], list[ast.AST]] = field(default_factory=dict)
    errors: set[str] = field(default_factory=set)
    magnitudes: set[str] = field(default_factory=set)


def _targets(target: ast.AST, value: ast.AST) -> list[tuple[ast.AST, ast.AST]]:
    """The targets a destructuring assignment writes, each with its own value where the value's shape
    matches the target's (tuples and lists, nested), each place of a call's result where the value is a
    call, else the whole target with the whole value."""
    flat = isinstance(target, ast.Tuple | ast.List) and not any(isinstance(element, ast.Starred)
                                                                for element in target.elts)
    if flat and isinstance(value, ast.Tuple | ast.List) and len(target.elts) == len(value.elts) and not any(
            isinstance(element, ast.Starred) for element in value.elts):
        parts = zip(target.elts, value.elts, strict=True)
        return [pair for inner, part in parts for pair in _targets(inner, part)]
    if flat and isinstance(value, ast.Call):
        places = [ast.Subscript(value=value, slice=ast.Constant(index), ctx=ast.Load())
                  for index in range(len(target.elts))]
        pairs = zip(target.elts, places, strict=True)
        return [pair for inner, place in pairs for pair in _targets(inner, place)]
    return [(target, value)]


def _bound_names(target: ast.AST) -> list[str]:
    """The names an assignment's target binds: a subscript or attribute target binds its base."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Tuple | ast.List):
        return [name for element in target.elts for name in _bound_names(element)]
    if isinstance(target, ast.Starred | ast.Subscript | ast.Attribute):
        return _bound_names(target.value)
    return []


def _returned(value: ast.Call, scope: _Scope, reading: _Reading, kinds: Set[int]) -> bool:
    """Whether a call may return what `kinds` holds the functions of: any definition it may reach."""
    return any(id(definition) in kinds for definition in reading.calls.reached(scope.node, value))


def _imported(value: ast.Call, reading: _Reading) -> Imported | None:
    """What is known of the imported function a call calls (`Imported`), or None."""
    return reading.imported(_called(value, reading.tables.get(id(value), {})))


STRUCTURAL = frozenset({"len", "isinstance", "type", "id", "hash", "str", "repr"})
"""Calls that read a value's structure, not its numbers: what they return carries none of its traits."""
SHAPES = frozenset({"shape", "size", "ndim", "dtype"})
"""Attributes that read a value's structure, likewise."""
CLOCKS = frozenset({"time.perf_counter", "time.monotonic", "time.time", "time.process_time",
                    "time.perf_counter_ns", "time.monotonic_ns", "time.time_ns"})
"""The clocks: a difference of their readings is a duration, which no tolerance compares."""
"""Reductions that keep a magnitude a magnitude, as methods or as numpy's and jax's functions."""


def _known(value: ast.Subscript, scope: _Scope) -> list[ast.AST] | None:
    """Every value a mapping's constant key may hold, or None where the scope does not show them: the
    values the scope writes to that key, else that key's values in the dict literals every binding of the
    mapping is (with no unpacked entry)."""
    container, key = value.value, value.slice
    if not (isinstance(container, ast.Name) and isinstance(key, ast.Constant)):
        return None
    written = scope.fields.get((container.id, key.value))
    if written:
        return list(written)
    bound = [found for names, found in scope.bindings if container.id in names]
    if not bound or not all(isinstance(found, ast.Dict) and None not in found.keys for found in bound):
        return None
    return [found for literal in bound for field_, found in zip(literal.keys, literal.values, strict=True)
            if isinstance(field_, ast.Constant) and field_.value == key.value]


def _placed(value: ast.AST, scope: _Scope, reading: _Reading | None) -> set[int] | None:
    """For a place of a call's result (`f(...)[i]`) where every function the call reaches returns only
    tuples, whether that place may hold an error, as the set of such places; else None."""
    if reading is None or not (isinstance(value, ast.Subscript) and isinstance(value.value, ast.Call)
                               and isinstance(value.slice, ast.Constant)):
        return None
    reached = list(reading.calls.reached(scope.node, value.value))
    if not reached:
        found = _imported(value.value, reading)
        return None if found is None or found.places is None else set(found.places)
    if not all(id(definition) in reading.erring_at for definition in reached):
        return None
    return set().union(*(reading.erring_at[id(definition)] for definition in reached))


def _parts(value: ast.AST, scope: _Scope, reading: _Reading | None = None) -> Iterator[ast.AST]:
    """`value` and what it is computed from, as `ast.walk` gives them, short of what only its structure
    is read from (`STRUCTURAL`, `SHAPES`), reading a mapping's constant key by the values it holds
    (`_known`), and a place of a tuple-returning function's result by that place alone (`_placed`)."""
    pending, expanded = [value], set()
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in STRUCTURAL:
            continue
        if isinstance(node, ast.Attribute) and node.attr in SHAPES:
            continue
        if isinstance(node, ast.Subscript) and (known := _known(node, scope)) is not None:
            key = (node.value.id, node.slice.value)
            if key not in expanded:
                expanded.add(key)
                pending.extend(known)
            continue
        if _placed(node, scope, reading) is not None:
            yield node
            continue
        yield node
        pending.extend(ast.iter_child_nodes(node))


def _clocked(value: ast.AST, scope: _Scope, reading: _Reading, seen: frozenset = frozenset()) -> bool:
    """Whether `value` is a clock's reading by construction: a call of a clock (`CLOCKS`), a name every
    binding of which is one, or a sum or difference of them. Nothing is a time by its name."""
    if isinstance(value, ast.Name) and value.id not in seen:
        bound = [found for names, found in scope.bindings if value.id in names]
        return bool(bound) and all(_clocked(found, scope, reading, seen | {value.id}) for found in bound)
    if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add | ast.Sub):
        return all(_clocked(side, scope, reading, seen) for side in (value.left, value.right))
    return isinstance(value, ast.Call) and _called(value, reading.tables.get(id(value), {})) in CLOCKS


def _bound_to(name: str, scope: _Scope) -> list[ast.AST]:
    return [found for names, found in scope.bindings if name in names]


def _differenced(value: ast.AST, scope: _Scope, reading: _Reading, seen: frozenset = frozenset()) -> bool:
    """Whether `value` is a difference itself: a subtraction (of anything but two clock readings, a
    duration), `float` or a negation of one, a name the scope binds to one, or a call of a function of
    the file that may return one. A value merely computed with a subtraction somewhere (a loss, a
    statistic) is not one."""
    if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Sub):
        return not all(_clocked(side, scope, reading) for side in (value.left, value.right))
    if isinstance(value, ast.UnaryOp):
        return _differenced(value.operand, scope, reading, seen)
    if isinstance(value, ast.Name) and value.id not in seen:
        return any(_differenced(found, scope, reading, seen | {value.id})
                   for found in _bound_to(value.id, scope))
    if isinstance(value, ast.Call):
        if isinstance(value.func, ast.Name) and value.func.id == "float" and len(value.args) == 1:
            return _differenced(value.args[0], scope, reading, seen)
        return _returned(value, scope, reading, reading.differencing)
    return False


def _counted(value: ast.AST, scope: _Scope | None = None, reading: _Reading | None = None,
             seen: frozenset = frozenset()) -> bool:
    """Whether `value` is a whole number, or holds only whole numbers, by an explicit construction: an
    int literal, a truth value, an `int`, `len`, one-argument `round` or `.count` call, `range`, a
    `sum`, `max` or `min` of whole numbers, a container, comprehension or element of them, arithmetic
    that keeps them whole, a name every binding of which is one, or a call of a function of the file, or
    one it imports (`Imported.whole`), every return of which is one (`delivered` returning its token
    counts). Nothing is whole by its name or its use."""
    def counted(part):
        return _counted(part, scope, reading, seen)

    if isinstance(value, ast.Constant):
        return type(value.value) in (int, bool)
    if isinstance(value, ast.Compare):
        return True
    if isinstance(value, ast.UnaryOp):
        return counted(value.operand)
    if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add | ast.Sub | ast.Mult | ast.FloorDiv
                                                   | ast.Mod):
        return counted(value.left) and counted(value.right)
    if isinstance(value, ast.List | ast.Tuple | ast.Set):
        return bool(value.elts) and all(counted(element) for element in value.elts)
    if isinstance(value, ast.GeneratorExp | ast.ListComp | ast.SetComp):
        return counted(value.elt)
    if isinstance(value, ast.Subscript):
        return counted(value.value)
    if isinstance(value, ast.Name):
        if scope is None:
            return False
        if value.id in seen:
            # A name built from itself and whole numbers (`targets[i] += int(count)`) stays whole: the
            # greatest set every binding keeps.
            return True
        bound = [found for names, found in scope.bindings if value.id in names]
        return bool(bound) and all(_counted(found, scope, reading, seen | {value.id}) for found in bound)
    if not isinstance(value, ast.Call):
        return False
    if isinstance(value.func, ast.Attribute) and value.func.attr == "count":
        return True
    if isinstance(value.func, ast.Name) and len(value.args) == 1 and not value.keywords and (
            value.func.id in {"int", "len", "round"}):
        return True
    if isinstance(value.func, ast.Name) and value.func.id in {"sum", "max", "min", "range"} and value.args \
            and not value.keywords:
        return all(counted(argument) for argument in value.args)
    if scope is None or reading is None:
        return False
    found = _imported(value, reading)
    return _returned(value, scope, reading, reading.counting) or (found is not None and found.whole)


def _measured(inner: ast.AST, scope: _Scope, reading: _Reading) -> tuple[bool, bool]:
    """Whether a node takes a magnitude at all (`MAGNITUDES`, a square, a distance), and whether it takes
    one of a difference (`_differenced`) or a distance, which is an error; a magnitude of explicitly whole
    numbers (`abs(int(a) - int(b))`) is a count's distance and neither."""
    if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Pow) and isinstance(
            inner.right, ast.Constant) and inner.right.value == 2:
        operands = [inner.left]
    elif isinstance(inner, ast.Call):
        called = _called(inner, reading.tables.get(id(inner), {}))
        # The value measured: the first argument, by position or as its keyword (`input=`, `x=`, `a=`).
        given = inner.args or [keyword.value for keyword in inner.keywords
                               if keyword.arg in {"input", "x", "a"}]
        if called not in MAGNITUDES | DISTANCES or not given or all(_counted(argument, scope, reading)
                                                                     for argument in given):
            return False, False
        if called in DISTANCES:
            return True, True
        operands = given[:1]
    else:
        return False, False
    if all(_counted(operand, scope, reading) for operand in operands):
        return False, False
    return True, any(_differenced(operand, scope, reading) for operand in operands)


def _erring(value: ast.AST, scope: _Scope, reading: _Reading) -> bool:
    """Whether `value` holds an error: a magnitude, square or norm of a difference, or a distance
    (`_measured`), a name the scope binds to one, what a function of the file may return of one (the
    call resolved to its definitions, `_Calls.reached`) or a function it imports (`Imported.erring`),
    or a place of a tuple one returns."""
    for inner in _parts(value, scope, reading):
        if isinstance(inner, ast.Name) and inner.id in scope.errors:
            return True
        if (places := _placed(inner, scope, reading)) is not None:
            if inner.slice.value in places:
                return True
            continue
        if _measured(inner, scope, reading)[1]:
            return True
        if isinstance(inner, ast.Call) and (_returned(inner, scope, reading, reading.erring) or (
                (found := _imported(inner, reading)) is not None and found.erring)):
            return True
    return False


def _magnitude(value: ast.AST, scope: _Scope, reading: _Reading) -> bool:
    """Whether `value` holds a magnitude of anything (`_measured`) or a name the scope binds to one."""
    return any((isinstance(inner, ast.Name) and inner.id in scope.magnitudes)
               or _measured(inner, scope, reading)[0] for inner in _parts(value, scope, reading))


def _floating(value: ast.AST) -> bool:
    return isinstance(value, ast.Constant) and type(value.value) is float and value.value != 0


def _literal(value: ast.AST) -> bool:
    """Whether `value` is written out: a constant, or arithmetic of constants."""
    return all(isinstance(inner, ast.Constant | ast.BinOp | ast.UnaryOp | ast.operator | ast.unaryop)
               for inner in ast.walk(value))


def _margin(value: ast.AST, scope: _Scope, reading: _Reading, seen: frozenset = frozenset()) -> bool:
    """Whether `value` is evidently a float tolerance: a nonzero float literal, a name or attribute
    spelled as a tolerance (`TOLERANCE_NAME`), a name every binding of which is one, a call of a function
    of the file that returns only one, a negation of one, or one scaled by anything (`1e-12 * rms`)."""
    if _floating(value):
        return True
    if isinstance(value, ast.UnaryOp):
        return _margin(value.operand, scope, reading, seen)
    if isinstance(value, ast.Attribute):
        return bool(TOLERANCE_NAME.search(value.attr))
    if isinstance(value, ast.Name):
        if TOLERANCE_NAME.search(value.id):
            return True
        bound = _bound_to(value.id, scope) if value.id not in seen else []
        return bool(bound) and all(_margin(found, scope, reading, seen | {value.id}) for found in bound)
    if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Mult | ast.Div):
        return any(_margin(side, scope, reading, seen) for side in (value.left, value.right))
    return isinstance(value, ast.Call) and _returned(value, scope, reading, reading.margining)


def _parameters(function: ast.AST) -> list[tuple[list[str], ast.AST]]:
    """A function's parameters bound to their defaults, where they have one."""
    if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
        return []
    arguments = function.args
    positional = [*arguments.posonlyargs, *arguments.args]
    pairs = list(zip(positional[len(positional) - len(arguments.defaults):], arguments.defaults, strict=True))
    pairs += [(argument, default) for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults,
                                                                strict=True) if default is not None]
    return [([argument.arg], default) for argument, default in pairs]


def _read_scope(scope_node: ast.AST, reading: _Reading) -> _Scope:
    """A scope's bindings (a member write binds its base name; a loop's or comprehension's target the
    iterable it takes elements of; a parameter its default) and its names of errors, differences and
    the names it binds to an error, the least set its bindings make: what is known only adds findings."""
    nodes = list(scopes.own(scope_node))
    bindings: list[tuple[list[str], ast.AST]] = _parameters(scope_node)
    fields: dict[tuple[str, object], list[ast.AST]] = {}
    for node in nodes:
        if isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign) and node.value is not None:
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                pairs = [(target, ast.BinOp(target, node.op, node.value))] if isinstance(
                    node, ast.AugAssign) else _targets(target, node.value)
                for written, found in pairs:
                    keyed = isinstance(written, ast.Subscript) and isinstance(written.value, ast.Name) and \
                        isinstance(written.slice, ast.Constant)
                    if keyed:
                        fields.setdefault((written.value.id, written.slice.value), []).append(found)
                    else:
                        bindings.append((_bound_names(written), found))
        elif isinstance(node, ast.NamedExpr):
            bindings.append((_bound_names(node.target), node.value))
        elif isinstance(node, ast.For | ast.AsyncFor | ast.comprehension):
            bindings.append((_bound_names(node.target), node.iter))
    scope = _Scope(scope_node, nodes, bindings, fields)
    while True:
        before = len(scope.errors) + len(scope.magnitudes)
        for bound, value in bindings:
            if _erring(value, scope, reading):
                scope.errors.update(bound)
            if _magnitude(value, scope, reading):
                scope.magnitudes.update(bound)
        if len(scope.errors) + len(scope.magnitudes) == before:
            return scope


def _exact(node: ast.Call, called: str) -> bool:
    """A tolerance call that compares exactly: each of its API's tolerances a literal zero, given by
    keyword or position, or zero by default, and none passed in a pack."""
    declared = TOLERANT.get(called)
    # A packed argument (`*args`, `**options`) may carry any tolerance: not provably exact.
    packed = any(isinstance(argument, ast.Starred) for argument in node.args) or any(
        keyword.arg is None for keyword in node.keywords)
    if declared is None or packed:
        return False
    named = {keyword.arg for keyword in node.keywords}
    for name, (place, default) in declared.items():
        # pytest's approx given only `abs` holds that tolerance alone, its relative one unused.
        if called == "pytest.approx" and name == "rel" and "abs" in named and "rel" not in named \
                and len(node.args) <= place:
            continue
        given = next((keyword.value for keyword in node.keywords if keyword.arg == name), None)
        if given is None and place is not None and len(node.args) > place:
            given = node.args[place]
        value = given.value if isinstance(given, ast.Constant) else (default if given is None else None)
        if value != 0:
            return False
    return True


def _slack(side: ast.AST, scope: _Scope, reading: _Reading) -> bool:
    """Whether a side of an ordering is a value given a float margin: a value that is not written out
    plus or minus a margin (`_margin`: `b + 1e-6`, `b - tolerance`), or scaled by one plus or minus a
    margin (`b * (1 + 1e-6)`). A bound written out (`2.0 - 0.4 - 0.1`) or a ratio (`0.5 * before`) is
    no margin."""
    if not isinstance(side, ast.BinOp):
        return False
    if isinstance(side.op, ast.Add | ast.Sub):
        for base, margin in ((side.left, side.right), (side.right, side.left)):
            if _margin(margin, scope, reading) and not _literal(base):
                return True
        return _slack(side.left, scope, reading) or _slack(side.right, scope, reading)
    if isinstance(side.op, ast.Mult):
        return any(isinstance(factor, ast.BinOp) and isinstance(factor.op, ast.Add | ast.Sub)
                   and _literal(factor) and any(_floating(part) for part in (factor.left, factor.right))
                   for factor in (side.left, side.right))
    return False


def _zero(side: ast.AST) -> bool:
    return isinstance(side, ast.Constant) and type(side.value) in (int, float) and side.value == 0


def _relation(left: ast.AST, right: ast.AST, scope: _Scope, reading: _Reading) -> str | None:
    """What one relation of an ordering compares within a tolerance, if anything. Against zero it asks a
    value's sign, whether a gradient moved or a loss fell, and is a tolerance only where a margin is
    folded into the value (`_slack`: `abs(a - b) - 1e-6 <= 0`). Otherwise: an error against any bound,
    a magnitude of anything against a sub-unit margin, a difference itself against a float tolerance
    (`_differenced`, `_margin`; a difference of positions against a length is index arithmetic, and one
    of two clock readings a duration), each read through the scope's bindings and the file's functions,
    or a float within a slack."""
    for side, other in ((left, right), (right, left)):
        if _zero(other):
            return "a float compared within a slack" if _slack(side, scope, reading) else None
    for side, other in ((left, right), (right, left)):
        if _erring(side, scope, reading):
            return "an error compared with a bound"
        if _differenced(side, scope, reading) and _margin(other, scope, reading):
            return "a difference compared with a bound"
        # A magnitude of anything held near zero (`abs(residual) < 1e-3`), by a margin below one or named
        # as a tolerance; against a domain threshold (a clamp at 20, an activation bound) it is none.
        unit = isinstance(other, ast.Constant) and isinstance(other.value, float) and abs(other.value) >= 1
        if not unit and _magnitude(side, scope, reading) and _margin(other, scope, reading):
            return "an error compared with a bound"
    if _slack(left, scope, reading) or _slack(right, scope, reading):
        return "a float compared within a slack"
    return None


def _same(one: ast.AST, other: ast.AST) -> bool:
    return ast.dump(one) == ast.dump(other)


def _fractional(bound: ast.AST, scope: _Scope, reading: _Reading) -> bool:
    """Whether a bound evidently holds a fraction: a nonzero float literal, a true division or a margin
    (`_margin`: `tol`, `eps`) anywhere in it. Whole bounds (`16 <= hits <= 48`, `3 * batches`) bound a
    count, as a count of unknown provenance is most often one."""
    return any(_floating(part) or (isinstance(part, ast.BinOp) and isinstance(part.op, ast.Div))
               or (isinstance(part, ast.Name | ast.Attribute) and _margin(part, scope, reading))
               for part in ast.walk(bound))


def _written(bound: ast.AST, scope: _Scope) -> ast.AST | None:
    """A bound written out (`_literal`), as it is or through a name the scope binds only to it."""
    if _literal(bound):
        return bound
    if isinstance(bound, ast.Name):
        found = _bound_to(bound.id, scope)
        if len(found) == 1 and _literal(found[0]):
            return found[0]
    return None


def _number(bound: ast.AST | None) -> float | None:
    """A written-out bound's value (`1 / 8`, `-1.0`), or None where it is not arithmetic of numbers."""
    if isinstance(bound, ast.Constant) and type(bound.value) in (int, float):
        return float(bound.value)
    if isinstance(bound, ast.UnaryOp) and isinstance(bound.op, ast.USub | ast.UAdd):
        operand = _number(bound.operand)
        return None if operand is None else -operand if isinstance(bound.op, ast.USub) else operand
    if isinstance(bound, ast.BinOp):
        left, right = _number(bound.left), _number(bound.right)
        operations = {ast.Add: float.__add__, ast.Sub: float.__sub__, ast.Mult: float.__mul__,
                      ast.Div: float.__truediv__}
        operation = operations.get(type(bound.op))
        if left is not None and right is not None and operation is not None and (
                right != 0 or not isinstance(bound.op, ast.Div)):
            return operation(left, right)
    return None


def _band(lower: ast.AST, value: ast.AST, upper: ast.AST, scope: _Scope, reading: _Reading) -> bool:
    """Whether `lower <= value <= upper` holds a float in a band around a target: a value neither whole
    nor a duration, between bounds one of which holds a fraction (`_fractional`), both written out, as
    they are or through names (`2.25 <= ratio <= 3.75`, `-1e-6 <= residual <= 1e-6`), a target less
    and plus one width (`c - w <= value <= c + w`), or a target scaled by two written-out factors
    (`0.75 * c <= value <= 1.25 * c`). A bound at a domain's edge leaves one side alone, which is no
    band: zero either way asks a sign, and 1 above or -1 below is a share's or a cosine's edge (the
    busiest of eight shares in [1 / 8, 1], a cosine in [-1, 1]). A duration's floor and timeout time
    it, and a bound that is a reading (a task's wall time) is no target."""
    low, high = _written(lower, scope), _written(upper, scope)
    bounds = [bound for bound in (lower, upper, low, high) if bound is not None]
    if _counted(value, scope, reading) or _clocked(value, scope, reading):
        return False
    if _number(low) in (0.0, -1.0) or _number(high) in (0.0, 1.0):
        return False
    if not any(_fractional(bound, scope, reading) for bound in bounds):
        return False
    if low is not None and high is not None:
        return True
    if not (isinstance(lower, ast.BinOp) and isinstance(upper, ast.BinOp)):
        return False
    if isinstance(lower.op, ast.Sub) and isinstance(upper.op, ast.Add):
        return (_same(lower.left, upper.left) and _same(lower.right, upper.right)) or (
            _same(lower.left, upper.right) and _same(lower.right, upper.left))
    if isinstance(lower.op, ast.Mult) and isinstance(upper.op, ast.Mult):
        return any(_same(target, other) and _literal(factor) and _literal(scale)
                   for target, factor in ((lower.left, lower.right), (lower.right, lower.left))
                   for other, scale in ((upper.left, upper.right), (upper.right, upper.left)))
    return False


def _chained_band(node: ast.Compare, scope: _Scope, reading: _Reading) -> bool:
    """Whether a chain bounds a value on both sides in a band (`_band`), written either way round."""
    sides = [node.left, *node.comparators]
    for place, (first, second) in enumerate(zip(node.ops[:-1], node.ops[1:], strict=True)):
        lower, value, upper = sides[place:place + 3]
        if all(isinstance(op, ast.Lt | ast.LtE) for op in (first, second)) and _band(
                lower, value, upper, scope, reading):
            return True
        if all(isinstance(op, ast.Gt | ast.GtE) for op in (first, second)) and _band(
                upper, value, lower, scope, reading):
            return True
    return False


def _conjunction(node: ast.AST) -> bool:
    return (isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And)) or (
        isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd))


def _bounds(node: ast.AST) -> Iterator[tuple[ast.AST, str, ast.AST, ast.Compare]]:
    """Each one-relation ordering a conjunction holds (`and`, or `&` of masks), as (value, "lower" or
    "upper", bound, the ordering) for both its sides."""
    assert isinstance(node, ast.BoolOp | ast.BinOp)
    parts = node.values if isinstance(node, ast.BoolOp) else [node.left, node.right]
    for part in parts:
        if _conjunction(part):
            yield from _bounds(part)
            continue
        if not (isinstance(part, ast.Compare) and len(part.ops) == 1):
            continue
        left, op, right = part.left, part.ops[0], part.comparators[0]
        if isinstance(op, ast.Gt | ast.GtE):
            left, right = right, left
        elif not isinstance(op, ast.Lt | ast.LtE):
            continue
        yield right, "lower", left, part
        yield left, "upper", right, part


def _conjoined_bands(node: ast.AST, scope: _Scope, reading: _Reading) -> int:
    """How many values a conjunction bounds on both sides in a band (`_band`): `lo <= x and x <= hi`,
    or `(x > lo) & (x < hi)`, each counted once. A band one of whose orderings is found a tolerance on
    its own (`_relation`) is found there, not again."""
    found: dict[str, dict[str, list[tuple[ast.AST, ast.Compare]]]] = {}
    values: dict[str, ast.AST] = {}
    for value, side, bound, part in _bounds(node):
        key = ast.dump(value)
        values.setdefault(key, value)
        found.setdefault(key, {"lower": [], "upper": []})[side].append((bound, part))

    def alone(part: ast.Compare) -> bool:
        return not _relation(part.left, part.comparators[0], scope, reading)

    return sum(any(_band(lower, values[key], upper, scope, reading) and alone(low) and alone(high)
                   for lower, low in sides["lower"] for upper, high in sides["upper"])
               for key, sides in found.items())


def _read(tree: ast.Module, tables: Mapping[int, Mapping[str, str]], calls: Calls,
          imported: Callable[[str], Imported | None]) -> tuple[_Reading, list[ast.AST]]:
    """What SLOP020 knows of a file's functions (`_Reading`), and the file's scopes."""
    reading = _Reading(tables, calls, set(), set(), set(), imported=imported)
    nodes = [tree, *(node for node in ast.walk(tree) if isinstance(node, scopes.SCOPES))]
    functions = [node for node in nodes if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)]
    # Which functions return only tuples is read first, from their returns' shapes alone, so a call of one
    # is read by place from the start and the places only grow: the least fixed point, reached.
    for function in functions:
        returned = [node.value for node in scopes.own(function) if isinstance(node, ast.Return)]
        if returned and all(isinstance(value, ast.Tuple) for value in returned):
            reading.erring_at[id(function)] = set()
    # A function may return an error or a difference where any of its returns is one; it returns a
    # margin, or whole numbers, only where every one is.
    kinds = ((reading.erring, _erring, any), (reading.differencing, _differenced, any),
             (reading.margining, _margin, all), (reading.counting, _counted, all))
    while True:
        grew = False
        for function in functions:
            scope = _read_scope(function, reading)
            returned = [node.value for node in scope.nodes
                        if isinstance(node, ast.Return) and node.value is not None]
            for held, holds, each in kinds:
                if id(function) not in held and returned and each(holds(value, scope, reading)
                                                                  for value in returned):
                    held.add(id(function))
                    grew = True
            if id(function) in reading.erring_at:
                places = {place for value in returned for place, element in enumerate(value.elts)
                          if _erring(element, scope, reading)}
                if not places <= reading.erring_at[id(function)]:
                    reading.erring_at[id(function)] |= places
                    grew = True
        if not grew:
            break
    return reading, nodes


def support_functions(tree: ast.Module, tables: Mapping[int, Mapping[str, str]], calls: Calls
                      ) -> dict[str, Imported]:
    """What a file's own reading proves of each of its module-level functions, by name, read for a file
    that imports one (`Imported`); what they call through an import of their own is not followed."""
    reading, _ = _read(tree, tables, calls, lambda called: None)
    places = {key: frozenset(value) for key, value in reading.erring_at.items()}
    return {node.name: Imported(whole=id(node) in reading.counting, erring=id(node) in reading.erring,
                                places=places.get(id(node)))
            for node in tree.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)}


def comparisons(tree: ast.Module, tables: Mapping[int, Mapping[str, str]], calls: Calls,
                imported: Callable[[str], Imported | None] = lambda called: None
                ) -> Iterator[tuple[ast.AST, str]]:
    """Each place a file compares floats within a tolerance, and what it is (its nodes' import tables
    `tables`, its calls' definitions `calls`, and what its own file proves of a function it imports,
    `imported` of its dotted name): a tolerance API not told to
    compare exactly (`_exact`), `atol` or `rtol` handed to any call, an ordering relation that compares
    within a tolerance (`_relation`), and a name bound to a tolerance. What is known of a value only
    adds findings: nothing is exempted as whole or as a time, so a bound on a count, a time or a mask
    of positions is written as a plain ordering (lo <= x <= hi, or x - y < w for a window). SLOP020
    reads conventional code: it does not prove what an obfuscated one computes (a shadowed built-in, a
    mutated alias), which review does."""
    reading, nodes = _read(tree, tables, calls, imported)
    # A band is a test's claim: what an assertion holds, not a schedule's window in a test's helper.
    asserted = {id(part) for statement in ast.walk(tree) if isinstance(statement, ast.Assert)
                for part in ast.walk(statement.test)}
    # The conjunctions a band was found in, the outermost first, so one held within it is not found again.
    conjoined: set[int] = set()
    for scope_node in nodes:
        scope = _read_scope(scope_node, reading)
        for node in scope.nodes:
            if isinstance(node, ast.Call):
                called = _called(node, tables.get(id(node), {}))
                method = isinstance(node.func, ast.Attribute) and node.func.attr in TOLERANT_METHODS
                if (called in TOLERANT or method) and not _exact(node, called):
                    yield node, f"`{called or node.func.attr}`"
                elif called in BOUNDED and node.args and _erring(node.args[0], scope, reading):
                    yield node, f"`{called}` of an error"
                elif called not in TOLERANT and any(keyword.arg in TOLERANCE_KEYWORDS
                                                    for keyword in node.keywords):
                    yield node, "a call given `atol`/`rtol`"
            elif isinstance(node, ast.Compare):
                sides = [node.left, *node.comparators]
                relations = [_relation(left, right, scope, reading)
                             for left, op, right in zip(sides[:-1], node.ops, sides[1:], strict=True)
                             if isinstance(op, ast.Lt | ast.LtE | ast.Gt | ast.GtE)]
                what = next((relation for relation in relations if relation), None)
                if what:
                    yield node, what
                elif id(node) in asserted and _chained_band(node, scope, reading):
                    yield node, "a float held in a band around a target"
            elif _conjunction(node) and id(node) in asserted and id(node) not in conjoined:
                inner = [part for part in ast.walk(node) if isinstance(part, ast.BoolOp | ast.BinOp)]
                bands = _conjoined_bands(node, scope, reading)
                if bands:
                    conjoined.update(id(part) for part in inner)
                for _ in range(bands):
                    yield node, "a float held in a band around a target"
            elif isinstance(node, ast.Assign | ast.AnnAssign):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                named = [target for target in targets
                         if isinstance(target, ast.Name) and TOLERANCE_NAME.search(target.id)]
                if named:
                    yield node, f"a tolerance, `{named[0].id}`"
