"""SLOP024's reading of a file, for tools/lint_slop.py: a JAX array divided by a compile-time constant no
power of two divides exactly by, which XLA folds into a product with the constant's rounded reciprocal (rows
1402, 1403), so the quotient is not correctly rounded; `dew.nn.precision.divided` divides correctly. A
constant is a literal, a named constant of the module or the library's, a number of math or numpy, a dtype's
facts, or a pure function of constants (`_constant`); an array is what the module proves one (`_array`).
"""

from __future__ import annotations

import ast
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import lint_scopes as scopes

ARRAY_MODULES = ("jax.numpy", "jax.lax", "jax.nn", "jax.random", "jax.scipy")
"""The modules whose functions return JAX arrays (but `ARRAY_FACTS`), which SLOP024 proves an operand by."""
ARRAY_FACTS = {"dtype", "finfo", "iinfo", "issubdtype", "result_type", "promote_types", "can_cast", "shape",
               "ndim", "size", "isdtype"}
"""The functions of `ARRAY_MODULES` that return no array: a dtype, its facts, a shape or a count."""
ARRAY_TYPES = {"jax.Array", "jax.numpy.ndarray"}
"""The annotations that make a parameter an array."""


def _array(node: ast.expr, table: Mapping[str, str], arrays: set[str]) -> bool:
    """Whether SLOP024 proves `node` a JAX array, its names read as the imports `table` gives them
    (`lint_scopes.scoped`): a call of a function of `ARRAY_MODULES` but its facts, a method or subscript of an
    array, arithmetic with one, or a name in `arrays` (a parameter annotated as one)."""
    if isinstance(node, ast.Call):
        called = scopes.resolved(node.func, table)
        module, _, function = called.rpartition(".")
        if module in ARRAY_MODULES or any(module.startswith(f"{each}.") for each in ARRAY_MODULES):
            return function not in ARRAY_FACTS
        return (isinstance(node.func, ast.Attribute) and node.func.attr not in ("item", "tolist")
                and _array(node.func.value, table, arrays))
    if isinstance(node, ast.Subscript):
        return _array(node.value, table, arrays)
    if isinstance(node, ast.BinOp):
        return _array(node.left, table, arrays) or _array(node.right, table, arrays)
    if isinstance(node, ast.UnaryOp):
        return _array(node.operand, table, arrays)
    return isinstance(node, ast.Name) and node.id in arrays


CONSTANT_NUMBERS = {f"{root}.{name}": value for root in ("math", "numpy", "jax.numpy")
                    for name, value in (("pi", math.pi), ("e", math.e), ("tau", math.tau))}
"""The modules' number constants a compile-time constant may read."""
CONSTANT_FUNCTIONS = {
    "float": float, "int": int, "abs": abs, "min": min, "max": max, "round": round,
    **{f"math.{name}": getattr(math, name)
       for name in ("sqrt", "log", "log2", "log10", "exp", "pow", "fabs")},
}
"""The pure scalar functions a compile-time constant may call."""
WIDE_DTYPES = ("float16", "float32", "float64")
NARROW_DTYPES = ("bfloat16", "float8_e4m3fn", "float8_e5m2", "float4_e2m1fn")
CONSTANT_DTYPES = WIDE_DTYPES + NARROW_DTYPES
"""The floating types a constant may be cast to (`jnp.float32(c)`) or read the limits of (`jnp.finfo(...)`),
under numpy, jax.numpy or ml_dtypes."""
FINFO_FIELDS = ("max", "min", "eps", "tiny", "smallest_normal", "resolution")
"""The limits of a floating type a constant may read."""
FINFOS = {"numpy.finfo", "jax.numpy.finfo", "ml_dtypes.finfo"}
"""The functions that give a floating type's limits."""
OPERATORS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
             ast.Div: lambda a, b: a / b, ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b,
             ast.Pow: lambda a, b: a ** b, ast.LShift: lambda a, b: a << b, ast.RShift: lambda a, b: a >> b}
"""The arithmetic a compile-time constant may do."""
DIVIDES = {"jax.numpy.divide", "jax.numpy.true_divide", "jax.lax.div"}
"""The functions that divide their first argument by their second, as `/` does."""


def _dtype(path: str):
    """The floating type a dotted `path` names (`jnp.float32`, `ml_dtypes.bfloat16`), or None."""
    import ml_dtypes
    import numpy as np

    root, _, name = path.rpartition(".")
    if root not in ("numpy", "jax.numpy", "ml_dtypes") or name not in CONSTANT_DTYPES:
        return None
    return np.dtype(getattr(ml_dtypes, name) if name in NARROW_DTYPES else name)


def _number(value) -> bool:
    import numpy as np

    return (isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_)))


@dataclass(frozen=True)
class Constants:
    """How `_constant` reads a name: the scope's imports (`table`), the module's constants (`own`), the
    names its enclosing scopes bind (`shadowed`, never a constant) and the library's sources (`root`)."""

    table: Mapping[str, str]
    own: Mapping[str, object]
    shadowed: frozenset[str]
    root: Path | None
    bound: frozenset[str] = frozenset()
    """Every name the module binds, which no builtin's spelling then names."""


def _constant(node: ast.expr, scope: Constants):
    """The value of `node` where it is a compile-time constant, else None, read by its structure alone and
    never by running the code: a number, arithmetic of constants (`OPERATORS`), a module-level constant of
    this module or of another of the library's (`_module_constants`, by a name or an attribute chain), a
    number of math, numpy or jax.numpy (`CONSTANT_NUMBERS`), a cast of a constant to a floating type or one
    of its limits (`CONSTANT_DTYPES`, `FINFO_FIELDS`), or a pure scalar function of constants
    (`CONSTANT_FUNCTIONS`). A name an enclosing scope binds is none."""
    if isinstance(node, ast.Constant):
        return node.value if _number(node.value) else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _constant(node.operand, scope)
        return None if value is None else -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        left, right = _constant(node.left, scope), _constant(node.right, scope)
        integral = isinstance(node.op, (ast.LShift, ast.RShift))
        dividing = isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod))
        if (left is None or right is None or (dividing and not right)
                or (integral and not (isinstance(left, int) and isinstance(right, int)))):
            return None
        value = OPERATORS[type(node.op)](left, right)
        return value if _number(value) and math.isfinite(float(value)) else None
    if isinstance(node, ast.Name) and node.id in scope.shadowed:
        return None
    if isinstance(node, ast.Name) and node.id in scope.own:
        return scope.own[node.id]
    if isinstance(node, (ast.Name, ast.Attribute)):
        return _named(node, scope)
    if isinstance(node, ast.Call) and not node.keywords:
        return _called(node, scope)
    return None


def _named(node: ast.Name | ast.Attribute, scope: Constants):
    """A name or attribute chain's constant (`_constant`): a library module's, a module's number, or a
    floating type's limit (`jnp.finfo(dtype).max`)."""
    if (isinstance(node, ast.Attribute) and node.attr in FINFO_FIELDS and isinstance(node.value, ast.Call)
            and len(node.value.args) == 1 and scopes.resolved(node.value.func, scope.table) in FINFOS
            and (dtype := _dtype(scopes.resolved(node.value.args[0], scope.table))) is not None):
        import ml_dtypes

        return getattr(ml_dtypes.finfo(dtype), node.attr)
    root = node
    while isinstance(root, ast.Attribute):
        root = root.value
    if isinstance(root, ast.Name) and root.id in scope.shadowed:
        return None
    path = scopes.resolved(node, scope.table)
    if path in CONSTANT_NUMBERS:
        return CONSTANT_NUMBERS[path]
    if path.startswith("dew.") and scope.root is not None:
        source = _source_of(path, scope.root)
        if source is not None:
            return _module_constants(source[0]).get(source[1])
    return None


def _called(node: ast.Call, scope: Constants):
    """A call's constant (`_constant`): a cast of a constant to a floating type, or a pure scalar function
    of constants."""
    arguments = [_constant(argument, scope) for argument in node.args]
    if any(argument is None for argument in arguments):
        return None
    if isinstance(node.func, ast.Name) and node.func.id in scope.shadowed:
        return None
    # A bare name is a builtin only where nothing of the module or around it binds it.
    builtin = (isinstance(node.func, ast.Name) and node.func.id not in scope.bound
               and node.func.id not in scope.shadowed)
    path = scopes.resolved(node.func, scope.table) or (node.func.id if builtin else "")
    if (dtype := _dtype(path)) is not None and len(arguments) == 1:
        # A scalar of its type, so arithmetic with it rounds as the module's own does.
        return dtype.type(arguments[0])
    function = CONSTANT_FUNCTIONS.get(path)
    if function is None or (path.startswith("math.") and any(argument <= 0 for argument in arguments[:1])
                            and path in ("math.sqrt", "math.log", "math.log2", "math.log10")):
        return None
    value = function(*arguments)
    return value if _number(value) else None


_READING: dict[Path, dict[str, object]] = {}
"""The modules whose constants are being read, each with those read so far: a module of an import cycle reads
another's as Python's import would see it, partly initialized."""
_PARTIAL: list[int] = [0]
"""How many reads of a module still being read there have been: a module whose reading met one is not kept
(`_READ`), as another order of reading would see more of it."""
_READ: dict[Path, dict[str, object]] = {}
"""Each module's constants once read whole (`_module_constants`)."""


def _module_constants(path: Path) -> dict[str, object]:
    """The module-level names the source at `path` binds exactly once (`lint_scopes.binders` over its own
    nodes, an augmented assignment or a conditional's binding counted) to a compile-time constant
    (`_constant`), each its value, read in the order bound; within an import cycle, those of a module still
    being read that it has bound so far."""
    path = path.resolve()
    if path in _READ:
        return _READ[path]
    if path in _READING:
        _PARTIAL[0] += 1
        return _READING[path]
    if not path.is_file():
        return {}
    constants: dict[str, object] = {}
    _READING[path] = constants
    partial = _PARTIAL[0]
    try:
        tree = ast.parse(path.read_text())
        root = next((parent for parent in path.parents if parent.name == "src"), None)
        name = ".".join(path.relative_to(root).with_suffix("").parts) if root else path.stem
        parent = scopes.package_of(name.removesuffix(".__init__"), path.name == "__init__.py")
        table = dict(next(iter(scopes.scoped(tree.body, {}, parent)), (None, {}))[1]) if tree.body else {}
        counts = scopes.binders(scopes.nodes_of(tree.body))
        bound = frozenset(counts)
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target, value = node.targets[0], node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                target, value = node.target, node.value
            else:
                continue
            if isinstance(target, ast.Name) and len(counts.get(target.id, [])) == 1:
                found = _constant(value, Constants(table, constants, frozenset(), root, bound))
                if found is not None:
                    constants[target.id] = found
    finally:
        del _READING[path]
    if _PARTIAL[0] == partial:
        _READ[path] = constants
    return constants


def _source_of(dotted: str, root: Path) -> tuple[Path, str] | None:
    """The source file under `root` (a checkout's src/) a dotted `dew.<module>.<name>` is a module-level
    name of, and the name; None where no such file holds it."""
    module, _, name = dotted.rpartition(".")
    for candidate in (root / Path(*module.split(".")).with_suffix(".py"),
                      root / Path(*module.split(".")) / "__init__.py"):
        if candidate.is_file():
            return candidate, name
    return None


def _rounded(value) -> bool:
    """Whether a constant divisor `value` is one no power of two divides exactly by: its reciprocal is no
    float, so XLA's product with it rounds twice."""
    return (value is not None and math.isfinite(float(value)) and value != 0
            and math.frexp(float(value))[0] not in (0.5, -0.5))


def _local_arrays(function: ast.AST, table: Mapping[str, str], arrays: set[str]) -> set[str]:
    """`arrays` and every name `function`'s own scope binds only by assignments of an expression `_array`
    proves (`lint_scopes.binders`: a name also bound otherwise, by a parameter, a loop or an augmented
    assignment, is none), to a fixed point: what a name holds read as what each of its bindings gives it."""
    assigned = {id(target): node.value for node in scopes.own(function) if isinstance(node, ast.Assign)
                for target in node.targets if isinstance(target, ast.Name)}
    parameters = scopes.arguments(function) if not isinstance(function, ast.ClassDef) else ()
    bound = scopes.binders(list(scopes.own(function)), parameters)
    candidates = {name: [assigned[id(node)] for node in nodes] for name, nodes in bound.items()
                  if name not in arrays and all(id(node) in assigned for node in nodes)}
    found = set(arrays)
    while True:
        more = {name for name, values in candidates.items() if name not in found
                and all(_array(value, table, found) for value in values)}
        if not more:
            return found
        found |= more


def _shadowing(tree: ast.Module) -> tuple[dict[int, frozenset[str]], dict[int, frozenset[str]]]:
    """Each node's names its enclosing scopes bind as Python resolves a name there (no module constant
    reaches one), and those of them its comprehensions bind within its own function (no array its function
    assigns reaches one). A function's or a class's statement's decorators, defaults and bases read the
    scope around it (`lint_scopes.outside`); a class's names reach its own body, never a function in it; a
    comprehension's targets reach all of it but its first iterable."""
    shadowed: dict[int, frozenset[str]] = {}
    comprehended: dict[int, frozenset[str]] = {}

    def visit(node: ast.AST, names: frozenset[str], functions: frozenset[str], local: frozenset[str]) -> None:
        shadowed[id(node)], comprehended[id(node)] = names, local
        if isinstance(node, scopes.SCOPES):
            parameters = scopes.arguments(node) if not isinstance(node, ast.ClassDef) else ()
            inner = functions | frozenset(scopes.binders(list(scopes.own(node)), parameters))
            for child in ast.iter_child_nodes(node):
                if isinstance(node, ast.ClassDef):
                    # A class's names reach its body's statements, not the functions they define.
                    visit(child, inner, functions, local)
                else:
                    visit(child, inner, inner, frozenset())
            # What the statement evaluates around it (decorators, defaults, annotations, bases), visited
            # again in the enclosing scope, as Python evaluates it.
            for part in scopes.outside(node):
                visit(part, names, functions, local)
            return
        if isinstance(node, scopes.COMPREHENSIONS):
            bound = frozenset(name.id for name in scopes.targets(node))
            first = node.generators[0].iter
            for child in ast.iter_child_nodes(node):
                visit(child, names | bound, functions | bound, local | bound)
            visit(first, names, functions, local)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, names, functions, local)

    visit(tree, frozenset(), frozenset(), frozenset())
    return shadowed, comprehended


def divisions(path: Path, tree: ast.Module, parent: str) -> Iterator[tuple[int, int, str]]:
    """SLOP024: a JAX array (`_array`, or a name its function binds only to one) divided by a compile-time
    constant no power of two divides exactly by (`_constant`, `_rounded`): a literal, a named constant of
    this module or the library's, a number of math or numpy, or a pure function of constants; by `/`, `/=`,
    or jax.numpy's or jax.lax's division. XLA folds each into a product with the constant's rounded
    reciprocal (rows 1402, 1403). Each name is read in its own scope (`lint_scopes.scoped`), `parent` the
    module's package for its relative imports. Each finding's line, column and message."""
    scoped = list(scopes.scoped(tree.body, {}, parent))
    tables = {id(node): table for node, table in scoped}
    own = _module_constants(path)
    root = next((parent for parent in path.resolve().parents if parent.name == "src"), None)
    bound = frozenset(scopes.binders(scopes.nodes_of(tree.body)))
    shadowed, comprehended = _shadowing(tree)
    arrays: dict[int, set[str]] = {}
    for function in ast.walk(tree):
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            table = tables.get(id(function), {})
            annotated = set() if isinstance(function, ast.Lambda) else {
                name for name, annotation in scopes.annotations(function)
                if scopes.resolved(annotation, table) in ARRAY_TYPES}
            held = _local_arrays(function, table, annotated)
            for node in scopes.own(function):
                arrays[id(node)] = held
    for node, table in scoped:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            numerator, divisor = node.left, node.right
        elif isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Div):
            numerator, divisor = node.target, node.value
        elif (isinstance(node, ast.Call) and len(node.args) == 2
              and scopes.resolved(node.func, table) in DIVIDES):
            numerator, divisor = node.args
        else:
            continue
        value = _constant(divisor, Constants(table, own, shadowed.get(id(node), frozenset()), root, bound))
        held = arrays.get(id(node), set()) - comprehended.get(id(node), frozenset())
        if _rounded(value) and _array(numerator, table, held):
            yield (node.lineno, node.col_offset + 1,
                   f"divides a JAX array by the constant {ast.unparse(divisor)} = {float(value)!r}, which "
                   "XLA computes as a product with its rounded reciprocal, not correctly rounded; divide "
                   "with precision.divided")
