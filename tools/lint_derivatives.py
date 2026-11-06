"""SLOP028's reading of a file, for tools/lint_slop.py: every custom derivative of the package's source
declares, at its seam, the contract that holds it (row 4411). A seam is a function a custom derivative
decorates (JAX's custom_jvp and custom_vjp, Dew's declaring functions of `dew.nn.derivatives`, as a
decorator, a call or within functools.partial), a name assigned one's making call, or a call of
`generation.linear_tangent`; its contract is `contracted(test)` of `dew.nn.derivatives`, a decorator of
the function or the call wrapping the making call, `test` the name of exactly one test function of
tests/test_contracts_*.py (`contract_support.assert_contract`: every order finite against its float64
truth and an independent plain formula). Blind spots: a derivative made through a name bound to a maker,
or by a function of the file's own that returns one.
"""

from __future__ import annotations

import ast
import collections
import functools
from collections.abc import Iterator
from pathlib import Path

DERIVATIVE_MAKERS = frozenset({"custom_jvp", "custom_vjp", "surrogate", "declared_derivative",
                               "declared_tangent", "linear_tangent"})
"""What makes a custom derivative (SLOP028): JAX's own, and Dew's (`dew.nn.derivatives`; a declaring builder's
kind its first argument; `generation.linear_tangent`, whose call makes one), as a decorator, a call, or
within functools.partial."""

def _maker_of(node: ast.expr, given: tuple = ()) -> str | None:
    """The custom derivative `node`, a decorator or a call, makes (`DERIVATIVE_MAKERS`), or None: a
    declaring builder's with the kind its first argument constructs (`declared_derivative:Surrogate`), as
    the algebra pass reads it (docs/design/algebra.md §1.3); `given`, a partial's arguments past the
    function."""
    if isinstance(node, ast.Call):
        name = _call_name(node.func)
        if name == "partial" and node.args:
            return _maker_of(node.args[0], tuple(node.args[1:]))
        if name in ("declared_derivative", "declared_tangent"):
            return _maker_of(node.func, tuple(node.args))
        return _maker_of(node.func)
    name = _call_name(node)
    if name in ("declared_derivative", "declared_tangent"):
        kind = given[0] if given else None
        return f"{name}:{_call_name(kind.func) if isinstance(kind, ast.Call) else None}"
    return name if name in DERIVATIVE_MAKERS - {"linear_tangent"} else None


def _call_name(node: ast.expr) -> str | None:
    return getattr(node, "attr", None) or getattr(node, "id", None)


def derivative_seams(relative: str, tree: ast.Module) -> Iterator[tuple[str, str, ast.AST, ast.expr | None]]:
    """`relative`'s custom derivatives: each seam's `relative::qualified name`, what makes it, the node to
    report, and its declaration (`contracted(...)`), if any: a decorated function's decorator, or the call
    wrapping an assigned maker's call or a linear_tangent call."""
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

    def declared(node: ast.expr) -> ast.expr | None:
        """The `contracted(...)` call wrapping the call `node` (`contracted(...)(node)`), if any."""
        outer = parents.get(node)
        if (isinstance(outer, ast.Call) and outer.args == [node] and isinstance(outer.func, ast.Call)
                and _call_name(outer.func.func) == "contracted"):
            return outer.func
        return None

    def walk(node: ast.AST, scope: list[str]) -> Iterator[tuple[str, str, ast.AST, ast.expr | None]]:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not isinstance(child, ast.ClassDef):
                    makers = [maker for maker in map(_maker_of, child.decorator_list) if maker]
                    if makers:
                        contract = next((decorator for decorator in child.decorator_list
                                         if isinstance(decorator, ast.Call)
                                         and _call_name(decorator.func) == "contracted"), None)
                        yield ".".join([*scope, child.name]), makers[0], child, contract
                yield from walk(child, [*scope, child.name])
                continue
            target = (child.targets[0] if isinstance(child, ast.Assign)
                      else child.target if isinstance(child, ast.AnnAssign) else None)
            if isinstance(target, ast.Name) and isinstance(child.value, ast.Call):
                made = child.value
                inner = (made.args[0] if isinstance(made.func, ast.Call)
                         and _call_name(made.func.func) == "contracted" and made.args else made)
                if isinstance(inner, ast.Call) and (maker := _maker_of(inner)):
                    yield ".".join([*scope, target.id]), maker, child, declared(inner)
            if isinstance(child, ast.Call) and _call_name(child.func) == "linear_tangent":
                yield ".".join([*scope, "linear_tangent"]), "linear_tangent", child, declared(child)
            yield from walk(child, scope)

    for name, maker, node, contract in walk(tree, []):
        yield f"{relative}::{name}", maker, node, contract


def contract_tests(checkout: Path) -> collections.Counter[str]:
    """The test functions of `checkout`'s tests/test_contracts_*.py, each the holder of a contract, with
    how many files define each name (a seam's contract names one alone), read once a version of the files
    (each file's modification time), whichever tree it lints."""
    files = sorted((checkout / "tests").glob("test_contracts_*.py"))
    return _contract_tests(tuple((path, path.stat().st_mtime_ns) for path in files))


@functools.cache
def _contract_tests(files: tuple[tuple[Path, int], ...]) -> collections.Counter[str]:
    return collections.Counter(node.name for path, _ in files for node in ast.parse(path.read_text()).body
                               if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"))


def missing(relative: str, tree: ast.Module, checkout: Path) -> Iterator[tuple[int, int, str]]:
    """SLOP028's findings in `relative` (`tree`), each `(line, column, message)`: a custom derivative with
    no contract at its seam, or one naming other than one test function of `checkout`'s
    tests/test_contracts_*.py."""
    for seam, maker, node, contract in derivative_seams(relative, tree):
        where = (node.lineno, node.col_offset + 1)
        name = seam.split("::")[1]
        if contract is None:
            yield (*where, f"{name} ({maker}) makes a custom derivative with no contract: declare "
                           "`contracted(test)` of dew.nn.derivatives at it")
            continue
        named = [argument.value for argument in contract.args if isinstance(argument, ast.Constant)]
        if len(named) != 1 or len(contract.args) != 1 or contract_tests(checkout)[named[0]] != 1:
            yield (*where, f"{name}'s contract names {named}, not one test function of "
                           "tests/test_contracts_*.py")
