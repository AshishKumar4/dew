"""SLOP011's read of production source and program IR as text into a test's assertions (row 4512), for
tools/lint_slop.py: the names a test binds to a production path, a program, or text of either, followed
through its assignments into its `assert` statements. A first-party call's result is the behaviour under
test, so text passed into one leaves it untainted.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

import lint_scopes as scopes

SOURCE_READERS = {"inspect.getsource", "inspect.getsourcelines", "inspect.getsourcefile"}
"""How a test reads a function's or module's source."""
IR_TEXTS = {"as_text", "as_hlo_text", "pretty_print", "as_hlo_dot_graph", "mlir_module"}
"""A lowered or compiled program's text, by the method that prints it (jax.stages, jax.core.Jaxpr)."""
PROGRAMS = {"jax.make_jaxpr", "lower", "compile", "trace"}
"""The calls whose value is a program, whose str or repr prints its IR: a jaxpr, a lowering, a compilation."""
FILE_READERS = {"read_text", "read_bytes", "read", "readlines"}
"""How a test reads a file's text."""
STRING_METHODS = {"splitlines", "split", "strip", "lstrip", "rstrip", "lower", "upper", "count", "find",
                  "index", "startswith", "endswith", "partition", "rpartition", "replace", "join", "decode",
                  "encode"}
"""A string's methods, whose result is the text matched or reshaped."""
TEXT_CALLS = {"re.search", "re.match", "re.fullmatch", "re.findall", "re.finditer", "re.sub", "re.split",
              "len", "str", "repr", "sorted", "set", "list", "tuple", "any", "all", "sum", "max", "min"}
"""The library calls whose result is still the text, or a measure of it, that they are handed."""


def _resolved(node: ast.expr, tables: Mapping[int, Mapping[str, str]]) -> str:
    """`scopes.resolved` of a name or an attribute chain at `node`, through its scope's imports."""
    if not isinstance(node, ast.Name | ast.Attribute):
        return ""
    return scopes.resolved(node, tables.get(id(node), {}))


@dataclass
class _Taint:
    """The names one test binds to a production path, a program, or text of either (`_texts`)."""

    tables: Mapping[int, Mapping[str, str]]
    package: str
    paths: set[str] = field(default_factory=set)
    programs: set[str] = field(default_factory=set)
    texts: set[str] = field(default_factory=set)

    def path(self, expr: ast.expr) -> bool:
        """A file under `src/<package>/`: spelled, joined from "src" and the package, or a first-party
        module's `__file__`, or a name bound to one."""
        parts = []
        for node in ast.walk(expr):
            if isinstance(node, ast.Name) and node.id in self.paths:
                return True
            if isinstance(node, ast.Constant):
                parts.append(node.value)
                if isinstance(node.value, str) and node.value.startswith(f"src/{self.package}"):
                    return True
            if isinstance(node, ast.Attribute) and node.attr == "__file__" and self.first_party(node.value):
                return True
        return "src" in parts and self.package in parts

    def program(self, expr: ast.expr) -> bool:
        """A jaxpr, a lowering or a compilation (`PROGRAMS`), or a name bound to one."""
        if isinstance(expr, ast.Name):
            return expr.id in self.programs
        if not isinstance(expr, ast.Call):
            return False
        func = expr.func.func if isinstance(expr.func, ast.Call) else expr.func
        method = isinstance(func, ast.Attribute) and not isinstance(func.value, ast.Constant)
        return _resolved(func, self.tables) in PROGRAMS or (method and func.attr in PROGRAMS)

    def read(self, node: ast.AST) -> bool:
        """One node that reads source or IR as text."""
        if isinstance(node, ast.Name):
            return node.id in self.texts
        if isinstance(node, ast.FormattedValue):
            return self.program(node.value)
        if not isinstance(node, ast.Call):
            return False
        func, called = node.func, _resolved(node.func, self.tables)
        attribute = func.attr if isinstance(func, ast.Attribute) else ""
        return (called in SOURCE_READERS or attribute in IR_TEXTS
                or (attribute in FILE_READERS and self.path(func.value))
                or (called == "open" and bool(node.args) and self.path(node.args[0]))
                or (called in ("str", "repr") and bool(node.args) and self.program(node.args[0])))

    def text(self, node: ast.AST) -> bool:
        """Whether `node` is source or IR text, or text matched, searched, split or compared from it: through
        operators, comprehensions, a string's own methods and `TEXT_CALLS`, never another call, whose result
        is a parse (Dew's own, `dew.telemetry.instrumentation.hlo_instructions`, or a helper's) and not the
        text."""
        if self.read(node):
            return True
        if isinstance(node, ast.Call):
            func = node.func
            if _resolved(func, self.tables) in TEXT_CALLS:
                return any(self.text(argument) for argument in node.args)
            return (isinstance(func, ast.Attribute) and func.attr in STRING_METHODS | FILE_READERS
                    and self.text(func.value))
        if isinstance(node, ast.Lambda | ast.Dict | ast.DictComp):
            return False
        return any(self.text(child) for child in ast.iter_child_nodes(node))

    def first_party(self, node: ast.expr) -> bool:
        name = _resolved(node, self.tables)
        return name == self.package or name.startswith(f"{self.package}.")

    def bind(self, target: ast.expr, value: ast.expr) -> None:
        """Bind `target`'s names to what `value` is, unbinding them from what it is not."""
        bound = {name.id for name in ast.walk(target) if isinstance(name, ast.Name)}
        for names, holds in ((self.paths, self.path(value)), (self.programs, self.program(value)),
                             (self.texts, self.text(value))):
            if holds:
                names |= bound
            else:
                names -= bound


def texts(function: ast.FunctionDef | ast.AsyncFunctionDef, tables: Mapping[int, Mapping[str, str]],
           package: str) -> Iterator[ast.Assert]:
    """Each `assert` of `function` whose test reads production source or a program's IR as text: a file
    under `src/<package>/` or a first-party module's `__file__` read (`FILE_READERS`, `open`), source read
    by `inspect` (`SOURCE_READERS`), or a jaxpr, lowering or compilation printed (`str`, `repr`, an
    f-string, `IR_TEXTS`), through the names the function binds to any of them, a `with open(...)`'s
    among them (row 4512). A program's structure read as objects (a jaxpr's equations, a compilation's
    cost analysis) is behaviour; its text is a snapshot of one lowering."""
    taint = _Taint(tables, package)
    for node in sorted((node for node in scopes.own(function)
                        if isinstance(node, ast.Assign | ast.AnnAssign | ast.Assert | ast.With)),
                       key=lambda node: (node.lineno, node.col_offset)):
        if isinstance(node, ast.Assert):
            if taint.text(node.test):
                yield node
        elif isinstance(node, ast.With):
            for item in node.items:
                if item.optional_vars is not None:
                    taint.bind(item.optional_vars, item.context_expr)
        elif node.value is not None:
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                taint.bind(target, node.value)
