"""Share straight-line fragments of printed component bodies without changing their primitive order."""

import ast
import copy
import dataclasses
import itertools
from collections import defaultdict
from collections.abc import Collection

from dew.algebra import printed

_MIN_SHARED = 6
"""Smaller fragments would add a call and arguments for little source reduction."""


class _Fragment(ast.NodeTransformer):
    """Alpha-rename a flat fragment. Only static broadcast shapes/axes may vary between copies."""

    def __init__(self, scope: set[str]):
        self.scope = scope
        self.inputs: dict[str, str] = {}
        self.bound: dict[str, str] = {}

    def _input(self, value: str) -> ast.Name:
        slot = self.inputs.setdefault(value, f"p{len(self.inputs)}")
        return ast.Name(id=slot, ctx=ast.Load())

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id == "_" or node.id not in self.scope:
            return node
        if isinstance(node.ctx, ast.Store):
            self.bound[node.id] = f"v{len(self.bound)}"
        if node.id in self.bound:
            return ast.Name(id=self.bound[node.id], ctx=node.ctx)
        return self._input(node.id)

    def visit_keyword(self, node: ast.keyword) -> ast.AST:
        if node.arg in ("shape", "broadcast_dimensions", "new_sizes") and _extent(node.value, self.bound):
            return ast.keyword(arg=node.arg, value=self._input(ast.unparse(node.value)))
        return self.generic_visit(node)


def _extent(node: ast.expr, bound: dict[str, str]) -> bool:
    """A static expression the printer takes from array inputs, not from a value made in the fragment."""
    if isinstance(node, ast.Tuple):
        return all(_extent(part, bound) for part in node.elts)
    if isinstance(node, ast.Constant):
        return type(node.value) is int
    return (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute)
            and node.value.attr == "shape" and isinstance(node.value.value, ast.Name)
            and node.value.value.id not in bound and isinstance(node.slice, ast.Constant)
            and type(node.slice.value) is int)


@dataclasses.dataclass(frozen=True)
class _Assignment:
    target: str
    call: str
    arguments: tuple[str, ...]

    def printed(self) -> list[str]:
        return printed.wrapped(f"    {self.target} = {self.call}(", list(self.arguments), ")")


@dataclasses.dataclass
class _Copy:
    name: str
    first: int
    statements: tuple[_Assignment, ...]
    arguments: tuple[str, ...]
    written: tuple[str, ...]


def _assigns(node: ast.stmt, fragments: set[str]) -> bool:
    if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
        return False
    call = node.value.func
    return (isinstance(call, ast.Name) and call.id in fragments
            or isinstance(call, ast.Attribute) and (call.attr == "bind"
                or (call.attr == "array" and isinstance(call.value, ast.Name) and call.value.id == "np")))


def _copy(name: str, first: int, nodes: list[ast.stmt], scope: set[str]) -> _Copy:
    normal = _Fragment(scope)
    statements = []
    for node in nodes:
        # Visit the operands before allocating the SSA result names.
        assert isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
        call = normal.visit(copy.deepcopy(node.value))
        assert isinstance(call, ast.Call)
        (target,) = node.targets
        target = ast.unparse(normal.visit(copy.deepcopy(target)))
        args = tuple(ast.unparse(arg) for arg in (*call.args, *call.keywords))
        statements.append(_Assignment(target, ast.unparse(call.func), args))
    return _Copy(name, first, tuple(statements), tuple(normal.inputs), tuple(normal.bound))


@dataclasses.dataclass(frozen=True)
class _Body:
    statements: list[ast.stmt]
    scope: set[str]


def _bodies(source: dict[str, str]) -> dict[str, _Body]:
    bodies = {}
    for name, text in source.items():
        (function,) = ast.parse(text).body
        assert isinstance(function, ast.FunctionDef)
        scope = {node.arg for node in ast.walk(function.args) if isinstance(node, ast.arg)}
        scope.update(node.id for node in ast.walk(function)
                     if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store))
        bodies[name] = _Body(function.body, scope)
    return bodies


def _largest(bodies: dict[str, _Body], fragments: set[str]) -> tuple[_Copy, ...]:
    """Find the shared region which saves the most source, including earlier emitted fragments."""
    candidates = []
    for name, body in bodies.items():
        eligible = (_assigns(node, fragments) for node in body.statements)
        runs = itertools.groupby(enumerate(eligible), key=lambda pair: pair[1])
        for accepted, group in runs:
            if accepted:
                indices = [index for index, _ in group]
                candidates.extend((name, first, indices[-1] + 1 - first)
                                  for first in indices[:len(indices) - _MIN_SHARED + 1])
    best: tuple[_Copy, ...] = ()
    saved = 0
    for width in itertools.count(_MIN_SHARED):
        by_body: dict[tuple[_Assignment, ...], list[tuple[_Copy, int]]] = defaultdict(list)
        for name, first, available in candidates:
            if available >= width:
                body = bodies[name]
                duplicate = _copy(name, first, body.statements[first:first + width], body.scope)
                by_body[duplicate.statements].append((duplicate, available))
        candidates = []
        for entries in by_body.values():
            copies = {found.name: found for found, _ in reversed(entries)}
            if len(copies) < 2:
                continue
            candidates.extend((found.name, found.first, available) for found, available in entries)
            avoided = width * (len(copies) - 1)
            if avoided > saved:
                best, saved = tuple(reversed(tuple(copies.values()))), avoided
        if not candidates:
            return best
    return best


def _reads(statements: list[ast.stmt]) -> set[str]:
    return {node.id for statement in statements for node in ast.walk(statement)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}


def shared(source: dict[str, str], *, reserved: Collection[str] = ()) -> dict[str, str]:
    """Emit private helpers for equivalent contiguous binds. Calls pass static axes and the original
    preceding values. They return only values used after at least one copy's span. No tracing decision
    moves into a run-time test, and each call executes the original primitive sequence in order."""
    forms = dict(source)
    occupied = set(source).union(reserved)
    numbers = itertools.count()
    fragments: set[str] = set()
    while True:
        bodies = _bodies(forms)
        copies = _largest(bodies, fragments)
        if not copies:
            return forms
        first = copies[0]
        width = len(first.statements)
        suffixes = [_reads(bodies[found.name].statements[found.first + width:]) for found in copies]
        kept = [index for index in range(len(first.written))
                if any(found.written[index] in reads for found, reads in zip(copies, suffixes, strict=True))]
        while (name := f"_shared_component_{next(numbers)}") in occupied:
            pass
        occupied.add(name)
        fragments.add(name)
        fragment_lines = printed.wrapped(f"def {name}(", [f"p{i}" for i in range(len(first.arguments))], "):")
        fragment_lines += [line for statement in first.statements for line in statement.printed()]
        if kept:
            tail = ",)" if len(kept) == 1 else ")"
            fragment_lines += printed.wrapped("    return (", [f"v{i}" for i in kept], tail)
        forms[name] = "\n".join(fragment_lines)
        for found, reads in zip(copies, suffixes, strict=True):
            nodes = bodies[found.name].statements
            begin, end = nodes[found.first].lineno - 1, nodes[found.first + width - 1].end_lineno
            outputs = ", ".join(found.written[index] if found.written[index] in reads else "_"
                                for index in kept)
            if len(kept) == 1:
                outputs += ","
            assigned = f"{outputs} = " if outputs else ""
            call = printed.wrapped(f"    {assigned}{name}(", list(found.arguments), ")")
            lines = forms[found.name].splitlines()
            forms[found.name] = "\n".join([*lines[:begin], *call, *lines[end:]])
