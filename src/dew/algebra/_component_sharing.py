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
_MIN_WHOLE = 3
_MAX_SHARED = 64
"""Long repeats can share several readable helpers rather than one oversized body."""


class _Fragment(ast.NodeTransformer):
    """Alpha-rename a fragment without changing its caller's geometry or logical shift direction."""

    def __init__(self, scope: set[str], fragments: set[str]):
        self.fragments = fragments
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

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        # Both logical shifts have the same bind interface. The caller passes the original primitive.
        if (node.attr in ("shift_left_p", "shift_right_logical_p")
                and isinstance(node.value, ast.Name) and node.value.id == "lax"):
            return self._input(ast.unparse(node))
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if isinstance(node.func, ast.Name) and node.func.id in self.fragments:
            operands = [self._input(ast.unparse(arg)) if isinstance(arg, ast.Tuple)
                        and _extent(arg, self.bound) else self.visit(arg) for arg in node.args]
            return ast.Call(func=node.func, args=operands, keywords=node.keywords)
        found = self.generic_visit(node)
        assert isinstance(found, ast.Call)
        call = found.func
        if (isinstance(call, ast.Attribute) and call.attr == "array" and isinstance(call.value, ast.Name)
                and call.value.id == "np" and found.args and isinstance(found.args[0], ast.Constant)
                and type(found.args[0].value) is int):
            # The printer's integer masks and shifts differ by value, not by dtype or primitive.
            # Pass the Python integer; construct its array at the original point of the trace.
            found.args[0] = self._input(str(found.args[0].value))
        return found

    def visit_keyword(self, node: ast.keyword) -> ast.AST:
        if (node.arg == "new_dtype" and isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == "jnp.dtype" and len(node.value.args) == 1
                and isinstance(node.value.args[0], ast.Constant)):
            return ast.keyword(arg=node.arg, value=self._input(ast.unparse(node.value)))
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
    if isinstance(call, ast.Name):
        return call.id in fragments
    if not isinstance(call, ast.Attribute):
        return False
    return call.attr == "bind" or (call.attr == "array" and isinstance(call.value, ast.Name)
                                   and call.value.id == "np")


@dataclasses.dataclass
class _Span:
    name: str
    first: int
    available: int
    normal: _Fragment
    whole: bool
    statements: list[_Assignment] = dataclasses.field(default_factory=list)

    def extend(self, nodes: list[ast.stmt], width: int) -> tuple[_Assignment, ...]:
        for index in range(self.first + len(self.statements), self.first + width):
            self.statements.append(_assignment(nodes[index], self.normal))
        return tuple(self.statements)

    def snapshot(self) -> _Copy:
        return _Copy(self.name, self.first, tuple(self.statements), tuple(self.normal.inputs),
                     tuple(self.normal.bound))


def _assignment(node: ast.stmt, normal: _Fragment) -> _Assignment:
    # Visit the operands before allocating the SSA result names.
    assert isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
    call = normal.visit(copy.deepcopy(node.value))
    assert isinstance(call, ast.Call)
    (target,) = node.targets
    target = ast.unparse(normal.visit(copy.deepcopy(target)))
    args = tuple(ast.unparse(arg) for arg in (*call.args, *call.keywords))
    return _Assignment(target, ast.unparse(call.func), args)


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
                last = indices[-1] + 1
                whole = (indices[0] == 0 and last == len(body.statements) - 1
                         and isinstance(body.statements[last], ast.Return))
                minimum = _MIN_WHOLE if whole else _MIN_SHARED
                candidates.extend(_Span(name, first, last - first, _Fragment(body.scope, fragments),
                                        whole and first == 0)
                                  for first in indices[:len(indices) - minimum + 1])
    best: tuple[_Copy, ...] = ()
    saved = 0
    for width in range(_MIN_WHOLE, _MAX_SHARED + 1):
        by_body: dict[tuple[_Assignment, ...], list[_Span]] = defaultdict(list)
        for span in candidates:
            if span.available >= width:
                signature = span.extend(bodies[span.name].statements, width)
                by_body[signature].append(span)
        candidates = []
        for entries in by_body.values():
            # Every group's starts remain in source order. Earliest disjoint spans leave the
            # most room for the next one of the same width in the same function.
            stops: dict[str, int] = {}
            copies = []
            for span in entries:
                if span.first >= stops.get(span.name, 0):
                    copies.append(span)
                    stops[span.name] = span.first + width
            if len(copies) < 2:
                continue
            candidates.extend(entries)
            if width < _MIN_SHARED:
                copies = [span for span in copies if span.whole and span.available == width]
            avoided = width * (len(copies) - 1)
            if avoided > saved:
                best, saved = tuple(span.snapshot() for span in copies), avoided
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
        # Later replacements cannot slide an earlier source range in the same function.
        for found, reads in reversed(tuple(zip(copies, suffixes, strict=True))):
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


def bound_call(source: str, helpers: set[str], shapes: tuple[tuple[int, ...], ...]) -> list[str] | None:
    """Bind a tail call's known case arguments instead of emitting a redundant wrapper function."""
    (function,) = ast.parse(source).body
    assert isinstance(function, ast.FunctionDef)
    if len(function.body) != 2:
        return None
    binding, returned = function.body
    if (not isinstance(binding, ast.Assign) or not isinstance(returned, ast.Return)
            or not isinstance(binding.value, ast.Call) or returned.value is None
            or ast.unparse(binding.targets[0]) != ast.unparse(returned.value)):
        return None
    call = binding.value
    if not isinstance(call.func, ast.Name) or call.func.id not in helpers or call.keywords:
        return None
    names = tuple(arg.arg for arg in function.args.args)
    if tuple(ast.unparse(arg) for arg in call.args[:len(names)]) != names:
        return None
    dimensions = dict(zip(names, shapes, strict=True))
    static = [_static_argument(arg, dimensions) for arg in call.args[len(names):]]
    if any(value is None for value in static):
        return None
    return [call.func.id, *(f"p{index}={value}" for index, value in enumerate(static, len(names)))]


def _static_argument(expr: ast.expr, dimensions: dict[str, tuple[int, ...]]) -> str | None:
    if isinstance(expr, ast.Constant):
        return repr(expr.value)
    if isinstance(expr, ast.Tuple):
        parts = [_static_argument(part, dimensions) for part in expr.elts]
        if any(part is None for part in parts):
            return None
        return f"({', '.join(part for part in parts if part is not None)}{',' if len(parts) == 1 else ''})"
    if isinstance(expr, ast.Subscript) and isinstance(expr.value, ast.Attribute) and expr.value.attr == "shape":
        base = expr.value.value
        if (isinstance(base, ast.Name) and base.id in dimensions and isinstance(expr.slice, ast.Constant)
                and type(expr.slice.value) is int):
            return repr(dimensions[base.id][expr.slice.value])
    if isinstance(expr, ast.Attribute) and ast.unparse(expr) in ("lax.shift_left_p", "lax.shift_right_logical_p"):
        return ast.unparse(expr)
    if (isinstance(expr, ast.Call) and ast.unparse(expr.func) == "jnp.dtype"
            and len(expr.args) == 1 and isinstance(expr.args[0], ast.Constant)):
        return ast.unparse(expr)
    return None
