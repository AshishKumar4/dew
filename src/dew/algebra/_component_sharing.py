"""Factor ordered primitive bindings before rendering any generated Python."""

import dataclasses
import itertools
from collections import defaultdict
from collections.abc import Collection, Iterator

import numpy as np
from jax import lax

from dew.algebra.printed import (
    Binding,
    Body,
    Call,
    Extent,
    Layout,
    Literal,
    Operand,
    Primitive,
    Ref,
    StaticTuple,
)

_MIN_SHARED = 6
_MIN_WHOLE = 3
_MAX_SHARED = 64
_GEOMETRY = frozenset({"shape", "broadcast_dimensions", "new_sizes"})
_SHIFTS = (lax.shift_left_p, lax.shift_right_logical_p)


class _Fragment:
    """Canonicalize SSA names and bind only explicit static inputs at the call site."""

    def __init__(self):
        self.inputs: dict[Operand, Ref] = {}
        self.bound: dict[str, Ref] = {}

    def input(self, value: Operand) -> Ref:
        if value not in self.inputs:
            self.inputs[value] = Ref(f"p{len(self.inputs)}")
        return self.inputs[value]

    def operand(self, value: Operand) -> Operand:
        if isinstance(value, Ref):
            return self.bound.get(value.name) or self.input(value)
        if isinstance(value, Extent):
            if value.operand.name not in self.bound:
                return self.input(value)
            return Extent(self.bound[value.operand.name], value.axis)
        if isinstance(value, Layout):
            if all(name not in self.bound for name in _reads(value)):
                return self.input(value)
            return Layout(tuple(n if isinstance(n, int) else self.dimension(n) for n in value.dimensions))
        if isinstance(value, StaticTuple) and all(type(n) is int for n in value.values):
            return self.input(value)
        if isinstance(value, Literal) and value.integer is not None:
            original = value.integer
            integer = self.operand(original) if isinstance(original, Ref) else self.input(original)
            assert isinstance(integer, Ref)
            return dataclasses.replace(value, integer=integer, data=b"")
        if isinstance(value, Primitive) and value.value in _SHIFTS:
            return self.input(value)
        return value

    def dimension(self, value: Extent) -> Extent:
        operand = self.operand(value.operand)
        assert isinstance(operand, Ref)
        return Extent(operand, value.axis)

    def binding(self, statement: Binding) -> Binding:
        raw = statement.value
        if isinstance(raw, Call):
            callee = self.operand(raw.callee) if not isinstance(raw.callee, str) else raw.callee
            assert isinstance(callee, str | Ref | Primitive)
            arguments = tuple(map(self.operand, raw.operands))
            keywords = tuple((name, self.input(value) if _static_parameter(name, value)
                              else self.operand(value)) for name, value in raw.keywords)
            raw = Call(callee, arguments, keywords)
        else:
            raw = self.operand(raw)
        targets = []
        for target in statement.targets:
            if target == "_":
                targets.append(target)
            else:
                name = f"v{len(self.bound)}"
                self.bound[target] = Ref(name)
                targets.append(name)
        return Binding(tuple(targets), raw, statement.unpack)


def _static_parameter(name: str, value: Operand) -> bool:
    return ((name == "new_dtype" and isinstance(value, np.dtype))
            or (name in _GEOMETRY and isinstance(value, StaticTuple)
                and all(type(n) is int for n in value.values)))


def _eligible(node: Binding | str) -> bool:
    return isinstance(node, Binding) and (isinstance(node.value, Call)
                                          or (isinstance(node.value, Literal) and not node.constant))


@dataclasses.dataclass(frozen=True)
class _Copy:
    name: str
    first: int
    statements: tuple[Binding, ...]
    arguments: tuple[Operand, ...]
    written: tuple[str, ...]


@dataclasses.dataclass
class _Span:
    name: str
    first: int
    available: int
    whole: bool
    normal: _Fragment = dataclasses.field(default_factory=_Fragment)
    statements: list[Binding] = dataclasses.field(default_factory=list)

    def extend(self, nodes: tuple[Binding | str, ...], width: int) -> tuple[Binding, ...]:
        for index in range(self.first + len(self.statements), self.first + width):
            node = nodes[index]
            assert isinstance(node, Binding)
            self.statements.append(self.normal.binding(node))
        return tuple(self.statements)

    def snapshot(self) -> _Copy:
        return _Copy(self.name, self.first, tuple(self.statements), tuple(self.normal.inputs),
                     tuple(self.normal.bound))


def _spans(bodies: dict[str, Body]) -> list[_Span]:
    candidates = []
    for name, body in bodies.items():
        runs = itertools.groupby(enumerate(body.statements), key=lambda numbered: _eligible(numbered[1]))
        for eligible, group in runs:
            if not eligible:
                continue
            starts = [index for index, _ in group]
            end = starts[-1] + 1
            whole = starts[0] == 0 and end == len(body.statements)
            minimum = _MIN_WHOLE if whole else _MIN_SHARED
            candidates.extend(_Span(name, start, end - start, whole and start == 0)
                              for start in starts[:len(starts) - minimum + 1])
    return candidates


def _disjoint(entries: list[_Span], width: int) -> list[_Span]:
    stops: dict[str, int] = {}
    chosen = []
    for span in entries:
        if span.first >= stops.get(span.name, 0):
            chosen.append(span)
            stops[span.name] = span.first + width
    return chosen


def _largest(bodies: dict[str, Body]) -> tuple[_Copy, ...]:
    candidates = _spans(bodies)
    best: tuple[_Copy, ...] = ()
    saved = 0
    for width in range(_MIN_WHOLE, _MAX_SHARED + 1):
        by_body: dict[tuple[Binding, ...], list[_Span]] = defaultdict(list)
        for span in candidates:
            if span.available >= width:
                by_body[span.extend(bodies[span.name].statements, width)].append(span)
        candidates = []
        for entries in by_body.values():
            copies = _disjoint(entries, width)
            if len(copies) < 2:
                continue
            candidates.extend(entries)
            if width < _MIN_SHARED:
                copies = [span for span in copies if span.whole and span.available == width]
            avoided = width * (len(copies) - 1)
            if avoided > saved:
                best, saved = tuple(span.snapshot() for span in copies), avoided
        if not candidates:
            break
    return best


def _reads(value: Operand) -> Iterator[str]:
    if isinstance(value, Ref):
        yield value.name
    elif isinstance(value, Extent):
        yield value.operand.name
    elif isinstance(value, Layout):
        for dimension in value.dimensions:
            yield from _reads(dimension)
    elif isinstance(value, Literal) and isinstance(value.integer, Ref):
        yield value.integer.name
    elif isinstance(value, Call):
        if isinstance(value.callee, Ref):
            yield value.callee.name
        for operand in value.operands:
            yield from _reads(operand)
        for _, operand in value.keywords:
            yield from _reads(operand)


def _live(body: Body, start: int) -> set[str]:
    live = {name for statement in body.statements[start:] if isinstance(statement, Binding)
            for name in _reads(statement.value)}
    live.update(name for operand in body.outputs for name in _reads(operand))
    return live


def shared(source: dict[str, Body], *, reserved: Collection[str] = ()) -> dict[str, Body]:
    """Factor disjoint spans with the same typed operations and preserve their ordered live outputs."""
    forms = dict(source)
    occupied = set(source).union(reserved)
    numbers = itertools.count()
    while copies := _largest(forms):
        first = copies[0]
        width = len(first.statements)
        suffixes = [_live(forms[found.name], found.first + width) for found in copies]
        kept = [i for i in range(len(first.written))
                if any(found.written[i] in reads for found, reads in zip(copies, suffixes, strict=True))]
        while (name := f"_shared_component_{next(numbers)}") in occupied:
            pass
        occupied.add(name)
        forms[name] = Body(tuple(f"p{i}" for i in range(len(first.arguments))), first.statements,
                           tuple(Ref(f"v{i}") for i in kept))
        for found, reads in reversed(tuple(zip(copies, suffixes, strict=True))):
            body = forms[found.name]
            outputs = tuple(found.written[i] if found.written[i] in reads else "_" for i in kept)
            invoke = Binding(outputs, Call(name, found.arguments), unpack=True)
            statements = (*body.statements[:found.first], invoke, *body.statements[found.first + width:])
            forms[found.name] = dataclasses.replace(body, statements=statements)
    return forms


def bound_call(body: Body, shared_names: set[str], shapes: tuple[tuple[int, ...], ...]) -> Call | None:
    """Bind a tail call's known case arguments directly in the case table."""
    if len(body.statements) != 1 or not isinstance(binding := body.statements[0], Binding):
        return None
    if not isinstance(call := binding.value, Call) or not isinstance(call.callee, str):
        return None
    if (call.callee not in shared_names or not binding.unpack
            or body.outputs != tuple(map(Ref, binding.targets))):
        return None
    count = len(body.inputs)
    if call.keywords or call.operands[:count] != tuple(map(Ref, body.inputs)):
        return None
    sizes = dict(zip(body.inputs, shapes, strict=True))
    fixed = []
    for position, value in enumerate(call.operands[count:], count):
        if any(name not in sizes for name in _reads(value)) or isinstance(value, Ref | Literal | Call):
            return None
        if isinstance(value, Layout):
            value = StaticTuple(tuple(sizes[dim.operand.name][dim.axis] if isinstance(dim, Extent) else dim
                                      for dim in value.dimensions))
        elif isinstance(value, Extent):
            value = sizes[value.operand.name][value.axis]
        fixed.append((f"p{position}", value))
    return Call(call.callee, (), tuple(fixed))
