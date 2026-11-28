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
_MIN_TAIL = _MIN_SHARED - 1
_MIN_WHOLE = 2
_MAX_SHARED = 64
_GEOMETRY = frozenset({"shape", "broadcast_dimensions", "new_sizes"})
_SHIFTS = (lax.shift_left_p, lax.shift_right_logical_p)


class _Fragment:
    """Canonicalize SSA names and bind only explicit static inputs at the call site."""

    def __init__(self):
        self.inputs: dict[Operand, Ref] = {}
        self.arguments: list[Operand] = []
        self.separate: set[int] = set()
        """The slots of a call's integer arguments, which `shared` bakes in where every copy agrees."""
        self.statics: set[int] = set()
        """Every static value's slot, which `shared` joins with another every copy fills alike."""
        self.bound: dict[str, Ref] = {}

    def input(self, value: Operand) -> Ref:
        if value not in self.inputs:
            self.inputs[value] = self.argument(value)
        return self.inputs[value]

    def argument(self, value: Operand) -> Ref:
        ref = Ref(f"p{len(self.arguments)}")
        self.arguments.append(value)
        return ref

    def static(self, value: Operand) -> Ref:
        """A static value at a slot of its own, whatever another slot holds: two copies match by their
        operations alone, and `shared` merges the slots every copy fills alike (`_coalesced`)."""
        self.statics.add(len(self.arguments))
        return self.argument(value)

    def integer(self, value: int) -> Ref:
        """A call's integer argument at a static slot of its own, baked into the helper where every copy
        passes it alike (`_shared_integer_arguments`)."""
        self.separate.add(len(self.arguments))
        return self.static(value)

    def operand(self, value: Operand) -> Operand:
        if isinstance(value, Ref):
            return self.bound.get(value.name) or self.input(value)
        if isinstance(value, Extent):
            if value.operand.name not in self.bound:
                return self.static(value)
            return Extent(self.bound[value.operand.name], value.axis)
        if isinstance(value, Layout):
            if all(name not in self.bound for name in _reads(value)):
                return self.static(value)
            return Layout(tuple(n if isinstance(n, int) else self.dimension(n) for n in value.dimensions))
        if isinstance(value, StaticTuple) and all(type(n) is int for n in value.values):
            return self.static(value)
        if isinstance(value, Literal) and value.integer is not None:
            original = value.integer
            integer = self.operand(original) if isinstance(original, Ref) else self.static(original)
            # Its type an input too, as a cast's: the same bindings over another word or float width.
            dtype = (self.operand(value.dtype) if isinstance(value.dtype, Ref)
                     else self.static(np.dtype(value.dtype)))
            assert isinstance(integer, Ref) and isinstance(dtype, Ref)
            return dataclasses.replace(value, integer=integer, dtype=dtype, data=b"")
        if isinstance(value, np.dtype):
            return self.static(value)
        if isinstance(value, Primitive) and value.value in _SHIFTS:
            return self.static(value)
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
            # Separate static positions in an already shared call can have the same value in
            # just one case (e.g. an exponent bias equal to a shift). That need not prevent sharing.
            arguments = tuple(self.integer(value) if type(value) is int and isinstance(raw.callee, str)
                              else self.operand(value) for value in raw.operands)
            keywords = tuple((name, self.static(value) if _static_parameter(name, value)
                              else self.operand(value)) for name, value in raw.keywords)
            raw = Call(callee, arguments, keywords)
        else:
            raw = self.operand(raw)
        targets = []
        for target in statement.targets:
            # A discarded output named as a kept one is: two copies match whether or not each reads it, and
            # an unread name (no program writes `_n`) is no output of the helper (`shared`'s `kept`).
            held = f"_{len(self.bound)}" if target == "_" else target
            name = f"v{len(self.bound)}"
            self.bound[held] = Ref(name)
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
    separate: frozenset[int]
    statics: frozenset[int]


@dataclasses.dataclass
class _Span:
    name: str
    first: int
    available: int
    whole: bool
    tail: bool = False
    """Whether its run ends the body, whose return statement follows its last binding."""
    normal: _Fragment = dataclasses.field(default_factory=_Fragment)
    statements: list[Binding] = dataclasses.field(default_factory=list)
    counts: list[tuple[int, int]] = dataclasses.field(default_factory=list)
    """Its fragment's arguments and bound names after each of its statements: a statement's normal form
    reads only those before it, so the span at any width it has reached is a prefix of each."""

    def extend(self, nodes: tuple[Binding | str, ...], width: int) -> Binding:
        """Its normal form `width` statements wide, its last statement (normalized at most once)."""
        for index in range(self.first + len(self.statements), self.first + width):
            node = nodes[index]
            assert isinstance(node, Binding)
            self.statements.append(self.normal.binding(node))
            self.counts.append((len(self.normal.arguments), len(self.normal.bound)))
        return self.statements[width - 1]

    def snapshot(self, width: int) -> _Copy:
        """Its copy `width` statements wide, its fragment's state as it was there."""
        arguments, bound = self.counts[width - 1]
        return _Copy(self.name, self.first, tuple(self.statements[:width]),
                     tuple(self.normal.arguments[:arguments]), tuple(self.normal.bound)[:bound],
                     frozenset(i for i in self.normal.separate if i < arguments),
                     frozenset(i for i in self.normal.statics if i < arguments))


def _spans(bodies: dict[str, Body], made: dict[str, tuple[Body, list[_Span]]]) -> list[_Span]:
    """Every body's candidate spans, in its order and the bodies': a body `made` holds as it is now keeps
    the spans an earlier search made of it, their statements normalized so far (`_Span.extend` only appends:
    a span's normal form at a width is its own whatever search reads it), a changed or new body's made
    afresh and held."""
    candidates = []
    for name, body in bodies.items():
        held = made.get(name)
        if held is None or held[0] is not body:
            held = made[name] = (body, list(_body_spans(name, body)))
        candidates.extend(held[1])
    return candidates


def _body_spans(name: str, body: Body) -> Iterator[_Span]:
    runs = itertools.groupby(enumerate(body.statements), key=lambda numbered: _eligible(numbered[1]))
    for eligible, group in runs:
        if not eligible:
            continue
        starts = [index for index, _ in group]
        end = starts[-1] + 1
        tail = end == len(body.statements)
        whole = starts[0] == 0 and tail
        minimum = _MIN_WHOLE if whole else _MIN_TAIL if tail else _MIN_SHARED
        yield from (_Span(name, start, end - start, whole and start == 0, tail)
                    for start in starts[:len(starts) - minimum + 1])


def _disjoint(entries: list[_Span], width: int) -> list[_Span]:
    stops: dict[str, int] = {}
    chosen = []
    for span in entries:
        if span.first >= stops.get(span.name, 0):
            chosen.append(span)
            stops[span.name] = span.first + width
    return chosen


def _largest(bodies: dict[str, Body], made: dict[str, tuple[Body, list[_Span]]]) -> tuple[_Copy, ...]:
    """The disjoint copies of one span's normal form that save the most statements, widening the spans
    whose forms another span shares a statement at a time. A width's groups refine the last's by each
    span's next statement (`_Span.extend`'s last), the first width's by its first: two spans' normal forms
    are equal where their forms one statement narrower were and their next statements are, so a group is
    read by its parent's number and one statement, never its whole form again (hashing whole forms at every
    width and renormalizing every span at every search took the generation check from 49 s to 102 s)."""
    # Each span's group one statement wide, by its first statement: the widths below refine it.
    firsts: dict[Binding, int] = {}
    candidates = [(firsts.setdefault(span.extend(bodies[span.name].statements, 1), len(firsts)), span)
                  for span in _spans(bodies, made)]
    best: tuple[_Copy, ...] = ()
    saved = 0
    for width in range(_MIN_WHOLE, _MAX_SHARED + 1):
        by_body: dict[tuple[int, Binding], list[_Span]] = defaultdict(list)
        for group, span in candidates:
            if span.available >= width:
                by_body[group, span.extend(bodies[span.name].statements, width)].append(span)
        candidates = []
        for group, entries in enumerate(by_body.values()):
            copies = _disjoint(entries, width)
            if len(copies) < 2:
                continue
            candidates.extend((group, span) for span in entries)
            if width < _MIN_SHARED:
                # Below a shared span's width, only a whole body, or a run ending one, which its return
                # statement completes to six statements as rendered.
                copies = [span for span in copies if span.available == width
                          and (span.whole or (span.tail and width >= _MIN_TAIL))]
            avoided = width * (len(copies) - 1)
            if avoided > saved:
                best, saved = tuple(span.snapshot(width) for span in copies), avoided
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
    elif isinstance(value, Literal):
        for part in (value.integer, value.dtype):
            if isinstance(part, Ref):
                yield part.name
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


def _shared_integer_arguments(copies: tuple[_Copy, ...]) -> dict[str, int]:
    first = copies[0]
    fixed = {}
    for i in first.separate:
        value = first.arguments[i]
        if type(value) is not int:
            continue
        if all(i in copy.separate and copy.arguments[i] == value for copy in copies[1:]):
            fixed[f"p{i}"] = value
    return fixed


def _coalesced(copies: tuple[_Copy, ...], fixed: dict[str, int]) -> dict[str, Ref]:
    """Each static slot every copy fills with the value another, earlier slot holds, by the parameter it
    joins: one parameter where the copies give the slots one value (a mask and a shift alike in each), two
    where any copy gives them two."""
    first = copies[0]
    slots = [i for i in sorted(first.statics) if f"p{i}" not in fixed]
    merged: dict[int, int] = {}
    for at, i in enumerate(slots):
        if i in merged:
            continue
        for j in slots[at + 1:]:
            if j not in merged and all(_same(copy.arguments[i], copy.arguments[j]) for copy in copies):
                merged[j] = i
    return {f"p{j}": Ref(f"p{i}") for j, i in merged.items()}


def _same(first: Operand, second: Operand) -> bool:
    return type(first) is type(second) and first == second


def _substituted(value: Operand, values: dict[str, Operand]) -> Operand:
    """`value` with each parameter `values` names in place of its reference: a static integer every copy
    shares baked in (`_shared_integer_arguments`), a slot joining another renamed (`_coalesced`)."""
    if isinstance(value, Ref):
        return values.get(value.name, value)
    if isinstance(value, Extent):
        operand = _substituted(value.operand, values)
        assert isinstance(operand, Ref)
        return Extent(operand, value.axis)
    if isinstance(value, Layout):
        substituted = (_substituted(dimension, values) for dimension in value.dimensions)
        dimensions = tuple(dimension for dimension in substituted if isinstance(dimension, int | Extent))
        assert len(dimensions) == len(value.dimensions)
        return Layout(dimensions)
    if isinstance(value, StaticTuple):
        return StaticTuple(tuple(_substituted(part, values) for part in value.values))
    if isinstance(value, Literal):
        integer, dtype = (_substituted(part, values) if isinstance(part, Ref) else part
                          for part in (value.integer, value.dtype))
        return dataclasses.replace(value, integer=integer, dtype=dtype)
    if isinstance(value, Call):
        callee = _substituted(value.callee, values) if isinstance(value.callee, Ref) else value.callee
        assert isinstance(callee, str | Ref | Primitive)
        return Call(callee, tuple(_substituted(operand, values) for operand in value.operands),
                    tuple((name, _substituted(operand, values)) for name, operand in value.keywords))
    return value


def shared(source: dict[str, Body], *, reserved: Collection[str] = ()) -> dict[str, Body]:
    """Factor disjoint spans with the same typed operations and preserve their ordered live outputs."""
    forms = dict(source)
    occupied = set(source).union(reserved)
    numbers = itertools.count()
    made: dict[str, tuple[Body, list[_Span]]] = {}
    while copies := _largest(forms, made):
        first = copies[0]
        width = len(first.statements)
        suffixes = [_live(forms[found.name], found.first + width) for found in copies]
        kept = [i for i in range(len(first.written))
                if any(found.written[i] in reads for found, reads in zip(copies, suffixes, strict=True))]
        while (name := f"shared_component_{next(numbers)}") in occupied:
            pass
        occupied.add(name)
        # Give every helper the dynamic-array prefix the case table can pass directly.
        fixed = _shared_integer_arguments(copies)
        merged = _coalesced(copies, fixed)
        remaining = (i for i in range(len(first.arguments)) if f"p{i}" not in fixed and f"p{i}" not in merged)
        order = sorted(remaining, key=lambda i: not isinstance(first.arguments[i], Ref))
        outputs = tuple(Ref(f"v{i}") for i in kept)
        substituted = {**fixed, **merged}
        statements = tuple(dataclasses.replace(statement, value=_substituted(statement.value, substituted))
                           for statement in first.statements)
        forms[name] = Body(tuple(f"p{i}" for i in order), statements, outputs)
        for found, reads in reversed(tuple(zip(copies, suffixes, strict=True))):
            body = forms[found.name]
            outputs = tuple(found.written[i] if found.written[i] in reads else "_" for i in kept)
            invoke = Binding(outputs, Call(name, tuple(found.arguments[i] for i in order)), unpack=True)
            statements = (*body.statements[:found.first], invoke, *body.statements[found.first + width:])
            forms[found.name] = dataclasses.replace(body, statements=statements)
    return _forwarded(forms, set(source))


def _forwarded(forms: dict[str, Body], kept: set[str]) -> dict[str, Body]:
    """`forms` without a helper that only forwards to another (its inputs passed in order, every output
    returned as it came), each call of it the other's: a span factored again whole leaves one. The
    declarations' own bodies (`kept`) stay, a caller's names."""
    target = {}
    for name, body in forms.items():
        if name in kept or len(body.statements) != 1:
            continue
        (binding,) = body.statements
        if (isinstance(binding, Binding) and binding.unpack and isinstance(call := binding.value, Call)
                and isinstance(call.callee, str) and call.callee in forms and not call.keywords
                and call.operands == tuple(map(Ref, body.inputs))
                and body.outputs == tuple(map(Ref, binding.targets))
                and len(forms[call.callee].outputs) == len(binding.targets)):
            target[name] = call.callee
    if not target:
        return forms

    def resolved(callee: str) -> str:
        # A chain's end: every forwarder found above, a rewrite of callees making none another.
        while callee in target:
            callee = target[callee]
        return callee

    def retargeted(statement):
        if (isinstance(statement, Binding) and isinstance(call := statement.value, Call)
                and isinstance(call.callee, str) and call.callee in target):
            moved = dataclasses.replace(call, callee=resolved(call.callee))
            return dataclasses.replace(statement, value=moved)
        return statement

    return {name: dataclasses.replace(body, statements=tuple(map(retargeted, body.statements)))
            for name, body in forms.items() if name not in target}


def bound_call(body: Body, shared_bodies: dict[str, Body],
               shapes: tuple[tuple[int, ...], ...]) -> Call | None:
    """Bind a tail call's known case arguments directly in the case table."""
    if len(body.statements) != 1 or not isinstance(binding := body.statements[0], Binding):
        return None
    if not isinstance(call := binding.value, Call) or not isinstance(call.callee, str):
        return None
    if (call.callee not in shared_bodies or not binding.unpack
            or body.outputs != tuple(map(Ref, binding.targets))):
        return None
    count = len(body.inputs)
    if call.keywords or call.operands[:count] != tuple(map(Ref, body.inputs)):
        return None
    sizes = dict(zip(body.inputs, shapes, strict=True))
    fixed = []
    bindings = zip(shared_bodies[call.callee].inputs[count:], call.operands[count:], strict=True)
    for parameter, value in bindings:
        if any(name not in sizes for name in _reads(value)) or isinstance(value, Ref | Literal | Call):
            return None
        if isinstance(value, Layout):
            value = StaticTuple(tuple(sizes[dim.operand.name][dim.axis] if isinstance(dim, Extent) else dim
                                      for dim in value.dimensions))
        elif isinstance(value, Extent):
            value = sizes[value.operand.name][value.axis]
        fixed.append((parameter, value))
    return Call(call.callee, (), tuple(fixed))
