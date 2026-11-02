"""Whether a traced forward is causal over its tokens, decided from its program alone, with no weights
(ledger 1595, the 1493 design's H7): tier 3 admits a source whose shrunken probe cannot run by a trace of
its full config over abstract weights, and proves each program it then runs here instead of reading its
causality from the probe's logits.

The walk composes, through every operation of the program (`jax.make_jaxpr`'s), which input positions
each entry of each value may read: a value the tokens reach holds an axis along which its entries read
their own sets of positions (`Reads`, Python integers as bit sets, any length), or reads one set
everywhere. Slices, concatenations, pads, reversals, gathers by indices the program computes without the
tokens, reshapes, broadcasts, cumulative operations, convolutions and scans (step by step, their carries
and initial state included) each move those sets as they move the entries; a contraction, reduction,
sort or selection across the positions' axis gives every entry the union, as does any operation the
walk holds no rule for (`_RULES`, `_ENTRYWISE`). Dew's causal attention (`CAUSAL_ATTENTION`) gives each
query its own position's set and its keys' up to its own; attention under a visibility its caller
states (`STATED_ATTENTION`) gives each position its own query's, key's and value's, the caller's
visibility being the call's and not the source's; a computation Dew states is token by token
(`TOKENWISE`) gives each token its own. Each of these three holds for a call of a program Dew's own
code bound under it (`attested`), identified as that very program and never by a name; a call of any
other program is walked inside. Every set is an over-approximation: the walk may refuse a causal
program, never admit a later read. A program is causal where every positional output's entry at a
position reads no later one; another length's or another call's program may differ (a branch taken on
the sequence's length), so each program a model runs is proven on its own."""

from __future__ import annotations

import dataclasses
import math
import weakref
from collections.abc import Callable, Mapping, Sequence

import jax
import numpy as np
from jax.core import ShapedArray
from jax.extend import core

from dew.nn.nested import nested

CAUSAL_ATTENTION = "causal attention"
"""The rule of a program tier 3's attention runs (`torchax_fallback._dew_attention`, `attested`), causal
over its keys by construction, its queries, keys and values `[B, S, H, D]`."""

STATED_ATTENTION = "stated attention"
"""The rule of a program tier 3's attention runs under a visibility its caller states (a decision row's
options, `torchax_fallback._Attending.visible`; `attested`), its queries, keys and values `[B, S, H, D]`:
which keys a query reads is the caller's, so the walk holds each position to its own."""

TOKENWISE = "token-wise"
"""The rule of a program Dew computes token by token (tier 3's experts' dispatch,
`torchax_fallback.functional_experts`; `attested`), whose outputs at a token read that token's inputs
alone."""

_ATTESTED: weakref.WeakKeyDictionary[core.Jaxpr, str] = weakref.WeakKeyDictionary()
"""Each program Dew's own code bound under a rule (`attested`), by that rule: the walk reads a call by
its rule where the call's program is one of these objects, as the program alone identifies what it
computes; a name any program may take."""


def attested[**P, R](function: Callable[P, R], rule: str) -> Callable[P, R]:
    """`function`, which Dew states computes as `rule` (`CAUSAL_ATTENTION`, `STATED_ATTENTION` or
    `TOKENWISE`) says, as a nested program (`dew.nn.nested.nested`) whose every staged call's program is
    recorded as `rule`'s, so a walk of the enclosing program reads that call by the rule."""
    assert rule in (CAUSAL_ATTENTION, STATED_ATTENTION, TOKENWISE), rule

    def record(program: core.Jaxpr) -> None:
        _ATTESTED[program] = rule

    return nested(function, traced=record)

CONCRETE = 1 << 20
"""The most entries of an integer value the walk evaluates while it walks (a gather's indices, a
slice's start), beyond which it reads none and keeps its rules' unions."""


@dataclasses.dataclass(frozen=True)
class Reads:
    """Which input positions a value the tokens reach may read: along `axis`, its entry at index i
    reads the positions `masks[i]` holds (bit j position j); where `axis` is None, every entry reads
    `masks[0]`."""

    axis: int | None
    masks: np.ndarray

    @property
    def union(self) -> int:
        return int(np.bitwise_or.reduce(self.masks))

    def flat(self) -> Reads:
        return Reads(None, np.array([self.union], object))


type Held = Reads | None
"""What the walk knows of a value: the positions it reads, or None where the tokens do not reach it."""


def causality_refusal(closed: core.Jaxpr, tokens: int, length: int,
                      positional: Sequence[int]) -> str | None:
    """Why the program `closed` is not provably causal at `length` tokens (its input `tokens`, `[B, S]`,
    the sequence on axis 1), or None where each of its `positional` outputs (by index, each holding its
    positions on axis 1: a forward's logits and states) reads at a position that position and earlier
    ones alone. It holds for this program alone: another length's, or another call's, may compute
    otherwise, so a caller proves each program it runs (`torchax_fallback.TorchGraph`)."""
    held: list[Held] = [None] * len(closed.jaxpr.invars)
    held[tokens] = Reads(1, np.array([1 << position for position in range(length)], object))
    walk = _Walk(length)
    consts = [np.asarray(const) if _small_integer(const) else None for const in closed.consts]
    # The values it evaluates (indices, their tests) computed now, also where a trace encloses the walk
    # (a program proven as it is traced, `torchax_fallback.TorchGraph`).
    with jax.ensure_compile_time_eval():
        outputs, _ = walk.run(closed.jaxpr, held, [None] * len(closed.jaxpr.invars), consts)
    for index in positional:
        output = outputs[index]
        if output is not None and _later(output, length):
            where = f"; first at {walk.first}" if walk.first else ""
            return f"its output {index}'s entry reads a later position at {length} tokens{where}"
    return None


def _ahead(value: Reads, length: int) -> bool:
    """Whether an entry of `value`, along its own axis of `length` entries, reads one past its index:
    the operation first doing so names a refusal (`_Walk.first`)."""
    assert value.axis is not None
    allowed = np.array([(1 << (index + 1)) - 1 for index in range(length)], object)
    return bool(np.any(value.masks & ~allowed))


def _later(value: Reads, length: int) -> bool:
    """Whether an entry of `value`, an output holding its positions on axis 1, reads a position later than
    its own: along that axis, entry i reads one past i; read along any other axis, or along none, every
    entry reads the union, which reads past position 0 where it holds any later one."""
    if value.axis != 1 or len(value.masks) != length:
        return bool(value.union & ~1)
    return _ahead(value, length)


def _programs(eqn: core.JaxprEqn) -> list[core.Jaxpr]:
    """The programs `eqn` runs within it: a call's, a scan's or loop's body, a branch's."""
    found = []
    for value in eqn.params.values():
        for program in value if isinstance(value, tuple | list) else (value,):
            if isinstance(program, core.Jaxpr):
                # A program holds its constants beside it on JAX 0.12, its `jaxpr` the program itself.
                found.append(program.jaxpr)
    return found


def _small_integer(value) -> bool:
    """Whether a program's constant is an integer or boolean array of at most `CONCRETE` entries, which
    the walk holds as its value."""
    return (isinstance(value, np.ndarray | jax.Array) and _indexing(value.dtype)
            and math.prod(value.shape) <= CONCRETE)


class _Walk:
    def __init__(self, length: int):
        self.length = length
        self.first = ""
        """The first operation whose output's entry at a position reads a later position, for the
        refusal's account."""

    def run(self, jaxpr: core.Jaxpr, inputs: Sequence[Held], values: Sequence[np.ndarray | None],
            consts: Sequence[np.ndarray | None] = ()) -> tuple[list[Held], list[np.ndarray | None]]:
        """`jaxpr`'s outputs' reads given its inputs', and their values where the walk evaluates them."""
        held: dict = dict(zip(jaxpr.invars, inputs, strict=True))
        known: dict = dict(zip(jaxpr.invars, values, strict=True))
        held.update(dict.fromkeys(jaxpr.constvars))
        known.update(zip(jaxpr.constvars, list(consts) or [None] * len(jaxpr.constvars), strict=True))
        for eqn in jaxpr.eqns:
            given = [None if isinstance(var, core.Literal) else held.get(var) for var in eqn.invars]
            operands = [np.asarray(var.val) if isinstance(var, core.Literal) else known.get(var)
                        for var in eqn.invars]
            outputs, evaluated = self.equation(eqn, given, operands)
            held.update(zip(eqn.outvars, outputs, strict=True))
            known.update(zip(eqn.outvars, evaluated, strict=True))
            if not self.first and any(isinstance(value, Reads) and value.axis is not None
                                      and len(value.masks) == self.length and _ahead(value, self.length)
                                      for value in outputs):
                self.first = eqn.primitive.name
        return ([None if isinstance(var, core.Literal) else held.get(var) for var in jaxpr.outvars],
                [np.asarray(var.val) if isinstance(var, core.Literal) else known.get(var)
                 for var in jaxpr.outvars])

    def equation(self, eqn: core.JaxprEqn, given: list[Held],
                 operands: list[np.ndarray | None]) -> tuple[list[Held], list[np.ndarray | None]]:
        name = eqn.primitive.name
        if name in _CALLS:
            return _call(self, eqn, given, operands)
        if not any(isinstance(value, Reads) for value in given):
            return [None] * len(eqn.outvars), self.evaluated(eqn, operands)
        rule = _RULES.get(name)
        if rule is not None:
            return rule(self, eqn, given, operands), [None] * len(eqn.outvars)
        shape = _shape(eqn.outvars[0])
        if name in _ENTRYWISE and all(_shape(var) == shape for var in eqn.outvars) and all(
                _fits(_shape(var), shape) for var, value in zip(eqn.invars, given, strict=True)
                if isinstance(value, Reads)):
            return [_combined(given, shape)] * len(eqn.outvars), [None] * len(eqn.outvars)
        return [_flat(given)] * len(eqn.outvars), [None] * len(eqn.outvars)

    def evaluated(self, eqn: core.JaxprEqn, operands: list[np.ndarray | None]) -> list[np.ndarray | None]:
        """The outputs' values of an operation the tokens do not reach whose operands the walk holds,
        where they are integers or booleans of at most `CONCRETE` entries: the indices a gather, a slice
        or an update reads, and the tests that choose them."""
        if (any(operand is None for operand in operands) or _programs(eqn)
                or not all(isinstance(var.aval, ShapedArray) and _indexing(var.aval.dtype)
                           and math.prod(_shape(var)) <= CONCRETE for var in eqn.outvars)):
            return [None] * len(eqn.outvars)
        bound = eqn.primitive.bind(*operands, **eqn.params)
        return [np.asarray(output) for output in (bound if eqn.primitive.multiple_results else [bound])]


def _shape(var) -> tuple[int, ...]:
    return tuple(var.aval.shape) if isinstance(var.aval, ShapedArray) else ()


def _indexing(dtype: np.dtype | jax.typing.DTypeLike) -> bool:
    """Whether `dtype` is an integer or boolean one, which indices and their tests hold (a key's
    extended dtype is neither)."""
    return isinstance(dtype, np.dtype) and (np.issubdtype(dtype, np.integer) or dtype == np.bool_)


def _fits(operand: tuple[int, ...], out: tuple[int, ...]) -> bool:
    """Whether an entrywise operand of shape `operand` broadcasts to `out` entry for entry: a scalar,
    or a shape of its rank whose every axis is the output's or 1."""
    if operand == ():
        return True
    return len(operand) == len(out) and all(size in (1, full)
                                            for size, full in zip(operand, out, strict=True))


def _flat(given: Sequence[Held]) -> Reads:
    """Every entry reading what any of `given` reads."""
    mask = 0
    for value in given:
        if isinstance(value, Reads):
            mask |= value.union
    return Reads(None, np.array([mask], object))


def _combined(given: Sequence[Held], shape: tuple[int, ...]) -> Reads:
    """An entrywise operation's reads over operands of `shape` (or scalars): the positions' axis the
    operands share keeps each index's union; any other operand's reads join every entry."""
    axes = [value for value in given if isinstance(value, Reads) and value.axis is not None
            and len(value.masks) == shape[value.axis]]
    base = axes[0] if axes else None
    extra = 0
    for value in given:
        if isinstance(value, Reads) and (base is None or value is not base):
            if base is not None and value.axis == base.axis and len(value.masks) == len(base.masks):
                base = Reads(base.axis, base.masks | value.masks)
            else:
                extra |= value.union
    if base is None:
        return Reads(None, np.array([extra], object))
    return Reads(base.axis, base.masks | extra)


def _call(walk: _Walk, eqn: core.JaxprEqn, given: list[Held],
          operands: list[np.ndarray | None]) -> tuple[list[Held], list[np.ndarray | None]]:
    """A call of a program of its own: a jit of a program Dew's code bound under a rule (`attested`) by
    that rule, any other walked inside."""
    program = eqn.params.get("jaxpr")
    rule = (_ATTESTED.get(program) if eqn.primitive.name in ("jit", "pjit")
            and isinstance(program, core.Jaxpr) else None)
    if rule in (CAUSAL_ATTENTION, STATED_ATTENTION) and any(isinstance(value, Reads) for value in given):
        return _attended(eqn, given, causal=rule == CAUSAL_ATTENTION), [None] * len(eqn.outvars)
    if rule == TOKENWISE and any(isinstance(value, Reads) for value in given):
        out = _shape(eqn.outvars[0])
        aligned = [value for value in given if isinstance(value, Reads)]
        if all(value.axis == 0 and len(value.masks) == out[0] for value in aligned):
            masks = np.bitwise_or.reduce([value.masks for value in aligned])
            return [Reads(0, masks)] * len(eqn.outvars), [None] * len(eqn.outvars)
        return [_flat(given)] * len(eqn.outvars), [None] * len(eqn.outvars)
    inner = next(iter(_programs(eqn)), None)
    if inner is None:
        return [_flat(given) if any(given) else None] * len(eqn.outvars), [None] * len(eqn.outvars)
    closed = next((value for value in eqn.params.values() if isinstance(value, core.Jaxpr)), None)
    consts = [] if closed is None else [np.asarray(const) if _small_integer(const) else None
                                        for const in closed.consts]
    return walk.run(inner, given, operands, consts)


def _attended(eqn: core.JaxprEqn, given: list[Held], *, causal: bool) -> list[Held]:
    """Dew's attention's outputs, its queries `[B, S, H, D]` its first operand: an output of the
    queries' positions (`[B, S, ...]`) reads at each position its own query's positions and its keys'
    and values' up to its own (`causal`), or its own (a visibility its caller states); any other output
    (a cache it writes) every position any operand reads."""
    length = _shape(eqn.invars[0])[1]
    own, keyed, extra = (np.zeros(length, object), np.zeros(length, object), 0)
    for index, value in enumerate(given):
        if not isinstance(value, Reads):
            continue
        if value.axis == 1 and len(value.masks) == length:
            if index == 0 or not causal:
                own = own | value.masks
            else:
                keyed = keyed | value.masks
        else:
            extra |= value.union
    rows = own | np.bitwise_or.accumulate(keyed) | extra
    flat = _flat(given)
    return [Reads(1, rows) if len(_shape(var)) > 1 and _shape(var)[1] == length else flat
            for var in eqn.outvars]


def _rows(value: Reads) -> np.ndarray:
    return value.masks


def _broadcast_in_dim(walk: _Walk, eqn, given, operands) -> list[Held]:
    value, dims, out = given[0], eqn.params["broadcast_dimensions"], _shape(eqn.outvars[0])
    assert isinstance(value, Reads)
    if value.axis is None:
        return [value]
    axis = dims[value.axis]
    masks = value.masks if len(value.masks) == out[axis] else np.full(out[axis], value.union, object)
    return [Reads(axis, masks)]


def _reshape(walk: _Walk, eqn, given, operands) -> list[Held]:
    value = given[0]
    assert isinstance(value, Reads)
    if value.axis is None:
        return [value]
    before, after = _shape(eqn.invars[0]), tuple(eqn.params["new_sizes"])
    order = eqn.params.get("dimensions")
    if order is not None:
        # Its operand transposed by `dimensions` first, then laid out row-major.
        before = tuple(before[dim] for dim in order)
        value = Reads(tuple(order).index(value.axis), value.masks)
    return [_moved(value, before, after)]


def _moved(value: Reads, before: tuple[int, ...], after: tuple[int, ...]) -> Reads:
    """`value`'s reads once its entries, in row-major order, take the shape `after`: along each new axis,
    each index reads the union of the rows its entries come from; the axis whose unions are the
    tightest holds them (exactly where its index alone decides the row: the old axis kept, merged with
    others, or split)."""
    axis = value.axis
    assert axis is not None
    count, inner = len(value.masks), math.prod(before[axis + 1:])
    span = _RangeOr(value.masks)
    best: tuple[int, Reads] | None = None
    for index, width in enumerate(after):
        lead, tail = math.prod(after[:index]), math.prod(after[index + 1:])
        if lead * width > CONCRETE:
            continue
        starts = (np.arange(lead)[:, None] * width + np.arange(width)[None, :]) * tail
        rows = np.bitwise_or.reduce(span(starts // inner, (starts + tail - 1) // inner), axis=0)
        if best is None or _tighter(rows, best[1].masks):
            best = (index, Reads(index, rows.astype(object)))
    return value.flat() if best is None or count == 0 else best[1]


def _tighter(rows: np.ndarray, held: np.ndarray) -> bool:
    """Whether `rows` give their indices fewer positions each, on average, than `held` (the finer axis
    where they tie): the axis a value's reads are kept along."""
    def mean(masks: np.ndarray) -> float:
        return sum(bin(int(mask)).count("1") for mask in masks) / len(masks)
    return (mean(rows), -len(rows)) < (mean(held), -len(held))


class _RangeOr:
    """The union of `masks`' rows over each range of row indices, taken modulo their count (a sparse
    table over them twice over), every row where a range covers them all."""

    def __init__(self, masks: np.ndarray):
        self.count = len(masks)
        self.union = int(np.bitwise_or.reduce(masks)) if self.count else 0
        table = [np.concatenate([masks, masks]).astype(object)]
        while (1 << len(table)) <= 2 * self.count:
            last, step = table[-1], 1 << (len(table) - 1)
            table.append(last[:-step] | last[step:])
        self.table = table

    def __call__(self, first: np.ndarray, last: np.ndarray) -> np.ndarray:
        size = last - first + 1
        whole = size >= self.count
        start = first % self.count
        clipped = np.where(whole, 1, size)
        level = np.floor(np.log2(clipped)).astype(int)
        out = np.zeros(first.shape, object)
        for height in np.unique(level):
            at = level == height
            row = self.table[height]
            out[at] = row[start[at]] | row[start[at] + clipped[at] - (1 << height)]
        return np.where(whole, self.union, out)


def _squeeze(walk: _Walk, eqn, given, operands) -> list[Held]:
    value = given[0]
    assert isinstance(value, Reads)
    if value.axis is None:
        return [value]
    dims = eqn.params["dimensions"]
    if value.axis in dims:
        return [value.flat()]
    return [Reads(value.axis - sum(1 for dim in dims if dim < value.axis), value.masks)]


def _transpose(walk: _Walk, eqn, given, operands) -> list[Held]:
    value, order = given[0], tuple(eqn.params["permutation"])
    assert isinstance(value, Reads)
    return [value if value.axis is None else Reads(order.index(value.axis), value.masks)]


def _reduce(walk: _Walk, eqn, given, operands) -> list[Held]:
    value, axes = _combined(given, _shape(eqn.invars[0])), tuple(eqn.params["axes"])
    if value.axis is None or value.axis in axes:
        return [value.flat()] * len(eqn.outvars)
    return [Reads(value.axis - sum(1 for dim in axes if dim < value.axis), value.masks)] * len(eqn.outvars)


def _cumulative(walk: _Walk, eqn, given, operands) -> list[Held]:
    value = given[0]
    assert isinstance(value, Reads)
    if value.axis != eqn.params["axis"]:
        return [value]
    if eqn.params.get("reverse", False):
        return [Reads(value.axis, np.bitwise_or.accumulate(value.masks[::-1])[::-1])]
    return [Reads(value.axis, np.bitwise_or.accumulate(value.masks))]


def _dot_general(walk: _Walk, eqn, given, operands) -> list[Held]:
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = eqn.params["dimension_numbers"]
    ranks = len(_shape(eqn.invars[0])), len(_shape(eqn.invars[1]))
    free = [[dim for dim in range(rank) if dim not in contract and dim not in batch]
            for rank, contract, batch in zip(ranks, (lhs_contract, rhs_contract), (lhs_batch, rhs_batch),
                                              strict=True)]
    placed, extra = [], 0
    for side, value in enumerate(given[:2]):
        if not isinstance(value, Reads):
            continue
        contract, batch = (lhs_contract, rhs_contract)[side], (lhs_batch, rhs_batch)[side]
        if value.axis is None or value.axis in contract:
            extra |= value.union
        elif value.axis in batch:
            placed.append(Reads(list(batch).index(value.axis), value.masks))
        else:
            start = len(lhs_batch) + (0 if side == 0 else len(free[0]))
            placed.append(Reads(start + free[side].index(value.axis), value.masks))
    if not placed:
        return [Reads(None, np.array([extra], object))]
    base = placed[0]
    for other in placed[1:]:
        if other.axis == base.axis:
            base = Reads(base.axis, base.masks | other.masks)
        else:
            extra |= other.union
    return [Reads(base.axis, base.masks | extra)]


def _conv(walk: _Walk, eqn, given, operands) -> list[Held]:
    lhs, rhs = given[0], given[1]
    if isinstance(rhs, Reads) or not isinstance(lhs, Reads) or lhs.axis is None:
        return [_flat(given)]
    numbers = eqn.params["dimension_numbers"]
    lhs_spec, rhs_spec, out_spec = numbers.lhs_spec, numbers.rhs_spec, numbers.out_spec
    if lhs.axis == lhs_spec[0]:
        if eqn.params.get("batch_group_count", 1) != 1:
            # Grouping the batch moves its entries into the output's features.
            return [lhs.flat()]
        return [Reads(out_spec[0], lhs.masks)]
    if lhs.axis == lhs_spec[1]:
        return [lhs.flat()]
    spatial = list(lhs_spec[2:]).index(lhs.axis)
    low, _ = eqn.params["padding"][spatial]
    width = _shape(eqn.invars[1])[rhs_spec[2 + spatial]]
    dilation, stride = eqn.params["rhs_dilation"][spatial], eqn.params["window_strides"][spatial]
    if eqn.params["lhs_dilation"][spatial] != 1:
        return [lhs.flat()]
    size = _shape(eqn.outvars[0])[out_spec[2 + spatial]]
    rows = []
    for out in range(size):
        mask = 0
        for tap in range(width):
            read = out * stride - low + tap * dilation
            if 0 <= read < len(lhs.masks):
                mask |= int(lhs.masks[read])
        rows.append(mask)
    return [Reads(out_spec[2 + spatial], np.array(rows, object))]


def _slice(walk: _Walk, eqn, given, operands) -> list[Held]:
    value = given[0]
    assert isinstance(value, Reads)
    if value.axis is None:
        return [value]
    start, limit = eqn.params["start_indices"][value.axis], eqn.params["limit_indices"][value.axis]
    strides = eqn.params.get("strides")
    step = 1 if strides is None else strides[value.axis]
    return [Reads(value.axis, value.masks[start:limit:step])]


def _rev(walk: _Walk, eqn, given, operands) -> list[Held]:
    value = given[0]
    assert isinstance(value, Reads)
    if value.axis is None or value.axis not in eqn.params["dimensions"]:
        return [value]
    return [Reads(value.axis, value.masks[::-1])]


def _pad(walk: _Walk, eqn, given, operands) -> list[Held]:
    value, padding = given[0], given[1]
    if isinstance(padding, Reads) or not isinstance(value, Reads) or value.axis is None:
        return [_flat(given)]
    low, high, interior = eqn.params["padding_config"][value.axis]
    rows: list[int] = []
    for index, mask in enumerate(value.masks):
        rows.append(int(mask))
        if index < len(value.masks) - 1:
            rows.extend([0] * interior)
    rows = [0] * max(low, 0) + rows[max(-low, 0):]
    rows = rows[:len(rows) + high] if high < 0 else rows + [0] * high
    return [Reads(value.axis, np.array(rows, object))]


def _concatenate(walk: _Walk, eqn, given, operands) -> list[Held]:
    dimension = eqn.params["dimension"]
    axes = {value.axis for value in given if isinstance(value, Reads)}
    if axes != {dimension}:
        flat = _flat(given)
        along = [value for value in given if isinstance(value, Reads) and value.axis is not None]
        if along and all(value.axis == along[0].axis for value in along) and dimension != along[0].axis:
            return [_combined(given, _shape(eqn.outvars[0]))]
        return [flat]
    rows = [value.masks if isinstance(value, Reads) else np.zeros(_shape(var)[dimension], object)
            for var, value in zip(eqn.invars, given, strict=True)]
    return [Reads(dimension, np.concatenate(rows))]


def _dynamic_slice(walk: _Walk, eqn, given, operands) -> list[Held]:
    value = given[0]
    if not isinstance(value, Reads) or any(isinstance(start, Reads) for start in given[1:]):
        return [_flat(given)]
    if value.axis is None:
        return [value]
    start = operands[1 + value.axis]
    size = eqn.params["slice_sizes"][value.axis]
    if start is None:
        return [Reads(value.axis, np.full(size, value.union, object))]
    first = int(np.clip(int(start), 0, len(value.masks) - size))
    return [Reads(value.axis, value.masks[first:first + size])]


def _dynamic_update_slice(walk: _Walk, eqn, given, operands) -> list[Held]:
    base, update = given[0], given[1]
    if any(isinstance(start, Reads) for start in given[2:]):
        return [_flat(given)]
    axis = next((value.axis for value in (base, update) if isinstance(value, Reads)), None)
    if axis is None or any(isinstance(value, Reads) and value.axis != axis for value in (base, update)):
        return [_flat(given)]
    length = _shape(eqn.outvars[0])[axis]
    rows = base.masks.copy() if isinstance(base, Reads) else np.zeros(length, object)
    width = _shape(eqn.invars[1])[axis]
    start = operands[2 + axis]
    if start is None:
        rows |= _flat([update]).masks[0] if isinstance(update, Reads) else 0
        return [Reads(axis, rows)]
    first = int(np.clip(int(start), 0, length - width))
    updated = update.masks if isinstance(update, Reads) else np.zeros(width, object)
    others = [size for dim, size in enumerate(_shape(eqn.outvars[0])) if dim != axis]
    covered = [size for dim, size in enumerate(_shape(eqn.invars[1])) if dim != axis]
    if covered == others:
        # The update overwrites each of its rows whole: they read what it reads alone.
        rows[first:first + width] = updated
    else:
        rows[first:first + width] = rows[first:first + width] | updated
    return [Reads(axis, rows)]


def _gather(walk: _Walk, eqn, given, operands) -> list[Held]:
    """A gather: by indices the walk holds, each output entry reads the operand rows it takes, moved
    to the output's axis that decides them; by indices the tokens reach (an embedding's lookup), each
    output entry reads its index's positions; otherwise the union."""
    operand, indices = given[0], given[1]
    numbers = eqn.params["dimension_numbers"]
    out = _shape(eqn.outvars[0])
    batch_out = [dim for dim in range(len(out)) if dim not in tuple(numbers.offset_dims)]
    if isinstance(indices, Reads):
        if isinstance(operand, Reads) or indices.axis is None:
            return [_flat(given)]
        vector = len(_shape(eqn.invars[1])) - 1
        if indices.axis == vector:
            return [indices.flat()]
        return [Reads(batch_out[indices.axis], indices.masks)]
    assert isinstance(operand, Reads)
    if operand.axis is None:
        return [operand]
    held = operands[1]
    shape = _shape(eqn.invars[0])
    if held is None or math.prod(shape) > CONCRETE or math.prod(out) > CONCRETE:
        return [operand.flat()]
    ids = np.broadcast_to(np.arange(shape[operand.axis]).reshape(
        [-1 if dim == operand.axis else 1 for dim in range(len(shape))]), shape)
    taken = np.asarray(jax.lax.gather(ids, held, numbers, eqn.params["slice_sizes"],
                                      mode=eqn.params["mode"], fill_value=-1))
    return [_decided(taken, operand.masks)]


def _decided(ids: np.ndarray, masks: np.ndarray) -> Reads:
    """Reads of a value whose entries hold the operand rows `ids` (-1 for none): along its axis whose
    index decides the rows, each index's union; the axis giving the fewest positions where none does
    alone."""
    best: Reads | None = None
    for axis in range(ids.ndim):
        moved = np.moveaxis(ids, axis, 0).reshape(ids.shape[axis], -1)
        rows = np.array([int(np.bitwise_or.reduce(np.append(masks[row[row >= 0]], 0))) for row in moved],
                        object)
        if best is None or _tighter(rows, best.masks):
            best = Reads(axis, rows)
    if best is None:
        return Reads(None, np.array([int(np.bitwise_or.reduce(np.append(masks[ids[ids >= 0]], 0)))],
                                    object))
    return best


def _sort(walk: _Walk, eqn, given, operands) -> list[Held]:
    value = _combined(given, _shape(eqn.invars[0]))
    if value.axis == eqn.params["dimension"]:
        return [value.flat()] * len(eqn.outvars)
    return [value] * len(eqn.outvars)


def _top_k(walk: _Walk, eqn, given, operands) -> list[Held]:
    value = given[0]
    assert isinstance(value, Reads)
    if value.axis == len(_shape(eqn.invars[0])) - 1:
        return [value.flat()] * len(eqn.outvars)
    return [value] * len(eqn.outvars)


def _split(walk: _Walk, eqn, given, operands) -> list[Held]:
    value = given[0]
    assert isinstance(value, Reads)
    if value.axis != eqn.params["axis"]:
        return [value] * len(eqn.outvars)
    pieces, start = [], 0
    for size in eqn.params["sizes"]:
        pieces.append(Reads(value.axis, value.masks[start:start + size]))
        start += size
    return pieces


def _scatter(walk: _Walk, eqn, given, operands) -> list[Held]:
    return [_flat(given)]


def _scan_groups(eqn: core.JaxprEqn) -> tuple[int, int]:
    """A scan's constant and carried inputs' counts: its parameters', or, where it states its inputs'
    and outputs' groups (`ft_in`, `ft_out`), theirs."""
    params = eqn.params
    if "num_consts" in params:
        return params["num_consts"], params["num_carry"]
    consts, carried, _ = (len(group) for group in params["ft_in"].elts)
    return consts, carried


def _joined(held: Held, found: Held) -> Held:
    """The reads of a loop's carried value over two passes: each entry the union of both."""
    if held is None or found is None:
        return found if held is None else held
    if held.axis == found.axis and len(held.masks) == len(found.masks):
        return Reads(held.axis, held.masks | found.masks)
    return _flat([held, found])


def _settled(walk: _Walk, body: core.Jaxpr, consts: list[Held], carried: list[Held], rest: list[Held],
             values: Sequence[np.ndarray | None]) -> tuple[list[Held], list[Held]]:
    """A loop's carried values over every number of passes, none included (each the union of its initial
    reads and every pass's, which only grow over finitely many positions), and `body`'s outputs past its
    carries over those values."""
    while True:
        out, _ = walk.run(body, [*consts, *carried, *rest], values)
        grown = [_joined(held, found) for held, found in zip(carried, out[:len(carried)], strict=True)]
        if all(_same(held, found) for held, found in zip(grown, carried, strict=True)):
            return grown, out[len(carried):]
        carried = grown


def _same(first: Held, second: Held) -> bool:
    if first is None or second is None:
        return first is second
    return first.axis == second.axis and np.array_equal(first.masks, second.masks)


def _scan(walk: _Walk, eqn, given, operands) -> list[Held]:
    """A scan: over the positions (a scanned input the tokens reach on its axis 0), step by step in its
    order, each step's output reading its carry, which holds its initial state's reads and every earlier
    step's, and that step's inputs; over any other axis, its body walked until its carried reads settle."""
    params = eqn.params
    consts, carries = _scan_groups(eqn)
    body = params["jaxpr"].jaxpr
    fixed, carried = list(given[:consts]), list(given[consts:consts + carries])
    xs = list(given[consts + carries:])
    length = params["length"]
    values = [None] * len(body.invars)
    if any(isinstance(value, Reads) and value.axis == 0 for value in xs):
        steps: list[list[Held]] = []
        order = range(length - 1, -1, -1) if params["reverse"] else range(length)
        for step in order:
            sliced = [value if not isinstance(value, Reads) else
                      Reads(None, value.masks[step:step + 1]) if value.axis == 0 else
                      Reads(None if value.axis is None else value.axis - 1, value.masks) for value in xs]
            out, _ = walk.run(body, [*fixed, *carried, *sliced], values)
            carried = list(out[:carries])
            steps.append(list(out[carries:]))
        if params["reverse"]:
            steps.reverse()
        ys = []
        for index in range(len(eqn.outvars) - carries):
            per_step = [step[index] for step in steps]
            if all(value is None for value in per_step):
                ys.append(None)
            elif all(value is None or value.axis is None for value in per_step):
                ys.append(Reads(0, np.array([0 if value is None else value.union for value in per_step],
                                            object)))
            else:
                ys.append(_flat(per_step))
        return [*carried, *ys]
    sliced = [value if not isinstance(value, Reads) or value.axis is None
              else Reads(value.axis - 1, value.masks) for value in xs]
    final, out = _settled(walk, body, fixed, carried, sliced, values)
    ys = [value if not isinstance(value, Reads) or value.axis is None
          else Reads(value.axis + 1, value.masks) for value in out]
    return [*final, *ys]


def _while(walk: _Walk, eqn, given, operands) -> list[Held]:
    params = eqn.params
    cond_consts, body_consts = params["cond_nconsts"], params["body_nconsts"]
    carried = list(given[cond_consts + body_consts:])
    body = params["body_jaxpr"].jaxpr
    # Its state on exit, after any number of passes, none included.
    out, _ = _settled(walk, body, list(given[cond_consts:cond_consts + body_consts]), carried, [],
                      [None] * len(body.invars))
    cond, _ = walk.run(params["cond_jaxpr"].jaxpr, [*given[:cond_consts], *out],
                       [None] * len(params["cond_jaxpr"].jaxpr.invars))
    if any(isinstance(value, Reads) for value in [*cond, *given[:cond_consts]]):
        # How many passes run is the tokens' to decide: every carried entry may read every one.
        return [_flat([*given, *out])] * len(out)
    return out


def _cond(walk: _Walk, eqn, given, operands) -> list[Held]:
    branches = eqn.params["branches"]
    outs = [walk.run(branch.jaxpr, given[1:], operands[1:])[0] for branch in branches]
    merged = []
    for values in zip(*outs, strict=True):
        held: Held = None
        for value in values:
            held = _joined(held, value)
        merged.append(held)
    if isinstance(given[0], Reads):
        # The branch is the tokens' to choose: every entry may read what chose it.
        return [_flat([value, given[0]]) for value in merged]
    return merged


_CALLS = frozenset({"pjit", "jit", "closed_call", "core_call", "custom_jvp_call", "custom_vjp_call",
                    "custom_vjp_call_jaxpr", "remat", "checkpoint", "custom_lin"})

_ENTRYWISE = frozenset({
    "add", "sub", "mul", "div", "rem", "pow", "max", "min", "neg", "abs", "sign", "floor", "ceil", "round",
    "exp", "exp2", "log", "log1p", "expm1", "tanh", "logistic", "sin", "cos", "tan", "asin", "acos", "atan",
    "atan2", "sinh", "cosh", "asinh", "acosh", "atanh", "sqrt", "rsqrt", "cbrt", "square", "integer_pow",
    "erf", "erfc", "erf_inv", "lgamma", "digamma", "is_finite", "not", "and", "or", "xor", "eq", "ne", "lt",
    "le", "gt", "ge", "select_n", "convert_element_type", "copy", "copy_p", "reduce_precision", "clamp",
    "nextafter", "shift_left", "shift_right_logical", "shift_right_arithmetic", "population_count", "clz",
    "real", "imag", "conj", "complex", "stop_gradient", "random_wrap", "random_unwrap", "random_bits",
    "random_fold_in", "random_split", "threefry2x32", "polygamma", "igamma", "igammac",
    "bessel_i0e", "bessel_i1e", "zeta"})
"""The operations whose every output entry reads its operands' entries at its own index (over operands
broadcast to its shape) and nothing else: the walk keeps their positions' axis. Any operation it holds
no rule for gives every entry the union of what its operands read."""

type _Rule = Callable[[_Walk, core.JaxprEqn, list[Held], list[np.ndarray | None]], list[Held]]

_RULES: Mapping[str, _Rule] = {
    "broadcast_in_dim": _broadcast_in_dim, "reshape": _reshape, "squeeze": _squeeze, "transpose": _transpose,
    **dict.fromkeys(("reduce_sum", "reduce_max", "reduce_min", "reduce_prod", "reduce_and", "reduce_or",
                     "reduce_xor", "argmax", "argmin"), _reduce),
    **dict.fromkeys(("cumsum", "cumprod", "cummax", "cummin", "cumlogsumexp"), _cumulative),
    "dot_general": _dot_general, "conv_general_dilated": _conv, "gather": _gather, "sort": _sort,
    "top_k": _top_k, "slice": _slice, "rev": _rev, "pad": _pad, "concatenate": _concatenate,
    "dynamic_slice": _dynamic_slice, "dynamic_update_slice": _dynamic_update_slice, "split": _split,
    **dict.fromkeys(("scatter", "scatter-add", "scatter_add", "scatter-mul", "scatter_mul", "scatter-min",
                     "scatter_min", "scatter-max", "scatter_max", "scatter_apply"), _scatter),
    "scan": _scan, "while": _while, "cond": _cond,
}
