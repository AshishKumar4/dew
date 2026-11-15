"""Whether a traced forward is causal over its tokens, decided from its program alone, with no weights
(ledger 1595, the 1493 design's H7): tier 3 admits a source whose shrunken probe cannot run by a trace of
its full config over abstract weights, and proves each program it then runs here instead of reading its
causality from the probe's logits.

The walk composes, through every operation of the program (`jax.make_jaxpr`'s), the latest input
position each entry of each value may read: a value the tokens reach holds an axis along which its
entries read up to their own latest positions (`Reads`), or reads up to one everywhere. Every rule joins
what its operands read, and the latest of a join is the largest of theirs, so the latest alone decides
causality as the whole set of positions read would. Slices, concatenations, pads, reversals, gathers by
indices the program computes without the tokens, reshapes, broadcasts, cumulative operations, convolutions and
scans (step by step, their carries and initial state included) each move those positions as they move the
entries; a contraction, reduction, sort or selection across the positions' axis gives every entry the latest
of all, as does any operation the walk holds no rule for (`_RULES`, `dew.nn.programs.positionwise`). A
contraction of two values along their positions on two axes of its own (attention's scores written out)
reads along both (`Pair`), so a mask that lowers each later key's score to minus infinity or its dtype's
lowest zeroes its softmax there, and the weighted values read no later key. Dew's
causal attention (`CAUSAL_ATTENTION`) gives each query its own position's latest and its keys' up to its own;
attention under a visibility its caller states (`STATED_ATTENTION`) gives each position its own query's, key's
and value's, the caller's visibility being the call's and not the source's; a computation Dew states is token
by token (`TOKENWISE`) gives each token its own. Each of these three holds for a call of a program Dew's own
code bound under it (`attested`), identified as that very program and never by a name; a call of any other
program is walked inside (`dew.nn.programs.called`).
Every position is an over-approximation: the walk may refuse a causal program, never admit a later
read. A program is causal where every positional output's entry at a
position reads no later one; another length's or another call's program may differ (a branch taken on
the sequence's length), so each program a model runs is proven on its own."""

from __future__ import annotations

import dataclasses
import functools
import math
import weakref
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import ShapedArray
from jax.extend import core
from jax.extend.core import primitives

from dew.nn.nested import nested
from dew.nn.programs import (
    bodies,
    branches,
    called,
    cumulative,
    loop_programs,
    on_host,
    positionwise,
    reduction,
    scan_body,
    scatter,
)

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
slice's start), and of a reshaped axis it sizes anew (`_moved`), beyond which it reads none and keeps
its rules' joins."""

NONE = -1
"""The latest position an entry reads where it reads none."""


@dataclasses.dataclass(frozen=True)
class Reads:
    """The latest input position a value the tokens reach may read: along `axis`, its entry at index i
    reads none past `latest[i]` (`NONE` for none); where `axis` is None, every entry reads none past
    `latest[0]`."""

    axis: int | None
    latest: np.ndarray

    @property
    def last(self) -> int:
        """The latest position any entry reads."""
        return int(self.latest.max(initial=NONE))

    def flat(self) -> Reads:
        return _everywhere(self.last)


def _everywhere(latest: int) -> Reads:
    """Every entry reading up to `latest`."""
    return Reads(None, np.array([latest], np.int64))


def _none(size: int) -> np.ndarray:
    """`size` entries reading no position."""
    return np.full(size, NONE, np.int64)


def _read_at(latest: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """The latest position each of `rows` (-1 for none, of any shape) reads among `latest`'s, which may
    hold none."""
    if len(latest) == 0:
        return np.full(rows.shape, NONE, np.int64)
    return np.where(rows >= 0, latest[np.clip(rows, 0, len(latest) - 1)], NONE)


@dataclasses.dataclass(frozen=True)
class Pair:
    """A value whose entries read along two of its axes (`axes`, a query's and a key's): its entry at
    (i, j) along them reads none past `latest[i, j]`, its value lies within `low[i, j]` and `high[i, j]`,
    and where `zero[i, j]` holds it is exactly zero, reading nothing (a product or a sum of it reads nothing
    of what made it). A contraction of two values the tokens reach along their positions makes one
    (attention's scores), its entries within a fraction of their dtype's largest (`_BOUND`); a mask's
    lowest values added to a score, or chosen in its place, bound it far below every unmasked score of its
    row, so that its softmax is exactly zero there, whatever the later key it scores read (masked
    attention written out, transformers' own). Every value but a mask's is finite, so a zero times any
    other value is zero; an operation the walk holds no such rule for reads it as `collapsed`."""

    axes: tuple[int, int]
    latest: np.ndarray
    low: np.ndarray
    high: np.ndarray
    zero: np.ndarray

    @property
    def last(self) -> int:
        return int(self.latest.max(initial=NONE))

    def collapsed(self) -> Reads:
        """Its reads along its first axis alone, each index the latest of its row."""
        return Reads(self.axes[0], self.latest.max(axis=1, initial=NONE))


_BOUND = 2.0 ** -10
"""The fraction of its dtype's largest value that bounds a contraction's entries along two positions'
axes (a score), which a mask's lowest value lies far below: a score past it overflows its softmax."""


type Held = Reads | Pair | None
"""What the walk knows of a value: the positions it reads, or None where the tokens do not reach it."""


def _reads(value: Held) -> Reads | None:
    """`value`'s reads along one axis at most: a `Pair` `collapsed`."""
    return value.collapsed() if isinstance(value, Pair) else value


def causality_refusal(closed: core.Jaxpr, tokens: int, length: int,
                      positional: Sequence[int]) -> str | None:
    """Why the program `closed` is not provably causal at `length` tokens (its input `tokens`, `[B, S]`,
    the sequence on axis 1), or None where each of its `positional` outputs (by index, each holding its
    positions on axis 1: a forward's logits and states) reads at a position that position and earlier
    ones alone. It holds for this program alone: another length's, or another call's, may compute
    otherwise, so a caller proves each program it runs (`torchax_fallback.TorchGraph`)."""
    held: list[Held] = [None] * len(closed.jaxpr.invars)
    held[tokens] = Reads(1, np.arange(length, dtype=np.int64))
    walk = _Walk(length)
    consts = [np.asarray(const) if _small_held(const) else None for const in closed.consts]
    # The values it evaluates (indices, their tests) computed now on this host, also where a trace and its
    # mesh enclose the walk (a program proven as it is traced, `torchax_fallback.TorchGraph`; a compile-only
    # mesh's devices run nothing here: `programs.on_host`).
    with on_host():
        outputs, _ = walk.run(closed.jaxpr, held, [None] * len(closed.jaxpr.invars), consts)
    for index in positional:
        output = _reads(outputs[index])
        if output is not None and _later(output, length):
            where = f"; first at {walk.first}" if walk.first else ""
            return f"its output {index}'s entry reads a later position at {length} tokens{where}"
    return None


def _ahead(value: Reads, length: int) -> bool:
    """Whether an entry of `value`, along its own axis of `length` entries, reads one past its index:
    the operation first doing so names a refusal (`_Walk.first`)."""
    assert value.axis is not None
    return bool(np.any(value.latest > np.arange(length)))


def _later(value: Reads, length: int) -> bool:
    """Whether an entry of `value`, an output holding its positions on axis 1, reads a position later than
    its own: along that axis, entry i reads one past i; read along any other axis, or along none, every
    entry reads the latest of all, which is past position 0 where it is any later one."""
    if value.axis != 1 or len(value.latest) != length:
        return value.last > 0
    return _ahead(value, length)


def _small_held(value) -> bool:
    """Whether a program's constant is an integer, boolean or float array of at most `CONCRETE` entries,
    which the walk holds as its value."""
    return (isinstance(value, np.ndarray | jax.Array) and _held_dtype(value.dtype)
            and math.prod(value.shape) <= CONCRETE)


type _Rule = Callable[[_Walk, core.JaxprEqn, list[Held], list[np.ndarray | None]], list[Held]]

_RULES: dict[core.Primitive, _Rule] = {}
"""The walk's rule for each operation it knows, by JAX's primitive object: what each of its results' entries
reads of its operands'. The reductions, the cumulative ones and the scatters take their classes' rules
(`dew.nn.programs.reduction`, `cumulative`, `scatter`); an operation that maps over its operands' positions
(`positionwise`, a key's among them) keeps their positions' axis; any other gives every entry the latest of
what its operands read."""


def _rule(*operations: core.Primitive) -> Callable[[_Rule], _Rule]:
    """Registers a walk's rule for `operations`."""
    def registered(rule: _Rule) -> _Rule:
        _RULES.update(dict.fromkeys(operations, rule))
        return rule
    return registered



@functools.cache
def _stop_gradient() -> core.Primitive:
    """The primitive `jax.lax.stop_gradient` binds, which computes its operand: as a trace of it shows."""
    (eqn,) = jax.make_jaxpr(jax.lax.stop_gradient)(jnp.zeros(1)).eqns
    return eqn.primitive

class _Walk:
    def __init__(self, length: int):
        self.length = length
        self.first = ""
        """The first operation whose output's entry at a position reads a later position, for the
        refusal's account."""
        self.partial: dict[int, np.ma.MaskedArray] = {}
        """The entries the walk holds of an integer value the tokens reach in part, by the value's
        variable's identity: a concatenation of indices it holds beside ones the tokens decide (a
        gather's row coordinates beside its routed ones)."""
        self.row_max: dict[int, Pair] = {}
        """Each value that is a `Pair`'s maximum along its second axis, by the value's variable's identity,
        kept through the broadcasts and reshapes that lay it beside the pair (a softmax's)."""

    def followed(self, eqn: core.JaxprEqn, given: list[Held], operands: list[np.ndarray | None]) -> None:
        """Keeps what the walk knows of `eqn`'s output past its reads: the entries it holds of a
        concatenation the tokens reach in part, and a pair's row maximum carried through a broadcast, a
        reshape or a conversion."""
        primitive = eqn.primitive
        if (primitive is jax.lax.concatenate_p and any(isinstance(value, Reads) for value in given)
                and any(operand is not None for operand in operands)
                and all(isinstance(var.aval, ShapedArray) and _indexing(var.aval.dtype) for var in eqn.invars)
                and math.prod(_shape(eqn.outvars[0])) <= CONCRETE):
            parts = [np.ma.masked_array(np.zeros(_shape(var), np.int64), mask=True) if operand is None
                     else np.ma.masked_array(np.broadcast_to(operand, _shape(var)), mask=False)
                     for var, operand in zip(eqn.invars, operands, strict=True)]
            self.partial[id(eqn.outvars[0])] = np.ma.concatenate(parts, axis=eqn.params["dimension"])
        elif (primitive in (jax.lax.broadcast_in_dim_p, jax.lax.reshape_p, jax.lax.convert_element_type_p,
                            jax.lax.squeeze_p, _stop_gradient()) and id(eqn.invars[0]) in self.row_max):
            self.row_max[id(eqn.outvars[0])] = self.row_max[id(eqn.invars[0])]
        elif primitive is jax.lax.max_p:
            # A maximum with a floor of minus infinity (a reduction's `initial`) is the maximum.
            for index, var in enumerate(eqn.invars[:2]):
                other = operands[1 - index]
                if (id(var) in self.row_max and other is not None
                        and bool(np.all(np.isneginf(np.asarray(other, np.float64))))):
                    self.row_max[id(eqn.outvars[0])] = self.row_max[id(var)]

    def run_reads(self, jaxpr: core.Jaxpr, inputs: Sequence[Held],
                  values: Sequence[np.ndarray | None]) -> list[Reads | None]:
        """`run`'s outputs' reads along one axis at most (`_reads`), as a loop or a branch joins them."""
        return [_reads(value) for value in self.run(jaxpr, inputs, values)[0]]

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
            self.followed(eqn, given, operands)
            if not self.first and any(isinstance(value, Reads) and value.axis is not None
                                      and len(value.latest) == self.length and _ahead(value, self.length)
                                      for value in map(_reads, outputs)):
                self.first = eqn.primitive.name
        return ([None if isinstance(var, core.Literal) else held.get(var) for var in jaxpr.outvars],
                [np.asarray(var.val) if isinstance(var, core.Literal) else known.get(var)
                 for var in jaxpr.outvars])

    def equation(self, eqn: core.JaxprEqn, given: list[Held],
                 operands: list[np.ndarray | None]) -> tuple[list[Held], list[np.ndarray | None]]:
        primitive = eqn.primitive
        if primitive.inlined_call:
            return _call(self, eqn, given, operands)
        if not any(isinstance(value, Reads | Pair) for value in given):
            return [None] * len(eqn.outvars), self.evaluated(eqn, operands)
        if any(isinstance(value, Pair) for value in given):
            paired = _paired(self, eqn, given, operands)
            if paired is not None:
                return paired, [None] * len(eqn.outvars)
            given = [_reads(value) for value in given]
        # An operation is known by its object: one of another library's reads its operands whole, whatever
        # its name.
        rule = _RULES.get(primitive) or (_reduce if reduction(primitive) else _cumulative if cumulative(
            primitive) else _scatter if scatter(primitive) else None)
        if rule is not None:
            return rule(self, eqn, given, operands), [None] * len(eqn.outvars)
        shape = _shape(eqn.outvars[0])
        if positionwise(primitive) and all(_shape(var) == shape for var in eqn.outvars) and all(
                _fits(_shape(var), shape) for var, value in zip(eqn.invars, given, strict=True)
                if isinstance(value, Reads)):
            return [_combined(given, shape)] * len(eqn.outvars), [None] * len(eqn.outvars)
        return [_flat(given)] * len(eqn.outvars), [None] * len(eqn.outvars)

    def evaluated(self, eqn: core.JaxprEqn, operands: list[np.ndarray | None]) -> list[np.ndarray | None]:
        """The outputs' values of an operation the tokens do not reach whose operands the walk holds,
        where they are integers, booleans or floats of at most `CONCRETE` entries: the indices a gather, a
        slice or an update reads, the tests that choose them, and a mask's values."""
        if (any(operand is None for operand in operands) or bodies(eqn)
                or not all(isinstance(var.aval, ShapedArray) and _held_dtype(var.aval.dtype)
                           and math.prod(_shape(var)) <= CONCRETE for var in eqn.outvars)):
            return [None] * len(eqn.outvars)
        bound = eqn.primitive.bind(*operands, **eqn.params)
        return [np.asarray(output) for output in (bound if eqn.primitive.multiple_results else [bound])]


def _shape(var) -> tuple[int, ...]:
    return tuple(var.aval.shape) if isinstance(var.aval, ShapedArray) else ()


def _held_dtype(dtype: np.dtype | jax.typing.DTypeLike) -> bool:
    """Whether the walk holds a value of `dtype` it computes without the tokens: an integer's, a
    boolean's or a real float's."""
    return _indexing(dtype) or jnp.issubdtype(dtype, jnp.floating)


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
    return _everywhere(max((value.last for value in given if isinstance(value, Reads)), default=NONE))


def _combined(given: Sequence[Held], shape: tuple[int, ...]) -> Reads:
    """An entrywise operation's reads over operands of `shape` (or scalars): the positions' axis the
    operands share keeps each index's join; any other operand's reads join every entry."""
    axes = [value for value in given if isinstance(value, Reads) and value.axis is not None
            and len(value.latest) == shape[value.axis]]
    base = axes[0] if axes else None
    extra = NONE
    for value in given:
        if isinstance(value, Reads) and (base is None or value is not base):
            if base is not None and value.axis == base.axis and len(value.latest) == len(base.latest):
                base = Reads(base.axis, np.maximum(base.latest, value.latest))
            else:
                extra = max(extra, value.last)
    if base is None:
        return _everywhere(extra)
    return Reads(base.axis, np.maximum(base.latest, extra))


def _call(walk: _Walk, eqn: core.JaxprEqn, given: list[Held],
          operands: list[np.ndarray | None]) -> tuple[list[Held], list[np.ndarray | None]]:
    """A call of a program of its own: a jit of a program Dew's code bound under a rule (`attested`) by
    that rule, any other walked inside."""
    program = called(eqn)
    rule = _ATTESTED.get(program) if eqn.primitive is primitives.jit_p and program is not None else None
    if rule is not None or program is None:
        given = [_reads(value) for value in given]
    if rule in (CAUSAL_ATTENTION, STATED_ATTENTION) and any(isinstance(value, Reads) for value in given):
        return _attended(eqn, given, causal=rule == CAUSAL_ATTENTION), [None] * len(eqn.outvars)
    if rule == TOKENWISE and any(isinstance(value, Reads) for value in given):
        out = _shape(eqn.outvars[0])
        aligned = [value for value in given if isinstance(value, Reads)]
        if all(value.axis == 0 and len(value.latest) == out[0] for value in aligned):
            latest = np.maximum.reduce([value.latest for value in aligned])
            return [Reads(0, latest)] * len(eqn.outvars), [None] * len(eqn.outvars)
        return [_flat(given)] * len(eqn.outvars), [None] * len(eqn.outvars)
    if program is None:
        return [_flat(given) if any(given) else None] * len(eqn.outvars), [None] * len(eqn.outvars)
    consts = [np.asarray(const) if _small_held(const) else None for const in program.consts]
    return walk.run(program, given, operands, consts)


def _attended(eqn: core.JaxprEqn, given: list[Held], *, causal: bool) -> list[Held]:
    """Dew's attention's outputs, its queries `[B, S, H, D]` its first operand: an output of the
    queries' positions (`[B, S, ...]`) reads at each position its own query's positions and its keys'
    and values' up to its own (`causal`), or its own (a visibility its caller states); any other output
    (a cache it writes) every position any operand reads."""
    length = _shape(eqn.invars[0])[1]
    own, keyed, extra = _none(length), _none(length), NONE
    for index, value in enumerate(given):
        if not isinstance(value, Reads):
            continue
        if value.axis == 1 and len(value.latest) == length:
            if index == 0 or not causal:
                own = np.maximum(own, value.latest)
            else:
                keyed = np.maximum(keyed, value.latest)
        else:
            extra = max(extra, value.last)
    rows = np.maximum(np.maximum(own, np.maximum.accumulate(keyed)), extra)
    flat = _flat(given)
    return [Reads(1, rows) if len(_shape(var)) > 1 and _shape(var)[1] == length else flat
            for var in eqn.outvars]


def _largest(var) -> float:
    """The largest finite value of `var`'s dtype, infinite for a dtype of no such bound (an integer's)."""
    aval = var.aval
    assert isinstance(aval, ShapedArray)
    return float(jnp.finfo(aval.dtype).max) if jnp.issubdtype(aval.dtype, jnp.floating) else math.inf


def _paired(walk: _Walk, eqn: core.JaxprEqn, given: list[Held],
            operands: list[np.ndarray | None]) -> list[Held] | None:
    """The outputs' reads of an operation over a `Pair`, where the walk keeps its two axes through it: a
    move of its axes, a reduction, a contraction, or an operation entry by entry; None elsewhere, where
    the pair is read `collapsed`."""
    primitive = eqn.primitive
    if primitive is jax.lax.dot_general_p:
        return _pair_dot(eqn, given)
    if primitive in (jax.lax.reduce_max_p, jax.lax.reduce_sum_p):
        return _pair_reduced(walk, eqn, given)
    if primitive in (jax.lax.broadcast_in_dim_p, jax.lax.reshape_p, jax.lax.transpose_p, jax.lax.squeeze_p):
        moved = _pair_moved(eqn, given[0])
        if moved is None:
            return None
        kept: list[Held] = [moved]
        return kept
    shape = _shape(eqn.outvars[0])
    if positionwise(primitive) and all(_shape(var) == shape for var in eqn.outvars) and all(
            _fits(_shape(var), shape) for var, value in zip(eqn.invars, given, strict=True)
            if isinstance(value, Reads | Pair)):
        entrywise = _pair_entrywise(walk, eqn, given, operands, shape)
        if entrywise is None:
            return None
        combined: list[Held] = [entrywise] * len(eqn.outvars)
        return combined
    return None


def _pair_moved(eqn: core.JaxprEqn, value: Held) -> Pair | None:
    """A pair through a broadcast, a reshape, a transpose or a squeeze that keeps each of its axes whole
    (as many entries, in their order), its entries as they are."""
    assert isinstance(value, Pair)
    primitive, before, after = eqn.primitive, _shape(eqn.invars[0]), _shape(eqn.outvars[0])
    if primitive is jax.lax.transpose_p:
        order = tuple(eqn.params["permutation"])
        axes = tuple(order.index(axis) for axis in value.axes)
    elif primitive is jax.lax.broadcast_in_dim_p:
        dims = eqn.params["broadcast_dimensions"]
        axes = tuple(dims[axis] for axis in value.axes)
        if any(after[axis] != before[old] for axis, old in zip(axes, value.axes, strict=True)):
            return None
    elif primitive is jax.lax.squeeze_p:
        dims = eqn.params["dimensions"]
        if any(axis in dims for axis in value.axes):
            return None
        axes = tuple(axis - sum(1 for dim in dims if dim < axis) for axis in value.axes)
    else:
        if eqn.params.get("dimensions") is not None:
            return None
        axes = tuple(_kept_axis(axis, before, after) for axis in value.axes)
        if None in axes:
            return None
    return dataclasses.replace(value, axes=axes)


def _kept_axis(axis: int, before: tuple[int, ...], after: tuple[int, ...]) -> int | None:
    """The axis of a row-major reshape from `before` to `after` holding `axis`'s entries in their order,
    whole, or None."""
    count, outer = before[axis], math.prod(before[:axis])
    for index, width in enumerate(after):
        if width == count and math.prod(after[:index]) == outer:
            return index
    return None


def _per_entry(value: np.ndarray, shape: tuple[int, ...], axes: tuple[int, int]) -> tuple[np.ndarray, ...]:
    """A value the walk holds, broadcast to `shape`, as each entry along `axes`' lowest and highest over
    the others."""
    full = np.moveaxis(np.broadcast_to(np.asarray(value, np.float64), shape), axes, (0, 1))
    flat = full.reshape(shape[axes[0]], shape[axes[1]], -1)
    return flat.min(axis=2), flat.max(axis=2)


def _pair_entrywise(walk: _Walk, eqn: core.JaxprEqn, given: list[Held], operands: list[np.ndarray | None],
                    shape: tuple[int, ...]) -> Pair | None:
    """An operation entry by entry over pairs alike and values the walk holds or reads along either axis:
    each entry reads what its operands' entries read, past a zero product's or quotient's, and its value
    bounds and zeros follow what the operation does (`_bounded`)."""
    pairs = [value for value in given if isinstance(value, Pair)]
    axes = pairs[0].axes
    size = (shape[axes[0]], shape[axes[1]])
    if any(value.axes != axes or value.latest.shape != size for value in pairs):
        return None
    entries = [_entries(value, operand, shape, axes) for value, operand in zip(given, operands, strict=True)]
    if any(entry is None for entry in entries):
        return None
    reads, lows, highs, zeros = (list(group) for group in zip(*entries, strict=True))
    latest = np.maximum.reduce(reads)
    low, high, zero = _bounded(walk, eqn, given, operands, shape, axes, lows, highs, zeros)
    if eqn.primitive is jax.lax.select_n_p:
        chosen = _chosen(operands[0], shape, axes, len(given) - 1)
        if chosen is not None:
            latest = np.maximum(reads[0], np.maximum.reduce(
                [np.where(taken, reads[case], NONE) for case, taken in enumerate(chosen, 1)]))
    if eqn.primitive in (jax.lax.mul_p, jax.lax.div_p, jax.lax.exp_p, jax.lax.convert_element_type_p,
                         jax.lax.select_n_p):
        # Zero whatever its operands' other entries read.
        latest = np.where(zero, NONE, latest)
    return Pair(axes, latest, low, high, zero)


def _entries(value: Held, operand: np.ndarray | None, shape: tuple[int, ...],
             axes: tuple[int, int]) -> tuple[np.ndarray, ...] | None:
    """An operand's entries along `axes` of an operation entry by entry over a pair: what each reads, its
    value's bounds and whether it is zero; None where it reads along another axis."""
    size = (shape[axes[0]], shape[axes[1]])
    if isinstance(value, Pair):
        return value.latest, value.low, value.high, value.zero
    if isinstance(value, Reads):
        if value.axis is None:
            reads = np.full(size, value.last, np.int64)
        elif value.axis == axes[0] and len(value.latest) == size[0]:
            reads = np.broadcast_to(value.latest[:, None], size)
        elif value.axis == axes[1] and len(value.latest) == size[1]:
            reads = np.broadcast_to(value.latest[None, :], size)
        else:
            return None
        return reads, np.full(size, -math.inf), np.full(size, math.inf), np.zeros(size, bool)
    reads = np.full(size, NONE, np.int64)
    if operand is None:
        return reads, np.full(size, -math.inf), np.full(size, math.inf), np.zeros(size, bool)
    low, high = _per_entry(operand, shape, axes)
    return reads, low, high, (low == 0) & (high == 0)


def _bounded(walk: _Walk, eqn: core.JaxprEqn, given: list[Held], operands: list[np.ndarray | None],
             shape: tuple[int, ...], axes: tuple[int, int], lows: list[np.ndarray], highs: list[np.ndarray],
             zeros: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """An entrywise operation's value bounds and zeros from its operands': a sum's and a difference's
    (a score less its row's maximum below the row's largest lower bound), a product's, a quotient's by a
    positive divisor, a negation's, a maximum's, an exponential's (zero where it underflows its dtype),
    a conversion's and a choice's; any other's unknown."""
    primitive = eqn.primitive
    size = lows[0].shape
    low, high, zero = np.full(size, -math.inf), np.full(size, math.inf), np.zeros(size, bool)
    with np.errstate(invalid="ignore", over="ignore"):
        if primitive is jax.lax.add_p:
            low, high, zero = lows[0] + lows[1], highs[0] + highs[1], zeros[0] & zeros[1]
        elif primitive is jax.lax.sub_p:
            low, high, zero = lows[0] - highs[1], highs[0] - lows[1], zeros[0] & zeros[1]
            row = walk.row_max.get(id(eqn.invars[1]))
            subtracted = given[1]
            if (row is not None and given[0] is row and isinstance(subtracted, Reads)
                    and subtracted.axis == axes[0]):
                # A score less its row's maximum, which is at least the row's largest lower bound.
                high = np.minimum(high, row.high - row.low.max(axis=1, keepdims=True))
                low = np.maximum(low, row.low - row.high.max(axis=1, keepdims=True))
        elif primitive in (jax.lax.mul_p, jax.lax.div_p):
            divided = primitive is jax.lax.div_p
            zero = zeros[0] if divided else zeros[0] | zeros[1]
            if not divided or bool(np.all(lows[1] > 0)):
                pairs = [(lows[0], lows[1]), (lows[0], highs[1]), (highs[0], lows[1]), (highs[0], highs[1])]
                corners = [first / second if divided else first * second for first, second in pairs]
                low, high = np.nanmin(corners, axis=0), np.nanmax(corners, axis=0)
        elif primitive is jax.lax.neg_p:
            low, high, zero = -highs[0], -lows[0], zeros[0]
        elif primitive is jax.lax.max_p:
            low, high, zero = np.maximum.reduce(lows), np.maximum.reduce(highs), np.logical_and.reduce(zeros)
        elif primitive is jax.lax.exp_p:
            aval = eqn.outvars[0].aval
            assert isinstance(aval, ShapedArray)
            # Below this an exponential rounds to zero in its dtype.
            floor = math.log(float(jnp.finfo(aval.dtype).smallest_subnormal) / 2)
            low, high, zero = np.exp(lows[0]), np.exp(highs[0]), highs[0] < floor
        elif primitive is jax.lax.convert_element_type_p:
            low, high, zero = lows[0], highs[0], zeros[0]
            if jnp.issubdtype(eqn.params["new_dtype"], jnp.floating):
                largest = float(jnp.finfo(eqn.params["new_dtype"]).max)
                high, low = np.where(high > largest, math.inf, high), np.where(low < -largest, -math.inf, low)
        elif primitive is jax.lax.select_n_p:
            chosen = _chosen(operands[0], shape, axes, len(given) - 1)
            if chosen is not None:
                low = np.minimum.reduce([np.where(taken, lows[case], math.inf)
                                         for case, taken in enumerate(chosen, 1)])
                high = np.maximum.reduce([np.where(taken, highs[case], -math.inf)
                                          for case, taken in enumerate(chosen, 1)])
                zero = np.logical_and.reduce([~taken | zeros[case] for case, taken in enumerate(chosen, 1)])
    return np.nan_to_num(low, nan=-math.inf), np.nan_to_num(high, nan=math.inf), zero


def _chosen(predicate: np.ndarray | None, shape: tuple[int, ...], axes: tuple[int, int],
            cases: int) -> list[np.ndarray] | None:
    """Along `axes`, where a choice of `cases` by a `predicate` the walk holds may take each case."""
    if predicate is None:
        return None
    full = np.moveaxis(np.broadcast_to(np.asarray(predicate).astype(np.int64), shape), axes, (0, 1))
    flat = full.reshape(shape[axes[0]], shape[axes[1]], -1)
    return [np.any(flat == case, axis=2) for case in range(cases)]


def _pair_reduced(walk: _Walk, eqn: core.JaxprEqn, given: list[Held]) -> list[Held] | None:
    """A maximum or a sum over a pair's axes: along its second axis, each row reads its entries that may
    take part (each but a zero of a sum; each a maximum may be, its upper bound past the row's largest
    lower one), kept along its first; along neither, a pair of its remaining axes."""
    value, axes = given[0], tuple(eqn.params["axes"])
    assert isinstance(value, Pair)
    first, second = value.axes
    maximum = eqn.primitive is jax.lax.reduce_max_p
    part = value.high >= value.low.max(axis=1, keepdims=True) if maximum else ~value.zero
    if second in axes and first not in axes:
        rows = np.where(part, value.latest, NONE).max(axis=1, initial=NONE)
        held = Reads(first - sum(1 for dim in axes if dim < first), rows)
        if maximum:
            walk.row_max[id(eqn.outvars[0])] = value
        return [held]
    if first in axes and second not in axes:
        if maximum:
            part = value.high >= value.low.max(axis=0, keepdims=True)
        cols = np.where(part, value.latest, NONE).max(axis=0, initial=NONE)
        return [Reads(second - sum(1 for dim in axes if dim < second), cols)]
    if first in axes:
        return [_everywhere(int(np.where(part, value.latest, NONE).max(initial=NONE)))]
    moved = (first - sum(1 for dim in axes if dim < first), second - sum(1 for dim in axes if dim < second))
    if maximum:
        return [dataclasses.replace(value, axes=moved)]
    return [Pair(moved, value.latest, np.full(value.latest.shape, -math.inf),
                 np.full(value.latest.shape, math.inf), value.zero)]


def _pair_dot(eqn: core.JaxprEqn, given: list[Held]) -> list[Held] | None:
    """A contraction of a pair along one of its axes with a value whose positions lie on the contracted
    axis (attention's weights with its values): each entry along the pair's other axis reads its row's
    entries but its zeros, and the values' positions those entries meet."""
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = eqn.params["dimension_numbers"]
    pairs = [index for index, value in enumerate(given[:2]) if isinstance(value, Pair)]
    if len(pairs) != 1:
        return None
    side = pairs[0]
    pair, other = given[side], given[1 - side]
    assert isinstance(pair, Pair)
    contract, other_contract = ((lhs_contract, rhs_contract) if side == 0 else (rhs_contract, lhs_contract))
    batch = (lhs_batch, rhs_batch)[side]
    first, second = pair.axes
    latest, zero = pair.latest, pair.zero
    if first in contract and second not in contract:
        first, second, latest, zero = second, first, latest.T, zero.T
    if second not in contract or first in contract or first in batch:
        return None
    matched = other_contract[list(contract).index(second)]
    extra = NONE
    if isinstance(other, Reads) and other.axis == matched and len(other.latest) == latest.shape[1]:
        cols = other.latest
    elif other is None or (isinstance(other, Reads) and other.axis is None):
        cols = _none(latest.shape[1])
        extra = NONE if other is None else other.last
    else:
        return None
    rows = np.where(zero, NONE, np.maximum(latest, cols[None, :])).max(axis=1, initial=NONE)
    ranks = len(_shape(eqn.invars[0])), len(_shape(eqn.invars[1]))
    free = [dim for dim in range(ranks[side]) if dim not in contract and dim not in batch]
    start = len(lhs_batch) + (0 if side == 0 else ranks[0] - len(lhs_contract) - len(lhs_batch))
    return [Reads(start + free.index(first), np.maximum(rows, extra))]


@_rule(jax.lax.broadcast_in_dim_p)
def _broadcast_in_dim(walk: _Walk, eqn, given, operands) -> list[Held]:
    value, dims, out = given[0], eqn.params["broadcast_dimensions"], _shape(eqn.outvars[0])
    assert isinstance(value, Reads)
    if value.axis is None:
        return [value]
    axis = dims[value.axis]
    latest = value.latest if len(value.latest) == out[axis] else np.full(out[axis], value.last, np.int64)
    return [Reads(axis, latest)]


@_rule(jax.lax.reshape_p)
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
        value = Reads(tuple(order).index(value.axis), value.latest)
    return [_moved(value, before, after)]


def _moved(value: Reads, before: tuple[int, ...], after: tuple[int, ...]) -> Reads:
    """`value`'s reads once its entries, in row-major order, take the shape `after`. An axis that keeps
    the old axis's entries in their order (as many of them, as many entries before it) holds their reads
    as they are, at any size. Otherwise along each new axis of at most `CONCRETE` index entries, each
    index reads the latest of the rows its entries come from, and the axis whose latest are the earliest
    holds them (exactly where its index alone decides the row: the old axis merged with others, or
    split)."""
    axis = value.axis
    assert axis is not None
    count, inner, outer = len(value.latest), math.prod(before[axis + 1:]), math.prod(before[:axis])
    if count == 0:
        return value.flat()
    for index, width in enumerate(after):
        if width == count and math.prod(after[:index]) == outer:
            return Reads(index, value.latest)
    span = _RangeMax(value.latest)
    best: Reads | None = None
    for index, width in enumerate(after):
        lead, tail = math.prod(after[:index]), math.prod(after[index + 1:])
        if lead * width > CONCRETE:
            continue
        starts = (np.arange(lead)[:, None] * width + np.arange(width)[None, :]) * tail
        rows = span(starts // inner, (starts + tail - 1) // inner).max(axis=0)
        if best is None or _tighter(rows, best.latest):
            best = Reads(index, rows)
    return value.flat() if best is None else best


def _tighter(rows: np.ndarray, held: np.ndarray) -> bool:
    """Whether `rows` give their indices earlier positions, on average, than `held` (the finer axis where
    they tie): the axis a value's reads are kept along."""
    return (rows.mean(), -len(rows)) < (held.mean(), -len(held))


class _RangeMax:
    """The latest of `latest`'s rows over each range of row indices, taken modulo their count (a sparse
    table over them twice over), every row's where a range covers them all."""

    def __init__(self, latest: np.ndarray):
        self.count = len(latest)
        self.last = int(latest.max(initial=NONE))
        table = [np.concatenate([latest, latest])]
        while (1 << len(table)) <= 2 * self.count:
            previous, step = table[-1], 1 << (len(table) - 1)
            table.append(np.maximum(previous[:-step], previous[step:]))
        self.table = table

    def __call__(self, first: np.ndarray, last: np.ndarray) -> np.ndarray:
        size = last - first + 1
        whole = size >= self.count
        start = first % self.count
        clipped = np.where(whole, 1, size)
        level = np.floor(np.log2(clipped)).astype(int)
        out = np.full(first.shape, NONE, np.int64)
        for height in np.unique(level):
            at = level == height
            row = self.table[height]
            out[at] = np.maximum(row[start[at]], row[start[at] + clipped[at] - (1 << height)])
        return np.where(whole, self.last, out)


@_rule(jax.lax.squeeze_p)
def _squeeze(walk: _Walk, eqn, given, operands) -> list[Held]:
    value = given[0]
    assert isinstance(value, Reads)
    if value.axis is None:
        return [value]
    dims = eqn.params["dimensions"]
    if value.axis in dims:
        return [value.flat()]
    return [Reads(value.axis - sum(1 for dim in dims if dim < value.axis), value.latest)]


@_rule(jax.lax.transpose_p)
def _transpose(walk: _Walk, eqn, given, operands) -> list[Held]:
    value, order = given[0], tuple(eqn.params["permutation"])
    assert isinstance(value, Reads)
    return [value if value.axis is None else Reads(order.index(value.axis), value.latest)]


def _reduce(walk: _Walk, eqn, given, operands) -> list[Held]:
    value, axes = _combined(given, _shape(eqn.invars[0])), tuple(eqn.params["axes"])
    if value.axis is None or value.axis in axes:
        return [value.flat()] * len(eqn.outvars)
    return [Reads(value.axis - sum(1 for dim in axes if dim < value.axis), value.latest)] * len(eqn.outvars)


def _cumulative(walk: _Walk, eqn, given, operands) -> list[Held]:
    value = given[0]
    assert isinstance(value, Reads)
    if value.axis != eqn.params["axis"]:
        return [value]
    if eqn.params.get("reverse", False):
        return [Reads(value.axis, np.maximum.accumulate(value.latest[::-1])[::-1])]
    return [Reads(value.axis, np.maximum.accumulate(value.latest))]


@_rule(jax.lax.dot_general_p)
def _dot_general(walk: _Walk, eqn, given, operands) -> list[Held]:
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = eqn.params["dimension_numbers"]
    ranks = len(_shape(eqn.invars[0])), len(_shape(eqn.invars[1]))
    free = [[dim for dim in range(rank) if dim not in contract and dim not in batch]
            for rank, contract, batch in zip(ranks, (lhs_contract, rhs_contract), (lhs_batch, rhs_batch),
                                              strict=True)]
    placed, extra = [], NONE
    for side, value in enumerate(given[:2]):
        if not isinstance(value, Reads):
            continue
        contract, batch = (lhs_contract, rhs_contract)[side], (lhs_batch, rhs_batch)[side]
        if value.axis is None or value.axis in contract:
            extra = max(extra, value.last)
        elif value.axis in batch:
            placed.append(Reads(list(batch).index(value.axis), value.latest))
        else:
            start = len(lhs_batch) + (0 if side == 0 else len(free[0]))
            placed.append(Reads(start + free[side].index(value.axis), value.latest))
    if not placed:
        return [_everywhere(extra)]
    if (len(placed) == 2 and placed[0].axis != placed[1].axis
            and len(placed[0].latest) * len(placed[1].latest) <= CONCRETE):
        # Each side's positions on an axis of its own (a query's and a key's): a score.
        rows, cols = placed
        latest = np.maximum(np.maximum(rows.latest[:, None], cols.latest[None, :]), extra)
        bound = _largest(eqn.outvars[0]) * _BOUND
        return [Pair((rows.axis, cols.axis), latest, np.full(latest.shape, -bound),
                     np.full(latest.shape, bound), np.zeros(latest.shape, bool))]
    base = placed[0]
    for other in placed[1:]:
        if other.axis == base.axis:
            base = Reads(base.axis, np.maximum(base.latest, other.latest))
        else:
            extra = max(extra, other.last)
    return [Reads(base.axis, np.maximum(base.latest, extra))]


@_rule(jax.lax.conv_general_dilated_p)
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
        return [Reads(out_spec[0], lhs.latest)]
    if lhs.axis == lhs_spec[1]:
        return [lhs.flat()]
    spatial = list(lhs_spec[2:]).index(lhs.axis)
    low, _ = eqn.params["padding"][spatial]
    width = _shape(eqn.invars[1])[rhs_spec[2 + spatial]]
    dilation, stride = eqn.params["rhs_dilation"][spatial], eqn.params["window_strides"][spatial]
    if eqn.params["lhs_dilation"][spatial] != 1:
        return [lhs.flat()]
    size = _shape(eqn.outvars[0])[out_spec[2 + spatial]]
    reads = np.arange(size)[:, None] * stride - low + np.arange(width)[None, :] * dilation
    inside = (reads >= 0) & (reads < len(lhs.latest))
    rows = _read_at(lhs.latest, np.where(inside, reads, -1)).max(axis=1, initial=NONE)
    return [Reads(out_spec[2 + spatial], rows)]


@_rule(jax.lax.slice_p)
def _slice(walk: _Walk, eqn, given, operands) -> list[Held]:
    value = given[0]
    assert isinstance(value, Reads)
    if value.axis is None:
        return [value]
    start, limit = eqn.params["start_indices"][value.axis], eqn.params["limit_indices"][value.axis]
    strides = eqn.params.get("strides")
    step = 1 if strides is None else strides[value.axis]
    return [Reads(value.axis, value.latest[start:limit:step])]


@_rule(jax.lax.rev_p)
def _rev(walk: _Walk, eqn, given, operands) -> list[Held]:
    value = given[0]
    assert isinstance(value, Reads)
    if value.axis is None or value.axis not in eqn.params["dimensions"]:
        return [value]
    return [Reads(value.axis, value.latest[::-1])]


@_rule(jax.lax.pad_p)
def _pad(walk: _Walk, eqn, given, operands) -> list[Held]:
    value, padding = given[0], given[1]
    if isinstance(padding, Reads) or not isinstance(value, Reads) or value.axis is None:
        return [_flat(given)]
    low, high, interior = eqn.params["padding_config"][value.axis]
    count = len(value.latest)
    spread = _none(max(count + (count - 1) * interior, 0))
    spread[::interior + 1] = value.latest
    rows = np.concatenate([_none(max(low, 0)), spread[max(-low, 0):]])
    rows = rows[:len(rows) + high] if high < 0 else np.concatenate([rows, _none(high)])
    return [Reads(value.axis, rows)]


@_rule(jax.lax.concatenate_p)
def _concatenate(walk: _Walk, eqn, given, operands) -> list[Held]:
    dimension = eqn.params["dimension"]
    axes = {value.axis for value in given if isinstance(value, Reads)}
    if axes != {dimension}:
        flat = _flat(given)
        along = [value for value in given if isinstance(value, Reads) and value.axis is not None]
        if along and all(value.axis == along[0].axis for value in along) and dimension != along[0].axis:
            return [_combined(given, _shape(eqn.outvars[0]))]
        return [flat]
    rows = [value.latest if isinstance(value, Reads) else _none(_shape(var)[dimension])
            for var, value in zip(eqn.invars, given, strict=True)]
    return [Reads(dimension, np.concatenate(rows))]


@_rule(jax.lax.dynamic_slice_p)
def _dynamic_slice(walk: _Walk, eqn, given, operands) -> list[Held]:
    value = given[0]
    if not isinstance(value, Reads) or any(isinstance(start, Reads) for start in given[1:]):
        return [_flat(given)]
    if value.axis is None:
        return [value]
    start = operands[1 + value.axis]
    size = eqn.params["slice_sizes"][value.axis]
    if start is None:
        return [Reads(value.axis, np.full(size, value.last, np.int64))]
    first = int(np.clip(int(start), 0, len(value.latest) - size))
    return [Reads(value.axis, value.latest[first:first + size])]


@_rule(jax.lax.dynamic_update_slice_p)
def _dynamic_update_slice(walk: _Walk, eqn, given, operands) -> list[Held]:
    base, update = given[0], given[1]
    if any(isinstance(start, Reads) for start in given[2:]):
        return [_flat(given)]
    axis = next((value.axis for value in (base, update) if isinstance(value, Reads)), None)
    if axis is None or any(isinstance(value, Reads) and value.axis != axis for value in (base, update)):
        return [_flat(given)]
    length = _shape(eqn.outvars[0])[axis]
    rows = base.latest.copy() if isinstance(base, Reads) else _none(length)
    width = _shape(eqn.invars[1])[axis]
    start = operands[2 + axis]
    if start is None:
        return [Reads(axis, np.maximum(rows, _flat([update]).last))]
    first = int(np.clip(int(start), 0, length - width))
    updated = update.latest if isinstance(update, Reads) else _none(width)
    others = [size for dim, size in enumerate(_shape(eqn.outvars[0])) if dim != axis]
    covered = [size for dim, size in enumerate(_shape(eqn.invars[1])) if dim != axis]
    if covered == others:
        # The update overwrites each of its rows whole: they read what it reads alone.
        rows[first:first + width] = updated
    else:
        rows[first:first + width] = np.maximum(rows[first:first + width], updated)
    return [Reads(axis, rows)]


@_rule(jax.lax.gather_p)
def _gather(walk: _Walk, eqn, given, operands) -> list[Held]:
    """A gather: by indices the walk holds, each output entry reads the operand rows it takes, moved
    to the output's axis that decides them; by indices the tokens reach (an embedding's lookup), each
    output entry reads its index's positions; otherwise the latest of all."""
    operand, indices = given[0], given[1]
    numbers = eqn.params["dimension_numbers"]
    out = _shape(eqn.outvars[0])
    batch_out = [dim for dim in range(len(out)) if dim not in tuple(numbers.offset_dims)]
    if isinstance(indices, Reads):
        routed = _routed(walk, eqn, operand, indices, batch_out)
        if routed is not None:
            return [routed]
        if isinstance(operand, Reads) or indices.axis is None:
            return [_flat(given)]
        vector = len(_shape(eqn.invars[1])) - 1
        if indices.axis == vector:
            return [indices.flat()]
        return [Reads(batch_out[indices.axis], indices.latest)]
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
    return [_decided(taken, operand.latest)]


def _routed(walk: _Walk, eqn: core.JaxprEqn, operand: Held, indices: Reads,
            batch_out: list[int]) -> Reads | None:
    """A gather from an operand along its positions, by indices the tokens reach whose coordinate on that
    axis the walk holds (`_Walk.partial`: each token's row beside its routed expert, a top-k's weights
    taken per token): each output entry reads the row it takes and what its index read, kept along the
    output's axis the indices' positions lie on. None where no such coordinate is held."""
    numbers = eqn.params["dimension_numbers"]
    held = walk.partial.get(id(eqn.invars[1]))
    if (held is None or not isinstance(operand, Reads) or operand.axis is None
            or operand.axis not in numbers.start_index_map or numbers.operand_batching_dims
            or eqn.params["slice_sizes"][operand.axis] != 1):
        return None
    vector = held.ndim - 1
    coordinate = held[..., list(numbers.start_index_map).index(operand.axis)]
    if np.ma.is_masked(coordinate) or indices.axis == vector:
        return None
    shape = _shape(eqn.invars[0])
    if math.prod(shape) > CONCRETE or math.prod(_shape(eqn.outvars[0])) > CONCRETE:
        return None
    ids = np.broadcast_to(np.arange(shape[operand.axis]).reshape(
        [-1 if dim == operand.axis else 1 for dim in range(len(shape))]), shape)
    # Its other coordinates read as row 0: the row along the operand's positions is the held one's alone.
    taken = np.asarray(jax.lax.gather(ids, held.filled(0), numbers, eqn.params["slice_sizes"],
                                      mode=eqn.params["mode"], fill_value=-1))
    read = _read_at(operand.latest, taken)
    if indices.axis is None:
        return _along(np.maximum(read, indices.last))
    axis = batch_out[indices.axis]
    along = indices.latest.reshape([-1 if dim == axis else 1 for dim in range(read.ndim)])
    read = np.maximum(read, along)
    return Reads(axis, np.moveaxis(read, axis, 0).reshape(read.shape[axis], -1).max(axis=1, initial=NONE))


def _decided(ids: np.ndarray, latest: np.ndarray) -> Reads:
    """Reads of a value whose entries hold the operand rows `ids` (-1 for none) (`_along`)."""
    return _along(_read_at(latest, ids))


def _along(read: np.ndarray) -> Reads:
    """Reads of a value whose entries read up to `read`'s positions: along its axis whose index decides
    them, each index's latest; the axis giving the earliest where none does alone."""
    best: Reads | None = None
    for axis in range(read.ndim):
        rows = np.moveaxis(read, axis, 0).reshape(read.shape[axis], -1).max(axis=1, initial=NONE)
        if best is None or _tighter(rows, best.latest):
            best = Reads(axis, rows)
    return _everywhere(int(read.max(initial=NONE))) if best is None else best


@_rule(jax.lax.sort_p)
def _sort(walk: _Walk, eqn, given, operands) -> list[Held]:
    value = _combined(given, _shape(eqn.invars[0]))
    if value.axis == eqn.params["dimension"]:
        return [value.flat()] * len(eqn.outvars)
    return [value] * len(eqn.outvars)


@_rule(jax.lax.top_k_p)
def _top_k(walk: _Walk, eqn, given, operands) -> list[Held]:
    value = given[0]
    assert isinstance(value, Reads)
    if value.axis == len(_shape(eqn.invars[0])) - 1:
        return [value.flat()] * len(eqn.outvars)
    return [value] * len(eqn.outvars)


@_rule(jax.lax.stack_p)
def _stack(walk: _Walk, eqn, given, operands) -> list[Held]:
    """A stack: its operands, alike in shape, side by side along a new axis `axis`, each operand's positions
    on the axis they held (one past it from `axis` on), joined where they share it."""
    axis = eqn.params["axis"]
    moved = [value if not isinstance(value, Reads) or value.axis is None
             else Reads(value.axis + (value.axis >= axis), value.latest) for value in given]
    return [_combined(moved, _shape(eqn.outvars[0]))]


@_rule(jax.lax.unstack_p)
def _unstack(walk: _Walk, eqn, given, operands) -> list[Held]:
    """An unstack: its operand's slices along `axis`, each its own result: along the positions, each
    slice reads its own position's; otherwise each its operand's positions on the axis they held (one
    before it past `axis`)."""
    value, axis = given[0], eqn.params["axis"]
    assert isinstance(value, Reads)
    if value.axis is None:
        return [value] * len(eqn.outvars)
    if value.axis == axis:
        return [_everywhere(int(latest)) for latest in value.latest]
    return [Reads(value.axis - (value.axis > axis), value.latest)] * len(eqn.outvars)


@_rule(jax.lax.split_p)
def _split(walk: _Walk, eqn, given, operands) -> list[Held]:
    value = given[0]
    assert isinstance(value, Reads)
    if value.axis != eqn.params["axis"]:
        return [value] * len(eqn.outvars)
    pieces, start = [], 0
    for size in eqn.params["sizes"]:
        pieces.append(Reads(value.axis, value.latest[start:start + size]))
        start += size
    return pieces


def _scatter(walk: _Walk, eqn, given, operands) -> list[Held]:
    """A scatter (an update, or one that combines, an add's or a max's): by indices the walk holds, each
    output entry reads its operand entry's positions and those of every update entry landing on it,
    placed as the scatter places them, kept along the operand's axis its operands' positions lie on
    (`_placed`), or the axis that decides them (`_along`); by indices the tokens reach, or too many entries
    to place, the latest of all."""
    operand, indices, updates = given[0], given[1], given[2]
    out = _shape(eqn.outvars[0])
    held = operands[1]
    if (isinstance(indices, Reads) or held is None or len(eqn.invars) != 3
            or math.prod(out) > CONCRETE or math.prod(_shape(eqn.invars[2])) > CONCRETE):
        return [_flat(given)]
    numbers = eqn.params["dimension_numbers"]
    landed = np.asarray(jax.lax.scatter_max(
        _spread(operand, out), held, _spread(updates, _shape(eqn.invars[2])), numbers,
        indices_are_sorted=eqn.params["indices_are_sorted"], unique_indices=eqn.params["unique_indices"],
        mode=eqn.params["mode"]))
    placed = _placed(operand, updates, numbers, len(out))
    if not placed:
        return [_along(landed)]
    kept = [Reads(axis, np.moveaxis(landed, axis, 0).reshape(out[axis], -1).max(axis=1, initial=NONE))
            for axis in placed]
    return [min(kept, key=lambda value: (value.latest.mean(), -len(value.latest)))]


def _placed(operand: Held, updates: Held, numbers: jax.lax.ScatterDimensionNumbers, rank: int) -> set[int]:
    """The output axes a scatter keeps its operands' positions on: the operand's own, and the axis an
    update's window dimension holding its positions lands on (`update_window_dims`, in order, onto the
    operand's dimensions neither inserted nor batched); an update's positions on a scattered dimension
    land where its indices say, on no one axis."""
    axes = set()
    if isinstance(operand, Reads) and operand.axis is not None:
        axes.add(operand.axis)
    if isinstance(updates, Reads) and updates.axis is not None and updates.axis in numbers.update_window_dims:
        windowed = [dim for dim in range(rank) if dim not in numbers.inserted_window_dims
                    and dim not in numbers.operand_batching_dims]
        axes.add(windowed[list(numbers.update_window_dims).index(updates.axis)])
    return axes


def _spread(value: Held, shape: tuple[int, ...]) -> np.ndarray:
    """The latest position each entry of a value of `shape` reads (`NONE` where it reads none)."""
    if not isinstance(value, Reads):
        return np.full(shape, NONE, np.int64)
    if value.axis is None:
        return np.full(shape, value.last, np.int64)
    along = value.latest.reshape([-1 if dim == value.axis else 1 for dim in range(len(shape))])
    return np.broadcast_to(along, shape).copy()


def _scan_groups(eqn: core.JaxprEqn) -> tuple[int, int]:
    """A scan's constant and carried inputs' counts: its parameters', or, where it states its inputs'
    and outputs' groups (`ft_in`, `ft_out`), theirs."""
    params = eqn.params
    if "num_consts" in params:
        return params["num_consts"], params["num_carry"]
    consts, carried, _ = (len(group) for group in params["ft_in"].elts)
    return consts, carried


def _joined(held: Reads | None, found: Reads | None) -> Reads | None:
    """The reads of a loop's carried value over two passes: each entry the join of both."""
    if held is None or found is None:
        return found if held is None else held
    if held.axis == found.axis and len(held.latest) == len(found.latest):
        return Reads(held.axis, np.maximum(held.latest, found.latest))
    return _flat([held, found])


def _settled(walk: _Walk, body: core.Jaxpr, consts: list[Held], carried: list[Reads | None],
             rest: list[Held],
             values: Sequence[np.ndarray | None]) -> tuple[list[Reads | None], list[Reads | None]]:
    """A loop's carried values over every number of passes, none included (each the join of its initial
    reads and every pass's, which only grow over finitely many positions), and `body`'s outputs past its
    carries over those values."""
    while True:
        out = walk.run_reads(body, [*consts, *carried, *rest], values)
        grown = [_joined(held, found) for held, found in zip(carried, out[:len(carried)], strict=True)]
        if all(_same(held, found) for held, found in zip(grown, carried, strict=True)):
            return grown, out[len(carried):]
        carried = grown


def _same(first: Reads | None, second: Reads | None) -> bool:
    if first is None or second is None:
        return first is second
    return first.axis == second.axis and np.array_equal(first.latest, second.latest)


@_rule(jax.lax.scan_p)
def _scan(walk: _Walk, eqn, given, operands) -> list[Held]:
    """A scan: over the positions (a scanned input the tokens reach on its axis 0), step by step in its
    order, each step's output reading its carry, which holds its initial state's reads and every earlier
    step's, and that step's inputs; over any other axis, its body walked until its carried reads settle."""
    params = eqn.params
    consts, carries = _scan_groups(eqn)
    body = scan_body(eqn).jaxpr
    fixed, carried = list(given[:consts]), list(given[consts:consts + carries])
    xs = list(given[consts + carries:])
    length = params["length"]
    values = [None] * len(body.invars)
    if any(isinstance(value, Reads) and value.axis == 0 for value in xs):
        steps: list[list[Reads | None]] = []
        order = range(length - 1, -1, -1) if params["reverse"] else range(length)
        for step in order:
            sliced = [value if not isinstance(value, Reads) else
                      Reads(None, value.latest[step:step + 1]) if value.axis == 0 else
                      Reads(None if value.axis is None else value.axis - 1, value.latest) for value in xs]
            out = walk.run_reads(body, [*fixed, *carried, *sliced], values)
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
                ys.append(Reads(0, np.array([NONE if value is None else value.last for value in per_step],
                                            np.int64)))
            else:
                ys.append(_flat(per_step))
        return [*carried, *ys]
    sliced = [value if not isinstance(value, Reads) or value.axis is None
              else Reads(value.axis - 1, value.latest) for value in xs]
    final, out = _settled(walk, body, fixed, carried, sliced, values)
    ys = [value if not isinstance(value, Reads) or value.axis is None
          else Reads(value.axis + 1, value.latest) for value in out]
    return [*final, *ys]


@_rule(jax.lax.while_p)
def _while(walk: _Walk, eqn, given, operands) -> list[Held]:
    params = eqn.params
    cond_consts, body_consts = params["cond_nconsts"], params["body_nconsts"]
    carried = list(given[cond_consts + body_consts:])
    predicate, body = (program.jaxpr for program in loop_programs(eqn))
    # Its state on exit, after any number of passes, none included.
    out, _ = _settled(walk, body, list(given[cond_consts:cond_consts + body_consts]), carried, [],
                      [None] * len(body.invars))
    cond = walk.run_reads(predicate, [*given[:cond_consts], *out], [None] * len(predicate.invars))
    if any(isinstance(value, Reads) for value in [*cond, *given[:cond_consts]]):
        # How many passes run is the tokens' to decide: every carried entry may read every one.
        return [_flat([*given, *out])] * len(out)
    settled: list[Held] = list(out)
    return settled


@_rule(jax.lax.cond_p)
def _cond(walk: _Walk, eqn, given, operands) -> list[Held]:
    outs = [walk.run_reads(branch.jaxpr, given[1:], operands[1:]) for branch in branches(eqn)]
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
