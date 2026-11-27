"""A template's trace-time derivations (`derive._derived`'s forms at one tile) stored across processes in
Dew's cache, so a process traces a fixed tile's forms from what an earlier one derived
(docs/design/algebra.md §6.5).

A derivation is a pure function of what it reads: its declaration's traced program (its structural digest,
`terms.jaxpr_digest`, and its captured constants' values), its walked axes, its tile's shapes and dtypes,
its numerics' class and limits, the context its forms are priced and traced in (`derive.context`), the rule
sets it saturates with (by name, each as the source registers it, immutable, `rules.sourced`; a set
registered at run time stores nothing), the algebra's code (`dew._code.code_digest`) and egglog's version.
An entry is keyed by all of them (`key`, each part in a canonical form: an unordered collection sorted, a
structure by its fields), written whole (`dew._files.write_atomically`) and read back exactly: each chosen
term through the Term language's constructors (`CONSTRUCTORS`, `get_callable_args`), its arguments as their
literals, a vector as written (egglog writes it typed or not, each a term of its own); the parameters and
literals the derivation appended to its region past its trace by their primitive's name and their values'
types, in their order, the region itself traced afresh. No value is pickled or evaluated. An entry is
checked against the region it extends as it decodes (each term's leaf, literal, parameter and state in
range, each state of a side its form reads, each count and part of its forms the region's) before any of it
is taken.

A derivation a key cannot state (a value held by identity, an outer trace's or a device's) or a value the
codec cannot state is derived afresh and not stored, as is one whose entry cannot be written; an entry that
fails to open, read or decode is derived afresh past. The extractor's oracle (`Passes.checked`) derives and
never reaches here. Each outcome is counted in the kernel report (`report.record_stored`)."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import importlib.metadata
import json
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

import jax
import numpy as np
from egglog import ConvertError, String, Vec, f64, get_callable_args, i64
from jax.extend.core import Primitive

from dew import _code, _files
from dew._builds import dew_cache_dir
from dew.algebra import terms
from dew.algebra.terms import Term
from dew.nn.kernels.templates import report as kernel_report

if TYPE_CHECKING:
    from dew.algebra.derive import _Forms

_log = logging.getLogger(__name__)

FORMAT = 3
"""The entries' layout, in every key: an entry of another is never read."""

CONSTRUCTORS: Mapping[str, Callable] = {
    **{written.constructor: getattr(Term, written.constructor) for written in terms.WRITTEN.values()},
    **{name: getattr(Term, name) for name in ("leaf", "literal", "state")}}
"""The constructors a derived form is written in, by name: each operation of the language (`terms.WRITTEN`)
and the terms an evaluation reads beside them, a tile's leaf, a literal and a carried state."""

_INDEXED = {"leaf": ("leaves", 0), "literal": ("literals", 0)}
"""Each terminal constructor's index argument and what it indexes (a state's, `_Decoded`'s own)."""


class Unstated(Exception):
    """A value the store's key or codec does not state: its derivation is not stored."""


def directory() -> str:
    """Where the entries are, in Dew's cache."""
    return os.path.join(dew_cache_dir(), "algebra", "derived")


def key(closed, sets: Sequence[str], parts: Sequence[object]) -> str | None:
    """The entry's name of a derivation of the traced `closed` saturated with the source's rule sets named
    `sets` (`rules.sourced`), in their order, beside `parts` (everything else it reads, `derive._derived`'s);
    None where its program, a captured constant (an outer trace's value, or one on a device) or a part is no
    value a key states by its contents."""
    digest, opaques = terms.jaxpr_digest(closed, in_process=False)
    if opaques or not all(isinstance(const, (np.ndarray, np.generic, bool, int, float))
                          for const in closed.consts):
        _log.info("a derivation not keyed: a program or constant held by identity")
        return None
    try:
        text = repr((FORMAT, digest, [_encoded_value(np.asarray(const)) for const in closed.consts],
                     list(sets), _canonical(tuple(parts)), importlib.metadata.version("egglog"),
                     _code.code_digest("dew.algebra.derive")))
    except Unstated as unstated:
        _log.info("a derivation not keyed: %s", unstated)
        return None
    if " at 0x" in text:
        _log.info("a derivation not keyed: a part held by identity")
        return None
    return hashlib.blake2b(text.encode(), digest_size=20).hexdigest()


def loaded(name: str, region: terms.Region, assembled: Callable[..., _Forms]) -> _Forms | None:
    """The stored derivation `name` over `region`, traced afresh, its appended parameters and literals
    replayed once the whole entry reads (`assembled(record, decoded=..., ...)` its forms, each decoded term
    checked against the region as it is built, `_Decoded`); None where none is stored, or where the entry
    fails to open, read or decode (counted `invalid`, derived afresh past)."""
    path = os.path.join(directory(), f"{name}.json")
    try:
        with open(path, "rb") as file:
            text = file.read()
    except FileNotFoundError:
        _log.debug("no stored derivation %s: derived afresh", path)
        return None
    except OSError as unreadable:
        _invalid(path, unreadable)
        return None
    try:
        record = json.loads(text)
        if not isinstance(record, dict) or (record.get("format"), record.get("params_from"),
                                            record.get("literals_from")) != (FORMAT, len(region.params),
                                                                             len(region.literals)):
            raise ValueError("an entry of another layout or another trace's region")
        params = [(_primitive(primitive), {name: _value(value) for name, value in given})
                  for primitive, given in record["params"]]
        literals = [_value(value) for value in record["literals"]]
        decoded = _Decoded(region.leaves, len(region.literals) + len(literals),
                           len(region.params) + len(params), len(record["steps"]))
        made = assembled(record, decoded=decoded, value=_value)
    except (KeyError, TypeError, ValueError, IndexError, ConvertError) as failed:
        # ConvertError: an argument of a sort its constructor does not take.
        _invalid(path, failed)
        return None
    # The region extended once the whole entry has read: one that fails leaves it as traced.
    region.params.extend(params)
    region.literals.extend(literals)
    kernel_report.record_stored("hit")
    return made


def _invalid(path: str, failed: Exception) -> None:
    _log.warning("the stored derivation %s fails to read (%s): derived afresh", path, failed)
    kernel_report.record_stored("invalid")


def stored(name: str, region: terms.Region, params_from: int, literals_from: int,
           encoded: Callable[..., Mapping[str, object]]) -> None:
    """Write a derivation's forms derived afresh over `region` as the entry `name` (`encoded(term=...,
    maybe=..., value=...)` its forms), with the parameters and literals the derivation appended to its region
    past `params_from` and `literals_from`: counted `miss` once written; `unstorable`, and not written, where
    the codec does not state it or the entry cannot be written."""
    try:
        record = {"format": FORMAT, "params_from": params_from, "literals_from": literals_from,
                  # Each parameter in its order: a primitive's parameters bind as traced.
                  "params": [[_primitive_name(primitive), [[key, _encoded_value(value)]
                                                           for key, value in params.items()]]
                             for primitive, params in region.params[params_from:]],
                  "literals": [_encoded_value(np.asarray(value))
                               for value in region.literals[literals_from:]],
                  **encoded(term=_encoded_term, maybe=_encoded_maybe, value=_encoded_value)}
        os.makedirs(directory(), exist_ok=True)
        _files.write_atomically(os.path.join(directory(), f"{name}.json"), json.dumps(record, sort_keys=True))
    except (Unstated, OSError) as unstated:
        _log.info("a derivation not stored: %s", unstated)
        kernel_report.record_stored("unstorable")
        return
    kernel_report.record_stored("miss")


def _canonical(value):
    """`value` as a key states it, the same in every process: a structure by its class and fields, a mapping
    or an unordered collection sorted by its entries' canonical text, a sequence in its order, anything else
    by its text."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (type(value).__qualname__, tuple((field.name, _canonical(getattr(value, field.name)))
                                                for field in dataclasses.fields(value)))
    if isinstance(value, Mapping):
        entries = ((_canonical(k), _canonical(v)) for k, v in value.items())
        return ("mapping", tuple(sorted(entries, key=repr)))
    if isinstance(value, (set, frozenset)):
        return ("set", tuple(sorted((_canonical(member) for member in value), key=repr)))
    if isinstance(value, (tuple, list)):
        return tuple(_canonical(part) for part in value)
    return repr(value)


def _encoded_term(term) -> list:
    for name, constructor in CONSTRUCTORS.items():
        if (args := get_callable_args(term, constructor)) is not None:
            return [name, [_encoded_argument(arg) for arg in args]]
    raise Unstated(f"a term of no stored constructor: {str(term)[:80]}")


def _encoded_maybe(term) -> list | None:
    return None if term is None else _encoded_term(term)


def _encoded_argument(argument) -> list:
    if isinstance(argument, Term):
        return ["t", _encoded_term(argument)]
    if isinstance(argument, i64):
        return ["i", terms.integer(argument)]
    if isinstance(argument, String):
        return ["s", terms.text(argument)]
    if isinstance(argument, f64):
        return ["f", terms.real(argument).hex()]
    if (axes := get_callable_args(argument, Vec[i64])) is not None:
        # egglog writes a vector typed or not, each a term of its own: kept as written.
        typed = str(argument) == str(Vec[i64](*axes))
        return ["v" if typed else "u", [_encoded_argument(axis) for axis in axes]]
    raise Unstated(f"an argument of no stored kind: {str(argument)[:80]}")


@dataclasses.dataclass(frozen=True)
class _Decoded:
    """An entry's terms decoded over a region of `leaves` leaves, `literals` literals and `params`
    parameters (its own appended ones counted), carrying `states` states, each checked as it is built: a
    leaf, a literal, an operation's parameters and a state in range, a state of a side the term's form reads
    (`sides`: a step's and a finish's the walk's own, 0; a merge's either walk's, 0 or 1; a region's term
    none), each tag a constructor's (`ValueError` otherwise)."""

    leaves: int
    literals: int
    params: int
    states: int

    def __call__(self, encoded: list, *, sides: tuple[int, ...]) -> Term:
        name, arguments = encoded
        constructor = CONSTRUCTORS[name]
        if name == "state":
            slot, side = (_argument(argument, self, sides) for argument in arguments)
            if not (0 <= terms.integer(slot) < self.states and terms.integer(side) in sides):
                raise ValueError(f"a state {terms.integer(slot)} of side {terms.integer(side)} no form reads")
            return constructor(slot, side)
        values = [_argument(argument, self, sides) for argument in arguments]
        if name in _INDEXED:
            held, at = _INDEXED[name]
            if not 0 <= terms.integer(values[at]) < getattr(self, held):
                raise ValueError(f"a {name} past its region's {held}")
        elif values and isinstance(values[-1], i64) and not 0 <= terms.integer(values[-1]) < self.params:
            # An operation's parameters by index, its last argument where it is one.
            raise ValueError(f"a {name}'s parameters past its region's")
        return constructor(*values)

    def maybe(self, encoded: list | None, *, sides: tuple[int, ...]) -> Term | None:
        return None if encoded is None else self(encoded, sides=sides)


def _argument(encoded: list, decoded: _Decoded, sides: tuple[int, ...]):
    kind, value = encoded
    if kind == "t":
        return decoded(value, sides=sides)
    if kind == "i":
        return i64(value)
    if kind == "s":
        return String(value)
    if kind == "f":
        return f64(float.fromhex(value))
    if kind in ("v", "u"):
        axes = [_argument(axis, decoded, sides) for axis in value]
        return Vec[i64](*axes) if kind == "v" else Vec(*axes)
    raise ValueError(f"an argument of no stored kind {kind!r}")


@functools.cache
def _primitives() -> Mapping[str, Primitive]:
    """jax.lax's primitives by their names: what a stored parameter names."""
    return {value.name: value for value in vars(jax.lax).values() if isinstance(value, Primitive)}


def _primitive_name(primitive) -> str:
    if _primitives().get(primitive.name) is not primitive:
        raise Unstated(f"a primitive jax.lax does not name: {primitive}")
    return primitive.name


def _primitive(name: str):
    return _primitives()[name]


def _encoded_value(value) -> list:
    """`value` as an entry writes it, its type kept: an integer an integer, a dtype a dtype and a NumPy
    scalar class that class."""
    if value is None or isinstance(value, (bool, int, str)):
        return ["p", value]
    if isinstance(value, float):
        return ["x", value.hex()]
    if isinstance(value, tuple):
        return ["T", [_encoded_value(part) for part in value]]
    if isinstance(value, np.dtype):
        return ["d", value.name]
    if isinstance(value, type) and issubclass(value, np.generic):
        return ["c", np.dtype(value).name]
    if isinstance(value, (np.ndarray, np.generic)):
        held = np.asarray(value)
        return ["a" if isinstance(value, np.ndarray) else "g", held.dtype.name, list(held.shape),
                held.tobytes().hex()]
    raise Unstated(f"a value of no stored kind: {type(value).__name__}")


def _value(encoded: list):
    kind = encoded[0]
    if kind == "p":
        return encoded[1]
    if kind == "x":
        return float.fromhex(encoded[1])
    if kind == "T":
        return tuple(_value(part) for part in encoded[1])
    if kind == "d":
        return np.dtype(encoded[1])
    if kind == "c":
        return np.dtype(encoded[1]).type
    if kind in ("a", "g"):
        held = np.frombuffer(bytes.fromhex(encoded[3]), np.dtype(encoded[1])).reshape(encoded[2]).copy()
        return held if kind == "a" else held[()]
    raise ValueError(f"a value of no stored kind {kind!r}")
