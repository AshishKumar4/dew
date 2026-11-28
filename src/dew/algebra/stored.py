"""A template's trace-time derivations (`derive._derived`'s forms at one tile) stored across processes in
Dew's cache, so a process traces a fixed tile's forms from what an earlier one derived, and read from a
store no run writes (`shipped`: a built package's, the derivations its declared steps' templates request,
derived as it was built by `tools/derived_catalogue.py`, or one a site built), after Dew's cache's
(docs/design/algebra.md §6.5).

A derivation is a pure function of what it reads: its declaration's traced program (its structural digest,
`terms.jaxpr_digest`, and its captured constants' values), its walked axes, its tile's shapes and dtypes,
its numerics' class and limits, the context its forms are priced and traced in (`derive.context`), the rule
sets it saturates with (by name, each as the source registers it, immutable, `rules.sourced`; a set
registered at run time stores nothing), the algebra's code (`dew._code.code_digest`) and egglog's and JAX's
versions, each its exact build's (Dew's patched egglog build where pyproject.toml pins one, upstream's
release elsewhere): an entry is read only by the egglog that derived it, so a shipped one is read where the
package's build was derived and derived afresh elsewhere, never an entry another build could saturate or
extract otherwise. The program's digest states each primitive by its name, so a program is keyed only where
each of its primitives, at every depth, is the one JAX binds to that name (`_bound`): a region tells
primitives apart by identity (`terms.OPERATIONS`), and a user's primitive of a JAX name is not keyed.
An entry is keyed by all of them (`key`, each part by the algebra's structural key, `terms.plain_key`: an
unordered collection sorted, a structure by its fields, a part held by identity keying nothing), every mesh by
its axes' names and types and none of their sizes (`terms.unsized`), beside its region in order (`_inventory`:
each output, opaque input and shaped term, its parameter slots normalized): a derivation reads whether an axis
is explicit or manual (`terms.unplaced_equation`, a tangent sum's add), never its size, so a run on one device
and one on eight read one entry (its forms in a process, which hold their mesh's avals, are kept per mesh,
`derive.context`). Written whole (`dew._files.write_atomically`) and read back exactly: each chosen term
through the Term language's constructors (`CONSTRUCTORS`, `get_callable_args`), its arguments as their
literals, a vector as written (egglog writes it typed or not, each a term of its own), each term once in a
table its forms index (`_Writer`: a term its forms share is written once); the parameters, literals and shapes
the derivation appended to its region past its trace by their primitive's name and their values' types, in
their order (the shapes it appended, its appended literals' own, `_appended`), the region itself traced
afresh. No value is pickled or evaluated. An entry's text is digested with its name (`_summed`), and one whose
digest is not its own is not decoded; an entry is checked against the region it extends as it decodes (each
value of its kind, each term's leaf, literal, parameter and state in range, each state of a side its form
reads, each count and part of its forms the region's) before any of it is taken, and the region extended once
all of it has read.

A derivation a key cannot state (a value held by identity, an outer trace's or a device's) or a value the
codec cannot state is derived afresh and not stored, as is one whose entry cannot be written; an entry that
fails to open, read or decode is derived afresh past. The extractor's oracle (`Passes.checked`) derives and
never reaches here. Each outcome is counted in the kernel report (`report.record_stored`)."""

from __future__ import annotations

import dataclasses
import enum
import functools
import hashlib
import importlib.metadata
import json
import logging
import os
import re
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

import jax
import numpy as np
from egglog import ConvertError, String, Vec, f64, get_callable_args, i64
from jax.core import ShapedArray
from jax.extend.core import Jaxpr, JaxprEqn, Primitive, primitives

from dew import _code, _files
from dew._builds import dew_cache_dir
from dew.algebra import rules, terms
from dew.algebra.terms import Term
from dew.nn.kernels.templates import report as kernel_report

if TYPE_CHECKING:
    from dew.algebra.derive import _Forms

_log = logging.getLogger(__name__)

FORMAT = 5
"""The entries' layout and key, in every key: an entry of another is never read."""

CONSTRUCTORS: Mapping[str, Callable] = {
    **{written.constructor: getattr(Term, written.constructor) for written in terms.WRITTEN.values()},
    **{name: getattr(Term, name) for name in ("leaf", "literal", "state")}}
"""The constructors a derived form is written in, by name: each operation of the language (`terms.WRITTEN`)
and the terms an evaluation reads beside them, a tile's leaf, a literal and a carried state."""

class Unstated(Exception):
    """A value the store's key or codec does not state: its derivation is not stored."""


SHIPPED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "derived")
"""The entries a built package ships (`tools/derived_catalogue.py`, its package data): none in a checkout
of the source, whose tree holds none (`.gitignore`)."""


def directory() -> str:
    """Where the entries a process writes are, in Dew's cache, read before the read-only ones (`shipped`)."""
    return os.path.join(dew_cache_dir(), "algebra", "derived")


def shipped() -> str:
    """Where the read-only entries are, read after Dew's cache's and never written: `DEW_DERIVATIONS` where
    it is set (a store `tools/derived_catalogue.py` built for a site's own GPUs and meshes, every run of it
    reading it), else the package's (`SHIPPED`)."""
    return os.environ.get("DEW_DERIVATIONS") or SHIPPED


def key(closed, region: terms.Region, sets: Sequence[str], parts: Sequence[object]) -> str | None:
    """The entry's name of a derivation of the traced `closed` saturated with the source's rule sets named
    `sets` (`rules.sourced`), in their order, beside `parts` (everything else it reads, `derive._derived`'s),
    each by the algebra's structural key (`terms.plain_key`), every mesh among them by its axes' names and
    types (`terms.unsized`); None where its program (a primitive its name does not identify, `_bound`), a
    captured constant (an outer trace's value, or one on a device) or a part is held by identity."""
    equations = _equations(closed)
    if equations is None or not _bound(equations):
        _log.info("a derivation not keyed: a primitive its name does not identify")
        return None
    if not all(isinstance(const, (np.ndarray, np.generic, bool, int, float)) for const in closed.consts):
        _log.info("a derivation not keyed: a constant on a device")
        return None
    with terms.unsized():
        digest, opaques = terms.jaxpr_digest(closed, in_process=False)
        interned = _inventory(region)
        stated = None if interned is None else terms.plain_key((list(closed.consts), interned, list(sets),
                                                                tuple(parts),
                                  importlib.metadata.version("egglog"), jax.__version__,
                                  _code.code_digest("dew.algebra.derive")))
    if opaques or stated is None:
        _log.info("a derivation not keyed: a program or part held by identity")
        return None
    return hashlib.blake2b(repr((FORMAT, digest, stated)).encode(), digest_size=20).hexdigest()


def loaded(name: str, region: terms.Region, assembled: Callable[..., _Forms], *,
           rescaled: frozenset[str]) -> tuple[_Forms, dict] | None:
    """The stored derivation `name` over `region`, and its record (which a class of sizes binds at its
    others, `dew.algebra.derive._classed`), traced afresh, its appended parameters, literals and
    shapes replayed once the whole entry reads (`assembled(record, decoded=..., ...)` its forms, each decoded
    term checked against the region, `_Decoded`, a rescale reading an exponential of `rescaled`, those its
    declaration's own rescale reads, `_operated`); None where none is stored, or where the entry fails to
    open, is not its own (`_summed`) or fails to decode (counted `invalid`, derived afresh past). Read from
    Dew's cache where it holds the entry, else from the read-only ones (`shipped`)."""
    for held in (directory(), shipped()):
        path = os.path.join(held, f"{name}.json")
        try:
            with open(path, "rb") as file:
                text = file.read()
            break
        except FileNotFoundError:
            continue
        except OSError as unreadable:
            _invalid(path, unreadable)
            return None
    else:
        _log.debug("no stored derivation %s: derived afresh", name)
        return None
    try:
        record = _record(name, text)
        made = bound(record, region, assembled, rescaled=rescaled)
    except (KeyError, TypeError, ValueError, IndexError, OverflowError, RecursionError,
            ConvertError) as failed:
        # ConvertError: an argument of a sort its constructor does not take; OverflowError: a float past
        # range; RecursionError: a value nested past Python's depth.
        _invalid(path, failed)
        return None
    kernel_report.record_stored("hit")
    return made, record


def bound(record: Mapping, region: terms.Region, assembled: Callable[..., _Forms], *,
          rescaled: frozenset[str], resized: Callable[[Mapping], dict] | None = None) -> _Forms:
    """The forms `record` holds (`record_of`'s) over `region`, traced afresh, its appended parameters,
    literals and shapes replayed once the whole record reads (`assembled(record, decoded=..., ...)` its
    forms, each decoded term checked against the region, `_Decoded`, a rescale reading an exponential of
    `rescaled`, `_operated`; each appended parameter's values `resized` where given, a record of another
    size of the region's class, `dew.algebra.derive._classed`): the region extended past what it traced once
    every form decodes, and left as traced where one fails (`KeyError`, `TypeError`, `ValueError`,
    `IndexError`, `OverflowError`, `RecursionError` or egglog's `ConvertError`)."""
    if (record.get("params_from"), record.get("literals_from"), record.get("shapes_from")) != (
            len(region.params), len(region.literals), len(region.shapes)):
        raise ValueError("an entry of another trace's region")
    params = [parameter(entry) for entry in _listed(record["params"])]
    if resized is not None:
        params = [(primitive, resized(values)) for primitive, values in params]
    literals = [_literal(value) for value in _listed(record["literals"])]
    decoded = _decoded(_listed(record["terms"]), leaves=region.leaves,
                       literals=len(region.literals) + len(literals),
                       primitives=[primitive for primitive, _ in (*region.params, *params)],
                       states=len(_listed(record["steps"])), rescaled=rescaled)
    made = assembled(record, decoded=decoded, value=_value)
    # Each literal a derivation appends is shaped as it is (`_appended`).
    shapes = [(Term.literal(i64(index)), literal.shape)
              for index, literal in enumerate(literals, start=len(region.literals))]
    region.params.extend(params)
    region.literals.extend(literals)
    for term, extent in shapes:
        region.shaped(term, extent)
    return made


def _inventory(region: terms.Region) -> list | None:
    """`region` (the traced program's, `key`'s) as a derivation reads it, in order, each term by its index in
    one table of the store's codec (`_Writer`: a term once, its slots' and literals' indices within it): each
    output by its position (None a literal's), each opaque equation's inputs by theirs, each shaped term and
    its extent in the region's order, and the parameter table its terms index, normalized as the region
    interns it (`terms.Region.interned`): an entry's forms and indices then bind alike in every region of
    one key. None where a term is of no stored kind."""
    writer = _Writer()
    try:
        outputs = [writer.maybe(term) for term in region.outputs]
        opaque = [[writer.maybe(term) for term in held.inputs] for held in region.opaque]
        shaped = [[writer.term(term), list(extent)] for term, extent in region.shapes]
    except Unstated as unstated:
        _log.info("a derivation not keyed: its region holds %s", unstated)
        return None
    return [writer.table, outputs, opaque, shaped,
            [(primitive.name, params) for primitive, params in region.params]]


def _one_directory(first: str, second: str) -> bool:
    """Whether `first` and `second` are one directory: one path once resolved (whatever either holds now, a
    directory replaced between two reads among them), or one file where both exist (`os.path.samefile`: two
    mounts of one). An error reading either but its absence is raised (`stored` counts it unstorable)."""
    if os.path.realpath(first) == os.path.realpath(second):
        return True
    try:
        return os.path.samefile(first, second)
    except FileNotFoundError:
        return False


def entry(path: str | os.PathLike) -> bool:
    """Whether the file at `path` is a store's entry: named by a key, its text its own and this format's
    (`_record`), as `loaded` reads one."""
    name = os.path.basename(path).removesuffix(".json")
    if not re.fullmatch(r"[0-9a-f]{40}", name) or not os.path.isfile(path):
        return False
    try:
        with open(path, "rb") as file:
            _record(name, file.read())
    except (OSError, ValueError, RecursionError):
        return False
    return True


def _record(name: str, text: bytes) -> dict:
    """The record an entry `name` of `text` holds, its digest checked before its body is parsed (`_summed`: a
    text not its own is never read) and its format this one's; `ValueError` otherwise (`RecursionError`
    where a body of its own nests past Python's depth)."""
    summed, _, body = text.decode().partition("\n")
    if summed != _summed(name, body):
        raise ValueError("an entry whose digest is not its own")
    record = json.loads(body)
    if not isinstance(record, dict) or record.get("format") != FORMAT:
        raise ValueError("an entry of another layout")
    return record


def _summed(name: str, body: str) -> str:
    """The digest an entry `name` of text `body` is written under: an entry is read only where it is its
    own."""
    return hashlib.blake2b(f"{name}\n{body}".encode(), digest_size=20).hexdigest()


def _invalid(path: str, failed: Exception) -> None:
    _log.warning("the stored derivation %s fails to read (%s): derived afresh", path, failed)
    kernel_report.record_stored("invalid")


def stored(name: str, region: terms.Region, params_from: int, literals_from: int, shapes_from: int,
           encoded: Callable[..., Mapping[str, object]]) -> dict | None:
    """Write a derivation's forms derived afresh over `region` as the entry `name` (`record_of`'s record),
    and the record, which a class of sizes binds at its others (`dew.algebra.derive._classed`): counted
    `miss` once written; `unstorable`, and not written, where the codec does not state it (no record) or the
    entry cannot be written."""
    record = None
    try:
        record = record_of(region, params_from, literals_from, shapes_from, encoded)
        written(name, record)
    except (Unstated, OSError) as unstated:
        _log.info("a derivation not stored: %s", unstated)
        kernel_report.record_stored("unstorable")
        return record
    kernel_report.record_stored("miss")
    return record


def record_of(region: terms.Region, params_from: int, literals_from: int, shapes_from: int,
              encoded: Callable[..., Mapping[str, object]]) -> dict:
    """A derivation's forms over `region` as a record (`encoded(term=..., maybe=..., value=...)` its forms,
    each term by its index in one table of the codec's), with the parameters, literals and shapes the
    derivation appended to its region past `params_from`, `literals_from` and `shapes_from`: what an entry
    holds (`stored`) and a family of its structure (`dew.algebra.derive._Family`); `Unstated` where the
    codec does not state it."""
    writer = _Writer()
    forms = encoded(term=writer.term, maybe=writer.maybe, value=encoded_value)
    _appended(region, literals_from, shapes_from)
    return {"format": FORMAT, "params_from": params_from, "literals_from": literals_from,
            "shapes_from": shapes_from,
            # Each parameter in its order: a primitive's parameters bind as traced.
            "params": [encoded_parameter(primitive, params)
                       for primitive, params in region.params[params_from:]],
            "literals": [encoded_value(np.asarray(value)) for value in region.literals[literals_from:]],
            **forms, "terms": writer.table}


def encoded_parameter(primitive: Primitive, params: Mapping) -> list:
    """A parameter as a record holds it (`record_of`; `_parameter` reads it back)."""
    return [_primitive_name(primitive), [[key, encoded_value(value)] for key, value in params.items()]]


def written(name: str, record: Mapping) -> None:
    """`record` written as the entry `name` in Dew's cache; `Unstated` where Dew's cache is the read-only
    store, `OSError` where it cannot be written."""
    body = json.dumps(record, sort_keys=True)
    if _one_directory(directory(), shipped()):
        raise Unstated("Dew's cache is the read-only store (`DEW_DERIVATIONS`), never written")
    os.makedirs(directory(), exist_ok=True)
    _files.write_atomically(os.path.join(directory(), f"{name}.json"), f"{_summed(name, body)}\n{body}")


def _appended(region: terms.Region, literals_from: int, shapes_from: int) -> None:
    """That the shapes a derivation appended to `region` past `shapes_from` are its literals' appended past
    `literals_from`, each in its order at its own shape, as a reading replays them (`loaded`); `Unstated`
    otherwise."""
    held = [(terms.key(term), tuple(extent)) for term, extent in region.shapes[shapes_from:]]
    if held != [(terms.key(Term.literal(i64(index))), np.shape(region.literals[index]))
                for index in range(literals_from, len(region.literals))]:
        raise Unstated("a derivation's appended shapes are not its appended literals'")


def _called(term) -> tuple[str, tuple]:
    """`term`'s constructor's name and its arguments (`CONSTRUCTORS`)."""
    for name, constructor in CONSTRUCTORS.items():
        if (args := get_callable_args(term, constructor)) is not None:
            return name, tuple(args)
    raise Unstated(f"a term of no stored constructor: {str(term)[:80]}")


class _Writer:
    """An entry's terms as it writes them: each term once, in a table (`table`) of its constructor's name and
    its arguments, a term argument by its index there, every argument of an entry's before it; a form by its
    term's index (`term`). A term its forms share is written once, as egglog holds it once (`terms.key`)."""

    def __init__(self):
        self.table: list[list] = []
        self.at: dict[object, int] = {}

    def term(self, term: Term) -> int:
        """`term`'s index in the table, it and each of its terms written where they are not yet."""
        pending: list[tuple[Term, tuple[str, tuple] | None]] = [(term, None)]
        while pending:
            held, called = pending.pop()
            if terms.key(held) in self.at:
                continue
            if called is None:
                called = _called(held)
                pending.append((held, called))
                pending.extend((argument, None) for argument in called[1] if isinstance(argument, Term))
                continue
            name, args = called
            self.table.append([name, [self._argument(argument) for argument in args]])
            self.at[terms.key(held)] = len(self.table) - 1
        return self.at[terms.key(term)]

    def maybe(self, term: Term | None) -> int | None:
        return None if term is None else self.term(term)

    def _argument(self, argument) -> list:
        if isinstance(argument, Term):
            return ["t", self.at[terms.key(argument)]]
        if isinstance(argument, i64):
            return ["i", terms.integer(argument)]
        if isinstance(argument, String):
            return ["s", terms.text(argument)]
        if isinstance(argument, f64):
            return ["f", terms.real(argument).hex()]
        if (axes := get_callable_args(argument, Vec[i64])) is not None:
            # egglog writes a vector typed or not, each a term of its own: kept as written.
            typed = str(argument) == str(Vec[i64](*axes))
            return ["v" if typed else "u", [self._argument(axis) for axis in axes]]
        raise Unstated(f"an argument of no stored kind: {str(argument)[:80]}")


@dataclasses.dataclass(frozen=True)
class _Decoded:
    """An entry's table of terms decoded (`_decoded`), each with the sides of the states it reads (`reads`):
    a form by its index, read where every state it reads is of a side the form reads (`sides`: a step's and
    a finish's the walk's own, 0; a merge's either walk's, 0 or 1; a region's term none; `ValueError`
    otherwise)."""

    terms: tuple[Term, ...]
    reads: tuple[frozenset[int], ...]

    def __call__(self, index: int, *, sides: tuple[int, ...]) -> Term:
        if not (integral(index) and 0 <= index < len(self.terms)):
            raise ValueError(f"a form {index!r} past its entry's terms")
        if not self.reads[index] <= frozenset(sides):
            raise ValueError(f"a form reading states of sides {sorted(self.reads[index])}, not {sides}")
        return self.terms[index]

    def maybe(self, index: int | None, *, sides: tuple[int, ...]) -> Term | None:
        return None if index is None else self(index, sides=sides)


def _decoded(table: list, *, leaves: int, literals: int, primitives: Sequence[Primitive],
             states: int, rescaled: frozenset[str] = frozenset()) -> _Decoded:
    """An entry's `table` (`_Writer.table`) decoded over a region of `leaves` leaves, `literals` literals and
    parameters of `primitives` (its own appended ones counted), carrying `states` states, each term checked
    as it is built: each argument of its kind and a term's one before it, a leaf, a literal, an operation's
    parameters and a state in range, an operation's parameters its own (`_operated`; a rescale's an
    exponential of `rescaled`, none by default), a state of side 0 or 1, each name a constructor's
    (`ValueError` otherwise)."""
    made: list[Term] = []
    reads: list[frozenset[int]] = []
    for entry in table:
        name, arguments = _listed(entry)
        constructor = CONSTRUCTORS[name]
        read: frozenset[int] = frozenset()
        values = []
        for argument in _listed(arguments):
            kind, value = _listed(argument)
            if kind == "t":
                if not (integral(value) and 0 <= value < len(made)):
                    raise ValueError(f"a term argument {value!r} not before its term")
                read |= reads[value]
                values.append(made[value])
            else:
                values.append(_argument(kind, value))
        if name == "state":
            slot, side = (terms.integer(value) for value in values)
            if not (0 <= slot < states and side in (0, 1)):
                raise ValueError(f"a state {slot} of side {side} past its forms'")
            read = frozenset({side})
        elif name in ("leaf", "literal"):
            # A leaf's and a literal's index, its first argument.
            if not 0 <= terms.integer(values[0]) < (leaves if name == "leaf" else literals):
                raise ValueError(f"a {name} past its region's")
        elif values and isinstance(values[-1], i64):
            # An operation's parameters by index, its last argument where it is one.
            if not 0 <= (index := terms.integer(values[-1])) < len(primitives):
                raise ValueError(f"a {name}'s parameters past its region's")
            _operated(name, values, primitives[index], rescaled)
        made.append(constructor(*values))
        reads.append(read)
    return _Decoded(tuple(made), tuple(reads))


def _operated(name: str, values: list, primitive: Primitive, rescaled: frozenset[str]) -> None:
    """That the parameters of `primitive` a term of constructor `name` over `values` reads are an operation's
    the term writes (`terms.operations_written`), the primitive's own (`terms.OPERATIONS`, by identity):
    where it holds a name (its second argument), the operation of that symbol or monoid, a fused
    multiply-add's name the operation it binds as (`rules.BOUND_AS`). A derived walk's rescale
    (`terms.RESCALE`), no program's operation, reads an exponential's (`derive.scales`) of `rescaled`, those
    its declaration's own rescale reads: none where it states no rescale, whose walk writes none. `ValueError`
    otherwise."""
    named = terms.text(values[1]) if len(values) > 1 and isinstance(values[1], String) else None
    operation = terms.OPERATIONS.get(primitive)
    if name == "binary" and named == terms.RESCALE:
        if operation is None or operation.symbol not in rescaled:
            raise ValueError(f"a rescale reading {primitive}'s parameters, no exponential of "
                             f"{sorted(rescaled)}")
        return
    bound = None if named is None else rules.BOUND_AS.get(named, named)
    if operation is None or operation not in (terms.operations_written(name, bound) or ()):
        raise ValueError(f"a {name} {named or ''} reading {primitive}'s parameters")


def _argument(kind: str, value):
    """A term's argument other than a term, of its `kind`, checked of its type."""
    if kind == "i" and integral(value):
        return i64(value)
    if kind == "s" and isinstance(value, str):
        return String(value)
    if kind == "f" and isinstance(value, str):
        return f64(float.fromhex(value))
    if kind in ("v", "u"):
        # A vector of integers, as `_Writer` writes one; egglog types an untyped one by its first axis.
        axes = [_listed(axis) for axis in _listed(value)]
        if not all(len(axis) == 2 and axis[0] == "i" and integral(axis[1]) for axis in axes) or (
                kind == "u" and not axes):
            raise ValueError(f"a vector {value!r} of no integers")
        return Vec[i64](*(i64(axis[1]) for axis in axes)) if kind == "v" else Vec(*(i64(axis[1])
                                                                                  for axis in axes))
    raise ValueError(f"an argument of no stored kind {kind!r}")


def integral(value) -> bool:
    """Whether `value`, read from JSON, is an integer (a boolean is not): an index or count an entry
    states."""
    return isinstance(value, int) and not isinstance(value, bool)


def _listed(value) -> list:
    """`value`, read from JSON, where it is a list (`ValueError` otherwise)."""
    if isinstance(value, list):
        return value
    raise ValueError(f"a {type(value).__name__} where a list is written")


def _extent(value) -> tuple[int, ...]:
    """A shape read from JSON: a list of nonnegative integers."""
    if not all(integral(size) and size >= 0 for size in _listed(value)):
        raise ValueError(f"a shape {value!r} of no sizes")
    return tuple(value)


def _literal(encoded) -> np.ndarray:
    """An appended literal read from JSON: an array, as a derivation appends one (`np.asarray`)."""
    value = _value(encoded)
    if not isinstance(value, np.ndarray):
        raise ValueError(f"a literal of {type(value).__name__}, no array")
    return value


def parameter(entry) -> tuple[Primitive, dict]:
    """An appended parameter read from JSON: its primitive by name and its values by theirs."""
    primitive, given = _listed(entry)
    values = {}
    for pair in _listed(given):
        name, value = _listed(pair)
        if not isinstance(name, str):
            raise ValueError(f"a parameter named {name!r}")
        values[name] = _value(value)
    return _primitive(primitive), values


@functools.cache
def _primitives() -> Mapping[str, Primitive]:
    """JAX's primitives by their names, as `jax.lax` and `jax.extend.core.primitives` bind them, a name two
    of them share left out: what a program's digest and a stored parameter name."""
    held: dict[str, dict[int, Primitive]] = {}
    for module in (jax.lax, primitives):
        for value in vars(module).values():
            if isinstance(value, Primitive):
                held.setdefault(value.name, {})[id(value)] = value
    return {name: next(iter(found.values())) for name, found in held.items() if len(found) == 1}


_PLAIN = (type(None), bool, int, float, complex, str, bytes, np.dtype, np.generic, np.ndarray, enum.Enum,
          type, ShapedArray, jax.sharding.Sharding, jax.sharding.Mesh, jax.sharding.AbstractMesh)
"""The parameter values that hold no program: what `_bound` reads past without descending."""


def _equations(closed: Jaxpr) -> list[JaxprEqn] | None:
    """Each equation of `closed` and of every program its parameters hold, at any depth, in one order: its
    parameters read through sequences and mappings (their keys and values), each once, to programs and plain
    values (`_PLAIN`); None where a parameter of any other kind, which may hold a program out of this walk's
    sight, is met."""
    found: list[JaxprEqn] = []
    pending: list = [closed]
    seen: set[int] = set()
    while pending:
        value = pending.pop()
        if isinstance(value, _PLAIN) or id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, Jaxpr):
            for equation in value.jaxpr.eqns:
                found.append(equation)
                pending.extend(equation.params.values())
        elif isinstance(value, (tuple, list, set, frozenset)):
            pending.extend(value)
        elif isinstance(value, Mapping):
            pending.extend((*value.keys(), *value.values()))
        else:
            return None
    return found


def _bound(equations: Sequence[JaxprEqn]) -> bool:
    """Whether each primitive of `equations` is the one JAX binds to its name (`_primitives`)."""
    bound = _primitives()
    return all(bound.get(equation.primitive.name) is equation.primitive for equation in equations)


def _primitive_name(primitive) -> str:
    if _primitives().get(primitive.name) is not primitive:
        raise Unstated(f"a primitive jax.lax does not name: {primitive}")
    return primitive.name


def _primitive(name) -> Primitive:
    """A primitive read from JSON by its name (`_primitives`)."""
    if not isinstance(name, str):
        raise ValueError(f"a primitive named {name!r}")
    return _primitives()[name]


def encoded_value(value) -> list:
    """`value` as an entry writes it, its type kept: an integer an integer, a dtype a dtype and a NumPy
    scalar class that class."""
    if value is None or isinstance(value, (bool, int, str)):
        return ["p", value]
    if isinstance(value, float):
        return ["x", value.hex()]
    if isinstance(value, tuple):
        return ["T", [encoded_value(part) for part in value]]
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
    """A value `encoded_value` wrote, checked of its kind's type (`ValueError` otherwise)."""
    kind, *parts = _listed(encoded)
    if kind == "p" and len(parts) == 1 and (parts[0] is None or isinstance(parts[0], (bool, int, str))):
        return parts[0]
    if kind == "x" and len(parts) == 1 and isinstance(parts[0], str):
        return float.fromhex(parts[0])
    if kind == "T" and len(parts) == 1:
        return tuple(_value(part) for part in _listed(parts[0]))
    if kind in ("d", "c") and len(parts) == 1:
        dtype = _dtype(parts[0])
        return dtype if kind == "d" else dtype.type
    if kind in ("a", "g") and len(parts) == 3 and isinstance(parts[2], str):
        dtype, shape = _dtype(parts[0]), _extent(parts[1])
        held = np.frombuffer(bytes.fromhex(parts[2]), dtype).reshape(shape).copy()
        return held if kind == "a" else held[()]
    raise ValueError(f"a value of no stored kind {kind!r}")


def _dtype(name) -> np.dtype:
    """A dtype read from JSON by its name: one of numbers, never of objects."""
    if not isinstance(name, str):
        raise ValueError(f"a dtype named {name!r}")
    dtype = np.dtype(name)
    if dtype.hasobject:
        raise ValueError(f"a dtype {name!r} of objects")
    return dtype
