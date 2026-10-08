"""Short names for the classes a run is made of, and the records that rebuild them.

A record names a class by its import path, `{"class": "dew.nn.dit:SimpleDiT",
"fields": {...}}`, and a function the same way, `{"function": "optax:adamw"}`.
Rebuilding one imports the module and calls the class with its fields, each
converted to the type its annotation declares, so a record loads in a process
that imported nothing beforehand, and a class outside Dew needs nothing to be
recorded: its import path is its name. A record Dew writes carries the path,
so rebuilding it never depends on an alias.

Where a person writes a record, in a config file or on the command line, the
class may be a short alias instead, `{"class": "simple_dit"}`. Each kind's
aliases are a static table below (`models`, `mixers`, ...) that maps a name to
a path and imports the class only when it is read. An unknown name or field
raises an error.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import datetime
import functools
import importlib
import inspect
import operator
import sys
import types
import typing
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Union, overload

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike
from typing_extensions import Format, get_annotations, is_protocol

from dew.records import JSON

if TYPE_CHECKING:
    from _typeshed import DataclassInstance
    from flax import linen as nn

    from dew.data.dataset import DatasetSpec
    from dew.diffusion.presets import Preset
    from dew.inputs.encoders import ConditionEncoder
    from dew.nn.mixers import MixerBase
    from dew.nn.vision import ProjectorBase, TowerBase
    from dew.objectives.base import Metric, Objective
    from dew.sampling.solvers import Solver
    from dew.training.optim import ScheduleBase

# A type annotation as a value: a class, a union of them, a subscripted
# generic, a PEP 695 alias, or the None a field with no resolvable annotation
# leaves behind. Every reader below takes one of these and asks it what it is.
type Annotation = type | types.UnionType | types.GenericAlias | typing.TypeAliasType | None

# One field of a member, as a caller writes it or a config file carries it:
# the JSON a file holds, a dtype, an enum member such as a matmul precision,
# an array a schedule was tabulated into, a callable a field takes as a hook,
# a value class already built (every value class here is a dataclass, a Flax
# module included), and the records and sequences of any of those. This is
# the width of what may arrive; `build` narrows each one again against the
# type its member declared before the member ever sees it.
type Configured = (JSON | DTypeLike | Enum | np.ndarray | np.generic
                   | types.FunctionType | types.BuiltinFunctionType
                   | DataclassInstance | Mapping[str, object]
                   | Mapping[RecordKey, object] | Sequence[Configured])

# A config mapping's key as a record reads it back (`_key`): a name, a
# number, or a tree path of them.
type RecordKey = str | int | float | tuple[str | int | float, ...]

# `build` called with no record at all, which is every caller that writes its
# fields as keywords. Shared because it is read and never written.
NO_RECORD: Mapping[str, object] = types.MappingProxyType({})



Record = TypedDict("Record", {"class": str, "fields": Mapping[str, object]})
"""A class as a record names it: its alias or import path and its constructor
fields, `{"class": "mla", "fields": {...}}`."""


type Importable = type | types.FunctionType | types.BuiltinFunctionType | types.MethodType
"""What a record names by import path: a class, a function, or a classmethod bound to its class."""


def import_path(member: Importable) -> str:
    """The path a record names a class or function by, `module:Qualified.name`.

    A lambda, a function defined inside another, or anything defined in
    `__main__` has no path another process can import, so it is refused here,
    where the record is written, rather than when it is read."""
    function = member.__func__ if isinstance(member, types.MethodType) else member
    module, name = function.__module__, function.__qualname__
    if not module or module == "__main__" or "<" in name:
        raise ValueError(f"{member!r} has no import path a record can name; define it at module level "
                         f"in an importable module to record it")
    return f"{module}:{name}"


def imported(path: str) -> Callable[..., Configured]:
    """The class or function an import path names, importing its module.

    Records arrive from the Hub, and importing a module runs its top level,
    so a record imports only Dew's own modules and modules already imported.
    Code that defines its own classes imports them before it loads a run of
    them; a run of a package not yet imported loads once the caller trusts
    the package (`import_trusted`), as `trust_remote_code` does in
    transformers."""
    module, _, name = path.partition(":")
    if not module or not name:
        raise ValueError(f"{path!r} is not an import path, `module:Qualified.name`")
    if module != "dew" and not module.startswith("dew.") and module not in sys.modules:
        package = module.split(".")[0]
        raise ValueError(f"{path!r} names {module}, which is outside Dew and not imported; import it "
                         f"first, or trust its package where the run loads (trust=({package!r},), "
                         f"--trust {package})")
    # Whatever the path names, untyped until `callable` says what it is.
    held: Any = importlib.import_module(module)
    for part in name.split("."):
        try:
            held = getattr(held, part)
        except AttributeError:
            raise ValueError(f"{module} has no {name}, which {path!r} names") from None
    if not isinstance(held, (type, types.FunctionType, types.BuiltinFunctionType, types.MethodType)):
        raise ValueError(f"{path!r} names {held!r}, which is neither a class nor a function")
    return held


_TRUSTED: set[str] = {"flax"}
"""The packages outside Dew the process trusts (`import_trusted`): from the
start Flax's, whose layers a model's record names (a `Sequential`'s)."""


def import_trusted(record: object, trust: Sequence[str]) -> None:
    """Trust the packages in `trust` for the rest of the process, and import
    every module of theirs that `record` names by an import path, so reading
    the record finds them imported (`imported`) and may construct their
    classes where a field declares no base class (`_owned`)."""
    _TRUSTED.update(trust)
    if isinstance(record, str):
        module, colon, name = record.partition(":")
        if colon and name and module.split(".")[0] in trust and module.replace(".", "").isidentifier():
            importlib.import_module(module)
    elif isinstance(record, Mapping):
        for value in record.values():
            import_trusted(value, trust)
    elif isinstance(record, list):
        for value in record:
            import_trusted(value, trust)
    elif record is not None and not isinstance(record, (bool, int, float)):
        raise ValueError(f"{record!r} is not JSON a record holds")


class Aliases[T: Callable[..., Any], Built](Mapping[str, T]):
    """The short names of one kind, each mapped to the import path of its class.

    `models["simple_dit"]` imports and returns `SimpleDiT`, as does
    `models["dew.nn.backbones.dit:SimpleDiT"]`: the table is static, and a
    class outside it, Dew's or a user's, is named by its path."""

    def __init__(self, kind: str, paths: Mapping[str, str], base: str | None = None):
        self.kind = kind
        self.paths = types.MappingProxyType(dict(paths))
        self.base = base
        """The import path of the class every member derives from, or None
        where the members share only a protocol."""
        # What a lookup imported, untyped as an import is; typed on the way out.
        self._imported: dict[str, Any] = {}

    def __getitem__(self, name: str) -> T:
        """The member an alias or import path names. The table's own entries
        are code's; a path may come from a record, name anything imported, and
        be built, so its member must derive from `base` (or be a function
        declared to return one), and with no base be a class of Dew's or of a
        trusted package (`_owned`)."""
        if ":" not in name and name not in self.paths:
            raise KeyError(f"no {self.kind} named {name!r}; known: {', '.join(sorted(self.paths))}, "
                           f"or any class by its import path")
        if name not in self._imported:
            member = imported(self.paths.get(name, name))
            if name not in self.paths and not self._holds(member):
                raise ValueError(f"{name!r} names {member!r}, which is no {self.kind}")
            self._imported[name] = member
        return self._imported[name]

    def _holds(self, member: Callable[..., Configured]) -> bool:
        if self.base is None:
            return _owned(member, called=False)
        module, _, qualified = self.base.partition(":")
        base = getattr(importlib.import_module(module), qualified)
        return issubclass(member, base) if isinstance(member, type) else _returns(member, base)

    def __iter__(self) -> Iterator[str]:
        return iter(self.paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __repr__(self) -> str:
        return f"Aliases({self.kind!r}, {sorted(self.paths)})"

    def label(self, path: str) -> str:
        """The alias of the class at import `path`, or its class name where
        it has none, for a run's human-readable name."""
        return next((alias for alias, held in self.paths.items() if held == path), path.rpartition(":")[2])

    def alias_of(self, member: Importable) -> str:
        """The alias of `member`, for a run's human-readable name; KeyError
        for a class this kind has no alias for."""
        path = import_path(member)
        for name, held in self.paths.items():
            if held == path:
                return name
        raise KeyError(f"{path} has no {self.kind} alias")

    def build(self, name: str, record: Mapping[str, object] = NO_RECORD, /,
              **fields: Configured) -> Built:
        """Construct the member an alias or import path names, from a record, keyword fields, or both.

        A field the member does not declare is an error. Fields come from JSON
        as often as from code, so a field whose declared type is a value class
        is built here from its record. For example,
        `models.build("m", attention={"heads": 8})` and
        `models.build("m", attention=Attention(heads=8))` build the same model.
        Keyword fields override the record's. The fields of a function, or of a
        class that is not a dataclass, are its parameters, converted to their
        annotations the same way.
        """
        member = self[name]
        given: Mapping[str, object] = {**record, **fields}
        held = _record_class(member)
        with _reading(given):
            given = (_declared(held, given, dtypes=True) if held is not None
                     else _arguments(member, given, dtypes=True))
        return member(**given)

    def from_record(self, record: Mapping[str, object]) -> Built:
        """Construct the member a class record names, `{"class": ..., "fields": {...}}`."""
        named = _class_record(record)
        if named is None:
            raise ValueError(f"a {self.kind} is the record that names it, "
                             f"{{'class': ..., 'fields': {{...}}}}, not {record!r}")
        name, fields = named
        try:
            self[name]
        except KeyError as error:
            # A record naming nothing known is a bad config, not a lookup.
            raise ValueError(error.args[0]) from None
        return self.build(name, fields)

    @property
    def union(self) -> type[Built] | types.UnionType:
        """The union of the members' types, for a tyro subcommand over the kind."""
        return functools.reduce(operator.or_, self.values())


def _declared_type(member: type, field: str) -> Annotation:
    """Resolve one field's annotation, without evaluating unrelated ones."""
    for owner in member.__mro__:
        annotations = get_annotations(owner, format=Format.FORWARDREF)
        if field not in annotations:
            continue
        selected = types.SimpleNamespace(__annotations__={field: annotations[field]})
        try:
            return typing.get_type_hints(
                selected, globalns=dict(vars(owner)),
                localns=vars(sys.modules[owner.__module__]))[field]
        except NameError:
            # Only this field's unavailable dependency leaves its value opaque.
            break
    return None


def resolve_alias(annotation: Annotation) -> Annotation:
    """Look a PEP 695 alias through to the type it declares. `get_origin`
    and `get_args` see nothing through one, so every reader goes through here
    before asking an annotation what it is."""
    while isinstance(annotation, typing.TypeAliasType):
        annotation = annotation.__value__
    return annotation


def _unwrapped(annotation: Annotation) -> Annotation:
    """Return `annotation` with an Optional looked through. A union of several
    members says nothing about its entries and answers None."""
    annotation = resolve_alias(annotation)
    if typing.get_origin(annotation) not in (Union, types.UnionType):
        return annotation
    members = [a for a in typing.get_args(annotation) if a is not type(None)]
    return members[0] if len(members) == 1 else None


def entry_types(annotation: Annotation, count: int) -> list[Annotation]:
    """Return the annotation of each of the `count` entries of an annotated
    container: a fixed tuple's per-position types, otherwise its one element
    type repeated (a mapping's value type, a sequence's element). None is an
    unannotated entry, which takes its value as given."""
    annotation = _unwrapped(annotation)
    arguments = typing.get_args(annotation)
    if typing.get_origin(annotation) is tuple and Ellipsis not in arguments:
        if len(arguments) != count:
            raise ValueError(f"{annotation} takes {len(arguments)} entries, got {count}")
        return list(arguments)
    element = next((a for a in reversed(arguments) if a is not Ellipsis), None)
    return [element] * count


def wants_tuple(annotation: Annotation) -> bool:
    """Return whether the annotation declares an immutable sequence. A record's
    list is rebuilt as a tuple so the frozen value stays hashable; `list`
    and `MutableSequence` keep their list."""
    annotation = _unwrapped(annotation)
    return (annotation in (tuple, Sequence)
            or typing.get_origin(annotation) in (tuple, Sequence))


def from_record[ValueT](annotation: type[ValueT], value: Configured, *, dtypes: bool = True) -> ValueT:
    """Return `value` as the class `annotation` names, from a record or already one.

    The class is the witness: what comes back is an instance of it or a
    `ValueError` naming what the record built instead, so a caller reads a
    value of the type it asked for rather than one it has to narrow again. A
    container annotation, `tuple[ParamGroup, ...]` or `Mapping[str, ...]`,
    builds every entry and witnesses the container.
    `dtypes` is the one policy the two readers differ in: a module field
    takes a `dtype` as the dtype its name says (True), and a run record
    keeps the name it wrote (`RunConfig.from_dict`, False).
    """
    with _reading(value):
        built = _rebuilt(annotation, value, dtypes=dtypes)
    # A union is its own witness: the value is any one of its members.
    union = typing.get_origin(annotation) in (Union, types.UnionType)
    witness = annotation if union else typing.get_origin(annotation) or annotation
    if not isinstance(built, witness):
        declared = annotation if union else witness.__name__
        raise ValueError(f"{value!r} builds {type(built).__name__}, not the {declared} the field declares")
    return built


def to_record(value, annotation) -> JSON:
    """Return `value` as the record `from_record(annotation, ...)` rebuilds it from.

    The record is JSON: a dict, a list, or a scalar json.dump can write. A
    dataclass is the record of its fields where `annotation` names its class
    exactly, and `{"class": <import path>, "fields": {...}}` where the field
    declares a base, a union or nothing, so the reader knows what to build;
    a function is `{"function": <import path>}`. A Flax module held twice in
    one record, as `nn.Sequential([dense, dense])` holds its one shared layer,
    is written once as a class record with `"share": <n>`, and each later
    place that holds it as `{"shared": <n>}`, so the reader builds one module
    and the model keeps one set of parameters for it (`recording`).
    """
    from flax import linen as nn

    from dew.records import recorded_duration

    if isinstance(value, datetime.timedelta):
        return recorded_duration(value)
    if isinstance(value, type) and value.__module__ in ('jax.numpy', 'numpy', 'ml_dtypes'):
        return dtype_name(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        with recording() as written:
            if isinstance(value, nn.Module) and id(value) in written:
                return _referred(written, written[id(value)])
            fields = record_fields(value, type(value))
            record = (fields if type(value) is _unwrapped(annotation)
                      else {"class": import_path(type(value)), "fields": fields})
            if isinstance(value, nn.Module):
                written[id(value)] = (value, record)
            return record
    if isinstance(value, (types.FunctionType, types.BuiltinFunctionType, types.MethodType)):
        return {"function": import_path(value)}
    if isinstance(value, (list, tuple)):
        entries = entry_types(annotation, len(value))
        return [to_record(entry_value, entry)
                for entry_value, entry in zip(value, entries, strict=True)]
    if isinstance(value, Mapping):
        entries = entry_types(annotation, len(value))
        keys = _key_type(annotation)
        return {_record_key(key, keys): to_record(entry_value, entry)
                for (key, entry_value), entry in zip(value.items(), entries, strict=True)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        f"{type(value).__name__} is not something a run record can carry; a "
        f"config field holds JSON scalars, sequences, mappings, dataclasses and "
        f"functions with an import path")


def _record_key(key: object, declared: Annotation) -> str:
    """Return one mapping key as JSON names it, since JSON has only string keys.

    A name is itself, a number its Python spelling, and a tree path its parts
    joined the way every message in this tree joins them,
    `params/layers_0/self_attn/q_proj`. The mapping's declared key type reads
    the spelling back (`_key`), so a key it would not read back as the same
    value of the same type, such as a number under an undeclared key type, a
    NaN, or a path part holding `/`, is refused here rather than written.
    """
    if isinstance(key, tuple):
        written = "/".join(_record_key(part, None) if isinstance(part, str) else str(part) for part in key)
    elif isinstance(key, (str, int, float)):
        written = str(key)
    else:
        raise TypeError(
            f"{key!r} is not a key a run record can carry; a config mapping is keyed "
            f"by a name, a number, or a tree path of them")
    try:
        back = _key(declared, written)
    except ValueError:
        back = written
    if not _same_key(back, key):
        raise ValueError(f"the key {key!r} would read back as {back!r} under the mapping's key type "
                         f"{declared}; declare the key type, or key it by a name")
    return written


def _same_key(read: RecordKey, key: RecordKey) -> bool:
    """Whether a key read back is the key written: equal, and of the same type
    part by part."""
    if isinstance(key, tuple):
        return (isinstance(read, tuple) and len(read) == len(key)
                and all(_same_key(a, b) for a, b in zip(read, key, strict=True)))
    return type(read) is type(key) and read == key


def _key_type(annotation: Annotation) -> Annotation:
    """A mapping annotation's declared key type; None where it declares none."""
    annotation = _unwrapped(annotation)
    return typing.get_args(annotation)[0] if typing.get_origin(annotation) in _MAPPINGS else None


_MAPPINGS = (dict, Mapping, MutableMapping)


def _rebuilt(annotation: Annotation, value: object, *, dtypes: bool, name: str = "") -> Configured:
    """Return `value` as its annotation asks for it: a record becomes the value it
    describes, and anything already built is left alone.

    This is the one walk from a record to a value, for a module field and a
    run record alike. A class record, `{"class": ..., "fields": {...}}`
    (`to_record` writes it), builds the class it names, which must be the
    field's class, derive from it, belong to its union, or be a function
    declared to return one of those; a function record is the function; any
    other mapping is the record of the field's own dataclass. Containers are
    walked, so a mapping of records and a tuple of records build their values
    too, a JSON list becomes the tuple a field declares, and a mapping's key
    becomes the tuple path its key type declares. `dtypes` is as
    `from_record` reads it, for a field or entry `name`d `dtype`.
    """
    if dtypes and name == "dtype":
        return resolve_dtype(value)
    annotation = resolve_alias(annotation)
    if isinstance(value, Mapping) and set(value) == {"function"} and isinstance(value["function"], str):
        return imported(value["function"])
    if (isinstance(value, Mapping) and annotation not in (None, object)
            and typing.get_origin(annotation) not in _MAPPINGS and _sharing(value) is not None):
        return _shared(annotation, value, dtypes=dtypes)
    if typing.get_origin(annotation) in (Union, types.UnionType):
        if value is None:
            return value
        inner = [member for member in typing.get_args(annotation) if member is not type(None)]
        if len(inner) == 1:
            return _rebuilt(inner[0], value, dtypes=dtypes)
        classes = tuple(member for member in inner if isinstance(member, type))
        named = _class_record(value) if isinstance(value, Mapping) else None
        if named is not None:
            if not classes:
                raise ValueError(f"{annotation} declares no class, so nothing builds the record {value!r}")
            return _built(_member(named[0], classes), named[1], dtypes=dtypes)
        return configured(value)
    if isinstance(value, Mapping):
        if typing.get_origin(annotation) is Callable or annotation is Callable:
            # A callable field, such as a Sequential's layers or a loss, takes
            # a record of a class whose instances are called, whatever class.
            named = _class_record(value)
            if named is not None:
                member = _member(named[0], (object,))
                if not _owned(member, called=True):
                    raise ValueError(
                        f"{named[0]!r} names {member!r}; a callable field's class record names a class "
                        f"whose instances are called, of Dew's or of a trusted package (trust=, --trust), "
                        f"and a function is the record {{'function': ...}}")
                return _built(member, named[1], dtypes=dtypes)
        if isinstance(annotation, type) and annotation is not object:
            # A field typed with a class takes a record of the class itself or
            # of any class derived from it or function returning it; `object`
            # declares nothing, and its record stays the record it is.
            named = _class_record(value)
            if named is not None:
                return _built(_member(named[0], (annotation,)), named[1], dtypes=dtypes)
            if dataclasses.is_dataclass(annotation):
                return _construct(annotation, value, dtypes=dtypes)
        if typing.get_origin(annotation) in _MAPPINGS:
            keys, entries = typing.get_args(annotation)
            return {_key(keys, str(name)): _rebuilt(entries, entry, dtypes=dtypes, name=str(name))
                    for name, entry in value.items()}
        # A record with no value class behind it, such as one of the unets'
        # per-stage attention settings: entries are walked.
        return {name: _rebuilt(entry, record, dtypes=dtypes, name=str(name))
                for entry, (name, record) in zip(entry_types(annotation, len(value)), value.items(),
                                                 strict=True)}
    if isinstance(value, (list, tuple)):
        rebuilt = [_rebuilt(entry, record, dtypes=dtypes)
                   for entry, record in zip(entry_types(annotation, len(value)), value, strict=True)]
        return tuple(rebuilt) if wants_tuple(annotation) else type(value)(rebuilt)
    return configured(value)


def _class_record(value: Mapping[str, object] | Mapping[RecordKey, object]
                  ) -> tuple[str, Mapping[str, object]] | None:
    """The class a class record, `{"class": ..., "fields": {...}}`, names and
    its fields (a record with no fields may leave them out); None for a
    mapping that is not one."""
    name, fields = value.get("class"), value.get("fields", {})
    if set(value) <= {"class", "fields"} and isinstance(name, str) and isinstance(fields, Mapping):
        return name, fields
    return None


type _Written = dict[int, tuple[nn.Module, dict[str, JSON]]]

_WRITTEN: contextvars.ContextVar[_Written | None] = contextvars.ContextVar("_WRITTEN", default=None)
"""The Flax modules the record being written holds, by identity, each with its record."""


@contextlib.contextmanager
def recording() -> Iterator[_Written]:
    """One record being written: a module held twice in it is written once
    and referred to after (`to_record`). The outermost call opens the record;
    one inside it writes into the same record."""
    held = _WRITTEN.get()
    if held is not None:
        yield held
        return
    opened: _Written = {}
    token = _WRITTEN.set(opened)
    try:
        yield opened
    finally:
        _WRITTEN.reset(token)


def _referred(written: _Written, first: tuple[nn.Module, dict[str, JSON]]) -> JSON:
    """The reference to a module the record already holds: its first record
    becomes a class record numbered `share`, in place, and the reference
    names that number."""
    module, record = first
    if "share" not in record:
        if "class" not in record:
            fields = dict(record)
            record.clear()
            record.update({"class": import_path(type(module)), "fields": fields})
        record["share"] = sum("share" in held for _, held in written.values())
    return {"shared": record["share"]}


@dataclasses.dataclass
class _Shared:
    """The shared modules of the record being read: each one's defining
    record, by number, and the module once it is built."""

    defined: dict[int, Mapping[str, object]]
    built: dict[int, Configured] = dataclasses.field(default_factory=dict)


_READING: contextvars.ContextVar[_Shared | None] = contextvars.ContextVar("_READING", default=None)


@contextlib.contextmanager
def _reading(record: Configured | Mapping[str, object]) -> Iterator[None]:
    """One record being read: every module it shares (`to_record`) is built
    once, where the record first holds it or refers to it. The outermost call
    opens the record; one inside it reads the same record."""
    if _READING.get() is not None:
        yield
        return
    defined: dict[int, Mapping[str, object]] = {}

    def scan(node) -> None:  # any value a record holds, a record of records or not
        if isinstance(node, Mapping):
            number = _sharing(node)
            if number is not None and "share" in node:
                if number in defined:
                    raise ValueError(f"the record defines its shared module {number} twice")
                defined[number] = {str(key): entry for key, entry in node.items() if key != "share"}
            for entry in node.values():
                scan(entry)
        elif isinstance(node, (list, tuple)):
            for entry in node:
                scan(entry)

    scan(record)
    token = _READING.set(_Shared(defined))
    try:
        yield
    finally:
        _READING.reset(token)


def _sharing(value: Mapping[str, object] | Mapping[RecordKey, object]) -> int | None:
    """The number of the shared module `value` defines, a class record with
    `"share": n`, or refers to, `{"shared": n}`; None for any other record."""
    keys = set(value)
    number = (value.get("shared") if keys == {"shared"} else
              value.get("share") if {"class", "share"} <= keys <= {"class", "fields", "share"} else None)
    return number if isinstance(number, int) and not isinstance(number, bool) else None


def _shared(annotation: Annotation, value: Mapping[str, object], *, dtypes: bool) -> Configured:
    """The one module a shared record (`{"share": n, ...}`) or a reference to
    it (`{"shared": n}`) stands for, built the first time either is read."""
    shared = _READING.get()
    number = _sharing(value)
    if shared is None or number is None:
        raise ValueError(f"{value!r} is no shared module of a record being read")
    if number not in shared.built:
        if number not in shared.defined:
            raise ValueError(f"the record refers to its shared module {number} and defines none")
        shared.built[number] = _rebuilt(annotation, shared.defined[number], dtypes=dtypes)
    return shared.built[number]


def _member(name: str, held: tuple[type, ...]) -> Callable[..., Configured]:
    """The class or function a record's `class` names, by import path or by
    an alias of any kind, that is one of `held`, derives from one, or is a
    function declared to return one."""
    def fits(member: Callable[..., Configured]) -> bool:
        if isinstance(member, type):
            return any(_derives(member, base) for base in held)
        return callable(member) and any(_returns(member, base) for base in held)

    found = ([imported(name)] if ":" in name else
             [imported(kind.paths[name]) for kind in KINDS if name in kind.paths])
    fitting = [member for member in found if fits(member)]
    if len(fitting) == 1:
        return fitting[0]
    wanted = " or ".join(base.__name__ for base in held)
    if not found:
        raise ValueError(f"no class is named {name!r}; a record names one by an alias or "
                         f"its import path, `module:Class`")
    raise ValueError(f"{name!r} names {', '.join(map(repr, found))}, not one {wanted}")


def _built(member: Callable[..., Configured], fields: Mapping[str, object], *, dtypes: bool) -> Configured:
    """What `member` builds from a record of its fields: a dataclass's fields
    as it declares them, a function's or another class's as its parameters."""
    held = _record_class(member)
    if held is not None:
        return _construct(held, fields, dtypes=dtypes)
    return configured(member(**_arguments(member, fields, dtypes=dtypes)))


def _key(annotation: Annotation, key: str) -> RecordKey:
    """One record key as the mapping declares its keys: a name, a number, a
    literal, or the tuple path `_record_key` joined with `/`, each part read
    by its own declared type. A spelling the type does not read is refused."""
    annotation = resolve_alias(annotation)
    if wants_tuple(annotation):
        parts = key.split("/")
        read: list[str | int | float] = []
        for part_type, part in zip(entry_types(annotation, len(parts)), parts, strict=True):
            value = _key(part_type, part)
            if isinstance(value, tuple):
                raise ValueError(f"the record key {key!r} nests a path inside a path")
            read.append(value)
        return tuple(read)
    if typing.get_origin(annotation) in (Union, types.UnionType):
        # An int spelling reads as an int before a float or a string reads it.
        for member in sorted(typing.get_args(annotation),
                             key=lambda member: (member is str, member is float)):
            try:
                return _key(member, key)
            except ValueError:
                continue
        raise ValueError(f"the record key {key!r} is none of {annotation}")
    if typing.get_origin(annotation) is Literal:
        held = [member for member in typing.get_args(annotation) if str(member) == key]
        if not held:
            raise ValueError(f"the record key {key!r} is none of {annotation}")
        return held[0]
    try:
        if annotation is int:
            return int(key)
        if annotation is float:
            return float(key)
    except ValueError:
        raise ValueError(f"the record key {key!r} is not the {annotation} its mapping declares") from None
    if annotation is type(None):
        raise ValueError(f"the record key {key!r} is not None")
    return key


def _owned(member: Callable[..., Configured], *, called: bool) -> bool:
    """Whether a record may construct `member` where no base class holds it:
    a class of Dew's or of a trusted package (`import_trusted`), and for a
    `called` field one whose instances are called. Anything imported can be
    named, and a function, or a class that does its work in its constructor
    as `subprocess.Popen` does, is refused before anything runs."""
    package = member.__module__.split(".")[0]
    return (isinstance(member, type) and (package == "dew" or package in _TRUSTED)
            and (not called or any("__call__" in vars(base) for base in member.__mro__[:-1])))


def _record_class(member: Callable[..., Configured]) -> type | None:
    """`member` when it is a dataclass, whose fields a record names; None for
    a function or a plain class, which declares no fields to narrow."""
    return member if isinstance(member, type) and dataclasses.is_dataclass(member) else None


def _construct(member: type, fields: Mapping[str, object], *, dtypes: bool) -> Configured:
    """Build the dataclass `member` from a record of its fields."""
    return configured(member(**_declared(member, fields, dtypes=dtypes)))


def _declared(member: type, fields: Mapping[str, object], *, dtypes: bool) -> Mapping[str, object]:
    """The record's fields as `member` declares them, each walked against its
    own annotation. A field the record lacks takes its declared default; a
    field `member` does not declare, or a required one the record lacks,
    raises. A field marked `metadata={"record": False}` is a binding the
    value picks up at runtime, not something a record carries."""
    declared = [f for f in dataclasses.fields(member) if _recorded(f)]
    names = sorted(f.name for f in declared)
    unknown = sorted(set(fields) - set(names))
    missing = [f.name for f in declared if f.name not in fields
               and f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING]
    if unknown or missing:
        raise ValueError(f"{member.__name__} does not match the record: unknown fields {unknown}, "
                         f"missing fields {missing}; its fields are {names}")
    return {name: _read(member, name, value, dtypes=dtypes) for name, value in fields.items()}


def _arguments(function: Callable[..., Configured], fields: Mapping[str, object], *,
               dtypes: bool) -> Mapping[str, object]:
    """The record's fields as the parameters of the registered `function`, or
    of a class's constructor (`parameters`), each walked against its own
    annotation (`_read`), as `_declared` walks a dataclass's. A field the
    function does not take, or a parameter without a default that the record
    lacks, raises; a function taking `**kwargs` takes any field."""
    named, open_ended = parameters(function)
    unknown = [] if open_ended else sorted(set(fields) - set(named))
    missing = [name for name, (parameter, _) in named.items()
               if name not in fields and parameter.default is parameter.empty]
    if unknown or missing:
        raise ValueError(f"{function.__name__} does not match the record: unknown fields {unknown}, "
                         f"missing fields {missing}; its parameters are {sorted(named)}")
    return {name: _read(function, name, value, dtypes=dtypes) for name, value in fields.items()}


def read_arguments(member: type | Callable[..., Configured], record: Mapping[str, object]
                   ) -> Mapping[str, object]:
    """`record`'s fields of the dataclass `member`, or arguments of the
    function or constructor `member`, each read as `build` reads it, without
    asking whether they are all that `member` takes."""
    with _reading(record):
        return {name: _read(member, name, value, dtypes=True) for name, value in record.items()}


def _read(member: type | Callable[..., Configured], name: str, value, *, dtypes: bool):
    """One field of the dataclass `member`, walked against its own annotation;
    or one argument of the function or constructor `member`, a record or a
    JSON list read against its parameter's annotation and a dtype's name the
    dtype, any other value, a scalar or a value the caller built, as given."""
    held = _record_class(member)
    if held is not None:
        return _rebuilt(_declared_type(held, name), configured(value), dtypes=dtypes, name=name)
    if not (isinstance(value, (Mapping, list)) or (dtypes and name == "dtype")):
        return value
    named = parameters(member)[0]
    return _rebuilt(_parameter_type(named[name][1], name) if name in named else None, value,
                    dtypes=dtypes, name=name)


def argument_records(member: type | Callable[..., Configured], fields: Mapping[str, object]
                     ) -> dict[str, JSON]:
    """`fields` as records of `member`'s parameters (`parameters`): each read
    against its parameter's annotation and written back, so an alias becomes
    its import path, a value already built becomes its record, and an
    argument `member` does not take, or a value no record can carry (a
    lambda), is refused here rather than where the run is built."""
    named, open_ended = parameters(member)
    unknown = [] if open_ended else sorted(set(fields) - set(named))
    if unknown:
        raise ValueError(f"{member.__name__} takes no {unknown}; its parameters are {sorted(named)}")
    records = {}
    with _reading(fields), recording():
        for name, value in fields.items():
            annotation = _parameter_type(named[name][1], name) if name in named else None
            built = (_rebuilt(annotation, value, dtypes=False, name=name)
                     if isinstance(value, (Mapping, list)) else value)
            records[name] = to_record(built, annotation)
    return records


type Parameters = dict[str, tuple[inspect.Parameter, Callable[..., Configured]]]


def parameters(member: type | Callable[..., Configured]) -> tuple[Parameters, bool]:
    """`member`'s named parameters, each with the function that declares it,
    and whether it takes any other keyword besides.

    A class's parameters are its constructor's, and a constructor that passes
    `**kwargs` on to its base's takes the base's too, as a few-step
    diffusion objective takes every `DiffusionObjective` argument. A subclass
    that names an argument again declares its own default for it."""
    owners = ([vars(owner)["__init__"] for owner in member.__mro__[:-1] if "__init__" in vars(owner)]
              if isinstance(member, type) else [member])
    named: Parameters = {}
    for owner in owners:
        held = list(inspect.signature(owner).parameters.values())[1 if isinstance(member, type) else 0:]
        for parameter in held:
            if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY):
                named.setdefault(parameter.name, (parameter, owner))
        if not any(parameter.kind is parameter.VAR_KEYWORD for parameter in held):
            return named, False
    return named, bool(owners)


def _parameter_type(function: Callable[..., Configured], name: str) -> Annotation:
    """Resolve the annotation of one parameter of `function`, or its return
    for `name="return"`, without evaluating the others. None where it has no
    annotation. An annotation naming what the function's module does not
    import at runtime raises, since a record cannot be read against it."""
    # A classmethod a record names, `Class.reader`, is a bound method; its
    # annotations and module are its function's.
    underlying = function.__func__ if isinstance(function, types.MethodType) else function
    if not isinstance(underlying, types.FunctionType):
        return None
    annotations = get_annotations(underlying, format=Format.FORWARDREF)
    if name not in annotations:
        return None
    selected = types.SimpleNamespace(__annotations__={name: annotations[name]})
    try:
        return typing.get_type_hints(selected, globalns=dict(underlying.__globals__))[name]
    except NameError as error:
        raise ValueError(f"{function.__name__}'s annotation of {name} names what {underlying.__module__} "
                         f"does not import at runtime ({error}); a record is read against it, so "
                         f"import it there") from error


def _returns(member: Callable[..., Configured], held: type) -> bool:
    """Whether the function `member` is declared to return `held` or a class
    derived from it."""
    returned = _parameter_type(member, "return")
    return isinstance(returned, type) and _derives(returned, held)


def _derives(member: type, base: type) -> bool:
    """Whether `member` is `base` or derives from it. A protocol declares an
    interface and no class to derive from, so a record's class for one is a
    class of Dew's or of a trusted package (`_owned`), as a protocol kind's
    aliases are."""
    return _owned(member, called=False) if is_protocol(base) else issubclass(member, base)


def record_fields(value: DataclassInstance, owner: type) -> dict[str, JSON]:
    """The record of a dataclass value's constructor fields, each as `owner`
    declares it, at every level of a nested value alike.

    A field still holding its declared callable default, such as a Flax
    layer's initializer closure, is left out: no record can name a closure,
    and the class supplies the same default again when the record is read."""
    from flax import linen as nn

    fields = {}
    with recording():
        for field in dataclasses.fields(value):
            held = getattr(value, field.name)
            if (_recorded(field) and not (isinstance(value, nn.Module) and field.name in ('parent', 'name'))
                    and not (callable(held) and held is field.default)):
                fields[field.name] = to_record(held, _declared_type(owner, field.name))
    return fields


def _recorded(field: dataclasses.Field) -> bool:
    """Return whether a field is part of a record: one the constructor takes,
    not marked `metadata={"record": False}`."""
    return field.init and field.metadata.get("record", True)


def configured(value: object) -> Configured:
    """Return one value a record carried, as a field holds it.

    Nothing is converted here: this is the one place that says what a field
    can carry at all, so a value no member could take is refused where the
    record was read instead of inside the module that would have used it.
    """
    if value is None or isinstance(value, (bool, int, float, str, bytes, type, Enum, np.dtype)):
        return value
    if isinstance(value, (np.ndarray, np.generic, jax.Array)):
        return value
    if dataclasses.is_dataclass(value) or isinstance(value, (Mapping, Sequence)) or callable(value):
        return value
    raise ValueError(f"{value!r} is not a value a member field carries")


DtypeName = Literal["float32", "bfloat16", "float16"]
_DTYPES: dict[DtypeName, DTypeLike] = {
    "float32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.float16}


def resolve_dtype(value: object) -> DTypeLike | None:
    """Resolve a dtype for a module field: a jnp dtype, one of its names, or None.

    Every field named `dtype` is read here, wherever it arrives from, so a
    dtype passes through and a name becomes the dtype it names. Anything
    else is refused here rather than inside a module's first cast.
    """
    if value is None:
        return None
    if isinstance(value, str):
        for name, dtype in _DTYPES.items():
            if value == name:
                return dtype
        raise ValueError(f"dtype {value!r} is not one of {sorted(_DTYPES)}")
    if isinstance(value, (type, np.dtype)):
        return value
    raise ValueError(
        f"dtype {value!r} is not a dtype, nor one of {sorted(_DTYPES)}")


@overload
def dtype_name(value: DTypeLike) -> DtypeName: ...
@overload
def dtype_name(value: None) -> None: ...
def dtype_name(value: DTypeLike | None) -> DtypeName | None:
    """Return the name `resolve_dtype` accepts for a dtype, for a logged config.

    A loader takes a dtype, `jnp.bfloat16`, or its name, and records the name."""
    if value is None:
        return None
    for name, dtype in _DTYPES.items():
        if jnp.dtype(value) == jnp.dtype(dtype):
            return name
    raise ValueError(f"{value!r} is not a dtype a config can name")


def with_dtype(name: str, fields: Mapping[str, object], dtype: str | None) -> Mapping[str, object]:
    """`fields`, the model `name` builds from, computing in `dtype`.

    The dtype is written into the model's own `dtype` field, or, for a
    composite that declares none (DiffusionGemma around its text decoder),
    into each model part it holds that declares one, as a record or built.
    A model with neither refuses a dtype. None leaves `fields` as they are."""
    from flax import linen as nn

    if dtype is None:
        return fields
    member = models[name]
    if isinstance(member, type) and "dtype" in {field.name for field in dataclasses.fields(member)}:
        return {**fields, "dtype": dtype}
    assert isinstance(member, type)
    parts = {}
    for key, value in fields.items():
        part = _part_type(member, key, configured(value))
        if part is None or "dtype" not in {field.name for field in dataclasses.fields(part)}:
            continue
        if isinstance(value, nn.Module):
            parts[key] = value.clone(dtype=resolve_dtype(dtype))
        elif isinstance(value, Mapping) and (named := _class_record(value)) is not None:
            parts[key] = {"class": named[0], "fields": {**named[1], "dtype": dtype}}
        elif isinstance(value, Mapping):
            parts[key] = {**value, "dtype": dtype}
    if not parts:
        raise ValueError(f"{member.__qualname__} declares no dtype field and holds no part that does, so "
                         f"it cannot compute in {dtype}")
    return {**fields, **parts}


def _part_type(member: type, field: str, value: Configured) -> type | None:
    """The model class a composite's `field` holds, built, as a class record,
    or as the record of the class the field declares; None for anything else."""
    from flax import linen as nn

    if isinstance(value, nn.Module):
        return type(value)
    named = _class_record(value) if isinstance(value, Mapping) else None
    if named is not None:
        held = _member(named[0], (nn.Module,))
        return held if isinstance(held, type) else None
    declared = _unwrapped(_declared_type(member, field))
    if isinstance(value, Mapping) and isinstance(declared, type) and issubclass(declared, nn.Module):
        return declared
    return None


# Each kind's aliases, for the command line and the records a person writes.
models: Aliases[type[nn.Module], nn.Module] = Aliases("model", {
    "causal_transformer": "dew.nn.backbones.causal_transformer:CausalTransformer",
    "diffusion_gemma": "dew.nn.diffusion_gemma:DiffusionGemma",
    "edm2_unet": "dew.nn.backbones.edm2:EDM2UNet",
    "flux2_transformer": "dew.nn.backbones.flux2:Flux2Transformer",
    "flux_transformer": "dew.nn.backbones.flux:FluxTransformer",
    "hierarchical_mmdit": "dew.nn.backbones.mmdit:HierarchicalMMDiT",
    "hybrid_dit": "dew.nn.backbones.ssm_dit:HybridSSMAttentionDiT",
    "jepa_encoder": "dew.nn.backbones.jepa:JepaEncoder",
    "jepa_predictor": "dew.nn.backbones.jepa:JepaPredictor",
    "jepa_video_encoder": "dew.nn.backbones.jepa:JepaVideoEncoder",
    "multimodal_transformer": "dew.nn.multimodal:MultimodalTransformer",
    "qwen_image_transformer": "dew.nn.backbones.qwen_image:QwenImageTransformer",
    "sd3_transformer": "dew.nn.backbones.sd3:SD3Transformer",
    "simple_dit": "dew.nn.backbones.dit:SimpleDiT",
    "simple_mmdit": "dew.nn.backbones.mmdit:SimpleMMDiT",
    "simple_udit": "dew.nn.backbones.uvit:SimpleUDiT",
    "unet": "dew.nn.backbones.unet:Unet",
    "unet_2d_condition": "dew.nn.backbones.unet_condition:UNet2DCondition",
    "unet_3d": "dew.nn.backbones.unet3d:UNet3D",
    "uvit": "dew.nn.backbones.uvit:UViT",
    "video_dit": "dew.nn.backbones.video_dit:VideoDiT",
    "wan_transformer": "dew.nn.backbones.wan:WanTransformer",
    "z_image_transformer": "dew.nn.backbones.z_image:ZImageTransformer",
}, base="flax.linen:Module")
presets: Aliases[type[Preset], Preset] = Aliases("preset", {
    "cosine": "dew.diffusion.presets:Cosine",
    "edm": "dew.diffusion.presets:EDM",
    "flow": "dew.diffusion.presets:Flow",
    "jit": "dew.diffusion.presets:JiT",
    "karras": "dew.diffusion.presets:Karras",
    "mdlm": "dew.diffusion.discrete:MDLM",
    "mean_flow": "dew.diffusion.presets:MeanFlow",
    "shortcut": "dew.diffusion.presets:Shortcut",
    "sqrt": "dew.diffusion.presets:Sqrt",
})
solvers: Aliases[type[Solver[Any]], Solver[Any]] = Aliases("solver", {
    "consistency": "dew.sampling.solvers.gaussian:Consistency",
    "ddim": "dew.sampling.solvers.gaussian:DDIM",
    "ddpm": "dew.sampling.solvers.gaussian:DDPM",
    "deis": "dew.sampling.solvers.dpm:DEIS",
    "dpmsolver_multistep": "dew.sampling.solvers.dpm:DPMSolverMultistep",
    "dpmsolver_sde": "dew.sampling.solvers.brownian:DPMSolverSDE",
    "dpmsolver_singlestep": "dew.sampling.solvers.dpm:DPMSolverSinglestep",
    "euler": "dew.sampling.solvers.sigma:Euler",
    "euler_ancestral": "dew.sampling.solvers.sigma:EulerAncestral",
    "flow_sde": "dew.sampling.flow:FlowSDE",
    "heun": "dew.sampling.solvers.sigma:Heun",
    "kdpm2": "dew.sampling.solvers.sigma:KDPM2",
    "lms": "dew.sampling.solvers.sigma:LMS",
    "multistep_dpm": "dew.sampling.solvers.sigma:MultiStepDPM",
    "pndm": "dew.sampling.solvers.gaussian:PNDM",
    "rk4": "dew.sampling.solvers.sigma:RK4",
    "tcd": "dew.sampling.solvers.gaussian:TCD",
    "unipc": "dew.sampling.solvers.unipc:UniPC",
    "unmask": "dew.diffusion.discrete:Unmask",
})
datasets: Aliases[type[DatasetSpec], DatasetSpec] = Aliases("dataset", {
    "array_record_images": "dew.data.images:ArrayRecordImages",
    "chat_messages": "dew.data.chat:ChatMessages",
    "decision_table": "dew.decision.data:DecisionTable",
    "hf": "dew.data.providers:HubDataset",
    "hf_images": "dew.data.images:HFImages",
    "local_videos": "dew.data.video:LocalVideos",
    "online_images": "dew.data.streaming:OnlineImages",
    "online_videos": "dew.data.streaming:OnlineVideos",
    "preference_pairs": "dew.data.preferences:PreferencePairs",
    "prompts": "dew.data.prompts:Prompts",
    "tfds": "dew.data.providers:PreparedTFDS",
    "tfds_images": "dew.data.images:TFDSImages",
    "token_windows": "dew.data.tokens:TokenWindows",
}, base="dew.data.dataset:DatasetSpec")
encoders: Aliases[type[ConditionEncoder[Any]], ConditionEncoder[Any]] = Aliases("encoder", {
    "char_table": "dew.inputs.encoders:CharTable",
    "clip_text": "dew.inputs.encoders:CLIPText",
    "diffusion_text": "dew.inputs.diffusion:DiffusionConditioner",
    "hf_audio": "dew.inputs.encoders:HFAudio",
    "hidden_states_text": "dew.inputs.diffusion:HiddenStatesConditioner",
    "qwen_image_text": "dew.inputs.diffusion:QwenImageConditioner",
    "t5": "dew.inputs.encoders:T5Text",
    "wan_text": "dew.inputs.diffusion:WanConditioner",
}, base="dew.inputs.encoders:ConditionEncoder")
metrics: Aliases[Callable[..., Metric], Metric] = Aliases("metric", {
    "accuracy": "dew.decision.metrics:Accuracy",
    "aurc": "dew.decision.metrics:AURC",
    "brier": "dew.decision.scoring:Brier",
    "clip": "dew.eval.images:CLIPDistance",
    "clip_score": "dew.eval.images:CLIPScore",
    "ece": "dew.decision.metrics:ECE",
    "fid": "dew.eval.fid:FID",
    "knn_probe": "dew.objectives.jepa.probes:KnnProbe",
    "linear_probe": "dew.objectives.jepa.probes:LinearProbe",
    "log_loss": "dew.decision.scoring:LogLoss",
    "lpips": "dew.eval.lpips:LPIPS",
    "perplexity": "dew.objectives.lm.objective:Perplexity",
    "psnr": "dew.eval.psnr:PSNR",
    "rps": "dew.decision.scoring:RankedProbability",
    "spherical": "dew.decision.scoring:Spherical",
    "ssim": "dew.eval.ssim:SSIM",
})
objectives: Aliases[type[Objective], Objective] = Aliases("objective", {
    "block_diffusion": "dew.objectives.diffusion.block:BlockDiffusionObjective",
    "decision": "dew.decision.objective:DecisionObjective",
    "diffusion": "dew.objectives.diffusion.objective:DiffusionObjective",
    "distillation": "dew.objectives.distillation:DistillationObjective",
    "dpo": "dew.objectives.rl.preference:DPOObjective",
    "flow_grpo": "dew.objectives.rl.flow:FlowGRPOObjective",
    "grpo": "dew.objectives.rl.grpo:GRPOObjective",
    "guidance_distillation": "dew.objectives.diffusion.guidance_distillation:GuidanceDistillationObjective",
    "jepa": "dew.objectives.jepa.objective:JepaObjective",
    "ladd": "dew.objectives.diffusion.adversarial:AdversarialDistillationObjective",
    "lm": "dew.objectives.lm.objective:LMObjective",
    "masked_diffusion": "dew.objectives.diffusion.masked:MaskedDiffusionObjective",
    "mean_flow": "dew.objectives.diffusion.few_step:MeanFlowObjective",
    "ppo": "dew.objectives.rl.ppo:PPOObjective",
    "rcm": "dew.objectives.diffusion.consistency:ConsistencyDistillationObjective",
    "shortcut": "dew.objectives.diffusion.few_step:ShortcutObjective",
    "supervised": "dew.objectives.supervised:Supervised",
}, base="dew.objectives.base:Objective")
mixers: Aliases[type[MixerBase], MixerBase] = Aliases("mixer", {
    "attention": "dew.nn.mixers.attention:AttentionMixer",
    "deepseek_v4": "dew.nn.deepseek_v4:DeepseekV4Mixer",
    "gated_delta_net": "dew.nn.mixers.gated_delta_net:GatedDeltaNetMixer",
    "kimi_delta_attention": "dew.nn.kda:KimiDeltaAttentionMixer",
    "kpool_sparse_attention": "dew.nn.dsa_kpool:KPoolSparseAttentionMixer",
    "llama4": "dew.nn.llama4:Llama4Mixer",
    "mamba2": "dew.nn.mixers.mamba2:Mamba2Mixer",
    "mla": "dew.nn.mla:MLAMixer",
    "mlp": "dew.nn.mixers.mlp:MLPMixer",
}, base="dew.nn.mixer_base:MixerBase")
towers: Aliases[type[TowerBase], TowerBase] = Aliases("tower", {
    "deepseek_v41": "dew.nn.vision.deepseek_v41:DeepseekV41Vision",
    "gemma3n": "dew.nn.vision.gemma3n:Gemma3nVision",
    "gemma3n_audio": "dew.nn.audio:Gemma3nAudio",
    "gemma4": "dew.nn.vision.gemma4:Gemma4Vision",
    "gemma4_audio": "dew.nn.audio:Gemma4Audio",
    "llama4": "dew.nn.vision.llama4:Llama4Vision",
    "qwen3_5": "dew.nn.vision.qwen35:Qwen35Vision",
    "siglip": "dew.nn.vision.siglip:SiglipVision",
}, base="dew.nn.vision.common:TowerBase")
projectors: Aliases[type[ProjectorBase], ProjectorBase] = Aliases("projector", {
    "deepseek_v41": "dew.nn.vision.deepseek_v41:DeepseekV41Projector",
    "gemma": "dew.nn.vision.siglip:GemmaProjector",
    "gemma3n": "dew.nn.vision.gemma3n:Gemma3nProjector",
    "gemma4": "dew.nn.vision.gemma4:Gemma4Projector",
    "llama4": "dew.nn.vision.llama4:Llama4Projector",
    "qwen3_5": "dew.nn.vision.qwen35:Qwen35Projector",
}, base="dew.nn.vision.common:ProjectorBase")
schedules: Aliases[type[ScheduleBase], ScheduleBase] = Aliases("schedule", {
    "cosine": "dew.training.optim:Cosine",
    "exponential": "dew.training.optim:Exponential",
    "linear": "dew.training.optim:Linear",
    "one_cycle": "dew.training.optim:OneCycle",
    "power": "dew.training.optim:Power",
}, base="dew.training.optim:ScheduleBase")


KINDS: tuple[Aliases, ...] = (models, presets, solvers, datasets, encoders, metrics, objectives, mixers,
                              towers, projectors, schedules)
"""Every kind, which a record's alias is looked up across."""

__all__ = [
    "KINDS",
    "Aliases",
    "Record",
    "argument_records",
    "datasets",
    "dtype_name",
    "encoders",
    "from_record",
    "import_path",
    "import_trusted",
    "imported",
    "metrics",
    "mixers",
    "models",
    "objectives",
    "parameters",
    "presets",
    "projectors",
    "record_fields",
    "schedules",
    "solvers",
    "to_record",
    "towers",
    "with_dtype",
]
