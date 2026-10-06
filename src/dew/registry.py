"""Registries that name the things a run is made of.

There is one `Registry` per kind, including model components such as mixers,
towers and projectors. Config files, the CLI and run records refer to a member
by name, and its registry maps that name to the class and back, so
`models["simple_dit"]` is `SimpleDiT`. Python code builds the class directly. An
unknown name or field raises an error.

The registries are empty at import. Each member registers itself where it is
defined, so importing a package fills its tables, and this module imports none
of them. When a lookup asks for a name its table does not hold yet, the registry
imports the Dew modules whose decorator registers that name, found by reading
the sources. So a record loads in a process that imported nothing beforehand,
and no other module is imported.

A package outside Dew registers its members the same way, and lists a module
whose import registers them under the `dew.plugins` entry-point group:

    [project.entry-points."dew.plugins"]
    sparx = "sparx"

The plugins are loaded only for a name that Dew's own sources do not register.
Every entry is then imported once (`entry.load()`), and the lookup tries again.
An entry that fails to import is reported only in the error of a lookup that
still misses after that. A plugin can also add a kind of its own. It creates a
`Registry` and `share`s it in the module that defines the kind's base class, so
a record whose field declares that class rebuilds the member.
"""

from __future__ import annotations

import dataclasses
import datetime
import functools
import importlib
import importlib.metadata
import inspect
import operator
import re
import sys
import types
import typing
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict, TypeVar, Union, overload

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike
from typing_extensions import Format, get_annotations

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
    from dew.objectives.diffusion.objective import Training
    from dew.sampling.solvers import Solver
    from dew.training.optim import ScheduleBase

M = TypeVar("M", bound=Callable[..., Any])
"""What calling a member builds: the module a model name builds, the spec a
dataset name builds. A registry is generic over both, since the table holds
the callable and `build` hands back what it returned."""

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
                   | Mapping[str | tuple[str, ...], object] | Sequence[Configured])

# `build` called with no record at all, which is every caller that writes its
# fields as keywords. Shared because it is read and never written.
NO_RECORD: Mapping[str, object] = types.MappingProxyType({})


_DECORATOR = re.compile(r"""^[ \t]*@(?:dew\.)?(?:registry\.)?(\w+)\(\s*["']([^"']+)["']\s*\)[ \t]*$""", re.M)
"""A registration as Dew's sources write it, `@models("simple_dit")` or
`@registry.objectives("lm")`: the registry's name in this module and the
member's."""


@functools.cache
def _registering_modules() -> Mapping[tuple[str, str], tuple[str, ...]]:
    """Each `(registry, name)` Dew's sources register, and the modules that do.

    The decorators are the one statement of what registers where, so this
    reads them rather than keeping a second table: about 270 files in 20 ms,
    once a process, and only when a lookup misses."""
    root = Path(__file__).parent
    found: dict[tuple[str, str], list[str]] = {}
    for path in sorted(root.rglob("*.py")):
        parts = path.relative_to(root.parent).with_suffix("").parts
        module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        for attribute, name in _DECORATOR.findall(path.read_text()):
            found.setdefault((attribute, name), []).append(module)
    return {key: tuple(modules) for key, modules in found.items()}


PLUGINS = "dew.plugins"
"""The entry-point group under which a package outside Dew lists the module that registers its members."""


@functools.cache
def _loaded_plugins() -> tuple[tuple[str, BaseException], ...]:
    """Import every `dew.plugins` entry, once a process, and return those
    that failed, each with its exception. Importing an entry is what
    registers its members; nothing of a plugin is read without importing it."""
    failed = []
    for entry in sorted(importlib.metadata.entry_points(group=PLUGINS), key=lambda entry: entry.name):
        try:
            entry.load()
        except Exception as error:
            # Reported by the lookup that needed the entry, and no other.
            failed.append((f"{entry.name} = {entry.value}", error))
    return tuple(failed)


_SHARED: list[Registry] = []
"""Every table a record may name a member of, in the order shared; written
only by `Registry.share`. The tables hold different kinds, so this names
none: a reader asks each table for a member and tests what it got."""


class Registry[T: Callable[..., Any], Built](Mapping[str, T]):
    """Holds the members of one kind by name, and registers them as a decorator.

    `registry[name]` returns a member. If the table does not hold the name yet,
    it first imports the Dew modules that register it, and then the plugins. An
    unknown name raises KeyError listing the known ones.

    Records can name members only of a shared table (`share`). Dew's twelve
    tables are shared, and a plugin shares a kind of its own the same way.
    """

    def __init__(self, kind: str):
        self.kind = kind
        # A decorator has no base class to test the member against, and it
        # hands back the class it decorated so a caller's checker keeps the
        # concrete type (`DiffusionObjective`, not `Objective`). The table is
        # untyped here and typed on the way out.
        self._members: dict[str, Any] = {}

    def __call__(self, name: str, /) -> Callable[[M], M]:
        """Return a decorator that registers a class or function under `name`, as in `@models("simple_dit")`.

        The decorator returns the member unchanged. Raises TypeError for a name
        that is not a non-empty string, and the decorator raises ValueError
        when the name already maps to another member.
        """
        if type(name) is not str or not name:
            raise TypeError(f"a {self.kind} name is a non-empty string, not {name!r}")

        def register(member: M) -> M:
            held = self._members.get(name)
            if held is not None and held is not member:
                raise ValueError(
                    f"{self.kind} {name!r} is already {held.__name__}; "
                    f"a name maps to one {self.kind}")
            self._members[name] = member
            return member

        return register

    def __getitem__(self, name: str) -> T:
        if name not in self._members:
            for module in self._registering(name):
                importlib.import_module(module)
        failed: tuple[tuple[str, BaseException], ...] = ()
        if name not in self._members:
            failed = _loaded_plugins()
        try:
            return self._members[name]
        except KeyError:
            known = set(self._members) | {held for attribute, held in _registering_modules()
                                          if getattr(sys.modules[__name__], attribute, None) is self}
            message = f"no {self.kind} named {name!r}; known: {', '.join(sorted(known))}"
            if not failed:
                raise KeyError(message) from None
            broken = "; ".join(f"{entry} ({type(error).__name__}: {error})" for entry, error in failed)
            raise KeyError(f"{message}. These {PLUGINS} entries failed to import, so a name they register "
                           f"is not known: {broken}") from failed[0][1]

    def _registering(self, name: str) -> tuple[str, ...]:
        """The modules of Dew whose decorator registers `name` in this table."""
        return tuple(module for (attribute, held), modules in _registering_modules().items()
                     if held == name and getattr(sys.modules[__name__], attribute, None) is self
                     for module in modules)

    def __iter__(self) -> Iterator[str]:
        return iter(self._members)

    def __len__(self) -> int:
        return len(self._members)

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}, {sorted(self._members)})"

    def share(self) -> Registry[T, Built]:
        """Share this table so records can name its members, and return it.

        A typical use is `activations = Registry("activation").share()`. A
        record then rebuilds a member wherever a field declares the member's
        base class, and writes the member back by name. Share a plugin's kind
        in the module that defines that base class, so every field typed with
        it finds the table. A kind has one shared table; sharing the same table
        again does nothing, and sharing a second table of that kind raises
        ValueError.
        """
        for held in _SHARED:
            if held.kind == self.kind and held is not self:
                raise ValueError(f"a {self.kind} registry is already shared; a kind has one table")
        if not any(held is self for held in _SHARED):
            _SHARED.append(self)
        return self

    @staticmethod
    def shared() -> tuple[Registry, ...]:
        """Return every shared table, Dew's twelve first, then any a plugin shared."""
        return tuple(_SHARED)

    def name_of(self, member: Named) -> str:
        """Return the name a member was registered under in this table.

        The table is searched by identity, so `member` can be any class or
        function, whatever the registry builds. A member this table does not
        hold raises KeyError, with the decorator line that would register it."""
        for name, held in self._members.items():
            if held is member:
                return name
        raise KeyError(f"{member.__name__} is not a registered {self.kind}; register it once with "
                       f"`@dew.registry.{self.kind}s(\"{member.__name__.lower()}\")` above its class")

    def build(self, name: str, record: Mapping[str, object] = NO_RECORD, /,
              **fields: Configured) -> Built:
        """Construct the member called `name` from a record, keyword fields, or both.

        A field the member does not declare is an error. Fields come from JSON
        as often as from code, so a field whose declared type is a value class
        is built here from its record. For example,
        `models.build("m", attention={"heads": 8})` and
        `models.build("m", attention=Attention(heads=8))` build the same model.

        Pass a whole parsed config as the positional `record`. Its values are
        converted to the member's declared types here, so a caller does not
        need to know the member's fields to unpack the config. Keyword fields
        override the record's. A function member's fields are its parameters,
        converted to their annotations the same way, so a parameter typed
        with what another table's functions return takes a record naming one
        of them.
        """
        member = self[name]
        given: Mapping[str, object] = {**record, **fields}
        held = _record_class(member)
        if held is not None:
            given = _declared(held, given, dtypes=True)
        elif not isinstance(member, type):
            given = _arguments(member, given, dtypes=True)
        return member(**given)

    def from_record(self, record: Mapping[str, object]) -> Built:
        """Construct the member a `{"name": ..., "fields": {...}}` record names.

        A config writes a mixer, a tower or a projector as such a record, where
        code passes the value `build` makes. A record that names no registered
        member raises ValueError listing the known ones.
        """
        name, fields = _named(self, record)
        return self.build(name, fields)

    @property
    def union(self) -> type[Built] | types.UnionType:
        """The union of the members' types, for a tyro subcommand over the table."""
        return functools.reduce(operator.or_, self._members.values())


class Record(TypedDict):
    """A registered member as a record writes it: the name its table holds it
    under and its constructor fields, `{"name": "mla", "fields": {...}}`."""

    name: str
    fields: Mapping[str, object]


class Named(Protocol):
    """Declares the name a registry member carries of its own.

    A member is a class or a function -- the decorator takes both -- and each
    declares `__name__`, which is what an error names a member by. The registry's table holds members as the
    concrete type their decorator handed back, so this is the one thing read
    off them without the caller's own type.
    """

    @property
    def __name__(self) -> str: ...


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
    built = _rebuilt(annotation, value, dtypes=dtypes)
    witness: type[ValueT] = typing.get_origin(annotation) or annotation
    if not isinstance(built, witness):
        raise ValueError(f"{value!r} builds {type(built).__name__}, "
                         f"not the {witness.__name__} the field declares")
    return built


def to_record(value, annotation) -> JSON:
    """Return `value` as the record `from_record(annotation, ...)` rebuilds it from.

    The record is JSON: a dict, a list, or a scalar json.dump can write. A
    registered member is `{"name": ..., "fields": {...}}` and any other
    dataclass the record of its fields. `annotation` is the declared field
    type, so the write side names the same registry and member types the read
    side rebuilds from, and a value no registry holds where the field names
    one is refused rather than written as a record nothing reads back.
    """
    from flax import linen as nn

    from dew.records import recorded_duration

    if isinstance(value, datetime.timedelta):
        return recorded_duration(value)
    if isinstance(value, type) and value.__module__ in ('jax.numpy', 'numpy', 'ml_dtypes'):
        return dtype_name(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        held = _table_of(annotation)
        if held is not None and not any(type(value) is member
                                      for member in held.values()):
            raise ValueError(
                f"{type(value).__qualname__} is not a registered {held.kind}; a "
                f"run config can only record members that load back, so register "
                f"it once with `@dew.registry.{held.kind}s(\"{type(value).__name__.lower()}\")`")
        if held is None:
            held = _table_of(type(value))
        fields = {f.name: to_record(getattr(value, f.name), _declared_type(type(value), f.name))
                  for f in dataclasses.fields(value) if _recorded(f)
                  and not (isinstance(value, nn.Module) and f.name in ('parent', 'name'))}
        if held is None:
            return fields
        return {"name": held.name_of(type(value)), "fields": fields}
    if isinstance(value, (list, tuple)):
        entries = entry_types(annotation, len(value))
        return [to_record(entry_value, entry)
                for entry_value, entry in zip(value, entries, strict=True)]
    if isinstance(value, Mapping):
        entries = entry_types(annotation, len(value))
        return {_record_key(key): to_record(entry_value, entry)
                for (key, entry_value), entry in zip(value.items(), entries, strict=True)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        f"{type(value).__name__} is not something a run record can carry; a "
        f"config field holds JSON scalars, sequences, mappings, and the "
        f"registered values this writes as their name and fields")


def _record_key(key: object) -> str:
    """Return one mapping key as JSON names it, since JSON has only string keys.

    A tree path is its parts joined the way every message in this tree joins
    them, `params/layers_0/self_attn/q_proj`, which is the spelling `_key`
    splits back into the tuple the field declares. A key that is
    neither a name nor a path of them has no spelling a record reads back,
    so it is refused here rather than written as its repr.
    """
    if isinstance(key, tuple):
        return "/".join(_record_key(part) for part in key)
    if not isinstance(key, (str, int, float)):
        raise TypeError(
            f"{key!r} is not a key a run record can carry; a config mapping is keyed "
            f"by a name, a number, or a tree path of names")
    return str(key)


_MAPPINGS = (dict, Mapping, MutableMapping)


def _rebuilt(annotation: Annotation, value: object, *, dtypes: bool, name: str = "") -> Configured:
    """Return `value` as its annotation asks for it: a record becomes the value it
    describes, and anything already built is left alone.

    This is the one walk from a record to a value, for a module field and a
    run record alike. A registered member is the record that names it,
    `{"name": ..., "fields": {...}}` (`to_record` writes it), where
    the field declares the table's members or a class the member derives
    from, or a function of a shared table declared to return such a class;
    a dataclass is the record of its fields. Containers are walked, so
    a mapping of records and a tuple of records build their values too, a
    JSON list becomes the tuple a field declares, and a mapping's key
    becomes the tuple path its key type declares. `dtypes` is as
    `from_record` reads it, for a field or entry `name`d `dtype`.
    """
    if dtypes and name == "dtype":
        return resolve_dtype(value)
    annotation = resolve_alias(annotation)
    # A union of a table's members takes a record naming one of them; a field
    # typed with one class takes its own fields or a member's record below.
    table = _table_of(annotation) if typing.get_args(annotation) else None
    if table is not None:
        if isinstance(value, tuple(member for member in table.values() if isinstance(member, type))):
            return configured(value)
        name, fields = _named(table, value)
        member = _record_class(table[name])
        if member is None:
            raise ValueError(f"the {table.kind} {name!r} is not a class, and a record names "
                             "the fields of one")
        return _construct(member, fields, dtypes=dtypes)
    if typing.get_origin(annotation) in (Union, types.UnionType):
        if value is None:
            return value
        inner = [member for member in typing.get_args(annotation) if member is not type(None)]
        if len(inner) == 1:
            return _rebuilt(inner[0], value, dtypes=dtypes)
        # A union of a table's members (a schedule or None) rebuilds through
        # the table; any other union holds the value as it is.
        members = functools.reduce(operator.or_, inner)
        return (_rebuilt(members, value, dtypes=dtypes) if _table_of(members) is not None
                else configured(value))
    if isinstance(value, Mapping):
        if isinstance(annotation, type) and annotation is not object:
            # A field typed with a base class takes a record of any shared
            # member derived from it or returning it, or of the class itself;
            # `object` declares nothing, and its record stays the record it is.
            member, fields = _nested(annotation, value)
            held = _record_class(member)
            if held is not None:
                return _construct(held, fields, dtypes=dtypes)
            if not isinstance(member, type):
                return configured(member(**_arguments(member, fields, dtypes=dtypes)))
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


def _key(annotation: Annotation, key: str) -> str | tuple[str, ...]:
    """One record key as the mapping declares it: a name, or the tuple path
    `dew.config._key` joined with `/`."""
    return tuple(key.split("/")) if wants_tuple(annotation) else key


def _record_class(member: Callable[..., Configured]) -> type | None:
    """`member` when it is a dataclass, whose fields a record names; None for
    a function or a plain class, which declares no fields to narrow."""
    return member if isinstance(member, type) and dataclasses.is_dataclass(member) else None


def _construct(member: type, fields: Mapping[str, object], *, dtypes: bool) -> Configured:
    """Build the dataclass `member` from a record of its fields."""
    return configured(member(**_declared(member, fields, dtypes=dtypes)))


def _declared(member: type, fields: Mapping[str, object], *, dtypes: bool) -> dict[str, Configured]:
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
    return {name: _rebuilt(_declared_type(member, name), configured(value), dtypes=dtypes, name=name)
            for name, value in fields.items()}


def _arguments(function: Callable[..., Configured], fields: Mapping[str, object], *,
               dtypes: bool) -> dict[str, Configured]:
    """The record's fields as the parameters of the registered `function`,
    each walked against its own annotation, as `_declared` walks a
    dataclass's. A field the function does not take, or a parameter without
    a default that the record lacks, raises; a function taking `**kwargs`
    takes any field."""
    parameters = inspect.signature(function).parameters.values()
    named = [parameter for parameter in parameters
             if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)]
    names = sorted(parameter.name for parameter in named)
    open_ended = any(parameter.kind is parameter.VAR_KEYWORD for parameter in parameters)
    unknown = [] if open_ended else sorted(set(fields) - set(names))
    missing = [parameter.name for parameter in named
               if parameter.name not in fields and parameter.default is parameter.empty]
    if unknown or missing:
        raise ValueError(f"{function.__name__} does not match the record: unknown fields {unknown}, "
                         f"missing fields {missing}; its parameters are {names}")
    return {name: _rebuilt(_parameter_type(function, name), configured(value), dtypes=dtypes, name=name)
            for name, value in fields.items()}


def _parameter_type(function: Callable[..., Configured], name: str) -> Annotation:
    """Resolve the annotation of one parameter of `function`, or its return
    for `name="return"`, without evaluating the others. None where it has no
    annotation. An annotation naming what the function's module does not
    import at runtime raises, since a record cannot be read against it."""
    # A classmethod registered as `table("name")(Class.reader)` is a bound
    # method; its annotations and module are its function's.
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
    return isinstance(returned, type) and issubclass(returned, held)


def _recorded(field: dataclasses.Field) -> bool:
    """Return whether a field is part of a record: one the constructor takes,
    not marked `metadata={"record": False}`."""
    return field.init and field.metadata.get("record", True)


def _table_of(annotation: Annotation) -> Registry | None:
    """The shared table whose members the annotation names, all of them, or None."""
    members = typing.get_args(annotation) or (annotation,)
    for table in Registry.shared():
        if all(any(member is held for held in table.values()) for member in members):
            return table
    return None


def _named(table: Registry, record: object) -> tuple[str, Mapping[str, object]]:
    """The name and fields of a `{"name": ..., "fields": {...}}` record of `table`."""
    if (not isinstance(record, Mapping) or set(record) != {"name", "fields"}
            or not isinstance(record["name"], str) or not isinstance(record["fields"], Mapping)):
        raise ValueError(f"a {table.kind} is the record that names it, {{'name': ..., 'fields': {{...}}}}, "
                         f"not {record!r}")
    try:
        table[record["name"]]
    except KeyError as error:
        # A record naming nothing registered is a bad config, not a lookup.
        raise ValueError(error.args[0]) from error
    return record["name"], record["fields"]


def _nested(held: type,
            record: Mapping[str, object]) -> tuple[Callable[..., Configured], Mapping[str, object]]:
    """The member and fields a record builds for a field typed `held`: the
    shared member a name/fields record names, where that member is `held`,
    derives from it, or is a function declared to return it, or else `held`
    and the record as its own fields."""
    if set(record) == {"name", "fields"} and isinstance(record["name"], str) and isinstance(
            record["fields"], Mapping):
        for table in Registry.shared():
            member = table.get(record["name"])
            if member is None:
                continue
            if issubclass(member, held) if isinstance(member, type) else _returns(member, held):
                return member, record["fields"]
    return held, record


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


# The flag each precision field is set by, for the error a model config that
# carries one of them raises.
_PRECISION_FLAGS = {"dtype": "--model.dtype", "attention_impl": "--model.attention-impl",
                    "param_dtype": "--model.param-dtype",
                    "precision": "--model.matmul-precision"}


class PrecisionFields(TypedDict, total=False):
    """Names what a run's precision settings write into a model config.

    Only the keys the named member declares are written, so the bag is
    partial by construction; `attention_configs` is the UNets' per-stage
    settings, which carry the dtype into each stage a config named. Its
    entries are a stage record, a built `Stage` or None, which is what a
    model config carries and what `build` narrows against the field.
    """

    dtype: str | None
    attention_impl: str
    param_dtype: str
    precision: str
    attention_configs: list[object]


def precision_fields(name: str, config: Mapping[str, object], *,
                     dtype: str | None, attention_impl: str, param_dtype: str | None = None,
                     matmul_precision: str | None = None) -> PrecisionFields:
    """Return the run's compute dtype and attention kernel as the fields a model takes.

    `with_precision` is the same settings merged into the config they belong
    to; this is them on their own, for a caller holding a typed field bag.

    `attention_impl` is an `AttentionImpl` and travels as it is written; the
    kernel resolves 'auto' against each call, so the name a run recorded is
    the name the model holds.

    `dtype` is the compute dtype; `param_dtype` is where the parameters are
    stored and `matmul_precision` what every matmul asks XLA for. Each is
    written into the field the model declares for it (`dtype`, `param_dtype`,
    `precision`), and a run that names neither of the last two writes
    neither. Unset, parameters stay float32 and the model keeps its own
    precision. A setting the model declares no field for raises ValueError
    naming the model and the field, since the run would not compute as it
    says; a composite that declares no `dtype` takes the run's dtype when a
    registered part in its config declares one (`with_precision`).

    The UNets keep per-stage attention settings in `attention_configs`,
    which do not inherit the model dtype and default `force_fp32_for_softmax`
    off, which no fused kernel can honour, so the knobs reach into them.

    A stage arrives either way: as a record, whose `dtype` name `build`
    resolves at the boundary with every other field, or as a built `Stage`,
    which nothing resolves afterwards, so its dtype is resolved here. The two
    agree once built, which `tests/test_models.py` asserts.
    """
    member = models[name]
    declared = {f.name for f in dataclasses.fields(member) if f.init}
    named = {"dtype": dtype, "param_dtype": param_dtype, "precision": matmul_precision}
    unreached = [field for field, value in named.items() if value is not None and field not in declared
                 and not (field == "dtype" and any(_part_takes_dtype(configured(part))
                                                    for part in config.values()))]
    if unreached:
        settings = ", ".join(f"{_PRECISION_FLAGS[field]} {named[field]}" for field in unreached)
        raise ValueError(
            f"the model {name!r} declares no {' or '.join(unreached)} field, so the run's {settings} "
            f"would not reach it; set {', '.join(_PRECISION_FLAGS[field] for field in unreached)} "
            f"to None")
    written: PrecisionFields = {}
    if "dtype" in declared:
        written["dtype"] = dtype
    if "attention_impl" in declared:
        written["attention_impl"] = attention_impl
    if param_dtype is not None and "param_dtype" in declared:
        written["param_dtype"] = param_dtype
    if matmul_precision is not None and "precision" in declared:
        written["precision"] = matmul_precision
    duplicate = sorted(set(config) & set(written))
    if duplicate:
        raise ValueError(
            f"the model config carries {duplicate}, which the run's precision "
            f"settings own; set {', '.join(_PRECISION_FLAGS[held] for held in duplicate)} instead")
    fields: PrecisionFields = {**written}
    stages = {f.name: f for f in dataclasses.fields(member)}.get("attention_configs")
    if stages is not None:
        carried = config.get("attention_configs", stages.default)
        if not isinstance(carried, (list, tuple)):
            raise ValueError(
                f"attention_configs is {carried!r}; a unet takes one entry per "
                f"resolution stage, each a record, a Stage, or None for a stage "
                f"that does not attend")
        resolved: list[object] = []
        for stage in carried:
            if stage is None:
                resolved.append(None)
            elif isinstance(stage, Mapping):
                resolved.append({**stage, "dtype": dtype, "force_fp32_for_softmax": True})
            elif dataclasses.is_dataclass(stage) and not isinstance(stage, type):
                resolved.append(dataclasses.replace(
                    stage, dtype=resolve_dtype(dtype), force_fp32_for_softmax=True))
            else:
                raise ValueError(
                    f"attention_configs carries {stage!r}; a stage is a record or "
                    f"a Stage")
        fields["attention_configs"] = resolved
    return fields


def float64_twin(config: Mapping[str, object]) -> Mapping[str, object]:
    """`config`, a model config `with_precision` wrote, computing in float64:
    its dtype and the dtype `precision_fields` wrote into each stage of the
    UNets' `attention_configs`, the only nested dtypes the policy writes.
    Under x64 the model computes in float64 throughout
    (`dew.nn.precision.at_least_fp32`): the twin a float64 reference runs. A
    run's dtype knob names no float64; only a reference computes in it."""
    twin: dict[str, object] = {**config, "dtype": jnp.float64}
    stages = config.get("attention_configs")
    if isinstance(stages, list | tuple):
        twin["attention_configs"] = [
            {**stage, "dtype": jnp.float64} if isinstance(stage, Mapping)
            else dataclasses.replace(stage, dtype=jnp.float64)
            if dataclasses.is_dataclass(stage) and not isinstance(stage, type) else stage
            for stage in stages]
    return twin


def with_precision(name: str, config: Mapping[str, object], *,
                   dtype: str | None, attention_impl: str, param_dtype: str | None = None,
                   matmul_precision: str | None = None) -> Mapping[str, object]:
    """Return a model config with the run's compute dtype and attention kernel in it.

    A composite that declares no `dtype` of its own, such as DiffusionGemma
    around its text decoder, passes the run's compute dtype on to the
    registered model parts nested in its record that declare one."""
    fields = {**config, **precision_fields(
        name, config, dtype=dtype, attention_impl=attention_impl,
        param_dtype=param_dtype, matmul_precision=matmul_precision)}
    if dtype is None or "dtype" in {field.name for field in dataclasses.fields(models[name])}:
        return fields
    return {key: _with_part_dtype(configured(value), dtype) for key, value in fields.items()}


def _part_takes_dtype(value: Configured) -> bool:
    """Whether a composite's field is a registered model, recorded or built,
    that declares a compute dtype."""
    from flax import linen as nn

    if isinstance(value, nn.Module):
        return type(value) in models.values() and "dtype" in {f.name for f in dataclasses.fields(value)}
    if not isinstance(value, Mapping) or set(value) != {"name", "fields"}:
        return False
    name, fields = value["name"], value["fields"]
    return (isinstance(name, str) and name in models and isinstance(fields, Mapping)
            and "dtype" in {field.name for field in dataclasses.fields(models[name])})


def _with_part_dtype(value: Configured, dtype: str) -> Configured:
    """A composite's part with the run's compute dtype, where the part is a
    registered model, recorded or built, that declares one."""
    from flax import linen as nn

    if not _part_takes_dtype(value):
        return value
    if isinstance(value, nn.Module):
        return value.clone(dtype=resolve_dtype(dtype))
    assert isinstance(value, Mapping) and isinstance(value["fields"], Mapping)
    return {"name": value["name"], "fields": {**value["fields"], "dtype": dtype}}


# Every table's record is `{"name": ..., "fields": {...}}`, which
# `Registry.from_record` and the run record's walk read alike.
models: Registry[type[nn.Module], nn.Module] = Registry("model").share()
presets: Registry[type[Preset], Preset] = Registry("preset").share()
solvers: Registry[type[Solver[Any]], Solver[Any]] = Registry("solver").share()
datasets: Registry[type[DatasetSpec], DatasetSpec] = Registry("dataset").share()
encoders: Registry[type[ConditionEncoder[Any]], ConditionEncoder[Any]] = Registry("encoder").share()
metrics: Registry[Callable[..., Metric], Metric] = Registry("metric").share()
objectives: Registry[type[Objective], Objective] = Registry("objective").share()
mixers: Registry[type[MixerBase], MixerBase] = Registry("mixer").share()
towers: Registry[type[TowerBase], TowerBase] = Registry("tower").share()
projectors: Registry[type[ProjectorBase], ProjectorBase] = Registry("projector").share()
schedules: Registry[type[ScheduleBase], ScheduleBase] = Registry("schedule").share()
trainings: Registry[type[Training], Training] = Registry("training").share()

__all__ = [
    "PLUGINS",
    "Record",
    "Registry",
    "datasets",
    "dtype_name",
    "encoders",
    "from_record",
    "metrics",
    "mixers",
    "models",
    "objectives",
    "presets",
    "projectors",
    "schedules",
    "solvers",
    "to_record",
    "towers",
    "with_precision",
]
