"""Short names for the classes a run is made of, and the records that rebuild them.

A record names a class by its import path, `{"class": "dew.nn.dit:SimpleDiT",
"fields": {...}}`, and a function the same way, `{"function": "optax:adamw"}`.
Rebuilding one imports the module and calls the class with its fields, each
converted to the type its annotation declares, so a record loads in a process
that imported nothing beforehand, and a class outside Dew needs nothing to be
recorded: its import path is its name. A record Dew writes carries the path,
so rebuilding it never depends on an alias, with one exception: a run's own
record names its run class by its alias in `runs`, so a saved run reads back
wherever its class moves, and only that kind is consulted for it.

Where a person writes a record, in a config file or on the command line, the
class may be a short alias instead, `{"class": "simple_dit"}`. Each kind's
aliases are one table (`models`, `mixers`, ...) that maps a name to a path:
Dew's own classes are its static entries, imported only when one is read, and a
user's class joins it under a name of its own with a decorator,
`@dew.register("my_model")` for a model (`register`) or a kind's own
`@mixers.register("my_mixer")`, as its module is imported. A record still
names the user's class by its path, so it loads without that module imported
first where the process trusts its package (`import_trusted`), as any class
outside Dew does. An unknown name or field raises an error.

A run's model is one such record, `ModelConfig`: the model's class, the fields
the run gives it, the defaults it took and the wrappers it was made in. A
component built around parameters (an encoder, an autoencoder) records the view
it is rebuilt from (`Recorded`), and its class is found back by `component`.

The tables are the one list of every kind, each typed by its members' base, which
this module names under TYPE_CHECKING alone: it imports no feature package as it
runs (tests/test_imports.py), and so sits beside the foundation modules without
being one of them (row 4266).
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import datetime
import functools
import importlib
import importlib.util
import inspect
import itertools
import json
import operator
import os
import sys
import types
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Iterator, Mapping, MutableMapping, MutableSequence, Sequence
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Literal,
    Protocol,
    Self,
    TypeAliasType,
    TypedDict,
    Union,
    get_args,
    overload,
    runtime_checkable,
)

import jax
import jax.numpy as jnp
import numpy as np
import tyro
from jax.typing import DTypeLike
from typing_extensions import Format, get_annotations, is_protocol

from dew import _records
from dew._records import JSON, boolean, class_record, integer, json_value, real, strings_of, text

if TYPE_CHECKING:
    from _typeshed import DataclassInstance
    from flax import linen as nn

    from dew.config import RunConfig
    from dew.diffusion.presets import Preset
    from dew.inputs.condition import ConditionEncoder
    from dew.interop.composite import SourceKind
    from dew.nn.mixers import MixerBase
    from dew.nn.vision import ProjectorBase, TowerBase
    from dew.objectives.base import Metric, Objective
    from dew.objectives.rl.control import ControlRollout
    from dew.objectives.rl.dreamer import DreamerRollout
    from dew.optim import OptimizerSpec
    from dew.optim_schedules import ScheduleBase
    from dew.sampling.solvers import Solver

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
type Configured = (JSON | DTypeLike | Enum | np.ndarray | np.generic | datetime.timedelta
                   | types.FunctionType | types.BuiltinFunctionType | functools.partial
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


class Unrecordable(TypeError, ValueError):
    """What a record cannot describe: a member no import path names, a value or key no record carries, a
    class whose constructor's parameters it does not keep. Written where the record is, it is a TypeError and
    a ValueError, as each of these was raised before."""


def import_path(member: Importable) -> str:
    """The path a record names a class or function by, `module:Qualified.name`.

    A lambda, a function defined inside another, or anything defined in
    `__main__` has no path another process can import, so it is refused here,
    where the record is written, rather than when it is read; as is a method
    bound to an instance, whose receiver a path does not name (a reload would
    import its unbound function: the API02 review)."""
    if isinstance(member, types.MethodType) and not isinstance(member.__self__, type):
        raise Unrecordable(f"{member!r} is bound to an instance its path cannot name; record a function, a "
                           "class or a dataclass that holds what it reads")
    function = member.__func__ if isinstance(member, types.MethodType) else member
    module, name = function.__module__, function.__qualname__
    if not module or module == "__main__" or "<" in name:
        raise Unrecordable(f"{member!r} has no import path a record can name; define it at module level "
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
    package = module.split(".")[0]
    if (module != "dew" and not module.startswith("dew.") and module not in sys.modules
            and not (package in _TRUSTED and module.replace(".", "").isidentifier() and _importable(module))):
        # A module of a trusted package a record names is imported where the registry reaches it.
        raise ValueError(f"{path!r} names {module}, which is outside Dew and not imported; import it "
                         f"first, or trust its package where the run loads (trust=({package!r},), "
                         f"--trust {package})")
    # Whatever the path names, an object until the check below says what it is.
    held: object = importlib.import_module(module)
    for part in name.split("."):
        try:
            held = getattr(held, part)
        except AttributeError:
            raise ValueError(f"{module} has no {name}, which {path!r} names") from None
    if not isinstance(held, (type, types.FunctionType, types.BuiltinFunctionType, types.MethodType)):
        raise ValueError(f"{path!r} names {held!r}, which is neither a class nor a function")
    return held


_FILES: dict[str, Path] = {}
"""The modules `module_from_file` imported, by name, each its file."""


def module_from_file(path: str | os.PathLike[str], name: str, *, beside: bool = False) -> types.ModuleType:
    """The Python file at `path` imported as the module `name`, which `sys.modules` holds before it runs,
    so what it defines is named by `name` (`import_path`) and a record of it reads back here
    (`imported`). Each call runs the file again; a file this imported before under `name` gives way
    to this one. A name that is another module's, imported or importable (the standard library's
    `json` for a `json.py`), or a package's module (a dotted name), is refused rather than shadowed
    or read in its place, the process as it was. `beside` puts the file's directory first on the path
    once the file is accepted, as `python -m` runs one beside its siblings, so it imports them."""
    location = Path(path).resolve()
    if "." in name:
        raise ValueError(f"{location} would import as {name}, a module of the package "
                         f"{name.rpartition('.')[0]}, which a file cannot stand in for; rename the file")
    held = sys.modules.get(name)
    if held is not None and _FILES.get(name) is None:
        other = (held.__spec__.origin if held.__spec__ is not None else None) or "a module without a file"
    elif held is None and name.isidentifier() and (found := importlib.util.find_spec(name)) is not None:
        other = found.origin or "a namespace package"
    else:
        other = None
    if other is not None and (other in ("built-in", "frozen") or Path(other).resolve() != location):
        raise ValueError(f"{location} would import as {name}, which names {other}; rename the file")
    spec = importlib.util.spec_from_file_location(name, location)
    if spec is None or spec.loader is None:
        raise ValueError(f"{location} is not a Python file")
    module = importlib.util.module_from_spec(spec)
    if beside and str(location.parent) not in sys.path:
        sys.path.insert(0, str(location.parent))
    sys.modules[name], _FILES[name] = module, location
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name], _FILES[name]
        raise
    return module


_TRUSTED: set[str] = {"flax"}
"""The packages outside Dew the process trusts (`import_trusted`): from the
start Flax's, whose layers a model's record names (a `Sequential`'s)."""


_TRUSTED_SOURCES: set[tuple[str, str]] = set()
"""The sources whose own code the process runs where a load reads them (`trusts_source`), by identity
(`_source_identity`): none from the start, unlike the packages, which hold Flax's."""


def _source_identity(name: str) -> tuple[str, str]:
    """A source's identity: an existing directory by its resolved path, which no other working directory
    and no Hub repository of the same name shares; any other name the Hub repository it names."""
    path = Path(name).expanduser()
    return ("directory", str(path.resolve())) if path.is_dir() else ("hub", name)


def trusts_source(name: str) -> bool:
    """Whether the process trusts the source `name`, a Hub repository or a directory, to run its own
    code where a load reads it (`trust_remote_code`): as `trust=(name,)` and `--trust name` say where a
    run or a task loads (`import_trusted`), or a load asked to run it (`trust_sources`)."""
    return _source_identity(name) in _TRUSTED_SOURCES


def trust_sources(names: Sequence[str]) -> None:
    """Trust the sources `names` for the rest of the process (`trusts_source`)."""
    _TRUSTED_SOURCES.update(_source_identity(str(name)) for name in names)


def import_trusted(record: JSON | Mapping[str, object], trust: Sequence[str]) -> None:
    """Trust the packages in `trust` for the rest of the process, import each
    of them, so the classes their modules name by a decorator join their kinds
    (`Aliases.register`, a run.json's `{"name": "my_model"}`), and import every
    module of theirs that `record` names by an import path, so reading the
    record finds them imported (`imported`) and may construct their classes
    where a field declares no base class (`_owned`). A source named there, a
    Hub repository or a directory, has its own code run where a load reads it
    (`trusts_source`)."""
    _TRUSTED.update(trust)
    trust_sources(trust)
    for package in trust:
        # A name shaped as a package may name a source directory alone (`trusts_source`), which no module
        # path holds: only a module the process can import is imported.
        if package.replace(".", "").isidentifier() and _importable(package):
            importlib.import_module(package)
    # Only JSON a record holds (`json_value` refuses any other), its every string read once.
    for named in strings_of(json_value(record, "record")):
        module, colon, name = named.partition(":")
        # A string shaped as an import path that names no module (a text field's) is data, not a module.
        if (colon and name and module.split(".")[0] in trust and module.replace(".", "").isidentifier()
                and _importable(module)):
            importlib.import_module(module)


def _importable(name: str) -> bool:
    """Whether the module `name` is one the process can import: its package's, then its own, each one the
    process holds already (a module installed without a spec included) or one found on the module path
    (`importlib.util.find_spec`)."""
    parts = name.split(".")
    prefixes = (".".join(parts[:end]) for end in range(1, len(parts) + 1))
    return all(sys.modules.get(prefix) is not None or importlib.util.find_spec(prefix) is not None
               for prefix in prefixes)


class Aliases[T: Callable[..., Any], Built](Mapping[str, T]):
    """The short names of one kind, each mapped to the import path of its class.

    `models["simple_dit"]` imports and returns `SimpleDiT`, as does
    `models["dew.nn.backbones.dit:SimpleDiT"]`: the table is static, and a
    class outside it, Dew's or a user's, is named by its path."""

    def __init__(self, kind: str, paths: Mapping[str, str], base: str | None = None):
        self.kind = kind
        self._paths = dict(paths)
        self.paths = types.MappingProxyType(self._paths)
        """Each alias's import path: Dew's own, the static table's, then each one a user's class took
        (`register`)."""
        self._own = frozenset(paths)
        self.base = base
        """The import path of the class every member derives from, or of the
        protocol they share, whose members are then classes of Dew's or of a
        trusted package, or functions declared to return one (`_derives`); None
        where the members share only an unnamed protocol."""
        # What a lookup imported, held to the kind's contract (`_holds`), its type `T`'s on the way out: a
        # bound cannot name the kind's `Built`, so a member's call is typed by the table it is read through.
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

    def register[M: Callable[..., Any]](self, name: str, /) -> Callable[[M], M]:
        """A decorator naming the class (or function) it decorates `name` in this kind, as its module is
        imported: `models.build("name", ...)`, a config's `{"class": "name"}` and the command line build it
        as they build Dew's own. The member must be of the kind (`base`), and the name no other member's; a
        record names the member by its import path still (`name_of`), so it loads in a process that has
        not imported the member's module, once that process trusts its package (`import_trusted`)."""
        def registered(member: M) -> M:
            path = import_path(member)
            held = self._paths.get(name)
            if held is not None and held != path:
                raise ValueError(f"{name!r} names the {self.kind} {held} already, not {path}")
            # A user's own code, imported to be registered: a kind of no base takes it as it is.
            if self.base is not None and not self._holds(member):
                raise ValueError(f"{path} is no {self.kind}, which {name!r} would name")
            self._paths[name] = path
            self._imported[name] = member
            return member
        return registered

    def _holds(self, member: Callable[..., Configured]) -> bool:
        if self.base is None:
            return _owned(member, called=False)
        module, _, qualified = self.base.partition(":")
        base = getattr(importlib.import_module(module), qualified)
        return _derives(member, base) if isinstance(member, type) else _returns(member, base)

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

    def name_of(self, member: Importable) -> str:
        """The name a record writes for `member`: its alias among Dew's own in this kind, else its import
        path, a user's registered class's among them (`register`)."""
        path = import_path(member)
        return next((alias for alias, held in self.paths.items() if held == path and alias in self._own),
                    path)

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
        with _reading(given):
            given = _taken(member, given, dtypes=True)
        return member(**given)

    def from_record(self, record: Mapping[str, object]) -> Built:
        """Construct the member a class record names, `{"class": ..., "fields": {...}}`."""
        named = class_record(record)
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

    @property
    def subcommands(self) -> Annotation:
        """The members' union with each subcommand named as the kind names it, hyphenated as a flag is
        (`optim.optimizer:gauss-newton`),
        rather than by its class, a subclass's before its base's: tyro chooses a default's subcommand as the
        first member it is an instance of, an AdamW's its own and not an Adam's."""
        # The special form subscripted at runtime, where each member is a value rather than a
        # type expression, which a type checker reads `typing.Annotated[...]` as.
        annotated = vars(typing)["Annotated"]
        names = sorted(self, key=lambda name: -len(kind.__mro__) if isinstance(kind := self[name], type)
                       else 0)
        members = (annotated[self[name], tyro.conf.subcommand(name.replace("_", "-"))] for name in names)
        return functools.reduce(operator.or_, members)

    def flag_choices(self, record: type[Bound]) -> Callable[[], type]:
        """The members' records (`record`, this kind's `Bound` class) as a command line chooses them
        (`record_flags`): a subcommand an alias (`env:flight`), its arguments plain flags, made when a
        command line is parsed (`tyro.conf.arg(constructor_factory=...)`), one class's first alias alone,
        a subclass's before its base's (tyro reads a union's first member a default is an instance of)."""

        def choices() -> type:
            # A union of the subcommands' annotated records, which tyro's `constructor_factory` takes where
            # its stubs say a type (`typing.Annotated` read by name, as tyro reads it).
            annotated = vars(typing)["Annotated"]
            named: dict[str, str] = {}
            for alias in self:
                named.setdefault(self.paths[alias], alias)
            members = [(self[alias], path, alias) for path, alias in named.items()]
            members.sort(key=lambda each: -len(each[0].__mro__) if isinstance(each[0], type) else 0)
            return functools.reduce(operator.or_, (annotated[record_flags(record, member, path, alias),
                                                             tyro.conf.subcommand(alias.replace("_", "-"))]
                                                   for member, path, alias in members))

        return choices


def declared_type(member: type, field: str) -> Annotation:
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


def _union_members(annotation: Annotation) -> tuple[Annotation, ...]:
    """A union's members, aliases within it expanded and each member once, by identity, however many of
    its aliases name it (`Stops | Pause` where `Stops` holds `Pause`)."""
    annotation = resolve_alias(annotation)
    if typing.get_origin(annotation) in (Union, types.UnionType):
        expanded = [inner for member in typing.get_args(annotation) for inner in _union_members(member)]
        return tuple(member for index, member in enumerate(expanded)
                     if not any(member is earlier for earlier in expanded[:index]))
    if typing.get_origin(annotation) is typing.Annotated:
        return _union_members(typing.get_args(annotation)[0])
    return (annotation,)


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


@overload
def from_record[ValueT](annotation: type[ValueT], value: Configured, *, dtypes: bool = True) -> ValueT: ...


@overload
def from_record(annotation: types.UnionType | TypeAliasType, value: Configured, *,
                dtypes: bool = True) -> Configured: ...


def from_record[ValueT](annotation: type[ValueT] | types.UnionType | TypeAliasType, value: Configured, *,
                        dtypes: bool = True) -> ValueT | Configured:
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
    if isinstance(annotation, (types.UnionType, TypeAliasType)):
        return built
    witness = typing.get_origin(annotation) or annotation
    if not isinstance(built, witness):
        raise ValueError(f"{value!r} builds {type(built).__name__}, "
                         f"not the {witness.__name__} the field declares")
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

    from dew._records import recorded_duration

    if isinstance(value, Bound):
        # A command line's subcommand (`record_flags`) is the record it makes, its base's; a bound member is
        # recorded as `{"name", "fields", "defaults", ...}`, its other settings beside (`Bound.record`), and
        # named by its kind's class where the annotation does not name it (a union of kinds).
        held = value.recorded()
        record = held.record()
        if type(held) is nominal_class(_unwrapped(annotation)):
            return record
        return {"class": import_path(type(held)), "fields": record}
    if isinstance(value, functools.partial):
        # A member bound by keyword (`Bound.bound`), recorded as the class record of what it builds, where its
        # reader reads one back (a field declaring `functools.partial[...]`, `_bound`; or a member's held
        # argument, read against the member's own annotation, `Bound.read`): any other field's reader builds
        # what such a record names, or refuses a function's, so the partial is refused here.
        if annotation is not None and not _admits_partial(annotation, value):
            raise Unrecordable(f"{value!r} is held where {annotation!r} is declared, whose record reads it "
                               "back as no bound member: a field declaring functools.partial of a kind its "
                               "member is reads one")
        if value.args:
            raise Unrecordable(f"{value!r} binds arguments by position, which a record names none of")
        return {"class": import_path(value.func), "fields": argument_records(value.func, value.keywords)}
    if isinstance(value, datetime.timedelta):
        try:
            return recorded_duration(value)
        except ValueError as refused:
            # A duration no record carries is a value no record describes (`Unrecordable`), as any other.
            raise Unrecordable(str(refused)) from refused
    if isinstance(value, np.dtype) or (isinstance(value, type)
                                       and value.__module__ in ('jax.numpy', 'numpy', 'ml_dtypes')):
        return dtype_name(value)
    if isinstance(value, jax.lax.Precision | jax.lax.DotAlgorithmPreset):
        # A matmul precision as the name jax reads back as it: a dot algorithm
        # preset's own ('F32_F32_F32'), a precision's alias ('highest').
        return value.name if isinstance(value, jax.lax.DotAlgorithmPreset) else value.name.lower()
    if ((isinstance(value, Recorded) or dataclasses.is_dataclass(value))
            and not isinstance(value, type)):
        with recording() as written:
            if isinstance(value, nn.Module) and id(value) in written.modules:
                return _referred(written, id(value))
            record = _record_of(value, annotation)
            if isinstance(value, nn.Module):
                written.modules[id(value)] = (value, record)
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
        mapping = {_record_key(key, keys): to_record(entry_value, entry)
                   for (key, entry_value), entry in zip(value.items(), entries, strict=True)}
        number = _sharing(mapping)
        if number is not None:
            # Data shaped like a shared module's record keeps its number to itself.
            with recording() as written:
                written.taken.add(number)
        return mapping
    if isinstance(value, Enum):
        # A member is recorded as its value, which `from_record` reads back as the member (`_rebuilt`).
        return to_record(value.value, None)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, type) and _declares_callable(annotation):
        # A class named where a callable is declared (a rollout's reward by its class), recorded by its import
        # path as a function is, which reads back as the class (`from_record`).
        return {"function": import_path(value)}
    if not isinstance(value, type) and _owned(type(value), called=False):
        # A plain class of Dew's or a trusted package's, rebuilt from its
        # constructor's parameters as `_built` builds one; it declares no
        # fields, so its record always names it.
        return {"class": import_path(type(value)), "fields": _constructed(value)}
    raise Unrecordable(
        f"{type(value).__name__} is not something a run record can carry; a "
        f"config field holds JSON scalars, sequences, mappings, dataclasses and "
        f"functions with an import path")


def _declares_callable(annotation) -> bool:
    """Whether `annotation` declares a callable (`Callable[...]`), any member of its union (`_union_members`,
    through its aliases and `Annotated`)."""
    return any(resolve_alias(member) is Callable or typing.get_origin(resolve_alias(member)) is Callable
               for member in _union_members(annotation))


def _record_of(value: Recorded | DataclassInstance, annotation: Annotation) -> dict[str, JSON]:
    """One value's record, written inside its `to_record` call's recording: a component built around
    parameters (an encoder, an autoencoder) by the constructor view it is rebuilt from, never its
    parameters, always naming its class (`Recorded`, `component`); any other dataclass by its fields,
    as a class record unless `annotation` names its class exactly."""
    if isinstance(value, Recorded):
        view = value.recorded_fields()
        return {"class": import_path(type(value)),
                "fields": {name: to_record(entry, object) for name, entry in view.items()}}
    fields = record_fields(value, type(value))
    # The class the annotation names, a subscripted generic's origin among them, as every reader takes it
    # (`nominal_class`, row 8403).
    if type(value) is nominal_class(_unwrapped(annotation)):
        return fields
    return {"class": import_path(type(value)), "fields": fields}

def _constructed(value: Configured) -> dict[str, JSON]:
    """The record of a plain class's instance: each constructor parameter as
    the attribute of that name holds it, which `_built` passes back."""
    cls = type(value)
    named, _ = parameters(cls)
    fields = {}
    for name, (_, owner) in named.items():
        if not hasattr(value, name):
            raise Unrecordable(f"{cls.__name__} keeps no attribute {name!r} for its constructor parameter, "
                            f"so a record cannot rebuild it")
        fields[name] = to_record(getattr(value, name), parameter_type(owner, name))
    return fields


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
        raise Unrecordable(
            f"{key!r} is not a key a run record can carry; a config mapping is keyed "
            f"by a name, a number, or a tree path of them")
    try:
        back = _key(declared, written)
    except ValueError:
        back = written
    if not _same_key(back, key):
        raise Unrecordable(f"the key {key!r} would read back as {back!r} under the mapping's key type "
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


def _admits_function(annotation: Annotation) -> bool:
    """Whether a value `annotation` declares may be a function, so that a function record
    (`{"function": name}`) is the function it names: not a mapping's entry type itself nor a union of
    data alone (`dew._records.JSON`: scalars, sequences and mappings), where `{"function": ...}` is data
    a JSON round trip keeps as it is."""
    origin = typing.get_origin(annotation)
    if origin in _MAPPINGS:
        return False
    if origin in (Union, types.UnionType):
        scalars = (bool, int, float, str, type(None))
        containers = (*_MAPPINGS, list, tuple, Sequence)
        return any(member not in scalars and typing.get_origin(member) not in containers
                   and _admits_function(member) for member in _union_members(annotation))
    return True


def _unnamed(annotation: Annotation, classes: tuple[type, ...], value: Mapping, *,
             dtypes: bool) -> Configured:
    """A mapping naming no class (`class_record`) for a field of a union declaring no mapping (`_rebuilt`):
    the record of the one dataclass of the union that reads it as its record (`_holds_record`: its keys
    fields the class records, its required fields among them) and builds from it, its own checks passing
    (a scaling's `rope_type`), as a field of the class reads it (row 11980: `TrainerConfig.stop`, a Pause
    or a Plateau, given `{"fraction": 0.5}` held a dict, and the run never paused; a saved decoder's
    `rope_scaling={"rope_type": "yarn", "factor": 4.0}` is a `YarnScaling`); refused where several
    classes build from it, as nothing says which it is, or none does. A mapping no class reads as its
    record stays as written, a source's own record its field reads."""
    # The classes that read the record as written first; one only where a saved record's retired field
    # left out (`_records.saved_fields`) where none does, so retirement never makes a record ambiguous.
    fitting = ([member for member in classes if member is not object and _holds_record(member, value)]
               or [member for member in classes if member is not object
                   and _holds_record(member, value, saved=True)])
    if not fitting:
        return configured(value)
    built = []
    shared = _READING.get()
    # Each trial from the shared modules as they stood before any (`before`), its own kept beside what it
    # built: the read keeps the one trial that builds, no other trial's, a refused or an ambiguous one's
    # neither (the API02 review: a trial read the shared modules another had built).
    before = None if shared is None else (dict(shared.built), dict(shared.sources))
    for member in fitting:
        # A class whose own check refuses the record (a scaling's `rope_type` another's) is not its class.
        if shared is not None and before is not None:
            shared.built, shared.sources = dict(before[0]), dict(before[1])
        try:
            made = _construct(member, value, dtypes=dtypes)
        except ValueError:
            if len(fitting) == 1:
                raise
            continue
        built.append((made, None if shared is None else (shared.built, shared.sources)))
    if shared is not None and before is not None:
        kept = built[0][1] if len(built) == 1 else None
        shared.built, shared.sources = kept if kept is not None else before
    if len(built) == 1:
        return built[0][0]
    names = ", ".join(f"{member.__module__}:{member.__qualname__}" for member in fitting)
    if not built:
        raise ValueError(f"{annotation} declares {names}, each refusing {value!r} as its record")
    raise ValueError(f"{annotation} declares {names}, each of whose fields hold {sorted(value)}: write the "
                     f"class record {{'class': ..., 'fields': {{...}}}} that names one")


def _holds_record(member: type, value: Mapping, *, saved: bool = False) -> bool:
    """Whether the dataclass `member` reads `value` as its record (`_declared`): its every key a field the
    class records, its every required field among them; with `saved`, as `_declared` reads a saved record,
    a field its class lost since left out (`_records.saved_fields`)."""
    if not dataclasses.is_dataclass(member):
        return False
    names, required = _record_fields(member)
    held = _records.saved_fields(member, value, names) if saved else value
    return set(held) <= set(names) and set(required) <= set(held)


def _record_fields(member: type) -> tuple[list[str], list[str]]:
    """The fields the dataclass `member`'s record holds, and those of them it must (no default): a field
    marked `metadata={"record": False}` is a binding the value picks up at runtime, not something a record
    carries."""
    declared = [field for field in dataclasses.fields(member) if _recorded(field)]
    required = [field.name for field in declared
                if field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING]
    return sorted(field.name for field in declared), required


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
    if annotation is JSON:
        # Data a record holds as written, whatever it spells (a chat message's `{"function": ...}`).
        return configured(value)
    # An alias resolved first, so a bound member's alias (`type Predictor = functools.partial[...]`) reads
    # as the partial it names, and an Annotated's type read as the type, as its writer reads it
    # (`_admits_partial`).
    annotation = _resolved(annotation)
    if typing.get_origin(annotation) is functools.partial or annotation is functools.partial:
        return _bound(value, typing.get_args(annotation))
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        # An enum's member is recorded as its value, an IntEnum's its int (`to_record`).
        return annotation(value)
    if annotation is datetime.timedelta and isinstance(value, str):
        # A duration is recorded as its text (`_records.recorded_duration`), read back by its reader.
        return _records.duration(value)
    if (isinstance(annotation, type) and annotation in _SCALARS
            and not isinstance(value, (np.generic, np.ndarray, jax.Array))):
        # A field declaring a scalar holds one of its type (`dew._records`' readers), kept as written: a
        # null, a list or a record (a function's among them) is refused as a mistyped scalar is; a NumPy
        # or JAX value, which a module built in Python may hold, is the field's as built.
        _SCALARS[annotation](value, name or annotation.__name__)
        return configured(value)
    if (isinstance(value, Mapping) and set(value) == {"function"} and isinstance(value["function"], str)
            and _admits_function(annotation)):
        return imported(value["function"])
    if (isinstance(value, Mapping) and annotation not in (None, object)
            and typing.get_origin(_unwrapped(annotation)) not in _MAPPINGS and _sharing(value) is not None):
        return _shared(annotation, value, dtypes=dtypes)
    if typing.get_origin(annotation) in (Union, types.UnionType):
        if value is None:
            return value
        inner = [member for member in _union_members(annotation) if member is not type(None)]
        if len(inner) == 1:
            return _rebuilt(inner[0], value, dtypes=dtypes)
        # A class record builds one of the union's classes; a JSON union (`dew._records.JSON`) declares
        # none but scalars and keeps a mapping, a class record among them, as the record it is.
        classes = tuple(member for member in inner
                        if isinstance(member, type) and member not in (bool, int, float, str))
        mappings = [member for member in inner if typing.get_origin(member) in _MAPPINGS]
        named = class_record(value) if isinstance(value, Mapping) else None
        if named is not None and (classes or not mappings):
            if not classes:
                raise ValueError(f"{annotation} declares no class, so nothing builds the record {value!r}")
            return _built(_member(named[0], classes), named[1], dtypes=dtypes)
        sequences = [member for member in inner if member in (tuple, Sequence, list)
                     or typing.get_origin(member) in (tuple, Sequence, list, MutableSequence)]
        if isinstance(value, list) and len(sequences) == 1:
            # A record's list is the union's one sequence member (`Files.url: str | tuple[str, ...]`).
            return _rebuilt(sequences[0], value, dtypes=dtypes)
        if isinstance(value, Mapping) and not mappings:
            return _unnamed(annotation, classes, value, dtypes=dtypes)
        return configured(value)
    if isinstance(value, Mapping):
        if typing.get_origin(annotation) is Callable or annotation is Callable:
            # A callable field, such as a Sequential's layers or a loss, takes
            # a record of a class whose instances are called, whatever class.
            named = class_record(value)
            if named is not None:
                member = _member(named[0], (object,))
                if not _owned(member, called=True):
                    raise ValueError(
                        f"{named[0]!r} names {member!r}; a callable field's class record names a class "
                        f"whose instances are called, of Dew's or of a trusted package (trust=, --trust), "
                        f"and a function is the record {{'function': ...}}")
                return _built(member, named[1], dtypes=dtypes)
        nominal = nominal_class(annotation)
        if nominal is not None and nominal is not object:
            # A field typed with a class, or a generic of one, takes a record of the class itself or
            # of any class derived from it or function returning it; `object`
            # declares nothing, and its record stays the record it is.
            named = class_record(value)
            if named is not None:
                return _built(_member(named[0], (nominal,)), named[1], dtypes=dtypes)
            if dataclasses.is_dataclass(nominal):
                # A record reads as its kind's own class: a command line's subcommands are its parser's
                # alone (`RunConfig.cli`, `Bound.flagged`), so a run saved and read back is the run.
                return _construct(nominal, value, dtypes=dtypes)
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


_SCALARS: Mapping[type, Callable[[object, str], object]] = {
    bool: boolean, int: integer, float: real, str: text}
"""The reader of each scalar type a field declares (`_rebuilt`), each refusing a value of another."""


def nominal_class(annotation: Annotation) -> type | None:
    """The class `annotation` names: a class itself, or a subscripted generic's origin (`Base[int]`'s
    `Base`), which every reader of a field's class takes alike (`_rebuilt`, `_read`); None for a union,
    a container generic (`Mapping[...]`, `tuple[...]`, whose entries are walked) or no class."""
    if isinstance(annotation, type):
        return annotation
    origin = typing.get_origin(annotation)
    if not isinstance(origin, type) or issubclass(origin, (Mapping, Sequence, tuple)):
        return None
    return origin


def owned_built(record: JSON, *, called: bool, what: str) -> Configured:
    """What a class record (`class_record`) builds where it names a class of Dew's or of a trusted package
    (`_owned`; with `called`, one whose instances are called), read as `from_record` reads a record (a
    trusted package's module imported where the registry reaches it, every module it shares built once, a
    `dtype` field kept as its record states it): a value a record carries beside the fields any class
    declares (a run's own processor). Any other record (a function's among them, which builds by running
    it) is refused, as is a class no trust admits, naming `what` and `trust=`."""
    named = class_record(_records.record(record, what))
    if named is None:
        raise ValueError(f"{what} is recorded as a class record, {{'class': ..., 'fields': ...}}; got "
                         f"{record!r}")
    member = imported(named[0])
    if not _owned(member, called=called):
        called_so = " whose instances are called" if called else ""
        raise ValueError(f"{what} names {named[0]!r}, which is no class of Dew's or of a package the load "
                         f"trusts{called_so} (trust=, --trust)")
    with _reading(_records.record(record, what)):
        return _built(member, named[1], dtypes=False)


@dataclasses.dataclass
class _Written:
    """The record being written: the Flax modules it holds, by identity, each
    with its record; the references to each one it holds more than once; and
    the numbers that data shaped like a shared module's record takes
    (`_sharing`). The shared modules are numbered once the record is whole,
    past every number its data takes, so a reader never takes one for the
    other (`_Shared`)."""

    modules: dict[int, tuple[nn.Module, dict[str, JSON]]] = dataclasses.field(default_factory=dict)
    references: dict[int, list[dict[str, JSON]]] = dataclasses.field(default_factory=dict)
    taken: set[int] = dataclasses.field(default_factory=set)

    def number(self) -> None:
        """Number each shared module's record and its references, in the order they were first referred to."""
        free = (number for number in itertools.count() if number not in self.taken)
        for key, references in self.references.items():
            number = next(free)
            self.modules[key][1]["share"] = number
            for reference in references:
                reference["shared"] = number


_WRITTEN: contextvars.ContextVar[_Written | None] = contextvars.ContextVar("_WRITTEN", default=None)
"""The record being written."""


@contextlib.contextmanager
def recording() -> Iterator[_Written]:
    """One record being written: a module held twice in it is written once
    and referred to after (`to_record`). The outermost call opens the record;
    one inside it writes into the same record."""
    held = _WRITTEN.get()
    if held is not None:
        yield held
        return
    opened = _Written()
    token = _WRITTEN.set(opened)
    try:
        yield opened
        opened.number()
    finally:
        _WRITTEN.reset(token)


def _referred(written: _Written, key: int) -> JSON:
    """A reference to a module the record already holds: its first record
    becomes a class record, in place, and both take the module's number when
    the record is whole (`_Written.number`)."""
    module, record = written.modules[key]
    if key not in written.references:
        if "class" not in record:
            fields = dict(record)
            record.clear()
            record.update({"class": import_path(type(module)), "fields": fields})
        written.references[key] = []
    reference: dict[str, JSON] = {}
    written.references[key].append(reference)
    return reference


@dataclasses.dataclass
class _Shared:
    """The shared modules of the record being read: by number, every mapping
    shaped like one's definition, the module once it is built, and the record
    it was built from.

    The shape alone does not make a definition: a field typed `object` or a
    plain mapping keeps such a mapping as the data it is (`_rebuilt`). So the
    scan only collects candidates, and a number becomes a module where the
    walk reads it at a module's place: a definition is built from itself, and
    a reference read before its definition (a record written with sorted
    keys) from the one candidate it can mean. `to_record` never gives data and
    a shared module one number, so only a record made by hand can leave a
    reference more than one.
    """

    candidates: dict[int, list[Mapping[str, object]]]
    built: dict[int, Configured] = dataclasses.field(default_factory=dict)
    sources: dict[int, Mapping[str, object]] = dataclasses.field(default_factory=dict)


_READING: contextvars.ContextVar[_Shared | None] = contextvars.ContextVar("_READING", default=None)


@contextlib.contextmanager
def _reading(record: Configured | Mapping[str, object]) -> Iterator[None]:
    """One record being read: every module it shares (`to_record`) is built
    once, where the record first holds it or refers to it. The outermost call
    opens the record; one inside it reads the same record."""
    if _READING.get() is not None:
        yield
        return
    candidates: dict[int, list[Mapping[str, object]]] = {}

    def scan(node) -> None:  # any value a record holds, a record of records or not
        if isinstance(node, Mapping):
            number = _sharing(node)
            if number is not None and "share" in node:
                candidates.setdefault(number, []).append(node)
            for entry in node.values():
                scan(entry)
        elif isinstance(node, (list, tuple)):
            for entry in node:
                scan(entry)

    scan(record)
    token = _READING.set(_Shared(candidates))
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
    if "share" in value:
        source = value
    elif number in shared.built:
        return _declared_as(annotation, shared.built[number], number)
    else:
        found = shared.candidates.get(number, [])
        if len(found) != 1:
            raise ValueError(f"the record refers to its shared module {number} before defining it, and "
                             f"holds {len(found) or 'no'} records numbered {number}")
        source = found[0]
    if number in shared.built:
        if shared.sources[number] != source:
            raise ValueError(f"the record defines its shared module {number} twice")
        return _declared_as(annotation, shared.built[number], number)
    shared.sources[number] = source
    shared.built[number] = _rebuilt(annotation, {str(key): entry for key, entry in source.items()
                                                 if key != "share"}, dtypes=dtypes)
    return shared.built[number]


def _declared_as(annotation: Annotation, built: Configured, number: int) -> Configured:
    """`built`, the shared module `number` read before, where the field reading it again declares it
    (`annotation`'s class, or one of its union's); refused where it declares only classes it is none of, so
    a union's trial whose field declares another kind refuses the record as a module built for it would
    (the API02 review: a module built for one field read as another's, and two trials built)."""
    members = (_union_members(annotation) if typing.get_origin(annotation) in (Union, types.UnionType)
               else (resolve_alias(annotation),))
    declared = [member for member in members if member is not type(None)]
    if any(member is typing.Any or member is object for member in declared):
        return built
    # A parameterized class (`Sequence[Module]`) by its class (`typing.get_origin`), as `isinstance` reads it.
    nominal = [typing.get_origin(member) or member for member in declared]
    classes = tuple(member for member in nominal if isinstance(member, type) and not is_protocol(member))
    if declared and len(classes) == len(declared) and not isinstance(built, classes):
        raise ValueError(f"the shared module {number} is a {type(built).__qualname__}, which the field "
                         f"reading it again ({annotation}) does not declare")
    return built

def component[Base](base: type[Base], record: Mapping[str, object]
                    ) -> tuple[type[Base], Mapping[str, object]]:
    """The class a component's record, `{"class": ..., "fields": {...}}` (`to_record`, `Recorded`), names
    by an alias or its import path, one that is `base` or derives from it, and the record's fields as
    written: the lookup `from_record` uses for a class record (`_member`). A component built around parameters
    (an encoder, an autoencoder, a decision head) reads its fields against its own constructor
    (`read_arguments`) and binds them itself. A record of another shape, of a function, or a protocol
    `base` (which no class need derive from) is refused."""
    if is_protocol(base):
        raise TypeError(f"a component's base is a class it derives from, not the protocol {base.__name__}")
    named = class_record(record)
    if named is None:
        raise ValueError(f"a {base.__name__} record is {{'class': ..., 'fields': {{...}}}}, not {record!r}")
    member = _member(named[0], (base,))
    if not isinstance(member, type):
        raise TypeError(f"{named[0]!r} names the function {member!r}; a {base.__name__} record names its "
                        f"class")
    return member, named[1]


def _of_kind(member: Callable[..., Configured], held: tuple[type, ...]) -> bool:
    """Whether `member` is one of `held`, derives from one, or is a function declared to return one."""
    if isinstance(member, type):
        return any(_derives(member, base) for base in held)
    return callable(member) and any(_returns(member, base) for base in held)


def _member(name: str, held: tuple[type, ...]) -> Callable[..., Configured]:
    """The class or function a record's `class` names, by import path or by
    an alias of any kind, that is one of `held`, derives from one, or is a
    function declared to return one."""
    found = ([imported(name)] if ":" in name else
             [imported(kind.paths[name]) for kind in KINDS if name in kind.paths])
    fitting = [member for member in found if _of_kind(member, held)]
    if len(fitting) == 1:
        return fitting[0]
    wanted = " or ".join(base.__name__ for base in held)
    if not found:
        raise ValueError(f"no class is named {name!r}; a record names one by an alias or "
                         f"its import path, `module:Class`")
    if any(is_protocol(base) for base in held) and all(isinstance(member, type) for member in found):
        package = found[0].__module__.split(".")[0]
        raise ValueError(f"{name!r} names a {wanted} outside Dew, which a record builds only from a "
                         f"trusted package (trust=({package!r},), --trust {package})")
    raise ValueError(f"{name!r} names {', '.join(map(repr, found))}, not one {wanted}")


def _resolved(annotation: Annotation) -> Annotation:
    """`annotation` as a record is read against it: its aliases resolved and an Annotated's type taken, at
    any depth of the two, the one normalization the reader (`_rebuilt`) and the writer (`_admits_partial`)
    read a declaration by."""
    held = resolve_alias(annotation)
    while typing.get_origin(held) is typing.Annotated:
        held = resolve_alias(typing.get_args(held)[0])
    return held


def _flattened(annotation: Annotation) -> Annotation:
    """`annotation` as its reader dispatches a value on it (`_rebuilt`): normalized (`_resolved`), and a union
    of one member besides None that member, flattened in turn; any other union as it is."""
    held = _resolved(annotation)
    if typing.get_origin(held) in (Union, types.UnionType):
        inner = [member for member in _union_members(held) if member is not type(None)]
        return _flattened(inner[0]) if len(inner) == 1 else held
    return held


def _admits_partial(annotation: Annotation, partial: functools.partial) -> bool:
    """Whether a field declaring `annotation` reads `partial` back from its class record (`_bound`), as its
    reader flattens it (`_flattened`): a partial its declaration takes, binding a member of the kind it
    names, as the reader checks (`_of_kind`)."""
    held = _flattened(annotation)
    if not (held is functools.partial or typing.get_origin(held) is functools.partial):
        return False
    kinds = tuple(each for each in typing.get_args(held) if isinstance(each, type) and each is not Any)
    return not kinds or _of_kind(partial.func, kinds)


def _unread_partial(value: Configured, annotation: Annotation) -> functools.partial | None:
    """The first partial `value` holds, at any depth of its containers, whose place `annotation` declares
    reads back no bound member (`_admits_partial`), each entry against its own type (`entry_types`; an
    entry of no type it states takes none); None where every one reads back."""
    if isinstance(value, functools.partial):
        return None if _admits_partial(annotation, value) else value
    if not isinstance(value, Mapping | list | tuple):
        return None
    try:
        # The container's entries typed as its reader types them, its alias and Optional flattened first.
        types_ = entry_types(_flattened(annotation), len(value))
    except ValueError:
        types_ = [None] * len(value)
    entries = value.values() if isinstance(value, Mapping) else value
    return next((found for entry, held in zip(entries, types_, strict=True)
                 if (found := _unread_partial(entry, held)) is not None), None)


def _argument_type(member: type | Callable[..., Configured], name: str) -> Annotation:
    """The annotation `member` declares its argument `name` with, as its record reads that argument
    (`member_fields`), None where it declares none its module resolves; no default built."""
    held = record_class(member)
    if held is not None:
        return declared_type(held, name)
    found = parameters(member)[0].get(name)
    if found is None:
        return None
    try:
        return parameter_type(found[1], name)
    except ValueError:
        return None


def _bound(value: object, declared: tuple[Annotation, ...]) -> functools.partial:
    """A member bound by keyword (`Bound.bound`) from its class record, `{"class": ..., "fields": {...}}`,
    each field read against the member's own annotation; or the partial given, its arguments as they are.
    A member the partial's annotation names (`functools.partial[JepaPredictor]`) is one of that class,
    checked before its arguments are read, a partial given among them; `Any` names none, and any other
    kind is its record's to check (`Bound._member`)."""
    held = tuple(each for each in declared if isinstance(each, type) and each is not Any)
    if isinstance(value, functools.partial):
        if held and not _of_kind(value.func, held):
            raise ValueError(f"{value!r} binds no {' or '.join(each.__name__ for each in held)}")
        return value
    named = class_record(value) if isinstance(value, Mapping) else None
    if named is None:
        raise ValueError(f"a bound member is the record that names it, {{'class': ..., 'fields': {{...}}}}, "
                         f"not {value!r}")
    member = _member(named[0], held or (object,))
    # Read here: a partial a caller calls hands its member the arguments it takes (a config's bound member
    # holds its records until it is built, `Bound.read`).
    return _Bound(member, **read_arguments(member, named[1]))


def _taken(member: Callable[..., Configured], fields: Mapping[str, object], *,
           dtypes: bool) -> Mapping[str, object]:
    """A record's `fields` as `member` takes them: a dataclass's as it declares
    them (`_declared`), a function's or another class's as its parameters
    (`_arguments`)."""
    held = record_class(member)
    return (_declared(held, fields, dtypes=dtypes) if held is not None
            else _arguments(member, fields, dtypes=dtypes))


def _built(member: Callable[..., Configured], fields: Mapping[str, object], *, dtypes: bool) -> Configured:
    """What `member` builds from a record of its fields, checked as a configured value (`configured`): a
    dataclass built from its fields as a field's own class is (`_construct`, a bound member's record by its
    own shape among them), anything else from its parameters (`_arguments`)."""
    # A bound member's record read by its own shape before any dataclass's fields (`_construct`, the one
    # dispatch to `Bound.read`): a kind of its own constructor is no dataclass `record_class` names.
    held = member if isinstance(member, type) and issubclass(member, Bound) else record_class(member)
    if held is not None:
        return _construct(held, fields, dtypes=dtypes)
    built = member(**_arguments(member, fields, dtypes=dtypes))
    # A plain class of Dew's or a trusted package's is the value its record
    # describes (`to_record` writes it from its constructor's parameters).
    plain = isinstance(member, type) and _owned(member, called=False)
    return built if plain else configured(built)


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


def record_class(member: Callable[..., Configured]) -> type | None:
    """`member` when it is a dataclass built from its fields, whose fields a
    record names; None for a function, a plain class, or a dataclass whose own
    constructor takes other arguments (a metric over its base's fields), each
    built from its parameters (`parameters`)."""
    if not (isinstance(member, type) and dataclasses.is_dataclass(member)):
        return None
    # The generated constructor takes the init fields, InitVars among them, in any order
    # keyword-only ones give it; a constructor of the class's own takes other names. The names
    # are read from its code (under any wrapper, as Flax wraps a module's), not its signature,
    # whose annotations may name a type not yet bound.
    constructor = inspect.unwrap(member.__init__)
    if not isinstance(constructor, types.FunctionType):
        return None
    code = constructor.__code__
    taken = set(code.co_varnames[1:code.co_argcount + code.co_kwonlyargcount])
    fields = {f.name for f in dataclasses.fields(member) if f.init}
    return member if fields <= taken <= set(member.__dataclass_fields__) else None


def _construct(member: type, fields: Mapping[str, object], *, dtypes: bool) -> Configured:
    """Build the dataclass `member` from a record of its fields; a bound member's record (`Bound`) by its
    own shape (`Bound.read`)."""
    if issubclass(member, Bound):
        return member.read(fields, dtypes=dtypes)
    return configured(member(**_declared(member, fields, dtypes=dtypes)))


def _declared(member: type, fields: Mapping[str, object], *, dtypes: bool) -> Mapping[str, object]:
    """The record's fields as `member` declares them, each walked against its
    own annotation. A field the record lacks takes its declared default; a
    field `member` does not declare, or a required one the record lacks,
    raises. A field marked `metadata={"record": False}` is a binding the
    value picks up at runtime, not something a record carries."""
    names, required = _record_fields(member)
    # A saved record's field its class lost since, read without (`_records.saved_fields`).
    fields = _records.saved_fields(member, fields, names)
    unknown = sorted(set(fields) - set(names))
    missing = [name for name in required if name not in fields]
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
    # A saved record's field its class lost since, read without (`_records.saved_fields`), as `_declared`
    # reads a dataclass's: a constructor taking `**kwargs` forwards only what its parameters take. A
    # function's record names no class, and none of its fields is retired.
    if isinstance(function, type):
        fields = _records.saved_fields(function, fields, named)
    unknown = [] if open_ended else sorted(set(fields) - set(named))
    missing = [name for name, (parameter, _) in named.items()
               if name not in fields and parameter.default is parameter.empty]
    if unknown or missing:
        raise ValueError(f"{function.__name__} does not match the record: unknown fields {unknown}, "
                         f"missing fields {missing}; its parameters are {sorted(named)}")
    return {name: _read(function, name, value, dtypes=dtypes) for name, value in fields.items()}


def read_arguments(member: type | Callable[..., Configured], record: Mapping[str, object], *,
                   declared: Mapping[str, Annotation] | None = None) -> Mapping[str, object]:
    """`record`'s fields of the dataclass `member`, or arguments of the
    function or constructor `member`, each read as `build` reads it, without
    asking whether they are all that `member` takes; one `declared` names is
    read against that annotation instead, as a generic constructor's own
    parameter (`model: M`) cannot say which class it takes. One record is read
    in one reading, so a module it holds twice is built once (`recording`)."""
    given = declared or {}
    with _reading(record):
        return {name: (_rebuilt(given[name], value, dtypes=True, name=name) if name in given
                       else _read(member, name, value, dtypes=True)) for name, value in record.items()}


def _read(member: type | Callable[..., Configured], name: str, value, *, dtypes: bool):
    """One field of the dataclass `member`, walked against its own annotation;
    or one argument of the function or constructor `member`, a record or a
    JSON list read against its parameter's annotation and a dtype's name the
    dtype, any other value, a scalar or a value the caller built, as given."""
    held = record_class(member)
    if held is not None:
        declared = declared_type(held, name)
        nominal = nominal_class(declared)
        if (not (dtypes and name == "dtype") and nominal is not None
                and nominal not in (Any, object, bool, int, float, str) and not is_protocol(nominal)
                and not issubclass(nominal, (Mapping, Sequence)) and isinstance(value, nominal)):
            # A value the caller built, of the field's own nominal class (an encoder a binder made),
            # is kept as it is, whatever its kind; `Any` and `object` name no class to check, and a
            # scalar or a dtype's name is read as before (`_rebuilt`).
            return value
        return _rebuilt(declared, configured(value), dtypes=dtypes, name=name)
    if not (isinstance(value, (Mapping, list, str)) or (dtypes and name == "dtype")):
        return value
    named = parameters(member)[0]
    # A text is kept as given but where its parameter declares a duration, recorded as its text
    # (`_records.recorded_duration`), optional or not.
    if isinstance(value, str) and not (dtypes and name == "dtype") and not _declares_duration(named, name):
        return value
    return _rebuilt(parameter_type(named[name][1], name) if name in named else None, value, dtypes=dtypes,
                    name=name)


def _declares_duration(named: Mapping[str, tuple], name: str) -> bool:
    """Whether parameter `name` of `named` (`parameters`) declares a duration, optional or not; False where
    its annotation names what its module does not import at runtime, whose text is passed as given."""
    if name not in named:
        return False
    try:
        declared = parameter_type(named[name][1], name)
    except ValueError:
        # An annotation no record is read against (`parameter_type`): the text is the argument as written.
        return False
    return datetime.timedelta in _union_members(declared)


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
            annotation = parameter_type(named[name][1], name) if name in named else None
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


def parameter_type(function: Callable[..., Configured], name: str) -> Annotation:
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
    # The names the annotation was written among are its module's: a linen module's constructor is written
    # by Flax's dataclass transform, whose globals are Flax's own. Code made by exec names a module that may
    # not be loaded, and its globals are where it was written.
    module = sys.modules.get(underlying.__module__)
    namespace = underlying.__globals__ if module is None else vars(module)
    try:
        return typing.get_type_hints(selected, globalns=dict(namespace))[name]
    except NameError as error:
        raise ValueError(f"{function.__name__}'s annotation of {name} names what {underlying.__module__} "
                         f"does not import at runtime ({error}); a record is read against it, so "
                         f"import it there") from error


def _returns(member: Callable[..., Configured], held: type) -> bool:
    """Whether the function `member` is declared to return `held` or a class
    derived from it."""
    returned = parameter_type(member, "return")
    return isinstance(returned, type) and _derives(returned, held)


def _derives(member: type, base: type) -> bool:
    """Whether `member` is `base` or derives from it. A protocol declares an
    interface and no class to derive from, so a record's class for one is a
    class of Dew's or of a trusted package (`_owned`), as a protocol kind's
    aliases are."""
    return _owned(member, called=False) if is_protocol(base) else issubclass(member, base)


def record_fields(value: DataclassInstance, owner: type, *, without: Collection[str] = ()) -> dict[str, JSON]:
    """The record of a dataclass value's constructor fields, each as `owner`
    declares it, at every level of a nested value alike; the fields named
    `without` left out, where another record carries them.

    A field still holding its declared callable default, such as a Flax
    layer's initializer closure, is left out: no record can name a closure,
    and the class supplies the same default again when the record is read.
    A Flax module's `parent` is its binding, never recorded, and its `name`
    is recorded where it was given one: a parent that keeps its submodules'
    own names (`preserve_adopted_names`) finds its weights under them."""
    from flax import linen as nn

    fields = {}
    with recording():
        for field in dataclasses.fields(value):
            held = getattr(value, field.name)
            binding = isinstance(value, nn.Module) and (field.name == 'parent' or (field.name == 'name'
                                                                                    and held is None))
            if (_recorded(field) and not binding and field.name not in without
                    and not (callable(held) and held is field.default)):
                fields[field.name] = to_record(held, declared_type(owner, field.name))
    return fields


@runtime_checkable
class Wrapper(Protocol):
    """A record that wraps a module in a class of its own and makes that
    class again over a base: a LoRA adapter (`dew.lora.AdapterRecord`), a
    run's quantized training or a checkpoint's input quantization
    (`dew.quantization`)."""

    def apply(self, model: nn.Module) -> nn.Module: ...

    def keeps_products_linear(self, train: bool) -> bool:
        """Whether every product the wrapper holds stays the linear map of its input by its kernel in a call,
        a training one where `train`: an adapter's where it drops nothing there, a quantization's never.
        A layer that reads forks along a linear path at no more work than one row (an in-context memory's
        recurrence) unfolds them under a wrapper that does not (`MixerBase.fork_sharing_refusal`)."""
        ...


@runtime_checkable
class Recorded(Protocol):
    """A component a run records by the constructor view it is rebuilt from, not by its attributes: an
    encoder's checkpoint and settings, an autoencoder's module and normalization, never the parameters
    it was built around (`to_record` writes it as `{"class", "fields"}`, `component` finds its class).
    `to_record` asks for this view before it walks a dataclass's fields (`record_fields`), so a
    dataclass that is `Recorded` is recorded by its view alone."""

    def recorded_fields(self) -> Mapping[str, object]: ...


@runtime_checkable
class WrapperClass(Protocol):
    """A module class a `Wrapper` made over `_dew_wrapped`; `_dew_wrapper`
    returns that wrapper, which a model record keeps (`ModelConfig.wrappers`)."""

    _dew_wrapped: ClassVar[type]

    @staticmethod
    def _dew_wrapper() -> Wrapper: ...


def constructor_fields(member: type) -> tuple[dataclasses.Field, ...]:
    """A dataclass's fields its constructor takes, a Flax module's `parent` and `name` aside: what
    rebuilding an instance under another class (a wrapper's, `wrapper_layers`) passes on."""
    return tuple(field for field in dataclasses.fields(member)
                 if field.init and field.name not in ("parent", "name"))


def computing_fields(member: type | Callable[..., Configured], *, kept: Collection[str] = ()
                     ) -> tuple[str, ...]:
    """The fields of `member`, a model's class, that say what it computes (`member_fields`): every one but
    those it marks `dew._records.RUNS`, which say how it runs or trains (its dtype, precision and kernels,
    remat, dropout, the initialization loaded weights replace), a field also marked one of the metadata
    keys `kept` (`_records.NUMBERS`, `_records.ROUNDS`) among them. The one statement of what makes two
    models one (`computing_differences`)."""
    held = record_class(member)
    runs = () if held is None else {field.name for field in constructor_fields(held)
                                    if field.metadata.get(_records.RUNS)
                                    and not any(field.metadata.get(key) for key in kept)}
    return tuple(name for name, _, _ in member_fields(member) if name not in runs)


def computing_differences(member: type | Callable[..., Configured], ours: Mapping[str, object],
                          theirs: Mapping[str, object], *, kept: Collection[str] = ()
                          ) -> dict[str, tuple[object, object]]:
    """Each field of `member` that says what a model computes (`computing_fields`, with `kept`) on which
    `ours` and `theirs`, two models' fields by name, differ, as its two values, ours first; a field one of
    them lacks is None there. Empty where the two are one model, whatever runs or trains them: a saved
    run's weights average with another's (`dew.checkpoints.averaged`), an export reads back as the model
    (`dew.interop.hf_decoders.refuse_rebuilt`), a run resumes on other devices (`RunConfig.check_resume`,
    which keeps the numbers it trains in)."""
    return {name: (ours.get(name), theirs.get(name)) for name in computing_fields(member, kept=kept)
            if ours.get(name) != theirs.get(name)}


def rebuilt[Built](member: Callable[..., Built], model: DataclassInstance) -> Built:
    """`model` built again by `member`, its own class or one that wraps it (`wrapper_layers`), from its
    constructor fields (`constructor_fields`) as it holds them; `rebuilt(dict, model)` is those fields
    by name. `model` is any dataclass instance, a Flax module among them."""
    return member(**{field.name: getattr(model, field.name) for field in constructor_fields(type(model))})


def wrapper_layers(model_type: type) -> tuple[type, tuple[WrapperClass, ...]]:
    """The class `model_type`'s wrappers wrap and its wrapper classes, the
    outermost first.

    Each layer is a subclass of the one inside it, so it inherits the inner
    layer's markers too; a layer counts once, where its own class declares
    `_dew_wrapped`."""
    layers: list[WrapperClass] = []
    while isinstance(model_type, WrapperClass) and "_dew_wrapped" in vars(model_type):
        layers.append(model_type)
        model_type = model_type._dew_wrapped
    return model_type, tuple(layers)


def recorded_names(owner: type) -> tuple[str, ...]:
    """The fields a record of the dataclass `owner` holds (`record_fields` writes them, `from_record`
    reads them): its constructor's, none marked unrecorded (`_recorded`)."""
    return tuple(field.name for field in dataclasses.fields(owner) if _recorded(field))


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
    if value is None or isinstance(value, (bool, int, float, str, bytes, type, Enum, np.dtype,
                                           datetime.timedelta)):
        return value
    if isinstance(value, (np.ndarray, np.generic, jax.Array)):
        return value
    if dataclasses.is_dataclass(value) or isinstance(value, (Mapping, Sequence)) or callable(value):
        return value
    raise ValueError(f"{value!r} is not a value a member field carries")


DtypeName = Literal["float32", "bfloat16", "float16", "float64"]
_DTYPES: dict[DtypeName, DTypeLike] = {name: getattr(jnp, name) for name in get_args(DtypeName)}
"""Each storage dtype a config names, by its name: `DtypeName`'s, the one declaration. float64 is
a model's own dtype where it computes in float64 end to end (Dory, nGPT)."""


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
    name = storage_name(value)
    if name is None:
        raise Unrecordable(f"{value!r} is not a dtype a config can name")
    return name


def storage_name(value: DTypeLike) -> DtypeName | None:
    """The name of the storage dtype `value` is (`DtypeName`), or None for any other dtype: a
    checkpoint's integer or float8 tensors are not storage a config names."""
    return next((name for name, dtype in _DTYPES.items() if jnp.dtype(value) == jnp.dtype(dtype)), None)


def with_dtype(name: str, fields: Mapping[str, object], dtype: str | None) -> Mapping[str, object]:
    """`fields`, the model `name` builds from, computing in `dtype` (`with_declared`); None leaves `fields`
    as they are."""
    if dtype is None:
        return fields
    return with_declared(name, fields, "dtype", dtype, built=resolve_dtype(dtype))


def with_declared(name: str, fields: Mapping[str, object], field: str, value: str | int, *,
                  built: DTypeLike | None = None) -> Mapping[str, object]:
    """`fields`, the model `name` builds from, with `value` in its field `field`: written into the model's
    own `field`, and into each model part it holds that declares one, as a record or built (`built`, the
    value a built part takes, `value` where None), so a composite that declares none (DiffusionGemma around
    its text decoder) takes it in its parts. A model with neither refuses it."""
    from flax import linen as nn

    member = models[name]
    assert isinstance(member, type)
    parts: dict[str, object] = ({field: value}
                              if field in {declared.name for declared in dataclasses.fields(member)} else {})
    for key, held in fields.items():
        part = _part_type(member, key, configured(held))
        if part is None or field not in {declared.name for declared in dataclasses.fields(part)}:
            continue
        if isinstance(held, nn.Module):
            parts[key] = dataclasses.replace(held, **{field: value if built is None else built})
        elif isinstance(held, Mapping) and (named := class_record(held)) is not None:
            parts[key] = {"class": named[0], "fields": {**named[1], field: value}}
        elif isinstance(held, Mapping):
            parts[key] = {**held, field: value}
    if not parts:
        raise ValueError(f"{member.__qualname__} declares no {field} field and holds no part that does, so "
                         f"it cannot take {field}={value!r}")
    return {**fields, **parts}


def _part_type(member: type, field: str, value: Configured) -> type | None:
    """The model class a composite's `field` holds, built, as a class record,
    or as the record of the class the field declares; None for anything else."""
    from flax import linen as nn

    if isinstance(value, nn.Module):
        return type(value)
    declared = tuple(part for part in _union_members(declared_type(member, field))
                     if isinstance(part, type) and issubclass(part, nn.Module))
    if not declared:
        return None
    named = class_record(value) if isinstance(value, Mapping) else None
    if named is not None:
        held = _member(named[0], declared)
        return held if isinstance(held, type) and issubclass(held, nn.Module) else None
    if isinstance(value, Mapping) and len(declared) == 1:
        return declared[0]
    return None


# Each kind's aliases, for the command line and the records a person writes.
source_kinds: Aliases[type[SourceKind], SourceKind] = Aliases("source", {
    "decision": "dew.decision.sources:DecisionSource",
    "slot": "dew.decision.slot_source:SlotSource",
}, base="dew.interop.composite:SourceKind")


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
    "cosine_continuous": "dew.diffusion.presets:CosineContinuous",
    "cosine_general": "dew.diffusion.presets:CosineGeneral",
    "edm": "dew.diffusion.presets:EDM",
    "exp": "dew.diffusion.presets:Exp",
    "flow": "dew.diffusion.presets:Flow",
    "jit": "dew.diffusion.presets:JiT",
    "karras": "dew.diffusion.presets:Karras",
    "mdlm": "dew.diffusion.discrete:MDLM",
    "mean_flow": "dew.diffusion.presets:MeanFlow",
    "shortcut": "dew.diffusion.presets:Shortcut",
    "sqrt": "dew.diffusion.presets:Sqrt",
})
solvers: Aliases[type[Solver], Solver] = Aliases("solver", {
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
    "confident": "dew.diffusion.discrete:Confident",
})
# An encoder's raw input is its own (`ConditionEncoder[Raw, Encoded]`, Raw invariant as an input is): the
# table admits an encoder of any, which only the encoder's run reads.
encoders: Aliases[type[ConditionEncoder[Any]], ConditionEncoder[Any]] = Aliases("encoder", {
    "char_table": "dew.inputs.encoders:CharTable",
    "clip_text": "dew.interop.encoders:CLIPText",
    "diffusion_text": "dew.interop.conditioners:DiffusionConditioner",
    "hf_audio": "dew.interop.encoders:HFAudio",
    "hidden_states_text": "dew.interop.conditioners:HiddenStatesConditioner",
    "qwen_image_text": "dew.interop.conditioners:QwenImageConditioner",
    "t5": "dew.interop.encoders:T5Text",
    "wan_text": "dew.interop.conditioners:WanConditioner",
}, base="dew.inputs.condition:ConditionEncoder")
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
    "diffusion_nft": "dew.objectives.rl.nft:DiffusionNFTObjective",
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
    "block_sparse_attention": "dew.nn.block_sparse:BlockSparseAttentionMixer",
    "deepseek_v4": "dew.nn.deepseek_v4:DeepseekV4Mixer",
    "gated_delta_net": "dew.nn.mixers.gated_delta_net:GatedDeltaNetMixer",
    "kimi_delta_attention": "dew.nn.kda:KimiDeltaAttentionMixer",
    "kpool_sparse_attention": "dew.nn.dsa_kpool:KPoolSparseAttentionMixer",
    "lightning_attention": "dew.nn.mixers.lightning:LightningAttentionMixer",
    "mamba": "dew.nn.mixers.mamba:MambaMixer",
    "mamba2": "dew.nn.mixers.mamba2:Mamba2Mixer",
    "mamba3": "dew.nn.mixers.mamba3:Mamba3Mixer",
    "memory": "dew.nn.mixers.memory:InContextMemoryMixer",
    "mla": "dew.nn.mla:MLAMixer",
    "mlp": "dew.nn.mixers.mlp:MLPMixer",
    "parallel": "dew.nn.mixers.parallel:ParallelMixer",
    "qsa_attention": "dew.nn.mixers.attention:QSAAttentionMixer",
    "short_conv": "dew.nn.mixers.short_conv:ShortConvMixer",
}, base="dew.nn.mixer_base:MixerBase")
towers: Aliases[type[TowerBase], TowerBase] = Aliases("tower", {
    "deepseek_v41": "dew.nn.vision.deepseek_v41:DeepseekV41Vision",
    "gemma3n": "dew.nn.vision.gemma3n:Gemma3nVision",
    "gemma3n_audio": "dew.nn.audio:Gemma3nAudio",
    "gemma4": "dew.nn.vision.gemma4:Gemma4Vision",
    "gemma4_unified": "dew.nn.vision.gemma4_unified:Gemma4UnifiedVision",
    "gemma4_audio": "dew.nn.audio:Gemma4Audio",
    "inkling2": "dew.nn.vision.inkling:InklingVision",
    "lfm3_vl": "dew.nn.vision.lfm3:Lfm3VlVision",
    "llama4": "dew.nn.vision.llama4:Llama4Vision",
    "mimo": "dew.nn.vision.mimo:MiMoVision",
    "mimo_audio": "dew.nn.mimo_audio:MiMoAudio",
    "moonvit": "dew.nn.vision.moonvit:MoonViT",
    "muse_glimmer": "dew.nn.vision.muse_glimmer:MuseGlimmerVision",
    "qwen3_5": "dew.nn.vision.qwen35:Qwen35Vision",
    "siglip": "dew.nn.vision.siglip:SiglipVision",
}, base="dew.nn.vision.common:TowerBase")
projectors: Aliases[type[ProjectorBase], ProjectorBase] = Aliases("projector", {
    "deepseek_v41": "dew.nn.vision.deepseek_v41:DeepseekV41Projector",
    "gemma": "dew.nn.vision.siglip:GemmaProjector",
    "gemma3n": "dew.nn.vision.gemma3n:Gemma3nProjector",
    "gemma4": "dew.nn.vision.gemma4:Gemma4Projector",
    "inkling2": "dew.nn.vision.inkling:InklingProjector",
    "lfm3_vl": "dew.nn.vision.lfm3:Lfm3VlProjector",
    "llama4": "dew.nn.vision.llama4:Llama4Projector",
    "mimo": "dew.nn.vision.mimo:MiMoProjector",
    "moonvit": "dew.nn.vision.moonvit:MoonViTProjector",
    "moonvit_v2": "dew.nn.vision.moonvit:MoonViTProjectorV2",
    "muse_glimmer": "dew.nn.vision.muse_glimmer:MuseGlimmerProjector",
    "qwen3_5": "dew.nn.vision.qwen35:Qwen35Projector",
}, base="dew.nn.vision.common:ProjectorBase")
schedules: Aliases[type[ScheduleBase], ScheduleBase] = Aliases("schedule", {
    "cosine": "dew.optim:Cosine",
    "exponential": "dew.optim:Exponential",
    "linear": "dew.optim:Linear",
    "one_cycle": "dew.optim:OneCycle",
    "power": "dew.optim:Power",
}, base="dew.optim_schedules:ScheduleBase")
optimizers: Aliases[type[OptimizerSpec], OptimizerSpec] = Aliases("optimizer", {
    "adam": "dew.optim:Adam",
    "adamw": "dew.optim:AdamW",
    "gauss_newton": "dew.optim:GaussNewton",
    "lamb": "dew.optim:Lamb",
    "laprop": "dew.optim:LaProp",
    "muon": "dew.optim:Muon",
    "muonclip": "dew.optim:MuonClip",
    "optax": "dew.optim:Optax",
}, base="dew.optim:OptimizerSpec")
"""The optimizers a run's `OptimConfig.optimizer` names, each with its own settings
(`dew.optim.OptimizerSpec`)."""


# A run config's own class sits above the registry, which names it for the type checker alone. A run
# record names its class by this table's alias (`RunConfig.record`), apart from every other kind.
runs: Aliases[type[RunConfig], RunConfig] = Aliases("run", {
    "decision": "dew.decision.config:DecisionRunConfig",
    "diffusion": "dew.config.diffusion:DiffusionRunConfig",
    "env": "dew.config.env:EnvRunConfig",
    "jepa": "dew.config.jepa:JepaRunConfig",
    "lm": "dew.config.lm:LMRunConfig",
    "world_model": "dew.config.world_model:WorldModelRunConfig",
}, base="dew.config:RunConfig")


# An environment's types live in `dew.rl`, which only `dew.rl` and `dew.objectives.rl` import
# (docs/design/architecture.md), so the table's type names none; its base is read by path.
envs: Aliases[type | Callable[..., Configured], Any] = Aliases("env", {
    "flight": "dew.rl.flight:FlightEnv",
    "gate_race": "dew.rl.flight:GateRace",
    "playground": "dew.rl.playground:Playground.load",
    "skydreamer": "dew.rl.skydreamer:SkyDreamerRace",
}, base="dew.rl.env:Env")
"""The environments an environment run names (`dew.config.env.EnvConfig`): a JAX `Env` class, a
host `Simulator` class, or a function or classmethod that returns one; any other by its import path."""
rollouts: Aliases[type[DreamerRollout] | type[ControlRollout], DreamerRollout | ControlRollout] = Aliases(
    "rollout", {"control": "dew.objectives.rl.control:ControlRollout",
                "dreamer": "dew.objectives.rl.dreamer:DreamerRollout"})
"""The rollouts an environment run collects its batches with (`dew.config.env.RolloutConfig`)."""


KINDS: tuple[Aliases, ...] = (models, presets, solvers, encoders, metrics, objectives, mixers, towers,
                              projectors, schedules, optimizers)
"""Every kind, which a record's alias is looked up across."""

register = models.register
"""A decorator naming a user's model class in `models` (`Aliases.register`): `@dew.register("my_model")` over
an `nn.Module`, which a config's `{"class": "my_model"}` then builds. Every other kind registers through its
own table, `mixers.register("my_mixer")`."""

_JSON_OBJECT = tyro.constructors.PrimitiveConstructorSpec(
    nargs=1,
    metavar="JSON",
    instance_from_str=lambda args: json.loads(args[0]),
    is_instance=lambda value: isinstance(value, dict),
    str_from_instance=lambda value: [json.dumps(value)],
)
"""A command-line flag holding one JSON object, written as a single JSON string."""

JsonDict = Annotated[Mapping[str, object], _JSON_OBJECT]
"""The fields a model is built from, written as a single JSON string on the
command line. The registry knows which architecture takes which field and
narrows each one where it builds it, so the values are read there."""

JsonRecords = Annotated[Mapping[str, JSON], _JSON_OBJECT]
"""Records by name, written as a single JSON string on the command line and
read back as the records they are (`JSON` entries, `_rebuilt`), never as the
values they describe: a class's defaults as a run recorded them
(`Bound.defaults`)."""


class _Bound(functools.partial):
    """A member bound by keyword (`Bound.bound`), equal to another of the same member and arguments, as a
    record of it is: a run read back equals the run written. Its base unsubscripted, as `Bound`'s kinds'
    are: tyro reads a field's default's class by its original bases, which Python 3.14 refuses a
    subscripted one of (`types.get_original_bases`)."""

    def __eq__(self, other) -> bool:
        # Another bound member alone, which hashes as this one does.
        return (isinstance(other, _Bound) and (self.func, self.args, self.keywords)
                == (other.func, other.args, other.keywords))

    def __hash__(self) -> int:
        return hash(self.func)


@dataclasses.dataclass(frozen=True)
class Bound(ABC):
    """A registered member, a class or the function that builds one, bound to the arguments a run states
    (`bound`: a `functools.partial` of the member by keyword alone, which a type checker reads against
    the member's own signature), and `defaults`, every other argument the member defaults as it defaulted
    them when the run was first recorded, so the record builds the same thing after a default changes
    (`arguments`). What a caller derives from its data (a vocabulary's size, the model an objective is
    built around) it passes at `build`. Each kind names its registry (`_member`), which the bound member
    must be one of, and whether its arguments must be ones a record can carry (`_recorded`).

    A record writes it as `{"name": <import path>, "fields": {...}, "defaults": {...}}` and its other
    settings beside, each by its field (`record`, `read`), the shape every record of it has held, so a run,
    a checkpoint or a release written before reads as written."""

    bound: functools.partial[Any]
    defaults: JsonRecords = dataclasses.field(default_factory=dict)
    """Each argument the member defaults and `bound` leaves out, as a record holds it, read back as that
    record (a function's `{"function": ...}`), so a run read back holds the defaults it was written with;
    one a record lacks, as an argument the member has since gained, takes the member's default now."""

    _recorded: ClassVar[bool] = True
    _aliases: ClassVar[Aliases | None] = None
    """The kind's aliases (`envs`), which name its command line's subcommands (`flagged`); None for a kind
    no command line chooses."""
    _flags: ClassVar[Mapping[str, Annotation]] = {}
    """A command line's subcommand's flags (`record_flags`), each by its flat annotation (`_flat`); none on
    a record of no subcommand."""

    @staticmethod
    @abstractmethod
    def _member(name: str) -> type | Callable[..., Configured]:
        """The class `name` names, from the kind's aliases (`models`, `objectives`, `metrics`), or a
        function or classmethod that builds one (`envs`)."""

    def __post_init__(self) -> None:
        if not isinstance(self.bound, functools.partial):
            raise TypeError(f"{type(self).__name__} binds its member with functools.partial(Member, ...), "
                            f"not {self.bound!r}")
        if self.bound.args:
            raise TypeError(f"{type(self).__name__} binds its member's arguments by name, which its record, "
                            f"its command line and an override address them by, not {self.bound.args!r}")
        member = self._member(import_path(self.bound.func))
        keywords = dict(self.bound.keywords)
        if self._recorded:
            # Held as their records, as a record writes them, and refused here, by name, rather than where
            # the run is built or saved: an argument the member does not take, or a value no record can
            # carry (a lambda). Read, a nested class built, only where the member is built (`arguments`),
            # so a record that names an environment's renderer opens none of its assets until then.
            keywords = argument_records(member, keywords)
        # Held as given otherwise (a model's: a world model's views, named until its environment widens
        # them, `dew.objectives.rl.run.filled`), read where the member is built.
        object.__setattr__(self, "bound", _Bound(member, **keywords))
        object.__setattr__(self, "defaults", recorded_defaults(member, keywords, self.defaults))

    @property
    def member(self) -> type | Callable[..., Configured]:
        """The class, or function, bound (`_member`), as the kind's table holds it."""
        return self._member(import_path(self.bound.func))

    @property
    def name(self) -> str:
        """The bound member's import path, which a record writes."""
        return import_path(self.bound.func)

    @property
    def fields(self) -> Mapping[str, Configured]:
        # The arguments the run states as held: records for a kind that holds records (`_recorded`).
        """The arguments the run states, as the member takes them."""
        return types.MappingProxyType(self.bound.keywords)

    @property
    def arguments(self) -> Mapping[str, object]:
        """The arguments the member is built from, before those the caller derives (`built_arguments`)."""
        return built_arguments(self.member, self.fields, self.defaults)

    @property
    def field_records(self) -> dict[str, JSON]:
        """The arguments the run states, each as a record holds it: what a record compares and writes,
        beside `defaults`. A recorded kind's as held (`argument_records`, at construction); another's each
        recorded as given, read against nothing until its member is built (a world model's views named
        alone, which its environment widens)."""
        # A partial held, at any depth of a container, where the member's argument reads none back is
        # refused, as `to_record` refuses it under a field's annotation (train 1429's review): its record
        # would read back as no partial. The rest written as held, read against nothing.
        for name, value in self.fields.items():
            found = (_unread_partial(value, _argument_type(self.member, name))
                     if isinstance(value, functools.partial | Mapping | list | tuple) else None)
            if found is not None:
                raise Unrecordable(f"{self.name}'s {name} holds {found!r}, which its annotation reads back "
                                   "as no bound member: an argument declaring functools.partial of a kind its "
                                   "member is reads one")
        with recording():
            # Each written as held, against no parameter's annotation, as the record held them before
            # (`JsonDict`): a recorded kind's records again as they are, and a world model's views named
            # alone, which its environment widens, no value of the parameter they become.
            return {name: to_record(value, None) for name, value in self.fields.items()}

    def record(self) -> dict[str, JSON]:
        """This record as a run's record holds it: the member's import path (`name`), the arguments the
        run states as records (`fields`), its `defaults`, and each other setting by its field."""
        settings = {field.name: to_record(vars(self)[field.name], declared_type(type(self), field.name))
                    for field in dataclasses.fields(self) if field.name not in ("bound", "defaults")}
        return {"name": self.name, "fields": self.field_records, "defaults": dict(self.defaults), **settings}

    @classmethod
    def read(cls, record: Mapping[str, object], *, dtypes: bool) -> Self:
        """The record `record` writes (`record`): the member its `name` names, by an alias or an import
        path, bound by its `fields`, and its `defaults` and other settings. A field the record lacks takes
        its default; one the kind does not declare raises."""
        own = {field.name for field in dataclasses.fields(cls)} - {"bound"}
        unknown = sorted(set(record) - own - {"name", "fields"})
        if unknown:
            raise ValueError(f"{cls.__name__} is the record {{'name': ..., 'fields': {{...}}, 'defaults': "
                             f"{{...}}}}; unknown fields {unknown}")
        default = next(field for field in dataclasses.fields(cls) if field.name == "bound").default_factory
        if "name" in record:
            member = cls._member(text(record["name"], f"{cls.__name__}.name"))
        elif default is not dataclasses.MISSING:
            # A record naming no member is the kind's default one's, as a record of an earlier Dew's was.
            member = default().func
        else:
            raise ValueError(f"{cls.__name__}'s record names no member, and the kind has no default one")
        # Fields absent are none; any other value but a record's is refused, as an empty list's is.
        fields = _records.record(record.get("fields", {}), f"{cls.__name__}.fields")
        settings: dict[str, Any] = {name: _rebuilt(declared_type(cls, name), value, dtypes=dtypes, name=name)
                                    for name, value in record.items() if name in own}
        return cls(functools.partial(member, **fields), **settings)

    def with_fields(self, **changed: Configured) -> Self:
        """This record with `changed` among the arguments it states, over those it stated."""
        return dataclasses.replace(self, bound=functools.partial(self.member, **{**self.fields, **changed}))

    def without_fields(self, *names: str) -> Self:
        """This record stating none of `names`, each then the member's default or the caller's."""
        kept = {name: value for name, value in self.fields.items() if name not in names}
        return dataclasses.replace(self, bound=functools.partial(self.member, **kept))

    def recorded(self) -> Bound:
        """This record as its kind's own class holds it, where it is a command line's subcommand
        (`record_flags`): its member, its arguments, its defaults and settings, the flags its arguments
        already hold; itself otherwise. What a run's record writes (`to_record`)."""
        if "_alias" not in vars(type(self)):
            return self
        base = type(self).__mro__[1]
        return base(**{field.name: vars(self)[field.name] for field in dataclasses.fields(base)})

    def flagged(self) -> Self:
        """This record as its command line's subcommand holds it (`record_flags`, `Aliases.flag_choices`): an
        argument a plain flag holds as that flag, the rest kept as they are, its other settings as they
        are; itself where its kind names no aliases or none names its member (a command line chooses none
        such), or where it is already such a subcommand's."""
        aliases = self._aliases
        named = {} if aliases is None else {aliases.paths[alias]: alias for alias in reversed(list(aliases))}
        if self.name not in named or "_alias" in vars(type(self)):
            return self
        made = record_flags(type(self), self.member, self.name, named[self.name])
        own = {each.name for each in dataclasses.fields(type(self))}
        held = {name: value for name, value in self.fields.items()
                if name in made._flags and _fits(value, made._flags[name])}
        settings = {name: vars(self)[name] for name in own if name != "bound"}
        rest = {name: value for name, value in self.fields.items() if name not in held}
        # Its flags and settings stated over the subcommand's defaults, `bound` the rest (its
        # `__post_init__` adding the flags set apart from their defaults).
        return made(bound=functools.partial(self.member, **rest), **settings, **held)


def _simple_dit() -> functools.partial[nn.Module]:
    """A model record's default member, the registry's own simple DiT (`models`), bound to nothing: the
    registry imports no model (`tools/lint_layers.py`), so it names its default by its table."""
    return functools.partial(models["simple_dit"])


@dataclasses.dataclass(frozen=True)
class ModelConfig(Bound):
    """A model's class bound to the fields the run gives it.

    `bound` is the class, of `dew.registry.models` or any model by its import
    path, by the keywords the run states: its own fields, its compute and
    storage dtypes and matmul precision included (its attention kernel is the
    run's, `Diagnostics.kernels`, since row 11938), which the command line
    sets one flag each (`RunConfig.cli`); a recipe fills the ones it derives
    from its data in `build`. `defaults` holds every other field the class
    defaults, as it defaulted them when the run was first recorded, so the
    record builds the same model after a default changes (`arguments`).
    `wrappers` are the classes the model was wrapped in, in the order they
    were applied.
    """

    bound: functools.partial[Any] = dataclasses.field(
        default_factory=_simple_dit)
    """The model's class, a Flax module, bound to the fields the run states."""
    wrappers: tuple[Wrapper, ...] = ()
    """What wraps the built model, innermost first, which `build` applies in
    order: a checkpoint's input quantization, a bound LoRA (its factors are
    stored with the checkpoint variables), the quantized training a run
    wrapped it in."""

    _recorded: ClassVar[bool] = False

    def __post_init__(self) -> None:
        super().__post_init__()
        stray = [type(wrapper).__name__ for wrapper in self.wrappers if not isinstance(wrapper, Wrapper)]
        if stray:
            raise TypeError(f"a model record's wrappers are dew.registry.Wrapper records, not {stray}")

    @staticmethod
    def _member(name: str) -> type:
        return models[name]

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> Self:
        """Read back the record `RunConfig.to_dict` writes for this field."""
        return from_record(cls, values, dtypes=False)

    @classmethod
    def from_model(cls, model) -> Self:
        """Return the module's class bound to its constructor fields, and each wrapper its class was made
        in, in the order they were applied."""
        model_type, layers = wrapper_layers(type(model))
        return cls(functools.partial(model_type, **record_fields(model, model_type)),
                   wrappers=tuple(layer._dew_wrapper() for layer in reversed(layers)))

    def with_dtype(self, dtype: str | None) -> Self:
        """This model computing in `dtype` (`with_dtype`); None keeps it."""
        return dataclasses.replace(
            self, bound=functools.partial(self.member, **with_dtype(self.name, self.fields, dtype)))

    def with_declared(self, field: str, value: str | int) -> Self:
        """This model with `value` in its field `field`, or its parts' (`with_declared`)."""
        return dataclasses.replace(
            self, bound=functools.partial(self.member, **with_declared(self.name, self.fields, field, value)))

    def build(self, **derived: Configured):
        """Build the model from its fields and the ones the recipe `derived`."""
        model = models.build(self.name, {**self.arguments, **derived})
        for wrapper in self.wrappers:
            model = wrapper.apply(model)
        return model


_PLAIN = (int, float, str, bool)
"""The values a command line sets a model's field to by a plain flag (`_flat`)."""


def _flat(annotation: Annotation) -> Annotation | None:
    """The part of a field's `annotation` a command line sets by a plain value (`_PLAIN`, a `Literal` of
    them, a tuple of them, None): a union's such members, its others (a record, a module) left to the
    record; None where no member but None is one."""
    origin = typing.get_origin(annotation)
    if origin is Annotated:
        return _flat(typing.get_args(annotation)[0])
    if annotation in _PLAIN or annotation is type(None):
        return annotation
    if origin is Literal:
        return annotation if all(isinstance(each, _PLAIN) for each in typing.get_args(annotation)) else None
    if origin is tuple:
        parts = typing.get_args(annotation)
        return annotation if parts and all(part is Ellipsis or part in _PLAIN for part in parts) else None
    if origin in (Union, types.UnionType):
        kept = [part for member in typing.get_args(annotation) if (part := _flat(member)) is not None]
        return functools.reduce(operator.or_, kept) if any(part is not type(None) for part in kept) else None
    return None


def _fits(value: Configured, annotation: Annotation) -> bool:
    """Whether `value` is one a flag of the flat `annotation` (`_flat`) holds."""
    origin = typing.get_origin(annotation)
    if origin in (Union, types.UnionType):
        return any(_fits(value, member) for member in typing.get_args(annotation))
    if origin is Literal:
        return any(value == each and type(value) is type(each) for each in typing.get_args(annotation))
    if origin is tuple:
        if not isinstance(value, tuple):
            return False
        parts = typing.get_args(annotation)
        kinds = parts[:1] * len(value) if len(parts) == 2 and parts[1] is Ellipsis else parts
        return len(value) == len(kinds) and all(_fits(each, kind)
                                                for each, kind in zip(value, kinds, strict=True))
    if annotation is type(None):
        return value is None
    return type(value) is annotation or (annotation is float and type(value) is int)


@functools.cache
def record_flags[R: Bound](record: type[R], member: type | Callable[..., Configured], path: str,
                           alias: str) -> type[R]:
    """`member`'s record (`record`, a `Bound` kind's) as a command line sets it (api03): a subclass whose
    own fields are the member's arguments a plain flag sets (`_flat`; one the run builds, an object, takes
    none), each defaulting to the member's default (None where it states none a flag holds, which leaves
    the member's), the record's `bound` and `defaults` hidden from the command line and its other settings
    flags as the record declares them; its `bound` the member, by the flags set apart from their defaults
    over any argument it was given. Its `_alias` names the subcommand it is (`implied_subcommand`)."""
    reserved = {field.name for field in dataclasses.fields(record)} | set(dir(record))
    # Each flag's annotation, `make_dataclass`'s field type, and its flat part (`_flat`), which `flagged`
    # reads a value by.
    flags = []
    flats: dict[str, Annotation] = {}
    stated_defaults: dict[str, Configured] = {}
    for name, declared, annotation in member_fields(member):
        flat = None if annotation is None or name in reserved else _flat(annotation)
        if flat is None:
            continue
        held = declared is not dataclasses.MISSING and _fits(declared, flat)
        default = declared if held else None
        flats[name], stated_defaults[name] = flat, default
        flags.append((name, flat if held else flat | None, dataclasses.field(default=default)))
    suppressed = [("bound", tyro.conf.Suppress[functools.partial],
                   dataclasses.field(default_factory=lambda: functools.partial(member))),
                  ("defaults", tyro.conf.Suppress[JsonRecords], dataclasses.field(default_factory=dict))]

    def __post_init__(self) -> None:
        given = {name: vars(self)[name] for name in stated_defaults
                 if vars(self)[name] != stated_defaults[name]}
        object.__setattr__(self, "bound", functools.partial(member, **{**self.bound.keywords, **given}))
        record.__post_init__(self)

    namespace = {"__post_init__": __post_init__, "__module__": __name__, "_alias": alias.replace("_", "-"),
                 "_flags": flats, "__doc__": (member.__doc__ or "").split("\n\n")[0]}
    title = "".join(part.title() for part in alias.split("_"))
    made = dataclasses.make_dataclass(f"{record.__name__}{title}", [*suppressed, *flags], bases=(record,),
                                      frozen=True, namespace=namespace)
    if not issubclass(made, record):
        raise TypeError(f"{made.__name__} is no {record.__name__}")
    return made


def implied_subcommand(default: Configured) -> str | None:
    """The subcommand a field's `default` is (`record_flags`'s `_alias`: `env:flight`), which a flag of the
    field given without one implies; None for any other default."""
    return vars(type(default)).get("_alias") if isinstance(default, Bound) else None



def member_fields(member: type | Callable[..., Configured]) -> Iterator[tuple[str, Configured, Annotation]]:
    """Each field `_member_flags` flags: its name, its default
    (`dataclasses.MISSING` for none) and its annotation (None where its
    module cannot resolve it). A dataclass with a constructor of its own (a
    metric over its base's fields) has that constructor's arguments, as the
    registry builds it (`record_class`)."""
    held = record_class(member)
    if held is not None:
        for field in constructor_fields(held):
            default = (field.default_factory() if callable(field.default_factory) else field.default)
            yield field.name, default, declared_type(held, field.name)
        return
    for name, (parameter, owner) in parameters(member)[0].items():
        if parameter.kind is parameter.POSITIONAL_OR_KEYWORD and parameter.default is parameter.empty:
            continue  # what the caller builds, the model first
        try:
            annotation = parameter_type(owner, name)
        except ValueError:
            annotation = None
        default = dataclasses.MISSING if parameter.default is parameter.empty else parameter.default
        yield name, default, annotation


def default_record(name: str, declared: Configured, annotation: Annotation) -> JSON:
    """A member's default `declared` as a record holds it: a dtype by its
    name, a matmul precision by its level, anything else as `to_record`
    writes it. TypeError or ValueError for a value no record carries."""
    if name in ("dtype", "param_dtype"):
        return dtype_name(resolve_dtype(declared))
    if name == "precision":
        return None if declared is None else str(declared).lower().removeprefix("precision.")
    # The record itself, not a copy: a module a later default shares turns
    # this one into its defining record.
    record = to_record(declared, annotation)
    json.dumps(record)
    return record


def recorded_defaults(member: type | Callable[..., Configured], stated: Mapping[str, object],
                      recorded: Mapping[str, object]) -> JsonDict:
    """The record of each field or argument `member` defaults that `stated`
    leaves out: as `recorded` holds it, or else as the class defaults it now.
    A default no record carries (a callable, say) is left to the class, and
    a recorded one the class no longer takes is refused."""
    defaults, named = {}, set()
    # One record: a module two defaults share is written once.
    with recording():
        for name, declared, annotation in member_fields(member):
            named.add(name)
            if name in stated or name in recorded or declared is dataclasses.MISSING:
                continue
            try:
                defaults[name] = default_record(name, declared, annotation)
            except (TypeError, ValueError):
                continue
    unknown = sorted(set(recorded) - named)
    if unknown:
        raise ValueError(f"{member.__name__} takes no {unknown}, which the record defaults")
    return {**defaults, **{name: value for name, value in recorded.items() if name not in stated}}


def built_arguments(member: type | Callable[..., Configured], fields: Mapping[str, object],
                    defaults: Mapping[str, object]) -> JsonDict:
    """What a config builds `member` from beside what its caller derives:
    `fields`, over each of `defaults` the class no longer defaults the same.
    A default the class still has is left to it, so a class that treats an
    argument left out in its own way, as `DPOObjective` holds its reference
    at unit decay, still does."""
    now = recorded_defaults(member, {}, {})
    return {**{name: value for name, value in defaults.items() if name not in now or now[name] != value},
            **fields}


def _live_record[Value](annotation: type[Value], value: Configured) -> Value:
    """A command line's record of `annotation`, read live: strict within a saved run's reading too
    (`_records.reading_saved`)."""
    with _records.reading_saved(saved=False):
        return from_record(annotation, value, dtypes=False)


def record_argument[Value](annotation: type[Value]) -> tyro.constructors.PrimitiveConstructorSpec[Value]:
    """A field typed `annotation` written as one JSON value on the command
    line, the record a run holds: a tuple of records as a list,
    `[{"field": ...}, ...]`, and a mapping of class records as an
    object, `{"sigma": {"class": "linear", "fields": {...}}}`, since a flag per
    field cannot spell a collection whose length or keys the command line
    decides.

    The value is read with `from_record` and written with `to_record`, so a
    flag, a default and `run.json` hold the same record. Use it as the
    field's metadata: `Annotated[Mapping[str, ScheduleSpec],
    record_argument(Mapping[str, ScheduleSpec])]`."""
    container = typing.get_origin(annotation) or annotation
    if container is Mapping:
        container = dict
    return tyro.constructors.PrimitiveConstructorSpec(
        nargs=1,
        metavar="JSON",
        instance_from_str=lambda given: _live_record(annotation, json.loads(given[0])),
        is_instance=lambda given: isinstance(given, container),
        str_from_instance=lambda given: [json.dumps(to_record(given, annotation))],
    )


_IMPORTABLE = (type, types.FunctionType, types.BuiltinFunctionType, types.MethodType)
"""What `import_path` names by its own path (`Importable`), as an instance check reads it."""


def imported_argument() -> tyro.constructors.PrimitiveConstructorSpec[Callable[..., Configured]]:
    """A callable field written on the command line as its import path, `module:name` (`imported`): the
    function or the class it names, a class read by the field's own class as its instance at its defaults
    is called (`dew.config.lm.PromptRollout.reward`); written back as the import path of the function or
    the class, or of an instance's class. A run's record holds the callable's own record (`to_record`). Use
    it as the field's metadata: `Annotated[Reward, imported_argument()]`."""
    return tyro.constructors.PrimitiveConstructorSpec(
        nargs=1,
        metavar="MODULE:NAME",
        instance_from_str=lambda given: imported(given[0]),
        is_instance=callable,
        str_from_instance=lambda given: [
            import_path(given if isinstance(given, _IMPORTABLE) else type(given))],
    )


__all__ = [
    "KINDS",
    "Aliases",
    "ModelConfig",
    "Record",
    "Recorded",
    "argument_records",
    "component",
    "dtype_name",
    "encoders",
    "from_record",
    "import_path",
    "import_trusted",
    "imported",
    "imported_argument",
    "metrics",
    "mixers",
    "models",
    "objectives",
    "optimizers",
    "parameters",
    "presets",
    "projectors",
    "record_argument",
    "record_fields",
    "runs",
    "schedules",
    "solvers",
    "to_record",
    "towers",
    "trust_sources",
    "trusts_source",
    "with_dtype",
]
