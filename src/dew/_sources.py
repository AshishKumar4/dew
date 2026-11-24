"""Where a process's code comes from: the file a loaded module is code of, whether a module is a user's own
code (loaded from a file no installed distribution versions and outside the interpreter's standard library)
or an installed distribution's, which its version states. One statement, which a data plan's step identity
(`dew.data.plan`) and a rung record's key (`dew._stated`) read."""

from __future__ import annotations

import abc
import contextlib
import contextvars
import dataclasses
import enum
import functools
import importlib.metadata
import inspect
import site
import sys
import sysconfig
import types
import typing
from collections.abc import Callable, Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, Protocol, runtime_checkable


def source_file(held: Callable | type) -> Path | None:
    """The file `held`, a function or a class, is defined in, None where it has none: a class of a notebook,
    a REPL or `python -c` (`__main__` with no file), a builtin's or an extension's, or a file since removed.
    Read where its module names a file, where Python reads its source file without raising."""
    module = sys.modules.get(held.__module__)
    if module is None or vars(module).get("__file__") is None:
        return None
    place = inspect.getsourcefile(held)
    return Path(place) if place is not None and Path(place).is_file() else None


def recorded(name: str) -> Path | None:
    """The file whose source a process record holds for the loaded module `name` (`file_of`), None for one
    it holds none of: a module of no file, a frozen one (the interpreter's own, whatever file it names),
    Dew's (its package digest's) and `__main__` (stated where a call names it)."""
    module = sys.modules[name]
    spec = vars(module).get("__spec__")
    frozen = spec is not None and spec.origin == "frozen"
    if (vars(module).get("__file__") is None or frozen or name in ("dew", "__main__")
            or name.startswith("dew.")):
        return None
    return file_of(name)



def file_of(name: str) -> Path:
    """The file the loaded module `name` is code of: its own, or, where the file it names is none (a
    module an extension makes of itself), its nearest package's that is one; refused where none is, as
    nothing then states its code."""
    names = name.split(".")
    for parts in range(len(names), 0, -1):
        module = sys.modules.get(".".join(names[:parts]))
        file = None if module is None else vars(module).get("__file__")
        if file is not None and Path(file).is_file():
            return Path(file).resolve()
    raise ValueError(f"the loaded module {name} names no file of its code, so no record of this process "
                     "can state it")



def own_source(name: str | None) -> bool:
    """Whether the module `name` is a user's own code: loaded from a file no distribution versions and
    Dew's digest does not hold, or the script the process runs.
    Read once a loaded module (`_OWN`), in the process's install (`environment`, which a block may set
    apart, read afresh): a module loaded again is read again."""
    if name == "__main__":
        return vars(sys.modules[name]).get("__file__") is not None
    if name is None or name not in sys.modules:
        return False
    module = sys.modules[name]
    given = _ENVIRONMENT.get()
    held = _OWN.get(name) if given is None else None
    if held is not None and held[0] is module:
        return held[1]
    path = recorded(name)
    placed = path is not None and within(path, installs())
    own = path is not None and not (placed and name.partition(".")[0] in packages()) and (
        placed or not within(path, standard()))
    if given is None:
        _OWN[name] = (module, own)
    return own


@dataclasses.dataclass(frozen=True)
class Environment:
    """An install: the directories distributions are installed into and the distributions each top-level
    package belongs to (`installed`, `distributed`, the process's)."""

    directories: tuple[Path, ...]
    packages: Mapping[str, Sequence[str]]


_ENVIRONMENT: contextvars.ContextVar[Environment | None] = contextvars.ContextVar("environment", default=None)
"""The install a block reads in place of the process's (`environment`), None outside one."""


@contextlib.contextmanager
def environment(installed: Environment):
    """Within the block, `installed` is the install code is classified by (`own_source`, `installs`,
    `packages`): a test's double of a distribution installed beside the process's."""
    token = _ENVIRONMENT.set(installed)
    try:
        yield
    finally:
        _ENVIRONMENT.reset(token)


def installs() -> tuple[Path, ...]:
    """The install directories in effect (`environment`'s, else `installed`)."""
    given = _ENVIRONMENT.get()
    return installed() if given is None else given.directories


def packages() -> Mapping[str, Sequence[str]]:
    """The distributions each top-level package belongs to in effect (`environment`'s, else
    `distributed`)."""
    given = _ENVIRONMENT.get()
    return distributed() if given is None else given.packages


_OWN: dict[str, tuple[object, bool]] = {}
"""Each loaded module's verdict (`own_source`) beside the module it was read of: a program's identity
asks it of every function it states, its file's path resolved and placed each time (13% of a small fit)."""


@functools.cache
def distributed() -> Mapping[str, list[str]]:
    """The installed distributions each top-level package belongs to (`packages_distributions`), read
    once: a process installs none while it runs."""
    return importlib.metadata.packages_distributions()


@functools.cache
def installed() -> tuple[Path, ...]:
    """The directories this interpreter installs distributions into."""
    paths = sysconfig.get_paths()
    return tuple({Path(path).resolve() for path in (paths["purelib"], paths["platlib"],
                                                    *site.getsitepackages(), site.getusersitepackages())})


@functools.cache
def standard() -> tuple[Path, ...]:
    """The directories of this interpreter's standard library, whose code its version states."""
    paths = sysconfig.get_paths()
    return tuple({Path(paths["stdlib"]).resolve(), Path(paths["platstdlib"]).resolve()})


def within(path: Path, directories: Sequence[Path]) -> bool:
    return any(path.is_relative_to(directory) for directory in directories)


def names_read(code: types.CodeType) -> set[str]:
    """Every name `code` and the code nested in it (a lambda, a comprehension, a local function) reads."""
    names = set(code.co_names)
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            names |= names_read(constant)
    return names



def wrapped(member):
    """The function a class's descriptor wraps: a static or class method's, a property's getter, a cached
    property's. Any other member is itself."""
    if isinstance(member, (staticmethod, classmethod)):
        return member.__func__
    if isinstance(member, property):
        return member.fget
    if isinstance(member, functools.cached_property):
        return member.func
    return member


@runtime_checkable
class Wrapping(Protocol):
    """A function that runs another, which it names (`functools.wraps`, `jax.jit`)."""

    __wrapped__: Callable


def hand_written(member, own: Callable[[str | None], bool] = own_source) -> bool:
    """Whether `member` is, or wraps (`__wrapped__`), a function a file of the user's own code wrote: the
    file of the globals it was defined in, as against one made for the class (a dataclass's `__init__`,
    whose code is no file's, and its `__repr__`, which a wrapper of Python's own holds). By its globals,
    not the module it names: a decorator's wrapper names the module of the function it wraps. `own` is
    which modules' code counts as the user's (`dew._stated`'s walk of an installed plugin's among it)."""
    seen: set[int] = set()
    while member is not None and id(member) not in seen:
        seen.add(id(member))
        if isinstance(member, types.FunctionType):
            defined = member.__globals__
            if (own(defined.get("__name__"))
                    and member.__code__.co_filename == defined.get("__file__")):
                return True
        member = member.__wrapped__ if isinstance(member, Wrapping) else None
    return False

def function_reads(function: types.FunctionType) -> tuple[dict, dict]:
    """What `function`'s body reads outside its arguments: the cells it closed over (None for one nothing
    has filled yet), and each name its code or the code nested in it reads (`names_read`) that its globals
    hold or that is a loaded module of the user's (one its body imports). A name is every use of it, an
    attribute's among them, so more is read than is used, never less. The one statement of what a function
    reads, which a stated value (`dew._stated`) and a data step's held code (`dew.data.plan`) walk."""
    code = function.__code__
    cells = {}
    for name, cell in zip(code.co_freevars, function.__closure__ or (), strict=True):
        try:
            cells[name] = cell.cell_contents
        except ValueError:
            cells[name] = None
    names = names_read(code)
    held = {name: function.__globals__[name] for name in names if name in function.__globals__}
    imported = {name: sys.modules[name] for name in names - held.keys()
                if name in sys.modules and own_source(name)}
    return cells, held | imported


_ABC_STATE = type(vars(abc.ABC)["_abc_impl"])
"""The type of what `abc.ABCMeta` keeps in each class it makes (its registry and caches), machinery no
statement of the user's code holds (`members`)."""


def members(held, own: Callable[[str | None], bool] = own_source) -> list[tuple]:
    """What a class or a module of the user's holds, by name, in order: a method as the function its
    descriptor wraps (`wrapped`), anything else as itself. Of Python's own entries (`__dict__`, `__doc__`,
    `__annotations__`) only the methods the user wrote (`__call__`), which a call runs; the rest are its
    machinery's, the methods a dataclass makes among them, which the class's description states, the
    state `abc.ABCMeta` writes into every class it makes (`_ABC_STATE`), a Grain transform's among them, the
    descriptor Python makes for each of a class's own slots (mangled or not), and a generated field's
    default where the class attribute still holds it, which `generated_fields` states with its factory:
    a slotted class's and a child overriding a default hold it where their class attributes do not (train
    947's review), and a value set on the class at run time, or a class variable, stays a member (train
    950's review: skipping every field name left them unstated). `own` is which modules' code counts as the
    user's (`hand_written`)."""
    defaults = {field.name: field.default for field in generated_fields(held)}
    found = []
    for name, member in sorted(vars(held).items()):
        member = wrapped(member)
        if ((name.startswith("__") and name.endswith("__") and not hand_written(member, own))
                or isinstance(member, _ABC_STATE)
                or (name in defaults and member is defaults[name])
                or (isinstance(member, types.MemberDescriptorType) and member.__objclass__ is held)):
            continue
        found.append((name, member))
    return found


MISSING = dataclasses.MISSING
"""What a `GeneratedField` holds where its field has no default or no factory."""


class GeneratedField(NamedTuple):
    """A field a class generator (`dataclasses`, attrs) recorded on a class: its name, its default and its
    default factory, each `MISSING` where it has none, and every other part its generator recorded of it
    (`parts`), each by its name (`generated_fields`)."""

    name: str
    default: Any
    factory: Any
    parts: tuple[tuple[str, Any], ...] = ()


def instance_fields(held: type) -> list[GeneratedField]:
    """The fields an instance of `held`, a generated class (`generated`), holds: those its nearest class in
    its method resolution order that a generator made itself recorded (`generated_fields`), so a subclass
    no generator decorated holds its base's (train 1013's review)."""
    made = next((base for base in held.__mro__ if "__dataclass_fields__" in vars(base) or attrs_made(base)),
                held)
    return generated_fields(made)


def decides(field: GeneratedField) -> bool:
    """Whether `field` takes part in its instances' equality, as its generator records it: a dataclass
    field's `compare`, an attrs field's `eq`. A field that declares it does not (`compare=False`,
    `eq=False`: credentials, a timeout) decides no rows, and an identity leaves it out."""
    parts = dict(field.parts)
    return parts.get("compare", parts.get("eq", True)) is not False


def attrs_made(held: type) -> bool:
    """Whether attrs made `held` itself (its own `__attrs_attrs__`, which a subclass it did not decorate
    inherits)."""
    return "__attrs_attrs__" in vars(held)


def generated(held: type) -> bool:
    """Whether a class generator made `held`: a dataclass (one it derives from among them) or a class attrs
    made itself."""
    return dataclasses.is_dataclass(held) or attrs_made(held)


def generated_instance(value) -> bool:
    """Whether `value` is an instance of a class a generator made (`generated`), which its fields state."""
    return not isinstance(value, type) and generated(type(value))


def field_names(held: type) -> list[str]:
    """The names of the fields a class generator recorded for `held` (`generated`), in order: what an
    instance of it holds, a record's parts."""
    return [field.name for field in instance_fields(held)]


def _setattr_hooks(held: type) -> Mapping[str, Callable]:
    """The hook the `__setattr__` attrs generated for `held` runs on each field it hooks, by the field's name:
    the field's own `on_setattr` or the class's, held in that method's closure alone (its `sa_attrs`, each
    field's `Attribute` beside its hook). Empty for a class whose `__setattr__` attrs did not generate."""
    method = vars(held).get("__setattr__")
    if not isinstance(method, types.FunctionType) or hand_written(method):
        return {}
    for cell in method.__closure__ or ():
        held_hooks = cell.cell_contents
        if isinstance(held_hooks, dict):
            return {name: hook for name, (_, hook) in held_hooks.items()}
    return {}


DATACLASS_OPTIONS = {name: parameter.default for name, parameter
                     in inspect.signature(dataclasses.dataclass).parameters.items() if name != "cls"}
"""What `dataclasses.dataclass` makes a class with unless told otherwise (eq, order, frozen, ...), read from
its signature: a class's departures from it are part of what it is (`generation_options`)."""

ATTRS_COMPARISONS = {"eq": "__eq__", "order": "__lt__"}
"""An attrs class's comparison options by the method each writes into it, and so whether it was given each:
attrs makes a class with equality and without ordering unless told otherwise (`attrs.define`'s eq=True,
order=False)."""

ATTRS_DEFAULTS = {"eq": True, "order": False}
"""What `attrs.define` makes a class with unless told otherwise, of `ATTRS_COMPARISONS`."""


ATTRS_PROPS_UNSTATED = frozenset({"on_setattr_hook", "field_transformer"})
"""The parts of attrs' record of how it made a class (`__attrs_props__`, attrs 25.4 and later) whose effect
its fields state: the hook each field's setter runs (`generated_fields`' effective hook) and the fields a
transformer made."""


def generation_options(held: type) -> tuple[tuple[str, bool | str], ...]:
    """How a class generator made `held` itself, each option by its name, as the generator records it, which
    with its fields (`generated_fields`) is what the methods it wrote are written from: a dataclass's options
    apart from `DATACLASS_OPTIONS` (eq, order, frozen, unsafe_hash, ...); an attrs class's every recorded
    option (`__attrs_props__`: frozen, slotted, hashability, ...) but `ATTRS_PROPS_UNSTATED`, or under an
    attrs that records none, its equality and ordering by the methods it holds (`ATTRS_COMPARISONS`) apart
    from `ATTRS_DEFAULTS`. The code it wrote is stated by none of them: attrs writes a process's salted
    string hash into a hashable class's `__hash__` (train 974's review). Empty for a dataclass made as
    `dataclasses.dataclass` makes one by default, or a class no generator made (base review of 67fd128db:
    an eq=False went unstated, and a step comparing its instances read other rows)."""
    params = vars(held).get("__dataclass_params__")
    if params is not None:
        return tuple((name, getattr(params, name)) for name, default in DATACLASS_OPTIONS.items()
                     if getattr(params, name) != default)
    props = vars(held).get("__attrs_props__")
    if props is not None:
        return tuple((name, _option(getattr(props, name))) for name in vars(type(props))["__slots__"]
                     if name not in ATTRS_PROPS_UNSTATED)
    if "__attrs_attrs__" in vars(held):
        given = {name: vars(held).get(method) is not None for name, method in ATTRS_COMPARISONS.items()}
        return tuple((name, value) for name, value in given.items() if value != ATTRS_DEFAULTS[name])
    return ()


def _option(value: bool | enum.Enum) -> bool | str:
    """An option attrs records of a class (`__attrs_props__`) as an identity states it: a flag as it is, a
    choice of its (`KeywordOnly`, `Hashability`) by its value's name."""
    return value.name if isinstance(value, enum.Enum) else value


def annotation_text(annotation: Hashable) -> str | tuple:
    """An annotation as an identity states it: a string as written, a class by its module and name, a form
    holding `typing.Annotated` metadata by its parts (`_annotated_parts`), any other (a generic alias, a
    union, a typing form) by the text Python gives it, which names what it holds. Metadata is any value, a
    function or an object whose text is its address in this process among them, so its text was stated, and
    an unchanged step's exact resume refused in every other process (train 977's review, row 7270)."""
    match annotation:
        case str():
            return annotation
        case type() if not isinstance(annotation, types.GenericAlias):
            return f"{annotation.__module__}.{annotation.__qualname__}"
        case _ if _holds_annotated(annotation):
            return _annotated_parts(annotation)
        case _:
            return repr(annotation)


def _holds_annotated(annotation: Hashable) -> bool:
    """Whether `annotation` is, or holds among its arguments, a `typing.Annotated` form."""
    if typing.get_origin(annotation) is typing.Annotated:
        return True
    return any(_holds_annotated(argument) for argument in _arguments(annotation))


def _arguments(annotation: Hashable) -> tuple[Hashable, ...]:
    """`annotation`'s arguments (`typing.get_args`), a callable's parameter list's own among them."""
    return tuple(inner for argument in typing.get_args(annotation)
                 for inner in (argument if isinstance(argument, list) else [argument]))


def _annotated_parts(annotation: Hashable) -> tuple:
    """A form holding `typing.Annotated` metadata as the values it holds, which an identity states as it
    states any value (`dew._stated`: a module's function by its name, an object by its form): an
    `Annotated` as its annotation's own statement and its metadata, any other form as its own type, its
    origin and its arguments' statements (`annotation_text`), a union among them."""
    if typing.get_origin(annotation) is typing.Annotated:
        annotated, *metadata = typing.get_args(annotation)
        return ("annotated", annotation_text(annotated), *metadata)
    # The form's own type beside its origin: `list[...]` and `typing.List[...]` share their origin, and a
    # converter reading a field's annotation (`attrs.Attribute.type`) tells them apart.
    form = type(annotation)
    return ("form", f"{form.__module__}.{form.__qualname__}", typing.get_origin(annotation),
            *map(annotation_text, _arguments(annotation)))


def _field_value[Value](name: str, value: Value) -> Value | str | tuple | dict:
    """A part of a field as its generator recorded it, as an identity states it: its annotation as text
    (`annotation_text`), its metadata as the mapping it is, a dataclass's field kind (a field, a class or an
    init variable, one of its sentinels) by its name, anything else as it is."""
    if name == "type" and isinstance(value, Hashable):
        return annotation_text(value)
    if isinstance(value, types.MappingProxyType):
        return dict(value)
    if type(value).__name__ == "_FIELD_BASE":
        return str(value)
    return value


ATTRS_UNSTATED = frozenset({"name", "default", "on_setattr"})
"""An attrs field's parts its `Attribute` records that `generated_fields` states apart: its name, its
default, split into a default and a factory, and its setter hook, which the hook its class's setter runs on
it states (`attrs.setters.NO_OP` among them, a bare sentinel no identity names; train 974's review)."""

DATACLASS_UNSTATED = frozenset({"name", "default", "default_factory"})
"""A dataclass field's parts its `dataclasses.Field` records that `generated_fields` states apart."""


def generated_fields(held: type) -> list[GeneratedField]:
    """The fields a class generator recorded on `held` itself, in order, its bases' among them and an
    override's default its own: a dataclass's (`dataclasses.fields`, its class and init variables, which are
    no fields, apart) or an attrs class's (`__attrs_attrs__`, a default `attrs.Factory`'s factory its
    factory). Each with every other part its generator records of it (`parts`): every slot of its
    `dataclasses.Field` or `attrs.Attribute`, by value, whichever this Python and this attrs record (its
    annotation, keyword-only flag, comparison flags and keys, converter, validator, metadata, alias, ...),
    so a part added later is stated too, and an attrs field's hook its class's generated `__setattr__` runs.
    Empty for a class no generator made itself (the evening audit of 2026-11-16: a field's keyword-only
    flag, a factory's `takes_self` and an annotation a converter reads went unstated; train 955's review: a
    class-level hook)."""
    if "__dataclass_fields__" in vars(held):
        found = []
        for field in dataclasses.fields(held):
            slots = vars(type(field))["__slots__"]
            parts = tuple((name, _field_value(name, getattr(field, name)))
                          for name in slots if name not in DATACLASS_UNSTATED)
            found.append(GeneratedField(field.name, field.default, field.default_factory, parts))
        return found
    if attrs_made(held):
        # attrs is loaded where a class of it exists; its sentinel and factory type are its own.
        attr = sys.modules["attr"]
        hooks = _setattr_hooks(held)
        found = []
        for field in vars(held)["__attrs_attrs__"]:
            default, factory = field.default, MISSING
            parts = [(name, _field_value(name, getattr(field, name)))
                     for name in vars(type(field))["__slots__"] if name not in ATTRS_UNSTATED]
            if isinstance(default, attr.Factory):
                # A factory by what it calls and whether it is handed the instance it fills.
                default, factory = MISSING, default.factory
                parts.append(("takes_self", field.default.takes_self))
            if (hook := hooks.get(field.name)) is not None:
                parts.append(("effective_on_setattr", hook))
            found.append(GeneratedField(field.name, MISSING if default is attr.NOTHING else default, factory,
                                        tuple(parts)))
        return found
    return []
